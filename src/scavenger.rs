//! APFS file scavenger: watches a local directory, chunks text files,
//! tokenizes each chunk through the Qwen tokenizer, looks up token vectors
//! in the memory-mapped 622 MB FP16 embedding table, and persists the
//! chunked text, metadata hashes, token IDs, and vector state to Sled.
//!
//! All processing is local and offline: no network sockets are opened and
//! the only external I/O is to the filesystem, tokenizer, embedding table,
//! and Sled database.

use anyhow::{anyhow, Context, Result};
use half::f16;
use notify::{Event, EventKind, RecommendedWatcher, RecursiveMode, Watcher};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use sled::Db;
use std::collections::HashMap;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, SystemTime};
use tokenizers::Tokenizer;
use tokio::sync::mpsc;
use tokio::task::spawn_blocking;

const CHUNK_TOKEN_TARGET: usize = 512;
const CHUNK_TOKEN_OVERLAP: usize = 64;
const CHUNK_BYTE_LIMIT: usize = 8_388_608; // 8 MiB
const WATCH_DEPTH: usize = 4;
const TRACKED_EXTENSIONS: &[&str] = &[
    "txt", "md", "rs", "py", "c", "cpp", "h", "hpp", "swift", "js", "ts", "sh", "bash", "zsh",
    "json", "yaml", "yml", "toml", "xml", "csv", "log", "ini", "cfg",
];
const METADATA_PREFIX: &[u8] = b"scavenger:meta:";
const CHUNK_PREFIX: &str = "scavenger:chunk";

/// Runtime configuration for the scavenger.
#[derive(Clone, Debug)]
pub struct ScavengerConfig {
    pub watch_dir: PathBuf,
    pub sled_db_path: PathBuf,
    pub tokenizer_path: PathBuf,
    pub embedding_path: PathBuf,
    pub hidden_size: usize,
    pub vocab_size: usize,
}

impl ScavengerConfig {
    /// Build from environment with sensible fallbacks.
    pub fn from_env() -> Result<Self> {
        let watch_dir = std::env::var_os("BADAPPLE_WILD_WORKSPACE_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("wild_workspace"));

        let sled_db_path = std::env::var_os("BADAPPLE_SLED_DB_PATH")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("strategy_db"));

        let model_path = std::env::var_os("BADAPPLE_ANE_MODEL")
            .map(PathBuf::from)
            .unwrap_or_default();

        let tokenizer_path = std::env::var_os("BADAPPLE_ANE_TOKENIZER")
            .map(PathBuf::from)
            .or_else(|| model_path.parent().map(|p| p.join("tokenizer.json")))
            .filter(|p| !p.as_os_str().is_empty())
            .context("BADAPPLE_ANE_TOKENIZER must be set or a tokenizer.json must sit beside BADAPPLE_ANE_MODEL")?;

        let embedding_path = model_path
            .parent()
            .map(|p| p.join("embedding.f16"))
            .filter(|p| !p.as_os_str().is_empty())
            .context("BADAPPLE_ANE_MODEL must be set and its parent must contain embedding.f16")?;

        let (hidden_size, vocab_size) = Self::parse_model_manifest(&model_path)?;

        Ok(Self {
            watch_dir,
            sled_db_path,
            tokenizer_path,
            embedding_path,
            hidden_size,
            vocab_size,
        })
    }

    fn parse_model_manifest(model_path: &Path) -> Result<(usize, usize)> {
        if let Ok(raw) = fs::read_to_string(model_path) {
            if let Ok(value) = serde_json::from_str::<serde_json::Value>(&raw) {
                if let Some(model) = value.get("model") {
                    let hidden = model
                        .get("hidden_size")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(2048) as usize;
                    let vocab = model
                        .get("vocab_size")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(151_936) as usize;
                    return Ok((hidden, vocab));
                }
            }
        }
        Ok((2048, 151_936))
    }
}

/// Memory-mapped FP16 embedding table for the local Qwen model.
struct EmbeddingTable {
    mmap: memmap2::Mmap,
    hidden_size: usize,
    vocab_size: usize,
}

impl EmbeddingTable {
    fn open(path: &Path, hidden_size: usize, vocab_size: usize) -> Result<Self> {
        let file = fs::File::open(path)
            .with_context(|| format!("unable to open embedding table: {}", path.display()))?;
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        let expected = vocab_size * hidden_size * 2;
        let actual = mmap.len();
        if actual < expected {
            tracing::warn!(
                "embedding table is smaller than expected ({} < {}); truncating lookups",
                actual,
                expected
            );
        }
        if mmap.len() % 2 != 0 {
            tracing::warn!("embedding table has an odd byte count; the last byte will be ignored");
        }
        Ok(Self {
            mmap,
            hidden_size,
            vocab_size,
        })
    }

    /// Return the FP16 row for `token_id` as a slice of `u16` bits.
    fn row(&self, token_id: u32) -> Option<&[u16]> {
        let id = token_id as usize;
        if id >= self.vocab_size {
            return None;
        }
        let start_u16 = id * self.hidden_size;
        let end_u16 = start_u16 + self.hidden_size;
        let byte_start = start_u16 * 2;
        let byte_end = end_u16 * 2;
        let u8_slice = self.mmap.get(byte_start..byte_end)?;
        // mmap memory is page-aligned, so each even byte offset is 2-byte aligned.
        Some(bytemuck::cast_slice(u8_slice))
    }
}

/// A single chunked record persisted to Sled.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct ChunkRecord {
    path: String,
    mtime: u64,
    chunk_index: usize,
    text: String,
    token_ids: Vec<u32>,
    /// Mean-pooled FP16 vector state for this chunk, stored as raw u16 bits.
    vector: Vec<u16>,
}

/// Persistent metadata for a watched file.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct FileMetadata {
    path: String,
    content_hash: String,
    chunk_count: usize,
}

/// Background scavenger state.
pub struct Scavenger {
    config: ScavengerConfig,
    tokenizer: Arc<Tokenizer>,
    embeddings: Arc<EmbeddingTable>,
    db: Arc<Db>,
}

impl Scavenger {
    /// Open the tokenizer, embedding table, and Sled database.
    pub fn open(config: ScavengerConfig) -> Result<Self> {
        fs::create_dir_all(&config.watch_dir).ok();
        let tokenizer = Arc::new(Tokenizer::from_file(&config.tokenizer_path).map_err(|e| {
            anyhow!(
                "unable to load tokenizer {}: {}",
                config.tokenizer_path.display(),
                e
            )
        })?);
        let embeddings = Arc::new(EmbeddingTable::open(
            &config.embedding_path,
            config.hidden_size,
            config.vocab_size,
        )?);
        let db = Arc::new(sled::open(&config.sled_db_path)?);
        Ok(Self {
            config,
            tokenizer,
            embeddings,
            db,
        })
    }

    /// Run the watcher loop.  The filesystem watcher itself runs on a
    /// dedicated background thread; all indexing work is offloaded to
    /// `spawn_blocking` so the main executor stays responsive.
    pub async fn run(self: Arc<Self>) -> Result<()> {
        let (tx, mut rx) = mpsc::unbounded_channel::<Event>();
        let watch_dir = self.config.watch_dir.clone();

        // Debounce map: path -> last seen mtime.
        let mut pending: HashMap<PathBuf, u64> = HashMap::new();

        // Spawn a dedicated, low-priority background thread for the watcher.
        let _watcher_thread = thread::Builder::new()
            .name("bad-apple-scavenger-watcher".into())
            .spawn(move || {
                set_background_qos();
                let mut watcher: RecommendedWatcher =
                    notify::recommended_watcher(move |res: Result<Event, notify::Error>| {
                        if let Ok(event) = res {
                            let _ = tx.send(event);
                        }
                    })
                    .expect("unable to create scavenger filesystem watcher");
                watcher
                    .watch(&watch_dir, RecursiveMode::Recursive)
                    .expect("unable to watch scavenger directory");
                // Keep the watcher alive until the process exits.
                loop {
                    thread::park();
                }
            })
            .context("unable to spawn scavenger watcher thread")?;

        // Seed existing files before watching for new events.
        self.seed_existing_files().await?;

        while let Some(event) = rx.recv().await {
            for path in &event.paths {
                match event.kind {
                    EventKind::Create(_) | EventKind::Modify(_) if is_tracked_file(path) => {
                        if let Ok(meta) = fs::metadata(path) {
                            let mtime = meta
                                .modified()
                                .ok()
                                .and_then(|t| t.duration_since(SystemTime::UNIX_EPOCH).ok())
                                .map(|d| d.as_secs())
                                .unwrap_or(0);
                            pending.insert(path.clone(), mtime);
                        }
                    }
                    EventKind::Remove(_) if is_tracked_file(path) => {
                        pending.remove(path);
                        let self_ref = self.clone();
                        let path = path.clone();
                        spawn_blocking(move || self_ref.purge_file(&path));
                    }
                    _ => {}
                }
            }

            // Debounce: collect events for one second, then process.
            tokio::time::sleep(Duration::from_secs(1)).await;
            while let Ok(event) = rx.try_recv() {
                for path in &event.paths {
                    match event.kind {
                        EventKind::Create(_) | EventKind::Modify(_) if is_tracked_file(path) => {
                            if let Ok(meta) = fs::metadata(path) {
                                let mtime = meta
                                    .modified()
                                    .ok()
                                    .and_then(|t| t.duration_since(SystemTime::UNIX_EPOCH).ok())
                                    .map(|d| d.as_secs())
                                    .unwrap_or(0);
                                pending.insert(path.clone(), mtime);
                            }
                        }
                        EventKind::Remove(_) if is_tracked_file(path) => {
                            pending.remove(path);
                            let self_ref = self.clone();
                            let path = path.clone();
                            spawn_blocking(move || self_ref.purge_file(&path));
                        }
                        _ => {}
                    }
                }
            }

            // Process pending files on a blocking thread.
            if !pending.is_empty() {
                let batch: Vec<(PathBuf, u64)> = pending.drain().collect();
                let self_ref = self.clone();
                spawn_blocking(move || {
                    for (path, mtime) in batch {
                        if let Err(error) = self_ref.index_file(&path, mtime) {
                            tracing::warn!(
                                "scavenger failed to index {}: {}",
                                path.display(),
                                error
                            );
                        }
                    }
                });
            }
        }

        Ok(())
    }

    /// Index existing files in the watch directory before live events arrive.
    async fn seed_existing_files(self: &Arc<Self>) -> Result<()> {
        let mut files = Vec::new();
        collect_files(&self.config.watch_dir, WATCH_DEPTH, &mut files);

        if !files.is_empty() {
            let self_ref = self.clone();
            spawn_blocking(move || {
                for (path, mtime) in files {
                    if let Err(error) = self_ref.index_file(&path, mtime) {
                        tracing::warn!("scavenger failed to seed {}: {}", path.display(), error);
                    }
                }
            })
            .await?;
        }
        Ok(())
    }

    /// Remove all records and metadata for a deleted file.
    fn purge_file(&self, path: &Path) {
        let path_key = path_key(path);
        let meta_key = meta_key(&path_key);
        if let Ok(Some(raw)) = self.db.get(&meta_key) {
            if let Ok(meta) = serde_json::from_slice::<FileMetadata>(&raw) {
                for i in 0..meta.chunk_count {
                    let chunk_key = chunk_key(&path_key, i);
                    let _ = self.db.remove(chunk_key);
                }
            }
        }
        let _ = self.db.remove(&meta_key);
        tracing::info!("scavenger purged {}", path.display());
    }

    /// Read, chunk, tokenize, embed, and persist a file.
    fn index_file(&self, path: &Path, mtime: u64) -> Result<()> {
        let path_key = path_key(path);
        let meta_key = meta_key(&path_key);

        // Skip if unchanged.
        if let Ok(Some(raw)) = self.db.get(&meta_key) {
            if let Ok(meta) = serde_json::from_slice::<FileMetadata>(&raw) {
                let current_hash = content_hash(path);
                if !current_hash.is_empty() && meta.content_hash == current_hash {
                    return Ok(());
                }
            }
        }

        // Delete any previous version before writing the new one.
        self.purge_file(path);

        let text = read_limited_text(path, CHUNK_BYTE_LIMIT)?;
        let current_hash = content_hash_bytes(text.as_bytes());
        let chunks = self.chunk_text(&text);
        let chunk_count = chunks.len();

        for (i, chunk_text) in chunks.iter().enumerate() {
            let record = self.embed_chunk(path, mtime, i, chunk_text)?;
            let key = chunk_key(&path_key, i);
            let value = serde_json::to_vec(&record)?;
            self.db.insert(key, value)?;
        }

        let metadata = FileMetadata {
            path: path.to_string_lossy().into_owned(),
            content_hash: current_hash,
            chunk_count,
        };
        self.db.insert(&meta_key, serde_json::to_vec(&metadata)?)?;

        tracing::info!(
            "scavenger indexed {} -> {} chunks",
            path.display(),
            chunk_count
        );
        Ok(())
    }

    /// Split text into overlapping token-bounded chunks.
    fn chunk_text(&self, text: &str) -> Vec<String> {
        let paragraphs: Vec<&str> = text
            .split("\n\n")
            .filter(|p| !p.trim().is_empty())
            .collect();
        if paragraphs.is_empty() {
            return vec![text.to_string()];
        }

        let mut chunks: Vec<String> = Vec::new();
        for paragraph in paragraphs {
            let tokens = match self.tokenizer.encode(paragraph, true) {
                Ok(encoding) => encoding.get_ids().to_vec(),
                Err(_) => {
                    chunks.push(paragraph.to_string());
                    continue;
                }
            };

            if tokens.len() <= CHUNK_TOKEN_TARGET {
                chunks.push(paragraph.to_string());
                continue;
            }

            // Split by tokens with overlap.
            let mut start = 0;
            while start < tokens.len() {
                let end = (start + CHUNK_TOKEN_TARGET).min(tokens.len());
                let slice = &tokens[start..end];
                if let Ok(decoded) = self.tokenizer.decode(slice, true) {
                    chunks.push(decoded);
                }
                if end == tokens.len() {
                    break;
                }
                start = end.saturating_sub(CHUNK_TOKEN_OVERLAP);
            }
        }

        // Merge tiny contiguous chunks to avoid excessive fragmentation.
        let mut merged: Vec<String> = Vec::new();
        for chunk in chunks {
            if let Some(last) = merged.last_mut() {
                let candidate = format!("{}\n\n{}", last, chunk);
                if let Ok(merged_tokens) = self.tokenizer.encode(candidate.as_str(), true) {
                    if merged_tokens.get_ids().len() <= CHUNK_TOKEN_TARGET {
                        *last = candidate;
                        continue;
                    }
                }
            }
            merged.push(chunk);
        }
        merged
    }

    /// Tokenize a single chunk and compute its mean-pooled FP16 vector.
    fn embed_chunk(
        &self,
        path: &Path,
        mtime: u64,
        index: usize,
        text: &str,
    ) -> Result<ChunkRecord> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow!("unable to tokenize chunk: {}", e))?;
        let token_ids = encoding.get_ids().to_vec();

        let mut mean = vec![0.0f32; self.config.hidden_size];
        let mut valid_tokens = 0;
        for &id in &token_ids {
            if let Some(row) = self.embeddings.row(id) {
                for (i, &v) in row.iter().enumerate() {
                    mean[i] += f16::from_bits(v).to_f32();
                }
                valid_tokens += 1;
            }
        }

        let vector: Vec<u16> = if valid_tokens > 0 {
            mean.iter()
                .map(|&v| f16::from_f32(v / valid_tokens as f32).to_bits())
                .collect()
        } else {
            vec![0; self.config.hidden_size]
        };

        Ok(ChunkRecord {
            path: path.to_string_lossy().into_owned(),
            mtime,
            chunk_index: index,
            text: text.to_string(),
            token_ids,
            vector,
        })
    }
}

#[cfg(target_os = "macos")]
fn set_background_qos() {
    // Mark this background watcher thread as Quality-of-Service "background".
    // This tells the kernel to schedule it on efficiency cores and to avoid
    // waking performance cores, matching the low-overhead posture.
    unsafe {
        let _ = libc::pthread_set_qos_class_self_np(libc::qos_class_t::QOS_CLASS_BACKGROUND, 0);
    }
}

#[cfg(not(target_os = "macos"))]
fn set_background_qos() {}

fn collect_files(dir: &Path, depth: usize, out: &mut Vec<(PathBuf, u64)>) {
    if depth == 0 {
        return;
    }
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.filter_map(Result::ok) {
            let path = entry.path();
            if path.is_dir() {
                collect_files(&path, depth - 1, out);
            } else if is_tracked_file(&path) {
                if let Ok(meta) = fs::metadata(&path) {
                    let mtime = meta
                        .modified()
                        .ok()
                        .and_then(|t| t.duration_since(SystemTime::UNIX_EPOCH).ok())
                        .map(|d| d.as_secs())
                        .unwrap_or(0);
                    out.push((path, mtime));
                }
            }
        }
    }
}

fn is_tracked_file(path: &Path) -> bool {
    if !path.is_file() {
        return false;
    }
    if let Some(ext) = path.extension() {
        if let Some(ext) = ext.to_str() {
            return TRACKED_EXTENSIONS.contains(&ext.to_lowercase().as_str());
        }
    }
    false
}

fn read_limited_text(path: &Path, limit: usize) -> Result<String> {
    let file = fs::File::open(path)?;
    let mut buf = Vec::with_capacity(limit.min(1_048_576));
    file.take(limit as u64).read_to_end(&mut buf)?;
    // Drop invalid UTF-8 sequences silently.
    Ok(String::from_utf8_lossy(&buf).into_owned())
}

fn content_hash(path: &Path) -> String {
    if let Ok(bytes) = fs::read(path) {
        content_hash_bytes(&bytes)
    } else {
        String::new()
    }
}

fn content_hash_bytes(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hasher
        .finalize()
        .iter()
        .map(|b| format!("{:02x}", b))
        .collect()
}

fn path_key(path: &Path) -> String {
    content_hash_bytes(path.as_os_str().as_encoded_bytes())
}

fn meta_key(path_key: &str) -> Vec<u8> {
    let mut key = METADATA_PREFIX.to_vec();
    key.extend_from_slice(path_key.as_bytes());
    key
}

fn chunk_key(path_key: &str, index: usize) -> Vec<u8> {
    format!("{}:{}:{}", CHUNK_PREFIX, path_key, index).into_bytes()
}
