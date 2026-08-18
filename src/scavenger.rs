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
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, SystemTime};
use sysinfo::Disks;
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
const SKIP_DIR_NAMES: &[&str] = &[
    ".git",
    "node_modules",
    "target",
    "qwen3b_ane_shards",
    "compiled",
    "logs",
    ".cache",
    "__pycache__",
    "dist",
    "build",
    "state-backup",
    "wild_workspace",
    ".venv",
    "sapient_agi_soul",
];
const METADATA_PREFIX: &[u8] = b"scavenger:meta:";
const CHUNK_PREFIX: &str = "scavenger:chunk";

/// Runtime configuration for the scavenger.
#[derive(Clone, Debug)]
pub struct ScavengerConfig {
    /// One or more directories to watch and index.
    pub watch_dirs: Vec<PathBuf>,
    pub sled_db_path: PathBuf,
    pub tokenizer_path: PathBuf,
    pub embedding_path: PathBuf,
    pub hidden_size: usize,
    pub vocab_size: usize,
    /// If true, remove an existing Sled database before opening it.
    pub reset_db: bool,
}

impl ScavengerConfig {
    /// Build from environment with sensible fallbacks.
    pub fn from_env() -> Result<Self> {
        // Support multiple colon-separated watch roots. This lets the daemon
        // index the Bad Apple repo plus adjacent local source folders. If no
        // explicit list is set, auto-discover repositories under /Users/savag3.
        let watch_dirs = if let Some(raw) = std::env::var_os("BADAPPLE_SCAVENGE_DIRS") {
            std::env::split_paths(&raw).collect::<Vec<_>>()
        } else {
            Self::discover_source_dirs()
        };

        let sled_db_path = std::env::var_os("BADAPPLE_SLED_DB_PATH")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("strategy_db"));

        let reset_db = std::env::var("BADAPPLE_SLED_DB_RESET")
            .ok()
            .is_some_and(|v| v == "1" || v.eq_ignore_ascii_case("true"));

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
            watch_dirs,
            sled_db_path,
            tokenizer_path,
            embedding_path,
            hidden_size,
            vocab_size,
            reset_db,
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

    /// Heuristic to identify adjacent local source-code repositories.
    fn looks_like_source_repo(path: &Path) -> bool {
        let markers = [
            ".git",
            "Cargo.toml",
            "Package.swift",
            "pyproject.toml",
            "setup.py",
            "go.mod",
            "package.json",
        ];
        markers.iter().any(|marker| path.join(marker).exists())
    }

    /// Discover the Bad Apple repo and any adjacent local source folders.
    fn discover_source_dirs() -> Vec<PathBuf> {
        let home = PathBuf::from("/Users/savag3");
        let primary = home.join("bad_apple");
        let mut dirs = vec![primary.clone()];
        if let Ok(entries) = fs::read_dir(&home) {
            for entry in entries.filter_map(Result::ok) {
                let path = entry.path();
                if path.is_dir() && path != primary && Self::looks_like_source_repo(&path) {
                    dirs.push(path);
                }
            }
        }
        dirs
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
        for dir in &config.watch_dirs {
            if !dir.is_file() {
                fs::create_dir_all(dir).ok();
            }
        }

        // Reset a corrupted Sled tree on request. This is intended for initial
        // installs where the strategy/scavenger DB has become unreadable.
        if config.reset_db && config.sled_db_path.exists() {
            tracing::info!(
                "scavenger resetting Sled database at {}",
                config.sled_db_path.display()
            );
            let _ = fs::remove_dir_all(&config.sled_db_path);
        }

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

        // Open Sled, recovering from corruption if necessary by deleting the
        // directory and trying once more. This prevents the daemon from giving
        // up on the whole subsystem because of a stale corrupted tree.
        let db = match sled::open(&config.sled_db_path) {
            Ok(db) => db,
            Err(error) => {
                tracing::warn!(
                    "scavenger Sled open failed ({}); removing and retrying",
                    error
                );
                let _ = fs::remove_dir_all(&config.sled_db_path);
                sled::open(&config.sled_db_path).with_context(|| {
                    format!(
                        "unable to open Sled database at {}",
                        config.sled_db_path.display()
                    )
                })?
            }
        };

        // Log disk headroom for the Sled database before we start writing.
        if let Some((free_mb, total_mb)) = disk_headroom_mb(&config.sled_db_path) {
            tracing::info!(
                "scavenger disk headroom: {} MB free / {} MB total",
                free_mb,
                total_mb
            );
        }

        Ok(Self {
            config,
            tokenizer,
            embeddings,
            db: Arc::new(db),
        })
    }

    /// Run the watcher loop.  The filesystem watcher itself runs on a
    /// dedicated background thread; all indexing work is offloaded to
    /// `spawn_blocking` so the main executor stays responsive.
    pub async fn run(self: Arc<Self>) -> Result<()> {
        let (tx, mut rx) = mpsc::unbounded_channel::<Event>();
        let watch_dirs: Vec<PathBuf> = self
            .config
            .watch_dirs
            .iter()
            .filter(|dir| dir.is_dir())
            .cloned()
            .collect();

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
                for dir in &watch_dirs {
                    if let Err(error) = watcher.watch(dir, RecursiveMode::Recursive) {
                        tracing::warn!("scavenger unable to watch {}: {}", dir.display(), error);
                    } else {
                        tracing::info!("scavenger watching {}", dir.display());
                    }
                }
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
                        spawn_blocking(move || {
                            self_ref.purge_file(&path);
                            if let Err(error) = self_ref.write_path_catalog() {
                                tracing::warn!(
                                    "scavenger failed to update path catalog: {}",
                                    error
                                );
                            }
                        });
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
                            spawn_blocking(move || {
                                self_ref.purge_file(&path);
                                if let Err(error) = self_ref.write_path_catalog() {
                                    tracing::warn!(
                                        "scavenger failed to update path catalog: {}",
                                        error
                                    );
                                }
                            });
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
                    if let Err(error) = self_ref.write_path_catalog() {
                        tracing::warn!("scavenger failed to update path catalog: {}", error);
                    }
                });
            }
        }

        Ok(())
    }

    /// Index existing files in the watch directories before live events arrive.
    pub async fn seed_existing_files(self: &Arc<Self>) -> Result<()> {
        let mut files = Vec::new();
        for dir in &self.config.watch_dirs {
            if dir.is_dir() {
                collect_files(dir, WATCH_DEPTH, &mut files);
            } else if dir.exists() {
                tracing::warn!("scavenger watch path is not a directory: {}", dir.display());
            } else {
                tracing::warn!("scavenger watch path does not exist: {}", dir.display());
            }
        }

        if !files.is_empty() {
            let file_count = files.len();
            tracing::info!(
                "scavenger starting first repository index pass: {} files across {} watch roots",
                file_count,
                self.config.watch_dirs.len()
            );

            if let Some((free_mb, _)) = disk_headroom_mb(&self.config.sled_db_path) {
                tracing::info!("scavenger pre-index disk headroom: {} MB free", free_mb);
            }

            let self_ref = self.clone();
            spawn_blocking(move || {
                for (path, mtime) in files {
                    if let Err(error) = self_ref.index_file(&path, mtime) {
                        tracing::warn!("scavenger failed to seed {}: {}", path.display(), error);
                    }
                }
            })
            .await?;

            self.write_path_catalog()?;
            let files_indexed = count_with_prefix(&self.db, METADATA_PREFIX);
            let chunks_indexed = count_with_prefix(&self.db, CHUNK_PREFIX.as_bytes());
            let (free_mb, total_mb) = disk_headroom_mb(&self.config.sled_db_path).unwrap_or((0, 0));
            tracing::info!(
                "scavenger first repository index pass cleared: {} files, {} chunks, {} MB free / {} MB total",
                files_indexed,
                chunks_indexed,
                free_mb,
                total_mb
            );
        } else {
            tracing::info!("scavenger found no tracked files in watch directories");
        }
        Ok(())
    }

    fn write_path_catalog(&self) -> Result<()> {
        let mut paths = self
            .db
            .scan_prefix(METADATA_PREFIX)
            .filter_map(|entry| entry.ok())
            .filter_map(|(_, raw)| serde_json::from_slice::<FileMetadata>(&raw).ok())
            .map(|metadata| metadata.path)
            .collect::<Vec<_>>();
        paths.sort();
        paths.dedup();

        let parent = self
            .config
            .sled_db_path
            .parent()
            .unwrap_or_else(|| Path::new("."));
        let catalog = parent.join("scavenger_paths.json");
        let temporary = parent.join("scavenger_paths.json.tmp");
        fs::write(&temporary, serde_json::to_vec(&paths)?)?;
        fs::rename(&temporary, &catalog)?;
        Ok(())
    }

    /// Remove all records and metadata for a deleted file.
    fn purge_file(&self, path: &Path) {
        let path_key = path_key(path);
        let meta_key = meta_key(&path_key);
        {
            if let Ok(Some(raw)) = self.db.get(&meta_key) {
                if let Ok(meta) = serde_json::from_slice::<FileMetadata>(&raw) {
                    for i in 0..meta.chunk_count {
                        let chunk_key = chunk_key(&path_key, i);
                        let _ = self.db.remove(chunk_key);
                    }
                }
            }
            let _ = self.db.remove(&meta_key);
            let _ = self.db.flush();
        }
        tracing::info!("scavenger purged {}", path.display());
    }

    /// Read, chunk, tokenize, embed, and persist a file.
    fn index_file(&self, path: &Path, mtime: u64) -> Result<()> {
        let path_key = path_key(path);
        let meta_key = meta_key(&path_key);

        // Eagerly slurp the file into memory and drop the descriptor before
        // any tokenization, embedding, or database work crosses a scheduling
        // boundary.
        let text = {
            let mut text = fs::read_to_string(path)
                .with_context(|| format!("scavenger unable to read {}", path.display()))?;
            if text.len() > CHUNK_BYTE_LIMIT {
                let mut byte_len = CHUNK_BYTE_LIMIT;
                while !text.is_char_boundary(byte_len) && byte_len > 0 {
                    byte_len -= 1;
                }
                text.truncate(byte_len);
            }
            text
        };
        let current_hash = content_hash_bytes(text.as_bytes());

        // Skip if unchanged.
        if let Ok(Some(raw)) = self.db.get(&meta_key) {
            if let Ok(meta) = serde_json::from_slice::<FileMetadata>(&raw) {
                if meta.content_hash == current_hash {
                    return Ok(());
                }
            }
        }

        // Delete any previous version before writing the new one.
        self.purge_file(path);

        let chunks = self.chunk_text(&text);
        let chunk_count = chunks.len();

        // Isolate all Sled writes in a block and force a flush before releasing
        // the page locks back to the OS kernel.
        {
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
            self.db.flush()?;
        }

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
            if is_ignored_path(&path) {
                continue;
            }
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

fn is_ignored_path(path: &Path) -> bool {
    path.components().any(|component| {
        if let Some(name) = component.as_os_str().to_str() {
            SKIP_DIR_NAMES
                .iter()
                .any(|skip| name.eq_ignore_ascii_case(skip) || name.starts_with(skip))
        } else {
            false
        }
    })
}

fn is_tracked_file(path: &Path) -> bool {
    if !path.is_file() || is_ignored_path(path) {
        return false;
    }
    if let Some(ext) = path.extension() {
        if let Some(ext) = ext.to_str() {
            return TRACKED_EXTENSIONS.contains(&ext.to_lowercase().as_str());
        }
    }
    false
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

/// Return the number of records whose keys start with `prefix`.
fn count_with_prefix(db: &sled::Db, prefix: &[u8]) -> usize {
    db.scan_prefix(prefix).count()
}

/// Return the free / total space in megabytes for the disk holding `path`.
fn disk_headroom_mb(path: &Path) -> Option<(u64, u64)> {
    // Normalize to an absolute path so we can find the containing mount point.
    let canonical = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
    let disks = Disks::new_with_refreshed_list();
    let mut best: Option<(u64, u64)> = None;
    let mut best_len = 0;
    for disk in disks.list() {
        if let Some(mount) = disk.mount_point().to_str() {
            if canonical.starts_with(mount) && mount.len() >= best_len {
                best = Some((
                    disk.available_space() / 1_048_576,
                    disk.total_space() / 1_048_576,
                ));
                best_len = mount.len();
            }
        }
    }
    best
}
