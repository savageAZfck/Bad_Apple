//! APFS file scavenger: watches a local directory, chunks text files,
//! tokenizes each chunk through the Qwen tokenizer, looks up token vectors
//! in the memory-mapped 622 MB FP16 embedding table, and persists the
//! chunked text, metadata hashes, token IDs, and vector state to redb.
//!
//! All processing is local and offline: no network sockets are opened and
//! the only external I/O is to the filesystem, tokenizer, embedding table,
//! and redb database.

use crate::redb_kv;
use anyhow::{anyhow, Context, Result};
use half::f16;
use notify::{Event, EventKind, RecommendedWatcher, RecursiveMode, Watcher};
use redb::Database;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
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
        // explicit list is set, auto-discover repositories under $HOME.
        let watch_dirs = if let Some(raw) = std::env::var_os("BADAPPLE_SCAVENGE_DIRS") {
            std::env::split_paths(&raw).collect::<Vec<_>>()
        } else {
            Self::discover_source_dirs()
        };

        let sled_db_path = std::env::var_os("BADAPPLE_SLED_DB_PATH")
            .map_or_else(|| PathBuf::from("strategy_db"), PathBuf::from);

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
                        .and_then(serde_json::Value::as_u64)
                        .unwrap_or(2048) as usize;
                    let vocab = model
                        .get("vocab_size")
                        .and_then(serde_json::Value::as_u64)
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
        let home =
            std::env::var("HOME").map_or_else(|_| PathBuf::from("/Users/YourName"), PathBuf::from);
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
        // SAFETY: Mmap::map is safe because `file` was just opened successfully and maps it
        // read-only. The returned mapping is bounds-checked against the file length and is
        // only read (never mutated), so the read-only shared mapping is sound.
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
    db: Arc<Database>,
}

impl Scavenger {
    /// Open the tokenizer, embedding table, and Sled database.
    pub fn open(config: ScavengerConfig) -> Result<Self> {
        for dir in &config.watch_dirs {
            if !dir.is_file() {
                fs::create_dir_all(dir).ok();
            }
        }

        // Reset a corrupted redb tree on request. This is intended for initial
        // installs where the strategy/scavenger DB has become unreadable.
        if config.reset_db && config.sled_db_path.exists() {
            tracing::info!(
                "scavenger resetting redb database at {}",
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

        // Open redb, recovering from corruption if necessary by deleting the
        // directory and trying once more. This prevents the daemon from giving
        // up on the whole subsystem because of a stale corrupted tree.
        let db = match redb_kv::open(&config.sled_db_path) {
            Ok(db) => db,
            Err(error) => {
                tracing::warn!(
                    "scavenger redb open failed ({}); removing and retrying",
                    error
                );
                let _ = fs::remove_dir_all(&config.sled_db_path);
                redb_kv::open(&config.sled_db_path).with_context(|| {
                    format!(
                        "unable to open redb database at {}",
                        config.sled_db_path.display()
                    )
                })?
            }
        };

        // Log disk headroom for the redb database before we start writing.
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
        // Canonical watch roots used to reject events whose path escapes the
        // watched tree (e.g. via a symlink inserted after the watcher started).
        let canonical_roots: Vec<PathBuf> = watch_dirs
            .iter()
            .filter_map(|d| fs::canonicalize(d).ok())
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
                        if !path_under_roots(path, &canonical_roots) {
                            continue;
                        }
                        if let Ok(meta) = fs::metadata(path) {
                            let mtime = meta
                                .modified()
                                .ok()
                                .and_then(|t| t.duration_since(SystemTime::UNIX_EPOCH).ok())
                                .map_or(0, |d| d.as_secs());
                            pending.insert(path.clone(), mtime);
                        }
                    }
                    EventKind::Remove(_) if is_tracked_file(path) => {
                        if !path_under_roots(path, &canonical_roots) {
                            continue;
                        }
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
                            if !path_under_roots(path, &canonical_roots) {
                                continue;
                            }
                            if let Ok(meta) = fs::metadata(path) {
                                let mtime = meta
                                    .modified()
                                    .ok()
                                    .and_then(|t| t.duration_since(SystemTime::UNIX_EPOCH).ok())
                                    .map_or(0, |d| d.as_secs());
                                pending.insert(path.clone(), mtime);
                            }
                        }
                        EventKind::Remove(_) if is_tracked_file(path) => {
                            if !path_under_roots(path, &canonical_roots) {
                                continue;
                            }
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

        if files.is_empty() {
            tracing::info!("scavenger found no tracked files in watch directories");
        } else {
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
        }
        Ok(())
    }

    fn write_path_catalog(&self) -> Result<()> {
        let mut paths = redb_kv::scan_prefix(&self.db, METADATA_PREFIX)
            .unwrap_or_default()
            .into_iter()
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
        let mut keys: Vec<Vec<u8>> = Vec::new();
        if let Ok(Some(raw)) = redb_kv::get(&self.db, &meta_key) {
            if let Ok(meta) = serde_json::from_slice::<FileMetadata>(&raw) {
                for i in 0..meta.chunk_count {
                    keys.push(chunk_key(&path_key, i));
                }
            }
        }
        keys.push(meta_key);
        let key_refs: Vec<&[u8]> = keys.iter().map(|k| k.as_slice()).collect();
        let _ = redb_kv::remove_many(&self.db, &key_refs);
        tracing::info!("scavenger purged {}", path.display());
    }

    /// Read, chunk, tokenize, embed, and persist a file.
    fn index_file(&self, path: &Path, mtime: u64) -> Result<()> {
        let path_key = path_key(path);
        let meta_key = meta_key(&path_key);

        // Eagerly read the file but cap at CHUNK_BYTE_LIMIT + 1 byte so we
        // never slurp a multi-GB file into memory.  Open with O_NOFOLLOW to
        // atomically reject symlinks at the leaf, eliminating the TOCTOU race
        // between a separate lstat check and the subsequent open().
        let text = {
            use std::io::Read;
            use std::os::unix::fs::OpenOptionsExt;
            let file = fs::OpenOptions::new()
                .read(true)
                .custom_flags(libc::O_NOFOLLOW)
                .open(path)
                .with_context(|| {
                    format!(
                        "scavenger unable to open (or symlink rejected) {}",
                        path.display()
                    )
                })?;
            // fstat on the fd to confirm a regular file (defends against
            // FIFOs, devices, etc. that O_NOFOLLOW alone doesn't catch).
            let meta = file
                .metadata()
                .with_context(|| format!("scavenger unable to fstat {}", path.display()))?;
            if !meta.file_type().is_file() {
                return Ok(());
            }
            let mut reader = std::io::BufReader::new(file);
            let mut buf = vec![0u8; CHUNK_BYTE_LIMIT + 1];
            let mut filled = 0;
            while filled < buf.len() {
                let n = reader
                    .read(&mut buf[filled..])
                    .with_context(|| format!("scavenger unable to read {}", path.display()))?;
                if n == 0 {
                    break;
                }
                filled += n;
            }
            buf.truncate(filled.min(CHUNK_BYTE_LIMIT));
            // Ensure we end on a UTF-8 char boundary.
            let mut byte_len = buf.len();
            while byte_len > 0 && std::str::from_utf8(&buf[..byte_len]).is_err() {
                byte_len -= 1;
            }
            buf.truncate(byte_len);
            String::from_utf8_lossy(&buf).into_owned()
        };
        let current_hash = content_hash_bytes(text.as_bytes());

        // Skip if unchanged.
        if let Ok(Some(raw)) = redb_kv::get(&self.db, &meta_key) {
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

        // Batch all redb writes for this file into a single transaction.
        let mut items: Vec<(Vec<u8>, Vec<u8>)> = Vec::with_capacity(chunk_count + 1);
        for (i, chunk_text) in chunks.iter().enumerate() {
            let record = self.embed_chunk(path, mtime, i, chunk_text)?;
            let key = chunk_key(&path_key, i);
            let value = serde_json::to_vec(&record)?;
            items.push((key, value));
        }
        let metadata = FileMetadata {
            path: path.to_string_lossy().into_owned(),
            content_hash: current_hash,
            chunk_count,
        };
        items.push((meta_key, serde_json::to_vec(&metadata)?));
        let item_refs: Vec<(&[u8], &[u8])> = items
            .iter()
            .map(|(k, v)| (k.as_slice(), v.as_slice()))
            .collect();
        redb_kv::insert_many(&self.db, &item_refs)?;

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
            let tokens = if let Ok(encoding) = self.tokenizer.encode(paragraph, true) {
                encoding.get_ids().to_vec()
            } else {
                chunks.push(paragraph.to_string());
                continue;
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
                let candidate = format!("{last}\n\n{chunk}");
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
            .map_err(|e| anyhow!("unable to tokenize chunk: {e}"))?;
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
    // SAFETY: `pthread_set_qos_class_self_np` only affects the calling thread's scheduling
    // class. `QOS_CLASS_BACKGROUND` with a relative priority of 0 is a valid combination,
    // and the return value is ignored (best-effort). No shared state is touched.
    unsafe {
        let _ = libc::pthread_set_qos_class_self_np(libc::qos_class_t::QOS_CLASS_BACKGROUND, 0);
    }
}

#[cfg(not(target_os = "macos"))]
fn set_background_qos() {}

/// Verify that a (canonicalized) event path still falls under one of the
/// canonical watch roots.  This prevents a symlink created after the watcher
/// started from injecting paths outside the watched tree into the pending map.
fn path_under_roots(path: &Path, roots: &[PathBuf]) -> bool {
    let Ok(canonical) = fs::canonicalize(path) else {
        return false;
    };
    roots.iter().any(|root| canonical.starts_with(root))
}

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
            // Before recursing, check that this is a real directory, not a
            // symlink.  Using `symlink_metadata` avoids following a symlinked
            // directory that could escape the watched tree.
            let meta = match fs::symlink_metadata(&path) {
                Ok(m) => m,
                Err(_) => continue,
            };
            if meta.file_type().is_symlink() {
                // Reject symlinks outright: they could point outside the tree.
                continue;
            }
            if meta.file_type().is_dir() {
                collect_files(&path, depth - 1, out);
            } else if is_tracked_file(&path) {
                if let Ok(meta) = fs::metadata(&path) {
                    let mtime = meta
                        .modified()
                        .ok()
                        .and_then(|t| t.duration_since(SystemTime::UNIX_EPOCH).ok())
                        .map_or(0, |d| d.as_secs());
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
                .any(|skip| name.eq_ignore_ascii_case(skip))
        } else {
            false
        }
    })
}

fn is_tracked_file(path: &Path) -> bool {
    // Use symlink_metadata to reject symlinks: a symlink with a tracked
    // extension could point at an arbitrary file outside the watched tree.
    let Ok(meta) = fs::symlink_metadata(path) else {
        return false;
    };
    if meta.file_type().is_symlink() || !meta.is_file() {
        return false;
    }
    if is_ignored_path(path) {
        return false;
    }
    if let Some(ext) = path.extension() {
        if let Some(ext) = ext.to_str() {
            return TRACKED_EXTENSIONS.contains(&ext.to_lowercase().as_str());
        }
    }
    false
}

// ============================================================================
// Grounded code index — self-contained semantic recall over local sources.
//
// Unlike the FP16-table path above (which requires the ANE embedding dump),
// this mode embeds chunks with `tensor_brain::semantic_embedding`, the same
// deterministic 64-D mean-pooled token space the gatekeeper brain uses. No
// external model files are required, so it runs on any stock install.
// ============================================================================

/// Default location of the grounded code index.
pub const GROUNDED_INDEX_PATH: &str = "/var/lib/bad_apple/grounded_index";
const GROUNDED_META_PREFIX: &[u8] = b"grounded:meta:";
const GROUNDED_CHUNK_PREFIX: &str = "grounded:chunk";
/// Chunk size target in bytes (~512 tokens at ~4 chars/token).
const GROUNDED_CHUNK_BYTES: usize = 2_048;
const GROUNDED_CHUNK_OVERLAP: usize = 256;
/// Cap on chunks scanned per recall so a huge index stays interactive.
const MAX_RECALL_CHUNKS: usize = 50_000;
/// Cap on snippet text returned per hit.
const RECALL_SNIPPET_BYTES: usize = 1_500;

#[derive(Clone, Debug, Serialize, Deserialize)]
struct GroundedChunk {
    path: String,
    mtime: u64,
    chunk_index: usize,
    text: String,
    /// 64-D mean-pooled embedding stored as f32 bits for compactness.
    vector: Vec<u32>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct GroundedFileMeta {
    path: String,
    content_hash: String,
    chunk_count: usize,
}

/// A single recall hit.
#[derive(Clone, Debug, Serialize)]
pub struct RecallHit {
    pub path: String,
    pub chunk_index: usize,
    pub score: f64,
    pub text: String,
}

fn grounded_chunk_key(path_key: &str, index: usize) -> Vec<u8> {
    format!("{GROUNDED_CHUNK_PREFIX}:{path_key}:{index}").into_bytes()
}

fn grounded_meta_key(path_key: &str) -> Vec<u8> {
    let mut key = GROUNDED_META_PREFIX.to_vec();
    key.extend_from_slice(path_key.as_bytes());
    key
}

fn f32_slice_to_bits(v: &[f64]) -> Vec<u32> {
    v.iter().map(|&x| (x as f32).to_bits()).collect()
}

fn bits_to_f32_vec(bits: &[u32]) -> Vec<f64> {
    bits.iter().map(|&b| f32::from_bits(b) as f64).collect()
}

fn cosine_sim(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0;
    let mut na = 0.0;
    let mut nb = 0.0;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom > 0.0 {
        dot / denom
    } else {
        0.0
    }
}

/// Split text into byte-bounded overlapping chunks that break on paragraph
/// or line boundaries where possible.
fn grounded_chunk_text(text: &str) -> Vec<String> {
    if text.len() <= GROUNDED_CHUNK_BYTES {
        return vec![text.to_string()];
    }
    let mut chunks = Vec::new();
    let mut start = 0usize;
    while start < text.len() {
        let mut end = (start + GROUNDED_CHUNK_BYTES).min(text.len());
        // Back up to a UTF-8 boundary.
        while end > start && !text.is_char_boundary(end) {
            end -= 1;
        }
        // Prefer to end on a paragraph or line break.
        if end < text.len() {
            if let Some(pos) = text[start..end].rfind("\n\n") {
                if pos > GROUNDED_CHUNK_BYTES / 2 {
                    end = start + pos + 2;
                }
            } else if let Some(pos) = text[start..end].rfind('\n') {
                if pos > GROUNDED_CHUNK_BYTES / 2 {
                    end = start + pos + 1;
                }
            }
        }
        chunks.push(text[start..end].to_string());
        if end >= text.len() {
            break;
        }
        let mut next = end.saturating_sub(GROUNDED_CHUNK_OVERLAP);
        while next > start && !text.is_char_boundary(next) {
            next -= 1;
        }
        start = if next > start { next } else { end };
    }
    chunks
}

/// Read a file with O_NOFOLLOW (symlink-safe) and a byte cap.
fn grounded_read_file(path: &Path) -> Result<String> {
    use std::io::Read;
    use std::os::unix::fs::OpenOptionsExt;
    let file = fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW)
        .open(path)
        .with_context(|| format!("unable to open {}", path.display()))?;
    let meta = file.metadata()?;
    if !meta.file_type().is_file() {
        anyhow::bail!("not a regular file");
    }
    let mut reader = std::io::BufReader::new(file);
    let mut buf = vec![0u8; CHUNK_BYTE_LIMIT + 1];
    let mut filled = 0;
    while filled < buf.len() {
        let n = reader.read(&mut buf[filled..])?;
        if n == 0 {
            break;
        }
        filled += n;
    }
    buf.truncate(filled.min(CHUNK_BYTE_LIMIT));
    let mut byte_len = buf.len();
    while byte_len > 0 && std::str::from_utf8(&buf[..byte_len]).is_err() {
        byte_len -= 1;
    }
    buf.truncate(byte_len);
    Ok(String::from_utf8_lossy(&buf).into_owned())
}

/// Index one file into the grounded index. Returns the number of chunks
/// written (0 when unchanged).
fn index_grounded_file(db: &Database, path: &Path, mtime: u64) -> Result<usize> {
    let pkey = path_key(path);
    let meta_key = grounded_meta_key(&pkey);
    let text = grounded_read_file(path)?;
    let current_hash = content_hash_bytes(text.as_bytes());

    if let Ok(Some(raw)) = redb_kv::get(db, &meta_key) {
        if let Ok(meta) = serde_json::from_slice::<GroundedFileMeta>(&raw) {
            if meta.content_hash == current_hash {
                return Ok(0);
            }
        }
    }

    // Purge any previous version.
    if let Ok(Some(raw)) = redb_kv::get(db, &meta_key) {
        if let Ok(meta) = serde_json::from_slice::<GroundedFileMeta>(&raw) {
            let keys: Vec<Vec<u8>> = (0..meta.chunk_count)
                .map(|i| grounded_chunk_key(&pkey, i))
                .collect();
            let refs: Vec<&[u8]> = keys.iter().map(|k| k.as_slice()).collect();
            let _ = redb_kv::remove_many(db, &refs);
        }
    }

    let chunks = grounded_chunk_text(&text);
    let chunk_count = chunks.len();
    let mut items: Vec<(Vec<u8>, Vec<u8>)> = Vec::with_capacity(chunk_count + 1);
    for (i, chunk_text) in chunks.iter().enumerate() {
        let emb = crate::tensor_brain::semantic_embedding(chunk_text);
        let record = GroundedChunk {
            path: path.to_string_lossy().into_owned(),
            mtime,
            chunk_index: i,
            text: chunk_text.clone(),
            vector: f32_slice_to_bits(&emb),
        };
        items.push((grounded_chunk_key(&pkey, i), serde_json::to_vec(&record)?));
    }
    let meta = GroundedFileMeta {
        path: path.to_string_lossy().into_owned(),
        content_hash: current_hash,
        chunk_count,
    };
    items.push((meta_key, serde_json::to_vec(&meta)?));
    let refs: Vec<(&[u8], &[u8])> = items
        .iter()
        .map(|(k, v)| (k.as_slice(), v.as_slice()))
        .collect();
    redb_kv::insert_many(db, &refs)?;
    Ok(chunk_count)
}

/// Index every tracked file under `dirs` (one-shot pass, no watcher).
/// Returns `(files_indexed, chunks_written)` — files whose content hash was
/// unchanged are skipped, so repeat runs are incremental and cheap.
pub fn index_directories(dirs: &[PathBuf]) -> Result<(usize, usize)> {
    index_directories_at(dirs, Path::new(GROUNDED_INDEX_PATH))
}

/// Same as `index_directories` but against an explicit database path
/// (used by tests and `BADAPPLE_GROUNDED_INDEX` overrides).
pub fn index_directories_at(dirs: &[PathBuf], db_path: &Path) -> Result<(usize, usize)> {
    let db = redb_kv::open(db_path)?;
    let mut files: Vec<(PathBuf, u64)> = Vec::new();
    for dir in dirs {
        if dir.is_dir() {
            collect_files(dir, WATCH_DEPTH, &mut files);
        } else if dir.is_file() && is_tracked_file(dir) {
            if let Ok(meta) = fs::metadata(dir) {
                let mtime = meta
                    .modified()
                    .ok()
                    .and_then(|t| t.duration_since(SystemTime::UNIX_EPOCH).ok())
                    .map_or(0, |d| d.as_secs());
                files.push((dir.clone(), mtime));
            }
        }
    }
    let mut indexed = 0usize;
    let mut chunks = 0usize;
    for (path, mtime) in files {
        match index_grounded_file(&db, &path, mtime) {
            Ok(n) => {
                if n > 0 {
                    indexed += 1;
                    chunks += n;
                }
            }
            Err(e) => {
                tracing::warn!("grounded index failed for {}: {}", path.display(), e);
            }
        }
    }
    Ok((indexed, chunks))
}

/// Recall the `k` most semantically similar indexed chunks for `query`.
pub fn recall(query: &str, k: usize) -> Result<Vec<RecallHit>> {
    recall_at(query, k, Path::new(GROUNDED_INDEX_PATH))
}

/// Same as `recall` but against an explicit database path.
pub fn recall_at(query: &str, k: usize, db_path: &Path) -> Result<Vec<RecallHit>> {
    if !db_path.join("db.redb").exists() && !db_path.exists() {
        return Ok(Vec::new());
    }
    let db = match redb_kv::open(db_path) {
        Ok(db) => db,
        Err(_) => return Ok(Vec::new()),
    };
    let qemb = crate::tensor_brain::semantic_embedding(query);
    let terms = distinctive_terms(query);
    let mut hits: Vec<RecallHit> = Vec::new();
    let mut scanned = 0usize;
    for (_key, raw) in redb_kv::scan_prefix(&db, GROUNDED_CHUNK_PREFIX.as_bytes())? {
        scanned += 1;
        if scanned > MAX_RECALL_CHUNKS {
            break;
        }
        let Ok(chunk) = serde_json::from_slice::<GroundedChunk>(&raw) else {
            continue;
        };
        let vec = bits_to_f32_vec(&chunk.vector);
        // Hybrid score: cosine similarity plus a lexical bonus for exact
        // identifier matches — the 64-D semantic space is coarse, and code
        // queries hinge on literal names (symlink, O_NOFOLLOW, module names).
        let score = cosine_sim(&qemb, &vec) + lexical_bonus(&terms, &chunk.text, &chunk.path);
        hits.push(RecallHit {
            path: chunk.path,
            chunk_index: chunk.chunk_index,
            score,
            text: chunk.text.chars().take(RECALL_SNIPPET_BYTES).collect(),
        });
    }
    hits.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    hits.truncate(k);
    Ok(hits)
}

/// Lowercase alphanumeric/underscore tokens ≥3 chars minus common stopwords —
/// the words that actually discriminate one chunk of code from another.
fn distinctive_terms(query: &str) -> Vec<String> {
    const STOPWORDS: &[&str] = &[
        "the",
        "and",
        "for",
        "with",
        "that",
        "this",
        "from",
        "what",
        "which",
        "how",
        "does",
        "are",
        "was",
        "were",
        "you",
        "your",
        "its",
        "his",
        "her",
        "their",
        "our",
        "can",
        "could",
        "should",
        "would",
        "will",
        "shall",
        "may",
        "might",
        "must",
        "not",
        "but",
        "all",
        "any",
        "each",
        "some",
        "into",
        "over",
        "under",
        "between",
        "about",
        "when",
        "where",
        "who",
        "whom",
        "why",
        "use",
        "used",
        "using",
        "code",
        "file",
        "function",
        "module",
        "implement",
        "implemented",
        "implements",
        "implementation",
        "work",
        "works",
        "working",
        "codebase",
        "repo",
        "repository",
        "bad",
        "apple",
        "system",
    ];
    query
        .split(|c: char| !(c.is_alphanumeric() || c == '_'))
        .flat_map(|tok| {
            // Split snake_case/camelCase so "automation_cage" also yields
            // "automation" and "cage" as query terms.
            let mut parts = vec![tok.to_string()];
            parts.extend(tok.split('_').map(str::to_string));
            let mut split_case = String::new();
            let mut extra: Vec<String> = Vec::new();
            for ch in tok.chars() {
                if ch.is_uppercase() && !split_case.is_empty() {
                    extra.push(split_case.clone());
                    split_case.clear();
                }
                split_case.push(ch);
            }
            if !split_case.is_empty() && split_case != tok {
                extra.push(split_case);
            }
            parts.extend(extra);
            parts
        })
        .map(|t| t.to_lowercase())
        .filter(|t| t.len() >= 3 && !STOPWORDS.contains(&t.as_str()))
        .fold(Vec::new(), |mut acc, t| {
            if !acc.contains(&t) {
                acc.push(t);
            }
            acc
        })
}

/// Bonus for literal term presence: fraction of distinctive terms found in the
/// chunk text (weight 0.4) plus a bonus when a term hits the file path itself
/// (weight 0.8) — a named module should outrank a passing mention, and a path
/// match is a far stronger signal than body text since filenames are chosen
/// identifiers rather than prose.
fn lexical_bonus(terms: &[String], text: &str, path: &str) -> f64 {
    if terms.is_empty() {
        return 0.0;
    }
    let hay = text.to_lowercase();
    let hay_path = path.to_lowercase();
    let mut text_hits = 0usize;
    let mut path_hits = 0usize;
    for term in terms {
        if hay.contains(term.as_str()) {
            text_hits += 1;
        }
        if hay_path.contains(term.as_str()) {
            path_hits += 1;
        }
    }
    0.4 * (text_hits as f64 / terms.len() as f64) + 0.8 * (path_hits as f64 / terms.len() as f64)
}

/// Number of grounded chunks currently in the index.
pub fn grounded_chunk_count() -> usize {
    grounded_chunk_count_at(Path::new(GROUNDED_INDEX_PATH))
}

/// Same as `grounded_chunk_count` but against an explicit database path.
pub fn grounded_chunk_count_at(db_path: &Path) -> usize {
    let Ok(db) = redb_kv::open(db_path) else {
        return 0;
    };
    redb_kv::count_prefix(&db, GROUNDED_CHUNK_PREFIX.as_bytes()).unwrap_or(0)
}

fn content_hash_bytes(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
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
    format!("{CHUNK_PREFIX}:{path_key}:{index}").into_bytes()
}

/// Return the number of records whose keys start with `prefix`.
fn count_with_prefix(db: &Database, prefix: &[u8]) -> usize {
    redb_kv::count_prefix(db, prefix).unwrap_or_default()
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
