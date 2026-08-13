//! Double-buffered, asynchronous state persistence.
//!
//! The main cognitive loop builds a lightweight snapshot of the soul matrix,
//! copies the live Candle `VarMap` tensors, and hands the payload to a
//! dedicated background thread.  The thread performs the physical 800+ ms
//! JSON + safetensors + connectome flush without ever blocking the main
//! async runtime.

use crate::{connectome_mmap, FullySapientSoulMatrix, MAX_MEMORY_NODES};
use candle_core::{safetensors, Tensor};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::thread;

/// Paths for the atomic state family.
#[derive(Clone, Debug)]
pub struct SavePaths {
    pub state: PathBuf,
    pub safetensors: PathBuf,
    pub weights: PathBuf,
    pub defense: PathBuf,
    pub network: PathBuf,
    pub connectome: PathBuf,
}

impl SavePaths {
    pub fn from(base: &Path) -> Self {
        Self {
            state: PathBuf::from(base),
            safetensors: PathBuf::from(base).with_extension("safetensors"),
            weights: PathBuf::from(base).with_extension("weights"),
            defense: PathBuf::from(base).with_extension("defense"),
            network: PathBuf::from(base).with_extension("network"),
            connectome: PathBuf::from(base).with_extension("connectome"),
        }
    }
}

/// A detached, immutable snapshot that the background writer owns.
#[repr(align(128))]
pub struct SavePayload {
    pub snapshot: FullySapientSoulMatrix,
    pub weights: Option<HashMap<String, Tensor>>,
    pub paths: SavePaths,
    /// Optional callback invoked with the flush duration in milliseconds once
    /// the physical write is complete.
    pub completion: Option<Box<dyn FnOnce(u64) + Send>>,
}

impl SavePayload {
    /// Build a payload from the live mind.  The live lock is only held while
    /// copying data, not while writing to disk.
    pub fn from_mind(
        mind: &FullySapientSoulMatrix,
        base: &Path,
        weights: Option<HashMap<String, Tensor>>,
    ) -> Self {
        Self {
            snapshot: mind.save_snapshot(),
            weights,
            paths: SavePaths::from(base),
            completion: None,
        }
    }

    /// Physical flush to disk.  This runs in the background thread.
    pub fn write(mut self) -> std::io::Result<()> {
        let start = std::time::Instant::now();
        // Stream compact JSON directly to a buffered, atomically-renamed file.
        let tmp = self.paths.state.with_extension("json.tmp");
        if let Ok(file) = std::fs::File::create(&tmp) {
            let writer = std::io::BufWriter::new(file);
            if serde_json::to_writer(writer, &self.snapshot).is_ok() {
                let _ = std::fs::rename(&tmp, &self.paths.state);
            }
        }

        // Real Candle tensor weights to a separate safetensors file.
        // The incoming `weights` map contains shallow Tensor Arc references;
        // copy the underlying storage on this background thread before writing,
        // then release the copies once the flush is complete.
        if let Some(weights) = self.weights {
            let mut copied = HashMap::with_capacity(weights.len());
            for (name, t) in &weights {
                if let Ok(t) = t.copy() {
                    copied.insert(name.clone(), t);
                }
            }
            let _ = safetensors::save(&copied, &self.paths.safetensors);
        }

        // Persist 2048-D connectome embeddings and 576-D brain states via mmap.
        if let Ok(store) =
            connectome_mmap::ConnectomeMmap::open(&self.paths.connectome, MAX_MEMORY_NODES)
        {
            let _ = store.persist(&self.snapshot.associative_memory_network);
        }

        // Legacy scalar weight persistence, file quarantine, and network stack.
        let _ = self.snapshot.weight_persistence.save_to_file(&self.paths.weights);
        let _ = self.snapshot.file_defense.save_state(&self.paths.defense);
        let _ = self.snapshot.network_stack.save_state(&self.paths.network);

        let duration_ms = start.elapsed().as_millis() as u64;
        if let Some(callback) = self.completion.take() {
            callback(duration_ms);
        }

        Ok(())
    }
}

/// Dedicated background worker that owns the state-save file flush.
pub struct StateSaveWorker {
    sender: mpsc::Sender<SavePayload>,
}

impl StateSaveWorker {
    /// Spawn a background thread that waits for payloads and writes them.
    pub fn new() -> Self {
        let (sender, receiver) = mpsc::channel::<SavePayload>();
        thread::spawn(move || {
            while let Ok(payload) = receiver.recv() {
                if let Err(e) = payload.write() {
                    tracing::warn!("Background state save flush failed: {}", e);
                }
            }
        });
        Self { sender }
    }

    /// Queue a snapshot to be flushed in the background.  Non-blocking.
    pub fn submit(&self, payload: SavePayload) {
        if self.sender.send(payload).is_err() {
            tracing::warn!("State save worker thread has exited; dropping payload");
        }
    }
}
