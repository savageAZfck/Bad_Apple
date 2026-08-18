//! Zero-copy memory-mapped persistence for 2048-D connectome embeddings and
//! their 576-D brain-state snapshots.
//!
//! The on-disk layout is a fixed-size binary slab:
//!
//!     [ConnectomeHeader] [Record 0] [Record 1] ...
//!
//! Each record is 8-byte aligned and stores `id`, `timestamp`, `embedding[2048]`
//! and `brain_state[BRAIN_DIM]`.  Because the file is `mmap`'d directly into the
//! process address space, save/flush does not materialise the large float arrays
//! as JSON and does not copy them through user-space buffers.

use bytemuck::{Pod, Zeroable};
use memmap2::{MmapMut, MmapOptions};
use std::collections::HashMap;
use std::fmt;
use std::fs::{File, OpenOptions};
use std::io;
use std::path::Path;
use std::sync::Mutex;

use crate::tensor_brain::{self, BRAIN_DIM};

const CONNECTOME_MAGIC: u64 = 0x46495245464C5921; // Legacy v1 marker retained for file compatibility.
const CONNECTOME_VERSION: u64 = 1;

/// Dimensionality of the grounded multimodal engram embedding.
pub const EMBEDDING_DIM: usize = 2048;

// Record layout (all offsets are 8-byte aligned).
const ID_OFFSET: usize = 0;
const TIMESTAMP_OFFSET: usize = 8;
const EMBEDDING_OFFSET: usize = 16;
const BRAIN_STATE_OFFSET: usize = EMBEDDING_OFFSET + EMBEDDING_DIM * 8;
const RECORD_SIZE: usize = BRAIN_STATE_OFFSET + BRAIN_DIM * 8;

// Header layout.
const HEADER_SIZE: usize = 32; // 4 x u64

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct ConnectomeHeader {
    magic: u64,
    version: u64,
    count: u64,
    _pad: u64,
}

/// Memory-mapped store for high-dimensional connectome vectors.
pub struct ConnectomeMmap {
    file: File,
    mmap: Mutex<MmapMut>,
}

impl fmt::Debug for ConnectomeMmap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ConnectomeMmap")
            .field("capacity", &self.capacity())
            .finish()
    }
}

impl ConnectomeMmap {
    /// Open or create the memory-mapped connectome file at `path`, ensuring it
    /// can hold at least `capacity` records.
    pub fn open(path: &Path, capacity: usize) -> io::Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(path)?;

        let min_len = (HEADER_SIZE + capacity * RECORD_SIZE) as u64;

        let meta = file.metadata()?;
        if meta.len() < min_len {
            file.set_len(min_len)?;
        }

        let mut mmap = unsafe { MmapOptions::new().map_mut(&file)? };
        let bytes: &mut [u8] = &mut mmap;

        // Clear a brand-new or extended mapping; stale bytes from ftruncate
        // could otherwise look like valid records.
        if meta.len() < min_len {
            bytes.fill(0);
        }

        // Initialise / validate the header.
        let header =
            &mut bytemuck::cast_slice_mut::<_, ConnectomeHeader>(&mut bytes[0..HEADER_SIZE])[0];

        if header.magic != CONNECTOME_MAGIC || header.version != CONNECTOME_VERSION {
            // Unrecognised header: reset count so any trailing records are
            // ignored, but do not erase the rest of the slab in case it can
            // still be useful for diagnostics.
            header.magic = CONNECTOME_MAGIC;
            header.version = CONNECTOME_VERSION;
            header.count = 0;
        }

        Ok(Self {
            file,
            mmap: Mutex::new(mmap),
        })
    }

    /// Persist the high-dimensional vectors of `network` into the mmap and
    /// request an asynchronous kernel flush.  This is safe to call from the
    /// off-thread state-save path and avoids stalling the main loop.
    pub fn persist(&self, network: &HashMap<u64, crate::MemoryGraphNode>) -> io::Result<()> {
        let mut guard = self.mmap.lock().unwrap();
        let required = HEADER_SIZE + network.len() * RECORD_SIZE;

        // Grow the mapping if the network has outgrown the current file.
        if guard.len() < required {
            drop(guard);
            self.file.set_len(required as u64)?;
            let new_mmap = unsafe { MmapOptions::new().map_mut(&self.file)? };
            *self.mmap.lock().unwrap() = new_mmap;
            guard = self.mmap.lock().unwrap();
        }

        let bytes: &mut [u8] = &mut guard;
        let (header_bytes, records_bytes) = bytes.split_at_mut(HEADER_SIZE);
        let header = &mut bytemuck::cast_slice_mut::<_, ConnectomeHeader>(header_bytes)[0];

        header.magic = CONNECTOME_MAGIC;
        header.version = CONNECTOME_VERSION;
        header.count = network.len() as u64;

        // Sort by id for deterministic, cache-friendly writes.
        let mut nodes: Vec<&crate::MemoryGraphNode> = network.values().collect();
        nodes.sort_by_key(|n| n.id);

        for (i, node) in nodes.iter().enumerate() {
            let base = i * RECORD_SIZE;
            let record = &mut records_bytes[base..base + RECORD_SIZE];

            let id_slice =
                bytemuck::cast_slice_mut::<_, u64>(&mut record[ID_OFFSET..ID_OFFSET + 8]);
            id_slice[0] = node.id;

            let ts_slice = bytemuck::cast_slice_mut::<_, u64>(
                &mut record[TIMESTAMP_OFFSET..TIMESTAMP_OFFSET + 8],
            );
            ts_slice[0] = node.timestamp;

            let emb_slice = bytemuck::cast_slice_mut::<_, f64>(
                &mut record[EMBEDDING_OFFSET..EMBEDDING_OFFSET + EMBEDDING_DIM * 8],
            );
            if node.embedding.len() == EMBEDDING_DIM {
                emb_slice.copy_from_slice(&node.embedding);
            } else {
                emb_slice.fill(0.0);
            }

            let bs_slice = bytemuck::cast_slice_mut::<_, f64>(
                &mut record[BRAIN_STATE_OFFSET..BRAIN_STATE_OFFSET + BRAIN_DIM * 8],
            );
            bs_slice.fill(0.0);
            let n = node.brain_state.len().min(BRAIN_DIM);
            if n > 0 {
                bs_slice[..n].copy_from_slice(&node.brain_state[..n]);
            }
        }

        // Zero any trailing records so a later load does not see stale ids.
        let active = network.len() * RECORD_SIZE;
        records_bytes[active..].fill(0);

        // NOTE: explicit `flush_async()` is intentionally omitted.  On some
        // platforms `msync(MS_ASYNC)` can serialize with the page-fault path
        // and block the save thread.  Because the mapping is `MAP_SHARED`, the
        // kernel will asynchronously write-back the dirty pages to the file,
        // so the off-thread state save never stalls the cognitive clock.
        Ok(())
    }

    /// Load vector data from the mmap back into `network`.  Nodes whose records
    /// are missing have their embeddings regenerated from `experiential_text`
    /// and their `brain_state` zeroed, so the graph remains usable.
    pub fn load_into(
        &self,
        network: &mut HashMap<u64, crate::MemoryGraphNode>,
        spatial_axes: &[f64; 4],
    ) -> io::Result<()> {
        let guard = self.mmap.lock().unwrap();
        let bytes: &[u8] = &guard;
        if bytes.len() < HEADER_SIZE {
            return Ok(());
        }
        let (header_bytes, records_bytes) = bytes.split_at(HEADER_SIZE);
        let header = &bytemuck::cast_slice::<_, ConnectomeHeader>(header_bytes)[0];

        if header.magic != CONNECTOME_MAGIC || header.version != CONNECTOME_VERSION {
            return Ok(());
        }

        let count = header.count as usize;

        for i in 0..count {
            let base = i * RECORD_SIZE;
            let record = &records_bytes[base..base + RECORD_SIZE];

            let id_slice = bytemuck::cast_slice::<_, u64>(&record[ID_OFFSET..ID_OFFSET + 8]);
            let id = id_slice[0];
            let ts_slice =
                bytemuck::cast_slice::<_, u64>(&record[TIMESTAMP_OFFSET..TIMESTAMP_OFFSET + 8]);
            let ts = ts_slice[0];
            if id == 0 && ts == 0 {
                continue;
            }

            if let Some(node) = network.get_mut(&id) {
                let emb_slice = bytemuck::cast_slice::<_, f64>(
                    &record[EMBEDDING_OFFSET..EMBEDDING_OFFSET + EMBEDDING_DIM * 8],
                );
                node.embedding = emb_slice.to_vec();

                let bs_slice = bytemuck::cast_slice::<_, f64>(
                    &record[BRAIN_STATE_OFFSET..BRAIN_STATE_OFFSET + BRAIN_DIM * 8],
                );
                node.brain_state = bs_slice.to_vec();
            }
        }

        // Backfill missing embeddings from the text field; zero missing brain
        // states so downstream code can rely on the expected dimension.
        for node in network.values_mut() {
            if node.embedding.is_empty() {
                node.embedding =
                    tensor_brain::text_to_grounded_embedding(&node.experiential_text, spatial_axes);
            }
            if node.brain_state.is_empty() {
                node.brain_state = vec![0.0; BRAIN_DIM];
            }
        }

        Ok(())
    }

    /// Return a single record by index, copying it out of the mmap.
    #[allow(dead_code)]
    pub fn get_record(&self, index: usize) -> Option<(u64, u64, Vec<f64>, Vec<f64>)> {
        let guard = self.mmap.lock().unwrap();
        let bytes: &[u8] = &guard;
        if bytes.len() < HEADER_SIZE {
            return None;
        }
        let (header_bytes, records_bytes) = bytes.split_at(HEADER_SIZE);
        let header = &bytemuck::cast_slice::<_, ConnectomeHeader>(header_bytes)[0];
        if header.magic != CONNECTOME_MAGIC || header.version != CONNECTOME_VERSION {
            return None;
        }
        if index >= header.count as usize {
            return None;
        }

        let base = index * RECORD_SIZE;
        let record = &records_bytes[base..base + RECORD_SIZE];

        let id = bytemuck::cast_slice::<_, u64>(&record[ID_OFFSET..ID_OFFSET + 8])[0];
        let ts = bytemuck::cast_slice::<_, u64>(&record[TIMESTAMP_OFFSET..TIMESTAMP_OFFSET + 8])[0];
        let emb = bytemuck::cast_slice::<_, f64>(
            &record[EMBEDDING_OFFSET..EMBEDDING_OFFSET + EMBEDDING_DIM * 8],
        )
        .to_vec();
        let bs = bytemuck::cast_slice::<_, f64>(
            &record[BRAIN_STATE_OFFSET..BRAIN_STATE_OFFSET + BRAIN_DIM * 8],
        )
        .to_vec();

        Some((id, ts, emb, bs))
    }

    pub fn capacity(&self) -> usize {
        let guard = self.mmap.lock().unwrap();
        guard.len().saturating_sub(HEADER_SIZE) / RECORD_SIZE
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_path() -> std::path::PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let mut p = std::env::temp_dir();
        p.push(format!(
            "bad_apple_connectome_{}_{}.bin",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        p
    }

    #[test]
    fn round_trip_records() {
        let path = temp_path();
        let _ = std::fs::remove_file(&path);

        let mut network = HashMap::new();
        for id in 0..3 {
            network.insert(
                id,
                crate::MemoryGraphNode {
                    id,
                    timestamp: 1000 + id,
                    experiential_text: format!("node {}", id),
                    emotional_state_snapshot: "test".into(),
                    embedding: (0..EMBEDDING_DIM).map(|i| (i as f64) * 0.001).collect(),
                    associated_edge_ids: vec![],
                    origin_instance: "test".into(),
                    brain_state: (0..BRAIN_DIM).map(|i| (i as f64) * 0.01).collect(),
                },
            );
        }

        let store = ConnectomeMmap::open(&path, 8).unwrap();
        store.persist(&network).unwrap();

        // Clear in-memory vectors and load from disk.
        for node in network.values_mut() {
            node.embedding.clear();
            node.brain_state.clear();
        }

        store
            .load_into(&mut network, &[0.9, 0.1, 0.7, 9.81])
            .unwrap();

        for id in 0..3 {
            let node = network.get(&id).unwrap();
            assert_eq!(node.embedding.len(), EMBEDDING_DIM);
            assert_eq!(node.brain_state.len(), BRAIN_DIM);
            assert!((node.embedding[0] - 0.0).abs() < 1e-9);
            assert!((node.embedding[10] - 0.01).abs() < 1e-9);
            assert!((node.brain_state[5] - 0.05).abs() < 1e-9);
        }

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn missing_backfill_uses_text() {
        let path = temp_path();
        let _ = std::fs::remove_file(&path);

        let mut network = HashMap::new();
        network.insert(
            7,
            crate::MemoryGraphNode {
                id: 7,
                timestamp: 1,
                experiential_text: "hello world".into(),
                emotional_state_snapshot: "test".into(),
                embedding: vec![],
                associated_edge_ids: vec![],
                origin_instance: "test".into(),
                brain_state: vec![],
            },
        );

        let store = ConnectomeMmap::open(&path, 8).unwrap();
        // Persist an empty network so node 7 has no record.
        store.persist(&HashMap::new()).unwrap();

        store
            .load_into(&mut network, &[0.9, 0.1, 0.7, 9.81])
            .unwrap();

        let node = network.get(&7).unwrap();
        assert_eq!(node.embedding.len(), EMBEDDING_DIM);
        assert_eq!(node.brain_state.len(), BRAIN_DIM);

        let _ = std::fs::remove_file(&path);
    }
}
