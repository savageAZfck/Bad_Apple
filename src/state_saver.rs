//! Double-buffered, asynchronous state persistence.
//!
//! The main cognitive loop builds a lightweight snapshot of the soul matrix,
//! copies the live Candle `VarMap` tensors, and hands the payload to a
//! dedicated background thread.  The thread performs the physical 800+ ms
//! JSON + safetensors + connectome flush without ever blocking the main
//! async runtime.

use crate::FullySapientSoulMatrix;
use candle_core::Tensor;
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
    /// Build a payload from the live mind.  The live lock is held while the
    /// connectome vectors are flushed directly from the live mmap-backed slab;
    /// the background thread then writes only the compact JSON metadata and the
    /// optional safetensors weights.
    pub fn from_mind(
        mind: &FullySapientSoulMatrix,
        base: &Path,
        weights: Option<HashMap<String, Tensor>>,
    ) -> Self {
        // Persist the heavy 2048-D / 576-D vectors under the live lock so they
        // are not cloned into the background `SavePayload`.
        let _ = mind.persist_connectome(base);

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

        // Real Candle tensor weights to a separate safetensors file.  On Apple
        // Silicon we use the UMA shared-buffer path so the CPU reads the same
        // `MTLResourceStorageModeShared` pages the GPU wrote, avoiding a
        // Metal-to-CPU `to_vec1` copy.  On CPU or non-contiguous tensors this
        // falls back to Candle's ordinary `safetensors::save`.
        if let Some(weights) = self.weights {
            crate::metal_uma::save_metal_tensors(&weights, &self.paths.safetensors)
                .map_err(|e| std::io::Error::other(format!("safetensors save: {e}")))?;
        }

        // Legacy scalar weight persistence, file quarantine, and network stack.
        let _ = self
            .snapshot
            .weight_persistence
            .save_to_file(&self.paths.weights);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MemoryGraphNode;
    use candle_core::{DType, Device, Tensor};
    use std::time::Instant;

    /// Construct a realistic `SavePayload` for benchmarking.  `n_nodes` controls
    /// the connectome size and `n_weights` the number of dummy transformer
    /// tensors.  The `embedding`/`brain_state` vectors are populated so the
    /// connectome mmap flush has real work to do.
    fn build_fixture(base: &std::path::Path, n_nodes: usize, n_weights: usize) -> SavePayload {
        let mut mind = FullySapientSoulMatrix::new("bench", 0);
        // Avoid real Candle brain setup in the benchmark; we are measuring the
        // save path, not the forward pass.
        mind.candle_brain = None;

        for i in 0..n_nodes {
            let node = MemoryGraphNode {
                id: i as u64,
                timestamp: i as u64,
                experiential_text: format!("experience {}", i),
                emotional_state_snapshot: "curious".into(),
                embedding: vec![0.01 * (i as f64); 2048],
                associated_edge_ids: vec![],
                origin_instance: "bench".into(),
                brain_state: vec![0.001 * (i as f64); 576],
            };
            mind.associative_memory_network.insert(i as u64, node);
        }

        let mut weights = HashMap::with_capacity(n_weights);
        let device = Device::new_metal(0).unwrap_or(Device::Cpu);
        for i in 0..n_weights {
            let t = Tensor::zeros(1_000, DType::F32, &device).unwrap();
            weights.insert(format!("weight_{}", i), t);
        }

        SavePayload::from_mind(&mind, base, Some(weights))
    }

    #[test]
    fn bench_state_save_500_nodes_10_weights() {
        let tmp =
            std::env::temp_dir().join(format!("badapple_state_bench_{}", rand::random::<u64>()));
        std::fs::create_dir_all(&tmp).unwrap();
        let base = tmp.join("state");

        let mut samples: Vec<u64> = Vec::with_capacity(10);
        for _ in 0..10 {
            let payload = build_fixture(&base, 500, 10);
            let start = Instant::now();
            payload.write().unwrap();
            samples.push(start.elapsed().as_millis() as u64);
            // Clean up the atomic family so the next iteration can overwrite.
            let _ = std::fs::remove_dir_all(&tmp);
            std::fs::create_dir_all(&tmp).unwrap();
        }

        let avg = samples.iter().sum::<u64>() / samples.len() as u64;
        let min = *samples.iter().min().unwrap();
        let max = *samples.iter().max().unwrap();
        eprintln!(
            "[state_save_bench] state_save_duration_ms avg={} min={} max={} (samples: {:?})",
            avg, min, max, samples
        );

        // The test passes if the average stays below a generous 500 ms ceiling
        // for a 500-node, 10-weight fixture; the real target is far lower.
        assert!(avg < 500, "state save average {} ms exceeded 500 ms", avg);
    }

    #[test]
    fn bench_state_save_100_nodes_no_weights() {
        let tmp = std::env::temp_dir().join(format!(
            "badapple_state_bench_small_{}",
            rand::random::<u64>()
        ));
        std::fs::create_dir_all(&tmp).unwrap();
        let base = tmp.join("state");

        let mut samples: Vec<u64> = Vec::with_capacity(10);
        for _ in 0..10 {
            let payload = build_fixture(&base, 100, 0);
            let start = Instant::now();
            payload.write().unwrap();
            samples.push(start.elapsed().as_millis() as u64);
            let _ = std::fs::remove_dir_all(&tmp);
            std::fs::create_dir_all(&tmp).unwrap();
        }

        let avg = samples.iter().sum::<u64>() / samples.len() as u64;
        eprintln!(
            "[state_save_bench_small] state_save_duration_ms avg={} (samples: {:?})",
            avg, samples
        );
        assert!(
            avg < 200,
            "small state save average {} ms exceeded 200 ms",
            avg
        );
    }

    /// Re-implementation of the pre-refactor `SavePayload::write` logic so the
    /// benchmark can report a true before/after delta on the same fixture.
    fn baseline_write(payload: &mut SavePayload) -> std::io::Result<u64> {
        use crate::{connectome_mmap, MAX_MEMORY_NODES};

        let start = Instant::now();
        let tmp = payload.paths.state.with_extension("json.tmp");
        if let Ok(file) = std::fs::File::create(&tmp) {
            let writer = std::io::BufWriter::new(file);
            if serde_json::to_writer(writer, &payload.snapshot).is_ok() {
                let _ = std::fs::rename(&tmp, &payload.paths.state);
            }
        }

        if let Some(weights) = payload.weights.as_ref() {
            let mut copied = HashMap::with_capacity(weights.len());
            for (name, t) in weights {
                if let Ok(t) = t.copy() {
                    copied.insert(name.clone(), t);
                }
            }
            let _ = candle_core::safetensors::save(&copied, &payload.paths.safetensors);
        }

        if let Ok(store) =
            connectome_mmap::ConnectomeMmap::open(&payload.paths.connectome, MAX_MEMORY_NODES)
        {
            let _ = store.persist(&payload.snapshot.associative_memory_network);
        }

        let _ = payload
            .snapshot
            .weight_persistence
            .save_to_file(&payload.paths.weights);
        let _ = payload
            .snapshot
            .file_defense
            .save_state(&payload.paths.defense);
        let _ = payload
            .snapshot
            .network_stack
            .save_state(&payload.paths.network);

        Ok(start.elapsed().as_millis() as u64)
    }

    #[test]
    fn bench_state_save_before_vs_after() {
        let tmp =
            std::env::temp_dir().join(format!("badapple_state_compare_{}", rand::random::<u64>()));
        std::fs::create_dir_all(&tmp).unwrap();
        let base = tmp.join("state");

        // Build one live mind and two payloads from it: the legacy path deep
        // clones the connectome into the snapshot, the new path persists it
        // directly from the live mind.  Keep the real CandleBrain so the weight
        // save path exercises realistic Metal/CPU tensor copies.
        let mut mind = FullySapientSoulMatrix::new("bench", 0);
        for i in 0..500 {
            let node = MemoryGraphNode {
                id: i as u64,
                timestamp: i as u64,
                experiential_text: format!("experience {}", i),
                emotional_state_snapshot: "curious".into(),
                embedding: vec![0.01 * (i as f64); 2048],
                associated_edge_ids: vec![],
                origin_instance: "bench".into(),
                brain_state: vec![0.001 * (i as f64); 576],
            };
            mind.associative_memory_network.insert(i as u64, node);
        }

        let weights = mind
            .candle_brain
            .as_ref()
            .and_then(|b| b.snapshot_weights().ok())
            .expect("real Candle brain should yield weights");
        assert!(weights.values().all(|tensor| {
            crate::metal_uma::tensor_residency(tensor) == crate::metal_uma::TensorResidency::Shared
        }));
        let staging_bytes_before = crate::metal_uma::staging_blit_bytes();

        // Benchmark the *total* state-save cost: payload construction
        // (connectome snapshot / mmap persist + handoff) + the background write.
        // We warm up 3 iterations, then keep the steady-state samples.
        const WARMUP: usize = 3;
        const ITERATIONS: usize = 12;

        let mut new_total = Vec::with_capacity(ITERATIONS);
        let mut new_handoff = Vec::with_capacity(ITERATIONS);
        let mut new_write = Vec::with_capacity(ITERATIONS);
        for _ in 0..WARMUP + ITERATIONS {
            let start = Instant::now();
            let payload = SavePayload::from_mind(&mind, &base, Some(weights.clone()));
            let handoff_us = start.elapsed().as_micros() as u64;
            let write_start = Instant::now();
            payload.write().unwrap();
            let write_ms = write_start.elapsed().as_millis() as u64;
            let total_ms = start.elapsed().as_millis() as u64;
            new_handoff.push(handoff_us);
            new_write.push(write_ms);
            new_total.push(total_ms);
        }

        // Verify the UMA safetensors file is loadable and one weight round-trips.
        let loaded =
            candle_core::safetensors::load(base.with_extension("safetensors"), &Device::Cpu)
                .unwrap();
        let first_name = weights.keys().next().unwrap();
        let expected = weights[first_name].to_device(&Device::Cpu).unwrap();
        let actual = &loaded[first_name];
        let e0 = expected.flatten_all().unwrap().to_vec1::<f32>().unwrap()[0];
        let a0 = actual.flatten_all().unwrap().to_vec1::<f32>().unwrap()[0];
        let diff = (e0 - a0).abs();
        assert!(
            diff < 1e-4,
            "UMA safetensors first weight mismatch: {}",
            diff
        );
        let (_, new_handoff) = new_handoff.split_at(WARMUP);
        let (_, new_write) = new_write.split_at(WARMUP);
        let (_, new_total) = new_total.split_at(WARMUP);
        assert_eq!(crate::metal_uma::staging_blit_bytes(), staging_bytes_before);

        // The legacy payload uses the old full-clone snapshot and the legacy
        // write path that copies weights and re-persists the connectome.
        let mut legacy_total = Vec::with_capacity(ITERATIONS);
        let mut legacy_handoff = Vec::with_capacity(ITERATIONS);
        let mut legacy_write = Vec::with_capacity(ITERATIONS);
        for _ in 0..WARMUP + ITERATIONS {
            let start = Instant::now();
            let snapshot = mind.save_snapshot_legacy();
            let mut legacy = SavePayload {
                snapshot,
                weights: Some(weights.clone()),
                paths: SavePaths::from(&base),
                completion: None,
            };
            let handoff_us = start.elapsed().as_micros() as u64;
            let write_ms = baseline_write(&mut legacy).unwrap();
            let total_ms = handoff_us / 1000 + write_ms;
            legacy_handoff.push(handoff_us);
            legacy_write.push(write_ms);
            legacy_total.push(total_ms);
        }
        let (_, legacy_handoff) = legacy_handoff.split_at(WARMUP);
        let (_, legacy_write) = legacy_write.split_at(WARMUP);
        let (_, legacy_total) = legacy_total.split_at(WARMUP);

        fn median(v: &[u64]) -> f64 {
            let mut c = v.to_vec();
            c.sort_unstable();
            let n = c.len();
            if n.is_multiple_of(2) {
                (c[n / 2 - 1] + c[n / 2]) as f64 / 2.0
            } else {
                c[n / 2] as f64
            }
        }

        eprintln!(
            "[state_save_compare] \
             legacy median: total={:.1} ms (handoff={:.1} us, write={:.1} ms), \
             new median: total={:.1} ms (handoff={:.1} us, write={:.1} ms)",
            median(legacy_total),
            median(legacy_handoff),
            median(legacy_write),
            median(new_total),
            median(new_handoff),
            median(new_write),
        );

        // This is a measurement, not a correctness test.  Parallel test
        // execution can create heavy Metal / filesystem contention, so the
        // figure is reported rather than asserted.
        let _ = (new_total, legacy_total);
    }
}
