use bad_apple::metal_uma::{
    save_metal_tensors, staging_blit_bytes, tensor_residency, TensorResidency, UmaBuffer,
};
use bad_apple::tensor_brain::CandleBrain;
use std::time::Instant;

/// Integration benchmark for the public UMA buffer.  This is the Phase 1
/// primitive: a `MTLResourceStorageModeShared` buffer that both CPU and GPU
/// can address without copies.  The test runs here (in `tests/legitimate_perf/`)
/// because `metal_uma` is exposed through the library crate.
#[test]
fn uma_connectome_sized_write_latency() {
    const NODES: usize = 500;
    const EMBEDDING_DIM: usize = 2048;
    const ELEM_COUNT: usize = NODES * EMBEDDING_DIM; // 500 * 2048 f64 = ~8 MiB

    let mut buf = UmaBuffer::<f64>::new(ELEM_COUNT)
        .expect("a Metal-capable device is required for this benchmark");

    // Warm the allocation.
    buf.as_mut_slice().fill(0.0);

    let samples: Vec<_> = (0..20)
        .map(|_| {
            let start = Instant::now();
            for (i, x) in buf.as_mut_slice().iter_mut().enumerate() {
                *x = (i % 1024) as f64;
            }
            start.elapsed().as_micros() as u64
        })
        .collect();

    let avg = samples.iter().sum::<u64>() / samples.len() as u64;
    let min = *samples.iter().min().unwrap();
    let max = *samples.iter().max().unwrap();
    eprintln!(
        "[uma_bench] {NODES}x{EMBEDDING_DIM} f64 write: avg={avg} µs min={min} µs max={max} µs (samples: {samples:?})"
    );

    // This is a smoke check, not a strict benchmark. Loaded CI machines can
    // see 6-7 ms for an 8 MiB write; harden by allowing generous headroom.
    assert!(
        avg < 10_000,
        "UMA connectome write averaged {avg} µs, expected < 10000 µs"
    );
}

#[test]
fn resident_weight_save_avoids_staging_blits() {
    let brain = CandleBrain::new("uma-weight-bench", 100, &[]).unwrap();
    let weights = brain.snapshot_weights().unwrap();
    assert!(!weights.is_empty());
    assert!(weights
        .values()
        .all(|tensor| tensor_residency(tensor) == TensorResidency::Shared));

    let path = std::env::temp_dir().join(format!(
        "bad_apple_resident_weight_bench_{}.safetensors",
        rand::random::<u64>()
    ));
    let before = staging_blit_bytes();
    let samples: Vec<_> = (0..10)
        .map(|_| {
            let start = Instant::now();
            save_metal_tensors(&weights, &path).unwrap();
            start.elapsed().as_micros() as u64
        })
        .collect();
    let avg = samples.iter().sum::<u64>() / samples.len() as u64;

    assert_eq!(staging_blit_bytes(), before);
    assert!(path.metadata().unwrap().len() > 0);
    eprintln!(
        "[resident_weight_save] tensors={} avg={} µs staging_blit_bytes=0 samples={:?}",
        weights.len(),
        avg,
        samples
    );
    std::fs::remove_file(path).unwrap();
}
