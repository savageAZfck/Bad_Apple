use firefly_edgeos::metal_uma::UmaBuffer;
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
        "[uma_bench] {}x{} f64 write: avg={} µs min={} µs max={} µs (samples: {:?})",
        NODES, EMBEDDING_DIM, avg, min, max, samples
    );

    assert!(
        avg < 2_000,
        "UMA connectome write averaged {} µs, expected < 2000 µs",
        avg
    );
}
