//! Probe a compiled ANE shard: loads the .mlmodelc, runs predictions on the
//! Neural Engine, prints placement ratio + average latency.
//! Usage: ane_probe <path-to-mlmodelc> [iterations]

use bad_apple::ane_core;
use std::path::PathBuf;

fn main() {
    let path = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .expect("usage: ane_probe <mlmodelc> [iters]");
    let iters: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);
    match ane_core::probe_compiled_shard(&path, 3, iters) {
        Ok(us) => println!("ANE prediction OK — avg latency {us} us ({path:?})"),
        Err(e) => {
            eprintln!("ANE probe failed: {e}");
            std::process::exit(1);
        }
    }
}
