//! Greedy-generate a fixed prompt through the ANE core; used to diff outputs
//! across artifact variants (manual attention vs fused SDPA).
//! Usage: BADAPPLE_ANE_MODEL=<manifest.json> BADAPPLE_ANE_TOKENIZER=<tok.json> ane_gen <tokens>

use bad_apple::ane_core;

fn main() {
    let tokens: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(24);
    ane_core::initialize_from_env().expect("init");
    let text = ane_core::generate_sync(
        "Explain quantum computing in one sentence.",
        tokens,
        ane_core::context_limit(),
    )
    .expect("generate");
    println!("=== RESPONSE ===");
    println!("{}", text);
}
