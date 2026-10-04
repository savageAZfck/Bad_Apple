//! Greedy-generate a prompt through the ANE core; used to diff outputs
//! across artifact variants (manual attention vs fused SDPA).
//! Usage: BADAPPLE_ANE_MODEL=<manifest.json> BADAPPLE_ANE_TOKENIZER=<tok.json> ane_gen [tokens] [prompt]
//! BADAPPLE_ANE_RAW_PROMPT=1 skips the chat template; prompt may also come from
//! a file via ANE_GEN_PROMPT_FILE.

use bad_apple::ane_core;

fn main() {
    let mut args = std::env::args().skip(1);
    let tokens: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(24);
    let prompt = std::env::var("ANE_GEN_PROMPT_FILE")
        .ok()
        .and_then(|p| std::fs::read_to_string(p).ok())
        .or_else(|| args.next())
        .unwrap_or_else(|| "Explain quantum computing in one sentence.".to_string());
    ane_core::initialize_from_env().expect("init");
    let text =
        ane_core::generate_sync(&prompt, tokens, ane_core::context_limit()).expect("generate");
    println!("=== RESPONSE ===");
    println!("{}", text);
}
