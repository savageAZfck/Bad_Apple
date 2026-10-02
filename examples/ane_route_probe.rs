// Probe: drive the daemon's raw `inference` route with brain=ane|gpu|auto.
// Prints the response text plus any timing/usage fields for throughput math.
use serde_json::json;
use std::time::Instant;

fn main() {
    let brain = std::env::args().nth(1).unwrap_or_else(|| "ane".into());
    let tokens: usize = std::env::args()
        .nth(2)
        .and_then(|v| v.parse().ok())
        .unwrap_or(60);
    let started = Instant::now();
    let result = bad_apple::bad_apple_ipc::call_agent(
        "inference",
        Some(json!({
            "prompt": "Explain how a hash-chained audit ledger detects tampering.",
            "system_prompt": "You are a precise assistant.",
            "max_new_tokens": tokens,
            "temperature": 0.0,
            "brain": brain,
        })),
        tokens,
    );
    let elapsed = started.elapsed().as_secs_f64();
    match result {
        Ok(v) => {
            let text = v.get("text").and_then(|t| t.as_str()).unwrap_or("");
            let mut stats = serde_json::Map::new();
            if let Some(obj) = v.as_object() {
                for (k, val) in obj {
                    if val.is_number() || k == "model" || k == "brain" || k == "tier" {
                        stats.insert(k.clone(), val.clone());
                    }
                }
            }
            println!(
                "brain={brain} wall={elapsed:.1}s stats={} words={}",
                serde_json::Value::Object(stats),
                text.split_whitespace().count()
            );
        }
        Err(e) => println!("brain={brain} error: {e}"),
    }
}
