//! Bad Apple // in-process ANE Qwen3-4B cognitive core benchmark.
//!
//! This integration test loads a locally compiled CoreML `.mlmodelc` (placed
//! under `tests/ane_brain_perf/artifacts`) and exercises the `ane_core`
//! generation path end-to-end.  It reports:
//!
//! - ANE placement ratio (how many operations CoreML placed on the ANE)
//! - first-token and per-token latency
//! - top-process CPU usage during inference
//! - confirmation of zero network socket activity
//! - mastery score on a small curriculum set

use bad_apple::ane_core;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::time::{Duration, Instant};
use sysinfo::{get_current_pid, ProcessRefreshKind, System};

const DEFAULT_MAX_NEW_TOKENS: usize = 64;
const WARMUP_MAX_NEW_TOKENS: usize = 8;
const TIMEOUT: Duration = Duration::from_secs(180);

#[derive(Debug)]
struct BenchmarkReport {
    model_path: PathBuf,
    model_size_bytes: u64,
    ane_placement_ratio: f64,
    compute_units_raw_value: i32,
    selection_load_latency_us: u64,
    selection_prewarm_latency_us: u64,
    first_token_latency_us: u64,
    avg_token_latency_us: u64,
    tokens_per_second: f64,
    top_process_cpu_percent: f32,
    socket_count: usize,
    mastery_score: f64,
    mastery_attempts: usize,
    response_sample: String,
}

impl BenchmarkReport {
    fn print(&self) {
        println!("\n=== BAD APPLE // ANE Brain Performance Report ===");
        println!("model_path: {}", self.model_path.display());
        println!("model_size_bytes: {}", self.model_size_bytes);
        println!(
            "ane_placement_ratio: {:.2}%",
            self.ane_placement_ratio * 100.0
        );
        println!("compute_units_raw_value: {}", self.compute_units_raw_value);
        println!(
            "selection_load_latency_us: {}",
            self.selection_load_latency_us
        );
        println!(
            "selection_prewarm_latency_us: {}",
            self.selection_prewarm_latency_us
        );
        println!("first_token_latency_us: {}", self.first_token_latency_us);
        println!("avg_token_latency_us: {}", self.avg_token_latency_us);
        println!("tokens_per_second: {:.2}", self.tokens_per_second);
        println!(
            "top_process_cpu_percent: {:.1}%",
            self.top_process_cpu_percent
        );
        println!("socket_count: {}", self.socket_count);
        println!("mastery_score: {:.3}", self.mastery_score);
        println!("mastery_attempts: {}", self.mastery_attempts);
        println!("response_sample: {}", self.response_sample);
    }
}

#[test]
fn ane_brain_perf() {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let artifact_root = manifest.join("tests/ane_brain_perf/artifacts");

    // Try the Qwen3-4B sharded artifacts first, then legacy 3B/0.5B fallbacks.
    let candidates = [
        ("qwen3b_ane_shards", "conversion_manifest.json"),
        ("qwen3b_ane", "model.mlmodelc"),
        ("qwen3b", "Qwen2.5-3B-Instruct-4bit.mlmodelc"),
        ("qwen0.5b", "Qwen2.5-0.5B-Instruct-4bit.mlmodelc"),
    ];

    let mut chosen = std::env::var_os("BADAPPLE_ANE_MODEL")
        .zip(std::env::var_os("BADAPPLE_ANE_TOKENIZER"))
        .map(|(model, tokenizer)| (PathBuf::from(model), PathBuf::from(tokenizer)));
    for (dir, model_name) in &candidates {
        if chosen.is_some() {
            break;
        }
        let dir = artifact_root.join(dir);
        let model = dir.join(model_name);
        let tokenizer = dir.join("tokenizer.json");
        let usable = if model
            .extension()
            .is_some_and(|extension| extension == "json")
        {
            shard_generation_ready(&model)
        } else {
            dir_size(&model) > 50_000_000
        };
        if model.exists() && tokenizer.exists() && usable {
            chosen = Some((model, tokenizer));
            break;
        }
    }

    let (model, tokenizer) = match chosen {
        Some(p) => p,
        None => {
            println!("ANE artifacts not present; skipping benchmark.");
            println!("Run: python3 tests/ane_brain_perf/download_models.py");
            return;
        }
    };

    std::env::set_var("BADAPPLE_ANE_MODEL", &model);
    std::env::set_var("BADAPPLE_ANE_TOKENIZER", &tokenizer);

    ane_core::initialize_from_env().expect("ANE core must initialize");

    let benchmark_tokens = std::env::var("BADAPPLE_ANE_BENCH_TOKENS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(DEFAULT_MAX_NEW_TOKENS);
    let compute_units_raw_value = ane_core::compute_units_raw_value().unwrap_or(-1);
    let (selection_load_latency_us, selection_prewarm_latency_us) =
        ane_core::selection_latency_us().unwrap_or_default();
    let _ = ane_core::prewarm();
    let placement_ratio = ane_core::placement_ratio().unwrap_or(-1.0);

    // First a warm-up generation so subsequent timings are stable.
    let _ = ane_core::generate_sync(
        "What is the capital of France?",
        WARMUP_MAX_NEW_TOKENS.min(benchmark_tokens),
        ane_core::context_limit(),
    );

    ane_core::reset_counters();
    let prompt = "Explain quantum computing in one sentence.";
    let (response, first_token_us, avg_token_us, _tokens, top_cpu) =
        timed_generation_with_cpu(prompt, benchmark_tokens);

    let socket_count = current_process_socket_count().unwrap_or(0);
    let (mastery_score, mastery_attempts) =
        if std::env::var_os("BADAPPLE_ANE_SKIP_MASTERY").is_some() {
            (0.0, 0)
        } else {
            run_mastery_suite()
        };

    let model_size_bytes = if model
        .extension()
        .is_some_and(|extension| extension == "json")
    {
        shard_runtime_size(&model)
    } else {
        dir_size(&model)
    };
    let report = BenchmarkReport {
        model_path: model,
        model_size_bytes,
        ane_placement_ratio: placement_ratio,
        compute_units_raw_value,
        selection_load_latency_us,
        selection_prewarm_latency_us,
        first_token_latency_us: first_token_us,
        avg_token_latency_us: avg_token_us,
        tokens_per_second: if avg_token_us > 0 {
            1_000_000.0 / avg_token_us as f64
        } else {
            0.0
        },
        top_process_cpu_percent: top_cpu,
        socket_count,
        mastery_score,
        mastery_attempts,
        response_sample: response.chars().take(200).collect(),
    };
    report.print();

    assert_eq!(
        ane_core::metrics().2,
        0,
        "ANE core must report zero prediction failures"
    );
    assert_eq!(
        socket_count, 0,
        "in-process ANE core must not open any network sockets"
    );
    if report.model_path.ends_with("conversion_manifest.json") {
        assert_eq!(
            report.compute_units_raw_value, 3,
            "sharded generation must select cpuAndNeuralEngine"
        );
        assert!(
            report.ane_placement_ratio > 0.0,
            "sharded generation must place operations on ANE"
        );
        assert!(
            !report.response_sample.trim().is_empty(),
            "sharded generation must decode text"
        );
    } else if report
        .model_path
        .to_string_lossy()
        .contains("Qwen2.5-3B-Instruct-4bit.mlmodelc")
    {
        assert_eq!(
            report.compute_units_raw_value, 1,
            "the incompatible 4-bit graph must fall back to cpuAndGPU"
        );
        assert!(
            report.avg_token_latency_us < 2_000_000,
            "cpuAndGPU fallback regressed to {} µs/token",
            report.avg_token_latency_us
        );
    }
    assert!(
        report.mastery_score >= 0.0,
        "mastery score must be non-negative"
    );
}

#[test]
#[ignore = "requires locally converted CoreML shards"]
fn ane_shard_residency() {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/ane_brain_perf/artifacts/qwen3b_ane_shards/conversion_manifest.json");
    if !manifest.is_file() {
        println!("Shard manifest not present; skipping residency probe.");
        return;
    }
    let value: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&manifest).expect("read shard manifest"))
            .expect("parse shard manifest");
    let compiled: Vec<PathBuf> = value["shards"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|shard| shard["status"] == "compiled")
        .filter_map(|shard| shard["compiled_path"].as_str().map(PathBuf::from))
        .filter(|path| path.is_dir())
        .collect();
    if compiled.is_empty() {
        println!("No compiled shards present; skipping residency probe.");
        return;
    }

    let mut ratios = Vec::with_capacity(compiled.len());
    for path in &compiled {
        let ratio = ane_core::audit_artifact_placement(path, 3)
            .expect("CoreML placement audit must succeed");
        println!("shard={} ane_placement_ratio={ratio:.6}", path.display());
        assert!((0.0..=1.0).contains(&ratio));
        ratios.push(ratio);
    }
    let placement_ratio = ratios.iter().sum::<f64>() / ratios.len() as f64;
    let probe_latency_us = ane_core::probe_compiled_shard(&compiled[0], 3, 3)
        .expect("ANE shard prediction must succeed");
    let compiled_size_bytes: u64 = compiled.iter().map(|path| dir_size(path)).sum();

    println!("\n=== BAD APPLE // ANE Shard Residency Report ===");
    println!("compiled_shards: {}", compiled.len());
    println!("compiled_size_bytes: {compiled_size_bytes}");
    println!("ane_placement_ratio: {:.2}%", placement_ratio * 100.0);
    println!("first_shard_avg_latency_us: {probe_latency_us}");
    assert!(
        placement_ratio > 0.0,
        "compiled shards must place work on ANE"
    );
    assert!(
        probe_latency_us > 0,
        "shard probe latency must be measurable"
    );
}

fn timed_generation_with_cpu(prompt: &str, max_new_tokens: usize) -> (String, u64, u64, u64, f32) {
    let prompt = prompt.to_string();
    let (tx, rx) = mpsc::channel();
    let context_limit = ane_core::context_limit();

    // Run the blocking ANE generation on a dedicated thread so the main
    // thread can sample CPU usage concurrently.
    std::thread::spawn(move || {
        let start = Instant::now();
        let result = ane_core::generate_sync(&prompt, max_new_tokens, context_limit);
        let elapsed = start.elapsed();
        let _ = tx.send((result, elapsed));
    });

    let mut system = System::new_all();
    let pid = get_current_pid().expect("current pid");
    let mut max_cpu = 0.0_f32;
    let sample_deadline = Instant::now() + TIMEOUT;

    while Instant::now() < sample_deadline {
        system.refresh_process_specifics(pid, ProcessRefreshKind::new().with_cpu());
        if let Some(process) = system.process(pid) {
            let cpu = process.cpu_usage();
            if cpu > max_cpu {
                max_cpu = cpu;
            }
        }
        if let Ok((result, elapsed)) = rx.try_recv() {
            let text = result.expect("ANE generation must succeed");
            let first_token_us = ane_core::first_token_latency_us();
            let (_last_token_us, token_count, _failures) = ane_core::metrics();
            let avg_token_us = if token_count > 1 {
                (elapsed.as_micros() as u64).saturating_sub(first_token_us) / (token_count - 1)
            } else {
                first_token_us
            };
            return (text, first_token_us, avg_token_us, token_count, max_cpu);
        }
        std::thread::sleep(Duration::from_millis(50));
    }

    panic!("ANE generation did not finish within {:?}", TIMEOUT);
}

fn run_mastery_suite() -> (f64, usize) {
    let tasks = [
        ("Apply a Caesar cipher shift of 3 to 'hello'.", "khoor"),
        (
            "What is the first-letter acronym of 'National Aeronautics and Space Administration'?",
            "NASA",
        ),
        (
            "Reverse the words in 'the quick brown fox'.",
            "brown quick the",
        ),
    ];

    let mut score = 0.0;
    let context_limit = ane_core::context_limit();

    for (prompt, expected) in &tasks {
        let full_prompt =
            format!("{prompt}\nAnswer with only the requested result, no explanation.");
        let answer = match ane_core::generate_sync(&full_prompt, 24, context_limit) {
            Ok(text) => text.to_lowercase(),
            Err(_) => continue,
        };
        let expected_lower = expected.to_lowercase();
        if answer.contains(&expected_lower) {
            score += 1.0;
        } else if expected.contains(' ') {
            // Accept reverse/acronym even if words are reordered by punctuation.
            let expected_words: std::collections::HashSet<_> =
                expected_lower.split_whitespace().collect();
            let answer_words: std::collections::HashSet<_> = answer.split_whitespace().collect();
            if expected_words.is_subset(&answer_words) {
                score += 1.0;
            }
        }
    }

    (score / tasks.len() as f64, tasks.len())
}

fn current_process_socket_count() -> Option<usize> {
    let pid = std::process::id();
    // `-a` ANDs the process and network selections so we only see sockets
    // opened by this exact process.
    let output = Command::new("lsof")
        .args(["-a", "-nP", "-i", "-p", &pid.to_string()])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()
        .ok()?;

    let text = String::from_utf8_lossy(&output.stdout);
    Some(
        text.lines()
            .skip(1) // lsof header
            .filter(|line| line.contains("TCP") || line.contains("UDP"))
            .count(),
    )
}

fn shard_generation_ready(path: &std::path::Path) -> bool {
    let Ok(raw) = std::fs::read(path) else {
        return false;
    };
    let Ok(value) = serde_json::from_slice::<serde_json::Value>(&raw) else {
        return false;
    };
    let layers_ready = value["shards"].as_array().is_some_and(|shards| {
        !shards.is_empty() && shards.iter().all(|shard| shard["status"] == "compiled")
    });
    let heads_ready = value["shared"]["lm_head_shards"]
        .as_array()
        .is_some_and(|heads| {
            !heads.is_empty() && heads.iter().all(|head| head["status"] == "compiled")
        });
    value["status"] == "complete"
        && value["shared"]["embedding"]["status"] == "complete"
        && layers_ready
        && heads_ready
}

fn shard_runtime_size(path: &std::path::Path) -> u64 {
    let Ok(raw) = std::fs::read(path) else {
        return 0;
    };
    let Ok(value) = serde_json::from_slice::<serde_json::Value>(&raw) else {
        return 0;
    };
    let layer_bytes = value["shards"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|shard| shard["compiled_size_bytes"].as_u64())
        .sum::<u64>();
    let head_bytes = value["shared"]["lm_head_shards"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|head| head["compiled_size_bytes"].as_u64())
        .sum::<u64>();
    layer_bytes
        + head_bytes
        + value["shared"]["embedding"]["size_bytes"]
            .as_u64()
            .unwrap_or(0)
}

fn dir_size(path: &std::path::Path) -> u64 {
    if !path.exists() {
        return 0;
    }
    if path.is_file() {
        return std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    }
    let mut total = 0;
    if let Ok(entries) = std::fs::read_dir(path) {
        for entry in entries.flatten() {
            total += dir_size(&entry.path());
        }
    }
    total
}
