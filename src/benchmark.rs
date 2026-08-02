//! Simple external task benchmark for the Firefly agent.
//!
//! The suite generates math / logic / code puzzles, lets the agent produce a
//! Python solution, and records the success rate over time.

use crate::strategy_library::{DialecticalEngine, Strategy, StrategyLibrary};
use tokio::task::spawn_blocking;

#[derive(Clone)]
pub struct BenchmarkTask {
    pub name: &'static str,
    pub prompt: &'static str,
    pub expected: &'static str,
    pub tolerance: Option<f64>,
}

pub struct BenchmarkSuite {
    pub tasks: Vec<BenchmarkTask>,
    pub history: Vec<(String, bool)>,
    pub index: usize,
}

impl BenchmarkSuite {
    pub fn new() -> Self {
        Self {
            tasks: vec![
                BenchmarkTask {
                    name: "simple_arithmetic",
                    prompt: "Write a short Python 3 script that computes (17 * 23) + (12 * 31) and prints only the final integer answer.",
                    expected: "763",
                    tolerance: Some(0.001),
                },
                BenchmarkTask {
                    name: "factorial",
                    prompt: "Write a short Python 3 script that computes 6! (6 factorial) and prints only the final integer answer.",
                    expected: "720",
                    tolerance: Some(0.001),
                },
                BenchmarkTask {
                    name: "fibonacci_10",
                    prompt: "Write a short Python 3 script that computes the 10th Fibonacci number (F(0)=0, F(1)=1, F(2)=1, ... F(10)=?) and prints only the final integer answer.",
                    expected: "55",
                    tolerance: Some(0.001),
                },
                BenchmarkTask {
                    name: "sum_of_squares",
                    prompt: "Write a short Python 3 script that computes 1*1 + 2*2 + 3*3 + 4*4 + 5*5 and prints only the final integer answer.",
                    expected: "55",
                    tolerance: Some(0.001),
                },
                BenchmarkTask {
                    name: "prime_check",
                    prompt: "Write a short Python 3 script that checks if 29 is prime and prints only the boolean True or False.",
                    expected: "True",
                    tolerance: None,
                },
            ],
            history: Vec::new(),
            index: 0,
        }
    }

    pub fn next_task(&mut self) -> &BenchmarkTask {
        let task = &self.tasks[self.index % self.tasks.len()];
        self.index += 1;
        task
    }

    pub fn record(&mut self, name: &str, success: bool) {
        self.history.push((name.to_string(), success));
        if self.history.len() > 1000 {
            self.history.remove(0);
        }
    }

    pub fn score(&self) -> f64 {
        if self.history.is_empty() {
            0.0
        } else {
            let successes = self.history.iter().filter(|(_, s)| *s).count();
            (successes as f64 / self.history.len() as f64) * 100.0
        }
    }

    pub fn history_summary(&self, n: usize) -> Vec<(String, bool)> {
        self.history.iter().rev().take(n).cloned().collect()
    }

    /// Check whether the produced output matches the expected answer.
    pub fn validate(&self, expected: &str, output: &str, tolerance: Option<f64>) -> bool {
        let tokens: Vec<&str> = output.split_whitespace().collect();
        let expected = expected.trim();

        if let (Ok(b), Some(tol)) = (expected.parse::<f64>(), tolerance) {
            for token in &tokens {
                if let Ok(a) = token.parse::<f64>() {
                    if (a - b).abs() <= tol {
                        return true;
                    }
                }
            }
        } else {
            for token in &tokens {
                if token == &expected {
                    return true;
                }
            }
        }
        false
    }

    pub fn running_time_ms(&self) -> f64 {
        // Placeholder for future wall-clock tracking; returns a fixed default.
        1000.0
    }
}

/// A transfer-learning task: learn a skill from one training example and
/// generalize to a held-out test example in a new domain.
#[derive(Clone)]
pub struct TransferTask {
    pub domain: &'static str,
    pub description: &'static str,
    pub train_input: &'static str,
    pub train_output: &'static str,
    pub test_input: &'static str,
    pub test_output: &'static str,
}

/// Autonomous domain-transfer benchmark suite.
pub struct TransferSuite {
    pub tasks: Vec<TransferTask>,
    pub history: Vec<(String, bool)>,
    pub index: usize,
}

impl TransferSuite {
    pub fn new() -> Self {
        Self {
            tasks: vec![
                TransferTask { domain: "vowel counting", description: "Count the number of vowels in a lowercase English string.", train_input: "hello", train_output: "2", test_input: "firefly agi", test_output: "4" },
                TransferTask { domain: "caesar cipher shift 3", description: "Shift each letter forward by 3 in the alphabet (wrap around from z to a). Keep non-letters unchanged.", train_input: "abc", train_output: "def", test_input: "xyz", test_output: "abc" },
                TransferTask { domain: "sum of digits", description: "Sum all decimal digits in the input string, ignoring non-digit characters.", train_input: "a1b2c3", train_output: "6", test_input: "fire5fly8agi2", test_output: "15" },
                TransferTask { domain: "count words", description: "Count the number of whitespace-separated words in the input string.", train_input: "hello world", train_output: "2", test_input: "firefly agi research runtime", test_output: "4" },
                TransferTask { domain: "title case", description: "Convert the input string to title case: first letter of each word uppercase, the rest lowercase.", train_input: "hello world", train_output: "Hello World", test_input: "firefly agi research", test_output: "Firefly Agi Research" },
                TransferTask { domain: "extract digits", description: "Extract all decimal digits from the input and return them as a single string in order.", train_input: "a1b2c3", train_output: "123", test_input: "fire5fly8agi2", test_output: "582" },
                TransferTask { domain: "reverse words", description: "Reverse the order of whitespace-separated words in the input string.", train_input: "hello world", train_output: "world hello", test_input: "firefly agi research", test_output: "research agi firefly" },
                TransferTask { domain: "isogram check", description: "Return True if all letters in the input appear at most once, otherwise False. Ignore non-letters and case.", train_input: "background", train_output: "False", test_input: "isogram", test_output: "True" },
                TransferTask { domain: "morse alphabet", description: "Convert each letter in the input to a dot for consonants and a dash for vowels. Keep spaces.", train_input: "abc de", train_output: "-.- -.", test_input: "firefly agi", test_output: "-.-.-.. ..-" },
                TransferTask { domain: "first letter acronym", description: "Return the first letter of each word, lowercased, concatenated.", train_input: "hello world", train_output: "hw", test_input: "firefly agi research runtime", test_output: "farr" },
                TransferTask { domain: "binary dotdash", description: "Replace every character that is not a space with a dot if it is a vowel and a dash otherwise.", train_input: "ab", train_output: "-.", test_input: "fire", test_output: "-.-." },
                TransferTask { domain: "palindrome words", description: "Return a space-separated list of the words that are palindromes. A single letter is not a palindrome.", train_input: "madam bob apple", train_output: "madam bob", test_input: "civic radar car", test_output: "civic radar" },
            ],
            history: Vec::new(),
            index: 0,
        }
    }

    pub fn next_task(&mut self) -> &TransferTask {
        let task = &self.tasks[self.index % self.tasks.len()];
        self.index += 1;
        task
    }

    pub fn record(&mut self, name: &str, success: bool) {
        self.history.push((name.to_string(), success));
        if self.history.len() > 1000 {
            self.history.remove(0);
        }
    }

    pub fn score(&self) -> f64 {
        if self.history.is_empty() {
            0.0
        } else {
            let successes = self.history.iter().filter(|(_, s)| *s).count();
            (successes as f64 / self.history.len() as f64) * 100.0
        }
    }

    pub fn history_summary(&self, n: usize) -> Vec<(String, bool)> {
        self.history.iter().rev().take(n).cloned().collect()
    }

    /// Current domain-transfer mastery in [0, 1].
    pub fn mastery(&self) -> f64 {
        if self.history.is_empty() {
            return 0.0;
        }
        let successes = self.history.iter().filter(|(_, s)| *s).count();
        successes as f64 / self.history.len() as f64
    }

    /// If mastery for a task falls below `threshold`, run a dialectical
    /// synthesis to produce a repaired internal strategy.
    pub async fn dialectical_repair_if_needed(
        &self,
        library: &StrategyLibrary,
        task: &TransferTask,
        succeeded: bool,
        error: &str,
    ) -> Option<String> {
        if succeeded || self.mastery() >= 0.85 {
            return None;
        }
        let key = format!("transfer_{}", task.domain.replace(' ', "_"));
        let engine = DialecticalEngine::new();
        // Build a thesis from the task description and antithesis from the error.
        let dummy = Strategy::new(
            key.clone(),
            task.description.to_string(),
            "python".to_string(),
            String::new(),
        );
        let thesis = engine.thesis(&dummy, task.test_output);
        let antithesis = engine.antithesis(error);
        let _contradiction = engine.contradiction_strength(&thesis, &antithesis);
        let repaired = engine.synthesize(&dummy, error);
        let new_key = format!("{}_synth_{}", key, engine.next_id());
        let mut repaired = repaired;
        repaired.key = new_key.clone();
        repaired.problem = format!("{} [domain: {}]", repaired.problem, task.domain);
        library.put(&repaired).await.ok()?;
        Some(new_key)
    }
}

// =========================================================================
// Closed-Loop Rust Compilation Validation
// =========================================================================

/// Result of validating a synthesized Rust utility.
#[derive(Clone, Debug)]
pub struct SynthesisResult {
    pub key: String,
    pub source: String,
    pub compiled: bool,
    pub diagnostics: String,
    pub competence: f64,
    /// Wall-clock validation time in milliseconds.
    pub wall_time_ms: f64,
    /// Approximate memory footprint: source size in bytes.
    pub footprint_bytes: usize,
}

/// Runs `cargo check` on a synthesized Rust snippet inside an isolated temp
/// Cargo workspace. Up to `max_retries` formatting/typo fixes are attempted
/// by running `cargo fmt` before each check.
pub struct RustValidator;

impl RustValidator {
    /// Check a Rust source string. Returns `SynthesisResult` with competence
    /// 1.0 on success, 0.0 on failure. The heavy compilation is offloaded to
    /// `spawn_blocking` to keep the Tokio runtime responsive.
    pub async fn validate(key: &str, source: &str, max_retries: usize) -> SynthesisResult {
        let key = key.to_string();
        let source = source.to_string();
        let result = spawn_blocking({
            let key = key.clone();
            let source = source.clone();
            move || Self::validate_blocking(&key, &source, max_retries)
        })
        .await
        .unwrap_or_else(|e| SynthesisResult {
            key,
            source,
            compiled: false,
            diagnostics: format!("spawn_blocking failed: {}", e),
            competence: 0.0,
            wall_time_ms: 0.0,
            footprint_bytes: 0,
        });
        result
    }

    fn validate_blocking(key: &str, source: &str, max_retries: usize) -> SynthesisResult {
        let start = std::time::Instant::now();
        let base = std::env::temp_dir().join(format!("firefly_rust_val_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);

        let Ok(()) = std::fs::create_dir_all(base.join("src")) else {
            return SynthesisResult {
                key: key.to_string(),
                source: source.to_string(),
                compiled: false,
                diagnostics: "failed to create temp dir".to_string(),
                competence: 0.0,
                wall_time_ms: 0.0,
                footprint_bytes: source.len(),
            };
        };

        let manifest = r#"[package]
name = "firefly_synthesized"
version = "0.1.0"
edition = "2021"

[dependencies]
tokio = { version = "1", features = ["full"] }
serde_json = "1"
sha2 = "0.11"
"#;

        let mut diagnostics = String::new();
        let mut compiled = false;

        for attempt in 0..=max_retries {
            let current = if attempt == 0 {
                source.to_string()
            } else {
                // Retry attempt: try to run rustfmt on the source first.
                match Self::format_source(source) {
                    Ok(fmt) => fmt,
                    Err(e) => {
                        diagnostics.push_str(&format!("fmt attempt {} failed: {}\n", attempt, e));
                        continue;
                    }
                }
            };

            if let Err(e) = std::fs::write(base.join("Cargo.toml"), manifest) {
                diagnostics.push_str(&format!("write Cargo.toml failed: {}\n", e));
                continue;
            }
            if let Err(e) = std::fs::write(base.join("src/main.rs"), &current) {
                diagnostics.push_str(&format!("write main.rs failed: {}\n", e));
                continue;
            }

            let output = std::process::Command::new("cargo")
                .arg("check")
                .current_dir(&base)
                .arg("--quiet")
                .arg("--offline")
                .output();

            match output {
                Ok(out) => {
                    if out.status.success() {
                        compiled = true;
                        diagnostics = String::from_utf8_lossy(&out.stderr).to_string();
                        break;
                    } else {
                        diagnostics = String::from_utf8_lossy(&out.stderr).to_string();
                    }
                }
                Err(e) => {
                    diagnostics.push_str(&format!("cargo check spawn failed: {}\n", e));
                }
            }
        }

        let _ = std::fs::remove_dir_all(&base);

        let competence = if compiled { 1.0 } else { 0.0 };
        let wall_time_ms = start.elapsed().as_secs_f64() * 1000.0;
        let footprint_bytes = source.len();
        SynthesisResult {
            key: key.to_string(),
            source: source.to_string(),
            compiled,
            diagnostics,
            competence,
            wall_time_ms,
            footprint_bytes,
        }
    }

    /// Best-effort format a Rust source snippet using `rustfmt`. Returns the
    /// original source if rustfmt is unavailable.
    fn format_source(source: &str) -> Result<String, String> {
        let mut child = std::process::Command::new("rustfmt")
            .arg("--emit=stdout")
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .map_err(|e| e.to_string())?;

        if let Some(stdin) = child.stdin.as_mut() {
            use std::io::Write;
            stdin
                .write_all(source.as_bytes())
                .map_err(|e| e.to_string())?;
        }

        let out = child.wait_with_output().map_err(|e| e.to_string())?;
        if out.status.success() {
            Ok(String::from_utf8_lossy(&out.stdout).to_string())
        } else {
            Err(String::from_utf8_lossy(&out.stderr).to_string())
        }
    }
}
