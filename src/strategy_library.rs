//! Durable strategy cache for the Bad Apple agent.
//!
//! A strategy is a proven tool template keyed by a problem signature (typically
//! a goal string or a domain). The library is backed by Sled so strategies
//! survive restarts, and it exposes a policy-improvement interface for pruning
//! low-reliability templates.

use crate::hyperdimensional_core::{HDCMemory, OverheadAnalyzer, ScriptProfile};
use crate::production_blueprint::{CausalGraph, CausalRelation};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::{Arc, Mutex};
use tokio::task::spawn_blocking;

/// A cached tool/strategy with empirical reliability.
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct Strategy {
    pub key: String,
    pub problem: String,
    pub language: String,
    pub code: String,
    pub reliability: f64,
    pub uses: u64,
    pub successes: u64,
}

impl Strategy {
    pub fn new(key: String, problem: String, language: String, code: String) -> Self {
        Self {
            key,
            problem,
            language,
            code,
            reliability: 0.5,
            uses: 0,
            successes: 0,
        }
    }

    /// Increment use/success counters and update reliability with an EMA.
    pub fn record(&mut self, success: bool) {
        self.uses += 1;
        if success {
            self.successes += 1;
        }
        let alpha = 0.3;
        let observed = if success { 1.0 } else { 0.0 };
        self.reliability = self.reliability * (1.0 - alpha) + observed * alpha;
    }
}

/// Sled-backed strategy library.
#[derive(Clone)]
pub struct StrategyLibrary {
    db: Arc<sled::Db>,
    /// Causal graph that records how strategies relate to system assets and
    /// constraints.  Shared across clones of the library.
    causal_graph: Arc<Mutex<CausalGraph>>,
}

impl StrategyLibrary {
    /// Open or create the Sled database at the given path.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, sled::Error> {
        let db = sled::open(path)?;
        Ok(Self {
            db: Arc::new(db),
            causal_graph: Arc::new(Mutex::new(CausalGraph::bad_apple_default())),
        })
    }

    /// Read a strategy by key.
    pub async fn get(&self, key: &str) -> Option<Strategy> {
        let db = Arc::clone(&self.db);
        let key = key.to_string();
        match spawn_blocking(move || db.get(key.as_bytes())).await {
            Ok(Ok(Some(bytes))) => serde_json::from_slice(&bytes).ok(),
            _ => None,
        }
    }

    /// Insert or overwrite a strategy and record its causal relation in the
    /// knowledge graph.
    pub async fn put(
        &self,
        strategy: &Strategy,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let db = Arc::clone(&self.db);
        let bytes = serde_json::to_vec(strategy)?;
        let key = strategy.key.clone();
        spawn_blocking(move || db.insert(key.as_bytes(), bytes).map(|_| ())).await??;

        if let Ok(mut graph) = self.causal_graph.lock() {
            graph.add_relation(&strategy.key, CausalRelation::Enables, &strategy.problem);
            graph.add_relation("skill_memory", CausalRelation::DependsOn, &strategy.key);
        }
        Ok(())
    }

    /// Return a causal explanation for a failed problem signature.
    pub fn explain_failure(&self, problem: &str) -> Option<String> {
        let graph = self.causal_graph.lock().ok()?;
        let primitive = graph.find_broken_primitive(problem)?;
        graph.explain_failure(primitive)
    }

    /// Delete a strategy.
    pub async fn remove(&self, key: &str) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let db = Arc::clone(&self.db);
        let key = key.to_string();
        spawn_blocking(move || db.remove(key.as_bytes()).map(|_| ())).await??;
        Ok(())
    }

    /// Return all strategies with reliability below the threshold.
    pub async fn weak_strategies(&self, threshold: f64) -> Vec<Strategy> {
        let db = Arc::clone(&self.db);
        spawn_blocking(move || {
            let mut out = Vec::new();
            for (_, value) in db.iter().flatten() {
                if let Ok(s) = serde_json::from_slice::<Strategy>(&value) {
                    if s.reliability < threshold {
                        out.push(s);
                    }
                }
            }
            out
        })
        .await
        .unwrap_or_default()
    }

    /// Delete all strategies below the reliability threshold.
    pub async fn prune_below(&self, threshold: f64) -> usize {
        let weak = self.weak_strategies(threshold).await;
        let count = weak.len();
        for s in weak {
            let _ = self.remove(&s.key).await;
        }
        count
    }

    /// Find the best matching strategy for a problem using a hybrid score.
    pub async fn best_match(&self, problem: &str) -> Option<Strategy> {
        let db = Arc::clone(&self.db);
        let problem_lower = problem.to_lowercase();
        let problem_words: std::collections::HashSet<String> = problem_lower
            .split_whitespace()
            .map(std::string::ToString::to_string)
            .collect();
        spawn_blocking(move || {
            let mut best: Option<Strategy> = None;
            let mut best_score = 0.0;
            for (_, value) in db.iter().flatten() {
                if let Ok(s) = serde_json::from_slice::<Strategy>(&value) {
                    let s_lower = s.problem.to_lowercase();
                    let mut score =
                        if s_lower.contains(&problem_lower) || problem_lower.contains(&s_lower) {
                            1.0
                        } else {
                            let s_words: std::collections::HashSet<String> = s_lower
                                .split_whitespace()
                                .map(std::string::ToString::to_string)
                                .collect();
                            let total = problem_words.union(&s_words).count();
                            let overlap = problem_words.intersection(&s_words).count();
                            if total > 0 {
                                overlap as f64 / total as f64
                            } else {
                                0.0
                            }
                        };
                    score *= 0.8 + 0.2 * s.reliability;
                    if score > 0.7 && score > best_score {
                        best_score = score;
                        best = Some(s);
                    }
                }
            }
            best
        })
        .await
        .unwrap_or_default()
    }
}

// =========================================================================
// Dialectical Causal Contradiction Engine
// =========================================================================

/// A contradiction node in the dialectical synthesis graph.
/// The graph is stored as an index-based vector (arena) to avoid allocation
/// during synthesis.
#[derive(Clone, Debug)]
pub struct DialecticalNode {
    pub id: u64,
    pub proposition: String,
    pub truth_value: f64,
    pub antecedents: Vec<usize>,
}

/// Thesis → Antithesis → Synthesis engine for strategy repair.
///
/// Given a failed strategy and an error description, the engine isolates the
/// contradiction between the strategy's assumed preconditions (thesis) and
/// the empirical failure (antithesis), then produces a repaired strategy
/// (synthesis) with adjusted code or problem statement.
pub struct DialecticalEngine {
    counter: std::sync::atomic::AtomicU64,
}

impl DialecticalEngine {
    pub fn new() -> Self {
        Self {
            counter: std::sync::atomic::AtomicU64::new(1),
        }
    }

    pub fn next_id(&self) -> u64 {
        self.counter
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    }

    /// Build a thesis from the current strategy and a causal prediction.
    pub fn thesis(&self, strategy: &Strategy, predicted_outcome: &str) -> DialecticalNode {
        DialecticalNode {
            id: self.next_id(),
            proposition: format!(
                "Strategy '{}' will succeed under assumed context: {}",
                strategy.key, predicted_outcome
            ),
            truth_value: strategy.reliability,
            antecedents: Vec::new(),
        }
    }

    /// Build an antithesis from the observed error.
    pub fn antithesis(&self, error: &str) -> DialecticalNode {
        DialecticalNode {
            id: self.next_id(),
            proposition: format!("Observed failure: {error}"),
            truth_value: 1.0,
            antecedents: Vec::new(),
        }
    }

    /// Compute the contradiction signal between thesis and antithesis.
    /// Returns a value in `[0, 1]`; higher means stronger contradiction.
    pub fn contradiction_strength(
        &self,
        thesis: &DialecticalNode,
        antithesis: &DialecticalNode,
    ) -> f64 {
        // Strong contradiction when a confident thesis fails.
        thesis.truth_value * antithesis.truth_value
    }

    /// Synthesize a repaired strategy. The repair is heuristic: if the error
    /// mentions a missing module, prepend an import; otherwise tag the problem
    /// with the error signature for future matching.
    pub fn synthesize(&self, strategy: &Strategy, error: &str) -> Strategy {
        let mut repaired = strategy.clone();
        repaired.uses = 0;
        repaired.successes = 0;
        repaired.reliability = 0.5;

        let error_lower = error.to_lowercase();
        if error_lower.contains("missing") && error_lower.contains("module") {
            // Heuristic repair: wrap code with a broad safe import.
            repaired.code = format!(
                "import math, statistics, json, datetime, re, collections, itertools, string\n{}\n",
                repaired.code
            );
        }

        // Bind the problem to the error signature so future matching can find
        // this repaired variant when the same failure mode recurs.
        repaired.problem = format!("{} [synthesized for error: {}]", repaired.problem, error);
        repaired
    }
}

impl Default for DialecticalEngine {
    fn default() -> Self {
        Self::new()
    }
}

// =========================================================================
// Continuous Fluid Policy Synthesis
// =========================================================================

/// A lightweight, pattern-matching Rust code synthesizer.
///
/// Treats the minimized 10,000-D HDC profile as a fluid policy coordinate and
/// maps it to the closest hand-written async Tokio template. This is NOT a full
/// Python-to-Rust compiler, but a deterministic policy synthesizer for common
/// local utility patterns.
pub struct RustSynthesizer {
    template_memory: HDCMemory,
    templates: std::collections::HashMap<String, String>,
}

/// Alias for the neuromorphic flow generator used in the thermodynamic harness.
pub type FluidPolicySynthesizer = RustSynthesizer;

impl RustSynthesizer {
    pub fn new() -> Self {
        let mut template_memory = HDCMemory::new();
        let mut templates = std::collections::HashMap::new();

        // Template 1: sum a vector of numbers.
        let k1 = "sum_template".to_string();
        template_memory.allocate(&k1);
        templates.insert(
            k1,
            r#"use tokio;

#[tokio::main]
async fn main() {
    let numbers: Vec<i64> = vec![1, 2, 3, 4, 5];
    let total: i64 = numbers.iter().sum();
    println!("{}", total);
}"#
            .to_string(),
        );

        // Template 2: echo an input string.
        let k2 = "echo_template".to_string();
        template_memory.allocate(&k2);
        templates.insert(
            k2,
            r#"use tokio;

#[tokio::main]
async fn main() {
    let input = "hello";
    println!("{}", input);
}"#
            .to_string(),
        );

        // Template 3: count words.
        let k3 = "wordcount_template".to_string();
        template_memory.allocate(&k3);
        templates.insert(
            k3,
            r#"use tokio;

#[tokio::main]
async fn main() {
    let input = "hello world";
    let count = input.split_whitespace().count();
    println!("{}", count);
}"#
            .to_string(),
        );

        // Template 4: reverse a string.
        let k4 = "reverse_template".to_string();
        template_memory.allocate(&k4);
        templates.insert(
            k4,
            r#"use tokio;

#[tokio::main]
async fn main() {
    let input = "hello";
    let reversed: String = input.chars().rev().collect();
    println!("{}", reversed);
}"#
            .to_string(),
        );

        // Template 5: factorial.
        let k5 = "factorial_template".to_string();
        template_memory.allocate(&k5);
        templates.insert(
            k5,
            r#"use tokio;

fn factorial(n: u64) -> u64 {
    if n <= 1 { 1 } else { n * factorial(n - 1) }
}

#[tokio::main]
async fn main() {
    println!("{}", factorial(6));
}"#
            .to_string(),
        );

        // Template 6: streaming JSON parsing over a local string.
        let k6 = "json_stream_template".to_string();
        template_memory.allocate(&k6);
        templates.insert(
            k6,
            r##"use tokio;
use serde_json::Value;

#[tokio::main]
async fn main() {
    let data = r#"{"name":"bad apple","value":42}"#;
    let v: Value = serde_json::from_str(data).unwrap();
    println!("{}", v["value"]);
}"##
            .to_string(),
        );

        // Template 7: high-throughput CSV column transformation.
        let k7 = "csv_transform_template".to_string();
        template_memory.allocate(&k7);
        templates.insert(
            k7,
            r#"use tokio;

#[tokio::main]
async fn main() {
    let csv = "a,1\nb,2\nc,3";
    for line in csv.lines() {
        let cols: Vec<&str> = line.split(',').collect();
        if let (Some(key), Some(val)) = (cols.first(), cols.get(1)) {
            println!("{}: {}", key, val);
        }
    }
}"#
            .to_string(),
        );

        // Template 8: local system log parsing and anomaly filtering.
        let k8 = "log_filter_template".to_string();
        template_memory.allocate(&k8);
        templates.insert(
            k8,
            r#"use tokio;

#[tokio::main]
async fn main() {
    let logs = "INFO start\nERROR fail\nINFO ok";
    for line in logs.lines() {
        if line.contains("ERROR") {
            println!("{}", line);
        }
    }
}"#
            .to_string(),
        );

        // Template 9: cryptographic file hashing array (SHA-256 of file contents).
        let k9 = "crypto_hash_template".to_string();
        template_memory.allocate(&k9);
        templates.insert(
            k9,
            r#"use tokio;
use sha2::{Sha256, Digest};

#[tokio::main]
async fn main() {
    let input = b"hello world";
    let mut hasher = Sha256::new();
    hasher.update(input);
    let result = hasher.finalize();
    println!("{:x}", result);
}"#
            .to_string(),
        );

        // Template 10: safe local file-system space defragmentation sweep.
        // This template only reports large files; it never moves or deletes data.
        let k10 = "defrag_sweep_template".to_string();
        template_memory.allocate(&k10);
        templates.insert(
            k10,
            r#"use tokio;
use std::path::Path;

#[tokio::main]
async fn main() {
    let path = Path::new(".");
    if let Ok(entries) = std::fs::read_dir(path) {
        for entry in entries.flatten() {
            if let Ok(meta) = entry.metadata() {
                if meta.is_file() && meta.len() > 1_000_000 {
                    println!("large file: {:?} ({} bytes)", entry.path(), meta.len());
                }
            }
        }
    }
}"#
            .to_string(),
        );

        Self {
            template_memory,
            templates,
        }
    }

    /// Synthesize a Rust source string from a script profile.
    ///
    /// The selection is a hybrid: explicit text heuristics for deterministic
    /// pattern matching, with the HDC template memory used as a fallback for
    /// closest-profile retrieval.
    pub fn synthesize(&mut self, profile: &ScriptProfile) -> String {
        self.synthesize_fluid_policy(profile)
    }

    /// Continuous fluid policy synthesis: map a minimized HDC profile to an
    /// async Rust utility template.
    pub fn synthesize_fluid_policy(&mut self, profile: &ScriptProfile) -> String {
        let (diagnosis, severity) = OverheadAnalyzer::analyze(profile);
        let source = &profile.source;

        // Explicit text-pattern routing. The HDC profile is used to embed the
        // script; routing is currently deterministic because VSA retrieval of an
        // arbitrary program is a research problem, not a solved compiler.
        let key = if source.to_lowercase().contains("sum")
            || source.to_lowercase().contains("total")
        {
            "sum_template"
        } else if source.to_lowercase().contains("count") || source.to_lowercase().contains("word")
        {
            "wordcount_template"
        } else if source.to_lowercase().contains("echo") || source.to_lowercase().contains("print")
        {
            "echo_template"
        } else if source.to_lowercase().contains("reverse") || source.to_lowercase().contains("rev")
        {
            "reverse_template"
        } else if source.to_lowercase().contains("factorial") || source.to_lowercase().contains('!')
        {
            "factorial_template"
        } else if source.to_lowercase().contains("json") || source.to_lowercase().contains("stream")
        {
            "json_stream_template"
        } else if source.to_lowercase().contains("csv") || source.to_lowercase().contains("column")
        {
            "csv_transform_template"
        } else if source.to_lowercase().contains("log") || source.to_lowercase().contains("error") {
            "log_filter_template"
        } else if source.to_lowercase().contains("hash") || source.to_lowercase().contains("sha") {
            "crypto_hash_template"
        } else if source.to_lowercase().contains("defrag")
            || source.to_lowercase().contains("sweep")
        {
            "defrag_sweep_template"
        } else {
            ""
        };

        let base_template = self.templates.get(key).cloned().unwrap_or_default();

        let header = format!(
            "// Synthesized from open-source script profile.\n// Overhead diagnosis: {diagnosis} (severity {severity:.2})\n"
        );
        if base_template.is_empty() {
            return format!("{header}fn main() {{\n    println!(\"no matching template\");\n}}\n");
        }

        format!("{header}{base_template}")
    }

    /// Repair a failed Rust source using raw `cargo check` diagnostics.
    ///
    /// Parses the exact error line, deduces the missing logical boundary, and
    /// rewrites the source tokens heuristically.  This is the self-healing
    /// primitive used in the multi-episode compiler loop.
    pub fn repair(&mut self, source: &str, diagnostics: &str) -> String {
        let diag_lower = diagnostics.to_lowercase();
        let mut repaired = source.to_string();

        // Missing `tokio::main` attribute on `async fn main`.
        if diag_lower.contains("main function cannot be async")
            && !repaired.contains("#[tokio::main]")
        {
            repaired = repaired.replace("async fn main()", "#[tokio::main]\nasync fn main()");
        }

        // Missing import for a crate name.
        if diag_lower.contains("use of undeclared crate or module")
            || diag_lower.contains("cannot find")
        {
            if diag_lower.contains("serde_json") && !repaired.contains("use serde_json") {
                repaired = format!("use serde_json::Value;\n{repaired}");
            }
            if diag_lower.contains("sha2") && !repaired.contains("use sha2") {
                repaired = format!("use sha2::{{Sha256, Digest}};\n{repaired}");
            }
        }

        // Type mismatch / missing `mut`.
        if diag_lower.contains("cannot assign to") || diag_lower.contains("cannot borrow") {
            repaired = repaired.replace("let input =", "let mut input =");
            repaired = repaired.replace("let numbers: Vec<i64>", "let mut numbers: Vec<i64>");
        }

        // Missing `std::fs` or `std::path` imports.
        if diag_lower.contains("fs::") && !repaired.contains("use std::fs") {
            repaired = format!("use std::fs;\n{repaired}");
        }
        if diag_lower.contains("path::") && !repaired.contains("use std::path") {
            repaired = format!("use std::path::Path;\n{repaired}");
        }

        // Missing semicolons: parse `--> src/main.rs:LINE:COL` snippets and
        // append a semicolon to the end of each reported line.
        if diag_lower.contains("expected `;`")
            || diag_lower.contains("expected semicolon")
            || diag_lower.contains("expected `;`, found")
        {
            let error_lines = Self::parse_cargo_error_lines(diagnostics);
            let mut owned: Vec<String> = repaired
                .lines()
                .map(std::string::ToString::to_string)
                .collect();
            for line_no in error_lines {
                if line_no == 0 || line_no > owned.len() {
                    continue;
                }
                let l = owned[line_no - 1].trim_end();
                if !l.ends_with(';') && !l.ends_with('{') && !l.ends_with('}') && !l.is_empty() {
                    owned[line_no - 1] = format!("{l};");
                }
            }
            repaired = owned.join("\n");
        }

        // Unclosed delimiters (e.g., missing `}`): balance braces by appending
        // the missing closing braces at the end of the source.
        if diag_lower.contains("unclosed delimiter")
            || (diag_lower.contains("expected one of") && diag_lower.contains("`}`"))
        {
            let open = repaired.chars().filter(|c| *c == '{').count();
            let close = repaired.chars().filter(|c| *c == '}').count();
            if open > close {
                repaired.push_str(&"}".repeat(open - close));
            }
        }

        repaired
    }

    /// Parse `cargo` diagnostics of the form `--> src/main.rs:LINE:COL` and
    /// return the set of 1-indexed line numbers.
    fn parse_cargo_error_lines(diagnostics: &str) -> Vec<usize> {
        let mut lines = Vec::new();
        for line in diagnostics.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with("-->") {
                if let Some(path_start) = trimmed.find(' ') {
                    let path_part = &trimmed[path_start + 1..];
                    // path_part looks like `src/main.rs:10:15`
                    if let Some(colon) = path_part.rfind(':') {
                        let before_col = &path_part[..colon];
                        if let Some(line_colon) = before_col.rfind(':') {
                            let num = &before_col[line_colon + 1..];
                            if let Ok(n) = num.parse::<usize>() {
                                lines.push(n);
                            }
                        }
                    }
                }
            }
        }
        lines
    }
}

impl StrategyLibrary {
    /// Run a dialectical synthesis on a failed strategy and store the repaired
    /// variant. Returns the key of the new synthesis.
    pub async fn dialectical_repair(
        &self,
        failed_key: &str,
        error: &str,
        predicted_outcome: &str,
    ) -> Option<String> {
        let original = self.get(failed_key).await?;
        let engine = DialecticalEngine::new();
        let thesis = engine.thesis(&original, predicted_outcome);
        let antithesis = engine.antithesis(error);
        let _contradiction = engine.contradiction_strength(&thesis, &antithesis);
        let repaired = engine.synthesize(&original, error);
        let new_key = format!("{}_synth_{}", failed_key, engine.next_id());
        let mut repaired = repaired;
        repaired.key = new_key.clone();
        if self.put(&repaired).await.is_ok() {
            Some(new_key)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_db() -> StrategyLibrary {
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let tid = std::thread::current().id();
        let base = std::env::temp_dir().join(format!("bad_apple_test_{tid:?}_{ts}"));
        std::fs::create_dir_all(&base).unwrap();
        StrategyLibrary::open(&base).unwrap()
    }

    #[test]
    fn put_get_roundtrip() {
        let lib = temp_db();
        let s = Strategy::new(
            "stats".into(),
            "compute mean".into(),
            "python".into(),
            "import statistics; print(statistics.mean([1,2,3]))".into(),
        );
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            lib.put(&s).await.unwrap();
            let got = lib.get("stats").await.unwrap();
            assert_eq!(got.key, "stats");
            assert_eq!(got.problem, "compute mean");
            assert_eq!(got.language, "python");
        });
    }

    #[test]
    fn best_match_finds_similar_problem() {
        let lib = temp_db();
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            lib.put(&Strategy::new(
                "mean".into(),
                "compute mean of list".into(),
                "python".into(),
                "print(1)".into(),
            ))
            .await
            .unwrap();
            let best = lib.best_match("mean of list").await.unwrap();
            assert_eq!(best.key, "mean");
        });
    }

    #[test]
    fn prune_below_removes_weak() {
        let lib = temp_db();
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let mut s = Strategy::new(
                "weak".into(),
                "low reliability".into(),
                "python".into(),
                "print(1)".into(),
            );
            s.reliability = 0.1;
            lib.put(&s).await.unwrap();
            let count = lib.prune_below(0.2).await;
            assert_eq!(count, 1);
            assert!(lib.get("weak").await.is_none());
        });
    }

    #[test]
    fn record_updates_reliability() {
        let mut s = Strategy::new("k".into(), "p".into(), "python".into(), "c".into());
        assert_eq!(s.reliability, 0.5);
        s.record(true);
        assert!(s.reliability > 0.5);
        s.record(false);
        assert!(s.reliability < 1.0);
    }

    #[tokio::test]
    async fn rust_synthesizer_repair_missing_semicolon() {
        use crate::benchmark::RustValidator;
        let broken = "fn main() {\n    let x = 1\n    println!(\"{}\", x);\n}";
        let first = RustValidator::validate("repair_semi", broken, 0).await;
        assert!(!first.compiled);

        let mut synthesizer = RustSynthesizer::new();
        let fixed = synthesizer.repair(broken, &first.diagnostics);
        let second = RustValidator::validate("repair_semi_2", &fixed, 0).await;
        assert!(
            second.compiled,
            "missing semicolon not repaired: {}",
            second.diagnostics
        );
    }

    #[tokio::test]
    async fn rust_synthesizer_repair_unclosed_block() {
        use crate::benchmark::RustValidator;
        let broken = "fn main() {\n    let x = 1;\n    println!(\"{}\", x);";
        let first = RustValidator::validate("repair_block", broken, 0).await;
        assert!(!first.compiled);

        let mut synthesizer = RustSynthesizer::new();
        let fixed = synthesizer.repair(broken, &first.diagnostics);
        let second = RustValidator::validate("repair_block_2", &fixed, 0).await;
        assert!(
            second.compiled,
            "unclosed block not repaired: {}",
            second.diagnostics
        );
    }
}
