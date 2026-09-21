//! Durable strategy cache for the Bad Apple agent.
//!
//! A strategy is a proven tool template keyed by a problem signature (typically
//! a goal string or a domain). The library is backed by redb so strategies
//! survive restarts, and it exposes a policy-improvement interface for pruning
//! low-reliability templates.

use crate::production_blueprint::{CausalGraph, CausalRelation};
use crate::redb_kv;
use anyhow::Result;
use redb::Database;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
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

/// redb-backed strategy library.
#[derive(Clone)]
pub struct StrategyLibrary {
    db: Arc<Database>,
    /// Causal graph that records how strategies relate to system assets and
    /// constraints.  Shared across clones of the library.
    causal_graph: Arc<Mutex<CausalGraph>>,
}

impl StrategyLibrary {
    /// Open or create the redb database at the given path.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let db = redb_kv::open(path.as_ref())?;
        Ok(Self {
            db: Arc::new(db),
            causal_graph: Arc::new(Mutex::new(CausalGraph::bad_apple_default())),
        })
    }

    /// Read a strategy by key.
    pub async fn get(&self, key: &str) -> Option<Strategy> {
        let db = Arc::clone(&self.db);
        let key = key.to_string();
        match spawn_blocking(move || redb_kv::get(&db, key.as_bytes())).await {
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
        spawn_blocking(move || redb_kv::insert(&db, key.as_bytes(), &bytes)).await??;

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
        spawn_blocking(move || redb_kv::remove(&db, key.as_bytes())).await??;
        Ok(())
    }

    /// Return all strategies with reliability below the threshold.
    pub async fn weak_strategies(&self, threshold: f64) -> Vec<Strategy> {
        let db = Arc::clone(&self.db);
        spawn_blocking(move || {
            let mut out = Vec::new();
            for (_, value) in redb_kv::iter(&db).unwrap_or_default() {
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
            for (_, value) in redb_kv::iter(&db).unwrap_or_default() {
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

/// A lightweight, pattern-matching Rust code repair helper.
///
/// Parses `cargo check` diagnostics and applies small, deterministic rewrites
/// (missing semicolons, unclosed delimiters, missing imports, etc.).
pub struct RustSynthesizer;

impl RustSynthesizer {
    pub fn new() -> Self {
        Self
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

/// Directory the dashboard scans for curious proposals
/// (`~/.bad_apple/notes/proposed_patches`).
pub fn strategy_proposals_dir() -> PathBuf {
    std::env::var("BADAPPLE_STRATEGY_PROPOSALS")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
            PathBuf::from(home)
                .join(".bad_apple")
                .join("notes")
                .join("proposed_patches")
        })
}

fn proposal_timestamp() -> String {
    chrono::Utc::now().format("%Y%m%d-%H%M%S").to_string()
}

impl StrategyLibrary {
    /// Dialectical synthesis on a failed strategy, kept *outside* the live
    /// library: the repaired candidate is written as a curious-format
    /// proposal for governed adoption (`strategy adopt <path>`), never
    /// auto-installed. Returns the proposal path, or None when the
    /// contradiction is too weak to carry new information (a strategy that
    /// was already unreliable failing again is noise, not signal).
    pub async fn propose_repair(
        &self,
        failed_key: &str,
        error: &str,
        predicted_outcome: &str,
    ) -> Option<PathBuf> {
        let original = self.get(failed_key).await?;
        let engine = DialecticalEngine::new();
        let thesis = engine.thesis(&original, predicted_outcome);
        let antithesis = engine.antithesis(error);
        let strength = engine.contradiction_strength(&thesis, &antithesis);
        if strength < 0.2 {
            return None;
        }
        let mut candidate = engine.synthesize(&original, error);
        candidate.key = format!("{}_synth_{}", failed_key, engine.next_id());

        let dir = strategy_proposals_dir();
        std::fs::create_dir_all(&dir).ok()?;
        let safe_key: String = failed_key
            .chars()
            .map(|c| if c.is_alphanumeric() { c } else { '-' })
            .collect();
        let path = dir.join(format!("strategy-{safe_key}-{}.md", proposal_timestamp()));
        let error_excerpt: String = error.chars().take(500).collect();
        let body = format!(
            "# Strategy repair proposal: {failed_key}\n\n\
             **When:** {when} UTC\n\
             **Workspace:** system\n\n\
             ## Finding\n\n\
             - Failed strategy: `{failed_key}`\n\
             - Prior reliability: {rel:.2} over {uses} use(s)\n\
             - Contradiction strength: {strength:.2}\n\
             - Observed error: {error_excerpt}\n\n\
             ## Proposal\n\n\
             ```json\n{json}\n```\n\n\
             ## Adoption\n\n\
             `badapple strategy adopt {path}`\n",
            when = chrono::Utc::now().format("%Y-%m-%d %H:%M:%S"),
            rel = original.reliability,
            uses = original.uses,
            strength = strength,
            path = path.display(),
            json = serde_json::to_string_pretty(&serde_json::json!({
                "no_patch": true,
                "strategy": {
                    "key": candidate.key,
                    "problem": candidate.problem,
                    "language": candidate.language,
                    "code": candidate.code,
                }
            }))
            .unwrap_or_default(),
        );
        std::fs::write(&path, body).ok()?;
        Some(path)
    }

    /// Adopt a repair proposal: parse the strategy block from its ```json
    /// section and install it into the library. Marks the proposal applied.
    pub async fn adopt_proposal<P: AsRef<Path>>(
        &self,
        path: P,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path)?;
        let proposal_section = text
            .split("## Proposal")
            .nth(1)
            .ok_or("no ## Proposal section")?;
        let json_text = proposal_section
            .split("```json")
            .nth(1)
            .and_then(|b| b.split("```").next())
            .ok_or("no ```json block in proposal")?;
        let v: serde_json::Value = serde_json::from_str(json_text.trim())?;
        let s = v
            .get("strategy")
            .ok_or("proposal json has no strategy field")?;
        let strategy = Strategy {
            key: s["key"].as_str().ok_or("strategy.key missing")?.to_string(),
            problem: s["problem"].as_str().unwrap_or_default().to_string(),
            language: s["language"].as_str().unwrap_or("shell").to_string(),
            code: s["code"].as_str().unwrap_or_default().to_string(),
            reliability: 0.5,
            uses: 0,
            successes: 0,
        };
        let key = strategy.key.clone();
        self.put(&strategy).await?;
        let mut marked = text;
        marked.push_str(&format!(
            "\n## Applied\n\nAdopted {} UTC via `strategy adopt`.\n",
            chrono::Utc::now().format("%Y-%m-%d %H:%M:%S")
        ));
        let _ = std::fs::write(path, marked);
        Ok(key)
    }

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

#[cfg(test)]
mod library_tests {
    use super::{Strategy, StrategyLibrary};

    fn temp_lib(name: &str) -> (StrategyLibrary, std::path::PathBuf) {
        let path =
            std::env::temp_dir().join(format!("ba-strategies-{name}-{}.redb", std::process::id()));
        let _ = std::fs::remove_file(&path);
        (StrategyLibrary::open(&path).unwrap(), path)
    }

    #[tokio::test]
    async fn record_then_match_roundtrip() {
        let (lib, path) = temp_lib("roundtrip");
        lib.put(&Strategy::new(
            "tool:list_files".into(),
            "list files in a directory".into(),
            "shell".into(),
            "ls -la".into(),
        ))
        .await
        .unwrap();
        let mut s = lib.get("tool:list_files").await.unwrap();
        s.record(true);
        s.record(true);
        lib.put(&s).await.unwrap();
        let best = lib.best_match("list files in a directory").await.unwrap();
        assert_eq!(best.key, "tool:list_files");
        assert!(best.reliability > 0.5);
        let _ = std::fs::remove_file(path);
    }

    #[tokio::test]
    async fn failures_decay_reliability() {
        let (lib, path) = temp_lib("decay");
        let mut s = Strategy::new("k".into(), "p".into(), "shell".into(), "c".into());
        for _ in 0..5 {
            s.record(false);
        }
        lib.put(&s).await.unwrap();
        let weak = lib.weak_strategies(0.3).await;
        assert_eq!(weak.len(), 1);
        let pruned = lib.prune_below(0.3).await;
        assert_eq!(pruned, 1);
        assert!(lib.get("k").await.is_none());
        let _ = std::fs::remove_file(path);
    }

    #[tokio::test]
    async fn repair_writes_proposal_not_library_entry() {
        let prop_dir = std::env::temp_dir().join(format!("ba-strat-props-{}", std::process::id()));
        std::env::set_var("BADAPPLE_STRATEGY_PROPOSALS", &prop_dir);
        let (lib, path) = temp_lib("repair");
        let mut s = Strategy::new(
            "tool:flash".into(),
            "flash the firmware".into(),
            "shell".into(),
            "flash.sh".into(),
        );
        // Raise reliability so the failure is a genuine contradiction.
        for _ in 0..4 {
            s.record(true);
        }
        lib.put(&s).await.unwrap();

        let proposal = lib
            .propose_repair("tool:flash", "missing module avrdude", "flash completes")
            .await
            .expect("proposal should be written");
        let text = std::fs::read_to_string(&proposal).unwrap();
        assert!(text.contains("## Proposal"));
        assert!(text.contains("strategy"));
        // The candidate must NOT be in the live library yet.
        assert_eq!(lib.weak_strategies(2.0).await.len(), 1);

        // Adoption is the explicit governed step.
        let key = lib.adopt_proposal(&proposal).await.unwrap();
        assert!(key.starts_with("tool:flash_synth_"));
        assert!(lib.get(&key).await.is_some());
        assert_eq!(lib.weak_strategies(2.0).await.len(), 2);
        assert!(std::fs::read_to_string(&proposal)
            .unwrap()
            .contains("## Applied"));

        std::env::remove_var("BADAPPLE_STRATEGY_PROPOSALS");
        let _ = std::fs::remove_dir_all(prop_dir);
        let _ = std::fs::remove_file(path);
    }

    #[tokio::test]
    async fn weak_contradiction_writes_no_proposal() {
        let prop_dir =
            std::env::temp_dir().join(format!("ba-strat-props-weak-{}", std::process::id()));
        std::env::set_var("BADAPPLE_STRATEGY_PROPOSALS", &prop_dir);
        let (lib, path) = temp_lib("weakprop");
        let mut s = Strategy::new("tool:x".into(), "p".into(), "shell".into(), "c".into());
        for _ in 0..6 {
            s.record(false); // reliability ~0.17 — failure is expected
        }
        lib.put(&s).await.unwrap();
        assert!(lib
            .propose_repair("tool:x", "it failed again", "works")
            .await
            .is_none());
        std::env::remove_var("BADAPPLE_STRATEGY_PROPOSALS");
        let _ = std::fs::remove_dir_all(prop_dir);
        let _ = std::fs::remove_file(path);
    }
}
