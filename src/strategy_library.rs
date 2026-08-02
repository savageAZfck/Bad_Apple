//! Durable strategy cache for the Firefly agent.
//!
//! A strategy is a proven tool template keyed by a problem signature (typically
//! a goal string or a domain). The library is backed by Sled so strategies
//! survive restarts, and it exposes a policy-improvement interface for pruning
//! low-reliability templates.

use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::Arc;
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
}

impl StrategyLibrary {
    /// Open or create the Sled database at the given path.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, sled::Error> {
        let db = sled::open(path)?;
        Ok(Self { db: Arc::new(db) })
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

    /// Insert or overwrite a strategy.
    pub async fn put(
        &self,
        strategy: &Strategy,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let db = Arc::clone(&self.db);
        let bytes = serde_json::to_vec(strategy)?;
        let key = strategy.key.clone();
        spawn_blocking(move || db.insert(key.as_bytes(), bytes).map(|_| ())).await??;
        Ok(())
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
            .map(|s| s.to_string())
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
                            let s_words: std::collections::HashSet<String> =
                                s_lower.split_whitespace().map(|s| s.to_string()).collect();
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

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_db() -> StrategyLibrary {
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let tid = std::thread::current().id();
        let base = std::env::temp_dir().join(format!("sapient_soul_test_{:?}_{}", tid, ts));
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
}
