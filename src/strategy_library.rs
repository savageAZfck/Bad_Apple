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
    pub async fn put(&self, strategy: &Strategy) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
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
        match spawn_blocking(move || {
            let mut out = Vec::new();
            for item in db.iter() {
                if let Ok((_, value)) = item {
                    if let Ok(s) = serde_json::from_slice::<Strategy>(&value) {
                        if s.reliability < threshold {
                            out.push(s);
                        }
                    }
                }
            }
            out
        }).await {
            Ok(v) => v,
            Err(_) => Vec::new(),
        }
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
        let problem_words: std::collections::HashSet<String> = problem_lower.split_whitespace().map(|s| s.to_string()).collect();
        match spawn_blocking(move || {
            let mut best: Option<Strategy> = None;
            let mut best_score = 0.0;
            for item in db.iter() {
                if let Ok((_, value)) = item {
                    if let Ok(s) = serde_json::from_slice::<Strategy>(&value) {
                        let s_lower = s.problem.to_lowercase();
                        let mut score = if s_lower.contains(&problem_lower) || problem_lower.contains(&s_lower) {
                            1.0
                        } else {
                            let s_words: std::collections::HashSet<String> = s_lower.split_whitespace().map(|s| s.to_string()).collect();
                            let total = problem_words.union(&s_words).count();
                            let overlap = problem_words.intersection(&s_words).count();
                            if total > 0 { overlap as f64 / total as f64 } else { 0.0 }
                        };
                        score *= 0.8 + 0.2 * s.reliability;
                        if score > 0.7 && score > best_score {
                            best_score = score;
                            best = Some(s);
                        }
                    }
                }
            }
            best
        }).await {
            Ok(v) => v,
            Err(_) => None,
        }
    }
}
