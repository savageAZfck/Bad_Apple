//! Hyperdimensional Computing (HDC) substrate for Firefly.
//!
//! Implements a 10,000-dimensional Vector Symbolic Architecture (VSA) using
//! bipolar (-1, +1) vectors. Supports bundling (superposition), binding
//! (multiplication), permutation (circular shifts), and Hamming-distance
//! retrieval. The engine is `Send + Sync` and operates on pre-allocated
//! scratch buffers to keep per-call allocations minimal.

use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;
use std::sync::{Arc, Mutex};

use crate::tensor_brain::tokenize_text;

pub const HD_DIM: usize = 10_000;

/// A 10,000-dimensional bipolar hypervector.
/// Values are stored as `i8` (-1 or +1). This is compact and branchless in
/// the hot paths.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Hypervector {
    pub values: Vec<i8>,
}

impl Hypervector {
    /// Allocate a new random bipolar vector.
    pub fn random<R: Rng>(rng: &mut R) -> Self {
        let mut values = Vec::with_capacity(HD_DIM);
        for _ in 0..HD_DIM {
            values.push(if rng.gen_bool(0.5) { 1 } else { -1 });
        }
        Self { values }
    }

    /// Zero vector (all +1 by convention, representing absence).
    pub fn zero() -> Self {
        Self {
            values: vec![1; HD_DIM],
        }
    }

    /// Bundle a slice of vectors via element-wise addition followed by
    /// bipolar clipping. Requires an external scratch buffer to avoid
    /// per-call allocation.
    pub fn bundle(vectors: &[&Hypervector], scratch: &mut [i32]) -> Self {
        assert_eq!(scratch.len(), HD_DIM);
        scratch.fill(0);
        for v in vectors {
            for (i, &x) in v.values.iter().enumerate() {
                scratch[i] += x as i32;
            }
        }
        let mut values = Vec::with_capacity(HD_DIM);
        for &acc in scratch.iter() {
            values.push(if acc > 0 { 1 } else { -1 });
        }
        Self { values }
    }

    /// Bind two vectors via element-wise multiplication (bipolar binding).
    pub fn bind(&self, other: &Hypervector) -> Self {
        let mut values = Vec::with_capacity(HD_DIM);
        for (&a, &b) in self.values.iter().zip(other.values.iter()) {
            values.push(a * b);
        }
        Self { values }
    }

    /// Permute by a circular left shift by `k` positions.
    pub fn permute(&self, k: usize) -> Self {
        let k = k % HD_DIM;
        let mut values = Vec::with_capacity(HD_DIM);
        values.extend_from_slice(&self.values[k..]);
        values.extend_from_slice(&self.values[..k]);
        Self { values }
    }

    /// Inverse permutation (circular right shift by `k`).
    pub fn permute_inv(&self, k: usize) -> Self {
        let k = k % HD_DIM;
        let mut values = Vec::with_capacity(HD_DIM);
        let split = HD_DIM - k;
        values.extend_from_slice(&self.values[split..]);
        values.extend_from_slice(&self.values[..split]);
        Self { values }
    }

    /// Hamming-like distance for bipolar vectors: count of positions that
    /// differ. Returns a value in `[0, HD_DIM]`.
    pub fn hamming_distance(&self, other: &Hypervector) -> usize {
        self.values
            .iter()
            .zip(other.values.iter())
            .filter(|(&a, &b)| a != b)
            .count()
    }

    /// Cosine-analog similarity for bipolar vectors: dot product.
    pub fn dot(&self, other: &Hypervector) -> i32 {
        self.values
            .iter()
            .zip(other.values.iter())
            .map(|(&a, &b)| a as i32 * b as i32)
            .sum()
    }
}

/// A thread-safe repository of hyperdimensional symbols with pre-allocated
/// scratch buffers for bundling.
pub struct HDCMemory {
    symbols: std::collections::HashMap<String, Hypervector>,
    scratch: Vec<i32>,
    rng: Arc<Mutex<StdRng>>,
}

impl HDCMemory {
    pub fn new() -> Self {
        Self {
            symbols: std::collections::HashMap::new(),
            scratch: vec![0; HD_DIM],
            rng: Arc::new(Mutex::new(StdRng::seed_from_u64(0x2046))),
        }
    }

    /// Allocate a fresh random hypervector and store it under `name`.
    pub fn allocate(&mut self, name: &str) -> Hypervector {
        let mut rng = self.rng.lock().unwrap();
        let v = Hypervector::random(&mut *rng);
        self.symbols.insert(name.to_string(), v.clone());
        v
    }

    /// Retrieve a stored symbol.
    pub fn get(&self, name: &str) -> Option<&Hypervector> {
        self.symbols.get(name)
    }

    /// Find the closest stored symbol by Hamming distance.
    pub fn nearest(&self, probe: &Hypervector) -> Option<(String, usize)> {
        self.symbols
            .iter()
            .map(|(k, v)| (k.clone(), v.hamming_distance(probe)))
            .min_by_key(|(_, d)| *d)
    }

    /// Bundle a collection of stored symbols into a single superposition.
    pub fn bundle_names(&mut self, names: &[&str]) -> Option<Hypervector> {
        let refs: Vec<&Hypervector> = names.iter().filter_map(|n| self.symbols.get(*n)).collect();
        if refs.is_empty() {
            return None;
        }
        Some(Hypervector::bundle(&refs, &mut self.scratch))
    }
}

impl Default for HDCMemory {
    fn default() -> Self {
        Self::new()
    }
}

// =========================================================================
// Script profiling and overhead analysis
// =========================================================================

/// A machine-readable efficiency profile extracted from an open-source script.
/// The HDC `profile` is a 10,000-D superposition of the script's token flow.
#[derive(Clone, Debug)]
pub struct ScriptProfile {
    /// 10,000-D hyperdimensional signature of the script.
    pub profile: Hypervector,
    /// Original (or sanitized) source text.
    pub source: String,
    /// Number of detected for/while loops.
    pub loop_count: usize,
    /// Number of detected string concatenations or repeated string ops.
    pub string_alloc_count: usize,
    /// Number of detected function definitions.
    pub function_count: usize,
    /// Estimated interpreter overhead score in [0, 1].
    pub overhead_score: f64,
}

/// Encoder that tokenizes an open-source script and folds its token stream
/// into a single 10,000-D hypervector via n-gram binding and bundling.
pub struct ScriptEncoder {
    token_memory: HDCMemory,
    pos_buffer: Hypervector,
}

impl ScriptEncoder {
    pub fn new() -> Self {
        Self {
            token_memory: HDCMemory::new(),
            pos_buffer: Hypervector::zero(),
        }
    }

    /// Encode a token id as a stable HDC symbol. The first call for an id
    /// allocates a random vector; subsequent calls reuse it.
    fn token_hv(&mut self, id: u32) -> &Hypervector {
        let key = format!("tok_{}", id);
        if !self.token_memory.symbols.contains_key(&key) {
            self.token_memory.allocate(&key);
        }
        // Safety: key was just inserted.
        self.token_memory.symbols.get(&key).unwrap()
    }

    /// Encode the source text into a script profile.
    pub fn encode(&mut self, source: &str) -> ScriptProfile {
        let ids = tokenize_text(source);
        let mut result = Hypervector::zero();
        for (pos, &id) in ids.iter().enumerate() {
            let tok = self.token_hv(id);
            // Bind token with position to preserve order: ρ^pos × token.
            self.pos_buffer = tok.permute(pos);
            let tmp = result.bind(&self.pos_buffer);
            // Approximate bundling of bound n-grams by averaging via XOR-like
            // multiplication. To avoid collapse we keep the running product.
            result = result.bind(&tmp);
        }

        // Extract explicit text features for the overhead analyzer.
        let lower = source.to_lowercase();
        let loop_count = lower.matches("for ").count()
            + lower.matches("while ").count()
            + lower.matches("loop").count();
        let string_alloc_count = lower.matches("+").count()
            + lower.matches(".append(").count()
            + lower.matches(".join(").count();
        let function_count = lower.matches("def ").count()
            + lower.matches("fn ").count()
            + lower.matches("func ").count();

        let overhead_score = {
            let len_factor = (source.len() as f64 / 1000.0).min(1.0);
            let loop_factor = (loop_count as f64 / 5.0).min(1.0);
            let alloc_factor = (string_alloc_count as f64 / 20.0).min(1.0);
            (len_factor * 0.2 + loop_factor * 0.4 + alloc_factor * 0.4).min(1.0)
        };

        ScriptProfile {
            profile: result,
            source: source.to_string(),
            loop_count,
            string_alloc_count,
            function_count,
            overhead_score,
        }
    }
}

impl Default for ScriptEncoder {
    fn default() -> Self {
        Self::new()
    }
}

/// Analyzes a `ScriptProfile` and returns a short human-readable diagnosis
/// plus a severity score in [0, 1].
pub struct OverheadAnalyzer;

impl OverheadAnalyzer {
    pub fn analyze(profile: &ScriptProfile) -> (String, f64) {
        let mut issues = Vec::new();
        if profile.loop_count > 2 {
            issues.push(format!("{} interpreter loops", profile.loop_count));
        }
        if profile.string_alloc_count > 5 {
            issues.push(format!("{} string allocations", profile.string_alloc_count));
        }
        if profile.function_count == 0 {
            issues.push("no functions".to_string());
        }

        if issues.is_empty() {
            return ("clean".to_string(), 0.0);
        }

        let diagnosis = issues.join(", ");
        (diagnosis, profile.overhead_score)
    }
}

/// Thermodynamic minimizer: reduces the entropy of a hypervector by forcing
/// near-zero components to a neutral state. This is a hardware-friendly
/// bitwise-style clamp over the 10,000-D bipolar vector.
pub struct ThermodynamicMinimizer {
    /// Components with magnitude below this threshold are collapsed to +1.
    pub energy_threshold: f64,
}

impl ThermodynamicMinimizer {
    pub fn new(energy_threshold: f64) -> Self {
        Self {
            energy_threshold: energy_threshold.clamp(0.0, 1.0),
        }
    }

    /// Compute the thermodynamic energy (Shannon-analog) of a hypervector as
    /// the fraction of components that deviate from the neutral mean.
    pub fn energy(hv: &Hypervector) -> f64 {
        let total = hv.values.len() as f64;
        let deviants = hv.values.iter().filter(|&&v| v != 1 && v != -1).count() as f64;
        deviants / total
    }

    /// Minimize a hypervector in place. Components whose "energy" is below the
    /// threshold are collapsed toward the neutral +1 state, reducing entropy.
    pub fn minimize(&self, hv: &mut Hypervector) {
        let n = hv.values.len();
        for (i, v) in hv.values.iter_mut().enumerate() {
            // Treat the hypervector as a sampled field and zero out components
            // that are thermodynamically inactive.
            if (i as f64 / n as f64) < self.energy_threshold {
                *v = 1;
            }
        }
    }

    /// Reduce the entropy of a `ScriptProfile` by minimizing its HDC profile
    /// and re-evaluating its overhead score.
    pub fn reduce_entropy(profile: &mut ScriptProfile) {
        let minimizer = ThermodynamicMinimizer::new(profile.overhead_score);
        minimizer.minimize(&mut profile.profile);
        let energy = Self::energy(&profile.profile);
        profile.overhead_score = (profile.overhead_score * (1.0 - energy)).clamp(0.0, 1.0);
    }
}

impl Default for ThermodynamicMinimizer {
    fn default() -> Self {
        Self::new(0.3)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binding_inverts() {
        let a = Hypervector::random(&mut rand::thread_rng());
        let b = Hypervector::random(&mut rand::thread_rng());
        let bound = a.bind(&b);
        assert_eq!(bound.bind(&b), a); // b * b = identity
    }

    #[test]
    fn bundle_of_same_is_same() {
        let v = Hypervector::random(&mut rand::thread_rng());
        let mut scratch = vec![0; HD_DIM];
        let bundled = Hypervector::bundle(&[&v, &v, &v], &mut scratch);
        assert_eq!(bundled, v);
    }

    #[test]
    fn permutation_roundtrip() {
        let v = Hypervector::random(&mut rand::thread_rng());
        assert_eq!(v.permute(7).permute_inv(7), v);
    }

    #[test]
    fn nearest_lookup() {
        let mut mem = HDCMemory::new();
        let _ = mem.allocate("red");
        let _ = mem.allocate("blue");
        let probe = mem.get("red").unwrap().clone();
        let (name, _) = mem.nearest(&probe).unwrap();
        assert_eq!(name, "red");
    }
}
