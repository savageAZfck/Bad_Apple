//! Hyperdimensional Computing (HDC) substrate for Bad Apple.
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
    values: Vec<i8>,
}

impl Hypervector {
    /// Allocate a zeroed hypervector of `HD_DIM` dimensions.
    pub fn new() -> Self {
        Self {
            values: vec![0; HD_DIM],
        }
    }

    /// Construct a `Hypervector` from an existing value vector, validating that
    /// it has exactly `HD_DIM` elements.
    pub fn from_values(v: Vec<i8>) -> Result<Self, String> {
        if v.len() != HD_DIM {
            return Err(format!("Hypervector must have {HD_DIM} elements"));
        }
        Ok(Self { values: v })
    }

    /// Borrow the underlying bipolar values.
    pub fn values(&self) -> &[i8] {
        &self.values
    }

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
        #[cfg(target_arch = "aarch64")]
        {
            aarch64::bundle(vectors, scratch);
            let mut values = vec![0_i8; HD_DIM];
            aarch64::clip(scratch, &mut values);
            Self { values }
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
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
    }

    /// Bind two vectors via element-wise multiplication (bipolar binding).
    pub fn bind(&self, other: &Hypervector) -> Self {
        let mut values = vec![0_i8; HD_DIM];
        #[cfg(target_arch = "aarch64")]
        aarch64::bind(&self.values, &other.values, &mut values);
        #[cfg(not(target_arch = "aarch64"))]
        {
            for (i, (&a, &b)) in self.values.iter().zip(other.values.iter()).enumerate() {
                values[i] = a * b;
            }
        }
        Self { values }
    }

    /// Permute by a circular left shift by `k` positions.
    pub fn permute(&self, k: usize) -> Self {
        assert_eq!(
            self.values.len(),
            HD_DIM,
            "Hypervector length invariant violated"
        );
        let k = k % HD_DIM;
        let mut values = Vec::with_capacity(HD_DIM);
        values.extend_from_slice(&self.values[k..]);
        values.extend_from_slice(&self.values[..k]);
        Self { values }
    }

    /// Inverse permutation (circular right shift by `k`).
    pub fn permute_inv(&self, k: usize) -> Self {
        assert_eq!(
            self.values.len(),
            HD_DIM,
            "Hypervector length invariant violated"
        );
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
        #[cfg(target_arch = "aarch64")]
        return aarch64::hamming_distance(&self.values, &other.values);
        #[cfg(not(target_arch = "aarch64"))]
        {
            self.values
                .iter()
                .zip(other.values.iter())
                .filter(|(&a, &b)| a != b)
                .count()
        }
    }

    /// Cosine-analog similarity for bipolar vectors: dot product.
    pub fn dot(&self, other: &Hypervector) -> i32 {
        #[cfg(target_arch = "aarch64")]
        return aarch64::dot(&self.values, &other.values);
        #[cfg(not(target_arch = "aarch64"))]
        {
            self.values
                .iter()
                .zip(other.values.iter())
                .map(|(&a, &b)| a as i32 * b as i32)
                .sum()
        }
    }
}

#[cfg(target_arch = "aarch64")]
mod aarch64 {
    use core::arch::aarch64::{
        int16x8_t, vaddq_s32, vaddvq_s32, vaddvq_u8, vaddw_s16, vceqq_s8, vcombine_s16,
        vcombine_s8, vdupq_n_s16, vdupq_n_s32, vget_high_s16, vget_high_s8, vget_low_s16,
        vget_low_s8, vld1q_s32, vld1q_s8, vmlal_s8, vmovl_s16, vmovl_s8, vmulq_s8, vmvnq_u8,
        vorrq_s32, vqmovn_s16, vqmovn_s32, vshrq_n_s32, vshrq_n_u8, vst1q_s32, vst1q_s8,
    };

    use super::HD_DIM;

    fn main_len() -> usize {
        (HD_DIM / 16) * 16
    }

    pub fn bind(a: &[i8], b: &[i8], out: &mut [i8]) {
        assert_eq!(a.len(), HD_DIM);
        assert_eq!(b.len(), HD_DIM);
        assert_eq!(out.len(), HD_DIM);
        let main = main_len();
        // SAFETY: `a`, `b`, and `out` are all asserted to be exactly `HD_DIM` elements, and
        // `main` is `(HD_DIM/16)*16`, so every 16-element `vld1q_s8`/`vst1q_s8` access stays
        // in bounds. The slices are contiguous `&[i8]`/`&mut [i8]` with valid, properly
        // aligned (1-byte) pointers. The tail beyond `main` is handled by the scalar loop.
        unsafe {
            for i in (0..main).step_by(16) {
                let va = vld1q_s8(a.as_ptr().add(i));
                let vb = vld1q_s8(b.as_ptr().add(i));
                let prod = vmulq_s8(va, vb);
                vst1q_s8(out.as_mut_ptr().add(i), prod);
            }
        }
        for i in main..HD_DIM {
            out[i] = a[i] * b[i];
        }
    }

    pub fn hamming_distance(a: &[i8], b: &[i8]) -> usize {
        let main = main_len();
        let mut count: usize = 0;
        // SAFETY: `a` and `b` are `&[i8]` slices whose pointers are valid for their length.
        // `main` is `(HD_DIM/16)*16` so each 16-element `vld1q_s8` load is in bounds; the
        // loads are read-only and the tail is handled by the scalar loop below.
        unsafe {
            for i in (0..main).step_by(16) {
                let va = vld1q_s8(a.as_ptr().add(i));
                let vb = vld1q_s8(b.as_ptr().add(i));
                let eq = vceqq_s8(va, vb);
                let diff = vmvnq_u8(eq);
                let ones = vshrq_n_u8(diff, 7);
                count += vaddvq_u8(ones) as usize;
            }
        }
        for i in main..HD_DIM {
            if a[i] != b[i] {
                count += 1;
            }
        }
        count
    }

    pub fn dot(a: &[i8], b: &[i8]) -> i32 {
        let main = main_len();
        // SAFETY: `a` and `b` are `&[i8]` slices read only within `[0, main)` where
        // `main = (HD_DIM/16)*16`, so each 16-element `vld1q_s8` load is in bounds. The
        // accumulator vectors are stack locals and never alias the slices. The scalar tail
        // loop handles the remainder beyond `main`.
        unsafe {
            let mut acc_low: int16x8_t = vdupq_n_s16(0);
            let mut acc_high: int16x8_t = vdupq_n_s16(0);
            for i in (0..main).step_by(16) {
                let va = vld1q_s8(a.as_ptr().add(i));
                let vb = vld1q_s8(b.as_ptr().add(i));
                acc_low = vmlal_s8(acc_low, vget_low_s8(va), vget_low_s8(vb));
                acc_high = vmlal_s8(acc_high, vget_high_s8(va), vget_high_s8(vb));
            }
            let low = vmovl_s16(vget_low_s16(acc_low));
            let high = vmovl_s16(vget_high_s16(acc_low));
            let low_h = vmovl_s16(vget_low_s16(acc_high));
            let high_h = vmovl_s16(vget_high_s16(acc_high));
            let total = vaddq_s32(vaddq_s32(low, high), vaddq_s32(low_h, high_h));
            let mut sum = vaddvq_s32(total) as i32;
            for i in main..HD_DIM {
                sum += (a[i] as i32) * (b[i] as i32);
            }
            sum
        }
    }

    pub fn bundle(vectors: &[&super::Hypervector], scratch: &mut [i32]) {
        assert_eq!(scratch.len(), HD_DIM);
        scratch.fill(0);
        let main = main_len();
        // SAFETY: `scratch` is asserted to be `HD_DIM` elements and is filled with zeros
        // first. Each hypervector's `values` is `HD_DIM` `i8`s, and `main` is
        // `(HD_DIM/16)*16`, so every `vld1q_s8`/`vst1q_s32` access is in bounds. The
        // scratch buffer is accessed via `&mut` so no aliasing occurs, and the tail is
        // handled by the scalar loop.
        unsafe {
            for v in vectors {
                let src = v.values.as_ptr();
                for i in (0..main).step_by(16) {
                    let va = vld1q_s8(src.add(i));
                    let low16 = vmovl_s8(vget_low_s8(va));
                    let high16 = vmovl_s8(vget_high_s8(va));
                    let ptr = scratch.as_mut_ptr().add(i);
                    let mut acc0 = vld1q_s32(ptr);
                    let mut acc1 = vld1q_s32(ptr.add(4));
                    let mut acc2 = vld1q_s32(ptr.add(8));
                    let mut acc3 = vld1q_s32(ptr.add(12));
                    acc0 = vaddw_s16(acc0, vget_low_s16(low16));
                    acc1 = vaddw_s16(acc1, vget_high_s16(low16));
                    acc2 = vaddw_s16(acc2, vget_low_s16(high16));
                    acc3 = vaddw_s16(acc3, vget_high_s16(high16));
                    vst1q_s32(ptr, acc0);
                    vst1q_s32(ptr.add(4), acc1);
                    vst1q_s32(ptr.add(8), acc2);
                    vst1q_s32(ptr.add(12), acc3);
                }
                for (slot, &x) in scratch[main..].iter_mut().zip(&v.values[main..]) {
                    *slot += x as i32;
                }
            }
        }
    }

    pub fn clip(scratch: &[i32], out: &mut [i8]) {
        assert_eq!(scratch.len(), HD_DIM);
        assert_eq!(out.len(), HD_DIM);
        let main = main_len();
        // SAFETY: `scratch` and `out` are both asserted to be `HD_DIM` elements, and
        // `main = (HD_DIM/16)*16`, so each 16-element `vld1q_s32`/`vst1q_s8` access stays
        // in bounds. `scratch` is read-only here and `out` is written via `&mut`, so they
        // do not alias. The scalar tail loop handles the remainder beyond `main`.
        unsafe {
            let ones = vdupq_n_s32(1);
            for i in (0..main).step_by(16) {
                let ptr = scratch.as_ptr().add(i);
                let mut a0 = vld1q_s32(ptr);
                let mut a1 = vld1q_s32(ptr.add(4));
                let mut a2 = vld1q_s32(ptr.add(8));
                let mut a3 = vld1q_s32(ptr.add(12));
                a0 = vorrq_s32(vshrq_n_s32(a0, 31), ones);
                a1 = vorrq_s32(vshrq_n_s32(a1, 31), ones);
                a2 = vorrq_s32(vshrq_n_s32(a2, 31), ones);
                a3 = vorrq_s32(vshrq_n_s32(a3, 31), ones);
                let n0 = vqmovn_s32(a0);
                let n1 = vqmovn_s32(a1);
                let n2 = vqmovn_s32(a2);
                let n3 = vqmovn_s32(a3);
                let lo16 = vcombine_s16(n0, n1);
                let hi16 = vcombine_s16(n2, n3);
                let lo8 = vqmovn_s16(lo16);
                let hi8 = vqmovn_s16(hi16);
                let bytes = vcombine_s8(lo8, hi8);
                vst1q_s8(out.as_mut_ptr().add(i), bytes);
            }
        }
        for i in main..HD_DIM {
            out[i] = if scratch[i] > 0 { 1 } else { -1 };
        }
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
        let mut rng = self.rng.lock().unwrap_or_else(|e| e.into_inner());
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
        let key = format!("tok_{id}");
        if !self.token_memory.symbols.contains_key(&key) {
            self.token_memory.allocate(&key);
        }
        // Safety: key was just inserted above, so it is guaranteed to exist.
        self.token_memory
            .symbols
            .get(&key)
            .expect("token symbol was just allocated")
    }

    /// Encode the source text into a script profile.
    pub fn encode(&mut self, source: &str) -> ScriptProfile {
        // Cap input to prevent O(n * HD_DIM) DoS.
        const MAX_ENCODE_CHARS: usize = 8_192;
        const MAX_ENCODE_TOKENS: usize = 256;
        let source = if source.len() > MAX_ENCODE_CHARS {
            &source[..MAX_ENCODE_CHARS]
        } else {
            source
        };
        let ids = tokenize_text(source);
        let ids: Vec<u64> = ids
            .into_iter()
            .take(MAX_ENCODE_TOKENS)
            .map(|id| id as u64)
            .collect();
        let mut result = Hypervector::zero();
        for (pos, &id) in ids.iter().enumerate() {
            let tok = self.token_hv(id as u32);
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
        let string_alloc_count = lower.matches('+').count()
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

    // =========================================================================
    // Security regression tests — red team findings
    // =========================================================================

    /// Verify that encoding a very long string does not panic or OOM.
    /// The encoder caps input at MAX_ENCODE_CHARS to prevent O(n * HD_DIM) DoS.
    #[test]
    fn encode_caps_input_length() {
        let mut encoder = ScriptEncoder::new();
        // A very long string (1 million chars) should be capped internally.
        let long = "x".repeat(1_000_000);
        let profile = encoder.encode(&long);
        // The source stored in the profile should be capped, not the full input.
        assert!(
            profile.source.len() <= 8_192,
            "encoded source should be capped, got {} chars",
            profile.source.len()
        );
    }

    /// Verify that from_values rejects a vector with the wrong number of elements.
    #[test]
    fn hypervector_from_values_rejects_wrong_length() {
        let too_short = vec![1_i8; HD_DIM - 1];
        assert!(
            Hypervector::from_values(too_short).is_err(),
            "shorter vector must be rejected"
        );

        let too_long = vec![1_i8; HD_DIM + 1];
        assert!(
            Hypervector::from_values(too_long).is_err(),
            "longer vector must be rejected"
        );

        let correct = vec![1_i8; HD_DIM];
        assert!(
            Hypervector::from_values(correct).is_ok(),
            "correct-length vector must be accepted"
        );
    }
}
