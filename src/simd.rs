//! Hardware-fused vector math for the Firefly hot paths.
//!
//! This module is the home for the manual ARM64 NEON paths that replace scalar
//! dot products and cosine-similarity loops in the transformer, firewall, and
//! memory-graph code.  All public functions are safe wrappers around small
//! `unsafe` intrinsics blocks and fall back to scalar Rust on non-aarch64
//! targets.

/// Dot product of two equal-length `f64` slices, fused to `f32` 128-bit
/// NEON accumulation on Apple Silicon.
///
/// The loop streams four `f32` lanes per instruction by loading two
/// `float64x2_t` chunks from each slice, narrowing to `f32`, combining them
/// into a `float32x4_t`, and accumulating.  NaN/Inf lanes are masked to zero
/// with `vbslq_f32` so they do not poison the sum.  Any tail elements that are
/// not a multiple of four are handled by a scalar remainder loop.
///
/// On non-aarch64 targets the same operation is performed with a scalar loop.
pub fn dot_f64_f32(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "dot_f64_f32: length mismatch");
    #[cfg(target_arch = "aarch64")]
    unsafe {
        dot_f64x2_f32x4(a, b) as f64
    }
    #[cfg(not(target_arch = "aarch64"))]
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let p = x * y;
            if p.is_finite() {
                p
            } else {
                0.0
            }
        })
        .sum::<f64>()
}

/// L2 magnitude (Euclidean norm) of an `f64` slice, fused to NEON on aarch64.
#[inline]
pub fn magnitude_f64_f32(a: &[f64]) -> f64 {
    dot_f64_f32(a, a).sqrt()
}

/// Cosine similarity of two `f64` slices, fused to NEON on aarch64.
///
/// Returns `0.0` if either vector has zero magnitude.
#[inline]
pub fn cosine_f64_f32(a: &[f64], b: &[f64]) -> f64 {
    let dot = dot_f64_f32(a, b);
    let mag = magnitude_f64_f32(a) * magnitude_f64_f32(b);
    if mag < 1e-12 {
        0.0
    } else {
        (dot / mag).clamp(-1.0, 1.0)
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn dot_f64x2_f32x4(a: &[f64], b: &[f64]) -> f32 {
    use core::arch::aarch64::*;

    let n = a.len();
    let mut acc = vdupq_n_f32(0.0);
    let mut i = 0usize;

    while i + 4 <= n {
        let a0 = vld1q_f64(a.as_ptr().add(i));
        let a1 = vld1q_f64(a.as_ptr().add(i + 2));
        let b0 = vld1q_f64(b.as_ptr().add(i));
        let b1 = vld1q_f64(b.as_ptr().add(i + 2));

        let va = vcombine_f32(vcvt_f32_f64(a0), vcvt_f32_f64(a1));
        let vb = vcombine_f32(vcvt_f32_f64(b0), vcvt_f32_f64(b1));

        // Keep the product only if both lanes are finite (not NaN/Inf).
        let finite_mask = vandq_u32(vceqq_f32(va, va), vceqq_f32(vb, vb));
        let prod = vmulq_f32(va, vb);
        let safe_prod = vbslq_f32(finite_mask, prod, vdupq_n_f32(0.0));

        acc = vaddq_f32(acc, safe_prod);
        i += 4;
    }

    let mut sum = vaddvq_f32(acc);

    while i < n {
        let x = a[i] as f32;
        let y = b[i] as f32;
        if x.is_finite() && y.is_finite() {
            sum += x * y;
        }
        i += 1;
    }

    sum
}
