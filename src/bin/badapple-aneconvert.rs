//! Bad Apple // badapple-aneconvert — native GGUF-free, Python-free converter.
//!
//! Reads a HuggingFace safetensors checkpoint (e.g. Qwen3-4B) and emits the
//! stateful ANE shard substrate that `BadAppleANEShardCore` drives:
//!
//!   <out>/qwen3b_ane_shards/layer_sXX-YY.mlpackage + .mlmodelc (one or more
//!       transformer layers, KV cache as MLState tensors)
//!   <out>/qwen3b_ane_shards/head_vXX-YY.mlpackage + .mlmodelc (final RMSNorm +
//!       sliced LM head)
//!   <out>/qwen3b_ane_shards/embedding.f16 (mmap'd fp16 [vocab, hidden])
//!   <out>/qwen3b_ane_shards/conversion_manifest.json
//!
//! Compilation goes through Xcode's `coremlc` — no Python in the pipeline.

#[cfg(test)]
use bad_apple::mil_spec::*;
use bad_apple::safetensors::*;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

// ======== safetensors (header-only index + on-demand slices) ========

// ======== model config ========

#[derive(Clone, Copy, PartialEq, Eq)]
enum Arch {
    Qwen3,
    Gemma3Text,
}

#[derive(Clone)]
struct Cfg {
    arch: Arch,
    d_model: i64,
    n_layers: i64,
    n_heads: i64,
    n_kv: i64,
    d_head: i64,
    d_ff: i64,
    vocab: i64,
    rope_theta: f64,
    /// Gemma3 local (sliding-window) layers use a different RoPE base
    /// (`rope_local_base_freq`, 1e4) than global layers (`rope_theta`, 1e6).
    /// Equals rope_theta for single-base architectures.
    rope_local_theta: f64,
    eps: f32,
    rope_dim: i64,
    has_qk_norm: bool,
    tie_embeddings: bool,
    /// Score scale: Qwen3 1/sqrt(d_head); Gemma3 query_pre_attn_scalar^-0.5.
    attn_scale: f32,
    /// Sliding-window width (0 = no windowing). Gemma3 local layers only.
    sliding_window: i64,
    /// Global layer every N: sliding iff (layer+1) % pattern != 0.
    sliding_pattern: i64,
    /// Gemma RMSNorm stores w but applies (1+w) — bake +1 into weights.
    norm_plus_one: bool,
    /// Gemma scales embeddings by sqrt(d_model) — baked into embedding.f16.
    embed_scale: f32,
}

impl Cfg {
    fn from_config(dir: &Path) -> Result<Self, String> {
        let raw = std::fs::read_to_string(dir.join("config.json")).map_err(|e| e.to_string())?;
        let root: Value = serde_json::from_str(&raw).map_err(|e| e.to_string())?;
        let model_type = root["model_type"].as_str().unwrap_or("");
        let arch = match model_type {
            "gemma3" | "gemma3_text" => Arch::Gemma3Text,
            _ => Arch::Qwen3,
        };
        // Gemma3 multimodal checkpoints nest the LM config under text_config.
        let c = if root["text_config"].is_object() {
            &root["text_config"]
        } else {
            &root
        };
        let g = |k: &str| c[k].as_i64().unwrap_or(0);
        let d_model = g("hidden_size");
        let n_heads = g("num_attention_heads");
        let d_head =
            c["head_dim"]
                .as_i64()
                .unwrap_or(if n_heads > 0 { d_model / n_heads } else { 0 });
        let rope_theta = c["rope_theta"].as_f64().unwrap_or(1_000_000.0);
        let rope_local_theta = c["rope_local_base_freq"].as_f64().unwrap_or(rope_theta);
        let attn_scale = c["query_pre_attn_scalar"]
            .as_f64()
            .map(|s| (s as f32).sqrt().recip())
            .unwrap_or_else(|| 1.0 / (d_head as f32).sqrt());
        let sliding_pattern = c["sliding_window_pattern"].as_i64().unwrap_or(0);
        Ok(Cfg {
            arch,
            d_model,
            n_layers: g("num_hidden_layers"),
            n_heads,
            n_kv: g("num_key_value_heads"),
            d_head,
            d_ff: g("intermediate_size"),
            vocab: g("vocab_size"),
            rope_theta,
            rope_local_theta,
            eps: c["rms_norm_eps"].as_f64().unwrap_or(1e-6) as f32,
            rope_dim: d_head,
            // Both Qwen3 and Gemma3 carry per-head q_norm/k_norm.
            has_qk_norm: true,
            tie_embeddings: c["tie_word_embeddings"].as_bool().unwrap_or(false),
            attn_scale,
            sliding_window: c["sliding_window"].as_i64().unwrap_or(0),
            sliding_pattern,
            norm_plus_one: arch == Arch::Gemma3Text,
            embed_scale: if arch == Arch::Gemma3Text {
                (d_model as f32).sqrt()
            } else {
                1.0
            },
        })
    }

    fn arch_name(&self) -> &'static str {
        match self.arch {
            Arch::Qwen3 => "qwen3",
            Arch::Gemma3Text => "gemma3_text",
        }
    }

    /// Two rope input rows (global + local) only when the arch has a
    /// second base frequency — keeps Qwen3 artifacts/inputs untouched.
    #[allow(dead_code)] // used by the cfg(test) parity oracle
    fn rope_rows(&self) -> i64 {
        if std::env::var("BADAPPLE_GEMMA_NO_LOCAL_ROPE").is_ok() {
            return 1;
        }
        if self.rope_local_theta != self.rope_theta {
            2
        } else {
            1
        }
    }

    /// RoPE input row for layer `i`: 1 = local base, 0 = global.
    #[allow(dead_code)] // used by the cfg(test) parity oracle
    fn rope_row(&self, i: i64) -> i64 {
        if self.rope_rows() == 2 && self.is_sliding(i) {
            1
        } else {
            0
        }
    }

    #[allow(dead_code)] // used by the cfg(test) parity oracle
    fn is_sliding(&self, i: i64) -> bool {
        if std::env::var("BADAPPLE_GEMMA_NO_SLIDING").is_ok() {
            return false;
        }
        self.sliding_window > 0 && self.sliding_pattern > 0 && (i + 1) % self.sliding_pattern != 0
    }
}

/// Gemma RMSNorm: out = x̂ · (1 + w) — fold the +1 into stored fp16 weights.

// ======== weight table assembled per layer ========
// WeightBin (in-memory weight.bin v2 builder) comes from mil_spec.

// ======== weight compression ========

/// 16 = FP16 weights (exact). 8 = int8 matmul weights, one fp16 scale per
/// output channel; norms, embeddings and all activations stay FP16.
static WEIGHT_BITS: std::sync::OnceLock<u32> = std::sync::OnceLock::new();

fn weight_bits() -> u32 {
    *WEIGHT_BITS.get().unwrap_or(&16)
}

/// Symmetric per-row int8: scale_r = max|w_r| / 127, q = round(w / scale_r).
/// `fp16` is row-major [rows, cols]. Returns (int8 bytes, fp16 scale bytes).
fn down_prescale() -> f32 {
    down_prescale_for(weight_bits())
}

fn down_prescale_for(bits: u32) -> f32 {
    if bits != 8 {
        return 1.0;
    }
    // 64 measured exact (cos 0.99997 vs CPU FP16 over 12 layers) with headroom
    // for deeper layers; 16–256 all behaved identically.
    std::env::var("BADAPPLE_Q8_DOWN_PRESCALE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(64.0)
}

/// Emit a matmul weight [cout, cin, 1, 1] as fp16 or int8 per `bits`.
#[cfg(test)]
fn weight_const(
    wb: &mut WeightBin,
    blk: &mut Block,
    name: &str,
    fp16: &[u8],
    shape: &[i64],
    bits: u32,
) -> String {
    if bits == 8 {
        let (q, s) = quantize_q8_rows(fp16, shape[0] as usize);
        let mut scale_shape = vec![shape[0]];
        scale_shape.extend(std::iter::repeat_n(1, shape.len() - 1));
        let qoff = wb.put(&format!("{name}_q8"), DType::Int8, shape, &q);
        let soff = wb.put(&format!("{name}_scale"), DType::Fp16, &scale_shape, &s);
        blk.konst_q8(name, WEIGHTS_NAME, qoff, soff, shape)
    } else if bits == 4 {
        let (q, lut) = quantize_q4_rows(fp16, shape[0] as usize);
        let mut lut_shape = vec![shape[0]];
        lut_shape.extend(std::iter::repeat_n(1, shape.len() - 1));
        lut_shape.push(16);
        lut_shape.push(1);
        let qoff = wb.put(&format!("{name}_q4"), DType::Uint4, shape, &q);
        let loff = wb.put(&format!("{name}_lut"), DType::Fp16, &lut_shape, &lut);
        blk.konst_q4(name, WEIGHTS_NAME, qoff, loff, shape)
    } else {
        let off = wb.put(name, DType::Fp16, shape, fp16);
        blk.konst_blob(name, WEIGHTS_NAME, off, DType::Fp16, shape)
    }
}

// ======== graph emitters ========

// iOS19/macOS26 opset (CoreML9 / spec v10): unlocked scaled_dot_product_attention
// and newer op registrations; CoreML8/v9 remains as fallback via env.
// slice_update (the packed-KV write op) fails ANE plan-build -14
// under the ios19/CoreML9 lowering; ios18 plans it correctly — mil_kvpack
// verified identical MIL passes at ios18 and fails at ios19.
#[cfg(test)]
const SPEC_VERSION: i32 = 9;
#[cfg(test)]
const OPSET: &str = "CoreML8";
#[cfg(test)]
const WEIGHTS_NAME: &str = "@model_path/weights/weight.bin";

/// tile a (dh,) weight to (n*dh) elements for conv-broadcast use
/// formulation (attention paths, mil_kvpack).
#[cfg(test)]
fn linear_w(blk: &mut Block, x: &str, w: &str, d: i64, cout: i64, s: i64, name: &str) -> String {
    let xr = blk.reshape(x, &[1, 1, d, s], &format!("{name}_xr"));
    let wr = blk.reshape(w, &[1, 1, cout, d], &format!("{name}_wr"));
    let mm = blk.matmul(&wr, &xr, false, &[1, 1, cout, s], name);
    blk.reshape(&mm, &[1, cout, 1, s], &format!("{name}_b"))
}

/// 1x1 conv whose output spatial width is `s`. `conv1x1` hardcodes width 1;
/// prefill shards (s>1) declare the real shape so MIL type-checks.
/// RMSNorm hardened for massive activations, with a subnormal floor.
///
/// mil_spec's safe-norm prescales x by 1/sqrt(d) before squaring, keeping
/// x^2 inside fp16 — but that only covers |x| < sqrt(65504)*sqrt(d) (~8.7k
/// for d=1152). Gemma3's massive activation channels exceed that: the fp16
/// accumulator saturates, the mean reads low, rsqrt overestimates, and the
/// channel amplifies ~10x per layer until the sum overflows to Inf and the
/// layer collapses to NaN.
///
/// rms_norm is scale-invariant — (x/c)/rms(x/c) == x/rms(x) — so this emits
/// xs = x/(8*sqrt(d)) and eps/(c^2*d), extending safe |x| to ~70k while the
/// math stays identical. A plain x/64 overcorrected: sq fell into fp16
/// subnormal territory and `mean` underflowed to 0 for small-rms inputs,
/// turning rsqrt(0) into +Inf (observed at l4, step 4 — inf, not the
/// earlier NaN). The maximum(var, 1e-7) floor bounds rsqrt <= ~3162 for
/// degenerate near-zero inputs, which are ~0 anyway.
#[cfg(test)]
fn rms_norm_big(
    blk: &mut Block,
    x: &str,
    w: &str,
    d: i64,
    eps: f32,
    shape4: &[i64],
    pfx: &str,
) -> String {
    const C: f32 = 8.0;
    let k = (d as f32).sqrt();
    let inv = blk.konst_f16(&format!("{pfx}_bigik"), 1.0 / (C * k));
    let xs = blk.mul(x, &inv, shape4, &format!("{pfx}_bigxs"));
    let sq = blk.mul(&xs, &xs, shape4, &format!("{pfx}_bigsq"));
    let axes = blk.konst_i32(&format!("{pfx}_bigax"), &[1]);
    let kd = blk.konst_bool(&format!("{pfx}_bigkd"), true);
    let mut mshape = shape4.to_vec();
    mshape[1] = 1;
    let mean = blk.o1(
        "reduce_mean",
        vec![
            ("x".into(), bind(&sq).1),
            ("axes".into(), bind(&axes).1),
            ("keep_dims".into(), bind(&kd).1),
        ],
        &format!("{pfx}_bigm"),
        ValueType::Tensor(TensorType::f16(&mshape)),
    );
    let e = blk.konst_f16(&format!("{pfx}_bige"), eps / (C * C * k * k));
    let vp = blk.add(&mean, &e, &mshape, &format!("{pfx}_bigvp"));
    let fl = blk.konst_f16(&format!("{pfx}_bigfl"), 1e-7);
    let vm = blk.o1(
        "maximum",
        vec![("x".into(), bind(&vp).1), ("y".into(), bind(&fl).1)],
        &format!("{pfx}_bigvm"),
        ValueType::Tensor(TensorType::f16(&mshape)),
    );
    let z = blk.konst_f16(&format!("{pfx}_bigz"), 0.0);
    let r = blk.o1(
        "rsqrt",
        vec![("x".into(), bind(&vm).1), ("epsilon".into(), bind(&z).1)],
        &format!("{pfx}_bigr"),
        ValueType::Tensor(TensorType::f16(&mshape)),
    );
    // Keep mil's naming: the pre-weight normalized tensor is `{pfx}_xn`,
    // the weighted output `{pfx}_out` — the monolith's fold path consumes
    // `{i}_fn_xn` directly.
    let xn = blk.mul(&xs, &r, shape4, &format!("{pfx}_xn"));
    blk.mul(&xn, w, shape4, &format!("{pfx}_out"))
}

#[allow(dead_code)] // used by the cfg(test) parity oracle
#[cfg(test)]
fn conv1x1_w(blk: &mut Block, x: &str, w: &str, cout: i64, s: i64, name: &str) -> String {
    if s == 1 {
        return blk.conv1x1(x, w, None, cout, name);
    }
    let strides = blk.konst_i32(&format!("{name}_cstrides"), &[1, 1]);
    let pad_type = {
        let vt = ValueType::Tensor(TensorType {
            dtype: DType::Str,
            shape: vec![],
        });
        blk.op(
            "const",
            vec![],
            vec![(&format!("{name}_cpadtype"), vt)],
            vec![(
                "val".into(),
                bad_apple::mil_spec::Value::Str("valid".into()),
            )],
        )[0]
        .clone()
    };
    let pads = blk.konst_i32(&format!("{name}_cpad"), &[0, 0, 0, 0]);
    let dil = blk.konst_i32(&format!("{name}_cdil"), &[1, 1]);
    let grp = blk.konst_scalar_i32(&format!("{name}_cgrp"), 1);
    let vt = ValueType::Tensor(TensorType::f16(&[1, cout, 1, s]));
    blk.o1(
        "conv",
        vec![
            ("x".into(), bind(x).1),
            ("weight".into(), bind(w).1),
            ("strides".into(), bind(&strides).1),
            ("pad_type".into(), bind(&pad_type).1),
            ("pad".into(), bind(&pads).1),
            ("dilations".into(), bind(&dil).1),
            ("groups".into(), bind(&grp).1),
        ],
        name,
        vt,
    )
}

/// Cfg → the Json `mil_layer_spec` consumes; the BADAPPLE_* debug knobs
/// fold in here (they were env vars in the Rust emitter, fields in .cy).
fn layer_cfg_json(cfg: &Cfg) -> Value {
    let env_b = |k: &str| std::env::var(k).is_ok();
    let mut j = json!({
        "arch": cfg.arch_name(),
        "d_model": cfg.d_model,
        "n_layers": cfg.n_layers,
        "vocab": cfg.vocab,
        "embed_scale": cfg.embed_scale as f64,
        "n_heads": cfg.n_heads,
        "n_kv": cfg.n_kv,
        "d_head": cfg.d_head,
        "rope_dim": cfg.rope_dim,
        "d_ff": cfg.d_ff,
        "eps": cfg.eps as f64,
        "attn_scale": cfg.attn_scale as f64,
        "weight_bits": weight_bits(),
        "norm_plus_one": cfg.norm_plus_one,
        "tie_embeddings": cfg.tie_embeddings,
        "has_qk_norm": cfg.has_qk_norm,
        "rope_theta": cfg.rope_theta,
        "rope_local_theta": cfg.rope_local_theta,
        "sliding_window": cfg.sliding_window,
        "sliding_pattern": cfg.sliding_pattern,
        "down_prescale": down_prescale(),
        "split_ffn": env_b("BADAPPLE_SPLIT_FFN"),
        "qkv_input": env_b("BADAPPLE_QKV_INPUT"),
        "gu_input": env_b("BADAPPLE_GU_INPUT"),
        "gu_mm": env_b("BADAPPLE_GU_MM"),
        "no_rope": env_b("BADAPPLE_NO_ROPE"),
        "no_qknorm": env_b("BADAPPLE_NO_QKNORM"),
        "ones_attn": env_b("BADAPPLE_ONES_ATTN"),
        "kv_dup": env_b("BADAPPLE_KVDUP"),
        "no_postnorm": env_b("BADAPPLE_NO_POSTNORM"),
        "gemma_silu": env_b("BADAPPLE_GEMMA_SILU"),
        "gemma_no_sliding": env_b("BADAPPLE_GEMMA_NO_SLIDING"),
        "gemma_no_local_rope": env_b("BADAPPLE_GEMMA_NO_LOCAL_ROPE"),
    });
    if let Ok(v) = std::env::var("BADAPPLE_CUT") {
        j["cut_at"] = json!(v);
    }
    if let Ok(v) = std::env::var("BADAPPLE_FFN_CUT") {
        j["ffn_cut"] = json!(v);
    }
    if let Ok(v) = std::env::var("BADAPPLE_STOP_AT") {
        j["stop_at"] = json!(v);
    }
    if let Ok(v) = std::env::var("BADAPPLE_SPLIT_GU") {
        j["split_gu"] = json!(v.parse::<i64>().unwrap_or(1));
    }
    if let Ok(v) = std::env::var("BADAPPLE_SPEC_VERSION") {
        if let Ok(n) = v.parse::<i64>() {
            j["spec_version"] = json!(n);
            j["opset"] = json!(std::env::var("BADAPPLE_OPSET").unwrap_or_else(|_| {
                if n > 9 {
                    "CoreML9".into()
                } else {
                    "CoreML8".into()
                }
            }));
        }
    }
    j
}

/// Emit one stateful transformer-layer shard covering layers [ls, le).
/// `s` is the token width per forward: 1 = decode shard, >1 = prefill shard
/// (same state names/shapes so it shares MLState with the decode shards).
///
/// The pre-port Rust emitter — kept as the byte-parity oracle for
/// `mil_cy_layer_*_parity` tests.
#[cfg(test)]
fn emit_layer_shard_rust(
    st: &StIndex,
    cfg: &Cfg,
    ls: i64,
    le: i64,
    seq: i64,
    out_dir: &Path,
    s: i64,
    is_prefill: bool,
    bits: u32,
) -> Result<(PathBuf, PathBuf, Vec<u8>), String> {
    // Shard names are role-based, not width-based: a --decode-width >1
    // decode shard is still `layer_s..` (the bridge's decode lookup),
    // while a bulk-priming shard is always `prefill_s..`.
    let name = if is_prefill {
        format!("prefill_s{:02}-{:02}", ls, le)
    } else {
        format!("layer_s{:02}-{:02}", ls, le)
    };
    let pkg_dir = out_dir.join(format!("{name}.mlpackage"));

    let mut blk = Block::new();
    let mut wb = WeightBin::new();
    let d = cfg.d_model;
    let nh = cfg.n_heads;
    let nkv = cfg.n_kv;
    let dh = cfg.d_head;
    let rope_dim = cfg.rope_dim;
    let rope_half = rope_dim / 2;
    let q_dim = nh * dh;
    let kv_dim = nkv * dh;
    let hpk = nh / nkv;
    let _scale = 1.0f32 / (dh as f32).sqrt();

    // Blob weight constants
    let blob =
        |wb: &mut WeightBin, blk: &mut Block, name: &str, bytes: &[u8], shape: &[i64]| -> String {
            let off = wb.put(name, DType::Fp16, shape, bytes);
            blk.konst_blob(name, WEIGHTS_NAME, off, DType::Fp16, shape)
        };

    let mut cur = "x".to_string();
    let shape_x = vec![1, d, 1, s];
    let n_state = le - ls;
    let split_ffn = std::env::var("BADAPPLE_SPLIT_FFN").is_ok();
    // Packed KV state: ONE buffer per shard [2*nl, nkv, seq, dh], rows
    // 2l (K) and 2l+1 (V), chained through every layer's slice_update and
    // committed once per call. One state object total (the documented
    // recipe) — and K^T for attention comes from matmul's fused
    // transpose_y, because a standalone transpose of a
    // slice_update-derived state slice fails ANE plan-build -14
    // (verified by mil_kvpack).
    let kvshape: Vec<i64> = vec![2 * n_state, nkv, seq, dh];
    let mut cur_kv = blk.read_state("kv_cache", &kvshape, "kv_init");

    // Gemma stores RMSNorm weights w but applies (1+w); fold the +1 in
    // at emission so the in-graph mul stays identical to Qwen3's.
    let norm_w = |st: &StIndex, name: &str| -> Result<Vec<u8>, String> {
        let raw = st.get_f16(name)?;
        Ok(if cfg.norm_plus_one {
            f16_plus_one(&raw)
        } else {
            raw
        })
    };

    // BADAPPLE_CUT=<stage> truncates emission mid-layer (single-layer
    // shards only) for plan-build bisection: `cur` becomes the named
    // intermediate and the layer loop exits before later stages emit.
    let cut_at = std::env::var("BADAPPLE_CUT").ok();
    let mut cut_shape: Option<Vec<i64>> = None;
    // Values kept alive by declaring them as extra model outputs: a dead
    // rope/slice value is itself an ANE plan-build (-14) suspect, so each
    // cut can nominate the graph values the truncated tail would have used.
    let mut cut_extra: Vec<(String, Vec<i64>)> = Vec::new();
    macro_rules! cut {
        ($stage:expr, $val:expr, $shape:expr $(; $($kv:expr => $kshape:expr),*)?) => {
            if cut_at.as_deref() == Some($stage) && le - ls == 1 {
                let z = blk.konst_f16(&format!("cutz_{}", $stage), 0.0);
                cur = blk.add(&$val, &z, &$shape, "hidden");
                cut_shape = Some($shape.to_vec());
                $($( cut_extra.push(($kv.clone(), $kshape.to_vec())); )*)?
                break;
            }
        };
    }

    for i in ls..le {
        let p = format!("model.layers.{i}");

        // ---- rope input row select ----
        // rope_cos/rope_sin arrive [R, half] (s==1) or [R,1,half,s] (s>1):
        // row 0 global base, row 1 local base (Gemma3 sliding layers).
        let rope_row = cfg.rope_row(i);
        let (cos_in, sin_in) = if cfg.rope_rows() == 2 {
            let (cs, ss) = if s == 1 {
                (
                    blk.slice(
                        "rope_cos",
                        &[rope_row as i32, 0],
                        &[rope_row as i32 + 1, rope_half as i32],
                        &[1, rope_half],
                        &format!("l{i}_cosr"),
                    ),
                    blk.slice(
                        "rope_sin",
                        &[rope_row as i32, 0],
                        &[rope_row as i32 + 1, rope_half as i32],
                        &[1, rope_half],
                        &format!("l{i}_sinr"),
                    ),
                )
            } else {
                (
                    blk.slice(
                        "rope_cos",
                        &[rope_row as i32, 0, 0, 0],
                        &[rope_row as i32 + 1, 1, rope_half as i32, s as i32],
                        &[1, 1, rope_half, s],
                        &format!("l{i}_cosr"),
                    ),
                    blk.slice(
                        "rope_sin",
                        &[rope_row as i32, 0, 0, 0],
                        &[rope_row as i32 + 1, 1, rope_half as i32, s as i32],
                        &[1, 1, rope_half, s],
                        &format!("l{i}_sinr"),
                    ),
                )
            };
            (cs, ss)
        } else {
            ("rope_cos".to_string(), "rope_sin".to_string())
        };

        // ---- attn_norm ----
        let w = norm_w(st, &format!("{p}.input_layernorm.weight"))?;
        let wn = blob(
            &mut wb,
            &mut blk,
            &format!("w_attn_norm_{i}"),
            &w,
            &[d, 1, 1],
        );
        let normed = rms_norm_big(
            &mut blk,
            &cur,
            &wn,
            d,
            cfg.eps,
            &shape_x,
            &format!("l{i}_an"),
        );
        cut!("an", normed, [1, d, 1, s]);

        // ---- fused QKV conv ----
        let qw = st.get_f16(&format!("{p}.self_attn.q_proj.weight"))?;
        let kw = st.get_f16(&format!("{p}.self_attn.k_proj.weight"))?;
        let vw = st.get_f16(&format!("{p}.self_attn.v_proj.weight"))?;
        let mut qkv_w = qw;
        qkv_w.extend_from_slice(&kw);
        qkv_w.extend_from_slice(&vw);
        let wqkv = weight_const(
            &mut wb,
            &mut blk,
            &format!("w_qkv_{i}"),
            &qkv_w,
            &[q_dim + 2 * kv_dim, d, 1, 1],
            bits,
        );
        // BADAPPLE_QKV_INPUT (debug): declare `qkv` as a raw input instead of
        // the weight conv — isolates conv/state-update plan interactions.
        // The projection runs as a matmul, not a conv: a `conv` op upstream
        // of the chained slice_update pair fails ANE plan-build (-14) on
        // newer CoreML, while the identical matmul plans clean (bisected:
        // conv+1 update passes, conv+2 fails; matmul+2 passes).
        // The projection runs as a matmul at s==1 (a `conv` upstream of the
        // chained slice_update pair fails ANE plan-build at s==1) but as a
        // conv at s>1 — the reverse pairing, where the s>1 matmul chain is
        // the one that fails plan-build (-14).
        let qkv = if std::env::var("BADAPPLE_QKV_INPUT").is_ok() {
            "qkv".to_string()
        } else if s == 1 {
            linear_w(
                &mut blk,
                &normed,
                &wqkv,
                d,
                q_dim + 2 * kv_dim,
                s,
                &format!("l{i}_qkv"),
            )
        } else {
            conv1x1_w(
                &mut blk,
                &normed,
                &wqkv,
                q_dim + 2 * kv_dim,
                s,
                &format!("l{i}_qkv"),
            )
        };

        let q = blk.slice(
            &qkv,
            &[0, 0, 0, 0],
            &[1, q_dim as i32, 1, s as i32],
            &[1, q_dim, 1, s],
            &format!("l{i}_q"),
        );
        let k = blk.slice(
            &qkv,
            &[0, q_dim as i32, 0, 0],
            &[1, (q_dim + kv_dim) as i32, 1, s as i32],
            &[1, kv_dim, 1, s],
            &format!("l{i}_k"),
        );
        let v = blk.slice(
            &qkv,
            &[0, (q_dim + kv_dim) as i32, 0, 0],
            &[1, (q_dim + 2 * kv_dim) as i32, 1, s as i32],
            &[1, kv_dim, 1, s],
            &format!("l{i}_v"),
        );
        cut!("qkv", qkv, [1, q_dim + 2 * kv_dim, 1, s]);

        // ---- per-head q/k RMSNorm (Qwen3/Gemma3) ----
        let (qn, kn) = if cfg.has_qk_norm && std::env::var("BADAPPLE_NO_QKNORM").is_err() {
            let qnw = norm_w(st, &format!("{p}.self_attn.q_norm.weight"))?;
            let knw = norm_w(st, &format!("{p}.self_attn.k_norm.weight"))?;
            let qn_tiled = tile_head_w(&qnw, nh);
            let kn_tiled = tile_head_w(&knw, nkv);
            let qn = blob(
                &mut wb,
                &mut blk,
                &format!("w_qn_{i}"),
                &qn_tiled,
                &[q_dim, 1, 1],
            );
            let kn = blob(
                &mut wb,
                &mut blk,
                &format!("w_kn_{i}"),
                &kn_tiled,
                &[kv_dim, 1, 1],
            );
            // reshape (1,nh*dh,1,s) -> (nh,dh,1,s); rms over axis1; back; mul w
            let qr = blk.reshape(&q, &[nh, dh, 1, s], &format!("l{i}_qh"));
            let qshape = vec![nh, dh, 1, s];
            // safe-norm per head over dh (axis 1)
            let qn_out = head_rms(&mut blk, &qr, dh, cfg.eps, &qshape, &format!("l{i}_qn"));
            let qb = blk.reshape(&qn_out, &[1, q_dim, 1, s], &format!("l{i}_qnb"));
            let qo = blk.mul(&qb, &qn, &[1, q_dim, 1, s], &format!("l{i}_qo"));

            let kr = blk.reshape(&k, &[nkv, dh, 1, s], &format!("l{i}_kh"));
            let kshape = vec![nkv, dh, 1, s];
            let kn_out = head_rms(&mut blk, &kr, dh, cfg.eps, &kshape, &format!("l{i}_kn"));
            let kb = blk.reshape(&kn_out, &[1, kv_dim, 1, s], &format!("l{i}_knb"));
            let ko = blk.mul(&kb, &kn, &[1, kv_dim, 1, s], &format!("l{i}_ko"));
            (qo, ko)
        } else {
            (q, k)
        };
        cut!("qn", qn, [1, q_dim, 1, s]);

        // ---- RoPE (host supplies rope_cos/rope_sin (1, rope_half)) ----
        let rope =
            |blk: &mut Block, xf: &str, n_h: i64, dim: i64, total: i64, tag: &str| -> String {
                let xr = blk.reshape(xf, &[1, n_h, dim], &format!("{tag}_r"));
                // slice rotated dims [0..rope_dim) and pass-through [rope_dim..dim)
                let rot_end = rope_dim as i32;
                let has_pass = rope_dim < dim;
                let x_rot = if has_pass {
                    blk.slice(
                        &xr,
                        &[0, 0, 0],
                        &[1, n_h as i32, rot_end],
                        &[1, n_h, rope_dim],
                        &format!("{tag}_rot"),
                    )
                } else {
                    xr.clone()
                };
                // empty pass-through: skip entirely — a zero-width slice is illegal in MIL
                let x_pass = if has_pass {
                    blk.slice(
                        &xr,
                        &[0, 0, rot_end],
                        &[1, n_h as i32, dim as i32],
                        &[1, n_h, dim - rope_dim],
                        &format!("{tag}_pass"),
                    )
                } else {
                    String::new()
                };
                let x_lo = blk.slice(
                    &x_rot,
                    &[0, 0, 0],
                    &[1, n_h as i32, rope_half as i32],
                    &[1, n_h, rope_half],
                    &format!("{tag}_lo"),
                );
                let x_hi = blk.slice(
                    &x_rot,
                    &[0, 0, rope_half as i32],
                    &[1, n_h as i32, rot_end],
                    &[1, n_h, rope_half],
                    &format!("{tag}_hi"),
                );
                let cos_b = blk.reshape(&cos_in, &[1, 1, rope_half], &format!("{tag}_cos"));
                let sin_b = blk.reshape(&sin_in, &[1, 1, rope_half], &format!("{tag}_sin"));
                let a = blk.mul(&x_lo, &cos_b, &[1, n_h, rope_half], &format!("{tag}_a"));
                let b = blk.mul(&x_hi, &sin_b, &[1, n_h, rope_half], &format!("{tag}_b"));
                let r_lo = blk.sub(&a, &b, &[1, n_h, rope_half], &format!("{tag}_rlo"));
                let c = blk.mul(&x_lo, &sin_b, &[1, n_h, rope_half], &format!("{tag}_c"));
                let d2 = blk.mul(&x_hi, &cos_b, &[1, n_h, rope_half], &format!("{tag}_d"));
                let r_hi = blk.add(&c, &d2, &[1, n_h, rope_half], &format!("{tag}_rhi"));
                let cat = if has_pass {
                    blk.concat(
                        &[r_lo, r_hi, x_pass],
                        -1,
                        &[1, n_h, dim],
                        &format!("{tag}_cat"),
                    )
                } else {
                    blk.concat(&[r_lo, r_hi], -1, &[1, n_h, dim], &format!("{tag}_cat"))
                };
                blk.reshape(&cat, &[1, total, 1, 1], &format!("{tag}_flat"))
            };
        // RoPE for s>1: rank-4 layout, x [1,n_h,dim,P], cos/sin already
        // [1,1,rope_half,P]; slice rot/pass/lo/hi along axis 2.
        let rope_p =
            |blk: &mut Block, xf: &str, n_h: i64, dim: i64, total: i64, tag: &str| -> String {
                let xr = blk.reshape(xf, &[1, n_h, dim, s], &format!("{tag}_r"));
                let rot_end = rope_dim as i32;
                let has_pass = rope_dim < dim;
                let x_rot = if has_pass {
                    blk.slice(
                        &xr,
                        &[0, 0, 0, 0],
                        &[1, n_h as i32, rot_end, s as i32],
                        &[1, n_h, rope_dim, s],
                        &format!("{tag}_rot"),
                    )
                } else {
                    xr.clone()
                };
                let x_pass = if has_pass {
                    blk.slice(
                        &xr,
                        &[0, 0, rot_end, 0],
                        &[1, n_h as i32, dim as i32, s as i32],
                        &[1, n_h, dim - rope_dim, s],
                        &format!("{tag}_pass"),
                    )
                } else {
                    String::new()
                };
                let x_lo = blk.slice(
                    &x_rot,
                    &[0, 0, 0, 0],
                    &[1, n_h as i32, rope_half as i32, s as i32],
                    &[1, n_h, rope_half, s],
                    &format!("{tag}_lo"),
                );
                let x_hi = blk.slice(
                    &x_rot,
                    &[0, 0, rope_half as i32, 0],
                    &[1, n_h as i32, rot_end, s as i32],
                    &[1, n_h, rope_half, s],
                    &format!("{tag}_hi"),
                );
                let a = blk.mul(&x_lo, &cos_in, &[1, n_h, rope_half, s], &format!("{tag}_a"));
                let b = blk.mul(&x_hi, &sin_in, &[1, n_h, rope_half, s], &format!("{tag}_b"));
                let r_lo = blk.sub(&a, &b, &[1, n_h, rope_half, s], &format!("{tag}_rlo"));
                let c = blk.mul(&x_lo, &sin_in, &[1, n_h, rope_half, s], &format!("{tag}_c"));
                let d2 = blk.mul(&x_hi, &cos_in, &[1, n_h, rope_half, s], &format!("{tag}_d"));
                let r_hi = blk.add(&c, &d2, &[1, n_h, rope_half, s], &format!("{tag}_rhi"));
                let cat = if has_pass {
                    blk.concat(
                        &[r_lo, r_hi, x_pass],
                        2,
                        &[1, n_h, dim, s],
                        &format!("{tag}_cat"),
                    )
                } else {
                    blk.concat(&[r_lo, r_hi], 2, &[1, n_h, dim, s], &format!("{tag}_cat"))
                };
                blk.reshape(&cat, &[1, total, 1, s], &format!("{tag}_flat"))
            };
        let skip_rope = std::env::var("BADAPPLE_NO_ROPE").is_ok();
        let (q_r, k_r) = if skip_rope {
            (qn.clone(), kn.clone())
        } else if s == 1 {
            (
                rope(&mut blk, &qn, nh, dh, q_dim, &format!("l{i}_rq")),
                rope(&mut blk, &kn, nkv, dh, kv_dim, &format!("l{i}_rk")),
            )
        } else {
            (
                rope_p(&mut blk, &qn, nh, dh, q_dim, &format!("l{i}_rq")),
                rope_p(&mut blk, &kn, nkv, dh, kv_dim, &format!("l{i}_rk")),
            )
        };

        cut!("rq", q_r, [1, q_dim, 1, s]);
        let (new_k, new_v) = if s == 1 {
            (
                blk.reshape(&k_r, &[1, nkv, 1, dh], &format!("l{i}_nk")),
                blk.reshape(&v, &[1, nkv, 1, dh], &format!("l{i}_nv")),
            )
        } else {
            // [1,kv_dim,1,P] -> [1,nkv,dh,P] -> transpose [0,1,3,2] -> [1,nkv,P,dh]
            let kr = blk.reshape(&k_r, &[1, nkv, dh, s], &format!("l{i}_nkr"));
            let kt = blk.transpose(&kr, &[0, 1, 3, 2], &[1, nkv, s, dh], &format!("l{i}_nk"));
            let vr = blk.reshape(&v, &[1, nkv, dh, s], &format!("l{i}_nvr"));
            let vt2 = blk.transpose(&vr, &[0, 1, 3, 2], &[1, nkv, s, dh], &format!("l{i}_nv"));
            (kt, vt2)
        };

        // ---- packed-KV state update ----
        // `kv_cache` [2*nl,nkv,seq,dh]; layer l owns rows 2l (K) and
        // 2l+1 (V). Two write paths, one per token width — each is the
        // only variant that plan-builds for its s (verified by bisect):
        //   s==1 (decode): pure recipe — runtime `pos` drives begin/end;
        //     slice_update writes ONLY the new KV [1,nkv,s,dh]; attention
        //     re-slices the full row from the chained buffer.
        //   s>1 (prefill): hybrid — the row is masked-merged in-graph
        //     (kv_write_mask encodes the write position), then written
        //     back with a STATIC-bounds slice_update; attention reads the
        //     in-graph merged row.
        // Runtime-pos updates at s>1 and static full-row write-backs at
        // s==1 each fail plan-build -14 under every spec version.
        let local = i - ls;
        let krow = 2 * local as i32;
        let (k_attn, v_attn) = if true {
            let pos_i32 = "pos".to_string();
            let pos_s = {
                let c = blk.konst_i32(&format!("l{i}_psc"), &[s as i32]);
                blk.o1(
                    "add",
                    vec![("x".into(), bind(&pos_i32).1), ("y".into(), bind(&c).1)],
                    &format!("l{i}_ps"),
                    ValueType::Tensor(TensorType::i32v(1)),
                )
            };
            let concat4 = |blk: &mut Block, pfx: &str, vals: &[&str]| -> String {
                let ax = blk.konst_scalar_i32(&format!("{pfx}_ax"), 0);
                let il = blk.konst_bool(&format!("{pfx}_il"), false);
                blk.o1(
                    "concat",
                    vec![
                        ("values".into(), bind_many(vals)),
                        ("axis".into(), bind(&ax).1),
                        ("interleave".into(), bind(&il).1),
                    ],
                    pfx,
                    ValueType::Tensor(TensorType::i32v(4)),
                )
            };
            let c0 = blk.konst_i32(&format!("l{i}_c0"), &[0]);
            let ck = blk.konst_i32(&format!("l{i}_ck"), &[krow]);
            let ck1 = blk.konst_i32(&format!("l{i}_ck1"), &[krow + 1]);
            let cv = blk.konst_i32(&format!("l{i}_cv"), &[krow + 2]);
            let cnkv = blk.konst_i32(&format!("l{i}_cnkv"), &[nkv as i32]);
            let cdh = blk.konst_i32(&format!("l{i}_cdh"), &[dh as i32]);
            // Per-update mask/stride consts: reusing one const across chained
            // slice_updates fails ANE plan-build on newer OS releases.
            let mk_masks = |blk: &mut Block, pfx: &str| -> (String, String) {
                let st = blk.konst_i32(&format!("{pfx}_st"), &[1, 1, 1, 1]);
                let bm = blk.op(
                    "const",
                    vec![],
                    vec![(
                        &format!("{pfx}_bm"),
                        ValueType::Tensor(TensorType {
                            dtype: DType::Bool,
                            shape: vec![4],
                        }),
                    )],
                    vec![(
                        "val".into(),
                        mil_spec::Value::bools(&[false, false, false, false]),
                    )],
                )[0]
                .clone();
                (st, bm)
            };
            let (kst, kbm) = mk_masks(&mut blk, &format!("l{i}_km"));
            let (vst, vbm) = mk_masks(&mut blk, &format!("l{i}_vm"));
            // K row: write [1,nkv,s,dh] at [2l,0,pos,0]..[2l+1,nkv,pos+s,dh]
            let kb = concat4(
                &mut blk,
                &format!("l{i}_kb"),
                &[ck.as_str(), c0.as_str(), pos_i32.as_str(), c0.as_str()],
            );
            let ke = concat4(
                &mut blk,
                &format!("l{i}_ke"),
                &[ck1.as_str(), cnkv.as_str(), pos_s.as_str(), cdh.as_str()],
            );
            cur_kv = blk.o1(
                "slice_update",
                vec![
                    ("x".into(), bind(&cur_kv).1),
                    ("update".into(), bind(&new_k).1),
                    ("begin".into(), bind(&kb).1),
                    ("end".into(), bind(&ke).1),
                    ("stride".into(), bind(&kst).1),
                    ("begin_mask".into(), bind(&kbm).1),
                    ("end_mask".into(), bind(&kbm).1),
                    ("squeeze_mask".into(), bind(&kbm).1),
                ],
                &format!("l{i}_kup"),
                ValueType::Tensor(TensorType::f16(&kvshape)),
            );
            cut!(
                "kupd",
                cur_kv,
                [kvshape[0], kvshape[1], kvshape[2], kvshape[3]]
            );
            // V row: write [1,nkv,s,dh] at [2l+1,0,pos,0]..[2l+2,nkv,pos+s,dh]
            // BADAPPLE_KVDUP (debug): write the K row twice instead — splits
            // "second chained update fails" from "row-1 update fails".
            let dup = std::env::var("BADAPPLE_KVDUP").is_ok();
            let vbs: [&str; 4] = if dup {
                [ck.as_str(), c0.as_str(), pos_i32.as_str(), c0.as_str()]
            } else {
                [ck1.as_str(), c0.as_str(), pos_i32.as_str(), c0.as_str()]
            };
            let vb = concat4(&mut blk, &format!("l{i}_vb"), &vbs);
            let ves: [&str; 4] = if dup {
                [ck1.as_str(), cnkv.as_str(), pos_s.as_str(), cdh.as_str()]
            } else {
                [cv.as_str(), cnkv.as_str(), pos_s.as_str(), cdh.as_str()]
            };
            let ve = concat4(&mut blk, &format!("l{i}_ve"), &ves);
            cur_kv = blk.o1(
                "slice_update",
                vec![
                    ("x".into(), bind(&cur_kv).1),
                    ("update".into(), bind(if dup { &new_k } else { &new_v }).1),
                    ("begin".into(), bind(&vb).1),
                    ("end".into(), bind(&ve).1),
                    ("stride".into(), bind(&vst).1),
                    ("begin_mask".into(), bind(&vbm).1),
                    ("end_mask".into(), bind(&vbm).1),
                    ("squeeze_mask".into(), bind(&vbm).1),
                ],
                &format!("l{i}_vup"),
                ValueType::Tensor(TensorType::f16(&kvshape)),
            );
            cut!("vupd", cur_kv, [kvshape[0], kvshape[1], kvshape[2], kvshape[3]]; q_r => [1, q_dim, 1, s]);
            let k_attn = blk.slice(
                &cur_kv,
                &[krow, 0, 0, 0],
                &[krow + 1, nkv as i32, seq as i32, dh as i32],
                &[1, nkv, seq, dh],
                &format!("l{i}_kfull"),
            );
            let v_attn = blk.slice(
                &cur_kv,
                &[krow + 1, 0, 0, 0],
                &[krow + 2, nkv as i32, seq as i32, dh as i32],
                &[1, nkv, seq, dh],
                &format!("l{i}_vfull"),
            );
            cut!("kf", k_attn, [1, nkv, seq, dh]);
            (k_attn, v_attn)
        } else {
            // Prefill hybrid: kv_write_mask [1,1,seq,s] encodes the write
            // position; merge the new KV into the sliced row in-graph,
            // then write the full row back with static bounds.
            let one = blk.konst_f16(&format!("l{i}_one"), 1.0);
            let ax = blk.konst_i32(&format!("l{i}_wax"), &[3]);
            let kd = blk.konst_bool(&format!("l{i}_wkd"), true);
            let wsum = blk.o1(
                "reduce_sum",
                vec![
                    ("x".into(), bind("kv_write_mask").1),
                    ("axes".into(), bind(&ax).1),
                    ("keep_dims".into(), bind(&kd).1),
                ],
                &format!("l{i}_wsum"),
                ValueType::Tensor(TensorType::f16(&[1, 1, seq, 1])),
            );
            let one_m = blk.sub(&one, &wsum, &[1, 1, seq, 1], &format!("l{i}_om"));
            let kshape = vec![1, nkv, seq, dh];
            let k_old = blk.slice(
                &cur_kv,
                &[krow, 0, 0, 0],
                &[krow + 1, nkv as i32, seq as i32, dh as i32],
                &kshape,
                &format!("l{i}_kold"),
            );
            let k_keep = blk.mul(&k_old, &one_m, &kshape, &format!("l{i}_kkeep"));
            // [1,1,seq,P] x [1,nkv,P,dh] -> [1,nkv,seq,dh] (batch broadcast).
            let k_new = blk.matmul(
                "kv_write_mask",
                &new_k,
                false,
                &kshape,
                &format!("l{i}_knew"),
            );
            let k_full = blk.add(&k_keep, &k_new, &kshape, &format!("l{i}_kfull"));
            let v_old = blk.slice(
                &cur_kv,
                &[krow + 1, 0, 0, 0],
                &[krow + 2, nkv as i32, seq as i32, dh as i32],
                &kshape,
                &format!("l{i}_vold"),
            );
            let v_keep = blk.mul(&v_old, &one_m, &kshape, &format!("l{i}_vkeep"));
            let v_new = blk.matmul(
                "kv_write_mask",
                &new_v,
                false,
                &kshape,
                &format!("l{i}_vnew"),
            );
            let v_full = blk.add(&v_keep, &v_new, &kshape, &format!("l{i}_vfull"));
            let st4 = blk.konst_i32(&format!("l{i}_st4"), &[1, 1, 1, 1]);
            let bm4 = blk.op(
                "const",
                vec![],
                vec![(
                    &format!("l{i}_bm4"),
                    ValueType::Tensor(TensorType {
                        dtype: DType::Bool,
                        shape: vec![4],
                    }),
                )],
                vec![(
                    "val".into(),
                    mil_spec::Value::bools(&[false, false, false, false]),
                )],
            )[0]
            .clone();
            let kb4 = blk.konst_i32(&format!("l{i}_kb4"), &[krow, 0, 0, 0]);
            let ke4 = blk.konst_i32(
                &format!("l{i}_ke4"),
                &[krow + 1, nkv as i32, seq as i32, dh as i32],
            );
            cur_kv = blk.o1(
                "slice_update",
                vec![
                    ("x".into(), bind(&cur_kv).1),
                    ("update".into(), bind(&k_full).1),
                    ("begin".into(), bind(&kb4).1),
                    ("end".into(), bind(&ke4).1),
                    ("stride".into(), bind(&st4).1),
                    ("begin_mask".into(), bind(&bm4).1),
                    ("end_mask".into(), bind(&bm4).1),
                    ("squeeze_mask".into(), bind(&bm4).1),
                ],
                &format!("l{i}_kup"),
                ValueType::Tensor(TensorType::f16(&kvshape)),
            );
            let vb4 = blk.konst_i32(&format!("l{i}_vb4"), &[krow + 1, 0, 0, 0]);
            let ve4 = blk.konst_i32(
                &format!("l{i}_ve4"),
                &[krow + 2, nkv as i32, seq as i32, dh as i32],
            );
            cur_kv = blk.o1(
                "slice_update",
                vec![
                    ("x".into(), bind(&cur_kv).1),
                    ("update".into(), bind(&v_full).1),
                    ("begin".into(), bind(&vb4).1),
                    ("end".into(), bind(&ve4).1),
                    ("stride".into(), bind(&st4).1),
                    ("begin_mask".into(), bind(&bm4).1),
                    ("end_mask".into(), bind(&bm4).1),
                    ("squeeze_mask".into(), bind(&bm4).1),
                ],
                &format!("l{i}_vup"),
                ValueType::Tensor(TensorType::f16(&kvshape)),
            );
            (k_full, v_full)
        };
        cut!("kv", k_attn, [1, nkv, seq, dh]);
        // ---- explicit attention (matmul·scale·+mask·softmax·matmul) ----
        // The fused scaled_dot_product_attention op is numerically unstable
        // on the folded GQA multi-query shape q [1,nkv,hpk,dh]: across model
        // instances the ANE plan alternately produces correct output,
        // suppressed activations, or outright inf channels (verified against
        // MLX ground truth — the prefill/explicit shard reproduces the 6144
        // slot-0 massive activation at channel 35, fused SDPA does not).
        // It also silently DROPS a non-broadcast mask for multi-query shapes
        // (verified: hidden identical under an all -10000 mask). Materialize
        // scores + softmax explicitly for every token width s.
        let attn4 = {
            // P queries folded into the query axis: row r = s_idx*hpk + j.
            // q [1,q_dim,1,P] -> [nkv,hpk*dh,1,P] -> [nkv,P,1,hpk*dh]
            //   -> [1,nkv,P*hpk,dh]; mask [1,1,hpk*P,seq] broadcasts.
            // For s==1 the fold is a single reshape — the transpose chain
            // degenerates to permuting size-1 axes, which the ANE compiler
            // miscompiles (produces amplified/inf attention outputs while
            // the identical math at s>=2 is exact).
            // For s==1 the [1,nkv,hpk,dh] attention shape (dim2=hpk=2) is
            // numerically miscompiled — identical MIL math produces correct
            // output at s>=2 (dim2>=4) but amplified/inf results at s==1
            // across three softmax formulations. Pad dim2 to 4 rows by
            // duplicating the queries (harmless extra compute), attend at
            // the proven s=2 shape, and slice the real rows back out.
            let rows = if s == 1 { 2 * hpk } else { s * hpk };
            let qf = if s == 1 {
                let q2 = blk.reshape(&q_r, &[1, nkv, hpk, dh], &format!("l{i}_qfold2"));
                blk.concat(
                    &[q2.clone(), q2],
                    2,
                    &[1, nkv, 2 * hpk, dh],
                    &format!("l{i}_qfold"),
                )
            } else {
                let q4 = blk.reshape(&q_r, &[nkv, hpk * dh, 1, s], &format!("l{i}_qheads"));
                let qt = blk.transpose(
                    &q4,
                    &[0, 3, 2, 1],
                    &[nkv, s, 1, hpk * dh],
                    &format!("l{i}_qt"),
                );
                blk.reshape(&qt, &[1, nkv, s * hpk, dh], &format!("l{i}_qfold"))
            };
            // Fused SDPA silently DROPS a non-broadcast mask for multi-query
            // shapes (verified: hidden identical under an all -10000 mask).
            // Materialize scores + softmax explicitly so the mask is real.
            // K^T comes from matmul's fused transpose_y — a standalone
            // transpose of a slice_update-derived slice fails plan-build
            // -14 (mil_kvpack verified the fused variant loads clean).
            let scores = blk.matmul(
                &qf,
                &k_attn,
                true,
                &[1, nkv, rows, seq],
                &format!("l{i}_sc"),
            );
            let scl = blk.konst_f16(&format!("l{i}_scl"), cfg.attn_scale);
            let sc = blk.mul(&scores, &scl, &[1, nkv, rows, seq], &format!("l{i}_scm"));
            cut!("sc", sc, [1, nkv, rows, seq]);
            let m0 = blk.add(&sc, "attn_mask", &[1, nkv, rows, seq], &format!("l{i}_msk"));
            // Gemma3 sliding-window layers additionally mask keys older than
            // q - (w-1). The band mask arrives from the bridge as a second
            // input (`attn_mask_slide`); the in-graph i32 compare/cast chain
            // it replaced failed ANE plan-build (-14) on the full layer.
            let masked = if cfg.is_sliding(i) {
                blk.add(
                    &m0,
                    "attn_mask_slide",
                    &[1, nkv, rows, seq],
                    &format!("l{i}_mskw"),
                )
            } else {
                m0
            };
            let probs = blk.softmax(&masked, 3, &[1, nkv, rows, seq], &format!("l{i}_pb"), false);
            cut!("pb", probs, [1, nkv, rows, seq]);
            // BADAPPLE_ONES_ATTN=1 (debug): swap the V operand for all-ones
            // (v_attn*0+1 keeps it live) so attn4 emits the softmax ROW SUMS
            // — a normalized softmax yields exactly 1.0.
            let v_op = if s == 1 && std::env::var("BADAPPLE_ONES_ATTN").is_ok() {
                let vzc = blk.konst_f16(&format!("l{i}_vzc"), 0.0);
                let vz = blk.mul(&v_attn, &vzc, &[1, nkv, seq, dh], &format!("l{i}_vz"));
                let voc = blk.konst_f16(&format!("l{i}_voc"), 1.0);
                blk.add(&vz, &voc, &[1, nkv, seq, dh], &format!("l{i}_vones"))
            } else {
                v_attn.clone()
            };
            let attn = blk.matmul(
                &probs,
                &v_op,
                false,
                &[1, nkv, rows, dh],
                &format!("l{i}_sdpa"),
            );
            cut!("attn", attn, [1, nkv, rows, dh]);
            if s == 1 {
                // slice the real hpk rows out of the padded result, then
                // unfold [1,nkv,hpk,dh] -> [1,q_dim,1,1] with one reshape.
                let slim = blk.slice(
                    &attn,
                    &[0, 0, 0, 0],
                    &[1, nkv as i32, hpk as i32, dh as i32],
                    &[1, nkv, hpk, dh],
                    &format!("l{i}_aslice"),
                );
                blk.reshape(&slim, &[1, q_dim, 1, 1], &format!("l{i}_attn4"))
            } else {
                let ar = blk.reshape(&attn, &[nkv, s, hpk * dh, 1], &format!("l{i}_ar"));
                let at = blk.transpose(
                    &ar,
                    &[0, 2, 3, 1],
                    &[nkv, hpk * dh, 1, s],
                    &format!("l{i}_at"),
                );
                blk.reshape(&at, &[1, q_dim, 1, s], &format!("l{i}_attn4"))
            }
        };
        let ow = st.get_f16(&format!("{p}.self_attn.o_proj.weight"))?;
        let wo = weight_const(
            &mut wb,
            &mut blk,
            &format!("w_o_{i}"),
            &ow,
            &[d, q_dim, 1, 1],
            bits,
        );
        let o = conv1x1_w(&mut blk, &attn4, &wo, d, s, &format!("l{i}_o"));
        cut!("o", o, [1, d, 1, s]);
        // Gemma3 post_attention_layernorm normalizes the sublayer OUTPUT
        // before the residual add; Qwen3 adds the raw o_proj result.
        let o2 = if cfg.arch == Arch::Gemma3Text && std::env::var("BADAPPLE_NO_POSTNORM").is_err() {
            let wp = norm_w(st, &format!("{p}.post_attention_layernorm.weight"))?;
            let wpn = blob(
                &mut wb,
                &mut blk,
                &format!("w_post_attn_{i}"),
                &wp,
                &[d, 1, 1],
            );
            rms_norm_big(
                &mut blk,
                &o,
                &wpn,
                d,
                cfg.eps,
                &shape_x,
                &format!("l{i}_pan"),
            )
        } else {
            o
        };
        let x1 = blk.add(&cur, &o2, &shape_x, &format!("l{i}_res1"));

        // ---- FFN ----
        // Qwen3's FFN norm is named post_attention_layernorm; Gemma3's is
        // pre_feedforward_layernorm (its post_attention_layernorm already
        // ran above, post-attention).
        let ffn_norm_name = if cfg.arch == Arch::Gemma3Text {
            "pre_feedforward_layernorm"
        } else {
            "post_attention_layernorm"
        };
        let wfn = norm_w(st, &format!("{p}.{ffn_norm_name}.weight"))?;
        let wfnn = blob(
            &mut wb,
            &mut blk,
            &format!("w_ffn_norm_{i}"),
            &wfn,
            &[d, 1, 1],
        );
        cut!("res1", x1, [1, d, 1, s]);
        // CUT=kvdbg: hidden = concat(K rows 0..8 read back from the packed
        // state post-update, res1) — reads the state's actual contents
        // through the terminal output buffer (extra outputs see reused
        // buffers, untrusted).
        if cut_at.as_deref() == Some("kvdbg") && le - ls == 1 {
            let ks8 = blk.slice(
                &k_attn,
                &[0, 0, 0, 0],
                &[1, nkv as i32, 8, dh as i32],
                &[1, nkv, 8, dh],
                &format!("l{i}_ks8"),
            );
            let kf = blk.reshape(&ks8, &[1, nkv * 8 * dh, 1, 1], &format!("l{i}_ks8f"));
            cur = blk.concat(&[kf, x1.clone()], 1, &[1, nkv * 8 * dh + d, 1, s], "hidden");
            cut_shape = Some(vec![1, nkv * 8 * dh + d, 1, s]);
            break;
        }
        let n2 = rms_norm_big(
            &mut blk,
            &x1,
            &wfnn,
            d,
            cfg.eps,
            &shape_x,
            &format!("l{i}_fn"),
        );
        // BADAPPLE_SPLIT_FFN: end the layer shard at concat(res1, fn_norm).
        // The wide FFN conv miscompiles when it shares a plan with the
        // stateful attention chain — kvdbg verified the K state and res1
        // are exact until the conv exists downstream in the same model.
        // The paired ffn_s shard computes the whole FFN tail on a real
        // model input (conv-on-input is the verified-exact configuration).
        if split_ffn {
            cur = blk.concat(&[x1.clone(), n2.clone()], 1, &[1, 2 * d, 1, s], "hidden");
            cut_shape = Some(vec![1, 2 * d, 1, s]);
            break;
        }
        cut!("fn", n2, [1, d, 1, s]; n2.clone() => [1, d, 1, s]);
        // The wide FFN projection must not consume the elementwise
        // norm-weight product (`l{i}_fn_out` = xn * w) directly: the ANE
        // compiler fuses the broadcast mul into the conv's producer and
        // miscomputes it (~8% error — verified: conv on the identical input
        // is exact at 0.003, and the same graph minus the conv reads 0.05).
        // Instead the norm weight is folded into the projection's weight
        // COLUMNS — conv(W·diag(w), xn) == conv(W, xn·w) exactly — and the
        // conv reads `l{i}_fn_xn`, the pre-weight norm output. Even that
        // isn't sufficient: the fused plan still miscomputes the norm
        // upstream of the conv. Drive the conv input through a
        // reverse_sequence round-trip — real data movement with no fold
        // rule in the compiler, so the tensor physically materializes
        // before the projection reads it.
        let ffn_in = {
            // lengths = channel count (1152): reverse the full seq_axis=1
            // for the single batch element.
            let seq_len_c = blk.konst_i32(&format!("l{i}_rsl"), &[d as i32]);
            let rsa = blk.konst_scalar_i32(&format!("l{i}_rsa"), 1);
            let rba = blk.konst_scalar_i32(&format!("l{i}_rba"), 0);
            let xn = format!("l{i}_fn_xn");
            let r1 = blk.o1(
                "reverse_sequence",
                vec![
                    ("x".into(), bind(&xn).1),
                    ("lengths".into(), bind(&seq_len_c).1),
                    ("seq_axis".into(), bind(&rsa).1),
                    ("batch_axis".into(), bind(&rba).1),
                ],
                &format!("l{i}_fnr1"),
                ValueType::Tensor(TensorType::f16(&shape_x)),
            );
            let rsa2 = blk.konst_scalar_i32(&format!("l{i}_rsa2"), 1);
            let rba2 = blk.konst_scalar_i32(&format!("l{i}_rba2"), 0);
            blk.o1(
                "reverse_sequence",
                vec![
                    ("x".into(), bind(&r1).1),
                    ("lengths".into(), bind(&seq_len_c).1),
                    ("seq_axis".into(), bind(&rsa2).1),
                    ("batch_axis".into(), bind(&rba2).1),
                ],
                &format!("l{i}_fnr2"),
                ValueType::Tensor(TensorType::f16(&shape_x)),
            )
        };
        // Column-fold the FFN norm weight into gate/up weights (fp16 math —
        // (1+w) already baked into `wfn` by norm_w for Gemma).
        let gw = f16_scale_cols(&st.get_f16(&format!("{p}.mlp.gate_proj.weight"))?, &wfn);
        let uw = f16_scale_cols(&st.get_f16(&format!("{p}.mlp.up_proj.weight"))?, &wfn);
        // BADAPPLE_SPLIT_GU (debug): separate gate/up projections — tests
        // whether the 2*d_ff-wide single projection itself fails plan-build.
        let (gate, up) = if let Ok(split_n) = std::env::var("BADAPPLE_SPLIT_GU") {
            let split_n: usize = split_n.parse().unwrap_or(1);
            let chunk = (cfg.d_ff as usize) / split_n;
            let chunk_b = chunk * 2; // f16 bytes per row
            let mut proj_chunks = |w: &[u8], tag: &str| -> String {
                let mut outs = Vec::with_capacity(split_n);
                for c in 0..split_n {
                    let wname = format!("w_{tag}{c}_{i}");
                    let wc = weight_const(
                        &mut wb,
                        &mut blk,
                        &wname,
                        &w[c * chunk * d as usize * 2..(c + 1) * chunk * d as usize * 2],
                        &[chunk as i64, d, 1, 1],
                        bits,
                    );
                    let _ = chunk_b;
                    outs.push(conv1x1_w(
                        &mut blk,
                        &ffn_in,
                        &wc,
                        chunk as i64,
                        s,
                        &format!("l{i}_{tag}{c}"),
                    ));
                }
                if split_n == 1 {
                    outs.pop().unwrap()
                } else {
                    blk.concat(&outs, 1, &[1, cfg.d_ff, 1, s], &format!("l{i}_{tag}cat"))
                }
            };
            (proj_chunks(&gw, "g"), proj_chunks(&uw, "u"))
        } else {
            let mut gu_w = gw;
            gu_w.extend_from_slice(&uw);
            let wgu = weight_const(
                &mut wb,
                &mut blk,
                &format!("w_gu_{i}"),
                &gu_w,
                &[2 * cfg.d_ff, d, 1, 1],
                bits,
            );
            // Post-update projections stay conv: matmul's reshapes on the
            // FFN-norm output fail plan-build where the conv passes
            // (bisected). The pre-update QKV projection is the inverse.
            // BADAPPLE_GU_MM (debug): route the fused projection through
            // linear_w's matmul path — the reshape pair changes the input
            // layout and breaks the norm→conv fusion miscompile.
            let gu = if std::env::var("BADAPPLE_GU_MM").is_ok() {
                linear_w(
                    &mut blk,
                    &ffn_in,
                    &wgu,
                    d,
                    2 * cfg.d_ff,
                    s,
                    &format!("l{i}_gu"),
                )
            } else {
                conv1x1_w(
                    &mut blk,
                    &ffn_in,
                    &wgu,
                    2 * cfg.d_ff,
                    s,
                    &format!("l{i}_gu"),
                )
            };
            // CUT=gufn: hidden = concat(attn4, o, res1, fn_xn, gu) — one
            // output buffer materializes the whole chain: extra outputs
            // read reused buffers and can't be trusted mid-graph, but the
            // terminal concat pins all five values.
            if cut_at.as_deref() == Some("gufn") && le - ls == 1 {
                let cat = blk.concat(
                    &[
                        format!("l{i}_attn4"),
                        format!("l{i}_o"),
                        x1.clone(),
                        format!("l{i}_fn_xn"),
                        gu.clone(),
                    ],
                    1,
                    &[1, q_dim + 3 * d + 2 * cfg.d_ff, 1, s],
                    "hidden",
                );
                cur = cat;
                cut_shape = Some(vec![1, q_dim + 3 * d + 2 * cfg.d_ff, 1, s]);
                break;
            }
            cut!("gu", gu, [1, 2 * cfg.d_ff, 1, s]);
            (
                blk.slice(
                    &gu,
                    &[0, 0, 0, 0],
                    &[1, cfg.d_ff as i32, 1, s as i32],
                    &[1, cfg.d_ff, 1, s],
                    &format!("l{i}_gate"),
                ),
                blk.slice(
                    &gu,
                    &[0, cfg.d_ff as i32, 0, 0],
                    &[1, (2 * cfg.d_ff) as i32, 1, s as i32],
                    &[1, cfg.d_ff, 1, s],
                    &format!("l{i}_up"),
                ),
            )
        };
        // Gate activation: Qwen3 silu, Gemma3 gelu (tanh approximation).
        let act = if cfg.arch == Arch::Gemma3Text && std::env::var("BADAPPLE_GEMMA_SILU").is_err() {
            blk.o1(
                "gelu",
                vec![
                    ("x".into(), bind(&gate).1),
                    (
                        "mode".into(),
                        bind_const(mil_spec::Value::Str("TANH_APPROXIMATION".into())),
                    ),
                ],
                &format!("l{i}_gelu"),
                ValueType::Tensor(TensorType::f16(&[1, cfg.d_ff, 1, s])),
            )
        } else {
            blk.o1(
                "silu",
                vec![("x".into(), bind(&gate).1)],
                &format!("l{i}_silu"),
                ValueType::Tensor(TensorType::f16(&[1, cfg.d_ff, 1, s])),
            )
        };
        let hidden = blk.mul(&act, &up, &[1, cfg.d_ff, 1, s], &format!("l{i}_hid"));
        cut!("hid", hidden, [1, cfg.d_ff, 1, s]);
        let dw = st.get_f16(&format!("{p}.mlp.down_proj.weight"))?;
        let wd = weight_const(
            &mut wb,
            &mut blk,
            &format!("w_down_{i}"),
            &dw,
            &[d, cfg.d_ff, 1, 1],
            bits,
        );
        // The FFN hidden is the only un-normalized matmul input; Qwen3's
        // massive activation (~8e3 at layer 6) overflows the ANE's int8-weight
        // conv. Pre-scaling by 1/α and restoring by α is exact in real
        // arithmetic and keeps the int8 path inside FP16 range.
        let alpha = down_prescale_for(bits);
        let ffn = if alpha > 1.0 {
            let inv = blk.konst_f16(&format!("l{i}_dinv"), 1.0 / alpha);
            let hs = blk.mul(&hidden, &inv, &[1, cfg.d_ff, 1, s], &format!("l{i}_hids"));
            let raw = conv1x1_w(&mut blk, &hs, &wd, d, s, &format!("l{i}_down_s"));
            let a = blk.konst_f16(&format!("l{i}_dalpha"), alpha);
            blk.mul(&raw, &a, &shape_x, &format!("l{i}_down"))
        } else {
            conv1x1_w(&mut blk, &hidden, &wd, d, s, &format!("l{i}_down"))
        };
        cut!("down", ffn, [1, d, 1, s]);
        // Gemma3 post_feedforward_layernorm normalizes the MLP output
        // before the residual add.
        let ffn_out =
            if cfg.arch == Arch::Gemma3Text && std::env::var("BADAPPLE_NO_POSTNORM").is_err() {
                let wp = norm_w(st, &format!("{p}.post_feedforward_layernorm.weight"))?;
                let wpn = blob(
                    &mut wb,
                    &mut blk,
                    &format!("w_post_ffn_{i}"),
                    &wp,
                    &[d, 1, 1],
                );
                rms_norm_big(
                    &mut blk,
                    &ffn,
                    &wpn,
                    d,
                    cfg.eps,
                    &shape_x,
                    &format!("l{i}_pfn"),
                )
            } else {
                ffn
            };
        // the last layer's output must be named `hidden` to match the declared
        // model output feature (CoreML binds program outputs by name).
        let stop_at = std::env::var("BADAPPLE_STOP_AT").ok();
        let out_nm = if i == le - 1 && (stop_at.is_none() || s != 1) {
            "hidden".to_string()
        } else {
            format!("l{i}_res2")
        };
        cur = blk.add(&x1, &ffn_out, &shape_x, &out_nm);
    }
    // Commit the chained packed-KV updates once per call.
    blk.write_state("kv_cache", &cur_kv);

    // BADAPPLE_STOP_AT=l0_<stage> emits a debug package whose single output is
    // an intermediate: a trailing `add(x, 0)` named `hidden` re-binds the
    // intermediate so the declared output is still the last-produced value
    // (E5 rejects outputs bound to mid-block intermediates).
    let stop_at = std::env::var("BADAPPLE_STOP_AT").ok();
    let mut stop_shape: Option<Vec<i64>> = None;
    if stop_at.is_some() && le - ls == 1 && s == 1 {
        let i = ls;
        for (suffix, shape) in [
            ("an_out", vec![1, d, 1, 1]),        // post attn_norm
            ("qo", vec![1, q_dim, 1, 1]),        // q post head-norm+weight
            ("ko", vec![1, kv_dim, 1, 1]),       // k post head-norm+weight
            ("rq_flat", vec![1, q_dim, 1, 1]),   // q post-rope
            ("rk_flat", vec![1, kv_dim, 1, 1]),  // k post-rope
            ("attn4", vec![1, q_dim, 1, 1]),     // attention out pre-o_proj
            ("v", vec![1, kv_dim, 1, 1]),        // v slice of qkv (pre-norm/rope)
            ("sc", vec![1, nkv, 2 * hpk, seq]),  // attention scores pre-mask (padded)
            ("msk", vec![1, nkv, 2 * hpk, seq]), // scores post-mask (padded)
            ("pb", vec![1, nkv, 2 * hpk, seq]),  // attention probs (padded)
            ("o", vec![1, d, 1, 1]),             // o_proj out
            ("res1", vec![1, d, 1, 1]),          // post-attn residual
            ("fn_out", vec![1, d, 1, 1]),        // post ffn_norm
            ("hid", vec![1, cfg.d_ff, 1, 1]),    // FFN hidden pre-down
        ] {
            let n = format!("l{i}_{suffix}");
            if stop_at.as_deref() == Some(n.as_str()) {
                // Flatten to [1,N,1,1]; skip the reshape when already flat —
                // an identical-shape reshape op breaks plan-build (-14).
                // Stopping before the state writes makes them dead code
                // (also -14), so fold a zero-weighted single-axis
                // reduce_mean of the block tail into the output to keep the
                // full graph live.
                let flat: i64 = shape.iter().product();
                let src = if shape == [1, flat, 1, 1] {
                    n.clone()
                } else {
                    blk.reshape(&n, &[1, flat, 1, 1], &format!("l{i}_{suffix}_flat"))
                };
                let kax = blk.konst_i32("stop_ax", &[1]);
                let kkd = blk.konst_bool("stop_kd", true);
                let tail = blk.o1(
                    "reduce_mean",
                    vec![
                        ("x".into(), bind(&cur).1),
                        ("axes".into(), bind(&kax).1),
                        ("keep_dims".into(), bind(&kkd).1),
                    ],
                    "stop_tail",
                    ValueType::Tensor(TensorType::f16(&[1, 1, 1, 1])),
                );
                let z = blk.konst_f16("stop_zero", 0.0);
                let keep = blk.mul(&tail, &z, &[1, 1, 1, 1], "stop_keep");
                cur = blk.add(&src, &keep, &[1, flat, 1, 1], "hidden");
                stop_shape = Some(vec![1, flat, 1, 1]);
            }
        }
    }
    blk.outputs = vec![cur.clone()];
    for (extra_name, _) in &cut_extra {
        blk.outputs.push(extra_name.clone());
    }
    let out_shape_decl = cut_shape.or(stop_shape).unwrap_or_else(|| vec![1, d, 1, s]);

    // ---- serialize ----
    // Rope inputs carry R rows: row 0 = global base, row 1 = local base
    // (Gemma3 sliding-window layers slice their own row in-graph).
    let rrows = cfg.rope_rows();
    let rope_shape: Vec<i64> = if s == 1 {
        vec![rrows, rope_half]
    } else {
        vec![rrows, 1, rope_half, s]
    };
    // s==1 declares a real dim2 on attn_mask ([1,1,rows,seq] where rows is
    // the padded query count): a [1,1,1,seq] mask broadcasts dim2 1->rows,
    // which the s==1 plan misaligns — probs then spread onto unwritten
    // (garbage) KV slots instead of staying one-hot. dim0/dim1 broadcast
    // is proven fine (shared with the s>1 path).
    let mask_shape: Vec<i64> = if s == 1 {
        vec![1, 1, 2 * hpk, seq]
    } else {
        vec![1, 1, hpk * s, seq]
    };
    let mut fn_inputs: Vec<NVT> = vec![
        NVT {
            name: "x".into(),
            ty: ValueType::Tensor(TensorType::f16(&[1, d, 1, s])),
        },
        NVT {
            name: "rope_cos".into(),
            ty: ValueType::Tensor(TensorType::f16(&rope_shape)),
        },
        NVT {
            name: "rope_sin".into(),
            ty: ValueType::Tensor(TensorType::f16(&rope_shape)),
        },
        NVT {
            name: "attn_mask".into(),
            ty: ValueType::Tensor(TensorType::f16(&mask_shape)),
        },
        NVT {
            name: "pos".into(),
            ty: ValueType::Tensor(TensorType::i32v(1)),
        },
    ];
    // Sliding-window shards take a second band mask from the bridge.
    let has_sliding = (ls..le).any(|i| cfg.is_sliding(i));
    if has_sliding {
        fn_inputs.push(NVT {
            name: "attn_mask_slide".into(),
            ty: ValueType::Tensor(TensorType::f16(&mask_shape)),
        });
    }
    fn_inputs.push(NVT {
        name: "kv_cache".into(),
        ty: ValueType::State(TensorType::f16(&kvshape)),
    });
    if std::env::var("BADAPPLE_QKV_INPUT").is_ok() {
        fn_inputs.push(NVT {
            name: "qkv".into(),
            ty: ValueType::Tensor(TensorType::f16(&[1, q_dim + 2 * kv_dim, 1, s])),
        });
    }
    if std::env::var("BADAPPLE_GU_INPUT").is_ok() {
        for n in ["wgu_g", "wgu_u"] {
            fn_inputs.push(NVT {
                name: n.into(),
                ty: ValueType::Tensor(TensorType::f16(&[cfg.d_ff, d, 1, 1])),
            });
        }
    }
    let states: Vec<Feature> = vec![Feature {
        name: "kv_cache".into(),
        shape: kvshape.clone(),
        dtype: DType::Fp16,
        is_state: true,
    }];
    let mut inputs: Vec<Feature> = vec![
        Feature {
            name: "x".into(),
            shape: vec![1, d, 1, s],
            dtype: DType::Fp16,
            is_state: false,
        },
        Feature {
            name: "rope_cos".into(),
            shape: rope_shape.clone(),
            dtype: DType::Fp16,
            is_state: false,
        },
        Feature {
            name: "rope_sin".into(),
            shape: rope_shape.clone(),
            dtype: DType::Fp16,
            is_state: false,
        },
        Feature {
            name: "attn_mask".into(),
            shape: mask_shape.clone(),
            dtype: DType::Fp16,
            is_state: false,
        },
        Feature {
            name: "pos".into(),
            shape: vec![1],
            dtype: DType::Int32,
            is_state: false,
        },
    ];
    if has_sliding {
        inputs.push(Feature {
            name: "attn_mask_slide".into(),
            shape: mask_shape.clone(),
            dtype: DType::Fp16,
            is_state: false,
        });
    }
    if std::env::var("BADAPPLE_QKV_INPUT").is_ok() {
        inputs.push(Feature {
            name: "qkv".into(),
            shape: vec![1, q_dim + 2 * kv_dim, 1, s],
            dtype: DType::Fp16,
            is_state: false,
        });
    }
    if std::env::var("BADAPPLE_GU_INPUT").is_ok() {
        for n in ["wgu_g", "wgu_u"] {
            inputs.push(Feature {
                name: n.into(),
                shape: vec![cfg.d_ff, d, 1, 1],
                dtype: DType::Fp16,
                is_state: false,
            });
        }
    }
    let mut outputs = vec![Feature {
        name: "hidden".into(),
        shape: out_shape_decl,
        dtype: DType::Fp16,
        is_state: false,
    }];
    for (extra_name, extra_shape) in &cut_extra {
        outputs.push(Feature {
            name: extra_name.clone(),
            shape: extra_shape.clone(),
            dtype: DType::Fp16,
            is_state: false,
        });
    }

    // The ANE plan-build coverage of the slice_update graph is
    // spec-dependent: the s==1 (decode) graph only plans under ios18/
    // CoreML8, while the s>1 (prefill) graph only plans under ios19/
    // CoreML9 — each fails -14 under the other spec (verified across
    // every shard and bisected to the spec flag itself). Emit each shard
    // at the spec its graph plans under; deployment target is per-model.
    let (sv, ops) = if s == 1 {
        (9, "CoreML8")
    } else {
        (10, "CoreML9")
    };
    // Spec-version sweep for new OS releases: ANE plan-build coverage
    // shifts per CoreML generation (documented above).
    let (sv, ops) = if let Ok(v) = std::env::var("BADAPPLE_SPEC_VERSION") {
        let n: i32 = v.parse().unwrap_or(sv);
        let o = std::env::var("BADAPPLE_OPSET").unwrap_or_else(|_| ops.to_string());
        (n, Box::leak(o.into_boxed_str()) as &'static str)
    } else {
        (sv, ops)
    };
    let meta = ModelMeta::new(sv, ops)
        .creator("badapple-aneconvert")
        .description("Bad Apple native ANE shard");
    let spec = encode_model(&inputs, &outputs, &states, &blk, &fn_inputs, &meta);
    let weights = wb.finish();
    write_mlpackage(&pkg_dir, &spec, Some(&weights)).map_err(|e| e.to_string())?;
    let mlmodelc = out_dir.join(format!("{name}.mlmodelc"));
    Ok((pkg_dir, mlmodelc, weights))
}

/// Pre-port Rust FFN-shard emitter — byte-parity oracle for the .cy path.
#[cfg(test)]
fn emit_ffn_shard_rust(
    st: &StIndex,
    cfg: &Cfg,
    ls: i64,
    le: i64,
    out_dir: &Path,
    s: i64,
    is_prefill: bool,
    bits: u32,
) -> Result<(PathBuf, PathBuf, Vec<u8>), String> {
    if le - ls != 1 {
        // Each FFN shard consumes exactly one layer's split output —
        // multi-layer coverage is impossible by construction (the
        // intermediate res1/fn values only exist inside the paired
        // attention shard), so split mode always runs lps=1.
        return Err("split FFN shards are per-layer (le-ls must be 1)".into());
    }
    let name = if is_prefill {
        format!("ffn_p{:02}-{:02}", ls, le)
    } else {
        format!("ffn_s{:02}-{:02}", ls, le)
    };
    let pkg_dir = out_dir.join(format!("{name}.mlpackage"));

    let mut blk = Block::new();
    let mut wb = WeightBin::new();
    let d = cfg.d_model;
    let shape_x = vec![1, d, 1, s];

    // Single input x = concat(res1, fn_norm) [1,2d,1,s]: slices take the
    // halves so the bridge can hand the attention shard's output over
    // verbatim. (Earlier "all zeros" reads were a readback artifact — the
    // output MLMultiArray carries channel-stride 32 and must be read
    // stride-aware; the slice path computes correctly.)
    let res1 = blk.slice(
        "x",
        &[0, 0, 0, 0],
        &[1, d as i32, 1, s as i32],
        &shape_x,
        "in_res1",
    );
    let n2 = blk.slice(
        "x",
        &[0, d as i32, 0, 0],
        &[1, (2 * d) as i32, 1, s as i32],
        &shape_x,
        "in_fn",
    );

    let ffn_cut = std::env::var("BADAPPLE_FFN_CUT").ok();
    let mut cur = res1.clone();
    let mut out_shape_override: Option<Vec<i64>> = None;
    for i in ls..le {
        let p = format!("model.layers.{i}");
        let gw = st.get_f16(&format!("{p}.mlp.gate_proj.weight"))?;
        let uw = st.get_f16(&format!("{p}.mlp.up_proj.weight"))?;
        let mut gu_w = gw;
        gu_w.extend_from_slice(&uw);
        let wgu = weight_const(
            &mut wb,
            &mut blk,
            &format!("w_gu_{i}"),
            &gu_w,
            &[2 * cfg.d_ff, d, 1, 1],
            bits,
        );
        // Matmul, not conv: post-update matmul reshapes are the documented
        // -14 case, but this shard has no slice_update at all — and the
        // conv form miscomputes here regardless (output written at stride
        // 32 into the flat buffer: ane[k*32] == ref[k] to 0.001).
        let gu = linear_w(&mut blk, &n2, &wgu, d, 2 * cfg.d_ff, s, &format!("l{i}_gu"));
        if ffn_cut.as_deref() == Some("gu") {
            let z = blk.konst_f16("fcz", 0.0);
            cur = blk.add(&gu, &z, &[1, 2 * cfg.d_ff, 1, s], "hidden");
            out_shape_override = Some(vec![1, 2 * cfg.d_ff, 1, s]);
            break;
        }
        let gate = blk.slice(
            &gu,
            &[0, 0, 0, 0],
            &[1, cfg.d_ff as i32, 1, s as i32],
            &[1, cfg.d_ff, 1, s],
            &format!("l{i}_gate"),
        );
        let up = blk.slice(
            &gu,
            &[0, cfg.d_ff as i32, 0, 0],
            &[1, (2 * cfg.d_ff) as i32, 1, s as i32],
            &[1, cfg.d_ff, 1, s],
            &format!("l{i}_up"),
        );
        let act = if cfg.arch == Arch::Gemma3Text && std::env::var("BADAPPLE_GEMMA_SILU").is_err() {
            blk.o1(
                "gelu",
                vec![
                    ("x".into(), bind(&gate).1),
                    (
                        "mode".into(),
                        bind_const(mil_spec::Value::Str("TANH_APPROXIMATION".into())),
                    ),
                ],
                &format!("l{i}_gelu"),
                ValueType::Tensor(TensorType::f16(&[1, cfg.d_ff, 1, s])),
            )
        } else {
            blk.o1(
                "silu",
                vec![("x".into(), bind(&gate).1)],
                &format!("l{i}_silu"),
                ValueType::Tensor(TensorType::f16(&[1, cfg.d_ff, 1, s])),
            )
        };
        let hidden = blk.mul(&act, &up, &[1, cfg.d_ff, 1, s], &format!("l{i}_hid"));
        if ffn_cut.as_deref() == Some("hid") {
            let z = blk.konst_f16("fcz", 0.0);
            cur = blk.add(&hidden, &z, &[1, cfg.d_ff, 1, s], "hidden");
            out_shape_override = Some(vec![1, cfg.d_ff, 1, s]);
            break;
        }
        let dw = st.get_f16(&format!("{p}.mlp.down_proj.weight"))?;
        let wd = weight_const(
            &mut wb,
            &mut blk,
            &format!("w_down_{i}"),
            &dw,
            &[d, cfg.d_ff, 1, 1],
            bits,
        );
        // Down projection is matmul for the same reason as gate/up: the
        // conv form scatters outputs at stride 32 inside this plan, and
        // the matmul -14 case (post-slice_update reshapes) does not exist
        // in a stateless shard.
        let alpha = down_prescale_for(bits);
        let ffn = if alpha > 1.0 {
            let inv = blk.konst_f16(&format!("l{i}_dinv"), 1.0 / alpha);
            let hs = blk.mul(&hidden, &inv, &[1, cfg.d_ff, 1, s], &format!("l{i}_hids"));
            let raw = linear_w(&mut blk, &hs, &wd, cfg.d_ff, d, s, &format!("l{i}_down_s"));
            let a = blk.konst_f16(&format!("l{i}_dalpha"), alpha);
            blk.mul(&raw, &a, &shape_x, &format!("l{i}_down"))
        } else {
            linear_w(
                &mut blk,
                &hidden,
                &wd,
                cfg.d_ff,
                d,
                s,
                &format!("l{i}_down"),
            )
        };
        let ffn_out =
            if cfg.arch == Arch::Gemma3Text && std::env::var("BADAPPLE_NO_POSTNORM").is_err() {
                // Gemma stores w but applies (1+w); fold the +1 at emission.
                let wp_raw = st.get_f16(&format!("{p}.post_feedforward_layernorm.weight"))?;
                let wp = if cfg.norm_plus_one {
                    f16_plus_one(&wp_raw)
                } else {
                    wp_raw
                };
                let wpn = wb.put(&format!("w_post_ffn_{i}"), DType::Fp16, &[d, 1, 1], &wp);
                let wpn = blk.konst_blob(
                    &format!("w_post_ffn_{i}"),
                    WEIGHTS_NAME,
                    wpn,
                    DType::Fp16,
                    &[d, 1, 1],
                );
                rms_norm_big(
                    &mut blk,
                    &ffn,
                    &wpn,
                    d,
                    cfg.eps,
                    &shape_x,
                    &format!("l{i}_pfn"),
                )
            } else {
                ffn
            };
        let out_nm = if i == le - 1 { "hidden" } else { "" };
        let nm = if out_nm.is_empty() {
            format!("l{i}_res2")
        } else {
            out_nm.to_string()
        };
        cur = blk.add(&res1, &ffn_out, &shape_x, &nm);
    }
    blk.outputs = vec![cur.clone()];

    let (sv, ops) = if s == 1 {
        (9, "CoreML8")
    } else {
        (10, "CoreML9")
    };
    let (sv, ops) = if let Ok(v) = std::env::var("BADAPPLE_SPEC_VERSION") {
        let n: i32 = v.parse().unwrap_or(sv);
        let o = std::env::var("BADAPPLE_OPSET").unwrap_or_else(|_| ops.to_string());
        (n, Box::leak(o.into_boxed_str()) as &'static str)
    } else {
        (sv, ops)
    };
    let shape_2d = vec![1, 2 * d, 1, s];
    let fn_inputs: Vec<NVT> = vec![NVT {
        name: "x".into(),
        ty: ValueType::Tensor(TensorType::f16(&shape_2d)),
    }];
    let inputs: Vec<Feature> = vec![Feature {
        name: "x".into(),
        shape: shape_2d,
        dtype: DType::Fp16,
        is_state: false,
    }];
    let outputs = vec![Feature {
        name: "hidden".into(),
        shape: out_shape_override.unwrap_or_else(|| vec![1, d, 1, s]),
        dtype: DType::Fp16,
        is_state: false,
    }];
    let states: Vec<Feature> = vec![];
    let meta = ModelMeta::new(sv, ops)
        .creator("badapple-aneconvert")
        .description("Bad Apple native ANE FFN shard (split layer tail)");
    let spec = encode_model(&inputs, &outputs, &states, &blk, &fn_inputs, &meta);
    let weights = wb.finish();
    write_mlpackage(&pkg_dir, &spec, Some(&weights)).map_err(|e| e.to_string())?;
    let mlmodelc = out_dir.join(format!("{name}.mlmodelc"));
    Ok((pkg_dir, mlmodelc, weights))
}

/// per-head RMSNorm over axis 1 of (n, dh, 1, 1); weight application handled by caller.
#[cfg(test)]
fn head_rms(blk: &mut Block, x: &str, d: i64, eps: f32, shape4: &[i64], pfx: &str) -> String {
    // Same two-sided fp16 guard as rms_norm_big: 1/(8*sqrt(d)) prescale so
    // per-head activations survive |x| up to ~70k, plus a subnormal floor
    // before rsqrt so near-zero heads can't produce rsqrt(0)=Inf.
    const C: f32 = 8.0;
    let k = (d as f32).sqrt();
    let inv_k = blk.konst_f16(&format!("{pfx}_ik"), 1.0 / (C * k));
    let xs = blk.mul(x, &inv_k, shape4, &format!("{pfx}_xs"));
    let sq = blk.mul(&xs, &xs, shape4, &format!("{pfx}_sq"));
    let axes = blk.konst_i32(&format!("{pfx}_ax"), &[1]);
    let kd = blk.konst_bool(&format!("{pfx}_kd"), true);
    let mshape = vec![shape4[0], 1, 1, shape4[3]];
    let mean = blk.o1(
        "reduce_mean",
        vec![
            ("x".into(), bind(&sq).1),
            ("axes".into(), bind(&axes).1),
            ("keep_dims".into(), bind(&kd).1),
        ],
        &format!("{pfx}_m"),
        ValueType::Tensor(TensorType::f16(&mshape)),
    );
    let e = blk.konst_f16(&format!("{pfx}_e"), eps / (C * C * k * k));
    let vp = blk.add(&mean, &e, &mshape, &format!("{pfx}_vp"));
    let fl = blk.konst_f16(&format!("{pfx}_fl"), 1e-7);
    let vm = blk.o1(
        "maximum",
        vec![("x".into(), bind(&vp).1), ("y".into(), bind(&fl).1)],
        &format!("{pfx}_vm"),
        ValueType::Tensor(TensorType::f16(&mshape)),
    );
    let z = blk.konst_f16(&format!("{pfx}_z"), 0.0);
    let r = blk.o1(
        "rsqrt",
        vec![("x".into(), bind(&vm).1), ("epsilon".into(), bind(&z).1)],
        &format!("{pfx}_r"),
        ValueType::Tensor(TensorType::f16(&mshape)),
    );
    blk.mul(&xs, &r, shape4, &format!("{pfx}_xn"))
}

/// Pre-port Rust head-shard emitter — byte-parity oracle for the .cy path.
#[cfg(test)]
fn emit_head_shard_rust(
    st: &StIndex,
    cfg: &Cfg,
    vstart: i64,
    vend: i64,
    out_dir: &Path,
    bits: u32,
) -> Result<(PathBuf, PathBuf), String> {
    let name = format!("head_v{:05}-{:05}", vstart, vend);
    let pkg_dir = out_dir.join(format!("{name}.mlpackage"));
    let mut blk = Block::new();
    let mut wb = WeightBin::new();
    let d = cfg.d_model;
    let vsz = vend - vstart;

    let mut nw = st.get_f16("model.norm.weight")?;
    if cfg.norm_plus_one {
        nw = f16_plus_one(&nw);
    }
    let noff = wb.put("w_out_norm", DType::Fp16, &[d, 1, 1], &nw);
    let wn = blk.konst_blob("w_out_norm", WEIGHTS_NAME, noff, DType::Fp16, &[d, 1, 1]);

    let normed = rms_norm_big(&mut blk, "x", &wn, d, cfg.eps, &[1, d, 1, 1], "hn");

    // sliced lm head weight: (vsz, d) -> conv (vsz, d, 1, 1)
    let lm_name = if cfg.tie_embeddings || !st.tensors.contains_key("lm_head.weight") {
        "model.embed_tokens.weight"
    } else {
        "lm_head.weight"
    };
    let slice_f16 = st.get_f16_rows(lm_name, vstart as u64, vend as u64)?;
    let wl = weight_const(&mut wb, &mut blk, "w_lm", &slice_f16, &[vsz, d, 1, 1], bits);
    let logits = blk.conv1x1(&normed, &wl, None, vsz, "logits");
    blk.outputs = vec![logits];

    let fn_inputs = vec![NVT {
        name: "x".into(),
        ty: ValueType::Tensor(TensorType::f16(&[1, d, 1, 1])),
    }];
    let inputs = vec![Feature {
        name: "x".into(),
        shape: vec![1, d, 1, 1],
        dtype: DType::Fp16,
        is_state: false,
    }];
    let outputs = vec![Feature {
        name: "logits".into(),
        shape: vec![1, vsz, 1, 1],
        dtype: DType::Fp16,
        is_state: false,
    }];
    let meta = ModelMeta::new(SPEC_VERSION, OPSET)
        .creator("badapple-aneconvert")
        .description("Bad Apple native ANE shard");
    let spec = encode_model(&inputs, &outputs, &[], &blk, &fn_inputs, &meta);
    let weights = wb.finish();
    write_mlpackage(&pkg_dir, &spec, Some(&weights)).map_err(|e| e.to_string())?;
    Ok((pkg_dir, out_dir.join(format!("{name}.mlmodelc"))))
}

/// Pre-port Rust probe emitter — byte-parity oracle for the .cy path.
#[cfg(test)]
fn emit_probe_rust(
    st: &StIndex,
    cfg: &Cfg,
    ls: i64,
    stage: &str,
    out_dir: &Path,
) -> Result<(PathBuf, PathBuf, Vec<u8>), String> {
    if cfg.arch != Arch::Qwen3 {
        return Err("probe emitter is qwen3-only".into());
    }
    let name = format!("probe_l{ls}_{stage}");
    let pkg_dir = out_dir.join(format!("{name}.mlpackage"));
    let mut blk = Block::new();
    let mut wb = WeightBin::new();
    let d = cfg.d_model;
    let nh = cfg.n_heads;
    let nkv = cfg.n_kv;
    let dh = cfg.d_head;
    let rope_dim = cfg.rope_dim;
    let rope_half = rope_dim / 2;
    let q_dim = nh * dh;
    let kv_dim = nkv * dh;
    let p = format!("model.layers.{ls}");

    let w = st.get_f16(&format!("{p}.input_layernorm.weight"))?;
    let noff = wb.put(&format!("w_attn_norm_{ls}"), DType::Fp16, &[d, 1, 1], &w);
    let wn = blk.konst_blob(
        &format!("w_attn_norm_{ls}"),
        WEIGHTS_NAME,
        noff,
        DType::Fp16,
        &[d, 1, 1],
    );
    let normed = rms_norm_big(
        &mut blk,
        "x",
        &wn,
        d,
        cfg.eps,
        &[1, d, 1, 1],
        &format!("l{ls}_an"),
    );
    if stage == "an" {
        let z = blk.konst_f16("pz", 0.0);
        let o = blk.add(&normed, &z, &[1, d, 1, 1], "out");
        blk.outputs = vec![o];
        return finish_probe(&pkg_dir, out_dir, &name, &blk, wb, d, d, rope_half);
    }
    // "xw": raw x * w broadcast test (no rms) — isolates blob-const broadcast
    if stage == "xw" {
        let o = blk.mul("x", &wn, &[1, d, 1, 1], "xw");
        let z = blk.konst_f16("pz", 0.0);
        let o = blk.add(&o, &z, &[1, d, 1, 1], "out");
        blk.outputs = vec![o];
        return finish_probe(&pkg_dir, out_dir, &name, &blk, wb, d, d, rope_half);
    }
    // primitive probes for scalar broadcast
    if stage == "muls" {
        let s = blk.konst_f16("cs", 0.5);
        let o = blk.mul("x", &s, &[1, d, 1, 1], "out");
        blk.outputs = vec![o];
        return finish_probe(&pkg_dir, out_dir, &name, &blk, wb, d, d, rope_half);
    }
    if stage == "adds" {
        let s = blk.konst_f16("cs", 1.5);
        let o = blk.add("x", &s, &[1, d, 1, 1], "out");
        blk.outputs = vec![o];
        return finish_probe(&pkg_dir, out_dir, &name, &blk, wb, d, d, rope_half);
    }
    if stage == "id" {
        let o = blk.add("x", "x", &[1, d, 1, 1], "out");
        blk.outputs = vec![o];
        return finish_probe(&pkg_dir, out_dir, &name, &blk, wb, d, d, rope_half);
    }
    // "xw4": same but weight declared rank-4 (1,d,1,1) — exact-rank broadcast
    if stage == "xw4" {
        let noff4 = wb.put("w_attn_norm4", DType::Fp16, &[1, d, 1, 1], &w);
        let wn4 = blk.konst_blob(
            "w_attn_norm4",
            WEIGHTS_NAME,
            noff4,
            DType::Fp16,
            &[1, d, 1, 1],
        );
        let o = blk.mul("x", &wn4, &[1, d, 1, 1], "xw4");
        let z = blk.konst_f16("pz", 0.0);
        let o = blk.add(&o, &z, &[1, d, 1, 1], "out");
        blk.outputs = vec![o];
        return finish_probe(&pkg_dir, out_dir, &name, &blk, wb, d, d, rope_half);
    }
    // "anr": rms WITHOUT weight — isolates the norm path
    if stage == "anr" {
        let ik = blk.konst_f16("anr_ik", 1.0 / (d as f32).sqrt());
        let xs = blk.mul("x", &ik, &[1, d, 1, 1], "anr_xs");
        let sq = blk.mul(&xs, &xs, &[1, d, 1, 1], "anr_sq");
        let ax = blk.konst_i32("anr_ax", &[1]);
        let kd = blk.konst_bool("anr_kd", true);
        let m = blk.o1(
            "reduce_mean",
            vec![
                ("x".into(), bind(&sq).1),
                ("axes".into(), bind(&ax).1),
                ("keep_dims".into(), bind(&kd).1),
            ],
            "anr_m",
            ValueType::Tensor(TensorType::f16(&[1, 1, 1, 1])),
        );
        let ep = blk.konst_f16("anr_e", cfg.eps / d as f32);
        let vp = blk.add(&m, &ep, &[1, 1, 1, 1], "anr_vp");
        let zz = blk.konst_f16("anr_z", 0.0);
        let r = blk.o1(
            "rsqrt",
            vec![("x".into(), bind(&vp).1), ("epsilon".into(), bind(&zz).1)],
            "anr_r",
            ValueType::Tensor(TensorType::f16(&[1, 1, 1, 1])),
        );
        let xn = blk.mul(&xs, &r, &[1, d, 1, 1], "anr_xn");
        let z2 = blk.konst_f16("pz", 0.0);
        let o = blk.add(&xn, &z2, &[1, d, 1, 1], "out");
        blk.outputs = vec![o];
        return finish_probe(&pkg_dir, out_dir, &name, &blk, wb, d, d, rope_half);
    }

    let qw = st.get_f16(&format!("{p}.self_attn.q_proj.weight"))?;
    let kw = st.get_f16(&format!("{p}.self_attn.k_proj.weight"))?;
    let vw = st.get_f16(&format!("{p}.self_attn.v_proj.weight"))?;
    let mut qkv_w = qw;
    qkv_w.extend_from_slice(&kw);
    qkv_w.extend_from_slice(&vw);
    let woff = wb.put(
        &format!("w_qkv_{ls}"),
        DType::Fp16,
        &[q_dim + 2 * kv_dim, d, 1, 1],
        &qkv_w,
    );
    let wqkv = blk.konst_blob(
        &format!("w_qkv_{ls}"),
        WEIGHTS_NAME,
        woff,
        DType::Fp16,
        &[q_dim + 2 * kv_dim, d, 1, 1],
    );
    let qkv = blk.conv1x1(
        &normed,
        &wqkv,
        None,
        q_dim + 2 * kv_dim,
        &format!("l{ls}_qkv"),
    );
    let q = blk.slice(
        &qkv,
        &[0, 0, 0, 0],
        &[1, q_dim as i32, 1, 1],
        &[1, q_dim, 1, 1],
        &format!("l{ls}_q"),
    );
    let k = blk.slice(
        &qkv,
        &[0, q_dim as i32, 0, 0],
        &[1, (q_dim + kv_dim) as i32, 1, 1],
        &[1, kv_dim, 1, 1],
        &format!("l{ls}_k"),
    );
    let v = blk.slice(
        &qkv,
        &[0, (q_dim + kv_dim) as i32, 0, 0],
        &[1, (q_dim + 2 * kv_dim) as i32, 1, 1],
        &[1, kv_dim, 1, 1],
        &format!("l{ls}_v"),
    );
    if stage == "q" {
        let z = blk.konst_f16("pz", 0.0);
        let o = blk.add(&q, &z, &[1, q_dim, 1, 1], "out");
        blk.outputs = vec![o];
        return finish_probe(&pkg_dir, out_dir, &name, &blk, wb, d, q_dim, rope_half);
    }
    if stage == "v" {
        let z = blk.konst_f16("pz", 0.0);
        let o = blk.add(&v, &z, &[1, kv_dim, 1, 1], "out");
        blk.outputs = vec![o];
        return finish_probe(&pkg_dir, out_dir, &name, &blk, wb, d, kv_dim, rope_half);
    }

    let qnw = st.get_f16(&format!("{p}.self_attn.q_norm.weight"))?;
    let knw = st.get_f16(&format!("{p}.self_attn.k_norm.weight"))?;
    let qn_tiled = tile_head_w(&qnw, nh);
    let kn_tiled = tile_head_w(&knw, nkv);
    let qnoff = wb.put(
        &format!("w_qn_{ls}"),
        DType::Fp16,
        &[q_dim, 1, 1],
        &qn_tiled,
    );
    let qn = blk.konst_blob(
        &format!("w_qn_{ls}"),
        WEIGHTS_NAME,
        qnoff,
        DType::Fp16,
        &[q_dim, 1, 1],
    );
    let knoff = wb.put(
        &format!("w_kn_{ls}"),
        DType::Fp16,
        &[kv_dim, 1, 1],
        &kn_tiled,
    );
    let kn = blk.konst_blob(
        &format!("w_kn_{ls}"),
        WEIGHTS_NAME,
        knoff,
        DType::Fp16,
        &[kv_dim, 1, 1],
    );
    let qr = blk.reshape(&q, &[nh, dh, 1, 1], &format!("l{ls}_qh"));
    let qshape = vec![nh, dh, 1, 1];
    let qn_out = head_rms(&mut blk, &qr, dh, cfg.eps, &qshape, &format!("l{ls}_qn"));
    let qb = blk.reshape(&qn_out, &[1, q_dim, 1, 1], &format!("l{ls}_qnb"));
    let qo = blk.mul(&qb, &qn, &[1, q_dim, 1, 1], &format!("l{ls}_qo"));
    let kr = blk.reshape(&k, &[nkv, dh, 1, 1], &format!("l{ls}_kh"));
    let kshape = vec![nkv, dh, 1, 1];
    let kn_out = head_rms(&mut blk, &kr, dh, cfg.eps, &kshape, &format!("l{ls}_kn"));
    let kb = blk.reshape(&kn_out, &[1, kv_dim, 1, 1], &format!("l{ls}_knb"));
    let ko = blk.mul(&kb, &kn, &[1, kv_dim, 1, 1], &format!("l{ls}_ko"));
    if stage == "qn" {
        let z = blk.konst_f16("pz", 0.0);
        let o = blk.add(&qo, &z, &[1, q_dim, 1, 1], "out");
        blk.outputs = vec![o];
        return finish_probe(&pkg_dir, out_dir, &name, &blk, wb, d, q_dim, rope_half);
    }
    if stage == "kn" {
        let z = blk.konst_f16("pz", 0.0);
        let o = blk.add(&ko, &z, &[1, kv_dim, 1, 1], "out");
        blk.outputs = vec![o];
        return finish_probe(&pkg_dir, out_dir, &name, &blk, wb, d, kv_dim, rope_half);
    }

    let rope = |blk: &mut Block, xf: &str, n_h: i64, dim: i64, total: i64, tag: &str| -> String {
        let xr = blk.reshape(xf, &[1, n_h, dim], &format!("{tag}_r"));
        let rot_end = rope_dim as i32;
        let has_pass = rope_dim < dim;
        let x_rot = if has_pass {
            blk.slice(
                &xr,
                &[0, 0, 0],
                &[1, n_h as i32, rot_end],
                &[1, n_h, rope_dim],
                &format!("{tag}_rot"),
            )
        } else {
            xr.clone()
        };
        let x_pass = if has_pass {
            blk.slice(
                &xr,
                &[0, 0, rot_end],
                &[1, n_h as i32, dim as i32],
                &[1, n_h, dim - rope_dim],
                &format!("{tag}_pass"),
            )
        } else {
            String::new()
        };
        let x_lo = blk.slice(
            &x_rot,
            &[0, 0, 0],
            &[1, n_h as i32, rope_half as i32],
            &[1, n_h, rope_half],
            &format!("{tag}_lo"),
        );
        let x_hi = blk.slice(
            &x_rot,
            &[0, 0, rope_half as i32],
            &[1, n_h as i32, rot_end],
            &[1, n_h, rope_half],
            &format!("{tag}_hi"),
        );
        let cos_b = blk.reshape("rope_cos", &[1, 1, rope_half], &format!("{tag}_cos"));
        let sin_b = blk.reshape("rope_sin", &[1, 1, rope_half], &format!("{tag}_sin"));
        let a = blk.mul(&x_lo, &cos_b, &[1, n_h, rope_half], &format!("{tag}_a"));
        let b = blk.mul(&x_hi, &sin_b, &[1, n_h, rope_half], &format!("{tag}_b"));
        let r_lo = blk.sub(&a, &b, &[1, n_h, rope_half], &format!("{tag}_rlo"));
        let c = blk.mul(&x_lo, &sin_b, &[1, n_h, rope_half], &format!("{tag}_c"));
        let d2 = blk.mul(&x_hi, &cos_b, &[1, n_h, rope_half], &format!("{tag}_d"));
        let r_hi = blk.add(&c, &d2, &[1, n_h, rope_half], &format!("{tag}_rhi"));
        let cat = if has_pass {
            blk.concat(
                &[r_lo, r_hi, x_pass],
                -1,
                &[1, n_h, dim],
                &format!("{tag}_cat"),
            )
        } else {
            blk.concat(&[r_lo, r_hi], -1, &[1, n_h, dim], &format!("{tag}_cat"))
        };
        blk.reshape(&cat, &[1, total, 1, 1], &format!("{tag}_flat"))
    };
    if stage == "kv" {
        // masked state update + read-back: write k into fresh state at pos0,
        // then read_state and return row 0 -> verifies state round-trip.
        let seq = 2048i64;
        let new_k = blk.reshape(&ko, &[1, nkv, 1, dh], "kv_nk");
        let one = blk.konst_f16("kv_one", 1.0);
        let one_m = blk.sub(&one, "kv_write_mask", &[1, 1, seq, 1], "kv_om");
        let kshape = vec![1, nkv, seq, dh];
        let k_old = blk.read_state("k_cache_0", &kshape, "kv_kold");
        let k_keep = blk.mul(&k_old, &one_m, &kshape, "kv_kkeep");
        let k_new = blk.mul(&new_k, "kv_write_mask", &kshape, "kv_knew");
        let k_full = blk.add(&k_keep, &k_new, &kshape, "kv_kfull");
        blk.write_state("k_cache_0", &k_full);
        let k_upd = blk.read_state("k_cache_0", &kshape, "kv_kupd");
        // extract row 0 of the read-back state: slice [0,0,0:dh] -> (1,nkv,1,dh)
        let row0 = blk.slice(
            &k_upd,
            &[0, 0, 0, 0],
            &[1, nkv as i32, 1, dh as i32],
            &[1, nkv, 1, dh],
            "kv_row0",
        );
        let z = blk.konst_f16("pz", 0.0);
        let o = blk.add(&row0, &z, &[1, nkv, 1, dh], "out");
        blk.outputs = vec![o];
        // this probe needs a state + write mask — dedicated finisher
        let fn_inputs = vec![
            NVT {
                name: "x".into(),
                ty: ValueType::Tensor(TensorType::f16(&[1, d, 1, 1])),
            },
            NVT {
                name: "rope_cos".into(),
                ty: ValueType::Tensor(TensorType::f16(&[1, rope_half])),
            },
            NVT {
                name: "rope_sin".into(),
                ty: ValueType::Tensor(TensorType::f16(&[1, rope_half])),
            },
            NVT {
                name: "kv_write_mask".into(),
                ty: ValueType::Tensor(TensorType::f16(&[1, 1, seq, 1])),
            },
            NVT {
                name: "k_cache_0".into(),
                ty: ValueType::State(TensorType::f16(&[1, nkv, seq, dh])),
            },
        ];
        let inputs = vec![
            Feature {
                name: "x".into(),
                shape: vec![1, d, 1, 1],
                dtype: DType::Fp16,
                is_state: false,
            },
            Feature {
                name: "rope_cos".into(),
                shape: vec![1, rope_half],
                dtype: DType::Fp16,
                is_state: false,
            },
            Feature {
                name: "rope_sin".into(),
                shape: vec![1, rope_half],
                dtype: DType::Fp16,
                is_state: false,
            },
            Feature {
                name: "kv_write_mask".into(),
                shape: vec![1, 1, seq, 1],
                dtype: DType::Fp16,
                is_state: false,
            },
        ];
        let states = vec![Feature {
            name: "k_cache_0".into(),
            shape: vec![1, nkv, seq, dh],
            dtype: DType::Fp16,
            is_state: true,
        }];
        let outputs = vec![Feature {
            name: "out".into(),
            shape: vec![1, nkv, 1, dh],
            dtype: DType::Fp16,
            is_state: false,
        }];
        let meta = ModelMeta::new(SPEC_VERSION, OPSET)
            .creator("badapple-aneconvert")
            .description("Bad Apple native ANE shard");
        let spec = encode_model(&inputs, &outputs, &states, &blk, &fn_inputs, &meta);
        let weights = wb.finish();
        write_mlpackage(&pkg_dir, &spec, Some(&weights)).map_err(|e| e.to_string())?;
        return Ok((pkg_dir, out_dir.join(format!("{name}.mlmodelc")), weights));
    }

    let q_r = rope(&mut blk, &qo, nh, dh, q_dim, &format!("l{ls}_rq"));
    if stage == "rq" {
        let z = blk.konst_f16("pz", 0.0);
        let o = blk.add(&q_r, &z, &[1, q_dim, 1, 1], "out");
        blk.outputs = vec![o];
        return finish_probe(&pkg_dir, out_dir, &name, &blk, wb, d, q_dim, rope_half);
    }
    let k_r = rope(&mut blk, &ko, nkv, dh, kv_dim, &format!("l{ls}_rk"));
    if stage == "rk" {
        let z = blk.konst_f16("pz", 0.0);
        let o = blk.add(&k_r, &z, &[1, kv_dim, 1, 1], "out");
        blk.outputs = vec![o];
        return finish_probe(&pkg_dir, out_dir, &name, &blk, wb, d, kv_dim, rope_half);
    }
    Err(format!("unknown probe stage {stage}"))
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn finish_probe(
    pkg_dir: &Path,
    out_dir: &Path,
    name: &str,
    blk: &Block,
    wb: WeightBin,
    d: i64,
    out_dim: i64,
    rope_half: i64,
) -> Result<(PathBuf, PathBuf, Vec<u8>), String> {
    let fn_inputs = vec![
        NVT {
            name: "x".into(),
            ty: ValueType::Tensor(TensorType::f16(&[1, d, 1, 1])),
        },
        NVT {
            name: "rope_cos".into(),
            ty: ValueType::Tensor(TensorType::f16(&[1, rope_half])),
        },
        NVT {
            name: "rope_sin".into(),
            ty: ValueType::Tensor(TensorType::f16(&[1, rope_half])),
        },
    ];
    let inputs = vec![
        Feature {
            name: "x".into(),
            shape: vec![1, d, 1, 1],
            dtype: DType::Fp16,
            is_state: false,
        },
        Feature {
            name: "rope_cos".into(),
            shape: vec![1, rope_half],
            dtype: DType::Fp16,
            is_state: false,
        },
        Feature {
            name: "rope_sin".into(),
            shape: vec![1, rope_half],
            dtype: DType::Fp16,
            is_state: false,
        },
    ];
    let outputs = vec![Feature {
        name: "out".into(),
        shape: vec![1, out_dim, 1, 1],
        dtype: DType::Fp16,
        is_state: false,
    }];
    let meta = ModelMeta::new(SPEC_VERSION, OPSET)
        .creator("badapple-aneconvert")
        .description("Bad Apple native ANE shard");
    let spec = encode_model(&inputs, &outputs, &[], blk, &fn_inputs, &meta);
    let weights = wb.finish();
    write_mlpackage(pkg_dir, &spec, Some(&weights)).map_err(|e| e.to_string())?;
    Ok((
        pkg_dir.to_path_buf(),
        out_dir.join(format!("{name}.mlmodelc")),
        weights,
    ))
}

// ======== compile driver ========

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut model_dir = PathBuf::from("models/qwen3-4b");
    let mut out_dir = PathBuf::from("tests/ane_brain_perf/artifacts/qwen3b_ane_shards");
    let mut seq: i64 = 2048;
    let mut lps: i64 = 1;
    let mut head_shards: i64 = 4;
    let mut only_layers: Option<(i64, i64)> = None;
    let mut no_compile = false;
    let mut keep_packages = false;
    let mut probe_stage: Option<String> = None;
    let mut prefill_chunk: i64 = 0;
    // Decode shard token width. s=1 emits degenerate dims that the ANE
    // compiler miscompiles (broadcast write masks produce garbage at
    // unwritten KV slots; dim2=1 attention masks misalign). s=2 routes a
    // real token + dummy slot through the proven s>1 path.
    let mut decode_width: i64 = 1;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => {
                model_dir = PathBuf::from(&args[i + 1]);
                i += 1;
            }
            "--out" => {
                out_dir = PathBuf::from(&args[i + 1]);
                i += 1;
            }
            "--seq-len" => {
                seq = args[i + 1].parse().unwrap_or(2048);
                i += 1;
            }
            "--layers-per-shard" => {
                lps = args[i + 1].parse().unwrap_or(1);
                i += 1;
            }
            "--lm-head-shards" => {
                head_shards = args[i + 1].parse().unwrap_or(4);
                i += 1;
            }
            "--weight-bits" => {
                let bits: u32 = args[i + 1].parse().unwrap_or(16);
                if bits != 4 && bits != 8 && bits != 16 {
                    eprintln!("--weight-bits must be 4, 8, or 16");
                    std::process::exit(2);
                }
                let _ = WEIGHT_BITS.set(bits);
                i += 1;
            }
            "--layers" => {
                let r: Vec<i64> = args[i + 1]
                    .split('-')
                    .map(|s| s.parse().unwrap_or(0))
                    .collect();
                if r.len() == 2 {
                    only_layers = Some((r[0], r[1]));
                }
                i += 1;
            }
            "--no-compile" => no_compile = true,
            "--keep-packages" => keep_packages = true,
            "--probe" => {
                probe_stage = Some(args[i + 1].clone());
                i += 1;
            }
            "--prefill-chunk" => {
                prefill_chunk = args[i + 1].parse().unwrap_or(0);
                i += 1;
            }
            "--decode-width" => {
                decode_width = args[i + 1].parse().unwrap_or(1);
                i += 1;
            }
            _ => {}
        }
        i += 1;
    }

    // BADAPPLE_SPLIT_FFN: each layer becomes two models — a stateful
    // attention shard emitting concat(res1, fn_norm) and a stateless
    // ffn_s shard computing the conv tail on a real model input. The
    // monolithic norm→conv chain miscompiles inside stateful plans on
    // multi-token decode (verified: state contents and res1 are exact
    // until the conv exists downstream in the same plan). Split forces
    // lps=1: an FFN shard can only consume its own layer's pair.
    let split_ffn = std::env::var("BADAPPLE_SPLIT_FFN").is_ok();
    if split_ffn {
        lps = 1;
    }

    eprintln!(
        "🏴‍☠️  BAD APPLE // aneconvert: {:?} -> {:?}",
        model_dir, out_dir
    );
    let cfg = match Cfg::from_config(&model_dir) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("config: {e}");
            std::process::exit(1);
        }
    };
    eprintln!(
        "  d={} L={} nh={} nkv={} dh={} dff={} vocab={} ropeθ={}",
        cfg.d_model,
        cfg.n_layers,
        cfg.n_heads,
        cfg.n_kv,
        cfg.d_head,
        cfg.d_ff,
        cfg.vocab,
        cfg.rope_theta
    );
    // Early safetensors validation; the .cy side re-indexes per extern call.
    if let Err(e) = StIndex::load(&model_dir) {
        eprintln!("safetensors: {e}");
        std::process::exit(1);
    }

    // The whole convert run is authored in Coyote (coyote/layer.cy):
    // embedding emit, layer/prefill/ffn/head shard loops, coremlc
    // compilation, and the manifest are all `mil_convert_run`. Rust keeps
    // argv parsing, the config load, and the exit code.
    let cfgj = layer_cfg_json(&cfg);
    let opts = json!({
        "seq": seq,
        "lps": lps,
        "head_shards": head_shards,
        "only_lo": only_layers.map(|(a, _)| a).unwrap_or(-1),
        "only_hi": only_layers.map(|(_, b)| b).unwrap_or(-1),
        "no_compile": no_compile,
        "keep_packages": keep_packages,
        "prefill_chunk": prefill_chunk,
        "decode_width": decode_width,
        "split_ffn": split_ffn,
        "probe_stage": probe_stage.clone().unwrap_or_default(),
        "probe_layer": only_layers.map(|(a, _)| a).unwrap_or(0),
    });
    let dir = model_dir.to_string_lossy().to_string();
    let out = out_dir.to_string_lossy().to_string();
    match bad_apple::coyote_sys::mil_convert_run(&cfgj, &dir, &out, &opts) {
        Ok(v) => {
            if let Some(p) = v.get("probe") {
                eprintln!("  probe result: {}", p["status"].as_str().unwrap_or("?"));
                if let Some(e) = p.get("error").and_then(|e| e.as_str()) {
                    eprintln!("    {e}");
                }
                return;
            }
            let ok = v["all_ok"].as_bool().unwrap_or(false);
            eprintln!("manifest: {}", if ok { "complete" } else { "partial" });
            if !ok {
                std::process::exit(1);
            }
        }
        Err(e) => {
            eprintln!("convert failed: {e}");
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Synthetic single-layer safetensors fixture — deterministic F16
    /// contents (parity needs identical bytes, not real weights).
    fn write_fixture(dir: &Path, d: usize, nh: usize, nkv: usize, dh: usize, dff: usize) {
        let (q_dim, kv_dim) = (nh * dh, nkv * dh);
        let mk = |shape: &[usize]| -> Vec<u8> {
            let n: usize = shape.iter().product();
            let mut b = Vec::with_capacity(n * 2);
            for i in 0..n {
                // Small well-conditioned fp16 bits — never inf/nan.
                let bits = 0x2000u16 + (i % 61) as u16 * 13;
                b.extend_from_slice(&bits.to_le_bytes());
            }
            b
        };
        let p = "model.layers.0";
        let tensors: Vec<(String, Vec<u64>, Vec<u8>)> = vec![
            (
                format!("{p}.input_layernorm.weight"),
                vec![d as u64],
                mk(&[d]),
            ),
            (
                format!("{p}.self_attn.q_proj.weight"),
                vec![q_dim as u64, d as u64],
                mk(&[q_dim, d]),
            ),
            (
                format!("{p}.self_attn.k_proj.weight"),
                vec![kv_dim as u64, d as u64],
                mk(&[kv_dim, d]),
            ),
            (
                format!("{p}.self_attn.v_proj.weight"),
                vec![kv_dim as u64, d as u64],
                mk(&[kv_dim, d]),
            ),
            (
                format!("{p}.self_attn.q_norm.weight"),
                vec![dh as u64],
                mk(&[dh]),
            ),
            (
                format!("{p}.self_attn.k_norm.weight"),
                vec![dh as u64],
                mk(&[dh]),
            ),
            (
                format!("{p}.self_attn.o_proj.weight"),
                vec![d as u64, q_dim as u64],
                mk(&[d, q_dim]),
            ),
            (
                format!("{p}.post_attention_layernorm.weight"),
                vec![d as u64],
                mk(&[d]),
            ),
            (
                format!("{p}.mlp.gate_proj.weight"),
                vec![dff as u64, d as u64],
                mk(&[dff, d]),
            ),
            (
                format!("{p}.mlp.up_proj.weight"),
                vec![dff as u64, d as u64],
                mk(&[dff, d]),
            ),
            (
                format!("{p}.mlp.down_proj.weight"),
                vec![d as u64, dff as u64],
                mk(&[d, dff]),
            ),
            ("model.norm.weight".to_string(), vec![d as u64], mk(&[d])),
            (
                "model.embed_tokens.weight".to_string(),
                vec![64, d as u64],
                mk(&[64, d]),
            ),
            (
                "lm_head.weight".to_string(),
                vec![64, d as u64],
                mk(&[64, d]),
            ),
        ];
        let mut hdr = serde_json::Map::new();
        let mut blob = Vec::new();
        for (name, shape, data) in &tensors {
            let begin = blob.len();
            blob.extend_from_slice(data);
            hdr.insert(
                name.clone(),
                json!({
                    "dtype": "F16",
                    "shape": shape,
                    "data_offsets": [begin, blob.len()],
                }),
            );
        }
        let hbuf = serde_json::to_vec(&Value::Object(hdr)).unwrap();
        let mut out = (hbuf.len() as u64).to_le_bytes().to_vec();
        out.extend_from_slice(&hbuf);
        out.extend_from_slice(&blob);
        std::fs::write(dir.join("model.safetensors"), out).unwrap();
    }

    fn qwen_cfg_json(cfg: &Cfg) -> Value {
        json!({
            "arch": cfg.arch_name(),
            "d_model": cfg.d_model,
            "n_heads": cfg.n_heads,
            "n_kv": cfg.n_kv,
            "d_head": cfg.d_head,
            "rope_dim": cfg.rope_dim,
            "d_ff": cfg.d_ff,
            "eps": cfg.eps as f64,
            "attn_scale": cfg.attn_scale as f64,
            "weight_bits": 16,
            "norm_plus_one": cfg.norm_plus_one,
            "has_qk_norm": cfg.has_qk_norm,
            "rope_theta": cfg.rope_theta,
            "rope_local_theta": cfg.rope_local_theta,
            "sliding_window": cfg.sliding_window,
            "sliding_pattern": cfg.sliding_pattern,
            "down_prescale": 1.0,
        })
    }

    fn layer_parity(s: i64) {
        let dir = std::env::temp_dir().join(format!("ba_cy_parity_{s}_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (d, nh, nkv, dh, dff, seq) = (16usize, 2usize, 1usize, 8usize, 32usize, 16i64);
        write_fixture(&dir, d, nh, nkv, dh, dff);
        let st = StIndex::load(&dir).unwrap();
        let cfg = Cfg {
            arch: Arch::Qwen3,
            d_model: d as i64,
            n_layers: 1,
            n_heads: nh as i64,
            n_kv: nkv as i64,
            d_head: dh as i64,
            d_ff: dff as i64,
            vocab: 64,
            rope_theta: 1e6,
            rope_local_theta: 1e6,
            eps: 1e-6,
            rope_dim: dh as i64,
            has_qk_norm: true,
            tie_embeddings: false,
            attn_scale: 1.0 / (dh as f32).sqrt(),
            sliding_window: 0,
            sliding_pattern: 0,
            norm_plus_one: false,
            embed_scale: 1.0,
        };
        let out_dir = dir.join("out");
        std::fs::create_dir_all(&out_dir).unwrap();
        let (pkg, _mlc, w_rust) =
            emit_layer_shard_rust(&st, &cfg, 0, 1, seq, &out_dir, s, false, 16).expect("rust emit");
        let spec_rust =
            std::fs::read(pkg.join("Data/com.apple.CoreML/model.mlmodel")).expect("spec bytes");
        let (spec_cy, w_cy) = bad_apple::coyote_sys::mil_layer_emit(
            &qwen_cfg_json(&cfg),
            dir.to_str().unwrap(),
            0,
            1,
            seq,
            s,
            false,
        )
        .expect("coyote emit");
        assert_eq!(spec_cy, spec_rust, "s={s}: .cy spec != rust spec");
        assert_eq!(w_cy, w_rust, "s={s}: .cy weight.bin != rust weight.bin");
    }

    /// Quantized weight paths: layer decode at bits=8 (constexpr
    /// blockwise_shift_scale + the down-projection prescale α ops) and
    /// bits=4 (constexpr_lut_to_dense). Explicit `bits` params keep this
    /// deterministic under parallel test threads — no env globals.
    fn quant_parity(bits: u32) {
        let dir = std::env::temp_dir().join(format!("ba_cy_q{bits}_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (d, nh, nkv, dh, dff, seq) = (16usize, 2usize, 1usize, 8usize, 32usize, 16i64);
        write_fixture(&dir, d, nh, nkv, dh, dff);
        let st = StIndex::load(&dir).unwrap();
        let cfg = Cfg {
            arch: Arch::Qwen3,
            d_model: d as i64,
            n_layers: 1,
            n_heads: nh as i64,
            n_kv: nkv as i64,
            d_head: dh as i64,
            d_ff: dff as i64,
            vocab: 64,
            rope_theta: 1e6,
            rope_local_theta: 1e6,
            eps: 1e-6,
            rope_dim: dh as i64,
            has_qk_norm: true,
            tie_embeddings: false,
            attn_scale: 1.0 / (dh as f32).sqrt(),
            sliding_window: 0,
            sliding_pattern: 0,
            norm_plus_one: false,
            embed_scale: 1.0,
        };
        let out_dir = dir.join("out");
        std::fs::create_dir_all(&out_dir).unwrap();
        let (pkg, _mlc, w_rust) =
            emit_layer_shard_rust(&st, &cfg, 0, 1, seq, &out_dir, 1, false, bits)
                .expect("rust emit");
        let spec_rust =
            std::fs::read(pkg.join("Data/com.apple.CoreML/model.mlmodel")).expect("spec bytes");
        let mut cfgj = qwen_cfg_json(&cfg);
        cfgj["weight_bits"] = json!(bits);
        cfgj["down_prescale"] = json!(down_prescale_for(bits) as f64);
        let (spec_cy, w_cy) = bad_apple::coyote_sys::mil_layer_emit(
            &cfgj,
            dir.to_str().unwrap(),
            0,
            1,
            seq,
            1,
            false,
        )
        .expect("coyote emit");
        assert_eq!(spec_cy, spec_rust, "bits={bits}: .cy spec != rust spec");
        assert_eq!(w_cy, w_rust, "bits={bits}: .cy weight.bin != rust");
    }

    #[test]
    fn mil_cy_layer_q8_parity() {
        quant_parity(8);
    }

    #[test]
    fn mil_cy_layer_q4_parity() {
        quant_parity(4);
    }

    #[test]
    fn mil_cy_layer_decode_parity() {
        layer_parity(1);
    }

    #[test]
    fn mil_cy_layer_prefill_parity() {
        layer_parity(4);
    }

    fn ffn_parity(s: i64) {
        let dir = std::env::temp_dir().join(format!("ba_cy_ffn_{s}_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (d, nh, nkv, dh, dff) = (16usize, 2usize, 1usize, 8usize, 32usize);
        write_fixture(&dir, d, nh, nkv, dh, dff);
        let st = StIndex::load(&dir).unwrap();
        let cfg = Cfg {
            arch: Arch::Qwen3,
            d_model: d as i64,
            n_layers: 1,
            n_heads: nh as i64,
            n_kv: nkv as i64,
            d_head: dh as i64,
            d_ff: dff as i64,
            vocab: 64,
            rope_theta: 1e6,
            rope_local_theta: 1e6,
            eps: 1e-6,
            rope_dim: dh as i64,
            has_qk_norm: true,
            tie_embeddings: false,
            attn_scale: 1.0 / (dh as f32).sqrt(),
            sliding_window: 0,
            sliding_pattern: 0,
            norm_plus_one: false,
            embed_scale: 1.0,
        };
        let out_dir = dir.join("out");
        std::fs::create_dir_all(&out_dir).unwrap();
        let (pkg, _mlc, w_rust) =
            emit_ffn_shard_rust(&st, &cfg, 0, 1, &out_dir, s, s != 1, 16).expect("rust emit");
        let spec_rust =
            std::fs::read(pkg.join("Data/com.apple.CoreML/model.mlmodel")).expect("spec bytes");
        let (spec_cy, w_cy) = bad_apple::coyote_sys::mil_ffn_emit(
            &qwen_cfg_json(&cfg),
            dir.to_str().unwrap(),
            0,
            1,
            s,
        )
        .expect("coyote emit");
        assert_eq!(spec_cy, spec_rust, "s={s}: .cy spec != rust spec");
        assert_eq!(w_cy, w_rust, "s={s}: .cy weight.bin != rust weight.bin");
    }

    #[test]
    fn mil_cy_ffn_decode_parity() {
        ffn_parity(1);
    }

    #[test]
    fn mil_cy_ffn_prefill_parity() {
        ffn_parity(4);
    }

    #[test]
    fn mil_cy_head_parity() {
        let dir = std::env::temp_dir().join(format!("ba_cy_head_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (d, nh, nkv, dh, dff) = (16usize, 2usize, 1usize, 8usize, 32usize);
        write_fixture(&dir, d, nh, nkv, dh, dff);
        let st = StIndex::load(&dir).unwrap();
        let cfg = Cfg {
            arch: Arch::Qwen3,
            d_model: d as i64,
            n_layers: 1,
            n_heads: nh as i64,
            n_kv: nkv as i64,
            d_head: dh as i64,
            d_ff: dff as i64,
            vocab: 64,
            rope_theta: 1e6,
            rope_local_theta: 1e6,
            eps: 1e-6,
            rope_dim: dh as i64,
            has_qk_norm: true,
            tie_embeddings: false,
            attn_scale: 1.0 / (dh as f32).sqrt(),
            sliding_window: 0,
            sliding_pattern: 0,
            norm_plus_one: false,
            embed_scale: 1.0,
        };
        let out_dir = dir.join("out");
        std::fs::create_dir_all(&out_dir).unwrap();
        let (pkg, _mlc) = emit_head_shard_rust(&st, &cfg, 0, 32, &out_dir, 16).expect("rust emit");
        let spec_rust =
            std::fs::read(pkg.join("Data/com.apple.CoreML/model.mlmodel")).expect("spec bytes");
        let w_rust =
            std::fs::read(pkg.join("Data/com.apple.CoreML/weights/weight.bin")).expect("weights");
        let (spec_cy, w_cy) = bad_apple::coyote_sys::mil_head_emit(
            &qwen_cfg_json(&cfg),
            dir.to_str().unwrap(),
            0,
            32,
        )
        .expect("coyote emit");
        assert_eq!(spec_cy, spec_rust, ".cy head spec != rust spec");
        assert_eq!(w_cy, w_rust, ".cy head weight.bin != rust weight.bin");
    }

    #[test]
    fn mil_cy_probe_parity() {
        let dir = std::env::temp_dir().join(format!("ba_cy_probe_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (d, nh, nkv, dh, dff) = (16usize, 2usize, 1usize, 8usize, 32usize);
        write_fixture(&dir, d, nh, nkv, dh, dff);
        let st = StIndex::load(&dir).unwrap();
        let cfg = Cfg {
            arch: Arch::Qwen3,
            d_model: d as i64,
            n_layers: 1,
            n_heads: nh as i64,
            n_kv: nkv as i64,
            d_head: dh as i64,
            d_ff: dff as i64,
            vocab: 64,
            rope_theta: 1e6,
            rope_local_theta: 1e6,
            eps: 1e-6,
            rope_dim: dh as i64,
            has_qk_norm: true,
            tie_embeddings: false,
            attn_scale: 1.0 / (dh as f32).sqrt(),
            sliding_window: 0,
            sliding_pattern: 0,
            norm_plus_one: false,
            embed_scale: 1.0,
        };
        let out_dir = dir.join("out");
        std::fs::create_dir_all(&out_dir).unwrap();
        // Deepest stages: rq (norm+qkv+slice+head_rms+rope) and kv (state round-trip).
        for stage in ["an", "qn", "rq", "kv"] {
            let (pkg, _mlc, w_rust) =
                emit_probe_rust(&st, &cfg, 0, stage, &out_dir).expect("rust probe");
            let spec_rust =
                std::fs::read(pkg.join("Data/com.apple.CoreML/model.mlmodel")).expect("spec bytes");
            let (spec_cy, w_cy) = bad_apple::coyote_sys::mil_probe_emit(
                &qwen_cfg_json(&cfg),
                dir.to_str().unwrap(),
                0,
                stage,
            )
            .expect("coyote probe");
            assert_eq!(spec_cy, spec_rust, "stage={stage}: .cy spec != rust spec");
            assert_eq!(w_cy, w_rust, "stage={stage}: .cy weight.bin != rust");
        }
    }

    /// End-to-end: mil_convert_run drives the whole pipeline in-language —
    /// embedding emit, layer shard, prefill shard, lm-head shard, manifest —
    /// with compile skipped. Asserts the manifest schema/statuses.
    #[test]
    fn mil_cy_convert_run_no_compile() {
        let dir = std::env::temp_dir().join(format!("ba_cy_run_{}", std::process::id()));
        let out = dir.join("artifacts");
        std::fs::create_dir_all(&dir).unwrap();
        let (d, nh, nkv, dh, dff) = (16usize, 2usize, 1usize, 8usize, 32usize);
        write_fixture(&dir, d, nh, nkv, dh, dff);
        let cfg = Cfg {
            arch: Arch::Qwen3,
            d_model: d as i64,
            n_layers: 1,
            n_heads: nh as i64,
            n_kv: nkv as i64,
            d_head: dh as i64,
            d_ff: dff as i64,
            vocab: 64,
            rope_theta: 1e6,
            rope_local_theta: 1e6,
            eps: 1e-6,
            rope_dim: dh as i64,
            has_qk_norm: true,
            tie_embeddings: false,
            attn_scale: 1.0 / (dh as f32).sqrt(),
            sliding_window: 0,
            sliding_pattern: 0,
            norm_plus_one: false,
            embed_scale: 1.0,
        };
        let opts = json!({
            "seq": 16,
            "lps": 1,
            "head_shards": 2,
            "only_lo": -1,
            "only_hi": -1,
            "no_compile": true,
            "keep_packages": true,
            "prefill_chunk": 4,
            "decode_width": 1,
            "split_ffn": false,
            "probe_stage": "",
            "probe_layer": 0,
        });
        let v = bad_apple::coyote_sys::mil_convert_run(
            &layer_cfg_json(&cfg),
            dir.to_str().unwrap(),
            out.to_str().unwrap(),
            &opts,
        )
        .expect("mil_convert_run");
        assert_eq!(v["all_ok"].as_bool(), Some(true), "run: {v}");

        let manifest: Value = serde_json::from_slice(
            &std::fs::read(out.join("conversion_manifest.json")).expect("manifest file"),
        )
        .expect("manifest json");
        assert_eq!(manifest["status"].as_str(), Some("complete"));
        assert_eq!(manifest["schema_version"].as_i64(), Some(1));

        let shards = manifest["shards"].as_array().expect("shards");
        assert_eq!(shards.len(), 1);
        assert_eq!(shards[0]["status"].as_str(), Some("packaged"));
        assert_eq!(shards[0]["layer_start"].as_i64(), Some(0));
        assert_eq!(shards[0]["layer_end"].as_i64(), Some(1));

        let prefill = manifest["shards_prefill"]
            .as_array()
            .expect("prefill shards");
        assert_eq!(prefill.len(), 1);
        assert_eq!(prefill[0]["status"].as_str(), Some("packaged"));

        let heads = manifest["shared"]["lm_head_shards"]
            .as_array()
            .expect("head shards");
        assert_eq!(heads.len(), 2);
        assert_eq!(heads[0]["status"].as_str(), Some("packaged"));
        assert_eq!(heads[0]["vocab_start"].as_i64(), Some(0));
        assert_eq!(heads[0]["vocab_end"].as_i64(), Some(32));
        assert_eq!(heads[1]["vocab_start"].as_i64(), Some(32));
        assert_eq!(heads[1]["vocab_end"].as_i64(), Some(64));

        assert_eq!(
            manifest["shared"]["embedding"]["status"].as_str(),
            Some("complete")
        );
        assert_eq!(
            std::fs::metadata(out.join("embedding.f16")).unwrap().len() as i64,
            64 * d as i64 * 2
        );
        // Packages were emitted.
        assert!(out
            .join("layer_s00-01.mlpackage/Data/com.apple.CoreML/model.mlmodel")
            .is_file());
        assert!(out
            .join("prefill_s00-01.mlpackage/Data/com.apple.CoreML/model.mlmodel")
            .is_file());
        assert!(out
            .join("head_v00000-00032.mlpackage/Data/com.apple.CoreML/model.mlmodel")
            .is_file());
    }
}
