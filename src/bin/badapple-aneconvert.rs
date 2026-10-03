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

use bad_apple::mil_spec::*;
use serde_json::{json, Map, Value};
use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::process::Command;

// ======== safetensors (header-only index + on-demand slices) ========

#[derive(Clone, Debug)]
struct TensorInfo {
    dtype: String,
    shape: Vec<usize>,
    file: PathBuf,
    begin: u64,
    end: u64,
}

struct StIndex {
    tensors: HashMap<String, TensorInfo>,
}

impl StIndex {
    fn load(dir: &Path) -> Result<Self, String> {
        let mut tensors = HashMap::new();
        // Index file maps tensor name -> shard file; single-file models work too.
        let idx_path = dir.join("model.safetensors.index.json");
        let mut files: Vec<PathBuf> = vec![];
        if idx_path.is_file() {
            let idx: Value =
                serde_json::from_slice(&std::fs::read(&idx_path).map_err(|e| e.to_string())?)
                    .map_err(|e| e.to_string())?;
            let wm = idx["weight_map"].as_object().ok_or("weight_map missing")?;
            for f in wm.values() {
                let p = dir.join(f.as_str().ok_or("bad weight_map entry")?);
                if !files.contains(&p) {
                    files.push(p);
                }
            }
        } else {
            let single = dir.join("model.safetensors");
            if !single.is_file() {
                return Err(format!(
                    "no safetensors index or model.safetensors in {}",
                    dir.display()
                ));
            }
            files.push(single);
        }
        for file in files {
            let mut fh = std::fs::File::open(&file).map_err(|e| format!("{e} {file:?}"))?;
            let mut lenb = [0u8; 8];
            fh.read_exact(&mut lenb).map_err(|e| e.to_string())?;
            let hlen = u64::from_le_bytes(lenb) as usize;
            let mut hbuf = vec![0u8; hlen];
            fh.read_exact(&mut hbuf).map_err(|e| e.to_string())?;
            let hdr: Value = serde_json::from_slice(&hbuf).map_err(|e| e.to_string())?;
            let obj = hdr.as_object().ok_or("bad header")?;
            for (name, meta) in obj {
                if name == "__metadata__" {
                    continue;
                }
                let dtype = meta["dtype"].as_str().unwrap_or("F16").to_string();
                let shape: Vec<usize> = meta["shape"]
                    .as_array()
                    .unwrap_or(&vec![])
                    .iter()
                    .map(|v| v.as_u64().unwrap_or(0) as usize)
                    .collect();
                let offs = meta["data_offsets"].as_array().ok_or("no data_offsets")?;
                let begin = offs[0].as_u64().unwrap_or(0);
                let end = offs[1].as_u64().unwrap_or(0);
                let base = 8 + hlen as u64;
                tensors.insert(
                    name.clone(),
                    TensorInfo {
                        dtype,
                        shape,
                        file: file.clone(),
                        begin: base + begin,
                        end: base + end,
                    },
                );
            }
        }
        Ok(StIndex { tensors })
    }

    fn get_raw(&self, name: &str) -> Result<Vec<u8>, String> {
        let t = self
            .tensors
            .get(name)
            .ok_or_else(|| format!("missing tensor {name}"))?;
        let mut fh = std::fs::File::open(&t.file).map_err(|e| e.to_string())?;
        fh.seek(SeekFrom::Start(t.begin))
            .map_err(|e| e.to_string())?;
        let mut buf = vec![0u8; (t.end - t.begin) as usize];
        fh.read_exact(&mut buf).map_err(|e| e.to_string())?;
        Ok(buf)
    }

    /// Read a tensor as fp16 bytes (converting bf16/f32 sources).
    fn get_f16(&self, name: &str) -> Result<Vec<u8>, String> {
        let t = self
            .tensors
            .get(name)
            .ok_or_else(|| format!("missing tensor {name}"))?;
        let raw = self.get_raw(name)?;
        match t.dtype.as_str() {
            "F16" => Ok(raw),
            "BF16" => Ok(bf16_bytes_to_f16(&raw)),
            "F32" => Ok(f32_bytes_to_f16(&raw)),
            other => Err(format!("{name}: unsupported dtype {other}")),
        }
    }
}

fn bf16_bytes_to_f16(raw: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(raw.len());
    for c in raw.chunks_exact(2) {
        let b = half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]]));
        out.extend_from_slice(&half::f16::from_f32(b.to_f32()).to_le_bytes());
    }
    out
}

fn f32_bytes_to_f16(raw: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(raw.len() / 2);
    for c in raw.chunks_exact(4) {
        let v = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
        out.extend_from_slice(&half::f16::from_f32(v).to_le_bytes());
    }
    out
}

// ======== model config ========

#[derive(Clone)]
struct Cfg {
    d_model: i64,
    n_layers: i64,
    n_heads: i64,
    n_kv: i64,
    d_head: i64,
    d_ff: i64,
    vocab: i64,
    rope_theta: f64,
    eps: f32,
    rope_dim: i64,
    has_qk_norm: bool,
    tie_embeddings: bool,
}

impl Cfg {
    fn from_config(dir: &Path) -> Result<Self, String> {
        let raw = std::fs::read_to_string(dir.join("config.json")).map_err(|e| e.to_string())?;
        let c: Value = serde_json::from_str(&raw).map_err(|e| e.to_string())?;
        let g = |k: &str| c[k].as_i64().unwrap_or(0);
        let d_model = g("hidden_size");
        let n_heads = g("num_attention_heads");
        let d_head =
            c["head_dim"]
                .as_i64()
                .unwrap_or(if n_heads > 0 { d_model / n_heads } else { 0 });
        Ok(Cfg {
            d_model,
            n_layers: g("num_hidden_layers"),
            n_heads,
            n_kv: g("num_key_value_heads"),
            d_head,
            d_ff: g("intermediate_size"),
            vocab: g("vocab_size"),
            rope_theta: c["rope_theta"].as_f64().unwrap_or(1_000_000.0),
            eps: c["rms_norm_eps"].as_f64().unwrap_or(1e-6) as f32,
            rope_dim: d_head,
            has_qk_norm: true, // Qwen3 always has q_norm/k_norm
            tie_embeddings: c["tie_word_embeddings"].as_bool().unwrap_or(false),
        })
    }
}

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
fn quantize_q8_rows(fp16: &[u8], rows: usize) -> (Vec<u8>, Vec<u8>) {
    let cols = fp16.len() / 2 / rows;
    let mut q = Vec::with_capacity(rows * cols);
    let mut scales = Vec::with_capacity(rows * 2);
    let mut row = vec![0f32; cols];
    for r in 0..rows {
        let base = r * cols * 2;
        let mut amax = 0f32;
        for (c, v) in row.iter_mut().enumerate() {
            let o = base + c * 2;
            *v = half::f16::from_le_bytes([fp16[o], fp16[o + 1]]).to_f32();
            amax = amax.max(v.abs());
        }
        // Round the scale to fp16 first so quantization uses the exact value
        // the ANE will multiply by.
        let scale = half::f16::from_f32(if amax > 0.0 { amax / 127.0 } else { 1.0 });
        let s = scale.to_f32();
        q.extend(
            row.iter()
                .map(|v| ((v / s).round().clamp(-127.0, 127.0) as i8) as u8),
        );
        scales.extend_from_slice(&scale.to_le_bytes());
    }
    (q, scales)
}

/// α for the down_proj input pre-scale (int8 only; 1 = off). `BADAPPLE_Q8_DOWN_PRESCALE`.
fn down_prescale() -> f32 {
    if weight_bits() != 8 {
        return 1.0;
    }
    // 64 measured exact (cos 0.99997 vs CPU FP16 over 12 layers) with headroom
    // for deeper layers; 16–256 all behaved identically.
    std::env::var("BADAPPLE_Q8_DOWN_PRESCALE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(64.0)
}

/// Emit a matmul weight [cout, cin, 1, 1] as fp16 or int8 per `weight_bits()`.
fn weight_const(
    wb: &mut WeightBin,
    blk: &mut Block,
    name: &str,
    fp16: &[u8],
    shape: &[i64],
) -> String {
    if weight_bits() == 8 {
        let (q, s) = quantize_q8_rows(fp16, shape[0] as usize);
        let mut scale_shape = vec![shape[0]];
        scale_shape.extend(std::iter::repeat(1).take(shape.len() - 1));
        let qoff = wb.put(&format!("{name}_q8"), DType::Int8, shape, &q);
        let soff = wb.put(&format!("{name}_scale"), DType::Fp16, &scale_shape, &s);
        blk.konst_q8(name, WEIGHTS_NAME, qoff, soff, shape)
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
const SPEC_VERSION: i32 = 9;
const OPSET: &str = "CoreML8";
const WEIGHTS_NAME: &str = "@model_path/weights/weight.bin";

/// tile a (dh,) weight to (n*dh) elements for conv-broadcast use
fn tile_head_w(bytes: &[u8], n: i64) -> Vec<u8> {
    let mut out = Vec::with_capacity(bytes.len() * n as usize);
    for _ in 0..n {
        out.extend_from_slice(bytes);
    }
    out
}

/// 1x1 conv whose output spatial width is `s`. `conv1x1` hardcodes width 1;
/// prefill shards (s>1) declare the real shape so MIL type-checks.
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
            vec![("val".into(), bad_apple::mil_spec::Value::Str("valid".into()))],
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

/// Emit one stateful transformer-layer shard covering layers [ls, le).
/// `s` is the token width per forward: 1 = decode shard, >1 = prefill shard
/// (same state names/shapes so it shares MLState with the decode shards).
fn emit_layer_shard(
    st: &StIndex,
    cfg: &Cfg,
    ls: i64,
    le: i64,
    seq: i64,
    out_dir: &Path,
    s: i64,
    is_prefill: bool,
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
    // Packed KV state: ONE buffer per shard [2*nl, nkv, seq, dh], rows
    // 2l (K) and 2l+1 (V), chained through every layer's slice_update and
    // committed once per call. One state object total (the documented
    // recipe) — and K^T for attention comes from matmul's fused
    // transpose_y, because a standalone transpose of a
    // slice_update-derived state slice fails ANE plan-build -14
    // (verified by mil_kvpack).
    let kvshape: Vec<i64> = vec![2 * n_state as i64, nkv, seq, dh];
    let mut cur_kv = blk.read_state("kv_cache", &kvshape, "kv_init");

    for i in ls..le {
        let p = format!("model.layers.{i}");

        // ---- attn_norm ----
        let w = st.get_f16(&format!("{p}.input_layernorm.weight"))?;
        let wn = blob(
            &mut wb,
            &mut blk,
            &format!("w_attn_norm_{i}"),
            &w,
            &[d, 1, 1],
        );
        let normed = blk.rms_norm(&cur, &wn, d, cfg.eps, &shape_x, &format!("l{i}_an"));

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
        );
        let qkv = conv1x1_w(
            &mut blk,
            &normed,
            &wqkv,
            q_dim + 2 * kv_dim,
            s,
            &format!("l{i}_qkv"),
        );

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

        // ---- per-head q/k RMSNorm (Qwen3) ----
        let (qn, kn) = if cfg.has_qk_norm {
            let qnw = st.get_f16(&format!("{p}.self_attn.q_norm.weight"))?;
            let knw = st.get_f16(&format!("{p}.self_attn.k_norm.weight"))?;
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
                let a = blk.mul(&x_lo, "rope_cos", &[1, n_h, rope_half, s], &format!("{tag}_a"));
                let b = blk.mul(&x_hi, "rope_sin", &[1, n_h, rope_half, s], &format!("{tag}_b"));
                let r_lo = blk.sub(&a, &b, &[1, n_h, rope_half, s], &format!("{tag}_rlo"));
                let c = blk.mul(&x_lo, "rope_sin", &[1, n_h, rope_half, s], &format!("{tag}_c"));
                let d2 = blk.mul(&x_hi, "rope_cos", &[1, n_h, rope_half, s], &format!("{tag}_d"));
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
        let (q_r, k_r) = if s == 1 {
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
                    vec![
                        ("x".into(), bind(&pos_i32).1),
                        ("y".into(), bind(&c).1),
                    ],
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
                vec![("val".into(), mil_spec::Value::bools(&[false, false, false, false]))],
            )[0]
            .clone();
            // K row: write [1,nkv,s,dh] at [2l,0,pos,0]..[2l+1,nkv,pos+s,dh]
            let kb = concat4(&mut blk, &format!("l{i}_kb"), &[ck.as_str(), c0.as_str(), pos_i32.as_str(), c0.as_str()]);
            let ke = concat4(&mut blk, &format!("l{i}_ke"), &[ck1.as_str(), cnkv.as_str(), pos_s.as_str(), cdh.as_str()]);
            cur_kv = blk.o1(
                "slice_update",
                vec![
                    ("x".into(), bind(&cur_kv).1),
                    ("update".into(), bind(&new_k).1),
                    ("begin".into(), bind(&kb).1),
                    ("end".into(), bind(&ke).1),
                    ("stride".into(), bind(&st4).1),
                    ("begin_mask".into(), bind(&bm4).1),
                    ("end_mask".into(), bind(&bm4).1),
                    ("squeeze_mask".into(), bind(&bm4).1),
                ],
                &format!("l{i}_kup"),
                ValueType::Tensor(TensorType::f16(&kvshape)),
            );
            // V row: write [1,nkv,s,dh] at [2l+1,0,pos,0]..[2l+2,nkv,pos+s,dh]
            let vb = concat4(&mut blk, &format!("l{i}_vb"), &[ck1.as_str(), c0.as_str(), pos_i32.as_str(), c0.as_str()]);
            let ve = concat4(&mut blk, &format!("l{i}_ve"), &[cv.as_str(), cnkv.as_str(), pos_s.as_str(), cdh.as_str()]);
            cur_kv = blk.o1(
                "slice_update",
                vec![
                    ("x".into(), bind(&cur_kv).1),
                    ("update".into(), bind(&new_v).1),
                    ("begin".into(), bind(&vb).1),
                    ("end".into(), bind(&ve).1),
                    ("stride".into(), bind(&st4).1),
                    ("begin_mask".into(), bind(&bm4).1),
                    ("end_mask".into(), bind(&bm4).1),
                    ("squeeze_mask".into(), bind(&bm4).1),
                ],
                &format!("l{i}_vup"),
                ValueType::Tensor(TensorType::f16(&kvshape)),
            );
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
            let k_new = blk.matmul("kv_write_mask", &new_k, false, &kshape, &format!("l{i}_knew"));
            let k_full = blk.add(&k_keep, &k_new, &kshape, &format!("l{i}_kfull"));
            let v_old = blk.slice(
                &cur_kv,
                &[krow + 1, 0, 0, 0],
                &[krow + 2, nkv as i32, seq as i32, dh as i32],
                &kshape,
                &format!("l{i}_vold"),
            );
            let v_keep = blk.mul(&v_old, &one_m, &kshape, &format!("l{i}_vkeep"));
            let v_new = blk.matmul("kv_write_mask", &new_v, false, &kshape, &format!("l{i}_vnew"));
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
                vec![("val".into(), mil_spec::Value::bools(&[false, false, false, false]))],
            )[0]
            .clone();
            let kb4 = blk.konst_i32(&format!("l{i}_kb4"), &[krow, 0, 0, 0]);
            let ke4 = blk.konst_i32(&format!("l{i}_ke4"), &[krow + 1, nkv as i32, seq as i32, dh as i32]);
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
            let ve4 = blk.konst_i32(&format!("l{i}_ve4"), &[krow + 2, nkv as i32, seq as i32, dh as i32]);
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
                blk.concat(&[q2.clone(), q2], 2, &[1, nkv, 2 * hpk, dh], &format!("l{i}_qfold"))
            } else {
                let q4 = blk.reshape(&q_r, &[nkv, hpk * dh, 1, s], &format!("l{i}_qheads"));
                let qt = blk.transpose(&q4, &[0, 3, 2, 1], &[nkv, s, 1, hpk * dh], &format!("l{i}_qt"));
                blk.reshape(&qt, &[1, nkv, s * hpk, dh], &format!("l{i}_qfold"))
            };
            // Fused SDPA silently DROPS a non-broadcast mask for multi-query
            // shapes (verified: hidden identical under an all -10000 mask).
            // Materialize scores + softmax explicitly so the mask is real.
            // K^T comes from matmul's fused transpose_y — a standalone
            // transpose of a slice_update-derived slice fails plan-build
            // -14 (mil_kvpack verified the fused variant loads clean).
            let scores = blk.matmul(&qf, &k_attn, true, &[1, nkv, rows, seq], &format!("l{i}_sc"));
            let scl = blk.konst_f16(&format!("l{i}_scl"), 1.0 / (dh as f32).sqrt());
            let sc = blk.mul(&scores, &scl, &[1, nkv, rows, seq], &format!("l{i}_scm"));
            let masked = blk.add(&sc, "attn_mask", &[1, nkv, rows, seq], &format!("l{i}_msk"));
            let probs = blk.softmax(&masked, 3, &[1, nkv, rows, seq], &format!("l{i}_pb"), false);
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
            let attn = blk.matmul(&probs, &v_op, false, &[1, nkv, rows, dh], &format!("l{i}_sdpa"));
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
                let at = blk.transpose(&ar, &[0, 2, 3, 1], &[nkv, hpk * dh, 1, s], &format!("l{i}_at"));
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
        );
        let o = conv1x1_w(&mut blk, &attn4, &wo, d, s, &format!("l{i}_o"));
        let x1 = blk.add(&cur, &o, &shape_x, &format!("l{i}_res1"));

        // ---- FFN ----
        let wfn = st.get_f16(&format!("{p}.post_attention_layernorm.weight"))?;
        let wfnn = blob(
            &mut wb,
            &mut blk,
            &format!("w_ffn_norm_{i}"),
            &wfn,
            &[d, 1, 1],
        );
        let n2 = blk.rms_norm(&x1, &wfnn, d, cfg.eps, &shape_x, &format!("l{i}_fn"));
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
        );
        let gu = conv1x1_w(&mut blk, &n2, &wgu, 2 * cfg.d_ff, s, &format!("l{i}_gu"));
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
        let silu = blk.o1(
            "silu",
            vec![("x".into(), bind(&gate).1)],
            &format!("l{i}_silu"),
            ValueType::Tensor(TensorType::f16(&[1, cfg.d_ff, 1, s])),
        );
        let hidden = blk.mul(&silu, &up, &[1, cfg.d_ff, 1, s], &format!("l{i}_hid"));
        let dw = st.get_f16(&format!("{p}.mlp.down_proj.weight"))?;
        let wd = weight_const(
            &mut wb,
            &mut blk,
            &format!("w_down_{i}"),
            &dw,
            &[d, cfg.d_ff, 1, 1],
        );
        // The FFN hidden is the only un-normalized matmul input; Qwen3's
        // massive activation (~8e3 at layer 6) overflows the ANE's int8-weight
        // conv. Pre-scaling by 1/α and restoring by α is exact in real
        // arithmetic and keeps the int8 path inside FP16 range.
        let alpha = down_prescale();
        let ffn = if alpha > 1.0 {
            let inv = blk.konst_f16(&format!("l{i}_dinv"), 1.0 / alpha);
            let hs = blk.mul(&hidden, &inv, &[1, cfg.d_ff, 1, s], &format!("l{i}_hids"));
            let raw = conv1x1_w(&mut blk, &hs, &wd, d, s, &format!("l{i}_down_s"));
            let a = blk.konst_f16(&format!("l{i}_dalpha"), alpha);
            blk.mul(&raw, &a, &shape_x, &format!("l{i}_down"))
        } else {
            conv1x1_w(&mut blk, &hidden, &wd, d, s, &format!("l{i}_down"))
        };
        // the last layer's output must be named `hidden` to match the declared
        // model output feature (CoreML binds program outputs by name).
        let stop_at = std::env::var("BADAPPLE_STOP_AT").ok();
        let out_nm = if i == le - 1 && (stop_at.is_none() || s != 1) {
            "hidden".to_string()
        } else {
            format!("l{i}_res2")
        };
        cur = blk.add(&x1, &ffn, &shape_x, &out_nm);
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
            ("an_out", vec![1, d, 1, 1]),       // post attn_norm
            ("qo", vec![1, q_dim, 1, 1]),       // q post head-norm+weight
            ("ko", vec![1, kv_dim, 1, 1]),      // k post head-norm+weight
            ("rq_flat", vec![1, q_dim, 1, 1]),  // q post-rope
            ("rk_flat", vec![1, kv_dim, 1, 1]), // k post-rope
            ("attn4", vec![1, q_dim, 1, 1]),    // attention out pre-o_proj
            ("v", vec![1, kv_dim, 1, 1]),       // v slice of qkv (pre-norm/rope)
            ("sc", vec![1, nkv, 2 * hpk, seq]), // attention scores pre-mask (padded)
            ("msk", vec![1, nkv, 2 * hpk, seq]),// scores post-mask (padded)
            ("pb", vec![1, nkv, 2 * hpk, seq]), // attention probs (padded)
            ("o", vec![1, d, 1, 1]),            // o_proj out
            ("res1", vec![1, d, 1, 1]),         // post-attn residual
            ("fn_out", vec![1, d, 1, 1]),       // post ffn_norm
            ("hid", vec![1, cfg.d_ff, 1, 1]),   // FFN hidden pre-down
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
    let out_shape_decl = stop_shape.unwrap_or_else(|| vec![1, d, 1, s]);

    // ---- serialize ----
    let rope_shape: Vec<i64> = if s == 1 { vec![1, rope_half] } else { vec![1, 1, rope_half, s] };
    // s==1 declares a real dim2 on attn_mask ([1,1,rows,seq] where rows is
    // the padded query count): a [1,1,1,seq] mask broadcasts dim2 1->rows,
    // which the s==1 plan misaligns — probs then spread onto unwritten
    // (garbage) KV slots instead of staying one-hot. dim0/dim1 broadcast
    // is proven fine (shared with the s>1 path).
    let mask_shape: Vec<i64> = if s == 1 { vec![1, 1, 2 * hpk, seq] } else { vec![1, 1, hpk * s, seq] };
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
        NVT {
            name: "kv_cache".into(),
            ty: ValueType::State(TensorType::f16(&kvshape)),
        },
    ];
    let states: Vec<Feature> = vec![Feature {
        name: "kv_cache".into(),
        shape: kvshape.clone(),
        dtype: DType::Fp16,
        is_state: true,
    }];
    let inputs: Vec<Feature> = vec![
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
    let outputs = vec![Feature {
        name: "hidden".into(),
        shape: out_shape_decl,
        dtype: DType::Fp16,
        is_state: false,
    }];

    // The ANE plan-build coverage of the slice_update graph is
    // spec-dependent: the s==1 (decode) graph only plans under ios18/
    // CoreML8, while the s>1 (prefill) graph only plans under ios19/
    // CoreML9 — each fails -14 under the other spec (verified across
    // every shard and bisected to the spec flag itself). Emit each shard
    // at the spec its graph plans under; deployment target is per-model.
    let (sv, ops) = if s == 1 { (9, "CoreML8") } else { (10, "CoreML9") };
    let meta = ModelMeta::new(sv, ops)
        .creator("badapple-aneconvert")
        .description("Bad Apple native ANE shard");
    let spec = encode_model(&inputs, &outputs, &states, &blk, &fn_inputs, &meta);
    let weights = wb.finish();
    write_mlpackage(&pkg_dir, &spec, Some(&weights)).map_err(|e| e.to_string())?;
    let mlmodelc = out_dir.join(format!("{name}.mlmodelc"));
    Ok((pkg_dir, mlmodelc, weights))
}

/// per-head RMSNorm over axis 1 of (n, dh, 1, 1); weight application handled by caller.
fn head_rms(blk: &mut Block, x: &str, d: i64, eps: f32, shape4: &[i64], pfx: &str) -> String {
    let k = (d as f32).sqrt();
    let inv_k = blk.konst_f16(&format!("{pfx}_ik"), 1.0 / k);
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
    let e = blk.konst_f16(&format!("{pfx}_e"), eps / (k * k));
    let vp = blk.add(&mean, &e, &mshape, &format!("{pfx}_vp"));
    let z = blk.konst_f16(&format!("{pfx}_z"), 0.0);
    let r = blk.o1(
        "rsqrt",
        vec![("x".into(), bind(&vp).1), ("epsilon".into(), bind(&z).1)],
        &format!("{pfx}_r"),
        ValueType::Tensor(TensorType::f16(&mshape)),
    );
    blk.mul(&xs, &r, shape4, &format!("{pfx}_xn"))
}

/// Emit a vocab-slice LM head shard (final norm + sliced projection).
fn emit_head_shard(
    st: &StIndex,
    cfg: &Cfg,
    vstart: i64,
    vend: i64,
    out_dir: &Path,
) -> Result<(PathBuf, PathBuf), String> {
    let name = format!("head_v{:05}-{:05}", vstart, vend);
    let pkg_dir = out_dir.join(format!("{name}.mlpackage"));
    let mut blk = Block::new();
    let mut wb = WeightBin::new();
    let d = cfg.d_model;
    let vsz = vend - vstart;

    let nw = st.get_f16("model.norm.weight")?;
    let noff = wb.put("w_out_norm", DType::Fp16, &[d, 1, 1], &nw);
    let wn = blk.konst_blob("w_out_norm", WEIGHTS_NAME, noff, DType::Fp16, &[d, 1, 1]);

    let normed = blk.rms_norm("x", &wn, d, cfg.eps, &[1, d, 1, 1], "hn");

    // sliced lm head weight: (vsz, d) -> conv (vsz, d, 1, 1)
    let lm_name = if cfg.tie_embeddings || !st.tensors.contains_key("lm_head.weight") {
        "model.embed_tokens.weight"
    } else {
        "lm_head.weight"
    };
    let t = &st.tensors[lm_name];
    let row_bytes = (d as u64) * 2;
    let mut fh = std::fs::File::open(&t.file).map_err(|e| e.to_string())?;
    fh.seek(SeekFrom::Start(t.begin + (vstart as u64) * row_bytes))
        .map_err(|e| e.to_string())?;
    let mut slice_raw = vec![0u8; (vsz as u64 * row_bytes) as usize];
    fh.read_exact(&mut slice_raw).map_err(|e| e.to_string())?;
    let slice_f16 = match t.dtype.as_str() {
        "F16" => slice_raw,
        "BF16" => bf16_bytes_to_f16(&slice_raw),
        "F32" => f32_bytes_to_f16(&slice_raw),
        o => return Err(format!("lm_head dtype {o}")),
    };
    let wl = weight_const(&mut wb, &mut blk, "w_lm", &slice_f16, &[vsz, d, 1, 1]);
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

/// Emit a NON-stateful debug package containing only the pre-attention prefix
/// of layer `ls`, for numeric bisection. stage selects where to stop:
/// an | q | qn | kn | rq | rk | v
fn emit_probe(
    st: &StIndex,
    cfg: &Cfg,
    ls: i64,
    stage: &str,
    out_dir: &Path,
) -> Result<(PathBuf, PathBuf, Vec<u8>), String> {
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
    let normed = blk.rms_norm("x", &wn, d, cfg.eps, &[1, d, 1, 1], &format!("l{ls}_an"));
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

/// Write the raw fp16 embedding table [vocab, d] row-major.
fn emit_embedding(st: &StIndex, cfg: &Cfg, path: &Path) -> Result<(), String> {
    let raw = st.get_f16("model.embed_tokens.weight")?;
    std::fs::write(path, &raw).map_err(|e| e.to_string())?;
    let want = (cfg.vocab * cfg.d_model * 2) as usize;
    if raw.len() != want {
        return Err(format!("embedding size {} != expected {}", raw.len(), want));
    }
    Ok(())
}

// ======== compile driver ========

fn compile_pkg(pkg: &Path, out_dir: &Path) -> Result<PathBuf, String> {
    let stem = pkg.file_stem().unwrap().to_string_lossy().to_string();
    let out = out_dir.join(format!("{stem}.mlmodelc"));
    if out.exists() {
        std::fs::remove_dir_all(&out).map_err(|e| e.to_string())?;
    }
    let status = Command::new("xcrun")
        .args(["coremlc", "compile"])
        .arg(pkg)
        .arg(out_dir)
        .status()
        .map_err(|e| format!("coremlc spawn: {e}"))?;
    if !status.success() {
        return Err(format!("coremlc failed for {pkg:?} (status {status})"));
    }
    if !out.exists() {
        return Err(format!("compiled output missing: {out:?}"));
    }
    Ok(out)
}

fn rel(p: &Path, base: &Path) -> String {
    p.strip_prefix(base)
        .map(|r| r.to_string_lossy().to_string())
        .unwrap_or_else(|_| p.to_string_lossy().to_string())
}

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
                if bits != 8 && bits != 16 {
                    eprintln!("--weight-bits must be 8 or 16");
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
    let st = match StIndex::load(&model_dir) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("safetensors: {e}");
            std::process::exit(1);
        }
    };
    std::fs::create_dir_all(&out_dir).unwrap();
    std::fs::create_dir_all(out_dir.join("logs")).unwrap();

    if let Some(stage) = &probe_stage {
        let ls = only_layers.map(|(a, _)| a).unwrap_or(0);
        match emit_probe(&st, &cfg, ls, stage, &out_dir) {
            Ok((pkg, mlpath, _)) => {
                eprintln!("  probe {stage} layer {ls}: {pkg:?}");
                if !no_compile {
                    match compile_pkg(&pkg, &out_dir) {
                        Ok(c) => eprintln!("  compiled -> {c:?}"),
                        Err(e) => eprintln!("  compile failed: {e}"),
                    }
                }
                if !keep_packages {
                    let _ = std::fs::remove_dir_all(&pkg);
                }
                let _ = mlpath;
            }
            Err(e) => eprintln!("  probe emit failed: {e}"),
        }
        return;
    }

    let (ls0, le0) = only_layers.unwrap_or((0, cfg.n_layers));

    let mut manifest = json!({
        "schema_version": 1,
        "status": "pending",
        "source": { "path": model_dir.to_string_lossy() },
        "model": {
            "architecture": "qwen3",
            "total_layers": cfg.n_layers,
            "hidden_size": cfg.d_model,
            "vocab_size": cfg.vocab,
            "seq_len": seq,
            "quant_bits": if weight_bits() == 8 { 8 } else { 0 },
            "compute_units": "all",
            "layers_per_shard": lps,
            "decode_width": decode_width,
            "stateful": true,
            "rope_dim": cfg.rope_dim,
            "rope_freq_base": cfg.rope_theta,
            "rms_norm_eps": cfg.eps,
            "native": true,
        },
        "shared": {
            "embedding": {
                "path": rel(&out_dir.join("embedding.f16"), &out_dir),
                "shape": [cfg.vocab, cfg.d_model],
                "dtype": "float16",
                "status": "pending",
            },
            "lm_head_shards": [],
        },
        "shards": [],
    });

    // embedding
    match emit_embedding(&st, &cfg, &out_dir.join("embedding.f16")) {
        Ok(()) => {
            manifest["shared"]["embedding"]["status"] = json!("complete");
            manifest["shared"]["embedding"]["size_bytes"] =
                json!((cfg.vocab * cfg.d_model * 2) as u64);
            eprintln!(
                "  embedding.f16 written ({} MB)",
                cfg.vocab * cfg.d_model * 2 / 1_000_000
            );
        }
        Err(e) => eprintln!("  embedding: {e}"),
    }

    // layer shards
    let mut shard_entries = vec![];
    let mut ls = ls0;
    while ls < le0 {
        let le = (ls + lps).min(le0);
        eprint!("  shard [{ls},{le}) ... ");
        std::io::stderr().flush().ok();
        match emit_layer_shard(&st, &cfg, ls, le, seq, &out_dir, decode_width, false) {
            Ok((pkg, mlmodelc, _w)) => {
                let mut ent = Map::new();
                ent.insert(
                    "name".into(),
                    json!(pkg.file_stem().unwrap().to_string_lossy()),
                );
                ent.insert("layer_start".into(), json!(ls));
                ent.insert("layer_end".into(), json!(le));
                if no_compile {
                    ent.insert("status".into(), json!("packaged"));
                    eprintln!("packaged (compile skipped)");
                } else {
                    match compile_pkg(&pkg, &out_dir) {
                        Ok(c) => {
                            ent.insert("status".into(), json!("compiled"));
                            ent.insert("compiled_path".into(), json!(rel(&c, &out_dir)));
                            let sz = dir_size(&c);
                            ent.insert("compiled_size_bytes".into(), json!(sz));
                            eprintln!("compiled ({:.1} MB)", sz as f64 / 1e6);
                            if !keep_packages {
                                // the runtime only consumes .mlmodelc; drop the
                                // source package to halve the artifact footprint
                                let _ = std::fs::remove_dir_all(&pkg);
                            }
                        }
                        Err(e) => {
                            ent.insert("status".into(), json!("compile_failed"));
                            ent.insert("last_error".into(), json!(e));
                            eprintln!("FAILED: {e}");
                        }
                    }
                    let _ = mlmodelc;
                }
                ent.insert("package_path".into(), json!(rel(&pkg, &out_dir)));
                shard_entries.push(Value::Object(ent));
            }
            Err(e) => {
                eprintln!("emit failed: {e}");
                let mut ent = Map::new();
                ent.insert("layer_start".into(), json!(ls));
                ent.insert("layer_end".into(), json!(le));
                ent.insert("status".into(), json!("emit_failed"));
                ent.insert("last_error".into(), json!(e));
                shard_entries.push(Value::Object(ent));
            }
        }
        // persist after every shard (resumable)
        manifest["shards"] = json!(shard_entries);
        std::fs::write(
            out_dir.join("conversion_manifest.json"),
            serde_json::to_string_pretty(&manifest).unwrap(),
        )
        .ok();
        ls = le;
    }

    // prefill shards: same layer groupings and state names as the decode
    // shards, emitted at token width `prefill_chunk`. They write into the
    // same MLState buffers, so a decode session can be primed in bulk.
    if prefill_chunk > 0 {
        manifest["model"]["prefill_chunk"] = json!(prefill_chunk);
        let mut prefill_entries = vec![];
        let mut pls = ls0;
        while pls < le0 {
            let ple = (pls + lps).min(le0);
            eprint!("  prefill shard [{pls},{ple}) ... ");
            std::io::stderr().flush().ok();
            match emit_layer_shard(&st, &cfg, pls, ple, seq, &out_dir, prefill_chunk, true) {
                Ok((pkg, mlmodelc, _w)) => {
                    let mut ent = Map::new();
                    ent.insert(
                        "name".into(),
                        json!(pkg.file_stem().unwrap().to_string_lossy()),
                    );
                    ent.insert("layer_start".into(), json!(pls));
                    ent.insert("layer_end".into(), json!(ple));
                    if no_compile {
                        ent.insert("status".into(), json!("packaged"));
                        eprintln!("packaged (compile skipped)");
                    } else {
                        match compile_pkg(&pkg, &out_dir) {
                            Ok(c) => {
                                ent.insert("status".into(), json!("compiled"));
                                ent.insert("compiled_path".into(), json!(rel(&c, &out_dir)));
                                let sz = dir_size(&c);
                                ent.insert("compiled_size_bytes".into(), json!(sz));
                                eprintln!("compiled ({:.1} MB)", sz as f64 / 1e6);
                                if !keep_packages {
                                    let _ = std::fs::remove_dir_all(&pkg);
                                }
                            }
                            Err(e) => {
                                ent.insert("status".into(), json!("compile_failed"));
                                ent.insert("last_error".into(), json!(e));
                                eprintln!("FAILED: {e}");
                            }
                        }
                        let _ = mlmodelc;
                    }
                    ent.insert("package_path".into(), json!(rel(&pkg, &out_dir)));
                    prefill_entries.push(Value::Object(ent));
                }
                Err(e) => {
                    eprintln!("emit failed: {e}");
                    let mut ent = Map::new();
                    ent.insert("layer_start".into(), json!(pls));
                    ent.insert("layer_end".into(), json!(ple));
                    ent.insert("status".into(), json!("emit_failed"));
                    ent.insert("last_error".into(), json!(e));
                    prefill_entries.push(Value::Object(ent));
                }
            }
            manifest["shards_prefill"] = json!(prefill_entries);
            std::fs::write(
                out_dir.join("conversion_manifest.json"),
                serde_json::to_string_pretty(&manifest).unwrap(),
            )
            .ok();
            pls = ple;
        }
    }

    // head shards
    let mut heads = vec![];
    let width = (cfg.vocab + head_shards.max(1) - 1) / head_shards.max(1);
    let mut vs = if head_shards == 0 { cfg.vocab } else { 0i64 };
    while vs < cfg.vocab {
        let ve = (vs + width).min(cfg.vocab);
        eprint!("  head [{vs},{ve}) ... ");
        std::io::stderr().flush().ok();
        match emit_head_shard(&st, &cfg, vs, ve, &out_dir) {
            Ok((pkg, _c)) => {
                let mut ent = Map::new();
                ent.insert(
                    "name".into(),
                    json!(pkg.file_stem().unwrap().to_string_lossy()),
                );
                ent.insert("vocab_start".into(), json!(vs));
                ent.insert("vocab_end".into(), json!(ve));
                ent.insert("package_path".into(), json!(rel(&pkg, &out_dir)));
                if no_compile {
                    ent.insert("status".into(), json!("packaged"));
                    eprintln!("packaged");
                } else {
                    match compile_pkg(&pkg, &out_dir) {
                        Ok(c) => {
                            ent.insert("status".into(), json!("compiled"));
                            ent.insert("compiled_path".into(), json!(rel(&c, &out_dir)));
                            eprintln!("compiled ({:.1} MB)", dir_size(&c) as f64 / 1e6);
                            if !keep_packages {
                                let _ = std::fs::remove_dir_all(&pkg);
                            }
                        }
                        Err(e) => {
                            ent.insert("status".into(), json!("compile_failed"));
                            ent.insert("last_error".into(), json!(e));
                            eprintln!("FAILED: {e}");
                        }
                    }
                }
                heads.push(Value::Object(ent));
            }
            Err(e) => {
                eprintln!("emit failed: {e}");
                let mut ent = Map::new();
                ent.insert("vocab_start".into(), json!(vs));
                ent.insert("vocab_end".into(), json!(ve));
                ent.insert("status".into(), json!("emit_failed"));
                ent.insert("last_error".into(), json!(e));
                heads.push(Value::Object(ent));
            }
        }
        manifest["shared"]["lm_head_shards"] = json!(heads);
        std::fs::write(
            out_dir.join("conversion_manifest.json"),
            serde_json::to_string_pretty(&manifest).unwrap(),
        )
        .ok();
        vs = ve;
    }

    let all_ok = manifest["shards"]
        .as_array()
        .map(|a| {
            a.iter()
                .all(|s| s["status"] == "compiled" || s["status"] == "packaged")
        })
        .unwrap_or(false)
        && manifest["shared"]["lm_head_shards"]
            .as_array()
            .map(|a| {
                a.iter()
                    .all(|s| s["status"] == "compiled" || s["status"] == "packaged")
            })
            .unwrap_or(false)
        && manifest["shared"]["embedding"]["status"] == "complete";
    // Runtimes tokenize from the artifact dir itself (tokenizer.json for the
    // vocab, tokenizer_config.json for the chat template).
    for f in ["tokenizer.json", "tokenizer_config.json"] {
        if let Err(e) = std::fs::copy(model_dir.join(f), out_dir.join(f)) {
            eprintln!("warning: could not copy {f}: {e}");
        }
    }
    manifest["status"] = json!(if all_ok { "complete" } else { "partial" });
    std::fs::write(
        out_dir.join("conversion_manifest.json"),
        serde_json::to_string_pretty(&manifest).unwrap(),
    )
    .ok();
    eprintln!("manifest: {}", if all_ok { "complete" } else { "partial" });
}

fn dir_size(p: &Path) -> u64 {
    let mut total = 0;
    if let Ok(rd) = std::fs::read_dir(p) {
        for e in rd.flatten() {
            let m = e.path();
            if m.is_dir() {
                total += dir_size(&m);
            } else {
                total += e.metadata().map(|m| m.len()).unwrap_or(0);
            }
        }
    }
    total
}
