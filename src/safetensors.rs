//! safetensors index + fp16/quantized weight access — extracted from
//! badapple-aneconvert so the Coyote `st_*`/`w_*` externs share the
//! implementation with the converter binary.

use serde_json::Value;
use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

#[derive(Clone, Debug)]
pub struct TensorInfo {
    pub dtype: String,
    pub file: PathBuf,
    pub begin: u64,
    pub end: u64,
    pub shape: Vec<u64>,
}

#[derive(Clone)]
pub struct StIndex {
    pub tensors: HashMap<String, TensorInfo>,
    /// MLX quantization group size (config.json → quantization.group_size).
    pub group_size: usize,
}

impl StIndex {
    pub fn load(dir: &Path) -> Result<Self, String> {
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
                let offs = meta["data_offsets"].as_array().ok_or("no data_offsets")?;
                let begin = offs[0].as_u64().unwrap_or(0);
                let end = offs[1].as_u64().unwrap_or(0);
                let shape: Vec<u64> = meta["shape"]
                    .as_array()
                    .map(|a| a.iter().filter_map(|v| v.as_u64()).collect())
                    .unwrap_or_default();
                let base = 8 + hlen as u64;
                tensors.insert(
                    name.clone(),
                    TensorInfo {
                        dtype,
                        file: file.clone(),
                        begin: base + begin,
                        end: base + end,
                        shape,
                    },
                );
            }
        }
        let group_size = std::fs::read_to_string(dir.join("config.json"))
            .ok()
            .and_then(|s| serde_json::from_str::<Value>(&s).ok())
            .and_then(|c| c["quantization"]["group_size"].as_u64())
            .unwrap_or(64) as usize;
        Ok(StIndex {
            tensors,
            group_size,
        })
    }

    pub fn get_raw(&self, name: &str) -> Result<Vec<u8>, String> {
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
    pub fn get_f16(&self, name: &str) -> Result<Vec<u8>, String> {
        let t = self
            .tensors
            .get(name)
            .ok_or_else(|| format!("missing tensor {name}"))?;
        let raw = self.get_raw(name)?;
        match t.dtype.as_str() {
            "F16" => Ok(raw),
            "BF16" => Ok(bf16_bytes_to_f16(&raw)),
            "F32" => Ok(f32_bytes_to_f16(&raw)),
            // MLX-quantized linear: packed U32 nibbles + F16 scales/biases.
            "U32" => {
                let base = name
                    .strip_suffix(".weight")
                    .ok_or_else(|| format!("{name}: quantized tensor without .weight suffix"))?;
                let scales = self.get_raw(&format!("{base}.scales"))?;
                let biases = self.get_raw(&format!("{base}.biases"))?;
                dequant_mlx_q4(&raw, &t.shape, &scales, &biases, self.group_size)
            }
            other => Err(format!("{name}: unsupported dtype {other}")),
        }
    }

    /// Read rows [r0, r1) of a 2-D tensor as fp16 bytes. For U32-quantized
    /// tensors the packed weight rows and their per-row scale/bias stripes
    /// are sliced independently, then dequantized.
    pub fn get_f16_rows(&self, name: &str, r0: u64, r1: u64) -> Result<Vec<u8>, String> {
        let t = self
            .tensors
            .get(name)
            .ok_or_else(|| format!("missing tensor {name}"))?;
        if t.shape.len() != 2 {
            return Err(format!("{name}: expected 2-D tensor, got {:?}", t.shape));
        }
        let rows = t.shape[0];
        if r1 > rows || r0 >= r1 {
            return Err(format!("{name}: bad row range {r0}..{r1} of {rows}"));
        }
        let read_at = |t: &TensorInfo, off: u64, len: u64| -> Result<Vec<u8>, String> {
            let mut fh = std::fs::File::open(&t.file).map_err(|e| e.to_string())?;
            fh.seek(SeekFrom::Start(t.begin + off))
                .map_err(|e| e.to_string())?;
            let mut buf = vec![0u8; len as usize];
            fh.read_exact(&mut buf).map_err(|e| e.to_string())?;
            Ok(buf)
        };
        if t.dtype == "U32" {
            let base = name
                .strip_suffix(".weight")
                .ok_or_else(|| format!("{name}: quantized tensor without .weight suffix"))?;
            let st_t = self
                .tensors
                .get(&format!("{base}.scales"))
                .ok_or_else(|| format!("{name}: missing scales"))?;
            let bt_t = self
                .tensors
                .get(&format!("{base}.biases"))
                .ok_or_else(|| format!("{name}: missing biases"))?;
            let packed_row = (t.end - t.begin) / rows; // cols*4 bytes
            let ngroups = t.shape[1] * 8 / self.group_size as u64; // unpacked groups
            let srow = ngroups * 2;
            let n = r1 - r0;
            let wr = read_at(t, r0 * packed_row, n * packed_row)?;
            let sr = read_at(st_t, r0 * srow, n * srow)?;
            let br = read_at(bt_t, r0 * srow, n * srow)?;
            return dequant_mlx_q4(&wr, &[n, t.shape[1]], &sr, &br, self.group_size);
        }
        let elem: u64 = match t.dtype.as_str() {
            "F16" | "BF16" => 2,
            "F32" => 4,
            o => return Err(format!("{name}: unsupported dtype {o}")),
        };
        let row_bytes = t.shape[1] * elem;
        let raw = read_at(t, r0 * row_bytes, (r1 - r0) * row_bytes)?;
        match t.dtype.as_str() {
            "F16" => Ok(raw),
            "BF16" => Ok(bf16_bytes_to_f16(&raw)),
            _ => Ok(f32_bytes_to_f16(&raw)),
        }
    }
}

/// Dequantize an MLX 4-bit affine-quantized weight to fp16 bytes.
/// Layout: `weight` is U32 [rows, cols/8] — 8 consecutive input-dim nibbles
pub fn dequant_mlx_q4(
    raw: &[u8],
    shape: &[u64],
    scales: &[u8],
    biases: &[u8],
    group: usize,
) -> Result<Vec<u8>, String> {
    if shape.len() != 2 {
        return Err(format!("quantized weight must be 2-D, got {shape:?}"));
    }
    let (rows, packed_cols) = (shape[0] as usize, shape[1] as usize);
    let cols = packed_cols * 8;
    if group == 0 || cols % group != 0 {
        return Err(format!("cols {cols} not divisible by group_size {group}"));
    }
    let ngroups = cols / group;
    if scales.len() != rows * ngroups * 2 || biases.len() != scales.len() {
        return Err(format!(
            "scale/bias size mismatch: {} bytes for [{rows}x{ngroups}]",
            scales.len()
        ));
    }
    let mut out = Vec::with_capacity(rows * cols * 2);
    for r in 0..rows {
        let row = &raw[r * packed_cols * 4..(r + 1) * packed_cols * 4];
        for w in 0..packed_cols {
            let word =
                u32::from_le_bytes([row[w * 4], row[w * 4 + 1], row[w * 4 + 2], row[w * 4 + 3]]);
            for n in 0..8usize {
                let i = w * 8 + n;
                let g = i / group;
                let so = (r * ngroups + g) * 2;
                let s = half::bf16::from_le_bytes([scales[so], scales[so + 1]]).to_f32();
                let b = half::bf16::from_le_bytes([biases[so], biases[so + 1]]).to_f32();
                let q = ((word >> (n * 4)) & 0xF) as f32;
                out.extend_from_slice(&half::f16::from_f32(s * q + b).to_le_bytes());
            }
        }
    }
    Ok(out)
}

pub fn bf16_bytes_to_f16(raw: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(raw.len());
    for c in raw.chunks_exact(2) {
        let b = half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]]));
        out.extend_from_slice(&half::f16::from_f32(b.to_f32()).to_le_bytes());
    }
    out
}

pub fn f32_bytes_to_f16(raw: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(raw.len() / 2);
    for c in raw.chunks_exact(4) {
        let v = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
        out.extend_from_slice(&half::f16::from_f32(v).to_le_bytes());
    }
    out
}
pub fn f16_plus_one(bytes: &[u8]) -> Vec<u8> {
    map_f16(bytes, |v| v + 1.0)
}

/// Multiply each fp16 column of a row-major [rows, cols] weight by the
/// matching fp16 scale element: out[r][c] = w[r][c] * col[c]. Folds a
/// broadcast elementwise norm-weight mul into a following projection.
pub fn f16_scale_cols(w: &[u8], col: &[u8]) -> Vec<u8> {
    let ncols = col.len() / 2;
    let cv: Vec<f32> = col
        .chunks_exact(2)
        .map(|b| half::f16::from_le_bytes([b[0], b[1]]).to_f32())
        .collect();
    w.chunks_exact(2)
        .enumerate()
        .flat_map(|(i, b)| {
            let v = half::f16::from_le_bytes([b[0], b[1]]).to_f32() * cv[i % ncols];
            half::f16::from_f32(v).to_le_bytes()
        })
        .collect()
}

/// Gemma embeddings enter the graph pre-scaled by sqrt(d_model).
pub fn f16_scale_all(bytes: &[u8], scale: f32) -> Vec<u8> {
    if (scale - 1.0).abs() < f32::EPSILON {
        return bytes.to_vec();
    }
    map_f16(bytes, |v| v * scale)
}

pub fn map_f16(bytes: &[u8], f: impl Fn(f32) -> f32) -> Vec<u8> {
    bytes
        .chunks_exact(2)
        .flat_map(|b| {
            let v = half::f16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32();
            half::f16::from_f32(f(v)).to_bits().to_le_bytes()
        })
        .collect()
}
pub fn quantize_q8_rows(fp16: &[u8], rows: usize) -> (Vec<u8>, Vec<u8>) {
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

/// Symmetric per-row 4-bit palettization: scale_r = max|w_r| / 7, index =
/// round(w / scale_r) + 8 clamped to [0, 15]. Packed two nibbles per byte,
/// low nibble = even index. The LUT per row holds scale_r*(i-8) for i in
/// 0..16 — reconstruction is identical to signed int4 + scale, packaged in
/// the `constexpr_lut_to_dense` form the ANE lowers.
/// `fp16` is row-major [rows, cols]. Returns (packed bytes, fp16 LUT bytes).
pub fn quantize_q4_rows(fp16: &[u8], rows: usize) -> (Vec<u8>, Vec<u8>) {
    let cols = fp16.len() / 2 / rows;
    let mut q = Vec::with_capacity(rows * cols.div_ceil(2));
    let mut lut = Vec::with_capacity(rows * 16 * 2);
    let mut row = vec![0f32; cols];
    for r in 0..rows {
        let base = r * cols * 2;
        let mut amax = 0f32;
        for (c, v) in row.iter_mut().enumerate() {
            let o = base + c * 2;
            *v = half::f16::from_le_bytes([fp16[o], fp16[o + 1]]).to_f32();
            amax = amax.max(v.abs());
        }
        let scale = half::f16::from_f32(if amax > 0.0 { amax / 7.0 } else { 1.0 });
        let s = scale.to_f32();
        let mut c = 0;
        while c < cols {
            let lo = ((row[c] / s).round().clamp(-8.0, 7.0) as i8 + 8) & 0xF;
            let hi = if c + 1 < cols {
                ((row[c + 1] / s).round().clamp(-8.0, 7.0) as i8 + 8) & 0xF
            } else {
                0
            };
            q.push((lo as u8) | ((hi as u8) << 4));
            c += 2;
        }
        for i in 0..16 {
            lut.extend_from_slice(&half::f16::from_f32(s * (i as f32 - 8.0)).to_le_bytes());
        }
    }
    (q, lut)
}

/// α for the down_proj input pre-scale (int8 only; 1 = off). `BADAPPLE_Q8_DOWN_PRESCALE`.
pub fn tile_head_w(bytes: &[u8], n: i64) -> Vec<u8> {
    let mut out = Vec::with_capacity(bytes.len() * n as usize);
    for _ in 0..n {
        out.extend_from_slice(bytes);
    }
    out
}
