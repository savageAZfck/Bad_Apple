//! CPU reference forward for gemma-3-1b, multi-token causal attention.
//! Per-layer |h| at the LAST position, vs the ANE probe's last-step stats.
//! Usage: gemma_ref <model_dir> <tok1> <tok2> ...
use serde_json::Value;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

fn bf16_to_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(2)
        .map(|c| f32::from_bits(((c[1] as u32) << 8 | c[0] as u32) << 16))
        .collect()
}

struct St {
    map: HashMap<String, (PathBuf, u64, u64, String)>,
}
impl St {
    fn load(dir: &Path) -> Self {
        let idx = dir.join("model.safetensors.index.json");
        let mut names: Vec<(String, PathBuf)> = vec![];
        if idx.is_file() {
            let j: Value = serde_json::from_slice(&fs::read(&idx).unwrap()).unwrap();
            for (n, f) in j["weight_map"].as_object().unwrap() {
                names.push((n.clone(), dir.join(f.as_str().unwrap())));
            }
        } else {
            let f = dir.join("model.safetensors");
            let data = fs::read(&f).unwrap();
            let hlen = u64::from_le_bytes(data[..8].try_into().unwrap()) as usize;
            let hdr: Value = serde_json::from_slice(&data[8..8 + hlen]).unwrap();
            for n in hdr.as_object().unwrap().keys() {
                if n != "__metadata__" {
                    names.push((n.clone(), f.clone()));
                }
            }
        }
        let mut map = HashMap::new();
        for (name, file) in names {
            let data = fs::read(&file).unwrap();
            let hlen = u64::from_le_bytes(data[..8].try_into().unwrap()) as usize;
            let hdr: Value = serde_json::from_slice(&data[8..8 + hlen]).unwrap();
            let meta = &hdr.as_object().unwrap()[&name];
            let o = meta["data_offsets"].as_array().unwrap();
            map.insert(
                name,
                (
                    file,
                    8 + hlen as u64 + o[0].as_u64().unwrap(),
                    8 + hlen as u64 + o[1].as_u64().unwrap(),
                    meta["dtype"].as_str().unwrap_or("F16").to_string(),
                ),
            );
        }
        St { map }
    }
    fn get(&self, name: &str) -> Vec<f32> {
        let mut v = self.get_raw(name);
        if std::env::var("Q8").is_ok() && name.ends_with("proj.weight") {
            let cols = if name.contains("down_proj") { 6912usize } else { 1152usize };
            let rows = v.len() / cols;
            for r in 0..rows {
                let row = &mut v[r * cols..(r + 1) * cols];
                let amax = row.iter().fold(0f32, |a, b| a.max(b.abs()));
                let sc = if amax > 0.0 { amax / 127.0 } else { 1.0 };
                for x in row.iter_mut() {
                    *x = (*x / sc).round().clamp(-127.0, 127.0) * sc;
                }
            }
        }
        if std::env::var("FP16").is_ok() {
            for x in v.iter_mut() { *x = h16(*x) }
        }
        return v;
    }
    fn get_raw(&self, name: &str) -> Vec<f32> {
        let (file, b, e, dt) = &self.map[name];
        use std::io::{Read, Seek, SeekFrom};
        let mut f = fs::File::open(file).unwrap();
        f.seek(SeekFrom::Start(*b)).unwrap();
        let mut buf = vec![0u8; (e - b) as usize];
        f.read_exact(&mut buf).unwrap();
        match dt.as_str() {
            "BF16" => bf16_to_f32(&buf),
            "F16" => buf
                .chunks_exact(2)
                .map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
                .collect(),
            "F32" => buf
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect(),
            _ => panic!("dtype {dt}"),
        }
    }
}

fn rms_norm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let ms = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let r = 1.0 / (ms + eps).sqrt();
    x.iter().zip(w).map(|(a, b)| a * r * (1.0 + b)).collect()
}

fn matmul(w: &[f32], rows: usize, cols: usize, x: &[f32]) -> Vec<f32> {
    let mut o: Vec<f32> = (0..rows)
        .map(|r| (0..cols).map(|c| w[r * cols + c] * x[c]).sum::<f32>())
        .collect();
    v16(&mut o);
    o
}

fn rope_half(v: &mut [f32], dh: usize, pos: usize, theta: f32) {
    let half = dh / 2;
    for i in 0..half {
        let ang = pos as f32 * theta.powf(-(i as f32) / half as f32);
        let (s, c) = ang.sin_cos();
        let lo = v[i];
        let hi = v[i + half];
        v[i] = lo * c - hi * s;
        v[i + half] = lo * s + hi * c;
    }
}

fn gelu_tanh(x: f32) -> f32 {
    0.5 * x * (1.0 + (0.7978845608 * (x + 0.044715 * x * x * x)).tanh())
}

const D: usize = 1152;
const NH: usize = 4;
const DH: usize = 256;
const DFF: usize = 6912;
const EPS: f32 = 1e-6;
const NL: usize = 26;

fn h16(v: f32) -> f32 {
    if std::env::var("FP16").is_err() { return v }
    half::f16::from_f32(v).to_f32()
}
fn v16(v: &mut [f32]) {
    if std::env::var("FP16").is_err() { return }
    for x in v.iter_mut() { *x = half::f16::from_f32(*x).to_f32() }
}

fn main() {
    let dir = PathBuf::from(std::env::args().nth(1).unwrap());
    let toks: Vec<usize> = std::env::args().skip(2).map(|a| a.parse().unwrap()).collect();
    let st = St::load(&dir);
    let emb_scale = (D as f32).sqrt();
    let s = toks.len();
    let emb = st.get("model.embed_tokens.weight");

    // hidden[s][D]
    let mut hid: Vec<Vec<f32>> = toks
        .iter()
        .map(|&t| emb[t * D..(t + 1) * D].iter().map(|v| v * emb_scale).collect())
        .collect();

    let dlayer: usize = std::env::var("REFLAYER")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    for i in 0..NL {
        let p = format!("model.layers.{i}.");
        let sliding = i % 6 != 5;
        let theta = if sliding { 1e4f32 } else { 1e6f32 };
        let anw = st.get(&format!("{p}input_layernorm.weight"));
        let wq = st.get(&format!("{p}self_attn.q_proj.weight"));
        let wk = st.get(&format!("{p}self_attn.k_proj.weight"));
        let wv = st.get(&format!("{p}self_attn.v_proj.weight"));
        let qnw = st.get(&format!("{p}self_attn.q_norm.weight"));
        let knw = st.get(&format!("{p}self_attn.k_norm.weight"));
        let wo = st.get(&format!("{p}self_attn.o_proj.weight"));
        let panw = st.get(&format!("{p}post_attention_layernorm.weight"));
        let fnw = st.get(&format!("{p}pre_feedforward_layernorm.weight"));
        let wg = st.get(&format!("{p}mlp.gate_proj.weight"));
        let wu = st.get(&format!("{p}mlp.up_proj.weight"));
        let wd = st.get(&format!("{p}mlp.down_proj.weight"));
        let pfnw = st.get(&format!("{p}post_feedforward_layernorm.weight"));

        // per-position normed input
        let an: Vec<Vec<f32>> = hid.iter().map(|x| rms_norm(x, &anw, EPS)).collect();
        // K/V cache [pos][nkv*dh]
        let mut ks: Vec<Vec<f32>> = Vec::with_capacity(s);
        let mut vs: Vec<Vec<f32>> = Vec::with_capacity(s);
        for t in 0..s {
            let mut k = matmul(&wk, DH, D, &an[t]);
            let kv = matmul(&wv, DH, D, &an[t]);
            let n = rms_norm(&k, &knw, EPS);
            k.copy_from_slice(&n);
            rope_half(&mut k, DH, t, theta);
            ks.push(k);
            vs.push(kv);
        }
        if i == dlayer && std::env::var("DUMPKV").is_ok() {
            let mut kk: Vec<f32> = Vec::new();
            let mut vv: Vec<f32> = Vec::new();
            for t in 0..8.min(s) {
                kk.extend_from_slice(&ks[t]);
                vv.extend_from_slice(&vs[t]);
            }
            // pad to 8 rows so the concat layout matches the probe
            while kk.len() < 8 * DH { kk.push(0.0); vv.push(0.0); }
            fs::write("/tmp/ref_k.bin", bytemuck::cast_slice(&kk)).unwrap();
            fs::write("/tmp/ref_v.bin", bytemuck::cast_slice(&vv)).unwrap();
        }
        let scale = (256f32).powf(-0.5);
        let mut out: Vec<Vec<f32>> = Vec::with_capacity(s);
        for t in 0..s {
            let mut q = matmul(&wq, NH * DH, D, &an[t]);
            for h in 0..NH {
                let sl = &mut q[h * DH..(h + 1) * DH];
                let n = rms_norm(sl, &qnw, EPS);
                sl.copy_from_slice(&n);
                rope_half(sl, DH, t, theta);
            }
            // attention: scores over keys 0..=t (causal; all within window)
            let mut heads: Vec<f32> = Vec::with_capacity(NH * DH);
            for h in 0..NH {
                let qh = &q[h * DH..(h + 1) * DH];
                let mut sc: Vec<f32> = (0..=t)
                    .map(|k| {
                        qh.iter().zip(&ks[k]).map(|(a, b)| a * b).sum::<f32>() * scale
                    })
                    .collect();
                let mx = sc.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0f32;
                for v in sc.iter_mut() {
                    *v = (*v - mx).exp();
                    sum += *v;
                }
                if i == dlayer && t == s - 1 {
                    let mut idx: Vec<(usize, f32)> = sc.iter().enumerate().map(|(k, &v)| (k, v / sum)).collect();
                    idx.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                    let tops: Vec<String> = idx.iter().take(8).map(|(k, v)| format!("{k}:{v:.3}")).collect();
                    println!("ref row{h}: {}", tops.join(" "));
                }
                if i == dlayer && t == s - 1 {
                    let mut idx: Vec<(usize, f32)> = sc.iter().enumerate().map(|(k, &v)| (k, v / sum)).collect();
                    idx.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                    let tops: Vec<String> = idx.iter().take(8).map(|(k, v)| format!("{k}:{v:.3}")).collect();
                    println!("ref row{h}: {}", tops.join(" "));
                }
                let mut acc = vec![0f32; DH];
                for (k, p) in sc.iter().enumerate() {
                    for j in 0..DH {
                        acc[j] += (*p / sum) * vs[k][j];
                    }
                }
                if i == dlayer && t == s - 1 && std::env::var("DUMPATTN").is_ok() {
                    fs::write(format!("/tmp/ref_attn_h{h}.bin"), bytemuck::cast_slice(&acc)).unwrap();
                }
                heads.extend(acc);
            }
            if std::env::var("NOATTN").is_ok() { heads.iter_mut().for_each(|v| *v = 0.0); }
            let o = matmul(&wo, D, NH * DH, &heads);
            if i == dlayer && t == s - 1 && std::env::var("DUMPO").is_ok() {
                fs::write("/tmp/ref_o.bin", bytemuck::cast_slice(&o)).unwrap();
                let mut hc: Vec<f32> = Vec::new();
                hc.extend_from_slice(&heads);
                fs::write("/tmp/ref_attn4.bin", bytemuck::cast_slice(&hc)).unwrap();
            }
            let pan = rms_norm(&o, &panw, EPS);
            let res1: Vec<f32> = hid[t].iter().zip(&pan).map(|(a, b)| a + b).collect();
            if i == dlayer && t == s - 1 && std::env::var("DUMPRES").is_ok() {
                fs::write("/tmp/ref_res1.bin", bytemuck::cast_slice(&res1)).unwrap();
                let f8: Vec<String> = res1[..8].iter().map(|v| format!("{v:.3}")).collect();
                println!("ref res1 first8: {}", f8.join(" "));
            }
            let fnx = rms_norm(&res1, &fnw, EPS);
            if i == dlayer && t == s - 1 && std::env::var("DUMPFN").is_ok() {
                fs::write("/tmp/ref_fn.bin", bytemuck::cast_slice(&fnx)).unwrap();
            }
            if i == dlayer && std::env::var("DUMPFNALL").is_ok() {
                fs::write(format!("/tmp/ref_fn_p{t}.bin"), bytemuck::cast_slice(&fnx)).unwrap();
                let r1f: Vec<String> = res1[..6].iter().map(|v| format!("{v:.3}")).collect();
                println!("  pos{t} fn[0..3]={:?} res1[0..6]={r1f:?}", &fnx[..3]);
            }
            let g = matmul(&wg, DFF, D, &fnx);
            let u = matmul(&wu, DFF, D, &fnx);
            let ffn: Vec<f32> = g.iter().zip(&u).map(|(a, b)| gelu_tanh(*a) * b).collect();
            if i == dlayer && t == s - 1 && std::env::var("DUMPHID").is_ok() {
                fs::write("/tmp/ref_hid.bin", bytemuck::cast_slice(&ffn)).unwrap();
                let mut gu = g.clone(); gu.extend_from_slice(&u);
                fs::write("/tmp/ref_gu.bin", bytemuck::cast_slice(&gu)).unwrap();
                fs::write("/tmp/ref_g.bin", bytemuck::cast_slice(&g)).unwrap();
            }
            let dn = matmul(&wd, D, DFF, &ffn);
            let pfn = rms_norm(&dn, &pfnw, EPS);
            let hidden: Vec<f32> = res1.iter().zip(&pfn).map(|(a, b)| a + b).collect();
            if i == dlayer && t == s - 1 && std::env::var("DUMPHID").is_ok() {
                fs::write("/tmp/ref_layer_out.bin", bytemuck::cast_slice(&hidden)).unwrap();
            }
            out.push(hidden);
        }
        let last = &out[s - 1];
        let mean_abs = last.iter().map(|v| v.abs()).sum::<f32>() / D as f32;
        if i == dlayer {
            let f8: Vec<String> = last[..8].iter().map(|v| format!("{v:.3}")).collect();
            println!("  l00 first8: {}", f8.join(" "));
        }
        println!("l{i:02} |h|={mean_abs:.4}");
        if std::env::var("DUMPALL").is_ok() {
            fs::write(format!("/tmp/ref_l{i:02}.bin"), bytemuck::cast_slice(last)).unwrap();
        }
        hid = out;
    }
    // final head: norm + tied lm head over first 65536 rows? logits = hidden_norm @ emb^T
    let nw = st.get("model.norm.weight");
    let h = rms_norm(&hid[s - 1], &nw, EPS);
    let vocab = emb.len() / D;
    let mut pairs: Vec<(usize, f32)> = (0..vocab)
        .map(|v| {
            (
                v,
                (0..D).map(|j| emb[v * D + j] * h[j]).sum::<f32>(),
            )
        })
        .collect();
    pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    print!("ref top8:");
    for (id, v) in pairs.iter().take(8) {
        print!(" {id}:{v:.2}");
    }
    println!();
}
