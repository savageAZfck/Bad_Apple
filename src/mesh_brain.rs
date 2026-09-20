//! Mesh-brain: pipeline-parallel model sharding across Bad Apple peers.
//!
//! One model too big for a single Mac is split by decoder layers across
//! mesh nodes. Each node gets a *shard directory* — a self-contained,
//! loadable model dir whose `config.json` declares only its slice of
//! layers and whose safetensors carry only its weights (re-keyed to
//! local layer indices). Activations cross between nodes per forward
//! pass; weights never leave their host.
//!
//! Layout produced per rank:
//!   shard-r<N>/
//!     config.json        — num_hidden_layers rewritten to slice length
//!     shard.json         — {rank, world, layer_start, layer_end}
//!     *.safetensors      — owned weights only, layer keys renumbered
//!     tokenizer files    — cloned verbatim (needed on edge ranks)
//!
//! Verified locally by splitting Qwen2.5-0.5B across two processes; the
//! same path carries 671B-class models across multi-node Studios.

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::fs;
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::time::Duration;

/// One node's shard assignment.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ShardSpec {
    pub rank: usize,
    /// Total ranks in the pipeline.
    #[serde(default)]
    pub world: usize,
    /// Host:port the shard engine listens on (rank 0 is the driver).
    pub host: String,
    /// Host:port of the next rank downstream — absent on the last rank.
    #[serde(default)]
    pub next_host: Option<String>,
    pub layer_start: usize,
    pub layer_end: usize,
    /// Owns embed_tokens (first rank).
    pub has_embed: bool,
    /// Owns norm + lm_head (last rank).
    pub has_head: bool,
    /// Model ties lm_head to embed_tokens — the last rank needs the
    /// embed weights too, since they double as the output head.
    #[serde(default)]
    pub tie_embeddings: bool,
}

/// A full mesh-brain plan.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MeshBrainPlan {
    pub model: String,
    pub world: usize,
    pub ranks: Vec<ShardSpec>,
}

fn layer_re(key: &str) -> Option<(usize, String, String)> {
    // Match "<prefix>layers.<N>.<rest>" — prefix varies per arch
    // ("model.", "language_model.model.", "transformer.", ...).
    let idx = key.find("layers.")?;
    let prefix = &key[..idx];
    let rest = &key[idx + "layers.".len()..];
    let dot = rest.find('.')?;
    let n: usize = rest[..dot].parse().ok()?;
    Some((n, prefix.to_string(), rest[dot + 1..].to_string()))
}

/// Read a model's config.json and return (config, num_hidden_layers).
pub fn read_config(model_dir: &Path) -> Result<(serde_json::Map<String, Value>, usize)> {
    let text = fs::read_to_string(model_dir.join("config.json"))
        .with_context(|| format!("no config.json in {}", model_dir.display()))?;
    let mut cfg: serde_json::Map<String, Value> =
        serde_json::from_str(&text).context("config.json is not a JSON object")?;
    let layers = cfg
        .get("num_hidden_layers")
        .and_then(|v| v.as_u64())
        .or_else(|| {
            cfg.get("text_config")
                .and_then(|t| t.get("num_hidden_layers"))
                .and_then(|v| v.as_u64())
        })
        .context("config.json has no num_hidden_layers")? as usize;
    if layers == 0 {
        bail!("num_hidden_layers is 0");
    }
    // Normalize nested text_config layers too so sharded configs stay honest.
    if let Some(t) = cfg.get_mut("text_config").and_then(|t| t.as_object_mut()) {
        t.insert("num_hidden_layers".into(), Value::from(layers));
    }
    Ok((cfg, layers))
}

/// Compute an even (or memory-weighted) layer split over `world` nodes.
/// `mem_gb` optionally gives each node's usable memory; layers are
/// allocated proportional to share, with edge weights (embed/head)
/// charged against the owning rank's budget.
pub fn plan(model_dir: &Path, hosts: &[String], mem_gb: Option<&[f64]>) -> Result<MeshBrainPlan> {
    let (cfg, total_layers) = read_config(model_dir)?;
    let world = hosts.len();
    if world == 0 {
        bail!("need at least one host");
    }
    if total_layers < world {
        bail!("model has {total_layers} layers — fewer than {world} nodes");
    }

    // Rough edge-weight cost in layer-equivalents: embed + lm_head is
    // ~vocab*hidden*2B*(1 or 2) bytes; estimate and charge rank 0/last.
    let hidden = cfg
        .get("hidden_size")
        .and_then(|v| v.as_u64())
        .unwrap_or(4096) as f64;
    let vocab = cfg
        .get("vocab_size")
        .and_then(|v| v.as_u64())
        .unwrap_or(32000) as f64;
    let layer_gb_est = {
        // crude per-layer estimate: hidden^2 * ~12 bytes (attn+mlp, 4-bit ~ halved)
        (hidden * hidden * 12.0 * 0.5) / 1_073_741_824.0
    };
    let edge_gb = (vocab * hidden * 2.0 * 2.0) / 1_073_741_824.0;
    let edge_layers = (edge_gb / layer_gb_est.max(0.001)).ceil() as usize;

    let weights: Vec<f64> = match mem_gb {
        Some(m) if m.len() == world => {
            let mut w = m.to_vec();
            if world > 1 {
                let budget = edge_layers as f64 * layer_gb_est;
                w[0] = (w[0] - budget).max(1.0);
                w[world - 1] = (w[world - 1] - budget).max(1.0);
            }
            w
        }
        _ => {
            let mut w = vec![1.0f64; world];
            if world > 1 {
                w[0] -= edge_layers as f64 / total_layers as f64;
                w[world - 1] -= edge_layers as f64 / total_layers as f64;
            }
            w
        }
    };
    let wsum: f64 = weights.iter().sum();

    let mut ranks = Vec::with_capacity(world);
    let mut cursor = 0usize;
    for (i, host) in hosts.iter().enumerate() {
        let remaining = total_layers - cursor;
        let nodes_left = world - i;
        let count = if i == world - 1 {
            remaining
        } else {
            let share = ((weights[i] / wsum) * total_layers as f64).round() as usize;
            share.clamp(1, remaining - (nodes_left - 1))
        };
        let tied = cfg
            .get("tie_word_embeddings")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        ranks.push(ShardSpec {
            rank: i,
            world,
            host: host.clone(),
            next_host: hosts.get(i + 1).cloned(),
            layer_start: cursor,
            layer_end: cursor + count,
            has_embed: i == 0,
            has_head: i == world - 1,
            tie_embeddings: tied,
        });
        cursor += count;
    }
    debug_assert_eq!(cursor, total_layers);

    Ok(MeshBrainPlan {
        model: model_dir.display().to_string(),
        world,
        ranks,
    })
}

// ---------------------------------------------------------------------------
// safetensors shard writer
// ---------------------------------------------------------------------------

struct TensorEntry {
    new_key: String,
    dtype: String,
    shape: Vec<u64>,
    data_begin: u64,
    data_end: u64,
}

/// Decide whether a weight key belongs to this rank and what its local
/// name should be. Layer weights renumber to `start..end` local indices —
/// they are the only thing actually split. Every non-layer weight
/// (embed_tokens, final norm, lm_head) ships on EVERY rank: MLX module
/// loading is strict and validates all declared parameters regardless of
/// which ops a rank executes.
fn remap_key(key: &str, spec: &ShardSpec) -> Option<String> {
    if let Some((n, prefix, rest)) = layer_re(key) {
        if n >= spec.layer_start && n < spec.layer_end {
            return Some(format!(
                "{}layers.{}.{}",
                prefix,
                n - spec.layer_start,
                rest
            ));
        }
        return None;
    }
    Some(key.to_string())
}

/// Rewrite one safetensors file into the shard dir, keeping only owned
/// keys and renumbering layer indices. Streams tensor bytes — constant
/// memory regardless of model size. Returns the kept (renamed) keys.
fn shard_safetensors(src: &Path, dst: &Path, spec: &ShardSpec) -> Result<Vec<String>> {
    let mut f = BufReader::with_capacity(8 << 20, fs::File::open(src)?);
    let mut len_buf = [0u8; 8];
    f.read_exact(&mut len_buf)?;
    let header_len = u64::from_le_bytes(len_buf) as usize;
    let mut header = vec![0u8; header_len];
    f.read_exact(&mut header)?;
    let map: BTreeMap<String, Value> =
        serde_json::from_slice(&header).context("bad safetensors header")?;

    let data_start = 8 + header_len as u64;
    let mut kept: Vec<TensorEntry> = Vec::new();
    for (key, meta) in &map {
        if key == "__metadata__" {
            continue;
        }
        let Some(new_key) = remap_key(key, spec) else {
            continue;
        };
        let offs = meta["data_offsets"]
            .as_array()
            .context("tensor entry missing data_offsets")?;
        kept.push(TensorEntry {
            new_key,
            dtype: meta["dtype"].as_str().unwrap_or("BF16").to_string(),
            shape: meta["shape"]
                .as_array()
                .unwrap_or(&vec![])
                .iter()
                .filter_map(|v| v.as_u64())
                .collect(),
            data_begin: offs[0].as_u64().unwrap_or(0),
            data_end: offs[1].as_u64().unwrap_or(0),
        });
    }
    if kept.is_empty() {
        return Ok(Vec::new());
    }
    kept.sort_by_key(|e| e.data_begin);

    // Build the new header with re-based offsets.
    let mut new_map = serde_json::Map::new();
    let mut cursor = 0u64;
    for e in &kept {
        let len = e.data_end - e.data_begin;
        new_map.insert(
            e.new_key.clone(),
            serde_json::json!({
                "dtype": e.dtype,
                "shape": e.shape,
                "data_offsets": [cursor, cursor + len],
            }),
        );
        cursor += len;
    }
    let new_header = serde_json::to_vec(&new_map)?;
    let mut out = BufWriter::with_capacity(8 << 20, fs::File::create(dst)?);
    out.write_all(&(new_header.len() as u64).to_le_bytes())?;
    out.write_all(&new_header)?;

    let mut buf = vec![0u8; 8 << 20];
    let mut file = f.into_inner();
    for e in &kept {
        file.seek(SeekFrom::Start(data_start + e.data_begin))?;
        let mut remaining = e.data_end - e.data_begin;
        while remaining > 0 {
            let chunk = remaining.min(buf.len() as u64) as usize;
            file.read_exact(&mut buf[..chunk])?;
            out.write_all(&buf[..chunk])?;
            remaining -= chunk as u64;
        }
    }
    out.flush()?;
    Ok(kept.into_iter().map(|e| e.new_key).collect())
}

/// Clone auxiliary (non-weight) files the engine still needs.
fn copy_aux(model_dir: &Path, out_dir: &Path, spec: &ShardSpec) -> Result<()> {
    for entry in fs::read_dir(model_dir)? {
        let entry = entry?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name.ends_with(".safetensors")
            || name == "config.json"
            || name.ends_with(".safetensors.index.json")
        {
            continue;
        }
        // Tokenizer only needed on edge ranks; keep shards lean elsewhere.
        let tok = name.contains("tokenizer")
            || name == "vocab.json"
            || name == "merges.txt"
            || name == "special_tokens_map.json";
        if tok && !(spec.has_embed || spec.has_head) {
            continue;
        }
        let dst = out_dir.join(name.as_ref());
        // APFS clonefile: instant, copy-on-write, zero duplicate bytes.
        #[cfg(target_os = "macos")]
        {
            use std::ffi::CString;
            let s = CString::new(entry.path().to_string_lossy().as_bytes())?;
            let d = CString::new(dst.to_string_lossy().as_bytes())?;
            if unsafe { libc::clonefile(s.as_ptr(), d.as_ptr(), 0) } == 0 {
                continue;
            }
        }
        fs::copy(entry.path(), &dst).with_context(|| format!("failed to copy {}", name))?;
    }
    Ok(())
}

/// Build the shard directory for `spec` from a full local model dir.
pub fn build_shard(model_dir: &Path, out_dir: &Path, spec: &ShardSpec) -> Result<u64> {
    fs::create_dir_all(out_dir)?;
    let (mut cfg, _total) = read_config(model_dir)?;
    cfg.insert(
        "num_hidden_layers".into(),
        Value::from(spec.layer_end - spec.layer_start),
    );
    cfg.insert(
        "badapple_shard".into(),
        serde_json::json!({
            "rank": spec.rank,
            "layer_start": spec.layer_start, "layer_end": spec.layer_end,
        }),
    );
    fs::write(
        out_dir.join("config.json"),
        serde_json::to_string_pretty(&Value::Object(cfg))?,
    )?;
    fs::write(
        out_dir.join("shard.json"),
        serde_json::to_string_pretty(spec)?,
    )?;
    // Runtime spec consumed by BadAppleShardRuntime (camelCase Codable keys).
    fs::write(
        out_dir.join("mesh_brain_rank.json"),
        serde_json::to_string_pretty(&serde_json::json!({
            "rank": spec.rank,
            "world": spec.world,
            "host": spec.host,
            "nextHost": spec.next_host,
            "hasEmbed": spec.has_embed,
            "hasHead": spec.has_head,
            "layerStart": spec.layer_start,
            "layerEnd": spec.layer_end,
        }))?,
    )?;

    let mut tensors = 0u64;
    let mut weight_map = serde_json::Map::new();
    for entry in fs::read_dir(model_dir)? {
        let path = entry?.path();
        if path.extension().and_then(|e| e.to_str()) != Some("safetensors") {
            continue;
        }
        let fname = path.file_name().unwrap().to_string_lossy().to_string();
        for key in shard_safetensors(&path, &out_dir.join(&fname), spec)? {
            weight_map.insert(key, Value::from(fname.clone()));
            tensors += 1;
        }
    }
    if tensors == 0 {
        bail!("shard produced no tensors — check layer range and weight naming");
    }
    // Regenerate the index if the source model had one — the original
    // references global keys that no longer exist in this shard.
    if model_dir.join("model.safetensors.index.json").exists() {
        let total: u64 = weight_map.len() as u64;
        fs::write(
            out_dir.join("model.safetensors.index.json"),
            serde_json::to_string_pretty(&serde_json::json!({
                "metadata": {"total_size": 0},
                "weight_map": Value::Object(weight_map),
                "_badapple_tensors": total,
            }))?,
        )?;
    }
    copy_aux(model_dir, out_dir, spec)?;
    Ok(tensors)
}

/// Locate a model's local dir: accept a direct path or an HF repo id
/// resolvable under the HF cache.
pub fn resolve_model_dir(model: &str) -> Result<PathBuf> {
    let p = PathBuf::from(model);
    if p.is_dir() {
        return Ok(p);
    }
    let cache = dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("~"))
        .join(".cache/huggingface/hub")
        .join(format!("models--{}", model.replace('/', "--")));
    let snaps = cache.join("snapshots");
    if snaps.is_dir() {
        for e in fs::read_dir(&snaps)? {
            let e = e?;
            if e.path().is_dir() {
                return Ok(e.path());
            }
        }
    }
    bail!("model not found locally: {model} (expected a dir or a cached HF repo id)")
}

// ---------------------------------------------------------------------------
// client — drive a rank-0 shard server over the mesh-brain wire protocol
// ---------------------------------------------------------------------------

/// Frame header shared with BadAppleShard.swift: [u32 BE jsonLen][json][payload].
#[derive(Serialize, Deserialize)]
struct MeshBrainFrame {
    op: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "maxTokens")]
    max_tokens: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    token: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    done: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

fn write_frame(s: &mut TcpStream, frame: &MeshBrainFrame) -> Result<()> {
    let json = serde_json::to_vec(frame)?;
    s.write_all(&(json.len() as u32).to_be_bytes())?;
    s.write_all(&json)?;
    Ok(())
}

fn read_frame(s: &mut TcpStream) -> Result<MeshBrainFrame> {
    let mut len_buf = [0u8; 4];
    s.read_exact(&mut len_buf)?;
    let len = u32::from_be_bytes(len_buf) as usize;
    if len > 64 * 1024 {
        bail!("frame header too large: {len}");
    }
    let mut buf = vec![0u8; len];
    s.read_exact(&mut buf)?;
    Ok(serde_json::from_slice(&buf)?)
}

/// Ping a rank — returns its rank number as a liveness check.
pub fn ping(host: &str) -> Result<i64> {
    let mut s = TcpStream::connect(host).with_context(|| format!("connect {host}"))?;
    s.set_read_timeout(Some(Duration::from_secs(30)))?;
    write_frame(
        &mut s,
        &MeshBrainFrame {
            op: "ping".into(),
            prompt: None,
            max_tokens: None,
            token: None,
            done: None,
            text: None,
            error: None,
        },
    )?;
    let resp = read_frame(&mut s)?;
    if let Some(e) = resp.error {
        bail!("peer error: {e}");
    }
    resp.token.context("ping response missing rank")
}

/// Ask the pipeline to generate: sends the prompt to the embed rank (rank 0)
/// and returns decoded text produced by the last rank's head.
pub fn ask(host: &str, prompt: &str, max_tokens: usize) -> Result<String> {
    let mut s = TcpStream::connect(host).with_context(|| format!("connect {host}"))?;
    s.set_read_timeout(Some(Duration::from_secs(600)))?;
    write_frame(
        &mut s,
        &MeshBrainFrame {
            op: "generate".into(),
            prompt: Some(prompt.into()),
            max_tokens: Some(max_tokens),
            token: None,
            done: None,
            text: None,
            error: None,
        },
    )?;
    let resp = read_frame(&mut s)?;
    if let Some(e) = resp.error {
        bail!("mesh-brain error: {e}");
    }
    resp.text.context("generate response missing text")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layer_re_parses() {
        let (n, p, r) = layer_re("model.layers.14.self_attn.q_proj.weight").unwrap();
        assert_eq!(
            (n, p.as_str(), r.as_str()),
            (14, "model.", "self_attn.q_proj.weight")
        );
    }

    #[test]
    fn plan_covers_all_layers() {
        let dir = std::env::temp_dir().join(format!("mbplan-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            dir.join("config.json"),
            r#"{"num_hidden_layers": 28, "hidden_size": 896, "vocab_size": 151936}"#,
        )
        .unwrap();
        let hosts = vec!["a:8741".to_string(), "b:8741".to_string()];
        let p = plan(&dir, &hosts, None).unwrap();
        assert_eq!(p.ranks.len(), 2);
        assert_eq!(p.ranks[0].layer_start, 0);
        assert_eq!(p.ranks[1].layer_end, 28);
        assert_eq!(p.ranks[0].layer_end, p.ranks[1].layer_start);
        assert!(p.ranks[0].has_embed && !p.ranks[0].has_head);
        assert!(!p.ranks[1].has_embed && p.ranks[1].has_head);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn remap_renumbers_layers() {
        let spec = ShardSpec {
            rank: 1,
            host: "b:8741".into(),
            layer_start: 14,
            layer_end: 28,
            has_embed: false,
            has_head: true,
        };
        assert_eq!(
            remap_key("model.layers.14.mlp.down_proj.weight", &spec).as_deref(),
            Some("model.layers.0.mlp.down_proj.weight")
        );
        assert_eq!(
            remap_key("model.layers.2.mlp.down_proj.weight", &spec),
            None
        );
        assert_eq!(
            remap_key("model.norm.weight", &spec).as_deref(),
            Some("model.norm.weight")
        );
        assert_eq!(remap_key("model.embed_tokens.weight", &spec), None);
    }
}
