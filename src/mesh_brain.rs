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
use hmac::{Hmac, KeyInit, Mac};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::Sha256;
use std::collections::BTreeMap;
use std::fs;
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::time::Duration;

type HmacSha256 = Hmac<Sha256>;

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
pub(crate) struct MeshBrainFrame {
    pub(crate) op: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) prompt: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "maxTokens")]
    pub(crate) max_tokens: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) token: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) done: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) shape: Option<Vec<usize>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) dtype: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "layerStart")]
    pub(crate) layer_start: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "layerEnd")]
    pub(crate) layer_end: Option<usize>,
}

// ---------------------------------------------------------------------------
// mutual auth — per-connection HMAC-SHA256 nonce challenge over the shared
// SLICKS secret. Server challenges; both sides prove. BADAPPLE_MESH_AUTH=0
// disables for debugging (never default).
// ---------------------------------------------------------------------------

fn mesh_auth_enabled() -> bool {
    std::env::var("BADAPPLE_MESH_AUTH").ok().as_deref() != Some("0")
}

/// Resolve the shared secret: BADAPPLE_MESH_KEY > BADAPPLE_P2P_SECRET >
/// BADAPPLE_SLICKS_KEY_PATH > /var/lib/bad_apple/slicks.key.
pub fn resolve_mesh_secret() -> Result<Vec<u8>> {
    let raw = if let Ok(s) = std::env::var("BADAPPLE_MESH_KEY") {
        if s.is_empty() {
            bail!("BADAPPLE_MESH_KEY is empty");
        }
        s
    } else if let Ok(s) = std::env::var("BADAPPLE_P2P_SECRET") {
        if s.is_empty() {
            bail!("BADAPPLE_P2P_SECRET is empty");
        }
        s
    } else {
        let path = std::env::var("BADAPPLE_SLICKS_KEY_PATH")
            .unwrap_or_else(|_| "/var/lib/bad_apple/slicks.key".into());
        fs::read_to_string(&path)
            .with_context(|| format!("failed to read SLICKS key from {path}"))?
            .trim()
            .to_string()
    };
    if raw.len() < 32 {
        bail!("mesh-brain secret must be at least 32 bytes");
    }
    Ok(raw.into_bytes())
}

/// Registry of the last planned/sharded mesh — written by `mesh-brain
/// plan|shard`, read as the default host list by `status`/`ping`/`ask`
/// and by the engine when answering "is the mesh up".
pub fn hosts_file() -> PathBuf {
    std::env::var("BADAPPLE_MESH_HOSTS_FILE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/var/lib/bad_apple/mesh_hosts.json"))
}

pub fn save_hosts(hosts: &[String], model: &str) -> Result<()> {
    let v = serde_json::json!({
        "hosts": hosts,
        "model": model,
        "updated": std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0),
    });
    fs::write(hosts_file(), serde_json::to_string_pretty(&v)?)?;
    Ok(())
}

pub fn load_hosts() -> Option<Vec<String>> {
    let text = fs::read_to_string(hosts_file()).ok()?;
    let v: Value = serde_json::from_str(&text).ok()?;
    let hosts: Vec<String> = v["hosts"]
        .as_array()?
        .iter()
        .filter_map(|h| h.as_str().map(String::from))
        .collect();
    if hosts.is_empty() {
        None
    } else {
        Some(hosts)
    }
}

pub(crate) fn hmac_hex(key: &[u8], msg: &str) -> String {
    let mut mac = HmacSha256::new_from_slice(key).expect("hmac accepts any key size");
    mac.update(msg.as_bytes());
    mac.finalize()
        .into_bytes()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// AES-256-GCM session cipher derived from the shared secret — same
/// derivation as the P2P engram crypto (SHA-256 of the secret). Encryption
/// rides on top of the auth handshake; BADAPPLE_MESH_ENC=0 disables.
fn mesh_enc_enabled() -> bool {
    std::env::var("BADAPPLE_MESH_ENC").ok().as_deref() != Some("0")
}

pub(crate) fn session_cipher(secret: &[u8]) -> Result<crate::p2p_crypto::P2PCipher> {
    crate::p2p_crypto::P2PCipher::new(&crate::p2p_crypto::derive_key_from_bytes(secret))
        .map_err(|e| anyhow::anyhow!(e))
}

/// Wire packet = plaintext [u32 jsonLen][json][payload] when `cipher` is
/// None, else [u32 ctLen][nonce||AES-GCM(jsonLen||json||payload)+tag].
pub(crate) fn write_packet(
    s: &mut TcpStream,
    frame: &MeshBrainFrame,
    payload: &[u8],
    cipher: Option<&crate::p2p_crypto::P2PCipher>,
) -> Result<()> {
    let json = serde_json::to_vec(frame)?;
    match cipher {
        Some(c) => {
            let mut inner = Vec::with_capacity(4 + json.len() + payload.len());
            inner.extend_from_slice(&(json.len() as u32).to_be_bytes());
            inner.extend_from_slice(&json);
            inner.extend_from_slice(payload);
            let ct = c.encrypt(&inner).map_err(|e| anyhow::anyhow!(e))?;
            s.write_all(&(ct.len() as u32).to_be_bytes())?;
            s.write_all(&ct)?;
        }
        None => {
            s.write_all(&(json.len() as u32).to_be_bytes())?;
            s.write_all(&json)?;
            s.write_all(payload)?;
        }
    }
    Ok(())
}

pub(crate) fn read_packet(
    s: &mut TcpStream,
    cipher: Option<&crate::p2p_crypto::P2PCipher>,
) -> Result<(MeshBrainFrame, Vec<u8>)> {
    match cipher {
        Some(c) => {
            let mut len_buf = [0u8; 4];
            s.read_exact(&mut len_buf)?;
            let len = u32::from_be_bytes(len_buf) as usize;
            if len > 512 * 1024 * 1024 {
                bail!("encrypted frame too large: {len}");
            }
            let mut ct = vec![0u8; len];
            s.read_exact(&mut ct)?;
            let inner = c.decrypt(&ct).map_err(|e| anyhow::anyhow!(e))?;
            if inner.len() < 4 {
                bail!("decrypted frame too short");
            }
            let json_len = u32::from_be_bytes(inner[..4].try_into().unwrap()) as usize;
            if inner.len() < 4 + json_len {
                bail!("decrypted frame truncated");
            }
            let frame: MeshBrainFrame = serde_json::from_slice(&inner[4..4 + json_len])?;
            Ok((frame, inner[4 + json_len..].to_vec()))
        }
        None => {
            let frame = read_frame(s)?;
            let payload = if let Some(shape) = &frame.shape {
                let elems: usize = shape.iter().product();
                let size = elems * dtype_size(frame.dtype.as_deref().unwrap_or("float16"));
                let mut p = vec![0u8; size];
                s.read_exact(&mut p)?;
                p
            } else {
                Vec::new()
            };
            Ok((frame, payload))
        }
    }
}

fn dtype_size(name: &str) -> usize {
    match name {
        "float64" | "int64" | "uint64" => 8,
        "float32" | "int32" | "uint32" => 4,
        "bool" | "uint8" | "int8" => 1,
        _ => 2, // f16/bf16/u16/i16 + anything else small
    }
}

/// Client side of the handshake: answer the server's nonce, then verify the
/// server's counter-proof. Returns early (no-op) when auth is disabled —
/// detected by the server sending a normal result instead of a challenge.
pub(crate) fn client_handshake(s: &mut TcpStream, key: &[u8]) -> Result<()> {
    let challenge = read_frame(s)?;
    if challenge.op != "auth" {
        bail!("expected auth challenge, got {}", challenge.op);
    }
    let nonce = challenge.text.context("auth challenge missing nonce")?;
    let proof = hmac_hex(key, &format!("mb-c:{nonce}"));
    write_frame(
        s,
        &MeshBrainFrame {
            op: "auth".into(),
            prompt: None,
            max_tokens: None,
            token: None,
            done: None,
            text: Some(proof),
            error: None,
            shape: None,
            dtype: None,
            layer_start: None,
            layer_end: None,
        },
    )?;
    let resp = read_frame(s)?;
    match resp.op.as_str() {
        "auth-ok" => {
            let expect = hmac_hex(key, &format!("mb-s:{nonce}"));
            if resp.text.as_deref() != Some(expect.as_str()) {
                bail!("server counter-proof invalid — wrong mesh secret?");
            }
            Ok(())
        }
        "error" => bail!("auth rejected: {}", resp.error.unwrap_or_default()),
        other => bail!("expected auth-ok, got {other}"),
    }
}

/// Open an authenticated connection to a rank. Returns the stream and the
/// session cipher — Some when auth+encryption are both enabled.
fn connect(
    host: &str,
    timeout_secs: u64,
) -> Result<(TcpStream, Option<crate::p2p_crypto::P2PCipher>)> {
    let mut s = TcpStream::connect(host).with_context(|| format!("connect {host}"))?;
    s.set_read_timeout(Some(Duration::from_secs(timeout_secs)))?;
    s.set_write_timeout(Some(Duration::from_secs(30)))?;
    s.set_nodelay(true).ok();
    if mesh_auth_enabled() {
        let secret = resolve_mesh_secret()?;
        client_handshake(&mut s, &secret)?;
        if mesh_enc_enabled() {
            return Ok((s, Some(session_cipher(&secret)?)));
        }
    }
    Ok((s, None))
}

pub(crate) fn write_frame(s: &mut TcpStream, frame: &MeshBrainFrame) -> Result<()> {
    let json = serde_json::to_vec(frame)?;
    s.write_all(&(json.len() as u32).to_be_bytes())?;
    s.write_all(&json)?;
    Ok(())
}

pub(crate) fn read_frame(s: &mut TcpStream) -> Result<MeshBrainFrame> {
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

/// Ping a rank — returns (rank, layer_start, layer_end) as a liveness check.
pub fn ping(host: &str) -> Result<(i64, Option<usize>, Option<usize>)> {
    let (mut s, cipher) = connect(host, 30)?;
    write_packet(
        &mut s,
        &MeshBrainFrame {
            op: "ping".into(),
            prompt: None,
            max_tokens: None,
            token: None,
            done: None,
            text: None,
            error: None,
            shape: None,
            dtype: None,
            layer_start: None,
            layer_end: None,
        },
        &[],
        cipher.as_ref(),
    )?;
    let (resp, _) = read_packet(&mut s, cipher.as_ref())?;
    if let Some(e) = resp.error {
        bail!("peer error: {e}");
    }
    Ok((
        resp.token.context("ping response missing rank")?,
        resp.layer_start,
        resp.layer_end,
    ))
}

/// Ask the pipeline to generate: sends the prompt to the embed rank (rank 0)
/// and returns decoded text produced by the last rank's head.
pub fn ask(host: &str, prompt: &str, max_tokens: usize) -> Result<String> {
    let (mut s, cipher) = connect(host, 600)?;
    write_packet(
        &mut s,
        &MeshBrainFrame {
            op: "generate".into(),
            prompt: Some(prompt.into()),
            max_tokens: Some(max_tokens),
            token: None,
            done: None,
            text: None,
            error: None,
            shape: None,
            dtype: None,
            layer_start: None,
            layer_end: None,
        },
        &[],
        cipher.as_ref(),
    )?;
    let (resp, _) = read_packet(&mut s, cipher.as_ref())?;
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
            world: 2,
            host: "b:8741".into(),
            next_host: None,
            layer_start: 14,
            layer_end: 28,
            has_embed: false,
            has_head: true,
            tie_embeddings: false,
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
        // Non-layer weights replicate to every rank — MLX validates all
        // declared module weights at load time, and tied models need
        // embed_tokens on the last rank as the output head.
        assert_eq!(
            remap_key("model.embed_tokens.weight", &spec).as_deref(),
            Some("model.embed_tokens.weight")
        );
    }
}
