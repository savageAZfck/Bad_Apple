//! Native organ layer — the Coyote-free implementation.
//!
//! Same public API as `coyote_organs.rs` (the `.cy` organs were ported from
//! this code), routed to the native Rust implementations: `axon` for the
//! reflex arc, `firewall_machine` for the output firewall, a pure-Rust MLP
//! for gatekeeper classify/train, and faithful ports of the marshal
//! (bounded autonomy), council (quorum tally), and camazotz (erasure
//! certificate) organ semantics.
//!
//! Enabled via the `native-organs` cargo feature; `lib.rs` re-exports this
//! module as `coyote_organs` so every call site compiles unchanged.

use anyhow::{bail, Context, Result};
use axon::{Reflex, ReflexAction, ReflexArc, SenseEvent, ThresholdOp, Trigger};
use half::f16;
use mil_infer::graph::{compile_graph, GraphRuntime};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

const GK_WEIGHT_MAGIC: &[u8; 8] = b"CYGKW01\0";
const GK_DIM: usize = 64;
const GK_LR: f64 = 0.15;

/// State root: `BADAPPLE_ORGANS_DIR` (default `/var/lib/bad_apple/organs`).
pub fn organs_dir() -> PathBuf {
    std::env::var("BADAPPLE_ORGANS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/var/lib/bad_apple/organs"))
}

fn ensure_state() -> Result<()> {
    std::fs::create_dir_all(organs_dir())?;
    Ok(())
}

fn now_ms() -> i64 {
    chrono::Utc::now().timestamp_millis()
}

fn sha256_hex(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

// ------------------------------------------------------------------- arc

fn arc_path() -> PathBuf {
    organs_dir().join("arc.json")
}

fn load_arc() -> Result<ReflexArc> {
    let path = arc_path();
    if path.exists() {
        let data = std::fs::read_to_string(&path)
            .with_context(|| format!("cannot read {}", path.display()))?;
        Ok(serde_json::from_str(&data).context("corrupt arc.json")?)
    } else {
        Ok(ReflexArc::new())
    }
}

fn save_arc(arc: &ReflexArc) -> Result<()> {
    ensure_state()?;
    let path = arc_path();
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, serde_json::to_string_pretty(arc)?)
        .with_context(|| format!("cannot write {}", tmp.display()))?;
    std::fs::rename(&tmp, &path)?;
    Ok(())
}

/// Trigger specs: `kind:<glob>` `field:<name>=<glob>` `gt|ge|lt|le:<field>=<n>`
fn parse_trigger(spec: &str) -> Result<Trigger> {
    let (prefix, rest) = spec
        .split_once(':')
        .context("trigger spec needs `kind:`/`field:`/`gt:`/`lt:`/`ge:`/`le:`")?;
    match prefix {
        "kind" => Ok(Trigger::KindIs(rest.to_string())),
        "field" => {
            let (name, glob) = rest.split_once('=').context("field: needs name=glob")?;
            Ok(Trigger::FieldMatch {
                field: name.to_string(),
                pattern: glob.to_string(),
            })
        }
        op @ ("gt" | "ge" | "lt" | "le") => {
            let (name, n) = rest.split_once('=').context("threshold needs field=n")?;
            Ok(Trigger::Threshold {
                field: name.to_string(),
                op: match op {
                    "gt" => ThresholdOp::Gt,
                    "ge" => ThresholdOp::Ge,
                    "lt" => ThresholdOp::Lt,
                    _ => ThresholdOp::Le,
                },
                value: n.parse().context("threshold value must be a number")?,
            })
        }
        other => bail!("unknown trigger prefix: {other}"),
    }
}

fn parse_action(action: &str) -> Result<ReflexAction> {
    match action {
        "brake" => Ok(ReflexAction::Brake),
        "alert" => Ok(ReflexAction::Alert),
        "quarantine" => Ok(ReflexAction::Quarantine),
        other => bail!("unknown action: {other}"),
    }
}

pub fn arc_install(
    id: &str,
    trigger: &str,
    action: &str,
    cooldown_ms: i64,
    max_firings: i64,
) -> Result<String> {
    let mut arc = load_arc()?;
    arc.install(
        Reflex::new(id, parse_trigger(trigger)?, parse_action(action)?)
            .with_bounds(cooldown_ms as u64, max_firings as u64),
    )?;
    save_arc(&arc)?;
    Ok(json!({"status": "installed", "reflex": id}).to_string())
}

pub fn arc_remove(id: &str) -> Result<String> {
    let mut arc = load_arc()?;
    if !arc.remove(id) {
        bail!("no such reflex: {id}");
    }
    save_arc(&arc)?;
    Ok(json!({"status": "removed", "reflex": id}).to_string())
}

pub fn arc_list() -> Result<String> {
    let arc = load_arc()?;
    Ok(serde_json::to_string_pretty(&arc.reflexes)?)
}

pub fn arc_sense(kind: &str, data: &serde_json::Value) -> Result<String> {
    let mut arc = load_arc()?;
    let fired = arc.sense(&SenseEvent::new(kind, data.clone()));
    save_arc(&arc)?;
    Ok(json!({"fired": fired, "frozen": arc.frozen}).to_string())
}

pub fn arc_clear_freeze() -> Result<String> {
    let mut arc = load_arc()?;
    let was = arc.clear_freeze();
    save_arc(&arc)?;
    Ok(json!({"status": "cleared", "was_frozen": was}).to_string())
}

pub fn arc_rearm(id: &str) -> Result<String> {
    let mut arc = load_arc()?;
    if !arc.rearm(id) {
        bail!("no such reflex: {id}");
    }
    save_arc(&arc)?;
    Ok(json!({"status": "rearmed", "reflex": id}).to_string())
}

pub fn arc_status() -> Result<String> {
    let arc = load_arc()?;
    Ok(serde_json::to_string_pretty(&arc)?)
}

pub fn arc_verify() -> Result<String> {
    let arc = load_arc()?;
    let ok = arc.verify().is_ok();
    Ok(json!({"chain_ok": ok}).to_string())
}

// --------------------------------------------------------------- marshal

// akicita — the bounded-autonomy marshal, direct from the crate the .cy
// organ was ported from. `Akicita::check` decides Permit/Hold/Deny over
// its own state (level, dedupe ring, rate limit, protected surfaces).
fn marshal() -> &'static std::sync::Mutex<akicita::Akicita> {
    static M: std::sync::OnceLock<std::sync::Mutex<akicita::Akicita>> = std::sync::OnceLock::new();
    M.get_or_init(|| {
        let level_file = organs_dir().join("marshal_level.txt");
        let _ = ensure_state();
        let level = akicita::Level::load(&level_file);
        let cfg = akicita::Config::default();
        std::sync::Mutex::new(akicita::Akicita::new(level, cfg, Some(level_file)))
    })
}

pub fn marshal_check(surface: &str, old: &str, new: &str, intent: &str) -> Result<String> {
    let m = marshal()
        .lock()
        .map_err(|e| anyhow::anyhow!("marshal lock: {e}"))?;
    let proposal = akicita::Proposal::patch(surface, old, new, intent);
    Ok(match m.check(&proposal) {
        akicita::Decision::Permit => "permit=".to_string(),
        akicita::Decision::Hold { reason } => format!("hold={reason}"),
        akicita::Decision::Deny { reason } => format!("deny={reason}"),
    })
}

pub fn marshal_set_level(level: &str) -> Result<String> {
    if !["off", "suggest", "safe-apply", "full"].contains(&level) {
        return Ok(format!("error=unknown level {level}"));
    }
    let m = marshal()
        .lock()
        .map_err(|e| anyhow::anyhow!("marshal lock: {e}"))?;
    m.set_level(akicita::Level::parse(level));
    Ok(format!("level={level}"))
}

pub fn marshal_status() -> Result<String> {
    let m = marshal()
        .lock()
        .map_err(|e| anyhow::anyhow!("marshal lock: {e}"))?;
    let level = match m.level() {
        akicita::Level::Off => "off",
        akicita::Level::Suggest => "suggest",
        akicita::Level::SafeApply => "safe-apply",
        akicita::Level::Full => "full",
    };
    Ok(format!("level={level}"))
}

// -------------------------------------------------------------- firewall

// macuahuitl — the blocklist DFA firewall, direct from the crate.
// Built-in key-material sentinels merge with the operator's blocklist,
// then the automaton scans the text.
pub fn firewall_scan(text: &str) -> Result<String> {
    let blocklist = std::env::var("BADAPPLE_BLOCKLIST")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/var/lib/bad_apple/blocklist.txt"));
    let body = std::fs::read_to_string(&blocklist).unwrap_or_default();
    let fw = macuahuitl::Firewall::with_blocklist(&body);
    let mut stream = fw.stream();
    let blocked = matches!(stream.push(text), macuahuitl::Verdict::Blocked);
    // `hit=-1` means clean — the gatekeeper checks that sentinel.
    let hit = if blocked { 1 } else { -1 };
    Ok(format!("hit={}|patterns={}", hit, fw.patterns().len()))
}

// --------------------------------------------------------------- council

/// tokala — votes arrive as `seat|yes|reason` rows; tally, preserve the first
/// dissenting reason, and return the verdict string.
pub fn council(topic: &str, votes: &[String]) -> Result<String> {
    let mut yes = 0i64;
    let mut no = 0i64;
    let mut dissent = String::new();
    let mut rows = Vec::with_capacity(votes.len());
    for v in votes {
        let p: Vec<&str> = v.split('|').collect();
        if p.len() == 3 {
            if p[1] == "yes" {
                yes += 1;
            } else {
                no += 1;
                if dissent.is_empty() {
                    dissent = p[2].to_string();
                }
            }
            rows.push(format!("{}|{}|{}", p[0], p[1], p[2]));
        }
    }
    let approved = yes > no;
    let hash = sha256_hex(format!("{topic}|{}", rows.join(",")).as_bytes());
    let mut acc = format!(
        "approved={}|yes={}|no={}|hash={}",
        if approved { 1 } else { 0 },
        yes,
        no,
        hash
    );
    if !dissent.is_empty() {
        acc.push_str(&format!("|dissent={dissent}"));
    }
    Ok(acc)
}

// --------------------------------------------------------------- erasure

// camazotz — certified cryptographic erasure, direct from the crate.
// The subject's payload is sealed under its own vault key, the key is
// destroyed, and the resulting certificate proves it cannot be read back.
pub fn erase_subject(subject: &str, payload: &str) -> Result<String> {
    let dir = organs_dir().join("camazotz");
    std::fs::create_dir_all(&dir)?;
    let vault = camazotz::Vault::new(dir.join("vault"))?;
    vault.seal(payload.as_bytes(), &camazotz::subject_ref(subject))?;
    let erasure = camazotz::Erasure::new(vault, dir.join("certs"));
    let cert = erasure.forget(subject)?;
    let cert_hash = sha256_hex(&cert.signing_body());
    Ok(format!("cert={cert_hash}"))
}

// ------------------------------------------------------ gatekeeper router

/// Deterministic 64-D L2-normalized signed feature-hashed embedding.
/// Verbatim port — the classifier trains/scores on this regardless of the
/// execution backend.
fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

pub fn embed_text_64(text: &str) -> Vec<f32> {
    let mut out = vec![0.0f32; 64];
    let lower = text.to_lowercase();
    let words: Vec<&str> = lower
        .split(|c: char| !c.is_alphanumeric() && c != '_' && c != '-')
        .filter(|w| !w.is_empty())
        .collect();
    for (idx, &w) in words.iter().enumerate() {
        let h1 = fnv1a64(w.as_bytes());
        let b1 = (h1 as usize) % 60;
        let s1 = if (h1 >> 8) & 1 == 0 { 1.0f32 } else { -1.0f32 };
        out[b1] += s1;
        let h2 = h1.wrapping_mul(0x9E3779B97F4A7C15);
        let b2 = ((h2 >> 16) as usize) % 60;
        let s2 = if (h2 >> 24) & 1 == 0 { 0.5f32 } else { -0.5f32 };
        out[b2] += s2;
        if idx + 1 < words.len() {
            let bg = format!("{w}_{}", words[idx + 1]);
            let hb = fnv1a64(bg.as_bytes());
            let bb = (hb as usize) % 60;
            let sb = if (hb >> 12) & 1 == 0 {
                0.75f32
            } else {
                -0.75f32
            };
            out[bb] += sb;
        }
    }
    out[60] = (words.len() as f32 / 12.0).min(2.0);
    out[61] = (lower.len() as f32 / 80.0).min(2.0);
    out[62] = if lower.contains('/') || lower.contains(".wasm") || lower.contains(".cy") {
        1.0
    } else {
        0.0
    };
    out[63] = 1.0;
    let norm: f32 = out.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for v in &mut out {
            *v /= norm;
        }
    }
    out
}

/// The trained router: y = sigmoid(x @ w1) @ w2. Two f32[64,64] matrices —
/// the same shape the Coyote program compiled to the ANE, executed here in
/// plain Rust with identical math.
struct GkBrain {
    w1: Vec<f64>,
    w2: Vec<f64>,
}

impl GkBrain {
    fn new() -> Self {
        // Deterministic small-random init so w1 isn't locked at zero grads.
        let mut w1 = vec![0.0f64; GK_DIM * GK_DIM];
        let mut w2 = vec![0.0f64; GK_DIM * GK_DIM];
        let mut state = 0x9E3779B97F4A7C15u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state % 2000) as f64 / 2000.0) * 0.02 - 0.01
        };
        for v in w1.iter_mut().chain(w2.iter_mut()) {
            *v = next();
        }
        GkBrain { w1, w2 }
    }

    fn forward(&self, x: &[f64]) -> (Vec<f64>, Vec<f64>) {
        // h = sigmoid(x @ w1)
        let mut h = vec![0.0f64; GK_DIM];
        for i in 0..GK_DIM {
            let mut s = 0.0;
            for j in 0..GK_DIM {
                s += x[j] * self.w1[j * GK_DIM + i];
            }
            h[i] = 1.0 / (1.0 + (-s).exp());
        }
        // y = h @ w2
        let mut y = vec![0.0f64; GK_DIM];
        for i in 0..GK_DIM {
            let mut s = 0.0;
            for j in 0..GK_DIM {
                s += h[j] * self.w2[j * GK_DIM + i];
            }
            y[i] = s;
        }
        (h, y)
    }

    /// One SGD step: MSE loss, lr = 0.15 — same hyperparameters the Coyote
    /// `organ_train` program used on-chip.
    fn train_step(&mut self, x: &[f64], t: &[f64]) -> f64 {
        let (h, y) = self.forward(x);
        let mut loss = 0.0;
        let mut dy = vec![0.0f64; GK_DIM];
        for i in 0..GK_DIM {
            let d = y[i] - t[i];
            loss += d * d;
            dy[i] = 2.0 * d / GK_DIM as f64;
        }
        loss /= GK_DIM as f64;
        // dw2 = hᵀ ⊗ dy ; dh = dy @ w2ᵀ ; dh *= h ⊙ (1-h) ; dw1 = xᵀ ⊗ dh
        let mut dh = vec![0.0f64; GK_DIM];
        for i in 0..GK_DIM {
            let mut s = 0.0;
            for j in 0..GK_DIM {
                s += dy[j] * self.w2[i * GK_DIM + j];
            }
            dh[i] = s * h[i] * (1.0 - h[i]);
        }
        for i in 0..GK_DIM {
            for j in 0..GK_DIM {
                self.w2[i * GK_DIM + j] -= GK_LR * h[i] * dy[j];
                self.w1[i * GK_DIM + j] -= GK_LR * x[i] * dh[j];
            }
        }
        loss
    }

    fn weights_path() -> PathBuf {
        organs_dir().join("gatekeeper_ane.bin")
    }

    fn load_or_init() -> Result<Self> {
        ensure_state()?;
        let path = Self::weights_path();
        if path.exists() {
            let bytes = std::fs::read(&path)?;
            let expected = 8 + (GK_DIM * GK_DIM * 2) * 4;
            if bytes.len() == expected && &bytes[..8] == GK_WEIGHT_MAGIC {
                let mut w1 = Vec::with_capacity(GK_DIM * GK_DIM);
                let mut w2 = Vec::with_capacity(GK_DIM * GK_DIM);
                for chunk in bytes[8..].chunks_exact(4) {
                    let v = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as f64;
                    if w1.len() < GK_DIM * GK_DIM {
                        w1.push(v);
                    } else {
                        w2.push(v);
                    }
                }
                return Ok(GkBrain { w1, w2 });
            }
        }
        Ok(Self::new())
    }

    fn save(&self, path: &Path) -> Result<()> {
        ensure_state()?;
        let mut buf = Vec::with_capacity(8 + (GK_DIM * GK_DIM * 2) * 4);
        buf.extend_from_slice(GK_WEIGHT_MAGIC);
        for v in self.w1.iter().chain(self.w2.iter()) {
            buf.extend_from_slice(&(*v as f32).to_le_bytes());
        }
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        std::fs::write(path, &buf).with_context(|| format!("cannot write {}", path.display()))?;
        let _ = std::fs::write(Self::weights_path(), &buf);
        Ok(())
    }

    fn load_from(&mut self, path: &Path) -> Result<()> {
        let bytes =
            std::fs::read(path).with_context(|| format!("cannot read {}", path.display()))?;
        let expected = 8 + (GK_DIM * GK_DIM * 2) * 4;
        if bytes.len() != expected || &bytes[..8] != GK_WEIGHT_MAGIC {
            bail!("bad gatekeeper weights file {}", path.display());
        }
        self.w1.clear();
        self.w2.clear();
        for chunk in bytes[8..].chunks_exact(4) {
            let v = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as f64;
            if self.w1.len() < GK_DIM * GK_DIM {
                self.w1.push(v);
            } else {
                self.w2.push(v);
            }
        }
        Ok(())
    }
}

fn gk_instance() -> Result<&'static std::sync::Mutex<GkBrain>> {
    // One resident brain per process — same lifecycle as the resident VM.
    static BRAIN: std::sync::OnceLock<std::sync::Mutex<GkBrain>> = std::sync::OnceLock::new();
    Ok(BRAIN.get_or_init(|| {
        std::sync::Mutex::new(GkBrain::load_or_init().unwrap_or_else(|_| GkBrain::new()))
    }))
}

/// Persist the router weights (`w1`, `w2`) to `path`, mirrored into
/// `organs_dir()/gatekeeper_ane.bin` — same `CYGKW01` file format.
pub fn gatekeeper_save_weights(path: &Path) -> Result<()> {
    let brain = gk_instance()?;
    let brain = brain.lock().map_err(|e| anyhow::anyhow!("gk lock: {e}"))?;
    brain.save(path)
}

/// Load router weights from `path` into the resident brain.
pub fn gatekeeper_load_weights(path: &Path) -> Result<()> {
    let brain = gk_instance()?;
    let mut brain = brain.lock().map_err(|e| anyhow::anyhow!("gk lock: {e}"))?;
    brain.load_from(path)
}

// -------------------------------------------------------------- variant

// The gatekeeper router graph — `y = W2 @ sigmoid(W1 @ x)` — emitted as a
// `.mil` and compiled through `mil_infer::graph::compile_graph`, the same
// sovereign HWX path verified on hardware in
// `graph::tests::ane_onchip_backprop_training_exec`. Weights are runtime
// externals, not baked constants — training rewrites the SharedSurface.
const GK_FWD_MIL: &str = r#"program(1.3)
{
    func main<ios18>(tensor<fp16, [1, 1, 1, 64]> x,
                     tensor<fp16, [1, 64, 1, 64]> W1,
                     tensor<fp16, [1, 64, 1, 64]> W2) {
        tensor<bool, []> bt = const()[val = bool(true)];
        tensor<fp16, [1, 64, 1, 1]> z1 = matmul(x = x, y = W1, transpose_y = bt)[name = string("z1")];
        tensor<fp16, [1, 64, 1, 1]> h = sigmoid(x = z1)[name = string("h")];
        tensor<fp16, [1, 64, 1, 1]> y = matmul(x = h, y = W2, transpose_y = bt)[name = string("y")];
    } -> (y);
}
"#;

const GK_TRAIN_MIL: &str = r#"program(1.3)
{
    func main<ios18>(tensor<fp16, [1, 1, 1, 64]> x,
                     tensor<fp16, [1, 64, 1, 1]> target,
                     tensor<fp16, [1, 64, 1, 1]> neg_lr,
                     tensor<fp16, [1, 64, 1, 1]> one,
                     tensor<fp16, [1, 64, 1, 64]> W1,
                     tensor<fp16, [1, 64, 1, 64]> W2) {
        tensor<bool, []> bt = const()[val = bool(true)];
        tensor<fp16, [1, 64, 1, 1]> z1 = matmul(x = x, y = W1, transpose_y = bt)[name = string("z1")];
        tensor<fp16, [1, 64, 1, 1]> h = sigmoid(x = z1)[name = string("h")];
        tensor<fp16, [1, 64, 1, 1]> y = matmul(x = h, y = W2, transpose_y = bt)[name = string("y")];
        tensor<fp16, [1, 64, 1, 1]> dy = sub(x = y, y = target)[name = string("dy")];
        tensor<fp16, [1, 64, 1, 1]> ndy = mul(x = dy, y = neg_lr)[name = string("ndy")];
        tensor<fp16, [1, 1, 1, 64]> dh = matmul(x = dy, y = W2, transpose_x = bt)[name = string("dh")];
        tensor<fp16, [1, 64, 1, 1]> omh = sub(x = one, y = h)[name = string("omh")];
        tensor<fp16, [1, 64, 1, 1]> ds = mul(x = h, y = omh)[name = string("ds")];
        tensor<fp16, [1, 64, 1, 1]> dz1 = mul(x = dh, y = ds)[name = string("dz1")];
        tensor<fp16, [1, 64, 1, 1]> ndz1 = mul(x = dz1, y = neg_lr)[name = string("ndz1")];
        tensor<fp16, [1, 64, 1, 64]> dW2 = matmul(x = ndy, y = h, transpose_x = bt)[name = string("dW2")];
        tensor<fp16, [1, 64, 1, 64]> W2_new = add(x = W2, y = dW2)[name = string("W2_new")];
        tensor<fp16, [1, 64, 1, 64]> dW1 = matmul(x = ndz1, y = x, transpose_x = bt)[name = string("dW1")];
        tensor<fp16, [1, 64, 1, 64]> W1_new = add(x = W1, y = dW1)[name = string("W1_new")];
    } -> (y, W1_new, W2_new);
}
"#;

fn gk_mil_path(name: &str) -> PathBuf {
    organs_dir().join(name)
}

/// Compile + load a gatekeeper graph into a resident `GraphRuntime`.
fn gk_runtime(mil_src: &str, mil_name: &str) -> Result<GraphRuntime> {
    ensure_state()?;
    let mil_path = gk_mil_path(mil_name);
    std::fs::write(&mil_path, mil_src)?;
    let out_dir = organs_dir().join(format!("{mil_name}.hwx"));
    std::fs::create_dir_all(&out_dir)?;
    let plan = compile_graph(
        mil_path.to_str().context("mil path utf8")?,
        out_dir.to_str().context("out path utf8")?,
    )
    .map_err(|e| anyhow::anyhow!("graph compile: {e}"))?;
    GraphRuntime::load(plan).map_err(|e| anyhow::anyhow!("graph runtime: {e}"))
}

// `GraphRuntime` holds `Rc<SharedSurface>` — single-threaded by design —
// so runtimes live thread-local, built lazily on first use.
thread_local! {
    static GK_FWD_RT: std::cell::RefCell<Option<GraphRuntime>> =
        std::cell::RefCell::new(None);
    static GK_TRAIN_RT: std::cell::RefCell<Option<GraphRuntime>> =
        std::cell::RefCell::new(None);
}

fn gk_fwd<F, R>(f: F) -> Result<R>
where
    F: FnOnce(&mut GraphRuntime) -> R,
{
    GK_FWD_RT.with(|cell| {
        let mut slot = cell.borrow_mut();
        if slot.is_none() {
            *slot = gk_runtime(GK_FWD_MIL, "gatekeeper_fwd").ok();
        }
        slot.as_mut()
            .map(|rt| f(rt))
            .ok_or_else(|| anyhow::anyhow!("no ANE forward runtime"))
    })
}

fn gk_train<F, R>(f: F) -> Result<R>
where
    F: FnOnce(&mut GraphRuntime) -> R,
{
    GK_TRAIN_RT.with(|cell| {
        let mut slot = cell.borrow_mut();
        if slot.is_none() {
            *slot = gk_runtime(GK_TRAIN_MIL, "gatekeeper_train").ok();
        }
        slot.as_mut()
            .map(|rt| f(rt))
            .ok_or_else(|| anyhow::anyhow!("no ANE training runtime"))
    })
}

/// Fill an external input surface with `vals` (fp16 LE), indexing through
/// the surface's `desc` — `off(i)` maps a linear element index to the
/// (b, d, c, h, w) tuple `desc.offset` expects.
fn ext_f16(
    rt: &mut GraphRuntime,
    name: &'static str,
    vals: &[f64],
    off: fn(usize) -> (u64, u64, u64, u64, u64),
) {
    let vals = vals.to_vec();
    rt.write_ext(name, move |b, desc| {
        for (i, v) in vals.iter().enumerate() {
            let (a, d, c, h, w) = off(i);
            let o = desc.offset(a, d, c, h, w);
            if o + 1 < b.len() {
                b[o..o + 2].copy_from_slice(&f16::from_f64(*v).to_le_bytes());
            }
        }
    });
}

/// Read a named graph output back as `d` f64s — indexing through the
/// surface's `desc` (outputs are strided, not contiguous). `off(i)` maps a
/// linear element index to the (b, d, c, h, w) `desc.offset` tuple.
fn read_out(
    rt: &GraphRuntime,
    name: &str,
    d: usize,
    off: fn(usize) -> (u64, u64, u64, u64, u64),
) -> Option<Vec<f64>> {
    let ni = rt.out_node(name)?;
    let desc = rt.out_desc(ni);
    let out = rt.output(ni);
    let mut v = vec![0.0f64; d];
    for (i, slot) in v.iter_mut().enumerate() {
        let (a, dd, c, h, w) = off(i);
        let o = desc.offset(a, dd, c, h, w);
        if o + 1 < out.len() {
            *slot = f16::from_le_bytes([out[o], out[o + 1]]).to_f64();
        }
    }
    Some(v)
}

/// Classify `text` (`0.0` = fast, `1.0` = deep). Runs the forward graph on
/// the Neural Engine when the runtime is resident; any emission, load, or
/// eval failure falls back to the pure-Rust MLP with identical math.
pub fn gatekeeper_classify_text(text: &str) -> Result<f64> {
    let emb = embed_text_64(text);
    let x: Vec<f64> = emb.iter().map(|v| *v as f64).collect();
    let ane = (|| {
        let brain = gk_instance().ok()?;
        let brain = brain.lock().ok()?;
        let w1 = brain.w1.clone();
        let w2 = brain.w2.clone();
        drop(brain);
        let x = x.clone();
        gk_fwd(move |rt| {
            ext_f16(rt, "x", &x, |i| (0, 0, 0, 0, i as u64));
            ext_f16(rt, "W1", &w1, |i| {
                (0, 0, (i / 64) as u64, 0, (i % 64) as u64)
            });
            ext_f16(rt, "W2", &w2, |i| {
                (0, 0, (i / 64) as u64, 0, (i % 64) as u64)
            });
            rt.run()
                .ok()
                .and_then(|_| read_out(rt, "y", GK_DIM, |i| (0, 0, i as u64, 0, 0)))
        })
        .ok()
        .flatten()
    })();
    if let Some(y) = ane {
        let s0: f64 = y[0..32].iter().sum::<f64>() / 32.0;
        let s1: f64 = y[32..64].iter().sum::<f64>() / 32.0;
        return Ok((0.5 + 0.5 * (s1 - s0)).clamp(0.0, 1.0));
    }
    let brain = gk_instance()?;
    let brain = brain.lock().map_err(|e| anyhow::anyhow!("gk lock: {e}"))?;
    let (_, y) = brain.forward(&x);
    let s0: f64 = y[0..32].iter().sum::<f64>() / 32.0;
    let s1: f64 = y[32..64].iter().sum::<f64>() / 32.0;
    Ok((0.5 + 0.5 * (s1 - s0)).clamp(0.0, 1.0))
}

/// One training step on `text` with `label` (`0` = fast, `1` = deep) —
/// the full forward+backward+in-place-SGD graph executes on the ANE when
/// resident; updated `W1`/`W2` are read back into the brain and persisted.
/// Falls back to the pure-Rust step on any failure. Returns step MSE.
pub fn gatekeeper_train_text(text: &str, label: usize) -> Result<f64> {
    let emb = embed_text_64(text);
    let x: Vec<f64> = emb.iter().map(|v| *v as f64).collect();
    let mut t = vec![0.0f64; GK_DIM];
    for i in 0..32 {
        if label == 1 {
            t[i] = -1.0;
            t[i + 32] = 1.0;
        } else {
            t[i] = 1.0;
            t[i + 32] = -1.0;
        }
    }
    let ane = (|| {
        let brain = gk_instance().ok()?;
        let brain = brain.lock().ok()?;
        let w1 = brain.w1.clone();
        let w2 = brain.w2.clone();
        drop(brain);
        let x = x.clone();
        let t_ext = t.clone();
        gk_train(move |rt| {
            ext_f16(rt, "x", &x, |i| (0, 0, 0, 0, i as u64));
            ext_f16(rt, "target", &t_ext, |i| (0, 0, i as u64, 0, 0));
            ext_f16(rt, "neg_lr", &[-GK_LR; GK_DIM], |i| (0, 0, i as u64, 0, 0));
            ext_f16(rt, "one", &[1.0; GK_DIM], |i| (0, 0, i as u64, 0, 0));
            ext_f16(rt, "W1", &w1, |i| {
                (0, 0, (i / 64) as u64, 0, (i % 64) as u64)
            });
            ext_f16(rt, "W2", &w2, |i| {
                (0, 0, (i / 64) as u64, 0, (i % 64) as u64)
            });
            rt.run().ok()?;
            let y = read_out(rt, "y", GK_DIM, |i| (0, 0, i as u64, 0, 0))?;
            let w1n = read_out(rt, "W1_new", GK_DIM * GK_DIM, |i| {
                (0, 0, (i / 64) as u64, 0, (i % 64) as u64)
            });
            let w2n = read_out(rt, "W2_new", GK_DIM * GK_DIM, |i| {
                (0, 0, (i / 64) as u64, 0, (i % 64) as u64)
            });
            Some((y, w1n, w2n))
        })
        .ok()
        .flatten()
    })();
    if let Some((y, w1n, w2n)) = ane {
        let mse = y.iter().zip(&t).map(|(a, b)| (a - b).powi(2)).sum::<f64>() / GK_DIM as f64;
        let brain = gk_instance()?;
        let mut brain = brain.lock().map_err(|e| anyhow::anyhow!("gk lock: {e}"))?;
        if let Some(w) = w1n {
            brain.w1 = w;
        }
        if let Some(w) = w2n {
            brain.w2 = w;
        }
        let _ = brain.save(&GkBrain::weights_path());
        return Ok(mse);
    }
    let brain = gk_instance()?;
    let mut brain = brain.lock().map_err(|e| anyhow::anyhow!("gk lock: {e}"))?;
    Ok(brain.train_step(&x, &t))
}

/// Emit + compile the organ programs into `build_dir` — the installer's
/// "compile into a directory" contract, through `graph::compile_graph`
/// (sovereign HWX emission, no Coyote).
pub fn compile_to_dir(build_dir: &Path) -> Result<(usize, bool)> {
    std::fs::create_dir_all(build_dir)?;
    let mut n = 0;
    for (src, name) in [
        (GK_FWD_MIL, "gatekeeper_fwd"),
        (GK_TRAIN_MIL, "gatekeeper_train"),
    ] {
        let mil_path = build_dir.join(format!("{name}.mil"));
        std::fs::write(&mil_path, src)?;
        let out_dir = build_dir.join(format!("{name}.hwx"));
        std::fs::create_dir_all(&out_dir)?;
        compile_graph(
            mil_path.to_str().context("mil path utf8")?,
            out_dir.to_str().context("out path utf8")?,
        )
        .map_err(|e| anyhow::anyhow!("graph compile: {e}"))?;
        n += 1;
    }
    Ok((n, true))
}

/// True when the forward graph compiled and the `GraphRuntime` loaded —
/// the Neural Engine classify path is actually running.
pub fn is_ane_resident() -> bool {
    gk_fwd(|_| ()).is_ok()
}
