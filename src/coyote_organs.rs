//! Coyote organ layer — `coyote/badapple_organs.cy` loaded and invoked
//! in-process through `coyote::load_vm` / `Vm::call_fn`.
//!
//! This replaces the per-crate, per-binary JSON-sidecar organ pattern
//! (axon arc in `badapple-reflex`, akicita marshal, firewall scan,
//! tokala quorum, camazotz erasure) with one compiled organism: one VM,
//! one witness chain, one state root. The `.cy` source is the organ —
//! `cap fs, patch, erase` and `invariant reflexes_brake_never_steer` are
//! enforced by the Coyote compiler, not by convention.
//!
//! State root: `BADAPPLE_ORGANS_DIR` (default `/var/lib/bad_apple/organs`).
//! Organ source: `BADAPPLE_ORGANS_CY` (default `coyote/badapple_organs.cy`
//! shipped with this crate). Timestamps are injected here at the Rust
//! boundary — the same role `SenseEvent::new`'s `now_millis` plays — so
//! every organ decision stays deterministic and replayable.

use anyhow::{bail, Context, Result};
use coyote::vm::{Val, Vm};
use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

const ZERO64: &str = "0000000000000000000000000000000000000000000000000000000000000000";
const GK_WEIGHT_MAGIC: &[u8; 8] = b"CYGKW01\0";

struct ResidentOrganism {
    vm: Vm,
    src_path: PathBuf,
    witness_flushed: usize,
}

static ORGANISM: OnceLock<Mutex<Option<ResidentOrganism>>> = OnceLock::new();

/// Where the organism keeps its arc, ledger, and marshal state.
pub fn organs_dir() -> PathBuf {
    std::env::var("BADAPPLE_ORGANS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/var/lib/bad_apple/organs"))
}

fn organs_src() -> PathBuf {
    std::env::var("BADAPPLE_ORGANS_CY")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            for candidate in [
                PathBuf::from("/var/lib/bad_apple/organs/badapple_organs.cy"),
                PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("coyote/badapple_organs.cy"),
            ] {
                if candidate.exists() {
                    return candidate;
                }
            }
            PathBuf::from("/Users/savag3/bad_apple/coyote/badapple_organs.cy")
        })
}

fn default_weights_candidates(root: &Path) -> Vec<PathBuf> {
    let mut v = Vec::new();
    if let Ok(p) = std::env::var("BADAPPLE_GATEKEEPER_WEIGHTS") {
        v.push(PathBuf::from(p));
    }
    v.push(root.join("gatekeeper_ane.bin"));
    v.push(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("data/gatekeeper_ane.bin"));
    v.push(PathBuf::from("/var/lib/bad_apple/data/gatekeeper_ane.bin"));
    v.push(PathBuf::from("data/gatekeeper_ane.bin"));
    v
}

/// First-run materialization — the .cy organ treats file presence as a
/// given (axon::load_arc does the same with a default ReflexArc).
fn ensure_state(root: &Path) -> Result<()> {
    std::fs::create_dir_all(root)?;
    let tip_line = format!("seq=0\ntip={ZERO64}\n");
    for (name, init) in [
        ("arc.txt", "v1\nfrozen=0\nwstart=0\nwcount=0\n"),
        ("arc.log", tip_line.as_str()),
        (
            "marshal.txt",
            "level=suggest\nlast_apply=0\nmin_interval=0\ndedupe=\n",
        ),
    ] {
        let p = root.join(name);
        if !p.exists() {
            std::fs::write(&p, init).with_context(|| format!("cannot write {}", p.display()))?;
        }
    }
    Ok(())
}

fn vs(s: impl AsRef<str>) -> Val {
    Val::S(Rc::from(s.as_ref()))
}

fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

/// JSON sense-event data -> `{str: str}` with typed tags, matching
/// axon's `v.as_f64()` (`n:`) vs `v.to_string()` (`s:`/`j:`) semantics.
fn data_val(v: &serde_json::Value) -> Val {
    let mut fields = Vec::new();
    if let serde_json::Value::Object(m) = v {
        for (k, val) in m {
            let tagged = match val {
                serde_json::Value::Number(_) => format!("n:{val}"),
                serde_json::Value::String(s) => format!("s:{s}"),
                other => format!("j:{other}"),
            };
            fields.push((vs(k), vs(tagged)));
        }
    }
    Val::M(Rc::new(RefCell::new(fields)))
}

fn load_weights_into_vm(vm: &mut Vm, path: &Path) -> Result<()> {
    let bytes = std::fs::read(path)
        .with_context(|| format!("cannot read gatekeeper ANE weights {}", path.display()))?;
    let expected = 8 + (64 * 64 * 4) * 2;
    if bytes.len() != expected || &bytes[..8] != GK_WEIGHT_MAGIC {
        bail!(
            "{}: invalid Coyote gatekeeper weight header/size ({}B)",
            path.display(),
            bytes.len()
        );
    }
    let mut w1 = Vec::with_capacity(64 * 64);
    let mut w2 = Vec::with_capacity(64 * 64);
    let mut pos = 8usize;
    for _ in 0..64 * 64 {
        w1.push(f32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap()));
        pos += 4;
    }
    for _ in 0..64 * 64 {
        w2.push(f32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap()));
        pos += 4;
    }
    vm.set_param("gk_w1", w1).map_err(|e| anyhow::anyhow!(e))?;
    vm.set_param("gk_w2", w2).map_err(|e| anyhow::anyhow!(e))?;
    Ok(())
}

fn build_resident(src: &Path, root: &Path) -> Result<ResidentOrganism> {
    let build = std::env::temp_dir().join("badapple_organs_build");
    let force_cpu = std::env::var("BADAPPLE_ORGANS_CPU")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let mut vm = if force_cpu {
        coyote::load_vm(src, &build, true, false, true)
            .map_err(|e| anyhow::anyhow!("coyote load {}: {e}", src.display()))?
    } else {
        match coyote::load_vm(src, &build, false, false, true) {
            Ok(vm) => vm,
            Err(_) => coyote::load_vm(src, &build, true, false, true)
                .map_err(|e| anyhow::anyhow!("coyote load {}: {e}", src.display()))?,
        }
    };
    for cand in default_weights_candidates(root) {
        if cand.is_file() && load_weights_into_vm(&mut vm, &cand).is_ok() {
            break;
        }
    }
    // Prewarm the resident ANE `organ_encode` program so the first live
    // request hits already-loaded silicon surfaces.
    let x0 = vm.alloc_tensor(vec![64], vec![0.0; 64]);
    let _ = vm.call_fn("gatekeeper_classify", vec![x0]);
    let prewarm_wit = vm.witness_log().len();
    Ok(ResidentOrganism {
        vm,
        src_path: src.to_path_buf(),
        witness_flushed: prewarm_wit,
    })
}

fn with_organism<T, F>(f: F) -> Result<(T, PathBuf)>
where
    F: FnOnce(&mut Vm, &Path) -> Result<T>,
{
    let root = organs_dir();
    ensure_state(&root)?;
    let src = organs_src();
    let mutex = ORGANISM.get_or_init(|| Mutex::new(None));
    let mut guard = mutex
        .lock()
        .map_err(|e| anyhow::anyhow!("organism lock poisoned: {e}"))?;
    let need_init = guard.as_ref().is_none_or(|org| org.src_path != src);
    if need_init {
        *guard = Some(build_resident(&src, &root)?);
    }
    let org = guard.as_mut().expect("organism initialized");
    let out = f(&mut org.vm, &root)?;
    let log = org.vm.witness_log();
    if log.len() > org.witness_flushed {
        if let Ok(mut wf) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(root.join("witness.jsonl"))
        {
            use std::io::Write;
            for w in &log[org.witness_flushed..] {
                let _ = writeln!(
                    wf,
                    "{}",
                    serde_json::json!({
                        "seq": w.seq, "tag": w.tag, "val": w.val, "hash": w.hash
                    })
                );
            }
        }
        org.witness_flushed = log.len();
    }
    Ok((out, root))
}

/// True when the resident organ VM is executing its tensor programs on
/// native ANE silicon (`ProgBackend::Ane(GraphRuntime)`).
pub fn is_ane_resident() -> bool {
    with_organism(|vm, _| Ok(vm.is_ane()))
        .map(|(b, _)| b)
        .unwrap_or(false)
}

/// Compile `badapple_organs.cy` into `build_dir` on the ANE backend and
/// return `(ane_program_count, is_ane)`.
pub fn compile_to_dir(build_dir: &Path) -> Result<(usize, bool)> {
    let src = organs_src();
    let vm = coyote::load_vm(&src, build_dir, false, true, true)
        .map_err(|e| anyhow::anyhow!("coyote ANE compile {}: {e}", src.display()))?;
    Ok((vm.prog_names().len(), vm.is_ane()))
}

/// Invoke `name` on the resident Coyote organism VM — with the state
/// root prepended when `rooted` — and flush new witness entries next to
/// the organ state.
fn call(name: &str, args: Vec<Val>, rooted: bool) -> Result<String> {
    let t0 = Instant::now();
    let (s, _) = with_organism(|vm, root| {
        let mut argv = Vec::with_capacity(args.len() + 1);
        if rooted {
            argv.push(vs(root.to_string_lossy()));
        }
        argv.extend(args);
        let v = vm
            .call_fn(name, argv)
            .map_err(|e| anyhow::anyhow!("organs {name}: {e}"))?;
        match v {
            Val::S(s) => Ok(s.to_string()),
            Val::I(i) => Ok(i.to_string()),
            Val::B(b) => Ok(b.to_string()),
            Val::F(f) => Ok(f.to_string()),
            _ => bail!("organs {name}: non-scalar return"),
        }
    })?;
    if std::env::var_os("BADAPPLE_ORGANS_QUIET").is_none() {
        eprintln!("  (organs {name}: {:.0}µs)", t0.elapsed().as_micros());
    }
    Ok(s)
}

// ------------------------------------------------------------- the arc

pub fn arc_install(
    id: &str,
    trigger: &str,
    action: &str,
    cooldown_ms: i64,
    max_firings: i64,
) -> Result<String> {
    call(
        "arc_install",
        vec![
            vs(id),
            vs(trigger),
            vs(action),
            Val::I(cooldown_ms),
            Val::I(max_firings),
        ],
        true,
    )
}

pub fn arc_remove(id: &str) -> Result<String> {
    call("arc_remove", vec![vs(id)], true)
}

pub fn arc_list() -> Result<String> {
    call("arc_list", vec![], true)
}

pub fn arc_sense(kind: &str, data: &serde_json::Value) -> Result<String> {
    call(
        "arc_sense",
        vec![vs(kind), data_val(data), Val::I(now_ms())],
        true,
    )
}

pub fn arc_clear_freeze() -> Result<String> {
    call("arc_clear_freeze", vec![], true)
}

pub fn arc_rearm(id: &str) -> Result<String> {
    call("arc_rearm", vec![vs(id)], true)
}

pub fn arc_status() -> Result<String> {
    call("arc_status", vec![], true)
}

pub fn arc_verify() -> Result<String> {
    call("arc_verify", vec![], true)
}

// ---------------------------------------------------------- the marshal

pub fn marshal_check(surface: &str, old: &str, new: &str, intent: &str) -> Result<String> {
    call(
        "marshal_check",
        vec![vs(surface), vs(old), vs(new), vs(intent), Val::I(now_ms())],
        true,
    )
}

pub fn marshal_set_level(level: &str) -> Result<String> {
    call("marshal_set_level", vec![vs(level)], true)
}

pub fn marshal_status() -> Result<String> {
    call("marshal_status", vec![], true)
}

// ------------------------------------------------------------ firewall

pub fn firewall_scan(text: &str) -> Result<String> {
    let blocklist = std::env::var("BADAPPLE_BLOCKLIST")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/var/lib/bad_apple/blocklist.txt"));
    let bl = if blocklist.exists() {
        blocklist.to_string_lossy().to_string()
    } else {
        String::new()
    };
    call("firewall_scan", vec![vs(bl), vs(text)], false)
}

// ------------------------------------------------------------- council

pub fn council(topic: &str, votes: &[String]) -> Result<String> {
    let vs_vec: Vec<Val> = votes.iter().map(vs).collect();
    call(
        "council",
        vec![vs(topic), Val::V(Rc::new(RefCell::new(vs_vec)))],
        false,
    )
}

// ------------------------------------------------------------- erasure

pub fn erase_subject(subject: &str, payload: &str) -> Result<String> {
    call("erase_subject", vec![vs(subject), vs(payload)], false)
}

// --------------------------------------------------- gatekeeper ANE router

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// Deterministic 64-D L2-normalized signed feature-hashed embedding for
/// Coyote's `f16[64]` ANE gatekeeper programs (`organ_encode` / `organ_train`).
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
    // Structural anchors in slots 60..64: token count, char length, path/script markers, bias
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

/// Classify `text` (`0.0` = fast, `1.0` = deep) on the ANE through the
/// resident Coyote organ VM (`gatekeeper_classify` -> `organ_encode`).
pub fn gatekeeper_classify_text(text: &str) -> Result<f64> {
    let emb = embed_text_64(text);
    let (score, _) = with_organism(|vm, _| {
        let x = vm.alloc_tensor(vec![64], emb);
        let v = vm
            .call_fn("gatekeeper_classify", vec![x])
            .map_err(|e| anyhow::anyhow!("gatekeeper_classify: {e}"))?;
        match v {
            Val::F(f) => Ok(f),
            Val::I(i) => Ok(i as f64),
            _ => bail!("gatekeeper_classify returned non-float"),
        }
    })?;
    Ok(score)
}

/// Run one on-chip ANE training step (`gatekeeper_train_step` -> `organ_train`)
/// for `text` with `label` (`0` = fast, `1` = deep) and return the step MSE.
pub fn gatekeeper_train_text(text: &str, label: usize) -> Result<f64> {
    let emb = embed_text_64(text);
    let (loss, _) = with_organism(|vm, _| {
        let x = vm.alloc_tensor(vec![64], emb);
        let v = vm
            .call_fn("gatekeeper_train_step", vec![x, Val::I(label as i64)])
            .map_err(|e| anyhow::anyhow!("gatekeeper_train_step: {e}"))?;
        match v {
            Val::F(f) => Ok(f),
            Val::I(i) => Ok(i as f64),
            _ => bail!("gatekeeper_train_step returned non-float"),
        }
    })?;
    Ok(loss)
}

/// Persist the trained Coyote ANE gatekeeper weights (`gk_w1`, `gk_w2`) to `path`
/// (and mirror them into `organs_dir()/gatekeeper_ane.bin`).
pub fn gatekeeper_save_weights(path: &Path) -> Result<()> {
    let (bytes, root) = with_organism(|vm, _| {
        let w1 = vm
            .param_slice("gk_w1")
            .context("missing gk_w1 in Coyote VM")?
            .to_vec();
        let w2 = vm
            .param_slice("gk_w2")
            .context("missing gk_w2 in Coyote VM")?
            .to_vec();
        let mut buf = Vec::with_capacity(8 + (w1.len() + w2.len()) * 4);
        buf.extend_from_slice(GK_WEIGHT_MAGIC);
        for v in w1.iter().chain(w2.iter()) {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        Ok(buf)
    })?;
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    std::fs::write(path, &bytes).with_context(|| format!("cannot write {}", path.display()))?;
    let _ = std::fs::write(root.join("gatekeeper_ane.bin"), &bytes);
    Ok(())
}

/// Load trained Coyote ANE gatekeeper weights (`gk_w1`, `gk_w2`) from `path`
/// into the resident Coyote VM.
pub fn gatekeeper_load_weights(path: &Path) -> Result<()> {
    with_organism(|vm, _| load_weights_into_vm(vm, path))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ane_core::{AneCore, AneCoreConfig};
    use std::collections::HashMap;

    #[test]
    fn resident_organs_and_gatekeeper_run_on_ane() {
        let tmp = std::env::temp_dir().join("badapple_coyote_organs_test");
        let _ = std::fs::remove_dir_all(&tmp);
        std::env::set_var("BADAPPLE_ORGANS_DIR", &tmp);

        assert!(
            is_ane_resident(),
            "resident Coyote organ VM must run on ANE"
        );

        // Warmed resident call latency
        let t0 = Instant::now();
        let st = arc_status().unwrap();
        let warm_status_us = t0.elapsed().as_micros();
        assert!(st.starts_with("reflexes="));
        println!("resident arc_status latency: {warm_status_us} µs");

        // Gatekeeper ANE classification
        let t1 = Instant::now();
        let fast_score = gatekeeper_classify_text("what time is it").unwrap();
        let deep_score =
            gatekeeper_classify_text("explain the architecture of speculative decoding").unwrap();
        let gk_us = t1.elapsed().as_micros() / 2;
        println!(
            "gatekeeper ANE classify: fast={fast_score:.4}, deep={deep_score:.4} ({gk_us} µs/call)"
        );
        assert!(fast_score < 0.5, "expected fast < 0.5, got {fast_score}");
        assert!(deep_score > 0.5, "expected deep > 0.5, got {deep_score}");
    }

    #[test]
    fn ane_core_runs_coyote_qwenorg_on_ane() {
        let cy_path = PathBuf::from("/Users/savag3/coyote/examples/qwenorg.cy");
        let tok_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tokenizer.json");
        if !cy_path.is_file() || !tok_path.is_file() {
            return;
        }
        let cfg = AneCoreConfig {
            model_path: cy_path,
            tokenizer_path: tok_path,
            max_context_tokens: 64,
            bucket_paths: HashMap::new(),
        };
        let mut core = AneCore::load(&cfg).expect("AneCore::load(qwenorg.cy) must succeed");
        assert_eq!(core.ane_placement_ratio(), 1.0);
        assert_eq!(core.position(), Some(0));

        let out = core
            .generate("Hello", None, 4, 64)
            .expect("Coyote ANE generate must succeed");
        println!(
            "AneCore Coyote qwenorg output: {out:?}, pos={:?}",
            core.position()
        );
        assert!(core.position().unwrap_or(0) > 0);
        assert!(core.rewind(1));
        assert_eq!(core.position(), Some(1));
    }
}
