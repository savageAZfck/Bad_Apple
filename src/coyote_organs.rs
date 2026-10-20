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
use coyote::vm::Val;
use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::time::Instant;

const ZERO64: &str =
    "0000000000000000000000000000000000000000000000000000000000000000";

/// Where the organism keeps its arc, ledger, and marshal state.
pub fn organs_dir() -> PathBuf {
    std::env::var("BADAPPLE_ORGANS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/var/lib/bad_apple/organs"))
}

fn organs_src() -> PathBuf {
    std::env::var("BADAPPLE_ORGANS_CY").map(PathBuf::from).unwrap_or_else(|_| {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("coyote/badapple_organs.cy")
    })
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
            std::fs::write(&p, init)
                .with_context(|| format!("cannot write {}", p.display()))?;
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

/// Compile the organ program (cached), invoke `name` — with the state
/// root prepended when `rooted` — and flush the call's witness entries
/// next to the organ state. The VM's chain is hash-linked within the
/// call; the durable cross-call chain lives in `arc.log` itself.
fn call(name: &str, args: Vec<Val>, rooted: bool) -> Result<String> {
    let root = organs_dir();
    ensure_state(&root)?;
    let src = organs_src();
    let build = std::env::temp_dir().join("badapple_organs_build");
    let t0 = Instant::now();
    let mut vm = coyote::load_vm(&src, &build, true, false, true)
        .map_err(|e| anyhow::anyhow!("coyote load {}: {e}", src.display()))?;
    let mut argv = Vec::with_capacity(args.len() + 1);
    if rooted {
        argv.push(vs(root.to_string_lossy()));
    }
    argv.extend(args);
    let v = vm
        .call_fn(name, argv)
        .map_err(|e| anyhow::anyhow!("organs {name}: {e}"))?;
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(root.join("witness.jsonl"))
    {
        use std::io::Write;
        for w in vm.witness_log() {
            let _ = writeln!(
                f,
                "{}",
                serde_json::json!({
                    "seq": w.seq, "tag": w.tag, "val": w.val, "hash": w.hash
                })
            );
        }
    }
    let s = match v {
        Val::S(s) => s.to_string(),
        Val::I(i) => i.to_string(),
        Val::B(b) => b.to_string(),
        _ => bail!("organs {name}: non-scalar return"),
    };
    eprintln!("  (organs {name}: {:.0}µs)", t0.elapsed().as_micros());
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
    call("arc_sense", vec![vs(kind), data_val(data), Val::I(now_ms())], true)
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

pub fn marshal_check(
    surface: &str,
    old: &str,
    new: &str,
    intent: &str,
) -> Result<String> {
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
