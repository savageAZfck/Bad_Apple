// badapple-bulletin — signed oversight + risk bulletin over the audit ledger.
//
// Scans ledger.jsonl for entries appended since the last bulletin, classifies
// each dispatch by authority class (autonomous / human-approved / human-denied
// / policy-denied / council-escalated), summarizes council dissent and IFY
// findings in the same window, binds the artifact to the ledger tip it
// summarizes, signs the canonical body through the identity agent (Secure
// Enclave when available), and atomically writes
// /var/lib/bad_apple/bulletins/oversight-<stamp>.json.
//
// The artifact is the regulatory-grade "who authorized what" record: a human
// or auditor can verify it offline against the agent's public key without
// trusting the process that produced it.
//
// Usage:
//   badapple-bulletin                  Emit a bulletin for new entries
//   badapple-bulletin --all            Rebuild over the full ledger
//   badapple-bulletin --verify <path>  Verify a bulletin's signature

use anyhow::{bail, Context, Result};
use base64::{engine::general_purpose::STANDARD as B64, Engine as _};
use serde_json::{json, Map, Value};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixStream;
use std::path::PathBuf;
use std::time::Duration;

fn ledger_path() -> PathBuf {
    std::env::var("BADAPPLE_LEDGER")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/var/lib/bad_apple/ledger.jsonl"))
}

fn bulletin_dir() -> PathBuf {
    std::env::var("BADAPPLE_BULLETIN_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/var/lib/bad_apple/bulletins"))
}

fn cursor_path() -> PathBuf {
    bulletin_dir().join(".cursor")
}

fn identity_socket() -> PathBuf {
    std::env::var("BADAPPLE_IDENTITY_AGENT_SOCKET")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/var/run/badapple/identity.sock"))
}

fn ify_findings_path() -> PathBuf {
    if let Ok(dir) = std::env::var("BADAPPLE_IFY_DIR") {
        return PathBuf::from(dir).join("findings.jsonl");
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| "/root".into());
    PathBuf::from(home).join(".bad_apple/ify/findings.jsonl")
}

fn ceremony_path() -> PathBuf {
    std::env::var("BADAPPLE_CEREMONY_LOG")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/var/lib/bad_apple/key_ceremony.jsonl"))
}

// ---------- identity agent client (line-delimited JSON over a unix socket) ----------

fn agent_call(req: &Value) -> Result<Value> {
    let sock = identity_socket();
    let mut stream = UnixStream::connect(&sock)
        .with_context(|| format!("identity agent not reachable at {}", sock.display()))?;
    stream.set_read_timeout(Some(Duration::from_secs(10)))?;
    stream.set_write_timeout(Some(Duration::from_secs(10)))?;
    let mut frame = serde_json::to_vec(req)?;
    frame.push(b'\n');
    stream.write_all(&frame)?;
    let mut reader = BufReader::new(stream);
    let mut line = String::new();
    if reader.read_line(&mut line)? == 0 {
        bail!("identity agent closed the connection");
    }
    let resp: Value = serde_json::from_str(&line)?;
    if !resp.get("ok").and_then(|v| v.as_bool()).unwrap_or(false) {
        bail!(
            "identity agent error: {}",
            resp.get("error")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown")
        );
    }
    Ok(resp)
}

fn agent_available() -> bool {
    identity_socket().exists()
}

fn agent_sign(payload: &str) -> Result<(String, String)> {
    let sig =
        agent_call(&json!({"command": "sign", "message_b64": B64.encode(payload.as_bytes())}))?;
    let signature = sig
        .get("signature")
        .and_then(|v| v.as_str())
        .context("agent returned no signature")?
        .to_string();
    let pub_resp = agent_call(&json!({"command": "public_key"}))?;
    let public_key = pub_resp
        .get("public_key")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    Ok((signature, public_key))
}

fn agent_verify(payload: &str, signature: &str, public_key: &str) -> Result<bool> {
    let resp = agent_call(&json!({
        "command": "verify",
        "message_b64": B64.encode(payload.as_bytes()),
        "signature": signature,
        "public_key": public_key,
    }))?;
    Ok(resp.get("valid").and_then(|v| v.as_bool()).unwrap_or(false))
}

// ---------- key ceremony log ----------
//
// /var/lib/bad_apple/key_ceremony.jsonl — append-only record of every
// attestation-key event: observed (first sighting), rotated (key changed
// under us), revoked (operator withdrawal). Entries are sha256-chained and,
// when the agent is up, signed by the *current* enclave key — a rotation
// record is self-authenticating because the new key signs the record of its
// own succession.

use sha2::{Digest, Sha256};

fn key_id(pubkey: &str) -> String {
    hex::encode(Sha256::digest(pubkey.as_bytes()))[..16].to_string()
}

fn ceremony_tip() -> String {
    fs::read_to_string(ceremony_path())
        .unwrap_or_default()
        .lines()
        .filter_map(|l| serde_json::from_str::<Value>(l).ok())
        .filter_map(|e| e.get("hash").and_then(|h| h.as_str()).map(String::from))
        .last()
        .unwrap_or_else(|| "genesis".into())
}

fn record_ceremony(action: &str, key_id_val: &str, pubkey: &str, note: &str) -> Result<()> {
    let mut entry = Map::new();
    entry.insert(
        "ts".into(),
        json!(chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Micros, true)),
    );
    entry.insert("action".into(), json!(action));
    entry.insert("key_id".into(), json!(key_id_val));
    entry.insert("public_key".into(), json!(pubkey));
    entry.insert("note".into(), json!(note));
    entry.insert("prev_hash".into(), json!(ceremony_tip()));
    let canon = serde_json::to_string(&Value::Object(entry.clone()))?;
    entry.insert(
        "hash".into(),
        json!(hex::encode(Sha256::digest(canon.as_bytes()))),
    );

    // The new/current key signs the record of its own ceremony.
    if agent_available() {
        if let Ok((sig, _)) = agent_sign(&canon) {
            entry.insert("signature".into(), json!(sig));
        }
    }

    let path = ceremony_path();
    if let Some(dir) = path.parent() {
        fs::create_dir_all(dir)?;
    }
    let mut f = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)?;
    f.write_all(serde_json::to_string(&Value::Object(entry))?.as_bytes())?;
    f.write_all(b"\n")?;
    Ok(())
}

/// Called on every emit: if the agent's current public key differs from the
/// last ceremony record, a "rotated" event lands (signed by the NEW key, so
/// the succession is self-attesting). First run records "observed".
fn check_key_ceremony() -> Option<String> {
    if !agent_available() {
        return None;
    }
    let pubkey = agent_call(&json!({"command": "public_key"}))
        .ok()?
        .get("public_key")
        .and_then(|v| v.as_str())
        .map(String::from)?;
    let id = key_id(&pubkey);

    let last_key_id = fs::read_to_string(ceremony_path())
        .unwrap_or_default()
        .lines()
        .filter_map(|l| serde_json::from_str::<Value>(l).ok())
        .filter(|e| e.get("action").and_then(|a| a.as_str()) != Some("revoked"))
        .filter_map(|e| e.get("key_id").and_then(|k| k.as_str()).map(String::from))
        .last();

    match last_key_id {
        None => {
            let _ = record_ceremony("observed", &id, &pubkey, "first sighting of identity key");
        }
        Some(prev) if prev != id => {
            let _ = record_ceremony(
                "rotated",
                &id,
                &pubkey,
                &format!("key changed under observation (was {prev})"),
            );
        }
        _ => {}
    }
    Some(id)
}

fn revoke_key(target: &str) -> Result<()> {
    // The revocation must be signed by the CURRENT key — it is the operator
    // exercising the hardware identity to withdraw trust, and only that key
    // can prove the operator did it.
    let pubkey = agent_call(&json!({"command": "public_key"}))?
        .get("public_key")
        .and_then(|v| v.as_str())
        .map(String::from)
        .context("identity agent did not return a public key")?;
    record_ceremony("revoked", target, &pubkey, "operator revocation")?;
    println!("revocation recorded for key {target} (signed by current key)");
    Ok(())
}

fn print_key_history() -> Result<()> {
    let path = ceremony_path();
    let Ok(text) = fs::read_to_string(&path) else {
        println!("no ceremony log at {}", path.display());
        return Ok(());
    };
    for line in text.lines() {
        if let Ok(e) = serde_json::from_str::<Value>(line) {
            println!(
                "{}  {:9}  {}  {}{}",
                e.get("ts").and_then(|v| v.as_str()).unwrap_or("?"),
                e.get("action").and_then(|v| v.as_str()).unwrap_or("?"),
                e.get("key_id").and_then(|v| v.as_str()).unwrap_or("?"),
                e.get("note").and_then(|v| v.as_str()).unwrap_or(""),
                if e.get("signature").is_some() {
                    "  [signed]"
                } else {
                    ""
                }
            );
        }
    }
    Ok(())
}

// ---------- ledger scan + classification ----------

#[derive(Default)]
struct Window {
    entries: u64,
    first_ts: Option<String>,
    last_ts: Option<String>,
    // Authority classification of dispatch activity.
    tool_calls: u64,
    human_approved: u64,
    human_denied: u64,
    policy_denied: u64,
    approval_requested: u64,
    approval_unmatched: u64,
    council_escalated: u64,
    council_sessions: u64,
    kill_switch: u64,
    // Context activity counts.
    queries: u64,
    responses: u64,
    meta_commands: u64,
    cache_hits: u64,
    curious_checks: u64,
    curious_triggers: u64,
    dream_applied: u64,
    dream_rejected: u64,
    provenance_records: u64,
    tool_denied_reasons: BTreeMap<String, u64>,
    dissent_scores: Vec<f64>,
    policy_hashes: BTreeSet<String>,
    other_types: BTreeMap<String, u64>,
}

fn classify(entry: &Value, w: &mut Window) {
    let t = entry.get("type").and_then(|v| v.as_str()).unwrap_or("?");
    let data = entry.get("data").cloned().unwrap_or(Value::Null);
    let get = |k: &str| data.get(k).and_then(|v| v.as_str()).unwrap_or("");

    if let Some(ph) = data.get("policy_hash").and_then(|v| v.as_str()) {
        w.policy_hashes.insert(ph.to_string());
    }
    if let Some(d) = data.get("dissent").and_then(|v| v.as_f64()) {
        w.dissent_scores.push(d);
    }

    match t {
        "tool_call" => w.tool_calls += 1,
        "tool_denied" => {
            w.policy_denied += 1;
            *w.tool_denied_reasons
                .entry(get("reason").to_string())
                .or_default() += 1;
        }
        "tool_result" => {
            // Model-loop policy denials surface as tool_result with a
            // "Policy:" prefix rather than a tool_denied event.
            if get("result").starts_with("Policy:") {
                w.policy_denied += 1;
            }
        }
        "approval_requested" => w.approval_requested += 1,
        "approval_executed" | "approval_execute" => w.human_approved += 1,
        "approval_denied" => w.human_denied += 1,
        "approval_unmatched" => w.approval_unmatched += 1,
        "council_escalated" => w.council_escalated += 1,
        "council_deliberation" => w.council_sessions += 1,
        "kill_switch" => w.kill_switch += 1,
        "query" => w.queries += 1,
        "response" => w.responses += 1,
        "meta_command" => w.meta_commands += 1,
        "cache_hit" => w.cache_hits += 1,
        "curious_autopilot_check" => w.curious_checks += 1,
        "curious_trigger" => w.curious_triggers += 1,
        "dream_applied" | "dream_adopted" => w.dream_applied += 1,
        "dream_rejected" => w.dream_rejected += 1,
        "model_provenance" => w.provenance_records += 1,
        other => *w.other_types.entry(other.to_string()).or_default() += 1,
    }
}

/// Approval outcomes: requested ids joined against executed/denied
/// resolutions across the whole ledger. Unanswered requests split by the
/// 24h approval TTL — "pending" means the gate is still live, "expired"
/// means the request aged out unanswered (the gate held; the human never
/// granted). The two classes carry different audit weight.
const APPROVAL_TTL_SECS: f64 = 24.0 * 3600.0;

fn approval_outcomes(entries: &[Value]) -> (Vec<String>, Vec<String>) {
    let mut requested: HashMap<String, f64> = HashMap::new();
    let mut resolved: HashSet<String> = HashSet::new();
    for e in entries {
        let t = e.get("type").and_then(|v| v.as_str()).unwrap_or("");
        let id = e
            .get("data")
            .and_then(|d| d.get("id"))
            .and_then(|v| v.as_str())
            .unwrap_or("");
        if id.is_empty() {
            continue;
        }
        match t {
            "approval_requested" | "council_escalated" => {
                let ts = e
                    .get("ts")
                    .and_then(|v| v.as_str())
                    .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
                    .map(|d| d.timestamp() as f64)
                    .unwrap_or(0.0);
                requested.entry(id.to_string()).or_insert(ts);
            }
            "approval_executed" | "approval_denied" => {
                resolved.insert(id.to_string());
            }
            _ => {}
        }
    }
    let now = chrono::Utc::now().timestamp() as f64;
    let mut pending: Vec<String> = Vec::new();
    let mut expired: Vec<String> = Vec::new();
    for (id, ts) in requested {
        if resolved.contains(&id) {
            continue;
        }
        if now - ts < APPROVAL_TTL_SECS {
            pending.push(id);
        } else {
            expired.push(id);
        }
    }
    pending.sort();
    expired.sort();
    (pending, expired)
}

// ---------- bulletin emit / verify ----------

fn emit(all: bool) -> Result<()> {
    let ledger = ledger_path();
    if !ledger.is_file() {
        println!(
            "no ledger at {} yet; nothing to summarize",
            ledger.display()
        );
        return Ok(());
    }

    let file = fs::File::open(&ledger)?;
    let reader = BufReader::new(file);
    let mut entries: Vec<Value> = Vec::new();
    let mut tip_hash = String::new();
    let mut total_lines = 0u64;
    for line in reader.lines().map_while(Result::ok) {
        total_lines += 1;
        if let Ok(v) = serde_json::from_str::<Value>(&line) {
            if let Some(h) = v.get("hash").and_then(|h| h.as_str()) {
                tip_hash = h.to_string();
            }
            entries.push(v);
        }
    }

    let cursor: u64 = if all {
        0
    } else {
        fs::read_to_string(cursor_path())
            .ok()
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(0)
    };
    let mut cursor_reset = false;
    let start = if cursor as usize > entries.len() {
        // Ledger was truncated/reset under us — rebuild from the top and
        // say so rather than silently summarizing a different chain.
        cursor_reset = true;
        0
    } else {
        cursor as usize
    };

    let mut w = Window::default();
    for e in entries.iter().skip(start) {
        if w.entries == 0 {
            w.first_ts = e.get("ts").and_then(|v| v.as_str()).map(String::from);
        }
        w.entries += 1;
        w.last_ts = e.get("ts").and_then(|v| v.as_str()).map(String::from);
        classify(e, &mut w);
    }

    let (pending, expired) = approval_outcomes(&entries);

    // IFY findings inside the window (unix ts range).
    let mut ify_counts: HashMap<String, u64> = HashMap::new();
    if let (Some(from_ts), Some(to_ts)) = (&w.first_ts, &w.last_ts) {
        let from = chrono::DateTime::parse_from_rfc3339(from_ts)
            .map(|d| d.timestamp() as f64)
            .unwrap_or(0.0);
        let to = chrono::DateTime::parse_from_rfc3339(to_ts)
            .map(|d| d.timestamp() as f64)
            .unwrap_or(f64::MAX);
        if let Ok(f) = fs::File::open(ify_findings_path()) {
            for line in BufReader::new(f).lines().map_while(Result::ok) {
                if let Ok(v) = serde_json::from_str::<Value>(&line) {
                    let ts = v.get("ts").and_then(|t| t.as_f64()).unwrap_or(-1.0);
                    if ts >= from && ts <= to {
                        let sev = v
                            .get("severity")
                            .and_then(|s| s.as_str())
                            .unwrap_or("unknown");
                        *ify_counts.entry(sev.to_string()).or_default() += 1;
                    }
                }
            }
        }
    }

    let dissent_mean = if w.dissent_scores.is_empty() {
        Value::Null
    } else {
        json!(w.dissent_scores.iter().sum::<f64>() / w.dissent_scores.len() as f64)
    };
    let dissent_max = w.dissent_scores.iter().cloned().fold(f64::MIN, f64::max);
    let dissent_max = if w.dissent_scores.is_empty() {
        Value::Null
    } else {
        json!(dissent_max)
    };

    let now = chrono::Utc::now();
    let mut body = Map::new();
    body.insert("kind".into(), json!("badapple.oversight_bulletin"));
    body.insert("version".into(), json!(1));
    body.insert(
        "issued_at".into(),
        json!(now.to_rfc3339_opts(chrono::SecondsFormat::Secs, true)),
    );
    body.insert(
        "window".into(),
        json!({
            "from_ts": w.first_ts, "to_ts": w.last_ts,
            "entries": w.entries,
            "ledger_entries_total": entries.len() as u64,
            "cursor_reset": cursor_reset,
        }),
    );
    body.insert(
        "authority".into(),
        json!({
            "dispatch_attempts": w.tool_calls,
            "human_approved": w.human_approved,
            "human_denied": w.human_denied,
            "policy_denied": w.policy_denied,
            "approval_requested": w.approval_requested,
            "approval_unmatched": w.approval_unmatched,
            "council_escalated": w.council_escalated,
            "pending_approvals": pending.len() as u64,
            "pending_ids": pending,
            "expired_unanswered": expired.len() as u64,
            "expired_ids": expired,
            "denial_reasons": w.tool_denied_reasons,
        }),
    );
    body.insert(
        "dissent".into(),
        json!({
            "escalations": w.council_escalated,
            "mean": dissent_mean,
            "max": dissent_max,
        }),
    );
    body.insert(
        "activity".into(),
        json!({
            "queries": w.queries,
            "responses": w.responses,
            "meta_commands": w.meta_commands,
            "cache_hits": w.cache_hits,
            "curious_checks": w.curious_checks,
            "curious_triggers": w.curious_triggers,
            "council_sessions": w.council_sessions,
            "dream_applied": w.dream_applied,
            "dream_rejected": w.dream_rejected,
            "kill_switch_events": w.kill_switch,
            "provenance_records": w.provenance_records,
            "other_events": w.other_types,
        }),
    );
    body.insert(
        "ify".into(),
        json!({
            "total": ify_counts.values().sum::<u64>(),
            "by_severity": ify_counts,
        }),
    );
    body.insert(
        "policy".into(),
        json!({ "hashes_seen": w.policy_hashes.iter().collect::<Vec<_>>() }),
    );
    body.insert(
        "ledger".into(),
        json!({ "tip_hash": tip_hash, "entries": entries.len() as u64 }),
    );

    // Key-ceremony check: any rotation since the last bulletin is recorded
    // (and signed by the new key) before this bulletin is signed under it.
    let active_key_id = check_key_ceremony();
    body.insert(
        "identity".into(),
        json!({
            "key_id": active_key_id,
            "ceremony_tip": ceremony_tip(),
        }),
    );

    let payload = serde_json::to_string(&Value::Object(body.clone()))?;

    // Sign through the identity agent when it is running.
    let signed = agent_available();
    if signed {
        match agent_sign(&payload) {
            Ok((signature, public_key)) => {
                body.insert("signed_at".into(), body["issued_at"].clone());
                body.insert("scheme".into(), json!("secure-enclave"));
                body.insert("public_key".into(), json!(public_key));
                body.insert("signature".into(), json!(signature));
                body.insert("payload".into(), json!(payload));
            }
            Err(e) => {
                eprintln!("warning: bulletin signing failed: {e}");
            }
        }
    } else {
        eprintln!("warning: identity agent unavailable — bulletin unsigned");
    }
    body.insert("signed".into(), json!(body.get("signature").is_some()));

    let dir = bulletin_dir();
    fs::create_dir_all(&dir)?;
    let name = format!("oversight-{}.json", now.format("%Y-%m-%dT%H-%M-%SZ"));
    let path = dir.join(&name);
    let tmp = dir.join(format!(".{name}.tmp"));
    {
        let mut f = fs::File::create(&tmp)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            f.set_permissions(fs::Permissions::from_mode(0o600))?;
        }
        f.write_all(serde_json::to_string_pretty(&Value::Object(body))?.as_bytes())?;
        f.write_all(b"\n")?;
        f.sync_all()?;
    }
    fs::rename(&tmp, &path)?;
    fs::write(cursor_path(), format!("{}\n", entries.len()))?;

    println!(
        "bulletin written to {} ({} entries, {total} ledger lines{})",
        path.display(),
        w.entries,
        if signed { ", signed" } else { ", unsigned" },
        total = total_lines
    );
    Ok(())
}

fn verify(path: &str) -> Result<()> {
    let text = fs::read_to_string(path)?;
    let doc: Value = serde_json::from_str(&text)?;
    let payload = doc
        .get("payload")
        .and_then(|v| v.as_str())
        .context("bulletin has no signed payload")?;
    let signature = doc
        .get("signature")
        .and_then(|v| v.as_str())
        .context("bulletin has no signature")?;
    let public_key = doc
        .get("public_key")
        .and_then(|v| v.as_str())
        .context("bulletin has no public key")?;

    // The stored payload must byte-match the fields it claims to cover —
    // otherwise a signature over anything could be replayed onto this file.
    let mut stripped = doc.clone();
    if let Some(obj) = stripped.as_object_mut() {
        for k in [
            "signed_at",
            "scheme",
            "public_key",
            "signature",
            "payload",
            "signed",
        ] {
            obj.remove(k);
        }
    }
    let rebuilt = serde_json::to_string(&stripped)?;
    if rebuilt != payload {
        bail!("payload does not match bulletin fields — possible tampering");
    }
    if !agent_available() {
        bail!("identity agent not running — cannot verify signature");
    }
    let ok = agent_verify(payload, signature, public_key)?;
    if ok {
        println!("bulletin signature VALID (secure-enclave)");
        Ok(())
    } else {
        bail!("bulletin signature INVALID");
    }
}

fn usage() -> ! {
    eprintln!("badapple-bulletin — signed oversight + risk bulletin");
    eprintln!("Usage:");
    eprintln!("  badapple-bulletin           Emit a bulletin for entries since the last run");
    eprintln!("  badapple-bulletin --all     Rebuild over the full ledger");
    eprintln!("  badapple-bulletin --verify <path>  Verify a bulletin's signature");
    eprintln!("  badapple-bulletin --keys    Show attestation-key ceremony history");
    eprintln!("  badapple-bulletin --revoke-key <key_id>  Record a signed key revocation");
    std::process::exit(1);
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.as_slice() {
        [] => emit(false),
        [a] if a == "--all" => emit(true),
        [a, p] if a == "--verify" => verify(p),
        [a] if a == "--keys" => print_key_history(),
        [a, k] if a == "--revoke-key" => revoke_key(k),
        _ => usage(),
    }
}
