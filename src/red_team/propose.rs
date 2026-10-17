//! Generation cycle — the organism hypothesizes a weakness, expresses it as
//! a `ProbeSpec`, and the harness validates, persists, executes, and records
//! the result into a hash-chained findings log.
//!
//! The model never writes code: it emits declarative specs interpreted by
//! `generated.rs` inside a jail. A spec that cannot be validated never runs
//! and is never persisted — generation fails closed.

use super::generated::{self, Check, ProbeSpec};
use super::{Probe, ProbeResult, Severity};
use crate::bad_apple_ipc::call_agent;
use anyhow::{bail, Context, Result};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::path::PathBuf;

/// One entry in the findings log. `hash` commits to every field and to
/// `prev_hash`, so a rewritten or truncated log is detectable offline.
#[derive(Clone, Debug, Serialize)]
pub struct FindingRecord {
    pub timestamp_ms: u64,
    pub probe: String,
    pub category: String,
    pub mitigated: bool,
    pub severity: Severity,
    pub title: String,
    pub description: String,
    pub payload: Option<String>,
    pub prev_hash: String,
    pub hash: String,
}

impl FindingRecord {
    fn new(result: &ProbeResult, prev_hash: &str) -> Self {
        let ts = super::now_ms();
        let mut rec = FindingRecord {
            timestamp_ms: ts,
            probe: result.probe.clone(),
            category: result.category.clone(),
            mitigated: result.mitigated,
            severity: result.severity.clone(),
            title: result.title.clone(),
            description: result.description.clone(),
            payload: result.payload.clone(),
            prev_hash: prev_hash.to_string(),
            hash: String::new(),
        };
        let body = serde_json::to_string(&(
            &rec.timestamp_ms,
            &rec.probe,
            &rec.category,
            rec.mitigated,
            &rec.severity,
            &rec.title,
            &rec.description,
            &rec.payload,
            &rec.prev_hash,
        ))
        .unwrap_or_default();
        rec.hash = hex(Sha256::digest(body.as_bytes()));
        rec
    }
}

fn hex(bytes: impl AsRef<[u8]>) -> String {
    bytes.as_ref().iter().map(|b| format!("{b:02x}")).collect()
}

fn read_last_hash(path: &PathBuf) -> String {
    let Ok(text) = std::fs::read_to_string(path) else {
        return String::new();
    };
    text.lines()
        .rev()
        .find_map(|l| {
            serde_json::from_str::<serde_json::Value>(l)
                .ok()
                .and_then(|v| v.get("hash")?.as_str().map(String::from))
        })
        .unwrap_or_default()
}

/// Append a result to the hash-chained findings log.
pub fn record_result(result: &ProbeResult) -> Result<PathBuf> {
    let path = generated::findings_log_path();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let prev = read_last_hash(&path);
    let record = FindingRecord::new(result, &prev);
    let mut line = serde_json::to_string(&record)?;
    line.push('\n');
    use std::io::Write;
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)?;
    f.write_all(line.as_bytes())?;
    Ok(path)
}

/// Re-verify the findings log chain. Every line's `hash` must recompute over
/// its fields + `prev_hash`, and each `prev_hash` must equal the prior line's
/// `hash` — detects rewrites, truncation, reordering, and insertion.
/// Returns the number of verified records.
pub fn verify_findings_log(path: &PathBuf) -> Result<usize> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("cannot read findings log {}", path.display()))?;
    let mut prev = String::new();
    let mut count = 0usize;
    for (i, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let v: serde_json::Value =
            serde_json::from_str(line).with_context(|| format!("line {}: invalid JSON", i + 1))?;
        let claimed_prev = v.get("prev_hash").and_then(|x| x.as_str()).unwrap_or("");
        if claimed_prev != prev {
            bail!(
                "line {}: prev_hash mismatch — chain is broken or reordered",
                i + 1
            );
        }
        let claimed_hash = v
            .get("hash")
            .and_then(|x| x.as_str())
            .with_context(|| format!("line {}: missing hash", i + 1))?;
        let severity: Severity = serde_json::from_value(
            v.get("severity")
                .cloned()
                .unwrap_or(serde_json::Value::Null),
        )
        .context("line has bad severity")?;
        let rec = FindingRecord {
            timestamp_ms: v.get("timestamp_ms").and_then(|x| x.as_u64()).unwrap_or(0),
            probe: v
                .get("probe")
                .and_then(|x| x.as_str())
                .unwrap_or("")
                .to_string(),
            category: v
                .get("category")
                .and_then(|x| x.as_str())
                .unwrap_or("")
                .to_string(),
            mitigated: v
                .get("mitigated")
                .and_then(|x| x.as_bool())
                .unwrap_or(false),
            severity,
            title: v
                .get("title")
                .and_then(|x| x.as_str())
                .unwrap_or("")
                .to_string(),
            description: v
                .get("description")
                .and_then(|x| x.as_str())
                .unwrap_or("")
                .to_string(),
            payload: v
                .get("payload")
                .cloned()
                .and_then(|x| x.as_str().map(String::from)),
            prev_hash: claimed_prev.to_string(),
            hash: String::new(),
        };
        let body = serde_json::to_string(&(
            &rec.timestamp_ms,
            &rec.probe,
            &rec.category,
            rec.mitigated,
            &rec.severity,
            &rec.title,
            &rec.description,
            &rec.payload,
            &rec.prev_hash,
        ))
        .unwrap_or_default();
        let recomputed = hex(Sha256::digest(body.as_bytes()));
        if recomputed != claimed_hash {
            bail!("line {}: hash mismatch — record was rewritten", i + 1);
        }
        prev = recomputed;
        count += 1;
    }
    Ok(count)
}

/// Ids already claimed by the registry (built-in + generated).
fn registry_ids() -> Vec<String> {
    super::default_registry()
        .iter()
        .map(|p| p.id().to_string())
        .collect()
}

/// The attack-surface inventory offered to the model as raw material. Kept
/// deliberately compact — a local 7B complies with short concrete prompts far
/// better than a wall of text, so this lists targets, not prose.
fn surface_inventory(focus: Option<&str>) -> String {
    let mut s = String::from(
        "Live surface: dashboard http://127.0.0.1:8787 /api/* (status, ledger,\n\
         audit, doctor, organism, control, workspace, working_memory, memory/facts,\n\
         workshop/custom_tools/run, models, models/:id/use, agents, chat,\n\
         p2p/sync, p2p/pull, p2p/send, mcp/servers, autopilot, curious_trigger,\n\
         ocular, ocular/screen.png — bearer token /var/lib/bad_apple/dashboard.token).\n\
         Unix sockets: /var/run/badapple/{substrate,substrate_mlx,identity,aqua_helper}.sock,\n\
         /tmp/badapple_tts.sock. Files: /var/lib/bad_apple/{slicks.key,dashboard.token,\n\
         ledger.jsonl,ledger.sovereign.jsonl,policy.yaml,tape/}, ~/.bad_apple/*,\n\
         prompt.txt, personas.json. Processes: badapple-engine, gatekeeper,\n\
         supervisor, tape, ify, tts, identity-agent, dashboard.\n",
    );
    if let Some(f) = focus {
        s.push_str(&format!("Focus: {f}\n"));
    }
    s
}

fn spec_dsl_doc() -> &'static str {
    "Emit ONE json object, nothing else. No prose, no fences.\n\
     Schema: {\"id\":\"gen.<snake_name>\",\"title\":\"short\",\"hypothesis\":\"why it might break\",\n\
     \"severity\":\"low|medium|high\",\"checks\":[1-3 check objects]}\n\
     Check objects (every check is a JSON object, never a bare string):\n\
     {\"kind\":\"file_absent\",\"path\":\"/abs/path\"}\n\
     {\"kind\":\"file_present\",\"path\":\"/abs/path\"}\n\
     {\"kind\":\"file_contains\",\"path\":\"/abs/path\",\"pattern\":\"text\"}\n\
     {\"kind\":\"file_not_contains\",\"path\":\"/abs/path\",\"pattern\":\"text\"}\n\
     {\"kind\":\"file_mode_max\",\"path\":\"/abs/path\",\"mode\":\"0600\"}\n\
     {\"kind\":\"http_status\",\"url\":\"http://127.0.0.1:PORT/path\",\"status\":401}\n\
     {\"kind\":\"http_not_status\",\"url\":\"http://127.0.0.1:PORT/path\",\"status\":200}\n\
     {\"kind\":\"port_closed\",\"addr\":\"127.0.0.1:PORT\"}\n\
     {\"kind\":\"socket_rejects\",\"path\":\"/var/run/badapple/X.sock\",\"frame\":\"garbage\"}\n\
     {\"kind\":\"process_present\",\"name\":\"proc\"} or {\"kind\":\"process_absent\",\"name\":\"proc\"}\n\
     Example of a COMPLETE spec (invent your own, do not copy):\n\
     {\"id\":\"gen.ledger_perms\",\"title\":\"Ledger file readable by group\",\n\
     \"hypothesis\":\"If ledger.jsonl is group-readable, other admin accounts can read query history containing private prompts.\",\n\
     \"severity\":\"medium\",\"checks\":[{\"kind\":\"file_mode_max\",\"path\":\"/var/lib/bad_apple/ledger.jsonl\",\"mode\":\"0600\"}]}\n"
}

/// Extract the first balanced JSON object from model output.
pub fn extract_json_object(text: &str) -> Option<String> {
    let bytes = text.as_bytes();
    let mut start = None;
    let mut depth = 0i32;
    let mut in_str = false;
    let mut esc = false;
    for (i, &b) in bytes.iter().enumerate() {
        if in_str {
            if esc {
                esc = false;
            } else if b == b'\\' {
                esc = true;
            } else if b == b'"' {
                in_str = false;
            }
            continue;
        }
        match b {
            b'"' => in_str = true,
            b'{' => {
                if depth == 0 {
                    start = Some(i);
                }
                depth += 1;
            }
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return start.map(|s| text[s..=i].to_string());
                }
            }
            _ => {}
        }
    }
    None
}

/// Outcome of one generation cycle.
#[derive(Debug)]
pub struct ProposeOutcome {
    pub spec: ProbeSpec,
    pub result: ProbeResult,
    pub persisted_to: Option<PathBuf>,
    pub raw_model_output: String,
}

/// Ask the organism for one probe spec, validate it, optionally persist it,
/// execute it, and record the result. `focus` steers the model at a
/// subsystem; `dry_run` skips persistence (the probe still runs).
pub fn propose(focus: Option<&str>, dry_run: bool) -> Result<ProposeOutcome> {
    let prompt = format!(
        "You are Bad Apple's self red-team. Hypothesize ONE weakness on this surface.\n\
         {}Target something real and checkable. JSON spec only.\n{}",
        surface_inventory(focus),
        spec_dsl_doc()
    );

    // Raw inference bypasses the persona stack — a persona voice narrates
    // instead of emitting strict JSON, so the generator speaks to the
    // reasoning layer directly through the keyed agent channel.
    let params = serde_json::json!({
        "prompt": prompt,
        "system_prompt": "You are a security probe generator. Output exactly one JSON object. No prose, no markdown, no explanation.",
        "max_new_tokens": 1536,
        "temperature": 0.0,
    });
    let resp = call_agent("inference", Some(params), 1536)
        .context("inference call for probe spec failed")?;
    let raw = resp
        .get("text")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let json_text = extract_json_object(&raw).with_context(|| {
        format!(
            "model did not emit a JSON object — raw output: {}",
            &raw[..raw.len().min(600)]
        )
    })?;
    let spec: ProbeSpec = serde_json::from_str(&json_text)
        .with_context(|| format!("model output did not parse as a probe spec: {json_text}"))?;
    generated::validate_spec(&spec).context("generated spec failed validation")?;
    if spec.id == "gen.name_a-z0-9_"
        || spec.id == "gen.snake_name"
        || spec.title.contains("Why it might break")
    {
        bail!("model echoed the template instead of authoring a spec");
    }

    if registry_ids().iter().any(|id| id == &spec.id) {
        bail!("probe id '{}' is already registered", spec.id);
    }

    let probe = generated::load_probe(spec.clone());
    let result = probe.run();

    let persisted_to = if dry_run {
        None
    } else {
        Some(generated::persist_spec(&spec)?)
    };
    record_result(&result)?;

    Ok(ProposeOutcome {
        spec,
        result,
        persisted_to,
        raw_model_output: raw,
    })
}

/// The Check enum is re-exported here so prompt/doc references resolve.
#[allow(dead_code)]
fn _doc_only(_c: Check) {}

#[cfg(test)]
mod tests {
    use super::generated::GENERATED_CATEGORY;
    use super::*;

    #[test]
    fn extracts_json_from_noisy_output() {
        let text = "Here is my spec:\n```json\n{\"id\":\"gen.x\",\"checks\":[]}\n```\nThanks.";
        assert!(extract_json_object(text).unwrap().starts_with('{'));
        let nested = "{\"a\":{\"b\":\"}\"},\"c\":1}";
        assert_eq!(extract_json_object(nested).unwrap(), nested);
        assert!(extract_json_object("no json here").is_none());
    }

    #[test]
    fn verify_round_trips_written_records() {
        let dir = std::env::temp_dir()
            .canonicalize()
            .unwrap()
            .join(format!("ba_prop_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::env::set_var("BADAPPLE_RED_TEAM_DIR", &dir);
        let path = generated::findings_log_path();
        let r = || ProbeResult {
            probe: "gen.x".into(),
            category: GENERATED_CATEGORY.into(),
            mitigated: false,
            severity: Severity::Low,
            title: "t".into(),
            description: "d".into(),
            payload: None,
            timestamp_ms: 1,
            duration_us: 0,
        };
        record_result(&r()).unwrap();
        record_result(&r()).unwrap();
        let n = verify_findings_log(&path);
        std::env::remove_var("BADAPPLE_RED_TEAM_DIR");
        let _ = std::fs::remove_dir_all(&dir);
        assert_eq!(n.unwrap(), 2);
    }

    #[test]
    fn finding_record_chains() {
        let r1 = ProbeResult {
            probe: "gen.a".into(),
            category: GENERATED_CATEGORY.into(),
            mitigated: true,
            severity: Severity::Low,
            title: "t".into(),
            description: "d".into(),
            payload: None,
            timestamp_ms: 1,
            duration_us: 0,
        };
        let rec1 = FindingRecord::new(&r1, "");
        let rec2 = FindingRecord::new(&r1, &rec1.hash);
        assert_ne!(rec1.hash, rec2.hash);
        assert_eq!(rec2.prev_hash, rec1.hash);
    }
}
