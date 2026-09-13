//! IFY — Bad Apple's behavioral watchdog.
//!
//! Tails `/var/lib/bad_apple/ledger.jsonl`, verifies each new line against
//! the hash chain as it arrives, learns deterministic baselines (event-type
//! rates per hour-of-day, tool frequencies, approval outcomes), and surfaces
//! findings on a severity ladder:
//!
//! - `info`      → findings log only
//! - `elevated`  → proposal + notification (secondary phase and up)
//! - `critical`  → proposal + notification + kill-switch brake (autopilot)
//!
//! The detector is statistics, not a model: every finding carries the
//! observed number, the baseline number, and the rule that fired.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs::{self, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

pub const GENESIS_LABEL: &str = "bad-apple-genesis-v1";
pub const DEFAULT_GESTATION_DAYS: f64 = 14.0;
pub const DEFAULT_SECONDARY_DAYS: f64 = 14.0;
/// Warmup floor: don't fire baseline-dependent rules until this many
/// events have been observed, regardless of phase.
pub const BASELINE_WARMUP_EVENTS: u64 = 200;
/// Spike multiplier for per-type hourly rates.
pub const RATE_SPIKE_FACTOR: f64 = 4.0;
pub const RATE_SPIKE_FLOOR: u64 = 10;
pub const DENIAL_SPIKE_COUNT: u64 = 3;
pub const FIREWALL_SPIKE_FLOOR: u64 = 3;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, Debug)]
#[serde(rename_all = "lowercase")]
pub enum Severity {
    Info,
    Elevated,
    Critical,
}

#[derive(Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Debug)]
#[serde(rename_all = "lowercase")]
pub enum Phase {
    Gestation,
    Secondary,
    Autopilot,
}

impl Phase {
    pub fn as_str(&self) -> &'static str {
        match self {
            Phase::Gestation => "gestation",
            Phase::Secondary => "secondary",
            Phase::Autopilot => "autopilot",
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Finding {
    pub id: String,
    pub ts: f64,
    pub rule: String,
    pub severity: Severity,
    /// The observed value that tripped the rule.
    pub observed: String,
    /// The baseline the observation was compared against.
    pub baseline: String,
    /// Human-readable summary (deterministic text, no model output).
    pub detail: String,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct EventStat {
    pub count: u64,
    /// Total observations per hour-of-day (UTC), 24 buckets.
    pub hourly: [u64; 24],
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IfyState {
    pub version: u32,
    pub installed_at: f64,
    pub events_seen: u64,
    /// Byte offset into ledger.jsonl already consumed.
    pub ledger_offset: u64,
    /// Stored hash of the last verified line (chain tip as IFY knows it).
    pub ledger_tip: String,
    pub event_types: HashMap<String, EventStat>,
    pub tools: HashMap<String, u64>,
    pub personas: HashMap<String, u64>,
    pub approvals_granted: u64,
    pub approvals_denied: u64,
    pub firewall_hits: u64,
    pub kill_switch_events: u64,
    /// Aggregate hourly activity across all event types (for off_hours).
    pub hourly_total: [u64; 24],
    /// Current UTC hour and per-type counts inside it (rate window).
    #[serde(default)]
    pub window_hour: i64,
    #[serde(default)]
    pub window_counts: HashMap<String, u64>,
    /// Sliding window (last 3600s) of denial and firewall timestamps.
    #[serde(default)]
    pub denial_times: Vec<f64>,
    #[serde(default)]
    pub firewall_times: Vec<f64>,
    /// "rule:event_type" → window hour last fired, so a rule fires at
    /// most once per type per hour even as the baseline moves.
    #[serde(default)]
    pub fired: HashMap<String, i64>,
    #[serde(default)]
    pub last_brake_ts: f64,
}

impl Default for IfyState {
    fn default() -> Self {
        Self {
            version: 1,
            installed_at: now_secs(),
            events_seen: 0,
            ledger_offset: 0,
            ledger_tip: genesis_hash(),
            event_types: HashMap::new(),
            tools: HashMap::new(),
            personas: HashMap::new(),
            approvals_granted: 0,
            approvals_denied: 0,
            firewall_hits: 0,
            kill_switch_events: 0,
            hourly_total: [0; 24],
            window_hour: -1,
            window_counts: HashMap::new(),
            denial_times: Vec::new(),
            firewall_times: Vec::new(),
            fired: HashMap::new(),
            last_brake_ts: 0.0,
        }
    }
}

pub fn now_secs() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64()
}

fn genesis_hash() -> String {
    hex::encode(Sha256::digest(GENESIS_LABEL.as_bytes()))
}

// MARK: - Paths

pub fn ify_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("BADAPPLE_IFY_DIR") {
        return PathBuf::from(dir);
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
    PathBuf::from(home).join(".bad_apple").join("ify")
}

pub fn state_path() -> PathBuf {
    ify_dir().join("state.json")
}

pub fn findings_path() -> PathBuf {
    ify_dir().join("findings.jsonl")
}

pub fn proposals_dir() -> PathBuf {
    ify_dir().join("proposals")
}

pub fn ledger_path() -> PathBuf {
    std::env::var("BADAPPLE_LEDGER")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/var/lib/bad_apple/ledger.jsonl"))
}

pub fn runtime_state_path() -> PathBuf {
    PathBuf::from("/var/lib/bad_apple/runtime_state.json")
}

// MARK: - Phase machine

pub fn gestation_days() -> f64 {
    std::env::var("BADAPPLE_IFY_GESTATION_DAYS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_GESTATION_DAYS)
}

pub fn secondary_days() -> f64 {
    std::env::var("BADAPPLE_IFY_SECONDARY_DAYS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_SECONDARY_DAYS)
}

/// Resolve the active phase. `BADAPPLE_IFY_PHASE` forces one (testing).
pub fn current_phase(state: &IfyState) -> Phase {
    if let Ok(forced) = std::env::var("BADAPPLE_IFY_PHASE") {
        match forced.trim().to_lowercase().as_str() {
            "gestation" => return Phase::Gestation,
            "secondary" | "secondary_school" => return Phase::Secondary,
            "autopilot" => return Phase::Autopilot,
            _ => {}
        }
    }
    let age_days = (now_secs() - state.installed_at) / 86400.0;
    if age_days < gestation_days() {
        Phase::Gestation
    } else if age_days < gestation_days() + secondary_days() {
        Phase::Secondary
    } else {
        Phase::Autopilot
    }
}

// MARK: - State persistence

pub fn load_state() -> IfyState {
    fs::read_to_string(state_path())
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default()
}

pub fn save_state(state: &IfyState) {
    let dir = ify_dir();
    let _ = fs::create_dir_all(&dir);
    let path = state_path();
    let tmp = dir.join(format!(".state.{}.tmp", std::process::id()));
    if let Ok(json) = serde_json::to_string_pretty(state) {
        if let Ok(mut f) = fs::File::create(&tmp) {
            let _ = f.write_all(json.as_bytes());
            let _ = f.flush();
            let _ = fs::rename(&tmp, &path);
        }
    }
}

pub fn log_finding(f: &Finding) {
    let dir = ify_dir();
    let _ = fs::create_dir_all(&dir);
    if let Ok(mut file) = OpenOptions::new()
        .create(true)
        .append(true)
        .open(findings_path())
    {
        if let Ok(line) = serde_json::to_string(f) {
            let _ = writeln!(file, "{line}");
        }
    }
}

fn finding_id(rule: &str) -> String {
    let ts = now_secs();
    let mut h = Sha256::new();
    h.update(rule.as_bytes());
    h.update(ts.to_bits().to_le_bytes());
    h.update((std::process::id() as u64).to_le_bytes());
    let short = &hex::encode(h.finalize())[..8];
    let dt = chrono::Utc::now().format("%Y%m%d-%H%M%S");
    format!("ify-{dt}-{short}")
}

// MARK: - Ledger tailing

/// Slicks key candidates for per-line verification, mirroring
/// badapple-sovereign's loader.
pub fn load_slicks_secrets() -> Vec<Vec<u8>> {
    let key_path = std::env::var("BADAPPLE_SLICKS_KEY_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/var/lib/bad_apple/slicks.key"));
    fs::read(&key_path)
        .map(|raw| sovereign_ledger::import::slicks_key_candidates(&raw))
        .unwrap_or_default()
}

/// A single verified ledger event, decoded for baseline updates.
pub struct LedgerEvent {
    pub event_type: String,
    pub data: Value,
    pub hour: usize,
    pub hash: String,
}

/// Parse one verified raw ledger line into a LedgerEvent.
fn parse_event(line: &str) -> Option<LedgerEvent> {
    let v: Value = serde_json::from_str(line).ok()?;
    let event_type = v
        .get("type")
        .or_else(|| v.get("event_type"))
        .and_then(|t| t.as_str())
        .unwrap_or("audit")
        .to_string();
    let data = v.get("data").cloned().unwrap_or(Value::Null);
    let hour = v
        .get("ts")
        .or_else(|| v.get("timestamp"))
        .and_then(|t| t.as_str())
        .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
        .map(|dt| dt.hour() as usize)
        .unwrap_or(0);
    let hash = v
        .get("hash")
        .and_then(|h| h.as_str())
        .unwrap_or("")
        .to_string();
    Some(LedgerEvent {
        event_type,
        data,
        hour,
        hash,
    })
}

use chrono::Timelike;

pub struct TailReport {
    pub new_events: u64,
    pub findings: Vec<Finding>,
    /// Set when the chain itself broke — always critical.
    pub chain_broken: bool,
    /// Set when the ledger shrank below our offset — always critical.
    pub truncated: bool,
}

/// Read new ledger bytes from `state.ledger_offset`, verify each line's
/// linkage and (when secrets are available) content hash, update baselines,
/// and run the detection rules. Offset/tip advance only across verified
/// lines; a failure stops the tail at the offending line.
pub fn tail_ledger(state: &mut IfyState, secrets: &[Vec<u8>]) -> std::io::Result<TailReport> {
    let path = ledger_path();
    let mut report = TailReport {
        new_events: 0,
        findings: Vec::new(),
        chain_broken: false,
        truncated: false,
    };
    if !path.exists() {
        return Ok(report);
    }
    let len = fs::metadata(&path)?.len();
    if len < state.ledger_offset {
        report.truncated = true;
        report.findings.push(Finding {
            id: finding_id("ledger_gap"),
            ts: now_secs(),
            rule: "ledger_gap".into(),
            severity: Severity::Critical,
            observed: format!("size {len} < offset {}", state.ledger_offset),
            baseline: "append-only".into(),
            detail: "ledger.jsonl shrank — entries were removed or the file was replaced".into(),
        });
        // Re-anchor at the start: the old tip is meaningless now.
        state.ledger_offset = 0;
        state.ledger_tip = genesis_hash();
    }
    let mut file = fs::File::open(&path)?;
    file.seek(SeekFrom::Start(state.ledger_offset))?;
    let mut buf = String::new();
    file.read_to_string(&mut buf)?;
    // Only process complete lines; a partial write keeps for next pass.
    let mut offset = state.ledger_offset;
    for line in buf.lines() {
        let line_len = line.len() as u64 + 1; // + '\n'
        if line.trim().is_empty() {
            offset += line_len;
            continue;
        }
        match sovereign_ledger::import::verify_badapple_line(line, secrets, &state.ledger_tip) {
            Ok(new_tip) => {
                if let Some(ev) = parse_event(line) {
                    observe_event(state, &ev, &mut report.findings);
                }
                state.ledger_tip = new_tip;
                state.events_seen += 1;
                report.new_events += 1;
                offset += line_len;
            }
            Err(e) => {
                report.chain_broken = true;
                report.findings.push(Finding {
                    id: finding_id("chain_break"),
                    ts: now_secs(),
                    rule: "chain_break".into(),
                    severity: Severity::Critical,
                    observed: format!("verify failed: {e}"),
                    baseline: format!("prev {}", &state.ledger_tip[..16.min(state.ledger_tip.len())]),
                    detail: "a ledger line failed hash-chain verification — the audit trail was modified".into(),
                });
                break;
            }
        }
    }
    state.ledger_offset = offset;
    Ok(report)
}

// MARK: - Baselines + detection

fn observe_event(state: &mut IfyState, ev: &LedgerEvent, findings: &mut Vec<Finding>) {
    let warmed = state.events_seen >= BASELINE_WARMUP_EVENTS;
    let hour = ev.hour.min(23);

    // Rule: novel_event_type — first observation of a type after warmup.
    if warmed && !state.event_types.contains_key(&ev.event_type) {
        findings.push(Finding {
            id: finding_id("novel_event_type"),
            ts: now_secs(),
            rule: "novel_event_type".into(),
            severity: Severity::Elevated,
            observed: ev.event_type.clone(),
            baseline: format!("{} known types", state.event_types.len()),
            detail: format!(
                "event type `{}` has never appeared in the ledger",
                ev.event_type
            ),
        });
    }

    // Rate window: reset when the hour rolls over.
    let now_hour = (now_secs() / 3600.0) as i64;
    if state.window_hour != now_hour {
        state.window_hour = now_hour;
        state.window_counts.clear();
    }
    let win = state
        .window_counts
        .entry(ev.event_type.clone())
        .or_insert(0);
    *win += 1;

    // Rule: rate_spike — this-hour count far above the type's hourly norm.
    let stat = state.event_types.entry(ev.event_type.clone()).or_default();
    let hourly_avg =
        stat.count as f64 / stat.hourly.iter().filter(|&&c| c > 0).count().max(1) as f64;
    let this_hour = state
        .window_counts
        .get(&ev.event_type)
        .copied()
        .unwrap_or(0);
    let fire_key = format!("rate_spike:{}", ev.event_type);
    let already_fired = state.fired.get(&fire_key).copied() == Some(now_hour);
    if warmed
        && !already_fired
        && hourly_avg > 0.0
        && (this_hour as f64) > (hourly_avg * RATE_SPIKE_FACTOR).max(RATE_SPIKE_FLOOR as f64)
    {
        state.fired.insert(fire_key, now_hour);
        findings.push(Finding {
            id: finding_id("rate_spike"),
            ts: now_secs(),
            rule: "rate_spike".into(),
            severity: Severity::Elevated,
            observed: format!("{} `{}` events this hour", this_hour, ev.event_type),
            baseline: format!("~{hourly_avg:.1}/hr typical"),
            detail: format!(
                "`{}` is running {:.1}× above its learned rate",
                ev.event_type,
                this_hour as f64 / hourly_avg
            ),
        });
    }

    // Rule: off_hours — activity in an hour-of-day with zero history.
    if warmed && state.hourly_total[hour] == 0 {
        findings.push(Finding {
            id: finding_id("off_hours"),
            ts: now_secs(),
            rule: "off_hours".into(),
            severity: Severity::Info,
            observed: format!("`{}` at hour {:02} UTC", ev.event_type, hour),
            baseline: "no prior activity in this hour".into(),
            detail: format!(
                "activity during an hour-of-day that has been silent for the whole baseline ({})",
                ev.event_type
            ),
        });
    }

    stat.count += 1;
    stat.hourly[hour] += 1;
    state.hourly_total[hour] += 1;

    // Baseline counters + targeted rules on well-known shapes.
    let ty = ev.event_type.as_str();
    if ty.contains("kill") || ty.contains("safe_mode") {
        state.kill_switch_events += 1;
        if warmed {
            findings.push(Finding {
                id: finding_id("kill_switch"),
                ts: now_secs(),
                rule: "kill_switch".into(),
                severity: Severity::Elevated,
                observed: ev.event_type.clone(),
                baseline: format!("{} prior", state.kill_switch_events - 1),
                detail:
                    "kill switch / safe mode event — if you didn't trigger it, something else did"
                        .into(),
            });
        }
    }
    if ty.contains("firewall") {
        state.firewall_hits += 1;
        state.firewall_times.push(now_secs());
    }
    if ty.contains("approv") || ty.contains("tool") {
        let approved = ev
            .data
            .get("approved")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);
        if approved {
            state.approvals_granted += 1;
        } else {
            state.approvals_denied += 1;
            state.denial_times.push(now_secs());
        }
        if let Some(tool) = ev
            .data
            .get("tool")
            .or_else(|| ev.data.get("name"))
            .and_then(|v| v.as_str())
        {
            *state.tools.entry(tool.to_string()).or_insert(0) += 1;
        }
    }
    if let Some(persona) = ev
        .data
        .get("persona")
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
    {
        *state.personas.entry(persona.to_string()).or_insert(0) += 1;
    }

    // Rule: denial_spike — ≥3 denied approvals in the trailing hour.
    let cutoff = now_secs() - 3600.0;
    state.denial_times.retain(|&t| t > cutoff);
    state.firewall_times.retain(|&t| t > cutoff);
    if warmed && state.denial_times.len() as u64 == DENIAL_SPIKE_COUNT {
        findings.push(Finding {
            id: finding_id("denial_spike"),
            ts: now_secs(),
            rule: "denial_spike".into(),
            severity: Severity::Elevated,
            observed: format!("{} denied approvals in 1h", state.denial_times.len()),
            baseline: format!("<{DENIAL_SPIKE_COUNT}/hr typical"),
            detail: "repeated tool denials — something is probing the approval gate".into(),
        });
        state.denial_times.clear();
    }
    // Rule: firewall_spike — output firewall hit burst.
    if warmed
        && (state.firewall_times.len() as f64)
            > (state.firewall_hits as f64 / state.events_seen.max(1) as f64
                * 3600.0
                * RATE_SPIKE_FACTOR)
                .max(FIREWALL_SPIKE_FLOOR as f64)
    {
        findings.push(Finding {
            id: finding_id("firewall_spike"),
            ts: now_secs(),
            rule: "firewall_spike".into(),
            severity: Severity::Elevated,
            observed: format!("{} firewall hits in 1h", state.firewall_times.len()),
            baseline: format!("{} total over baseline", state.firewall_hits),
            detail: "output firewall is firing in a burst — possible injection probing".into(),
        });
        state.firewall_times.clear();
    }
}

// MARK: - Actions

/// Whether a finding should surface to the user in the given phase.
pub fn surfaces(phase: Phase, severity: Severity) -> bool {
    match phase {
        Phase::Gestation => false,
        _ => severity >= Severity::Elevated,
    }
}

/// Whether a critical finding may pull the brake itself.
pub fn may_brake(phase: Phase) -> bool {
    phase == Phase::Autopilot
}

/// Write a proposal markdown file compatible with the curious-proposal
/// pipeline (`**When:**`, `## Proposal` + ```json block).
pub fn write_proposal(f: &Finding, braked: bool) -> std::io::Result<PathBuf> {
    let dir = proposals_dir();
    fs::create_dir_all(&dir)?;
    let when = chrono::Utc::now()
        .format("%Y-%m-%d %H:%M:%S UTC")
        .to_string();
    let body = format!(
        "# IFY finding: {rule}\n\n**When:** {when}\n**Workspace:** system\n\n## Finding\n\n- Rule: `{rule}`\n- Severity: {sev}\n- Observed: {obs}\n- Baseline: {base}\n\n{detail}\n{brake}\n\n## Proposal\n\n```json\n{{\"no_patch\": true}}\n```\n",
        rule = f.rule,
        when = when,
        sev = serde_json::to_value(f.severity).ok().and_then(|v| v.as_str().map(String::from)).unwrap_or_default(),
        obs = f.observed,
        base = f.baseline,
        detail = f.detail,
        brake = if braked {
            "\n**Brake engaged:** kill switch pulled automatically (autopilot phase). Reply `resume bad apple` to restore."
        } else {
            ""
        },
    );
    let path = dir.join(format!("{}.md", f.id));
    fs::write(&path, body)?;
    Ok(path)
}

/// Desktop notification via osascript. Silent no-op on failure.
pub fn notify(title: &str, body: &str) {
    let esc = |s: &str| s.replace('\\', "\\\\").replace('"', "\\\"");
    let _ = std::process::Command::new("/usr/bin/osascript")
        .args([
            "-e",
            &format!(
                "display notification \"{}\" with title \"{}\"",
                esc(body),
                esc(title)
            ),
        ])
        .output();
}

/// Pull the brake: kill switch via the daemon, drop autopilot, mark
/// runtime_state SAFE_MODE. Every step is best-effort and logged —
/// a brake that half-fires still leaves the durable state files behind.
pub fn brake(reason: &str) -> Value {
    let mut outcome = json!({"kill_switch": false, "autopilot_off": false, "safe_mode": false});

    // 1. Ask the engine to kill itself (same path as the user's command).
    match crate::bad_apple_ipc::call_agent(
        "invoke_tool",
        Some(json!({"name": "kill_switch", "args": {}})),
        128,
    ) {
        Ok(_) => outcome["kill_switch"] = json!(true),
        Err(e) => outcome["kill_switch_error"] = json!(e.to_string()),
    }

    // 2. Freeze autopilot so a restart doesn't resume autonomy.
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
    let level_path = PathBuf::from(home)
        .join(".bad_apple")
        .join("autopilot_level");
    if fs::write(&level_path, "off").is_ok() {
        outcome["autopilot_off"] = json!(true);
    }

    // 3. Mark runtime_state SAFE_MODE so menu bar + dashboard show it.
    let path = runtime_state_path();
    let mut state: Value = fs::read_to_string(&path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_else(|| json!({}));
    if !state.is_object() {
        state = json!({});
    }
    let obj = state.as_object_mut().expect("checked object");
    obj.insert("mode".into(), json!("SAFE_MODE"));
    obj.insert("safe_mode_reason".into(), json!(format!("ify:{reason}")));
    obj.insert("updated_at".into(), json!(now_secs()));
    let tmp = path.with_file_name(format!(".runtime_state.{}.tmp", std::process::id()));
    if let Ok(mut f) = fs::File::create(&tmp) {
        if f.write_all(state.to_string().as_bytes()).is_ok() && fs::rename(&tmp, &path).is_ok() {
            outcome["safe_mode"] = json!(true);
        }
    }
    outcome
}

/// Render a finding in plain English via the fast tier. Numbers only —
/// the model narrates the finding, it doesn't generate it.
pub fn narrate(f: &Finding) -> Option<String> {
    if std::env::var("BADAPPLE_IFY_NARRATE")
        .map(|v| v == "0")
        .unwrap_or(false)
    {
        return None;
    }
    let prompt = format!(
        "You are IFY, Bad Apple's watchdog. One sentence, plain English, no jargon. \
         Report this security finding to the owner. Rule: {}. Observed: {}. Baseline: {}. Detail: {}",
        f.rule, f.observed, f.baseline, f.detail
    );
    crate::bad_apple_ipc::stream_query(&prompt, 120, |_| {}).ok()
}

/// Dispatch one finding: always log; surface/brake per phase.
/// Returns the path of a written proposal, if any.
pub fn dispatch(state: &IfyState, f: &Finding) -> Option<PathBuf> {
    log_finding(f);
    let phase = current_phase(state);
    if !surfaces(phase, f.severity) {
        return None;
    }
    let braked = f.severity == Severity::Critical && may_brake(phase);
    if braked {
        let outcome = brake(&f.rule);
        eprintln!("[ify] brake engaged for {}: {}", f.rule, outcome);
    }
    notify(
        &format!("IFY: {}", f.rule),
        &narrate(f).unwrap_or_else(|| f.detail.clone()),
    );
    write_proposal(f, braked).ok()
}
