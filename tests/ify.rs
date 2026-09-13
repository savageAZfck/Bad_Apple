//! IFY watchdog tests: baseline learning, detection rules, phase gating,
//! and ledger tail/verify semantics over a synthetic ledger.

use bad_apple::ify;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::fs;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

/// Serializes tests: env vars and filesystem fixtures are process-global.
fn lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(())).lock().unwrap()
}

fn sha256_hex(s: &str) -> String {
    hex::encode(Sha256::digest(s.as_bytes()))
}

fn genesis() -> String {
    sha256_hex(ify::GENESIS_LABEL)
}

/// Build a chained ledger line in the current Bad Apple schema:
/// `{"data":{...},"prev_hash":"…","ts":"…","type":"…","hash":"…"}`
/// where hash = sha256 of the body (the line minus the hash field).
fn ledger_line(prev: &str, ty: &str, data: &serde_json::Value, ts: &str) -> (String, String) {
    let body = format!(
        "{{\"data\":{},\"prev_hash\":\"{}\",\"ts\":\"{}\",\"type\":\"{}\"}}",
        serde_json::to_string(data).unwrap(),
        prev,
        ts,
        ty
    );
    let hash = sha256_hex(&body);
    (
        format!("{},\"hash\":\"{}\"}}", &body[..body.len() - 1], hash),
        hash,
    )
}

struct Fixture {
    dir: PathBuf,
    ledger: PathBuf,
}

fn fixture(name: &str) -> Fixture {
    let dir = std::env::temp_dir().join(format!("ify-test-{}-{}", name, std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let ledger = dir.join("ledger.jsonl");
    fs::write(&ledger, "").unwrap();
    std::env::set_var("BADAPPLE_LEDGER", &ledger);
    std::env::set_var("BADAPPLE_IFY_DIR", dir.join("ify"));
    Fixture { dir, ledger }
}

fn append_line(fx: &Fixture, line: &str) {
    use std::io::Write;
    let mut f = fs::OpenOptions::new()
        .append(true)
        .open(&fx.ledger)
        .unwrap();
    writeln!(f, "{line}").unwrap();
}

/// Feed `n` chained "query" events spread over known hours, return tip.
fn seed_baseline(fx: &Fixture, state: &mut ify::IfyState, n: usize) -> String {
    let mut tip = genesis();
    for i in 0..n {
        // Fixed daytime hour so hourly buckets are warm.
        let (line, hash) = ledger_line(
            &tip,
            "query",
            &json!({"persona": "default"}),
            &format!("2026-10-0{}T1{}:00:00Z", (i % 9) + 1, i % 10),
        );
        append_line(fx, &line);
        tip = hash;
    }
    let r = ify::tail_ledger(state, &[]).unwrap();
    assert!(!r.chain_broken);
    tip
}

#[test]
fn gestation_learns_silently_then_surfaces() {
    let _g = lock();
    let fx = fixture("gestation");
    std::env::set_var("BADAPPLE_IFY_PHASE", "secondary");
    let mut state = ify::IfyState::default();

    // Warm the baseline past BASELINE_WARMUP_EVENTS.
    let tip = seed_baseline(&fx, &mut state, (ify::BASELINE_WARMUP_EVENTS + 10) as usize);
    assert!(state.events_seen >= ify::BASELINE_WARMUP_EVENTS);
    assert!(state.event_types["query"].count >= ify::BASELINE_WARMUP_EVENTS);

    // A novel event type now fires elevated.
    let (line, _) = ledger_line(&tip, "kernel_panic", &json!({}), "2026-10-10T12:00:00Z");
    append_line(&fx, &line);
    let r = ify::tail_ledger(&mut state, &[]).unwrap();
    let novel = r.findings.iter().find(|f| f.rule == "novel_event_type");
    assert!(
        novel.is_some(),
        "expected novel_event_type finding: {:?}",
        r.findings
    );
    assert_eq!(novel.unwrap().severity, ify::Severity::Elevated);

    // Phase gating: gestation hides it, secondary surfaces it.
    assert!(!ify::surfaces(
        ify::Phase::Gestation,
        ify::Severity::Elevated
    ));
    assert!(ify::surfaces(
        ify::Phase::Secondary,
        ify::Severity::Elevated
    ));
    assert!(!ify::surfaces(ify::Phase::Secondary, ify::Severity::Info));
}

#[test]
fn chain_break_is_critical() {
    let _g = lock();
    let fx = fixture("chainbreak");
    let mut state = ify::IfyState::default();
    let tip = seed_baseline(&fx, &mut state, 5);

    // Forge a line: valid JSON, right fields, wrong hash.
    let bad = format!(
        "{{\"data\":{{}},\"prev_hash\":\"{}\",\"ts\":\"2026-10-10T12:00:00Z\",\"type\":\"query\",\"hash\":\"{}\"}}",
        tip,
        sha256_hex("forged")
    );
    append_line(&fx, &bad);
    let r = ify::tail_ledger(&mut state, &[]).unwrap();
    assert!(r.chain_broken);
    let f = r.findings.iter().find(|f| f.rule == "chain_break").unwrap();
    assert_eq!(f.severity, ify::Severity::Critical);
    // Offset did NOT advance past the bad line.
    let meta_len = fs::metadata(&fx.ledger).unwrap().len();
    assert!(state.ledger_offset < meta_len);
}

#[test]
fn truncation_is_critical() {
    let _g = lock();
    let fx = fixture("truncate");
    let mut state = ify::IfyState::default();
    seed_baseline(&fx, &mut state, 5);
    let offset = state.ledger_offset;
    assert!(offset > 0);

    // Replace the ledger with a shorter file.
    fs::write(&fx.ledger, "").unwrap();
    let r = ify::tail_ledger(&mut state, &[]).unwrap();
    assert!(r.truncated);
    assert!(r
        .findings
        .iter()
        .any(|f| f.rule == "ledger_gap" && f.severity == ify::Severity::Critical));
}

#[test]
fn phase_machine_ages_and_forces() {
    let _g = lock();
    std::env::remove_var("BADAPPLE_IFY_PHASE");
    std::env::set_var("BADAPPLE_IFY_GESTATION_DAYS", "14");
    std::env::set_var("BADAPPLE_IFY_SECONDARY_DAYS", "14");

    let mut state = ify::IfyState::default();
    assert_eq!(ify::current_phase(&state), ify::Phase::Gestation);

    state.installed_at = ify::now_secs() - 15.0 * 86400.0;
    assert_eq!(ify::current_phase(&state), ify::Phase::Secondary);

    state.installed_at = ify::now_secs() - 29.0 * 86400.0;
    assert_eq!(ify::current_phase(&state), ify::Phase::Autopilot);

    std::env::set_var("BADAPPLE_IFY_PHASE", "gestation");
    assert_eq!(ify::current_phase(&state), ify::Phase::Gestation);
    std::env::remove_var("BADAPPLE_IFY_PHASE");

    // Brake only exists in autopilot.
    assert!(!ify::may_brake(ify::Phase::Gestation));
    assert!(!ify::may_brake(ify::Phase::Secondary));
    assert!(ify::may_brake(ify::Phase::Autopilot));
}

#[test]
fn kill_switch_event_surfaces() {
    let _g = lock();
    let fx = fixture("killswitch");
    let mut state = ify::IfyState::default();
    let tip = seed_baseline(&fx, &mut state, (ify::BASELINE_WARMUP_EVENTS + 5) as usize);

    let (line, _) = ledger_line(&tip, "kill_switch", &json!({}), "2026-10-10T12:00:00Z");
    append_line(&fx, &line);
    let r = ify::tail_ledger(&mut state, &[]).unwrap();
    assert!(r
        .findings
        .iter()
        .any(|f| f.rule == "kill_switch" && f.severity == ify::Severity::Elevated));
    assert_eq!(state.kill_switch_events, 1);
}

#[test]
fn proposals_use_curious_format() {
    let _g = lock();
    let fx = fixture("proposals");
    std::env::set_var("BADAPPLE_IFY_DIR", fx.dir.join("ify"));
    let f = ify::Finding {
        id: "ify-test-abc123".into(),
        ts: ify::now_secs(),
        rule: "novel_event_type".into(),
        severity: ify::Severity::Elevated,
        observed: "kernel_panic".into(),
        baseline: "5 known types".into(),
        detail: "test finding".into(),
    };
    let path = ify::write_proposal(&f, false).unwrap();
    let text = fs::read_to_string(&path).unwrap();
    // Compatible with parse_proposal_file in badapple-dashboard.rs.
    assert!(text.contains("**When:**"));
    assert!(text.contains("**Workspace:**"));
    assert!(text.contains("## Proposal"));
    assert!(text.contains("```json"));
    assert!(text.contains("\"no_patch\": true"));
}
