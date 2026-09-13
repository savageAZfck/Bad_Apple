//! badapple-ify — IFY watchdog daemon.
//!
//! Polls the Bad Apple audit ledger, verifies each new line against the
//! hash chain, updates behavioral baselines, and dispatches findings
//! per the phase state machine (gestation → secondary → autopilot).
//!
//! Usage:
//!   badapple-ify            run the watch loop (LaunchAgent mode)
//!   badapple-ify --once     single tail+detect pass, print JSON report
//!   badapple-ify --status   print phase, baselines, recent findings

use bad_apple::ify;
use serde_json::json;
use std::env;
use std::fs;
use std::io::{BufRead, BufReader};
use std::process;

fn poll_interval() -> u64 {
    env::var("BADAPPLE_IFY_INTERVAL")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(15)
}

fn status_report(state: &ify::IfyState) -> serde_json::Value {
    let phase = ify::current_phase(state);
    let mut top_types: Vec<(&String, &u64)> = state
        .event_types
        .iter()
        .map(|(k, v)| (k, &v.count))
        .collect();
    top_types.sort_by(|a, b| b.1.cmp(a.1));
    let recent: Vec<serde_json::Value> = fs::File::open(ify::findings_path())
        .ok()
        .map(|f| {
            BufReader::new(f)
                .lines()
                .map_while(Result::ok)
                .filter(|l| !l.trim().is_empty())
                .collect::<Vec<_>>()
        })
        .unwrap_or_default()
        .iter()
        .rev()
        .take(10)
        .filter_map(|l| serde_json::from_str(l).ok())
        .collect();
    json!({
        "phase": phase.as_str(),
        "installed_days_ago": ((ify::now_secs() - state.installed_at) / 86400.0 * 10.0).round() / 10.0,
        "events_seen": state.events_seen,
        "ledger_offset": state.ledger_offset,
        "ledger_tip": state.ledger_tip,
        "event_types": state.event_types.len(),
        "top_event_types": top_types.iter().take(10).map(|(k, v)| json!({(*k).clone(): v})).collect::<Vec<_>>(),
        "approvals": {"granted": state.approvals_granted, "denied": state.approvals_denied},
        "firewall_hits": state.firewall_hits,
        "kill_switch_events": state.kill_switch_events,
        "recent_findings": recent,
    })
}

fn run_pass(state: &mut ify::IfyState, secrets: &[Vec<u8>]) -> ify::TailReport {
    match ify::tail_ledger(state, secrets) {
        Ok(report) => report,
        Err(e) => ify::TailReport {
            new_events: 0,
            findings: vec![ify::Finding {
                id: format!("ify-error-{}", ify::now_secs() as u64),
                ts: ify::now_secs(),
                rule: "tail_error".into(),
                severity: ify::Severity::Info,
                observed: e.to_string(),
                baseline: "readable ledger".into(),
                detail: format!("could not tail ledger: {e}"),
            }],
            chain_broken: false,
            truncated: false,
        },
    }
}

fn main() {
    if env::var("BADAPPLE_IFY").map(|v| v == "0").unwrap_or(false) {
        println!("ify disabled via BADAPPLE_IFY=0");
        return;
    }
    let args: Vec<String> = env::args().collect();
    let once = args.iter().any(|a| a == "--once");
    let status = args.iter().any(|a| a == "--status");

    if status {
        let state = ify::load_state();
        println!(
            "{}",
            serde_json::to_string_pretty(&status_report(&state)).unwrap_or_else(|_| "{}".into())
        );
        return;
    }

    let secrets = ify::load_slicks_secrets();
    let mut state = ify::load_state();

    if once {
        let report = run_pass(&mut state, &secrets);
        for f in &report.findings {
            ify::dispatch(&state, f);
        }
        ify::save_state(&state);
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({
                "phase": ify::current_phase(&state).as_str(),
                "new_events": report.new_events,
                "findings": report.findings.len(),
                "chain_broken": report.chain_broken,
                "truncated": report.truncated,
            }))
            .unwrap_or_else(|_| "{}".into())
        );
        return;
    }

    let interval = poll_interval();
    println!(
        "[ify] watching {} (phase {}, {}s interval)",
        ify::ledger_path().display(),
        ify::current_phase(&state).as_str(),
        interval
    );
    loop {
        let report = run_pass(&mut state, &secrets);
        for f in &report.findings {
            ify::dispatch(&state, f);
        }
        if report.new_events > 0 || !report.findings.is_empty() {
            ify::save_state(&state);
        }
        std::thread::sleep(std::time::Duration::from_secs(interval));
    }
}

#[allow(dead_code)]
fn _exit() -> ! {
    process::exit(0)
}
