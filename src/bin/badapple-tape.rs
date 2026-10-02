//! badapple-tape — flight recorder daemon for Bad Apple.
//!
//! Thin wrapper over the public `flight_tape` crate (pattern matches
//! badapple-sovereign / badapple-respawn). Tails the primary ledger,
//! drains the engine's intent drop-stream, and freezes signed incident
//! bundles on kill-switch events, engine death, or manual request.
//!
//! Runs as the user LaunchAgent `com.badapple.tape`.

use flight_tape::daemon::{Daemon, DaemonConfig};
use flight_tape::freeze::{list_incidents, FreezeTrigger, StateRoot};
use flight_tape::replay;
use flight_tape::ring::{Ring, RingConfig};
use flight_tape::trigger::TriggerConfig;
use flight_tape::verify::verify_bundle;
use std::error::Error;
use std::path::PathBuf;
use std::process;
use std::sync::atomic::AtomicBool;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

const STATE_DIR: &str = "/var/lib/bad_apple";
const LEDGER: &str = "/var/lib/bad_apple/ledger.jsonl";
const ENGINE_PROC: &str = "badapple-engine";

fn tape_dir() -> PathBuf {
    PathBuf::from(STATE_DIR).join("tape")
}

fn learned_dir() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
    PathBuf::from(home).join(".bad_apple")
}

fn default_state_roots() -> Vec<StateRoot> {
    vec![
        StateRoot {
            label: "platform".into(),
            dir: PathBuf::from(STATE_DIR),
            // Heavy, non-decision state — excluded from snapshot manifests.
            exclude: vec![
                "kv_cache".into(),
                "generated_images".into(),
                "install_backups".into(),
                "tape".into(),
            ],
        },
        StateRoot {
            label: "learned".into(),
            dir: learned_dir(),
            exclude: vec![".respawn".into()],
        },
    ]
}

fn usage() -> ! {
    eprintln!("badapple-tape — Bad Apple flight recorder");
    eprintln!("Usage: badapple-tape <command> [options]");
    eprintln!("Commands:");
    eprintln!("  daemon              Run the recorder (LaunchAgent payload)");
    eprintln!("  freeze [reason]     Freeze the tape into a signed incident bundle now");
    eprintln!("  verify <bundle>     Verify an incident bundle (offline)");
    eprintln!("  replay <bundle>     Render the incident timeline");
    eprintln!("  report <bundle>     Emit a structured incident disclosure (JSON)");
    eprintln!("       [--kind K]     Filter to event kinds (repeatable)");
    eprintln!("       [--around N]   Center on seq N (with --context C, default 20)");
    eprintln!("  status              Ring stats, chain head, incident list");
    eprintln!("  incidents           List frozen incident bundles");
    eprintln!("  -h, --help          Show this help");
    eprintln!();
    eprintln!(
        "Paths: tape={} ledger={} watch={}",
        tape_dir().display(),
        LEDGER,
        ENGINE_PROC
    );
    process::exit(if std::env::args().any(|a| a == "-h" || a == "--help") {
        0
    } else {
        1
    });
}

fn daemon() -> Result<(), Box<dyn Error>> {
    let dir = tape_dir();
    std::fs::create_dir_all(&dir)?;
    let cfg = DaemonConfig {
        tape_dir: dir.clone(),
        ledger_sources: vec![PathBuf::from(LEDGER)],
        intent_path: dir.join(flight_tape::INTENT_STREAM),
        subject_name: "badapple".into(),
        subject_version: env!("CARGO_PKG_VERSION").to_string(),
        state_roots: default_state_roots(),
        poll: Duration::from_secs(2),
        ring: RingConfig::default(),
        triggers: TriggerConfig {
            freeze_kinds: vec!["kill_switch".into()],
            watch_process: Some(ENGINE_PROC.into()),
        },
    };
    let stop = std::sync::Arc::new(AtomicBool::new(false));
    signal_hook::flag::register(signal_hook::consts::SIGTERM, stop.clone())?;
    signal_hook::flag::register(signal_hook::consts::SIGINT, stop.clone())?;
    let mut d = Daemon::open(cfg)?.with_stop(stop);
    eprintln!("[tape] recording {} → {}", LEDGER, dir.display());
    d.run()?;
    Ok(())
}

fn freeze(reason: Option<String>) -> Result<(), Box<dyn Error>> {
    let dir = tape_dir();
    let reason = reason.unwrap_or_else(|| "manual freeze".into());

    // If the daemon holds the ring lock, signal it through the trigger file —
    // it freezes with live state on its next poll (≤2s) instead of racing us.
    let prior = list_incidents(&dir);
    match Ring::open(&dir, RingConfig::default()) {
        Err(flight_tape::Error::Lock(_)) => {
            std::fs::write(dir.join(flight_tape::FREEZE_FILE), &reason)?;
            for _ in 0..60 {
                std::thread::sleep(Duration::from_millis(200));
                let now = list_incidents(&dir);
                if let Some(newest) = now.iter().find(|i| !prior.contains(i)) {
                    println!("froze via daemon → {}", newest);
                    return Ok(());
                }
            }
            bail_timeout()
        }
        Err(e) => Err(e.into()),
        Ok(ring) => {
            let trig = FreezeTrigger {
                kind: "manual".into(),
                detail: reason,
                ts: SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs(),
            };
            let bundle = flight_tape::freeze::freeze(
                &ring,
                trig,
                "badapple",
                env!("CARGO_PKG_VERSION"),
                &default_state_roots(),
                prior,
            )?;
            println!(
                "froze {} frames (seq {}..{}) → {}",
                bundle.frames,
                bundle.first_seq,
                bundle.last_seq,
                bundle.dir.display()
            );
            Ok(())
        }
    }
}

fn bail_timeout() -> Result<(), Box<dyn Error>> {
    Err("daemon did not freeze within 12s — check tape.log".into())
}

fn status() -> Result<(), Box<dyn Error>> {
    let dir = tape_dir();
    // Lock-free read: the daemon may hold the ring lock. Parse ring.jsonl
    // directly and tolerate a torn tail line mid-append.
    let path = dir.join("ring.jsonl");
    let mut count = 0u64;
    let mut first_seq = 0u64;
    let mut last_seq = 0u64;
    let mut head = String::new();
    if let Ok(content) = std::fs::read_to_string(&path) {
        for line in content.lines() {
            let Ok(f) = serde_json::from_str::<serde_json::Value>(line) else {
                continue;
            };
            let seq = f["seq"].as_u64().unwrap_or(0);
            if first_seq == 0 {
                first_seq = seq;
            }
            last_seq = seq;
            if let Some(h) = f["hash"].as_str() {
                head = h.to_string();
            }
            count += 1;
        }
    }
    println!("tape:     {}", dir.display());
    println!("frames:   {} (seq {}..{})", count, first_seq, last_seq);
    println!(
        "head:     {}",
        if head.is_empty() {
            "genesis".into()
        } else {
            head
        }
    );
    println!(
        "daemon:   {}",
        if daemon_running() {
            "running"
        } else {
            "stopped"
        }
    );
    println!("incidents:");
    for i in list_incidents(&dir).iter().rev().take(10) {
        println!("  {}", i);
    }
    Ok(())
}

fn daemon_running() -> bool {
    std::process::Command::new("pgrep")
        .args(["-f", "badapple-tape daemon"])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// `badapple tape report <bundle>` — render a frozen incident bundle as a
/// structured disclosure document: trigger, integrity verification, an
/// authority-classified action summary, condensed timeline, and the state
/// snapshot ids an auditor can materialize to reconstruct exact state.
/// Deliberately factual — severity labels are hints, not legal conclusions.
fn report(bundle_dir: &PathBuf) -> Result<(), Box<dyn Error>> {
    use sha2::{Digest, Sha256};

    let manifest_text = std::fs::read_to_string(bundle_dir.join("manifest.json"))?;
    let manifest: serde_json::Value = serde_json::from_str(&manifest_text)?;
    let manifest_sha = hex::encode(Sha256::digest(manifest_text.trim_end().as_bytes()));

    let verified = verify_bundle(bundle_dir).ok();
    let frames = replay::load_frames(bundle_dir)?;

    // Authority classification over the frozen window.
    let mut intent_verdicts = std::collections::BTreeMap::<String, u64>::new();
    let mut kind_counts = std::collections::BTreeMap::<String, u64>::new();
    let mut kill_events = 0u64;
    let mut escalations = 0u64;
    for f in &frames {
        *kind_counts.entry(f.kind.clone()).or_default() += 1;
        if f.kind == "kill_switch" {
            kill_events += 1;
        }
        if f.kind == "tool_intent" {
            let verdict = f.body["verdict"].as_str().unwrap_or("unknown");
            *intent_verdicts.entry(verdict.to_string()).or_default() += 1;
        }
        if f.kind == "ledger" && f.body["type"].as_str() == Some("council_escalated") {
            escalations += 1;
        }
    }

    let trigger_kind = manifest["trigger"]["kind"].as_str().unwrap_or("unknown");
    let severity_hint = match trigger_kind {
        "kill_switch" => "operator_brake",
        "process_death" | "crash" => "engine_fault",
        "manual" => "manual_freeze",
        _ => "unclassified",
    };
    // An operator brake or unexpected engine death is what an auditor wants
    // to see disclosed; a manual freeze is routine evidence capture.
    let statutory_candidate = matches!(severity_hint, "operator_brake" | "engine_fault");

    let trigger_ts = manifest["trigger"]["ts"].as_u64().unwrap_or(0);
    let trigger_iso = chrono::DateTime::from_timestamp(trigger_ts as i64, 0)
        .map(|t| t.to_rfc3339())
        .unwrap_or_else(|| trigger_ts.to_string());

    let timeline: Vec<serde_json::Value> = frames
        .iter()
        .map(|f| {
            let mut body = serde_json::to_string(&f.body).unwrap_or_else(|_| "{}".into());
            if body.len() > 200 {
                let mut end = 197;
                while !body.is_char_boundary(end) {
                    end -= 1;
                }
                body.truncate(end);
                body.push_str("...");
            }
            serde_json::json!({
                "seq": f.seq, "ts": f.ts, "kind": f.kind, "src": f.src, "body": body,
            })
        })
        .collect();

    let report = serde_json::json!({
        "kind": "badapple.incident_report",
        "version": 1,
        "generated_at": chrono::Utc::now().to_rfc3339(),
        "incident": {
            "id": bundle_dir.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default(),
            "bundle": bundle_dir.display().to_string(),
            "trigger": {"kind": trigger_kind, "detail": manifest["trigger"]["detail"], "detected_at": trigger_iso},
            "subject": manifest["subject"],
            "window": manifest["window"],
        },
        "classification": {
            "severity_hint": severity_hint,
            "disclosure_candidate": statutory_candidate,
            "note": "severity_hint is mechanical; statutory classification requires human assessment",
        },
        "integrity": {
            "manifest_sha256": manifest_sha,
            "head_hash": manifest["head_hash"],
            "verification": verified.map(|r| serde_json::json!({
                "frames_checked": r.frames_checked,
                "chain_ok": r.chain_ok,
                "manifest_ok": r.manifest_ok,
                "signature_ok": r.signature_ok,
            })),
            "manifest_pubkey": manifest["pubkey"],
        },
        "authority": {
            "tool_intent_verdicts": intent_verdicts,
            "kill_switch_events": kill_events,
            "council_escalations": escalations,
            "frame_kind_counts": kind_counts,
        },
        "state_snapshots": manifest["snapshots"],
        "prior_incidents": manifest["prior_incidents"],
        "timeline": timeline,
    });
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

fn main() {
    let mut args = std::env::args().skip(1);
    let cmd = match args.next() {
        Some(c) => c,
        None => usage(),
    };
    let rest: Vec<String> = args.collect();

    let result: Result<(), Box<dyn Error>> = match cmd.as_str() {
        "daemon" => daemon(),
        "freeze" => freeze(rest.first().cloned()),
        "verify" => {
            let target = match rest.first().map(PathBuf::from) {
                Some(t) => t,
                None => usage(),
            };
            match verify_bundle(&target) {
                Ok(r) => {
                    for p in &r.problems {
                        eprintln!("  ✗ {}", p);
                    }
                    println!(
                        "{}: {} frames, chain={}, manifest={}, signature={}",
                        if r.ok() { "VALID" } else { "INVALID" },
                        r.frames_checked,
                        r.chain_ok,
                        r.manifest_ok,
                        r.signature_ok
                    );
                    if r.ok() {
                        Ok(())
                    } else {
                        process::exit(1)
                    }
                }
                Err(e) => Err(e.into()),
            }
        }
        "replay" => {
            let target = match rest.first().map(PathBuf::from) {
                Some(t) => t,
                None => usage(),
            };
            let mut kinds = Vec::new();
            let mut around = None;
            let mut context = 20usize;
            let mut i = 1;
            while i < rest.len() {
                match rest[i].as_str() {
                    "--kind" => {
                        if let Some(k) = rest.get(i + 1) {
                            kinds.push(k.clone());
                            i += 1;
                        }
                    }
                    "--around" => {
                        around = rest.get(i + 1).and_then(|v| v.parse().ok());
                        i += 1;
                    }
                    "--context" => {
                        context = rest.get(i + 1).and_then(|v| v.parse().ok()).unwrap_or(20);
                        i += 1;
                    }
                    _ => {}
                }
                i += 1;
            }
            match replay::load_frames(&target) {
                Ok(f) => {
                    print!(
                        "{}",
                        replay::render(&f, &kinds, around.map(|s| (s, context)))
                    );
                    Ok(())
                }
                Err(e) => Err(e.into()),
            }
        }
        "report" => {
            let target = match rest.first().map(PathBuf::from) {
                Some(t) => t,
                None => usage(),
            };
            report(&target)
        }
        "status" => status(),
        "incidents" => {
            for i in list_incidents(&tape_dir()) {
                println!("{}", i);
            }
            Ok(())
        }
        _ => usage(),
    };

    if let Err(e) = result {
        eprintln!("badapple-tape: {}", e);
        process::exit(1);
    }
}
