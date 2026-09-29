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
