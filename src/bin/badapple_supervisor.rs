//! Bounded launchd health supervisor for the Bad Apple service family.
//!
//! Replaces the Python badapple_supervisor.py with a native Rust binary.
//! Checks gatekeeper, MLX daemon, and TTS services, restarts them within
//! a bounded budget, and enters safe mode if the budget is exhausted.

use std::collections::HashMap;
use std::env;
use std::fs;
use std::io::Write;
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use serde_json::json;

const DATA_DIR: &str = "/var/lib/bad_apple";
const STATE_FILE: &str = "supervisor_state.json";
const RUNTIME_FILE: &str = "runtime_state.json";
const FAILURE_THRESHOLD: u32 = 3;
const RESTART_BUDGET: usize = 2;
const RESTART_WINDOW_SECS: f64 = 600.0;
const STARTUP_GRACE_SECS: f64 = 180.0;

#[derive(Clone)]
struct Service {
    name: String,
    domain: String,
    socket: Option<String>,
}

#[derive(Serialize, Deserialize, Clone, Default)]
struct ServiceEntry {
    consecutive_failures: u32,
    restart_attempts: Vec<f64>,
    start_at: f64,
    last_check: f64,
    healthy: bool,
}

#[derive(Serialize, Deserialize, Clone, Default)]
struct SupervisorState {
    services: HashMap<String, ServiceEntry>,
    last_report: Option<serde_json::Value>,
    #[serde(default)]
    last_index_check: f64,
}

/// Seconds between watched-index rescans (default 120 — the check loop runs
/// every ~30s, so this is roughly every fourth tick).
fn watch_interval() -> f64 {
    env::var("BADAPPLE_WATCH_INTERVAL")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(120.0)
}

fn cli_path() -> Option<PathBuf> {
    if let Ok(p) = env::var("BADAPPLE_CLI") {
        return Some(PathBuf::from(p));
    }
    for candidate in [
        "/Applications/Bad Apple.app/Contents/Helpers/badapple",
        "/usr/local/bin/badapple",
    ] {
        let p = PathBuf::from(candidate);
        if p.exists() {
            return Some(p);
        }
    }
    None
}

/// Re-index every registered watch path by shelling the CLI as the console
/// user — the supervisor runs as root and a root-owned redb would break
/// user-side `index`/`recall` writes. `launchctl asuser` makes the CLI do
/// the work with the user's permissions and HOME.
fn refresh_watched_indexes(state: &mut SupervisorState) -> serde_json::Value {
    let now = now_secs();
    if now - state.last_index_check < watch_interval() {
        return json!({"skipped": "interval"});
    }
    state.last_index_check = now;
    let dirs = bad_apple::scavenger::watched_dirs();
    if dirs.is_empty() {
        return json!({"watched": 0});
    }
    let Some(cli) = cli_path() else {
        return json!({"watched": dirs.len(), "error": "no badapple cli found"});
    };
    let uid = console_uid();
    if uid == 0 {
        // No console user (or /dev/console owned by root): indexing as root
        // would chown the redb away from the user, so skip rather than fail.
        return json!({"watched": dirs.len(), "skipped": "no console user"});
    }
    let mut child = match Command::new("launchctl")
        .args(["asuser", &uid.to_string()])
        .arg(&cli)
        .arg("index")
        .arg("--watched")
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => return json!({"watched": dirs.len(), "error": e.to_string()}),
    };
    // Bounded wait: an incremental rescan is seconds, but a stalled CLI must
    // never wedge the health loop.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(240);
    loop {
        match child.try_wait() {
            Ok(Some(_)) => {
                let out = child.wait_with_output();
                return match out {
                    Ok(o) => json!({
                        "watched": dirs.len(),
                        "ok": o.status.success(),
                        "output": String::from_utf8_lossy(&o.stdout).trim(),
                    }),
                    Err(e) => {
                        json!({"watched": dirs.len(), "error": e.to_string()})
                    }
                };
            }
            Ok(None) if std::time::Instant::now() < deadline => {
                std::thread::sleep(std::time::Duration::from_millis(250));
            }
            Ok(None) => {
                let _ = child.kill();
                let _ = child.wait();
                return json!({"watched": dirs.len(), "error": "index timed out"});
            }
            Err(e) => {
                let _ = child.kill();
                return json!({"watched": dirs.len(), "error": e.to_string()});
            }
        }
    }
}

fn now_secs() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64()
}

fn data_dir() -> PathBuf {
    PathBuf::from(env::var("BADAPPLE_DATA_DIR").unwrap_or_else(|_| DATA_DIR.to_string()))
}

fn check_interval() -> u64 {
    env::var("BADAPPLE_SUPERVISOR_INTERVAL")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(30)
}

fn startup_grace() -> f64 {
    env::var("BADAPPLE_SUPERVISOR_STARTUP_GRACE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(STARTUP_GRACE_SECS)
}

fn console_uid() -> u32 {
    let output = Command::new("stat")
        .args(["-f", "%Su", "/dev/console"])
        .output();
    if let Ok(out) = output {
        let user = String::from_utf8_lossy(&out.stdout).trim().to_string();
        if let Ok(uid) = Command::new("id").args(["-u", &user]).output() {
            if let Ok(s) = String::from_utf8(uid.stdout) {
                if let Ok(n) = s.trim().parse::<u32>() {
                    return n;
                }
            }
        }
    }
    unsafe { libc::getuid() }
}

fn services() -> Vec<Service> {
    let uid = console_uid();
    vec![
        Service {
            name: "gatekeeper".into(),
            domain: "system/com.badapple.gatekeeper".into(),
            socket: Some("/var/run/badapple/substrate.sock".into()),
        },
        Service {
            name: "mlx".into(),
            domain: "system/com.badapple.mlx".into(),
            socket: Some("/var/run/badapple/substrate_mlx.sock".into()),
        },
        Service {
            name: "tts".into(),
            domain: format!("gui/{}/com.badapple.tts", uid),
            socket: Some("/tmp/badapple_tts.sock".into()),
        },
    ]
}

fn launchd_running(domain: &str) -> bool {
    let result = Command::new("launchctl").args(["print", domain]).output();
    match result {
        Ok(out) => {
            out.status.success() && String::from_utf8_lossy(&out.stdout).contains("state = running")
        }
        Err(_) => false,
    }
}

fn socket_ready_timed(path: &str) -> bool {
    if !Path::new(path).exists() {
        return false;
    }
    // UnixStream::connect is already fast for local sockets.
    // Set a read timeout after connect to avoid hanging.
    match UnixStream::connect(path) {
        Ok(stream) => {
            let _ = stream.shutdown(std::net::Shutdown::Write);
            true
        }
        Err(_) => false,
    }
}

fn restart_service(domain: &str) -> bool {
    Command::new("launchctl")
        .args(["kickstart", "-k", domain])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn load_state() -> SupervisorState {
    let path = data_dir().join(STATE_FILE);
    match fs::read_to_string(&path) {
        Ok(content) => serde_json::from_str(&content).unwrap_or_default(),
        Err(_) => SupervisorState::default(),
    }
}

fn save_state(state: &SupervisorState) {
    let dir = data_dir();
    let _ = fs::create_dir_all(&dir);
    let path = dir.join(STATE_FILE);
    let tmp = dir.join(format!(".{}.{}.tmp", STATE_FILE, std::process::id()));
    if let Ok(json) = serde_json::to_string_pretty(state) {
        if let Ok(mut f) = fs::File::create(&tmp) {
            let _ = f.write_all(json.as_bytes());
            let _ = f.flush();
            let _ = fs::rename(&tmp, &path);
        }
    }
}

fn write_runtime_state(mode: &str, safe_mode_reason: Option<&str>) {
    let dir = data_dir();
    let _ = fs::create_dir_all(&dir);
    let path = dir.join(RUNTIME_FILE);
    // Read existing state, update mode, write back.
    let mut state: serde_json::Value = match fs::read_to_string(&path) {
        Ok(content) => serde_json::from_str(&content).unwrap_or(json!({})),
        Err(_) => json!({}),
    };
    if let Some(obj) = state.as_object_mut() {
        obj.insert("mode".into(), json!(mode));
        if let Some(reason) = safe_mode_reason {
            obj.insert("safe_mode_reason".into(), json!(reason));
        } else {
            obj.remove("safe_mode_reason");
        }
        obj.insert(
            "revision".into(),
            json!(obj.get("revision").and_then(|v| v.as_i64()).unwrap_or(0) + 1),
        );
        obj.insert("updated_at".into(), json!(now_secs()));
    } else {
        state = json!({
            "mode": mode,
            "safe_mode_reason": safe_mode_reason,
            "revision": 1,
            "updated_at": now_secs(),
            "killed": false,
            "kill_reason": null,
            "private_mode": false,
            "schema_version": 1,
        });
    }
    let tmp = dir.join(format!(".{}.{}.tmp", RUNTIME_FILE, std::process::id()));
    if let Ok(json) = serde_json::to_string_pretty(&state) {
        if let Ok(mut f) = fs::File::create(&tmp) {
            let _ = f.write_all(json.as_bytes());
            let _ = f.flush();
            let _ = fs::rename(&tmp, &path);
        }
    }
}

fn restart_allowed(entry: &mut ServiceEntry, now: f64) -> bool {
    entry
        .restart_attempts
        .retain(|&t| now - t < RESTART_WINDOW_SECS);
    entry.restart_attempts.len() < RESTART_BUDGET
}

fn check_once(repair: bool) -> serde_json::Value {
    let now = now_secs();
    let mut state = load_state();
    let mut report = json!({
        "timestamp": now,
        "services": {},
        "safe_mode": false,
    });
    report["thermal"] = bad_apple::production_blueprint::probe_thermal();
    report["index"] = refresh_watched_indexes(&mut state);
    let grace = startup_grace();

    for svc in services() {
        let entry = state.services.entry(svc.name.clone()).or_default();

        let running = launchd_running(&svc.domain);
        if running {
            if entry.start_at == 0.0 {
                entry.start_at = now;
            }
        } else {
            entry.start_at = 0.0;
        }

        let socket_ok = svc
            .socket
            .as_ref()
            .map(|s| socket_ready_timed(s))
            .unwrap_or(true);

        let in_grace = running
            && svc.socket.is_some()
            && !socket_ok
            && entry.start_at > 0.0
            && (now - entry.start_at) < grace;

        let healthy = running && (socket_ok || in_grace);
        let mut action;

        if healthy {
            entry.consecutive_failures = 0;
            action = if in_grace { "starting" } else { "none" };
        } else {
            entry.consecutive_failures += 1;
            action = "observe";
            if repair && entry.consecutive_failures >= FAILURE_THRESHOLD {
                if restart_allowed(entry, now) {
                    let attempted = restart_service(&svc.domain);
                    entry.restart_attempts.push(now);
                    if attempted {
                        entry.consecutive_failures = 0;
                        action = "restart";
                    } else {
                        action = "restart_failed";
                    }
                } else {
                    let reason = format!("{} exceeded restart budget", svc.name);
                    write_runtime_state("SAFE_MODE", Some(&reason));
                    action = "safe_mode";
                    if let Some(obj) = report.as_object_mut() {
                        obj.insert("safe_mode".into(), json!(true));
                    }
                }
            }
        }

        entry.last_check = now;
        entry.healthy = healthy;

        if let Some(services) = report
            .as_object_mut()
            .and_then(|r| r.get_mut("services"))
            .and_then(|s| s.as_object_mut())
        {
            services.insert(
                svc.name.clone(),
                json!({
                    "running": running,
                    "socket_ready": socket_ok,
                    "in_grace": in_grace,
                    "healthy": healthy,
                    "action": action,
                    "consecutive_failures": entry.consecutive_failures,
                }),
            );
        }
    }

    state.last_report = Some(report.clone());
    save_state(&state);

    // Keep runtime_state.json current so the menu bar doesn't show a stale
    // safe-mode message after the supervisor has recovered.
    let all_healthy = state.services.values().all(|e| e.healthy);
    let in_safe_mode = report
        .get("safe_mode")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    if !in_safe_mode && all_healthy {
        write_runtime_state("READY", None);
    }

    report
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let once = args.iter().any(|a| a == "--once");
    let no_repair = args.iter().any(|a| a == "--no-repair");

    if once {
        let report = check_once(!no_repair);
        println!(
            "{}",
            serde_json::to_string_pretty(&report)
                .unwrap_or_else(|_| "{\"error\":\"serialization failed\"}".to_string())
        );
        return;
    }

    let interval = check_interval();
    loop {
        let report = check_once(!no_repair);
        println!(
            "{}",
            serde_json::to_string(&report)
                .unwrap_or_else(|_| "{\"error\":\"serialization failed\"}".to_string())
        );
        std::thread::sleep(Duration::from_secs(interval));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn index_refresh_empty_registry_is_noop() {
        let file = std::env::temp_dir().join(format!("ba-sup-watch-{}", std::process::id()));
        std::fs::write(&file, "").unwrap();
        std::env::set_var("BADAPPLE_WATCH_PATHS", &file);
        let mut state = SupervisorState::default();
        let report = refresh_watched_indexes(&mut state);
        assert_eq!(report["watched"], 0);
        std::env::remove_var("BADAPPLE_WATCH_PATHS");
        let _ = std::fs::remove_file(&file);
    }

    #[test]
    fn index_refresh_missing_cli_reports_error() {
        let file = std::env::temp_dir().join(format!("ba-sup-watch2-{}", std::process::id()));
        let dir = std::env::temp_dir().join(format!("ba-sup-dir-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(&file, format!("{}\n", dir.display())).unwrap();
        std::env::set_var("BADAPPLE_WATCH_PATHS", &file);
        std::env::set_var("BADAPPLE_CLI", "/nonexistent/badapple");
        let mut state = SupervisorState::default();
        let report = refresh_watched_indexes(&mut state);
        // Spawn failure, launchctl-reported child failure, or uid-0 skip —
        // any of these is a bounded, nonfatal report; the loop survives.
        assert!(
            report.get("error").is_some()
                || report.get("skipped").is_some()
                || report["ok"] == false
        );
        std::env::remove_var("BADAPPLE_WATCH_PATHS");
        std::env::remove_var("BADAPPLE_CLI");
        let _ = std::fs::remove_file(&file);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
