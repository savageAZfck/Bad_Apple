//! Integration test for the Bad Apple launchd-ready background daemon.
//!
//! Spawns the release binary in `BADAPPLE_ANE_DAEMON=1` mode, waits for the
//! daemon heartbeat, then asserts that the process has zero network sockets
//! (`lsof -i`).  This is a dry-run substitute for the actual `launchd` install.

use std::io::{BufRead, BufReader};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

const ARTIFACT_MODEL: &str =
    "tests/ane_brain_perf/artifacts/qwen3b_ane_shards/conversion_manifest.json";
const ARTIFACT_TOKENIZER: &str = "tests/ane_brain_perf/artifacts/qwen3b_ane_shards/tokenizer.json";
const DAEMON_READY_LOG: &str = "BAD APPLE SLICKS ingress live at";
const TEST_SLICKS_SECRET: &str = "0123456789abcdef0123456789abcdef";

fn artifact_dir() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn find_binary(name: &str) -> Option<std::path::PathBuf> {
    if let Ok(exe) = std::env::var(format!("CARGO_BIN_EXE_{name}")) {
        let path = std::path::PathBuf::from(exe);
        if path.exists() {
            return Some(path);
        }
    }
    for profile in ["release", "debug"] {
        let path = std::path::PathBuf::from(format!("target/{profile}/{name}"));
        if path.exists() {
            return Some(path);
        }
    }
    None
}

#[test]
fn daemon_mode_initializes_and_holds_zero_network_sockets() {
    let manifest = artifact_dir().join(ARTIFACT_MODEL);
    let tokenizer = artifact_dir().join(ARTIFACT_TOKENIZER);
    if !manifest.exists() || !tokenizer.exists() {
        eprintln!(
            "skipping daemon test: ANE artifacts not found at {} / {}",
            manifest.display(),
            tokenizer.display()
        );
        return;
    }

    let binary = find_binary("badappled").expect("badappled binary must be built");
    let cli = find_binary("badapple").expect("badapple binary must be built");
    let socket_path = std::env::temp_dir().join(format!(
        "badapple-test-{}-{}.sock",
        std::process::id(),
        rand::random::<u64>()
    ));

    let mut child = Command::new(&binary)
        .arg("--daemon")
        .env("BADAPPLE_ANE_DAEMON", "1")
        .env("BADAPPLE_ANE_MODEL", &manifest)
        .env("BADAPPLE_ANE_TOKENIZER", &tokenizer)
        .env("BADAPPLE_SOCKET_PATH", &socket_path)
        .env("BADAPPLE_SLICKS_SECRET", TEST_SLICKS_SECRET)
        .env("RUST_LOG", "info,bad_apple=info")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("failed to spawn daemon");

    let start = Instant::now();
    let mut reader = BufReader::new(child.stdout.take().expect("daemon stdout must be captured"));
    let mut line = String::new();

    loop {
        if start.elapsed() > Duration::from_secs(120) {
            let _ = child.kill();
            panic!("daemon did not emit a heartbeat within 120 seconds");
        }
        reader
            .read_line(&mut line)
            .expect("daemon stdout must be readable");
        if line.contains(DAEMON_READY_LOG) {
            break;
        }
        line.clear();
    }

    let cli_output = Command::new(cli)
        .args(["--max-tokens", "4", "Say hello"])
        .env("BADAPPLE_SOCKET_PATH", &socket_path)
        .env("BADAPPLE_SLICKS_SECRET", TEST_SLICKS_SECRET)
        .output()
        .expect("failed to run badapple CLI");

    let pid = child.id() as i32;
    let socket_count = count_network_sockets(pid);

    let _ = child.kill();
    let _ = child.wait();
    let _ = std::fs::remove_file(&socket_path);

    assert!(
        cli_output.status.success(),
        "badapple CLI failed: {}",
        String::from_utf8_lossy(&cli_output.stderr)
    );
    assert!(
        !cli_output.stdout.is_empty(),
        "badapple CLI returned no text"
    );
    assert_eq!(
        socket_count, 0,
        "Bad Apple daemon must have zero network sockets; lsof -i reported {}",
        socket_count
    );
}

fn count_network_sockets(pid: i32) -> usize {
    let output = Command::new("lsof")
        .arg("-i")
        .arg("-a")
        .arg("-p")
        .arg(pid.to_string())
        .output()
        .expect("lsof must be available");

    if !output.status.success() {
        return 0;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    // `lsof -i` prints a header line; non-empty lines after the header are sockets.
    stdout
        .lines()
        .skip(1)
        .filter(|l| !l.trim().is_empty())
        .count()
}
