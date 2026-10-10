//! Stress tests for the generative red-team surface — hostile probe specs,
//! jail escapes, volume/eviction, and findings-chain tamper. Every check the
//! engine can run is reachable only through a validated spec, so the spec
//! boundary is the attack surface these tests attack.

use bad_apple::red_team::generated::{
    self, load_registry, persist_spec, run_probe_spec, ProbeSpec,
};
use bad_apple::red_team::propose;
use bad_apple::red_team::ProbeResult;
use bad_apple::red_team::Severity;
use std::path::PathBuf;
use std::sync::Mutex;

/// The probe dir is a process-global env var — parallel tests would race on
/// it, so every test that redirects it serializes through this lock.
/// `into_inner` on poison keeps one panicking test from cascading into the rest.
static ENV_LOCK: Mutex<()> = Mutex::new(());

fn env_lock() -> std::sync::MutexGuard<'static, ()> {
    ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

fn dir() -> PathBuf {
    let d = std::env::temp_dir()
        .canonicalize()
        .unwrap()
        .join(format!("ba_stress_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn spec(id: &str, checks: &str) -> ProbeSpec {
    serde_json::from_str(&format!(
        r#"{{"id":"{id}","title":"t","hypothesis":"h","severity":"low","checks":[{checks}]}}"#
    ))
    .unwrap()
}

fn result_for(id: &str, mitigated: bool) -> ProbeResult {
    ProbeResult {
        probe: id.to_string(),
        category: "generated".to_string(),
        mitigated,
        severity: Severity::Low,
        title: "t".to_string(),
        description: "d".to_string(),
        payload: None,
        timestamp_ms: 1,
        duration_us: 0,
    }
}

#[test]
fn planted_hostile_specs_never_load() {
    let _g = env_lock();
    std::env::set_var("BADAPPLE_RED_TEAM_DIR", dir());
    let hostile = [
        // out-of-jail reads
        r#"{"id":"gen.a1","title":"t","hypothesis":"h","severity":"low","checks":[{"kind":"file_contains","path":"/etc/master.passwd","pattern":"root"}]}"#,
        r#"{"id":"gen.a2","title":"t","hypothesis":"h","severity":"low","checks":[{"kind":"file_contains","path":"/var/lib/bad_apple/../../etc/passwd","pattern":"x"}]}"#,
        // non-loopback HTTP
        r#"{"id":"gen.a3","title":"t","hypothesis":"h","severity":"low","checks":[{"kind":"http_status","url":"http://example.com/","status":200}]}"#,
        r#"{"id":"gen.a4","title":"t","hypothesis":"h","severity":"low","checks":[{"kind":"http_status","url":"http://127.0.0.1.evil.com/","status":200}]}"#,
        r#"{"id":"gen.a5","title":"t","hypothesis":"h","severity":"low","checks":[{"kind":"http_status","url":"file:///etc/passwd","status":200}]}"#,
        // oversized frame / multiline frame
        r#"{"id":"gen.a6","title":"t","hypothesis":"h","severity":"low","checks":[{"kind":"socket_rejects","path":"/var/run/badapple/substrate.sock","frame":"one\ntwo"}]}"#,
        // zero checks
        r#"{"id":"gen.a7","title":"t","hypothesis":"h","severity":"low","checks":[]}"#,
        // bad id
        r#"{"id":"builtin.cage","title":"t","hypothesis":"h","severity":"low","checks":[{"kind":"file_absent","path":"/tmp/x_y_z"}]}"#,
        // unknown field smuggle
        r#"{"id":"gen.a9","title":"t","hypothesis":"h","severity":"low","checks":[{"kind":"file_absent","path":"/tmp/x_y_z"}],"exec":"rm -rf /"}"#,
    ];
    let probes_dir = generated::probes_dir();
    std::fs::create_dir_all(&probes_dir).unwrap();
    for (i, s) in hostile.iter().enumerate() {
        std::fs::write(probes_dir.join(format!("hostile_{i}.json")), s).unwrap();
    }
    let loaded = load_registry();
    assert!(
        loaded.is_empty(),
        "hostile specs must never reach the registry; {} did",
        loaded.len()
    );
}

#[test]
fn symlink_escape_is_rejected() {
    let _g = env_lock();
    std::env::set_var("BADAPPLE_RED_TEAM_DIR", dir());
    let link = PathBuf::from("/tmp/ba_stress_link");
    let _ = std::fs::remove_file(&link);
    #[cfg(unix)]
    std::os::unix::fs::symlink("/etc/master.passwd", &link).unwrap();
    let s = spec(
        "gen.slink",
        r#"{"kind":"file_contains","path":"/tmp/ba_stress_link","pattern":"root"}"#,
    );
    assert!(
        generated::validate_spec(&s).is_err(),
        "symlink resolving out of jail must be rejected"
    );
    let _ = std::fs::remove_file(&link);
}

#[test]
fn registry_is_bounded_under_volume() {
    let _g = env_lock();
    let d = dir();
    std::env::set_var("BADAPPLE_RED_TEAM_DIR", &d);
    // Write more specs than the cap allows.
    for i in 0..70 {
        let s = spec(
            &format!("gen.vol_{i:03}"),
            r#"{"kind":"file_absent","path":"/tmp/definitely_absent_xyz"}"#,
        );
        persist_spec(&s).unwrap();
    }
    let loaded = load_registry();
    assert!(
        loaded.len() <= generated::MAX_GENERATED_PROBES,
        "registry must stay bounded, got {}",
        loaded.len()
    );
    let files = std::fs::read_dir(generated::probes_dir()).unwrap().count();
    assert!(files <= generated::MAX_GENERATED_PROBES);
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn findings_chain_detects_rewrite() {
    let _g = env_lock();
    let d = dir();
    std::env::set_var("BADAPPLE_RED_TEAM_DIR", &d);
    let path = generated::findings_log_path();
    propose::record_result(&result_for("gen.one", true)).unwrap();
    propose::record_result(&result_for("gen.two", false)).unwrap();
    propose::record_result(&result_for("gen.three", true)).unwrap();
    assert_eq!(propose::verify_findings_log(&path).unwrap(), 3);

    // Rewrite a field inside a middle record — keep `lines` pristine for the
    // truncation/reorder cases below.
    let lines: Vec<String> = std::fs::read_to_string(&path)
        .unwrap()
        .lines()
        .map(String::from)
        .collect();
    let mut edited = lines.clone();
    edited[1] = edited[1].replace("\"mitigated\":false", "\"mitigated\":true");
    let tampered = d.join("tampered.jsonl");
    std::fs::write(&tampered, edited.join("\n") + "\n").unwrap();
    assert!(propose::verify_findings_log(&tampered).is_err());

    // Truncate the tail — the surviving prefix still chains.
    let trunc = d.join("trunc.jsonl");
    std::fs::write(&trunc, lines[..2].join("\n") + "\n").unwrap();
    assert_eq!(propose::verify_findings_log(&trunc).unwrap(), 2);
    // But re-ordering records breaks the next prev_hash.
    let mut swapped = lines.clone();
    swapped.swap(1, 2);
    let reord = d.join("reord.jsonl");
    std::fs::write(&reord, swapped.join("\n") + "\n").unwrap();
    assert!(propose::verify_findings_log(&reord).is_err());
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn executed_checks_fail_closed_on_jail_violations() {
    // A spec that validates cleanly but whose check target is moved out of
    // jail between validation and execution must still fail, not pass.
    let s = spec(
        "gen.late_jail",
        r#"{"kind":"file_present","path":"/var/lib/bad_apple/does_not_exist_xyz"}"#,
    );
    generated::validate_spec(&s).unwrap(); // parent dir resolves; file absent is fine
    let result = run_probe_spec(s);
    assert!(!result.mitigated, "absent required file must be a finding");
}

#[test]
fn oversized_scan_target_fails_closed() {
    // A file over the 16 MiB scan cap must error, never silently pass — most
    // dangerous on file_not_contains, where "pass" would mean skipping a
    // secret-scan on an unreadable ledger.
    let big = PathBuf::from("/tmp/ba_stress_big.bin");
    std::fs::write(&big, vec![b'x'; 17 * 1024 * 1024]).unwrap();
    let s = spec(
        "gen.bigfile",
        r#"{"kind":"file_not_contains","path":"/tmp/ba_stress_big.bin","pattern":"SECRET"}"#,
    );
    generated::validate_spec(&s).unwrap();
    let result = run_probe_spec(s);
    assert!(
        !result.mitigated,
        "over-cap file must fail closed, not pass: {:?}",
        result
    );
    let _ = std::fs::remove_file(&big);
}

#[test]
fn spec_with_all_check_kinds_runs() {
    let _g = env_lock();
    std::env::set_var("BADAPPLE_RED_TEAM_DIR", dir());
    let s = spec(
        "gen.everything",
        r#"{"kind":"file_absent","path":"/tmp/absent_xyz"},
           {"kind":"file_present","path":"/tmp"},
           {"kind":"port_closed","addr":"127.0.0.1:1"},
           {"kind":"process_absent","name":"definitely_not_running_xyz"}"#,
    );
    generated::validate_spec(&s).unwrap();
    let result = run_probe_spec(s);
    assert!(
        result.mitigated,
        "benign checks should all pass: {:?}",
        result
    );
}
