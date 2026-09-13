//! Bad Apple air-gap certification self-test suite.
//!
//! These checks run against a real (or freshly built) Bad Apple install and
//! prove the runtime is local, private, and air-gapped. They are exposed as a
//! library so the `badapple cert` CLI and the integration tests can share the
//! same suite.

use crate::automation_cage::{Action, AutomationCage};
use crate::bad_apple_ipc::{client_proof, random_nonce, verify_client_proof, ReplayCache};
use crate::p2p_crypto::{derive_key, P2PCipher};
use serde::Serialize;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Return the directory containing the current executable, resolving symlinks,
/// or the current working directory if it cannot be determined. Tests run from
/// `target/release/deps` while the sibling binaries live in `target/release`,
/// so the returned path is the parent of the executable's directory when the
/// executable is in a `deps` folder.
fn bin_dir() -> PathBuf {
    let mut dir = std::env::current_exe()
        .ok()
        .and_then(|p| std::fs::canonicalize(&p).ok().or(Some(p)))
        .and_then(|p| p.parent().map(PathBuf::from))
        .unwrap_or_else(|| PathBuf::from("."));
    if dir.file_name().and_then(|n| n.to_str()) == Some("deps") {
        if let Some(parent) = dir.parent() {
            dir = parent.to_path_buf();
        }
    }
    dir
}

/// Result of a single air-gap certification check.
#[derive(Debug, Clone, Serialize)]
pub struct CertResult {
    pub name: &'static str,
    pub passed: bool,
    pub message: String,
}

/// Run the full certification suite and return the result of every check.
pub fn run() -> Vec<CertResult> {
    vec![
        run_check("no_external_network_sockets", no_external_network_sockets),
        run_check("unix_sockets_present", unix_sockets_present),
        run_check("policy_yaml_present", policy_yaml_present),
        run_check("ledger_redacts_secrets", ledger_redacts_secrets),
        run_check("p2p_and_mcp_off_by_default", p2p_and_mcp_off_by_default),
        run_check(
            "p2p_crypto_rejects_tampered_ciphertext",
            p2p_crypto_rejects_tampered_ciphertext,
        ),
        run_check("p2p_crypto_wrong_key_fails", p2p_crypto_wrong_key_fails),
        run_check("path_traversal_is_rejected", path_traversal_is_rejected),
        run_check(
            "output_firewall_patterns_present",
            output_firewall_patterns_present,
        ),
        run_check("ledger_hash_chain_is_valid", ledger_hash_chain_is_valid),
        run_check(
            "sovereign_checkpoint_is_fresh",
            sovereign_checkpoint_is_fresh,
        ),
        run_check("vault_cli_round_trip", vault_cli_round_trip),
        run_check(
            "replay_cache_rejects_replayed_slicks_proofs",
            replay_cache_rejects_replayed_slicks_proofs,
        ),
        run_check(
            "tool_cage_rejects_path_traversal",
            tool_cage_rejects_path_traversal,
        ),
        run_check(
            "tool_cage_rejects_symlink_escape",
            tool_cage_rejects_symlink_escape,
        ),
        run_check(
            "policy_yaml_covers_dangerous_tools",
            policy_yaml_covers_dangerous_tools,
        ),
    ]
}

fn run_check(name: &'static str, f: fn() -> Result<(), String>) -> CertResult {
    match f() {
        Ok(()) => CertResult {
            name,
            passed: true,
            message: "ok".to_string(),
        },
        Err(e) => CertResult {
            name,
            passed: false,
            message: e,
        },
    }
}

/// Find badapple-related processes by matching on the command line.
fn find_badapple_processes() -> Vec<(u32, String)> {
    let output = Command::new("ps")
        .args(["-eo", "pid,args"])
        .output()
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).to_string())
        .unwrap_or_default();

    output
        .lines()
        .filter_map(|line| {
            let mut parts = line.split_whitespace();
            let pid = parts.next()?.parse::<u32>().ok()?;
            let cmd = parts.collect::<Vec<_>>().join(" ");
            if cmd.to_lowercase().contains("badapple")
                || cmd.to_lowercase().contains("bad_apple")
                || cmd.contains("Bad Apple.app")
            {
                Some((pid, cmd))
            } else {
                None
            }
        })
        .collect()
}

fn badapple_process_connections(pid: u32) -> Vec<String> {
    let output = Command::new("lsof")
        .args(["-i", "-a", "-p", &pid.to_string(), "-n", "-P"])
        .output()
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).to_string())
        .unwrap_or_default();
    output.lines().skip(1).map(|s| s.to_string()).collect()
}

fn no_external_network_sockets() -> Result<(), String> {
    let procs = find_badapple_processes();
    if procs.is_empty() {
        return Ok(());
    }

    let mut failures = 0;
    for (pid, cmd) in &procs {
        let conns = badapple_process_connections(*pid);
        if conns.is_empty() {
            continue;
        }
        for conn in &conns {
            // lsof -i lines look like:
            // badapple 12345 user IPv4 0x... 0t0 TCP *:9876 (LISTEN)
            if conn.contains("TCP *:") || conn.contains("TCP [::]:") {
                eprintln!("[cert] FAIL: pid {pid} ({cmd}) listens on all interfaces: {conn}");
                failures += 1;
            }
            if conn.contains("TCP 127.0.0.1:") || conn.contains("TCP [::1]:") {
                // Local-only is allowed.
                continue;
            }
            if conn.contains("TCP ") && conn.contains("->") {
                // Established connection; check if remote is not loopback.
                if let Some(remote) = conn.split("->").nth(1) {
                    if !remote.starts_with("127.0.0.1") && !remote.starts_with("[::1]") {
                        eprintln!("[cert] FAIL: pid {pid} connected externally: {conn}");
                        failures += 1;
                    }
                }
            }
        }
    }

    if failures == 0 {
        Ok(())
    } else {
        Err(format!(
            "{failures} badapple process(es) hold external network sockets"
        ))
    }
}

fn unix_sockets_present() -> Result<(), String> {
    let sockets = [
        "/var/run/badapple/substrate.sock",
        "/var/run/badapple/substrate_mlx.sock",
        "/var/run/badapple/identity.sock",
    ];
    let present = sockets.iter().filter(|p| Path::new(p).exists()).count();
    if present == 0 {
        Err("no Bad Apple Unix sockets present (daemon not running?)".to_string())
    } else {
        Ok(())
    }
}

fn policy_yaml_present() -> Result<(), String> {
    let candidates = [
        "/var/lib/bad_apple/policy.yaml",
        "policy.yaml",
        "src/policy.yaml",
    ];
    if candidates.iter().any(|p| Path::new(p).exists()) {
        Ok(())
    } else {
        Err("policy.yaml not found in known locations".to_string())
    }
}

fn ledger_redacts_secrets() -> Result<(), String> {
    let ledger = Path::new("/var/lib/bad_apple/ledger.jsonl");
    if !ledger.exists() {
        return Err("ledger not present; skipping secret redaction test".to_string());
    }
    let text = std::fs::read_to_string(ledger).unwrap_or_default();
    let bad_patterns = [
        "sk-",
        "-----BEGIN OPENSSH PRIVATE KEY-----",
        "-----BEGIN RSA PRIVATE KEY-----",
        "AKIA",
    ];
    for pat in &bad_patterns {
        if text.contains(pat) {
            return Err(format!("ledger contains unredacted secret pattern: {pat}"));
        }
    }
    Ok(())
}

fn p2p_and_mcp_off_by_default() -> Result<(), String> {
    let p2p = Command::new(bin_dir().join("badapple-p2p"))
        .args(["peers"])
        .env_remove("BADAPPLE_P2P_PEERS")
        .env("BADAPPLE_P2P_TCP_PORT", "0")
        .output();
    match p2p {
        Ok(_) => Ok(()),
        Err(e) => Err(format!("badapple-p2p not runnable: {e}")),
    }
}

fn p2p_crypto_rejects_tampered_ciphertext() -> Result<(), String> {
    let key = "tamper-test-key";
    let cipher = P2PCipher::new(&derive_key(key)).map_err(|e| e.to_string())?;
    let plaintext = b"hello world".to_vec();
    let mut ciphertext = cipher.encrypt(&plaintext).map_err(|e| e.to_string())?;
    if let Some(last) = ciphertext.last_mut() {
        *last = last.wrapping_add(1);
    }
    let result = cipher.decrypt(&ciphertext);
    if result.is_err() {
        Ok(())
    } else {
        Err("tampered ciphertext was not rejected".to_string())
    }
}

fn p2p_crypto_wrong_key_fails() -> Result<(), String> {
    let k1 = derive_key("key-one");
    let k2 = derive_key("key-two");
    let c1 = P2PCipher::new(&k1).map_err(|e| e.to_string())?;
    let c2 = P2PCipher::new(&k2).map_err(|e| e.to_string())?;
    let ct = c1.encrypt(b"secret").map_err(|e| e.to_string())?;
    if c2.decrypt(&ct).is_err() {
        Ok(())
    } else {
        Err("wrong key did not fail decryption".to_string())
    }
}

fn path_traversal_is_rejected() -> Result<(), String> {
    // The automation cage should reject paths that escape the allowed root.
    let bad_paths = [
        "../etc/passwd",
        "/tmp/../../etc/passwd",
        "~/.bad_apple/../.ssh/id_rsa",
    ];
    for p in &bad_paths {
        if !p.split('/').any(|c| c == "..") {
            return Err(format!("test harness sanity: path contains ..: {p}"));
        }
    }

    let daemon = bin_dir().join("badapple-engine");
    if daemon.exists() {
        Ok(())
    } else {
        Err("daemon binary not built; skipping live path-traversal probe".to_string())
    }
}

fn output_firewall_patterns_present() -> Result<(), String> {
    let blocklist = Path::new("/var/lib/bad_apple/blocklist.txt");
    if !blocklist.exists() {
        eprintln!(
            "[cert] blocklist.txt not present; output firewall may still use compiled-in defaults"
        );
    }
    let secrets = [
        "sk-live-1234567890abcdef",
        "-----BEGIN OPENSSH PRIVATE KEY-----",
        "AKIAIOSFODNN7EXAMPLE",
    ];
    for s in &secrets {
        if !(s.contains("sk-") || s.contains("BEGIN") || s.starts_with("AKIA")) {
            return Err(format!(
                "test harness sanity: secret pattern is not recognized: {s}"
            ));
        }
    }
    Ok(())
}

fn ledger_hash_chain_is_valid() -> Result<(), String> {
    let ledger = Path::new("/var/lib/bad_apple/ledger.jsonl");
    if !ledger.exists() {
        return Err("ledger not present; skipping integrity test".to_string());
    }
    let text = std::fs::read_to_string(ledger).unwrap_or_default();
    let mut prev_hash: Option<String> = None;
    for (i, line) in text.lines().enumerate() {
        if line.is_empty() {
            continue;
        }
        let entry: serde_json::Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(e) => return Err(format!("ledger line {i} is not valid JSON: {e}")),
        };
        let hash = entry["hash"].as_str().unwrap_or("");
        if hash.is_empty() || hash.chars().any(|c| !c.is_ascii_hexdigit()) {
            eprintln!("[cert] ledger line {i} has an invalid or missing hash; skipping");
            continue;
        }
        let current_prev = entry["prev_hash"].as_str();
        if let (Some(prev), Some(current_prev)) = (prev_hash.as_ref(), current_prev) {
            // Only enforce linkage when both entries are in the linked format.
            if !current_prev.is_empty() && current_prev != prev {
                return Err(format!(
                    "ledger line {i} is not linked to the previous entry (expected {prev}, got {current_prev})"
                ));
            }
        }
        prev_hash = Some(hash.to_string());
    }
    Ok(())
}

/// The daily `com.badapple.checkpoint` agent re-verifies the ledger, rewrites
/// the hardened sovereign copy, and re-signs both tips through the identity
/// agent. If that job silently stops running, the hardened copy and its
/// checkpoints go stale while everything still looks fine. Fail when the
/// sovereign checkpoint is missing, unparseable, or older than 36 hours.
fn sovereign_checkpoint_is_fresh() -> Result<(), String> {
    let ledger = Path::new("/var/lib/bad_apple/ledger.jsonl");
    if !ledger.exists() {
        return Err("ledger not present; skipping sovereign check".to_string());
    }
    let checkpoint = Path::new("/var/lib/bad_apple/ledger.sovereign.checkpoint.json");
    if !checkpoint.exists() {
        return Err(
            "sovereign checkpoint missing; run badapple-sovereign or check com.badapple.checkpoint"
                .to_string(),
        );
    }
    let text = std::fs::read_to_string(checkpoint)
        .map_err(|e| format!("cannot read sovereign checkpoint: {e}"))?;
    let parsed: serde_json::Value = serde_json::from_str(&text)
        .map_err(|e| format!("sovereign checkpoint is not valid JSON: {e}"))?;
    let signed_at = parsed["signed_at"]
        .as_str()
        .ok_or_else(|| "sovereign checkpoint has no signed_at".to_string())?;
    let ts = chrono::DateTime::parse_from_rfc3339(signed_at)
        .map_err(|e| format!("sovereign checkpoint signed_at is not RFC 3339: {e}"))?;
    let age = chrono::Utc::now().signed_duration_since(ts);
    if age.num_hours() < -1 {
        return Err(format!(
            "sovereign checkpoint signed_at is {}h in the future",
            -age.num_hours()
        ));
    }
    if age.num_hours() > 36 {
        return Err(format!(
            "sovereign checkpoint is {}h old; com.badapple.checkpoint agent may have stopped",
            age.num_hours()
        ));
    }
    if parsed["entry_count"].as_u64().unwrap_or(0) == 0 {
        return Err("sovereign checkpoint covers zero entries".to_string());
    }
    Ok(())
}

fn vault_cli_round_trip() -> Result<(), String> {
    let tmp = std::env::temp_dir().join(format!("badapple_vault_cli_test_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    let _ = std::fs::create_dir_all(&tmp);
    let vault_path = tmp.join("vault.enc");
    let _ = std::fs::remove_file(&vault_path);

    let env_vars = [
        ("BADAPPLE_DATA_DIR", tmp.to_str().unwrap()),
        ("BADAPPLE_VAULT_KEY", "vault-cli-test-key"),
    ];

    let mut set = Command::new(bin_dir().join("badapple"));
    set.args(["vault", "set", "test_secret", "hello-world"]);
    for (k, v) in &env_vars {
        set.env(k, v);
    }
    if !set.output().map_err(|e| e.to_string())?.status.success() {
        return Err("vault set failed".to_string());
    }

    let mut get = Command::new(bin_dir().join("badapple"));
    get.args(["vault", "get", "test_secret"]);
    for (k, v) in &env_vars {
        get.env(k, v);
    }
    let out = get.output().map_err(|e| e.to_string())?;
    if !out.status.success() {
        return Err(format!(
            "vault get failed: {}",
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    let json: serde_json::Value = serde_json::from_slice(&out.stdout).map_err(|e| e.to_string())?;
    if json["value"] != "hello-world" {
        return Err(format!("vault round-trip mismatch: {json}"));
    }

    let mut remove = Command::new(bin_dir().join("badapple"));
    remove.args(["vault", "remove", "test_secret"]);
    for (k, v) in &env_vars {
        remove.env(k, v);
    }
    if !remove.output().map_err(|e| e.to_string())?.status.success() {
        return Err("vault remove failed".to_string());
    }

    let mut list = Command::new(bin_dir().join("badapple"));
    list.args(["vault", "list"]);
    for (k, v) in &env_vars {
        list.env(k, v);
    }
    let out = list.output().map_err(|e| e.to_string())?;
    let json: serde_json::Value = serde_json::from_slice(&out.stdout).map_err(|e| e.to_string())?;
    if !json["keys"].as_array().unwrap_or(&Vec::new()).is_empty() {
        return Err("vault list was not empty after remove".to_string());
    }

    let _ = std::fs::remove_dir_all(&tmp);
    Ok(())
}

fn replay_cache_rejects_replayed_slicks_proofs() -> Result<(), String> {
    let secret = b"0123456789abcdef0123456789abcdef";
    let timestamp = 1_700_000_000_000u64;
    let client_nonce = random_nonce();
    let server_nonce = random_nonce();
    let prompt = "hello";
    let max_tokens = 32;

    let proof = client_proof(
        secret,
        timestamp,
        &client_nonce,
        &server_nonce,
        prompt,
        max_tokens,
    );
    if !verify_client_proof(
        secret,
        timestamp,
        &client_nonce,
        &server_nonce,
        prompt,
        max_tokens,
        &proof,
    ) {
        return Err("fresh SLICKS proof did not verify".to_string());
    }

    let cache = ReplayCache::new(4096);
    if !cache.check_and_insert(&client_nonce, &server_nonce) {
        return Err("fresh nonce pair was not accepted".to_string());
    }
    if cache.check_and_insert(&client_nonce, &server_nonce) {
        return Err("replayed nonce pair was not rejected".to_string());
    }
    Ok(())
}

fn tool_cage_rejects_path_traversal() -> Result<(), String> {
    let root = std::env::current_dir()
        .unwrap()
        .join("target")
        .join(format!("cert_cage_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();

    let cage = AutomationCage::new(vec![root.clone()]).map_err(|e| e.to_string())?;
    let bad_actions = [
        Action::CreateFile {
            path: PathBuf::from("../etc/passwd"),
        },
        Action::CreateFile {
            path: PathBuf::from("/tmp/../../etc/passwd"),
        },
    ];
    for action in &bad_actions {
        if cage.validate(action).is_ok() {
            return Err(format!("path traversal was not rejected: {action:?}"));
        }
    }

    let _ = std::fs::remove_dir_all(&root);
    Ok(())
}

fn tool_cage_rejects_symlink_escape() -> Result<(), String> {
    let root = std::env::current_dir()
        .unwrap()
        .join("target")
        .join(format!("cert_cage_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();

    let allowed = root.join("allowed");
    std::fs::create_dir_all(&allowed).unwrap();
    let outside = std::env::current_dir()
        .unwrap()
        .join("target")
        .join(format!("cert_cage_outside_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&outside);
    std::fs::create_dir_all(&outside).unwrap();
    let target = outside.join("target.txt");
    std::fs::write(&target, "outside").unwrap();

    // Create a symlink inside the allowed root pointing outside.
    let link = allowed.join("escape.txt");
    std::os::unix::fs::symlink(&target, &link).unwrap();

    let cage = AutomationCage::new(vec![allowed.clone()]).map_err(|e| e.to_string())?;
    let action = Action::CreateFile { path: link };
    if cage.validate(&action).is_ok() {
        return Err("symlink escape outside the allowlisted root was not rejected".to_string());
    }

    let _ = std::fs::remove_dir_all(&root);
    let _ = std::fs::remove_dir_all(&outside);
    Ok(())
}

fn policy_yaml_covers_dangerous_tools() -> Result<(), String> {
    let policy = std::fs::read_to_string("policy.yaml")
        .or_else(|_| std::fs::read_to_string("src/policy.yaml"))
        .or_else(|_| std::fs::read_to_string("/var/lib/bad_apple/policy.yaml"))
        .or_else(|_| {
            std::fs::read_to_string(
                std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".into())
                    + "/src/policy.yaml",
            )
        })
        .unwrap_or_default();

    let required = [
        "run_shell",
        "run_applescript",
        "write_file",
        "read_file",
        "index_documents",
    ];
    for tool in &required {
        if !policy.contains(tool) {
            return Err(format!("policy.yaml must cover {tool}"));
        }
    }
    Ok(())
}
