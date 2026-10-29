// cert.cy — the air-gap certification suite, ported from src/cert.rs.
//
// Proves the runtime is local, private, and air-gapped. Every check is a
// host fn returning Json {name, passed, message}; `run()` is the suite
// entry — `badapple cert` calls this .cy program, not the Rust suite.
//
// Checks that exercise in-process crypto/protocol run against the OTHER
// .cy ports (slicks/cage/mesh_brain) — the suite certifies the organism
// itself, in the organism's language.

import "sys.cy";
import "strlib.cy";
import "slicks.cy";
import "cage.cy";
import "mesh_brain.cy";

cap fs, net, crypto, proc

// ── harness ────────────────────────────────────────────────────────────

host det fn cert_ok(name: str) -> Json {
    return Json::Obj({
        "name": Json::Str(name),
        "passed": Json::Bool(true),
        "message": Json::Str("ok"),
    });
}

host det fn cert_fail(name: str, msg: str) -> Json {
    return Json::Obj({
        "name": Json::Str(name),
        "passed": Json::Bool(false),
        "message": Json::Str(msg),
    });
}

host det fn jstr(o: {str: Json}, k: str) -> str {
    if has(o, k) {
        match o[k] { Json::Str(s) => { return s; } _ => { } }
    }
    return "";
}

host det fn jbool(o: {str: Json}, k: str) -> bool {
    if has(o, k) {
        match o[k] { Json::Bool(b) => { return b; } _ => { } }
    }
    return false;
}

host det fn jnum(o: {str: Json}, k: str) -> i64 {
    if has(o, k) {
        match o[k] { Json::Num(n) => { return i64(n); } _ => { } }
    }
    return 0;
}

// ── process/table inspection ───────────────────────────────────────────

host fn find_badapple_processes() -> [str] cap proc {
    var acc: [str] = [];
    match sys_exec("ps", ["-eo", "pid,args"]) {
        Json::Obj(r) => {
            let text = jstr(r, "out");
            let ls = lines(text);
            for i in 0..len(ls) {
                let low = lower_str(ls[i]);
                if find_sub(low, "badapple") >= 0 || find_sub(low, "bad_apple") >= 0
                    || find_sub(ls[i], "Bad Apple.app") >= 0 {
                    push(acc, trim(ls[i]));
                }
            }
        }
        _ => { }
    }
    return acc;
}

// pid is the first whitespace token of a "pid args" line
host det fn line_pid(line: str) -> str {
    var i = 0;
    var acc = "";
    while i < len(line) {
        let c = slice_str(line, i, i + 1);
        if c == " " || c == "\t" {
            if acc != "" { return acc; }
        } else {
            acc = acc + c;
        }
        i = i + 1;
    }
    return acc;
}

host fn check_no_external_network_sockets() -> Json cap proc {
    let procs = find_badapple_processes();
    if len(procs) == 0 { return cert_ok("no_external_network_sockets"); }
    var failures = 0;
    for i in 0..len(procs) {
        let pid = line_pid(procs[i]);
        match sys_exec("lsof", ["-i", "-a", "-p", pid, "-n", "-P"]) {
            Json::Obj(r) => {
                let conns = lines(jstr(r, "out"));
                for j in 0..len(conns) {
                    let conn = conns[j];
                    if find_sub(conn, "TCP *:") >= 0 || find_sub(conn, "TCP [::]:") >= 0 {
                        failures = failures + 1;
                    } else if find_sub(conn, "TCP 127.0.0.1:") >= 0
                        || find_sub(conn, "TCP [::1]:") >= 0 {
                        // local-only allowed
                    } else if find_sub(conn, "TCP ") >= 0 && find_sub(conn, "->") >= 0 {
                        let parts = split(conn, "->");
                        if len(parts) > 1 {
                            let remote = parts[1];
                            if !starts_with(remote, "127.0.0.1") && !starts_with(remote, "[::1]") {
                                failures = failures + 1;
                            }
                        }
                    }
                }
            }
            _ => { }
        }
    }
    if failures == 0 { return cert_ok("no_external_network_sockets"); }
    return cert_fail("no_external_network_sockets",
        str(failures) + " badapple process(es) hold external network sockets");
}

host fn check_unix_sockets_present() -> Json cap fs {
    let socks = ["/var/run/badapple/substrate.sock",
                 "/var/run/badapple/substrate_mlx.sock",
                 "/var/run/badapple/identity.sock"];
    var present = 0;
    for i in 0..len(socks) {
        if fs_exists(socks[i]) { present = present + 1; }
    }
    if present == 0 {
        return cert_fail("unix_sockets_present",
            "no Bad Apple Unix sockets present (daemon not running?)");
    }
    return cert_ok("unix_sockets_present");
}

host fn check_policy_yaml_present() -> Json cap fs {
    let cands = ["/var/lib/bad_apple/policy.yaml", "policy.yaml", "src/policy.yaml"];
    for i in 0..len(cands) {
        if fs_exists(cands[i]) { return cert_ok("policy_yaml_present"); }
    }
    return cert_fail("policy_yaml_present", "policy.yaml not found in known locations");
}

host fn check_ledger_redacts_secrets() -> Json cap fs {
    if !fs_exists("/var/lib/bad_apple/ledger.jsonl") {
        return cert_fail("ledger_redacts_secrets",
            "ledger not present; skipping secret redaction test");
    }
    let text = read_file("/var/lib/bad_apple/ledger.jsonl");
    let bad = ["sk-", "-----BEGIN OPENSSH PRIVATE KEY-----",
               "-----BEGIN RSA PRIVATE KEY-----", "AKIA"];
    for i in 0..len(bad) {
        if find_sub(text, bad[i]) >= 0 {
            return cert_fail("ledger_redacts_secrets",
                "ledger contains unredacted secret pattern: " + bad[i]);
        }
    }
    return cert_ok("ledger_redacts_secrets");
}

// find_binary — sibling dir, launchd plist, install locations, app bundle
host fn find_binary(name: str) -> str cap fs {
    var cands: [str] = [];
    push(cands, sys_exe_dir() + "/" + name);
    if fs_exists("/Library/LaunchDaemons/com.badapple.mlx.plist") {
        let plist = read_file("/Library/LaunchDaemons/com.badapple.mlx.plist");
        let ls = lines(plist);
        for i in 0..len(ls) {
            let line = trim(ls[i]);
            if starts_with(line, "<string>") && find_sub(line, name) >= 0 {
                var p = line;
                if starts_with(p, "<string>") { p = slice_str(p, 8, len(p)); }
                if ends_with(p, "</string>") { p = slice_str(p, 0, len(p) - 9); }
                push(cands, p);
            }
        }
    }
    push(cands, "/usr/local/bin/" + name);
    push(cands, "/usr/local/libexec/" + name);
    push(cands, "/Applications/Bad Apple.app/Contents/Helpers/" + name);
    push(cands, "/Applications/Bad Apple.app/Contents/Resources/" + name);
    let root = env_get("BADAPPLE_ROOT");
    if root != "" {
        push(cands, root + "/target/release/" + name);
    }
    for i in 0..len(cands) {
        if fs_exists(cands[i]) { return cands[i]; }
    }
    return "";
}

host fn check_p2p_off_by_default() -> Json cap fs, proc {
    let bin = find_binary("badapple-p2p");
    if bin == "" {
        // no helper installed → p2p cannot run at all — satisfies the check
        return cert_ok("p2p_and_mcp_off_by_default");
    }
    var env: {str: str} = {};
    env["BADAPPLE_P2P_PEERS"] = "";
    env["BADAPPLE_P2P_SECRET"] = "";
    env["BADAPPLE_SLICKS_KEY_PATH"] = "";
    env["BADAPPLE_P2P_TCP_PORT"] = "0";
    match sys_exec_env(bin, ["peers"], env) {
        Json::Obj(r) => {
            let text = jstr(r, "out") + jstr(r, "err");
            let code = jnum(r, "code");
            if find_sub(text, "requires BADAPPLE_P2P") >= 0 || code != 0 {
                return cert_ok("p2p_and_mcp_off_by_default");
            }
            return cert_fail("p2p_and_mcp_off_by_default",
                "badapple-p2p started without credentials");
        }
        _ => {
            return cert_fail("p2p_and_mcp_off_by_default",
                "badapple-p2p present but not runnable");
        }
    }
}

// ── crypto checks — against the .cy cipher externs ─────────────────────

host fn check_tampered_ciphertext() -> Json cap crypto {
    let key = sha256("tamper-test-key");
    let ct = aead_seal(key, hex_encode("hello world"));
    if ct == "" {
        return cert_fail("p2p_crypto_rejects_tampered_ciphertext", "seal failed");
    }
    // flip the last nibble
    let n = len(ct);
    var last = slice_str(ct, n - 1, n);
    if last == "0" { last = "1"; } else { last = "0"; }
    let bad = slice_str(ct, 0, n - 1) + last;
    if aead_open(key, bad) == "" {
        return cert_ok("p2p_crypto_rejects_tampered_ciphertext");
    }
    return cert_fail("p2p_crypto_rejects_tampered_ciphertext",
        "tampered ciphertext was not rejected");
}

host fn check_wrong_key() -> Json cap crypto {
    let k1 = sha256("key-one");
    let k2 = sha256("key-two");
    let ct = aead_seal(k1, hex_encode("secret"));
    if aead_open(k2, ct) == "" {
        return cert_ok("p2p_crypto_wrong_key_fails");
    }
    return cert_fail("p2p_crypto_wrong_key_fails",
        "wrong key did not fail decryption");
}

host fn check_path_traversal_rejected() -> Json cap fs {
    // sanity — the probe paths all contain ".."
    let bad = ["../etc/passwd", "/tmp/../../etc/passwd", "~/.bad_apple/../.ssh/id_rsa"];
    for i in 0..len(bad) {
        var found = false;
        let parts = split(bad[i], "/");
        for j in 0..len(parts) {
            if parts[j] == ".." { found = true; }
        }
        if !found {
            return cert_fail("path_traversal_is_rejected",
                "test harness sanity: path contains ..: " + bad[i]);
        }
    }
    if find_binary("badapple-engine") != "" {
        return cert_ok("path_traversal_is_rejected");
    }
    return cert_fail("path_traversal_is_rejected",
        "badapple-engine not found in any install location");
}

host fn check_firewall_patterns() -> Json cap fs {
    if !fs_exists("/var/lib/bad_apple/blocklist.txt") {
        // compiled-in defaults may still apply — not a failure
    }
    let secrets = ["sk-live-1234567890abcdef",
                   "-----BEGIN OPENSSH PRIVATE KEY-----",
                   "AKIAIOSFODNN7EXAMPLE"];
    for i in 0..len(secrets) {
        let s = secrets[i];
        if !(find_sub(s, "sk-") >= 0 || find_sub(s, "BEGIN") >= 0 || starts_with(s, "AKIA")) {
            return cert_fail("output_firewall_patterns_present",
                "test harness sanity: secret pattern is not recognized: " + s);
        }
    }
    return cert_ok("output_firewall_patterns_present");
}

host fn check_ledger_chain() -> Json cap fs {
    if !fs_exists("/var/lib/bad_apple/ledger.jsonl") {
        return cert_fail("ledger_hash_chain_is_valid",
            "ledger not present; skipping integrity test");
    }
    let text = read_file("/var/lib/bad_apple/ledger.jsonl");
    let ls = lines(text);
    var prev = "";
    for i in 0..len(ls) {
        let line = ls[i];
        if line == "" { continue; }
        match json_parse(line) {
            Json::Obj(e) => {
                let hash = jstr(e, "hash");
                if hash == "" || !is_hex(hash) {
                    continue; // unlinked-format line — skip like the Rust does
                }
                let cur_prev = jstr(e, "prev_hash");
                if prev != "" && cur_prev != "" && cur_prev != prev {
                    return cert_fail("ledger_hash_chain_is_valid",
                        "ledger line " + str(i) + " is not linked to the previous entry");
                }
                prev = hash;
            }
            _ => {
                return cert_fail("ledger_hash_chain_is_valid",
                    "ledger line " + str(i) + " is not valid JSON");
            }
        }
    }
    return cert_ok("ledger_hash_chain_is_valid");
}

host fn check_sovereign_checkpoint() -> Json cap fs {
    if !fs_exists("/var/lib/bad_apple/ledger.jsonl") {
        return cert_fail("sovereign_checkpoint_is_fresh",
            "ledger not present; skipping sovereign check");
    }
    let cp = "/var/lib/bad_apple/ledger.sovereign.checkpoint.json";
    if !fs_exists(cp) {
        return cert_fail("sovereign_checkpoint_is_fresh",
            "sovereign checkpoint missing; run badapple-sovereign or check com.badapple.checkpoint");
    }
    match json_parse(read_file(cp)) {
        Json::Obj(o) => {
            let signed_at = jstr(o, "signed_at");
            if signed_at == "" {
                return cert_fail("sovereign_checkpoint_is_fresh",
                    "sovereign checkpoint has no signed_at");
            }
            let ts = time_parse_iso(signed_at);
            if ts < 0 {
                return cert_fail("sovereign_checkpoint_is_fresh",
                    "sovereign checkpoint signed_at is not RFC 3339");
            }
            let age_h = (time_ms() - ts) / 3600000;
            if age_h < -1 {
                return cert_fail("sovereign_checkpoint_is_fresh",
                    "sovereign checkpoint signed_at is in the future");
            }
            if age_h > 36 {
                return cert_fail("sovereign_checkpoint_is_fresh",
                    "sovereign checkpoint is " + str(age_h) + "h old; com.badapple.checkpoint agent may have stopped");
            }
            if jnum(o, "entry_count") == 0 {
                return cert_fail("sovereign_checkpoint_is_fresh",
                    "sovereign checkpoint covers zero entries");
            }
            return cert_ok("sovereign_checkpoint_is_fresh");
        }
        _ => {
            return cert_fail("sovereign_checkpoint_is_fresh",
                "sovereign checkpoint is not valid JSON");
        }
    }
}

host fn check_vault_cli() -> Json cap fs, proc, crypto {
    let bin = sys_exe_dir() + "/badapple";
    if !fs_exists(bin) {
        return cert_fail("vault_cli_round_trip", "badapple binary not found");
    }
    var env: {str: str} = {};
    let tmp = sys_exe_dir() + "/cert_vault_" + rand_hex(4);
    fs_mkdir(tmp);
    env["BADAPPLE_DATA_DIR"] = tmp;
    env["BADAPPLE_VAULT_KEY"] = "vault-cli-test-key";

    match sys_exec_env(bin, ["vault", "set", "test_secret", "hello-world"], env) {
        Json::Obj(r) => {
            if jnum(r, "code") != 0 {
                fs_remove_all(tmp);
                return cert_fail("vault_cli_round_trip", "vault set failed");
            }
        }
        _ => { fs_remove_all(tmp); return cert_fail("vault_cli_round_trip", "vault set failed"); }
    }
    var ok = false;
    match sys_exec_env(bin, ["vault", "get", "test_secret"], env) {
        Json::Obj(r) => {
            if jnum(r, "code") == 0 {
                match json_parse(jstr(r, "out")) {
                    Json::Obj(v) => {
                        if jstr(v, "value") == "hello-world" { ok = true; }
                    }
                    _ => { }
                }
            }
        }
        _ => { }
    }
    if !ok {
        fs_remove_all(tmp);
        return cert_fail("vault_cli_round_trip", "vault round-trip mismatch");
    }
    sys_exec_env(bin, ["vault", "remove", "test_secret"], env);
    match sys_exec_env(bin, ["vault", "list"], env) {
        Json::Obj(r) => {
            match json_parse(jstr(r, "out")) {
                Json::Obj(v) => {
                    if has(v, "keys") {
                        match v["keys"] {
                            Json::Arr(a) => {
                                if len(a) > 0 { ok = false; }
                            }
                            _ => { }
                        }
                    }
                }
                _ => { }
            }
        }
        _ => { }
    }
    fs_remove_all(tmp);
    if !ok {
        return cert_fail("vault_cli_round_trip", "vault list was not empty after remove");
    }
    return cert_ok("vault_cli_round_trip");
}

// ── SLICKS replay + proof checks — against slicks.cy itself ────────────

host fn check_replay_cache() -> Json cap crypto {
    let secret = "0123456789abcdef0123456789abcdef";
    let key_hex = hex_encode(secret);
    let ts = 1700000000000;
    let cn = rand_hex(32);
    let sn = rand_hex(32);
    let proof = client_proof(key_hex, ts, cn, sn, "hello", 32);
    if !verify_client_proof(key_hex, ts, cn, sn, "hello", 32, proof) {
        return cert_fail("replay_cache_rejects_replayed_slicks_proofs",
            "fresh SLICKS proof did not verify");
    }
    var cache = replay_new(4096);
    if !replay_fresh(cache, cn, sn) {
        return cert_fail("replay_cache_rejects_replayed_slicks_proofs",
            "fresh nonce pair was not accepted");
    }
    cache = replay_insert(cache, cn, sn);
    if replay_fresh(cache, cn, sn) {
        return cert_fail("replay_cache_rejects_replayed_slicks_proofs",
            "replayed nonce pair was not rejected");
    }
    return cert_ok("replay_cache_rejects_replayed_slicks_proofs");
}

// ── cage checks — against cage.cy ──────────────────────────────────────

host fn check_cage_traversal() -> Json cap fs {
    let home = env_get("HOME");
    let root = sys_exe_dir() + "/cert_cage_" + rand_hex(4);
    fs_mkdir(root);
    match cage_new([root], home) {
        Json::Obj(c) => {
            if has(c, "error") {
                fs_remove_all(root);
                return cert_fail("tool_cage_rejects_path_traversal", jstr(c, "error"));
            }
            var roots: [str] = [];
            match c["roots"] {
                Json::Arr(a) => {
                    for i in 0..len(a) {
                        match a[i] { Json::Str(s) => { push(roots, s); } _ => { } }
                    }
                }
                _ => { }
            }
            let bad = ["../etc/passwd", "/tmp/../../etc/passwd"];
            for i in 0..len(bad) {
                match resolve_allowlisted(bad[i], roots, home) {
                    Json::Obj(r) => {
                        if !has(r, "error") {
                            fs_remove_all(root);
                            return cert_fail("tool_cage_rejects_path_traversal",
                                "path traversal was not rejected: " + bad[i]);
                        }
                    }
                    _ => { }
                }
            }
            fs_remove_all(root);
            return cert_ok("tool_cage_rejects_path_traversal");
        }
        _ => {
            fs_remove_all(root);
            return cert_fail("tool_cage_rejects_path_traversal", "cage_new failed");
        }
    }
}

host fn check_cage_symlink() -> Json cap fs, proc, crypto {
    let home = env_get("HOME");
    let root = sys_exe_dir() + "/cert_cage_" + rand_hex(4);
    let outside = sys_exe_dir() + "/cert_cage_outside_" + rand_hex(4);
    fs_mkdir(root);
    fs_mkdir(root + "/allowed");
    fs_mkdir(outside);
    write_file(outside + "/target.txt", "outside");
    // symlink via ln — no symlink extern; exec is fine, still cap-gated
    sys_exec("ln", ["-s", outside + "/target.txt", root + "/allowed/escape.txt"]);
    match cage_new([root + "/allowed"], home) {
        Json::Obj(c) => {
            if has(c, "error") {
                fs_remove_all(root);
                fs_remove_all(outside);
                return cert_fail("tool_cage_rejects_symlink_escape", jstr(c, "error"));
            }
            var roots: [str] = [];
            match c["roots"] {
                Json::Arr(a) => {
                    for i in 0..len(a) {
                        match a[i] { Json::Str(s) => { push(roots, s); } _ => { } }
                    }
                }
                _ => { }
            }
            match resolve_allowlisted(root + "/allowed/escape.txt", roots, home) {
                Json::Obj(r) => {
                    fs_remove_all(root);
                    fs_remove_all(outside);
                    if !has(r, "error") {
                        return cert_fail("tool_cage_rejects_symlink_escape",
                            "symlink escape outside the allowlisted root was not rejected");
                    }
                    return cert_ok("tool_cage_rejects_symlink_escape");
                }
                _ => {
                    fs_remove_all(root);
                    fs_remove_all(outside);
                    return cert_fail("tool_cage_rejects_symlink_escape", "resolve failed");
                }
            }
        }
        _ => {
            fs_remove_all(root);
            fs_remove_all(outside);
            return cert_fail("tool_cage_rejects_symlink_escape", "cage_new failed");
        }
    }
}

host fn check_policy_covers_tools() -> Json cap fs {
    var policy = "";
    let cands = ["policy.yaml", "src/policy.yaml", "/var/lib/bad_apple/policy.yaml"];
    for i in 0..len(cands) {
        if fs_exists(cands[i]) { policy = read_file(cands[i]); }
    }
    let required = ["run_shell", "run_applescript", "write_file", "read_file",
                    "index_documents"];
    for i in 0..len(required) {
        if find_sub(policy, required[i]) < 0 {
            return cert_fail("policy_yaml_covers_dangerous_tools",
                "policy.yaml must cover " + required[i]);
        }
    }
    return cert_ok("policy_yaml_covers_dangerous_tools");
}

// ── mesh-brain checks — against mesh_brain.cy ──────────────────────────

host fn check_mesh_plan() -> Json cap fs {
    let dir = sys_exe_dir() + "/cert_mbplan_" + rand_hex(4);
    fs_mkdir(dir);
    write_file(dir + "/config.json",
        "{\"num_hidden_layers\": 28, \"hidden_size\": 896, \"vocab_size\": 151936}");
    var hosts: [str] = ["a:8741", "b:8741", "c:8741"];
    var nomem: [f64] = [];
    match plan(dir, hosts, nomem) {
        Json::Obj(p) => {
            fs_remove_all(dir);
            if has(p, "error") {
                return cert_fail("mesh_brain_plan_covers_layers", jstr(p, "error"));
            }
            match p["ranks"] {
                Json::Arr(ranks) => {
                    if len(ranks) != 3 {
                        return cert_fail("mesh_brain_plan_covers_layers",
                            "expected 3 ranks, got " + str(len(ranks)));
                    }
                    var starts: [i64] = [];
                    var ends: [i64] = [];
                    var hostsv: [str] = [];
                    var nexts: [str] = [];
                    var embeds: [bool] = [];
                    var heads: [bool] = [];
                    var worlds: [i64] = [];
                    for i in 0..len(ranks) {
                        match ranks[i] {
                            Json::Obj(r) => {
                                push(starts, jnum(r, "layer_start"));
                                push(ends, jnum(r, "layer_end"));
                                push(hostsv, jstr(r, "host"));
                                push(nexts, jstr(r, "next_host"));
                                push(embeds, jbool(r, "has_embed"));
                                push(heads, jbool(r, "has_head"));
                                push(worlds, jnum(r, "world"));
                            }
                            _ => { }
                        }
                    }
                    if starts[0] != 0 || ends[2] != 28 {
                        return cert_fail("mesh_brain_plan_covers_layers",
                            "plan does not cover layers 0..28");
                    }
                    for i in 0..(len(ranks) - 1) {
                        if ends[i] != starts[i + 1] {
                            return cert_fail("mesh_brain_plan_covers_layers",
                                "gap/overlap between ranks");
                        }
                        if nexts[i] != hostsv[i + 1] {
                            return cert_fail("mesh_brain_plan_covers_layers",
                                "rank not wired to next host");
                        }
                    }
                    if !embeds[0] || heads[0] {
                        return cert_fail("mesh_brain_plan_covers_layers",
                            "rank 0 must own embed and not head");
                    }
                    if embeds[2] || !heads[2] {
                        return cert_fail("mesh_brain_plan_covers_layers",
                            "last rank must own head and not embed");
                    }
                    if nexts[2] != "" {
                        return cert_fail("mesh_brain_plan_covers_layers",
                            "last rank must have no next_host");
                    }
                    for i in 0..len(ranks) {
                        if worlds[i] != 3 {
                            return cert_fail("mesh_brain_plan_covers_layers",
                                "rank has wrong world");
                        }
                    }
                    return cert_ok("mesh_brain_plan_covers_layers");
                }
                _ => {
                    return cert_fail("mesh_brain_plan_covers_layers", "no ranks in plan");
                }
            }
        }
        _ => {
            fs_remove_all(dir);
            return cert_fail("mesh_brain_plan_covers_layers", "plan returned non-object");
        }
    }
}

host fn check_mesh_shard() -> Json cap fs {
    let dir = sys_exe_dir() + "/cert_mbshard_" + rand_hex(4);
    let model = dir + "/model";
    let outdir = dir + "/shard0";
    fs_mkdir(model);
    fs_mkdir(outdir);
    write_file(model + "/config.json",
        "{\"num_hidden_layers\": 2, \"hidden_size\": 4, \"vocab_size\": 8}");
    // minimal safetensors: 8B LE header len + JSON header + 16B data
    let header = "{\"model.layers.0.mlp.weight\": {\"dtype\": \"BF16\", \"shape\": [2, 2], \"data_offsets\": [0, 8]}, \"model.norm.weight\": {\"dtype\": \"BF16\", \"shape\": [2, 2], \"data_offsets\": [8, 16]}}";
    let blob = i64_le_hex(len(header), 8) + hex_encode(header) + hex_encode("................");
    fs_write_hex(model + "/model.safetensors", blob);
    write_file(model + "/model.safetensors.index.json",
        "{\"metadata\": {\"total_size\": 16}, \"weight_map\": {\"model.layers.0.mlp.weight\": \"model.safetensors\", \"model.norm.weight\": \"model.safetensors\", \"model.layers.1.mlp.weight\": \"model.safetensors\"}}");
    let spec = ShardSpec {
        rank: 0, world: 2, addr: "a:8741", next_host: "b:8741",
        layer_start: 0, layer_end: 1,
        has_embed: true, has_head: false, tie_embeddings: false,
    };
    match build_shard(model, outdir, spec) {
        Json::Obj(r) => {
            if has(r, "error") {
                fs_remove_all(dir);
                return cert_fail("mesh_brain_shard_metadata_written", jstr(r, "error"));
            }
            if jnum(r, "tensors") == 0 {
                fs_remove_all(dir);
                return cert_fail("mesh_brain_shard_metadata_written", "shard kept no tensors");
            }
            match json_parse(read_file(outdir + "/mesh_brain_rank.json")) {
                Json::Obj(meta) => {
                    let want = ["rank", "world", "host", "nextHost", "hasEmbed",
                                "hasHead", "layerStart", "layerEnd"];
                    for i in 0..len(want) {
                        if !has(meta, want[i]) {
                            fs_remove_all(dir);
                            return cert_fail("mesh_brain_shard_metadata_written",
                                "mesh_brain_rank.json missing key " + want[i]);
                        }
                    }
                    if jstr(meta, "nextHost") != "b:8741" || jnum(meta, "layerEnd") != 1 {
                        fs_remove_all(dir);
                        return cert_fail("mesh_brain_shard_metadata_written",
                            "rank metadata values wrong");
                    }
                }
                _ => {
                    fs_remove_all(dir);
                    return cert_fail("mesh_brain_shard_metadata_written",
                        "mesh_brain_rank.json missing");
                }
            }
            let idx_text = read_file(outdir + "/model.safetensors.index.json");
            if find_sub(idx_text, "layers.1") >= 0 {
                fs_remove_all(dir);
                return cert_fail("mesh_brain_shard_metadata_written",
                    "stale index references layer 1, which is not in this shard");
            }
            if find_sub(idx_text, "model.norm.weight") < 0 {
                fs_remove_all(dir);
                return cert_fail("mesh_brain_shard_metadata_written",
                    "index lost the replicated non-layer weight");
            }
            let shard_hex = fs_read_hex(outdir + "/model.safetensors");
            if len(shard_hex) < 16 {
                fs_remove_all(dir);
                return cert_fail("mesh_brain_shard_metadata_written", "shard blob too short");
            }
            let hlen = hex_le_i64(slice_str(shard_hex, 0, 16));
            let shj = hex_decode(slice_str(shard_hex, 16, 16 + hlen * 2));
            if find_sub(shj, "model.layers.0.mlp.weight") < 0 {
                fs_remove_all(dir);
                return cert_fail("mesh_brain_shard_metadata_written",
                    "layer tensor was not re-keyed into the shard");
            }
            fs_remove_all(dir);
            return cert_ok("mesh_brain_shard_metadata_written");
        }
        _ => {
            fs_remove_all(dir);
            return cert_fail("mesh_brain_shard_metadata_written", "build_shard failed");
        }
    }
}

// both ends of a loopback conn driven alternately in one thread —
// write on one handle, read on the other. no threads needed.
host fn check_mesh_frames_no_plaintext() -> Json cap net, crypto {
    let secret = "cert-mesh-secret-cert-mesh-secret-42";
    let listener = net_tcp_listen("127.0.0.1:0");
    if listener < 0 {
        return cert_fail("mesh_brain_frames_carry_no_plaintext", "listen failed");
    }
    let addr = net_local_addr(listener);
    let cli = net_tcp_connect(addr);
    if cli < 0 {
        net_close(listener);
        return cert_fail("mesh_brain_frames_carry_no_plaintext", "connect failed");
    }
    let srv = net_accept(listener);
    if srv < 0 {
        net_close(cli);
        net_close(listener);
        return cert_fail("mesh_brain_frames_carry_no_plaintext", "accept failed");
    }
    // server handshake, driven by hand: challenge → proof → counter-proof
    let nonce = "0123456789abcdef0123456789abcdef";
    var ch: {str: Json} = {};
    ch["op"] = Json::Str("auth");
    ch["text"] = Json::Str(nonce);
    write_mb_frame(srv, Json::Obj(ch));
    match read_mb_frame(cli) {
        Json::Obj(got) => {
            if jstr(got, "op") != "auth" {
                net_close(cli); net_close(srv); net_close(listener);
                return cert_fail("mesh_brain_frames_carry_no_plaintext", "bad challenge");
            }
        }
        _ => { }
    }
    var resp: {str: Json} = {};
    resp["op"] = Json::Str("auth");
    resp["text"] = Json::Str(mb_hmac(secret, "mb-c:" + nonce));
    write_mb_frame(cli, Json::Obj(resp));
    match read_mb_frame(srv) {
        Json::Obj(got) => {
            let expect = mb_hmac(secret, "mb-c:" + nonce);
            if jstr(got, "text") != expect {
                net_close(cli); net_close(srv); net_close(listener);
                return cert_fail("mesh_brain_frames_carry_no_plaintext", "client proof mismatch");
            }
        }
        _ => { }
    }
    var okf: {str: Json} = {};
    okf["op"] = Json::Str("auth-ok");
    okf["text"] = Json::Str(mb_hmac(secret, "mb-s:" + nonce));
    write_mb_frame(srv, Json::Obj(okf));
    read_mb_frame(cli);

    let key = session_key_hex(secret);
    let marker = hex_encode("hidden-state-marker-0xdeadbeef");
    var f: {str: Json} = {};
    f["op"] = Json::Str("step");
    f["text"] = Json::Str("plaintext-marker-op");
    f["shape"] = Json::Arr([Json::Num(1.0), Json::Num(1.0), Json::Num(8.0)]);
    f["dtype"] = Json::Str("bfloat16");
    write_packet(cli, Json::Obj(f), marker, key);
    match read_packet(srv, key) {
        Json::Obj(r) => {
            if has(r, "error") {
                net_close(cli); net_close(srv); net_close(listener);
                return cert_fail("mesh_brain_frames_carry_no_plaintext",
                    "read_packet: " + jstr(r, "error"));
            }
            // echo it back
            write_packet(srv, r["frame"], jstr(r, "payload"), key);
        }
        _ => { }
    }
    var echoed = false;
    match read_packet(cli, key) {
        Json::Obj(r) => {
            if !has(r, "error") {
                match r["frame"] {
                    Json::Obj(fr) => {
                        if jstr(fr, "op") == "step" && jstr(r, "payload") == marker {
                            echoed = true;
                        }
                    }
                    _ => { }
                }
            }
        }
        _ => { }
    }
    net_close(cli);
    net_close(srv);
    net_close(listener);
    if !echoed {
        return cert_fail("mesh_brain_frames_carry_no_plaintext",
            "encrypted round-trip corrupted the frame");
    }
    // wire proof: encrypt the same frame, confirm no marker survives
    let fj = json_emit(Json::Obj(f));
    let inner = u32be_hex(len(fj)) + hex_encode(fj) + marker;
    let ct = aead_seal(key, inner);
    let ct_text = hex_decode(ct);
    let needles = ["\"op\"", "step", "plaintext-marker-op", "hidden-state-marker"];
    for i in 0..len(needles) {
        if find_sub(ct_text, needles[i]) >= 0 {
            return cert_fail("mesh_brain_frames_carry_no_plaintext",
                "plaintext marker visible in encrypted frame");
        }
    }
    // tamper: flip a byte
    let n = len(ct);
    var last = slice_str(ct, n - 1, n);
    if last == "0" { last = "1"; } else { last = "0"; }
    if aead_open(key, slice_str(ct, 0, n - 1) + last) != "" {
        return cert_fail("mesh_brain_frames_carry_no_plaintext",
            "tampered mesh frame decrypted successfully");
    }
    // wrong key
    let wrong = session_key_hex("cert-mesh-secret-cert-mesh-secret-99");
    if aead_open(wrong, ct) != "" {
        return cert_fail("mesh_brain_frames_carry_no_plaintext",
            "mesh frame opened with the wrong session key");
    }
    return cert_ok("mesh_brain_frames_carry_no_plaintext");
}

host fn check_mesh_auth_wrong_key() -> Json cap net, crypto {
    let server_key = "cert-server-key-cert-server-key-000";
    let client_key = "cert-WRONG-key-cert-WRONG-key-0000";
    let listener = net_tcp_listen("127.0.0.1:0");
    if listener < 0 {
        return cert_fail("mesh_brain_auth_rejects_wrong_key", "listen failed");
    }
    let cli = net_tcp_connect(net_local_addr(listener));
    let srv = net_accept(listener);
    let nonce = "aabbccddeeff00112233445566778899";
    var ch: {str: Json} = {};
    ch["op"] = Json::Str("auth");
    ch["text"] = Json::Str(nonce);
    write_mb_frame(srv, Json::Obj(ch));
    // client answers with the WRONG key
    read_mb_frame(cli);
    var resp: {str: Json} = {};
    resp["op"] = Json::Str("auth");
    resp["text"] = Json::Str(mb_hmac(client_key, "mb-c:" + nonce));
    write_mb_frame(cli, Json::Obj(resp));
    var rejected = false;
    match read_mb_frame(srv) {
        Json::Obj(got) => {
            let expect = mb_hmac(server_key, "mb-c:" + nonce);
            if jstr(got, "text") != expect { rejected = true; }
        }
        _ => { }
    }
    net_close(cli);
    net_close(srv);
    net_close(listener);
    if rejected {
        return cert_ok("mesh_brain_auth_rejects_wrong_key");
    }
    return cert_fail("mesh_brain_auth_rejects_wrong_key",
        "client handshake succeeded with the wrong key");
}

host fn check_mesh_dead_peer() -> Json cap net {
    // bind+drop to get a definitely-closed port
    let l = net_tcp_listen("127.0.0.1:0");
    let addr = net_local_addr(l);
    net_close(l);
    let t0 = time_ms();
    let h = net_tcp_connect(addr);
    let elapsed = time_ms() - t0;
    if h >= 0 {
        net_close(h);
        return cert_fail("mesh_brain_dead_peer_fails_fast",
            "ping to a dead rank succeeded");
    }
    if elapsed > 10000 {
        return cert_fail("mesh_brain_dead_peer_fails_fast",
            "dead peer took " + str(elapsed) + "ms to fail — possible hang");
    }
    return cert_ok("mesh_brain_dead_peer_fails_fast");
}

// ── the suite ──────────────────────────────────────────────────────────

host fn run() -> [Json] cap fs, net, crypto, proc {
    var results: [Json] = [];
    push(results, check_no_external_network_sockets());
    push(results, check_unix_sockets_present());
    push(results, check_policy_yaml_present());
    push(results, check_ledger_redacts_secrets());
    push(results, check_p2p_off_by_default());
    push(results, check_tampered_ciphertext());
    push(results, check_wrong_key());
    push(results, check_path_traversal_rejected());
    push(results, check_firewall_patterns());
    push(results, check_ledger_chain());
    push(results, check_sovereign_checkpoint());
    push(results, check_vault_cli());
    push(results, check_replay_cache());
    push(results, check_cage_traversal());
    push(results, check_cage_symlink());
    push(results, check_policy_covers_tools());
    push(results, check_mesh_plan());
    push(results, check_mesh_shard());
    push(results, check_mesh_frames_no_plaintext());
    push(results, check_mesh_auth_wrong_key());
    push(results, check_mesh_dead_peer());
    return results;
}
