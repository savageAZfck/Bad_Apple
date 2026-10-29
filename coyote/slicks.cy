// slicks.cy — SLICKS local IPC protocol, ported from src/bad_apple_ipc.rs.
//
// Newline-delimited JSON frames over a Unix socket, mutually authenticated
// with HMAC-SHA256 proofs bound to fresh nonces and a prompt hash. This is
// the organism's front door — the gatekeeper/daemon live on the far side.

import "sys.cy";
import "strlib.cy";

cap fs, net, crypto, proc

host det fn slicks_v1() -> i64 { return 1; }
host det fn slicks_v2() -> i64 { return 2; }
host det fn default_socket() -> str { return "/var/run/badapple/substrate.sock"; }
host det fn default_mlx_socket() -> str { return "/var/run/badapple/substrate_mlx.sock"; }
host det fn default_identity_socket() -> str { return "/var/run/badapple/identity.sock"; }
host det fn default_key_path() -> str { return "/var/lib/bad_apple/slicks.key"; }
host det fn max_prompt_bytes() -> i64 { return 65536; }
host det fn max_new_tokens() -> i64 { return 4096; }
host det fn ipc_max_frame_bytes() -> i64 { return 1048576; }
host det fn handshake_max_skew_ms() -> i64 { return 30000; }

host det fn socket_path() -> str {
    let v = env_get("BADAPPLE_SOCKET_PATH");
    if v != "" { return v; }
    return default_socket();
}
host det fn mlx_socket_path() -> str {
    let v = env_get("BADAPPLE_MLX_SOCKET_PATH");
    if v != "" { return v; }
    return default_mlx_socket();
}
host det fn identity_socket_path() -> str {
    let v = env_get("BADAPPLE_IDENTITY_AGENT_SOCKET");
    if v != "" { return v; }
    return default_identity_socket();
}
host det fn key_path() -> str {
    let v = env_get("BADAPPLE_SLICKS_KEY_PATH");
    if v != "" { return v; }
    return default_key_path();
}

// secret bytes as hex — hex-keyed files decode; anything else is raw utf8.
// "" on failure (missing/short) — callers fail closed.
host fn slicks_secret_hex() -> str cap fs {
    var raw = env_get("BADAPPLE_SLICKS_SECRET");
    if raw == "" {
        if !fs_exists(key_path()) { return ""; }
        raw = trim(read_file(key_path()));
    } else {
        raw = trim(raw);
    }
    var hexkey = hex_encode(raw);
    if len(raw) % 2 == 0 && is_hex(raw) { hexkey = raw; }
    if len(hexkey) < 32 { return ""; } // <128 bits fails closed
    return hexkey;
}

host fn timestamp_fresh(ts_ms: i64) -> bool cap fs {
    let now = time_ms();
    var d = now - ts_ms;
    if d < 0 { d = -d; }
    return d <= handshake_max_skew_ms();
}

host fn random_nonce() -> str cap crypto { return rand_hex(32); }

host det fn nonce_is_valid(n: str) -> bool {
    return len(n) == 64 && is_hex(n);
}

// ── proof material — byte-exact with client_material/server_material ───

host det fn server_material(ts_ms: i64, cn: str, sn: str) -> str {
    return "BADAPPLE-SLICKS/1|server|" + str(ts_ms) + "|" + cn + "|" + sn;
}

host det fn client_material(ts_ms: i64, cn: str, sn: str, prompt: str, max_tok: i64) -> str {
    return "BADAPPLE-SLICKS/1|client|" + str(ts_ms) + "|" + cn + "|" + sn + "|"
        + str(max_tok) + "|" + sha256(prompt);
}

host det fn v2_server_material(ts_ms: i64, cn: str, sn: str) -> str {
    return "BADAPPLE-SLICKS/2|server|" + str(ts_ms) + "|" + cn + "|" + sn;
}

host det fn v2_client_material(ts_ms: i64, cn: str, sn: str, prompt: str, max_tok: i64) -> str {
    return "BADAPPLE-SLICKS/2|client|" + str(ts_ms) + "|" + cn + "|" + sn + "|"
        + str(max_tok) + "|" + sha256(prompt);
}

// hmac over the raw secret bytes — key_hex form of the loaded secret.
host det fn proof_for(key_hex: str, material: str) -> str {
    return hmac_sha256b(key_hex, hex_encode(material));
}

host det fn proof_valid(key_hex: str, material: str, proof: str) -> bool {
    if len(proof) % 2 != 0 || !is_hex(proof) { return false; }
    return hmac_verify(key_hex, hex_encode(material), proof);
}

host det fn client_proof(key_hex: str, ts_ms: i64, cn: str, sn: str, prompt: str, max_tok: i64) -> str {
    return proof_for(key_hex, client_material(ts_ms, cn, sn, prompt, max_tok));
}

host det fn verify_client_proof(key_hex: str, ts_ms: i64, cn: str, sn: str, prompt: str, max_tok: i64, proof: str) -> bool {
    return proof_valid(key_hex, client_material(ts_ms, cn, sn, prompt, max_tok), proof);
}

host det fn server_proof(key_hex: str, ts_ms: i64, cn: str, sn: str) -> str {
    return proof_for(key_hex, server_material(ts_ms, cn, sn));
}

host det fn verify_server_proof(key_hex: str, ts_ms: i64, cn: str, sn: str, proof: str) -> bool {
    return proof_valid(key_hex, server_material(ts_ms, cn, sn), proof);
}

// "" on a violated bound — callers treat as rejection.
host det fn validate_request(prompt: str, max_tok: i64) -> str {
    if trim(prompt) == "" { return "prompt must not be empty"; }
    if len(prompt) > max_prompt_bytes() {
        return "prompt exceeds the 65536-byte limit";
    }
    if max_tok < 1 || max_tok > max_new_tokens() {
        return "max_new_tokens must be between 1 and 4096";
    }
    return "";
}

// ── replay cache — FIFO-evicting nonce-pair set ────────────────────────

struct ReplayCache {
    seen: {str: bool},
    order: [str],
    head: i64,
    max: i64,
}

host det fn replay_new(max: i64) -> ReplayCache {
    var empty: {str: bool} = {};
    var none: [str] = [];
    return ReplayCache { seen: empty, order: none, head: 0, max: max };
}

// true when the pair is fresh; false = replay
host det fn replay_fresh(c: ReplayCache, cn: str, sn: str) -> bool {
    let key = cn + "|" + sn;
    return !has(c.seen, key);
}

host det fn replay_insert(c: ReplayCache, cn: str, sn: str) -> ReplayCache {
    var rc = c;
    let key = cn + "|" + sn;
    rc.seen[key] = true;
    push(rc.order, key);
    while len(rc.order) - rc.head > rc.max {
        let old = rc.order[rc.head];
        rc.head = rc.head + 1;
        rc.seen[old] = false;
    }
    return rc;
}

// ── frame io — newline-terminated JSON, ≤1 MiB ─────────────────────────

host fn ipc_write(h: i64, j: Json) -> bool cap net {
    let frame = json_emit(j) + "\n";
    return net_send(h, hex_encode(frame));
}

// reads until a newline byte (0x0a) appears at a byte boundary in the hex.
// "" on close/overflow — caller distinguishes via json_parse.
host fn ipc_read(h: i64) -> str cap net {
    var hex = "";
    let max_hex = 2 * ipc_max_frame_bytes() + 2;
    var got = true;
    while got {
        // scan for the 0x0a byte — hex pairs only
        var at = -1;
        var i = 0;
        while i + 1 < len(hex) {
            if slice_str(hex, i, i + 2) == "0a" { at = i; break; }
            i = i + 2;
        }
        if at >= 0 {
            return hex_decode(slice_str(hex, 0, at));
        }
        if len(hex) > max_hex { return ""; }
        let chunk = net_recv(h, 65536);
        if chunk == "" { got = false; } else { hex = hex + chunk; }
    }
    return "";
}

// ── client side — Hello → Challenge → Execute → stream tokens ──────────

// connects, performs the v1 handshake, sends the execute frame.
// returns the socket handle, or -1 and sets nothing — caller checks.
host fn slicks_connect(sock: str) -> i64 cap fs, net {
    let h = net_unix_connect(sock);
    return h;
}

// full v1 handshake on an open handle. returns server nonce "" on failure.
host fn slicks_handshake(h: i64, key_hex: str, cn: str) -> str cap net {
    let ts = time_ms();
    if !ipc_write(h, Json::Obj({
        "type": Json::Str("hello"),
        "version": Json::Num(f64(slicks_v1())),
        "timestamp_ms": Json::Num(f64(ts)),
        "client_nonce": Json::Str(cn),
    })) { return ""; }
    let line = ipc_read(h);
    if line == "" { return ""; }
    match json_parse(line) {
        Json::Obj(o) => {
            var ty = "";
            if has(o, "type") { match o["type"] { Json::Str(t) => { ty = t; } _ => { } } }
            if ty != "challenge" { return ""; }
            var sn = "";
            var proof = "";
            if has(o, "server_nonce") { match o["server_nonce"] { Json::Str(x) => { sn = x; } _ => { } } }
            if has(o, "proof") { match o["proof"] { Json::Str(x) => { proof = x; } _ => { } } }
            if sn == "" || proof == "" { return ""; }
            if !verify_server_proof(key_hex, ts, cn, sn, proof) { return ""; }
            return sn;
        }
        _ => { return ""; }
    }
}

host fn slicks_execute(h: i64, key_hex: str, cn: str, sn: str, prompt: str, max_tok: i64) -> bool cap net {
    let ts = time_ms();
    let proof = proof_for(key_hex, client_material(ts, cn, sn, prompt, max_tok));
    return ipc_write(h, Json::Obj({
        "type": Json::Str("execute"),
        "version": Json::Num(f64(slicks_v1())),
        "timestamp_ms": Json::Num(f64(ts)),
        "client_nonce": Json::Str(cn),
        "server_nonce": Json::Str(sn),
        "prompt": Json::Str(prompt),
        "max_new_tokens": Json::Num(f64(max_tok)),
        "proof": Json::Str(proof),
    }));
}

// ── full query — the `query_with_metrics` port ─────────────────────────
// connect → hello → challenge/verify → execute → stream Token frames
// until Done/Response/Error. → Json {text, tier, metrics} or {error}.

host fn slicks_query_once(sock: str, key_hex: str, prompt: str, max_tok: i64) -> Json cap fs, net {
    let bad = validate_request(prompt, max_tok);
    if bad != "" {
        return Json::Obj({ "error": Json::Str(bad) });
    }
    if key_hex == "" {
        return Json::Obj({ "error": Json::Str("no SLICKS key") });
    }
    let h = net_unix_connect(sock);
    if h < 0 {
        return Json::Obj({ "error": Json::Str("daemon socket unavailable: " + sock) });
    }
    let cn = rand_hex(32);
    let sn = slicks_handshake(h, key_hex, cn);
    if sn == "" {
        net_close(h);
        return Json::Obj({ "error": Json::Str("SLICKS handshake failed") });
    }
    if !slicks_execute(h, key_hex, cn, sn, prompt, max_tok) {
        net_close(h);
        return Json::Obj({ "error": Json::Str("execute write failed") });
    }
    // stream until Done/Response/Error
    var accepted = false;
    var done = false;
    var text = "";
    var result: Json = Json::Null;
    while !done {
        let line = ipc_read(h);
        if line == "" {
            net_close(h);
            return Json::Obj({ "error": Json::Str("connection closed mid-stream") });
        }
        match json_parse(line) {
            Json::Obj(f) => {
                var ty = "";
                if has(f, "type") { match f["type"] { Json::Str(t) => { ty = t; } _ => { } } }
                if ty == "accepted" {
                    accepted = true;
                } else if ty == "token" {
                    if !accepted {
                        net_close(h);
                        return Json::Obj({ "error": Json::Str("out-of-order IPC frame") });
                    }
                    if has(f, "text") {
                        match f["text"] { Json::Str(t) => { text = text + t; } _ => { } }
                    }
                } else if ty == "done" {
                    if !accepted {
                        net_close(h);
                        return Json::Obj({ "error": Json::Str("out-of-order IPC frame") });
                    }
                    var res: {str: Json} = {};
                    if has(f, "text") { res["text"] = f["text"]; } else { res["text"] = Json::Str(text); }
                    if has(f, "metrics") { res["metrics"] = f["metrics"]; }
                    net_close(h);
                    return Json::Obj(res);
                } else if ty == "response" {
                    if !accepted {
                        net_close(h);
                        return Json::Obj({ "error": Json::Str("out-of-order IPC frame") });
                    }
                    var res: {str: Json} = {};
                    if has(f, "result") {
                        res["text"] = Json::Str(json_emit(f["result"]));
                    }
                    net_close(h);
                    return Json::Obj(res);
                } else if ty == "error" {
                    var msg = "Bad Apple request failed";
                    if has(f, "message") { match f["message"] { Json::Str(m) => { msg = m; } _ => { } } }
                    net_close(h);
                    return Json::Obj({ "error": Json::Str(msg) });
                } else {
                    net_close(h);
                    return Json::Obj({ "error": Json::Str("unexpected frame: " + ty) });
                }
            }
            _ => {
                net_close(h);
                return Json::Obj({ "error": Json::Str("invalid IPC frame") });
            }
        }
        if done { break; }
    }
    net_close(h);
    return Json::Obj({ "error": Json::Str("stream ended without done") });
}

// query with the "not loaded yet" retry budget — mirrors
// query_with_metrics (90s window, BADAPPLE_NO_WAIT=1 disables).
host fn slicks_query(sock: str, prompt: str, max_tok: i64) -> Json cap fs, net, proc {
    let key_hex = slicks_secret_hex();
    let budget_ms = 90000;
    let t0 = time_ms();
    var last: Json = Json::Null;
    var tries = 0;
    var keep = true;
    while keep {
        tries = tries + 1;
        last = slicks_query_once(sock, key_hex, prompt, max_tok);
        var retry = false;
        match last {
            Json::Obj(o) => {
                if has(o, "error") {
                    let e = jstr_of(o, "error");
                    if find_sub(e, "not loaded yet") >= 0
                        && env_get("BADAPPLE_NO_WAIT") == "" {
                        retry = time_ms() - t0 < budget_ms;
                    }
                }
            }
            _ => { }
        }
        if retry { sys_sleep_ms(2000); } else { keep = false; }
        if tries > 60 { keep = false; }
    }
    return last;
}

host det fn jstr_of(o: {str: Json}, k: str) -> str {
    if has(o, k) {
        match o[k] { Json::Str(s) => { return s; } _ => { } }
    }
    return "";
}

// call_agent — the `__BADAPPLE_AGENT__` envelope over SLICKS.
// → Json result value or {error}.
host fn slicks_call_agent(sock: str, method: str, params: Json, max_tok: i64) -> Json cap fs, net, proc {
    var req: {str: Json} = {};
    req["id"] = Json::Str("cli-1");
    req["method"] = Json::Str(method);
    match params {
        Json::Null => { }
        _ => { req["params"] = params; }
    }
    let prompt = "__BADAPPLE_AGENT__ " + json_emit(Json::Obj(req));
    match slicks_query(sock, prompt, max_tok) {
        Json::Obj(r) => {
            if has(r, "error") { return Json::Obj({ "error": r["error"] }); }
            let text = jstr_of(r, "text");
            let parsed = json_parse(text);
            match parsed {
                Json::Null => { return Json::Str(text); }  // raw text fallback
                _ => { return parsed; }
            }
        }
        _ => { return Json::Obj({ "error": Json::Str("agent call failed") }); }
    }
}

// ── identity agent — v2 signing over its own unix socket ───────────────
// JSON-RPC one-shot: {"command": ...} + \n → single JSON line reply.

host fn identity_call(req: Json) -> Json cap fs, net {
    let path = identity_socket_path();
    if !fs_exists(path) {
        return Json::Obj({ "error": Json::Str("identity agent socket not found") });
    }
    let h = net_unix_connect(path);
    if h < 0 {
        return Json::Obj({ "error": Json::Str("unable to connect to identity agent") });
    }
    if !ipc_write(h, req) {
        net_close(h);
        return Json::Obj({ "error": Json::Str("identity agent write failed") });
    }
    let line = ipc_read(h);
    net_close(h);
    if line == "" {
        return Json::Obj({ "error": Json::Str("identity agent closed the connection") });
    }
    return json_parse(line);
}

host fn identity_available() -> bool cap fs, net {
    return fs_exists(identity_socket_path());
}

host fn identity_public_key() -> Json cap fs, net {
    match identity_call(Json::Obj({ "command": Json::Str("public_key") })) {
        Json::Obj(r) => {
            if jstr_of(r, "ok") == "" {
                match r["ok"] {
                    Json::Bool(b) => {
                        if !b {
                            return Json::Obj({ "error": Json::Str("identity agent public_key failed") });
                        }
                    }
                    _ => { }
                }
            }
            if has(r, "public_key") {
                return r["public_key"];
            }
            return Json::Obj({ "error": Json::Str("identity agent did not return a public key") });
        }
        _ => { return Json::Obj({ "error": Json::Str("identity agent call failed") }); }
    }
}

// v2 proof — identity agent signs the base64'd material (P-256 in the
// Secure Enclave). → signature str or Json {error} flattened to "".
host fn v2_sign(material: str) -> Json cap fs, net {
    return identity_call(Json::Obj({
        "command": Json::Str("sign"),
        "message_b64": Json::Str(b64_encode(material)),
    }));
}

// TOFU key pin — first connection writes daemon.pub; later ones verify.
host fn v2_pinned_pubkey() -> str cap fs {
    var path = env_get("BADAPPLE_SERVER_KEY_PATH");
    if path == "" { path = "/var/lib/bad_apple/keys/daemon.pub"; }
    if !fs_exists(path) { return ""; }
    return trim(read_file(path));
}

host fn v2_pin_pubkey(pubkey: str) -> bool cap fs {
    var path = env_get("BADAPPLE_SERVER_KEY_PATH");
    if path == "" { path = "/var/lib/bad_apple/keys/daemon.pub"; }
    fs_mkdir(dir_of(path));
    write_file(path, pubkey);
    fs_chmod(path, 420); // 0644
    return true;
}
