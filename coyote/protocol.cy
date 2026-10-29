// protocol.cy — engram packet signing, ported from src/protocol.rs.
//
// The wire format is unchanged: `sig = hex(hmac_sha256(secret, payload))`,
// payload travels base64. Peers signed by the Rust stack verify under this
// stack and vice versa. Concurrency/transport stayed in Rust — this file is
// the pure signing logic, checked det end to end.
//
// Convention: `secret` and `payload` are utf8 strs; `secret_hex` variants
// take the hex form for binary key material (multi_agent.key is raw bytes).

import "sys.cy";

cap fs

struct SignedPacket { sender: str, payload_b64: str, signature_hex: str }

// non-short-circuit equality — the port of protocol.rs::constant_time_eq.
host det fn ct_eq(a: str, b: str) -> bool {
    if len(a) != len(b) { return false; }
    var diff = 0;
    for i in 0..len(a) {
        if slice_str(a, i, i + 1) != slice_str(b, i, i + 1) { diff += 1; }
    }
    return diff == 0;
}

host det fn sign_packet(sender: str, payload: str, secret: str) -> SignedPacket {
    return SignedPacket {
        sender: sender,
        payload_b64: b64_encode(payload),
        signature_hex: hmac_sha256(secret, payload),
    };
}

host det fn packet_payload(p: SignedPacket) -> str {
    return b64_decode(p.payload_b64);
}

host det fn verify_packet(p: SignedPacket, secret: str) -> bool {
    let payload = packet_payload(p);
    if p.payload_b64 != "" && payload == "" { return false; }
    return ct_eq(hmac_sha256(secret, payload), p.signature_hex);
}

// hex-keyed variant — binary key material (the p2p cipher key form).
host det fn verify_packet_hex(p: SignedPacket, secret_hex: str) -> bool {
    let payload = packet_payload(p);
    if p.payload_b64 != "" && payload == "" { return false; }
    return ct_eq(hmac_sha256b(secret_hex, hex_encode(payload)), p.signature_hex);
}

// ── multi-agent secret — the port of protocol.rs::multi_agent_secret ────
// Returns the secret as hex (binary key material travels as hex). Order of
// precedence matches the Rust: env → key file → fresh random 32B.
host fn multi_agent_secret() -> str cap fs {
    let e = env_get("MULTI_AGENT_SECRET");
    if e != "" { return hex_encode(e); }
    let path = "/var/lib/bad_apple/multi_agent.key";
    let existing = fs_read_hex(path);
    if existing != "" { return existing; }
    var seed = "";
    for i in 0..8 { seed = seed + str(rand()); }
    let key = sha256(seed); // 32 bytes of fresh entropy, hex form
    fs_mkdir("/var/lib/bad_apple");
    fs_write_hex(path, key);
    return key;
}

// ── the engram gate — ported from protocol.rs::handle_frame ─────────────
// The whole accept/drop chain runs in-language: envelope decrypt → signature
// verify → bounded truncation → replay window → freshness → goal-similarity
// gate. State travels as a Json the caller holds:
//   { "seen": [ids...], "goals": [[f64...], ...], "dropped": n }
// Returns { action: "accept"|"drop", reason: str, state: <updated>,
//           packet: <CompactEngramPacket Json> }.

cap crypto

import "strlib.cy";

host det fn gate_state_new() -> Json {
    let seen: [Json] = [];
    let goals: [Json] = [];
    return Json::Obj({
        "seen": Json::Arr(seen),
        "goals": Json::Arr(goals),
        "dropped": Json::Num(0.0)
    });
}

host det fn gate_state_field(st: Json, k: str) -> Json {
    match st {
        Json::Obj(m) => {
            if has(m, k) { return m[k]; }
        }
        _ => { }
    }
    return Json::Null;
}

host det fn gate_state_set(st: Json, k: str, v: Json) -> Json {
    match st {
        Json::Obj(m) => { m[k] = v; }
        _ => { }
    }
    return st;
}

host det fn jnum_of(j: Json) -> f64 {
    match j { Json::Num(n) => { return n; } _ => { } }
    return 0.0;
}

host det fn gate_js(j: Json) -> str {
    match j { Json::Str(x) => { return x; } _ => { } }
    return "";
}

// cosine similarity via f64 loops — the math the gate needs on-machine.
host det fn vec_dot(a: Json, b: Json) -> f64 {
    var av: [Json] = [];
    var bv: [Json] = [];
    match a { Json::Arr(x) => { av = x; } _ => { } }
    match b { Json::Arr(x) => { bv = x; } _ => { } }
    let n = len(av);
    if n != len(bv) { return 0.0; }
    var acc = 0.0;
    for i in 0..n {
        acc = acc + jnum_of(av[i]) * jnum_of(bv[i]);
    }
    return acc;
}

host det fn vec_norm(v: Json) -> f64 {
    var vv: [Json] = [];
    match v { Json::Arr(x) => { vv = x; } _ => { } }
    var acc = 0.0;
    for i in 0..len(vv) {
        let x = jnum_of(vv[i]);
        acc = acc + x * x;
    }
    return math_sqrt(acc);
}

// max cosine similarity of `emb` against the goals list. 1.0 when no goals
// configured (gate open), -1.0 when goals exist but none comparable.
host det fn max_goal_sim(emb: Json, goals: Json) -> f64 {
    var gv: [Json] = [];
    match goals { Json::Arr(x) => { gv = x; } _ => { } }
    var ev: [Json] = [];
    match emb { Json::Arr(x) => { ev = x; } _ => { } }
    if len(gv) == 0 || len(ev) == 0 { return 1.0; }
    let na = vec_norm(emb);
    if na == 0.0 { return 0.0; }
    var best = -1.0;
    for i in 0..len(gv) {
        best = goal_sim1(emb, gv[i], na, best);
    }
    return best;
}

host det fn goal_sim1(emb: Json, goal: Json, na: f64, best: f64) -> f64 {
    var gv: [Json] = [];
    match goal { Json::Arr(x) => { gv = x; } _ => { } }
    var ev: [Json] = [];
    match emb { Json::Arr(x) => { ev = x; } _ => { } }
    if len(gv) != len(ev) { return best; }
    let nb = vec_norm(goal);
    if nb == 0.0 { return best; }
    let sim = vec_dot(emb, goal) / (na * nb);
    if sim > best { return sim; }
    return best;
}

// parse + policy-check one incoming frame. `frame` is the raw frame text
// (JSON); `secret`/`key_hex` are hex strings; `now_secs` is unix seconds.
// Pure — no caps needed beyond crypto for the decrypt path.
host det fn engram_gate(
    frame: str, secret_hex: str, key_hex: str, st: Json, now_secs: i64
) -> Json {
    // ── envelope: EncryptedPacket first, then legacy signed ─────────────
    var signed_json = "";
    let env = json_parse(frame);
    var decrypted = false;
    match env {
        Json::Obj(e) => {
            if has(e, "ciphertext_b64") {
                let ct_hex = hex_encode(b64_decode(gate_js(e["ciphertext_b64"])));
                let pt_hex = aead_open(key_hex, ct_hex);
                if pt_hex != "" { signed_json = hex_decode(pt_hex); decrypted = true; }
            }
        }
        _ => { }
    }
    if !decrypted {
        match env {
            Json::Obj(e) => {
                if has(e, "signature_hex") { signed_json = frame; }
            }
            _ => { }
        }
    }
    if signed_json == "" {
        return gate_drop(st, "unparseable envelope", Json::Null);
    }
    // ── signature verify ────────────────────────────────────────────────
    let sp = json_parse(signed_json);
    var sender = "";
    var payload_b64 = "";
    var sig = "";
    match sp {
        Json::Obj(s) => {
            sender = gate_js(s["sender"]);
            payload_b64 = gate_js(s["payload_b64"]);
            sig = gate_js(s["signature_hex"]);
        }
        _ => { return gate_drop(st, "bad signed packet", Json::Null); }
    }
    let pkt = SignedPacket { sender: sender, payload_b64: payload_b64, signature_hex: sig };
    if !verify_packet_hex(pkt, secret_hex) {
        return gate_drop(st, "tampered/unsigned", Json::Null);
    }
    // ── compact packet ──────────────────────────────────────────────────
    let payload = b64_decode(payload_b64);
    let compact = json_parse(payload);
    match compact {
        Json::Obj(_) => { }
        _ => { return gate_drop(st, "bad compact packet", Json::Null); }
    }
    let bounded = engram_truncate(compact);
    // ── replay window ───────────────────────────────────────────────────
    let pid = jnum_of(jobj_get(bounded, "id"));
    let seen = gate_state_field(st, "seen");
    if seen_has(seen, pid) {
        return gate_drop(st, "replay", bounded);
    }
    let seen2 = seen_push(seen, pid);
    var st2 = gate_state_set(st, "seen", seen2);
    // ── freshness ───────────────────────────────────────────────────────
    let ts = i64(jnum_of(jobj_get(bounded, "timestamp")));
    if ts > now_secs + 60 {
        return gate_drop(st2, "future-dated", bounded);
    }
    if now_secs - ts > 300 {
        return gate_drop(st2, "stale", bounded);
    }
    // ── mesh-sync control payloads bypass the similarity gate ───────────
    let pl = jobj_get(bounded, "payload");
    match pl {
        Json::Str(_) => {
            return Json::Obj({
                "action": Json::Str("accept"),
                "reason": Json::Str("sync"),
                "state": st2,
                "packet": bounded
            });
        }
        _ => { }
    }
    // ── similarity gate ─────────────────────────────────────────────────
    let emb = jobj_get(bounded, "embedding");
    var ev: [Json] = [];
    match emb { Json::Arr(x) => { ev = x; } _ => { } }
    if len(ev) == 0 {
        return gate_drop(st2, "empty embedding", bounded);
    }
    let goals = gate_state_field(st2, "goals");
    let sim = max_goal_sim(emb, goals);
    if sim < 0.35 {
        let d = jnum_of(gate_state_field(st2, "dropped")) + 1.0;
        st2 = gate_state_set(st2, "dropped", Json::Num(d));
        return gate_drop(st2, "similarity " + str(sim), bounded);
    }
    return Json::Obj({
        "action": Json::Str("accept"),
        "reason": Json::Str("similarity " + str(sim)),
        "state": st2,
        "packet": bounded
    });
}

host det fn gate_drop(st: Json, reason: str, packet: Json) -> Json {
    return Json::Obj({
        "action": Json::Str("drop"),
        "reason": Json::Str(reason),
        "state": st,
        "packet": packet
    });
}

host det fn jobj_get(j: Json, k: str) -> Json {
    match j {
        Json::Obj(m) => {
            if has(m, k) { return m[k]; }
        }
        _ => { }
    }
    return Json::Null;
}

// truncate text to 1024 chars, brain_state to 100, embedding to 2048.
host det fn engram_truncate(compact: Json) -> Json {
    match compact {
        Json::Obj(m) => {
            if has(m, "experiential_text") {
                let t = gate_js(m["experiential_text"]);
                if len(t) > 1024 { m["experiential_text"] = Json::Str(slice_str(t, 0, 1024)); }
            }
            if has(m, "brain_state") {
                match m["brain_state"] {
                    Json::Arr(b) => {
                        if len(b) > 100 { m["brain_state"] = Json::Arr(take_n(b, 100)); }
                    }
                    _ => { }
                }
            }
            if has(m, "embedding") {
                match m["embedding"] {
                    Json::Arr(e) => {
                        if len(e) > 2048 { m["embedding"] = Json::Arr(take_n(e, 2048)); }
                    }
                    _ => { }
                }
            }
            return Json::Obj(m);
        }
        _ => { return compact; }
    }
}

host det fn take_n(v: [Json], n: i64) -> [Json] {
    var out_v: [Json] = [];
    for i in 0..n { push(out_v, v[i]); }
    return out_v;
}

host det fn seen_has(seen: Json, pid: f64) -> bool {
    match seen {
        Json::Arr(a) => {
            for i in 0..len(a) {
                if jnum_of(a[i]) == pid { return true; }
            }
        }
        _ => { }
    }
    return false;
}

// push + FIFO-evict at 1024 — mirrors VecDeque::push_back/pop_front.
host det fn seen_push(seen: Json, pid: f64) -> Json {
    var a: [Json] = [];
    match seen { Json::Arr(x) => { a = x; } _ => { } }
    push(a, Json::Num(pid));
    if len(a) > 1024 {
        var kept: [Json] = [];
        for i in 1..len(a) { push(kept, a[i]); }
        a = kept;
    }
    return Json::Arr(a);
}

// ── peer-spec parsing — the SSRF guard from parse_peer_spec ─────────────

// → {addr, transport:"tcp"|"websocket"} — empty addr = rejected host.
host det fn parse_peer_spec(peer: str) -> Json {
    let p = trim(peer);
    var host_part = p;
    if starts_with(p, "ws://") { host_part = slice_str(p, 5, len(p)); }
    if starts_with(p, "wss://") { host_part = slice_str(p, 6, len(p)); }
    let c = find_sub(host_part, ":");
    if c >= 0 { host_part = slice_str(host_part, 0, c); }
    if host_part == "169.254.169.254" {
        return Json::Obj({ "addr": Json::Str(""), "transport": Json::Str("tcp") });
    }
    if starts_with(p, "ws://") {
        return Json::Obj({
            "addr": Json::Str(slice_str(p, 5, len(p))),
            "transport": Json::Str("websocket")
        });
    }
    if starts_with(p, "wss://") {
        return Json::Obj({
            "addr": Json::Str(slice_str(p, 6, len(p))),
            "transport": Json::Str("websocket")
        });
    }
    return Json::Obj({ "addr": Json::Str(p), "transport": Json::Str("tcp") });
}
