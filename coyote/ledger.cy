// ledger.cy — the organism's audit ledger, in-language.
//
// Port of the append path in BadAppleSecurity.swift / the Python ledger:
// entries are `{ts, type, data, prev_hash, hash}` JSONL, prev_hash links to
// the last entry's hash, and `hash` is HMAC-SHA256(slicks, canonical body)
// — or plain SHA-256 when no key exists. Canonical = sorted-key compact
// JSON, which `json_emit` produces. Appends hold flock(ledger.lock), the
// same cross-process contract the Swift daemon keeps.
//
// `sha256` and `json_emit` are native/extern — the chain math below has no
// Rust equivalent left to port; this file IS the implementation.

import "sys.cy";
import "strlib.cy";

cap fs

// ── ledger identity ────────────────────────────────────────────────────

host fn ledger_path() -> str cap fs {
    let e = env_get("BADAPPLE_LEDGER");
    if e != "" { return e; }
    return "/var/lib/bad_apple/ledger.jsonl";
}

// SLICKS key as hex (the daemon hex-decodes when the file looks hex,
// else uses utf8 bytes). "" = no key → plain sha256 chain.
host fn ledger_key_hex() -> str cap fs {
    var p = env_get("BADAPPLE_SLICKS_KEY_PATH");
    if p == "" { p = "/var/lib/bad_apple/slicks.key"; }
    if !fs_exists(p) { return ""; }
    let raw = trim(read_file(p));
    if raw == "" { return ""; }
    if len(raw) >= 32 && is_hex(raw) { return raw; }
    return hex_encode(raw);
}

// prev_hash for the next entry — last entry's `hash`, or genesis.
host fn ledger_last_hash(path: str) -> str cap fs {
    let genesis = sha256("bad-apple-genesis-v1");
    if !fs_exists(path) { return genesis; }
    let lines = split(read_file(path), "\n");
    var i = len(lines) - 1;
    while i >= 0 {
        let line = trim(lines[i]);
        if line != "" {
            match json_parse(line) {
                Json::Obj(m) => {
                    if has(m, "hash") {
                        match m["hash"] {
                            Json::Str(h) => { return h; }
                            _ => { }
                        }
                    }
                }
                _ => { }
            }
        }
        i -= 1;
    }
    return genesis;
}

// ── append ─────────────────────────────────────────────────────────────

host fn ledger_append(ev_type: str, data: Json) -> bool cap fs {
    let path = ledger_path();
    let lock = fs_flock(dir_of(path) + "/ledger.lock");
    if lock < 0 { return false; }
    let prev = ledger_last_hash(path);
    let ts = time_iso();
    let canon = json_emit(Json::Obj({
        "data": data,
        "prev_hash": Json::Str(prev),
        "ts": Json::Str(ts),
        "type": Json::Str(ev_type),
    }));
    let key = ledger_key_hex();
    var hash = "";
    if key != "" {
        hash = hmac_sha256b(key, hex_encode(canon));
    } else {
        hash = sha256(canon);
    }
    let line = json_emit(Json::Obj({
        "data": data,
        "hash": Json::Str(hash),
        "prev_hash": Json::Str(prev),
        "ts": Json::Str(ts),
        "type": Json::Str(ev_type),
    }));
    let ok = fs_append(path, line + "\n");
    fs_funlock(lock);
    return ok;
}

// ── delegated inference — asker-side attestation ──────────────────────
// The serving peer ledgers `delegated_query`/`delegated_response`; these
// mirror them on the asking machine so "I asked, they answered" is
// provable on BOTH chains. Prompts hash, never record — attest, don't leak.

host fn ledger_delegated_ask(to_peer: str, request_id: str, prompt: str) -> bool cap fs {
    return ledger_append("delegated_ask", Json::Obj({
        "to_peer": Json::Str(to_peer),
        "request_id": Json::Str(request_id),
        "prompt_sha256": Json::Str(sha256(prompt)),
        "prompt_chars": Json::Num(f64(len(prompt))),
    }));
}

host fn ledger_delegated_answer(to_peer: str, request_id: str, status: str, response: str) -> bool cap fs {
    return ledger_append("delegated_answer", Json::Obj({
        "to_peer": Json::Str(to_peer),
        "request_id": Json::Str(request_id),
        "status": Json::Str(status),
        "response_sha256": Json::Str(sha256(response)),
        "response_chars": Json::Num(f64(len(response))),
    }));
}
