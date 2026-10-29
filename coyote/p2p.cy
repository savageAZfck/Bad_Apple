// p2p.cy — delegated inference wire protocol, ported from src/p2p_model.rs.
//
// Wire format (unchanged, peer-compatible with the Rust stack):
//   key    = sha256(preshared secret bytes)          → AES-256-GCM
//   frame  = u32-BE length + [nonce(12) || ct || tag(16)]
//   body   = ciborium serde — externally-tagged enum {"Variant": {...}}
//   fields in serde declaration order — index order IS the wire order.
//
// Transport/cipher live as externs (cap net/crypto); the protocol — frame
// construction, length prefixes, request lifecycle, dual attestation —
// is all in-language here.

import "sys.cy";
import "strlib.cy";
import "ledger.cy";
import "slicks.cy";

cap net, crypto, fs

// wire limits — match p2p_model.rs
host det fn max_frame_bytes() -> i64 { return 2097152; }   // 2 MiB
host det fn max_delegated_prompt() -> i64 { return 32768; }
host det fn max_delegated_tokens() -> i64 { return 2048; }
host det fn chunk_size() -> i64 { return 262144; }         // 256 KiB
host det fn max_model_bytes() -> i64 { return 85899345920; } // 80 GiB

// ── p2p key — port of resolve_secret + derive_key ─────────────────────
// BADAPPLE_P2P_SECRET (utf8) else the slicks key file, trimmed.
// key = sha256(secret bytes), returned as hex.
host fn p2p_key() -> str cap fs {
    var secret = env_get("BADAPPLE_P2P_SECRET");
    if secret == "" {
        var p = env_get("BADAPPLE_SLICKS_KEY_PATH");
        if p == "" { p = "/var/lib/bad_apple/slicks.key"; }
        if !fs_exists(p) { return ""; }
        secret = trim(read_file(p));
    }
    if len(secret) < 32 { return ""; }
    return sha256(secret);
}

// ── frames ─────────────────────────────────────────────────────────────

host fn net_read_exact(h: i64, nbytes: i64) -> str cap net {
    var acc = "";
    while len(acc) < nbytes * 2 {
        let chunk = net_recv(h, nbytes - len(acc) / 2);
        if chunk == "" { return ""; }
        acc = acc + chunk;
    }
    return acc;
}

host fn send_frame(h: i64, key: str, frame: Json) -> bool cap net, crypto {
    let ct = aead_seal(key, cbor_encode(frame));
    if ct == "" { return false; }
    return net_send(h, u32be_hex(len(ct) / 2) + ct);
}

host fn recv_frame(h: i64, key: str) -> Json cap net, crypto {
    let hdr = net_read_exact(h, 4);
    if hdr == "" { return Json::Str("hdr_empty"); }
    let n = hex_to_i64(hdr);
    if n <= 0 || n > max_frame_bytes() { return Json::Str("bad_len:" + hdr); }
    let ct = net_read_exact(h, n);
    if ct == "" { return Json::Str("ct_short:" + str(n)); }
    let pt = aead_open(key, ct);
    if pt == "" { return Json::Str("decrypt_fail:" + slice_str(ct, 0, 40)); }
    let decoded = cbor_decode(pt);
    match decoded {
        Json::Null => { return Json::Str("cbor_fail:" + slice_str(pt, 0, 60)); }
        _ => { return decoded; }
    }
}

// ── delegated inference — the port of P2PModelTransfer::infer ─────────
// Returns Json: {"status":"ok","text":...,"tier":...,"elapsed_ms":...}
// or {"status":"error","message":...}. Both outcomes attest on the local
// chain — the serving peer's ledger mirrors it.

host fn infer(peer: str, prompt: str, max_tokens: i64, from_peer: str) -> Json cap net, crypto, fs {
    if len(prompt) == 0 || len(prompt) > max_delegated_prompt() {
        return Json::Obj({ "status": Json::Str("error"), "message": Json::Str("prompt empty or too large") });
    }
    let key = p2p_key();
    if key == "" {
        return Json::Obj({ "status": Json::Str("error"), "message": Json::Str("no p2p secret") });
    }
    var toks = max_tokens;
    if toks < 1 { toks = 1; }
    if toks > max_delegated_tokens() { toks = max_delegated_tokens(); }
    let h = net_tcp_connect(peer);
    if h < 0 {
        return Json::Obj({ "status": Json::Str("error"), "message": Json::Str("connect failed") });
    }
    let request_id = rand_hex(8); // {:016x}
    let frame = Json::Obj({
        "InferRequest": Json::Obj({
            "request_id": Json::Str(request_id),
            "prompt": Json::Str(prompt),
            "max_tokens": Json::Num(f64(toks)),
            "from_peer": Json::Str(from_peer),
        }),
    });
    if !send_frame(h, key, frame) {
        net_close(h);
        return Json::Obj({ "status": Json::Str("error"), "message": Json::Str("send failed") });
    }
    ledger_delegated_ask(peer, request_id, prompt);
    let resp = recv_frame(h, key);
    net_close(h);
    match resp {
        Json::Obj(m) => {
            if has(m, "InferResponse") {
                match m["InferResponse"] {
                    Json::Obj(r) => {
                        if has(r, "request_id") && has(r, "text") {
                            match r["request_id"] {
                                Json::Str(rid) => {
                                    if rid == request_id {
                                        var text = "";
                                        match r["text"] { Json::Str(t) => { text = t; } _ => { } }
                                        var tier = Json::Null;
                                        if has(r, "tier") { tier = r["tier"]; }
                                        var ms = Json::Null;
                                        if has(r, "elapsed_ms") { ms = r["elapsed_ms"]; }
                                        ledger_delegated_answer(peer, request_id, "ok", text);
                                        return Json::Obj({
                                            "status": Json::Str("ok"),
                                            "text": Json::Str(text),
                                            "tier": tier,
                                            "elapsed_ms": ms,
                                        });
                                    }
                                    ledger_delegated_answer(peer, request_id, "error", "request_id mismatch");
                                    return Json::Obj({ "status": Json::Str("error"), "message": Json::Str("request_id mismatch") });
                                }
                                _ => {
                                    ledger_delegated_answer(peer, request_id, "error", "bad request_id");
                                    return Json::Obj({ "status": Json::Str("error"), "message": Json::Str("bad request_id") });
                                }
                            }
                        }
                        ledger_delegated_answer(peer, request_id, "error", "malformed response");
                        return Json::Obj({ "status": Json::Str("error"), "message": Json::Str("malformed response") });
                    }
                    _ => { }
                }
            }
            if has(m, "Error") {
                match m["Error"] {
                    Json::Obj(e) => {
                        var msg = "peer error";
                        if has(e, "message") {
                            match e["message"] { Json::Str(x) => { msg = x; } _ => { } }
                        }
                        ledger_delegated_answer(peer, request_id, "error", msg);
                        return Json::Obj({ "status": Json::Str("error"), "message": Json::Str(msg) });
                    }
                    _ => { }
                }
            }
        }
        _ => { }
    }
    ledger_delegated_answer(peer, request_id, "error", "unexpected frame");
    return Json::Obj({ "status": Json::Str("error"), "message": Json::Str("unexpected frame: " + slice_str(json_emit(resp), 0, 120)) });
}

// ── serving side — the port of handle_infer_request ────────────────────
// Serving is opt-in per peer (BADAPPLE_P2P_INFER=1), inference-only:
// the __BADAPPLE_DELEGATED__ envelope routes through the engine's
// delegated path — no meta commands, no tools, no approvals. The serving
// daemon ledgers the query under the requester's peer label.

host det fn infer_serving_enabled() -> bool {
    return env_get("BADAPPLE_P2P_INFER") == "1";
}

host fn send_error(h: i64, key: str, msg: str) -> bool cap net, crypto {
    var e: {str: Json} = {};
    e["message"] = Json::Str(msg);
    var f: {str: Json} = {};
    f["Error"] = Json::Obj(e);
    return send_frame(h, key, Json::Obj(f));
}

// handles one InferRequest frame on an accepted conn. Returns "" ok or
// an error string — either way a frame went back to the peer.
host fn handle_infer(h: i64, key: str, req: Json, peer: str) -> str cap fs, net, crypto, proc {
    if !infer_serving_enabled() {
        send_error(h, key, "this peer does not serve delegated inference (set BADAPPLE_P2P_INFER=1 to enable)");
        return "delegated inference refused — serving disabled on this peer";
    }
    var prompt = "";
    var request_id = "";
    var from_peer = "";
    var max_tok = 512;
    match req {
        Json::Obj(r) => {
            if has(r, "prompt") { match r["prompt"] { Json::Str(x) => { prompt = x; } _ => { } } }
            if has(r, "request_id") { match r["request_id"] { Json::Str(x) => { request_id = x; } _ => { } } }
            if has(r, "from_peer") { match r["from_peer"] { Json::Str(x) => { from_peer = x; } _ => { } } }
            if has(r, "max_tokens") { match r["max_tokens"] { Json::Num(n) => { max_tok = i64(n); } _ => { } } }
        }
        _ => { }
    }
    if prompt == "" || len(prompt) > max_delegated_prompt() {
        send_error(h, key, "prompt empty or exceeds 32768 bytes");
        return "delegated prompt rejected: " + str(len(prompt)) + " bytes";
    }
    if max_tok < 1 { max_tok = 1; }
    if max_tok > max_delegated_tokens() { max_tok = max_delegated_tokens(); }

    // serving-side ledger record under the requester's label
    ledger_delegated_ask(peer + " (" + from_peer + ")", request_id, prompt);

    var env: {str: Json} = {};
    env["prompt"] = Json::Str(prompt);
    env["from_peer"] = Json::Str(from_peer);
    env["max_tokens"] = Json::Num(f64(max_tok));
    let daemon_prompt = "__BADAPPLE_DELEGATED__ " + json_emit(Json::Obj(env));

    let t0 = time_ms();
    match slicks_query(socket_path(), daemon_prompt, max_tok) {
        Json::Obj(r) => {
            if has(r, "error") {
                var msg = "delegated inference failed";
                match r["error"] { Json::Str(x) => { msg = x; } _ => { } }
                send_error(h, key, msg);
                ledger_delegated_answer(peer, request_id, "error", msg);
                return msg;
            }
            var text = "";
            if has(r, "text") { match r["text"] { Json::Str(x) => { text = x; } _ => { } } }
            var tier: Json = Json::Null;
            if has(r, "metrics") {
                match r["metrics"] {
                    Json::Obj(mm) => {
                        if has(mm, "tier") { tier = mm["tier"]; }
                    }
                    _ => { }
                }
            }
            var resp: {str: Json} = {};
            resp["request_id"] = Json::Str(request_id);
            resp["text"] = Json::Str(text);
            match tier {
                Json::Str(_) => { resp["tier"] = tier; }
                Json::Null => { }
                _ => { }
            }
            resp["elapsed_ms"] = Json::Num(f64(time_ms() - t0));
            var f: {str: Json} = {};
            f["InferResponse"] = Json::Obj(resp);
            send_frame(h, key, Json::Obj(f));
            ledger_delegated_answer(peer, request_id, "ok", text);
            return "";
        }
        _ => {
            send_error(h, key, "delegated inference failed");
            ledger_delegated_answer(peer, request_id, "error", "daemon call failed");
            return "daemon call failed";
        }
    }
}

// ── model transfer — ports of pull/push + serve-side handlers ─────────
//
// ChunkFrame.bytes travels as CBOR int array (serde Vec<u8> encoding);
// the codec surfaces it to us as Json::Bytes — a hex str. file names map
// to model ids by stripping known extensions.

host det fn model_id_of(file_name: str) -> str {
    var id = file_name;
    let exts = [".safetensors", ".bin", ".gguf", ".pt", ".ckpt"];
    for i in 0..len(exts) {
        let e = exts[i];
        if ends_with(id, e) {
            id = slice_str(id, 0, len(id) - len(e));
        }
    }
    return id;
}

// scan a model dir → manifest Json for model_id, or Json::Null.
// stateless port of advertised-manifest lookup — rescanning per request
// means pushes take effect without bookkeeping.
host fn find_manifest(model_dir: str, model_id: str) -> Json cap fs {
    if !fs_is_dir(model_dir) { return Json::Null; }
    let names = fs_list(model_dir);
    for i in 0..len(names) {
        let name = names[i];
        let path = model_dir + "/" + name;
        if !fs_is_dir(path) {
            let sz = fs_size(path);
            if sz > 0 && sz <= max_model_bytes() && model_id_of(name) == model_id {
                let sha = sha256_file(path);
                if sha != "" {
                    return Json::Obj({
                        "model_id": Json::Str(model_id),
                        "file_name": Json::Str(name),
                        "total_bytes": Json::Num(f64(sz)),
                        "sha256": Json::Str(sha),
                        "chunk_size": Json::Num(f64(chunk_size())),
                    });
                }
            }
        }
    }
    return Json::Null;
}

// every advertised manifest — the scan_and_advertise port.
host fn scan_manifests(model_dir: str, origin_peer: str) -> Json cap fs {
    var arr: [Json] = [];
    if !fs_is_dir(model_dir) { return Json::Arr(arr); }
    let names = fs_list(model_dir);
    for i in 0..len(names) {
        let name = names[i];
        let path = model_dir + "/" + name;
        if !fs_is_dir(path) {
            let sz = fs_size(path);
            if sz > 0 && sz <= max_model_bytes() {
                let sha = sha256_file(path);
                if sha != "" {
                    push(arr, Json::Obj({
                        "model_id": Json::Str(model_id_of(name)),
                        "file_name": Json::Str(name),
                        "total_bytes": Json::Num(f64(sz)),
                        "sha256": Json::Str(sha),
                        "origin_peer": Json::Str(origin_peer),
                        "chunk_size": Json::Num(f64(chunk_size())),
                    }));
                }
            }
        }
    }
    return Json::Arr(arr);
}

// send all chunks of `path` after an accepted offer; wait Ack(i) after
// each. "" on success, error string otherwise.
host fn send_chunks(h: i64, key: str, path: str, total: i64, resume: Json) -> str cap net, crypto, fs {
    for index in 0..total {
        var skip = false;
        match resume {
            Json::Arr(rs) => {
                for i in 0..len(rs) {
                    match rs[i] {
                        Json::Num(n) => { if i64(n) == index { skip = true; } }
                        _ => { }
                    }
                }
            }
            _ => { }
        }
        if !skip {
            let off = index * chunk_size();
            let want = chunk_size();
            let hexb = fs_read_at(path, off, want);
            var f: {str: Json} = {};
            var cf: {str: Json} = {};
            cf["index"] = Json::Num(f64(index));
            cf["total"] = Json::Num(f64(total));
            cf["bytes"] = Json::Bytes(hexb);
            f["Chunk"] = Json::Obj(cf);
            if !send_frame(h, key, Json::Obj(f)) { return "chunk send failed"; }
            match recv_frame(h, key) {
                Json::Obj(m) => {
                    if has(m, "Ack") {
                        match m["Ack"] {
                            Json::Obj(a) => {
                                var ai = -1;
                                if has(a, "index") {
                                    match a["index"] { Json::Num(n) => { ai = i64(n); } _ => { } }
                                }
                                if ai != index { return "ack index mismatch"; }
                            }
                            _ => { return "bad ack"; }
                        }
                    } else {
                        return "expected Ack " + str(index);
                    }
                }
                _ => { return "ack recv failed"; }
            }
        }
    }
    return "";
}

// receive chunks into part_path until Done; verify sha256 + count.
// "" on success (file complete + verified), error string otherwise.
host fn recv_chunks(h: i64, key: str, offer: Json, part_path: str) -> str cap net, crypto, fs {
    var total_chunks = 0;
    var csize = 0;
    var want_sha = "";
    match offer {
        Json::Obj(o) => {
            if has(o, "total_chunks") { match o["total_chunks"] { Json::Num(n) => { total_chunks = i64(n); } _ => { } } }
            if has(o, "chunk_size") { match o["chunk_size"] { Json::Num(n) => { csize = i64(n); } _ => { } } }
            if has(o, "sha256") { match o["sha256"] { Json::Str(x) => { want_sha = x; } _ => { } } }
        }
        _ => { }
    }
    if total_chunks < 1 || csize < 1 { return "malformed offer"; }
    var received: {str: bool} = {};
    var done = false;
    while !done {
        let fr = recv_frame(h, key);
        match fr {
            Json::Obj(m) => {
                if has(m, "Chunk") {
                    match m["Chunk"] {
                        Json::Obj(c) => {
                            var idx = -1;
                            var data = "";
                            if has(c, "index") { match c["index"] { Json::Num(n) => { idx = i64(n); } _ => { } } }
                            if has(c, "bytes") {
                                match c["bytes"] {
                                    Json::Bytes(hx) => { data = hx; }
                                    Json::Arr(_) => { return "chunk bytes not compacted"; }
                                    _ => { return "bad chunk bytes"; }
                                }
                            }
                            if idx < 0 || idx >= total_chunks { return "chunk index out of range"; }
                            if fs_write_at(part_path, idx * csize, data) < 0 {
                                return "chunk write failed";
                            }
                            received[str(idx)] = true;
                            var af: {str: Json} = {};
                            var ai: {str: Json} = {};
                            ai["index"] = Json::Num(f64(idx));
                            af["Ack"] = Json::Obj(ai);
                            if !send_frame(h, key, Json::Obj(af)) { return "ack send failed"; }
                        }
                        _ => { return "bad chunk frame"; }
                    }
                } else {
                    if has(m, "Error") { return "peer error during transfer"; }
                    return "unexpected frame mid-transfer";
                }
            }
            // serde unit variants decode as bare strings
            Json::Str(u) => {
                if u == "Done" { done = true; }
                else { return "unexpected unit frame: " + u; }
            }
            _ => { return "frame recv failed"; }
        }
    }
    var got = 0;
    for i in 0..total_chunks {
        if has(received, str(i)) { got = got + 1; }
    }
    if got != total_chunks { return "missing chunks: " + str(total_chunks - got); }
    let actual = sha256_file(part_path);
    if actual != want_sha { return "sha256 mismatch"; }
    return "";
}

// serve side: a peer asked to pull a model — port of handle_pull_request.
host fn handle_pull(h: i64, key: str, req: Json, model_dir: str) -> str cap fs, net, crypto {
    var model_id = "";
    var resume = Json::Null;
    match req {
        Json::Obj(r) => {
            if has(r, "model_id") { match r["model_id"] { Json::Str(x) => { model_id = x; } _ => { } } }
            if has(r, "resume_chunks") { resume = r["resume_chunks"]; }
        }
        _ => { }
    }
    match find_manifest(model_dir, model_id) {
        Json::Obj(mf) => {
            var fname = "";
            var total_bytes = 0;
            var sha = "";
            match mf["file_name"] { Json::Str(x) => { fname = x; } _ => { } }
            match mf["total_bytes"] { Json::Num(n) => { total_bytes = i64(n); } _ => { } }
            match mf["sha256"] { Json::Str(x) => { sha = x; } _ => { } }
            let path = model_dir + "/" + fname;
            let total_chunks = (total_bytes + chunk_size() - 1) / chunk_size();
            var of: {str: Json} = {};
            var oo: {str: Json} = {};
            oo["model_id"] = Json::Str(model_id);
            oo["file_name"] = Json::Str(fname);
            oo["total_bytes"] = Json::Num(f64(total_bytes));
            oo["total_chunks"] = Json::Num(f64(total_chunks));
            oo["chunk_size"] = Json::Num(f64(chunk_size()));
            oo["sha256"] = Json::Str(sha);
            of["Offer"] = Json::Obj(oo);
            if !send_frame(h, key, Json::Obj(of)) { return "offer send failed"; }
            match recv_frame(h, key) {
                Json::Str(u) => {
                    if u == "Accept" {
                        let err = send_chunks(h, key, path, total_chunks, resume);
                        if err != "" { return err; }
                        send_frame(h, key, Json::Str("Done"));
                        return "";
                    }
                    if u == "Cancel" { return "puller cancelled"; }
                    return "expected Accept, got " + u;
                }
                Json::Obj(af) => {
                    if has(af, "Reject") { return "puller rejected offer"; }
                    return "expected Accept";
                }
                _ => { return "accept recv failed"; }
            }
        }
        _ => {
            send_error(h, key, "model " + model_id + " not available from this peer");
            return "model not found: " + model_id;
        }
    }
}

// serve side: a peer is pushing a model to us — port of
// handle_push_receive. Receives into <name>.part, verifies, renames.
host fn handle_push_receive(h: i64, key: str, req: Json, model_dir: str) -> str cap fs, net, crypto {
    if !send_frame(h, key, Json::Str("PushAccept")) { return "pushaccept send failed"; }

    let offer = recv_frame(h, key);
    match offer {
        Json::Obj(m) => {
            if has(m, "Error") { return "peer error"; }
            if !has(m, "Offer") { return "expected Offer"; }
            match m["Offer"] {
                Json::Obj(o) => {
                    var fname = "";
                    var model_id = "";
                    var total_bytes = 0;
                    if has(o, "file_name") { match o["file_name"] { Json::Str(x) => { fname = x; } _ => { } } }
                    if has(o, "model_id") { match o["model_id"] { Json::Str(x) => { model_id = x; } _ => { } } }
                    if has(o, "total_bytes") { match o["total_bytes"] { Json::Num(n) => { total_bytes = i64(n); } _ => { } } }
                    if fname == "" || contains(fname, "/") || contains(fname, "..") {
                        send_error(h, key, "bad file name");
                        return "bad file name";
                    }
                    if total_bytes < 1 || total_bytes > max_model_bytes() {
                        var rf: {str: Json} = {};
                        var rr: {str: Json} = {};
                        rr["reason"] = Json::Str("model is too large");
                        rf["Reject"] = Json::Obj(rr);
                        send_frame(h, key, Json::Obj(rf));
                        return "model is too large";
                    }
                    if !send_frame(h, key, Json::Str("Accept")) { return "accept send failed"; }
                    fs_mkdir(model_dir);
                    let part = model_dir + "/" + fname + ".part";
                    if fs_size(part) != total_bytes {
                        if !fs_truncate(part, total_bytes) { return "part alloc failed"; }
                    }
                    let err = recv_chunks(h, key, m["Offer"], part);
                    if err != "" { return err; }
                    if !fs_rename(part, model_dir + "/" + fname) { return "finalize rename failed"; }
                    return "";
                }
                _ => { return "bad offer"; }
            }
        }
        _ => { return "offer recv failed"; }
    }
}

// client: pull a model — port of pull_over_stream. resume is a Json Arr
// of chunk indices already on disk (usually empty).
host fn pull(peer: str, model_id: str, resume: Json, out_dir: str) -> Json cap net, crypto, fs {
    let key = p2p_key();
    if key == "" { return Json::Obj({"status": Json::Str("error"), "message": Json::Str("no p2p secret")}); }
    let h = net_tcp_connect(peer);
    if h < 0 { return Json::Obj({"status": Json::Str("error"), "message": Json::Str("connect failed")}); }
    var rf: {str: Json} = {};
    var rr: {str: Json} = {};
    rr["model_id"] = Json::Str(model_id);
    rr["resume_chunks"] = resume;
    rf["Request"] = Json::Obj(rr);
    if !send_frame(h, key, Json::Obj(rf)) {
        net_close(h);
        return Json::Obj({"status": Json::Str("error"), "message": Json::Str("request send failed")});
    }
    let offer = recv_frame(h, key);
    match offer {
        Json::Obj(m) => {
            if has(m, "Error") {
                net_close(h);
                return Json::Obj({"status": Json::Str("error"), "message": Json::Str("peer error")});
            }
            if !has(m, "Offer") {
                net_close(h);
                return Json::Obj({"status": Json::Str("error"), "message": Json::Str("expected Offer")});
            }
            match m["Offer"] {
                Json::Obj(o) => {
                    var fname = "";
                    var total_bytes = 0;
                    if has(o, "file_name") { match o["file_name"] { Json::Str(x) => { fname = x; } _ => { } } }
                    if has(o, "total_bytes") { match o["total_bytes"] { Json::Num(n) => { total_bytes = i64(n); } _ => { } } }
                    if fname == "" || contains(fname, "/") || contains(fname, "..") {
                        net_close(h);
                        return Json::Obj({"status": Json::Str("error"), "message": Json::Str("bad file name in offer")});
                    }
                    if total_bytes < 1 || total_bytes > max_model_bytes() {
                        net_close(h);
                        return Json::Obj({"status": Json::Str("error"), "message": Json::Str("model is too large")});
                    }
                    if !send_frame(h, key, Json::Str("Accept")) {
                        net_close(h);
                        return Json::Obj({"status": Json::Str("error"), "message": Json::Str("accept send failed")});
                    }
                    fs_mkdir(out_dir);
                    let part = out_dir + "/" + fname + ".part";
                    if fs_size(part) != total_bytes {
                        if !fs_truncate(part, total_bytes) {
                            net_close(h);
                            return Json::Obj({"status": Json::Str("error"), "message": Json::Str("part alloc failed")});
                        }
                    }
                    let err = recv_chunks(h, key, m["Offer"], part);
                    if err != "" {
                        net_close(h);
                        return Json::Obj({"status": Json::Str("error"), "message": Json::Str(err)});
                    }
                    let finalp = out_dir + "/" + fname;
                    if !fs_rename(part, finalp) {
                        net_close(h);
                        return Json::Obj({"status": Json::Str("error"), "message": Json::Str("finalize failed")});
                    }
                    net_close(h);
                    return Json::Obj({"status": Json::Str("ok"), "path": Json::Str(finalp)});
                }
                _ => {
                    net_close(h);
                    return Json::Obj({"status": Json::Str("error"), "message": Json::Str("bad offer")});
                }
            }
        }
        _ => {
            net_close(h);
            return Json::Obj({"status": Json::Str("error"), "message": Json::Str("offer recv failed")});
        }
    }
}

// client: push a model to a listening peer — port of push().
host fn push_model(peer: str, model_id: str, model_dir: str) -> Json cap net, crypto, fs {
    let key = p2p_key();
    if key == "" { return Json::Obj({"status": Json::Str("error"), "message": Json::Str("no p2p secret")}); }
    match find_manifest(model_dir, model_id) {
        Json::Obj(mf) => {
            let h = net_tcp_connect(peer);
            if h < 0 { return Json::Obj({"status": Json::Str("error"), "message": Json::Str("connect failed")}); }
            var pf: {str: Json} = {};
            var pr: {str: Json} = {};
            pr["model_id"] = Json::Str(model_id);
            pf["PushRequest"] = Json::Obj(pr);
            if !send_frame(h, key, Json::Obj(pf)) {
                net_close(h);
                return Json::Obj({"status": Json::Str("error"), "message": Json::Str("pushrequest send failed")});
            }
            match recv_frame(h, key) {
                Json::Str(u) => {
                    if u != "PushAccept" {
                        net_close(h);
                        return Json::Obj({"status": Json::Str("error"), "message": Json::Str("push rejected: " + u)});
                    }
                }
                Json::Obj(acc) => {
                    if has(acc, "PushReject") || has(acc, "Reject") {
                        net_close(h);
                        return Json::Obj({"status": Json::Str("error"), "message": Json::Str("push rejected")});
                    }
                    net_close(h);
                    return Json::Obj({"status": Json::Str("error"), "message": Json::Str("expected PushAccept")});
                }
                _ => {
                    net_close(h);
                    return Json::Obj({"status": Json::Str("error"), "message": Json::Str("push accept recv failed")});
                }
            }
            var fname = "";
            var total_bytes = 0;
            var sha = "";
            match mf["file_name"] { Json::Str(x) => { fname = x; } _ => { } }
            match mf["total_bytes"] { Json::Num(n) => { total_bytes = i64(n); } _ => { } }
            match mf["sha256"] { Json::Str(x) => { sha = x; } _ => { } }
            let path = model_dir + "/" + fname;
            let total_chunks = (total_bytes + chunk_size() - 1) / chunk_size();
            var of: {str: Json} = {};
            var oo: {str: Json} = {};
            oo["model_id"] = Json::Str(model_id);
            oo["file_name"] = Json::Str(fname);
            oo["total_bytes"] = Json::Num(f64(total_bytes));
            oo["total_chunks"] = Json::Num(f64(total_chunks));
            oo["chunk_size"] = Json::Num(f64(chunk_size()));
            oo["sha256"] = Json::Str(sha);
            of["Offer"] = Json::Obj(oo);
            if !send_frame(h, key, Json::Obj(of)) {
                net_close(h);
                return Json::Obj({"status": Json::Str("error"), "message": Json::Str("offer send failed")});
            }
            match recv_frame(h, key) {
                Json::Str(u2) => {
                    if u2 != "Accept" {
                        net_close(h);
                        return Json::Obj({"status": Json::Str("error"), "message": Json::Str("offer rejected: " + u2)});
                    }
                }
                Json::Obj(_) => {
                    net_close(h);
                    return Json::Obj({"status": Json::Str("error"), "message": Json::Str("offer rejected")});
                }
                _ => {
                    net_close(h);
                    return Json::Obj({"status": Json::Str("error"), "message": Json::Str("accept recv failed")});
                }
            }
            let err = send_chunks(h, key, path, total_chunks, Json::Null);
            if err != "" {
                net_close(h);
                return Json::Obj({"status": Json::Str("error"), "message": Json::Str(err)});
            }
            send_frame(h, key, Json::Str("Done"));
            net_close(h);
            return Json::Obj({"status": Json::Str("ok"), "model_id": Json::Str(model_id)});
        }
        _ => {
            return Json::Obj({"status": Json::Str("error"), "message": Json::Str("model not found: " + model_id)});
        }
    }
}

// one accepted connection — dispatch on the first frame.
host fn serve_conn(h: i64, key: str, peer: str, model_dir: str) -> str cap fs, net, crypto, proc {
    let first = recv_frame(h, key);
    match first {
        Json::Obj(f) => {
            if has(f, "InferRequest") {
                return handle_infer(h, key, f["InferRequest"], peer);
            }
            if has(f, "Request") {
                return handle_pull(h, key, f["Request"], model_dir);
            }
            if has(f, "PushRequest") {
                return handle_push_receive(h, key, f["PushRequest"], model_dir);
            }
            send_error(h, key, "expected Request, PushRequest, or InferRequest frame");
            return "unexpected first frame";
        }
        _ => {
            return "bad first frame";
        }
    }
}

// the serve loop — bind, accept, handle, repeat. blocking, one conn at a
// time (the Rust spawned a task per conn; the .cy runtime stays serial —
// delegated inference is rare and idempotent).
// single-connection variant — tests and one-shot receives.
host fn serve_once(port: i64, model_dir: str) -> str cap fs, net, crypto, proc {
    let key_hex = p2p_key();
    if key_hex == "" { return "no p2p secret"; }
    let l = net_tcp_listen("127.0.0.1:" + str(port));
    if l < 0 { return "listen failed"; }
    let h = net_accept(l);
    if h < 0 { return "accept failed"; }
    let r = serve_conn(h, key_hex, "peer", model_dir);
    net_close(h);
    net_close(l);
    return r;
}

host fn serve(port: i64, model_dir: str) -> str cap fs, net, crypto, proc {
    let key_hex = p2p_key();
    if key_hex == "" { return "no p2p secret"; }
    let l = net_tcp_listen("0.0.0.0:" + str(port));
    if l < 0 { return "failed to bind P2P port " + str(port); }
    while true {
        let h = net_accept(l);
        if h >= 0 {
            serve_conn(h, key_hex, "peer", model_dir);
            net_close(h);
        }
    }
    return "listener closed";
}
