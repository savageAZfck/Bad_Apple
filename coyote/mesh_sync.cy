// mesh_sync.cy — cross-device document sync, ported from src/mesh_sync.rs.
//
// Small encrypted documents (personas, prompts, settings, checkpoints,
// dream shares) replicated across peers as CompactEngramPacket payloads.
// Conflict resolution: last-write-wins by (timestamp, version). One JSON
// file per document under the mesh root; saves are atomic (tmp + rename).
// Wire format preserved: serde variant names capitalized, packets tagged
// "op", payloads base64'd JSON.

import "sys.cy";
import "strlib.cy";

cap fs, crypto, proc

// ── doc kinds — serde variant names on the wire ────────────────────────

host det fn kind_wire(kind: str) -> str {
    if kind == "personas" { return "Personas"; }
    if kind == "prompt" { return "Prompt"; }
    if kind == "settings" { return "Settings"; }
    if kind == "model_manifests" { return "ModelManifests"; }
    if kind == "sovereign_checkpoint" { return "SovereignCheckpoint"; }
    if kind == "ledger_checkpoint" { return "LedgerCheckpoint"; }
    if kind == "dream_share" { return "DreamShare"; }
    return "";
}

host det fn kind_of(wire: str) -> str {
    if wire == "Personas" { return "personas"; }
    if wire == "Prompt" { return "prompt"; }
    if wire == "Settings" { return "settings"; }
    if wire == "ModelManifests" { return "model_manifests"; }
    if wire == "SovereignCheckpoint" { return "sovereign_checkpoint"; }
    if wire == "LedgerCheckpoint" { return "ledger_checkpoint"; }
    if wire == "DreamShare" { return "dream_share"; }
    // the FromStr aliases
    if wire == "prompts" { return "prompt"; }
    if wire == "models" { return "model_manifests"; }
    if wire == "checkpoint" || wire == "checkpoints" { return "sovereign_checkpoint"; }
    if wire == "dream" || wire == "dreams" { return "dream_share"; }
    return "";
}

host det fn default_filename(kind: str) -> str {
    if kind == "personas" { return "personas.json"; }
    if kind == "prompt" { return "prompt.txt"; }
    if kind == "settings" { return "settings.json"; }
    if kind == "model_manifests" { return "model_manifests.json"; }
    if kind == "sovereign_checkpoint" { return "sovereign_checkpoint.json"; }
    if kind == "ledger_checkpoint" { return "ledger_checkpoint.json"; }
    if kind == "dream_share" { return "dream_share.json"; }
    return "";
}

// data_dir = ~/Library/Application Support (macOS dirs::data_dir)
host det fn data_base() -> str {
    let home = env_get("HOME");
    if home == "" { return "/var/lib/bad_apple"; }
    return home + "/Library/Application Support/bad_apple";
}

host det fn local_source(kind: str) -> str {
    if kind == "sovereign_checkpoint" {
        return "/var/lib/bad_apple/ledger.sovereign.checkpoint.json";
    }
    if kind == "ledger_checkpoint" {
        return "/var/lib/bad_apple/ledger_checkpoint.json";
    }
    if kind == "dream_share" {
        return "/var/lib/bad_apple/dream_share.json";
    }
    return data_base() + "/" + default_filename(kind);
}

host det fn mesh_root() -> str {
    return data_base() + "/mesh";
}

host det fn doc_id(kind: str, origin: str) -> str {
    return kind + "@" + origin;
}

host det fn sanitize(s: str) -> str {
    let ok = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.";
    var acc = "";
    for i in 0..len(s) {
        let c = slice_str(s, i, i + 1);
        if contains(ok, c) { acc = acc + c; } else { acc = acc + "_"; }
    }
    return acc;
}

// ── store ──────────────────────────────────────────────────────────────

// the store is the directory — no resident cache needed; reads hit disk.
// merge/save/load semantics match MeshStore exactly.

host fn doc_path(id: str) -> str cap fs {
    return mesh_root() + "/" + sanitize(id) + ".json";
}

// doc → Json on the wire: {kind: <VariantName>, body, timestamp, origin, version}
host fn load_doc(id: str) -> Json cap fs {
    let p = doc_path(id);
    if !fs_exists(p) { return Json::Null; }
    return json_parse(read_file(p));
}

// last-write-wins: (timestamp, version) — mirrors merge()
host fn merge_doc(doc: Json) -> bool cap fs {
    match doc {
        Json::Obj(d) => {
            let id = jstr(d, "kind") + "@" + jstr(d, "origin");
            // wire kind → the stored doc_id uses the kind's DISPLAY name
            let kind = kind_of(jstr(d, "kind"));
            let real_id = kind + "@" + jstr(d, "origin");
            var ts = jnum(d, "timestamp");
            var ver = jnum(d, "version");
            match load_doc(real_id) {
                Json::Obj(existing) => {
                    let ets = jnum(existing, "timestamp");
                    let ever = jnum(existing, "version");
                    if ts > ets || (ts == ets && ver > ever) {
                        save_doc(real_id, doc);
                        return true;
                    }
                    return false;
                }
                _ => {
                    save_doc(real_id, doc);
                    return true;
                }
            }
        }
        _ => { return false; }
    }
}

host fn jstr(o: {str: Json}, k: str) -> str cap fs {
    if has(o, k) {
        match o[k] { Json::Str(s) => { return s; } _ => { } }
    }
    return "";
}

host det fn jnum_d(o: {str: Json}, k: str) -> i64 {
    if has(o, k) {
        match o[k] { Json::Num(n) => { return i64(n); } _ => { } }
    }
    return 0;
}

host fn jnum(o: {str: Json}, k: str) -> i64 cap fs { return jnum_d(o, k); }

// atomic save — tmp + rename, mirrors MeshStore::save
host fn save_doc(doc_id: str, doc: Json) -> bool cap fs {
    fs_mkdir(mesh_root());
    let path = doc_path(doc_id);
    let tmp = path + ".tmp";
    write_file(tmp, json_emit(doc));
    return fs_rename(tmp, path);
}

// every persisted doc → [Json]
host fn list_docs() -> [Json] cap fs {
    var acc: [Json] = [];
    let names = fs_list(mesh_root());
    for i in 0..len(names) {
        let n = names[i];
        if ends_with(n, ".json") {
            match json_parse(read_file(mesh_root() + "/" + n)) {
                Json::Obj(d) => { push(acc, Json::Obj(d)); }
                _ => { }
            }
        }
    }
    return acc;
}

host fn load_kind(kind: str) -> Json cap fs {
    let docs = list_docs();
    for i in 0..len(docs) {
        match docs[i] {
            Json::Obj(d) => {
                if kind_of(jstr(d, "kind")) == kind {
                    return Json::Obj(d);
                }
            }
            _ => { }
        }
    }
    return Json::Null;
}

// ── packets — {"op":"Push|Pull|Ack", ...} base64'd inside the engram ───

host det fn packet_push(id: str, doc: Json) -> Json {
    var o: {str: Json} = {};
    o["op"] = Json::Str("Push");
    o["doc_id"] = Json::Str(id);
    o["doc"] = doc;
    return Json::Obj(o);
}

host det fn packet_pull(id: str) -> Json {
    return Json::Obj({ "op": Json::Str("Pull"), "doc_id": Json::Str(id) });
}

host det fn packet_ack(id: str, ts: i64) -> Json {
    return Json::Obj({
        "op": Json::Str("Ack"),
        "doc_id": Json::Str(id),
        "timestamp": Json::Num(f64(ts)),
    });
}

host det fn encode_packet(p: Json) -> str {
    return b64_encode(json_emit(p));
}

host det fn decode_packet(payload: str) -> Json {
    let s = b64_decode(payload);
    if s == "" { return Json::Null; }
    return json_parse(s);
}

// ── local doc read + engram wrap ───────────────────────────────────────

host fn hostname() -> str cap fs, proc {
    let v = env_get("BADAPPLE_ORIGIN_INSTANCE");
    if v != "" { return v; }
    var noargs: [str] = [];
    match sys_exec("hostname", noargs) {
        Json::Obj(r) => { return trim(jstr(r, "out")); }
        _ => { return "unknown"; }
    }
}

host fn read_local_doc(kind: str) -> Json cap fs, proc {
    var body = "";
    let path = local_source(kind);
    if fs_exists(path) { body = read_file(path); }
    var o: {str: Json} = {};
    o["kind"] = Json::Str(kind_wire(kind));
    o["body"] = Json::Str(body);
    o["timestamp"] = Json::Num(f64(time_ms() / 1000));
    o["origin"] = Json::Str(hostname());
    o["version"] = Json::Num(1.0);
    return Json::Obj(o);
}

// CompactEngramPacket — payload carries the packet; experiential_text is
// a mesh marker, brain_state is zeros (ENGRAM_DIM=64 in protocol.rs).
host fn build_engram(packet: Json, origin: str) -> Json cap crypto {
    var brain: [Json] = [];
    for i in 0..64 { push(brain, Json::Num(0.0)); }
    var o: {str: Json} = {};
    o["id"] = Json::Num(f64(time_ms() % 9223372036854775807));
    o["timestamp"] = Json::Num(f64(time_ms() / 1000));
    o["experiential_text"] = Json::Str("mesh:" + jstr_d(packet));
    o["emotional_state_snapshot"] = Json::Str("");
    o["origin_instance"] = Json::Str(origin);
    o["brain_state"] = Json::Arr(brain);
    var none: [Json] = [];
    o["embedding"] = Json::Arr(none);
    o["priority"] = Json::Num(255.0);
    o["payload"] = Json::Str(encode_packet(packet));
    return Json::Obj(o);
}

host det fn jstr_d(packet: Json) -> str {
    match packet {
        Json::Obj(o) => {
            if has(o, "doc_id") {
                match o["doc_id"] { Json::Str(s) => { return s; } _ => { } }
            }
            return "";
        }
        _ => { return ""; }
    }
}

// ── incoming — merge a pushed doc / answer a pull / ignore acks ────────
// → Json packet to reply (Push for a known doc, Ack for a merged push)
// or Json::Null when nothing to send.
host fn handle_incoming(engram: Json) -> Json cap fs {
    match engram {
        Json::Obj(e) => {
            if !has(e, "payload") { return Json::Null; }
            var payload = "";
            match e["payload"] { Json::Str(p) => { payload = p; } _ => { } }
            if payload == "" { return Json::Null; }
            match decode_packet(payload) {
                Json::Obj(pkt) => {
                    var op = "";
                    var id = "";
                    match pkt["op"] { Json::Str(x) => { op = x; } _ => { } }
                    if has(pkt, "doc_id") {
                        match pkt["doc_id"] { Json::Str(x) => { id = x; } _ => { } }
                    }
                    if op == "Push" {
                        if has(pkt, "doc") {
                            let doc = pkt["doc"];
                            if merge_doc(doc) {
                                var ts = 0;
                                match doc {
                                    Json::Obj(d) => { ts = jnum(d, "timestamp"); }
                                    _ => { }
                                }
                                return packet_ack(id, ts);
                            }
                            return packet_ack(id, 0);
                        }
                        return Json::Null;
                    }
                    if op == "Pull" {
                        match load_doc(id) {
                            Json::Obj(d) => { return packet_push(id, Json::Obj(d)); }
                            _ => { return Json::Null; }
                        }
                    }
                    return Json::Null; // Ack
                }
                _ => { return Json::Null; }
            }
        }
        _ => { return Json::Null; }
    }
}
