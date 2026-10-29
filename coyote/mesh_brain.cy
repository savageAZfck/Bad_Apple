// mesh_brain.cy — pipeline-parallel mesh client + shard tools,
// ported from src/mesh_brain.rs.
//
// Wire: [u32 BE jsonLen][json frame][optional payload] per packet; with a
// session cipher the packet is [u32 ctLen][nonce||AES-GCM(jsonLen||json||payload)].
// Mutual auth: server challenges with a nonce, client proves
// hmac(secret, "mb-c:"+nonce), server counter-proves "mb-s:"+nonce.
// Session key = sha256(secret) — same derivation as the p2p engram cipher.

import "sys.cy";
import "strlib.cy";

cap fs, net, crypto

// ── types ──────────────────────────────────────────────────────────────

struct ShardSpec {
    rank: i64,
    world: i64,
    addr: str,
    next_host: str,        // "" = none
    layer_start: i64,
    layer_end: i64,
    has_embed: bool,
    has_head: bool,
    tie_embeddings: bool,
}

struct MeshPlan {
    model: str,
    world: i64,
    ranks: [ShardSpec],
}

// a tensor kept by a shard — new key + byte range into the source blob
struct TensorEntry {
    new_key: str,
    dtype: str,
    shape: [i64],
    data_begin: i64,
    data_end: i64,
}

// ── config + plan ──────────────────────────────────────────────────────

host det fn json_num(o: {str: Json}, k: str, d: f64) -> f64 {
    if has(o, k) {
        match o[k] { Json::Num(n) => { return n; } _ => { } }
    }
    return d;
}

host det fn json_bool(o: {str: Json}, k: str) -> bool {
    if has(o, k) {
        match o[k] { Json::Bool(b) => { return b; } _ => { } }
    }
    return false;
}

// num_hidden_layers at top level or under text_config. -1 when absent.
host det fn cfg_layers(cfg: {str: Json}) -> i64 {
    var n = i64(json_num(cfg, "num_hidden_layers", -1.0));
    if n < 0 {
        if has(cfg, "text_config") {
            match cfg["text_config"] {
                Json::Obj(t) => { n = i64(json_num(t, "num_hidden_layers", -1.0)); }
                _ => { }
            }
        }
    }
    return n;
}

// "<prefix>layers.<N>.<rest>" → N or -1. Returns the pieces via Json.
host det fn layer_re(key: str) -> Json {
    let idx = find_sub(key, "layers.");
    if idx < 0 { return Json::Null; }
    let prefix = slice_str(key, 0, idx);
    let rest = slice_str(key, idx + 7, len(key));
    let dot = find_sub(rest, ".");
    if dot < 0 { return Json::Null; }
    let nstr = slice_str(rest, 0, dot);
    let nf = parse_num(nstr);
    if nf != nf { return Json::Null; }
    return Json::Obj({
        "n": Json::Num(nf),
        "prefix": Json::Str(prefix),
        "rest": Json::Str(slice_str(rest, dot + 1, len(rest))),
    });
}

host fn plan(model_dir: str, hosts: [str], mem_gb: [f64]) -> Json cap fs {
    let cfgj = json_parse(read_file(model_dir + "/config.json"));
    match cfgj {
        Json::Obj(cfg) => {
            let total = cfg_layers(cfg);
            if total < 0 {
                return Json::Obj({ "error": Json::Str("config.json has no num_hidden_layers") });
            }
            if total == 0 {
                return Json::Obj({ "error": Json::Str("num_hidden_layers is 0") });
            }
            let world = len(hosts);
            if world == 0 {
                return Json::Obj({ "error": Json::Str("need at least one host") });
            }
            if total < world {
                return Json::Obj({ "error": Json::Str("model has " + str(total) + " layers — fewer than " + str(world) + " nodes") });
            }
            let hidden = json_num(cfg, "hidden_size", 4096.0);
            let vocab = json_num(cfg, "vocab_size", 32000.0);
            let layer_gb = (hidden * hidden * 12.0 * 0.5) / 1073741824.0;
            let edge_gb = (vocab * hidden * 2.0 * 2.0) / 1073741824.0;
            var div = layer_gb;
            if div < 0.001 { div = 0.001; }
            var edge_layers = i64(edge_gb / div);
            if f64(edge_layers) < edge_gb / div { edge_layers = edge_layers + 1; } // ceil

            var weights: [f64] = [];
            var wsum = 0.0;
            if len(mem_gb) == world && world > 0 {
                for i in 0..world { push(weights, mem_gb[i]); }
                if world > 1 {
                    let budget = f64(edge_layers) * layer_gb;
                    weights[0] = weights[0] - budget;
                    if weights[0] < 1.0 { weights[0] = 1.0; }
                    weights[world - 1] = weights[world - 1] - budget;
                    if weights[world - 1] < 1.0 { weights[world - 1] = 1.0; }
                }
            } else {
                for i in 0..world { push(weights, 1.0); }
                if world > 1 {
                    weights[0] = weights[0] - f64(edge_layers) / f64(total);
                    weights[world - 1] = weights[world - 1] - f64(edge_layers) / f64(total);
                }
            }
            for i in 0..world { wsum = wsum + weights[i]; }

            let tied = json_bool(cfg, "tie_word_embeddings");
            var ranks: [ShardSpec] = [];
            var cursor = 0;
            for i in 0..world {
                let remaining = total - cursor;
                let nodes_left = world - i;
                var count = remaining;
                if i < world - 1 {
                    // round() halves away from zero — positives only here
                    count = i64((weights[i] / wsum) * f64(total) + 0.5);
                    var hi = remaining - (nodes_left - 1);
                    if count < 1 { count = 1; }
                    if count > hi { count = hi; }
                }
                var nxt = "";
                if i + 1 < world { nxt = hosts[i + 1]; }
                push(ranks, ShardSpec {
                    rank: i,
                    world: world,
                    addr: hosts[i],
                    next_host: nxt,
                    layer_start: cursor,
                    layer_end: cursor + count,
                    has_embed: i == 0,
                    has_head: i == world - 1,
                    tie_embeddings: tied,
                });
                cursor = cursor + count;
            }
            return Json::Obj({
                "plan": Json::Str("ok"),
                "model": Json::Str(model_dir),
                "world": Json::Num(f64(world)),
                "ranks": Json::Arr(specs_json(ranks)),
            });
        }
        _ => {
            return Json::Obj({ "error": Json::Str("config.json is not a JSON object") });
        }
    }
}

host det fn specs_json(ranks: [ShardSpec]) -> [Json] {
    var acc: [Json] = [];
    for i in 0..len(ranks) {
        let r = ranks[i];
        var o: {str: Json} = {};
        o["rank"] = Json::Num(f64(r.rank));
        o["world"] = Json::Num(f64(r.world));
        o["host"] = Json::Str(r.addr);
        if r.next_host != "" { o["next_host"] = Json::Str(r.next_host); }
        o["layer_start"] = Json::Num(f64(r.layer_start));
        o["layer_end"] = Json::Num(f64(r.layer_end));
        o["has_embed"] = Json::Bool(r.has_embed);
        o["has_head"] = Json::Bool(r.has_head);
        o["tie_embeddings"] = Json::Bool(r.tie_embeddings);
        push(acc, Json::Obj(o));
    }
    return acc;
}

// key belongs to this rank? returns the (possibly renumbered) local key
// or Json::Null when the weight is dropped.
host det fn remap_key(key: str, layer_start: i64, layer_end: i64) -> Json {
    match layer_re(key) {
        Json::Obj(l) => {
            let n = i64(json_num(l, "n", -1.0));
            if n >= layer_start && n < layer_end {
                var pfx = "";
                var rst = "";
                match l["prefix"] { Json::Str(x) => { pfx = x; } _ => { } }
                match l["rest"] { Json::Str(x) => { rst = x; } _ => { } }
                return Json::Str(pfx + "layers." + str(n - layer_start) + "." + rst);
            }
            return Json::Null;
        }
        _ => { return Json::Str(key); }
    }
}

// ── safetensors shard writer ───────────────────────────────────────────
// hex-str copy model — correct for any size the hex fits in memory; the
// streaming 8 MiB-chunk path of the Rust is a shell optimization that
// stays available behind the same entry point on huge models.

// reads a safetensors blob (as hex), keeps owned keys, renumbers layers.
// returns Json {header_hex, body_hex, keys:[str]} or {error}
host fn shard_safetensors(src_hex: str, layer_start: i64, layer_end: i64) -> Json cap fs {
    if len(src_hex) < 16 {
        return Json::Obj({ "error": Json::Str("safetensors too short") });
    }
    let hlen = hex_le_i64(slice_str(src_hex, 0, 16));
    if hlen < 0 || 16 + hlen * 2 > len(src_hex) {
        return Json::Obj({ "error": Json::Str("bad safetensors header length") });
    }
    let header_text = hex_decode(slice_str(src_hex, 16, 16 + hlen * 2));
    let data_off_hex = 16 + hlen * 2;
    match json_parse(header_text) {
        Json::Obj(map) => {
            var kept: [TensorEntry] = [];
            let ks = keys(map);
            for i in 0..len(ks) {
                let key = ks[i];
                if key == "__metadata__" { continue; }
                match remap_key(key, layer_start, layer_end) {
                    Json::Str(nk) => {
                        match map[key] {
                            Json::Obj(meta) => {
                                var dt = "BF16";
                                if has(meta, "dtype") {
                                    match meta["dtype"] { Json::Str(d) => { dt = d; } _ => { } }
                                }
                                var shape: [i64] = [];
                                if has(meta, "shape") {
                                    match meta["shape"] {
                                        Json::Arr(a) => {
                                            for j in 0..len(a) {
                                                match a[j] { Json::Num(n) => { push(shape, i64(n)); } _ => { } }
                                            }
                                        }
                                        _ => { }
                                    }
                                }
                                var b0 = 0;
                                var b1 = 0;
                                if has(meta, "data_offsets") {
                                    match meta["data_offsets"] {
                                        Json::Arr(o) => {
                                            if len(o) >= 2 {
                                                match o[0] { Json::Num(n) => { b0 = i64(n); } _ => { } }
                                                match o[1] { Json::Num(n) => { b1 = i64(n); } _ => { } }
                                            }
                                        }
                                        _ => { }
                                    }
                                }
                                push(kept, TensorEntry {
                                    new_key: nk, dtype: dt, shape: shape,
                                    data_begin: b0, data_end: b1,
                                });
                            }
                            _ => { }
                        }
                    }
                    _ => { }
                }
            }
            if len(kept) == 0 {
                var none: [Json] = [];
                return Json::Obj({ "header_hex": Json::Str(""), "body_hex": Json::Str(""), "keys": Json::Arr(none) });
            }
            // sort by data_begin — insertion order on a small vec
            var sorted: [TensorEntry] = [];
            var used: {str: bool} = {};
            while len(sorted) < len(kept) {
                var bi = -1;
                var bv = 0;
                for i in 0..len(kept) {
                    if has(used, str(i)) { continue; }
                    if bi < 0 || kept[i].data_begin < bv {
                        bi = i;
                        bv = kept[i].data_begin;
                    }
                }
                used[str(bi)] = true;
                push(sorted, kept[bi]);
            }
            // rebuild header with re-based offsets
            var new_map: {str: Json} = {};
            var cur = 0;
            var body_hex = "";
            var kept_keys: [Json] = [];
            for i in 0..len(sorted) {
                let e = sorted[i];
                let blen = e.data_end - e.data_begin;
                var m: {str: Json} = {};
                m["dtype"] = Json::Str(e.dtype);
                var shp: [Json] = [];
                for j in 0..len(e.shape) { push(shp, Json::Num(f64(e.shape[j]))); }
                m["shape"] = Json::Arr(shp);
                m["data_offsets"] = Json::Arr([Json::Num(f64(cur)), Json::Num(f64(cur + blen))]);
                new_map[e.new_key] = Json::Obj(m);
                // slice the tensor bytes from the source blob
                body_hex = body_hex + slice_str(src_hex,
                    data_off_hex + e.data_begin * 2,
                    data_off_hex + e.data_end * 2);
                cur = cur + blen;
                push(kept_keys, Json::Str(e.new_key));
            }
            let new_header = hex_encode(json_emit(Json::Obj(new_map)));
            return Json::Obj({
                "header_hex": Json::Str(i64_le_hex(len(new_header) / 2, 8) + new_header),
                "body_hex": Json::Str(body_hex),
                "keys": Json::Arr(kept_keys),
            });
        }
        _ => {
            return Json::Obj({ "error": Json::Str("bad safetensors header") });
        }
    }
}

// write a shard dir from a model dir. mirrors build_shard(): patched
// config.json, shard.json, mesh_brain_rank.json (camelCase Codable keys),
// re-keyed safetensors, regenerated index, aux copy.
// NOTE: hex-str based — correct, memory-bound on multi-GB models.
host fn build_shard(model_dir: str, out_dir: str, spec: ShardSpec) -> Json cap fs {
    fs_mkdir(out_dir);
    let cfgj = json_parse(read_file(model_dir + "/config.json"));
    match cfgj {
        Json::Obj(cfg) => {
            cfg["num_hidden_layers"] = Json::Num(f64(spec.layer_end - spec.layer_start));
            var shard_meta: {str: Json} = {};
            shard_meta["rank"] = Json::Num(f64(spec.rank));
            shard_meta["layer_start"] = Json::Num(f64(spec.layer_start));
            shard_meta["layer_end"] = Json::Num(f64(spec.layer_end));
            cfg["badapple_shard"] = Json::Obj(shard_meta);
            write_file(out_dir + "/config.json", json_emit(Json::Obj(cfg)));

            var sj: {str: Json} = {};
            sj["rank"] = Json::Num(f64(spec.rank));
            sj["world"] = Json::Num(f64(spec.world));
            sj["host"] = Json::Str(spec.addr);
            if spec.next_host != "" { sj["next_host"] = Json::Str(spec.next_host); }
            sj["layer_start"] = Json::Num(f64(spec.layer_start));
            sj["layer_end"] = Json::Num(f64(spec.layer_end));
            sj["has_embed"] = Json::Bool(spec.has_embed);
            sj["has_head"] = Json::Bool(spec.has_head);
            sj["tie_embeddings"] = Json::Bool(spec.tie_embeddings);
            write_file(out_dir + "/shard.json", json_emit(Json::Obj(sj)));

            var rk: {str: Json} = {};
            rk["rank"] = Json::Num(f64(spec.rank));
            rk["world"] = Json::Num(f64(spec.world));
            rk["host"] = Json::Str(spec.addr);
            if spec.next_host != "" { rk["nextHost"] = Json::Str(spec.next_host); }
            rk["hasEmbed"] = Json::Bool(spec.has_embed);
            rk["hasHead"] = Json::Bool(spec.has_head);
            rk["layerStart"] = Json::Num(f64(spec.layer_start));
            rk["layerEnd"] = Json::Num(f64(spec.layer_end));
            write_file(out_dir + "/mesh_brain_rank.json", json_emit(Json::Obj(rk)));

            var tensors = 0;
            var weight_map: {str: Json} = {};
            let entries = fs_list(model_dir);
            for i in 0..len(entries) {
                let fname = entries[i];
                if !ends_with(fname, ".safetensors") { continue; }
                let src_hex = fs_read_hex(model_dir + "/" + fname);
                match shard_safetensors(src_hex, spec.layer_start, spec.layer_end) {
                    Json::Obj(res) => {
                        if has(res, "error") {
                            return Json::Obj({ "error": res["error"] });
                        }
                        var hh = "";
                        var bb = "";
                        match res["header_hex"] { Json::Str(x) => { hh = x; } _ => { } }
                        match res["body_hex"] { Json::Str(x) => { bb = x; } _ => { } }
                        fs_write_hex(out_dir + "/" + fname, hh + bb);
                        match res["keys"] {
                            Json::Arr(ks) => {
                                for j in 0..len(ks) {
                                    match ks[j] {
                                        Json::Str(k) => {
                                            weight_map[k] = Json::Str(fname);
                                            tensors = tensors + 1;
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
            }
            if tensors == 0 {
                return Json::Obj({ "error": Json::Str("shard produced no tensors — check layer range and weight naming") });
            }
            if fs_exists(model_dir + "/model.safetensors.index.json") {
                var idx: {str: Json} = {};
                var meta: {str: Json} = {};
                meta["total_size"] = Json::Num(0.0);
                idx["metadata"] = Json::Obj(meta);
                idx["weight_map"] = Json::Obj(weight_map);
                idx["_badapple_tensors"] = Json::Num(f64(tensors));
                write_file(out_dir + "/model.safetensors.index.json", json_emit(Json::Obj(idx)));
            }
            // aux files: everything except safetensors/config/index; tokenizer
            // only on edge ranks
            for i in 0..len(entries) {
                let name = entries[i];
                if ends_with(name, ".safetensors") || name == "config.json"
                    || ends_with(name, ".safetensors.index.json") {
                    continue;
                }
                let tok = find_sub(name, "tokenizer") >= 0 || name == "vocab.json"
                    || name == "merges.txt" || name == "special_tokens_map.json";
                if tok && !(spec.has_embed || spec.has_head) { continue; }
                let data = fs_read_hex(model_dir + "/" + name);
                if data != "" {
                    fs_write_hex(out_dir + "/" + name, data);
                }
            }
            return Json::Obj({ "tensors": Json::Num(f64(tensors)) });
        }
        _ => {
            return Json::Obj({ "error": Json::Str("no config.json") });
        }
    }
}

// ── secrets + hosts registry ───────────────────────────────────────────

host fn resolve_mesh_secret() -> str cap fs {
    var raw = env_get("BADAPPLE_MESH_KEY");
    if raw == "" {
        raw = env_get("BADAPPLE_P2P_SECRET");
        if raw == "" {
            let kp = env_get("BADAPPLE_SLICKS_KEY_PATH");
            var path = kp;
            if kp == "" { path = "/var/lib/bad_apple/slicks.key"; }
            if !fs_exists(path) { return ""; }
            raw = trim(read_file(path));
        }
    }
    if len(raw) < 32 { return ""; }
    return raw;
}

host det fn hosts_file() -> str {
    let v = env_get("BADAPPLE_MESH_HOSTS_FILE");
    if v != "" { return v; }
    return "/var/lib/bad_apple/mesh_hosts.json";
}

host fn load_hosts() -> [str] cap fs {
    var acc: [str] = [];
    if !fs_exists(hosts_file()) { return acc; }
    match json_parse(read_file(hosts_file())) {
        Json::Obj(o) => {
            if has(o, "hosts") {
                match o["hosts"] {
                    Json::Arr(a) => {
                        for i in 0..len(a) {
                            match a[i] { Json::Str(s) => { push(acc, s); } _ => { } }
                        }
                    }
                    _ => { }
                }
            }
        }
        _ => { }
    }
    return acc;
}

// ── wire io — frames and packets ───────────────────────────────────────

host det fn mb_hmac(secret: str, msg: str) -> str {
    return hmac_sha256b(hex_encode(secret), hex_encode(msg));
}

host det fn session_key_hex(secret: str) -> str {
    return sha256(secret); // sha256 hex → the aead key hex directly
}

host det fn mb_auth_enabled() -> bool {
    return env_get("BADAPPLE_MESH_AUTH") != "0";
}

host det fn mb_enc_enabled() -> bool {
    return env_get("BADAPPLE_MESH_ENC") != "0";
}

// frame as Json object — op always present, optional fields omitted
host det fn mb_frame(op: str) -> Json {
    var o: {str: Json} = {};
    o["op"] = Json::Str(op);
    return Json::Obj(o);
}

// read exactly n bytes (returns hex, "" on eof/err)
host fn read_exact(h: i64, n: i64) -> str cap net {
    var acc = "";
    var need = n * 2;
    while len(acc) < need {
        let chunk = net_recv(h, (need - len(acc)) / 2);
        if chunk == "" { return ""; }
        acc = acc + chunk;
    }
    return acc;
}

// u32-BE len-prefixed JSON frame
host fn write_mb_frame(h: i64, f: Json) -> bool cap net {
    let j = json_emit(f);
    return net_send(h, u32be_hex(len(j)) + hex_encode(j));
}

host fn read_mb_frame(h: i64) -> Json cap net {
    let lh = read_exact(h, 4);
    if lh == "" { return Json::Null; }
    let n = hex_to_i64(lh);
    if n < 0 || n > 65536 { return Json::Null; }
    let body = read_exact(h, n);
    if body == "" { return Json::Null; }
    return json_parse(hex_decode(body));
}

host det fn dtype_size(name: str) -> i64 {
    if name == "float64" || name == "int64" || name == "uint64" { return 8; }
    if name == "float32" || name == "int32" || name == "uint32" { return 4; }
    if name == "bool" || name == "uint8" || name == "int8" { return 1; }
    return 2;
}

// packet = frame + raw payload, or AES-GCM'd [jsonLen||json||payload]
host fn write_packet(h: i64, f: Json, payload_hex: str, key_hex: str) -> bool cap net, crypto {
    let j = json_emit(f);
    if key_hex != "" {
        let inner = u32be_hex(len(j)) + hex_encode(j) + payload_hex;
        let ct = aead_seal(key_hex, inner);
        if ct == "" { return false; }
        return net_send(h, u32be_hex(len(ct) / 2) + ct);
    }
    return net_send(h, u32be_hex(len(j)) + hex_encode(j) + payload_hex);
}

// → Json {frame: Json, payload: str(hex)} or {error}
host fn read_packet(h: i64, key_hex: str) -> Json cap net, crypto {
    if key_hex != "" {
        let lh = read_exact(h, 4);
        if lh == "" { return Json::Obj({ "error": Json::Str("eof") }); }
        let n = hex_to_i64(lh);
        if n < 0 || n > 536870912 {
            return Json::Obj({ "error": Json::Str("encrypted frame too large") });
        }
        let ct = read_exact(h, n);
        if ct == "" { return Json::Obj({ "error": Json::Str("eof") }); }
        let inner = aead_open(key_hex, ct);
        if len(inner) < 8 {
            return Json::Obj({ "error": Json::Str("decrypted frame too short") });
        }
        let jlen = hex_to_i64(slice_str(inner, 0, 8));
        if jlen < 0 || len(inner) < 8 + jlen * 2 {
            return Json::Obj({ "error": Json::Str("decrypted frame truncated") });
        }
        let fj = json_parse(hex_decode(slice_str(inner, 8, 8 + jlen * 2)));
        return Json::Obj({
            "frame": fj,
            "payload": Json::Str(slice_str(inner, 8 + jlen * 2, len(inner))),
        });
    }
    let fj = read_mb_frame(h);
    match fj {
        Json::Obj(o) => {
            var pay = "";
            var shape_elems = 0;
            var ds = 2;
            if has(o, "shape") {
                match o["shape"] {
                    Json::Arr(s) => {
                        shape_elems = 1;
                        for i in 0..len(s) {
                            match s[i] { Json::Num(n) => { shape_elems = shape_elems * i64(n); } _ => { } }
                        }
                    }
                    _ => { }
                }
            }
            if has(o, "dtype") {
                match o["dtype"] { Json::Str(d) => { ds = dtype_size(d); } _ => { } }
            }
            if shape_elems > 0 {
                pay = read_exact(h, shape_elems * ds);
            }
            return Json::Obj({ "frame": fj, "payload": Json::Str(pay) });
        }
        _ => {
            return Json::Obj({ "error": Json::Str("bad frame") });
        }
    }
}

// ── handshakes ─────────────────────────────────────────────────────────

// client side — returns "" on success (empty error string), else message
host fn client_handshake(h: i64, secret: str) -> str cap net {
    match read_mb_frame(h) {
        Json::Obj(ch) => {
            var op = "";
            var nonce = "";
            if has(ch, "op") { match ch["op"] { Json::Str(x) => { op = x; } _ => { } } }
            if has(ch, "text") { match ch["text"] { Json::Str(x) => { nonce = x; } _ => { } } }
            if op != "auth" { return "expected auth challenge, got " + op; }
            if nonce == "" { return "auth challenge missing nonce"; }
            var f: {str: Json} = {};
            f["op"] = Json::Str("auth");
            f["text"] = Json::Str(mb_hmac(secret, "mb-c:" + nonce));
            if !write_mb_frame(h, Json::Obj(f)) { return "write failed"; }
            match read_mb_frame(h) {
                Json::Obj(r) => {
                    var rop = "";
                    var rtext = "";
                    var rerr = "";
                    if has(r, "op") { match r["op"] { Json::Str(x) => { rop = x; } _ => { } } }
                    if has(r, "text") { match r["text"] { Json::Str(x) => { rtext = x; } _ => { } } }
                    if has(r, "error") { match r["error"] { Json::Str(x) => { rerr = x; } _ => { } } }
                    if rop == "auth-ok" {
                        let expect = mb_hmac(secret, "mb-s:" + nonce);
                        if rtext != expect {
                            return "server counter-proof invalid — wrong mesh secret?";
                        }
                        return "";
                    }
                    if rop == "error" { return "auth rejected: " + rerr; }
                    return "expected auth-ok, got " + rop;
                }
                _ => { return "bad auth-ok frame"; }
            }
        }
        _ => { return "bad challenge frame"; }
    }
}

// server side — challenge, verify client proof, counter-prove
host fn server_handshake(h: i64, secret: str) -> str cap net, crypto {
    let nonce = rand_hex(32);
    var ch: {str: Json} = {};
    ch["op"] = Json::Str("auth");
    ch["text"] = Json::Str(nonce);
    if !write_mb_frame(h, Json::Obj(ch)) { return "write failed"; }
    match read_mb_frame(h) {
        Json::Obj(r) => {
            var op = "";
            var text = "";
            if has(r, "op") { match r["op"] { Json::Str(x) => { op = x; } _ => { } } }
            if has(r, "text") { match r["text"] { Json::Str(x) => { text = x; } _ => { } } }
            let expect = mb_hmac(secret, "mb-c:" + nonce);
            if op != "auth" || text != expect { return "peer auth failed"; }
            var ok: {str: Json} = {};
            ok["op"] = Json::Str("auth-ok");
            ok["text"] = Json::Str(mb_hmac(secret, "mb-s:" + nonce));
            if !write_mb_frame(h, Json::Obj(ok)) { return "write failed"; }
            return "";
        }
        _ => { return "bad response frame"; }
    }
}

// open an authenticated conn — returns {h, key} or {error}
host fn mesh_connect(addr: str) -> Json cap fs, net, crypto {
    let h = net_tcp_connect(addr);
    if h < 0 {
        return Json::Obj({ "error": Json::Str("connect " + addr + " failed") });
    }
    if mb_auth_enabled() {
        let secret = resolve_mesh_secret();
        if secret == "" {
            net_close(h);
            return Json::Obj({ "error": Json::Str("no mesh secret") });
        }
        let e = client_handshake(h, secret);
        if e != "" {
            net_close(h);
            return Json::Obj({ "error": Json::Str(e) });
        }
        if mb_enc_enabled() {
            return Json::Obj({ "h": Json::Num(f64(h)), "key": Json::Str(session_key_hex(secret)) });
        }
    }
    return Json::Obj({ "h": Json::Num(f64(h)), "key": Json::Str("") });
}

// ── ops ────────────────────────────────────────────────────────────────

// ping → Json {rank, layer_start, layer_end} or {error}
host fn ping(addr: str) -> Json cap fs, net, crypto {
    match mesh_connect(addr) {
        Json::Obj(c) => {
            if has(c, "error") { return Json::Obj({ "error": c["error"] }); }
            var h = -1;
            var key = "";
            match c["h"] { Json::Num(n) => { h = i64(n); } _ => { } }
            match c["key"] { Json::Str(x) => { key = x; } _ => { } }
            if !write_packet(h, mb_frame("ping"), "", key) {
                net_close(h);
                return Json::Obj({ "error": Json::Str("write failed") });
            }
            match read_packet(h, key) {
                Json::Obj(r) => {
                    net_close(h);
                    if has(r, "error") { return Json::Obj({ "error": r["error"] }); }
                    match r["frame"] {
                        Json::Obj(f) => {
                            if has(f, "error") {
                                return Json::Obj({ "error": f["error"] });
                            }
                            var res: {str: Json} = {};
                            if has(f, "token") { res["rank"] = f["token"]; }
                            if has(f, "layerStart") { res["layer_start"] = f["layerStart"]; }
                            if has(f, "layerEnd") { res["layer_end"] = f["layerEnd"]; }
                            return Json::Obj(res);
                        }
                        _ => { return Json::Obj({ "error": Json::Str("bad frame") }); }
                    }
                }
                _ => { net_close(h); return Json::Obj({ "error": Json::Str("bad packet") }); }
            }
        }
        _ => { return Json::Obj({ "error": Json::Str("connect failed") }); }
    }
}

// ask the pipeline to generate → Json {text} or {error}
host fn ask(addr: str, prompt: str, max_tokens: i64) -> Json cap fs, net, crypto {
    match mesh_connect(addr) {
        Json::Obj(c) => {
            if has(c, "error") { return Json::Obj({ "error": c["error"] }); }
            var h = -1;
            var key = "";
            match c["h"] { Json::Num(n) => { h = i64(n); } _ => { } }
            match c["key"] { Json::Str(x) => { key = x; } _ => { } }
            var f = mb_frame("generate");
            match f {
                Json::Obj(o) => {
                    o["prompt"] = Json::Str(prompt);
                    o["maxTokens"] = Json::Num(f64(max_tokens));
                    f = Json::Obj(o);
                }
                _ => { }
            }
            if !write_packet(h, f, "", key) {
                net_close(h);
                return Json::Obj({ "error": Json::Str("write failed") });
            }
            match read_packet(h, key) {
                Json::Obj(r) => {
                    net_close(h);
                    if has(r, "error") { return Json::Obj({ "error": r["error"] }); }
                    match r["frame"] {
                        Json::Obj(fr) => {
                            if has(fr, "error") { return Json::Obj({ "error": fr["error"] }); }
                            if has(fr, "text") {
                                return Json::Obj({ "text": fr["text"] });
                            }
                            return Json::Obj({ "error": Json::Str("generate response missing text") });
                        }
                        _ => { return Json::Obj({ "error": Json::Str("bad frame") }); }
                    }
                }
                _ => { net_close(h); return Json::Obj({ "error": Json::Str("bad packet") }); }
            }
        }
        _ => { return Json::Obj({ "error": Json::Str("connect failed") }); }
    }
}

// step → Json {token, done} or {error}
host fn step(addr: str, hidden_hex: str, shape: [i64], dtype: str) -> Json cap fs, net, crypto {
    match mesh_connect(addr) {
        Json::Obj(c) => {
            if has(c, "error") { return Json::Obj({ "error": c["error"] }); }
            var h = -1;
            var key = "";
            match c["h"] { Json::Num(n) => { h = i64(n); } _ => { } }
            match c["key"] { Json::Str(x) => { key = x; } _ => { } }
            var f = mb_frame("step");
            match f {
                Json::Obj(o) => {
                    var shp: [Json] = [];
                    for i in 0..len(shape) { push(shp, Json::Num(f64(shape[i]))); }
                    o["shape"] = Json::Arr(shp);
                    o["dtype"] = Json::Str(dtype);
                    f = Json::Obj(o);
                }
                _ => { }
            }
            if !write_packet(h, f, hidden_hex, key) {
                net_close(h);
                return Json::Obj({ "error": Json::Str("write failed") });
            }
            match read_packet(h, key) {
                Json::Obj(r) => {
                    net_close(h);
                    if has(r, "error") { return Json::Obj({ "error": r["error"] }); }
                    match r["frame"] {
                        Json::Obj(fr) => {
                            if has(fr, "error") { return Json::Obj({ "error": fr["error"] }); }
                            var tok = -1;
                            var done = false;
                            if has(fr, "token") { match fr["token"] { Json::Num(n) => { tok = i64(n); } _ => { } } }
                            if has(fr, "done") { match fr["done"] { Json::Bool(b) => { done = b; } _ => { } } }
                            return Json::Obj({ "token": Json::Num(f64(tok)), "done": Json::Bool(done) });
                        }
                        _ => { return Json::Obj({ "error": Json::Str("bad frame") }); }
                    }
                }
                _ => { net_close(h); return Json::Obj({ "error": Json::Str("bad packet") }); }
            }
        }
        _ => { return Json::Obj({ "error": Json::Str("connect failed") }); }
    }
}
