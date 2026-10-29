// scavenger.cy — the grounded file index + recall, in Coyote.
//
// Port of the grounded-index half of src/scavenger.rs: chunks tracked
// text files on byte-ish boundaries, embeds each chunk through
// tensor_brain::semantic_embedding (shell extern), and persists
// chunk/meta records to redb under the same key layout the Rust code
// uses — `grounded:meta:<sha256(path)>` / `grounded:chunk:<sha>:<i>`.
//
// Recall is hybrid: cosine similarity plus a lexical bonus for exact
// identifier hits (the 64-dim space is coarse; code queries hinge on
// literal names). All local, all offline — `cap fs` only, no net.

cap fs

import "sys.cy";
import "strlib.cy";

// ── constants ──────────────────────────────────────────────────────────
host det fn grounded_index_path() -> str { return "/var/lib/bad_apple/grounded_index"; }
host det fn meta_prefix() -> str { return "grounded:meta:"; }
host det fn chunk_prefix() -> str { return "grounded:chunk"; }
host det fn chunk_bytes() -> i64 { return 2048; }
host det fn chunk_overlap() -> i64 { return 256; }
host det fn max_recall_chunks() -> i64 { return 50000; }
host det fn recall_snippet_chars() -> i64 { return 1500; }
host det fn watch_depth() -> i64 { return 4; }
host det fn chunk_byte_limit() -> i64 { return 8388608; }

host det fn tracked_extensions() -> [str] {
    return ["txt", "md", "rs", "py", "c", "cpp", "h", "hpp", "swift", "js",
        "ts", "sh", "bash", "zsh", "json", "yaml", "yml", "toml", "xml",
        "csv", "log", "ini", "cfg"];
}

host det fn skip_dir_names() -> [str] {
    return [".git", "node_modules", "target", "qwen3b_ane_shards",
        "compiled", "logs", ".cache", "__pycache__", "dist", "build",
        "state-backup", "wild_workspace", "sapient_agi_soul"];
}

// ── json helpers ───────────────────────────────────────────────────────
host det fn jstr(o: {str: Json}, k: str) -> str {
    if has(o, k) {
        match o[k] { Json::Str(s) => { return s; } _ => { } }
    }
    return "";
}

host det fn jnumf(o: {str: Json}, k: str) -> f64 {
    if has(o, k) {
        match o[k] { Json::Num(n) => { return n; } _ => { } }
    }
    return 0.0;
}

host det fn jnumi(o: {str: Json}, k: str) -> i64 {
    if has(o, k) {
        match o[k] { Json::Num(n) => { return i64(n); } _ => { } }
    }
    return 0;
}

host det fn jbool(o: {str: Json}, k: str) -> bool {
    if has(o, k) {
        match o[k] { Json::Bool(b) => { return b; } _ => { } }
    }
    return false;
}

// ── path classification ────────────────────────────────────────────────
host det fn file_extension(path: str) -> str {
    let dot = rfind_sub(path, ".");
    let slash = rfind_sub(path, "/");
    if dot < 0 || dot < slash { return ""; }
    return str_lower(slice_str(path, dot + 1, len(path)));
}

host det fn is_tracked_file(path: str) -> bool {
    let ext = file_extension(path);
    let exts = tracked_extensions();
    for i in 0..len(exts) {
        if exts[i] == ext { return true; }
    }
    return false;
}

host det fn is_ignored_path(path: str) -> bool {
    let parts = split(path, "/");
    for i in 0..len(parts) {
        if is_skip_name(parts[i]) { return true; }
    }
    return false;
}

host det fn is_skip_name(p: str) -> bool {
    let skips = skip_dir_names();
    for j in 0..len(skips) {
        if skips[j] == p { return true; }
    }
    return false;
}

host det fn path_under_roots(path: str, roots: [str]) -> bool {
    for i in 0..len(roots) {
        if starts_with(path, roots[i]) { return true; }
    }
    return false;
}

// ── keys ───────────────────────────────────────────────────────────────
host det fn content_hash(s: str) -> str { return sha256(s); }
host det fn path_key(path: str) -> str { return sha256(path); }
host det fn meta_key_of(pkey: str) -> str { return meta_prefix() + pkey; }
host det fn chunk_key_of(pkey: str, index: i64) -> str {
    return chunk_prefix() + ":" + pkey + ":" + str(index);
}

// ── chunking ───────────────────────────────────────────────────────────
// byte-ish bounded overlapping chunks that break on paragraph or line
// boundaries where possible (char offsets — slice_str is char-based).
host det fn grounded_chunk_text(text: str) -> [str] {
    var chunks: [str] = [];
    let n = len(text);
    if n <= chunk_bytes() {
        push(chunks, text);
        return chunks;
    }
    var start = 0;
    while start < n {
        var end = start + chunk_bytes();
        if end > n { end = n; }
        // prefer to end on a paragraph or line break
        if end < n {
            let seg = slice_str(text, start, end);
            let p2 = rfind_sub(seg, "\n\n");
            let p1 = rfind_sub(seg, "\n");
            if p2 > chunk_bytes() / 2 {
                end = start + p2 + 2;
            } else {
                if p1 > chunk_bytes() / 2 { end = start + p1 + 1; }
            }
        }
        push(chunks, slice_str(text, start, end));
        if end >= n { break; }
        var next = end - chunk_overlap();
        if next <= start { next = end; }
        start = next;
    }
    return chunks;
}

// ── indexing ───────────────────────────────────────────────────────────
// GroundedFileMeta: {path, content_hash, chunk_count, mtime, size}
// GroundedChunk:   {path, mtime, chunk_index, text, vector: [u32 bits]}

// index one file; → chunks written (0 = skipped unchanged / unreadable)
host fn index_grounded_file(db: str, path: str, mtime: i64, size: i64) -> i64 cap fs {
    let canon = fs_realpath(path);
    var cp = path;
    if canon != "" { cp = canon; }
    let pkey = path_key(cp);
    let mk = meta_key_of(pkey);

    // fast path: mtime+size unchanged → skip without reading
    var prior_ok = false;
    var prior_chunks = 0;
    var prior_hash = "";
    let prior_raw = kv_get(db, mk);
    if prior_raw != "" {
        match json_parse(prior_raw) {
            Json::Obj(pm) => {
                prior_ok = true;
                prior_chunks = jnumi(pm, "chunk_count");
                prior_hash = jstr(pm, "content_hash");
                if jnumi(pm, "mtime") == mtime && jnumi(pm, "size") == size && mtime != 0 {
                    return 0;
                }
            }
            _ => { }
        }
    }

    let text = fs_read_nofollow(cp, chunk_byte_limit());
    if text == "" { return 0; }
    let current_hash = content_hash(text);

    if prior_ok && prior_hash == current_hash {
        // touched but unchanged: refresh stored mtime/size
        var um: {str: Json} = {};
        um["path"] = Json::Str(cp);
        um["content_hash"] = Json::Str(prior_hash);
        um["chunk_count"] = Json::Num(f64(prior_chunks));
        um["mtime"] = Json::Num(f64(mtime));
        um["size"] = Json::Num(f64(size));
        let _w = kv_put(db, mk, json_emit(Json::Obj(um)));
        return 0;
    }

    // purge any previous version's chunks
    if prior_ok {
        var i = 0;
        while i < prior_chunks {
            let _d = kv_del(db, chunk_key_of(pkey, i));
            i += 1;
        }
    }

    let chunks = grounded_chunk_text(text);
    var written = 0;
    for i in 0..len(chunks) {
        index_one_chunk(db, pkey, cp, mtime, i, chunks[i]);
        written += 1;
    }

    var meta: {str: Json} = {};
    meta["path"] = Json::Str(cp);
    meta["content_hash"] = Json::Str(current_hash);
    meta["chunk_count"] = Json::Num(f64(written));
    meta["mtime"] = Json::Num(f64(mtime));
    meta["size"] = Json::Num(f64(size));
    let _m = kv_put(db, mk, json_emit(Json::Obj(meta)));
    return written;
}

// index_directories_at(dirs, db) → Json {files, chunks}
host fn index_directories_at(dirs: [str], db: str) -> Json cap fs {
    var indexed = 0;
    var chunks = 0;
    for i in 0..len(dirs) {
        let d = dirs[i];
        if fs_is_dir(d) {
            let files = fs_walk(d, watch_depth());
            let part = index_walk_files(files, db);
            match part {
                Json::Obj(pm) => {
                    indexed += jnumi(pm, "files");
                    chunks += jnumi(pm, "chunks");
                }
                _ => { }
            }
        } else {
            if fs_exists(d) && is_tracked_file(d) && !is_ignored_path(d) {
                let n = index_grounded_file(db, d, fs_mtime(d), fs_size(d));
                if n > 0 {
                    indexed += 1;
                    chunks += n;
                }
            }
        }
    }
    var r: {str: Json} = {};
    r["files"] = Json::Num(f64(indexed));
    r["chunks"] = Json::Num(f64(chunks));
    return Json::Obj(r);
}

// ── watch registry ─────────────────────────────────────────────────────
host fn watch_paths_file() -> str {
    let e = env_get("BADAPPLE_WATCH_PATHS");
    if e != "" { return e; }
    return "/var/lib/bad_apple/watch_paths";
}

host fn watched_dirs() -> [str] {
    var out_v: [str] = [];
    let f = watch_paths_file();
    if fs_exists(f) {
        let lines = split(read_file(f), "\n");
        for i in 0..len(lines) {
            let l = trim(lines[i]);
            if len(l) > 0 && slice_str(l, 0, 1) != "#" { push(out_v, l); }
        }
    }
    return out_v;
}

host fn strs_contains(list: [str], v: str) -> bool {
    for i in 0..len(list) {
        if list[i] == v { return true; }
    }
    return false;
}

// register a dir for persistent re-indexing → true when newly added
host fn register_watch(dir: str) -> bool cap fs {
    var canon = fs_realpath(dir);
    if canon == "" { canon = dir; }
    var existing = watched_dirs();
    if strs_contains(existing, canon) { return false; }
    fs_mkdir(dir_of(watch_paths_file()));
    push(existing, canon);
    write_file(watch_paths_file(), join_str(existing, "\n") + "\n");
    return true;
}

// unregister → true when it was registered
host fn unregister_watch(dir: str) -> bool cap fs {
    var canon = fs_realpath(dir);
    if canon == "" { canon = dir; }
    let existing = watched_dirs();
    var kept: [str] = [];
    var removed = false;
    for i in 0..len(existing) {
        if existing[i] == canon { removed = true; } else { push(kept, existing[i]); }
    }
    if !removed { return false; }
    write_file(watch_paths_file(), join_str(kept, "\n") + "\n");
    return true;
}

host det fn join_str(items: [str], sep: str) -> str {
    var acc = "";
    for i in 0..len(items) {
        if i > 0 { acc = acc + sep; }
        acc = acc + items[i];
    }
    return acc;
}

// ── recall ─────────────────────────────────────────────────────────────
host det fn cosine_sim(a: [Json], b: [Json]) -> f64 {
    if len(a) != len(b) || len(a) == 0 { return 0.0; }
    var dot = 0.0;
    var na = 0.0;
    var nb = 0.0;
    for i in 0..len(a) {
        var x = 0.0;
        var y = 0.0;
        match a[i] { Json::Num(n) => { x = f32_from_bits(i64(n)); } _ => { } }
        match b[i] { Json::Num(n) => { y = f32_from_bits(i64(n)); } _ => { } }
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    let denom = math_sqrt(na) * math_sqrt(nb);
    if denom > 0.0 { return dot / denom; }
    return 0.0;
}

host det fn is_alnum(c: str) -> bool {
    return contains("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_", c);
}

host det fn is_upper(c: str) -> bool {
    return contains("ABCDEFGHIJKLMNOPQRSTUVWXYZ", c);
}

// distinctive_terms — lowercase alnum/underscore tokens ≥3 chars minus
// stopwords; snake_case and camelCase tokens also yield their parts.
host det fn distinctive_terms(query: str) -> [str] {
    let stops = ["the","and","for","with","that","this","from","what",
        "which","how","does","are","was","were","you","your","its","his",
        "her","their","our","can","could","should","would","will","shall",
        "may","might","must","not","but","all","any","each","some","into",
        "over","under","between","about","when","where","who","whom","why",
        "use","used","using","code","file","function","module","implement",
        "implemented","implements","implementation","work","works",
        "working","codebase","repo","repository","bad","apple","system"];
    var terms: [str] = [];
    // tokenize on non-alnum/underscore
    var toks: [str] = [];
    var cur = "";
    for i in 0..len(query) {
        let c = slice_str(query, i, i + 1);
        if is_alnum(c) { cur = cur + c; }
        else {
            if len(cur) > 0 { push(toks, cur); cur = ""; }
        }
    }
    if len(cur) > 0 { push(toks, cur); }
    for ti in 0..len(toks) {
        collect_token_terms(toks[ti], stops, terms);
    }
    return terms;
}

// per-token expansion — factored so the loop body stays flat (the host
// compiler mis-slots vars declared in a for body containing nested fors)
host det fn collect_token_terms(tok: str, stops: [str], terms: [str]) -> unit {
    var parts: [str] = [];
    push(parts, tok);
    // snake split
    let subs = split(tok, "_");
    for si in 0..len(subs) {
        if subs[si] != tok { push(parts, subs[si]); }
    }
    // camel split
    var split_case = "";
    var extra: [str] = [];
    for ci in 0..len(tok) {
        let c = slice_str(tok, ci, ci + 1);
        if is_upper(c) && len(split_case) > 0 {
            push(extra, split_case);
            split_case = "";
        }
        split_case = split_case + c;
    }
    if len(split_case) > 0 && split_case != tok { push(extra, split_case); }
    for ei in 0..len(extra) { push(parts, extra[ei]); }
    for pi in 0..len(parts) {
        let t = str_lower(parts[pi]);
        if len(t) >= 3 && !stopword_in(stops, t) && !strs_contains(terms, t) {
            push(terms, t);
        }
    }
}

host det fn stopword_in(stops: [str], t: str) -> bool {
    for i in 0..len(stops) {
        if stops[i] == t { return true; }
    }
    return false;
}

// lexical_bonus — 0.4·(text hits / terms) + 0.8·(path hits / terms)
host det fn lexical_bonus(terms: [str], text: str, path: str) -> f64 {
    if len(terms) == 0 { return 0.0; }
    let hay = str_lower(text);
    let hayp = str_lower(path);
    var text_hits = 0;
    var path_hits = 0;
    for i in 0..len(terms) {
        if contains(hay, terms[i]) { text_hits += 1; }
        if contains(hayp, terms[i]) { path_hits += 1; }
    }
    let n = f64(len(terms));
    return 0.4 * (f64(text_hits) / n) + 0.8 * (f64(path_hits) / n);
}

// recall_at(query, k, db) → Json Arr of {path, chunk_index, score, text}
host fn recall_at(query: str, k: i64, db: str) -> Json cap fs {
    var out_a: [Json] = [];
    if !fs_exists(db) { return Json::Arr(out_a); }
    let qemb = scav_embed(query);
    var qvec: [Json] = [];
    match qemb {
        Json::Obj(qm) => {
            if jbool(qm, "ok") {
                match qm["vec"] {
                    Json::Arr(v) => {
                        for i in 0..len(v) {
                            match v[i] {
                                Json::Num(x) => {
                                    push(qvec, Json::Num(f64(f32_bits(x))));
                                }
                                _ => { }
                            }
                        }
                    }
                    _ => { }
                }
            }
        }
        _ => { }
    }
    let terms = distinctive_terms(query);
    let rows = kv_scan_prefix(db, chunk_prefix());
    var scanned = 0;
    match rows {
        Json::Arr(ra) => {
            for i in 0..len(ra) {
                if scanned > max_recall_chunks() { break; }
                scanned += 1;
                match ra[i] {
                    Json::Obj(row) => {
                        match json_parse(jstr(row, "value")) {
                            Json::Obj(cm) => {
                                let cvec = jarr_of(cm, "vector");
                                let ctext = jstr(cm, "text");
                                let cpath = jstr(cm, "path");
                                let score = cosine_sim(qvec, cvec)
                                    + lexical_bonus(terms, ctext, cpath);
                                var hit: {str: Json} = {};
                                hit["path"] = Json::Str(cpath);
                                hit["chunk_index"] = Json::Num(jnumf(cm, "chunk_index"));
                                hit["score"] = Json::Num(score);
                                var snip = ctext;
                                if len(snip) > recall_snippet_chars() {
                                    snip = slice_str(snip, 0, recall_snippet_chars());
                                }
                                hit["text"] = Json::Str(snip);
                                push(out_a, Json::Obj(hit));
                            }
                            _ => { }
                        }
                    }
                    _ => { }
                }
            }
        }
        _ => { }
    }
    // top-k by score descending — insertion-sort, hit lists are small
    var sorted: [Json] = [];
    for i in 0..len(out_a) {
        var ins = len(sorted);
        var j = 0;
        while j < len(sorted) {
            match sorted[j] {
                Json::Obj(sj) => {
                    match out_a[i] {
                        Json::Obj(h) => {
                            if jnumf(h, "score") > jnumf(sj, "score") {
                                ins = j;
                                j = len(sorted);
                            }
                        }
                        _ => { }
                    }
                }
                _ => { }
            }
            j += 1;
        }
        var tmp: [Json] = [];
        var a = 0;
        while a < ins { push(tmp, sorted[a]); a += 1; }
        push(tmp, out_a[i]);
        var b = ins;
        while b < len(sorted) { push(tmp, sorted[b]); b += 1; }
        sorted = tmp;
    }
    var top: [Json] = [];
    var i = 0;
    while i < len(sorted) && i < k {
        push(top, sorted[i]);
        i += 1;
    }
    return Json::Arr(top);
}

host det fn jarr_of(o: {str: Json}, k: str) -> [Json] {
    if has(o, k) {
        match o[k] { Json::Arr(a) => { return a; } _ => { } }
    }
    var empty: [Json] = [];
    return empty;
}

host fn recall(query: str, k: i64) -> Json cap fs {
    return recall_at(query, k, grounded_index_path());
}

host fn grounded_chunk_count_at(db: str) -> i64 cap fs {
    return kv_count_prefix(db, chunk_prefix());
}

// embed + persist one chunk record — factored so the enclosing loops stay
// flat (nested for/match/if var slots mis-compile in host bytecode)
host fn index_one_chunk(db: str, pkey: str, cp: str, mtime: i64, i: i64, text: str) -> unit cap fs {
    var vec_bits: [Json] = [];
    embed_into_bits(text, vec_bits);
    var rec: {str: Json} = {};
    rec["path"] = Json::Str(cp);
    rec["mtime"] = Json::Num(f64(mtime));
    rec["chunk_index"] = Json::Num(f64(i));
    rec["text"] = Json::Str(text);
    rec["vector"] = Json::Arr(vec_bits);
    let _p = kv_put(db, chunk_key_of(pkey, i), json_emit(Json::Obj(rec)));
}

host fn embed_into_bits(text: str, vec_bits: [Json]) -> unit cap fs {
    let emb = scav_embed(text);
    match emb {
        Json::Obj(em) => {
            if jbool(em, "ok") {
                match em["vec"] {
                    Json::Arr(v) => {
                        for k in 0..len(v) {
                            match v[k] {
                                Json::Num(x) => {
                                    push(vec_bits, Json::Num(f64(f32_bits(x))));
                                }
                                _ => { }
                            }
                        }
                    }
                    _ => { }
                }
            }
        }
        _ => { }
    }
}

// index every tracked file in a fs_walk result array → {files, chunks}
host fn index_walk_files(files: Json, db: str) -> Json cap fs {
    var indexed = 0;
    var chunks = 0;
    match files {
        Json::Arr(fa) => {
            for j in 0..len(fa) {
                let n = index_walk_entry(fa[j], db);
                if n > 0 {
                    indexed += 1;
                    chunks += n;
                }
            }
        }
        _ => { }
    }
    var r: {str: Json} = {};
    r["files"] = Json::Num(f64(indexed));
    r["chunks"] = Json::Num(f64(chunks));
    return Json::Obj(r);
}

host fn index_walk_entry(f: Json, db: str) -> i64 cap fs {
    match f {
        Json::Obj(m) => {
            let p = jstr(m, "path");
            if is_tracked_file(p) && !is_ignored_path(p) {
                return index_grounded_file(db, p, jnumi(m, "mtime"), jnumi(m, "size"));
            }
            return 0;
        }
        _ => { return 0; }
    }
}
