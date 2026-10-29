// strategy.cy — the durable strategy library, in Coyote.
//
// Port of src/strategy_library.rs: proven tool templates keyed by a
// problem signature, backed by redb (kv_* externs) so they survive
// restarts. Includes the dialectical engine (thesis → antithesis →
// synthesis repair) and the RustSynthesizer cargo-diagnostic healer.
//
// A Strategy is a Json::Obj with the serde field names — kv stores the
// emitted JSON verbatim so Rust and Coyote share the database.

cap fs, proc

import "sys.cy";
import "strlib.cy";
import "blueprint.cy";

// ── strategy records ───────────────────────────────────────────────────
host det fn strategy_new(key: str, problem: str, language: str, code: str) -> Json {
    var s: {str: Json} = {};
    s["key"] = Json::Str(key);
    s["problem"] = Json::Str(problem);
    s["language"] = Json::Str(language);
    s["code"] = Json::Str(code);
    s["reliability"] = Json::Num(0.5);
    s["uses"] = Json::Num(0.0);
    s["successes"] = Json::Num(0.0);
    return Json::Obj(s);
}

// record(success): uses+1, EMA reliability update (alpha 0.3)
host det fn strategy_record(st: Json, success: bool) -> Json {
    match st {
        Json::Obj(m) => {
            m["uses"] = Json::Num(jnumf(m, "uses") + 1.0);
            if success {
                m["successes"] = Json::Num(jnumf(m, "successes") + 1.0);
            }
            var observed = 0.0;
            if success { observed = 1.0; }
            m["reliability"] = Json::Num(jnumf(m, "reliability") * 0.7 + observed * 0.3);
            return Json::Obj(m);
        }
        _ => { return st; }
    }
}

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

host det fn jarr(o: {str: Json}, k: str) -> [Json] {
    if has(o, k) {
        match o[k] { Json::Arr(a) => { return a; } _ => { } }
    }
    var empty: [Json] = [];
    return empty;
}

// ── database ───────────────────────────────────────────────────────────
host fn db_path_default() -> str {
    let home = env_get("HOME");
    if home == "" { return "/tmp/.bad_apple/strategies.redb"; }
    return home + "/.bad_apple/strategies.redb";
}

host fn db_get(db: str, key: str) -> Json cap fs {
    let v = kv_get(db, key);
    if v == "" { return Json::Null; }
    return json_parse(v);
}

host fn db_put(db: str, st: Json) -> bool cap fs {
    match st {
        Json::Obj(m) => {
            let ok = kv_put(db, jstr(m, "key"), json_emit(st));
            if ok {
                cg_relate(jstr(m, "key"), "enables", jstr(m, "problem"));
                cg_relate("skill_memory", "depends_on", jstr(m, "key"));
            }
            return ok;
        }
        _ => { return false; }
    }
}

// every stored strategy, parsed — the `strategy list` source.
host fn db_all(db: str) -> [Json] cap fs {
    var out_v: [Json] = [];
    let rows = kv_list(db);
    match rows {
        Json::Arr(a) => {
            for i in 0..len(a) {
                match a[i] {
                    Json::Obj(row) => {
                        let sv = json_parse(jstr(row, "value"));
                        match sv { Json::Obj(_) => { push(out_v, sv); } _ => { } }
                    }
                    _ => { }
                }
            }
        }
        _ => { }
    }
    return out_v;
}

// Strategy::record — uses+1, successes+=ok, EMA reliability α=0.3.
host det fn strategy_record_outcome(st: Json, ok: bool) -> Json {
    match st {
        Json::Obj(m) => {
            let uses = jnumf(m, "uses") + 1.0;
            var succ = jnumf(m, "successes");
            if ok { succ = succ + 1.0; }
            var obs = 0.0;
            if ok { obs = 1.0; }
            let rel = jnumf(m, "reliability") * 0.7 + obs * 0.3;
            m["uses"] = Json::Num(uses);
            m["successes"] = Json::Num(succ);
            m["reliability"] = Json::Num(rel);
            return Json::Obj(m);
        }
        _ => { return st; }
    }
}

// all strategies below the reliability threshold
host fn weak_strategies(db: str, threshold: f64) -> [Json] cap fs {
    var out_v: [Json] = [];
    let rows = kv_list(db);
    match rows {
        Json::Arr(a) => {
            for i in 0..len(a) {
                match a[i] {
                    Json::Obj(row) => {
                        let sv = json_parse(jstr(row, "value"));
                        match sv {
                            Json::Obj(sm) => {
                                if jnumf(sm, "reliability") < threshold {
                                    push(out_v, sv);
                                }
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
    return out_v;
}

host fn prune_below(db: str, threshold: f64) -> i64 cap fs {
    let weak = weak_strategies(db, threshold);
    var n = 0;
    for i in 0..len(weak) {
        match weak[i] {
            Json::Obj(m) => {
                if kv_del(db, jstr(m, "key")) { n += 1; }
            }
            _ => { }
        }
    }
    return n;
}

// word set membership
host det fn word_in(words: [str], w: str) -> bool {
    for i in 0..len(words) {
        if words[i] == w { return true; }
    }
    return false;
}

// Jaccard overlap |A∩B| / |A∪B|
host det fn jaccard(a: [str], b: [str]) -> f64 {
    var inter = 0;
    var uni = 0;
    for i in 0..len(a) {
        if word_in(b, a[i]) { inter += 1; }
        uni += 1;
    }
    for j in 0..len(b) {
        if !word_in(a, b[j]) { uni += 1; }
    }
    if uni == 0 { return 0.0; }
    return f64(inter) / f64(uni);
}

// best_match(problem) — hybrid score: containment ⇒ 1.0 else Jaccard,
// discounted by 0.8 + 0.2·reliability; threshold 0.7.
host fn best_match(db: str, problem: str) -> Json cap fs {
    let pl = str_lower(problem);
    let pw = split_ws(pl);
    var best = Json::Null;
    var best_score = 0.0;
    let rows = kv_list(db);
    match rows {
        Json::Arr(a) => {
            for i in 0..len(a) {
                match a[i] {
                    Json::Obj(row) => {
                        let sv = json_parse(jstr(row, "value"));
                        match sv {
                            Json::Obj(sm) => {
                                let sl = str_lower(jstr(sm, "problem"));
                                var score = 0.0;
                                if contains(sl, pl) || contains(pl, sl) {
                                    score = 1.0;
                                } else {
                                    score = jaccard(pw, split_ws(sl));
                                }
                                score = score * (0.8 + 0.2 * jnumf(sm, "reliability"));
                                if score > 0.7 && score > best_score {
                                    best_score = score;
                                    best = sv;
                                }
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
    return best;
}

// ── dialectical engine ─────────────────────────────────────────────────
// nodes are Json {id, proposition, truth_value} — antecedents exist in
// Rust for arena wiring; synthesis only consumes truth values.
host det fn thesis_of(st: Json, predicted: str, id: i64) -> Json {
    var n: {str: Json} = {};
    match st {
        Json::Obj(m) => {
            n["id"] = Json::Num(f64(id));
            n["proposition"] = Json::Str(
                "Strategy '" + jstr(m, "key") + "' will succeed under assumed context: " + predicted);
            n["truth_value"] = Json::Num(jnumf(m, "reliability"));
        }
        _ => { }
    }
    return Json::Obj(n);
}

host det fn antithesis_of(error: str, id: i64) -> Json {
    var n: {str: Json} = {};
    n["id"] = Json::Num(f64(id));
    n["proposition"] = Json::Str("Observed failure: " + error);
    n["truth_value"] = Json::Num(1.0);
    return Json::Obj(n);
}

host det fn contradiction_strength(thesis: Json, anti: Json) -> f64 {
    var t = 0.0;
    var a = 0.0;
    match thesis {
        Json::Obj(m) => { t = jnumf(m, "truth_value"); }
        _ => { }
    }
    match anti {
        Json::Obj(m) => { a = jnumf(m, "truth_value"); }
        _ => { }
    }
    return t * a;
}

// synthesize: reset counters, heuristic repair, bind problem to the error
// signature so future matching finds this variant.
host det fn synthesize(st: Json, error: str) -> Json {
    match st {
        Json::Obj(m) => {
            m["uses"] = Json::Num(0.0);
            m["successes"] = Json::Num(0.0);
            m["reliability"] = Json::Num(0.5);
            let el = str_lower(error);
            if contains(el, "missing") && contains(el, "module") {
                m["code"] = Json::Str(
                    "import math, statistics, json, datetime, re, collections, itertools, string\n"
                    + jstr(m, "code") + "\n");
            }
            m["problem"] = Json::Str(
                jstr(m, "problem") + " [synthesized for error: " + error + "]");
            return Json::Obj(m);
        }
        _ => { return st; }
    }
}

// dialectical_repair(db, failed_key, error, predicted) → new key or ""
host fn dialectical_repair(db: str, failed_key: str, error: str, predicted: str) -> str cap fs {
    let orig = db_get(db, failed_key);
    match orig {
        Json::Obj(m) => {
            let t = thesis_of(orig, predicted, 1);
            let a = antithesis_of(error, 2);
            let _strength = contradiction_strength(t, a);
            var rep = synthesize(orig, error);
            let new_key = failed_key + "_synth_3";
            match rep {
                Json::Obj(rm) => { rm["key"] = Json::Str(new_key); }
                _ => { }
            }
            if db_put(db, rep) { return new_key; }
            return "";
        }
        _ => { return ""; }
    }
}

// ── RustSynthesizer — deterministic cargo-diagnostic repair ────────────
// parse `--> src/main.rs:LINE:COL` diagnostics → [line numbers]
host det fn cargo_error_lines(diagnostics: str) -> [i64] {
    var out_v: [i64] = [];
    let ls = split(diagnostics, "\n");
    for i in 0..len(ls) {
        let t = trim(ls[i]);
        if starts_with(t, "-->") {
            let space = find_sub(t, " ");
            if space >= 0 {
                let path_part = slice_str(t, space + 1, len(t));
                let colon = rfind_sub(path_part, ":");
                if colon >= 0 {
                    let before = slice_str(path_part, 0, colon);
                    let lc = rfind_sub(before, ":");
                    if lc >= 0 {
                        let num = trim(slice_str(before, lc + 1, len(before)));
                        let n = parse_num(num);
                        if n > 0.0 { push(out_v, i64(n)); }
                    }
                }
            }
        }
    }
    return out_v;
}

// repair(source, diagnostics) — the deterministic rewrites, in order.
host det fn rust_repair(source: str, diagnostics: str) -> str {
    let diag = str_lower(diagnostics);
    var repaired = source;

    if contains(diag, "main function cannot be async") && !contains(repaired, "#[tokio::main]") {
        repaired = replace_all(repaired, "async fn main()", "#[tokio::main]\nasync fn main()");
    }

    if contains(diag, "use of undeclared crate or module") || contains(diag, "cannot find") {
        if contains(diag, "serde_json") && !contains(repaired, "use serde_json") {
            repaired = "use serde_json::Value;\n" + repaired;
        }
        if contains(diag, "sha2") && !contains(repaired, "use sha2") {
            repaired = "use sha2::{Sha256, Digest};\n" + repaired;
        }
    }

    if contains(diag, "cannot assign to") || contains(diag, "cannot borrow") {
        repaired = replace_all(repaired, "let input =", "let mut input =");
        repaired = replace_all(repaired, "let numbers: Vec<i64>", "let mut numbers: Vec<i64>");
    }

    if contains(diag, "fs::") && !contains(repaired, "use std::fs") {
        repaired = "use std::fs;\n" + repaired;
    }
    if contains(diag, "path::") && !contains(repaired, "use std::path") {
        repaired = "use std::path::Path;\n" + repaired;
    }

    if contains(diag, "expected `;`") || contains(diag, "expected semicolon") {
        let err_lines = cargo_error_lines(diagnostics);
        var owned = split(repaired, "\n");
        for i in 0..len(err_lines) {
            let ln = err_lines[i];
            if ln > 0 && ln <= len(owned) {
                let l = trim_end(owned[ln - 1]);
                if !ends_with(l, ";") && !ends_with(l, "{") && !ends_with(l, "}") && len(l) > 0 {
                    owned[ln - 1] = l + ";";
                }
            }
        }
        repaired = join_str(owned, "\n");
    }

    if contains(diag, "unclosed delimiter")
        || (contains(diag, "expected one of") && contains(diag, "`}`"))
    {
        let open_n = count_char(repaired, "{");
        let close_n = count_char(repaired, "}");
        if open_n > close_n {
            var i = 0;
            while i < open_n - close_n {
                repaired = repaired + "}";
                i += 1;
            }
        }
    }

    return repaired;
}

host det fn join_str(items: [str], sep: str) -> str {
    var acc = "";
    for i in 0..len(items) {
        if i > 0 { acc = acc + sep; }
        acc = acc + items[i];
    }
    return acc;
}

// ── proposals ──────────────────────────────────────────────────────────
host fn strategy_proposals_dir() -> str {
    let e = env_get("BADAPPLE_STRATEGY_PROPOSALS");
    if e != "" { return e; }
    let home = env_get("HOME");
    if home == "" { return "/tmp/.bad_apple/notes/proposed_patches"; }
    return home + "/.bad_apple/notes/proposed_patches";
}

host det fn safe_key_of(key: str) -> str {
    var r = "";
    for i in 0..len(key) {
        let c = slice_str(key, i, i + 1);
        if contains("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", c) {
            r = r + c;
        } else {
            r = r + "-";
        }
    }
    return r;
}

// propose_repair — synthesis goes to a curious-format proposal for
// governed adoption, never auto-installed. → path or ""
host fn propose_repair(db: str, failed_key: str, error: str, predicted: str) -> str cap fs {
    let orig = db_get(db, failed_key);
    match orig {
        Json::Obj(m) => {
            let t = thesis_of(orig, predicted, 1);
            let a = antithesis_of(error, 2);
            let strength = contradiction_strength(t, a);
            if strength < 0.2 { return ""; }
            var cand = synthesize(orig, error);
            let ckey = failed_key + "_synth_3";
            match cand {
                Json::Obj(cm) => { cm["key"] = Json::Str(ckey); }
                _ => { }
            }
            let dir = strategy_proposals_dir();
            fs_mkdir(dir);
            let stamp = digits_compact(time_stamp());
            let path = dir + "/strategy-" + safe_key_of(failed_key) + "-" + stamp + ".md";
            var excerpt = error;
            if len(excerpt) > 500 { excerpt = slice_str(excerpt, 0, 500); }
            var sjson: {str: Json} = {};
            match cand {
                Json::Obj(cm) => {
                    var sj: {str: Json} = {};
                    sj["key"] = Json::Str(jstr(cm, "key"));
                    sj["problem"] = Json::Str(jstr(cm, "problem"));
                    sj["language"] = Json::Str(jstr(cm, "language"));
                    sj["code"] = Json::Str(jstr(cm, "code"));
                    sjson["strategy"] = Json::Obj(sj);
                }
                _ => { }
            }
            sjson["no_patch"] = Json::Bool(true);
            let body = "# Strategy repair proposal: " + failed_key + "\n\n**When:** "
                + time_stamp() + "\n**Workspace:** system\n\n## Finding\n\n- Failed strategy: `"
                + failed_key + "`\n- Prior reliability: " + str(jnumf(m, "reliability"))
                + " over " + str(jnumf(m, "uses")) + " use(s)\n- Contradiction strength: "
                + str(strength) + "\n- Observed error: " + excerpt
                + "\n\n## Proposal\n\n```json\n" + json_emit(Json::Obj(sjson))
                + "\n```\n\n## Adoption\n\n`badapple strategy adopt " + path + "`\n";
            write_file(path, body);
            return path;
        }
        _ => { return ""; }
    }
}

// "YYYYMMDD-HHMMSS" from the time_stamp shape
host det fn digits_compact(stamp: str) -> str {
    var compact = "";
    for i in 0..len(stamp) {
        let c = slice_str(stamp, i, i + 1);
        if contains("0123456789", c) { compact = compact + c; }
    }
    if len(compact) >= 14 {
        return slice_str(compact, 0, 8) + "-" + slice_str(compact, 8, 14);
    }
    return compact;
}

// adopt_proposal(path) → adopted strategy key or "" on malformed
host fn adopt_proposal(db: str, path: str) -> str cap fs {
    if !fs_exists(path) { return ""; }
    let text = read_file(path);
    let sections = split(text, "## Proposal");
    if len(sections) < 2 { return ""; }
    let jblocks = split(sections[1], "```json");
    if len(jblocks) < 2 { return ""; }
    let jtail = split(jblocks[1], "```");
    if len(jtail) < 1 { return ""; }
    let v = json_parse(trim(jtail[0]));
    match v {
        Json::Obj(m) => {
            if !has(m, "strategy") { return ""; }
            match m["strategy"] {
                Json::Obj(sm) => {
                    var lang = jstr(sm, "language");
                    if lang == "" { lang = "shell"; }
                    let st = strategy_new(
                        jstr(sm, "key"), jstr(sm, "problem"), lang, jstr(sm, "code"));
                    var key = "";
                    match st {
                        Json::Obj(nm) => { key = jstr(nm, "key"); }
                        _ => { }
                    }
                    if key == "" { return ""; }
                    if !db_put(db, st) { return ""; }
                    write_file(path, text + "\n## Applied\n\nAdopted "
                        + time_stamp() + " via `strategy adopt`.\n");
                    return key;
                }
                _ => { return ""; }
            }
        }
        _ => { return ""; }
    }
}
