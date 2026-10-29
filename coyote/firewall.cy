// firewall.cy — the hardware output firewall, in Coyote.
//
// Port of src/firewall_machine.rs: the blocklist is merged into a pattern
// set (built-ins first, then non-comment blocklist lines, verbatim) and
// compiled into a stateful Aho-Corasick DFA running on the ANE — the
// shell performs the actual DFA→mlmodelc emit through dfa_ane_compile.
//
// The fingerprint contract is load-bearing: SHA256(lowercased patterns
// joined by '\n') must match the Swift engine's live snapshot; the
// fingerprint file is written LAST so a partial build never looks fresh.

cap fs, proc

import "sys.cy";
import "strlib.cy";

// Built-in patterns — mirrors defaultPatterns in
// src/platform/apple_desktop/BadAppleSecurity.swift. Keep in sync.
host fn builtin_patterns() -> [str] {
    var pats: [str] = [];
    push(pats, "sk-");
    push(pats, "ssh-rsa");
    push(pats, "-----BEGIN");
    push(pats, "-----END");
    push(pats, "BEGIN PRIVATE KEY");
    push(pats, "BEGIN OPENSSH PRIVATE KEY");
    return pats;
}

host fn firewall_dir() -> str { return "/var/lib/bad_apple"; }
host fn package_name() -> str { return "firewall.mlpackage"; }
host fn compiled_name() -> str { return "firewall.mlmodelc"; }
host fn fingerprint_name() -> str { return "firewall.patterns.sha"; }

// BADAPPLE_BLOCKLIST overrides; engine uses the same env.
host fn blocklist_path() -> str {
    let v = env_get("BADAPPLE_BLOCKLIST");
    if v != "" { return v; }
    return firewall_dir() + "/blocklist.txt";
}

// The merged pattern set in the same order the engine builds it:
// built-ins first, then each non-comment blocklist line, verbatim.
host fn merged_patterns() -> [str] cap fs {
    var pats = builtin_patterns();
    if fs_exists(blocklist_path()) {
        let text = read_file(blocklist_path());
        let lines = split(text, "\n");
        var i = 0;
        while i < len(lines) {
            let t = trim(lines[i]);
            if len(t) > 0 {
                if slice_str(t, 0, 1) != "#" { push(pats, t); }
            }
            i = i + 1;
        }
    }
    return pats;
}

// Canonical fingerprint — SHA-256 of the lowercased patterns joined by
// '\n', hex-encoded. Must match the Swift engine's.
host det fn fingerprint(patterns: [str]) -> str {
    let joined = join_lower(patterns, "\n");
    return sha256(joined);
}

host det fn join_lower(items: [str], sep: str) -> str {
    var acc = "";
    var i = 0;
    while i < len(items) {
        if i > 0 { acc = acc + sep; }
        acc = acc + str_lower(items[i]);
        i = i + 1;
    }
    return acc;
}

// Freshness: does the on-disk fingerprint match the live pattern set?
host fn is_fresh(dir: str) -> bool cap fs {
    let fp = dir + "/" + fingerprint_name();
    if !fs_exists(fp) { return false; }
    let on_disk = trim(read_file(fp));
    if on_disk != fingerprint(merged_patterns()) { return false; }
    return fs_is_dir(dir + "/" + compiled_name());
}

// compile_to(dir) — emit the DFA + ANE organ binaries into
// dir/firewall.mlmodelc through the shell's compiler bridge, then write
// the fingerprint LAST (a partial build must never look fresh).
// → Json {ok, states, patterns, err}
host fn compile_to(dir: str) -> Json cap fs {
    let patterns = merged_patterns();
    let compiled = dir + "/" + compiled_name();
    if !fs_exists(compiled) { fs_mkdir(compiled); }
    var pat_json: [Json] = [];
    var i = 0;
    while i < len(patterns) {
        push(pat_json, Json::Str(patterns[i]));
        i = i + 1;
    }
    let res = dfa_ane_compile(Json::Arr(pat_json), compiled);
    match res {
        Json::Obj(r) => {
            if jbool(r, "ok") {
                let fp = dir + "/" + fingerprint_name();
                write_file(fp, fingerprint(patterns));
            }
            set_key_num(r, "patterns", f64_of(len(patterns)));
            return Json::Obj(r);
        }
        _ => { return res; }
    }
}

host det fn jbool(o: {str: Json}, k: str) -> bool {
    if has(o, k) {
        match o[k] { Json::Bool(b) => { return b; } _ => { } }
    }
    return false;
}

host fn f64_of(n: i64) -> f64 {
    return f64(n);
}

// ── CLI entry: `badapple firewall compile` / `status` via .cy ──────────
host fn firewall_status(dir: str) -> Json cap fs {
    var m: {str: Json} = {};
    let pats = merged_patterns();
    set_key_num(m, "patterns", f64_of(len(pats)));
    m["fingerprint"] = Json::Str(fingerprint(pats));
    m["fresh"] = Json::Bool(is_fresh(dir));
    m["compiled"] = Json::Str(dir + "/" + compiled_name());
    m["blocklist"] = Json::Str(blocklist_path());
    return Json::Obj(m);
}

host fn set_key_num(m: {str: Json}, k: str, v: f64) -> unit {
    m[k] = Json::Num(v);
}
