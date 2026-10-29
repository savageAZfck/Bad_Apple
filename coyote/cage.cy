// cage.cy — the automation cage, ported from src/automation_cage.rs.
//
// Path jailing for tool actions: absolute paths only, no "." / ".."
// components, security-sensitive names and system prefixes forbidden,
// symlinks resolved via the deepest existing ancestor, and every resolved
// path must stay inside an allowlisted root. Returns "" on accept or a
// rejection reason — matching the Rust Result<Plan> split.

import "sys.cy";
import "strlib.cy";

cap fs

host det fn forbidden_roots() -> [str] {
    return ["/", "/System", "/private", "/Library", "/root", "/etc",
            "/usr", "/bin", "/sbin", "/var"];
}

host det fn forbidden_names() -> [str] {
    return [".ssh", ".gnupg", ".aws", ".azure", ".kube", ".docker",
            ".git", ".env", ".netrc", ".npmrc", ".pypirc",
            "authorized_keys", "known_hosts"];
}

host det fn lower_str(s: str) -> str {
    var acc = "";
    for i in 0..len(s) {
        let c = slice_str(s, i, i + 1);
        let D = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
        let d = "abcdefghijklmnopqrstuvwxyz";
        var lc = c;
        for j in 0..26 {
            if slice_str(D, j, j + 1) == c { lc = slice_str(d, j, j + 1); }
        }
        acc = acc + lc;
    }
    return acc;
}

// "" on accept, reason on reject — mirrors reject_lexical_path
host det fn reject_lexical(path: str) -> str {
    if !starts_with(path, "/") {
        return "automation paths must be absolute: " + path;
    }
    let parts = split(path, "/");
    for i in 0..len(parts) {
        let p = parts[i];
        if p == ".." || p == "." {
            return "path traversal components are forbidden: " + path;
        }
        if p != "" {
            let e = reject_security_name(p, path);
            if e != "" { return e; }
        }
    }
    return "";
}

host det fn reject_security_name(name: str, path: str) -> str {
    let lowered = lower_str(name);
    let bad = forbidden_names();
    for i in 0..len(bad) {
        if lowered == bad[i] {
            return "hidden security path is forbidden: " + path;
        }
    }
    if starts_with(lowered, ".env.") || ends_with(lowered, ".keychain-db") {
        return "hidden security path is forbidden: " + path;
    }
    return "";
}

host det fn path_eq_or_under(path: str, prefix: str) -> bool {
    if path == prefix { return true; }
    return starts_with(path, prefix + "/");
}

// "" on accept — mirrors reject_forbidden_path (home = $HOME str)
host det fn reject_forbidden(path: str, home: str) -> str {
    let bad = forbidden_roots();
    for i in 0..len(bad) {
        if path_eq_or_under(path, bad[i]) {
            return "system or private path is forbidden: " + path;
        }
    }
    if path_eq_or_under(path, home + "/Library") {
        return "system or private path is forbidden: " + path;
    }
    return "";
}

// resolve via deepest existing ancestor, reject symlink components,
// re-append the missing tail — mirrors canonicalize_existing_ancestor.
host fn resolve_ancestor(path: str) -> Json cap fs {
    var ancestor = path;
    var missing: [str] = [];
    while !fs_exists(ancestor) {
        let name = slice_str(ancestor, len(dir_of(ancestor)) + 1, len(ancestor));
        if name == "" {
            return Json::Obj({ "error": Json::Str("path has no existing ancestor: " + path) });
        }
        push(missing, name);
        ancestor = dir_of(ancestor);
    }
    if fs_is_symlink(ancestor) {
        return Json::Obj({ "error": Json::Str("symlink path components are forbidden: " + path) });
    }
    var resolved = fs_realpath(ancestor);
    if resolved == "" {
        return Json::Obj({ "error": Json::Str("cannot canonicalize: " + ancestor) });
    }
    for i in 0..len(missing) {
        resolved = resolved + "/" + missing[len(missing) - 1 - i];
    }
    return Json::Obj({ "path": Json::Str(resolved) });
}

// a full resolve_allowlisted: lexical + forbidden + ancestor + root jail.
// → Json {path} or {error}
host fn resolve_allowlisted(path: str, roots: [str], home: str) -> Json cap fs {
    var e = reject_lexical(path);
    if e != "" { return Json::Obj({ "error": Json::Str(e) }); }
    e = reject_forbidden(path, home);
    if e != "" { return Json::Obj({ "error": Json::Str(e) }); }
    match resolve_ancestor(path) {
        Json::Obj(r) => {
            if has(r, "error") { return Json::Obj({ "error": r["error"] }); }
            var resolved = "";
            match r["path"] { Json::Str(p) => { resolved = p; } _ => { } }
            let e2 = reject_forbidden(resolved, home);
            if e2 != "" { return Json::Obj({ "error": Json::Str(e2) }); }
            for i in 0..len(roots) {
                if path_eq_or_under(resolved, roots[i]) {
                    return Json::Obj({ "path": Json::Str(resolved) });
                }
            }
            return Json::Obj({ "error": Json::Str("path escapes all allowlisted roots: " + path) });
        }
        _ => { return Json::Obj({ "error": Json::Str("resolve failed") }); }
    }
}

// canonicalize the allowlisted roots at cage construction —
// mirrors AutomationCage::new. → Json {roots:[str]} or {error}
host fn cage_new(roots: [str], home: str) -> Json cap fs {
    if len(roots) == 0 {
        return Json::Obj({ "error": Json::Str("automation requires at least one allowlisted root") });
    }
    var canon: [str] = [];
    for i in 0..len(roots) {
        let e = reject_lexical(roots[i]);
        if e != "" { return Json::Obj({ "error": Json::Str(e) }); }
        if !fs_exists(roots[i]) {
            return Json::Obj({ "error": Json::Str("allowlisted root does not exist: " + roots[i]) });
        }
        let c = fs_realpath(roots[i]);
        if c == "" || !fs_is_dir(c) {
            return Json::Obj({ "error": Json::Str("allowlisted root is not a directory: " + roots[i]) });
        }
        let e2 = reject_forbidden(c, home);
        if e2 != "" { return Json::Obj({ "error": Json::Str(e2) }); }
        var seen = false;
        for j in 0..len(canon) {
            if canon[j] == c { seen = true; }
        }
        if !seen { push(canon, c); }
    }
    var acc: [Json] = [];
    for i in 0..len(canon) { push(acc, Json::Str(canon[i])); }
    return Json::Obj({ "roots": Json::Arr(acc) });
}

// ── action layer — ported from automation_cage_impl.rs ─────────────────
// The policy lives here in-language; the mutation itself crosses a single
// extern boundary (cage_exec) where the Rust fd-dance runs. serde shape:
// {"operation":"<snake_case>", ...fields} — deny_unknown_fields parity is
// enforced by the whitelist below.

host det fn max_copy_bytes() -> i64 { return 104857600; } // 100 MiB

host det fn action_ops() -> [str] {
    return ["create_file","create_directory","copy_file","move_file","move_to_trash"];
}

// exact field whitelist per op — deny_unknown_fields equivalent.
// → "" ok, or the offending key.
host det fn action_bad_field(op: str, m: {str: Json}) -> str {
    let ks = keys(m);
    for i in 0..len(ks) {
        let k = ks[i];
        if k == "operation" { }
        else {
            if op == "create_file" || op == "create_directory" || op == "move_to_trash" {
                if k != "path" { return k; }
            } else {
                if k != "source" && k != "destination" { return k; }
            }
        }
    }
    return "";
}

host det fn action_field(m: {str: Json}, k: str) -> str {
    if has(m, k) {
        match m[k] { Json::Str(x) => { return x; } _ => { } }
    }
    return "";
}

// validate one action's Json shape → "" ok or error reason.
host det fn action_check(m: {str: Json}) -> str {
    if !has(m, "operation") { return "action missing operation"; }
    var op = "";
    match m["operation"] {
        Json::Str(o) => { op = o; }
        _ => { return "operation must be a string"; }
    }
    var known = false;
    let ops = action_ops();
    for i in 0..len(ops) { if ops[i] == op { known = true; } }
    if !known { return "unknown operation: " + op; }
    let bad = action_bad_field(op, m);
    if bad != "" { return "unknown field in action: " + bad; }
    if op == "create_file" || op == "create_directory" || op == "move_to_trash" {
        if action_field(m, "path") == "" { return op + " requires path"; }
    } else {
        if action_field(m, "source") == "" || action_field(m, "destination") == "" {
            return op + " requires source and destination";
        }
    }
    return "";
}

// parse every ```badapple-action / ```json fenced block →
// Json {actions:[obj]} or {error}. Fail-closed like the Rust.
host det fn cage_parse_actions(input: str) -> Json {
    let norm = replace_all(input, "\r\n", "\n");
    let lines = split(norm, "\n");
    var acts: [Json] = [];
    var i = 0;
    while i < len(lines) {
        let t = trim(lines[i]);
        if t == "```badapple-action" || t == "```json" {
            var body = "";
            var closed = false;
            var j = i + 1;
            while j < len(lines) && !closed {
                if trim(lines[j]) == "```" {
                    closed = true;
                } else {
                    if body != "" { body = body + "\n"; }
                    body = body + lines[j];
                    j = j + 1;
                }
            }
            if !closed || trim(body) == "" {
                return Json::Obj({ "error": Json::Str("invalid or unterminated action fence") });
            }
            let act = json_parse(trim(body));
            match act {
                Json::Obj(m) => {
                    let e = action_check(m);
                    if e != "" {
                        return Json::Obj({ "error": Json::Str(e) });
                    }
                    push(acts, act);
                }
                _ => {
                    return Json::Obj({ "error": Json::Str("invalid fenced action JSON") });
                }
            }
            i = j;
        }
        i = i + 1;
    }
    if len(acts) == 0 {
        return Json::Obj({ "error": Json::Str("no badapple-action or compatible json fenced action found") });
    }
    return Json::Obj({ "actions": Json::Arr(acts) });
}

// ── placeholder substitution — ~/ and /Users/<me>/ → $HOME ─────────────
host det fn cage_substitute(path: str, home: str) -> str {
    if starts_with(path, "~/") {
        return home + slice_str(path, 1, len(path));
    }
    let user = slice_str(home, rfind_sub(home, "/") + 1, len(home));
    let prefix = "/Users/" + user + "/";
    if starts_with(path, prefix) {
        return home + "/" + slice_str(path, len(prefix), len(path));
    }
    return path;
}

// ── validate — Action → Plan, or {error} ───────────────────────────────
host fn cage_validate(action: Json, roots: [str], home: str) -> Json cap fs {
    var op = "";
    var p_src = "";
    var p_dst = "";
    match action {
        Json::Obj(m) => {
            if has(m, "operation") { match m["operation"] { Json::Str(x) => { op = x; } _ => { } } }
            if has(m, "path") { match m["path"] { Json::Str(x) => { p_src = x; } _ => { } } }
            if has(m, "source") { match m["source"] { Json::Str(x) => { p_src = x; } _ => { } } }
            if has(m, "destination") { match m["destination"] { Json::Str(x) => { p_dst = x; } _ => { } } }
        }
        _ => { return Json::Obj({ "error": Json::Str("bad action") }); }
    }
    // substitute placeholders, then jail-resolve each path.
    let a_src = cage_substitute(p_src, home);
    var a_dst = p_dst;
    if p_dst != "" { a_dst = cage_substitute(p_dst, home); }
    var resolved: [str] = [];
    match resolve_allowlisted(a_src, roots, home) {
        Json::Obj(r) => {
            if has(r, "error") { return Json::Obj({ "error": r["error"] }); }
            match r["path"] { Json::Str(p) => { push(resolved, p); } _ => { } }
        }
        _ => { return Json::Obj({ "error": Json::Str("resolve failed") }); }
    }
    if a_dst != "" {
        match resolve_allowlisted(a_dst, roots, home) {
            Json::Obj(r) => {
                if has(r, "error") { return Json::Obj({ "error": r["error"] }); }
                match r["path"] { Json::Str(p) => { push(resolved, p); } _ => { } }
            }
            _ => { return Json::Obj({ "error": Json::Str("resolve failed") }); }
        }
    }

    if op == "create_file" || op == "create_directory" {
        if fs_exists(resolved[0]) {
            return Json::Obj({ "error": Json::Str("destination exists; overwrite forbidden") });
        }
    }
    if op == "copy_file" || op == "move_file" {
        let src = resolved[0];
        if fs_lstat_symlink(src) || !fs_is_file(src) {
            return Json::Obj({ "error": Json::Str("source must be a regular non-symlink file") });
        }
        if fs_nlink(src) > 1 {
            return Json::Obj({ "error": Json::Str("source has multiple hard links and may cross trust boundaries") });
        }
        if op == "copy_file" && fs_size(src) > max_copy_bytes() {
            return Json::Obj({ "error": Json::Str("copy source is oversized") });
        }
        // destination checks
        let dst = resolved[1];
        if src == dst {
            return Json::Obj({ "error": Json::Str("source equals destination") });
        }
        if fs_lstat_symlink(dst) {
            return Json::Obj({ "error": Json::Str("destination is a symlink; overwrite via symlink is forbidden") });
        }
        if fs_exists(dst) {
            return Json::Obj({ "error": Json::Str("overwrite forbidden") });
        }
        let parent = dir_of(dst);
        if !fs_exists(parent) {
            return Json::Obj({ "error": Json::Str("destination parent missing") });
        }
        if fs_lstat_symlink(parent) || !fs_is_dir(parent) {
            return Json::Obj({ "error": Json::Str("unsafe destination parent") });
        }
    }
    var trash = "";
    if op == "move_to_trash" {
        let src = resolved[0];
        if fs_lstat_symlink(src) {
            return Json::Obj({ "error": Json::Str("special or symlink trash source forbidden") });
        }
        if !fs_is_file(src) && !fs_is_dir(src) {
            return Json::Obj({ "error": Json::Str("trash source does not exist") });
        }
        // ~/.Trash must be a real dir, not a symlink
        let tdir = home + "/.Trash";
        if !fs_exists(tdir) { return Json::Obj({ "error": Json::Str("~/.Trash unavailable") }); }
        if fs_lstat_symlink(tdir) { return Json::Obj({ "error": Json::Str("~/.Trash must not be a symlink") }); }
        let canont = fs_realpath(tdir);
        if canont == "" || !starts_with(canont, home) || !fs_is_dir(canont) {
            return Json::Obj({ "error": Json::Str("unsafe trash directory") });
        }
        let name = slice_str(src, rfind_sub(src, "/") + 1, len(src));
        if name == "" { return Json::Obj({ "error": Json::Str("source has no name") }); }
        var cand = canont + "/" + name;
        var n = 0;
        while fs_exists(cand) && n <= 10000 {
            n = n + 1;
            cand = canont + "/" + name + " " + str(n);
        }
        if n > 10000 { return Json::Obj({ "error": Json::Str("no collision-safe trash name available") }); }
        trash = cand;
    }
    var paths: [Json] = [];
    for i in 0..len(resolved) { push(paths, Json::Str(resolved[i])); }
    return Json::Obj({
        "operation": Json::Str(op),
        "paths": Json::Arr(paths),
        "trash": Json::Str(trash),
    });
}

// ── from_env / default roots ───────────────────────────────────────────
host fn cage_home() -> Json cap fs {
    let h = env_get("HOME");
    if h == "" { return Json::Obj({ "error": Json::Str("HOME is not set") }); }
    let c = fs_realpath(h);
    if c == "" { return Json::Obj({ "error": Json::Str("cannot canonicalize HOME") }); }
    return Json::Obj({ "path": Json::Str(c) });
}

host fn cage_default_roots(home: str) -> Json cap fs {
    var roots: [str] = [];
    let names = ["Desktop","Documents","Downloads","Developer","Projects","src"];
    for i in 0..len(names) {
        let p = home + "/" + names[i];
        if fs_is_dir(p) { push(roots, p); }
    }
    // any home dir entry that looks like a source root
    let entries = fs_list(home);
    let markers = [".git","Cargo.toml","Package.swift","pyproject.toml","go.mod","package.json"];
    for i in 0..len(entries) {
        let p = home + "/" + entries[i];
        if fs_is_dir(p) {
            var is_src = false;
            for k in 0..len(markers) {
                if fs_exists(p + "/" + markers[k]) { is_src = true; }
            }
            if is_src { push(roots, p); }
        }
    }
    // cwd under home with repo markers
    let cwd = fs_realpath(".");
    if cwd != "" && starts_with(cwd, home) {
        if fs_exists(cwd + "/.git") || fs_is_file(cwd + "/Cargo.toml") {
            push(roots, cwd);
        }
    }
    if len(roots) == 0 {
        return Json::Obj({ "error": Json::Str("no safe default roots; set BADAPPLE_AUTOMATION_ROOTS") });
    }
    return Json::Obj({ "roots": strs_json(roots) });
}

host det fn strs_json(v: [str]) -> Json {
    var a: [Json] = [];
    for i in 0..len(v) { push(a, Json::Str(v[i])); }
    return Json::Arr(a);
}

// → Json {roots:[str], home:str, log_path:str} or {error}
host fn cage_from_env() -> Json cap fs {
    match cage_home() {
        Json::Obj(h) => {
            if has(h, "error") { return Json::Obj({ "error": h["error"] }); }
            var home = "";
            match h["path"] { Json::Str(x) => { home = x; } _ => { } }
            var roots: [str] = [];
            let env_roots = env_get("BADAPPLE_AUTOMATION_ROOTS");
            if env_roots != "" {
                let parts = split(env_roots, ":");
                for i in 0..len(parts) {
                    if parts[i] != "" { push(roots, parts[i]); }
                }
                if len(roots) == 0 {
                    return Json::Obj({ "error": Json::Str("BADAPPLE_AUTOMATION_ROOTS is empty") });
                }
            } else {
                match cage_default_roots(home) {
                    Json::Obj(r) => {
                        if has(r, "error") { return Json::Obj({ "error": r["error"] }); }
                        match r["roots"] {
                            Json::Arr(a) => {
                                for i in 0..len(a) {
                                    match a[i] { Json::Str(x) => { push(roots, x); } _ => { } }
                                }
                            }
                            _ => { }
                        }
                    }
                    _ => { }
                }
            }
            var log = env_get("BADAPPLE_AUTOMATION_LOG");
            if log == "" { log = home + "/.badapple/automation.jsonl"; }
            // canonicalize + jail the roots (mirrors AutomationCage::new)
            match cage_new(roots, home) {
                Json::Obj(c) => {
                    if has(c, "error") { return Json::Obj({ "error": c["error"] }); }
                    var canon: [str] = [];
                    match c["roots"] {
                        Json::Arr(a) => {
                            for i in 0..len(a) {
                                match a[i] { Json::Str(x) => { push(canon, x); } _ => { } }
                            }
                        }
                        _ => { }
                    }
                    return Json::Obj({
                        "roots": strs_json(canon),
                        "home": Json::Str(home),
                        "log_path": Json::Str(log),
                    });
                }
                _ => { return Json::Obj({ "error": Json::Str("cage init failed") }); }
            }
        }
        _ => { return Json::Obj({ "error": Json::Str("home resolve failed") }); }
    }
}

// ── execute — validate in-language, mutate across the extern boundary ──
// → Json report {operation, paths, result, elapsed_ms} or {error}
host fn cage_execute(action: Json, cage: Json) -> Json cap fs {
    var roots: [str] = [];
    var home = "";
    var log = "";
    match cage {
        Json::Obj(c) => {
            match c["roots"] {
                Json::Arr(a) => {
                    for i in 0..len(a) {
                        match a[i] { Json::Str(x) => { push(roots, x); } _ => { } }
                    }
                }
                _ => { }
            }
            match c["home"] { Json::Str(x) => { home = x; } _ => { } }
            match c["log_path"] { Json::Str(x) => { log = x; } _ => { } }
        }
        _ => { }
    }
    let plan = cage_validate(action, roots, home);
    match plan {
        Json::Obj(p) => {
            if has(p, "error") {
                return Json::Obj({ "error": p["error"] });
            }
        }
        _ => { return Json::Obj({ "error": Json::Str("validate failed") }); }
    }
    // the mutation: serialize action → Rust cage_exec re-validates +
    // performs the fd-safe op + appends the audit line.
    var csv = "";
    for i in 0..len(roots) {
        if i > 0 { csv = csv + ":"; }
        csv = csv + roots[i];
    }
    let r = cage_exec(json_emit(action), csv, home, log);
    return r;
}

// convenience: run a fenced text end-to-end — parse, env cage, execute all.
host fn cage_run(input: str) -> Json cap fs {
    let cj = cage_from_env();
    match cj {
        Json::Obj(cage) => {
            if has(cage, "error") { return Json::Obj({ "error": cage["error"] }); }
            match cage_parse_actions(input) {
                Json::Obj(parsed) => {
                    if has(parsed, "error") { return Json::Obj({ "error": parsed["error"] }); }
                    var results: [Json] = [];
                    match parsed["actions"] {
                        Json::Arr(acts) => {
                            for i in 0..len(acts) {
                                push(results, cage_execute(acts[i], cj));
                            }
                        }
                        _ => { }
                    }
                    return Json::Obj({ "results": Json::Arr(results) });
                }
                _ => { return Json::Obj({ "error": Json::Str("parse failed") }); }
            }
        }
        _ => { return Json::Obj({ "error": Json::Str("cage init failed") }); }
    }
}
