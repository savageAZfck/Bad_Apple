// cli.cy — the organism's own front door.
//
// The Rust `badapple` binary consults this module first: `cli_dispatch`
// takes the argv tail and returns {handled: true, out, err} for every
// subcommand the organism runs in-language, or {handled: false} so the
// Rust falls back to its legacy path. Ports land subcommand at a time —
// vault and the non-interactive strategy surface are first.

cap fs, crypto

import "sys.cy";
import "strlib.cy";
import "vault.cy";
import "strategy.cy";

// ── result helpers ─────────────────────────────────────────────────────

host det fn cli_ok(o: str) -> Json {
    return Json::Obj({
        "handled": Json::Bool(true),
        "code": Json::Num(0.0),
        "out": Json::Str(o),
        "err": Json::Str("")
    });
}

host det fn cli_fail(err: str) -> Json {
    return Json::Obj({
        "handled": Json::Bool(true),
        "code": Json::Num(1.0),
        "out": Json::Str(""),
        "err": Json::Str(err)
    });
}

host det fn cli_unhandled() -> Json {
    return Json::Obj({ "handled": Json::Bool(false) });
}

// ── dispatch ───────────────────────────────────────────────────────────

// args = argv after "badapple" (e.g. ["vault","list"]).
host fn cli_dispatch(args: [str]) -> Json cap fs, crypto {
    if len(args) == 0 { return cli_unhandled(); }
    let sub = args[0];
    if sub == "vault" { return cli_vault(drop_first(args)); }
    if sub == "strategy" { return cli_strategy(drop_first(args)); }
    return cli_unhandled();
}

host det fn drop_first(v: [str]) -> [str] {
    var out_v: [str] = [];
    for i in 1..len(v) { push(out_v, v[i]); }
    return out_v;
}

// ── vault ──────────────────────────────────────────────────────────────

host fn cli_vault(args: [str]) -> Json cap fs, crypto {
    if len(args) == 0 {
        return cli_fail("usage: badapple vault <get|set|remove|list|import> [args]");
    }
    let key_hex = master_key_hex();
    if key_hex == "" {
        return cli_fail("failed to open vault; set BADAPPLE_VAULT_KEY, BADAPPLE_SLICKS_KEY_PATH, or BADAPPLE_SLICKS_SECRET");
    }
    let path = default_vault_path();
    let sub = args[0];
    if sub == "list" { return cli_vault_list(path); }
    if sub == "get" {
        if len(args) < 2 { return cli_fail("usage: badapple vault get <key>"); }
        return cli_vault_get(path, args[1]);
    }
    if sub == "set" {
        if len(args) < 3 { return cli_fail("usage: badapple vault set <key> <value>"); }
        return cli_vault_set(path, args[1], args[2]);
    }
    if sub == "remove" {
        if len(args) < 2 { return cli_fail("usage: badapple vault remove <key>"); }
        return cli_vault_remove(path, args[1]);
    }
    if sub == "import" {
        if len(args) < 2 { return cli_fail("usage: badapple vault import <dotenv-file>"); }
        return cli_vault_import(path, args[1]);
    }
    return cli_fail("unknown vault subcommand: " + sub);
}

host fn cli_vault_list(path: str) -> Json cap fs, crypto {
    let ks = vault_list(path);
    var arr: [Json] = [];
    for i in 0..len(ks) { push(arr, Json::Str(ks[i])); }
    return cli_ok(json_emit(Json::Obj({ "keys": Json::Arr(arr) })) + "\n");
}

host fn cli_vault_get(path: str, k: str) -> Json cap fs, crypto {
    let v = vault_get(path, k);
    if v == "" {
        return Json::Obj({
            "handled": Json::Bool(true),
            "code": Json::Num(1.0),
            "out": Json::Str("{\"key\": \"" + k + "\", \"error\": \"not found\"}\n"),
            "err": Json::Str("")
        });
    }
    return cli_ok(json_emit(Json::Obj({ "key": Json::Str(k), "value": Json::Str(v) })) + "\n");
}

host fn cli_vault_set(path: str, k: str, v: str) -> Json cap fs, crypto {
    if !vault_set(path, k, v) {
        return cli_fail("vault write failed");
    }
    return cli_ok("{\"status\": \"ok\", \"key\": \"" + k + "\"}\n");
}

host fn cli_vault_remove(path: str, k: str) -> Json cap fs, crypto {
    if !vault_remove(path, k) {
        return cli_fail("vault remove failed");
    }
    return cli_ok("{\"status\": \"removed\", \"key\": \"" + k + "\"}\n");
}

// import a dotenv file — parse in-language, write through vault_set.
host fn cli_vault_import(path: str, file: str) -> Json cap fs, crypto {
    if !fs_exists(file) { return cli_fail("failed to read " + file); }
    let content = read_file(file);
    let lines = split(content, "\n");
    var count = 0;
    for i in 0..len(lines) {
        let line = trim(lines[i]);
        if line != "" && !starts_with(line, "#") {
            let eq = find_sub(line, "=");
            if eq > 0 {
                let k = trim(slice_str(line, 0, eq));
                var v = trim(slice_str(line, eq + 1, len(line)));
                v = unquote(v);
                if k != "" {
                    if vault_set(path, k, v) { count = count + 1; }
                }
            }
        }
    }
    return cli_ok("{\"status\": \"ok\", \"imported\": " + str(count) + "}\n");
}

host det fn unquote(s: str) -> str {
    if len(s) >= 2 {
        let f = slice_str(s, 0, 1);
        let l = slice_str(s, len(s) - 1, len(s));
        if (f == "\"" && l == "\"") || (f == "'" && l == "'") {
            return slice_str(s, 1, len(s) - 1);
        }
    }
    return s;
}

// ── strategy — the non-interactive surface ─────────────────────────────
// learn / record / match / list run in-language. prune/repair/adopt carry
// interactive confirmation or proposal-file mechanics → fall through to Rust.

host fn cli_strategy(args: [str]) -> Json cap fs, crypto {
    var sub = "list";
    if len(args) > 0 { sub = args[0]; }
    if sub == "list" { return cli_strategy_list(); }
    if sub == "learn" { return cli_strategy_learn(args); }
    if sub == "record" { return cli_strategy_record(args); }
    if sub == "match" { return cli_strategy_match(args); }
    return cli_unhandled();
}

host det fn strategies_db() -> str {
    let e = env_get("BADAPPLE_STRATEGIES");
    if e != "" { return e; }
    let home = env_get("HOME");
    if home == "" { return "/tmp/.bad_apple/strategies.redb"; }
    return home + "/.bad_apple/strategies.redb";
}

host det fn flag_value(args: [str], name: str) -> str {
    var i = 0;
    while i < len(args) {
        if args[i] == name && i + 1 < len(args) { return args[i + 1]; }
        i = i + 1;
    }
    return "";
}

host det fn positional_args(args: [str]) -> [str] {
    var out_v: [str] = [];
    var i = 1;
    while i < len(args) {
        if starts_with(args[i], "--") {
            i = i + 2;
        } else {
            push(out_v, args[i]);
            i = i + 1;
        }
    }
    return out_v;
}

host fn cli_strategy_list() -> Json cap fs, crypto {
    let all = db_all(strategies_db());
    var rows = "";
    for i in 0..len(all) {
        rows = rows + strategy_row(all[i]);
    }
    return cli_ok(rows);
}

host det fn strategy_row(st: Json) -> str {
    match st {
        Json::Obj(m) => {
            let k = jstr(m, "key");
            let rel = jnumf(m, "reliability");
            let uses = jnumf(m, "uses");
            let prob = jstr(m, "problem");
            return k + "\t" + str(rel) + "\t" + str(uses) + " use(s)\t" + prob + "\n";
        }
        _ => { return ""; }
    }
}

host fn cli_strategy_learn(args: [str]) -> Json cap fs, crypto {
    let pos = positional_args(args);
    if len(pos) < 1 { return cli_fail("usage: badapple strategy learn <key> --problem <p> [--lang <l>] [--code <c>]"); }
    let problem = flag_value(args, "--problem");
    if problem == "" { return cli_fail("usage: badapple strategy learn <key> --problem <p> [--lang <l>] [--code <c>]"); }
    var lang = flag_value(args, "--lang");
    if lang == "" { lang = "shell"; }
    let code = flag_value(args, "--code");
    let st = Json::Obj({
        "key": Json::Str(pos[0]),
        "problem": Json::Str(problem),
        "language": Json::Str(lang),
        "code": Json::Str(code),
        "uses": Json::Num(0.0),
        "successes": Json::Num(0.0),
        "reliability": Json::Num(0.5)
    });
    if !db_put(strategies_db(), st) { return cli_fail("strategy write failed"); }
    return cli_ok("learned strategy '" + pos[0] + "'\n");
}

host fn cli_strategy_record(args: [str]) -> Json cap fs, crypto {
    let pos = positional_args(args);
    if len(pos) < 2 {
        return cli_fail("usage: badapple strategy record <key> <ok|fail> [--problem p] [--lang l] [--code c]");
    }
    let key = pos[0];
    var ok = true;
    if pos[1] == "ok" || pos[1] == "success" { ok = true; }
    else if pos[1] == "fail" || pos[1] == "failure" { ok = false; }
    else { return cli_fail("outcome must be ok|fail, got '" + pos[1] + "'"); }
    var st = db_get(strategies_db(), key);
    match st {
        Json::Obj(_) => { }
        _ => {
            var prob = flag_value(args, "--problem");
            if prob == "" { prob = key; }
            var lang = flag_value(args, "--lang");
            if lang == "" { lang = "shell"; }
            st = Json::Obj({
                "key": Json::Str(key),
                "problem": Json::Str(prob),
                "language": Json::Str(lang),
                "code": Json::Str(flag_value(args, "--code")),
                "uses": Json::Num(0.0),
                "successes": Json::Num(0.0),
                "reliability": Json::Num(0.5)
            });
        }
    }
    st = strategy_record_outcome(st, ok);
    if !db_put(strategies_db(), st) { return cli_fail("strategy write failed"); }
    match st {
        Json::Obj(m) => {
            return cli_ok(key + ": reliability " + str(jnumf(m, "reliability"))
                + " over " + str(jnumf(m, "uses")) + " use(s)\n");
        }
        _ => { return cli_fail("record failed"); }
    }
}

host fn cli_strategy_match(args: [str]) -> Json cap fs, crypto {
    let pos = positional_args(args);
    if len(pos) == 0 { return cli_fail("usage: badapple strategy match <problem text>"); }
    var problem = "";
    for i in 0..len(pos) {
        if i > 0 { problem = problem + " "; }
        problem = problem + pos[i];
    }
    let found = best_match(strategies_db(), problem);
    match found {
        Json::Obj(m) => {
            let out_j = Json::Obj({
                "key": m["key"],
                "problem": m["problem"],
                "language": m["language"],
                "code": m["code"],
                "uses": m["uses"],
                "reliability": m["reliability"]
            });
            return cli_ok(json_emit(out_j) + "\n");
        }
        _ => { return cli_fail("no matching strategy"); }
    }
}
