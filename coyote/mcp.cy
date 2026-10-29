// mcp.cy — the MCP marketplace registry, ported from src/mcp_marketplace.rs.
//
// Portable: the catalog (JSON on disk), install/uninstall/list/get/upsert
// state, and ALL admission policy — id shape, command allowlisting,
// arg metacharacter screening, env blocklist. What stays shell: process
// spawn/stdio piping (start/stop/send/request) — a running MCP server is a
// child process, and child lifecycles are the shell's job.

cap fs

import "sys.cy";
import "strlib.cy";

// ── admission policy ───────────────────────────────────────────────────

host det fn mcp_id_ok(id: str) -> str {
    if len(id) == 0 || len(id) > 64 {
        return "MCP server id must be 1-64 characters";
    }
    for i in 0..len(id) {
        let c = slice_str(id, i, i + 1);
        let ok = contains("abcdefghijklmnopqrstuvwxyz", str_lower(c))
            || contains("0123456789", c)
            || c == "_"
            || c == "-";
        if !ok {
            return "MCP server id must be ASCII alphanumeric, underscore, or hyphen";
        }
    }
    return "";
}

// commands that fetch/execute remote code — air-gap policy.
host det fn mcp_disallowed_commands() -> [str] {
    return ["npx","npm","pip","pip3","curl","wget","git","ssh","scp","ftp","telnet"];
}

host det fn mcp_meta_chars() -> str {
    return ";&|`$(){}[]<>!\\*?\"'";
}

host det fn mcp_arg_bad(arg: str) -> bool {
    let t = trim(arg);
    if t == "" { return true; }
    let mc = mcp_meta_chars();
    for i in 0..len(t) {
        if contains(mc, slice_str(t, i, i + 1)) { return true; }
    }
    if starts_with(t, "~") || contains(t, "..") { return true; }
    if t == "/" { return true; }
    return false;
}

// → "" ok or the rejection reason.
host det fn mcp_validate_command(command: str, args: [str]) -> str {
    if command == "" { return "MCP server command cannot be empty"; }
    // basename for the allowlist check
    var base = command;
    let sl = rfind_sub(command, "/");
    if sl >= 0 { base = slice_str(command, sl + 1, len(command)); }
    let bad = mcp_disallowed_commands();
    let lb = str_lower(base);
    for i in 0..len(bad) {
        if lb == bad[i] {
            return "MCP server command '" + command + "' is disallowed for air-gapped operation";
        }
    }
    let mc = mcp_meta_chars();
    for i in 0..len(command) {
        if contains(mc, slice_str(command, i, i + 1)) {
            return "MCP server command contains disallowed shell metacharacters";
        }
    }
    if starts_with(command, "~") || contains(command, "..") {
        return "MCP server command cannot use ~ or .. paths";
    }
    for i in 0..len(args) {
        if mcp_arg_bad(args[i]) {
            return "MCP server arg[" + str(i) + "] is disallowed: '" + args[i] + "'";
        }
    }
    return "";
}

host det fn mcp_env_blocked() -> [str] {
    return ["PATH","LD_PRELOAD","LD_LIBRARY_PATH","DYLD_INSERT_LIBRARIES",
        "DYLD_LIBRARY_PATH","DYLD_FALLBACK_LIBRARY_PATH","HOME","USER",
        "SHELL","SSH_AUTH_SOCK","SUDO_COMMAND","PS4"];
}

// is `k` a permitted env name? — factored out (nested-for slot bug).
host det fn mcp_env_key_bad(k: str, blocked: [str]) -> str {
    for j in 0..len(blocked) {
        if k == blocked[j] { return "MCP server environment variable " + k + " is blocked"; }
    }
    if starts_with(k, "BADAPPLE_") {
        return "MCP server environment variable name is not allowed";
    }
    for c in 0..len(k) {
        let ch = slice_str(k, c, c + 1);
        let ok = contains("abcdefghijklmnopqrstuvwxyz", str_lower(ch))
            || contains("0123456789", ch) || ch == "_";
        if !ok { return "MCP server environment variable name is not allowed"; }
    }
    return "";
}

host det fn mcp_validate_env(env: Json) -> str {
    let blocked = mcp_env_blocked();
    var ks: [str] = [];
    match env { Json::Obj(eo) => { ks = keys(eo); } _ => { } }
    for i in 0..len(ks) {
        let bad = mcp_env_key_bad(ks[i], blocked);
        if bad != "" { return bad; }
    }
    return "";
}

// filtered env — the Rust's allowed_env(): blocked keys dropped silently.
host det fn mcp_allowed_env(env: Json) -> {str: str} {
    let blocked = mcp_env_blocked();
    var out_env: {str: str} = {};
    var ks: [str] = [];
    match env { Json::Obj(eo) => { ks = keys(eo); } _ => { } }
    for i in 0..len(ks) {
        let k = ks[i];
        if !mcp_in_list(k, blocked) {
            match env { Json::Obj(eo) => {
                match eo[k] {
                    Json::Str(v) => { out_env[k] = v; }
                    _ => { }
                }
            } _ => { } }
        }
    }
    return out_env;
}

host det fn mcp_in_list(k: str, list: [str]) -> bool {
    for i in 0..len(list) {
        if k == list[i] { return true; }
    }
    return false;
}

// ── catalog — one JSON doc {servers: [...], version: N} on disk ────────

host det fn mcp_catalog_path() -> str {
    var d = env_get("BADAPPLE_DATA_DIR");
    if d == "" { d = "/var/lib/bad_apple"; }
    return d + "/mcp_catalog.json";
}

host fn mcp_load(path: str) -> Json cap fs {
    if !fs_exists(path) {
        return Json::Obj({ "servers": Json::Arr(empty_json_arr()), "version": Json::Num(0.0) });
    }
    let raw = read_file(path);
    let c = json_parse(raw);
    match c {
        Json::Obj(_) => { return c; }
        _ => { return Json::Obj({ "servers": Json::Arr(empty_json_arr()), "version": Json::Num(0.0) }); }
    }
}

host det fn empty_json_arr() -> [Json] { let e: [Json] = []; return e; }

host det fn empty_json_map() -> {str: Json} { let e: {str: Json} = {}; return e; }

host fn mcp_save(path: str, catalog: Json) -> bool cap fs {
    let parent = dir_of(path);
    fs_mkdir(parent);
    let tmp = path + ".tmp";
    write_file(tmp, json_emit(catalog));
    return fs_rename(tmp, path);
}

host det fn mcp_servers(catalog: Json) -> [Json] {
    match catalog {
        Json::Obj(c) => {
            if has(c, "servers") {
                match c["servers"] { Json::Arr(a) => { return a; } _ => { } }
            }
        }
        _ => { }
    }
    return empty_json_arr();
}

// the id of a server record, "" when missing.
host det fn mcp_record_id(server: Json) -> str {
    match server {
        Json::Obj(sm) => {
            if has(sm, "id") {
                match sm["id"] { Json::Str(x) => { return x; } _ => { } }
            }
        }
        _ => { }
    }
    return "";
}

// validate then upsert a server record — admission gate in-language.
host det fn mcp_record_str(server: Json, k: str) -> str {
    match server {
        Json::Obj(sm) => {
            if has(sm, k) {
                match sm[k] { Json::Str(x) => { return x; } _ => { } }
            }
        }
        _ => { }
    }
    return "";
}

host det fn mcp_record_args(server: Json) -> [str] {
    var args: [str] = [];
    match server {
        Json::Obj(sm) => {
            if has(sm, "args") {
                match sm["args"] {
                    Json::Arr(a) => {
                        for i in 0..len(a) {
                            match a[i] { Json::Str(x) => { push(args, x); } _ => { } }
                        }
                    }
                    _ => { }
                }
            }
        }
        _ => { }
    }
    return args;
}

host fn mcp_upsert(path: str, server: Json) -> Json cap fs {
    let id = mcp_record_str(server, "id");
    let command = mcp_record_str(server, "command");
    var envj = Json::Null;
    match server {
        Json::Obj(sm) => { if has(sm, "env") { envj = sm["env"]; } }
        _ => { return Json::Obj({ "error": Json::Str("server must be an object") }); }
    }
    let e1 = mcp_id_ok(id);
    if e1 != "" { return Json::Obj({ "error": Json::Str(e1) }); }
    let e2 = mcp_validate_command(command, mcp_record_args(server));
    if e2 != "" { return Json::Obj({ "error": Json::Str(e2) }); }
    match envj {
        Json::Obj(_) => {
            let e3 = mcp_validate_env(envj);
            if e3 != "" { return Json::Obj({ "error": Json::Str(e3) }); }
        }
        _ => { }
    }
    let cat = mcp_load(path);
    let svrs = mcp_servers(cat);
    // replace by id or append
    var found = false;
    for i in 0..len(svrs) {
        if mcp_record_id(svrs[i]) == id { svrs[i] = server; found = true; }
    }
    if !found { push(svrs, server); }
    let newcat = Json::Obj({ "servers": Json::Arr(svrs), "version": Json::Num(1.0) });
    if !mcp_save(path, newcat) { return Json::Obj({ "error": Json::Str("catalog save failed") }); }
    return Json::Obj({ "ok": Json::Bool(true), "id": Json::Str(id) });
}

host fn mcp_get(path: str, id: str) -> Json cap fs {
    let svrs = mcp_servers(mcp_load(path));
    for i in 0..len(svrs) {
        if mcp_record_id(svrs[i]) == id { return svrs[i]; }
    }
    return Json::Null;
}

// per-server filter for remove — kept when ids differ.
host det fn mcp_keep_or(server: Json, id: str) -> Json {
    if mcp_record_id(server) == id { return Json::Null; }
    return server;
}

host fn mcp_remove(path: str, id: str) -> Json cap fs {
    let cat = mcp_load(path);
    let svrs = mcp_servers(cat);
    var kept: [Json] = [];
    var had = false;
    for i in 0..len(svrs) {
        if mcp_record_id(svrs[i]) == id {
            had = true;
        } else {
            push(kept, svrs[i]);
        }
    }
    if !had { return Json::Obj({ "error": Json::Str("server not found: " + id) }); }
    mcp_save(path, Json::Obj({ "servers": Json::Arr(kept), "version": Json::Num(1.0) }));
    return Json::Obj({ "ok": Json::Bool(true) });
}

// per-server flag set — returns the (possibly) updated record + a hit flag.
host det fn mcp_flag1(server: Json, id: str, flag: str, val: bool) -> Json {
    if mcp_record_id(server) != id { return server; }
    match server {
        Json::Obj(sm) => {
            sm[flag] = Json::Bool(val);
            return Json::Obj(sm);
        }
        _ => { return server; }
    }
}

// set a flag on a server record — install/uninstall/enable/disable.
host fn mcp_set_flag(path: str, id: str, flag: str, val: bool) -> Json cap fs {
    let cat = mcp_load(path);
    let svrs = mcp_servers(cat);
    var had = false;
    for i in 0..len(svrs) {
        if mcp_record_id(svrs[i]) == id { had = true; }
        svrs[i] = mcp_flag1(svrs[i], id, flag, val);
    }
    if !had { return Json::Obj({ "error": Json::Str("server not found: " + id) }); }
    mcp_save(path, Json::Obj({ "servers": Json::Arr(svrs), "version": Json::Num(1.0) }));
    return Json::Obj({ "ok": Json::Bool(true) });
}

// ── the MCP JSON-RPC server — ported from src/mcp.rs ────────────────────
// Request policy (size, version, method dispatch, tool gates, arg-depth)
// runs in-language; the transport is the unix-socket extern surface and
// the agent call bridges to bad_apple_ipc::call_agent.

cap net
cap cortex

host det fn mcp_server_info() -> Json {
    return Json::Obj({
        "protocolVersion": Json::Str("2024-11-05"),
        "serverInfo": Json::Obj({
            "name": Json::Str("Bad Apple MCP"),
            "version": Json::Str("0.1.9")
        }),
        "capabilities": Json::Obj({
            "tools": Json::Obj({ "listChanged": Json::Bool(true) }),
            "resources": Json::Obj({ "subscribe": Json::Bool(false) }),
            "prompts": Json::Obj({ "listChanged": Json::Bool(false) })
        })
    });
}

host det fn mcp_tools() -> [Json] {
    var t: [Json] = [];
    push(t, Json::Obj({
        "name": Json::Str("runtime_status"),
        "description": Json::Str("Get the current Bad Apple runtime status."),
        "inputSchema": Json::Obj({ "type": Json::Str("object"), "properties": Json::Obj(empty_json_map()), "required": Json::Arr(empty_json_arr()) })
    }));
    push(t, Json::Obj({
        "name": Json::Str("invoke_tool"),
        "description": Json::Str("Invoke a Bad Apple local tool by name with arguments."),
        "inputSchema": Json::Obj({
            "type": Json::Str("object"),
            "properties": Json::Obj({
                "name": Json::Obj({ "type": Json::Str("string") }),
                "args": Json::Obj({ "type": Json::Str("object") })
            }),
            "required": Json::Arr(mcp_reqd())
        })
    }));
    push(t, Json::Obj({
        "name": Json::Str("list_models"),
        "description": Json::Str("List tracked and cached models."),
        "inputSchema": Json::Obj({ "type": Json::Str("object"), "properties": Json::Obj(empty_json_map()), "required": Json::Arr(empty_json_arr()) })
    }));
    return t;
}

host det fn mcp_reqd() -> [Json] {
    var r: [Json] = [];
    push(r, Json::Str("name"));
    return r;
}

// tools that mutate the system — gated to daemon approval.
host det fn mcp_dangerous() -> [str] {
    return ["run_shell","run_applescript","write_file","delete_file","index_documents","run_shortcut"];
}

// agent methods MCP may reach — governance mutations deliberately absent.
host det fn mcp_direct() -> [str] {
    return ["runtime_status","invoke_tool","p2p_peers","p2p_sync","inference","list_models","model_info"];
}

host det fn mcp_err(id: Json, code: i64, msg: str) -> Json {
    return Json::Obj({
        "jsonrpc": Json::Str("2.0"),
        "id": id,
        "error": Json::Obj({ "code": Json::Num(f64(code)), "message": Json::Str(msg) })
    });
}

host det fn mcp_ok(id: Json, result: Json) -> Json {
    return Json::Obj({
        "jsonrpc": Json::Str("2.0"),
        "id": id,
        "result": result
    });
}

host det fn mcp_req_id(req: Json) -> Json {
    match req {
        Json::Obj(r) => {
            if has(r, "id") { return r["id"]; }
        }
        _ => { }
    }
    return Json::Null;
}

// recursive depth/type guard — is_safe_value(0..8).
host det fn mcp_safe_value(v: Json, depth: i64) -> bool {
    if depth > 8 { return false; }
    match v {
        Json::Arr(a) => { return mcp_safe_arr(a, depth + 1); }
        Json::Obj(o) => { return mcp_safe_obj(o, depth + 1); }
        _ => { return true; }
    }
}

host det fn mcp_safe_arr(a: [Json], depth: i64) -> bool {
    for i in 0..len(a) {
        if !mcp_safe_value(a[i], depth) { return false; }
    }
    return true;
}

host det fn mcp_safe_obj(o: {str: Json}, depth: i64) -> bool {
    let ks = keys(o);
    for i in 0..len(ks) {
        if len(ks[i]) > 256 { return false; }
        if !mcp_safe_value(o[ks[i]], depth) { return false; }
    }
    return true;
}

// tools/call → the agent params the Rust builds, or an error Json.
host det fn mcp_call_gate(req: Json) -> Json {
    let id = mcp_req_id(req);
    var name = "";
    var args = Json::Null;
    var has_params = false;
    match req {
        Json::Obj(r) => {
            if has(r, "params") {
                match r["params"] {
                    Json::Obj(p) => {
                        has_params = true;
                        if has(p, "name") { name = jstrv(p["name"]); }
                        if has(p, "arguments") { args = p["arguments"]; }
                    }
                    _ => { }
                }
            }
        }
        _ => { }
    }
    if !has_params { return mcp_err(id, -32602, "params must be an object"); }
    if name == "" || len(name) > 256 {
        return mcp_err(id, -32602, "name is required or too long");
    }
    if !mcp_safe_value(args, 0) {
        return mcp_err(id, -32602, "arguments exceed depth or type limits");
    }
    if !mcp_in_list(name, mcp_direct()) {
        return mcp_err(id, -32602, "tool " + name + " is not exposed");
    }
    if mcp_in_list(name, mcp_dangerous()) {
        return mcp_err(id, -32602, "tool " + name + " requires daemon approval");
    }
    // invoke_tool wraps an inner tool — re-gate the inner name so the
    // dangerous check can't be bypassed.
    if name == "invoke_tool" {
        var inner = "";
        var inner_args = Json::Obj(empty_json_map());
        match args {
            Json::Obj(am) => {
                inner = jstrv(am["name"]);
                if has(am, "args") { inner_args = am["args"]; }
            }
            _ => { }
        }
        if mcp_in_list(inner, mcp_dangerous()) {
            return mcp_err(id, -32602, "tool " + inner + " requires daemon approval");
        }
        return Json::Obj({
            "name": Json::Str("invoke_tool"),
            "params": Json::Obj({ "name": Json::Str(inner), "args": inner_args }),
            "id": id
        });
    }
    return Json::Obj({
        "name": Json::Str(name),
        "params": args,
        "id": id
    });
}

host det fn jstrv(j: Json) -> str {
    match j { Json::Str(x) => { return x; } _ => { } }
    return "";
}

// one request line → response Json, or Null when the request is a
// notification that gets no response.
host det fn mcp_handle_request(line: str) -> Json {
    if len(line) > 65536 {
        return mcp_err(Json::Null, -32700, "request too large");
    }
    let req = json_parse(line);
    match req {
        Json::Obj(r) => {
            if !has(r, "jsonrpc") || jstrv(r["jsonrpc"]) != "2.0" {
                return mcp_err(mcp_req_id(req), -32600, "invalid jsonrpc version");
            }
            let method = jstrv(r["method"]);
            let id = mcp_req_id(req);
            if method == "initialize" { return mcp_ok(id, mcp_server_info()); }
            if method == "initialized" { return mcp_ok(id, Json::Null); }
            if method == "tools/list" {
                return mcp_ok(id, Json::Obj({ "tools": Json::Arr(mcp_tools()) }));
            }
            if method == "resources/list" {
                return mcp_ok(id, Json::Obj({ "resources": Json::Arr(empty_json_arr()) }));
            }
            if method == "prompts/list" {
                return mcp_ok(id, Json::Obj({ "prompts": Json::Arr(empty_json_arr()) }));
            }
            if method == "tools/call" {
                let gated = mcp_call_gate(req);
                match gated {
                    Json::Obj(g) => {
                        if has(g, "error") { return gated; }
                    }
                    _ => { }
                }
                return gated;  // caller runs the agent call (needs cortex cap)
            }
            return mcp_err(id, -32601, "method not found: " + method);
        }
        _ => {
            return mcp_err(Json::Null, -32700, "parse error");
        }
    }
}

// full tools/call — gate + agent round-trip.
host fn mcp_tools_call(req: Json) -> Json cap cortex {
    let gated = mcp_call_gate(req);
    match gated {
        Json::Obj(g) => {
            if has(g, "error") { return gated; }
            let name = jstrv(g["name"]);
            let params = g["params"];
            let id = g["id"];
            let res = mcp_agent_call(name, params);
            match res {
                Json::Obj(o) => {
                    if has(o, "error") {
                        return mcp_err(id, -32603, jstrv(o["error"]));
                    }
                }
                _ => { }
            }
            return mcp_ok(id, res);
        }
        _ => { return mcp_err(Json::Null, -32603, "call gate failed"); }
    }
}
