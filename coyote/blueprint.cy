// blueprint.cy — the neuro-symbolic + thermodynamic layer, ported from
// src/production_blueprint.rs.
//
// What moved: the CausalGraph (nodes/edges/traversal), the governor math
// (stress → sparsity → throttle), the HomeostaticController learning-rate
// loop, thermal sampling, and the workspace broadcast selection. The graph
// is persisted through the kv externs — the Rust version lived in a Mutex
// and died with the process; ours is learned state, it survives.
//
// What stayed shell-side: nothing essential. sys_exec supplies the
// sysctl/pmset/memory_pressure probes under `cap proc`.

cap fs, proc

import "sys.cy";
import "strlib.cy";

host det fn cg_db_path() -> str {
    var d = env_get("BADAPPLE_DATA_DIR");
    if d == "" { d = env_get("HOME") + "/.bad_apple"; }
    return d + "/causal.redb";
}

// ── causal graph ───────────────────────────────────────────────────────
// Json representation: {"nodes": [str], "edges": [{"s": i, "rel": str, "d": i}]}
// Relations use the Rust names verbatim: ENFORCES PROTECTS ENABLES CAUSES
// VIOLATES DEPENDS_ON.

host det fn cg_inverse(rel: str) -> str {
    if rel == "ENFORCES" { return "VIOLATED BY"; }
    if rel == "PROTECTS" { return "ENDANGERED BY"; }
    if rel == "ENABLES" { return "BLOCKED BY"; }
    if rel == "CAUSES" { return "RESULTED FROM"; }
    if rel == "VIOLATES" { return "VIOLATES"; }
    return "REQUIRED BY"; // DEPENDS_ON
}

host det fn cg_new() -> Json {
    let e: [Json] = [];
    let f: [Json] = [];
    return Json::Obj({ "nodes": Json::Arr(e), "edges": Json::Arr(f) });
}

host det fn cg_nodes(g: Json) -> [Json] {
    match g {
        Json::Obj(m) => {
            if has(m, "nodes") {
                match m["nodes"] { Json::Arr(a) => { return a; } _ => { } }
            }
        }
        _ => { }
    }
    let e: [Json] = [];
    return e;
}

host det fn cg_edges(g: Json) -> [Json] {
    match g {
        Json::Obj(m) => {
            if has(m, "edges") {
                match m["edges"] { Json::Arr(a) => { return a; } _ => { } }
            }
        }
        _ => { }
    }
    let e: [Json] = [];
    return e;
}

// node index, adding if absent — mutates the nodes Arr in place (vec is
// interior-mutable behind the enum).
host det fn cg_node_idx(g: Json, name: str) -> i64 {
    let nodes = cg_nodes(g);
    for i in 0..len(nodes) {
        match nodes[i] {
            Json::Str(n) => { if n == name { return i64(i); } }
            _ => { }
        }
    }
    push(nodes, Json::Str(name));
    return i64(len(nodes) - 1);
}

host det fn cg_add_relation(g: Json, src: str, rel: str, dst: str) -> Json {
    let s = cg_node_idx(g, src);
    let d = cg_node_idx(g, dst);
    push(cg_edges(g), Json::Obj({
        "s": Json::Num(f64(s)),
        "rel": Json::Str(rel),
        "d": Json::Num(f64(d)),
    }));
    return g;
}

// match-priority order — specific technical terms precede generic ones.
host det fn cg_primitives() -> [str] {
    return ["vram","tokenizer","socket","ledger","identity","slicks",
        "menubar","dashboard","tts","ears","microphone","speech","screen",
        "index","blocklist","permission","launch","agent","disk","mesh",
        "peer","council","approval","firewall","cache","data_dir","recall",
        "model","engine","memory"];
}

// the built-in causal map — bad_apple_default() verbatim.
host det fn cg_default() -> Json {
    var g = cg_new();
    let prims = cg_primitives();
    for i in 0..len(prims) {
        cg_node_idx(g, prims[i]);
    }
    g = cg_add_relation(g, "apple_intelligence_bridge", "ENABLES", "llm_oracle");
    g = cg_add_relation(g, "llm_oracle", "ENABLES", "planning_head");
    g = cg_add_relation(g, "llm_oracle", "ENABLES", "tool_synthesis");
    g = cg_add_relation(g, "candle_brain", "ENABLES", "local_conscience");
    g = cg_add_relation(g, "local_conscience", "PROTECTS", "goal_integrity");
    g = cg_add_relation(g, "conscience_oracle", "ENFORCES", "safety_policy");
    g = cg_add_relation(g, "safety_policy", "PROTECTS", "system_asset");
    g = cg_add_relation(g, "wild_workspace", "PROTECTS", "system_asset");
    g = cg_add_relation(g, "transfer_evaluator", "ENABLES", "skill_memory");
    g = cg_add_relation(g, "skill_memory", "ENABLES", "planning_head");
    g = cg_add_relation(g, "strategy_library", "DEPENDS_ON", "skill_memory");

    g = cg_add_relation(g, "launch_plist", "ENABLES", "socket");
    g = cg_add_relation(g, "stale_engine_process", "VIOLATES", "socket");
    g = cg_add_relation(g, "engine", "DEPENDS_ON", "socket");
    g = cg_add_relation(g, "gatekeeper", "DEPENDS_ON", "socket");
    g = cg_add_relation(g, "cli", "DEPENDS_ON", "socket");
    g = cg_add_relation(g, "vram", "ENABLES", "model");
    g = cg_add_relation(g, "memory_pressure", "VIOLATES", "vram");
    g = cg_add_relation(g, "stale_engine_process", "CAUSES", "memory_pressure");
    g = cg_add_relation(g, "model", "ENABLES", "generation");
    g = cg_add_relation(g, "model", "ENABLES", "engine");
    g = cg_add_relation(g, "socket", "ENABLES", "engine");
    g = cg_add_relation(g, "memory_pressure", "VIOLATES", "memory");
    g = cg_add_relation(g, "memory", "ENABLES", "vram");
    g = cg_add_relation(g, "model", "DEPENDS_ON", "vram");
    g = cg_add_relation(g, "model_bundle", "ENABLES", "tokenizer");
    g = cg_add_relation(g, "tokenizer", "ENABLES", "index");
    g = cg_add_relation(g, "tokenizer", "ENABLES", "engine");
    g = cg_add_relation(g, "index", "DEPENDS_ON", "tokenizer");
    g = cg_add_relation(g, "engine", "DEPENDS_ON", "tokenizer");
    g = cg_add_relation(g, "index", "ENABLES", "recall");
    g = cg_add_relation(g, "helper_binary", "ENABLES", "recall");
    g = cg_add_relation(g, "recall", "DEPENDS_ON", "index");
    g = cg_add_relation(g, "hf_cache", "ENABLES", "model");
    g = cg_add_relation(g, "app_bundle", "PROTECTS", "helper_binary");
    g = cg_add_relation(g, "secure_enclave", "ENABLES", "identity");
    g = cg_add_relation(g, "identity_agent", "ENABLES", "identity");
    g = cg_add_relation(g, "launch_plist", "ENABLES", "identity_agent");
    g = cg_add_relation(g, "identity", "ENABLES", "slicks");
    g = cg_add_relation(g, "slicks", "DEPENDS_ON", "identity");
    g = cg_add_relation(g, "peer_tls", "DEPENDS_ON", "identity");
    g = cg_add_relation(g, "launchd_service", "ENABLES", "launch");
    g = cg_add_relation(g, "launch_plist", "ENABLES", "agent");
    g = cg_add_relation(g, "platform_installer", "ENABLES", "data_dir");
    g = cg_add_relation(g, "data_dir", "ENABLES", "ledger");
    g = cg_add_relation(g, "disk", "ENABLES", "ledger");
    g = cg_add_relation(g, "state_growth", "VIOLATES", "disk");
    g = cg_add_relation(g, "ledger", "ENABLES", "ify");
    g = cg_add_relation(g, "ledger", "ENABLES", "audit");
    g = cg_add_relation(g, "ify", "DEPENDS_ON", "ledger");
    g = cg_add_relation(g, "audit", "DEPENDS_ON", "ledger");
    g = cg_add_relation(g, "bge_embedding_model", "ENABLES", "cache");
    g = cg_add_relation(g, "microphone", "ENABLES", "ears");
    g = cg_add_relation(g, "speech", "ENABLES", "ears");
    g = cg_add_relation(g, "menubar", "ENABLES", "ears");
    g = cg_add_relation(g, "ears", "DEPENDS_ON", "menubar");
    g = cg_add_relation(g, "ears", "ENABLES", "ambient_hearing");
    g = cg_add_relation(g, "screen", "ENABLES", "ocular");
    g = cg_add_relation(g, "vision_model", "ENABLES", "ocular");
    g = cg_add_relation(g, "user_consent", "ENABLES", "permission");
    g = cg_add_relation(g, "permission", "ENABLES", "microphone");
    g = cg_add_relation(g, "permission", "ENABLES", "speech");
    g = cg_add_relation(g, "permission", "ENABLES", "screen");
    g = cg_add_relation(g, "launch_plist", "ENABLES", "menubar");
    g = cg_add_relation(g, "launch_plist", "ENABLES", "tts");
    g = cg_add_relation(g, "launch_plist", "ENABLES", "dashboard");
    g = cg_add_relation(g, "launch_plist", "ENABLES", "ify");
    g = cg_add_relation(g, "council", "ENABLES", "approval");
    g = cg_add_relation(g, "engine", "ENABLES", "council");
    g = cg_add_relation(g, "firewall", "PROTECTS", "secrets");
    g = cg_add_relation(g, "firewall", "DEPENDS_ON", "blocklist");
    g = cg_add_relation(g, "blocklist", "ENABLES", "firewall");
    g = cg_add_relation(g, "identity", "ENABLES", "peer_tls");
    g = cg_add_relation(g, "peer_tls", "ENABLES", "peer");
    g = cg_add_relation(g, "peer", "ENABLES", "mesh");
    g = cg_add_relation(g, "mesh", "ENABLES", "delegated_inference");
    return g;
}

// ── persistence — the Rust graph lived in a Mutex; ours is durable ─────

host fn cg_load() -> Json cap fs {
    let raw = kv_get(cg_db_path(), "graph");
    if raw != "" {
        let g = json_parse(raw);
        match g {
            Json::Obj(_) => { return g; }
            _ => { }
        }
    }
    return cg_default();
}

host fn cg_save(g: Json) -> bool cap fs {
    return kv_put(cg_db_path(), "graph", json_emit(g));
}

// record a relation — strategy put path calls this instead of the
// shell-side Mutex graph. Persists immediately.
host fn cg_relate(from: str, rel: str, to: str) -> bool cap fs {
    var g = cg_load();
    g = cg_add_relation(g, from, str_upper(rel), to);
    return cg_save(g);
}

// backward DFS from a failed node — explain_failure() verbatim.
host det fn cg_explain(g: Json, failed: str) -> str {
    let nodes = cg_nodes(g);
    var start = -1;
    for i in 0..len(nodes) {
        match nodes[i] {
            Json::Str(n) => { if n == failed { start = i64(i); } }
            _ => { }
        }
    }
    if start < 0 { return ""; }
    let edges = cg_edges(g);
    var visited: {str: bool} = {};
    var stack: [i64] = [start];
    var root_used: {str: bool} = {};
    var antecedents: [str] = [];
    while len(stack) > 0 {
        let idx = stack[len(stack) - 1];
        var rest: [i64] = [];
        for si in 0..(len(stack) - 1) { push(rest, stack[si]); }
        stack = rest;
        let key = str(idx);
        if !has(visited, key) {
            visited[key] = true;
            for ei in 0..len(edges) {
                match edges[ei] {
                    Json::Obj(e) => {
                        var es = -1;
                        var ed = -1;
                        var rel = "";
                        if has(e, "s") { match e["s"] { Json::Num(n) => { es = i64(n); } _ => { } } }
                        if has(e, "d") { match e["d"] { Json::Num(n) => { ed = i64(n); } _ => { } } }
                        if has(e, "rel") { match e["rel"] { Json::Str(r) => { rel = r; } _ => { } } }
                        if ed == idx && !has(visited, str(es)) {
                            var sname = "";
                            var dname = "";
                            match nodes[es] { Json::Str(n) => { sname = n; } _ => { } }
                            match nodes[idx] { Json::Str(n) => { dname = n; } _ => { } }
                            if has(root_used, key) {
                                push(antecedents, "[" + sname + "] " + rel + " [" + dname + "]");
                            } else {
                                push(antecedents, "[" + sname + "] " + cg_inverse(rel) + " [" + dname + "]");
                                root_used[key] = true;
                            }
                            push(stack, es);
                        }
                    }
                    _ => { }
                }
            }
        }
    }
    if len(antecedents) == 0 {
        return "Failure at isolated node [" + failed + "]";
    }
    var chain = "Causal chain for failure at [" + failed + "]:\n";
    for i in 0..len(antecedents) {
        chain = chain + antecedents[i];
        if i < len(antecedents) - 1 { chain = chain + "\n"; }
    }
    return chain;
}

// first node name (in priority order) appearing in the failure text.
host det fn cg_find_broken(g: Json, failure: str) -> str {
    let nodes = cg_nodes(g);
    let low = str_lower(failure);
    for i in 0..len(nodes) {
        match nodes[i] {
            Json::Str(n) => {
                if contains(low, str_lower(n)) { return n; }
            }
            _ => { }
        }
    }
    return "";
}

// the ify.explain path: load graph → find primitive → explain. "" when
// nothing matches (mirrors Option::None).
host fn cg_explain_text(text: str) -> str cap fs {
    let g = cg_load();
    let prim = cg_find_broken(g, text);
    if prim == "" { return ""; }
    return cg_explain(g, prim);
}

// ── thermodynamic governor ─────────────────────────────────────────────

host det fn governor_update(st: Json) -> Json {
    var cpu = 0.0;
    var mem = 0.0;
    var batt = 1.0;
    var temp = 0.0;
    match st {
        Json::Obj(s) => {
            if has(s, "cpu_usage") { match s["cpu_usage"] { Json::Num(n) => { cpu = n; } _ => { } } }
            if has(s, "memory_pressure") { match s["memory_pressure"] { Json::Num(n) => { mem = n; } _ => { } } }
            if has(s, "battery_health") { match s["battery_health"] { Json::Num(n) => { batt = n; } _ => { } } }
            if has(s, "cpu_temp") { match s["cpu_temp"] { Json::Num(n) => { temp = n; } _ => { } } }
        }
        _ => { }
    }
    var stress = cpu * 0.25 + mem * 0.25 + (1.0 - batt) * 0.35 + temp * 0.15;
    if stress < 0.0 { stress = 0.0; }
    if stress > 1.0 { stress = 1.0; }
    var sparsity = stress;
    if sparsity > 0.9 { sparsity = 0.9; }
    return Json::Obj({
        "stress": Json::Num(stress),
        "target_sparsity": Json::Num(sparsity),
        "throttle": Json::Bool(stress > 0.7),
    });
}

host det fn throttle_lr(base: f64, stress: f64) -> f64 {
    var f = 1.0 - stress * 0.8;
    if f < 0.0001 { f = 0.0001; }
    if f > 1.0 { f = 1.0; }
    return base * f;
}

// keep the top (1 - target_sparsity) activations by |x|, zero the rest.
// vec is a [f64]; the mask approach avoids sorting machinery.
host det fn sparsify(v: [f64], target_sparsity: f64) -> [f64] {
    if target_sparsity <= 0.0 || len(v) == 0 { return v; }
    var keep_ratio = 1.0 - target_sparsity;
    if keep_ratio < 0.0 { keep_ratio = 0.0; }
    if keep_ratio > 1.0 { keep_ratio = 1.0; }
    var keep_count = i64(f64(len(v)) * keep_ratio);
    if f64(keep_count) < f64(len(v)) * keep_ratio { keep_count = keep_count + 1; } // ceil
    var mags: [f64] = [];
    for i in 0..len(v) {
        var x = v[i];
        if x < 0.0 { x = 0.0 - x; }
        push(mags, x);
    }
    // find the keep_count-th largest magnitude by threshold descent —
    // selection, not sort (n is small in practice).
    var thresh = 0.0;
    var got = 0;
    // max magnitude first
    var mx = 0.0;
    for i in 0..len(mags) { if mags[i] > mx { mx = mags[i]; } }
    // walk candidate thresholds: repeatedly find the largest magnitude
    // still above thresh — keep_count passes, O(k·n) worst case.
    var picked: [bool] = [];
    for i in 0..len(v) { push(picked, false); }
    for k in 0..keep_count {
        var best = -1;
        var bv = -1.0;
        for i in 0..len(mags) {
            if !picked[i] && mags[i] > bv {
                bv = mags[i];
                best = i64(i);
            }
        }
        if best >= 0 { picked[best] = true; }
    }
    var result_v: [f64] = [];
    for i in 0..len(v) {
        if picked[i] { push(result_v, v[i]); } else { push(result_v, 0.0); }
    }
    return result_v;
}

// homeostatic loop — returns the new active learning rate.
host det fn homeostatic_step(active_lr: f64, empirical_loss: f64, stress: f64) -> f64 {
    var lr = active_lr;
    let err = empirical_loss - 2.5; // TARGET_STABILITY
    if err > 0.0 {
        lr = lr + err * 0.001;
    } else {
        var a = err;
        if a < 0.0 { a = 0.0 - a; }
        lr = lr - a * 0.0005;
    }
    if lr < 0.0001 { lr = 0.0001; }
    if lr > 0.001 { lr = 0.001; } // MAXIMUM_META_RATE
    return throttle_lr(lr, stress);
}

// ── thermal sampling — sysctl/pmset through cap proc ───────────────────

host det fn no_args() -> [str] { let e: [str] = []; return e; }

host det fn thermal_path() -> str {
    var p = env_get("BADAPPLE_THERMAL_FILE");
    if p == "" { p = "/var/lib/bad_apple/thermal.json"; }
    return p;
}

host fn cmd_stdout(cmd: str, args: [str]) -> str cap proc {
    match sys_exec(cmd, args) {
        Json::Obj(r) => {
            var code = 0.0;
            if has(r, "code") { match r["code"] { Json::Num(n) => { code = n; } _ => { } } }
            if code == 0.0 {
                if has(r, "out") {
                    match r["out"] { Json::Str(o) => { return o; } _ => { } }
                }
            }
        }
        _ => { }
    }
    return "";
}

host fn sample_cpu_usage() -> f64 cap proc {
    var load = 0.0;
    let lout = cmd_stdout("sysctl", ["-n", "vm.loadavg"]);
    // "{ 2.31 2.05 1.87 }" — first parseable token is the 1-min load
    let toks = split_ws(lout);
    var found_load = false;
    for i in 0..len(toks) {
        if !found_load {
            let t = trim(toks[i]);
            let n = parse_num(t);
            if n > 0.0 {
                load = n;
                found_load = true;
            }
        }
    }
    var cores = 1.0;
    let cout = cmd_stdout("sysctl", ["-n", "hw.ncpu"]);
    let c = parse_num(trim(cout));
    if c >= 1.0 { cores = c; }
    var u = load / cores;
    if u < 0.0 { u = 0.0; }
    if u > 1.0 { u = 1.0; }
    return u;
}

host fn sample_memory_pressure() -> f64 cap proc {
    let stdout_s = cmd_stdout("memory_pressure", no_args());
    let lines = split(stdout_s, "\n");
    for i in 0..len(lines) {
        if contains(lines[i], "free percentage") {
            let toks = split_ws(lines[i]);
            for j in 0..len(toks) {
                let t = toks[j];
                if ends_with(t, "%") {
                    let n = parse_num(slice_str(t, 0, len(t) - 1));
                    if n > 0.0 {
                        var p = 1.0 - n / 100.0;
                        if p < 0.0 { p = 0.0; }
                        if p > 1.0 { p = 1.0; }
                        return p;
                    }
                }
            }
        }
    }
    return 0.0;
}

host fn sample_battery_health() -> f64 cap proc {
    let stdout_s = cmd_stdout("pmset", ["-g", "batt"]);
    if stdout_s == "" { return 1.0; }
    if !contains(stdout_s, "Battery Power") && !contains(stdout_s, "discharging") {
        return 1.0;
    }
    let toks = split_ws(stdout_s);
    for i in 0..len(toks) {
        var t = toks[i];
        if ends_with(t, "%;") { t = slice_str(t, 0, len(t) - 2); }
        else { if ends_with(t, "%") { t = slice_str(t, 0, len(t) - 1); } }
        let n = parse_num(t);
        if n > 0.0 && (ends_with(toks[i], "%") || ends_with(toks[i], "%;")) {
            var p = n / 100.0;
            if p > 1.0 { p = 1.0; }
            return p;
        }
    }
    return 1.0;
}

host fn sample_cpu_thermal() -> f64 cap proc {
    let stdout_s = cmd_stdout("pmset", ["-g", "therm"]);
    let lines = split(stdout_s, "\n");
    for i in 0..len(lines) {
        if contains(lines[i], "CPU_Speed_Limit") {
            let parts = split(lines[i], "=");
            if len(parts) > 1 {
                let n = parse_num(trim(parts[1]));
                var t = 1.0 - n / 100.0;
                if t < 0.0 { t = 0.0; }
                if t > 1.0 { t = 1.0; }
                return t;
            }
        }
    }
    return 0.0;
}

host fn thermal_sample_state() -> Json cap proc {
    return Json::Obj({
        "cpu_usage": Json::Num(sample_cpu_usage()),
        "memory_pressure": Json::Num(sample_memory_pressure()),
        "battery_health": Json::Num(sample_battery_health()),
        "cpu_temp": Json::Num(sample_cpu_thermal()),
    });
}

// probe: sample → governor → persist report → return it.
host fn probe_thermal() -> Json cap fs, proc {
    let st = thermal_sample_state();
    let gov = governor_update(st);
    var cpu = 0.0;
    var mem = 0.0;
    var batt = 1.0;
    var temp = 0.0;
    var stress = 0.0;
    var sparsity = 0.0;
    var throttle = false;
    match st {
        Json::Obj(s) => {
            if has(s, "cpu_usage") { match s["cpu_usage"] { Json::Num(n) => { cpu = n; } _ => { } } }
            if has(s, "memory_pressure") { match s["memory_pressure"] { Json::Num(n) => { mem = n; } _ => { } } }
            if has(s, "battery_health") { match s["battery_health"] { Json::Num(n) => { batt = n; } _ => { } } }
            if has(s, "cpu_temp") { match s["cpu_temp"] { Json::Num(n) => { temp = n; } _ => { } } }
        }
        _ => { }
    }
    match gov {
        Json::Obj(g) => {
            if has(g, "stress") { match g["stress"] { Json::Num(n) => { stress = n; } _ => { } } }
            if has(g, "target_sparsity") { match g["target_sparsity"] { Json::Num(n) => { sparsity = n; } _ => { } } }
            if has(g, "throttle") { match g["throttle"] { Json::Bool(b) => { throttle = b; } _ => { } } }
        }
        _ => { }
    }
    let report = Json::Obj({
        "stress": Json::Num(stress),
        "throttle": Json::Bool(throttle),
        "target_sparsity": Json::Num(sparsity),
        "cpu_usage": Json::Num(cpu),
        "memory_pressure": Json::Num(mem),
        "battery_health": Json::Num(batt),
        "cpu_temp": Json::Num(temp),
        "ts": Json::Num(f64(time_ms()) / 1000.0),
    });
    let p = thermal_path();
    fs_mkdir(slice_str(p, 0, rfind_sub(p, "/")));
    write_file(p, json_emit(report));
    return report;
}

// ── neuro-symbolic engine — rule register + activation gating ──────────
// rules live in kv: key "ns:<premise>" → conclusion.

host fn ns_add_rule(premise: str, conclusion: str) -> bool cap fs {
    return kv_put(cg_db_path(), "ns:" + premise, conclusion);
}

host det fn ns_rules() -> Json cap fs {
    return kv_scan_prefix(cg_db_path(), "ns:");
}

// deliberate_execution_chain — premise activation > 0.85 → conclusion.
// activations: {str: f64-as-Json}
host fn deliberate(activations: {str: Json}) -> [str] cap fs {
    var results: [str] = [];
    match ns_rules() {
        Json::Arr(rs) => {
            for i in 0..len(rs) {
                match rs[i] {
                    Json::Obj(pair) => {
                        var k = "";
                        var v = "";
                        if has(pair, "key") { match pair["key"] { Json::Str(x) => { k = x; } _ => { } } }
                        if has(pair, "value") { match pair["value"] { Json::Str(x) => { v = x; } _ => { } } }
                        if len(k) > 3 {
                            let prem = slice_str(k, 3, len(k));
                            if has(activations, prem) {
                                match activations[prem] {
                                    Json::Num(a) => {
                                        if a > 0.85 { push(results, v); }
                                    }
                                    _ => { }
                                }
                            }
                        }
                    }
                    _ => { }
                }
            }
        }
        _ => { }
    }
    return results;
}

// ── global workspace — broadcast selection ─────────────────────────────
// The Rust version spawned a callback per channel; the .cy port returns
// the winning (payload, saliency) so the caller dispatches — the threshold
// gate is the policy that matters.

host det fn select_broadcast(signals: {str: Json}) -> Json {
    var best_payload = "";
    var best_sal = -1.0;
    let ks = keys(signals);
    for i in 0..len(ks) {
        match signals[ks[i]] {
            Json::Arr(pair) => {
                if len(pair) == 2 {
                    match pair[1] {
                        Json::Num(sal) => {
                            if sal > best_sal {
                                best_sal = sal;
                                match pair[0] { Json::Str(p) => { best_payload = p; } _ => { } }
                            }
                        }
                        _ => { }
                    }
                }
            }
            _ => { }
        }
    }
    if best_sal < 0.82 { return Json::Null; } // ATTENTION_THRESHOLD
    return Json::Obj({
        "payload": Json::Str(best_payload),
        "saliency": Json::Num(best_sal),
    });
}
