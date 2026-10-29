// config.cy — runtime configuration, ported from src/config.rs.
//
// Values load from BADAPPLE_* env vars with defaults; a JSON file at
// BADAPPLE_CONFIG_FILE overlays non-default fields on top.

import "sys.cy";
import "strlib.cy";

cap fs

struct Config {
    telemetry_port: i64,
    multi_agent_port_start: i64,
    multi_agent_port_end: i64,
    apple_intelligence_model: str,
    state_file: str,
    curriculum_dir: str,
    wild_workspace_dir: str,
    sled_db_path: str,
    metrics_log: str,
    skills_dir: str,
    tools_dir: str,
    clock_interval_secs: i64,
    max_engram_batch: i64,
    engram_timeout_ms: i64,
    multi_agent_secret: str,      // "" = unset (was Option)
    peer_nodes: [str],
    config_file: str,             // "" = unset
    wan_tcp_port: i64,
    wan_ws_port: i64,
    max_wan_peers: i64,
    peer_retry_base_ms: i64,
    peer_retry_max_ms: i64,
}

host det fn env_i64(name: str, default: i64) -> i64 {
    let v = env_get(name);
    if v == "" { return default; }
    let n = parse_num(v);
    if n != n { return default; }  // NaN check
    return i64(n);
}

host det fn env_str(name: str, default: str) -> str {
    let v = env_get(name);
    if v == "" { return default; }
    return v;
}

// peer list: separators , ; \n \r — flat char-scan, trim, drop empties
host det fn parse_peer_list(s: str) -> [str] {
    var acc: [str] = [];
    var cur = "";
    for i in 0..len(s) {
        let c = slice_str(s, i, i + 1);
        if c == "," || c == ";" || c == "\n" || c == "\r" {
            let p = trim(cur);
            if p != "" { push(acc, p); }
            cur = "";
        } else {
            cur = cur + c;
        }
    }
    let p = trim(cur);
    if p != "" { push(acc, p); }
    return acc;
}

host fn config_from_env() -> Config cap fs {
    var cf = env_get("BADAPPLE_CONFIG_FILE");
    if cf != "" && !fs_exists(cf) { cf = ""; }
    var c = Config {
        telemetry_port: env_i64("BADAPPLE_TELEMETRY_PORT", 8080),
        multi_agent_port_start: env_i64("BADAPPLE_MULTI_AGENT_PORT_START", 5001),
        multi_agent_port_end: env_i64("BADAPPLE_MULTI_AGENT_PORT_END", 5010),
        apple_intelligence_model: env_str("BADAPPLE_APPLE_INTELLIGENCE_MODEL", "apple-intelligence"),
        state_file: env_str("BADAPPLE_STATE_FILE", "state.json"),
        curriculum_dir: env_str("BADAPPLE_CURRICULUM_DIR", "curriculum"),
        wild_workspace_dir: env_str("BADAPPLE_WILD_WORKSPACE_DIR", "wild_workspace"),
        sled_db_path: env_str("BADAPPLE_SLED_DB_PATH", "strategy_db"),
        metrics_log: env_str("BADAPPLE_METRICS_LOG", "metrics.jsonl"),
        skills_dir: env_str("BADAPPLE_SKILLS_DIR", "skills"),
        tools_dir: env_str("BADAPPLE_TOOLS_DIR", "tools"),
        clock_interval_secs: env_i64("BADAPPLE_CLOCK_INTERVAL_SECS", 6),
        max_engram_batch: env_i64("BADAPPLE_MAX_ENGRAM_BATCH", 64),
        engram_timeout_ms: env_i64("BADAPPLE_ENGRAM_TIMEOUT_MS", 5),
        multi_agent_secret: env_get("BADAPPLE_MULTI_AGENT_SECRET"),
        peer_nodes: parse_peer_list(env_get("BADAPPLE_PEERS")),
        config_file: cf,
        wan_tcp_port: env_i64("BADAPPLE_WAN_TCP_PORT", 6001),
        wan_ws_port: env_i64("BADAPPLE_WAN_WS_PORT", 6002),
        max_wan_peers: env_i64("BADAPPLE_MAX_WAN_PEERS", 64),
        peer_retry_base_ms: env_i64("BADAPPLE_PEER_RETRY_BASE_MS", 250),
        peer_retry_max_ms: env_i64("BADAPPLE_PEER_RETRY_MAX_MS", 30000),
    };
    if cf != "" {
        match json_parse(read_file(cf)) {
            Json::Obj(o) => { c = merge_config(c, o); }
            _ => { }
        }
    }
    return c;
}

// JSON overlay — non-default fields win (same rule as merge_config).
host det fn merge_config(base: Config, o: {str: Json}) -> Config {
    var c = base;
    if has(o, "telemetry_port") { match o["telemetry_port"] { Json::Num(n) => { if n != 0.0 { c.telemetry_port = i64(n); } } _ => { } } }
    if has(o, "multi_agent_port_start") { match o["multi_agent_port_start"] { Json::Num(n) => { if n != 0.0 { c.multi_agent_port_start = i64(n); } } _ => { } } }
    if has(o, "multi_agent_port_end") { match o["multi_agent_port_end"] { Json::Num(n) => { if n != 0.0 { c.multi_agent_port_end = i64(n); } } _ => { } } }
    if has(o, "apple_intelligence_model") { match o["apple_intelligence_model"] { Json::Str(v) => { if v != "" { c.apple_intelligence_model = v; } } _ => { } } }
    if has(o, "state_file") { match o["state_file"] { Json::Str(v) => { if v != "" { c.state_file = v; } } _ => { } } }
    if has(o, "curriculum_dir") { match o["curriculum_dir"] { Json::Str(v) => { if v != "" { c.curriculum_dir = v; } } _ => { } } }
    if has(o, "wild_workspace_dir") { match o["wild_workspace_dir"] { Json::Str(v) => { if v != "" { c.wild_workspace_dir = v; } } _ => { } } }
    if has(o, "sled_db_path") { match o["sled_db_path"] { Json::Str(v) => { if v != "" { c.sled_db_path = v; } } _ => { } } }
    if has(o, "metrics_log") { match o["metrics_log"] { Json::Str(v) => { if v != "" { c.metrics_log = v; } } _ => { } } }
    if has(o, "skills_dir") { match o["skills_dir"] { Json::Str(v) => { if v != "" { c.skills_dir = v; } } _ => { } } }
    if has(o, "tools_dir") { match o["tools_dir"] { Json::Str(v) => { if v != "" { c.tools_dir = v; } } _ => { } } }
    if has(o, "clock_interval_secs") { match o["clock_interval_secs"] { Json::Num(n) => { if n != 0.0 { c.clock_interval_secs = i64(n); } } _ => { } } }
    if has(o, "max_engram_batch") { match o["max_engram_batch"] { Json::Num(n) => { if n != 0.0 { c.max_engram_batch = i64(n); } } _ => { } } }
    if has(o, "engram_timeout_ms") { match o["engram_timeout_ms"] { Json::Num(n) => { if n != 0.0 { c.engram_timeout_ms = i64(n); } } _ => { } } }
    if has(o, "multi_agent_secret") { match o["multi_agent_secret"] { Json::Str(v) => { if v != "" { c.multi_agent_secret = v; } } _ => { } } }
    if has(o, "peer_nodes") { match o["peer_nodes"] { Json::Arr(a) => { c.peer_nodes = json_str_list(a); } _ => { } } }
    if has(o, "wan_tcp_port") { match o["wan_tcp_port"] { Json::Num(n) => { if n != 0.0 { c.wan_tcp_port = i64(n); } } _ => { } } }
    if has(o, "wan_ws_port") { match o["wan_ws_port"] { Json::Num(n) => { if n != 0.0 { c.wan_ws_port = i64(n); } } _ => { } } }
    if has(o, "max_wan_peers") { match o["max_wan_peers"] { Json::Num(n) => { if n != 0.0 { c.max_wan_peers = i64(n); } } _ => { } } }
    if has(o, "peer_retry_base_ms") { match o["peer_retry_base_ms"] { Json::Num(n) => { if n != 0.0 { c.peer_retry_base_ms = i64(n); } } _ => { } } }
    if has(o, "peer_retry_max_ms") { match o["peer_retry_max_ms"] { Json::Num(n) => { if n != 0.0 { c.peer_retry_max_ms = i64(n); } } _ => { } } }
    return c;
}

host det fn json_str_list(a: [Json]) -> [str] {
    var acc: [str] = [];
    for i in 0..len(a) {
        match a[i] { Json::Str(s) => { push(acc, s); } _ => { } }
    }
    return acc;
}
