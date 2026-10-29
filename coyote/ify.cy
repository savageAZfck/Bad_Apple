// ify.cy — IFY, the behavioral watchdog, in Coyote.
//
// Port of src/ify.rs: tails the hash-chained audit ledger, verifies each
// new line against the chain, learns deterministic baselines (event-type
// rates per hour-of-day, tool frequencies, approval outcomes), and
// surfaces findings on the severity ladder:
//
//   info     → findings log only
//   elevated → proposal + notification (secondary phase and up)
//   critical → proposal + notification + kill-switch brake (autopilot)
//
// The detector is statistics, not a model: every finding carries the
// observed number, the baseline number, and the rule that fired.
//
// IfyState travels as a Json::Obj — same field names as the serde
// struct so state.json stays cross-compatible with the Rust file.

cap fs, net, crypto, proc

import "sys.cy";
import "strlib.cy";
import "ledger.cy";
import "slicks.cy";
import "blueprint.cy";

// ── constants ──────────────────────────────────────────────────────────
host det fn genesis_label() -> str { return "bad-apple-genesis-v1"; }
host det fn default_gestation_days() -> f64 { return 14.0; }
host det fn default_secondary_days() -> f64 { return 14.0; }
host det fn baseline_warmup_events() -> i64 { return 200; }
host det fn rate_spike_factor() -> f64 { return 4.0; }
host det fn rate_spike_floor() -> i64 { return 10; }
host det fn denial_spike_count() -> i64 { return 3; }
host det fn firewall_spike_floor() -> i64 { return 3; }

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

host det fn jobj(o: {str: Json}, k: str) -> {str: Json} {
    if has(o, k) {
        match o[k] { Json::Obj(m) => { return m; } _ => { } }
    }
    var empty: {str: Json} = {};
    return empty;
}

host det fn jarr(o: {str: Json}, k: str) -> [Json] {
    if has(o, k) {
        match o[k] { Json::Arr(a) => { return a; } _ => { } }
    }
    var empty: [Json] = [];
    return empty;
}

host det fn jarr_push(a: Json, v: Json) -> unit {
    match a { Json::Arr(items) => { push(items, v); } _ => { } }
}

// map.get(k).count += 1 inside an Obj-of-Objs
host det fn jcount_bump(m: {str: Json}, k: str) -> unit {
    var cur = 0.0;
    if has(m, k) {
        match m[k] { Json::Num(n) => { cur = n; } _ => { } }
    }
    m[k] = Json::Num(cur + 1.0);
}

host det fn jmap_num(m: {str: Json}, k: str) -> f64 {
    if has(m, k) {
        match m[k] { Json::Num(n) => { return n; } _ => { } }
    }
    return 0.0;
}

// ── time ───────────────────────────────────────────────────────────────
host fn now_secs() -> f64 { return f64(time_ms()) / 1000.0; }

host det fn genesis_hash() -> str { return sha256(genesis_label()); }

// hour-of-day (UTC) from an RFC3339/ISO timestamp string
host det fn hour_of(ts: str) -> i64 {
    let ms = time_parse_iso(ts);
    if ms < 0 { return 0; }
    return (ms / 3600000) % 24;
}

// "ify-YYYYMMDD-HHMMSS-XXXXXXXX" — same shape as Rust finding_id
host fn finding_id(rule: str) -> str {
    let stamp = time_stamp();          // "YYYY-MM-DD HH:MM:SS UTC"
    var compact = "";
    for i in 0..len(stamp) {
        let c = slice_str(stamp, i, i + 1);
        if contains("0123456789", c) { compact = compact + c; }
    }
    // "YYYYMMDDHHMMSS" → "YYYYMMDD-HHMMSS"
    let short = slice_str(compact, 0, 8) + "-" + slice_str(compact, 8, 14);
    let h = sha256(rule + stamp + rand_hex(4));
    return "ify-" + short + "-" + slice_str(h, 0, 8);
}

// ── paths ──────────────────────────────────────────────────────────────
host fn ify_dir() -> str cap fs {
    let e = env_get("BADAPPLE_IFY_DIR");
    if e != "" { return e; }
    let home = env_get("HOME");
    if home == "" { return "/tmp/.bad_apple/ify"; }
    return home + "/.bad_apple/ify";
}

host fn state_path() -> str cap fs { return ify_dir() + "/state.json"; }
host fn findings_path() -> str cap fs { return ify_dir() + "/findings.jsonl"; }
host fn proposals_dir() -> str cap fs { return ify_dir() + "/proposals"; }
host fn runtime_state_path() -> str { return "/var/lib/bad_apple/runtime_state.json"; }

// ── phases ─────────────────────────────────────────────────────────────
host fn gestation_days() -> f64 {
    let v = env_get("BADAPPLE_IFY_GESTATION_DAYS");
    if v == "" { return default_gestation_days(); }
    return parse_num(v);
}

host fn secondary_days() -> f64 {
    let v = env_get("BADAPPLE_IFY_SECONDARY_DAYS");
    if v == "" { return default_secondary_days(); }
    return parse_num(v);
}

// → "gestation" | "secondary" | "autopilot"; BADAPPLE_IFY_PHASE forces.
host fn current_phase(st: Json) -> str {
    let forced = str_lower(trim(env_get("BADAPPLE_IFY_PHASE")));
    if forced == "gestation" { return "gestation"; }
    if forced == "secondary" || forced == "secondary_school" { return "secondary"; }
    if forced == "autopilot" { return "autopilot"; }
    var installed = 0.0;
    match st {
        Json::Obj(s) => { installed = jnumf(s, "installed_at"); }
        _ => { }
    }
    let age_days = (now_secs() - installed) / 86400.0;
    if age_days < gestation_days() { return "gestation"; }
    if age_days < gestation_days() + secondary_days() { return "secondary"; }
    return "autopilot";
}

host det fn severity_num(sev: str) -> i64 {
    if sev == "elevated" { return 1; }
    if sev == "critical" { return 2; }
    return 0;
}

host det fn surfaces(phase: str, severity: str) -> bool {
    if phase == "gestation" { return false; }
    return severity_num(severity) >= 1;
}

host det fn may_brake(phase: str) -> bool { return phase == "autopilot"; }

// ── state ──────────────────────────────────────────────────────────────
host fn zeros24() -> Json {
    var a: [Json] = [];
    for i in 0..24 { push(a, Json::Num(0.0)); }
    return Json::Arr(a);
}

host fn state_new() -> Json {
    var s: {str: Json} = {};
    s["version"] = Json::Num(1.0);
    s["installed_at"] = Json::Num(now_secs());
    s["events_seen"] = Json::Num(0.0);
    s["ledger_offset"] = Json::Num(0.0);
    s["ledger_tip"] = Json::Str(genesis_hash());
    var et0: {str: Json} = {};
    s["event_types"] = Json::Obj(et0);
    var t0: {str: Json} = {};
    s["tools"] = Json::Obj(t0);
    var p0: {str: Json} = {};
    s["personas"] = Json::Obj(p0);
    s["approvals_granted"] = Json::Num(0.0);
    s["approvals_denied"] = Json::Num(0.0);
    s["firewall_hits"] = Json::Num(0.0);
    s["kill_switch_events"] = Json::Num(0.0);
    s["hourly_total"] = zeros24();
    s["window_hour"] = Json::Num(-1.0);
    var w0: {str: Json} = {};
    s["window_counts"] = Json::Obj(w0);
    var dt0: [Json] = [];
    s["denial_times"] = Json::Arr(dt0);
    var ft0: [Json] = [];
    s["firewall_times"] = Json::Arr(ft0);
    var f0: {str: Json} = {};
    s["fired"] = Json::Obj(f0);
    s["last_brake_ts"] = Json::Num(0.0);
    return Json::Obj(s);
}

host fn load_state() -> Json cap fs {
    let p = state_path();
    if fs_exists(p) {
        let j = json_parse(read_file(p));
        match j {
            Json::Obj(s) => {
                if has(s, "version") { return j; }
            }
            _ => { }
        }
    }
    return state_new();
}

host fn save_state(st: Json) -> bool cap fs {
    let dir = ify_dir();
    fs_mkdir(dir);
    let tmp = dir + "/.state.tmp";
    write_file(tmp, json_emit(st));
    return fs_rename(tmp, state_path());
}

host fn log_finding(f: Json) -> bool cap fs {
    fs_mkdir(ify_dir());
    return fs_append(findings_path(), json_emit(f) + "\n");
}

// ── ledger tailing ─────────────────────────────────────────────────────
host fn load_slicks_secrets() -> [str] cap fs {
    var p = env_get("BADAPPLE_SLICKS_KEY_PATH");
    if p == "" { p = "/var/lib/bad_apple/slicks.key"; }
    return ledger_key_candidates(p);
}

host fn make_finding(rule: str, sev: str, observed: str, baseline: str, detail: str) -> Json {
    var f: {str: Json} = {};
    f["id"] = Json::Str(finding_id(rule));
    f["ts"] = Json::Num(now_secs());
    f["rule"] = Json::Str(rule);
    f["severity"] = Json::Str(sev);
    f["observed"] = Json::Str(observed);
    f["baseline"] = Json::Str(baseline);
    f["detail"] = Json::Str(detail);
    return Json::Obj(f);
}

// → Json {type, data, hour, hash} or Null on malformed
host det fn parse_event(line: str) -> Json {
    let v = json_parse(line);
    match v {
        Json::Obj(m) => {
            var ty = jstr(m, "type");
            if ty == "" { ty = jstr(m, "event_type"); }
            if ty == "" { ty = "audit"; }
            var ts = jstr(m, "ts");
            if ts == "" { ts = jstr(m, "timestamp"); }
            var data = Json::Null;
            if has(m, "data") { data = m["data"]; }
            var e: {str: Json} = {};
            e["type"] = Json::Str(ty);
            e["data"] = data;
            e["hour"] = Json::Num(f64(hour_of(ts)));
            e["hash"] = Json::Str(jstr(m, "hash"));
            return Json::Obj(e);
        }
        _ => { return Json::Null; }
    }
}

// tail_ledger(state, secrets) — read new bytes from ledger_offset, verify
// each line's linkage + content hash, update baselines, run the rules.
// → Json {state, new_events, findings: [..], chain_broken, truncated}
host fn tail_ledger(stin: Json, secrets: [str]) -> Json cap fs, crypto {
    var findings: [Json] = [];
    var new_events = 0;
    var chain_broken = false;
    var truncated = false;
    var st = stin;
    let path = ledger_path();

    match st {
        Json::Obj(s) => {
            if !fs_exists(path) {
                return tail_report(st, new_events, findings, chain_broken, truncated);
            }
            let size = fs_size(path);
            var offset = jnumi(s, "ledger_offset");
            if size < offset {
                truncated = true;
                push(findings, make_finding(
                    "ledger_gap", "critical",
                    "size " + str(size) + " < offset " + str(offset),
                    "append-only",
                    "ledger.jsonl shrank — entries were removed or the file was replaced"));
                s["ledger_offset"] = Json::Num(0.0);
                s["ledger_tip"] = Json::Str(genesis_hash());
                offset = 0;
            }
            let buf = fs_read_from(path, offset);
            var tip = jstr(s, "ledger_tip");
            let lines = split(buf, "\n");
            var li = 0;
            var stop = false;
            while li < len(lines) {
                if stop { break; }
                let raw = lines[li];
                let line = trim(raw);
                // complete-line accounting: +1 for '\n' matches the Rust
                // byte math on the ASCII JSONL the ledger writes
                if line == "" {
                    // split yields a phantom "" after a trailing '\n' that
                    // Rust's .lines() never produces — don't count it
                    if li < len(lines) - 1 {
                        offset += len(raw) + 1;
                    }
                } else {
                    let vr = ledger_verify_line(raw, secrets, tip);
                    match vr {
                        Json::Obj(v) => {
                            if jbool(v, "ok") {
                                let ev = parse_event(raw);
                                match ev {
                                    Json::Obj(_) => {
                                        observe_event(s, ev, findings);
                                    }
                                    _ => { }
                                }
                                tip = jstr(v, "tip");
                                s["ledger_tip"] = Json::Str(tip);
                                s["events_seen"] = Json::Num(jnumf(s, "events_seen") + 1.0);
                                new_events += 1;
                                offset += len(raw) + 1;
                            } else {
                                chain_broken = true;
                                push(findings, make_finding(
                                    "chain_break", "critical",
                                    "verify failed: " + jstr(v, "err"),
                                    "prev " + slice_str(tip, 0, 16),
                                    "a ledger line failed hash-chain verification — the audit trail was modified"));
                                stop = true;
                            }
                        }
                        _ => { stop = true; }
                    }
                }
                li += 1;
            }
            s["ledger_offset"] = Json::Num(f64(offset));
        }
        _ => { }
    }
    return tail_report(st, new_events, findings, chain_broken, truncated);
}

host det fn tail_report(st: Json, new_events: i64, findings: [Json], chain_broken: bool, truncated: bool) -> Json {
    var r: {str: Json} = {};
    r["state"] = st;
    r["new_events"] = Json::Num(f64(new_events));
    r["findings"] = Json::Arr(findings);
    r["chain_broken"] = Json::Bool(chain_broken);
    r["truncated"] = Json::Bool(truncated);
    return Json::Obj(r);
}

// ── baselines + detection ──────────────────────────────────────────────
// observe_event(s, ev, findings) — s is the state Obj (mutated in place).
host fn observe_event(s: {str: Json}, ev: Json, findings: [Json]) -> unit {
    let warmed = jnumi(s, "events_seen") >= baseline_warmup_events();
    match ev {
        Json::Obj(e) => {
            var hour = jnumi(e, "hour");
            if hour > 23 { hour = 23; }
            if hour < 0 { hour = 0; }
            let ty = jstr(e, "type");

            // rule: novel_event_type — first sighting after warmup
            var ets = jobj(s, "event_types");
            if warmed && !has(ets, ty) {
                push(findings, make_finding(
                    "novel_event_type", "elevated", ty,
                    str(len(keys(ets))) + " known types",
                    "event type `" + ty + "` has never appeared in the ledger"));
            }

            // rate window rolls over with the hour
            let now_hour = i64(now_secs() / 3600.0);
            if jnumi(s, "window_hour") != now_hour {
                s["window_hour"] = Json::Num(f64(now_hour));
                var w0: {str: Json} = {};
    s["window_counts"] = Json::Obj(w0);
            }
            var wc = jobj(s, "window_counts");
            jcount_bump(wc, ty);
            s["window_counts"] = Json::Obj(wc);
            let this_hour = jmap_num(wc, ty);

            // per-type stat (also bumps count + hourly bucket)
            bump_event_type(s, ty, hour);
            let stat = jobj(jobj(s, "event_types"), ty);
            let count = jnumf(stat, "count");
            var nonzero = 0.0;
            let buckets = jarr(stat, "hourly");
            for i in 0..len(buckets) {
                match buckets[i] {
                    Json::Num(n) => { if n > 0.0 { nonzero += 1.0; } }
                    _ => { }
                }
            }
            var denom = nonzero;
            if denom < 1.0 { denom = 1.0; }
            let hourly_avg = count / denom;

            // rule: rate_spike
            let fire_key = "rate_spike:" + ty;
            var fired_m = jobj(s, "fired");
            let already = has(fired_m, fire_key) && jnumi(fired_m, fire_key) == now_hour;
            var floor = f64(rate_spike_floor());
            var thresh = hourly_avg * rate_spike_factor();
            if thresh < floor { thresh = floor; }
            if warmed && !already && hourly_avg > 0.0 && this_hour > thresh {
                fired_m[fire_key] = Json::Num(f64(now_hour));
                s["fired"] = Json::Obj(fired_m);
                push(findings, make_finding(
                    "rate_spike", "elevated",
                    str(i64(this_hour)) + " `" + ty + "` events this hour",
                    "~" + str(hourly_avg) + "/hr typical",
                    "`" + ty + "` is running " + str(this_hour / hourly_avg) + "x above its learned rate"));
            }

            // rule: off_hours — activity in an hour-of-day with no history
            var ht = jarr(s, "hourly_total");
            var prior = 0.0;
            match ht[hour] { Json::Num(n) => { prior = n; } _ => { } }
            if warmed && prior == 0.0 {
                push(findings, make_finding(
                    "off_hours", "info",
                    "`" + ty + "` at hour " + str(hour) + " UTC",
                    "no prior activity in this hour",
                    "activity during an hour-of-day that has been silent for the whole baseline (" + ty + ")"));
            }

            // targeted counters on well-known shapes
            let data = jobj(e, "data");
            if contains(ty, "kill") || contains(ty, "safe_mode") {
                let n = jnumi(s, "kill_switch_events") + 1;
                s["kill_switch_events"] = Json::Num(f64(n));
                if warmed {
                    push(findings, make_finding(
                        "kill_switch", "elevated", ty,
                        str(n - 1) + " prior",
                        "kill switch / safe mode event — if you didn't trigger it, something else did"));
                }
            }
            if contains(ty, "firewall") {
                s["firewall_hits"] = Json::Num(jnumf(s, "firewall_hits") + 1.0);
                let ft = s["firewall_times"];
                jarr_push(ft, Json::Num(now_secs()));
                s["firewall_times"] = ft;
            }
            if contains(ty, "approv") || contains(ty, "tool") {
                var approved = true;
                if has(data, "approved") {
                    match data["approved"] { Json::Bool(b) => { approved = b; } _ => { } }
                }
                if approved {
                    s["approvals_granted"] = Json::Num(jnumf(s, "approvals_granted") + 1.0);
                } else {
                    s["approvals_denied"] = Json::Num(jnumf(s, "approvals_denied") + 1.0);
                    let dt = s["denial_times"];
                    jarr_push(dt, Json::Num(now_secs()));
                    s["denial_times"] = dt;
                }
                var tool = jstr(data, "tool");
                if tool == "" { tool = jstr(data, "name"); }
                if tool != "" {
                    var tools = jobj(s, "tools");
                    jcount_bump(tools, tool);
                    s["tools"] = Json::Obj(tools);
                }
            }
            let persona = jstr(data, "persona");
            if persona != "" {
                var pers = jobj(s, "personas");
                jcount_bump(pers, persona);
                s["personas"] = Json::Obj(pers);
            }

            // rule: denial_spike — >=3 denied approvals in trailing hour
            let cutoff = now_secs() - 3600.0;
            s["denial_times"] = prune_old(s["denial_times"], cutoff);
            s["firewall_times"] = prune_old(s["firewall_times"], cutoff);
            let denials = jarr(s, "denial_times");
            if warmed && len(denials) == i64(denial_spike_count()) {
                push(findings, make_finding(
                    "denial_spike", "elevated",
                    str(len(denials)) + " denied approvals in 1h",
                    "<" + str(denial_spike_count()) + "/hr typical",
                    "repeated tool denials — something is probing the approval gate"));
                var dt0: [Json] = [];
    s["denial_times"] = Json::Arr(dt0);
            }
            // rule: firewall_spike — output firewall hit burst
            let ftimes = jarr(s, "firewall_times");
            let hits = jnumf(s, "firewall_hits");
            var seen = jnumf(s, "events_seen");
            if seen < 1.0 { seen = 1.0; }
            var fw_thresh = hits / seen * 3600.0 * rate_spike_factor();
            let fw_floor = f64(firewall_spike_floor());
            if fw_thresh < fw_floor { fw_thresh = fw_floor; }
            if warmed && f64(len(ftimes)) > fw_thresh {
                push(findings, make_finding(
                    "firewall_spike", "elevated",
                    str(len(ftimes)) + " firewall hits in 1h",
                    str(hits) + " total over baseline",
                    "output firewall is firing in a burst — possible injection probing"));
                var ft0: [Json] = [];
    s["firewall_times"] = Json::Arr(ft0);
            }
        }
        _ => { }
    }
}

// bump event_types[ty].count + .hourly[hour] + hourly_total[hour]
host fn bump_event_type(s: {str: Json}, ty: str, hour: i64) -> unit {
    var ets = jobj(s, "event_types");
    var stat = jobj(ets, ty);
    stat["count"] = Json::Num(jnumf(stat, "count") + 1.0);
    var buckets = jarr(stat, "hourly");
    if len(buckets) < 24 {
        match zeros24() {
            Json::Arr(z) => { buckets = z; }
            _ => { }
        }
    }
    var cur = 0.0;
    match buckets[hour] { Json::Num(n) => { cur = n; } _ => { } }
    buckets[hour] = Json::Num(cur + 1.0);
    stat["hourly"] = Json::Arr(buckets);
    ets[ty] = Json::Obj(stat);
    s["event_types"] = Json::Obj(ets);
    var ht = jarr(s, "hourly_total");
    if len(ht) < 24 {
        match zeros24() {
            Json::Arr(z) => { ht = z; }
            _ => { }
        }
    }
    var hc = 0.0;
    match ht[hour] { Json::Num(n) => { hc = n; } _ => { } }
    ht[hour] = Json::Num(hc + 1.0);
    s["hourly_total"] = Json::Arr(ht);
}

// keep only timestamps > cutoff
host det fn prune_old(times: Json, cutoff: f64) -> Json {
    var kept: [Json] = [];
    match times {
        Json::Arr(a) => {
            for i in 0..len(a) {
                match a[i] {
                    Json::Num(n) => { if n > cutoff { push(kept, a[i]); } }
                    _ => { }
                }
            }
        }
        _ => { }
    }
    return Json::Arr(kept);
}

// ── actions ────────────────────────────────────────────────────────────
host det fn esc_notify(s: str) -> str {
    var r = "";
    for i in 0..len(s) {
        let c = slice_str(s, i, i + 1);
        if c == "\\" { r = r + "\\\\"; }
        else { if c == "\"" { r = r + "\\\""; } else { r = r + c; } }
    }
    return r;
}

host fn notify(title: str, body: str) -> unit cap proc {
    var args: [str] = [];
    push(args, "-e");
    push(args, "display notification \"" + esc_notify(body) + "\" with title \"" + esc_notify(title) + "\"");
    let _r = sys_exec("/usr/bin/osascript", args);
}

// write_proposal(f, braked) → proposal path or ""
host fn write_proposal(f: {str: Json}, braked: bool) -> str cap fs {
    let dir = proposals_dir();
    fs_mkdir(dir);
    var brake_txt = "";
    if braked {
        brake_txt = "\n**Brake engaged:** kill switch pulled automatically (autopilot phase). Reply `resume bad apple` to restore.";
    }
    let rule = jstr(f, "rule");
    let body = "# IFY finding: " + rule + "\n\n**When:** " + time_stamp()
        + "\n**Workspace:** system\n\n## Finding\n\n- Rule: `" + rule
        + "`\n- Severity: " + jstr(f, "severity")
        + "\n- Observed: " + jstr(f, "observed")
        + "\n- Baseline: " + jstr(f, "baseline") + "\n\n"
        + jstr(f, "detail") + "\n" + brake_txt
        + "\n\n## Proposal\n\n```json\n{\"no_patch\": true}\n```\n";
    let path = dir + "/" + jstr(f, "id") + ".md";
    write_file(path, body);
    return path;
}

// brake(reason) — kill switch via the daemon, freeze autopilot, mark
// runtime_state SAFE_MODE. Best-effort, every step logged in the result.
host fn brake(reason: str) -> Json cap fs, net, proc {
    var o: {str: Json} = {};
    o["kill_switch"] = Json::Bool(false);
    o["autopilot_off"] = Json::Bool(false);
    o["safe_mode"] = Json::Bool(false);

    // 1. ask the engine to kill itself (same path as the user's command)
    var eargs: {str: Json} = {};
    let r = slicks_call_agent(default_socket(), "invoke_tool",
        Json::Obj({"name": Json::Str("kill_switch"), "args": Json::Obj(eargs)}), 128);
    match r {
        Json::Obj(m) => {
            if jstr(m, "error") == "" && jbool(m, "ok") {
                o["kill_switch"] = Json::Bool(true);
            } else {
                o["kill_switch_error"] = Json::Str(jstr(m, "error"));
            }
        }
        _ => { o["kill_switch_error"] = Json::Str("no response"); }
    }

    // 2. freeze autopilot so a restart doesn't resume autonomy
    let home = env_get("HOME");
    if home != "" {
        write_file(home + "/.bad_apple/autopilot_level", "off");
        o["autopilot_off"] = Json::Bool(true);
    }

    // 3. runtime_state SAFE_MODE — atomic tmp+rename
    let rp = runtime_state_path();
    var rtstate: {str: Json} = {};
    if fs_exists(rp) {
        match json_parse(read_file(rp)) {
            Json::Obj(prev) => { rtstate = prev; }
            _ => { }
        }
    }
    rtstate["mode"] = Json::Str("SAFE_MODE");
    rtstate["safe_mode_reason"] = Json::Str("ify:" + reason);
    rtstate["updated_at"] = Json::Num(now_secs());
    let tmp = dir_of(rp) + "/.runtime_state.tmp";
    write_file(tmp, json_emit(Json::Obj(rtstate)));
    if fs_rename(tmp, rp) { o["safe_mode"] = Json::Bool(true); }
    return Json::Obj(o);
}

// narrate(f) — plain-English one-liner via the fast tier; "" when
// narration is disabled or the daemon is unreachable.
host fn narrate(f: {str: Json}) -> str cap fs, net, proc {
    if env_get("BADAPPLE_IFY_NARRATE") == "0" { return ""; }
    let prompt = "You are IFY, Bad Apple's watchdog. One sentence, plain English, no jargon. Report this security finding to the owner. Rule: "
        + jstr(f, "rule") + ". Observed: " + jstr(f, "observed")
        + ". Baseline: " + jstr(f, "baseline") + ". Detail: " + jstr(f, "detail");
    let r = slicks_query(default_socket(), prompt, 120);
    match r {
        Json::Obj(m) => {
            if jbool(m, "ok") { return jstr(m, "text"); }
            return "";
        }
        _ => { return ""; }
    }
}

// enrich a finding's detail with the causal-chain explanation when the
// observed text names a known runtime primitive
host det fn with_causal_explanation(f: Json) -> Json {
    match f {
        Json::Obj(m) => {
            let text = jstr(m, "rule") + " " + jstr(m, "observed") + " " + jstr(m, "detail");
            let chain = cg_explain_text(text);
            if chain == "" { return f; }
            m["detail"] = Json::Str(jstr(m, "detail") + "\n\n" + chain);
            return Json::Obj(m);
        }
        _ => { return f; }
    }
}

// dispatch(state, f) — always log; surface/brake per phase.
// → proposal path or ""
host fn dispatch(st: Json, f: Json) -> str cap fs, net, proc {
    let fe = with_causal_explanation(f);
    log_finding(fe);
    let phase = current_phase(st);
    var sev = "";
    match fe {
        Json::Obj(m) => { sev = jstr(m, "severity"); }
        _ => { }
    }
    if !surfaces(phase, sev) { return ""; }
    var braked = false;
    if sev == "critical" && may_brake(phase) {
        var rule = "";
        match fe { Json::Obj(m) => { rule = jstr(m, "rule"); } _ => { } }
        let _outcome = brake(rule);
        braked = true;
    }
    match fe {
        Json::Obj(m) => {
            var body = narrate(m);
            if body == "" { body = jstr(m, "detail"); }
            notify("IFY: " + jstr(m, "rule"), body);
            return write_proposal(m, braked);
        }
        _ => { return ""; }
    }
}

// ── entry points ───────────────────────────────────────────────────────
// ify_once() — one full pass: tail the ledger, verify, detect, dispatch.
// → Json {new_events, findings_dispatched, chain_broken, truncated, phase}
host fn ify_once() -> Json cap fs, net, crypto, proc {
    var st = load_state();
    let secrets = load_slicks_secrets();
    let rep = tail_ledger(st, secrets);
    var dispatched = 0;
    var cb = false;
    var tr = false;
    var ne = 0;
    match rep {
        Json::Obj(r) => {
            st = r["state"];
            ne = jnumi(r, "new_events");
            cb = jbool(r, "chain_broken");
            tr = jbool(r, "truncated");
            let fs2 = jarr(r, "findings");
            for i in 0..len(fs2) {
                dispatch(st, fs2[i]);
                dispatched += 1;
            }
        }
        _ => { }
    }
    save_state(st);
    var o: {str: Json} = {};
    o["new_events"] = Json::Num(f64(ne));
    o["findings_dispatched"] = Json::Num(f64(dispatched));
    o["chain_broken"] = Json::Bool(cb);
    o["truncated"] = Json::Bool(tr);
    o["phase"] = Json::Str(current_phase(st));
    match st {
        Json::Obj(s) => { o["events_seen"] = Json::Num(jnumf(s, "events_seen")); }
        _ => { }
    }
    witness("ify", "pass");
    return Json::Obj(o);
}

// ify_status() — phase + baseline summary for `badapple ify status`.
host fn ify_status() -> Json cap fs {
    let st = load_state();
    var o: {str: Json} = {};
    o["phase"] = Json::Str(current_phase(st));
    match st {
        Json::Obj(s) => {
            o["events_seen"] = Json::Num(jnumf(s, "events_seen"));
            o["installed_at"] = Json::Num(jnumf(s, "installed_at"));
            o["ledger_offset"] = Json::Num(jnumf(s, "ledger_offset"));
            o["tools"] = Json::Num(f64(len(keys(jobj(s, "tools")))));
            o["event_types"] = Json::Num(f64(len(keys(jobj(s, "event_types")))));
            o["approvals_granted"] = Json::Num(jnumf(s, "approvals_granted"));
            o["approvals_denied"] = Json::Num(jnumf(s, "approvals_denied"));
            o["firewall_hits"] = Json::Num(jnumf(s, "firewall_hits"));
            o["kill_switch_events"] = Json::Num(jnumf(s, "kill_switch_events"));
        }
        _ => { }
    }
    return Json::Obj(o);
}
