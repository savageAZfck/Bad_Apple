// badapple_organs.cy — Bad Apple's organ layer, written in Coyote.
//
// Ports what used to be five separate Rust organs — the reflex arc
// (badapple-reflex / axon), the autonomy marshal (akicita), the output
// firewall scan (macuahuitl), the dissenting council (tokala), and
// certified erasure (camazotz) — into one compiled organism. Instead of
// five crates each keeping JSON sidecars, every organ runs in one VM
// whose witness chain is the audit ledger.
//
// State is Coyote-owned text under `root` (passed per call — the same
// role BADAPPLE_REFLEX_DIR plays):
//
//   arc.txt      frz flag, storm window, installed reflex table
//   arc.log      seq= / tip= header + hash-chained firing ledger
//   marshal.txt  autonomy level, rate-limit clock, dedupe ring
//
// Reflex rows: `R|id|trigger|action|cooldown_ms|max_firings|last_ms|count`
// Trigger grammar is badapple-reflex's own:
//   kind:<glob>    field:<name>=<glob>    gt|lt|ge|le:<field>=<n>
// `*` only globs as a suffix — same as axon::glob_match.
//
// Sense-event data rides a {str: str} map where each value is typed:
//   `n:<num>` (JSON number)   `s:<str>`   `j:<json>` (anything else)
// matching axon's `v.as_f64()` for thresholds vs `to_string()` for globs.

cap fs, patch, erase
invariant reflexes_brake_never_steer
invariant sealed_keys_unnameable

// ==================================================== ANE silicon programs
//
// Two compiled ANE programs live inside the organ VM:
//   - `organ_encode`: 2-layer 64->64->64 MLP on the Apple Neural Engine
//     used by the gatekeeper semantic router and organ pulse.
//   - `organ_train`: on-chip forward + reverse-mode `grad` + in-place
//     packed weight update (`gk_w1`, `gk_w2`) on the Apple Neural Engine.

param gk_w1: trainable f16[64, 64]
param gk_w2: trainable f16[64, 64]
param enclave_root: sealed f16[64]

fn gk_loss<D>(x: f16[D], t: f16[D], gk_w1: f16[D, D], gk_w2: f16[D, D]) -> f32 {
    let h = sigmoid(x @ gk_w1);
    let p = h @ gk_w2;
    mse(p, t)
}

fn organ_encode(x: f16[64]) -> f16[64] {
    sigmoid(x @ gk_w1) @ gk_w2
}

fn organ_train(x: f16[64], t: f16[64]) -> f16[64] {
    let dw2 = grad(gk_loss(x, t, gk_w1, gk_w2), gk_w2);
    let dw1 = grad(gk_loss(x, t, gk_w1, gk_w2), gk_w1);
    gk_w2 -= 0.15 * dw2;
    gk_w1 -= 0.15 * dw1;
    sigmoid(x @ gk_w1) @ gk_w2
}

// ============================================================ helpers

host det fn starts_with(s: str, prefix: str) -> bool {
    if len(s) < len(prefix) {
        return false;
    }
    return slice_str(s, 0, len(prefix)) == prefix;
}

host det fn index_of(s: str, sub: str) -> i64 {
    let n = len(sub);
    if n == 0 {
        return 0;
    }
    var i = 0;
    while i + n <= len(s) {
        if slice_str(s, i, i + n) == sub {
            return i;
        }
        i = i + 1;
    }
    return -1;
}

host det fn ends_with(s: str, suffix: str) -> bool {
    let n = len(suffix);
    if len(s) < n {
        return false;
    }
    return slice_str(s, len(s) - n, len(s)) == suffix;
}

host det fn trim(s: str) -> str {
    var lo = 0;
    var hi = len(s);
    var done = false;
    while lo < hi {
        if done {
            break;
        }
        let c = slice_str(s, lo, lo + 1);
        if c == " " || c == "\t" || c == "\r" {
            lo = lo + 1;
        } else {
            done = true;
        }
    }
    done = false;
    while hi > lo {
        if done {
            break;
        }
        let c = slice_str(s, hi - 1, hi);
        if c == " " || c == "\t" || c == "\r" {
            hi = hi - 1;
        } else {
            done = true;
        }
    }
    return slice_str(s, lo, hi);
}

// axon::glob_match — a single trailing `*` is a prefix glob.
host det fn glob_match(pattern: str, value: str) -> bool {
    if ends_with(pattern, "*") {
        return starts_with(value, slice_str(pattern, 0, len(pattern) - 1));
    }
    return pattern == value;
}

host det fn join_str(xs: [str], sep: str) -> str {
    var acc = "";
    for i in 0..len(xs) {
        if i > 0 {
            acc = acc + sep;
        }
        acc = acc + xs[i];
    }
    return acc;
}

host det fn drop_front(xs: [str]) -> [str] {
    var acc: [str] = [];
    for i in 1..len(xs) {
        push(acc, xs[i]);
    }
    return acc;
}

// typed field values: strip the `n:`/`s:`/`j:` tag.
host det fn field_val(raw: str) -> str {
    return slice_str(raw, 2, len(raw));
}

host det fn field_is_num(raw: str) -> bool {
    return starts_with(raw, "n:");
}

// map -> deterministic string (insertion order is stable in the VM).
host det fn map_repr(m: {str: str}) -> str {
    var acc = "{";
    let ks = keys(m);
    for i in 0..len(ks) {
        if i > 0 {
            acc = acc + ",";
        }
        acc = acc + ks[i] + "=" + m[ks[i]];
    }
    return acc + "}";
}

// ============================================================ triggers

// badapple-reflex parse_trigger grammar, evaluated against an event.
host det fn valid_trigger(spec: str) -> bool {
    let ci = index_of(spec, ":");
    if ci < 1 {
        return false;
    }
    let head = slice_str(spec, 0, ci);
    let rest = slice_str(spec, ci + 1, len(spec));
    if head == "kind" {
        return len(rest) > 0;
    }
    if head == "field" {
        let eq = index_of(rest, "=");
        return eq > 0 && eq + 1 < len(rest);
    }
    if head == "gt" || head == "lt" || head == "ge" || head == "le" {
        let eq = index_of(rest, "=");
        if eq < 1 || eq + 1 >= len(rest) {
            return false;
        }
        let n = parse_num(slice_str(rest, eq + 1, len(rest)));
        return n == n;
    }
    return false;
}

// axon Trigger::matches — KindIs / FieldMatch / Threshold.
host det fn trigger_hit(spec: str, kind: str, data: {str: str}) -> bool {
    let ci = index_of(spec, ":");
    if ci < 1 {
        return false;
    }
    let head = slice_str(spec, 0, ci);
    let rest = slice_str(spec, ci + 1, len(spec));
    if head == "kind" {
        return glob_match(rest, kind);
    }
    let eq = index_of(rest, "=");
    if eq < 1 {
        return false;
    }
    let fname = slice_str(rest, 0, eq);
    let arg = slice_str(rest, eq + 1, len(rest));
    if !has(data, fname) {
        return false;
    }
    let raw = data[fname];
    if head == "field" {
        return glob_match(arg, field_val(raw));
    }
    if head == "gt" || head == "lt" || head == "ge" || head == "le" {
        if !field_is_num(raw) {
            return false;
        }
        let got = parse_num(field_val(raw));
        let want = parse_num(arg);
        if got != got || want != want {
            return false;
        }
        if head == "gt" {
            return got > want;
        }
        if head == "lt" {
            return got < want;
        }
        if head == "ge" {
            return got >= want;
        }
        return got <= want;
    }
    return false;
}

// ============================================================= the arc

// axon invariant #1, enforced by the Coyote compiler: this `reflex` is
// det-only, holds no capabilities, and cannot steer — it can only ever
// return Pass/Alert/Brake/Quarantine. Every non-Pass return lands on the
// witness chain automatically.
reflex assert_action(id: str, action: str, why: str) -> Reflex {
    if action == "brake" {
        return Reflex::Brake(id + ": " + why);
    }
    if action == "quarantine" {
        return Reflex::Quarantine(id + ": " + why);
    }
    return Reflex::Alert(id + ": " + why);
}

struct ArcState {
    frz: bool,
    wstart: i64,
    wcount: i64,
    reflexes: [str],
}

host det fn arc_parse(text: str) -> ArcState {
    var frz = false;
    var wstart = 0;
    var wcount = 0;
    var refs: [str] = [];
    let lines = split(text, "\n");
    for i in 0..len(lines) {
        let line = lines[i];
        if line == "frozen=1" {
            frz = true;
        }
        if starts_with(line, "wstart=") {
            wstart = i64(parse_num(slice_str(line, 7, len(line))));
        }
        if starts_with(line, "wcount=") {
            wcount = i64(parse_num(slice_str(line, 7, len(line))));
        }
        if starts_with(line, "R|") {
            push(refs, line);
        }
    }
    return ArcState { frz: frz, wstart: wstart, wcount: wcount, reflexes: refs };
}

host fn arc_load(root: str) -> ArcState cap fs {
    return arc_parse(read_file(root + "/arc.txt"));
}

host fn arc_save(root: str, st: ArcState) -> i64 cap fs {
    var text = "v1\n";
    if st.frz {
        text = text + "frozen=1\n";
    } else {
        text = text + "frozen=0\n";
    }
    text = text + "wstart=" + str(st.wstart) + "\n";
    text = text + "wcount=" + str(st.wcount) + "\n";
    for i in 0..len(st.reflexes) {
        text = text + st.reflexes[i] + "\n";
    }
    write_file(root + "/arc.txt", text);
    return 0;
}

// append a firing to arc.log — the same fields axon's Firing carries,
// hash-chained: seq|reflex|action|evidence|kind|ts|prev|hash
host fn arc_record(root: str, rid: str, action: str, kind: str, evidence: str, ts: i64) -> i64 cap fs {
    let path = root + "/arc.log";
    let lines = split(read_file(path), "\n");
    var seq = 0;
    var prev = "0000000000000000000000000000000000000000000000000000000000000000";
    var acc = "";
    for i in 0..len(lines) {
        let line = lines[i];
        if starts_with(line, "seq=") {
            seq = i64(parse_num(slice_str(line, 4, len(line))));
            acc = acc + "seq=" + str(seq + 1) + "\n";
        } else if starts_with(line, "tip=") {
            prev = slice_str(line, 4, len(line));
            acc = acc + "NEWTIP\n";
        } else {
            acc = acc + line + "\n";
        }
    }
    let hash = sha256(str(seq) + "|" + rid + "|" + action + "|" + evidence + "|" + kind + "|" + str(ts) + "|" + prev);
    let body = acc + str(seq) + "|" + rid + "|" + action + "|" + evidence + "|" + kind + "|" + str(ts) + "|" + prev + "|" + hash + "\n";
    var fixed = "";
    let blines = split(body, "\n");
    for i in 0..len(blines) {
        if blines[i] == "NEWTIP" {
            fixed = fixed + "tip=" + hash + "\n";
        } else if len(blines[i]) > 0 {
            fixed = fixed + blines[i] + "\n";
        }
    }
    write_file(path, fixed);
    return seq + 1;
}

// `badapple-reflex install` — validate like axon::Reflex::validate and
// badapple-reflex::parse_trigger, then append the row.
host fn arc_install(root: str, id: str, trigger: str, action: str, cooldown: i64, max: i64) -> str cap fs {
    if id == "" {
        return "error=empty id";
    }
    if index_of(id, "|") >= 0 || index_of(trigger, "|") >= 0 {
        return "error=|` is the field separator and cannot appear in id or trigger";
    }
    if !(action == "brake" || action == "alert" || action == "quarantine") {
        return "error=unknown action " + action;
    }
    if !valid_trigger(trigger) {
        return "error=trigger spec must be kind:<glob> | field:<name>=<glob> | gt|lt|ge|le:<field>=<n>";
    }
    var st = arc_load(root);
    for i in 0..len(st.reflexes) {
        let p = split(st.reflexes[i], "|");
        if len(p) == 8 {
            if p[1] == id {
                return "error=duplicate id: " + id;
            }
        }
    }
    push(st.reflexes, "R|" + id + "|" + trigger + "|" + action + "|" + str(cooldown) + "|" + str(max) + "|0|0");
    arc_save(root, st);
    witness("arc:install", id + " -> " + action);
    return "installed=" + id;
}

host fn arc_remove(root: str, id: str) -> str cap fs {
    var st = arc_load(root);
    var kept: [str] = [];
    var found = false;
    for i in 0..len(st.reflexes) {
        let p = split(st.reflexes[i], "|");
        if len(p) == 8 && p[1] == id {
            found = true;
        } else {
            push(kept, st.reflexes[i]);
        }
    }
    if !found {
        return "error=no such reflex: " + id;
    }
    st.reflexes = kept;
    arc_save(root, st);
    witness("arc:remove", id);
    return "removed=" + id;
}

host fn arc_list(root: str) -> str cap fs {
    let st = arc_load(root);
    if len(st.reflexes) == 0 {
        return "reflexes=[]";
    }
    return "reflexes=[" + join_str(st.reflexes, ", ") + "]";
}

// axon ReflexArc::sense — storm window, freeze, per-reflex cooldown and
// max-firings, hash-chained record, brake engages freeze. Timestamps are
// injected by the caller so the whole decision stays `det`-replayable.
host fn arc_sense(root: str, kind: str, data: {str: str}, now_ms: i64) -> str cap fs {
    var st = arc_load(root);
    if now_ms - st.wstart > 60000 {
        st.wstart = now_ms;
        st.wcount = 0;
    }
    var fired: [str] = [];
    let _ane = organ_encode(onehot(len(kind) % 64, 64));
    let evidence = sha256(kind + "|" + map_repr(data) + "|" + str(now_ms));
    if st.frz {
        if st.wcount < 64 {
            arc_record(root, "freeze", "brake", kind, evidence, now_ms);
            st.wcount = st.wcount + 1;
            push(fired, "freeze:brake");
        }
        arc_save(root, st);
        return "fired=" + join_str(fired, ",") + "|frozen=1";
    }
    for i in 0..len(st.reflexes) {
        if st.wcount >= 64 {
            break;
        }
        let p = split(st.reflexes[i], "|");
        if len(p) != 8 {
            continue;
        }
        if !trigger_hit(p[2], kind, data) {
            continue;
        }
        let cd = i64(parse_num(p[4]));
        let mx = i64(parse_num(p[5]));
        let last = i64(parse_num(p[6]));
        let cnt = i64(parse_num(p[7]));
        if cd > 0 && now_ms - last < cd {
            continue;
        }
        if mx > 0 && cnt >= mx {
            continue;
        }
        let r = assert_action(p[1], p[3], "event:" + kind);
        var action = "alert";
        var freeze_now = false;
        match r {
            Reflex::Pass => {}
            Reflex::Alert(m) => { action = "alert"; }
            Reflex::Brake(m) => { action = "brake"; freeze_now = true; }
            Reflex::Quarantine(m) => { action = "quarantine"; }
        }
        arc_record(root, p[1], action, kind, evidence, now_ms);
        st.wcount = st.wcount + 1;
        if freeze_now {
            st.frz = true;
        }
        push(fired, p[1] + ":" + action);
        st.reflexes[i] = "R|" + p[1] + "|" + p[2] + "|" + p[3] + "|" + p[4] + "|" + p[5] + "|" + str(now_ms) + "|" + str(cnt + 1);
    }
    arc_save(root, st);
    var fz = "0";
    if st.frz {
        fz = "1";
    }
    return "fired=" + join_str(fired, ",") + "|frozen=" + fz;
}

host fn arc_clear_freeze(root: str) -> str cap fs {
    var st = arc_load(root);
    let was = st.frz;
    st.frz = false;
    arc_save(root, st);
    witness("arc:clear-freeze", str(was));
    return "cleared|was_frozen=" + str(was);
}

host fn arc_rearm(root: str, id: str) -> str cap fs {
    var st = arc_load(root);
    var found = false;
    for i in 0..len(st.reflexes) {
        let p = split(st.reflexes[i], "|");
        if len(p) == 8 && p[1] == id {
            st.reflexes[i] = "R|" + p[1] + "|" + p[2] + "|" + p[3] + "|" + p[4] + "|" + p[5] + "|" + p[6] + "|0";
            found = true;
        }
    }
    if !found {
        return "error=no such reflex: " + id;
    }
    arc_save(root, st);
    witness("arc:rearm", id);
    return "rearmed=" + id;
}

host fn arc_status(root: str) -> str cap fs {
    let st = arc_load(root);
    var firings = 0;
    var tip = "genesis";
    let lines = split(read_file(root + "/arc.log"), "\n");
    for i in 0..len(lines) {
        let line = lines[i];
        if starts_with(line, "seq=") {
            firings = i64(parse_num(slice_str(line, 4, len(line))));
        }
        if starts_with(line, "tip=") {
            tip = slice_str(line, 4, len(line));
        }
    }
    return "reflexes=" + str(len(st.reflexes)) + "|firings=" + str(firings) + "|frozen=" + str(st.frz) + "|tip=" + tip;
}

// axon ReflexArc::verify — replay the firing ledger's hash chain.
host fn arc_verify(root: str) -> str cap fs {
    let lines = split(read_file(root + "/arc.log"), "\n");
    var prev = "0000000000000000000000000000000000000000000000000000000000000000";
    var seq = 0;
    var n = 0;
    for i in 0..len(lines) {
        let line = lines[i];
        if line == "" || starts_with(line, "seq=") || starts_with(line, "tip=") {
            continue;
        }
        let p = split(line, "|");
        if len(p) != 8 {
            return "BROKEN: malformed line " + str(i);
        }
        if i64(parse_num(p[0])) != seq {
            return "BROKEN: seq gap at " + str(seq);
        }
        if p[6] != prev {
            return "BROKEN: prev mismatch at " + str(seq);
        }
        let want = sha256(p[0] + "|" + p[1] + "|" + p[2] + "|" + p[3] + "|" + p[4] + "|" + p[5] + "|" + prev);
        if want != p[7] {
            return "BROKEN: hash mismatch at " + str(seq);
        }
        prev = p[7];
        seq = seq + 1;
        n = n + 1;
    }
    return "verified=" + str(n);
}

// ========================================================== the marshal

// akicita — the camp marshal. Level ladder, dedupe ring (512), rate
// limit; the bounded decision itself is delegated to the `propose_patch`
// native (protected surfaces + blast radius, witnessed by the VM).

struct MarshalState {
    level: str,
    last_apply: i64,
    min_interval: i64,
    dedupe: [str],
}

host det fn marshal_parse(text: str) -> MarshalState {
    var level = "suggest";
    var last_apply = 0;
    var min_interval = 0;
    var ded: [str] = [];
    let lines = split(text, "\n");
    for i in 0..len(lines) {
        let line = lines[i];
        if starts_with(line, "level=") {
            level = trim(slice_str(line, 6, len(line)));
        }
        if starts_with(line, "last_apply=") {
            last_apply = i64(parse_num(slice_str(line, 11, len(line))));
        }
        if starts_with(line, "min_interval=") {
            min_interval = i64(parse_num(slice_str(line, 13, len(line))));
        }
        if starts_with(line, "dedupe=") {
            let rest = trim(slice_str(line, 7, len(line)));
            if len(rest) > 0 {
                ded = split(rest, ",");
            }
        }
    }
    return MarshalState { level: level, last_apply: last_apply, min_interval: min_interval, dedupe: ded };
}

host fn marshal_load(root: str) -> MarshalState cap fs {
    return marshal_parse(read_file(root + "/marshal.txt"));
}

host fn marshal_save(root: str, st: MarshalState) -> i64 cap fs {
    write_file(root + "/marshal.txt",
        "level=" + st.level + "\n" +
        "last_apply=" + str(st.last_apply) + "\n" +
        "min_interval=" + str(st.min_interval) + "\n" +
        "dedupe=" + join_str(st.dedupe, ",") + "\n");
    return 0;
}

// akicita::Akicita::check — off denies, dedupe denies, suggest holds,
// protected + blast caps hold (inside propose_patch), rate limit denies,
// otherwise permit. Every outcome is journaled on the witness chain.
host fn marshal_check(root: str, surface: str, old: str, new: str, intent: str, now_ms: i64) -> str cap fs, patch {
    var st = marshal_load(root);
    if st.level == "off" {
        witness("marshal:deny", surface + " (autonomy off)");
        return "deny=autonomy level is off";
    }
    let fp = sha256(surface + "|" + old + "|" + new);
    for i in 0..len(st.dedupe) {
        if st.dedupe[i] == fp {
            witness("marshal:deny", surface + " (refile)");
            return "deny=proposal already decided — not refiling";
        }
    }
    let d = propose_patch(surface, old, new, intent);
    var verdict = "permit";
    var reason = "";
    match d {
        Decision::Permit => {}
        Decision::Hold(m) => { verdict = "hold"; reason = m; }
        Decision::Deny(m) => { verdict = "deny"; reason = m; }
    }
    if verdict == "permit" {
        if st.level == "suggest" {
            verdict = "hold";
            reason = "suggest level — recorded for review";
        }
    }
    if verdict == "permit" {
        if st.min_interval > 0 && st.last_apply > 0 && now_ms - st.last_apply < st.min_interval {
            verdict = "deny";
            reason = "rate limited";
        }
    }
    if verdict == "permit" {
        st.last_apply = now_ms;
    }
    push(st.dedupe, fp);
    if len(st.dedupe) > 512 {
        st.dedupe = drop_front(st.dedupe);
    }
    marshal_save(root, st);
    return verdict + "=" + reason;
}

host fn marshal_set_level(root: str, level: str) -> str cap fs {
    if !(level == "off" || level == "suggest" || level == "safe-apply" || level == "full") {
        return "error=unknown level " + level;
    }
    var st = marshal_load(root);
    st.level = level;
    marshal_save(root, st);
    witness("marshal:level", level);
    return "level=" + level;
}

host fn marshal_status(root: str) -> str cap fs {
    let st = marshal_load(root);
    return "level=" + st.level + "|last_apply=" + str(st.last_apply) + "|min_interval=" + str(st.min_interval) + "|dedupe=" + str(len(st.dedupe));
}

// ============================================================ firewall

// firewall_machine::merged_patterns — built-ins first, then each
// non-comment blocklist line — then macuahuitl's Aho-Corasick scan.
host fn firewall_scan(blocklist: str, text: str) -> str cap fs {
    var pats: [str] = [
        "sk-", "ssh-rsa", "-----BEGIN", "-----END", "BEGIN PRIVATE KEY", "BEGIN OPENSSH PRIVATE KEY",
        "-----begin", "-----end", "begin private key", "begin openssh private key"
    ];
    if len(blocklist) > 0 {
        let lines = split(read_file(blocklist), "\n");
        for i in 0..len(lines) {
            let t = trim(lines[i]);
            if len(t) > 0 && !starts_with(t, "#") {
                push(pats, t);
            }
        }
    }
    let fw = dfa_compile(pats);
    let hit = dfa_scan(fw, text);
    let _ane = organ_encode(onehot(len(text) % 64, 64));
    witness("firewall:scan", "patterns=" + str(len(pats)) + " hit=" + str(hit));
    return "hit=" + str(hit) + "|patterns=" + str(len(pats));
}

host fn firewall_scan_live(text: str) -> str cap fs {
    return firewall_scan("/var/lib/bad_apple/blocklist.txt", text);
}

// =================================================== gatekeeper ANE router

// Classify a 64-D grounded semantic embedding on the ANE (`organ_encode`).
// Channels 0..32 represent class 0 (fast), channels 32..64 represent class 1 (deep).
// Returns a probability score in [0.0, 1.0] (> 0.5 => deep).
host fn gatekeeper_classify(x: f16[64]) -> f64 {
    let y = organ_encode(x);
    var s0 = 0.0;
    var s1 = 0.0;
    for i in 0..32 {
        s0 = s0 + y[i];
        s1 = s1 + y[i + 32];
    }
    s0 = s0 / 32.0;
    s1 = s1 / 32.0;
    var p = 0.5 + 0.5 * (s1 - s0);
    if p < 0.0 {
        p = 0.0;
    }
    if p > 1.0 {
        p = 1.0;
    }
    witness("gatekeeper:classify", str(i64(p * 1000.0)));
    return p;
}

// On-chip ANE training step (`organ_train`): computes forward + backward `grad`
// + in-place update of `gk_w1` and `gk_w2` on the Neural Engine and returns MSE.
host fn gatekeeper_train_step(x: f16[64], label: i64) -> f64 {
    var t = zeros(64);
    for i in 0..32 {
        if label == 1 {
            t[i] = -1.0;
            t[i + 32] = 1.0;
        } else {
            t[i] = 1.0;
            t[i + 32] = -1.0;
        }
    }
    let y = organ_train(x, t);
    var loss = 0.0;
    for i in 0..64 {
        let d = y[i] - t[i];
        loss = loss + d * d;
    }
    return loss / 64.0;
}

// ============================================================= council

// tokala — votes arrive as `seat|yes|reason` rows; the quorum native
// tallies, preserves minority dissent, and witnesses the verdict.
host fn council(topic: str, votes: [str]) -> str {
    var vs: [Vote] = [];
    for i in 0..len(votes) {
        let p = split(votes[i], "|");
        if len(p) == 3 {
            push(vs, Vote { seat: p[0], approve: p[1] == "yes", reason: p[2] });
        }
    }
    let qv = quorum(topic, vs);
    var acc = "approved=" + str(qv.approved) + "|yes=" + str(qv.yes) + "|no=" + str(qv.no) + "|hash=" + qv.hash;
    if len(qv.dissent) > 0 {
        acc = acc + "|dissent=" + qv.dissent[0];
    }
    return acc;
}

// ============================================================ erasure

// camazotz — certified erasure with a decrypt-failure proof, witnessed.
host fn erase_subject(subject: str, payload: str) -> str cap erase {
    return "cert=" + erase(subject, payload);
}

// =========================================================== self-test

host fn main() -> i64 {
    // state files land as /tmp/arc.txt etc — selftest only.
    let root = "/tmp";
    write_file(root + "/arc.txt", "v1\nfrozen=0\nwstart=0\nwcount=0\n");
    write_file(root + "/arc.log", "seq=0\ntip=0000000000000000000000000000000000000000000000000000000000000000\n");
    write_file(root + "/marshal.txt", "level=suggest\nlast_apply=0\nmin_interval=0\ndedupe=\n");

    print(arc_install(root, "exfil-brake", "field:dest=external*", "quarantine", 0, 0));
    var data: {str: str} = {};
    data["dest"] = "s:external-host";
    print(arc_sense(root, "tool_dispatch", data, 1000));
    print(arc_status(root));
    print(arc_verify(root));
    print(marshal_check(root, "src/app.rs", "a", "b", "small fix", 2000));
    print(firewall_scan("", "here is sk-livekey material"));
    print(council("deploy", ["A|yes|within bounds", "B|no|need margin", "C|yes|verified"]));
    print(erase_subject("subj_1", "ephemeral transcript"));
    let x0 = onehot(0, 64);
    let l0 = gatekeeper_train_step(x0, 1);
    let p0 = gatekeeper_classify(x0);
    print("gatekeeper ane train loss=" + str(l0) + " classify=" + str(p0));
    return 0;
}
