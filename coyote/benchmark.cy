// benchmark.cy — the transfer-evaluation harness, ported from
// src/benchmark.rs.
//
// The suites are stateless transforms: a suite is a Json
// {kind, index, history: [[name, ok], ...]} and every operation returns
// the updated record — callers hold the state (the Rust held it in-memory).
// Task tables, validate math, score/mastery, and the PILOT report format
// are exact ports. RustValidator stays shell: running `cargo check` in an
// isolated workspace is a process-sandbox concern, not a policy one.

cap fs

import "sys.cy";
import "strlib.cy";

// ── the standard suite — byte-identical task table ──────────────────────

host det fn bench_tasks() -> [Json] {
    var t: [Json] = [];
    push(t, Json::Obj({
        "name": Json::Str("simple_arithmetic"),
        "prompt": Json::Str("Write a short Python 3 script that computes (17 * 23) + (12 * 31) and prints only the final integer answer."),
        "expected": Json::Str("763"),
        "tolerance": Json::Num(0.001)
    }));
    push(t, Json::Obj({
        "name": Json::Str("factorial"),
        "prompt": Json::Str("Write a short Python 3 script that computes 6! (6 factorial) and prints only the final integer answer."),
        "expected": Json::Str("720"),
        "tolerance": Json::Num(0.001)
    }));
    push(t, Json::Obj({
        "name": Json::Str("fibonacci_10"),
        "prompt": Json::Str("Write a short Python 3 script that computes the 10th Fibonacci number (F(0)=0, F(1)=1, F(2)=1, ... F(10)=?) and prints only the final integer answer."),
        "expected": Json::Str("55"),
        "tolerance": Json::Num(0.001)
    }));
    push(t, Json::Obj({
        "name": Json::Str("sum_of_squares"),
        "prompt": Json::Str("Write a short Python 3 script that computes 1*1 + 2*2 + 3*3 + 4*4 + 5*5 and prints only the final integer answer."),
        "expected": Json::Str("55"),
        "tolerance": Json::Num(0.001)
    }));
    push(t, Json::Obj({
        "name": Json::Str("prime_check"),
        "prompt": Json::Str("Write a short Python 3 script that checks if 29 is prime and prints only the boolean True or False."),
        "expected": Json::Str("True"),
        "tolerance": Json::Null
    }));
    return t;
}

// ── the transfer suite — byte-identical task table ──────────────────────

host det fn xfer_tasks() -> [Json] {
    var t: [Json] = [];
    push(t, xfer_task("vowel counting",
        "Count the number of vowels in a lowercase English string.",
        "hello", "2", "badapple agi", "5"));
    push(t, xfer_task("caesar cipher shift 3",
        "Shift each letter forward by 3 in the alphabet (wrap around from z to a). Keep non-letters unchanged.",
        "abc", "def", "xyz", "abc"));
    push(t, xfer_task("sum of digits",
        "Sum all decimal digits in the input string, ignoring non-digit characters.",
        "a1b2c3", "6", "bad5apple8agi2", "15"));
    push(t, xfer_task("count words",
        "Count the number of whitespace-separated words in the input string.",
        "hello world", "2", "badapple agi research runtime", "4"));
    push(t, xfer_task("title case",
        "Convert the input string to title case: first letter of each word uppercase, the rest lowercase.",
        "hello world", "Hello World", "badapple agi research", "Badapple Agi Research"));
    push(t, xfer_task("extract digits",
        "Extract all decimal digits from the input and return them as a single string in order.",
        "a1b2c3", "123", "bad5apple8agi2", "582"));
    push(t, xfer_task("reverse words",
        "Reverse the order of whitespace-separated words in the input string.",
        "hello world", "world hello", "badapple agi research", "research agi badapple"));
    push(t, xfer_task("isogram check",
        "Return True if all letters in the input appear at most once, otherwise False. Ignore non-letters and case.",
        "background", "False", "isogram", "True"));
    push(t, xfer_task("morse alphabet",
        "Convert each letter in the input to a dot for consonants and a dash for vowels. Keep spaces.",
        "abc de", "-.. .-", "badapple agi", ".-.-...- -.-"));
    push(t, xfer_task("first letter acronym",
        "Return the first letter of each word, lowercased, concatenated.",
        "hello world", "hw", "badapple agi research runtime", "barr"));
    push(t, xfer_task("binary dotdash",
        "Replace every character that is not a space with a dot if it is a vowel and a dash otherwise.",
        "ab", ".-", "apple", ".---."));
    push(t, xfer_task("palindrome words",
        "Return a space-separated list of the words that are palindromes. A single letter is not a palindrome.",
        "madam bob apple", "madam bob", "civic radar car", "civic radar"));
    return t;
}

host det fn xfer_task(
    domain: str, desc: str, tin: str, tout: str, xin: str, xout: str
) -> Json {
    return Json::Obj({
        "domain": Json::Str(domain),
        "description": Json::Str(desc),
        "train_input": Json::Str(tin),
        "train_output": Json::Str(tout),
        "test_input": Json::Str(xin),
        "test_output": Json::Str(xout)
    });
}

// ── suite state — a Json the caller holds ───────────────────────────────

host det fn bench_new_suite(kind: str) -> Json {
    var tasks: [Json] = [];
    if kind == "transfer" { tasks = xfer_tasks(); } else { tasks = bench_tasks(); }
    let hist: [Json] = [];
    return Json::Obj({
        "kind": Json::Str(kind),
        "index": Json::Num(0.0),
        "history": Json::Arr(hist)
    });
}

host det fn jnum(j: Json) -> f64 {
    match j { Json::Num(n) => { return n; } _ => { } }
    return 0.0;
}

host det fn jstr_of(j: Json) -> str {
    match j { Json::Str(x) => { return x; } _ => { } }
    return "";
}

// next task — returns {"task": ..., "suite": <updated>}.
host det fn bench_next(suite: Json) -> Json {
    var tasks = bench_tasks();
    var idx = 0.0;
    var hist: [Json] = [];
    var kind = "basic";
    match suite {
        Json::Obj(s) => {
            if has(s, "kind") { kind = jstr_of(s["kind"]); }
            if has(s, "index") { idx = jnum(s["index"]); }
            if has(s, "history") {
                match s["history"] { Json::Arr(h) => { hist = h; } _ => { } }
            }
        }
        _ => { }
    }
    if kind == "transfer" { tasks = xfer_tasks(); }
    let n = len(tasks);
    let i = i64(idx) % n;
    let next = Json::Num(f64(i64(idx) + 1));
    let upd = Json::Obj({
        "kind": Json::Str(kind),
        "index": next,
        "history": Json::Arr(hist)
    });
    return Json::Obj({ "task": tasks[i], "suite": upd });
}

// record a result into history (cap 1000, drop oldest like the Rust).
host det fn bench_record(suite: Json, name: str, ok: bool) -> Json {
    var hist: [Json] = [];
    var kind = "basic";
    var idx = 0.0;
    match suite {
        Json::Obj(s) => {
            if has(s, "kind") { kind = jstr_of(s["kind"]); }
            if has(s, "index") { idx = jnum(s["index"]); }
            if has(s, "history") {
                match s["history"] { Json::Arr(h) => { hist = h; } _ => { } }
            }
        }
        _ => { }
    }
    push(hist, Json::Arr(bench_pair(name, ok)));
    if len(hist) > 1000 {
        var kept: [Json] = [];
        for i in 1..len(hist) { push(kept, hist[i]); }
        hist = kept;
    }
    return Json::Obj({
        "kind": Json::Str(kind),
        "index": Json::Num(idx),
        "history": Json::Arr(hist)
    });
}

host det fn bench_pair(name: str, ok: bool) -> [Json] {
    var p: [Json] = [];
    push(p, Json::Str(name));
    push(p, Json::Bool(ok));
    return p;
}

// success ratio * 100 — matches Rust score().
host det fn bench_score(suite: Json) -> f64 {
    var total = 0;
    var good = 0;
    match suite {
        Json::Obj(s) => {
            if has(s, "history") {
                match s["history"] {
                    Json::Arr(h) => {
                        for i in 0..len(h) {
                            match h[i] {
                                Json::Arr(pair) => {
                                    total = total + 1;
                                    if len(pair) > 1 {
                                        match pair[1] {
                                            Json::Bool(b) => { if b { good = good + 1; } }
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
        }
        _ => { }
    }
    if total == 0 { return 0.0; }
    return (f64(total) / f64(total)) * 100.0 * (f64(good) / f64(total));
}

// transfer mastery in [0,1] — Rust TransferSuite::mastery().
host det fn bench_mastery(suite: Json) -> f64 {
    return bench_score(suite) / 100.0;
}

// last n history entries, newest first — Rust history_summary().
host det fn bench_history(suite: Json, n: i64) -> [Json] {
    var hist: [Json] = [];
    match suite {
        Json::Obj(s) => {
            if has(s, "history") {
                match s["history"] { Json::Arr(h) => { hist = h; } _ => { } }
            }
        }
        _ => { }
    }
    var out_h: [Json] = [];
    var i = len(hist);
    var taken = 0;
    while i > 0 && taken < n {
        i = i - 1;
        push(out_h, hist[i]);
        taken = taken + 1;
    }
    return out_h;
}

// ── output validation — exact port of BenchmarkSuite::validate ──────────

host det fn bench_token_match(expected: str, out_tokens: [str], tol: f64, use_tol: bool) -> bool {
    let exp = trim(expected);
    let eb = str_parse_f64(exp);
    match eb {
        Json::Num(b) => {
            if use_tol {
                for i in 0..len(out_tokens) {
                    match str_parse_f64(out_tokens[i]) {
                        Json::Num(a) => {
                            var ad = a - b;
                            if ad < 0.0 { ad = 0.0 - ad; }
                            if ad <= tol { return true; }
                        }
                        _ => { }
                    }
                }
                return false;
            }
        }
        _ => { }
    }
    for i in 0..len(out_tokens) {
        if out_tokens[i] == exp { return true; }
    }
    return false;
}

// tolerance < 0 means None.
host det fn bench_validate(expected: str, output: str, tolerance: f64) -> bool {
    let toks = split_ws(output);
    return bench_token_match(expected, toks, tolerance, tolerance >= 0.0);
}

// ── dialectical repair — wired through strategy.cy ──────────────────────

import "strategy.cy";

// if mastery < 0.85 and the task failed, synthesize a repaired strategy.
// → new strategy key, or "" when no repair is needed.
host fn bench_repair_if_needed(
    db: str, suite: Json, domain: str, desc: str, succeeded: bool, error: str
) -> str cap fs {
    if succeeded || bench_mastery(suite) >= 0.85 { return ""; }
    let key = "transfer_" + replace_all(domain, " ", "_");
    // dummy strategy shell, same fields the Rust builds
    let dummy = Json::Obj({
        "key": Json::Str(key),
        "problem": Json::Str(desc),
        "language": Json::Str("python"),
        "code": Json::Str("")
    });
    // thesis/antithesis/synthesize on the dummy — mirrors the Rust flow.
    let t = thesis_of(dummy, "x", 1);
    let a = antithesis_of(error, 2);
    let _strength = contradiction_strength(t, a);
    var rep = synthesize(dummy, error);
    let new_key = key + "_synth_3";
    match rep {
        Json::Obj(rm) => {
            rm["key"] = Json::Str(new_key);
            rm["problem"] = Json::Str(jstr_of(rm["problem"]) + " [domain: " + domain + "]");
        }
        _ => { }
    }
    if db_put(db, rep) { return new_key; }
    return "";
}

// ── the PILOT report — exact port of PilotReport::write_report ─────────

host det fn bench_report(runs_json: Json) -> str {
    var runs: [Json] = [];
    match runs_json { Json::Arr(a) => { runs = a; } _ => { } }
    let total = len(runs);
    var passed = 0;
    var lat_sum = 0.0;
    var lat_max = 0.0;
    var rss_peak = 0.0;
    var tok_total = 0.0;
    var leak_max = 0.0;
    for i in 0..len(runs) {
        match runs[i] {
            Json::Obj(r) => {
                if has(r, "success") {
                    match r["success"] { Json::Bool(b) => { if b { passed = passed + 1; } } _ => { } }
                }
                lat_sum = lat_sum + jnum(r["latency_us"]);
                let l = jnum(r["latency_us"]);
                if l > lat_max { lat_max = l; }
                let rs = jnum(r["peak_rss_mb"]);
                if rs > rss_peak { rss_peak = rs; }
                tok_total = tok_total + jnum(r["token_count"]);
                let lk = jnum(r["leak_score"]);
                if lk > leak_max { leak_max = lk; }
            }
            _ => { }
        }
    }
    var avg_lat = 0.0;
    if total > 0 { avg_lat = lat_sum / f64(total); }
    var lines = "";
    for i in 0..len(runs) {
        lines = lines + bench_report_line(runs[i]);
    }
    return "# Pilot Evaluation Metrics\n\nGenerated by the Bad Apple transfer-evaluation harness.\n\n## Summary\n\n- Total runs: "
        + str(total) + "\n- Passed: " + str(passed) + " / " + str(total)
        + "\n- Average latency: " + str(avg_lat) + " µs\n- Max latency: " + str(lat_max)
        + " µs\n- Peak RSS: " + str(rss_peak) + " MB\n- Max memory-leak score: " + str(leak_max)
        + " (0.00 floor)\n- Total tokens tracked: " + str(tok_total)
        + "\n\n## Per-run telemetry\n\n| Domain | Result | Latency (µs) | Peak RSS (MB) | Tokens | Leak Score |\n|---|---|---|---|---|---|\n"
        + lines;
}

host det fn bench_report_line(run: Json) -> str {
    var domain = "";
    var status = "FAIL";
    var lat = 0.0;
    var rss = 0.0;
    var tok = 0.0;
    var leak = 0.0;
    match run {
        Json::Obj(r) => {
            domain = jstr_of(r["domain"]);
            match r["success"] { Json::Bool(b) => { if b { status = "PASS"; } } _ => { } }
            lat = jnum(r["latency_us"]);
            rss = jnum(r["peak_rss_mb"]);
            tok = jnum(r["token_count"]);
            leak = jnum(r["leak_score"]);
        }
        _ => { }
    }
    return "| " + domain + " | " + status + " | " + str(lat) + " | " + str(rss)
        + " | " + str(tok) + " | " + str(leak) + " |\n";
}
