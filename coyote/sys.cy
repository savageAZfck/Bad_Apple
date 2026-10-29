// sys.cy — the sovereign syscall layer.
//
// The extern surface every Bad Apple organism shares. Each declaration is
// bound from the Rust shell via `coyote_sys::bind_all` (src/coyote_sys.rs)
// and is checked under the same constitution as natives: capabilities are
// transitive, `det`/`stoch` are enforced, and using an extern whose cap the
// program never declared is a compile error.
//
// Convention: `str` is utf8. Binary payloads travel as hex strings
// (`hex_encode`/`fs_read_hex`/`fs_write_hex`) — no bytes type, no leakage.

// Bytes is a compact in-memory form for binary payloads (hex string
// inside). The codecs translate it to the wire form each format uses:
// cbor/json emit it as an array of small ints (what serde produces for
// Vec<u8> without serde_bytes); decode collapses homogeneous small-int
// arrays back into Bytes.
enum Json { Null, Bool(bool), Num(f64), Str(str), Arr([Json]), Obj({str: Json}), Bytes(str) }

// ── codec ──────────────────────────────────────────────────────────────
extern det fn json_parse(s: str) -> Json;   // Json::Null on malformed input
extern det fn json_valid(s: str) -> bool;
extern det fn json_emit(j: Json) -> str;

extern det fn hex_encode(s: str) -> str;
extern det fn hex_decode(h: str) -> str;    // "" on invalid hex / non-utf8
extern det fn b64_encode(s: str) -> str;
extern det fn b64_decode(s: str) -> str;    // "" on invalid base64

// utf8 key + message → hex signature (ledger entries, utf8 payloads)
extern det fn hmac_sha256(key: str, msg: str) -> str;
// binary-safe form — key and message are hex strings
extern det fn hmac_sha256b(key_hex: str, msg_hex: str) -> str;
// constant-time verification — proof_hex compared via verify_slice
extern det fn hmac_verify(key_hex: str, msg_hex: str, proof_hex: str) -> bool;

// ── process ────────────────────────────────────────────────────────────
extern stoch fn time_ms() -> i64;           // unix epoch ms — nondeterministic
extern stoch fn time_iso() -> str;          // ISO8601 Z — matches ledger `ts`
extern det fn env_get(name: str) -> str;    // "" when unset
extern det fn sys_args() -> [str];

// ── filesystem (cap fs) ────────────────────────────────────────────────
extern det fn fs_exists(path: str) -> bool cap fs;
extern det fn fs_size(path: str) -> i64 cap fs;   // -1 on error
extern det fn fs_list(path: str) -> [str] cap fs;
extern fn fs_append(path: str, content: str) -> bool cap fs;
extern fn fs_mkdir(path: str) -> bool cap fs;
extern fn fs_rename(a: str, b: str) -> bool cap fs;
extern fn fs_read_hex(path: str) -> str cap fs;   // binary file → hex, "" on error
extern fn fs_write_hex(path: str, hex: str) -> bool cap fs;

// advisory flock — fs_flock returns a handle (>=0) or -1; fs_funlock releases.
extern fn fs_flock(path: str) -> i64 cap fs;
extern fn fs_funlock(h: i64) -> unit cap fs;

// ── crypto (cap crypto) ────────────────────────────────────────────────
// binary-safe forms — everything is hex on the wire between .cy and Rust.
extern stoch fn rand_hex(n: i64) -> str;                    // n random bytes → hex
extern stoch fn aead_seal(key_hex: str, pt_hex: str) -> str cap crypto;  // [nonce||ct||tag]
extern det fn aead_open(key_hex: str, ct_hex: str) -> str cap crypto;    // pt hex, "" on auth fail

// cbor — ciborium-compatible encode/decode for serde's externally-tagged
// enum representation ({Variant: {fields}}). det, no cap — a pure codec.
extern det fn cbor_encode(j: Json) -> str;                  // → hex, "" on error
extern det fn cbor_decode(hex: str) -> Json;                // Json::Null on error

// ── network (cap net) ──────────────────────────────────────────────────
// blocking handles — the VM is synchronous; streams get a read timeout in
// the binding. connect/listen/accept/send/recv are stoch (the world
// answers), close is det.
extern stoch fn net_tcp_connect(addr: str) -> i64 cap net;  // handle, -1 fail
extern stoch fn net_tcp_listen(addr: str) -> i64 cap net;
extern stoch fn net_unix_connect(path: str) -> i64 cap net; // SLICKS sockets
extern stoch fn net_unix_listen(path: str) -> i64 cap net;  // unlinks stale path first
extern stoch fn net_accept(h: i64) -> i64 cap net;          // conn handle (tcp or unix)
extern stoch fn net_send(h: i64, hex: str) -> bool cap net;
extern stoch fn net_recv(h: i64, max: i64) -> str cap net;  // ≤max bytes as hex, "" on eof/err
extern det fn net_close(h: i64) -> unit cap net;
extern det fn net_local_addr(h: i64) -> str cap net;        // bound addr of a listener, "" on err

// ── ed25519 (cap crypto) — org policy signatures, hex on the wire ─────
extern stoch fn ed25519_keygen() -> str cap crypto;         // → secret hex (32B)
extern det fn ed25519_pub(sec_hex: str) -> str cap crypto;  // → pubkey hex
extern det fn ed25519_sign(sec_hex: str, msg_hex: str) -> str cap crypto;
extern det fn ed25519_verify(pub_hex: str, msg_hex: str, sig_hex: str) -> bool cap crypto;

// ── process exec (cap proc) ────────────────────────────────────────────
// runs cmd with args, captures output → Json {code, out, err}.
// deliberately coarse: the checker still requires `cap proc` transitively.
extern stoch fn sys_exec(cmd: str, args: [str]) -> Json cap proc;
// bridge into the daemon's agent surface — the .cy MCP server calls the
// local agent exactly like the Rust one does.
extern stoch fn mcp_agent_call(name: str, params: Json) -> Json cap cortex;
// env override — {str:str} pairs applied on top of the inherited env;
// keys mapped to "" are REMOVED from the child env.
extern stoch fn sys_exec_env(cmd: str, args: [str], env: {str: str}) -> Json cap proc;
extern stoch fn sys_sleep_ms(ms: i64) -> unit cap proc;

// ── fs extras ──────────────────────────────────────────────────────────
extern fn fs_chmod(path: str, mode_octal: i64) -> bool cap fs;
extern fn fs_unlink(path: str) -> bool cap fs;
extern fn fs_remove_all(path: str) -> bool cap fs;   // file or dir tree; true if absent
extern det fn fs_realpath(path: str) -> str cap fs;  // canonicalized, "" on error
extern det fn fs_is_symlink(path: str) -> bool cap fs;
extern det fn fs_is_dir(path: str) -> bool cap fs;
extern det fn fs_is_file(path: str) -> bool cap fs;   // regular file, follows nothing
extern det fn fs_nlink(path: str) -> i64 cap fs;     // lstat hard-link count, -1 err
extern det fn fs_lstat_symlink(path: str) -> bool cap fs; // leaf symlink even if dangling
// ── automation cage exec — the fd-safe mutation boundary ───────────────
// The .cy side owns parsing + validation policy (cage.cy); the mutation
// itself stays in Rust — openat chains + O_NOFOLLOW are the TOCTOU-safe
// mechanism and are not portable to the VM. The extern re-runs the Rust
// validator before touching anything, so this boundary fails closed.
// → Json report {operation, paths, result, elapsed_ms} or {error}
extern fn cage_exec(action_json: str, roots_csv: str, home: str, log_path: str) -> Json cap fs;
// O_NOFOLLOW append for the cage audit log (parent symlink-checked).
extern fn cage_log_append(path: str, line: str) -> bool cap fs;
extern stoch fn sys_exe_dir() -> str;                // dir of the running binary
extern det fn time_parse_iso(s: str) -> i64;         // RFC3339 → unix ms, -1 on error

// ── string extras ──────────────────────────────────────────────────────
extern det fn str_lower(s: str) -> str;
extern det fn str_parse_f64(s: str) -> Json; // Json::Num on success, Null on parse error
extern det fn rss_mb() -> f64;               // process resident set in MB, 0.0 unavailable
extern det fn str_upper(s: str) -> str;

// ── ledger forensics (cap fs, crypto) ──────────────────────────────────
// read bytes [offset..EOF] of a file as utf8-lossy text — ledger tailing.
extern det fn fs_read_from(path: str, offset: i64) -> str cap fs;
// random-access io for model chunk transfers — offsets in bytes.
extern fn fs_read_at(path: str, off: i64, len: i64) -> str cap fs;  // → hex, "" on err/eof-0
extern fn fs_write_at(path: str, off: i64, hex: str) -> i64 cap fs; // → bytes written, -1 err
extern fn fs_truncate(path: str, len: i64) -> bool cap fs;          // create+set_len (prealloc)
extern fn sha256_file(path: str) -> str cap fs;                     // streaming digest → hex
// verify one JSONL line against the sovereign format set — all three
// historical layouts. secrets are hex candidates; → Json {ok, tip, err}.
extern det fn ledger_verify_line(line: str, secrets: [str], prev_tip: str) -> Json cap crypto;
// SLICKS key candidates from a key file — hex per candidate.
extern det fn ledger_key_candidates(path: str) -> [str] cap fs;
// causal-graph explanation for a finding's observed text → "" or the chain.
extern det fn causal_explain(text: str) -> str;
// record a relation in the shared causal graph (strategy put path).
extern det fn causal_relate(from: str, rel: str, to: str) -> unit;

// ── durable KV (cap fs) — redb-backed, values are utf8 JSON ───────────
extern det fn kv_get(db: str, key: str) -> str cap fs;        // → value or ""
extern fn kv_put(db: str, key: str, val: str) -> bool cap fs;
extern fn kv_del(db: str, key: str) -> bool cap fs;
extern det fn kv_list(db: str) -> Json cap fs;   // Arr of {key, value}
extern det fn kv_scan_prefix(db: str, prefix: str) -> Json cap fs;
extern det fn kv_count_prefix(db: str, prefix: str) -> i64 cap fs;

// ── scavenger (cap fs) — the offline file index ────────────────────────
// recursive walk: → Json Arr of {path, size, mtime} (mtime in ns)
extern det fn fs_walk(root: str, depth: i64) -> Json cap fs;
extern det fn fs_mtime(path: str) -> i64 cap fs;  // ns since epoch, 0 on err
// symlink-safe read (O_NOFOLLOW) with a byte cap and utf8-boundary trim
extern det fn fs_read_nofollow(path: str, cap_bytes: i64) -> str cap fs;
// semantic embedding via tensor_brain — → Json {ok, vec: [f64]} or {ok:false}
extern det fn scav_embed(text: str) -> Json cap fs;
// f32 bit-cast helpers — grounded index stores vectors as u32 bits for
// byte-compat with the Rust record layout
extern det fn math_sqrt(v: f64) -> f64;
extern det fn f32_bits(v: f64) -> i64;          // (v as f32).to_bits()
extern det fn f32_from_bits(b: i64) -> f64;     // f32::from_bits → f64
extern det fn f32_sqrt(v: f64) -> f64;          // (v as f32).sqrt() as f64 — exact f32 rounding
extern det fn f32_round(v: f64) -> f64;         // (v as f32) as f64 — f32 rounding
// "%Y-%m-%d %H:%M:%S UTC" — proposal timestamps.
extern stoch fn time_stamp() -> str;

// ── sentinel bridge (cap sentinel) ────────────────────────────────────
// the dedicated oversight model: verdict logic lives in sentinel.cy, the
// resident AneCore engine call lives in the shell.
// generate → Json {status: "unavailable"|"ok"|"err", text}
extern stoch fn sentinel_generate(prompt: str, system: str, max_tokens: i64, max_ctx: i64) -> Json cap sentinel;
extern det fn sentinel_tally(verdict: str, latency_us: i64) -> unit cap sentinel;

// ── coyote self-bridge (cap fs) ────────────────────────────────────────
// the organism asks its own shell to compile a DFA pattern set into the
// ANE organ binaries (firewall.mlmodelc). patterns is a JSON array of
// str → Json {states, ok, err}. Writes artifacts, so `cap fs`.
extern det fn dfa_ane_compile(patterns: Json, out_dir: str) -> Json cap fs;

// ── mil_spec bridge — .cy authors MIL programs ──────────────────────────
// mil_emit(spec_json) → encoded model.mlmodel bytes as hex ("ERR:..." on
// failure). mil_write_package(dir, spec_hex, weight_src) writes the
// .mlpackage (weight_src = "" for no weights dir).
extern det fn mil_emit(spec: Json) -> Json cap proc;

// ── safetensors + weight surgery (converters author specs with real bytes)
// st_f16/st_f16_rows read tensors as fp16 hex ("" on miss). w_* are pure
// byte transforms on fp16 hex payloads; w_q8/q4_rows return Json {q, s|lut}.
extern det fn st_f16(dir: str, name: str) -> str cap fs;
extern det fn st_f16_rows(dir: str, name: str, r0: i64, r1: i64) -> str cap fs;
extern det fn st_tensors(dir: str) -> Json cap fs;   // Json::Arr of names
extern det fn w_f16_plus_one(hexs: str) -> str;
extern det fn w_f16_scale_cols(w: str, col: str) -> str;
extern det fn w_f16_scale_all(hexs: str, scale: f64) -> str;
extern det fn w_tile(hexs: str, n: i64) -> str;
extern det fn w_q8_rows(hexs: str, rows: i64) -> Json;
extern det fn w_q4_rows(hexs: str, rows: i64) -> Json;
extern fn mil_write_package(dir: str, spec_hex: str, weight_src: str) -> bool cap fs;
extern fn log_err(s: str) -> unit cap proc;            // eprintln — progress lines
