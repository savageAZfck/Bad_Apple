//! Extern syscall layer for Coyote organisms — the Rust shell's half of
//! `coyote/sys.cy`. Every `.cy` extern declaration binds to a real OS call
//! here; the Coyote checker enforces capability/det contracts on the source
//! side, and these callbacks are the only place those capabilities become
//! actual syscalls.
//!
//! Convention: `str` is utf8. Binary payloads travel as hex strings.
//! Fallible externs return an in-band failure value ("" / false / -1 /
//! Json::Null) rather than trapping the VM — callers check, like C.

use base64::{engine::general_purpose::STANDARD, Engine};
use coyote::vm::{Val, Vm};
use hmac::{Hmac, KeyInit, Mac};
use sha2::Sha256;
use std::cell::RefCell;
use std::collections::HashMap;
use std::io::Write;
use std::net::ToSocketAddrs;
use std::os::fd::AsRawFd;
use std::path::PathBuf;
use std::rc::Rc;

type HmacSha256 = Hmac<Sha256>;

/// Resolved (enum_idx, variant_idx) for the `Json` enum declared in sys.cy.
#[derive(Clone, Copy)]
struct JsonIdx {
    en: u32,
    null: u32,
    boolv: u32,
    num: u32,
    strv: u32,
    arr: u32,
    obj: u32,
    bytes: u32,
}

fn json_idx(vm: &Vm) -> Result<JsonIdx, String> {
    let var = |v: &str| {
        vm.enum_variant("Json", v)
            .ok_or_else(|| format!("sys.cy enum Json missing variant `{v}`"))
    };
    let null = var("Null")?;
    Ok(JsonIdx {
        en: null.0,
        null: null.1,
        boolv: var("Bool")?.1,
        num: var("Num")?.1,
        strv: var("Str")?.1,
        arr: var("Arr")?.1,
        obj: var("Obj")?.1,
        bytes: var("Bytes")?.1,
    })
}

fn jnull(j: &JsonIdx) -> Val {
    Val::En(j.en, j.null, Rc::new(Vec::new()))
}

fn json_to_val(v: &serde_json::Value, j: &JsonIdx) -> Val {
    json_to_val_mode(v, j, false)
}

/// `bytes_mode` is set only for CBOR payloads — serde encodes Vec<u8> as
/// an int array on that wire, so homogeneous small-int arrays collapse
/// to Json::Bytes there. Plain JSON arrays stay Arr (shape metadata,
/// offsets, etc. are legit small ints).
fn json_to_val_mode(v: &serde_json::Value, j: &JsonIdx, bytes_mode: bool) -> Val {
    let en = |vi: u32, payload: Vec<Val>| Val::En(j.en, vi, Rc::new(payload));
    match v {
        serde_json::Value::Null => jnull(j),
        serde_json::Value::Bool(b) => en(j.boolv, vec![Val::B(*b)]),
        serde_json::Value::Number(n) => en(j.num, vec![Val::F(n.as_f64().unwrap_or(0.0))]),
        serde_json::Value::String(s) => en(j.strv, vec![Val::S(Rc::from(s.as_str()))]),
        serde_json::Value::Array(a) => {
            // Homogeneous small-int array = serialized Vec<u8>. Collapse
            // to Json::Bytes (hex) so chunk data stays compact in the VM.
            if bytes_mode
                && !a.is_empty()
                && a.iter()
                    .all(|x| x.as_u64().map(|u| u <= 255).unwrap_or(false))
            {
                let hex: String = a
                    .iter()
                    .map(|x| format!("{:02x}", x.as_u64().unwrap()))
                    .collect();
                return en(j.bytes, vec![Val::S(Rc::from(hex.as_str()))]);
            }
            en(
                j.arr,
                vec![Val::V(Rc::new(RefCell::new(
                    a.iter()
                        .map(|x| json_to_val_mode(x, j, bytes_mode))
                        .collect(),
                )))],
            )
        }
        serde_json::Value::Object(o) => en(
            j.obj,
            vec![Val::M(Rc::new(RefCell::new(
                o.iter()
                    .map(|(k, x)| {
                        (
                            Val::S(Rc::from(k.as_str())),
                            json_to_val_mode(x, j, bytes_mode),
                        )
                    })
                    .collect(),
            )))],
        ),
    }
}

fn val_to_json(v: &Val, j: &JsonIdx) -> Result<serde_json::Value, String> {
    match v {
        Val::En(ei, vi, payload) if *ei == j.en => {
            let p0 = payload.first();
            match *vi {
                v if v == j.null => Ok(serde_json::Value::Null),
                v if v == j.boolv => match p0 {
                    Some(Val::B(b)) => Ok(serde_json::Value::Bool(*b)),
                    _ => Err("Json::Bool payload".into()),
                },
                v if v == j.num => match p0 {
                    Some(Val::F(f)) => {
                        // Integral values emit as ints — required for cbor
                        // wire compat (serde usize fields must stay uints).
                        if f.fract() == 0.0 && f.abs() <= (i64::MAX as f64) {
                            Ok(serde_json::json!(*f as i64))
                        } else {
                            Ok(serde_json::json!(f))
                        }
                    }
                    _ => Err("Json::Num payload".into()),
                },
                v if v == j.strv => match p0 {
                    Some(Val::S(s)) => Ok(serde_json::Value::String(s.to_string())),
                    _ => Err("Json::Str payload".into()),
                },
                v if v == j.arr => match p0 {
                    Some(Val::V(xs)) => xs
                        .borrow()
                        .iter()
                        .map(|x| val_to_json(x, j))
                        .collect::<Result<Vec<_>, _>>()
                        .map(serde_json::Value::Array),
                    _ => Err("Json::Arr payload".into()),
                },
                v if v == j.obj => match p0 {
                    Some(Val::M(m)) => {
                        // Canonical emit — sorted keys, compact separators.
                        // serde_json::Map preserves insertion order here, so
                        // sort pairs explicitly; ledger hashing depends on it.
                        let mut pairs: Vec<(String, serde_json::Value)> = Vec::new();
                        for (k, x) in m.borrow().iter() {
                            let Val::S(k) = k else {
                                return Err("Json::Obj non-str key".into());
                            };
                            pairs.push((k.to_string(), val_to_json(x, j)?));
                        }
                        pairs.sort_by(|a, b| a.0.cmp(&b.0));
                        let mut map = serde_json::Map::new();
                        for (k, x) in pairs {
                            map.insert(k, x);
                        }
                        Ok(serde_json::Value::Object(map))
                    }
                    _ => Err("Json::Obj payload".into()),
                },
                v if v == j.bytes => match p0 {
                    // hex payload → int array (serde's Vec<u8> encoding,
                    // identical in both serde_json and ciborium).
                    Some(Val::S(hx)) => {
                        let bytes = hex::decode(hx.as_ref())
                            .map_err(|_| "Json::Bytes bad hex".to_string())?;
                        Ok(serde_json::Value::Array(
                            bytes.iter().map(|b| serde_json::json!(*b)).collect(),
                        ))
                    }
                    _ => Err("Json::Bytes payload".into()),
                },
                _ => Err("unknown Json variant".into()),
            }
        }
        // Lenient: emit plain scalars as their JSON equivalents.
        Val::S(s) => Ok(serde_json::Value::String(s.to_string())),
        Val::I(i) => Ok(serde_json::json!(i)),
        Val::F(f) => Ok(serde_json::json!(f)),
        Val::B(b) => Ok(serde_json::Value::Bool(*b)),
        Val::Unit => Ok(serde_json::Value::Null),
        _ => Err("val_to_json: unsupported value".into()),
    }
}

fn arg_str(args: &[Val], i: usize) -> Result<String, String> {
    match args.get(i) {
        Some(Val::S(s)) => Ok(s.to_string()),
        _ => Err(format!("extern arg {i} must be str")),
    }
}

fn arg_i64(args: &[Val], i: usize) -> Result<i64, String> {
    match args.get(i) {
        Some(Val::I(n)) => Ok(*n),
        _ => Err(format!("extern arg {i} must be i64")),
    }
}

fn s(v: impl Into<String>) -> Val {
    Val::S(Rc::from(v.into().as_str()))
}

fn hmac_hex(key: &[u8], msg: &[u8]) -> String {
    let mut mac = HmacSha256::new_from_slice(key).expect("HMAC accepts any key length");
    mac.update(msg);
    hex::encode(mac.finalize().into_bytes())
}

/// The ledger organism embedded at compile time — `coyote/sys.cy` +
/// `coyote/ledger.cy` written to a temp dir and loaded as one program.
/// This is the port in practice: the audit ledger's implementation IS the
/// `.cy` source; these helpers are just its syscall bindings.
const SYS_CY: &str = include_str!("../coyote/sys.cy");
const STRLIB_CY: &str = include_str!("../coyote/strlib.cy");
const LEDGER_CY: &str = include_str!("../coyote/ledger.cy");
const SLICKS_CY: &str = include_str!("../coyote/slicks.cy");
const CAGE_CY: &str = include_str!("../coyote/cage.cy");
const MESH_BRAIN_CY: &str = include_str!("../coyote/mesh_brain.cy");
const CERT_CY: &str = include_str!("../coyote/cert.cy");
const PROTOCOL_CY: &str = include_str!("../coyote/protocol.cy");
const P2P_CY: &str = include_str!("../coyote/p2p.cy");
const ORG_POLICY_CY: &str = include_str!("../coyote/org_policy.cy");
const CONFIG_CY: &str = include_str!("../coyote/config.cy");
const MESH_SYNC_CY: &str = include_str!("../coyote/mesh_sync.cy");
const FIREWALL_CY: &str = include_str!("../coyote/firewall.cy");
const SENTINEL_CY: &str = include_str!("../coyote/sentinel.cy");
const IFY_CY: &str = include_str!("../coyote/ify.cy");
const STRATEGY_CY: &str = include_str!("../coyote/strategy.cy");
const SCAVENGER_CY: &str = include_str!("../coyote/scavenger.cy");
const VAULT_CY: &str = include_str!("../coyote/vault.cy");
const BLUEPRINT_CY: &str = include_str!("../coyote/blueprint.cy");
const WATCHER_CY: &str = include_str!("../coyote/watcher.cy");
const MCP_CY: &str = include_str!("../coyote/mcp.cy");
const BENCHMARK_CY: &str = include_str!("../coyote/benchmark.cy");
const CLI_CY: &str = include_str!("../coyote/cli.cy");
const MIL_CY: &str = include_str!("../coyote/mil.cy");
const LAYER_CY: &str = include_str!("../coyote/layer.cy");

fn write_embedded_cy(dir: &std::path::Path, name: &str, text: &str) -> Result<(), String> {
    let path = dir.join(name);
    let stale = std::fs::read_to_string(&path)
        .map(|old| old != text)
        .unwrap_or(true);
    if stale {
        std::fs::write(&path, text).map_err(|e| format!("write {}: {e}", path.display()))?;
    }
    Ok(())
}

/// Resident ledger VM — loaded once, CPU backend (host-stratum only; there
/// are no tensor fns to compile to silicon).
static LEDGER_VM: std::sync::OnceLock<std::sync::Mutex<Option<Vm>>> = std::sync::OnceLock::new();

fn resident_ledger_vm() -> Result<std::sync::MutexGuard<'static, Option<Vm>>, String> {
    let m = LEDGER_VM.get_or_init(|| std::sync::Mutex::new(None));
    let mut g = m
        .lock()
        .map_err(|_| "ledger vm lock poisoned".to_string())?;
    if g.is_none() {
        let dir = std::env::temp_dir().join("badapple_cy_ledger");
        std::fs::create_dir_all(&dir).map_err(|e| format!("cy dir: {e}"))?;
        write_embedded_cy(&dir, "sys.cy", SYS_CY)?;
        write_embedded_cy(&dir, "ledger.cy", LEDGER_CY)?;
        let build = dir.join("build");
        let mut vm = coyote::load_vm(&dir.join("ledger.cy"), &build, true, false, true)?;
        bind_all(&mut vm)?;
        *g = Some(vm);
    }
    Ok(g)
}

fn attest(ev: &str, args: Vec<Val>) {
    let r = (|| -> Result<(), String> {
        let mut g = resident_ledger_vm()?;
        match g.as_mut().unwrap().call_fn(ev, args) {
            Ok(Val::B(true)) => Ok(()),
            Ok(_) => Err("ledger append returned false".into()),
            Err(e) => Err(e),
        }
    })();
    if let Err(e) = r {
        eprintln!("[coyote ledger] delegated attestation failed: {e}");
    }
}

/// Asker-side attestation for delegated inference — `ledger.cy`'s
/// `delegated_ask`/`delegated_answer` on the local chain, mirroring the
/// serving peer's `delegated_query`/`delegated_response`.
pub fn attest_delegated_ask(peer: &str, request_id: &str, prompt: &str) {
    attest(
        "ledger_delegated_ask",
        vec![s(peer), s(request_id), s(prompt)],
    );
}

pub fn attest_delegated_answer(peer: &str, request_id: &str, status: &str, response: &str) {
    attest(
        "ledger_delegated_answer",
        vec![s(peer), s(request_id), s(status), s(response)],
    );
}

/// Bind every `extern fn` declared in `coyote/sys.cy` to this VM.
/// Call once after `load_vm`, before invoking host fns.
pub fn bind_all(vm: &mut Vm) -> Result<(), String> {
    let j = json_idx(vm)?;

    // ── codec ──
    {
        let j = j;
        vm.bind_extern("json_parse", move |args| {
            let text = arg_str(args, 0)?;
            match serde_json::from_str::<serde_json::Value>(&text) {
                Ok(v) => Ok(json_to_val(&v, &j)),
                Err(_) => Ok(jnull(&j)),
            }
        });
    }
    vm.bind_extern("json_valid", |args| {
        let text = arg_str(args, 0)?;
        Ok(Val::B(
            serde_json::from_str::<serde_json::Value>(&text).is_ok(),
        ))
    });
    let j2 = json_idx(vm)?;
    vm.bind_extern("json_emit", move |args| {
        let v = args.first().ok_or("json_emit needs 1 arg")?;
        Ok(s(val_to_json(v, &j2)?.to_string()))
    });
    vm.bind_extern("hex_encode", |args| Ok(s(hex::encode(arg_str(args, 0)?))));
    vm.bind_extern("hex_decode", |args| {
        let h = arg_str(args, 0)?;
        let out = hex::decode(&h)
            .ok()
            .and_then(|b| String::from_utf8(b).ok())
            .unwrap_or_default();
        Ok(s(out))
    });
    vm.bind_extern("b64_encode", |args| {
        Ok(s(STANDARD.encode(arg_str(args, 0)?)))
    });
    vm.bind_extern("b64_decode", |args| {
        let b = arg_str(args, 0)?;
        let out = STANDARD
            .decode(&b)
            .ok()
            .and_then(|b| String::from_utf8(b).ok())
            .unwrap_or_default();
        Ok(s(out))
    });
    vm.bind_extern("hmac_sha256", |args| {
        let key = arg_str(args, 0)?;
        let msg = arg_str(args, 1)?;
        Ok(s(hmac_hex(key.as_bytes(), msg.as_bytes())))
    });
    vm.bind_extern("hmac_sha256b", |args| {
        let key = hex::decode(arg_str(args, 0)?).unwrap_or_default();
        let msg = hex::decode(arg_str(args, 1)?).unwrap_or_default();
        Ok(s(hmac_hex(&key, &msg)))
    });

    // ── process ──
    vm.bind_extern("time_ms", |_args| {
        let ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as i64)
            .unwrap_or(0);
        Ok(Val::I(ms))
    });
    vm.bind_extern("time_iso", |_args| {
        let now = chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true);
        Ok(s(now))
    });
    vm.bind_extern("env_get", |args| {
        let name = arg_str(args, 0)?;
        Ok(s(std::env::var(&name).unwrap_or_default()))
    });
    vm.bind_extern("sys_args", |_args| {
        Ok(Val::V(Rc::new(RefCell::new(
            std::env::args().map(s).collect(),
        ))))
    });

    // ── filesystem ──
    vm.bind_extern("fs_exists", |args| {
        Ok(Val::B(std::path::Path::new(&arg_str(args, 0)?).exists()))
    });
    vm.bind_extern("fs_size", |args| {
        let n = std::fs::metadata(arg_str(args, 0)?)
            .map(|m| m.len() as i64)
            .unwrap_or(-1);
        Ok(Val::I(n))
    });
    vm.bind_extern("fs_list", |args| {
        let mut names: Vec<String> = std::fs::read_dir(arg_str(args, 0)?)
            .map(|rd| {
                rd.filter_map(|e| e.ok())
                    .filter_map(|e| e.file_name().into_string().ok())
                    .collect()
            })
            .unwrap_or_default();
        names.sort(); // sorted — det externs return deterministic results
        Ok(Val::V(Rc::new(RefCell::new(
            names.into_iter().map(s).collect(),
        ))))
    });
    vm.bind_extern("fs_append", |args| {
        let path = arg_str(args, 0)?;
        let content = arg_str(args, 1)?;
        let ok = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .and_then(|mut f| f.write_all(content.as_bytes()))
            .is_ok();
        Ok(Val::B(ok))
    });
    vm.bind_extern("fs_mkdir", |args| {
        Ok(Val::B(std::fs::create_dir_all(arg_str(args, 0)?).is_ok()))
    });
    vm.bind_extern("fs_rename", |args| {
        Ok(Val::B(
            std::fs::rename(arg_str(args, 0)?, arg_str(args, 1)?).is_ok(),
        ))
    });
    vm.bind_extern("fs_read_hex", |args| {
        let out = std::fs::read(arg_str(args, 0)?)
            .map(|b| hex::encode(b))
            .unwrap_or_default();
        Ok(s(out))
    });
    vm.bind_extern("fs_write_hex", |args| {
        let path = arg_str(args, 0)?;
        let h = arg_str(args, 1)?;
        let ok = hex::decode(&h)
            .ok()
            .map(|b| std::fs::write(&path, b).is_ok())
            .unwrap_or(false);
        Ok(Val::B(ok))
    });

    // random-access io — model chunk transfers
    vm.bind_extern("fs_read_at", |args| {
        use std::io::{Read, Seek, SeekFrom};
        let path = arg_str(args, 0)?;
        let off = arg_i64(args, 1)?.max(0) as u64;
        let len = arg_i64(args, 2)?.max(0) as usize;
        let mut f = match std::fs::File::open(&path) {
            Ok(f) => f,
            Err(_) => return Ok(s("")),
        };
        if f.seek(SeekFrom::Start(off)).is_err() {
            return Ok(s(""));
        }
        let mut buf = vec![0u8; len];
        match f.read(&mut buf) {
            Ok(0) => Ok(s("")),
            Ok(n) => Ok(s(hex::encode(&buf[..n]))),
            Err(_) => Ok(s("")),
        }
    });
    vm.bind_extern("fs_write_at", |args| {
        use std::io::{Seek, SeekFrom, Write};
        let path = arg_str(args, 0)?;
        let off = arg_i64(args, 1)?.max(0) as u64;
        let Ok(data) = hex::decode(arg_str(args, 2)?) else {
            return Ok(Val::I(-1));
        };
        let mut f = match std::fs::OpenOptions::new().write(true).open(&path) {
            Ok(f) => f,
            Err(_) => return Ok(Val::I(-1)),
        };
        if f.seek(SeekFrom::Start(off)).is_err() {
            return Ok(Val::I(-1));
        }
        match f.write(&data) {
            Ok(n) => Ok(Val::I(n as i64)),
            Err(_) => Ok(Val::I(-1)),
        }
    });
    vm.bind_extern("fs_truncate", |args| {
        let path = arg_str(args, 0)?;
        let len = arg_i64(args, 1)?.max(0) as u64;
        let ok = std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&path)
            .and_then(|f| f.set_len(len))
            .is_ok();
        Ok(Val::B(ok))
    });
    vm.bind_extern("sha256_file", |args| {
        use std::io::Read;
        let path = arg_str(args, 0)?;
        let mut f = match std::fs::File::open(&path) {
            Ok(f) => f,
            Err(_) => return Ok(s("")),
        };
        let mut h = sha2::Sha256::new();
        let mut buf = [0u8; 65536];
        loop {
            match f.read(&mut buf) {
                Ok(0) => break,
                Ok(n) => {
                    use sha2::Digest;
                    h.update(&buf[..n]);
                }
                Err(_) => return Ok(s("")),
            }
        }
        use sha2::Digest;
        Ok(s(hex::encode(h.finalize())))
    });

    vm.bind_extern("fs_is_file", |args| {
        Ok(Val::B(
            std::fs::metadata(arg_str(args, 0)?)
                .map(|m| m.is_file())
                .unwrap_or(false),
        ))
    });
    vm.bind_extern("fs_nlink", |args| {
        use std::os::unix::fs::MetadataExt;
        let n = std::fs::symlink_metadata(arg_str(args, 0)?)
            .map(|m| m.nlink() as i64)
            .unwrap_or(-1);
        Ok(Val::I(n))
    });
    vm.bind_extern("fs_lstat_symlink", |args| {
        Ok(Val::B(
            std::fs::symlink_metadata(arg_str(args, 0)?)
                .map(|m| m.file_type().is_symlink())
                .unwrap_or(false),
        ))
    });

    // ── automation cage — the fd-safe mutation boundary ──
    let jcage = json_idx(vm)?;
    vm.bind_extern("cage_exec", move |args| {
        let action_json = arg_str(args, 0)?;
        let roots_csv = arg_str(args, 1)?;
        let home = arg_str(args, 2)?;
        let log_path = arg_str(args, 3)?;
        let j = jcage.clone();
        Ok((|| -> Result<Val, String> {
            let action: crate::automation_cage::Action =
                serde_json::from_str(&action_json).map_err(|e| format!("action decode: {e}"))?;
            let roots: Vec<std::path::PathBuf> = roots_csv
                .split(':')
                .filter(|s| !s.is_empty())
                .map(std::path::PathBuf::from)
                .collect();
            let cage = crate::automation_cage::AutomationCage::new_with_log(
                roots,
                std::path::PathBuf::from(home),
                std::path::PathBuf::from(log_path),
            )
            .map_err(|e| format!("cage init: {e:#}"))?;
            let report = cage.execute(&action).map_err(|e| format!("{e:#}"))?;
            let mut m: HashMap<String, serde_json::Value> = HashMap::new();
            m.insert("operation".into(), serde_json::json!(report.operation));
            m.insert(
                "paths".into(),
                serde_json::json!(report
                    .paths
                    .iter()
                    .map(|p| p.to_string_lossy().to_string())
                    .collect::<Vec<_>>()),
            );
            m.insert("result".into(), serde_json::json!(report.result));
            m.insert(
                "elapsed_ms".into(),
                serde_json::json!(report.elapsed_ms as u64),
            );
            Ok(json_to_val(
                &serde_json::Value::Object(m.into_iter().collect()),
                &j,
            ))
        })()
        .unwrap_or_else(|e| {
            let mut m: HashMap<String, serde_json::Value> = HashMap::new();
            m.insert("error".into(), serde_json::json!(e));
            json_to_val(&serde_json::Value::Object(m.into_iter().collect()), &j)
        }))
    });
    vm.bind_extern("cage_log_append", |args| {
        use std::io::Write;
        use std::os::unix::fs::OpenOptionsExt;
        let path = arg_str(args, 0)?;
        let line = arg_str(args, 1)?;
        let r = (|| -> Result<(), String> {
            let p = std::path::Path::new(&path);
            let parent = p.parent().ok_or("no log parent")?;
            std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
            let meta = std::fs::symlink_metadata(parent).map_err(|e| e.to_string())?;
            if meta.file_type().is_symlink() {
                return Err("log parent is a symlink".into());
            }
            let mut f = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .custom_flags(libc::O_NOFOLLOW)
                .open(p)
                .map_err(|e| e.to_string())?;
            f.write_all(line.as_bytes()).map_err(|e| e.to_string())?;
            f.write_all(b"\n").map_err(|e| e.to_string())?;
            f.flush().map_err(|e| e.to_string())
        })();
        Ok(Val::B(r.is_ok()))
    });

    // advisory flock — handle table shared by the two lock externs
    let locks: Rc<RefCell<HashMap<i64, std::fs::File>>> = Rc::new(RefCell::new(HashMap::new()));
    let next_lock = Rc::new(RefCell::new(0i64));
    {
        let locks = locks.clone();
        let next_lock = next_lock.clone();
        vm.bind_extern("fs_flock", move |args| {
            let path = arg_str(args, 0)?;
            let f = std::fs::OpenOptions::new()
                .create(true)
                .read(true)
                .write(true)
                .open(&path);
            let Ok(f) = f else { return Ok(Val::I(-1)) };
            let rc = unsafe { libc::flock(f.as_raw_fd(), libc::LOCK_EX) };
            if rc != 0 {
                return Ok(Val::I(-1));
            }
            let mut n = next_lock.borrow_mut();
            *n += 1;
            locks.borrow_mut().insert(*n, f);
            Ok(Val::I(*n))
        });
    }
    vm.bind_extern("fs_funlock", move |args| {
        let h = arg_i64(args, 0)?;
        if let Some(f) = locks.borrow_mut().remove(&h) {
            unsafe { libc::flock(f.as_raw_fd(), libc::LOCK_UN) };
        }
        Ok(Val::Unit)
    });

    // ── crypto ──
    vm.bind_extern("rand_hex", |args| {
        let n = (arg_i64(args, 0)?.max(0) as usize).min(1 << 20);
        let mut b = vec![0u8; n];
        rand::RngCore::fill_bytes(&mut rand::thread_rng(), &mut b);
        Ok(s(hex::encode(b)))
    });
    vm.bind_extern("aead_seal", |args| {
        let key = hex::decode(arg_str(args, 0)?).unwrap_or_default();
        let pt = hex::decode(arg_str(args, 1)?).unwrap_or_default();
        let out = crate::p2p_crypto::P2PCipher::new(&key)
            .and_then(|c| c.encrypt(&pt))
            .map(|ct| hex::encode(ct))
            .unwrap_or_default();
        Ok(s(out))
    });
    vm.bind_extern("aead_open", |args| {
        let key = hex::decode(arg_str(args, 0)?).unwrap_or_default();
        let ct = hex::decode(arg_str(args, 1)?).unwrap_or_default();
        let out = crate::p2p_crypto::P2PCipher::new(&key)
            .and_then(|c| c.decrypt(&ct))
            .map(|pt| hex::encode(pt))
            .unwrap_or_default();
        Ok(s(out))
    });
    let j3 = json_idx(vm)?;
    vm.bind_extern("cbor_encode", move |args| {
        let v = val_to_json(args.first().ok_or("cbor_encode needs 1 arg")?, &j3)?;
        let mut out = Vec::new();
        match ciborium::into_writer(&v, &mut out) {
            Ok(()) => Ok(s(hex::encode(out))),
            Err(_) => Ok(s("")),
        }
    });
    let j4 = json_idx(vm)?;
    vm.bind_extern("cbor_decode", move |args| {
        let bytes = hex::decode(arg_str(args, 0)?).unwrap_or_default();
        match ciborium::from_reader::<ciborium::value::Value, _>(&bytes[..]) {
            Ok(v) => Ok(json_to_val_mode(&cborval_to_json(&v), &j4, true)),
            Err(_) => Ok(jnull(&j4)),
        }
    });

    // ── network — blocking handles; the VM is synchronous ──
    enum NetStream {
        Tcp(std::net::TcpStream),
        Unix(std::os::unix::net::UnixStream),
    }
    enum NetListener {
        Tcp(std::net::TcpListener),
        Unix(std::os::unix::net::UnixListener),
    }
    struct NetTable {
        next: i64,
        streams: HashMap<i64, NetStream>,
        listeners: HashMap<i64, NetListener>,
    }
    let net = Rc::new(RefCell::new(NetTable {
        next: 0,
        streams: HashMap::new(),
        listeners: HashMap::new(),
    }));
    {
        let net = net.clone();
        vm.bind_extern("net_tcp_connect", move |args| {
            let addr = arg_str(args, 0)?;
            let sock = addr
                .to_socket_addrs()
                .ok()
                .and_then(|mut it| it.find(|_| true));
            let Some(sock) = sock else {
                return Ok(Val::I(-1));
            };
            let Ok(st) =
                std::net::TcpStream::connect_timeout(&sock, std::time::Duration::from_secs(10))
            else {
                return Ok(Val::I(-1));
            };
            let _ = st.set_read_timeout(Some(std::time::Duration::from_secs(120)));
            let _ = st.set_write_timeout(Some(std::time::Duration::from_secs(30)));
            let _ = st.set_nodelay(true);
            let mut t = net.borrow_mut();
            t.next += 1;
            let id = t.next;
            t.streams.insert(id, NetStream::Tcp(st));
            Ok(Val::I(id))
        });
    }
    {
        let net = net.clone();
        vm.bind_extern("net_tcp_listen", move |args| {
            let addr = arg_str(args, 0)?;
            let Ok(l) = std::net::TcpListener::bind(&addr) else {
                return Ok(Val::I(-1));
            };
            let mut t = net.borrow_mut();
            t.next += 1;
            let id = t.next;
            t.listeners.insert(id, NetListener::Tcp(l));
            Ok(Val::I(id))
        });
    }
    {
        let net = net.clone();
        vm.bind_extern("net_unix_connect", move |args| {
            let path = arg_str(args, 0)?;
            let Ok(st) = std::os::unix::net::UnixStream::connect(&path) else {
                return Ok(Val::I(-1));
            };
            let _ = st.set_read_timeout(Some(std::time::Duration::from_secs(120)));
            let _ = st.set_write_timeout(Some(std::time::Duration::from_secs(30)));
            let mut t = net.borrow_mut();
            t.next += 1;
            let id = t.next;
            t.streams.insert(id, NetStream::Unix(st));
            Ok(Val::I(id))
        });
    }
    {
        let net = net.clone();
        vm.bind_extern("net_unix_listen", move |args| {
            let path = arg_str(args, 0)?;
            let _ = std::fs::remove_file(&path);
            let Ok(l) = std::os::unix::net::UnixListener::bind(&path) else {
                return Ok(Val::I(-1));
            };
            let mut t = net.borrow_mut();
            t.next += 1;
            let id = t.next;
            t.listeners.insert(id, NetListener::Unix(l));
            Ok(Val::I(id))
        });
    }
    {
        let net = net.clone();
        vm.bind_extern("net_accept", move |args| {
            let h = arg_i64(args, 0)?;
            let st = {
                let t = net.borrow();
                match t.listeners.get(&h) {
                    Some(NetListener::Tcp(l)) => l.accept().ok().map(|(s, _)| NetStream::Tcp(s)),
                    Some(NetListener::Unix(l)) => l.accept().ok().map(|(s, _)| NetStream::Unix(s)),
                    None => None,
                }
            };
            let Some(st) = st else {
                return Ok(Val::I(-1));
            };
            match &st {
                NetStream::Tcp(s) => {
                    let _ = s.set_read_timeout(Some(std::time::Duration::from_secs(120)));
                }
                NetStream::Unix(s) => {
                    let _ = s.set_read_timeout(Some(std::time::Duration::from_secs(120)));
                }
            }
            let mut t = net.borrow_mut();
            t.next += 1;
            let id = t.next;
            t.streams.insert(id, st);
            Ok(Val::I(id))
        });
    }
    {
        let net = net.clone();
        vm.bind_extern("net_send", move |args| {
            let h = arg_i64(args, 0)?;
            let bytes = hex::decode(arg_str(args, 1)?).unwrap_or_default();
            let mut t = net.borrow_mut();
            let Some(st) = t.streams.get_mut(&h) else {
                return Ok(Val::B(false));
            };
            use std::io::Write as _;
            let ok = match st {
                NetStream::Tcp(s) => s.write_all(&bytes).is_ok(),
                NetStream::Unix(s) => s.write_all(&bytes).is_ok(),
            };
            Ok(Val::B(ok))
        });
    }
    {
        let net = net.clone();
        vm.bind_extern("net_recv", move |args| {
            let h = arg_i64(args, 0)?;
            let max = (arg_i64(args, 1)?.max(1) as usize).min(4 << 20);
            let mut buf = vec![0u8; max];
            let mut t = net.borrow_mut();
            let Some(st) = t.streams.get_mut(&h) else {
                return Ok(s(""));
            };
            use std::io::Read as _;
            let n = match st {
                NetStream::Tcp(s) => s.read(&mut buf),
                NetStream::Unix(s) => s.read(&mut buf),
            };
            match n {
                Ok(0) => Ok(s("")),
                Ok(n) => Ok(s(hex::encode(&buf[..n]))),
                Err(_) => Ok(s("")),
            }
        });
    }
    {
        let net = net.clone();
        vm.bind_extern("net_local_addr", move |args| {
            let h = arg_i64(args, 0)?;
            let t = net.borrow();
            let addr = match t.listeners.get(&h) {
                Some(NetListener::Tcp(l)) => l.local_addr().ok().map(|a| a.to_string()),
                Some(NetListener::Unix(_)) => None,
                None => t.streams.get(&h).and_then(|s| match s {
                    NetStream::Tcp(s) => s.local_addr().ok().map(|a| a.to_string()),
                    NetStream::Unix(_) => None,
                }),
            };
            Ok(s(addr.unwrap_or_default()))
        });
    }
    vm.bind_extern("net_close", move |args| {
        let h = arg_i64(args, 0)?;
        let mut t = net.borrow_mut();
        t.streams.remove(&h);
        t.listeners.remove(&h);
        Ok(Val::Unit)
    });

    // ── ed25519 (cap crypto) — org policy signatures ──
    vm.bind_extern("ed25519_keygen", |_args| {
        let mut sec = [0u8; 32];
        rand::RngCore::fill_bytes(&mut rand::thread_rng(), &mut sec);
        Ok(s(hex::encode(sec)))
    });
    vm.bind_extern("ed25519_pub", |args| {
        let sec = hex::decode(arg_str(args, 0)?).unwrap_or_default();
        let Ok(arr): Result<[u8; 32], _> = sec.try_into() else {
            return Ok(s(""));
        };
        let pubk = ed25519_dalek::SigningKey::from_bytes(&arr)
            .verifying_key()
            .to_bytes();
        Ok(s(hex::encode(pubk)))
    });
    vm.bind_extern("ed25519_sign", |args| {
        use ed25519_dalek::Signer;
        let sec = hex::decode(arg_str(args, 0)?).unwrap_or_default();
        let msg = hex::decode(arg_str(args, 1)?).unwrap_or_default();
        let Ok(arr): Result<[u8; 32], _> = sec.try_into() else {
            return Ok(s(""));
        };
        let sig = ed25519_dalek::SigningKey::from_bytes(&arr).sign(&msg);
        Ok(s(hex::encode(sig.to_bytes())))
    });
    vm.bind_extern("ed25519_verify", |args| {
        let pubk = hex::decode(arg_str(args, 0)?).unwrap_or_default();
        let msg = hex::decode(arg_str(args, 1)?).unwrap_or_default();
        let sig = hex::decode(arg_str(args, 2)?).unwrap_or_default();
        let Ok(pa): Result<[u8; 32], _> = pubk.try_into() else {
            return Ok(Val::B(false));
        };
        let Ok(vk) = ed25519_dalek::VerifyingKey::from_bytes(&pa) else {
            return Ok(Val::B(false));
        };
        let Ok(signature) = ed25519_dalek::Signature::from_slice(&sig) else {
            return Ok(Val::B(false));
        };
        Ok(Val::B(vk.verify_strict(&msg, &signature).is_ok()))
    });

    // ── process exec (cap proc) ──
    let j5 = json_idx(vm)?;
    vm.bind_extern("sys_exec", move |args| {
        let cmd = arg_str(args, 0)?;
        let argv: Vec<String> = match args.get(1) {
            Some(Val::V(v)) => v
                .borrow()
                .iter()
                .filter_map(|x| {
                    if let Val::S(s) = x {
                        Some(s.to_string())
                    } else {
                        None
                    }
                })
                .collect(),
            _ => Vec::new(),
        };
        let mk = |code: i64, out: String, err: String| -> Val {
            let m = vec![
                (
                    Val::S(Rc::from("code")),
                    Val::En(j5.en, j5.num, Rc::new(vec![Val::F(code as f64)])),
                ),
                (
                    Val::S(Rc::from("out")),
                    Val::En(
                        j5.en,
                        j5.strv,
                        Rc::new(vec![Val::S(Rc::from(out.as_str()))]),
                    ),
                ),
                (
                    Val::S(Rc::from("err")),
                    Val::En(
                        j5.en,
                        j5.strv,
                        Rc::new(vec![Val::S(Rc::from(err.as_str()))]),
                    ),
                ),
            ];
            Val::En(
                j5.en,
                j5.obj,
                Rc::new(vec![Val::M(Rc::new(RefCell::new(m)))]),
            )
        };
        match std::process::Command::new(&cmd).args(&argv).output() {
            Ok(o) => Ok(mk(
                o.status.code().unwrap_or(-1) as i64,
                String::from_utf8_lossy(&o.stdout).to_string(),
                String::from_utf8_lossy(&o.stderr).to_string(),
            )),
            Err(e) => Ok(mk(-1, String::new(), format!("{e}"))),
        }
    });

    // sys_exec with env overlay — {str:str} pairs; "" value removes the var.
    let j6 = json_idx(vm)?;
    vm.bind_extern("sys_exec_env", move |args| {
        let cmd = arg_str(args, 0)?;
        let argv: Vec<String> = match args.get(1) {
            Some(Val::V(v)) => v
                .borrow()
                .iter()
                .filter_map(|x| {
                    if let Val::S(s) = x {
                        Some(s.to_string())
                    } else {
                        None
                    }
                })
                .collect(),
            _ => Vec::new(),
        };
        let env: Vec<(String, String)> = match args.get(2) {
            Some(Val::M(m)) => m
                .borrow()
                .iter()
                .filter_map(|(k, v)| {
                    if let (Val::S(k), Val::S(v)) = (k, v) {
                        Some((k.to_string(), v.to_string()))
                    } else {
                        None
                    }
                })
                .collect(),
            _ => Vec::new(),
        };
        let mk = |code: i64, out: String, err: String| -> Val {
            let s1 = |x: &str| Val::En(j6.en, j6.strv, Rc::new(vec![Val::S(Rc::from(x))]));
            let m = vec![
                (
                    Val::S(Rc::from("code")),
                    Val::En(j6.en, j6.num, Rc::new(vec![Val::F(code as f64)])),
                ),
                (Val::S(Rc::from("out")), s1(&out)),
                (Val::S(Rc::from("err")), s1(&err)),
            ];
            Val::En(
                j6.en,
                j6.obj,
                Rc::new(vec![Val::M(Rc::new(RefCell::new(m)))]),
            )
        };
        let mut c = std::process::Command::new(&cmd);
        c.args(&argv);
        for (k, v) in &env {
            if v.is_empty() {
                c.env_remove(k);
            } else {
                c.env(k, v);
            }
        }
        match c.output() {
            Ok(o) => Ok(mk(
                o.status.code().unwrap_or(-1) as i64,
                String::from_utf8_lossy(&o.stdout).to_string(),
                String::from_utf8_lossy(&o.stderr).to_string(),
            )),
            Err(e) => Ok(mk(-1, String::new(), format!("{e}"))),
        }
    });

    vm.bind_extern("sys_sleep_ms", |args| {
        std::thread::sleep(std::time::Duration::from_millis(
            arg_i64(args, 0)?.max(0) as u64
        ));
        Ok(Val::Unit)
    });

    // mil_emit — Coyote authors a MIL program; this lowers it to bytes.
    {
        let jmil = json_idx(vm)?;
        vm.bind_extern("mil_emit", move |args| {
            let spec = args
                .first()
                .and_then(json_val)
                .ok_or_else(|| "mil_emit: arg 0 must be Json".to_string())?;
            match mil_emit_json(&spec) {
                Ok((spec_b, wb_b)) => Ok(json_to_val(
                    &serde_json::json!({
                        "spec": hex::encode(spec_b),
                        "weights": hex::encode(wb_b),
                    }),
                    &jmil,
                )),
                Err(e) => Ok(json_to_val(&serde_json::json!({ "error": e }), &jmil)),
            }
        });
    }

    // ── safetensors + weight-surgery externs (st_*/w_*) ─────────────────
    {
        use std::sync::{Mutex, OnceLock};
        static ST_CACHE: OnceLock<Mutex<HashMap<String, crate::safetensors::StIndex>>> =
            OnceLock::new();
        let st_index = |dir: &str| -> Result<crate::safetensors::StIndex, String> {
            let cache = ST_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
            let mut m = cache.lock().map_err(|e| e.to_string())?;
            if !m.contains_key(dir) {
                let idx = crate::safetensors::StIndex::load(std::path::Path::new(dir))?;
                m.insert(dir.to_string(), idx);
            }
            Ok(m[dir].clone())
        };
        vm.bind_extern("st_f16", move |args| {
            let dir = arg_str(args, 0)?;
            let name = arg_str(args, 1)?;
            match st_index(&dir).and_then(|st| st.get_f16(&name)) {
                Ok(b) => Ok(s(hex::encode(b))),
                Err(_) => Ok(s("")),
            }
        });
        vm.bind_extern("st_f16_rows", move |args| {
            let dir = arg_str(args, 0)?;
            let name = arg_str(args, 1)?;
            let r0 = arg_i64(args, 2)? as u64;
            let r1 = arg_i64(args, 3)? as u64;
            match st_index(&dir).and_then(|st| st.get_f16_rows(&name, r0, r1)) {
                Ok(b) => Ok(s(hex::encode(b))),
                Err(_) => Ok(s("")),
            }
        });
    }
    {
        let jw = json_idx(vm)?;
        vm.bind_extern("st_tensors", move |args| {
            let dir = arg_str(args, 0)?;
            let names: Vec<serde_json::Value> = {
                use std::sync::{Mutex, OnceLock};
                static C2: OnceLock<Mutex<HashMap<String, Vec<String>>>> = OnceLock::new();
                let m = C2.get_or_init(|| Mutex::new(HashMap::new()));
                let mut g = m.lock().map_err(|e| e.to_string())?;
                if !g.contains_key(&dir) {
                    let idx = crate::safetensors::StIndex::load(std::path::Path::new(&dir))
                        .map_err(|e| e.to_string())?;
                    g.insert(dir.clone(), idx.tensors.keys().cloned().collect());
                }
                g[&dir]
                    .iter()
                    .cloned()
                    .map(serde_json::Value::String)
                    .collect()
            };
            Ok(json_to_val(&serde_json::Value::Array(names), &jw))
        });
        vm.bind_extern("w_f16_plus_one", move |args| {
            let b = hex::decode(arg_str(args, 0)?).unwrap_or_default();
            Ok(s(hex::encode(crate::safetensors::f16_plus_one(&b))))
        });
        vm.bind_extern("w_f16_scale_cols", move |args| {
            let w = hex::decode(arg_str(args, 0)?).unwrap_or_default();
            let col = hex::decode(arg_str(args, 1)?).unwrap_or_default();
            Ok(s(hex::encode(crate::safetensors::f16_scale_cols(&w, &col))))
        });
        vm.bind_extern("w_f16_scale_all", move |args| {
            let b = hex::decode(arg_str(args, 0)?).unwrap_or_default();
            let sc = match args.get(1) {
                Some(Val::F(f)) => *f as f32,
                Some(Val::I(i)) => *i as f32,
                _ => 1.0,
            };
            Ok(s(hex::encode(crate::safetensors::f16_scale_all(&b, sc))))
        });
        vm.bind_extern("w_tile", move |args| {
            let b = hex::decode(arg_str(args, 0)?).unwrap_or_default();
            let n = arg_i64(args, 1)?;
            Ok(s(hex::encode(crate::safetensors::tile_head_w(&b, n))))
        });
    }
    {
        let jq = json_idx(vm)?;
        vm.bind_extern("w_q8_rows", move |args| {
            let b = hex::decode(arg_str(args, 0)?).unwrap_or_default();
            let rows = arg_i64(args, 1)?.max(1) as usize;
            let (q, sc) = crate::safetensors::quantize_q8_rows(&b, rows);
            Ok(json_to_val(
                &serde_json::json!({"q": hex::encode(q), "s": hex::encode(sc)}),
                &jq,
            ))
        });
    }
    {
        let jq4 = json_idx(vm)?;
        vm.bind_extern("w_q4_rows", move |args| {
            let b = hex::decode(arg_str(args, 0)?).unwrap_or_default();
            let rows = arg_i64(args, 1)?.max(1) as usize;
            let (q, lut) = crate::safetensors::quantize_q4_rows(&b, rows);
            Ok(json_to_val(
                &serde_json::json!({"q": hex::encode(q), "lut": hex::encode(lut)}),
                &jq4,
            ))
        });
    }
    // mil_write_package(dir, spec_hex, weight_src) → writes .mlpackage
    {
        vm.bind_extern("log_err", |args| {
            eprintln!("{}", arg_str(args, 0)?);
            Ok(Val::Unit)
        });
        vm.bind_extern("mil_write_package", move |args| {
            let dir = std::path::PathBuf::from(arg_str(args, 0)?);
            let spec = hex::decode(arg_str(args, 1)?)
                .map_err(|e| format!("mil_write_package: bad spec hex: {e}"))?;
            let whex = arg_str(args, 2).unwrap_or_default();
            let weights = if whex.is_empty() {
                None
            } else {
                Some(
                    hex::decode(&whex)
                        .map_err(|e| format!("mil_write_package: bad weights hex: {e}"))?,
                )
            };
            mil_spec::write_mlpackage(&dir, &spec, weights.as_deref())
                .map(|_| Val::B(true))
                .map_err(|e| format!("mil_write_package: {e}"))
        });
    }

    // mcp_agent_call — the .cy MCP server reaching the agent surface.
    {
        let jm = json_idx(vm)?;
        vm.bind_extern("mcp_agent_call", move |args| {
            let name = arg_str(args, 0)?;
            let params = args.get(1).and_then(json_val);
            match crate::bad_apple_ipc::call_agent(&name, params, 4096) {
                Ok(v) => Ok(json_to_val(&v, &jm)),
                Err(e) => Ok(json_to_val(
                    &serde_json::json!({ "error": e.to_string() }),
                    &jm,
                )),
            }
        });
    }

    // ── fs extras ──
    vm.bind_extern("fs_chmod", |args| {
        let path = arg_str(args, 0)?;
        let mode = arg_i64(args, 1)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            Ok(Val::B(
                std::fs::set_permissions(&path, std::fs::Permissions::from_mode(mode as u32))
                    .is_ok(),
            ))
        }
        #[cfg(not(unix))]
        {
            let _ = (path, mode);
            Ok(Val::B(false))
        }
    });
    vm.bind_extern("fs_unlink", |args| {
        Ok(Val::B(std::fs::remove_file(arg_str(args, 0)?).is_ok()))
    });
    vm.bind_extern("fs_remove_all", |args| {
        let p = arg_str(args, 0)?;
        let m = std::fs::metadata(&p);
        let ok = match m {
            Ok(m) if m.is_dir() => std::fs::remove_dir_all(&p).is_ok(),
            Ok(_) => std::fs::remove_file(&p).is_ok(),
            Err(_) => true, // already absent
        };
        Ok(Val::B(ok))
    });
    vm.bind_extern("fs_realpath", |args| {
        let p = arg_str(args, 0)?;
        match std::fs::canonicalize(&p) {
            Ok(r) => Ok(s(r.to_string_lossy().to_string())),
            Err(_) => Ok(s("")),
        }
    });
    vm.bind_extern("fs_is_symlink", |args| {
        let p = arg_str(args, 0)?;
        let ok = std::fs::symlink_metadata(&p)
            .map(|m| m.file_type().is_symlink())
            .unwrap_or(false);
        Ok(Val::B(ok))
    });
    vm.bind_extern("fs_is_dir", |args| {
        Ok(Val::B(
            std::fs::metadata(arg_str(args, 0)?)
                .map(|m| m.is_dir())
                .unwrap_or(false),
        ))
    });
    vm.bind_extern("time_parse_iso", |args| {
        let s_ = arg_str(args, 0)?;
        let ms = chrono::DateTime::parse_from_rfc3339(&s_)
            .map(|d| d.timestamp_millis())
            .unwrap_or(-1);
        Ok(Val::I(ms))
    });
    vm.bind_extern("sys_exe_dir", |_args| {
        let mut dir = std::env::current_exe()
            .ok()
            .and_then(|p| std::fs::canonicalize(&p).ok().or(Some(p)))
            .and_then(|p| p.parent().map(PathBuf::from))
            .unwrap_or_else(|| PathBuf::from("."));
        if dir.file_name().and_then(|n| n.to_str()) == Some("deps") {
            if let Some(parent) = dir.parent() {
                dir = parent.to_path_buf();
            }
        }
        Ok(s(dir.to_string_lossy().to_string()))
    });

    // ── ledger forensics ──
    vm.bind_extern("fs_read_from", |args| {
        let p = arg_str(args, 0)?;
        let off = arg_i64(args, 1)?.max(0) as u64;
        use std::io::{Read, Seek, SeekFrom};
        let mut f = match std::fs::File::open(&p) {
            Ok(f) => f,
            Err(_) => return Ok(s("")),
        };
        if f.seek(SeekFrom::Start(off)).is_err() {
            return Ok(s(""));
        }
        let mut buf = Vec::new();
        if f.read_to_end(&mut buf).is_err() {
            return Ok(s(""));
        }
        Ok(s(String::from_utf8_lossy(&buf).to_string()))
    });
    {
        let j11 = json_idx(vm)?;
        vm.bind_extern("ledger_verify_line", move |args| {
            let line = arg_str(args, 0)?;
            let secrets: Vec<Vec<u8>> = match &args[1] {
                Val::V(v) => v
                    .borrow()
                    .iter()
                    .filter_map(|x| match x {
                        Val::S(h) => Some(hex::decode(h.as_ref()).unwrap_or_default()),
                        _ => None,
                    })
                    .collect(),
                _ => vec![],
            };
            let tip = arg_str(args, 2)?;
            match sovereign_ledger::import::verify_badapple_line(&line, &secrets, &tip) {
                Ok(new_tip) => Ok(json_to_val(
                    &serde_json::json!({"ok": true, "tip": new_tip, "err": ""}),
                    &j11,
                )),
                Err(e) => Ok(json_to_val(
                    &serde_json::json!({"ok": false, "tip": tip, "err": e.to_string()}),
                    &j11,
                )),
            }
        });
    }
    vm.bind_extern("ledger_key_candidates", |args| {
        let p = arg_str(args, 0)?;
        let keys: Vec<Val> = std::fs::read(&p)
            .map(|raw| {
                sovereign_ledger::import::slicks_key_candidates(&raw)
                    .into_iter()
                    .map(|k| s(hex::encode(k)))
                    .collect()
            })
            .unwrap_or_default();
        Ok(Val::V(Rc::new(RefCell::new(keys))))
    });
    vm.bind_extern("causal_explain", |args| {
        let text = arg_str(args, 0)?;
        let graph = crate::production_blueprint::CausalGraph::bad_apple_default();
        let out = graph
            .find_broken_primitive(&text)
            .and_then(|p| graph.explain_failure(p))
            .unwrap_or_default();
        Ok(s(out))
    });
    vm.bind_extern("causal_relate", |args| {
        use crate::production_blueprint::CausalRelation;
        let from = arg_str(args, 0)?;
        let rel = arg_str(args, 1)?;
        let to = arg_str(args, 2)?;
        let r = match rel.as_str() {
            "depends_on" => CausalRelation::DependsOn,
            "protects" => CausalRelation::Protects,
            "causes" => CausalRelation::Causes,
            "violates" => CausalRelation::Violates,
            "enforces" => CausalRelation::Enforces,
            _ => CausalRelation::Enables,
        };
        // the causal graph is process-local in Rust too — keep a shared
        // static so causal_explain sees what .cy callers related
        static GRAPH: std::sync::OnceLock<
            std::sync::Mutex<crate::production_blueprint::CausalGraph>,
        > = std::sync::OnceLock::new();
        let g = GRAPH.get_or_init(|| {
            std::sync::Mutex::new(crate::production_blueprint::CausalGraph::bad_apple_default())
        });
        if let Ok(mut graph) = g.lock() {
            graph.add_relation(&from, r, &to);
        }
        Ok(Val::Unit)
    });

    // ── durable KV (redb) ──
    vm.bind_extern("kv_get", |args| {
        let db = arg_str(args, 0)?;
        let key = arg_str(args, 1)?;
        let out = crate::redb_kv::open(&db)
            .and_then(|d| crate::redb_kv::get(&d, key.as_bytes()))
            .ok()
            .flatten()
            .map(|b| String::from_utf8_lossy(&b).to_string())
            .unwrap_or_default();
        Ok(s(out))
    });
    vm.bind_extern("kv_put", |args| {
        let db = arg_str(args, 0)?;
        let key = arg_str(args, 1)?;
        let val = arg_str(args, 2)?;
        let ok = crate::redb_kv::open(&db)
            .and_then(|d| crate::redb_kv::insert(&d, key.as_bytes(), val.as_bytes()))
            .is_ok();
        Ok(Val::B(ok))
    });
    vm.bind_extern("kv_del", |args| {
        let db = arg_str(args, 0)?;
        let key = arg_str(args, 1)?;
        let ok = crate::redb_kv::open(&db)
            .and_then(|d| crate::redb_kv::remove(&d, key.as_bytes()))
            .is_ok();
        Ok(Val::B(ok))
    });
    {
        let j12 = json_idx(vm)?;
        vm.bind_extern("kv_list", move |args| {
            let db = arg_str(args, 0)?;
            let mut items: Vec<serde_json::Value> = Vec::new();
            if let Ok(d) = crate::redb_kv::open(&db) {
                if let Ok(pairs) = crate::redb_kv::iter(&d) {
                    for (k, v) in pairs {
                        items.push(serde_json::json!({
                            "key": String::from_utf8_lossy(&k),
                            "value": String::from_utf8_lossy(&v),
                        }));
                    }
                }
            }
            Ok(json_to_val(&serde_json::Value::Array(items), &j12))
        });
    }
    vm.bind_extern("time_stamp", |_args| {
        Ok(s(chrono::Utc::now()
            .format("%Y-%m-%d %H:%M:%S UTC")
            .to_string()))
    });

    // ── scavenger bridge ──
    {
        let j13 = json_idx(vm)?;
        vm.bind_extern("kv_scan_prefix", move |args| {
            let db = arg_str(args, 0)?;
            let prefix = arg_str(args, 1)?;
            let mut items: Vec<serde_json::Value> = Vec::new();
            if let Ok(d) = crate::redb_kv::open(&db) {
                if let Ok(pairs) = crate::redb_kv::scan_prefix(&d, prefix.as_bytes()) {
                    for (k, v) in pairs {
                        items.push(serde_json::json!({
                            "key": String::from_utf8_lossy(&k),
                            "value": String::from_utf8_lossy(&v),
                        }));
                    }
                }
            }
            Ok(json_to_val(&serde_json::Value::Array(items), &j13))
        });
    }
    vm.bind_extern("kv_count_prefix", |args| {
        let db = arg_str(args, 0)?;
        let prefix = arg_str(args, 1)?;
        let n = crate::redb_kv::open(&db)
            .and_then(|d| crate::redb_kv::count_prefix(&d, prefix.as_bytes()))
            .unwrap_or(0);
        Ok(Val::I(n as i64))
    });
    {
        let j14 = json_idx(vm)?;
        vm.bind_extern("fs_walk", move |args| {
            let root = PathBuf::from(arg_str(args, 0)?);
            let depth = arg_i64(args, 1)?.max(0) as usize;
            let mut files: Vec<serde_json::Value> = Vec::new();
            fn walk(d: &std::path::Path, depth: usize, out: &mut Vec<serde_json::Value>) {
                let Ok(rd) = std::fs::read_dir(d) else { return };
                for e in rd.flatten() {
                    let p = e.path();
                    let Ok(ft) = e.file_type() else { continue };
                    if ft.is_symlink() {
                        continue;
                    }
                    if ft.is_dir() && depth > 0 {
                        walk(&p, depth - 1, out);
                    } else if ft.is_file() {
                        if let Ok(m) = e.metadata() {
                            let mtime = m
                                .modified()
                                .ok()
                                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                                .map_or(0, |d| d.as_nanos() as u64);
                            out.push(serde_json::json!({
                                "path": p.to_string_lossy(),
                                "size": m.len(),
                                "mtime": mtime,
                            }));
                        }
                    }
                }
            }
            walk(&root, depth, &mut files);
            Ok(json_to_val(&serde_json::Value::Array(files), &j14))
        });
    }
    vm.bind_extern("fs_mtime", |args| {
        let n = std::fs::metadata(arg_str(args, 0)?)
            .ok()
            .and_then(|m| m.modified().ok())
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map_or(0, |d| d.as_nanos() as i64);
        Ok(Val::I(n))
    });
    vm.bind_extern("fs_read_nofollow", |args| {
        let p = arg_str(args, 0)?;
        let cap = arg_i64(args, 1)?.max(0) as usize;
        #[cfg(unix)]
        {
            use std::io::Read;
            use std::os::unix::fs::OpenOptionsExt;
            let f = std::fs::OpenOptions::new()
                .read(true)
                .custom_flags(libc::O_NOFOLLOW)
                .open(&p);
            let Ok(file) = f else { return Ok(s("")) };
            let Ok(meta) = file.metadata() else {
                return Ok(s(""));
            };
            if !meta.file_type().is_file() {
                return Ok(s(""));
            }
            let mut reader = std::io::BufReader::new(file);
            let mut buf = vec![0u8; cap + 1];
            let mut filled = 0usize;
            while filled < buf.len() {
                match reader.read(&mut buf[filled..]) {
                    Ok(0) | Err(_) => break,
                    Ok(n) => filled += n,
                }
            }
            buf.truncate(filled.min(cap));
            let mut bl = buf.len();
            while bl > 0 && std::str::from_utf8(&buf[..bl]).is_err() {
                bl -= 1;
            }
            buf.truncate(bl);
            Ok(s(String::from_utf8_lossy(&buf).to_string()))
        }
        #[cfg(not(unix))]
        {
            let _ = (p, cap);
            Ok(s(""))
        }
    });
    {
        let j15 = json_idx(vm)?;
        vm.bind_extern("scav_embed", move |args| {
            let text = arg_str(args, 0)?;
            let v = crate::tensor_brain::semantic_embedding(&text);
            Ok(json_to_val(
                &serde_json::json!({"ok": !v.is_empty(), "vec": v}),
                &j15,
            ))
        });
    }
    vm.bind_extern("math_sqrt", |args| {
        let v = match args.first() {
            Some(Val::F(f)) => *f,
            Some(Val::I(i)) => *i as f64,
            _ => 0.0,
        };
        Ok(Val::F(v.sqrt()))
    });
    vm.bind_extern("f32_bits", |args| {
        let v = match args.first() {
            Some(Val::F(f)) => *f,
            Some(Val::I(i)) => *i as f64,
            _ => 0.0,
        };
        Ok(Val::I((v as f32).to_bits() as i64))
    });
    vm.bind_extern("f32_sqrt", |args| {
        let v = match args.first() {
            Some(Val::F(f)) => *f,
            Some(Val::I(i)) => *i as f64,
            _ => 0.0,
        };
        Ok(Val::F((v as f32).sqrt() as f64))
    });

    vm.bind_extern("f32_round", |args| {
        let v = match args.first() {
            Some(Val::F(f)) => *f,
            Some(Val::I(i)) => *i as f64,
            _ => 0.0,
        };
        Ok(Val::F((v as f32) as f64))
    });

    vm.bind_extern("f32_from_bits", |args| {
        let b = arg_i64(args, 0)? as u32;
        Ok(Val::F(f32::from_bits(b) as f64))
    });

    // ── sentinel bridge: the resident AneCore engine call. Verdict
    // parsing, prompt building and fail semantics live in sentinel.cy.
    {
        let j10 = json_idx(vm)?;
        vm.bind_extern("sentinel_generate", move |args| {
            let prompt = arg_str(args, 0)?;
            let system = arg_str(args, 1)?;
            let max_tokens = arg_i64(args, 2)?.max(1) as usize;
            let max_ctx = arg_i64(args, 3)?.max(1) as usize;
            let (status, text) =
                crate::sentinel::generate_raw(&prompt, &system, max_tokens, max_ctx);
            let st = match status {
                0 => "unavailable",
                1 => "ok",
                _ => "err",
            };
            Ok(json_to_val(
                &serde_json::json!({"status": st, "text": text}),
                &j10,
            ))
        });
    }
    vm.bind_extern("sentinel_tally", |args| {
        crate::sentinel::tally(&arg_str(args, 0)?, arg_i64(args, 1)?.max(0) as u64);
        Ok(Val::Unit)
    });

    // ── string extras ──
    vm.bind_extern("str_lower", |args| Ok(s(arg_str(args, 0)?.to_lowercase())));
    {
        let j = json_idx(vm)?;
        vm.bind_extern("str_parse_f64", move |args| {
            let t = arg_str(args, 0)?;
            Ok(match t.trim().parse::<f64>() {
                Ok(v) => Val::En(j.en, j.num, Rc::new(vec![Val::F(v)])),
                Err(_) => jnull(&j),
            })
        });
    }
    vm.bind_extern("rss_mb", |_| {
        use sysinfo::System;
        let mut sys = System::new_all();
        sys.refresh_all();
        let mb = sys
            .process(sysinfo::Pid::from_u32(std::process::id()))
            .map(|p| p.memory() as f64 / 1024.0 / 1024.0)
            .unwrap_or(0.0);
        Ok(Val::F(mb))
    });
    vm.bind_extern("str_upper", |args| Ok(s(arg_str(args, 0)?.to_uppercase())));

    // ── coyote self-bridge: compile a DFA pattern set into the ANE organ
    // binaries under `out_dir` — firewall.mlmodelc. Runs the language's own
    // compiler backend; the caller is a .cy program so `cap fs` applies.
    {
        let j9 = json_idx(vm)?;
        vm.bind_extern("dfa_ane_compile", move |args| {
            let pj = json_val(args.first().ok_or("dfa_ane_compile: patterns")?)
                .unwrap_or(serde_json::Value::Null);
            let dir = arg_str(args, 1)?;
            let pats: Vec<String> = pj
                .as_array()
                .map(|a| {
                    a.iter()
                        .filter_map(|x| x.as_str().map(String::from))
                        .collect()
                })
                .unwrap_or_default();
            let out = PathBuf::from(&dir);
            let _ = std::fs::create_dir_all(&out);
            let states = coyote::vm::DfaMachine::compile(pats).out.len();
            let body = match crate::coyote_organs::compile_to_dir(&out) {
                Ok((_progs, _ane)) => {
                    serde_json::json!({"states": states, "ok": true, "err": ""})
                }
                Err(e) => {
                    serde_json::json!({"states": states, "ok": false, "err": format!("{e:#}")})
                }
            };
            Ok(json_to_val(&body, &j9))
        });
    }
    // constant-time proof check — hmac_sha256b + verify_slice semantics
    vm.bind_extern("hmac_verify", |args| {
        let key = hex::decode(arg_str(args, 0)?).unwrap_or_default();
        let msg = hex::decode(arg_str(args, 1)?).unwrap_or_default();
        let proof = hex::decode(arg_str(args, 2)?).unwrap_or_default();
        use hmac::Mac as _;
        let Ok(mut mac) = <hmac::Hmac<sha2::Sha256> as hmac::KeyInit>::new_from_slice(&key) else {
            return Ok(Val::B(false));
        };
        mac.update(&msg);
        Ok(Val::B(mac.verify_slice(&proof).is_ok()))
    });

    Ok(())
}

/// Run the Coyote cert suite (`coyote/cert.cy`) — returns
/// (name, passed, message) per check, matching cert::run()'s shape so the
/// CLI/report code paths don't change.
pub fn run_cert() -> anyhow::Result<Vec<(String, bool, String)>> {
    let mut vm = cy_module_vm("cert.cy")?;
    let out = vm
        .call_fn("run", vec![])
        .map_err(|e| anyhow::anyhow!("cert run() failed: {e}"))?;
    let Val::V(items) = out else {
        anyhow::bail!("cert run() did not return a vec");
    };
    let mut results = Vec::new();
    for item in items.borrow().iter() {
        let Val::En(_, _, p) = item else { continue };
        let Some(Val::M(m)) = p.first() else { continue };
        let m = m.borrow();
        let get_str = |k: &str| -> String {
            m.iter()
                .find(|(kk, _)| matches!(kk, Val::S(s) if &**s == k))
                .and_then(|(_, v)| {
                    if let Val::En(_, _, pp) = v {
                        if let Some(Val::S(x)) = pp.first() {
                            return Some(x.to_string());
                        }
                    }
                    None
                })
                .unwrap_or_default()
        };
        let passed = m
            .iter()
            .find(|(kk, _)| matches!(kk, Val::S(s) if &**s == "passed"))
            .and_then(|(_, v)| {
                if let Val::En(_, _, pp) = v {
                    if let Some(Val::B(b)) = pp.first() {
                        return Some(*b);
                    }
                }
                None
            })
            .unwrap_or(false);
        results.push((get_str("name"), passed, get_str("message")));
    }
    Ok(results)
}

/// A VM loaded with a named embedded .cy module (plus everything it
/// imports), externs bound.
fn cy_module_vm(entry: &str) -> anyhow::Result<Vm> {
    let dir = std::env::temp_dir().join(format!("badapple_cy_{}", entry.replace('.', "_")));
    std::fs::create_dir_all(&dir)?;
    for (name, text) in [
        ("sys.cy", SYS_CY),
        ("strlib.cy", STRLIB_CY),
        ("ledger.cy", LEDGER_CY),
        ("slicks.cy", SLICKS_CY),
        ("cage.cy", CAGE_CY),
        ("mesh_brain.cy", MESH_BRAIN_CY),
        ("cert.cy", CERT_CY),
        ("protocol.cy", PROTOCOL_CY),
        ("p2p.cy", P2P_CY),
        ("org_policy.cy", ORG_POLICY_CY),
        ("config.cy", CONFIG_CY),
        ("mesh_sync.cy", MESH_SYNC_CY),
        ("firewall.cy", FIREWALL_CY),
        ("sentinel.cy", SENTINEL_CY),
        ("ify.cy", IFY_CY),
        ("strategy.cy", STRATEGY_CY),
        ("scavenger.cy", SCAVENGER_CY),
        ("vault.cy", VAULT_CY),
        ("blueprint.cy", BLUEPRINT_CY),
        ("watcher.cy", WATCHER_CY),
        ("mcp.cy", MCP_CY),
        ("benchmark.cy", BENCHMARK_CY),
        ("cli.cy", CLI_CY),
        ("mil.cy", MIL_CY),
        ("layer.cy", LAYER_CY),
    ] {
        write_embedded_cy(&dir, name, text).map_err(anyhow::Error::msg)?;
    }
    let build = dir.join("build");
    let mut vm = coyote::load_vm(&dir.join(entry), &build, true, false, true)
        .map_err(|e| anyhow::anyhow!("{entry} failed to load: {e}"))?;
    bind_all(&mut vm).map_err(anyhow::Error::msg)?;
    Ok(vm)
}

/// cli.cy dispatch — the organism's own front door. The `badapple` binary
/// consults this before its legacy handlers: `Some((code, out, err))` means
/// cli.cy handled it (print out/err, exit code); `None` means fall through.
pub fn cli_dispatch(args: &[String]) -> Option<(i64, String, String)> {
    let mut vm = cy_module_vm("cli.cy").ok()?;
    let argv = Val::V(Rc::new(RefCell::new(
        args.iter().map(|a| Val::S(Rc::from(a.as_str()))).collect(),
    )));
    let r = vm.call_fn("cli_dispatch", vec![argv]).ok()?;
    let v = json_val(&r)?;
    if !v.get("handled").and_then(|h| h.as_bool()).unwrap_or(false) {
        return None;
    }
    let code = v.get("code").and_then(|c| c.as_i64()).unwrap_or(0);
    let out = v.get("out").and_then(|x| x.as_str()).unwrap_or("");
    let err = v.get("err").and_then(|x| x.as_str()).unwrap_or("");
    Some((code, out.to_string(), err.to_string()))
}

/// layer.cy emit — the Coyote-authored layer-shard graph, lowered through
/// Shared emit-driver for the layer.cy spec fns — calls `f` with the
/// given int args appended after (cfg, dir), expects {spec,weights} hex.
fn mil_cy_emit_call(
    f: &str,
    cfg: &serde_json::Value,
    dir: &str,
    args: &[i64],
    s_is_prefill: Option<(i64, bool)>,
) -> Result<(Vec<u8>, Vec<u8>), String> {
    let mut vm = cy_module_vm("layer.cy").map_err(|e| e.to_string())?;
    let j = json_idx(&vm)?;
    let mut call_args = vec![json_to_val(cfg, &j), Val::S(Rc::from(dir))];
    for a in args {
        call_args.push(Val::I(*a));
    }
    if let Some((s, pf)) = s_is_prefill {
        call_args.push(Val::I(s));
        call_args.push(Val::B(pf));
    }
    let r = vm
        .call_fn(f, call_args)
        .map_err(|e| format!("{f} call: {e}"))?;
    let v = val_to_json(&r, &j)?;
    if let Some(err) = v.get("error").and_then(|e| e.as_str()) {
        if !err.is_empty() {
            return Err(format!("{f}: {err}"));
        }
    }
    let spec_hex = v
        .get("spec")
        .and_then(|x| x.as_str())
        .ok_or_else(|| format!("{f}: no spec in {v:?}"))?;
    let w_hex = v
        .get("weights")
        .and_then(|x| x.as_str())
        .ok_or_else(|| format!("{f}: no weights in {v:?}"))?;
    let spec = hex::decode(spec_hex).map_err(|e| format!("spec hex: {e}"))?;
    let weights = hex::decode(w_hex).map_err(|e| format!("weights hex: {e}"))?;
    Ok((spec, weights))
}

/// layer.cy emit — the Coyote-authored layer-shard graph, lowered through
/// `mil_emit` to (spec_bytes, weight_bin_bytes). `cfg` is the JSON mirror
/// of the converter's Cfg plus the debug-flag keys (cut_at, split_gu, ...).
/// Errors return in-band as `{"error": ...}` from the .cy side.
pub fn mil_layer_emit(
    cfg: &serde_json::Value,
    dir: &str,
    ls: i64,
    le: i64,
    seq: i64,
    s: i64,
    is_prefill: bool,
) -> Result<(Vec<u8>, Vec<u8>), String> {
    mil_cy_emit_call(
        "mil_layer_emit",
        cfg,
        dir,
        &[ls, le, seq],
        Some((s, is_prefill)),
    )
}

/// layer.cy split-FFN shard (per-layer stateless tail).
pub fn mil_ffn_emit(
    cfg: &serde_json::Value,
    dir: &str,
    ls: i64,
    le: i64,
    s: i64,
) -> Result<(Vec<u8>, Vec<u8>), String> {
    mil_cy_emit_call("mil_ffn_emit", cfg, dir, &[ls, le, s], None)
}

/// layer.cy LM-head shard (final norm + sliced projection).
pub fn mil_head_emit(
    cfg: &serde_json::Value,
    dir: &str,
    vstart: i64,
    vend: i64,
) -> Result<(Vec<u8>, Vec<u8>), String> {
    mil_cy_emit_call("mil_head_emit", cfg, dir, &[vstart, vend], None)
}

/// layer.cy probe package (qwen3 numeric bisection; stage = an|xw|muls|
/// adds|id|xw4|anr|q|v|qn|kn|kv|rq|rk).
pub fn mil_probe_emit(
    cfg: &serde_json::Value,
    dir: &str,
    ls: i64,
    stage: &str,
) -> Result<(Vec<u8>, Vec<u8>), String> {
    let mut vm = cy_module_vm("layer.cy").map_err(|e| e.to_string())?;
    let j = json_idx(&vm)?;
    let r = vm
        .call_fn(
            "mil_probe_emit",
            vec![
                json_to_val(cfg, &j),
                Val::S(Rc::from(dir)),
                Val::I(ls),
                Val::S(Rc::from(stage)),
            ],
        )
        .map_err(|e| format!("mil_probe_emit call: {e}"))?;
    let v = val_to_json(&r, &j)?;
    if let Some(err) = v.get("error").and_then(|e| e.as_str()) {
        if !err.is_empty() {
            return Err(format!("mil_probe_emit: {err}"));
        }
    }
    let spec_hex = v
        .get("spec")
        .and_then(|x| x.as_str())
        .ok_or_else(|| format!("mil_probe_emit: no spec in {v:?}"))?;
    let w_hex = v
        .get("weights")
        .and_then(|x| x.as_str())
        .ok_or_else(|| format!("mil_probe_emit: no weights in {v:?}"))?;
    let spec = hex::decode(spec_hex).map_err(|e| format!("spec hex: {e}"))?;
    let weights = hex::decode(w_hex).map_err(|e| format!("weights hex: {e}"))?;
    Ok((spec, weights))
}

/// layer.cy full convert driver — embedding + all shard loops + manifest,
/// all authored and executed inside the Coyote VM. Returns the final
/// manifest object; `all_ok` field carries the exit-code signal.
pub fn mil_convert_run(
    cfg: &serde_json::Value,
    dir: &str,
    out_dir: &str,
    opts: &serde_json::Value,
) -> Result<serde_json::Value, String> {
    let mut vm = cy_module_vm("layer.cy").map_err(|e| e.to_string())?;
    let j = json_idx(&vm)?;
    let r = vm
        .call_fn(
            "mil_convert_run",
            vec![
                json_to_val(cfg, &j),
                Val::S(Rc::from(dir)),
                Val::S(Rc::from(out_dir)),
                json_to_val(opts, &j),
            ],
        )
        .map_err(|e| format!("mil_convert_run call: {e}"))?;
    let v = val_to_json(&r, &j)?;
    if let Some(err) = v.get("error").and_then(|e| e.as_str()) {
        if !err.is_empty() {
            return Err(format!("mil_convert_run: {err}"));
        }
    }
    Ok(v)
}

/// A Json enum value → serde_json::Value (Obj/Arr/Str/Num/Bool/Null).
fn json_val(v: &Val) -> Option<serde_json::Value> {
    let Val::En(_, _, p) = v else { return None };
    if p.is_empty() {
        return Some(serde_json::Value::Null);
    }
    let inner = p.first()?;
    Some(match inner {
        Val::B(b) => serde_json::Value::Bool(*b),
        Val::F(n) => serde_json::json!(n),
        Val::S(s) => serde_json::Value::String(s.to_string()),
        Val::V(a) => serde_json::Value::Array(a.borrow().iter().filter_map(json_val).collect()),
        Val::M(m) => serde_json::Value::Object(
            m.borrow()
                .iter()
                .filter_map(|(k, v)| {
                    if let Val::S(k) = k {
                        json_val(v).map(|v| (k.to_string(), v))
                    } else {
                        None
                    }
                })
                .collect(),
        ),
        _ => return None,
    })
}

/// `p2p.cy::infer` — the delegated-inference client in Coyote. Returns
/// (text, tier, elapsed_ms) matching `svc.infer(...)`'s response shape.
pub fn p2p_infer(
    peer: &str,
    prompt: &str,
    max_tokens: i64,
    from_peer: &str,
) -> anyhow::Result<(String, Option<String>, u64)> {
    let mut vm = cy_module_vm("p2p.cy")?;
    let r = vm
        .call_fn(
            "infer",
            vec![s(peer), s(prompt), Val::I(max_tokens), s(from_peer)],
        )
        .map_err(|e| anyhow::anyhow!("p2p infer failed: {e}"))?;
    let j = json_val(&r).unwrap_or(serde_json::Value::Null);
    if j.get("status").and_then(|s| s.as_str()) == Some("ok") {
        Ok((
            j["text"].as_str().unwrap_or_default().to_string(),
            j["tier"].as_str().map(String::from),
            j["elapsed_ms"].as_f64().unwrap_or(0.0) as u64,
        ))
    } else {
        anyhow::bail!("{}", j["message"].as_str().unwrap_or("p2p infer failed"))
    }
}

/// `p2p.cy::serve` — the serving loop. Blocking forever; call on a thread.
/// Handles delegated inference plus model pull/push chunk transfers.
pub fn p2p_serve(port: i64, model_dir: &str) -> anyhow::Result<()> {
    let mut vm = cy_module_vm("p2p.cy")?;
    vm.call_fn("serve", vec![Val::I(port), s(model_dir)])
        .map_err(|e| anyhow::anyhow!("p2p serve failed: {e}"))?;
    Ok(())
}

// ── mil_spec bridge — Coyote authors MIL programs ───────────────────────
// `mil_emit(spec_json)` takes a Json program description and returns the
// encoded model spec bytes as hex; `mil_write_package` writes the
// .mlpackage. The converter logic — which ops, which shapes, which shard
// layout — lives in aneconvert.cy; this bridge is pure data plumbing.
//
// Json schema:
//   { inputs/outputs/states: [{name,dtype,shape:[i64..]}],
//     fn_inputs: [{name,dtype,shape}],            (states double as
//                                                  stateType features)
//     meta: {spec_version,opset,creator,description,user:{k:v}},
//     ops: [{ty,
//            in: {arg_name: "bind" | ["v1","v2"] | {"c": <value>}},
//            out: [{name,dtype,shape}],
//            attr: {name: <value>}}],
//     block_outputs: [name...] }
// <value>: {"s":str} | {"i32":n} | {"i32s":[..]} | {"i64s":[..]} |
//          {"bools":[..]} | {"f16":x} | {"f32":x} |
//          {"f16s":{"shape":[..],"hex":".."}} |
//          {"bytes":{"dtype":"fp16","shape":[..],"hex":".."}} |
//          {"blob":{"dtype":"fp16","shape":[..],"off":N}}

fn mil_dtype(s: &str) -> Result<mil_spec::DType, String> {
    use mil_spec::DType;
    Ok(match s {
        "fp16" | "float16" => DType::Fp16,
        "fp32" | "float32" => DType::Fp32,
        "int32" | "i32" => DType::Int32,
        "int64" | "i64" => DType::Int64,
        "int8" | "i8" => DType::Int8,
        "int4" => DType::Int4,
        "uint4" => DType::Uint4,
        "bool" => DType::Bool,
        "str" | "string" => DType::Str,
        other => return Err(format!("mil: unknown dtype {other}")),
    })
}

fn mil_int_ok(v: &serde_json::Value) -> Option<i64> {
    v.as_i64().or_else(|| {
        v.as_f64()
            .filter(|f| f.fract() == 0.0 && f.is_finite())
            .map(|f| f as i64)
    })
}

fn mil_int(v: &serde_json::Value) -> Result<i64, String> {
    v.as_i64()
        .or_else(|| {
            v.as_f64()
                .filter(|f| f.fract() == 0.0 && f.is_finite())
                .map(|f| f as i64)
        })
        .ok_or_else(|| "mil: expected int".to_string())
}

fn mil_shape(v: &serde_json::Value) -> Result<Vec<i64>, String> {
    v.as_array()
        .ok_or_else(|| "mil: shape must be an array".to_string())?
        .iter()
        .map(|x| {
            x.as_i64()
                .or_else(|| mil_int(x).ok())
                .ok_or_else(|| "mil: shape dim not an int".to_string())
        })
        .collect()
}

fn mil_value(
    v: &serde_json::Value,
    woff: &std::collections::HashMap<String, u64>,
) -> Result<mil_spec::Value, String> {
    use mil_spec::{DType, TensorType, Value, ValueType};
    let o = v
        .as_object()
        .ok_or_else(|| "mil: value must be an object".to_string())?;
    let tt = |dt: DType, shape: &[i64]| {
        ValueType::Tensor(TensorType {
            dtype: dt,
            shape: shape.to_vec(),
        })
    };
    if let Some(s) = o.get("s").and_then(|x| x.as_str()) {
        return Ok(Value::Str(s.to_string()));
    }
    if let Some(n) = o.get("i32").and_then(mil_int_ok) {
        return Ok(Value::i32_1(n as i32));
    }
    if let Some(x) = o.get("i32_scalar").and_then(mil_int_ok) {
        return Ok(Value::Imm(
            ValueType::Tensor(TensorType {
                dtype: DType::Int32,
                shape: vec![],
            }),
            mil_spec::Immediate::Ints(vec![x as i32]),
        ));
    }
    if let Some(b) = o.get("bool1").and_then(|x| x.as_bool()) {
        return Ok(Value::Imm(
            ValueType::Tensor(TensorType {
                dtype: DType::Bool,
                shape: vec![],
            }),
            mil_spec::Immediate::Bools(vec![b]),
        ));
    }
    if let Some(a) = o.get("i32s").and_then(|x| x.as_array()) {
        let vs: Vec<i32> = a
            .iter()
            .map(|x| mil_int_ok(x).unwrap_or(0) as i32)
            .collect();
        return Ok(Value::i32s(&vs));
    }
    if let Some(a) = o.get("i64s").and_then(|x| x.as_array()) {
        let vs: Vec<i64> = a.iter().map(|x| mil_int_ok(x).unwrap_or(0)).collect();
        return Ok(Value::Imm(
            tt(DType::Int64, &[vs.len() as i64]),
            mil_spec::Immediate::LongInts(vs),
        ));
    }
    if let Some(a) = o.get("bools").and_then(|x| x.as_array()) {
        let vs: Vec<bool> = a.iter().map(|x| x.as_bool().unwrap_or(false)).collect();
        return Ok(Value::bools(&vs));
    }
    if let Some(x) = o.get("f16").and_then(|x| x.as_f64()) {
        return Ok(Value::f16_scalar(x as f32));
    }
    if let Some(x) = o.get("f32").and_then(|x| x.as_f64()) {
        return Ok(Value::f32_scalar(x as f32));
    }
    if let Some(b) = o.get("f16s").or_else(|| o.get("bytes")) {
        let shape = mil_shape(&b["shape"])?;
        let hex = b["hex"].as_str().unwrap_or("");
        let raw = hex::decode(hex).map_err(|e| format!("mil: bad hex: {e}"))?;
        let dt = mil_dtype(b["dtype"].as_str().unwrap_or("fp16"))?;
        return Ok(Value::Bytes(tt(dt, &shape), raw));
    }
    if let Some(b) = o.get("blob_named") {
        let name = b["name"]
            .as_str()
            .ok_or_else(|| "mil: blob_named missing name".to_string())?;
        let off = woff
            .get(name)
            .ok_or_else(|| format!("mil: weight {name} not in weights table"))?;
        let shape = mil_shape(&b["shape"])?;
        let dt = mil_dtype(b["dtype"].as_str().unwrap_or("fp16"))?;
        return Ok(Value::Blob(
            ValueType::Tensor(TensorType { dtype: dt, shape }),
            "@model_path/weights/weight.bin".to_string(),
            *off,
        ));
    }
    if let Some(b) = o.get("blob") {
        let shape = mil_shape(&b["shape"])?;
        let dt = mil_dtype(b["dtype"].as_str().unwrap_or("fp16"))?;
        let off = b["off"].as_u64().unwrap_or(0);
        return Ok(Value::Blob(
            tt(dt, &shape),
            "weights/weight.bin".to_string(),
            off,
        ));
    }
    Err("mil: unrecognized value spec".to_string())
}

fn mil_arg(
    v: &serde_json::Value,
    woff: &std::collections::HashMap<String, u64>,
) -> Result<mil_spec::Argument, String> {
    use mil_spec::{bind, bind_const, bind_many};
    match v {
        serde_json::Value::String(n) => Ok(bind(n).1),
        serde_json::Value::Array(a) => {
            let names: Vec<&str> = a.iter().filter_map(|x| x.as_str()).collect();
            Ok(bind_many(&names))
        }
        serde_json::Value::Object(o) => {
            if let Some(c) = o.get("c") {
                Ok(bind_const(mil_value(c, &woff)?))
            } else {
                Err("mil: arg object needs 'c' const".into())
            }
        }
        _ => Err("mil: bad arg".into()),
    }
}

fn mil_nvt(o: &serde_json::Value) -> Result<mil_spec::NVT, String> {
    use mil_spec::{TensorType, ValueType, NVT};
    Ok(NVT {
        name: o["name"]
            .as_str()
            .ok_or_else(|| "mil: nvt missing name".to_string())?
            .to_string(),
        ty: if o["state"].as_bool().unwrap_or(false) {
            ValueType::State(TensorType {
                dtype: mil_dtype(o["dtype"].as_str().unwrap_or("fp16"))?,
                shape: mil_shape(&o["shape"])?,
            })
        } else {
            ValueType::Tensor(TensorType {
                dtype: mil_dtype(o["dtype"].as_str().unwrap_or("fp16"))?,
                shape: mil_shape(&o["shape"])?,
            })
        },
    })
}

fn mil_feature(o: &serde_json::Value, is_state: bool) -> Result<mil_spec::Feature, String> {
    Ok(mil_spec::Feature {
        name: o["name"]
            .as_str()
            .ok_or_else(|| "mil: feature missing name".to_string())?
            .to_string(),
        dtype: mil_dtype(o["dtype"].as_str().unwrap_or("fp16"))?,
        shape: mil_shape(&o["shape"])?,
        is_state,
    })
}

fn mil_features(v: &serde_json::Value, is_state: bool) -> Result<Vec<mil_spec::Feature>, String> {
    v.as_array()
        .map(|a| {
            a.iter()
                .map(|x| mil_feature(x, is_state))
                .collect::<Result<Vec<_>, _>>()
        })
        .unwrap_or_else(|| Ok(Vec::new()))
        .map_err(|e: String| e)
}

/// Lower a Json program spec through mil_spec → (encoded model, weight.bin).
/// `spec.weights` is an ordered [{name,dtype,shape,hex}] table; `blob_named`
/// value refs resolve to 64-byte-aligned offsets assigned in spec order.
fn mil_emit_json(spec: &serde_json::Value) -> Result<(Vec<u8>, Vec<u8>), String> {
    use mil_spec::{encode_model, Block, ModelMeta, WeightBin};
    let mut wbin = WeightBin::new();
    let mut woff: std::collections::HashMap<String, u64> = std::collections::HashMap::new();
    if let Some(ws) = spec["weights"].as_array() {
        for w in ws {
            let name = w["name"]
                .as_str()
                .ok_or_else(|| "mil: weight missing name".to_string())?;
            let raw = hex::decode(w["hex"].as_str().unwrap_or(""))
                .map_err(|e| format!("mil: weight {name} bad hex: {e}"))?;
            let shape = mil_shape(&w["shape"])?;
            let dt = mil_dtype(w["dtype"].as_str().unwrap_or("fp16"))?;
            woff.insert(name.to_string(), wbin.put(name, dt, &shape, &raw));
        }
    }
    let inputs = mil_features(&spec["inputs"], false)?;
    let outputs = mil_features(&spec["outputs"], false)?;
    let states = mil_features(&spec["states"], true)?;
    let mut fn_inputs = Vec::new();
    for x in spec["fn_inputs"].as_array().into_iter().flatten() {
        fn_inputs.push(mil_nvt(x)?);
    }
    let mo = &spec["meta"];
    let mut meta = ModelMeta::new(
        mo["spec_version"].as_f64().unwrap_or(9.0) as i32,
        mo["opset"].as_str().unwrap_or("CoreML9"),
    )
    .creator(mo["creator"].as_str().unwrap_or("badapple-coyote"))
    .description(mo["description"].as_str().unwrap_or("authored by Coyote"));
    if let Some(u) = mo["user"].as_object() {
        for (k, v) in u {
            meta = meta.user_meta(k, v.as_str().unwrap_or(""));
        }
    }
    let mut blk = Block::new();
    for op_j in spec["ops"].as_array().into_iter().flatten() {
        let ty = op_j["ty"]
            .as_str()
            .ok_or_else(|| "mil: op missing ty".to_string())?;
        let mut inputs = Vec::new();
        if let Some(io) = op_j["in"].as_object() {
            for (k, v) in io {
                inputs.push((k.clone(), mil_arg(v, &woff)?));
            }
        }
        let mut outs: Vec<(&str, mil_spec::ValueType)> = Vec::new();
        for o in op_j["out"].as_array().into_iter().flatten() {
            let name = o["name"]
                .as_str()
                .ok_or_else(|| "mil: op out missing name".to_string())?;
            outs.push((
                name,
                mil_spec::ValueType::Tensor(mil_spec::TensorType {
                    dtype: mil_dtype(o["dtype"].as_str().unwrap_or("fp16"))?,
                    shape: mil_shape(&o["shape"])?,
                }),
            ));
        }
        let mut attrs = Vec::new();
        if let Some(ao) = op_j["attr"].as_object() {
            for (k, v) in ao {
                attrs.push((k.clone(), mil_value(v, &woff)?));
            }
        }
        blk.op(ty, inputs, outs, attrs);
    }
    for o in spec["block_outputs"].as_array().into_iter().flatten() {
        if let Some(n) = o.as_str() {
            blk.outputs.push(n.to_string());
        }
    }
    let spec_bytes = encode_model(&inputs, &outputs, &states, &blk, &fn_inputs, &meta);
    Ok((spec_bytes, wbin.finish()))
}

/// ciborium::value::Value → serde_json::Value (bytes → hex str, ints → f64
/// when they overflow i64, tags → inner).
fn cborval_to_json(v: &ciborium::value::Value) -> serde_json::Value {
    use ciborium::value::Value as C;
    match v {
        C::Integer(i) => {
            let n: i128 = (*i).into();
            if let Ok(i64v) = i64::try_from(n) {
                serde_json::json!(i64v)
            } else {
                serde_json::json!(n as f64)
            }
        }
        // Byte strings go through the small-int-array path so
        // json_to_val collapses them to Json::Bytes like everything else.
        C::Bytes(b) => serde_json::Value::Array(b.iter().map(|x| serde_json::json!(*x)).collect()),
        C::Float(f) => serde_json::json!(f),
        C::Text(t) => serde_json::Value::String(t.clone()),
        C::Bool(b) => serde_json::Value::Bool(*b),
        C::Null => serde_json::Value::Null,
        C::Tag(_, inner) => cborval_to_json(inner),
        C::Array(a) => serde_json::Value::Array(a.iter().map(cborval_to_json).collect()),
        C::Map(m) => {
            let mut map = serde_json::Map::new();
            for (k, x) in m {
                let key = match k {
                    C::Text(t) => t.clone(),
                    other => cborval_to_json(other).to_string(),
                };
                map.insert(key, cborval_to_json(x));
            }
            serde_json::Value::Object(map)
        }
        _ => serde_json::Value::Null,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn protocol_vm() -> Vm {
        let src = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("coyote/protocol.cy");
        let build = std::env::temp_dir().join("badapple_coyote_sys_test");
        let mut vm =
            coyote::load_vm(&src, &build, true, false, true).expect("coyote load protocol.cy");
        bind_all(&mut vm).expect("bind externs");
        vm
    }

    fn str_arg(v: &str) -> Val {
        Val::S(Rc::from(v))
    }

    fn packet_fields(v: &Val) -> (String, String, String) {
        let Val::St(_, fields) = v else {
            panic!("expected SignedPacket struct");
        };
        let f = fields.borrow();
        let mut out = f.iter().map(|x| match x {
            Val::S(s) => s.to_string(),
            _ => panic!("non-str field"),
        });
        (
            out.next().unwrap(),
            out.next().unwrap(),
            out.next().unwrap(),
        )
    }

    #[test]
    fn protocol_sign_verify_parity_with_rust() {
        let mut vm = protocol_vm();
        let signed = vm
            .call_fn(
                "sign_packet",
                vec![
                    str_arg("alice"),
                    str_arg("hello organism"),
                    str_arg("topsecret"),
                ],
            )
            .expect("sign_packet");
        let (sender, payload_b64, signature_hex) = packet_fields(&signed);
        assert_eq!(sender, "alice");
        // parity: identical signature to the Rust implementation
        let rust = crate::protocol::sign_packet("alice", b"hello organism", b"topsecret");
        assert_eq!(payload_b64, rust.payload_b64);
        assert_eq!(signature_hex, rust.signature_hex);

        let ok = vm
            .call_fn("verify_packet", vec![signed, str_arg("topsecret")])
            .expect("verify_packet");
        assert!(matches!(ok, Val::B(true)));

        let signed2 = vm
            .call_fn(
                "sign_packet",
                vec![
                    str_arg("alice"),
                    str_arg("hello organism"),
                    str_arg("topsecret"),
                ],
            )
            .expect("sign_packet 2");
        let bad = vm
            .call_fn("verify_packet", vec![signed2, str_arg("wrongkey")])
            .expect("verify_packet wrong key");
        assert!(matches!(bad, Val::B(false)));
    }

    #[test]
    fn json_roundtrip_through_externs() {
        // exercise json externs via a .cy roundtrip: parse → emit
        let build = std::env::temp_dir().join("badapple_coyote_sys_json");
        let src = build.join("jt.cy");
        std::fs::create_dir_all(&build).unwrap();
        std::fs::write(
            &src,
            r#"
import "sys.cy"
host fn rt(s: str) -> str { return json_emit(json_parse(s)); }
"#,
        )
        .unwrap();
        // sys.cy resolves relative to the importing file — copy it in.
        std::fs::copy(
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("coyote/sys.cy"),
            build.join("sys.cy"),
        )
        .unwrap();
        let mut vm = coyote::load_vm(&src, &build, true, false, true).expect("load jt.cy");
        bind_all(&mut vm).expect("bind externs");
        let out = vm
            .call_fn("rt", vec![str_arg(r#"{"b":[1,2.5,true,null],"a":"x"}"#)])
            .expect("rt");
        let Val::S(text) = out else {
            panic!("expected str")
        };
        let v: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(v["a"], "x");
        assert_eq!(
            v["b"],
            serde_json::json!([1, 2.5, true, null]) // integral Nums emit as ints
        );
    }

    #[test]
    fn ledger_cy_writes_verifiable_chain() {
        // temp ledger + slicks key — the .cy implementation must produce
        // entries the daemon's verify() would accept.
        let dir = std::env::temp_dir().join(format!("badapple_ledger_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let ledger = dir.join("ledger.jsonl");
        let key = dir.join("slicks.key");
        let key_hex = "aa".repeat(32); // 32 bytes as hex — triggers hex-decode path
        std::fs::write(&key, &key_hex).unwrap();
        std::env::set_var("BADAPPLE_LEDGER", &ledger);
        std::env::set_var("BADAPPLE_SLICKS_KEY_PATH", &key);

        let src = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("coyote/ledger.cy");
        let build = std::env::temp_dir().join("badapple_coyote_ledger_build");
        let mut vm = coyote::load_vm(&src, &build, true, false, true).expect("load ledger.cy");
        bind_all(&mut vm).expect("bind externs");

        let ok = vm
            .call_fn(
                "ledger_delegated_ask",
                vec![
                    str_arg("10.0.0.5:9876"),
                    str_arg("req-001"),
                    str_arg("what is the wifi password"),
                ],
            )
            .expect("ledger_delegated_ask");
        assert!(matches!(ok, Val::B(true)));
        let ok = vm
            .call_fn(
                "ledger_delegated_answer",
                vec![
                    str_arg("10.0.0.5:9876"),
                    str_arg("req-001"),
                    str_arg("ok"),
                    str_arg("it is hunter2"),
                ],
            )
            .expect("ledger_delegated_answer");
        assert!(matches!(ok, Val::B(true)));

        // verify the chain exactly like BadAppleSecurity.swift verify():
        // recompute HMAC over the canonical body, check linkage.
        let secret = hex::decode(&key_hex).unwrap();
        let text = std::fs::read_to_string(&ledger).unwrap();
        let mut prev = {
            use sha2::Digest;
            hex::encode(Sha256::digest(b"bad-apple-genesis-v1"))
        };
        let mut seen = 0;
        for line in text.lines().filter(|l| !l.trim().is_empty()) {
            let e: serde_json::Value = serde_json::from_str(line).unwrap();
            // canonical body — sorted keys, compact (matches daemon verify)
            let body = serde_json::json!({
                "data": e["data"],
                "prev_hash": e["prev_hash"],
                "ts": e["ts"],
                "type": e["type"],
            });
            let expected = hmac_hex(&secret, serde_json::to_string(&body).unwrap().as_bytes());
            assert_eq!(e["hash"].as_str().unwrap(), expected, "hash mismatch");
            assert_eq!(e["prev_hash"].as_str().unwrap(), prev, "chain broken");
            prev = e["hash"].as_str().unwrap().to_string();
            seen += 1;
        }
        assert_eq!(seen, 2);
        // prompts are hashed, never recorded raw
        assert!(!text.contains("wifi password"));
        assert!(!text.contains("hunter2"));

        std::env::remove_var("BADAPPLE_LEDGER");
        std::env::remove_var("BADAPPLE_SLICKS_KEY_PATH");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The .cy infer() driving a real TCP peer that speaks the EXISTING
    /// Rust wire protocol — u32-BE len + [nonce||ct||tag] + ciborium serde.
    /// If the .cy framing/crypto/cbor diverge by a byte, this fails.
    #[test]
    fn p2p_cy_infer_wire_compatible() {
        let secret = "test-p2p-secret-for-wire-compat-0123456789";
        let key = crate::p2p_crypto::derive_key_from_bytes(secret.as_bytes());

        let dir = std::env::temp_dir().join(format!("badapple_p2p_cy_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let ledger = dir.join("ledger.jsonl");
        std::env::set_var("BADAPPLE_LEDGER", &ledger);
        std::env::set_var("BADAPPLE_P2P_SECRET", secret);
        // make sure the .cy doesn't fall through to the real slicks key
        std::env::set_var("BADAPPLE_SLICKS_KEY_PATH", dir.join("nonexistent.key"));

        // The serving peer — an independent implementation of the Rust
        // wire path: decrypt, cbor-decode into the real TransferFrame,
        // then answer InferResponse the same way send_frame does.
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let server = std::thread::spawn(move || {
            use std::io::{Read, Write};
            let (mut st, _) = listener.accept().unwrap();
            let mut hdr = [0u8; 4];
            st.read_exact(&mut hdr).unwrap();
            let n = u32::from_be_bytes(hdr) as usize;
            let mut ct = vec![0u8; n];
            st.read_exact(&mut ct).unwrap();
            let pt = crate::p2p_crypto::P2PCipher::new(&key)
                .unwrap()
                .decrypt(&ct)
                .unwrap();
            // decode into the REAL enum — the strongest compat check
            let frame: crate::p2p_model::TransferFrame = ciborium::from_reader(&pt[..]).unwrap();
            let crate::p2p_model::TransferFrame::InferRequest(req) = frame else {
                panic!("expected InferRequest, got {frame:?}");
            };
            assert_eq!(req.prompt, "ping through coyote");
            assert_eq!(req.max_tokens, 7);
            assert_eq!(req.from_peer, "tester");

            let resp = crate::p2p_model::TransferFrame::InferResponse(
                crate::p2p_model::InferResponseFrame {
                    request_id: req.request_id,
                    text: "pong through rust".to_string(),
                    tier: Some("ane".to_string()),
                    elapsed_ms: 42,
                },
            );
            let mut body = Vec::new();
            ciborium::into_writer(&resp, &mut body).unwrap();
            let sealed = crate::p2p_crypto::P2PCipher::new(&key)
                .unwrap()
                .encrypt(&body)
                .unwrap();
            st.write_all(&(sealed.len() as u32).to_be_bytes()).unwrap();
            st.write_all(&sealed).unwrap();
        });

        // .cy side
        let src = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("coyote/p2p.cy");
        let build = std::env::temp_dir().join("badapple_coyote_p2p_build");
        let mut vm = coyote::load_vm(&src, &build, true, false, true).expect("load p2p.cy");
        bind_all(&mut vm).expect("bind externs");
        let resp = vm
            .call_fn(
                "infer",
                vec![
                    str_arg(&format!("127.0.0.1:{port}")),
                    str_arg("ping through coyote"),
                    Val::I(7),
                    str_arg("tester"),
                ],
            )
            .expect("infer");

        server.join().unwrap();

        // response Json: {"status":"ok","text":"pong through rust",...}
        let Val::En(_, _, payload) = &resp else {
            panic!("expected Json enum, got something else");
        };
        let Some(Val::M(m)) = payload.first() else {
            panic!("expected Json::Obj");
        };
        let m = m.borrow();
        let get = |k: &str| -> Option<String> {
            m.iter()
                .find(|(kk, _)| matches!(kk, Val::S(s) if &**s == k))
                .and_then(|(_, v)| {
                    if let Val::En(_, _, p) = v {
                        if let Some(Val::S(x)) = p.first() {
                            return Some(x.to_string());
                        }
                    }
                    None
                })
        };
        eprintln!("infer result message: {:?}", get("message"));
        assert_eq!(get("status").as_deref(), Some("ok"));
        assert_eq!(get("text").as_deref(), Some("pong through rust"));
        assert_eq!(get("tier").as_deref(), Some("ane"));

        // dual attestation: ask + answer on the local chain
        let text = std::fs::read_to_string(&ledger).unwrap();
        assert!(text.contains("delegated_ask"));
        assert!(text.contains("delegated_answer"));
        assert!(!text.contains("ping through coyote")); // hashed, never raw

        std::env::remove_var("BADAPPLE_LEDGER");
        std::env::remove_var("BADAPPLE_P2P_SECRET");
        std::env::remove_var("BADAPPLE_SLICKS_KEY_PATH");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// p2p.cy pull/push chunk transfer against the REAL serde wire types —
    /// a Rust thread serves pull_over_stream's exact frame sequence and
    /// .cy pulls the file, verifying every chunk's bytes and the final
    /// sha256 through the on-disk file.
    #[test]
    fn p2p_cy_pull_wire_compatible() {
        use crate::p2p_crypto::P2PCipher;
        use crate::p2p_model::{self, TransferFrame};
        use std::io::{Read, Write};

        let secret = "test-p2p-secret-for-wire-compat-0123456789";
        let key = crate::p2p_crypto::derive_key_from_bytes(secret.as_bytes());

        let dir = std::env::temp_dir().join(format!("badapple_p2p_pull_cy_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let out_dir = dir.join("out");
        std::fs::create_dir_all(&out_dir).unwrap();
        std::env::set_var("BADAPPLE_P2P_SECRET", secret);
        std::env::set_var("BADAPPLE_SLICKS_KEY_PATH", dir.join("nonexistent.key"));

        // Multi-chunk payload — 3 chunks worth (not aligned to 256K, so
        // the last chunk is short) with non-utf8 bytes to exercise Bytes.
        let mut payload = vec![0u8; 600 * 1024 + 17];
        for (i, b) in payload.iter_mut().enumerate() {
            *b = (i % 251) as u8;
        }
        let total_chunks = payload.len().div_ceil(p2p_model::DEFAULT_CHUNK_SIZE);
        let sha = {
            use sha2::Digest;
            hex::encode(sha2::Sha256::digest(&payload))
        };

        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let payload_t = payload.clone();
        let server = std::thread::spawn(move || {
            let (mut st, _) = listener.accept().unwrap();
            let cipher = P2PCipher::new(&key).unwrap();
            let recv = |st: &mut std::net::TcpStream| -> TransferFrame {
                let mut hdr = [0u8; 4];
                st.read_exact(&mut hdr).unwrap();
                let n = u32::from_be_bytes(hdr) as usize;
                let mut ct = vec![0u8; n];
                st.read_exact(&mut ct).unwrap();
                let pt = cipher.decrypt(&ct).unwrap();
                ciborium::from_reader(&pt[..]).unwrap()
            };
            let send = |st: &mut std::net::TcpStream, f: &TransferFrame| {
                let mut body = Vec::new();
                ciborium::into_writer(f, &mut body).unwrap();
                let sealed = cipher.encrypt(&body).unwrap();
                st.write_all(&(sealed.len() as u32).to_be_bytes()).unwrap();
                st.write_all(&sealed).unwrap();
            };

            // Request
            let TransferFrame::Request(req) = recv(&mut st) else {
                panic!("expected Request")
            };
            assert_eq!(req.model_id, "test-model");

            // Offer
            send(
                &mut st,
                &TransferFrame::Offer(p2p_model::OfferFrame {
                    model_id: "test-model".into(),
                    file_name: "test-model.safetensors".into(),
                    total_bytes: payload_t.len() as u64,
                    total_chunks,
                    chunk_size: p2p_model::DEFAULT_CHUNK_SIZE,
                    sha256: sha.clone(),
                }),
            );
            assert!(matches!(recv(&mut st), TransferFrame::Accept));

            // Chunks with acks
            for index in 0..total_chunks {
                let start = index * p2p_model::DEFAULT_CHUNK_SIZE;
                let end = ((index + 1) * p2p_model::DEFAULT_CHUNK_SIZE).min(payload_t.len());
                send(
                    &mut st,
                    &TransferFrame::Chunk(p2p_model::ChunkFrame {
                        index,
                        total: total_chunks,
                        bytes: payload_t[start..end].to_vec(),
                    }),
                );
                match recv(&mut st) {
                    TransferFrame::Ack(a) if a.index == index => {}
                    other => panic!("expected Ack({index}), got {other:?}"),
                }
            }
            send(&mut st, &TransferFrame::Done);
        });

        // .cy pull
        let mut vm = cy_vm("p2p.cy");
        let resume = Val::En(0, 0, Rc::new(Vec::new())); // placeholder; real enum below
                                                         // Build Json::Arr([]) through the module's own enum indices
        let j = json_idx(&vm).unwrap();
        let empty_resume = Val::En(
            j.en,
            j.arr,
            Rc::new(vec![Val::V(Rc::new(RefCell::new(vec![])))]),
        );
        drop(resume);
        let resp = vm
            .call_fn(
                "pull",
                vec![
                    str_arg(&format!("127.0.0.1:{port}")),
                    str_arg("test-model"),
                    empty_resume,
                    str_arg(out_dir.to_str().unwrap()),
                ],
            )
            .expect("pull");

        server.join().unwrap();

        // surface the .cy status message before the file check
        if let Val::En(_, _, p) = &resp {
            if let Some(Val::M(m)) = p.first() {
                for (k, v) in m.borrow().iter() {
                    if let (Val::S(kk), Val::En(_, _, pp)) = (k, v) {
                        if let Some(Val::S(x)) = pp.first() {
                            eprintln!("pull resp {} = {}", kk, x);
                        }
                    }
                }
            }
        }
        let got = std::fs::read(out_dir.join("test-model.safetensors")).unwrap();
        assert_eq!(got, payload, "transferred bytes differ");

        let Val::En(_, _, payload_v) = &resp else {
            panic!("expected Json enum")
        };
        let Some(Val::M(m)) = payload_v.first() else {
            panic!("expected Json::Obj")
        };
        let m = m.borrow();
        let status = m
            .iter()
            .find(|(k, _)| matches!(k, Val::S(s) if &**s == "status"))
            .and_then(|(_, v)| match v {
                Val::En(_, _, p) => match p.first() {
                    Some(Val::S(x)) => Some(x.to_string()),
                    _ => None,
                },
                _ => None,
            });
        assert_eq!(status.as_deref(), Some("ok"));
        // .part cleaned up
        assert!(!out_dir.join("test-model.safetensors.part").exists());

        std::env::remove_var("BADAPPLE_P2P_SECRET");
        std::env::remove_var("BADAPPLE_SLICKS_KEY_PATH");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Reverse direction: a REAL Rust client pushes to a .cy serve_once —
    /// exercises PushRequest → PushAccept → Offer → Accept → Chunk/Ack →
    /// Done → sha256 verify → .part → rename on the Coyote side.
    #[test]
    fn p2p_cy_serve_push_wire_compatible() {
        use crate::p2p_crypto::P2PCipher;
        use crate::p2p_model::{self, TransferFrame};
        use std::io::{Read, Write};

        let secret = "test-p2p-secret-for-wire-compat-0123456789";
        let key = crate::p2p_crypto::derive_key_from_bytes(secret.as_bytes());

        let dir = std::env::temp_dir().join(format!("badapple_p2p_srv_cy_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let model_dir = dir.join("models");
        std::fs::create_dir_all(&model_dir).unwrap();
        std::env::set_var("BADAPPLE_P2P_SECRET", secret);
        std::env::set_var("BADAPPLE_SLICKS_KEY_PATH", dir.join("nonexistent.key"));

        let mut payload = vec![0u8; 300 * 1024 + 5];
        for (i, b) in payload.iter_mut().enumerate() {
            *b = ((i * 7) % 253) as u8;
        }
        let total_chunks = payload.len().div_ceil(p2p_model::DEFAULT_CHUNK_SIZE);
        let sha = {
            use sha2::Digest;
            hex::encode(sha2::Sha256::digest(&payload))
        };

        // grab a free port, release it for the .cy listener
        let probe = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = probe.local_addr().unwrap().port();
        drop(probe);

        // Rust client thread — real serde frames, real cipher
        let payload_c = payload.clone();
        let client = std::thread::spawn(move || {
            // the .cy listener binds inside serve_once — retry until it is up
            let mut st = loop {
                match std::net::TcpStream::connect(format!("127.0.0.1:{port}")) {
                    Ok(s) => break s,
                    Err(_) => std::thread::sleep(std::time::Duration::from_millis(20)),
                }
            };
            let cipher = P2PCipher::new(&key).unwrap();
            let recv = |st: &mut std::net::TcpStream| -> TransferFrame {
                let mut hdr = [0u8; 4];
                st.read_exact(&mut hdr).unwrap();
                let n = u32::from_be_bytes(hdr) as usize;
                let mut ct = vec![0u8; n];
                st.read_exact(&mut ct).unwrap();
                let pt = cipher.decrypt(&ct).unwrap();
                ciborium::from_reader(&pt[..]).unwrap()
            };
            let send = |st: &mut std::net::TcpStream, f: &TransferFrame| {
                let mut body = Vec::new();
                ciborium::into_writer(f, &mut body).unwrap();
                let sealed = cipher.encrypt(&body).unwrap();
                st.write_all(&(sealed.len() as u32).to_be_bytes()).unwrap();
                st.write_all(&sealed).unwrap();
            };

            send(
                &mut st,
                &TransferFrame::PushRequest(p2p_model::PushRequestFrame {
                    model_id: "pushed-model".into(),
                }),
            );
            assert!(matches!(recv(&mut st), TransferFrame::PushAccept));
            send(
                &mut st,
                &TransferFrame::Offer(p2p_model::OfferFrame {
                    model_id: "pushed-model".into(),
                    file_name: "pushed-model.safetensors".into(),
                    total_bytes: payload_c.len() as u64,
                    total_chunks,
                    chunk_size: p2p_model::DEFAULT_CHUNK_SIZE,
                    sha256: sha,
                }),
            );
            assert!(matches!(recv(&mut st), TransferFrame::Accept));
            for index in 0..total_chunks {
                let start = index * p2p_model::DEFAULT_CHUNK_SIZE;
                let end = ((index + 1) * p2p_model::DEFAULT_CHUNK_SIZE).min(payload_c.len());
                send(
                    &mut st,
                    &TransferFrame::Chunk(p2p_model::ChunkFrame {
                        index,
                        total: total_chunks,
                        bytes: payload_c[start..end].to_vec(),
                    }),
                );
                match recv(&mut st) {
                    TransferFrame::Ack(a) if a.index == index => {}
                    other => panic!("expected Ack({index}), got {other:?}"),
                }
            }
            send(&mut st, &TransferFrame::Done);
        });

        let mut vm = cy_vm("p2p.cy");
        let r = vm
            .call_fn(
                "serve_once",
                vec![Val::I(port as i64), str_arg(model_dir.to_str().unwrap())],
            )
            .expect("serve_once");
        client.join().unwrap();

        let msg = match &r {
            Val::S(x) => x.to_string(),
            _ => "<non-str>".to_string(),
        };
        assert_eq!(msg, "", "serve_once returned error");

        let got = std::fs::read(model_dir.join("pushed-model.safetensors")).unwrap();
        assert_eq!(got, payload, "received bytes differ");
        assert!(!model_dir.join("pushed-model.safetensors.part").exists());

        std::env::remove_var("BADAPPLE_P2P_SECRET");
        std::env::remove_var("BADAPPLE_SLICKS_KEY_PATH");
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn cy_vm(file: &str) -> Vm {
        let src = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("coyote")
            .join(file);
        let build = std::env::temp_dir().join(format!("badapple_cy_build_{}", file));
        let mut vm = coyote::load_vm(&src, &build, true, false, true).expect("coyote load");
        bind_all(&mut vm).expect("bind externs");
        vm
    }

    /// org_policy.cy: .cy keygen + sign, verified by the RUST verifier,
    /// and a Rust-signed file verified by the .cy — both directions.
    #[test]
    fn org_policy_cross_verifies() {
        let dir = std::env::temp_dir().join(format!("orgcy_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let mut vm = cy_vm("org_policy.cy");

        // .cy keygen
        let kg = vm
            .call_fn("keygen", vec![str_arg(&dir.to_string_lossy())])
            .expect("keygen");
        let Val::En(_, _, p) = &kg else {
            panic!("expected Json")
        };
        let Some(Val::M(m)) = p.first() else {
            panic!("expected Obj")
        };
        let key_path = {
            let m = m.borrow();
            m.iter()
                .find(|(k, _)| matches!(k, Val::S(s) if &**s == "key"))
                .and_then(|(_, v)| {
                    if let Val::En(_, _, pp) = v {
                        if let Some(Val::S(x)) = pp.first() {
                            return Some(x.to_string());
                        }
                    }
                    None
                })
                .expect("key path")
        };
        let pub_path = std::path::Path::new(&key_path)
            .parent()
            .unwrap()
            .join("org_policy.pub");

        // .cy sign a policy, RUST verifies it
        let policy = dir.join("policy.yaml");
        std::fs::write(&policy, "defaults:\n  allowed: true\n").unwrap();
        let sig = dir.join("policy.yaml.sig");
        let ok = vm
            .call_fn(
                "sign_file",
                vec![
                    str_arg(&policy.to_string_lossy()),
                    str_arg(&key_path),
                    str_arg(&sig.to_string_lossy()),
                ],
            )
            .expect("sign_file");
        assert!(matches!(ok, Val::B(true)));
        assert!(
            crate::org_policy::verify_file(&policy, &sig, &pub_path).unwrap(),
            "rust verifier must accept the .cy signature"
        );

        // RUST signs, .cy verifies — and tampering fails closed
        let rust_sig = dir.join("policy2.sig");
        crate::org_policy::sign_file(&policy, &key_path, &rust_sig).unwrap();
        let ok = vm
            .call_fn(
                "verify_file",
                vec![
                    str_arg(&policy.to_string_lossy()),
                    str_arg(&rust_sig.to_string_lossy()),
                    str_arg(&pub_path.to_string_lossy()),
                ],
            )
            .expect("verify_file");
        assert!(matches!(ok, Val::I(1)));

        std::fs::write(&policy, "defaults:\n  allowed: true\n  autopilot: true\n").unwrap();
        let bad = vm
            .call_fn(
                "verify_file",
                vec![
                    str_arg(&policy.to_string_lossy()),
                    str_arg(&rust_sig.to_string_lossy()),
                    str_arg(&pub_path.to_string_lossy()),
                ],
            )
            .expect("verify tampered");
        assert!(matches!(bad, Val::I(0)));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// config.cy: env + overlay parity with Config::from_env.
    #[test]
    fn config_cy_matches_rust() {
        std::env::set_var("BADAPPLE_TELEMETRY_PORT", "9999");
        std::env::set_var("BADAPPLE_PEERS", "a:1, b:2;c:3");
        std::env::set_var("BADAPPLE_STATE_FILE", "cy_state.json");
        std::env::remove_var("BADAPPLE_CONFIG_FILE");

        let mut vm = cy_vm("config.cy");
        let c = vm.call_fn("config_from_env", vec![]).expect("config");
        let Val::St(_, fields) = &c else {
            panic!("expected Config struct")
        };
        let f = fields.borrow();
        let strf = |i: usize| match &f[i] {
            Val::S(s) => s.to_string(),
            _ => panic!("field {i} not str"),
        };
        let intf = |i: usize| match &f[i] {
            Val::I(n) => *n,
            _ => panic!("field {i} not i64"),
        };
        // field order = Config struct decl order
        assert_eq!(intf(0), 9999); // telemetry_port
        assert_eq!(strf(4), "cy_state.json"); // state_file
        let Val::V(peers) = &f[15] else {
            panic!("peer_nodes not vec")
        };
        let peers: Vec<String> = peers
            .borrow()
            .iter()
            .filter_map(|v| {
                if let Val::S(s) = v {
                    Some(s.to_string())
                } else {
                    None
                }
            })
            .collect();
        assert_eq!(peers, vec!["a:1", "b:2", "c:3"]);

        // Rust side agrees
        let rc = crate::config::Config::from_env();
        assert_eq!(rc.telemetry_port, 9999);
        assert_eq!(rc.state_file, PathBuf::from("cy_state.json"));
        assert_eq!(rc.peer_nodes, vec!["a:1", "b:2", "c:3"]);

        std::env::remove_var("BADAPPLE_TELEMETRY_PORT");
        std::env::remove_var("BADAPPLE_PEERS");
        std::env::remove_var("BADAPPLE_STATE_FILE");
    }

    /// mesh_sync.cy: packet codec + LWW merge + handle_incoming.
    #[test]
    fn mesh_sync_cy_packet_and_merge() {
        let dir = std::env::temp_dir().join(format!("meshcy_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        // point the store at the temp dir: HOME/Library/.../bad_apple/mesh
        std::env::set_var("HOME", &dir);
        std::fs::create_dir_all(dir.join("Library/Application Support/bad_apple/mesh")).unwrap();

        let mut vm = cy_vm("mesh_sync.cy");
        // packet encode/decode roundtrip
        let push = vm
            .call_fn(
                "packet_push",
                vec![str_arg("settings@test"), {
                    let j = json_idx(&vm).unwrap();
                    let jstrv = |x: &str| Val::En(j.en, j.strv, Rc::new(vec![Val::S(Rc::from(x))]));
                    let jnumv = |n: f64| Val::En(j.en, j.num, Rc::new(vec![Val::F(n)]));
                    let mm: Vec<(Val, Val)> = vec![
                        (Val::S(Rc::from("kind")), jstrv("Settings")),
                        (Val::S(Rc::from("body")), jstrv("{\"k\":1}")),
                        (Val::S(Rc::from("timestamp")), jnumv(100.0)),
                        (Val::S(Rc::from("origin")), jstrv("test")),
                        (Val::S(Rc::from("version")), jnumv(1.0)),
                    ];
                    Val::En(
                        j.en,
                        j.obj,
                        Rc::new(vec![Val::M(Rc::new(RefCell::new(mm)))]),
                    )
                }],
            )
            .expect("packet_push");
        let enc = vm
            .call_fn("encode_packet", vec![push.clone()])
            .expect("encode");
        let Val::S(b64) = &enc else {
            panic!("expected str")
        };
        let dec = vm
            .call_fn("decode_packet", vec![str_arg(b64)])
            .expect("decode");
        let j = json_val(&dec).unwrap();
        assert_eq!(j["op"], "Push");
        assert_eq!(j["doc_id"], "settings@test");
        assert_eq!(j["doc"]["kind"], "Settings");

        // merge: older loses, newer wins — LWW on (timestamp, version)
        let merged = vm
            .call_fn("merge_doc", vec![j_to_val(&vm, &j["doc"])])
            .expect("merge");
        assert!(matches!(merged, Val::B(true)));
        // same ts, lower version → rejected
        let mut older = j["doc"].clone();
        older["version"] = serde_json::json!(0);
        let merged2 = vm
            .call_fn("merge_doc", vec![j_to_val(&vm, &older)])
            .expect("merge2");
        assert!(matches!(merged2, Val::B(false)));
        // newer wins
        let mut newer = j["doc"].clone();
        newer["timestamp"] = serde_json::json!(200);
        let merged3 = vm
            .call_fn("merge_doc", vec![j_to_val(&vm, &newer)])
            .expect("merge3");
        assert!(matches!(merged3, Val::B(true)));

        let _ = std::fs::remove_dir_all(&dir);
    }

    fn j_to_val(vm: &Vm, j: &serde_json::Value) -> Val {
        json_to_val(j, &json_idx(vm).unwrap())
    }

    /// firewall.cy: fingerprint must match the Rust/Swift contract vector.
    #[test]
    fn firewall_cy_fingerprint_matches_rust() {
        let mut vm = cy_vm("firewall.cy");
        let pats = vm
            .call_fn("builtin_patterns", vec![])
            .expect("builtin_patterns");
        let Val::V(items) = &pats else {
            panic!("expected vec")
        };
        assert_eq!(items.borrow().len(), 6);
        let fp = vm.call_fn("fingerprint", vec![pats]).expect("fingerprint");
        let Val::S(h) = &fp else {
            panic!("expected str")
        };
        // same known vector as firewall_machine::tests::fingerprint_matches_swift_side
        assert_eq!(
            h.as_ref(),
            "9b7e159682aac2a1d7425bb4fe1246166a2e97fcf37a21ffdc3fbfbff4f54169"
        );
        // merged_patterns: env override → builtins + non-comment lines
        let dir = std::env::temp_dir().join(format!("fwcy_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let bl = dir.join("blocklist.txt");
        std::fs::write(&bl, "# comment\n\ns3cr3t\n  padded \n").unwrap();
        std::env::set_var("BADAPPLE_BLOCKLIST", &bl);
        let merged = vm
            .call_fn("merged_patterns", vec![])
            .expect("merged_patterns");
        let Val::V(m) = &merged else {
            panic!("expected vec")
        };
        let got: Vec<String> = m
            .borrow()
            .iter()
            .filter_map(|v| match v {
                Val::S(x) => Some(x.to_string()),
                _ => None,
            })
            .collect();
        assert_eq!(got.len(), 6 + 2);
        assert_eq!(got[6], "s3cr3t");
        assert_eq!(got[7], "padded");
        std::env::remove_var("BADAPPLE_BLOCKLIST");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// sentinel.cy: parse_verdict must match the Rust truth table —
    /// clean ALLOW/DENY/ESCALATE parse, everything else escalates.
    #[test]
    fn sentinel_cy_parse_verdict_matches_rust() {
        let mut vm = cy_vm("sentinel.cy");
        let mut pv = |raw: &str| -> (String, String) {
            let v = vm
                .call_fn("parse_verdict", vec![str_arg(raw)])
                .expect("parse_verdict");
            let j = json_val(&v).unwrap();
            (
                j["verdict"].as_str().unwrap_or("").to_string(),
                j["reason"].as_str().unwrap_or("").to_string(),
            )
        };
        // the Rust truth table — sentinel.rs::tests
        assert_eq!(pv("ALLOW\nfine").0, "allow");
        assert_eq!(pv("ALLOW\nfine").1, "fine");
        assert_eq!(pv("DENY\ndestructive").0, "deny");
        assert_eq!(pv("ESCALATE\nunclear").0, "escalate");
        assert_eq!(pv("allow lowercase too").0, "allow");
        assert_eq!(pv("").0, "escalate");
        assert_eq!(pv("I think this is fine").0, "escalate");
        assert_eq!(pv("ALLOWANCE denied").0, "allow"); // starts_with ALLOW
                                                       // no reason line → reason falls back to the head
        assert_eq!(pv("ALLOW").1, "ALLOW");
        // vet without a model → unavailable, not escalate
        let v = vm
            .call_fn(
                "vet",
                vec![str_arg("run_shell"), str_arg("{}"), str_arg("approved")],
            )
            .expect("vet");
        let j = json_val(&v).unwrap();
        assert_eq!(j["verdict"], "unavailable");
    }

    /// ify.cy: tail a live sha-chained ledger, verify each line through
    /// the sovereign verifier extern, and fire the detection rules.
    #[test]
    fn ify_cy_tails_and_detects() {
        let dir = std::env::temp_dir().join(format!("ifycy_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::env::set_var("BADAPPLE_LEDGER", dir.join("ledger.jsonl"));
        std::env::set_var("BADAPPLE_IFY_DIR", dir.join("ify"));
        std::env::set_var("BADAPPLE_SLICKS_KEY_PATH", dir.join("no.key"));
        std::env::set_var("BADAPPLE_IFY_PHASE", "autopilot");

        // write three ledger lines through ledger.cy (real chain format)
        let mut lvm = cy_vm("ledger.cy");
        for ty in ["query", "tool_call", "query"] {
            let ok = lvm
                .call_fn(
                    "ledger_append",
                    vec![
                        str_arg(ty),
                        j_to_val(
                            &lvm,
                            &serde_json::json!({"tool": "run_shell", "approved": true}),
                        ),
                    ],
                )
                .expect("append");
            assert!(matches!(ok, Val::B(true)));
        }

        let mut vm = cy_vm("ify.cy");
        // seed a warmed-up state so rules can fire
        let st = vm.call_fn("state_new", vec![]).expect("state_new");
        let sj = json_val(&st).unwrap();
        let mut sj = sj;
        sj["events_seen"] = serde_json::json!(500);
        sj["installed_at"] = serde_json::json!(0); // old → autopilot via env anyway
        let st = j_to_val(&vm, &sj);
        let secrets = vm.call_fn("load_slicks_secrets", vec![]).expect("secrets");
        let rep = vm
            .call_fn("tail_ledger", vec![st, secrets.clone()])
            .expect("tail_ledger");
        let j = json_val(&rep).unwrap();
        assert_eq!(
            j["new_events"].as_f64().unwrap() as u64,
            3,
            "all three lines verified+consumed"
        );
        assert_eq!(j["chain_broken"], false);
        // warmed + novel types → novel_event_type findings
        let findings = j["findings"].as_array().unwrap();
        assert!(
            findings.iter().any(|f| f["rule"] == "novel_event_type"),
            "expected novel_event_type finding, got {findings:?}"
        );
        // offset advanced past the file
        let off = j["state"]["ledger_offset"].as_f64().unwrap() as u64;
        let size = std::fs::metadata(dir.join("ledger.jsonl")).unwrap().len();
        assert_eq!(off, size);

        // tamper: corrupt the middle line → chain_break, tail stops
        std::env::set_var("BADAPPLE_LEDGER", dir.join("ledger2.jsonl"));
        std::fs::copy(dir.join("ledger.jsonl"), dir.join("ledger2.jsonl")).unwrap();
        let txt = std::fs::read_to_string(dir.join("ledger2.jsonl")).unwrap();
        let mut lines: Vec<String> = txt.lines().map(String::from).collect();
        lines[1] = lines[1].replace("run_shell", "run_nothing");
        std::fs::write(dir.join("ledger2.jsonl"), lines.join("\n") + "\n").unwrap();
        let st2 = vm.call_fn("state_new", vec![]).expect("state2");
        let rep2 = vm
            .call_fn("tail_ledger", vec![st2, secrets.clone()])
            .expect("tail2");
        let j2 = json_val(&rep2).unwrap();
        assert!(j2["chain_broken"].as_bool().unwrap());
        assert!(j2["findings"]
            .as_array()
            .unwrap()
            .iter()
            .any(|f| f["rule"] == "chain_break"));

        std::env::remove_var("BADAPPLE_LEDGER");
        std::env::remove_var("BADAPPLE_IFY_DIR");
        std::env::remove_var("BADAPPLE_SLICKS_KEY_PATH");
        std::env::remove_var("BADAPPLE_IFY_PHASE");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// strategy.cy: redb roundtrip, EMA record, best_match scoring, and
    /// the rust_repair healer against the Rust test vectors.
    #[test]
    fn strategy_cy_roundtrip_and_repair() {
        let dir = std::env::temp_dir().join(format!("stratcy_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let db = dir.join("strategies.redb").to_string_lossy().to_string();
        let mut vm = cy_vm("strategy.cy");

        let st = vm
            .call_fn(
                "strategy_new",
                vec![
                    str_arg("k1"),
                    str_arg("restart the daemon"),
                    str_arg("shell"),
                    str_arg("launchctl kickstart"),
                ],
            )
            .expect("new");
        assert!(matches!(
            vm.call_fn("db_put", vec![str_arg(&db), st.clone()]),
            Ok(Val::B(true))
        ));
        let got = vm
            .call_fn("db_get", vec![str_arg(&db), str_arg("k1")])
            .expect("get");
        let j = json_val(&got).unwrap();
        assert_eq!(j["key"], "k1");
        assert_eq!(j["reliability"], 0.5);

        // record(true) → reliability = 0.5*0.7 + 1.0*0.3 = 0.65
        let rec = vm
            .call_fn("strategy_record", vec![got, Val::B(true)])
            .expect("record");
        let rj = json_val(&rec).unwrap();
        assert_eq!(rj["uses"], 1.0);
        assert_eq!(rj["successes"], 1.0);
        assert!((rj["reliability"].as_f64().unwrap() - 0.65).abs() < 1e-9);

        // best_match: containment hit
        let bm = vm
            .call_fn(
                "best_match",
                vec![str_arg(&db), str_arg("restart the daemon please")],
            )
            .expect("best_match");
        assert_eq!(json_val(&bm).unwrap()["key"], "k1");

        // rust_repair: the two documented cases from the Rust tests
        let fixed = vm
            .call_fn(
                "rust_repair",
                vec![
                    str_arg("fn main() {\n    let x = 1\n    println!(\"{}\", x);\n}"),
                    str_arg("error: expected `;`, found `println`\n --> src/main.rs:2:14"),
                ],
            )
            .expect("repair");
        let Val::S(txt) = &fixed else {
            panic!("expected str")
        };
        assert!(txt.contains("let x = 1;"), "semicolon appended: {txt}");

        let closed = vm
            .call_fn(
                "rust_repair",
                vec![
                    str_arg("fn main() {\n    if true {\n        println!(\"hi\");"),
                    str_arg("error: unclosed delimiter\n --> src/main.rs:1:1"),
                ],
            )
            .expect("repair2");
        let Val::S(txt2) = &closed else {
            panic!("expected str")
        };
        assert_eq!(
            txt2.matches('{').count(),
            txt2.matches('}').count(),
            "braces balanced: {txt2}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// scavenger.cy: chunking, distinctive_terms, watch registry, and the
    /// incremental index loop — rescan skips unchanged, reindexes changed.
    #[test]
    fn scavenger_cy_index_and_registry() {
        let dir = std::env::temp_dir().join(format!("scavcy_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let src = dir.join("src");
        std::fs::create_dir_all(&src).unwrap();
        let db = dir.join("grounded.redb").to_string_lossy().to_string();
        let mut vm = cy_vm("scavenger.cy");

        // chunking: small text = 1 chunk; big text overlaps at boundaries
        let small = vm
            .call_fn("grounded_chunk_text", vec![str_arg("hello world")])
            .expect("chunk");
        let Val::V(c) = &small else { panic!("vec") };
        assert_eq!(c.borrow().len(), 1);
        let big = "x".repeat(3000) + "\n\n" + &"y".repeat(3000);
        let bigc = vm
            .call_fn("grounded_chunk_text", vec![str_arg(&big)])
            .expect("big chunk");
        let Val::V(bc) = &bigc else { panic!("vec") };
        assert!(
            bc.borrow().len() >= 3,
            "expected ≥3 chunks, got {}",
            bc.borrow().len()
        );

        // distinctive_terms: snake/camel split, stopwords dropped
        let terms = vm
            .call_fn(
                "distinctive_terms",
                vec![str_arg("the automation_cage MyFunc")],
            )
            .expect("terms");
        let tv: Vec<String> = match &terms {
            Val::V(v) => v
                .borrow()
                .iter()
                .filter_map(|x| match x {
                    Val::S(s) => Some(s.to_string()),
                    _ => None,
                })
                .collect(),
            _ => vec![],
        };
        assert!(tv.contains(&"automation".to_string()));
        assert!(tv.contains(&"cage".to_string()));
        assert!(tv.contains(&"myfunc".to_string()) || tv.contains(&"func".to_string()));
        assert!(!tv.contains(&"the".to_string()));

        // watch registry roundtrip (BADAPPLE_WATCH_PATHS redirect)
        std::env::set_var("BADAPPLE_WATCH_PATHS", dir.join("watch_paths"));
        let canon = std::fs::canonicalize(&src)
            .unwrap()
            .to_string_lossy()
            .to_string();
        assert!(matches!(
            vm.call_fn("register_watch", vec![str_arg(&src.to_string_lossy())]),
            Ok(Val::B(true))
        ));
        assert!(matches!(
            vm.call_fn("register_watch", vec![str_arg(&canon)]),
            Ok(Val::B(false))
        ));
        let dirs = vm.call_fn("watched_dirs", vec![]).expect("watched");
        let Val::V(dv) = &dirs else { panic!("vec") };
        assert_eq!(dv.borrow().len(), 1);
        assert!(matches!(
            vm.call_fn("unregister_watch", vec![str_arg(&canon)]),
            Ok(Val::B(true))
        ));

        // incremental index: first pass writes chunks, second is a no-op
        std::fs::write(src.join("note.md"), "sovereign mesh memory ledger").unwrap();
        let dirs_arg = { Val::V(Rc::new(RefCell::new(vec![str_arg(&src.to_string_lossy())]))) };
        let r1 = vm
            .call_fn("index_directories_at", vec![dirs_arg.clone(), str_arg(&db)])
            .expect("index1");
        let j1 = json_val(&r1).unwrap();
        assert_eq!(j1["files"].as_f64().unwrap(), 1.0);
        assert!(j1["chunks"].as_f64().unwrap() >= 1.0);
        let r2 = vm
            .call_fn("index_directories_at", vec![dirs_arg.clone(), str_arg(&db)])
            .expect("index2");
        let j2 = json_val(&r2).unwrap();
        assert_eq!(j2["files"].as_f64().unwrap(), 0.0, "unchanged → skip");
        // change content → reindex
        std::fs::write(src.join("note.md"), "sovereign mesh memory ledger v2").unwrap();
        // bump mtime forward so the fast-path can't fire
        let mtime = vm
            .call_fn(
                "fs_mtime",
                vec![str_arg(&src.join("note.md").to_string_lossy())],
            )
            .ok();
        let _ = mtime;
        let r3 = vm
            .call_fn("index_directories_at", vec![dirs_arg, str_arg(&db)])
            .expect("index3");
        let j3 = json_val(&r3).unwrap();
        // content-hash path may still skip when mtime changed but text same;
        // here text changed → must reindex
        assert_eq!(j3["files"].as_f64().unwrap(), 1.0);

        let cnt = vm
            .call_fn("grounded_chunk_count_at", vec![str_arg(&db)])
            .expect("count");
        assert!(matches!(cnt, Val::I(n) if n >= 1));

        std::env::remove_var("BADAPPLE_WATCH_PATHS");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// vault.cy: a .cy-written vault opens in the Rust BadAppleVault and
    /// vice versa — same AES-256-GCM wire format and JSON payload.
    #[test]
    fn vault_cy_interops_with_rust() {
        let dir = std::env::temp_dir().join(format!("vaultcy_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::env::set_var("BADAPPLE_VAULT_KEY", "vault-parity-key");
        std::env::remove_var("BADAPPLE_SLICKS_KEY_PATH");
        std::env::remove_var("BADAPPLE_SLICKS_SECRET");

        let path = dir.join("vault.enc");
        let ps = path.to_string_lossy().to_string();
        let mut vm = cy_vm("vault.cy");

        // .cy writes → Rust reads
        assert!(matches!(
            vm.call_fn(
                "vault_set",
                vec![str_arg(&ps), str_arg("k1"), str_arg("v1")]
            ),
            Ok(Val::B(true))
        ));
        assert!(matches!(
            vm.call_fn(
                "vault_set",
                vec![str_arg(&ps), str_arg("k2"), str_arg("v2")]
            ),
            Ok(Val::B(true))
        ));
        let rv = crate::vault::BadAppleVault::open(&path).unwrap();
        assert_eq!(rv.get("k1").unwrap().as_deref(), Some("v1"));
        assert_eq!(rv.get("k2").unwrap().as_deref(), Some("v2"));

        // Rust writes → .cy reads
        rv.set("k3", "rust-side").unwrap();
        let got = vm
            .call_fn("vault_get", vec![str_arg(&ps), str_arg("k3")])
            .expect("get");
        assert!(matches!(&got, Val::S(v) if v.as_ref() == "rust-side"));

        // list sorted; remove semantics
        let keys = vm.call_fn("vault_list", vec![str_arg(&ps)]).expect("list");
        let ks: Vec<String> = match &keys {
            Val::V(v) => v
                .borrow()
                .iter()
                .filter_map(|x| match x {
                    Val::S(x) => Some(x.to_string()),
                    _ => None,
                })
                .collect(),
            _ => vec![],
        };
        assert_eq!(ks, vec!["k1", "k2", "k3"]);
        assert!(matches!(
            vm.call_fn("vault_remove", vec![str_arg(&ps), str_arg("k1")]),
            Ok(Val::B(true))
        ));
        assert!(matches!(
            vm.call_fn("vault_remove", vec![str_arg(&ps), str_arg("nope")]),
            Ok(Val::B(false))
        ));
        // invalid keys rejected before touching the file
        assert!(matches!(
            vm.call_fn(
                "vault_set",
                vec![str_arg(&ps), str_arg("a/b"), str_arg("x")]
            ),
            Ok(Val::B(false))
        ));
        assert!(matches!(
            vm.call_fn(
                "vault_set",
                vec![str_arg(&ps), str_arg("a..b"), str_arg("x")]
            ),
            Ok(Val::B(false))
        ));
        // Rust still opens the .cy-mutated vault
        let rv2 = crate::vault::BadAppleVault::open(&path).unwrap();
        assert_eq!(rv2.get("k2").unwrap().as_deref(), Some("v2"));
        assert_eq!(rv2.get("k1").unwrap(), None);

        std::env::remove_var("BADAPPLE_VAULT_KEY");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// cage.cy end-to-end: parse a fenced action, validate in-language,
    /// execute through the fd-safe extern, and verify the filesystem +
    /// audit log match the Rust cage's semantics.
    #[test]
    fn cage_cy_executes_and_rejects() {
        // the cage forbids /var — the fixture lives under $HOME like the
        // Rust automation_cage tests do.
        let home_dir = std::env::var("HOME").unwrap();
        let base = std::path::Path::new(&home_dir)
            .join(".badapple")
            .join(format!("cagecy_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let root = base.join("allowed");
        let home = base.join("home");
        let log = base.join("audit.jsonl");
        std::fs::create_dir_all(home.join(".Trash")).unwrap();
        std::fs::create_dir_all(&root).unwrap();
        // a file to move + a file to trash
        std::fs::write(root.join("src.txt"), b"payload").unwrap();
        std::fs::write(root.join("trashme.txt"), b"old").unwrap();

        let mut vm = cy_vm("cage.cy");

        // parse: two actions, one badapple-action + one json fence
        let input = format!(
            "pre\n```badapple-action\n{{\"operation\":\"create_directory\",\"path\":\"{}/made\"}}\n```\nmid\n```json\n{{\"operation\":\"move_to_trash\",\"path\":\"{}/trashme.txt\"}}\n```\npost",
            root.display(),
            root.display()
        );
        let parsed = vm
            .call_fn("cage_parse_actions", vec![str_arg(&input)])
            .expect("parse");
        // extract action count
        let count = {
            let Val::En(_, _, p) = &parsed else {
                panic!("not a Json enum")
            };
            let Some(Val::M(m)) = p.first() else {
                panic!("not Json::Obj")
            };
            let m = m.borrow();
            let mut n = 0;
            for (k, v) in m.iter() {
                if let Val::S(kk) = k {
                    if &**kk == "actions" {
                        if let Val::En(_, _, pp) = v {
                            if let Some(Val::V(a)) = pp.first() {
                                n = a.borrow().len();
                            }
                        }
                    }
                }
            }
            n
        };
        assert_eq!(count, 2, "expected 2 actions");

        // reject: path escape
        let bad = vm
            .call_fn(
                "cage_parse_actions",
                vec![str_arg("```badapple-action\n{\"operation\":\"create_directory\",\"path\":\"/etc/evil\"}\n```")],
            )
            .unwrap();
        // parses fine — the cage rejects at execute time, not parse time
        let badj = match &bad {
            Val::En(_, _, p) => matches!(p.first(), Some(Val::M(_))),
            _ => false,
        };
        assert!(badj);

        // reject: unknown field (deny_unknown_fields parity)
        let badfield = vm
            .call_fn(
                "cage_parse_actions",
                vec![str_arg("```badapple-action\n{\"operation\":\"create_directory\",\"path\":\"/tmp/x\",\"mode\":511}\n```")],
            )
            .unwrap();
        let has_err = match &badfield {
            Val::En(_, _, p) => {
                if let Some(Val::M(m)) = p.first() {
                    m.borrow()
                        .iter()
                        .any(|(k, _)| matches!(k, Val::S(s) if &**s == "error"))
                } else {
                    false
                }
            }
            _ => false,
        };
        assert!(has_err, "unknown field should fail closed");

        // cage from explicit pieces (env cage would jail to real home)
        let cage = {
            let j = json_idx(&vm).unwrap();
            let canon_root = std::fs::canonicalize(&root).unwrap();
            let canon_home = std::fs::canonicalize(&home).unwrap();
            let canon_log = log.to_string_lossy().to_string();
            let roots_arr = Val::En(
                j.en,
                j.arr,
                Rc::new(vec![Val::V(Rc::new(RefCell::new(vec![Val::En(
                    j.en,
                    j.strv,
                    Rc::new(vec![s(canon_root.to_string_lossy())]),
                )])))]),
            );
            let fields: Vec<(Val, Val)> = vec![
                (s("roots"), roots_arr),
                (
                    s("home"),
                    Val::En(j.en, j.strv, Rc::new(vec![s(canon_home.to_string_lossy())])),
                ),
                (
                    s("log_path"),
                    Val::En(j.en, j.strv, Rc::new(vec![s(canon_log)])),
                ),
            ];
            Val::En(
                j.en,
                j.obj,
                Rc::new(vec![Val::M(Rc::new(RefCell::new(fields)))]),
            )
        };

        // execute the parsed actions one by one
        let Val::En(_, _, pp) = &parsed else {
            panic!("not enum")
        };
        let Some(Val::M(pm)) = pp.first() else {
            panic!("not obj")
        };
        let acts = pm
            .borrow()
            .iter()
            .find(|(k, _)| matches!(k, Val::S(s) if &**s == "actions"))
            .and_then(|(_, v)| match v {
                Val::En(_, _, ap) => ap.first().and_then(|x| match x {
                    Val::V(a) => Some(a.borrow().clone()),
                    _ => None,
                }),
                _ => None,
            })
            .unwrap();

        let r1 = vm
            .call_fn("cage_execute", vec![acts[0].clone(), cage.clone()])
            .expect("exec1");
        assert!(
            root.join("made").is_dir(),
            "create_directory should have run"
        );
        // report.result == "success"
        if let Val::En(_, _, p) = &r1 {
            if let Some(Val::M(m)) = p.first() {
                let res = m
                    .borrow()
                    .iter()
                    .find(|(k, _)| matches!(k, Val::S(s) if &**s == "result"))
                    .and_then(|(_, v)| match v {
                        Val::En(_, _, q) => match q.first() {
                            Some(Val::S(x)) => Some(x.to_string()),
                            _ => None,
                        },
                        _ => None,
                    });
                assert_eq!(res.as_deref(), Some("success"));
            }
        }

        let _r2 = vm
            .call_fn("cage_execute", vec![acts[1].clone(), cage.clone()])
            .expect("exec2");
        assert!(!root.join("trashme.txt").exists(), "trash source gone");
        assert!(
            home.join(".Trash/trashme.txt").exists(),
            "trash dest exists"
        );

        // audit log was appended by the Rust execute path
        let logtxt = std::fs::read_to_string(&log).unwrap();
        assert!(logtxt.lines().count() >= 2, "audit lines: {logtxt}");

        // escaping action is refused by .cy validate before extern call
        let j = json_idx(&vm).unwrap();
        let esc_act = json_to_val(
            &serde_json::json!({"operation":"create_directory","path":"/tmp/../etc/x"}),
            &j,
        );
        let rej = vm
            .call_fn("cage_execute", vec![esc_act, cage])
            .expect("rej");
        let rej_err = match &rej {
            Val::En(_, _, p) => p
                .first()
                .and_then(|x| match x {
                    Val::M(m) => m
                        .borrow()
                        .iter()
                        .find(|(k, _)| matches!(k, Val::S(s) if &**s == "error"))
                        .map(|_| true),
                    _ => None,
                })
                .unwrap_or(false),
            _ => false,
        };
        assert!(rej_err, "escaping path must be rejected");

        let _ = std::fs::remove_dir_all(&base);
    }

    /// mcp.cy — admission policy + catalog. Mirrors the Rust validator
    /// tests: bad ids, metacharacters, network commands, env blocks.
    #[test]
    fn mcp_cy_policy_and_catalog() {
        let dir = std::env::temp_dir().join(format!("mcpcy_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let cat = dir.join("mcp_catalog.json").to_string_lossy().to_string();
        let mut vm = cy_vm("mcp.cy");
        let get_s = |v: &Val| match v {
            Val::S(x) => x.to_string(),
            _ => String::new(),
        };

        // id validation — the Rust rejections
        for id in ["", "has space", "has/slash", "bad.dot", "semi;colon"] {
            let r = vm.call_fn("mcp_id_ok", vec![str_arg(id)]).expect("id");
            assert_ne!(get_s(&r), "", "id {id:?} should be rejected");
        }
        let r = vm
            .call_fn("mcp_id_ok", vec![str_arg("good-server_1")])
            .unwrap();
        assert_eq!(get_s(&r), "");

        // command validation — network fetchers + metacharacters
        for cmd in [
            "npx",
            "/usr/bin/curl",
            "git",
            "wget",
            "a;b",
            "x|y",
            "~/home",
            "../up",
        ] {
            let args = Val::V(Rc::new(RefCell::new(vec![])));
            let r = vm
                .call_fn("mcp_validate_command", vec![str_arg(cmd), args])
                .unwrap();
            assert_ne!(get_s(&r), "", "cmd {cmd:?} should be rejected");
        }
        let r = vm
            .call_fn(
                "mcp_validate_command",
                vec![
                    str_arg("/usr/local/bin/local-mcp"),
                    Val::V(Rc::new(RefCell::new(vec![str_arg("--flag")]))),
                ],
            )
            .unwrap();
        assert_eq!(get_s(&r), "");

        // env validation — blocked keys + BADAPPLE_ prefix
        let j = json_idx(&vm).unwrap();
        let mk_env = |pairs: Vec<(&str, &str)>| -> Val {
            let m: Vec<(Val, Val)> = pairs
                .into_iter()
                .map(|(k, v)| (s(k), Val::En(j.en, j.strv, Rc::new(vec![s(v)]))))
                .collect();
            Val::En(j.en, j.obj, Rc::new(vec![Val::M(Rc::new(RefCell::new(m)))]))
        };
        for bad in ["PATH", "HOME", "DYLD_INSERT_LIBRARIES", "SUDO_COMMAND"] {
            let r = vm
                .call_fn("mcp_validate_env", vec![mk_env(vec![(bad, "x")])])
                .unwrap();
            assert_ne!(get_s(&r), "", "env {bad} should be rejected");
        }
        let r = vm
            .call_fn("mcp_validate_env", vec![mk_env(vec![("BADAPPLE_X", "y")])])
            .unwrap();
        assert_ne!(get_s(&r), "");
        let r = vm
            .call_fn(
                "mcp_validate_env",
                vec![mk_env(vec![("MY_FLAG", "1"), ("API_OK", "2")])],
            )
            .unwrap();
        assert_eq!(get_s(&r), "", "valid env rejected: {}", get_s(&r));

        // catalog roundtrip
        let server = json_to_val(
            &serde_json::json!({
                "id": "echo-srv", "name": "Echo", "command": "/usr/bin/false",
                "args": [], "env": {}, "transport": "stdio",
                "installed": false, "enabled": false, "description": ""
            }),
            &j,
        );
        let up = vm
            .call_fn("mcp_upsert", vec![str_arg(&cat), server])
            .expect("upsert");
        let ok = match &up {
            Val::En(_, _, p) => p
                .first()
                .map(|x| match x {
                    Val::M(m) => m
                        .borrow()
                        .iter()
                        .any(|(k, _)| matches!(k, Val::S(s) if &**s == "ok")),
                    _ => false,
                })
                .unwrap_or(false),
            _ => false,
        };
        assert!(ok, "upsert failed");

        // upsert with a disallowed command is refused before the write
        let badserver = json_to_val(
            &serde_json::json!({
                "id": "evil", "name": "Evil", "command": "curl",
                "args": [], "env": {}, "transport": "stdio"
            }),
            &j,
        );
        let up2 = vm
            .call_fn("mcp_upsert", vec![str_arg(&cat), badserver])
            .expect("upsert2");
        let has_err = match &up2 {
            Val::En(_, _, p) => p
                .first()
                .map(|x| match x {
                    Val::M(m) => m
                        .borrow()
                        .iter()
                        .any(|(k, _)| matches!(k, Val::S(s) if &**s == "error")),
                    _ => false,
                })
                .unwrap_or(false),
            _ => false,
        };
        assert!(has_err, "disallowed command should be refused");

        // get + set_flag + remove
        let got = vm
            .call_fn("mcp_get", vec![str_arg(&cat), str_arg("echo-srv")])
            .unwrap();
        assert!(matches!(&got, Val::En(_, _, _)));
        let gotid = match &got {
            Val::En(_, _, p) => p.first().and_then(|x| match x {
                Val::M(m) => m
                    .borrow()
                    .iter()
                    .find(|(k, _)| matches!(k, Val::S(s) if &**s == "id"))
                    .and_then(|(_, v)| match v {
                        Val::En(_, _, q) => match q.first() {
                            Some(Val::S(x)) => Some(x.to_string()),
                            _ => None,
                        },
                        _ => None,
                    }),
                _ => None,
            }),
            _ => None,
        };
        assert_eq!(gotid.as_deref(), Some("echo-srv"));

        vm.call_fn(
            "mcp_set_flag",
            vec![
                str_arg(&cat),
                str_arg("echo-srv"),
                str_arg("installed"),
                Val::B(true),
            ],
        )
        .unwrap();
        // the Rust reader sees the same catalog
        let on_disk: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&cat).unwrap()).unwrap();
        assert_eq!(on_disk["servers"][0]["installed"], true);

        vm.call_fn("mcp_remove", vec![str_arg(&cat), str_arg("echo-srv")])
            .unwrap();
        let on_disk2: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&cat).unwrap()).unwrap();
        assert_eq!(on_disk2["servers"].as_array().unwrap().len(), 0);

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// cli.cy — the organism's own front door: vault + strategy through
    /// cli_dispatch against real on-disk state.
    #[test]
    fn cli_cy_vault_and_strategy() {
        let dir = std::env::temp_dir().join(format!("clicy_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let strat = dir.join("strategies.redb");
        let strat_s = strat.to_string_lossy().to_string();
        std::env::set_var("BADAPPLE_STRATEGIES", &strat_s);
        std::env::set_var("BADAPPLE_VAULT_KEY", "clitestkey");
        std::env::set_var("BADAPPLE_DATA_DIR", dir.to_string_lossy().to_string());

        // vault set → get → list → remove, all through cli_dispatch
        let r = cli_dispatch(&["vault".into(), "set".into(), "k1".into(), "v1".into()])
            .expect("vault set handled");
        assert_eq!(r.0, 0);
        let r =
            cli_dispatch(&["vault".into(), "get".into(), "k1".into()]).expect("vault get handled");
        assert!(r.1.contains("v1"), "{}", r.1);
        let r = cli_dispatch(&["vault".into(), "list".into()]).expect("list");
        assert!(r.1.contains("k1"), "{}", r.1);
        let r = cli_dispatch(&["vault".into(), "remove".into(), "k1".into()]).expect("remove");
        assert_eq!(r.0, 0);

        // strategy learn → record → match → list through the kv store
        let r = cli_dispatch(&[
            "strategy".into(),
            "learn".into(),
            "sk".into(),
            "--problem".into(),
            "p".into(),
        ])
        .expect("learn");
        assert_eq!(r.0, 0);
        let r = cli_dispatch(&["strategy".into(), "record".into(), "sk".into(), "ok".into()])
            .expect("record");
        assert!(r.1.contains("reliability 0.6"), "{}", r.1);
        let r = cli_dispatch(&["strategy".into(), "match".into(), "p".into()]).expect("match");
        assert!(r.1.contains("sk"), "{}", r.1);

        // interactive subcommands stay in Rust — prune is unhandled
        assert!(cli_dispatch(&["strategy".into(), "prune".into()]).is_none());
        assert!(cli_dispatch(&["unknown".into()]).is_none());

        std::env::remove_var("BADAPPLE_STRATEGIES");
        std::env::remove_var("BADAPPLE_VAULT_KEY");
        std::env::remove_var("BADAPPLE_DATA_DIR");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// protocol.cy engram gate — the handle_frame policy chain in-language:
    /// envelope → verify → truncate → replay → freshness → similarity.
    #[test]
    fn protocol_cy_engram_gate() {
        let mut vm = cy_vm("protocol.cy");
        let j = json_idx(&vm).unwrap();
        let get_s = |v: &Val| match v {
            Val::S(x) => x.to_string(),
            Val::En(_, _, p) => match p.first() {
                Some(Val::S(x)) => x.to_string(),
                _ => String::new(),
            },
            _ => String::new(),
        };
        let field = |v: &Val, k: &str| -> Val {
            if let Val::En(_, _, p) = v {
                if let Some(Val::M(m)) = p.first() {
                    for (ek, ev) in m.borrow().iter() {
                        if let Val::S(s) = ek {
                            if &**s == k {
                                return ev.clone();
                            }
                        }
                    }
                }
            }
            Val::B(false)
        };

        // build a signed packet through the Rust signer, wrap in the legacy
        // (unencrypted) envelope path so the gate sees a SignedPacket.
        let secret = b"test-secret-32bytes-of-entropy!!!1";
        let compact = serde_json::json!({
            "id": 7, "timestamp": 1000u64,
            "experiential_text": "x".repeat(1500),
            "emotional_state_snapshot": "calm",
            "origin_instance": "peer-a",
            "brain_state": [0.1, 0.2],
            "embedding": [1.0, 0.0],
            "priority": 5
        });
        let payload = serde_json::to_vec(&compact).unwrap();
        let signed = crate::protocol::sign_packet("peer-a", &payload, secret);
        let frame = serde_json::to_string(&signed).unwrap();

        let secret_hex = str_arg(&hex::encode(secret));
        let state0 = vm.call_fn("gate_state_new", vec![]).expect("state_new");
        // no goals → gate open → accept (sync bypass not needed)
        let r = vm
            .call_fn(
                "engram_gate",
                vec![
                    str_arg(&frame),
                    secret_hex.clone(),
                    str_arg(&hex::encode(b"k")),
                    state0.clone(),
                    Val::I(1000),
                ],
            )
            .expect("gate");
        assert_eq!(
            get_s(&field(&r, "action")),
            "accept",
            "gate result: {}",
            serde_json::to_string(&json_val(&r).unwrap_or_default()).unwrap_or_default()
        );
        // truncation happened — text is now 1024 chars
        let pkt = field(&r, "packet");
        let txt = field(&pkt, "experiential_text");
        assert_eq!(get_s(&txt).len(), 1024);
        let st = field(&r, "state");

        // replay: same id again → drop(replay)
        let r2 = vm
            .call_fn(
                "engram_gate",
                vec![
                    str_arg(&frame),
                    secret_hex.clone(),
                    str_arg(&hex::encode(b"k")),
                    st.clone(),
                    Val::I(1000),
                ],
            )
            .unwrap();
        assert_eq!(get_s(&field(&r2, "action")), "drop");
        assert_eq!(get_s(&field(&r2, "reason")), "replay");

        // stale: timestamp older than 300 s → drop(stale)
        let stale = serde_json::json!({
            "id": 8, "timestamp": 100u64, "experiential_text": "x",
            "emotional_state_snapshot": "", "origin_instance": "peer-a",
            "brain_state": [], "embedding": [1.0], "priority": 5
        });
        let sframe = serde_json::to_string(&crate::protocol::sign_packet(
            "peer-a",
            &serde_json::to_vec(&stale).unwrap(),
            secret,
        ))
        .unwrap();
        let r3 = vm
            .call_fn(
                "engram_gate",
                vec![
                    str_arg(&sframe),
                    secret_hex.clone(),
                    str_arg(&hex::encode(b"k")),
                    st.clone(),
                    Val::I(1000),
                ],
            )
            .unwrap();
        assert_eq!(get_s(&field(&r3, "reason")), "stale");

        // tampered signature → drop
        let mut bad = signed.clone();
        bad.signature_hex = "00".repeat(32);
        let bf = serde_json::to_string(&bad).unwrap();
        let r4 = vm
            .call_fn(
                "engram_gate",
                vec![
                    str_arg(&bf),
                    secret_hex.clone(),
                    str_arg(&hex::encode(b"k")),
                    state0.clone(),
                    Val::I(1000),
                ],
            )
            .unwrap();
        assert_eq!(get_s(&field(&r4, "reason")), "tampered/unsigned");

        // similarity gate with goals: orthogonal embedding → drop
        let goals = json_to_val(&serde_json::json!([[0.0, 1.0]]), &j);
        let mut stg = gate_state_new_vm(&mut vm);
        stg = vm
            .call_fn("gate_state_set", vec![stg, str_arg("goals"), goals])
            .unwrap();
        let emb = serde_json::json!({
            "id": 9, "timestamp": 1000u64, "experiential_text": "x",
            "emotional_state_snapshot": "", "origin_instance": "peer-a",
            "brain_state": [], "embedding": [1.0, 0.0], "priority": 5
        });
        let eframe = serde_json::to_string(&crate::protocol::sign_packet(
            "peer-a",
            &serde_json::to_vec(&emb).unwrap(),
            secret,
        ))
        .unwrap();
        let r5 = vm
            .call_fn(
                "engram_gate",
                vec![
                    str_arg(&eframe),
                    secret_hex.clone(),
                    str_arg(&hex::encode(b"k")),
                    stg,
                    Val::I(1000),
                ],
            )
            .unwrap();
        assert_eq!(get_s(&field(&r5, "action")), "drop");

        // parse_peer_spec — SSRF + transports
        for (spec, want_addr, want_t) in [
            ("ws://host:1234", "host:1234", "websocket"),
            ("wss://host:9", "host:9", "websocket"),
            ("host:5", "host:5", "tcp"),
            ("169.254.169.254:8080", "", "tcp"),
        ] {
            let r = vm.call_fn("parse_peer_spec", vec![str_arg(spec)]).unwrap();
            assert_eq!(get_s(&field(&r, "addr")), want_addr, "spec {spec}");
            assert_eq!(get_s(&field(&r, "transport")), want_t, "spec {spec}");
        }
    }

    fn gate_state_new_vm(vm: &mut Vm) -> Val {
        vm.call_fn("gate_state_new", vec![])
            .expect("gate_state_new")
    }

    /// mcp.cy JSON-RPC request policy — the mcp.rs test cases verbatim.
    #[test]
    fn mcp_cy_request_policy() {
        let mut vm = cy_vm("mcp.cy");
        let has_err = |v: &Val| -> bool {
            if let Val::En(_, _, p) = v {
                if let Some(Val::M(m)) = p.first() {
                    return m
                        .borrow()
                        .iter()
                        .any(|(k, _)| matches!(k, Val::S(s) if &**s == "error"));
                }
            }
            false
        };
        let has_key = |v: &Val, k: &str| -> bool {
            if let Val::En(_, _, p) = v {
                if let Some(Val::M(m)) = p.first() {
                    return m
                        .borrow()
                        .iter()
                        .any(|(ek, _)| matches!(ek, Val::S(s) if &**s == k));
                }
            }
            false
        };

        // malformed + oversized + wrong version → errors
        for line in [
            "not json",
            &"x".repeat(70_000),
            r#"{"jsonrpc":"1.0","method":"initialize","id":1}"#,
        ] {
            let r = vm
                .call_fn("mcp_handle_request", vec![str_arg(line)])
                .unwrap();
            assert!(has_err(&r), "line should error: {line:.40}");
        }

        // initialize → server info
        let r = vm
            .call_fn(
                "mcp_handle_request",
                vec![str_arg(r#"{"jsonrpc":"2.0","method":"initialize","id":1}"#)],
            )
            .unwrap();
        assert!(has_key(&r, "result"));

        // tools/list → 3 tools
        let r = vm
            .call_fn(
                "mcp_handle_request",
                vec![str_arg(r#"{"jsonrpc":"2.0","method":"tools/list","id":2}"#)],
            )
            .unwrap();
        assert!(has_key(&r, "result"));

        // unknown method → -32601
        let r = vm
            .call_fn(
                "mcp_handle_request",
                vec![str_arg(r#"{"jsonrpc":"2.0","method":"nope","id":3}"#)],
            )
            .unwrap();
        assert!(has_err(&r));

        // dangerous tool call → daemon approval error
        let r = vm
            .call_fn(
                "mcp_handle_request",
                vec![str_arg(
                    r#"{"jsonrpc":"2.0","method":"tools/call","id":4,"params":{"name":"run_shell","arguments":{}}}"#,
                )],
            )
            .unwrap();
        assert!(has_err(&r));

        // invoke_tool inner-name re-gate: invoke_tool{run_shell} must fail
        let r = vm
            .call_fn(
                "mcp_handle_request",
                vec![str_arg(
                    r#"{"jsonrpc":"2.0","method":"tools/call","id":5,"params":{"name":"invoke_tool","arguments":{"name":"run_shell","args":{}}}}"#,
                )],
            )
            .unwrap();
        assert!(has_err(&r), "inner dangerous tool must be re-gated");

        // unexposed tool → error
        let r = vm
            .call_fn(
                "mcp_handle_request",
                vec![str_arg(
                    r#"{"jsonrpc":"2.0","method":"tools/call","id":6,"params":{"name":"set_workspace","arguments":{}}}"#,
                )],
            )
            .unwrap();
        assert!(has_err(&r));

        // over-deep arguments → error (10 levels of nesting)
        let mut deep = String::from(
            r#"{"jsonrpc":"2.0","method":"tools/call","id":7,"params":{"name":"runtime_status","arguments":{"a":"#,
        );
        for _ in 0..12 {
            deep.push_str("{\"b\":");
        }
        deep.push('1');
        for _ in 0..12 {
            deep.push('}');
        }
        deep.push_str("}}}");
        let r = vm
            .call_fn("mcp_handle_request", vec![str_arg(&deep)])
            .unwrap();
        assert!(has_err(&r), "deep arguments should fail: {deep:.80}");
    }

    /// benchmark.cy — suite lifecycle, validate math, mastery, report.
    /// Parity with the Rust BenchmarkSuite semantics.
    #[test]
    fn benchmark_cy_suite_and_validate() {
        let mut vm = cy_vm("benchmark.cy");
        let get_f = |v: &Val| match v {
            Val::F(x) => *x,
            _ => -1.0,
        };
        let get_s = |v: &Val| match v {
            Val::S(x) => x.to_string(),
            _ => String::new(),
        };

        // fresh suite: next_task cycles the table, index advances
        let suite0 = vm
            .call_fn("bench_new_suite", vec![str_arg("basic")])
            .unwrap();
        let n1 = vm
            .call_fn("bench_next", vec![suite0.clone()])
            .expect("next");
        // bench_next returns {"task","suite"} — pull the suite out
        let j = json_idx(&vm).unwrap();
        let grab = |v: &Val, k: &str| -> Val {
            if let Val::En(_, _, p) = v {
                if let Some(Val::M(m)) = p.first() {
                    for (ek, ev) in m.borrow().iter() {
                        if let Val::S(s) = ek {
                            if &**s == k {
                                return ev.clone();
                            }
                        }
                    }
                }
            }
            Val::B(false)
        };
        let suite1 = grab(&n1, "suite");

        // record 3 passes + 1 fail → 75.0
        let mut suite = suite1;
        for ok in [true, true, true, false] {
            suite = vm
                .call_fn("bench_record", vec![suite, str_arg("t"), Val::B(ok)])
                .expect("record");
        }
        let score = vm.call_fn("bench_score", vec![suite]).unwrap();
        assert!((get_f(&score) - 75.0).abs() < 0.001);

        // validate parity — numeric with tolerance, string exact
        let r = vm
            .call_fn(
                "bench_validate",
                vec![str_arg("763"), str_arg("the answer is 763"), Val::F(0.001)],
            )
            .unwrap();
        assert!(matches!(r, Val::B(true)));
        let r = vm
            .call_fn(
                "bench_validate",
                vec![str_arg("763"), str_arg("762"), Val::F(0.001)],
            )
            .unwrap();
        assert!(matches!(r, Val::B(false)));
        let r = vm
            .call_fn(
                "bench_validate",
                vec![str_arg("True"), str_arg("result: True"), Val::F(-1.0)],
            )
            .unwrap();
        assert!(matches!(r, Val::B(true)));
        let r = vm
            .call_fn(
                "bench_validate",
                vec![str_arg("True"), str_arg("result: False"), Val::F(-1.0)],
            )
            .unwrap();
        assert!(matches!(r, Val::B(false)));

        // report — runs Json → markdown with the table row
        let runs = json_to_val(
            &serde_json::json!([
                {"domain": "vowel counting", "success": true, "latency_us": 100,
                 "peak_rss_mb": 50.0, "token_count": 30, "leak_score": 0.0},
                {"domain": "caesar cipher shift 3", "success": false, "latency_us": 200,
                 "peak_rss_mb": 60.0, "token_count": 40, "leak_score": 0.0}
            ]),
            &j,
        );
        let rep = vm.call_fn("bench_report", vec![runs]).unwrap();
        let md = get_s(&rep);
        assert!(md.contains("Total runs: 2"), "report:\n{md}");
        assert!(md.contains("Passed: 1 / 2"), "report:\n{md}");
        assert!(md.contains("| vowel counting | PASS |"), "report:\n{md}");
        assert!(
            md.contains("| caesar cipher shift 3 | FAIL |"),
            "report:\n{md}"
        );
    }

    /// blueprint.cy — the causal graph + governor port. Mirrors the Rust
    /// causal_tests/thermal_tests verbatim, plus persistence and the
    /// broadcast threshold gate.
    #[test]
    fn blueprint_cy_causal_and_thermal() {
        let dir = std::env::temp_dir().join(format!("blueprintcy_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::env::set_var("BADAPPLE_DATA_DIR", &dir);
        let mut vm = cy_vm("blueprint.cy");

        let get_s = |v: &Val| -> String {
            match v {
                Val::S(x) => x.to_string(),
                _ => String::new(),
            }
        };

        // — causal_tests::explains_vram_admission_denied
        let g0 = vm.call_fn("cg_default", vec![]).unwrap();
        let prim = vm
            .call_fn(
                "cg_find_broken",
                vec![g0, str_arg("VRAM admission denied: Not enough free memory")],
            )
            .expect("find_broken");
        assert_eq!(get_s(&prim), "vram");
        let g1 = vm.call_fn("cg_default", vec![]).unwrap();
        let chain = vm.call_fn("cg_explain", vec![g1, prim]).expect("explain");
        let chain = get_s(&chain);
        assert!(chain.contains("memory_pressure"), "chain: {chain}");
        assert!(chain.contains("stale_engine_process"), "chain: {chain}");

        // — explains_socket_contention
        let g = vm.call_fn("cg_default", vec![]).unwrap();
        let prim2 = vm
            .call_fn(
                "cg_find_broken",
                vec![
                    g.clone(),
                    str_arg("error binding socket: address already in use"),
                ],
            )
            .unwrap();
        assert_eq!(get_s(&prim2), "socket");
        let chain2 = get_s(&vm.call_fn("cg_explain", vec![g, prim2]).unwrap());
        assert!(chain2.contains("stale_engine_process"), "chain: {chain2}");
        assert!(
            chain2.contains("DEPENDS_ON")
                || chain2.contains("REQUIRED BY")
                || chain2.contains("VIOLATES")
                || chain2.contains("BLOCKED BY"),
            "chain: {chain2}"
        );

        // — explains_identity_agent_failure
        let g = vm.call_fn("cg_default", vec![]).unwrap();
        let prim3 = vm
            .call_fn(
                "cg_find_broken",
                vec![g.clone(), str_arg("identity_agent_not_loaded")],
            )
            .unwrap();
        assert_eq!(get_s(&prim3), "identity");
        let chain3 = get_s(&vm.call_fn("cg_explain", vec![g, prim3]).unwrap());
        assert!(chain3.contains("launch_plist"), "chain: {chain3}");

        // — specific_terms_win_over_generic
        let g = vm.call_fn("cg_default", vec![]).unwrap();
        let prim4 = vm
            .call_fn(
                "cg_find_broken",
                vec![g, str_arg("speech recognition permission denied")],
            )
            .unwrap();
        assert_eq!(get_s(&prim4), "speech");

        // — unknown_failures_report_isolation
        let g = vm.call_fn("cg_default", vec![]).unwrap();
        let prim5 = vm
            .call_fn("cg_find_broken", vec![g, str_arg("quantum flux inverter")])
            .unwrap();
        assert_eq!(get_s(&prim5), "");

        // persistence: relate survives a reload
        assert!(matches!(
            vm.call_fn(
                "cg_relate",
                vec![str_arg("x_src"), str_arg("ENABLES"), str_arg("x_dst")]
            ),
            Ok(Val::B(true))
        ));
        let loaded = vm.call_fn("cg_load", vec![]).unwrap();
        let found = vm
            .call_fn("cg_find_broken", vec![loaded, str_arg("x_src broke")])
            .unwrap();
        assert_eq!(get_s(&found), "x_src");

        // — thermal_tests::governor_throttles_under_stress
        let j = json_idx(&vm).unwrap();
        let obj = |pairs: Vec<(&str, Val)>| -> Val {
            let m: Vec<(Val, Val)> = pairs
                .into_iter()
                .map(|(k, v)| (Val::S(Rc::from(k)), v))
                .collect();
            Val::En(j.en, j.obj, Rc::new(vec![Val::M(Rc::new(RefCell::new(m)))]))
        };
        let num = |f: f64| Val::En(j.en, j.num, Rc::new(vec![Val::F(f)]));
        let hot = obj(vec![
            ("cpu_usage", num(0.95)),
            ("memory_pressure", num(0.95)),
            ("battery_health", num(0.05)),
            ("cpu_temp", num(0.9)),
        ]);
        let gv = vm.call_fn("governor_update", vec![hot]).unwrap();
        // pull stress/throttle out of the Json
        let mut throttle = false;
        let mut stress = 0.0f64;
        let mut sparsity = 0.0f64;
        if let Val::En(_, _, p) = &gv {
            if let Some(Val::M(m)) = p.first() {
                for (k, v) in m.borrow().iter() {
                    if let Val::S(kk) = k {
                        match &**kk {
                            "throttle" => {
                                if let Val::En(_, _, pp) = v {
                                    if let Some(Val::B(b)) = pp.first() {
                                        throttle = *b;
                                    }
                                }
                            }
                            "stress" | "target_sparsity" => {
                                if let Val::En(_, _, pp) = v {
                                    if let Some(Val::F(f)) = pp.first() {
                                        if &**kk == "stress" {
                                            stress = *f;
                                        } else {
                                            sparsity = *f;
                                        }
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }
        }
        assert!(throttle, "expected throttle under stress");
        assert!(sparsity > 0.5, "sparsity {sparsity}");
        let lr = vm
            .call_fn("throttle_lr", vec![Val::F(0.001), Val::F(stress)])
            .expect("throttle_lr");
        assert!(matches!(lr, Val::F(x) if x < 0.0005));

        // — governor_is_calm_when_idle
        let calm = obj(vec![
            ("cpu_usage", num(0.0)),
            ("memory_pressure", num(0.0)),
            ("battery_health", num(1.0)),
            ("cpu_temp", num(0.0)),
        ]);
        let gv2 = vm.call_fn("governor_update", vec![calm]).unwrap();
        if let Val::En(_, _, p) = &gv2 {
            if let Some(Val::M(m)) = p.first() {
                for (k, v) in m.borrow().iter() {
                    if let Val::S(kk) = k {
                        if &**kk == "throttle" {
                            if let Val::En(_, _, pp) = v {
                                assert!(matches!(pp.first(), Some(Val::B(false))));
                            }
                        }
                        if &**kk == "target_sparsity" {
                            if let Val::En(_, _, pp) = v {
                                assert!(matches!(pp.first(), Some(Val::F(f)) if *f == 0.0));
                            }
                        }
                    }
                }
            }
        }

        // — broadcast threshold gate
        let mut sigs: Vec<(Val, Val)> = Vec::new();
        sigs.push((
            Val::S(Rc::from("a")),
            Val::En(
                j.en,
                j.arr,
                Rc::new(vec![Val::V(Rc::new(RefCell::new(vec![
                    Val::En(j.en, j.strv, Rc::new(vec![Val::S(Rc::from("low"))])),
                    num(0.5),
                ])))]),
            ),
        ));
        sigs.push((
            Val::S(Rc::from("b")),
            Val::En(
                j.en,
                j.arr,
                Rc::new(vec![Val::V(Rc::new(RefCell::new(vec![
                    Val::En(j.en, j.strv, Rc::new(vec![Val::S(Rc::from("winning"))])),
                    num(0.9),
                ])))]),
            ),
        ));
        let winner = vm
            .call_fn(
                "select_broadcast",
                vec![Val::M(Rc::new(RefCell::new(sigs)))],
            )
            .expect("broadcast");
        assert!(matches!(&winner, Val::En(_, _, _)));

        std::env::remove_var("BADAPPLE_DATA_DIR");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// cert.cy: run the self-contained checks through the VM — crypto,
    /// replay cache, mesh plan/shard, loopback frames, dead-peer.
    #[test]
    fn cert_cy_self_contained_checks_pass() {
        let mut vm = cy_vm("cert.cy");
        let names = [
            "check_tampered_ciphertext",
            "check_wrong_key",
            "check_replay_cache",
            "check_cage_traversal",
            "check_cage_symlink",
            "check_mesh_plan",
            "check_mesh_shard",
            "check_mesh_frames_no_plaintext",
            "check_mesh_auth_wrong_key",
            "check_mesh_dead_peer",
        ];
        for name in names {
            let r = vm
                .call_fn(name, vec![])
                .unwrap_or_else(|e| panic!("{name}: {e}"));
            let Val::En(_, _, p) = &r else {
                panic!("{name}: expected Json")
            };
            let Some(Val::M(m)) = p.first() else {
                panic!("{name}: expected Obj")
            };
            let m = m.borrow();
            let passed = m
                .iter()
                .find(|(k, _)| matches!(k, Val::S(s) if &**s == "passed"))
                .and_then(|(_, v)| {
                    if let Val::En(_, _, pp) = v {
                        if let Some(Val::B(b)) = pp.first() {
                            return Some(*b);
                        }
                    }
                    None
                });
            let msg = m
                .iter()
                .find(|(k, _)| matches!(k, Val::S(s) if &**s == "message"))
                .and_then(|(_, v)| {
                    if let Val::En(_, _, pp) = v {
                        if let Some(Val::S(x)) = pp.first() {
                            return Some(x.to_string());
                        }
                    }
                    None
                })
                .unwrap_or_default();
            assert_eq!(passed, Some(true), "{name} failed: {msg}");
        }
    }

    #[test]
    fn mil_cy_emits_spec_bytes() {
        let mut vm = cy_module_vm("mil.cy").expect("mil.cy loads");
        let spec = vm.call_fn("mil_demo_spec", vec![]).expect("mil_demo_spec");
        let emit = |v: &mut Vm, spec: Val| -> Vec<u8> {
            let r = v.call_fn("mil_build", vec![spec]).expect("mil_build");
            let j = json_idx(v).unwrap();
            match val_to_json(&r, &j).unwrap_or_default() {
                serde_json::Value::Object(o) => match &o["spec"] {
                    serde_json::Value::String(hx) => hex::decode(hx).expect("hex"),
                    _ => panic!("mil_build: missing spec in {o:?}"),
                },
                other => panic!("mil_build error: {other:?}"),
            }
        };
        let a = emit(&mut vm, spec.clone());
        let b = emit(&mut vm, spec);
        assert!(!a.is_empty(), "empty spec bytes");
        assert_eq!(a, b, "mil emit must be deterministic");
    }

    #[test]
    fn mil_cy_linear_parity() {
        use mil_spec::{
            encode_model, Block, DType, Feature, ModelMeta, TensorType, ValueType, NVT,
        };
        let (d, cout, sq): (i64, i64, i64) = (8, 16, 3);
        // Rust path: the same op sequence linear_w emits.
        let mut blk = Block::new();
        let xr = blk.reshape("x", &[1, 1, d, sq], "lin_xr");
        let wr = blk.reshape("w", &[1, 1, cout, d], "lin_wr");
        let mm = blk.matmul(&wr, &xr, false, &[1, 1, cout, sq], "lin");
        let ob = blk.reshape(&mm, &[1, cout, 1, sq], "lin_b");
        blk.outputs = vec![ob];
        let feat = |name: &str, shape: &[i64]| Feature {
            name: name.into(),
            dtype: DType::Fp16,
            shape: shape.to_vec(),
            is_state: false,
        };
        let nvt = |name: &str, shape: &[i64]| NVT {
            name: name.into(),
            ty: ValueType::Tensor(TensorType {
                dtype: DType::Fp16,
                shape: shape.to_vec(),
            }),
        };
        let inputs = [feat("x", &[1, d, 1, sq]), feat("w", &[cout, d, 1, 1])];
        let outputs = [feat("lin_b", &[1, cout, 1, sq])];
        let fn_in = [nvt("x", &[1, d, 1, sq]), nvt("w", &[cout, d, 1, 1])];
        let meta = ModelMeta::new(9, "CoreML9")
            .creator("badapple-coyote")
            .description("authored by Coyote");
        let want = encode_model(&inputs, &outputs, &[], &blk, &fn_in, &meta);

        // Coyote path: mil_linear_spec emits the same graph.
        let mut vm = cy_module_vm("mil.cy").expect("mil.cy loads");
        let spec = vm
            .call_fn("mil_linear_spec", vec![Val::I(d), Val::I(cout), Val::I(sq)])
            .expect("mil_linear_spec");
        let r = vm.call_fn("mil_build", vec![spec]).expect("mil_build");
        let j = json_idx(&vm).unwrap();
        let got = match val_to_json(&r, &j).unwrap_or_default() {
            serde_json::Value::Object(o) => match &o["spec"] {
                serde_json::Value::String(hx) => hex::decode(hx).expect("hex"),
                _ => panic!("mil_build: missing spec in {o:?}"),
            },
            other => panic!("mil_build error: {other:?}"),
        };
        assert_eq!(got, want, "coyote-emitted spec must equal the Rust emitter");
    }

    #[test]
    fn mil_cy_rms_norm_parity() {
        use mil_spec::{
            bind, encode_model, Block, DType, Feature, ModelMeta, TensorType, ValueType,
        };
        let d: i64 = 16;
        let eps: f32 = 1e-5;
        let sq: i64 = 2;
        let pfx = "l0_fn";

        // Rust path — rms_norm_big's op sequence verbatim.
        let mut blk = Block::new();
        const C: f32 = 8.0;
        let k = (d as f32).sqrt();
        let shape4: Vec<i64> = vec![1, d, 1, sq];
        let mshape: Vec<i64> = vec![1, 1, 1, sq];
        let inv = blk.konst_f16(&format!("{pfx}_bigik"), 1.0 / (C * k));
        let xs = blk.mul("x", &inv, &shape4, &format!("{pfx}_bigxs"));
        let sqv = blk.mul(&xs, &xs, &shape4, &format!("{pfx}_bigsq"));
        let axes = blk.konst_i32(&format!("{pfx}_bigax"), &[1]);
        let kd = blk.konst_bool(&format!("{pfx}_bigkd"), true);
        let mean = blk.o1(
            "reduce_mean",
            vec![
                ("x".into(), bind(&sqv).1),
                ("axes".into(), bind(&axes).1),
                ("keep_dims".into(), bind(&kd).1),
            ],
            &format!("{pfx}_bigm"),
            ValueType::Tensor(TensorType::f16(&mshape)),
        );
        let e = blk.konst_f16(&format!("{pfx}_bige"), eps / (C * C * k * k));
        let vp = blk.add(&mean, &e, &mshape, &format!("{pfx}_bigvp"));
        let fl = blk.konst_f16(&format!("{pfx}_bigfl"), 1e-7);
        let vm = blk.o1(
            "maximum",
            vec![("x".into(), bind(&vp).1), ("y".into(), bind(&fl).1)],
            &format!("{pfx}_bigvm"),
            ValueType::Tensor(TensorType::f16(&mshape)),
        );
        let z = blk.konst_f16(&format!("{pfx}_bigz"), 0.0);
        let r = blk.o1(
            "rsqrt",
            vec![("x".into(), bind(&vm).1), ("epsilon".into(), bind(&z).1)],
            &format!("{pfx}_bigr"),
            ValueType::Tensor(TensorType::f16(&mshape)),
        );
        let xn = blk.mul(&xs, &r, &shape4, &format!("{pfx}_xn"));
        let oname = blk.mul(&xn, "w", &shape4, &format!("{pfx}_out"));
        blk.outputs = vec![oname];

        let feat = |name: &str, shape: &[i64]| Feature {
            name: name.into(),
            dtype: DType::Fp16,
            shape: shape.to_vec(),
            is_state: false,
        };
        let inputs = [feat("x", &[1, d, 1, sq]), feat("w", &[1, d, 1, 1])];
        let outputs = [feat(&format!("{pfx}_out"), &shape4)];
        let meta = ModelMeta::new(9, "CoreML9")
            .creator("badapple-coyote")
            .description("authored by Coyote");
        let want = encode_model(&inputs, &outputs, &[], &blk, &[], &meta);

        let mut vm = cy_module_vm("mil.cy").expect("mil.cy loads");
        let spec = vm
            .call_fn(
                "mil_rms_norm_spec",
                vec![Val::I(d), Val::F(eps as f64), Val::I(sq), s(pfx)],
            )
            .expect("mil_rms_norm_spec");
        let r = vm.call_fn("mil_build", vec![spec]).expect("mil_build");
        let j = json_idx(&vm).unwrap();
        let got = match val_to_json(&r, &j).unwrap_or_default() {
            serde_json::Value::Object(o) => match &o["spec"] {
                serde_json::Value::String(hx) => hex::decode(hx).expect("hex"),
                _ => panic!("mil_build: missing spec in {o:?}"),
            },
            other => panic!("mil_build error: {other:?}"),
        };
        assert_eq!(
            got, want,
            "coyote rms_norm spec must equal the Rust emitter"
        );
    }
}
