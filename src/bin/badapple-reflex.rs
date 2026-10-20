//! badapple-reflex — the organism's audited reflex arc, running through Coyote on the ANE.
//!
//! Executes `coyote/badapple_organs.cy` via `bad_apple::coyote_organs` on the
//! Apple Neural Engine. Reflexes fire below the council and the policy engine —
//! they can brake, alert, or quarantine, never steer. Every firing lands on the
//! organism's hash-chained witness and arc ledger.

use anyhow::{bail, Context, Result};
use bad_apple::coyote_organs;
use serde_json::{json, Value};

fn sync_env_dir() {
    if std::env::var_os("BADAPPLE_ORGANS_DIR").is_none() {
        if let Some(dir) = std::env::var_os("BADAPPLE_REFLEX_DIR") {
            std::env::set_var("BADAPPLE_ORGANS_DIR", dir);
        }
    }
}

fn check_ok(res: String) -> Result<String> {
    if let Some(err) = res.strip_prefix("error=") {
        bail!("{err}");
    }
    Ok(res)
}

fn main() -> Result<()> {
    sync_env_dir();
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        bail!(
            "usage: badapple-reflex <install|remove|list|sense|clear-freeze|rearm|status|verify>"
        );
    }
    let named = |flag: &str| -> Option<String> {
        args.iter()
            .position(|a| a == flag)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };

    match args[0].as_str() {
        "install" => {
            let id = args.get(1).context("install <id>")?;
            let spec = named("--trigger").context("--trigger required")?;
            let action = named("--action").unwrap_or_else(|| "alert".to_string());
            let cooldown: i64 = named("--cooldown-ms")
                .and_then(|s| s.parse().ok())
                .unwrap_or(0);
            let max: i64 = named("--max-firings")
                .and_then(|s| s.parse().ok())
                .unwrap_or(0);
            check_ok(coyote_organs::arc_install(
                id, &spec, &action, cooldown, max,
            )?)?;
            println!("{}", json!({"status":"installed","reflex":id}));
        }
        "remove" => {
            let id = args.get(1).context("remove <id>")?;
            check_ok(coyote_organs::arc_remove(id)?)?;
            println!("{}", json!({"status":"removed","reflex":id}));
        }
        "list" => {
            let res = coyote_organs::arc_list()?;
            let inner = res
                .strip_prefix("reflexes=[")
                .and_then(|s| s.strip_suffix(']'))
                .unwrap_or("");
            let mut items = Vec::new();
            for row in inner.split(", ").filter(|r| !r.is_empty()) {
                let p: Vec<&str> = row.split('|').collect();
                if p.len() == 8 {
                    items.push(json!({
                        "id": p[1],
                        "trigger": p[2],
                        "action": p[3],
                        "cooldown_ms": p[4].parse::<i64>().unwrap_or(0),
                        "max_firings": p[5].parse::<i64>().unwrap_or(0),
                        "last_fired_ms": p[6].parse::<i64>().unwrap_or(0),
                        "firings": p[7].parse::<i64>().unwrap_or(0),
                    }));
                }
            }
            println!("{}", serde_json::to_string_pretty(&items)?);
        }
        "sense" => {
            let kind = args.get(1).context("sense <kind>")?;
            let data: Value = named("--data")
                .map(|s| serde_json::from_str(&s).unwrap_or(Value::Null))
                .unwrap_or(Value::Null);
            let res = coyote_organs::arc_sense(kind, &data)?;
            let mut fired = Vec::new();
            let mut frozen = false;
            for part in res.split('|') {
                if let Some(f) = part.strip_prefix("fired=") {
                    for entry in f.split(',').filter(|s| !s.is_empty()) {
                        if let Some((rid, act)) = entry.split_once(':') {
                            fired
                                .push(json!({"reflex_id": rid, "action": act, "event_kind": kind}));
                        }
                    }
                } else if let Some(frz) = part.strip_prefix("frozen=") {
                    frozen = frz == "1" || frz == "true";
                }
            }
            println!(
                "{}",
                serde_json::to_string_pretty(&json!({
                    "fired": fired,
                    "frozen": frozen,
                }))?
            );
        }
        "clear-freeze" => {
            let res = coyote_organs::arc_clear_freeze()?;
            let was = res.ends_with("true");
            println!("{}", json!({"status":"cleared","was_frozen":was}));
        }
        "rearm" => {
            let id = args.get(1).context("rearm <id>")?;
            check_ok(coyote_organs::arc_rearm(id)?)?;
            println!("{}", json!({"status":"rearmed","reflex":id}));
        }
        "status" => {
            let res = coyote_organs::arc_status()?;
            let mut reflexes = 0i64;
            let mut firings = 0i64;
            let mut frozen = false;
            let mut tip: Option<String> = None;
            for part in res.split('|') {
                if let Some(v) = part.strip_prefix("reflexes=") {
                    reflexes = v.parse().unwrap_or(0);
                } else if let Some(v) = part.strip_prefix("firings=") {
                    firings = v.parse().unwrap_or(0);
                } else if let Some(v) = part.strip_prefix("frozen=") {
                    frozen = v == "true" || v == "1";
                } else if let Some(v) = part.strip_prefix("tip=") {
                    if v != "genesis" && !v.chars().all(|c| c == '0') {
                        tip = Some(v.to_string());
                    }
                }
            }
            println!(
                "{}",
                json!({
                    "reflexes": reflexes,
                    "firings": firings,
                    "frozen": frozen,
                    "tip": tip,
                })
            );
        }
        "verify" => {
            let res = coyote_organs::arc_verify()?;
            if res.starts_with("BROKEN:") {
                bail!("arc chain broken: {res}");
            }
            let firings: i64 = res
                .strip_prefix("verified=")
                .and_then(|v| v.parse().ok())
                .unwrap_or(0);
            println!("{}", json!({"status":"verified","firings":firings}));
        }
        other => bail!("unknown command: {other}"),
    }
    Ok(())
}
