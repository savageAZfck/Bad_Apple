//! badapple-reflex — the organism's audited reflex arc.
//!
//! Thin wrapper over the public `axon` crate. Reflexes fire below the
//! council and the policy engine — they can brake, alert, or quarantine,
//! never steer. Every firing lands on the arc's own hash-chained log.
//!
//! State under /var/lib/bad_apple/reflex/:
//!   arc.json — installed reflexes, firing state, hash-chained arc log
//!
//!   badapple-reflex install <id> --trigger <spec> --action <brake|alert|quarantine>
//!       [--cooldown-ms <n>] [--max-firings <n>]
//!     trigger specs: kind:<glob>  field:<name>=<glob>  gt:<field>=<n>  lt|ge|le:<field>=<n>
//!   badapple-reflex remove <id>
//!   badapple-reflex list
//!   badapple-reflex sense <kind> [--data <json>]
//!   badapple-reflex clear-freeze
//!   badapple-reflex rearm <id>
//!   badapple-reflex status
//!   badapple-reflex verify

use anyhow::{bail, Context, Result};
use axon::{Reflex, ReflexAction, ReflexArc, SenseEvent, ThresholdOp, Trigger};
use serde_json::Value;
use std::fs;
use std::path::PathBuf;

const ROOT: &str = "/var/lib/bad_apple/reflex";

fn root() -> PathBuf {
    std::env::var("BADAPPLE_REFLEX_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| ROOT.into())
}

fn arc_path() -> PathBuf {
    root().join("arc.json")
}

fn load_arc() -> Result<ReflexArc> {
    let path = arc_path();
    if path.exists() {
        Ok(serde_json::from_str(&fs::read_to_string(&path)?)?)
    } else {
        Ok(ReflexArc::new())
    }
}

fn save_arc(arc: &ReflexArc) -> Result<()> {
    fs::create_dir_all(root())?;
    fs::write(arc_path(), serde_json::to_string_pretty(arc)?)?;
    Ok(())
}

fn parse_trigger(spec: &str) -> Result<Trigger> {
    let (kind, rest) = spec.split_once(':').context(
        "trigger spec must be kind:<glob> | field:<name>=<glob> | gt|lt|ge|le:<field>=<n>",
    )?;
    Ok(match kind {
        "kind" => Trigger::KindIs(rest.to_string()),
        "field" => {
            let (name, pattern) = rest.split_once('=').context("field:<name>=<glob>")?;
            Trigger::FieldMatch {
                field: name.to_string(),
                pattern: pattern.to_string(),
            }
        }
        op @ ("gt" | "lt" | "ge" | "le") => {
            let (field, n) = rest.split_once('=').context("op:<field>=<n>")?;
            Trigger::Threshold {
                field: field.to_string(),
                op: match op {
                    "gt" => ThresholdOp::Gt,
                    "lt" => ThresholdOp::Lt,
                    "ge" => ThresholdOp::Ge,
                    _ => ThresholdOp::Le,
                },
                value: n.parse().context("threshold value must be numeric")?,
            }
        }
        other => bail!("unknown trigger kind: {other}"),
    })
}

fn main() -> Result<()> {
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
            let action = match named("--action").as_deref() {
                Some("brake") => ReflexAction::Brake,
                Some("alert") | None => ReflexAction::Alert,
                Some("quarantine") => ReflexAction::Quarantine,
                Some(other) => bail!("unknown action: {other}"),
            };
            let cooldown = named("--cooldown-ms")
                .and_then(|s| s.parse().ok())
                .unwrap_or(0);
            let max = named("--max-firings")
                .and_then(|s| s.parse().ok())
                .unwrap_or(0);
            let mut arc = load_arc()?;
            arc.install(Reflex::new(id, parse_trigger(&spec)?, action).with_bounds(cooldown, max))?;
            save_arc(&arc)?;
            println!("{}", serde_json::json!({"status":"installed","reflex":id}));
        }
        "remove" => {
            let id = args.get(1).context("remove <id>")?;
            let mut arc = load_arc()?;
            if !arc.remove(id) {
                bail!("no such reflex: {id}");
            }
            save_arc(&arc)?;
            println!("{}", serde_json::json!({"status":"removed","reflex":id}));
        }
        "list" => {
            let arc = load_arc()?;
            println!("{}", serde_json::to_string_pretty(&arc.reflexes)?);
        }
        "sense" => {
            let kind = args.get(1).context("sense <kind>")?;
            let data: Value = named("--data")
                .map(|s| serde_json::from_str(&s).unwrap_or(Value::Null))
                .unwrap_or(Value::Null);
            let mut arc = load_arc()?;
            let fired = arc.sense(&SenseEvent::new(kind, data));
            save_arc(&arc)?;
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "fired": fired,
                    "frozen": arc.frozen,
                }))?
            );
        }
        "clear-freeze" => {
            let mut arc = load_arc()?;
            let was = arc.clear_freeze();
            save_arc(&arc)?;
            println!(
                "{}",
                serde_json::json!({"status":"cleared","was_frozen":was})
            );
        }
        "rearm" => {
            let id = args.get(1).context("rearm <id>")?;
            let mut arc = load_arc()?;
            if !arc.rearm(id) {
                bail!("no such reflex: {id}");
            }
            save_arc(&arc)?;
            println!("{}", serde_json::json!({"status":"rearmed","reflex":id}));
        }
        "status" => {
            let arc = load_arc()?;
            println!(
                "{}",
                serde_json::json!({
                    "reflexes": arc.reflexes.len(),
                    "firings": arc.log.len(),
                    "frozen": arc.frozen,
                    "tip": arc.log.last().map(|f| f.hash.clone()),
                })
            );
        }
        "verify" => {
            let arc = load_arc()?;
            arc.verify()?;
            println!(
                "{}",
                serde_json::json!({"status":"verified","firings":arc.log.len()})
            );
        }
        other => bail!("unknown command: {other}"),
    }
    Ok(())
}
