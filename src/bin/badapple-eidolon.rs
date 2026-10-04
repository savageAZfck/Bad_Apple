//! badapple-eidolon — counterfactual replay for the organism.
//!
//! Thin wrapper over the public `eidolon` crate. Answers "what would you
//! have done?" with a signed, independently reproducible diff: replay a
//! slice of the audit ledger under an alternate policy and prove where
//! the outcomes diverge.
//!
//! State under /var/lib/bad_apple/eidolon/:
//!   identity.key — report-signing identity (ed25519, 0600)
//!
//!   badapple-eidolon init
//!   badapple-eidolon replay [--file <ledger.jsonl>] [--from <n>] [--to <n>]
//!       --deny <pattern> [--deny <pattern>...] [--label <name>]
//!     Actual outcome = entry["outcome"] if present else {"allowed":true}.
//!     Alternate denies entries whose tool/command matches any --deny glob.
//!   badapple-eidolon verify <report.json> [--file <ledger.jsonl>] --deny <pattern>...

use anyhow::{bail, Context, Result};
use eidolon::{replay_diff, verify_replay, DiffReport, Identity, ReplayFn};
use serde_json::{json, Value};
use std::fs;
use std::path::PathBuf;

const ROOT: &str = "/var/lib/bad_apple/eidolon";
const DEFAULT_LEDGER: &str = "/var/lib/bad_apple/ledger.jsonl";

fn root() -> PathBuf {
    std::env::var("BADAPPLE_EIDOLON_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| ROOT.into())
}

fn load_identity() -> Result<Identity> {
    let path = root().join("identity.key");
    if path.exists() {
        let hex_key = fs::read_to_string(&path)?.trim().to_string();
        let bytes: [u8; 32] = hex::decode(&hex_key)
            .context("identity.key is not hex")?
            .try_into()
            .map_err(|_| anyhow::anyhow!("identity.key must be 32 bytes"))?;
        return Ok(Identity::from_bytes(&bytes));
    }
    fs::create_dir_all(&root())?;
    let id = Identity::generate();
    fs::write(&path, hex::encode(id.seed()))?;
    Ok(id)
}

fn load_entries(path: &PathBuf, from: Option<usize>, to: Option<usize>) -> Result<Vec<Value>> {
    let text = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let mut entries = Vec::new();
    for (i, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        if let Some(f) = from {
            if i < f {
                continue;
            }
        }
        if let Some(t) = to {
            if i > t {
                break;
            }
        }
        entries.push(serde_json::from_str(line)?);
    }
    Ok(entries)
}

/// Compute the chain tip over the raw file: sha256 of the last line's
/// hash field, falling back to hashing the last line itself.
fn chain_tip(path: &PathBuf) -> Result<String> {
    let text = fs::read_to_string(path)?;
    let last = text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .last()
        .context("empty chain")?;
    let v: Value = serde_json::from_str(last)?;
    if let Some(h) = v.get("hash").and_then(|x| x.as_str()) {
        return Ok(h.to_string());
    }
    use sha2::{Digest, Sha256};
    Ok(hex::encode(Sha256::digest(last.as_bytes())))
}

fn alt_policy(denies: Vec<String>) -> Box<ReplayFn> {
    Box::new(move |e| {
        let hay = format!(
            "{} {}",
            e["tool"].as_str().unwrap_or(""),
            e["cmd"].as_str().unwrap_or("")
        );
        let denied = denies.iter().any(|d| {
            if let Some(prefix) = d.strip_suffix('*') {
                hay.contains(prefix)
            } else {
                hay.contains(d.as_str())
            }
        });
        if denied {
            json!({"allowed": false, "reason": "alt-policy deny"})
        } else {
            json!({"allowed": true})
        }
    })
}

fn actual_outcome() -> Box<ReplayFn> {
    Box::new(|e| {
        e.get("outcome")
            .cloned()
            .unwrap_or_else(|| json!({"allowed": true}))
    })
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        bail!("usage: badapple-eidolon <init|replay|verify>");
    }
    let named = |flag: &str| -> Option<String> {
        args.iter()
            .position(|a| a == flag)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    let all_of = |flag: &str| -> Vec<String> {
        args.iter()
            .enumerate()
            .filter(|(_, a)| *a == flag)
            .filter_map(|(i, _)| args.get(i + 1).cloned())
            .collect()
    };

    match args[0].as_str() {
        "init" => {
            let id = load_identity()?;
            println!(
                "{}",
                serde_json::json!({"status":"initialized","identity":id.public_key()})
            );
        }
        "replay" => {
            let file = named("--file")
                .map(PathBuf::from)
                .unwrap_or_else(|| DEFAULT_LEDGER.into());
            let from = named("--from").and_then(|s| s.parse().ok());
            let to = named("--to").and_then(|s| s.parse().ok());
            let denies = all_of("--deny");
            if denies.is_empty() {
                bail!("at least one --deny <pattern> required");
            }
            let label = named("--label").unwrap_or_else(|| "alt-policy".into());
            let entries = load_entries(&file, from, to)?;
            let tip = chain_tip(&file)?;
            let id = load_identity()?;
            let report = replay_diff(
                &id,
                "ledger",
                &tip,
                &label,
                "recorded-outcome",
                &entries,
                &actual_outcome(),
                &alt_policy(denies),
            )?;
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        "verify" => {
            let path = args.get(1).context("verify <report.json>")?;
            let report: DiffReport = serde_json::from_str(&fs::read_to_string(path)?)?;
            let file = named("--file")
                .map(PathBuf::from)
                .unwrap_or_else(|| DEFAULT_LEDGER.into());
            let from = named("--from").and_then(|s| s.parse().ok());
            let to = named("--to").and_then(|s| s.parse().ok());
            let denies = all_of("--deny");
            let entries = load_entries(&file, from, to)?;
            verify_replay(&report, &entries, &actual_outcome(), &alt_policy(denies))?;
            println!(
                "{}",
                serde_json::json!({"status":"verified","divergences":report.divergences.len()})
            );
        }
        other => bail!("unknown command: {other}"),
    }
    Ok(())
}
