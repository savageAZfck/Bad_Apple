//! badapple-treaty — inter-organism treaties for the organism.
//!
//! Thin wrapper over the public `wolakota` crate. Manages consent
//! contracts between this organism's owner key and foreign owner keys:
//! propose, ratify, check scope, record invocation, revoke. The treaty
//! book is hash-chained — both parties can hold it and verify offline.
//!
//! State under /var/lib/bad_apple/treaties/:
//!   identity.key  — this organism's treaty identity (ed25519, 0600)
//!   book.json     — the hash-chained treaty book
//!
//!   badapple-treaty init
//!   badapple-treaty propose <counterparty-pubkey> --scope <name> [--scope ...] --terms <text> [--ttl <secs>]
//!   badapple-treaty ratify <treaty-file>
//!   badapple-treaty state <treaty-id>
//!   badapple-treaty invoke <treaty-id> --scope <name>
//!   badapple-treaty revoke <treaty-id> --reason <text>
//!   badapple-treaty verify
//!   badapple-treaty export [treaty-id]

use anyhow::{bail, Context, Result};
use std::fs;
use std::path::PathBuf;
use wolakota::{Identity, Revocation, Scope, Treaty, TreatyBook, TreatyState};

const ROOT: &str = "/var/lib/bad_apple/treaties";

fn root() -> PathBuf {
    std::env::var("BADAPPLE_TREATY_DIR")
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

fn load_book() -> Result<TreatyBook> {
    let path = root().join("book.json");
    if path.exists() {
        Ok(serde_json::from_str(&fs::read_to_string(&path)?)?)
    } else {
        Ok(TreatyBook::new())
    }
}

fn save_book(book: &TreatyBook) -> Result<()> {
    fs::create_dir_all(&root())?;
    fs::write(root().join("book.json"), serde_json::to_string_pretty(book)?)?;
    Ok(())
}

fn now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn parse_scope(s: &str) -> Result<Scope> {
    Ok(match s {
        "share_dreams" => Scope::ShareDreams,
        _ if s.starts_with("inference:") => Scope::Inference {
            model: s.trim_start_matches("inference:").to_string(),
        },
        _ if s.starts_with("read:") => Scope::Read {
            resource: s.trim_start_matches("read:").to_string(),
        },
        _ if s.starts_with("custody:") => Scope::Custody {
            ceremony: s.trim_start_matches("custody:").to_string(),
        },
        _ => Scope::Custom { name: s.into() },
    })
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        bail!(
            "usage: badapple-treaty <init|propose|ratify|state|invoke|revoke|verify|export>"
        );
    }
    let named = |flag: &str| -> Option<String> {
        args.iter()
            .position(|a| a == flag)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    let scopes_of = || -> Result<Vec<Scope>> {
        args.iter()
            .enumerate()
            .filter(|(_, a)| *a == "--scope")
            .map(|(i, _)| {
                let s = args.get(i + 1).context("--scope needs a value")?;
                parse_scope(s)
            })
            .collect()
    };

    match args[0].as_str() {
        "init" => {
            let id = load_identity()?;
            save_book(&load_book().unwrap_or_default())?;
            println!(
                "{}",
                serde_json::json!({"status":"initialized","identity":id.public_key()})
            );
        }
        "propose" => {
            let id = load_identity()?;
            let mut book = load_book()?;
            let counterparty = args.get(1).context("propose <counterparty-pubkey>")?;
            let scopes = scopes_of()?;
            if scopes.is_empty() {
                bail!("at least one --scope is required");
            }
            let terms = named("--terms").unwrap_or_else(|| "federation treaty".into());
            let ttl: u64 = named("--ttl").and_then(|s| s.parse().ok()).unwrap_or(0);
            let treaty = Treaty::propose(&id, counterparty, scopes, &terms, ttl);
            book.record_proposal(treaty.clone())?;
            save_book(&book)?;
            let out = root().join(format!("proposed-{}.json", &treaty.id[..16]));
            fs::write(&out, serde_json::to_string_pretty(&treaty)?)?;
            println!(
                "{}",
                serde_json::json!({"status":"proposed","id":treaty.id,"file":out})
            );
        }
        "ratify" => {
            let id = load_identity()?;
            let mut book = load_book()?;
            let file = args.get(1).context("ratify <treaty-file>")?;
            let mut treaty: Treaty =
                serde_json::from_str(&fs::read_to_string(file)?)?;
            treaty.ratify(&id)?;
            if book.treaties.iter().any(|t| t.id == treaty.id) {
                book.apply_ratification(treaty.clone())?;
            } else {
                book.record_proposal(treaty.clone())?;
                book.apply_ratification(treaty.clone())?;
            }
            save_book(&book)?;
            let out = root().join(format!("ratified-{}.json", &treaty.id[..16]));
            fs::write(&out, serde_json::to_string_pretty(&treaty)?)?;
            println!(
                "{}",
                serde_json::json!({"status":"ratified","id":treaty.id,"file":out})
            );
        }
        "state" => {
            let book = load_book()?;
            let id = args.get(1).context("state <treaty-id>")?;
            let treaty = book
                .treaties
                .iter()
                .find(|t| t.id == *id || t.id.starts_with(id.as_str()))
                .context("no such treaty")?;
            println!(
                "{}",
                serde_json::json!({
                    "id": treaty.id,
                    "state": format!("{:?}", book.state(&treaty.id, now())),
                    "scopes": treaty.scopes,
                    "initiator": treaty.initiator,
                    "counterparty": treaty.counterparty,
                })
            );
        }
        "invoke" => {
            let mut book = load_book()?;
            let id = args.get(1).context("invoke <treaty-id>")?;
            let treaty = book
                .treaties
                .iter()
                .find(|t| t.id == *id || t.id.starts_with(id.as_str()))
                .context("no such treaty")?
                .clone();
            let scopes = scopes_of()?;
            let scope = scopes.first().context("invoke requires --scope")?;
            book.record_invocation(&treaty.id, scope, now())?;
            save_book(&book)?;
            println!(
                "{}",
                serde_json::json!({"status":"invoked","treaty":treaty.id,"scope":scope})
            );
        }
        "revoke" => {
            let id = load_identity()?;
            let mut book = load_book()?;
            let tid = args.get(1).context("revoke <treaty-id>")?;
            let treaty = book
                .treaties
                .iter()
                .find(|t| t.id == *tid || t.id.starts_with(tid.as_str()))
                .context("no such treaty")?
                .clone();
            let reason = named("--reason").unwrap_or_else(|| "owner exit".into());
            let rev = Revocation::issue(&id, &treaty, &reason);
            book.apply_revocation(rev.clone())?;
            save_book(&book)?;
            let out = root().join(format!("revocation-{}.json", &treaty.id[..16]));
            fs::write(&out, serde_json::to_string_pretty(&rev)?)?;
            println!(
                "{}",
                serde_json::json!({"status":"revoked","treaty":treaty.id,"by":rev.revoked_by})
            );
        }
        "verify" => {
            let book = load_book()?;
            println!(
                "{}",
                serde_json::json!({
                    "chain_ok": book.verify_chain(),
                    "treaties": book.treaties.len(),
                    "revocations": book.revocations.len(),
                    "events": book.events.len(),
                })
            );
            if !book.verify_chain() {
                std::process::exit(1);
            }
        }
        "export" => {
            let book = load_book()?;
            if let Some(tid) = args.get(1) {
                let treaty = book
                    .treaties
                    .iter()
                    .find(|t| t.id == *tid || t.id.starts_with(tid.as_str()))
                    .context("no such treaty")?;
                println!("{}", serde_json::to_string_pretty(treaty)?);
            } else {
                println!("{}", serde_json::to_string_pretty(&book)?);
            }
        }
        other => bail!("unknown subcommand: {other}"),
    }
    // Silence unused-import lint for TreatyState when the compiler sees only
    // Debug formatting through it.
    let _ = TreatyState::Proposed;
    Ok(())
}
