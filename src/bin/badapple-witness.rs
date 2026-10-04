//! badapple-witness — cross-organism witnessing for the organism.
//!
//! Thin wrapper over the public `kola` crate. This organism's ledger is
//! self-signed — witnessing makes the past unforgeable: peers notarize
//! the chain tip into their own hash-chained journals, and a rewritten
//! history contradicts every attestation they hold.
//!
//! State under /var/lib/bad_apple/witness/:
//!   identity.key  — this organism's witness identity (ed25519, 0600)
//!   journal.json  — the hash-chained witness journal
//!
//!   badapple-witness init
//!   badapple-witness tip [--file <chain.jsonl>] [--chain <id>]
//!   badapple-witness attest <subject-pubkey> <head.json>
//!   badapple-witness record <attestation.json>
//!   badapple-witness audit <subject-pubkey> --chain <id> --history <map.json>
//!   badapple-witness status
//!   badapple-witness verify

use anyhow::{bail, Context, Result};
use kola::{
    audit_against_history, journal_from_json, journal_to_json, Attestation, ChainHead, Identity,
    WitnessJournal,
};
use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

const ROOT: &str = "/var/lib/bad_apple/witness";
const DEFAULT_LEDGER: &str = "/var/lib/bad_apple/ledger.jsonl";

fn root() -> PathBuf {
    std::env::var("BADAPPLE_WITNESS_DIR")
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

fn load_journal() -> Result<WitnessJournal> {
    let path = root().join("journal.json");
    if path.exists() {
        Ok(journal_from_json(&fs::read_to_string(&path)?)?)
    } else {
        let id = load_identity()?;
        Ok(WitnessJournal::new(id.public_key()))
    }
}

fn save_journal(journal: &WitnessJournal) -> Result<()> {
    fs::create_dir_all(&root())?;
    fs::write(root().join("journal.json"), journal_to_json(journal)?)?;
    Ok(())
}

/// Compute a notarizable head for a hash-chained JSONL file: seq is the
/// line count, tip hash is the `hash` (or `head_hash`/`digest`) field of
/// the last entry.
fn chain_tip(path: &PathBuf, chain_id: &str) -> Result<ChainHead> {
    let text = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
    let last = lines.last().context("chain is empty")?;
    let v: serde_json::Value = serde_json::from_str(last).context("last line is not JSON")?;
    let tip = v
        .get("hash")
        .or_else(|| v.get("head_hash"))
        .or_else(|| v.get("digest"))
        .and_then(|x| x.as_str())
        .context("last entry has no hash field")?
        .to_string();
    Ok(ChainHead {
        chain_id: chain_id.to_string(),
        seq: lines.len() as u64,
        tip_hash: tip,
    })
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        bail!("usage: badapple-witness <init|tip|attest|record|audit|status|verify>");
    }
    let named = |flag: &str| -> Option<String> {
        args.iter()
            .position(|a| a == flag)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };

    match args[0].as_str() {
        "init" => {
            let id = load_identity()?;
            let j = load_journal().unwrap_or_else(|_| WitnessJournal::new(id.public_key()));
            save_journal(&j)?;
            println!(
                "{}",
                serde_json::json!({"status":"initialized","identity":id.public_key()})
            );
        }
        "tip" => {
            let file = named("--file")
                .map(PathBuf::from)
                .unwrap_or_else(|| DEFAULT_LEDGER.into());
            let chain = named("--chain").unwrap_or_else(|| "ledger".into());
            let head = chain_tip(&file, &chain)?;
            println!("{}", serde_json::to_string_pretty(&head)?);
        }
        "attest" => {
            let subject = args.get(1).context("attest <subject-pubkey> <head.json>")?;
            let head_path = args.get(2).context("attest <subject-pubkey> <head.json>")?;
            let head: ChainHead = serde_json::from_str(&fs::read_to_string(head_path)?)?;
            let id = load_identity()?;
            let mut journal = load_journal()?;
            let att = journal.attest(&id, subject, &head)?;
            save_journal(&journal)?;
            println!("{}", serde_json::to_string_pretty(&att)?);
        }
        "record" => {
            let path = args.get(1).context("record <attestation.json>")?;
            let att: Attestation = serde_json::from_str(&fs::read_to_string(path)?)?;
            let mut journal = load_journal()?;
            journal.record_received(att.clone())?;
            save_journal(&journal)?;
            println!(
                "{}",
                serde_json::json!({"status":"recorded","id":att.id(),"witness":att.witness})
            );
        }
        "audit" => {
            let subject = args
                .get(1)
                .context("audit <subject-pubkey> --chain <id> --history <map.json>")?;
            let chain = named("--chain").unwrap_or_else(|| "ledger".into());
            let hist_path = named("--history").context("--history <map.json> required")?;
            let history: BTreeMap<u64, String> =
                serde_json::from_str(&fs::read_to_string(&hist_path)?)?;
            let journal = load_journal()?;
            let report = audit_against_history(&journal, subject, &chain, &history);
            println!(
                "{}",
                serde_json::json!({
                    "checked": report.checked,
                    "confirmed": report.confirmed,
                    "contradicted": report.contradicted.iter().map(|a| a.id()).collect::<Vec<_>>(),
                    "clean": report.clean(),
                })
            );
            if !report.clean() {
                std::process::exit(2);
            }
        }
        "status" => {
            let journal = load_journal()?;
            let head = journal.head();
            println!(
                "{}",
                serde_json::json!({
                    "owner": journal.owner,
                    "events": journal.events.len(),
                    "journal_seq": head.seq,
                    "journal_tip": head.tip_hash,
                })
            );
        }
        "verify" => {
            let journal = load_journal()?;
            journal.verify()?;
            println!(
                "{}",
                serde_json::json!({"status":"verified","events":journal.events.len()})
            );
        }
        other => bail!("unknown command: {other}"),
    }
    Ok(())
}
