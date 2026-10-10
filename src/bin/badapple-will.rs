//! badapple-will — posthumous agency for the organism.
//!
//! Thin wrapper over the public `mictlan` crate. The owner signs standing
//! orders while alive; after a cryptographically-proven death (heartbeat
//! silence + custodian quorum), the executor runs only those orders, in
//! scope, on a hash-chained log.
//!
//! State under /var/lib/bad_apple/will/:
//!   owner.key       — the owner's signing identity (ed25519, 0600)
//!   will.json       — the signed will (orders + death condition)
//!   attestations/   — custodian unreachable-attestations received
//!   cert.json       — the issued death certificate (once proven)
//!   execution.json  — the hash-chained execution log
//!
//!   badapple-will init --silence <secs> --quorum <n>
//!   badapple-will order <action> [--params <json>] [--max <n>] [--expires <unix>]
//!   badapple-will heartbeat
//!   badapple-will attest <file.json>           # record custodian attestation
//!   badapple-will certify --last-hb <unix>     # attempt certification
//!   badapple-will execute <order-id> [--outcome <json>]
//!   badapple-will status
//!   badapple-will verify

use anyhow::{bail, Context, Result};
use mictlan::{
    DeadManSwitch, DeathCertificate, Executor, Heartbeat, Identity, StandingOrder,
    UnreachableAttestation, Will,
};
use serde_json::Value;
use std::fs;
use std::path::PathBuf;

const ROOT: &str = "/var/lib/bad_apple/will";

fn root() -> PathBuf {
    std::env::var("BADAPPLE_WILL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| ROOT.into())
}

fn load_owner() -> Result<Identity> {
    let path = root().join("owner.key");
    if path.exists() {
        let hex_key = fs::read_to_string(&path)?.trim().to_string();
        let bytes: [u8; 32] = hex::decode(&hex_key)
            .context("owner.key is not hex")?
            .try_into()
            .map_err(|_| anyhow::anyhow!("owner.key must be 32 bytes"))?;
        return Ok(Identity::from_bytes(&bytes));
    }
    fs::create_dir_all(root())?;
    let id = Identity::generate();
    fs::write(&path, hex::encode(id.seed()))?;
    Ok(id)
}

fn will_path() -> PathBuf {
    root().join("will.json")
}

fn load_will() -> Result<Will> {
    let text = fs::read_to_string(will_path()).context("no will — run init first")?;
    Ok(serde_json::from_str(&text)?)
}

fn save_will(will: &Will) -> Result<()> {
    fs::create_dir_all(root())?;
    fs::write(will_path(), serde_json::to_string_pretty(will)?)?;
    Ok(())
}

fn load_executor() -> Result<Executor> {
    let path = root().join("execution.json");
    if path.exists() {
        let log: mictlan::ExecutionLog = serde_json::from_str(&fs::read_to_string(&path)?)?;
        Ok(Executor { log })
    } else {
        Ok(Executor::new())
    }
}

fn save_executor(ex: &Executor) -> Result<()> {
    fs::create_dir_all(root())?;
    fs::write(
        root().join("execution.json"),
        serde_json::to_string_pretty(&ex.log)?,
    )?;
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        bail!("usage: badapple-will <init|order|heartbeat|attest|certify|execute|status|verify>");
    }
    let named = |flag: &str| -> Option<String> {
        args.iter()
            .position(|a| a == flag)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };

    match args[0].as_str() {
        "init" => {
            let silence: u64 = named("--silence")
                .and_then(|s| s.parse().ok())
                .unwrap_or(86_400);
            let quorum: usize = named("--quorum").and_then(|s| s.parse().ok()).unwrap_or(1);
            let owner = load_owner()?;
            if will_path().exists() {
                println!(
                    "{}",
                    serde_json::json!({"status":"exists","owner":owner.public_key()})
                );
            } else {
                let will = Will::sign(&owner, vec![], silence, quorum);
                save_will(&will)?;
                println!(
                    "{}",
                    serde_json::json!({"status":"initialized","owner":owner.public_key()})
                );
            }
        }
        "order" => {
            let action = args.get(1).context("order <action>")?;
            let params: Value = named("--params")
                .map(|s| serde_json::from_str(&s).unwrap_or(Value::Null))
                .unwrap_or(Value::Null);
            let max: u32 = named("--max").and_then(|s| s.parse().ok()).unwrap_or(0);
            let expires: u64 = named("--expires").and_then(|s| s.parse().ok()).unwrap_or(0);
            let owner = load_owner()?;
            let mut will = load_will()?;
            let order = StandingOrder::sign(&owner, action, params, max, expires);
            let id = order.id.clone();
            will.orders.push(order);
            // Re-sign the will over the expanded order set.
            let will = Will::sign(&owner, will.orders, will.silence_secs, will.quorum);
            save_will(&will)?;
            println!("{}", serde_json::json!({"status":"ordered","order":id}));
        }
        "heartbeat" => {
            let owner = load_owner()?;
            let hb = Heartbeat::sign(&owner);
            hb.verify()?;
            fs::create_dir_all(root())?;
            fs::write(
                root().join("last_heartbeat.json"),
                serde_json::to_string_pretty(&hb)?,
            )?;
            println!(
                "{}",
                serde_json::json!({"status":"alive","ts":hb.ts,"owner":hb.owner})
            );
        }
        "attest" => {
            let path = args.get(1).context("attest <file.json>")?;
            let att: UnreachableAttestation = serde_json::from_str(&fs::read_to_string(path)?)?;
            att.verify()?;
            let will = load_will()?;
            if att.owner != will.owner {
                bail!("attestation is for a different owner");
            }
            let dir = root().join("attestations");
            fs::create_dir_all(&dir)?;
            fs::write(
                dir.join(format!("{}.json", &att.custodian[..16])),
                fs::read_to_string(path)?,
            )?;
            println!(
                "{}",
                serde_json::json!({"status":"recorded","custodian":att.custodian})
            );
        }
        "certify" => {
            let will = load_will()?;
            let last_hb: u64 = named("--last-hb")
                .and_then(|s| s.parse().ok())
                .or_else(|| {
                    fs::read_to_string(root().join("last_heartbeat.json"))
                        .ok()
                        .and_then(|t| serde_json::from_str::<Heartbeat>(&t).ok())
                        .map(|h| h.ts)
                })
                .context("--last-hb required (no heartbeat on file)")?;
            let mut switch = DeadManSwitch::new(&will);
            let dir = root().join("attestations");
            if dir.exists() {
                for entry in fs::read_dir(&dir)? {
                    let att: UnreachableAttestation =
                        serde_json::from_str(&fs::read_to_string(entry?.path())?)?;
                    switch.record_unreachable(att)?;
                }
            }
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_secs();
            let cert = switch.certify(last_hb, now)?;
            fs::write(
                root().join("cert.json"),
                serde_json::to_string_pretty(&cert)?,
            )?;
            println!(
                "{}",
                serde_json::json!({"status":"certified","certified_at":cert.certified_at})
            );
        }
        "execute" => {
            let order_id = args.get(1).context("execute <order-id>")?;
            let outcome: Value = named("--outcome")
                .map(|s| serde_json::from_str(&s).unwrap_or(Value::Null))
                .unwrap_or(Value::Null);
            let will = load_will()?;
            let cert: DeathCertificate = serde_json::from_str(
                &fs::read_to_string(root().join("cert.json"))
                    .context("no death certificate — run certify first")?,
            )?;
            let mut executor = load_executor()?;
            let e = executor.execute(&will, &cert, order_id, outcome)?;
            save_executor(&executor)?;
            println!(
                "{}",
                serde_json::json!({"status":"executed","seq":e.seq,"order":e.order_id})
            );
        }
        "status" => {
            let will = load_will()?;
            let executor = load_executor()?;
            let certified = root().join("cert.json").exists();
            println!(
                "{}",
                serde_json::json!({
                    "owner": will.owner,
                    "orders": will.orders.len(),
                    "silence_secs": will.silence_secs,
                    "quorum": will.quorum,
                    "death_certified": certified,
                    "executions": executor.log.entries.len(),
                })
            );
        }
        "verify" => {
            let will = load_will()?;
            will.verify()?;
            let executor = load_executor()?;
            executor.log.verify()?;
            if root().join("cert.json").exists() {
                let cert: DeathCertificate =
                    serde_json::from_str(&fs::read_to_string(root().join("cert.json"))?)?;
                cert.verify(&will)?;
            }
            println!("{}", serde_json::json!({"status":"verified"}));
        }
        other => bail!("unknown command: {other}"),
    }
    Ok(())
}
