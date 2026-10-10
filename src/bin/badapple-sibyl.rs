//! badapple-sibyl — oracle-gated memory for the organism.
//!
//! Thin wrapper over the public `sibyl` crate. Memories sealed behind
//! events: the organism carries ciphertext and half a key — it cannot
//! recall until the oracle signs the claim and releases the second share.
//!
//! State under /var/lib/bad_apple/sibyl/:
//!   oracle.key   — when this organism acts as oracle for peers' gates
//!   gates/       — held gates (<id>.json)
//!   shares/      — custody shares this organism guards (<gate-id>.share)
//!   gatelog.json — hash-chained record of seals, releases, opens
//!
//!   badapple-sibyl init
//!   badapple-sibyl seal <file|-> --condition <claim> --oracle <pubkey>
//!   badapple-sibyl release <gate-id>            # as oracle: sign + release share
//!   badapple-sibyl unlock <gate-id> <artifact.json>
//!   badapple-sibyl gates
//!   badapple-sibyl custody                      # shares held for others
//!   badapple-sibyl verify

use anyhow::{bail, Context, Result};
use sibyl::{oracle_issue, Gate, GateLog, Identity, ReleaseArtifact};
use std::fs;
use std::io::Read;
use std::path::PathBuf;

const ROOT: &str = "/var/lib/bad_apple/sibyl";

fn root() -> PathBuf {
    std::env::var("BADAPPLE_SIBYL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| ROOT.into())
}

fn oracle_key_path() -> PathBuf {
    root().join("oracle.key")
}

fn load_oracle() -> Result<Identity> {
    let path = oracle_key_path();
    if path.exists() {
        let hex_key = fs::read_to_string(&path)?.trim().to_string();
        let bytes: [u8; 32] = hex::decode(&hex_key)
            .context("oracle.key is not hex")?
            .try_into()
            .map_err(|_| anyhow::anyhow!("oracle.key must be 32 bytes"))?;
        return Ok(Identity::from_bytes(&bytes));
    }
    fs::create_dir_all(root())?;
    let id = Identity::generate();
    fs::write(&path, hex::encode(id.seed()))?;
    Ok(id)
}

fn gate_path(id: &str) -> PathBuf {
    root().join("gates").join(format!("{id}.json"))
}

fn load_gate(id: &str) -> Result<Gate> {
    let text = fs::read_to_string(gate_path(id)).with_context(|| format!("no gate {id}"))?;
    Ok(serde_json::from_str(&text)?)
}

fn save_gate(gate: &Gate) -> Result<()> {
    fs::create_dir_all(root().join("gates"))?;
    fs::write(gate_path(&gate.id), serde_json::to_string_pretty(gate)?)?;
    Ok(())
}

fn share_path(gate_id: &str) -> PathBuf {
    root().join("shares").join(format!("{gate_id}.share"))
}

fn load_log() -> GateLog {
    let path = root().join("gatelog.json");
    if path.exists() {
        serde_json::from_str(&fs::read_to_string(&path).unwrap_or_default()).unwrap_or_default()
    } else {
        GateLog::new()
    }
}

fn save_log(log: &GateLog) -> Result<()> {
    fs::create_dir_all(root())?;
    fs::write(
        root().join("gatelog.json"),
        serde_json::to_string_pretty(log)?,
    )?;
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        bail!("usage: badapple-sibyl <init|seal|release|unlock|gates|custody|verify>");
    }
    let named = |flag: &str| -> Option<String> {
        args.iter()
            .position(|a| a == flag)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };

    match args[0].as_str() {
        "init" => {
            let oracle = load_oracle()?;
            fs::create_dir_all(root().join("gates"))?;
            fs::create_dir_all(root().join("shares"))?;
            println!(
                "{}",
                serde_json::json!({"status":"initialized","oracle":oracle.public_key()})
            );
        }
        "seal" => {
            let src = args.get(1).context("seal <file|->")?;
            let condition = named("--condition").context("--condition required")?;
            let oracle = named("--oracle").context("--oracle <pubkey> required")?;
            let payload: Vec<u8> = if src == "-" {
                let mut buf = Vec::new();
                std::io::stdin().read_to_end(&mut buf)?;
                buf
            } else {
                fs::read(src)?
            };
            // The sealer signs with the oracle identity file if it exists
            // (self-seal), else a fresh ephemeral sealer key.
            let sealer = load_oracle().unwrap_or_else(|_| Identity::generate());
            let (gate, k2) = Gate::seal(&payload, &condition, &oracle, &sealer);
            gate.verify_seal()?;
            save_gate(&gate)?;
            // Custody share lands in shares/ — hand it to the oracle.
            fs::create_dir_all(root().join("shares"))?;
            fs::write(share_path(&gate.id), &k2)?;
            let mut log = load_log();
            log.record("sealed", &gate.id, &gate.condition);
            save_log(&log)?;
            println!(
                "{}",
                serde_json::json!({
                    "status": "sealed",
                    "gate": gate.id,
                    "condition": gate.condition,
                    "oracle": gate.oracle,
                    "custody_share": share_path(&gate.id),
                })
            );
        }
        "release" => {
            let gate_id = args.get(1).context("release <gate-id>")?;
            let gate = load_gate(gate_id)?;
            let k2 = fs::read_to_string(share_path(gate_id))
                .with_context(|| "no custody share for this gate")?
                .trim()
                .to_string();
            let oracle = load_oracle()?;
            let artifact = oracle_issue(&oracle, &gate, &k2)?;
            let mut log = load_log();
            log.record("released", &gate.id, &artifact.claim);
            save_log(&log)?;
            println!("{}", serde_json::to_string_pretty(&artifact)?);
        }
        "unlock" => {
            let gate_id = args.get(1).context("unlock <gate-id> <artifact.json>")?;
            let artifact_path = args.get(2).context("unlock <gate-id> <artifact.json>")?;
            let gate = load_gate(gate_id)?;
            let artifact: ReleaseArtifact =
                serde_json::from_str(&fs::read_to_string(artifact_path)?)?;
            let payload = gate.unlock(&artifact)?;
            let mut log = load_log();
            log.record("opened", &gate.id, "payload disclosed");
            save_log(&log)?;
            println!(
                "{}",
                serde_json::json!({
                    "status": "opened",
                    "gate": gate.id,
                    "payload_utf8": String::from_utf8_lossy(&payload),
                })
            );
        }
        "gates" => {
            let dir = root().join("gates");
            let mut out = Vec::new();
            if dir.exists() {
                for entry in fs::read_dir(&dir)? {
                    let gate: Gate = serde_json::from_str(&fs::read_to_string(entry?.path())?)?;
                    out.push(serde_json::json!({
                        "id": gate.id,
                        "condition": gate.condition,
                        "oracle": gate.oracle,
                        "sealed_by": gate.sealed_by,
                    }));
                }
            }
            println!("{}", serde_json::to_string_pretty(&out)?);
        }
        "custody" => {
            let dir = root().join("shares");
            let mut out = Vec::new();
            if dir.exists() {
                for entry in fs::read_dir(&dir)? {
                    out.push(entry?.file_name().to_string_lossy().to_string());
                }
            }
            println!("{}", serde_json::to_string_pretty(&out)?);
        }
        "verify" => {
            let log = load_log();
            log.verify()?;
            println!(
                "{}",
                serde_json::json!({"status":"verified","events":log.events.len()})
            );
        }
        other => bail!("unknown command: {other}"),
    }
    Ok(())
}
