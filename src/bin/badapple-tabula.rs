//! badapple-tabula — proof of non-knowledge over the organism's own history.
//!
//! Thin wrapper over the public `tabula` crate. Builds a commitment chain
//! from the audit ledger and signs certified-absence proofs — "this content
//! never occurred in my history," replayable offline by anyone holding
//! the derived chain.
//!
//! State under /var/lib/bad_apple/tabula/:
//!   identity.key    — prover key (ed25519, 0600)
//!   chain.json      — commitment chain derived from the ledger
//!   proofs/         — signed NonKnowledgeProof artifacts
//!
//!   badapple-tabula init
//!   badapple-tabula scan                      # build chain from ledger
//!   badapple-tabula prove "<content>"         # sign an absence proof
//!   badapple-tabula prove-hash <hex64>
//!   badapple-tabula verify <proof.json>       # replay against the chain
//!   badapple-tabula status

use anyhow::{bail, Context, Result};
use std::fs;
use std::path::PathBuf;
use tabula::{Chain, Identity, NonKnowledgeProof, Tabula};

const ROOT: &str = "/var/lib/bad_apple/tabula";
const LEDGER: &str = "/var/lib/bad_apple/ledger.jsonl";

fn root() -> PathBuf {
    std::env::var("BADAPPLE_TABULA_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| ROOT.into())
}

fn ledger_path() -> PathBuf {
    std::env::var("BADAPPLE_LEDGER")
        .map(PathBuf::from)
        .unwrap_or_else(|_| LEDGER.into())
}

fn load_identity() -> Result<Identity> {
    let path = root().join("identity.key");
    if path.exists() {
        let bytes: [u8; 32] = hex_decode(fs::read_to_string(&path)?.trim())
            .context("identity.key is not hex")?
            .try_into()
            .map_err(|_| anyhow::anyhow!("identity.key must be 32 bytes"))?;
        return Ok(Identity::from_bytes(&bytes));
    }
    fs::create_dir_all(&root())?;
    let id = Identity::generate();
    fs::write(&path, hex_encode(id.seed()))?;
    Ok(id)
}

fn hex_encode(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}
fn hex_decode(s: &str) -> Result<Vec<u8>> {
    if s.len() % 2 != 0 {
        bail!("odd-length hex");
    }
    (0..s.len())
        .step_by(2)
        .map(|i| Ok(u8::from_str_radix(&s[i..i + 2], 16)?))
        .collect()
}

fn load_chain() -> Result<Chain> {
    let path = root().join("chain.json");
    if path.exists() {
        let chain: Chain = serde_json::from_str(&fs::read_to_string(&path)?)?;
        chain.verify()?;
        Ok(chain)
    } else {
        Ok(Chain::new())
    }
}

fn save_chain(chain: &Chain) -> Result<()> {
    fs::create_dir_all(&root())?;
    fs::write(
        root().join("chain.json"),
        serde_json::to_string_pretty(chain)?,
    )?;
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        bail!("usage: badapple-tabula <init|scan|prove|prove-hash|verify|status>");
    }

    match args[0].as_str() {
        "init" => {
            let id = load_identity()?;
            println!(
                "{}",
                serde_json::json!({"status":"initialized","identity":id.public_key()})
            );
        }
        "scan" => {
            // Rebuild the commitment chain from the live ledger.
            let mut chain = Chain::new();
            let ledger = ledger_path();
            if ledger.exists() {
                for line in fs::read_to_string(&ledger)?.lines() {
                    if !line.trim().is_empty() {
                        chain.append(line.as_bytes());
                    }
                }
            }
            chain.verify()?;
            save_chain(&chain)?;
            println!(
                "{}",
                serde_json::json!({
                    "status": "scanned",
                    "entries": chain.len(),
                    "head": chain.head(),
                })
            );
        }
        "prove" | "prove-hash" => {
            let query = args.get(1).context("query content/hash required")?;
            let chain = load_chain()?;
            let id = load_identity()?;
            let t = Tabula::new(chain.clone(), id);
            let proof = if args[0] == "prove" {
                t.prove_absence(query)
            } else {
                t.prove_absence_hash(query)
            };
            match proof {
                Ok(p) => {
                    let dir = root().join("proofs");
                    fs::create_dir_all(&dir)?;
                    let path = dir.join(format!("{}.json", &p.query_hash[..16]));
                    fs::write(&path, serde_json::to_string_pretty(&p)?)?;
                    println!(
                        "{}",
                        serde_json::json!({
                            "status": "certified-absent",
                            "entries_scanned": p.entries_scanned,
                            "head": p.head,
                            "proof": path,
                        })
                    );
                }
                Err(tabula::Error::KnowledgeFound { seq }) => {
                    println!(
                        "{}",
                        serde_json::json!({
                            "status": "refused",
                            "reason": format!("content present at seq {seq} — cannot certify absence"),
                        })
                    );
                    std::process::exit(2);
                }
                Err(e) => return Err(e.into()),
            }
        }
        "verify" => {
            let path = args.get(1).context("verify <proof.json>")?;
            let proof: NonKnowledgeProof =
                serde_json::from_str(&fs::read_to_string(path)?)?;
            let chain = load_chain()?;
            match proof.check(&chain) {
                Ok(true) => println!(
                    "{}",
                    serde_json::json!({"status":"verified","entries":proof.entries_scanned})
                ),
                Ok(false) => bail!("proof checked but reported false"),
                Err(e) => {
                    println!("{}", serde_json::json!({"status":"invalid","reason":e.to_string()}));
                    std::process::exit(2);
                }
            }
        }
        "status" => {
            let chain = load_chain().unwrap_or_default();
            let proofs = fs::read_dir(root().join("proofs"))
                .map(|d| d.count())
                .unwrap_or(0);
            println!(
                "{}",
                serde_json::json!({
                    "entries": chain.len(),
                    "head": chain.head(),
                    "proofs_issued": proofs,
                })
            );
        }
        other => bail!("unknown command: {other}"),
    }
    Ok(())
}
