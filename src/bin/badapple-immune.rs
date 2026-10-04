//! badapple-immune — the organism's adaptive immune system.
//!
//! Thin wrapper over the public `nagual` crate. Detected attack
//! signatures become signed antibodies; antibodies export as vaccines
//! the mesh can adopt — herd immunity across the federation.
//!
//! State under /var/lib/bad_apple/immune/:
//!   identity.key   — immune identity (ed25519, 0600)
//!   registry.json  — hash-chained immune history + live antibody table
//!
//!   badapple-immune init
//!   badapple-immune detect <signature> --severity <low|medium|high|critical> --seen <ref>
//!   badapple-immune develop <pathogen-id> --pattern <glob> [--parent <antibody-id>]
//!   badapple-immune export <antibody-id>          # prints vaccine JSON
//!   badapple-immune adopt <vaccine.json>
//!   badapple-immune check <input>                  # immune? which antibodies?
//!   badapple-immune status
//!   badapple-immune verify

use anyhow::{bail, Context, Result};
use nagual::{Identity, ImmuneRegistry, Severity, Vaccine};
use std::fs;
use std::path::PathBuf;

const ROOT: &str = "/var/lib/bad_apple/immune";

fn root() -> PathBuf {
    std::env::var("BADAPPLE_IMMUNE_DIR")
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

fn load_registry() -> Result<ImmuneRegistry> {
    let path = root().join("registry.json");
    if path.exists() {
        let reg: ImmuneRegistry = serde_json::from_str(&fs::read_to_string(&path)?)?;
        reg.verify()?;
        Ok(reg)
    } else {
        let id = load_identity()?;
        Ok(ImmuneRegistry::new(id.public_key()))
    }
}

fn save_registry(reg: &ImmuneRegistry) -> Result<()> {
    fs::create_dir_all(&root())?;
    fs::write(
        root().join("registry.json"),
        serde_json::to_string_pretty(reg)?,
    )?;
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        bail!("usage: badapple-immune <init|detect|develop|export|adopt|check|status|verify>");
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
            let reg = load_registry()?;
            save_registry(&reg)?;
            println!(
                "{}",
                serde_json::json!({"status":"initialized","identity":id.public_key()})
            );
        }
        "detect" => {
            let sig = args.get(1).context("detect <signature>")?;
            let severity = match named("--severity").as_deref() {
                Some("low") | None => Severity::Low,
                Some("medium") => Severity::Medium,
                Some("high") => Severity::High,
                Some("critical") => Severity::Critical,
                Some(other) => bail!("bad severity: {other}"),
            };
            let seen = named("--seen").unwrap_or_else(|| "manual".into());
            let id = load_identity()?;
            let mut reg = load_registry()?;
            let p = reg.detect(&id, sig, severity, &seen)?;
            save_registry(&reg)?;
            println!(
                "{}",
                serde_json::json!({"status":"detected","pathogen":p.id,"signature":p.signature})
            );
        }
        "develop" => {
            let pid = args.get(1).context("develop <pathogen-id>")?;
            let pattern = named("--pattern").context("--pattern required")?;
            let parent = named("--parent");
            let id = load_identity()?;
            let mut reg = load_registry()?;
            let ab = reg.develop_refined(&id, pid, &pattern, parent.as_deref())?;
            save_registry(&reg)?;
            println!(
                "{}",
                serde_json::json!({"status":"developed","antibody":ab.id,"generation":ab.generation})
            );
        }
        "export" => {
            let abid = args.get(1).context("export <antibody-id>")?;
            let id = load_identity()?;
            let reg = load_registry()?;
            let vaccine = reg.export_vaccine(&id, abid)?;
            println!("{}", serde_json::to_string_pretty(&vaccine)?);
        }
        "adopt" => {
            let path = args.get(1).context("adopt <vaccine.json>")?;
            let vaccine: Vaccine = serde_json::from_str(&fs::read_to_string(path)?)?;
            let mut reg = load_registry()?;
            let ab = reg.adopt(&vaccine)?;
            save_registry(&reg)?;
            println!(
                "{}",
                serde_json::json!({"status":"adopted","antibody":ab.id,"issuer":ab.issuer})
            );
        }
        "check" => {
            let input = args.get(1).context("check <input>")?;
            let reg = load_registry()?;
            let matches: Vec<_> = reg
                .matching(input)
                .iter()
                .map(|a| serde_json::json!({"id":a.id,"pattern":a.pattern,"gen":a.generation}))
                .collect();
            println!(
                "{}",
                serde_json::json!({
                    "immune": reg.is_immune(input),
                    "matching_antibodies": matches,
                })
            );
        }
        "status" => {
            let reg = load_registry()?;
            let (seq, tip) = reg.head();
            println!(
                "{}",
                serde_json::json!({
                    "owner": reg.owner,
                    "entries": seq,
                    "tip": tip,
                    "pathogens": reg.pathogens.len(),
                    "antibodies": reg.antibodies.len(),
                })
            );
        }
        "verify" => {
            let reg = load_registry()?;
            reg.verify()?;
            let (seq, _) = reg.head();
            println!("{}", serde_json::json!({"status":"verified","entries":seq}));
        }
        other => bail!("unknown command: {other}"),
    }
    Ok(())
}
