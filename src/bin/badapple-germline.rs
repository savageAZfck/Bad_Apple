//! badapple-germline — organism reproduction for the organism.
//!
//! Thin wrapper over the public `germline` crate. Birth a child organism:
//! dual-signed genesis, bequested memories, checkable genealogy.
//!
//! State under /var/lib/bad_apple/germline/:
//!   identity.key   — this organism's lineage identity (ed25519, 0600)
//!   lineage.json   — the family tree (geneses, births, revocations)
//!   children/      — accepted genesis blocks issued by this organism
//!
//!   badapple-germline init
//!   badapple-germline bear --child <pubkey> --tip <hex64>
//!       [--bequest <resource>:<hash>:<revocable>]...
//!   badapple-germline accept <genesis.json>        # as the child
//!   badapple-germline record <genesis.json>        # add to the tree
//!   badapple-germline revoke <genesis-id> <bequest-id> --reason <text>
//!   badapple-germline ancestry <pubkey>
//!   badapple-germline descends <descendant> <ancestor>
//!   badapple-germline tree
//!   badapple-germline verify

use anyhow::{bail, Context, Result};
use germline::{Bequest, GenesisBlock, Identity, Lineage, Revocation};
use std::fs;
use std::path::PathBuf;

const ROOT: &str = "/var/lib/bad_apple/germline";

fn root() -> PathBuf {
    std::env::var("BADAPPLE_GERMLINE_DIR")
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

fn load_lineage() -> Result<Lineage> {
    let path = root().join("lineage.json");
    if path.exists() {
        let tree: Lineage = serde_json::from_str(&fs::read_to_string(&path)?)?;
        tree.verify()?;
        Ok(tree)
    } else {
        Ok(Lineage::new())
    }
}

fn save_lineage(tree: &Lineage) -> Result<()> {
    fs::create_dir_all(&root())?;
    fs::write(
        root().join("lineage.json"),
        serde_json::to_string_pretty(tree)?,
    )?;
    Ok(())
}

fn my_generation(tree: &Lineage, pubkey: &str) -> u32 {
    tree.births
        .get(pubkey)
        .map(|id| tree.geneses[id].generation)
        .unwrap_or(0)
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        bail!("usage: badapple-germline <init|bear|accept|record|revoke|ancestry|descends|tree|verify>");
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
            fs::create_dir_all(root().join("children"))?;
            println!(
                "{}",
                serde_json::json!({"status":"initialized","identity":id.public_key()})
            );
        }
        "bear" => {
            // Note: --child is a pubkey; the child's actual Identity
            // signs acceptance on its own side. For the demo/CLI path
            // we issue the parent's half; `accept` completes it.
            let child_pk = named("--child").context("--child <pubkey> required")?;
            let tip = named("--tip").context("--tip <hex64> required")?;
            let bequests: Vec<Bequest> = all_of("--bequest")
                .iter()
                .map(|spec| {
                    let parts: Vec<&str> = spec.splitn(3, ':').collect();
                    if parts.len() != 3 {
                        bail!("--bequest <resource>:<hash>:<revocable>");
                    }
                    Ok(Bequest::grant(parts[0], parts[1], parts[2] == "true"))
                })
                .collect::<Result<_>>()?;
            let id = load_identity()?;
            let tree = load_lineage().unwrap_or_default();
            let generation = my_generation(&tree, &id.public_key()) + 1;
            // Parent signs its half over the real child pubkey; the
            // child countersigns via `accept` on its own machine.
            let genesis = GenesisBlock::issue_for(&id, &child_pk, generation, &tip, bequests)?;
            let dir = root().join("children");
            fs::create_dir_all(&dir)?;
            let path = dir.join(format!("{}.json", &genesis.id[..16]));
            fs::write(&path, serde_json::to_string_pretty(&genesis)?)?;
            println!(
                "{}",
                serde_json::json!({
                    "status": "issued-pending-acceptance",
                    "genesis": genesis.id,
                    "child": genesis.child,
                    "file": path,
                })
            );
        }
        "accept" => {
            let path = args.get(1).context("accept <genesis.json>")?;
            let mut genesis: GenesisBlock = serde_json::from_str(&fs::read_to_string(path)?)?;
            let id = load_identity()?;
            genesis.accept(&id)?;
            fs::write(path, serde_json::to_string_pretty(&genesis)?)?;
            println!(
                "{}",
                serde_json::json!({"status":"accepted","genesis":genesis.id})
            );
        }
        "record" => {
            let path = args.get(1).context("record <genesis.json>")?;
            let genesis: GenesisBlock = serde_json::from_str(&fs::read_to_string(path)?)?;
            let mut tree = load_lineage().unwrap_or_default();
            tree.add_genesis(&genesis)?;
            save_lineage(&tree)?;
            println!(
                "{}",
                serde_json::json!({"status":"recorded","genesis":genesis.id})
            );
        }
        "revoke" => {
            let genesis_id = args.get(1).context("revoke <genesis-id> <bequest-id>")?;
            let bequest_id = args.get(2).context("revoke <genesis-id> <bequest-id>")?;
            let reason = named("--reason").unwrap_or_else(|| "revoked".into());
            let id = load_identity()?;
            let mut tree = load_lineage()?;
            let genesis = tree
                .geneses
                .get(genesis_id)
                .context("unknown genesis")?
                .clone();
            let r = Revocation::sign(&id, &genesis, bequest_id, &reason)?;
            tree.add_revocation(&r)?;
            save_lineage(&tree)?;
            println!(
                "{}",
                serde_json::json!({"status":"revoked","bequest":bequest_id})
            );
        }
        "ancestry" => {
            let pk = args.get(1).context("ancestry <pubkey>")?;
            let tree = load_lineage()?;
            println!("{}", serde_json::to_string_pretty(&tree.ancestry(pk))?);
        }
        "descends" => {
            let desc = args.get(1).context("descends <descendant> <ancestor>")?;
            let anc = args.get(2).context("descends <descendant> <ancestor>")?;
            let tree = load_lineage()?;
            let yes = tree.descends_from(desc, anc);
            println!("{}", serde_json::json!({"descends": yes}));
            if !yes {
                std::process::exit(2);
            }
        }
        "tree" => {
            let tree = load_lineage()?;
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "births": tree.births.len(),
                    "geneses": tree.geneses.len(),
                    "revocations": tree.revocations.len(),
                }))?
            );
        }
        "verify" => {
            let tree = load_lineage()?;
            tree.verify()?;
            println!(
                "{}",
                serde_json::json!({"status":"verified","geneses":tree.geneses.len()})
            );
        }
        other => bail!("unknown command: {other}"),
    }
    Ok(())
}
