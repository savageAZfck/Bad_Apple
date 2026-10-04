//! badapple-escrow — dead-man memory escrow for the organism.
//!
//! Thin wrapper over the public `nagi` crate. Escrows the organism's
//! memory-vault master key across owned custodian machines as signed
//! Shamir shards; a daily heartbeat beacon keeps dead-man recovery armed;
//! a successor key can recover the mind if the machine — or its owner —
//! goes silent.
//!
//! State under /var/lib/bad_apple/escrow/:
//!   identity.key     — the escrow identity (ed25519, root-owned 0600)
//!   plan.json        — the escrow plan in force
//!   shards/          — signed shards awaiting distribution
//!   held/            — shards this machine holds as a custodian
//!   beacon.json      — newest signed heartbeat
//!   recoveries/      — signed recovery certificates
//!
//!   badapple-escrow init <k> <n> [--heir <pubkey>] [--staleness <secs>]
//!   badapple-escrow seal --secret-file <path>
//!   badapple-escrow beacon
//!   badapple-escrow custodian-accept <shard-file>
//!   badapple-escrow judge <claim-file> [--release]
//!   badapple-escrow claim --heir-key <hex> --last-beacon <file>
//!   badapple-escrow recover <shard-file>...

use anyhow::{bail, Context, Result};
use nagi::{
    Beacon, ClaimVerdict, Custodian, Escrow, Identity, Plan, Recovery, RecoveryClaim, Shard,
};
use std::fs;
use std::path::PathBuf;

const ROOT: &str = "/var/lib/bad_apple/escrow";

fn root() -> PathBuf {
    std::env::var("BADAPPLE_ESCROW_DIR")
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

fn load_plan() -> Result<Plan> {
    let raw = fs::read_to_string(root().join("plan.json")).context("no plan.json — run init")?;
    Ok(serde_json::from_str(&raw)?)
}

fn now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        bail!(
            "usage: badapple-escrow <init|seal|beacon|custodian-accept|judge|claim|recover|status>"
        );
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
            let k: usize = args.get(1).context("init <k> <n>")?.parse()?;
            let n: usize = args.get(2).context("init <k> <n>")?.parse()?;
            let plan = match named("--heir") {
                Some(heir) => {
                    let staleness = named("--staleness")
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(86_400 * 30);
                    Plan::dead_man(k, n, staleness, heir)
                }
                None => Plan::owner_unlock(k, n),
            };
            fs::create_dir_all(root().join("shards"))?;
            fs::create_dir_all(root().join("held"))?;
            fs::create_dir_all(root().join("recoveries"))?;
            fs::write(
                root().join("plan.json"),
                serde_json::to_string_pretty(&plan)?,
            )?;
            println!(
                "{}",
                serde_json::json!({"status":"initialized","identity":id.public_key(),"plan":plan})
            );
        }
        "seal" => {
            let id = load_identity()?;
            let plan = load_plan()?;
            let secret_path = named("--secret-file")
                .or_else(|| {
                    named("--secret").map(|s| {
                        fs::write(root().join(".tmp_secret"), &s).ok();
                        root().join(".tmp_secret").display().to_string()
                    })
                })
                .context("seal requires --secret-file <path> or --secret <value>")?;
            let secret = fs::read(&secret_path).context("cannot read secret file")?;
            let shards = Escrow::seal(&id, &secret, &plan)?;
            let ceremony = shards[0].ceremony_id.clone();
            let dir = root().join("shards").join(&ceremony[..16]);
            fs::create_dir_all(&dir)?;
            for shard in &shards {
                fs::write(
                    dir.join(format!("shard-{}.json", shard.index)),
                    serde_json::to_string_pretty(shard)?,
                )?;
            }
            let _ = fs::remove_file(root().join(".tmp_secret"));
            println!(
                "{}",
                serde_json::json!({"status":"sealed","ceremony":ceremony,"shards":shards.len(),"dir":dir})
            );
        }
        "beacon" => {
            let id = load_identity()?;
            let plan = load_plan()?;
            let ceremony = plan.ceremony_id(&id.public_key());
            let beacon = Beacon::sign(&id, &ceremony);
            fs::write(
                root().join("beacon.json"),
                serde_json::to_string_pretty(&beacon)?,
            )?;
            println!("{}", serde_json::json!({"status":"alive","ts":beacon.ts}));
        }
        "custodian-accept" => {
            let file = args.get(1).context("custodian-accept <shard-file>")?;
            let shard: Shard = serde_json::from_str(&fs::read_to_string(file)?)?;
            let plan = load_plan()?;
            let c = Custodian::accept(shard.clone(), &plan)?;
            let held = root().join("held");
            fs::create_dir_all(&held)?;
            let name = format!("{}-{}.json", &shard.ceremony_id[..16], shard.index);
            fs::write(held.join(&name), serde_json::to_string_pretty(&shard)?)?;
            println!(
                "{}",
                serde_json::json!({"status":"held","shard":shard.index})
            );
            drop(c);
        }
        "judge" => {
            let file = args.get(1).context("judge <claim-file>")?;
            let claim: RecoveryClaim = serde_json::from_str(&fs::read_to_string(file)?)?;
            let held_dir = root().join("held");
            let mut released = false;
            for entry in fs::read_dir(&held_dir)? {
                let shard: Shard = serde_json::from_str(&fs::read_to_string(entry?.path())?)?;
                if shard.ceremony_id != claim.ceremony_id {
                    continue;
                }
                let plan = load_plan()?;
                let c = Custodian::accept(shard.clone(), &plan)?;
                match c.judge(&claim, now()) {
                    (ClaimVerdict::Release, Some(s)) => {
                        let out = root().join("recoveries").join(format!(
                            "release-{}-{}.json",
                            &claim.ceremony_id[..16],
                            s.index
                        ));
                        fs::create_dir_all(out.parent().unwrap())?;
                        fs::write(&out, serde_json::to_string_pretty(&s)?)?;
                        println!("{}", serde_json::json!({"verdict":"release","shard":out}));
                        released = true;
                    }
                    (ClaimVerdict::NotYet(m), _) => {
                        println!("{}", serde_json::json!({"verdict":"not_yet","reason":m}))
                    }
                    (ClaimVerdict::Deny(m), _) => {
                        println!("{}", serde_json::json!({"verdict":"deny","reason":m}))
                    }
                    (ClaimVerdict::Release, None) => {}
                }
            }
            if !released {
                std::process::exit(1);
            }
        }
        "claim" => {
            let heir_hex = named("--heir-key").context("claim requires --heir-key <hex>")?;
            let seed: [u8; 32] = hex::decode(&heir_hex)?
                .try_into()
                .map_err(|_| anyhow::anyhow!("bad key"))?;
            let heir = Identity::from_bytes(&seed);
            let beacon_file =
                named("--last-beacon").context("claim requires --last-beacon <file>")?;
            let beacon: Beacon = serde_json::from_str(&fs::read_to_string(beacon_file)?)?;
            if !beacon.verify() {
                bail!("beacon signature invalid");
            }
            let claim = RecoveryClaim::dead_man_claim(&heir, &beacon.ceremony_id, beacon.ts);
            let out = root().join(format!("claim-{}.json", &beacon.ceremony_id[..16]));
            fs::write(&out, serde_json::to_string_pretty(&claim)?)?;
            println!(
                "{}",
                serde_json::json!({"status":"claim_written","file":out})
            );
        }
        "recover" => {
            let id = load_identity()?;
            let files = &args[1..];
            let mut shards = Vec::new();
            for f in files {
                shards.push(serde_json::from_str::<Shard>(&fs::read_to_string(f)?)?);
            }
            let (secret, cert) = Recovery::assemble(&id, &shards)?;
            let out = root()
                .join("recoveries")
                .join(format!("cert-{}.json", &cert.ceremony_id[..16]));
            fs::create_dir_all(out.parent().unwrap())?;
            fs::write(&out, serde_json::to_string_pretty(&cert)?)?;
            println!(
                "{}",
                serde_json::json!({
                    "status": "recovered",
                    "bytes": secret.len(),
                    "secret": hex::encode(&secret),
                    "certificate": out,
                })
            );
        }
        "status" => {
            let plan = load_plan()?;
            let held = fs::read_dir(root().join("held"))
                .map(|d| d.count())
                .unwrap_or(0);
            let beacon = fs::read_to_string(root().join("beacon.json"))
                .ok()
                .and_then(|s| serde_json::from_str::<Beacon>(&s).ok());
            println!(
                "{}",
                serde_json::json!({
                    "plan": plan,
                    "shards_held": held,
                    "last_beacon": beacon.map(|b| b.ts),
                })
            );
        }
        other => bail!("unknown subcommand: {other}"),
    }
    Ok(())
}
