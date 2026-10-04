//! badapple-hansard — bicameral concurrence for the organism.
//!
//! Thin wrapper over the public `bicameral` crate. The GPU brain and the
//! ANE brain are separate chambers: consequential motions (treaty
//! ratification, custodian rotation, self-mod adoption) resolve only
//! when both faculties vote — and a split is preserved as signed dissent.
//!
//! State under /var/lib/bad_apple/hansard/:
//!   gpu.key / ane.key — chamber identities (ed25519, 0600)
//!   chambers.json     — seated chamber registry
//!   hansard.json      — the hash-chained parliamentary record
//!
//!   badapple-hansard init
//!   badapple-hansard seat <name> <pubkey>
//!   badapple-hansard table <subject> [--context <text>] [--deadline <unix>]
//!   badapple-hansard vote <chamber> <motion-id> <aye|nay|abstain> --reason <text>
//!   badapple-hansard resolution <motion-id>
//!   badapple-hansard record [--motion <id>]
//!   badapple-hansard verify

use anyhow::{bail, Context, Result};
use bicameral::{Chamber, Hansard, Identity, Rule, Vote};
use std::fs;
use std::path::PathBuf;

const ROOT: &str = "/var/lib/bad_apple/hansard";

fn root() -> PathBuf {
    std::env::var("BADAPPLE_HANSARD_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| ROOT.into())
}

fn key_path(chamber: &str) -> PathBuf {
    root().join(format!("{chamber}.key"))
}

fn load_key(chamber: &str) -> Result<Identity> {
    let path = key_path(chamber);
    if path.exists() {
        let hex_key = fs::read_to_string(&path)?.trim().to_string();
        let bytes: [u8; 32] = hex::decode(&hex_key)
            .context("key file is not hex")?
            .try_into()
            .map_err(|_| anyhow::anyhow!("key file must be 32 bytes"))?;
        return Ok(Identity::from_bytes(&bytes));
    }
    fs::create_dir_all(&root())?;
    let id = Identity::generate();
    fs::write(&path, hex::encode(id.seed()))?;
    Ok(id)
}

fn load_hansard() -> Result<Hansard> {
    let path = root().join("hansard.json");
    if path.exists() {
        Ok(serde_json::from_str(&fs::read_to_string(&path)?)?)
    } else {
        // Default seating: the organism's two brains.
        let gpu = load_key("gpu")?;
        let ane = load_key("ane")?;
        let chambers = if root().join("chambers.json").exists() {
            serde_json::from_str(&fs::read_to_string(root().join("chambers.json"))?)?
        } else {
            vec![
                Chamber::new("gpu", &gpu.public_key()),
                Chamber::new("ane", &ane.public_key()),
            ]
        };
        Ok(Hansard::new(chambers, Rule::Unanimous))
    }
}

fn save_hansard(h: &Hansard) -> Result<()> {
    fs::create_dir_all(&root())?;
    fs::write(
        root().join("chambers.json"),
        serde_json::to_string_pretty(&h.chambers)?,
    )?;
    fs::write(
        root().join("hansard.json"),
        serde_json::to_string_pretty(h)?,
    )?;
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        bail!("usage: badapple-hansard <init|seat|table|vote|resolution|record|verify>");
    }
    let named = |flag: &str| -> Option<String> {
        args.iter()
            .position(|a| a == flag)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };

    match args[0].as_str() {
        "init" => {
            let gpu = load_key("gpu")?;
            let ane = load_key("ane")?;
            let h = load_hansard()?;
            save_hansard(&h)?;
            println!(
                "{}",
                serde_json::json!({
                    "status": "initialized",
                    "gpu": gpu.public_key(),
                    "ane": ane.public_key(),
                    "chambers": h.chambers.iter().map(|c| c.name.clone()).collect::<Vec<_>>(),
                })
            );
        }
        "seat" => {
            let name = args.get(1).context("seat <name> <pubkey>")?;
            let pubkey = args.get(2).context("seat <name> <pubkey>")?;
            let mut h = load_hansard()?;
            if h.chambers.iter().any(|c| c.name == *name) {
                bail!("chamber already seated: {name}");
            }
            h.chambers.push(Chamber::new(name, pubkey));
            save_hansard(&h)?;
            println!("{}", serde_json::json!({"status":"seated","chamber":name}));
        }
        "table" => {
            let subject = args.get(1).context("table <subject>")?;
            let context = named("--context").unwrap_or_default();
            let deadline = named("--deadline").and_then(|s| s.parse::<u64>().ok());
            let mut h = load_hansard()?;
            let m = h.table_with_deadline(subject, &context, deadline);
            save_hansard(&h)?;
            println!(
                "{}",
                serde_json::json!({"status":"tabled","motion":m.id,"subject":m.subject})
            );
        }
        "vote" => {
            let chamber = args.get(1).context("vote <chamber> <motion-id> <aye|nay|abstain>")?;
            let motion_id = args.get(2).context("vote <chamber> <motion-id> <aye|nay|abstain>")?;
            let vote = match args.get(3).map(|s| s.as_str()) {
                Some("aye") => Vote::Aye,
                Some("nay") => Vote::Nay,
                Some("abstain") => Vote::Abstain,
                _ => bail!("vote must be aye|nay|abstain"),
            };
            let reason = named("--reason").unwrap_or_default();
            let id = load_key(chamber)?;
            let mut h = load_hansard()?;
            let ballot = h.vote(&id, motion_id, vote, &reason)?;
            let resolution = h.resolution(motion_id)?;
            save_hansard(&h)?;
            println!(
                "{}",
                serde_json::json!({
                    "status": "voted",
                    "chamber": ballot.chamber,
                    "vote": ballot.vote,
                    "resolution": resolution,
                })
            );
        }
        "resolution" => {
            let motion_id = args.get(1).context("resolution <motion-id>")?;
            let h = load_hansard()?;
            println!(
                "{}",
                serde_json::json!({"motion":motion_id,"resolution":h.resolution(motion_id)?})
            );
        }
        "record" => {
            let h = load_hansard()?;
            let filter = named("--motion");
            let (seq, tip) = h.head();
            let mut out = serde_json::json!({"entries": seq, "tip": tip, "events": []});
            let events: Vec<serde_json::Value> = h
                .events
                .iter()
                .filter(|e| match (&e.entry, &filter) {
                    (bicameral::HansardEntry::MotionTabled(m), Some(f)) => m.id == *f,
                    (bicameral::HansardEntry::VoteCast(b), Some(f)) => b.motion_id == *f,
                    (bicameral::HansardEntry::Resolved { motion_id, .. }, Some(f)) => motion_id == f,
                    (bicameral::HansardEntry::DissentRecorded(d), Some(f)) => d.motion_id == *f,
                    (_, None) => true,
                })
                .map(|e| serde_json::to_value(&e.entry).unwrap())
                .collect();
            out["events"] = serde_json::json!(events);
            println!("{}", serde_json::to_string_pretty(&out)?);
        }
        "verify" => {
            let h = load_hansard()?;
            h.verify()?;
            let (seq, tip) = h.head();
            println!(
                "{}",
                serde_json::json!({"status":"verified","entries":seq,"tip":tip})
            );
        }
        other => bail!("unknown command: {other}"),
    }
    Ok(())
}
