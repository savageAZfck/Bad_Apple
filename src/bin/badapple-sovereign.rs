use sovereign_ledger::anchor::{write_checkpoint, Anchor, IdentityAgentAnchor, Tip};
use sovereign_ledger::import::{self, BADAPPLE_GENESIS_LABEL};
use sovereign_ledger::SovereignLedger;
use std::error::Error;
use std::fs;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::process;

fn usage() -> ! {
    eprintln!("badapple-sovereign — hardened parallel audit ledger");
    eprintln!("Usage: badapple-sovereign [options]");
    eprintln!("Options:");
    eprintln!(
        "  -i, --input <path>   Bad Apple ledger.jsonl (default: /var/lib/bad_apple/ledger.jsonl)"
    );
    eprintln!(
        "  -o, --output <path>  Hardened ledger output (default: /var/lib/bad_apple/ledger.sovereign.jsonl)"
    );
    eprintln!("  -k, --key <seed>     Sovereign key seed (default: Bad Apple SLICKS key)");
    eprintln!(
        "  --checkpoint         Verify the chain and write a Secure Enclave-signed checkpoint only"
    );
    eprintln!("  -h, --help           Show this help");
    process::exit(1);
}

fn parse_args() -> (PathBuf, PathBuf, Option<Vec<u8>>, bool) {
    let mut input = PathBuf::from("/var/lib/bad_apple/ledger.jsonl");
    let mut output = PathBuf::from("/var/lib/bad_apple/ledger.sovereign.jsonl");
    let mut sovereign_seed: Option<String> = None;
    let mut checkpoint_only = false;

    let mut iter = std::env::args().skip(1);
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "-h" | "--help" => usage(),
            "-i" | "--input" => {
                input = iter.next().expect("missing value for --input").into();
            }
            "-o" | "--output" => {
                output = iter.next().expect("missing value for --output").into();
            }
            "-k" | "--key" => {
                sovereign_seed = Some(iter.next().expect("missing value for --key"));
            }
            "--checkpoint" => {
                checkpoint_only = true;
            }
            _ => usage(),
        }
    }

    (
        input,
        output,
        sovereign_seed.map(|s| s.into_bytes()),
        checkpoint_only,
    )
}

fn load_slicks_secrets(input: &Path) -> Vec<Vec<u8>> {
    let key_path = std::env::var("BADAPPLE_SLICKS_KEY_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            input
                .parent()
                .unwrap_or(Path::new("/var/lib/bad_apple"))
                .join("slicks.key")
        });
    fs::read(&key_path)
        .map(|raw| import::slicks_key_candidates(&raw))
        .unwrap_or_default()
}

fn hex_to_32(s: &str) -> Result<[u8; 32], Box<dyn Error>> {
    let v = hex::decode(s)?;
    Ok(<[u8; 32]>::try_from(v.as_slice())?)
}

fn sign_checkpoint(
    agent: &IdentityAgentAnchor,
    ledger_dir: &Path,
    checkpoint_name: &str,
    tip: &Tip,
) -> Result<PathBuf, Box<dyn Error>> {
    let checkpoint = agent.attest(tip)?;
    Ok(write_checkpoint(ledger_dir, checkpoint_name, &checkpoint)?)
}

fn main() -> Result<(), Box<dyn Error>> {
    let (input, output, sovereign_seed, checkpoint_only) = parse_args();

    if !input.exists() {
        // Fresh installs have no ledger yet; a scheduled run with nothing
        // to harden is a no-op, not an integrity failure.
        println!("no ledger at {} yet; nothing to harden", input.display());
        return Ok(());
    }
    if input == output {
        eprintln!("input and output paths must be different");
        process::exit(1);
    }

    let slicks_secrets = load_slicks_secrets(&input);
    let input_dir = input.parent().unwrap_or(Path::new(".")).to_path_buf();
    let output_dir = output.parent().unwrap_or(Path::new(".")).to_path_buf();
    let agent = IdentityAgentAnchor::default_socket();

    // The sovereign chain uses the decoded (first) key form as its seed.
    let seed = sovereign_seed
        .as_deref()
        .or_else(|| slicks_secrets.first().map(|v| v.as_slice()));

    // Build the hardened ledger in a temporary file and atomically replace
    // the output. Repeated runs are idempotent; a partial file can never be
    // mistaken for a completed ledger.
    let output_file = output
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_default();
    let tmp_file_name = format!("{}.{pid}.tmp", output_file, pid = process::id());
    let tmp_path = output.with_file_name(&tmp_file_name);
    let tmp_lock_path = output.with_file_name(format!("{tmp_file_name}.lock"));
    let _ = fs::remove_file(&tmp_path);
    let _ = fs::remove_file(&tmp_lock_path);

    let file = fs::File::open(&input)?;
    let reader = BufReader::new(file);
    let (report, sovereign_tip, sovereign_root) = {
        let mut ledger = SovereignLedger::new(&tmp_path, seed)?;
        let report = match import::import_badapple(reader, &slicks_secrets, &mut ledger) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("{e}");
                process::exit(1);
            }
        };
        ledger.verify()?;
        ledger.sync()?;
        (report, ledger.last_hash(), ledger.merkle_root()?)
    };

    let source_tip = Tip {
        tip_hash: hex_to_32(&report.source_tip)?,
        merkle_root: hex_to_32(report.source_merkle_root.as_deref().unwrap_or_default())?,
        entry_count: report.entries as u64,
        genesis: BADAPPLE_GENESIS_LABEL.to_string(),
    };

    if checkpoint_only {
        match sign_checkpoint(&agent, &input_dir, "ledger_checkpoint.json", &source_tip) {
            Ok(p) => {
                println!(
                    "verified {} entries; checkpoint signed at {}",
                    report.entries,
                    p.display()
                );
                return Ok(());
            }
            Err(e) => {
                eprintln!("chain verified but checkpoint signing failed: {e}");
                process::exit(1);
            }
        }
    }

    fs::rename(&tmp_path, &output)?;
    let _ = fs::remove_file(&tmp_lock_path);

    println!(
        "verified and hardened {} entries from {}",
        report.entries,
        input.display()
    );
    println!("wrote {}", output.display());

    // Anchor both chains under the Secure Enclave identity. Signing failure
    // is loud but does not destroy the verified output above.
    let sovereign = Tip {
        tip_hash: sovereign_tip,
        merkle_root: sovereign_root,
        entry_count: report.entries as u64,
        genesis: hex::encode([0u8; 32]),
    };
    for (dir, name, tip) in [
        (input_dir.as_path(), "ledger_checkpoint.json", &source_tip),
        (
            output_dir.as_path(),
            "ledger.sovereign.checkpoint.json",
            &sovereign,
        ),
    ] {
        match sign_checkpoint(&agent, dir, name, tip) {
            Ok(p) => println!("checkpoint signed at {}", p.display()),
            Err(e) => eprintln!("warning: checkpoint signing failed for {name}: {e}"),
        }
    }

    Ok(())
}
