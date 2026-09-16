use respawn::revert;
use respawn::snapshot;
use respawn::store::Store;
use std::error::Error;
use std::path::PathBuf;
use std::process;

fn usage() -> ! {
    eprintln!("badapple-respawn — filesystem undo for Bad Apple state");
    eprintln!("Usage: badapple-respawn [options]");
    eprintln!("Options:");
    eprintln!("  -d, --dir <path>     State root to version (default: /var/lib/bad_apple)");
    eprintln!("  -m, --message <msg>  Snapshot message (default: scheduled checkpoint)");
    eprintln!("  --status             Report drift against HEAD instead of snapshotting");
    eprintln!("  --revert <ref>       Revert the state root to a snapshot (hash or 'head~N')");
    eprintln!("  --full               With --status: rehash every file (forged-mtime safe)");
    eprintln!("  -h, --help           Show this help");
    process::exit(1);
}

struct Args {
    dir: PathBuf,
    message: String,
    status: bool,
    revert_ref: Option<String>,
    full: bool,
}

fn parse_args() -> Args {
    let mut a = Args {
        dir: PathBuf::from("/var/lib/bad_apple"),
        message: "scheduled state checkpoint".to_string(),
        status: false,
        revert_ref: None,
        full: false,
    };
    let mut iter = std::env::args().skip(1);
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "-h" | "--help" => usage(),
            "-d" | "--dir" => a.dir = iter.next().expect("missing value for --dir").into(),
            "-m" | "--message" => a.message = iter.next().expect("missing value for --message"),
            "--status" => a.status = true,
            "--full" => a.full = true,
            "--revert" => a.revert_ref = Some(iter.next().expect("missing value for --revert")),
            _ => usage(),
        }
    }
    a
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args();

    if !args.dir.is_dir() {
        // Fresh installs have no state dir yet; a scheduled run with
        // nothing to version is a no-op, not a failure.
        println!(
            "no state dir at {} yet; nothing to snapshot",
            args.dir.display()
        );
        return Ok(());
    }

    let store = match Store::open(&args.dir) {
        Ok(s) => s,
        Err(_) => Store::init(&args.dir)?,
    };

    if let Some(reference) = &args.revert_ref {
        let id = snapshot::resolve(&store, reference)?;
        let target = snapshot::load(&store, &id)?;
        // keep_extra=false, force=true: the scheduled snapshot just ran, so
        // reverting to HEAD destroys nothing that isn't already versioned.
        let (_drift, report) = revert::check_then_apply(&store, &args.dir, &target, false, true)?;
        match report {
            Some(r) => println!(
                "reverted {} to {}: {} restored, {} removed, {} kept-extra",
                args.dir.display(),
                respawn::short(&id),
                r.restored.len(),
                r.removed.len(),
                r.skipped_extra.len(),
            ),
            None => println!("revert refused: unsnapshotted changes present"),
        }
        return Ok(());
    }

    let head = store.head()?;
    if args.status {
        let Some(id) = head else {
            println!("no snapshots yet");
            return Ok(());
        };
        let manifest = snapshot::load(&store, &id)?;
        let report = respawn::drift::detect(&args.dir, &manifest, args.full)?;
        if report.clean() {
            println!(
                "clean: {} matches snapshot {}",
                args.dir.display(),
                respawn::short(&id)
            );
        } else {
            println!(
                "drift vs {}: {} added, {} modified, {} deleted, {} touched",
                respawn::short(&id),
                report.added.len(),
                report.modified.len(),
                report.deleted.len(),
                report.touched.len(),
            );
        }
        return Ok(());
    }

    // Skip the snapshot when nothing changed since HEAD — a daily agent
    // should not pile up identical manifests.
    if let Some(id) = &head {
        let manifest = snapshot::load(&store, id)?;
        if respawn::drift::detect(&args.dir, &manifest, false)?.clean() {
            println!("clean: state unchanged since {}", respawn::short(id));
            return Ok(());
        }
    }

    let id = snapshot::create(&store, &args.dir, &args.message)?;
    println!("snapshot {} -> {}", respawn::short(&id), args.dir.display());
    Ok(())
}
