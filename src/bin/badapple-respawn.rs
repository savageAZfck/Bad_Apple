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
    eprintln!("  -d, --dir <path>     State root to version (repeatable; default:");
    eprintln!("                      /var/lib/bad_apple and ~/.bad_apple)");
    eprintln!("  -m, --message <msg>  Snapshot message (default: scheduled checkpoint)");
    eprintln!("  --status             Report drift against HEAD instead of snapshotting");
    eprintln!("  --revert <ref>       Revert the state root to a snapshot (hash or 'head~N')");
    eprintln!("  --full               With --status: rehash every file (forged-mtime safe)");
    eprintln!("  -h, --help           Show this help");
    process::exit(1);
}

/// State roots the scheduled run versions. `/var/lib/bad_apple` holds platform
/// state; `~/.bad_apple` holds learned state — strategy memory, IFY baseline,
/// notes, proposals — which is just as irreversible when lost.
fn default_dirs() -> Vec<PathBuf> {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
    vec![
        PathBuf::from("/var/lib/bad_apple"),
        PathBuf::from(home).join(".bad_apple"),
    ]
}

struct Args {
    dirs: Vec<PathBuf>,
    message: String,
    status: bool,
    revert_ref: Option<String>,
    full: bool,
}

fn parse_args() -> Args {
    let mut a = Args {
        dirs: Vec::new(),
        message: "scheduled state checkpoint".to_string(),
        status: false,
        revert_ref: None,
        full: false,
    };
    let mut iter = std::env::args().skip(1);
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "-h" | "--help" => usage(),
            "-d" | "--dir" => a
                .dirs
                .push(iter.next().expect("missing value for --dir").into()),
            "-m" | "--message" => a.message = iter.next().expect("missing value for --message"),
            "--status" => a.status = true,
            "--full" => a.full = true,
            "--revert" => a.revert_ref = Some(iter.next().expect("missing value for --revert")),
            _ => usage(),
        }
    }
    if a.dirs.is_empty() {
        a.dirs = default_dirs();
    }
    a
}

fn snapshot_dir(dir: &PathBuf, message: &str) -> Result<(), Box<dyn Error>> {
    if !dir.is_dir() {
        // Fresh installs have no state dir yet; a scheduled run with
        // nothing to version is a no-op, not a failure.
        println!("no state dir at {} yet; nothing to snapshot", dir.display());
        return Ok(());
    }

    let store = match Store::open(dir) {
        Ok(s) => s,
        Err(_) => Store::init(dir)?,
    };

    // Skip the snapshot when nothing changed since HEAD — a daily agent
    // should not pile up identical manifests.
    if let Some(id) = store.head()? {
        let manifest = snapshot::load(&store, &id)?;
        if respawn::drift::detect(dir, &manifest, false)?.clean() {
            println!(
                "clean: {} unchanged since {}",
                dir.display(),
                respawn::short(&id)
            );
            return Ok(());
        }
    }

    let id = snapshot::create(&store, dir, message)?;
    println!("snapshot {} -> {}", respawn::short(&id), dir.display());
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args();

    if args.revert_ref.is_some() || args.status {
        // Point-in-time operations act on exactly one root.
        let dir = args.dirs.first().cloned().ok_or("no state dir specified")?;
        if !dir.is_dir() {
            println!("no state dir at {} yet", dir.display());
            return Ok(());
        }
        let store = match Store::open(&dir) {
            Ok(s) => s,
            Err(_) => Store::init(&dir)?,
        };

        if let Some(reference) = &args.revert_ref {
            let id = snapshot::resolve(&store, reference)?;
            let target = snapshot::load(&store, &id)?;
            // keep_extra=false, force=true: the scheduled snapshot just ran, so
            // reverting to HEAD destroys nothing that isn't already versioned.
            let (_drift, report) = revert::check_then_apply(&store, &dir, &target, false, true)?;
            match report {
                Some(r) => println!(
                    "reverted {} to {}: {} restored, {} removed, {} kept-extra",
                    dir.display(),
                    respawn::short(&id),
                    r.restored.len(),
                    r.removed.len(),
                    r.skipped_extra.len(),
                ),
                None => println!("revert refused: unsnapshotted changes present"),
            }
            return Ok(());
        }

        // --status
        let Some(id) = store.head()? else {
            println!("no snapshots yet");
            return Ok(());
        };
        let manifest = snapshot::load(&store, &id)?;
        let report = respawn::drift::detect(&dir, &manifest, args.full)?;
        if report.clean() {
            println!(
                "clean: {} matches snapshot {}",
                dir.display(),
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

    let mut failed = false;
    for dir in &args.dirs {
        if let Err(e) = snapshot_dir(dir, &args.message) {
            eprintln!("snapshot failed for {}: {}", dir.display(), e);
            failed = true;
        }
    }
    if failed {
        return Err("one or more state roots failed to snapshot".into());
    }
    Ok(())
}
