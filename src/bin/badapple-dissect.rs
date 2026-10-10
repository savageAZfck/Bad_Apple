//! badapple-dissect — the reverse-engineering organ.
//!
//! Thin wrapper over the public `haruspex` crate. Lets the organism
//! see inside the machine it lives on: what a binary links, imports,
//! embeds, and asks permission for — all read-only, all evidence-backed.
//!
//! State under /var/lib/bad_apple/dissect/:
//!   reports/<sha256[:12]>.json — audit reports, kept as the record
//!   of what was inspected and what was found.
//!
//!   badapple-dissect inspect <binary>
//!   badapple-dissect strings <binary> [--min N] [--grep <pat>]
//!   badapple-dissect audit <binary> [--save]
//!   badapple-dissect diff <old> <new>
//!   badapple-dissect status

use anyhow::{Context, Result};
use haruspex::{audit_file, diff, strings, Report};
use std::fs;
use std::path::PathBuf;

const ROOT: &str = "/var/lib/bad_apple/dissect";

fn root() -> PathBuf {
    std::env::var("BADAPPLE_DISSECT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| ROOT.into())
}

fn save_report(r: &Report) -> Result<PathBuf> {
    let dir = root().join("reports");
    fs::create_dir_all(&dir)?;
    let path = dir.join(format!("{}.json", &r.sha256[..12]));
    fs::write(&path, serde_json::to_string_pretty(r)?)?;
    Ok(path)
}

fn usage() -> ! {
    eprintln!(
        "badapple-dissect — see inside binaries (read-only)\n\n\
         inspect <binary>               slices, dylibs, symbols, entitlements\n\
         strings <binary> [--min N] [--grep <pat>]\n\
         audit <binary> [--save]        full report + flags (--save keeps it)\n\
         diff <old> <new>               what changed between two binaries\n\
         status                         report inventory"
    );
    std::process::exit(2);
}

fn arg_flag<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.windows(2)
        .find(|w| w[0] == name)
        .map(|w| w[1].as_str())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        usage();
    }
    match args[0].as_str() {
        "inspect" => {
            let path = args.get(1).context("inspect needs a binary path")?;
            let r = audit_file(std::path::Path::new(path))?;
            for s in &r.slices {
                println!(
                    "{} {} {} ({} sections, {} imports, {} exports)",
                    s.arch,
                    s.filetype,
                    s.uuid.as_deref().unwrap_or("no-uuid"),
                    s.sections.len(),
                    s.imports.len(),
                    s.exports.len()
                );
                for d in &s.dylibs {
                    println!("  dylib      {}", d);
                }
                for p in &s.rpaths {
                    println!("  rpath      {}", p);
                }
                if let Some(name) = &s.install_name {
                    println!("  install    {}", name);
                }
                if let Some(ent) = &s.entitlements {
                    println!("  entitlements:");
                    for line in ent.lines() {
                        println!("    {}", line);
                    }
                }
                if !s.objc_classes.is_empty() {
                    println!("  objc classes ({}):", s.objc_classes.len());
                    for c in &s.objc_classes {
                        println!("    {}", c);
                    }
                }
            }
            Ok(())
        }
        "strings" => {
            let path = args.get(1).context("strings needs a binary path")?;
            let min: usize = arg_flag(&args, "--min")
                .and_then(|v| v.parse().ok())
                .unwrap_or(5);
            let grep = arg_flag(&args, "--grep");
            let bytes = fs::read(path)?;
            for s in strings(&bytes, min) {
                if let Some(g) = grep {
                    if !s.text.contains(g) {
                        continue;
                    }
                }
                println!("{:>8x} {}{}", s.offset, s.text, if s.wide { " (wide)" } else { "" });
            }
            Ok(())
        }
        "audit" => {
            let path = args.get(1).context("audit needs a binary path")?;
            let r = audit_file(std::path::Path::new(path))?;
            println!("{}", serde_json::to_string_pretty(&r)?);
            if args.iter().any(|a| a == "--save") {
                let saved = save_report(&r)?;
                eprintln!("saved {}", saved.display());
            }
            Ok(())
        }
        "diff" => {
            let (old, new) = (
                args.get(1).context("diff needs two binary paths")?,
                args.get(2).context("diff needs two binary paths")?,
            );
            let a = audit_file(std::path::Path::new(old))?;
            let b = audit_file(std::path::Path::new(new))?;
            println!("{}", serde_json::to_string_pretty(&diff(&a, &b))?);
            Ok(())
        }
        "status" => {
            let dir = root().join("reports");
            let n = fs::read_dir(&dir).map(|d| d.count()).unwrap_or(0);
            println!("{} saved reports under {}", n, dir.display());
            Ok(())
        }
        _ => {
            eprintln!("unknown subcommand: {}", args[0]);
            usage();
        }
    }
}
