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
//!   badapple-dissect funcs <binary>            function table + exports
//!   badapple-dissect disasm <binary> <hexaddr> disasm the function at addr
//!   badapple-dissect hunt <binary> <substr>    string -> xref -> disasm context
//!   badapple-dissect cache list [grep]           dyld cache image index
//!   badapple-dissect cache audit <name> [--save] carve + audit a cached image
//!   badapple-dissect cache extract <name> [-o out]
//!   badapple-dissect cache hunt <name> <substr>  same hunt on a cached image
//!   badapple-dissect status

use anyhow::{Context, Result};
use haruspex::{
    audit_bytes, audit_file,
    cache::SharedCache,
    code::{disasm_listing, index_code, CodeIndex},
    diff, parse, strings, Report, Slice,
};
use std::fs;
use std::path::{Path, PathBuf};

const ROOT: &str = "/var/lib/bad_apple/dissect";
// Darwin 27 keeps the cache inside the OS cryptex, not /System/Library.
const DYLD_CACHE: &str = "/System/Cryptexes/OS/System/Library/dyld/dyld_shared_cache_arm64e";

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
         funcs <binary>                function table + exported symbols\n\
         disasm <binary> <hexaddr>      disasm the function containing addr\n\
         hunt <binary> <substr>         string -> xrefs -> disasm context\n\
         cache list [grep]              images inside the dyld shared cache\n\
         cache audit <name> [--save]    carve + audit a cached framework\n\
         cache extract <name> [-o out]  carve an image to disk\n\
         cache funcs <name>             function table of a cached image\n\
         cache disasm <name> <hexaddr>  disasm inside a cached image\n\
         cache hunt <name> <substr>     xref a string inside a cached image\n\
         status                         report inventory"
    );
    std::process::exit(2);
}

fn arg_flag<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.windows(2)
        .find(|w| w[0] == name)
        .map(|w| w[1].as_str())
}

/// Function-table + export listing shared by file and cache paths.
fn show_funcs(slice: &Slice, idx: &CodeIndex) {
    println!(
        "{} functions, {} exports, {} xrefs",
        idx.functions.len(),
        idx.exports.len(),
        idx.xrefs.len()
    );
    let _ = slice;
    for (_, n) in &idx.exports {
        println!("  export {n}");
    }
}

/// Disassemble the function containing `addr` (hex or decimal).
fn show_disasm(bytes: &[u8], slice: &Slice, idx: &CodeIndex, addr: u64) -> Result<()> {
    let f = idx
        .function_at(addr)
        .with_context(|| format!("no function contains {addr:#x}"))?;
    println!("{} ({:#x}..{:#x})", f.label(), f.addr, f.end);
    let t = slice.text().context("no __text section")?;
    let lo = (f.addr - t.addr) as usize;
    let hi = (f.end - t.addr) as usize;
    let code = &bytes[t.fileoff as usize + lo..t.fileoff as usize + hi];
    for line in disasm_listing(code, f.addr) {
        println!("  {line}");
    }
    Ok(())
}

/// The hunt: for every string matching `grep`, find code references to
/// it and disassemble the surrounding context.
fn hunt(bytes: &[u8], slice: &Slice, idx: &CodeIndex, grep: &str) -> Result<()> {
    let mut found = false;
    for os in strings(bytes, 6) {
        if !os.text.contains(grep) {
            continue;
        }
        let Some(va) = slice.vmaddr_of_fileoff(os.offset) else {
            continue;
        };
        let sites: Vec<u64> = idx
            .xrefs
            .iter()
            .filter(|x| x.target == va)
            .map(|x| x.site)
            .collect();
        found = true;
        if sites.is_empty() {
            println!("{:?} — no code refs", os.text);
            continue;
        }
        println!("{:?} @ {va:#x}", os.text);
        for site in sites.iter().take(4) {
            let f = match idx.function_at(*site) {
                Some(f) => f,
                None => continue,
            };
            println!(
                "  ref {site:#x} in {} ({:#x}..{:#x})",
                f.label(),
                f.addr,
                f.end
            );
            let t = slice.text().context("no __text")?;
            let lo = site.saturating_sub(96).max(f.addr);
            let hi = (site + 24).min(f.end);
            let a = t.fileoff as usize + (lo - t.addr) as usize;
            let b = t.fileoff as usize + (hi - t.addr) as usize;
            for line in disasm_listing(&bytes[a..b], lo) {
                let mark = if line.starts_with(&format!("{site:x}:")) {
                    " <=="
                } else {
                    ""
                };
                println!("      {line}{mark}");
            }
        }
    }
    if !found {
        eprintln!("no strings matching {grep:?}");
    }
    Ok(())
}

fn hex_or_dec(s: &str) -> Result<u64> {
    let s = s.strip_prefix("0x").unwrap_or(s);
    Ok(u64::from_str_radix(s, 16).or_else(|_| s.parse::<u64>())?)
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
                println!(
                    "{:>8x} {}{}",
                    s.offset,
                    s.text,
                    if s.wide { " (wide)" } else { "" }
                );
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
        "funcs" | "disasm" | "hunt" => {
            let path = args.get(1).context("needs a binary path")?;
            let bytes = fs::read(path)?;
            let m = parse(&bytes)?;
            let slice = m.slices.first().context("no slice")?;
            let idx = index_code(&bytes, slice, |off, len| {
                let (lo, hi) = (off as usize, off as usize + len as usize);
                bytes
                    .get(lo..hi)
                    .map(|s| s.to_vec())
                    .ok_or(haruspex::Error::Malformed("range outside file"))
            })?;
            match args[0].as_str() {
                "funcs" => {
                    show_funcs(slice, &idx);
                    Ok(())
                }
                "disasm" => {
                    let a = hex_or_dec(args.get(2).context("disasm needs an address")?)?;
                    show_disasm(&bytes, slice, &idx, a)
                }
                _ => {
                    let g = args.get(2).context("hunt needs a substring")?;
                    hunt(&bytes, slice, &idx, g)
                }
            }
        }
        "cache" => {
            let mut cache = SharedCache::open(Path::new(
                std::env::var("BADAPPLE_DYLD_CACHE")
                    .as_deref()
                    .unwrap_or(DYLD_CACHE),
            ))?;
            match args.get(1).map(|s| s.as_str()) {
                Some("list") => {
                    cache.enumerate_images()?;
                    let grep = args.get(2).map(|s| s.as_str());
                    for img in &cache.images {
                        if let Some(g) = grep {
                            if !img.path.contains(g) {
                                continue;
                            }
                        }
                        println!("{}", img.path);
                    }
                }
                Some("audit") => {
                    let name = args.get(2).context("cache audit needs an image name")?;
                    let (path, bytes) = cache.extract(name)?;
                    eprintln!("carved {} ({} bytes)", path, bytes.len());
                    let r = audit_bytes(&bytes, &path)?;
                    println!("{}", serde_json::to_string_pretty(&r)?);
                    if args.iter().any(|a| a == "--save") {
                        let saved = save_report(&r)?;
                        eprintln!("saved {}", saved.display());
                    }
                }
                Some("extract") => {
                    let name = args.get(2).context("cache extract needs an image name")?;
                    let out = args
                        .windows(2)
                        .find(|w| w[0] == "-o")
                        .map(|w| w[1].clone())
                        .unwrap_or_else(|| {
                            format!("/tmp/{}.carved", name.rsplit('/').next().unwrap_or("image"))
                        });
                    let (path, bytes) = cache.extract(name)?;
                    fs::write(&out, &bytes)?;
                    eprintln!("{} -> {} ({} bytes)", path, out, bytes.len());
                }
                Some("funcs") | Some("disasm") | Some("hunt") => {
                    let sub = args[1].as_str();
                    let name = args.get(2).context("needs an image name")?;
                    let (_, bytes) = cache.extract(name)?;
                    let m = parse(&bytes)?;
                    let slice = m.slices.first().context("no slice")?;
                    let (_, idx) = cache.code_index(name)?;
                    (match sub {
                        "funcs" => {
                            show_funcs(slice, &idx);
                            Ok(())
                        }
                        "disasm" => {
                            let a = hex_or_dec(args.get(3).context("disasm needs an address")?)?;
                            show_disasm(&bytes, slice, &idx, a)
                        }
                        _ => {
                            let g = args.get(3).context("hunt needs a substring")?;
                            hunt(&bytes, slice, &idx, g)
                        }
                    })?
                }
                _ => {
                    eprintln!("cache needs list|audit|extract|funcs|disasm|hunt");
                    usage();
                }
            }
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
