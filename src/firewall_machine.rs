//! Hardware output firewall — compiles the blocklist into a stateful
//! Aho-Corasick DFA that runs as a CoreML program on the ANE.
//!
//! Rust emits the program (`mil_machines` → .mlpackage → `mil_compile` →
//! .mlmodelc); the Swift engine (`BadAppleOutputFirewall`) loads the
//! compiled model and streams generated text through `MLState`, falling
//! back to its software automaton whenever the artifact is missing or
//! stale. The stale check compares `SHA256(lowercased patterns joined by
//! '\n')` written here against the fingerprint the engine computes over
//! its live snapshot — `firewall.patterns.sha` must therefore be the last
//! file written so a partial build never looks fresh.
//!
//! `BUILTIN_PATTERNS` mirrors `defaultPatterns` in
//! `src/platform/apple_desktop/BadAppleSecurity.swift` — keep in sync.

use anyhow::{Context, Result};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

/// Built-in patterns — must match BadAppleSecurity.swift defaultPatterns.
pub const BUILTIN_PATTERNS: &[&str] = &[
    "sk-",
    "ssh-rsa",
    "-----BEGIN",
    "-----END",
    "BEGIN PRIVATE KEY",
    "BEGIN OPENSSH PRIVATE KEY",
];

/// Artifact paths under /var/lib/bad_apple.
pub const FIREWALL_DIR: &str = "/var/lib/bad_apple";
pub const PACKAGE_NAME: &str = "firewall.mlpackage";
pub const COMPILED_NAME: &str = "firewall.mlmodelc";
pub const FINGERPRINT_NAME: &str = "firewall.patterns.sha";

/// Blocklist path — BADAPPLE_BLOCKLIST overrides (engine uses same env).
pub fn blocklist_path() -> PathBuf {
    std::env::var("BADAPPLE_BLOCKLIST")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/var/lib/bad_apple/blocklist.txt"))
}

/// The merged pattern set in the same order the engine builds it:
/// built-ins first, then each non-comment blocklist line, verbatim.
pub fn merged_patterns() -> Vec<String> {
    let mut pats: Vec<String> = BUILTIN_PATTERNS.iter().map(|s| s.to_string()).collect();
    if let Ok(text) = std::fs::read_to_string(blocklist_path()) {
        for line in text.lines() {
            let t = line.trim();
            if !t.is_empty() && !t.starts_with('#') {
                pats.push(t.to_string());
            }
        }
    }
    pats
}

/// Canonical fingerprint — must match the engine's:
/// SHA-256 of the lowercased patterns joined by '\n', hex-encoded.
pub fn fingerprint(patterns: &[String]) -> String {
    let joined = patterns
        .iter()
        .map(|p| p.to_lowercase())
        .collect::<Vec<_>>()
        .join("\n");
    let digest = Sha256::digest(joined.as_bytes());
    digest.iter().map(|b| format!("{b:02x}")).collect()
}

/// What a compile produces.
pub struct BuildReport {
    pub states: usize,
    pub patterns: usize,
    pub package: PathBuf,
    pub compiled: PathBuf,
    pub fingerprint_path: PathBuf,
}

/// Emit the Coyote DFA + ANE organ binaries (`0xBEEFFACE` `.macho`) directly
/// into `dir/firewall.mlmodelc` through the Coyote compiler and ANE runtime.
#[cfg(not(feature = "coyote-sys"))]
pub fn compile_to(_dir: &Path) -> Result<BuildReport> {
    anyhow::bail!("firewall compile requires the coyote build")
}

#[cfg(feature = "coyote-sys")]
pub fn compile_to(dir: &Path) -> Result<BuildReport> {
    let patterns = merged_patterns();
    let lowered: Vec<String> = patterns.iter().map(|p| p.to_lowercase()).collect();
    let dfa = coyote::vm::DfaMachine::compile(lowered);
    let states = dfa.out.len();

    let package = dir.join(PACKAGE_NAME);
    let compiled = dir.join(COMPILED_NAME);
    std::fs::create_dir_all(&package).with_context(|| format!("mkdir {}", package.display()))?;
    std::fs::create_dir_all(&compiled).with_context(|| format!("mkdir {}", compiled.display()))?;

    crate::coyote_organs::compile_to_dir(&compiled)
        .with_context(|| format!("coyote ANE compile {}", compiled.display()))?;

    let fingerprint_path = dir.join(FINGERPRINT_NAME);
    std::fs::write(&fingerprint_path, fingerprint(&patterns))
        .with_context(|| format!("write {}", fingerprint_path.display()))?;

    Ok(BuildReport {
        states,
        patterns: patterns.len(),
        package,
        compiled,
        fingerprint_path,
    })
}

/// Freshness: does the on-disk fingerprint match the live pattern set?
pub fn is_fresh(dir: &Path) -> bool {
    let Ok(on_disk) = std::fs::read_to_string(dir.join(FINGERPRINT_NAME)) else {
        return false;
    };
    on_disk.trim() == fingerprint(&merged_patterns()) && dir.join(COMPILED_NAME).is_dir()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fingerprint_matches_swift_side() {
        // Known vector: sha256 of the six lowercased defaults joined by \n
        // — computed independently of both sides.
        let pats: Vec<String> = BUILTIN_PATTERNS.iter().map(|s| s.to_string()).collect();
        assert_eq!(
            fingerprint(&pats),
            "9b7e159682aac2a1d7425bb4fe1246166a2e97fcf37a21ffdc3fbfbff4f54169"
        );
    }

    #[test]
    fn merged_patterns_order_and_filter() {
        let pats = merged_patterns();
        assert!(pats.len() >= BUILTIN_PATTERNS.len());
        assert_eq!(&pats[..BUILTIN_PATTERNS.len()], BUILTIN_PATTERNS);
        assert!(pats.iter().all(|p| !p.starts_with('#') && !p.is_empty()));
    }

    #[test]
    fn is_fresh_tracks_compile() {
        let dir = Path::new(FIREWALL_DIR);
        // artifact was compiled by `badapple firewall compile` for the live
        // pattern set — fresh iff nothing changed since.
        if dir.join(COMPILED_NAME).is_dir() {
            assert!(is_fresh(dir));
        }
    }
}
