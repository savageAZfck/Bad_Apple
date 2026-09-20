//! Org-signed policy: Ed25519 signatures over `policy.yaml`.
//!
//! Personal installs are sovereign: the owner edits `policy.yaml` directly
//! and no trust root exists. An organization can pin a trust root at
//! `/var/lib/bad_apple/org_policy.pub`; once that file exists, the policy
//! engine refuses to load `policy.yaml` unless it verifies against
//! `policy.yaml.sig` signed by the matching secret key. Tampered or
//! unsigned policy fails closed (deny-all), and the denial is explicit
//! rather than silent.
//!
//! Key and signature files are hex-encoded, one value per file, matching
//! the format the Swift policy engine verifies via CryptoKit.

use anyhow::{bail, Context, Result};
use ed25519_dalek::{Signature, Signer, SigningKey, VerifyingKey};
use std::fs;
use std::path::{Path, PathBuf};

/// Policy file the signature covers.
pub const DEFAULT_POLICY_PATH: &str = "/var/lib/bad_apple/policy.yaml";
/// Detached hex signature written next to the policy.
pub const DEFAULT_SIG_PATH: &str = "/var/lib/bad_apple/policy.yaml.sig";
/// Pinned org trust root. Its presence switches the engine into
/// org-verified mode; its absence means personal sovereignty.
pub const DEFAULT_PUB_PATH: &str = "/var/lib/bad_apple/org_policy.pub";
/// Default location for the org signing secret (should live offline,
/// not on managed machines).
pub const DEFAULT_KEY_PATH: &str = "/var/lib/bad_apple/org_policy.key";

/// Whether an org trust root is installed on this machine.
pub fn org_mode_active() -> bool {
    Path::new(DEFAULT_PUB_PATH).exists()
}

/// Generate a new org keypair into `out_dir` as `org_policy.key` /
/// `org_policy.pub`. Returns (key_path, pub_path). Refuses to overwrite
/// existing files.
pub fn keygen(out_dir: impl AsRef<Path>) -> Result<(PathBuf, PathBuf)> {
    let dir = out_dir.as_ref();
    fs::create_dir_all(dir).with_context(|| format!("failed to create {}", dir.display()))?;
    let key_path = dir.join("org_policy.key");
    let pub_path = dir.join("org_policy.pub");
    for p in [&key_path, &pub_path] {
        if p.exists() {
            bail!("refusing to overwrite existing file: {}", p.display());
        }
    }
    let mut secret = [0u8; 32];
    rand::Rng::fill(&mut rand::thread_rng(), &mut secret);
    let signing = SigningKey::from_bytes(&secret);
    let verifying = signing.verifying_key();
    write_hex_secret(&key_path, signing.to_bytes().as_slice())?;
    fs::write(&pub_path, hex::encode(verifying.to_bytes()))
        .with_context(|| format!("failed to write {}", pub_path.display()))?;
    Ok((key_path, pub_path))
}

/// Sign `policy_path` with the secret at `key_path`, writing the hex
/// signature to `sig_path`.
pub fn sign_file(
    policy_path: impl AsRef<Path>,
    key_path: impl AsRef<Path>,
    sig_path: impl AsRef<Path>,
) -> Result<()> {
    let policy = fs::read(policy_path.as_ref())
        .with_context(|| format!("failed to read {}", policy_path.as_ref().display()))?;
    let signing = load_signing_key(key_path.as_ref())?;
    let sig = signing.sign(&policy);
    fs::write(sig_path.as_ref(), hex::encode(sig.to_bytes()))
        .with_context(|| format!("failed to write {}", sig_path.as_ref().display()))?;
    Ok(())
}

/// Verify `policy_path` against `sig_path` using the pinned public key.
/// Returns `Ok(true)` when valid; `Ok(false)` when the signature is
/// missing or does not verify; `Err` on unreadable inputs.
pub fn verify_file(
    policy_path: impl AsRef<Path>,
    sig_path: impl AsRef<Path>,
    pub_path: impl AsRef<Path>,
) -> Result<bool> {
    let policy = fs::read(policy_path.as_ref())
        .with_context(|| format!("failed to read {}", policy_path.as_ref().display()))?;
    let verifying = load_verifying_key(pub_path.as_ref())?;
    let sig_hex = match fs::read_to_string(sig_path.as_ref()) {
        Ok(s) => s,
        Err(_) => return Ok(false),
    };
    let sig_bytes = match hex::decode(sig_hex.trim()) {
        Ok(b) => b,
        Err(_) => return Ok(false),
    };
    let sig = match Signature::from_slice(&sig_bytes) {
        Ok(s) => s,
        Err(_) => return Ok(false),
    };
    Ok(verifying.verify_strict(&policy, &sig).is_ok())
}

fn load_signing_key(path: &Path) -> Result<SigningKey> {
    let hex_key =
        fs::read_to_string(path).with_context(|| format!("failed to read {}", path.display()))?;
    let bytes = hex::decode(hex_key.trim())
        .with_context(|| format!("invalid hex key in {}", path.display()))?;
    let arr: [u8; 32] = bytes
        .try_into()
        .map_err(|_| anyhow::anyhow!("org key must be 32 bytes: {}", path.display()))?;
    Ok(SigningKey::from_bytes(&arr))
}

fn load_verifying_key(path: &Path) -> Result<VerifyingKey> {
    let hex_key =
        fs::read_to_string(path).with_context(|| format!("failed to read {}", path.display()))?;
    let bytes = hex::decode(hex_key.trim())
        .with_context(|| format!("invalid hex pubkey in {}", path.display()))?;
    let arr: [u8; 32] = bytes
        .try_into()
        .map_err(|_| anyhow::anyhow!("org pubkey must be 32 bytes: {}", path.display()))?;
    VerifyingKey::from_bytes(&arr)
        .with_context(|| format!("invalid org pubkey in {}", path.display()))
}

fn write_hex_secret(path: &Path, bytes: &[u8]) -> Result<()> {
    fs::write(path, hex::encode(bytes))
        .with_context(|| format!("failed to write {}", path.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(path, fs::Permissions::from_mode(0o600))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sign_and_verify_round_trip() {
        let dir = std::env::temp_dir().join(format!("orgpolicy-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let (key, pub_key) = keygen(&dir).unwrap();
        let policy = dir.join("policy.yaml");
        let sig = dir.join("policy.yaml.sig");
        fs::write(&policy, "defaults:\n  allowed: true\n").unwrap();
        sign_file(&policy, &key, &sig).unwrap();
        assert!(verify_file(&policy, &sig, &pub_key).unwrap());

        // Tampered policy must not verify.
        fs::write(&policy, "defaults:\n  allowed: true\n  autopilot: true\n").unwrap();
        assert!(!verify_file(&policy, &sig, &pub_key).unwrap());

        // Missing signature must not verify.
        fs::remove_file(&sig).unwrap();
        assert!(!verify_file(&policy, &sig, &pub_key).unwrap());
        let _ = fs::remove_dir_all(&dir);
    }
}
