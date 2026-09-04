//! Local encrypted vault for user secrets and API keys.
//!
//! This replaces `badapple_vault.py` with a Rust implementation that encrypts
//! secrets at rest using AES-256-GCM. The master key is derived from the
//! SLICKS secret, so the vault is tied to the Bad Apple install. In a future
//! iteration the master key can be wrapped by the Secure Enclave identity agent.

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

const VAULT_FILE: &str = "vault.enc";

#[derive(Serialize, Deserialize, Default)]
struct VaultPayload {
    entries: HashMap<String, String>,
    version: u32,
}

/// Encrypted key-value vault for API keys, tokens, and other user secrets.
pub struct BadAppleVault {
    path: PathBuf,
    cipher: crate::p2p_crypto::P2PCipher,
}

impl BadAppleVault {
    /// Open the default vault under the Bad Apple data directory.
    pub fn open_default() -> Result<Self> {
        let data_dir = std::env::var("BADAPPLE_DATA_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("/var/lib/bad_apple"));
        Self::open(data_dir.join(VAULT_FILE))
    }

    /// Open or create a vault at the given path.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let key = Self::derive_master_key()?;
        let cipher = crate::p2p_crypto::P2PCipher::new(&key)
            .map_err(|e| anyhow::anyhow!("failed to initialize vault cipher: {e}"))?;
        Ok(Self { path, cipher })
    }

    fn derive_master_key() -> Result<[u8; 32]> {
        if let Ok(secret) = std::env::var("BADAPPLE_VAULT_KEY") {
            return Ok(crate::p2p_crypto::derive_key(&secret));
        }
        if let Ok(path) = std::env::var("BADAPPLE_SLICKS_KEY_PATH") {
            let key = fs::read_to_string(&path)
                .with_context(|| format!("failed to read SLICKS key from {path}"))?;
            return Ok(crate::p2p_crypto::derive_key_from_bytes(
                key.trim().as_bytes(),
            ));
        }
        if let Ok(secret) = std::env::var("BADAPPLE_SLICKS_SECRET") {
            return Ok(crate::p2p_crypto::derive_key_from_bytes(secret.as_bytes()));
        }
        bail!("vault requires BADAPPLE_VAULT_KEY, BADAPPLE_SLICKS_KEY_PATH, or BADAPPLE_SLICKS_SECRET")
    }

    /// Store a secret in the vault.
    pub fn set(&self, key: &str, value: &str) -> Result<()> {
        if key.is_empty() || key.contains('/') || key.contains("..") {
            bail!("invalid vault key");
        }
        let mut payload = self.load()?;
        payload.entries.insert(key.to_string(), value.to_string());
        self.save(&payload)
    }

    /// Retrieve a secret from the vault.
    pub fn get(&self, key: &str) -> Result<Option<String>> {
        let payload = self.load()?;
        Ok(payload.entries.get(key).cloned())
    }

    /// Remove a secret from the vault.
    pub fn remove(&self, key: &str) -> Result<bool> {
        let mut payload = self.load()?;
        let removed = payload.entries.remove(key).is_some();
        self.save(&payload)?;
        Ok(removed)
    }

    /// List all vault keys (not values).
    pub fn list(&self) -> Result<Vec<String>> {
        let payload = self.load()?;
        let mut keys: Vec<String> = payload.entries.keys().cloned().collect();
        keys.sort();
        Ok(keys)
    }

    fn load(&self) -> Result<VaultPayload> {
        if !self.path.exists() {
            return Ok(VaultPayload {
                entries: HashMap::new(),
                version: 1,
            });
        }
        let ciphertext = fs::read(&self.path).context("failed to read vault file")?;
        let plaintext = self
            .cipher
            .decrypt(&ciphertext)
            .map_err(|e| anyhow::anyhow!("failed to decrypt vault: {e}"))?;
        let payload: VaultPayload =
            serde_json::from_slice(&plaintext).context("vault is corrupt")?;
        Ok(payload)
    }

    fn save(&self, payload: &VaultPayload) -> Result<()> {
        let parent = self.path.parent().context("vault path has no parent")?;
        fs::create_dir_all(parent)?;
        let plaintext = serde_json::to_vec(payload).context("failed to serialize vault")?;
        let ciphertext = self
            .cipher
            .encrypt(&plaintext)
            .map_err(|e| anyhow::anyhow!("failed to encrypt vault: {e}"))?;
        let tmp = self.path.with_extension("tmp");
        fs::write(&tmp, &ciphertext).context("failed to write vault temp file")?;
        fs::rename(&tmp, &self.path).context("failed to finalize vault file")?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    fn tmp_vault_path() -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("badapple_vault_test_{}", std::process::id()));
        let _ = fs::remove_file(&p);
        p
    }

    #[test]
    fn round_trip() {
        let path = tmp_vault_path();
        env::set_var("BADAPPLE_VAULT_KEY", "test-key-for-vault");
        let vault = BadAppleVault::open(&path).unwrap();
        vault.set("openai_api_key", "sk-test").unwrap();
        let value = vault.get("openai_api_key").unwrap();
        assert_eq!(value, Some("sk-test".to_string()));
        let keys = vault.list().unwrap();
        assert_eq!(keys, vec!["openai_api_key"]);
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn wrong_key_fails() {
        let path = tmp_vault_path();
        env::set_var("BADAPPLE_VAULT_KEY", "test-key-for-vault");
        let vault = BadAppleVault::open(&path).unwrap();
        vault.set("openai_api_key", "sk-test").unwrap();

        env::set_var("BADAPPLE_VAULT_KEY", "wrong-key");
        let result = BadAppleVault::open(&path).unwrap().get("openai_api_key");
        assert!(result.is_err());
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn rejects_invalid_keys() {
        let path = tmp_vault_path();
        env::set_var("BADAPPLE_VAULT_KEY", "test-key-for-vault");
        let vault = BadAppleVault::open(&path).unwrap();
        assert!(vault.set("", "x").is_err());
        assert!(vault.set("foo/bar", "x").is_err());
        assert!(vault.set("foo..bar", "x").is_err());
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn remove_and_list() {
        let path = tmp_vault_path();
        env::set_var("BADAPPLE_VAULT_KEY", "test-key-for-vault");
        let vault = BadAppleVault::open(&path).unwrap();
        assert_eq!(vault.list().unwrap(), Vec::<String>::new());
        vault.set("a", "1").unwrap();
        vault.set("b", "2").unwrap();
        assert_eq!(vault.list().unwrap(), vec!["a", "b"]);
        assert!(vault.remove("a").unwrap());
        assert!(!vault.remove("missing").unwrap());
        assert_eq!(vault.list().unwrap(), vec!["b"]);
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn missing_key_returns_none() {
        let path = tmp_vault_path();
        env::set_var("BADAPPLE_VAULT_KEY", "test-key-for-vault");
        let vault = BadAppleVault::open(&path).unwrap();
        assert_eq!(vault.get("nonexistent").unwrap(), None);
        let _ = fs::remove_file(&path);
    }
}
