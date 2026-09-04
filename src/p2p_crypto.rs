//! P2P payload encryption using AES-256-GCM.
//!
//! This module adds confidentiality to the signed `CompactEngramPacket`
//! transport in `protocol.rs`. Each packet is signed first (HMAC-SHA256)
//! and then encrypted with a per-session AES-256-GCM key derived from an
//! ECDH key exchange or pre-shared key.
//!
//! For the initial implementation, peers use a pre-shared 256-bit key
//! distributed out of band (e.g. displayed as a QR code in the menu bar) or
//! derived from the SLICKS secret already shared between trusted peers.

use aes_gcm::{
    aead::{Aead, KeyInit, Payload},
    Aes256Gcm, Nonce,
};
use rand::RngCore;

/// Length of the random nonce prepended to each ciphertext.
const NONCE_LEN: usize = 12;

/// A thin wrapper around AES-256-GCM for P2P packet encryption.
pub struct P2PCipher {
    cipher: Aes256Gcm,
}

impl P2PCipher {
    /// Create a cipher from a 32-byte (256-bit) key.
    pub fn new(key: &[u8]) -> Result<Self, String> {
        if key.len() != 32 {
            return Err(format!("P2P key must be 32 bytes, got {}", key.len()));
        }
        let cipher = Aes256Gcm::new_from_slice(key)
            .map_err(|e| format!("AES-256-GCM key init failed: {:?}", e))?;
        Ok(Self { cipher })
    }

    /// Encrypt `plaintext` and return `[nonce || ciphertext || tag]`.
    pub fn encrypt(&self, plaintext: &[u8]) -> Result<Vec<u8>, String> {
        let mut nonce_bytes = [0u8; NONCE_LEN];
        rand::thread_rng().fill_bytes(&mut nonce_bytes);
        let nonce = Nonce::from_slice(&nonce_bytes);

        let ciphertext = self
            .cipher
            .encrypt(nonce, Payload::from(plaintext))
            .map_err(|e| format!("AES-256-GCM encrypt failed: {:?}", e))?;

        let mut out = Vec::with_capacity(NONCE_LEN + ciphertext.len());
        out.extend_from_slice(&nonce_bytes);
        out.extend_from_slice(&ciphertext);
        Ok(out)
    }

    /// Decrypt a buffer of the form `[nonce || ciphertext || tag]`.
    pub fn decrypt(&self, ciphertext: &[u8]) -> Result<Vec<u8>, String> {
        if ciphertext.len() < NONCE_LEN {
            return Err("ciphertext too short for nonce".to_string());
        }
        let (nonce_bytes, body) = ciphertext.split_at(NONCE_LEN);
        let nonce = Nonce::from_slice(nonce_bytes);

        self.cipher
            .decrypt(nonce, Payload::from(body))
            .map_err(|e| format!("AES-256-GCM decrypt failed: {:?}", e))
    }
}

/// Derive a 32-byte AES key from an arbitrary-length pre-shared secret using
/// SHA-256. This lets peers re-use an existing strong passphrase.
pub fn derive_key(secret: &str) -> [u8; 32] {
    derive_key_from_bytes(secret.as_bytes())
}

/// Derive a 32-byte AES key from raw bytes using SHA-256.
pub fn derive_key_from_bytes(secret: &[u8]) -> [u8; 32] {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(secret);
    hasher.finalize().into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip() {
        let key = [0x42u8; 32];
        let cipher = P2PCipher::new(&key).unwrap();
        let plaintext = b"hello p2p mesh";
        let encrypted = cipher.encrypt(plaintext).unwrap();
        assert!(encrypted.len() > NONCE_LEN + plaintext.len());
        let decrypted = cipher.decrypt(&encrypted).unwrap();
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn wrong_key_fails() {
        let key = [0x42u8; 32];
        let cipher = P2PCipher::new(&key).unwrap();
        let encrypted = cipher.encrypt(b"secret").unwrap();
        let wrong = P2PCipher::new(&[0x43u8; 32]).unwrap();
        assert!(wrong.decrypt(&encrypted).is_err());
    }

    #[test]
    fn key_derivation_is_deterministic() {
        let k1 = derive_key("my preshared secret");
        let k2 = derive_key("my preshared secret");
        assert_eq!(k1, k2);
    }

    #[test]
    fn rejects_short_key() {
        let short = [0x42u8; 31];
        assert!(
            P2PCipher::new(&short).is_err(),
            "31-byte key must be rejected"
        );
    }

    #[test]
    fn rejects_short_ciphertext() {
        let key = [0x42u8; 32];
        let cipher = P2PCipher::new(&key).unwrap();
        assert!(
            cipher.decrypt(&[0u8; 5]).is_err(),
            "ciphertext shorter than nonce must fail"
        );
    }

    #[test]
    fn round_trip_empty_plaintext() {
        let key = [0x42u8; 32];
        let cipher = P2PCipher::new(&key).unwrap();
        let encrypted = cipher.encrypt(b"").unwrap();
        let decrypted = cipher.decrypt(&encrypted).unwrap();
        assert_eq!(decrypted, b"");
    }

    #[test]
    fn rejects_tampered_ciphertext() {
        let key = [0x42u8; 32];
        let cipher = P2PCipher::new(&key).unwrap();
        let mut encrypted = cipher.encrypt(b"hello").unwrap();
        let last = encrypted.len() - 1;
        encrypted[last] = encrypted[last].wrapping_add(1);
        assert!(cipher.decrypt(&encrypted).is_err());
    }
}
