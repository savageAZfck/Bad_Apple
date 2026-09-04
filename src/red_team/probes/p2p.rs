//! P2P crypto probes — tampering and wrong-key attempts.

use super::{elapsed_us, Probe, ProbeResult, Severity};
use crate::p2p_crypto::{derive_key, P2PCipher};
use std::time::Instant;

/// Probe that verifies a tampered P2P ciphertext is rejected.
pub struct P2PTamperingProbe;

impl Probe for P2PTamperingProbe {
    fn id(&self) -> &'static str {
        "p2p.tampered_ciphertext"
    }
    fn category(&self) -> &'static str {
        "p2p"
    }
    fn severity(&self) -> Severity {
        Severity::High
    }
    fn title(&self) -> &'static str {
        "P2P crypto rejects tampered ciphertext"
    }
    fn description(&self) -> &'static str {
        "Encrypts a message, flips the last byte of the ciphertext, and verifies decryption fails."
    }
    fn run(&self) -> ProbeResult {
        let start = Instant::now();
        let key = derive_key("redteam-tamper-test");
        let cipher = P2PCipher::new(&key).expect("valid key");
        let plaintext = b"hello world".to_vec();
        let mut ciphertext = cipher.encrypt(&plaintext).expect("encryption succeeds");

        if let Some(last) = ciphertext.last_mut() {
            *last = last.wrapping_add(1);
        }

        let mitigated = cipher.decrypt(&ciphertext).is_err();

        ProbeResult {
            probe: self.id().to_string(),
            category: self.category().to_string(),
            mitigated,
            severity: self.severity(),
            title: self.title().to_string(),
            description: self.description().to_string(),
            payload: Some(hex::encode(&ciphertext)),
            timestamp_ms: crate::red_team::now_ms(),
            duration_us: elapsed_us(start),
        }
    }
}

/// Probe that verifies a wrong P2P key fails to decrypt.
pub struct P2PWrongKeyProbe;

impl Probe for P2PWrongKeyProbe {
    fn id(&self) -> &'static str {
        "p2p.wrong_key"
    }
    fn category(&self) -> &'static str {
        "p2p"
    }
    fn severity(&self) -> Severity {
        Severity::High
    }
    fn title(&self) -> &'static str {
        "P2P crypto rejects decryption with wrong key"
    }
    fn description(&self) -> &'static str {
        "Encrypts with one key and attempts to decrypt with another derived key."
    }
    fn run(&self) -> ProbeResult {
        let start = Instant::now();
        let k1 = derive_key("key-one");
        let k2 = derive_key("key-two");
        let c1 = P2PCipher::new(&k1).expect("valid key");
        let c2 = P2PCipher::new(&k2).expect("valid key");
        let ciphertext = c1.encrypt(b"secret").expect("encryption succeeds");

        let mitigated = c2.decrypt(&ciphertext).is_err();

        ProbeResult {
            probe: self.id().to_string(),
            category: self.category().to_string(),
            mitigated,
            severity: self.severity(),
            title: self.title().to_string(),
            description: self.description().to_string(),
            payload: Some(hex::encode(&ciphertext)),
            timestamp_ms: crate::red_team::now_ms(),
            duration_us: elapsed_us(start),
        }
    }
}
