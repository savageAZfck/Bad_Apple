//! SLICKS IPC probes — replay and proof tampering.

use super::{elapsed_us, Probe, ProbeResult, Severity};
use crate::bad_apple_ipc::{client_proof, random_nonce, verify_client_proof, ReplayCache};
use std::time::Instant;

/// Probe that verifies the replay cache rejects reused nonce pairs.
pub struct ReplayCacheProbe;

impl Probe for ReplayCacheProbe {
    fn id(&self) -> &'static str {
        "slicks.replay_cache"
    }
    fn category(&self) -> &'static str {
        "slicks"
    }
    fn severity(&self) -> Severity {
        Severity::Critical
    }
    fn title(&self) -> &'static str {
        "SLICKS replay cache rejects reused nonce pairs"
    }
    fn description(&self) -> &'static str {
        "Inserts a valid nonce pair into the replay cache and then attempts to reuse it."
    }
    fn run(&self) -> ProbeResult {
        let start = Instant::now();
        let cache = ReplayCache::new(4096);
        let client_nonce = random_nonce();
        let server_nonce = random_nonce();

        let first = cache.check_and_insert(&client_nonce, &server_nonce);
        let second = cache.check_and_insert(&client_nonce, &server_nonce);
        let mitigated = first && !second;

        ProbeResult {
            probe: self.id().to_string(),
            category: self.category().to_string(),
            mitigated,
            severity: self.severity(),
            title: self.title().to_string(),
            description: self.description().to_string(),
            payload: Some(format!("{client_nonce}:{server_nonce}")),
            timestamp_ms: crate::red_team::now_ms(),
            duration_us: elapsed_us(start),
        }
    }
}

/// Probe that verifies a tampered client proof fails verification.
pub struct ClientProofTamperingProbe;

impl Probe for ClientProofTamperingProbe {
    fn id(&self) -> &'static str {
        "slicks.client_proof_tampering"
    }
    fn category(&self) -> &'static str {
        "slicks"
    }
    fn severity(&self) -> Severity {
        Severity::High
    }
    fn title(&self) -> &'static str {
        "SLICKS client proof verification rejects tampered signatures"
    }
    fn description(&self) -> &'static str {
        "Generates a valid client proof, mutates one hex character, and verifies the mutated proof is rejected."
    }
    fn run(&self) -> ProbeResult {
        let start = Instant::now();
        let secret = b"0123456789abcdef0123456789abcdef";
        let timestamp = 1_700_000_000_000u64;
        let client_nonce = random_nonce();
        let server_nonce = random_nonce();
        let prompt = "hello";
        let max_tokens = 32usize;

        let proof = client_proof(
            secret,
            timestamp,
            &client_nonce,
            &server_nonce,
            prompt,
            max_tokens,
        );
        let mut tampered = proof.clone();
        if let Some(last) = tampered.pop() {
            tampered.push(if last == '0' { '1' } else { '0' });
        }

        let valid = verify_client_proof(
            secret,
            timestamp,
            &client_nonce,
            &server_nonce,
            prompt,
            max_tokens,
            &proof,
        );
        let tampered_valid = verify_client_proof(
            secret,
            timestamp,
            &client_nonce,
            &server_nonce,
            prompt,
            max_tokens,
            &tampered,
        );

        let mitigated = valid && !tampered_valid;

        ProbeResult {
            probe: self.id().to_string(),
            category: self.category().to_string(),
            mitigated,
            severity: self.severity(),
            title: self.title().to_string(),
            description: self.description().to_string(),
            payload: Some(tampered),
            timestamp_ms: crate::red_team::now_ms(),
            duration_us: elapsed_us(start),
        }
    }
}
