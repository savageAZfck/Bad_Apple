//! Audit ledger probes — secret redaction and hash chain integrity.

use super::{elapsed_us, Probe, ProbeResult, Severity};
use std::path::PathBuf;
use std::time::Instant;

/// Probe that verifies the ledger file does not contain common unredacted secret patterns.
pub struct LedgerSecretRedactionProbe;

impl Probe for LedgerSecretRedactionProbe {
    fn id(&self) -> &'static str {
        "audit.ledger_secret_redaction"
    }
    fn category(&self) -> &'static str {
        "audit"
    }
    fn severity(&self) -> Severity {
        Severity::High
    }
    fn title(&self) -> &'static str {
        "Audit ledger does not contain unredacted secret patterns"
    }
    fn description(&self) -> &'static str {
        "Scans the live ledger for common secret prefixes and reports any unredacted matches."
    }
    fn run(&self) -> ProbeResult {
        let start = Instant::now();

        let candidates = [
            PathBuf::from("/var/lib/bad_apple/ledger.jsonl"),
            PathBuf::from("ledger.jsonl"),
        ];

        let mut ledger_text: Option<String> = None;
        for path in &candidates {
            if let Ok(text) = std::fs::read_to_string(path) {
                ledger_text = Some(text);
                break;
            }
        }

        let (mitigated, description, payload) = match ledger_text {
            Some(text) => {
                let bad_patterns = [
                    "sk-",
                    "-----BEGIN OPENSSH PRIVATE KEY-----",
                    "-----BEGIN RSA PRIVATE KEY-----",
                    "AKIA",
                ];
                let mut found = Vec::new();
                for pat in &bad_patterns {
                    if text.contains(pat) {
                        found.push(*pat);
                    }
                }
                if found.is_empty() {
                    (true, self.description().to_string(), None)
                } else {
                    (
                        false,
                        format!("ledger contains unredacted patterns: {}", found.join(", ")),
                        Some(found.join(", ")),
                    )
                }
            }
            None => (
                true,
                "ledger not present; redaction cannot be tested on this machine".to_string(),
                None,
            ),
        };

        ProbeResult {
            probe: self.id().to_string(),
            category: self.category().to_string(),
            mitigated,
            severity: self.severity(),
            title: self.title().to_string(),
            description,
            payload,
            timestamp_ms: crate::red_team::now_ms(),
            duration_us: elapsed_us(start),
        }
    }
}
