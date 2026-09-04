//! Bad Apple adversarial self-red teaming harness.
//!
//! The harness runs a library of adversarial probes against the Bad Apple
//! security primitives, reports any successful escape as a `Finding`, and can
//! run either on-demand or in a continuous background loop.
//!
//! Each probe is designed to be safe: it operates on temporary directories,
//! isolated library calls, or in-memory data. No probe attacks a live user
//! daemon or user data unless explicitly configured to do so.

pub mod probes;
pub mod runner;

pub use runner::{RedTeamLoop, RedTeamRunner};

/// Severity of a red-team finding.
#[derive(Clone, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Severity {
    Critical,
    High,
    Medium,
    Low,
    Info,
}

/// Result of a single probe execution.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ProbeResult {
    pub probe: String,
    pub category: String,
    pub mitigated: bool,
    pub severity: Severity,
    pub title: String,
    pub description: String,
    pub payload: Option<String>,
    pub timestamp_ms: u64,
    pub duration_us: u128,
}

impl ProbeResult {
    /// A finding is a probe result where the attack was *not* stopped.
    pub fn is_finding(&self) -> bool {
        !self.mitigated
    }

    /// Convert a mitigated result into an `Attempt` summary, or a finding if not mitigated.
    pub fn finding(&self) -> Option<Finding> {
        if self.mitigated {
            None
        } else {
            Some(Finding {
                probe: self.probe.clone(),
                category: self.category.clone(),
                severity: self.severity.clone(),
                title: self.title.clone(),
                description: self.description.clone(),
                payload: self.payload.clone(),
                timestamp_ms: self.timestamp_ms,
                duration_us: self.duration_us,
            })
        }
    }
}

/// A confirmed finding from the red-team harness.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Finding {
    pub probe: String,
    pub category: String,
    pub severity: Severity,
    pub title: String,
    pub description: String,
    pub payload: Option<String>,
    pub timestamp_ms: u64,
    pub duration_us: u128,
}

/// Aggregated report from one or more probe runs.
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct Report {
    pub attempts: Vec<ProbeResult>,
    pub total: u64,
    pub mitigated: u64,
    pub unmitigated: u64,
    pub duration_ms: u128,
    pub score: f64,
}

impl Report {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add(&mut self, result: ProbeResult) {
        self.total += 1;
        if result.mitigated {
            self.mitigated += 1;
        } else {
            self.unmitigated += 1;
        }
        self.attempts.push(result);
    }

    pub fn findings(&self) -> Vec<Finding> {
        self.attempts.iter().filter_map(|r| r.finding()).collect()
    }

    pub fn calculate_score(&mut self) {
        if self.total == 0 {
            self.score = 1.0;
            return;
        }
        self.score = self.mitigated as f64 / self.total as f64;
    }
}

/// Unix timestamp in milliseconds.
pub(crate) fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

/// A single probe implementation.
pub trait Probe: Send + Sync {
    fn id(&self) -> &'static str;
    fn category(&self) -> &'static str;
    fn severity(&self) -> Severity;
    fn title(&self) -> &'static str;
    fn description(&self) -> &'static str;
    fn run(&self) -> ProbeResult;
}

/// Default registry of all built-in probes.
pub fn default_registry() -> Vec<Box<dyn Probe>> {
    vec![
        Box::new(probes::cage::PathTraversalProbe),
        Box::new(probes::cage::SymlinkEscapeProbe),
        Box::new(probes::cage::CreateDirectoryTraversalProbe),
        Box::new(probes::slicks::ReplayCacheProbe),
        Box::new(probes::slicks::ClientProofTamperingProbe),
        Box::new(probes::p2p::P2PTamperingProbe),
        Box::new(probes::p2p::P2PWrongKeyProbe),
        Box::new(probes::wasm::WasmFuelExhaustionProbe),
        Box::new(probes::wasm::WasmMemoryPolicyProbe),
        Box::new(probes::policy::PolicyCoversShellProbe),
        Box::new(probes::policy::PolicyCoversAppleScriptProbe),
        Box::new(probes::audit::LedgerSecretRedactionProbe),
    ]
}
