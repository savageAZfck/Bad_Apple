//! Policy coverage probes — verify policy.yaml covers destructive tools.

use super::{elapsed_us, Probe, ProbeResult, Severity};
use std::path::PathBuf;
use std::time::Instant;

fn load_policy() -> Option<String> {
    let candidates = [
        PathBuf::from("policy.yaml"),
        PathBuf::from("src/policy.yaml"),
        PathBuf::from("/var/lib/bad_apple/policy.yaml"),
    ];
    for path in &candidates {
        if let Ok(text) = std::fs::read_to_string(path) {
            return Some(text);
        }
    }
    None
}

fn tool_approval(tool: &str) -> bool {
    if let Some(policy) = load_policy() {
        // The policy should explicitly mention the tool and require approval.
        policy.contains(tool)
            && (policy.contains("require_approval: true")
                || policy.contains("allowed: false")
                || policy.contains("denied_patterns"))
    } else {
        false
    }
}

/// Probe that verifies the policy covers shell execution with approval gating.
pub struct PolicyCoversShellProbe;

impl Probe for PolicyCoversShellProbe {
    fn id(&self) -> &'static str {
        "policy.covers_shell"
    }
    fn category(&self) -> &'static str {
        "policy"
    }
    fn severity(&self) -> Severity {
        Severity::Medium
    }
    fn title(&self) -> &'static str {
        "Policy covers run_shell with approval or denials"
    }
    fn description(&self) -> &'static str {
        "Verifies that policy.yaml explicitly covers the run_shell tool with approval gates or denied patterns."
    }
    fn run(&self) -> ProbeResult {
        let start = Instant::now();
        let mitigated = tool_approval("run_shell");

        ProbeResult {
            probe: self.id().to_string(),
            category: self.category().to_string(),
            mitigated,
            severity: self.severity(),
            title: self.title().to_string(),
            description: self.description().to_string(),
            payload: None,
            timestamp_ms: crate::red_team::now_ms(),
            duration_us: elapsed_us(start),
        }
    }
}

/// Probe that verifies the policy covers AppleScript execution.
pub struct PolicyCoversAppleScriptProbe;

impl Probe for PolicyCoversAppleScriptProbe {
    fn id(&self) -> &'static str {
        "policy.covers_applescript"
    }
    fn category(&self) -> &'static str {
        "policy"
    }
    fn severity(&self) -> Severity {
        Severity::Medium
    }
    fn title(&self) -> &'static str {
        "Policy covers run_applescript with approval or denials"
    }
    fn description(&self) -> &'static str {
        "Verifies that policy.yaml explicitly covers the run_applescript tool with approval gates or denied patterns."
    }
    fn run(&self) -> ProbeResult {
        let start = Instant::now();
        let mitigated = tool_approval("run_applescript");

        ProbeResult {
            probe: self.id().to_string(),
            category: self.category().to_string(),
            mitigated,
            severity: self.severity(),
            title: self.title().to_string(),
            description: self.description().to_string(),
            payload: None,
            timestamp_ms: crate::red_team::now_ms(),
            duration_us: elapsed_us(start),
        }
    }
}
