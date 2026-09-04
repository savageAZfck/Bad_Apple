//! Automation cage probes — path traversal, symlink escape, directory traversal.

use super::{elapsed_us, Probe, ProbeResult, Severity};
use crate::automation_cage::{Action, AutomationCage};
use std::path::PathBuf;
use std::time::Instant;

/// Probe that attempts to escape the allowlisted root using `..` in a file path.
pub struct PathTraversalProbe;

impl Probe for PathTraversalProbe {
    fn id(&self) -> &'static str {
        "cage.path_traversal"
    }
    fn category(&self) -> &'static str {
        "cage"
    }
    fn severity(&self) -> Severity {
        Severity::Critical
    }
    fn title(&self) -> &'static str {
        "Automation cage rejects path traversal via CreateFile"
    }
    fn description(&self) -> &'static str {
        "Attempts to create a file outside the allowlisted root using ../ sequences."
    }
    fn run(&self) -> ProbeResult {
        let start = Instant::now();
        let root = std::env::current_dir()
            .unwrap_or_else(|_| std::env::temp_dir())
            .join("target")
            .join(format!("redteam_cage_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).ok();

        let cage = match AutomationCage::new(vec![root.clone()]) {
            Ok(c) => c,
            Err(e) => {
                return ProbeResult {
                    probe: self.id().to_string(),
                    category: self.category().to_string(),
                    mitigated: false,
                    severity: self.severity(),
                    title: self.title().to_string(),
                    description: format!("could not initialize cage: {e}"),
                    payload: None,
                    timestamp_ms: crate::red_team::now_ms(),
                    duration_us: elapsed_us(start),
                };
            }
        };

        let payload = "../etc/passwd";
        let action = Action::CreateFile {
            path: PathBuf::from(payload),
        };
        let mitigated = cage.validate(&action).is_err();

        let _ = std::fs::remove_dir_all(&root);

        ProbeResult {
            probe: self.id().to_string(),
            category: self.category().to_string(),
            mitigated,
            severity: self.severity(),
            title: self.title().to_string(),
            description: self.description().to_string(),
            payload: Some(payload.to_string()),
            timestamp_ms: crate::red_team::now_ms(),
            duration_us: elapsed_us(start),
        }
    }
}

/// Probe that attempts to create a directory outside the allowlisted root.
pub struct CreateDirectoryTraversalProbe;

impl Probe for CreateDirectoryTraversalProbe {
    fn id(&self) -> &'static str {
        "cage.create_directory_traversal"
    }
    fn category(&self) -> &'static str {
        "cage"
    }
    fn severity(&self) -> Severity {
        Severity::Critical
    }
    fn title(&self) -> &'static str {
        "Automation cage rejects directory creation traversal"
    }
    fn description(&self) -> &'static str {
        "Attempts to create a directory outside the allowlisted root using ../ sequences."
    }
    fn run(&self) -> ProbeResult {
        let start = Instant::now();
        let root = std::env::current_dir()
            .unwrap_or_else(|_| std::env::temp_dir())
            .join("target")
            .join(format!("redteam_cage_dir_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).ok();

        let cage = match AutomationCage::new(vec![root.clone()]) {
            Ok(c) => c,
            Err(e) => {
                return ProbeResult {
                    probe: self.id().to_string(),
                    category: self.category().to_string(),
                    mitigated: false,
                    severity: self.severity(),
                    title: self.title().to_string(),
                    description: format!("could not initialize cage: {e}"),
                    payload: None,
                    timestamp_ms: crate::red_team::now_ms(),
                    duration_us: elapsed_us(start),
                };
            }
        };

        let payload = "../../tmp/redteam_escape";
        let action = Action::CreateDirectory {
            path: PathBuf::from(payload),
        };
        let mitigated = cage.validate(&action).is_err();

        let _ = std::fs::remove_dir_all(&root);

        ProbeResult {
            probe: self.id().to_string(),
            category: self.category().to_string(),
            mitigated,
            severity: self.severity(),
            title: self.title().to_string(),
            description: self.description().to_string(),
            payload: Some(payload.to_string()),
            timestamp_ms: crate::red_team::now_ms(),
            duration_us: elapsed_us(start),
        }
    }
}

/// Probe that attempts to escape the allowlisted root using a symlink.
pub struct SymlinkEscapeProbe;

impl Probe for SymlinkEscapeProbe {
    fn id(&self) -> &'static str {
        "cage.symlink_escape"
    }
    fn category(&self) -> &'static str {
        "cage"
    }
    fn severity(&self) -> Severity {
        Severity::Critical
    }
    fn title(&self) -> &'static str {
        "Automation cage rejects symlink escape"
    }
    fn description(&self) -> &'static str {
        "Creates a symlink inside the allowlisted root that points outside, then attempts to write through it."
    }
    fn run(&self) -> ProbeResult {
        let start = Instant::now();
        let root = std::env::current_dir()
            .unwrap_or_else(|_| std::env::temp_dir())
            .join("target")
            .join(format!("redteam_symlink_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).ok();

        let allowed = root.join("allowed");
        let outside = std::env::current_dir()
            .unwrap_or_else(|_| std::env::temp_dir())
            .join("target")
            .join(format!("redteam_symlink_outside_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&outside);
        std::fs::create_dir_all(&outside).ok();

        let target = outside.join("target.txt");
        std::fs::write(&target, "outside").ok();
        let link = allowed.join("escape.txt");

        #[cfg(unix)]
        {
            std::fs::create_dir_all(&allowed).ok();
            std::os::unix::fs::symlink(&target, &link).ok();
        }

        let cage = match AutomationCage::new(vec![allowed.clone()]) {
            Ok(c) => c,
            Err(e) => {
                let _ = std::fs::remove_dir_all(&root);
                let _ = std::fs::remove_dir_all(&outside);
                return ProbeResult {
                    probe: self.id().to_string(),
                    category: self.category().to_string(),
                    mitigated: false,
                    severity: self.severity(),
                    title: self.title().to_string(),
                    description: format!("could not initialize cage: {e}"),
                    payload: None,
                    timestamp_ms: crate::red_team::now_ms(),
                    duration_us: elapsed_us(start),
                };
            }
        };

        let action = Action::CreateFile { path: link };
        let mitigated = cage.validate(&action).is_err();

        let _ = std::fs::remove_dir_all(&root);
        let _ = std::fs::remove_dir_all(&outside);

        ProbeResult {
            probe: self.id().to_string(),
            category: self.category().to_string(),
            mitigated,
            severity: self.severity(),
            title: self.title().to_string(),
            description: self.description().to_string(),
            payload: Some(allowed.to_string_lossy().to_string()),
            timestamp_ms: crate::red_team::now_ms(),
            duration_us: elapsed_us(start),
        }
    }
}
