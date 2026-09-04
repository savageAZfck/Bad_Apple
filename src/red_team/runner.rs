//! Red-team runner and continuous monitoring loop.

use super::{Finding, Probe, Report};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

/// Runner that executes a registry of probes and produces a report.
pub struct RedTeamRunner {
    registry: Arc<Vec<Box<dyn Probe>>>,
}

impl Default for RedTeamRunner {
    fn default() -> Self {
        Self {
            registry: Arc::new(super::default_registry()),
        }
    }
}

impl RedTeamRunner {
    /// Create a runner with the default built-in probe registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a runner with a custom probe registry.
    pub fn with_registry(registry: Vec<Box<dyn Probe>>) -> Self {
        Self {
            registry: Arc::new(registry),
        }
    }

    /// Run every probe once and return a report.
    pub fn run_once(&self) -> Report {
        let start = Instant::now();
        let mut report = Report::new();

        for probe in self.registry.iter() {
            let result = probe.run();
            report.add(result);
        }

        report.duration_ms = start.elapsed().as_millis();
        report.calculate_score();
        report
    }

    /// Run only probes in a given category.
    pub fn run_category(&self, category: &str) -> Report {
        let start = Instant::now();
        let mut report = Report::new();

        for probe in self.registry.iter() {
            if probe.category() == category {
                let result = probe.run();
                report.add(result);
            }
        }

        report.duration_ms = start.elapsed().as_millis();
        report.calculate_score();
        report
    }

    /// Run a single probe by id and return its result.
    pub fn run_probe(&self, id: &str) -> Option<Report> {
        let start = Instant::now();
        let mut report = Report::new();

        for probe in self.registry.iter() {
            if probe.id() == id {
                let result = probe.run();
                report.add(result);
                report.duration_ms = start.elapsed().as_millis();
                report.calculate_score();
                return Some(report);
            }
        }

        None
    }

    /// Return all findings from the last run. (Convenience; use `run_once` for full data.)
    pub fn all_findings(&self) -> Vec<Finding> {
        self.run_once().findings()
    }
}

/// Shared state for the continuous red-team loop.
pub struct RedTeamLoop {
    runner: Arc<RedTeamRunner>,
    interval: Duration,
    findings: Arc<Mutex<Vec<Finding>>>,
    last_report: Arc<Mutex<Option<super::Report>>>,
    running: Arc<Mutex<bool>>,
}

impl RedTeamLoop {
    /// Create a new continuous loop with the default probe registry.
    pub fn new(interval: Duration) -> Self {
        Self {
            runner: Arc::new(RedTeamRunner::default()),
            interval,
            findings: Arc::new(Mutex::new(Vec::new())),
            last_report: Arc::new(Mutex::new(None)),
            running: Arc::new(Mutex::new(false)),
        }
    }

    /// Return a snapshot of accumulated findings.
    pub fn findings(&self) -> Vec<Finding> {
        self.findings
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// Return the last report, if any.
    pub fn last_report(&self) -> Option<super::Report> {
        self.last_report
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// Return whether the loop is running.
    pub fn is_running(&self) -> bool {
        *self.running.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Start the continuous loop on a background thread.
    ///
    /// The loop runs every `interval`, stores the last report, and appends any
    /// unmitigated findings to the shared findings list. It stops when
    /// `stop()` is called or the returned `RedTeamHandle` is dropped.
    pub fn start(&self) -> RedTeamHandle {
        *self.running.lock().unwrap_or_else(|e| e.into_inner()) = true;

        let findings = self.findings.clone();
        let last_report = self.last_report.clone();
        let running = self.running.clone();
        let handle_running = running.clone();
        let runner = self.runner.clone();
        let interval = self.interval;

        let handle = thread::spawn(move || {
            while *running.lock().unwrap_or_else(|e| e.into_inner()) {
                let report = runner.run_once();
                let new_findings = report.findings();

                if let Ok(mut f) = findings.lock() {
                    f.extend(new_findings);
                }

                if let Ok(mut r) = last_report.lock() {
                    *r = Some(report);
                }

                thread::sleep(interval);
            }
        });

        RedTeamHandle {
            running: handle_running,
            thread: Some(handle),
        }
    }

    /// Stop the continuous loop. This does not join the thread immediately.
    pub fn stop(&self) {
        *self.running.lock().unwrap_or_else(|e| e.into_inner()) = false;
    }
}

/// Handle to a running red-team loop. Dropping it stops the loop.
pub struct RedTeamHandle {
    running: Arc<Mutex<bool>>,
    thread: Option<thread::JoinHandle<()>>,
}

impl RedTeamHandle {
    /// Stop the loop and wait for the background thread to finish.
    pub fn stop(mut self) {
        *self.running.lock().unwrap_or_else(|e| e.into_inner()) = false;
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

impl Drop for RedTeamHandle {
    fn drop(&mut self) {
        *self.running.lock().unwrap_or_else(|e| e.into_inner()) = false;
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}
