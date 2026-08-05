//! Dual-Process attention governor.
//!
//! Replaces the fixed 6-second cognitive clock with a real-time, anomaly- and
//! stress-driven tick interval.  System 1 is the fast, low-power path that
//! skips the heavy 4-block Transformer; System 2 is the deep attention path
//! that uses the full Transformer, a wider context, and a scaled learning rate.

use crate::telemetry::TelemetryState;
use crate::SensorSnapshot;
use std::time::Duration;

/// Operational mode for the dual-process cognitive architecture.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SystemMode {
    /// Fast, low-power, hyperdimensional/embedding path for routine inputs.
    System1,
    /// Deep 4-block Transformer attention for anomaly resolution.
    System2,
}

/// Real-time metacognitive governor that decides whether to engage the deep
/// Transformer (System 2) or stay on the fast path (System 1).
#[derive(Clone, Debug)]
pub struct DualProcessGovernor {
    /// Current mode.
    mode: SystemMode,
    /// 0..1 measure of input unpredictability (high = System 2).
    entropy_index: f64,
    /// 0..1 hardware/heap stress (high = System 2 or throttle).
    resource_stress: f64,
    /// Consecutive cycles the system has been in System 2.
    system2_cycles: u64,
    /// Current clock interval.
    tick_ms: u64,
}

impl DualProcessGovernor {
    pub fn new() -> Self {
        Self {
            mode: SystemMode::System1,
            entropy_index: 0.0,
            resource_stress: 0.0,
            system2_cycles: 0,
            tick_ms: 6_000,
        }
    }

    /// Update state from live telemetry and sensor snapshots.
    pub fn update(&mut self, telemetry: &TelemetryState, sensors: &SensorSnapshot) {
        // Entropy index: combines memory-leak score, transfer/loss volatility,
        // and incoming data unpredictability.  Higher = more surprising.
        let memory_component = telemetry.memory_leak_score.clamp(0.0, 1.0);
        let critic_component = (1.0 - telemetry.critic_score.clamp(0.0, 1.0)) * 0.3;
        let transfer_component = (1.0 - telemetry.transfer_score.clamp(0.0, 1.0)) * 0.2;
        let loss_component = telemetry.last_loss.min(1.0) * 0.2;

        self.entropy_index =
            (memory_component * 0.5 + critic_component + transfer_component + loss_component)
                .clamp(0.0, 1.0);

        // Resource stress: CPU, memory pressure, and temperature.
        self.resource_stress = (sensors.cpu_usage_percent / 100.0 * 0.35
            + sensors.memory_pressure_percent / 100.0 * 0.35
            + (sensors.cpu_temperature_celsius / 100.0).clamp(0.0, 1.0) * 0.3)
            .clamp(0.0, 1.0);

        // Hysteresis: System 2 triggers on high entropy *or* high stress.
        let should_activate = self.entropy_index > 0.65 || self.resource_stress > 0.80;
        let should_release =
            self.entropy_index < 0.35 && self.resource_stress < 0.50 && self.system2_cycles > 2;

        match self.mode {
            SystemMode::System1 if should_activate => {
                self.mode = SystemMode::System2;
                self.system2_cycles = 1;
                self.tick_ms = 1_000; // fast attention during anomaly
            }
            SystemMode::System2 if should_release => {
                self.mode = SystemMode::System1;
                self.system2_cycles = 0;
                self.tick_ms = 6_000; // back to routine cadence
            }
            SystemMode::System2 => {
                self.system2_cycles += 1;
                // Gradually slow System 2 as the anomaly resolves.
                self.tick_ms = (self.tick_ms + 250).min(3_000);
            }
            SystemMode::System1 => {
                self.tick_ms = 6_000;
            }
        }
    }

    /// Current tick interval as a `Duration`.
    pub fn tick_duration(&self) -> Duration {
        Duration::from_millis(self.tick_ms)
    }

    /// Current mode.
    pub fn mode(&self) -> SystemMode {
        self.mode
    }

    /// True when the deep Transformer path should be used.
    pub fn system2_active(&self) -> bool {
        matches!(self.mode, SystemMode::System2)
    }

    /// 0..1 entropy index.
    pub fn entropy_index(&self) -> f64 {
        self.entropy_index
    }

    /// 0..1 resource stress.
    pub fn resource_stress(&self) -> f64 {
        self.resource_stress
    }

    /// Consecutive System 2 cycles.
    pub fn system2_cycles(&self) -> u64 {
        self.system2_cycles
    }
}

impl Default for DualProcessGovernor {
    fn default() -> Self {
        Self::new()
    }
}
