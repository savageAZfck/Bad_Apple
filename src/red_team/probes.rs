//! Adversarial probes for the Bad Apple self-red teaming harness.
//!
//! Each probe attempts a known attack against an isolated security primitive.
//! Probes must be safe: they use temporary directories, synthetic secrets, and
//! in-memory data. No probe targets a live user daemon unless it is launched
//! explicitly by the harness with an isolated environment.

pub use super::{Probe, ProbeResult, Severity};

pub(crate) fn elapsed_us(start: std::time::Instant) -> u128 {
    start.elapsed().as_micros()
}

pub(crate) fn now_ms() -> u64 {
    crate::red_team::now_ms()
}

pub mod audit;
pub mod cage;
pub mod p2p;
pub mod policy;
pub mod slicks;
pub mod wasm;
