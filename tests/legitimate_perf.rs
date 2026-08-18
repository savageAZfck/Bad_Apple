//! Legitimate low-level performance benchmarks.
//!
//! The state-save benchmark is implemented as a binary unit test in
//! `src/state_saver.rs` because its `SavePayload` depends on
//! `FullySapientSoulMatrix`, which lives in the binary crate.  This integration
//! crate benchmarks the public UMA and connectome paths.

#[path = "legitimate_perf/uma_bench.rs"]
mod uma_bench;
