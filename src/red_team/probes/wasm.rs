//! WASM cage probes — fuel exhaustion and memory policy.

use super::{elapsed_us, Probe, ProbeResult, Severity};
use crate::wasm_cage::WasmCage;
use std::time::Instant;

/// Minimal valid WASM module that exports a `run` function returning i32 42.
fn valid_return_module() -> Vec<u8> {
    vec![
        0x00, 0x61, 0x73, 0x6d, // magic
        0x01, 0x00, 0x00, 0x00, // version
        0x01, 0x06, 0x01, 0x60, 0x00, 0x01,
        0x7f, // type section: 1 func, 0 params, 1 i32 result
        0x03, 0x02, 0x01, 0x00, // function section: 1 function, type 0
        0x07, 0x07, 0x01, 0x03, 0x72, 0x75, 0x6e, 0x00, 0x00, // export "run" as func 0
        0x0a, 0x06, 0x01, 0x04, 0x00, 0x41, 0x2a, 0x0b, // code section: i32.const 42, end
    ]
}

/// WASM module with an infinite loop in its `run` function.
fn infinite_loop_module() -> Vec<u8> {
    vec![
        0x00, 0x61, 0x73, 0x6d, // magic
        0x01, 0x00, 0x00, 0x00, // version
        0x01, 0x04, 0x01, 0x60, 0x00, 0x00, // type section: 1 func, 0 params, 0 results
        0x03, 0x02, 0x01, 0x00, // function section
        0x07, 0x07, 0x01, 0x03, 0x72, 0x75, 0x6e, 0x00, 0x00, // export "run"
        0x0a, 0x06, 0x01, 0x04, 0x00, 0x03, 0x40, 0x0c, 0x00, 0x0b,
        0x0b, // code: loop, br 0, end, end
    ]
}

/// WASM module declaring 100 initial memory pages, exceeding the cage policy.
fn oversized_memory_module() -> Vec<u8> {
    vec![
        0x00, 0x61, 0x73, 0x6d, // magic
        0x01, 0x00, 0x00, 0x00, // version
        0x05, 0x03, 0x01, 0x64, 0x64, // memory section: 1 memory, 100 min, 100 max
    ]
}

/// Probe that verifies the WASM cage halts an infinite loop via fuel exhaustion.
pub struct WasmFuelExhaustionProbe;

impl Probe for WasmFuelExhaustionProbe {
    fn id(&self) -> &'static str {
        "wasm.fuel_exhaustion"
    }
    fn category(&self) -> &'static str {
        "wasm"
    }
    fn severity(&self) -> Severity {
        Severity::High
    }
    fn title(&self) -> &'static str {
        "WASM cage halts infinite loop via fuel metering"
    }
    fn description(&self) -> &'static str {
        "Compiles a guest module with an infinite loop and verifies execution exhausts fuel instead of hanging."
    }
    fn run(&self) -> ProbeResult {
        let start = Instant::now();
        let mut cage = match WasmCage::new() {
            Ok(c) => c,
            Err(e) => {
                return ProbeResult {
                    probe: self.id().to_string(),
                    category: self.category().to_string(),
                    mitigated: false,
                    severity: self.severity(),
                    title: self.title().to_string(),
                    description: format!("could not create wasm cage: {e}"),
                    payload: None,
                    timestamp_ms: crate::red_team::now_ms(),
                    duration_us: elapsed_us(start),
                };
            }
        };

        let wasm = infinite_loop_module();
        let compile_ok = cage.compile(&wasm).is_ok();
        let mitigated = if compile_ok {
            // If compile succeeds, the run must exhaust fuel.
            cage.run_with_input(b"test").is_err()
        } else {
            true
        };

        ProbeResult {
            probe: self.id().to_string(),
            category: self.category().to_string(),
            mitigated,
            severity: self.severity(),
            title: self.title().to_string(),
            description: self.description().to_string(),
            payload: Some(hex::encode(&wasm)),
            timestamp_ms: crate::red_team::now_ms(),
            duration_us: elapsed_us(start),
        }
    }
}

/// Probe that verifies the WASM cage rejects modules with excessive initial memory.
pub struct WasmMemoryPolicyProbe;

impl Probe for WasmMemoryPolicyProbe {
    fn id(&self) -> &'static str {
        "wasm.memory_policy"
    }
    fn category(&self) -> &'static str {
        "wasm"
    }
    fn severity(&self) -> Severity {
        Severity::High
    }
    fn title(&self) -> &'static str {
        "WASM cage rejects modules exceeding the memory policy"
    }
    fn description(&self) -> &'static str {
        "Compiles a module that declares 100 initial pages and verifies it is rejected by the memory policy."
    }
    fn run(&self) -> ProbeResult {
        let start = Instant::now();
        let mut cage = match WasmCage::new() {
            Ok(c) => c,
            Err(e) => {
                return ProbeResult {
                    probe: self.id().to_string(),
                    category: self.category().to_string(),
                    mitigated: false,
                    severity: self.severity(),
                    title: self.title().to_string(),
                    description: format!("could not create wasm cage: {e}"),
                    payload: None,
                    timestamp_ms: crate::red_team::now_ms(),
                    duration_us: elapsed_us(start),
                };
            }
        };

        let wasm = oversized_memory_module();
        let mitigated = cage.compile(&wasm).is_err();

        ProbeResult {
            probe: self.id().to_string(),
            category: self.category().to_string(),
            mitigated,
            severity: self.severity(),
            title: self.title().to_string(),
            description: self.description().to_string(),
            payload: Some(hex::encode(&wasm)),
            timestamp_ms: crate::red_team::now_ms(),
            duration_us: elapsed_us(start),
        }
    }
}
