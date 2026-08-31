#![no_main]
use libfuzzer_sys::fuzz_target;

// Fuzz the WASM compiler and execution cage with arbitrary bytes.
//
// The cage must reject invalid modules without panicking and must safely
// execute valid ones under the fuel and memory limits.  This target:
//
// 1. Creates a fresh `WasmCage` (bounded memory, fuel metering).
// 2. Attempts to compile the fuzzer-provided bytes as a WASM module.
// 3. If compilation succeeds, runs the module with the same bytes as input.
//
// A panic, hang, or unbounded resource consumption in either `compile` or
// `run_with_input` constitutes a finding.
fuzz_target!(|data: &[u8]| {
    use bad_apple::wasm_cage::WasmCage;

    // Construct a fresh cage for each iteration.  Creating a new cage is
    // cheap (engine + store + fuel) and avoids state leakage between runs.
    let mut cage = match WasmCage::new() {
        Ok(c) => c,
        Err(_) => return,
    };

    // compile() validates the module, enforces the memory policy (single
    // memory, bounded initial/max pages), and instantiates it with the
    // Bad Apple host ABI.  Invalid input should yield a WasmError, never a
    // panic.
    if cage.compile(data).is_err() {
        return;
    }

    // If the module compiled and instantiated, exercise the run path.
    // run_with_input resets fuel and host state, calls the exported
    // `run()` function, and returns the accumulated output.
    if cage.is_loaded() {
        // Use the fuzzer data as guest input.  This exercises the host
        // ABI (input_size, input_read, alloc, output_write) under
        // arbitrary payloads.
        let _ = cage.run_with_input(data);

        // Also try with an empty input to test the no-input edge case.
        let _ = cage.run_with_input(b"");
    }
});
