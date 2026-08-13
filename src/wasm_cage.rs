//! WebAssembly execution cage for untrusted tool synthesis.
//!
//! This is a design scaffold for the Phase 4 hard isolation mandate.  The
//! synthesizer currently emits Python and raw shell commands.  The long-term
//! target is to compile or interpret every generated tool inside a tightly
//! bounded WebAssembly linear memory sandbox, preventing any generated code
//! from touching the host filesystem, environment, or core memory.
//!
//! # Design constraints
//!
//! * **Pure Rust runtime** — `wasmi` is the preferred engine: it is entirely
//!   written in Rust, has no C dependencies, and adds only a few megabytes to
//!   the release binary.  `wasmtime` would give more speed but pulls in a
//!   larger native object surface; `wasmer` is heavier still.
//! * **Bounded linear memory** — instantiate the module with a single
//!   pre-allocated 64 KiB or 1 MiB memory page.  No `memory.grow` is permitted.
//! * **No host imports** — the guest module receives only a tiny `env` with
//!   `abort` and `trace` functions.  There is no `fd_write`, `getenv`, or
//!   `sock_open`.
//! * **Tool-chain migration** — the `wild_workspace` synthesizer must be
//!   extended to either emit Rust/WAT or to transpile the existing Python
//!   plan into a tiny deterministic subset (`no_std`) that compiles to `wasm32`.
//!   A Python interpreter inside WASM is explicitly out of scope; it would be
//!   larger and slower than the entire EdgeOS binary.
//!
//! # Usage sketch (not yet wired)
//!
//! ```text
//! let mut cage = WasmCage::new().expect("init wasm engine");
//! let gas = cage.compile(&wasm_bytes)?;
//! let result: i64 = cage.call("run", &[1, 2, 3])?;
//! ```

use std::fmt;

/// Opaque handle to an isolated WebAssembly execution cage.
///
/// In this scaffold the engine is not yet linked; the struct holds the
/// configuration and a placeholder state so the API can stabilize.  The
/// `#[cfg(feature = "wasm-sandbox")]` block below shows the intended
/// integration shape once `wasmi` is added as an optional dependency.
pub struct WasmCage {
    memory_limit_bytes: usize,
    fuel: Option<u64>,
}

/// Error type for sandbox failures.
#[derive(Debug)]
pub struct WasmError {
    pub reason: String,
}

impl fmt::Display for WasmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "WasmCage error: {}", self.reason)
    }
}

impl std::error::Error for WasmError {}

impl WasmCage {
    /// Create a new cage with a default 1 MiB memory ceiling and a fuel limit.
    pub fn new() -> Result<Self, WasmError> {
        Ok(Self {
            memory_limit_bytes: 1024 * 1024,
            fuel: Some(1_000_000),
        })
    }

    /// Compile and validate an untrusted WASM module.
    ///
    /// The scaffold currently only checks the size and a WASM magic prefix;
    /// the real implementation will fully validate and instantiate with
    /// bounded memory and no host imports.
    pub fn compile(&mut self, wasm_bytes: &[u8]) -> Result<(), WasmError> {
        if wasm_bytes.len() > self.memory_limit_bytes {
            return Err(WasmError {
                reason: format!(
                    "WASM module {} bytes exceeds {} byte limit",
                    wasm_bytes.len(),
                    self.memory_limit_bytes
                ),
            });
        }
        if !wasm_bytes.starts_with(&[0x00, 0x61, 0x73, 0x6d]) {
            return Err(WasmError {
                reason: "not a valid WASM module".to_string(),
            });
        }
        Ok(())
    }

    /// Call a guest export with a small fixed-size argument list.
    ///
    /// The real implementation will meter instructions via `wasmi` fuel and
    /// trap on out-of-bounds memory or host import violations.
    pub fn call(&mut self, _name: &str, _args: &[i64]) -> Result<i64, WasmError> {
        Err(WasmError {
            reason: "WASM execution is not yet wired; cage is a design scaffold".to_string(),
        })
    }

    /// Maximum guest linear memory in bytes.
    pub fn memory_limit(&self) -> usize {
        self.memory_limit_bytes
    }
}

impl Default for WasmCage {
    fn default() -> Self {
        Self::new().expect("WasmCage::new cannot fail in scaffold")
    }
}
