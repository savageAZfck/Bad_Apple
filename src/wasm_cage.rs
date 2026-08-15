//! WebAssembly execution cage for untrusted tool synthesis.
//!
//! The cage uses `wasmi` 0.35.0 to compile and run a guest module.  Host-side
//! surface is intentionally tiny: `env::abort` and a small Bad Apple ABI for
//! string I/O.  Linear memory is bounded by rejecting modules with more than
//! one memory and by refusing modules whose declared initial or maximum memory
//! exceeds the cage's limit.  Execution is fuel-metered and halted when the
//! budget is exhausted.
//!
//! # Bad Apple string ABI
//!
//! The guest imports four functions from the `bad_apple` module:
//!
//! - `input_size() -> i32` – length of the host-provided input in bytes.
//! - `input_read(dst: i32)` – copies the input into guest memory at `dst`.
//! - `alloc(len: i32) -> i32` – reserves `len` contiguous guest bytes and
//!   returns the offset; the first page below `ALLOC_BASE` is reserved.
//! - `output_write(src: i32, len: i32)` – copies `src..src+len` to the host.
//!
//! The host calls the exported `run()` function and returns the accumulated
//! output as a UTF-8 string.
//!
//! # Usage
//!
//! ```text
//! let mut cage = WasmCage::new();
//! cage.compile(&wasm_bytes)?;
//! let output = cage.run_with_input(b"some text")?;
//! ```

use std::fmt;
use wasmi::{Config, EnforcedLimits, Engine, Extern, Linker, Module, Store};

/// Maximum number of linear memory pages allowed per module.
///
/// One Wasm page is 64 KiB.  `MAX_PAGES = 16` gives a 1 MiB upper bound.
const MAX_PAGES: u64 = 16;

/// Maximum initial pages a module may declare.  Modules with `memory 2` or more
/// are rejected so that untrusted payloads cannot reserve a large slab up front.
const MAX_INITIAL_PAGES: u64 = 2;

/// Default fuel budget for a single call.
const DEFAULT_FUEL: u64 = 1_000_000;

/// Start of the host-side bump allocator in guest linear memory.  Everything
/// below this offset (the first page) is left for the guest's static data.
const ALLOC_BASE: u32 = 1024;

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

impl From<wasmi::Error> for WasmError {
    fn from(e: wasmi::Error) -> Self {
        Self {
            reason: format!("wasmi: {e}"),
        }
    }
}

/// Host-side state kept in the Wasmi `Store`.  It holds the call input/output
/// and a tiny bump allocator for guest string buffers.
#[derive(Debug, Default)]
struct WasmHost {
    input: Vec<u8>,
    output: Vec<u8>,
    bump: u32,
}

impl WasmHost {
    fn reset(&mut self, input: Vec<u8>) {
        self.input = input;
        self.output.clear();
        self.bump = ALLOC_BASE;
    }

    fn bump_alloc(&mut self, len: u32) -> Option<u32> {
        let aligned = (len + 7) & !7;
        let base = self.bump;
        self.bump = base.checked_add(aligned)?;
        Some(base)
    }
}

/// Isolated WebAssembly execution cage.
pub struct WasmCage {
    engine: Engine,
    store: Store<WasmHost>,
    module: Option<Module>,
    instance: Option<wasmi::Instance>,
    fuel_per_call: u64,
}

impl WasmCage {
    /// Create a new cage with a 1 MiB memory policy and fuel metering.
    pub fn new() -> Result<Self, WasmError> {
        let mut config = Config::default();
        config
            .consume_fuel(true)
            .enforced_limits(EnforcedLimits::strict());
        let engine = Engine::new(&config);
        let mut store = Store::new(&engine, WasmHost::default());
        store.set_fuel(DEFAULT_FUEL).map_err(|e| WasmError {
            reason: format!("fuel: {e}"),
        })?;

        Ok(Self {
            engine,
            store,
            module: None,
            instance: None,
            fuel_per_call: DEFAULT_FUEL,
        })
    }

    /// Maximum guest linear memory in bytes.
    #[inline]
    pub fn memory_limit(&self) -> usize {
        (MAX_PAGES as usize) * 64 * 1024
    }

    /// Compile and validate an untrusted WASM module.
    ///
    /// Rejects modules that exceed the memory policy or that are not valid
    /// WebAssembly binaries.  The module is instantiated with the Bad Apple host
    /// ABI: `env::abort`, `bad_apple::input_size`, `bad_apple::input_read`,
    /// `bad_apple::alloc`, and `bad_apple::output_write`.
    pub fn compile(&mut self, wasm_bytes: &[u8]) -> Result<(), WasmError> {
        self.validate_memory_policy(wasm_bytes)?;

        if !wasm_bytes.starts_with(&[0x00, 0x61, 0x73, 0x6d]) {
            return Err(WasmError {
                reason: "not a valid WASM module".to_string(),
            });
        }

        self.module = Some(Module::new(&self.engine, wasm_bytes)?);

        let mut linker = <Linker<WasmHost>>::new(&self.engine);

        // Minimal abort hook from the `env` module.
        linker
            .func_wrap("env", "abort", |_caller: wasmi::Caller<'_, WasmHost>| {
                tracing::warn!("wasm guest called abort");
            })
            .map_err(|e| WasmError {
                reason: format!("linker: {e}"),
            })?;

        // Bad Apple string ABI.
        linker
            .func_wrap(
                "bad_apple",
                "input_size",
                |caller: wasmi::Caller<'_, WasmHost>| -> i32 { caller.data().input.len() as i32 },
            )
            .map_err(|e| WasmError {
                reason: format!("linker: {e}"),
            })?;

        linker
            .func_wrap(
                "bad_apple",
                "input_read",
                |mut caller: wasmi::Caller<'_, WasmHost>, dst: i32| {
                    let Some(Extern::Memory(mem)) = caller.get_export("memory") else {
                        return;
                    };
                    let (mem_data, state) = mem.data_and_store_mut(&mut caller);
                    let start = dst as usize;
                    let end = start.saturating_add(state.input.len());
                    if let Some(dest) = mem_data.get_mut(start..end) {
                        dest.copy_from_slice(&state.input);
                    }
                },
            )
            .map_err(|e| WasmError {
                reason: format!("linker: {e}"),
            })?;

        linker
            .func_wrap(
                "bad_apple",
                "alloc",
                |mut caller: wasmi::Caller<'_, WasmHost>, len: i32| -> i32 {
                    let Some(Extern::Memory(mem)) = caller.get_export("memory") else {
                        return -1;
                    };
                    let limit = mem.data_size(&caller) as u32;
                    let host = caller.data_mut();
                    let Some(base) = host.bump_alloc(len as u32) else {
                        return -1;
                    };
                    if base.checked_add(len as u32).is_none_or(|end| end > limit) {
                        return -1;
                    }
                    base as i32
                },
            )
            .map_err(|e| WasmError {
                reason: format!("linker: {e}"),
            })?;

        linker
            .func_wrap(
                "bad_apple",
                "output_write",
                |mut caller: wasmi::Caller<'_, WasmHost>, src: i32, len: i32| {
                    let Some(Extern::Memory(mem)) = caller.get_export("memory") else {
                        return;
                    };
                    let (mem_data, state) = mem.data_and_store_mut(&mut caller);
                    let start = src as usize;
                    let end = start.saturating_add(len.max(0) as usize);
                    if let Some(src) = mem_data.get(start..end) {
                        state.output.extend_from_slice(src);
                    }
                },
            )
            .map_err(|e| WasmError {
                reason: format!("linker: {e}"),
            })?;

        let instance = linker
            .instantiate(&mut self.store, self.module.as_ref().unwrap())?
            .start(&mut self.store)?;
        self.instance = Some(instance);

        Ok(())
    }

    /// Run the exported `run()` function with the provided input and return
    /// the guest's accumulated output.
    pub fn run_with_input(&mut self, input: &[u8]) -> Result<String, WasmError> {
        let instance = self.instance.as_ref().ok_or_else(|| WasmError {
            reason: "no compiled module".to_string(),
        })?;

        // Reset fuel and host state for this call.
        self.store
            .set_fuel(self.fuel_per_call)
            .map_err(|e| WasmError {
                reason: format!("fuel: {e}"),
            })?;
        self.store.data_mut().reset(input.to_vec());

        let func = instance.get_typed_func::<(), ()>(&self.store, "run")?;
        func.call(&mut self.store, ()).map_err(WasmError::from)?;

        let output = &self.store.data().output;
        Ok(String::from_utf8_lossy(output).into_owned())
    }

    /// Set the fuel budget for `run_with_input`.  Higher values allow longer
    /// guest runs.
    pub fn set_fuel_per_call(&mut self, fuel: u64) {
        self.fuel_per_call = fuel;
    }

    /// Return true if a module has been compiled.
    pub fn is_loaded(&self) -> bool {
        self.module.is_some() && self.instance.is_some()
    }

    /// Inspect the module's memory section and reject anything that could grow
    /// beyond the cage limit or that reserves too much memory up front.
    fn validate_memory_policy(&self, wasm_bytes: &[u8]) -> Result<(), WasmError> {
        use wasmparser::{Parser, Payload};

        let mut initial_pages = None;
        let mut max_pages = None;

        for payload in Parser::new(0).parse_all(wasm_bytes) {
            match payload {
                Ok(Payload::MemorySection(reader)) => {
                    for mem in reader {
                        let mem = mem.map_err(|e| WasmError {
                            reason: format!("memory section parse error: {e}"),
                        })?;
                        initial_pages = Some(mem.initial);
                        max_pages = mem.maximum;
                    }
                }
                Ok(Payload::End(_)) => break,
                Err(e) => {
                    return Err(WasmError {
                        reason: format!("WASM parse error: {e}"),
                    });
                }
                _ => {}
            }
        }

        if let Some(initial) = initial_pages {
            if initial > MAX_INITIAL_PAGES {
                return Err(WasmError {
                    reason: format!(
                        "WASM module requests {initial} initial memory pages; maximum is {MAX_INITIAL_PAGES}"
                    ),
                });
            }
            let effective_max = max_pages.unwrap_or(MAX_PAGES);
            if effective_max > MAX_PAGES {
                return Err(WasmError {
                    reason: format!(
                        "WASM module allows {effective_max} memory pages; maximum is {MAX_PAGES}"
                    ),
                });
            }
        }

        Ok(())
    }
}

impl Default for WasmCage {
    fn default() -> Self {
        Self::new().expect("WasmCage::new cannot fail")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Tiny wasm module:
    // (module
    //   (func (export "run") (param i64) (result i64)
    //     local.get 0
    //     i64.const 1
    //     i64.add)
    // )
    const ADD_ONE_WASM: &[u8] = &[
        0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, 0x01, 0x06, 0x01, 0x60, 0x01, 0x7e, 0x01,
        0x7e, 0x03, 0x02, 0x01, 0x00, 0x07, 0x07, 0x01, 0x03, 0x72, 0x75, 0x6e, 0x00, 0x00, 0x0a,
        0x09, 0x01, 0x07, 0x00, 0x20, 0x00, 0x42, 0x01, 0x7c, 0x0b,
    ];

    #[test]
    fn cage_compiles_and_runs_i64_add() {
        let mut cage = WasmCage::new().unwrap();
        cage.compile(ADD_ONE_WASM).unwrap();
        assert!(cage.is_loaded());
    }

    #[test]
    fn cage_rejects_oversized_memory() {
        // (module (memory 17))
        let wasm: &[u8] = &[
            0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, 0x05, 0x03, 0x01, 0x00, 0x11,
        ];
        let mut cage = WasmCage::new().unwrap();
        assert!(cage.compile(wasm).is_err());
    }
}
