//! Apple Intelligence / Foundation Models integration layer.
//!
//! Provides a thread-safe, global C-callback registration primitive and a
//! small runtime loader that can `dlopen` the optional `FireflySiriBridge`
//! Swift dylib.  The design keeps the main async runtime unblocked by
//! running all model calls inside `tokio::task::spawn_blocking`.
//!
//! Safety note: the C string exchange necessarily allocates the required
//! null-terminated strings, but the data itself is passed by raw pointer
//! through the C FFI rather than via HTTP/JSON, and the returned string is
//! freed on the Rust side as soon as it is converted.

use std::ffi::{c_char, c_void, CStr, CString};
use std::panic::catch_unwind;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;
use std::time::Instant;

/// C-compatible callback signature for an Apple Intelligence provider.
///
/// The caller passes a valid, null-terminated UTF-8 prompt.  The callee
/// returns ownership of a null-terminated C string (or `NULL` on failure).
/// Rust frees the returned pointer with `libc::free`, so the bridge must
/// allocate with the C library allocator (e.g. `strdup`).
pub type AppleIntelligenceCallback = extern "C" fn(*const c_char) -> *mut c_char;

static CALLBACK: OnceLock<AppleIntelligenceCallback> = OnceLock::new();

static LAST_LATENCY_US: AtomicU64 = AtomicU64::new(0);
static CALL_COUNT: AtomicU64 = AtomicU64::new(0);
static FAIL_COUNT: AtomicU64 = AtomicU64::new(0);

/// Register the Apple Intelligence callback.  This is the public FFI
/// primitive used by the embedded Swift bridge and by external daemons.
#[no_mangle]
pub extern "C" fn register_apple_intelligence_oracle(callback: AppleIntelligenceCallback) {
    let _ = CALLBACK.set(callback);
    tracing::info!("🍎 Apple Intelligence oracle callback registered");
}

/// True when a callback has been registered and is available for use.
pub fn is_available() -> bool {
    CALLBACK.get().is_some()
}

/// Synchronously invoke the registered Apple Intelligence callback.
///
/// Panics inside the foreign function are caught and treated as a failure,
/// preserving the calling async task.  The returned C string is freed with
/// `free` before the function returns.
pub fn call_sync(prompt: &str) -> Option<String> {
    let cb = *CALLBACK.get()?;
    let c_prompt = CString::new(prompt).ok()?;
    let start = Instant::now();

    let result = catch_unwind(std::panic::AssertUnwindSafe(|| cb(c_prompt.as_ptr()))).ok()?;

    let latency_us = start.elapsed().as_micros() as u64;
    LAST_LATENCY_US.store(latency_us, Ordering::Relaxed);
    CALL_COUNT.fetch_add(1, Ordering::Relaxed);

    if result.is_null() {
        FAIL_COUNT.fetch_add(1, Ordering::Relaxed);
        return None;
    }

    let output = unsafe {
        let out = CStr::from_ptr(result).to_str().ok()?.to_string();
        libc::free(result as *mut c_void);
        out
    };

    Some(output)
}

/// Asynchronously invoke the callback on a blocking thread so the async
/// runtime is never paused by the model inference.
pub async fn call(prompt: &str) -> Option<String> {
    let prompt = prompt.to_string();
    tokio::task::spawn_blocking(move || call_sync(&prompt))
        .await
        .ok()?
}

/// Attempt to load the optional `libFireflySiriBridge.dylib` from a set of
/// common locations and call its `init_firefly_siri_bridge` symbol.  This
/// is the runtime path for the in-process Swift bridge.
pub fn try_load_bridge() -> Result<(), String> {
    use libloading::{Library, Symbol};

    type InitFn = extern "C" fn();

    let paths = [
        "libFireflySiriBridge.dylib",
        "./libFireflySiriBridge.dylib",
        "target/release/libFireflySiriBridge.dylib",
        "target/debug/libFireflySiriBridge.dylib",
    ];

    for path in paths {
        let lib = match unsafe { Library::new(path) } {
            Ok(l) => l,
            Err(_) => continue,
        };

        let init: Symbol<InitFn> = unsafe { lib.get(b"init_firefly_siri_bridge\0") }
            .map_err(|e| format!("Swift bridge lacks init symbol: {}", e))?;

        init();

        // Intentionally leak the library handle so the callback remains
        // valid for the lifetime of the process.  This is a daemon-style
        // bridge; cleanup happens on process exit.
        std::mem::forget(lib);

        return Ok(());
    }

    Err("FireflySiriBridge dylib not found in any search path".to_string())
}

/// Try to load the bridge once on startup, logging success or a warning.
pub fn initialize() {
    match try_load_bridge() {
        Ok(()) => tracing::info!("🍎 FireflySiriBridge loaded and initialized"),
        Err(e) => tracing::warn!("🍎 FireflySiriBridge not available: {}", e),
    }
}

/// Return the latency (in microseconds) of the most recent call.
pub fn last_latency_us() -> u64 {
    LAST_LATENCY_US.load(Ordering::Relaxed)
}

/// Return the total number of successful Apple Intelligence calls.
pub fn call_count() -> u64 {
    CALL_COUNT.load(Ordering::Relaxed)
}

/// Return the total number of failed Apple Intelligence calls.
pub fn fail_count() -> u64 {
    FAIL_COUNT.load(Ordering::Relaxed)
}

/// Reset latency/call counters.  Useful for telemetry snapshots.
pub fn reset_counters() {
    LAST_LATENCY_US.store(0, Ordering::Relaxed);
    CALL_COUNT.store(0, Ordering::Relaxed);
    FAIL_COUNT.store(0, Ordering::Relaxed);
}
