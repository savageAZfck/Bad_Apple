//! Apple Intelligence / Foundation Models integration layer.
//!
//! Provides a thread-safe, global C-callback registration primitive and a
//! small runtime loader that can `dlopen` the optional `BadAppleBridge`
//! Swift dylib.  The design keeps the main async runtime unblocked by
//! running all model calls inside `tokio::task::spawn_blocking`.
//!
//! Safety note: the C string exchange necessarily allocates the required
//! null-terminated strings, but the data itself is passed by raw pointer
//! through the C FFI rather than via HTTP/JSON, and the returned string is
//! freed on the Rust side as soon as it is converted.

use crossbeam_channel::{bounded, unbounded, Sender};
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

/// C-compatible deallocator for strings the bridge allocates and returns.
pub type FreeStringCallback = extern "C" fn(*mut c_char);

/// C-compatible desktop notification dispatcher.
pub type DesktopNotificationCallback = extern "C" fn(*const c_char, *const c_char);

static CALLBACK: OnceLock<AppleIntelligenceCallback> = OnceLock::new();
static FREE_CB: OnceLock<FreeStringCallback> = OnceLock::new();
static NOTIFY_CB: OnceLock<DesktopNotificationCallback> = OnceLock::new();

static LAST_LATENCY_US: AtomicU64 = AtomicU64::new(0);
static CALL_COUNT: AtomicU64 = AtomicU64::new(0);
static FAIL_COUNT: AtomicU64 = AtomicU64::new(0);

/// Request sent to the dedicated Apple Intelligence actor thread.
///
/// The actor is the only thread that ever calls the registered C callback,
/// serializing all native framework invocations.  Callers receive the result
/// through the supplied one-shot reply channel.
type ActorRequest = (String, Sender<Option<String>>);

/// MPMC send handle for the actor thread.  Lazily initialized on the first
/// Apple Intelligence request.
static FFI_TX: OnceLock<Sender<ActorRequest>> = OnceLock::new();

/// Spawn the single background actor that owns the Apple Intelligence callback.
fn spawn_actor() -> Sender<ActorRequest> {
    let (tx, rx) = unbounded::<ActorRequest>();
    std::thread::spawn(move || {
        while let Ok((prompt, reply)) = rx.recv() {
            let result = invoke_callback(&prompt);
            let _ = reply.send(result);
        }
    });
    tx
}

/// Register the Apple Intelligence callback.  This is the public FFI
/// primitive used by the embedded Swift bridge and by external daemons.
#[no_mangle]
pub extern "C" fn register_apple_intelligence_oracle(callback: AppleIntelligenceCallback) {
    let _ = CALLBACK.set(callback);
    tracing::info!("🏴‍☠️  BAD APPLE // Apple Intelligence oracle callback registered");
}

/// True when a callback has been registered and is available for use.
pub fn is_available() -> bool {
    CALLBACK.get().is_some()
}

/// Internal, unsynchronized C callback invocation.
///
/// This is intentionally private: the single actor thread is the only caller,
/// so NLEmbedding / SystemLanguageModel calls never race.  Panics inside the
/// foreign function are caught and treated as a failure.
fn invoke_callback(prompt: &str) -> Option<String> {
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

    let guard = CStringOwner(result);

    let output = unsafe {
        // to_string_lossy() guarantees we always consume and free the C string,
        // even if the bytes are not strict UTF-8.
        CStr::from_ptr(guard.0).to_string_lossy().into_owned()
    };

    Some(output)
}

/// Synchronously invoke the Apple Intelligence callback through the actor.
///
/// This keeps all native framework calls on a single, dedicated thread so the
/// main runtime never contends the NLEmbedding lock.  The caller blocks only
/// on the reply channel, not on the Apple framework itself.
pub fn call_sync(prompt: &str) -> Option<String> {
    let actor = FFI_TX.get_or_init(spawn_actor);
    let (reply_tx, reply_rx) = bounded(1);
    actor.send((prompt.to_string(), reply_tx)).ok()?;
    reply_rx.recv().ok()?
}

/// RAII guard that owns a foreign-allocated C string and frees it on drop.
struct CStringOwner(*mut c_char);

impl Drop for CStringOwner {
    fn drop(&mut self) {
        if self.0.is_null() {
            return;
        }
        if let Some(free) = FREE_CB.get() {
            free(self.0);
        } else {
            unsafe { libc::free(self.0 as *mut c_void) };
        }
    }
}

/// Asynchronously invoke the callback through the dedicated actor thread.
///
/// The caller pushes the request into the lock-free MPMC queue and awaits the
/// one-shot reply, keeping the async runtime unblocked and the native
/// framework calls serialized on a single background thread.
pub async fn call(prompt: &str) -> Option<String> {
    let actor = FFI_TX.get_or_init(spawn_actor);
    let (reply_tx, reply_rx) = bounded(1);
    actor.send((prompt.to_string(), reply_tx)).ok()?;
    tokio::task::spawn_blocking(move || reply_rx.recv().ok()?)
        .await
        .ok()?
}

/// Attempt to load the optional `libBadAppleBridge.dylib` from a set of
/// common locations and call its `init_bad_apple_bridge` symbol.  This
/// is the runtime path for the in-process Swift bridge.
pub fn try_load_bridge() -> Result<(), String> {
    use libloading::{Library, Symbol};

    type InitFn = extern "C" fn();
    type FreeFn = extern "C" fn(*mut c_char);
    type NotifyFn = extern "C" fn(*const c_char, *const c_char);

    let mut paths: Vec<String> = vec![
        "libBadAppleBridge.dylib".to_string(),
        "./libBadAppleBridge.dylib".to_string(),
    ];

    // Prefer the bridge sitting next to the running executable, then the
    // build target directory matching the executable, then the default
    // `target/{release,debug}` fallback for manual runs.
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            if let Some(d) = dir.to_str() {
                paths.push(format!("{}/libBadAppleBridge.dylib", d));
            }
            if let Some(d) = dir.parent().and_then(|p| p.to_str()) {
                paths.push(format!("{}/libBadAppleBridge.dylib", d));
            }
        }
    }

    if let Ok(target_dir) = std::env::var("CARGO_TARGET_DIR") {
        paths.push(format!("{}/release/libBadAppleBridge.dylib", target_dir));
        paths.push(format!("{}/debug/libBadAppleBridge.dylib", target_dir));
    }

    paths.push("target/release/libBadAppleBridge.dylib".to_string());
    paths.push("target/debug/libBadAppleBridge.dylib".to_string());

    for path in &paths {
        let lib = match unsafe { Library::new(path) } {
            Ok(l) => l,
            Err(_) => continue,
        };

        let init: Symbol<InitFn> = unsafe { lib.get(b"init_bad_apple_bridge\0") }
            .map_err(|e| format!("Swift bridge lacks init symbol: {}", e))?;

        init();

        // Load the bridge's matching string deallocator if it exposes one.
        // This lets us free returned C strings in the same runtime that
        // allocated them, avoiding cross-runtime allocator drift.
        if let Ok(free) = unsafe { lib.get::<FreeFn>(b"free_swift_string\0") } {
            let _ = FREE_CB.set(*free);
            tracing::info!("🏴‍☠️  BAD APPLE // Swift string deallocator registered");
        }

        if let Ok(dispatch) = unsafe { lib.get::<NotifyFn>(b"dispatch_desktop_notification\0") } {
            let _ = NOTIFY_CB.set(*dispatch);
            tracing::info!("🏴‍☠️  BAD APPLE // Desktop notification dispatcher registered");
        }

        // Intentionally leak the library handle so the callback remains
        // valid for the lifetime of the process.  This is a daemon-style
        // bridge; cleanup happens on process exit.
        std::mem::forget(lib);

        return Ok(());
    }

    Err("BadAppleBridge dylib not found in any search path".to_string())
}

/// Try to load the bridge once on startup, logging success or a warning.
pub fn initialize() {
    match try_load_bridge() {
        Ok(()) => tracing::info!("🏴‍☠️  BAD APPLE // Bad Apple bridge loaded and initialized"),
        Err(e) => tracing::warn!("🏴‍☠️  BAD APPLE // Bad Apple bridge not available: {}", e),
    }
}

/// Dispatch a native macOS desktop notification if the bridge is loaded.
///
/// Title and body are passed as null-terminated C strings to the Swift
/// `dispatch_desktop_notification` hook.  If the bridge is unavailable or
/// notification authorization was denied, this logs and returns silently.
pub fn dispatch_desktop_notification(title: &str, body: &str) {
    let cb = match NOTIFY_CB.get() {
        Some(cb) => *cb,
        None => return,
    };
    let title_c = match CString::new(title) {
        Ok(s) => s,
        Err(_) => return,
    };
    let body_c = match CString::new(body) {
        Ok(s) => s,
        Err(_) => return,
    };
    cb(title_c.as_ptr(), body_c.as_ptr());
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
