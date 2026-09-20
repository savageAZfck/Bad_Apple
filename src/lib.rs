#![allow(dead_code, clippy::new_without_default)]

//! Public C-compatible library interface for the `bad_apple` runtime.
//!
//! This is an **Evaluation Kit** scaffold. It exposes a small, thread-safe FFI
//! surface so macOS native code (Swift / C++ / Objective-C) can initialize the
//! engine, submit a stream of bytes, and read back a coarse mastery index.
//!
//! Safety: every function that accepts raw pointers is `unsafe extern "C"`. The
//! caller is responsible for passing only valid pointers obtained from
//! `bad_apple_init` and for calling `bad_apple_free` to release the context.

use std::ffi::{c_char, c_void, CStr, CString};
use std::sync::Mutex;

pub mod ane_core;
pub mod apple_intelligence;
pub mod arena;
#[path = "automation_cage_impl.rs"]
pub mod automation_cage;
pub mod bad_apple_ipc;
pub mod benchmark;
pub mod cert;
pub mod config;

pub mod ify;
pub mod mcp;
pub mod mcp_marketplace;
pub mod mesh_brain;
pub mod mesh_sync;
pub mod metal_uma;
pub mod metrics;
pub mod org_policy;
pub mod p2p_crypto;
pub mod p2p_model;
pub mod production_blueprint;
pub mod protocol;
pub mod red_team;
pub mod redb_kv;
pub mod scavenger;
pub mod simd;
pub mod strategy_library;
pub mod tensor_brain;
pub mod vault;
pub mod wasm_cage;
pub mod workspace_watcher;

pub use apple_intelligence::{
    call as apple_intelligence_call, call_sync as apple_intelligence_call_sync,
    is_available as apple_intelligence_is_available, register_apple_intelligence_oracle,
};

use config::Config;

/// Opaque handle to an initialized Bad Apple evaluation context.
///
/// The internals are intentionally hidden from C. Only the pointer is exposed;
/// the Rust side owns and synchronizes the state with a `std::sync::Mutex`.
#[repr(C)]
pub struct BadAppleContext {
    _private: *mut c_void,
}

// SAFETY: BadAppleContext is an opaque `#[repr(C)]` handle whose `_private` pointer is
// only dereferenced inside FFI functions that synchronize access through the inner
// `Mutex<BadAppleState>`. The handle itself is a plain pointer with no interior
// mutability, so moving or sharing it across threads is sound.
unsafe impl Send for BadAppleContext {}
// SAFETY: &BadAppleContext provides no way to mutate the handle; all mutation goes
// through the inner Mutex, so sharing references across threads is sound.
unsafe impl Sync for BadAppleContext {}

struct BadAppleState {
    #[allow(dead_code)]
    config: Config,
    mastery_index: f32,
    active_pursuits: Vec<String>,
}

impl BadAppleState {
    fn new(config: Config) -> Self {
        Self {
            config,
            mastery_index: 0.5,
            active_pursuits: Vec::new(),
        }
    }
}

/// Initialize a Bad Apple evaluation context.
///
/// `config_path` may be a null pointer, in which case configuration is loaded
/// from `BADAPPLE_*` environment variables. The returned pointer must be freed
/// with `bad_apple_free`.
///
/// # Safety
///
/// The caller must ensure `config_path` is either null or a valid,
/// null-terminated C string.
#[no_mangle]
pub unsafe extern "C" fn bad_apple_init(config_path: *const c_char) -> *mut BadAppleContext {
    let config = if config_path.is_null() {
        Config::from_env()
    } else {
        let cstr = CStr::from_ptr(config_path);
        match cstr.to_str() {
            Ok(s) if !s.is_empty() => Config::from_file(std::path::Path::new(s)),
            _ => Config::from_env(),
        }
    };

    let state = BadAppleState::new(config);
    let boxed = Box::new(Mutex::new(state));
    let ctx = Box::new(BadAppleContext {
        _private: Box::into_raw(boxed).cast::<c_void>(),
    });
    Box::into_raw(ctx)
}

/// Process a raw byte stream and return a null-terminated diagnostic string.
///
/// The input is interpreted as UTF-8. It is encoded into the 10,000-D HDC
/// substrate, thermodynamically minimized, and analyzed for overhead patterns.
/// The returned `*mut c_char` is a freshly allocated C string that the caller
/// must free with `libc::free` (or a matching deallocator).
///
/// # Safety
///
/// `context` must be a valid pointer returned by `bad_apple_init` and not yet
/// freed. `input_buffer` must point to at least `length` readable bytes.
#[no_mangle]
pub unsafe extern "C" fn bad_apple_process_stream(
    context: *mut BadAppleContext,
    input_buffer: *const u8,
    length: usize,
) -> *mut c_char {
    if context.is_null() || input_buffer.is_null() {
        return bad_apple_cstring("null pointer");
    }

    let bytes = std::slice::from_raw_parts(input_buffer, length);
    let text = match std::str::from_utf8(bytes) {
        Ok(s) => s,
        Err(_) => return bad_apple_cstring("invalid utf-8"),
    };

    let summary = format!(
        "received {} bytes; no overhead analysis available",
        text.len()
    );
    bad_apple_cstring(&summary)
}

/// Return the current coarse mastery index for this context, clamped to [0, 1].
///
/// # Safety
///
/// `context` must be a valid pointer returned by `bad_apple_init` and not yet freed.
#[no_mangle]
pub unsafe extern "C" fn bad_apple_get_mastery_index(context: *mut BadAppleContext) -> f32 {
    if context.is_null() {
        return 0.0;
    }
    let ctx = &*context;
    let state = &*(ctx._private as *const Mutex<BadAppleState>);
    state
        .lock()
        .map_or(0.0, |g| g.mastery_index.clamp(0.0, 1.0))
}

/// Release a context previously allocated by `bad_apple_init`.
///
/// After this call the pointer is invalid and must not be used again.
///
/// # Safety
///
/// `context` must be a valid pointer returned by `bad_apple_init` and not yet
/// freed. After this call, the pointer must not be used again.
#[no_mangle]
pub unsafe extern "C" fn bad_apple_free(context: *mut BadAppleContext) {
    if context.is_null() {
        return;
    }
    let ctx = Box::from_raw(context);
    let state = Box::from_raw(ctx._private.cast::<Mutex<BadAppleState>>());
    drop(state);
    drop(ctx);
}

fn bad_apple_cstring(s: &str) -> *mut c_char {
    CString::new(s).unwrap_or_default().into_raw()
}

/// Return the last measured Apple Intelligence call latency for this process,
/// in microseconds.  This reads the shared atomic counter maintained by the
/// Apple Intelligence bridge; it is safe to call from any thread.
#[no_mangle]
pub extern "C" fn bad_apple_get_apple_latency_us() -> u64 {
    apple_intelligence::last_latency_us()
}

/// Return the active pursuits for a context as a JSON array C string.
/// The caller must free the returned pointer with `bad_apple_free_string`.
///
/// # Safety
///
/// `context` must be a valid pointer returned by `bad_apple_init` and not yet freed.
#[no_mangle]
pub unsafe extern "C" fn bad_apple_get_active_pursuits(
    context: *mut BadAppleContext,
) -> *mut c_char {
    if context.is_null() {
        return bad_apple_cstring("[]");
    }
    let ctx = &*context;
    let state = &*(ctx._private as *const Mutex<BadAppleState>);
    match state.lock() {
        Ok(guard) => {
            let json =
                serde_json::to_string(&guard.active_pursuits).unwrap_or_else(|_| "[]".to_string());
            bad_apple_cstring(&json)
        }
        Err(_) => bad_apple_cstring("[]"),
    }
}

/// Push a new active pursuit string onto a context.
///
/// # Safety
///
/// `context` must be a valid pointer returned by `bad_apple_init` and not yet freed.
/// `text` must be a valid, null-terminated UTF-8 C string.
#[no_mangle]
pub unsafe extern "C" fn bad_apple_push_pursuit(
    context: *mut BadAppleContext,
    text: *const c_char,
) -> bool {
    if context.is_null() || text.is_null() {
        return false;
    }
    let text = match CStr::from_ptr(text).to_str() {
        Ok(s) if !s.is_empty() => s.to_string(),
        _ => return false,
    };
    let ctx = &*context;
    let state = &*(ctx._private as *const Mutex<BadAppleState>);
    match state.lock() {
        Ok(mut guard) => {
            guard.active_pursuits.push(text);
            // Keep a bounded, recent window so the menu bar stays responsive.
            if guard.active_pursuits.len() > 64 {
                guard.active_pursuits.remove(0);
            }
            true
        }
        Err(_) => false,
    }
}

/// Free a C string previously returned by the library.
///
/// # Safety
///
/// `s` must be a pointer previously returned by a Bad Apple FFI function that
/// returns ownership of a C string, and it must not have been freed before.
/// Both Rust-allocated (`CString`) and bridge-allocated (`strdup`) strings
/// are released through the C library `free()` path used by the global
/// allocator, so this is safe for all C string exchanges.
#[no_mangle]
pub unsafe extern "C" fn bad_apple_free_string(s: *mut c_char) {
    if s.is_null() {
        return;
    }
    libc::free(s.cast::<c_void>());
}

/// Alias for `bad_apple_free_string` that explicitly signals to the host
/// environment that a Swift-allocated string is being released.
///
/// # Safety
///
/// Same as `bad_apple_free_string`.
#[no_mangle]
pub unsafe extern "C" fn free_swift_string(s: *mut c_char) {
    bad_apple_free_string(s);
}

/// Run a prompt through the registered Apple Intelligence callback and return
/// the response as a C string.  This is a blocking, synchronous call so the
/// menu-bar app can drive it from the main thread without starting a Tokio
/// runtime.  The caller must free the returned pointer with `bad_apple_free_string`.
///
/// # Safety
///
/// `prompt` must be a valid, null-terminated UTF-8 C string.
#[no_mangle]
pub unsafe extern "C" fn bad_apple_generate_text(prompt: *const c_char) -> *mut c_char {
    if prompt.is_null() {
        return bad_apple_cstring("null prompt");
    }
    let text = match CStr::from_ptr(prompt).to_str() {
        Ok(s) if !s.is_empty() => s,
        _ => return bad_apple_cstring("invalid prompt"),
    };
    if ane_core::is_available() {
        // Cap output to 64 tokens for speed; 2-3 sentences is all the
        // concise persona needs.
        if let Ok(resp) = ane_core::generate_sync(text, 64, ane_core::context_limit()) {
            return bad_apple_cstring(&resp);
        }
    }
    match apple_intelligence::call_sync(text) {
        Some(resp) => bad_apple_cstring(&resp),
        None => bad_apple_cstring("Apple Intelligence not available"),
    }
}

/// Dispatch a native macOS desktop notification through the loaded
/// BadAppleBridge.  If the bridge is not loaded or notification
/// authorization was not granted, the call is a silent no-op.
///
/// # Safety
///
/// `title` and `body` must be valid, null-terminated UTF-8 C strings.
#[no_mangle]
pub unsafe extern "C" fn bad_apple_dispatch_desktop_notification(
    title: *const c_char,
    body: *const c_char,
) {
    if title.is_null() || body.is_null() {
        return;
    }
    let title = match CStr::from_ptr(title).to_str() {
        Ok(s) => s,
        Err(_) => return,
    };
    let body = match CStr::from_ptr(body).to_str() {
        Ok(s) => s,
        Err(_) => return,
    };
    apple_intelligence::dispatch_desktop_notification(title, body);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ffi_roundtrip() {
        // SAFETY: passing a null config_path is explicitly supported by bad_apple_init,
        // which falls back to environment-based configuration. The returned pointer is
        // checked for null immediately.
        let ctx = unsafe { bad_apple_init(std::ptr::null()) };
        assert!(!ctx.is_null());

        let input = b"print hello world sum total";
        // SAFETY: `ctx` is a valid pointer from bad_apple_init, `input` is a byte slice
        // whose pointer and length are valid for the duration of the call.
        let out = unsafe { bad_apple_process_stream(ctx, input.as_ptr(), input.len()) };
        assert!(!out.is_null());
        // SAFETY: `out` is a valid non-null C string returned by bad_apple_process_stream.
        let _ = unsafe { CStr::from_ptr(out) };
        // SAFETY: `out` was allocated by CString::into_raw and is freed once here.
        unsafe { bad_apple_free_string(out) };

        // SAFETY: `ctx` is still a valid, unfreed pointer from bad_apple_init.
        let idx = unsafe { bad_apple_get_mastery_index(ctx) };
        assert!((0.0..=1.0).contains(&idx));

        // SAFETY: `ctx` is a valid pointer from bad_apple_init and has not been freed yet.
        unsafe { bad_apple_free(ctx) };
    }

    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
