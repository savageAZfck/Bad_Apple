// Per-artifact ANE placement audit + per-op census via the bridge FFI.
// usage: ane_audit <model.mlmodelc> [computeUnitsRaw] [report_path]
use libloading::Library;
use std::ffi::CString;
use std::os::raw::c_char;
use std::path::PathBuf;

type ReportFn = unsafe extern "C" fn(*const c_char, i32, *const c_char) -> f64;

fn main() {
    let path = PathBuf::from(std::env::args().nth(1).expect("model path"));
    let cu: i32 = std::env::args()
        .nth(2)
        .and_then(|v| v.parse().ok())
        .unwrap_or(3);
    match bad_apple::ane_core::audit_artifact_placement(&path, cu) {
        Ok(r) => println!("{} -> {:.1}% ANE", path.display(), r * 100.0),
        Err(e) => println!("{} -> audit failed: {e}", path.display()),
    }
    if let Some(report) = std::env::args().nth(3) {
        let lib = unsafe { Library::new("target/release/libBadAppleBridge.dylib") }
            .expect("bridge dylib");
        let report_fn: libloading::Symbol<ReportFn> =
            unsafe { lib.get(b"bad_apple_coreml_op_placement_report\0") }.expect("report symbol");
        let model_c = CString::new(path.to_string_lossy().as_bytes()).unwrap();
        let report_c = CString::new(report.as_str()).unwrap();
        let ratio = unsafe { report_fn(model_c.as_ptr(), cu, report_c.as_ptr()) };
        println!("report ratio {:.1}% -> {}", ratio * 100.0, report);
    }
}
