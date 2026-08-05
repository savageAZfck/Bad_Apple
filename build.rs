use std::env;
use std::path::PathBuf;

fn main() {
    // macOS: the Swift bridge dylib calls back into the executable to register
    // the Apple Intelligence callback.  In release builds with LTO enabled, the
    // linker may not put public C symbols in the dynamic symbol table by
    // default, causing the dylib to call a NULL function pointer and segfault.
    // `-Wl,-export_dynamic` keeps the required symbols visible at runtime.
    if let Ok(target) = env::var("TARGET") {
        if target.contains("apple-darwin") {
            println!("cargo:rustc-link-arg=-Wl,-export_dynamic");
        }
    }

    let crate_dir = env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR missing");
    let target_dir = env::var("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(&crate_dir).join("target"))
        .join(env::var("PROFILE").unwrap_or_else(|_| "debug".into()));
    std::fs::create_dir_all(&target_dir).ok();

    let header_path = target_dir.join("firefly_core.h");

    cbindgen::Builder::new()
        .with_crate(crate_dir)
        .with_language(cbindgen::Language::C)
        .with_no_includes()
        .generate()
        .expect("Unable to generate C bindings")
        .write_to_file(&header_path);

    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=Cargo.toml");
}
