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
        .map_or_else(|_| PathBuf::from(&crate_dir).join("target"), PathBuf::from)
        .join(env::var("PROFILE").unwrap_or_else(|_| "debug".into()));
    std::fs::create_dir_all(&target_dir).ok();

    let header_path = target_dir.join("bad_apple_core.h");
    let config_path = PathBuf::from(&crate_dir).join("cbindgen.toml");

    match cbindgen::Builder::new()
        .with_crate(crate_dir)
        .with_config(cbindgen::Config::from_root_or_default(&config_path))
        .generate()
    {
        Ok(bindings) => {
            bindings.write_to_file(&header_path);
        }
        Err(cbindgen::Error::ParseSyntaxError { ref src_path, .. }) => {
            // cbindgen cannot parse every valid Rust file (e.g. modules that use
            // tokio types with attributes it doesn't understand). These modules
            // are not part of the C-ABI surface, so a stale/pre-existing header
            // is acceptable. If no header exists yet, write an empty one so the
            // rest of the build can proceed; the Swift bridge can regenerate it
            // from a successful run.
            eprintln!("cbindgen parse error in {src_path}; keeping existing C header if present");
            if !header_path.exists() {
                std::fs::write(
                    &header_path,
                    "/* cbindgen parse error; regenerate later */\n",
                )
                .expect("failed to write fallback header");
            }
        }
        Err(e) => panic!("Unable to generate C bindings: {e:?}"),
    }

    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=Cargo.toml");
    println!("cargo:rerun-if-changed=cbindgen.toml");
}
