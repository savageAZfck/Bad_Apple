use std::env;
use std::path::PathBuf;

fn main() {
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
