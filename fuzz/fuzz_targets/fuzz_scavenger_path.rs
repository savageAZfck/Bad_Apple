#![no_main]
use libfuzzer_sys::fuzz_target;

// Fuzz the scavenger's file-path handling with arbitrary strings.
//
// The scavenger's security model depends on several std-library path
// operations that must never panic on hostile input:
//
// * `Path::components` – used by `is_ignored_path` to filter directory names.
// * `Path::starts_with` – used by `path_under_roots` to enforce root containment.
// * `fs::canonicalize` – used by `path_under_roots` to resolve symlinks.
// * `fs::symlink_metadata` – used by `is_tracked_file` and `collect_files`
//   to reject symlinks before following them.
// * `Path::extension` – used by `is_tracked_file` to filter file types.
// * `Path::parent` / `Path::join` – used by `write_path_catalog` and
//   `ScavengerConfig::from_env` to derive sibling paths.
//
// Additionally, we construct a `ScavengerConfig` from the fuzzer data to
// verify that path-based config construction does not panic on arbitrary
// input (it should return an error instead).
fuzz_target!(|data: &[u8]| {
    use std::path::Path;

    let Ok(s) = std::str::from_utf8(data) else {
        return;
    };
    let path = Path::new(s);

    // --- Component iteration (is_ignored_path) ----------------------
    // The scavenger iterates path components to check for skipped
    // directory names.  Verify this never panics on weird paths.
    let _ = path.components().count();
    for component in path.components() {
        let _ = component.as_os_str().to_str();
    }

    // --- Root containment (path_under_roots) -------------------------
    // The scavenger canonicalises the event path and checks starts_with
    // against each canonical root.  Exercise both operations.
    let _ = path.starts_with("/");
    let _ = path.is_absolute();
    let _ = std::fs::canonicalize(path);

    // --- Symlink rejection (is_tracked_file / collect_files) ---------
    // The scavenger uses symlink_metadata to avoid following symlinks.
    let _ = std::fs::symlink_metadata(path);

    // --- Extension filtering (is_tracked_file) -----------------------
    let _ = path.extension();
    if let Some(ext) = path.extension() {
        if let Some(ext_str) = ext.to_str() {
            let _ = ext_str.to_lowercase();
        }
    }

    // --- Parent / join (write_path_catalog / from_env) ---------------
    let _ = path.parent();
    if let Some(parent) = path.parent() {
        let _ = parent.join("scavenger_paths.json");
        let _ = parent.join("scavenger_paths.json.tmp");
        let _ = parent.join("tokenizer.json");
        let _ = parent.join("embedding.f16");
    }

    // --- ScavengerConfig construction --------------------------------
    // Building a config from arbitrary path strings must not panic.
    // We bypass from_env() (which reads environment variables) and
    // construct the struct directly to focus on path handling.
    use bad_apple::scavenger::ScavengerConfig;
    use std::path::PathBuf;
    let _config = ScavengerConfig {
        watch_dirs: vec![PathBuf::from(s)],
        sled_db_path: PathBuf::from(s),
        tokenizer_path: PathBuf::from(s),
        embedding_path: PathBuf::from(s),
        hidden_size: 2048,
        vocab_size: 151_936,
        reset_db: false,
    };

    // --- Path with NUL bytes -----------------------------------------
    // NUL bytes in paths can cause issues on Unix.  Verify that path
    // operations handle them gracefully (they should not panic).
    if data.iter().any(|&b| b == 0) {
        let _ = path.to_string_lossy();
        let _ = path.as_os_str().to_str();
    }
});
