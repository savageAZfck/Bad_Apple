//! Air-gap certification self-test suite.
//!
//! The implementation lives in `src/cert.rs`; this file is a thin integration
//! wrapper so the suite runs as part of `cargo test --test cert_suite`.

#[test]
fn air_gap_cert_suite() {
    let results = bad_apple::cert::run();
    let mut failed = Vec::new();
    for r in &results {
        eprintln!(
            "[cert] {}: {}",
            if r.passed { "PASS" } else { "FAIL" },
            r.name
        );
        if !r.passed {
            failed.push(format!("{}: {}", r.name, r.message));
        }
    }
    assert!(
        failed.is_empty(),
        "cert suite failed:\n{}",
        failed.join("\n")
    );
}
