# Bad Apple Fuzz Targets

Coverage-guided fuzzing for the three most security-critical parsing and IPC
boundaries in the Bad Apple runtime.

## Targets

| Binary                   | Boundary                                        |
| ------------------------ | ----------------------------------------------- |
| `fuzz_ipc_frame`         | SLICKS IPC frame parsing (`read_frame`, nonce / timestamp / request validation) |
| `fuzz_wasm_cage`         | WASM cage compilation and fuel-metered execution |
| `fuzz_scavenger_path`    | Scavenger file-path handling (canonicalisation, symlink rejection, root containment) |
| `fuzz_protocol_frame`    | Multi-agent protocol frame parsing (`SignedUdpPacket`, `CompactEngramPacket`, HMAC verification) |

## Prerequisites

Install `cargo-fuzz` (a cargo subcommand that wires up libFuzzer, sanitizer
flags, and coverage instrumentation):

```bash
cargo install cargo-fuzz
```

`cargo-fuzz` requires a nightly Rust toolchain for sanitizer support:

```bash
rustup toolchain install nightly
```

## Running a target

From the repository root:

```bash
# Run the IPC frame fuzzer for 60 seconds
cargo +nightly fuzz run fuzz_ipc_frame -- -max_total_time=60

# Run the WASM cage fuzzer with a larger memory limit
cargo +nightly fuzz run fuzz_wasm_cage -- -max_total_time=120 -rss_limit_mb=4096

# Run the scavenger path fuzzer
cargo +nightly fuzz run fuzz_scavenger_path -- -max_total_time=60

# Run the protocol frame fuzzer
cargo +nightly fuzz run fuzz_protocol_frame -- -max_total_time=60
```

### Useful libFuzzer flags

| Flag                    | Description                                      |
| ----------------------- | ------------------------------------------------ |
| `-max_total_time=N`     | Stop after N seconds.                            |
| `-max_len=N`            | Cap input length at N bytes.                     |
| `-rss_limit_mb=N`       | Abort if RSS exceeds N MB (default 2048).        |
| `-timeout=N`            | Abort if a single run takes more than N seconds. |
| `-only_ascii=1`         | Restrict inputs to ASCII.                        |
| `-dict=path`            | Load a mutation dictionary.                      |
| `-artifact_prefix=dir/` | Write crash inputs to `dir/`.                    |

## Reproducing a crash

When a target finds a crash, libFuzzer writes the offending input to a file
(typically `fuzz/artifacts/fuzz_<name>/crash-<hash>`).  Reproduce it with:

```bash
cargo +nightly fuzz run fuzz_ipc_frame -- reproduce fuzz/artifacts/fuzz_ipc_frame/crash-<hash>
```

## Minimising a crash

To reduce a crash input to the smallest equivalent:

```bash
cargo +nightly fuzz run fuzz_ipc_frame -- minimize fuzz/artifacts/fuzz_ipc_frame/crash-<hash>
```

## Corpus management

libFuzzer maintains a corpus directory under `fuzz/corpus/fuzz_<name>/`.
Commit interesting seed inputs there to guide future runs:

```bash
# Add a seed
echo '{"type":"hello","version":1,"timestamp_ms":0,"client_nonce":"0000000000000000000000000000000000000000000000000000000000000000"}' > fuzz/corpus/fuzz_ipc_frame/seed_hello.json

# Merge corpora from multiple runs
cargo +nightly fuzz run fuzz_ipc_frame -- merge fuzz/corpus/fuzz_ipc_frame
```

## Architecture

The fuzz crate is a standalone Cargo package (`fuzz/Cargo.toml`) that depends
on the `bad_apple` library via a path dependency.  It is **not** part of the
root workspace, so it can be built and run independently without affecting the
main build.

Each fuzz target is a `#![no_main]` binary that uses the `libfuzzer_sys`
`fuzz_target!` macro to expose a test function to libFuzzer.  The targets call
only the **public** API of `bad_apple` — no private functions are exercised.

### What each target covers

**`fuzz_ipc_frame`** — Feeds arbitrary bytes through `read_frame` (newline-
delimited JSON deserialisation for both `ClientFrame` and `ServerFrame`),
`nonce_is_valid`, `timestamp_is_fresh`, and `validate_request`.  The parser
must return an error for malformed input and must never panic, hang, or cause
undefined behaviour.

**`fuzz_wasm_cage`** — Creates a fresh `WasmCage` and calls `compile` on the
fuzzer data.  If compilation succeeds, `run_with_input` is called with the same
data and with an empty buffer.  The cage enforces a 1 MiB linear-memory cap,
single-memory policy, and fuel metering — all of which must hold under
adversarial input.

**`fuzz_scavenger_path`** — Exercises the std-library path operations that the
scavenger's security model depends on: `Path::components` (used by
`is_ignored_path`), `Path::starts_with` and `fs::canonicalize` (used by
`path_under_roots`), `fs::symlink_metadata` (used by `is_tracked_file` and
`collect_files`), and `Path::extension` / `Path::parent` / `Path::join`.  Also
constructs a `ScavengerConfig` from arbitrary paths to verify that config
construction is panic-free.

**`fuzz_protocol_frame`** — Deserialises arbitrary bytes as `SignedUdpPacket`
and `CompactEngramPacket`, then exercises `decode_payload` (base64) and
`verify_packet` (HMAC-SHA256 constant-time comparison).  Also parses as a
generic `serde_json::Value` as a baseline.
