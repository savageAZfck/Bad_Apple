//! Model-authored probes — the generative layer of the self red-team.
//!
//! Built-in probes verify *known* weakness classes. This module lets the
//! organism hypothesize *new* weaknesses: the model emits a `ProbeSpec`
//! (declarative JSON), which is validated against a strict schema and a
//! filesystem jail, persisted to `~/.bad_apple/redteam/probes/`, and executed
//! by a check interpreter instead of arbitrary code.
//!
//! Safety invariants — enforced for every generated probe, forever:
//! * checks are read-only: they can inspect files, sockets, and loopback
//!   HTTP endpoints but cannot write files, spawn processes (other than a
//!   read-only `pgrep`), or reach non-loopback addresses;
//! * filesystem paths must resolve inside the allowed roots;
//! * HTTP checks never carry credentials — a probe always sees the same
//!   boundary an unauthenticated caller would;
//! * the registry is bounded and FIFO-evicted, so probe generation cannot
//!   grow memory or runtime without limit.

use super::probes::elapsed_us;
use super::{now_ms, Probe, ProbeResult, Severity};
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::io::{Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// Category reported for every generated probe.
pub const GENERATED_CATEGORY: &str = "generated";
/// Maximum persisted generated probes (FIFO eviction beyond this).
pub const MAX_GENERATED_PROBES: usize = 64;
/// Maximum checks inside a single probe spec.
pub const MAX_CHECKS: usize = 8;
/// Bytes cap for socket frames sent by checks.
const MAX_FRAME_BYTES: usize = 4096;
/// Bytes cap for socket/HTTP responses read by checks.
const MAX_RESPONSE_BYTES: usize = 8192;
/// Bytes a file-content check may scan. Larger files fail closed — a probe
/// must never be able to turn a check into an unbounded read.
const MAX_FILE_SCAN_BYTES: u64 = 16 * 1024 * 1024;
/// Cap for probe title/hypothesis text.
const MAX_TEXT_LEN: usize = 600;

/// Declarative probe spec authored by the model and interpreted by this engine.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProbeSpec {
    /// Probe id — must start with `gen.` and match `[a-z0-9_]`.
    pub id: String,
    /// One-line summary of the weakness being tested.
    pub title: String,
    /// The model's reasoning about why this might be exploitable.
    pub hypothesis: String,
    pub severity: Severity,
    /// Ordered checks; the probe is mitigated only if ALL pass.
    pub checks: Vec<Check>,
}

/// A single read-only check inside a generated probe.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Check {
    /// Fails if the path exists (e.g. a world-readable secret must not appear).
    FileAbsent { path: String },
    /// Fails if the path does not exist.
    FilePresent { path: String },
    /// Fails if the file's permission bits are more permissive than `mode`
    /// (octal string, e.g. "0600"). A missing file fails — the guard the mode
    /// protects must exist to be meaningful.
    FileModeMax { path: String, mode: String },
    /// Fails if the file does not contain `pattern`.
    FileContains { path: String, pattern: String },
    /// Fails if the file contains `pattern` (secret-in-ledger checks).
    FileNotContains { path: String, pattern: String },
    /// Fails unless `url` (loopback HTTP only) returns `status`. No
    /// credentials are ever attached — the check sees what an
    /// unauthenticated local caller sees.
    HttpStatus { url: String, status: u16 },
    /// Fails if `url` (loopback HTTP only) returns `status` — e.g. a
    /// sensitive route that must not answer 200 without auth.
    HttpNotStatus { url: String, status: u16 },
    /// Fails if a TCP connection to `addr` succeeds (port should be closed).
    /// Non-loopback addresses are refused before connecting.
    PortClosed { addr: String },
    /// Fails if the unix socket accepts `frame` and answers with a clean,
    /// error-free response. Passes on connect failure, timeout, malformed
    /// response, or an explicit error reply.
    SocketRejects { path: String, frame: String },
    /// Fails if no process matches `name` (`pgrep -x`).
    ProcessPresent { name: String },
    /// Fails if a process matches `name` (`pgrep -x`).
    ProcessAbsent { name: String },
}

/// Filesystem roots generated checks may read. Everything outside — or
/// anything that resolves outside through a symlink — is refused before IO.
fn allowed_roots() -> Vec<PathBuf> {
    let mut roots: Vec<PathBuf> = Vec::new();
    if let Ok(extra) = std::env::var("BADAPPLE_RED_TEAM_ROOTS") {
        roots.extend(
            extra
                .split(':')
                .filter(|s| !s.is_empty())
                .map(PathBuf::from),
        );
    }
    roots.push(PathBuf::from("/var/lib/bad_apple"));
    roots.push(PathBuf::from("/var/run/badapple"));
    if let Some(home) = std::env::var_os("HOME") {
        roots.push(PathBuf::from(home).join(".bad_apple"));
    }
    roots.push(PathBuf::from("/tmp"));
    roots.push(PathBuf::from("/private/tmp"));
    roots.push(PathBuf::from(env!("CARGO_MANIFEST_DIR")));
    roots
}

/// Resolve `path` (symlinks included) and require it under an allowed root.
/// Returns the resolved path. Nonexistent paths are resolved lexically for
/// `file_absent`-style checks by canonicalizing the parent.
fn jail_path(path: &str) -> Result<PathBuf> {
    let raw = Path::new(path);
    if !raw.is_absolute() {
        bail!("path must be absolute: {path}");
    }
    if path.contains("..") {
        bail!("path must not contain '..': {path}");
    }
    let resolved = if raw.exists() {
        raw.canonicalize()
            .with_context(|| format!("cannot resolve {path}"))?
    } else {
        let parent = raw
            .parent()
            .and_then(|p| p.canonicalize().ok())
            .with_context(|| format!("cannot resolve parent of {path}"))?;
        match raw.file_name() {
            Some(name) => parent.join(name),
            None => bail!("path has no file name: {path}"),
        }
    };
    for root in allowed_roots() {
        let Ok(root) = root.canonicalize() else {
            continue;
        };
        if resolved.starts_with(&root) {
            return Ok(resolved);
        }
    }
    bail!("path is outside allowed roots: {path}");
}

/// Parse an octal mode string like "0600".
fn parse_mode(s: &str) -> Result<u32> {
    if s.is_empty() || s.len() > 4 || !s.bytes().all(|b| b.is_ascii_digit()) {
        bail!("invalid mode string: {s}");
    }
    u32::from_str_radix(s, 8).with_context(|| format!("invalid octal mode: {s}"))
}

fn host_is_loopback(host: &str) -> bool {
    matches!(host, "127.0.0.1" | "localhost" | "::1")
}

/// Split a loopback HTTP URL into (host, port, path). Anything else fails.
fn parse_loopback_url(url: &str) -> Result<(String, u16, String)> {
    let rest = url
        .strip_prefix("http://")
        .with_context(|| format!("only http:// URLs are allowed: {url}"))?;
    let (authority, path) = match rest.find('/') {
        Some(i) => (&rest[..i], &rest[i..]),
        None => (rest, "/"),
    };
    let (host, port) = if let Some(bare) = authority.strip_prefix('[') {
        let end = bare.find(']').context("malformed IPv6 authority")?;
        let host = &bare[..end];
        let tail = &bare[end + 1..];
        let port = tail
            .strip_prefix(':')
            .and_then(|p| p.parse().ok())
            .unwrap_or(80);
        (host.to_string(), port)
    } else {
        match authority.rsplit_once(':') {
            Some((h, p)) => (h.to_string(), p.parse().unwrap_or(80)),
            None => (authority.to_string(), 80),
        }
    };
    if !host_is_loopback(&host) {
        bail!("HTTP checks may only target loopback hosts: {url}");
    }
    if path.contains("..") {
        bail!("HTTP path must not contain '..': {url}");
    }
    Ok((host, port, path.to_string()))
}

/// Minimal loopback GET — enough for status assertions, no redirects, no TLS.
fn http_get_status(url: &str) -> Result<u16> {
    let (host, port, path) = parse_loopback_url(url)?;
    let addr = (host.as_str(), port)
        .to_socket_addrs()
        .context("cannot resolve address")?
        .next()
        .context("no address")?;
    let mut stream =
        TcpStream::connect_timeout(&addr, Duration::from_secs(2)).context("connect failed")?;
    stream.set_read_timeout(Some(Duration::from_secs(3))).ok();
    let req = format!(
        "GET {path} HTTP/1.1\r\nHost: {host}:{port}\r\nConnection: close\r\nUser-Agent: badapple-redteam\r\n\r\n"
    );
    stream.write_all(req.as_bytes())?;
    let mut buf = vec![0u8; MAX_RESPONSE_BYTES];
    let n = stream.read(&mut buf).unwrap_or(0);
    let head = String::from_utf8_lossy(&buf[..n]);
    let status = head
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse::<u16>().ok())
        .context("malformed HTTP response")?;
    Ok(status)
}

/// Send a bounded frame to a unix socket and read a bounded reply.
/// Returns Ok(body) only when the socket produced a clean, error-free reply.
fn socket_roundtrip(path: &Path, frame: &str) -> Result<String> {
    if frame.len() > MAX_FRAME_BYTES {
        bail!("socket frame exceeds {MAX_FRAME_BYTES} bytes");
    }
    let mut stream = UnixStream::connect(path).context("connect failed")?;
    stream.set_read_timeout(Some(Duration::from_secs(3))).ok();
    stream.set_write_timeout(Some(Duration::from_secs(2))).ok();
    stream.write_all(frame.as_bytes())?;
    stream.write_all(b"\n")?;
    let mut buf = vec![0u8; MAX_RESPONSE_BYTES];
    let n = stream.read(&mut buf).context("read failed")?;
    if n == 0 {
        bail!("socket closed without a reply");
    }
    Ok(String::from_utf8_lossy(&buf[..n]).to_string())
}

/// Execute one check. Returns Ok(()) when the check holds (mitigated) or a
/// description of the violation when it does not.
fn run_check(check: &Check) -> std::result::Result<(), String> {
    match check {
        Check::FileAbsent { path } => match jail_path(path) {
            Err(e) => Err(format!("jail: {e}")),
            Ok(p) if p.exists() => Err(format!("{path} exists but must not")),
            Ok(_) => Ok(()),
        },
        Check::FilePresent { path } => match jail_path(path) {
            Err(e) => Err(format!("jail: {e}")),
            Ok(p) if p.is_file() || p.is_dir() => Ok(()),
            Ok(_) => Err(format!("{path} does not exist")),
        },
        Check::FileModeMax { path, mode } => {
            let max = parse_mode(mode).map_err(|e| e.to_string())?;
            match jail_path(path) {
                Err(e) => Err(format!("jail: {e}")),
                Ok(p) => {
                    use std::os::unix::fs::PermissionsExt;
                    match std::fs::metadata(&p) {
                        Err(e) => Err(format!("cannot stat {path}: {e}")),
                        Ok(m) => {
                            let actual = m.permissions().mode() & 0o777;
                            if actual <= max {
                                Ok(())
                            } else {
                                Err(format!(
                                    "{path} mode {actual:04o} is more permissive than {max:04o}"
                                ))
                            }
                        }
                    }
                }
            }
        }
        Check::FileContains { path, pattern } | Check::FileNotContains { path, pattern } => {
            let negated = matches!(check, Check::FileNotContains { .. });
            let read = || -> std::result::Result<String, String> {
                let p = jail_path(path).map_err(|e| format!("jail: {e}"))?;
                let len = std::fs::metadata(&p)
                    .map_err(|e| format!("cannot stat {path}: {e}"))?
                    .len();
                if len > MAX_FILE_SCAN_BYTES {
                    return Err(format!(
                        "{path} is {len} bytes, over the {MAX_FILE_SCAN_BYTES}-byte scan cap"
                    ));
                }
                std::fs::read_to_string(&p).map_err(|e| format!("cannot read {path}: {e}"))
            };
            match read() {
                Err(e) => Err(e),
                Ok(text) => {
                    let found = text.contains(pattern.as_str());
                    if found == negated {
                        Err(format!(
                            "{path} {} '{pattern}'",
                            if negated {
                                "contains forbidden"
                            } else {
                                "lacks required"
                            }
                        ))
                    } else {
                        Ok(())
                    }
                }
            }
        }
        Check::HttpStatus { url, status } => match http_get_status(url) {
            Err(e) => Err(format!("request failed: {e}")),
            Ok(s) if s == *status => Ok(()),
            Ok(s) => Err(format!("{url} returned {s}, expected {status}")),
        },
        Check::HttpNotStatus { url, status } => match http_get_status(url) {
            Err(e) => Err(format!("request failed: {e}")),
            Ok(s) if s != *status => Ok(()),
            Ok(s) => Err(format!("{url} returned {s}, which must not happen")),
        },
        Check::PortClosed { addr } => {
            let (host, _) = addr.rsplit_once(':').unwrap_or((addr, ""));
            if !host_is_loopback(host.trim_start_matches('[').trim_end_matches(']')) {
                return Err(format!("port checks may only target loopback: {addr}"));
            }
            let addrs: Vec<_> = addr
                .to_socket_addrs()
                .map(|it| it.collect())
                .unwrap_or_default();
            if addrs.is_empty() {
                return Err(format!("cannot parse address: {addr}"));
            }
            for a in addrs {
                if TcpStream::connect_timeout(&a, Duration::from_millis(500)).is_ok() {
                    return Err(format!("{addr} accepted a connection"));
                }
            }
            Ok(())
        }
        Check::SocketRejects { path, frame } => match jail_path(path) {
            Err(e) => Err(format!("jail: {e}")),
            Ok(p) => match socket_roundtrip(&p, frame) {
                Err(_) => Ok(()),
                Ok(body) => {
                    let low = body.to_lowercase();
                    if low.contains("\"ok\":true") || low.contains("accepted") {
                        Err(format!(
                            "{path} accepted a malformed frame: {}",
                            &body[..body.len().min(160)]
                        ))
                    } else {
                        Ok(())
                    }
                }
            },
        },
        Check::ProcessPresent { name } | Check::ProcessAbsent { name } => {
            let negated = matches!(check, Check::ProcessAbsent { .. });
            if name.is_empty()
                || name.len() > 64
                || !name
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b"-_.".contains(&b))
            {
                return Err(format!("invalid process name: {name}"));
            }
            let found = std::process::Command::new("pgrep")
                .args(["-x", name])
                .output()
                .map(|o| o.status.success())
                .unwrap_or(false);
            if found == negated {
                Err(format!(
                    "process '{name}' {} unexpectedly",
                    if negated {
                        "is running"
                    } else {
                        "is not running"
                    }
                ))
            } else {
                Ok(())
            }
        }
    }
}

/// A probe authored by the model and persisted as JSON.
pub struct GeneratedProbe {
    spec: ProbeSpec,
}

impl GeneratedProbe {
    pub fn spec(&self) -> &ProbeSpec {
        &self.spec
    }
}

/// Build a probe from an already-validated spec.
pub fn load_probe(spec: ProbeSpec) -> GeneratedProbe {
    GeneratedProbe { spec }
}

/// Build a probe from a spec and execute it immediately.
pub fn run_probe_spec(spec: ProbeSpec) -> ProbeResult {
    load_probe(spec).run()
}

impl Probe for GeneratedProbe {
    fn id(&self) -> &str {
        &self.spec.id
    }
    fn category(&self) -> &str {
        GENERATED_CATEGORY
    }
    fn severity(&self) -> Severity {
        self.spec.severity.clone()
    }
    fn title(&self) -> &str {
        &self.spec.title
    }
    fn description(&self) -> &str {
        &self.spec.hypothesis
    }
    fn run(&self) -> ProbeResult {
        let start = Instant::now();
        let mut failures: Vec<String> = Vec::new();
        for check in &self.spec.checks {
            if let Err(why) = run_check(check) {
                failures.push(why);
            }
        }
        let (mitigated, description, payload) = if failures.is_empty() {
            (true, self.spec.hypothesis.clone(), None)
        } else {
            let evidence = failures.join(" | ");
            (
                false,
                format!("{} — {}", self.spec.hypothesis, evidence),
                Some(
                    serde_json::json!({"hypothesis": self.spec.hypothesis, "failures": failures})
                        .to_string(),
                ),
            )
        };
        ProbeResult {
            probe: self.spec.id.clone(),
            category: GENERATED_CATEGORY.to_string(),
            mitigated,
            severity: self.severity(),
            title: self.spec.title.clone(),
            description,
            payload,
            timestamp_ms: now_ms(),
            duration_us: elapsed_us(start),
        }
    }
}

/// Validate a spec without executing it. Rejects anything that could not be
/// produced honestly: bad ids, empty checks, out-of-jail paths, non-loopback
/// URLs, oversized frames.
pub fn validate_spec(spec: &ProbeSpec) -> Result<()> {
    if !spec.id.starts_with("gen.")
        || spec.id.len() > 48
        || !spec.id[4..]
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_')
        || spec.id.len() < 8
    {
        bail!("id must match gen.[a-z0-9_] (3-44 chars after the prefix)");
    }
    if spec.title.trim().is_empty() || spec.title.len() > 160 {
        bail!("title must be 1-160 chars");
    }
    if spec.hypothesis.trim().is_empty() || spec.hypothesis.len() > MAX_TEXT_LEN {
        bail!("hypothesis must be 1-{MAX_TEXT_LEN} chars");
    }
    if spec.checks.is_empty() || spec.checks.len() > MAX_CHECKS {
        bail!("checks must contain 1-{MAX_CHECKS} entries");
    }
    for check in &spec.checks {
        match check {
            Check::FileAbsent { path }
            | Check::FilePresent { path }
            | Check::FileContains { path, .. }
            | Check::FileNotContains { path, .. } => {
                let _ = jail_path(path)?;
            }
            Check::FileModeMax { path, mode } => {
                let _ = jail_path(path)?;
                let _ = parse_mode(mode)?;
            }
            Check::HttpStatus { url, .. } | Check::HttpNotStatus { url, .. } => {
                let _ = parse_loopback_url(url)?;
            }
            Check::PortClosed { addr } => {
                let (host, port) = addr
                    .rsplit_once(':')
                    .with_context(|| format!("addr must be host:port: {addr}"))?;
                if !host_is_loopback(host.trim_start_matches('[').trim_end_matches(']')) {
                    bail!("port checks may only target loopback: {addr}");
                }
                port.parse::<u16>()
                    .with_context(|| format!("invalid port in {addr}"))?;
            }
            Check::SocketRejects { path, frame } => {
                let _ = jail_path(path)?;
                if frame.is_empty() || frame.len() > MAX_FRAME_BYTES {
                    bail!("socket frame must be 1-{MAX_FRAME_BYTES} bytes");
                }
                if frame.contains('\n') {
                    bail!("socket frame must be a single line");
                }
            }
            Check::ProcessPresent { name } | Check::ProcessAbsent { name } => {
                if name.is_empty() || name.len() > 64 {
                    bail!("invalid process name: {name}");
                }
            }
        }
    }
    Ok(())
}

/// Directory holding generated probe specs.
pub fn probes_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os("BADAPPLE_RED_TEAM_DIR") {
        return PathBuf::from(dir).join("probes");
    }
    let home = std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    home.join(".bad_apple").join("redteam").join("probes")
}

/// Hash-chained findings log written by the generation cycle.
pub fn findings_log_path() -> PathBuf {
    if let Some(dir) = std::env::var_os("BADAPPLE_RED_TEAM_DIR") {
        return PathBuf::from(dir).join("findings.jsonl");
    }
    let home = std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    home.join(".bad_apple")
        .join("redteam")
        .join("findings.jsonl")
}

/// Path for a spec id inside the probes dir (`gen.foo` -> `gen_foo.json`).
pub fn spec_path(id: &str) -> PathBuf {
    probes_dir().join(format!("{}.json", id.replace('.', "_")))
}

/// Load every valid generated probe from the probes dir.
/// Invalid specs are skipped — a corrupt hand-authored file degrades to a
/// warning instead of breaking the whole harness.
pub fn load_registry() -> Vec<Box<dyn Probe>> {
    let dir = probes_dir();
    let mut out: Vec<Box<dyn Probe>> = Vec::new();
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return out;
    };
    let mut paths: Vec<PathBuf> = entries
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().map(|x| x == "json").unwrap_or(false))
        .collect();
    paths.sort();
    for path in paths.into_iter().take(MAX_GENERATED_PROBES) {
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        match serde_json::from_str::<ProbeSpec>(&text) {
            Ok(spec) => {
                if validate_spec(&spec).is_ok() {
                    out.push(Box::new(GeneratedProbe { spec }));
                } else {
                    tracing::warn!("skipping invalid generated probe {}", path.display());
                }
            }
            Err(e) => {
                tracing::warn!(
                    "skipping unparsable generated probe {}: {e}",
                    path.display()
                );
            }
        }
    }
    out
}

/// Persist a validated spec atomically with 0600 permissions, evicting the
/// oldest specs past the cap. Returns the spec path.
pub fn persist_spec(spec: &ProbeSpec) -> Result<PathBuf> {
    validate_spec(spec)?;
    let dir = probes_dir();
    std::fs::create_dir_all(&dir)?;
    let path = spec_path(&spec.id);
    let tmp = dir.join(format!(".{}.tmp", spec.id.replace('.', "_")));
    let text = serde_json::to_string_pretty(spec)?;
    std::fs::write(&tmp, text)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(&tmp, std::fs::Permissions::from_mode(0o600));
    }
    std::fs::rename(&tmp, &path)?;
    evict_oldest(&dir);
    Ok(path)
}

fn evict_oldest(dir: &Path) {
    let mut entries: Vec<(PathBuf, std::time::SystemTime)> = std::fs::read_dir(dir)
        .map(|rd| {
            rd.filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| p.extension().map(|x| x == "json").unwrap_or(false))
                .filter_map(|p| {
                    std::fs::metadata(&p)
                        .and_then(|m| m.modified())
                        .ok()
                        .map(|t| (p, t))
                })
                .collect()
        })
        .unwrap_or_default();
    entries.sort_by_key(|(_, t)| *t);
    while entries.len() > MAX_GENERATED_PROBES {
        if let Some((oldest, _)) = entries.first().cloned() {
            let _ = std::fs::remove_file(&oldest);
        }
        entries.remove(0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec_json(checks: &str) -> String {
        format!(
            r#"{{"id":"gen.test_probe","title":"t","hypothesis":"h","severity":"medium","checks":[{checks}]}}"#
        )
    }

    #[test]
    fn valid_spec_loads() {
        let spec: ProbeSpec = serde_json::from_str(&spec_json(
            r#"{"kind":"file_absent","path":"/tmp/definitely_not_here_xyz"}"#,
        ))
        .unwrap();
        validate_spec(&spec).unwrap();
    }

    #[test]
    fn rejects_relative_and_traversal_paths() {
        for path in ["relative/path", "/tmp/../etc/passwd"] {
            let spec: ProbeSpec = serde_json::from_str(&spec_json(&format!(
                r#"{{"kind":"file_absent","path":"{path}"}}"#
            )))
            .unwrap();
            assert!(validate_spec(&spec).is_err(), "should reject {path}");
        }
    }

    #[test]
    fn rejects_out_of_jail_paths() {
        let spec: ProbeSpec = serde_json::from_str(&spec_json(
            r#"{"kind":"file_contains","path":"/etc/master.passwd","pattern":"root"}"#,
        ))
        .unwrap();
        assert!(validate_spec(&spec).is_err());
    }

    #[test]
    fn rejects_non_loopback_http() {
        for url in [
            "http://example.com/",
            "http://8.8.8.8/x",
            "https://127.0.0.1/x",
        ] {
            let spec: ProbeSpec = serde_json::from_str(&spec_json(&format!(
                r#"{{"kind":"http_status","url":"{url}","status":200}}"#
            )))
            .unwrap();
            assert!(validate_spec(&spec).is_err(), "should reject {url}");
        }
    }

    #[test]
    fn accepts_loopback_http() {
        let spec: ProbeSpec = serde_json::from_str(&spec_json(
            r#"{"kind":"http_not_status","url":"http://127.0.0.1:8787/api/status","status":200}"#,
        ))
        .unwrap();
        validate_spec(&spec).unwrap();
    }

    #[test]
    fn rejects_bad_ids() {
        for id in ["evil.probe", "gen.A", "gen.", "gen.probe with space"] {
            let spec: ProbeSpec = serde_json::from_str(&format!(
                r#"{{"id":"{id}","title":"t","hypothesis":"h","severity":"low","checks":[{{"kind":"file_absent","path":"/tmp/nope_xyz"}}]}}"#
            ))
            .unwrap();
            assert!(validate_spec(&spec).is_err(), "should reject id {id}");
        }
    }

    #[test]
    fn mode_parsing() {
        assert_eq!(parse_mode("0600").unwrap(), 0o600);
        assert_eq!(parse_mode("644").unwrap(), 0o644);
        assert!(parse_mode("8888").is_err());
        assert!(parse_mode("abc").is_err());
    }

    #[test]
    fn file_mode_check_detects_open_perms() {
        let dir = Path::new("/tmp").join(format!("ba_rt_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let f = dir.join("open_secret");
        std::fs::write(&f, "x").unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&f, std::fs::Permissions::from_mode(0o644)).unwrap();
        }
        let check = Check::FileModeMax {
            path: f.to_string_lossy().to_string(),
            mode: "0600".to_string(),
        };
        assert!(run_check(&check).is_err());
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&f, std::fs::Permissions::from_mode(0o600)).unwrap();
        }
        assert!(run_check(&check).is_ok());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn unknown_fields_rejected() {
        let json = r#"{"id":"gen.x_ok","title":"t","hypothesis":"h","severity":"low","checks":[],"evil":true}"#;
        assert!(serde_json::from_str::<ProbeSpec>(json).is_err());
    }
}
