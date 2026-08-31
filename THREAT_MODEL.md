# Bad Apple — Threat Model

## System Overview

Bad Apple is a baremetal AI OS layer for Apple Silicon macOS. It runs as three system launchd daemons (gatekeeper, MLX server, health supervisor) with root privileges, communicating via SLICKS-authenticated Unix domain sockets.

## Trust Boundaries

### Boundary 1: Local User → Gatekeeper
- **Trust level**: Semi-trusted. The user has valid SLICKS credentials but may send malformed or adversarial inputs.
- **Attack surface**: SLICKS handshake, prompt content, tool invocation requests.
- **Defense**: SLICKS v1 (HMAC-SHA256) and v2 (Secure Enclave ECDSA) authentication, nonce validation, timestamp freshness, replay cache, request validation (prompt size, token limits).

### Boundary 2: Gatekeeper → MLX Daemon
- **Trust level**: Trusted (both run as root on the same machine).
- **Attack surface**: Internal Unix socket, v2 proxy forwarding.
- **Defense**: Socket permissions 0o660, SLICKS end-to-end for v2, request validation in proxy path.

### Boundary 3: MLX Daemon → Local Filesystem
- **Trust level**: Semi-trusted. User-provided paths and tool arguments.
- **Attack surface**: File read/write tools, AppleScript execution, shell commands, document indexing.
- **Defense**: Fail-closed filesystem cage with openat-based operations, O_NOFOLLOW on all file creation, path jailing in Python tools, AppleScript escaping, shell allowlist (no interpreters), human-in-the-loop approval policy.

### Boundary 4: MLX Daemon → WebAssembly Sandbox
- **Trust level**: Untrusted. Model-generated code executed in sandbox.
- **Attack surface**: WASM module compilation and execution, host function ABI.
- **Defense**: Fuel metering, StoreLimits (memory cap, instance limit), output vector cap (256 KiB), input validation (negative lengths rejected), memory policy enforcement (max 1 memory, max pages).

### Boundary 5: P2P Mesh → Local Daemon
- **Trust level**: Untrusted. Remote peers on link-local network.
- **Attack surface**: P2P protocol frames, adapter transfer, model sync.
- **Defense**: AES-256-GCM encryption, HMAC-SHA256 or Secure Enclave ECDSA signing, replay protection (nonce window), timestamp freshness, peer spec SSRF prevention (cloud metadata blocked), zip slip prevention (path validation on extract), frame size limits (MAX_FRAME_BYTES on TCP and WebSocket).

### Boundary 6: Dashboard → Local System
- **Trust level**: Semi-trusted. Local browser on 127.0.0.1.
- **Attack surface**: HTTP endpoints, POST requests.
- **Defense**: CSRF token with constant-time comparison, Origin/Referer validation, 1 MB body size limit, no CORS headers, 127.0.0.1 only binding.

### Boundary 7: Aqua Helper → UI Automation
- **Trust level**: Semi-trusted. Local daemon requesting UI actions.
- **Attack surface**: Accessibility actions (click, type, focus), screen capture, Shortcuts.
- **Defense**: SLICKS v1 HMAC authentication, timestamp freshness, nonce replay protection, AppleScript escaping in all UI actions, O_NOFOLLOW on response files, socket permissions 0o600.

## Threat Agents

### Agent 1: Local Unprivileged User
- **Motivation**: Privilege escalation, data exfiltration.
- **Capabilities**: Can create files, symlinks, set environment variables for their own processes, connect to Unix sockets.
- **Mitigations**: Socket permissions 0o660, SLICKS authentication required, openat cage prevents symlink-based escape, replay cache prevents frame replay.

### Agent 2: Malicious Model Output
- **Motivation**: Execute arbitrary code, exfiltrate data.
- **Capabilities**: Can emit tool calls (shell, AppleScript, file write), can generate text with secrets.
- **Mitigations**: Human-in-the-loop approval for destructive tools, WASM sandbox for code synthesis, output firewall with Aho-Corasick secret redaction, shell allowlist excludes interpreters, AppleScript escaping in all integrations, autopilot off by default.

### Agent 3: Network Attacker (P2P)
- **Motivation**: Inject malicious engrams, steal model weights, impersonate peers.
- **Capabilities**: Can observe and inject traffic on link-local network.
- **Mitigations**: P2P off by default, AES-256-GCM encryption, HMAC/ECDSA signing, replay protection, frame size limits, SSRF prevention, zip slip prevention.

### Agent 4: Physical Access Attacker
- **Motivation**: Read conversation history, extract model, tamper with audit log.
- **Capabilities**: Can read files accessible to their user account.
- **Mitigations**: Audit ledger uses HMAC with SLICKS-derived secret, conversation file in /var/lib/bad_apple (0o770), Secure Enclave-signed checkpoints, model pinned to commit hash with integrity verification.

## Not Protected Against

- **Root-level attacker**: An attacker with root can modify any file, intercept any socket, and bypass all protections. Bad Apple does not defend against a compromised root account.
- **Firmware/hardware attack**: Secure Enclave provides key storage but cannot prevent hardware-level attacks (e.g., JTAG, chip-off).
- **Side-channel attacks**: Timing attacks on non-cryptographic operations are not fully mitigated. Constant-time comparison is used for HMAC verification but not for all comparisons.
- **Memory corruption in unsafe code**: The 114 unsafe blocks are individually commented but not formally verified. A memory safety bug in unsafe code could bypass all protections.
- **Supply chain attacks**: Dependencies are audited with cargo-audit but transitive dependency poisoning is not fully mitigated.

## Security Verification

| Method | Coverage | Results |
|---|---|---|
| Manual audit pass 1 | All 4 layers | 30 bugs found, all fixed |
| Manual audit pass 2 (red team) | All 4 layers | 39 bugs found, all fixed |
| cargo-fuzz (IPC) | SLICKS frame parsing | 127M iterations, 0 crashes |
| cargo-fuzz (WASM) | Compilation + execution | 638K iterations, 0 crashes |
| cargo-fuzz (Protocol) | P2P frame parsing | 694K iterations, 0 crashes |
| cargo-fuzz (Scavenger) | Path handling | 760K iterations, 0 crashes |
| Regression tests | Security boundaries | 78 Rust + 46 Python tests |
| Install test | Distribution integrity | 32 checks |
| cargo audit | Dependency vulnerabilities | 0 vulnerabilities |
