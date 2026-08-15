# Security Policy

Bad Apple is a local-first, sovereign AGI research runtime. It runs entirely on the user's machine and does not require cloud services. This document outlines the security model, boundaries, and known limitations.

## Threat model

### In scope

- **Unauthorized access to the local HTTP API.** A malicious local process could call `/tools/run`, `/skills/learn`, `/pursuits/add`, or `/identity`.
- **Tool sandbox escape.** A learned or injected Python tool could attempt filesystem, network, or shell access.
- **Data exfiltration.** The runtime must not send source code, embeddings, or state to remote services.
- **Resource exhaustion.** Unbounded memory, disk, or CPU usage from runaway training, planning, or tool execution.
- **Swarm packet forgery.** A peer on the local network could send unsigned or malicious engrams.
- **Dylib tampering.** A compromised `libbad_apple.dylib` or Apple Intelligence bridge could crash or mislead the runtime.

### Out of scope

- Physical access to the machine.
- Compromise of the operating system or Apple Intelligence frameworks themselves.
- Attacks on the host compiler toolchain (`cargo`, `rustc`) beyond what the tool sandbox can control.

## Mitigations

| Threat | Mitigation |
|--------|------------|
| Unauthorized HTTP access | The telemetry server binds to `127.0.0.1:8080` by default; no remote interface is exposed. |
| Tool sandbox escape | The Python tool runner uses a restricted module list; `wild_workspace` operates read-only and cannot touch the network. |
| Data exfiltration | No cloud APIs are called; embeddings and `state.*` files remain on disk in the runtime directory. |
| Resource exhaustion | The `DualProcessGovernor` scales tick rate and regularization from thermal state; `LockFreeRing` drops lowest-priority engrams above 85% occupancy. |
| Swarm forgery | Every inbound TCP/UDP/WebSocket engram is HMAC-signed and checked against the active-goal 2048-D cosine firewall. |
| Dylib tampering | The C FFI bridge validates null, alignment, and bounds on every call; release builds are `lto` + `panic = "abort"`. |
| Secret leakage | No API keys or credentials are present in source. |

## Data flow

```
Sensors / curriculum / wild_workspace
                │
                ▼
       Bad Apple (same process)
                │
    ┌───────────┼───────────┐
    ▼           ▼           ▼
 tensor_brain  connectome  strategy_db
 (local)       (mmap)      (Sled, local)
    │
    ▼
 HTTP (localhost) / swarm (signed engrams) / FFI dylib (optional)
```

No data leaves the machine unless the user explicitly configures a swarm peer on another host.

## Reporting

Because this is pre-acquisition research code, security issues should be reported directly to the author at the repository contact email. Do not open public issues for undisclosed vulnerabilities.

## Known limitations

- The runtime runs as the user who starts it and has the same filesystem permissions. It does not use macOS sandbox entitlements.
- Learned Python tools run under a soft module restriction, not a full kernel sandbox such as `sandbox-exec` or a container.
- The HTTP API has no authentication. Any local process on the machine can access `http://127.0.0.1:8080` while the runtime is running.
- The 10,000-entry identity journal and `state.*` files are world-readable by default if the user's home directory is not restricted.
