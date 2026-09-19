# Security Policy

Bad Apple is a local-first, sovereign AI runtime. It runs entirely on the user's machine and does not require cloud services after the first model download. This document outlines the security model, boundaries, and known limitations. The full adversary-by-adversary analysis lives in [THREAT_MODEL.md](THREAT_MODEL.md).

## Threat model

### In scope

- **Unauthorized access to the local dashboard API.** A malicious local process could call tools, read the audit log, or change runtime settings through `http://127.0.0.1:8787`.
- **Tool sandbox escape.** A learned or injected tool could attempt filesystem, shell, or network access outside its allowlist.
- **Data exfiltration.** The runtime must not send prompts, source code, embeddings, or state to remote services.
- **Resource exhaustion.** Unbounded memory, disk, or CPU usage from runaway generation, tool execution, or model loading.
- **P2P packet forgery.** A peer on the local network could send unsigned or malicious mesh packets.
- **Dylib / binary tampering.** A compromised `libBadAppleMLX.dylib`, `libbad_apple.dylib`, or app bundle could crash or mislead the runtime.
- **Model supply chain.** Poisoned, typosquatted, or adversarially-trained model weights arriving through the download/update path.
- **Prompt injection.** A document or webpage could inject instructions into the model context.

### Out of scope

- Physical access to the machine.
- Compromise of the operating system, compiler toolchain, or Apple Intelligence/MLX frameworks themselves.
- Network attacks on the host's internet connection outside of Bad Apple's control.

## Mitigations

| Threat | Mitigation |
|--------|------------|
| Unauthorized dashboard access | `badapple-dashboard` binds to `127.0.0.1:8787` by default; no remote interface is exposed. P2P and dashboard are off by default for air-gap certification. |
| Tool sandbox escape | The policy engine enforces per-tool allowlists, timeouts, output limits, and approval rules. Filesystem tools run through the `automation_cage` (`openat`/`O_NOFOLLOW`). Untrusted synthesis runs in the `wasm_cage` with bounded fuel, memory, and output. |
| Data exfiltration | `HF_HUB_OFFLINE=1` is set in the daemon plist. No cloud APIs are called during normal operation. P2P and MCP are off by default. |
| Resource exhaustion | The Swift `MemoryGovernor` polls macOS memory pressure and purges optional models. Rust UMA reservations and `policy.yaml` limits bound tool execution. The `wasm_cage` refuses oversized modules and halts on fuel exhaustion. |
| P2P packet forgery | Every mesh packet is HMAC-SHA256 signed and AES-256-GCM encrypted. Peer keys are derived from a local pre-shared key or authenticated exchange. |
| Dylib tampering | Release builds use `lto`, `codegen-units = 1`, and `panic = "abort"`. The Swift/Rust IPC validates frame lengths, signatures, and nonce freshness. Ad-hoc or Apple Developer ID signing is used depending on the build. |
| Model supply chain | SHA-256 provenance manifests per model; `config.json` hash verified on each load; `BADAPPLE_MODEL_REVISION` pins upstream commits. Provenance verifies origin, not behavior — IFY and the ledger watch what the model *does* after install. |
| Prompt injection | Workspace and document text is embedded and truncated. The output firewall scans generated text for secrets and redacts matches. The policy engine requires explicit user approval for destructive actions. |
| Secret leakage | The audit ledger redacts secrets, emails, SSNs, phones, API keys, and long random tokens before writing. No API keys are required for core operation. |

## Data flow

```
User prompt / menu bar / dashboard
                │
                ▼
    badapple CLI / gatekeeper
                │
                ▼
    badapple-engine (Swift MLX daemon)
                │
    ┌───────────┼───────────┐
    ▼           ▼           ▼
  MLX model   policy.yaml   audit ledger
  local       local         local
                │
                ▼
  Bad Apple tools / automation cage / wasm cage
                │
                ▼
  Filesystem / shell / Shortcuts / AppleScript
```

No data leaves the machine unless the user explicitly enables a P2P peer or downloads a model from Hugging Face.

## Reporting

Report vulnerabilities via GitHub private security advisory on the repository, or directly to the author through LinkedIn DM. Do not open public issues for undisclosed vulnerabilities. A response is targeted within 72 hours.

## Known limitations

- The runtime runs as the user who starts it and has the same filesystem permissions. It does not use macOS sandbox entitlements.
- The `badapple-dashboard` HTTP server has no authentication. Any local process on the machine can access `http://127.0.0.1:8787` while the dashboard is running.
- The consumer release is unsigned and not notarized by default. Users must strip Gatekeeper quarantine manually for direct-download zips, or install via the Homebrew Cask which does this in `postflight`.
- Model weights are downloaded once from Hugging Face at first use. The daemon is configured for offline mode after the initial cache is populated.
- The app is not yet tested on a clean-machine VM. A notarized DMG is planned but not the default artifact.
