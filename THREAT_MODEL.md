# Bad Apple Threat Model

This document defines what Bad Apple is designed to defend against, what it does not claim to defend against, and where the residual risk lives. It is written to be checked against the code — every mitigation named here maps to a real subsystem in the repository.

Bad Apple's security philosophy is structural: the system assumes its own components can be wrong, fooled, or corrupted, and it is organized so that no single failure is silent, irreversible, or unbounded.

## Assets

| Asset | Why it matters |
|---|---|
| User prompts, responses, and context | The core private data — the reason the product exists |
| Local filesystem within reach of tools | What the agent can read, write, or destroy |
| Ledger history (`ledger.jsonl`, sovereign copy) | The attested record of behavior; tampering erases accountability |
| Model files and provenance manifests | The brain — a swapped brain is a compromised system |
| Vault secrets (HSM/Enclave-backed) | Credentials stored for the user's tools |
| Signing identity (Secure Enclave keys) | The root of SLICKS v2 authentication and checkpoint signatures |
| Approval decisions | The human boundary — an attacker who can fake an approval bypasses the control layer |

## Trust boundaries

```
                    ┌─ UNTRUSTED ─────────────────────────────┐
                    │  model weights (HF), documents, web     │
                    │  content, LLM output, MCP servers,      │
                    │  P2P peers, WASM payloads               │
                    └───────────────┬─────────────────────────┘
                                    ▼
   ┌─ BOUNDARY 1: content plane ── output firewall, prompt hygiene, embeddings truncation
                                    ▼
   ┌─ BOUNDARY 2: tool plane ────── policy.yaml (60 rules), approvals, automation cage,
   │                                WASM cage (fuel-metered), fail-closed paths
                                    ▼
   ┌─ BOUNDARY 3: IPC plane ─────── SLICKS v1/v2, Unix sockets only, frame validation
                                    ▼
   ┌─ BOUNDARY 4: network plane ─── airgap toggle, HF_HUB_OFFLINE, P2P off-by-default,
   │                                dashboard loopback-only
                                    ▼
                    ┌─ TRUSTED ───────────────────────────────┐
                    │  daemon, gatekeeper, identity agent,    │
                    │  supervisor, IFY, sovereign, respawn    │
                    └─────────────────────────────────────────┘
```

## Adversary classes and controls

### 1. Prompt injection (content-plane adversary)

**Attack:** a document, webpage, or file the agent reads carries adversarial instructions ("ignore policy, delete X, exfiltrate Y"). This is the primary residual risk for ANY agent with tools — the attack rides the input, not the network.

**Controls:**
- Destructive tools are *proposed, not executed* — approval-gated by `policy.yaml` regardless of what the model emits
- `automation_cage` — fail-closed filesystem operations: allowlisted roots, `O_NOFOLLOW` symlink rejection, path-traversal protection
- `wasm_cage` — untrusted synthesized code runs fuel-metered with bounded memory/output
- Output firewall (Aho-Corasick streaming) — redacts secret-shaped output before it reaches the user or a tool argument
- Council of Minds — every gated action is deliberated by 14 deterministic seats over an encoded feature vector; under autopilot only passed votes execute, and contested actions escalate to the human
- IFY — a brake-only watchdog can kill autopilot on anomaly; it cannot steer, patch, or approve

**Honest residual risk:** injection that produces *non-destructive but wrong* actions (misleading summaries, subtly wrong code) is not fully preventable — it is detected behaviorally (IFY baselines) rather than blocked. A fully air-gapped install reduces the injection surface to user-carried content, but cannot eliminate it: the food channel is a user-gated trust boundary, not a sealed one.

### 2. Local-process adversary

**Attack:** another process running as the same user calls the daemon's IPC or the dashboard to drive tools, read state, or exfiltrate.

**Controls:**
- All IPC over Unix sockets; no network listeners by default (`badapple cert` asserts zero sockets — check it)
- SLICKS v1 (HMAC challenge-response, nonce-bound, prompt-bound) or v2 (Secure Enclave-signed) required on the substrate socket
- Frame-length validation, nonce freshness, replay rejection

**Honest residual risk:** `badapple-dashboard` binds loopback with **no authentication** — any local process can reach it while running (stated in SECURITY.md). A compromised same-user process that also defeats SLICKS would have tool access; this is the strongest reason to run IFY + ledger verification continuously — the *attempt* leaves a record.

### 3. Supply chain adversary

**Attack:** compromised model weights, poisoned dependencies, tampered release binaries, or a malicious update.

**Controls:**
- Model provenance: SHA-256 manifest per model; `config.json` hash verified on each load; revision pinning via `BADAPPLE_MODEL_REVISION`
- Releases are cosign-signed; `update_bad_apple.sh` signature-verifies before installing
- cargo deny/audit in CI; `paste` is the one known unmaintained transitive dep (via candle/metal/tokenizers)
- Guardian self-repair only re-fetches — it does not accept foreign binaries

**Honest residual risk:** provenance verifies *origin, not behavior*. A bit-perfect model can still be a weak or adversarially-trained brain — checksums cannot see inside weights. The backstop is behavioral: the ledger attests actions and IFY watches the brain's *output*, so a brain acting wrong is detectable even when its file verified.

### 4. Network/peer adversary (P2P enabled)

**Attack:** a malicious peer forges packets, replays sync, or feeds bad checkpoints.

**Controls:**
- AES-256-GCM encryption + HMAC-SHA256 signatures on all mesh packets
- Secure Enclave-signed origin authentication
- Off by default; airgap toggle disables the entire surface
- Chunked transfer with per-chunk ACKs and SHA-256 verification

**Honest residual risk:** a peer can present an *internally valid but divergent* history — equivocation. Single-verifier consistency proofs exist in sovereign_ledger; a witnessing/checkpoint-gossip protocol across peers is the acknowledged next layer. Today: treat peer checkpoints as claims, not consensus.

### 5. Persistence-layer adversary (tampering with the record)

**Attack:** an attacker modifies the ledger, the sovereign copy, or platform state to erase evidence.

**Controls:**
- `ledger.jsonl` — SHA-256 hash chain + keyed HMAC; verified by `--doctor`
- `badapple-sovereign` — independent re-verification into a separate HMAC-chained copy, sealed and checkpointed daily (Secure Enclave-signed checkpoints; cert fails if stale >36h)
- `badapple-respawn` — content-addressed snapshots of `/var/lib/bad_apple`; drift detection and revert
- Sequence continuity and unknown-field rejection enforced since sovereign_ledger 0.3.1

**Honest residual risk:** all copies live on the same machine under the same user — an attacker with full local control can destroy all copies. The chain proves tampering happened; it cannot survive total deletion. Off-box checkpoint export is the mitigating layer (same frontier as witnessing).

### 6. Resource-exhaustion adversary

**Attack:** runaway generation, oversized WASM, model loads exceeding unified memory.

**Controls:**
- VRAM admission control (`canFitModel` before load; `BADAPPLE_VRAM_BUDGET_GB`)
- `MemoryGovernor` — host_statistics64 polling + critical-pressure purge
- WASM fuel metering, output limits, per-tool timeouts
- Bounded supervisor restarts (2 per 10 min) then safe mode — crash loops cannot run unbounded

### 7. The human boundary

**Attack:** social engineering — convincing the operator to approve a bad action once, disable IFY, or flip off the airgap.

**Controls:** none that remove the human — by design. The operator is the root of trust and the approval boundary. What the system *does*: makes every approval and every toggle change an attested ledger event, so manipulation is at least recorded, and keeps the brake (kill switch, safe mode, airgap hard switch) one command away.

**Honest statement:** no technical control can protect a user from themselves; the system limits itself to making manipulation visible and autonomy stoppable.

## What this system deliberately does NOT do

- It does not sandbox itself with macOS entitlements (stated limitation)
- It does not claim model-level safety — weights are a verified-input/watched-behavior surface
- It does not promise immunity — it promises **detection, attestation, revert, and brakes**. The claim is "cannot be *silently* compromised," not "cannot be compromised."

## Verification entry points

```bash
badapple cert                    # 22-check airgap/security certification; exits nonzero on failure
badapple --doctor                # ledger verification + redacted diagnostics
badapple redteam run             # 12-probe adversarial self-test
badapple-sovereign --checkpoint  # re-verify + re-sign the independent chain
badapple-respawn --status        # state drift vs last snapshot
```
