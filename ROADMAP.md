# Bad Apple — Cognitive Bare-Metal AI OS Roadmap

> **Goal:** make Bad Apple the reference implementation for a sovereign, on-device, bare-metal cognitive operating system — one that proves personal AGI can be more private, more capable, and more personally aligned than anything rented from the cloud.

This is a living list. Items are grouped by theme and roughly ordered from near-term natural extensions to longer-term research bets. Items marked ✅ have shipped in the current tree.

---

## 1. Sovereignty & Identity

| Feature | Why it matters |
|---|---|
| **Sovereign Identity Token (SLICKS 2.0)** ✅ | Cryptographic proof that the assistant, the user, and every action all originated on this machine. Bind the assistant to the user's biometric / device key so she is *yours*, not a commodity model. |
| **On-device key ceremony** ✅ | Generate and store all secrets (SLICKS, ledger, vault) in the Secure Enclave or SEP-backed keychain. No key material in userland. |
| **Air-gapped boot & model provenance** | Verify every downloaded model with signed hashes and a local attestation chain. Know exactly what code is running and that it never phoned home. |
| **Self-hosting model registry** | Cache, compress, and version every model locally. One-click import of MLX-compatible weights from Hugging Face without leaving the Mac. |

## 2. Memory & Cognition

| Feature | Why it matters |
|---|---|
| **Long-horizon episodic memory** | A structured, lifetime memory graph: what the user said, when, where, how they felt. Not just RAG chunks, but *experiences* the assistant can reason over. |
| **Procedural memory / skills** ✅ | Learn workflows from demonstrations ("every morning, do X") and replay them as compound tool plans — shipped as standing orders + the scheduler. |
| **Working memory dashboard** ✅ | A visible scratchpad the model can read/write to during a turn — shipped as the native task board (⌘T). |
| **Dream / offline consolidation** ✅ | While idle, consolidate conversation, memory, and ledger into summaries and updated beliefs without the user asking — shipped as the nightly dream cycle with eval-gated LoRA adoption. |
| **Personal fine-tuning on device** ✅ | LoRA-style adapters trained on the user's own data, locally — shipped via `badapple-lora` + the nightly dream gate. |

## 3. Sensory & Multimodal

| Feature | Why it matters |
|---|---|
| **On-device vision** ✅ | MLX vision models (e.g., Qwen2-VL, Florence) to describe screen captures, parse UI, read documents, and act on what she sees — shipped as the ocular organ. |
| **Audio understanding** | Local speech-to-text with diarization, plus audio event detection (knocks, alarms, baby cries). Partially shipped: ambient ears + meeting capture use on-device `SFSpeechRecognizer`; the `BadAppleASR` seam is whisper-ready for an MLX upgrade. |
| **Video / screen comprehension** | Real-time or recorded screen understanding for "what just happened?" and hands-free workflow help. |
| **Local image generation** | Diffusion on Apple Silicon for quick sketches, diagrams, and avatars — no cloud credits. |

## 4. OS Integration & Automation

| Feature | Why it matters |
|---|---|
| **System-level Shortcuts integration** ✅ | Trigger and compose macOS Shortcuts as first-class tools — shipped via `run_shortcut`. |
| **Persistent workspace / project mode** ✅ | Attach to a directory or project; the assistant knows the files, git state, build system — shipped via the workspace watcher + grounded code index. |
| **Local email / calendar / reminders** ✅ | Read and act on the user's local accounts with explicit approval — shipped via the Aqua bridge tools. |
| **Focused application automation** | Inspect and drive the active app via macOS Accessibility APIs + AppleScript. Partially shipped: `run_applescript` covers the scripting surface; Accessibility-API inspection is open. |
| **Clipboard & drag-and-drop agent** | The assistant can watch the clipboard, accept dragged files, and act on them. Partially shipped: clipboard recall (`recall_clipboard`) is live; drag-and-drop ingestion is open. |
| **Local browser agent** | Drive a headless WebKit/Safari instance for tasks the user approves, all on device. |

## 5. Security, Safety & Governance

| Feature | Why it matters |
|---|---|
| **Policy language for the cage** ✅ | A declarable DSL (YAML or natural language) for what the assistant is allowed to do, read, write, or execute — shipped as `policy.yaml`. |
| **Behavioral permissions per persona** | A "drill" persona should not be allowed to run shell commands; a "researcher" persona should. |
| **Tamper-evident ledger 2.0** ✅ | Signed, hash-chained, and timestamped by a local hardware key — shipped as the sovereign ledger layer with Secure Enclave checkpoints. |
| **Private mode / guest mode** ✅ | One-shot sessions that leave no memory, no cache, and no ledger — shipped as private mode. |
| **Adversarial output checks** | Beyond the blocklist: a second-pass classifier that catches jailbreaks, exfiltration, and hallucinated secrets. |
| **User override & kill switch** ✅ | A global hotkey or voice command that immediately stops the model, TTS, and any pending actions — shipped (`⌘⇧⎋` interrupts speech; the kill switch halts actions). |

## 6. Developer & Extensibility Platform

| Feature | Why it matters |
|---|---|
| **Local MCP server** ✅ | Expose Bad Apple's tools and memory to local clients via the Model Context Protocol — shipped (`badapple-mcp`, `/var/run/badapple/mcp.sock`). |
| **A2A / agent-agent protocol** | Let multiple local agents (Bad Apple, Blade, EdgeOS) discover and delegate to each other without cloud brokers. |
| **Plugin / tool registry** ✅ | Install local tools from signed bundles — shipped as `PluginRegistry` with signed manifests. |
| **Local API with SLICKS** ✅ | Give other apps on the Mac a safe, authenticated way to ask Bad Apple for inference, memory, or tool execution — shipped via the SLICKS Unix socket. |
| **Observability dashboard** ✅ | Latency, token throughput, memory, cache hit rate, audit log — shipped as `badapple-dashboard` Control Center on 127.0.0.1:8787. |

## 7. Collaboration & Sync

| Feature | Why it matters |
|---|---|
| **Encrypted peer-to-peer sync** ✅ | Sync memory, persona, and model manifests between a user's devices over AES-256-GCM encrypted mesh — shipped (opt-in); plus layer-range brain sharding across peer Macs (`MESH_BRAIN.md`). |
| **Family / team rings** | Allow multiple devices in the same household to share a private knowledge base while each keeps its own identity. |
| **Offline-first everything** | The assistant degrades gracefully to smaller local models or cached plans when disconnected, rather than stopping. |

## 8. Performance & Efficiency

| Feature | Why it matters |
|---|---|
| **Dynamic model tiering** | Route simple queries to a tiny on-device model, reasoning to the 9B, and heavy lifting to an optional local quant, all automatically. |
| **Memory-pressure-aware inference** ✅ | Purge, quantize, or page-out draft caches when other apps need RAM — shipped via `MemoryGovernor`. |
| **Battery / thermal scheduling** | Defer heavy training or indexing until plugged in and cool. |
| **Speculative + medusa + look-ahead decoding** | Push token throughput closer to what cloud GPUs deliver. Partially shipped: external-draft speculative decoding is armed in the daemon plist, and the reusable prompt-prefix KV cache cut measured turn wall time ~47%. Medusa/MTP heads are open. |
| **Streaming first-token preview** | Emit a low-confidence preview token immediately so the UI feels alive, then correct if the model changes its mind. |

## 9. Standards & Philosophy

| Feature | Why it matters |
|---|---|
| **The Bare-Metal AI OS Manifesto** ✅ | A public document defining what a sovereign cognitive OS promises — shipped as `MANIFESTO.md`. |
| **Reference architecture & protocol specs** ✅ | Open descriptions of SLICKS, the cage, the audit ledger, and the persona protocol — shipped (`docs/SLICKS_PROTOCOL.md`, `docs/LEDGER_FORMAT.md`, `ARCHITECTURE.md`). |
| **Certification / self-test suite** ✅ | Automated tests that prove a given build is air-gapped, does not call network, and redacts secrets — shipped as the 22-check `badapple cert` suite + `personal_agi_proof.sh`. |
| **Model & data portability** | Export your assistant's memory, personas, and fine-tuned adapters as a self-contained archive you can move to another machine. |

---

## What makes this a "standard" rather than a product

A standard is something others can build against. For Bad Apple to become the cognitive bare-metal AGI OS standard, it should expose:

1. **A well-defined local trust boundary** — the assistant, the user, and the hardware are one trust domain.
2. **An open tool and memory interface** — any local app can extend her capabilities.
3. **A portable identity and audit format** — your assistant's state is yours and provably so.
4. **Reference implementations** — the menu bar, the CLI, the daemon, and the cage serve as proof that the standard works.
5. **A public design rationale** — why local, why sovereign, why anti-cloud, and how to stay safe.

The cloud AI giants are building the exact opposite: multi-tenant, rented, surveilled, centralized. A bare-metal AGI OS standard is the counter-architecture.
