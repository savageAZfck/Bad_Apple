# Bad Apple

> **A sovereign personal AGI organism for Apple Silicon — with two brains on
> two different blocks of silicon, senses, memory, and proof of everything
> it has ever done.**
>
> One brain on the GPU. One on the Neural Engine. A fast tier for reflexes,
> eyes on the screen, ears in the room, and a hash-chained ledger that can be
> verified by strangers without trusting a word this file says. After the
> models are cached, inference needs no network. Under full air-gap,
> `badapple cert` asserts the daemon holds **zero** network sockets.
>
> Don't trust this README. Verify it.

---

## The organism

Bad Apple is not a chat app and not a cloud wrapper. It is a persistent
organism made of `launchd` daemons, native Swift/Rust services, and a menu
bar — with the parts a creature needs, not just the parts a demo needs:

| Organ | Implementation |
|---|---|
| **Main brain** | Qwen2.5-Coder-7B, 4-bit, MLX, on the **GPU** (~16–23 tok/s warm) |
| **Second brain** | Qwen3-4B, int8, sharded stateful CoreML, on the **Neural Engine** (~7–10 tok/s at <1% CPU — 100% of runtime ops prefer ANE) |
| **Reflexes** | Qwen2.5-0.5B fast tier for simple queries — answers without waking a big brain |
| **Eyes** | Screen capture → Qwen2-VL-2B change-narration → `Saw:` percepts |
| **Ears** | Whisper large-v3 turbo ASR, ambient hearing (opt-in) |
| **Memory** | bge-small embeddings, RAG, semantic cache, learned facts, respawn snapshots |
| **Identity** | Secure Enclave signing; SLICKS v2 authenticated IPC |
| **Immune system** | IFY watchdog — learns baselines, files approval-gated findings; brakes, never steers |
| **Flight recorder** | `badapple-tape` — bounded forensic ring + signed incident bundles |
| **Conscience** | 14-seat Council of Minds votes on every gated action, votes journaled |
| **Governance** | Consent registry, certified erasure, subject vault, provenance manifest, constitutional amendments, signed oversight bulletins |
| **Hands** | Files, shell, AppleScript, Shortcuts, mail/calendar/reminders, local MCP tools — all under per-tool policy |

Both big brains run **at the same time**. Per request, the `brain` field on
the inference route selects `ane`, `gpu`, or `auto`; the daemon admits or
defers each brain based on live memory pressure, unloads under pressure, and
reloads on fault. Every routing decision lands on the ledger as a
`brain_route` event.

### Continuity of self

The organs above sum to a property most systems never achieve: she is a
**continuity-of-self layer**. Not "persistence" — files surviving a reboot —
but provable continuity of identity across substrate changes. The same keys
sign every epoch of her existence; the same hash-chained ledger is her spine;
every brain load, model swap, respawn, and delegated inference is attested
into that chain. Brains are slots, machines are hosts, mesh nodes are
fungible — the self is the chain, and the chain is verifiable offline by
anyone holding `cosign.pub`. A process is something she runs; a continuity
is what she *is*.

## The ANE brain — and the toolchain that made it

Every Apple Silicon Mac ships a Neural Engine that sits idle while the GPU
does all the work. Bad Apple puts a whole second brain on it.

`badapple-aneconvert` converts a Hugging Face safetensors checkpoint into
sharded, stateful, weight-only-int8 CoreML artifacts — **in pure Rust, with
no Python and no coremltools**. It writes Apple's model format (MIL) directly,
including stateful KV caches and runtime int8 dequantization ops, then
compiles each layer shard with `coremlc`. The graph targets the CoreML9
opset with fused `scaled_dot_product_attention` (GQA folded into the query
sequence axis) and fp16 SiLU — measured result: **every runtime operation
prefers the Neural Engine**, decode under 1% CPU, zero sockets.

```bash
badapple-aneconvert --model ~/models/qwen3-4b-src \
                    --out ane_artifacts/qwen3b_ane_shards_q8_sdpa \
                    --weight-bits 8 --seq-len 2048
cargo run --release --example ane_audit -- <layer.mlmodelc>   # placement census
cargo test --release --test ane_brain_perf                    # perf + mastery
```

Enable the second brain by dropping the manifest path into the control file
`~/.bad_apple/ane_brain` (or `BADAPPLE_ANE_MODEL` in the daemon plist). Per
request, the `brain` field on the inference route selects `ane`, `gpu`, or
`auto` — drive it directly with
`cargo run --release --example ane_route_probe -- ane`.

## See it prove itself

`badapple demo` is a narrated self-demonstration. Every line is a real call
against live subsystem state — nothing is scripted:

```text
  Bad Apple — self-demonstration
  ────────────────────────────

  Verifying my chain...        8726 attested actions, tip 378f8a4452869859…
  Checking my air gap...       16 checks, zero network sockets — clean
  Consulting my watchdog...    IFY is gestation — watching, brake-only
  Reading my vitals...         Secure Enclave signing — identity is hardware-bound
  Checking my sovereign seal... 8634 entries sealed · secure-enclave

  As far as I can prove: I am alone with your data.
  Don't trust me — verify me: `badapple cert` · `badapple receipts`
```

## The proof commands

| Command | What it does |
|---|---|
| `badapple demo` | Narrated self-demo. Every line is a real subsystem read. |
| `badapple receipts` | Proof card: attested-action count, organism age, chain tip, sovereign seal, watchdog phase, identity. |
| `badapple cert` | The certification suite — air-gap, ledger integrity, policy coverage, cages, crypto, mesh. Exits non-zero on any failure. |
| `badapple export-proof` | Self-contained verification bundle of the organism's attested history — publicly verifiable, **no secrets required**. |
| `badapple --doctor` | Diagnostics, binary checks, ledger hash-chain verification. |
| `badapple ify status` | Watchdog phase, baseline, findings. |
| `badapple tape status` | Flight-recorder ring: frames, head hash, incidents. |
| `badapple bulletin` | Signed oversight bulletin over new ledger entries. |

`export-proof` bundles the sovereign ledger (`ledger.sovereign.jsonl`), its
Secure Enclave-signed checkpoint, and verification instructions. The sealed
segments verify **without any key material** — using the published
[`sovereign_ledger`](https://crates.io/crates/sovereign_ledger) crate or the
zero-dependency JS verifier:

```sh
node verify.mjs ledger.sovereign.jsonl --public
# public verification passed: 8634 sealed entries in 1 segments, 1 unsealed
# anchor: secure-enclave
```

You can post your AI's receipts and let strangers check them. That is the point.

## Install

### Homebrew Cask (recommended)

```bash
brew tap savageAZfck/bad-apple https://github.com/savageAZfck/homebrew-bad-apple
brew install --cask bad-apple
```

### Direct download

See [bad-apple-releases](https://github.com/savageAZfck/bad-apple-releases)
for the signed-artifact zips. Every release ships both packages, checksums,
an SBOM, and cosign sigstore bundles — `update_bad_apple.sh` verifies the
signature against the pinned `cosign.pub` before installing. Unsigned builds
trip Gatekeeper; the bundled `strip_quarantine.sh` handles this locally.
**Notarization is on the roadmap** — see "Honest limits" below.

### From source (unsigned, no Apple Developer ID needed)

```bash
cargo build --release
BADAPPLE_NO_SIGN=1 src/platform/apple_desktop/build_bad_apple_menu_bar.sh
sudo src/platform/apple_desktop/strip_quarantine.sh
osascript -e 'do shell script "cd /path/to/bad_apple && src/platform/apple_bridge/install_badapple_platform.sh --install --unsigned-install" with administrator privileges'
```

**Hardware:** Apple Silicon (M1+). 8 GB unified memory minimum, 16 GB
recommended — both brains resident plus senses fits in 16. ~45 GB free disk
for models, ANE artifacts, and state.

## Architecture

```text
badapple CLI / menu bar / voice / dashboard (127.0.0.1:8787)
              │
              ▼
   /var/run/badapple/substrate.sock  (SLICKS v1/v2)
              │
              ▼
     gatekeeper (Rust, launchd)
     ├─ SLICKS v1/v2 auth + replay cache
     ├─ Candle semantic router
     ├─ Automation cage (openat, O_NOFOLLOW)
     └─ WASM sandbox (fuel-metered)
              │
              ▼
   badapple-engine  (Swift daemon — the organism's body)
   ├─ Brain GPU:  Qwen2.5-Coder-7B 4-bit via MLX
   │              + resident 0.5B fast tier
   │              + reusable prompt-prefix KV cache
   ├─ Brain ANE:  Qwen3-4B int8 via sharded stateful CoreML
   │              (fused SDPA, fp16 SiLU — 100% runtime-op ANE placement)
   ├─ Routing:    brain=gpu|ane|auto, memory admission, pressure
   │              unload, fault reload — all ledgered
   ├─ Senses:     eyes (screen → VLM narration) · ears (Whisper ASR)
   │              · clipboard · meeting capture · workspace watcher
   ├─ RAG, semantic cache, native embeddings
   ├─ Policy engine (80+ per-tool rules) + human-in-the-loop approvals
   ├─ Council of Minds — 14 deterministic seats vote on gated actions
   ├─ Agent loop with failure replanning + native task board (⌘T)
   ├─ Vigilance: watchers · standing orders · calendar lookahead ·
   │  sentinel (persistence/network perimeter diffing)
   ├─ Aqua bridge: mail · calendar · reminders · messages
   ├─ Governance: consent registry · certified erasure · subject vault ·
   │  provenance brain_manifest · constitutional amendments
   ├─ Streaming output firewall (Aho-Corasick secret redaction)
   ├─ Hash-chained audit ledger (policy_hash on every line)
   ├─ IFY watchdog (brake-only) · Curious autopilot (policy-gated)
   └─ Air-gap certification self-audit
              │
              ▼
   badapple-tts  (native AVSpeechSynthesizer)

Independent verification & oversight layers:
   badapple-sovereign → re-verifies ledger into sealed sovereign chain,
                        Secure Enclave-signed daily checkpoints
   badapple-respawn   → content-addressed snapshots of all state roots,
                        drift detection, revert-to-any-point
   badapple-tape      → forensic ring + signed incident bundles
   badapple-bulletin  → signed oversight artifacts + key-ceremony log
```

## The verification stack

| Layer | What it proves | How to check it |
|---|---|---|
| Audit ledger | Every action is hash-chained and secret-redacted | `badapple --doctor` |
| Sovereign ledger | Independent re-verified copy, sealed, publicly verifiable | `badapple export-proof` + `verify_public` |
| Secure Enclave checkpoints | Daily signed chain-tip + Merkle root | `badapple receipts` |
| Flight recorder | Pre-execution intent frames + incident bundles verify offline | `badapple tape verify <bundle>` |
| Oversight bulletins | Periodic signed summary of dispatches by authority class | `badapple-bulletin --verify <path>` |
| Provenance | Every model load/swap anchored to a signed brain_manifest | `/var/lib/bad_apple/provenance.jsonl` |
| IFY watchdog | Anomalies and drift surfaced as approval-gated findings | `badapple ify status` |
| Council | Every gated action carries a journaled 14-seat vote | `council_deliberation` events |
| Air-gap cert | Runtime checks incl. zero external sockets | `badapple cert` |
| Organism proof | End-to-end organ battery: senses, council, dream, vigilance | `./personal_agi_proof.sh` |
| Respawn | All state revertible to any snapshot | `badapple-respawn --status` |

Conformance is externally checkable:
[Touchstone](https://github.com/savageAZfck/touchstone) — the open
conformance battery for sovereign personal AGI organisms — scores Bad Apple
on its published scoreboard.

Companion open components:
[`sovereign_ledger`](https://github.com/savageAZfck/sovereign_ledger),
[`respawn`](https://github.com/savageAZfck/respawn),
[`edge_gate`](https://github.com/savageAZfck/edge_gate),
[`flight_tape`](https://github.com/savageAZfck/flight_tape).

## Security model — and honest limits

Read [THREAT_MODEL.md](THREAT_MODEL.md) and [SECURITY.md](SECURITY.md) first.
The short version:

- **Air-gap is a certified posture, not a magic property.** Downloads, P2P,
  and MCP are optional doors; `badapple cert` proves them shut when they
  should be. Nothing leaving also means nothing gets in — the same boundary
  that keeps your data home keeps remote control out.
- **Unsigned binaries.** Until notarization lands, first install shows a
  Gatekeeper warning. That is a real UX cost, stated plainly.
- **Local model ceiling.** The brains are 7B/4B-class open weights — not
  frontier cloud scale. The organism's answer is swappable brains and mesh
  federation: better open models drop in without touching memory, identity,
  or governance.
- **Prompt injection is a real residual.** The policy engine, cages, and
  firewall bound it; they do not eliminate it. See THREAT_MODEL.md §5.
- **Fail-closed by design.** Denied tools stay denied; approvals are
  explicit, logged, and deniable (`deny <id>`).
- **ANE artifacts are local builds.** The second brain's shards are
  generated on your machine from a checkpoint you supply — the converter is
  open, the artifacts are not shipped.

## Quick use

```bash
badapple "What is 2+2?"                          # reflex tier answers
badapple --speak "What do you think of Siri?"    # voice with native TTS
badapple "are you alone"                         # on-demand self-audit, in persona
badapple "approve <id>" / "deny <id>"            # human-in-the-loop
badapple --benchmark                             # performance suite
badapple "switch to wicket"                      # persona switch
badapple tape freeze                             # snapshot an incident bundle now
cargo run --release --example ane_route_probe -- ane   # probe the ANE route
```

## Build & test

```bash
cargo build --release
cargo fmt --check
cargo test --release                            # unit + integration suites (tests/)
cargo test --release --test cert_suite
cargo test --release --test ane_brain_perf      # ANE brain perf + mastery
badapple cert                                   # live air-gap certification
```

Swift engine & app bundle:

```bash
BADAPPLE_NO_SIGN=1 src/platform/apple_desktop/build_bad_apple_menu_bar.sh
```

## Documentation

- [THREAT_MODEL.md](THREAT_MODEL.md) — adversary classes, defenses, residuals
- [SECURITY.md](SECURITY.md) — security posture and reporting
- [ARCHITECTURE.md](ARCHITECTURE.md) — component topology and invariants
- [docs/LEDGER_FORMAT.md](docs/LEDGER_FORMAT.md) — primary ledger format spec
- [docs/SLICKS_PROTOCOL.md](docs/SLICKS_PROTOCOL.md) — IPC protocol spec
- [BAD_APPLE.md](BAD_APPLE.md) — technical deep dive and live benchmarks
- [AGENTS.md](AGENTS.md) — build commands and project conventions
- [IFY.md](IFY.md) — watchdog design spec
- [CHANGELOG.md](CHANGELOG.md) — development history

## License

Source-available proprietary license — see [LICENSE.txt](LICENSE.txt).
Copyright 2026 Adam Clark.

Reading, auditing, modifying, and non-commercial redistribution are granted
to everyone; commercial use requires a written license from the Owner. The
verification companions
([sovereign_ledger](https://github.com/savageAZfck/sovereign_ledger),
[respawn](https://github.com/savageAZfck/respawn),
[edge_gate](https://github.com/savageAZfck/edge_gate),
[flight_tape](https://github.com/savageAZfck/flight_tape))
are released under the same terms: the tools that verify are open; the thing
they verify is auditable too.

## How this was built

Bad Apple was architected, threat-modeled, and audited by a human; large
portions of the implementation were produced with AI coding assistance and
then reviewed, red-teamed, and tested before being kept. That is disclosed
here because the project's own standard — *don't trust, verify* — applies to
its provenance as much as to its runtime. Judge the artifact: run the cert,
read the spec, verify the chain.
