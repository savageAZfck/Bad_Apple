# Bad Apple

> **A sovereign, local AI operating layer for macOS — one that can prove what it did.**
>
> On-device MLX inference, Secure Enclave identity, a hash-chained audit
> ledger, an independent verification layer, and a brake-only watchdog —
> running entirely on Apple Silicon. After the models are cached, inference
> needs no network. Under full air-gap, `badapple cert` asserts the daemon
> holds **zero** network sockets.
>
> Don't trust this README. Verify it.

---

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
| `badapple receipts` | Prints a proof card: attested-action count, organism age, chain tip, sovereign seal, watchdog phase, identity. |
| `badapple cert` | Runs the **16-check** air-gap/security certification suite; exits non-zero on any failure. |
| `badapple export-proof` | Exports a self-contained verification bundle of the AI's own attested history — publicly verifiable, **no secrets required**. |
| `badapple --doctor` | Diagnostics, binary checks, ledger hash-chain verification. |
| `badapple ify status` | Shows the watchdog's phase, baseline, and findings. |

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

## What Bad Apple is

Not a chat app. Not a cloud wrapper. A set of `launchd` daemons, native
Swift/Rust services, and a menu bar that turn an Apple Silicon Mac into a
private, air-gap-certifiable assistant with real OS-level hands — files,
shell, AppleScript, Shortcuts, workspace indexing, local MCP tools, native
TTS — under a declarative security policy and a tamper-evident ledger.

The design thesis: **an assistant should be able to prove its own behavior.**
Every query, tool call, approval, and refusal is appended to a hash-chained,
secret-redacting ledger. A second, independent layer re-verifies that ledger
into a sealed sovereign chain and signs daily checkpoints through the Secure
Enclave. A watchdog (IFY) tails the ledger, learns baselines, and files
approval-gated findings — it can brake, never steer. And a 14-seat
**Council of Minds** deliberates before gated actions run: under
autopilot, actions that pass the vote execute and contested ones come
back to you; every vote is journaled on the ledger.

## Install

### Homebrew Cask (recommended)

```bash
brew tap savageAZfck/bad-apple https://github.com/savageAZfck/homebrew-bad-apple
brew install --cask bad-apple
```

### Direct download

See [bad-apple-releases](https://github.com/savageAZfck/bad-apple-releases)
for the signed-artifact beta zips. Unsigned builds trip Gatekeeper; the
bundled `strip_quarantine.sh` handles this locally. **Notarization is on the
roadmap** — see "Honest limits" below.

### From source (unsigned, no Apple Developer ID needed)

```bash
cargo build --release
BADAPPLE_NO_SIGN=1 src/platform/apple_desktop/build_bad_apple_menu_bar.sh
sudo src/platform/apple_desktop/strip_quarantine.sh
osascript -e 'do shell script "cd /path/to/bad_apple && src/platform/apple_bridge/install_badapple_platform.sh --install --unsigned-install" with administrator privileges'
```

**Hardware:** Apple Silicon (M1+). 8 GB unified memory minimum, 16 GB
recommended. ~40 GB free disk for models and state.

## Architecture

```text
badapple CLI / menu bar / voice / dashboard
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
   badapple-engine  (Swift MLX daemon)
   ├─ Local models: 7B default · 9B switchable · 0.5B fast tier
   ├─ RAG, semantic cache, native embeddings
   ├─ Policy engine (60+ rules) + human-in-the-loop approvals
   ├─ Council of Minds — 14 deterministic seats vote on every gated
   │  action; passed votes run, failed votes escalate to the human
   ├─ Streaming output firewall (Aho-Corasick secret redaction)
   ├─ Hash-chained audit ledger
   ├─ IFY watchdog (brake-only) · Curious autopilot (policy-gated)
   ├─ Workspace watcher · MCP host · optional P2P mesh
   │    (model transfer + delegated inference — borrow a peer's brain)
   └─ Air-gap certification self-audit
              │
              ▼
   badapple-tts  (native AVSpeechSynthesizer)

Verification layers (independent of the engine):
   badapple-sovereign → re-verifies ledger into sealed sovereign chain,
                        Secure Enclave-signed daily checkpoints
   badapple-respawn   → content-addressed snapshots of all state,
                        drift detection, revert-to-any-point
```

## The verification stack

| Layer | What it proves | How to check it |
|---|---|---|
| Audit ledger | Every action is hash-chained and secret-redacted | `badapple --doctor` |
| Sovereign ledger | Independent re-verified copy, sealed, publicly verifiable | `badapple export-proof` + `verify_public` |
| Secure Enclave checkpoints | Daily signed chain-tip + Merkle root | `badapple receipts` |
| IFY watchdog | Anomalies and drift surfaced as approval-gated findings | `badapple ify status` |
| Council | Every gated action carries a journaled 14-seat vote with rationales | `badapple "council <q>"` · `council_deliberation` events |
| Air-gap cert | 16 runtime checks, incl. zero external sockets | `badapple cert` |
| Respawn | All platform state revertible to any snapshot | `badapple-respawn --status` |

Companion open components: [`sovereign_ledger`](https://github.com/savageAZfck/sovereign_ledger),
[`respawn`](https://github.com/savageAZfck/respawn),
[`edge_gate`](https://github.com/savageAZfck/edge_gate).

## Security model — and honest limits

Read [THREAT_MODEL.md](THREAT_MODEL.md) and [SECURITY.md](SECURITY.md) first.
The short version:

- **Air-gap is a certified posture, not a magic property.** Downloads, P2P,
  and MCP are optional doors; `badapple cert` proves them shut when they
  should be. Model files you carry in are explicit trust-boundary inputs —
  the chain verifies provenance, not benevolence.
- **Unsigned binaries.** Until notarization lands, first install shows a
  Gatekeeper warning. That is a real UX cost, stated plainly.
- **Local model ceiling.** The default brain is a 7B-class local model. It is
  not a frontier cloud model and will not pretend to be one.
- **Prompt injection is a real residual.** The policy engine, cages, and
  firewall bound it; they do not eliminate it. See THREAT_MODEL.md §5.
- **Fail-closed by design.** Denied tools stay denied; approvals are
  explicit, logged, and deniable (`deny <id>`).

## Quick use

```bash
badapple "What is 2+2?"                  # text query
badapple --speak "What do you think of Siri?"   # voice with native TTS
badapple "are you alone"                 # on-demand self-audit, in persona
badapple "approve <id>" / "deny <id>"    # human-in-the-loop approvals
badapple --benchmark                     # performance suite
badapple "switch to wicket"              # persona switch
badapple-p2p ask <peer> "prompt"         # delegate a query to a peer's brain
```

## Build & test

```bash
cargo build --release
cargo fmt --check
cargo clippy --all-targets --all-features --release -- -D warnings
cargo test --release          # 103 unit + 21 integration tests
cargo test --release --test cert_suite
badapple cert                 # live 16-check air-gap certification
```

Swift engine & app bundle:

```bash
BADAPPLE_NO_SIGN=1 src/platform/apple_desktop/build_bad_apple_menu_bar.sh
```

## Documentation

- [THREAT_MODEL.md](THREAT_MODEL.md) — adversary classes, defenses, residuals
- [SECURITY.md](SECURITY.md) — security posture and reporting
- [docs/LEDGER_FORMAT.md](docs/LEDGER_FORMAT.md) — primary ledger format spec
- [docs/SLICKS_PROTOCOL.md](docs/SLICKS_PROTOCOL.md) — IPC protocol spec
- [BAD_APPLE.md](BAD_APPLE.md) — technical deep dive and live benchmarks
- [AGENTS.md](AGENTS.md) — build commands and project conventions
- [IFY.md](IFY.md) — watchdog design spec
- [CHANGELOG.md](CHANGELOG.md) — development history

## License

**FSL-1.1-ALv2** — Functional Source License 1.1, Apache-2.0 future license.
Copyright 2026 Adam Clark.

This is a **source-available** license, not an OSI-approved open-source
license — stated plainly. The source is fully auditable; for two years it may
not be used to offer a competing product, after which it converts to
Apache-2.0. The verification tooling
([sovereign_ledger](https://github.com/savageAZfck/sovereign_ledger),
[respawn](https://github.com/savageAZfck/respawn),
[edge_gate](https://github.com/savageAZfck/edge_gate)) is released under the
same terms: the tools that verify are open; the thing they verify is
auditable too.

## How this was built

Bad Apple was architected, threat-modeled, and audited by a human; large
portions of the implementation were produced with AI coding assistance and
then reviewed, red-teamed, and tested before being kept. That is disclosed
here because the project's own standard — *don't trust, verify* — applies to
its provenance as much as to its runtime. Judge the artifact: run the cert,
read the spec, verify the chain.
