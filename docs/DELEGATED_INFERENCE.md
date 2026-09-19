# Delegated Inference — Trust Model & Protocol

`badapple-p2p ask` lets one Bad Apple instance borrow a trusted peer's loaded
model. This document is the normative spec for the feature's security model:
what exists today, and the roadmap to cross-identity delegation ("share your
brain").

The invariants, in every phase:

1. **Inference-only.** A delegated prompt never reaches the serving machine's
   tools, meta commands, approval flow, persona switching, or chat history.
   It is text in → model → text out.
2. **Attestation at both ends.** The action is recorded on the serving
   machine's ledger under the requester's identity, and the request is
   recorded on the asking machine's ledger under the server's identity.
   "I asked, they answered" is provable on both chains.
3. **Fail-closed.** Serving is opt-in per peer (`BADAPPLE_P2P_INFER=1`).
   The owner's kill switch, private mode, and output firewall apply to
   delegated traffic exactly as they do to local traffic.
4. **No context leakage.** Delegated queries carry no history, no workspace,
   no memory facts, no ambient context. The serving machine's output firewall
   redacts secrets before text leaves.

---

## Phase A — shipped: same-owner delegation

Single trust domain: all peers share one secret (`BADAPPLE_P2P_SECRET` or the
SLICKS key). This is correct for "my own machines" — the mesh is a private
LAN fabric where every peer is already trusted.

- `TransferFrame::InferRequest { request_id, prompt, max_tokens, from_peer }`
- `TransferFrame::InferResponse { request_id, text, tier, elapsed_ms }`
- Caps: prompt ≤ 32 KiB, tokens ≤ 2048 (enforced on both sides).
- Serving gate: `BADAPPLE_P2P_INFER=1` on the serving peer's process env.
- Serving path: `__BADAPPLE_DELEGATED__` daemon envelope →
  `generateDelegated()` — bypasses `handleMetaRequest`,
  `toolAwareGeneration`, approvals, personas, history, and `saveTurn`.
- Ledger: `delegated_query { from_peer, prompt }` →
  `delegated_response { from_peer, chars|error }` on the serving chain.

Known limits of Phase A: one shared secret means one blast radius;
`from_peer` is self-reported; the asking side does not yet ledger the
delegation; no per-peer permissions, rate limits, or revocation.

---

## Phase B — cross-identity trust: "share your brain"

Goal: delegate between **different people's machines** — your friend's 70B
answers your query, attested on their ledger. This is the network-effect
primitive: every new install makes every existing install smarter.

### B.1 Identity

A peer identity is the machine's **Secure Enclave P-256 public key
fingerprint** (`badapple-identity`, SLICKS v2). Hardware-bound, per-machine,
unforgeable — and already the trust root used for checkpoint signing.

Peer trust store: `/var/lib/bad_apple/peers.json`

```json
{
  "peers": [
    {
      "fingerprint": "sha256:ab12…",
      "label": "studio-m4",
      "added_at": "2026-09-19T00:00:00Z",
      "approved_by": "owner",
      "can_infer": true,
      "can_transfer": true,
      "max_tokens": 2048,
      "rate_limit_per_hour": 60,
      "daily_cap": 500
    }
  ]
}
```

### B.2 Pairing

Two flows, both requiring **owner consent on the serving side**:

1. **TOFU (trust on first use)** — first contact presents the peer's
   fingerprint; the owner approves it once (menu-bar prompt / CLI confirm),
   SSH-style. Subsequent packets verify against the pinned fingerprint.
2. **Invite codes** — `badapple-p2p invite` emits a short one-time code;
   the friend runs `badapple-p2p pair <addr> <code>`; the code authenticates
   the first exchange and both sides pin each other's fingerprint.

Either flow produces a **per-pair session key** — compromise of one pairing
exposes nothing else.

### B.3 Wire authentication (Phase B)

Phase A's single symmetric key becomes:

- Session key per pair (from invite flow) — AES-256-GCM, unchanged framing.
- Each `InferRequest` carries the requester's **Enclave-signed identity
  proof** (signature over `request_id ‖ ts ‖ prompt_hash`). The serving
  machine verifies it against the pinned fingerprint and records the
  *fingerprint* — not the self-reported label — in `delegated_query`.
- This makes `delegated_query` events non-repudiable: the serving chain
  proves *which physical machine* asked, not just *who claimed to*.

### B.4 Abuse controls

- Per-peer token bucket (`rate_limit_per_hour`) + daily cap.
- `badapple-p2p revoke <fingerprint>` — removes the pairing and rotates any
  pair material; takes effect on the next packet.
- Serving machine's `private mode` suspends delegated serving while active.
- IFY baseline learns `delegated_query` rates; a requester turning abusive
  surfaces as an anomaly finding on the serving machine.

---

## Phase C — engine-native delegation

Today `ask` is a CLI verb. The breakout UX is the organism delegating
**itself**: you ask Bad Apple a hard question and it borrows the Studio's
bigger brain transparently.

- New tool `delegate_query { peer, prompt }` in the tool router.
- Policy-gated and approval-required by default — it is an action with
  external effects (work performed on another machine).
- The serving machine sees it as a normal `InferRequest` — the engine path
  on the serving side is unchanged.
- The asker's own chain now attests the request through the normal
  `tool_call`/`tool_result` events: "it decided to borrow a bigger brain"
  is itself a receipt.
- Response rendering: `served by sha256:ab12… · tier main · attested on
  their ledger`.

Phase C closes the dual-attestation gap without a Rust-side ledger writer:
the asker's delegation lives in its chain as a tool call.

---

## Threat analysis

| Threat | Mitigation |
|---|---|
| Malicious peer prompt | Inference-only path — cannot reach tools, meta, approvals, persona, or history |
| Forged requests | Enclave-signed identity proof verified against pinned fingerprint; per-pair session keys |
| Requester abuse/spam | Per-peer rate limits, daily caps, `revoke`, IFY anomaly detection |
| Privacy leak via delegation | No context/history/workspace is attached; output firewall redacts before text leaves |
| Ledger ambiguity | `delegated_*` event types are explicit; verifiers can filter delegated traffic from organic actions |
| Serving-side coercion | Owner consent at pairing; kill switch and private mode suspend serving |
| One pair compromised | Per-pair keys; rotation via re-pair; other pairs unaffected |

## Open questions (post-Phase C)

- **Untrusted-peer economics** — rate limits make casual sharing safe, but
  stranger-scale sharing wants accounting. Ledger receipts are a natural
  credit primitive later ("your chain shows you served N tokens").
- **WAN reach** — transport is LAN-first; WAN needs relay or NAT traversal.
  Design explicitly does not require it.
- **Streaming responses** — Phase A returns whole text; token streaming over
  the mesh is a frame-protocol extension, not a security change.
- **Asker-side redaction** — prompts delegated outbound should pass through
  the requester's output firewall before hitting the wire, same as inbound.
