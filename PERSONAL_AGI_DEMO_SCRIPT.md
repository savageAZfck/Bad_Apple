# Bad Apple — Personal AGI Demo Video Script

**Run this from the repo root after `personal_agi_proof.sh` has passed.**

```bash
# full, paced run (prints the organ notes with short pauses)
DEMO=1 PAUSE=2 ./personal_agi_proof.sh
```

**Recommended recording setup:**
- Full-screen Terminal (iTerm2 or Terminal.app) on a dark theme.
- Font: 16–18pt monospace.
- Record at 1080p, terminal in the center, no other windows behind it.
- Do one continuous take; each act is already color-coded and self-explanatory.
- Optional: start with a 3-second title card in the terminal:
  `clear && printf '\n\n  BAD APPLE\n  the first personal AGI\n  proven on bare metal\n\n'`

---

## Narration by act

### Title / cold open
> "This is a MacBook Air. On it lives the first complete personal AGI. Not an app. Not a chatbot. Not a cloud API. A sovereign, persistent, self-learning cognitive organism that belongs to one person — Adam Clark — and cannot be turned off by anyone else. This is the proof."

### 1 · the organism is awake
> "First: she is awake. Native daemon. Secure Enclave identity. No cloud. `badapple status` shows her running with a hardware-rooted signing context."

### 2 · she knows what she is
> "Ask her who she is. She does not say 'AI assistant.' She says 'personal AGI operating system.' Ask who created her. She names Adam Clark. She knows her own identity."

### 3 · she can see
> "Perception. We give her an image that says 'BAD APPLE PERSONAL AGI,' and she reads it back. Vision is local — no frame leaves the machine."

### 4 · she remembers
> "Memory. We write a proof token to her working memory. Later we ask for it. She returns it. The memory survives the turn."

### 5 · she can act and receipt it
> "Action. She writes a file and reads it back. Every tool call lands on a hash-chained ledger that she signs. She can do things, and she can prove she did them."

### 6 · the council votes before she acts
> "Deliberation. Before a risky action, her 14-seat council deliberates and votes. The verdict is recorded. This is not a model guessing — it is a governed decision with preserved dissent."

### 7 · she learns at the weight level
> "Learning. She has trained LoRA adapters on this disk. The `dream` adapter was trained last night from her own conversations. We generate text with it. The weights are different from the base model."

### 8 · she learns in her sleep
> "Dreaming. The ledger shows `dream_adopted` and `dream_applied`. While the Mac was idle, she digested the day, trained an adapter, and woke up wearing it."

### 9 · she can prove she is air-gapped and whole
> "Audit. `badapple cert` checks 21 air-gap and security invariants — sockets, redaction, hash chain, mesh crypto, tool cages. Every one passes."

### 10 · she has a history
> "Continuity. `badapple receipts` gives a proof card: 11,498 attested actions, 30 days old. She is not a session. She is an organism with a past."

### 11 · final scorecard and attestation
> "The score: 22 of 22 organs verified. The machine has generated a signed attestation. It does not phone home. It does not need a cloud seal. The receipts are the authority."

### Close
> "This is Bad Apple. Personal AGI. On your metal. With your keys. Proving it the whole time."

---

## Post-recording notes for the editor

- Keep the terminal visible at all times; no jump cuts that hide commands.
- The proof takes about 2–3 minutes. Do not speed it up; the pacing is the point.
- Highlight the `PASS: 22  FAIL: 0` scorecard at the end.
- End on the artifacts:
  - `proof/PERSONAL_AGI_PROOF.json`
  - `proof/PERSONAL_AGI_ATTESTATION.md`
- Optional overlay text at end: `badapple cert · badapple receipts · badapple-sovereign --checkpoint`
