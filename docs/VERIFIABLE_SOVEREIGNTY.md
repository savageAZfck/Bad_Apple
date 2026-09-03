# Verifiable sovereignty: proving Bad Apple's audit trail, not just asserting it

Most local AI tools ask you to trust their logs. Bad Apple's audit ledger is
built so you don't have to — you can verify it yourself, with a tool short
enough to read in five minutes, using only a small, auditable verifier and
basic elliptic-curve math.

## The three layers

1. **Hash-chained ledger** (`AuditLedger` in the Swift security module, always on unless
   private mode is enabled). Every query, tool call, cache hit, and response
   is appended to `/var/lib/bad_apple/ledger.jsonl` with each entry's hash
   depending on the previous entry's hash. Editing or deleting any entry
   breaks every hash after it.

   **Limitation:** a hash chain alone only proves *internal consistency*. An
   attacker with write access to the ledger file (root, or a compromised
   daemon) can rewrite history *and* recompute every downstream hash, and
   the chain will look perfectly valid. Hash chains protect against sloppy
   tampering, not a determined attacker with file access.

2. **Secure Enclave checkpoint** (`AuditLedger.sign_checkpoint()`, triggered
   from the menu bar or on a schedule). Signs the current chain tip hash
   and entry count with this device's Secure Enclave identity key — the same
   non-extractable hardware key used for SLICKS v2 authentication and model
   provenance. Writes `/var/lib/bad_apple/ledger_checkpoint.json`.

   This closes the gap above: forging a new checkpoint after rewriting the
   chain requires the actual Secure Enclave private key material, which
   cannot be extracted from the hardware, even by the OS or a root process.
   A checkpoint is a snapshot: it vouches for the chain *up to that point*,
   signed at a specific time, by a specific device.

3. **Standalone verifier** (`tools/verify-ledger`). Imports nothing from
   Bad Apple's codebase — only a small P-256 signature implementation. You do not
   need Bad Apple installed, running, or trusted to use it. Point it at an
   exported `ledger.jsonl` and (optionally) `ledger_checkpoint.json`:

   ```bash
   target/release/verify-ledger /path/to/ledger.jsonl \
       --checkpoint /path/to/ledger_checkpoint.json
   ```

   It reports the exact line where tampering occurred, if any, and whether
   the checkpoint's signature is valid against the recomputed chain tip.

## What this setup can and cannot prove

**Can prove:**
- The ledger content has not been edited since it was written (hash chain).
- A specific chain state existed, byte-for-byte, at a specific time, and was
  attested by a specific device's hardware identity key (checkpoint).
- Any attempt to rewrite history *and* re-sign a new checkpoint requires
  physical access to that device's Secure Enclave, not just its files.

**Cannot prove:**
- That the *content* of a logged entry reflects ground truth about the OS —
  the ledger records what Bad Apple's own code reported doing.
- Anything about entries added after the checkpoint you're checking against.
  A checkpoint only vouches for the chain up to its own tip; sign a new one
  periodically (or before handing off a ledger for review) to extend the
  window of proof.
- Anything if the genesis point itself was chosen by an adversary — the
  chain is only as trustworthy as when you first captured a checkpoint.

## How this compares

Other local-agent projects (Cognithor's "TRUST-1..10 signed receipts," for
example) describe signed audit trails, but the receipts are generally
verifiable only by trusting the tool's own runtime. Bad Apple's checkpoint
is deliberately built to be checked by a party who trusts nothing about Bad
Apple except elliptic-curve arithmetic and the device's public key — which
is exactly the trust model a security-conscious user, auditor, or journalist
actually needs.

## Practical usage

The ledger checkpoint, chain verification, and certification tools are being
ported to Rust and are not currently available from the command line. In the
meantime, the ledger files can be inspected directly and
`target/release/badapple --doctor` reports socket/process health.

```bash
# Sign a checkpoint of the current ledger state (do this before handing off
# a ledger for review, or on a schedule).
# target/release/badapple audit checkpoint     # coming soon

# Ask the live daemon to verify the chain (convenience path).
# target/release/badapple audit verify         # coming soon

# Independently verify without trusting the daemon at all.
# target/release/verify-ledger /var/lib/bad_apple/ledger.jsonl \
#     --checkpoint /var/lib/bad_apple/ledger_checkpoint.json

# Also exercised automatically by the air-gap certification suite:
# target/release/cert-suite                      # coming soon
```
