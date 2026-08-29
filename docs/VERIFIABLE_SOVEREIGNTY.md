# Verifiable sovereignty: proving Bad Apple's audit trail, not just asserting it

Most local AI tools ask you to trust their logs. Bad Apple's audit ledger is
built so you don't have to — you can verify it yourself, with a tool short
enough to read in five minutes, using nothing but Python's standard library
and basic elliptic-curve math.

## The three layers

1. **Hash-chained ledger** (`badapple_extras.AuditLedger`, always on unless
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
   with `agent_client.py audit checkpoint`). Signs the current chain tip hash
   and entry count with this device's Secure Enclave identity key — the same
   non-extractable hardware key used for SLICKS v2 authentication and model
   provenance. Writes `/var/lib/bad_apple/ledger_checkpoint.json`.

   This closes the gap above: forging a new checkpoint after rewriting the
   chain requires the actual Secure Enclave private key material, which
   cannot be extracted from the hardware, even by the OS or a root process.
   A checkpoint is a snapshot: it vouches for the chain *up to that point*,
   signed at a specific time, by a specific device.

3. **Standalone verifier** (`tools/verify_ledger.py`). Imports nothing from
   Bad Apple's codebase — only Python's standard library, plus the
   widely-used `cryptography` package for the signature check. You do not
   need Bad Apple installed, running, or trusted to use it. Point it at an
   exported `ledger.jsonl` and (optionally) `ledger_checkpoint.json`:

   ```bash
   python3 tools/verify_ledger.py /path/to/ledger.jsonl \
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

```bash
# Sign a checkpoint of the current ledger state (do this before handing off
# a ledger for review, or on a schedule).
.venv/bin/python agent_client.py audit checkpoint

# Ask the live daemon to verify the chain (convenience path).
.venv/bin/python agent_client.py audit verify

# Independently verify without trusting the daemon at all.
.venv/bin/python tools/verify_ledger.py /var/lib/bad_apple/ledger.jsonl \
    --checkpoint /var/lib/bad_apple/ledger_checkpoint.json

# Also exercised automatically by the air-gap certification suite:
sudo .venv/bin/python cert_suite.py
```
