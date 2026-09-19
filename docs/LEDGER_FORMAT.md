# Bad Apple Audit Ledger Format

The primary audit ledger (`/var/lib/bad_apple/ledger.jsonl`) is the runtime's own attestation of everything it did. It is hash-chained, HMAC-authenticated, secret-redacted, and re-verified independently by `badapple-sovereign` (which writes a second, sealed chain using the public `sovereign_ledger` crate — see its SPEC.md for the sovereign-chain format).

## Record structure

One JSON object per line. Fields:

| field | type | meaning |
|---|---|---|
| `seq` | uint | monotonically increasing sequence, starting at 1 |
| `ts_ms` | int | Unix milliseconds |
| `kind` | string | event kind: `query`, `response`, `tool_call`, `tool_result`, `approval`, `cache_hit`, `error`, `control` |
| `actor` | string | `user`, `model`, `system`, `ify` |
| `payload` | object | event-specific data (redacted before write) |
| `prev_hash` | hex | SHA-256 of the previous record's canonical preimage |
| `hash` | hex | SHA-256 of this record's canonical preimage |
| `mac` | hex | HMAC-SHA256 over the record, keyed by the local ledger key |

## Chain construction

```
preimage = version || seq || ts_ms || kind || actor || canonical(payload) || prev_hash
hash     = SHA-256(preimage)
mac      = HMAC-SHA256(ledger_key, preimage)
```

- `seq` must be `prev.seq + 1`; first record has `seq = 1` and `prev_hash = ""`
- `canonical(payload)` is deterministic JSON serialization (sorted keys, no whitespace)
- Any gap in `seq`, mismatch in `prev_hash`, or MAC failure fails verification — the verifier does not skip or reorder

## Redaction (before write)

Secrets are removed from `payload` before hashing — so the chain never contains the secret it redacts:

- private key blocks, `sk-`-style API keys, `AKIA`-style access keys
- emails, phone numbers, SSNs
- long high-entropy tokens

Redaction is a one-way transform: the ledger records *that* a secret was present (`[REDACTED]`), never the secret. `badapple cert` includes a check that scans the ledger for known secret patterns and fails if any survive.

## Verification

```bash
badapple --doctor                # walks the chain, checks hashes/MACs/sequence
badapple-sovereign               # re-verifies every entry into the sovereign copy
```

The verifier recomputes each preimage from the record's fields and compares `hash` and `mac`; it does not trust the stored values. A record that parses but recomputes differently fails.

## What the chain proves — and doesn't

**Proves:** the recorded history is internally consistent and was written under the ledger key; any modification, deletion, reorder, or insertion after the fact breaks the chain detectably.

**Does not prove:** that the recorded events are *true* (the writer could record a lie — the chain attests consistency, not honesty); that all copies survive (same-machine deletion destroys them — off-box checkpoints are the mitigating layer); or that two viewers saw the same history (equivocation — witnessing is the acknowledged frontier, tracked in THREAT_MODEL.md).
