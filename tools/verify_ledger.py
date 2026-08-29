#!/usr/bin/env python3
"""Standalone, third-party verifier for the Bad Apple audit ledger.

This tool deliberately imports NOTHING from the Bad Apple codebase. It only
needs Python's standard library (plus `cryptography`, a standard package, for
the optional Secure-Enclave-signed checkpoint check). You do not need Bad
Apple installed, running, or even present on this machine to use it -- you
only need an exported `ledger.jsonl` (and optionally `ledger_checkpoint.json`).

This is the whole point: you should not have to trust Bad Apple's own code,
its daemon, or its operator to know whether the audit trail it produced is
intact. You should be able to check it yourself, independently, with a tool
short enough to read in five minutes.

What it proves:
- The hash chain in ledger.jsonl is unbroken from genesis to tip: no entry
  was inserted, deleted, or edited without invalidating everything after it.
- (If a checkpoint file is supplied) the chain tip existed, byte-for-byte,
  at the time it was signed by a specific device's Secure Enclave identity
  key -- verified against that device's PUBLIC key, so you do not need to
  trust anything except elliptic-curve math.

What it does NOT prove:
- That the *content* of any entry is true (the ledger records what Bad
  Apple's daemon reported doing, not an independent audit of the OS).
- That entries were not omitted entirely before the chain started (the
  genesis point is only as trustworthy as when you first captured it).
- Anything about entries added *after* a checkpoint you're verifying against
  -- a checkpoint only vouches for the chain up to its own tip.

Usage:
    python3 tools/verify_ledger.py /path/to/ledger.jsonl
    python3 tools/verify_ledger.py /path/to/ledger.jsonl --checkpoint /path/to/ledger_checkpoint.json
    python3 tools/verify_ledger.py /path/to/ledger.jsonl --secret <hex-or-string>

If the ledger was written with BADAPPLE_LEDGER_SECRET set (HMAC mode), pass
the same secret with --secret to verify; without it you can still check
prev_hash linkage but not the per-entry HMAC.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import sys
from pathlib import Path
from typing import Any

DEFAULT_GENESIS = "bad-apple-genesis-v1"


def _canonical_json(data: Any) -> str:
    """Must exactly match badapple_extras._safe_json()."""
    return json.dumps(data, sort_keys=True, ensure_ascii=True, default=str)


def verify_chain(ledger_path: Path, genesis: str, secret: bytes | None) -> dict[str, Any]:
    """Recompute the hash chain independently and report the first break, if any."""
    if not ledger_path.is_file():
        return {"ok": False, "error": f"ledger file not found: {ledger_path}"}

    prev = hashlib.sha256(genesis.encode()).hexdigest()
    total = 0
    tip_hash = prev
    with open(ledger_path, encoding="utf-8") as f:
        for lineno, raw_line in enumerate(f, 1):
            line = raw_line.strip()
            if not line:
                continue
            total += 1
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as e:
                return {"ok": False, "error": f"line {lineno}: invalid JSON: {e}", "entries_checked": total - 1}

            for field in ("ts", "type", "data", "prev_hash", "hash"):
                if field not in entry:
                    return {"ok": False, "error": f"line {lineno}: missing field {field!r}", "entries_checked": total - 1}

            if entry["prev_hash"] != prev:
                return {
                    "ok": False,
                    "error": f"line {lineno}: chain broken -- prev_hash does not match the previous entry's hash "
                             f"(expected {prev[:16]}..., got {entry['prev_hash'][:16]}...)",
                    "entries_checked": total - 1,
                }

            calc = _canonical_json({
                "ts": entry["ts"],
                "type": entry["type"],
                "data": entry["data"],
                "prev_hash": entry["prev_hash"],
            })
            if secret:
                expected = hmac.new(secret, calc.encode(), hashlib.sha256).hexdigest()
            else:
                expected = hashlib.sha256(calc.encode()).hexdigest()

            if expected != entry["hash"]:
                return {
                    "ok": False,
                    "error": f"line {lineno}: hash mismatch -- entry content does not match its recorded hash "
                             "(the entry was edited after being written, or the wrong --secret was supplied)",
                    "entries_checked": total - 1,
                }

            prev = entry["hash"]
            tip_hash = prev

    return {"ok": True, "entries_checked": total, "tip_hash": tip_hash}


def verify_checkpoint(checkpoint_path: Path, chain_tip_hash: str, chain_entry_count: int) -> dict[str, Any]:
    """Verify a Secure-Enclave-signed checkpoint against a specific chain state.

    This is the low-level signature verifier. ``chain_tip_hash`` and
    ``chain_entry_count`` must describe the chain state **at the checkpoint's
    recorded position**, not necessarily the current tip — a checkpoint vouches
    for the chain up to its own tip; entries added after are protected by
    ongoing hash-chain linkage but not by this signature.

    Use :func:`verify_checkpoint_in_ledger` for the common case of verifying
    a checkpoint against a live, growing ledger.
    """
    try:
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        return {"ok": False, "error": f"could not read checkpoint: {e}"}

    for field in ("tip_hash", "entry_count", "signature", "public_key", "genesis", "signed_at"):
        if field not in checkpoint:
            return {"ok": False, "error": f"checkpoint missing field {field!r}"}

    if checkpoint["tip_hash"] != chain_tip_hash:
        return {
            "ok": False,
            "error": "checkpoint tip_hash does not match the chain state at the recorded "
                     f"position (entry {checkpoint['entry_count']}) -- either the ledger was "
                     "rewritten below that point, or the checkpoint is for a different ledger.",
        }
    if checkpoint["entry_count"] != chain_entry_count:
        return {
            "ok": False,
            "error": f"checkpoint recorded {checkpoint['entry_count']} entries but the chain "
                     f"state supplied has {chain_entry_count}.",
        }

    payload = _canonical_json({
        "genesis": checkpoint["genesis"],
        "tip_hash": checkpoint["tip_hash"],
        "entry_count": checkpoint["entry_count"],
        "signed_at": checkpoint["signed_at"],
    }).encode("utf-8")

    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import ec
    except ImportError:
        return {"ok": False, "error": "the `cryptography` package is required to verify the signature: pip install cryptography"}

    try:
        pubkey_bytes = base64.b64decode(checkpoint["public_key"])
        signature_bytes = base64.b64decode(checkpoint["signature"])
        pub = ec.EllipticCurvePublicKey.from_encoded_point(ec.SECP256R1(), pubkey_bytes)
        pub.verify(signature_bytes, payload, ec.ECDSA(hashes.SHA256()))
    except InvalidSignature:
        return {"ok": False, "error": "signature does not match -- the checkpoint was forged, corrupted, or signed by a different key than the one supplied"}
    except (ValueError, TypeError) as e:
        return {"ok": False, "error": f"could not verify signature: {e}"}

    return {
        "ok": True,
        "signed_at": checkpoint["signed_at"],
        "signed_entry_count": checkpoint["entry_count"],
        "public_key": checkpoint["public_key"],
    }


def _hash_at_position(ledger_path: Path, genesis: str, secret: bytes | None, target_count: int) -> dict[str, Any]:
    """Recompute the chain up to ``target_count`` entries and return the hash at that position.

    This lets us verify a checkpoint against the chain state *at the time it
    was signed*, even if the ledger has grown since.
    """
    if target_count < 0:
        return {"ok": False, "error": f"invalid target_count: {target_count}"}

    prev = hashlib.sha256(genesis.encode()).hexdigest()
    total = 0
    with open(ledger_path, encoding="utf-8") as f:
        for lineno, raw_line in enumerate(f, 1):
            line = raw_line.strip()
            if not line:
                continue
            total += 1
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as e:
                return {"ok": False, "error": f"line {lineno}: invalid JSON: {e}"}

            for field in ("ts", "type", "data", "prev_hash", "hash"):
                if field not in entry:
                    return {"ok": False, "error": f"line {lineno}: missing field {field!r}"}

            if entry["prev_hash"] != prev:
                return {"ok": False, "error": f"line {lineno}: chain broken before checkpoint position"}

            calc = _canonical_json({
                "ts": entry["ts"],
                "type": entry["type"],
                "data": entry["data"],
                "prev_hash": entry["prev_hash"],
            })
            if secret:
                expected = hmac.new(secret, calc.encode(), hashlib.sha256).hexdigest()
            else:
                expected = hashlib.sha256(calc.encode()).hexdigest()

            if expected != entry["hash"]:
                return {"ok": False, "error": f"line {lineno}: hash mismatch before checkpoint position"}

            prev = entry["hash"]

            if total == target_count:
                return {"ok": True, "hash_at_position": prev, "entries_checked": total}

    return {
        "ok": False,
        "error": f"ledger has only {total} entries but checkpoint was signed at position {target_count}",
    }


def verify_checkpoint_in_ledger(
    checkpoint_path: Path, ledger_path: Path, genesis: str = DEFAULT_GENESIS, secret: bytes | None = None
) -> dict[str, Any]:
    """Verify a Secure-Enclave-signed checkpoint against a live, growing ledger.

    Unlike :func:`verify_checkpoint`, this reads the ledger to find the chain
    state at the checkpoint's recorded position, so it works correctly even
    when new entries have been appended after the checkpoint was signed.
    The checkpoint vouches for the chain up to its own tip; entries after
    are protected by the ongoing hash-chain linkage (verified separately by
    :func:`verify_chain`).
    """
    try:
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        return {"ok": False, "error": f"could not read checkpoint: {e}"}

    for field in ("tip_hash", "entry_count", "signature", "public_key", "genesis", "signed_at"):
        if field not in checkpoint:
            return {"ok": False, "error": f"checkpoint missing field {field!r}"}

    if checkpoint["genesis"] != genesis:
        return {"ok": False, "error": f"checkpoint genesis mismatch: {checkpoint['genesis']!r} != {genesis!r}"}

    pos_result = _hash_at_position(ledger_path, genesis, secret, checkpoint["entry_count"])
    if not pos_result["ok"]:
        return pos_result

    # Count total entries in the ledger (for informational reporting)
    total_entries = 0
    with open(ledger_path, encoding="utf-8") as f:
        for raw_line in f:
            if raw_line.strip():
                total_entries += 1

    result = verify_checkpoint(checkpoint_path, pos_result["hash_at_position"], checkpoint["entry_count"])
    if result.get("ok"):
        result["current_ledger_entries"] = total_entries
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ledger", type=Path, help="Path to ledger.jsonl")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Path to ledger_checkpoint.json (optional)")
    parser.add_argument("--genesis", default=DEFAULT_GENESIS, help=f"Genesis string (default: {DEFAULT_GENESIS!r})")
    parser.add_argument("--secret", default=None, help="HMAC secret, if the ledger was written with BADAPPLE_LEDGER_SECRET set")
    args = parser.parse_args()

    secret = args.secret.encode() if args.secret else None

    print(f"Verifying chain integrity: {args.ledger}")
    chain_result = verify_chain(args.ledger, args.genesis, secret)
    if not chain_result["ok"]:
        print(f"  [FAIL] {chain_result['error']}")
        print(f"  {chain_result.get('entries_checked', 0)} entries verified before the break.")
        return 1
    print(f"  [PASS] {chain_result['entries_checked']} entries, unbroken chain from genesis to tip.")
    print(f"  tip hash: {chain_result['tip_hash']}")

    if args.checkpoint:
        print(f"\nVerifying Secure Enclave checkpoint: {args.checkpoint}")
        cp_result = verify_checkpoint_in_ledger(args.checkpoint, args.ledger, args.genesis, secret)
        if not cp_result["ok"]:
            print(f"  [FAIL] {cp_result['error']}")
            print("\nOverall: FAILED")
            return 1
        extra = ""
        if "current_ledger_entries" in cp_result and cp_result["current_ledger_entries"] != cp_result["signed_entry_count"]:
            extra = (f" (ledger has grown to {cp_result['current_ledger_entries']} entries since; "
                     f"those are protected by chain linkage)")
        print(f"  [PASS] Signature valid. Chain state at entry {cp_result['signed_entry_count']} "
              f"(tip {cp_result.get('tip_hash', '???')[:16]}...) was attested by device key "
              f"{cp_result['public_key'][:20]}... at {cp_result['signed_at']}.{extra}")

    print("\nOverall: VERIFIED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
