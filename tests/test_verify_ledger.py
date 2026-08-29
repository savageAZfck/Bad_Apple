"""Tests for the hash-chained audit ledger and its standalone verifier.

Covers both sides of the trust story:
- badapple_extras.AuditLedger: the live daemon-side ledger and checkpoint signer.
- tools.verify_ledger: the standalone, dependency-free verifier a third party
  (with no Bad Apple code, daemon, or trust in the operator) would run.
"""

import base64
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec

from badapple_extras import AuditLedger

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.verify_ledger import verify_chain, verify_checkpoint  # noqa: E402


def _canonical(data) -> str:
    return json.dumps(data, sort_keys=True, ensure_ascii=True, default=str)


class AuditLedgerCheckpointTests(unittest.TestCase):
    def test_sign_checkpoint_without_identity_reports_clear_error(self):
        # On a machine (or CI runner) without a real Secure Enclave identity,
        # sign_checkpoint() must fail cleanly rather than write a bogus file.
        with tempfile.TemporaryDirectory() as tmp:
            ledger = AuditLedger(Path(tmp))
            ledger.record("query", {"prompt": "hello"})
            result = ledger.sign_checkpoint()
            # badapple_slicks may or may not report an identity depending on
            # the host; either a clean failure or a real signed checkpoint
            # (verified in the software-key tests below) is acceptable here.
            if not result["ok"]:
                self.assertIn("error", result)
            else:
                self.assertTrue(Path(result["path"]).is_file())


class VerifyChainTests(unittest.TestCase):
    def _write_ledger(self, path: Path, n: int) -> None:
        ledger = AuditLedger(path.parent, genesis="test-genesis")
        assert ledger.ledger_path == path
        for i in range(n):
            ledger.record("query", {"prompt": f"message {i}"})

    def test_valid_chain_verifies(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger_path = Path(tmp) / "ledger.jsonl"
            self._write_ledger(ledger_path, 10)
            result = verify_chain(ledger_path, "test-genesis", secret=None)
            self.assertTrue(result["ok"], result.get("error"))
            self.assertEqual(result["entries_checked"], 10)

    def test_tampered_entry_is_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger_path = Path(tmp) / "ledger.jsonl"
            self._write_ledger(ledger_path, 10)

            lines = ledger_path.read_text(encoding="utf-8").splitlines()
            entry = json.loads(lines[5])
            entry["data"]["prompt"] = "TAMPERED"
            lines[5] = json.dumps(entry)
            ledger_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

            result = verify_chain(ledger_path, "test-genesis", secret=None)
            self.assertFalse(result["ok"])
            self.assertIn("hash mismatch", result["error"])
            self.assertEqual(result["entries_checked"], 5)

    def test_broken_prev_hash_linkage_is_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger_path = Path(tmp) / "ledger.jsonl"
            self._write_ledger(ledger_path, 5)

            lines = ledger_path.read_text(encoding="utf-8").splitlines()
            # Delete a middle line entirely -- classic tamper: erase an entry.
            del lines[2]
            ledger_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

            result = verify_chain(ledger_path, "test-genesis", secret=None)
            self.assertFalse(result["ok"])
            self.assertIn("chain broken", result["error"])

    def test_missing_file_reports_error_not_crash(self):
        result = verify_chain(Path("/nonexistent/ledger.jsonl"), "test-genesis", secret=None)
        self.assertFalse(result["ok"])
        self.assertIn("not found", result["error"])

    def test_hmac_secret_mode_round_trips(self):
        with tempfile.TemporaryDirectory() as tmp:
            secret = b"a shared secret"
            ledger = AuditLedger(Path(tmp), genesis="test-genesis")
            ledger._secret = secret
            ledger.record("query", {"prompt": "hi"})
            ledger.record("response", {"text": "hello"})

            ok_result = verify_chain(ledger.ledger_path, "test-genesis", secret=secret)
            self.assertTrue(ok_result["ok"])

            wrong_secret_result = verify_chain(ledger.ledger_path, "test-genesis", secret=b"wrong secret")
            self.assertFalse(wrong_secret_result["ok"])


class VerifyCheckpointTests(unittest.TestCase):
    """Exercise the signature-verification logic with a software test keypair,
    independent of whether real Secure Enclave hardware is available.
    """

    def _make_signed_checkpoint(self, tmp: Path, tip_hash: str, entry_count: int):
        private_key = ec.generate_private_key(ec.SECP256R1())
        public_key = private_key.public_key()
        pubkey_bytes = public_key.public_bytes(
            encoding=serialization.Encoding.X962,
            format=serialization.PublicFormat.UncompressedPoint,
        )

        checkpoint = {
            "genesis": "test-genesis",
            "tip_hash": tip_hash,
            "entry_count": entry_count,
            "signed_at": "2026-01-01T00:00:00+00:00",
        }
        payload = _canonical(checkpoint).encode("utf-8")
        signature = private_key.sign(payload, ec.ECDSA(hashes.SHA256()))

        checkpoint["signature"] = base64.b64encode(signature).decode("ascii")
        checkpoint["public_key"] = base64.b64encode(pubkey_bytes).decode("ascii")

        checkpoint_path = tmp / "ledger_checkpoint.json"
        checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")
        return checkpoint_path, private_key

    def test_valid_checkpoint_verifies(self):
        with tempfile.TemporaryDirectory() as tmp:
            tip_hash = hashlib.sha256(b"fake tip").hexdigest()
            checkpoint_path, _ = self._make_signed_checkpoint(Path(tmp), tip_hash, 42)
            result = verify_checkpoint(checkpoint_path, tip_hash, 42)
            self.assertTrue(result["ok"], result.get("error"))

    def test_checkpoint_rejects_mismatched_tip_hash(self):
        # Simulates an attacker who rewrote the chain (recomputing all
        # downstream hashes) after the checkpoint was signed.
        with tempfile.TemporaryDirectory() as tmp:
            tip_hash = hashlib.sha256(b"real tip").hexdigest()
            checkpoint_path, _ = self._make_signed_checkpoint(Path(tmp), tip_hash, 42)
            forged_tip_hash = hashlib.sha256(b"forged tip").hexdigest()
            result = verify_checkpoint(checkpoint_path, forged_tip_hash, 42)
            self.assertFalse(result["ok"])
            self.assertIn("does not match", result["error"])

    def test_checkpoint_rejects_forged_signature(self):
        with tempfile.TemporaryDirectory() as tmp:
            tip_hash = hashlib.sha256(b"real tip").hexdigest()
            checkpoint_path, _ = self._make_signed_checkpoint(Path(tmp), tip_hash, 42)

            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            # Attacker without the private key swaps in a signature from a
            # different (also attacker-controlled) keypair, but the public
            # key on file is the real one -- must fail.
            forger_key = ec.generate_private_key(ec.SECP256R1())
            forged_sig = forger_key.sign(b"anything", ec.ECDSA(hashes.SHA256()))
            checkpoint["signature"] = base64.b64encode(forged_sig).decode("ascii")
            checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")

            result = verify_checkpoint(checkpoint_path, tip_hash, 42)
            self.assertFalse(result["ok"])

    def test_checkpoint_rejects_wrong_entry_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            tip_hash = hashlib.sha256(b"real tip").hexdigest()
            checkpoint_path, _ = self._make_signed_checkpoint(Path(tmp), tip_hash, 42)
            result = verify_checkpoint(checkpoint_path, tip_hash, 999)
            self.assertFalse(result["ok"])
            self.assertIn("entries", result["error"])


if __name__ == "__main__":
    unittest.main()
