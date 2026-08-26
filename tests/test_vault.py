#!/usr/bin/env python3
"""Unit tests for the Bad Apple vault (generations and encrypted backup)."""

import tempfile
import unittest
from pathlib import Path

from badapple_vault import ArtifactManifest, EncryptedBackup, GenerationStore


class VaultTests(unittest.TestCase):
    def test_generation_store_commit_and_restore(self) -> None:
        """GenerationStore can commit and restore a set of files."""
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "data"
            source_dir = Path(tmp) / "source"
            source_dir.mkdir()
            (source_dir / "prompt.txt").write_text("hello")
            (source_dir / "notes.txt").write_text("world")

            store = GenerationStore(data_dir, keep=3)
            generation_id = store.commit(
                "test-gen",
                {"prompt.txt": source_dir / "prompt.txt", "notes.txt": source_dir / "notes.txt"},
            )
            self.assertIsInstance(generation_id, str)
            self.assertTrue((data_dir / "generations" / generation_id).is_dir())

            destination = Path(tmp) / "restored"
            restore = store.restore(generation_id, destination)
            self.assertEqual(restore["generation_id"], generation_id)
            self.assertEqual(len(restore["restored"]), 2)
            self.assertEqual((destination / "prompt.txt").read_text(), "hello")
            self.assertEqual((destination / "notes.txt").read_text(), "world")

    def test_manifest_validates_hashes(self) -> None:
        """ArtifactManifest detects a modified artifact."""
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "file.txt"
            f.write_text("good")
            manifest = ArtifactManifest.create([f])
            self.assertTrue(manifest["artifacts"][0]["sha256"])

            # Tamper with the file after manifest creation.
            f.write_text("bad")
            verify = ArtifactManifest.verify(manifest)
            self.assertFalse(verify["valid"])

    def test_encrypted_backup_round_trip(self) -> None:
        """EncryptedBackup creates and restores an archive."""
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source"
            source.mkdir()
            (source / "a.txt").write_text("alpha")
            (source / "b.txt").write_text("beta")
            output = Path(tmp) / "backup.bin"

            info = EncryptedBackup.create(source, output, "super-secret-12")
            self.assertIn("path", info)
            self.assertTrue(output.is_file())

            restore_dir = Path(tmp) / "restore"
            restored = EncryptedBackup.extract(output, restore_dir, "super-secret-12")
            self.assertEqual(restored["files"], 2)
            self.assertEqual((restore_dir / "a.txt").read_text(), "alpha")
            self.assertEqual((restore_dir / "b.txt").read_text(), "beta")

    def test_encrypted_backup_requires_long_passphrase(self) -> None:
        """EncryptedBackup rejects short passphrases."""
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source"
            source.mkdir()
            (source / "x.txt").write_text("x")
            output = Path(tmp) / "backup.bin"
            with self.assertRaises(ValueError):
                EncryptedBackup.create(source, output, "short")


if __name__ == "__main__":
    unittest.main()
