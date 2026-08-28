#!/usr/bin/env python3
"""Property-style tests for model provenance verification."""

import os
import shutil
import tempfile
import unittest
from pathlib import Path

from badapple_model_provenance import FileEntry, ModelManifest, ModelProvenance


class ModelProvenanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        os.environ["BADAPPLE_DATA_DIR"] = self.tmpdir

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        os.environ.pop("BADAPPLE_DATA_DIR", None)

    def _make_model(self, root: Path, files: dict[str, str]) -> None:
        for rel, content in files.items():
            p = root / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")

    def test_record_and_verify(self) -> None:
        data = Path(self.tmpdir) / "data"
        data.mkdir()
        model = Path(self.tmpdir) / "model"
        model.mkdir()
        self._make_model(model, {"config.json": "{}", "weights.safetensors": "abc" * 1000})

        prov = ModelProvenance(data)
        result = prov.record("m1", "repo/m1", str(model))
        self.assertEqual(result["status"], "recorded")

        v = prov.verify("m1", str(model))
        self.assertEqual(v["status"], "verified")
        self.assertEqual(v["files"], 2)

    def test_tamper_detected(self) -> None:
        data = Path(self.tmpdir) / "data"
        data.mkdir()
        model = Path(self.tmpdir) / "model"
        model.mkdir()
        self._make_model(model, {"config.json": "{}", "weights.safetensors": "abc"})

        prov = ModelProvenance(data)
        prov.record("m1", "repo/m1", str(model))

        # Tamper
        (model / "weights.safetensors").write_text("xyz")
        v = prov.verify("m1", str(model))
        self.assertEqual(v["status"], "mismatch")
        self.assertIn("weights.safetensors", v["mismatches"][0])

    def test_missing_file_detected(self) -> None:
        data = Path(self.tmpdir) / "data"
        data.mkdir()
        model = Path(self.tmpdir) / "model"
        model.mkdir()
        self._make_model(model, {"config.json": "{}", "weights.safetensors": "abc"})

        prov = ModelProvenance(data)
        prov.record("m1", "repo/m1", str(model))

        (model / "config.json").unlink()
        v = prov.verify("m1", str(model))
        self.assertEqual(v["status"], "mismatch")
        self.assertIn("missing", v["mismatches"][0])

    def test_symlink_handling(self) -> None:
        data = Path(self.tmpdir) / "data"
        data.mkdir()
        model = Path(self.tmpdir) / "model"
        model.mkdir()
        blob = Path(self.tmpdir) / "blob"
        blob.write_text("shared-weights")

        # HF-style symlink layout.
        (model / "config.json").write_text("{}")
        (model / "weights.safetensors").symlink_to(blob)

        prov = ModelProvenance(data)
        prov.record("m1", "repo/m1", str(model))
        v = prov.verify("m1", str(model))
        self.assertEqual(v["status"], "verified")

        # Tamper the blob via the symlink target.
        blob.write_text("corrupt")
        v2 = prov.verify("m1", str(model))
        self.assertEqual(v2["status"], "mismatch")

    def test_rejects_path_traversal_model_id(self) -> None:
        prov = ModelProvenance(Path(self.tmpdir))
        with self.assertRaises(ValueError):
            prov.record("../etc/passwd", "repo/m", "/tmp")

    def test_model_id_with_slash_is_rejected(self) -> None:
        prov = ModelProvenance(Path(self.tmpdir))
        with self.assertRaises(ValueError):
            prov._manifest_path("a/b")

    def test_manifest_round_trip(self) -> None:
        prov = ModelProvenance(Path(self.tmpdir))
        manifest = ModelManifest(
            repo_id="r",
            local_path="/tmp/m",
            recorded_at=0.0,
            files={
                "a.json": FileEntry(relative="a.json", size=3, mtime=1.0, sha256="x" * 64),
            },
        )
        prov._save_manifest("m1", manifest)
        loaded = prov._load_manifest(prov._manifest_path("m1"))
        self.assertEqual(loaded.repo_id, "r")
        self.assertEqual(loaded.files["a.json"].sha256, "x" * 64)

    def test_unknown_status_when_no_manifest(self) -> None:
        prov = ModelProvenance(Path(self.tmpdir))
        model = Path(self.tmpdir) / "model"
        model.mkdir()
        (model / "x").write_text("y")
        v = prov.verify("m1", str(model))
        self.assertEqual(v["status"], "unknown")

    def test_empty_model_is_recorded(self) -> None:
        data = Path(self.tmpdir) / "data"
        data.mkdir()
        model = Path(self.tmpdir) / "model"
        model.mkdir()
        prov = ModelProvenance(data)
        r = prov.record("m1", "repo/m1", str(model))
        self.assertEqual(r["status"], "recorded")
        self.assertEqual(r["files"], 0)


if __name__ == "__main__":
    unittest.main()
