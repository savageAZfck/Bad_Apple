#!/usr/bin/env python3
"""Interactive local backup, restore, generation, and provenance CLI."""

import argparse
import getpass
import json
import os
from pathlib import Path

import badapple_identity
from badapple_vault import ArtifactManifest, EncryptedBackup, GenerationStore


def _data_dir() -> Path:
    return Path(os.environ.get("BADAPPLE_DATA_DIR", "/var/lib/bad_apple")).expanduser()


def main() -> int:
    parser = argparse.ArgumentParser(prog="badapple-vault")
    sub = parser.add_subparsers(dest="command", required=True)

    backup = sub.add_parser("backup")
    backup.add_argument("output", type=Path)
    backup.add_argument("--source", type=Path, default=None)

    restore = sub.add_parser("extract")
    restore.add_argument("archive", type=Path)
    restore.add_argument("destination", type=Path)

    snapshot = sub.add_parser("snapshot")
    snapshot.add_argument("name")
    snapshot.add_argument("files", nargs="+", type=Path)

    verify_generation = sub.add_parser("verify-generation")
    verify_generation.add_argument("generation_id")

    manifest = sub.add_parser("manifest")
    manifest.add_argument("output", type=Path)
    manifest.add_argument("artifacts", nargs="+", type=Path)

    verify_manifest = sub.add_parser("verify-manifest")
    verify_manifest.add_argument("manifest", type=Path)

    args = parser.parse_args()
    data_dir = _data_dir()

    if args.command == "backup":
        passphrase = getpass.getpass("Backup passphrase: ")
        confirmation = getpass.getpass("Confirm passphrase: ")
        if passphrase != confirmation:
            raise SystemExit("passphrases do not match")
        result = EncryptedBackup.create(args.source or data_dir, args.output, passphrase)
    elif args.command == "extract":
        passphrase = getpass.getpass("Backup passphrase: ")
        result = EncryptedBackup.extract(args.archive, args.destination, passphrase)
    elif args.command == "snapshot":
        files = {path.name: path for path in args.files}
        generation_id = GenerationStore(data_dir).commit(args.name, files)
        result = GenerationStore(data_dir).verify(generation_id)
    elif args.command == "verify-generation":
        result = GenerationStore(data_dir).verify(args.generation_id)
    elif args.command == "manifest":
        manifest_data = ArtifactManifest.create(args.artifacts)
        manifest_data = ArtifactManifest.seal(
            manifest_data,
            badapple_identity.ensure_identity(),
            badapple_identity.sign(ArtifactManifest.signing_payload(manifest_data)),
        )
        args.output.write_text(json.dumps(manifest_data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        result = {"path": str(args.output), "verification": ArtifactManifest.verify(manifest_data)}
    else:
        loaded = json.loads(args.manifest.read_text(encoding="utf-8"))
        result = ArtifactManifest.verify(loaded)

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("valid", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
