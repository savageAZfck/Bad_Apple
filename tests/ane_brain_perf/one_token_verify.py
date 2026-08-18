#!/usr/bin/env python3
"""One-token live ANE verification: run a prompt through badapple CLI and print the first token."""
import subprocess, shlex, sys

prompt = sys.argv[1] if len(sys.argv) > 1 else "Explain quantum computing in one sentence."
out = subprocess.check_output(["./target/release/badapple", prompt, "--max-tokens", "1"], text=True)
print(f"prompt: {prompt!r}")
print(f"first generated token text: {out.strip()!r}")
