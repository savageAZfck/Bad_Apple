#!/usr/bin/env python3
"""Reference client for the Bad Apple local agent protocol (LAP).

Usage:
    python3 agent_client.py discover
    python3 agent_client.py invoke run_shell '{"command":"ls"}'
    python3 agent_client.py workspace /Users/savag3/bad_apple
    python3 agent_client.py infer 'What is 2+2?'

This is a reference / test client, not the production `badapple` CLI.
"""

import hashlib
import hmac
import json
import os
import random
import socket
import sys
import time
from pathlib import Path

SLICKS_VERSION = 1
DEFAULT_SOCKET_PATH = "/var/run/badapple/substrate_mlx.sock"
DEFAULT_KEY_PATH = "/var/lib/bad_apple/slicks.key"


def load_secret() -> bytes:
    if "BADAPPLE_SLICKS_SECRET" in os.environ:
        return os.environ["BADAPPLE_SLICKS_SECRET"].encode()
    key_path = os.environ.get("BADAPPLE_SLICKS_KEY_PATH", DEFAULT_KEY_PATH)
    if Path(key_path).is_file():
        raw = Path(key_path).read_text(encoding="utf-8").strip()
        if all(c in "0123456789abcdefABCDEF" for c in raw) and len(raw) >= 32:
            return bytes.fromhex(raw)
        return raw.encode()
    raise RuntimeError("No SLICKS secret found")


def sign(secret: bytes, material: bytes) -> str:
    return hmac.new(secret, material, hashlib.sha256).hexdigest()


def client_material(timestamp_ms, client_nonce, server_nonce, prompt, max_new_tokens):
    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
    return f"BADAPPLE-SLICKS/{SLICKS_VERSION}|client|{timestamp_ms}|{client_nonce}|{server_nonce}|{max_new_tokens}|{prompt_hash}".encode()


def random_nonce() -> str:
    return "".join(f"{random.randrange(16):x}" for _ in range(64))


def send_frame(sock, obj: dict):
    line = json.dumps(obj, ensure_ascii=True).encode() + b"\n"
    sock.sendall(line)


def recv_frame(sock) -> dict:
    f = sock.makefile("r")
    line = f.readline()
    if not line:
        raise RuntimeError("server closed connection")
    return json.loads(line)


def call_agent(method: str, params: dict | None = None, prompt_text: str | None = None, max_tokens: int = 1):
    secret = load_secret()
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(os.environ.get("BADAPPLE_SOCKET_PATH", DEFAULT_SOCKET_PATH))

    timestamp_ms = int(time.time() * 1000)
    client_nonce = random_nonce()

    if prompt_text is not None:
        prompt = prompt_text
    else:
        req = {"id": "cli-1", "method": method}
        if params:
            req["params"] = params
        prompt = "__BADAPPLE_AGENT__ " + json.dumps(req)

    hello = {
        "type": "hello",
        "version": SLICKS_VERSION,
        "timestamp_ms": timestamp_ms,
        "client_nonce": client_nonce,
        "client_pubkey": "",
        "signature": sign(secret, f"slicks/{SLICKS_VERSION}/{timestamp_ms}/{client_nonce}".encode()),
    }
    # The server currently only checks type/version/timestamp/nonce.
    send_frame(sock, hello)
    resp = recv_frame(sock)
    if resp.get("type") != "challenge":
        raise RuntimeError(f"unexpected challenge: {resp}")
    server_nonce = resp["server_nonce"]

    proof = sign(secret, client_material(timestamp_ms, client_nonce, server_nonce, prompt, max_tokens))
    execute = {
        "type": "execute",
        "version": SLICKS_VERSION,
        "timestamp_ms": timestamp_ms,
        "client_nonce": client_nonce,
        "server_nonce": server_nonce,
        "prompt": prompt,
        "max_new_tokens": max_tokens,
        "proof": proof,
    }
    send_frame(sock, execute)
    accepted = recv_frame(sock)
    if accepted.get("type") != "accepted":
        raise RuntimeError(f"not accepted: {accepted}")

    return recv_frame(sock)


def main():
    if len(sys.argv) < 2:
        print("Usage: agent_client.py <discover | invoke <tool> <json-args> | workspace <path> | infer <prompt> | status | kill | resume | private <on|off> | identity | p2p_sync | p2p_peers | p2p <on|off|peers|sync> | flush | unload <vision|image|all> | dashboard [--open] | work <read|write|clear> [content] | tier <on|off> | autopilot <on|off> | ambient <on|off>>")
        return 1

    cmd = sys.argv[1]
    if cmd == "discover":
        print(json.dumps(call_agent("discover_tools"), indent=2))
    elif cmd == "invoke":
        if len(sys.argv) < 4:
            print("Usage: agent_client.py invoke <tool> <json-args>")
            return 1
        tool = sys.argv[2]
        args = json.loads(sys.argv[3])
        print(json.dumps(call_agent("invoke_tool", {"name": tool, "args": args}), indent=2))
    elif cmd == "workspace":
        if len(sys.argv) < 3:
            print("Usage: agent_client.py workspace <path>")
            return 1
        print(json.dumps(call_agent("set_workspace", {"path": sys.argv[2]}), indent=2))
    elif cmd == "infer":
        prompt = " ".join(sys.argv[2:])
        print(json.dumps(call_agent("inference", {"prompt": prompt, "max_new_tokens": 120}), indent=2))
    elif cmd == "status":
        print(json.dumps(call_agent("runtime_status"), indent=2))
    elif cmd == "kill":
        print(json.dumps(call_agent("kill_switch", {"enabled": True, "reason": "CLI requested"}), indent=2))
    elif cmd == "resume":
        print(json.dumps(call_agent("kill_switch", {"enabled": False}), indent=2))
    elif cmd == "private":
        if len(sys.argv) != 3 or sys.argv[2] not in ("on", "off"):
            print("Usage: agent_client.py private <on|off>")
            return 1
        print(json.dumps(call_agent("private_mode", {"enabled": sys.argv[2] == "on"}), indent=2))
    elif cmd == "identity":
        print(json.dumps(call_agent("identity_status"), indent=2))
    elif cmd == "p2p_sync":
        print(json.dumps(call_agent("p2p_sync"), indent=2))
    elif cmd == "p2p_peers":
        print(json.dumps(call_agent("p2p_peers"), indent=2))
    elif cmd == "flush":
        print(json.dumps(call_agent("flush_vram"), indent=2))
    elif cmd == "unload":
        model_type = sys.argv[2] if len(sys.argv) > 2 else "vision"
        print(json.dumps(call_agent("unload_model", {"type": model_type}), indent=2))
    elif cmd == "work":
        if len(sys.argv) < 3:
            print("Usage: agent_client.py work <read|write|clear> [content]")
            return 1
        sub = sys.argv[2]
        if sub == "read":
            print(json.dumps(call_agent("invoke_tool", {"name": "read_working_memory", "args": {}}), indent=2))
        elif sub == "write":
            if len(sys.argv) < 4:
                print("Usage: agent_client.py work write <content>")
                return 1
            print(json.dumps(call_agent("invoke_tool", {"name": "write_working_memory", "args": {"content": " ".join(sys.argv[3:])}}), indent=2))
        elif sub == "clear":
            print(json.dumps(call_agent("invoke_tool", {"name": "clear_working_memory", "args": {}}), indent=2))
        else:
            print(f"Unknown work command: {sub}")
            return 1
    elif cmd == "tier":
        if len(sys.argv) < 3 or sys.argv[2] not in ("on", "off"):
            print("Usage: agent_client.py tier <on|off>")
            return 1
        print(json.dumps(call_agent("set_fast_tier", {"enabled": sys.argv[2] == "on"}), indent=2))
    elif cmd == "autopilot":
        if len(sys.argv) < 3 or sys.argv[2] not in ("on", "off"):
            print("Usage: agent_client.py autopilot <on|off>")
            return 1
        print(json.dumps(call_agent("set_autopilot", {"enabled": sys.argv[2] == "on"}), indent=2))
    elif cmd == "p2p":
        if len(sys.argv) < 3 or sys.argv[2] not in ("on", "off", "peers", "sync"):
            print("Usage: agent_client.py p2p <on|off|peers|sync>")
            return 1
        if sys.argv[2] in ("on", "off"):
            print(json.dumps(call_agent("set_p2p", {"enabled": sys.argv[2] == "on"}), indent=2))
        elif sys.argv[2] == "peers":
            print(json.dumps(call_agent("p2p_peers", {}), indent=2))
        else:
            print(json.dumps(call_agent("p2p_sync", {}), indent=2))
    elif cmd == "ambient":
        if len(sys.argv) < 3 or sys.argv[2] not in ("on", "off"):
            print("Usage: agent_client.py ambient <on|off>")
            return 1
        print(json.dumps(call_agent("set_ambient", {"enabled": sys.argv[2] == "on"}), indent=2))
    elif cmd == "dashboard":
        url = "http://127.0.0.1:8787/"
        print(url)
        if len(sys.argv) > 2 and sys.argv[2] == "--open":
            import webbrowser

            webbrowser.open(url)
    else:
        print(f"Unknown command: {cmd}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
