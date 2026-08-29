#!/usr/bin/env python3
"""Reference client for the Bad Apple local agent protocol (LAP).

Usage:
    python3 agent_client.py discover
    python3 agent_client.py invoke run_shell '{"command":"ls"}'
    python3 agent_client.py workspace /path/to/workspace
    python3 agent_client.py infer 'What is 2+2?'

This is a reference / test client, not the production `badapple` CLI.
"""

import base64
import json
import os
import random
import socket
import sys
import time
from pathlib import Path

import badapple_slicks

DEFAULT_SOCKET_PATH = "/var/run/badapple/substrate_mlx.sock"
DEFAULT_KEY_PATH = badapple_slicks.DEFAULT_KEY_PATH


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

    use_v2 = os.environ.get("BADAPPLE_SLICKS2") == "1" and badapple_slicks.v2_available()
    if use_v2:
        client_version = badapple_slicks.SLICKS_VERSION_2
        client_pubkey_b64 = badapple_slicks.v2_public_key_b64() or ""
    else:
        client_version = badapple_slicks.SLICKS_VERSION
        client_pubkey_b64 = ""

    hello = {
        "type": "hello",
        "version": client_version,
        "timestamp_ms": timestamp_ms,
        "client_nonce": client_nonce,
        "client_pubkey": client_pubkey_b64,
    }
    send_frame(sock, hello)
    resp = recv_frame(sock)
    if resp.get("type") != "challenge":
        raise RuntimeError(f"unexpected challenge: {resp}")
    server_nonce = resp["server_nonce"]
    server_version = resp.get("version", client_version)

    if server_version == badapple_slicks.SLICKS_VERSION:
        secret = load_secret()
        if not badapple_slicks.v1_verify_server_proof(secret, timestamp_ms, client_nonce, server_nonce, resp["proof"]):
            raise RuntimeError("SLICKS v1 server authentication failed")
        proof = badapple_slicks.v1_client_proof(secret, timestamp_ms, client_nonce, server_nonce, prompt, max_tokens)
    else:
        server_pubkey_b64 = resp.get("server_pubkey")
        server_pubkey = base64.b64decode(server_pubkey_b64) if server_pubkey_b64 else None
        if not badapple_slicks.v2_verify_server_proof(timestamp_ms, client_nonce, server_nonce, resp["proof"], server_pubkey):
            raise RuntimeError("SLICKS v2 server authentication failed")
        proof = badapple_slicks.v2_client_proof(timestamp_ms, client_nonce, server_nonce, prompt, max_tokens)

    execute = {
        "type": "execute",
        "version": server_version,
        "timestamp_ms": timestamp_ms,
        "client_nonce": client_nonce,
        "server_nonce": server_nonce,
        "prompt": prompt,
        "max_new_tokens": max_tokens,
        "client_pubkey": client_pubkey_b64,
        "proof": proof,
    }
    send_frame(sock, execute)
    accepted = recv_frame(sock)
    if accepted.get("type") != "accepted":
        raise RuntimeError(f"not accepted: {accepted}")

    return recv_frame(sock)


def main():
    if len(sys.argv) < 2:
        print("Usage: agent_client.py <discover | invoke <tool> <json-args> | workspace <path> | infer <prompt> | status | kill | resume | private <on|off> | airgap [on|off] | identity | p2p_sync | p2p_peers | p2p <on|off|peers|sync> | audit <checkpoint|verify> | flush | unload <vision|image|all> | dashboard [--open] | work <read|write|clear> [content] | tier <on|off> | autopilot <on|off> | ambient <on|off> | agent <list | create '<goal>' [max_steps] | status <task_id> | cancel <task_id> | pause <task_id> | resume <task_id>> | model <list | recommend [query] | switch <id> | preload [id,id,...] | admit <id>>>")
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
    elif cmd == "airgap":
        if len(sys.argv) == 2:
            print(json.dumps(call_agent("airgap_status"), indent=2))
        elif sys.argv[2] in ("on", "off"):
            print(json.dumps(call_agent("set_airgap", {"enabled": sys.argv[2] == "on"}), indent=2))
        else:
            print("Usage: agent_client.py airgap [on|off]")
            return 1
    elif cmd == "identity":
        print(json.dumps(call_agent("identity_status"), indent=2))
    elif cmd == "p2p_sync":
        print(json.dumps(call_agent("p2p_sync"), indent=2))
    elif cmd == "p2p_peers":
        print(json.dumps(call_agent("p2p_peers"), indent=2))
    elif cmd == "audit":
        if len(sys.argv) < 3 or sys.argv[2] not in ("checkpoint", "verify"):
            print("Usage: agent_client.py audit <checkpoint|verify>")
            return 1
        print(json.dumps(call_agent(f"audit_{sys.argv[2]}"), indent=2))
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
    elif cmd == "agent":
        if len(sys.argv) < 3:
            print("Usage: agent_client.py agent <list | create '<goal>' [max_steps] | status <task_id> | cancel <task_id> | pause <task_id> | resume <task_id>>")
            return 1
        sub = sys.argv[2]
        if sub == "list":
            print(json.dumps(call_agent("list_agent_tasks"), indent=2))
        elif sub == "create":
            if len(sys.argv) < 4:
                print("Usage: agent_client.py agent create '<goal>' [max_steps]")
                return 1
            goal = sys.argv[3]
            max_steps = int(sys.argv[4]) if len(sys.argv) > 4 else 10
            print(json.dumps(call_agent("run_agent_task", {"goal": goal, "max_steps": max_steps}), indent=2))
        elif sub == "status":
            if len(sys.argv) < 4:
                print("Usage: agent_client.py agent status <task_id>")
                return 1
            print(json.dumps(call_agent("get_agent_task", {"task_id": sys.argv[3]}), indent=2))
        elif sub == "cancel":
            if len(sys.argv) < 4:
                print("Usage: agent_client.py agent cancel <task_id>")
                return 1
            print(json.dumps(call_agent("cancel_agent_task", {"task_id": sys.argv[3]}), indent=2))
        elif sub == "pause":
            if len(sys.argv) < 4:
                print("Usage: agent_client.py agent pause <task_id>")
                return 1
            print(json.dumps(call_agent("pause_agent_task", {"task_id": sys.argv[3]}), indent=2))
        elif sub == "resume":
            if len(sys.argv) < 4:
                print("Usage: agent_client.py agent resume <task_id>")
                return 1
            print(json.dumps(call_agent("resume_agent_task", {"task_id": sys.argv[3]}), indent=2))
        else:
            print(f"Unknown agent command: {sub}")
            return 1
    elif cmd == "model":
        if len(sys.argv) < 3:
            print("Usage: agent_client.py model <list | recommend [query] | switch <id> | preload [id,id,...] | admit <id>>")
            return 1
        sub = sys.argv[2]
        if sub == "list":
            print(json.dumps(call_agent("model_status"), indent=2))
        elif sub == "recommend":
            query = " ".join(sys.argv[3:])
            print(json.dumps(call_agent("recommend_model", {"query": query}), indent=2))
        elif sub == "switch":
            if len(sys.argv) < 4:
                print("Usage: agent_client.py model switch <id or repo>")
                return 1
            print(json.dumps(call_agent("switch_main_model", {"model_ref": sys.argv[3]}), indent=2))
        elif sub == "preload":
            ids = [m.strip() for m in sys.argv[3].split(",")] if len(sys.argv) > 3 else []
            print(json.dumps(call_agent("preload_models", {"model_ids": ids}), indent=2))
        elif sub == "admit":
            if len(sys.argv) < 4:
                print("Usage: agent_client.py model admit <id or repo>")
                return 1
            print(json.dumps(call_agent("admit_model", {"model_ref": sys.argv[3]}), indent=2))
        else:
            print(f"Unknown model command: {sub}")
            return 1
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
