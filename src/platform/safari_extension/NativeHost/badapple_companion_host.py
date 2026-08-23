#!/usr/bin/env python3
"""Native messaging host for the Bad Apple Safari companion.

Reads JSON messages from stdin, writes JSON responses to stdout.
Connects to the local Bad Apple SLICKS socket for on-device answers.
"""

import json
import socket
import struct
import sys
from pathlib import Path

SOCKET_PATH = "/var/run/badapple/substrate.sock"


def read_message():
    raw = sys.stdin.buffer.read(4)
    if not raw:
        return None
    length = struct.unpack("=I", raw)[0]
    data = sys.stdin.buffer.read(length)
    return json.loads(data.decode("utf-8"))


def write_message(message):
    data = json.dumps(message).encode("utf-8")
    sys.stdout.buffer.write(struct.pack("=I", len(data)))
    sys.stdout.buffer.write(data)
    sys.stdout.buffer.flush()


def ask_badapple(text: str) -> str:
    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.connect(SOCKET_PATH)
        sock.sendall((text + "\n").encode("utf-8"))
        parts = []
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                break
            parts.append(chunk)
        return b"".join(parts).decode("utf-8", "ignore") or "No response."
    except Exception as e:
        return f"Bad Apple connection error: {e}"


def main():
    while True:
        msg = read_message()
        if msg is None:
            break
        prompt = msg.get("prompt", "")
        if msg.get("type") == "summarize_page" and not prompt:
            prompt = f"Summarize this page in three bullets:\nTitle: {msg.get('title','')}\nURL: {msg.get('url','')}\nText: {msg.get('text','')[:4000]}"
        answer = ask_badapple(prompt)
        write_message({"text": answer[:4000]})


if __name__ == "__main__":
    main()
