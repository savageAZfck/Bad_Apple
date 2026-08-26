#!/usr/bin/env python3
"""Native messaging host for the Bad Apple Safari companion.

Reads JSON messages from stdin, writes JSON responses to stdout.
Connects to the local Bad Apple SLICKS socket for on-device answers.
"""

import json
import struct
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent_client import call_agent


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
        response = call_agent("inference", {"prompt": text, "max_new_tokens": 320})
        if response.get("type") == "error":
            return f"Bad Apple connection error: {response.get('message', response)}"
        result = response.get("result", {})
        return result.get("text") or response.get("text") or "No response."
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
        elif msg.get("type") == "query":
            page = msg.get("page") or {}
            prompt = (
                f"Answer the question using only this local page context.\n"
                f"Question: {prompt}\nTitle: {page.get('title', '')}\n"
                f"URL: {page.get('url', '')}\nText: {page.get('text', '')[:6000]}"
            )
        answer = ask_badapple(prompt)
        write_message({"text": answer[:4000]})


if __name__ == "__main__":
    main()
