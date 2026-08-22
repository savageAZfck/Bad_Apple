# Bad Apple — Bare-Metal AI OS Standards

This document defines the public interfaces and formats of the Bad Apple bare-metal AI operating system. Implementations that follow this standard can interoperate with Bad Apple and with each other.

---

## 1. SLICKS — Secure Local Identity & Key Exchange

SLICKS is the local authentication protocol. It ensures that only code running on the same device can talk to the Bad Apple daemon.

### 1.1 Identity

- The client and daemon each hold a SLICKS key pair stored in the device keychain.
- The daemon's public key is pinned at install time.
- The client proves possession of its private key in the handshake.

### 1.2 Handshake (one-shot per connection)

```json
{
  "type": "hello",
  "version": "slicks/1.0",
  "timestamp_ms": 1724371200000,
  "client_nonce": "<16 random bytes, base64>",
  "client_pubkey": "<base64 public key>",
  "signature": "<signature of (version|timestamp_ms|client_nonce|client_pubkey)>"
}
```

The daemon responds with:

```json
{
  "type": "hello",
  "version": "slicks/1.0",
  "server_nonce": "<16 random bytes, base64>",
  "signature": "<signature of (client_nonce|server_nonce)>"
}
```

### 1.3 Transport

- After the handshake, all frames are length-prefixed JSON lines over a Unix domain socket.
- The socket path is `/var/run/badapple/substrate_mlx.sock` by default.
- No TCP listener is permitted in a certified bare-metal build.

---

## 2. Local Agent Protocol (LAP)

LAP is a minimal JSON-RPC protocol that lets other local apps use Bad Apple as a cognitive service.

### 2.1 Transport

Same as SLICKS: Unix socket, length-prefixed JSON lines. Every request must include a request ID and a SLICKS handshake must precede method calls.

### 2.2 Request envelope

```json
{
  "id": "req-1",
  "method": "inference",
  "params": {
    "prompt": "what is the capital of France?",
    "max_new_tokens": 240,
    "persona": "default",
    "voice": false,
    "stream": false
  }
}
```

### 2.3 Response envelope

```json
{
  "id": "req-1",
  "type": "response",
  "text": "Paris, babe.",
  "metrics": {
    "tokens": 4,
    "decode_tps": 18.5,
    "peak_memory_gb": 6.2
  }
}
```

### 2.4 Core methods

| Method | Description |
|---|---|
| `inference` | Run a single-turn inference. |
| `inference_stream` | Stream token/sentence chunks. |
| `invoke_tool` | Call a local tool directly. |
| `discover_tools` | Return the list of available tools and their schemas. |
| `query_memory` | Query the local knowledge/memory index. |
| `set_workspace` | Set the active project directory. |
| `switch_persona` | Change the active persona. |
| `propose_approve` | Propose a destructive tool and return the proposal ID. |
| `approve` | Approve a previously proposed tool. |
| `audit_tail` | Return the last N ledger entries. |

### 2.5 Streaming

For `inference_stream`, each chunk is a frame:

```json
{"id": "req-2", "type": "token", "text": "Paris, "}
{"id": "req-2", "type": "token", "text": "babe. "}
{"id": "req-2", "type": "done", "text": "Paris, babe.", "metrics": {...}}
```

---

## 3. Policy Schema (`policy.yaml`)

The policy file is the declarative cage. It is human-editable and enforced by the OS.

### 3.1 Top-level fields

```yaml
policy_version: "1.0"
autopilot: false
defaults:
  allowed: true
  require_approval: true
tools:
  <tool_name>:
    allowed: true | false
    require_approval: true | false
    <tool-specific rules>
```

### 3.2 Common tool rules

| Rule | Type | Meaning |
|---|---|---|
| `allowed` | bool | Whether the tool may be used at all. |
| `require_approval` | bool | Whether the tool requires a human approve step. |
| `allowed_paths` | list | Paths the tool may read/write under. Empty means unrestricted. |
| `denied_patterns` | list | Substrings that cause immediate denial. |

### 3.3 Tool-specific rules

- `run_shell`: `allowed_commands`, `denied_patterns`, `max_timeout`
- `run_applescript`: `allowed_apps`, `denied_patterns`, `max_timeout`
- `write_file`: `notes_dir`, `denied_patterns`
- `index_documents`: `allowed_paths`
- `read_file`: `max_size`

---

## 4. Audit Ledger Format

The ledger is a JSON-Lines file where every line is an entry and every entry is hash-chained.

### 4.1 Entry format

```json
{
  "timestamp": "2026-08-22T23:00:00Z",
  "event": "tool_execute",
  "data": {"tool": "write_file", "args": {"filename": "note.txt"}},
  "previous_hash": "sha256-of-previous-entry",
  "hash": "sha256-of-this-entry"
}
```

### 4.2 Hash computation

```python
payload = canonical_json({
    "timestamp": entry.timestamp,
    "event": entry.event,
    "data": entry.data,
    "previous_hash": entry.previous_hash,
})
entry.hash = sha256(payload).hexdigest()
```

### 4.3 Genesis

The first entry's `previous_hash` is the SHA-256 of the genesis string `bad-apple-genesis-v1`.

---

## 5. Persona Pack Format (`personas.json`)

A persona pack is a JSON object where each key is a persona name.

### 5.1 Persona object

```json
{
  "name": "Wicket",
  "description": "A quick-witted London AI.",
  "system_prompt": "...",
  "voice_system_prompt": "...",
  "roast_bank": ["..."]
}
```

### 5.2 Default persona

The default persona is loaded from `prompt.txt` in the repo root. Alternative personas override it via `personas.json`.

---

## 6. Output Firewall

The streaming firewall sits between the model and the user. It scans generated text using Aho-Corasick for banned patterns.

### 6.1 Banned pattern categories

- PII: emails, phone numbers, physical addresses
- Secrets: API keys, tokens, passwords, private keys
- Disallowed content: configured per build

### 6.2 Behavior

When a banned pattern is detected, the stream is replaced with a redaction marker and the match is logged to the audit ledger. The model is not informed of the block.

---

## 7. Certification Suite

A bare-metal AI OS build must pass `cert_suite.py`.

### 7.1 Required checks

1. No external network sockets held by badapple processes.
2. No TCP listeners; only Unix sockets.
3. A valid `policy.yaml` is loaded.
4. The audit ledger hash chain is intact (if present).
5. No unredacted secret patterns in logs or data.
6. Model weights are stored locally.
7. No hard-coded third-party cloud endpoints in source.

### 7.2 Running the suite

```bash
sudo /path/to/venv/bin/python cert_suite.py
```

Root is preferred for full socket/process visibility, but many checks run unprivileged.
