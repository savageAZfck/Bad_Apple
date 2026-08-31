# SLICKS Protocol Specification

## Overview

SLICKS (Secure Local IPC for Knowledge and Execution Services) is the authenticated Unix domain socket protocol used by Bad Apple for all communication between the CLI, menu bar app, and system daemons.

## Versions

| Version | Authentication | Key Material |
|---|---|---|
| v1 | HMAC-SHA256 challenge-response | Shared secret in /var/lib/bad_apple/slicks.key |
| v2 | ECDSA P-256 via Secure Enclave | Per-device key pair in Secure Enclave |

## Transport

- **Socket type**: Unix domain socket (SOCK_STREAM)
- **Default path**: /var/run/badapple/substrate.sock (gatekeeper), /var/run/badapple/substrate_mlx.sock (MLX daemon)
- **Permissions**: 0o660 (owner and group only)
- **Framing**: Newline-delimited JSON, max 1 MiB per frame

## Handshake Flow

### Version 1 (HMAC-SHA256)

```
Client                              Server
  |                                    |
  | --- Hello -----------------------> |
  |     {                              |
  |       type: "hello",               |
  |       version: 1,                  |
  |       timestamp_ms: <unix_ms>,     |
  |       client_nonce: <64 hex>       |
  |     }                              |
  |                                    |
  | <--- Challenge -------------------- |
  |     {                              |
  |       type: "challenge",           |
  |       version: 1,                  |
  |       server_nonce: <64 hex>,      |
  |       proof: <HMAC-SHA256 hex>     |
  |     }                              |
  |                                    |
  | --- Execute ---------------------> |
  |     {                              |
  |       type: "execute",             |
  |       version: 1,                  |
  |       timestamp_ms: <same>,        |
  |       client_nonce: <same>,        |
  |       server_nonce: <same>,        |
  |       prompt: <string>,            |
  |       max_new_tokens: <int>,       |
  |       proof: <HMAC-SHA256 hex>     |
  |     }                              |
  |                                    |
  | <--- Accepted --------------------- |
  | <--- Token { text: "..." } -------> |
  | <--- Token { text: "..." } -------> |
  | <--- Done { text, metrics } ------> |
  |                                    |
```

### Server Proof Material (v1)

```
BADAPPLE-SLICKS/1|server|{timestamp_ms}|{client_nonce}|{server_nonce}
```

### Client Proof Material (v1)

```
BADAPPLE-SLICKS/1|client|{timestamp_ms}|{client_nonce}|{server_nonce}|{max_new_tokens}|{SHA256(prompt)}
```

### Version 2 (Secure Enclave ECDSA)

Same flow, but:
- Hello includes `client_pubkey` (base64 P-256 uncompressed point)
- Challenge includes `server_pubkey` and `proof` is an ECDSA signature
- Execute proof is an ECDSA signature over the material
- Server public key is pinned via TOFU trust store at /var/lib/bad_apple/keys/daemon.pub
- Client public key is extracted from the Hello frame (not the Execute frame)

### V2 Server Proof Material

```
BADAPPLE-SLICKS/2|server|{timestamp_ms}|{client_nonce}|{server_nonce}
```

### V2 Client Proof Material

```
BADAPPLE-SLICKS/2|client|{timestamp_ms}|{client_nonce}|{server_nonce}|{max_new_tokens}|{SHA256(prompt)}
```

## Security Properties

| Property | Mechanism |
|---|---|
| Mutual authentication | Challenge-response with shared secret (v1) or ECDSA (v2) |
| Prompt binding | SHA256(prompt) included in client proof material |
| Token binding | max_new_tokens included in client proof material |
| Replay protection | Server maintains in-memory nonce cache (4096 entries) |
| Timestamp freshness | ±30 second window from server's wall clock |
| Nonce uniqueness | 32 random bytes via OsRng, validated as 64 hex chars |
| Constant-time comparison | HMAC verification uses verify_slice (constant-time) |
| Key pinning (v2) | TOFU trust store, first-connection pins the server pubkey |
| Frame size limit | 1 MiB max per frame, rejected if exceeded |

## Gatekeeper v2 Proxy

When v2 is used, the gatekeeper acts as a transparent proxy:
1. Receives client Hello, forwards to MLX daemon
2. Receives MLX Challenge, forwards to client
3. Receives client Execute, validates prompt and max_tokens, forwards to MLX
4. Streams MLX response frames back to client with automation cage post-processing

The gatekeeper does not hold a Secure Enclave key — the end-to-end authentication is between the client and the MLX daemon. The gatekeeper enforces request validation (validate_request) before forwarding.

## Error Frames

```json
{ "type": "error", "message": "human-readable error" }
```

Error messages do not leak sensitive information (no key material, no internal paths, no stack traces).

## Limits

| Limit | Value |
|---|---|
| MAX_PROMPT_BYTES | 65,536 (64 KiB) |
| MAX_NEW_TOKENS | 4,096 |
| MAX_FRAME_BYTES | 1,048,576 (1 MiB) |
| HANDSHAKE_MAX_SKEW | 30 seconds |
| Nonce length | 64 hex chars (32 bytes) |
| Replay cache size | 4,096 entries |
| Socket permissions | 0o660 |
