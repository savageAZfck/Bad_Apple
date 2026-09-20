# Mesh-Brain — one model, several Macs

Bad Apple can split a single AI model across trusted Macs and run it as one
brain. Each machine holds a slice of the model's layers. When you ask a
question, the partial "thought" (a hidden activation) passes from machine to
machine over an authenticated, AES-256-GCM encrypted link — once per token —
until the last machine produces words.

**Why:** a model that doesn't fit on your Mac might fit on your Mac *and* your
friend's Mac. A maxed-out Studio already carries 671B alone — mesh-brain is
how a crew of smaller Macs pools memory into the same league, and how the
brain keeps thinking when a box drops out.

## The one rule

Every rank must share the same key — at least 32 characters:

```bash
export BADAPPLE_MESH_KEY="some-long-secret-at-least-32-characters"
```

Set it on every machine. Wrong key = rejected at the handshake. No key = no
mesh. This key is also what encrypts the traffic, so keep it private.

## Try it on one Mac (5 minutes)

Split the small 0.5B model across two processes on the same machine:

```bash
# 1. Plan the split — this remembers the mesh for later
badapple mesh-brain plan --model mlx-community/Qwen2.5-0.5B-Instruct-4bit \
    --hosts 127.0.0.1:8741,127.0.0.1:8742

# 2. Build the two shard directories
badapple mesh-brain shard --model mlx-community/Qwen2.5-0.5B-Instruct-4bit \
    --rank 0 --of 2 --hosts 127.0.0.1:8741,127.0.0.1:8742 --out /tmp/r0
badapple mesh-brain shard --model mlx-community/Qwen2.5-0.5B-Instruct-4bit \
    --rank 1 --of 2 --hosts 127.0.0.1:8741,127.0.0.1:8742 --out /tmp/r1

# 3. Start a rank engine for each shard (two terminals, or nohup)
BADAPPLE_SHARD_DIR=/tmp/r0 badapple-engine
BADAPPLE_SHARD_DIR=/tmp/r1 badapple-engine

# 4. Check it and ask it something
badapple mesh-brain status
badapple mesh-brain ask --prompt "The capital of France is" --max-tokens 8
```

Rank 0 is the driver — you always talk to it. It embeds the tokens, hands the
activations downstream, and the last rank owns the output head.

## Two real Macs

Same thing, except `--hosts` gets real LAN addresses:

```bash
badapple mesh-brain plan --model <model> --hosts 192.168.1.10:8741,192.168.1.11:8742
```

Shard each rank, copy each shard dir to its machine, start an engine with
`BADAPPLE_SHARD_DIR` on each, same `BADAPPLE_MESH_KEY` on both. Done.

## What happens when things go wrong

- **A rank crashes** — the pipeline fails fast (no silent hang), reports
  itself degraded, and heals when the rank rejoins. Ask again and it answers.
- **Wrong key** — rejected during the handshake, before anything is sent.
- **Someone tampers with a frame** — AES-GCM authentication fails, the frame
  is dropped.
- **A rank goes silent but stays alive** (network partition) — the watchdog
  timeout (`BADAPPLE_MESH_TIMEOUT`, default 120s) bounds the wait, then the
  pipeline resets and errors honestly.

## Remembering the mesh

`plan` and `shard` save the host list to `/var/lib/bad_apple/mesh_hosts.json`,
so `status`/`ping`/`ask` work without retyping `--hosts`, and Bad Apple can
answer "is the mesh up?" herself. `badapple mesh-brain forget` clears it.

## The knobs

| env var | what it does |
|---|---|
| `BADAPPLE_MESH_KEY` | shared secret (≥32 chars) — auth + encryption |
| `BADAPPLE_SHARD_DIR` | puts `badapple-engine` into rank mode serving that shard |
| `BADAPPLE_MESH_TIMEOUT` | watchdog timeout in seconds (default 120) |
| `BADAPPLE_MESH_AUTH=0` | disables authentication — debugging only |
| `BADAPPLE_MESH_ENC=0` | disables frame encryption — debugging only |
| `BADAPPLE_MESH_HOSTS_FILE` | override the saved-mesh registry path |

Nothing here phones home. The mesh only talks to the hosts you name, with the
key you chose.
