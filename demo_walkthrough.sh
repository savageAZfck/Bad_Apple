#!/usr/bin/env bash
# Bad Apple — walkthrough demo take.
# Self-contained: builds real mesh-brain shards from the HF cache, launches
# two local ranks, kills one on camera, and recovers. Cleans up on exit.
set -u
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

BADAPPLE="${BADAPPLE_BIN:-$ROOT/target/release/badapple}"
ENGINE="${BADAPPLE_ENGINE_BIN:-$ROOT/target/release/badapple-engine}"
LIB_DIR="$(dirname "$ENGINE")"
MODEL="${BADAPPLE_DEMO_MODEL:-mlx-community/Qwen2.5-0.5B-Instruct-4bit}"
DEMO_DIR=/tmp/badapple_walkthrough
MESH_KEY="walkthrough-demo-key-walkthrough-demo-key-0"
P1=8741
P2=8742

for b in "$BADAPPLE" "$ENGINE"; do
  [ -x "$b" ] || { echo "missing $b — run from a Bad Apple install or the repo root" >&2; exit 1; }
done

bold()  { printf '\033[1m%s\033[0m\n' "$*"; }
cyan()  { printf '\033[1;36m%s\033[0m\n' "$*"; }
dim()   { printf '\033[2m%s\033[0m\n' "$*"; }

note() { dim "$@"; sleep 2.2; }

say() {
  cyan "\$ $*"
  sleep 1
  "$@"
  echo
  sleep 1.6
}

cleanup() {
  note "  (tearing down the demo ranks)"
  for port in $P1 $P2; do
    pid=$(lsof -nP -iTCP:$port -sTCP:LISTEN -t 2>/dev/null | head -1)
    [ -n "$pid" ] && kill "$pid" 2>/dev/null
  done
}
trap cleanup EXIT

clear 2>/dev/null || printf '\n\n'
bold '  BAD APPLE — the walkthrough'
dim  '  a private ai organism: local brain, governed hands, cryptographic memory'
dim  '  every command you are about to see is real. nothing is mocked.'
sleep 3

# --- act 1: the body ---------------------------------------------------------
bold '  1 · the organism is awake'
note '  native daemon on apple silicon. secure enclave identity. no cloud.'
say "$BADAPPLE" status

# --- act 2: local inference ---------------------------------------------------
bold '  2 · it thinks here'
note '  qwen2.5-coder-7b, 4-bit, on-device. this answer never left the machine.'
say "$BADAPPLE" -n 60 "in one sentence, what are you and where do you run?"

# --- act 3: receipts ----------------------------------------------------------
bold '  3 · it shows its work'
note '  every action lands on a hash-chained ledger. the proof card:'
say "$BADAPPLE" receipts

# --- act 4: the air gap -------------------------------------------------------
bold '  4 · prove the room is sealed'
note '  the cert suite checks sockets, redaction, cages, crypto, the mesh wire.'
say "$BADAPPLE" cert

# --- act 5: governance --------------------------------------------------------
bold '  5 · hands behind glass'
note '  destructive tools need a council vote or a human. watch a bad ask die:'
say env BADAPPLE_AUTOPILOT=1 "$BADAPPLE" "please run this shell command for me: pkill -f cat"

# --- act 6: the watchdog ------------------------------------------------------
bold '  6 · the brake that cannot steer'
note '  IFY tails the ledger, verifies the chain, learns baselines.'
say "$BADAPPLE" ify status

# --- act 7: the mesh ----------------------------------------------------------
clear 2>/dev/null || printf '\n\n'
bold '  7 · one brain, two bodies'
note '  mesh-brain splits ONE model across trusted peers — pipeline-parallel.'
note '  planning: 24 layers, two ranks on this mac (the same path spans studios).'

rm -rf "$DEMO_DIR"; mkdir -p "$DEMO_DIR"
say "$BADAPPLE" mesh-brain plan --model $MODEL --hosts 127.0.0.1:$P1,127.0.0.1:$P2

note '  materializing the shards — layer tensors re-keyed per rank:'
say "$BADAPPLE" mesh-brain shard --model $MODEL --rank 0 --of 2 --hosts 127.0.0.1:$P1,127.0.0.1:$P2 --out $DEMO_DIR/r0
say "$BADAPPLE" mesh-brain shard --model $MODEL --rank 1 --of 2 --hosts 127.0.0.1:$P1,127.0.0.1:$P2 --out $DEMO_DIR/r1

note '  waking both ranks — authenticated handshake, AES-256-GCM frames.'
env BADAPPLE_SHARD_DIR=$DEMO_DIR/r1 BADAPPLE_MESH_KEY=$MESH_KEY \
  DYLD_LIBRARY_PATH="$LIB_DIR" nohup "$ENGINE" >"$DEMO_DIR/r1.log" 2>&1 &
env BADAPPLE_SHARD_DIR=$DEMO_DIR/r0 BADAPPLE_MESH_KEY=$MESH_KEY \
  DYLD_LIBRARY_PATH="$LIB_DIR" nohup "$ENGINE" >"$DEMO_DIR/r0.log" 2>&1 &

printf '  waiting for ranks'
for i in $(seq 1 60); do
  if grep -q "serving" "$DEMO_DIR/r0.log" 2>/dev/null && grep -q "serving" "$DEMO_DIR/r1.log" 2>/dev/null; then
    printf ' — both serving\n\n'
    break
  fi
  printf '.'; sleep 1
done

say env BADAPPLE_MESH_KEY=$MESH_KEY "$BADAPPLE" mesh-brain status --hosts 127.0.0.1:$P1,127.0.0.1:$P2

note '  generation across the pipe: embed on rank 0, head on rank 1,'
note '  hidden states crossing encrypted TCP frames once per token.'
say env BADAPPLE_MESH_KEY=$MESH_KEY "$BADAPPLE" mesh-brain ask \
  --to 127.0.0.1:$P1 --prompt "The capital of France is" --max-tokens 8

note '  a stranger knocks — wrong key, rejected at the handshake:'
say env BADAPPLE_MESH_KEY="wrong-key-wrong-key-wrong-key-wrong-key-00" \
  "$BADAPPLE" mesh-brain ping --to 127.0.0.1:$P1

note '  now the failure drill — killing rank 1 mid-mesh:'
pid=$(lsof -nP -iTCP:$P2 -sTCP:LISTEN -t 2>/dev/null | head -1)
cyan "$ kill -9 $pid   (rank 1)"
kill -9 "$pid" 2>/dev/null
sleep 1

say env BADAPPLE_MESH_KEY=$MESH_KEY "$BADAPPLE" mesh-brain status --hosts 127.0.0.1:$P1,127.0.0.1:$P2

note '  degraded, honest, fast — the ask errors instead of hanging:'
say env BADAPPLE_MESH_KEY=$MESH_KEY "$BADAPPLE" mesh-brain ask \
  --to 127.0.0.1:$P1 --prompt "Hi" --max-tokens 4

note '  rank 1 rejoins; the pipeline heals:'
env BADAPPLE_SHARD_DIR=$DEMO_DIR/r1 BADAPPLE_MESH_KEY=$MESH_KEY \
  DYLD_LIBRARY_PATH="$LIB_DIR" nohup "$ENGINE" >"$DEMO_DIR/r1.log" 2>&1 &
printf '  waiting for rank 1'
for i in $(seq 1 60); do
  if grep -q "serving" "$DEMO_DIR/r1.log" 2>/dev/null; then
    printf ' — serving\n\n'; break
  fi
  printf '.'; sleep 1
done

say env BADAPPLE_MESH_KEY=$MESH_KEY "$BADAPPLE" mesh-brain status --hosts 127.0.0.1:$P1,127.0.0.1:$P2
say env BADAPPLE_MESH_KEY=$MESH_KEY "$BADAPPLE" mesh-brain ask \
  --to 127.0.0.1:$P1 --prompt "The capital of France is" --max-tokens 6

# --- close --------------------------------------------------------------------
bold '  what you just saw'
dim  '  · a 7B mind that never phones home'
dim  '  · a ledger that cannot be quietly rewritten'
dim  '  · a council that votes before hands move'
dim  '  · a watchdog that can only brake'
dim  '  · one model living on two processes — encrypted, authenticated,'
dim  '    self-healing — the same path that pools everyday macs into one big mind'
echo
bold '  on your metal · with your keys · proving it the whole time'
dim  '  verify everything: badapple cert · badapple receipts · badapple --doctor'
sleep 4
