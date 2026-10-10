#!/bin/bash
# Bring up the 14B GPU+ANE split-brain mesh on loopback.
#
#   GPU rank 0  :8741  layers 0-20  (MLX, embed, drives the mesh)
#   ANE rank 1  :8744  layers 20-40 (CoreML/Neural Engine, head)
#
# Ask it:  badapple mesh-brain ask --to 127.0.0.1:8741 --prompt "..." --max-tokens N
# Kill it: tools/mesh14_up.sh --down
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/target/release"
ART="$ROOT/ane_artifacts"
# Rank-0 prefix lives under the HF cache so `badapple model add` accepts it
# (the daemon jails custom model paths to the cache root).
PREFIX_DIR="$HOME/.cache/huggingface/hub/q14-mesh/model"  # mesh_brain::build_shard output, layers 0-20
TAIL_MANIFEST="$ART/qwen14b_tail/tail20_manifest.json"  # layers 20-40 relabeled 0-20
TOKENIZER="$ART/qwen14b_tail/tokenizer.json"
LOG0=/tmp/mesh14_gpu.log
LOG1=/tmp/mesh14_ane.log

if [ "${1:-}" = "--down" ]; then
    pkill -f 'badapple-ane-rank --listen 127.0.0.1:8744' || true
    pkill -f 'BADAPPLE_SHARD_DIR=.*q14-mesh' || true
    echo "mesh14 ranks stopped"
    exit 0
fi

for f in "$PREFIX_DIR/shard.json" "$TAIL_MANIFEST" "$TOKENIZER" "$BIN/badapple-ane-rank" "$BIN/badapple-engine"; do
    [ -e "$f" ] || { echo "missing: $f" >&2; exit 1; }
done

# ANE tail first — it is the slow boot (plan-builds ~100-140s for 14 shards).
# fp16 shards trip a cumulative plan-build -14 wall past ~17; 8-bit shards (330MB) plan 20+ clean.
if ! pgrep -f 'badapple-ane-rank --listen 127.0.0.1:8744' >/dev/null; then
    echo "booting ANE tail :8744 (20 q8 shards, ~30s)..."
    nohup env BADAPPLE_ANE_MAX_PREWARM_MS=600000 \
        "$BIN/badapple-ane-rank" --listen 127.0.0.1:8744 \
        --manifest "$TAIL_MANIFEST" --tokenizer "$TOKENIZER" \
        --rank 1 --world 2 --layers 20-40 >"$LOG1" 2>&1 &
fi

# GPU prefix rank — fast boot (~10s for 20 layers).
if ! pgrep -f 'BADAPPLE_SHARD_DIR=.*q14-mesh' >/dev/null && \
   ! lsof -nP -iTCP:8741 -sTCP:LISTEN >/dev/null 2>&1; then
    echo "booting GPU prefix :8741 (layers 0-20)..."
    nohup env BADAPPLE_SHARD_DIR="$PREFIX_DIR" \
        "$BIN/badapple-engine" >"$LOG0" 2>&1 &
fi

echo "waiting for ranks..."
for i in $(seq 1 40); do
    a=$(lsof -nP -iTCP:8741 -sTCP:LISTEN >/dev/null 2>&1 && echo up || echo -)
    b=$(lsof -nP -iTCP:8744 -sTCP:LISTEN >/dev/null 2>&1 && echo up || echo -)
    [ "$a" = up ] && [ "$b" = up ] && { echo "mesh14 up: :8741 (GPU 0-20) -> :8744 (ANE 20-40)"; exit 0; }
    sleep 5
done
echo "timed out — check $LOG0 and $LOG1" >&2
exit 1
