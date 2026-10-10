#!/usr/bin/env bash
# Live-daemon decode benchmark: runs N novel prose prompts through the
# installed badapple daemon and prints per-turn decode tok/s, draft
# acceptance, prompt size and prefix-cache state from the ledger, plus a
# memory snapshot. Run this with the IDE / browsers closed to see what the
# organism does when it has the machine to itself.
set -euo pipefail
cd "$(dirname "$0")/.."
BIN=target/release/badapple
LEDGER=/var/lib/bad_apple/ledger.jsonl
N=${1:-4}
TOK=${2:-100}

PROMPTS=(
  "Write three sentences about the creak of a wooden staircase."
  "Write three sentences about frost patterns on a car windshield."
  "Write three sentences about the smell of a library in summer."
  "Write three sentences about the glow of a streetlight in fog."
  "Write three sentences about the sound of gravel under bicycle tires."
  "Write three sentences about the color of tea left too long."
)

echo "== memory before =="
vm_stat | awk '/Pages (free|inactive|occupied by compressor)/ {gsub(/\./,"",$NF); printf "  %-12s %6.2f GB\n", $0 ~ /compressor/ ? "compressor" : $2, $NF*16384/1073741824}'
sysctl -n vm.swapusage | sed 's/^/  swap: /'

$BIN -n 10 "Say ready." >/dev/null 2>&1 || true

# --benchmark bypasses the semantic cache and tool loops and prints its own
# table (tok, ttft, decode tok/s, total, mem); it does not write a ledger
# row, so read the table directly. Errors are shown, not swallowed.
echo "== turns (ttft = prefill wait, decode = tok/s after first token) =="
for ((i=0; i<N; i++)); do
  p=${PROMPTS[$((i % ${#PROMPTS[@]}))]}
  start=$(date +%s)
  out=$($BIN --benchmark -n "$TOK" "$p" 2>&1) || true
  wall=$(( $(date +%s) - start ))
  row=$(printf '%s\n' "$out" | grep -E "^Write" | head -1)
  if [[ -n "$row" ]]; then
    printf '  wall=%3ss  %s\n' "$wall" "$(printf '%s' "$row" | sed -E 's/^Write[^…]*…[[:space:]]+//')"
  else
    printf '  wall=%3ss  FAILED: %s\n' "$wall" "$(printf '%s\n' "$out" | tail -2 | tr '\n' ' ')"
  fi
done
echo "  (columns: tok  ttft(s)  decode(tok/s)  total(s)  engine mem(GB))"
echo "== last spec log lines (draft acceptance) =="
grep "BadAppleSpec\]" /var/log/bad_apple_mlx_server.log | tail -"$N" | sed -E 's/^.*\[BadAppleSpec\]/  /'

echo "== memory after =="
vm_stat | awk '/Pages (free|inactive|occupied by compressor)/ {gsub(/\./,"",$NF); printf "  %-12s %6.2f GB\n", $0 ~ /compressor/ ? "compressor" : $2, $NF*16384/1073741824}'
sysctl -n vm.swapusage | sed 's/^/  swap: /'
