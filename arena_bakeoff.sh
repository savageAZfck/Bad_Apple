#!/usr/bin/env bash
# arena_bakeoff.sh — blind bake-off: Bad Apple vs arena frontier models.
#
#   ./arena_bakeoff.sh prepare "your prompt"
#       → ~/.bad_apple/arena/roundNN/{prompt.md,badapple.md}
#       → paste prompt.md into arena battle mode, save the two replies as
#         arena_a.md and arena_b.md in the same round dir
#
#   ./arena_bakeoff.sh judge
#       → badapple judges her answer vs each arena reply (order shuffled),
#         appends ~/.bad_apple/arena/verdicts.jsonl, prints the scoreboard
#
# BAKEOFF_JUDGE overrides the judge binary (default target/release/badapple).

set -euo pipefail
ROOT="${BAKEOFF_DIR:-$HOME/.bad_apple/arena}"
JUDGE="${BAKEOFF_JUDGE:-$(dirname "$0")/target/release/badapple}"
mkdir -p "$ROOT"

cmd="${1:-}"; shift || true

next_round() {
    local n=1
    while [[ -d "$ROOT/round$(printf '%02d' "$n")" ]]; do n=$((n+1)); done
    printf '%02d' "$n"
}

case "$cmd" in
prepare)
    prompt="${*:?usage: $0 prepare \"<prompt>\"}"
    dir="$ROOT/round$(next_round)"
    mkdir -p "$dir"
    printf '%s\n' "$prompt" > "$dir/prompt.md"
    "$JUDGE" "$prompt" > "$dir/badapple.md"
    cat <<EOF
round: $dir
  prompt.md    → paste into arena battle mode
  badapple.md  → her answer (already generated)
save the two arena replies as:
  $dir/arena_a.md
  $dir/arena_b.md
then run: $0 judge
EOF
    ;;
judge)
    scoreboard="$ROOT/verdicts.jsonl"
    touch "$scoreboard"
    for dir in "$ROOT"/round*/; do
        [[ -f "$dir/arena_a.md" && -f "$dir/arena_b.md" ]] || continue
        name=$(basename "$dir")
        grep -q "\"round\": \"$name\"" "$scoreboard" && continue
        prompt=$(cat "$dir/prompt.md")
        for side in a b; do
            if (( RANDOM % 2 )); then
                first="$dir/badapple.md"; first_is="her"
                second="$dir/arena_$side.md"
            else
                first="$dir/arena_$side.md"; first_is="arena"
                second="$dir/badapple.md"
            fi
            verdict=$("$JUDGE" "Judge two responses to the same prompt. Reply with exactly 'A' or 'B' then one sentence why.

PROMPT: $prompt

RESPONSE A:
$(cat "$first")

RESPONSE B:
$(cat "$second")" | head -1)
            letter=${verdict:0:1}
            if [[ "$letter" == "A" ]]; then winner=$first_is; elif [[ "$letter" == "B" ]]; then [[ "$first_is" == her ]] && winner=arena || winner=her; else winner=unclear; fi
            printf '{"round": "%s", "vs": "arena_%s", "winner": "%s", "verdict": %s}\n' \
                "$name" "$side" "$winner" "$(printf '%s' "$verdict" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read().strip()))')" \
                >> "$scoreboard"
        done
    done
    her=$(grep -c '"winner": "her"' "$scoreboard" || true)
    arena=$(grep -c '"winner": "arena"' "$scoreboard" || true)
    unclear=$(grep -c '"winner": "unclear"' "$scoreboard" || true)
    echo "scoreboard — her: $her  arena: $arena  unclear: $unclear   ($scoreboard)"
    ;;
*)
    sed -n '2,14p' "$0"
    ;;
esac
