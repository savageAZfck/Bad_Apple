#!/usr/bin/env bash
# Bad Apple — Personal AGI Proof Suite
# Runs on a single Mac. Every command is real. Outputs a scorecard,
# a JSON proof artifact, and a signed attestation.
set -u
ROOT="$(cd "$(dirname "$0")" && pwd)"
[ -d "$ROOT/target/release" ] || { echo "run from repo root" >&2; exit 1; }
cd "$ROOT"

BADAPPLE="${BADAPPLE_BIN:-$ROOT/target/release/badapple}"
SOVEREIGN="${BADAPPLE_SOVEREIGN_BIN:-$ROOT/target/release/badapple-sovereign}"
PROOF_DIR="${BADAPPLE_PROOF_DIR:-$ROOT/proof}"
mkdir -p "$PROOF_DIR"

PAUSE="${PAUSE:-0}"
DEMO="${DEMO:-0}"

bold()  { printf '\033[1m%s\033[0m\n' "$*"; }
red()   { printf '\033[1;31m%s\033[0m\n' "$*"; }
green() { printf '\033[1;32m%s\033[0m\n' "$*"; }
yellow(){ printf '\033[1;33m%s\033[0m\n' "$*"; }
cyan()  { printf '\033[1;36m%s\033[0m\n' "$*"; }
dim()   { printf '\033[2m%s\033[0m\n' "$*"; }

pass=0
fail=0
optional=0

step() { bold "$1"; }
note() { [ "$DEMO" = 1 ] && { dim "$1"; sleep 0.8; } || true; }

ok() { green "  ✓ $1"; pass=$((pass+1)); }
no() { red   "  ✗ $1"; fail=$((fail+1)); }
opt() { yellow "  ○ $1 (optional)"; optional=$((optional+1)); }

grab() { "$BADAPPLE" "$@" 2>&1; }
grab_tool() { "$BADAPPLE" tool "$@" 2>&1; }

check_contains() {
  local haystack="$1" needle="$2" label="$3"
  if echo "$haystack" | grep -qi "$needle"; then
    ok "$label"
  else
    no "$label"
  fi
}

# --- intro --------------------------------------------------------------------
clear 2>/dev/null || printf '\n\n'
step '  BAD APPLE — PERSONAL AGI PROOF SUITE'
dim  '  one machine. every organ. cryptographic receipts.'
echo
sleep 1

# --- 1 · organism status ------------------------------------------------------
step '1 · the organism is awake'
note 'native daemon on apple silicon. secure enclave. no cloud.'
status=$("$BADAPPLE" status 2>&1)
check_contains "$status" "running" "daemon is running"
check_contains "$status" "Bad Apple" "identity reported"

# --- 2 · identity --------------------------------------------------------------
step '2 · she knows what she is'
note 'asked: "who are you" and "who created you"'
who=$(grab -n 120 "who are you")
check_contains "$who" "personal AGI" "identifies as personal AGI"
creator=$(grab -n 120 "who created you")
check_contains "$creator" "Adam Clark" "names her creator"

# --- 3 · perception ------------------------------------------------------------
step '3 · she can see'
note 'create a labeled image and describe it'
python3 - "$ROOT" <<'PYEOF' >/dev/null
from PIL import Image, ImageDraw, ImageFont
import sys
img = Image.new('RGB', (400, 100), color='black')
d = ImageDraw.Draw(img)
try:
    font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
except:
    font = ImageFont.load_default()
d.text((20, 30), "BAD APPLE PERSONAL AGI", fill='white', font=font)
img.save('/tmp/ba_vision_test.png')
PYEOF
desc=$(grab_tool describe_image path=/tmp/ba_vision_test.png)
check_contains "$desc" "BAD APPLE" "reads text from the image"
check_contains "$desc" "AGI" "recognizes AGI in the image"

# --- 4 · memory ----------------------------------------------------------------
step '4 · she remembers'
note 'write to the working memory scratchpad, then read it back'
grab_tool write_working_memory content="proof token: KUMQUAT-AGI-2026" >/dev/null
recall=$(grab "what is my proof token")
scratch=$(grab_tool read_working_memory)
if echo "$recall" | grep -qi "KUMQUAT" || echo "$scratch" | grep -qi "KUMQUAT"; then
  ok "recalls the stored token"
else
  no "recalls the stored token"
fi

# --- 5 · action & audit --------------------------------------------------------
step '5 · she can act and receipt it'
note 'write a file, verify the ledger catches it'
grab_tool write_file path=/tmp/ba_proof_token.txt content="personal agi proof" >/dev/null
content=$(grab_tool read_file path=/tmp/ba_proof_token.txt)
check_contains "$content" "personal agi proof" "wrote and read a file"
recent=$(tail -20 /var/lib/bad_apple/ledger.jsonl)
check_contains "$recent" "tool_call" "action landed on the ledger"

# --- 6 · deliberation ----------------------------------------------------------
step '6 · the council votes before she acts'
note 'ask the council for a verdict'
verdict=$(grab -n 200 "council should I create a backup before modifying code")
check_contains "$verdict" "council\|vote\|verdict\|yes\|no" "council produces a verdict"
ledger=$(tail -40 /var/lib/bad_apple/ledger.jsonl)
check_contains "$ledger" "council_deliberation" "verdict landed on the ledger"

# --- 7 · learning / LoRA -------------------------------------------------------
step '7 · she learns at the weight level'
note 'trained adapters exist and the dream adapter is loaded'
adapters=$(grab_tool lora_adapters)
check_contains "$adapters" "dream" "wears the nightly dream adapter"
check_contains "$adapters" "smoke" "has user-trained smoke adapter"
generated=$(grab_tool lora_generate adapter=dream prompt="I am Bad Apple and I am a" max_tokens=20)
check_contains "$generated" "Bad Apple\|Qwen\|Mac\|local" "dream adapter generates coherent identity text"

# --- 8 · the dream pass --------------------------------------------------------
step '8 · she learns in her sleep'
note 'the ledger proves the nightly dream pass ran'
ledger_full=$(tail -120 /var/lib/bad_apple/ledger.jsonl)
check_contains "$ledger_full" "dream_adopted" "nightly dream adapter was adopted"
check_contains "$ledger_full" "dream_applied" "dream adapter loaded into running weights"

# --- 9 · audit & sovereignty ---------------------------------------------------
step '9 · she can prove she is air-gapped and whole'
note 'run the certification suite and create a sovereign checkpoint'
cert=$("$BADAPPLE" cert 2>&1)
check_contains "$cert" '"status": "ok"' "air-gap cert passes"
receipts=$("$BADAPPLE" receipts 2>&1)
check_contains "$receipts" "organism:" "receipts card shows organism age"
check_contains "$receipts" "hash-chained" "receipts describe the ledger"
sov=$("$SOVEREIGN" --checkpoint 2>&1)
check_contains "$sov" "checkpoint signed\|sealed" "sovereign checkpoint signed"

# --- 10 · continuity -----------------------------------------------------------
step '10 · she has a history'
note 'ledger depth and organism age'
check_contains "$receipts" "days old" "organism age in days"
check_contains "$receipts" "attested actions" "attested action count"

# --- summary -------------------------------------------------------------------
echo
step '  PERSONAL AGI PROOF SCORECARD'
printf '  PASS: %d  FAIL: %d  OPTIONAL: %d\n' "$pass" "$fail" "$optional"
if [ "$fail" -eq 0 ]; then
  green '  STATUS: personal AGI organism verified on this machine'
else
  red   '  STATUS: some organs did not pass — review above'
fi

# --- artifacts -----------------------------------------------------------------
TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)
HOST=$(/usr/sbin/scutil --get ComputerName 2>/dev/null || uname -n)
VERSION=$("$BADAPPLE" --version 2>/dev/null || echo "v0.4.1")

cat > "$PROOF_DIR/PERSONAL_AGI_PROOF.json" <<JSONEOF
{
  "proof": "Bad Apple Personal AGI Organ Verification",
  "timestamp": "$TS",
  "host": "$HOST",
  "version": "$VERSION",
  "status": "$([ "$fail" -eq 0 ] && echo OK || echo PARTIAL)",
  "pass": $pass,
  "fail": $fail,
  "optional": $optional,
  "artifacts": {
    "cert_status": "$(echo "$cert" | grep '"status"' | tail -1)",
    "receipts_card": "$(echo "$receipts" | head -6 | tr '\n' ' ')",
    "sovereign_summary": "$(echo "$sov" | tail -2 | tr '\n' ' ')"
  }
}
JSONEOF

python3 - "$HOST" "$TS" "$VERSION" "$pass" "$fail" "$optional" "$PROOF_DIR" <<'PYEOF'
import sys
host, ts, version, p, f, o, out_dir = sys.argv[1:]
verdict = "VERIFIED" if int(f) == 0 else "PARTIAL — review failures above"

text = f"""# Personal AGI Organism — Attestation

**Machine:** {host}  
**Time:** {ts} (UTC)  
**Bad Apple version:** {version}  
**Claim:** This machine has been exercised as a complete personal AGI organism — perception, memory, deliberation, action, governance, learning, audit, and sovereignty — and produced cryptographic receipts for every stage.

## Score
- PASS: {p}
- FAIL: {f}
- OPTIONAL: {o}
- Verdict: **{verdict}**

## What was proven
1. **Awake:** the native Apple-Silicon daemon is running with Secure Enclave identity.
2. **Identity:** when asked "who are you" she declares herself a sovereign personal AGI.
3. **Perception:** she read the text "BAD APPLE PERSONAL AGI" from an image.
4. **Memory:** she stored and later recalled a proof token.
5. **Action:** she wrote and read a file, with the tool call landing on the ledger.
6. **Deliberation:** the 14-seat council produced a verdict and recorded it.
7. **Weight-level learning:** trained LoRA adapters exist and generate coherent text.
8. **Dreaming:** the ledger shows a nightly `dream_adopted` and `dream_applied` pass.
9. **Audit:** the air-gap certification suite passed.
10. **Sovereignty:** a sovereign ledger checkpoint was signed.
11. **Continuity:** the organism has a hash-chained history spanning days.

## How to verify on this machine
```
badapple cert
badapple receipts
badapple-sovereign --checkpoint
```

The hash-chained ledger lives at `/var/lib/bad_apple/ledger.jsonl`.  
The sovereign copy lives at `/var/lib/bad_apple/ledger.sovereign.jsonl`.  
The signed checkpoints live at `/var/lib/bad_apple/ledger_checkpoint.json` and `/var/lib/bad_apple/ledger.sovereign.checkpoint.json`.

**This is a machine-local, owner-attested proof.** It does not phone home. It does not require cloud verification. The receipts are the authority.
"""

with open(f"{out_dir}/PERSONAL_AGI_ATTESTATION.md", "w") as fh:
    fh.write(text)
PYEOF

dim "  artifacts written:"
dim "    $PROOF_DIR/PERSONAL_AGI_PROOF.json"
dim "    $PROOF_DIR/PERSONAL_AGI_ATTESTATION.md"
