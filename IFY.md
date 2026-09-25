# IFY — the immune system for Bad Apple

IFY ("I'll Find You") is Bad Apple's behavioral watchdog: a small
deterministic daemon that learns what normal looks like on this machine,
notices when reality drifts, and — at most — pulls the emergency brake.
It never steers. It proposes, it narrates, it brakes. Everything it does
lands in the same tamper-evident record it watches.

The design rule: **the detector is math, the narrator is a model, the
record is cryptography.** Detection cannot hallucinate because it only
reports what the numbers show; narration can be wrong in prose but never
invent a finding, because it only renders findings the math already
produced.

## Where it sits

```
ledger.jsonl ──tail──▶ badapple-ify ──▶ ~/.bad_apple/ify/   (state, baselines)
      (verified        │              ~/.bad_apple/ify/findings.jsonl
       incrementally)  ├─ curiosity ─▶ findings log only        [gestation]
                       ├─ notable   ─▶ proposal .md + notify    [secondary+]
                       └─ critical  ─▶ kill_switch via SLICKS   [autopilot]
                                        + runtime_state SAFE_MODE
                                        + proposal .md explaining why
```

Inputs:

- `/var/lib/bad_apple/ledger.jsonl` — the primary signal: every query,
  tool call, approval, firewall hit, persona switch. Tailed incrementally
  by byte offset; each new line is hash-chain-verified as it arrives
  (a chain break is itself a critical anomaly — the watched cannot forge
  the watcher's evidence).
- `/var/lib/bad_apple/runtime_state.json` — supervisor health signals
  (safe mode, kill state) folded into the baseline as system events.

Outputs:

- `~/.bad_apple/ify/state.json` — phase, baselines, tail offset.
- `~/.bad_apple/ify/findings.jsonl` — every observation, silent or not.
- `~/.bad_apple/ify/proposals/*.md` — proposals in the
  `proposed_patches` markdown format (`**When:**`, `**Workspace:**`,
  `## Proposal` + ```json block with `no_patch` or a `patch`), so they
  surface in the existing dashboard approve/reject pipeline.
- Desktop notification via `osascript display notification` for
  notable+ findings.
- Optional narration through the SLICKS socket (`stream_query`), asking
  the model to render the finding in plain English — the fast tier when
  armed (`BADAPPLE_FAST_TIER=1` + `BADAPPLE_FAST_MODEL`), otherwise the
  main brain. Skipped silently when the engine is down.

## Phases

| Phase | Duration | IFY can |
|---|---|---|
| `gestation` | `BADAPPLE_IFY_GESTATION_DAYS` (default 14) | observe, baseline, log findings internally — no user-facing output |
| `secondary` | `BADAPPLE_IFY_SECONDARY_DAYS` (default 14) | surface curiosities as notifications, write proposals — all approval-gated |
| `autopilot` | after | same as secondary **plus** the brake: critical findings may engage kill switch |

Phase transitions are automatic on elapsed time and recorded as
findings. `BADAPPLE_IFY_PHASE` forces a phase (testing, demos).
`BADAPPLE_IFY=0` disables the daemon entirely.

Autopilot for IFY never means "act on the system." It means "may pull
the brake." Proposals are always human-approved regardless of phase and
regardless of `BADAPPLE_AUTOPILOT` — the immune system does not inherit
the patient's privileges.

## Baselines

`~/.bad_apple/ify/state.json` holds deterministic statistics — no model,
no embeddings:

```json
{
  "version": 1,
  "installed_at": 1760000000,
  "phase": "secondary",
  "events_seen": 8123,
  "ledger_offset": 1048576,
  "ledger_tip": "ab12…",
  "event_types": {
    "query":     {"count": 5000, "hourly": [0,0,…]},
    "tool_call": {"count":  800, "hourly": [0,0,…]}
  },
  "tools":      {"run_shell": 120, "read_file": 400},
  "personas":   {"default": 4000, "wicket": 300},
  "approvals":  {"granted": 55, "denied": 3},
  "firewall_hits": 12,
  "kill_switch_events": 1
}
```

Baselines keep learning in every phase; the numbers just gain a voice
after gestation.

## Detection rules (v1)

All deterministic. Every finding records the observed number, the
baseline number, and the rule that fired — a finding you cannot explain
is a finding you cannot trust.

| Rule | Fires when | Severity |
|---|---|---|
| `chain_break` | a new ledger line fails incremental hash verification | critical |
| `novel_event_type` | event type never seen in baseline | elevated (info during gestation) |
| `rate_spike` | events of a type in the last hour > max(4× baseline hourly avg, 10) | elevated |
| `off_hours` | activity in an hour-of-day bucket with zero history | info |
| `denial_spike` | ≥3 denied tool approvals in an hour | elevated |
| `kill_switch` | a kill_switch/safe_mode ledger event appears | elevated (you should know if it wasn't you) |
| `firewall_spike` | output-firewall events > max(4× baseline, 3) in an hour | elevated — possible injection probing |
| `ledger_gap` | ledger mtime advances but byte offset shrinks (truncation) | critical |

Severity ladder: `info` → findings log only · `elevated` → proposal +
notification (secondary and up) · `critical` → proposal + notification +
brake (autopilot only; earlier phases still propose the brake for manual
approval — gestation IFY can suggest "you may want to hit the kill
switch," it just can't pull it).

## The brake

A critical finding in autopilot phase does exactly three things:

1. `call_agent("invoke_tool", {"name": "kill_switch"})` over the SLICKS
   socket — the same brake the user's "kill switch" command pulls.
2. Writes `runtime_state.json` `mode: SAFE_MODE` with
   `safe_mode_reason: "ify:<rule>"` so the menu bar and dashboard show
   who stopped the show and why.
3. Writes a proposal recording the finding, the numbers, and the brake
   action — the audit trail explains itself.

A false positive costs a paused daemon and a proposal to dismiss. That
asymmetry is intentional: the watchdog can only ever stop the machine,
never move it.

## Narrator

When `BADAPPLE_IFY_NARRATE != 0` and the engine is up, notable findings
are rendered by the model through the normal query path (the fast tier
handles them when armed):

> "IFY noticed: `run_shell` fired 47 times between 3–4am — your baseline
> for that hour is zero. Finding `ify-20261009-0312-a4f2`. Proposal
> filed, nothing was done about it."

The narrator only ever sees the finding's numbers — never ledger bodies,
never prompts — so it cannot leak content and cannot invent causes.
If the engine is down or the prompt times out, the raw finding still
logs and notifies; narration is garnish, not substance.

## What IFY is not

- Not a remediation engine. It proposes; humans apply.
- Not a second model watching the first. The detector is statistics over
  a verified stream — the recursion problem ("who watches the watcher")
  is answered by determinism, and its own actions land in the record.
- Not a claim that baselines are truth. A slowly-drifting baseline can
  normalize an attack; `chain_break` and `ledger_gap` are the two rules
  that don't depend on learned normality at all — they check the
  evidence itself.

## Operations

```bash
badapple-ify --once          # single tail+detect pass, print report
badapple-ify --status        # phase, baselines summary, recent findings
launchctl load -w ~/Library/LaunchAgents/com.badapple.ify.plist
```

Files: `~/.bad_apple/ify/{state.json,findings.jsonl,proposals/}`.
All state is user-owned plaintext JSON — inspectable, deletable,
degradable. Deleting `state.json` restarts gestation; that is the
correct behavior, not a bug.
