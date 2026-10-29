# Bad Apple — User Guide

Plain-English handbook for installing and using Bad Apple on a Mac. If you only
read three sections, read **Trust and sudo**, **Installing**, and **Talking to
her**.

## What Bad Apple is

Bad Apple is a personal AI that lives entirely on your Mac. She is not a
chatbot window or a cloud app — she is a small set of background programs
(daemons) that give her a brain, senses, memory, and rules she cannot break.

A few things worth knowing up front:

- **Everything is local.** Your prompts, her memory, and her activity log stay
  on your machine. Nothing leaves your Mac unless you ask it to.
- **She has rules, not just settings.** Some actions require your consent
  through her built-in approval system — that is on purpose.
- **She is one of a kind per machine.** Each install is its own creature with
  its own keys, memory, and history. Two Macs, two Bad Apples.

## Trust and sudo — what you are actually installing

Installing her requires your admin password, and that deserves an explanation.
Here is exactly what happens and why:

**The app is unsigned by choice.** Apple notarization would require uploading
her binaries to Apple's servers. As a sovereignty stance, we do not do that.
The cost: macOS Gatekeeper warns on first launch (expected — click **Open**, or
System Settings → Privacy & Security → **Open Anyway**).

**Why your password:** the installer registers three system background services
(the request-classifier proxy, the engine, and a health supervisor), creates
her state directory `/var/lib/bad_apple`, and links the `badapple` command into
`/usr/local/bin`. Those paths need admin rights; nothing else does. The
installer prints this note when it runs.

**What each piece does:**

| Program | Runs as | Job |
|---|---|---|
| `gatekeeper` | root (system daemon) | Front proxy — runs each request through her organ checks before it reaches the brain (no relation to Apple's Gatekeeper) |
| `badapple-engine` | you | The brain itself — loads the model and serves inference |
| `badapple-supervisor` | root (system daemon) | Health watchdog — restarts daemons that die |
| identity agent | you (per-user agent) | Secure Enclave signing for her log checkpoints |
| menu bar app + dashboard | you | The UI you actually touch |

("organs" = her rule-checking programs that sit between a request and the
brain — classify, policy, consent. You never have to touch them.)

**Network access:** the only outbound traffic is model downloads from
HuggingFace (on by default, turnable-off — see **Turning downloads off**) plus
the release check when you ask for an update. There is no telemetry, no
analytics, no phone-home.

**Verify it yourself** (for technical users): the release zip is cosign-signed
(key in `cosign.pub`), ships an SBOM on the release page, and the package
includes a source subset — the installer script and daemon configs it runs are
in `bad_apple/src/platform/` inside the zip.

**If it fails:** the installer asks before filing anything — nothing is sent
without your yes.

## Installing

Requirements: Apple Silicon Mac (M1 or newer), macOS 26+, **8 GB RAM minimum**
(16 GB more comfortable), ~40 GB free disk, and [Homebrew](https://brew.sh)
installed. On macOS 27 her organ checks run on the Apple Neural Engine; on
26.x they automatically run on CPU instead — everything else works the same.

Open the **Terminal** app (Applications → Utilities), then run:

```bash
brew tap savageAZfck/bad-apple https://github.com/savageAZfck/homebrew-bad-apple
brew install --cask bad-apple
```

What happens next:

1. Homebrew downloads the package (~200 MB), then the installer it runs
   downloads her main brain — about **4–6 GB**. It prints `still installing`
   progress lines while it works. On a slow connection this can take 20+
   minutes — do not kill it.
2. The first time you open the menu bar app you may see the unsigned-app
   warning covered above.

When it finishes you have: `/Applications/Bad Apple.app`, the `badapple`
command, and her daemons running.

**How to tell it worked:** run `badapple "hello"` — she should answer. The very
first reply can take ~45 seconds while her brain warms up; after that it is
fast. (Same after a reboot.)

## Talking to her

### From the terminal

```bash
badapple "What can you do?"
badapple -n 200 "Write me a haiku about bare metal"
```

`-n` sets the maximum answer length. `badapple --help` lists everything.

### From the menu bar

Click the **apple icon** in your menu bar. You get a prompt field, conversation
history, persona switching, model switching, tasks, and status at a glance.

### Voice

```bash
badapple --speak "Tell me a story"   # she answers out loud
```

The first spoken reply may lag while her voice wakes — check your volume.

## The dashboard (Control Center)

Open it from the menu bar — **Organs → Web Dashboard** (it signs you in
automatically; the plain "Dashboard" item opens the native window instead).
If you open `http://127.0.0.1:8787` directly and it asks for a token, it is at
`/var/lib/bad_apple/dashboard.token`.

The dashboard is her control surface: status, models, agents, logs.

## Approvals — when she asks permission

Some actions need your yes, not just your request. Approvals arrive two ways:

- **macOS notification** with Approve/Deny buttons, or
- **in chat** — reply `approve <id>` or `deny <id>`

What triggers them: shell commands, AppleScript, Shortcuts, writing files,
memory writes, document indexing, screen capture, and memory consolidation.
Reads outside her file jail are simply denied (no approval possible — hard
boundary). That is the system working, not a bug.

## Models

She ships with a main brain (a ~5 GB model file) that fits any modern Apple
Silicon Mac. Bigger brains exist in the catalog — the Models page greys out
anything your RAM cannot hold, so you cannot pick a model that will not fit.

To switch: menu bar → **GPU Brain Model**, or Dashboard → Models → Switch, or
`badapple model list` then `badapple model use <model-id>`.

Other model commands: `model scan` (find downloaded weights), `model verify`
(check integrity), `model recommend` (ask her which fits).

## Personas

Switchable personalities: default, cali, curious, wicket, genz, drill, midwest.

```bash
badapple "switch to cali"
badapple --persona drill "What do you think of Siri?"
```

## What she can do

- **Assistant work**: answer, write code, summarize, reason, plan multi-step
  tasks (her Tasks view in the menu bar shows active runs).
- **Use tools with permission**: files, shell commands, AppleScript, mail,
  calendar, reminders — all gated by the approval rules above.
- **Senses (opt-in)**: screen narration (eyes) and ambient speech recognition
  (ears). Off unless you enable them — she links you to the System Settings
  panes for Accessibility, Microphone, and Speech Recognition.
- **Remember**: teach her facts that persist across sessions:

  ```bash
  badapple "teach The deploy server is on the office mini"
  ```

- **Watch herself**: her IFY watchdog tails her own activity log, learns what
  normal looks like, and flags anomalies — she literally has a self-monitoring
  reflex: `badapple ify status`.
- **Prove herself**: every action lands in a tamper-evident log
  (`/var/lib/bad_apple/ledger.jsonl`). `badapple cert` verifies its chain and
  checkpoints; `badapple --doctor` checks daemon health and prints a support
  report.

Under the hood she has more safety machinery than the average install needs to
think about — an output firewall (blocklist patterns she cannot emit), a WASM
cage for untrusted code, daily state snapshots she can revert to, a dead-man
switch, and a council that votes on gated actions. You never have to touch any
of it; it is just how she is built.

## Turning downloads off

Downloads are **on by default** — a fresh machine cannot work without her brain
file. To run sealed/air-gapped, opt out at install time:

```bash
BADAPPLE_AIRGAP=1 brew install --cask bad-apple
```

(you would then need to seed her model cache yourself at `~/.cache/huggingface`).

On an installed system, the "Allow Bad Apple to download models" checkbox on
the dashboard Models page controls whether she may fetch new models.

## Troubleshooting

- **Install failed?** The installer asks if you want to file a bug report — it
  writes a diagnostic bundle to `/tmp` and opens an issue page for you.
  Nothing sends without your yes.
- **`badapple` hangs or errors?** Check the logs —
  `tail -20 /var/lib/bad_apple/mlx_server.log` (engine) and
  `tail -15 /var/log/bad_apple_gatekeeper.log` (front proxy).
- **Nothing responds?** `badapple --doctor` prints a support report — daemon
  health, sockets, launchd states, cache.
- **After a reboot she is slow on the first reply** — her brain reloads on
  first use (~45 s). Normal.
- **Bugs and questions**: `github.com/savageAZfck/bad-apple-releases/issues`.

## Updating and removing

- **Update**: `brew upgrade --cask bad-apple` — no extra tooling needed.
  (The menu bar's **Check for Updates…** also works but requires `cosign`
  from brew to verify the release signature.)
- **Remove**: quit the menu bar app first, then
  `brew uninstall --cask bad-apple`.
- **Remove her memory and state too**:
  `brew uninstall --cask --zap bad-apple` clears `/var/lib/bad_apple`,
  `~/.bad_apple`, and her logs — then delete the model cache at
  `~/.cache/huggingface` (the ~4–6 GB brain files live there, not in the app).
  For a truly clean slate also delete `/Library/Application Support/Bad Apple`
  (needs sudo), `~/Library/Application Support/BadApple`, and
  `~/.cache/badapple`.

## One-line mental model

She is a creature, not a service: a brain on your GPU, organs with rules,
memory, and a log she cannot forge — and she only lives on your Mac.
