# Bad Apple — User Guide

Plain-English handbook for installing and using Bad Apple on a Mac.

## What Bad Apple is

Bad Apple is a personal AI that lives entirely on your Mac. She is not a chatbot
window or a cloud app — she is a small set of background programs that give her
a brain, senses, memory, and rules she cannot break.

A few things worth knowing up front:

- **Everything is local.** Your prompts, her memory, and her activity log stay
  on your machine. Nothing leaves your Mac unless you ask it to.
- **She has rules, not just settings.** Some actions require your consent
  through her built-in approval system — that is on purpose.
- **She is one of a kind per machine.** Each install is its own creature with
  its own keys, memory, and history. Two Macs, two Bad Apples.

## Installing

Requirements: Apple Silicon Mac (M1 or newer), macOS 14+, ~10 GB free disk, and
[Homebrew](https://brew.sh) installed. You will be asked for your Mac's admin
password during install — that is the daemons being registered with the system.

Open the **Terminal** app (Applications → Utilities), then run:

```bash
brew tap savageAZfck/bad-apple https://github.com/savageAZfck/homebrew-bad-apple
brew install --cask bad-apple
```

What happens next:

1. Homebrew downloads the package (~200 MB) and installs the app + daemons.
2. The `brew install` command itself then downloads her main brain — about
   **4–6 GB**. This is the longest part; the installer prints
   `still installing` progress lines while it works. On a slow connection this
   can take 20+ minutes. That is normal — do not kill it.
3. The first time you open the menu bar app, macOS may warn that it is unsigned.
   That is expected — click **Open** at the prompt (or System Settings →
   Privacy & Security → **Open Anyway**).

When the installer finishes, you get:

- `/Applications/Bad Apple.app` — the menu bar app
- `badapple` — the command-line tool
- Background programs that keep her alive and restart themselves if they crash

**How to tell it worked:** run `badapple "hello"` — she should answer. The very
first reply can take ~45 seconds while her brain warms up; after that it is
fast.

## Talking to her

### From the terminal

```bash
badapple "What can you do?"
badapple -n 200 "Write me a haiku about bare metal"
```

`-n` sets the maximum length of her answer.

### From the menu bar

Click the **apple icon** in your menu bar. You get a prompt field, conversation
history, persona switching, model switching, and status at a glance.

### Voice

```bash
badapple --speak "Tell me a story"   # she answers out loud
```

The first spoken reply may take a moment while her voice wakes up — make sure
your volume is on.

## The dashboard (Control Center)

Open it from the menu bar — **Bad Apple → Web Dashboard**. (Opening it this way
signs you in automatically. If you ever open `http://127.0.0.1:8787` directly in
a browser and it asks for a token, it is at `/var/lib/bad_apple/dashboard.token`.)

The dashboard is her control surface: status, models, agents, consent
approvals, logs. The **Models** page shows which models fit your machine's
memory — anything too big is greyed out so you cannot pick it by accident.

## Models

She ships with a main brain (a ~5 GB model file) that fits any modern Apple
Silicon Mac. Bigger brains exist in the catalog — the Models page marks which
ones your RAM can hold. More RAM = bigger brains.

To switch brains:

- Menu bar → **GPU Brain Model** → pick one, or
- Dashboard → Models → Switch, or
- `badapple model list` then `badapple model use <model-id>` from the terminal.

## Personas

She has switchable personalities: default, cali, curious, wicket, genz, drill,
midwest.

```bash
badapple "switch to cali"
badapple --persona drill "What do you think of Siri?"
```

## What she can do

- Answer questions, write code, summarize, reason — normal assistant work.
- Use **tools** with your permission: files, shell commands, AppleScript,
  mail/calendar/reminders — gated by her consent system, not just your word.
- **Watch and listen** (opt-in senses): screen narration and ambient speech
  recognition, off unless you grant the permissions — she will link you to the
  right System Settings panes (Accessibility, Speech Recognition, Microphone).
- **Remember**: facts you teach her persist. Teach her one:

  ```bash
  badapple "teach The deploy server is on the office mini"
  ```

- **Prove herself**: every action lands in a tamper-evident log
  (`/var/lib/bad_apple/ledger.jsonl`) you can audit:

  ```bash
  badapple --doctor        # health check + log verification
  badapple ify status      # her self-watchdog's status
  badapple cert            # certification checks
  ```

## Governance in 60 seconds

She is built to be sovereign — some actions need your approval rather than just
being allowed. If she asks for consent on something (destructive actions,
reading outside her data vault, network calls), the request shows up in the
dashboard's approvals section and waits for your yes or no. That is the system
working as designed, not a bug.

## Turning downloads off

Downloads are **on by default** — a fresh machine cannot work without her brain
file. To run sealed/air-gapped, you need to opt out at install time:

```bash
BADAPPLE_AIRGAP=1 brew install --cask bad-apple
```

On an installed system, the "Allow Bad Apple to download models" checkbox on
the dashboard Models page controls whether she may fetch new models. (Her first
brain is always fetched during install regardless.)

## Troubleshooting

- **Install failed?** The installer asks if you want to file a bug report — it
  writes a diagnostic bundle to `/tmp` and opens an issue page for you.
  Nothing sends without your yes.
- **`badapple` hangs or errors?** Check the logs —
  `tail -20 /var/lib/bad_apple/mlx_server.log` (engine) and
  `tail -15 /var/log/bad_apple_gatekeeper.log` (the front proxy). Paste them
  into an issue if you file one.
- **Nothing responds?** `badapple --doctor` checks daemon health, the log
  chain, and signing.
- **Everything else?** File at
  `github.com/savageAZfck/bad-apple-releases/issues`.

## Updating and removing

- **Update**: menu bar → **Check for Updates…** — or
  `brew upgrade --cask bad-apple`.
- **Remove**: quit the menu bar app first, then
  `brew uninstall --cask bad-apple`.
- **Remove everything, including her memory and state**:
  `brew uninstall --cask --zap bad-apple`, then delete the model cache at
  `~/.cache/huggingface` (the ~4–6 GB brain files live there, not in the app).

## One-line mental model

She is a creature, not a service: a brain on your GPU, organs with rules,
memory, and a log she cannot forge — and she only lives on your Mac.
