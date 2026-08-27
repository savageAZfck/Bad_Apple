# Bad Apple Shortcuts & Siri Samples

The `BadAppleIntent` module in `src/platform/apple_bridge/BadAppleIntent.swift` registers AppIntents for Shortcuts and Siri on macOS. These are the default, air-gapped samples that ship with the app.

## Default App Shortcuts

| Shortcut | Siri phrase | What it does |
|---|---|---|
| **Execute Bad Apple** | "Ask Bad Apple" or "Execute Bad Apple" | Sends spoken text to the local daemon and speaks the reply. |
| **Bad Apple Status** | "What's my Bad Apple status" | Returns runtime, active model, and memory summary. |
| **Bad Apple Kill Switch** | "Stop Bad Apple" | Engages the kill switch to halt generation and tool use. |
| **Bad Apple Kill Switch** | "Resume Bad Apple" | Resumes from kill switch / safe mode when supported by the intent. |

## Sample Shortcuts you can build in the Shortcuts app

1. **Good Morning** — run at 7 AM
   - Action: *Execute Bad Apple*
   - Spoken text: `summarize my working memory and ambient context`
   - Output: spoken summary while you are getting ready.

2. **Focus Kill Switch** — a button or Siri
   - Action: *Bad Apple Kill Switch*
   - Set *Engage* to `On`
   - Use phrase: "Hey Siri, stop Bad Apple"

3. **Read My Day** — voice or widget
   - Action: *Execute Bad Apple*
   - Spoken text: `read my working memory`
   - Use phrase: "Hey Siri, read my Bad Apple notes"

## Siri Phrase Reference

- "Ask Bad Apple what is the capital of France"
- "Ask Bad Apple what do you think of Siri"
- "Show Bad Apple status"
- "Stop Bad Apple" (engages kill switch)
- "Resume Bad Apple" (resumes generation)

All commands run through the local SLICKS socket (`/var/run/badapple/substrate.sock`) and never leave the Mac.
