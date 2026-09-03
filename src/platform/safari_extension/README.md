# Bad Apple Safari Companion

A Safari web extension scaffold for Bad Apple. It lets the assistant see the
current web page and answer questions about it, all routed through the local
SLICKS Unix socket.

## Layout

- `BadAppleCompanion/Resources/` — Safari web extension bundle (manifest, content
  script, popup, background worker).
- `BadAppleCompanion/Resources/` — Safari web extension bundle.

## Status

The Safari companion native messaging host has been removed as part of the
Python-free migration. A native `badapple-companion-host` binary may be added
later. The web extension bundle remains for a future port.
