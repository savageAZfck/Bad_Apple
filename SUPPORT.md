# Support

Bad Apple is a sovereign, local AI OS. Because it runs entirely on your machine, most issues can be diagnosed from the logs and a few built-in commands.

## Quick health check

```bash
.venv/bin/python agent_client.py status
```

This returns runtime mode, health checks, active models, memory usage, and P2P state.

## Check the daemon log

```bash
tail -n 50 /var/log/bad_apple_mlx_server.log
```

Look for:

- `[daemon] model loaded` — the 9B brain is ready.
- `safe_mode` — the supervisor shut something down due to repeated failures.
- `Socket path ... does not exist` — the gatekeeper or MLX daemon is not running.

## Restart the daemons

If the status is not `READY` or a socket is missing, restart the platform:

```bash
osascript -e 'do shell script "launchctl unload /Library/LaunchDaemons/com.badapple.mlx.plist 2>/dev/null; launchctl unload /Library/LaunchDaemons/com.badapple.gatekeeper.plist 2>/dev/null; launchctl unload /Library/LaunchDaemons/com.badapple.supervisor.plist 2>/dev/null; sleep 2; launchctl load -w /Library/LaunchDaemons/com.badapple.gatekeeper.plist; launchctl load -w /Library/LaunchDaemons/com.badapple.mlx.plist; launchctl load -w /Library/LaunchDaemons/com.badapple.supervisor.plist" with administrator privileges'
```

Wait ~45 seconds for the model to load again.

## Common issues

### “Bad Apple can’t be opened” / Gatekeeper

The unsigned app is not notarized. Right-click `/Applications/Bad Apple.app` and choose **Open**, or go to **System Settings → Privacy & Security → Open Anyway**.

### Menu bar icon does not appear

```bash
launchctl print gui/$(id -u)/com.badapple.menubar
```

If it is not loaded, reinstall the agent:

```bash
src/platform/apple_desktop/install_menu_bar_agent.sh
```

### The dashboard shows the daemon is offline

Check that `/api/status` returns JSON:

```bash
curl -s http://127.0.0.1:8787/api/status
```

If that fails, the dashboard server is not running; restart the platform daemons above.

### Voice input does not work

- Make sure `badapple_tts_server.py` is running and the TTS LaunchAgent is loaded.
- Approve Microphone and Speech Recognition permissions when prompted.
- Check `/tmp/badapple_voice_debug.log` for transcription errors.

### Model loads slowly or runs out of memory

- Close other large apps before starting Bad Apple.
- Use `agent_client.py flush` or `agent_client.py unload all` to free VRAM.
- Disable `BADAPPLE_FAST_TIER` if you are troubleshooting inference.

## Built-in diagnostic tools

```bash
# Runtime and health
.venv/bin/python agent_client.py status

# Flush Metal cache
.venv/bin/python agent_client.py flush

# Unload optional models
.venv/bin/python agent_client.py unload all

# Air-gap / security certification
osascript -e 'do shell script "cd /Users/savag3/bad_apple && /Users/savag3/bad_apple/.venv/bin/python cert_suite.py" with administrator privileges'
```

## Reporting issues

If you hit a bug that the steps above do not fix:

1. Note the output of `agent_client.py status`.
2. Capture the last 100 lines of `/var/log/bad_apple_mlx_server.log`.
3. Include your macOS version, Mac model, and whether you are running the signed or unsigned build.
4. File an issue with that information.

Bad Apple is under active development. Logs rule everything around here.
