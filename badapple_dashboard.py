#!/usr/bin/env python3
"""Power and performance dashboard for Bad Apple.

Gathers local system metrics, the Bad Apple daemon process stats, and the
latest performance snapshot from the log. No cloud.

This module also exposes a local-only HTTP dashboard on 127.0.0.1:8787.

Endpoints:
    GET /                  - Dark-themed HTML dashboard that auto-refreshes every 2s.
    GET /api/snapshot      - JSON system/daemon snapshot (same as the system_dashboard tool).
    GET /api/status        - JSON daemon runtime/health/resources/models/breakers status.
    GET /api/tail?n=20     - Last N lines of /var/log/bad_apple_mlx_server.log.
    GET /api/ledger?n=20   - Last N entries of /var/lib/bad_apple/ledger.jsonl.
"""

import http.server
import json
import os
import secrets
import shutil
import socketserver
import subprocess
import threading
import time
import urllib.parse
from dataclasses import asdict
from pathlib import Path
from typing import Any

import psutil

import badapple_ambient
import badapple_mcp_marketplace
import badapple_ocular
from badapple_dashboard_data import (
    _run,  # noqa: F401
    _battery,
    _thermal,
    _badapple_proc,
    _latest_log_perf,
    _tail_lines,
    _tail_ledger,
    _voice_activity,
    _query_int,
    _load_mcp_servers,
)

# The MLXServer instance is set here by badapple_mlx_server.py at startup so
# the HTTP handler can return daemon-internal status.
_server_instance: Any | None = None


def _csrf_token_path() -> Path:
    data_dir = Path(os.environ.get("BADAPPLE_DATA_DIR") or "/var/lib/bad_apple").expanduser()
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir / "dashboard_csrf.token"


def _generate_csrf_token() -> str:
    return secrets.token_urlsafe(32)


def _get_csrf_token() -> str:
    token_path = _csrf_token_path()
    if token_path.is_file():
        return token_path.read_text(encoding="utf-8").strip()
    token = _generate_csrf_token()
    token_path.write_text(token, encoding="utf-8")
    return token


def _csrf_cookie() -> str:
    return f"csrf_token={_get_csrf_token()}; Path=/; SameSite=Strict"


def _check_csrf_token(headers: dict[str, str], body: dict[str, Any] | None = None) -> bool:
    token = headers.get("X-CSRF-Token") or headers.get("X-Csrf-Token")
    if not token and body is not None:
        token = body.get("csrf_token")
    return token == _get_csrf_token()


# New web UX assets live in the web/ directory next to this module.
WEB_ROOT = Path(__file__).with_name("web").resolve()
STATIC_ROOT = WEB_ROOT / "static"


def set_server_instance(instance: Any) -> None:
    """Bind the running MLXServer instance to the dashboard handlers."""
    global _server_instance
    _server_instance = instance


def snapshot() -> str:
    """Return a JSON string with the current system/daemon snapshot."""
    mem = psutil.virtual_memory()
    swap = psutil.swap_memory()
    disk = psutil.disk_usage("/")
    data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "cpu": {
            "count": psutil.cpu_count(logical=True),
            "percent": psutil.cpu_percent(interval=0.5),
            "load_avg_1m": os.getloadavg()[0],
        },
        "memory": {
            "total_gb": round(mem.total / 1e9, 2),
            "used_gb": round(mem.used / 1e9, 2),
            "free_gb": round(mem.free / 1e9, 2),
            "percent": mem.percent,
        },
        "swap": {
            "total_gb": round(swap.total / 1e9, 2),
            "used_gb": round(swap.used / 1e9, 2),
        },
        "disk_root": {
            "total_gb": round(disk.total / 1e9, 2),
            "used_gb": round(disk.used / 1e9, 2),
            "free_gb": round(disk.free / 1e9, 2),
            "percent": round((disk.used / disk.total) * 100, 1),
        },
        "battery": _battery(),
        "thermal_pressure": _thermal(),
        "bad_apple_process": _badapple_proc(),
        "latest_log_perf": _latest_log_perf(),
    }
    return json.dumps(data, indent=2, default=str)


def _daemon_status() -> dict[str, Any]:
    """Build the daemon runtime status payload."""
    if _server_instance is None:
        return {
            "runtime": None,
            "health": None,
            "resources": None,
            "active_models": [],
            "breakers": {},
            "autopilot": None,
            "fast_tier": None,
            "ambient_running": False,
            "ambient": None,
            "ocular_running": False,
            "ocular": None,
            "workspace": None,
            "p2p_enabled": False,
            "p2p_peers": [],
            "mcp_socket": os.environ.get("BADAPPLE_MCP_SOCKET", "/var/run/badapple/mcp.sock"),
            "active_persona": "default",
            "timestamp": time.time(),
        }
    try:
        ambient = json.loads(badapple_ambient.get_context()) if badapple_ambient._CONTEXT_FILE.is_file() else None
    except (json.JSONDecodeError, TypeError, ValueError, AttributeError, OSError):
        ambient = None
    return {
        "runtime": _server_instance.runtime.status(),
        "health": _server_instance.health.snapshot(),
        "resources": _server_instance.resources.snapshot(),
        "active_models": _server_instance.active_models(),
        "breakers": {
            name: asdict(cb.snapshot())
            for name, cb in _server_instance.breakers.items()
        },
        "autopilot": _server_instance.policy.autopilot,
        "fast_tier": _server_instance.fast_tier_enabled,
        "ambient_running": badapple_ambient.is_running(),
        "ambient": ambient,
        "ocular_running": badapple_ocular.is_running(),
        "ocular": badapple_ocular.status() if badapple_ocular.is_running() or badapple_ocular.OCULAR_CONTEXT.is_file() else None,
        "workspace": str(_server_instance.workspace.path) if _server_instance.workspace.path else None,
        "airgap": _server_instance.airgap,
        "p2p_enabled": _server_instance.p2p is not None and _server_instance.p2p.is_running(),
        "p2p_peers": _server_instance.p2p.get_peers() if _server_instance.p2p is not None and _server_instance.p2p.is_running() else [],
        "mcp_socket": os.environ.get("BADAPPLE_MCP_SOCKET", "/var/run/badapple/mcp.sock"),
        "fast_model": _server_instance.fast_model_info,
        "active_persona": _server_instance.personas.active,
        "hibernating": _server_instance.hibernating,
        "idle_seconds": round(time.time() - _server_instance.last_activity, 1),
        "hibernate_after": _server_instance.hibernate_after,
        "timestamp": time.time(),
    }


def _recent_ambient_facts(n: int = 10) -> list[dict[str, Any]]:
    if _server_instance is None:
        return []
    try:
        facts = _server_instance.memory._state.get("facts", [])
        ambient = [f for f in facts if f.get("source") == "ambient"]
        return [
            {
                "fact": f.get("text", ""),
                "created": f.get("created", ""),
            }
            for f in ambient[-n:]
        ]
    except Exception as e:  # noqa: BLE001
        print(f"[dashboard] ambient facts error: {e}", flush=True)
        return []


def _recent_ambient_episodes(n: int = 10) -> list[dict[str, Any]]:
    if _server_instance is None:
        return []
    try:
        eps = _server_instance.memory._state.get("episodes", [])
        ambient = [e for e in eps if e.get("context", {}).get("source") == "ambient"]
        return [
            {
                "user": e.get("user", ""),
                "context": e.get("context", {}),
                "created": e.get("created", ""),
            }
            for e in ambient[-n:]
        ]
    except Exception as e:  # noqa: BLE001
        print(f"[dashboard] ambient episodes error: {e}", flush=True)
        return []


_PERSONA_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Bad Apple Persona Editor</title>
    <style>
        :root { --bg: #0d0d0d; --card: #151515; --text: #e6e6e6; --muted: #888; --accent: #4ea043; --danger: #c94e4e; }
        * { box-sizing: border-box; }
        body { margin: 0; background: var(--bg); color: var(--text); font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; line-height: 1.5; height: 100vh; display: flex; flex-direction: column; }
        header { padding: 1rem 1.5rem; border-bottom: 1px solid #222; display: flex; justify-content: space-between; align-items: center; }
        h1 { margin: 0; font-size: 1.3rem; }
        a { color: var(--accent); text-decoration: none; }
        #main { flex: 1; padding: 1.5rem; display: flex; flex-direction: column; gap: 1rem; max-width: 900px; width: 100%; margin: 0 auto; }
        label { color: var(--muted); font-size: 0.9rem; }
        select, textarea, button { background: #1a1a1a; border: 1px solid #333; color: var(--text); padding: 0.75rem; border-radius: 8px; font-size: 1rem; }
        textarea { flex: 1; font-family: ui-monospace, monospace; line-height: 1.4; resize: none; }
        button { background: var(--accent); color: #000; font-weight: 600; cursor: pointer; border: none; }
        #status { color: var(--muted); font-size: 0.85rem; min-height: 1.2rem; }
    </style>
</head>
<body>
    <header>
        <h1>Persona Editor</h1>
        <a href="/">Dashboard</a> <a href="/chat">Chat</a>
    </header>
    <div id="main">
        <label for="persona">Active persona</label>
        <select id="persona"></select>
        <label for="prompt">System prompt</label>
        <textarea id="prompt" placeholder="System prompt..."></textarea>
        <button onclick="save()">Save &amp; switch</button>
        <div id="status"></div>
    </div>
    <script>
        const personaSel = document.getElementById('persona');
        const promptEl = document.getElementById('prompt');
        const statusEl = document.getElementById('status');

        async function loadList() {
            const r = await fetch('/api/personas');
            const data = await r.json();
            personaSel.innerHTML = '';
            for (const name of data.personas) {
                const opt = document.createElement('option');
                opt.value = name;
                opt.textContent = name;
                if (name === data.active) opt.selected = true;
                personaSel.appendChild(opt);
            }
            promptEl.value = data.prompt || '';
        }

        async function save() {
            statusEl.textContent = 'Saving...';
            const r = await fetch('/api/personas', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name: personaSel.value, prompt: promptEl.value })
            });
            const data = await r.json();
            statusEl.textContent = data.ok ? 'Saved. Active next query.' : (data.error || 'Error');
        }

        loadList();
    </script>
</body>
</html>
"""


_DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Bad Apple Dashboard</title>
    <style>
        :root { --bg: #0d0d0d; --card: #151515; --card2: #1a1a1a; --text: #e6e6e6; --muted: #888; --accent: #4ea043; --warn: #d9a441; --danger: #c94e4e; }
        * { box-sizing: border-box; }
        body { margin: 0; background: var(--bg); color: var(--text); font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; line-height: 1.4; }
        header { padding: 1.25rem 1.5rem; border-bottom: 1px solid #222; display: flex; justify-content: space-between; align-items: center; }
        h1 { margin: 0; font-size: 1.4rem; }
        #last-update { color: var(--muted); font-size: 0.85rem; }
        main { padding: 1.5rem; display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 1rem; }
        .card { background: var(--card); border: 1px solid #222; border-radius: 10px; padding: 1rem; }
        .card h2 { margin: 0 0 0.6rem 0; font-size: 1rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.03em; }
        .value { font-size: 1.75rem; font-weight: 600; }
        .sub { color: var(--muted); font-size: 0.85rem; margin-top: 0.25rem; }
        .ok { color: var(--accent); }
        .warn { color: var(--warn); }
        .bad { color: var(--danger); }
        .muted { color: var(--muted); }
        pre { background: var(--card2); border-radius: 6px; padding: 0.75rem; overflow-x: auto; white-space: pre-wrap; word-break: break-word; font-size: 0.8rem; max-height: 220px; overflow-y: auto; }
        table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
        th, td { text-align: left; padding: 0.3rem 0.5rem; }
        th { color: var(--muted); border-bottom: 1px solid #333; }
        tr:nth-child(even) { background: rgba(255,255,255,0.03); }
        .breaker-list { display: flex; flex-wrap: wrap; gap: 0.5rem; }
        .breaker { background: var(--card2); border-radius: 6px; padding: 0.35rem 0.6rem; font-size: 0.85rem; }
        .breaker span { font-weight: 600; }
    </style>
</head>
<body>
    <header>
        <h1>Bad Apple Dashboard</h1>
        <div id="last-update">waiting...</div>
    </header>
    <main>
        <div class="card">
            <h2>Runtime Mode</h2>
            <div class="value" id="runtime-mode">—</div>
            <div class="sub" id="runtime-flags"></div>
        </div>
        <div class="card">
            <h2>Memory Used / Total</h2>
            <div class="value" id="memory-usage">—</div>
            <div class="sub" id="memory-percent">—</div>
        </div>
        <div class="card">
            <h2>Active Models</h2>
            <div class="value" id="active-models">—</div>
            <div class="sub" id="active-models-count"></div>
        </div>
        <div class="card">
            <h2>Breakers</h2>
            <div id="breakers" class="breaker-list"></div>
        </div>
        <div class="card">
            <h2>Battery</h2>
            <div class="value" id="battery">—</div>
            <div class="sub" id="battery-source"></div>
        </div>
        <div class="card">
            <h2>Latest Perf</h2>
            <div class="sub" id="latest-perf">—</div>
        </div>
        <div class="card" style="grid-column: 1 / -1;">
            <h2>Workspace</h2>
            <div class="value" id="workspace">—</div>
            <pre id="workspace-summary" class="muted" style="background: transparent; padding: 0;">—</pre>
        </div>
        <div class="card" style="grid-column: 1 / -1;">
            <h2>Ambient Context</h2>
            <div id="ambient"></div>
        </div>
        <div class="card" style="grid-column: 1 / -1;">
            <h2>P2P Peers</h2>
            <div id="p2p-peers"></div>
        </div>
        <div class="card" style="grid-column: 1 / -1;">
            <h2>Log Tail</h2>
            <pre id="log-tail">—</pre>
        </div>
        <div class="card" style="grid-column: 1 / -1;">
            <h2>Ledger Tail</h2>
            <div id="ledger-tail"></div>
        </div>
    </main>
    <script>
        const $ = id => document.getElementById(id);
        function fmtMem(used, total) {
            if (used == null || total == null) return '—';
            return `${used.toFixed(2)} / ${total.toFixed(2)} GB`;
        }
        function setText(id, text, cls) {
            const el = $(id);
            el.textContent = text;
            el.className = 'value' + (cls ? ' ' + cls : '');
        }
        function renderStatus(data) {
            const rt = data.runtime || {};
            const mode = rt.mode || 'unknown';
            const flags = [];
            if (rt.killed) flags.push('killed');
            if (rt.private_mode) flags.push('private');
            if (rt.safe_mode_reason) flags.push(`safe: ${rt.safe_mode_reason}`);
            if (data.autopilot) flags.push('autopilot');
            if (data.fast_tier) flags.push('fast tier');
            if (data.p2p_enabled) flags.push('p2p on');
            setText('runtime-mode', mode);
            $('runtime-flags').textContent = flags.length ? flags.join(' · ') : 'normal';

            const res = data.resources || {};
            const snapMem = (window._lastSnapshot && window._lastSnapshot.memory) || {};
            const used = snapMem.used_gb;
            const total = snapMem.total_gb;
            $('memory-usage').textContent = fmtMem(used, total);
            const pct = res.memory_percent != null ? res.memory_percent : snapMem.percent;
            $('memory-percent').textContent = pct != null ? `${pct}% used` : '';

            const models = data.active_models || [];
            $('active-models').textContent = models.length ? models.join(', ') : 'none';
            $('active-models-count').textContent = `${models.length} loaded`;

            const breakers = data.breakers || {};
            const bContainer = $('breakers');
            bContainer.innerHTML = '';
            Object.entries(breakers).forEach(([name, b]) => {
                const div = document.createElement('div');
                div.className = 'breaker';
                let cls = 'ok';
                if (b.state === 'open') cls = 'bad';
                else if (b.state === 'half_open') cls = 'warn';
                div.innerHTML = `<span class="${cls}">${name}</span> · ${b.state} (${b.failures})`;
                bContainer.appendChild(div);
            });

            const bat = res.battery_percent != null ? res.battery_percent : (window._lastSnapshot && window._lastSnapshot.battery && window._lastSnapshot.battery.percent);
            const plugged = res.plugged_in ? 'plugged in' : (window._lastSnapshot && window._lastSnapshot.battery && window._lastSnapshot.battery.source);
            $('battery').textContent = bat != null ? `${bat}%` : '—';
            $('battery-source').textContent = plugged ? (plugged === 'ac' ? 'AC power' : plugged === 'battery' ? 'on battery' : plugged) : '';
        }
        function renderSnapshot(data) {
            window._lastSnapshot = data;
            const perf = data.latest_log_perf || {};
            $('latest-perf').textContent = perf.raw || '—';
        }
        function renderWorkspace(status) {
            const ws = status.workspace;
            $('workspace').textContent = ws ? ws.replace(/^\//, '').split('/').pop() : 'No workspace set';
            $('workspace-summary').textContent = 'Set workspace to attach project context, babe.';
        }
        function renderAmbient(status) {
            const container = $('ambient');
            const running = status.ambient_running;
            const ambient = status.ambient;
            if (!running || !ambient || !ambient.app) {
                container.innerHTML = `<div class="breaker"><span class="${running ? 'warn' : 'bad'}">${running ? 'running' : 'stopped'}</span></div>`;
                return;
            }
            container.innerHTML = `<div class="breaker"><span class="ok">running</span></div><div>app: <strong>${ambient.app}</strong></div><div>window: <span class="muted">${ambient.window}</span></div><div class="muted">${ambient.timestamp}</div>`;
        }
        function renderP2P(peers) {
            const container = $('p2p-peers');
            if (!peers || !peers.length) { container.innerHTML = '<div class="muted">—</div>'; return; }
            container.innerHTML = peers.map(p => `<div class="breaker"><span>${p}</span></div>`).join('');
        }
        function renderTail(lines) {
            $('log-tail').textContent = lines.length ? lines.join('') : '—';
        }
        function renderLedger(entries) {
            const container = $('ledger-tail');
            if (!entries.length) { container.innerHTML = '<div class="muted">—</div>'; return; }
            let html = '<table><thead><tr><th>type</th><th>time</th><th>summary</th></tr></thead><tbody>';
            entries.forEach(e => {
                const type = e.event_type || e.type || '—';
                const ts = e.timestamp || e.t || '—';
                const summary = JSON.stringify(e).slice(0, 120) + (JSON.stringify(e).length > 120 ? '…' : '');
                html += `<tr><td>${type}</td><td class="muted">${ts}</td><td class="muted">${summary.replace(/</g, '&lt;')}</td></tr>`;
            });
            html += '</tbody></table>';
            container.innerHTML = html;
        }
        async function update() {
            try {
                const [status, snap, tail, ledger] = await Promise.all([
                    fetch('/api/status').then(r => r.ok ? r.json() : null),
                    fetch('/api/snapshot').then(r => r.ok ? r.json() : null),
                    fetch('/api/tail?n=20').then(r => r.ok ? r.json() : {lines: []}),
                    fetch('/api/ledger?n=20').then(r => r.ok ? r.json() : {entries: []})
                ]);
                if (snap) renderSnapshot(snap);
                if (status) {
                    renderStatus(status);
                    renderWorkspace(status);
                    renderAmbient(status);
                    renderP2P(status.p2p_peers);
                }
                renderTail(tail.lines || []);
                renderLedger(ledger.entries || []);
                $('last-update').textContent = 'updated ' + new Date().toLocaleTimeString();
            } catch (e) {
                $('last-update').textContent = 'refresh failed: ' + e.message;
            }
        }
        update();
        setInterval(update, 2000);
    </script>
</body>
</html>
"""


_CHAT_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Bad Apple Chat</title>
    <style>
        :root { --bg: #0d0d0d; --card: #151515; --text: #e6e6e6; --muted: #888; --accent: #4ea043; --danger: #c94e4e; }
        * { box-sizing: border-box; }
        body { margin: 0; background: var(--bg); color: var(--text); font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; line-height: 1.5; height: 100vh; display: flex; flex-direction: column; }
        header { padding: 1rem 1.5rem; border-bottom: 1px solid #222; display: flex; justify-content: space-between; align-items: center; }
        h1 { margin: 0; font-size: 1.3rem; }
        a { color: var(--accent); text-decoration: none; }
        #chat { flex: 1; overflow-y: auto; padding: 1.5rem; display: flex; flex-direction: column; gap: 0.75rem; }
        .msg { max-width: 80%; padding: 0.8rem 1rem; border-radius: 12px; white-space: pre-wrap; word-break: break-word; }
        .user { align-self: flex-end; background: #1f3a1f; }
        .bot { align-self: flex-start; background: #222; }
        .error { align-self: flex-start; background: var(--danger); color: #fff; }
        #controls { padding: 1rem 1.5rem; border-top: 1px solid #222; display: flex; gap: 0.5rem; }
        #prompt { flex: 1; background: #1a1a1a; border: 1px solid #333; color: var(--text); padding: 0.75rem 1rem; border-radius: 8px; font-size: 1rem; }
        button { background: var(--accent); border: none; color: #000; padding: 0.75rem 1.25rem; border-radius: 8px; font-weight: 600; cursor: pointer; }
        button:disabled { opacity: 0.5; }
        #status { color: var(--muted); font-size: 0.85rem; }
        .tool { font-size: 0.8rem; color: var(--muted); margin-top: 0.25rem; }
    </style>
</head>
<body>
    <header>
        <h1>Bad Apple Chat</h1>
        <a href="/">Dashboard</a>
    </header>
    <div id="chat"></div>
    <div id="controls">
        <input type="text" id="prompt" placeholder="Say something to Bad Apple..." autocomplete="off" autofocus>
        <button id="send" onclick="send()">Send</button>
    </div>
    <script>
        const chat = document.getElementById('chat');
        const promptEl = document.getElementById('prompt');
        const sendBtn = document.getElementById('send');
        function append(text, cls) {
            const d = document.createElement('div');
            d.className = 'msg ' + cls;
            d.textContent = text;
            chat.appendChild(d);
            chat.scrollTop = chat.scrollHeight;
            return d;
        }
        async function send() {
            const p = promptEl.value.trim();
            if (!p) return;
            append(p, 'user');
            promptEl.value = '';
            sendBtn.disabled = true;
            const waiting = append('...', 'bot');
            try {
                const r = await fetch('/api/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ prompt: p, stream: true })
                });
                const reader = r.body.getReader();
                const decoder = new TextDecoder();
                let buffer = '';
                let started = false;
                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;
                    buffer += decoder.decode(value, { stream: true });
                    const chunks = buffer.split('\n\n');
                    buffer = chunks.pop();
                    for (const chunk of chunks) {
                        const line = chunk.split('\n').find(l => l.startsWith('data:'));
                        if (!line) continue;
                        const data = line.slice(5).trim();
                        let msg;
                        try { msg = JSON.parse(data); } catch (e) { continue; }
                        if (!started) { waiting.textContent = ''; started = true; }
                        if (msg.type === 'token') {
                            waiting.textContent += msg.text;
                            chat.scrollTop = chat.scrollHeight;
                        } else if (msg.type === 'tool') {
                            const t = document.createElement('div');
                            t.className = 'tool';
                            t.textContent = `⚡ ${msg.tool}`;
                            waiting.appendChild(t);
                            chat.scrollTop = chat.scrollHeight;
                        } else if (msg.type === 'done') {
                            const imgMatch = msg.text.match(/^Generated image:\s*(.+\.png)$/);
                            if (imgMatch) {
                                waiting.textContent = 'Generated image:';
                                const img = document.createElement('img');
                                img.src = '/api/image/' + encodeURIComponent(imgMatch[1].split('/').pop());
                                img.style.maxWidth = '100%';
                                img.style.borderRadius = '8px';
                                img.style.marginTop = '0.5rem';
                                waiting.appendChild(img);
                            } else {
                                waiting.textContent = msg.text;
                            }
                            if (msg.metrics && !imgMatch) {
                                const t = document.createElement('div');
                                t.className = 'tool';
                                t.textContent = JSON.stringify(msg.metrics);
                                waiting.appendChild(t);
                            }
                            chat.scrollTop = chat.scrollHeight;
                        } else if (msg.type === 'error') {
                            waiting.textContent = msg.error;
                            waiting.className = 'msg error';
                        }
                    }
                }
            } catch (e) {
                waiting.textContent = 'Error: ' + e;
                waiting.className = 'msg error';
            }
            sendBtn.disabled = false;
            promptEl.focus();
        }
        promptEl.addEventListener('keydown', e => { if (e.key === 'Enter') send(); });
    </script>
</body>
</html>
"""


def _repo_root() -> Path:
    """Return the directory that contains this dashboard module."""
    return Path(__file__).resolve().parent


def _personas_file() -> Path:
    return (
        Path(os.environ["BADAPPLE_PERSONAS_FILE"]).expanduser()
        if os.environ.get("BADAPPLE_PERSONAS_FILE")
        else _repo_root() / "personas.json"
    )


def _prompt_file() -> Path:
    return _repo_root() / "prompt.txt"


def _active_persona() -> str:
    try:
        if _server_instance is not None:
            return _server_instance.personas.active
    except Exception as e:  # noqa: BLE001 - logged
        print(f"[dashboard] error failed: {e}", flush=True)
    return "default"


def _list_personas() -> tuple[list[str], str, str]:
    pf = _personas_file()
    active = _active_persona()
    personas = {"default": "Bad Apple"}
    if pf.is_file():
        try:
            data = json.loads(pf.read_text(encoding="utf-8"))
            for k in data:
                personas[k] = data[k].get("name", k)
        except (json.JSONDecodeError, TypeError, ValueError, AttributeError, OSError, LookupError) as e:
            print(f"[dashboard] loads failed: {e}", flush=True)
    prompt, _ = _load_persona_prompt(active)
    return list(personas.keys()), active, prompt


def _load_persona_prompt(name: str) -> tuple[str, Path | None]:
    active_file: Path | None = None
    if name == "default":
        active_file = _prompt_file()
        if active_file.is_file():
            return active_file.read_text(encoding="utf-8").strip(), active_file
        return "", active_file
    pf = _personas_file()
    if pf.is_file():
        data = json.loads(pf.read_text(encoding="utf-8"))
        if name in data:
            return data[name].get("system_prompt", ""), pf
    return "", None


def _save_persona_prompt(name: str, prompt: str) -> None:
    if name == "default":
        _prompt_file().write_text(prompt, encoding="utf-8")
        return
    pf = _personas_file()
    if pf.is_file():
        data = json.loads(pf.read_text(encoding="utf-8"))
        if name in data:
            data[name]["system_prompt"] = prompt
            pf.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _badapple_cli() -> str | None:
    """Return the badapple CLI path: prefer the current build, then PATH."""
    candidate = _repo_root() / "target" / "release" / "badapple"
    if candidate.is_file():
        return str(candidate)
    cli = shutil.which("badapple")
    return cli


def _run_cli(prompt: str, max_tokens: int = 240) -> str:
    """Call the local badapple CLI as a simple bridge for the web chat."""
    exe = _badapple_cli()
    if not exe:
        return "Error: badapple CLI not found."
    try:
        result = subprocess.run(
            [exe, "-n", str(max_tokens), prompt],
            capture_output=True,
            text=True,
            timeout=120,
        check=False)
        if result.returncode == 0:
            return result.stdout.strip()
        return f"Error: {result.stderr.strip() or 'CLI failed'}"
    except (subprocess.SubprocessError, OSError, ValueError) as e:
        return f"Error: {e}"


def _stream_cli(handler: http.server.BaseHTTPRequestHandler, prompt: str, max_tokens: int = 240) -> None:
    """Stream the badapple CLI --json output as Server-Sent Events."""

    exe = _badapple_cli()
    if not exe:
        handler.send_response(503)
        handler.send_header("Content-Type", "text/plain")
        handler.end_headers()
        handler.wfile.write(b"Error: badapple CLI not found.")
        return
    handler.send_response(200)
    handler.send_header("Content-Type", "text/event-stream; charset=utf-8")
    handler.send_header("Cache-Control", "no-cache")
    handler.send_header("Connection", "close")
    handler.end_headers()

    def _emit(obj: dict) -> None:
        try:
            line = json.dumps(obj, default=str)
            handler.wfile.write(f"data: {line}\n\n".encode())
            handler.wfile.flush()
        except Exception:  # noqa: BLE001,S110 - cleanup
            pass

    try:
        proc = subprocess.Popen(
            [exe, "--json", "-n", str(max_tokens), prompt],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        while True:
            line = proc.stdout.readline()
            if not line:
                if proc.poll() is not None:
                    break
                time.sleep(0.05)
                continue
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            _emit(msg)
            if msg.get("type") == "done":
                break
        err = proc.stderr.read().strip() if proc.stderr else ""
        if err and proc.returncode != 0:
            _emit({"type": "error", "error": err or "CLI failed"})
    except (subprocess.SubprocessError, OSError, ValueError) as e:
        _emit({"type": "error", "error": str(e)})
    try:
        handler.wfile.flush()
        handler.connection.close()
    except Exception:  # noqa: BLE001,S110 - cleanup
        pass


class DashboardHandler(http.server.BaseHTTPRequestHandler):
    """Local-only request handler for the Bad Apple observability dashboard."""

    def log_message(self, fmt: str, *args: Any) -> None:
        # Keep the daemon log clean; dashboard access is not logged per-request.
        pass

    def _send_json(self, data: Any, status: int = 200) -> None:
        body = json.dumps(data, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._set_csrf_cookie()
        self.end_headers()
        self.wfile.write(body)

    def _set_csrf_cookie(self) -> None:
        self.send_header("Set-Cookie", _csrf_cookie())

    def _send_html(self, html: str, status: int = 200) -> None:
        body = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self._set_csrf_cookie()
        self.end_headers()
        self.wfile.write(body)

    def _send_text(self, text: str, status: int = 200) -> None:
        body = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._set_csrf_cookie()
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, file_path: Path, content_type: str, status: int = 200) -> None:
        body = file_path.read_bytes()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        self._set_csrf_cookie()
        self.end_headers()
        self.wfile.write(body)

    def _serve_static(self, relative: str) -> None:
        target = (STATIC_ROOT / relative).resolve()
        if not str(target).startswith(str(STATIC_ROOT.resolve())):
            self._send_text("Not found", 404)
            return
        if not target.is_file():
            self._send_text("Not found", 404)
            return
        content_type = "text/plain"
        if target.suffix == ".css":
            content_type = "text/css"
        elif target.suffix == ".js":
            content_type = "application/javascript"
        elif target.suffix in (".png", ".jpg", ".jpeg", ".gif", ".svg"):
            content_type = f"image/{target.suffix.lstrip('.')}"
        self._send_file(target, content_type)

    def _serve_html_file(self, name: str) -> None:
        file_path = WEB_ROOT / name
        if file_path.is_file():
            html = file_path.read_text(encoding="utf-8")
            token = _get_csrf_token()
            html = html.replace("{{CSRF_TOKEN}}", token)
            if "</head>" in html:
                html = html.replace(
                    "</head>",
                    f'<meta name="csrf-token" content="{token}">\n<script src="/static/csrf.js?v=7"></script>\n</head>',
                )
            self._send_html(html)
        else:
            self._send_text("Not found", 404)

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path.startswith("/static/"):
            self._serve_static(path[8:].lstrip("/"))
            return

        if path == "/":
            self._serve_html_file("index.html")
            return
        if path == "/api/csrf":
            self._send_json({"csrf_token": _get_csrf_token()})
            return
        if path == "/models" or path == "/models.html":
            self._serve_html_file("models.html")
            return

        if path == "/splash":
            self._serve_html_file("splash.html")
            return

        if path in ("/chat", "/persona", "/settings", "/logs", "/dashboard"):
            self._serve_html_file("index.html")
            return
        if path == "/ambient" or path == "/ambient.html":
            self._serve_html_file("ambient.html")
            return
        if path == "/mcp" or path == "/mcp.html":
            self._serve_html_file("mcp.html")
            return
        if path == "/control" or path == "/control.html":
            self._serve_html_file("control.html")
            return

        if path == "/api/snapshot":
            self._send_json(json.loads(snapshot()))
            return
        if path == "/api/status":
            self._send_json(_daemon_status())
            return
        if path == "/api/tail":
            n = _query_int(self.path, "n", 20)
            lines = _tail_lines("/var/log/bad_apple_mlx_server.log", n)
            self._send_json({"lines": lines})
            return
        if path == "/api/ledger":
            n = _query_int(self.path, "n", 20)
            entries = _tail_ledger("/var/lib/bad_apple/ledger.jsonl", n)
            self._send_json({"entries": entries})
            return
        if path == "/api/voice":
            n = _query_int(self.path, "n", 20)
            self._send_json({"events": _voice_activity(n)})
            return
        if path == "/api/personas":
            try:
                names, active, prompt = _list_personas()
                name = urllib.parse.parse_qs(parsed.query).get("name", [active])[0]
                prompt, _ = _load_persona_prompt(name)
                self._send_json({"personas": names, "active": active, "prompt": prompt})
            except (LookupError, TypeError, ValueError) as e:
                self._send_json({"error": str(e)}, 500)
            return
        if path == "/api/mcp_servers":
            try:
                data = _load_mcp_servers()
                self._send_json(data)
            except Exception as e:  # noqa: BLE001 - catch-all wrapper
                self._send_json({"error": str(e)}, 500)
            return
        if path == "/api/mcp_registry":
            try:
                catalog_path = Path(__file__).with_name("mcp_registry.json")
                self._send_json({"servers": badapple_mcp_marketplace.list_catalog_servers(catalog_path)})
            except Exception as e:  # noqa: BLE001 - catch-all wrapper
                self._send_json({"error": str(e)}, 500)
            return
        if path == "/api/ocular":
            try:
                self._send_json(badapple_ocular.status())
            except Exception as e:  # noqa: BLE001 - catch-all wrapper
                self._send_json({"error": str(e)}, 500)
            return
        if path == "/api/ocular/screen.png":
            try:
                screen_path = badapple_ocular.OCULAR_SCREEN
                if screen_path.is_file():
                    self._send_file(screen_path, "image/png")
                else:
                    self._send_text("No screen capture yet", 404)
            except Exception as e:  # noqa: BLE001 - catch-all wrapper
                self._send_json({"error": str(e)}, 500)
            return
        if path.startswith("/api/image/"):
            filename = urllib.parse.unquote(path[11:])
            safe = Path(filename).name
            img_dir = Path(os.environ.get("BADAPPLE_DATA_DIR", "/var/lib/bad_apple")) / "generated_images"
            img_path = img_dir / safe
            if img_path.is_file():
                self._send_file(img_path, "image/png")
            else:
                self._send_text("Image not found", 404)
            return

        if path == "/api/models":
            try:
                model_id = urllib.parse.parse_qs(parsed.query).get("model_id", [None])[0]
                if _server_instance is not None and hasattr(_server_instance, "model_manager"):
                    self._send_json(_server_instance.model_manager.status(model_id))
                else:
                    self._send_json({"error": "daemon not running"}, 503)
            except Exception as e:  # noqa: BLE001
                self._send_json({"error": str(e)}, 500)
            return

        if path == "/api/agents":
            try:
                if _server_instance is not None and hasattr(_server_instance, "list_agent_tasks"):
                    self._send_json({"tasks": _server_instance.list_agent_tasks()})
                else:
                    self._send_json({"error": "daemon not running"}, 503)
            except Exception as e:  # noqa: BLE001
                self._send_json({"error": str(e)}, 500)
            return

        if path == "/api/ambient":
            try:
                import badapple_ambient

                self._send_json({
                    "running": badapple_ambient.is_running(),
                    "context": json.loads(badapple_ambient.get_context()) if badapple_ambient._CONTEXT_FILE.is_file() else {},
                    "recent_facts": _recent_ambient_facts(10),
                    "recent_episodes": _recent_ambient_episodes(10),
                })
            except Exception as e:  # noqa: BLE001
                self._send_json({"error": str(e)}, 500)
            return

        if path == "/agents":
            self._serve_html_file("agents.html")
            return

        # SPA fallback
        self._serve_html_file("index.html")

    def do_POST(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        length = int(self.headers.get("Content-Length", 0))
        body_bytes = self.rfile.read(length) if length > 0 else b"{}"
        try:
            self._post_body = json.loads(body_bytes.decode("utf-8"))
        except json.JSONDecodeError:
            self._post_body = {}

        if not _check_csrf_token({k: v for k, v in self.headers.items()}, self._post_body):
            self._send_json({"error": "invalid or missing CSRF token"}, 403)
            return

        def _read_body() -> dict:
            return self._post_body

        if path == "/api/chat":
            try:
                payload = _read_body()
                prompt = payload.get("prompt", "").strip()
                if not prompt:
                    self._send_json({"error": "prompt is required"}, 400)
                    return
                max_tokens = max(1, min(4000, int(payload.get("max_tokens", 240))))
                if payload.get("stream"):
                    _stream_cli(self, prompt, max_tokens)
                    return
                text = _run_cli(prompt, max_tokens)
                self._send_json({"text": text})
            except Exception as e:  # noqa: BLE001 - catch-all wrapper
                self._send_json({"error": f"bad request: {e}"}, 400)
            return

        if path == "/api/personas":
            try:
                payload = _read_body()
                name = payload.get("name", "").strip()
                prompt = payload.get("prompt", "")
                if not name:
                    self._send_json({"error": "name is required"}, 400)
                    return
                _save_persona_prompt(name, prompt)
                _run_cli(f"switch to {name}")
                self._send_json({"ok": True})
            except Exception as e:  # noqa: BLE001 - catch-all wrapper
                self._send_json({"error": f"bad request: {e}"}, 400)
            return

        if path == "/api/workspace":
            try:
                payload = _read_body()
                path = payload.get("path", "").strip()
                if _server_instance is not None:
                    result = _server_instance.workspace.set(path)
                else:
                    result = "Workspace set to " + path
                self._send_json({"ok": True, "result": result})
            except Exception as e:  # noqa: BLE001 - catch-all wrapper
                self._send_json({"error": str(e)}, 500)
            return

        if path == "/api/airgap":
            try:
                if _server_instance is None:
                    self._send_json({"error": "daemon not running"}, 503)
                    return
                if self.command == "GET":
                    self._send_json({"airgap": _server_instance.airgap})
                    return
                if self.command == "POST":
                    payload = _read_body()
                    enabled = bool(payload.get("enabled", False))
                    _server_instance._set_airgap(enabled)
                    self._send_json({"ok": True, "airgap": _server_instance.airgap})
                    return
                self._send_json({"error": "method not allowed"}, 405)
            except Exception as e:  # noqa: BLE001 - catch-all wrapper
                self._send_json({"error": str(e)}, 500)
            return

        if path == "/api/control":
            try:
                payload = _read_body()
                command = payload.get("command", "").strip()
                if not command:
                    self._send_json({"error": "command required"}, 400)
                    return
                result = _run_cli(command, max_tokens=512)
                self._send_json({"ok": True, "result": result})
            except Exception as e:  # noqa: BLE001 - catch-all wrapper
                self._send_json({"error": str(e)}, 500)
            return

        if path == "/api/mcp_servers":
            try:
                payload = _read_body()
                action = payload.get("action", "")
                if action == "add":
                    result = badapple_mcp_marketplace.add_mcp_server(
                        payload.get("name", ""), payload.get("command", "")
                    )
                elif action == "remove":
                    result = badapple_mcp_marketplace.remove_mcp_server(payload.get("name", ""))
                elif action == "install":
                    catalog_path = Path(__file__).with_name("mcp_registry.json")
                    result = badapple_mcp_marketplace.install_catalog_server(payload.get("name", ""), catalog_path)
                elif action == "list":
                    result = _load_mcp_servers()
                elif action == "tools":
                    result = {"tools": badapple_mcp_marketplace._MARKETPLACE.list_tools(payload.get("name", ""))}
                elif action == "invoke":
                    server = payload.get("server", "")
                    tool = payload.get("tool", "")
                    arguments = payload.get("args", {})
                    result = badapple_mcp_marketplace._MARKETPLACE.invoke(server, tool, arguments)
                    self._send_json({"ok": True, "result": result})
                    return
                elif action == "stop_all":
                    badapple_mcp_marketplace._MARKETPLACE.stop_all()
                    result = "Stopped all MCP servers."
                else:
                    self._send_json({"error": "unknown action"}, 400)
                    return
                self._send_json({"ok": True, "result": result} if isinstance(result, str) else result)
            except Exception as e:  # noqa: BLE001 - catch-all wrapper
                self._send_json({"error": str(e)}, 500)
            return

        if path == "/api/ocular":
            try:
                payload = _read_body()
                action = payload.get("action", "")
                if action == "start":
                    result = badapple_ocular.start(
                        float(payload.get("capture_interval", 5)),
                        float(payload.get("describe_interval", 0)),
                        payload.get("prompt"),
                    )
                elif action == "stop":
                    result = badapple_ocular.stop()
                elif action == "capture":
                    result = badapple_ocular.capture_now(payload.get("prompt"))
                else:
                    self._send_json({"error": "unknown action"}, 400)
                    return
                self._send_json({"ok": True, "result": result} if isinstance(result, str) else result)
            except Exception as e:  # noqa: BLE001 - catch-all wrapper
                self._send_json({"error": str(e)}, 500)
            return

        if path == "/api/models":
            try:
                payload = _read_body()
                action = payload.get("action", "")
                if _server_instance is None or not hasattr(_server_instance, "model_manager"):
                    self._send_json({"error": "daemon not running"}, 503)
                    return
                if action == "status":
                    model_id = payload.get("model_id")
                    self._send_json(_server_instance.model_manager.status(model_id))
                elif action == "download":
                    model_id = payload.get("model_id", "").strip()
                    if not model_id:
                        self._send_json({"error": "model_id required"}, 400)
                        return
                    if not _server_instance.model_manager.allow_downloads:
                        self._send_json({"error": "downloads disabled"}, 403)
                        return
                    # Memory check before pre-download (does not block the download).
                    admit = _server_instance.admit_model(model_id, auto_unload=False)
                    result = _server_instance.model_manager.start_download(model_id)
                    self._send_json({**result, "memory_check": admit})
                elif action == "allow_downloads":
                    enabled = bool(payload.get("enabled", False))
                    if _server_instance.airgap and enabled:
                        self._send_json({"error": "downloads cannot be enabled while air-gap mode is on"}, 403)
                        return
                    _server_instance.model_manager.set_allow_downloads(enabled)
                    self._send_json({"ok": True, "allow_downloads": enabled})
                elif action == "refresh":
                    model_id = payload.get("model_id")
                    if model_id:
                        self._send_json(_server_instance.model_manager.refresh_cache_status(str(model_id)))
                    else:
                        _server_instance.model_manager.background_refresh_all()
                        self._send_json({"ok": True})
                elif action == "recommend":
                    query = payload.get("query", "")
                    if query:
                        self._send_json(_server_instance.model_manager.recommend_for_query(str(query)))
                    else:
                        self._send_json(_server_instance.model_manager.recommend_for_memory())
                elif action == "switch":
                    model_ref = payload.get("model_ref", "").strip()
                    if not model_ref:
                        self._send_json({"error": "model_ref required"}, 400)
                        return
                    if not _server_instance.loop or not _server_instance.executor:
                        self._send_json({"error": "server not ready"}, 503)
                        return
                    _server_instance.loop.run_in_executor(
                        _server_instance.executor, _server_instance.switch_main_model, model_ref
                    )
                    self._send_json({"ok": True, "message": f"Switching to {model_ref}"})
                elif action == "admit":
                    model_ref = payload.get("model_ref", "").strip()
                    if not model_ref:
                        self._send_json({"error": "model_ref required"}, 400)
                        return
                    self._send_json(_server_instance.admit_model(model_ref))
                elif action == "preload":
                    model_ids = payload.get("model_ids")
                    self._send_json({"ok": True, "preloaded": _server_instance.submit_preload_models(model_ids)})
                else:
                    self._send_json({"error": "unknown action"}, 400)
            except Exception as e:  # noqa: BLE001 - catch-all wrapper
                self._send_json({"error": str(e)}, 500)
            return

        if path == "/api/agents":
            try:
                payload = _read_body()
                action = payload.get("action", "")
                if _server_instance is None or not hasattr(_server_instance, "agent_task_manager"):
                    self._send_json({"error": "daemon not running"}, 503)
                    return
                if action == "list":
                    self._send_json({"tasks": _server_instance.list_agent_tasks()})
                elif action == "create":
                    goal = payload.get("goal", "").strip()
                    max_steps = max(1, min(50, int(payload.get("max_steps", 10))))
                    if not goal:
                        self._send_json({"error": "goal is required"}, 400)
                        return
                    task = _server_instance.submit_agent_task(goal, max_steps)
                    self._send_json({"ok": True, "task": task.to_dict()})
                elif action == "status":
                    task = _server_instance.get_agent_task(payload.get("task_id", ""))
                    self._send_json({"task": task})
                elif action == "cancel":
                    ok = _server_instance.cancel_agent_task(payload.get("task_id", ""))
                    self._send_json({"ok": ok})
                elif action == "pause":
                    ok = _server_instance.pause_agent_task(payload.get("task_id", ""))
                    self._send_json({"ok": ok})
                elif action == "resume":
                    ok = _server_instance.resume_agent_task(payload.get("task_id", ""))
                    self._send_json({"ok": ok})
                elif action == "delete":
                    ok = _server_instance.agent_task_manager.delete(payload.get("task_id", ""))
                    self._send_json({"ok": ok})
                else:
                    self._send_json({"error": "unknown action"}, 400)
            except Exception as e:  # noqa: BLE001 - catch-all wrapper
                self._send_json({"error": str(e)}, 500)
            return

        self._send_text("Not found", 404)


class ThreadedHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    """Allow concurrent dashboard requests and daemon-thread workers."""
    allow_reuse_address = True
    daemon_threads = True


class DashboardWebServer:
    """Start the local-only HTTP dashboard in a daemon thread."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8787):
        self.host = host
        self.port = port
        self._server: ThreadedHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self, mlx_server: Any) -> None:
        set_server_instance(mlx_server)
        self._server = ThreadedHTTPServer((self.host, self.port), DashboardHandler)
        self._thread = threading.Thread(target=self._server.serve_forever, name="badapple-dashboard", daemon=True)
        self._thread.start()
