const $ = (sel, el = document) => el.querySelector(sel);
const $$ = (sel, el = document) => Array.from(el.querySelectorAll(sel));

const ROUTES = {
  dashboard: 'Dashboard',
  chat: 'Chat',
  ocular: 'Ocular',
  persona: 'Persona',
  settings: 'Settings',
  logs: 'Logs',
};

let currentView = 'dashboard';

function initTheme() {
  const saved = localStorage.getItem('badapple-theme');
  const prefersLight = window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches;
  const light = saved ? saved === 'light' : prefersLight;
  document.body.classList.toggle('light', light);
  const checkbox = $('#theme-toggle');
  if (checkbox) checkbox.checked = light;
  updateThemeIcon();
  window.matchMedia('(prefers-color-scheme: light)').addEventListener('change', e => {
    if (!localStorage.getItem('badapple-theme')) {
      document.body.classList.toggle('light', e.matches);
      const cb = $('#theme-toggle');
      if (cb) cb.checked = e.matches;
      updateThemeIcon();
    }
  });
}
function toggleTheme(force) {
  const light = typeof force === 'boolean' ? force : !document.body.classList.contains('light');
  document.body.classList.toggle('light', light);
  localStorage.setItem('badapple-theme', light ? 'light' : 'dark');
  const checkbox = $('#theme-toggle');
  if (checkbox) checkbox.checked = light;
  updateThemeIcon();
}
function updateThemeIcon() {
  const btn = $('#theme-btn');
  if (btn) btn.textContent = document.body.classList.contains('light') ? '☀️' : '🌙';
}

function init() {
  initTheme();
  setupNav();
  setupKeyboard();
  window.addEventListener('popstate', route);
  if (location.pathname === '/splash') {
    runSplash();
  } else {
    route();
  }
  initStatusDot();
}

let daemonReady = false;

function showStartupOverlay() {
  const el = $('#startup-overlay');
  if (el) el.classList.remove('hidden');
}

function hideStartupOverlay() {
  const el = $('#startup-overlay');
  if (el) el.classList.add('hidden');
  if (!daemonReady) {
    daemonReady = true;
    maybeShowOnboarding();
  }
}

function updateStartupOverlay(status) {
  const title = $('#startup-title');
  const st = $('#startup-status');
  const fill = $('#startup-fill');
  if (!title || !st || !fill) return;

  const mode = status.runtime?.mode || 'STARTING';
  const mainModel = status.main_model_loaded === true || status.health?.checks?.main_model?.ok === true;
  const mainModelStatus = status.models?.main_9b?.status;
  const elapsed = Math.round(performance.now() / 1000);

  let msg = 'Waking up the local brain...';
  let pct = 20;

  if (mode === 'OFFLINE') {
    msg = 'Cannot reach the Bad Apple daemon. Is it running?';
    pct = 0;
  } else if (mainModel) {
    msg = 'Deep brain loaded and ready.';
    pct = 100;
  } else if (mode === 'READY' && mainModelStatus === 'downloading') {
    msg = 'Downloading the 9B model in the background...';
    pct = Math.round((status.models.main_9b.progress || 0) * 100);
  } else if (mode === 'READY' && mainModelStatus === 'queued') {
    msg = 'Preparing the 9B model download...';
    pct = 20;
  } else if (mode === 'READY') {
    msg = 'Lazy brain is ready. Fast tier active; 9B will load on first deep question.';
    pct = 100;
  } else if (elapsed > 15) {
    msg = 'Loading the 9B model... this can take ~45 seconds.';
    pct = 60;
  } else if (elapsed > 5) {
    msg = 'Loading the 9B model...';
    pct = 40;
  }

  title.textContent = (mode === 'READY' || mainModel) ? 'Bad Apple is ready' : 'Starting Bad Apple';
  st.textContent = msg;
  fill.style.width = pct + '%';
}

function setupKeyboard() {
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape') {
      closeHelp();
      const modal = document.getElementById('onboarding');
      if (modal) modal.style.display = 'none';
      return;
    }
    if (e.key === '?' && !e.metaKey && !e.ctrlKey && !e.altKey) {
      e.preventDefault();
      showHelp();
      return;
    }
    const meta = e.metaKey || e.ctrlKey;
    if (!meta) return;
    if (e.key === 'n') { e.preventDefault(); newChat(); }
    if (e.key === 'Enter') { e.preventDefault(); sendChat(); }
    if (e.key >= '1' && e.key <= '6') {
      e.preventDefault();
      const map = { '1': 'dashboard', '2': 'chat', '3': 'ocular', '4': 'persona', '5': 'settings', '6': 'logs' };
      navigate(map[e.key]);
    }
  });
}
function showHelp() { const h = $('#help'); if (h) h.style.display = 'flex'; }
function closeHelp() { const h = $('#help'); if (h) h.style.display = 'none'; }

const tourSteps = [
  { title: 'Dashboard', body: 'See runtime mode, memory, models, P2P, MCP status, tool calls, and live charts.', view: 'dashboard' },
  { title: 'Chat', body: 'Talk to Bad Apple. Conversations are saved locally, and you can copy code, retry, or delete messages.', view: 'chat' },
  { title: 'Ocular', body: 'Live screen capture and VLM description. Toggle the stream or capture a single frame.', view: 'ocular' },
  { title: 'Persona', body: 'Switch voices and edit the system prompt. Changes apply on the next query.', view: 'persona' },
  { title: 'Settings', body: 'Set workspace, add MCP servers, switch models, toggle autopilot/fast-tier/P2P, and change theme.', view: 'settings' },
  { title: 'Logs', body: 'Tail the daemon log for debugging and performance details.', view: 'logs' },
];
let tourIndex = 0;
function startTour() {
  dismissOnboarding();
  tourIndex = 0;
  showTourStep();
}
function showTourStep() {
  const step = tourSteps[tourIndex];
  const modal = document.getElementById('tour');
  $('#tour-title').textContent = step.title;
  $('#tour-body').textContent = step.body;
  $('#tour-prev').style.visibility = tourIndex === 0 ? 'hidden' : 'visible';
  $('#tour-next').textContent = tourIndex === tourSteps.length - 1 ? 'Finish' : 'Next';
  modal.style.display = 'flex';
  navigate(step.view);
}
function nextTour() {
  if (tourIndex < tourSteps.length - 1) { tourIndex++; showTourStep(); }
  else closeTour();
}
function prevTour() {
  if (tourIndex > 0) { tourIndex--; showTourStep(); }
}
function closeTour() { const t = $('#tour'); if (t) t.style.display = 'none'; }

let wizardStep = 0;
const wizardSteps = ['intro', 'workspace', 'features', 'models', 'done'];

function maybeShowOnboarding() {
  if (localStorage.getItem('badapple-onboarded')) return;
  const modal = document.getElementById('onboarding');
  if (!modal) return;
  wizardStep = 0;
  renderWizardStep();
  modal.style.display = 'flex';
}

function renderWizardStep() {
  $$('.wizard-step').forEach((el, i) => el.classList.toggle('hidden', i !== wizardStep));
}

function nextWizard() {
  if (wizardStep < wizardSteps.length - 1) {
    wizardStep++;
    renderWizardStep();
  }
}

function prevWizard() {
  if (wizardStep > 0) {
    wizardStep--;
    renderWizardStep();
  }
}

function skipWizard() {
  localStorage.setItem('badapple-onboarded', '1');
  const modal = document.getElementById('onboarding');
  if (modal) modal.style.display = 'none';
}

function dismissOnboarding() {
  localStorage.setItem('badapple-onboarded', '1');
  const modal = document.getElementById('onboarding');
  if (modal) modal.style.display = 'none';
}

async function finishWizard() {
  const ws = $('#setup-workspace').value.trim();
  const fast = $('#setup-fast-tier').checked;
  const autopilot = $('#setup-autopilot').checked;
  const p2p = $('#setup-p2p').checked;

  localStorage.setItem('badapple-setup', JSON.stringify({ workspace: ws, fastTier: fast, autopilot, p2p }));

  if (ws) {
    try {
      await api('/api/workspace', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path: ws }),
      });
    } catch (e) {
      console.error('workspace setup failed', e);
      toast('Workspace setup failed: ' + e.message, 'error');
    }
  }

  const commands = [];
  commands.push(fast ? 'fast tier on' : 'fast tier off');
  commands.push(autopilot ? 'autopilot on' : 'autopilot off');
  commands.push(p2p ? 'p2p on' : 'p2p off');
  for (const cmd of commands) {
    try {
      await api('/api/control', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ command: cmd }),
      });
    } catch (e) {
      console.error(`${cmd} failed`, e);
      toast(`${cmd} failed: ${e.message}`, 'error');
    }
  }

  wizardStep = wizardSteps.length - 1;
  renderWizardStep();
}

function setupNav() {
  $$('.nav-item').forEach(el => {
    el.addEventListener('click', e => {
      e.preventDefault();
      const view = el.dataset.view;
      navigate(view);
    });
  });
}

function navigate(view) {
  if (!ROUTES[view]) return;
  currentView = view;
  history.pushState({}, '', '/' + view);
  route();
}

function route() {
  const path = location.pathname.replace(/^\//, '') || 'dashboard';
  currentView = ROUTES[path] ? path : 'dashboard';
  $$('.view').forEach(v => v.classList.add('hidden'));
  const view = $('#view-' + currentView);
  if (view) view.classList.remove('hidden');
  $$('.nav-item').forEach(n => n.classList.toggle('active', n.dataset.view === currentView));
  $('h2#view-title').textContent = ROUTES[currentView];
  onViewEnter(currentView);
}

function onViewEnter(view) {
  if (view === 'dashboard') loadDashboard();
  if (view === 'chat') setupChat();
  if (view === 'ocular') loadOcular();
  if (view === 'persona') loadPersona();
  if (view === 'settings') loadSettings();
  if (view === 'logs') loadLogs();
}

/* ---------- UI helpers ---------- */
function toast(message, type = 'ok') {
  const t = document.createElement('div');
  t.className = `toast ${type}`;
  t.textContent = message;
  document.body.appendChild(t);
  requestAnimationFrame(() => t.classList.add('show'));
  setTimeout(() => { t.classList.remove('show'); setTimeout(() => t.remove(), 250); }, 3000);
}

async function api(path, opts = {}) {
  const r = await fetch(path, opts);
  if (!r.ok) {
    const txt = await r.text();
    throw new Error(txt || `HTTP ${r.status}`);
  }
  return r.json();
}

function formatBytes(bytes) {
  if (bytes == null) return '—';
  const gb = bytes / 1024 / 1024 / 1024;
  return `${gb.toFixed(2)} GB`;
}

function simpleMarkdown(text) {
  if (!text) return '';
  const codeBlocks = [];
  const saveCode = (code) => {
    codeBlocks.push(code);
    return `\x00CODE\x00${codeBlocks.length - 1}\x00`;
  };

  // Escape HTML
  let html = text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

  // Code blocks
  html = html.replace(/```(\w+)?\n([\s\S]*?)```/g, (m, lang, code) => {
    const unescaped = code.replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&amp;/g, '&');
    return saveCode(`<pre><code>${unescaped}</code></pre>`);
  });

  // Inline code
  html = html.replace(/`([^`]+)`/g, (m, code) => saveCode(`<code>${code}</code>`));

  // Bold/italic
  html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');

  // Blockquote
  html = html.replace(/^> (.+)$/gm, '<blockquote>$1</blockquote>');

  // Headers
  html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');
  html = html.replace(/^## (.+)$/gm, '<h2>$1</h2>');
  html = html.replace(/^# (.+)$/gm, '<h1>$1</h1>');

  // Lists
  html = html.replace(/^- (.+)$/gm, '<li>$1</li>');
  html = html.replace(/(<li>.+<\/li>\n?)+/g, '<ul>$&</ul>');
  html = html.replace(/<\/ul>\n?<ul>/g, '');

  // Paragraphs (only for lines that are not already tags)
  html = html.split('\n\n').map(p => {
    p = p.trim();
    if (!p) return '';
    if (/^<[a-zA-Z]/.test(p)) return p;
    return `<p>${p.replace(/\n/g, '<br>')}</p>`;
  }).join('\n');

  // Restore code placeholders
  codeBlocks.forEach((code, i) => {
    html = html.replace(`\x00CODE\x00${i}\x00`, code);
  });

  return html;
}

/* ---------- Status dot ---------- */
async function initStatusDot() {
  const dot = $('.sidebar-footer .status-dot');
  const text = $('.sidebar-footer .status-text');
  async function check() {
    try {
      const s = await api('/api/status');
      const ok = s.runtime && !s.runtime.killed && !s.runtime.safe_mode_reason;
      dot.className = 'status-dot ' + (ok ? 'ok' : 'warn');
      text.textContent = ok ? 'Daemon online' : (s.runtime?.safe_mode_reason || 'Check daemon');
    } catch (e) {
      dot.className = 'status-dot bad';
      text.textContent = 'Daemon offline';
    }
  }
  check();
  setInterval(check, 5000);
}

/* ---------- Dashboard ---------- */
let dashboardInterval;
let dashboardInitialLoad = true;
async function loadDashboard() {
  if (dashboardInterval) clearInterval(dashboardInterval);
  await updateDashboard();
  dashboardInterval = setInterval(updateDashboard, 2000);
}

async function updateDashboard() {
  const banner = $('#offline-banner');
  let status;
  try {
    status = await api('/api/status');
  } catch (e) {
    const msg = e?.message || String(e);
    console.error('status fetch failed:', msg, e);
    if (banner) banner.classList.remove('hidden');
    const statusText = $('.status-text');
    if (statusText) statusText.textContent = 'Offline — ' + msg;
    showStartupOverlay();
    updateStartupOverlay({ runtime: { mode: 'OFFLINE' } });
    return;
  }
  if (banner) banner.classList.add('hidden');

  const ready = status.runtime?.mode === 'READY';
  if (!ready) {
    showStartupOverlay();
    updateStartupOverlay(status);
    return;
  }
  hideStartupOverlay();

  if (dashboardInitialLoad) {
    setDashboardSkeletons(true);
    dashboardInitialLoad = false;
  }
  const endpoints = [
    { key: 'snap', path: '/api/snapshot' },
    { key: 'tail', path: '/api/tail?n=20' },
    { key: 'ledger', path: '/api/ledger?n=50' },
    { key: 'mcp', path: '/api/mcp_servers' },
    { key: 'voice', path: '/api/voice?n=12' },
  ];
  const settled = await Promise.allSettled(endpoints.map(e => api(e.path)));
  const data = {};
  const errors = {};
  settled.forEach((r, i) => {
    const key = endpoints[i].key;
    if (r.status === 'fulfilled') data[key] = r.value;
    else {
      console.error(`dashboard endpoint ${key} failed:`, r.reason);
      errors[key] = r.reason?.message || String(r.reason);
    }
  });
  setDashboardSkeletons(false);
  renderDashboard(status, data.snap || {}, data.tail || [], data.ledger || [], data.mcp || [], data.voice || [], errors);
}

function setDashboardSkeletons(loading) {
  const cards = $('#dashboard-cards');
  if (!cards) return;
  if (loading) {
    cards.innerHTML = Array.from({ length: 8 }, () => `
      <div class="card skeleton">
        <div class="skeleton-title"></div>
        <div class="skeleton-value"></div>
        <div class="skeleton-sub"></div>
      </div>
    `).join('');
    const panels = [
      { id: 'p2p-peers', lines: 2 },
      { id: 'mcp-status', lines: 2 },
      { id: 'tool-calls', lines: 4 },
      { id: 'voice-activity', lines: 4 },
    ];
    panels.forEach(p => {
      const el = $('#' + p.id);
      if (!el) return;
      el.innerHTML = Array.from({ length: p.lines }, () => `<div class="skeleton-line"></div>`).join('');
    });
    const perf = $('#latest-perf');
    if (perf) perf.innerHTML = '<span class="skeleton-line short"></span>';
    const log = $('#log-tail');
    if (log) log.innerHTML = '<span class="skeleton-line"></span><span class="skeleton-line"></span><span class="skeleton-line"></span>';
  }
}

function emptyState(title, body, action) {
  const btn = action ? `<button class="secondary" onclick="navigate('${action.view}')">${action.text}</button>` : '';
  return `<div class="empty-state"><div class="empty-title">${title}</div><div class="empty-body">${body}</div>${btn}</div>`;
}

function errorState(title, detail) {
  const short = escapeHtml(String(detail || ''));
  return `<div class="error-state"><div class="error-title">${title}</div><div class="error-body">${short}</div><button class="secondary" onclick="loadDashboard()">Retry</button></div>`;
}

function renderDashboard(status, snap, tail, ledger, mcp, voice, errors = {}) {
  const rt = status.runtime || {};
  const flags = [];
  if (rt.killed) flags.push('killed');
  if (rt.private_mode) flags.push('private');
  if (rt.safe_mode_reason) flags.push(`safe: ${rt.safe_mode_reason}`);
  if (status.autopilot) flags.push('autopilot');
  if (status.fast_tier) flags.push('fast tier');
  if (status.p2p_enabled) flags.push('p2p on');

  const memUsed = snap.memory ? (snap.memory.used_gb ?? 0) : 0;
  const memTotal = snap.memory ? (snap.memory.total_gb ?? 0) : 0;
  const memPct = snap.memory ? (snap.memory.percent ?? 0) : 0;

  const rows = [
    { label: 'Runtime', value: rt.mode || 'unknown', sub: flags.join(' · ') || 'normal' },
    { label: 'Memory', value: `${memUsed.toFixed(2)} / ${memTotal.toFixed(2)} GB`, sub: `${memPct}% used` },
    { label: 'Active models', value: (status.active_models || []).join(', ') || 'none', sub: `${(status.active_models || []).length} loaded` },
    { label: 'Battery', value: snap.battery ? `${snap.battery.percent}%` : '—', sub: snap.battery?.source === 'ac' ? 'AC power' : 'on battery' },
    { label: 'Workspace', value: status.workspace ? status.workspace.replace(/^\//, '').split('/').pop() : 'No workspace set', sub: status.workspace || '' },
    { label: 'P2P Mesh', value: status.p2p_enabled ? 'on' : 'off', sub: status.p2p_peers || 'No peers on the local network.' },
    { label: 'Ambient', value: (status.ambient_running && status.ambient?.app) ? status.ambient.app : 'off', sub: (status.ambient_running && status.ambient?.window) ? status.ambient.window : '' },
    { label: 'Hibernation', value: status.hibernating ? 'asleep' : 'awake', sub: status.hibernating ? `idle for ${Math.round(status.idle_seconds || 0)}s` : `idle ${Math.round(status.idle_seconds || 0)}s / ${Math.round(status.hibernate_after || 300)}s` },
  ];

  const cards = $('#dashboard-cards');
  cards.innerHTML = rows.map(r => `
    <div class="card">
      <h3>${r.label}</h3>
      <div class="value">${r.value}</div>
      <div class="sub">${r.sub}</div>
    </div>
  `).join('');

  if (status.active_persona) {
    $('#top-persona').textContent = status.active_persona;
  }

  const peers = status.p2p_peers || [];
  const peersEl = $('#p2p-peers');
  if (peers.length) {
    peersEl.innerHTML = peers.map(p => `<span class="persona-pill">${p}</span>`).join(' ');
  } else if (!status.p2p_enabled) {
    peersEl.innerHTML = emptyState('P2P is off', 'Enable P2P sync in Settings to discover peers on your local network.', { text: 'Open Settings', view: 'settings' });
  } else {
    peersEl.innerHTML = emptyState('No peers found', 'Bad Apple is listening on the local network. Peers will appear here when they join.');
  }

  const perf = snap.latest_log_perf?.raw || '—';
  $('#latest-perf').textContent = perf;

  const logEl = $('#log-tail');
  if (errors.tail) {
    logEl.innerHTML = errorState('Could not load log tail', errors.tail);
  } else {
    logEl.textContent = (tail.lines || []).join('') || 'Log is empty — the daemon may still be starting.';
  }

  const mcpList = mcp?.servers || [];
  const mcpEl = $('#mcp-status');
  if (errors.mcp) {
    mcpEl.innerHTML = errorState('Could not load MCP servers', errors.mcp);
  } else if (mcpList.length) {
    mcpEl.innerHTML = mcpList.map(s => `<span class="persona-pill" title="${s.command.join(' ')}">${s.name}</span>`).join(' ');
  } else {
    mcpEl.innerHTML = emptyState('No MCP servers', 'Add local MCP servers in Settings to expand Bad Apple’s tools.', { text: 'Add server', view: 'settings' });
  }

  const toolEvents = (ledger?.entries || []).filter(e => e.event_type === 'tool' || e.type === 'tool').slice(0, 12);
  const toolEl = $('#tool-calls');
  if (errors.ledger) {
    toolEl.innerHTML = errorState('Could not load tool calls', errors.ledger);
  } else if (toolEvents.length) {
    toolEl.innerHTML = toolEvents.map(e => `<div class="tool-call"><span class="name">⚡ ${e.tool || e.name || 'tool'}</span> <span class="muted">${e.timestamp || ''}</span></div>`).join('');
  } else {
    toolEl.innerHTML = emptyState('No recent tool calls', 'Ask Bad Apple to do something on your Mac, like “list my Downloads” or “run a benchmark”.');
  }

  const voiceEvents = voice?.events || [];
  const voiceEl = $('#voice-activity');
  if (errors.voice) {
    voiceEl.innerHTML = errorState('Could not load voice activity', errors.voice);
  } else if (voiceEvents.length) {
    voiceEl.innerHTML = voiceEvents.map(e => {
        const icon = { transcript: '🎤', command: '▶', response: '💬', spoken: '🔊', error: '⚠' }[e.type] || '•';
        const cls = e.type === 'error' ? 'bad' : 'muted';
        const text = e.text.length > 120 ? e.text.slice(0, 120) + '…' : e.text;
        return `<div class="tool-call"><span class="name">${icon} ${e.type}</span> <span class="${cls}">${text}</span></div>`;
      }).join('');
  } else {
    voiceEl.innerHTML = emptyState('No voice activity yet', 'Voice events appear here when voice mode is enabled.');
  }

  updateMetricsChart(snap, status);
}

/* ---------- Metrics chart ---------- */
const metricsHistory = { labels: [], memory: [], tps: [] };
const maxPoints = 60;
function parseTPS(perf) {
  if (!perf || !perf.raw) return 0;
  const m = perf.raw.match(/decode t\/s[:\s]+([\d.]+)/);
  return m ? parseFloat(m[1]) : 0;
}
function updateMetricsChart(snap, status) {
  const canvas = $('#metrics-chart');
  if (!canvas) return;
  const now = new Date().toLocaleTimeString();
  const mem = snap.memory ? (snap.memory.percent ?? 0) : 0;
  const tps = parseTPS(snap.latest_log_perf);
  metricsHistory.labels.push(now);
  metricsHistory.memory.push(mem);
  metricsHistory.tps.push(tps);
  if (metricsHistory.labels.length > maxPoints) {
    metricsHistory.labels.shift();
    metricsHistory.memory.shift();
    metricsHistory.tps.shift();
  }
  drawChart(canvas, metricsHistory);
}
function drawChart(canvas, data) {
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  ctx.scale(dpr, dpr);
  const w = rect.width, h = rect.height;
  const pad = 24;
  const chartW = w - pad * 2;
  const chartH = h - pad * 2;
  ctx.clearRect(0, 0, w, h);

  // grid
  ctx.strokeStyle = getComputedStyle(document.body).getPropertyValue('--border-subtle').trim() || '#1f1f22';
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let i = 0; i <= 4; i++) {
    const y = pad + chartH / 4 * i;
    ctx.moveTo(pad, y);
    ctx.lineTo(w - pad, y);
  }
  ctx.stroke();

  const maxMem = Math.max(10, ...data.memory);
  const maxTps = Math.max(1, ...data.tps);

  function drawLine(values, color, max) {
    if (values.length < 2) return;
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    values.forEach((v, i) => {
      const x = pad + (chartW / (maxPoints - 1)) * i;
      const y = pad + chartH - (v / max) * chartH;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }

  drawLine(data.memory, '#4ade80', maxMem);
  drawLine(data.tps, '#3b82f6', maxTps);

  // legend
  ctx.fillStyle = '#4ade80'; ctx.fillRect(pad, 8, 10, 10);
  ctx.fillStyle = '#fff'; ctx.fillText('Mem %', pad + 14, 16);
  ctx.fillStyle = '#3b82f6'; ctx.fillRect(pad + 70, 8, 10, 10);
  ctx.fillStyle = '#fff'; ctx.fillText('Decode t/s', pad + 84, 16);
}

/* ---------- Chat / conversation ---------- */
let chatReady = false;
let chatHistory = [];
let currentChatId = null;
let conversations = {};

function generateId() { return Math.random().toString(36).slice(2) + Date.now().toString(36); }
function storageKey() { return 'badapple-chats'; }
function loadConversations() {
  try { return JSON.parse(localStorage.getItem(storageKey()) || '{}'); }
  catch (e) { return {}; }
}
function saveConversations() {
  try { localStorage.setItem(storageKey(), JSON.stringify(conversations)); }
  catch (e) { console.warn('Could not save conversations', e); }
}
function chatTitle(messages) {
  const first = messages.find(m => m.role === 'user');
  if (!first) return 'New chat';
  return first.text.slice(0, 40) + (first.text.length > 40 ? '…' : '');
}
function saveCurrentChat() {
  if (!currentChatId) currentChatId = generateId();
  if (chatHistory.length === 0) return;
  conversations[currentChatId] = {
    id: currentChatId,
    title: chatTitle(chatHistory),
    updated: Date.now(),
    messages: chatHistory,
  };
  saveConversations();
  updateHistorySelect();
}
function deleteCurrentChat() {
  if (currentChatId && conversations[currentChatId]) {
    delete conversations[currentChatId];
    saveConversations();
  }
  newChat();
}
function newChat() {
  currentChatId = generateId();
  chatHistory = [];
  renderChat();
  updateHistorySelect();
}
function loadChatHistory(id) {
  if (!id || id === currentChatId) return;
  if (!conversations[id]) return;
  currentChatId = id;
  chatHistory = JSON.parse(JSON.stringify(conversations[id].messages));
  renderChat();
  updateHistorySelect();
}
function updateHistorySelect() {
  const sel = $('#chat-history-select');
  if (!sel) return;
  const sorted = Object.values(conversations).sort((a, b) => b.updated - a.updated);
  const options = ['<option value="">Current chat</option>', ...sorted.map(c =>
    `<option value="${c.id}" ${c.id === currentChatId ? 'selected' : ''}>${escapeHtml(c.title)}</option>`
  )];
  sel.innerHTML = options.join('');
}
function useSuggestion(text) {
  const input = $('#chat-input');
  input.value = text;
  input.focus();
}

function setupChat() {
  if (chatReady) return;
  chatReady = true;
  conversations = loadConversations();
  const input = $('#chat-input');
  const sendBtn = $('#chat-send');
  input.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey && !e.metaKey && !e.ctrlKey) { e.preventDefault(); sendChat(); }
  });
  input.addEventListener('input', () => {
    input.style.height = 'auto';
    input.style.height = Math.min(input.scrollHeight, 180) + 'px';
  });
  document.body.addEventListener('dragover', e => { e.preventDefault(); });
  document.body.addEventListener('drop', handleFileDrop);
  if (!currentChatId) newChat();
  else renderChat();
}

function handleFileDrop(e) {
  if (!e.target.closest('#view-chat')) return;
  e.preventDefault();
  const file = e.dataTransfer.files[0];
  if (!file) return;
  if (file.type.startsWith('image/')) {
    const reader = new FileReader();
    reader.onload = ev => {
      const dataUrl = ev.target.result;
      addMessage('user', '[image] ' + file.name, { image: dataUrl, local: true });
      sendChat(`Describe this image: ${file.name}`, true);
    };
    reader.readAsDataURL(file);
  } else if (file.size < 100000) {
    const reader = new FileReader();
    reader.onload = ev => {
      const text = ev.target.result;
      addMessage('user', '[file] ' + file.name, { fileText: text });
      sendChat(`Summarize this file (${file.name}):\n\n${text}`, true);
    };
    reader.readAsText(file);
  } else {
    toast('File too large. Try a text or image file under ~100 KB.', 'error');
  }
}

function addMessage(role, text, extra = {}) {
  const msg = { id: generateId(), role, text, ...extra };
  chatHistory.push(msg);
  const el = renderMessage(msg);
  $('#chat-messages').appendChild(el);
  scrollChat();
  return msg;
}

function renderMessage(msg) {
  const chatEl = $('#chat-messages');
  const d = document.createElement('div');
  d.className = `message ${msg.role}`;
  d.dataset.id = msg.id;
  const isUser = msg.role === 'user';
  const html = isUser ? escapeHtml(msg.text) : simpleMarkdown(msg.text || '');
  d.innerHTML = `
    <div class="avatar">${isUser ? 'You' : 'BA'}</div>
    <div class="bubble">
      <div class="bubble-content">${html}</div>
      ${msg.error ? `<div class="msg-error">Error: ${escapeHtml(msg.error)}</div>` : ''}
      <div class="msg-actions">
        ${isUser ? `<button onclick="editMessage('${msg.id}')">Edit</button>` : `<button onclick="copyMessage('${msg.id}')">Copy</button><button onclick="retryMessage('${msg.id}')">Retry</button>`}
        <button onclick="deleteMessage('${msg.id}')">Delete</button>
      </div>
    </div>
  `;
  const bubble = $('.bubble', d);
  if (msg.image) {
    const img = document.createElement('img');
    img.src = msg.image;
    img.alt = '';
    bubble.appendChild(img);
  }
  if (msg.tool) {
    const tc = document.createElement('div');
    tc.className = 'tool-call';
    tc.innerHTML = `<span class="name">⚡ ${msg.tool}</span>`;
    bubble.appendChild(tc);
  }
  if (msg.metrics) {
    const m = document.createElement('div');
    m.className = 'metrics';
    m.textContent = JSON.stringify(msg.metrics);
    bubble.appendChild(m);
  }
  attachCopyButtons(bubble);
  chatEl.appendChild(d);
  return d;
}

function attachCopyButtons(bubble) {
  $$('pre', bubble).forEach(pre => {
    if (pre.querySelector('.copy-btn')) return;
    const btn = document.createElement('button');
    btn.className = 'copy-btn';
    btn.textContent = 'Copy';
    btn.onclick = () => navigator.clipboard.writeText(pre.textContent).then(() => toast('Copied'));
    pre.style.position = 'relative';
    pre.appendChild(btn);
  });
}

function renderChat() {
  const chatEl = $('#chat-messages');
  chatEl.innerHTML = '';
  if (chatHistory.length === 0) {
    addMessage('bot', "Hey, babe. I'm here. Ask me anything or tell me what to do on your Mac.");
    return;
  }
  chatHistory.forEach(msg => chatEl.appendChild(renderMessage(msg)));
  scrollChat();
}

function escapeHtml(t) {
  return t.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function updateMessage(id, updates) {
  const msg = chatHistory.find(m => m.id === id);
  if (!msg) return;
  Object.assign(msg, updates);
  const el = document.querySelector(`.message[data-id="${id}"]`);
  if (el) {
    el.replaceWith(renderMessage(msg));
  }
  saveCurrentChat();
}

function deleteMessage(id) {
  const idx = chatHistory.findIndex(m => m.id === id);
  if (idx < 0) return;
  chatHistory.splice(idx, 1);
  const el = document.querySelector(`.message[data-id="${id}"]`);
  if (el) el.remove();
  saveCurrentChat();
}

function editMessage(id) {
  const msg = chatHistory.find(m => m.id === id);
  if (!msg) return;
  const newText = prompt('Edit message:', msg.text);
  if (newText === null) return;
  msg.text = newText.trim();
  renderChat();
  saveCurrentChat();
  // Remove all messages after this one and re-send
  const idx = chatHistory.findIndex(m => m.id === id);
  chatHistory.splice(idx + 1);
  sendChat(msg.text, false);
}

function copyMessage(id) {
  const msg = chatHistory.find(m => m.id === id);
  if (!msg) return;
  navigator.clipboard.writeText(msg.text).then(() => toast('Copied to clipboard'));
}

async function retryMessage(id) {
  const idx = chatHistory.findIndex(m => m.id === id);
  if (idx <= 0) return;
  const promptMsg = chatHistory[idx - 1];
  if (promptMsg.role !== 'user') return;
  // Remove the bot response and resend
  chatHistory.splice(idx);
  const el = document.querySelectorAll('.message');
  if (el[idx]) el[idx].remove();
  await sendChat(promptMsg.text, false);
}

async function sendChat(textOverride, isSystem) {
  const input = $('#chat-input');
  const sendBtn = $('#chat-send');
  let p;
  if (textOverride !== undefined && isSystem) {
    p = textOverride;
  } else if (textOverride !== undefined) {
    p = textOverride;
    input.value = '';
  } else {
    p = input.value.trim();
    if (!p) return;
    input.value = '';
  }
  input.style.height = 'auto';

  if (!isSystem) {
    // Check if this is a new continuation or a new message
    const lastUser = chatHistory.length && chatHistory[chatHistory.length - 1].role === 'user' && chatHistory[chatHistory.length - 1].text === p;
    if (!lastUser) addMessage('user', p);
  }
  sendBtn.disabled = true;

  const botMsg = { id: generateId(), role: 'bot', text: '', typing: true };
  chatHistory.push(botMsg);
  const el = renderMessage(botMsg);
  const bubble = $('.bubble', el);
  const content = $('.bubble-content', bubble);
  content.innerHTML = '<div class="typing"><span></span><span></span><span></span></div>';

  let finalText = '';
  try {
    const r = await fetch(window.location.origin + '/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt: p, stream: true }),
    });
    if (!r.ok) {
      const err = await r.text();
      throw new Error(err || `HTTP ${r.status}`);
    }
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
        if (!data) continue;
        let msg;
        try { msg = JSON.parse(data); } catch (e) { continue; }

        if (!started) { content.innerHTML = '<span class="stream"></span>'; started = true; }
        const streamSpan = $('.stream', content);

        if (msg.type === 'token') {
          if (streamSpan) streamSpan.textContent += msg.text;
          finalText += msg.text;
          scrollChat();
        } else if (msg.type === 'tool') {
          botMsg.tool = msg.tool;
          const tc = document.createElement('div');
          tc.className = 'tool-call';
          tc.innerHTML = `<span class="name">⚡ ${msg.tool}</span>`;
          bubble.appendChild(tc);
          scrollChat();
        } else if (msg.type === 'done') {
          const imgMatch = (msg.text || '').match(/^Generated image:\s*(.+\.png)$/);
          if (imgMatch) {
            botMsg.text = 'Generated image:';
            botMsg.image = '/api/image/' + encodeURIComponent(imgMatch[1].split('/').pop());
          } else {
            finalText = msg.text || finalText;
            botMsg.text = finalText;
          }
          botMsg.metrics = msg.metrics;
          botMsg.typing = false;
          if (imgMatch) {
            content.innerHTML = '';
            const t = document.createElement('div');
            t.textContent = botMsg.text;
            content.appendChild(t);
            const img = document.createElement('img');
            img.src = botMsg.image;
            img.alt = 'generated image';
            content.appendChild(img);
          } else {
            content.innerHTML = simpleMarkdown(botMsg.text);
            attachCopyButtons(bubble);
          }
          if (msg.metrics) {
            const m = document.createElement('div');
            m.className = 'metrics';
            m.textContent = JSON.stringify(msg.metrics);
            bubble.appendChild(m);
          }
          scrollChat();
        } else if (msg.type === 'error') {
          botMsg.error = msg.error;
          botMsg.typing = false;
          content.innerHTML = '';
          const err = document.createElement('div');
          err.className = 'msg-error';
          err.textContent = 'Error: ' + msg.error;
          content.appendChild(err);
          scrollChat();
        }
      }
    }
  } catch (e) {
    botMsg.error = e.message;
    botMsg.typing = false;
    const err = document.createElement('div');
    err.className = 'msg-error';
    err.textContent = 'Error: ' + e.message;
    if (content) content.appendChild(err);
    toast('Send failed: ' + e.message, 'error');
  } finally {
    sendBtn.disabled = false;
    input.focus();
    delete botMsg.typing;
    saveCurrentChat();
  }
}

function scrollChat() {
  const chatEl = $('#chat-messages');
  chatEl.scrollTop = chatEl.scrollHeight;
}

/* ---------- Persona ---------- */
async function loadPersona() {
  const sel = $('#persona-select');
  const ta = $('#persona-prompt');
  const data = await api('/api/personas');
  sel.innerHTML = data.personas.map(n => `<option value="${n}" ${n === data.active ? 'selected' : ''}>${n}</option>`).join('');
  ta.value = data.prompt || '';
}

async function savePersona() {
  const sel = $('#persona-select');
  const ta = $('#persona-prompt');
  try {
    await api('/api/personas', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: sel.value, prompt: ta.value }),
    });
    toast('Persona saved. Active next query.');
    loadPersona();
  } catch (e) {
    toast('Error: ' + e.message, 'error');
  }
}

/* ---------- Settings ---------- */
async function loadSettings() {
  const status = await api('/api/status');
  $('#ws-path').value = status.workspace || '';
  $('#auto-pilot').checked = !!status.autopilot;
  $('#fast-tier').checked = !!status.fast_tier;
  $('#p2p-enabled').checked = !!status.p2p_enabled;
  renderModels(status.active_models || []);
  await loadMcpServers();
  await loadMcpRegistry();
  await loadModelList(status.active_models?.[0]);
}

function renderModels(models) {
  const el = $('#model-list');
  el.innerHTML = models.length
    ? models.map(m => `<div class="persona-pill">${m}</div>`).join(' ')
    : '<span class="text-tertiary">No active models</span>';
}

async function loadModelList(active) {
  const sel = $('#model-select');
  try {
    const r = await api('/api/control', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ command: 'list models' }),
    });
    const lines = (r.result || '').split('\n');
    const models = lines.map(l => l.trim()).filter(l => l && !l.startsWith('Local') && !l.startsWith('Available'));
    sel.innerHTML = models.map(m => {
      const id = m.split(' —')[0].trim();
      return `<option value="${id}" ${id === active ? 'selected' : ''}>${m}</option>`;
    }).join('');
  } catch (e) {
    sel.innerHTML = '<option>Error loading models</option>';
  }
}

async function switchModel() {
  const sel = $('#model-select');
  try {
    const r = await api('/api/control', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ command: `use model ${sel.value}` }),
    });
    toast(r.result || 'Model switched');
  } catch (e) {
    toast('Error: ' + e.message, 'error');
  }
}

async function invokeMcpTool() {
  const server = $('#mcp-invoke-server').value.trim();
  const tool = $('#mcp-invoke-tool').value.trim();
  const arg = $('#mcp-invoke-arg').value.trim();
  const out = $('#mcp-invoke-result');
  if (!server || !tool) { out.textContent = 'Server and tool are required'; return; }
  const command = arg ? `invoke mcp tool ${tool} on server ${server} with text ${arg}` : `invoke mcp tool ${tool} on server ${server}`;
  try {
    const r = await api('/api/control', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ command }),
    });
    out.textContent = r.result || 'No output';
  } catch (e) {
    out.textContent = 'Error: ' + e.message;
  }
}

async function loadMcpServers() {
  try {
    const data = await api('/api/mcp_servers');
    const tbody = $('#mcp-table tbody');
    tbody.innerHTML = (data.servers || []).map(s => `
      <tr>
        <td>${s.name}</td>
        <td>${s.command.join(' ')}</td>
        <td><button class="secondary" onclick="removeMcpServer('${s.name}')">Remove</button></td>
      </tr>
    `).join('');
  } catch (e) {
    $('#mcp-table tbody').innerHTML = `<tr><td colspan="3" class="text-tertiary">${e.message}</td></tr>`;
  }
}

async function loadMcpRegistry() {
  try {
    const data = await api('/api/mcp_registry');
    const installed = (await api('/api/mcp_servers').catch(() => ({ servers: [] }))).servers || [];
    const names = new Set(installed.map(s => s.name));
    const tbody = $('#mcp-registry-table tbody');
    const servers = data.servers || [];
    if (!servers.length) {
      tbody.innerHTML = '<tr><td colspan="5" class="text-tertiary">No registry entries yet.</td></tr>';
      return;
    }
    tbody.innerHTML = servers.map(s => `
      <tr>
        <td>${s.name}</td>
        <td>${s.publisher || '-'}</td>
        <td>${s.description || ''}</td>
        <td>${s.install_type || 'command'}</td>
        <td>
          ${names.has(s.name)
            ? '<span class="text-tertiary">Installed</span>'
            : `<button class="secondary" onclick="installMcpRegistry('${s.name}')">Install</button>`}
        </td>
      </tr>
    `).join('');
  } catch (e) {
    $('#mcp-registry-table tbody').innerHTML = `<tr><td colspan="5" class="text-tertiary">${e.message}</td></tr>`;
  }
}

async function installMcpRegistry(name) {
  try {
    const res = await api('/api/mcp_servers', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action: 'install', name }),
    });
    toast(res.ok ? `Installed ${name}` : res);
    loadMcpServers();
    loadMcpRegistry();
  } catch (e) {
    toast('Error: ' + e.message, 'error');
  }
}

async function addMcpServer() {
  const name = $('#mcp-name').value.trim();
  const cmd = $('#mcp-command').value.trim();
  if (!name || !cmd) return toast('Name and command required', 'error');
  try {
    await api('/api/mcp_servers', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action: 'add', name, command: cmd }),
    });
    toast('MCP server added');
    $('#mcp-name').value = '';
    $('#mcp-command').value = '';
    loadMcpServers();
  } catch (e) {
    toast('Error: ' + e.message, 'error');
  }
}

async function removeMcpServer(name) {
  if (!confirm(`Remove ${name}?`)) return;
  try {
    await api('/api/mcp_servers', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action: 'remove', name }),
    });
    toast('MCP server removed');
    loadMcpServers();
  } catch (e) {
    toast('Error: ' + e.message, 'error');
  }
}

async function toggleControl(name, command, checkbox) {
  try {
    const r = await api('/api/control', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ command }),
    });
    toast(`${name}: ${r.result || 'ok'}`);
  } catch (e) {
    toast('Error: ' + e.message, 'error');
    if (checkbox) checkbox.checked = !checkbox.checked;
  }
}

async function saveWorkspace() {
  const path = $('#ws-path').value.trim();
  try {
    await api('/api/workspace', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ path }),
    });
    toast('Workspace updated');
  } catch (e) {
    toast('Error: ' + e.message, 'error');
  }
}

/* ---------- Logs ---------- */
let logsInterval;
async function loadLogs() {
  if (logsInterval) clearInterval(logsInterval);
  await updateLogs();
  logsInterval = setInterval(updateLogs, 3000);
}

async function updateLogs() {
  try {
    const tail = await api('/api/tail?n=50');
    $('#log-view').textContent = (tail.lines || []).join('') || '—';
  } catch (e) {
    $('#log-view').textContent = 'Error loading log';
  }
}

/* ---------- Splash ---------- */
function runSplash() {
  const splash = document.getElementById('splash');
  const bar = $('.splash-bar .fill', splash);
  const text = $('#splash-status', splash);
  let attempts = 0;

  const poll = async () => {
    attempts++;
    try {
      const status = await api('/api/status');
      const ready = status.runtime?.mode === 'READY';
      if (ready) {
        text.textContent = 'Ready.';
        bar.style.width = '100%';
        setTimeout(() => location.href = '/', 500);
        return;
      }
      const mainModel = status.health?.checks?.main_model?.ok === true;
      const elapsed = attempts * 0.5;
      let msg = 'Waking up the local brain...';
      let pct = Math.min(95, Math.round(elapsed * 2));
      if (elapsed > 30) {
        msg = 'Loading the 9B model... this can take ~45 seconds on first launch.';
        pct = 70;
      } else if (elapsed > 10) {
        msg = 'Loading the 9B model and embedding model...';
        pct = 50;
      } else if (mainModel) {
        msg = 'Warming up...';
        pct = 85;
      }
      text.textContent = msg;
      bar.style.width = pct + '%';
    } catch (e) {
      text.textContent = 'Waiting for daemon...';
      bar.style.width = Math.min(95, attempts * 2) + '%';
    }
    setTimeout(poll, 500);
  };

  poll();
}

/* ---------- Ocular ---------- */
let ocularInterval = null;
let ocularRunning = false;

function _updateOcularStatus(data) {
  const statusEl = $('#ocular-status');
  const img = $('#ocular-image');
  const desc = $('#ocular-description');
  const btn = $('#ocular-toggle');
  if (!statusEl) return;

  ocularRunning = data.running;
  if (data.running) {
    statusEl.textContent = `Streaming: capture every ${data.context?.capture_interval || data.capture_interval}s, describe every ${data.context?.describe_interval || data.describe_interval}s.`;
    btn.textContent = 'Stop stream';
    btn.classList.remove('secondary');
  } else {
    statusEl.textContent = 'Stream stopped. Capture a frame or start the stream.';
    btn.textContent = 'Start stream';
    btn.classList.add('secondary');
  }

  if (data.context?.description) {
    desc.textContent = data.context.description;
  } else if (data.context?.error) {
    desc.textContent = 'Error: ' + data.context.error;
  }

  if (img) {
    img.style.display = 'block';
    img.src = '/api/ocular/screen.png?t=' + Date.now();
  }
}

async function loadOcular() {
  try {
    const data = await api('/api/ocular');
    _updateOcularStatus(data);
    if (data.context) {
      $('#ocular-capture-interval').value = data.context.capture_interval || 5;
      $('#ocular-describe-interval').value = data.context.describe_interval || 0;
    }
  } catch (e) {
    $('#ocular-status').textContent = 'Error: ' + e.message;
  }
  if (ocularInterval) clearInterval(ocularInterval);
  ocularInterval = setInterval(() => {
    if (currentView === 'ocular') loadOcular();
  }, 5000);
}

async function toggleOcular() {
  const btn = $('#ocular-toggle');
  btn.disabled = true;
  try {
    const action = ocularRunning ? 'stop' : 'start';
    const payload = { action };
    if (action === 'start') {
      payload.capture_interval = parseFloat($('#ocular-capture-interval').value) || 5;
      payload.describe_interval = parseFloat($('#ocular-describe-interval').value) || 0;
      payload.prompt = $('#ocular-prompt').value;
    }
    const res = await api('/api/ocular', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    toast(res.result || res);
    await loadOcular();
  } catch (e) {
    toast('Error: ' + e.message, 'error');
  } finally {
    btn.disabled = false;
  }
}

async function captureOcular() {
  try {
    const res = await api('/api/ocular', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action: 'capture', prompt: $('#ocular-prompt').value }),
    });
    toast(res.result || res);
    await loadOcular();
  } catch (e) {
    toast('Error: ' + e.message, 'error');
  }
}

document.addEventListener('DOMContentLoaded', init);
