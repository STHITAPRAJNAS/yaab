"""`yaab web` — a zero-build local dev console for an agent.

Serves a single self-contained HTML page (no bundler, no npm, no CDN) that is a
real dev console rather than a bare chat box. It mounts the agent's full API
(``/run``, ``/run/stream``, ``/runs``, the agent card — see :mod:`yaab.serve`)
and layers tabs over it:

* **Chat** — token streaming over ``/chat/stream`` (the original behavior).
* **Events** — a live event-stream inspector: sends over ``POST /run/stream``
  (SSE) and renders every typed event (run_start, text_delta, tool_call,
  tool_result, agent_transfer, final_output, run_end) as a colour-coded,
  expandable timeline with payload JSON, plus a run-summary header showing the
  run's total tokens, cost, and latency from the ``run_end`` payload.
* **Runs** — lists runs from ``GET /runs`` with status badges, a per-run Cancel
  button (``POST /runs/{id}/cancel``), plus per-run Trace and Replay actions,
  auto-refreshing while the tab is open.
* **Trace** — a per-step span/waterfall from ``GET /runs/{id}/trace``: typed
  spans (model call, tool call, transfer, approval) with per-span latency and
  token/cost badges, and run totals. Needs a configured trace store.
* **State** — a session-state inspector from ``GET /sessions/{id}/state``.
* **Approvals** — out-of-band human sign-off: lists pending requests from
  ``GET /approvals`` with Approve/Deny buttons. Needs a configured approval store.
* **Agent** — renders ``GET /.well-known/agent.json`` (the agent card) plus the
  agent's tool list and instructions from ``GET /agent/info`` (mounted here on
  the web app, not on the serve app).

Tabs whose backing store is not configured degrade gracefully: the matching
endpoints return a clean ``404`` and the tab shows a "configure a store" hint.

Everything is vanilla JS — ``fetch`` with a ``ReadableStream`` reader for the
POST-SSE event stream (``EventSource`` can't POST) — and dark-mode friendly via
``color-scheme``. The Chat tab is plain enough to work even if the fancier tabs
can't (graceful degradation).

    from yaab.web import web_app
    # uvicorn module:app   where app = web_app(agent)

or via the CLI: ``yaab web mymodule:agent``.
"""

from __future__ import annotations

from typing import Any

# The page is a Python format-free template: we substitute only ``{name}`` via a
# literal ``str.replace`` so the page's own ``{ ... }`` CSS/JS braces are left
# untouched (no ``str.format`` — it would choke on every CSS rule).
_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>YAAB · {name}</title>
<style>
  :root { color-scheme: light dark; --line: #8884; --accent: #4f8cff; }
  * { box-sizing: border-box; }
  body { font-family: ui-sans-serif, system-ui, sans-serif; max-width: 920px;
         margin: 0 auto; padding: 1rem; }
  h1 { font-size: 1.1rem; opacity: .8; margin: .2rem 0 .8rem; }
  h1 .sub { opacity: .5; font-weight: 400; }
  nav { display: flex; gap: .25rem; border-bottom: 1px solid var(--line); margin-bottom: .8rem; }
  nav button { background: transparent; border: 0; border-bottom: 2px solid transparent;
               padding: .5rem .9rem; cursor: pointer; opacity: .65; font: inherit; }
  nav button.active { opacity: 1; border-bottom-color: var(--accent); font-weight: 600; }
  .pane { display: none; }
  .pane.active { display: block; }
  #log, #timeline, #runs, #agent { border: 1px solid var(--line); border-radius: 8px;
         padding: 1rem; min-height: 50vh; overflow-y: auto; }
  #log { white-space: pre-wrap; }
  .msg { margin: .5rem 0; }
  .user { font-weight: 600; }
  .assistant { opacity: .95; }
  form { display: flex; gap: .5rem; margin-top: .75rem; }
  input { flex: 1; padding: .6rem; border-radius: 8px; border: 1px solid #8888;
          background: transparent; color: inherit; }
  button.send, button.act { padding: .6rem 1rem; border-radius: 8px; border: 0; cursor: pointer;
          background: var(--accent); color: #fff; }
  button.cancel { padding: .3rem .7rem; border-radius: 6px; border: 1px solid #c55;
          background: transparent; color: inherit; cursor: pointer; }
  /* event timeline */
  .ev { border: 1px solid var(--line); border-left-width: 4px; border-radius: 6px;
        margin: .4rem 0; padding: .4rem .6rem; }
  .ev summary { cursor: pointer; font-weight: 600; list-style: none; }
  .ev pre { margin: .4rem 0 0; white-space: pre-wrap; word-break: break-word;
        font-size: .8rem; opacity: .9; }
  /* latency waterfall bar: width is proportional to the span's share of the
     slowest span, so slow steps read as visibly wider at a glance. */
  .lat-bar { height: 6px; border-radius: 3px; margin: .3rem 0 0;
        background: var(--accent); opacity: .55; min-width: 2px; }
  /* a small inline chip (e.g. the model name on a model_response row). */
  .chip { display: inline-block; padding: .05rem .45rem; margin-left: .4rem;
        border-radius: 999px; font-size: .7rem; border: 1px solid var(--line);
        opacity: .8; vertical-align: middle; font-weight: 400; }
  .ev-run_start   { border-left-color: #7a7; }
  .ev-text_delta  { border-left-color: #888; }
  .ev-tool_call   { border-left-color: #d90; }
  .ev-tool_result { border-left-color: #5b9; }
  .ev-agent_transfer { border-left-color: #b6f; }
  .ev-final_output { border-left-color: var(--accent); }
  .ev-run_end     { border-left-color: #7a7; }
  .ev-error       { border-left-color: #e55; }
  /* runs + agent */
  .run-row { display: flex; align-items: center; gap: .6rem; padding: .4rem 0;
        border-bottom: 1px solid var(--line); }
  .run-row code { flex: 1; }
  .badge { padding: .1rem .5rem; border-radius: 999px; font-size: .75rem;
           border: 1px solid var(--line); }
  .badge-running { color: #d90; border-color: #d90; }
  .badge-completed { color: #5b9; border-color: #5b9; }
  .badge-failed { color: #e55; border-color: #e55; }
  .badge-cancelled { color: #999; }
  .kv { display: grid; grid-template-columns: max-content 1fr; gap: .25rem .8rem; }
  .kv dt { font-weight: 600; opacity: .8; }
  .kv dd { margin: 0; word-break: break-word; }
  .tool { border: 1px solid var(--line); border-radius: 6px;
          padding: .4rem .6rem; margin: .3rem 0; }
  .tool b { font-family: ui-monospace, monospace; }
  .muted { opacity: .6; }
</style>
</head>
<body>
<h1>YAAB · {name} <span class="sub">dev console</span></h1>
<nav id="tabs">
  <button data-tab="chat" class="active" onclick="switchTab('chat')">Chat</button>
  <button data-tab="events" onclick="switchTab('events')">Events</button>
  <button data-tab="runs" onclick="switchTab('runs')">Runs</button>
  <button data-tab="trace" onclick="switchTab('trace')">Trace</button>
  <button data-tab="state" onclick="switchTab('state')">State</button>
  <button data-tab="approvals" onclick="switchTab('approvals')">Approvals</button>
  <button data-tab="agent" onclick="switchTab('agent')">Agent</button>
</nav>

<!-- CHAT -->
<section class="pane active" data-pane="chat">
  <div id="log"></div>
  <form id="chatForm"><input id="chatQ" placeholder="Ask the agent…" autocomplete="off"/>
  <button class="send">Send</button></form>
</section>

<!-- EVENTS -->
<section class="pane" data-pane="events">
  <div id="evSummary" class="muted"></div>
  <div id="timeline"><p class="muted">Send a message to inspect the event stream.</p></div>
  <form id="eventForm"><input id="eventQ" placeholder="Run and inspect events…" autocomplete="off"/>
  <button class="send">Run</button></form>
</section>

<!-- RUNS -->
<section class="pane" data-pane="runs">
  <p class="muted">Auto-refreshing every 3s while this tab is open.
    <button class="act" onclick="refreshRuns()">Refresh now</button></p>
  <div id="runs"><p class="muted">No runs yet.</p></div>
</section>

<!-- TRACE: per-step model/tool/token/cost/latency waterfall -->
<section class="pane" data-pane="trace">
  <form id="traceForm">
  <input id="traceRunId" placeholder="run id (blank for the latest run)" autocomplete="off"/>
  <button class="send">Load trace</button></form>
  <div id="traceTotals" class="muted"></div>
  <div id="trace"><p class="muted">Enter a run id to replay its trace (model calls, tools,
  tokens, cost, latency). Configure a trace store to persist runs.</p></div>
</section>

<!-- STATE: session KV inspector -->
<section class="pane" data-pane="state">
  <form id="stateForm"><input id="stateSession" placeholder="session id" autocomplete="off"/>
  <button class="send">Inspect</button>
  <button type="button" class="act" onclick="refreshState()">Refresh</button></form>
  <div id="state"><p class="muted">Enter a session id to inspect its stored state.
  Configure a session service to persist state.</p></div>
</section>

<!-- APPROVALS: out-of-band human sign-off -->
<section class="pane" data-pane="approvals">
  <p class="muted">Pending sign-offs.
  <button class="act" onclick="loadApprovals()">Refresh</button></p>
  <div id="approvals"><p class="muted">No pending approvals. Configure an approval store
  to enable sign-off.</p></div>
</section>

<!-- AGENT -->
<section class="pane" data-pane="agent">
  <div id="agent"><p class="muted">Loading agent info…</p></div>
</section>

<script>
// ---- tab switching --------------------------------------------------
let runsTimer = null;
function switchTab(name) {
  document.querySelectorAll('#tabs button').forEach(b =>
    b.classList.toggle('active', b.dataset.tab === name));
  document.querySelectorAll('.pane').forEach(p =>
    p.classList.toggle('active', p.dataset.pane === name));
  // Lazy-load + lifecycle per tab.
  if (name === 'runs') { refreshRuns(); startRunsPolling(); } else { stopRunsPolling(); }
  if (name === 'agent') loadAgent();
  if (name === 'approvals') loadApprovals();
}
function startRunsPolling() {
  if (runsTimer) return;
  runsTimer = setInterval(refreshRuns, 3000);  // auto-refresh while Runs is open
}
function stopRunsPolling() {
  if (runsTimer) { clearInterval(runsTimer); runsTimer = null; }
}

// ---- CHAT tab: token streaming over /chat/stream --------------------
const log = document.getElementById('log');
function addMsg(role, text) {
  const d = document.createElement('div');
  d.className = 'msg ' + role;
  d.textContent = (role === 'user' ? 'You: ' : '') + text;
  log.appendChild(d); log.scrollTop = log.scrollHeight; return d;
}
document.getElementById('chatForm').addEventListener('submit', async (e) => {
  e.preventDefault();
  const q = document.getElementById('chatQ'); const prompt = q.value.trim();
  if (!prompt) return; q.value = '';
  addMsg('user', prompt);
  const out = addMsg('assistant', '');
  const resp = await fetch('/chat/stream', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({prompt})
  });
  const reader = resp.body.getReader(); const dec = new TextDecoder();
  let buf = '';
  while (true) {
    const {value, done} = await reader.read(); if (done) break;
    buf += dec.decode(value, {stream: true});
    let idx;
    while ((idx = buf.indexOf('\\n\\n')) >= 0) {
      const frame = buf.slice(0, idx); buf = buf.slice(idx + 2);
      for (const line of frame.split('\\n')) {
        if (line.startsWith('data: ')) {
          const data = line.slice(6);
          if (data === '[DONE]') continue;
          out.textContent += data;
          log.scrollTop = log.scrollHeight;
        }
      }
    }
  }
});

// ---- EVENTS tab: POST /run/stream SSE via fetch ReadableStream ------
// EventSource can only GET, so POST-SSE needs a streaming fetch reader. We
// parse SSE frames (event:/data: separated by a blank line) by hand.
const timeline = document.getElementById('timeline');
const evSummary = document.getElementById('evSummary');
const EV_KNOWN = ['run_start','text_delta','tool_call','tool_result',
                  'agent_transfer','final_output','run_end','approval_required','error'];
// A run_end payload now carries usage (tokens/cost) and per-event duration_ms;
// the summary header surfaces those totals so cost/latency are visible at a glance.
function renderUsageSummary(payload) {
  const u = (payload && payload.usage) || {};
  const total = u.total_tokens != null ? u.total_tokens : '?';
  const cost = u.cost_usd != null ? u.cost_usd : (u.cost != null ? u.cost : '?');
  const dur = payload && payload.duration_ms != null ? payload.duration_ms + ' ms' : '';
  evSummary.textContent = 'run_end · total_tokens=' + total + ' · cost_usd=' + cost +
    (dur ? ' · ' + dur : '');
}
// The model name lives flat on a live SSE event (_safe_event_payload) but nested
// under .payload on a replayed event (_safe_event); look in both so the chip
// shows for live runs and replays alike.
function eventModel(payload) {
  if (!payload) return null;
  if (payload.model) return payload.model;
  if (payload.payload && payload.payload.model) return payload.payload.model;
  return null;
}
function renderEvent(type, payload) {
  const el = document.createElement('details');
  el.className = 'ev ev-' + (EV_KNOWN.includes(type) ? type : 'run_start');
  const sum = document.createElement('summary');
  const dur = payload && payload.duration_ms != null ? ' (' + payload.duration_ms + ' ms)' : '';
  sum.textContent = type + dur;
  // Model-name chip: which model answered this step, visible without expanding.
  const model = eventModel(payload);
  if (model) {
    const chip = document.createElement('span');
    chip.className = 'chip'; chip.textContent = model;
    sum.appendChild(chip);
  }
  const pre = document.createElement('pre');
  pre.textContent = JSON.stringify(payload, null, 2);
  el.appendChild(sum); el.appendChild(pre);
  timeline.appendChild(el); timeline.scrollTop = timeline.scrollHeight;
  if (type === 'run_end') renderUsageSummary(payload);
}
async function streamRun(prompt) {
  timeline.innerHTML = ''; evSummary.textContent = '';
  const resp = await fetch('/run/stream', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({prompt})
  });
  const reader = resp.body.getReader(); const dec = new TextDecoder();
  let buf = '';
  while (true) {
    const {value, done} = await reader.read(); if (done) break;
    buf += dec.decode(value, {stream: true});
    let idx;
    while ((idx = buf.indexOf('\\n\\n')) >= 0) {
      const frame = buf.slice(0, idx); buf = buf.slice(idx + 2);
      let evType = 'message', data = '';
      for (const line of frame.split('\\n')) {
        if (line.startsWith('event: ')) evType = line.slice(7).trim();
        else if (line.startsWith('data: ')) data += line.slice(6);
      }
      if (evType === 'done') continue;
      let payload = data;
      try { payload = JSON.parse(data); } catch (_) {}
      renderEvent(evType, payload);
    }
  }
}
document.getElementById('eventForm').addEventListener('submit', (e) => {
  e.preventDefault();
  const q = document.getElementById('eventQ'); const prompt = q.value.trim();
  if (!prompt) return; q.value = '';
  streamRun(prompt);
});

// ---- RUNS tab: GET /runs + per-run cancel ---------------------------
const runsEl = document.getElementById('runs');
async function refreshRuns() {
  let items;
  try { items = await (await fetch('/runs')).json(); }
  catch (_) { runsEl.innerHTML = '<p class="muted">Could not load runs.</p>'; return; }
  if (!items.length) { runsEl.innerHTML = '<p class="muted">No runs yet.</p>'; return; }
  runsEl.innerHTML = '';
  for (const r of items) {
    const row = document.createElement('div'); row.className = 'run-row';
    const code = document.createElement('code'); code.textContent = r.id;
    const badge = document.createElement('span');
    badge.className = 'badge badge-' + r.status; badge.textContent = r.status;
    row.appendChild(code); row.appendChild(badge);
    // Per-run Trace + Replay actions (the Runs-tab upgrade).
    const traceBtn = document.createElement('button');
    traceBtn.className = 'cancel'; traceBtn.textContent = 'Trace';
    traceBtn.onclick = () => openTrace(r.id);
    row.appendChild(traceBtn);
    const replayBtn = document.createElement('button');
    replayBtn.className = 'cancel'; replayBtn.textContent = 'Replay';
    replayBtn.onclick = () => replayRun(r.id);
    row.appendChild(replayBtn);
    if (r.status === 'running') {
      const btn = document.createElement('button');
      btn.className = 'cancel'; btn.textContent = 'Cancel';
      btn.onclick = () => cancelRun(r.id);
      row.appendChild(btn);
    }
    runsEl.appendChild(row);
  }
}
async function cancelRun(id) {
  try { await fetch('/runs/' + encodeURIComponent(id) + '/cancel', {method: 'POST'}); }
  catch (_) {}
  refreshRuns();
}
// Jump from a Runs row to that run's Trace tab, pre-filled and loaded.
function openTrace(id) {
  switchTab('trace');
  document.getElementById('traceRunId').value = id;
  loadTrace(id);
}

// ---- TRACE tab: GET /runs/{id}/trace -> span waterfall --------------
// Renders the computed span/waterfall: each span is typed (model_call,
// tool_call, transfer, approval), carries a duration_ms latency bar, and model
// spans show token + cost_usd badges. Run totals (total_tokens, cost) head it.
const traceEl = document.getElementById('trace');
const traceTotals = document.getElementById('traceTotals');
function renderSpan(s, maxDur) {
  const el = document.createElement('details');
  el.className = 'ev ev-' + (s.type === 'model_call' ? 'final_output'
    : s.type === 'tool_call' ? 'tool_call'
    : s.type === 'transfer' ? 'agent_transfer'
    : s.type === 'approval' ? 'agent_transfer' : 'run_start');
  const sum = document.createElement('summary');
  const dur = s.duration_ms != null ? s.duration_ms + ' ms' : '';
  let badge = '';
  if (s.type === 'model_call') {
    const tin = s.input_tokens != null ? s.input_tokens : '?';
    const tout = s.output_tokens != null ? s.output_tokens : '?';
    const cost = s.cost_usd != null ? s.cost_usd : '?';
    badge = ' · in=' + tin + ' out=' + tout + ' · cost_usd=' + cost;
  }
  sum.textContent = (s.type || 'span') + (s.name ? ' ' + s.name : '') +
    (dur ? ' · ' + dur : '') + badge;
  // Model spans carry which model answered — surface it as a chip.
  if (s.type === 'model_call' && s.model) {
    const chip = document.createElement('span');
    chip.className = 'chip'; chip.textContent = s.model;
    sum.appendChild(chip);
  }
  el.appendChild(sum);
  // Proportional latency bar: this span's share of the slowest span, so the
  // waterfall reads as a waterfall (wide = slow) without expanding each row.
  const d = s.duration_ms != null ? Number(s.duration_ms) : 0;
  if (maxDur > 0) {
    const bar = document.createElement('div');
    bar.className = 'lat-bar';
    bar.style.width = Math.max(2, Math.round((d / maxDur) * 100)) + '%';
    el.appendChild(bar);
  }
  const pre = document.createElement('pre');
  pre.textContent = JSON.stringify(s, null, 2);
  el.appendChild(pre);
  return el;
}
async function loadTrace(runId) {
  const id = runId || document.getElementById('traceRunId').value.trim();
  if (!id) { traceEl.innerHTML = '<p class="muted">Enter a run id.</p>'; return; }
  traceEl.innerHTML = '<p class="muted">Loading…</p>'; traceTotals.textContent = '';
  let body;
  try {
    const resp = await fetch('/runs/' + encodeURIComponent(id) + '/trace');
    if (!resp.ok) {
      traceEl.innerHTML = '<p class="muted">No trace (enable a trace store to persist runs).</p>';
      return;
    }
    body = await resp.json();
  } catch (_) { traceEl.innerHTML = '<p class="muted">Could not load trace.</p>'; return; }
  const t = body.totals || {};
  const tokens = (t.total_tokens != null ? t.total_tokens : '?');
  const cost = (t.cost_usd != null ? t.cost_usd : (t.total_cost != null ? t.total_cost : '?'));
  const dur = (t.duration_ms != null ? t.duration_ms : '?');
  traceTotals.textContent =
    'totals · total_tokens=' + tokens + ' · cost_usd=' + cost + ' · duration_ms=' + dur;
  traceEl.innerHTML = '';
  const spans = body.spans || [];
  // The slowest span anchors the latency bars (every other bar is its share).
  let maxDur = 0;
  for (const s of spans) {
    const d = s.duration_ms != null ? Number(s.duration_ms) : 0;
    if (d > maxDur) maxDur = d;
  }
  for (const s of spans) traceEl.appendChild(renderSpan(s, maxDur));
}
document.getElementById('traceForm').addEventListener('submit', (e) => {
  e.preventDefault(); loadTrace();
});

// ---- RUNS tab: Replay -> GET /runs/{id}/events re-render (no re-run) -
async function replayRun(id) {
  switchTab('events');
  timeline.innerHTML = '<p class="muted">Replaying ' + id + '…</p>';
  evSummary.textContent = '';
  let body;
  try {
    const resp = await fetch('/runs/' + encodeURIComponent(id) + '/events');
    if (!resp.ok) {
      timeline.innerHTML = '<p class="muted">No persisted events (enable a trace store).</p>';
      return;
    }
    body = await resp.json();
  } catch (_) { timeline.innerHTML = '<p class="muted">Could not replay.</p>'; return; }
  timeline.innerHTML = '';
  for (const ev of (body.events || [])) renderEvent(ev.type || 'message', ev);
}

// ---- STATE tab: GET /sessions/{id}/state -> KV table ----------------
const stateEl = document.getElementById('state');
async function loadState(sid) {
  const id = sid || document.getElementById('stateSession').value.trim();
  if (!id) { stateEl.innerHTML = '<p class="muted">Enter a session id.</p>'; return; }
  stateEl.innerHTML = '<p class="muted">Loading…</p>';
  let body;
  try {
    const resp = await fetch('/sessions/' + encodeURIComponent(id) + '/state');
    if (!resp.ok) { stateEl.innerHTML = '<p class="muted">Unknown session.</p>'; return; }
    body = await resp.json();
  } catch (_) { stateEl.innerHTML = '<p class="muted">Could not load state.</p>'; return; }
  stateEl.innerHTML = kvBlock('Session ' + (body.session_id || id), body.state || {});
}
document.getElementById('stateForm').addEventListener('submit', (e) => {
  e.preventDefault(); loadState();
});
// One-click re-pull of the session already in the box (state changes between
// turns) — no need to retype the id.
function refreshState() { loadState(); }

// ---- APPROVALS tab: list pending + approve/deny ---------------------
const approvalsEl = document.getElementById('approvals');
async function loadApprovals() {
  approvalsEl.innerHTML = '<p class="muted">Loading…</p>';
  let items;
  try {
    const resp = await fetch('/approvals?status=pending');
    if (!resp.ok) {
      approvalsEl.innerHTML = '<p class="muted">No approval store configured.</p>';
      return;
    }
    items = await resp.json();
  } catch (_) { approvalsEl.innerHTML = '<p class="muted">Could not load approvals.</p>'; return; }
  if (!items.length) {
    approvalsEl.innerHTML = '<p class="muted">No pending approvals.</p>';
    return;
  }
  approvalsEl.innerHTML = '';
  for (const a of items) {
    const el = document.createElement('div'); el.className = 'tool';
    el.innerHTML = '<b>' + escapeHtml(a.tool || '?') + '</b> ' +
      '<span class="muted">' + escapeHtml(JSON.stringify(a.arguments || {})) + '</span>';
    const ok = document.createElement('button'); ok.className = 'act'; ok.textContent = 'Approve';
    ok.onclick = () => decideApproval(a.approval_id, 'approve');
    const no = document.createElement('button'); no.className = 'cancel'; no.textContent = 'Deny';
    no.onclick = () => decideApproval(a.approval_id, 'deny');
    el.appendChild(document.createElement('br')); el.appendChild(ok); el.appendChild(no);
    approvalsEl.appendChild(el);
  }
}
async function decideApproval(id, action) {
  try {
    await fetch('/approvals/' + encodeURIComponent(id) + '/' + action, {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({reviewer: 'console'})
    });
  } catch (_) {}
  loadApprovals();
}

// ---- AGENT tab: agent card + /agent/info ----------------------------
const agentEl = document.getElementById('agent');
function kvBlock(title, obj) {
  const parts = ['<h2 style="font-size:1rem">' + title + '</h2><dl class="kv">'];
  for (const [k, v] of Object.entries(obj || {})) {
    const val = (typeof v === 'object') ? JSON.stringify(v) : String(v);
    parts.push('<dt></dt><dd></dd>'.replace('<dt></dt>', '<dt>' + escapeHtml(k) + '</dt>')
                                    .replace('<dd></dd>', '<dd>' + escapeHtml(val) + '</dd>'));
  }
  parts.push('</dl>');
  return parts.join('');
}
function escapeHtml(s) {
  return String(s).replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
}
async function loadAgent() {
  agentEl.innerHTML = '<p class="muted">Loading…</p>';
  let info = {}, card = {};
  try { info = await (await fetch('/agent/info')).json(); } catch (_) {}
  try { card = await (await fetch('/.well-known/agent.json')).json(); } catch (_) {}
  let html = kvBlock('Agent', {
    name: info.name, model: info.model, instructions: info.instructions
  });
  html += '<h2 style="font-size:1rem">Tools (' + (info.tools || []).length + ')</h2>';
  if ((info.tools || []).length) {
    for (const t of info.tools) {
      html += '<div class="tool"><b>' + escapeHtml(t.name) + '</b><div class="muted">' +
              escapeHtml(t.description || '') + '</div></div>';
    }
  } else {
    html += '<p class="muted">No tools.</p>';
  }
  if (card && card.name) html += kvBlock('Agent card', card);
  agentEl.innerHTML = html;
}
</script>
</body>
</html>"""


def _agent_info(agent: Any) -> dict[str, Any]:
    """Introspect ``agent`` into a JSON-safe document for the Agent tab.

    ``instructions`` may be a plain string or a callable (dynamic instructions);
    callables aren't JSON-serializable and would otherwise 500 the endpoint, so
    we stringify non-string values defensively. The model spec is stringified
    too — it's an arbitrary provider object, not necessarily JSON-able.
    """
    instructions = getattr(agent, "instructions", "")
    if not isinstance(instructions, str):
        # Dynamic/callable instructions: surface a readable placeholder rather
        # than dumping a repr the UI can't use.
        instructions = "<dynamic instructions>" if callable(instructions) else str(instructions)
    tools = [
        {
            "name": getattr(t, "name", ""),
            "description": getattr(t, "description", "") or "",
        }
        for t in getattr(agent, "tools", [])
    ]
    return {
        "name": agent.name,
        "model": str(agent._model_spec),
        "tools": tools,
        "instructions": instructions,
    }


def web_app(
    agent: Any,
    *,
    runner: Any | None = None,
    auth: Any | None = None,
    run_store: Any | None = None,
    approval_store: Any | None = None,
    trace_store: Any | None = None,
    run_checkpointer: Any | None = None,
    cron_store: Any | None = None,
) -> Any:
    """Build a FastAPI app: the agent's API + the browser dev console at ``/``.

    The console mounts the full serve app (so ``/run/stream``, ``/runs``,
    ``/chat/stream`` and the agent card are reachable here) and adds two
    web-only routes: ``GET /`` (the single-page console) and ``GET /agent/info``
    (introspection for the Agent tab). ``/agent/info`` lives on the web app, not
    on serve, because it's a dev-console convenience rather than part of the
    agent's served contract.

    The durable backends are forwarded straight to :func:`fastapi_server_app`, so
    passing a ``trace_store`` lights up the Trace + State tabs (per-step
    model/tool/token/cost/latency detail and the session-state inspector), and an
    ``approval_store`` (+ ``run_store``) lights up the Approvals tab for
    out-of-band human sign-off. When a backend is omitted, its tab degrades
    gracefully — the matching endpoints return a clean ``404`` and the tab shows
    a "configure a store to enable this" hint rather than erroring.
    """
    from .serve import fastapi_server_app

    try:
        from fastapi.responses import HTMLResponse, JSONResponse
    except ImportError as exc:  # pragma: no cover - optional extra
        raise RuntimeError("FastAPI is required. `pip install fastapi uvicorn`.") from exc

    app = fastapi_server_app(
        agent,
        runner=runner,
        auth=auth,
        run_store=run_store,
        approval_store=approval_store,
        trace_store=trace_store,
        run_checkpointer=run_checkpointer,
        cron_store=cron_store,
    )
    page = _PAGE.replace("{name}", agent.name)

    @app.get("/")
    async def playground() -> Any:
        return HTMLResponse(page)

    @app.get("/agent/info")
    async def agent_info() -> Any:
        return JSONResponse(_agent_info(agent))

    return app


def serve_web(
    agent: Any, *, host: str = "127.0.0.1", port: int = 8080, auth: Any | None = None
) -> None:  # pragma: no cover - thin uvicorn wrapper
    """Run the dev console with uvicorn (blocking)."""
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError("uvicorn is required. `pip install uvicorn`.") from exc
    uvicorn.run(web_app(agent, auth=auth), host=host, port=port)


__all__ = ["web_app", "serve_web"]
