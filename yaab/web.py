"""`yaab web` — a zero-build local dev console for an agent.

Serves a single self-contained HTML page (no bundler, no npm, no CDN) that is a
real dev console rather than a bare chat box. It mounts the agent's full API
(``/run``, ``/run/stream``, ``/runs``, the agent card — see :mod:`yaab.serve`)
and layers four tabs over it:

* **Chat** — token streaming over ``/chat/stream`` (the original behavior).
* **Events** — a live event-stream inspector: sends over ``POST /run/stream``
  (SSE) and renders every typed event (run_start, text_delta, tool_call,
  tool_result, agent_transfer, final_output, run_end) as a colour-coded,
  expandable timeline with payload JSON.
* **Runs** — lists runs from ``GET /runs`` with status badges, a per-run Cancel
  button (``POST /runs/{id}/cancel``), auto-refreshing while the tab is open.
* **Agent** — renders ``GET /.well-known/agent.json`` (the agent card) plus the
  agent's tool list and instructions from ``GET /agent/info`` (mounted here on
  the web app, not on the serve app).

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
const EV_KNOWN = ['run_start','text_delta','tool_call','tool_result',
                  'agent_transfer','final_output','run_end','error'];
function renderEvent(type, payload) {
  const el = document.createElement('details');
  el.className = 'ev ev-' + (EV_KNOWN.includes(type) ? type : 'run_start');
  const sum = document.createElement('summary');
  sum.textContent = type;
  const pre = document.createElement('pre');
  pre.textContent = JSON.stringify(payload, null, 2);
  el.appendChild(sum); el.appendChild(pre);
  timeline.appendChild(el); timeline.scrollTop = timeline.scrollHeight;
}
async function streamRun(prompt) {
  timeline.innerHTML = '';
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


def web_app(agent: Any, *, runner: Any | None = None, auth: Any | None = None) -> Any:
    """Build a FastAPI app: the agent's API + the browser dev console at ``/``.

    The console mounts the full serve app (so ``/run/stream``, ``/runs``,
    ``/chat/stream`` and the agent card are reachable here) and adds two
    web-only routes: ``GET /`` (the single-page console) and ``GET /agent/info``
    (introspection for the Agent tab). ``/agent/info`` lives on the web app, not
    on serve, because it's a dev-console convenience rather than part of the
    agent's served contract.
    """
    from .serve import fastapi_server_app

    try:
        from fastapi.responses import HTMLResponse, JSONResponse
    except ImportError as exc:  # pragma: no cover - optional extra
        raise RuntimeError("FastAPI is required. `pip install fastapi uvicorn`.") from exc

    app = fastapi_server_app(agent, runner=runner, auth=auth)
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
