"""`yaab web` — a zero-build local dev playground for an agent.

Serves a single self-contained HTML page (no bundler, no npm) that streams the
agent's tokens over the existing ``/chat/stream`` SSE endpoint, and mounts the
agent's full API (``/run``, ``/run/stream``, ``/a2a/tasks``, the agent card). It
is the local "open it in a browser and talk to your agent" experience the spec
called for.

    from yaab.web import web_app
    # uvicorn module:app   where app = web_app(agent)

or via the CLI: ``yaab web mymodule:agent``.
"""

from __future__ import annotations

from typing import Any

_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>YAAB · {name}</title>
<style>
  :root { color-scheme: light dark; }
  body { font-family: ui-sans-serif, system-ui, sans-serif; max-width: 820px;
         margin: 0 auto; padding: 1rem; }
  h1 { font-size: 1.1rem; opacity: .8; }
  #log { border: 1px solid #8884; border-radius: 8px; padding: 1rem; min-height: 50vh;
         white-space: pre-wrap; overflow-y: auto; }
  .msg { margin: .5rem 0; }
  .user { font-weight: 600; }
  .assistant { opacity: .95; }
  form { display: flex; gap: .5rem; margin-top: .75rem; }
  input { flex: 1; padding: .6rem; border-radius: 8px; border: 1px solid #8888; }
  button { padding: .6rem 1rem; border-radius: 8px; border: 0; cursor: pointer; }
</style>
</head>
<body>
<h1>YAAB · {name} <span style="opacity:.5">dev playground</span></h1>
<div id="log"></div>
<form id="f"><input id="q" placeholder="Ask the agent…" autocomplete="off"/>
<button>Send</button></form>
<script>
const log = document.getElementById('log');
function add(role, text) {
  const d = document.createElement('div');
  d.className = 'msg ' + role;
  d.textContent = (role === 'user' ? 'You: ' : '') + text;
  log.appendChild(d); log.scrollTop = log.scrollHeight; return d;
}
document.getElementById('f').addEventListener('submit', async (e) => {
  e.preventDefault();
  const q = document.getElementById('q'); const prompt = q.value.trim();
  if (!prompt) return; q.value = '';
  add('user', prompt);
  const out = add('assistant', '');
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
      const line = buf.slice(0, idx); buf = buf.slice(idx + 2);
      if (line.startsWith('data: ')) {
        const data = line.slice(6);
        if (data === '[DONE]') continue;
        out.textContent += data;
        log.scrollTop = log.scrollHeight;
      }
    }
  }
});
</script>
</body>
</html>"""


def web_app(agent: Any, *, runner: Any | None = None, auth: Any | None = None) -> Any:
    """Build a FastAPI app: the agent's API + a browser chat playground at ``/``."""
    from .serve import fastapi_server_app

    try:
        from fastapi.responses import HTMLResponse
    except ImportError as exc:  # pragma: no cover - optional extra
        raise RuntimeError("FastAPI is required. `pip install fastapi uvicorn`.") from exc

    app = fastapi_server_app(agent, runner=runner, auth=auth)
    page = _PAGE.replace("{name}", agent.name)

    @app.get("/")
    async def playground() -> Any:
        return HTMLResponse(page)

    return app


def serve_web(
    agent: Any, *, host: str = "127.0.0.1", port: int = 8080, auth: Any | None = None
) -> None:  # pragma: no cover - thin uvicorn wrapper
    """Run the dev playground with uvicorn (blocking)."""
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError("uvicorn is required. `pip install uvicorn`.") from exc
    uvicorn.run(web_app(agent, auth=auth), host=host, port=port)


__all__ = ["web_app", "serve_web"]
