#!/usr/bin/env python
"""Live server check — runs the out-of-the-box FastAPI server over a real TCP
port with a REAL model behind it, and exercises every endpoint over HTTP.

Unlike tests/test_serve_endpoints.py (in-process ASGI, TestModel), this binds
uvicorn to a localhost port and drives it with httpx, with a live model serving
/run and /a2a/tasks — proving the serving stack works end to end.

    GEMINI_API_KEY=...  YAAB_LIVE_MODEL=gemini/gemini-2.5-flash
    python scripts/live_serve_check.py

Requires: pip install 'yaab[serve,litellm,http]'
"""

from __future__ import annotations

import os
import socket
import sys
import threading
import time
from pathlib import Path


def _load_dotenv() -> None:
    env = Path(__file__).resolve().parent.parent / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())


_load_dotenv()
MODEL = os.environ.get("YAAB_LIVE_MODEL", "gemini/gemini-2.5-flash")


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def main() -> int:
    if not os.environ.get("GEMINI_API_KEY") and MODEL.startswith("gemini/"):
        print("GEMINI_API_KEY not set (add to .env).")
        return 2

    import httpx
    import uvicorn

    from yaab import Agent
    from yaab.auth import BearerTokenAuth
    from yaab.serve import fastapi_server_app

    agent = Agent("assistant", model=MODEL, registry_id="assistant",
                  instructions="Answer in one short sentence.")
    auth = BearerTokenAuth({"secret-token": "alice"})
    port = _free_port()
    app = fastapi_server_app(agent, auth=auth, base_url=f"http://127.0.0.1:{port}")

    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    base = f"http://127.0.0.1:{port}"
    auth_h = {"Authorization": "Bearer secret-token"}
    results: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        results.append((name, ok, detail))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")

    # Wait for the server to come up.
    deadline = time.time() + 20
    with httpx.Client(timeout=30) as client:
        while time.time() < deadline:
            try:
                if client.get(f"{base}/health").status_code == 200:
                    break
            except Exception:  # noqa: BLE001
                time.sleep(0.2)
        else:
            print("server did not start")
            return 1

        try:
            r = client.get(f"{base}/health")
            check("GET /health", r.status_code == 200 and r.json()["status"] == "ok",
                  r.json().get("agent", ""))
        except Exception as e:  # noqa: BLE001
            check("GET /health", False, str(e)[:60])

        try:
            r = client.get(f"{base}/.well-known/agent.json")
            card = r.json()
            check("GET /.well-known/agent.json",
                  r.status_code == 200 and card["name"] == "assistant"
                  and "bearer" in card.get("securitySchemes", {}),
                  f"card={card['name']}")
        except Exception as e:  # noqa: BLE001
            check("agent card", False, str(e)[:60])

        # Auth: missing token rejected.
        try:
            r = client.post(f"{base}/run", json={"prompt": "hi"})
            check("POST /run (no auth -> 401)", r.status_code == 401, f"status={r.status_code}")
        except Exception as e:  # noqa: BLE001
            check("auth 401", False, str(e)[:60])

        # /run with a real model.
        try:
            r = client.post(f"{base}/run", json={"prompt": "What is the capital of France?"},
                            headers=auth_h)
            out = r.json().get("output", "")
            check("POST /run (live model)", r.status_code == 200 and "paris" in out.lower(),
                  f"{out[:40]!r} | tok={r.json().get('usage', {}).get('total_tokens')}")
        except Exception as e:  # noqa: BLE001
            check("POST /run", False, str(e)[:60])

        # A2A task submit + poll.
        try:
            r = client.post(f"{base}/a2a/tasks",
                            json={"message": {"parts": [{"text": "Say the word OK."}]}},
                            headers=auth_h)
            task = r.json()
            tid = task["id"]
            polled = client.get(f"{base}/a2a/tasks/{tid}", headers=auth_h)
            check("POST /a2a/tasks + poll (live)",
                  r.status_code == 200 and task["status"]["state"] == "completed"
                  and polled.json()["id"] == tid,
                  f"task={tid[:18]} state={task['status']['state']}")
        except Exception as e:  # noqa: BLE001
            check("A2A task", False, str(e)[:60])

        # SSE event stream over HTTP.
        try:
            got_start = got_end = False
            with client.stream("POST", f"{base}/run/stream",
                               json={"prompt": "Count to three."}, headers=auth_h) as resp:
                for line in resp.iter_lines():
                    if "event: run_start" in line:
                        got_start = True
                    if "event: run_end" in line:
                        got_end = True
            check("POST /run/stream (SSE)", got_start and got_end, "run_start+run_end seen")
        except Exception as e:  # noqa: BLE001
            check("run/stream", False, str(e)[:60])

        # Token streaming over HTTP.
        try:
            chunks = 0
            with client.stream("POST", f"{base}/chat/stream",
                               json={"prompt": "Say hello."}, headers=auth_h) as resp:
                for line in resp.iter_lines():
                    if line.startswith("data:") and "[DONE]" not in line:
                        chunks += 1
            check("POST /chat/stream (tokens)", chunks >= 1, f"{chunks} data lines")
        except Exception as e:  # noqa: BLE001
            check("chat/stream", False, str(e)[:60])

    server.should_exit = True
    passed = sum(1 for _, ok, _ in results if ok)
    print(f"\n{passed}/{len(results)} server checks passed against {MODEL}.")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
