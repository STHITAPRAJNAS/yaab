"""A2A client — discover and delegate to a remote agent over HTTP.

``httpx`` is an optional dependency, imported lazily. An injectable
``transport`` callable (``method, path, json -> dict``) keeps the client fully
testable without a network — the test suite drives a real :func:`get_fastapi_app`
server through an in-process transport.
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable, Optional

from ..types import RunContext, RunResult, Usage

Transport = Callable[[str, str, Optional[dict]], Awaitable[dict]]


class RemoteAgent:
    """A handle to a remote A2A agent — usable as a tool or as an agent."""

    def __init__(
        self,
        url: str,
        *,
        name: Optional[str] = None,
        auth_token: Optional[str] = None,
        transport: Optional[Transport] = None,
    ) -> None:
        self.url = url.rstrip("/")
        self.name = name or "remote_agent"
        self.auth_token = auth_token
        self._transport = transport
        self._card: Optional[dict[str, Any]] = None

    # --- transport -----------------------------------------------------
    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.auth_token}"} if self.auth_token else {}

    async def _request(self, method: str, path: str, json: Optional[dict] = None) -> dict:
        if self._transport is not None:
            return await self._transport(method, path, json)
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover - optional extra
            raise RuntimeError(
                "httpx is required for the A2A client. `pip install httpx`."
            ) from exc
        async with httpx.AsyncClient() as client:
            resp = await client.request(
                method, f"{self.url}{path}", json=json, headers=self._headers()
            )
            resp.raise_for_status()
            return resp.json()

    # --- discovery -----------------------------------------------------
    async def fetch_card(self, *, refresh: bool = False) -> dict[str, Any]:
        """Fetch (and cache) the remote Agent Card."""
        if self._card is None or refresh:
            self._card = await self._request("GET", "/.well-known/agent.json", None)
            if not self._card.get("name"):
                self._card["name"] = self.name
            else:
                self.name = self.name or self._card["name"]
        return self._card

    # --- delegation ----------------------------------------------------
    async def run(
        self,
        prompt: str,
        *,
        deps: Any = None,
        session_id: Optional[str] = None,
        identity: Optional[str] = None,
    ) -> RunResult[str]:
        """Send the prompt as an A2A task and return the remote agent's output."""
        body = {"message": {"role": "user", "parts": [{"text": prompt}]}}
        task = await self._request("POST", "/a2a/tasks", body)
        text = _extract_text(task)
        return RunResult(output=text, usage=Usage(requests=1), run_id=task.get("id", ""))

    def run_sync(self, prompt: str, **kwargs: Any) -> RunResult[str]:
        import asyncio

        return asyncio.run(self.run(prompt, **kwargs))

    # --- Tool protocol -------------------------------------------------
    @property
    def description(self) -> str:
        if self._card:
            return self._card.get("description", f"Remote agent at {self.url}")
        return f"Remote A2A agent at {self.url}"

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prompt": {"type": "string", "description": "Task for the remote agent."}
                    },
                    "required": ["prompt"],
                },
            },
        }

    async def execute(self, ctx: RunContext, *, prompt: str) -> Any:
        result = await self.run(prompt, identity=ctx.identity)
        ctx.usage.add(result.usage)
        return result.output


def _extract_text(task: dict[str, Any]) -> str:
    """Pull the text out of an A2A task's artifacts."""
    parts_text: list[str] = []
    for artifact in task.get("artifacts", []):
        for part in artifact.get("parts", []):
            if "text" in part:
                parts_text.append(part["text"])
    return "\n".join(parts_text)


__all__ = ["RemoteAgent"]
