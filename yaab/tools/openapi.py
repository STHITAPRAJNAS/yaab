"""OpenAPI/Swagger toolset auto-generation.

Turns an OpenAPI 3.x document into a list of :class:`Tool` objects — one per
operation (``path`` x ``method``) — so a model can call a REST API the same way
it calls a native function. Rather than hand-writing wrappers for every
endpoint, point YAAB at the spec and get typed, schema-bearing tools for free.

The design keeps the heavy bits optional and injectable:

* ``httpx`` is imported lazily so the SDK installs without it; tests inject a
  pre-built ``AsyncClient`` (backed by ``httpx.MockTransport``) via ``client=``.
* ``pyyaml`` is only imported when a YAML string is passed and isn't valid JSON,
  so JSON/dict specs never pull the dependency.
* Tools never raise on HTTP errors — a non-2xx response becomes an ``error:``
  string so a bad call can't crash the agent loop; the model sees the failure
  and can correct or give up.
"""

from __future__ import annotations

import json
import re
from typing import Any

from ..types import RunContext

# HTTP methods OpenAPI defines on a path item; anything else (``parameters``,
# ``summary``, ``$ref`` ...) is metadata, not an operation.
_HTTP_METHODS = ("get", "put", "post", "delete", "patch", "head", "options", "trace")

# Cap on the error-body excerpt fed back to the model: enough to diagnose, not
# so much that a giant HTML error page floods the context window.
_ERROR_EXCERPT = 500


class OpenAPITool:
    """A single OpenAPI operation exposed through the YAAB :class:`Tool` protocol.

    Carries the resolved request recipe (method, path template, the names of
    path/query params, whether a JSON body is expected) plus the transport
    config (base URL, static headers, optional injected client). The model-facing
    input schema is the union of path params, query params, and the JSON
    requestBody's properties, so the model fills one flat argument object and the
    tool routes each value to its correct place in the request.
    """

    def __init__(
        self,
        *,
        name: str,
        description: str,
        method: str,
        path: str,
        base_url: str,
        parameters: dict[str, Any],
        required: list[str],
        path_params: set[str],
        query_params: set[str],
        body_params: set[str],
        headers: dict[str, str] | None = None,
        client: Any | None = None,
    ) -> None:
        self.name = name
        self.description = description
        self._method = method.upper()
        self._path = path
        self._base_url = base_url.rstrip("/")
        self._parameters = parameters
        self._required = required
        self._path_params = path_params
        self._query_params = query_params
        self._body_params = body_params
        self._headers = headers or {}
        self._client = client

    def schema(self) -> dict[str, Any]:
        """Return the OpenAI function-calling schema (same shape as FunctionTool)."""
        params: dict[str, Any] = {
            "type": "object",
            "properties": self._parameters,
        }
        if self._required:
            params["required"] = self._required
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": params,
            },
        }

    async def execute(self, ctx: RunContext, **kwargs: Any) -> Any:
        """Issue the HTTP request, routing each kwarg to path/query/body.

        Returns parsed JSON on success (falling back to text), or an ``error:``
        string on transport failure or any non-2xx status — never raises, so a
        failed call surfaces to the model instead of crashing the run.
        """
        try:
            import httpx
        except ImportError:
            return "error: httpx is not installed (`pip install httpx`)"

        # Substitute path params into the template; the model supplies them as
        # plain kwargs, so anything left in {braces} means a required arg is
        # missing — report it rather than sending a malformed URL.
        path = self._path
        for pname in self._path_params:
            if pname in kwargs:
                path = path.replace(f"{{{pname}}}", _quote(kwargs[pname]))
        if "{" in path:
            missing = re.findall(r"{([^}]+)}", path)
            return f"error: missing required path parameter(s): {', '.join(missing)}"

        url = f"{self._base_url}{path}"
        query = {k: kwargs[k] for k in self._query_params if k in kwargs}
        body = {k: kwargs[k] for k in self._body_params if k in kwargs}

        request_kwargs: dict[str, Any] = {}
        if query:
            request_kwargs["params"] = query
        if body:
            request_kwargs["json"] = body
        if self._headers:
            request_kwargs["headers"] = self._headers

        owns_client = self._client is None
        client: Any = (
            httpx.AsyncClient(follow_redirects=True, timeout=30) if owns_client else self._client
        )
        try:
            resp = await client.request(self._method, url, **request_kwargs)
        except Exception as exc:  # noqa: BLE001 - report transport failures, don't crash the loop
            return f"error: request to {url} failed: {exc}"
        finally:
            if owns_client:
                await client.aclose()

        if resp.status_code >= 300:
            excerpt = resp.text[:_ERROR_EXCERPT]
            return f"error: HTTP {resp.status_code}: {excerpt}"
        try:
            return resp.json()
        except Exception:  # noqa: BLE001 - non-JSON body: hand back the raw text
            return resp.text


def openapi_toolset(
    spec: dict[str, Any] | str,
    *,
    base_url: str | None = None,
    headers: dict[str, str] | None = None,
    client: Any | None = None,
    operations: list[str] | None = None,
) -> list[OpenAPITool]:
    """Build one :class:`OpenAPITool` per operation in an OpenAPI 3.x document.

    Args:
        spec: A parsed OpenAPI dict, a JSON string, or a YAML string. YAML is
            only attempted (via lazily-imported pyyaml) when the string isn't
            valid JSON, so JSON/dict callers never need pyyaml installed.
        base_url: Overrides the spec's ``servers[0].url``. Useful for pointing a
            published spec at a staging host or a test transport.
        headers: Static headers attached to every request (e.g. an API key). The
            model can't see or alter these — they're caller-controlled auth.
        client: An injected ``httpx.AsyncClient`` (e.g. one backed by
            ``httpx.MockTransport`` in tests). When omitted, each call creates and
            closes its own client.
        operations: Optional allowlist of ``operationId``\\ s to include; anything
            not listed is skipped. The (slug) fallback name also matches.

    Returns:
        A list of tools, one per included operation.
    """
    doc = _load_spec(spec)
    server_url = base_url or _default_base_url(doc)
    allow = set(operations) if operations is not None else None

    tools: list[OpenAPITool] = []
    paths = doc.get("paths") or {}
    for path, item in paths.items():
        if not isinstance(item, dict):
            continue
        # Parameters declared at the path level apply to every operation on it.
        shared_params = item.get("parameters", [])
        for method in _HTTP_METHODS:
            op = item.get(method)
            if not isinstance(op, dict):
                continue
            op_id = op.get("operationId") or _slug(method, path)
            if allow is not None and op_id not in allow:
                continue
            tools.append(
                _build_tool(
                    op_id=op_id,
                    op=op,
                    method=method,
                    path=path,
                    base_url=server_url,
                    shared_params=shared_params,
                    headers=headers,
                    client=client,
                )
            )
    return tools


def _build_tool(
    *,
    op_id: str,
    op: dict[str, Any],
    method: str,
    path: str,
    base_url: str,
    shared_params: list[dict[str, Any]],
    headers: dict[str, str] | None,
    client: Any | None,
) -> OpenAPITool:
    properties: dict[str, Any] = {}
    required: list[str] = []
    path_params: set[str] = set()
    query_params: set[str] = set()
    body_params: set[str] = set()

    # Path-level + operation-level parameters (path & query only; header/cookie
    # params are out of scope for the model-facing schema).
    for param in [*shared_params, *op.get("parameters", [])]:
        loc = param.get("in")
        pname = param.get("name")
        if not pname or loc not in ("path", "query"):
            continue
        prop = dict(param.get("schema") or {})
        if param.get("description"):
            prop.setdefault("description", param["description"])
        properties[pname] = prop
        if loc == "path":
            path_params.add(pname)
        else:
            query_params.add(pname)
        # Path params are always required; query params honor their own flag.
        if loc == "path" or param.get("required"):
            if pname not in required:
                required.append(pname)

    # Merge the JSON requestBody's object properties as flat top-level args.
    body_schema = _json_body_schema(op.get("requestBody"))
    if body_schema:
        for pname, prop in (body_schema.get("properties") or {}).items():
            properties[pname] = prop
            body_params.add(pname)
        for pname in body_schema.get("required", []):
            if pname not in required:
                required.append(pname)

    return OpenAPITool(
        name=op_id,
        description=_description(op),
        method=method,
        path=path,
        base_url=base_url,
        parameters=properties,
        required=required,
        path_params=path_params,
        query_params=query_params,
        body_params=body_params,
        headers=headers,
        client=client,
    )


def _json_body_schema(request_body: Any) -> dict[str, Any] | None:
    """Pull the ``application/json`` schema out of a requestBody, if any."""
    if not isinstance(request_body, dict):
        return None
    content = request_body.get("content") or {}
    media = content.get("application/json")
    if not isinstance(media, dict):
        return None
    schema = media.get("schema")
    return schema if isinstance(schema, dict) else None


def _description(op: dict[str, Any]) -> str:
    """Combine ``summary`` and ``description`` into the model-facing blurb."""
    parts = [op.get("summary", ""), op.get("description", "")]
    return "\n\n".join(p.strip() for p in parts if p and p.strip())


def _load_spec(spec: dict[str, Any] | str) -> dict[str, Any]:
    """Coerce a dict / JSON string / YAML string into a parsed dict."""
    if isinstance(spec, dict):
        return spec
    if not isinstance(spec, str):
        raise TypeError(f"spec must be a dict or str, got {type(spec).__name__}")
    try:
        return json.loads(spec)
    except (ValueError, json.JSONDecodeError):
        pass
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - depends on env
        raise ValueError(
            "spec string is not valid JSON and pyyaml is not installed "
            "(`pip install pyyaml`) to parse it as YAML"
        ) from exc
    loaded = yaml.safe_load(spec)
    if not isinstance(loaded, dict):
        raise ValueError("OpenAPI spec did not parse to a mapping")
    return loaded


def _default_base_url(doc: dict[str, Any]) -> str:
    """Return ``servers[0].url`` from the spec, or empty if absent."""
    servers = doc.get("servers") or []
    if servers and isinstance(servers[0], dict):
        return servers[0].get("url", "")
    return ""


def _slug(method: str, path: str) -> str:
    """Build a fallback tool name from method + path when no operationId.

    e.g. ``GET /pets/{petId}`` -> ``get_pets_petId``. Strips braces and slashes
    so the name is a valid function identifier the model can call.
    """
    cleaned = re.sub(r"[{}]", "", path)
    parts = [p for p in cleaned.split("/") if p]
    return "_".join([method.lower(), *parts])


def _quote(value: Any) -> str:
    """URL-encode a single path-segment value (so ``a/b`` doesn't add a segment)."""
    from urllib.parse import quote

    return quote(str(value), safe="")
