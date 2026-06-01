"""Tool protocol and typed function tools.

A :class:`Tool` exposes a JSON schema (for the model) and an async ``execute``
(for the runtime). :func:`tool` turns a plain typed Python function into a
tool: the parameter schema is generated from type hints, argument validation
is handled by Pydantic, and the description comes from the docstring — the same
ergonomics as Pydantic AI's ``@agent.tool``.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any, Protocol, get_type_hints, runtime_checkable

from pydantic import create_model

from ..exceptions import ToolError
from ..types import RunContext
from .auth import ToolAuth, ToolAuthRequired, as_headers

#: Parameter names FunctionTool fills with the resolved credential rather than
#: model-supplied arguments. They're hidden from the model-facing schema so the
#: model can't (and needn't) provide them.
_AUTH_HEADERS_PARAM = "auth_headers"
_CREDENTIAL_PARAM = "credential"


@runtime_checkable
class Tool(Protocol):
    """The pluggable tool interface."""

    name: str
    description: str

    def schema(self) -> dict[str, Any]:
        """Return the OpenAI-style function-calling schema."""
        ...

    async def execute(self, ctx: RunContext, **kwargs: Any) -> Any:
        """Run the tool with validated keyword arguments."""
        ...


class FunctionTool:
    """Wrap a typed Python function as a :class:`Tool`.

    The wrapped callable may optionally take a :class:`RunContext` as its first
    parameter (named ``ctx``); if present it is injected and excluded from the
    model-facing schema. All other parameters are validated against a Pydantic
    model derived from the signature before the function runs.
    """

    def __init__(
        self,
        fn: Callable[..., Any],
        *,
        name: str | None = None,
        description: str | None = None,
        timeout: float | None = None,
        auth: ToolAuth | None = None,
    ) -> None:
        self.fn = fn
        self.name = name or fn.__name__
        self.description = description or (inspect.getdoc(fn) or "").strip()
        #: Optional per-tool execution timeout (seconds); overrides the runner's
        #: ``default_tool_timeout``. ``None`` defers to the runner default.
        self.timeout = timeout
        #: Optional tool-level auth. When set, ``execute`` resolves a credential
        #: before calling ``fn`` and injects it (see :meth:`execute`).
        self.auth = auth
        self._is_async = inspect.iscoroutinefunction(fn)
        # Discover the auth-injection params up front so the schema can hide them
        # and ``execute`` knows where to route the resolved credential.
        params = inspect.signature(fn).parameters
        self._wants_auth_headers = _AUTH_HEADERS_PARAM in params
        self._wants_credential = _CREDENTIAL_PARAM in params
        self._takes_ctx, self._arg_model = self._build_model(fn)

    @staticmethod
    def _build_model(fn: Callable[..., Any]) -> tuple[bool, Any]:
        sig = inspect.signature(fn)
        try:
            hints = get_type_hints(fn)
        except Exception:  # noqa: BLE001 - tolerate un-resolvable annotations
            hints = {}
        fields: dict[str, Any] = {}
        takes_ctx = False
        for pname, param in sig.parameters.items():
            if pname == "ctx" or _is_run_context(hints.get(pname)):
                takes_ctx = True
                continue
            # Auth-injection params are filled by the framework, never the model,
            # so exclude them from the validated/model-facing schema.
            if pname in (_AUTH_HEADERS_PARAM, _CREDENTIAL_PARAM):
                continue
            if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                continue
            annotation = hints.get(pname, Any)
            default = param.default if param.default is not inspect.Parameter.empty else ...
            fields[pname] = (annotation, default)
        model = create_model(f"{fn.__name__}_Args", **fields)  # type: ignore[call-overload]
        return takes_ctx, model

    def schema(self) -> dict[str, Any]:
        params = self._arg_model.model_json_schema()
        params.pop("title", None)
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": params,
            },
        }

    async def execute(self, ctx: RunContext, **kwargs: Any) -> Any:
        try:
            validated = self._arg_model(**kwargs)
        except Exception as exc:  # noqa: BLE001 - surface as ToolError for retry
            raise ToolError(f"invalid arguments for tool '{self.name}': {exc}") from exc
        call_kwargs = validated.model_dump()
        if self._takes_ctx:
            call_kwargs = {"ctx": ctx, **call_kwargs}
        if self.auth is not None:
            # Resolve the credential first. A missing/unresolvable one becomes a
            # model-visible ``error:`` string (the agent loop turns tool results
            # into model input) instead of raising — so the model can tell the
            # user how to authorize and the run continues uninterrupted.
            try:
                cred = await self.auth.resolve(ctx, tool_name=self.name)
            except ToolAuthRequired as exc:
                return exc.as_model_error()
            if self._wants_auth_headers:
                call_kwargs[_AUTH_HEADERS_PARAM] = as_headers(cred)
            elif self._wants_credential:
                call_kwargs[_CREDENTIAL_PARAM] = cred
            else:
                # No injection param on the signature: stash it on ctx.state so a
                # function that reads its own credential can pick it up.
                ctx.state["__tool_credential__"] = cred
        result = self.fn(**call_kwargs)
        if inspect.isawaitable(result):
            result = await result
        return result


def _is_run_context(annotation: Any) -> bool:
    origin = getattr(annotation, "__origin__", None)
    return annotation is RunContext or origin is RunContext


def tool(
    fn: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    timeout: float | None = None,
    auth: ToolAuth | None = None,
) -> Any:
    """Decorator turning a typed function into a :class:`FunctionTool`.

    Usable bare (``@tool``) or parameterized (``@tool(name=..., timeout=...,
    auth=...)``). When ``auth`` is given, the framework resolves and injects a
    credential before each call — see :class:`FunctionTool` and :mod:`.auth`.
    """

    def wrap(func: Callable[..., Any]) -> FunctionTool:
        return FunctionTool(func, name=name, description=description, timeout=timeout, auth=auth)

    if fn is not None:
        return wrap(fn)
    return wrap


def coerce_tools(items: list[Any]) -> list[Tool]:
    """Coerce a mixed list of callables/tools into :class:`Tool` instances."""
    out: list[Tool] = []
    for item in items:
        if isinstance(item, FunctionTool) or (hasattr(item, "schema") and hasattr(item, "execute")):
            out.append(item)
        elif callable(item):
            out.append(FunctionTool(item))
        else:
            raise ToolError(f"cannot use {item!r} as a tool")
    return out
