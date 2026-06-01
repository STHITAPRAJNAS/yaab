"""Dynamic model routing — pick a model per request.

A :class:`ModelRouter` is itself a :class:`ModelProvider`: it picks one of
several downstream models *per request* via a classifier, then delegates. This
is a cost/quality lever — cheap models for simple turns, capable models when
the prompt is large or carries tools — but
expressed as a plain provider so it drops into ``Agent(model=...)`` anywhere a
model is accepted, including inside fallback chains.

The classifier is either a built-in name (``"length"``) or any callable
(sync or async) taking ``(messages, tools)`` and returning a route key. String
route specs (e.g. ``"openai/gpt-4o"``) are resolved lazily through
:func:`yaab.models.resolve_model` only when first chosen, and the resolved
provider is cached so repeated routing does not rebuild it.
"""

from __future__ import annotations

import inspect
from collections.abc import AsyncIterator, Callable
from typing import Any

from ..types import Message
from .base import ModelProvider, ModelResponse, StreamChunk

#: A classifier sees the request and names a route. It may be sync or async.
Classifier = Callable[[list[Message], "list[dict[str, Any]] | None"], Any]


def _total_prompt_chars(messages: list[Message]) -> int:
    """Approximate prompt size as total content characters.

    A deliberately cheap proxy for "how big/complex is this turn" — no tokenizer
    needed, and good enough to separate trivial turns from large ones.
    """
    return sum(len(m.content or "") for m in messages)


def length_classifier(complexity_threshold: int) -> Classifier:
    """Build the built-in length/complexity classifier.

    Routes to ``"complex"`` when the prompt exceeds ``complexity_threshold``
    characters OR any tools are present (tool use generally wants a more capable
    model), else ``"simple"``. These two keys are the convention; a router whose
    routes are named otherwise should supply a custom classifier or rely on the
    default-route fallback.
    """

    def classify(messages: list[Message], tools: list[dict[str, Any]] | None) -> str:
        if tools:
            return "complex"
        if _total_prompt_chars(messages) > complexity_threshold:
            return "complex"
        return "simple"

    return classify


#: Built-in classifiers addressable by name.
_BUILTIN: dict[str, Callable[[int], Classifier]] = {"length": length_classifier}


class ModelRouter:
    """A :class:`ModelProvider` that dispatches each request to one of ``routes``.

    Parameters
    ----------
    routes:
        Map of route key -> model. Values may be a :class:`ModelProvider`
        instance or a string spec resolved lazily via
        :func:`yaab.models.resolve_model`.
    classifier:
        A built-in name (``"length"``), a callable ``(messages, tools) -> key``
        (sync or async), or ``None`` to use the default ``"length"`` classifier.
    default:
        Route key used when the classifier returns an unknown key. Defaults to
        the first route key, so routing always resolves to a real model.
    complexity_threshold:
        Char threshold for the built-in ``"length"`` classifier.
    """

    def __init__(
        self,
        routes: dict[str, str | ModelProvider],
        *,
        classifier: Classifier | str | None = None,
        default: str | None = None,
        complexity_threshold: int = 2000,
    ) -> None:
        if not routes:
            raise ValueError("ModelRouter requires at least one route")
        self.name = "router"
        self.routes = dict(routes)
        self.complexity_threshold = complexity_threshold
        # Default route: explicit, else the first declared key (insertion order).
        self.default = default if default is not None else next(iter(self.routes))
        self.classifier = self._resolve_classifier(classifier)
        #: The route key chosen on the most recent call (observability/tests).
        self.last_route: str | None = None
        #: Cache of resolved providers for string route specs.
        self._resolved: dict[str, ModelProvider] = {}

    def _resolve_classifier(self, classifier: Classifier | str | None) -> Classifier:
        if classifier is None:
            return length_classifier(self.complexity_threshold)
        if isinstance(classifier, str):
            try:
                builder = _BUILTIN[classifier]
            except KeyError:
                raise ValueError(
                    f"unknown built-in classifier {classifier!r}; available: {sorted(_BUILTIN)}"
                ) from None
            return builder(self.complexity_threshold)
        return classifier

    async def _classify(self, messages: list[Message], tools: list[dict[str, Any]] | None) -> str:
        """Run the classifier (awaiting it if async) and snap to a valid route."""
        key = self.classifier(messages, tools)
        if inspect.isawaitable(key):
            key = await key
        if key not in self.routes:
            key = self.default
        self.last_route = key
        return key

    def _provider_for(self, key: str) -> ModelProvider:
        """Resolve (and cache) the provider for route ``key``.

        String specs are coerced through :func:`resolve_model` lazily, so a
        route that is never chosen is never built (and never needs litellm).
        """
        target = self.routes[key]
        if isinstance(target, str):
            if key not in self._resolved:
                # Reference the module-level ``resolve_model`` (not a local import)
                # so tests can monkeypatch ``yaab.models.router.resolve_model``.
                self._resolved[key] = resolve_model(target)
            return self._resolved[key]
        return target

    async def complete(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        output_schema: dict[str, Any] | None = None,
        tool_choice: Any | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        key = await self._classify(messages, tools)
        provider = self._provider_for(key)
        return await provider.complete(
            messages,
            tools=tools,
            output_schema=output_schema,
            tool_choice=tool_choice,
            **kwargs,
        )

    async def stream(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        key = await self._classify(messages, tools)
        provider = self._provider_for(key)
        async for chunk in provider.stream(messages, tools=tools, **kwargs):
            yield chunk


def resolve_model(model: str | ModelProvider) -> ModelProvider:
    """Coerce a model spec into a :class:`ModelProvider`.

    A thin module-level indirection over :func:`yaab.models.resolve_model`. It is
    defined here (rather than imported at module top) to (a) avoid a circular
    import, since ``yaab.models.__init__`` imports this module before it defines
    ``resolve_model``, and (b) give tests a stable name to monkeypatch
    (``yaab.models.router.resolve_model``).
    """
    from . import resolve_model as _resolve

    return _resolve(model)


def _register() -> None:
    """Register the router as the ``("model", "router")`` component."""
    from ..extensions import register

    register("model", "router", lambda **kw: ModelRouter(**kw))


_register()
