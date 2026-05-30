"""Component registry — the extensibility backbone.

Every swappable concern in YAAB (models, tools, session/memory/artifact
backends, checkpointers, guardrails, embedders, compliance mappers, plugins) is
a *named component*. Third-party developers add features two ways:

1. **In-process** — call :func:`register` (or use the ``@component`` decorator)
   to add a factory under a ``(kind, name)`` key.
2. **Out-of-process** — ship a package that advertises an entry point in the
   matching ``yaab.<kind>s`` group; it is discovered lazily on first lookup.

Nothing in the core imports plugins eagerly, so a broken third-party component
never breaks ``import yaab``.

    from yaab.extensions import register, get, available

    @register("embedder", "myco")
    def _make(**kw):
        return MyEmbedder(**kw)

    embedder = get("embedder", "myco")
"""

from __future__ import annotations

from typing import Any, Callable, Iterable

Factory = Callable[..., Any]

# kind -> name -> factory
_REGISTRY: dict[str, dict[str, Factory]] = {}

# kind -> entry-point group name (for out-of-process discovery)
_ENTRY_POINT_GROUPS: dict[str, str] = {
    "model": "yaab.models",
    "tool": "yaab.tools",
    "session": "yaab.sessions",
    "memory": "yaab.memory",
    "artifact": "yaab.artifacts",
    "checkpointer": "yaab.checkpointers",
    "guardrail": "yaab.guardrails",
    "embedder": "yaab.embedders",
    "vectorstore": "yaab.vectorstores",
    "reranker": "yaab.rerankers",
    "plugin": "yaab.plugins",
    "compliance": "yaab.compliance",
    "skill": "yaab.skills",
}

_discovered: set[str] = set()


class ComponentError(KeyError):
    """Raised when a requested component is not registered."""


def register(kind: str, name: str, factory: Factory | None = None) -> Any:
    """Register a component factory. Usable as a call or a decorator.

        register("model", "echo", EchoModel)
        @register("model", "echo")
        def make(**kw): ...
    """

    def _do(f: Factory) -> Factory:
        _REGISTRY.setdefault(kind, {})[name] = f
        return f

    return _do(factory) if factory is not None else _do


def component(kind: str, name: str) -> Callable[[Factory], Factory]:
    """Decorator form of :func:`register`."""
    return register(kind, name)


def _discover(kind: str) -> None:
    """Lazily load entry-point components for ``kind`` (once)."""
    if kind in _discovered:
        return
    _discovered.add(kind)
    group = _ENTRY_POINT_GROUPS.get(kind)
    if not group:
        return
    try:
        from importlib.metadata import entry_points

        for ep in entry_points(group=group):
            if ep.name in _REGISTRY.get(kind, {}):
                continue
            try:
                _REGISTRY.setdefault(kind, {})[ep.name] = _as_factory(ep.load())
            except Exception:  # noqa: BLE001 - a broken plugin must not break discovery
                continue
    except Exception:  # noqa: BLE001
        pass


def _as_factory(obj: Any) -> Factory:
    """Wrap a class/callable so ``get`` always returns an instance."""
    if callable(obj):
        return obj
    return lambda **_: obj


def get(kind: str, name: str, /, **kwargs: Any) -> Any:
    """Instantiate the named component, passing ``kwargs`` to its factory."""
    _discover(kind)
    table = _REGISTRY.get(kind, {})
    if name not in table:
        raise ComponentError(
            f"no '{kind}' component named '{name}'. Available: {sorted(table)}"
        )
    return table[name](**kwargs)


def available(kind: str) -> list[str]:
    """List registered component names for ``kind`` (including entry points)."""
    _discover(kind)
    return sorted(_REGISTRY.get(kind, {}))


def kinds() -> Iterable[str]:
    """All known component kinds."""
    return sorted(set(_REGISTRY) | set(_ENTRY_POINT_GROUPS))


__all__ = ["register", "component", "get", "available", "kinds", "ComponentError"]
