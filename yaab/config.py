"""Declarative agent configuration — build agents from YAML/dict.

Defining an agent as data (not code) makes it an auditable artifact ops teams
and non-coders can review and diff — and it pairs naturally with the governance
registry. ``yaml`` is imported lazily; :func:`agent_from_dict` works with no
extra dependency.

This mirrors Google ADK's "Agent Config" YAML: a single document fully
describes an agent (or a *workflow* of agents) so the same artifact can be
linted, versioned, and promoted through environments without code.

Schema (all keys optional except ``name``)::

    kind: agent                 # agent (default) | sequential | parallel | loop | swarm
    name: support-bot
    model: openai/gpt-4o
    instructions: You are a helpful support agent.
    registry_id: support-bot
    max_steps: 8
    output_retries: 2
    tool_choice: auto
    parallel_tools: true
    max_parallel_tools: 0
    instrument: true
    model_settings: {temperature: 0.2, seed: 7}
    tools:                      # built-in names, registered tool components, or:
      - calculator
      - {openapi: {spec: ./petstore.yaml, base_url: https://api.example.com}}
      - {mcp: {command: [python, my_server.py]}}   # deferred (needs async start)
    skills: []
    guardrails: [pii, prompt_injection, {topics: {banned: [weapons]}}]
    output_type: str
    sub_agents: [ ...nested agent specs... ]        # if Agent supports it

Workflow kinds put nested agent specs under ``agents:`` (recursively built) plus
kind-specific keys (``loop`` → ``max_iterations``; ``swarm`` → ``entry``,
``max_handoffs``).

Tools/skills reference built-ins by name or components registered via
:mod:`yaab.extensions`; unknown names raise so typos fail loudly. Truly unknown
*top-level* keys are warned about (not silently dropped) and ignored.

:func:`runner_from_dict` builds a :class:`~yaab.runner.Runner` from the same
data style: ``session_service``/``memory_service``/``plugins`` are resolved by
registry name and ``governance`` wires a :class:`GovernanceService`.
"""

from __future__ import annotations

import inspect
import logging
from typing import Any

from .agent import Agent

logger = logging.getLogger(__name__)

# Composition / meta keys handled explicitly by the builder; everything else is
# forwarded to Agent(**kwargs) (filtered by the constructor's real signature).
_COMPOSITION_KEYS = frozenset(
    {"kind", "name", "tools", "skills", "guardrails", "output_type", "agents", "sub_agents"}
)


def _resolve_tools(specs: list[Any]) -> list[Any]:
    """Resolve a YAML ``tools:`` list into concrete tool objects.

    Entries may be:

    * a built-in tool name or a registered ``tool`` component name (``str``);
    * ``{openapi: {spec: ..., base_url: ..., headers: ..., operations: ...}}``
      which expands to one tool per OpenAPI operation;
    * ``{mcp: {command: [...]}}`` which is *deferred*: MCP needs an async
      handshake that YAML construction (sync) can't perform, so we attach a
      lazy placeholder tool that errors with a clear message if invoked before
      :meth:`LazyMCPTool.start` is awaited — keeping construction side-effect
      free and never spawning a subprocess at build time.
    """
    from .tools import builtin

    builtin_map = {
        "calculator": builtin.calculator,
        "current_time": builtin.current_time,
        "http_get": builtin.http_get,
        "web_search": builtin.web_search,
        "python_exec": builtin.python_exec,
    }
    resolved: list[Any] = []
    for spec in specs:
        if isinstance(spec, str):
            if spec in builtin_map:
                resolved.append(builtin_map[spec])
                continue
            from .extensions import get

            try:
                resolved.append(get("tool", spec))
            except Exception as exc:  # noqa: BLE001
                raise ValueError(f"unknown tool in config: {spec!r}") from exc
            continue
        if isinstance(spec, dict):
            if "openapi" in spec:
                resolved.extend(_build_openapi_tools(spec["openapi"]))
                continue
            if "mcp" in spec:
                resolved.append(_build_mcp_tool(spec["mcp"]))
                continue
        raise ValueError(f"unsupported tool spec in config: {spec!r}")
    return resolved


def _build_openapi_tools(cfg: dict[str, Any]) -> list[Any]:
    """Build an OpenAPI toolset from ``{spec: ..., base_url: ..., ...}``."""
    from .tools.openapi import openapi_toolset

    if not isinstance(cfg, dict) or "spec" not in cfg:
        raise ValueError("openapi tool spec requires a 'spec' (path/JSON/YAML/dict)")
    spec = _load_openapi_spec(cfg["spec"])
    return openapi_toolset(
        spec,
        base_url=cfg.get("base_url"),
        headers=cfg.get("headers"),
        operations=cfg.get("operations"),
    )


def _load_openapi_spec(spec: Any) -> Any:
    """Accept a parsed dict, an inline JSON/YAML string, or a file path."""
    if isinstance(spec, dict):
        return spec
    if isinstance(spec, str):
        # A path (no newline + file exists) is read; otherwise treat the string
        # as inline JSON/YAML and let openapi_toolset parse it.
        import os

        if "\n" not in spec and os.path.exists(spec):
            with open(spec) as fh:
                return fh.read()
        return spec
    raise ValueError(f"openapi spec must be a dict, str, or path; got {type(spec).__name__}")


class LazyMCPTool:
    """A placeholder for an MCP server's tools, deferred until an async start.

    MCP tool discovery requires connecting to the server (a subprocess + JSON-RPC
    handshake) — inherently async, while YAML/dict construction is synchronous.
    Rather than block construction or silently spawn a process, we attach this
    placeholder. Call :meth:`start` (awaitable) to perform the handshake and get
    the real toolset, then add those tools to the agent. If the model tries to
    invoke the placeholder before that, :meth:`execute` returns a clear error
    instead of crashing the run.
    """

    def __init__(self, command: list[str]) -> None:
        self._mcp_command = list(command)
        self.name = "mcp_pending"
        self.description = (
            "MCP toolset (deferred). Await LazyMCPTool.start() to connect and "
            "load the server's real tools before use."
        )

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {"type": "object", "properties": {}},
            },
        }

    async def start(self) -> list[Any]:
        """Connect to the MCP server and return its discovered tools."""
        from .tools.mcp_client import MCPClient

        client = MCPClient.stdio(self._mcp_command)
        await client.start()
        return await client.list_tools()

    async def execute(self, ctx: Any, **kwargs: Any) -> str:  # noqa: ARG002
        return (
            "error: MCP tools are not loaded. They require an async start; "
            "await LazyMCPTool.start() and add the returned tools to the agent."
        )


def _build_mcp_tool(cfg: dict[str, Any]) -> LazyMCPTool:
    if not isinstance(cfg, dict) or "command" not in cfg:
        raise ValueError("mcp tool spec requires a 'command' (list of argv)")
    command = cfg["command"]
    if not isinstance(command, list) or not command:
        raise ValueError("mcp 'command' must be a non-empty list of argv strings")
    return LazyMCPTool(command)


def _resolve_skills(names: list[str]) -> list[Any]:
    from .skills import load_skills

    available = load_skills()
    out = []
    for n in names:
        if n not in available:
            raise ValueError(f"unknown skill in config: {n!r}")
        out.append(available[n])
    return out


def _resolve_guardrails(specs: list[Any]) -> list[Any]:
    """Instantiate guardrail scanners from the component registry.

    Entries are either a registry name (``str``) or a ``{name: {kwargs}}`` dict
    whose kwargs are forwarded to the factory (e.g. ``{topics: {banned: [...]}}``).
    Unknown names raise so a typo fails loudly rather than silently disabling a
    guardrail.
    """
    from .extensions import ComponentError, get

    out: list[Any] = []
    for spec in specs:
        name: str
        kwargs: dict[str, Any]
        if isinstance(spec, str):
            name, kwargs = spec, {}
        elif isinstance(spec, dict) and len(spec) == 1:
            (name, raw_kwargs) = next(iter(spec.items()))
            kwargs = raw_kwargs or {}
            if not isinstance(kwargs, dict):
                raise ValueError(f"guardrail {name!r} kwargs must be a mapping, got {kwargs!r}")
        else:
            raise ValueError(f"unsupported guardrail spec in config: {spec!r}")
        try:
            out.append(get("guardrail", name, **kwargs))
        except ComponentError as exc:
            raise ValueError(f"unknown guardrail in config: {name!r}") from exc
    return out


def _agent_kwargs(cfg: dict[str, Any]) -> dict[str, Any]:
    """Forward remaining keys to Agent, filtered by its real signature.

    Replaces the old fixed allowlist: anything the :class:`Agent` constructor
    actually accepts is passed through (so new constructor params work without
    touching this file); truly unknown keys are warned about and dropped.
    """
    accepted = set(inspect.signature(Agent.__init__).parameters) - {"self", "name"}
    kwargs: dict[str, Any] = {}
    for key, value in cfg.items():
        if key in accepted:
            kwargs[key] = value
        else:
            logger.warning("ignoring unknown agent config key %r", key)
    return kwargs


def _build_leaf_agent(cfg: dict[str, Any]) -> Agent:
    """Build a plain :class:`Agent` (kind: agent) from a spec dict."""
    name = cfg["name"]
    tools = _resolve_tools(cfg.get("tools") or [])
    skills = _resolve_skills(cfg.get("skills") or [])
    guardrails = _resolve_guardrails(cfg.get("guardrails") or [])

    # output_type is referenced by name; only "str" is supported declaratively.
    output_type_name = cfg.get("output_type", "str")
    output_type = str if output_type_name in ("str", None) else str

    # Everything not handled specially is forwarded to the constructor.
    rest = {k: v for k, v in cfg.items() if k not in _COMPOSITION_KEYS}
    kwargs = _agent_kwargs(rest)

    # sub_agents is forward-compatible: only pass it if the constructor accepts
    # it (a concurrent change may add it). Build nested specs recursively.
    if "sub_agents" in cfg:
        if "sub_agents" not in inspect.signature(Agent.__init__).parameters:
            raise ValueError(
                "sub_agents requires an Agent constructor that accepts a 'sub_agents' "
                "parameter; this build of yaab.Agent does not support it yet"
            )
        kwargs["sub_agents"] = [agent_from_dict(s) for s in cfg["sub_agents"]]

    return Agent(
        name,
        tools=tools,
        skills=skills,
        guardrails=guardrails,
        output_type=output_type,
        **kwargs,
    )


def _build_workflow(kind: str, cfg: dict[str, Any]) -> Any:
    """Build a multiagent workflow (sequential/parallel/loop/swarm)."""
    name = cfg["name"]
    agent_specs = cfg.get("agents")
    if not agent_specs:
        raise ValueError(f"workflow kind {kind!r} requires a non-empty 'agents:' list")
    agents = [agent_from_dict(s) for s in agent_specs]

    if kind == "sequential":
        from .multiagent import SequentialAgent

        return SequentialAgent(name, agents, pipe_output=cfg.get("pipe_output", True))
    if kind == "parallel":
        from .multiagent import ParallelAgent

        return ParallelAgent(name, agents)
    if kind == "loop":
        from .multiagent import LoopAgent

        # LoopAgent loops over a single agent; take the first of the list.
        return LoopAgent(name, agents[0], max_iterations=cfg.get("max_iterations", 5))
    if kind == "swarm":
        from .multiagent import Swarm

        return Swarm(
            name,
            agents,
            entry=cfg.get("entry"),
            max_handoffs=cfg.get("max_handoffs", 6),
        )
    raise ValueError(  # pragma: no cover - guarded by agent_from_dict
        f"unknown workflow kind: {kind!r}"
    )


_WORKFLOW_KINDS = frozenset({"sequential", "parallel", "loop", "swarm"})


def agent_from_dict(config: dict[str, Any]) -> Any:
    """Build an :class:`Agent` (or a workflow agent) from a config dict.

    The ``kind`` key selects the shape: ``agent`` (default) returns an
    :class:`Agent`; ``sequential``/``parallel``/``loop``/``swarm`` return the
    corresponding :mod:`yaab.multiagent` class wrapping nested specs.
    """
    if not isinstance(config, dict):
        raise ValueError("agent config must be a mapping")
    if "name" not in config:
        raise ValueError("agent config requires a 'name'")

    kind = config.get("kind", "agent")
    if kind in _WORKFLOW_KINDS:
        return _build_workflow(kind, config)
    if kind != "agent":
        raise ValueError(
            f"unknown config kind: {kind!r} (expected 'agent' or one of {sorted(_WORKFLOW_KINDS)})"
        )
    return _build_leaf_agent(config)


def agent_from_yaml(path_or_text: str) -> Any:
    """Build an agent (or workflow) from a YAML file path or a YAML string."""
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - optional extra
        raise RuntimeError("PyYAML is required: `pip install pyyaml`.") from exc

    import os

    text = path_or_text
    if "\n" not in path_or_text and os.path.exists(path_or_text):
        with open(path_or_text) as fh:
            text = fh.read()
    config = yaml.safe_load(text)
    if not isinstance(config, dict):
        raise ValueError("YAML config must be a mapping")
    return agent_from_dict(config)


def runner_from_dict(config: dict[str, Any]) -> Any:
    """Build a :class:`~yaab.runner.Runner` from a config dict.

    Resolves swappable services by registry name and wires governance::

        session_service: memory                # registry 'session' component
        memory_service: {name: mem0, ...}       # registry 'memory' component
        plugins: [audit, rate_limit]            # registry 'plugin' components
        governance:
          mode: enforcing
          guardrails: [pii, prompt_injection]   # scanners added to the policy

    A bare string is shorthand for ``{name: <string>}`` with no kwargs. Unknown
    component names raise ``ValueError`` so a typo fails loudly.
    """
    from .runner import Runner

    session_service = _resolve_service("session", config.get("session_service"))
    memory_service = _resolve_service("memory", config.get("memory_service"))
    artifact_service = _resolve_service("artifact", config.get("artifact_service"))
    plugins = [_resolve_named("plugin", p) for p in (config.get("plugins") or [])]
    governance = _build_governance(config.get("governance"))

    kwargs: dict[str, Any] = {"plugins": plugins, "governance": governance}
    if session_service is not None:
        kwargs["session_service"] = session_service
    if memory_service is not None:
        kwargs["memory_service"] = memory_service
    if artifact_service is not None:
        kwargs["artifact_service"] = artifact_service
    return Runner(**kwargs)


def _resolve_service(kind: str, spec: Any) -> Any:
    """Resolve an optional service spec (``None`` → ``None``)."""
    if spec is None:
        return None
    return _resolve_named(kind, spec)


def _resolve_named(kind: str, spec: Any) -> Any:
    """Resolve a registry component from a name or ``{name: {kwargs}}`` dict."""
    from .extensions import ComponentError, get

    if isinstance(spec, str):
        name, kwargs = spec, {}
    elif isinstance(spec, dict):
        # Either {name: <str>, ...kwargs} or {<name>: {kwargs}}.
        if "name" in spec:
            kwargs = {k: v for k, v in spec.items() if k != "name"}
            name = spec["name"]
        elif len(spec) == 1:
            (name, kwargs) = next(iter(spec.items()))
            kwargs = kwargs or {}
        else:
            raise ValueError(f"ambiguous {kind} component spec: {spec!r}")
    else:
        raise ValueError(f"{kind} component spec must be a str or mapping, got {spec!r}")
    try:
        return get(kind, name, **kwargs)
    except ComponentError as exc:
        raise ValueError(f"unknown {kind} component in config: {name!r}") from exc


def _build_governance(spec: Any) -> Any:
    """Build a :class:`GovernanceService` from a ``governance:`` block."""
    if spec is None:
        return None
    from .governance.policy import PolicyEngine
    from .governance.service import GovernanceMode, GovernanceService

    if not isinstance(spec, dict):
        raise ValueError(f"governance config must be a mapping, got {spec!r}")
    mode = spec.get("mode", "observe")
    try:
        mode_enum = GovernanceMode(mode)
    except ValueError as exc:
        raise ValueError(
            f"unknown governance mode: {mode!r} "
            f"(expected one of {[m.value for m in GovernanceMode]})"
        ) from exc

    guardrails = _resolve_guardrails(spec.get("guardrails") or [])
    policy = PolicyEngine(scanners=guardrails) if guardrails else None
    return GovernanceService(mode=mode_enum, policy=policy)


__all__ = ["agent_from_dict", "agent_from_yaml", "runner_from_dict", "LazyMCPTool"]
