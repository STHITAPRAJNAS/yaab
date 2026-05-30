"""Declarative agent configuration — build agents from YAML/dict.

Defining an agent as data (not code) makes it an auditable artifact ops teams
and non-coders can review and diff — and it pairs naturally with the governance
registry. ``yaml`` is imported lazily; :func:`agent_from_dict` works with no
extra dependency.

Schema (all keys optional except ``name``)::

    name: support-bot
    model: openai/gpt-4o
    instructions: You are a helpful support agent.
    registry_id: support-bot
    max_steps: 8
    tool_choice: auto
    tools: [calculator, current_time, http_get]   # built-in tool names
    skills: []                                     # registered skill names
    output_type: str

Tools/skills reference built-ins by name or components registered via
:mod:`yaab.extensions`; unknown names raise so typos fail loudly.
"""

from __future__ import annotations

from typing import Any

from .agent import Agent


def _resolve_tools(names: list[str]) -> list[Any]:
    from .tools import builtin

    resolved: list[Any] = []
    builtin_map = {
        "calculator": builtin.calculator,
        "current_time": builtin.current_time,
        "http_get": builtin.http_get,
        "web_search": builtin.web_search,
        "python_exec": builtin.python_exec,
    }
    for n in names:
        if n in builtin_map:
            resolved.append(builtin_map[n])
            continue
        # Fall back to a registered "tool" component.
        from .extensions import get

        try:
            resolved.append(get("tool", n))
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"unknown tool in config: {n!r}") from exc
    return resolved


def _resolve_skills(names: list[str]) -> list[Any]:
    from .skills import load_skills

    available = load_skills()
    out = []
    for n in names:
        if n not in available:
            raise ValueError(f"unknown skill in config: {n!r}")
        out.append(available[n])
    return out


def agent_from_dict(config: dict[str, Any]) -> Agent:
    """Build an :class:`Agent` from a config dict."""
    if "name" not in config:
        raise ValueError("agent config requires a 'name'")
    cfg = dict(config)
    name = cfg.pop("name")

    tools = _resolve_tools(cfg.pop("tools", []) or [])
    skills = _resolve_skills(cfg.pop("skills", []) or [])

    # output_type is referenced by name; only "str" is supported declaratively.
    output_type_name = cfg.pop("output_type", "str")
    output_type = str if output_type_name in ("str", None) else str

    return Agent(
        name,
        tools=tools,
        skills=skills,
        output_type=output_type,
        **{k: v for k, v in cfg.items() if k in _AGENT_KEYS},
    )


_AGENT_KEYS = {
    "model",
    "instructions",
    "registry_id",
    "max_steps",
    "output_retries",
    "tool_choice",
    "instrument",
}


def agent_from_yaml(path_or_text: str) -> Agent:
    """Build an agent from a YAML file path or a YAML string."""
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


__all__ = ["agent_from_dict", "agent_from_yaml"]
