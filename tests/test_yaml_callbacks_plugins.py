"""A declarative agent can wire callbacks and plugins by name.

``callbacks: {before_agent: name, after_agent: name}`` resolves registered
``callback`` components onto the agent's per-agent hooks (B4); ``plugins: [name]``
resolves registered ``plugin`` components onto the agent's Runner.
"""

from __future__ import annotations

import pytest

from yaab import register_component
from yaab.config import agent_from_dict
from yaab.plugins import Plugin
from yaab.testing import TestModel

_LOG: list[str] = []


@pytest.fixture(autouse=True)
def _clear():
    _LOG.clear()
    yield
    _LOG.clear()


class _RecorderPlugin(Plugin):
    async def before_run(self, ctx, agent, prompt):
        _LOG.append(f"plugin-before:{agent}")


def _before(ag, prompt):
    _LOG.append(f"cb-before:{ag.name}")


@pytest.mark.asyncio
async def test_yaml_callbacks_resolve_onto_agent():
    register_component("callback", "rec_before", lambda: _before)
    agent = agent_from_dict(
        {
            "name": "a",
            "model": TestModel("hi"),
            "callbacks": {"before_agent": "rec_before"},
        }
    )
    await agent.run("go")
    assert _LOG == ["cb-before:a"]


@pytest.mark.asyncio
async def test_yaml_plugins_resolve_onto_runner():
    register_component("plugin", "recorder", lambda: _RecorderPlugin())
    agent = agent_from_dict({"name": "b", "model": TestModel("hi"), "plugins": ["recorder"]})
    await agent.run("go")
    assert any(e.startswith("plugin-before:") for e in _LOG)


def test_unknown_callback_is_a_clear_error():
    with pytest.raises(ValueError, match="callback"):
        agent_from_dict(
            {"name": "a", "model": TestModel("x"), "callbacks": {"before_agent": "nope"}}
        )
