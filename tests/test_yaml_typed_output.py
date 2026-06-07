"""A YAML/declarative agent can emit a structured (typed) output, not just str.

``output_type`` in a spec is resolved by name: built-in scalars (``str``/``int``/
``float``/``bool``) plus any Pydantic model registered under the ``output_type``
component kind. This closes the gap where declarative agents were hard-pinned to
``str`` output.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from yaab import register_component
from yaab.config import agent_from_dict
from yaab.testing import TestModel


class Ticket(BaseModel):
    title: str
    priority: int


def test_yaml_output_type_defaults_to_str():
    agent = agent_from_dict({"name": "a", "model": TestModel("hi")})
    assert agent.output_type is str


def test_yaml_output_type_builtin_scalar():
    agent = agent_from_dict({"name": "a", "model": TestModel("7"), "output_type": "int"})
    assert agent.output_type is int


def test_yaml_output_type_registered_model():
    register_component("output_type", "Ticket", lambda: Ticket)
    agent = agent_from_dict(
        {"name": "a", "model": TestModel('{"title": "x", "priority": 2}'), "output_type": "Ticket"}
    )
    assert agent.output_type is Ticket


@pytest.mark.asyncio
async def test_yaml_typed_agent_actually_produces_the_model():
    register_component("output_type", "Ticket", lambda: Ticket)
    agent = agent_from_dict(
        {
            "name": "tk",
            "model": TestModel('{"title": "Reset password", "priority": 1}'),
            "output_type": "Ticket",
        }
    )
    result = await agent.run("make a ticket")
    assert isinstance(result.output, Ticket)
    assert result.output.title == "Reset password"
    assert result.output.priority == 1


def test_yaml_unknown_output_type_is_a_clear_error():
    with pytest.raises(ValueError, match="output_type"):
        agent_from_dict({"name": "a", "model": TestModel("x"), "output_type": "NoSuchType"})
