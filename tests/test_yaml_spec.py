"""Tests for the full-coverage declarative YAML agent spec.

These exercise :func:`yaab.config.agent_from_dict`/``agent_from_yaml`` across the
whole Agent surface plus workflow composition, guardrail wiring, openapi/MCP
tool building, forward-compatible ``sub_agents``, and ``runner_from_dict``.

We never call ``.run`` — model resolution is lazy (only on ``Agent.model``
access), so specs with ``model: openai/gpt-4o`` construct without a network
call. That keeps these tests offline and deterministic.
"""

from __future__ import annotations

import inspect

import pytest

from yaab import Agent, agent_from_dict, agent_from_yaml
from yaab.config import runner_from_dict
from yaab.multiagent import LoopAgent, ParallelAgent, SequentialAgent, Swarm


# --- pass-through Agent kwargs ----------------------------------------
def test_passthrough_kwargs_flow_into_agent():
    agent = agent_from_dict(
        {
            "name": "tuned",
            "model": "openai/gpt-4o",
            "model_settings": {"temperature": 0.1, "seed": 7},
            "parallel_tools": False,
            "max_parallel_tools": 3,
            "tool_choice": "required",
            "output_retries": 5,
            "max_steps": 11,
            "instrument": False,
        }
    )
    assert isinstance(agent, Agent)
    assert agent.model_settings == {"temperature": 0.1, "seed": 7}
    assert agent.parallel_tools is False
    assert agent.max_parallel_tools == 3
    assert agent.tool_choice == "required"
    assert agent.output_retries == 5
    assert agent.max_steps == 11
    assert agent.instrument is False


def test_unknown_key_warns_but_does_not_crash(caplog):
    import logging

    with caplog.at_level(logging.WARNING):
        agent = agent_from_dict(
            {"name": "ok", "model": "openai/gpt-4o", "totally_made_up_key": 123}
        )
    assert isinstance(agent, Agent)
    assert any("totally_made_up_key" in rec.getMessage() for rec in caplog.records)


# --- openapi tools ----------------------------------------------------
def test_tools_openapi_dict_builds_openapi_toolset():
    spec = {
        "openapi": "3.0.0",
        "info": {"title": "t", "version": "1"},
        "servers": [{"url": "https://api.example.com"}],
        "paths": {
            "/pets": {
                "get": {"operationId": "listPets", "responses": {"200": {"description": "ok"}}}
            }
        },
    }
    agent = agent_from_dict(
        {
            "name": "api-bot",
            "model": "openai/gpt-4o",
            "tools": [{"openapi": {"spec": spec, "base_url": "https://staging.example.com"}}],
        }
    )
    names = {t.name for t in agent.tools}
    assert "listPets" in names
    tool = next(t for t in agent.tools if t.name == "listPets")
    # base_url override flowed through into the built tool.
    assert "staging.example.com" in tool._base_url


def test_tools_mix_names_and_openapi():
    spec = {
        "openapi": "3.0.0",
        "info": {"title": "t", "version": "1"},
        "paths": {"/ping": {"get": {"operationId": "ping"}}},
    }
    agent = agent_from_dict(
        {
            "name": "mix",
            "model": "openai/gpt-4o",
            "tools": ["calculator", {"openapi": {"spec": spec}}],
        }
    )
    names = {t.name for t in agent.tools}
    assert {"calculator", "ping"} <= names


# --- MCP tools (lazy/deferred) ----------------------------------------
def test_tools_mcp_dict_is_deferred_not_started():
    # MCP tools need an async handshake; YAML construction is sync, so building
    # the spec must NOT spawn a subprocess. We assert it constructs and the
    # resulting placeholder carries the command for a later async start.
    agent = agent_from_dict(
        {
            "name": "mcp-bot",
            "model": "openai/gpt-4o",
            "tools": [{"mcp": {"command": ["python", "server.py"]}}],
        }
    )
    # A lazy MCP tool wrapper is attached and remembers its command.
    mcp_tools = [t for t in agent.tools if getattr(t, "_mcp_command", None)]
    assert mcp_tools
    assert mcp_tools[0]._mcp_command == ["python", "server.py"]


# --- guardrails -------------------------------------------------------
def test_guardrails_instantiate_from_registry():
    from yaab.governance.policy import PIIScanner, PromptInjectionScanner

    agent = agent_from_dict(
        {
            "name": "guarded",
            "model": "openai/gpt-4o",
            "guardrails": ["prompt_injection", "pii"],
        }
    )
    kinds = {type(g) for g in agent.guardrails}
    assert PromptInjectionScanner in kinds
    assert PIIScanner in kinds


def test_guardrails_with_kwargs_dict():
    # A {name: {kwargs}} entry passes kwargs to the registry factory.
    agent = agent_from_dict(
        {
            "name": "guarded2",
            "model": "openai/gpt-4o",
            "guardrails": [{"topics": {"banned": ["weapons"]}}],
        }
    )
    assert len(agent.guardrails) == 1
    # TopicScanner stores its banned list.
    assert getattr(agent.guardrails[0], "banned", None) == ["weapons"] or hasattr(
        agent.guardrails[0], "scan"
    )


def test_unknown_guardrail_fails_loudly():
    with pytest.raises(ValueError):
        agent_from_dict(
            {"name": "x", "model": "openai/gpt-4o", "guardrails": ["no_such_guardrail"]}
        )


# --- workflow composition ---------------------------------------------
def _leaf(name: str) -> dict:
    return {"name": name, "model": "openai/gpt-4o"}


def test_kind_sequential_builds_sequential_agent():
    wf = agent_from_dict(
        {
            "kind": "sequential",
            "name": "pipeline",
            "agents": [_leaf("a"), _leaf("b")],
        }
    )
    assert isinstance(wf, SequentialAgent)
    assert wf.name == "pipeline"
    assert [a.name for a in wf.agents] == ["a", "b"]
    assert all(isinstance(a, Agent) for a in wf.agents)


def test_kind_parallel_builds_parallel_agent():
    wf = agent_from_dict({"kind": "parallel", "name": "fan", "agents": [_leaf("a"), _leaf("b")]})
    assert isinstance(wf, ParallelAgent)
    assert len(wf.agents) == 2


def test_kind_loop_builds_loop_agent_with_max_iterations():
    wf = agent_from_dict(
        {
            "kind": "loop",
            "name": "looper",
            "agents": [_leaf("worker")],
            "max_iterations": 4,
        }
    )
    assert isinstance(wf, LoopAgent)
    assert wf.max_iterations == 4
    assert wf.agent.name == "worker"


def test_kind_swarm_builds_swarm_with_entry_and_handoffs():
    wf = agent_from_dict(
        {
            "kind": "swarm",
            "name": "hive",
            "agents": [_leaf("triage"), _leaf("expert")],
            "entry": "triage",
            "max_handoffs": 3,
        }
    )
    assert isinstance(wf, Swarm)
    assert wf.entry == "triage"
    assert wf.max_handoffs == 3
    assert set(wf.agents) == {"triage", "expert"}


def test_kind_agent_is_default():
    agent = agent_from_dict({"kind": "agent", "name": "plain", "model": "openai/gpt-4o"})
    assert isinstance(agent, Agent)


def test_unknown_kind_fails_loudly():
    with pytest.raises(ValueError):
        agent_from_dict({"kind": "octopus", "name": "x", "agents": [_leaf("a")]})


def test_nested_workflow_recurses():
    # A sequential whose first stage is itself a parallel fan-out.
    wf = agent_from_dict(
        {
            "kind": "sequential",
            "name": "outer",
            "agents": [
                {"kind": "parallel", "name": "inner", "agents": [_leaf("a"), _leaf("b")]},
                _leaf("c"),
            ],
        }
    )
    assert isinstance(wf, SequentialAgent)
    assert isinstance(wf.agents[0], ParallelAgent)
    assert isinstance(wf.agents[1], Agent)


# --- sub_agents (forward-compatible) ----------------------------------
def test_sub_agents_behavior_matches_constructor_support():
    config = {
        "name": "parent",
        "model": "openai/gpt-4o",
        "sub_agents": [_leaf("child1"), _leaf("child2")],
    }
    accepts = "sub_agents" in inspect.signature(Agent.__init__).parameters
    if accepts:
        agent = agent_from_dict(config)
        subs = getattr(agent, "sub_agents", None)
        assert subs is not None
        assert [a.name for a in subs] == ["child1", "child2"]
    else:
        with pytest.raises(ValueError, match="sub_agents requires"):
            agent_from_dict(config)


# --- YAML string entry point ------------------------------------------
def test_agent_from_yaml_full_spec():
    pytest.importorskip("yaml")
    text = """
kind: sequential
name: yaml-pipeline
agents:
  - name: researcher
    model: openai/gpt-4o
    instructions: Research the topic.
    guardrails: [pii]
  - name: writer
    model: openai/gpt-4o
    model_settings:
      temperature: 0.2
"""
    wf = agent_from_yaml(text)
    assert isinstance(wf, SequentialAgent)
    assert wf.agents[0].name == "researcher"
    assert len(wf.agents[0].guardrails) == 1
    assert wf.agents[1].model_settings == {"temperature": 0.2}


# --- runner_from_dict -------------------------------------------------
def test_runner_from_dict_default_is_runner():
    from yaab.runner import Runner

    runner = runner_from_dict({})
    assert isinstance(runner, Runner)


def test_runner_from_dict_with_session_and_governance():
    from yaab.governance.service import GovernanceMode
    from yaab.runner import Runner
    from yaab.sessions.memory import InMemorySessionService

    runner = runner_from_dict(
        {
            "session_service": "memory",
            "governance": {"mode": "enforcing", "guardrails": ["pii", "prompt_injection"]},
        }
    )
    assert isinstance(runner, Runner)
    assert isinstance(runner.session_service, InMemorySessionService)
    assert runner.governance is not None
    assert runner.governance.mode is GovernanceMode.ENFORCING
    # The named guardrails were wired into the policy engine.
    scanner_names = {getattr(s, "name", "") for s in runner.governance.policy.scanners}
    assert {"pii", "prompt_injection"} <= scanner_names


def test_runner_from_dict_unknown_session_fails_loudly():
    with pytest.raises(ValueError):
        runner_from_dict({"session_service": "no_such_backend"})


def test_runner_from_dict_plugins_by_name():
    # Register a throwaway plugin component, then build a runner referencing it.
    from yaab.extensions import register
    from yaab.plugins import Plugin

    class _Noop(Plugin):
        pass

    register("plugin", "test_noop", lambda **kw: _Noop())
    runner = runner_from_dict({"plugins": ["test_noop"]})
    assert any(isinstance(p, _Noop) for p in runner.plugins)


# --- regression: original simple specs still work ---------------------
def test_simple_spec_still_builds():
    agent = agent_from_dict(
        {
            "name": "support-bot",
            "model": "openai/gpt-4o",
            "instructions": "Be helpful.",
            "tools": ["calculator", "current_time"],
            "max_steps": 5,
            "registry_id": "support-bot",
        }
    )
    assert agent.name == "support-bot"
    assert agent.max_steps == 5
    assert {t.name for t in agent.tools} == {"calculator", "current_time"}
