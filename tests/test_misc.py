"""Tests for memory, prompts, skills, optimize, plugins, serve, and CLI."""

from __future__ import annotations

import pytest

from yaab import Agent
from yaab.memory import InMemoryVectorMemory
from yaab.models.test_model import TestModel
from yaab.optimize import ChainOfThought, Predict, Signature
from yaab.plugins.builtins import AuditPlugin, CachingPlugin, CostBudgetPlugin
from yaab.prompts import PromptRegistry
from yaab.skills import Skill


@pytest.mark.asyncio
async def test_vector_memory_retrieval():
    mem = InMemoryVectorMemory()
    await mem.add("Paris is the capital of France")
    await mem.add("Bananas are yellow fruit")
    hits = await mem.search("What is the capital of France?", k=1)
    assert hits
    assert "Paris" in hits[0][0].text


def test_prompt_versioning():
    reg = PromptRegistry()
    reg.register("greet", "Hello {name} (v1)")
    reg.register("greet", "Hi {name} (v2)")
    pt = reg.get("greet")
    assert pt.active == 2
    assert reg.render("greet", name="Bob") == "Hi Bob (v2)"
    assert pt.render(version=1, name="Bob") == "Hello Bob (v1)"
    # Distinct content hashes per version.
    assert pt.get(1).hash != pt.get(2).hash


def test_skill_attaches_tools_and_instructions():
    from yaab import tool

    @tool
    def search(q: str) -> str:
        """Search."""
        return "result"

    skill = Skill("research", instructions="Always search first.", tools=[search],
                  permissions=["net:read"])
    agent = Agent("a", model=TestModel("ok"), instructions="Base.", skills=[skill])
    assert any(t.name == "search" for t in agent.tools)
    assert "Always search first." in agent.instructions
    assert "net:read" in agent.permissions


def test_signature_parse_and_render():
    sig = Signature.parse("question -> answer", instructions="Answer well.")
    assert [f.name for f in sig.inputs] == ["question"]
    assert [f.name for f in sig.outputs] == ["answer"]
    prompt = sig.render_prompt({"question": "2+2?"})
    assert "2+2?" in prompt
    assert sig.parse_output("answer: 4") == {"answer": "4"}


def test_chain_of_thought_adds_reasoning():
    cot = ChainOfThought("q -> a", model=TestModel("x"))
    assert any(f.name == "reasoning" for f in cot.signature.outputs)


@pytest.mark.asyncio
async def test_bootstrap_optimizer_freezes_artifact():
    from yaab.governance.eval import Case
    from yaab.optimize import BootstrapFewShot

    module = Predict("input -> output", model=TestModel("output: yes"))
    train = [Case(name="c1", inputs={"input": "q"}, expected="yes")]

    def metric(case, pred):
        return 1.0 if pred.get("output") == case.expected else 0.0

    artifact = await BootstrapFewShot().compile(module, train, metric)
    assert artifact.optimizer == "bootstrap_few_shot"
    assert artifact.train_score == 1.0
    assert artifact.artifact_id.startswith("opt_")


@pytest.mark.asyncio
async def test_cost_budget_plugin_aborts():
    from yaab import Runner
    from yaab.plugins.builtins import BudgetExceeded

    # TestModel reports cost 0, so simulate via a plugin that bumps cost.
    from yaab.models.base import ModelResponse
    from yaab.plugins import Plugin

    class Expensive(Plugin):
        async def after_model(self, ctx, agent, response):
            ctx.usage.cost_usd = 999.0
            return None

    runner = Runner(plugins=[Expensive(), CostBudgetPlugin(max_usd=1.0)])
    agent = Agent("a", model=TestModel("hi"))
    with pytest.raises(BudgetExceeded):
        await runner.run(agent, "hi")


@pytest.mark.asyncio
async def test_caching_plugin_short_circuits():
    from yaab import Runner

    model = TestModel("cached")
    runner = Runner(plugins=[CachingPlugin()])
    agent = Agent("a", model=model)
    await runner.run(agent, "same")
    n_calls_after_first = len(model.calls)
    await runner.run(agent, "same")
    # Second identical run should hit cache (no new model call).
    assert len(model.calls) == n_calls_after_first


def test_audit_plugin_records(tmp_path):
    from yaab import Runner
    from yaab.governance import AuditLog

    log = AuditLog()
    runner = Runner(plugins=[AuditPlugin(log)])
    agent = Agent("a", model=TestModel("hi"))
    runner.run_sync(agent, "hi")
    from yaab.governance.audit import AuditKind

    assert any(e.kind == AuditKind.MODEL_CALL for e in log.events)


def test_fastapi_app_builds():
    fastapi = pytest.importorskip("fastapi")
    from yaab.serve import fastapi_server_app

    agent = Agent("a", model=TestModel("hi"))
    app = fastapi_server_app(agent)
    routes = {r.path for r in app.routes}
    assert "/run" in routes
    assert "/.well-known/agent.json" in routes
    assert "/a2a/tasks" in routes


def test_cli_info(capsys):
    from yaab.cli import main

    assert main(["info"]) == 0
    out = capsys.readouterr().out
    assert "performance backend" in out
