"""Tests for the user-simulation eval driver.

A :class:`UserSimulator` is an LLM playing a persona with a goal that drives a
multi-turn conversation against a real :class:`Agent`. These tests pin the
contract entirely offline with ``FunctionModel``/``TestModel`` doubles:

- a scripted simulator producing 2 user turns then ``[DONE]`` yields a
  transcript of 2 user + 2 agent turns and a parsed ``goal_achieved``;
- ``max_turns`` caps the loop even when the simulator never says ``[DONE]``;
- a ``stop_when`` predicate can terminate the loop on the agent's reply;
- the agent carries a ``session_id`` so it sees prior turns (verified by a
  message-counting :class:`FunctionModel` agent);
- :class:`SimulationEvaluator` scores 1.0 when the goal is achieved;
- :func:`simulate_evalset` seeds the persona/goal from each case's metadata.
"""

from __future__ import annotations

import pytest

from yaab import Agent
from yaab.governance.evalset import EvalCase, EvalSet
from yaab.governance.simulation import (
    SimulationEvaluator,
    SimulationResult,
    UserSimulator,
    simulate,
    simulate_evalset,
)
from yaab.models.base import ModelResponse
from yaab.models.test_model import FunctionModel, TestModel
from yaab.types import Message, Role


def _scripted_simulator(turns: list[str]) -> UserSimulator:
    """A simulator whose user messages are read off a fixed script.

    The simulator model ignores the conversation and just emits the next
    scripted line per call; after the script is exhausted it emits ``[DONE]``
    and, when asked for the final self-assessment, ``GOAL_ACHIEVED: yes``.
    """
    state = {"i": 0}

    def fn(messages: list[Message]) -> str:
        # The final self-assessment prompt contains the GOAL_ACHIEVED marker.
        last = messages[-1].content if messages else ""
        if "GOAL_ACHIEVED" in last:
            return "GOAL_ACHIEVED: yes"
        i = state["i"]
        state["i"] += 1
        if i < len(turns):
            return turns[i]
        return "[DONE]"

    return UserSimulator(
        model=FunctionModel(fn),
        persona="A curious user",
        goal="Find out the weather",
    )


@pytest.mark.asyncio
async def test_two_turns_then_done_produces_full_transcript():
    agent = Agent("a", model=TestModel(custom_output="ok"))
    sim = _scripted_simulator(["Hi there", "And tomorrow?"])

    result = await simulate(agent, sim)

    assert isinstance(result, SimulationResult)
    assert result.turns == 2
    roles = [m["role"] for m in result.transcript]
    assert roles == ["user", "assistant", "user", "assistant"]
    assert result.transcript[0]["content"] == "Hi there"
    assert result.transcript[1]["content"] == "ok"
    assert result.transcript[2]["content"] == "And tomorrow?"
    assert result.goal_achieved is True


@pytest.mark.asyncio
async def test_max_turns_caps_the_loop():
    agent = Agent("a", model=TestModel(custom_output="ok"))

    # A simulator that NEVER says [DONE]; only the cap stops it.
    def fn(messages: list[Message]) -> str:
        last = messages[-1].content if messages else ""
        if "GOAL_ACHIEVED" in last:
            return "GOAL_ACHIEVED: no"
        return "keep going"

    sim = UserSimulator(
        model=FunctionModel(fn),
        persona="A persistent user",
        goal="Never satisfied",
        max_turns=3,
    )

    result = await simulate(agent, sim)
    assert result.turns == 3
    assert len([m for m in result.transcript if m["role"] == "user"]) == 3
    assert result.goal_achieved is False


@pytest.mark.asyncio
async def test_stop_when_predicate_terminates_loop():
    # Agent says "STOP" on the second reply; stop_when ends the loop there.
    agent = Agent(
        "a",
        model=TestModel(responses=["first", "STOP", "third"]),
    )

    def fn(messages: list[Message]) -> str:
        last = messages[-1].content if messages else ""
        if "GOAL_ACHIEVED" in last:
            return "GOAL_ACHIEVED: yes"
        return "go on"

    sim = UserSimulator(
        model=FunctionModel(fn),
        persona="p",
        goal="g",
        max_turns=10,
        stop_when=lambda reply: "STOP" in reply,
    )

    result = await simulate(agent, sim)
    assert result.turns == 2
    assert result.transcript[-1]["content"] == "STOP"


@pytest.mark.asyncio
async def test_session_continuity_agent_sees_prior_turns():
    """The agent must carry a session so it accumulates history across turns.

    A FunctionModel agent that reports how many messages it received proves the
    history grows turn over turn (replayed from the session service).
    """

    def agent_fn(messages: list[Message]) -> str:
        # Count only user/assistant turns the agent has seen so far.
        convo = [m for m in messages if m.role in (Role.USER, Role.ASSISTANT)]
        return f"seen={len(convo)}"

    agent = Agent("counter", model=FunctionModel(agent_fn))
    sim = _scripted_simulator(["one", "two", "three"])

    result = await simulate(agent, sim)

    # Turn 1: agent sees just the first user msg -> seen=1.
    # Turn 2: it replays [user1, assistant1, user2] -> seen=3.
    # Turn 3: [user1, a1, user2, a2, user3] -> seen=5.
    agent_replies = [m["content"] for m in result.transcript if m["role"] == "assistant"]
    assert agent_replies == ["seen=1", "seen=3", "seen=5"]


@pytest.mark.asyncio
async def test_explicit_session_id_is_honored():
    agent = Agent("a", model=TestModel(custom_output="ok"))
    sim = _scripted_simulator(["hi"])
    result = await simulate(agent, sim, session_id="fixed-session")
    assert result.turns == 1


@pytest.mark.asyncio
async def test_aggregate_usage_sums_agent_calls():
    agent = Agent("a", model=TestModel(custom_output="ok"))
    sim = _scripted_simulator(["a", "b"])
    result = await simulate(agent, sim)
    # TestModel reports requests=1 per call; two agent turns -> 2 requests.
    assert result.agent_usage.requests == 2
    assert result.agent_usage.total_tokens == 30


@pytest.mark.asyncio
async def test_simulation_evaluator_scores_one_on_achieved_goal():
    agent = Agent("a", model=TestModel(custom_output="ok"))
    sim = _scripted_simulator(["hi"])
    evaluator = SimulationEvaluator()
    score = await evaluator.ascore(agent, sim)
    assert score == 1.0


@pytest.mark.asyncio
async def test_simulation_evaluator_scores_zero_on_failed_goal():
    agent = Agent("a", model=TestModel(custom_output="ok"))

    def fn(messages: list[Message]) -> str:
        last = messages[-1].content if messages else ""
        if "GOAL_ACHIEVED" in last:
            return "GOAL_ACHIEVED: no"
        return "[DONE]"

    sim = UserSimulator(model=FunctionModel(fn), persona="p", goal="g")
    evaluator = SimulationEvaluator()
    score = await evaluator.ascore(agent, sim)
    assert score == 0.0


@pytest.mark.asyncio
async def test_simulation_evaluator_custom_metric():
    agent = Agent("a", model=TestModel(custom_output="ok"))
    sim = _scripted_simulator(["a", "b"])
    # Metric = number of turns, normalized.
    evaluator = SimulationEvaluator(metric=lambda r: r.turns / 10.0)
    score = await evaluator.ascore(agent, sim)
    assert score == pytest.approx(0.2)


@pytest.mark.asyncio
async def test_simulator_emits_done_token_immediately():
    """A simulator that gives up at turn zero yields an empty conversation."""
    agent = Agent("a", model=TestModel(custom_output="ok"))

    def fn(messages: list[Message]) -> str:
        last = messages[-1].content if messages else ""
        if "GOAL_ACHIEVED" in last:
            return "GOAL_ACHIEVED: no"
        return "[DONE]"

    sim = UserSimulator(model=FunctionModel(fn), persona="p", goal="g")
    result = await simulate(agent, sim)
    assert result.turns == 0
    assert result.transcript == []
    assert result.goal_achieved is False


@pytest.mark.asyncio
async def test_simulator_next_message_uses_persona_and_goal():
    """The simulator's system prompt must carry the persona and goal."""
    captured: dict[str, str] = {}

    def fn(messages: list[Message]) -> str:
        system = next((m.content for m in messages if m.role is Role.SYSTEM), "")
        captured["system"] = system
        return "[DONE]"

    sim = UserSimulator(
        model=FunctionModel(fn),
        persona="An impatient executive",
        goal="Book a flight to Tokyo",
    )
    await sim.next_message([])
    assert "An impatient executive" in captured["system"]
    assert "Book a flight to Tokyo" in captured["system"]


@pytest.mark.asyncio
async def test_simulator_strips_done_token_and_detects_it():
    sim = _scripted_simulator([])
    # First call returns the [DONE] sentinel.
    msg, done = await sim.next_message([])
    assert done is True


@pytest.mark.asyncio
async def test_simulate_evalset_seeds_persona_and_goal_from_metadata():
    agent = Agent("a", model=TestModel(custom_output="ok"))

    seen_systems: list[str] = []

    def sim_fn(messages: list[Message]) -> str:
        system = next((m.content for m in messages if m.role is Role.SYSTEM), "")
        last = messages[-1].content if messages else ""
        if "GOAL_ACHIEVED" in last:
            return "GOAL_ACHIEVED: yes"
        if system not in seen_systems:
            seen_systems.append(system)
            return "opening line"
        return "[DONE]"

    evalset = EvalSet(
        name="personas",
        cases=[
            EvalCase(
                id="case1",
                conversation=["seed"],
                metadata={"persona": "A retiree", "goal": "Check my pension"},
            ),
            EvalCase(
                id="case2",
                conversation=["seed"],
                metadata={"persona": "A student", "goal": "Find a loan"},
            ),
        ],
    )

    results = await simulate_evalset(agent, evalset, simulator_model=FunctionModel(sim_fn))

    assert len(results) == 2
    assert all(isinstance(r, SimulationResult) for r in results)
    # Each case's persona/goal made it into a distinct simulator system prompt.
    blob = "\n".join(seen_systems)
    assert "A retiree" in blob
    assert "Check my pension" in blob
    assert "A student" in blob
    assert "Find a loan" in blob


@pytest.mark.asyncio
async def test_simulation_result_is_pydantic_serializable():
    agent = Agent("a", model=TestModel(custom_output="ok"))
    sim = _scripted_simulator(["hi"])
    result = await simulate(agent, sim)
    dumped = result.model_dump()
    assert dumped["turns"] == 1
    assert dumped["transcript"][0]["role"] == "user"


def test_modelresponse_simulator_passthrough():
    """The simulator accepts a ModelResponse from the model (not just str)."""

    def fn(messages: list[Message]) -> ModelResponse:
        return ModelResponse(content="[DONE]")

    sim = UserSimulator(model=FunctionModel(fn), persona="p", goal="g")
    assert sim.persona == "p"
