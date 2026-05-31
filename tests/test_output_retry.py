"""Output-validation reflection/retry — and its idempotency across runs.

When the model returns content that fails the output schema, the runner feeds
the validation error back and retries up to ``output_retries`` times. These
tests pin that behavior AND guard against a regression where the per-run retry
budget leaked into the agent's shared state, silently reducing retries on every
subsequent run.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from yaab import Agent
from yaab.models.test_model import TestModel


class Strict(BaseModel):
    value: int


@pytest.mark.asyncio
async def test_invalid_then_valid_is_repaired():
    # First completion is non-JSON (fails validation), second is valid.
    model = TestModel(responses=["not json at all", '{"value": 7}'])
    agent = Agent("s", model=model, output_type=Strict, output_retries=2)
    result = await agent.run("go")
    assert isinstance(result.output, Strict)
    assert result.output.value == 7


@pytest.mark.asyncio
async def test_output_retries_not_consumed_across_runs():
    """The configured retry budget must be the same for run 2 as for run 1.

    Regression: the runner used to do ``agent.output_retries -= 1`` on the shared
    Agent, so a reused agent lost a retry per failed validation forever. After
    two runs that each needed one repair, a third run had zero retries left and
    would fail spuriously.
    """
    agent = Agent("s", model=TestModel("x"), output_type=Strict, output_retries=2)

    # Run 1: needs exactly one repair.
    agent._model = TestModel(responses=["nope", '{"value": 1}'])
    r1 = await agent.run("go")
    assert r1.output.value == 1

    # Run 2: same agent, again needs one repair. Must still succeed.
    agent._model = TestModel(responses=["nope", '{"value": 2}'])
    r2 = await agent.run("go")
    assert r2.output.value == 2

    # The agent's configured budget is unchanged (not silently depleted).
    assert agent.output_retries == 2

    # Run 3: a run that needs the FULL budget still has it.
    agent._model = TestModel(responses=["bad", "still bad", '{"value": 3}'])
    r3 = await agent.run("go")
    assert r3.output.value == 3


@pytest.mark.asyncio
async def test_retries_exhausted_raises():
    from pydantic import ValidationError

    # Every response is invalid; with one retry, both attempts fail -> raise.
    agent = Agent(
        "s",
        model=TestModel(responses=["bad", "also bad", "still bad"]),
        output_type=Strict,
        output_retries=1,
    )
    with pytest.raises(ValidationError):
        await agent.run("go")
    # Budget unchanged even on the failure path.
    assert agent.output_retries == 1
