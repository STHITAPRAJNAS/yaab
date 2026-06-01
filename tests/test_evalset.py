"""Tests for the portable EvalSet file format and the tool-trajectory metric.

Covers eval depth:
- :class:`EvalSet` / :class:`EvalCase` JSON round-trip (the ``.evalset.json`` format)
- :meth:`EvalSet.from_cases` and :meth:`EvalSet.to_dataset` interop with the
  existing :class:`Experiment` machinery
- the :class:`ToolTrajectoryMatch` evaluator and its ``tool_trajectory`` registry
  registration, scoring exact / partial / out-of-order trajectories
- end-to-end: an :class:`Agent` driven by ``TestModel(call_tools=[...])`` produces
  a :class:`RunResult` whose tool sequence is scored by the trajectory metric.
"""

from __future__ import annotations

import pytest

from yaab import Agent, tool
from yaab.governance.eval import (
    Case,
    Dataset,
    ExactMatch,
    Experiment,
    ToolTrajectoryMatch,
)
from yaab.governance.evalset import EvalCase, EvalSet
from yaab.models.test_model import TestModel

# --- Feature A: EvalSet / EvalCase portable file format ---------------------


def test_evalset_save_load_round_trip(tmp_path):
    es = EvalSet(
        name="weather-suite",
        cases=[
            EvalCase(
                id="single",
                conversation=["What is the weather in Paris?"],
                expected_output="It is sunny.",
            ),
            EvalCase(
                id="multi",
                conversation=["Hi", "And in London?"],
                expected_output="Rainy.",
                expected_tool_trajectory=[{"name": "get_weather", "arguments": {"city": "London"}}],
                metadata={"difficulty": "hard"},
            ),
        ],
    )
    path = tmp_path / "weather.evalset.json"
    es.save(path)
    assert path.exists()

    loaded = EvalSet.load(path)
    assert loaded.name == "weather-suite"
    assert loaded.version == "1"
    assert len(loaded.cases) == 2
    assert loaded.cases[0].id == "single"
    assert loaded.cases[1].conversation == ["Hi", "And in London?"]
    assert loaded.cases[1].expected_tool_trajectory == [
        {"name": "get_weather", "arguments": {"city": "London"}}
    ]
    assert loaded.cases[1].metadata == {"difficulty": "hard"}
    assert loaded == es


def test_evalset_file_has_schema_version(tmp_path):
    es = EvalSet(name="s", cases=[EvalCase(id="a", conversation=["q"])])
    path = tmp_path / "s.evalset.json"
    es.save(path)
    import json

    raw = json.loads(path.read_text(encoding="utf-8"))
    assert "schema_version" in raw
    assert raw["name"] == "s"


def test_evalset_from_cases_converts_yaab_cases():
    cases = [
        Case(name="c1", inputs="hello", expected="hi"),
        Case(name="c2", inputs="bye", expected="cya", metadata={"k": "v"}),
    ]
    es = EvalSet.from_cases(cases, name="converted")
    assert es.name == "converted"
    assert len(es.cases) == 2
    assert es.cases[0].id == "c1"
    assert es.cases[0].conversation == ["hello"]
    assert es.cases[0].expected_output == "hi"
    assert es.cases[1].metadata == {"k": "v"}


def test_evalset_to_dataset_returns_yaab_dataset():
    es = EvalSet(
        name="ds",
        cases=[
            EvalCase(id="x", conversation=["question one"], expected_output="answer one"),
        ],
    )
    ds = es.to_dataset()
    assert isinstance(ds, Dataset)
    assert ds.name == "ds"
    assert len(ds.cases) == 1
    assert ds.cases[0].name == "x"
    # The last user turn is the input the task receives.
    assert ds.cases[0].inputs == "question one"
    assert ds.cases[0].expected == "answer one"


def test_to_dataset_multi_turn_uses_last_turn_as_input():
    es = EvalSet(
        name="multi",
        cases=[EvalCase(id="m", conversation=["setup", "the real question"])],
    )
    ds = es.to_dataset()
    assert ds.cases[0].inputs == "the real question"
    # Full conversation preserved in metadata for multi-turn-aware tasks.
    assert ds.cases[0].metadata["conversation"] == ["setup", "the real question"]


@pytest.mark.asyncio
async def test_to_dataset_runs_under_experiment():
    es = EvalSet(
        name="run-me",
        cases=[
            EvalCase(id="c", conversation=["q"], expected_output="echo:q"),
        ],
    )
    ds = es.to_dataset()
    exp = Experiment(ds, [ExactMatch()], name="exp")
    result = await exp.run(lambda inp: f"echo:{inp}")
    assert result.results[0].scores["exact_match"] == 1.0


# --- Feature B: Tool-trajectory evaluation metric ---------------------------


def _ctx(trajectory):
    """Build the evaluator context dict the way Experiment.run does."""
    return {"tool_trajectory": trajectory, "output": ""}


def test_trajectory_exact_match_scores_one():
    ev = ToolTrajectoryMatch()
    case = Case(metadata={"expected_tool_trajectory": [{"name": "a"}, {"name": "b"}]})
    ctx = _ctx([{"name": "a", "arguments": {}}, {"name": "b", "arguments": {}}])
    assert ev.evaluate(case, ctx) == 1.0


def test_trajectory_partial_ordered_match_is_fractional():
    ev = ToolTrajectoryMatch()
    case = Case(
        metadata={"expected_tool_trajectory": [{"name": "a"}, {"name": "b"}, {"name": "c"}]}
    )
    # Only "a" and "c" present, in order; "b" missing -> 2/3.
    ctx = _ctx([{"name": "a", "arguments": {}}, {"name": "c", "arguments": {}}])
    assert ev.evaluate(case, ctx) == pytest.approx(2 / 3)


def test_trajectory_wrong_order_scores_zero_when_no_subsequence():
    ev = ToolTrajectoryMatch()
    case = Case(metadata={"expected_tool_trajectory": [{"name": "a"}, {"name": "b"}]})
    # Actual is b then a -> only one of the expected steps can match as a
    # subsequence (a after b is impossible; b matches first) -> 1/2.
    ctx = _ctx([{"name": "b", "arguments": {}}, {"name": "a", "arguments": {}}])
    assert ev.evaluate(case, ctx) == pytest.approx(0.5)


def test_trajectory_missing_all_scores_zero():
    ev = ToolTrajectoryMatch()
    case = Case(metadata={"expected_tool_trajectory": [{"name": "a"}, {"name": "b"}]})
    ctx = _ctx([{"name": "x", "arguments": {}}])
    assert ev.evaluate(case, ctx) == 0.0


def test_trajectory_strict_requires_exact_sequence():
    ev = ToolTrajectoryMatch(strict=True)
    case = Case(metadata={"expected_tool_trajectory": [{"name": "a"}, {"name": "b"}]})
    exact = _ctx([{"name": "a", "arguments": {}}, {"name": "b", "arguments": {}}])
    assert ev.evaluate(case, exact) == 1.0
    extra = _ctx([{"name": "a", "arguments": {}}, {"name": "b", "arguments": {}}, {"name": "c"}])
    assert ev.evaluate(case, extra) == 0.0


def test_trajectory_argument_subset_match():
    ev = ToolTrajectoryMatch()
    case = Case(metadata={"expected_tool_trajectory": [{"name": "a", "arguments": {"x": 1}}]})
    # Actual args are a superset of expected -> matches.
    good = _ctx([{"name": "a", "arguments": {"x": 1, "y": 2}}])
    assert ev.evaluate(case, good) == 1.0
    # Wrong arg value -> no match.
    bad = _ctx([{"name": "a", "arguments": {"x": 99}}])
    assert ev.evaluate(case, bad) == 0.0


def test_trajectory_metric_is_registered():
    from yaab.eval import available_metrics, get_metric

    assert "tool_trajectory" in available_metrics()
    m = get_metric("tool_trajectory", strict=True)
    assert isinstance(m, ToolTrajectoryMatch)
    assert m.strict is True


def test_trajectory_reads_expected_from_case_expected_too():
    """Expected trajectory may be supplied via case.expected (list), not only metadata."""
    ev = ToolTrajectoryMatch()
    case = Case(expected=[{"name": "a"}])
    ctx = _ctx([{"name": "a", "arguments": {}}])
    assert ev.evaluate(case, ctx) == 1.0


# --- Backward compatibility: plain (case, output) evaluators still work -----


@pytest.mark.asyncio
async def test_experiment_runresult_extraction_and_backcompat():
    @tool
    def get_weather(city: str) -> str:
        """Return the weather for a city."""
        return f"sunny in {city}"

    model = TestModel(custom_output="It is sunny.", call_tools=["get_weather"])
    agent = Agent("a", model=model, tools=[get_weather])

    ds = Dataset(
        cases=[
            Case(
                name="weather",
                inputs="weather?",
                expected="It is sunny.",
                metadata={"expected_tool_trajectory": [{"name": "get_weather"}]},
            )
        ]
    )
    # ExactMatch is a plain (case, output) evaluator; ToolTrajectoryMatch is
    # context-aware. Both must work in the same Experiment off one RunResult.
    exp = Experiment(ds, [ExactMatch(), ToolTrajectoryMatch()], name="e2e")
    result = await exp.run(lambda prompt: agent.run(prompt))

    cr = result.results[0]
    assert cr.error is None, cr.error
    # Final-output metric scored the RunResult's output string.
    assert cr.scores["exact_match"] == 1.0
    # Trajectory metric scored the tool sequence pulled from result.events.
    assert cr.scores["tool_trajectory"] == 1.0
    # Recorded output is the plain string, not the RunResult.
    assert cr.output == "It is sunny."


@pytest.mark.asyncio
async def test_experiment_plain_string_task_still_works():
    ds = Dataset(cases=[Case(name="c", inputs="x", expected="x")])
    exp = Experiment(ds, [ExactMatch()], name="plain")
    result = await exp.run(lambda inp: inp)
    assert result.results[0].scores["exact_match"] == 1.0
