"""The ``yaab`` command-line interface.

Dependency-light (argparse only) so it works on a bare install. Commands:

* ``yaab info``               — environment + active performance backend;
* ``yaab init <name>``        — scaffold a starter agent file;
* ``yaab registry list``      — show the model inventory;
* ``yaab compliance report``  — generate a compliance report for a regime;
* ``yaab serve <module:agent>`` — serve an agent over HTTP (FastAPI);
* ``yaab eval <module:agent> <set.evalset.json>`` — score an agent on an evalset.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import sys
from typing import Any

from . import __version__, _core


def _info() -> int:
    import platform

    print(f"YAAB {__version__}")
    print(f"  performance backend : {_core.backend()}")
    print(f"  python              : {platform.python_version()}")
    try:
        import yaab_core  # noqa: F401

        print(f"  yaab-core (rust)    : {yaab_core.__version__}")
        print("  graph engine        : rust available (compile(engine='auto') uses it)")
    except ImportError:
        print("  yaab-core (rust)    : not installed (pure-Python fallback active)")
        print("  graph engine        : python (build yaab-core for the native engine)")
    return 0


_STARTER = '''"""A starter YAAB agent."""

from yaab import Agent


agent = Agent(
    "{name}",
    model="openai/gpt-4o",
    instructions="You are a helpful assistant.",
)


if __name__ == "__main__":
    print(agent.run_sync("Say hello in one sentence.").output)
'''


def _init(name: str) -> int:
    path = f"{name}.py"
    with open(path, "w") as fh:
        fh.write(_STARTER.format(name=name))
    print(f"created {path}")
    return 0


def _load_attr(spec: str):
    module_name, _, attr = spec.partition(":")
    module = importlib.import_module(module_name)
    return getattr(module, attr or "agent")


def _registry_list(db: str | None) -> int:
    from .governance.registry import AgentRegistry, SQLiteRegistryBackend

    registry = AgentRegistry(SQLiteRegistryBackend(db) if db else None)
    rows = registry.inventory()
    if not rows:
        print("(registry is empty)")
        return 0
    for row in rows:
        print(
            f"{row['agent_id']:<24} {row['name']:<20} "
            f"tier={row['risk_tier']:<8} status={row['approval_status']:<10} "
            f"state={row['lifecycle_state']}"
        )
    return 0


def _compliance_report(regime: str, db: str | None, agent_id: str | None) -> int:
    from .governance.audit import AuditLog
    from .governance.compliance import get_mapper
    from .governance.registry import AgentRegistry, SQLiteRegistryBackend

    mapper = get_mapper(regime)
    if mapper is None:
        print(f"unknown regime '{regime}'. Try: sr_11_7, eu_ai_act, nist_ai_rmf, iso_42001, soc2")
        return 1
    registry = AgentRegistry(SQLiteRegistryBackend(db) if db else None)
    report = mapper.map(registry, AuditLog(), agent_id)
    print(report.to_markdown())
    return 0


def _serve(spec: str, host: str, port: int) -> int:
    from .serve import serve

    serve(_load_attr(spec), host=host, port=port)
    return 0


def _web(spec: str, host: str, port: int) -> int:
    from .web import serve_web

    serve_web(_load_attr(spec), host=host, port=port)
    return 0


def _evaluators_for_case(case: Any, requested: list[str]) -> list[Any]:
    """Pick the metrics to score one case with.

    Explicit ``--metric`` flags win and apply to every case (the caller asked
    for them by name, so we don't second-guess). With no flags we *auto-detect*
    per case: ``exact_match`` when an expected output is present and
    ``tool_trajectory`` when an expected tool trajectory is — so
    a mixed suite of output-cases and tool-cases scores the right thing for each
    without the author wiring metrics up by hand.
    """
    from .eval import get_metric

    if requested:
        return [get_metric(name) for name in requested]
    names: list[str] = []
    if case.expected is not None:
        names.append("exact_match")
    if case.metadata.get("expected_tool_trajectory"):
        names.append("tool_trajectory")
    return [get_metric(name) for name in names]


async def _run_eval(agent: Any, dataset: Any, requested: list[str]) -> list[dict[str, Any]]:
    """Run every case against ``agent`` and score it, returning per-case rows.

    Reuses the eval engine's own output-unpacking and context-aware scoring
    (``_unpack_task_output`` / ``_score_evaluator``) so trajectory metrics get
    the tool-call sequence mined from the run's events — exactly what the
    ``Experiment`` machinery does — while plain string metrics see the final
    output. Errors are recorded per case (never abort the suite), matching the
    in-memory ``Experiment`` semantics.
    """
    from .governance.eval import _score_evaluator, _unpack_task_output

    rows: list[dict[str, Any]] = []
    for case in dataset.cases:
        row: dict[str, Any] = {"case": case.name, "scores": {}, "error": None, "output": None}
        evaluators = _evaluators_for_case(case, requested)
        try:
            result = await agent.run(str(case.inputs))
            output, context = _unpack_task_output(result)
            # Keep the report JSON-serializable: scalars pass through, anything
            # else is stringified rather than risking a non-serializable payload.
            if isinstance(output, (str, int, float, bool, type(None))):
                row["output"] = output
            else:
                row["output"] = str(output)
            for ev in evaluators:
                row["scores"][ev.name] = await _score_evaluator(ev, case, output, context)
        except Exception as exc:  # noqa: BLE001 - record, don't abort the suite
            row["error"] = str(exc)
        rows.append(row)
    return rows


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Mean score per metric across all cases (ignoring cases that errored)."""
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for row in rows:
        for name, value in row["scores"].items():
            sums[name] = sums.get(name, 0.0) + value
            counts[name] = counts.get(name, 0) + 1
    return {name: sums[name] / counts[name] for name in sums}


def _print_eval_table(
    rows: list[dict[str, Any]], aggregate: dict[str, float], fail_under: float | None
) -> None:
    """Print a plain-text per-case table plus a per-metric summary line.

    Dependency-free formatting (str padding, like the rest of this CLI) so the
    report is readable in any terminal/CI log without rich/tabulate. A case
    "passes" when every metric it was scored on is >= ``fail_under`` (or, with no
    gate, > 0); the column makes regressions obvious at a glance.
    """
    metrics = list(aggregate.keys())
    threshold = fail_under if fail_under is not None else 0.0
    header = f"{'case':<24} " + " ".join(f"{m:<18}" for m in metrics) + " result"
    print(header)
    print("-" * len(header))
    for row in rows:
        cells = []
        passed = True
        for m in metrics:
            score = row["scores"].get(m)
            if score is None:
                cells.append(f"{'-':<18}")
            else:
                cells.append(f"{score:<18.2f}")
                if score < threshold or (fail_under is None and score <= 0.0):
                    passed = False
        if row["error"]:
            verdict = f"ERROR: {row['error']}"
            passed = False
        else:
            verdict = "PASS" if passed else "FAIL"
        print(f"{row['case']:<24} " + " ".join(cells) + f" {verdict}")
    print("-" * len(header))
    summary = "  ".join(f"{m}={aggregate[m]:.2f}" for m in metrics) or "(no metrics scored)"
    print(f"mean: {summary}")


def _eval(
    spec: str,
    evalset_path: str,
    *,
    metrics: list[str],
    output: str | None,
    fail_under: float | None,
) -> int:
    """Score ``spec``'s agent on ``evalset_path`` and gate CI on the mean.

    Exit codes follow the CI-gate convention: ``0`` when every metric mean is at
    or above ``--fail-under`` (or always, when no gate is set), ``1`` when any
    metric mean falls below the gate — so this command drops straight into a
    pipeline as a quality gate.
    """
    from .governance.evalset import EvalSet

    agent = _load_attr(spec)
    evalset = EvalSet.load(evalset_path)
    dataset = evalset.to_dataset()

    rows = asyncio.run(_run_eval(agent, dataset, metrics))
    aggregate = _aggregate(rows)
    _print_eval_table(rows, aggregate, fail_under)

    if output:
        report = {
            "evalset": evalset.name,
            "agent": spec,
            "aggregate": aggregate,
            "cases": rows,
        }
        with open(output, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False)
        print(f"wrote report to {output}")

    if fail_under is not None:
        below = {m: v for m, v in aggregate.items() if v < fail_under}
        if below:
            failed = ", ".join(f"{m}={v:.2f}" for m, v in below.items())
            print(f"FAIL: below --fail-under {fail_under:.2f}: {failed}")
            return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="yaab", description="Yet Another Agent Builder")
    parser.add_argument("--version", action="version", version=f"yaab {__version__}")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("info", help="show environment and performance backend")

    p_init = sub.add_parser("init", help="scaffold a starter agent")
    p_init.add_argument("name")

    p_reg = sub.add_parser("registry", help="agent registry commands")
    reg_sub = p_reg.add_subparsers(dest="registry_command")
    p_reg_list = reg_sub.add_parser("list", help="show the model inventory")
    p_reg_list.add_argument("--db", default=None)

    p_comp = sub.add_parser("compliance", help="compliance commands")
    comp_sub = p_comp.add_subparsers(dest="compliance_command")
    p_comp_report = comp_sub.add_parser("report", help="generate a compliance report")
    p_comp_report.add_argument("regime")
    p_comp_report.add_argument("--db", default=None)
    p_comp_report.add_argument("--agent-id", default=None)

    p_serve = sub.add_parser("serve", help="serve an agent over HTTP (module:agent)")
    p_serve.add_argument("spec")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)

    p_web = sub.add_parser("web", help="open a browser dev playground for an agent")
    p_web.add_argument("spec")
    p_web.add_argument("--host", default="127.0.0.1")
    p_web.add_argument("--port", type=int, default=8080)

    p_eval = sub.add_parser("eval", help="score an agent on a .evalset.json (module:agent)")
    p_eval.add_argument("spec", help="agent as module:attribute (resolved like `yaab serve`)")
    p_eval.add_argument("evalset", help="path to a .evalset.json file")
    p_eval.add_argument(
        "--metric",
        action="append",
        default=[],
        dest="metrics",
        help="metric name (repeatable); defaults to auto-detect per case",
    )
    p_eval.add_argument("--output", default=None, help="write a JSON report to this path")
    p_eval.add_argument(
        "--fail-under",
        type=float,
        default=None,
        help="exit 1 if any metric mean is below this score (CI gate)",
    )

    args = parser.parse_args(argv)

    if args.command == "info" or args.command is None:
        return _info()
    if args.command == "init":
        return _init(args.name)
    if args.command == "registry":
        if args.registry_command == "list":
            return _registry_list(args.db)
        print("usage: yaab registry list [--db PATH]")
        return 1
    if args.command == "compliance":
        if args.compliance_command == "report":
            return _compliance_report(args.regime, args.db, args.agent_id)
        print("usage: yaab compliance report <regime> [--db PATH] [--agent-id ID]")
        return 1
    if args.command == "serve":
        return _serve(args.spec, args.host, args.port)
    if args.command == "web":
        return _web(args.spec, args.host, args.port)
    if args.command == "eval":
        return _eval(
            args.spec,
            args.evalset,
            metrics=args.metrics,
            output=args.output,
            fail_under=args.fail_under,
        )
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
