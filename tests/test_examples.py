"""Every example and sample must actually run.

Two layers:

1. **Smoke (subprocess)** -- each ``examples/*.py`` script and ``python -m samples.<pkg>``
   runs to completion with exit code 0 under a forced ``cp1252`` console encoding.
   That is the default encoding of a legacy Windows console, so this reproduces on
   every platform the environment where a ``print()`` of a non-encodable character
   (e.g. U+2192) crashes for real Windows users.
2. **Logic (in-process)** -- each example is imported and its ``main()`` result is
   asserted on, so an example that "runs" but silently does the wrong thing fails.
"""

from __future__ import annotations

import asyncio
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

EXAMPLE_SCRIPTS = sorted(p.name for p in (ROOT / "examples").glob("*.py"))

SAMPLE_PACKAGES = sorted(p.parent.name for p in (ROOT / "samples").glob("*/__main__.py"))


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    """Run a script the way a user on a legacy Windows console would."""
    env = os.environ.copy()
    # Force the encoding of the child's stdout/stderr to cp1252 so a print() of a
    # character outside that codepage raises UnicodeEncodeError -- exactly what
    # happens on an out-of-the-box legacy Windows console.
    env["PYTHONIOENCODING"] = "cp1252"
    # Import yaab/examples/samples from THIS repo checkout, not from whatever an
    # editable install elsewhere on the machine points at.
    env["PYTHONPATH"] = str(ROOT)
    # Samples must never reach for a real provider in tests.
    env.pop("YAAB_SAMPLE_MODEL", None)
    return subprocess.run(
        args,
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        encoding="cp1252",
        errors="replace",
        timeout=180,
    )


def _assert_clean_exit(proc: subprocess.CompletedProcess[str], what: str) -> None:
    assert proc.returncode == 0, (
        f"{what} exited with {proc.returncode}\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    )


@pytest.mark.parametrize("script", EXAMPLE_SCRIPTS)
def test_example_runs_as_script(script: str) -> None:
    proc = _run([sys.executable, str(ROOT / "examples" / script)])
    _assert_clean_exit(proc, f"examples/{script}")


@pytest.mark.parametrize("package", SAMPLE_PACKAGES)
def test_sample_runs_as_module(package: str) -> None:
    proc = _run([sys.executable, "-m", f"samples.{package}"])
    _assert_clean_exit(proc, f"python -m samples.{package}")


# --------------------------------------------------------------------------
# Logic layer: import each example and assert on what main() actually returns.
# --------------------------------------------------------------------------


def _load_example(name: str):
    """Import an example module by file path (names start with digits)."""
    path = ROOT / "examples" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"example_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _main_result(name: str) -> dict:
    """Run an example's main(), awaiting it if it is async."""
    module = _load_example(name)
    result = module.main()
    if asyncio.iscoroutine(result):
        result = asyncio.run(result)
    assert isinstance(result, dict), f"{name}.main() must return a dict of results"
    return result


def test_quickstart_results() -> None:
    out = _main_result("01_quickstart")
    assert out["simple"] == "Hello! How can I help?"
    assert "5" in out["with_tool"]
    assert out["structured"].city == "Paris"
    assert out["structured"].temp_c == 21


def test_graph_hitl_pauses_then_completes() -> None:
    out = _main_result("02_graph_hitl")
    assert out["paused"].interrupted is True
    assert out["paused"].interrupt_value["amount"] == 10_000
    assert out["done"].state["status"] == "EXECUTED"


def test_governance_lifecycle_and_audit() -> None:
    out = _main_result("03_governance")
    assert out["approved"] is True
    assert out["output"] == "Customer appears low-risk."
    assert out["audit_events"] > 0
    assert out["chain_intact"] is True
    assert 0.0 < out["report"].coverage <= 1.0


def test_multi_agent_patterns() -> None:
    out = _main_result("04_multi_agent")
    assert out["sequential"] == "a tidy summary"
    # ParallelAgent returns a dict of sub-agent name -> output.
    assert out["parallel"]["legal"] == "legally fine"
    assert out["parallel"]["finance"] == "budget approved"
    assert out["swarm"] == "refund processed"


def test_streaming_tokens_and_events() -> None:
    from yaab import EventType

    out = _main_result("05_streaming")
    assert "".join(out["tokens"]) == "Hello there, this is streamed."
    assert EventType.RUN_START in out["event_types"]
    assert EventType.TEXT_DELTA in out["event_types"]
    assert EventType.RUN_END in out["event_types"]


def test_managers_store_and_recall() -> None:
    out = _main_result("06_managers")
    assert len(out["sessions"]) == 1
    assert out["state"]["tier"] == "gold"
    assert "Alice" in out["recall"]
    # list_versions returns the number of saved versions.
    assert out["versions"] == 2
    assert out["latest"] == "February statement"


def test_rag_index_retrieve_and_tool() -> None:
    out = _main_result("07_rag")
    assert out["chunks"] >= 3
    assert any("finance portal" in r.text for r in out["results"])
    assert out["answer"] == "File it in the finance portal."


def test_robust_agent_features() -> None:
    out = _main_result("08_robust_agent")
    assert "calculator" in out["tool_names"]
    assert "42" in str(out["calc"])
    assert out["config_agent_tools"] == 2
    assert out["robust_out"] == "The answer is 42."
    assert out["approved_run"] == "Refund processed."


def test_loaders_streaming_batch() -> None:
    out = _main_result("09_loaders_streaming_batch")
    # The CSV loader yields one document per row, so 1 markdown + 2 rows = 3.
    assert out["loaded"] >= 2
    assert "5 business days" in out["retrieved"]
    assert out["partials"] and out["partials"][-1].city == "Paris"
    assert out["batch_succeeded"] == 5
    assert out["batch_failed"] == 0


def test_serve_app_serves_http() -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from yaab.serve import fastapi_server_app

    serve_app = _load_example("serve_app")
    client = TestClient(fastapi_server_app(serve_app.agent))
    resp = client.post("/run", json={"prompt": "hi"})
    assert resp.status_code == 200
    assert "Hello from a served YAAB agent." in resp.text
