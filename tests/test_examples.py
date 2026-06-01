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
