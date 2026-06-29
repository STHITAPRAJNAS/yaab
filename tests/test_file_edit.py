from __future__ import annotations

import pytest

from yaab.tools.builtin.files import file_toolset
from yaab.types import RunContext


def _tools(root):
    return {t.name: t for t in file_toolset(root=root)}


@pytest.mark.asyncio
async def test_edit_requires_read_first(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n")
    by = _tools(str(tmp_path))
    ctx = RunContext()
    # Edit before reading -> rejected (staleness guard).
    out = await by["file_edit"].fn(ctx, path="a.py", old_string="x = 1", new_string="x = 9")
    assert "error" in out.lower() and "read" in out.lower()
    assert (tmp_path / "a.py").read_text() == "x = 1\n"  # unchanged


@pytest.mark.asyncio
async def test_edit_exact_single_match(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\ny = 2\n")
    by = _tools(str(tmp_path))
    ctx = RunContext()
    await by["file_read"].fn(ctx, path="a.py")
    out = await by["file_edit"].fn(ctx, path="a.py", old_string="x = 1", new_string="x = 9")
    assert "diff" in out.lower() or "x = 9" in out
    assert (tmp_path / "a.py").read_text() == "x = 9\ny = 2\n"


@pytest.mark.asyncio
async def test_edit_ambiguous_is_error(tmp_path):
    (tmp_path / "b.py").write_text("a\na\n")
    by = _tools(str(tmp_path))
    ctx = RunContext()
    await by["file_read"].fn(ctx, path="b.py")
    out = await by["file_edit"].fn(ctx, path="b.py", old_string="a", new_string="z")
    assert "error" in out.lower() and "2 times" in out
    assert (tmp_path / "b.py").read_text() == "a\na\n"  # unchanged


@pytest.mark.asyncio
async def test_edit_not_found_is_error(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n")
    by = _tools(str(tmp_path))
    ctx = RunContext()
    await by["file_read"].fn(ctx, path="a.py")
    out = await by["file_edit"].fn(ctx, path="a.py", old_string="zzz", new_string="q")
    assert "error" in out.lower() and "not found" in out.lower()


@pytest.mark.asyncio
async def test_edit_refuses_oversized_file_no_truncation(tmp_path):
    # Regression: editing a >1MB file used to silently truncate to the 1MB cap
    # (the edit operates on a capped view that is then written back wholesale).
    big = tmp_path / "big.txt"
    payload = b"AAAA" + b"x" * 1_000_000 + b"-TAIL-MARKER-"
    big.write_bytes(payload)
    by = _tools(str(tmp_path))
    ctx = RunContext()
    await by["file_read"].fn(ctx, path="big.txt")
    out = await by["file_edit"].fn(ctx, path="big.txt", old_string="AAAA", new_string="BBBB")
    assert "error" in out.lower() and "exceed" in out.lower()
    # The file is byte-for-byte intact — nothing was truncated.
    assert big.read_bytes() == payload


@pytest.mark.asyncio
async def test_edit_replace_all(tmp_path):
    (tmp_path / "b.py").write_text("a\na\n")
    by = _tools(str(tmp_path))
    ctx = RunContext()
    await by["file_read"].fn(ctx, path="b.py")
    out = await by["file_edit"].fn(
        ctx, path="b.py", old_string="a", new_string="z", replace_all=True
    )
    assert "error" not in out.lower()
    assert (tmp_path / "b.py").read_text() == "z\nz\n"
