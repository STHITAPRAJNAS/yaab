from __future__ import annotations

import os

import pytest

from yaab.tools.builtin import make_file_tools


@pytest.mark.asyncio
async def test_write_is_atomic_and_protects_git(tmp_path):
    _read, write, _list = make_file_tools(root=str(tmp_path))
    out = await write.fn(path=".git/config", content="evil")
    assert "error" in out.lower()
    assert not (tmp_path / ".git" / "config").exists()

    ok = await write.fn(path="notes.txt", content="hello")
    assert "wrote" in ok
    assert (tmp_path / "notes.txt").read_text() == "hello"


@pytest.mark.asyncio
async def test_write_rejects_lock_files(tmp_path):
    _read, write, _list = make_file_tools(root=str(tmp_path))
    out = await write.fn(path="poetry.lock", content="x")
    assert "error" in out.lower()


@pytest.mark.asyncio
async def test_write_rejects_symlink_escape(tmp_path):
    if os.name != "posix":
        pytest.skip("symlink test is POSIX-specific")
    outside = tmp_path / "outside.txt"
    outside.write_text("original")
    root = tmp_path / "root"
    root.mkdir()
    (root / "link.txt").symlink_to(outside)
    _read, write, _list = make_file_tools(root=str(root))
    out = await write.fn(path="link.txt", content="pwned")
    assert "error" in out.lower()
    assert outside.read_text() == "original"  # write did NOT follow the symlink
