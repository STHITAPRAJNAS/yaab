from __future__ import annotations

import pytest

from yaab.tools.sandbox import SubprocessSandbox


@pytest.mark.asyncio
async def test_subprocess_does_not_inherit_secret_env(monkeypatch):
    monkeypatch.setenv("SECRET_API_KEY", "sk-leak-me")
    sb = SubprocessSandbox()
    out = await sb.run("import os; print(os.environ.get('SECRET_API_KEY', 'ABSENT'))", timeout=10)
    assert "sk-leak-me" not in out
    assert "ABSENT" in out


@pytest.mark.asyncio
async def test_timeout_reports_cleanly():
    sb = SubprocessSandbox()
    out = await sb.run("import time; time.sleep(30)", timeout=1)
    assert "timeout" in out.lower()


@pytest.mark.asyncio
async def test_normal_output_still_works():
    sb = SubprocessSandbox()
    out = await sb.run("print(sum(range(11)))", timeout=10)
    assert out.strip() == "55"
