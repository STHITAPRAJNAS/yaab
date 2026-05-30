"""Tests for prefix-scoped State (ADK temp:/user:/app:)."""

from __future__ import annotations

import pytest

from yaab import SessionManager, State
from yaab.state import scope_of


def test_scope_of():
    assert scope_of("x") == "session"
    assert scope_of("app:cfg") == "app"
    assert scope_of("user:pref") == "user"
    assert scope_of("temp:flag") == "temp"


def test_state_routes_by_prefix():
    app, user = {}, {}
    st = State(user=user, app=app)
    st["k"] = 1            # session
    st["app:cfg"] = "v"    # app
    st["user:pref"] = "p"  # user
    st["temp:tmp"] = "t"   # temp (ephemeral)

    assert st["k"] == 1
    assert app["app:cfg"] == "v"
    assert user["user:pref"] == "p"
    assert st["temp:tmp"] == "t"
    assert st.session == {"k": 1}
    assert st.temp == {"temp:tmp": "t"}


def test_persisted_excludes_temp():
    st = State()
    st["keep"] = 1
    st["temp:drop"] = 2
    persisted = st.persisted()
    assert "keep" in persisted
    assert "temp:drop" not in persisted


def test_state_mapping_protocol():
    st = State()
    st["a"] = 1
    st["app:b"] = 2
    assert len(st) == 2
    assert set(st) == {"a", "app:b"}
    del st["a"]
    assert "a" not in st


@pytest.mark.asyncio
async def test_session_manager_resolve_state_scopes():
    mgr = SessionManager()
    s1 = await mgr.create_session(app_name="bank", user_id="alice")
    s2 = await mgr.create_session(app_name="bank", user_id="alice")

    st1 = await mgr.resolve_state(s1.id, app_name="bank", user_id="alice")
    st1["user:tier"] = "gold"   # shared across alice's sessions
    st1["app:region"] = "eu"    # shared across the app
    st1["local"] = "only-s1"    # session-scoped

    # A second session for the same user sees user:/app: but not session-local.
    st2 = await mgr.resolve_state(s2.id, app_name="bank", user_id="alice")
    assert st2["user:tier"] == "gold"
    assert st2["app:region"] == "eu"
    assert "local" not in st2

    # A different user does NOT see alice's user: state.
    st3 = await mgr.resolve_state(
        (await mgr.create_session(app_name="bank", user_id="bob")).id,
        app_name="bank", user_id="bob",
    )
    assert "user:tier" not in st3
    assert st3["app:region"] == "eu"  # but app: state is global
