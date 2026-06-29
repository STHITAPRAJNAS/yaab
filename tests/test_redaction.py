from __future__ import annotations

from yaab._redaction import scrub_secrets


def test_scrub_secrets_covers_common_token_shapes():
    secrets = [
        "sk-ABCDEFGHIJ1234567890",
        "ghp_ABCDEFGHIJKLMNOPQRST12345",
        "xoxb-1234567890-abcdefghij",
        "AIzaSyA1234567890abcdefghijklmnopqrstuvw",
        "-----BEGIN PRIVATE KEY-----",
    ]
    for s in secrets:
        out = scrub_secrets({"k": f"value={s}"})
        assert s not in str(out), f"{s!r} was not redacted"
        assert "REDACTED" in str(out)


def test_trace_safe_event_redacts():
    from yaab.runs.trace import _safe_event

    ev = _safe_event({"type": "tool_result", "result": "token=sk-ABCDEFGHIJ1234567890"})
    assert "sk-ABCDEFGHIJ1234567890" not in str(ev)


def test_audit_record_redacts_payload():
    from yaab.governance.audit import AuditKind, AuditLog

    log = AuditLog()
    event = log.record(AuditKind.TOOL_CALL, tool="fetch", result="key=sk-ABCDEFGHIJ1234567890")
    assert "sk-ABCDEFGHIJ1234567890" not in str(event.payload)
    # The chain still verifies (redaction happened before hashing).
    assert log.verify() is True
