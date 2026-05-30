"""Current-date/time tool."""

from __future__ import annotations

from datetime import UTC, datetime

from ..base import tool


@tool
def current_time(tz_offset_hours: float = 0.0) -> str:
    """Return the current date and time in ISO 8601 format (UTC by default).

    ``tz_offset_hours`` shifts the result from UTC (e.g. -5 for US Eastern).
    """
    from datetime import timedelta

    now = datetime.now(UTC) + timedelta(hours=tz_offset_hours)
    return now.isoformat()
