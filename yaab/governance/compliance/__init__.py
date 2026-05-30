"""Compliance mappers — pluggable, regime-agnostic, additive.

Resolve a mapper by regime name with :func:`get_mapper`, or discover the full
set (including third-party ones registered via the ``yaab.compliance`` entry
point) with :func:`available_mappers`.
"""

from __future__ import annotations

from .base import ComplianceMapper, ComplianceReport, ControlResult, ControlStatus
from .eu_ai_act import EUAIActMapper
from .iso_42001 import ISO42001Mapper
from .nist_ai_rmf import NISTAIRMFMapper
from .soc2 import SOC2Mapper
from .sr_11_7 import SR117Mapper

_BUILTINS: dict[str, type] = {
    "sr_11_7": SR117Mapper,
    "eu_ai_act": EUAIActMapper,
    "nist_ai_rmf": NISTAIRMFMapper,
    "iso_42001": ISO42001Mapper,
    "soc2": SOC2Mapper,
}


def available_mappers() -> dict[str, type]:
    """Built-in mappers plus any registered via the ``yaab.compliance`` entry point."""
    mappers = dict(_BUILTINS)
    try:
        from importlib.metadata import entry_points

        for ep in entry_points(group="yaab.compliance"):
            try:
                mappers[ep.name] = ep.load()
            except Exception:  # noqa: BLE001 - a broken plugin shouldn't break discovery
                continue
    except Exception:  # noqa: BLE001
        pass
    return mappers


def get_mapper(regime: str) -> ComplianceMapper | None:
    cls = available_mappers().get(regime)
    return cls() if cls else None


__all__ = [
    "ComplianceMapper",
    "ComplianceReport",
    "ControlResult",
    "ControlStatus",
    "SR117Mapper",
    "EUAIActMapper",
    "NISTAIRMFMapper",
    "ISO42001Mapper",
    "SOC2Mapper",
    "available_mappers",
    "get_mapper",
]
