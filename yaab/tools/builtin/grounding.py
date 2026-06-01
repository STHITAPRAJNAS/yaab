"""grounded_search — provider-native search grounding passthrough.

Provider-native search grounding (e.g. Gemini's ``googleSearch``) isn't a
function tool: it flips on the provider's *built-in* grounding so the model
searches and cites for itself. There is nothing for the SDK to execute — the
grounding happens inside the provider. Instead we expose
:func:`grounding_settings`, a pure helper that returns the ``model_settings``
fragment LiteLLM forwards to enable native grounding.

For Gemini via LiteLLM that fragment is ``{"tools": [{"googleSearch": {}}]}``::

    from yaab import Agent
    from yaab.tools.builtin.grounding import grounding_settings

    agent = Agent(
        "researcher",
        model="gemini/gemini-2.0-flash",
        model_settings=grounding_settings(provider="gemini"),
    )

It's a pure function (no network, no I/O) so it's trivially testable and safe to
call anywhere. Each call returns a fresh dict so callers can mutate the result
without affecting later calls.
"""

from __future__ import annotations

from typing import Any

#: provider -> the ``model_settings`` fragment enabling native search grounding.
#: Stored as builder callables so every call hands back an independent dict.
_GROUNDING_BUILDERS: dict[str, Any] = {
    # Gemini (via LiteLLM): the Google Search retrieval tool.
    "gemini": lambda: {"tools": [{"googleSearch": {}}]},
    "google": lambda: {"tools": [{"googleSearch": {}}]},
}


def grounding_settings(provider: str = "gemini") -> dict[str, Any]:
    """Return the ``model_settings`` fragment that enables native search grounding.

    Merge this into an agent's ``model_settings`` to turn on the provider's
    built-in web grounding (no YAAB-side tool execution involved). Currently
    supports ``"gemini"`` (alias ``"google"``), which yields
    ``{"tools": [{"googleSearch": {}}]}``.

    Raises ``ValueError`` for an unsupported provider so misconfiguration fails
    loudly instead of silently producing an empty fragment.
    """
    builder = _GROUNDING_BUILDERS.get(provider.lower())
    if builder is None:
        supported = sorted(_GROUNDING_BUILDERS)
        raise ValueError(
            f"no native search grounding for provider {provider!r}; supported: {supported}"
        )
    return builder()


__all__ = ["grounding_settings"]
