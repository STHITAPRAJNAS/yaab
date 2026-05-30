"""YAAB exception hierarchy.

All SDK errors derive from :class:`YaabError` so callers can catch the whole
family with one ``except``.
"""

from __future__ import annotations


class YaabError(Exception):
    """Base class for every YAAB error."""


class ModelError(YaabError):
    """Raised when the model layer fails (provider error, bad response)."""


class OutputValidationError(YaabError):
    """Raised when a model's structured output fails schema validation.

    The runner uses this to drive reflection/retry: the validation message is
    fed back to the model so it can correct itself.
    """

    def __init__(self, message: str, *, attempts: int = 0) -> None:
        super().__init__(message)
        self.attempts = attempts


class ToolError(YaabError):
    """Raised when a tool cannot be found, validated, or executed."""


class MaxStepsExceeded(YaabError):
    """Raised when the agent loop exceeds its configured step budget."""


class GovernanceError(YaabError):
    """Base for governance/registry/policy failures."""


class PolicyViolation(GovernanceError):
    """Raised when a guardrail blocks a request or response."""

    def __init__(self, message: str, *, scanner: str, stage: str) -> None:
        super().__init__(message)
        self.scanner = scanner
        self.stage = stage


class NotRegisteredError(GovernanceError):
    """Raised in enforcing mode when an unregistered/unapproved agent runs."""


class LifecycleError(GovernanceError):
    """Raised on an illegal lifecycle state transition."""


class Interrupt(YaabError):
    """Raised by a graph node to pause execution for human-in-the-loop.

    The graph runtime catches this, checkpoints, and surfaces the payload to
    the caller; execution resumes from the same node on the next invocation.
    """

    def __init__(self, value: object) -> None:
        super().__init__("graph interrupted for human input")
        self.value = value
