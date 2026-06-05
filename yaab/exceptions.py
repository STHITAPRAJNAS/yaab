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


class UsageLimitExceeded(YaabError):
    """Raised when a run exceeds a configured usage limit (tokens/requests/tools)."""

    def __init__(self, message: str, *, limit: str) -> None:
        super().__init__(message)
        self.limit = limit


class RunCancelled(YaabError):
    """Raised when a run is cancelled via a CancellationToken or times out."""

    def __init__(self, message: str = "run cancelled", *, reason: str = "cancelled") -> None:
        super().__init__(message)
        self.reason = reason


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


class ApprovalRequired(GovernanceError):
    """Raised when a tool call needs human approval that wasn't granted.

    Carries the pending tool call so an out-of-band approval flow can surface it
    to a human and resume.
    """

    def __init__(self, tool: str, arguments: dict, *, reason: str = "approval required") -> None:
        super().__init__(f"approval required for tool '{tool}': {reason}")
        self.tool = tool
        self.arguments = arguments
        self.reason = reason


class ApprovalPending(ApprovalRequired):
    """Raised when a sensitive tool call has been parked for out-of-band sign-off.

    Unlike a bare :class:`ApprovalRequired`, this carries the correlation ids the
    run needs to durably park and later resume: the ``approval_id`` of the stored
    request, the ``run_id`` it belongs to, and the ``resume_id`` (checkpoint key)
    the loop resumes from once a reviewer decides. It subclasses
    :class:`ApprovalRequired` so existing ``except ApprovalRequired`` handlers keep
    working unchanged.
    """

    def __init__(
        self,
        tool: str,
        arguments: dict,
        *,
        approval_id: str,
        run_id: str,
        resume_id: str,
        reason: str = "approval required",
        kind: str = "approval",
        prompt: str | None = None,
        answer_schema: dict | None = None,
        correlation_key: str | None = None,
        expires_at: float | None = None,
    ) -> None:
        super().__init__(tool, arguments, reason=reason)
        self.approval_id = approval_id
        self.run_id = run_id
        self.resume_id = resume_id
        #: Which pause source this is: ``"approval"`` | ``"question"`` |
        #: ``"flow_pause"`` — so the resume seam flows the decided value back the
        #: right way (run the tool, or return a typed answer).
        self.kind = kind
        #: The ``ask_user`` question text (``"question"`` kind only).
        self.prompt = prompt
        #: A JSON Schema the typed answer is validated against, if declared.
        self.answer_schema = answer_schema
        #: An optional business key for key-based lookup.
        self.correlation_key = correlation_key
        #: An optional timeout deadline (epoch seconds).
        self.expires_at = expires_at


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
