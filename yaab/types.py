"""Core shared types: messages, usage, run context, events, and results.

These are deliberately framework-neutral Pydantic models so they serialize
cleanly into sessions, checkpoints, and the audit log.
"""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from .state import ReadonlyState, State  # noqa: F401

Deps = TypeVar("Deps")
Output = TypeVar("Output")


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ToolCall(BaseModel):
    """A model's request to invoke a tool."""

    id: str = Field(default_factory=lambda: f"call_{uuid.uuid4().hex[:12]}")
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class Message(BaseModel):
    """A single conversation message in the canonical OpenAI-ish shape."""

    role: Role
    content: str = ""
    name: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_call_id: str | None = None
    # Pre-rendered multimodal content items (set when the message carries a
    # multimodal Content); when present they take precedence over `content`.
    content_parts: list[Any] | None = None
    timestamp: float = Field(default_factory=time.time)

    def to_provider_dict(self) -> dict[str, Any]:
        """Render to the dict shape LiteLLM/OpenAI expect."""
        body: Any = self.content_parts if self.content_parts is not None else self.content
        msg: dict[str, Any] = {"role": self.role.value, "content": body}
        if self.name:
            msg["name"] = self.name
        if self.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": _json_args(tc.arguments)},
                }
                for tc in self.tool_calls
            ]
        if self.tool_call_id:
            msg["tool_call_id"] = self.tool_call_id
        return msg


def _json_args(args: dict[str, Any]) -> str:
    import json

    return json.dumps(args)


class Usage(BaseModel):
    """Token and cost accounting for a run, aggregated across model calls."""

    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    #: Prompt tokens served from the provider's prompt cache (subset of
    #: input_tokens) — billed cheaper; surfaced for cost attribution.
    cached_input_tokens: int = 0
    cost_usd: float = 0.0

    def add(self, other: Usage) -> None:
        self.requests += other.requests
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.total_tokens += other.total_tokens
        self.cached_input_tokens += other.cached_input_tokens
        self.cost_usd += other.cost_usd


class RunContext(Generic[Deps]):
    """Typed, dependency-injected context handed to tools and instructions.

    Holds the caller-supplied ``deps`` (the DI payload), identity, the live
    usage counter, and the run's shared ``state``. ``state`` is a prefix-scoped
    :class:`~yaab.state.State` — one object per run, shared by every agent,
    sub-agent, workflow child, tool, and plugin, so a value written in one place
    is read in another by key. Passing context explicitly keeps tools testable
    and free of global state.

    For backward compatibility ``state`` still behaves exactly like a dict
    (``ctx.state["k"]``, ``.get``, ``.setdefault``, ``in``, iteration); the only
    addition is that ``temp:``/``user:``/``app:`` prefixed keys route to their
    own scope. A bare ``dict`` passed in is adopted as the session scope.
    """

    __slots__ = (
        "deps",
        "session_id",
        "identity",
        "usage",
        "state",
        "run_id",
        # Reserved for the human-in-the-loop pause surface (set by the runner to
        # a callable during a run; ``None`` outside one). Reserved here so later
        # pillars need not reopen ``__slots__`` — adding a slot later is a
        # breaking change to every RunContext constructor.
        "pause_for",
        # Resume value injected on a resumed run (set by the runner).
        "_resume",
    )

    def __init__(
        self,
        deps: Deps = None,  # type: ignore[assignment]
        *,
        session_id: str | None = None,
        identity: str | None = None,
        usage: Usage | None = None,
        state: State | dict[str, Any] | None = None,
    ) -> None:
        from .state import State

        self.deps = deps
        self.session_id = session_id
        self.identity = identity
        self.usage = usage or Usage()
        # Accept a shared State, wrap a bare dict as the session scope
        # (back-compat), or build a fresh run-local State.
        if isinstance(state, State):
            self.state: State = state
        elif isinstance(state, dict):
            self.state = State(session=state)
        else:
            self.state = State()
        self.run_id = f"run_{uuid.uuid4().hex[:12]}"
        self.pause_for: Any = None
        self._resume: Any = None

    def readonly(self) -> RunContextView:
        """A read-only projection for instruction rendering and routing.

        Same ``deps``/``identity``/``usage``, but ``.state`` is a
        :class:`~yaab.state.ReadonlyState` so rendering and routing predicates
        physically cannot mutate shared state.
        """
        return RunContextView(self)


class RunContextView(Generic[Deps]):
    """A read-only projection of a :class:`RunContext`.

    Exposes the same fields, but ``state`` is an immutable
    :class:`~yaab.state.ReadonlyState`. Handed to instruction providers and
    routing predicates — the surfaces where a write would be a bug.
    """

    __slots__ = ("_ctx", "state")

    def __init__(self, ctx: RunContext[Deps]) -> None:
        from .state import ReadonlyState

        self._ctx = ctx
        self.state = ReadonlyState(ctx.state)

    @property
    def deps(self) -> Deps:
        return self._ctx.deps

    @property
    def session_id(self) -> str | None:
        return self._ctx.session_id

    @property
    def identity(self) -> str | None:
        return self._ctx.identity

    @property
    def usage(self) -> Usage:
        return self._ctx.usage

    @property
    def run_id(self) -> str:
        return self._ctx.run_id


class EventType(str, Enum):
    RUN_START = "run_start"
    USER_MESSAGE = "user_message"
    MODEL_REQUEST = "model_request"
    MODEL_DELTA = "model_delta"
    #: A token-level text delta during a streaming run (Runner.stream_run).
    TEXT_DELTA = "text_delta"
    MODEL_RESPONSE = "model_response"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    #: A framework-managed handoff to a sub-agent (transfer_to_agent). The
    #: payload carries ``to`` (the sub-agent name); the sub-agent's answer then
    #: becomes the parent run's output.
    AGENT_TRANSFER = "agent_transfer"
    GUARDRAIL = "guardrail"
    #: An agent (or workflow step) captured its validated output into shared state
    #: under its ``writes=`` key. The payload carries ``key`` (the state key,
    #: including any ``temp:``/``user:``/``app:`` scope prefix) and ``scope`` (the
    #: resolved scope name) so the inter-agent handoff is visible in the trace.
    STATE_WRITE = "state_write"
    #: An instruction's ``{key}`` placeholders were substituted from shared state
    #: before the model call. The payload carries ``keys`` (the resolved field
    #: names) so a reader can see exactly which state values shaped the prompt.
    STATE_TEMPLATE = "state_template"
    #: A sensitive tool call has been parked for out-of-band human sign-off; the
    #: run is durably paused and resumes once a reviewer decides. The payload
    #: carries ``approval_id``, ``tool``, and ``arguments``.
    APPROVAL_REQUIRED = "approval_required"
    #: A ``when=`` input guard was false, so a unit was skipped. The payload
    #: carries the condition label, the boolean result, and the resolved operand
    #: values (``operands``) so a reader can answer *why* it was skipped without
    #: re-running.
    CONDITION_SKIP = "condition_skip"
    #: A ``stop=`` output guard fired, so the enclosing pattern stopped. Same
    #: payload shape (label/result/operands) as the skip event.
    CONDITION_STOP = "condition_stop"
    #: An ``else=`` fallback unit ran in place of a unit that was skipped, failed,
    #: or timed out. The payload's ``status`` distinguishes the three triggers.
    CONDITION_FALLBACK = "condition_fallback"
    #: A :class:`~yaab.multiagent.RouterAgent` is about to evaluate its branches;
    #: the payload carries the static branch set and the no-match policy.
    ROUTER_EVALUATED = "router_evaluated"
    #: A router chose a branch (or the default, or matched nothing). The payload
    #: carries the chosen branch name, its index, and the resolved operands.
    ROUTER_MATCHED = "router_matched"
    FINAL_OUTPUT = "final_output"
    RUN_END = "run_end"
    ERROR = "error"


class Event(BaseModel):
    """An item in the Runner's typed event stream."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    type: EventType
    agent: str
    run_id: str
    payload: dict[str, Any] = Field(default_factory=dict)
    timestamp: float = Field(default_factory=time.time)
    #: Wall-clock time this step took (model call, tool call, or whole run), in
    #: milliseconds. Populated for timed steps so the trace UI need not infer it
    #: from timestamps; ``None`` for events that don't measure a span.
    duration_ms: float | None = None


class Pending(BaseModel):
    """One thing a human must decide before a paused run continues.

    A paused :class:`RunResult` carries a *list* of these (one element for the
    common single-approval case, several when one model turn guarded two tools).
    One shape covers all three pause kinds: a guarded tool (``"approval"``), an
    :func:`~yaab.tools.builtin.ask_user` question (``"question"``), or a graph
    ``interrupt()`` (``"flow_pause"``). Every kind carries the same correlation
    keys (``approval_id``/``run_id``/``resume_id``) so resume is uniform, with
    kind-specific fields left nullable.
    """

    kind: Literal["approval", "question", "flow_pause"]
    # --- correlation (every kind) ---
    #: == ``ApprovalRequest.approval_id`` — the durable resume key a human cites.
    approval_id: str = ""
    run_id: str = ""
    #: The checkpoint key the loop resumes from.
    resume_id: str = ""
    # --- approval kind ---
    #: The guarded tool awaiting sign-off.
    tool: str | None = None
    #: The proposed (editable) tool arguments.
    arguments: dict[str, Any] = Field(default_factory=dict)
    # --- question kind (ask_user) ---
    #: The question text shown to the human.
    prompt: str | None = None
    #: A JSON Schema the typed answer is validated against (if declared).
    answer_schema: dict[str, Any] | None = None
    # --- flow_pause kind (graph interrupt) ---
    #: The value passed to ``interrupt()``.
    payload: Any = None
    # --- lifecycle (any kind) ---
    #: An optional business key (e.g. ``"customer:42"``) for key-based lookup.
    correlation_key: str | None = None
    #: A timeout deadline (epoch seconds), if one was set.
    expires_at: float | None = None


class RunResult(BaseModel, Generic[Output]):
    """The result of an agent run.

    ``output`` is defined and meaningful **iff** ``not paused``. When a run pauses
    for human sign-off (an approval gate, a question, or an explicit step pause)
    it returns with ``paused=True``; ``pause_value`` carries what the single
    blocking human decision is, and ``pending`` lists *every* parked decision when
    more than one is outstanding at once (e.g. concurrent branches each awaiting a
    reviewer). ``output`` is then a placeholder (``None`` for the common case) and
    should not be read. Resume the same run to get a final, non-paused result.
    Keeping one result type with this documented invariant avoids fragmenting
    result shapes or weakening ``output``'s type for the common case.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    output: Output
    #: The terminal status of the unit that produced this result: ``"ok"`` |
    #: ``"skipped"`` | ``"failed"`` | ``"timeout"``. ``output`` is meaningful
    #: only when this is ``"ok"`` (and ``not paused``). A unit skipped by a
    #: ``when=`` guard returns its pass-through input with status ``"skipped"``;
    #: the status channel is orthogonal to the output value. Stored as a plain
    #: string (the :class:`~yaab.conditions.Status` enum is a ``str`` subclass,
    #: so ``result.status == Status.SKIPPED`` compares cleanly) to keep this core
    #: type free of an import cycle.
    status: str = "ok"
    messages: list[Message] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    events: list[Event] = Field(default_factory=list)
    run_id: str = ""
    #: ``True`` when the run is durably paused awaiting a human decision. Read
    #: ``output`` only when this is ``False``.
    paused: bool = False
    #: When ``paused``, the payload describing what a human must decide (e.g. the
    #: parked tool call and its arguments). ``None`` on a normal completion.
    pause_value: Any = None
    #: When ``paused``, the list of all outstanding human decisions for this run.
    #: A run may pause on more than one at once (concurrent branches each parking
    #: an approval/question), so this is the complete set a caller iterates to
    #: resolve them; ``pause_value`` is the single/primary one for the common
    #: case. Empty on a normal completion. ``result.paused == bool(pending)``.
    pending: list[Pending] = Field(default_factory=list)

    @property
    def all_messages(self) -> list[Message]:
        return self.messages
