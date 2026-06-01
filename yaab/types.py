"""Core shared types: messages, usage, run context, events, and results.

These are deliberately framework-neutral Pydantic models so they serialize
cleanly into sessions, checkpoints, and the audit log.
"""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field

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
    cost_usd: float = 0.0

    def add(self, other: Usage) -> None:
        self.requests += other.requests
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.total_tokens += other.total_tokens
        self.cost_usd += other.cost_usd


class RunContext(Generic[Deps]):
    """Typed, dependency-injected context handed to tools and instructions.

    Holds the caller-supplied ``deps`` (the DI payload), identity, the live
    usage counter, and a scratch ``state`` dict. Modeled on Pydantic AI's
    ``RunContext`` — it keeps tools testable and free of global state.
    """

    __slots__ = ("deps", "session_id", "identity", "usage", "state", "run_id")

    def __init__(
        self,
        deps: Deps = None,  # type: ignore[assignment]
        *,
        session_id: str | None = None,
        identity: str | None = None,
        usage: Usage | None = None,
        state: dict[str, Any] | None = None,
    ) -> None:
        self.deps = deps
        self.session_id = session_id
        self.identity = identity
        self.usage = usage or Usage()
        self.state = state if state is not None else {}
        self.run_id = f"run_{uuid.uuid4().hex[:12]}"


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
    GUARDRAIL = "guardrail"
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


class RunResult(BaseModel, Generic[Output]):
    """The result of an agent run."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    output: Output
    messages: list[Message] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    events: list[Event] = Field(default_factory=list)
    run_id: str = ""

    @property
    def all_messages(self) -> list[Message]:
        return self.messages
