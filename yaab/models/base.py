"""Model layer protocol and shared response types.

A :class:`ModelProvider` is the only thing the runtime needs from a model: a
way to turn a list of messages (plus optional tool schemas and a structured
output schema) into a :class:`ModelResponse`. LiteLLM, TestModel, and any
custom client all implement this one protocol.
"""

from __future__ import annotations

from typing import Any, AsyncIterator, Optional, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from ..types import Message, ToolCall, Usage


class ModelResponse(BaseModel):
    """A normalized model response in OpenAI-ish shape."""

    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    finish_reason: str = "stop"
    usage: Usage = Field(default_factory=Usage)
    model: str = ""
    raw: Optional[dict[str, Any]] = None

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


class StreamChunk(BaseModel):
    """An incremental delta during streaming."""

    delta: str = ""
    tool_call: Optional[ToolCall] = None
    done: bool = False


@runtime_checkable
class ModelProvider(Protocol):
    """The pluggable model interface."""

    name: str

    async def complete(
        self,
        messages: list[Message],
        *,
        tools: Optional[list[dict[str, Any]]] = None,
        output_schema: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> ModelResponse:
        """Return a single completion for ``messages``."""
        ...

    def stream(
        self,
        messages: list[Message],
        *,
        tools: Optional[list[dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Yield incremental chunks for ``messages``."""
        ...
