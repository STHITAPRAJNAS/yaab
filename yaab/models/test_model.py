"""Deterministic models for testing — no API keys, no network.

``TestModel`` returns canned text or a scripted sequence of responses, and can
auto-call tools so the agent loop can be exercised end to end. ``FunctionModel``
lets a test author compute the response from the conversation.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable
from typing import Any

from ..types import Message, ToolCall, Usage
from .base import ModelResponse, StreamChunk


class TestModel:
    """A scripted, deterministic model.

    Parameters
    ----------
    custom_output:
        Text to return as the final assistant message.
    responses:
        A sequence of :class:`ModelResponse` (or strings) returned in order,
        one per ``complete`` call. Overrides ``custom_output`` when set.
    call_tools:
        Tool names to call (with empty args) on the first response, before
        producing a final text answer. Useful for exercising the tool loop.
    structured_output:
        A dict returned (JSON-encoded) when the agent requests structured
        output, so output validation can be tested without a real model.
    """

    __test__ = False  # tell pytest this is not a test class

    def __init__(
        self,
        custom_output: str = "test-response",
        *,
        responses: list[ModelResponse | str] | None = None,
        call_tools: list[str] | None = None,
        structured_output: dict[str, Any] | None = None,
        reasoning: str | None = None,
    ) -> None:
        self.name = "test"
        self.custom_output = custom_output
        self.responses = responses
        self.call_tools = call_tools or []
        self.structured_output = structured_output
        self.reasoning = reasoning
        self._index = 0
        self.calls: list[list[Message]] = []
        #: Records the tool_choice passed on each complete() call, for assertions.
        self.tool_choices: list[Any] = []
        #: Records the extra provider kwargs (model_settings) per call.
        self.call_kwargs: list[dict[str, Any]] = []
        self._tools_called = False

    async def complete(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        output_schema: dict[str, Any] | None = None,
        tool_choice: Any | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        self.calls.append(list(messages))
        self.tool_choices.append(tool_choice)
        self.call_kwargs.append(dict(kwargs))
        usage = Usage(requests=1, input_tokens=10, output_tokens=5, total_tokens=15)

        if self.responses is not None:
            item = self.responses[min(self._index, len(self.responses) - 1)]
            self._index += 1
            if isinstance(item, str):
                return ModelResponse(content=item, usage=usage, model="test")
            item.usage = usage
            return item

        if self.call_tools and not self._tools_called:
            self._tools_called = True
            return ModelResponse(
                tool_calls=[ToolCall(name=name, arguments={}) for name in self.call_tools],
                finish_reason="tool_calls",
                usage=usage,
                model="test",
                reasoning=self.reasoning,
            )

        if output_schema is not None and self.structured_output is not None:
            return ModelResponse(
                content=json.dumps(self.structured_output), usage=usage, model="test"
            )

        return ModelResponse(
            content=self.custom_output, usage=usage, model="test", reasoning=self.reasoning
        )

    async def stream(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        resp = await self.complete(messages, tools=tools, **kwargs)
        # Surface tool calls so the streaming tool loop can drive them; otherwise
        # stream the text answer token-by-token.
        if resp.tool_calls:
            for tc in resp.tool_calls:
                yield StreamChunk(tool_call=tc)
        else:
            # Tokenize so the deltas reconstruct the content exactly (no spurious
            # trailing space): first word as-is, each subsequent prefixed by space.
            for i, word in enumerate(resp.content.split(" ")):
                yield StreamChunk(delta=word if i == 0 else " " + word)
        yield StreamChunk(done=True)


class FunctionModel:
    """A model whose response is computed by a user-supplied function."""

    def __init__(self, fn: Callable[[list[Message]], str | ModelResponse]) -> None:
        self.name = "function"
        self.fn = fn

    async def complete(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        output_schema: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        result = self.fn(messages)
        usage = Usage(requests=1, input_tokens=10, output_tokens=5, total_tokens=15)
        if isinstance(result, ModelResponse):
            result.usage = usage
            return result
        return ModelResponse(content=result, usage=usage, model="function")

    async def stream(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        resp = await self.complete(messages, tools=tools, **kwargs)
        yield StreamChunk(delta=resp.content)
        yield StreamChunk(done=True)
