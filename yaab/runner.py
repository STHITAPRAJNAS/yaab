"""The Runner — the orchestration engine for the model-driven fast path.

Runs the ReAct-style loop: call the model, execute any requested tools, feed
results back, repeat until a final (optionally typed-and-validated) output. It
owns the session/memory/artifact services, the plugin chain, and the optional
governance service, and yields a typed event stream.

The Runner is the deterministic seam between the developer API and the engine:
the same loop backs ``Agent.run`` and is reused by the graph and optimizer
layers.
"""

from __future__ import annotations

import json
import time
from typing import Any, AsyncIterator, Optional

from pydantic import TypeAdapter, ValidationError

from .exceptions import MaxStepsExceeded, ToolError
from .governance.audit import AuditKind
from .governance.policy import Stage
from .governance.service import GovernanceService
from .limits import CancellationToken, UsageLimits
from .models.base import ModelResponse
from .plugins import Plugin
from .sessions.base import SessionService
from .sessions.memory import InMemorySessionService
from .types import (
    Event,
    EventType,
    Message,
    Role,
    RunContext,
    RunResult,
    ToolCall,
    Usage,
)


class Runner:
    """Drives agents; holds services, plugins, and governance."""

    def __init__(
        self,
        *,
        session_service: Optional[SessionService] = None,
        memory_service: Optional[Any] = None,
        artifact_service: Optional[Any] = None,
        governance: Optional[GovernanceService] = None,
        plugins: Optional[list[Plugin]] = None,
    ) -> None:
        self.session_service = session_service or InMemorySessionService()
        self.memory_service = memory_service
        self.artifact_service = artifact_service
        self.governance = governance
        self.plugins: list[Plugin] = list(plugins or [])

    def add_plugin(self, plugin: Plugin) -> "Runner":
        self.plugins.append(plugin)
        return self

    # ------------------------------------------------------------------
    async def run(
        self,
        agent: Any,
        prompt: str,
        *,
        deps: Any = None,
        session_id: Optional[str] = None,
        identity: Optional[str] = None,
        usage_limits: Optional["UsageLimits"] = None,
        cancellation: Optional["CancellationToken"] = None,
        timeout: Optional[float] = None,
    ) -> RunResult[Any]:
        events: list[Event] = []
        async for event in self.run_stream(
            agent,
            prompt,
            deps=deps,
            session_id=session_id,
            identity=identity,
            usage_limits=usage_limits,
            cancellation=cancellation,
            timeout=timeout,
        ):
            events.append(event)
        final = events[-1]
        if final.type is EventType.ERROR:
            raise final.payload["error"]
        result: RunResult = final.payload["result"]
        result.events = events
        return result

    def run_sync(self, agent: Any, prompt: str, **kwargs: Any) -> RunResult[Any]:
        import asyncio

        return asyncio.run(self.run(agent, prompt, **kwargs))

    # ------------------------------------------------------------------
    async def run_stream(
        self,
        agent: Any,
        prompt: str,
        *,
        deps: Any = None,
        session_id: Optional[str] = None,
        identity: Optional[str] = None,
        usage_limits: Optional["UsageLimits"] = None,
        cancellation: Optional["CancellationToken"] = None,
        timeout: Optional[float] = None,
    ) -> AsyncIterator[Event]:
        """Execute the loop, yielding a typed :class:`Event` per step."""
        ctx: RunContext = RunContext(
            deps=deps, session_id=session_id, identity=identity, usage=Usage()
        )
        gov = self.governance

        # A timeout is just a deadline on the cancellation token.
        if timeout is not None:
            if cancellation is None:
                cancellation = CancellationToken.with_timeout(timeout)
            elif cancellation.deadline is None:
                cancellation.deadline = time.monotonic() + timeout
        tool_counts: dict[str, int] = {}

        def check_controls() -> None:
            if cancellation is not None:
                cancellation.raise_if_cancelled()
            if usage_limits is not None:
                usage_limits.check_usage(ctx.usage)

        def emit(etype: EventType, **payload: Any) -> Event:
            return Event(type=etype, agent=agent.name, run_id=ctx.run_id, payload=payload)

        try:
            # Registry gate (enforcing mode).
            if gov is not None:
                gov.check_registered(agent.registry_id, identity)
                gov.record_run_start(agent.registry_id, identity, prompt)

            yield emit(EventType.RUN_START, prompt=prompt)

            for plugin in self.plugins:
                await plugin.before_run(ctx, agent.name, prompt)

            # Input guardrails (scan the text; multimodal parts pass through).
            from .content import Content

            prompt_text = prompt.text if isinstance(prompt, Content) else str(prompt)
            scanned_prompt = prompt_text
            if gov is not None:
                scanned_prompt = gov.scan(
                    prompt_text, Stage.INPUT, agent_id=agent.registry_id, identity=identity
                )

            messages = await self._build_messages(
                agent, ctx, scanned_prompt, original=prompt
            )
            user_msg = messages[-1]
            yield emit(EventType.USER_MESSAGE, content=scanned_prompt)
            for plugin in self.plugins:
                await plugin.on_user_message(ctx, agent.name, user_msg)

            tool_schemas = [t.schema() for t in agent.tools] or None
            output_adapter, output_schema = _output_spec(agent.output_type)
            tool_choice = _normalize_tool_choice(getattr(agent, "tool_choice", None), tool_schemas)

            final_output: Any = None
            produced = False

            context_strategy = getattr(agent, "context_strategy", None)

            for _step in range(agent.max_steps):
                check_controls()  # cancellation / timeout / usage caps
                # Keep the conversation within the model's context window.
                if context_strategy is not None:
                    messages = await context_strategy.apply(messages, model=agent.model)
                response = await self._call_model(
                    agent, ctx, messages, tool_schemas, output_schema, tool_choice
                )
                # Re-check usage now that this request's tokens are counted.
                if usage_limits is not None:
                    usage_limits.check_usage(ctx.usage)
                if response.reasoning:
                    yield emit(EventType.MODEL_DELTA, reasoning=response.reasoning)
                yield emit(
                    EventType.MODEL_RESPONSE,
                    content=response.content,
                    tool_calls=[tc.model_dump() for tc in response.tool_calls],
                )

                if response.has_tool_calls:
                    messages.append(
                        Message(role=Role.ASSISTANT, content=response.content,
                                tool_calls=response.tool_calls)
                    )
                    for tc in response.tool_calls:
                        if cancellation is not None:
                            cancellation.raise_if_cancelled()
                        tool_counts[tc.name] = tool_counts.get(tc.name, 0) + 1
                        if usage_limits is not None:
                            usage_limits.check_tool_call(tc.name, tool_counts)
                        yield emit(EventType.TOOL_CALL, name=tc.name, arguments=tc.arguments)
                        result_value = await self._run_tool(agent, ctx, tc)
                        yield emit(EventType.TOOL_RESULT, name=tc.name, result=_safe(result_value))
                        messages.append(
                            Message(
                                role=Role.TOOL,
                                name=tc.name,
                                tool_call_id=tc.id,
                                content=_to_text(result_value),
                            )
                        )
                    continue

                # No tool calls -> attempt to finalize.
                try:
                    final_output = self._coerce_output(
                        agent, response.content, output_adapter
                    )
                    produced = True
                    break
                except ValidationError as exc:
                    # Reflection/retry: feed the validation error back to the model.
                    if agent.output_retries <= 0:
                        raise
                    agent.output_retries -= 1
                    messages.append(Message(role=Role.ASSISTANT, content=response.content))
                    messages.append(
                        Message(
                            role=Role.USER,
                            content=(
                                "Your previous response did not match the required schema. "
                                f"Validation error: {exc}. Respond again with valid JSON only."
                            ),
                        )
                    )
                    continue

            if not produced:
                raise MaxStepsExceeded(
                    f"agent '{agent.name}' did not finish within {agent.max_steps} steps"
                )

            # Output guardrails.
            text_out = _to_text(final_output)
            if gov is not None:
                scanned = gov.scan(
                    text_out, Stage.OUTPUT, agent_id=agent.registry_id, identity=identity
                )
                if scanned != text_out and isinstance(final_output, str):
                    final_output = scanned

            yield emit(EventType.FINAL_OUTPUT, output=_safe(final_output))

            for plugin in self.plugins:
                await plugin.after_run(ctx, agent.name, final_output)

            await self._persist(agent, ctx, scanned_prompt, final_output)
            if gov is not None:
                gov.record_run_end(agent.registry_id, identity, text_out)

            result = RunResult(
                output=final_output,
                messages=messages,
                usage=ctx.usage,
                run_id=ctx.run_id,
            )
            yield emit(EventType.RUN_END, result=result)

        except Exception as exc:  # noqa: BLE001 - surface as a terminal ERROR event
            if gov is not None:
                gov.audit.record(
                    AuditKind.ERROR,
                    agent_id=agent.registry_id,
                    identity=identity,
                    error=str(exc),
                )
            yield emit(EventType.ERROR, error=exc)

    # ------------------------------------------------------------------
    async def stream_text(
        self,
        agent: Any,
        prompt: Any,
        *,
        deps: Any = None,
        session_id: Optional[str] = None,
        identity: Optional[str] = None,
    ) -> AsyncIterator[str]:
        """Token-level streaming for a single answering turn (no tool loop).

        Yields text deltas as they arrive from the model. Use this for chat-style
        streaming UX; use :meth:`run_stream` when you need the full tool loop and
        semantic events.
        """
        from .content import Content

        ctx: RunContext = RunContext(deps=deps, session_id=session_id, identity=identity)
        prompt_text = prompt.text if isinstance(prompt, Content) else str(prompt)
        if self.governance is not None:
            prompt_text = self.governance.scan(
                prompt_text, Stage.INPUT, agent_id=agent.registry_id, identity=identity
            )
        messages = await self._build_messages(agent, ctx, prompt_text, original=prompt)
        async for chunk in agent.model.stream(messages):
            if chunk.delta:
                yield chunk.delta

    # ------------------------------------------------------------------
    async def _build_messages(
        self, agent: Any, ctx: RunContext, prompt: str, *, original: Any = None
    ) -> list[Message]:
        messages: list[Message] = []
        instructions = agent.instructions
        if callable(instructions):
            instructions = instructions(ctx)
        if instructions:
            messages.append(Message(role=Role.SYSTEM, content=str(instructions)))

        # Replay prior session history if a durable session exists.
        if ctx.session_id is not None:
            session = await self.session_service.get(ctx.session_id)
            if session is not None:
                messages.extend(session.messages)

        # Optionally fold in retrieved long-term memory.
        if self.memory_service is not None:
            hits = await self.memory_service.search(prompt, k=3)
            if hits:
                recalled = "\n".join(f"- {rec.text}" for rec, _ in hits)
                messages.append(
                    Message(role=Role.SYSTEM, content=f"Relevant memory:\n{recalled}")
                )

        messages.append(_user_message(prompt, original))
        return messages

    async def _call_model(
        self,
        agent: Any,
        ctx: RunContext,
        messages: list[Message],
        tool_schemas: Optional[list[dict[str, Any]]],
        output_schema: Optional[dict[str, Any]],
        tool_choice: Optional[Any] = None,
    ) -> ModelResponse:
        for plugin in self.plugins:
            short = await plugin.before_model(ctx, agent.name, messages)
            if short is not None:
                ctx.usage.add(short.usage)
                return short
        response = await agent.model.complete(
            messages, tools=tool_schemas, output_schema=output_schema, tool_choice=tool_choice
        )
        ctx.usage.add(response.usage)
        for plugin in self.plugins:
            amended = await plugin.after_model(ctx, agent.name, response)
            if amended is not None:
                response = amended
        return response

    async def _run_tool(self, agent: Any, ctx: RunContext, tc: ToolCall) -> Any:
        # Let plugins repair/coerce raw args before validation/execution.
        for plugin in self.plugins:
            repaired = await plugin.repair_tool_args(ctx, agent.name, tc.name, tc.arguments)
            if repaired is not None:
                tc.arguments = repaired
        for plugin in self.plugins:
            short = await plugin.before_tool(ctx, agent.name, tc.name, tc.arguments)
            if short is not None:
                return short
        tool = next((t for t in agent.tools if t.name == tc.name), None)
        if tool is None:
            return f"error: unknown tool '{tc.name}'"
        try:
            result = await tool.execute(ctx, **tc.arguments)
        except ToolError as exc:
            result = f"error: {exc}"
        except Exception as exc:  # noqa: BLE001 - tools shouldn't crash the loop
            result = f"error: tool '{tc.name}' raised {type(exc).__name__}: {exc}"
        for plugin in self.plugins:
            amended = await plugin.after_tool(ctx, agent.name, tc.name, result)
            if amended is not None:
                result = amended
        return result

    def _coerce_output(self, agent: Any, content: str, adapter: Optional[TypeAdapter]) -> Any:
        if adapter is None:
            return content
        return adapter.validate_json(content)

    async def _persist(
        self, agent: Any, ctx: RunContext, prompt: str, output: Any
    ) -> None:
        if ctx.session_id is None:
            return
        await self.session_service.append(
            ctx.session_id, Message(role=Role.USER, content=prompt)
        )
        await self.session_service.append(
            ctx.session_id, Message(role=Role.ASSISTANT, content=_to_text(output))
        )


def _user_message(text: str, original: Any = None) -> Message:
    """Build the user message, preserving multimodal parts from a Content."""
    from .content import Content

    if isinstance(original, Content) and original.is_multimodal():
        return Message(
            role=Role.USER, content=text, content_parts=original.to_provider_content()
        )
    return Message(role=Role.USER, content=text)


def _normalize_tool_choice(choice: Any, tool_schemas: Optional[list[dict[str, Any]]]) -> Any:
    """Normalize a tool_choice into the provider form.

    Passes through ``None``/``"auto"``/``"required"``/``"none"`` and dicts. A bare
    string naming one of the agent's tools is expanded to the OpenAI
    ``{"type": "function", "function": {"name": ...}}`` form. Ignored entirely
    when the agent has no tools.
    """
    if choice is None or not tool_schemas:
        return None if not tool_schemas else choice
    if isinstance(choice, dict) or choice in ("auto", "required", "none"):
        return choice
    if isinstance(choice, str):
        known = {s["function"]["name"] for s in tool_schemas if "function" in s}
        if choice in known:
            return {"type": "function", "function": {"name": choice}}
    return choice


def _output_spec(output_type: type) -> tuple[Optional[TypeAdapter], Optional[dict[str, Any]]]:
    """Build a validator + JSON schema for non-string output types."""
    if output_type is str or output_type is type(None):
        return None, None
    adapter = TypeAdapter(output_type)
    try:
        schema = adapter.json_schema()
    except Exception:  # noqa: BLE001
        schema = None
    return adapter, schema


def _to_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        if hasattr(value, "model_dump"):
            return json.dumps(value.model_dump())
        return json.dumps(value)
    except (TypeError, ValueError):
        return str(value)


def _safe(value: Any) -> Any:
    """Make a value JSON-safe for embedding in an Event payload."""
    if isinstance(value, (str, int, float, bool, type(None), dict, list)):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return str(value)
