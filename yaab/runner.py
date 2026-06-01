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
from collections.abc import AsyncIterator
from typing import Any

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
        session_service: SessionService | None = None,
        memory_service: Any | None = None,
        artifact_service: Any | None = None,
        governance: GovernanceService | None = None,
        plugins: list[Plugin] | None = None,
        memory_app_name: str | None = None,
        default_tool_timeout: float | None = None,
    ) -> None:
        self.session_service = session_service or InMemorySessionService()
        self.memory_service = memory_service
        self.artifact_service = artifact_service
        self.governance = governance
        self.plugins: list[Plugin] = list(plugins or [])
        #: App scope passed to a namespace-aware memory backend's search.
        self.memory_app_name = memory_app_name
        #: Default per-tool execution timeout (seconds); a tool's own ``timeout``
        #: overrides it. ``None`` means no timeout.
        self.default_tool_timeout = default_tool_timeout

    def add_plugin(self, plugin: Plugin) -> Runner:
        self.plugins.append(plugin)
        return self

    # ------------------------------------------------------------------
    async def run(
        self,
        agent: Any,
        prompt: str,
        *,
        deps: Any = None,
        session_id: str | None = None,
        identity: str | None = None,
        usage_limits: UsageLimits | None = None,
        cancellation: CancellationToken | None = None,
        timeout: float | None = None,
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
        session_id: str | None = None,
        identity: str | None = None,
        usage_limits: UsageLimits | None = None,
        cancellation: CancellationToken | None = None,
        timeout: float | None = None,
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
        run_started = time.monotonic()

        def check_controls() -> None:
            if cancellation is not None:
                cancellation.raise_if_cancelled()
            if usage_limits is not None:
                usage_limits.check_usage(ctx.usage)
                usage_limits.check_wall_clock(run_started)

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

            messages = await self._build_messages(agent, ctx, scanned_prompt, original=prompt)
            user_msg = messages[-1]
            yield emit(EventType.USER_MESSAGE, content=scanned_prompt)
            for plugin in self.plugins:
                await plugin.on_user_message(ctx, agent.name, user_msg)

            tool_schemas = [t.schema() for t in agent.tools] or None
            output_adapter, output_schema = _output_spec(agent.output_type)
            tool_choice = _normalize_tool_choice(getattr(agent, "tool_choice", None), tool_schemas)
            # A forcing tool_choice ("required" or a pinned function) must apply to
            # the first model call only — otherwise every turn is forced to call a
            # tool and the model can never emit a final answer (infinite loop until
            # max_steps). After a tool round we relax it to "auto" so the loop can
            # finalize. "auto"/"none"/None are not forcing and pass through unchanged.
            effective_tool_choice = tool_choice

            final_output: Any = None
            produced = False
            # Per-run retry budget. Kept local so a reused agent isn't mutated
            # (the configured agent.output_retries must hold for every run).
            output_retries_left = agent.output_retries

            context_strategy = getattr(agent, "context_strategy", None)

            for _step in range(agent.max_steps):
                check_controls()  # cancellation / timeout / usage caps
                # Keep the conversation within the model's context window.
                if context_strategy is not None:
                    messages = await context_strategy.apply(messages, model=agent.model)
                response = await self._call_model(
                    agent, ctx, messages, tool_schemas, output_schema, effective_tool_choice
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
                        Message(
                            role=Role.ASSISTANT,
                            content=response.content,
                            tool_calls=response.tool_calls,
                        )
                    )
                    tcs = response.tool_calls
                    # Pre-flight in call order: cancellation, counts, usage caps.
                    for tc in tcs:
                        if cancellation is not None:
                            cancellation.raise_if_cancelled()
                        tool_counts[tc.name] = tool_counts.get(tc.name, 0) + 1
                        if usage_limits is not None:
                            usage_limits.check_tool_call(tc.name, tool_counts)

                    parallel = getattr(agent, "parallel_tools", True) and len(tcs) > 1
                    if parallel:
                        # Announce all calls (in order), run concurrently, then
                        # emit results (in order) so traces stay deterministic.
                        for tc in tcs:
                            yield emit(EventType.TOOL_CALL, name=tc.name, arguments=tc.arguments)
                        results = await self._run_tools_parallel(agent, ctx, tcs)
                        for tc, result_value in zip(tcs, results, strict=False):
                            yield emit(
                                EventType.TOOL_RESULT, name=tc.name, result=_safe(result_value)
                            )
                            messages.append(
                                Message(
                                    role=Role.TOOL,
                                    name=tc.name,
                                    tool_call_id=tc.id,
                                    content=_to_text(result_value),
                                )
                            )
                    else:
                        for tc in tcs:
                            if cancellation is not None:
                                cancellation.raise_if_cancelled()
                            yield emit(EventType.TOOL_CALL, name=tc.name, arguments=tc.arguments)
                            result_value = await self._run_tool(agent, ctx, tc)
                            yield emit(
                                EventType.TOOL_RESULT, name=tc.name, result=_safe(result_value)
                            )
                            messages.append(
                                Message(
                                    role=Role.TOOL,
                                    name=tc.name,
                                    tool_call_id=tc.id,
                                    content=_to_text(result_value),
                                )
                            )
                    # Forcing choice has done its job; let the next turn finalize.
                    if _is_forcing_tool_choice(effective_tool_choice):
                        effective_tool_choice = "auto"
                    continue

                # No tool calls -> attempt to finalize.
                try:
                    final_output = self._coerce_output(agent, response.content, output_adapter)
                    produced = True
                    break
                except ValidationError as exc:
                    # Reflection/retry: feed the validation error back to the model.
                    if output_retries_left <= 0:
                        raise
                    output_retries_left -= 1
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

            result: RunResult = RunResult(
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
    async def stream_run(
        self,
        agent: Any,
        prompt: str,
        *,
        deps: Any = None,
        session_id: str | None = None,
        identity: str | None = None,
        usage_limits: UsageLimits | None = None,
        cancellation: CancellationToken | None = None,
        timeout: float | None = None,
    ) -> AsyncIterator[Event]:
        """Run the full multi-step loop while streaming token deltas.

        Unlike :meth:`run_stream` (which calls the model with ``complete`` and so
        only emits whole responses), this drives each turn with ``model.stream``:
        text arrives as :attr:`EventType.TEXT_DELTA` events *as it generates*,
        tools execute mid-run (``TOOL_CALL``/``TOOL_RESULT``), and the loop
        continues until a final answer — the LangGraph/ADK "stream through the
        tool loop" behavior. Terminates with ``FINAL_OUTPUT`` then ``RUN_END``.
        """
        ctx: RunContext = RunContext(
            deps=deps, session_id=session_id, identity=identity, usage=Usage()
        )
        gov = self.governance
        if timeout is not None:
            if cancellation is None:
                cancellation = CancellationToken.with_timeout(timeout)
            elif cancellation.deadline is None:
                cancellation.deadline = time.monotonic() + timeout
        tool_counts: dict[str, int] = {}
        run_started = time.monotonic()

        def emit(etype: EventType, **payload: Any) -> Event:
            return Event(type=etype, agent=agent.name, run_id=ctx.run_id, payload=payload)

        try:
            if gov is not None:
                gov.check_registered(agent.registry_id, identity)
                gov.record_run_start(agent.registry_id, identity, prompt)
            yield emit(EventType.RUN_START, prompt=prompt)

            for plugin in self.plugins:
                await plugin.before_run(ctx, agent.name, prompt)

            from .content import Content

            prompt_text = prompt.text if isinstance(prompt, Content) else str(prompt)
            scanned_prompt = prompt_text
            if gov is not None:
                scanned_prompt = gov.scan(
                    prompt_text, Stage.INPUT, agent_id=agent.registry_id, identity=identity
                )
            messages = await self._build_messages(agent, ctx, scanned_prompt, original=prompt)
            yield emit(EventType.USER_MESSAGE, content=scanned_prompt)

            tool_schemas = [t.schema() for t in agent.tools] or None
            output_adapter, _ = _output_spec(agent.output_type)
            effective_tool_choice = _normalize_tool_choice(
                getattr(agent, "tool_choice", None), tool_schemas
            )
            context_strategy = getattr(agent, "context_strategy", None)
            final_output: Any = None
            produced = False

            for _step in range(agent.max_steps):
                if cancellation is not None:
                    cancellation.raise_if_cancelled()
                if usage_limits is not None:
                    usage_limits.check_usage(ctx.usage)
                    usage_limits.check_wall_clock(run_started)
                if context_strategy is not None:
                    messages = await context_strategy.apply(messages, model=agent.model)

                # Stream this turn: accumulate text + tool calls.
                text_parts: list[str] = []
                turn_tool_calls: list[ToolCall] = []
                stream_kwargs: dict[str, Any] = dict(getattr(agent, "model_settings", {}))
                if effective_tool_choice is not None and tool_schemas:
                    stream_kwargs["tool_choice"] = effective_tool_choice
                async for chunk in agent.model.stream(
                    messages, tools=tool_schemas, **stream_kwargs
                ):
                    if chunk.delta:
                        text_parts.append(chunk.delta)
                        yield emit(EventType.TEXT_DELTA, delta=chunk.delta)
                    if chunk.tool_call is not None:
                        turn_tool_calls.append(chunk.tool_call)
                content = "".join(text_parts)

                if turn_tool_calls:
                    messages.append(
                        Message(
                            role=Role.ASSISTANT, content=content, tool_calls=turn_tool_calls
                        )
                    )
                    for tc in turn_tool_calls:
                        tool_counts[tc.name] = tool_counts.get(tc.name, 0) + 1
                        if usage_limits is not None:
                            usage_limits.check_tool_call(tc.name, tool_counts)
                    for tc in turn_tool_calls:
                        yield emit(EventType.TOOL_CALL, name=tc.name, arguments=tc.arguments)
                    if getattr(agent, "parallel_tools", True) and len(turn_tool_calls) > 1:
                        results = await self._run_tools_parallel(agent, ctx, turn_tool_calls)
                    else:
                        results = [await self._run_tool(agent, ctx, tc) for tc in turn_tool_calls]
                    for tc, result_value in zip(turn_tool_calls, results, strict=False):
                        yield emit(EventType.TOOL_RESULT, name=tc.name, result=_safe(result_value))
                        messages.append(
                            Message(
                                role=Role.TOOL,
                                name=tc.name,
                                tool_call_id=tc.id,
                                content=_to_text(result_value),
                            )
                        )
                    if _is_forcing_tool_choice(effective_tool_choice):
                        effective_tool_choice = "auto"
                    continue

                # No tool calls -> finalize.
                final_output = self._coerce_output(agent, content, output_adapter)
                produced = True
                break

            if not produced:
                raise MaxStepsExceeded(
                    f"agent '{agent.name}' did not finish within {agent.max_steps} steps"
                )

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
            result: RunResult = RunResult(
                output=final_output, messages=messages, usage=ctx.usage, run_id=ctx.run_id
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
        session_id: str | None = None,
        identity: str | None = None,
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
        async for chunk in agent.model.stream(messages, **getattr(agent, "model_settings", {})):
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

        # Optionally fold in retrieved long-term memory. Thread the run's
        # identity (and the runner's app scope) into the search when the memory
        # backend is namespace-aware (e.g. MemoryManager) so per-user/app scoped
        # memory is actually reachable from the Agent path — not just the
        # "default" namespace.
        if self.memory_service is not None:
            hits = await self._memory_search(prompt, ctx)
            if hits:
                recalled = "\n".join(f"- {rec.text}" for rec, _ in hits)
                messages.append(Message(role=Role.SYSTEM, content=f"Relevant memory:\n{recalled}"))

        messages.append(_user_message(prompt, original))
        return messages

    async def _memory_search(self, prompt: str, ctx: RunContext) -> Any:
        """Search long-term memory, threading identity/app into namespace-aware
        backends while staying compatible with the plain ``search(query, k)``
        protocol.
        """
        import inspect

        assert self.memory_service is not None  # only called when set (guarded by caller)
        search = self.memory_service.search
        params: Any = {}
        try:
            params = inspect.signature(search).parameters
        except (TypeError, ValueError):  # builtins / C funcs without a signature
            params = {}
        kwargs: dict[str, Any] = {"k": 3}
        if "user_id" in params and ctx.identity is not None:
            kwargs["user_id"] = ctx.identity
        if "app_name" in params and self.memory_app_name is not None:
            kwargs["app_name"] = self.memory_app_name
        return await search(prompt, **kwargs)

    async def _call_model(
        self,
        agent: Any,
        ctx: RunContext,
        messages: list[Message],
        tool_schemas: list[dict[str, Any]] | None,
        output_schema: dict[str, Any] | None,
        tool_choice: Any | None = None,
    ) -> ModelResponse:
        for plugin in self.plugins:
            short = await plugin.before_model(ctx, agent.name, messages)
            if short is not None:
                ctx.usage.add(short.usage)
                return short
        response = await agent.model.complete(
            messages,
            tools=tool_schemas,
            output_schema=output_schema,
            tool_choice=tool_choice,
            **getattr(agent, "model_settings", {}),
        )
        ctx.usage.add(response.usage)
        for plugin in self.plugins:
            amended = await plugin.after_model(ctx, agent.name, response)
            if amended is not None:
                response = amended
        return response

    async def _run_tools_parallel(
        self, agent: Any, ctx: RunContext, tcs: list[ToolCall]
    ) -> list[Any]:
        """Execute a turn's tool calls concurrently, returning results in order."""
        import asyncio

        max_parallel = getattr(agent, "max_parallel_tools", 0) or 0
        sem = asyncio.Semaphore(max_parallel) if max_parallel > 0 else None

        async def _one(tc: ToolCall) -> Any:
            if sem is not None:
                async with sem:
                    return await self._run_tool(agent, ctx, tc)
            return await self._run_tool(agent, ctx, tc)

        return await asyncio.gather(*(_one(tc) for tc in tcs))

    def _tool_timeout(self, tool: Any) -> float | None:
        """Resolve the effective per-tool timeout (tool's own overrides default)."""
        own = getattr(tool, "timeout", None)
        return own if own is not None else self.default_tool_timeout

    async def _run_tool(self, agent: Any, ctx: RunContext, tc: ToolCall) -> Any:
        import asyncio

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
        timeout = self._tool_timeout(tool)
        try:
            if timeout is not None:
                result = await asyncio.wait_for(tool.execute(ctx, **tc.arguments), timeout)
            else:
                result = await tool.execute(ctx, **tc.arguments)
        except TimeoutError:
            result = f"error: tool '{tc.name}' timed out after {timeout}s"
        except ToolError as exc:
            result = f"error: {exc}"
        except Exception as exc:  # noqa: BLE001 - tools shouldn't crash the loop
            result = f"error: tool '{tc.name}' raised {type(exc).__name__}: {exc}"
        for plugin in self.plugins:
            amended = await plugin.after_tool(ctx, agent.name, tc.name, result)
            if amended is not None:
                result = amended
        return result

    def _coerce_output(self, agent: Any, content: str, adapter: TypeAdapter | None) -> Any:
        if adapter is None:
            return content
        return adapter.validate_json(content)

    async def _persist(self, agent: Any, ctx: RunContext, prompt: str, output: Any) -> None:
        if ctx.session_id is None:
            return
        await self.session_service.append(ctx.session_id, Message(role=Role.USER, content=prompt))
        await self.session_service.append(
            ctx.session_id, Message(role=Role.ASSISTANT, content=_to_text(output))
        )


def _user_message(text: str, original: Any = None) -> Message:
    """Build the user message, preserving multimodal parts from a Content."""
    from .content import Content

    if isinstance(original, Content) and original.is_multimodal():
        parts = original.to_provider_content()
        return Message(
            role=Role.USER,
            content=text,
            content_parts=parts if isinstance(parts, list) else None,
        )
    return Message(role=Role.USER, content=text)


def _normalize_tool_choice(choice: Any, tool_schemas: list[dict[str, Any]] | None) -> Any:
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


def _is_forcing_tool_choice(choice: Any) -> bool:
    """True if ``choice`` compels a tool call (so it must be relaxed after one).

    Forcing forms: the literal ``"required"`` and a pinned-function dict
    (``{"type": "function", ...}``). ``None``/``"auto"``/``"none"`` don't force.
    """
    if choice == "required":
        return True
    return isinstance(choice, dict) and choice.get("type") == "function"


def _output_spec(output_type: type) -> tuple[TypeAdapter | None, dict[str, Any] | None]:
    """Build a validator + JSON schema for non-string output types."""
    if output_type is str or output_type is type(None):
        return None, None
    adapter: TypeAdapter = TypeAdapter(output_type)
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
