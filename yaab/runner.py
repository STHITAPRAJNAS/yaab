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

import asyncio
import inspect
import json
import re
import time
from collections.abc import AsyncIterator
from typing import Any

from pydantic import TypeAdapter, ValidationError

from .exceptions import ApprovalRequired, MaxStepsExceeded, ToolError
from .governance.audit import AuditKind
from .governance.policy import Stage
from .governance.service import GovernanceService
from .limits import CancellationToken, UsageLimits
from .models.base import ModelResponse
from .plugins import Plugin
from .sessions.base import Session, SessionService
from .sessions.memory import InMemorySessionService
from .state import ReadonlyState, State, StateKeyError
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

#: Identifier-start required so JSON braces (``{"role": ...}``), numeric
#: placeholders (``{0}``), and CSS braces are left untouched — only
#: ``{name}`` / ``{user:name}`` / ``{name?}`` are treated as state fields.
_STATE_FIELD = re.compile(r"\{([a-zA-Z_][\w]*(?::[a-zA-Z_][\w]*)?)(\?)?\}")


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
        run_checkpointer: Any | None = None,
        checkpoint_mode: str = "step",
        trace_store: Any | None = None,
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
        #: Optional :class:`~yaab.graph.checkpoint.Checkpointer`. When set *and* a
        #: ``resume_id`` is passed to ``run``/``run_stream``, the fast-path loop is
        #: made fault-tolerant: progress is persisted after every completed step so
        #: a crashed run can resume where it left off. ``None`` keeps the
        #: classic zero-overhead fast path.
        self.run_checkpointer = run_checkpointer
        #: Durability granularity for the resumable fast path. ``"step"`` (the
        #: default) persists progress after every completed step so a run can
        #: resume from any point; ``"final"`` writes only the terminal marker
        #: (cheap for short runs, still idempotent on a finished re-invoke).
        if checkpoint_mode not in ("step", "final"):
            raise ValueError("checkpoint_mode must be 'step' or 'final'")
        self.checkpoint_mode = checkpoint_mode
        #: Optional durable per-run trace store. When set, every emitted event is
        #: appended (its JSON-safe payload) keyed by ``(run_id, seq)`` so a run's
        #: full timeline survives the run and a restart for the trace console.
        #: Code against the duck-typed ``append(run_id, seq, event_dict)`` so any
        #: compatible backend can be dropped in. ``None`` keeps today's behavior.
        self.trace_store = trace_store
        #: Process-local backing stores for the cross-session/cross-app state
        #: scopes (``user:`` / ``app:`` prefixes). Session-scoped state lives on
        #: the session itself; these hold the wider scopes so a run's State can
        #: route them. A durable backend swap (the persistence layer) replaces
        #: these without touching the build seam.
        self._app_state: dict[str, dict[str, Any]] = {}
        self._user_state: dict[tuple[str, str], dict[str, Any]] = {}

    def add_plugin(self, plugin: Plugin) -> Runner:
        self.plugins.append(plugin)
        return self

    # ------------------------------------------------------------------
    async def _build_state(self, session: Session | None, identity: str | None) -> State:
        """Build the one shared :class:`State` for a run.

        With a durable session, the State's session scope **is** ``session.state``
        (the same dict object, so unprefixed writes land in the session and
        persist for free); ``user:``/``app:`` keys route to the runner's scoped
        stores so they survive across the session/app as scoped. Without a
        session, a run-local State (still routes ``temp:``/``user:``/``app:``).
        """
        if session is None:
            return State()
        app = self._app_state.setdefault(self.memory_app_name or "default", {})
        user = self._user_state.setdefault(
            (self.memory_app_name or "default", identity or "default"), {}
        )
        return State(session=session.state, user=user, app=app)

    # ------------------------------------------------------------------
    async def run(
        self,
        agent: Any,
        prompt: str,
        *,
        deps: Any = None,
        session_id: str | None = None,
        identity: str | None = None,
        state: State | None = None,
        usage_limits: UsageLimits | None = None,
        cancellation: CancellationToken | None = None,
        timeout: float | None = None,
        resume_id: str | None = None,
        approval_decision: str | None = None,
    ) -> RunResult[Any]:
        events: list[Event] = []
        async for event in self.run_stream(
            agent,
            prompt,
            deps=deps,
            session_id=session_id,
            identity=identity,
            state=state,
            usage_limits=usage_limits,
            cancellation=cancellation,
            timeout=timeout,
            resume_id=resume_id,
            approval_decision=approval_decision,
        ):
            events.append(event)
        final = events[-1]
        if final.type is EventType.ERROR:
            raise final.payload["error"]
        # A run that durably paused for human sign-off ends on APPROVAL_REQUIRED
        # (no RUN_END). Surface that as a paused RunResult per the documented
        # invariant (read ``output`` only when ``not paused``) instead of raising.
        if final.type is EventType.APPROVAL_REQUIRED:
            return RunResult(
                output=None,
                events=events,
                run_id=final.run_id,
                paused=True,
                pause_value=dict(final.payload),
            )
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
        state: State | None = None,
        usage_limits: UsageLimits | None = None,
        cancellation: CancellationToken | None = None,
        timeout: float | None = None,
        resume_id: str | None = None,
        approval_decision: str | None = None,
        _transfer_depth: int = 0,
        _transfer_cap: int | None = None,
    ) -> AsyncIterator[Event]:
        """Execute the loop, yielding a typed :class:`Event` per step.

        When the runner has a ``run_checkpointer`` and ``resume_id`` is set, loop
        progress is persisted after every completed step. A crashed run can then
        be resumed by re-invoking with the same ``resume_id`` (the captured model
        turns are NOT re-requested), and a finished ``resume_id`` returns the
        persisted result idempotently. Sub-agent handoffs ignore ``resume_id`` —
        only the root run is checkpointed.

        A sensitive tool call guarded for out-of-band sign-off pauses the run
        durably: the loop checkpoints a pending-approval marker, emits an
        :attr:`~yaab.types.EventType.APPROVAL_REQUIRED` event, and returns (no
        thread is blocked). Re-invoking with the same ``resume_id`` and an
        ``approval_decision`` of ``"approved"`` (run the tool now) or ``"denied"``
        (feed the model a denial) finishes the loop without re-requesting the
        captured model turns. Without a checkpointer the call raises instead —
        backward compatible.

        ``_transfer_depth``/``_transfer_cap`` are internal: they track how many
        sub-agent handoffs deep this run already is (and the root's cap) so a
        chain of ``transfer_to_agent`` calls can't loop forever. External callers
        leave them at their defaults.
        """
        # Resumable fast path: only the root run (never a sub-agent handoff) is
        # checkpointed, and only when both a checkpointer and a resume_id exist.
        # ``ckpt_id`` is the narrowed (non-None) resume key when checkpointing is on.
        ckpt_id: str | None = None
        if self.run_checkpointer is not None and _transfer_depth == 0:
            ckpt_id = resume_id
        resume_state: dict[str, Any] | None = None
        if self.run_checkpointer is not None and ckpt_id is not None:
            saved = self.run_checkpointer.get(ckpt_id)
            if saved is not None:
                _, saved_state = saved
                if saved_state.get("finished"):
                    # Idempotent re-invoke: rebuild the final result, no model calls.
                    yield self._replay_finished(agent, saved_state)
                    return
                resume_state = saved_state

        # Build-once / inherit-always: a child invoked by a workflow agent or by
        # a model-driven transfer is handed the parent's State; only the
        # outermost entity builds one (over the session, the source of truth).
        if state is not None:
            shared_state = state
        else:
            session = (
                await self.session_service.get_or_create(session_id)
                if session_id is not None
                else None
            )
            shared_state = await self._build_state(session, identity)
        ctx: RunContext = RunContext(
            deps=deps,
            session_id=session_id,
            identity=identity,
            usage=Usage(),
            state=shared_state,
        )
        # Thread the checkpoint key into the run context so a queue-mode approval
        # plugin can correlate its durable pending record to the exact checkpoint
        # the loop will resume from. Run-local (temp:) so it never persists.
        if ckpt_id is not None:
            ctx.state["temp:__resume_id__"] = ckpt_id
            # Key the run (and therefore its trace) by the durable resume id so the
            # persisted timeline correlates with the run record and is retrievable
            # via ``GET /runs/{id}/trace`` — and so a resume appends to the same
            # trace rather than starting a disconnected one.
            ctx.run_id = ckpt_id
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

        trace_store = self.trace_store
        seq = [0]
        # Trace appends are async; ``emit`` is sync (called as ``yield emit(...)``)
        # so it schedules each append as a task and we drain them at run end. This
        # guarantees the timeline (including APPROVAL_REQUIRED) is actually
        # persisted — previously the coroutine was created and dropped unawaited,
        # so durable traces were silently never written.
        trace_tasks: list[Any] = []

        def emit(etype: EventType, *, duration_ms: float | None = None, **payload: Any) -> Event:
            event = Event(
                type=etype,
                agent=agent.name,
                run_id=ctx.run_id,
                payload=payload,
                duration_ms=duration_ms,
            )
            # Persist the full timeline for the trace console, keyed by sequence,
            # in a JSON-safe shape (so it survives the run and a restart). The
            # append may be sync (duck-typed store) or async (the durable
            # backends); a returned coroutine is scheduled and drained at run end.
            if trace_store is not None:
                safe = _safe_event(event)
                safe["seq"] = seq[0]
                maybe = trace_store.append(ctx.run_id, seq[0], safe)
                if maybe is not None and inspect.isawaitable(maybe):
                    trace_tasks.append(asyncio.ensure_future(maybe))
            seq[0] += 1
            return event

        async def _drain_trace() -> None:
            """Await any in-flight async trace appends so the timeline is durable."""
            if not trace_tasks:
                return
            pending = list(trace_tasks)
            trace_tasks.clear()
            await asyncio.gather(*pending, return_exceptions=True)

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

            tool_schemas = [t.schema() for t in _available_tools(agent, ctx)] or None
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

            if resume_state is not None:
                # Resume: rehydrate the loop from the last checkpoint instead of
                # rebuilding the prompt. The captured model turns are already in
                # ``messages`` so they are never re-requested; the forced first
                # model call (if any) has already happened, so relax tool_choice.
                messages = [Message.model_validate(m) for m in resume_state["messages"]]
                tool_counts = dict(resume_state.get("tool_counts", {}))
                ctx.usage = Usage.model_validate(resume_state["usage"])
                output_retries_left = resume_state.get("output_retries_left", output_retries_left)
                start_step = int(resume_state["step"]) + 1
                # Restore the committed (durable) state the crashed/paused run had
                # written, so a resumed tool/step sees exactly what prior steps
                # produced — never an empty dict, never a lost temp: scratch value.
                committed = resume_state.get("state")
                if committed:
                    ctx.state.session.update(committed)
                # Continue the trace sequence past the events recorded before the
                # pause so a resume appends to the same timeline instead of
                # overwriting it (the approval event and prior steps are kept).
                seq[0] = int(resume_state.get("trace_seq", 0))
                if _is_forcing_tool_choice(effective_tool_choice):
                    effective_tool_choice = "auto"
                yield emit(EventType.USER_MESSAGE, content=scanned_prompt)

                # Resume-from-pending-approval: a sensitive tool call parked this
                # run awaiting human sign-off. The reviewer has now decided, so
                # resolve the pending tool here — run it (approved) or feed the
                # model a denial — then continue WITHOUT re-requesting any of the
                # captured model turns. The model never re-decides.
                pending = resume_state.get("pending_approval")
                if pending is not None:
                    tool_msg, call_ev, result_ev = await self._resume_pending(
                        agent, ctx, pending, approval_decision, emit
                    )
                    yield call_ev
                    yield result_ev
                    messages.append(tool_msg)
            else:
                messages, templated_keys = await self._build_messages(
                    agent, ctx, scanned_prompt, original=prompt
                )
                user_msg = messages[-1]
                yield emit(EventType.USER_MESSAGE, content=scanned_prompt)
                # Make the state-shaped instruction visible in the trace: which
                # shared-state fields fed this run's prompt (the read side of the
                # writes=/{key} handoff). Emitted only when something resolved, so
                # plain instructions stay byte-for-byte as before.
                if templated_keys:
                    yield emit(EventType.STATE_TEMPLATE, keys=list(templated_keys))
                for plugin in self.plugins:
                    await plugin.on_user_message(ctx, agent.name, user_msg)
                start_step = 0

            context_strategy = getattr(agent, "context_strategy", None)

            for _step in range(start_step, agent.max_steps):
                check_controls()  # cancellation / timeout / usage caps
                # Keep the conversation within the model's context window.
                if context_strategy is not None:
                    messages = await context_strategy.apply(messages, model=agent.model)
                _model_started = time.monotonic()
                response = await self._call_model(
                    agent, ctx, messages, tool_schemas, output_schema, effective_tool_choice
                )
                _model_ms = (time.monotonic() - _model_started) * 1000.0
                # Re-check usage now that this request's tokens are counted.
                if usage_limits is not None:
                    usage_limits.check_usage(ctx.usage)
                if response.reasoning:
                    yield emit(EventType.MODEL_DELTA, reasoning=response.reasoning)
                # Enrich with the model name, finish reason, and this call's token
                # delta so the trace console can render per-call cost and latency.
                yield emit(
                    EventType.MODEL_RESPONSE,
                    duration_ms=_model_ms,
                    content=response.content,
                    tool_calls=[tc.model_dump() for tc in response.tool_calls],
                    model=response.model,
                    finish_reason=response.finish_reason,
                    usage=response.usage.model_dump(),
                    reasoning=bool(response.reasoning),
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
                        _tool_started = time.monotonic()
                        try:
                            results = await self._run_tools_parallel(agent, ctx, tcs)
                        except ApprovalRequired as exc:
                            paused = self._pause_for_approval(
                                ctx,
                                exc,
                                _step,
                                messages,
                                tool_counts,
                                output_retries_left,
                                emit,
                                seq,
                            )
                            if paused is None:
                                raise
                            yield paused
                            return
                        _tool_ms = (time.monotonic() - _tool_started) * 1000.0
                        for tc, result_value in zip(tcs, results, strict=False):
                            yield emit(
                                EventType.TOOL_RESULT,
                                duration_ms=_tool_ms,
                                name=tc.name,
                                result=_safe(result_value),
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
                            _tool_started = time.monotonic()
                            try:
                                result_value = await self._run_tool(agent, ctx, tc)
                            except ApprovalRequired as exc:
                                # A sensitive tool needs out-of-band sign-off:
                                # checkpoint a pending-approval marker and pause
                                # the run durably (or re-raise without a
                                # checkpointer — backward compatible).
                                paused = self._pause_for_approval(
                                    ctx,
                                    exc,
                                    _step,
                                    messages,
                                    tool_counts,
                                    output_retries_left,
                                    emit,
                                    seq,
                                )
                                if paused is None:
                                    raise
                                yield paused
                                return
                            _tool_ms = (time.monotonic() - _tool_started) * 1000.0
                            yield emit(
                                EventType.TOOL_RESULT,
                                duration_ms=_tool_ms,
                                name=tc.name,
                                result=_safe(result_value),
                            )
                            messages.append(
                                Message(
                                    role=Role.TOOL,
                                    name=tc.name,
                                    tool_call_id=tc.id,
                                    content=_to_text(result_value),
                                )
                            )

                    # transfer_to_agent: a sub-agent handoff was requested.
                    # Delegate the ORIGINAL prompt to it and adopt its answer as
                    # this run's final output (skip the parent's own coercion).
                    sub, transfer_err = self._pop_transfer(
                        agent, ctx, _transfer_depth, _transfer_cap
                    )
                    if transfer_err is not None:
                        # Over the depth cap: surface as a tool error so the loop
                        # continues and the agent answers itself instead of looping.
                        messages.append(Message(role=Role.TOOL, content=transfer_err))
                    elif sub is not None:
                        yield emit(EventType.AGENT_TRANSFER, to=sub.name)
                        sub_output: Any = None
                        async for sub_ev in self.run_stream(
                            sub,
                            prompt,
                            deps=deps,
                            session_id=session_id,
                            identity=identity,
                            state=ctx.state,
                            usage_limits=usage_limits,
                            cancellation=cancellation,
                            _transfer_depth=_transfer_depth + 1,
                            _transfer_cap=_resolve_cap(agent, _transfer_cap),
                        ):
                            if sub_ev.type is EventType.RUN_END:
                                sub_output = sub_ev.payload["result"].output
                                ctx.usage.add(sub_ev.payload["result"].usage)
                            elif sub_ev.type is EventType.ERROR:
                                raise sub_ev.payload["error"]
                            else:
                                yield sub_ev
                        final_output = sub_output
                        produced = True
                        break

                    # Tool round complete: checkpoint progress so a crash after
                    # this point resumes from here (the captured model+tool turn
                    # is never re-requested). In ``final`` mode only the terminal
                    # marker is written, so per-step writes are skipped.
                    if ckpt_id is not None and self.checkpoint_mode == "step":
                        self._save_step(
                            ckpt_id,
                            _step,
                            messages,
                            tool_counts,
                            ctx.usage,
                            output_retries_left,
                            ctx,
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
                    # A validation retry is also a completed model step.
                    if ckpt_id is not None and self.checkpoint_mode == "step":
                        self._save_step(
                            ckpt_id,
                            _step,
                            messages,
                            tool_counts,
                            ctx.usage,
                            output_retries_left,
                            ctx,
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

            # writes= capture: land this run's typed output into shared state
            # under its declared key (before persistence, so the value is folded
            # into the durable subset) and surface the handoff in the trace.
            captured = _capture_writes(agent, ctx, final_output)
            if captured is not None:
                yield emit(EventType.STATE_WRITE, key=captured[0], scope=captured[1])

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
            # Terminal marker: a finished resume_id must never re-run; a later
            # re-invoke reconstructs this exact result from the checkpoint.
            if ckpt_id is not None:
                self._save_finished(
                    ckpt_id, agent.max_steps, messages, ctx.usage, final_output, ctx.run_id
                )
            yield emit(
                EventType.RUN_END,
                duration_ms=(time.monotonic() - run_started) * 1000.0,
                result=result,
            )

        except Exception as exc:  # noqa: BLE001 - surface as a terminal ERROR event
            if gov is not None:
                gov.audit.record(
                    AuditKind.ERROR,
                    agent_id=agent.registry_id,
                    identity=identity,
                    error=str(exc),
                )
            yield emit(EventType.ERROR, error=exc)
        finally:
            # Drain any scheduled trace appends so the durable timeline is fully
            # written before the run returns — including on the approval-pause
            # and error exits, not only the normal RUN_END path.
            await _drain_trace()

    # ------------------------------------------------------------------
    # Resumable fast-path checkpoint helpers.
    # ------------------------------------------------------------------
    def _save_step(
        self,
        resume_id: str,
        step: int,
        messages: list[Message],
        tool_counts: dict[str, int],
        usage: Usage,
        output_retries_left: int,
        ctx: RunContext | None = None,
    ) -> None:
        """Persist loop progress after a completed step (model turn + tools)."""
        assert self.run_checkpointer is not None
        state = {
            "step": step,
            "messages": [m.model_dump(mode="json") for m in messages],
            "tool_counts": dict(tool_counts),
            "usage": usage.model_dump(mode="json"),
            "output_retries_left": output_retries_left,
            # The durable subset only (temp: excluded) — identical to what session
            # write-back persists, so resume and scoping agree on one durable view.
            "state": _persisted_state(ctx),
        }
        self.run_checkpointer.put(resume_id, step, state)

    def _save_finished(
        self,
        resume_id: str,
        step: int,
        messages: list[Message],
        usage: Usage,
        output: Any,
        run_id: str,
    ) -> None:
        """Write the terminal ``finished`` marker carrying the final result."""
        assert self.run_checkpointer is not None
        state = {
            "step": step,
            "finished": True,
            "messages": [m.model_dump(mode="json") for m in messages],
            "usage": usage.model_dump(mode="json"),
            "output": _safe(output),
            "run_id": run_id,
        }
        # Use a high step number so this marker sorts last in history().
        self.run_checkpointer.put(resume_id, step + 1, state)

    def _replay_finished(self, agent: Any, state: dict[str, Any]) -> Event:
        """Reconstruct a terminal RUN_END event from a finished checkpoint."""
        messages = [Message.model_validate(m) for m in state.get("messages", [])]
        usage = Usage.model_validate(state["usage"]) if "usage" in state else Usage()
        result: RunResult = RunResult(
            output=state.get("output"),
            messages=messages,
            usage=usage,
            run_id=state.get("run_id", ""),
        )
        return Event(
            type=EventType.RUN_END,
            agent=agent.name,
            run_id=result.run_id,
            payload={"result": result},
        )

    # ------------------------------------------------------------------
    # Durable human-in-the-loop: pause for approval, then resume on a decision.
    # ------------------------------------------------------------------
    def _pause_for_approval(
        self,
        ctx: RunContext,
        exc: ApprovalRequired,
        step: int,
        messages: list[Message],
        tool_counts: dict[str, int],
        output_retries_left: int,
        emit: Any,
        seq: list[int],
    ) -> Event | None:
        """Park the run for out-of-band human sign-off, or signal a re-raise.

        When a checkpointer + resume key are configured, this writes a
        pending-approval checkpoint (so the run sleeps durably and can resume on
        any replica) and returns the :attr:`EventType.APPROVAL_REQUIRED` event to
        emit before the loop returns. Without a checkpointer it returns ``None``
        so the caller re-raises — preserving the classic block behavior.

        The APPROVAL_REQUIRED event is emitted *before* the checkpoint is written
        so the saved ``trace_seq`` points past it; a later resume then continues
        the trace after the approval event instead of overwriting it.
        """
        resume_key = ctx.state.get("temp:__resume_id__")
        if self.run_checkpointer is None or resume_key is None:
            return None
        approval_id = getattr(exc, "approval_id", "")
        pending = {
            "tool": exc.tool,
            "arguments": dict(exc.arguments),
            "approval_id": approval_id,
        }
        event = emit(
            EventType.APPROVAL_REQUIRED,
            approval_id=approval_id,
            tool=exc.tool,
            arguments=dict(exc.arguments),
        )
        self._save_pending(
            resume_key,
            step,
            messages,
            tool_counts,
            ctx.usage,
            output_retries_left,
            pending,
            trace_seq=seq[0],
            ctx=ctx,
        )
        return event

    def _save_pending(
        self,
        resume_id: str,
        step: int,
        messages: list[Message],
        tool_counts: dict[str, int],
        usage: Usage,
        output_retries_left: int,
        pending: dict[str, Any],
        *,
        trace_seq: int = 0,
        ctx: RunContext | None = None,
    ) -> None:
        """Checkpoint a parked run that is awaiting a human approval decision.

        Reuses the per-step state shape plus a ``pending_approval`` marker so the
        resume path knows to resolve the held tool before continuing — never
        re-requesting the captured model turns. ``trace_seq`` records how far the
        trace got so the resume appends after the approval event. The committed
        state is carried so a run that pauses on one pod and resumes on another
        restores exactly what it had written.
        """
        assert self.run_checkpointer is not None
        state = {
            "step": step,
            "messages": [m.model_dump(mode="json") for m in messages],
            "tool_counts": dict(tool_counts),
            "usage": usage.model_dump(mode="json"),
            "output_retries_left": output_retries_left,
            "pending_approval": pending,
            "trace_seq": trace_seq,
            "state": _persisted_state(ctx),
        }
        self.run_checkpointer.put(resume_id, step, state)

    async def _resume_pending(
        self,
        agent: Any,
        ctx: RunContext,
        pending: dict[str, Any],
        approval_decision: str | None,
        emit: Any,
    ) -> tuple[Message, Event, Event]:
        """Resolve a parked approval, returning the tool message + its events.

        On ``"approved"`` the held tool runs now (bypassing the approval gate,
        since a human already signed off). On ``"denied"`` — or any non-approval
        decision — the model is fed a denial message instead and the tool never
        runs. The caller appends the returned :class:`Message` to the conversation
        and yields the two events, then the loop continues from the next step.
        """
        tool_name = pending["tool"]
        arguments = dict(pending.get("arguments", {}))
        call_ev = emit(EventType.TOOL_CALL, name=tool_name, arguments=arguments)

        approved = approval_decision == "approved"
        if approved:
            result_value = await self._execute_approved_tool(agent, ctx, tool_name, arguments)
        else:
            reviewer = approval_decision or "reviewer"
            result_value = (
                f"error: tool '{tool_name}' denied by reviewer ({reviewer}); do not retry it."
            )
        result_ev = emit(
            EventType.TOOL_RESULT,
            duration_ms=0.0,
            name=tool_name,
            result=_safe(result_value),
        )
        tool_msg = Message(role=Role.TOOL, name=tool_name, content=_to_text(result_value))
        return tool_msg, call_ev, result_ev

    async def _execute_approved_tool(
        self, agent: Any, ctx: RunContext, tool_name: str, arguments: dict[str, Any]
    ) -> Any:
        """Run a now-approved tool directly, skipping the approval gate.

        The human has already decided, so the approval plugin's ``before_tool``
        must not park the run again. Other plugins' post-processing still applies.
        """
        import asyncio

        tool = next((t for t in agent.tools if t.name == tool_name), None)
        if tool is None:
            return f"error: unknown tool '{tool_name}'"
        timeout = self._tool_timeout(tool)
        try:
            if timeout is not None:
                result = await asyncio.wait_for(tool.execute(ctx, **arguments), timeout)
            else:
                result = await tool.execute(ctx, **arguments)
        except TimeoutError:
            result = f"error: tool '{tool_name}' timed out after {timeout}s"
        except ToolError as exc:
            result = f"error: {exc}"
        except Exception as exc:  # noqa: BLE001 - tools shouldn't crash the loop
            result = f"error: tool '{tool_name}' raised {type(exc).__name__}: {exc}"
        for plugin in self.plugins:
            amended = await plugin.after_tool(ctx, agent.name, tool_name, result)
            if amended is not None:
                result = amended
        return result

    # ------------------------------------------------------------------
    async def stream_run(
        self,
        agent: Any,
        prompt: str,
        *,
        deps: Any = None,
        session_id: str | None = None,
        identity: str | None = None,
        state: State | None = None,
        usage_limits: UsageLimits | None = None,
        cancellation: CancellationToken | None = None,
        timeout: float | None = None,
        _transfer_depth: int = 0,
        _transfer_cap: int | None = None,
    ) -> AsyncIterator[Event]:
        """Run the full multi-step loop while streaming token deltas.

        Unlike :meth:`run_stream` (which calls the model with ``complete`` and so
        only emits whole responses), this drives each turn with ``model.stream``:
        text arrives as :attr:`EventType.TEXT_DELTA` events *as it generates*,
        tools execute mid-run (``TOOL_CALL``/``TOOL_RESULT``), and the loop
        continues until a final answer — the stream-through-the-tool-loop
        behavior. Terminates with ``FINAL_OUTPUT`` then ``RUN_END``.

        ``_transfer_depth``/``_transfer_cap`` are internal handoff-loop guards
        (see :meth:`run_stream`).
        """
        # Build-once / inherit-always (see run_stream): a child inherits the
        # parent's State; only the outermost entity builds one.
        if state is not None:
            shared_state = state
        else:
            session = (
                await self.session_service.get_or_create(session_id)
                if session_id is not None
                else None
            )
            shared_state = await self._build_state(session, identity)
        ctx: RunContext = RunContext(
            deps=deps,
            session_id=session_id,
            identity=identity,
            usage=Usage(),
            state=shared_state,
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
            messages, templated_keys = await self._build_messages(
                agent, ctx, scanned_prompt, original=prompt
            )
            yield emit(EventType.USER_MESSAGE, content=scanned_prompt)
            if templated_keys:
                yield emit(EventType.STATE_TEMPLATE, keys=list(templated_keys))

            tool_schemas = [t.schema() for t in _available_tools(agent, ctx)] or None
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
                        Message(role=Role.ASSISTANT, content=content, tool_calls=turn_tool_calls)
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

                    # transfer_to_agent: delegate the ORIGINAL prompt to the
                    # requested sub-agent and adopt its answer (see run_stream).
                    sub, transfer_err = self._pop_transfer(
                        agent, ctx, _transfer_depth, _transfer_cap
                    )
                    if transfer_err is not None:
                        messages.append(Message(role=Role.TOOL, content=transfer_err))
                    elif sub is not None:
                        yield emit(EventType.AGENT_TRANSFER, to=sub.name)
                        sub_output: Any = None
                        async for sub_ev in self.stream_run(
                            sub,
                            prompt,
                            deps=deps,
                            session_id=session_id,
                            identity=identity,
                            state=ctx.state,
                            usage_limits=usage_limits,
                            cancellation=cancellation,
                            _transfer_depth=_transfer_depth + 1,
                            _transfer_cap=_resolve_cap(agent, _transfer_cap),
                        ):
                            if sub_ev.type is EventType.RUN_END:
                                sub_output = sub_ev.payload["result"].output
                                ctx.usage.add(sub_ev.payload["result"].usage)
                            elif sub_ev.type is EventType.ERROR:
                                raise sub_ev.payload["error"]
                            else:
                                yield sub_ev
                        final_output = sub_output
                        produced = True
                        break

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
            captured = _capture_writes(agent, ctx, final_output)
            if captured is not None:
                yield emit(EventType.STATE_WRITE, key=captured[0], scope=captured[1])
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

        session = (
            await self.session_service.get_or_create(session_id) if session_id is not None else None
        )
        shared_state = await self._build_state(session, identity)
        ctx: RunContext = RunContext(
            deps=deps, session_id=session_id, identity=identity, state=shared_state
        )
        prompt_text = prompt.text if isinstance(prompt, Content) else str(prompt)
        if self.governance is not None:
            prompt_text = self.governance.scan(
                prompt_text, Stage.INPUT, agent_id=agent.registry_id, identity=identity
            )
        messages, _ = await self._build_messages(agent, ctx, prompt_text, original=prompt)
        async for chunk in agent.model.stream(messages, **getattr(agent, "model_settings", {})):
            if chunk.delta:
                yield chunk.delta

    # ------------------------------------------------------------------
    async def _build_messages(
        self, agent: Any, ctx: RunContext, prompt: str, *, original: Any = None
    ) -> tuple[list[Message], list[str]]:
        """Assemble the model messages, returning ``(messages, templated_keys)``.

        ``templated_keys`` lists the state field names a string instruction's
        ``{key}`` placeholders resolved to (empty for callable/plain
        instructions), so the caller can emit a :attr:`EventType.STATE_TEMPLATE`
        event making the state-shaped prompt visible in the trace.
        """
        messages: list[Message] = []
        templated_keys: list[str] = []
        instructions = agent.instructions
        if callable(instructions):
            # The function form gets a read-only context so rendering can never
            # mutate shared state; the author has full control (no auto-injection).
            instructions = instructions(ctx.readonly())
        elif isinstance(instructions, str) and instructions:
            # The string form gets {key} injection from the same read-only view.
            instructions, templated_keys = _inject_state(instructions, ReadonlyState(ctx.state))
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
        return messages, templated_keys

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

    def _pop_transfer(
        self, agent: Any, ctx: RunContext, depth: int, cap: int | None
    ) -> tuple[Any | None, str | None]:
        """Consume a pending ``transfer_to_agent`` request from ``ctx.state``.

        Returns ``(sub_agent, None)`` when a valid, in-budget handoff is pending,
        ``(None, error_message)`` when the transfer-depth cap would be exceeded
        (so the loop can surface the error and continue), or ``(None, None)`` when
        no transfer was requested. Always clears the flag so it never leaks into
        the next turn or the sub-agent's run.
        """
        name = ctx.state.pop("temp:__transfer_to__", None)
        if name is None:
            return None, None
        sub_step = next((s for s in _sub_steps(agent) if _sub_agent_of(s).name == name), None)
        if sub_step is None:
            # The tool already validated the name; defensively ignore unknowns.
            return None, None
        sub = _sub_agent_of(sub_step)
        # A when= guard on the sub-agent gates a model-driven transfer to it. If
        # the guard is false, refuse with a model-visible error so the model can
        # answer itself or pick another target — it does not crash the run. A
        # plain (un-wrapped) sub-agent has no guard.
        from .conditions import Step as _Step

        guard_spec = sub_step.when if isinstance(sub_step, _Step) else None
        if guard_spec is not None:
            from .conditions import Guard, Phase, as_condition

            cond = as_condition(guard_spec, phase=Phase.INPUT)
            g = Guard(
                value=None,
                state=ctx.readonly().state,
                ctx=ctx,
                phase=Phase.INPUT,
            )
            if not cond.check(g):
                return None, (
                    f"error: transfer to '{name}' is not permitted (guard '{cond.label}' is false)"
                )
        effective_cap = cap if cap is not None else getattr(agent, "transfer_depth", 3)
        if depth + 1 > effective_cap:
            return None, (
                f"error: transfer to '{name}' refused — transfer depth cap "
                f"({effective_cap}) exceeded"
            )
        return sub, None

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
        tool = next((t for t in _available_tools(agent, ctx) if t.name == tc.name), None)
        if tool is None:
            # The tool is either unknown or currently gated off by a when= guard;
            # tell the model so it can recover rather than crashing the loop.
            if any(t.name == tc.name for t in _all_tools(agent)):
                return f"error: tool '{tc.name}' is not available right now"
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
        # Write back the durable subset of shared state (temp: excluded). Because
        # the State's session scope *is* session.state, unprefixed writes are
        # already there; this also folds user:/app: writes (separate stores) back
        # and saves the session so it survives to the next run. A child run that
        # inherited the parent's State already shares session.state, so this is a
        # cheap no-op for them; only the root that owns the session saves.
        if isinstance(ctx.state, State):
            persisted = ctx.state.persisted()
            session = await self.session_service.get(ctx.session_id)
            if session is not None:
                session.state.update(persisted)
                await self.session_service.save(session)


def _resolve_cap(agent: Any, cap: int | None) -> int:
    """The transfer-depth cap for delegating below ``agent``.

    The root run carries ``cap is None``; we then anchor the cap to the root
    agent's ``transfer_depth`` so the whole chain shares one budget.
    """
    return cap if cap is not None else getattr(agent, "transfer_depth", 3)


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


def _inject_state(template: str, state: ReadonlyState) -> tuple[str, list[str]]:
    """Substitute ``{key}`` / ``{key?}`` placeholders from shared state.

    Returns ``(rendered, resolved_keys)`` where ``resolved_keys`` is the ordered,
    de-duplicated list of fields actually pulled from state (so the runner can
    surface them as a :attr:`~yaab.types.EventType.STATE_TEMPLATE` trace event).

    Only identifier-style fields (``{name}``, ``{user:name}``) are treated as
    state — JSON braces, numeric placeholders, and CSS braces pass through. A
    missing **required** key raises :class:`~yaab.state.StateKeyError`; a trailing
    ``?`` makes the field optional and substitutes empty string when absent.
    """
    resolved: list[str] = []

    def sub(m: re.Match) -> str:
        key, optional = m.group(1), m.group(2)
        try:
            value = _to_text(state[key])
        except KeyError as exc:
            if optional:
                return ""
            raise StateKeyError(
                f"instruction references {{{key}}} but it is not in state; "
                f"mark it optional as {{{key}?}} or ensure an upstream step writes it"
            ) from exc
        if key not in resolved:
            resolved.append(key)
        return value

    return _STATE_FIELD.sub(sub, template), resolved


def _capture_writes(agent: Any, ctx: RunContext, output: Any) -> tuple[str, str] | None:
    """Land an agent's validated output into shared state under its ``writes=`` key.

    Returns ``(key, scope)`` when a capture happened (so the caller can emit a
    :attr:`~yaab.types.EventType.STATE_WRITE` trace event), or ``None`` when the
    agent declares no ``writes=`` key. The *typed* ``output`` is stored exactly as
    produced — it never round-trips through text — and the key's prefix
    (``temp:``/``user:``/``app:``) selects the scope for free. Idempotent: a
    workflow that also captures the same child writes the identical value.
    """
    from .state import scope_of

    key = getattr(agent, "writes", None)
    if not key:
        return None
    ctx.state[key] = output
    return key, scope_of(key)


def _persisted_state(ctx: RunContext | None) -> dict[str, Any]:
    """JSON-safe durable subset of a run's shared state (temp: excluded)."""
    if ctx is None or not isinstance(ctx.state, State):
        return {}
    return {k: _safe(v) for k, v in ctx.state.persisted().items()}


def _to_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        if hasattr(value, "model_dump"):
            return json.dumps(value.model_dump())
        return json.dumps(value)
    except (TypeError, ValueError):
        return str(value)


def _all_tools(agent: Any) -> list[Any]:
    """Every tool on an agent, unwrapping any conditional ``Step`` wrappers."""
    from .conditions import Step

    out: list[Any] = []
    for entry in getattr(agent, "tools", []):
        out.append(entry.unit if isinstance(entry, Step) else entry)
    return out


def _available_tools(agent: Any, ctx: RunContext) -> list[Any]:
    """The tools currently exposed to the model, honoring each tool's ``when=``.

    A bare tool is always available. A ``Step(tool, when=...)`` is available only
    when its input guard is true against the current state; otherwise it is
    omitted from the model-facing schema (no sentinel, no map mutation), so the
    model simply cannot call it this step.
    """
    from .conditions import Guard, Phase, Step, as_condition

    out: list[Any] = []
    ro = ctx.readonly().state
    for entry in getattr(agent, "tools", []):
        if not isinstance(entry, Step):
            out.append(entry)
            continue
        if entry.when is None:
            out.append(entry.unit)
            continue
        cond = as_condition(entry.when, phase=Phase.INPUT)
        guard = Guard(value=None, state=ro, ctx=ctx, phase=Phase.INPUT)
        if cond.check(guard):
            out.append(entry.unit)
    return out


def _sub_steps(agent: Any) -> list[Any]:
    """The sub-agent entries on an agent (each a plain agent or a ``Step``)."""
    return list(getattr(agent, "sub_agents", []))


def _sub_agent_of(entry: Any) -> Any:
    """Unwrap a sub-agent entry: a ``Step`` yields its wrapped agent."""
    from .conditions import Step

    return entry.unit if isinstance(entry, Step) else entry


def _safe(value: Any) -> Any:
    """Make a value JSON-safe for embedding in an Event payload."""
    if isinstance(value, (str, int, float, bool, type(None), dict, list)):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return str(value)


def _safe_event(event: Event) -> dict[str, Any]:
    """Render an emitted event into a fully JSON-safe dict for the trace store.

    The live payload may carry a :class:`RunResult` (under ``result``) or an
    exception (under ``error``); both are coerced so the persisted timeline
    survives the run and a restart. ``RunResult.usage`` is preserved so the
    trace console can attribute tokens and cost per run.
    """
    payload: dict[str, Any] = {}
    for key, value in event.payload.items():
        if key == "result" and hasattr(value, "model_dump"):
            payload[key] = value.model_dump(mode="json")
        elif isinstance(value, BaseException):
            payload[key] = str(value)
        else:
            payload[key] = _safe(value)
    return {
        "type": event.type.value,
        "agent": event.agent,
        "run_id": event.run_id,
        "seq": None,
        "timestamp": event.timestamp,
        "duration_ms": event.duration_ms,
        "payload": payload,
    }
