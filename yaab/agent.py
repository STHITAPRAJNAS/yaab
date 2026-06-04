"""The typed :class:`Agent` — YAAB's primary developer-facing abstraction.

``Agent[Deps, Output]`` is generic over a dependency-injection type and an
output type, fusing type-safety with a clean agent/runner split.
The three-line "hello agent" works with zero ceremony; every layer underneath
(runner, sessions, governance, graph) is openable when you need it.

    agent = Agent("assistant", model="openai/gpt-4o", instructions="Be helpful.")
    result = agent.run_sync("Hello!")
    print(result.output)
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Generic

from .models import ModelProvider, resolve_model
from .tools.base import Tool, coerce_tools
from .types import Deps, Output, RunContext, RunResult

if TYPE_CHECKING:
    from .state import State

_NoneType = type(None)  # module-level singleton (avoids a call in arg defaults)


def _sub_unit(entry: Any) -> Any:
    """Unwrap a sub-agent entry: a :class:`~yaab.conditions.Step` yields its unit.

    A sub-agent may be wrapped in a ``Step(sub_agent, when=...)`` to gate a
    model-driven transfer to it; the underlying agent (for the roster and the
    transfer roster names) is its ``.unit``.
    """
    from .conditions import Step

    return entry.unit if isinstance(entry, Step) else entry


class Agent(Generic[Deps, Output]):
    """A type-safe agent: a model + instructions + tools + an output contract."""

    def __init__(
        self,
        name: str,
        *,
        model: str | ModelProvider = "openai/gpt-4o",
        instructions: str | Callable[[RunContext[Deps]], str] = "",
        description: str = "",
        tools: list[Any] | None = None,
        deps_type: type = _NoneType,
        output_type: type = str,
        sub_agents: list[Agent[Any, Any]] | None = None,
        transfer_depth: int = 3,
        guardrails: list[Any] | None = None,
        capabilities: list[Any] | None = None,
        skills: list[Any] | None = None,
        registry_id: str | None = None,
        max_steps: int = 8,
        output_retries: int = 2,
        tool_choice: Any | None = None,
        context_strategy: Any | None = None,
        parallel_tools: bool = True,
        max_parallel_tools: int = 0,
        model_settings: dict[str, Any] | None = None,
        runner: Any | None = None,
        instrument: bool = True,
        writes: str | None = None,
    ) -> None:
        self.name = name
        self._model_spec = model
        #: Capture this run's final (typed) output into shared state under this
        #: key after it completes, so a downstream step reads it by name. A prefix
        #: on the key (``temp:``/``user:``/``app:``) selects the scope.
        self.writes = writes
        self.deps_type = deps_type
        self.output_type = output_type
        self.guardrails = guardrails or []
        self.capabilities = capabilities or []
        self.skills = skills or []
        self.registry_id = registry_id
        self.max_steps = max_steps
        self.output_retries = output_retries
        #: Tool-choice policy passed to the model: "auto" | "required" | "none" |
        #: a tool name (forces that function) | an OpenAI tool_choice dict.
        self.tool_choice = tool_choice
        #: Optional ContextStrategy that trims/summarizes history before each
        #: model call to stay within the context window.
        self.context_strategy = context_strategy
        #: Execute a turn's multiple tool calls concurrently (default). Set False
        #: for ordering-sensitive tools that must run one at a time.
        self.parallel_tools = parallel_tools
        #: Cap on concurrent tool executions (0 = unbounded) when parallel.
        self.max_parallel_tools = max_parallel_tools
        #: Arbitrary provider kwargs forwarded to the model on every call
        #: (temperature, top_p, seed, max_tokens, reasoning_effort, stop,
        #: extra_body, extra_headers, …) — anything LiteLLM / the model accepts.
        self.model_settings: dict[str, Any] = dict(model_settings or {})
        self.instrument = instrument
        self.permissions: list[str] = []

        self.tools: list[Tool] = coerce_tools(tools or [])
        # Capabilities and skills are reusable bundles of tools/instructions.
        for cap in self.capabilities:
            self.tools.extend(coerce_tools(getattr(cap, "tools", [])))

        instruction_fragments: list[str] = []
        for skill in self.skills:
            self.tools.extend(getattr(skill, "tools", []))
            self.permissions.extend(getattr(skill, "permissions", []))
            if getattr(skill, "instructions", ""):
                instruction_fragments.append(skill.instructions)

        # Derive the routing description BEFORE skill fragments are folded in, so
        # it reflects the developer's own one-liner — not appended skill prose.
        # Falls back to the first line of string instructions.
        if description:
            self.description = description
        elif isinstance(instructions, str) and instructions:
            self.description = instructions.splitlines()[0].strip()
        else:
            self.description = ""

        # Compose skill instruction fragments after the base instructions.
        if instruction_fragments and isinstance(instructions, str):
            base = [instructions] if instructions else []
            instructions = "\n\n".join(base + instruction_fragments)
        self.instructions = instructions

        #: Named sub-agents this agent may hand the conversation off to. When
        #: non-empty, a framework-managed ``transfer_to_agent`` tool is injected
        #: below so the model can route by name (the multi-agent pattern).
        self.sub_agents: list[Agent[Any, Any]] = list(sub_agents or [])
        #: Max chained transfers from this run, to prevent delegation loops.
        self.transfer_depth = transfer_depth
        if self.sub_agents:
            self.tools.append(self._build_transfer_tool())

        self._model: ModelProvider | None = None
        self._runner = runner

    def _build_transfer_tool(self) -> Tool:
        """Build the built-in ``transfer_to_agent`` tool from ``sub_agents``.

        The tool itself only *records* the requested handoff into
        ``ctx.state['temp:__transfer_to__']``; the Runner inspects that flag after the
        turn's tools run and performs the actual delegation. Keeping the tool a
        pure state-setter means the model-facing contract (pick a name) is
        decoupled from the orchestration (run the sub-agent), which is what lets
        the Runner enforce validity and the transfer-depth cap centrally.
        """
        from .tools.base import FunctionTool

        agents = [_sub_unit(a) for a in self.sub_agents]
        roster = "\n".join(f"- {a.name}: {a.description}" for a in agents)
        valid_names = [a.name for a in agents]

        async def transfer_to_agent(ctx: RunContext, agent_name: str) -> str:
            # Validate here so the model sees a useful error and can recover
            # (answer itself or pick a real name) without the Runner having to
            # rewrite tool results. Only a *valid* name arms the delegation flag.
            if agent_name not in valid_names:
                return f"error: unknown agent {agent_name}; available: {', '.join(valid_names)}"
            ctx.state["temp:__transfer_to__"] = agent_name
            return f"transferring to {agent_name}"

        # The docstring is the model-facing description: it must enumerate the
        # sub-agents (name: description) so the LLM can choose the right target.
        transfer_to_agent.__doc__ = (
            "Hand the conversation off to a specialized sub-agent by name. "
            "Choose the single best-matching agent for the user's request from:\n"
            f"{roster}"
        )
        return FunctionTool(transfer_to_agent)

    @property
    def model(self) -> ModelProvider:
        """Resolve (and cache) the model provider, wrapping it for tracing."""
        if self._model is None:
            provider = resolve_model(self._model_spec)
            if self.instrument:
                from .models.instrumented import InstrumentedModel

                provider = InstrumentedModel(provider)
            self._model = provider
        return self._model

    def tool(self, fn: Callable[..., Any] | None = None, **kwargs: Any) -> Any:
        """Register a tool on this agent (decorator form)."""
        from .tools.base import FunctionTool

        def wrap(func: Callable[..., Any]) -> Callable[..., Any]:
            self.tools.append(FunctionTool(func, **kwargs))
            return func

        return wrap(fn) if fn is not None else wrap

    def as_tool(self, *, name: str | None = None, description: str | None = None) -> Any:
        """Expose this agent as a tool for another agent (Agent-as-Tool)."""
        from .tools.agent_tool import AgentTool

        return AgentTool(self, name=name, description=description)

    def _get_runner(self) -> Any:
        if self._runner is None:
            from .runner import Runner

            self._runner = Runner()
        return self._runner

    def reset(self) -> Agent[Deps, Output]:
        """Reset per-agent run state so the instance is clean for reuse.

        Clears the cached (lazily-resolved, tracing-wrapped) model provider so the
        next run re-resolves it. Conversation history is **not** held on the Agent
        — it lives in the session service — so to clear a conversation, start a new
        ``session_id`` or clear it via your :class:`SessionManager`. Returns
        ``self`` for chaining.
        """
        self._model = None
        return self

    async def run(
        self,
        prompt: str,
        *,
        deps: Deps = None,  # type: ignore[assignment]
        session_id: str | None = None,
        identity: str | None = None,
        state: State | None = None,
        usage_limits: Any | None = None,
        cancellation: Any | None = None,
        timeout: float | None = None,
        resume_id: str | None = None,
    ) -> RunResult[Output]:
        """Run the agent's model-driven loop and return a typed result.

        ``usage_limits`` (:class:`~yaab.limits.UsageLimits`) caps tokens/requests/
        tool calls; ``cancellation`` (:class:`~yaab.limits.CancellationToken`) and
        ``timeout`` (seconds) stop the run cooperatively between steps.

        ``state`` is the run's shared :class:`~yaab.state.State`. Leave it ``None``
        for a standalone run (the runner builds one over the session); a workflow
        agent or a delegation passes its own State so every participant shares one
        object — values written in one step are read in the next by key.

        ``resume_id`` makes a run fault-tolerant: when the runner has a
        checkpointer, loop progress is persisted under this key after every
        completed step, so a crashed or paused run resumes from where it left
        off if re-invoked with the same ``resume_id`` — without re-requesting the
        model turns already captured. It is inert (zero overhead) when the runner
        has no checkpointer.
        """
        return await self._get_runner().run(
            self,
            prompt,
            deps=deps,
            session_id=session_id,
            identity=identity,
            state=state,
            usage_limits=usage_limits,
            cancellation=cancellation,
            timeout=timeout,
            resume_id=resume_id,
        )

    def stream(
        self,
        prompt: Any,
        *,
        deps: Deps = None,  # type: ignore[assignment]
        session_id: str | None = None,
        identity: str | None = None,
    ) -> Any:
        """Stream the answer token-by-token (single turn, no tool loop).

        Returns an async iterator of text deltas::

            async for token in agent.stream("tell me a joke"):
                print(token, end="")
        """
        return self._get_runner().stream_text(
            self, prompt, deps=deps, session_id=session_id, identity=identity
        )

    def stream_structured(
        self,
        prompt: Any,
        *,
        output_type: type | None = None,
        deps: Deps = None,  # type: ignore[assignment]
        identity: str | None = None,
    ) -> Any:
        """Stream partial typed objects as the model generates JSON.

        Yields successive partial instances of ``output_type`` (defaults to the
        agent's ``output_type``); the final yield is the fully-validated object::

            async for partial in agent.stream_structured("...", output_type=Weather):
                render(partial)
        """
        from .streaming import stream_structured as _stream_structured

        ot = output_type or self.output_type
        return _stream_structured(self, prompt, output_type=ot, deps=deps, identity=identity)

    def stream_events(
        self,
        prompt: str,
        *,
        deps: Deps = None,  # type: ignore[assignment]
        session_id: str | None = None,
        identity: str | None = None,
        state: State | None = None,
        usage_limits: Any | None = None,
        cancellation: Any | None = None,
        timeout: float | None = None,
    ) -> Any:
        """Stream the full multi-step run as typed events, tokens included.

        Yields :class:`~yaab.types.Event` objects: ``TEXT_DELTA`` token deltas as
        the model generates, ``TOOL_CALL``/``TOOL_RESULT`` as tools run mid-run,
        and a terminal ``FINAL_OUTPUT`` + ``RUN_END`` (which carries the
        :class:`~yaab.types.RunResult`). Unlike :meth:`stream` this drives the
        whole tool loop, not just the answering turn::

            async for event in agent.stream_events("..."):
                if event.type is EventType.TEXT_DELTA:
                    print(event.payload["delta"], end="")
        """
        return self._get_runner().stream_run(
            self,
            prompt,
            deps=deps,
            session_id=session_id,
            identity=identity,
            state=state,
            usage_limits=usage_limits,
            cancellation=cancellation,
            timeout=timeout,
        )

    def run_sync(
        self,
        prompt: str,
        *,
        deps: Deps = None,  # type: ignore[assignment]
        session_id: str | None = None,
        identity: str | None = None,
        state: State | None = None,
        usage_limits: Any | None = None,
        cancellation: Any | None = None,
        timeout: float | None = None,
        resume_id: str | None = None,
    ) -> RunResult[Output]:
        """Synchronous convenience wrapper around :meth:`run`."""
        return asyncio.run(
            self.run(
                prompt,
                deps=deps,
                session_id=session_id,
                identity=identity,
                state=state,
                usage_limits=usage_limits,
                cancellation=cancellation,
                timeout=timeout,
                resume_id=resume_id,
            )
        )

    def __repr__(self) -> str:
        return f"Agent(name={self.name!r}, model={self._model_spec!r}, tools={len(self.tools)})"
