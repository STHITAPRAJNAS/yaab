"""The typed :class:`Agent` — YAAB's primary developer-facing abstraction.

``Agent[Deps, Output]`` is generic over a dependency-injection type and an
output type, fusing Pydantic AI's type-safety with ADK's agent/runner split.
The three-line "hello agent" works with zero ceremony; every layer underneath
(runner, sessions, governance, graph) is openable when you need it.

    agent = Agent("assistant", model="openai/gpt-4o", instructions="Be helpful.")
    result = agent.run_sync("Hello!")
    print(result.output)
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Generic, Optional, Union

from .models import ModelProvider, resolve_model
from .tools.base import Tool, coerce_tools
from .types import Deps, Output, RunContext, RunResult

_NoneType = type(None)  # module-level singleton (avoids a call in arg defaults)


class Agent(Generic[Deps, Output]):
    """A type-safe agent: a model + instructions + tools + an output contract."""

    def __init__(
        self,
        name: str,
        *,
        model: Union[str, ModelProvider] = "openai/gpt-4o",
        instructions: Union[str, Callable[[RunContext[Deps]], str]] = "",
        tools: Optional[list[Any]] = None,
        deps_type: type = _NoneType,
        output_type: type = str,
        guardrails: Optional[list[Any]] = None,
        capabilities: Optional[list[Any]] = None,
        skills: Optional[list[Any]] = None,
        registry_id: Optional[str] = None,
        max_steps: int = 8,
        output_retries: int = 2,
        tool_choice: Optional[Any] = None,
        context_strategy: Optional[Any] = None,
        runner: Optional[Any] = None,
        instrument: bool = True,
    ) -> None:
        self.name = name
        self._model_spec = model
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

        # Compose skill instruction fragments after the base instructions.
        if instruction_fragments and isinstance(instructions, str):
            base = [instructions] if instructions else []
            instructions = "\n\n".join(base + instruction_fragments)
        self.instructions = instructions

        self._model: Optional[ModelProvider] = None
        self._runner = runner

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

    def tool(self, fn: Optional[Callable[..., Any]] = None, **kwargs: Any) -> Any:
        """Register a tool on this agent (decorator form)."""
        from .tools.base import FunctionTool

        def wrap(func: Callable[..., Any]) -> Callable[..., Any]:
            self.tools.append(FunctionTool(func, **kwargs))
            return func

        return wrap(fn) if fn is not None else wrap

    def as_tool(self, *, name: Optional[str] = None, description: Optional[str] = None) -> Any:
        """Expose this agent as a tool for another agent (Agent-as-Tool)."""
        from .tools.agent_tool import AgentTool

        return AgentTool(self, name=name, description=description)

    def _get_runner(self) -> Any:
        if self._runner is None:
            from .runner import Runner

            self._runner = Runner()
        return self._runner

    async def run(
        self,
        prompt: str,
        *,
        deps: Deps = None,  # type: ignore[assignment]
        session_id: Optional[str] = None,
        identity: Optional[str] = None,
        usage_limits: Optional[Any] = None,
        cancellation: Optional[Any] = None,
        timeout: Optional[float] = None,
    ) -> RunResult[Output]:
        """Run the agent's model-driven loop and return a typed result.

        ``usage_limits`` (:class:`~yaab.limits.UsageLimits`) caps tokens/requests/
        tool calls; ``cancellation`` (:class:`~yaab.limits.CancellationToken`) and
        ``timeout`` (seconds) stop the run cooperatively between steps.
        """
        return await self._get_runner().run(
            self,
            prompt,
            deps=deps,
            session_id=session_id,
            identity=identity,
            usage_limits=usage_limits,
            cancellation=cancellation,
            timeout=timeout,
        )

    def stream(
        self,
        prompt: Any,
        *,
        deps: Deps = None,  # type: ignore[assignment]
        session_id: Optional[str] = None,
        identity: Optional[str] = None,
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
        output_type: Optional[type] = None,
        deps: Deps = None,  # type: ignore[assignment]
        identity: Optional[str] = None,
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

    def run_sync(
        self,
        prompt: str,
        *,
        deps: Deps = None,  # type: ignore[assignment]
        session_id: Optional[str] = None,
        identity: Optional[str] = None,
        usage_limits: Optional[Any] = None,
        cancellation: Optional[Any] = None,
        timeout: Optional[float] = None,
    ) -> RunResult[Output]:
        """Synchronous convenience wrapper around :meth:`run`."""
        return asyncio.run(
            self.run(
                prompt,
                deps=deps,
                session_id=session_id,
                identity=identity,
                usage_limits=usage_limits,
                cancellation=cancellation,
                timeout=timeout,
            )
        )

    def __repr__(self) -> str:
        return f"Agent(name={self.name!r}, model={self._model_spec!r}, tools={len(self.tools)})"
