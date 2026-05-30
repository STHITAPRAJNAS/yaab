"""YAAB — Yet Another Agent Builder.

A type-safe, governance-first agent SDK with a Rust performance core. Type-safe
like Pydantic AI, optimizable like DSPy, durable like LangGraph, clean like
Google ADK, simple like Strands — on a universal LiteLLM model layer.

Quickstart::

    from yaab import Agent

    agent = Agent("assistant", model="openai/gpt-4o", instructions="Be concise.")
    print(agent.run_sync("Say hello").output)

Offline (no API key)::

    from yaab import Agent
    from yaab.testing import TestModel

    agent = Agent("assistant", model=TestModel("hi!"))
    assert agent.run_sync("hello").output == "hi!"
"""

from __future__ import annotations

from . import _core
from .agent import Agent
from .content import Content, Part, PartKind
from .exceptions import (
    GovernanceError,
    MaxStepsExceeded,
    ModelError,
    OutputValidationError,
    PolicyViolation,
    ToolError,
    YaabError,
)
from .runner import Runner
from .tools import AgentTool, FunctionTool, tool
from .types import Event, EventType, Message, RunContext, RunResult, Usage

__version__ = "0.1.0"

#: Which performance backend is active: ``"rust"`` or ``"python"``.
BACKEND = _core.backend()

__all__ = [
    "__version__",
    "BACKEND",
    "Agent",
    "Runner",
    "tool",
    "FunctionTool",
    "AgentTool",
    "RunContext",
    "RunResult",
    "Message",
    "Content",
    "Part",
    "PartKind",
    "Event",
    "EventType",
    "Usage",
    # exceptions
    "YaabError",
    "ModelError",
    "ToolError",
    "OutputValidationError",
    "MaxStepsExceeded",
    "GovernanceError",
    "PolicyViolation",
]
