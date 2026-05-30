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
from .artifacts.manager import ArtifactManager
from .batch import batch_embed, batch_map, batch_run
from .content import Content, Part, PartKind
from .extensions import available as available_components
from .extensions import get as get_component
from .extensions import register as register_component
from .limits import CancellationToken, UsageLimits
from .memory.manager import MemoryManager
from .multiagent import LoopAgent, MapAgent, ParallelAgent, SequentialAgent, Swarm
from .prompts import PromptRegistry
from .rag import Document, KnowledgeBase
from .sessions.manager import SessionManager
from .skills import Skill
from .state import State
from .config import agent_from_dict, agent_from_yaml
from .context import KeepAll, SummarizeHistory, TruncateMessages
from .exceptions import (
    ApprovalRequired,
    GovernanceError,
    MaxStepsExceeded,
    ModelError,
    OutputValidationError,
    PolicyViolation,
    RunCancelled,
    ToolError,
    UsageLimitExceeded,
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
    # multi-agent workflow patterns
    "SequentialAgent",
    "ParallelAgent",
    "MapAgent",
    "LoopAgent",
    "Swarm",
    # managers (ADK-style)
    "SessionManager",
    "MemoryManager",
    "ArtifactManager",
    "State",
    # extensibility
    "register_component",
    "get_component",
    "available_components",
    # reusable building blocks
    "Skill",
    "PromptRegistry",
    # RAG
    "KnowledgeBase",
    "Document",
    # run controls
    "UsageLimits",
    "CancellationToken",
    # context-window management
    "TruncateMessages",
    "SummarizeHistory",
    "KeepAll",
    # declarative config
    "agent_from_dict",
    "agent_from_yaml",
    # batch / offline inference
    "batch_run",
    "batch_map",
    "batch_embed",
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
    "UsageLimitExceeded",
    "RunCancelled",
    "GovernanceError",
    "PolicyViolation",
    "ApprovalRequired",
]
