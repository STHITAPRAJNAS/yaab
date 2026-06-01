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
from .config import agent_from_dict, agent_from_yaml, runner_from_dict
from .content import Content, Part, PartKind
from .context import KeepAll, SummarizeHistory, TruncateMessages
from .eval import available_metrics, get_metric, register_metric
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
from .extensions import available as available_components
from .extensions import get as get_component
from .extensions import register as register_component
from .governance.eval import ToolTrajectoryMatch
from .governance.evalset import EvalCase, EvalSet
from .graph.state import RetryPolicy
from .limits import CancellationToken, UsageLimits
from .memory.extraction import MemoryExtractor
from .memory.manager import MemoryManager
from .models.router import ModelRouter
from .multiagent import LoopAgent, MapAgent, ParallelAgent, SequentialAgent, Swarm
from .prompts import PromptRegistry
from .rag import Document, KnowledgeBase
from .rag.memory_service import KnowledgeBaseMemory
from .runner import Runner
from .sessions.manager import SessionManager
from .skills import Skill
from .state import State
from .tools import AgentTool, FunctionTool, tool
from .tools.auth import ToolAuth, ToolAuthRequired, ToolCredential
from .tools.openapi import OpenAPITool, openapi_toolset
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
    # eval metrics (built-in + RAGAS/DeepEval adapters)
    "register_metric",
    "get_metric",
    "available_metrics",
    # reusable building blocks
    "Skill",
    "PromptRegistry",
    # RAG
    "KnowledgeBase",
    "Document",
    # memory intelligence (ADK MemoryBank parity)
    "MemoryExtractor",
    "KnowledgeBaseMemory",
    # model intelligence
    "ModelRouter",
    # graph retry policies
    "RetryPolicy",
    # eval depth (portable evalsets + trajectory metric)
    "EvalSet",
    "EvalCase",
    "ToolTrajectoryMatch",
    # OpenAPI toolset
    "openapi_toolset",
    "OpenAPITool",
    # tool-level auth (credentials + OAuth2 consent)
    "ToolAuth",
    "ToolCredential",
    "ToolAuthRequired",
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
    "runner_from_dict",
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
