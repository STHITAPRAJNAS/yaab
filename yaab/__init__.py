"""YAAB — Yet Another Agent Builder.

A type-safe, governance-first agent SDK with a Rust performance core. Type-safe,
optimizable, durable, and simple — the best ideas from across the agent
ecosystem on one runtime, on a universal LiteLLM model layer.

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

# Side-effect import: registers the in-process + shared (Redis) rate limiters
# under the ``ratelimiter`` component kind so a global rate budget is selectable
# by name. Cheap — the redis driver is only imported when that backend is built.
from . import ratelimiters as _ratelimiters  # noqa: F401
from .agent import Agent
from .artifacts.manager import ArtifactManager
from .batch import batch_embed, batch_map, batch_run
from .conditions import (
    Branch,
    Condition,
    Guard,
    Step,
    and_,
    failed,
    loop_exhausted,
    not_,
    ok,
    or_,
    output_contains,
    skipped,
    state_eq,
    state_ge,
    timed_out,
    when,
)
from .config import agent_from_dict, agent_from_yaml, runner_from_dict
from .content import Content, Part, PartKind
from .context import KeepAll, RelevanceFilter, SummarizeHistory, TruncateMessages
from .deploy_backends import DurableBackends, durable_backends
from .eval import available_metrics, get_metric, register_metric
from .exceptions import (
    ApprovalPending,
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
from .flow import Flow
from .governance.approval import ToolApprovalPlugin
from .governance.approvals import (
    ApprovalDecision,
    ApprovalRequest,
    ApprovalStore,
    InMemoryApprovalStore,
    SQLiteApprovalStore,
)
from .governance.approvals_decide import Decision, ResumeBundle
from .governance.eval import ToolTrajectoryMatch
from .governance.evalset import EvalCase, EvalSet
from .graph.state import RetryPolicy
from .limits import CancellationToken, UsageLimits
from .memory.extraction import MemoryExtractor
from .memory.manager import MemoryManager
from .models.router import ModelRouter
from .multiagent import (
    LoopAgent,
    MapAgent,
    ParallelAgent,
    RouterAgent,
    SequentialAgent,
    Swarm,
)
from .prompts import PromptRegistry
from .rag import Document, KnowledgeBase
from .rag.memory_service import KnowledgeBaseMemory
from .runner import Runner
from .runs import (
    CronStore,
    InMemoryRunStore,
    InMemoryTraceStore,
    RunRecord,
    RunStatus,
    RunStore,
    RunWorker,
    SQLiteRunStore,
    SQLiteTraceStore,
    StoreCancellationToken,
    TraceStore,
    warn_if_ephemeral,
)
from .sessions.manager import SessionManager
from .skills import Skill
from .state import State
from .tools import AgentTool, FunctionTool, tool
from .tools.auth import ToolAuth, ToolAuthRequired, ToolCredential
from .tools.builtin.ask_user import ask_user
from .tools.openapi import OpenAPITool, openapi_toolset
from .types import Event, EventType, Message, Pending, RunContext, RunResult, Usage

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
    "RouterAgent",
    "Flow",
    # conditional execution + routing
    "Step",
    "Branch",
    "Condition",
    "Guard",
    "when",
    "and_",
    "or_",
    "not_",
    "failed",
    "timed_out",
    "ok",
    "skipped",
    "loop_exhausted",
    "output_contains",
    "state_eq",
    "state_ge",
    # managers
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
    # memory intelligence
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
    # durable background runs (survive restarts, span replicas)
    "RunStore",
    "RunRecord",
    "RunStatus",
    "InMemoryRunStore",
    "SQLiteRunStore",
    "RunWorker",
    "CronStore",
    "StoreCancellationToken",
    # out-of-band human sign-off
    "ToolApprovalPlugin",
    "ApprovalStore",
    "ApprovalRequest",
    "ApprovalDecision",
    "InMemoryApprovalStore",
    "SQLiteApprovalStore",
    # unified human-in-the-loop surface (pause -> decide -> resume)
    "Pending",
    "Decision",
    "ResumeBundle",
    "ask_user",
    # per-run trace store (replay a run with full per-step detail)
    "TraceStore",
    "InMemoryTraceStore",
    "SQLiteTraceStore",
    # multi-replica wiring + durability guardrail
    "durable_backends",
    "DurableBackends",
    "warn_if_ephemeral",
    # shared rate-limit budget across replicas (lazy: needs the redis driver)
    "RedisRateLimiter",
    # durable artifact backends (lazy: psycopg / redis only when used)
    "SQLiteArtifactService",
    "PostgresArtifactService",
    "RedisArtifactService",
    # context-window management
    "TruncateMessages",
    "SummarizeHistory",
    "RelevanceFilter",
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
    "ApprovalPending",
]


def __getattr__(name: str) -> object:
    # Lazy top-level exports for backends whose drivers are optional extras, so
    # ``import yaab`` never needs psycopg / redis installed. The class is only
    # imported on first attribute access (e.g. ``yaab.RedisRateLimiter``).
    if name == "RedisRateLimiter":
        from .models.distributed_ratelimit import RedisRateLimiter

        return RedisRateLimiter
    if name in ("SQLiteArtifactService", "PostgresArtifactService", "RedisArtifactService"):
        from . import artifacts

        return getattr(artifacts, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
