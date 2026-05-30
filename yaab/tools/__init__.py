"""Tools: typed function tools, agent-as-tool, and MCP interop."""

from __future__ import annotations

from .agent_tool import AgentTool
from .base import FunctionTool, Tool, coerce_tools, tool
from .mcp import MCPTool, mcp_toolset

__all__ = [
    "Tool",
    "FunctionTool",
    "tool",
    "coerce_tools",
    "AgentTool",
    "MCPTool",
    "mcp_toolset",
]
