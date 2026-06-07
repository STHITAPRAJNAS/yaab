"""Tools: typed function tools, agent-as-tool, and MCP interop."""

from __future__ import annotations

from .adapters import adapt_tool, from_crewai_tool, from_langchain_tool
from .agent_tool import AgentTool
from .auth import ToolAuth, ToolAuthRequired, ToolCredential, as_headers
from .base import FunctionTool, Tool, coerce_tools, tool
from .mcp import MCPTool, mcp_toolset
from .openapi import OpenAPITool, openapi_toolset

__all__ = [
    "Tool",
    "FunctionTool",
    "tool",
    "coerce_tools",
    "AgentTool",
    "MCPTool",
    "mcp_toolset",
    "OpenAPITool",
    "openapi_toolset",
    "ToolAuth",
    "ToolAuthRequired",
    "ToolCredential",
    "as_headers",
    # reuse tools from other ecosystems
    "adapt_tool",
    "from_langchain_tool",
    "from_crewai_tool",
]
