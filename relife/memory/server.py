"""In-process MCP server exposing memory tools to the agent.

Named ``relife_memory`` so tools surface as ``mcp__relife_memory__*`` — matched
by the permission policy's trusted ``mcp__relife`` prefix (auto-allowed).

Shipping memory as an MCP server (even in-process) means the agent-facing
contract is identical when it is served out of process — and it now is: the
tool definitions live in ``tools.py`` and the standalone ``mcp_server.py``
serves the same specs to agents on any LLM.

``memory_server(client)`` binds the tools to one memory client (an agent's
``ScopedMemoryClient``). The module-level tool objects below serve the main
agent and resolve ``default_client`` from this module at call time.
"""

from __future__ import annotations

from typing import Any, Callable

from claude_agent_sdk import create_sdk_mcp_server, tool

from . import tools as _tools
from .client import default_client


def _sdk_tool(spec: _tools.ToolSpec, get_client: Callable[[], Any]):
    """Wrap one transport-neutral spec as a Claude Agent SDK tool."""

    @tool(spec.name, spec.description, spec.input_schema)
    async def _call(args: dict[str, Any]) -> dict[str, Any]:
        text, is_error = await spec.handler(get_client(), args)
        out: dict[str, Any] = {"content": [{"type": "text", "text": text}]}
        if is_error:
            out["is_error"] = True
        return out

    return _call


def _main_client():
    return default_client()


memory_save = _sdk_tool(_tools.MEMORY_SAVE, _main_client)
memory_recall = _sdk_tool(_tools.MEMORY_RECALL, _main_client)
memory_forget = _sdk_tool(_tools.MEMORY_FORGET, _main_client)
skill_write = _sdk_tool(_tools.SKILL_WRITE, _main_client)
skill_find = _sdk_tool(_tools.SKILL_FIND, _main_client)
workflow_save = _sdk_tool(_tools.WORKFLOW_SAVE, _main_client)
workflow_find = _sdk_tool(_tools.WORKFLOW_FIND, _main_client)
memory_consolidate = _sdk_tool(_tools.MEMORY_CONSOLIDATE, _main_client)
memory_dream = _sdk_tool(_tools.MEMORY_DREAM, _main_client)

_MAIN_TOOLS = [
    memory_save,
    memory_recall,
    memory_forget,
    skill_write,
    skill_find,
    workflow_save,
    workflow_find,
    memory_consolidate,
    memory_dream,
]


def memory_server(client: Any = None):
    """Return the McpSdkServerConfig for the memory + skills + workflows server.

    ``client`` binds the tools to one memory client (an agent with its own
    space gets its ``ScopedMemoryClient``); omitted, they serve the main agent.
    """
    if client is None:
        sdk_tools = _MAIN_TOOLS
    else:
        sdk_tools = [_sdk_tool(spec, lambda: client) for spec in _tools.INTERNAL_TOOLS]
    return create_sdk_mcp_server(name="relife_memory", version="0.3.0", tools=sdk_tools)
