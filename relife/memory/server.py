"""In-process MCP server exposing memory tools to the agent.

Named ``relife_memory`` so tools surface as ``mcp__relife_memory__*`` — matched
by the permission policy's trusted ``mcp__relife`` prefix (auto-allowed).

Shipping memory as an MCP server (even in-process) means the agent-facing
contract is identical when we later split it into a standalone server.
"""

from __future__ import annotations

from typing import Any

from claude_agent_sdk import create_sdk_mcp_server, tool

from . import skills, store


@tool(
    "memory_save",
    "Save a durable fact, user preference, or task lesson to long-term memory so "
    "future sessions can recall it. Use for things worth remembering across tasks "
    "(preferences, project conventions, where things live) — not transient detail.",
    {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "The thing to remember, self-contained."},
            "kind": {
                "type": "string",
                "enum": ["fact", "preference", "episode"],
                "description": "fact = general truth; preference = how the user likes things; episode = a task outcome.",
            },
            "tags": {"type": "string", "description": "Optional comma-separated keywords to aid recall."},
        },
        "required": ["text"],
    },
)
async def memory_save(args: dict[str, Any]) -> dict[str, Any]:
    try:
        mid = store.save(
            text=args["text"],
            kind=args.get("kind", "fact"),
            tags=args.get("tags", ""),
        )
    except Exception as e:  # noqa: BLE001 - surface to the model
        return {"content": [{"type": "text", "text": f"Error saving memory: {e}"}], "is_error": True}
    return {"content": [{"type": "text", "text": f"Saved memory #{mid}."}]}


@tool(
    "memory_recall",
    "Search long-term memory for facts/preferences/lessons relevant to a query. "
    "Call this when starting a task to see what you already know about the user "
    "or project.",
    {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to look up."},
            "k": {"type": "integer", "description": "Max results (default 5)."},
        },
        "required": ["query"],
    },
)
async def memory_recall(args: dict[str, Any]) -> dict[str, Any]:
    hits = store.recall(args["query"], k=int(args.get("k", 5)))
    if not hits:
        return {"content": [{"type": "text", "text": "(no relevant memories)"}]}
    lines = [f"- [{m.kind}] {m.text}" + (f"  ({m.tags})" if m.tags else "") for m in hits]
    return {"content": [{"type": "text", "text": "\n".join(lines)}]}


@tool(
    "skill_write",
    "Record a reusable procedure (a 'skill') after you succeed at a task that you "
    "(or future-you) will likely do again — e.g. 'scaffold a Python CLI', 'push a "
    "new repo to GitHub here'. Writing skills is how you get faster and more "
    "reliable over time. Capture the concrete steps that worked. Re-writing an "
    "existing skill name updates it.",
    {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Short skill name, e.g. 'scaffold-python-cli'."},
            "when_to_use": {"type": "string", "description": "One line: the situation this applies to."},
            "steps": {"type": "string", "description": "The procedure, as concrete steps (Markdown)."},
        },
        "required": ["name", "when_to_use", "steps"],
    },
)
async def skill_write(args: dict[str, Any]) -> dict[str, Any]:
    try:
        slug = skills.write_skill(args["name"], args.get("when_to_use", ""), args["steps"])
    except Exception as e:  # noqa: BLE001
        return {"content": [{"type": "text", "text": f"Error writing skill: {e}"}], "is_error": True}
    return {"content": [{"type": "text", "text": f"Saved skill '{slug}'."}]}


@tool(
    "skill_find",
    "Search your saved skills for procedures relevant to the current task. Call "
    "this when starting a task to reuse a proven approach instead of figuring it "
    "out from scratch.",
    {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What you're about to do."},
            "k": {"type": "integer", "description": "Max results (default 3)."},
        },
        "required": ["query"],
    },
)
async def skill_find(args: dict[str, Any]) -> dict[str, Any]:
    hits = skills.find_skills(args["query"], k=int(args.get("k", 3)))
    if not hits:
        return {"content": [{"type": "text", "text": "(no matching skills yet)"}]}
    blocks = [f"## {s.name}\nWhen to use: {s.when_to_use}\n\n{s.body}" for s in hits]
    return {"content": [{"type": "text", "text": "\n\n---\n\n".join(blocks)}]}


def memory_server():
    """Return the McpSdkServerConfig for the memory + skills server."""
    return create_sdk_mcp_server(
        name="relife_memory",
        version="0.1.0",
        tools=[memory_save, memory_recall, skill_write, skill_find],
    )
