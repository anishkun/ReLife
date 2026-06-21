"""In-process MCP server exposing memory tools to the agent.

Named ``relife_memory`` so tools surface as ``mcp__relife_memory__*`` — matched
by the permission policy's trusted ``mcp__relife`` prefix (auto-allowed).

Shipping memory as an MCP server (even in-process) means the agent-facing
contract is identical when we later split it into a standalone server.
"""

from __future__ import annotations

from typing import Any

from claude_agent_sdk import create_sdk_mcp_server, tool

from . import consolidate, skills, store, workflows


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
                "enum": ["fact", "preference", "episode", "pattern"],
                "description": "fact = general truth; preference = how the user likes things; episode = a task outcome; pattern = a recurring regularity.",
            },
            "tags": {"type": "string", "description": "Optional comma-separated keywords to aid recall."},
            "importance": {
                "type": "number",
                "description": "0..1 salience. Higher = resists fading. Use ~0.8+ for things that must persist (key preferences/conventions); ~0.5 default; lower for transient context.",
            },
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
            importance=args.get("importance"),
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
    hits = store.recall(args["query"], k=int(args.get("k", 5)), reinforce=True)
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


@tool(
    "memory_forget",
    "Soft-forget a memory that is no longer useful (e.g. a finished work item). "
    "It is archived (kept but excluded from recall), not destroyed. Prefer this "
    "over leaving stale memories to clutter recall — though unused memories also "
    "fade on their own over time.",
    {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Recall query identifying the memory to forget."},
        },
        "required": ["query"],
    },
)
async def memory_forget(args: dict[str, Any]) -> dict[str, Any]:
    hits = store.recall(args["query"], k=1)
    if not hits:
        return {"content": [{"type": "text", "text": "(nothing matched; nothing forgotten)"}]}
    store.archive(hits[0].id)
    return {"content": [{"type": "text", "text": f"Archived: {hits[0].text}"}]}


@tool(
    "workflow_save",
    "Record a reusable multi-step WORKFLOW — an ordered chain of steps (often "
    "stitching several skills/actions together) for a recurring multi-stage job, "
    "e.g. 'scaffold → test → create repo → push'. Use this (vs a single skill) "
    "when the value is in the sequence. Re-saving the same name updates it.",
    {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Short workflow name, e.g. 'ship-new-service'."},
            "when_to_use": {"type": "string", "description": "One line: the multi-step situation this applies to."},
            "steps": {"type": "string", "description": "Ordered steps (Markdown); may reference skills by name."},
            "trigger": {"type": "string", "description": "Optional comma-separated keywords/tools that signal this workflow."},
        },
        "required": ["name", "when_to_use", "steps"],
    },
)
async def workflow_save(args: dict[str, Any]) -> dict[str, Any]:
    try:
        slug = workflows.write_workflow(
            args["name"], args.get("when_to_use", ""), args["steps"], args.get("trigger", "")
        )
    except Exception as e:  # noqa: BLE001
        return {"content": [{"type": "text", "text": f"Error writing workflow: {e}"}], "is_error": True}
    return {"content": [{"type": "text", "text": f"Saved workflow '{slug}'."}]}


@tool(
    "workflow_find",
    "Search your saved workflows for a multi-step plan relevant to the current "
    "job. Call this when a task looks like a recurring multi-stage process.",
    {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What multi-step job you're about to do."},
            "k": {"type": "integer", "description": "Max results (default 3)."},
        },
        "required": ["query"],
    },
)
async def workflow_find(args: dict[str, Any]) -> dict[str, Any]:
    hits = workflows.find_workflows(args["query"], k=int(args.get("k", 3)))
    if not hits:
        return {"content": [{"type": "text", "text": "(no matching workflows yet)"}]}
    blocks = [
        f"## {w.name}\nWhen to use: {w.when_to_use}\n\n{w.body}" for w in hits
    ]
    return {"content": [{"type": "text", "text": "\n\n---\n\n".join(blocks)}]}


@tool(
    "memory_consolidate",
    "Run a consolidation ('sleep') pass over memory: fade/archive unused "
    "memories, merge duplicates, and detect recurring patterns and tool "
    "sequences (synthesizing workflows from them). This usually runs "
    "automatically; call it explicitly to reflect and tidy memory now.",
    {"type": "object", "properties": {}},
)
async def memory_consolidate(args: dict[str, Any]) -> dict[str, Any]:
    try:
        report = consolidate.run_consolidation()
    except Exception as e:  # noqa: BLE001
        return {"content": [{"type": "text", "text": f"Error consolidating: {e}"}], "is_error": True}
    lines = [f"Consolidation: {report.summary()}."]
    if report.workflows_created:
        lines.append("New workflows: " + ", ".join(report.workflows_created))
    if report.patterns:
        lines.append("Patterns:\n" + "\n".join(f"- {p}" for p in report.patterns[:10]))
    return {"content": [{"type": "text", "text": "\n".join(lines)}]}


def memory_server():
    """Return the McpSdkServerConfig for the memory + skills + workflows server."""
    return create_sdk_mcp_server(
        name="relife_memory",
        version="0.2.0",
        tools=[
            memory_save,
            memory_recall,
            memory_forget,
            skill_write,
            skill_find,
            workflow_save,
            workflow_find,
            memory_consolidate,
        ],
    )
