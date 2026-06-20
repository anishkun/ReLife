"""In-process MCP server exposing the build ledger to the orchestrator.

Named ``relife_build`` so tools surface as ``mcp__relife_build__*`` — already
matched by the permission policy's trusted ``mcp__relife`` prefix (auto-allowed,
no permission change needed).

Unlike the memory server, this one is bound to a *specific* ``BuildLedger``
instance for the current run (via closure), so the tools mutate the right
ledger on disk.
"""

from __future__ import annotations

from typing import Any

from claude_agent_sdk import create_sdk_mcp_server, tool

from .ledger import BuildLedger


def build_tools(ledger: BuildLedger):
    """Return the ledger MCP tools bound to ``ledger`` (also used by tests)."""

    @tool(
        "build_plan_set",
        "Record the decomposition of this build as an ordered list of milestones. "
        "Call this once after you've thought through the architecture, before "
        "starting work. Each milestone should be independently implementable and "
        "verifiable. Replaces any existing plan.",
        {
            "type": "object",
            "properties": {
                "milestones": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Ordered milestone titles, e.g. ['Scaffold FastAPI app', 'Add SQLite models', ...].",
                }
            },
            "required": ["milestones"],
        },
    )
    async def build_plan_set(args: dict[str, Any]) -> dict[str, Any]:
        titles = [str(t).strip() for t in args.get("milestones", []) if str(t).strip()]
        if not titles:
            return {"content": [{"type": "text", "text": "No milestones provided."}], "is_error": True}
        ledger.set_plan(titles)
        return {"content": [{"type": "text", "text": f"Recorded {len(titles)} milestones.\n\n{ledger.status_brief()}"}]}

    @tool(
        "build_milestone_update",
        "Update one milestone's status after delegating it to a builder subagent. "
        "Set 'in_progress' before delegating, 'done' (with a concise summary of "
        "what was built and how it was verified) after it succeeds, or 'failed'.",
        {
            "type": "object",
            "properties": {
                "id": {"type": "integer", "description": "The milestone id from build_status."},
                "status": {
                    "type": "string",
                    "enum": ["pending", "in_progress", "done", "failed"],
                },
                "summary": {
                    "type": "string",
                    "description": "Concise result: what was built, key files, how verified. Keep it short.",
                },
            },
            "required": ["id", "status"],
        },
    )
    async def build_milestone_update(args: dict[str, Any]) -> dict[str, Any]:
        try:
            ledger.update_milestone(
                int(args["id"]), str(args["status"]), str(args.get("summary", ""))
            )
        except (KeyError, ValueError) as e:
            return {"content": [{"type": "text", "text": f"Error: {e}"}], "is_error": True}
        return {"content": [{"type": "text", "text": ledger.status_brief()}]}

    @tool(
        "build_status",
        "Read the current build ledger: the spec and every milestone with its "
        "status and summary. Call this at the start (especially on resume) to see "
        "what's already done and what's left.",
        {"type": "object", "properties": {}},
    )
    async def build_status(args: dict[str, Any]) -> dict[str, Any]:
        text = f"Spec: {ledger.spec}\n\n{ledger.status_brief()}"
        return {"content": [{"type": "text", "text": text}]}

    return [build_plan_set, build_milestone_update, build_status]


def build_server(ledger: BuildLedger):
    """Return the MCP server config bound to ``ledger``."""
    return create_sdk_mcp_server(
        name="relife_build",
        version="0.1.0",
        tools=build_tools(ledger),
    )
