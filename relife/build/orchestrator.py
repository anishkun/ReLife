"""Drive an orchestrated, resumable build.

``run_build`` wires the build ledger + ledger MCP server + ``builder`` subagent
into the standard agent loop, then streams the orchestrator to completion. It
persists the CLI ``session_id`` so a later ``--resume`` can continue the same
conversation; the ledger + on-disk files make resume robust even if that
session is gone.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from claude_agent_sdk import ClaudeSDKClient, ResultMessage

from .. import config
from ..agent import _render, build_options, console, preset_system_prompt
from .agents import ORCHESTRATOR_PROMPT_FILE, build_agents
from .ledger import BuildLedger
from .server import build_server


def _initial_prompt(spec: str) -> str:
    return (
        "Build the following project. First think through the architecture and "
        "decompose it into ordered milestones, then record them with "
        "`build_plan_set` and work through them by delegating each to a `builder` "
        "subagent.\n\nPROJECT SPEC:\n" + spec
    )


def _resume_prompt(ledger: BuildLedger) -> str:
    return (
        "Resume this in-progress build. Here is the current ledger — call "
        "`build_status` to confirm, spot-check that 'done' milestones really exist "
        "in the workspace, then continue from the first unfinished milestone "
        "(do not redo completed work).\n\n"
        f"SPEC:\n{ledger.spec}\n\nPROGRESS:\n{ledger.status_brief()}"
    )


async def run_build(
    spec: str | None,
    *,
    cwd: Path,
    can_use_tool: Any,
    hooks: dict[str, Any] | None = None,
    resume_id: str | None = None,
    budget: float | None = None,
) -> None:
    """Run (or resume) an orchestrated build to completion, streaming output.

    Fresh build: pass ``spec``. Resume: pass ``resume_id`` (or omit it to resume
    the most recent build for ``cwd``); ``spec`` is then ignored.
    """
    resuming = resume_id is not None or not spec
    if resuming:
        ledger = BuildLedger.load(resume_id) if resume_id else BuildLedger.latest_for(cwd)
        if ledger is None:
            console.print(
                "[red]No prior build found to resume for this workspace.[/] "
                "Start one with [bold]relife build \"<spec>\"[/]."
            )
            return
        prompt = _resume_prompt(ledger)
        console.print(f"[dim]resuming build {ledger.build_id}[/]")
    else:
        ledger = BuildLedger.create(spec, cwd)  # type: ignore[arg-type]
        prompt = _initial_prompt(spec)  # type: ignore[arg-type]
        console.print(f"[dim]build {ledger.build_id} — ledger: {ledger.md_path}[/]")

    mcp_servers = config.default_mcp_servers()
    mcp_servers["relife_build"] = build_server(ledger)

    options = build_options(
        cwd=cwd,
        can_use_tool=can_use_tool,
        mcp_servers=mcp_servers,
        hooks=hooks,
        system_prompt=preset_system_prompt(ORCHESTRATOR_PROMPT_FILE),
        agents=build_agents(),
        resume=ledger.session_id if resuming else None,
        max_budget_usd=budget,
    )

    async with ClaudeSDKClient(options=options) as client:
        await client.query(prompt)
        async for msg in client.receive_response():
            _render(msg)
            if isinstance(msg, ResultMessage) and getattr(msg, "session_id", None):
                # Persist the session handle so the next --resume can continue it.
                ledger.set_session_id(msg.session_id)

    if ledger.is_complete():
        console.print(f"[green]build {ledger.build_id} complete[/] — {len(ledger.done())} milestones")
    else:
        pend = len(ledger.pending())
        console.print(
            f"[yellow]build {ledger.build_id} paused[/] — {pend} milestone(s) left. "
            f"Resume with [bold]relife build --resume {ledger.build_id}[/]."
        )
