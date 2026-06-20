"""ReLife command-line interface.

    relife do "<task>"   [--workspace PATH]
    relife chat          [--workspace PATH]
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import anyio
import typer

from . import config
from .agent import run_chat, run_task
from .hooks import memory_hooks
from .permissions import make_permission_callback

app = typer.Typer(
    add_completion=False,
    help="ReLife — a personal agent that acts through MCP and learns over time.",
)


def _resolve_workspace(workspace: Optional[Path]) -> Path:
    config.ensure_dirs()
    ws = (workspace or config.DEFAULT_WORKSPACE).resolve()
    ws.mkdir(parents=True, exist_ok=True)
    return ws


@app.command("do")
def do(
    task: str = typer.Argument(..., help="What you want done, in plain language."),
    workspace: Optional[Path] = typer.Option(
        None, "--workspace", "-w", help="Directory the agent works in."
    ),
) -> None:
    """Run a single task to completion."""
    ws = _resolve_workspace(workspace)
    typer.secho(f"workspace: {ws}", fg=typer.colors.BRIGHT_BLACK)
    can_use_tool = make_permission_callback(ws)
    mcp_servers = config.default_mcp_servers()
    hooks = memory_hooks()
    anyio.run(
        lambda: run_task(
            task, cwd=ws, can_use_tool=can_use_tool, mcp_servers=mcp_servers, hooks=hooks
        )
    )


@app.command("chat")
def chat(
    workspace: Optional[Path] = typer.Option(
        None, "--workspace", "-w", help="Directory the agent works in."
    ),
) -> None:
    """Start an interactive multi-turn session."""
    ws = _resolve_workspace(workspace)
    typer.secho(f"workspace: {ws}", fg=typer.colors.BRIGHT_BLACK)
    can_use_tool = make_permission_callback(ws)
    mcp_servers = config.default_mcp_servers()
    hooks = memory_hooks()
    anyio.run(
        lambda: run_chat(
            cwd=ws, can_use_tool=can_use_tool, mcp_servers=mcp_servers, hooks=hooks
        )
    )


if __name__ == "__main__":
    app()
