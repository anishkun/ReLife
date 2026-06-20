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
from .build.orchestrator import run_build
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


@app.command("build")
def build(
    spec: Optional[str] = typer.Argument(
        None, help="What to build, in plain language. Omit when using --resume."
    ),
    workspace: Optional[Path] = typer.Option(
        None, "--workspace", "-w", help="Directory the agent builds in."
    ),
    resume: Optional[str] = typer.Option(
        None,
        "--resume",
        help="Resume a build. Pass a build id, or use the flag with no value "
        "to resume the most recent build for this workspace.",
        is_flag=False,
        flag_value="__latest__",
    ),
    budget: Optional[float] = typer.Option(
        None, "--budget", help="Optional max usage-equivalent budget (USD) for the run."
    ),
) -> None:
    """Orchestrate a large, multi-milestone build (decompose → delegate → resume)."""
    ws = _resolve_workspace(workspace)
    typer.secho(f"workspace: {ws}", fg=typer.colors.BRIGHT_BLACK)

    resume_id: Optional[str] = None
    if resume is not None:
        resume_id = None if resume == "__latest__" else resume
    elif not spec:
        typer.secho("Provide a spec to build, or --resume a prior build.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    can_use_tool = make_permission_callback(ws)
    hooks = memory_hooks()
    anyio.run(
        lambda: run_build(
            spec,
            cwd=ws,
            can_use_tool=can_use_tool,
            hooks=hooks,
            resume_id=resume_id,
            budget=budget,
        )
    )


if __name__ == "__main__":
    app()
