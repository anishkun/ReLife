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
        None,
        help="What to build, in plain language. With --resume, optionally the "
        "build id to resume (omit it to resume the most recent build).",
    ),
    workspace: Optional[Path] = typer.Option(
        None, "--workspace", "-w", help="Directory the agent builds in."
    ),
    resume: bool = typer.Option(
        False,
        "--resume",
        help="Resume a paused build. Optionally pass its build id as the "
        "argument; otherwise resume the most recent build for this workspace.",
    ),
    budget: Optional[float] = typer.Option(
        None, "--budget", help="Optional max usage-equivalent budget (USD) for the run."
    ),
) -> None:
    """Orchestrate a large, multi-milestone build (decompose → delegate → resume)."""
    ws = _resolve_workspace(workspace)
    typer.secho(f"workspace: {ws}", fg=typer.colors.BRIGHT_BLACK)

    # `--resume` is a boolean flag (so it never swallows the next option like
    # --workspace). The positional doubles as the optional build id on resume.
    resume_id: Optional[str] = None
    if resume:
        resume_id = spec  # may be None → resume most recent for this workspace
        spec = None
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


@app.command("consolidate")
def consolidate_cmd() -> None:
    """Run a memory consolidation ('sleep') pass now: fade unused memories, merge
    duplicates, detect recurring patterns, and synthesize workflows."""
    config.ensure_dirs()
    from .memory.client import default_client

    report = default_client().consolidate()
    typer.secho(f"Consolidation: {report.summary()}", fg=typer.colors.GREEN)
    for name in report.workflows_created:
        typer.secho(f"  + workflow: {name}", fg=typer.colors.CYAN)
    for p in report.patterns[:10]:
        typer.secho(f"  · pattern: {p}", fg=typer.colors.BRIGHT_BLACK)


@app.command("dream")
def dream_cmd(
    max_memories: Optional[int] = typer.Option(
        None, "--max", help="Max memories to review this pass (default from config)."
    ),
) -> None:
    """Run an opt-in REM ('dream') pass: the model reviews recent memories as an
    adversarial critic and reversibly prunes/reweights them. Unlike `consolidate`
    this uses the model (spends Max budget) — run it when budget is comfortable."""
    config.ensure_dirs()

    typer.secho("Dreaming (REM pass) — reviewing recent memories…", fg=typer.colors.BRIGHT_BLACK)
    if config.MEMORY_URL:
        # A daemon owns the DB (sole writer) — run REM there. `--max` has no
        # wire field, so the daemon uses its config default; warn if it was set.
        if max_memories is not None:
            typer.secho(
                "  (note: --max is ignored when RELIFE_MEMORY_URL is set; "
                "the daemon uses its configured batch size)",
                fg=typer.colors.YELLOW,
            )
        from .memory.client import default_client

        report = anyio.run(lambda: default_client().dream())
    else:
        from .memory import rem

        report = anyio.run(lambda: rem.run_rem(batch_max=max_memories))
    typer.secho(f"REM pass: {report.summary()}", fg=typer.colors.GREEN)
    for note in report.notes:
        typer.secho(f"  · {note}", fg=typer.colors.BRIGHT_BLACK)


memory_app = typer.Typer(help="Inspect long-term memory.")
app.add_typer(memory_app, name="memory")


@memory_app.command("stats")
def memory_stats() -> None:
    """Show memory counts, activation, and what has faded."""
    config.ensure_dirs()
    from .memory import events
    from .memory.client import default_client

    client = default_client()
    mems = client.all_memories(include_archived=True)
    active = [m for m in mems if m.status == "active"]
    archived = [m for m in mems if m.status != "active"]
    by_kind: dict[str, int] = {}
    for m in active:
        by_kind[m.kind] = by_kind.get(m.kind, 0) + 1

    typer.secho("Long-term memory", fg=typer.colors.BRIGHT_WHITE, bold=True)
    typer.echo(f"  active: {len(active)}   archived (faded): {len(archived)}")
    for kind, n in sorted(by_kind.items()):
        typer.echo(f"    {kind}: {n}")
    # skills/workflows follow the client (daemon-side when RELIFE_MEMORY_URL is
    # set); the event log is still local, so its count reflects this process.
    typer.echo(f"  skills: {client.skill_count()}   workflows: {client.workflow_count()}   events: {events.count()}")

    top = sorted(active, key=lambda m: m.activation(), reverse=True)[:5]
    if top:
        typer.secho("  strongest right now:", fg=typer.colors.BRIGHT_BLACK)
        for m in top:
            snippet = m.text if len(m.text) <= 60 else m.text[:57] + "..."
            typer.echo(f"    [{m.activation():.2f}] {snippet}")


@memory_app.command("serve")
def memory_serve(
    host: Optional[str] = typer.Option(None, "--host", help="Bind address (default 127.0.0.1)."),
    port: Optional[int] = typer.Option(None, "--port", help="Bind port (default 8787)."),
) -> None:
    """Run the long-lived memory daemon. Point clients at it with
    RELIFE_MEMORY_URL=http://<host>:<port>. Keeps the embedding model + DB warm
    across invocations. Requires the optional extra: pip install -e ".[daemon]"."""
    config.ensure_dirs()
    try:
        from .memory.remote import daemon
    except ImportError as e:  # noqa: BLE001
        typer.secho(
            f'Daemon deps missing ({e}). Install with: pip install -e ".[daemon]"',
            fg=typer.colors.RED,
        )
        raise typer.Exit(1)

    h = host or config.MEMORY_HOST
    p = port or config.MEMORY_PORT
    typer.secho(f"Memory daemon on http://{h}:{p}  (Ctrl-C to stop)", fg=typer.colors.GREEN)
    daemon.serve(config.MEMORY_DB_PATH, host=h, port=p, token=config.MEMORY_TOKEN)


@memory_app.command("ping")
def memory_ping() -> None:
    """Check that the memory daemon is reachable (GET /health)."""
    import httpx

    base = (config.MEMORY_URL or f"http://{config.MEMORY_HOST}:{config.MEMORY_PORT}").rstrip("/")
    headers = {"Authorization": f"Bearer {config.MEMORY_TOKEN}"} if config.MEMORY_TOKEN else {}
    try:
        r = httpx.get(f"{base}/health", headers=headers, timeout=5.0)
        r.raise_for_status()
    except Exception as e:  # noqa: BLE001
        typer.secho(f"unreachable at {base}: {e}", fg=typer.colors.RED)
        raise typer.Exit(1)
    typer.secho(f"ok — {base} {r.json()}", fg=typer.colors.GREEN)


if __name__ == "__main__":
    app()
