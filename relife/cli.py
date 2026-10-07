"""ReLife command-line interface.

    relife do "<task>"   [--workspace PATH]
    relife chat          [--workspace PATH]
    relife work          [REF] [--repo R] [--dry-run]
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
    # Locals in a traceback can hold a token, a memory's text, an email body.
    pretty_exceptions_show_locals=False,
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


@app.command("work")
def work(
    ref: Optional[str] = typer.Argument(
        None,
        help="Issue to work on: owner/repo#12, an issue URL, or 12 with --repo. "
        "Omit to list your open assigned issues.",
    ),
    repo: Optional[str] = typer.Option(None, "--repo", "-R", help="owner/repo (filter or default)."),
    workspace: Optional[Path] = typer.Option(
        None, "--workspace", "-w", help="Directory repos are cloned into."
    ),
    limit: int = typer.Option(20, "--limit", "-n", help="Max issues to list."),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Fetch and clone, print the agent's task, but don't run it."
    ),
) -> None:
    """Work a GitHub issue end-to-end: branch → fix → test → push → PR (asks first)."""
    from . import workitems as wi

    try:
        if ref is None:
            items = wi.list_assigned(repo=repo, limit=limit)
            if not items:
                typer.echo("No open issues assigned to you" + (f" in {repo}." if repo else "."))
                return
            for it in items:
                labels = f"  [{', '.join(it.labels)}]" if it.labels else ""
                typer.echo(f"{it.ref:<32} {it.title[:70]}{labels}")
            typer.secho(f"\nrelife work {items[0].ref}   — to work one", fg=typer.colors.BRIGHT_BLACK)
            return

        repo_name, number = wi.parse_ref(ref, repo)
        item = wi.fetch(repo_name, number)
        if item.state != "OPEN":
            typer.secho(f"{item.ref} is {item.state.lower()} — nothing to do.", fg=typer.colors.YELLOW)
            raise typer.Exit(code=1)
        ws = _resolve_workspace(workspace)
        checkout, cloned = wi.ensure_checkout(ws, repo_name)
    except wi.WorkItemError as e:
        typer.secho(f"error: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from e

    branch = wi.branch_name(item)
    prompt = wi.task_prompt(item, branch)
    typer.secho(
        f"{item.ref}: {item.title}\ncheckout: {checkout}" + (" (cloned)" if cloned else "")
        + f"\nbranch:   {branch}",
        fg=typer.colors.BRIGHT_BLACK,
    )
    if dry_run:
        typer.echo("\n" + prompt)
        return
    # The checkout *is* the workspace: auto-allowed writes stay inside this repo.
    can_use_tool = make_permission_callback(checkout)
    mcp_servers = config.default_mcp_servers()
    hooks = memory_hooks()
    anyio.run(
        lambda: run_task(
            prompt, cwd=checkout, can_use_tool=can_use_tool, mcp_servers=mcp_servers, hooks=hooks
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


@app.command("doctor")
def doctor_cmd(
    as_json: bool = typer.Option(False, "--json", help="Machine-readable output (one JSON object)."),
) -> None:
    """Check the environment: Claude CLI + login, Node, gh, SQLite FTS5, optional
    extras, the memory daemon, the agent server + schedules (do they fire? did the
    last runs need you?), and the claude.ai connectors (Gmail etc.). Says what to
    fix. Exits 1 if anything would stop a run."""
    import json as _json

    from .doctor import default_probes, run_checks, to_json, worst

    glyph = {"ok": ("✓", typer.colors.GREEN), "warn": ("!", typer.colors.YELLOW),
             "fail": ("✗", typer.colors.RED), "skip": ("·", typer.colors.BRIGHT_BLACK)}
    checks = run_checks(default_probes())
    if as_json:
        typer.echo(_json.dumps(to_json(checks), indent=2))
        raise typer.Exit(1 if worst(checks) == "fail" else 0)
    width = max(len(c.name) for c in checks)
    for c in checks:
        mark, color = glyph[c.status]
        typer.secho(f" {mark} ", fg=color, nl=False)
        typer.secho(f"{c.name.ljust(width)}  ", bold=True, nl=False)
        typer.echo(c.detail)
        if c.fix and c.status in {"warn", "fail"}:
            typer.secho(f"   {' ' * width}  → {c.fix}", fg=typer.colors.BRIGHT_BLACK)
    overall = worst(checks)
    if overall == "fail":
        typer.secho("\nSomething above will stop a run.", fg=typer.colors.RED)
        raise typer.Exit(1)
    if overall == "warn":
        typer.secho("\nRuns will work; the warnings limit what the agent can do.", fg=typer.colors.YELLOW)
    else:
        typer.secho("\nAll good.", fg=typer.colors.GREEN)


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


@app.command("serve")
def serve(
    host: Optional[str] = typer.Option(None, "--host", help="Bind address (default 127.0.0.1)."),
    port: Optional[int] = typer.Option(None, "--port", help="Bind port (default 8600)."),
) -> None:
    """Run the always-on agent server + web UI.

    Opens a long-lived process hosting persistent agent sessions and serves the
    web console. Outward actions are routed to the browser for approval. Requires
    the optional extra: pip install -e ".[server]"."""
    config.ensure_dirs()
    try:
        from .server import app as server_app
    except ImportError as e:  # noqa: BLE001
        typer.secho(
            f'Server deps missing ({e}). Install with: pip install -e ".[server]"',
            fg=typer.colors.RED,
        )
        raise typer.Exit(1)

    h = host or config.AGENT_HOST
    p = port or config.AGENT_PORT
    from .server.security import guard_bind

    try:
        guard_bind(h, config.AGENT_TOKEN)
    except ValueError as e:
        typer.secho(str(e), fg=typer.colors.RED)
        raise typer.Exit(2)

    typer.secho(f"ReLife agent console on http://{h}:{p}  (Ctrl-C to stop)", fg=typer.colors.GREEN)
    if config.AGENT_TOKEN:
        typer.secho(
            "auth: on — the console will ask for RELIFE_AGENT_TOKEN once, then use a cookie.",
            fg=typer.colors.BRIGHT_BLACK,
        )
    typer.secho(
        f"workspaces confined to {config.AGENT_WORKSPACE_ROOT}", fg=typer.colors.BRIGHT_BLACK
    )
    server_app.serve(host=h, port=p, token=config.AGENT_TOKEN)


memory_app = typer.Typer(help="Inspect and correct long-term memory.")
app.add_typer(memory_app, name="memory")


def _fmt_age(ts: float) -> str:
    import time

    secs = max(0.0, time.time() - ts)
    for unit, size in (("d", 86400), ("h", 3600), ("m", 60)):
        if secs >= size:
            return f"{int(secs // size)}{unit} ago"
    return "just now"


def _print_memory_line(m, *, width: int = 72) -> None:
    """One memory as a table row: id, kind, activation, importance, text."""
    text = " ".join(m.text.split())
    if len(text) > width:
        text = text[: width - 1] + "…"
    status = "" if m.status == "active" else f" [{m.status}]"
    typer.secho(f"  #{m.id:<5}", fg=typer.colors.BRIGHT_BLACK, nl=False)
    typer.secho(f"{m.kind:<10}", fg=typer.colors.CYAN, nl=False)
    typer.secho(f"act {m.activation():4.2f}  imp {m.importance:3.1f}  ", fg=typer.colors.BRIGHT_BLACK, nl=False)
    typer.echo(f"{text}{status}")


@memory_app.command("search")
def memory_search(
    query: str = typer.Argument(..., help="What to look for (same ranking the agent gets)."),
    k: int = typer.Option(10, "-k", help="Max results."),
    archived: bool = typer.Option(False, "--archived", help="Include faded (archived) memories."),
) -> None:
    """Search memory the way the recall hook does — but WITHOUT reinforcing the
    hits, so looking never changes what the agent will be shown."""
    config.ensure_dirs()
    from .memory.client import default_client

    hits = default_client().recall(query, k=k, reinforce=False, include_archived=archived)
    if not hits:
        typer.secho("no matching memories", fg=typer.colors.BRIGHT_BLACK)
        return
    for m in hits:
        _print_memory_line(m)
    typer.secho(f"\n{len(hits)} hit(s) · `relife memory show <id>` for the full record", fg=typer.colors.BRIGHT_BLACK)


@memory_app.command("list")
def memory_list(
    kind: Optional[str] = typer.Option(None, "--kind", help="fact | preference | episode | pattern"),
    archived: bool = typer.Option(False, "--archived", help="Show only faded (archived) memories."),
    n: int = typer.Option(30, "-n", help="How many to show."),
    sort: str = typer.Option("recent", "--sort", help="recent | strong | oldest"),
) -> None:
    """List what the agent has learned (newest first by default)."""
    config.ensure_dirs()
    from .memory.client import default_client

    mems = default_client().all_memories(include_archived=True)
    mems = [m for m in mems if (m.status != "active") == archived]
    if kind:
        mems = [m for m in mems if m.kind == kind]
    keys = {
        "recent": lambda m: -m.created_at,
        "oldest": lambda m: m.created_at,
        "strong": lambda m: -m.activation(),
    }
    if sort not in keys:
        typer.secho("--sort must be recent | strong | oldest", fg=typer.colors.RED)
        raise typer.Exit(2)
    mems.sort(key=keys[sort])
    if not mems:
        typer.secho("nothing here", fg=typer.colors.BRIGHT_BLACK)
        return
    for m in mems[:n]:
        _print_memory_line(m)
    if len(mems) > n:
        typer.secho(f"\n… {len(mems) - n} more (raise -n)", fg=typer.colors.BRIGHT_BLACK)


@memory_app.command("show")
def memory_show(mem_id: int = typer.Argument(..., help="Memory id (from search/list).")) -> None:
    """Show one memory in full: text, tags, importance, activation, use history."""
    config.ensure_dirs()
    import datetime as dt

    from .memory.client import default_client

    m = default_client().get(mem_id)
    if m is None:
        typer.secho(f"no memory #{mem_id}", fg=typer.colors.RED)
        raise typer.Exit(1)
    when = lambda ts: dt.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M") + f"  ({_fmt_age(ts)})"  # noqa: E731
    typer.secho(f"#{m.id}  {m.kind}  {m.status}", bold=True)
    typer.echo("")
    typer.echo("  " + m.text)
    typer.echo("")
    if m.tags:
        typer.echo(f"  tags        {m.tags}")
    typer.echo(f"  importance  {m.importance:.2f}")
    typer.echo(f"  activation  {m.activation():.2f}   (rises with use, fades when idle)")
    typer.echo(f"  used        {m.use_count}×, last {when(m.last_used_at or m.created_at)}")
    typer.echo(f"  created     {when(m.created_at)}")


@memory_app.command("forget")
def memory_forget(
    ids: Optional[list[int]] = typer.Argument(None, help="Memory id(s) to archive."),
    query: Optional[str] = typer.Option(None, "--query", "-q", help="Archive the single best match for this text instead."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Don't ask for confirmation."),
) -> None:
    """Archive memories so the agent stops being shown them. Reversible: archived
    memories keep their row (see `list --archived`) until the slow hard-delete tier."""
    config.ensure_dirs()
    from .memory.client import default_client

    client = default_client()
    if bool(ids) == bool(query):
        typer.secho("give memory id(s), or --query TEXT — not both, not neither", fg=typer.colors.RED)
        raise typer.Exit(2)

    if query:
        hits = client.recall(query, k=1, reinforce=False)
        if not hits:
            typer.secho("no matching memory", fg=typer.colors.BRIGHT_BLACK)
            raise typer.Exit(1)
        targets = hits
    else:
        targets, missing = [], []
        for mid in ids or []:
            m = client.get(mid)
            (targets if m else missing).append(m if m else mid)
        for mid in missing:
            typer.secho(f"no memory #{mid}", fg=typer.colors.YELLOW)
        if not targets:
            raise typer.Exit(1)

    for m in targets:
        _print_memory_line(m)
    if not yes and not typer.confirm(f"archive {len(targets)} memor{'y' if len(targets) == 1 else 'ies'}?", default=False):
        typer.secho("left alone", fg=typer.colors.BRIGHT_BLACK)
        raise typer.Exit(0)
    for m in targets:
        client.archive(m.id)
    typer.secho(f"archived {len(targets)}", fg=typer.colors.GREEN)


@memory_app.command("stats")
def memory_stats() -> None:
    """Show memory counts, activation, and what has faded."""
    config.ensure_dirs()
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
    # skills/workflows/events all follow the client (daemon-side when
    # RELIFE_MEMORY_URL is set, in-process otherwise).
    typer.echo(f"  skills: {client.skill_count()}   workflows: {client.workflow_count()}   events: {client.event_count()}")

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


def _friendly_error(e: BaseException) -> str | None:
    """A one-line explanation for failures that are the environment's, not a bug."""
    import sqlite3

    if isinstance(e, sqlite3.DatabaseError):
        return (
            f"memory database is unreadable ({e}): {config.MEMORY_DB_PATH}\n"
            "  move it aside (or restore a backup) and ReLife will start a fresh one"
        )
    if type(e).__name__ in {"ConnectError", "ConnectTimeout"} and config.MEMORY_URL:
        return (
            f"memory daemon not reachable at {config.MEMORY_URL} ({e})\n"
            "  start it with `relife memory serve`, or unset RELIFE_MEMORY_URL to use in-process memory"
        )
    return None


def main() -> None:
    """Console entry point: ``app()`` with environment failures made readable."""
    try:
        app()
    except Exception as e:  # noqa: BLE001
        msg = _friendly_error(e)
        if msg is None:
            raise
        typer.secho(f"error: {msg}", fg=typer.colors.RED, err=True)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
