"""ReLife command-line interface.

    relife do "<task>"   [--workspace PATH] [--agent NAME]
    relife chat          [--workspace PATH] [--agent NAME]
    relife work          [REF] [--repo R] [--dry-run]
    relife agent …       register agents and hand memory between them
    relife crew "<task>" CrewAI plans a team of agents and runs it ([crewai] extra)
    relife mcp --agent N serve ReLife memory to any MCP client (stdio)
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


_AGENT_OPTION_HELP = (
    "Run as this registered agent: it remembers into its own memory space and "
    "reads what it inherited (see `relife agent`). Default: the main agent."
)


def _agent_memory(agent: Optional[str]):
    """The scoped memory client for ``--agent NAME`` (``None`` = the main agent)."""
    if agent is None:
        return None
    from .agents import AgentStore
    from .memory.client import ScopedMemoryClient, default_client

    try:
        scope = AgentStore().require(agent).scope()
    except LookupError as e:
        typer.secho(f"error: {e}", fg=typer.colors.RED)
        raise typer.Exit(2) from e
    typer.secho(
        f"agent: {agent} (writes {scope.write}; reads {', '.join(scope.read)})",
        fg=typer.colors.BRIGHT_BLACK,
    )
    return ScopedMemoryClient(default_client(), scope)


@app.command("do")
def do(
    task: str = typer.Argument(..., help="What you want done, in plain language."),
    workspace: Optional[Path] = typer.Option(
        None, "--workspace", "-w", help="Directory the agent works in."
    ),
    agent: Optional[str] = typer.Option(None, "--agent", "-a", help=_AGENT_OPTION_HELP),
) -> None:
    """Run a single task to completion."""
    ws = _resolve_workspace(workspace)
    typer.secho(f"workspace: {ws}", fg=typer.colors.BRIGHT_BLACK)
    memory = _agent_memory(agent)
    can_use_tool = make_permission_callback(ws)
    mcp_servers = config.default_mcp_servers(memory)
    hooks = memory_hooks(memory)
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
    agent: Optional[str] = typer.Option(None, "--agent", "-a", help=_AGENT_OPTION_HELP),
) -> None:
    """Start an interactive multi-turn session."""
    ws = _resolve_workspace(workspace)
    typer.secho(f"workspace: {ws}", fg=typer.colors.BRIGHT_BLACK)
    memory = _agent_memory(agent)
    can_use_tool = make_permission_callback(ws)
    mcp_servers = config.default_mcp_servers(memory)
    hooks = memory_hooks(memory)
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
    """One memory as a table row: id, kind, activation, importance, text — plus
    the space it lives in when that isn't the main agent's."""
    text = " ".join(m.text.split())
    if len(text) > width:
        text = text[: width - 1] + "…"
    status = "" if m.status == "active" else f" [{m.status}]"
    space = getattr(m, "space", "default")
    typer.secho(f"  #{m.id:<5}", fg=typer.colors.BRIGHT_BLACK, nl=False)
    if space != "default":
        typer.secho(f"{space}/", fg=typer.colors.MAGENTA, nl=False)
    typer.secho(f"{m.kind:<10}", fg=typer.colors.CYAN, nl=False)
    typer.secho(f"act {m.activation():4.2f}  imp {m.importance:3.1f}  ", fg=typer.colors.BRIGHT_BLACK, nl=False)
    typer.echo(f"{text}{status}")


@memory_app.command("search")
def memory_search(
    query: str = typer.Argument(..., help="What to look for (same ranking the agent gets)."),
    k: int = typer.Option(10, "-k", help="Max results."),
    archived: bool = typer.Option(False, "--archived", help="Include faded (archived) memories."),
    space: Optional[list[str]] = typer.Option(
        None, "--space", "-s", help="Search these memory spaces (repeatable). Default: the main agent's."
    ),
) -> None:
    """Search memory the way the recall hook does — but WITHOUT reinforcing the
    hits, so looking never changes what the agent will be shown."""
    config.ensure_dirs()
    from .memory.client import default_client

    hits = default_client().recall(
        query, k=k, reinforce=False, include_archived=archived, spaces=space or None
    )
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
    space: Optional[list[str]] = typer.Option(
        None, "--space", "-s", help="Only these memory spaces (repeatable). Default: every space."
    ),
) -> None:
    """List what the agents have learned (newest first by default)."""
    config.ensure_dirs()
    from .memory.client import default_client

    mems = default_client().all_memories(include_archived=True, spaces=space or None)
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
    typer.echo(f"  space       {m.space}")
    if m.source:
        typer.echo(f"  written by  {m.source}")
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
    space: str = typer.Option("default", "--space", "-s", help="With --query: the memory space to search."),
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
        hits = client.recall(query, k=1, reinforce=False, spaces=[space])
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


@memory_app.command("spaces")
def memory_spaces() -> None:
    """List memory spaces: the main agent's (default) and one per agent."""
    config.ensure_dirs()
    from .agents import AgentStore
    from .memory.client import default_client

    owners: dict[str, list[str]] = {}
    readers: dict[str, list[str]] = {}
    for a in AgentStore().list():
        owners.setdefault(a.own_space, []).append(a.name)
        for s_ in a.inherits:
            readers.setdefault(s_, []).append(a.name)
    for name, c in default_client().spaces().items():
        who = "main agent" if name == "default" else ", ".join(owners.get(name, [])) or "no agent"
        typer.secho(f"  {name:<20}", bold=True, nl=False)
        typer.echo(
            f"{c['memories']:>5} memories  {c['skills']:>3} skills  {c['workflows']:>3} workflows"
            f"   · {who}"
            + (f"; read by {', '.join(readers[name])}" if name in readers else "")
        )


@memory_app.command("export")
def memory_export(
    space: str = typer.Argument("default", help="The memory space to export."),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="File to write (default: stdout)."),
) -> None:
    """Export a space's active memories, skills and workflows as a portable
    JSON pack — to move knowledge to another install, or another agent."""
    import json as _json

    config.ensure_dirs()
    from .memory.client import default_client

    try:
        pack = default_client().export_space(space)
    except ValueError as e:
        typer.secho(f"error: {e}", fg=typer.colors.RED)
        raise typer.Exit(2) from e
    text = _json.dumps(pack, indent=2, ensure_ascii=False)
    if output is None:
        typer.echo(text)
        return
    output.write_text(text, encoding="utf-8")
    typer.secho(
        f"exported {space}: {len(pack['memories'])} memories, {len(pack['skills'])} skills, "
        f"{len(pack['workflows'])} workflows → {output}",
        fg=typer.colors.GREEN,
    )


@memory_app.command("import")
def memory_import(
    pack_file: Path = typer.Argument(..., help="A pack written by `relife memory export`."),
    space: str = typer.Option(..., "--space", "-s", help="The memory space to load it into."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Don't ask for confirmation."),
) -> None:
    """Load an exported pack into a space. A memory the space already holds is
    reinforced, not duplicated; existing skills/workflows are kept. Imported
    memories are marked `import:<origin>` so their provenance stays visible."""
    import json as _json

    config.ensure_dirs()
    from .memory.client import default_client
    from .memory.service import validate_pack

    try:
        pack = validate_pack(_json.loads(pack_file.read_text(encoding="utf-8")))
    except (OSError, ValueError) as e:
        typer.secho(f"error: {e}", fg=typer.colors.RED)
        raise typer.Exit(2) from e
    typer.echo(
        f"{pack_file.name}: {len(pack.get('memories', []))} memories, "
        f"{len(pack.get('skills', []))} skills, {len(pack.get('workflows', []))} workflows "
        f"from space {pack.get('space')!r} → into {space!r}"
    )
    if space == "default":
        typer.secho(
            "  the default space is what your main agent trusts — import only packs you trust",
            fg=typer.colors.YELLOW,
        )
    if not yes and not typer.confirm("import?", default=False):
        typer.secho("left alone", fg=typer.colors.BRIGHT_BLACK)
        raise typer.Exit(0)
    try:
        out = default_client().import_pack(pack, space)
    except ValueError as e:
        typer.secho(f"error: {e}", fg=typer.colors.RED)
        raise typer.Exit(2) from e
    typer.secho(
        f"imported {out['memories']} memories, {out['skills']} skills, {out['workflows']} workflows",
        fg=typer.colors.GREEN,
    )


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


agent_app = typer.Typer(
    help="Register agents and hand memory between them (each agent has its own memory space)."
)
app.add_typer(agent_app, name="agent")


def _agents():
    config.ensure_dirs()
    from .agents import AgentStore
    from .memory.client import default_client

    return AgentStore(), default_client()


def _agent_or_exit(store, name: str):
    try:
        return store.require(name)
    except LookupError as e:
        typer.secho(f"error: {e}", fg=typer.colors.RED)
        raise typer.Exit(2) from e


def _print_connect(name: str, token: str | None = None) -> None:
    """Paste-ready MCP config for attaching ReLife memory as this agent."""
    import json as _json
    import sys

    from .agents import mcp_config

    http_url = f"http://{config.MEMORY_HOST}:{config.MEMORY_PORT}/mcp"
    cfg = mcp_config(
        name, python=sys.executable, home=str(config.PROJECT_ROOT), http_url=http_url, token=token
    )
    typer.secho("\nConnect any MCP client (stdio — runs on this machine):", bold=True)
    typer.echo(_json.dumps(cfg["stdio"], indent=2))
    if "http" in cfg:
        typer.secho(
            "\nOr over HTTP (needs `relife memory serve` running; the token is shown only now):",
            bold=True,
        )
        typer.echo(_json.dumps(cfg["http"], indent=2))


@agent_app.command("list")
def agent_list() -> None:
    """Every registered agent, its runtime, and the memory it reads and writes."""
    store, client = _agents()
    agents = store.list()
    if store.problem:
        typer.secho(f"agents.json: {store.problem}", fg=typer.colors.YELLOW)
    if not agents:
        typer.secho(
            "no agents yet — the main agent uses the default space.\n"
            "  relife agent create NAME [--inherit OTHER] [--fork OTHER]",
            fg=typer.colors.BRIGHT_BLACK,
        )
        return
    counts = client.spaces()
    for a in agents:
        n = counts.get(a.own_space, {}).get("memories", 0)
        typer.secho(f"  {a.name:<20}", bold=True, nl=False)
        typer.secho(f"{a.runtime:<9}", fg=typer.colors.CYAN, nl=False)
        reads = ", ".join(a.scope().read[1:]) or "nothing else"
        typer.echo(
            f"{n:>4} memories in {a.own_space}  · reads {reads}"
            + (f"  · {a.model}" if a.model else "")
            + ("  · token" if a.token_hash else "")
        )


@agent_app.command("create")
def agent_create(
    name: str = typer.Argument(..., help="Agent name (lowercase, digits, - or _)."),
    runtime: str = typer.Option(
        "relife", "--runtime", "-r",
        help="relife = a full ReLife (Claude) agent · llm = a CrewAI agent on another model · "
        "external = any MCP client that attaches ReLife memory.",
    ),
    model: str = typer.Option("", "--model", "-m", help="Model for an llm agent, e.g. ollama/llama3.1, gpt-4.1."),
    description: str = typer.Option("", "--description", "-d", help="What this agent is for."),
    inherit: Optional[list[str]] = typer.Option(
        None, "--inherit", "-i", help="Read this agent's memory live, read-only (repeatable)."
    ),
    fork: Optional[str] = typer.Option(None, "--fork", help="Start from a snapshot copy of this agent's memory."),
    space: Optional[str] = typer.Option(
        None, "--space", help="Write to this space instead of one named after the agent (agents given "
        "the same space share one memory)."
    ),
    isolated: bool = typer.Option(
        False, "--isolated", help="Don't let it read your main (default) memory — e.g. an external "
        "agent whose model provider shouldn't see it."
    ),
) -> None:
    """Register an agent and hand it memory from older agents."""
    from .agents import create_agent

    store, client = _agents()
    try:
        profile, copied = create_agent(
            store, client, name,
            runtime=runtime, model=model, description=description,
            inherit=list(inherit or []), fork=fork, space=space, isolated=isolated,
        )
    except (ValueError, LookupError) as e:
        typer.secho(f"error: {e}", fg=typer.colors.RED)
        raise typer.Exit(2) from e
    scope = profile.scope()
    typer.secho(f"created agent {profile.name} ({profile.runtime})", fg=typer.colors.GREEN)
    typer.echo(f"  writes: {scope.write}")
    typer.echo(f"  reads:  {', '.join(scope.read)}")
    if copied:
        typer.echo(
            f"  forked from {fork}: {copied['memories']} memories, {copied['skills']} skills, "
            f"{copied['workflows']} workflows"
        )
    if profile.runtime == "external":
        _print_connect(profile.name)
        typer.secho(
            f"\n  for HTTP access: relife agent token {profile.name}", fg=typer.colors.BRIGHT_BLACK
        )


@agent_app.command("show")
def agent_show(name: str = typer.Argument(..., help="Agent name.")) -> None:
    """One agent in full: what it reads and writes, and how much each space holds."""
    store, client = _agents()
    a = _agent_or_exit(store, name)
    counts = client.spaces()
    scope = a.scope()
    typer.secho(f"{a.name}  ({a.runtime}{', ' + a.model if a.model else ''})", bold=True)
    if a.description:
        typer.echo(f"  {a.description}")
    if a.parent:
        typer.echo(f"  parent: {a.parent}")
    for sp in scope.read:
        c = counts.get(sp, {"memories": 0, "skills": 0, "workflows": 0})
        role = "writes + reads" if sp == scope.write else "reads"
        typer.echo(
            f"  {role:<15}{sp:<20}{c['memories']:>5} memories {c['skills']:>3} skills "
            f"{c['workflows']:>3} workflows"
        )
    typer.echo(f"  http token: {'yes' if a.token_hash else 'none'}")


@agent_app.command("attach")
def agent_attach(
    name: str = typer.Argument(..., help="Agent that should see more."),
    other: str = typer.Argument(..., help="Agent (or space) whose memory it may read."),
) -> None:
    """Let an agent read another agent's memory, live and read-only."""
    from .agents import attach

    store, _ = _agents()
    try:
        a = attach(store, name, other)
    except (ValueError, LookupError) as e:
        typer.secho(f"error: {e}", fg=typer.colors.RED)
        raise typer.Exit(2) from e
    typer.secho(f"{a.name} now reads: {', '.join(a.scope().read)}", fg=typer.colors.GREEN)


@agent_app.command("detach")
def agent_detach(
    name: str = typer.Argument(..., help="Agent name."),
    other: str = typer.Argument(..., help="Agent (or space) to stop reading."),
) -> None:
    """Stop an agent reading another agent's memory."""
    from .agents import detach

    store, _ = _agents()
    try:
        a = detach(store, name, other)
    except (ValueError, LookupError) as e:
        typer.secho(f"error: {e}", fg=typer.colors.RED)
        raise typer.Exit(2) from e
    typer.secho(f"{a.name} now reads: {', '.join(a.scope().read)}", fg=typer.colors.GREEN)


@agent_app.command("promote")
def agent_promote(
    name: str = typer.Argument(..., help="Agent whose learnings to promote."),
    to: str = typer.Option("default", "--to", help="Destination space (default: your main memory)."),
    ids: Optional[list[int]] = typer.Option(None, "--id", help="Only these memory ids (repeatable)."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Don't ask for confirmation."),
) -> None:
    """Copy what an agent learned into shared memory — the explicit act of
    trusting it. Without --id, everything active in the agent's space (plus its
    skills and workflows) is promoted."""
    from .agents import promote

    store, client = _agents()
    a = _agent_or_exit(store, name)
    mems = client.all_memories(include_archived=False, spaces=[a.own_space])
    if ids:
        wanted = set(ids)
        mems = [m for m in mems if m.id in wanted]
    if not mems and ids:
        typer.secho(f"none of those ids are active memories of {name}", fg=typer.colors.YELLOW)
        raise typer.Exit(1)
    for m in mems[:20]:
        _print_memory_line(m)
    if len(mems) > 20:
        typer.secho(f"  … and {len(mems) - 20} more", fg=typer.colors.BRIGHT_BLACK)
    extra = "" if ids else " (plus its skills and workflows)"
    if not yes and not typer.confirm(
        f"promote {len(mems)} memor{'y' if len(mems) == 1 else 'ies'}{extra} from {a.own_space} into {to}?",
        default=False,
    ):
        typer.secho("left alone", fg=typer.colors.BRIGHT_BLACK)
        raise typer.Exit(0)
    try:
        out = promote(store, client, name, to=to, ids=list(ids) if ids else None)
    except ValueError as e:
        typer.secho(f"error: {e}", fg=typer.colors.RED)
        raise typer.Exit(2) from e
    typer.secho(
        f"promoted {out['memories']} memories, {out['skills']} skills, {out['workflows']} workflows into {to}",
        fg=typer.colors.GREEN,
    )


@agent_app.command("token")
def agent_token(
    name: str = typer.Argument(..., help="Agent name."),
    revoke: bool = typer.Option(False, "--revoke", help="Remove the agent's token instead."),
) -> None:
    """Mint (or rotate) the agent's token for the memory daemon's HTTP MCP
    endpoint. Shown once — only its hash is stored."""
    from .agents import issue_token, revoke_token

    store, _ = _agents()
    _agent_or_exit(store, name)
    if revoke:
        revoke_token(store, name)
        typer.secho(f"{name}: token revoked", fg=typer.colors.GREEN)
        return
    token = issue_token(store, name)
    typer.secho(f"{name}: new token (any previous one stops working)", fg=typer.colors.GREEN)
    typer.echo(f"  {token}")
    _print_connect(name, token)


@agent_app.command("connect")
def agent_connect(name: str = typer.Argument(..., help="Agent name.")) -> None:
    """Print the MCP config that attaches ReLife memory to a client as this agent."""
    store, _ = _agents()
    a = _agent_or_exit(store, name)
    _print_connect(a.name)
    if a.token_hash:
        typer.secho(
            "\nHTTP: this agent has a token; it can't be shown again — "
            f"`relife agent token {a.name}` rotates it.",
            fg=typer.colors.BRIGHT_BLACK,
        )


@agent_app.command("delete")
def agent_delete(
    name: str = typer.Argument(..., help="Agent name."),
    keep_memory: bool = typer.Option(False, "--keep-memory", help="Leave its memory space active."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Don't ask for confirmation."),
) -> None:
    """Unregister an agent. Its memory space is archived (reversible) unless
    --keep-memory or another agent shares that space."""
    from .agents import delete_agent

    store, client = _agents()
    a = _agent_or_exit(store, name)
    if not yes and not typer.confirm(f"delete agent {a.name}?", default=False):
        typer.secho("left alone", fg=typer.colors.BRIGHT_BLACK)
        raise typer.Exit(0)
    archived = delete_agent(store, client, name, keep_memory=keep_memory)
    typer.secho(
        f"deleted {name}" + (f"; archived {archived} memories in {a.own_space}" if archived else ""),
        fg=typer.colors.GREEN,
    )


def _crewai_or_exit() -> None:
    """`relife crew` needs the optional [crewai] extra (CrewAI needs Python < 3.14)."""
    import importlib.util
    import sys

    if importlib.util.find_spec("crewai") is not None:
        return
    if sys.version_info >= (3, 14):
        hint = (
            f"CrewAI doesn't support Python {sys.version_info.major}.{sys.version_info.minor} yet. "
            'Run ReLife from a 3.12 venv:  py -3.12 -m venv .venv  then  '
            '.venv\\Scripts\\pip install -e ".[crewai]"'
        )
    else:
        hint = 'CrewAI is not installed:  pip install -e ".[crewai]"'
    typer.secho(hint, fg=typer.colors.RED)
    raise typer.Exit(1)


@app.command("crew")
def crew_cmd(
    task: Optional[str] = typer.Argument(None, help="What the crew should get done, in plain language."),
    spec: Optional[Path] = typer.Option(
        None, "--spec", help="Run your own crew plan (JSON or YAML) instead of having one planned."
    ),
    plan_only: bool = typer.Option(False, "--plan-only", help="Plan and show the crew, but don't run it."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Run without asking to confirm the plan."),
    workspace: Optional[Path] = typer.Option(
        None, "--workspace", "-w", help="Base directory; the crew works in <workspace>/crews/<id>/."
    ),
) -> None:
    """CrewAI plans a team for the task and ReLife staffs it: ReLife agents (Claude
    with tools and memory) and, if configured (RELIFE_CREW_LLMS), CrewAI agents on
    other models with ReLife memory attached. New agents inherit what older ones
    learned. Each member spends budget — the plan is shown before it runs."""
    _crewai_or_exit()
    from .crew.runner import run_crew
    from .crew.spec import load_spec_file

    if not task and spec is None:
        typer.secho('give a task, or --spec FILE', fg=typer.colors.RED)
        raise typer.Exit(2)
    ws = _resolve_workspace(workspace)
    try:
        raw = load_spec_file(spec) if spec is not None else None
        record = run_crew(
            task,
            workspace=ws,
            spec_raw=raw,
            plan_only=plan_only,
            yes=yes,
            echo=typer.echo,
            confirm=lambda q: typer.confirm(q, default=False),
        )
    except (ValueError, OSError) as e:
        typer.secho(f"error: {e}", fg=typer.colors.RED)
        raise typer.Exit(2) from e
    if record.status in ("planned", "cancelled"):
        return
    color = typer.colors.GREEN if record.status == "done" else typer.colors.RED
    typer.secho(
        f"\ncrew {record.id}: {record.status} · {len(record.tasks)} task(s) · "
        f"usage-equiv ${record.cost_usd:.2f}" + (f" · {record.error}" if record.error else ""),
        fg=color,
    )
    if record.final_output:
        typer.echo("\n" + record.final_output)
    denied = [d for t in record.tasks for d in t.denied]
    if denied:
        typer.secho(f"\nneeded you — {len(denied)} action(s) were denied:", fg=typer.colors.YELLOW)
        for d in denied:
            typer.echo(f"  {d.get('tool')}: {d.get('brief')}")
    typer.secho(
        "\nwhat each agent learned stays in its own memory space — "
        "`relife agent promote NAME` moves it into your main memory",
        fg=typer.colors.BRIGHT_BLACK,
    )


@app.command("crews")
def crews_cmd(
    run_id: Optional[str] = typer.Argument(None, help="Show this crew run in full."),
    limit: int = typer.Option(10, "-n", help="How many recent runs to list."),
) -> None:
    """Recent crew runs, or one run's plan and per-task outcomes."""
    config.ensure_dirs()
    from .crew.record import CrewRunStore

    store = CrewRunStore()
    if run_id is None:
        runs = store.list(limit)
        if not runs:
            typer.secho("no crew runs yet", fg=typer.colors.BRIGHT_BLACK)
            return
        for r in runs:
            typer.secho(f"  {r.id}  ", bold=True, nl=False)
            typer.echo(f"{r.status:<11} ${r.cost_usd:5.2f}  {r.task[:70]}")
        return
    r = store.get(run_id)
    if r is None:
        typer.secho(f"no crew run {run_id!r}", fg=typer.colors.RED)
        raise typer.Exit(1)
    from .crew.runner import describe_plan
    from .crew.spec import CrewSpec

    typer.secho(f"crew {r.id} — {r.status} · usage-equiv ${r.cost_usd:.2f}", bold=True)
    typer.echo(f"task: {r.task}\nworkspace: {r.workspace}")
    for line in describe_plan(CrewSpec.from_dict(r.spec)):
        typer.echo(line)
    for t in r.tasks:
        typer.secho(f"\n## {t.name} ({t.agent}, {t.runtime})", bold=True)
        if t.error:
            typer.secho(f"error: {t.error}", fg=typer.colors.RED)
        typer.echo(t.output or "(no output)")
    if r.error:
        typer.secho(f"\nerror: {r.error}", fg=typer.colors.RED)


@app.command("mcp")
def mcp_cmd(
    agent: str = typer.Option(
        ..., "--agent", "-a",
        help="The registered agent this connection acts as (its memory scope). Required: "
        "an MCP client never gets your main memory's write access by default.",
    ),
) -> None:
    """Serve ReLife memory over MCP (stdio) to any agent — Claude Desktop, Cursor,
    Gemini CLI, a CrewAI agent on another model. Run by the MCP client itself;
    see `relife agent connect NAME` for the config."""
    from .agents import AgentStore

    config.ensure_dirs()
    try:
        profile = AgentStore().require(agent)
    except LookupError as e:
        # stdout is the protocol stream — errors go to stderr only.
        typer.secho(f"error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(2) from e
    from .memory.mcp_server import run_stdio

    run_stdio(profile.scope())


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
