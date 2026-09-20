"""The agent runner: builds options and drives the Claude Agent SDK loop.

Two entry points:
- ``run_task``  — one-shot: give it a task string, it works to completion.
- ``run_chat``  — interactive multi-turn session in one workspace.

Rendering is intentionally simple (stream text, announce tool use). Permissions,
MCP servers, and memory hooks are layered in by later build stages via the
``can_use_tool``, ``mcp_servers``, and ``hooks`` parameters of ``build_options``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Awaitable, Callable

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    PermissionResultDeny,
    ResultMessage,
    SystemMessage,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)
from rich.console import Console

from . import config

# Windows consoles default to a legacy code page (cp1252) that can't encode the
# glyphs we (and the model) emit. Force UTF-8 and use ANSI rendering so output
# never crashes on an unencodable character.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass

console = Console(legacy_windows=False)

CanUseTool = Callable[[str, dict[str, Any], Any], Awaitable[Any]]


def preset_system_prompt(append_file: Path | None = None) -> dict[str, Any]:
    """Claude Code preset (keeps strong coding behavior) + an appended persona.

    ``append_file`` defaults to the standard ReLife persona; callers like the
    build orchestrator pass their own file to swap in a different persona.
    """
    path = append_file or config.SYSTEM_PROMPT_FILE
    append = path.read_text(encoding="utf-8") if path.exists() else ""
    return {"type": "preset", "preset": "claude_code", "append": append}


def _system_prompt() -> dict[str, Any]:
    """Default persona append (the standard `do`/`chat` system prompt)."""
    return preset_system_prompt()


def build_options(
    *,
    cwd: Path,
    permission_mode: str | None = None,
    can_use_tool: CanUseTool | None = None,
    mcp_servers: dict[str, Any] | None = None,
    hooks: dict[str, Any] | None = None,
    system_prompt: dict[str, Any] | None = None,
    agents: dict[str, Any] | None = None,
    resume: str | None = None,
    max_budget_usd: float | None = None,
) -> ClaudeAgentOptions:
    """Assemble ClaudeAgentOptions from config + per-run overrides.

    ``agents`` defines subagents the model can delegate to via the Task tool
    (used by ``relife build``). ``resume`` continues a prior CLI session by id.
    ``system_prompt`` overrides the default persona append (the orchestrator
    swaps in its own).
    """
    return ClaudeAgentOptions(
        model=config.MODEL,
        effort=config.EFFORT,
        system_prompt=system_prompt or _system_prompt(),
        cwd=str(cwd),
        permission_mode=permission_mode or config.DEFAULT_PERMISSION_MODE,
        can_use_tool=can_use_tool,
        mcp_servers=mcp_servers or {},
        hooks=hooks,
        agents=agents,
        resume=resume,
        max_budget_usd=max_budget_usd,
        # Ensure CLIs like `gh` are on PATH for the agent subprocess.
        env=config.agent_env(),
        # Don't inherit the surrounding repo's Claude Code settings — ReLife is
        # self-contained and defines its own behavior.
        setting_sources=None,
    )


async def _deny_all_tools(tool_name: str, tool_input: dict[str, Any], context: Any):
    """A ``can_use_tool`` that denies everything — for pure reasoning calls."""
    return PermissionResultDeny(message="Tool use is disabled for this call.")


async def ask_model_oneshot(
    system_prompt: str, prompt: str, *, cwd: Path | None = None
) -> tuple[str, float | None]:
    """One-shot text-in / text-out model judgment (no tools, no MCP, no hooks).

    Used by the REM ("dream") pass to run the model as an adversarial critic over
    memory. Returns ``(text, cost_usd)``. Tools are hard-denied so the call is
    fully non-interactive and can never touch the filesystem or take an action —
    the model is a pure advisor here. Streaming transport is required because we
    attach a ``can_use_tool`` callback (the deny-all gate).
    """
    options = build_options(
        cwd=cwd or config.PROJECT_ROOT,
        can_use_tool=_deny_all_tools,
        mcp_servers={},
        hooks=None,
        system_prompt=system_prompt,  # plain critic prompt, not the ReLife preset
    )
    chunks: list[str] = []
    cost: float | None = None
    async with ClaudeSDKClient(options=options) as client:
        await client.query(prompt)
        async for msg in client.receive_response():
            if isinstance(msg, AssistantMessage):
                for block in msg.content:
                    if isinstance(block, TextBlock):
                        chunks.append(block.text)
            elif isinstance(msg, ResultMessage):
                cost = getattr(msg, "total_cost_usd", None)
    return "".join(chunks), cost


def to_event(msg: Any) -> list[dict[str, Any]]:
    """Map one streamed SDK message to zero-or-more structured events.

    Pure and JSON-serializable — the single source of the streaming message
    taxonomy, consumed by both the terminal renderer (``_render``) and the
    web server's SSE stream. ``SystemMessage`` (init/status frames) yields
    nothing, as it's dropped in the terminal too.

    **Tool results arrive on a ``UserMessage``**, not the assistant one: the CLI
    reports each result as a ``user``-type frame carrying ``ToolResultBlock``s
    (see the SDK's ``message_parser``). Walking only ``AssistantMessage`` meant
    the UI streamed every tool *call* and never a single result. Text on a
    ``UserMessage`` is the caller's own turn echoed back, so it's dropped —
    the server already publishes that itself when the turn is submitted.
    """
    events: list[dict[str, Any]] = []
    if isinstance(msg, UserMessage):
        content = msg.content
        if isinstance(content, list):
            for block in content:
                if isinstance(block, ToolResultBlock):
                    events.append({"type": "tool_result", "brief": _tool_result_brief(block)})
    elif isinstance(msg, AssistantMessage):
        for block in msg.content:
            if isinstance(block, TextBlock):
                if block.text.strip():
                    events.append({"type": "text", "text": block.text})
            elif isinstance(block, ThinkingBlock):
                events.append({"type": "thinking"})
            elif isinstance(block, ToolUseBlock):
                events.append(
                    {"type": "tool_use", "name": block.name, "brief": _tool_brief(block.input)}
                )
            elif isinstance(block, ToolResultBlock):
                events.append({"type": "tool_result", "brief": _tool_result_brief(block)})
    elif isinstance(msg, ResultMessage):
        events.append({"type": "result", "cost_usd": getattr(msg, "total_cost_usd", None)})
    return events


def _render(msg: Any) -> None:
    """Pretty-print a streamed SDK message (via the shared ``to_event`` taxonomy)."""
    for ev in to_event(msg):
        kind = ev["type"]
        if kind == "text":
            console.print(ev["text"])
        elif kind == "thinking":
            console.print("[dim italic]…thinking[/]")
        elif kind == "tool_use":
            console.print(f"[cyan]→ {ev['name']}[/] [dim]{ev['brief']}[/]")
        elif kind == "tool_result":
            # Tool results were historically not printed in the terminal; keep it
            # quiet there to avoid noise, but they still stream to the UI.
            pass
        elif kind == "result":
            cost = ev.get("cost_usd")
            note = f"  [dim](usage-equiv ${cost:.4f})[/]" if cost else ""
            console.print(f"[green]✓ done[/]{note}")


def _maybe_consolidate() -> None:
    """Run a background-style consolidation pass if enough has accrued.

    Brain-like upkeep after a run: fade unused memories, merge duplicates, and
    learn workflows from recurring tool sequences. Deterministic and cheap (no
    LLM); throttled by event volume and fully fail-safe so it never disrupts a
    completed task.
    """
    try:
        from .memory.client import default_client

        # The throttle + run are decided together on the memory side (server-side
        # under a daemon), so the "enough events accrued?" gate reads the same
        # event log the consolidation runs against. Gating here in the agent
        # process would read this process's local event log while the work runs
        # on the daemon — so it would never fire against a remote daemon.
        report = default_client().maybe_consolidate()
        if report and (
            report.archived or report.deleted or report.merged or report.workflows_created
        ):
            console.print(f"[dim]· memory consolidated: {report.summary()}[/]")
    except Exception:
        pass


def _tool_brief(inp: dict[str, Any], limit: int = 80) -> str:
    """One-line hint of what a tool call is doing.

    Built-in tools have one obvious key; a connector call (send an email, create
    an event) has none, so fall back to the leading scalar fields — the approval
    card must show *what* is about to leave the machine, not a blank line.
    """
    if not isinstance(inp, dict):
        return ""
    for key in ("command", "file_path", "path", "pattern", "url", "query"):
        if key in inp:
            return _clip(str(inp[key]), limit)
    pairs = []
    for key, val in inp.items():
        if isinstance(val, (str, int, float, bool)) and str(val).strip():
            pairs.append(f"{key}={_clip(' '.join(str(val).split()), 60)}")
        if len(pairs) == 4:
            break
    return _clip("  ".join(pairs), limit)


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _tool_result_brief(block: Any) -> str:
    """One-line summary of a tool result for the UI stream.

    Tool result content is either a string or a list of content parts; collapse
    it to a short single line and mark errors.
    """
    content = getattr(block, "content", None)
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                parts.append(str(part.get("text", "")))
            else:
                parts.append(str(getattr(part, "text", part)))
        text = " ".join(p for p in parts if p)
    else:
        text = str(content or "")
    text = " ".join(text.split())  # collapse whitespace/newlines
    if len(text) > 80:
        text = text[:77] + "..."
    prefix = "error: " if getattr(block, "is_error", False) else ""
    return f"{prefix}{text}"


async def run_task(
    prompt: str,
    *,
    cwd: Path,
    permission_mode: str | None = None,
    can_use_tool: CanUseTool | None = None,
    mcp_servers: dict[str, Any] | None = None,
    hooks: dict[str, Any] | None = None,
) -> None:
    """Run a single task to completion, streaming output.

    Uses ClaudeSDKClient (streaming transport) rather than the one-shot
    ``query`` helper because ``can_use_tool`` requires streaming mode.
    """
    options = build_options(
        cwd=cwd,
        permission_mode=permission_mode,
        can_use_tool=can_use_tool,
        mcp_servers=mcp_servers,
        hooks=hooks,
    )
    async with ClaudeSDKClient(options=options) as client:
        await client.query(prompt)
        async for msg in client.receive_response():
            _render(msg)
    _maybe_consolidate()


async def run_chat(
    *,
    cwd: Path,
    permission_mode: str | None = None,
    can_use_tool: CanUseTool | None = None,
    mcp_servers: dict[str, Any] | None = None,
    hooks: dict[str, Any] | None = None,
) -> None:
    """Interactive multi-turn session. Ctrl-C or 'exit' to quit."""
    options = build_options(
        cwd=cwd,
        permission_mode=permission_mode,
        can_use_tool=can_use_tool,
        mcp_servers=mcp_servers,
        hooks=hooks,
    )
    console.print("[bold]ReLife chat[/] — type 'exit' to quit.\n")
    async with ClaudeSDKClient(options=options) as client:
        while True:
            try:
                user = console.input("[bold blue]you ›[/] ").strip()
            except (EOFError, KeyboardInterrupt):
                console.print("\n[dim]bye[/]")
                _maybe_consolidate()
                return
            if user.lower() in {"exit", "quit"}:
                console.print("[dim]bye[/]")
                _maybe_consolidate()
                return
            if not user:
                continue
            await client.query(user)
            async for msg in client.receive_response():
                _render(msg)
