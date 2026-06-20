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
    ResultMessage,
    SystemMessage,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
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


def _system_prompt() -> dict[str, Any]:
    """Use the Claude Code preset (keeps strong coding behavior) + our persona."""
    append = ""
    if config.SYSTEM_PROMPT_FILE.exists():
        append = config.SYSTEM_PROMPT_FILE.read_text(encoding="utf-8")
    return {"type": "preset", "preset": "claude_code", "append": append}


def build_options(
    *,
    cwd: Path,
    permission_mode: str | None = None,
    can_use_tool: CanUseTool | None = None,
    mcp_servers: dict[str, Any] | None = None,
    hooks: dict[str, Any] | None = None,
) -> ClaudeAgentOptions:
    """Assemble ClaudeAgentOptions from config + per-run overrides."""
    return ClaudeAgentOptions(
        model=config.MODEL,
        effort=config.EFFORT,
        system_prompt=_system_prompt(),
        cwd=str(cwd),
        permission_mode=permission_mode or config.DEFAULT_PERMISSION_MODE,
        can_use_tool=can_use_tool,
        mcp_servers=mcp_servers or {},
        hooks=hooks,
        # Ensure CLIs like `gh` are on PATH for the agent subprocess.
        env=config.agent_env(),
        # Don't inherit the surrounding repo's Claude Code settings — ReLife is
        # self-contained and defines its own behavior.
        setting_sources=None,
    )


def _render(msg: Any) -> None:
    """Pretty-print a streamed SDK message."""
    if isinstance(msg, AssistantMessage):
        for block in msg.content:
            if isinstance(block, TextBlock):
                if block.text.strip():
                    console.print(block.text)
            elif isinstance(block, ThinkingBlock):
                console.print(f"[dim italic]…thinking[/]")
            elif isinstance(block, ToolUseBlock):
                console.print(f"[cyan]→ {block.name}[/] [dim]{_tool_brief(block.input)}[/]")
    elif isinstance(msg, ResultMessage):
        cost = getattr(msg, "total_cost_usd", None)
        note = f"  [dim](usage-equiv ${cost:.4f})[/]" if cost else ""
        console.print(f"[green]✓ done[/]{note}")
    elif isinstance(msg, SystemMessage):
        # init / status frames — keep quiet unless debugging
        pass


def _tool_brief(inp: dict[str, Any]) -> str:
    """One-line hint of what a tool call is doing."""
    if not isinstance(inp, dict):
        return ""
    for key in ("command", "file_path", "path", "pattern", "url", "query"):
        if key in inp:
            val = str(inp[key])
            return val if len(val) <= 80 else val[:77] + "..."
    return ""


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
                return
            if user.lower() in {"exit", "quit"}:
                console.print("[dim]bye[/]")
                return
            if not user:
                continue
            await client.query(user)
            async for msg in client.receive_response():
                _render(msg)
