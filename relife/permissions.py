"""Permission policy: decide which tool calls run autonomously vs. need approval.

User-defined autonomy model for v1:
- **Auto-allow**: reading, browsing, editing files inside the workspace, building
  and testing code, and git (including ``git push``).
- **Always-ask**: anything outward-facing — sending email/messages, posting data
  off the machine, publishing, remote shells, writing outside the workspace, and
  any tool we don't yet recognize (safe default).

``classify()`` is a pure function (easy to unit-test). ``make_permission_callback()``
wraps it with an interactive terminal y/n prompt for the ask cases.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Awaitable, Callable

import anyio
from claude_agent_sdk import PermissionResultAllow, PermissionResultDeny
from rich.console import Console

console = Console(legacy_windows=False)

Decision = tuple[str, str]  # ("allow" | "ask", reason)

# Built-in tools that are always safe (read-only / planning / shell control).
_ALWAYS_ALLOW_TOOLS = {
    "Read", "Glob", "Grep", "LS", "NotebookRead",
    "TodoWrite", "WebFetch", "WebSearch", "Task",
    "BashOutput", "KillShell", "KillBash",
}

# Shell tools: same outward/destructive gating applies to whichever shell the
# agent picks (Bash on POSIX, PowerShell on Windows).
_SHELL_TOOLS = {"Bash", "PowerShell"}

# Built-in tools that write to the filesystem — allowed only inside the workspace.
_FILE_WRITE_TOOLS = {"Write", "Edit", "MultiEdit", "NotebookEdit"}

# MCP servers whose tools run autonomously:
#   mcp__relife  — our own memory/skills server (added in later stages)
#   mcp__browser — Playwright browsing (navigate/read/click/fill); a core v1
#                  capability the user wants the agent to use freely.
_TRUSTED_MCP_PREFIXES = ("mcp__relife", "mcp__browser")

# Bash commands that reach off the machine or are destructive → always ask.
# git (incl. push) is intentionally NOT here: the user authorized it.
_OUTWARD_BASH = re.compile(
    r"""
    \b(?:sendmail|mailx|mutt|mail)\b           # email senders
    | \bgh\s+(?:pr|issue|release|api|gist)\b   # GitHub CLI outward ops
    | \bcurl\b[^|;&]*\s(?:-d|--data\S*|-T|--upload-file|-X\s*(?:POST|PUT|DELETE|PATCH))\b
    | \bwget\b[^|;&]*--post                    # wget uploads
    | \b(?:scp|sftp|rsync|ssh)\b               # remote shells / copies
    | \b(?:twine\s+upload|npm\s+publish|yarn\s+publish|poetry\s+publish)\b  # package publish
    | \bsudo\b                                 # privilege escalation
    | \brm\s+-rf?\s+/(?:\s|$)                   # catastrophic deletes at root
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _under(path_str: str, workspace: Path) -> bool:
    """True if ``path_str`` resolves to a location inside ``workspace``."""
    try:
        target = Path(path_str)
        if not target.is_absolute():
            target = workspace / target
        target = target.resolve()
        workspace = workspace.resolve()
        return target == workspace or workspace in target.parents
    except Exception:
        return False


def classify(tool_name: str, tool_input: dict[str, Any], workspace: Path) -> Decision:
    """Decide whether a tool call may run autonomously.

    Returns ("allow", reason) or ("ask", reason).
    """
    if tool_name in _ALWAYS_ALLOW_TOOLS:
        return "allow", "read-only / planning tool"

    if tool_name in _FILE_WRITE_TOOLS:
        path = tool_input.get("file_path") or tool_input.get("path") or ""
        if path and _under(str(path), workspace):
            return "allow", "file write inside workspace"
        return "ask", f"file write outside workspace: {path or '?'}"

    if tool_name in _SHELL_TOOLS:
        command = str(tool_input.get("command", ""))
        if _OUTWARD_BASH.search(command):
            return "ask", "shell command looks outward-facing or destructive"
        return "allow", "build/test/git shell command"

    if tool_name.startswith(_TRUSTED_MCP_PREFIXES):
        return "allow", "ReLife-owned MCP tool"

    # Unknown MCP tools and anything else: ask (safe default; allowlist grows
    # as concrete git/browser tool names are wired in later stages).
    return "ask", "unrecognized tool — approval required by default"


def make_permission_callback(
    workspace: Path,
    *,
    interactive: bool | None = None,
) -> Callable[[str, dict[str, Any], Any], Awaitable[Any]]:
    """Build a ``can_use_tool`` callback bound to a workspace.

    ``interactive`` defaults to whether stdin is a TTY. When non-interactive,
    ask-cases are denied (so unattended runs never block, but also never take an
    unapproved outward action).
    """
    if interactive is None:
        interactive = sys.stdin.isatty()

    async def can_use_tool(tool_name: str, tool_input: dict[str, Any], context: Any):
        decision, reason = classify(tool_name, tool_input, workspace)
        if decision == "allow":
            return PermissionResultAllow()

        # ask path
        detail = tool_input.get("command") or tool_input.get("file_path") or ""
        console.print(
            f"\n[yellow]⚠ approval needed[/] [bold]{tool_name}[/] — {reason}"
        )
        if detail:
            console.print(f"  [dim]{str(detail)[:200]}[/]")

        if not interactive:
            console.print("  [red]denied[/] [dim](non-interactive run)[/]")
            return PermissionResultDeny(message=f"Denied (non-interactive): {reason}")

        # Prompt the user. If the prompt can't be read (no real input attached,
        # even when a pseudo-TTY makes isatty() true), fail closed → deny.
        try:
            raw = await anyio.to_thread.run_sync(input, "  allow this? [y/N] ")
        except (EOFError, KeyboardInterrupt):
            console.print("  [red]denied[/] [dim](no input available)[/]")
            return PermissionResultDeny(message=f"Denied (no approval input): {reason}")
        if raw.strip().lower() in {"y", "yes"}:
            return PermissionResultAllow()
        return PermissionResultDeny(message="User declined this action.")

    return can_use_tool
