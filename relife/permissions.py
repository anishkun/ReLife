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

# Shell commands that reach off the machine, escalate privilege, or are
# unrecoverable → always ask. Covers BOTH shells the agent is given: POSIX tools
# and their PowerShell equivalents. On Windows PowerShell is the shell the agent
# actually reaches for, so a Bash-only pattern set left the policy open on the
# very platform ReLife runs on (`Send-MailMessage` was auto-allowed).
# git (incl. push) is intentionally NOT here: the user authorized it.
_OUTWARD_SHELL = re.compile(
    r"""
    # --- email senders -----------------------------------------------------
      \b(?:sendmail|mailx|mutt|mail)\b
    | \bSend-MailMessage\b
    # --- GitHub CLI outward ops --------------------------------------------
    | \bgh\s+(?:pr|issue|release|api|gist)\b
    # --- HTTP writes / uploads / downloads-to-disk -------------------------
    | \bcurl\b[^|;&]*\s(?:-d|--data\S*|-T|--upload-file|-X\s*(?:POST|PUT|DELETE|PATCH))\b
    | \bwget\b[^|;&]*--post
    | \b(?:Invoke-RestMethod|Invoke-WebRequest|irm|iwr)\b[^|;&]*
        (?:-Method\s*(?:POST|PUT|DELETE|PATCH)|-InFile|-OutFile)\b
    | \bStart-BitsTransfer\b
    # --- a download piped straight into an interpreter ---------------------
    | \b(?:curl|wget|Invoke-WebRequest|iwr)\b[^;&]*\|\s*(?:sudo\s+)?(?:ba|z|k)?sh\b
    | \b(?:curl|wget|Invoke-WebRequest|iwr)\b[^;&]*\|\s*(?:pwsh|powershell|python3?|node|perl|ruby)\b
    # --- remote shells / copies --------------------------------------------
    | \b(?:scp|sftp|rsync|ssh)\b
    | \b(?:New-PSSession|Enter-PSSession|Enable-PSRemoting)\b
    | \bInvoke-Command\b[^|;&]*-ComputerName\b
    # --- package publish ----------------------------------------------------
    | \b(?:twine\s+upload|npm\s+publish|yarn\s+publish|poetry\s+publish)\b
    | \b(?:Publish-Module|Publish-Script)\b
    | \bdotnet\s+nuget\s+push\b
    # --- privilege escalation / policy tampering ---------------------------
    | \bsudo\b
    | \bStart-Process\b[^|;&]*-Verb\s+RunAs\b
    | \bSet-ExecutionPolicy\b
    # --- catastrophic or device-level, i.e. unrecoverable ------------------
    | \brm\s+-[rRf]*\s*/(?:\s|$)
    | \b(?:mkfs(?:\.\w+)?|fdisk|diskpart)\b
    | \bdd\b[^|;&]*\bof=/dev/
    | \b(?:Format-Volume|Clear-Disk|Initialize-Disk)\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

# --- shell path analysis ----------------------------------------------------
# The workspace boundary used to be enforced only on the Write/Edit tools, so
# any shell redirect walked straight around it (`echo x > ~/.bashrc` was
# auto-allowed). These helpers give ``classify()`` a best-effort view of what a
# shell command *writes to* and *deletes*, so one containment rule covers both
# doors.
#
# Deliberately heuristic: shell grammar isn't parseable with a regex, and the
# goal is to close the common escapes, not to reimplement a shell. Unresolvable
# targets (shell variables, subshells) count as outside for deletes — an
# unevaluated `rm -rf "$DIR"` is exactly the case worth a prompt.

_TOKEN = re.compile(r"\"[^\"]*\"|'[^']*'|\S+")
# A switch, not a path: POSIX `-rf`, or a cmd.exe single-letter switch (`/s`,
# `/q`, `/a:h`). Single-letter-only keeps real POSIX paths (`/etc`) out of it.
_FLAG = re.compile(r"^-|^/[a-zA-Z](?::\w+)?$")
# Redirections: `> f`, `>> f`, `2> f`. `>&1` / `2>&1` can't match (the target
# class excludes `&`), so fd duplication is ignored, as it should be.
_REDIRECT = re.compile(r"\d?>>?\s*(\"[^\"]*\"|'[^']*'|[^\s|;&<>]+)")
# Segment separators: `;` `|` `||` `&&` `&` and newlines.
_SEPARATOR = re.compile(r"\|\||&&|[;|&\n]")

# Sinks that discard output — never a real file write.
_NULL_SINKS = {"/dev/null", "/dev/stdout", "/dev/stderr", "nul", "nul:", "con", "$null"}

# Commands that delete. Recursion/force changes the blast radius, not whether
# the containment rule applies, so plain `rm` belongs here too.
_DELETE_VERBS = {
    "rm", "rmdir", "unlink", "shred", "srm",
    "del", "erase", "rd",
    "remove-item", "ri", "clear-content",
}
# Commands that write a file named as an argument (rather than via `>`).
_WRITE_VERBS = {
    "tee", "out-file", "set-content", "add-content", "tee-object",
    "new-item", "export-csv", "export-clixml",
}
# Same, but the destination is the LAST positional argument.
_COPY_VERBS = {"cp", "copy", "copy-item", "mv", "move", "move-item", "install"}
# Named parameters whose value is a destination path.
_DEST_PARAMS = {"-path", "-filepath", "-literalpath", "-destination", "-outfile"}


def _unquote(tok: str) -> str:
    if len(tok) >= 2 and tok[0] == tok[-1] and tok[0] in "\"'":
        return tok[1:-1]
    return tok


def _verb(tok: str) -> str:
    r"""Normalize a command word: ``C:\Windows\System32\del.exe`` → ``del``."""
    name = _unquote(tok).replace("\\", "/").rsplit("/", 1)[-1].lower()
    return name[:-4] if name.endswith(".exe") else name


def _unresolvable(path_str: str) -> bool:
    """True if the target depends on runtime expansion we can't evaluate."""
    return any(ch in path_str for ch in ("$", "%", "`"))


def _segments(command: str) -> list[list[str]]:
    """Split a command line into rough token segments (quotes respected)."""
    out: list[list[str]] = []
    for part in _SEPARATOR.split(command):
        toks = [_unquote(t) for t in _TOKEN.findall(part)]
        if toks:
            out.append(toks)
    return out


def _positionals(tokens: list[str]) -> list[str]:
    """Argument tokens that look like operands rather than switches."""
    return [t for t in tokens[1:] if t and not _FLAG.match(t)]


def _named_dests(tokens: list[str]) -> list[str]:
    """Values of destination-style named parameters (`-OutFile x`, `-Path x`)."""
    dests: list[str] = []
    for i, tok in enumerate(tokens):
        if tok.lower() in _DEST_PARAMS and i + 1 < len(tokens):
            nxt = tokens[i + 1]
            if nxt and not _FLAG.match(nxt):
                dests.append(nxt)
    return dests


def _write_targets(command: str) -> list[str]:
    """Paths this command plausibly writes to (redirects + writer commands)."""
    targets = [
        t
        for t in (_unquote(m) for m in _REDIRECT.findall(command))
        if t and not t.startswith("&") and t.lower() not in _NULL_SINKS
    ]
    for tokens in _segments(command):
        verb = _verb(tokens[0])
        if verb in _WRITE_VERBS:
            targets += _positionals(tokens) + _named_dests(tokens)
        elif verb in _COPY_VERBS:
            positional = _positionals(tokens)
            if positional:
                targets.append(positional[-1])  # destination is last
            targets += _named_dests(tokens)
        else:
            # `-OutFile`/`-Destination` are unambiguous wherever they appear.
            targets += _named_dests(tokens)
    return targets


def _delete_targets(command: str) -> list[str]:
    """Paths this command plausibly deletes."""
    targets: list[str] = []
    for tokens in _segments(command):
        if _verb(tokens[0]) in _DELETE_VERBS:
            targets += _positionals(tokens) + _named_dests(tokens)
    return targets


def _escapes(targets: list[str], workspace: Path, *, strict: bool) -> str | None:
    """The first target not provably inside ``workspace``, else None.

    ``strict`` rejects *every* target we can't resolve (shell variables) — the
    right default for deletes, which are irreversible. Without it, only an
    unresolvable target that also names a path (`$HOME/.ssh/x`) is rejected, so
    an ordinary `… > $LOG` in the workspace doesn't start prompting.
    """
    for target in targets:
        if _unresolvable(target):
            if strict or "/" in target or "\\" in target:
                return target
            continue
        if not _under(target, workspace):
            return target
    return None


def _under(path_str: str, workspace: Path) -> bool:
    """True if ``path_str`` resolves to a location inside ``workspace``."""
    try:
        # `~` must expand before the containment test, or `rm -rf ~/Documents`
        # looks like a relative path *inside* the workspace.
        target = Path(path_str).expanduser()
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
        if _OUTWARD_SHELL.search(command):
            return "ask", "shell command looks outward-facing or destructive"
        # The workspace is a boundary for the shell too, not just for Write/Edit
        # — otherwise a single redirect walks around the whole file-write policy.
        escaped = _escapes(_write_targets(command), workspace, strict=False)
        if escaped:
            return "ask", f"shell writes outside the workspace: {escaped}"
        escaped = _escapes(_delete_targets(command), workspace, strict=True)
        if escaped:
            return "ask", f"shell deletes outside the workspace: {escaped}"
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


def make_approval_callback(
    workspace: Path,
    broker: Any,
    *,
    timeout: float,
) -> Callable[[str, dict[str, Any], Any], Awaitable[Any]]:
    """Build a ``can_use_tool`` that routes ask-cases to a UI approval broker.

    Same policy as the terminal path — the pure ``classify()`` decides — but
    instead of prompting a TTY, ask-cases are pushed to ``broker`` (which surfaces
    them in the web UI) and the run blocks awaiting the browser's decision. If no
    decision arrives within ``timeout`` seconds the broker returns ``False`` and
    the action is denied (safe default, matching the non-interactive TTY path).

    ``broker`` must expose an async ``request(tool_name, tool_input, reason, *,
    timeout) -> bool``.
    """

    async def can_use_tool(tool_name: str, tool_input: dict[str, Any], context: Any):
        decision, reason = classify(tool_name, tool_input, workspace)
        if decision == "allow":
            return PermissionResultAllow()

        approved = await broker.request(tool_name, tool_input, reason, timeout=timeout)
        if approved:
            return PermissionResultAllow()
        return PermissionResultDeny(message=f"Denied via UI: {reason}")

    return can_use_tool
