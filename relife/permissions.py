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

import os
import re
import shlex
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

# claude.ai connectors (Gmail, Google Calendar, Google Drive). They ride the
# logged-in subscription — no keys, no local server — and surface as
# ``mcp__claude_ai_<Service>__<tool>``. These are the first *real* outward
# capability, so the policy is verb-based and fail-closed: a tool whose name
# says it only reads (search/list/get/…) runs autonomously, a tool whose name
# says it changes the outside world (send/create/delete/…) always asks, and a
# name that says neither asks too. The write check wins over the read check.
_CONNECTOR_PREFIX = "mcp__claude_ai_"
_CONNECTOR_READ = re.compile(
    r"(?:^|_)(?:search|list|get|read|fetch|find|query|lookup|show|view|count|"
    r"suggest|authenticate|complete_authentication)(?:_|$)",
    re.IGNORECASE,
)
# "draft" is a write verb (create_draft) *and* a noun: reading one is still a read.
_CONNECTOR_READ_DRAFT = re.compile(r"^(?:get|list|search|read|find)_drafts?$", re.IGNORECASE)
_CONNECTOR_WRITE = re.compile(
    r"(?:^|_)(?:send|create|delete|trash|untrash|modify|update|patch|move|archive|"
    r"reply|forward|draft|label|mark|insert|batch|write|remove|upload|share|"
    r"add|set|edit|rename|copy|import|export|accept|decline|rsvp|invite)(?:_|$)",
    re.IGNORECASE,
)


def _connector_tool(tool_name: str) -> str:
    """``mcp__claude_ai_Gmail__gmail_send_message`` → ``gmail_send_message``."""
    return tool_name[len(_CONNECTOR_PREFIX):].split("__", 1)[-1]

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
    # (GitHub CLI is verb-based — see ``_gh_outward``.)
    # --- HTTP writes / uploads / downloads-to-disk -------------------------
    | \bcurl\b[^|;&]*\s(?:-d|--data\S*|-F|--form\S*|-T|--upload-file|-X\s*(?:POST|PUT|DELETE|PATCH))\b
    | \bwget\b[^|;&]*--post
    | \b(?:Invoke-RestMethod|Invoke-WebRequest|irm|iwr)\b[^|;&]*
        (?:-Method\s*(?:POST|PUT|DELETE|PATCH)|-InFile|-OutFile)\b
    | \bStart-BitsTransfer\b
    | \.(?:Upload(?:File|String|Data|Values)|Download(?:String|File|Data))\w*\s*\(
    # --- text evaluated as code (download cradles, decoded payloads) --------
    | \b(?:curl|wget|Invoke-WebRequest|iwr)\b[^;&]*\|\s*(?:pwsh|powershell|python3?|node|perl|ruby)\b
    | \|\s*(?:sudo\s+)?(?:ba|z|k|da)?sh\b(?![.\w-])
    | \|\s*(?:pwsh|powershell)(?:\.exe)?\b
    | \b(?:Invoke-Expression|iex)\b
    | \b(?:ba|z|da)?sh\s+-c\s+["']?\$\(
    | \bbase64\b[^|;&]*\s(?:-d|--decode)\b[^;&]*\|
    | \s-(?:EncodedCommand|enc|ec)\s
    # --- remote shells / copies / raw sockets -------------------------------
    | \b(?:scp|sftp|rsync|ssh|telnet|socat|netcat|ncat)\b
    | (?:^|[;&|(]\s*)nc(?:\.exe)?\s
    | /dev/(?:tcp|udp)/
    | \bgit\s+push\b[^|;&]*\s(?:https?://|ssh://|git@)
    | \bgit\s+remote\s+(?:add|set-url)\b(?![^|;&\n]*github\.com[:/])
    | \b(?:New-PSSession|Enter-PSSession|Enable-PSRemoting)\b
    | \bInvoke-Command\b[^|;&]*-ComputerName\b
    # --- package publish ----------------------------------------------------
    | \b(?:twine\s+upload|(?:npm|pnpm|yarn|poetry|cargo|uv|hatch|flit)\s+publish)\b
    | \b(?:docker|podman|helm)\s+push\b
    | \bgem\s+push\b
    | \bmvn\b[^|;&]*\bdeploy\b
    | \b(?:Publish-Module|Publish-Script)\b
    | \bdotnet\s+nuget\s+push\b
    # --- privilege escalation / policy tampering ---------------------------
    | \b(?:sudo|doas|pkexec|gsudo|runas)\b
    | (?:^|[;&|(]\s*)su(?:\s|$)
    | \bStart-Process\b[^|;&]*-Verb\s+RunAs\b
    | \bSet-ExecutionPolicy\b
    # --- persistence / machine-wide config outside the workspace ------------
    | \b(?:crontab|schtasks|setx)\b
    | \b(?:Register|New|Set)-ScheduledTask\w*\b
    | \breg(?:\.exe)?\s+(?:add|delete|import|load|restore|copy)\b
    | \b(?:Set|New|Remove)-ItemProperty\b[^|;&]*(?:HK(?:CU|LM|CR|U)|Registry::)
    | \bSetEnvironmentVariable\b
    | \b(?:systemctl|launchctl)\s+(?:enable|disable|mask|stop|load|bootstrap)\b
    | \bNew-Service\b
    | \bsc(?:\.exe)?\s+(?:create|config|delete)\b
    | \bgit\s+config\b[^|;&]*--(?:global|system)\b
    | \b(?:npm|pnpm|yarn|pip3?|uv)\s+config\s+set\b
    | \b(?:Stop-Computer|Restart-Computer|shutdown|reboot|poweroff|Clear-RecycleBin)\b
    # --- catastrophic or device-level, i.e. unrecoverable ------------------
    | \brm\s+-[rRf]*\s*/(?:\s|$)
    | \b(?:mkfs(?:\.\w+)?|fdisk|diskpart)\b
    | \bdd\b[^|;&]*\bof=/dev/
    | \b(?:Format-Volume|Clear-Disk|Initialize-Disk)\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Package installs that land outside the workspace: the user's global Python,
# a global npm prefix, ~/.cargo, ~/go, the OS package manager. A release smoke
# run caught `python -m pip install -e .` silently installing into the user's
# global site-packages. Installs through a workspace venv (an explicit
# `.venv/…/python`, or a venv activated in the same command) or into an explicit
# `--target`/`--prefix` (checked by the write-target rule) stay autonomous.
_GLOBAL_INSTALL = re.compile(
    r"""
      (?:^|[;&|(]\s*|(?<!uv)\s)(?:pip3?|pip3?\.exe|(?:python3?|py)(?:\.exe)?(?:\s+-3[\d.]*)?\s+-m\s+pip)
        \s+install\b
    | \buv\s+pip\s+install\b[^|;&]*--system\b
    | \b(?:npm|pnpm)\s+(?:i|install|add)\b[^|;&]*\s(?:-g|--global)\b
    | \byarn\s+global\s+add\b
    | \b(?:cargo|go|gem|pipx)\s+install\b
    | \buv\s+tool\s+install\b
    | \bdotnet\s+tool\s+install\b[^|;&]*\s(?:-g|--global)\b
    | \b(?:winget|choco|scoop|brew)\s+install\b
    | \b(?:apt|apt-get|dnf|yum|pacman|zypper|apk)\s+(?:install|add|-S)\b
    | \bInstall-(?:Module|Package|Script)\b
    """,
    re.IGNORECASE | re.VERBOSE,
)
# A workspace-local interpreter or activation in the same command line.
_VENV_HINT = re.compile(
    r"""(?:^|[\s"'/\\;&(])\.?venv[/\\](?:bin|Scripts)[/\\]
      | \b(?:source|\.)\s+\S*activate\b | \bactivate(?:\.ps1|\.bat)?\b
      | \s(?:--target|-t|--prefix|--root)[\s=]""",
    re.IGNORECASE | re.VERBOSE,
)


def _global_install(command: str) -> bool:
    """True if ``command`` installs packages somewhere outside the workspace."""
    for segment in re.split(r"\|\||&&|[;\n]", command):
        if _GLOBAL_INSTALL.search(segment) and not (
            re.search(r"pip", segment, re.IGNORECASE) and _VENV_HINT.search(command)
        ):
            return True
    return False


# --- GitHub CLI ------------------------------------------------------------
# `gh` used to be gated by group (`gh pr|issue|release|api|gist` → ask), which
# was wrong in both directions: reading the user's own work items (`gh issue
# list --assignee @me`) prompted — so a scheduled "triage my issues" run could
# not even look — while `gh repo delete`, `gh secret set` and `gh workflow run`
# ran unasked. The policy is now verb-based and fail-closed, like connectors:
# every `gh` invocation in the command must be a known read (or a local op);
# anything else asks. `gh repo create` stays autonomous — it is the v1
# build → create → push flow the user authorized, same as `git push`.
_GH_READ: dict[str, frozenset[str] | None] = {
    # group → read-only subcommands (None = every subcommand reads)
    "pr": frozenset({"list", "view", "status", "diff", "checks", "checkout"}),
    "issue": frozenset({"list", "view", "status"}),
    "release": frozenset({"list", "view", "download"}),
    "run": frozenset({"list", "view", "watch", "download"}),
    "workflow": frozenset({"list", "view"}),
    "repo": frozenset({"list", "view", "clone", "create"}),
    "gist": frozenset({"list", "view", "clone"}),
    "label": frozenset({"list"}),
    "secret": frozenset({"list"}),
    "variable": frozenset({"list", "get"}),
    "cache": frozenset({"list"}),
    "ruleset": frozenset({"list", "view", "check"}),
    "project": frozenset({"list", "view", "field-list", "item-list"}),
    "auth": frozenset({"status"}),
    "search": None,
    "status": None,
    "help": None,
    "version": None,
}
# Flags that may precede the group and consume the next token as their value.
_GH_VALUE_FLAGS = {"-R", "--repo", "--hostname"}
# Every `gh` occurrence anywhere in the command (inside `bash -c "…"`, after
# `time`/`xargs`, a full path to gh.exe) — not just at a segment start, so a
# wrapper can't smuggle a write past the check. Prose that happens to say
# "gh …" asks, which is the safe direction.
_GH_CALL = re.compile(r"""\bgh(?:\.exe)?["']?(?=\s|$)([^;|&\n]*)""", re.IGNORECASE)
# `gh api` field flags switch the request to POST unless a method is given.
_GH_API_FIELDS = re.compile(r"^(?:-[fF]|--field|--raw-field|--input)(?:=|$)|^-[fF].")


def _gh_api_reads(args: list[str]) -> bool:
    """True if a `gh api …` call is a plain GET of a REST endpoint."""
    method: str | None = None
    has_fields = False
    endpoint: str | None = None
    i = 0
    while i < len(args):
        tok = args[i]
        if tok in ("-X", "--method"):
            method = args[i + 1] if i + 1 < len(args) else ""
            i += 2
            continue
        if tok.startswith("--method="):
            method = tok.split("=", 1)[1]
        elif tok.startswith("-X") and len(tok) > 2:
            method = tok[2:]
        elif _GH_API_FIELDS.match(tok):
            has_fields = True
        elif tok in ("-H", "--header", "-q", "--jq", "-t", "--template",
                     "--hostname", "--cache", "-p", "--preview"):
            i += 2  # these take a value that must not be read as the endpoint
            continue
        elif not tok.startswith("-") and endpoint is None:
            endpoint = tok
        i += 1
    if endpoint is None or endpoint.lower().lstrip("/") == "graphql":
        return False  # graphql can mutate; reading via REST/`gh search` suffices
    if method is not None:
        return method.upper() == "GET"
    return not has_fields


def _gh_outward(command: str) -> str | None:
    """The first `gh` invocation in ``command`` that isn't a known read, else None."""
    for match in _GH_CALL.finditer(command):
        args = [t.strip("\"'()`") for t in match.group(1).split()]
        args = [a for a in args if a]
        positional: list[str] = []
        i = 0
        while i < len(args) and len(positional) < 2:
            tok = args[i]
            if tok in _GH_VALUE_FLAGS:
                i += 2
                continue
            if not tok.startswith("-"):
                positional.append(tok.lower())
            i += 1
        if not positional:
            continue  # bare `gh`, `gh --version`, `gh --help`
        group = positional[0]
        if group == "api":
            if _gh_api_reads(_after(args, "api")):
                continue
            return "gh api " + " ".join(_after(args, "api"))[:60]
        if group not in _GH_READ:
            return f"gh {group}"
        subs = _GH_READ[group]
        if subs is None:
            continue
        sub = positional[1] if len(positional) > 1 else ""
        if sub not in subs:
            return f"gh {group} {sub}".rstrip()
    return None


def _after(args: list[str], word: str) -> list[str]:
    """Tokens following the first case-insensitive ``word`` in ``args``."""
    for i, tok in enumerate(args):
        if tok.lower() == word:
            return args[i + 1:]
    return []


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
    "rm", "rmdir", "unlink", "shred", "srm", "truncate",
    "del", "erase", "rd",
    "remove-item", "ri", "clear-content", "clc",
}
# Commands that write a file named as an argument (rather than via `>`).
_WRITE_VERBS = {
    "tee", "out-file", "set-content", "add-content", "tee-object",
    "new-item", "export-csv", "export-clixml",
    "touch", "mkdir", "md", "ni", "sc", "ac", "ln",
    "chmod", "chown", "chgrp", "icacls", "attrib",
}
# In-place editors (`sed -i 's/a/b/' f`): every operand counts — the script
# itself is harmless, it resolves as a relative path inside the workspace.
_INPLACE_VERBS = {"sed", "perl"}
# Same, but the destination is the LAST positional argument.
_COPY_VERBS = {"cp", "copy", "copy-item", "mv", "move", "move-item", "install"}
# Named parameters whose value is a destination path.
_DEST_PARAMS = {
    "-path", "-filepath", "-literalpath", "-destination", "-outfile", "-destinationpath",
    "-o", "--output", "--output-document", "--directory-prefix", "--target",
    "--directory", "--prefix", "--root",
}
# Destination switches that mean "output file/dir" only for one command.
_VERB_DEST_PARAMS = {
    "wget": {"-O", "-P"}, "tar": {"-C"}, "unzip": {"-d"}, "pip": {"-t"}, "pip3": {"-t"},
}
# Wrappers that run the next word as the command (`time rm …`, `xargs rm`, `& { … }`).
_PREFIX_WORDS = {
    "time", "env", "command", "exec", "nohup", "nice", "builtin", "xargs",
    "{", "(", "&", ".", "call",
}
_ENV_ASSIGN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
# Shells that take a whole command line as one argument.
_NESTED_SHELL = re.compile(
    r"""(?:^|[\s;&|(])(?:
          (?:ba|z|da|k)?sh(?:\.exe)?\s+(?:-\w+\s+)*-c
        | cmd(?:\.exe)?\s+(?:/[a-zA-Z]\s+)*/[cCkK]
        | (?:powershell|pwsh)(?:\.exe)?\s+(?:-\w+\s+)*-(?:c|Command)
        | Start-Process\b[^|;&]*-ArgumentList
    )\s+(?P<rest>.+)$""",
    re.IGNORECASE | re.VERBOSE | re.DOTALL,
)
# .NET file APIs called from PowerShell with a literal path.
_DOTNET_FILE = re.compile(
    r"::(?:(?:Write|Append)All\w*|Delete|Move|Copy|Replace|Create\w*)\(\s*['\"]([^'\"]+)['\"]",
    re.IGNORECASE,
)


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
    """Split a command line into rough token segments (quotes respected).

    Wrapper words (`time`, `env`, `VAR=x`, `xargs`, a PowerShell `& {`) are
    dropped so the real command comes first. `xargs` feeds operands we can't
    see, so its command gets an unresolvable one — a delete then asks.
    """
    out: list[list[str]] = []
    for part in _SEPARATOR.split(command):
        toks = [_unquote(t) for t in _TOKEN.findall(part)]
        fed = False
        while toks and (_verb(toks[0]) in _PREFIX_WORDS or _ENV_ASSIGN.match(toks[0])):
            if _verb(toks[0]) == "xargs":
                fed = True
                toks = toks[1:]
                while toks and toks[0].startswith("-"):
                    toks = toks[1:]  # xargs' own switches
                continue
            toks = toks[1:]
        toks = [t.strip("{}") for t in toks if t.strip("{}")]
        if toks:
            out.append(toks + (["$XARGS"] if fed else []))
    return out


# `/s` `/q` are switches only to cmd built-ins. To rm/cp/mv/tee… a token like
# `/d` is a path — in Git Bash the root of drive D: — so reading it as a
# switch auto-allowed `rm -rf /d`.
_SLASH_SWITCH_VERBS = {"del", "erase", "rd", "rmdir", "copy", "move", "xcopy", "md", "mkdir", "attrib", "icacls"}


def _positionals(tokens: list[str]) -> list[str]:
    """Argument tokens that look like operands rather than switches."""
    slash_switches = bool(tokens) and _verb(tokens[0]) in _SLASH_SWITCH_VERBS
    return [
        t for t in tokens[1:]
        if t and not t.startswith("-") and not (slash_switches and _FLAG.match(t))
    ]


def _named_dests(tokens: list[str]) -> list[str]:
    """Values of destination-style named parameters (`-OutFile x`, `-o x`,
    `--target=x`, `dd of=x`)."""
    dests: list[str] = []
    verb = _verb(tokens[0]) if tokens else ""
    verb_params = _VERB_DEST_PARAMS.get(verb, set())
    for i, tok in enumerate(tokens):
        name, eq, val = tok.partition("=") if tok.startswith("--") else (tok, "", "")
        if eq and name.lower() in _DEST_PARAMS:
            dests.append(val)
        elif (tok.lower() in _DEST_PARAMS or tok in verb_params) and i + 1 < len(tokens):
            nxt = tokens[i + 1]
            if nxt and not _FLAG.match(nxt):
                dests.append(nxt)
        elif verb == "dd" and tok.startswith("of="):
            dests.append(tok[3:])
    return dests


def _find_targets(tokens: list[str]) -> list[str]:
    """`find ROOTS… -delete` / `-exec rm …`: the roots are what gets deleted."""
    low = [t.lower() for t in tokens]
    deletes = "-delete" in low or any(
        low[i] in ("-exec", "-execdir", "-ok") and i + 1 < len(low)
        and _verb(low[i + 1]) in _DELETE_VERBS
        for i in range(len(low))
    )
    if not deletes:
        return []
    roots = []
    for t in tokens[1:]:
        if t.startswith(("-", "(", "!")):
            break
        roots.append(t)
    return roots or ["."]


def _inner_commands(command: str) -> list[str]:
    """Command lines handed to a nested shell (`bash -c "…"`, `cmd /c …`,
    `powershell -Command …`, `Start-Process … -ArgumentList …`), outermost first."""
    out: list[str] = []
    m = _NESTED_SHELL.search(command)
    while m and len(out) < 8:
        rest = m.group("rest").strip()
        if len(rest) >= 2 and rest[0] in "\"'" and rest[-1] == rest[0]:
            rest = rest[1:-1]
        elif rest[:1] in "\"'":
            rest = rest[1:]
        # `Start-Process cmd -ArgumentList '/c del …'`: the args still carry the switch.
        rest = re.sub(r"^(?:/[cCkK]|-c|-Command)\s+", "", rest)
        out.append(rest)
        m = _NESTED_SHELL.search(rest)
    return out


def _write_targets(command: str) -> list[str]:
    """Paths this command plausibly writes to (redirects + writer commands)."""
    targets = [
        t
        for t in (_unquote(m) for m in _REDIRECT.findall(command))
        if t and not t.startswith("&") and t.lower() not in _NULL_SINKS
    ]
    targets += _DOTNET_FILE.findall(command)
    for tokens in _segments(command):
        verb = _verb(tokens[0])
        if verb in _WRITE_VERBS:
            targets += _positionals(tokens) + _named_dests(tokens)
        elif verb in _INPLACE_VERBS and any(t.startswith("-i") for t in tokens[1:]):
            targets += _positionals(tokens)
        elif verb == "git" and len(tokens) > 1 and tokens[1].lower() == "clone":
            operands = _positionals(tokens[1:])
            if len(operands) >= 2:
                targets.append(operands[-1])  # explicit clone destination
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
        verb = _verb(tokens[0])
        if verb in _DELETE_VERBS:
            targets += _positionals(tokens) + _named_dests(tokens)
        elif verb == "find":
            targets += _find_targets(tokens)
    return targets


_MSYS_DRIVE = re.compile(r"^/([A-Za-z])(?:/|$)")


def _msys_to_windows(path_str: str) -> str:
    """``/d/relife/x`` → ``D:/relife/x``: how Git Bash (the ``Bash`` tool on
    Windows) spells a drive path — the agent's own ``pwd`` prints it that way.
    Only the drive form is translated; ``/tmp``, ``/usr`` … stay as they are
    (they map to install-specific places, so they keep asking)."""
    return _MSYS_DRIVE.sub(lambda m: f"{m.group(1).upper()}:/", path_str, count=1)  # bare /d → D:/, never drive-relative D:


def _escapes(targets: list[str], workspace: Path, *, strict: bool, msys: bool = False) -> str | None:
    """The first target not provably inside ``workspace``, else None.

    ``strict`` rejects *every* target we can't resolve (shell variables) — the
    right default for deletes, which are irreversible. Without it, only an
    unresolvable target that also names a path (`$HOME/.ssh/x`) is rejected, so
    an ordinary `… > $LOG` in the workspace doesn't start prompting.
    ``msys`` reads Git Bash drive paths (``/d/…``) as the Windows paths they are.
    """
    for target in targets:
        if _unresolvable(target):
            if strict or "/" in target or "\\" in target:
                return target
            continue
        if not _under(_msys_to_windows(target) if msys else target, workspace):
            return target
    return None


def _under(path_str: str, workspace: Path) -> bool:
    """True if ``path_str`` resolves to a location inside ``workspace``."""
    try:
        if os.name != "nt":
            # A Windows path on POSIX (`C:\Users\…`, `\\server\share`) isn't
            # absolute to pathlib, so it would resolve *inside* the workspace;
            # it can never be inside a POSIX workspace. And `..\..` still means
            # "up two" to whatever shell a Windows-minded command targets.
            if re.match(r"^(?:[A-Za-z]:[\\/]|\\\\)", path_str):
                return False
            path_str = path_str.replace("\\", "/")
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
        # A nested shell (`cmd /c "del C:\\x"`, `pwsh -Command …`) hides its
        # command line inside one quoted token; judge that line on its own too.
        for inner in _inner_commands(command):
            decision, reason = classify(tool_name, {"command": inner}, workspace)
            if decision == "ask":
                return decision, f"{reason} (nested shell)"
        if _OUTWARD_SHELL.search(command):
            return "ask", "shell command looks outward-facing or destructive"
        if _global_install(command):
            return "ask", "installs packages outside the workspace (global environment)"
        gh = _gh_outward(command)
        if gh:
            return "ask", f"GitHub CLI action changes the outside world: {gh}"
        # The workspace is a boundary for the shell too, not just for Write/Edit
        # — otherwise a single redirect walks around the whole file-write policy.
        # Git Bash on Windows spells D:\x as /d/x; PowerShell doesn't.
        msys = tool_name == "Bash" and os.name == "nt"
        escaped = _escapes(_write_targets(command), workspace, strict=False, msys=msys)
        if escaped:
            return "ask", f"shell writes outside the workspace: {escaped}"
        escaped = _escapes(_delete_targets(command), workspace, strict=True, msys=msys)
        if escaped:
            return "ask", f"shell deletes outside the workspace: {escaped}"
        return "allow", "build/test/git shell command"

    if tool_name.startswith(_TRUSTED_MCP_PREFIXES):
        return "allow", "ReLife-owned MCP tool"

    if tool_name.startswith(_CONNECTOR_PREFIX):
        op = _connector_tool(tool_name)
        if _CONNECTOR_READ_DRAFT.match(op):
            return "allow", f"connector read: {op}"
        if _CONNECTOR_WRITE.search(op):
            return "ask", f"connector action changes the outside world: {op}"
        if _CONNECTOR_READ.search(op):
            return "allow", f"connector read: {op}"
        return "ask", f"connector tool with unrecognized effect: {op}"

    # Unknown MCP tools and anything else: ask (safe default; allowlist grows
    # as concrete git/browser tool names are wired in later stages).
    return "ask", "unrecognized tool — approval required by default"


# --- pre-authorized grants (scheduled runs) ---------------------------------
# An unattended scheduled run has nobody at the approval card, so an ask-case
# times out to deny — which makes "summarize my inbox and email me the digest"
# impossible to finish. A schedule may therefore carry a few *grants*: narrow,
# user-set pre-approvals that turn specific connector asks into allows.
#
# This is a deliberate policy widening, so it is kept pure and fail-closed:
#   * grants only ever cover claude.ai connector actions — never the shell,
#     file writes, or an unknown tool;
#   * each kind names a service and an operation shape, and anything
#     destructive (delete/modify/share/…) is excluded even if it also matches;
#   * every email address in the call — outside free-text content — must be on
#     the grant's list, and a recipient field holding anything that isn't an
#     allowlisted address (a group alias, garbage) fails the check;
#   * email must name at least one recipient we can see (a reply that infers
#     its recipient from the thread, or a raw MIME blob, falls back to asking).
#
# The one exception to "connectors only" is ``pull_request``, and it is the
# narrowest grant there is: stored on a *work schedule* as a bare
# ``{"kind": "pull_request"}`` that matches nothing; the scheduler binds it per
# turn to the issue's repo and branch, and only then does it allow exactly one
# command shape — a single ``gh pr create`` whose ``--repo``/``--head`` are
# that repo and branch, with only title/body/base/draft/fill flags, and no
# ``$``, backtick, redirect, pipe, chain or extra argument anywhere.
GRANT_KINDS = ("email", "calendar", "pull_request")
GRANT_MAX_ADDRESSES = 5
_EMAIL = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
_EMAIL_FULL = re.compile(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$")
# Free-text fields: an address mentioned in a digest body is content, not a recipient.
_CONTENT_KEY = re.compile(
    r"^(?:body|content|text|html|plain|message|subject|snippet|description|summary|"
    r"title|notes?|location)(?:_|$)",
    re.IGNORECASE,
)
_RECIPIENT_KEY = re.compile(
    r"^(?:to|cc|bcc|recipients?|attendees?|guests?|invitees?|email|(?:added_)?attendee_emails)$",
    re.IGNORECASE,
)
# Fields that make an email call reach people its visible recipients don't
# name (checked against the real Gmail connector schema): `draftId` sends a
# stored draft as-is and ignores to/cc/bcc, and `replyAll` keeps the thread's
# CC list. Either one ⇒ the grant can't vouch for the call ⇒ ask.
_EMAIL_HIDDEN_RECIPIENTS = ("draft_id", "reply_all")


def _snake(key: str) -> str:
    """``htmlBody`` → ``html_body``: the connectors use camelCase field names."""
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", key).lower()


def _is_content_key(key: str) -> bool:
    k = _snake(key)
    return bool(_CONTENT_KEY.match(k) or k == "forward_text") and not k.endswith("_id")
_GRANT_OPS: dict[str, tuple[str, re.Pattern[str]]] = {
    # kind → (service substring in the tool name, op shape)
    "email": ("gmail", re.compile(r"(?:^|_)(?:send|reply|forward|draft)(?:_|$)", re.IGNORECASE)),
    "calendar": (
        "calendar",
        re.compile(r"(?:^|_)(?:create|insert|add|quick_?add)(?:_\w*)?_?event", re.IGNORECASE),
    ),
}
_GRANT_NEVER = re.compile(
    r"(?:^|_)(?:delete|trash|remove|modify|update|patch|move|archive|label|share|batch|"
    r"import|export|upload|decline|accept|rsvp)(?:_|$)",
    re.IGNORECASE,
)


def normalize_grants(raw: Any) -> list[dict[str, Any]]:
    """Validate a schedule's grants → canonical list. Raises ``ValueError``.

    Shape: ``[{"kind": "email", "addresses": ["me@x.com"]},
    {"kind": "calendar", "addresses": []}]`` — at most one grant per kind.
    An email grant needs at least one address; a calendar grant with none
    means "events with no guests at all".
    """
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError("grants must be a list")
    out: dict[str, dict[str, Any]] = {}
    for g in raw:
        if not isinstance(g, dict):
            raise ValueError("each grant must be an object")
        kind = str(g.get("kind", "")).strip().lower()
        if kind not in GRANT_KINDS:
            raise ValueError(f"unknown grant kind {kind!r} (expected one of {', '.join(GRANT_KINDS)})")
        if kind in out:
            raise ValueError(f"duplicate {kind} grant")
        if kind == "pull_request":
            # Stored unbound: repo/branch are the scheduler's to fill per turn,
            # never the caller's — anything else in the object is dropped.
            out[kind] = {"kind": kind}
            continue
        addrs = g.get("addresses", [])
        if isinstance(addrs, str):
            addrs = [a for a in re.split(r"[,;\s]+", addrs) if a]
        if not isinstance(addrs, list):
            raise ValueError(f"{kind} grant: addresses must be a list")
        clean: list[str] = []
        for a in addrs:
            a = str(a).strip().lower()
            if not _EMAIL_FULL.match(a):
                raise ValueError(f"{kind} grant: not an email address: {a!r}")
            if a not in clean:
                clean.append(a)
        if len(clean) > GRANT_MAX_ADDRESSES:
            raise ValueError(f"{kind} grant: at most {GRANT_MAX_ADDRESSES} addresses")
        if kind == "email" and not clean:
            raise ValueError("email grant needs at least one address (who may be emailed)")
        out[kind] = {"kind": kind, "addresses": clean}
    return [out[k] for k in GRANT_KINDS if k in out]


def describe_grant(grant: dict[str, Any]) -> str:
    if grant.get("kind") == "pull_request":
        if grant.get("repo") and grant.get("branch"):
            return f"open one pull request in {grant['repo']} from {grant['branch']}"
        return "open the pull request for the issue branch it pushed"
    addrs = ", ".join(grant.get("addresses") or [])
    if grant.get("kind") == "email":
        return f"send email only to {addrs}"
    return f"add calendar events with guests only {addrs}" if addrs else "add calendar events with no guests"


def _addresses_in(value: Any, key: str = "") -> tuple[set[str], bool]:
    """Every email address in ``value`` outside free-text content, plus whether a
    recipient-shaped field held something that isn't a plain address."""
    found: set[str] = set()
    junk = False
    if isinstance(value, dict):
        for k, v in value.items():
            f, j = _addresses_in(v, str(k))
            found |= f
            junk |= j
    elif isinstance(value, (list, tuple)):
        for v in value:
            f, j = _addresses_in(v, key)
            found |= f
            junk |= j
    elif isinstance(value, str):
        if key and _is_content_key(key):
            return found, junk
        found |= {a.lower() for a in _EMAIL.findall(value)}
        if key and _RECIPIENT_KEY.match(_snake(key)):
            # "Me <me@x.com>, ops-team" — every token must be an address.
            for part in re.split(r"[,;]", value):
                part = part.strip()
                if part and not _EMAIL.search(part):
                    junk = True
    return found, junk


# `gh pr create` flags a bound pull_request grant tolerates → takes a value?
_PR_FLAGS: dict[str, bool] = {
    "--repo": True, "-R": True, "--head": True, "-H": True, "--base": True, "-B": True,
    "--title": True, "-t": True, "--body": True, "-b": True, "--draft": False, "-d": False,
    "--fill": False, "-f": False,
}
_PR_ALIASES = {"-R": "--repo", "-H": "--head", "-B": "--base", "-t": "--title",
               "-b": "--body", "-d": "--draft", "-f": "--fill"}
_PR_FORBIDDEN_CHARS = re.compile(r"[$`]")  # expansion/substitution in bash *and* PowerShell
_PR_OPERATOR = re.compile(r"^[;&|<>()]+$")


def _pr_create_matches(grant: dict[str, Any], tool_input: dict[str, Any]) -> bool:
    """Pure: is this shell call exactly one bound-safe ``gh pr create``?"""
    repo, branch = str(grant.get("repo") or ""), str(grant.get("branch") or "")
    command = tool_input.get("command") if isinstance(tool_input, dict) else None
    if not repo or not branch or not isinstance(command, str):
        return False  # an unbound grant matches nothing
    if _PR_FORBIDDEN_CHARS.search(command):
        return False
    try:
        lex = shlex.shlex(command, posix=True, punctuation_chars=True)
        lex.whitespace_split = True
        lex.commenters = ""  # a `#` must not hide the rest of the line from us
        tokens = list(lex)
    except ValueError:
        return False  # unbalanced quotes
    if any(_PR_OPERATOR.match(t) for t in tokens):
        return False  # ; | & && || > < ( ) — a second command or a redirect
    if len(tokens) < 3 or Path(tokens[0]).name.lower() not in ("gh", "gh.exe"):
        return False
    if [t.lower() for t in tokens[1:3]] != ["pr", "create"]:
        return False
    seen: dict[str, str] = {}
    i = 3
    while i < len(tokens):
        tok = tokens[i]
        flag, eq, val = tok.partition("=") if tok.startswith("--") else (tok, "", "")
        if flag not in _PR_FLAGS:
            return False  # positional args, --body-file, --reviewer, --label, --web, …
        name = _PR_ALIASES.get(flag, flag)
        if name in seen:
            return False
        if _PR_FLAGS[flag]:
            if not eq:
                if i + 1 >= len(tokens):
                    return False
                i += 1
                val = tokens[i]
            seen[name] = val
        elif eq:
            return False
        else:
            seen[name] = ""
        i += 1
    return (
        seen.get("--repo", "").lower() == repo.lower()
        and seen.get("--head") == branch
        and not seen.get("--base", "").startswith("-")
    )


def grant_allows(
    grants: list[dict[str, Any]], tool_name: str, tool_input: dict[str, Any]
) -> str | None:
    """Pure: the description of the grant that pre-authorizes this call, or
    ``None`` (→ the normal ask path). Never consulted for a call ``classify``
    already allows, and never widens anything but connector actions."""
    if not grants:
        return None
    if tool_name in _SHELL_TOOLS:
        for grant in grants:
            if grant.get("kind") == "pull_request" and _pr_create_matches(grant, tool_input):
                return describe_grant(grant)
        return None
    if not tool_name.startswith(_CONNECTOR_PREFIX):
        return None
    service = tool_name[len(_CONNECTOR_PREFIX):].split("__", 1)[0].lower()
    op = _connector_tool(tool_name)
    if _GRANT_NEVER.search(op):
        return None
    for grant in grants:
        needle, shape = _GRANT_OPS.get(grant.get("kind", ""), ("", None))
        if not needle or needle not in service or shape is None or not shape.search(op):
            continue
        inp = tool_input if isinstance(tool_input, dict) else {}
        if grant["kind"] == "email" and any(
            _snake(k) in _EMAIL_HIDDEN_RECIPIENTS and v not in (None, False, "") for k, v in inp.items()
        ):
            continue  # draft sent as stored / reply-all ⇒ recipients we can't see ⇒ ask
        found, junk = _addresses_in(inp)
        allowed = {a.lower() for a in grant.get("addresses") or []}
        if junk or not found <= allowed:
            continue
        if grant["kind"] == "email" and not found:
            continue  # recipient we can't see ⇒ ask
        return describe_grant(grant)
    return None


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

        # ask path — show what is about to happen. A shell command or a path
        # for the built-ins; for a connector call (an email about to be sent)
        # the leading fields, so the user approves something concrete.
        from .agent import _tool_brief  # lazy: agent.py sets up the console

        detail = _tool_brief(tool_input, limit=400)
        console.print(
            f"\n[yellow]⚠ approval needed[/] [bold]{tool_name}[/] — {reason}"
        )
        if detail:
            console.print(f"  [dim]{detail}[/]")

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
    preauthorize: Callable[[str, dict[str, Any]], Awaitable[str | None]] | None = None,
) -> Callable[[str, dict[str, Any], Any], Awaitable[Any]]:
    """Build a ``can_use_tool`` that routes ask-cases to a UI approval broker.

    Same policy as the terminal path — the pure ``classify()`` decides — but
    instead of prompting a TTY, ask-cases are pushed to ``broker`` (which surfaces
    them in the web UI) and the run blocks awaiting the browser's decision. If no
    decision arrives within ``timeout`` seconds the broker returns ``False`` and
    the action is denied (safe default, matching the non-interactive TTY path).

    ``broker`` must expose an async ``request(tool_name, tool_input, reason, *,
    timeout) -> bool``.

    ``preauthorize`` (optional) is consulted for ask-cases *before* the broker:
    if it returns a grant description the call is allowed without a card. The
    session supplies it, applying the current turn's grants (``grant_allows``)
    plus a per-run use cap.
    """

    async def can_use_tool(tool_name: str, tool_input: dict[str, Any], context: Any):
        decision, reason = classify(tool_name, tool_input, workspace)
        if decision == "allow":
            return PermissionResultAllow()

        if preauthorize is not None and await preauthorize(tool_name, tool_input):
            return PermissionResultAllow()

        approved = await broker.request(tool_name, tool_input, reason, timeout=timeout)
        if approved:
            return PermissionResultAllow()
        return PermissionResultDeny(message=f"Denied via UI: {reason}")

    return can_use_tool
