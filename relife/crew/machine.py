"""Machine tools for crew members on other models — every call through ReLife's gate.

A ``runtime: llm`` crew member runs CrewAI's agent loop, which never passes
through the Claude SDK's ``can_use_tool`` — so until now it got memory and
nothing that touches the machine. These tools close that gap the only safe
way: the *tool* asks ReLife before it acts. Each call is described in the
terms ReLife's policy already understands (``Read``/``Glob``/``Write``/
``Edit``/``PowerShell``/``Bash`` with the same input fields) and handed to the
same ``can_use_tool`` callback a ReLife member gets — the terminal prompt under
``relife crew``, a browser card under ``relife serve``, a deny when nobody is
there. Nothing here decides what is allowed; ``permissions`` does.

The policy is ``classify_confined``: ``classify()`` plus one tightening — a read
outside the workspace asks, because what this member reads goes to its model's
provider.

Two opt-in toolsets, named on the agent in the plan (``"tools": [...]``):

- ``files`` — ``read_file``, ``list_files``, ``write_file``, ``edit_file``;
- ``shell`` — ``run_shell`` (PowerShell on Windows, bash elsewhere), run in the
  workspace with a timeout and bounded output.

The shell gate is ReLife's denylist, not a sandbox — the same caveat as for a
ReLife agent (``tests/test_permissions_corpus.py``). In particular the read
confinement binds the *file* tools only: ``run_shell`` can still ``cat`` a file
outside the workspace, so ``shell`` is the toolset to grant sparingly.

Pure of CrewAI (``machine_tools.py`` wraps these as CrewAI tools), so it is
tested on any Python. Every call — allowed or denied — is journaled into the
member's space under the ReLife tool name, so consolidation reads a shell step
by its command exactly as it does a ReLife agent's.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any, Awaitable, Callable

from .. import config

TOOLSETS = ("files", "shell")
SHELL_TIMEOUT = 120.0   # seconds per command
OUTPUT_MAX_CHARS = 20000  # what a call hands back to the model
LIST_MAX_ENTRIES = 500

CanUseTool = Callable[[str, dict[str, Any], Any], Awaitable[Any]]
Journal = Callable[[str, str], None]

# CrewAI-facing name -> the toolset that exposes it.
TOOL_NAMES = {
    "read_file": "files",
    "list_files": "files",
    "write_file": "files",
    "edit_file": "files",
    "run_shell": "shell",
}


def shell_tool() -> str:
    """The ReLife shell this platform's ``run_shell`` is judged (and run) as."""
    return "PowerShell" if os.name == "nt" else "Bash"


def _clip(text: str, limit: int = OUTPUT_MAX_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 40].rstrip() + f"\n… [truncated, {len(text) - limit + 40} more chars]"


class MachineTools:
    """The machine tools for one member in one workspace, gated by ``can_use_tool``."""

    def __init__(
        self,
        workspace: Path,
        can_use_tool: CanUseTool,
        *,
        journal: Journal | None = None,
        shell_timeout: float = SHELL_TIMEOUT,
        run: Callable[..., Any] = subprocess.run,
    ) -> None:
        self.workspace = Path(workspace)
        self._can_use_tool = can_use_tool
        self._journal = journal
        self._timeout = shell_timeout
        self._run = run

    # --- the gate ---------------------------------------------------------------
    async def _permit(self, tool: str, tool_input: dict[str, Any], brief: str) -> str | None:
        """None if the call may run, else the text to hand back to the model."""
        if self._journal is not None:
            try:
                self._journal(tool, brief)
            except Exception:  # noqa: BLE001 — journaling never breaks a run
                pass
        try:
            result = await self._can_use_tool(tool, tool_input, None)
        except Exception as e:  # noqa: BLE001 — a broken gate denies, never allows
            return f"DENIED: the permission check failed ({e})."
        if getattr(result, "behavior", None) == "allow":
            return None
        why = getattr(result, "message", "") or "not approved"
        return (
            f"DENIED: {why}. This action needs the user's approval and did not happen. "
            "Do not retry it; say in your answer what you could not do."
        )

    def _path(self, path: str) -> Path:
        """``path`` as the tool will touch it: relative ones are under the workspace.
        (Containment is the policy's job; it resolves ``..`` and symlinks.)"""
        p = Path(os.path.expanduser(str(path or ".")))
        return p if p.is_absolute() else self.workspace / p

    # --- files ----------------------------------------------------------------------
    async def read_file(self, path: str, offset: int = 0, limit: int | None = None) -> str:
        target = self._path(path)
        denied = await self._permit("Read", {"file_path": str(target)}, str(path))
        if denied:
            return denied
        try:
            text = target.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            return f"ERROR: {e}"
        lines = text.splitlines()
        start = max(0, int(offset or 0))
        end = len(lines) if not limit else start + max(0, int(limit))
        body = "\n".join(f"{i + 1:>6}\t{line}" for i, line in enumerate(lines[start:end], start))
        return _clip(body) if body else "(empty)"

    async def list_files(self, path: str = ".", pattern: str = "*") -> str:
        target = self._path(path)
        pattern = (pattern or "*").replace("\\", "/")
        if pattern.startswith("/") or ".." in pattern.split("/") or ":" in pattern:
            # The policy judges ``path``; a pattern must not walk out of it.
            return "ERROR: the pattern must stay inside path (no '..', no absolute pattern)"
        denied = await self._permit("Glob", {"path": str(target), "pattern": pattern}, f"{path} {pattern}")
        if denied:
            return denied
        if not target.is_dir():
            return f"ERROR: not a directory: {path}"
        out: list[str] = []
        try:
            for p in sorted(target.glob(pattern)):
                if ".git" in p.relative_to(target).parts:
                    continue
                out.append(p.relative_to(target).as_posix() + ("/" if p.is_dir() else ""))
                if len(out) >= LIST_MAX_ENTRIES:
                    out.append(f"… (stopped at {LIST_MAX_ENTRIES} entries)")
                    break
        except (OSError, ValueError) as e:
            return f"ERROR: {e}"
        return "\n".join(out) or "(no matches)"

    async def write_file(self, path: str, content: str) -> str:
        target = self._path(path)
        denied = await self._permit("Write", {"file_path": str(target), "content": content}, str(path))
        if denied:
            return denied
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        except OSError as e:
            return f"ERROR: {e}"
        return f"wrote {len(content)} chars to {path}"

    async def edit_file(self, path: str, old_text: str, new_text: str, replace_all: bool = False) -> str:
        target = self._path(path)
        denied = await self._permit(
            "Edit",
            {"file_path": str(target), "old_string": old_text, "new_string": new_text},
            str(path),
        )
        if denied:
            return denied
        try:
            text = target.read_text(encoding="utf-8")
        except OSError as e:
            return f"ERROR: {e}"
        count = text.count(old_text) if old_text else 0
        if count == 0:
            return "ERROR: old_text not found in the file (it must match exactly, whitespace included)"
        if count > 1 and not replace_all:
            return f"ERROR: old_text occurs {count} times; add surrounding lines to make it unique, or set replace_all"
        updated = text.replace(old_text, new_text) if replace_all else text.replace(old_text, new_text, 1)
        try:
            target.write_text(updated, encoding="utf-8")
        except OSError as e:
            return f"ERROR: {e}"
        return f"edited {path} ({count if replace_all else 1} replacement{'s' if replace_all and count > 1 else ''})"

    # --- shell ----------------------------------------------------------------------
    async def run_shell(self, command: str) -> str:
        command = (command or "").strip()
        if not command:
            return "ERROR: empty command"
        tool = shell_tool()
        denied = await self._permit(tool, {"command": command}, command)
        if denied:
            return denied
        argv = (
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", command]
            if tool == "PowerShell"
            else ["bash", "-c", command]
        )
        try:
            proc = self._run(
                argv,
                cwd=str(self.workspace),
                env={**os.environ, **config.agent_env()},
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self._timeout,
            )
        except subprocess.TimeoutExpired:
            return f"ERROR: timed out after {self._timeout:.0f}s"
        except OSError as e:
            return f"ERROR: {e}"
        out = (proc.stdout or "") + (("\n" + proc.stderr) if proc.stderr else "")
        return _clip(f"exit code {proc.returncode}\n{out.strip()}".rstrip())

    # --- what a toolset exposes ---------------------------------------------------
    def methods(self, toolsets: list[str] | tuple[str, ...]) -> dict[str, Callable[..., Awaitable[str]]]:
        """CrewAI-facing name -> bound coroutine method, for the chosen toolsets."""
        return {name: getattr(self, name) for name, ts in TOOL_NAMES.items() if ts in toolsets}
