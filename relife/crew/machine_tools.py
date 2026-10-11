"""The gated machine tools (``machine.py``) as CrewAI tools.

Each one is a thin ``BaseTool`` over a :class:`MachineTools` method: CrewAI
calls ``_run`` synchronously on the crew's thread, and the coroutine — the
permission check first, then the action — runs to completion there through
``turns.run_sync``. The gate itself (terminal prompt, browser card, or deny)
is whatever ``can_use_tool`` the crew was built with.
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable, Optional

from crewai.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr

from .machine import MachineTools, shell_tool
from .turns import run_sync


class ReadFileArgs(BaseModel):
    path: str = Field(..., description="File to read, relative to the workspace.")
    offset: Optional[int] = Field(None, description="First line to return (0-based).")
    limit: Optional[int] = Field(None, description="How many lines to return.")


class ListFilesArgs(BaseModel):
    path: Optional[str] = Field(None, description="Directory, relative to the workspace (default: the workspace).")
    pattern: Optional[str] = Field(None, description="Glob pattern, e.g. '*.py' or '**/*.md' (default '*').")


class WriteFileArgs(BaseModel):
    path: str = Field(..., description="File to create or overwrite, relative to the workspace.")
    content: str = Field(..., description="The whole new content of the file.")


class EditFileArgs(BaseModel):
    path: str = Field(..., description="File to edit, relative to the workspace.")
    old_text: str = Field(..., description="Exact text to replace (must occur once unless replace_all).")
    new_text: str = Field(..., description="Replacement text.")
    replace_all: Optional[bool] = Field(None, description="Replace every occurrence.")


class RunShellArgs(BaseModel):
    command: str = Field(..., description="The command line to run in the workspace.")


def _descriptions() -> dict[str, tuple[str, type[BaseModel]]]:
    shell = shell_tool()
    return {
        "read_file": ("Read a text file (numbered lines). Paths are relative to the workspace.", ReadFileArgs),
        "list_files": ("List files in a directory of the workspace, optionally by glob pattern.", ListFilesArgs),
        "write_file": ("Create or overwrite a file with the given content.", WriteFileArgs),
        "edit_file": ("Replace exact text in a file (old_text must be unique unless replace_all).", EditFileArgs),
        "run_shell": (
            f"Run a {shell} command in the workspace and return its exit code and output "
            f"(build, test, git, …). Times out; outward or destructive commands need the user's approval.",
            RunShellArgs,
        ),
    }


class GatedMachineTool(BaseTool):
    """One machine tool; every call is checked by ReLife's permission gate."""

    _call: Callable[..., Awaitable[str]] = PrivateAttr()

    def _run(self, **kwargs: Any) -> str:
        args = {k: v for k, v in kwargs.items() if v is not None}
        try:
            return run_sync(lambda: self._call(**args))
        except Exception as e:  # noqa: BLE001 — surface to the model
            return f"ERROR: {e}"


def machine_tools(tools: MachineTools, toolsets: list[str] | tuple[str, ...]) -> list[BaseTool]:
    """The chosen toolsets of ``tools`` as CrewAI tools."""
    desc = _descriptions()
    out: list[BaseTool] = []
    for name, method in tools.methods(toolsets).items():
        text, schema = desc[name]
        tool = GatedMachineTool(name=name, description=text, args_schema=schema)
        tool._call = method
        out.append(tool)
    return out
