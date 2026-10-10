"""ReLife memory as CrewAI tools — for crew members on other LLMs.

A CrewAI agent running on Llama, GPT or Gemini gets ReLife memory through the
very same tool specs every other agent sees (``memory/tools.py``
``EXTERNAL_TOOLS``), bound to *its* ``ScopedMemoryClient``: it saves into its
own space and recalls from its own, what it inherited, and the user's default.

Inside a ReLife-run crew these are in-process CrewAI ``BaseTool`` objects — same
contract as ``relife mcp --agent NAME``, minus a subprocess per agent. A CrewAI
app running *outside* ReLife attaches the identical tools over MCP instead
(``MCPServerStdio(command=python, args=["-m", "relife", "mcp", "--agent", NAME])``
or the memory daemon's ``/mcp``).
"""

from __future__ import annotations

from typing import Any, Optional

from crewai.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr, create_model

from ..memory.tools import EXTERNAL_TOOLS, ToolSpec
from .turns import run_sync

_JSON_TYPES: dict[str, Any] = {"string": str, "integer": int, "number": float, "boolean": bool}


def _args_model(spec: ToolSpec) -> type[BaseModel]:
    """A pydantic args schema for CrewAI, built from the spec's JSON schema."""
    props = spec.input_schema.get("properties", {})
    required = set(spec.input_schema.get("required", []))
    fields: dict[str, Any] = {}
    for name, p in props.items():
        typ = _JSON_TYPES.get(p.get("type", "string"), str)
        desc = p.get("description", "")
        if name in required:
            fields[name] = (typ, Field(..., description=desc))
        else:
            fields[name] = (Optional[typ], Field(None, description=desc))
    model_name = "".join(part.title() for part in spec.name.split("_")) + "Args"
    return create_model(model_name, **fields)


class RelifeMemoryTool(BaseTool):
    """One ReLife memory tool bound to one agent's scoped memory client."""

    _spec: ToolSpec = PrivateAttr()
    _client: Any = PrivateAttr()

    def _run(self, **kwargs: Any) -> str:
        args = {k: v for k, v in kwargs.items() if v is not None}
        text, is_error = run_sync(lambda: self._spec.handler(self._client, args))
        return f"ERROR: {text}" if is_error else text


def memory_tools(client: Any, specs: tuple[ToolSpec, ...] = EXTERNAL_TOOLS) -> list[BaseTool]:
    """The external memory tools as CrewAI tools over ``client`` (a scoped client)."""
    out: list[BaseTool] = []
    for spec in specs:
        tool = RelifeMemoryTool(
            name=spec.name, description=spec.description, args_schema=_args_model(spec)
        )
        tool._spec = spec
        tool._client = client
        out.append(tool)
    return out
