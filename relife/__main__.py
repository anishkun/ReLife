"""``python -m relife …`` — the same CLI as the ``relife`` console script.

MCP clients (and CrewAI's ``MCPServerStdio``) launch ``python -m relife mcp
--agent NAME`` with an explicit interpreter, which works even where the
``relife`` script isn't on PATH (a venv that isn't activated, a GUI app's
minimal environment).
"""

from .cli import main

if __name__ == "__main__":
    main()
