"""The always-on agent server: a long-lived process hosting persistent agent
sessions and serving the self-contained web UI.

Two modules:
- ``session`` — ``AgentSession`` (one long-lived ``ClaudeSDKClient`` per session),
  ``ApprovalBroker`` (UI-routed outward-action approvals), and ``SessionManager``.
- ``app`` — the FastAPI ``create_app`` factory + ``serve`` (needs the ``[server]``
  extra: ``pip install -e ".[server]"``).

``app`` imports fastapi/uvicorn lazily-at-use only via module import, so importing
this package requires the extra. The CLI (`relife serve`) guards the import and
prints an install hint if the extra is missing.
"""
