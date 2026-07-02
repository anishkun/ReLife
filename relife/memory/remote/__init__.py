"""Out-of-process transport for long-term memory (Phase 2 of the split).

This package is the *only* thing that changes when memory moves from an
in-process ``LocalMemoryClient`` to a standalone daemon: consumers still depend
on the ``MemoryClient`` protocol (``memory/client.py``), and the daemon wraps the
same ``MemoryService`` the in-process path uses.

- ``wire``        — DTO (de)serialization shared by both sides (no heavy deps).
- ``daemon``      — the FastAPI service core (needs the ``[daemon]`` extra).
- ``http_client`` — ``HttpMemoryClient`` talking to it over pooled httpx.

``wire`` is import-safe without the ``[daemon]`` extra; ``daemon`` and
``http_client`` import ``fastapi`` / ``httpx`` lazily so this package can be
imported for its wire format alone.
"""
