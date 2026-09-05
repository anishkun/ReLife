"""Pure security helpers for the agent server.

Deliberately I/O-free and framework-free (same discipline as
``memory/cognitive.py``): every decision the hardened server makes about *who*
may talk to it and *where* a session may write is a plain function here, so the
policy is unit-testable without HTTP, a socket, or a model.
"""

from __future__ import annotations

import ipaddress
import secrets
import time
from pathlib import Path
from urllib.parse import urlsplit

from .. import config


# --- token ------------------------------------------------------------------
def token_matches(expected: str | None, presented: str | None) -> bool:
    """Constant-time token comparison. ``expected is None`` = auth disabled."""
    if expected is None:
        return True
    if not presented:
        return False
    return secrets.compare_digest(expected, presented)


def presented_token(authorization: str | None, cookie: str | None) -> str | None:
    """Extract the caller's token from either accepted carrier.

    ``Authorization: Bearer …`` is for programmatic clients; the cookie exists
    because a browser ``EventSource`` cannot set request headers — without it the
    SSE stream (which carries the whole transcript) would be the one route the UI
    could never authenticate.
    """
    if authorization:
        scheme, _, value = authorization.partition(" ")
        if scheme.lower() == "bearer" and value.strip():
            return value.strip()
    return cookie or None


# --- CSRF -------------------------------------------------------------------
def same_origin(origin: str | None, host: str | None) -> bool:
    """Whether a mutating request's ``Origin`` matches the served ``Host``.

    Cookie auth means the browser attaches credentials automatically, so a
    cross-site page could otherwise drive the agent. ``SameSite=Strict`` is the
    primary defence; this is the belt to that suspenders. A missing ``Origin``
    (curl, an SDK client) is allowed — those carry a bearer header, not a cookie
    a third-party site could ride on.
    """
    if not origin or origin == "null":
        return True
    if not host:
        return False
    return urlsplit(origin).netloc.lower() == host.lower()


# --- bind address -----------------------------------------------------------
def is_loopback(host: str) -> bool:
    h = (host or "").strip().strip("[]").lower()
    if h in {"localhost", "localhost.localdomain"}:
        return True
    try:
        return ipaddress.ip_address(h).is_loopback
    except ValueError:
        return False


def guard_bind(host: str, token: str | None) -> None:
    """Fail closed: never expose an unauthenticated agent beyond loopback.

    The agent server can run shell commands and edit files. Binding it to a
    reachable interface with no token would hand that to the network, so this
    refuses rather than warns.
    """
    if is_loopback(host) or token:
        return
    raise ValueError(
        f"refusing to bind {host} without authentication: the agent server can run "
        "shell commands and edit files. Set RELIFE_AGENT_TOKEN, or bind 127.0.0.1.\n"
        "Note: there is still no TLS — put a reverse proxy in front before exposing it."
    )


# --- workspace confinement --------------------------------------------------
def resolve_workspace(raw: str | None, root: Path | None = None) -> Path:
    """Resolve a session workspace requested over HTTP, confined to ``root``.

    The permission policy auto-allows writes *inside a session's workspace*, so an
    unconstrained path in the request body would let the caller pick how much of
    the disk is auto-writable. Relative paths are taken as relative to the root;
    absolute paths must land inside it. Resolution happens before the check, so
    ``..`` and symlinked escapes are caught.
    """
    base = Path(root if root is not None else config.AGENT_WORKSPACE_ROOT).expanduser().resolve()
    if not raw or not str(raw).strip():
        return base
    candidate = Path(str(raw).strip()).expanduser()
    candidate = (base / candidate) if not candidate.is_absolute() else candidate
    resolved = candidate.resolve()
    if resolved != base and base not in resolved.parents:
        raise ValueError(f"workspace must be inside {base}")
    return resolved


# --- throttling -------------------------------------------------------------
class AttemptLimiter:
    """Fixed-window per-key attempt counter (used to throttle token guessing)."""

    def __init__(self, max_attempts: int, window: float) -> None:
        self.max_attempts = max_attempts
        self.window = window
        self._hits: dict[str, list[float]] = {}

    def allow(self, key: str, now: float | None = None) -> bool:
        """Record an attempt for ``key``; False once the window is exhausted."""
        t = time.monotonic() if now is None else now
        hits = [h for h in self._hits.get(key, []) if t - h < self.window]
        if len(hits) >= self.max_attempts:
            self._hits[key] = hits
            return False
        hits.append(t)
        self._hits[key] = hits
        return True

    def reset(self, key: str) -> None:
        self._hits.pop(key, None)
