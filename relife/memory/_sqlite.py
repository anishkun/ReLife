"""SQLite connection settings shared by the memory store and the event log
(one file, ``data/relife.db``)."""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path


def fast_writes(conn: sqlite3.Connection) -> None:
    """WAL + ``synchronous=NORMAL`` for every connection to a memory DB.

    The default rollback journal with full sync creates, flushes and deletes a
    journal file on every commit: ~50 ms per save on Windows (measured), and a
    recall that reinforces 5 hits commits 5 times — on every prompt. In WAL a
    commit is an append (readers no longer wait for a writer, either), and
    NORMAL stays crash-safe for the application; at worst a power cut loses
    the last few commits, which for memory means a reinforcement or a save.
    WAL is persistent per DB file, so setting it again is a no-op."""
    try:
        conn.execute("PRAGMA journal_mode = WAL")
    except sqlite3.OperationalError:  # pragma: no cover - e.g. a read-only medium
        pass
    conn.execute("PRAGMA synchronous = NORMAL")


class Connection(sqlite3.Connection):
    """A memory-DB connection that can carry per-connection state (which
    extensions are loaded — plain ``sqlite3.Connection`` has no ``__dict__``)."""

    loaded: set[str]


_local = threading.local()


def connect(path: Path) -> Connection:
    """This thread's open connection to ``path`` (one per thread per file).

    The store and the event log used to open a fresh connection per call and
    drop it, and the open/close itself was the cost: in WAL mode the last
    connection to close checkpoints and deletes the WAL file, so each
    ``log_event`` paid for that (~17 ms each in the suite, measured); under
    the old rollback journal each open was a full sync. sqlite3 connections
    are per thread, so the cache is too — the server's worker threads each
    keep their own. Callers keep using ``with conn:`` (commit / rollback);
    nothing closes these. A file deleted underneath (POSIX allows it) gets a
    fresh connection rather than writes into an unlinked inode.
    """
    cache: dict[str, Connection] | None = getattr(_local, "conns", None)
    if cache is None:
        cache = _local.conns = {}
    key = str(path)
    conn = cache.get(key)
    if conn is not None:
        if path.exists():
            return conn
        conn.close()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, factory=Connection)
    conn.loaded = set()
    conn.row_factory = sqlite3.Row
    # Wait briefly instead of erroring instantly on SQLITE_BUSY (another
    # process — the daemon, a CLI — holding the write lock for a moment).
    conn.execute("PRAGMA busy_timeout = 5000")
    fast_writes(conn)
    cache[key] = conn
    return conn


def close_all() -> None:
    """Close this thread's cached connections (tests that remove DB files)."""
    for conn in (getattr(_local, "conns", None) or {}).values():
        conn.close()
    _local.conns = {}
