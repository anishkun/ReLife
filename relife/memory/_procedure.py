"""Bounds shared by the two procedural-memory file formats (skills, workflows).

Both are Markdown files with a ``---`` frontmatter header whose values come
from a model (``skill_write`` / ``workflow_save``, including external MCP
agents) or from an imported pack. A header value is written on one line, so a
newline in it could close the header early and turn the rest into the body, or
forge a ``name:`` line; and every procedure found is injected into later
prompts, so its size is bounded too. (Header ``---`` fences matter here too:
the parser ends the header at the first one.)
"""

from __future__ import annotations

import os
import re
import threading
from pathlib import Path
from typing import Any, Callable, TypeVar

T = TypeVar("T")

MAX_NAME_CHARS = 120
MAX_HEADER_CHARS = 500
MAX_BODY_CHARS = 20_000


def header_value(value: object, *, limit: int = MAX_HEADER_CHARS) -> str:
    """One header line: whitespace runs (newlines included) collapse to one
    space, and ``---`` (the header fence — the parser ends the header at the
    first one) shrinks to ``--``; then it is cut to ``limit``."""
    text = " ".join(str(value or "").split())
    text = re.sub(r"-{3,}", "--", text)
    return text[:limit]


def check_procedure(name: object, body: object) -> None:
    """Raise ``ValueError`` for a name or body a procedure file can't hold."""
    if not isinstance(name, str) or not isinstance(body, str):
        raise ValueError("name and steps must be text")
    if not name.strip() or not body.strip():
        raise ValueError("needs a name and steps")
    if len(header_value(name, limit=10**9)) > MAX_NAME_CHARS:
        raise ValueError(f"name is longer than {MAX_NAME_CHARS} characters")
    if len(body.strip()) > MAX_BODY_CHARS:
        raise ValueError(f"steps are longer than {MAX_BODY_CHARS} characters")


# --- parsed-file cache ------------------------------------------------------------
# Every prompt searches the skills and workflows of every space the agent reads,
# and opening a file is the whole cost (~10 ms each on Windows, measured: 200
# skills took 2.4 s per search). Parsed files are kept per directory, keyed by
# (mtime, size), so a search stats each file — cheap with ``os.scandir`` — and
# re-reads only what changed. Another process's write changes the stamp, so the
# daemon and a CLI on the same files still see each other's edits.
_CACHE: dict[str, dict[str, tuple[tuple[int, int], Any, frozenset[str], frozenset[str]]]] = {}
_CACHE_LOCK = threading.Lock()


def cached_dir(
    directory: Path,
    parse: Callable[[Path], T],
    index: Callable[[T], tuple[set[str], set[str]]],
) -> list[tuple[T, frozenset[str], frozenset[str]]]:
    """``(item, name_tokens, body_tokens)`` for every ``*.md`` in
    ``directory``, sorted by file name, parsing only files new or changed since
    the last call. Entries for files that are gone are dropped."""
    key = str(directory)
    try:
        entries = sorted(
            (e for e in os.scandir(directory) if e.name.endswith(".md") and e.is_file()),
            key=lambda e: e.name,
        )
    except FileNotFoundError:
        return []
    with _CACHE_LOCK:
        old = _CACHE.get(key, {})
    fresh: dict[str, tuple[tuple[int, int], Any, frozenset[str], frozenset[str]]] = {}
    out: list[tuple[T, frozenset[str], frozenset[str]]] = []
    for e in entries:
        try:
            st = e.stat()
        except OSError:
            continue  # removed between listing and stat
        sig = (st.st_mtime_ns, st.st_size)
        hit = old.get(e.name)
        if hit is None or hit[0] != sig:
            try:
                item = parse(Path(e.path))
            except OSError:
                continue
            name_tok, body_tok = index(item)
            hit = (sig, item, frozenset(name_tok), frozenset(body_tok))
        fresh[e.name] = hit
        out.append((hit[1], hit[2], hit[3]))
    with _CACHE_LOCK:
        _CACHE[key] = fresh
    return out


def clear_cache() -> None:
    with _CACHE_LOCK:
        _CACHE.clear()
