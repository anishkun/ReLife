"""Memory spaces — named partitions of memory, one per agent.

A *space* holds its own memories, skills, workflows and tool events. The user's
main ReLife agent lives in ``default`` (everything that existed before spaces
was introduced is there, untouched); every other agent registered in
``relife/agents.py`` writes only to a space of its own and *reads* a set of
spaces — its own, the ones it inherited from older agents, and ``default``.

That is how memory moves between agents without becoming one pool any agent
can poison: a new agent sees an old agent's knowledge live (read-only), forks a
snapshot copy, or receives an exported pack — and its own learnings reach shared
memory only through an explicit promote.

Pure and import-light (``config`` only), so the store, skills, workflows and
the agent registry can all depend on it without cycles.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .. import config

DEFAULT_SPACE = "default"

# Reassignable (tests redirect it; the memory daemon binds it next to its DB).
# The default space keeps the historical skills/workflows dirs, so no file moves.
_SPACES_DIR = config.DATA_DIR / "spaces"

_SPACE_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,47}$")


def validate_space(name: object) -> str:
    """Return ``name`` if it is a legal space name, else raise ``ValueError``.

    Space names become directory names, so the rule is strict: lowercase
    letters, digits, ``-`` and ``_``, at most 48 characters, no leading
    punctuation (no ``..``, no path separators, no drive letters).
    """
    if not isinstance(name, str) or not _SPACE_RE.match(name):
        raise ValueError(
            f"invalid space name {name!r}: use lowercase letters, digits, '-' or '_' "
            "(at most 48 characters)"
        )
    return name


def space_dir(kind: str, space: str, default_dir: Path) -> Path:
    """Directory holding ``kind`` (``skills``/``workflows``) files for ``space``."""
    if space == DEFAULT_SPACE:
        return default_dir
    return _SPACES_DIR / validate_space(space) / kind


@dataclass(frozen=True)
class MemoryScope:
    """What one agent may see and change.

    ``read`` is every space recall/find searches (own space first); ``write``
    is the single space saves, skills, workflows and events go to, and the only
    one ``forget``/``archive`` may touch. ``source`` is stamped on what it
    writes, so provenance survives a promote or an export.

    There is deliberately no second writable space: the memory tools never take
    a space argument (scope comes from identity, not from the model), so a save
    can only ever land in one place. Agents that should share writes are given
    the same ``write`` space instead.
    """

    read: tuple[str, ...] = (DEFAULT_SPACE,)
    write: str = DEFAULT_SPACE
    source: str = ""

    def __post_init__(self) -> None:
        validate_space(self.write)
        for s in self.read:
            validate_space(s)
        # An agent always sees what it writes, own space first; no repeats.
        ordered = dict.fromkeys((self.write, *self.read))
        object.__setattr__(self, "read", tuple(ordered))


DEFAULT_SCOPE = MemoryScope()
