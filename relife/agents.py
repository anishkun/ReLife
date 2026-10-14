"""Agent registry — who the agents are, and which memory each one may touch.

ReLife started as one agent with one memory. As a platform it hosts many: full
ReLife (Claude) agents a CrewAI crew spins up, CrewAI agents on other LLMs, and
external agents (any MCP client) that attach ReLife memory. Each is a named
``AgentProfile`` here, and each profile resolves to a ``MemoryScope``:

- it **writes** only its own space (``space``, by default its own name);
- it **reads** its own space, the spaces it inherited from older agents (live,
  read-only), and the user's ``default`` space unless it is ``isolated``.

That is the handoff model: a new agent *inherits* an old one's knowledge without
being able to change or poison it, can *fork* a snapshot copy instead, and its
own learnings reach shared memory only through an explicit :func:`promote` by
the user. No agent may write ``default`` — that space is the main agent's, and
changes to it come only from the main agent or a deliberate promote/import.

**Memory mode.** An agent's memory is ``individual`` (its own space, above) or
``collective``: it writes a shared pool instead — the global ``collective``
space, or a named ``collective-<name>`` for one team. Every non-isolated agent
*reads* the global collective, so what a collective member learns reaches the
others without each keeping (and paying for) a memory of its own — the right
home for an agent on a free-tier model, or one whose work doesn't need a
private history. Saves into a collective are importance-capped
(``ScopedMemoryClient``) and stamped with the writer, and the user can still
promote the collective into ``default``.

External agents authenticate over HTTP MCP with a per-agent token; only its
sha256 is stored here, so the registry file is not a credential store.

Records live in one JSON file (``config.AGENTS_PATH``) with the
``ScheduleStore`` discipline: nothing written until the first change, atomic
tmp + ``os.replace`` writes, unreadable records dropped (never widened) and the
original kept aside before the first overwrite.
"""

from __future__ import annotations

import hashlib
import json
import secrets
import shutil
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from . import _atomic, config
from ._filelock import file_lock
from .memory.spaces import (
    COLLECTIVE_PREFIX,
    COLLECTIVE_SPACE,
    DEFAULT_SCOPE,
    DEFAULT_SPACE,
    MemoryScope,
    collective_space,
    is_collective,
    validate_space,
)

RUNTIMES = ("relife", "llm", "external")
MEMORY_MODES = ("individual", "collective")
MAX_INHERITS = 16
MAX_DESCRIPTION = 500
MAX_MODEL = 200
TOKEN_PREFIX = "rla_"

# Every change is read-modify-write of one file, and under `relife serve` the
# routes (on worker threads) and a crew's worker can change the registry at the
# same time: writes are serialized, and each re-reads the file first so one
# writer never drops another's agent. Across processes (the CLI while the
# server runs) the same section also holds an OS file lock (``_filelock``).
_WRITE_LOCK = threading.RLock()


def validate_agent_name(name: Any) -> str:
    """Agent names double as their default space name, so the same rule applies
    — plus ``default`` is reserved for the main agent."""
    name = validate_space(name)
    if name == DEFAULT_SPACE:
        raise ValueError("'default' is the main agent's space, not an agent name")
    return name


def validate_new_agent_name(name: Any) -> str:
    """A name for a *new* agent: also not a collective's space, or the agent's
    private space would be everyone's pool. (Only checked on create — an older
    record with such a name still loads.)"""
    name = validate_agent_name(name)
    if is_collective(name):
        raise ValueError(f"{name!r} names a collective memory space, not an agent")
    return name


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


@dataclass
class AgentProfile:
    name: str
    runtime: str = "relife"
    model: str = ""
    description: str = ""
    # Own (write) space; "" means the agent's name. Two agents given the same
    # space deliberately share one memory.
    space: str = ""
    # Spaces read live, read-only — what the agent inherited from older agents.
    inherits: list[str] = field(default_factory=list)
    # True: do not read the user's default space (e.g. an external agent whose
    # model provider shouldn't see your personal memory) — nor the collective.
    isolated: bool = False
    # individual: writes its own space · collective: writes the shared one
    # (``collective`` names which; "" = the global collective).
    memory: str = "individual"
    collective: str = ""
    parent: str | None = None
    created_at: float = field(default_factory=time.time)
    token_hash: str | None = None

    @property
    def own_space(self) -> str:
        if self.memory == "collective":
            return collective_space(self.collective)
        return self.space or self.name

    @property
    def memory_label(self) -> str:
        """``individual`` · ``collective`` · ``collective:<name>``."""
        if self.memory != "collective":
            return "individual"
        return "collective" + (f":{self.collective}" if self.collective else "")

    def scope(self) -> MemoryScope:
        read = [self.own_space, *self.inherits]
        if not self.isolated:
            read += [COLLECTIVE_SPACE, DEFAULT_SPACE]
        return MemoryScope(read=tuple(read), write=self.own_space, source=self.name)

    def public(self) -> dict[str, Any]:
        """The record minus the token hash (what CLIs and APIs show)."""
        d = asdict(self)
        d.pop("token_hash")
        d["own_space"] = self.own_space
        d["memory_label"] = self.memory_label
        d["has_token"] = self.token_hash is not None
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AgentProfile:
        """Build a profile from untrusted data (a hand-edited file, a request
        body), validating every field. Raises ``ValueError``/``TypeError``."""
        if not isinstance(d, dict):
            raise TypeError("agent record must be an object")
        name = validate_agent_name(d.get("name"))
        runtime = d.get("runtime", "relife")
        if runtime not in RUNTIMES:
            raise ValueError(f"runtime must be one of {', '.join(RUNTIMES)}")
        model = str(d.get("model") or "")
        if len(model) > MAX_MODEL:
            raise ValueError("model name too long")
        description = str(d.get("description") or "")[:MAX_DESCRIPTION]
        space = d.get("space") or ""
        if space:
            validate_space(space)
            if space == DEFAULT_SPACE:
                raise ValueError(
                    "an agent may not write the default space; promote its memories instead"
                )
        memory = d.get("memory") or "individual"
        if memory not in MEMORY_MODES:
            raise ValueError(f"memory must be one of {', '.join(MEMORY_MODES)}")
        collective = d.get("collective") or ""
        if not isinstance(collective, str):
            raise ValueError("collective must be a name")
        if memory == "collective":
            if space:
                raise ValueError("a collective agent writes the collective; it has no space of its own")
            # Stored as the bare name ("" = the global collective).
            collective = collective_space(collective)
            collective = "" if collective == COLLECTIVE_SPACE else collective[len(COLLECTIVE_PREFIX):]
        elif collective:
            raise ValueError("collective is only for memory: collective")
        inherits = d.get("inherits") or []
        if not isinstance(inherits, list) or len(inherits) > MAX_INHERITS:
            raise ValueError(f"inherits must be a list of at most {MAX_INHERITS} spaces")
        own = collective_space(collective) if memory == "collective" else (space or name)
        clean: list[str] = []
        for s in inherits:
            validate_space(s)
            if s not in (own, DEFAULT_SPACE) and s not in clean:
                clean.append(s)
        parent = d.get("parent")
        if parent is not None:
            parent = validate_agent_name(parent)
        token_hash = d.get("token_hash")
        if token_hash is not None and (
            not isinstance(token_hash, str)
            or len(token_hash) != 64
            or any(c not in "0123456789abcdef" for c in token_hash)
        ):
            raise ValueError("malformed token hash")
        created = d.get("created_at", time.time())
        if not isinstance(created, (int, float)):
            raise ValueError("created_at must be a number")
        return cls(
            name=name,
            runtime=runtime,
            model=model,
            description=description,
            space=space,
            inherits=clean,
            isolated=bool(d.get("isolated", False)),
            memory=memory,
            collective=collective,
            parent=parent,
            created_at=float(created),
            token_hash=token_hash,
        )


class AgentStore:
    """Every registered agent, persisted as one JSON file (see module doc)."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path is not None else config.AGENTS_PATH
        self._items: dict[str, AgentProfile] = {}
        self.problem: str | None = None
        self._load()

    def _load(self) -> None:
        with _WRITE_LOCK:  # never read a file this process is half-way replacing
            self._items, self.problem = self._read()

    def _read(self) -> tuple[dict[str, AgentProfile], str | None]:
        items: dict[str, AgentProfile] = {}
        if not self.path.exists():
            return items, None
        try:
            raw = json.loads(_atomic.read_text(self.path))
        except (OSError, ValueError) as e:
            return items, f"unreadable ({type(e).__name__}: {e})"
        records = raw.get("agents", []) if isinstance(raw, dict) else None
        if not isinstance(records, list):
            return items, "unexpected format (no agents list)"
        dropped = 0
        for d in records:
            try:
                a = AgentProfile.from_dict(d)
            except (TypeError, ValueError):
                dropped += 1
                continue
            items[a.name] = a
        return items, (f"{dropped} unreadable agent record(s) skipped" if dropped else None)

    def _refresh(self) -> None:
        """Pick up what other writers saved since this store was loaded. A file
        that reads as broken now is left to :meth:`save`'s backup path — the
        records this store already holds are kept, never thrown away."""
        items, problem = self._read()
        if problem is None:
            self._items = items
        elif self.problem is None:
            self.problem = problem  # so save() keeps the broken file aside first

    def _preserve_original(self) -> None:
        if self.problem is None or not self.path.exists():
            return
        stamp = time.strftime("%Y%m%d-%H%M%S")
        backup = self.path.with_name(f"{self.path.name}.corrupt-{stamp}")
        try:
            shutil.copy2(self.path, backup)
        except OSError:
            return
        self.problem = f"{self.problem} — original kept as {backup.name}"

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.problem and "original kept as" not in self.problem:
            self._preserve_original()
        payload = {"version": 1, "agents": [asdict(a) for a in self._items.values()]}
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        _atomic.replace(tmp, self.path)

    def list(self) -> list[AgentProfile]:
        return sorted(self._items.values(), key=lambda a: a.created_at)

    def get(self, name: str) -> AgentProfile | None:
        return self._items.get(name)

    def require(self, name: str) -> AgentProfile:
        a = self._items.get(name)
        if a is None:
            raise LookupError(f"no agent named {name!r} (see `relife agent list`)")
        return a

    def put(self, profile: AgentProfile, *, new: bool = False) -> AgentProfile:
        """Save one agent. ``new``: refuse if the name was taken meanwhile (two
        creates racing must not have the second silently replace the first)."""
        # Round-trip through from_dict so a profile built in code obeys the
        # same rules as one read from disk.
        profile = AgentProfile.from_dict(asdict(profile))
        with _WRITE_LOCK, file_lock(self.path):
            self._refresh()
            if new and profile.name in self._items:
                raise ValueError(f"an agent named {profile.name!r} already exists")
            self._items[profile.name] = profile
            self.save()
        return profile

    def remove(self, name: str) -> bool:
        with _WRITE_LOCK, file_lock(self.path):
            self._refresh()
            if self._items.pop(name, None) is None:
                return False
            self.save()
        return True

    def by_token(self, token: str | None) -> AgentProfile | None:
        """The agent a presented token belongs to (constant-time compare)."""
        if not token:
            return None
        presented = _hash_token(token)
        found = None
        for a in self._items.values():
            if a.token_hash is not None and secrets.compare_digest(a.token_hash, presented):
                found = a
        return found


# --- scope resolution --------------------------------------------------------
def scope_for(agent: str | None, store: AgentStore | None = None) -> MemoryScope:
    """The memory scope for ``agent`` (``None`` = the main agent's default
    scope). An unknown name raises ``LookupError`` — never a silent fallback to
    the default space's read/write access."""
    if agent is None:
        return DEFAULT_SCOPE
    store = store or AgentStore()
    return store.require(agent).scope()


# --- lifecycle + handoff ------------------------------------------------------
def create_agent(
    store: AgentStore,
    client: Any,
    name: str,
    *,
    runtime: str = "relife",
    model: str = "",
    description: str = "",
    inherit: list[str] | tuple[str, ...] = (),
    fork: str | None = None,
    space: str | None = None,
    isolated: bool = False,
    memory: str = "individual",
    collective: str = "",
) -> tuple[AgentProfile, dict[str, int] | None]:
    """Register a new agent, handing it memory from older agents.

    ``inherit`` (agent names): read their memory live, read-only — including
    what *they* inherited, so knowledge flows down a lineage. ``fork`` (an
    agent name): start from a snapshot copy of its own memories, skills and
    workflows, and read what it inherited. ``memory="collective"`` makes it
    write the shared collective (``collective`` names a team's; "" = the
    global one) instead of a space of its own — it can't fork then: a private
    snapshot would be copied into everyone's pool. Returns the profile and, for
    a fork, what was copied.
    """
    name = validate_new_agent_name(name)
    if memory == "collective" and fork:
        raise ValueError("a collective agent can't fork: it would copy a private memory into the shared pool")
    if store.get(name) is not None:
        raise ValueError(f"an agent named {name!r} already exists")
    inherits: list[str] = []
    for parent_name in inherit:
        p = store.require(parent_name)
        inherits += [p.own_space, *p.inherits]
    fork_profile = store.require(fork) if fork else None
    if fork_profile is not None:
        inherits += fork_profile.inherits
    profile = store.put(
        AgentProfile(
            name=name,
            runtime=runtime,
            model=model,
            description=description,
            space=space or "",
            inherits=inherits,
            isolated=isolated,
            parent=fork or (inherit[0] if inherit else None),
            memory=memory,
            collective=collective,
        ),
        new=True,
    )
    copied = None
    if fork_profile is not None:
        copied = client.copy_space(fork_profile.own_space, profile.own_space)
    return profile, copied


def attach(store: AgentStore, name: str, other: str) -> AgentProfile:
    """Let ``name`` read ``other``'s memory (an agent name or a space), read-only."""
    a = store.require(name)
    target = store.get(other)
    space = target.own_space if target is not None else validate_space(other)
    if space in (a.own_space, DEFAULT_SPACE) or space in a.inherits:
        return a
    a.inherits = [*a.inherits, space]
    return store.put(a)


def set_memory(store: AgentStore, name: str, memory: str, collective: str = "") -> AgentProfile:
    """Move an agent's memory home (individual ⇄ collective, or to another
    collective). Future writes go to the new home; the old one is attached
    read-only, so nothing the agent learned stops being visible to it."""
    a = store.require(name)
    old = a.own_space
    a.memory = memory
    a.collective = collective if memory == "collective" else ""
    if memory == "collective":
        a.space = ""
    new = AgentProfile.from_dict(asdict(a))  # validates before anything is saved
    if old != new.own_space and old not in new.inherits and old != DEFAULT_SPACE:
        new.inherits = [*new.inherits, old]
    return store.put(new)


def detach(store: AgentStore, name: str, other: str) -> AgentProfile:
    a = store.require(name)
    target = store.get(other)
    space = target.own_space if target is not None else other
    a.inherits = [s for s in a.inherits if s != space]
    return store.put(a)


def promote(
    store: AgentStore,
    client: Any,
    name: str,
    *,
    to: str = DEFAULT_SPACE,
    ids: list[int] | None = None,
) -> dict[str, int]:
    """Copy an agent's own memories (all, or ``ids``) into ``to`` — the user's
    explicit act of trusting what the agent learned. Provenance is kept, so a
    promoted memory still says which agent wrote it."""
    a = store.require(name)
    validate_space(to)
    return client.copy_space(a.own_space, to, ids=ids)


def delete_agent(
    store: AgentStore, client: Any, name: str, *, keep_memory: bool = False
) -> int:
    """Unregister an agent. Its space is archived (reversibly) unless
    ``keep_memory`` or another agent still writes to that space. Returns the
    number of memories archived."""
    a = store.require(name)
    store.remove(name)
    shared = any(o.own_space == a.own_space for o in store.list())
    # A collective is everyone's: deleting one member never archives it.
    if keep_memory or shared or is_collective(a.own_space):
        return 0
    return client.archive_space(a.own_space)


def issue_token(store: AgentStore, name: str) -> str:
    """Mint (or rotate) the agent's HTTP MCP token. Returned once; only its
    hash is stored, so a lost token is replaced, not recovered."""
    a = store.require(name)
    token = TOKEN_PREFIX + secrets.token_urlsafe(32)
    a.token_hash = _hash_token(token)
    store.put(a)
    return token


def revoke_token(store: AgentStore, name: str) -> None:
    a = store.require(name)
    a.token_hash = None
    store.put(a)


def mcp_config(
    name: str,
    *,
    python: str,
    home: str,
    http_url: str | None = None,
    token: str | None = None,
) -> dict[str, Any]:
    """Paste-ready MCP client config attaching ReLife memory as agent ``name``.

    ``stdio`` launches ``python -m relife mcp --agent NAME`` (the local user's
    own process; no token). ``http`` targets the memory daemon's ``/mcp`` with
    the agent's bearer token — only when one is given, since it can't be shown
    again after it was issued. ``RELIFE_HOME`` pins the data dir so a client
    that launches from another directory still reaches the same memory.
    """
    out: dict[str, Any] = {
        "stdio": {
            "mcpServers": {
                "relife-memory": {
                    "command": python,
                    "args": ["-m", "relife", "mcp", "--agent", name],
                    "env": {"RELIFE_HOME": home},
                }
            }
        }
    }
    if http_url and token:
        out["http"] = {
            "mcpServers": {
                "relife-memory": {
                    "type": "http",
                    "url": http_url,
                    "headers": {"Authorization": f"Bearer {token}"},
                }
            }
        }
    return out
