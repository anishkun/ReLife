"""Procedural memory, level 2: workflows the agent assembles for itself.

A *skill* (``skills.py``) is a single reusable procedure. A *workflow* is a
higher-level, ordered chain of steps — often stitching several skills/actions
together — for a recurring multi-step job ("set up a new service: scaffold →
test → repo → push"). Workflows are what let ReLife notice that it keeps doing
the same sequence and capture it as one named, replayable plan.

Stored exactly like skills — one human-readable / diffable Markdown file per
workflow with a small frontmatter header — so the format stays consistent and
the consolidation pass can write them mechanically:

    ---
    name: ship-new-service
    when_to_use: Standing up and publishing a brand-new service.
    trigger: scaffold,test,repo,push
    ---
    1. ...ordered steps, may reference skills...

Recall is keyword + recency over name + when_to_use + trigger + body, with the
name weighted — same approach as skills. Workflows live per memory space with the
same layout and shadowing rule as skills (see ``skills.py``).
"""

from __future__ import annotations

import re
import shutil
from collections.abc import Sequence
from dataclasses import dataclass

from .. import config
from ._text import tokenize as _tokens
from ._procedure import cached_dir, check_procedure, header_value
from .spaces import DEFAULT_SPACE, space_dir

_WORKFLOWS_DIR = config.WORKFLOWS_DIR
_SLUG_OK = re.compile(r"[^a-z0-9]+")


def _dir(space: str = DEFAULT_SPACE):
    d = space_dir("workflows", space, _WORKFLOWS_DIR)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _slug(name: str) -> str:
    s = _SLUG_OK.sub("-", name.strip().lower()).strip("-")
    return s or "workflow"


@dataclass
class Workflow:
    name: str
    when_to_use: str
    trigger: str
    body: str
    slug: str
    space: str = DEFAULT_SPACE


def _parse(path, space: str = DEFAULT_SPACE) -> Workflow:
    text = path.read_text(encoding="utf-8")
    name, when, trigger, body = path.stem, "", "", text
    if text.startswith("---"):
        _, _, rest = text.partition("---")
        header, _, body = rest.partition("---")
        body = body.strip()
        for line in header.strip().splitlines():
            key, _, val = line.partition(":")
            key, val = key.strip().lower(), val.strip()
            if key == "name" and val:
                name = val
            elif key == "when_to_use":
                when = val
            elif key == "trigger":
                trigger = val
    return Workflow(
        name=name, when_to_use=when, trigger=trigger, body=body, slug=path.stem, space=space
    )


def write_workflow(
    name: str,
    when_to_use: str,
    steps: str,
    trigger: str = "",
    *,
    space: str = DEFAULT_SPACE,
) -> str:
    """Create or overwrite a workflow in ``space``. Returns its slug."""
    try:
        check_procedure(name, steps)
    except ValueError as e:
        raise ValueError(f"workflow {e}") from None
    name = header_value(name)
    slug = _slug(name)
    path = _dir(space) / f"{slug}.md"
    content = (
        f"---\nname: {name}\n"
        f"when_to_use: {header_value(when_to_use)}\n"
        f"trigger: {header_value(trigger)}\n---\n"
        f"{steps.strip()}\n"
    )
    path.write_text(content, encoding="utf-8")
    return slug


def _index(wf: Workflow) -> tuple[set[str], set[str]]:
    return _tokens(wf.name + " " + wf.slug), _tokens(wf.when_to_use + " " + wf.trigger + " " + wf.body)


def _indexed(space: str):
    # Parsed once per file version (see _procedure.cached_dir), not per search.
    return cached_dir(_dir(space), lambda p: _parse(p, space), _index)


def list_workflows(space: str = DEFAULT_SPACE) -> list[Workflow]:
    return [wf for wf, _, _ in _indexed(space)]


def find_workflows(
    query: str, k: int = 3, *, spaces: Sequence[str] | None = None
) -> list[Workflow]:
    """Return workflows relevant to ``query`` (keyword overlap, name weighted),
    searching ``spaces`` in order (``None`` = the default space); an earlier
    space's slug shadows a later one's."""
    q = _tokens(query)
    if not q:
        return []
    scored: list[tuple[int, Workflow]] = []
    seen: set[str] = set()
    for space in spaces if spaces is not None else (DEFAULT_SPACE,):
        for wf, name_tok, body_tok in _indexed(space):
            if wf.slug in seen:
                continue
            seen.add(wf.slug)
            score = 2 * len(q & name_tok) + len(q & body_tok)
            if score:
                scored.append((score, wf))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [w for _, w in scored[:k]]


def read_workflow(name: str, space: str = DEFAULT_SPACE) -> Workflow | None:
    path = _dir(space) / f"{_slug(name)}.md"
    return _parse(path, space) if path.exists() else None


def count(space: str = DEFAULT_SPACE) -> int:
    return len(list(_dir(space).glob("*.md")))


def copy_space(src: str, dst: str) -> int:
    """Copy every workflow of ``src`` that ``dst`` lacks into ``dst`` (fork)."""
    if src == dst:
        return 0
    target = _dir(dst)
    n = 0
    for p in sorted(_dir(src).glob("*.md")):
        if not (target / p.name).exists():
            shutil.copyfile(p, target / p.name)
            n += 1
    return n
