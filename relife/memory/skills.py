"""Procedural memory (layer B): reusable skills the agent writes for itself.

A *skill* is a named procedure ("how I scaffold a Python CLI", "how I push to
git here") the agent records after succeeding, then reuses later. Stored as
human-readable / diffable Markdown files, one per skill, with a small frontmatter
header:

    ---
    name: scaffold-python-cli
    when_to_use: Setting up a new Python command-line project.
    ---
    1. ...steps...

Recall is keyword + recency over name + when_to_use + body — same approach as the
fact store. This is what makes ReLife get better at *doing* things, not just
remembering facts.

Skills live per memory space (see ``spaces.py``): the default space keeps the
historical ``data/skills/`` dir, every other space has its own. A search over
several spaces lets the first (the agent's own) shadow a same-named skill it
inherited, so an agent can refine a procedure without editing its parent's.
"""

from __future__ import annotations

import re
import shutil
from collections.abc import Sequence
from dataclasses import dataclass

from .. import config
from ._text import tokenize as _tokens
from .spaces import DEFAULT_SPACE, space_dir

_SKILLS_DIR = config.SKILLS_DIR
_SLUG_OK = re.compile(r"[^a-z0-9]+")


def _dir(space: str = DEFAULT_SPACE):
    d = space_dir("skills", space, _SKILLS_DIR)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _slug(name: str) -> str:
    s = _SLUG_OK.sub("-", name.strip().lower()).strip("-")
    return s or "skill"


@dataclass
class Skill:
    name: str
    when_to_use: str
    body: str
    slug: str
    space: str = DEFAULT_SPACE


def _parse(path, space: str = DEFAULT_SPACE) -> Skill:
    text = path.read_text(encoding="utf-8")
    name, when, body = path.stem, "", text
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
    return Skill(name=name, when_to_use=when, body=body, slug=path.stem, space=space)


def write_skill(
    name: str, when_to_use: str, steps: str, *, space: str = DEFAULT_SPACE
) -> str:
    """Create or overwrite a skill in ``space``. Returns its slug."""
    if not name.strip() or not steps.strip():
        raise ValueError("skill needs a name and steps")
    slug = _slug(name)
    path = _dir(space) / f"{slug}.md"
    content = (
        f"---\nname: {name.strip()}\n"
        f"when_to_use: {when_to_use.strip()}\n---\n"
        f"{steps.strip()}\n"
    )
    path.write_text(content, encoding="utf-8")
    return slug


def list_skills(space: str = DEFAULT_SPACE) -> list[Skill]:
    return [_parse(p, space) for p in sorted(_dir(space).glob("*.md"))]


def find_skills(
    query: str, k: int = 3, *, spaces: Sequence[str] | None = None
) -> list[Skill]:
    """Return skills relevant to ``query`` (keyword overlap, name weighted),
    searching ``spaces`` in order (``None`` = the default space). A slug found
    in an earlier space shadows the same slug in a later one."""
    q = _tokens(query)
    if not q:
        return []
    scored: list[tuple[int, Skill]] = []
    seen: set[str] = set()
    for space in spaces if spaces is not None else (DEFAULT_SPACE,):
        for sk in list_skills(space):
            if sk.slug in seen:
                continue
            seen.add(sk.slug)
            name_tok = _tokens(sk.name + " " + sk.slug)
            body_tok = _tokens(sk.when_to_use + " " + sk.body)
            score = 2 * len(q & name_tok) + len(q & body_tok)
            if score:
                scored.append((score, sk))
    # Stable sort: at equal score the earlier (own) space wins.
    scored.sort(key=lambda x: x[0], reverse=True)
    return [s for _, s in scored[:k]]


def read_skill(name: str, space: str = DEFAULT_SPACE) -> Skill | None:
    path = _dir(space) / f"{_slug(name)}.md"
    return _parse(path, space) if path.exists() else None


def count(space: str = DEFAULT_SPACE) -> int:
    return len(list(_dir(space).glob("*.md")))


def copy_space(src: str, dst: str) -> int:
    """Copy every skill of ``src`` into ``dst`` that ``dst`` doesn't already
    have (the fork primitive — never overwrites the destination's own)."""
    if src == dst:
        return 0
    target = _dir(dst)
    n = 0
    for p in sorted(_dir(src).glob("*.md")):
        if not (target / p.name).exists():
            shutil.copyfile(p, target / p.name)
            n += 1
    return n
