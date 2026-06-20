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
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .. import config
from ._text import tokenize as _tokens

_SKILLS_DIR = config.DATA_DIR / "skills"
_SLUG_OK = re.compile(r"[^a-z0-9]+")


def _dir():
    _SKILLS_DIR.mkdir(parents=True, exist_ok=True)
    return _SKILLS_DIR


def _slug(name: str) -> str:
    s = _SLUG_OK.sub("-", name.strip().lower()).strip("-")
    return s or "skill"


@dataclass
class Skill:
    name: str
    when_to_use: str
    body: str
    slug: str


def _parse(path) -> Skill:
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
    return Skill(name=name, when_to_use=when, body=body, slug=path.stem)


def write_skill(name: str, when_to_use: str, steps: str) -> str:
    """Create or overwrite a skill. Returns its slug."""
    if not name.strip() or not steps.strip():
        raise ValueError("skill needs a name and steps")
    slug = _slug(name)
    path = _dir() / f"{slug}.md"
    content = (
        f"---\nname: {name.strip()}\n"
        f"when_to_use: {when_to_use.strip()}\n---\n"
        f"{steps.strip()}\n"
    )
    path.write_text(content, encoding="utf-8")
    return slug


def list_skills() -> list[Skill]:
    return [_parse(p) for p in sorted(_dir().glob("*.md"))]


def find_skills(query: str, k: int = 3) -> list[Skill]:
    """Return skills relevant to ``query`` (keyword overlap, name weighted)."""
    q = _tokens(query)
    if not q:
        return []
    scored: list[tuple[int, Skill]] = []
    for sk in list_skills():
        name_tok = _tokens(sk.name + " " + sk.slug)
        body_tok = _tokens(sk.when_to_use + " " + sk.body)
        score = 2 * len(q & name_tok) + len(q & body_tok)
        if score:
            scored.append((score, sk))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [s for _, s in scored[:k]]


def read_skill(name: str) -> Skill | None:
    path = _dir() / f"{_slug(name)}.md"
    return _parse(path) if path.exists() else None


def count() -> int:
    return len(list(_dir().glob("*.md")))
