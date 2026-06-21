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
name weighted — same approach as skills.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .. import config
from ._text import tokenize as _tokens

_WORKFLOWS_DIR = config.WORKFLOWS_DIR
_SLUG_OK = re.compile(r"[^a-z0-9]+")


def _dir():
    _WORKFLOWS_DIR.mkdir(parents=True, exist_ok=True)
    return _WORKFLOWS_DIR


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


def _parse(path) -> Workflow:
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
    return Workflow(name=name, when_to_use=when, trigger=trigger, body=body, slug=path.stem)


def write_workflow(name: str, when_to_use: str, steps: str, trigger: str = "") -> str:
    """Create or overwrite a workflow. Returns its slug."""
    if not name.strip() or not steps.strip():
        raise ValueError("workflow needs a name and steps")
    slug = _slug(name)
    path = _dir() / f"{slug}.md"
    content = (
        f"---\nname: {name.strip()}\n"
        f"when_to_use: {when_to_use.strip()}\n"
        f"trigger: {trigger.strip()}\n---\n"
        f"{steps.strip()}\n"
    )
    path.write_text(content, encoding="utf-8")
    return slug


def list_workflows() -> list[Workflow]:
    return [_parse(p) for p in sorted(_dir().glob("*.md"))]


def find_workflows(query: str, k: int = 3) -> list[Workflow]:
    """Return workflows relevant to ``query`` (keyword overlap, name weighted)."""
    q = _tokens(query)
    if not q:
        return []
    scored: list[tuple[int, Workflow]] = []
    for wf in list_workflows():
        name_tok = _tokens(wf.name + " " + wf.slug)
        body_tok = _tokens(wf.when_to_use + " " + wf.trigger + " " + wf.body)
        score = 2 * len(q & name_tok) + len(q & body_tok)
        if score:
            scored.append((score, wf))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [w for _, w in scored[:k]]


def read_workflow(name: str) -> Workflow | None:
    path = _dir() / f"{_slug(name)}.md"
    return _parse(path) if path.exists() else None


def count() -> int:
    return len(list(_dir().glob("*.md")))
