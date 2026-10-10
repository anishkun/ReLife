"""Memory spaces: one partition per agent, isolated where it matters.

Covers the v3 schema migration (incl. the fresh-DB stamp), per-space save
dedupe and recall, consolidation never crossing a space boundary, per-space
skills/workflows with own-space shadowing, and the copy / archive / export /
import primitives the agent handoff is built on. Deterministic (embeddings off).
"""

from __future__ import annotations

import json
import sqlite3
import time

import pytest

from relife.memory import consolidate as consol
from relife.memory import events as ev
from relife.memory import skills as sk
from relife.memory import spaces
from relife.memory import store as store_mod
from relife.memory import workflows as wf
from relife.memory.service import MemoryService, validate_pack


@pytest.fixture
def iso(tmp_path, monkeypatch):
    db = tmp_path / "relife.db"
    monkeypatch.setattr(store_mod, "_DB_PATH", db)
    monkeypatch.setattr(ev, "_DB_PATH", db)
    monkeypatch.setattr(consol, "_STATE_PATH", tmp_path / "consolidate_state.json")
    monkeypatch.setattr(sk, "_SKILLS_DIR", tmp_path / "skills")
    monkeypatch.setattr(wf, "_WORKFLOWS_DIR", tmp_path / "workflows")
    monkeypatch.setattr(spaces, "_SPACES_DIR", tmp_path / "spaces")
    return tmp_path


# --- schema -----------------------------------------------------------------
def test_fresh_db_is_v3_with_space_columns(tmp_path):
    db = tmp_path / "fresh.db"
    store_mod.MemoryStore(db).init_db()
    with sqlite3.connect(db) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 3
        cols = {r[1] for r in conn.execute("PRAGMA table_info(memories)")}
    assert {"space", "source"} <= cols


def test_v2_store_upgrades_in_place(tmp_path):
    """A store stamped v2 before spaces existed gets the columns, keeps every
    row (in the default space), and still recalls."""
    db = tmp_path / "v2.db"
    with sqlite3.connect(db) as conn:
        conn.execute(
            "CREATE TABLE memories (id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT NOT NULL, "
            "text TEXT NOT NULL, tags TEXT NOT NULL DEFAULT '', importance REAL NOT NULL DEFAULT 0.5, "
            "created_at REAL NOT NULL, last_used_at REAL NOT NULL DEFAULT 0, "
            "use_count INTEGER NOT NULL DEFAULT 0, status TEXT NOT NULL DEFAULT 'active', embedding BLOB)"
        )
        now = time.time()
        conn.execute(
            "INSERT INTO memories (kind, text, created_at, last_used_at) VALUES "
            "('fact', 'deploys go through the staging cluster', ?, ?)",
            (now, now),
        )
        conn.execute("PRAGMA user_version = 2")
    s = store_mod.MemoryStore(db)
    s.init_db()
    with sqlite3.connect(db) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 3
    [m] = s.recall("staging cluster deploys")
    assert m.space == "default" and m.source == ""


def test_space_names_are_validated(iso):
    for bad in ("../evil", "Upper", "", "a/b", "-lead", "x" * 49, "c:", None):
        with pytest.raises(ValueError):
            spaces.validate_space(bad)
    with pytest.raises(ValueError):
        store_mod.save("x", space="../evil")
    with pytest.raises(ValueError):
        sk.write_skill("n", "w", "s", space="../evil")


def test_scope_reads_its_write_space_first():
    sc = spaces.MemoryScope(read=("default", "alpha"), write="beta")
    assert sc.read == ("beta", "default", "alpha")
    assert spaces.MemoryScope(read=("a", "a", "b"), write="a").read == ("a", "b")


# --- save / recall ------------------------------------------------------------
def test_dedupe_is_per_space(iso):
    a = store_mod.save("ruff is the linter here", space="default")
    b = store_mod.save("ruff is the linter here", space="alpha")
    assert a != b  # same sentence, two agents: two memories
    assert store_mod.save("ruff is the linter here", space="alpha") == b  # reinforced
    assert store_mod.get(b).use_count == 1
    assert store_mod.get(a).use_count == 0  # alpha's save never strengthened default's


def test_recall_respects_spaces(iso):
    svc = MemoryService()
    svc.save("deploys go through the staging cluster", space="alpha", source="alpha")
    svc.save("the user prefers tabs over spaces", space="default")
    # Agent-facing default = the main agent's space only.
    assert svc.recall("staging cluster deploys") == []
    [m] = svc.recall("staging cluster deploys", spaces=["alpha"])
    assert (m.space, m.source) == ("alpha", "alpha")
    both = svc.recall("staging cluster deploys tabs prefers", spaces=["alpha", "default"])
    assert {m.space for m in both} == {"alpha", "default"}
    # An empty scope reads nothing (never "everything").
    assert store_mod.recall("staging cluster deploys", spaces=[]) == []


def test_forget_and_archive_stay_in_their_space(iso):
    svc = MemoryService()
    mid = svc.save("the deploy key rotates monthly", space="alpha")
    assert svc.forget("deploy key rotates", space="default") is None
    assert svc.archive(mid, space="beta") is False
    assert svc.get(mid).status == "active"
    assert svc.archive(mid, space="alpha") is True


# --- consolidation --------------------------------------------------------------
_BASE = "alpha bravo charlie delta echo foxtrot golf hotel india juliet"


def test_dedupe_never_merges_across_spaces(iso):
    # Token Jaccard 10/11 ≥ 0.9: duplicates — but in different spaces.
    store_mod.save(_BASE, space="default")
    store_mod.save(_BASE + " kilo", space="team")
    report = consol.run_consolidation()
    assert report.merged == 0
    assert store_mod.count(include_archived=False) == 2


def test_dedupe_still_merges_within_a_space(iso):
    store_mod.save(_BASE, space="team")
    store_mod.save(_BASE + " kilo", space="team")
    assert consol.run_consolidation().merged == 1
    assert store_mod.count(include_archived=False, spaces=["team"]) == 1


def test_recurring_sequence_is_learned_into_its_own_space(iso):
    for task in ("t1", "t2", "t3"):
        for cmd in ("git clone repo", "pytest -q", "git push"):
            ev.log_event("Bash", cmd, task_id=task, space="alpha")
    report = consol.run_consolidation()
    name = "auto-git-clone-test-git-push"
    assert wf.read_workflow(name, "alpha") is not None
    assert wf.read_workflow(name) is None  # not the main agent's
    assert f"[alpha] {name}" in report.workflows_created
    pats = store_mod.all_memories(include_archived=False, spaces=["alpha"])
    assert any(m.kind == "pattern" for m in pats)
    assert not any(m.kind == "pattern" for m in store_mod.all_memories(spaces=["default"]))


def test_sequences_from_two_spaces_do_not_combine(iso):
    """Two runs in alpha + one in beta: neither space reaches the threshold."""
    for task, space in (("t1", "alpha"), ("t2", "alpha"), ("t3", "beta")):
        for cmd in ("git clone repo", "pytest -q", "git push"):
            ev.log_event("Bash", cmd, task_id=task, space=space)
    assert consol.run_consolidation().workflows_created == []


# --- skills / workflows -----------------------------------------------------------
def test_own_space_skill_shadows_inherited(iso):
    sk.write_skill("deploy-service", "ship it", "1. old way")
    sk.write_skill("deploy-service", "ship it", "1. the refined way", space="alpha")
    hits = sk.find_skills("deploy service", spaces=["alpha", "default"])
    assert [(h.space, h.body) for h in hits] == [("alpha", "1. the refined way")]
    assert sk.find_skills("deploy service")[0].body == "1. old way"  # default unchanged
    assert sk.count() == 1 and sk.count("alpha") == 1


# --- copy / archive / export / import -----------------------------------------------
def test_copy_space_keeps_strength_and_provenance(iso):
    svc = MemoryService()
    mid = svc.save("staging deploys need a feature flag", importance=0.9, space="alpha", source="alpha")
    store_mod.reinforce(mid)
    store_mod.reinforce(mid)
    sk.write_skill("flag-deploy", "deploy behind a flag", "1. flag", space="alpha")
    out = svc.copy_space("alpha", "beta")
    assert out == {"memories": 1, "skills": 1, "workflows": 0}
    [copy] = svc.all_memories(spaces=["beta"])
    assert (copy.importance, copy.use_count, copy.source) == (0.9, 2, "alpha")
    # Copying again reinforces rather than duplicates.
    assert svc.copy_space("alpha", "beta")["memories"] == 1
    assert svc.count(spaces=["beta"]) == 1
    assert svc.all_memories(spaces=["beta"])[0].use_count == 3


def test_copy_space_by_ids_skips_procedural(iso):
    svc = MemoryService()
    keep = svc.save("keep this lesson about retries", space="alpha")
    svc.save("not this one about caching", space="alpha")
    sk.write_skill("retry", "retries", "1. back off", space="alpha")
    out = svc.copy_space("alpha", "default", ids=[keep])
    assert out == {"memories": 1, "skills": 0, "workflows": 0}
    assert [m.text for m in svc.all_memories(spaces=["default"])] == ["keep this lesson about retries"]


def test_archive_space(iso):
    svc = MemoryService()
    svc.save("one fact about alpha", space="alpha")
    svc.save("another fact about alpha", space="alpha")
    svc.save("a default fact", space="default")
    assert svc.archive_space("alpha") == 2
    assert svc.count(include_archived=False) == 1


def test_export_import_round_trip(iso):
    svc = MemoryService()
    svc.save("the CI matrix covers py3.11 to py3.13", kind="fact", tags="ci", space="alpha")
    svc.save("prefers small PRs", kind="preference", space="alpha")
    sk.write_skill("cut-release", "make a release", "1. tag\n2. push", space="alpha")
    wf.write_workflow("ship", "ship a change", "1. test\n2. pr", trigger="ship", space="alpha")
    pack = json.loads(json.dumps(svc.export_space("alpha")))  # survives JSON
    assert pack["format"] == "relife-memory-pack" and pack["space"] == "alpha"

    out = svc.import_pack(pack, "gamma")
    assert out == {"memories": 2, "skills": 1, "workflows": 1}
    got = {m.text: m for m in svc.all_memories(spaces=["gamma"])}
    assert got["prefers small PRs"].kind == "preference"
    assert {m.source for m in got.values()} == {"import:alpha"}
    assert sk.read_skill("cut-release", "gamma").body == "1. tag\n2. push"
    # Importing twice reinforces, never duplicates, and keeps existing skills.
    assert svc.import_pack(pack, "gamma")["skills"] == 0
    assert svc.count(spaces=["gamma"]) == 2


@pytest.mark.parametrize(
    "pack",
    [
        None,
        {"format": "something-else", "version": 1},
        {"format": "relife-memory-pack", "version": 99},
        {"format": "relife-memory-pack", "version": 1, "memories": "nope"},
        {"format": "relife-memory-pack", "version": 1, "memories": [{"text": "  "}]},
        {"format": "relife-memory-pack", "version": 1, "skills": [{"name": "x"}]},
    ],
)
def test_bad_packs_are_rejected_before_any_write(iso, pack):
    svc = MemoryService()
    with pytest.raises(ValueError):
        validate_pack(pack)
    bad = dict(pack) if isinstance(pack, dict) else pack
    if isinstance(bad, dict):
        bad.setdefault("memories", [{"text": "would have been written"}])
    with pytest.raises(ValueError):
        svc.import_pack(bad, "gamma")
    assert svc.count(spaces=["gamma"]) == 0


def test_spaces_listing(iso):
    svc = MemoryService()
    svc.save("a fact in alpha", space="alpha")
    sk.write_skill("only-skill", "x", "y", space="beta")
    listing = svc.spaces()
    assert listing["default"]["memories"] == 0
    assert listing["alpha"]["memories"] == 1
    assert listing["beta"] == {"memories": 0, "skills": 1, "workflows": 0}
    (iso / "spaces" / "Not A Space").mkdir(parents=True)
    assert "Not A Space" not in svc.spaces()  # stray dirs aren't spaces
