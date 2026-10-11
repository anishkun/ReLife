"""Scaling fixes — each pinned for *behaviour* (same answers as the old,
slow way) and, where it is the point, for staying fast at size.

The architecture review measured four hot paths that grew badly: the
consolidation dedupe (all pairs, re-embedding everything: 19 s at 3,000
memories), skill/workflow search (every file opened on every prompt: 2.4 s at
200 skills), the event log (never pruned; one 500-event window shared by every
space), and recall reinforcing every memory it returned — including memory the
reader may only read. Deterministic; no model calls.
"""

from __future__ import annotations

import json
import math
import os
import random
import time

import pytest

from relife.memory import _procedure
from relife.memory import consolidate as consol
from relife.memory import embeddings
from relife.memory import events as ev
from relife.memory import skills as sk
from relife.memory import spaces
from relife.memory import store as store_mod
from relife.memory import workflows as wf
from relife.memory._text import tokenize
from relife.memory.client import LocalMemoryClient, ScopedMemoryClient


@pytest.fixture
def env(tmp_path, monkeypatch):
    db = tmp_path / "relife.db"
    monkeypatch.setattr(store_mod, "_DB_PATH", db)
    monkeypatch.setattr(ev, "_DB_PATH", db)
    monkeypatch.setattr(consol, "_STATE_PATH", tmp_path / "consolidate_state.json")
    monkeypatch.setattr(sk, "_SKILLS_DIR", tmp_path / "skills")
    monkeypatch.setattr(wf, "_WORKFLOWS_DIR", tmp_path / "workflows")
    monkeypatch.setattr(spaces, "_SPACES_DIR", tmp_path / "spaces")
    _procedure.clear_cache()
    return LocalMemoryClient(), tmp_path


WORDS = [f"term{i}" for i in range(400)]


def _corpus(rng, n, dup_every=7):
    """Random memories with planted near-duplicates (one word swapped in a
    long text, so Jaccard stays >= 0.9) and exact token-set repeats."""
    texts = []
    for i in range(n):
        if texts and i % dup_every == 0:
            base = rng.choice(texts).split()
            if len(base) >= 20 and rng.random() < 0.5:
                base[rng.randrange(len(base))] = rng.choice(WORDS)  # near-dup
            else:
                rng.shuffle(base)  # same tokens, different order
            texts.append(" ".join(base))
        else:
            texts.append(" ".join(rng.sample(WORDS, rng.randint(3, 30))))
    return texts


def _reference_dedupe(texts):
    """The pre-fix algorithm: greedy, all pairs, in id order."""
    seen, dropped = [], set()
    for i, t in enumerate(texts):
        toks = tokenize(t)
        if not toks:
            continue
        hit = next((j for j, o in seen if len(toks & o) / len(toks | o) >= 0.9), None)
        if hit is None:
            seen.append((i, toks))
        else:
            dropped.add(i)
    return dropped


# --- consolidation dedupe -------------------------------------------------------------
@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_indexed_dedupe_merges_exactly_what_all_pairs_did(env, seed):
    rng = random.Random(seed)
    texts = _corpus(rng, 300)
    ids = [store_mod.save(t) for t in texts]  # exact repeats reinforce, not insert
    unique = list(dict.fromkeys(ids))
    by_id = {mid: texts[ids.index(mid)] for mid in unique}
    expected = {unique[i] for i in _reference_dedupe([by_id[m] for m in unique])}
    rep = consol.ConsolidationReport()
    consol._dedupe(rep)
    survivors = {m.id for m in store_mod.all_memories(include_archived=False)}
    assert set(unique) - survivors == expected
    assert rep.merged == len(expected) and rep.merged > 0


def test_dedupe_scales(env):
    rng = random.Random(7)
    s = store_mod._store()
    for _ in range(3000):
        s.save(" ".join(rng.sample(WORDS, 12)) + f" n{rng.random()}")
    t = time.perf_counter()
    consol._dedupe(consol.ConsolidationReport())
    # All pairs took ~19 s here; the indexed pass takes well under a second.
    assert time.perf_counter() - t < 8


def test_a_later_pass_only_compares_new_memories_but_still_catches_their_duplicates(env, monkeypatch):
    old = store_mod.save("deploy the service with blue green switching on friday")
    report = consol.run_consolidation()
    state = json.loads(consol._STATE_PATH.read_text(encoding="utf-8"))
    assert state["dedupe_marks"]["kw:default"][0] == old and report.merged == 0

    compared: list[int] = []
    real = consol._dedupe_group

    def spy(mems, rep, *, since=0, use_sem=False):
        compared.extend(m.id for m in mems if m.id > since)
        return real(mems, rep, since=since, use_sem=use_sem)

    monkeypatch.setattr(consol, "_dedupe_group", spy)
    dup = store_mod.save("friday deploy the service with blue green switching on")
    other = store_mod.save("an unrelated note about invoices")
    report = consol.run_consolidation()
    assert compared == [dup, other]  # the old memory was not re-checked against itself
    assert report.merged == 1 and store_mod.get(dup) is None and store_mod.get(old) is not None


def test_a_stale_mark_from_another_db_never_skips_memories(env):
    consol._STATE_PATH.write_text(
        json.dumps({"dedupe_marks": {"kw:default": [999, 1.0]}}), encoding="utf-8"
    )
    a = store_mod.save("rotate the api keys every quarter for the payments service")
    store_mod.save("every quarter rotate the api keys for the payments service")
    assert consol.run_consolidation().merged == 1  # ids restarted; the mark was ignored
    assert store_mod.get(a) is not None


class _FakeModel:
    """Embeddings without fastembed: texts sharing a topic word are paraphrases."""

    TOPICS = ("invoice", "deploy", "backup")

    def __init__(self):
        self.embedded: list[str] = []

    def vec(self, text):
        v = [0.0] * 4
        hit = next((i for i, t in enumerate(self.TOPICS) if t in text), 3)
        v[hit] = 1.0
        v[3] += 0.01 * (len(text) % 5)  # not identical, still cosine > 0.99
        return v

    def embed(self, texts):
        self.embedded += list(texts)
        return [self.vec(t) for t in texts]

    def embed_one(self, text):
        return self.vec(text)


def test_semantic_dedupe_reuses_stored_vectors(env, monkeypatch):
    model = _FakeModel()
    monkeypatch.setattr(embeddings, "available", lambda: True)
    monkeypatch.setattr(embeddings, "embed_one", model.embed_one)
    monkeypatch.setattr(embeddings, "embed", model.embed)
    monkeypatch.setattr(store_mod.config, "SAVE_DEDUP_SIM", 2.0)  # let save keep both
    keep = store_mod.save("send the invoice reminder on the first")
    store_mod.save("billing: invoice nudges go out monthly")  # a paraphrase
    other = store_mod.save("nightly backup of the database")
    report = consol.run_consolidation()
    assert report.merged == 1
    survivors = {m.id for m in store_mod.all_memories(include_archived=False)}
    assert survivors == {keep, other}
    assert model.embedded == []  # every vector came from the store, none recomputed
    marks = json.loads(consol._STATE_PATH.read_text(encoding="utf-8"))["dedupe_marks"]
    assert "sem:default" in marks


def test_semantic_index_matches_plain_cosine():
    rng = random.Random(3)
    idx = consol._SemIndex(50)

    class M:
        def __init__(self, i):
            self.id = i

    vecs = [[rng.uniform(-1, 1) for _ in range(8)] for _ in range(50)]
    items = [M(i) for i in range(50)]
    for m, v in zip(items, vecs):
        idx.add(m, v)
    for _ in range(20):
        q = [rng.uniform(-1, 1) for _ in range(8)]
        want = next((m for m, v in zip(items, vecs) if embeddings.cosine(q, v) >= 0.6), None)
        got = idx.first_match(q, 0.6)
        assert (got.id if got else None) == (want.id if want else None)


# --- skill / workflow search ------------------------------------------------------------------
def test_search_parses_each_file_once_per_version(env, monkeypatch):
    for i in range(30):
        sk.write_skill(f"skill {i} deploy", "when deploying", f"steps for {i}")
    parsed: list[str] = []
    real = sk._parse
    monkeypatch.setattr(sk, "_parse", lambda p, space="default": (parsed.append(p.name), real(p, space))[1])
    assert sk.find_skills("deploy", k=3)
    first = len(parsed)
    assert first == 30
    sk.find_skills("deploy", k=3)
    sk.find_skills("other words", k=3)
    assert len(parsed) == first  # nothing re-read

    sk.write_skill("skill 3 deploy", "when deploying", "new steps, longer than before")
    [hit] = [s for s in sk.list_skills() if s.slug == "skill-3-deploy"]
    assert hit.body == "new steps, longer than before" and len(parsed) == first + 1


def test_the_cache_sees_another_process_edit_and_delete_files(env):
    sk.write_skill("restart service", "when it hangs", "1. systemctl restart")
    assert sk.find_skills("restart service")[0].body == "1. systemctl restart"
    path = sk._dir() / "restart-service.md"
    # Another process rewrites it (different size, later mtime).
    path.write_text("---\nname: restart service\nwhen_to_use: hangs\n---\n1. kill -HUP the pid\n", encoding="utf-8")
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 10_000_000))
    assert sk.find_skills("restart service")[0].body == "1. kill -HUP the pid"
    path.unlink()
    assert sk.find_skills("restart service") == [] and sk.list_skills() == []


def test_workflows_use_the_same_cache(env, monkeypatch):
    wf.write_workflow("ship", "releasing", "1. tag\n2. push", trigger="release")
    parsed = []
    real = wf._parse
    monkeypatch.setattr(wf, "_parse", lambda p, space="default": (parsed.append(p), real(p, space))[1])
    assert wf.find_workflows("release ship")[0].trigger == "release"
    wf.find_workflows("release ship")
    assert len(parsed) == 1


def test_search_stays_fast_with_many_skills(env):
    for i in range(300):
        sk.write_skill(f"procedure {i} " + " ".join(random.sample(WORDS, 3)), "when", "steps " * 20)
    sk.find_skills("warm the cache")
    t = time.perf_counter()
    for _ in range(10):
        sk.find_skills("term1 term2 deploy", k=2)
    assert (time.perf_counter() - t) / 10 < 0.5  # was ~3.5 s per search at this size


# --- reinforcement is the reader's own -------------------------------------------------------
def _scoped(client, name, reads):
    return ScopedMemoryClient(client, spaces.MemoryScope(read=(name, *reads), write=name, source=name))


def test_reading_inherited_or_default_memory_never_strengthens_it(env):
    client, _ = env
    mine = client.save("the build cache lives under .cache/build", space="agent-a", source="agent-a")
    theirs = client.save("the build server is ci-01", space="default")
    agent = _scoped(client, "agent-a", ["default"])
    for _ in range(5):
        hits = agent.recall("build cache server", reinforce=True)
    assert {m.id for m in hits} == {mine, theirs}
    assert client.get(theirs).use_count == 0  # read-only means its strength too
    assert client.get(mine).use_count == 5


def test_the_memory_context_tool_reinforces_only_the_callers_space(env):
    import anyio

    from relife.memory.tools import EXTERNAL_TOOLS

    client, _ = env
    theirs = client.save("payroll runs on the 25th", space="default")
    agent = _scoped(client, "ext", ["default"])
    ctx = next(t for t in EXTERNAL_TOOLS if t.name == "memory_context")
    for _ in range(3):
        text, is_error = anyio.run(ctx.handler, agent, {"task": "when does payroll run"})
        assert not is_error and "25th" in text
    assert client.get(theirs).use_count == 0


def test_the_main_agent_still_reinforces_its_own_memory(env):
    client, _ = env
    mid = client.save("the staging database is pg-stage")
    client.recall("staging database", reinforce=True)
    assert client.get(mid).use_count == 1


# --- the event log ---------------------------------------------------------------------------
def test_events_are_pruned_and_the_throttle_keeps_counting(env, monkeypatch):
    monkeypatch.setattr(consol.config, "EVENTS_KEEP", 50)
    monkeypatch.setattr(consol.config, "CONSOLIDATE_EVERY", 5)
    monkeypatch.setattr(consol.config, "AUTO_CONSOLIDATE", True)
    for i in range(120):
        ev.log_event("Bash", f"cmd {i}", task_id=f"t{i % 4}")
    consol.run_consolidation()
    assert ev.count() == 50 and ev.max_id() == 120
    assert [e.brief for e in ev.recent_events(1)] == ["cmd 119"]  # the newest survive
    assert not consol.should_auto_run()
    for i in range(5):
        ev.log_event("Bash", f"more {i}", task_id="t9")
    assert consol.should_auto_run()  # by id: pruning didn't stall it


def test_a_pre_pruning_state_file_still_throttles(env, monkeypatch):
    monkeypatch.setattr(consol.config, "AUTO_CONSOLIDATE", True)
    monkeypatch.setattr(consol.config, "CONSOLIDATE_EVERY", 5)
    for i in range(10):
        ev.log_event("Read", f"f{i}")
    consol._STATE_PATH.write_text(json.dumps({"last_event_count": 10}), encoding="utf-8")
    assert not consol.should_auto_run()
    for i in range(5):
        ev.log_event("Read", f"g{i}")
    assert consol.should_auto_run()


def test_a_busy_space_no_longer_hides_a_quiet_spaces_patterns(env):
    # The quiet agent repeats a meaningful sequence 3 times...
    for t in range(3):
        for tool, brief in (("Bash", "git clone repo"), ("Bash", "pytest -q"), ("Bash", "git push")):
            ev.log_event(tool, brief, task_id=f"quiet-{t}", space="quiet")
    # ...then a busy agent logs far more than one window's worth.
    for i in range(800):
        ev.log_event("Read", f"file{i}", task_id=f"busy-{i // 10}", space="busy")
    report = consol.run_consolidation()
    assert any(p.startswith("[quiet] ") for p in report.patterns), report.patterns
    assert any(w.startswith("[quiet] ") for w in report.workflows_created)


def test_a_turn_finds_its_own_events_however_busy_the_log(env):
    client, _ = env
    client.log_event("Edit", "app.py", task_id="mine")
    for i in range(700):  # other sessions, all newer
        client.log_event("Read", f"x{i}", task_id=f"other-{i % 7}")
    client.log_event("Bash", "pytest", task_id="mine")
    assert [e.tool for e in client.events_for_task("mine")] == ["Edit", "Bash"]


def test_reinforce_space_is_ignored_when_reinforce_is_off(env):
    client, _ = env
    mid = client.save("ignored", space="default")
    client.recall("ignored", reinforce=False, reinforce_space="default")
    assert client.get(mid).use_count == 0
    assert not math.isnan(client.get(mid).importance)


# --- connections --------------------------------------------------------------------------------
def test_connections_are_reused_per_thread(tmp_path):
    import threading

    from relife.memory import _sqlite

    path = tmp_path / "c.db"
    a, b = _sqlite.connect(path), _sqlite.connect(path)
    assert a is b
    assert a.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    other = []
    t = threading.Thread(target=lambda: other.append(_sqlite.connect(path)))
    t.start()
    t.join()
    assert other[0] is not a  # sqlite3 connections belong to their thread
    _sqlite.close_all()
    assert _sqlite.connect(path) is not a
    _sqlite.close_all()


def test_writes_are_cheap(env):
    t = time.perf_counter()
    for i in range(200):
        ev.log_event("Bash", f"cmd {i}", task_id="t")
    # A connection per call (open, sync, close) cost ~15 ms per event here.
    assert (time.perf_counter() - t) / 200 < 0.005
