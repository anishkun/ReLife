"""Tests for the REM ('dream') pass — the opt-in, LLM-driven memory critic.

The model call is injected as a stub returning canned JSON, so these tests are
fully deterministic and never touch the Agent SDK. They verify that, whatever the
critic says, the *application* of its verdicts is safe: reversible (archive, never
delete), confidence-gated, capped, and journaled.
"""

import asyncio
import inspect
import json

from relife import config
from relife.memory import rem, store


def _isolate(tmp_path):
    db = tmp_path / "relife.db"
    store._DB_PATH = db
    rem._STATE_PATH = tmp_path / "rem_state.json"
    rem._JOURNAL_PATH = tmp_path / "rem_journal.jsonl"
    store.init_db()


def _stub(payload):
    async def ask(system_prompt, user_prompt):
        return json.dumps(payload), 0.0123
    return ask


def _run(stub):
    return asyncio.run(rem.run_rem(ask_model=stub))


def test_prune_is_reversible_archive(tmp_path, monkeypatch):
    _isolate(tmp_path)
    monkeypatch.setattr(config, "REM_MAX_PRUNE_FRACTION", 1.0)
    a = store.save("A contradictory, junk memory.", importance=0.5)
    b = store.save("A perfectly good memory.", importance=0.5)
    report = _run(_stub({"verdicts": [
        {"id": a, "action": "prune", "confidence": 0.95, "reason": "junk"},
        {"id": b, "action": "keep", "confidence": 0.9},
    ]}))

    assert report.pruned == 1
    assert store.get(a) is not None              # NOT deleted …
    assert store.get(a).status == "archived"     # … only archived (recoverable)
    assert store.get(b).status == "active"
    # The action was journaled.
    lines = (tmp_path / "rem_journal.jsonl").read_text(encoding="utf-8").splitlines()
    assert any(json.loads(l)["id"] == a and json.loads(l)["action"] == "archive" for l in lines)


def test_reweight_clamps(tmp_path, monkeypatch):
    _isolate(tmp_path)
    monkeypatch.setattr(config, "REM_MAX_PRUNE_FRACTION", 1.0)
    hi = store.save("Should be more salient.", importance=0.5)
    lo = store.save("Should be less salient.", importance=0.5)
    report = _run(_stub({"verdicts": [
        {"id": hi, "action": "reweight", "importance": 1.5, "confidence": 0.9, "reason": "durable"},
        {"id": lo, "action": "reweight", "importance": -0.2, "confidence": 0.9, "reason": "transient"},
    ]}))

    assert report.reweighted == 2
    assert store.get(hi).importance == 1.0   # clamped up
    assert store.get(lo).importance == 0.0   # clamped down


def test_contradiction_archives_loser(tmp_path, monkeypatch):
    _isolate(tmp_path)
    monkeypatch.setattr(config, "REM_MAX_PRUNE_FRACTION", 1.0)
    keep = store.save("The user deploys from main.", kind="preference", importance=0.5)
    stale = store.save("The user deploys from a staging branch.", kind="preference", importance=0.5)
    report = _run(_stub({
        "verdicts": [
            {"id": keep, "action": "keep", "confidence": 0.9},
            {"id": stale, "action": "keep", "confidence": 0.5},
        ],
        "contradictions": [
            {"keep": keep, "archive": stale, "confidence": 0.95, "reason": "conflict"},
        ],
    }))

    assert report.contradictions == 1
    assert report.pruned == 1
    assert store.get(stale).status == "archived"
    assert store.get(keep).status == "active"


def test_low_confidence_is_ignored(tmp_path, monkeypatch):
    _isolate(tmp_path)
    monkeypatch.setattr(config, "REM_MAX_PRUNE_FRACTION", 1.0)
    a = store.save("Borderline memory.", importance=0.5)
    report = _run(_stub({"verdicts": [
        {"id": a, "action": "prune", "confidence": 0.5, "reason": "maybe"},
    ]}))

    assert report.pruned == 0
    assert report.skipped_low_conf == 1
    assert store.get(a).status == "active"   # untouched


def test_prune_cap_bounds_blast_radius(tmp_path, monkeypatch):
    _isolate(tmp_path)
    monkeypatch.setattr(config, "REM_MAX_PRUNE_FRACTION", 0.25)
    ids = [store.save(f"Memory number {i}.", importance=0.5) for i in range(8)]
    # Critic wants to prune four of eight; cap = int(8 * 0.25) = 2.
    report = _run(_stub({"verdicts": [
        {"id": ids[i], "action": "prune", "confidence": 0.9, "reason": "x"} for i in range(4)
    ] + [
        {"id": ids[i], "action": "keep", "confidence": 0.9} for i in range(4, 8)
    ]}))

    assert report.pruned == 2
    assert report.skipped_cap == 2
    archived = [i for i in ids if store.get(i).status == "archived"]
    assert len(archived) == 2


def test_malformed_json_is_a_noop(tmp_path):
    _isolate(tmp_path)
    a = store.save("Memory A.", importance=0.5)
    b = store.save("Memory B.", importance=0.5)

    async def ask(system_prompt, user_prompt):
        return "I'm sorry, I cannot comply.", None

    report = asyncio.run(rem.run_rem(ask_model=ask))

    assert report.reviewed == 2
    assert report.pruned == 0 and report.reweighted == 0
    assert store.get(a).status == "active" and store.get(b).status == "active"


def test_watermark_advances_and_limits_next_buffer(tmp_path, monkeypatch):
    _isolate(tmp_path)
    monkeypatch.setattr(config, "REM_MAX_PRUNE_FRACTION", 1.0)
    a = store.save("First memory.", importance=0.5)
    b = store.save("Second memory.", importance=0.5)
    keep_all = _stub({"verdicts": [
        {"id": a, "action": "keep", "confidence": 0.9},
        {"id": b, "action": "keep", "confidence": 0.9},
    ]})
    first = _run(keep_all)
    assert first.reviewed == 2
    assert rem._read_state()["last_reviewed_id"] == b

    # A new memory arrives; the next pass should review only what's new.
    c = store.save("Third memory.", importance=0.5)
    second = _run(_stub({"verdicts": [{"id": c, "action": "keep", "confidence": 0.9}]}))
    assert second.reviewed == 1
    assert rem._read_state()["last_reviewed_id"] == c


def test_empty_store_is_a_noop(tmp_path):
    _isolate(tmp_path)
    report = _run(_stub({"verdicts": []}))
    assert report.reviewed == 0
    assert report.pruned == 0


def test_consolidate_stays_llm_free(tmp_path):
    """The deterministic 'sleep' pass must never reach for the model (invariant)."""
    from relife.memory import consolidate

    src = inspect.getsource(consolidate)
    assert "claude_agent_sdk" not in src
    assert "ask_model" not in src


def test_rem_module_keeps_sdk_lazy():
    """rem.py must not import the Agent SDK at module scope (keeps memory SDK-free)."""
    src = inspect.getsource(rem)
    assert "claude_agent_sdk" not in src
    # The only agent import is the lazy one inside _default_ask_model.
    assert "from ..agent import" in inspect.getsource(rem._default_ask_model)
