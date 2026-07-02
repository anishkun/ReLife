"""Tests for the out-of-process memory transport (Phase 2 daemon split).

Two guarantees:
1. **Wire round-trip** — every ``Memory`` field survives JSON (dependency-free).
2. **Behavioral equivalence** — the same save→recall→forget→count→consolidate
   sequence yields the same results against ``LocalMemoryClient`` (in-process)
   and ``HttpMemoryClient`` (over the FastAPI app, driven by ``TestClient`` so no
   real socket is opened). ``dream`` is excluded — it needs the model.

The daemon deps (fastapi/httpx) are an optional extra, so the conformance test
skips cleanly when they're absent, mirroring the ``[vector]`` suite.
"""

from __future__ import annotations

import pytest

from relife.memory import events as ev
from relife.memory import store as store_mod
from relife.memory.client import LocalMemoryClient
from relife.memory.remote import wire
from relife.memory.store import Memory


# --- wire round-trip (no extra needed) -------------------------------------
def test_wire_memory_roundtrip_all_fields():
    m = Memory(
        id=42,
        kind="preference",
        text="The user prefers ruff and pytest.",
        tags="python,tools",
        created_at=1_700_000_000.0,
        importance=0.83,
        last_used_at=1_700_000_500.0,
        use_count=7,
        status="active",
    )
    back = wire.memory_from_dict(wire.memory_to_dict(m))
    assert back == m  # dataclass equality → every serialized field matches


def test_wire_memory_or_none():
    assert wire.memory_or_none_to_dict(None) is None
    assert wire.memory_or_none_from_dict(None) is None


def test_wire_reports_roundtrip():
    from relife.memory.consolidate import ConsolidationReport
    from relife.memory.rem import RemReport

    cr = ConsolidationReport(
        archived=2, deleted=1, merged=3, patterns=["p1", "p2"], workflows_created=["w"]
    )
    cr2 = wire.consolidation_from_dict(wire.consolidation_to_dict(cr))
    assert cr2 == cr

    rr = RemReport(reviewed=5, pruned=1, reweighted=2, notes=["n"], cost_usd=0.01)
    rr2 = wire.rem_from_dict(wire.rem_to_dict(rr))
    assert rr2 == rr


# --- parametrized conformance: Local vs Http -------------------------------
def _bind_default_store(tmp_path, monkeypatch):
    """Point the module-level store + event log at an isolated DB (restored on
    teardown) so consolidate/dream — which use the module default — are isolated."""
    db = tmp_path / "relife.db"
    monkeypatch.setattr(store_mod, "_DB_PATH", db)
    monkeypatch.setattr(ev, "_DB_PATH", db)
    store_mod.init_db()
    return db


@pytest.fixture(params=["local", "http"])
def client(request, tmp_path, monkeypatch):
    _bind_default_store(tmp_path, monkeypatch)
    if request.param == "local":
        yield LocalMemoryClient()
        return

    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from relife.memory.remote.daemon import create_app
    from relife.memory.remote.http_client import HttpMemoryClient

    app = create_app(tmp_path / "relife.db")  # binds the same _DB_PATH
    tc = TestClient(app)  # httpx.Client subclass; drives the ASGI app in-process
    yield HttpMemoryClient("http://testserver", client=tc)
    tc.close()


def test_conformance_save_recall(client):
    mid = client.save("The user prefers ruff and pytest.", kind="preference", tags="python")
    assert isinstance(mid, int)
    hits = client.recall("what python tools?")
    assert hits and "ruff" in hits[0].text.lower()
    assert isinstance(hits[0], Memory)
    assert client.count() == 1


def test_conformance_forget(client):
    client.save("The staging deploy runs nightly.", tags="deploy")
    forgotten = client.forget("staging deploy")
    assert forgotten is not None and "staging" in forgotten.text.lower()
    assert client.recall("staging deploy") == []            # archived → not recalled
    assert client.count(include_archived=False) == 0
    assert client.forget("nonexistent topic xyz") is None


def test_conformance_all_memories(client):
    client.save("Fact one about the project.")
    client.save("Fact two about the project.")
    everything = client.all_memories(include_archived=True)
    assert len(everything) == 2
    assert all(isinstance(m, Memory) for m in everything)


def test_conformance_consolidate(client):
    client.save("Something to keep around.")
    report = client.consolidate()
    assert hasattr(report, "summary") and isinstance(report.summary(), str)


# --- daemon-specific: health + token auth ----------------------------------
def test_daemon_health_and_token(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from relife.memory.remote.daemon import create_app

    _bind_default_store(tmp_path, monkeypatch)
    app = create_app(tmp_path / "relife.db", token="s3cret")
    tc = TestClient(app)

    assert tc.get("/health").status_code == 200        # health needs no token
    assert tc.get("/count").status_code == 401          # missing token → rejected
    ok = tc.get("/count", headers={"Authorization": "Bearer s3cret"})
    assert ok.status_code == 200 and ok.json()["count"] == 0
    tc.close()
