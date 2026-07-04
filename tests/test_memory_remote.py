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

from relife.memory import consolidate as consol
from relife.memory import events as ev
from relife.memory import skills as sk
from relife.memory import store as store_mod
from relife.memory import workflows as wf
from relife.memory.client import LocalMemoryClient, MemoryClient
from relife.memory.remote import wire
from relife.memory.skills import Skill
from relife.memory.store import Memory
from relife.memory.workflows import Workflow


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


def test_wire_skill_workflow_roundtrip():
    s = Skill(
        name="scaffold-python-cli",
        when_to_use="Setting up a new Python CLI.",
        body="1. init\n2. add typer",
        slug="scaffold-python-cli",
    )
    assert wire.skill_from_dict(wire.skill_to_dict(s)) == s

    w = Workflow(
        name="ship-new-service",
        when_to_use="Standing up a new service.",
        trigger="scaffold,test,push",
        body="1. scaffold\n2. test\n3. push",
        slug="ship-new-service",
    )
    assert wire.workflow_from_dict(wire.workflow_to_dict(w)) == w


# --- parametrized conformance: Local vs Http -------------------------------
def _bind_default_store(tmp_path, monkeypatch):
    """Point the module-level store + event log + consolidate state at isolated
    paths (restored on teardown) so consolidate/dream — which use the module
    default — are isolated."""
    db = tmp_path / "relife.db"
    monkeypatch.setattr(store_mod, "_DB_PATH", db)
    monkeypatch.setattr(ev, "_DB_PATH", db)
    monkeypatch.setattr(consol, "_STATE_PATH", tmp_path / "consolidate_state.json")
    store_mod.init_db()
    ev.init_db()
    return db


@pytest.fixture(params=["local", "http"])
def client(request, tmp_path, monkeypatch):
    _bind_default_store(tmp_path, monkeypatch)
    # Isolate procedural memory the same way as the store: point the module
    # globals at tmp dirs. The http branch passes the *same* dirs into create_app
    # so the daemon binds them too — otherwise daemon-side consolidate would write
    # workflows the client never sees (the split this phase closes).
    skills_dir = tmp_path / "skills"
    workflows_dir = tmp_path / "workflows"
    monkeypatch.setattr(sk, "_SKILLS_DIR", skills_dir)
    monkeypatch.setattr(wf, "_WORKFLOWS_DIR", workflows_dir)

    if request.param == "local":
        yield LocalMemoryClient()
        return

    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from relife.memory.remote.daemon import create_app
    from relife.memory.remote.http_client import HttpMemoryClient

    app = create_app(  # binds the same _DB_PATH + skills/workflows dirs
        tmp_path / "relife.db",
        skills_dir=skills_dir,
        workflows_dir=workflows_dir,
    )
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


# --- procedural memory conformance (skills / workflows) --------------------
def test_conformance_skill_roundtrip(client):
    slug = client.skill_write(
        "scaffold-python-cli", "Setting up a new Python CLI.", "1. init\n2. add typer"
    )
    assert slug == "scaffold-python-cli"
    hits = client.skill_find("python cli scaffold")
    assert hits and isinstance(hits[0], Skill)
    assert hits[0].slug == "scaffold-python-cli"
    assert "typer" in hits[0].body
    assert client.skill_count() == 1
    assert client.skill_find("totally unrelated quantum chromodynamics") == []


def test_conformance_workflow_roundtrip(client):
    slug = client.workflow_write(
        "ship-new-service",
        "Standing up a new service.",
        "1. scaffold\n2. test\n3. push",
        trigger="scaffold,test,push",
    )
    assert slug == "ship-new-service"
    hits = client.workflow_find("ship service scaffold")
    assert hits and isinstance(hits[0], Workflow)
    assert hits[0].slug == "ship-new-service"
    assert hits[0].trigger == "scaffold,test,push"  # trigger survives the wire
    assert client.workflow_count() == 1
    assert client.workflow_find("totally unrelated quantum chromodynamics") == []


def test_conformance_write_validation(client):
    # Blank name or steps raise ValueError on BOTH transports (the daemon maps
    # its 400 back to ValueError so consumers fail identically).
    with pytest.raises(ValueError):
        client.skill_write("", "when", "steps")
    with pytest.raises(ValueError):
        client.skill_write("name", "when", "")
    with pytest.raises(ValueError):
        client.workflow_write("", "when", "steps")


def test_conformance_unicode_content(client):
    # Non-ASCII body must survive the JSON/UTF-8 round-trip byte-identical. The
    # tokenizer is ASCII-only, so match on ASCII words in when_to_use.
    body = "Deploy notes: café ☕ → naïve fix ✓ — déjà vu"
    client.skill_write("deploy-notes", "Deployment checklist reference.", body)
    hits = client.skill_find("deployment checklist reference")
    assert hits and hits[0].body == body


def test_conformance_consolidate_workflow_visible(client):
    # Headline regression: a recurring git-clone → test → push procedure across
    # tasks. Consolidation (server-side under the daemon) must synthesize a
    # workflow that the SAME client can then find — proving daemon-written
    # workflows land where clients read.
    for t in ("task1", "task2", "task3"):
        ev.log_event("Bash", "git clone https://github.com/x/y", task_id=t)
        ev.log_event("Bash", "mvn test", task_id=t)
        ev.log_event("Bash", "git push origin feat/x", task_id=t)

    report = client.consolidate()
    assert report.workflows_created, "expected a synthesized workflow"
    assert client.workflow_count() >= 1
    assert client.workflow_find("git clone test push")


# --- daemon-specific: health + token auth ----------------------------------
def test_daemon_health_and_token(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from relife.memory.remote.daemon import create_app

    _bind_default_store(tmp_path, monkeypatch)
    monkeypatch.setattr(sk, "_SKILLS_DIR", tmp_path / "skills")
    monkeypatch.setattr(wf, "_WORKFLOWS_DIR", tmp_path / "workflows")
    app = create_app(
        tmp_path / "relife.db",
        token="s3cret",
        skills_dir=tmp_path / "skills",
        workflows_dir=tmp_path / "workflows",
    )
    tc = TestClient(app)

    assert tc.get("/health").status_code == 200        # health needs no token
    assert tc.get("/count").status_code == 401          # missing token → rejected
    assert tc.get("/skills/count").status_code == 401   # new routes gated too
    ok = tc.get("/count", headers={"Authorization": "Bearer s3cret"})
    assert ok.status_code == 200 and ok.json()["count"] == 0
    # Health reports procedural-memory counts so a remote check is one call.
    health = tc.get("/health").json()
    assert health["skills"] == 0 and health["workflows"] == 0
    tc.close()
