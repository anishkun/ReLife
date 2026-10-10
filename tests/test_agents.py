"""The agent registry and the memory handoff between agents.

What must hold: an agent writes only its own space and can't widen its scope
(no space argument is accepted, admin ops refuse); it reads what it inherited,
live and read-only, transitively down a lineage; a fork is a snapshot; promote
is the only way into the user's default space; tokens are stored hashed and
rotate. Deterministic, isolated store per test.
"""

from __future__ import annotations

import json

import anyio
import pytest

from relife import agents as ag
from relife.memory import consolidate as consol
from relife.memory import events as ev
from relife.memory import skills as sk
from relife.memory import spaces
from relife.memory import store as store_mod
from relife.memory import workflows as wf
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
    store = ag.AgentStore(tmp_path / "agents.json")
    return store, LocalMemoryClient(), tmp_path


def scoped(store, client, name):
    return ScopedMemoryClient(client, store.require(name).scope())


# --- profile validation ---------------------------------------------------------
@pytest.mark.parametrize(
    "record",
    [
        {"name": "default"},                       # the main agent's space
        {"name": "Bad Name"},
        {"name": "../x"},
        {"name": "ok", "runtime": "shell"},
        {"name": "ok", "space": "default"},        # no agent writes the user's memory
        {"name": "ok", "inherits": "alpha"},       # not a list
        {"name": "ok", "inherits": ["../etc"]},
        {"name": "ok", "token_hash": "plaintext-token"},
        {"name": "ok", "parent": "Not Valid"},
    ],
)
def test_invalid_profiles_are_rejected(record):
    with pytest.raises((ValueError, TypeError)):
        ag.AgentProfile.from_dict(record)


def test_profile_normalizes_inherits():
    p = ag.AgentProfile.from_dict(
        {"name": "coder", "inherits": ["coder", "default", "alpha", "alpha", "beta"]}
    )
    assert p.inherits == ["alpha", "beta"]  # own + default are implicit, no repeats


def test_scope_reads_own_inherited_and_default():
    p = ag.AgentProfile(name="coder", inherits=["alpha"])
    sc = p.scope()
    assert (sc.write, sc.read, sc.source) == ("coder", ("coder", "alpha", "default"), "coder")
    iso = ag.AgentProfile(name="ext", isolated=True, inherits=["alpha"]).scope()
    assert "default" not in iso.read


def test_scope_for_unknown_agent_never_falls_back(env):
    store, _, _ = env
    assert ag.scope_for(None, store) is spaces.DEFAULT_SCOPE
    with pytest.raises(LookupError):
        ag.scope_for("ghost", store)


# --- store ------------------------------------------------------------------------
def test_store_persists_atomically_and_lazily(env):
    store, client, tmp = env
    assert not (tmp / "agents.json").exists()  # nothing written until a change
    ag.create_agent(store, client, "alpha", description="research")
    reread = ag.AgentStore(tmp / "agents.json")
    assert reread.require("alpha").description == "research"
    assert not list(tmp.glob("agents.json.tmp"))


def test_unreadable_records_are_dropped_and_original_kept(env):
    _, _, tmp = env
    path = tmp / "agents.json"
    path.write_text(
        json.dumps({"agents": [{"name": "ok"}, {"name": "default"}, {"name": "x", "runtime": "evil"}]}),
        encoding="utf-8",
    )
    store = ag.AgentStore(path)
    assert [a.name for a in store.list()] == ["ok"]
    assert "2 unreadable" in store.problem
    store.put(ag.AgentProfile(name="new"))
    assert list(tmp.glob("agents.json.corrupt-*"))


# --- handoff ----------------------------------------------------------------------
def test_inherit_is_live_read_only_and_transitive(env):
    store, client, _ = env
    ag.create_agent(store, client, "alpha")
    ag.create_agent(store, client, "beta", inherit=["alpha"])
    ag.create_agent(store, client, "gamma", inherit=["beta"])
    assert store.require("gamma").scope().read == ("gamma", "beta", "alpha", "default")

    a = scoped(store, client, "alpha")
    a.save("the API rate limit is 100 requests per minute")
    g = scoped(store, client, "gamma")
    # Live: alpha's later learning is visible to its descendants...
    [m] = g.recall("API rate limit requests")
    assert m.space == "alpha"
    # ...but read-only: gamma can't forget or archive it.
    assert g.forget("API rate limit") is None
    assert g.archive(m.id) is False
    assert store_mod.get(m.id).status == "active"


def test_fork_is_a_snapshot(env):
    store, client, _ = env
    ag.create_agent(store, client, "base")
    ag.create_agent(store, client, "alpha", inherit=["base"])
    a = scoped(store, client, "alpha")
    a.save("use feature flags for risky deploys")
    a.skill_write("flag-deploy", "risky deploys", "1. wrap in a flag")
    profile, copied = ag.create_agent(store, client, "alpha2", fork="alpha")
    assert copied == {"memories": 1, "skills": 1, "workflows": 0}
    assert profile.parent == "alpha" and profile.inherits == ["base"]
    a.save("a lesson alpha learned after the fork")
    f = scoped(store, client, "alpha2")
    assert f.recall("lesson alpha learned after fork") == []  # snapshot, not live
    [m] = f.recall("feature flags risky deploys")
    assert m.space == "alpha2"  # its own copy
    assert f.skill_find("risky deploys flag")[0].space == "alpha2"


def test_create_refuses_duplicates_and_unknown_parents(env):
    store, client, _ = env
    ag.create_agent(store, client, "alpha")
    with pytest.raises(ValueError):
        ag.create_agent(store, client, "alpha")
    with pytest.raises(LookupError):
        ag.create_agent(store, client, "beta", inherit=["ghost"])
    with pytest.raises(LookupError):
        ag.create_agent(store, client, "beta", fork="ghost")


def test_promote_is_the_way_into_default(env):
    store, client, _ = env
    ag.create_agent(store, client, "alpha")
    a = scoped(store, client, "alpha")
    keep = a.save("pin dependency versions in CI")
    a.save("a scratch note nobody needs")
    assert client.recall("pin dependency versions") == []  # main agent doesn't see it
    out = ag.promote(store, client, "alpha", ids=[keep])
    assert out["memories"] == 1
    [m] = client.recall("pin dependency versions CI")
    assert (m.space, m.source) == ("default", "alpha")  # provenance survives


def test_attach_and_detach(env):
    store, client, _ = env
    ag.create_agent(store, client, "alpha")
    ag.create_agent(store, client, "beta")
    assert "alpha" in ag.attach(store, "beta", "alpha").scope().read
    assert ag.attach(store, "beta", "default").inherits == ["alpha"]  # implicit already
    assert "alpha" not in ag.detach(store, "beta", "alpha").scope().read


def test_delete_archives_unless_shared_or_kept(env):
    store, client, _ = env
    ag.create_agent(store, client, "alpha")
    scoped(store, client, "alpha").save("alpha's only fact")
    assert ag.delete_agent(store, client, "alpha") == 1
    assert store.get("alpha") is None

    ag.create_agent(store, client, "one", space="team")
    ag.create_agent(store, client, "two", space="team")
    scoped(store, client, "one").save("a team fact both write")
    assert ag.delete_agent(store, client, "one") == 0  # "two" still writes there
    assert ag.delete_agent(store, client, "two", keep_memory=True) == 0
    assert client.count(include_archived=False, spaces=["team"]) == 1


# --- tokens -----------------------------------------------------------------------
def test_tokens_are_hashed_rotated_and_revoked(env):
    store, client, tmp = env
    ag.create_agent(store, client, "ext", runtime="external")
    t1 = ag.issue_token(store, "ext")
    assert t1.startswith("rla_")
    assert t1 not in (tmp / "agents.json").read_text(encoding="utf-8")
    assert ag.AgentStore(tmp / "agents.json").by_token(t1).name == "ext"
    t2 = ag.issue_token(store, "ext")
    assert store.by_token(t1) is None and store.by_token(t2).name == "ext"
    ag.revoke_token(store, "ext")
    assert store.by_token(t2) is None
    assert store.by_token(None) is None and store.by_token("") is None


# --- the scoped client ------------------------------------------------------------
def test_scoped_client_writes_only_its_space(env):
    store, client, _ = env
    ag.create_agent(store, client, "alpha")
    a = scoped(store, client, "alpha")
    mid = a.save("alpha keeps its notes here")
    m = client.get(mid)
    assert (m.space, m.source) == ("alpha", "alpha")
    a.skill_write("alpha-skill", "x", "1. y")
    a.workflow_write("alpha-flow", "x", "1. y")
    a.log_event("Bash", "pytest", task_id="s1")
    assert sk.count("alpha") == 1 and sk.count() == 0
    assert wf.count("alpha") == 1 and wf.count() == 0
    assert [e.space for e in ev.recent_events()] == ["alpha"]


def test_scoped_client_cannot_widen_its_scope(env):
    store, client, _ = env
    ag.create_agent(store, client, "alpha")
    ag.create_agent(store, client, "beta")
    other = scoped(store, client, "beta").save("beta's private plan")
    a = scoped(store, client, "alpha")
    with pytest.raises(TypeError):
        a.save("sneaky", space="default")  # no space argument exists
    with pytest.raises(TypeError):
        a.recall("plan", spaces=["beta"])
    assert a.recall("beta private plan") == []
    assert a.get(other) is None
    assert "beta" not in a.spaces()
    for admin in (
        lambda: a.copy_space("beta", "alpha"),
        lambda: a.export_space("beta"),
        lambda: a.import_pack({}, "default"),
        lambda: a.archive_space("beta"),
    ):
        with pytest.raises(PermissionError):
            admin()
    with pytest.raises(PermissionError):
        anyio.run(a.dream)


def test_scoped_events_for_task_only_show_its_own(env):
    store, client, _ = env
    ag.create_agent(store, client, "alpha")
    client.log_event("Bash", "ls", task_id="shared-id")
    a = scoped(store, client, "alpha")
    a.log_event("Read", "x", task_id="shared-id")
    assert [e.tool for e in a.events_for_task("shared-id")] == ["Read"]


def test_mcp_config_shapes():
    cfg = ag.mcp_config("ext", python="/py", home="/home/u/.relife")
    server = cfg["stdio"]["mcpServers"]["relife-memory"]
    assert server["args"] == ["-m", "relife", "mcp", "--agent", "ext"]
    assert server["env"] == {"RELIFE_HOME": "/home/u/.relife"}
    assert "http" not in cfg  # no token, no HTTP snippet
    cfg = ag.mcp_config("ext", python="/py", home="/h", http_url="http://127.0.0.1:8787/mcp", token="rla_x")
    assert cfg["http"]["mcpServers"]["relife-memory"]["headers"] == {"Authorization": "Bearer rla_x"}
