"""ReLife memory over MCP — for agents on any LLM.

- The in-process SDK server and the standalone server are built from one set of
  tool specs, so they can't drift; the external set leaves out the upkeep and
  budget-spending tools, and no tool takes a memory space.
- The standalone server speaks the real protocol (driven in-memory via the MCP
  SDK's own client session — no subprocess, no socket, no model).
- The caller's scope is enforced through MCP exactly as in-process.
- The daemon's ``/mcp`` endpoint requires a registered agent's token, scopes
  each request to that agent, and refuses foreign Host headers.
"""

from __future__ import annotations

import anyio
import pytest
from mcp.shared.memory import create_connected_server_and_client_session

from relife import agents as ag
from relife.memory import consolidate as consol
from relife.memory import events as ev
from relife.memory import server as mem_server
from relife.memory import skills as sk
from relife.memory import spaces
from relife.memory import store as store_mod
from relife.memory import tools as T
from relife.memory import workflows as wf
from relife.memory.client import LocalMemoryClient, ScopedMemoryClient
from relife.memory.mcp_server import allowed_hosts_for, build_server


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


# --- one source of truth ----------------------------------------------------------
def test_in_process_tools_are_the_specs():
    for spec in T.INTERNAL_TOOLS:
        sdk_tool = getattr(mem_server, spec.name)
        assert (sdk_tool.name, sdk_tool.description, sdk_tool.input_schema) == (
            spec.name,
            spec.description,
            spec.input_schema,
        )


def test_external_set_leaves_out_upkeep_and_budget():
    names = {t.name for t in T.EXTERNAL_TOOLS}
    assert "memory_dream" not in names and "memory_consolidate" not in names
    assert {"memory_context", "memory_save", "memory_recall", "skill_write"} <= names


@pytest.mark.parametrize("spec", T.INTERNAL_TOOLS + T.EXTERNAL_TOOLS, ids=lambda s: s.name)
def test_no_tool_lets_the_model_choose_a_space(spec):
    props = spec.input_schema.get("properties", {})
    assert not {"space", "spaces", "scope", "agent"} & set(props)


# --- the real protocol (in-memory streams) ------------------------------------------
def _run_mcp(scoped_client, steps):
    async def go():
        server = build_server(lambda _srv: scoped_client)
        async with create_connected_server_and_client_session(server) as session:
            return await steps(session)

    return anyio.run(go)


def _text(result) -> str:
    return result.content[0].text


def test_protocol_round_trip(env):
    store, client, _ = env
    ag.create_agent(store, client, "researcher", runtime="llm", model="ollama/llama3.1")
    me = ScopedMemoryClient(client, store.require("researcher").scope())

    async def steps(session):
        listed = await session.list_tools()
        saved = await session.call_tool(
            "memory_save", {"text": "the vendor API paginates with a cursor", "tags": "api"}
        )
        recalled = await session.call_tool("memory_recall", {"query": "vendor API cursor"})
        context = await session.call_tool("memory_context", {"task": "page through the vendor API"})
        unknown = await session.call_tool("rm_rf", {})
        bad = await session.call_tool("memory_save", {})  # missing required text
        return listed, saved, recalled, context, unknown, bad

    listed, saved, recalled, context, unknown, bad = _run_mcp(me, steps)
    assert [t.name for t in listed.tools] == [t.name for t in T.EXTERNAL_TOOLS]
    assert not saved.isError and "Saved memory #" in _text(saved)
    assert "paginates with a cursor" in _text(recalled)
    assert "Relevant long-term memory" in _text(context)
    assert unknown.isError and bad.isError
    [m] = client.all_memories(spaces=["researcher"])
    assert (m.source, m.tags) == ("researcher", "api")


def test_scope_holds_through_mcp(env):
    store, client, _ = env
    client.save("the user's home address is private")  # main agent, default space
    ag.create_agent(store, client, "insider")
    ag.create_agent(store, client, "outsider", runtime="external", isolated=True)
    ScopedMemoryClient(client, store.require("insider").scope()).save("insider's design notes on caching")
    outsider = ScopedMemoryClient(client, store.require("outsider").scope())

    async def steps(session):
        a = await session.call_tool("memory_recall", {"query": "home address private"})
        b = await session.call_tool("memory_recall", {"query": "design notes caching"})
        c = await session.call_tool("memory_forget", {"query": "design notes caching"})
        return a, b, c

    a, b, c = _run_mcp(outsider, steps)
    assert _text(a) == "(no relevant memories)"  # isolated: no default space
    assert _text(b) == "(no relevant memories)"  # not inherited
    assert _text(c).startswith("(nothing matched")
    assert client.count(include_archived=False) == 2


# --- HTTP /mcp on the daemon ------------------------------------------------------------
def _rpc(method, params=None, rid=1):
    return {"jsonrpc": "2.0", "id": rid, "method": method, "params": params or {}}


_ACCEPT = {"Accept": "application/json, text/event-stream"}


@pytest.fixture
def daemon(env):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from relife.memory.remote.daemon import create_app

    store, client, tmp = env
    ag.create_agent(store, client, "ext", runtime="external")
    token = ag.issue_token(store, "ext")
    app = create_app(
        tmp / "relife.db",
        skills_dir=tmp / "skills",
        workflows_dir=tmp / "workflows",
        spaces_dir=tmp / "spaces",
        agents_path=tmp / "agents.json",
        mcp_hosts=allowed_hosts_for("testserver"),
    )
    with TestClient(app) as tc:  # the lifespan runs the MCP session manager
        yield tc, token, store, client


def test_http_mcp_requires_an_agent_token(daemon):
    tc, token, _, _ = daemon
    assert tc.post("/mcp", json=_rpc("tools/list"), headers=_ACCEPT).status_code == 401
    bad = {**_ACCEPT, "Authorization": "Bearer rla_not-a-real-token"}
    assert tc.post("/mcp", json=_rpc("tools/list"), headers=bad).status_code == 401
    ok = {**_ACCEPT, "Authorization": f"Bearer {token}"}
    r = tc.post("/mcp", json=_rpc("tools/list"), headers=ok)
    assert r.status_code == 200, r.text
    names = [t["name"] for t in r.json()["result"]["tools"]]
    assert names == [t.name for t in T.EXTERNAL_TOOLS]


def test_http_mcp_scopes_each_call_to_the_token_agent(daemon):
    tc, token, store, client = daemon
    hdr = {**_ACCEPT, "Authorization": f"Bearer {token}"}
    r = tc.post(
        "/mcp",
        json=_rpc("tools/call", {"name": "memory_save", "arguments": {"text": "ext prefers JSON logs"}}),
        headers=hdr,
    )
    assert r.status_code == 200, r.text
    assert not r.json()["result"].get("isError")
    [m] = client.all_memories(spaces=["ext"])
    assert (m.text, m.source) == ("ext prefers JSON logs", "ext")
    assert client.recall("JSON logs prefers") == []  # not in the user's default space

    # A revoked token stops working immediately (the registry is re-read).
    ag.revoke_token(store, "ext")
    assert tc.post("/mcp", json=_rpc("tools/list"), headers=hdr).status_code == 401


def test_http_mcp_refuses_foreign_host(daemon):
    tc, token, _, _ = daemon
    hdr = {**_ACCEPT, "Authorization": f"Bearer {token}", "Host": "attacker.example"}
    r = tc.post("/mcp", json=_rpc("tools/list"), headers=hdr)
    assert r.status_code in (400, 403, 421)


def test_allowed_hosts_for_bind():
    assert "127.0.0.1:*" in allowed_hosts_for()
    assert "10.0.0.5" in allowed_hosts_for("10.0.0.5")
    assert "0.0.0.0" not in allowed_hosts_for("0.0.0.0")  # a wildcard bind is not a Host name


def test_stdio_warms_up_before_it_starts_reading(env, monkeypatch):
    """On Windows, loading the embedding model on a worker thread while the
    stdio reader blocks on stdin hung the first tool call; the model must load
    before the loop starts reading."""
    from relife.memory import mcp_server as ms

    order = []
    monkeypatch.setattr(ms, "_warm_up", lambda client: order.append("warm"))
    monkeypatch.setattr("anyio.run", lambda fn: order.append("serve"))
    _, client, _ = env
    ms.run_stdio(spaces.DEFAULT_SCOPE, client)
    assert order == ["warm", "serve"]
