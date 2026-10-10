"""The platform on the server: agents, memory spaces and crews over HTTP, crew
runs hosted like sessions (approvals to the browser), and crew schedules.

All model-free. The crew planner's model call, a ReLife member's turn and the
CrewAI member's LLM are injected stubs; the tests that run a real
``Crew.kickoff()`` skip without the [crewai] extra (Python <= 3.13). The
thread bridge — a member's turn on the crew's worker loop asking for an
approval the browser answers on the server loop — is tested without CrewAI.
"""

from __future__ import annotations

import asyncio
import json
import time

import anyio
import pytest

from relife import agents as ag
from relife.memory import consolidate as consol
from relife.memory import events as ev
from relife.memory import skills as sk
from relife.memory import spaces
from relife.memory import store as store_mod
from relife.memory import workflows as wf
from relife.memory.client import LocalMemoryClient
from relife.server.crews import LoopBroker, member_event
from relife.server.schedules import Schedule
from relife.server.session import ApprovalBroker

SPEC = {
    "goal": "add CSV export",
    "agents": [
        {"name": "builder", "role": "Engineer", "goal": "build features", "runtime": "relife"},
        {"name": "critic", "role": "Reviewer", "goal": "review work", "runtime": "llm",
         "llm": "claude-max", "inherit": ["builder"]},
    ],
    "tasks": [
        {"name": "build", "agent": "builder", "description": "Add CSV export to the report page.",
         "expected_output": "Working export with tests."},
        {"name": "review", "agent": "critic", "description": "Review the export change.",
         "expected_output": "A verdict.", "context": ["build"]},
    ],
}


@pytest.fixture
def env(tmp_path, monkeypatch):
    db = tmp_path / "relife.db"
    monkeypatch.setattr(store_mod, "_DB_PATH", db)
    monkeypatch.setattr(ev, "_DB_PATH", db)
    monkeypatch.setattr(consol, "_STATE_PATH", tmp_path / "consolidate_state.json")
    monkeypatch.setattr(sk, "_SKILLS_DIR", tmp_path / "skills")
    monkeypatch.setattr(wf, "_WORKFLOWS_DIR", tmp_path / "workflows")
    monkeypatch.setattr(spaces, "_SPACES_DIR", tmp_path / "spaces")
    for k in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    return LocalMemoryClient(), tmp_path


class StubModel:
    """Stands in for ``ask_model_oneshot`` (planner and ClaudeMaxLLM)."""

    def __init__(self, answer="Thought: done\nFinal Answer: looks good, ship it"):
        self.answer = answer
        self.calls = 0

    async def __call__(self, system, prompt):
        self.calls += 1
        return self.answer, 0.02


def make_app(env, **kw):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from relife.server.app import create_app

    client, tmp = env
    app = create_app(
        schedules_path=tmp / "schedules.json",
        runs_dir=tmp / "runs",
        run_scheduler=False,
        reap=False,
        workspace_root=tmp / "ws",
        agents_path=tmp / "agents.json",
        crews_dir=tmp / "crews",
        memory_client=client,
        **kw,
    )
    return app, TestClient(app)


def wait_for(cond, timeout=10.0, every=0.02):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        v = cond()
        if v:
            return v
        time.sleep(every)
    raise AssertionError("condition never held")


# --- the bridge (no CrewAI) ----------------------------------------------------------
def test_member_events_never_end_the_crew_for_a_watcher():
    assert member_event("a", {"type": "result", "cost_usd": 1})["type"] == "member_done"
    assert member_event("a", {"type": "error", "message": "x"})["type"] == "member_error"
    out = member_event("a", {"type": "tool_use", "name": "Bash"})
    assert out == {"type": "tool_use", "name": "Bash", "agent": "a"}


def test_loop_broker_carries_an_approval_from_a_worker_loop_to_the_server_loop():
    """A crew member's turn runs on the worker thread's own loop; its approval
    must surface (and be answered) on the server loop where the browser is."""

    async def flow():
        seen: list[dict] = []
        broker = ApprovalBroker(lambda e: _append(seen, e))
        loop = asyncio.get_running_loop()
        bridge = LoopBroker(broker, loop, {"agent": "builder"})

        def worker():  # another thread, another loop — like CrewAI's (no copied context)
            return anyio.run(lambda: bridge.request("Bash", {"command": "curl -X POST x"}, "outward", timeout=5))

        pending = loop.run_in_executor(None, worker)
        for _ in range(200):
            if seen:
                break
            await asyncio.sleep(0.01)
        req = seen[0]
        assert req["type"] == "approval_request" and req["agent"] == "builder"
        assert broker.resolve(req["approval_id"], True)
        assert await pending is True
        assert seen[-1] == {"type": "approval_resolved", "approval_id": req["approval_id"],
                            "approved": True, "agent": "builder"}

        # Stopping a run denies what is still waiting, at once.
        late = loop.run_in_executor(None, worker)
        for _ in range(200):
            if len(seen) >= 3:
                break
            await asyncio.sleep(0.01)
        assert broker.deny_all() == 1
        assert await late is False

    anyio.run(flow)


async def _append(seen, e):
    seen.append(e)


# --- agents + spaces over HTTP -----------------------------------------------------------
def test_agents_routes_register_hand_down_and_promote(env):
    client, tmp = env
    app, tc = make_app(env)
    r = tc.post("/agents", json={"name": "builder", "description": "writes code"})
    assert r.status_code == 201 and r.json()["own_space"] == "builder"
    ag.AgentStore(tmp / "agents.json")  # the CLI sees what the server wrote
    store = ag.AgentStore(tmp / "agents.json")
    from relife.memory.client import ScopedMemoryClient

    ScopedMemoryClient(client, store.require("builder").scope()).save("exports live in export.py")

    r = tc.post("/agents", json={"name": "critic", "inherit": ["builder"]})
    assert r.status_code == 201 and r.json()["reads"] == ["critic", "builder", "default"]
    assert tc.post("/agents", json={"name": "critic"}).status_code == 400  # exists
    assert tc.post("/agents", json={"name": "x", "inherit": ["nobody"]}).status_code == 404
    assert tc.post("/agents", json={"name": "Bad Name"}).status_code == 400
    assert tc.post("/agents", json={"name": "y", "inherit": "builder"}).status_code == 400

    listed = tc.get("/agents").json()["agents"]
    assert [a["name"] for a in listed] == ["builder", "critic"]
    assert listed[0]["memories"] == 1 and "token_hash" not in listed[0]
    assert tc.get("/agents/nobody").status_code == 404
    assert tc.get("/spaces").json()["spaces"]["builder"]["memories"] == 1

    r = tc.post("/agents/critic/detach", json={"other": "builder"})
    assert r.json()["reads"] == ["critic", "default"]
    r = tc.post("/agents/critic/attach", json={"other": "builder"})
    assert "builder" in r.json()["reads"]

    assert tc.post("/agents/builder/promote", json={"ids": ["1"]}).status_code == 400
    r = tc.post("/agents/builder/promote", json={})
    assert r.status_code == 200 and r.json()["copied"]["memories"] == 1
    assert [m.text for m in client.all_memories(spaces=["default"])] == ["exports live in export.py"]

    r = tc.delete("/agents/critic")
    assert r.json() == {"removed": True, "archived": 0}
    assert tc.delete("/agents/critic").status_code == 404


def test_agent_routes_are_guarded_like_every_mutating_route(env):
    _, tc = make_app(env, token="secret")
    assert tc.get("/agents").status_code == 401
    h = {"authorization": "Bearer secret"}
    assert tc.get("/agents", headers=h).status_code == 200
    cross = {**h, "origin": "https://evil.example"}
    assert tc.post("/agents", json={"name": "x"}, headers=cross).status_code == 403
    assert tc.post("/crews", json={"task": "x"}, headers=cross).status_code == 403


# --- crews over HTTP (planning needs no CrewAI) ----------------------------------------------
def test_plan_a_crew_then_discard_it(env):
    planner = StubModel(json.dumps(SPEC))
    _, tc = make_app(env, crew_ask_model=planner)
    assert tc.post("/crews", json={}).status_code == 400
    r = tc.post("/crews", json={"task": "add CSV export"})
    assert r.status_code == 201, r.text
    crew = r.json()
    assert planner.calls == 1 and crew["status"] == "planned"
    assert any("builder — Engineer" in line for line in crew["plan"])
    assert crew["workspace"].replace("\\", "/").endswith(f"ws/crews/{crew['id']}")

    listed = tc.get("/crews").json()
    assert [c["id"] for c in listed["crews"]] == [crew["id"]]
    assert "claude-max" in listed["llms"]
    assert tc.delete(f"/crews/{crew['id']}").json() == {"discarded": True}
    assert tc.get(f"/crews/{crew['id']}").json()["status"] == "cancelled"
    assert tc.delete(f"/crews/{crew['id']}").status_code == 409
    assert tc.post(f"/crews/{crew['id']}/run").status_code in (409, 501)
    assert tc.get("/crews/../../etc").status_code == 404


def test_a_spec_over_http_may_only_name_configured_models(env, monkeypatch):
    from relife import config

    monkeypatch.setattr(config, "CREW_LLMS", ())
    _, tc = make_app(env)
    bad = json.loads(json.dumps(SPEC))
    bad["agents"][1]["llm"] = "openai/gpt-4.1"
    r = tc.post("/crews", json={"spec": bad})
    assert r.status_code == 400 and "gpt-4.1" in r.json()["detail"]
    assert tc.post("/crews", json={"spec": SPEC}).status_code == 201  # claude-max is always allowed


def test_running_a_crew_without_crewai_says_how_to_get_it(env, monkeypatch):
    from relife.server import crews as crews_mod

    monkeypatch.setattr(crews_mod, "crewai_available", lambda: False)
    _, tc = make_app(env)
    crew = tc.post("/crews", json={"spec": SPEC}).json()
    r = tc.post(f"/crews/{crew['id']}/run")
    assert r.status_code == 501 and "3.12 venv" in r.json()["detail"]
    assert tc.get(f"/crews/{crew['id']}").json()["status"] == "planned"  # still runnable later


# --- crew schedules ------------------------------------------------------------------------
def test_crew_schedules_validate_like_a_crew_and_take_no_grants(env):
    client, tmp = env
    ag.create_agent(ag.AgentStore(tmp / "agents.json"), client, "builder")
    agents = {"builder"}
    s = Schedule.new(name="nightly", task="", spec={"every": "1d"}, crew=SPEC, agents=agents)
    assert s.crew["agents"][0]["name"] == "builder"
    assert s.to_dict()["crew_text"] == "runs a crew (builder, critic) · 2 tasks"
    with pytest.raises(ValueError, match="no pre-approvals"):
        Schedule.new(name="n", task="", spec={"every": "1d"}, crew=SPEC, agents=agents,
                     grants=[{"kind": "email", "addresses": ["me@example.com"]}])
    with pytest.raises(ValueError, match="not both"):
        Schedule.new(name="n", task="", spec={"every": "1d"}, crew=SPEC, agents=agents, work={})
    with pytest.raises(ValueError):
        Schedule.new(name="n", task="", spec={"every": "1d"}, crew={"goal": "x"}, agents=agents)
    # A corrupt crew on disk loads disabled, never as a blank task.
    broken = Schedule.from_dict({**s.to_dict(), "crew": {"goal": "x"}})
    assert broken.crew is None and broken.enabled is False

    _, tc = make_app(env)
    r = tc.post("/schedules", json={"name": "nightly", "every": "1d", "crew": SPEC})
    assert r.status_code == 201 and r.json()["crew"]["goal"] == "add CSV export"
    sid = r.json()["id"]
    assert tc.patch(f"/schedules/{sid}", json={"crew": None}).status_code == 400  # then it needs a task
    r = tc.patch(f"/schedules/{sid}", json={"crew": None, "task": "just do it"})
    assert r.status_code == 200 and r.json()["crew"] is None


def test_a_crew_schedule_without_crewai_records_the_error(env, monkeypatch):
    from relife.server import crews as crews_mod

    monkeypatch.setattr(crews_mod, "crewai_available", lambda: False)
    app, tc = make_app(env)
    sid = tc.post("/schedules", json={"name": "nightly", "every": "1d", "crew": SPEC}).json()["id"]
    r = tc.post(f"/schedules/{sid}/run")
    assert r.json()["status"].startswith("error: CrewAI isn't installed")


# --- a real crew under the server (needs CrewAI) --------------------------------------------
def _crew_app(env, *, turn, approval_timeout=5.0):
    pytest.importorskip("crewai")
    from relife.crew.llm import ClaudeMaxLLM

    return make_app(
        env,
        crew_turn=turn,
        crew_llm_factory=lambda model: ClaudeMaxLLM(ask_model=StubModel()),
        crew_approval_timeout=approval_timeout,
    )


def _events(app, sid):
    host = app.state.manager.get(sid)
    return [e for _, e in host._ring] if host else []


def test_a_crew_runs_under_the_server_and_its_approval_reaches_the_browser(env):
    from relife.crew.turns import TurnResult

    decisions = []

    async def turn(prompt, *, workspace, memory_client, can_use_tool, extra_mcp, on_event):
        on_event({"type": "tool_use", "name": "Bash", "brief": "curl -X POST https://example.com"})
        res = await can_use_tool("Bash", {"command": "curl -X POST https://example.com"}, None)
        decisions.append(type(res).__name__)
        memory_client.save("the export endpoint posts to example.com")
        on_event({"type": "result", "cost_usd": 0.4})
        return TurnResult(output="built the CSV export; tests pass", tool_calls=1, cost_usd=0.4)

    app, tc = _crew_app(env, turn=turn)
    with tc:
        crew = tc.post("/crews", json={"spec": SPEC}).json()
        r = tc.post(f"/crews/{crew['id']}/run")
        assert r.status_code == 200, r.text
        sid = r.json()["session_id"]
        assert tc.get(f"/sessions/{sid}").json()["kind"] == "crew"
        assert tc.post(f"/sessions/{sid}/messages", json={"text": "hi"}).status_code == 409
        assert tc.post(f"/crews/{crew['id']}/run").status_code == 409  # not twice
        assert tc.get(f"/crews/{crew['id']}").json()["status"] == "running"

        req = wait_for(lambda: next((e for e in _events(app, sid) if e["type"] == "approval_request"), None))
        assert req["agent"] == "builder" and "curl -X POST" in req["brief"]
        r = tc.post(f"/sessions/{sid}/approvals/{req['approval_id']}", json={"decision": "allow"})
        assert r.json() == {"resolved": True}

        wait_for(lambda: any(e["type"] in ("result", "error") for e in _events(app, sid)))
        evs = _events(app, sid)
        assert decisions == ["PermissionResultAllow"]
        assert [e["type"] for e in evs].count("result") == 1  # the member's became member_done
        assert any(e["type"] == "member_done" and e["agent"] == "builder" for e in evs)
        assert evs[0]["type"] == "user"
        rec = tc.get(f"/crews/{crew['id']}").json()
        assert rec["status"] == "done", rec["error"]
        assert rec["final_output"] == "looks good, ship it"
        assert [t["name"] for t in rec["tasks"]] == ["build", "review"]
        assert rec["session_id"] == sid
        client, _ = env
        assert [m.text for m in client.all_memories(spaces=["builder"])] == [
            "the export endpoint posts to example.com"
        ]


def test_unattended_crew_approvals_time_out_to_deny(env):
    from relife.crew.turns import TurnResult

    decisions = []

    async def turn(prompt, *, workspace, memory_client, can_use_tool, extra_mcp, on_event):
        res = await can_use_tool("Bash", {"command": "curl -X POST https://example.com"}, None)
        decisions.append(type(res).__name__)
        return TurnResult(output="could not post; said so", tool_calls=1)

    app, tc = _crew_app(env, turn=turn, approval_timeout=0.2)
    with tc:
        crew = tc.post("/crews", json={"spec": SPEC}).json()
        sid = tc.post(f"/crews/{crew['id']}/run").json()["session_id"]
        wait_for(lambda: tc.get(f"/crews/{crew['id']}").json()["status"] in ("done", "error"))
        assert decisions == ["PermissionResultDeny"]
        resolved = [e for e in _events(app, sid) if e["type"] == "approval_resolved"]
        assert resolved and resolved[0]["approved"] is False


def test_stopping_a_crew_denies_the_pending_approval_and_starts_nothing_new(env):
    """The live check caught the gap this pins: stop only refused the next
    *ReLife* turn, so a CrewAI member still ran after it (and the crew ended
    "done"). It must stop after the task in flight, whoever runs the next one."""
    pytest.importorskip("crewai")
    from relife.crew.llm import ClaudeMaxLLM
    from relife.crew.turns import TurnResult

    async def turn(prompt, *, workspace, memory_client, can_use_tool, extra_mcp, on_event):
        await can_use_tool("Bash", {"command": "curl -X POST https://example.com"}, None)
        return TurnResult(output="the post was denied; said so", tool_calls=1)

    reviewer = StubModel()
    app, tc = make_app(
        env, crew_turn=turn, crew_approval_timeout=30,
        crew_llm_factory=lambda model: ClaudeMaxLLM(ask_model=reviewer),
    )
    with tc:
        crew = tc.post("/crews", json={"spec": SPEC}).json()
        sid = tc.post(f"/crews/{crew['id']}/run").json()["session_id"]
        wait_for(lambda: any(e["type"] == "approval_request" for e in _events(app, sid)))
        assert tc.post(f"/crews/{crew['id']}/stop").json() == {"stopping": True, "denied": 1}
        rec = wait_for(lambda: (r := tc.get(f"/crews/{crew['id']}").json())["status"] in ("done", "error") and r)
        assert rec["status"] == "error" and "stopped by the user" in rec["error"]
        assert reviewer.calls == 0  # the CrewAI member never started
        assert [t["name"] for t in rec["tasks"]] == ["build"]  # what finished is kept
        wait_for(lambda: not app.state.manager.get(sid).busy)  # the host publishes last
        assert tc.post(f"/crews/{crew['id']}/stop").status_code == 409


def test_a_crew_schedule_fires_the_crew_and_records_the_run(env):
    from relife.crew.turns import TurnResult

    async def turn(prompt, *, workspace, memory_client, can_use_tool, extra_mcp, on_event):
        on_event({"type": "tool_use", "name": "Write", "brief": "export.py"})
        return TurnResult(output="built it", tool_calls=1, cost_usd=0.3)

    app, tc = _crew_app(env, turn=turn)
    with tc:
        sched = tc.post("/schedules", json={"name": "nightly", "every": "1d", "crew": SPEC}).json()
        r = tc.post(f"/schedules/{sched['id']}/run").json()
        assert r["status"] == "submitted", r
        entry = r["schedule"]["runs"][-1]
        crew_id, run_id = entry["crew_id"], entry["run_id"]

        def finished():
            s = tc.get(f"/schedules/{sched['id']}").json()
            return s["runs"][-1]["status"] not in ("submitted",) and s

        s = wait_for(finished)
        assert s["runs"][-1]["status"] == "done", s["runs"][-1]
        run = tc.get(f"/schedules/{sched['id']}/runs/{run_id}").json()
        assert run["summary"] == "looks good, ship it" and run["tool_calls"] == 1
        assert run["events"][0]["text"].startswith("[Scheduled run: nightly]")
        assert tc.get(f"/crews/{crew_id}").json()["status"] == "done"
        # The schedule's session is the crew's host: "watch" attaches to it.
        assert tc.get(f"/sessions/{s['session_id']}").json()["kind"] == "crew"


def test_crew_records_survive_windows_read_replace_races(tmp_path, monkeypatch):
    """Under the server a crew's worker rewrites its record while routes read
    it; on Windows either side can get a PermissionError for an instant (the
    full suite hit it). Both retry instead of failing."""
    import os

    from relife.crew import record as rec_mod
    from relife.crew.record import CrewRunRecord, CrewRunStore

    monkeypatch.setattr(rec_mod, "_RETRY_DELAY", 0)
    store = CrewRunStore(tmp_path)
    rec = CrewRunRecord(id=CrewRunRecord.new_id(), task="t", workspace=str(tmp_path), spec=SPEC)
    real_replace, fails = os.replace, {"n": 2}

    def flaky_replace(a, b):
        if fails["n"]:
            fails["n"] -= 1
            raise PermissionError(13, "in use")
        return real_replace(a, b)

    monkeypatch.setattr(rec_mod.os, "replace", flaky_replace)
    store.save(rec)
    assert fails["n"] == 0 and store.get(rec.id).task == "t"
    monkeypatch.setattr(rec_mod.os, "replace", lambda a, b: (_ for _ in ()).throw(PermissionError(13, "stuck")))
    with pytest.raises(PermissionError):
        store.save(rec)  # a lock that never clears still surfaces
