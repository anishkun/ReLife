"""The shared Max budget: one spend ledger for everything ReLife starts, and an
optional daily cap that refuses new work (``relife/budget.py``).

No model calls: costs are fed in directly, or through a scripted SDK client
whose results carry the CLI's *cumulative* ``total_cost_usd`` — the detail that
made per-turn costs in long-lived sessions wrong before the ledger existed.
"""

from __future__ import annotations

import asyncio
import math
from pathlib import Path

import pytest
from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock

from relife import budget, config


def _cap(monkeypatch, usd):
    monkeypatch.setattr(config, "DAILY_BUDGET_USD", usd)


# --- the ledger ------------------------------------------------------------------------------
def test_record_ignores_junk_and_sums_by_kind():
    assert budget.record(None, "cli") == 0
    assert budget.record(math.nan, "cli") == 0
    assert budget.record(-1, "cli") == 0
    assert budget.record("abc", "cli") == 0
    budget.record(0.25, "cli")
    budget.record(0.5, "crew", "c1")
    budget.record("0.25", "crew")
    assert budget.spent() == pytest.approx(1.0)
    assert budget.by_kind() == {"crew": pytest.approx(0.75), "cli": pytest.approx(0.25)}


def test_spend_is_per_day():
    budget.record(2.0, "cli", now=0.0)  # 1970
    budget.record(0.1, "cli")
    assert budget.spent() == pytest.approx(0.1)


def test_a_meter_turns_running_totals_into_turn_costs():
    m = budget.CostMeter("server", "s1")
    assert m.observe(0.10) == pytest.approx(0.10)
    assert m.observe(0.25) == pytest.approx(0.15)  # the CLI's total is cumulative
    assert m.observe(None) is None
    assert m.observe(0.05) == pytest.approx(0.05)  # lower ⇒ a new process
    assert budget.spent() == pytest.approx(0.30)


def test_the_cap_is_off_until_set(monkeypatch):
    budget.record(100.0, "cli")
    budget.check()  # no cap: never refuses
    assert not budget.exceeded()
    assert budget.status()["cap_usd"] is None
    _cap(monkeypatch, 50.0)
    assert budget.exceeded()
    with pytest.raises(budget.BudgetExceeded, match=r"\$100.00 of \$50.00"):
        budget.check("this")
    s = budget.status()
    assert s["exceeded"] and s["remaining_usd"] == 0


def test_a_broken_ledger_never_fails_the_work(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "SPEND_PATH", tmp_path)  # a directory: unopenable
    assert budget.record(1.0, "cli") == 0
    assert budget.spent() == 0
    budget.check()


# --- server sessions publish each turn's own cost ------------------------------------------
class _ScriptedClient:
    """Stands in for ClaudeSDKClient: each query answers with text and a
    result carrying the CLI's running total."""

    def __init__(self, totals):
        self.totals = list(totals)

    async def query(self, prompt):
        self._next = self.totals.pop(0)

    async def receive_response(self):
        yield AssistantMessage(content=[TextBlock(text="done")], model="m")
        yield ResultMessage(
            subtype="success", duration_ms=1, duration_api_ms=1, is_error=False,
            num_turns=1, session_id="cli-sess", total_cost_usd=self._next,
        )


def test_a_server_session_reports_per_turn_cost_not_the_running_total(tmp_path, monkeypatch):
    from relife.server import session as sess_mod

    async def no_upkeep():
        return None

    monkeypatch.setattr(sess_mod, "maybe_consolidate_off_loop", no_upkeep)

    async def flow():
        s = sess_mod.AgentSession(tmp_path)
        s._client = _ScriptedClient([0.10, 0.30, 0.35])
        s._worker = asyncio.create_task(s._run())
        q, _ = s.subscribe()
        costs = []
        for text in ("one", "two", "three"):
            await s.submit(text)
            while True:
                _, ev = await asyncio.wait_for(q.get(), 5)
                if ev["type"] == "result":
                    costs.append(ev["cost_usd"])
                    break
        s._worker.cancel()
        return costs

    costs = asyncio.run(flow())
    assert costs == [pytest.approx(0.10), pytest.approx(0.20), pytest.approx(0.05)]
    assert budget.spent() == pytest.approx(0.35)  # not 0.10 + 0.30 + 0.35


# --- enforcement --------------------------------------------------------------------------------
@pytest.fixture
def server(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from relife.server.app import create_app

    submitted = []

    class Session:
        def __init__(self, workspace: Path):
            self.id, self.workspace = "s1", workspace

        async def start(self):
            pass

        async def aclose(self):
            pass

        async def submit(self, text, *, grants=None):
            submitted.append(text)

    app = create_app(
        session_factory=Session, schedules_path=tmp_path / "schedules.json",
        runs_dir=tmp_path / "runs", run_scheduler=False, reap=False,
        workspace_root=tmp_path / "ws", agents_path=tmp_path / "agents.json",
        crews_dir=tmp_path / "crews",
    )
    return TestClient(app), submitted


def test_the_server_refuses_new_work_over_the_cap(server, monkeypatch):
    tc, submitted = server
    sid = tc.post("/sessions", json={}).json()["session_id"]
    assert tc.post(f"/sessions/{sid}/messages", json={"text": "hi"}).status_code == 200
    budget.record(5.0, "server")
    _cap(monkeypatch, 4.0)
    r = tc.post(f"/sessions/{sid}/messages", json={"text": "again"})
    assert r.status_code == 429 and "daily budget reached" in r.json()["detail"]
    assert submitted == ["hi"]
    r = tc.post("/crews", json={"task": "plan something"})
    assert r.status_code == 429  # planning is a model call too
    b = tc.get("/budget").json()
    assert b["exceeded"] and b["spent_usd"] == 5.0 and b["by_kind"] == {"server": 5.0}


def test_a_schedule_over_the_cap_is_skipped_not_run(tmp_path, monkeypatch):
    from relife.server.runs import RunStore
    from relife.server.scheduler import Scheduler
    from relife.server.schedules import Schedule, ScheduleStore

    class Manager:
        created = []

        async def create(self, workspace):
            self.created.append(workspace)
            raise AssertionError("must not start a session over budget")

        def get(self, sid):
            return None

    store = ScheduleStore(tmp_path / "schedules.json")
    s = store.add(Schedule.new(name="n", task="do it", spec={"every": "1h"}, now=0.0))
    sch = Scheduler(store, Manager(), workspace_root=tmp_path, runs=RunStore(tmp_path / "runs"))
    budget.record(3.0, "server")
    _cap(monkeypatch, 1.0)
    status = asyncio.run(sch.fire(s, now=10.0))
    assert status == "skipped: daily budget reached"
    assert s.runs[-1]["status"] == status and s.next_run_at > 10.0  # advanced, no retry storm


def test_the_cli_refuses_to_start_over_the_cap(monkeypatch, tmp_path):
    from typer.testing import CliRunner

    from relife import cli

    def boom(*a, **kw):
        raise AssertionError("no model call over budget")

    monkeypatch.setattr(cli, "run_task", boom)
    budget.record(2.0, "cli")
    _cap(monkeypatch, 1.0)
    out = CliRunner().invoke(cli.app, ["do", "anything", "-w", str(tmp_path)])
    assert out.exit_code == 1 and "daily budget reached" in out.output


def test_doctor_reports_spend_and_the_cap():
    from relife.doctor import Probes, _check_budget

    def probes(spend):
        return Probes(
            which=lambda n: None, run=lambda c: (1, ""), env={}, python_version=(3, 12, 0),
            import_ok=lambda m: True, bundled_cli=None, fts5_ok=lambda: True,
            data_dir=Path("."), data_dir_writable=lambda p: True, memory_url=None,
            http_get=lambda u: (200, "{}"), spend=spend,
        )

    budget.record(1.5, "crew")
    c = _check_budget(probes(budget.status()))
    assert c.status == "ok" and "$1.50 today" in c.detail and "no daily cap" in c.detail
    over = {**budget.status(), "cap_usd": 1.0, "exceeded": True}
    assert _check_budget(probes(over)).status == "warn"
    assert _check_budget(probes(None)).status == "skip"
