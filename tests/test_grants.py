"""Pre-authorized grants: a schedule's narrow permission to act unattended.

Four layers, none of which call a model:

1. **Policy** — ``normalize_grants`` / ``grant_allows`` are pure: which calls a
   grant covers, and every way it must fail closed.
2. **Callback** — ``make_approval_callback`` consults ``preauthorize`` before the
   broker, and only for ask-cases.
3. **Session** — grants belong to one turn, are capped per turn, and every use
   is published as ``approval_auto``.
4. **Scheduler / runs / HTTP** — a schedule hands its grants to its own turn,
   tells the agent about them, and the run record lists what was done.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import anyio
import pytest

from relife import config
from relife.permissions import (
    classify,
    describe_grant,
    grant_allows,
    make_approval_callback,
    normalize_grants,
)

ME = "me@example.com"
GMAIL_SEND = "mcp__claude_ai_Gmail__gmail_send_message"
CAL_CREATE = "mcp__claude_ai_Google_Calendar__gcal_create_event"
EMAIL = [{"kind": "email", "addresses": [ME]}]
CAL = [{"kind": "calendar", "addresses": []}]


# =============================================================================
# Layer 1 — pure policy
# =============================================================================
def test_normalize_grants_canonicalizes():
    got = normalize_grants(
        [{"kind": "Calendar"}, {"kind": "email", "addresses": "Me@Example.com, me@example.com; b@x.org"}]
    )
    assert got == [
        {"kind": "email", "addresses": ["me@example.com", "b@x.org"]},
        {"kind": "calendar", "addresses": []},
    ]
    assert normalize_grants(None) == [] and normalize_grants([]) == []


@pytest.mark.parametrize(
    "bad",
    [
        "email",
        [{"kind": "shell"}],
        [{"kind": "email"}],  # email must say who
        [{"kind": "email", "addresses": ["not-an-address"]}],
        [{"kind": "email", "addresses": [ME]}, {"kind": "email", "addresses": [ME]}],
        [{"kind": "calendar", "addresses": [f"a{i}@x.org" for i in range(6)]}],
        ["email"],
    ],
)
def test_normalize_grants_rejects(bad):
    with pytest.raises(ValueError):
        normalize_grants(bad)


def test_email_to_me_is_preauthorized():
    for inp in (
        {"to": ME, "subject": "digest", "body": "3 new mails"},
        {"to": [f"Me <{ME.upper()}>"], "body": "x"},
        {"message": {"to": ME, "body": "nested"}},
    ):
        assert classify(GMAIL_SEND, inp, Path("."))[0] == "ask"  # still an ask-case…
        assert grant_allows(EMAIL, GMAIL_SEND, inp) == f"send email only to {ME}"  # …that the grant covers
    assert grant_allows(EMAIL, "mcp__claude_ai_Gmail__create_draft", {"to": ME})


def test_addresses_in_content_are_not_recipients():
    """A digest quoting who wrote in must not look like mailing them."""
    inp = {"to": ME, "subject": "from boss@corp.com", "body": "boss@corp.com asked about Q3"}
    assert grant_allows(EMAIL, GMAIL_SEND, inp)


@pytest.mark.parametrize(
    "inp",
    [
        {"to": "someone@else.com", "body": "x"},
        {"to": ME, "cc": "someone@else.com"},
        {"to": [ME, "someone@else.com"]},
        {"to": f"{ME}, all-staff"},  # a recipient that isn't an address
        {"recipient_list": "someone@else.com", "to": ME},  # unknown key still counts
        {"thread_id": "abc", "body": "thanks!"},  # recipient we can't see
        {"raw": "VG86IHNvbWVvbmVAZWxzZS5jb20="},  # base64 MIME: opaque
        {},
    ],
)
def test_email_grant_fails_closed(inp):
    assert grant_allows(EMAIL, GMAIL_SEND, inp) is None


@pytest.mark.parametrize(
    "tool",
    [
        "mcp__claude_ai_Gmail__delete_draft",  # matches 'draft' but destructive
        "mcp__claude_ai_Gmail__batch_modify_labels",
        "mcp__claude_ai_Gmail__gmail_search_messages",  # not a send
        "mcp__claude_ai_Google_Drive__share_file",
        "mcp__claude_ai_Google_Calendar__gcal_create_event",  # email grant ≠ calendar
        "mcp__other_server__send_message",  # not a connector
        "Bash",
        "Write",
    ],
)
def test_email_grant_covers_only_email_sends(tool):
    assert grant_allows(EMAIL, tool, {"to": ME, "command": f"mail {ME}", "file_path": "/etc/x"}) is None


def test_calendar_grant_allows_events_without_outside_guests():
    both = CAL + [{"kind": "email", "addresses": [ME]}]
    assert grant_allows(CAL, CAL_CREATE, {"summary": "focus", "start": "09:00"}) == describe_grant(CAL[0])
    assert grant_allows(CAL, CAL_CREATE, {"summary": "1:1", "attendees": [ME]}) is None  # no guests allowed
    guests = [{"kind": "calendar", "addresses": [ME]}]
    assert grant_allows(guests, CAL_CREATE, {"attendees": [{"email": ME}], "description": "x@y.com"})
    assert grant_allows(guests, CAL_CREATE, {"attendees": [ME, "x@y.com"]}) is None
    for op in ("gcal_update_event", "gcal_delete_event", "gcal_create_calendar", "gcal_list_events"):
        assert grant_allows(both, f"mcp__claude_ai_Google_Calendar__{op}", {}) is None, op
    assert grant_allows(CAL, "mcp__claude_ai_Google_Calendar__quick_add_event", {"text": "lunch"})


def test_no_grants_no_widening():
    assert grant_allows([], GMAIL_SEND, {"to": ME}) is None


# =============================================================================
# Layer 2 — the approval callback
# =============================================================================
class Broker:
    def __init__(self, answer: bool = False) -> None:
        self.asked: list[str] = []
        self.answer = answer

    async def request(self, tool_name, tool_input, reason, *, timeout):
        self.asked.append(tool_name)
        return self.answer


def test_callback_consults_preauthorize_before_the_broker(tmp_path):
    from claude_agent_sdk import PermissionResultAllow, PermissionResultDeny

    seen: list[str] = []

    async def pre(tool_name, tool_input):
        seen.append(tool_name)
        return grant_allows(EMAIL, tool_name, tool_input)

    broker = Broker()
    cb = make_approval_callback(tmp_path, broker, timeout=1.0, preauthorize=pre)

    async def flow():
        assert isinstance(await cb("Read", {}, None), PermissionResultAllow)
        assert seen == []  # an allow never reaches the grant check
        assert isinstance(await cb(GMAIL_SEND, {"to": ME}, None), PermissionResultAllow)
        assert broker.asked == []
        assert isinstance(await cb(GMAIL_SEND, {"to": "x@y.com"}, None), PermissionResultDeny)
        assert broker.asked == [GMAIL_SEND]  # no grant ⇒ the normal card

    anyio.run(flow)


# =============================================================================
# Layer 3 — the session: per-turn, capped, visible
# =============================================================================
class FakeClient:
    """Stands in for ClaudeSDKClient: each query runs the scripted tool calls
    through the session's permission callback."""

    def __init__(self, session, calls: dict[str, list[tuple[str, dict]]]) -> None:
        self.session = session
        self.calls = calls
        self.cb = make_approval_callback(
            session.workspace, session._broker, timeout=0.01, preauthorize=session._preauthorize
        )
        self.outcomes: dict[str, list[str]] = {}
        self._turn = ""

    async def query(self, text: str) -> None:
        self._turn = text

    async def receive_response(self):
        out = self.outcomes.setdefault(self._turn, [])
        for tool, inp in self.calls.get(self._turn, []):
            res = await self.cb(tool, inp, None)
            out.append(type(res).__name__)
        return
        yield  # pragma: no cover - makes this an async generator


def _session(tmp_path, monkeypatch, calls):
    from relife.server import session as session_mod

    async def no_consolidate():
        return None

    monkeypatch.setattr(session_mod, "maybe_consolidate_off_loop", no_consolidate)
    s = session_mod.AgentSession(tmp_path)
    s._client = FakeClient(s, calls)
    return s


async def _drain(s) -> None:
    for _ in range(200):
        if not s.busy:
            return
        await asyncio.sleep(0.005)
    raise AssertionError("turn never finished")


def test_grants_apply_to_their_turn_only_and_are_capped(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "AGENT_GRANT_MAX_USES", 2)
    send = (GMAIL_SEND, {"to": ME, "body": "digest"})
    s = _session(tmp_path, monkeypatch, {"scheduled": [send, send, send], "typed": [send]})

    async def flow():
        worker = asyncio.create_task(s._run())
        try:
            await s.submit("scheduled", grants=EMAIL)
            await _drain(s)
            await s.submit("typed")  # same session, no grants
            await _drain(s)
        finally:
            worker.cancel()

    anyio.run(flow)
    got = s._client.outcomes
    allow, deny = "PermissionResultAllow", "PermissionResultDeny"
    assert got["scheduled"] == [allow, allow, deny]  # third use is past the cap → card → timeout
    assert got["typed"] == [deny]  # grants did not leak into the next turn
    autos = [ev for _, ev in s._ring if ev["type"] == "approval_auto"]
    assert len(autos) == 2 and autos[0]["tool"] == GMAIL_SEND and autos[0]["grant"] == f"send email only to {ME}"
    assert ME in autos[0]["brief"]


# =============================================================================
# Layer 4 — scheduler, run records, HTTP
# =============================================================================
def test_scheduler_passes_grants_and_tells_the_agent(tmp_path):
    from test_scheduler import FakeManager

    from relife.server.runs import RunStore
    from relife.server.scheduler import Scheduler
    from relife.server.schedules import Schedule, ScheduleStore

    store = ScheduleStore(tmp_path / "s.json")
    manager = FakeManager()
    scheduler = Scheduler(store, manager, workspace_root=tmp_path, runs=RunStore(tmp_path / "runs"))
    g = store.add(Schedule.new(name="g", task="digest", spec={"every": "1h"}, grants=EMAIL, now=0.0))
    plain = store.add(Schedule.new(name="p", task="t", spec={"every": "1h"}, now=0.0))

    async def flow():
        await scheduler.tick(3600.0)
        for task in list(scheduler._recorders):
            task.cancel()

    anyio.run(flow)
    sg, sp = manager.sessions[g.session_id], manager.sessions[plain.session_id]
    assert sg.grants == [EMAIL] and sp.grants == [[]]
    pre, task = sg.submitted[0].split("\n\n", 1)
    assert task == "digest" and f"send email only to {ME}" in pre and "pre-approved" in pre
    assert "pre-approved" not in sp.submitted[0]


def test_run_record_lists_what_was_done_on_your_behalf():
    from relife.server.runs import RunRecord, summarize_events

    events: list[dict[str, Any]] = [
        {"type": "user", "text": "p"},
        {"type": "tool_use", "name": GMAIL_SEND, "brief": ME},
        {"type": "approval_auto", "tool": GMAIL_SEND, "grant": f"send email only to {ME}", "brief": f"to={ME}"},
        {"type": "tool_result", "brief": "sent"},
        {"type": "text", "text": "Emailed you the digest."},
        {"type": "result", "cost_usd": 0.01},
    ]
    info = summarize_events(events)
    assert info["acted"] == [{"tool": GMAIL_SEND, "brief": f"to={ME}", "grant": f"send email only to {ME}"}]
    assert info["denied"] == [] and info["summary"] == "Emailed you the digest."
    rec = RunRecord(schedule_id="s", run_id="20260924-090000-000", started_at=0.0)
    rec.finish("done", events, now=1.0)
    assert len(rec.acted) == 1
    old = rec.to_dict(with_events=True)
    old.pop("acted")
    assert RunRecord(**old).acted == []  # records written before grants still load


def test_grants_over_http(tmp_path):

    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from relife.server.app import create_app
    from relife.server.schedules import ScheduleStore

    root = tmp_path / "ws"
    root.mkdir()
    app = create_app(
        token=None,
        session_factory=lambda ws: None,
        workspace_root=root,
        reap=False,
        schedules_path=tmp_path / "schedules.json",
        runs_dir=tmp_path / "runs",
        run_scheduler=False,
    )
    tc = TestClient(app)
    base = {"name": "digest", "task": "email me my inbox", "every": "1h"}
    r = tc.post("/schedules", json={**base, "grants": [{"kind": "email", "addresses": [ME]}]})
    assert r.status_code == 201, r.text
    sc = r.json()
    assert sc["grants"] == EMAIL and sc["grants_text"] == [f"send email only to {ME}"]
    assert tc.post("/schedules", json={**base, "grants": [{"kind": "shell"}]}).status_code == 400
    assert tc.post("/schedules", json={**base, "grants": [{"kind": "email"}]}).status_code == 400

    r = tc.patch(f"/schedules/{sc['id']}", json={"grants": EMAIL + CAL})
    assert r.status_code == 200 and [g["kind"] for g in r.json()["grants"]] == ["email", "calendar"]
    assert tc.patch(f"/schedules/{sc['id']}", json={"grants": [{"kind": "email", "addresses": ["nope"]}]}).status_code == 400
    assert tc.patch(f"/schedules/{sc['id']}", json={"grants": []}).json()["grants"] == []
    assert ScheduleStore(tmp_path / "schedules.json").get(sc["id"]).grants == []


def test_hand_edited_invalid_grant_is_dropped_on_load(tmp_path):
    import json

    from relife.server.schedules import Schedule, ScheduleStore

    store = ScheduleStore(tmp_path / "s.json")
    s = store.add(Schedule.new(name="n", task="t", spec={"every": "1h"}, grants=EMAIL, now=0.0))
    data = json.loads((tmp_path / "s.json").read_text(encoding="utf-8"))
    data["schedules"][0]["grants"] = [{"kind": "anything", "addresses": ["*"]}]
    (tmp_path / "s.json").write_text(json.dumps(data), encoding="utf-8")
    assert ScheduleStore(tmp_path / "s.json").get(s.id).grants == []
