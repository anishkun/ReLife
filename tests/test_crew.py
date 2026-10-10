"""CrewAI integration — a real ``Crew.kickoff()`` with zero model calls.

Needs the [crewai] extra (CrewAI requires Python < 3.14), so the whole module
skips without it. The ReLife turn runner and ClaudeMaxLLM's model call are
injected stubs; everything between them — the registry/handoff, the adapter
contract, CrewAI's sequential process, context passing, memory tools, the
durable record — is the real code.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("crewai")

from relife import agents as ag  # noqa: E402
from relife.crew import runner as crew_runner  # noqa: E402
from relife.crew.agent import ReLifeAgent  # noqa: E402
from relife.crew.build import build_crew, ensure_profiles  # noqa: E402
from relife.crew.llm import ClaudeMaxLLM, split_messages  # noqa: E402
from relife.crew.memory_tools import memory_tools  # noqa: E402
from relife.crew.record import CrewRunStore  # noqa: E402
from relife.crew.spec import normalize_spec  # noqa: E402
from relife.crew.turns import TurnResult, result_from_events  # noqa: E402
from relife.memory import consolidate as consol  # noqa: E402
from relife.memory import events as ev  # noqa: E402
from relife.memory import skills as sk  # noqa: E402
from relife.memory import spaces  # noqa: E402
from relife.memory import store as store_mod  # noqa: E402
from relife.memory import workflows as wf  # noqa: E402
from relife.memory.client import LocalMemoryClient, ScopedMemoryClient  # noqa: E402
from relife.permissions import classify  # noqa: E402


@pytest.fixture
def env(tmp_path, monkeypatch):
    db = tmp_path / "relife.db"
    monkeypatch.setattr(store_mod, "_DB_PATH", db)
    monkeypatch.setattr(ev, "_DB_PATH", db)
    monkeypatch.setattr(consol, "_STATE_PATH", tmp_path / "consolidate_state.json")
    monkeypatch.setattr(sk, "_SKILLS_DIR", tmp_path / "skills")
    monkeypatch.setattr(wf, "_WORKFLOWS_DIR", tmp_path / "workflows")
    monkeypatch.setattr(spaces, "_SPACES_DIR", tmp_path / "spaces")
    # No key of any kind: the crew must still build and run.
    for k in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    store = ag.AgentStore(tmp_path / "agents.json")
    return store, LocalMemoryClient(), tmp_path


class StubModel:
    """Stands in for ``ask_model_oneshot``: answers in CrewAI's ReAct format."""

    def __init__(self, answer="Thought: done\nFinal Answer: looks good, ship it"):
        self.answer = answer
        self.calls: list[tuple[str, str]] = []

    async def __call__(self, system, prompt):
        self.calls.append((system, prompt))
        return self.answer, 0.02


class FakeRunner:
    """Stands in for a ReLife turn: records the prompt, writes to memory."""

    def __init__(self, output="built the CSV export; tests pass"):
        self.output = output
        self.prompts: list[tuple[str, str]] = []

    async def __call__(self, agent, prompt, extra_mcp):
        self.prompts.append((agent.agent_name, prompt))
        agent.memory_client.save(f"{agent.agent_name} learned: CSV export lives in export.py")
        return TurnResult(output=self.output, tool_calls=3, cost_usd=0.5)


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


# --- ClaudeMaxLLM -------------------------------------------------------------------
def test_split_messages():
    assert split_messages("hi") == ("", "hi")
    sys_, prompt = split_messages([{"role": "system", "content": "be terse"}, {"role": "user", "content": "go"}])
    assert (sys_, prompt) == ("be terse", "go")
    _, convo = split_messages([
        {"role": "user", "content": [{"type": "text", "text": "q1"}]},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
    ])
    assert convo == "[user]\nq1\n\n[assistant]\na1\n\n[user]\nq2\n\n[assistant]\n"


def test_claude_max_llm_truncates_at_stop_words_and_tracks_cost():
    stub = StubModel("Thought: x\nAction: search\nObservation: invented result")
    llm = ClaudeMaxLLM(ask_model=stub, stop=["\nObservation:"])
    out = llm.call([{"role": "user", "content": "find it"}])
    assert out == "Thought: x\nAction: search"  # no hallucinated observation
    assert llm.calls == 1 and llm.cost_usd == 0.02
    assert llm.supports_function_calling() is False
    assert llm.get_context_window_size() >= 100_000


# --- the adapter --------------------------------------------------------------------
def test_relife_agent_implements_the_adapter_contract(env, tmp_path):
    store, client, _ = env
    ag.create_agent(store, client, "builder")
    a = ReLifeAgent(
        role="Engineer", goal="build", agent_name="builder", workspace=tmp_path,
        memory_client=ScopedMemoryClient(client, store.require("builder").scope()),
    )
    assert a.get_delegation_tools([]) == [] and a.get_mcp_tools([]) == []
    from crewai import Task

    t = Task(description="Add a flag.", expected_output="done", agent=a)
    p = a.task_prompt(t, "previous output text")
    assert "**Engineer**" in p and "Add a flag." in p
    assert "<<<CONTEXT\nprevious output text\nCONTEXT>>>" in p
    # The episode the Stop hook saves names the task, not the crew role (a live
    # smoke caught every crew episode reading "You are working on a crew as …").
    from relife.hooks import _episode_text

    assert _episode_text(p, ["Write", "Bash"]).startswith("Task: Add a flag.")


def test_failed_turn_without_output_raises(env, tmp_path):
    store, client, _ = env
    ag.create_agent(store, client, "builder")

    async def failing(agent, prompt, extra):
        return TurnResult(output="", error="CLI session limit reached")

    a = ReLifeAgent(role="Engineer", goal="build", agent_name="builder", workspace=tmp_path,
                    memory_client=client, runner=failing)
    from crewai import Task

    with pytest.raises(RuntimeError, match="session limit"):
        a.execute_task(Task(description="x", expected_output="y", agent=a))


def test_crew_tools_are_not_trusted(tmp_path):
    decision, _ = classify("mcp__crew_tools__web_search", {"q": "x"}, tmp_path)
    assert decision == "ask"


def test_result_from_events_keeps_the_full_closing_text():
    events = [
        {"type": "text", "text": "starting"},
        {"type": "tool_use", "name": "Bash", "brief": "pytest"},
        {"type": "tool_result", "brief": "ok"},
        {"type": "text", "text": "All done: " + "x" * 3000},
        {"type": "result", "cost_usd": 0.4},
    ]
    r = result_from_events(events)
    assert r.output.startswith("All done:") and len(r.output) > 2000  # not the 2000-char summary
    assert (r.tool_calls, r.cost_usd) == (1, 0.4)


# --- memory tools for non-Claude members ------------------------------------------------
def test_memory_tools_write_into_the_agents_space(env):
    store, client, _ = env
    ag.create_agent(store, client, "critic", runtime="llm")
    tools = {t.name: t for t in memory_tools(ScopedMemoryClient(client, store.require("critic").scope()))}
    assert "memory_dream" not in tools and "memory_context" in tools
    assert "Saved memory" in tools["memory_save"].run(text="reviews must cite the diff")
    [m] = client.all_memories(spaces=["critic"])
    assert m.source == "critic"
    assert "cite the diff" in tools["memory_recall"].run(query="reviews cite diff")


def test_llm_agent_builds_without_the_checkpointing_warning(env, recwarn):
    # The step callback is a closure (it carries the agent's scoped client), which
    # CrewAI flags as unserializable for checkpointing — noise for ReLife, which
    # never checkpoints a crew. The live smoke printed it; it must stay quiet.
    from relife.crew.native import llm_agent
    from relife.crew.spec import AgentSpec

    store, client, _ = env
    ag.create_agent(store, client, "critic", runtime="llm")
    scoped = ScopedMemoryClient(client, store.require("critic").scope())
    spec = AgentSpec(name="critic", role="Reviewer", goal="review", backstory="", runtime="llm",
                     llm="claude-max", inherit=[], fork=None)
    agent = llm_agent(spec, memory_client=scoped, llm=ClaudeMaxLLM(ask_model=StubModel()), task_id="t1")
    assert agent.step_callback is not None
    assert not [w for w in recwarn if "checkpointing" in str(w.message)]


# --- the whole crew ------------------------------------------------------------------------
def test_crew_kickoff_end_to_end(env, tmp_path):
    store, client, _ = env
    spec = normalize_spec(SPEC)
    notes = ensure_profiles(spec, store, client)
    assert notes[0] == "new agent builder" and "reads builder" in notes[1]
    assert store.require("critic").scope().read == ("critic", "builder", "default")

    stub, runner, outcomes = StubModel(), FakeRunner(), {}
    crew = build_crew(
        spec, store=store, client=client, workspace=tmp_path, run_id="r1",
        runner=runner,
        llm_factory=lambda model: ClaudeMaxLLM(ask_model=stub),
        on_outcome=lambda task, agent, res: outcomes.setdefault(task.name, res),
    )
    out = crew.kickoff()
    assert out.raw == "looks good, ship it"
    assert [n for n, _ in runner.prompts] == ["builder"]
    assert outcomes["build"].tool_calls == 3
    # The reviewer saw the builder's output as context.
    assert any("built the CSV export" in prompt for _, prompt in stub.calls)
    # What builder learned stays in builder's space — readable by critic, which inherits it.
    [m] = client.all_memories(spaces=["builder"])
    assert m.source == "builder"
    critic = ScopedMemoryClient(client, store.require("critic").scope())
    assert critic.recall("CSV export lives")[0].space == "builder"


def test_run_crew_plans_confirms_and_records(env, tmp_path):
    store, client, _ = env
    planner = StubModel(json.dumps(SPEC))
    stub, runner = StubModel(), FakeRunner()
    lines: list[str] = []
    records = CrewRunStore(tmp_path / "crews")

    def go(**kw):
        return crew_runner.run_crew(
            "add CSV export", workspace=tmp_path, echo=lines.append,
            ask_model=planner, runner=runner,
            llm_factory=lambda model: ClaudeMaxLLM(ask_model=stub),
            store=store, client=client, records=records, quiet_events=True, **kw,
        )

    planned = go(plan_only=True)
    assert planned.status == "planned" and not runner.prompts
    assert any("builder — Engineer" in ln for ln in lines)

    cancelled = go(confirm=lambda q: False)
    assert cancelled.status == "cancelled" and not runner.prompts

    rec = go(yes=True)
    assert rec.status == "done", rec.error
    assert [t.name for t in rec.tasks] == ["build", "review"]
    assert rec.outcome("build").tool_calls == 3 and rec.outcome("review").runtime == "llm"
    assert rec.final_output == "looks good, ship it"
    again = records.get(rec.id)
    assert again.status == "done" and round(again.cost_usd, 2) == round(rec.cost_usd, 2)
    assert (tmp_path / "crews" / rec.id).is_dir()  # the crew's own workspace
    assert records.get("../../etc") is None


def test_a_failing_member_is_recorded_not_raised(env, tmp_path):
    store, client, _ = env

    async def boom(agent, prompt, extra):
        raise RuntimeError("disk full")

    rec = crew_runner.run_crew(
        None, workspace=tmp_path, spec_raw=SPEC, yes=True, echo=lambda s: None,
        runner=boom, llm_factory=lambda m: ClaudeMaxLLM(ask_model=StubModel()),
        store=store, client=client, records=CrewRunStore(tmp_path / "crews"),
    )
    assert rec.status == "error" and "disk full" in rec.error
