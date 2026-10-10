"""``relife crew`` — plan a team for a task, show it, run it, keep the outcome.

    task ─► plan (Claude via the CLI, or the user's spec file) ─► validated CrewSpec
         ─► record (data/crews/<id>/) ─► show plan ─► confirm ─► agents + memory
         ─► CrewAI kickoff (ReLife agents + CrewAI agents on other models)
         ─► per-task outcomes + final output recorded ─► memory consolidation

Everything that spends budget is behind the confirmation (unless ``--yes``):
planning is one tool-less call; ``--plan-only`` stops there. The crew works in
its own directory, ``<workspace>/crews/<id>/`` — the auto-allow radius of its
ReLife agents. CrewAI's anonymous telemetry is off unless the user turned it on
(ReLife is local-first).
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Callable

from ..agents import AgentStore
from .planner import plan_crew
from .record import CrewRunRecord, CrewRunStore, TaskOutcome
from .spec import CrewSpec, normalize_spec
from .turns import TurnResult, run_sync

Echo = Callable[[str], None]


def _quiet_telemetry() -> None:
    os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
    os.environ.setdefault("CREWAI_DISABLE_TRACKING", "true")
    os.environ.setdefault("OTEL_SDK_DISABLED", "true")


def roster(store: AgentStore, client: Any) -> list[dict[str, Any]]:
    """Existing agents as the planner sees them (name, runtime, what they hold)."""
    counts = client.spaces()
    return [
        {
            "name": a.name,
            "runtime": a.runtime,
            "model": a.model,
            "description": a.description,
            "memories": counts.get(a.own_space, {}).get("memories", 0),
        }
        for a in store.list()
    ]


def describe_plan(spec: CrewSpec) -> list[str]:
    """The plan as the user reviews it before anything runs."""
    lines = [f"goal: {spec.goal}", f"process: {spec.process}", "agents:"]
    for a in spec.agents:
        engine = "ReLife agent (Claude + tools)" if a.runtime == "relife" else f"CrewAI agent on {a.llm} (no tools)"
        lineage = []
        if a.inherit:
            lineage.append("inherits " + ", ".join(a.inherit))
        if a.fork:
            lineage.append(f"forks {a.fork}")
        lines.append(f"  • {a.name} — {a.role}  [{engine}]" + (f"  ({'; '.join(lineage)})" if lineage else ""))
    lines.append("tasks:")
    for i, t in enumerate(spec.tasks, 1):
        ctx = f"  ← {', '.join(t.context)}" if t.context else ""
        desc = " ".join(t.description.split())
        lines.append(f"  {i}. {t.name} → {t.agent}{ctx}: {desc[:110]}{'…' if len(desc) > 110 else ''}")
    return lines


def run_crew(
    task: str | None,
    *,
    workspace: Path,
    spec_raw: Any = None,
    plan_only: bool = False,
    yes: bool = False,
    echo: Echo = print,
    confirm: Callable[[str], bool] | None = None,
    ask_model: Any = None,
    runner: Any = None,
    llm_factory: Any = None,
    manager_llm: Any = None,
    store: AgentStore | None = None,
    client: Any = None,
    records: CrewRunStore | None = None,
    quiet_events: bool = False,
) -> CrewRunRecord:
    """Plan (or load), confirm, and run a crew; return its durable record."""
    _quiet_telemetry()
    from ..memory.client import default_client

    store = store or AgentStore()
    client = client or default_client()
    records = records or CrewRunStore()
    existing = {a.name for a in store.list()}

    if spec_raw is not None:
        # The user's own file: any model string is theirs to choose.
        spec = normalize_spec(spec_raw, existing_agents=existing)
        notes: list[str] = []
        plan_cost = 0.0
    else:
        if not task or not task.strip():
            raise ValueError("give a task (or --spec FILE)")
        echo("planning the crew…")
        spec, notes, plan_cost = run_sync(
            lambda: plan_crew(task, roster=roster(store, client), ask_model=ask_model)
        )

    run_id = CrewRunRecord.new_id()
    crew_ws = (workspace / "crews" / run_id).resolve()
    record = CrewRunRecord(
        id=run_id,
        task=(task or spec.goal).strip(),
        workspace=str(crew_ws),
        spec=spec.to_dict(),
        plan_notes=notes,
        plan_cost_usd=plan_cost,
    )
    records.save(record)
    for n in notes:
        echo(f"  · {n}")
    for line in describe_plan(spec):
        echo(line)
    if plan_only:
        echo(f"plan only — saved as crew {run_id}")
        return record
    if not yes and confirm is not None and not confirm("run this crew?"):
        record.status = "cancelled"
        record.finished_at = time.time()
        records.save(record)
        echo("left alone")
        return record

    from .build import build_crew, ensure_profiles

    crew_ws.mkdir(parents=True, exist_ok=True)
    for n in ensure_profiles(spec, store, client):
        echo(f"  · {n}")

    relife_results: dict[str, TurnResult] = {}

    def on_outcome(task_obj: Any, agent: Any, result: TurnResult) -> None:
        relife_results[getattr(task_obj, "name", "") or ""] = result
        if result.denied:
            echo(f"  ! {agent.agent_name}: {len(result.denied)} action(s) denied")

    def on_event(agent_name: str, ev: dict[str, Any]) -> None:
        if quiet_events:
            return
        if ev.get("type") == "tool_use":
            echo(f"  [{agent_name}] → {ev.get('name')} {ev.get('brief', '')}")

    def on_step(agent_name: str, step: Any) -> None:
        if quiet_events:
            return
        tool = getattr(step, "tool", None)
        if tool:
            echo(f"  [{agent_name}] → {tool}")

    record.status = "running"
    records.save(record)
    try:
        crew = build_crew(
            spec,
            store=store,
            client=client,
            workspace=crew_ws,
            run_id=run_id,
            runner=runner,
            llm_factory=llm_factory,
            manager_llm=manager_llm,
            on_event=on_event,
            on_outcome=on_outcome,
            on_step=on_step,
        )
        echo(f"running crew {run_id} in {crew_ws}")
        out = crew.kickoff()
        record.final_output = str(getattr(out, "raw", out) or "")
        outputs = list(getattr(out, "tasks_output", []) or [])
        for i, t in enumerate(spec.tasks):
            agent_spec = spec.agent(t.agent)
            raw = str(getattr(outputs[i], "raw", "")) if i < len(outputs) else ""
            r = relife_results.get(t.name)
            record.tasks.append(
                TaskOutcome(
                    name=t.name,
                    agent=t.agent,
                    runtime=agent_spec.runtime,
                    output=raw or (r.output if r else ""),
                    tool_calls=r.tool_calls if r else 0,
                    cost_usd=r.cost_usd if r else None,
                    error=r.error if r else None,
                    denied=r.denied if r else [],
                )
            )
        record.status = "done"
    except KeyboardInterrupt:
        record.status = "interrupted"
        raise
    except Exception as e:  # noqa: BLE001 — the record says what happened
        record.status = "error"
        record.error = f"{type(e).__name__}: {e}"
        for name, r in relife_results.items():  # keep whatever finished
            if record.outcome(name) is None:
                t = next((x for x in spec.tasks if x.name == name), None)
                record.tasks.append(
                    TaskOutcome(
                        name=name, agent=t.agent if t else "?", runtime="relife",
                        output=r.output, tool_calls=r.tool_calls, cost_usd=r.cost_usd,
                        error=r.error, denied=r.denied,
                    )
                )
    finally:
        record.finished_at = time.time()
        records.save(record)
        # Brain upkeep once for the whole crew (deterministic, throttled, per space).
        from ..agent import maybe_consolidate

        maybe_consolidate()
    return record
