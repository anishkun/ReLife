"""Crew specs — the plan a crew runs, as validated plain data.

A spec says who is on the team and what each does: agents (a ReLife agent, or a
CrewAI agent on another model) and an ordered list of tasks, each assigned to
one agent and able to read the outputs of *earlier* tasks. It comes from the
planner (a model's JSON) or from a file the user wrote, so it is untrusted
input either way: :func:`normalize_spec` checks every field, refuses forward or
circular task references, unknown agents and lineage, and caps the team size
(every member spends budget).

Pure — no CrewAI import — so it is unit-tested on any Python, and a spec can be
shown (``relife crew --plan-only``) without the optional extra installed.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .. import config
from ..agents import MAX_INHERITS, validate_agent_name

RUNTIMES = ("relife", "llm")
PROCESSES = ("sequential", "hierarchical")
MAX_TEXT = 4000  # per free-text field; the planner can ramble


@dataclass
class AgentSpec:
    name: str
    role: str
    goal: str
    backstory: str = ""
    runtime: str = "relife"  # relife = full ReLife agent (Claude, tools) · llm = CrewAI agent
    llm: str = ""            # runtime llm: a CrewAI/LiteLLM model string, or "claude-max"
    inherit: list[str] = field(default_factory=list)  # agents whose memory it reads
    fork: str | None = None  # agent whose memory it starts from (a copy)


@dataclass
class TaskSpec:
    name: str
    description: str
    expected_output: str
    agent: str
    context: list[str] = field(default_factory=list)  # earlier tasks whose output it reads


@dataclass
class CrewSpec:
    goal: str
    agents: list[AgentSpec]
    tasks: list[TaskSpec]
    process: str = "sequential"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CrewSpec:
        """Rebuild a spec that was validated when it was recorded (for display;
        untrusted input goes through :func:`normalize_spec`)."""
        return cls(
            goal=d["goal"],
            agents=[AgentSpec(**a) for a in d["agents"]],
            tasks=[TaskSpec(**t) for t in d["tasks"]],
            process=d.get("process", "sequential"),
        )

    def agent(self, name: str) -> AgentSpec:
        return next(a for a in self.agents if a.name == name)


def _text(d: dict[str, Any], key: str, *, required: bool = True, where: str) -> str:
    v = d.get(key, "")
    if not isinstance(v, str):
        raise ValueError(f"{where}: {key} must be a string")
    v = v.strip()
    if required and not v:
        raise ValueError(f"{where}: {key} is required")
    return v[:MAX_TEXT]


def _names(v: Any, *, where: str) -> list[str]:
    if v is None:
        return []
    if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
        raise ValueError(f"{where} must be a list of names")
    return list(dict.fromkeys(x.strip() for x in v if x.strip()))


def normalize_spec(
    raw: Any,
    *,
    existing_agents: set[str] | frozenset[str] = frozenset(),
    allowed_llms: tuple[str, ...] | None = None,
    max_agents: int | None = None,
    max_tasks: int | None = None,
) -> CrewSpec:
    """Validate an untrusted crew spec; raise ``ValueError`` saying what's wrong.

    ``existing_agents`` are registered agent names a spec agent may reuse or
    inherit from. ``allowed_llms`` limits non-Claude models (``None`` = any;
    ``claude-max`` is always allowed).
    """
    max_agents = config.CREW_MAX_AGENTS if max_agents is None else max_agents
    max_tasks = config.CREW_MAX_TASKS if max_tasks is None else max_tasks
    if not isinstance(raw, dict):
        raise ValueError("a crew spec is a JSON object")
    goal = _text(raw, "goal", where="crew")
    process = raw.get("process", "sequential")
    if process not in PROCESSES:
        raise ValueError(f"process must be one of {', '.join(PROCESSES)}")

    raw_agents = raw.get("agents")
    if not isinstance(raw_agents, list) or not raw_agents:
        raise ValueError("a crew needs at least one agent")
    if len(raw_agents) > max_agents:
        raise ValueError(f"at most {max_agents} agents per crew (got {len(raw_agents)})")
    agents: list[AgentSpec] = []
    seen: set[str] = set()
    for i, d in enumerate(raw_agents):
        if not isinstance(d, dict):
            raise ValueError(f"agent #{i + 1} must be an object")
        try:
            name = validate_agent_name(d.get("name"))
        except ValueError as e:
            raise ValueError(f"agent #{i + 1}: {e}") from None
        where = f"agent {name}"
        if name in seen:
            raise ValueError(f"{where}: defined twice")
        runtime = d.get("runtime", "relife")
        if runtime not in RUNTIMES:
            raise ValueError(f"{where}: runtime must be one of {', '.join(RUNTIMES)}")
        llm = d.get("llm") or ""
        if not isinstance(llm, str):
            raise ValueError(f"{where}: llm must be a string")
        llm = llm.strip()
        if runtime == "llm":
            if not llm:
                raise ValueError(f"{where}: an llm agent needs an llm (a model string or {config.CREW_CLAUDE_LLM!r})")
            if (
                allowed_llms is not None
                and llm != config.CREW_CLAUDE_LLM
                and llm not in allowed_llms
            ):
                raise ValueError(
                    f"{where}: model {llm!r} isn't configured (RELIFE_CREW_LLMS allows: "
                    f"{', '.join(allowed_llms) or 'none'}; {config.CREW_CLAUDE_LLM!r} is always available)"
                )
        else:
            llm = ""  # a ReLife agent runs on ReLife's own model
        lineage_ok = existing_agents | seen
        inherit = _names(d.get("inherit"), where=f"{where}: inherit")
        if len(inherit) > MAX_INHERITS:
            raise ValueError(f"{where}: inherits from at most {MAX_INHERITS} agents")
        fork = d.get("fork") or None
        if fork is not None:
            if not isinstance(fork, str):
                raise ValueError(f"{where}: fork must be an agent name")
            fork = fork.strip() or None
        for ref in [*inherit, *([fork] if fork else [])]:
            if ref == name:
                raise ValueError(f"{where}: can't inherit from or fork itself")
            if ref not in lineage_ok:
                raise ValueError(f"{where}: unknown agent {ref!r} to inherit from / fork (define it first)")
        if fork and name in existing_agents:
            raise ValueError(f"{where}: already exists, so it can't be forked into")
        agents.append(
            AgentSpec(
                name=name,
                role=_text(d, "role", where=where),
                goal=_text(d, "goal", where=where),
                backstory=_text(d, "backstory", required=False, where=where),
                runtime=runtime,
                llm=llm,
                inherit=inherit,
                fork=fork,
            )
        )
        seen.add(name)

    raw_tasks = raw.get("tasks")
    if not isinstance(raw_tasks, list) or not raw_tasks:
        raise ValueError("a crew needs at least one task")
    if len(raw_tasks) > max_tasks:
        raise ValueError(f"at most {max_tasks} tasks per crew (got {len(raw_tasks)})")
    tasks: list[TaskSpec] = []
    done: set[str] = set()
    for i, d in enumerate(raw_tasks):
        if not isinstance(d, dict):
            raise ValueError(f"task #{i + 1} must be an object")
        tname = d.get("name") or f"task-{i + 1}"
        if not isinstance(tname, str) or not tname.strip():
            raise ValueError(f"task #{i + 1}: name must be a string")
        tname = tname.strip()[:80]
        where = f"task {tname}"
        if tname in done:
            raise ValueError(f"{where}: defined twice")
        agent = d.get("agent")
        if not isinstance(agent, str) or agent not in seen:
            raise ValueError(f"{where}: agent {agent!r} isn't on the crew")
        context = _names(d.get("context"), where=f"{where}: context")
        for ref in context:
            if ref not in done:
                raise ValueError(f"{where}: context {ref!r} must name an earlier task")
        tasks.append(
            TaskSpec(
                name=tname,
                description=_text(d, "description", where=where),
                expected_output=_text(d, "expected_output", where=where),
                agent=agent,
                context=context,
            )
        )
        done.add(tname)

    idle = seen - {t.agent for t in tasks}
    if idle:
        raise ValueError(f"agents with no task: {', '.join(sorted(idle))}")
    return CrewSpec(goal=goal, agents=agents, tasks=tasks, process=process)


def load_spec_file(path: Path) -> Any:
    """Read a spec file the user wrote: JSON, or YAML when PyYAML is installed
    (CrewAI depends on it). Returns the raw data for :func:`normalize_spec`."""
    text = Path(path).read_text(encoding="utf-8")
    if Path(path).suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore
        except ImportError as e:  # pragma: no cover - crewai brings it
            raise ValueError("YAML specs need PyYAML (pip install pyyaml), or use JSON") from e
        return yaml.safe_load(text)
    return json.loads(text)


def single_agent_spec(task: str, *, name: str = "generalist") -> CrewSpec:
    """The fallback plan: one ReLife agent does the whole task."""
    return CrewSpec(
        goal=task[:MAX_TEXT],
        agents=[
            AgentSpec(
                name=name,
                role="Generalist engineer",
                goal="Complete the user's task end to end",
                backstory="A capable ReLife agent with tools, permissions and long-term memory.",
            )
        ],
        tasks=[
            TaskSpec(
                name="do-the-task",
                description=task[:MAX_TEXT],
                expected_output="The task done, with a short summary of what changed and anything left open.",
                agent=name,
            )
        ],
    )
