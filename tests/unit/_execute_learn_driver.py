"""Driver for the execute-and-learn characterization and the confirmed-path gates (#1133).

Runs the REAL ``run_agentic_loop`` once, substrate-primary (no LLM worker), with the substrate proposer
replaced by one that returns ONE proposal carrying clusters (the ``test_approved_proposal_situation_1083``
driver). Everything the outcome touches is observed from OUTSIDE the code under test, so the same
observations hold whether the dispatch lives inline in ``agent_loop`` or in ``tool_dispatch``:

- the credit: ``agent_loop._record_outcome`` (the patch seam ``loop_setup`` binds into the run's
  ``rec_outcome`` partial), spied and then called for real against a recording NAc, so the booked
  VALENCE is observed too;
- the hub: a minimal stand-in carrying a hub ``agent_id`` that differs from the loop's agent name, the
  recording NAc and ``record_plan_outcome``;
- the Hippocampus: a stand-in that records ``capture_from_loop_async`` (the capture's ``situation``);
- ``environment.step``, the result cache's ``invalidate`` and the ``tool_called`` abstraction event,
  each spied on the instance the loop uses;
- the controller, captured at construction, for the follow-up it is left holding.

``level="supervised"`` routes the proposal through the SUPERVISED confirmation path: the probe tool is
in ``requires_confirmation`` and interactive mode is OFF, so the loop answers "yes" itself
(``should_prompt("confirmation")`` is False) and the answer is handled on the next tick.

``level="planning"`` routes it through the PLANNING queue (#1085): the loop queues every proposal, and an
EMBEDDER approves it (the ``test_approved_proposal_situation_1083`` pattern: ``proposal_queue.submit`` is
wrapped so the queued proposals are approved, in order, once ``approve_batch`` of them have been queued).
``forbid`` makes the probe tool a hard ``SafetyConstraints`` forbid; ``pause_on_approve`` pauses the
controller right after the approval (``emergency_halt``), before the loop drains the queue.

The mode (#963): ``state_mode`` is the loop state's ``"mode"`` (``None`` leaves it unset), ``grant`` the operator's
launch grant set on the executor before the run, and ``grant_at[k]`` a grant set just before the k-th proposal
(0-based) is handed to the loop, so between two ticks. ``Observed.followups`` holds every follow-up the loop was
left holding, in order (sampled before each later proposal and at the end).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

HUB_AGENT_ID = "hub_aut_1133"
CLUSTERS = {"interoception": "c_intero_7", "world": "c_world_3"}
REASONING = "probe the world"
TRIGGER = "what is out there"


@dataclass
class Observed:
    outcomes: list[dict[str, Any]] = field(default_factory=list)
    nac_observations: list[dict[str, Any]] = field(default_factory=list)
    plan_outcomes: list[dict[str, Any]] = field(default_factory=list)
    captures: list[dict[str, Any]] = field(default_factory=list)
    env_steps: list[Any] = field(default_factory=list)
    invalidations: list[dict[str, Any]] = field(default_factory=list)
    events: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    executed: list[dict[str, Any]] = field(default_factory=list)
    proposed: list[str] = field(default_factory=list)
    ctrls: list[Any] = field(default_factory=list)
    # Every recorder call, whatever its tool (``outcomes`` keeps the probe tool's only).
    all_outcomes: list[dict[str, Any]] = field(default_factory=list)
    # Every OTHER NAc call (``update_cluster_reward``, ``credit_goal``, ...), as (method, kwargs).
    nac_calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    # Every ``reset_deliberation`` the think stand-in received (``think_probe=True``).
    deliberation_resets: list[int] = field(default_factory=list)
    # The PLANNING arm: the proposals the loop queued, in order.
    submitted: list[Any] = field(default_factory=list)
    autonomy: Any = None
    raised: BaseException | None = None
    # Every distinct follow-up the controller held, in order (#963).
    followups: list[Any] = field(default_factory=list)

    def note_followup(self) -> None:
        held = self.ctrls[0].pending_action_followup if self.ctrls else None
        if held is not None and not any(held is f for f in self.followups):
            self.followups.append(held)

    @property
    def ctrl(self) -> Any:
        assert len(self.ctrls) == 1, self.ctrls
        return self.ctrls[0]


class _RecordingNAc:
    """Stands in for the hub's NAc: records every causal observation, and every other call (as a no-op)."""

    def __init__(self, sink: list[dict[str, Any]], calls: list[tuple[str, dict[str, Any]]] | None = None) -> None:
        self._sink = sink
        self._calls = calls if calls is not None else []

    def observe(self, **kw: Any) -> None:
        self._sink.append(kw)
        return None

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)

        def _noop(*a: Any, **k: Any) -> None:
            self._calls.append((name, dict(k)))
            return None

        return _noop


class _Hub:
    """The few MemoryHub members the loop reads (``loop_setup``, ``bio_integration``)."""

    def __init__(self, obs: Observed) -> None:
        from maxim.runtime.substrate_proposal import NO_SITUATION_CUE

        self.agent_id = HUB_AGENT_ID
        self.nac = _RecordingNAc(obs.nac_observations, obs.nac_calls)
        self.ec = None
        self.hippocampus = None
        self.situation_cue = NO_SITUATION_CUE
        self._obs = obs

    def on_session_start(self) -> dict[str, Any]:
        return {}

    def on_session_end(self) -> dict[str, Any]:
        return {}

    def on_session_end_lightweight(self) -> dict[str, Any]:
        return {}

    def record_plan_outcome(self, **kw: Any) -> None:
        self._obs.plan_outcomes.append(kw)


class _Hippocampus:
    def __init__(self, obs: Observed, *, raises: bool) -> None:
        self._obs = obs
        self._raises = raises
        self.config = SimpleNamespace(persistence_path=None)

    def capture_from_loop_async(self, **kw: Any) -> None:
        self._obs.captures.append(kw)
        if self._raises:
            from maxim.memory.encoding import EncodingContractError

            raise EncodingContractError("probe: capture contract broken")

    def observe_episode_event(self, event: Any) -> None:
        return None

    def start_capture_worker(self) -> None:
        return None

    def stop_capture_worker(self) -> None:
        return None

    def flush(self, timeout: float = 0.0) -> None:
        return None


class _Cache:
    def __init__(self, obs: Observed) -> None:
        self._obs = obs

    def invalidate(self, **kw: Any) -> int:
        self._obs.invalidations.append(kw)
        return 0

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)

        def _noop(*a: Any, **k: Any) -> None:
            return None

        return _noop


def run_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    tool: str = "probe_learn_1133",
    params: dict[str, Any] | None = None,
    success: bool = True,
    output: Any = "done",
    error: str | None = None,
    side_effects: dict[str, Any] | None = None,
    level: str = "autonomous",
    display_raises: bool = False,
    capture_raises: bool = False,
    then_propose: str | None = None,
    submit_interval: float | None = None,
    max_steps: int = 6,
    fail_unless_overwrite: bool = False,
    think_probe: bool = False,
    approve_batch: int = 1,
    forbid: bool = False,
    pause_on_approve: bool = False,
    record_raises_for: str | tuple[str, ...] | None = None,
    catch: bool = False,
    active_goal: str | None = None,
    policy: dict[str, Any] | None = None,
    tool_raises: type[BaseException] | None = None,
    state_mode: str | None = "active",
    grant: str | None = None,
    grant_at: dict[int, str | None] | None = None,
) -> Observed:
    """Run the loop until the one proposal has been executed (and, supervised, confirmed).

    ``fail_unless_overwrite``: the probe fails as ``write_file`` does on an existing file, unless it is
    called with ``overwrite=True``. ``think_probe``: a ``think`` stand-in is registered, recording
    ``reset_deliberation``. ``record_raises_for``: the outcome recorder raises for that tool, or
    those tools (after recording the call; the message names the tool). ``catch``: an exception out of the loop is kept in ``Observed.raised``.
    ``active_goal``: the state's active goal, so a credit booked against the NAc also credits the goal.
    ``policy``: the PLANNING controller's ``SupervisionPolicy`` fields (its hard denials, for instance); at
    SUPERVISED, fields added to the confirmation policy (``forbidden_tools`` makes a hard rejection, #963).
    ``tool_raises``: the probe tool raises this (a ``KeyboardInterrupt``, say) instead of returning; with
    ``catch``, any ``BaseException`` out of the loop is kept in ``Observed.raised``.
    """
    from maxim.agents.autonomy import AutonomyController, AutonomyLevel, SafetyConstraints, SupervisionPolicy
    from maxim.agents.llm_types import LLMProposal
    from maxim.agents.maxim_agent import MaximAgent
    from maxim.environment.filesystem_env import FileSystemEnv
    from maxim.runtime import agent_loop as AL
    from maxim.runtime import substrate_proposal
    from maxim.runtime import prefetch
    from maxim.runtime.bootstrap import build_decision_engine, build_executor, build_memory
    from maxim.runtime.loop_controller import LoopController
    from maxim.runtime.state import RuntimeState
    from maxim.simulation import sim_logger
    from maxim.tools.base import Tool, ToolOutput
    from maxim.tools.registry import ToolRegistry
    from maxim.utils import structured_logging

    obs = Observed()
    tmp_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(tmp_path)

    class _Probe(Tool):
        name = tool
        description = "stub"
        input_schema: dict = {}

        def execute(self, **kwargs: Any) -> Any:
            obs.executed.append({"_tool": tool, **kwargs})
            if tool_raises is not None:
                raise tool_raises("probe: the tool raised")
            if fail_unless_overwrite and not kwargs.get("overwrite"):
                return ToolOutput(success=False, output=None, error=f"File already exists: {kwargs.get('path')}")
            return ToolOutput(success=success, output=output, error=error, side_effects=side_effects)

        def reset_deliberation(self) -> None:
            obs.deliberation_resets.append(1)

    registry = ToolRegistry()
    registry.register(_Probe())
    if think_probe:

        class _Think(Tool):
            name = "think"
            description = "stub"
            input_schema: dict = {}

            def execute(self, **kwargs: Any) -> Any:
                return ToolOutput(success=True, output="thought")

            def reset_deliberation(self) -> None:
                obs.deliberation_resets.append(1)

        registry.register(_Think())
    if then_propose is not None:

        class _Second(Tool):
            name = then_propose
            description = "stub"
            input_schema: dict = {}

            def execute(self, **kwargs: Any) -> Any:
                obs.executed.append({"_tool": then_propose, **kwargs})
                return ToolOutput(success=True, output="second")

        registry.register(_Second())
    executor = build_executor(registry, pain_bus=None, permissions=None)
    if grant is not None:
        executor.set_operational_override(grant)

    proposed: list[int] = []

    queue = [tool] if then_propose is None else [tool, then_propose]

    def _propose(**_kw: Any) -> Any:
        if len(proposed) >= len(queue):
            return None
        name = queue[len(proposed)]
        obs.note_followup()
        if grant_at and len(proposed) in grant_at:
            executor.set_operational_override(grant_at[len(proposed)])
        proposed.append(1)
        obs.proposed.append(name)
        return LLMProposal(
            request_id=f"p-1133-{len(proposed)}",
            action={"tool_name": name, "params": dict(params or {}) if name == tool else {}},
            reasoning=REASONING,
            strategy_used="substrate-primary",
            confidence=0.9,
            mode_goal_achieved=False,
            triggering_input=TRIGGER,
            cluster_id=CLUSTERS["interoception"],
            clusters=dict(CLUSTERS),
        )

    real_record = AL._record_outcome

    def _spy_record(**kw: Any) -> Any:
        obs.all_outcomes.append(dict(kw))
        if kw.get("tool_name") == tool:
            obs.outcomes.append(dict(kw))
        raising = (record_raises_for,) if isinstance(record_raises_for, str) else (record_raises_for or ())
        if kw.get("tool_name") in raising:
            raise RuntimeError(f"probe: the recorder broke on {kw.get('tool_name')}")
        return real_record(**kw)

    monkeypatch.setattr(substrate_proposal, "propose_via_substrate", _propose)
    monkeypatch.setattr(AL, "_record_outcome", _spy_record)

    cache = _Cache(obs)
    monkeypatch.setattr(prefetch, "get_result_cache", lambda: cache)

    buffer = structured_logging.get_abstraction_buffer()
    real_append = buffer.append

    def _append(record: Any) -> Any:
        if getattr(record, "source", None) == "agent_loop":
            obs.events.append((record.event, dict(record.data or {})))
        return real_append(record)

    monkeypatch.setattr(buffer, "append", _append)

    real_init = LoopController.__init__

    def _init(self: Any, *a: Any, **k: Any) -> None:
        real_init(self, *a, **k)
        if submit_interval is not None:
            self.llm_submit_interval = submit_interval  # the substrate proposer's cadence
        obs.ctrls.append(self)

    monkeypatch.setattr(LoopController, "__init__", _init)

    monkeypatch.setattr(sim_logger, "_interactive_mode", sim_logger.InteractiveMode.OFF)
    if display_raises:

        def _boom(*a: Any, **k: Any) -> None:
            raise RuntimeError("display broke")

        monkeypatch.setattr(sim_logger, "display_action", _boom)

    if level == "autonomous":
        controller = AutonomyController(initial_level=AutonomyLevel.AUTONOMOUS)
    elif level == "supervised":
        controller = AutonomyController(
            initial_level=AutonomyLevel.SUPERVISED,
            supervision_policy=SupervisionPolicy(
                requires_confirmation={tool}, min_confidence_autonomous=0.0, **(policy or {})
            ),
        )
    elif level == "planning":
        controller = AutonomyController(
            initial_level=AutonomyLevel.PLANNING,
            safety_constraints=SafetyConstraints(forbidden_tools=frozenset({tool})) if forbid else None,
            supervision_policy=SupervisionPolicy(**policy) if policy else None,
        )
        pq = controller.proposal_queue
        real_submit = pq.submit

        def _submit_then_approve(proposal: Any) -> None:
            # The embedder: approves the queued proposals, in order, once ``approve_batch`` are queued.
            real_submit(proposal)
            obs.submitted.append(proposal)
            if len(obs.submitted) % approve_batch == 0:
                for queued in obs.submitted[-approve_batch:]:
                    assert pq.approve(queued.id, approved_by="test:embedder")
                if pause_on_approve:
                    controller.emergency_halt("probe: paused after the approval")

        monkeypatch.setattr(pq, "submit", _submit_then_approve)
    else:
        raise ValueError(level)
    obs.autonomy = controller

    env = FileSystemEnv(str(tmp_path / "ws"))
    (tmp_path / "ws").mkdir(exist_ok=True)
    real_step = env.step

    def _step(result: Any) -> Any:
        obs.env_steps.append(result)
        return real_step(result)

    monkeypatch.setattr(env, "step", _step)

    state = RuntimeState()
    if state_mode is not None:
        state.data["mode"] = state_mode
    if active_goal is not None:
        state.data["active_goal"] = active_goal
    try:
        AL.run_agentic_loop(
            MaximAgent(),
            env,
            state,
            build_memory(),
            build_decision_engine(),
            executor,
            autonomy_controller=controller,
            hippocampus=_Hippocampus(obs, raises=capture_raises),
            memory_hub=_Hub(obs),
            aut_mode="substrate-primary",
            max_steps=max_steps,
            target_hz=200.0,
            idle_sleep_s=0.0,
            stop_event=threading.Event(),
        )
    except BaseException as exc:
        if not catch:
            raise
        obs.raised = exc
    obs.note_followup()
    return obs


def audit_view(obs: Observed, action_type: str = "executed") -> list[dict[str, Any]]:
    """The autonomy audit entries of one type, as comparable data (no timestamp)."""
    out = []
    for entry in obs.autonomy.get_audit_log():
        if entry.action_type != action_type:
            continue
        d = dict(vars(entry))
        d.pop("timestamp")
        out.append(d)
    return out


def credit_view(kw: dict[str, Any], obs: Observed) -> dict[str, Any]:
    """A recorder call as comparable data: the per-run sinks by identity, the rest by value."""
    out = dict(kw)
    ctrl = obs.ctrl
    assert out.pop("recent_outcomes") is ctrl.recent_outcomes
    assert out.pop("context_pool") is ctrl.context_pool
    nac = out.pop("nac")
    out["nac_is_hub_nac"] = isinstance(nac, _RecordingNAc)
    return out


def tool_events(obs: Observed, event: str, tool: str) -> list[dict[str, Any]]:
    return [d for e, d in obs.events if e == event and d.get("tool") == tool]
