"""Characterization of ``run_agentic_loop``'s §6 submit, second half (1.3.6 loop phase 2, slice 7b).

Pins, through the PUBLIC entry ``agent_loop.run_agentic_loop`` on the LLM-primary path, what §6 does once slice 7a's
``prepare_submission`` has returned a context: the ``ModeInfo`` (the operational mode, an unknown one described as
passive under its own name, the ``llm.max_response_tokens`` override), the internet fields, the context's inputs cut
to the one being answered, the tool surface (the mode filter, the wired body's always-active SEM tools joined past it,
Wire 3's disabled filter, felt-phrase annotation with the earlier phrase stripped and its ``WIRE_3_FILTER`` sim event,
the shape-mismatch WARNING, the hard-forbid filter, the Wire 1 call), the context pool and history budgets, the
popped ``pending_modification``, the prefetch, the protocol context, the ``inference_start`` event (and a raising
listener), the ``MAXIM_TOOL_FAILURE_HINTS`` list, the triggering input, the ``take_replan_inputs`` fallback for the held
inputs (#1451), the submit clock, the submit logs and reports, and the outer "Failed to submit context to LLM"
handler; then the PFC deliberation: the cycle-1 wait, the ready branch (install, thinking-panel update, close,
refractory reset), the novelty skip, the not-ready branch (cycle 2's own request, an installed answer or the IDLE
line), the ``llm.deliberation_max_cycles`` override, and a run stopped during the wait. Written BEFORE slice 7b moves
the block out of ``_run_agentic_loop_body`` and kept green unchanged by that move (coverage-first rule).

The three values the body carries across passes are pinned exactly as they behave today:

- ``last_surfaced_tools``: §6 rebinds the body's local to the advertised list, and the NEXT pass's §2 records that
  list surfaced-but-unused against the chosen tool; ``ctrl.last_surfaced_tools`` (the setup alias) stays ``[]``.
- ``pending_prefetch``: a fresh input's prefetch rides that pass's submit (text and ``skip_exploration``) and no later
  one; ``ctrl.pending_prefetch`` is never cleared.
- ``_salience_c1``: what reaches ``sim_deliberation_update`` is this pass's cycle-1 salience, ``None`` when no pass
  ever computed it (§1.2's enrichment raised after the gate passed), or a STALE value from an earlier pass.

Observation is location-independent: a scripted worker records each ``submit_context`` and serves one proposal per
submit; the memory is a real ``InMemoryMemory``; the hub is slice 7a's stand-in carrying a real ``NAc`` subclass;
the controller is captured by a ``LoopController.__init__`` spy; the adapter records ``sim.log``; collaborators are
spied at their own modules (``sim_logger.sim_log``/``sim_deliberation_update``, ``tool_bridge.always_active_sem_tools``,
``prefetch.SpeculativePrefetcher``, ``ContextPool``, the abstraction buffer). Nothing patches ``agent_loop`` or the
module 7b moves to.

Covered elsewhere through the same public entry: the deliberation's cycle-2 continuation inbound feed and the
autonomy level re-read while it waits (``test_inbound_comms_1481``); the follow-up/held-input/triggering-input
precedence of the request (slice 7a's file). Wire 3's integrity bands, felt phrases and regex
(``test_wire_3_embodiment_filter``), the SEM always-active set (``test_sem_motor_binding``), Wire 1's variance bands
(``test_wire_1_risk_annotation``) and the cycles-2+ internals (``test_pfc_deliberation``) are pinned at their units.

Unreachable: the ``except ImportError`` around Wire 3's sim event (``maxim.simulation.sim_logger`` is in-package and
already imported by the loop). Equivalent mutant: ``new_inputs[-1:]`` -> ``new_inputs`` (with no new input the only
unprocessed line left is the follow-up's; the gate marks every other one processed or makes it the new input).

Pinned AS-IS, each a defect with its own issue (never fixed inside an extraction): (1) ``ctrl.last_surfaced_tools`` never
holds the surfaced list, only the body's local does (#1744); (2) ``ctrl.pending_prefetch`` keeps a consumed prefetch (#1744);
(3) ``locals().get("_salience_c1")`` reports a stale salience from an earlier pass, or ``None`` (#1744); (4) FIXED by #1743:
the ready-on-cycle-1 branch installed the answer directly, past §2's install checks, so an action-less one stayed
pending and §6 never submitted again (now ``test_1743_*``: the deliberation's answers go through §2's intake); (5) a submit opened with no input and no
follow-up (spoken "maxim") is reported to the sim as ``LLM submit: followup`` (#1746); (6) an unknown or passive mode advertises
its whole roster whether or not those tools are registered, and not the run's own (#1745).
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pytest

from tests.policy_helpers import TEST_POLICY

pytestmark = pytest.mark.timeout(60)

LOGGER = "maxim.runtime.agent_loop"
PROBE = "probe_submit"
PROBE2 = "probe_other"


def _proposal(*, ready: bool = True, reasoning: str = "", tool: str | None = None) -> Any:
    from maxim.agents.llm_types import LLMProposal

    return LLMProposal(
        request_id="p-7b",
        action={"tool_name": tool, "params": {}} if tool else {},
        reasoning=reasoning,
        strategy_used="llm",
        confidence=0.9,
        mode_goal_achieved=True,
        ready_to_act=ready,
    )


class _Worker:
    """Records each ``submit_context``. ``answers[n]`` is the proposal the first poll after the n-th submit returns
    (``None``: nothing); ``accept`` is what ``submit_context`` answers."""

    def __init__(
        self, answers: list[Any] | None = None, accept: bool = True, stop: Any = None, raise_on: int | None = None
    ) -> None:
        self.submits: list[dict[str, Any]] = []
        self.answers, self.accept, self.stop, self.raise_on = list(answers or []), accept, stop, raise_on
        self._owed: list[Any] = []

    def latest_attempt_state(self) -> Any:
        from maxim.agents.llm_types import LLMAttemptState

        return LLMAttemptState.NONE

    def get_latest_proposal(self) -> Any:
        return self._owed.pop(0) if self._owed else None

    def submit_context(self, **kw: Any) -> bool:
        ctx = kw.get("context")
        self.submits.append({"cli_inputs": list(getattr(ctx, "cli_inputs", None) or []), "context": ctx, "kw": kw})
        n = len(self.submits) - 1
        if n < len(self.answers) and self.answers[n] is not None:
            answer = self.answers[n]
            if getattr(answer, "original_request", None) is None:  # as the real worker stamps it (D13), same object
                request = SimpleNamespace(triggering_input=kw.get("triggering_input", ""), context=ctx)
                object.__setattr__(answer, "original_request", request)
            self._owed.append(answer)
        if self.stop is not None:  # the run stops while the submit is in flight
            self.stop.set()
        if n == self.raise_on:  # recorded, then the submit itself raises
            raise RuntimeError("submit boom")
        return self.accept

    def requeue_request(self, *_a: Any, **_k: Any) -> bool:
        return True

    def requeue_last_request(self, **_k: Any) -> bool:
        return True

    def record_outcome(self, *_a: Any, **_k: Any) -> None:
        return None


class _Agent:
    name = "submit-request-probe"

    def propose_intent(self, state: Any, memory: Any) -> None:
        return None


def _enriched(memories: int) -> Any:
    return SimpleNamespace(
        memories=[f"m{i}" for i in range(memories)], predictions=[], concepts=[], affordances=[], recent_context=None
    )


class _Pipeline:
    """``bio_enrichment_pipeline``: ``script[n]`` answers the n-th ``enrich`` (an Exception instance is raised; past
    the end, one memory)."""

    def __init__(self, script: list[Any] | None = None) -> None:
        self.script, self.calls = list(script or []), []

    def enrich(self, text: str, **kw: Any) -> Any:
        n = len(self.calls)
        self.calls.append(text)
        out = self.script[n] if n < len(self.script) else _enriched(1)
        if isinstance(out, Exception):
            raise out
        return out

    def format_thought_response(self, result: Any) -> str:
        return f"ENRICHED x{len(result.memories)}"


class _Gate:
    """``thought_gate``: always passes; records ``reset_refractory``."""

    def __init__(self) -> None:
        self.resets: list[int] = []

    def should_think(self, **kw: Any) -> Any:
        return SimpleNamespace(passed=True, score=SimpleNamespace(combined=0.9), threshold_used=0.5, reason="")

    def reset_refractory(self, step: int) -> None:
        self.resets.append(step)


class _Body:
    """``executor.embodiment`` with Wire 3's three reads scripted (an Exception instance is raised)."""

    def __init__(self, disabled: Any = (), degraded: Any = None) -> None:
        self.disabled, self.degraded = disabled, degraded or {}

    def get_disabled_affordances(self) -> Any:
        if isinstance(self.disabled, Exception):
            raise self.disabled
        return set(self.disabled)

    def get_degraded_affordances(self) -> Any:
        return dict(self.degraded)

    def integrity_to_felt_phrase(self, integrity: float) -> str:
        return "feels strained" if integrity >= 0.45 else ""

    def evaluate_failures(self) -> None:  # the pre-tick gate's embodiment tick
        return None


class _Protocols:
    def __init__(self, text: Any) -> None:
        self.text = text

    def get_context_for_llm(self) -> str:
        if isinstance(self.text, Exception):
            raise self.text
        return self.text


class _KEEP:
    pass


class _Hub:
    """The few MemoryHub members the loop reads (slice 7a's stand-in), carrying a real NAc."""

    agent_id = "submit_hub"
    ec = None
    hippocampus = None

    def __init__(self, nac: Any) -> None:
        from maxim.runtime.substrate_proposal import NO_SITUATION_CUE

        self.nac = nac
        self.situation_cue = NO_SITUATION_CUE

    def on_session_start(self, *, owner: Any, live: bool = False, deadline_s: float | None = None) -> Any:
        return SimpleNamespace(generation=1, owner=owner, start_stats={})

    def on_session_end(self, ticket: Any, *, deadline: Any = None, consolidation_bound: Any = None) -> dict:
        return {}

    def on_session_end_lightweight(self, ticket: Any, *, deadline: Any = None) -> dict:
        return {}

    def release_session_claim(self, ticket: Any) -> None:
        return None

    def record_plan_outcome(self, **kw: Any) -> None:
        return None


def _run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    steps: int = 3,
    inputs: dict[int, str] | None = None,
    amend: Callable[[int, Any], Any] | None = None,
    hooks: dict[int, Callable[[Any], None]] | None = None,
    state_data: dict[str, Any] | None = None,
    worker: _Worker | None = None,
    pipeline: Any = None,
    gate: Any = None,
    body: Any = None,
    on_event: Any = None,
    protocols: Any = None,
    autonomy: Any = None,
    planning_liveness: bool = False,
    extra_tools: tuple[str, ...] = (),
    descriptions: dict[str, str] | None = None,
    executor_hook: Callable[[Any], None] | None = None,
    stop_event: threading.Event | None = None,
    nac: Any = None,
    observations: dict[int, dict[str, Any]] | None = None,
    raise_on_log: str | None = None,
    wake: Callable[[], bool] | None = None,
    skip_fallback: bool = False,
    action_sink: Any = None,
) -> SimpleNamespace:
    """Run the real loop for ``steps`` passes (slice 7a's driver, with the 7b collaborators passed in)."""
    import maxim.simulation.sim_logger as sim_logger
    import maxim.utils.structured_logging as sl
    from maxim.agents.autonomy import AutonomyController, AutonomyLevel
    from maxim.environment.filesystem_env import FileSystemEnv
    from maxim.memory.base import InMemoryMemory
    from maxim.runtime import agent_loop as AL
    from maxim.runtime.bootstrap import build_decision_engine, build_executor
    from maxim.runtime.loop_controller import LoopController
    from maxim.runtime.sim_adapter import NullSimulationAdapter
    from maxim.runtime.state import RuntimeState
    from maxim.tools.base import Tool, ToolOutput
    from maxim.tools.registry import ToolRegistry

    tmp_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sim_logger, "_interactive_mode", sim_logger.InteractiveMode.OFF)
    obs = SimpleNamespace(
        ctrl=None,
        state=None,
        memory=None,
        worker=worker or _Worker(),
        sim_logs=[],
        adapter_logs=[],
        delib_updates=[],
        agentic=[],
        passes=-1,
    )
    real_sim_log = sim_logger.sim_log

    def _sim_log(category: str, msg: str, data: Any = None, **kw: Any) -> None:
        obs.sim_logs.append((category, msg, data))
        real_sim_log(category, msg, data, **kw)

    monkeypatch.setattr(sim_logger, "sim_log", _sim_log)
    real_update = sim_logger.sim_deliberation_update

    def _update(text: str, **kw: Any) -> None:
        obs.delib_updates.append((text, kw))
        real_update(text, **kw)

    monkeypatch.setattr(sim_logger, "sim_deliberation_update", _update)

    real_buffer = sl.get_abstraction_buffer()

    class _Buffer:
        def append(self, record: Any) -> None:
            if record.source == "agent_loop" and record.event == "llm_submit":
                obs.agentic.append(dict(record.data))
            real_buffer.append(record)

        def __getattr__(self, name: str) -> Any:
            return getattr(real_buffer, name)

    monkeypatch.setattr(sl, "get_abstraction_buffer", lambda: _Buffer())

    real_init = LoopController.__init__

    def _init(self: Any, *a: Any, **k: Any) -> None:
        real_init(self, *a, **k)
        self.llm_submit_interval = 0.0  # every pass may submit; the passes take microseconds
        obs.ctrl = self

    monkeypatch.setattr(LoopController, "__init__", _init)

    class _Memory(InMemoryMemory):
        def build_context(self) -> Any:
            ctx = super().build_context()
            if amend is not None:
                out = amend(obs.passes, ctx)
                return ctx if out is _KEEP else out
            return ctx

    class _Env(FileSystemEnv):
        def observe(self) -> dict[str, Any]:
            obs.passes += 1
            if hooks and obs.passes in hooks:
                hooks[obs.passes](obs)
            text = (inputs or {}).get(obs.passes)
            return {**({"cli_input": text} if text else {}), **(observations or {}).get(obs.passes, {})}

    class _Adapter(NullSimulationAdapter):
        def has_carried_percept(self) -> bool:
            return True if wake is None else wake()  # wake every pass, unless the test says when

        def should_skip_fallback_proposal(self, proposal: Any) -> bool:
            return skip_fallback and getattr(proposal, "reasoning", "") == "llm_fallback"

        def log(self, category: str, msg: str, data: dict | None = None) -> None:
            obs.adapter_logs.append((category, msg))
            if raise_on_log is not None and msg.startswith(raise_on_log):
                raise RuntimeError(f"sim log boom: {msg}")

    def _tool(tool_name: str) -> Tool:
        class _Probe(Tool):
            name = tool_name
            description = (descriptions or {}).get(tool_name, f"stub {tool_name}")
            input_schema: dict = {}

            def execute(self, **kwargs: Any) -> Any:
                return ToolOutput(success=True, output="ok")

        return _Probe()

    registry = ToolRegistry()
    for name in (PROBE, *extra_tools):
        registry.register(_tool(name))
    state = RuntimeState()
    state.data["mode"] = "active"
    state.data.update(state_data or {})
    obs.state = state
    memory = _Memory()
    obs.memory = memory
    executor = build_executor(registry, pain_bus=None, policy=TEST_POLICY)
    if body is not None:
        executor.embodiment = body
    if executor_hook is not None:
        executor_hook(executor)
    obs.executor = executor
    (tmp_path / "ws").mkdir(exist_ok=True)
    kwargs: dict[str, Any] = {"sim_adapter": _Adapter()}
    for key, value in (
        ("bio_enrichment_pipeline", pipeline),
        ("thought_gate", gate),
        ("on_event", on_event),
        ("protocol_registry", protocols),
        ("action_sink", action_sink),
    ):
        if value is not None:
            kwargs[key] = value
    if planning_liveness:
        kwargs["planning_liveness"] = True
    if nac is not None:
        kwargs["memory_hub"] = _Hub(nac)
    AL.run_agentic_loop(
        _Agent(),
        _Env(str(tmp_path / "ws")),
        state,
        memory,
        build_decision_engine(),
        executor,
        llm_worker=obs.worker,
        autonomy_controller=autonomy or AutonomyController(initial_level=AutonomyLevel.AUTONOMOUS),
        max_steps=steps,
        target_hz=1e6,
        idle_sleep_s=0.0,
        stop_event=stop_event or threading.Event(),
        **kwargs,
    )
    return obs


def _logs(caplog: Any, *prefixes: str, level: int = logging.DEBUG) -> list[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if r.name == LOGGER and r.levelno >= level and r.getMessage().startswith(prefixes)
    ]


@pytest.fixture(autouse=True)
def _loop_logs(caplog: Any) -> None:
    caplog.set_level(logging.DEBUG, logger="maxim")


def _kw(obs: Any, n: int = 0) -> dict[str, Any]:
    return obs.worker.submits[n]["kw"]


def _ctx_add(**fields: Any) -> Callable[[int, Any], Any]:
    def amend(k: int, ctx: Any) -> Any:
        if k == 0:
            for name, value in fields.items():
                setattr(ctx, name, value)
        return _KEEP

    return amend


# ── ModeInfo and the request's fixed fields ──────────────────────────────────


def test_the_request_carries_the_operational_modes_definition(monkeypatch, tmp_path):
    from maxim.modes.definitions import get_mode

    obs = _run(monkeypatch, tmp_path, inputs={0: "hi"}, steps=2)
    mode = _kw(obs)["mode"]
    active = get_mode("active")
    assert (mode.name, mode.goal, mode.context_prompt) == ("active", active.goal, active.context_prompt)
    assert mode.allowed_tools == {PROBE} and mode.forbidden_tools == set() and mode.max_response_tokens == 512
    assert (mode.can_access_filesystem, mode.can_access_network, mode.uses_tool_relevance_filter) == (True, True, False)


def test_an_unknown_mode_is_described_as_passive_under_its_own_name(monkeypatch, tmp_path):
    from maxim.modes.definitions import get_mode

    obs = _run(monkeypatch, tmp_path, inputs={0: "hi"}, state_data={"mode": "no_such_mode"}, steps=2)
    mode, passive = _kw(obs)["mode"], get_mode("passive")
    assert mode.name == "no_such_mode" and mode.goal == passive.goal and mode.context_prompt == passive.context_prompt
    assert mode.forbidden_tools == passive.forbidden_tools and mode.uses_tool_relevance_filter is True
    # Pinned as-is (#1745): the passive roster is advertised whole, none of it registered here, and the run's own tool is not.
    assert set(_kw(obs)["available_tools"]) == mode.allowed_tools and PROBE not in mode.allowed_tools


def test_the_response_token_override_reaches_the_mode(monkeypatch, tmp_path):
    monkeypatch.setenv("MAXIM_LLM_MAX_RESPONSE_TOKENS", "300")
    obs = _run(monkeypatch, tmp_path, inputs={0: "hi"}, steps=2)
    assert _kw(obs)["mode"].max_response_tokens == 300


def test_internet_access_fields_come_from_state(monkeypatch, tmp_path):
    obs = _run(monkeypatch, tmp_path, inputs={0: "hi"}, steps=2)
    assert (_kw(obs)["internet_access"], _kw(obs)["internet_policy_summary"]) == (False, "")
    net = {"internet_access": True, "internet_policy_summary": "allowlist only"}
    obs = _run(monkeypatch, tmp_path / "net", inputs={0: "hi"}, state_data=net, steps=2)
    assert (_kw(obs)["internet_access"], _kw(obs)["internet_policy_summary"]) == (True, "allowlist only")


# ── the context's inputs ─────────────────────────────────────────────────────


def test_only_the_new_input_rides_the_context(monkeypatch, tmp_path):
    """Two unprocessed inputs in the built context: the request keeps only the one the gate chose."""
    obs = _run(monkeypatch, tmp_path, amend=_ctx_add(cli_inputs=["older", "newer"]), steps=2)
    assert [(s["cli_inputs"], s["kw"]["triggering_input"]) for s in obs.worker.submits] == [(["older"], "older")]


def _followup_hook(obs: Any) -> None:
    from maxim.runtime.loop_types import ActionFollowup

    obs.ctrl.pending_action_followup = ActionFollowup(
        mode="active",
        timestamp=time.time(),
        tool=PROBE,
        result="the result",
        original_query="what was asked",
        followup_type="process",
    )


def test_with_no_new_input_the_newest_unprocessed_line_rides(monkeypatch, tmp_path):
    """A follow-up pass: memory still lists the processed "first"; only the follow-up line is kept."""
    obs = _run(monkeypatch, tmp_path, inputs={0: "first"}, hooks={1: _followup_hook}, steps=3)
    second = obs.worker.submits[1]["cli_inputs"]
    assert len(second) == 1 and second[0].startswith("[ACTION_FOLLOWUP")


# ── carried value 2: the prefetch ────────────────────────────────────────────


def _prefetch_spy(monkeypatch: pytest.MonkeyPatch, skip: bool = True) -> SimpleNamespace:
    from maxim.runtime import prefetch

    spy = SimpleNamespace(results=[], formatted=[])

    def _prefetch(self: Any, user_input: str, cwd: str | None = None) -> Any:
        result = prefetch.PrefetchResult(file_references=[], intent="unknown", skip_exploration=skip)
        spy.results.append((user_input, result))
        return result

    def _format(self: Any, result: Any, max_total_chars: int = 3000) -> str:
        spy.formatted.append(result)
        return f"PREFETCH#{len(spy.formatted)}"

    monkeypatch.setattr(prefetch.SpeculativePrefetcher, "prefetch_for_input", _prefetch)
    monkeypatch.setattr(prefetch.SpeculativePrefetcher, "format_prefetch_context", _format)
    return spy


def test_the_prefetch_rides_the_submit_of_its_own_pass_only(monkeypatch, tmp_path):
    """Carried value: a fresh input's prefetch (§1's ``ctrl.run_prefetch``) is formatted into that pass's submit with
    its ``skip_exploration``; the body then drops its LOCAL copy, so the next submit carries none, but
    ``ctrl.pending_prefetch`` is never cleared (pinned as-is, #1744)."""
    spy = _prefetch_spy(monkeypatch)
    obs = _run(monkeypatch, tmp_path, inputs={0: "open notes.txt"}, hooks={1: _followup_hook}, steps=3)
    [(text, result)] = spy.results
    assert text == "open notes.txt" and spy.formatted == [result]
    assert [(s["kw"]["prefetch_context"], s["kw"]["skip_exploration"]) for s in obs.worker.submits] == [
        ("PREFETCH#1", True),
        ("", False),
    ]
    assert obs.ctrl.pending_prefetch is result


def test_a_prefetch_that_says_explore_is_submitted_with_skip_off(monkeypatch, tmp_path):
    _prefetch_spy(monkeypatch, skip=False)
    obs = _run(monkeypatch, tmp_path, inputs={0: "open notes.txt"}, steps=2)
    assert (_kw(obs)["prefetch_context"], _kw(obs)["skip_exploration"]) == ("PREFETCH#1", False)


def test_a_new_inputs_prefetch_replaces_the_carried_one(monkeypatch, tmp_path):
    spy = _prefetch_spy(monkeypatch)
    obs = _run(monkeypatch, tmp_path, inputs={0: "one", 2: "two"}, steps=4)
    assert [s["kw"]["prefetch_context"] for s in obs.worker.submits] == ["PREFETCH#1", "PREFETCH#2"]
    assert spy.formatted == [r for _t, r in spy.results] and obs.ctrl.pending_prefetch is spy.results[1][1]


def test_a_prefetch_consumed_by_a_submit_that_raised_is_not_submitted_again(monkeypatch, tmp_path):
    """Carried value after a raise: the submit carrying the prefetch raises (logged, the pass goes on), yet the body's
    local was already dropped, so the next submit carries no prefetch and the prefetch is formatted once (review fold,
    executor lens: the carry is returned after the ``except``)."""
    spy = _prefetch_spy(monkeypatch)
    worker = _Worker(raise_on=0)
    obs = _run(monkeypatch, tmp_path, inputs={0: "open notes.txt"}, hooks={1: _followup_hook}, worker=worker, steps=3)
    assert [s["kw"]["prefetch_context"] for s in obs.worker.submits] == ["PREFETCH#1", ""]
    assert len(spy.formatted) == 1


# ── carried value 1: the surfaced tools ──────────────────────────────────────


class _ToolIndex:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str], str]] = []

    def record_surfaced_but_unused(self, goal: str, surfaced: Any, chosen: str) -> None:
        self.calls.append((goal, list(surfaced), chosen))


def test_the_next_pass_learns_from_the_tools_this_submit_surfaced(monkeypatch, tmp_path):
    """Carried value: §6 REBINDS the body's ``last_surfaced_tools`` to the advertised list; the next pass's §2, on
    installing the answer, records it surfaced-but-unused against the chosen tool. The controller's own list, the one
    the body aliased at setup, stays empty (pinned as-is, #1744)."""
    index = _ToolIndex()

    def _bridge(executor: Any) -> None:
        executor._tool_pain_bridge = SimpleNamespace(_tool_index=index)

    worker = _Worker(answers=[_proposal(tool=PROBE)])
    obs = _run(
        monkeypatch,
        tmp_path,
        inputs={0: "hi"},
        worker=worker,
        executor_hook=_bridge,
        extra_tools=(PROBE2,),
        state_data={"active_goal": "g1"},
        steps=3,
    )
    surfaced = _kw(obs)["available_tools"]
    assert sorted(surfaced) == [PROBE2, PROBE]
    assert index.calls == [("g1", list(surfaced), PROBE)]
    assert obs.ctrl.last_surfaced_tools == []


def test_a_raise_after_the_submit_still_carries_its_surfaced_tools(monkeypatch, tmp_path):
    """Carried value after a raise: the submit went out, then its sim log raised (logged, the pass goes on); the next
    pass's §2 still records THIS submit's roster surfaced-but-unused (review fold, executor lens)."""
    index = _ToolIndex()

    def _bridge(executor: Any) -> None:
        executor._tool_pain_bridge = SimpleNamespace(_tool_index=index)

    worker = _Worker(answers=[_proposal(tool=PROBE)])
    obs = _run(
        monkeypatch,
        tmp_path,
        inputs={0: "hi"},
        worker=worker,
        executor_hook=_bridge,
        extra_tools=(PROBE2,),
        state_data={"active_goal": "g1"},
        raise_on_log="LLM submit:",
        steps=3,
    )
    assert ("EXEC", "LLM submit: hi") in obs.adapter_logs  # the raise point was reached
    surfaced = _kw(obs)["available_tools"]
    assert sorted(surfaced) == [PROBE2, PROBE]
    assert index.calls == [("g1", list(surfaced), PROBE)]


def test_before_any_submit_nothing_is_recorded_surfaced(monkeypatch, tmp_path):
    """The first pass's §2 sees the setup alias (empty): an answer arriving before any submit records nothing."""
    index = _ToolIndex()
    worker = _Worker()
    worker._owed.append(_proposal(tool=PROBE))

    def _bridge(executor: Any) -> None:
        executor._tool_pain_bridge = SimpleNamespace(_tool_index=index)

    _run(monkeypatch, tmp_path, worker=worker, executor_hook=_bridge, state_data={"active_goal": "g1"}, steps=2)
    assert index.calls == []


# ── PFC deliberation; carried value 3: cycle 1's salience ────────────────────

SAL_1 = 0.3 * (1 / 5) + 0.3 * (1 / 4) + 0.4  # one section, one memory, novel: 0.535
SAL_3 = 0.3 * (1 / 5) + 0.3 * (3 / 4) + 0.4  # one section, three memories: 0.685


def _cycle1(obs: Any) -> list[tuple[str, Any]]:
    return [(text, kw.get("salience")) for text, kw in obs.delib_updates if kw.get("cycle") == 1]


def test_ready_on_cycle_1_installs_the_answer_and_closes_the_deliberation(monkeypatch, tmp_path):
    first = _proposal(reasoning="I will act now", tool=PROBE)  # with an action: §2's intake installs only those (#1743)
    gate, worker = _Gate(), _Worker(answers=[first])
    obs = _run(monkeypatch, tmp_path, inputs={0: "hi"}, worker=worker, pipeline=_Pipeline(), gate=gate, steps=1)
    assert obs.ctrl.pending_proposal is first and len(obs.worker.submits) == 1
    assert obs.delib_updates == [("I will act now", {"cycle": 1, "max_cycles": 2, "salience": pytest.approx(SAL_1)})]
    assert gate.resets == [0, 0]  # §1.2's cycle-1 reset, then §6's
    assert _kw(obs)["deliberation_available"] is True
    assert obs.ctrl.last_proposal_time > 0  # the wait stamps it (D13)


def test_cycle_1_salience_is_none_when_this_run_never_computed_it(monkeypatch, tmp_path):
    """§1.2's enrichment raised: the gate already passed, so §6 deliberates and reads ``_salience_c1`` unbound."""
    worker = _Worker(answers=[_proposal(reasoning="acting without recall", tool=PROBE)])
    pipeline = _Pipeline([RuntimeError("recall down")])
    obs = _run(monkeypatch, tmp_path, inputs={0: "hi"}, worker=worker, pipeline=pipeline, steps=1)
    assert _cycle1(obs) == [("acting without recall", None)]
    assert obs.ctrl.pending_proposal is not None


def test_cycle_1_salience_is_a_stale_value_from_an_earlier_pass(monkeypatch, tmp_path):
    """Pinned as-is (#1744): pass 0 computes salience 0.685; on pass 2 enrichment raises, yet the update reads pass 0's."""
    worker = _Worker(answers=[_proposal(reasoning="first thought here"), _proposal(reasoning="a different idea")])
    pipeline = _Pipeline([_enriched(3), RuntimeError("recall down")])
    obs = _run(
        monkeypatch,
        tmp_path,
        inputs={0: "one", 2: "two"},
        worker=worker,
        pipeline=pipeline,
        steps=3,
    )
    assert _cycle1(obs) == [
        ("first thought here", pytest.approx(SAL_3)),
        ("a different idea", pytest.approx(SAL_3)),
    ]


def test_cycle_1_salience_is_this_passes_when_recomputed(monkeypatch, tmp_path):
    worker = _Worker(answers=[_proposal(reasoning="first thought here"), _proposal(reasoning="a different idea")])
    pipeline = _Pipeline([_enriched(3), _enriched(1)])
    obs = _run(
        monkeypatch,
        tmp_path,
        inputs={0: "one", 2: "two"},
        worker=worker,
        pipeline=pipeline,
        steps=3,
    )
    assert [s for _t, s in _cycle1(obs)] == [pytest.approx(SAL_3), pytest.approx(SAL_1)]


# #1743: a deliberation answer goes through §2's one intake (``loop_intake.accept_llm_proposal``), as every other
# proposal does, so an action-less answer is never installed and the loop keeps hearing. The first two were the strict
# red gates (c3446ec1); before the fix the ready branch installed the answer and §6 never submitted again.
def test_1743_an_action_less_ready_answer_does_not_wedge_the_submit(monkeypatch, tmp_path):
    # The second answer is for the submit the fix lets through (else the deliberation's wait blocks on it); pre-fix
    # that submit never happens, so it is never read.
    worker = _Worker(answers=[_proposal(reasoning="nothing to do"), _proposal(reasoning="still nothing to do")])
    obs = _run(monkeypatch, tmp_path, inputs={0: "one", 2: "two"}, worker=worker, pipeline=_Pipeline(), steps=4)
    assert [s["cli_inputs"] for s in obs.worker.submits] == [["one"], ["two"]]
    assert obs.ctrl.pending_proposal is None


def test_1743_an_action_less_cycle_2_answer_does_not_wedge_the_submit(monkeypatch, tmp_path):
    # The third answer is for the submit the fix lets through (as in the ready gate above).
    answers = [_proposal(ready=False, reasoning="let me think more"), _proposal(reasoning="still nothing")]
    worker = _Worker(answers=[*answers, _proposal(reasoning="nothing for two either")])
    obs = _run(monkeypatch, tmp_path, inputs={0: "one", 2: "two"}, worker=worker, pipeline=_Pipeline(), steps=4)
    assert [s["cli_inputs"] for s in obs.worker.submits if s["cli_inputs"]][-1] == ["two"]


def test_1743_a_declined_deliberation_turn_is_dropped_not_retried(monkeypatch, tmp_path, caplog):
    """Under planning liveness §2 retries an action-less answer as a planning failure; the deliberation's intake
    drops it with no retry (owner decision 2026-11-24: a declined turn is not requeued). It still clears the pending
    user input, as §2's no-action branch does, so nothing waits on an answer that came."""
    worker = _Worker(answers=[_proposal(reasoning="nothing to do")])
    obs = _run(
        monkeypatch,
        tmp_path,
        inputs={0: "one"},
        state_data={"pending_user_input": "one", "pending_user_input_time": 1.0},
        planning_liveness=True,
        worker=worker,
        pipeline=_Pipeline(),
        steps=1,
    )
    assert obs.ctrl.pending_proposal is None and obs.ctrl.planning_failure_streak == 0
    assert _logs(caplog, "LLM proposal had no executable action") == [
        "LLM proposal had no executable action: proposal_completed_without_action"
    ]
    assert not _logs(caplog, "planning turn failed")
    assert "pending_user_input" not in obs.state.data and "pending_user_input_time" not in obs.state.data


def _error_answer_spends_the_budget(obs: Any) -> None:
    obs.ctrl.planning_retry_limit = 0  # the first planning failure exhausts


@pytest.mark.parametrize(
    "answers, raise_on_log, steps",
    [
        pytest.param([dict(error="model boom")], None, 40, id="ready-answer"),
        # the sim log after cycle 2's intake raises: submit_tick's except swallows the intake's outcome
        pytest.param([dict(ready=False), dict(error="model boom")], "multi-cycle deliberation yielded", 40, id="raise"),
        # spent on the run's last pass: no next gate runs, so the body's return reads the latch
        pytest.param([dict(error="model boom")], None, 1, id="last-pass"),
    ],
)
def test_1743_a_budget_spent_in_the_deliberation_aborts_on_the_next_pass(
    monkeypatch, tmp_path, answers, raise_on_log, steps
):
    """The deliberation's intake can spend the D13 budget (an error answer is a planning failure), after §2's break
    check. On a quiet pass §2 never runs, so the gate reads the latch: the next pass aborts, it does not idle."""
    import dataclasses

    from maxim.runtime.loop_controller import PlanningLivenessExhausted

    wakes: list[int] = []

    def _wake() -> bool:
        wakes.append(1)
        return len(wakes) == 1  # only the first pass has anything to react to

    proposals = [
        dataclasses.replace(_proposal(ready=a.get("ready", True), reasoning="r"), error=a.get("error")) for a in answers
    ]
    with pytest.raises(PlanningLivenessExhausted):
        _run(
            monkeypatch,
            tmp_path,
            inputs={0: "one"},
            hooks={0: _error_answer_spends_the_budget},
            planning_liveness=True,
            worker=_Worker(answers=proposals),
            pipeline=_Pipeline(),
            raise_on_log=raise_on_log,
            wake=_wake,
            steps=steps,
        )
    assert wakes == [1]  # the second pass's gate aborted before its idle check


def test_1743_the_stall_counter_carries_through_the_deliberation(monkeypatch, tmp_path):
    """A deliberation answer that timed out counts as a failed turn (#1185) and the count reaches the body: the
    next sim-skipped fallback, also the deliberation's, is the second in a row."""
    worker = _Worker(answers=[_proposal(reasoning="llm_timeout"), _proposal(reasoning="llm_fallback")])
    obs = _run(
        monkeypatch,
        tmp_path,
        inputs={0: "one", 2: "two"},
        worker=worker,
        pipeline=_Pipeline(),
        skip_fallback=True,
        steps=3,
    )
    assert ("EXEC", "DROPPED: fallback proposal (sim mode, #2)") in obs.adapter_logs


def test_1743_the_deliberation_closes_its_inference_event(monkeypatch, tmp_path):
    """The submit opens ``inference_start``; the deliberation's answer, through §2's intake, closes it."""
    events: list[Any] = []
    worker = _Worker(answers=[_proposal(tool=PROBE)])
    _run(monkeypatch, tmp_path, inputs={0: "one"}, worker=worker, pipeline=_Pipeline(), on_event=events.append, steps=1)
    assert [(e.kind, e.payload) for e in events if e.kind.startswith("inference")] == [
        ("inference_start", {}),
        ("inference_end", {"has_proposal": True}),
    ]


def test_1743_a_wait_that_brings_nothing_back_closes_the_inference_event(monkeypatch, tmp_path):
    events: list[Any] = []
    stop = threading.Event()
    worker = _Worker(answers=[None], stop=stop)  # the run stops while the submit is in flight: the wait returns None
    _run(
        monkeypatch,
        tmp_path,
        inputs={0: "one"},
        worker=worker,
        pipeline=_Pipeline(),
        on_event=events.append,
        stop_event=stop,
        steps=1,
    )
    assert [(e.kind, e.payload) for e in events if e.kind.startswith("inference")] == [
        ("inference_start", {}),
        ("inference_end", {"has_proposal": False}),
    ]


def test_1743_a_declined_ready_answer_does_not_say_ready_to_act(monkeypatch, tmp_path):
    import maxim.simulation.sim_logger as sim_logger

    ends: list[str] = []
    monkeypatch.setattr(sim_logger, "sim_deliberation_end", lambda **kw: ends.append(kw["summary"]))
    _run(monkeypatch, tmp_path, inputs={0: "one"}, worker=_Worker(answers=[_proposal()]), pipeline=_Pipeline(), steps=1)
    _run(
        monkeypatch,
        tmp_path / "acts",
        inputs={0: "one"},
        worker=_Worker(answers=[_proposal(tool=PROBE)]),
        pipeline=_Pipeline(),
        steps=1,
    )
    assert ends == ["Declined (cycle 1)", "Ready to act (cycle 1)"]


def test_1743_the_wait_resets_the_transport_streak_as_the_poll_does():
    """§2's poll resets the planning transport-failure streak on ANY proposal; the deliberation's wait does too."""
    from maxim.runtime.agent_loop import _wait_for_proposal

    resets: list[int] = []
    ctrl = SimpleNamespace(last_proposal_time=0.0, reset_planning_transport_failures=lambda: resets.append(1))
    worker = SimpleNamespace(get_latest_proposal=lambda: _proposal())
    assert _wait_for_proposal(worker, None, ctrl=ctrl) is not None
    assert resets == [1] and ctrl.last_proposal_time > 0


def test_the_blocked_submit_is_traced_in_sim_mode_only():
    """§6's diagnostic, moved out of the loop's body verbatim (#1743's rule-T carry)."""
    from maxim.runtime.loop_request import log_submit_blocked

    logs: list[tuple[str, str]] = []
    ctrl = SimpleNamespace(pending_proposal=_proposal(tool=PROBE))
    sim = SimpleNamespace(is_sim_mode=True, log=lambda c, m: logs.append((c, m)))
    log_submit_blocked(llm_worker=object(), ctrl=ctrl, cli_input="hello", sim=sim)
    log_submit_blocked(llm_worker=object(), ctrl=ctrl, cli_input=None, sim=sim)
    sim.is_sim_mode = False
    log_submit_blocked(llm_worker=object(), ctrl=ctrl, cli_input="hello", sim=sim)
    assert logs == [("PIPELINE", f"LLM gate BLOCKED: ctrl.pending_proposal={PROBE}, new cli_input=hello")]


@pytest.mark.parametrize(
    "deliberates, liveness, answer_tool, marked",
    [
        pytest.param(True, False, None, True, id="declined-deliberation"),
        pytest.param(True, True, None, True, id="declined-deliberation-liveness"),  # dropped, not retried
        pytest.param(False, False, None, True, id="section-2-no-action"),
        pytest.param(False, True, None, False, id="section-2-liveness-retries"),  # the retry's answer ends the turn
        pytest.param(True, False, PROBE, False, id="an-action"),
    ],
)
def test_1743_a_turn_ending_with_no_action_records_the_noop_marker(
    monkeypatch, tmp_path, deliberates, liveness, answer_tool, marked
):
    """The sim bridge settles on any record; a turn over with nothing done records ``DELIBERATION_NOOP`` (owner
    decision 2026-11-24) so the bridge does not wait out its timeout. A retried turn is not over."""
    from maxim.simulation.sinks import DELIBERATION_NOOP, RecordingSink

    sink = RecordingSink()
    _run(
        monkeypatch,
        tmp_path,
        inputs={0: "one"},
        worker=_Worker(answers=[_proposal(tool=answer_tool)]),
        pipeline=_Pipeline() if deliberates else None,
        planning_liveness=liveness,
        action_sink=sink,
        steps=2,
    )
    noops = [(a.tool_name, a.result_output) for a in sink.actions if a.tool_name == DELIBERATION_NOOP]
    assert noops == ([(DELIBERATION_NOOP, "proposal_completed_without_action")] if marked else [])


def _noops(sink: Any) -> list[str]:
    from maxim.simulation.sinks import DELIBERATION_NOOP

    return [a.result_output for a in sink.actions if a.tool_name == DELIBERATION_NOOP]


@pytest.mark.parametrize(
    "answer, liveness, marked",
    [
        pytest.param(dict(error="model boom"), False, ["proposal_error:model boom"], id="an-error-not-retried"),
        pytest.param(dict(error="shutdown"), False, [], id="shutdown-ends-the-run"),
        pytest.param(dict(stale=True), False, ["stale_proposal_dropped"], id="a-stale-drop-not-retried"),
        pytest.param(dict(stale=True), True, [], id="a-stale-drop-retried-under-liveness"),  # the retry ends the turn
    ],
)
def test_1743_other_turn_ends_with_nothing_done_record_the_marker(monkeypatch, tmp_path, answer, liveness, marked):
    """Planning liveness off, §2 (no deliberation): an error or a stale drop is not retried either, so the turn is over
    (owner decision 2026-11-24); a shutdown error ends the run, not a turn."""
    import dataclasses

    from maxim.simulation.sinks import RecordingSink

    proposal = dataclasses.replace(_proposal(), error=answer.get("error"))
    if answer.get("stale"):
        object.__setattr__(proposal, "timestamp", time.time() - 3600)
    sink = RecordingSink()
    _run(
        monkeypatch,
        tmp_path,
        inputs={0: "one"},
        worker=_Worker(answers=[proposal]),
        planning_liveness=liveness,
        action_sink=sink,
        steps=2,
    )
    assert _noops(sink) == marked


def test_1743_a_late_follow_up_answer_does_not_end_the_next_turn(monkeypatch, tmp_path):
    """A tool follow-up's action-less answer can land after the bridge settled on the tool's record and moved on:
    it must not record a marker into the next turn's window (delta round, executor SF1). Neither may a deliberation
    continuation's late answer (no triggering input)."""
    from maxim.agents.llm_types import LLMProposal
    from maxim.simulation.sinks import RecordingSink

    sink = RecordingSink()
    followup = SimpleNamespace(triggering_input="one", context=SimpleNamespace(cli_inputs=["[ACTION_FOLLOWUP x"]))
    continuation = SimpleNamespace(triggering_input="", context=SimpleNamespace(cli_inputs=["one"]))
    answers = []
    for request in (followup, continuation):
        p = _proposal(reasoning="done")
        object.__setattr__(p, "original_request", request)
        answers.append(p)
    worker = _Worker(answers=answers)
    _run(monkeypatch, tmp_path, inputs={0: "one", 2: "two"}, worker=worker, action_sink=sink, steps=4)
    assert len(worker.submits) == 2 and isinstance(answers[0], LLMProposal)
    assert _noops(sink) == []


@pytest.mark.parametrize("deliberates", [True, False], ids=["deliberated", "section-2"])
def test_1743_a_declined_answer_to_a_tool_follow_up_records_no_marker(monkeypatch, tmp_path, deliberates):
    """The composed path: a real pending follow-up (``_scan_context_inputs`` adds its line, ``_keep_newest_cli_input``
    keeps it), answered with no action. The bridge settled on the tool's own record, so a marker here could land in
    its next turn (delta round 2: the deliberation decided with cycle 1's request, as §2 does)."""
    from maxim.simulation.sinks import RecordingSink

    sink = RecordingSink()
    worker = _Worker(answers=[_proposal(reasoning="nothing more to do")])
    obs = _run(
        monkeypatch,
        tmp_path,
        hooks={0: _followup_hook},
        observations={0: {"raw_transcript_text": "a sound nearby"}},  # percept text: §1.2's gate passes
        worker=worker,
        pipeline=_Pipeline() if deliberates else None,
        action_sink=sink,
        steps=2,
    )
    assert worker.submits and worker.submits[0]["cli_inputs"][0].startswith("[ACTION_FOLLOWUP")
    assert (len(obs.delib_updates) > 0) is deliberates  # the deliberation did (or did not) run
    assert _noops(sink) == []


def test_1743_a_ready_answer_queues_its_next_actions(monkeypatch, tmp_path):
    """The deliberation's answers dropped their ``next_actions``; §2's intake queues them for ``multi_step``."""
    import dataclasses

    _next = {"tool_name": PROBE2, "params": {}}
    worker = _Worker(answers=[dataclasses.replace(_proposal(tool=PROBE), next_actions=[_next])])
    obs = _run(monkeypatch, tmp_path, inputs={0: "one"}, worker=worker, pipeline=_Pipeline(), steps=1)
    assert obs.ctrl.pending_proposal.action["tool_name"] == PROBE
    assert obs.ctrl.pending_next_actions == [_next]


def test_a_repeated_cycle_1_thought_is_not_shown_again(monkeypatch, tmp_path):
    worker = _Worker(answers=[_proposal(reasoning="same thought exactly"), _proposal(reasoning="same thought exactly")])
    obs = _run(
        monkeypatch,
        tmp_path,
        inputs={0: "one", 2: "two"},
        worker=worker,
        pipeline=_Pipeline(),
        steps=3,
    )
    assert len(obs.worker.submits) == 2 and [t for t, _s in _cycle1(obs)] == ["same thought exactly"]


def test_an_empty_cycle_1_reasoning_shows_nothing_but_still_closes(monkeypatch, tmp_path):
    gate = _Gate()
    obs = _run(
        monkeypatch,
        tmp_path,
        inputs={0: "hi"},
        worker=_Worker(answers=[_proposal(tool=PROBE)]),
        pipeline=_Pipeline(),
        gate=gate,
        steps=1,
    )
    assert obs.delib_updates == [] and gate.resets == [0, 0] and obs.ctrl.pending_proposal is not None


def test_not_ready_runs_cycle_2_with_its_own_request_and_installs_its_answer(monkeypatch, tmp_path):
    """Cycle 2's request carries the shared arguments unchanged and its own: no triggering input, modification,
    prefetch or held inputs, deliberation on, no inbound feed (no inbox here; the continuation feed is pinned in
    ``test_inbound_comms_1481``)."""
    _prefetch_spy(monkeypatch)
    second = _proposal(reasoning="now I act", tool=PROBE)
    worker = _Worker(answers=[_proposal(ready=False, reasoning="let me think more"), second])
    obs = _run(monkeypatch, tmp_path, inputs={0: "hi"}, worker=worker, pipeline=_Pipeline(), steps=1)
    assert obs.ctrl.pending_proposal is second
    first_kw, second_kw = _kw(obs, 0), _kw(obs, 1)
    shared = (
        "mode internet_access internet_policy_summary available_tools tool_descriptions context_pool_text "
        "agent_states recent_outcomes use_tool_prompting conversation_history_text is_sleeping protocol_context"
    ).split()
    assert {k: first_kw[k] for k in shared} == {k: second_kw[k] for k in shared}
    assert first_kw["prefetch_context"] == "PREFETCH#1" and first_kw["skip_exploration"] is True
    assert {k: v for k, v in second_kw.items() if k not in shared and k != "context"} == {
        "autonomy_level": first_kw["autonomy_level"],
        "triggering_input": "",
        "pending_modification": None,
        "prefetch_context": "",
        "skip_exploration": False,
        "deliberation_available": True,
        "deferred_inputs": [],
        "inbound_feed": None,
    }
    assert second_kw["context"] is first_kw["context"]
    assert ("DELIBERATION", "multi-cycle deliberation yielded proposal: tool=probe_submit") in obs.adapter_logs


def test_not_ready_with_no_actionable_answer_idles(monkeypatch, tmp_path):
    """Cycle 2's enrichment comes back empty: the deliberation stops with the action-less cycle-1 answer, which goes
    to §2's intake and is dropped there (#1743; it used to come back as None, past the intake), so nothing is
    installed."""
    worker = _Worker(answers=[_proposal(ready=False, reasoning="let me think more")])
    pipeline = _Pipeline([_enriched(1), None])
    obs = _run(monkeypatch, tmp_path, inputs={0: "hi"}, worker=worker, pipeline=pipeline, steps=1)
    assert obs.ctrl.pending_proposal is None and len(obs.worker.submits) == 1
    assert ("DELIBERATION", "multi-cycle deliberation declined: the intake took no action") in obs.adapter_logs


def test_the_deliberation_cap_override_reaches_the_cycles(monkeypatch, tmp_path):
    monkeypatch.setenv("MAXIM_LLM_DELIBERATION_MAX_CYCLES", "5")
    obs = _run(
        monkeypatch,
        tmp_path,
        inputs={0: "hi"},
        worker=_Worker(answers=[_proposal(reasoning="go now")]),
        pipeline=_Pipeline(),
        steps=1,
    )
    assert [kw["max_cycles"] for _t, kw in obs.delib_updates] == [5]


def test_no_deliberation_without_a_passed_gate_a_pipeline_or_a_submit(monkeypatch, tmp_path):
    """No percept text (a follow-up pass): the gate never passes, the answer waits for §2. A refused submit: no wait."""
    worker = _Worker(answers=[_proposal(reasoning="answer to the follow-up")])
    obs = _run(monkeypatch, tmp_path, hooks={0: _followup_hook}, worker=worker, pipeline=_Pipeline(), steps=1)
    assert len(obs.worker.submits) == 1 and obs.delib_updates == [] and obs.ctrl.pending_proposal is None
    assert worker._owed  # left for the next pass's §2
    worker = _Worker(answers=[_proposal(reasoning="never read")], accept=False)
    obs = _run(monkeypatch, tmp_path / "refused", inputs={0: "hi"}, worker=worker, pipeline=_Pipeline(), steps=1)
    assert obs.delib_updates == [] and obs.ctrl.pending_proposal is None and worker._owed


# ── the tool surface ─────────────────────────────────────────────────────────


def _sem_spy(monkeypatch: pytest.MonkeyPatch, names: Any) -> list[Any]:
    import maxim.embodiment.tool_bridge as tool_bridge

    calls: list[Any] = []

    def spy(registry: Any) -> list[Any]:
        calls.append(registry)
        if isinstance(names, Exception):
            raise names
        return [SimpleNamespace(name=n) for n in names]

    monkeypatch.setattr(tool_bridge, "always_active_sem_tools", spy)
    return calls


def test_a_wired_bodys_always_active_tools_join_past_the_mode_filter(monkeypatch, tmp_path):
    """Passive's roster does not list the body's tool; with a body it is advertised anyway, if registered and
    active (an unregistered name is not)."""
    calls = _sem_spy(monkeypatch, [PROBE2, "not_registered"])
    passive = {"mode": "passive"}
    run = {"inputs": {0: "hi"}, "extra_tools": (PROBE2,), "state_data": passive, "steps": 2}
    obs = _run(monkeypatch, tmp_path, body=_Body(), **run)
    assert PROBE2 in _kw(obs)["available_tools"] and "not_registered" not in _kw(obs)["available_tools"]
    assert PROBE2 in _kw(obs)["tool_descriptions"] and calls == [obs.executor.registry]
    assert PROBE2 not in _kw(obs)["mode"].allowed_tools  # the union is past the ModeInfo, not in it
    calls.clear()
    obs = _run(monkeypatch, tmp_path / "bodiless", **run)
    assert PROBE2 not in _kw(obs)["available_tools"] and calls == []


def test_a_failing_body_union_is_logged_and_the_submit_goes_on(monkeypatch, tmp_path, caplog):
    _sem_spy(monkeypatch, RuntimeError("bridge down"))
    obs = _run(monkeypatch, tmp_path, inputs={0: "hi"}, body=_Body(), steps=2)
    assert _kw(obs)["available_tools"] == [PROBE]
    assert _logs(caplog, "SEM prompt union failed") == ["SEM prompt union failed"]


def test_wire_3_hides_disabled_tools_and_annotates_degraded_ones(monkeypatch, tmp_path):
    """The felt phrase replaces any earlier one (the regex strip), copy-on-write; the sim log carries the names and
    the integrities, rounded."""
    _sem_spy(monkeypatch, [])
    body = _Body(disabled={PROBE}, degraded={PROBE2: 0.512345, "x_not_surfaced": 0.5})
    descriptions = {PROBE2: "moves the arm (feels weakened, prone to failing)"}
    obs = _run(
        monkeypatch, tmp_path, inputs={0: "hi"}, body=body, extra_tools=(PROBE2,), descriptions=descriptions, steps=2
    )
    assert _kw(obs)["available_tools"] == [PROBE2]
    assert _kw(obs)["tool_descriptions"][PROBE2]["description"] == "moves the arm (feels strained)"
    [(msg, data)] = [(m, d) for c, m, d in obs.sim_logs if c == "WIRE_3_FILTER"]
    assert msg == "wire_3: disabled=1 degraded=2"
    assert data["disabled_tools"] == [PROBE] and isinstance(data["tick"], int)
    assert data["degraded_integrities"] == {PROBE2: 0.5123, "x_not_surfaced": 0.5}
    assert obs.ctrl.last_surfaced_tools == []  # carried locally only (see the surfaced-tools arm)


def test_wire_3_leaves_a_tool_without_a_felt_phrase_as_described(monkeypatch, tmp_path):
    _sem_spy(monkeypatch, [])
    obs = _run(monkeypatch, tmp_path, inputs={0: "hi"}, body=_Body(degraded={PROBE: 0.31}), steps=2)
    assert _kw(obs)["tool_descriptions"][PROBE]["description"] == f"stub {PROBE}"
    assert [c for c, _m, _d in obs.sim_logs].count("WIRE_3_FILTER") == 1


def test_a_healthy_body_logs_no_wire_3_event(monkeypatch, tmp_path):
    _sem_spy(monkeypatch, [])
    obs = _run(monkeypatch, tmp_path, inputs={0: "hi"}, body=_Body(), steps=2)
    assert _kw(obs)["available_tools"] == [PROBE] and "WIRE_3_FILTER" not in [c for c, _m, _d in obs.sim_logs]


@pytest.mark.parametrize("error", [AttributeError, TypeError])
def test_a_wire_3_shape_mismatch_warns_and_filters_nothing(monkeypatch, tmp_path, caplog, error):
    _sem_spy(monkeypatch, [])
    obs = _run(monkeypatch, tmp_path, inputs={0: "hi"}, body=_Body(disabled=error("no such method")), steps=2)
    assert _kw(obs)["available_tools"] == [PROBE] and "WIRE_3_FILTER" not in [c for c, _m, _d in obs.sim_logs]
    [record] = [r for r in caplog.records if r.getMessage().startswith("Wire 3:")]
    assert record.levelno == logging.WARNING and record.getMessage().endswith("filter no-op: no such method")


def test_any_other_raise_in_the_block_drops_the_submit_with_a_warning(monkeypatch, tmp_path, caplog):
    _sem_spy(monkeypatch, [])
    obs = _run(monkeypatch, tmp_path, inputs={0: "hi"}, body=_Body(disabled=RuntimeError("modulator broke")), steps=2)
    assert obs.worker.submits == [] and obs.ctrl.last_llm_submit_time == 0.0
    [record] = [r for r in caplog.records if r.getMessage().startswith("Failed to submit context to LLM")]
    assert record.levelno == logging.WARNING
    assert record.getMessage() == "Failed to submit context to LLM: RuntimeError: modulator broke"
    assert _logs(caplog, "Full traceback:")


def test_a_hard_forbidden_tool_is_not_advertised(monkeypatch, tmp_path):
    from maxim.agents.autonomy import AutonomyController, AutonomyLevel, SafetyConstraints

    autonomy = AutonomyController(
        initial_level=AutonomyLevel.AUTONOMOUS,
        safety_constraints=SafetyConstraints(forbidden_tools=frozenset({PROBE2})),
    )
    obs = _run(monkeypatch, tmp_path, inputs={0: "hi"}, extra_tools=(PROBE2,), autonomy=autonomy, steps=2)
    assert _kw(obs)["available_tools"] == [PROBE] and PROBE2 not in _kw(obs)["tool_descriptions"]
    assert _kw(obs)["use_tool_prompting"] is True


def test_no_advertised_tool_turns_tool_prompting_off(monkeypatch, tmp_path):
    from maxim.agents.autonomy import AutonomyController, AutonomyLevel, SafetyConstraints

    autonomy = AutonomyController(
        initial_level=AutonomyLevel.AUTONOMOUS, safety_constraints=SafetyConstraints(forbidden_tools=frozenset({PROBE}))
    )
    obs = _run(monkeypatch, tmp_path, inputs={0: "hi"}, autonomy=autonomy, steps=2)
    assert _kw(obs)["available_tools"] == [] and _kw(obs)["use_tool_prompting"] is False


# ── the rest of the request ──────────────────────────────────────────────────


def test_the_context_pool_text_and_history_are_read_with_their_budgets(monkeypatch, tmp_path):
    from maxim.agents.context_pool import ContextPool

    reads: list[tuple] = []
    monkeypatch.setattr(
        ContextPool, "get_context_text", lambda self, max_tokens=None: reads.append(("ctx", max_tokens)) or "POOL"
    )
    monkeypatch.setattr(
        ContextPool, "get_conversation_text", lambda self, max_turns=10: reads.append(("conv", max_turns)) or "HISTORY"
    )
    obs = _run(monkeypatch, tmp_path, inputs={0: "hi"}, steps=2)
    assert (_kw(obs)["context_pool_text"], _kw(obs)["conversation_history_text"]) == ("POOL", "HISTORY")
    assert ("ctx", 1024) in reads and ("conv", 5) in reads  # half the mode's 2048-token window; five turns


def test_a_pending_modification_is_popped_into_the_request_and_triggers_it(monkeypatch, tmp_path):
    """With no new input or follow-up, the modification's text is the triggering input (spoken "maxim" opens the gate)."""
    mod = {"user_modification": "use the other file"}
    obs = _run(
        monkeypatch,
        tmp_path,
        amend=_ctx_add(detected_speech=["hi maxim"]),
        state_data={"pending_modification": mod},
        steps=2,
    )
    assert _kw(obs)["pending_modification"] == mod and _kw(obs)["triggering_input"] == "use the other file"
    assert "pending_modification" not in obs.state.data
    obs = _run(monkeypatch, tmp_path / "input", inputs={0: "hi"}, state_data={"pending_modification": mod}, steps=2)
    assert _kw(obs)["triggering_input"] == "hi" and _kw(obs)["pending_modification"] == mod


def test_the_protocol_context_is_read_fresh_per_submit(monkeypatch, tmp_path):
    obs = _run(monkeypatch, tmp_path, inputs={0: "hi"}, protocols=_Protocols("PROTOCOLS"), steps=2)
    assert _kw(obs)["protocol_context"] == "PROTOCOLS"
    assert _kw(_run(monkeypatch, tmp_path / "none", inputs={0: "hi"}, steps=2))["protocol_context"] == ""


def test_inference_start_is_streamed_before_the_submit_and_a_raising_listener_is_survived(
    monkeypatch, tmp_path, caplog
):
    seen: list[tuple[str, int]] = []
    holder: dict[str, Any] = {}

    def on_event(event: Any) -> None:
        seen.append((event.kind, len(holder["worker"].submits)))
        if holder.get("raise"):
            raise RuntimeError("listener broke")

    holder["worker"] = _Worker()
    obs = _run(monkeypatch, tmp_path, inputs={0: "hi"}, worker=holder["worker"], on_event=on_event, steps=2)
    assert [s for s in seen if s[0] == "inference_start"] == [("inference_start", 0)]
    holder.update(worker=_Worker(), **{"raise": True})
    obs = _run(monkeypatch, tmp_path / "raises", inputs={0: "hi"}, worker=holder["worker"], on_event=on_event, steps=2)
    assert len(obs.worker.submits) == 1
    assert any("on_event:inference_start" in r.getMessage() for r in caplog.records)


def test_hallucinated_tools_ride_only_with_the_hint_switch(monkeypatch, tmp_path):
    def _seen(executor: Any) -> None:
        executor._tools_hallucinated = ["ghost_tool"]

    obs = _run(monkeypatch, tmp_path, inputs={0: "hi"}, executor_hook=_seen, steps=2)
    assert _kw(obs)["failed_tools"] == []
    monkeypatch.setenv("MAXIM_TOOL_FAILURE_HINTS", "1")
    obs = _run(monkeypatch, tmp_path / "on", inputs={0: "hi"}, executor_hook=_seen, steps=2)
    assert _kw(obs)["failed_tools"] == ["ghost_tool"]
    monkeypatch.setenv("MAXIM_TOOL_FAILURE_HINTS", "0")
    assert (
        _kw(_run(monkeypatch, tmp_path / "off", inputs={0: "hi"}, executor_hook=_seen, steps=2))["failed_tools"] == []
    )


def test_the_autonomy_level_is_read_at_submit_time(monkeypatch, tmp_path):
    from maxim.agents.autonomy import AutonomyLevel

    obs = _run(monkeypatch, tmp_path, inputs={0: "hi"}, steps=2)
    assert _kw(obs)["autonomy_level"] == AutonomyLevel.AUTONOMOUS and _kw(obs)["deliberation_available"] is False
    assert _kw(obs)["inbound_feed"] is None and _kw(obs)["is_sleeping"] is False


def _held_lines(obs: Any) -> None:
    obs.memory.record_command("held one")
    obs.memory.record_command("held two")
    obs.ctrl.run_end_set_aside = True


def test_a_set_aside_finish_carries_every_held_input_to_the_replan(monkeypatch, tmp_path):
    """No follow-up held inputs: the submit falls back to ``take_replan_inputs`` (#1451), which needs planning
    liveness and a set-aside run-ending answer. The oldest unread line is the request; the rest ride."""
    obs = _run(monkeypatch, tmp_path, hooks={0: _held_lines}, planning_liveness=True, steps=1)
    assert (_kw(obs)["triggering_input"], _kw(obs)["deferred_inputs"]) == ("held one", ["held two"])
    assert list(obs.ctrl.processed_cli_inputs) == ["held one", "held two"]
    obs = _run(monkeypatch, tmp_path / "off", hooks={0: _held_lines}, steps=1)
    assert (_kw(obs)["triggering_input"], _kw(obs)["deferred_inputs"]) == ("held one", [])


# ── after the submit ─────────────────────────────────────────────────────────


def test_a_submit_logs_stamps_the_clock_and_reports(monkeypatch, tmp_path, caplog):
    long = "x" * 60
    before = time.time()
    obs = _run(monkeypatch, tmp_path, inputs={0: long}, steps=2)
    assert obs.ctrl.last_llm_submit_time >= before
    assert _logs(caplog, "Submitted", level=logging.INFO) == [f"Submitted to LLM: {'x' * 50}"]
    assert obs.agentic == [{"input": "x" * 50, "mode": "active", "autonomy": "autonomous", "tools_available": 1}]
    assert ("EXEC", f"LLM submit: {long}") in obs.adapter_logs


def test_a_follow_up_submit_logs_as_a_follow_up(monkeypatch, tmp_path, caplog):
    obs = _run(monkeypatch, tmp_path, hooks={0: _followup_hook}, steps=1)
    assert _logs(caplog, "Submitted", level=logging.INFO) == ["Submitted followup to LLM for processing"]
    assert obs.agentic == [{"input": "followup", "mode": "active", "autonomy": "autonomous", "tools_available": 1}]
    assert ("EXEC", "LLM submit: followup") in obs.adapter_logs


def test_a_gate_opened_submit_without_input_reports_none(monkeypatch, tmp_path, caplog):
    obs = _run(monkeypatch, tmp_path, amend=_ctx_add(detected_speech=["hi maxim"]), steps=1)
    assert _logs(caplog, "Submitted") == [] and obs.agentic[0]["input"] is None
    assert (
        "EXEC",
        "LLM submit: followup",
    ) in obs.adapter_logs  # pinned as-is (#1746): sim says "followup" for no input


def test_a_refused_submit_stamps_the_clock_but_reports_nothing(monkeypatch, tmp_path, caplog):
    obs = _run(monkeypatch, tmp_path, inputs={0: "hi"}, worker=_Worker(accept=False), steps=1)
    assert len(obs.worker.submits) == 1 and obs.ctrl.last_llm_submit_time > 0
    assert _logs(caplog, "Submitted") == [] and obs.agentic == [] and obs.adapter_logs == []


def test_a_run_stopped_during_the_cycle_1_wait_installs_nothing(monkeypatch, tmp_path):
    stop = threading.Event()
    obs = _run(
        monkeypatch,
        tmp_path,
        inputs={0: "hi"},
        worker=_Worker(stop=stop),
        pipeline=_Pipeline(),
        stop_event=stop,
        steps=2,
    )
    assert len(obs.worker.submits) == 1 and obs.ctrl.pending_proposal is None and obs.delib_updates == []


def test_wire_3_skips_a_degraded_tool_whose_description_is_not_text(monkeypatch, tmp_path):
    _sem_spy(monkeypatch, [])
    obs = _run(
        monkeypatch,
        tmp_path,
        inputs={0: "hi"},
        body=_Body(degraded={PROBE: 0.5}),
        descriptions={PROBE: None},  # type: ignore[dict-item]
        steps=2,
    )
    assert _kw(obs)["tool_descriptions"][PROBE]["description"] is None


def test_wire_1_annotates_a_variable_tool_from_the_runs_nac(monkeypatch, tmp_path):
    """The call only (Wire 1's bands are ``annotate_tool_variance``'s own): the run's NAc, under the hub's agent id,
    is asked for its risk profile and a high-variance surfaced tool's description gains the phrase."""
    from maxim.decisions.nac import NAc
    from maxim.runtime import loop_planning

    asked: list[str] = []

    class _RiskyNAc(NAc):
        def get_action_risk_profile(self, *, agent_id: str, min_observations: int = 0) -> dict:  # type: ignore[override]
            asked.append(agent_id)
            return {f"tool:{PROBE}": 1.0}

    obs = _run(monkeypatch, tmp_path, inputs={0: "hi"}, nac=_RiskyNAc(), steps=2)
    assert asked == ["submit_hub"]
    description = _kw(obs)["tool_descriptions"][PROBE]["description"]
    assert description.startswith(f"stub {PROBE}") and loop_planning._WIRE1_HIGH_PHRASE in description


def test_cycle_2_carries_no_held_inputs_even_when_cycle_1_did(monkeypatch, tmp_path):
    """Planning liveness, a follow-up and a held input, on a pass with percept text (a raw transcript) so the gate
    passes: cycle 1 carries the held input, cycle 2 none."""
    worker = _Worker(answers=[_proposal(ready=False, reasoning="let me think more"), _proposal(tool=PROBE)])
    obs = _run(
        monkeypatch,
        tmp_path,
        hooks={0: _followup_hook},
        amend=_ctx_add(cli_inputs=["held words"]),
        observations={0: {"raw_transcript_text": "a sound nearby"}},
        planning_liveness=True,
        worker=worker,
        pipeline=_Pipeline(),
        steps=1,
    )
    assert [s["kw"]["deferred_inputs"] for s in obs.worker.submits] == [["held words"], []]
