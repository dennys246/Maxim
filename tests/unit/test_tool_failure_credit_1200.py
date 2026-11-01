"""#1200 Stage 2: a tool failure that RAN is booked once, by the tool-pain bridge, with a real surprise.

Before Stage 2 the executor's failure branches never called the bridge: a failed tool booked only
``tool_dispatch.record_outcome``'s NEGATIVE ``tool_result`` link, and ``ToolOutput.rpe`` was ``None`` for every
failure, so a failed action entered memory with no surprise. Stage 2 (``docs/plans/tool_failure_credit.md``,
owner decisions TF1-TF9) books ONE attributed NEGATIVE through the bridge, stamps the booked signature on the
output, and dispatch skips its own NEGATIVE for exactly that signature. Speculative prefetch calls book
nothing (TF9); a human-driven, bridge-less, aliased or compound-signature failure books as before.

Driven through the REAL executor (``build_executor`` with a real NAc, so the bridge is the production one),
the real ``tool_dispatch.execute_and_learn`` (stand-ins only for the parts the failure path does not learn
through) and the real ``SpeculativePrefetcher``.
"""

from __future__ import annotations

import functools
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from maxim.decisions.causal_link import Valence
from maxim.decisions.nac import NAc, NACConfig
from maxim.runtime import tool_dispatch
from maxim.runtime.bootstrap import build_executor
from maxim.tools.base import Tool, ToolOutput
from maxim.tools.registry import ToolRegistry

TOOL = "flaky_1200"


class _Flaky(Tool):
    """Returns a failure while ``fail`` is set, a success otherwise."""

    name = TOOL
    description = "stub"
    input_schema: dict = {}

    def __init__(self) -> None:
        super().__init__()
        self.fail = True

    def execute(self, **kwargs: Any) -> Any:
        if self.fail:
            return ToolOutput(success=False, error="boom")
        return ToolOutput(success=True, output="ok")


class _Raiser(Tool):
    """Raises out of ``run`` itself: ``Tool.run`` turns an ``execute`` exception into a failed output, so only a
    tool overriding ``run`` reaches the executor's raised branch."""

    name = "raiser_1200"
    description = "stub"
    input_schema: dict = {}

    def execute(self, **kwargs: Any) -> Any:
        raise AssertionError("unreachable: run() raises first")

    def run(self, **kwargs: Any) -> ToolOutput:
        raise RuntimeError("kaput")


class _Reader(Tool):
    """Stands in for ``read_file``, the tool prefetch calls speculatively."""

    name = "read_file"
    description = "stub"
    input_schema: dict = {}

    def __init__(self) -> None:
        super().__init__()
        self.ok = True

    def execute(self, **kwargs: Any) -> Any:
        if self.ok:
            return ToolOutput(success=True, output="contents")
        return ToolOutput(success=False, error="no such file")


def _executor(*tools: Tool, nac: NAc | None = None) -> tuple[Any, NAc]:
    nac = nac if nac is not None else NAc(NACConfig())
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    return build_executor(registry, pain_bus=None, permissions=None, nac=nac), nac


def _links(nac: NAc, signature: str, valence: Valence) -> list[Any]:
    return [link for link in nac._links.get(signature, []) if link.outcome_valence is valence]


def _execute_and_learn(executor: Any, nac: NAc, action: dict[str, Any], autonomy_controller: Any = None) -> Any:
    """The real dispatch, with the loop's own ``rec_outcome`` (a partial of the real ``record_outcome``)."""
    sim = MagicMock()
    sim.is_sim_mode = False
    return tool_dispatch.execute_and_learn(
        agent=MagicMock(_execution_tracker=None),
        agent_name="aut",
        agent_id="aut_1200",
        executor=executor,
        sim=sim,
        state=MagicMock(data={}),
        environment=MagicMock(),
        memory=MagicMock(),
        hippocampus=None,
        memory_hub=None,
        result_cache=MagicMock(),
        autonomy_controller=autonomy_controller if autonomy_controller is not None else MagicMock(),
        rec_outcome=functools.partial(tool_dispatch.record_outcome, drive_relief_only=False),
        recent_outcomes=[],
        max_recent=5,
        llm_worker=None,
        context_pool=MagicMock(),
        nac=nac,
        run_id="run_1200",
        overlay={},
        action=action,
        confidence=0.9,
        proposal=SimpleNamespace(reasoning="probe", citations=[], cluster_id=None, clusters=None, triggering_input=""),
        observation={},
        human_involved=False,
    )


# ── red gates: a failure that ran ─────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("tool", ["returned", "raised"])
def test_a_failure_that_ran_carries_its_surprise(tool: str) -> None:
    executor, _ = _executor(_Flaky(), _Raiser())
    name = TOOL if tool == "returned" else "raiser_1200"
    out = executor.execute({"tool_name": name, "params": {}})
    assert out.success is False
    assert out.rpe is not None and out.rpe > 0.0
    assert out.nac_failure_booked == f"tool:{name}"


def test_a_dispatched_failure_books_exactly_one_negative_on_the_bridge_link() -> None:
    executor, nac = _executor(_Flaky())
    _execute_and_learn(executor, nac, {"tool_name": TOOL, "params": {}})
    negatives = _links(nac, f"tool:{TOOL}", Valence.NEGATIVE)
    assert [(link.outcome_type, link.outcome_signature, link.observation_count) for link in negatives] == [
        ("result", f"tool:{TOOL}:negative", 1)
    ]


def test_a_model_written_speculative_key_books_as_normal() -> None:
    """TF9 is carried by the executor's context, never by the action dict: that is the model's own JSON."""
    executor, nac = _executor(_Flaky())
    _execute_and_learn(executor, nac, {"tool_name": TOOL, "params": {}, "_speculative": True})
    assert [link.outcome_signature for link in _links(nac, f"tool:{TOOL}", Valence.NEGATIVE)] == [
        f"tool:{TOOL}:negative"
    ]


def test_a_raised_failure_is_booked_once_through_dispatch() -> None:
    executor, nac = _executor(_Raiser())
    _execute_and_learn(executor, nac, {"tool_name": "raiser_1200", "params": {}})
    negatives = _links(nac, "tool:raiser_1200", Valence.NEGATIVE)
    assert [link.outcome_type for link in negatives] == ["result"]


def test_a_raise_after_execute_returned_does_not_book_the_failure_twice() -> None:
    """A step between ``execute`` and the credit raising sends the dispatch to its except branch (#1145's
    shape): that branch must read the same stamp, or the bridge-booked failure gains a second NEGATIVE."""
    executor, nac = _executor(_Flaky())
    autonomy = MagicMock()
    autonomy.log_action.side_effect = [RuntimeError("audit down"), None]
    _execute_and_learn(executor, nac, {"tool_name": TOOL, "params": {}}, autonomy_controller=autonomy)
    negatives = _links(nac, f"tool:{TOOL}", Valence.NEGATIVE)
    assert [link.outcome_type for link in negatives] == ["result"]


def test_a_tool_cannot_set_its_own_failure_stamp() -> None:
    """The stamp tells dispatch to skip its NEGATIVE, so a tool-authored one would erase a failure's dispatch
    link; the executor overwrites it on every path (mirrors #847's ``test_a_tool_cannot_set_its_own_surprise``)."""

    class _Forger(_Flaky):
        def execute(self, **kwargs: Any) -> Any:
            return ToolOutput(success=False, error="boom", nac_failure_booked=f"tool:{TOOL}")

    registry = ToolRegistry()
    registry.register(_Forger())
    executor = build_executor(registry, pain_bus=None, permissions=None)  # no bridge: nothing was booked
    assert executor.execute({"tool_name": TOOL, "params": {}}).nac_failure_booked is None


def test_a_record_outcome_call_without_the_stamp_is_a_type_error() -> None:
    """TF8 step 3 (#1221): the default is gone, so forgetting the stamp fails loudly."""
    with pytest.raises(TypeError):
        tool_dispatch.record_outcome(
            agent_id="aut_1200",
            tool_name=TOOL,
            success=False,
            result_summary=None,
            error="boom",
            reasoning="",
            recent_outcomes=[],
            max_recent=5,
            llm_worker=None,
            context_pool=MagicMock(),
        )


def test_the_bridge_books_nothing_without_a_started_invocation() -> None:
    """No pending entry (never started, a human driving, a speculative call): ``None``, nothing booked."""
    from maxim.bridges.tool_pain_bridge import ToolPainBridge

    nac = NAc(NACConfig())
    bridge = ToolPainBridge(nac=nac, pain_detector=None)
    assert bridge.record_tool_failure(TOOL, "never-started") is None
    assert nac._links == {}


def test_a_failure_whose_event_is_gone_carries_no_stamp_and_no_surprise() -> None:
    """The event aged out or was consumed before the booking: no link, so no stamp (dispatch books as before) and
    no surprise noted (``rpe`` stays unmeasured, not a measured 0.0)."""
    executor, nac = _executor(_Flaky())
    real_start = executor._tool_pain_bridge.record_tool_start

    def _start_then_lose_the_event(tool_name: str, invocation_id: str, **kw: Any) -> Any:
        out = real_start(tool_name, invocation_id, **kw)
        nac._pending_events.clear()  # stands in for the 300 s window expiring mid-run
        return out

    executor._tool_pain_bridge.record_tool_start = _start_then_lose_the_event
    out = executor.execute({"tool_name": TOOL, "params": {}})
    assert (out.nac_failure_booked, out.rpe) == (None, None)


def test_a_bridge_error_degrades_to_dispatch_booking() -> None:
    """A raising bridge never fails the call: the output carries no stamp, so dispatch books the NEGATIVE."""
    executor, nac = _executor(_Flaky())

    def _boom(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("bridge down")

    executor._tool_pain_bridge.record_tool_failure = _boom
    out = executor.execute({"tool_name": TOOL, "params": {}})  # the executor's own contract: it does not raise
    assert (out.success, out.nac_failure_booked) == (False, None)
    _execute_and_learn(executor, nac, {"tool_name": TOOL, "params": {}})
    assert [link.outcome_type for link in _links(nac, f"tool:{TOOL}", Valence.NEGATIVE)] == ["tool_result"]


def _prefetch(ok: bool) -> NAc:
    from maxim.runtime.prefetch import SpeculativePrefetcher

    reader = _Reader()
    reader.ok = ok
    executor, nac = _executor(reader)
    SpeculativePrefetcher(executor=executor)._execute_cached("read_file", {"path": "x.txt"})
    return nac


def _booked_read_file(nac: NAc) -> tuple[list[Any], list[Any]]:
    return nac._links.get("tool:read_file", []), [e for e in nac._pending_events if e["signature"] == "tool:read_file"]


def test_a_speculative_prefetch_hit_books_nothing() -> None:
    assert _booked_read_file(_prefetch(ok=True)) == ([], [])


# ── guards: what Stage 2 must not change ──────────────────────────────────────────────────────────────


def test_a_success_books_as_before() -> None:
    """Guard (green before and after): one bridge POSITIVE plus dispatch's own success link."""
    flaky = _Flaky()
    flaky.fail = False
    executor, nac = _executor(flaky)
    _execute_and_learn(executor, nac, {"tool_name": TOOL, "params": {}})
    positives = _links(nac, f"tool:{TOOL}", Valence.POSITIVE)
    assert sorted(link.outcome_type for link in positives) == ["result", "tool_result"]
    assert _links(nac, f"tool:{TOOL}", Valence.NEGATIVE) == []


def test_a_human_driven_failure_books_through_dispatch_as_before(monkeypatch: pytest.MonkeyPatch) -> None:
    """Guard (TF3): the interactive gate suppresses the bridge, so dispatch keeps its NEGATIVE."""
    import maxim.simulation.sim_logger as sim_logger

    monkeypatch.setattr(sim_logger, "get_interactive_mode", lambda: sim_logger.InteractiveMode.ON)
    executor, nac = _executor(_Flaky())
    _execute_and_learn(executor, nac, {"tool_name": TOOL, "params": {}})
    negatives = _links(nac, f"tool:{TOOL}", Valence.NEGATIVE)
    assert [link.outcome_type for link in negatives] == ["tool_result"]


def test_a_failure_with_no_bridge_books_through_dispatch_as_before() -> None:
    """Guard: an executor without an NAc has no bridge (``api.py``'s learning=False, the sandbox); dispatch
    still books into the loop's NAc."""
    registry = ToolRegistry()
    registry.register(_Flaky())
    executor = build_executor(registry, pain_bus=None, permissions=None)
    assert executor._tool_pain_bridge is None
    nac = NAc(NACConfig())
    _execute_and_learn(executor, nac, {"tool_name": TOOL, "params": {}})
    assert [link.outcome_type for link in _links(nac, f"tool:{TOOL}", Valence.NEGATIVE)] == ["tool_result"]


def test_a_speculative_prefetch_miss_books_nothing() -> None:
    """Guard (green today only because no failure reaches the bridge yet): once Stage 2 books failures, a
    speculative miss must still book nothing (TF9)."""
    assert _booked_read_file(_prefetch(ok=False)) == ([], [])


def test_an_aliased_failure_keeps_dispatchs_alias_link() -> None:
    """Guard (TF7): the bridge books the canonical name, dispatch the alias the model wrote; they differ, so
    dispatch's link under the alias is not cut."""

    class _Recall(_Flaky):
        name = "memory_recall"

    executor, nac = _executor(_Recall())
    _execute_and_learn(executor, nac, {"tool_name": "recall", "params": {}})
    assert [link.outcome_type for link in _links(nac, "tool:recall", Valence.NEGATIVE)] == ["tool_result"]


# ── the cut itself ────────────────────────────────────────────────────────────────────────────────────


def _record(nac: NAc, *, success: bool, bridge_booked: str | None, tool: str = TOOL, params: Any = None) -> None:
    tool_dispatch.record_outcome(
        agent_id="aut_1200",
        tool_name=tool,
        success=success,
        result_summary=None if not success else "ok",
        error="boom" if not success else None,
        reasoning="probe",
        recent_outcomes=[],
        max_recent=5,
        llm_worker=None,
        context_pool=MagicMock(),
        nac=nac,
        tool_params=params,
        bridge_booked=bridge_booked,
    )


def _dispatch_links(nac: NAc, signature: str) -> list[str]:
    return [str(link.outcome_valence) for link in nac._links.get(signature, []) if link.outcome_type == "tool_result"]


def test_the_cut_skips_only_the_matching_failure() -> None:
    nac = NAc(NACConfig())
    _record(nac, success=False, bridge_booked=f"tool:{TOOL}")
    assert _dispatch_links(nac, f"tool:{TOOL}") == []


@pytest.mark.parametrize(
    ("success", "booked", "params"),
    [
        (False, None, None),  # nothing booked by a bridge
        (False, "tool:memory_recall", None),  # the bridge booked a different (canonical) signature
        (False, "tool:use", {"action": "dig"}),  # dispatch's signature is compound: tool:use:dig
        (True, f"tool:{TOOL}", None),  # a success is never cut
    ],
    ids=["none", "other_signature", "compound", "success"],
)
def test_the_cut_books_as_before_otherwise(success: bool, booked: str | None, params: Any) -> None:
    nac = NAc(NACConfig())
    tool = "use" if params else TOOL
    _record(nac, success=success, bridge_booked=booked, tool=tool, params=params)
    sig = tool_dispatch.build_tool_signature(tool, params)
    assert len(_dispatch_links(nac, sig)) == 1


def test_a_parallel_failure_is_booked_once() -> None:
    """The batch path threads the stamp through its per-action dict."""
    executor, nac = _executor(_Flaky())
    autonomy = MagicMock()
    autonomy.can_execute_action.return_value = (True, "")
    tool_dispatch.execute_parallel_actions(
        agent_id="aut_1200",
        actions=[{"tool_name": TOOL, "params": {}}],
        executor=executor,
        autonomy_controller=autonomy,
        confidence=0.9,
        reasoning="probe",
        recent_outcomes=[],
        max_recent=5,
        llm_worker=None,
        context_pool=MagicMock(),
        nac=nac,
    )
    negatives = _links(nac, f"tool:{TOOL}", Valence.NEGATIVE)
    assert [link.outcome_type for link in negatives] == ["result"]


def test_a_speculative_failure_carries_no_stamp() -> None:
    from maxim.runtime.executor import speculative_calls

    executor, nac = _executor(_Flaky())
    with speculative_calls():
        out = executor.execute({"tool_name": TOOL, "params": {}})
    assert out.nac_failure_booked is None
    assert nac._links.get(f"tool:{TOOL}", []) == []


# ── agent_loop's two recorder sites (TF8, Session A) ──────────────────────────────────────────────────


class _Hub:
    """The few MemoryHub members the loop reads, carrying the SAME real NAc the executor's bridge books into."""

    agent_id = "hub_aut_1200"
    ec = None
    hippocampus = None

    def __init__(self, nac: NAc) -> None:
        from maxim.runtime.substrate_proposal import NO_SITUATION_CUE

        self.nac = nac
        self.situation_cue = NO_SITUATION_CUE
        self.ends: list[bool] = []

    # #1438 PR 0: the session API takes an owner and hands back a ticket that every end must present.
    def on_session_start(self, *, owner: Any, live: bool = False, deadline_s: float | None = None) -> Any:
        self._ticket = SimpleNamespace(generation=1, owner=owner, start_stats={})
        return self._ticket

    def _end(self, ticket: Any) -> dict[str, Any]:
        # Recorded, not asserted: end_bio_session logs a raising end at WARNING, so the driver checks ``ends``.
        self.ends.append(ticket is not None and ticket is getattr(self, "_ticket", None))
        self._ticket = None
        return {}

    def on_session_end(self, ticket: Any, *, deadline: Any = None, consolidation_bound: Any = None) -> dict[str, Any]:
        return self._end(ticket)

    def on_session_end_lightweight(self, ticket: Any, *, deadline: Any = None) -> dict[str, Any]:
        return self._end(ticket)

    def release_session_claim(self, ticket: Any) -> None:
        return None

    def record_plan_outcome(self, **kw: Any) -> None:
        return None


def _run_agent_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
    *,
    actions: int,
    execute_raises_on: int | None = None,
    log_raises_on: int | None = None,
) -> tuple[NAc, list[dict[str, Any]]]:
    """Drive the REAL ``run_agentic_loop``'s §3 agent-fallback (``agent.propose_intent``) ``actions`` times with
    ``_Flaky`` (a returned failure, so the production bridge books it and stamps the output). No LLM worker and no
    substrate proposal, so every action is the agent's own. ``execute_raises_on=k``: the k-th (0-based)
    ``executor.execute`` raises before it returns (the tool never runs). ``log_raises_on=k``: the k-th action's
    audit ``log_action("executed")`` raises, AFTER ``execute`` returned. Returns the NAc and every recorder call."""
    import threading

    from maxim.agents.autonomy import AutonomyController, AutonomyLevel
    from maxim.environment.filesystem_env import FileSystemEnv
    from maxim.runtime import agent_loop as AL
    from maxim.runtime import substrate_proposal
    from maxim.runtime.bootstrap import build_decision_engine, build_memory
    from maxim.runtime.loop_controller import LoopController
    from maxim.runtime.state import RuntimeState
    from maxim.simulation import sim_logger

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sim_logger, "_interactive_mode", sim_logger.InteractiveMode.OFF)
    executor, nac = _executor(_Flaky())

    calls = {"execute": 0}
    real_execute = executor.execute

    def _execute(action: dict[str, Any], *a: Any, **k: Any) -> Any:
        n = calls["execute"]
        calls["execute"] += 1
        if n == execute_raises_on:
            raise RuntimeError("probe: execute raised before it returned")
        return real_execute(action, *a, **k)

    monkeypatch.setattr(executor, "execute", _execute)

    proposed = {"n": 0}

    class _Agent:
        name = "fallback-probe-1200"

        def propose_intent(self, state: Any, memory: Any) -> dict[str, Any] | None:
            if proposed["n"] >= actions:
                return None
            proposed["n"] += 1
            return {"goal": {"tool_name": TOOL, "params": {}}, "confidence": 0.9, "source": "probe"}

    controller = AutonomyController(initial_level=AutonomyLevel.AUTONOMOUS)
    logged = {"n": 0}
    real_log = controller.log_action

    def _log_action(*, action_type: str, **kw: Any) -> Any:
        if action_type == "executed":
            n = logged["n"]
            logged["n"] += 1
            if n == log_raises_on:
                raise RuntimeError("probe: the audit broke after execute returned")
        return real_log(action_type=action_type, **kw)

    monkeypatch.setattr(controller, "log_action", _log_action)

    recorded: list[dict[str, Any]] = []
    real_record = tool_dispatch.record_outcome

    def _spy(**kw: Any) -> Any:
        recorded.append(dict(kw))
        return real_record(**kw)

    monkeypatch.setattr(tool_dispatch, "record_outcome", _spy)
    monkeypatch.setattr(substrate_proposal, "propose_via_substrate", lambda **_kw: None)
    real_init = LoopController.__init__

    def _init(self: Any, *a: Any, **k: Any) -> None:
        real_init(self, *a, **k)
        self.llm_submit_interval = 0.0  # the substrate cadence wakes every pass, so §3 runs every pass

    monkeypatch.setattr(LoopController, "__init__", _init)

    (tmp_path / "ws").mkdir(exist_ok=True)
    state = RuntimeState()
    state.data["mode"] = "active"
    hub = _Hub(nac)
    AL.run_agentic_loop(
        _Agent(),
        FileSystemEnv(str(tmp_path / "ws")),
        state,
        build_memory(),
        build_decision_engine(),
        executor,
        autonomy_controller=controller,
        memory_hub=hub,
        aut_mode="substrate-primary",
        max_steps=actions + 3,
        target_hz=200.0,
        idle_sleep_s=0.0,
        stop_event=threading.Event(),
    )
    assert calls["execute"] == actions, calls  # every proposed action reached the executor
    assert hub.ends == [True], f"the loop's own session end did not run exactly once, on its own ticket: {hub.ends}"
    return nac, [kw for kw in recorded if kw.get("tool_name") == TOOL]


def _negatives(nac: NAc) -> list[tuple[str, int]]:
    """The tool's NEGATIVE links as (family, observation count): ``result`` is the bridge's, ``tool_result``
    dispatch's."""
    return sorted((link.outcome_type, link.observation_count) for link in _links(nac, f"tool:{TOOL}", Valence.NEGATIVE))


# Red on main before #1231 (#1200, TF8): the hard-rejection call omitted bridge_booked; since #1221 that is a TypeError
def test_a_hard_rejection_records_with_no_failure_stamp(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    """The tool never ran, so nothing was bridge-booked: the call says so EXPLICITLY (``bridge_booked=None``), and
    the keyword is required since #1221 (TF8 step 3), so an omitted one is a ``TypeError``. The rejection's own
    booking is unchanged (``None`` books as before)."""
    from tests.unit._execute_learn_driver import run_once

    tool = "probe_reject_1200"
    obs = run_once(monkeypatch, tmp_path, tool=tool, level="supervised", policy={"forbidden_tools": {tool}})
    assert obs.executed == []
    [kw] = obs.outcomes
    assert kw["error"].startswith("Rejected by autonomy") and kw["success"] is False
    assert [o["outcome_type"] for o in obs.nac_observations if o["event_signature"] == f"tool:{tool}"] == [
        "tool_result"
    ]
    assert "bridge_booked" in kw, sorted(kw)
    assert kw["bridge_booked"] is None


# Red on main (#1200, TF8): the agent-fallback's except books dispatch's NEGATIVE on top of the bridge's
def test_an_agent_fallback_raise_after_execute_books_the_failure_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """``execute`` returned a failure the bridge booked and stamped; the audit then raised, so the except branch
    records. It passes the stamp, so dispatch skips its NEGATIVE: one booking, the bridge's."""
    nac, recorded = _run_agent_fallback(monkeypatch, tmp_path, actions=1, log_raises_on=0)
    [kw] = recorded
    assert kw["success"] is False and "audit broke" in kw["error"]
    assert _negatives(nac) == [("result", 1)]
    assert kw.get("bridge_booked") == f"tool:{TOOL}"


# Red on main (#1200, TF8): the except omits bridge_booked; its behavioural half guards the stale read the fix
# would otherwise create (``result`` is bound only by a RETURNED ``execute``)
@pytest.mark.parametrize("earlier", [False, True], ids=["first_action", "after_a_booked_failure"])
def test_an_agent_fallback_raise_before_execute_returns_reads_no_earlier_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any, earlier: bool
) -> None:
    """``execute`` raised, so the tool never ran and nothing was bridge-booked: dispatch books the NEGATIVE.
    ``after_a_booked_failure``: an EARLIER fallback action failed and was bridge-booked, so a stale ``result``
    would carry ITS stamp (same signature) and dispatch would skip this failure's only booking. ``first_action``:
    no ``result`` was ever bound, so reading it would raise inside the except and record nothing."""
    nac, recorded = _run_agent_fallback(
        monkeypatch, tmp_path, actions=2 if earlier else 1, execute_raises_on=1 if earlier else 0
    )
    kw = recorded[-1]
    assert kw["success"] is False and "raised before it returned" in kw["error"]
    assert _negatives(nac) == ([("result", 1), ("tool_result", 1)] if earlier else [("tool_result", 1)])
    assert "bridge_booked" in kw, sorted(kw)
    assert kw["bridge_booked"] is None


# ── every caller passes the stamp (TF8) ───────────────────────────────────────────────────────────────


def _record_outcome_calls_missing_the_stamp() -> dict[str, int]:
    """Calls of ``tool_dispatch.record_outcome`` (direct, through the module, or through the loop's bound
    ``rec_outcome`` partial) in ``src/`` and ``scripts/`` that omit ``bridge_booked``. Fails CLOSED on shapes it
    cannot follow: a plain or relative import of the module, and any non-call reference to ``record_outcome``
    other than ``loop_setup``'s one ``functools.partial``."""
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    missing: dict[str, int] = {}
    for base in ("src", "scripts"):
        for path in sorted((root / base).rglob("*.py")):
            rel = str(path.relative_to(root))
            tree = ast.parse(path.read_text(encoding="utf-8"))
            names = {"rec_outcome", "_rec_outcome"}
            imported: set[str] = set()  # ``record_outcome`` itself, under whatever name it was imported
            modules: set[str] = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    assert not any(a.name == "maxim.runtime.tool_dispatch" for a in node.names), (
                        f"{rel}: `import maxim.runtime.tool_dispatch` is a shape this guard does not follow"
                    )
                if isinstance(node, ast.ImportFrom) and node.level and (node.module or "").endswith("tool_dispatch"):
                    raise AssertionError(f"{rel}: a relative import of tool_dispatch is not followed")
                if isinstance(node, ast.ImportFrom) and node.module == "maxim.runtime.tool_dispatch":
                    imported |= {a.asname or a.name for a in node.names if a.name == "record_outcome"}
                if isinstance(node, ast.ImportFrom) and node.module == "maxim.runtime":
                    modules |= {a.asname or a.name for a in node.names if a.name == "tool_dispatch"}
            if path.name == "tool_dispatch.py":
                imported.add("record_outcome")
            names |= imported
            called: set[int] = set()
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                f = node.func
                hit = (isinstance(f, ast.Name) and f.id in names) or (
                    # a bound recorder reached as an attribute (``run.rec_outcome(...)``), or the module's own
                    isinstance(f, ast.Attribute)
                    and (
                        f.attr in ("rec_outcome", "_rec_outcome")
                        or (f.attr == "record_outcome" and isinstance(f.value, ast.Name) and f.value.id in modules)
                    )
                )
                if hit:
                    called.add(id(f))
                    if not any(k.arg == "bridge_booked" for k in node.keywords):
                        missing[rel] = missing.get(rel, 0) + 1
            for node in ast.walk(tree):
                ref = (
                    isinstance(node, ast.Attribute)
                    and node.attr == "record_outcome"
                    and isinstance(node.value, ast.Name)
                    and node.value.id in modules
                ) or (isinstance(node, ast.Name) and node.id in imported and isinstance(node.ctx, ast.Load))
                if ref and id(node) not in called:
                    missing[f"{rel}:non-call reference"] = missing.get(f"{rel}:non-call reference", 0) + 1
    return missing


# loop_setup binds the loop's ``rec_outcome`` as ``functools.partial(_td.record_outcome, ...)``: the one sanctioned
# non-call reference (its calls are the ``rec_outcome`` / ``_rec_outcome`` calls counted above).
_SANCTIONED_REFERENCES = {"src/maxim/runtime/loop_setup.py:non-call reference": 1}


def test_every_record_outcome_caller_passes_the_failure_stamp() -> None:
    """A caller that forgets ``bridge_booked`` books a bridge-booked failure twice, silently (the adversarial
    design pass's D1). The type enforces it since #1221; this guard stays because test spies
    (``lambda **kw``) and attribute calls bypass the signature check."""
    assert _record_outcome_calls_missing_the_stamp() == _SANCTIONED_REFERENCES


# ── #1221 item 2: a raise AFTER the bridge booked must not lose the stamp ──────────────────────────────


class _RaisingCollector:
    """A provenance collector whose logging fails: NAc calls it after the link is already written."""

    verbosity = 1

    def log_activity(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("collector down")


class _RaisingDetector:
    def record_tool_error(self, **kwargs: Any) -> None:
        raise RuntimeError("detector down")


def _kinds(nac: NAc, tool: str = TOOL) -> list[str]:
    return sorted(link.outcome_type for link in _links(nac, f"tool:{tool}", Valence.NEGATIVE))


def test_a_collector_raise_after_the_link_is_written_books_the_failure_once() -> None:
    executor, nac = _executor(_Flaky())
    nac._collector = _RaisingCollector()
    _execute_and_learn(executor, nac, {"tool_name": TOOL, "params": {}})
    assert _kinds(nac) == ["result"]


def test_a_raise_after_a_parallel_result_is_recorded_keeps_one_result(monkeypatch: pytest.MonkeyPatch) -> None:
    real = tool_dispatch.log_agentic

    def _log(source: str, event: str, data: Any = None, **kwargs: Any) -> None:
        if event == "parallel_action_complete":
            raise RuntimeError("log down")
        real(source, event, data, **kwargs)

    monkeypatch.setattr(tool_dispatch, "log_agentic", _log)
    executor, nac = _executor(_Flaky())
    autonomy = MagicMock()
    autonomy.can_execute_action.return_value = (True, "")
    results, _ = tool_dispatch.execute_parallel_actions(
        agent_id="aut_1200",
        actions=[{"tool_name": TOOL, "params": {}}],
        executor=executor,
        autonomy_controller=autonomy,
        confidence=0.9,
        reasoning="probe",
        recent_outcomes=[],
        max_recent=5,
        llm_worker=None,
        context_pool=MagicMock(),
        nac=nac,
    )
    assert [r["tool"] for r in results] == [TOOL]
    assert _kinds(nac) == ["result"]


@pytest.mark.parametrize("where", ["detector", "stamp"])
@pytest.mark.parametrize("tool", ["returned", "raised"])
def test_a_raise_after_the_bridge_booked_returns_the_stamp(where: str, tool: str) -> None:
    executor, nac = _executor(_Flaky(), _Raiser())
    name = TOOL if tool == "returned" else "raiser_1200"
    if where == "detector":
        executor._pain_detector = _RaisingDetector()
    else:

        def _boom(result: Any) -> Any:
            raise RuntimeError("relief down")

        executor._drive_relief = _boom
    out = executor.execute({"tool_name": name, "params": {}})
    assert (out.success, out.nac_failure_booked) == (False, f"tool:{name}")
    _execute_and_learn(executor, nac, {"tool_name": name, "params": {}})
    assert _kinds(nac, name) == ["result"]


def test_a_raise_after_the_bridge_wrote_its_links_still_returns_the_stamp() -> None:
    """The bridge's own step after the write (noting the surprise) is the same window (#1221 review)."""
    executor, nac = _executor(_Flaky())

    def _boom(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("rpe map down")

    executor._tool_pain_bridge._note_invocation_rpe = _boom
    out = executor.execute({"tool_name": TOOL, "params": {}})
    assert out.nac_failure_booked == f"tool:{TOOL}"
    _execute_and_learn(executor, nac, {"tool_name": TOOL, "params": {}})
    assert _kinds(nac) == ["result"]


def test_the_stamped_fallback_retires_the_invocations_surprise() -> None:
    """A post-booking raise still pops the invocation's surprise and pain: neither outlives it (2S-c)."""
    executor, _ = _executor(_Flaky())
    executor._pain_detector = _RaisingDetector()
    out = executor.execute({"tool_name": TOOL, "params": {}})
    assert out.rpe is not None and out.rpe > 0.0
    bridge = executor._tool_pain_bridge
    assert (bridge._rpe_by_invocation, bridge._pain_by_invocation) == ({}, {})


def test_a_raise_before_a_parallel_result_is_recorded_keeps_the_stamp(monkeypatch: pytest.MonkeyPatch) -> None:
    """The batch's except reads the stamp off the result it already has, as dispatch's other two callers do."""

    def _boom(result: Any) -> Any:
        raise RuntimeError("side effects unreadable")

    monkeypatch.setattr(tool_dispatch, "read_learning_side_effects", _boom)
    executor, nac = _executor(_Flaky())
    autonomy = MagicMock()
    autonomy.can_execute_action.return_value = (True, "")
    results, _ = tool_dispatch.execute_parallel_actions(
        agent_id="aut_1200",
        actions=[{"tool_name": TOOL, "params": {}}],
        executor=executor,
        autonomy_controller=autonomy,
        confidence=0.9,
        reasoning="probe",
        recent_outcomes=[],
        max_recent=5,
        llm_worker=None,
        context_pool=MagicMock(),
        nac=nac,
    )
    assert [(r["tool"], r["success"]) for r in results] == [(TOOL, False)]
    assert _kinds(nac) == ["result"]  # the batch strips its private keys: the stamp shows in what was booked


def test_a_raise_after_a_failure_nothing_booked_still_propagates() -> None:
    """With nothing booked there is no stamp to keep, so the post-booking raise reaches the caller as before."""
    executor, _ = _executor(_Flaky())
    executor._tool_pain_bridge = None
    executor._pain_detector = _RaisingDetector()
    with pytest.raises(RuntimeError, match="detector down"):
        executor.execute({"tool_name": TOOL, "params": {}})


# ── #1221 item 3: dispatch traces only the link it books ──────────────────────────────────────────────


def _traced(monkeypatch: pytest.MonkeyPatch) -> list[tuple[Any, ...]]:
    import maxim.simulation.sim_logger as sim_logger

    calls: list[tuple[Any, ...]] = []
    monkeypatch.setattr(sim_logger, "sim_nac", lambda *a, **k: calls.append(a))
    return calls


def test_a_bridge_booked_failure_is_not_traced_by_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _traced(monkeypatch)
    _record(NAc(NACConfig()), success=False, bridge_booked=f"tool:{TOOL}")
    assert calls == []


def test_a_booked_outcome_is_still_traced(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _traced(monkeypatch)
    _record(NAc(NACConfig()), success=False, bridge_booked=None)
    assert [c[:2] for c in calls] == [(f"tool:{TOOL}", "negative")]
