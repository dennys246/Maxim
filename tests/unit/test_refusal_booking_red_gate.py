"""#1315 §1A.2 behavioural red gate: a gate refusal books NO learning, through both dispatch paths.

WHAT THIS FILE IS. Tests only (``docs/plans/atonia_gate_approach_notes.md`` DRAFT v6.3 §1A.2, "The behavioural red
gate is the enforcement, not this table"). For every gate refusal kind, one refusal is driven through
``runtime/tool_dispatch.py::execute_and_learn`` AND ``execute_parallel_actions``, with ``drive_relief_only`` both
False (substrate-primary) and True (the shipped llm-primary default), on the cheapest REAL fixtures:
- a real ``NAc``;
- a real ``PainBus`` from ``build_pain_bus``, wired the way production wires it (``build_executor(pain_bus=...)``,
  as ``create_full_agent``, bio_stack and agentic_runtime do; no ``pain_detector=``);
- the real ``ToolPainBridge`` that ``build_executor`` builds, a real ``ToolRegistry``, a real
  ``AutonomyController`` and the real ``record_outcome``.

THE KINDS (``KIND_TO_ERROR_KIND``): mode, permissions, unregistered and inactive_scene exist today. POLICY_CEILING,
POLICY_PIN, the offline ceiling (V7: ``policy.ceiling & OFFLINE_CEILING`` with the latch engaged) and an
``OfflineRefusal`` raised inside a tool body (the ``_fail_ran`` booking-order case, §1A.2) assert their API exists
first, so they fail today with that AssertionError. The tool body raises from ``execute`` -- the shipped path for a
tool whose library call meets the latch -- so ``Tool.run``'s catch must also let the refusal reach the mapping.
``test_cases_cover_every_gate_refusal_kind`` pins that every member of ``GATE_REFUSAL_KINDS`` has a case here.

THE ASSERTION. Zero learning writes:
- ``NAc.observe``, goal credit (``NAc.credit_goal``) and cluster reward (``NAc.update_cluster_reward``);
- the pain bus's ``publish``;
- ``ToolPainBridge``'s booking methods (``record_tool_start``/``_complete``/``_failure``/``_embodiment_failure``);
- the plan bridge (``MemoryHub.record_plan_outcome``, the only path to ``PlanHistoryBridge.record_plan_outcome``);
- and NAc's state itself unchanged (a snapshot of its link and bias maps, ``_NAC_STATE``), so a write through an
  unspied NAc method cannot hide.

AND exactly one entry in each of the carryover (``llm_worker.record_outcome``), ``recent_outcomes`` and the
autonomy audit log (``AutonomyController.audit_log``). §1A.2 keeps those, so the model is told, and the
``max_consecutive_failures`` counter still stops a model that keeps naming a refused tool.

TODAY every case books a negative through ``record_outcome`` (``NAc.observe``, plus goal credit, plus cluster reward
when ``drive_relief_only`` is False) and ``execute_and_learn`` books the plan bridge. Under production wiring no
pain is published for these refusals (the executor has no ``pain_detector``). The parallel batch writes NO autonomy
audit entry at all, so the gate also demands one there (new work for item 3). Every case is xfail, so the one
parametrised def carries one literal ``xfail(strict=True)`` (lint rule 3b).

THE ONE LINE ITEM 3 TOUCHES. ``_build_executor`` builds through ``ToolPolicy`` once ``maxim.runtime.tool_policy``
exists (``deny`` for the permissions kind, which then refuses as ``POLICY_PERMISSIONS``; ``unrestricted``
otherwise), and through today's ``permissions=`` until then. So the item-3 PR flips this gate without re-pointing
it.

The scenario (the refusal happened, of the expected kind, and the autonomy gate admitted the action) is checked
first and raises ``ScenarioBroken``, never an ``AssertionError``. One exception: for the new kinds, the refusal being
TYPED (its ``error_kind``) is item 3's own behaviour, so a wrong kind there is an AssertionError.
"""

from __future__ import annotations

import functools
import importlib
import importlib.util
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Callable
from unittest.mock import MagicMock, patch

import pytest

from maxim.agents.autonomy import AutonomyController, AutonomyLevel
from maxim.agents.permissions import AgentPermissions
from maxim.bridges.tool_pain_bridge import ToolPainBridge
from maxim.decisions.nac import NAc, NACConfig
from maxim.proprioception.pain_bus import build_pain_bus
from maxim.runtime import tool_dispatch
from maxim.runtime.bootstrap import build_executor
from maxim.tools.base import Tool, ToolErrorKind, ToolOutput
from maxim.tools.registry import ToolRegistry

AGENT_ID = "refusal_gate"
GOAL = "probe the gate"
CLUSTER = "c-refusal"
POLICY_REASON = "red-gate probe (§1A.2 booking)"

# Each case's expected ToolErrorKind member name(s). The legacy kinds also accept today's error text (``_LEGACY_TEXT``).
KIND_TO_ERROR_KIND: dict[str, frozenset[str]] = {
    "mode": frozenset({"MODE"}),
    "permissions": frozenset({"POLICY_PERMISSIONS"}),
    "unregistered": frozenset({"NOT_REGISTERED"}),
    "inactive_scene": frozenset({"INACTIVE_SCENE"}),
    "policy_ceiling": frozenset({"POLICY_CEILING"}),
    "policy_pin": frozenset({"POLICY_PIN"}),
    "offline_ceiling": frozenset({"POLICY_CEILING", "OFFLINE"}),
    "offline_in_tool_body": frozenset({"OFFLINE"}),
}
_LEGACY_TEXT = {
    "mode": "passive mode",
    "permissions": "probe_denied",
    "unregistered": "Tool not registered",
    "inactive_scene": "is not active",
}
KINDS = tuple(KIND_TO_ERROR_KIND)
PATHS = ("execute_and_learn", "execute_parallel_actions")

# NAc's learned state: every map a booking could touch (snapshot before/after, beside the method spies).
_NAC_STATE = (
    "_links",
    "_outcome_index",
    "_priors",
    "_reward_bias",
    "_visit_count",
    "_eligibility",
    "_cluster_reward_bias",
    "_cluster_reward_source",
    "_pending_operant_action",
    "_goal_reward_bias",
    "_percept_valences",
    "_cluster_fear",
    "_event_outcome_welford",
)


class ScenarioBroken(Exception):
    """The scenario did not happen as designed: NOT the defect a gate pins (module docstring)."""


def require(cond: Any, msg: str) -> None:
    if not cond:
        raise ScenarioBroken(msg)


# ── lazy access to the not-yet-built API (existence asserts) ──────────────


def _require_effect() -> Any:
    import maxim.tools.base as base

    assert hasattr(base, "Effect"), "item 2 not built: tools/base.py has no Effect"
    return base


def _require_policy() -> Any:
    spec = importlib.util.find_spec("maxim.runtime.tool_policy")
    assert spec is not None, "item 3 not built: maxim/runtime/tool_policy.py does not exist"
    return importlib.import_module("maxim.runtime.tool_policy")


def _require_offline(*names: str) -> Any:
    try:
        spec = importlib.util.find_spec("maxim._offline")
    except ModuleNotFoundError:
        spec = None
    assert spec is not None, "step 0b not built: maxim._offline does not exist"
    off = importlib.import_module("maxim._offline")
    for name in names:
        assert hasattr(off, name), f"step 0b not built: maxim._offline has no {name}"
    return off


def _require_kinds(kind: str) -> None:
    for member in KIND_TO_ERROR_KIND[kind]:
        assert hasattr(ToolErrorKind, member), f"item 3 not built: ToolErrorKind has no {member}"


# ── the rig ───────────────────────────────────────────────────────────────


class _Probe(Tool):
    """A tool that would succeed if it ran; every case but the in-body one must refuse it before it runs."""

    description = "stub"
    input_schema: dict = {}

    def __init__(self, name: str) -> None:
        self.name = name
        super().__init__()
        self.ran = False

    def execute(self, **kwargs: Any) -> Any:
        self.ran = True
        return ToolOutput(success=True, output="ran")


def _effect_probe(name: str, effect: Any, body: Callable[[], Any] | None = None) -> Any:
    """A fresh Tool subclass declaring ``effect`` in its own body (§2: the class's own ``__dict__``)."""

    def execute(self: Any, **kwargs: Any) -> Any:
        self.ran = True
        if body is not None:
            return body()
        return ToolOutput(success=True, output="ran")

    try:
        cls = type(
            f"Probe_{name}",
            (Tool,),
            {"name": name, "description": "stub", "input_schema": {}, "effect": effect, "execute": execute},
        )
        tool = cls()
    except Exception as e:  # noqa: BLE001 -- a fixture that cannot be built is a broken scenario
        raise ScenarioBroken(f"fixture tool {name!r} could not be built: {type(e).__name__}: {e}") from e
    tool.ran = False
    return tool


def _build_executor(
    registry: ToolRegistry, *, nac: NAc, bus: Any, deny: tuple[str, ...] = (), policy: Any = None
) -> Any:
    """The one line item 3 touches (module docstring): ``ToolPolicy`` once it exists, ``permissions=`` until then."""
    if importlib.util.find_spec("maxim.runtime.tool_policy") is None:
        require(policy is None, "a ToolPolicy case reached the legacy builder")
        permissions = AgentPermissions(tool_deny=frozenset(deny)) if deny else None
        return build_executor(registry, pain_bus=bus, permissions=permissions, nac=nac)
    if policy is None:
        tp = importlib.import_module("maxim.runtime.tool_policy")
        policy = (
            tp.ToolPolicy.deny(list(deny), reason=POLICY_REASON)
            if deny
            else tp.ToolPolicy.unrestricted(reason=POLICY_REASON)
        )
    return build_executor(registry, pain_bus=bus, policy=policy, nac=nac)


@dataclass
class Rig:
    executor: Any
    nac: NAc
    bus: Any
    autonomy: AutonomyController
    llm_worker: MagicMock
    memory_hub: MagicMock
    tool_name: str
    probe: Any = None
    must_not_run: bool = True
    recent_outcomes: list[dict[str, Any]] = field(default_factory=list)
    cleanup: Callable[[], None] | None = None


def _rig(kind: str) -> Rig:
    """One executor refusing ``tool_name`` for ``kind``, on the real learning sinks, production pain wiring."""
    nac = NAc(NACConfig())
    bus = build_pain_bus(hippocampus=None, nac=nac)
    registry = ToolRegistry()
    probe: Any = None
    deny: tuple[str, ...] = ()
    policy: Any = None
    must_not_run = True
    cleanup: Callable[[], None] | None = None
    engage: Callable[[], None] | None = None
    if kind in ("policy_ceiling", "policy_pin", "offline_ceiling", "offline_in_tool_body"):
        _require_kinds(kind)
    if kind == "mode":
        probe = _Probe("bash")  # HOST_ACTING_TOOLS: passive mode refuses it at dispatch (#826)
        registry.register(probe)
    elif kind == "permissions":
        probe = _Probe("probe_denied")
        registry.register(probe)
        deny = ("probe_denied",)
    elif kind == "unregistered":
        pass
    elif kind == "inactive_scene":
        probe = _Probe("probe_scene")
        registry.register_scene_tools([probe], scene_id="scene-1")
        registry.deactivate_scene("scene-1")
    elif kind == "policy_ceiling":
        base, tp = _require_effect(), _require_policy()
        probe = _effect_probe("probe_network", base.Effect.NETWORK)
        registry.register(probe)
        policy = tp.ToolPolicy.only(["probe_network"], ceiling=base.Effect.READ, reason=POLICY_REASON)
    elif kind == "policy_pin":
        base, tp = _require_effect(), _require_policy()
        pinned = _effect_probe("probe_pin", base.Effect.READ)
        probe = _effect_probe("probe_pin", base.Effect.READ)  # same name, a different object: the pin refuses it
        registry.register(probe)
        policy = tp.ToolPolicy.pinned({"probe_pin": pinned}, ceiling=base.Effect.READ, reason=POLICY_REASON)
    elif kind == "offline_ceiling":
        off = _require_offline("engage", "_reset_for_tests")
        base, _ = _require_effect(), _require_policy()
        probe = _effect_probe("probe_offline_net", base.Effect.NETWORK)
        registry.register(probe)

        def engage() -> None:
            off.engage("red-gate probe: V7 offline ceiling")

        cleanup = off._reset_for_tests
    elif kind == "offline_in_tool_body":
        off = _require_offline("OfflineRefusal")
        _require_effect()

        def body() -> Any:
            raise off.OfflineRefusal("red-gate probe: the latch refused this tool's fetch")

        probe = _effect_probe("probe_lazy_fetch", _require_effect().Effect.READ, body)
        registry.register(probe)
        must_not_run = False  # it runs, and its body meets the latch
    else:  # pragma: no cover - the parametrisation is closed
        raise ScenarioBroken(f"unknown refusal kind {kind!r}")
    executor = _build_executor(registry, nac=nac, bus=bus, deny=deny, policy=policy)
    if kind == "mode":
        executor.set_operational_override("passive")
    if engage is not None:
        engage()  # AFTER the build: V7 is applied at dispatch, so a policy built before a raise still narrows
    require(isinstance(executor._tool_pain_bridge, ToolPainBridge), "build_executor built no ToolPainBridge")
    return Rig(
        executor=executor,
        nac=nac,
        bus=bus,
        autonomy=AutonomyController(initial_level=AutonomyLevel.AUTONOMOUS),
        llm_worker=MagicMock(),
        memory_hub=MagicMock(spec_set=["record_plan_outcome"]),
        tool_name=probe.name if probe is not None else "probe_never_registered",
        probe=probe,
        must_not_run=must_not_run,
        cleanup=cleanup,
    )


def _drive_execute_and_learn(rig: Rig, action: dict[str, Any], drive_relief_only: bool) -> Any:
    sim = MagicMock()
    sim.is_sim_mode = False
    outcome = tool_dispatch.execute_and_learn(
        agent=MagicMock(_execution_tracker=None),
        agent_name="aut",
        agent_id=AGENT_ID,
        executor=rig.executor,
        sim=sim,
        state=MagicMock(data={"active_goal": GOAL}),
        environment=MagicMock(),
        memory=MagicMock(),
        hippocampus=None,
        memory_hub=rig.memory_hub,
        result_cache=MagicMock(),
        autonomy_controller=rig.autonomy,
        rec_outcome=functools.partial(tool_dispatch.record_outcome, drive_relief_only=drive_relief_only),
        recent_outcomes=rig.recent_outcomes,
        max_recent=10,
        llm_worker=rig.llm_worker,
        context_pool=MagicMock(),
        nac=rig.nac,
        run_id="run_refusal",
        overlay={},
        action=action,
        confidence=0.9,
        proposal=SimpleNamespace(
            reasoning=GOAL, citations=[], cluster_id=CLUSTER, clusters=None, triggering_input="probe"
        ),
        observation={},
        human_involved=False,
    )
    require(not outcome.raised, f"execute_and_learn raised instead of refusing: {outcome.error}")


def _drive_parallel(rig: Rig, action: dict[str, Any], drive_relief_only: bool) -> None:
    results, _ = tool_dispatch.execute_parallel_actions(
        agent_id=AGENT_ID,
        actions=[action],
        executor=rig.executor,
        autonomy_controller=rig.autonomy,
        confidence=0.9,
        reasoning=GOAL,
        recent_outcomes=rig.recent_outcomes,
        max_recent=10,
        llm_worker=rig.llm_worker,
        context_pool=MagicMock(),
        nac=rig.nac,
        active_goal=GOAL,
        cluster_id=CLUSTER,
        drive_relief_only=drive_relief_only,
    )
    require(len(results) == 1, f"the batch returned {len(results)} results")
    require(not str(results[0]["error"]).startswith("Rejected:"), "the autonomy gate, not the executor, refused")


def _nac_state(nac: NAc) -> dict[str, str]:
    missing = [name for name in _NAC_STATE if not hasattr(nac, name)]
    require(not missing, f"NAc no longer has {missing}: update _NAC_STATE")
    return {name: repr(getattr(nac, name)) for name in _NAC_STATE}


def _check_refusal(kind: str, outputs: list[Any]) -> None:
    require(len(outputs) == 1, f"the executor dispatched {len(outputs)} times")
    out = outputs[0]
    require(getattr(out, "success", None) is False, f"the {kind} call was not refused: {out!r}")
    error_kind = getattr(getattr(out, "error_kind", None), "name", None)
    expected = KIND_TO_ERROR_KIND[kind]
    if kind in _LEGACY_TEXT:
        require(
            error_kind in expected or _LEGACY_TEXT[kind] in str(out.error or ""),
            f"not the {kind} refusal: {error_kind} / {out.error!r}",
        )
    else:  # typing the refusal is item 3's own behaviour, so a wrong kind is the defect
        assert error_kind in expected, (
            f"the {kind} refusal is not typed {sorted(expected)}: {error_kind} / {out.error!r}"
        )


@pytest.mark.parametrize("drive_relief_only", [False, True], ids=["substrate_primary", "llm_primary"])
@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason=(
        "#1315 item 3 / §1A.2: a gate refusal books learning today (NAc observe, goal and cluster credit, plan "
        "bridge) and the batch writes no autonomy audit entry; the new kinds' APIs do not exist. Flips when "
        "record_outcome's refused= branch (GATE_REFUSAL_KINDS) lands for this kind on this path"
    ),
)
def test_a_gate_refusal_books_no_learning(kind: str, path: str, drive_relief_only: bool) -> None:
    """One refusal of ``kind`` through ``path``: zero learning writes and unchanged NAc state, and exactly one
    carryover, one ``recent_outcomes`` and one autonomy audit entry (§1A.2's booking table, "After" column)."""
    rig = _rig(kind)
    try:
        action = {"tool_name": rig.tool_name, "params": {}}
        require(rig.autonomy.can_execute_action(action, confidence=0.9)[0], "the autonomy gate refuses the action")
        outputs: list[Any] = []
        real_execute = rig.executor.execute

        def capture(act: dict[str, Any]) -> Any:
            out = real_execute(act)
            outputs.append(out)
            return out

        rig.executor.execute = capture
        bridge = rig.executor._tool_pain_bridge
        before = _nac_state(rig.nac)
        with (
            patch.object(rig.nac, "observe", wraps=rig.nac.observe) as nac_observe,
            patch.object(rig.nac, "credit_goal", wraps=rig.nac.credit_goal) as credit_goal,
            patch.object(rig.nac, "update_cluster_reward", wraps=rig.nac.update_cluster_reward) as cluster_reward,
            patch.object(rig.bus, "publish", wraps=rig.bus.publish) as pain_publish,
            patch.object(bridge, "record_tool_start", wraps=bridge.record_tool_start) as b_start,
            patch.object(bridge, "record_tool_complete", wraps=bridge.record_tool_complete) as b_complete,
            patch.object(bridge, "record_tool_failure", wraps=bridge.record_tool_failure) as b_failure,
            patch.object(
                bridge, "record_tool_embodiment_failure", wraps=bridge.record_tool_embodiment_failure
            ) as b_embodiment,
        ):
            spies = {
                "NAc.observe": nac_observe,
                "NAc.credit_goal (goal credit)": credit_goal,
                "NAc.update_cluster_reward (cluster reward)": cluster_reward,
                "PainBus.publish": pain_publish,
                "ToolPainBridge.record_tool_start": b_start,
                "ToolPainBridge.record_tool_complete": b_complete,
                "ToolPainBridge.record_tool_failure": b_failure,
                "ToolPainBridge.record_tool_embodiment_failure": b_embodiment,
                "MemoryHub.record_plan_outcome (PlanHistoryBridge)": rig.memory_hub.record_plan_outcome,
            }
            if path == "execute_and_learn":
                _drive_execute_and_learn(rig, action, drive_relief_only)
            else:
                _drive_parallel(rig, action, drive_relief_only)
        after = _nac_state(rig.nac)

        require(not (rig.must_not_run and getattr(rig.probe, "ran", False)), "the refused tool ran")
        _check_refusal(kind, outputs)

        writes = {name: spy.call_count for name, spy in spies.items() if spy.call_count}
        changed = sorted(name for name in _NAC_STATE if before[name] != after[name])
        kept = {
            "llm_worker.record_outcome (carryover)": rig.llm_worker.record_outcome.call_count,
            "recent_outcomes": len(rig.recent_outcomes),
            "AutonomyController.audit_log": len(rig.autonomy.audit_log),
        }
        wrong_kept = {name: n for name, n in kept.items() if n != 1}
        assert not writes and not changed and not wrong_kept, (
            f"{kind} via {path}: learning writes {writes or '{}'}; NAc state changed in {changed or '[]'}; "
            f"sinks that must hold exactly one entry: {wrong_kept or 'ok'}"
        )
    finally:
        if rig.cleanup is not None:
            rig.cleanup()


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="#1315 item 3 / §1A.2: tools/base.py has no GATE_REFUSAL_KINDS yet",
)
def test_cases_cover_every_gate_refusal_kind() -> None:
    """Every member of ``GATE_REFUSAL_KINDS`` has a case in ``KIND_TO_ERROR_KIND``, so a new kind joins the gate.
    Fails today: the frozenset does not exist."""
    import maxim.tools.base as base

    assert hasattr(base, "GATE_REFUSAL_KINDS"), "item 3 not built: tools/base.py has no GATE_REFUSAL_KINDS"
    covered = set().union(*KIND_TO_ERROR_KIND.values())
    missing = sorted(m.name for m in base.GATE_REFUSAL_KINDS if m.name not in covered)
    assert not missing, f"GATE_REFUSAL_KINDS members with no booking case: {missing}"
