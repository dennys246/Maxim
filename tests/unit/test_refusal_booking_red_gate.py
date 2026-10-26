"""#1315 §1A.2 behavioural red gate: a gate refusal books NO learning, through both dispatch paths.

WHAT THIS FILE IS. Tests only (``docs/plans/atonia_gate_approach_notes.md`` DRAFT v6.2 §1A.2, "The behavioural red
gate is the enforcement, not this table"). For every refusal kind that exists TODAY at the executor -- the live
mode, ``AgentPermissions``, an unregistered tool and an inactive scene tool -- one refusal is driven through
``runtime/tool_dispatch.py::execute_and_learn`` AND ``execute_parallel_actions`` on the cheapest REAL fixtures: a
real ``NAc``, a real ``PainBus`` (``build_pain_bus``) behind a real ``PainDetector``, the real ``ToolPainBridge``
from ``build_executor``, a real ``ToolRegistry``, a real ``AutonomyController`` and the real ``record_outcome``.

THE ASSERTION. Zero learning writes -- ``NAc.observe``, the pain bus's ``publish``, ``ToolPainBridge``'s booking
methods (``record_tool_start``/``_complete``/``_failure``/``_embodiment_failure``), the plan bridge
(``MemoryHub.record_plan_outcome``, the only path to ``PlanHistoryBridge.record_plan_outcome``), goal credit
(``NAc.credit_goal``) and cluster reward (``NAc.update_cluster_reward``) -- AND exactly one entry in each of the
carryover (``llm_worker.record_outcome``), ``recent_outcomes`` and the autonomy audit log
(``AutonomyController.audit_log``), because §1A.2 keeps those so the model is told and the
``max_consecutive_failures`` counter still stops a model that keeps naming a refused tool.

TODAY every case books a negative through ``record_outcome`` (``NAc.observe`` plus goal and cluster credit), the
unregistered and inactive-scene cases also book pain through ``_report_failure``, and the parallel batch writes no
autonomy audit entry at all. So each case is its own ``xfail(strict=True, raises=AssertionError)``: it flips when
item 3's ``refused=`` keyword (and its booking branch) lands for that kind on that path. A case that already books
zero writes is not marked xfail (none does today).

The scenario (the executor refused, with the expected error text, and the autonomy gate admitted the action, so
the refusal under test is the executor's) is checked first and raises ``ScenarioBroken``, never an
``AssertionError``.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from maxim.agents.autonomy import AutonomyController, AutonomyLevel
from maxim.agents.permissions import AgentPermissions
from maxim.bridges.tool_pain_bridge import ToolPainBridge
from maxim.decisions.nac import NAc, NACConfig
from maxim.proprioception.pain import PainDetector
from maxim.proprioception.pain_bus import build_pain_bus
from maxim.runtime import tool_dispatch
from maxim.runtime.bootstrap import build_executor
from maxim.tools.base import Tool, ToolOutput
from maxim.tools.registry import ToolRegistry

AGENT_ID = "refusal_gate"
GOAL = "probe the gate"
CLUSTER = "c-refusal"


class ScenarioBroken(Exception):
    """The scenario did not happen as designed: NOT the defect a gate pins (module docstring)."""


def require(cond: Any, msg: str) -> None:
    if not cond:
        raise ScenarioBroken(msg)


class _Probe(Tool):
    """A tool that would succeed if it ran; every case here must refuse it before it runs."""

    description = "stub"
    input_schema: dict = {}

    def __init__(self, name: str) -> None:
        self.name = name
        super().__init__()
        self.ran = False

    def execute(self, **kwargs: Any) -> Any:
        self.ran = True
        return ToolOutput(success=True, output="ran")


@dataclass
class Rig:
    executor: Any
    nac: NAc
    bus: Any
    autonomy: AutonomyController
    llm_worker: MagicMock
    memory_hub: MagicMock
    recent_outcomes: list[dict[str, Any]] = field(default_factory=list)
    probe: _Probe | None = None
    tool_name: str = ""
    expect_error: str = ""


def _rig(kind: str) -> Rig:
    """One executor refusing ``tool_name`` for ``kind``, on the real learning sinks."""
    nac = NAc(NACConfig())
    bus = build_pain_bus(hippocampus=None, nac=nac)
    detector = PainDetector(pain_bus=bus, agent_id=AGENT_ID)
    registry = ToolRegistry()
    permissions = None
    probe: _Probe | None = None
    if kind == "mode":
        probe = _Probe("bash")  # HOST_ACTING_TOOLS: passive mode refuses it at dispatch (#826)
        registry.register(probe)
        expect = "passive mode"
    elif kind == "permissions":
        probe = _Probe("probe_denied")
        registry.register(probe)
        permissions = AgentPermissions(tool_deny=frozenset({"probe_denied"}))
        expect = "probe_denied"
    elif kind == "unregistered":
        expect = "Tool not registered"
    elif kind == "inactive_scene":
        probe = _Probe("probe_scene")
        registry.register_scene_tools([probe], scene_id="scene-1")
        registry.deactivate_scene("scene-1")
        expect = "is not active"
    else:  # pragma: no cover - the parametrisation is closed
        raise ScenarioBroken(f"unknown refusal kind {kind!r}")
    executor = build_executor(registry, pain_bus=None, permissions=permissions, nac=nac, pain_detector=detector)
    if kind == "mode":
        executor.set_operational_override("passive")
    tool_name = probe.name if probe is not None else "probe_never_registered"
    require(isinstance(executor._tool_pain_bridge, ToolPainBridge), "build_executor built no ToolPainBridge")
    autonomy = AutonomyController(initial_level=AutonomyLevel.AUTONOMOUS)
    return Rig(
        executor=executor,
        nac=nac,
        bus=bus,
        autonomy=autonomy,
        llm_worker=MagicMock(),
        memory_hub=MagicMock(spec_set=["record_plan_outcome"]),
        probe=probe,
        tool_name=tool_name,
        expect_error=expect,
    )


def _drive_execute_and_learn(rig: Rig, action: dict[str, Any]) -> Any:
    sim = MagicMock()
    sim.is_sim_mode = False
    return tool_dispatch.execute_and_learn(
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
        rec_outcome=functools.partial(tool_dispatch.record_outcome, drive_relief_only=False),
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


def _drive_parallel(rig: Rig, action: dict[str, Any]) -> Any:
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
    )
    return results


KINDS = ("mode", "permissions", "unregistered", "inactive_scene")
PATHS = ("execute_and_learn", "execute_parallel_actions")

# Today's verdict per (kind, path), measured: every case books learning writes (module docstring). A case that
# books zero writes today goes in ``PASSES_TODAY`` and is not marked xfail.
PASSES_TODAY: frozenset[tuple[str, str]] = frozenset()


def _case(kind: str, path: str) -> Any:
    marks = ()
    if (kind, path) not in PASSES_TODAY:
        marks = (
            pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason=(
                    f"#1315 item 3 / §1A.2: the {kind} refusal through {path} books learning today; flips when "
                    "record_outcome's refused= branch (GATE_REFUSAL_KINDS) books no NAc/pain/goal/cluster/plan write"
                    + (", and the batch writes its autonomy audit entry" if path == "execute_parallel_actions" else "")
                ),
            ),
        )
    return pytest.param(kind, path, marks=marks, id=f"{kind}-{path}")


@pytest.mark.parametrize(("kind", "path"), [_case(k, p) for k in KINDS for p in PATHS])
def test_a_gate_refusal_books_no_learning(kind: str, path: str) -> None:
    """One refusal of ``kind`` through ``path``: zero learning writes, and exactly one carryover, one
    ``recent_outcomes`` and one autonomy audit entry (§1A.2's booking table, "After" column)."""
    rig = _rig(kind)
    action = {"tool_name": rig.tool_name, "params": {}}
    require(rig.autonomy.can_execute_action(action, confidence=0.9)[0], "the autonomy gate refuses the action itself")

    bridge = rig.executor._tool_pain_bridge
    spies: dict[str, MagicMock] = {}
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
            outcome = _drive_execute_and_learn(rig, action)
            error = outcome.error
            require(not outcome.raised, f"execute_and_learn raised instead of refusing: {outcome.error}")
        else:
            results = _drive_parallel(rig, action)
            require(len(results) == 1, f"the batch returned {len(results)} results")
            error = results[0]["error"]
            require(not str(error).startswith("Rejected:"), f"the autonomy gate, not the executor, refused: {error}")

    require(rig.probe is None or not rig.probe.ran, "the refused tool ran")
    require(rig.expect_error in str(error or ""), f"not the {kind} refusal: {error!r}")

    writes = {name: spy.call_count for name, spy in spies.items() if spy.call_count}
    kept = {
        "llm_worker.record_outcome (carryover)": rig.llm_worker.record_outcome.call_count,
        "recent_outcomes": len(rig.recent_outcomes),
        "AutonomyController.audit_log": len(rig.autonomy.audit_log),
    }
    wrong_kept = {name: n for name, n in kept.items() if n != 1}
    assert not writes and not wrong_kept, (
        f"{kind} via {path}: learning writes {writes or '{}'}; sinks that must hold exactly one entry: {wrong_kept or 'ok'}"
    )
