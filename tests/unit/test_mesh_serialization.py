"""Unit tests for mesh pre-work: serialization + NAc imported link registration.

Covers to_dict/from_dict round-trips for SubGoal, ProposedGoal, ToolResult,
and RuntimeCapabilities, plus NAc._register_imported_link().
"""

from __future__ import annotations

import time

from maxim.agents.bus import (
    FailureStrategy,
    GoalPriority,
    ProposedGoal,
    SubGoal,
    SubGoalStatus,
    ToolResult,
)
from maxim.runtime.capabilities import RuntimeCapabilities


# ─────────────────────────────────────────────────────────────────────────────
# SubGoal serialization
# ─────────────────────────────────────────────────────────────────────────────


class TestSubGoalSerialization:
    def test_round_trip_defaults(self):
        sg = SubGoal(id="sg1", description="look around", tool_name="look_around")
        d = sg.to_dict()
        restored = SubGoal.from_dict(d)
        assert restored.id == sg.id
        assert restored.tool_name == sg.tool_name
        assert restored.status == SubGoalStatus.PENDING
        assert restored.on_failure == FailureStrategy.RETRY
        assert restored.depends_on == []
        assert restored.priority_override is None

    def test_round_trip_all_fields(self):
        sg = SubGoal(
            id="sg2",
            description="grab cup",
            tool_name="grasp",
            tool_params={"target": "cup"},
            status=SubGoalStatus.COMPLETED,
            result={"grasped": True},
            error=None,
            attempts=1,
            max_retries=3,
            priority_override=GoalPriority.HIGH,
            on_failure=FailureStrategy.ESCALATE,
            depends_on=["sg1"],
            created_at=1000.0,
            started_at=1001.0,
            completed_at=1002.0,
        )
        d = sg.to_dict()
        restored = SubGoal.from_dict(d)
        assert restored.status == SubGoalStatus.COMPLETED
        assert restored.priority_override == GoalPriority.HIGH
        assert restored.on_failure == FailureStrategy.ESCALATE
        assert restored.depends_on == ["sg1"]
        assert restored.tool_params == {"target": "cup"}
        assert restored.started_at == 1001.0
        assert restored.completed_at == 1002.0

    def test_enum_serialized_as_name(self):
        sg = SubGoal(
            id="sg3",
            description="test",
            tool_name="t",
            status=SubGoalStatus.FAILED,
            on_failure=FailureStrategy.ABORT_PARENT,
        )
        d = sg.to_dict()
        assert d["status"] == "FAILED"
        assert d["on_failure"] == "ABORT_PARENT"


# ─────────────────────────────────────────────────────────────────────────────
# ProposedGoal serialization
# ─────────────────────────────────────────────────────────────────────────────


class TestProposedGoalSerialization:
    def test_round_trip_simple(self):
        goal = ProposedGoal(
            id="g1",
            description="find the cup",
            priority=GoalPriority.MEDIUM,
            tool_name="look_around",
        )
        d = goal.to_dict()
        restored = ProposedGoal.from_dict(d)
        assert restored.id == "g1"
        assert restored.priority == GoalPriority.MEDIUM
        assert restored.tool_name == "look_around"
        assert restored.sub_goals == []

    def test_round_trip_with_sub_goals(self):
        sg = SubGoal(id="sg1", description="step 1", tool_name="look_around")
        goal = ProposedGoal(
            id="g2",
            description="complex goal",
            priority=GoalPriority.HIGH,
            reasoning="because",
            confidence=0.9,
            sub_goals=[sg],
        )
        d = goal.to_dict()
        restored = ProposedGoal.from_dict(d)
        assert len(restored.sub_goals) == 1
        assert restored.sub_goals[0].id == "sg1"
        assert restored.confidence == 0.9
        assert restored.reasoning == "because"

    def test_priority_serialized_as_name(self):
        goal = ProposedGoal(id="g3", description="test", priority=GoalPriority.CRITICAL)
        d = goal.to_dict()
        assert d["priority"] == "CRITICAL"


# ─────────────────────────────────────────────────────────────────────────────
# ToolResult serialization
# ─────────────────────────────────────────────────────────────────────────────


class TestToolResultSerialization:
    def test_round_trip_success(self):
        tr = ToolResult(
            tool_call_id="tc1",
            tool_name="look_around",
            success=True,
            result={"objects": ["cup", "table"]},
            params={"direction": "left"},
        )
        d = tr.to_dict()
        restored = ToolResult.from_dict(d)
        assert restored.tool_call_id == "tc1"
        assert restored.success is True
        assert restored.result == {"objects": ["cup", "table"]}
        assert restored.params == {"direction": "left"}
        assert restored.error is None

    def test_round_trip_failure(self):
        tr = ToolResult(
            tool_call_id="tc2",
            tool_name="grasp",
            success=False,
            error="object out of reach",
        )
        d = tr.to_dict()
        restored = ToolResult.from_dict(d)
        assert restored.success is False
        assert restored.error == "object out of reach"
        assert restored.result is None


# ─────────────────────────────────────────────────────────────────────────────
# RuntimeCapabilities serialization
# ─────────────────────────────────────────────────────────────────────────────


class TestRuntimeCapabilitiesSerialization:
    def test_round_trip_defaults(self):
        caps = RuntimeCapabilities()
        d = caps.to_dict()
        restored = RuntimeCapabilities.from_dict(d)
        assert restored.has_gpu is False
        assert restored.vram_gb == 0.0
        assert restored.connected_devices == []

    def test_round_trip_populated(self):
        caps = RuntimeCapabilities(
            has_gpu=True,
            has_robot=True,
            gpu_type="RTX 5080",
            robot_type="reachy",
            vram_gb=16.0,
            ram_gb=64.0,
            connected_devices=["camera", "gripper"],
        )
        d = caps.to_dict()
        restored = RuntimeCapabilities.from_dict(d)
        assert restored.has_gpu is True
        assert restored.gpu_type == "RTX 5080"
        assert restored.robot_type == "reachy"
        assert restored.vram_gb == 16.0
        assert restored.connected_devices == ["camera", "gripper"]

    def test_ignores_unknown_fields(self):
        d = {"has_gpu": True, "future_field": "ignored", "vram_gb": 8.0}
        caps = RuntimeCapabilities.from_dict(d)
        assert caps.has_gpu is True
        assert caps.vram_gb == 8.0


# ─────────────────────────────────────────────────────────────────────────────
# NAc._register_imported_link
# ─────────────────────────────────────────────────────────────────────────────


class TestNAcRegisterImportedLink:
    def test_imported_link_appears_in_get_links(self, nac):
        from maxim.decisions.causal_link import CausalLink, Valence

        link = CausalLink(
            id="imported_001",
            event_type="tool",
            event_signature="tool:grasp",
            event_context={"_imported_from": "peer-A", "_import_trust": "verified"},
            outcome_type="result",
            outcome_signature="grasp_failed",
            outcome_valence=Valence.NEGATIVE,
            predicted_value=0.5,
            temporal_delta=1.0,
            observation_count=1,
            confidence=0.3,
            last_observed=time.time(),
        )
        nac._register_imported_link(link)

        found = nac.get_links_for_event("tool:grasp")
        assert len(found) == 1
        assert found[0].id == "imported_001"
        assert found[0].confidence == 0.3

    def test_imported_link_in_outcome_index(self, nac):
        from maxim.decisions.causal_link import CausalLink, Valence

        link = CausalLink(
            id="imported_002",
            event_type="tool",
            event_signature="tool:navigate",
            event_context={},
            outcome_type="result",
            outcome_signature="collision",
            outcome_valence=Valence.NEGATIVE,
            predicted_value=0.5,
            temporal_delta=2.0,
            observation_count=1,
            confidence=0.25,
            last_observed=time.time(),
        )
        nac._register_imported_link(link)

        outcome_links = nac.get_links_for_outcome("collision")
        assert any(lk.id == "imported_002" for lk in outcome_links)

    def test_imported_link_coexists_with_local(self, nac, valence_positive):
        from maxim.decisions.causal_link import CausalLink, Valence

        # Create a local link via observe
        nac.observe(
            event_type="tool",
            event_signature="tool:look",
            outcome_type="result",
            outcome_signature="found_object",
            outcome_valence=valence_positive,
            delta_seconds=0.5,
        )

        # Import a peer link for the same event
        imported = CausalLink(
            id="imported_003",
            event_type="tool",
            event_signature="tool:look",
            event_context={"_imported_from": "peer-B"},
            outcome_type="result",
            outcome_signature="found_nothing",
            outcome_valence=Valence.NEUTRAL,
            predicted_value=0.5,
            temporal_delta=1.0,
            observation_count=1,
            confidence=0.2,
            last_observed=time.time(),
        )
        nac._register_imported_link(imported)

        links = nac.get_links_for_event("tool:look")
        assert len(links) == 2
        ids = {lk.id for lk in links}
        assert "imported_003" in ids
