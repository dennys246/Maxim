"""Unit tests for plan_manager.py — PlanManager lifecycle, phase advancement, replan."""

from __future__ import annotations

import json
import os
import time
from unittest.mock import Mock

import pytest

from maxim.agents.bus import (
    AgentBus,
    FailureStrategy,
    PhaseCompleted,
    PhaseStarted,
    PlanCompleted,
    PlanCreated,
    PlanReplanRequested,
    PlanRestored,
    SubGoal,
    SubGoalStatus,
)
from maxim.planning.plan_document import (
    LongHorizonConfig,
    Phase,
    PhaseEnergyBudget,
    PhaseStatus,
    PlanDocument,
    PlanEnergyBudget,
    PlanStatus,
)
from maxim.planning.plan_manager import PlanManager, PlanServices


# ─────────────────────────────────────────────────────────────────────────────
# Factories
# ─────────────────────────────────────────────────────────────────────────────


def make_phase(
    plan_id="plan-1", index=0, description="Test phase", status="PENDING",
    sub_goals=None, expected_inputs=None, expected_outputs=None, phase_id=None,
):
    return Phase(
        id=phase_id or f"phase-{plan_id}-{index}",
        description=description, status=PhaseStatus[status], plan_id=plan_id,
        index=index, sub_goals=sub_goals or [],
        expected_inputs=expected_inputs or {}, expected_outputs=expected_outputs or {},
    )


def make_plan_document(
    num_phases=3, objective="Test objective", status="ACTIVE",
    plan_id="plan-1", with_sub_goals=False,
):
    phases = []
    for i in range(num_phases):
        phase_status = PhaseStatus.ACTIVE if i == 0 else PhaseStatus.PENDING
        sgs = []
        if with_sub_goals:
            sg1 = SubGoal(
                id=f"sg-{plan_id}-{i}-0", description=f"Sub-goal {i}.0",
                tool_name=f"tool_{i}_0", on_failure=FailureStrategy.REPLAN,
            )
            sg2 = SubGoal(
                id=f"sg-{plan_id}-{i}-1", description=f"Sub-goal {i}.1",
                tool_name=f"tool_{i}_1", depends_on=[sg1.id],
            )
            sgs = [sg1, sg2]
        phase = Phase(
            id=f"phase-{plan_id}-{i}", description=f"Phase {i}: step {i}",
            status=phase_status, plan_id=plan_id, index=i, sub_goals=sgs,
            expected_outputs={f"output_{i}": f"result from phase {i}"},
        )
        if i == 0:
            phase.started_at = time.time()
        phases.append(phase)
    now = time.time()
    return PlanDocument(
        id=plan_id, objective=objective, created_at=now, updated_at=now,
        status=PlanStatus[status], phases=phases, current_phase_index=0,
    )


def make_plan_manager(tmp_path, bus=None, config=None, services=None):
    from pathlib import Path
    plans_dir = str(Path(tmp_path) / "plans")
    return PlanManager(
        plans_dir=plans_dir, bus=bus or AgentBus(),
        config=config or LongHorizonConfig(), services=services or PlanServices(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


class EventCollector:
    """Subscribes to bus events and collects them for assertions."""

    def __init__(self, bus: AgentBus):
        self.events: list = []
        for event_type in [
            PlanCreated, PhaseStarted, PhaseCompleted,
            PlanCompleted, PlanRestored, PlanReplanRequested,
        ]:
            bus.subscribe(event_type, self.events.append)

    def of_type(self, event_type: type) -> list:
        return [e for e in self.events if isinstance(e, event_type)]


# ─────────────────────────────────────────────────────────────────────────────
# Phase advancement
# ─────────────────────────────────────────────────────────────────────────────


class TestAdvancePhase:
    def test_advance_phase_success(self, tmp_path):
        bus = AgentBus()
        collector = EventCollector(bus)
        mgr = make_plan_manager(tmp_path, bus=bus)

        phases = [
            make_phase(index=0, description="Phase 0", status="ACTIVE"),
            make_phase(index=1, description="Phase 1"),
        ]
        mgr.create_plan("p-1", "Test", phases)

        mgr.advance_phase(success=True, result="done")

        # Phase 0 completed, phase 1 active
        assert mgr.active_plan is not None
        assert mgr.active_plan.current_phase_index == 1
        assert mgr.active_plan.phases[0].status == PhaseStatus.COMPLETED
        assert mgr.active_plan.phases[1].status == PhaseStatus.ACTIVE

        # Verify events
        phase_completed = collector.of_type(PhaseCompleted)
        assert len(phase_completed) == 1
        assert phase_completed[0].success is True

        phase_started = collector.of_type(PhaseStarted)
        # 1 from create_plan (phase 0) + 1 from advance (phase 1)
        assert len(phase_started) == 2

    def test_advance_to_plan_completion(self, tmp_path):
        bus = AgentBus()
        collector = EventCollector(bus)
        mgr = make_plan_manager(tmp_path, bus=bus)

        phases = [make_phase(index=0, description="Only phase", status="ACTIVE")]
        mgr.create_plan("p-1", "Test", phases)

        mgr.advance_phase(success=True, result="all done")

        # Plan should be completed and cleared
        assert mgr.active_plan is None
        plan_completed = collector.of_type(PlanCompleted)
        assert len(plan_completed) == 1
        assert plan_completed[0].success is True

    def test_advance_publishes_events_in_order(self, tmp_path):
        bus = AgentBus()
        collector = EventCollector(bus)
        mgr = make_plan_manager(tmp_path, bus=bus)

        phases = [
            make_phase(index=0, description="P0", status="ACTIVE"),
            make_phase(index=1, description="P1"),
        ]
        mgr.create_plan("p-1", "Test", phases)
        collector.events.clear()

        mgr.advance_phase(success=True, result="done")

        # PhaseCompleted should come before PhaseStarted
        types = [type(e).__name__ for e in collector.events]
        assert types.index("PhaseCompleted") < types.index("PhaseStarted")

    def test_advance_no_plan(self, tmp_path):
        mgr = make_plan_manager(tmp_path)
        # Should not raise
        mgr.advance_phase(success=True, result="nothing")


# ─────────────────────────────────────────────────────────────────────────────
# Phase failure handling
# ─────────────────────────────────────────────────────────────────────────────


class TestHandlePhaseFailure:
    def test_replan_eligible_triggers_replanning(self, tmp_path):
        bus = AgentBus()
        collector = EventCollector(bus)
        mgr = make_plan_manager(tmp_path, bus=bus)

        sg = SubGoal(
            id="sg-1", description="search", tool_name="search",
            on_failure=FailureStrategy.REPLAN,
        )
        phases = [
            make_phase(index=0, description="Research", status="ACTIVE", sub_goals=[sg]),
            make_phase(index=1, description="Report"),
        ]
        mgr.create_plan("p-1", "Test", phases)

        mgr.advance_phase(success=False, error="no results")

        assert mgr.active_plan.status == PlanStatus.REPLANNING
        replan_events = collector.of_type(PlanReplanRequested)
        assert len(replan_events) == 1
        assert replan_events[0].reason == "no results"

    def test_exhaust_replan_attempts_fails_plan(self, tmp_path):
        config = LongHorizonConfig(max_replan_attempts=1)
        bus = AgentBus()
        mgr = make_plan_manager(tmp_path, bus=bus, config=config)

        sg = SubGoal(
            id="sg-1", description="search", tool_name="search",
            on_failure=FailureStrategy.REPLAN,
        )
        phases = [
            make_phase(index=0, description="Research", status="ACTIVE",
                       sub_goals=[sg], phase_id="phase-p-1-0"),
            make_phase(index=1, description="Report"),
        ]
        mgr.create_plan("p-1", "Test", phases)

        # First failure triggers replan
        mgr.advance_phase(success=False, error="fail 1")
        assert mgr.active_plan.status == PlanStatus.REPLANNING

        # Accept replan and fail again
        revised = make_phase(index=0, description="Research v2", status="ACTIVE",
                             sub_goals=[sg])
        mgr.accept_replan(revised)
        assert mgr.active_plan.status == PlanStatus.ACTIVE

        # Second failure exhausts attempts
        mgr.advance_phase(success=False, error="fail 2")
        assert mgr.active_plan is None  # Plan should be cleared after failure

    def test_non_replan_failure_fails_plan(self, tmp_path):
        bus = AgentBus()
        mgr = make_plan_manager(tmp_path, bus=bus)

        sg = SubGoal(
            id="sg-1", description="critical", tool_name="tool",
            on_failure=FailureStrategy.ABORT_PARENT,
        )
        phases = [
            make_phase(index=0, description="Critical", status="ACTIVE", sub_goals=[sg]),
        ]
        mgr.create_plan("p-1", "Test", phases)

        mgr.advance_phase(success=False, error="catastrophic")

        # Plan should fail immediately (no replan)
        assert mgr.active_plan is None


# ─────────────────────────────────────────────────────────────────────────────
# Accept replan
# ─────────────────────────────────────────────────────────────────────────────


class TestAcceptReplan:
    def test_accept_replan_returns_to_active(self, tmp_path):
        bus = AgentBus()
        collector = EventCollector(bus)
        mgr = make_plan_manager(tmp_path, bus=bus)

        sg = SubGoal(id="sg-1", description="search", tool_name="search",
                      on_failure=FailureStrategy.REPLAN)
        phases = [
            make_phase(index=0, description="Research", status="ACTIVE", sub_goals=[sg]),
            make_phase(index=1, description="Report"),
        ]
        mgr.create_plan("p-1", "Test", phases)
        mgr.advance_phase(success=False, error="timeout")

        assert mgr.active_plan.status == PlanStatus.REPLANNING

        new_sg = SubGoal(id="sg-2", description="alt search", tool_name="alt_search")
        revised = make_phase(index=0, description="Research v2", sub_goals=[new_sg])
        mgr.accept_replan(revised)

        assert mgr.active_plan.status == PlanStatus.ACTIVE
        current = mgr.active_plan.get_current_phase()
        assert len(current.sub_goals) == 1
        assert current.sub_goals[0].id == "sg-2"

        # PhaseStarted with [REPLAN] prefix
        started = collector.of_type(PhaseStarted)
        replan_started = [s for s in started if "[REPLAN]" in s.description]
        assert len(replan_started) == 1


# ─────────────────────────────────────────────────────────────────────────────
# Session lifecycle
# ─────────────────────────────────────────────────────────────────────────────


class TestSessionLifecycle:
    def test_session_start_no_plans(self, tmp_path):
        mgr = make_plan_manager(tmp_path)
        result = mgr.on_session_start()
        assert result["plan_restored"] is None

    def test_session_start_restores_active_plan(self, tmp_path):
        bus = AgentBus()
        mgr = make_plan_manager(tmp_path, bus=bus)

        # Create and save a plan
        phases = [
            make_phase(index=0, description="P0", status="ACTIVE"),
            make_phase(index=1, description="P1"),
        ]
        mgr.create_plan("p-1", "Test", phases)

        # New manager, same directory
        mgr2 = PlanManager(
            plans_dir=mgr._plans_dir, bus=AgentBus(),
            config=LongHorizonConfig(),
        )
        collector = EventCollector(mgr2._bus)
        result = mgr2.on_session_start()

        assert result["plan_restored"] == "p-1"
        assert mgr2.active_plan is not None
        restored_events = collector.of_type(PlanRestored)
        assert len(restored_events) == 1

    def test_session_start_resets_in_progress_sub_goals(self, tmp_path):
        bus = AgentBus()
        mgr = make_plan_manager(tmp_path, bus=bus)

        sg = SubGoal(id="sg-1", description="mid-exec", tool_name="tool")
        sg.status = SubGoalStatus.IN_PROGRESS
        sg.started_at = time.time()
        phases = [make_phase(index=0, description="P0", status="ACTIVE", sub_goals=[sg])]
        mgr.create_plan("p-1", "Test", phases)

        # New manager, restoring
        mgr2 = PlanManager(
            plans_dir=mgr._plans_dir, bus=AgentBus(),
            config=LongHorizonConfig(),
        )
        mgr2.on_session_start()

        current = mgr2.active_plan.get_current_phase()
        assert current.sub_goals[0].status == SubGoalStatus.PENDING
        assert current.sub_goals[0].started_at is None

    def test_session_end_pauses_active_plan(self, tmp_path):
        bus = AgentBus()
        mgr = make_plan_manager(tmp_path, bus=bus)

        phases = [make_phase(index=0, description="P0", status="ACTIVE")]
        mgr.create_plan("p-1", "Test", phases)

        result = mgr.on_session_end()
        assert result["plan_saved"] == "p-1"

        # Verify it was saved as PAUSED
        loaded = PlanDocument.load(os.path.join(mgr._plans_dir, "p-1.json"))
        assert loaded.status == PlanStatus.PAUSED
        assert loaded.paused_reason == "session_end"

    def test_session_end_no_plan(self, tmp_path):
        mgr = make_plan_manager(tmp_path)
        result = mgr.on_session_end()
        assert result["plan_saved"] is None


# ─────────────────────────────────────────────────────────────────────────────
# Soft preemption
# ─────────────────────────────────────────────────────────────────────────────


class TestSoftPreemption:
    def test_preempt_pauses_old_plan(self, tmp_path):
        bus = AgentBus()
        mgr = make_plan_manager(tmp_path, bus=bus)

        # Create plan A
        phases_a = [make_phase(plan_id="pa", index=0, status="ACTIVE")]
        mgr.create_plan("pa", "Plan A", phases_a)

        # Create plan B (preempts A)
        phases_b = [make_phase(plan_id="pb", index=0, status="ACTIVE")]
        mgr.create_plan("pb", "Plan B", phases_b)

        assert mgr.active_plan.id == "pb"
        assert len(mgr._preempted_plans) == 1
        assert mgr._preempted_plans[0].status == PlanStatus.PAUSED
        assert mgr._preempted_plans[0].paused_reason == "soft_preempted"

    def test_preemption_disabled_abandons_immediately(self, tmp_path):
        config = LongHorizonConfig(enable_soft_preemption=False)
        bus = AgentBus()
        mgr = make_plan_manager(tmp_path, bus=bus, config=config)

        phases_a = [make_phase(plan_id="pa", index=0, status="ACTIVE")]
        mgr.create_plan("pa", "Plan A", phases_a)

        phases_b = [make_phase(plan_id="pb", index=0, status="ACTIVE")]
        mgr.create_plan("pb", "Plan B", phases_b)

        assert mgr.active_plan.id == "pb"
        assert len(mgr._preempted_plans) == 0

        # Load plan A — should be ABANDONED
        loaded = PlanDocument.load(os.path.join(mgr._plans_dir, "pa.json"))
        assert loaded.status == PlanStatus.ABANDONED

    def test_finalize_preempted_on_viability(self, tmp_path):
        config = LongHorizonConfig(preemption_viability_phase=1)
        bus = AgentBus()
        mgr = make_plan_manager(tmp_path, bus=bus, config=config)

        # Plan A
        phases_a = [make_phase(plan_id="pa", index=0, status="ACTIVE")]
        mgr.create_plan("pa", "Plan A", phases_a)

        # Plan B preempts A
        phases_b = [
            make_phase(plan_id="pb", index=0, status="ACTIVE"),
            make_phase(plan_id="pb", index=1),
        ]
        mgr.create_plan("pb", "Plan B", phases_b)

        # Advance B past viability threshold (phase 0 complete = index 0+1 = 1 >= 1)
        mgr.advance_phase(success=True, result="done")

        # A should be abandoned
        assert len(mgr._preempted_plans) == 0
        loaded = PlanDocument.load(os.path.join(mgr._plans_dir, "pa.json"))
        assert loaded.status == PlanStatus.ABANDONED

    def test_preemption_rollback_on_failure(self, tmp_path):
        config = LongHorizonConfig(preemption_viability_phase=2)
        bus = AgentBus()
        collector = EventCollector(bus)
        mgr = make_plan_manager(tmp_path, bus=bus, config=config)

        # Plan A
        phases_a = [
            make_phase(plan_id="pa", index=0, status="ACTIVE"),
            make_phase(plan_id="pa", index=1),
        ]
        mgr.create_plan("pa", "Plan A", phases_a)

        # Plan B preempts A (no replan-eligible sub-goals)
        sg = SubGoal(id="sg-b", description="b task", tool_name="tool",
                      on_failure=FailureStrategy.ABORT_PARENT)
        phases_b = [make_phase(plan_id="pb", index=0, status="ACTIVE", sub_goals=[sg])]
        mgr.create_plan("pb", "Plan B", phases_b)

        # B fails before viability
        mgr.advance_phase(success=False, error="fail")

        # A should be restored
        assert mgr.active_plan is not None
        assert mgr.active_plan.id == "pa"
        assert mgr.active_plan.status == PlanStatus.ACTIVE

        restored = collector.of_type(PlanRestored)
        assert len(restored) >= 1
        assert any(r.plan_id == "pa" for r in restored)

    def test_stale_preemption_auto_abandon(self, tmp_path):
        config = LongHorizonConfig(stale_preemption_hours=0.0)  # Immediate stale
        bus = AgentBus()
        mgr = make_plan_manager(tmp_path, bus=bus, config=config)

        # Create and manually save a stale preempted plan
        plans_dir = mgr._plans_dir
        os.makedirs(plans_dir, exist_ok=True)
        stale = make_plan_document(plan_id="stale", status="PAUSED")
        stale.paused_reason = "soft_preempted"
        stale.save(os.path.join(plans_dir, "stale.json"))

        # Patch updated_at on disk AFTER save (save() overwrites updated_at)
        path = os.path.join(plans_dir, "stale.json")
        with open(path) as f:
            data = json.load(f)
        data["updated_at"] = time.time() - 100
        with open(path, "w") as f:
            json.dump(data, f)

        mgr.on_session_start()

        # Stale plan should have been abandoned
        loaded = PlanDocument.load(os.path.join(plans_dir, "stale.json"))
        assert loaded.status == PlanStatus.ABANDONED


# ─────────────────────────────────────────────────────────────────────────────
# Plan creation
# ─────────────────────────────────────────────────────────────────────────────


class TestPlanCreation:
    def test_create_plan_publishes_events(self, tmp_path):
        bus = AgentBus()
        collector = EventCollector(bus)
        mgr = make_plan_manager(tmp_path, bus=bus)

        phases = [make_phase(index=0, description="P0")]
        mgr.create_plan("p-1", "Test", phases)

        created = collector.of_type(PlanCreated)
        assert len(created) == 1
        assert created[0].plan_id == "p-1"
        assert created[0].objective == "Test"

        started = collector.of_type(PhaseStarted)
        assert len(started) == 1

    def test_create_plan_saves_to_disk(self, tmp_path):
        mgr = make_plan_manager(tmp_path)

        phases = [make_phase(index=0, description="P0")]
        mgr.create_plan("p-1", "Test", phases)

        assert os.path.exists(os.path.join(mgr._plans_dir, "p-1.json"))

    def test_create_plan_first_phase_active(self, tmp_path):
        mgr = make_plan_manager(tmp_path)

        phases = [make_phase(index=0), make_phase(index=1)]
        mgr.create_plan("p-1", "Test", phases)

        assert mgr.active_plan.phases[0].status == PhaseStatus.ACTIVE
        assert mgr.active_plan.phases[0].started_at is not None


# ─────────────────────────────────────────────────────────────────────────────
# Budget allocation with no services
# ─────────────────────────────────────────────────────────────────────────────


class TestBudgetAllocation:
    def test_allocate_phase_budget_no_services(self, tmp_path):
        """Budget allocation should not crash without services."""
        mgr = make_plan_manager(tmp_path)
        phases = [make_phase(index=0, status="ACTIVE")]
        mgr.create_plan("p-1", "Test", phases)

        # This should be a no-op since there's no energy budget or registry
        mgr._allocate_phase_budget(mgr.active_plan.phases[0])

    def test_allocate_phase_budget_with_mock_registry(self, tmp_path):
        mock_budget = Mock()
        mock_budget.current_level = 500.0
        mock_registry = Mock()
        mock_registry.get_budget = Mock(return_value=mock_budget)

        services = PlanServices(registry=mock_registry)
        mgr = make_plan_manager(tmp_path, services=services)

        phases = [make_phase(index=0, status="ACTIVE"), make_phase(index=1)]
        mgr.create_plan("p-1", "Test", phases)
        mgr.active_plan.total_energy_budget = PlanEnergyBudget(plan_id="p-1", total={"llm": 500.0})

        mgr._allocate_phase_budget(mgr.active_plan.phases[0])

        assert mgr.active_plan.phases[0].energy_budget is not None
        assert mgr.active_plan.phases[0].energy_budget.allocations.get("llm", 0) > 0


# ─────────────────────────────────────────────────────────────────────────────
# Replan context building
# ─────────────────────────────────────────────────────────────────────────────


class TestBuildReplanContext:
    def test_build_replan_context_no_services(self, tmp_path):
        """Should not crash with empty services."""
        mgr = make_plan_manager(tmp_path)
        phases = [make_phase(index=0, status="ACTIVE")]
        mgr.create_plan("p-1", "Test", phases)

        phase = mgr.active_plan.phases[0]
        ctx = mgr._build_replan_context(mgr.active_plan, phase, "test error", 0)

        assert ctx.failure_reason == "test error"
        assert ctx.original_objective == "Test"
        assert ctx.similar_past_failures == []
        assert ctx.alternative_approaches == []
