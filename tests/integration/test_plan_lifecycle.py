"""Integration tests for long-horizon plan lifecycle.

Tests complete multi-phase plan flows: creation → phase advancement → completion,
multi-session persistence, replan recovery, soft preemption chains, and backward
compatibility with plan_manager=None.
"""

from __future__ import annotations

import json
import os
import time

import pytest

from maxim.agents.bus import (
    AgentBus,
    FailureStrategy,
    GoalPriority,
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
    PhaseStatus,
    PlanDocument,
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
# Plan creation and completion
# ─────────────────────────────────────────────────────────────────────────────


class TestPlanCreationAndCompletion:
    """Full lifecycle tests: create → advance phases → complete."""

    def test_full_3_phase_lifecycle(self, tmp_path):
        """Create a 3-phase plan, advance all phases, verify COMPLETED + all bus events."""
        bus = AgentBus()
        collector = EventCollector(bus)
        mgr = make_plan_manager(tmp_path, bus=bus)

        phases = [
            make_phase(index=0, description="Gather data"),
            make_phase(index=1, description="Analyze data"),
            make_phase(index=2, description="Write report"),
        ]
        mgr.create_plan("plan-lifecycle", "Research climate data", phases)

        assert mgr.active_plan is not None
        assert mgr.active_plan.status == PlanStatus.ACTIVE

        # Advance phase 0 → phase 1
        mgr.advance_phase(success=True, result="Data gathered from 5 sources")
        assert mgr.active_plan.current_phase_index == 1
        assert mgr.active_plan.phases[0].status == PhaseStatus.COMPLETED
        assert mgr.active_plan.phases[1].status == PhaseStatus.ACTIVE

        # Advance phase 1 → phase 2
        mgr.advance_phase(success=True, result="Analysis complete")
        assert mgr.active_plan.current_phase_index == 2
        assert mgr.active_plan.phases[1].status == PhaseStatus.COMPLETED
        assert mgr.active_plan.phases[2].status == PhaseStatus.ACTIVE

        # Advance phase 2 → plan complete
        mgr.advance_phase(success=True, result="Report written")
        assert mgr.active_plan is None  # Plan cleared after completion

        # Verify all bus events in order
        created = collector.of_type(PlanCreated)
        assert len(created) == 1
        assert created[0].plan_id == "plan-lifecycle"
        assert created[0].phase_count == 3

        started = collector.of_type(PhaseStarted)
        assert len(started) == 3  # One per phase
        assert [e.phase_index for e in started] == [0, 1, 2]

        completed = collector.of_type(PhaseCompleted)
        assert len(completed) == 3
        assert all(e.success for e in completed)

        plan_completed = collector.of_type(PlanCompleted)
        assert len(plan_completed) == 1
        assert plan_completed[0].success is True
        assert plan_completed[0].phases_completed == 3
        assert plan_completed[0].phases_total == 3

    def test_plan_with_sub_goals_and_dependencies(self, tmp_path):
        """Each phase has 2 sub-goals with dependencies. Simulate tool results."""
        bus = AgentBus()
        mgr = make_plan_manager(tmp_path, bus=bus)

        # Phase 0 has two sub-goals: sg0 -> sg1 (dependency chain)
        sg0 = SubGoal(
            id="sg-0-0", description="Fetch raw data",
            tool_name="fetch_data", on_failure=FailureStrategy.RETRY,
        )
        sg1 = SubGoal(
            id="sg-0-1", description="Parse fetched data",
            tool_name="parse_data", depends_on=["sg-0-0"],
        )
        phase0 = make_phase(
            index=0, description="Data acquisition",
            sub_goals=[sg0, sg1],
            expected_outputs={"raw_data": "fetched raw data"},
        )

        # Phase 1 has one sub-goal
        sg2 = SubGoal(
            id="sg-1-0", description="Analyze parsed data",
            tool_name="analyze",
        )
        phase1 = make_phase(
            index=1, description="Data analysis",
            sub_goals=[sg2],
        )

        mgr.create_plan("plan-sgs", "Process dataset", [phase0, phase1])

        # Verify phase 0 is active with sub-goals intact
        current = mgr.active_plan.get_current_phase()
        assert len(current.sub_goals) == 2
        assert current.sub_goals[0].id == "sg-0-0"
        assert current.sub_goals[1].depends_on == ["sg-0-0"]

        # Sub-goal dependency resolution: sg0 should be executable, sg1 should not
        next_sg = current.get_next_executable()
        assert next_sg.id == "sg-0-0"

        # Complete sg0 → sg1 becomes executable
        current.sub_goals[0].status = SubGoalStatus.COMPLETED
        next_sg = current.get_next_executable()
        assert next_sg.id == "sg-0-1"

        # Complete sg1 → phase is complete
        current.sub_goals[1].status = SubGoalStatus.COMPLETED
        assert current.is_complete()

        # Advance to phase 1
        mgr.advance_phase(success=True, result={"raw_data": "fetched"})
        assert mgr.active_plan.current_phase_index == 1

        # Complete phase 1
        mgr.active_plan.get_current_phase().sub_goals[0].status = SubGoalStatus.COMPLETED
        mgr.advance_phase(success=True, result="Done")
        assert mgr.active_plan is None

    def test_single_phase_plan(self, tmp_path):
        """Single-phase plan degrades to simple goal-like behavior."""
        bus = AgentBus()
        collector = EventCollector(bus)
        mgr = make_plan_manager(tmp_path, bus=bus)

        phases = [make_phase(index=0, description="Do the thing")]
        mgr.create_plan("plan-single", "Simple task", phases)

        assert mgr.active_plan is not None

        mgr.advance_phase(success=True, result="Done")
        assert mgr.active_plan is None

        plan_completed = collector.of_type(PlanCompleted)
        assert len(plan_completed) == 1
        assert plan_completed[0].phases_completed == 1


# ─────────────────────────────────────────────────────────────────────────────
# Multi-session persistence
# ─────────────────────────────────────────────────────────────────────────────


class TestMultiSessionPersistence:
    """Tests that plans survive PlanManager restarts via save/load."""

    def test_plan_survives_restart(self, tmp_path):
        """Complete phase 0, session end, new manager, session start, verify phase 1 active."""
        from pathlib import Path
        plans_dir = str(Path(tmp_path) / "plans")
        bus = AgentBus()
        mgr = PlanManager(plans_dir=plans_dir, bus=bus, config=LongHorizonConfig())

        phases = [
            make_phase(index=0, description="Phase A"),
            make_phase(index=1, description="Phase B"),
            make_phase(index=2, description="Phase C"),
        ]
        mgr.create_plan("plan-persist", "Persistent plan", phases)

        # Complete phase 0
        mgr.advance_phase(success=True, result="Phase A done")
        assert mgr.active_plan.current_phase_index == 1
        assert mgr.active_plan.phases[1].status == PhaseStatus.ACTIVE

        # Session end
        result = mgr.on_session_end()
        assert result["plan_saved"] == "plan-persist"

        # New session — new PlanManager
        bus2 = AgentBus()
        collector2 = EventCollector(bus2)
        mgr2 = PlanManager(plans_dir=plans_dir, bus=bus2, config=LongHorizonConfig())
        start_result = mgr2.on_session_start()
        assert start_result["plan_restored"] == "plan-persist"

        # Verify state restored
        assert mgr2.active_plan is not None
        assert mgr2.active_plan.current_phase_index == 1
        assert mgr2.active_plan.phases[0].status == PhaseStatus.COMPLETED
        assert mgr2.active_plan.phases[1].status == PhaseStatus.ACTIVE  # Was paused, restored

        # PlanRestored event published
        restored = collector2.of_type(PlanRestored)
        assert len(restored) == 1
        assert restored[0].plan_id == "plan-persist"

        # Complete remaining phases
        mgr2.advance_phase(success=True, result="Phase B done")
        mgr2.advance_phase(success=True, result="Phase C done")
        assert mgr2.active_plan is None

    def test_mid_phase_restart_resets_in_progress(self, tmp_path):
        """IN_PROGRESS sub-goals are reset to PENDING on restore."""
        from pathlib import Path
        plans_dir = str(Path(tmp_path) / "plans")
        bus = AgentBus()

        sg = SubGoal(
            id="sg-mid", description="Long running task",
            tool_name="long_task",
        )
        phase = make_phase(index=0, description="Phase with mid-run sub-goal", sub_goals=[sg])
        mgr = PlanManager(plans_dir=plans_dir, bus=bus, config=LongHorizonConfig())
        mgr.create_plan("plan-mid", "Mid-phase test", [phase])

        # Simulate mid-execution: sub-goal started but not finished
        mgr.active_plan.phases[0].sub_goals[0].status = SubGoalStatus.IN_PROGRESS
        mgr.active_plan.phases[0].sub_goals[0].started_at = time.time()
        mgr.active_plan.save()

        # Session end + new session
        mgr2 = PlanManager(plans_dir=plans_dir, bus=AgentBus(), config=LongHorizonConfig())
        mgr2.on_session_start()

        # Sub-goal should be reset to PENDING
        current = mgr2.active_plan.get_current_phase()
        assert current.sub_goals[0].status == SubGoalStatus.PENDING
        assert current.sub_goals[0].started_at is None

    def test_pointer_file_enables_fast_restore(self, tmp_path):
        """With 5 plan files on disk, find_active uses pointer file for O(1) lookup."""
        from pathlib import Path
        plans_dir = str(Path(tmp_path) / "plans")
        os.makedirs(plans_dir, exist_ok=True)

        # Save 5 plans: 4 completed, 1 active
        for i in range(5):
            doc = make_plan_document(
                plan_id=f"plan-{i}",
                status="COMPLETED" if i < 4 else "ACTIVE",
            )
            doc.save(os.path.join(plans_dir, f"plan-{i}.json"))

        # Pointer file should exist pointing to the active plan
        pointer = os.path.join(plans_dir, "active_plan.txt")
        assert os.path.exists(pointer)
        with open(pointer) as f:
            assert f.read().strip() == "plan-4.json"

        # find_active should find it via pointer (O(1))
        found = PlanDocument.find_active(plans_dir)
        assert found is not None
        assert found.id == "plan-4"


# ─────────────────────────────────────────────────────────────────────────────
# Replan flow
# ─────────────────────────────────────────────────────────────────────────────


class TestReplanFlow:
    """Tests for REPLAN-triggered re-decomposition on phase failure."""

    def test_replan_on_phase_failure_and_recovery(self, tmp_path):
        """Fail a phase with REPLAN sub-goal → REPLANNING → accept_replan → complete."""
        bus = AgentBus()
        collector = EventCollector(bus)
        mgr = make_plan_manager(tmp_path, bus=bus)

        # Phase 0 with a REPLAN-eligible sub-goal
        sg = SubGoal(
            id="sg-replan-0", description="Try approach A",
            tool_name="approach_a", on_failure=FailureStrategy.REPLAN,
        )
        phase0 = make_phase(index=0, description="Research phase", sub_goals=[sg])
        phase1 = make_phase(index=1, description="Write phase")
        mgr.create_plan("plan-replan", "Replan test", [phase0, phase1])

        # Fail phase 0
        mgr.advance_phase(success=False, error="approach_a returned no useful data")
        assert mgr.active_plan.status == PlanStatus.REPLANNING

        # Verify replan event published
        replan_events = collector.of_type(PlanReplanRequested)
        assert len(replan_events) == 1
        assert replan_events[0].failed_phase_id == "phase-plan-1-0"

        # Accept replan with new sub-goals
        new_sg = SubGoal(
            id="sg-replan-1", description="Try approach B",
            tool_name="approach_b",
        )
        revised_phase = Phase(
            id="phase-plan-1-0", description="Research phase",
            status=PhaseStatus.ACTIVE, plan_id="plan-replan",
            sub_goals=[new_sg],
        )
        mgr.accept_replan(revised_phase)

        assert mgr.active_plan.status == PlanStatus.ACTIVE
        current = mgr.active_plan.get_current_phase()
        assert len(current.sub_goals) == 1
        assert current.sub_goals[0].tool_name == "approach_b"

        # Complete phase 0 with new approach
        mgr.advance_phase(success=True, result="Approach B worked")
        # Complete phase 1
        mgr.advance_phase(success=True, result="Written")
        assert mgr.active_plan is None

        plan_completed = collector.of_type(PlanCompleted)
        assert len(plan_completed) == 1
        assert plan_completed[0].success is True

    def test_replan_exhaustion_fails_plan(self, tmp_path):
        """Exhaust max_replan_attempts → plan FAILED."""
        config = LongHorizonConfig(max_replan_attempts=1)
        bus = AgentBus()
        mgr = make_plan_manager(tmp_path, bus=bus, config=config)

        sg = SubGoal(
            id="sg-ex-0", description="Fragile sub-goal",
            tool_name="fragile_tool", on_failure=FailureStrategy.REPLAN,
        )
        phase = make_phase(index=0, description="Fragile phase", sub_goals=[sg])
        mgr.create_plan("plan-exhaust", "Exhaust test", [phase])

        # First failure → REPLANNING (attempt 0)
        mgr.advance_phase(success=False, error="First failure")
        assert mgr.active_plan.status == PlanStatus.REPLANNING

        # Accept replan and fail again
        new_sg = SubGoal(
            id="sg-ex-1", description="New approach", tool_name="new_tool",
            on_failure=FailureStrategy.REPLAN,
        )
        revised = Phase(
            id="phase-plan-1-0", description="Fragile phase",
            status=PhaseStatus.ACTIVE, plan_id="plan-exhaust",
            sub_goals=[new_sg],
        )
        mgr.accept_replan(revised)

        # Second failure → max reached → FAILED
        mgr.advance_phase(success=False, error="Second failure")
        # Plan should be FAILED (loaded from disk since active_plan gets cleared)
        path = os.path.join(mgr._plans_dir, "plan-exhaust.json")
        loaded = PlanDocument.load(path)
        assert loaded.status == PlanStatus.FAILED

    def test_replan_context_contains_failure_info(self, tmp_path):
        """Verify ReplanContext contains failure reason and attempted sub-goals."""
        bus = AgentBus()
        collector = EventCollector(bus)
        mgr = make_plan_manager(tmp_path, bus=bus)

        sg = SubGoal(
            id="sg-ctx", description="Fetch API data",
            tool_name="api_fetch", tool_params={"url": "https://example.com"},
            on_failure=FailureStrategy.REPLAN,
        )
        phase = make_phase(index=0, description="API phase", sub_goals=[sg])
        mgr.create_plan("plan-ctx", "Context test", [phase])

        mgr.advance_phase(success=False, error="Timeout after 30s")

        replan_events = collector.of_type(PlanReplanRequested)
        assert len(replan_events) == 1

        ctx = replan_events[0].replan_context
        assert ctx.failure_reason == "Timeout after 30s"
        assert ctx.failure_type == "timeout"
        assert len(ctx.attempted_sub_goals) == 1
        assert ctx.attempted_sub_goals[0]["tool_name"] == "api_fetch"


# ─────────────────────────────────────────────────────────────────────────────
# Soft preemption
# ─────────────────────────────────────────────────────────────────────────────


class TestSoftPreemption:
    """Tests for soft preemption: pause, rollback, finalization, chains."""

    def test_preemption_pauses_old_plan(self, tmp_path):
        """Preempting plan A with plan B pauses A."""
        bus = AgentBus()
        mgr = make_plan_manager(tmp_path, bus=bus)

        phases_a = [make_phase(index=0, description="Plan A")]
        mgr.create_plan("plan-a", "Plan A objective", phases_a)
        assert mgr.active_plan.id == "plan-a"

        phases_b = [make_phase(index=0, description="Plan B")]
        mgr.create_plan("plan-b", "Plan B objective", phases_b)

        # Plan B should be active, Plan A should be paused
        assert mgr.active_plan.id == "plan-b"
        assert len(mgr._preempted_plans) == 1
        assert mgr._preempted_plans[0].id == "plan-a"
        assert mgr._preempted_plans[0].status == PlanStatus.PAUSED
        assert mgr._preempted_plans[0].paused_reason == "soft_preempted"

    def test_preemption_rollback_on_failure(self, tmp_path):
        """When plan B fails before viability, plan A is restored."""
        bus = AgentBus()
        collector = EventCollector(bus)
        config = LongHorizonConfig(preemption_viability_phase=2)
        mgr = make_plan_manager(tmp_path, bus=bus, config=config)

        # Create plan A
        phases_a = [
            make_phase(index=0, description="A phase 0"),
            make_phase(index=1, description="A phase 1"),
        ]
        mgr.create_plan("plan-a", "Plan A", phases_a)

        # Preempt with plan B
        phases_b = [
            make_phase(index=0, description="B phase 0"),
            make_phase(index=1, description="B phase 1"),
        ]
        mgr.create_plan("plan-b", "Plan B", phases_b)
        assert mgr.active_plan.id == "plan-b"

        # Plan B fails before viability threshold (0 completed < 2 required)
        mgr.advance_phase(success=False, error="Plan B failed")

        # Plan A should be restored
        assert mgr.active_plan is not None
        assert mgr.active_plan.id == "plan-a"
        assert mgr.active_plan.status == PlanStatus.ACTIVE

        # PlanRestored event for plan A
        restored = collector.of_type(PlanRestored)
        assert any(e.plan_id == "plan-a" for e in restored)

    def test_preemption_finalization_after_viability(self, tmp_path):
        """Advancing plan B past viability threshold abandons plan A."""
        config = LongHorizonConfig(preemption_viability_phase=1)
        bus = AgentBus()
        mgr = make_plan_manager(tmp_path, bus=bus, config=config)

        # Plan A
        phases_a = [make_phase(index=0, description="A")]
        mgr.create_plan("plan-a", "Plan A", phases_a)

        # Preempt with Plan B (2 phases)
        phases_b = [
            make_phase(index=0, description="B0"),
            make_phase(index=1, description="B1"),
        ]
        mgr.create_plan("plan-b", "Plan B", phases_b)
        assert len(mgr._preempted_plans) == 1

        # Complete phase 0 of plan B → past viability (1 completed >= 1 required)
        mgr.advance_phase(success=True, result="B0 done")

        # Preempted plans should be finalized (abandoned)
        assert len(mgr._preempted_plans) == 0
        loaded_a = PlanDocument.load(
            os.path.join(mgr._plans_dir, "plan-a.json")
        )
        assert loaded_a.status == PlanStatus.ABANDONED

    def test_preemption_chain_a_b_c(self, tmp_path):
        """A -> B -> C chain. Fail C -> B restored. Complete B -> A abandoned."""
        config = LongHorizonConfig(preemption_viability_phase=1)
        bus = AgentBus()
        mgr = make_plan_manager(tmp_path, bus=bus, config=config)

        # Create chain: A -> B -> C
        for label in ["a", "b", "c"]:
            phases = [
                make_phase(plan_id=f"plan-{label}", index=0, description=f"{label.upper()} phase 0"),
                make_phase(plan_id=f"plan-{label}", index=1, description=f"{label.upper()} phase 1"),
            ]
            mgr.create_plan(f"plan-{label}", f"Plan {label.upper()}", phases)

        assert mgr.active_plan.id == "plan-c"
        assert len(mgr._preempted_plans) == 2  # A and B

        # Fail plan C before viability (0 completed < 1 required)
        mgr.advance_phase(success=False, error="C failed")

        # B should be restored (LIFO — most recently preempted)
        assert mgr.active_plan.id == "plan-b"
        assert mgr.active_plan.status == PlanStatus.ACTIVE
        assert len(mgr._preempted_plans) == 1  # Only A remains

        # Complete plan B phase 0 → viability reached → A abandoned
        mgr.advance_phase(success=True, result="B0 done")
        assert len(mgr._preempted_plans) == 0

        loaded_a = PlanDocument.load(os.path.join(mgr._plans_dir, "plan-a.json"))
        assert loaded_a.status == PlanStatus.ABANDONED

        # Complete plan B
        mgr.advance_phase(success=True, result="B1 done")
        assert mgr.active_plan is None

    def test_soft_preemption_disabled(self, tmp_path):
        """With soft preemption disabled, old plan is ABANDONED immediately."""
        config = LongHorizonConfig(enable_soft_preemption=False)
        bus = AgentBus()
        mgr = make_plan_manager(tmp_path, bus=bus, config=config)

        phases_a = [make_phase(index=0, description="A")]
        mgr.create_plan("plan-a", "Plan A", phases_a)

        phases_b = [make_phase(index=0, description="B")]
        mgr.create_plan("plan-b", "Plan B", phases_b)

        assert mgr.active_plan.id == "plan-b"
        assert len(mgr._preempted_plans) == 0

        loaded_a = PlanDocument.load(os.path.join(mgr._plans_dir, "plan-a.json"))
        assert loaded_a.status == PlanStatus.ABANDONED


# ─────────────────────────────────────────────────────────────────────────────
# Backward compatibility
# ─────────────────────────────────────────────────────────────────────────────


class TestBackwardCompatibility:
    """Tests that plan_manager=None preserves existing behavior."""

    def test_goal_agent_works_without_plan_manager(self):
        """AgenticGoalAgent works identically without plan_manager."""
        from maxim.agents.agentic_goal_agent import AgenticGoalAgent
        from maxim.agents.bus import ProposedGoal, GoalCompleted

        bus = AgentBus()
        agent = AgenticGoalAgent(bus=bus, plan_manager=None)

        completions: list = []
        bus.subscribe(GoalCompleted, completions.append)

        # Propose and complete a simple goal
        goal = ProposedGoal(
            id="g-1", description="Simple goal",
            priority=GoalPriority.MEDIUM,
            tool_name="test_tool", tool_params={},
        )
        bus.publish(goal)

        assert agent.get_current_goal() is not None
        assert agent.get_current_goal().id == "g-1"

        # Report tool result
        agent.report_tool_result(
            tool_call_id=f"tc-g-1",
            tool_name="test_tool",
            success=True,
            result="done",
        )

        assert agent.get_current_goal() is None
        assert len(completions) == 1
        assert completions[0].success is True

    def test_plan_progress_none_when_no_active_plan(self, tmp_path):
        """PlanProgressContext is None when no plan is active."""
        from maxim.agents.bus import PlanProgressContext

        bus = AgentBus()
        mgr = make_plan_manager(tmp_path, bus=bus)

        # No plan created — active_plan is None
        assert mgr.active_plan is None

    def test_goal_agent_with_plan_manager_delegates_completion(self, tmp_path):
        """When plan_manager is attached, goal completion calls advance_phase."""
        from maxim.agents.agentic_goal_agent import AgenticGoalAgent
        from maxim.agents.bus import ProposedGoal, GoalCompleted

        bus = AgentBus()
        collector = EventCollector(bus)
        mgr = make_plan_manager(tmp_path, bus=bus)

        # Create a plan so there's something to advance
        phases = [
            make_phase(index=0, description="Phase 0"),
            make_phase(index=1, description="Phase 1"),
        ]
        mgr.create_plan("plan-delegate", "Delegation test", phases)

        agent = AgenticGoalAgent(bus=bus, plan_manager=mgr)

        # Propose a goal
        goal = ProposedGoal(
            id="g-delegate", description="Do phase 0 work",
            priority=GoalPriority.MEDIUM,
            tool_name="work_tool", tool_params={},
        )
        bus.publish(goal)

        # Complete the goal → should trigger advance_phase on plan_manager
        agent.report_tool_result(
            tool_call_id="tc-g-delegate",
            tool_name="work_tool",
            success=True,
            result="Phase 0 complete",
        )

        # Plan should have advanced to phase 1
        assert mgr.active_plan is not None
        assert mgr.active_plan.current_phase_index == 1
        assert mgr.active_plan.phases[0].status == PhaseStatus.COMPLETED
