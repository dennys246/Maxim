"""Unit tests for plan_document.py — data types, serialization, persistence."""

from __future__ import annotations

import json
import os
import time

import pytest

from maxim.agents.bus import FailureStrategy, SubGoal, SubGoalStatus
from maxim.planning.plan_document import (
    DepthExtension,
    LongHorizonConfig,
    Phase,
    PhaseEnergyBudget,
    PhaseStatus,
    PlanDocument,
    PlanEnergyBudget,
    PlanStatus,
    ReplanContext,
    ReplanRecord,
    classify_failure_type,
)



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
                tool_name=f"tool_{i}_0", tool_params={"phase": i, "step": 0},
                status=SubGoalStatus.PENDING, on_failure=FailureStrategy.REPLAN,
            )
            sg2 = SubGoal(
                id=f"sg-{plan_id}-{i}-1", description=f"Sub-goal {i}.1",
                tool_name=f"tool_{i}_1", tool_params={"phase": i, "step": 1},
                status=SubGoalStatus.PENDING, depends_on=[sg1.id],
                on_failure=FailureStrategy.RETRY,
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


# ─────────────────────────────────────────────────────────────────────────────
# PlanDocument creation and field validation
# ─────────────────────────────────────────────────────────────────────────────


class TestPlanDocumentCreation:
    def test_create_with_3_phases(self):
        plan = make_plan_document(num_phases=3)
        assert len(plan.phases) == 3
        assert plan.status == PlanStatus.ACTIVE
        assert plan.current_phase_index == 0
        assert plan.objective == "Test objective"

    def test_phase_indexes_match(self):
        plan = make_plan_document(num_phases=4)
        for i, phase in enumerate(plan.phases):
            assert phase.index == i
            assert phase.plan_id == plan.id

    def test_first_phase_is_active(self):
        plan = make_plan_document(num_phases=3)
        assert plan.phases[0].status == PhaseStatus.ACTIVE
        assert plan.phases[1].status == PhaseStatus.PENDING
        assert plan.phases[2].status == PhaseStatus.PENDING


class TestGetCurrentPhase:
    def test_returns_first_phase_when_index_0(self):
        plan = make_plan_document(num_phases=3)
        current = plan.get_current_phase()
        assert current is not None
        assert current.index == 0

    def test_returns_correct_phase_after_advancement(self):
        plan = make_plan_document(num_phases=3)
        plan.current_phase_index = 2
        current = plan.get_current_phase()
        assert current is not None
        assert current.index == 2

    def test_returns_none_if_index_out_of_range(self):
        plan = make_plan_document(num_phases=2)
        plan.current_phase_index = 5
        assert plan.get_current_phase() is None


class TestHasNextPhase:
    def test_true_when_phases_remain(self):
        plan = make_plan_document(num_phases=3)
        assert plan.has_next_phase() is True

    def test_false_at_last_phase(self):
        plan = make_plan_document(num_phases=3)
        plan.current_phase_index = 2
        assert plan.has_next_phase() is False


# ─────────────────────────────────────────────────────────────────────────────
# Phase lifecycle
# ─────────────────────────────────────────────────────────────────────────────


class TestPhaseLifecycle:
    def test_status_transitions(self):
        phase = make_phase()
        assert phase.status == PhaseStatus.PENDING

        phase.status = PhaseStatus.ACTIVE
        assert phase.status == PhaseStatus.ACTIVE

        phase.status = PhaseStatus.COMPLETED
        assert phase.status == PhaseStatus.COMPLETED

    def test_get_next_executable_with_dependencies(self):
        sg1 = SubGoal(id="sg-1", description="first", tool_name="tool_a")
        sg2 = SubGoal(
            id="sg-2", description="second", tool_name="tool_b",
            depends_on=["sg-1"],
        )
        phase = make_phase(sub_goals=[sg1, sg2])

        # sg2 depends on sg1, so sg1 should be next
        nxt = phase.get_next_executable()
        assert nxt is not None
        assert nxt.id == "sg-1"

        # After sg1 completes, sg2 should be next
        sg1.status = SubGoalStatus.COMPLETED
        nxt = phase.get_next_executable()
        assert nxt is not None
        assert nxt.id == "sg-2"

    def test_get_next_executable_retryable(self):
        sg = SubGoal(
            id="sg-1", description="retryable", tool_name="tool_a",
            on_failure=FailureStrategy.RETRY, max_retries=3,
        )
        sg.status = SubGoalStatus.FAILED
        sg.attempts = 1
        phase = make_phase(sub_goals=[sg])

        nxt = phase.get_next_executable()
        assert nxt is not None
        assert nxt.id == "sg-1"

    def test_get_next_executable_none_when_all_done(self):
        sg = SubGoal(id="sg-1", description="done", tool_name="tool_a")
        sg.status = SubGoalStatus.COMPLETED
        phase = make_phase(sub_goals=[sg])

        assert phase.get_next_executable() is None

    def test_is_complete(self):
        sg1 = SubGoal(id="sg-1", description="a", tool_name="t1")
        sg2 = SubGoal(id="sg-2", description="b", tool_name="t2")
        phase = make_phase(sub_goals=[sg1, sg2])

        assert phase.is_complete() is False

        sg1.status = SubGoalStatus.COMPLETED
        assert phase.is_complete() is False

        sg2.status = SubGoalStatus.SKIPPED
        assert phase.is_complete() is True

    def test_all_failed(self):
        sg = SubGoal(
            id="sg-1", description="critical", tool_name="t1",
            on_failure=FailureStrategy.ABORT_PARENT, max_retries=0,
        )
        sg.status = SubGoalStatus.FAILED
        sg.attempts = 1
        phase = make_phase(sub_goals=[sg])

        assert phase.all_failed() is True


# ─────────────────────────────────────────────────────────────────────────────
# Serialization round-trips
# ─────────────────────────────────────────────────────────────────────────────


class TestSerialization:
    def test_phase_roundtrip(self):
        sg = SubGoal(
            id="sg-1", description="search", tool_name="search",
            tool_params={"query": "test"}, on_failure=FailureStrategy.REPLAN,
            depends_on=["sg-0"],
        )
        phase = Phase(
            id="phase-1", description="Research", status=PhaseStatus.ACTIVE,
            plan_id="plan-1", index=0, sub_goals=[sg],
            expected_inputs={"data": "raw"}, expected_outputs={"result": "processed"},
        )

        data = phase.to_dict()
        restored = Phase.from_dict(data)

        assert restored.id == phase.id
        assert restored.description == phase.description
        assert restored.status == phase.status
        assert len(restored.sub_goals) == 1
        assert restored.sub_goals[0].on_failure == FailureStrategy.REPLAN
        assert restored.sub_goals[0].depends_on == ["sg-0"]

    def test_plan_document_roundtrip(self):
        plan = make_plan_document(num_phases=3, with_sub_goals=True)
        plan.tags = ["test", "research"]
        plan.source = "llm"
        plan.confidence = 0.85

        # Serialize via to_dict round-trip (mimic save/load)
        data = {
            "version": "1.0",
            "id": plan.id,
            "objective": plan.objective,
            "created_at": plan.created_at,
            "updated_at": plan.updated_at,
            "status": plan.status.name,
            "phases": [p.to_dict() for p in plan.phases],
            "current_phase_index": plan.current_phase_index,
            "default_depth_bound": plan.default_depth_bound,
            "confidence": plan.confidence,
            "source": plan.source,
            "tags": plan.tags,
            "module_refs": plan.module_refs,
            "paused_reason": plan.paused_reason,
            "abandoned_reason": plan.abandoned_reason,
            "active_depth_extensions": {},
            "total_energy_budget": None,
            "replan_history": [],
        }

        json_str = json.dumps(data, default=str)
        loaded_data = json.loads(json_str)

        restored = PlanDocument(
            id=loaded_data["id"],
            objective=loaded_data["objective"],
            created_at=loaded_data["created_at"],
            updated_at=loaded_data["updated_at"],
            status=PlanStatus[loaded_data["status"]],
            phases=[Phase.from_dict(p) for p in loaded_data["phases"]],
            current_phase_index=loaded_data["current_phase_index"],
        )

        assert restored.id == plan.id
        assert len(restored.phases) == 3
        assert len(restored.phases[0].sub_goals) == 2

    def test_replan_context_roundtrip(self):
        stub = Phase(id="p-1", description="Research", status=PhaseStatus.FAILED, plan_id="plan-1")
        ctx = ReplanContext(
            failed_phase=stub,
            failure_reason="timeout",
            failure_type="timeout",
            attempted_sub_goals=[{"description": "search", "tool_name": "search", "error": "timeout"}],
            attempted_tools=["search"],
            attempt_count=1,
            completed_phases=[{"description": "Setup", "result_summary": "done", "energy_spent": 10}],
            preserved_results={"setup_data": {"key": "value"}},
            remaining_phases=[{"description": "Report"}],
            original_objective="Write report",
            energy_remaining={"llm": 80.0},
            budget_remaining={"llm": 50.0},
            time_elapsed=120.0,
        )

        data = ctx.to_dict()
        restored = ReplanContext.from_dict(data)

        assert restored.failure_reason == "timeout"
        assert restored.failure_type == "timeout"
        assert restored.attempt_count == 1
        assert restored.original_objective == "Write report"
        assert "setup_data" in restored.preserved_results

    def test_replan_record_roundtrip(self):
        stub = Phase(id="p-1", description="Fail", status=PhaseStatus.FAILED, plan_id="plan-1")
        ctx = ReplanContext(failed_phase=stub, failure_reason="err", failure_type="tool_error")
        record = ReplanRecord(
            timestamp=time.time(),
            failed_phase_id="p-1",
            failure_reason="err",
            replan_context=ctx,
            original_sub_goals=[{"id": "sg-1", "tool_name": "t1"}],
            energy_spent_on_failure=15.0,
        )

        data = record.to_dict()
        restored = ReplanRecord.from_dict(data)

        assert restored.failed_phase_id == "p-1"
        assert restored.failure_reason == "err"
        assert restored.energy_spent_on_failure == 15.0
        assert restored.replan_context.failure_type == "tool_error"

    def test_phase_energy_budget_roundtrip(self):
        budget = PhaseEnergyBudget(
            phase_id="p-1",
            allocations={"llm": 200.0, "movement": 50.0},
            warning_threshold=0.8,
        )
        budget.spent["llm"] = 120.0

        data = budget.to_dict()
        restored = PhaseEnergyBudget.from_dict(data)

        assert restored.phase_id == "p-1"
        assert restored.allocations["llm"] == 200.0
        assert restored.spent["llm"] == 120.0
        assert restored.remaining["llm"] == 80.0

    def test_plan_energy_budget_roundtrip(self):
        phase_budget = PhaseEnergyBudget(phase_id="p-1", allocations={"llm": 100.0})
        budget = PlanEnergyBudget(
            plan_id="plan-1",
            total={"llm": 500.0},
            replan_reserve={"llm": 50.0},
        )
        budget.phase_budgets["p-1"] = phase_budget

        data = budget.to_dict()
        restored = PlanEnergyBudget.from_dict(data)

        assert restored.plan_id == "plan-1"
        assert restored.total["llm"] == 500.0
        assert "p-1" in restored.phase_budgets


# ─────────────────────────────────────────────────────────────────────────────
# Persistence (save/load)
# ─────────────────────────────────────────────────────────────────────────────


class TestPersistence:
    def test_save_and_load(self, tmp_path):
        plan = make_plan_document(num_phases=3, with_sub_goals=True)
        save_path = str(tmp_path / "plans" / "test_plan.json")
        plan.save(save_path)

        assert os.path.exists(save_path)

        loaded = PlanDocument.load(save_path)
        assert loaded.id == plan.id
        assert loaded.objective == plan.objective
        assert loaded.status == plan.status
        assert len(loaded.phases) == 3
        assert len(loaded.phases[0].sub_goals) == 2

    def test_save_load_with_energy_budget(self, tmp_path):
        plan = make_plan_document(num_phases=2)
        plan.total_energy_budget = PlanEnergyBudget(
            plan_id=plan.id, total={"llm": 500.0},
        )
        save_path = str(tmp_path / "plans" / "energy_plan.json")
        plan.save(save_path)

        loaded = PlanDocument.load(save_path)
        assert loaded.total_energy_budget is not None
        assert loaded.total_energy_budget.total["llm"] == 500.0

    def test_save_load_with_replan_history(self, tmp_path):
        plan = make_plan_document(num_phases=2)
        stub = Phase(id="p-0", description="fail", status=PhaseStatus.FAILED, plan_id=plan.id)
        ctx = ReplanContext(failed_phase=stub, failure_reason="err", failure_type="tool_error")
        plan.replan_history.append(ReplanRecord(
            timestamp=time.time(), failed_phase_id="p-0",
            failure_reason="err", replan_context=ctx,
        ))

        save_path = str(tmp_path / "plans" / "replan_plan.json")
        plan.save(save_path)

        loaded = PlanDocument.load(save_path)
        assert len(loaded.replan_history) == 1
        assert loaded.replan_history[0].failure_reason == "err"

    def test_save_load_with_depth_extensions(self, tmp_path):
        plan = make_plan_document(num_phases=2)
        plan.active_depth_extensions["phase-0"] = DepthExtension(
            phase_id="phase-0", requested_depth=3, reason="complex task", granted=True,
        )

        save_path = str(tmp_path / "plans" / "depth_plan.json")
        plan.save(save_path)

        loaded = PlanDocument.load(save_path)
        assert "phase-0" in loaded.active_depth_extensions
        assert loaded.active_depth_extensions["phase-0"].granted is True

    def test_load_with_recovery_corrupt_main(self, tmp_path):
        plan = make_plan_document(num_phases=2)
        save_path = str(tmp_path / "plans" / "corrupt.json")
        plan.save(save_path)

        # Create backup
        backup_path = f"{save_path}.backup"
        with open(save_path, "r") as f:
            backup_data = f.read()
        with open(backup_path, "w") as f:
            f.write(backup_data)

        # Corrupt main file
        with open(save_path, "w") as f:
            f.write("{invalid json!!!")

        loaded = PlanDocument.load_with_recovery(save_path)
        assert loaded is not None
        assert loaded.id == plan.id

    def test_load_with_recovery_both_corrupt(self, tmp_path):
        save_path = str(tmp_path / "plans" / "unrecoverable.json")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            f.write("not json")
        with open(f"{save_path}.backup", "w") as f:
            f.write("also not json")

        loaded = PlanDocument.load_with_recovery(save_path)
        assert loaded is None


# ─────────────────────────────────────────────────────────────────────────────
# Pointer file management
# ─────────────────────────────────────────────────────────────────────────────


class TestPointerFile:
    def test_pointer_written_on_active(self, tmp_path):
        plans_dir = str(tmp_path / "plans")
        plan = make_plan_document(num_phases=2, status="ACTIVE")
        plan.save(os.path.join(plans_dir, f"{plan.id}.json"))

        pointer_path = os.path.join(plans_dir, PlanDocument.ACTIVE_POINTER)
        assert os.path.exists(pointer_path)
        with open(pointer_path) as f:
            assert f.read().strip() == f"{plan.id}.json"

    def test_pointer_cleared_on_completed(self, tmp_path):
        plans_dir = str(tmp_path / "plans")
        plan = make_plan_document(num_phases=2, status="ACTIVE")
        save_path = os.path.join(plans_dir, f"{plan.id}.json")
        plan.save(save_path)

        # Now complete the plan
        plan.status = PlanStatus.COMPLETED
        plan.save(save_path)

        pointer_path = os.path.join(plans_dir, PlanDocument.ACTIVE_POINTER)
        assert not os.path.exists(pointer_path)

    def test_find_active_via_pointer(self, tmp_path):
        plans_dir = str(tmp_path / "plans")
        plan = make_plan_document(num_phases=2, status="ACTIVE")
        plan.save(os.path.join(plans_dir, f"{plan.id}.json"))

        found = PlanDocument.find_active(plans_dir)
        assert found is not None
        assert found.id == plan.id

    def test_find_active_directory_scan_fallback(self, tmp_path):
        plans_dir = str(tmp_path / "plans")
        plan = make_plan_document(num_phases=2, status="ACTIVE")
        plan.save(os.path.join(plans_dir, f"{plan.id}.json"))

        # Remove pointer to force scan
        pointer_path = os.path.join(plans_dir, PlanDocument.ACTIVE_POINTER)
        os.remove(pointer_path)

        found = PlanDocument.find_active(plans_dir)
        assert found is not None
        assert found.id == plan.id
        # Should have backfilled the pointer
        assert os.path.exists(pointer_path)

    def test_find_active_returns_none_when_empty(self, tmp_path):
        plans_dir = str(tmp_path / "plans")
        os.makedirs(plans_dir)
        assert PlanDocument.find_active(plans_dir) is None

    def test_find_active_returns_none_when_dir_missing(self, tmp_path):
        assert PlanDocument.find_active(str(tmp_path / "nonexistent")) is None


# ─────────────────────────────────────────────────────────────────────────────
# Energy budget properties
# ─────────────────────────────────────────────────────────────────────────────


class TestPhaseEnergyBudget:
    def test_remaining(self):
        budget = PhaseEnergyBudget(phase_id="p-1", allocations={"llm": 200.0})
        budget.spent["llm"] = 75.0
        assert budget.remaining["llm"] == 125.0

    def test_utilization(self):
        budget = PhaseEnergyBudget(phase_id="p-1", allocations={"llm": 200.0})
        budget.spent["llm"] = 100.0
        assert budget.utilization["llm"] == 0.5

    def test_is_over_warning(self):
        budget = PhaseEnergyBudget(
            phase_id="p-1", allocations={"llm": 100.0}, warning_threshold=0.75,
        )
        budget.spent["llm"] = 74.0
        assert budget.is_over_warning("llm") is False

        budget.spent["llm"] = 76.0
        assert budget.is_over_warning("llm") is True

    def test_is_over_budget(self):
        budget = PhaseEnergyBudget(phase_id="p-1", allocations={"llm": 100.0})
        budget.spent["llm"] = 99.0
        assert budget.is_over_budget("llm") is False

        budget.spent["llm"] = 100.0
        assert budget.is_over_budget("llm") is True


# ─────────────────────────────────────────────────────────────────────────────
# ReplanContext prompt generation
# ─────────────────────────────────────────────────────────────────────────────


class TestReplanContextPrompt:
    def test_to_llm_prompt_section(self):
        stub = Phase(id="p-1", description="Research", status=PhaseStatus.FAILED, plan_id="plan-1")
        ctx = ReplanContext(
            failed_phase=stub,
            failure_reason="API timeout",
            failure_type="timeout",
            attempted_sub_goals=[
                {"description": "search API", "tool_name": "search", "error": "timeout"},
            ],
            completed_phases=[
                {"description": "Setup", "result_summary": "done"},
            ],
            remaining_phases=[{"description": "Report"}],
            original_objective="Write report",
            energy_remaining={"llm": 80.0},
        )

        prompt = ctx.to_llm_prompt_section()
        assert "REPLAN REQUIRED" in prompt
        assert "Research" in prompt
        assert "DO NOT REPEAT" in prompt
        assert "Setup" in prompt
        assert "Report" in prompt
        assert "Write report" in prompt


# ─────────────────────────────────────────────────────────────────────────────
# classify_failure_type
# ─────────────────────────────────────────────────────────────────────────────


class TestClassifyFailureType:
    def test_timeout(self):
        assert classify_failure_type("Connection timeout after 30s") == "timeout"

    def test_budget(self):
        assert classify_failure_type("Energy budget exceeded") == "budget_exceeded"

    def test_no_results(self):
        assert classify_failure_type("No result found") == "no_results"
        assert classify_failure_type("empty response") == "no_results"
        assert classify_failure_type("File not found") == "no_results"

    def test_tool_error_default(self):
        assert classify_failure_type("Something unexpected happened") == "tool_error"

    def test_empty_error(self):
        assert classify_failure_type("") == "tool_error"
