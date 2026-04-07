"""PlanManager: Central coordinator for long-horizon plan lifecycle.

Single owner of the active plan under RLock. All plan mutations go
through PlanManager — no other component holds a mutable reference.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from threading import RLock
from typing import Any

from maxim.agents.bus import (
    AgentBus,
    FailureStrategy,
    PhaseCompleted,
    PhaseStarted,
    PlanCompleted,
    PlanCreated,
    PlanReplanRequested,
    PlanRestored,
    SubGoalStatus,
)
from maxim.planning.plan_document import (
    CodingReplanContext,
    LongHorizonConfig,
    Phase,
    PhaseStatus,
    PlanDocument,
    PlanStatus,
    ReplanContext,
    ReplanRecord,
    classify_failure_type,
    _sg_to_dict,
)

logger = logging.getLogger(__name__)

_CODING_TOOL_NAMES = {"run_tests", "lint_code", "type_check", "run_build"}


def _extract_coding_context(phase: Phase) -> CodingReplanContext | None:
    """Extract structured coding errors from phase sub-goal results."""
    context = CodingReplanContext()
    has_coding_data = False

    for sg in phase.sub_goals:
        if sg.result is None:
            continue

        result = sg.result if isinstance(sg.result, dict) else {}
        tool_name = result.get("tool_name", "") or sg.tool_name or ""
        metadata = result.get("metadata", {}) if isinstance(sg.result, dict) else {}

        if sg.status == SubGoalStatus.FAILED and tool_name in _CODING_TOOL_NAMES:
            entry = {
                "command": metadata.get("command", ""),
                "returncode": metadata.get("returncode"),
                "error": (result.get("error", "") if isinstance(sg.result, dict) else str(sg.result)),
            }
            if tool_name == "run_tests":
                context.test_failures.append(entry)
            elif tool_name == "run_build":
                context.build_errors.append(entry)
            elif tool_name in ("lint_code", "type_check"):
                context.lint_errors.append(entry)
            has_coding_data = True

        elif tool_name in ("edit_file", "write_file"):
            path = result.get("path", "") if isinstance(sg.result, dict) else ""
            if path:
                context.files_modified.append(path)

    return context if has_coding_data else None


@dataclass
class PlanServices:
    """Dependency bundle for PlanManager.

    Groups external dependencies so PlanManager's constructor stays clean
    and tests can inject mocks easily. All fields are optional — PlanManager
    gracefully degrades when services are absent.
    """

    registry: Any = None  # EnergyRegistry
    nac: Any = None  # NAc
    hippocampus: Any = None  # Hippocampus
    plan_history: Any = None  # PlanHistoryBridge
    statistician: Any = None  # StatisticianAgent


class PlanManager:
    """Central coordinator for long-horizon plan lifecycle.

    Owns the active plan under RLock. Handles phase transitions,
    failure cascades, replanning, session persistence, and soft preemption.
    """

    def __init__(
        self,
        plans_dir: str,
        bus: AgentBus,
        config: LongHorizonConfig | None = None,
        services: PlanServices | None = None,
    ) -> None:
        self._plans_dir = plans_dir
        self._bus = bus
        self._config = config or LongHorizonConfig()
        self._services = services or PlanServices()
        self._lock = RLock()
        self._active_plan: PlanDocument | None = None
        self._preempted_plans: list[PlanDocument] = []

    @property
    def active_plan(self) -> PlanDocument | None:
        """Read-only access to the active plan."""
        return self._active_plan

    # ── Plan Creation ────────────────────────────────────────────────────

    def create_plan(
        self,
        plan_id: str,
        objective: str,
        phases: list[Phase],
        source: str = "llm",
        confidence: float = 1.0,
    ) -> PlanDocument:
        """Create and activate a new plan."""
        with self._lock:
            now = time.time()
            plan = PlanDocument(
                id=plan_id,
                objective=objective,
                created_at=now,
                updated_at=now,
                status=PlanStatus.ACTIVE,
                phases=phases,
                current_phase_index=0,
                source=source,
                confidence=confidence,
            )

            # Set persistence path
            os.makedirs(self._plans_dir, exist_ok=True)
            plan._persistence_path = os.path.join(self._plans_dir, f"{plan_id}.json")

            # Activate the first phase
            if phases:
                phases[0].status = PhaseStatus.ACTIVE
                phases[0].started_at = now

            # If there's an active plan, preempt it
            if self._active_plan:
                self.preempt_for_new_plan(plan)
            else:
                self._active_plan = plan
                plan.save()

                self._bus.publish(
                    PlanCreated(
                        plan_id=plan.id,
                        objective=plan.objective,
                        phase_count=len(plan.phases),
                        timestamp=now,
                    )
                )

                if phases:
                    self._bus.publish(
                        PhaseStarted(
                            plan_id=plan.id,
                            phase_id=phases[0].id,
                            phase_index=0,
                            description=phases[0].description,
                            timestamp=now,
                        )
                    )

            return plan

    # ── Phase Advancement ────────────────────────────────────────────────

    def advance_phase(
        self,
        success: bool,
        result: Any = None,
        error: str | None = None,
    ) -> None:
        """Advance the active plan's current phase. Thread-safe."""
        with self._lock:
            plan = self._active_plan
            if not plan:
                return

            current = plan.get_current_phase()
            if not current:
                return

            if success:
                current.status = PhaseStatus.COMPLETED
                current.result_summary = str(result)[:500] if result else None
                current.completed_at = time.time()

                # Extract structured outputs
                if isinstance(result, dict):
                    current.outputs = {k: v for k, v in result.items() if k in current.expected_outputs}

                self._bus.publish(
                    PhaseCompleted(
                        plan_id=plan.id,
                        phase_id=current.id,
                        success=True,
                        result_summary=current.result_summary,
                        energy_spent=current.energy_spent,
                        timestamp=time.time(),
                    )
                )

                # Finalize soft-preempted plans once we've proven viable
                viability = self._config.preemption_viability_phase
                completed_count = current.index + 1
                if completed_count >= viability and self._preempted_plans:
                    self._finalize_preempted_plans()

                if plan.has_next_phase():
                    plan.current_phase_index += 1
                    next_phase = plan.get_current_phase()
                    if next_phase:
                        next_phase.status = PhaseStatus.ACTIVE
                        next_phase.started_at = time.time()

                        self._allocate_phase_budget(next_phase)

                        self._bus.publish(
                            PhaseStarted(
                                plan_id=plan.id,
                                phase_id=next_phase.id,
                                phase_index=next_phase.index,
                                description=next_phase.description,
                                timestamp=time.time(),
                            )
                        )
                else:
                    plan.status = PlanStatus.COMPLETED
                    total_energy = sum(p.energy_spent for p in plan.phases)
                    self._bus.publish(
                        PlanCompleted(
                            plan_id=plan.id,
                            success=True,
                            phases_completed=len([p for p in plan.phases if p.status == PhaseStatus.COMPLETED]),
                            phases_total=len(plan.phases),
                            total_energy_spent=total_energy,
                            timestamp=time.time(),
                        )
                    )
                    plan.save()
                    self._clear_active_plan()
                    return

                plan.save()
            else:
                self._handle_phase_failure(current, error or "Unknown error")

    # ── Phase Failure Handling ───────────────────────────────────────────

    def _handle_phase_failure(self, phase: Phase, error: str) -> None:
        """Unified phase failure handler: REPLAN -> escalate -> fail -> preemption rollback."""
        plan = self._active_plan
        if not plan:
            return

        # Stage 1: REPLAN (if eligible)
        replan_eligible = phase.sub_goals and any(sg.on_failure == FailureStrategy.REPLAN for sg in phase.sub_goals)
        attempt_count = len([r for r in plan.replan_history if r.failed_phase_id == phase.id])

        if replan_eligible and attempt_count < self._config.max_replan_attempts:
            replan_ctx = self._build_replan_context(plan, phase, error, attempt_count)

            plan.replan_history.append(
                ReplanRecord(
                    timestamp=time.time(),
                    failed_phase_id=phase.id,
                    failure_reason=error,
                    replan_context=replan_ctx,
                    original_sub_goals=[_sg_to_dict(sg) for sg in phase.sub_goals],
                    revised_sub_goals=[],
                    revision_succeeded=None,
                    energy_spent_on_failure=phase.energy_spent,
                )
            )

            plan.status = PlanStatus.REPLANNING
            self._bus.publish(
                PlanReplanRequested(
                    plan_id=plan.id,
                    failed_phase_id=phase.id,
                    reason=error,
                    replan_context=replan_ctx,
                    timestamp=time.time(),
                )
            )
            plan.save()
            return

        # Stage 2: ESCALATE or FAIL
        if attempt_count >= self._config.max_replan_attempts:
            logger.warning(
                "Plan '%s' exhausted %d replan attempts on phase '%s'",
                plan.objective,
                attempt_count,
                phase.description,
            )

        phase.status = PhaseStatus.FAILED
        phase.error = error
        plan.status = PlanStatus.FAILED
        plan.save()

        self._bus.publish(
            PhaseCompleted(
                plan_id=plan.id,
                phase_id=phase.id,
                success=False,
                error=error,
                energy_spent=phase.energy_spent,
                timestamp=time.time(),
            )
        )

        # Stage 3: PREEMPTION ROLLBACK
        viability = self._config.preemption_viability_phase
        completed_count = sum(1 for p in plan.phases if p.status == PhaseStatus.COMPLETED)

        if completed_count < viability and self._preempted_plans:
            resumed_plan = self._preempted_plans.pop()
            resumed_plan.status = PlanStatus.ACTIVE
            resumed_plan.paused_reason = None
            resumed_plan.save()
            self._active_plan = resumed_plan
            self._bus.publish(
                PlanRestored(
                    plan_id=resumed_plan.id,
                    objective=resumed_plan.objective,
                    current_phase_index=resumed_plan.current_phase_index,
                    status=resumed_plan.status.name,
                    timestamp=time.time(),
                )
            )
            logger.info(
                "Rolled back to preempted plan '%s' after plan '%s' failed before viability threshold (%d/%d phases)",
                resumed_plan.objective,
                plan.objective,
                completed_count,
                viability,
            )
        else:
            self._clear_active_plan()

    def accept_replan(self, revised_phase: Phase) -> None:
        """Accept LLM's revised decomposition and return plan to ACTIVE."""
        with self._lock:
            plan = self._active_plan
            if not plan or plan.status != PlanStatus.REPLANNING:
                return

            current = plan.get_current_phase()
            if not current:
                return

            # Update the latest ReplanRecord
            if plan.replan_history:
                latest = plan.replan_history[-1]
                latest.revised_sub_goals = [_sg_to_dict(sg) for sg in revised_phase.sub_goals]

            # Replace the failed phase's sub-goals
            current.sub_goals = revised_phase.sub_goals
            current.status = PhaseStatus.ACTIVE
            current.error = None

            plan.status = PlanStatus.ACTIVE
            plan.save()

            self._bus.publish(
                PhaseStarted(
                    plan_id=plan.id,
                    phase_id=current.id,
                    phase_index=current.index,
                    description=f"[REPLAN] {current.description}",
                    timestamp=time.time(),
                )
            )

    # ── Energy Budget Allocation ─────────────────────────────────────────

    def _allocate_phase_budget(self, phase: Phase) -> None:
        """Allocate energy for a single phase on-demand."""
        plan = self._active_plan
        registry = self._services.registry
        if not plan or not plan.total_energy_budget or not registry:
            return

        from maxim.planning.plan_document import PhaseEnergyBudget

        available: dict[str, float] = {}
        for domain in ["llm", "movement", "vision"]:
            budget = registry.get_budget(domain)
            if budget:
                available[domain] = budget.current_level

        allocatable: dict[str, float] = {}
        for d, v in available.items():
            replan_hold = v * plan.total_energy_budget.replan_reserve_fraction
            buffer_hold = v * plan.total_energy_budget.buffer_fraction
            already_reserved = plan.total_energy_budget.replan_reserve.get(d, 0)
            new_reserve = max(0, replan_hold - already_reserved)
            plan.total_energy_budget.replan_reserve[d] = already_reserved + new_reserve
            allocatable[d] = v - new_reserve - buffer_hold

        remaining_phases = [
            p for p in plan.phases if p.status in (PhaseStatus.PENDING, PhaseStatus.BLOCKED, PhaseStatus.ACTIVE)
        ]
        n_remaining = max(1, len(remaining_phases))

        nac = self._services.nac
        weight = 1.0
        if nac:
            prediction = nac.predict("phase", phase.description)
            if prediction:
                weight = prediction.predicted_value

        phase_fraction = weight / n_remaining
        phase_alloc = {d: max(v * phase_fraction, 10.0) for d, v in allocatable.items()}

        phase.energy_budget = PhaseEnergyBudget(
            phase_id=phase.id,
            allocations=phase_alloc,
        )

        if plan.total_energy_budget:
            plan.total_energy_budget.phase_budgets[phase.id] = phase.energy_budget

    # ── Replan Context Building ──────────────────────────────────────────

    def _build_replan_context(self, plan: PlanDocument, phase: Phase, error: str, attempt_count: int) -> ReplanContext:
        """Build rich context for replan using available services."""
        svc = self._services

        completed_phases = [
            {
                "description": p.description,
                "result_summary": p.result_summary or "completed",
                "energy_spent": p.energy_spent,
            }
            for p in plan.phases[: plan.current_phase_index]
            if p.status == PhaseStatus.COMPLETED
        ]

        remaining_phases = [
            {
                "description": p.description,
                "estimated_energy": (sum(p.energy_budget.allocations.values()) if p.energy_budget else None),
            }
            for p in plan.phases[plan.current_phase_index + 1 :]
        ]

        energy_remaining: dict[str, float] = {}
        if svc.registry:
            for domain in ["llm", "movement", "vision"]:
                budget = svc.registry.get_budget(domain)
                if budget:
                    energy_remaining[domain] = budget.current_level

        similar_past_failures: list[dict] = []
        if svc.hippocampus:
            try:
                similar = svc.hippocampus.recall_similar(f"plan failure: {phase.description} — {error}", limit=3)
                similar_past_failures = [
                    {
                        "objective": getattr(m, "content", str(m))[:100],
                        "failed_phase": phase.description,
                        "resolution": (
                            m.metadata.get("resolution", "unknown")
                            if hasattr(m, "metadata") and m.metadata
                            else "unknown"
                        ),
                    }
                    for m in similar
                ]
            except Exception:
                pass

        alternative_approaches: list[str] = []
        if svc.plan_history:
            try:
                modules = svc.plan_history.get_matching_modules(plan.objective, context=None)
                alternative_approaches = [f"{m.name} ({m.success_rate:.0%} success)" for m, _ in modules[:3]]
            except Exception:
                pass

        tool_success_rates: dict[str, float] = {}
        if svc.statistician:
            try:
                for sg in phase.sub_goals:
                    if sg.tool_name:
                        rate = svc.statistician.get_tool_success_rate(sg.tool_name)
                        if rate is not None:
                            tool_success_rates[sg.tool_name] = rate
            except Exception:
                pass

        # Extract coding-specific context if available
        coding_context = _extract_coding_context(phase)

        return ReplanContext(
            failed_phase=phase,
            failure_reason=error,
            failure_type=classify_failure_type(error),
            attempted_sub_goals=[
                {
                    "description": sg.description,
                    "tool_name": sg.tool_name,
                    "params": sg.tool_params,
                    "result": (str(sg.result)[:200] if sg.result else None),
                    "error": sg.error,
                }
                for sg in phase.sub_goals
            ],
            attempted_tools=list({sg.tool_name for sg in phase.sub_goals if sg.tool_name}),
            attempt_count=attempt_count,
            completed_phases=completed_phases,
            preserved_results={
                k: v
                for p in plan.phases[: plan.current_phase_index]
                if p.status == PhaseStatus.COMPLETED
                for k, v in p.outputs.items()
            },
            remaining_phases=remaining_phases,
            original_objective=plan.objective,
            energy_remaining=energy_remaining,
            budget_remaining=(phase.energy_budget.remaining if phase.energy_budget else {}),
            time_elapsed=time.time() - plan.created_at,
            similar_past_failures=similar_past_failures,
            alternative_approaches=alternative_approaches,
            tool_success_rates=tool_success_rates,
            pattern_warnings=[],
            coding_context=coding_context,
        )

    # ── Session Lifecycle ────────────────────────────────────────────────

    def on_session_start(self) -> dict[str, Any]:
        """Restore active plan and preemption chain on session start."""
        self._active_plan = PlanDocument.find_active(self._plans_dir)

        if self._active_plan:
            logger.info(
                "Resumed plan '%s' — phase %d/%d (%s)",
                self._active_plan.objective,
                self._active_plan.current_phase_index + 1,
                len(self._active_plan.phases),
                self._active_plan.status.name,
            )

            self._bus.publish(
                PlanRestored(
                    plan_id=self._active_plan.id,
                    objective=self._active_plan.objective,
                    current_phase_index=self._active_plan.current_phase_index,
                    status=self._active_plan.status.name,
                    timestamp=time.time(),
                )
            )

            # Reset mid-execution sub-goals
            current = self._active_plan.get_current_phase()
            if current:
                for sg in current.sub_goals:
                    if sg.status == SubGoalStatus.IN_PROGRESS:
                        sg.status = SubGoalStatus.PENDING
                        sg.started_at = None

        # Check if restored plan is actually a stale preempted plan
        if (
            self._active_plan
            and self._active_plan.status == PlanStatus.PAUSED
            and self._active_plan.paused_reason == "soft_preempted"
        ):
            stale_cutoff = time.time() - (self._config.stale_preemption_hours * 3600)
            if self._active_plan.updated_at < stale_cutoff:
                self._active_plan.status = PlanStatus.ABANDONED
                self._active_plan.abandoned_reason = "Stale — paused for >24 hours"
                self._active_plan.save()
                self._clear_active_plan()

        self._rebuild_preemption_stack()

        # Auto-abandon stale preempted plans in the stack
        now = time.time()
        stale_cutoff = now - (self._config.stale_preemption_hours * 3600)
        for plan in list(self._preempted_plans):
            if plan.updated_at < stale_cutoff:
                plan.status = PlanStatus.ABANDONED
                plan.abandoned_reason = "Stale — paused for >24 hours"
                plan.save()
                self._preempted_plans.remove(plan)

        return {"plan_restored": self._active_plan.id if self._active_plan else None}

    def on_session_end(self) -> dict[str, Any]:
        """Persist active plan on session end."""
        if self._active_plan:
            if self._active_plan.status == PlanStatus.ACTIVE:
                self._active_plan.status = PlanStatus.PAUSED
                self._active_plan.paused_reason = "session_end"

            self._active_plan.save()
            logger.info(
                "Saved plan '%s' (status: %s)",
                self._active_plan.objective,
                self._active_plan.status.name,
            )
            return {"plan_saved": self._active_plan.id}
        return {"plan_saved": None}

    # ── Soft Preemption ──────────────────────────────────────────────────

    def preempt_for_new_plan(self, new_plan: PlanDocument) -> None:
        """Soft-preempt the active plan in favor of a new one."""
        with self._lock:
            if not self._config.enable_soft_preemption:
                if self._active_plan:
                    self._active_plan.status = PlanStatus.ABANDONED
                    self._active_plan.abandoned_reason = "Preempted (soft preemption disabled)"
                    self._active_plan.save()
            elif self._active_plan:
                self._active_plan.status = PlanStatus.PAUSED
                self._active_plan.paused_reason = "soft_preempted"
                self._active_plan.save()
                self._preempted_plans.append(self._active_plan)

            self._active_plan = new_plan
            new_plan.status = PlanStatus.ACTIVE
            new_plan.save()

            self._bus.publish(
                PlanCreated(
                    plan_id=new_plan.id,
                    objective=new_plan.objective,
                    phase_count=len(new_plan.phases),
                    timestamp=time.time(),
                )
            )

    def _finalize_preempted_plans(self) -> None:
        """Called when active plan advances past viability threshold."""
        for plan in self._preempted_plans:
            plan.status = PlanStatus.ABANDONED
            plan.abandoned_reason = (
                (f"Superseded by plan {self._active_plan.id} which passed viability threshold")
                if self._active_plan
                else "Superseded"
            )
            plan.save()
        self._preempted_plans.clear()

    def _rebuild_preemption_stack(self) -> None:
        """Scan plans directory for soft-preempted PAUSED plans."""
        if not os.path.isdir(self._plans_dir):
            return
        for filename in sorted(os.listdir(self._plans_dir)):
            if not filename.endswith(".json") or filename.endswith(".tmp"):
                continue
            path = os.path.join(self._plans_dir, filename)
            try:
                plan = PlanDocument.load(path)
                if (
                    plan.status == PlanStatus.PAUSED
                    and plan.paused_reason == "soft_preempted"
                    and plan.id != getattr(self._active_plan, "id", None)
                ):
                    self._preempted_plans.append(plan)
            except Exception:
                continue
        # Sort by updated_at so most recently preempted is last (LIFO)
        self._preempted_plans.sort(key=lambda p: p.updated_at)

    # ── Internal Helpers ─────────────────────────────────────────────────

    def _clear_active_plan(self) -> None:
        """Clear the active plan reference and pointer file."""
        if self._active_plan:
            PlanDocument._clear_pointer(self._plans_dir)
            self._active_plan = None
