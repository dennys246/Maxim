"""Persistent plan document and supporting data types for long-horizon planning.

Pure data module — no bus, no agents. All types are serializable and
designed for atomic persistence following the Hippocampus pattern
(versioned JSON, tmp + os.replace, load_with_recovery).
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from enum import Enum, auto
from typing import Any

from maxim.agents.bus import FailureStrategy, GoalPriority, SubGoal, SubGoalStatus

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────────────────────


class PlanStatus(Enum):
    """Lifecycle status of a plan."""

    DRAFT = auto()
    ACTIVE = auto()
    PAUSED = auto()
    REPLANNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    ABANDONED = auto()


class PhaseStatus(Enum):
    """Lifecycle status of a phase within a plan."""

    PENDING = auto()
    BLOCKED = auto()
    ACTIVE = auto()
    COMPLETED = auto()
    FAILED = auto()
    SKIPPED = auto()


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class LongHorizonConfig:
    """Configuration for the long-horizon planning system."""

    # Core
    enable_plans: bool = True
    default_depth_bound: int = 2
    max_depth_bound: int = 4
    max_replan_attempts: int = 2
    plan_persistence_path: str = "planning"

    # Energy budgets
    phase_budget_hard_cap: bool = False

    # Modules
    enable_modules: bool = True
    module_promotion_threshold: int = 5
    module_promotion_nac_confidence: float = 0.7

    # Soft preemption
    enable_soft_preemption: bool = True
    preemption_viability_phase: int = 1
    stale_preemption_hours: float = 24.0


# ─────────────────────────────────────────────────────────────────────────────
# Energy Budgets
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class PhaseEnergyBudget:
    """Energy allocation for a single phase, drawn from EnergyRegistry domain budgets."""

    phase_id: str
    allocations: dict[str, float] = field(default_factory=dict)
    spent: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    warning_threshold: float = 0.75
    hard_cap: bool = False
    downstream_reserve: dict[str, float] = field(default_factory=dict)

    @property
    def remaining(self) -> dict[str, float]:
        return {
            domain: self.allocations.get(domain, 0) - self.spent.get(domain, 0)
            for domain in self.allocations
        }

    @property
    def utilization(self) -> dict[str, float]:
        return {
            domain: self.spent.get(domain, 0) / alloc if alloc > 0 else 0
            for domain, alloc in self.allocations.items()
        }

    def is_over_warning(self, domain: str) -> bool:
        alloc = self.allocations.get(domain, 0)
        return alloc > 0 and self.spent.get(domain, 0) / alloc >= self.warning_threshold

    def is_over_budget(self, domain: str) -> bool:
        alloc = self.allocations.get(domain, 0)
        return alloc > 0 and self.spent.get(domain, 0) >= alloc

    def to_dict(self) -> dict:
        return {
            "phase_id": self.phase_id,
            "allocations": self.allocations,
            "spent": dict(self.spent),
            "warning_threshold": self.warning_threshold,
            "hard_cap": self.hard_cap,
            "downstream_reserve": self.downstream_reserve,
        }

    @classmethod
    def from_dict(cls, data: dict) -> PhaseEnergyBudget:
        budget = cls(
            phase_id=data["phase_id"],
            allocations=data.get("allocations", {}),
            warning_threshold=data.get("warning_threshold", 0.75),
            hard_cap=data.get("hard_cap", False),
            downstream_reserve=data.get("downstream_reserve", {}),
        )
        budget.spent = defaultdict(float, data.get("spent", {}))
        return budget


@dataclass
class PlanEnergyBudget:
    """Top-level energy budget for an entire plan, allocated lazily."""

    plan_id: str
    total: dict[str, float] = field(default_factory=dict)
    phase_budgets: dict[str, PhaseEnergyBudget] = field(default_factory=dict)
    replan_reserve: dict[str, float] = field(default_factory=dict)
    replan_reserve_fraction: float = 0.10
    buffer_fraction: float = 0.05

    def to_dict(self) -> dict:
        return {
            "plan_id": self.plan_id,
            "total": self.total,
            "phase_budgets": {
                pid: pb.to_dict() for pid, pb in self.phase_budgets.items()
            },
            "replan_reserve": self.replan_reserve,
            "replan_reserve_fraction": self.replan_reserve_fraction,
            "buffer_fraction": self.buffer_fraction,
        }

    @classmethod
    def from_dict(cls, data: dict) -> PlanEnergyBudget:
        budget = cls(
            plan_id=data["plan_id"],
            total=data.get("total", {}),
            replan_reserve=data.get("replan_reserve", {}),
            replan_reserve_fraction=data.get("replan_reserve_fraction", 0.10),
            buffer_fraction=data.get("buffer_fraction", 0.05),
        )
        for pid, pb_data in data.get("phase_budgets", {}).items():
            budget.phase_budgets[pid] = PhaseEnergyBudget.from_dict(pb_data)
        return budget


# ─────────────────────────────────────────────────────────────────────────────
# Depth Extension
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class DepthExtension:
    """Temporary permission for deeper decomposition on a specific phase."""

    phase_id: str
    requested_depth: int
    reason: str
    granted: bool = False
    max_allowed: int = 4
    expires_at: float | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Replan Types
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ReplanContext:
    """Everything the LLM needs to re-decompose after a phase failure."""

    # What failed
    failed_phase: Phase
    failure_reason: str
    failure_type: str

    # What was tried
    attempted_sub_goals: list[dict] = field(default_factory=list)
    attempted_tools: list[str] = field(default_factory=list)
    attempt_count: int = 0

    # What succeeded
    completed_phases: list[dict] = field(default_factory=list)
    preserved_results: dict[str, Any] = field(default_factory=dict)

    # What remains
    remaining_phases: list[dict] = field(default_factory=list)
    original_objective: str = ""

    # Resource state
    energy_remaining: dict[str, float] = field(default_factory=dict)
    budget_remaining: dict[str, float] = field(default_factory=dict)
    time_elapsed: float = 0.0

    # Relevant memories
    similar_past_failures: list[dict] = field(default_factory=list)
    alternative_approaches: list[str] = field(default_factory=list)

    # Statistical context
    tool_success_rates: dict[str, float] = field(default_factory=dict)
    pattern_warnings: list[str] = field(default_factory=list)

    def to_llm_prompt_section(self) -> str:
        """Format as a structured prompt section for the LLM."""
        sections: list[str] = []

        sections.append("## REPLAN REQUIRED\n")
        sections.append(f"Original objective: {self.original_objective}\n")

        sections.append(f"### Failed Phase: {self.failed_phase.description}")
        sections.append(f"Failure reason: {self.failure_reason}")
        sections.append(f"Failure type: {self.failure_type}")
        sections.append(f"Attempts on this phase: {self.attempt_count}")

        if self.attempted_sub_goals:
            sections.append("\n### What Was Tried (DO NOT REPEAT):")
            for sg in self.attempted_sub_goals:
                status = "FAILED" if sg.get("error") else "partial"
                sections.append(
                    f"  - {sg['description']} [{sg['tool_name']}] → {status}: "
                    f"{sg.get('error') or sg.get('result', 'no result')}"
                )

        if self.completed_phases:
            sections.append("\n### Completed Phases (PRESERVE):")
            for cp in self.completed_phases:
                sections.append(
                    f"  - {cp['description']} → {cp['result_summary']}"
                )

        if self.remaining_phases:
            sections.append("\n### Remaining Phases:")
            for rp in self.remaining_phases:
                sections.append(f"  - {rp['description']}")

        sections.append("\n### Available Resources:")
        for domain, level in self.energy_remaining.items():
            sections.append(f"  - {domain}: {level:.0f}%")

        if self.alternative_approaches:
            sections.append("\n### Alternative Approaches Available:")
            for alt in self.alternative_approaches:
                sections.append(f"  - {alt}")

        if self.similar_past_failures:
            sections.append("\n### Similar Past Failures (learn from these):")
            for pf in self.similar_past_failures[:3]:
                sections.append(
                    f"  - {pf['objective']}: failed at {pf['failed_phase']} "
                    f"→ resolution: {pf.get('resolution', 'none')}"
                )

        if self.pattern_warnings:
            sections.append("\n### Statistical Warnings:")
            for pw in self.pattern_warnings:
                sections.append(f"  - {pw}")

        sections.append(
            "\n### Instructions:\n"
            "Re-decompose the FAILED phase using a DIFFERENT approach.\n"
            "You may also restructure remaining phases if the failure reveals\n"
            "the original plan structure was flawed.\n"
            "Preserve all completed phase results.\n"
            "Stay within energy budget constraints."
        )

        return "\n".join(sections)

    def to_dict(self) -> dict:
        return {
            "failed_phase_id": self.failed_phase.id,
            "failed_phase_description": self.failed_phase.description,
            "failure_reason": self.failure_reason,
            "failure_type": self.failure_type,
            "attempted_sub_goals": self.attempted_sub_goals,
            "attempted_tools": self.attempted_tools,
            "attempt_count": self.attempt_count,
            "completed_phases": self.completed_phases,
            "preserved_results": {
                k: json.dumps(v, default=str) for k, v in self.preserved_results.items()
            },
            "remaining_phases": self.remaining_phases,
            "original_objective": self.original_objective,
            "energy_remaining": self.energy_remaining,
            "budget_remaining": self.budget_remaining,
            "time_elapsed": self.time_elapsed,
            "similar_past_failures": self.similar_past_failures,
            "alternative_approaches": self.alternative_approaches,
            "tool_success_rates": self.tool_success_rates,
            "pattern_warnings": self.pattern_warnings,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ReplanContext:
        stub_phase = Phase(
            id=data.get("failed_phase_id", "unknown"),
            description=data.get("failed_phase_description", ""),
            status=PhaseStatus.FAILED,
            plan_id="",
        )
        return cls(
            failed_phase=stub_phase,
            failure_reason=data["failure_reason"],
            failure_type=data.get("failure_type", "unknown"),
            attempted_sub_goals=data.get("attempted_sub_goals", []),
            attempted_tools=data.get("attempted_tools", []),
            attempt_count=data.get("attempt_count", 0),
            completed_phases=data.get("completed_phases", []),
            preserved_results={
                k: json.loads(v) if isinstance(v, str) else v
                for k, v in data.get("preserved_results", {}).items()
            },
            remaining_phases=data.get("remaining_phases", []),
            original_objective=data.get("original_objective", ""),
            energy_remaining=data.get("energy_remaining", {}),
            budget_remaining=data.get("budget_remaining", {}),
            time_elapsed=data.get("time_elapsed", 0.0),
            similar_past_failures=data.get("similar_past_failures", []),
            alternative_approaches=data.get("alternative_approaches", []),
            tool_success_rates=data.get("tool_success_rates", {}),
            pattern_warnings=data.get("pattern_warnings", []),
        )


@dataclass
class ReplanRecord:
    """Stored in plan.replan_history for learning and context."""

    timestamp: float
    failed_phase_id: str
    failure_reason: str
    replan_context: ReplanContext
    original_sub_goals: list[dict] = field(default_factory=list)
    revised_sub_goals: list[dict] = field(default_factory=list)
    revision_succeeded: bool | None = None
    energy_spent_on_failure: float = 0.0

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "failed_phase_id": self.failed_phase_id,
            "failure_reason": self.failure_reason,
            "replan_context": self.replan_context.to_dict(),
            "original_sub_goals": self.original_sub_goals,
            "revised_sub_goals": self.revised_sub_goals,
            "revision_succeeded": self.revision_succeeded,
            "energy_spent_on_failure": self.energy_spent_on_failure,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ReplanRecord:
        return cls(
            timestamp=data["timestamp"],
            failed_phase_id=data["failed_phase_id"],
            failure_reason=data["failure_reason"],
            replan_context=ReplanContext.from_dict(data["replan_context"]),
            original_sub_goals=data.get("original_sub_goals", []),
            revised_sub_goals=data.get("revised_sub_goals", []),
            revision_succeeded=data.get("revision_succeeded"),
            energy_spent_on_failure=data.get("energy_spent_on_failure", 0.0),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Phase
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Phase:
    """A major stage of a plan. Contains sub-goals.

    Phases are the primary unit of plan structure. Each phase
    maps to one ProposedGoal cycle in the agent loop.
    """

    id: str
    description: str
    status: PhaseStatus
    plan_id: str

    # Sub-goals within this phase (existing SubGoal type from bus.py)
    sub_goals: list[SubGoal] = field(default_factory=list)

    # Ordering and dependencies
    index: int = 0
    depends_on_phases: list[str] = field(default_factory=list)

    # Phase contracts: declare what data flows between phases
    expected_inputs: dict[str, str] = field(default_factory=dict)
    expected_outputs: dict[str, str] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)

    # Results
    result_summary: str | None = None
    error: str | None = None

    # Energy budget (allocated lazily)
    energy_budget: PhaseEnergyBudget | None = None
    energy_spent: float = 0.0

    # Depth extension
    depth_extension: DepthExtension | None = None

    # Module reference
    module_id: str | None = None

    # Timing
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None
    estimated_duration: float | None = None

    def get_next_executable(self) -> SubGoal | None:
        """Get the next sub-goal that can be executed.

        Same logic as ProposedGoal.get_next_executable() — respects
        depends_on ordering within the phase.
        """
        completed_ids = {
            sg.id
            for sg in self.sub_goals
            if sg.status in (SubGoalStatus.COMPLETED, SubGoalStatus.SKIPPED)
        }

        executable = [
            sg
            for sg in self.sub_goals
            if sg.status in (SubGoalStatus.PENDING, SubGoalStatus.BLOCKED)
            and sg.can_execute(completed_ids)
        ]

        if not executable:
            retryable = [sg for sg in self.sub_goals if sg.should_retry()]
            if retryable:
                retryable.sort(
                    key=lambda sg: sg.effective_priority(GoalPriority.MEDIUM).value
                )
                return retryable[0]
            return None

        executable.sort(
            key=lambda sg: sg.effective_priority(GoalPriority.MEDIUM).value
        )
        return executable[0]

    def is_complete(self) -> bool:
        """Check if all sub-goals are completed or skipped."""
        if not self.sub_goals:
            return False
        return all(
            sg.status in (SubGoalStatus.COMPLETED, SubGoalStatus.SKIPPED)
            for sg in self.sub_goals
        )

    def all_failed(self) -> bool:
        """Check if all non-skippable sub-goals have failed."""
        if not self.sub_goals:
            return False
        critical = [
            sg for sg in self.sub_goals
            if sg.on_failure not in (FailureStrategy.SKIP,)
        ]
        return bool(critical) and all(
            sg.status == SubGoalStatus.FAILED and not sg.should_retry()
            for sg in critical
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "description": self.description,
            "status": self.status.name,
            "plan_id": self.plan_id,
            "index": self.index,
            "depends_on_phases": self.depends_on_phases,
            "expected_inputs": self.expected_inputs,
            "expected_outputs": self.expected_outputs,
            "outputs": {
                k: json.dumps(v, default=str)[:500]
                for k, v in self.outputs.items()
            },
            "result_summary": self.result_summary,
            "error": self.error,
            "energy_spent": self.energy_spent,
            "module_id": self.module_id,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "estimated_duration": self.estimated_duration,
            "sub_goals": [
                {
                    "id": sg.id,
                    "description": sg.description,
                    "tool_name": sg.tool_name,
                    "tool_params": sg.tool_params,
                    "status": sg.status.name,
                    "result": str(sg.result)[:500] if sg.result else None,
                    "error": sg.error,
                    "attempts": sg.attempts,
                    "max_retries": sg.max_retries,
                    "on_failure": sg.on_failure.name,
                    "depends_on": sg.depends_on,
                }
                for sg in self.sub_goals
            ],
            "energy_budget": (
                self.energy_budget.to_dict() if self.energy_budget else None
            ),
            "depth_extension": (
                asdict(self.depth_extension) if self.depth_extension else None
            ),
        }

    @classmethod
    def from_dict(cls, data: dict) -> Phase:
        phase = cls(
            id=data["id"],
            description=data["description"],
            status=PhaseStatus[data["status"]],
            plan_id=data["plan_id"],
            index=data.get("index", 0),
            depends_on_phases=data.get("depends_on_phases", []),
            expected_inputs=data.get("expected_inputs", {}),
            expected_outputs=data.get("expected_outputs", {}),
            outputs={
                k: json.loads(v) if isinstance(v, str) else v
                for k, v in data.get("outputs", {}).items()
            },
            result_summary=data.get("result_summary"),
            error=data.get("error"),
            energy_spent=data.get("energy_spent", 0.0),
            module_id=data.get("module_id"),
            created_at=data.get("created_at", time.time()),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            estimated_duration=data.get("estimated_duration"),
        )

        for sg_data in data.get("sub_goals", []):
            phase.sub_goals.append(SubGoal(
                id=sg_data["id"],
                description=sg_data["description"],
                tool_name=sg_data["tool_name"],
                tool_params=sg_data.get("tool_params", {}),
                status=SubGoalStatus[sg_data["status"]],
                error=sg_data.get("error"),
                attempts=sg_data.get("attempts", 0),
                max_retries=sg_data.get("max_retries", 2),
                on_failure=FailureStrategy[sg_data.get("on_failure", "RETRY")],
                depends_on=sg_data.get("depends_on", []),
            ))

        if data.get("energy_budget"):
            phase.energy_budget = PhaseEnergyBudget.from_dict(data["energy_budget"])

        if data.get("depth_extension"):
            phase.depth_extension = DepthExtension(**data["depth_extension"])

        return phase


# ─────────────────────────────────────────────────────────────────────────────
# PlanDocument
# ─────────────────────────────────────────────────────────────────────────────


def _sg_to_dict(sg: SubGoal) -> dict:
    """Serialize a SubGoal to a dict for ReplanRecord storage."""
    return {
        "id": sg.id,
        "description": sg.description,
        "tool_name": sg.tool_name,
        "tool_params": sg.tool_params,
        "status": sg.status.name,
        "result": str(sg.result)[:500] if sg.result else None,
        "error": sg.error,
        "attempts": sg.attempts,
        "on_failure": sg.on_failure.name,
        "depends_on": sg.depends_on,
    }


@dataclass
class PlanDocument:
    """A persistent, multi-phase plan that survives across goal cycles.

    Stored in the workspace (plans/ directory) and injected into
    StructuredContext so the LLM always knows where it stands.
    """

    VERSION = "1.0"

    id: str
    objective: str
    created_at: float
    updated_at: float
    status: PlanStatus

    # Hierarchy
    phases: list[Phase] = field(default_factory=list)
    current_phase_index: int = 0

    # Depth management
    default_depth_bound: int = 2
    active_depth_extensions: dict[str, DepthExtension] = field(default_factory=dict)

    # Budget allocation
    total_energy_budget: PlanEnergyBudget | None = None

    # Replan history
    replan_history: list[ReplanRecord] = field(default_factory=list)

    # Modular composition
    module_refs: list[str] = field(default_factory=list)

    # Soft preemption
    paused_reason: str | None = None
    abandoned_reason: str | None = None

    # Metadata
    source: str = "llm"
    confidence: float = 1.0
    tags: list[str] = field(default_factory=list)

    # Internal — not serialized
    _persistence_path: str | None = field(default=None, repr=False, compare=False)

    def get_current_phase(self) -> Phase | None:
        """Get the currently active phase."""
        if 0 <= self.current_phase_index < len(self.phases):
            return self.phases[self.current_phase_index]
        return None

    def has_next_phase(self) -> bool:
        """Check if there are more phases after the current one."""
        return self.current_phase_index < len(self.phases) - 1

    # ── Persistence ──────────────────────────────────────────────────────

    def save(self, path: str | None = None) -> None:
        """Persist plan to disk. Atomic write with tmp file."""
        save_path = path or self._persistence_path
        if not save_path:
            return

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        self.updated_at = time.time()

        payload = {
            "version": self.VERSION,
            "saved_at": self.updated_at,
            "id": self.id,
            "objective": self.objective,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "status": self.status.name,
            "current_phase_index": self.current_phase_index,
            "default_depth_bound": self.default_depth_bound,
            "confidence": self.confidence,
            "source": self.source,
            "tags": self.tags,
            "module_refs": self.module_refs,
            "paused_reason": self.paused_reason,
            "abandoned_reason": self.abandoned_reason,
            "phases": [phase.to_dict() for phase in self.phases],
            "active_depth_extensions": {
                k: asdict(v) for k, v in self.active_depth_extensions.items()
            },
            "total_energy_budget": (
                self.total_energy_budget.to_dict()
                if self.total_energy_budget else None
            ),
            "replan_history": [r.to_dict() for r in self.replan_history],
        }

        tmp_path = f"{save_path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)
        os.replace(tmp_path, save_path)

        # Update persistence path for future saves
        self._persistence_path = save_path

        # Maintain active plan pointer
        plans_dir = os.path.dirname(save_path)
        if self.status in (PlanStatus.ACTIVE, PlanStatus.PAUSED, PlanStatus.REPLANNING):
            self._write_pointer(plans_dir, os.path.basename(save_path))
        elif self.status in (PlanStatus.COMPLETED, PlanStatus.FAILED, PlanStatus.ABANDONED):
            self._clear_pointer(plans_dir)

    @classmethod
    def load(cls, path: str) -> PlanDocument:
        """Restore plan from disk."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        version = data.get("version", "1.0")
        if version not in ("1.0",):
            raise ValueError(f"Unknown plan version: {version}")

        plan = cls(
            id=data["id"],
            objective=data["objective"],
            created_at=data["created_at"],
            updated_at=data.get("updated_at", time.time()),
            status=PlanStatus[data["status"]],
            phases=[Phase.from_dict(p) for p in data["phases"]],
            current_phase_index=data["current_phase_index"],
            default_depth_bound=data.get("default_depth_bound", 2),
            confidence=data.get("confidence", 1.0),
            source=data.get("source", "restored"),
            tags=data.get("tags", []),
            module_refs=data.get("module_refs", []),
        )

        plan.paused_reason = data.get("paused_reason")
        plan.abandoned_reason = data.get("abandoned_reason")

        for k, v in data.get("active_depth_extensions", {}).items():
            plan.active_depth_extensions[k] = DepthExtension(**v)

        if data.get("total_energy_budget"):
            plan.total_energy_budget = PlanEnergyBudget.from_dict(
                data["total_energy_budget"]
            )

        for r in data.get("replan_history", []):
            plan.replan_history.append(ReplanRecord.from_dict(r))

        plan._persistence_path = path
        return plan

    @classmethod
    def load_with_recovery(cls, path: str) -> PlanDocument | None:
        """Load with fallback to backup, returns None if unrecoverable."""
        try:
            return cls.load(path)
        except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
            logger.warning("Plan file corrupted: %s, trying backup", e)
            backup = f"{path}.backup"
            if os.path.exists(backup):
                try:
                    return cls.load(backup)
                except Exception:
                    logger.error("Backup also corrupted")
            return None

    # ── Pointer file management ──────────────────────────────────────────

    ACTIVE_POINTER = "active_plan.txt"

    @classmethod
    def find_active(cls, plans_dir: str) -> PlanDocument | None:
        """Find an active plan via pointer file (O(1)) with directory scan fallback."""
        if not os.path.isdir(plans_dir):
            return None

        # Fast path: read pointer file
        pointer_path = os.path.join(plans_dir, cls.ACTIVE_POINTER)
        if os.path.exists(pointer_path):
            try:
                with open(pointer_path, "r") as f:
                    filename = f.read().strip()
                if filename:
                    path = os.path.join(plans_dir, filename)
                    if os.path.exists(path):
                        plan = cls.load(path)
                        if plan.status in (
                            PlanStatus.ACTIVE, PlanStatus.PAUSED, PlanStatus.REPLANNING
                        ):
                            return plan
                    # Pointer stale — clear it
                    os.remove(pointer_path)
            except Exception:
                pass

        # Slow fallback: directory scan (backward compat)
        for filename in sorted(os.listdir(plans_dir), reverse=True):
            if not filename.endswith(".json") or filename.endswith(".tmp"):
                continue
            path = os.path.join(plans_dir, filename)
            try:
                plan = cls.load(path)
                if plan.status in (
                    PlanStatus.ACTIVE, PlanStatus.PAUSED, PlanStatus.REPLANNING
                ):
                    cls._write_pointer(plans_dir, filename)
                    return plan
            except Exception:
                continue
        return None

    @classmethod
    def _write_pointer(cls, plans_dir: str, filename: str) -> None:
        """Write active plan pointer. Atomic via tmp+replace."""
        os.makedirs(plans_dir, exist_ok=True)
        pointer_path = os.path.join(plans_dir, cls.ACTIVE_POINTER)
        tmp = f"{pointer_path}.tmp"
        with open(tmp, "w") as f:
            f.write(filename)
        os.replace(tmp, pointer_path)

    @classmethod
    def _clear_pointer(cls, plans_dir: str) -> None:
        """Clear active plan pointer on completion/abandonment."""
        pointer_path = os.path.join(plans_dir, cls.ACTIVE_POINTER)
        if os.path.exists(pointer_path):
            os.remove(pointer_path)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def classify_failure_type(error: str) -> str:
    """Heuristic classification of failure reason into a category."""
    error_lower = error.lower() if error else ""
    if "timeout" in error_lower:
        return "timeout"
    if "budget" in error_lower or "energy" in error_lower:
        return "budget_exceeded"
    if "no result" in error_lower or "empty" in error_lower or "not found" in error_lower:
        return "no_results"
    return "tool_error"
