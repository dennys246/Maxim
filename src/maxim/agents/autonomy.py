"""Autonomy system for graduated agent control.

Provides three autonomy levels (PLANNING, SUPERVISED, AUTONOMOUS) with
associated policies, safety constraints, and transition management.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────────────────────


class AutonomyLevel(Enum):
    """Graduated autonomy levels for agent operation."""

    PLANNING = "planning"  # Propose only, human approves all
    SUPERVISED = "supervised"  # Act within bounds, escalate edge cases
    AUTONOMOUS = "autonomous"  # Full agency within safety constraints


# ─────────────────────────────────────────────────────────────────────────────
# Data Classes
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Violation:
    """A safety constraint violation."""

    constraint_name: str
    description: str
    severity: str  # "warning", "error", "critical"
    timestamp: float = field(default_factory=time.time)
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class ApprovalPattern:
    """A pattern of previously approved actions for pattern matching."""

    tool_name: str
    param_patterns: dict[str, Any] = field(default_factory=dict)
    approved_at: float = field(default_factory=time.time)
    approved_count: int = 1
    description: str = ""

    def matches(self, proposal: Proposal) -> bool:
        """Check if this pattern matches a proposal."""
        if proposal.action is None:
            return False

        action_tool = proposal.action.get("tool_name", "")
        if action_tool != self.tool_name:
            return False

        # Check param patterns (simple prefix/exact matching)
        action_params = proposal.action.get("params", {})
        for key, pattern in self.param_patterns.items():
            if key not in action_params:
                return False
            if pattern != "*" and action_params[key] != pattern:
                return False

        return True


@dataclass
class Proposal:
    """A proposed action awaiting approval."""

    id: str
    action: dict[str, Any] | None
    reasoning: str
    confidence: float
    strategy_used: str | None = None
    mode_goal_achieved: bool = False
    citations: list[dict[str, str]] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    status: str = "pending"  # "pending", "approved", "rejected", "expired"
    approved_by: str | None = None
    rejected_reason: str | None = None


@dataclass
class AutonomyRequest:
    """Agent request for higher autonomy level."""

    current: AutonomyLevel
    requested: AutonomyLevel
    duration: float | None = None  # seconds, None = indefinite
    justification: str = ""
    timestamp: float = field(default_factory=time.time)
    status: str = "pending"  # "pending", "approved", "rejected"


@dataclass
class AuditEntry:
    """Audit log entry for all actions and decisions."""

    timestamp: float
    autonomy_level: AutonomyLevel
    mode: str
    action_type: str  # "proposed", "approved", "executed", "rejected", "escalated"
    action: dict[str, Any] | None
    reasoning: str
    confidence: float = 0.0
    citations: list[dict[str, str]] = field(default_factory=list)
    source_urls: list[str] = field(default_factory=list)
    salient_extracts: list[str] = field(default_factory=list)
    outcome: str | None = None
    human_involved: bool = False
    error: str | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Safety Constraints
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class SafetyConstraints:
    """Hard limits that apply even in AUTONOMOUS mode."""

    # Absolute forbidden actions (cannot be overridden)
    forbidden_tools: frozenset[str] = frozenset(
        {
            "execute_file",  # Unless explicitly enabled
            "delete_file",
            "system_shutdown",
        }
    )

    # Rate limits
    max_actions_per_second: float = 5.0
    max_mode_switches_per_hour: int = 20
    max_consecutive_failures: int = 5

    # Resource limits
    max_memory_mb: int = 1000
    max_cpu_percent: float = 80.0

    # Self-preservation
    min_battery_for_autonomous: float = 0.2
    require_network_for_autonomous: bool = False

    # Anomaly detection
    abort_on_unexpected_state: bool = True
    abort_on_safety_violation: bool = True

    def check_constraints(self, context: RuntimeContext) -> list[Violation]:
        """Check all safety constraints, return violations."""
        violations = []

        # Check forbidden tools
        if context.pending_tool and context.pending_tool in self.forbidden_tools:
            violations.append(
                Violation(
                    constraint_name="forbidden_tool",
                    description=f"Tool '{context.pending_tool}' is forbidden",
                    severity="critical",
                    context={"tool": context.pending_tool},
                )
            )

        # Check action rate
        if context.actions_last_second > self.max_actions_per_second:
            violations.append(
                Violation(
                    constraint_name="action_rate",
                    description=f"Action rate {context.actions_last_second:.1f}/s exceeds limit",
                    severity="warning",
                    context={"rate": context.actions_last_second},
                )
            )

        # Check consecutive failures
        if context.consecutive_failures >= self.max_consecutive_failures:
            violations.append(
                Violation(
                    constraint_name="consecutive_failures",
                    description=f"{context.consecutive_failures} consecutive failures",
                    severity="error",
                    context={"failures": context.consecutive_failures},
                )
            )

        # Check mode switch rate
        if context.mode_switches_last_hour > self.max_mode_switches_per_hour:
            violations.append(
                Violation(
                    constraint_name="mode_switch_rate",
                    description=f"{context.mode_switches_last_hour} mode switches in last hour",
                    severity="warning",
                    context={"switches": context.mode_switches_last_hour},
                )
            )

        # Check battery for autonomous
        if (
            context.autonomy_level == AutonomyLevel.AUTONOMOUS
            and context.battery_level is not None
            and context.battery_level < self.min_battery_for_autonomous
        ):
            violations.append(
                Violation(
                    constraint_name="low_battery",
                    description=f"Battery {context.battery_level:.0%} too low for autonomous",
                    severity="error",
                    context={"battery": context.battery_level},
                )
            )

        return violations


@dataclass
class RuntimeContext:
    """Context for safety constraint checking."""

    autonomy_level: AutonomyLevel
    pending_tool: str | None = None
    actions_last_second: float = 0.0
    consecutive_failures: int = 0
    mode_switches_last_hour: int = 0
    battery_level: float | None = None
    memory_usage_mb: float = 0.0
    cpu_percent: float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Supervision Policy
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PermissionDenial:
    """First-class denial record — inspectable and loggable."""

    tool_name: str
    reason: str
    rule: str  # "forbidden_tool", "forbidden_prefix", "forbidden_category", "autonomy_level"


@dataclass
class SupervisionPolicy:
    """Defines what the agent can do autonomously in SUPERVISED mode."""

    # Tool permissions
    allowed_tools: set[str] = field(default_factory=set)
    forbidden_tools: set[str] = field(default_factory=set)

    # Confidence thresholds
    min_confidence_autonomous: float = 0.8
    min_confidence_with_explanation: float = 0.5

    # Mode switching permissions
    allowed_mode_transitions: dict[str, set[str]] = field(default_factory=dict)

    # Impact limits
    max_actions_per_minute: int = 10
    requires_confirmation: set[str] = field(default_factory=set)  # Tool names

    # Deny-prefix and deny-category rules
    forbidden_prefixes: tuple[str, ...] = ()
    forbidden_categories: frozenset[str] = frozenset()

    # Escalation triggers
    escalate_on_repeated_failure: int = 3
    escalate_on_novel_situation: bool = True

    # Sandbox and filesystem policies
    sandbox_write_auto_approve: bool = True  # Sandbox writes don't need approval
    sandbox_execute_requires_approval: bool = True  # Sandbox execution needs approval
    cwd_write_requires_approval: bool = True  # CWD writes need approval in supervised

    def can_execute(self, action: dict[str, Any], *, confidence: float | None = None) -> tuple[bool, str | None]:
        """Check if action can be executed autonomously."""
        tool_name = str(action.get("tool_name", "") or "")

        # Check forbidden prefixes
        for prefix in self.forbidden_prefixes:
            if tool_name.startswith(prefix):
                return False, f"Tool '{tool_name}' blocked by prefix rule: {prefix}"

        # Check forbidden categories
        tool_category = str(action.get("category", "") or "")
        if tool_category and tool_category in self.forbidden_categories:
            return False, f"Tool '{tool_name}' blocked by category: {tool_category}"

        if tool_name in self.forbidden_tools:
            return False, f"Tool '{tool_name}' is forbidden"

        if self.allowed_tools and tool_name not in self.allowed_tools:
            return False, f"Tool '{tool_name}' requires approval"

        if confidence is not None and confidence < self.min_confidence_autonomous:
            return False, f"Confidence {confidence:.2f} below threshold"

        if tool_name in self.requires_confirmation:
            return False, f"Tool '{tool_name}' requires confirmation"

        # Check sandbox/filesystem-specific policies
        params = action.get("params", {})

        if tool_name == "write_file":
            path = str(params.get("path", ""))
            # Check if writing to sandbox (auto-approve) or CWD (requires approval)
            from maxim.utils.filesystem_policy import is_path_in_workspace

            if is_path_in_workspace(path):
                if self.sandbox_write_auto_approve:
                    return True, None
            elif self.cwd_write_requires_approval:
                return False, "CWD write requires approval in supervised mode"

        if tool_name == "execute_file":
            if self.sandbox_execute_requires_approval:
                return False, "Sandbox execution requires approval"

        return True, None


# ─────────────────────────────────────────────────────────────────────────────
# Proposal Queue
# ─────────────────────────────────────────────────────────────────────────────


class ProposalQueue:
    """Holds proposals awaiting human approval in PLANNING mode."""

    def __init__(self, max_size: int = 100, expiry_seconds: float = 300.0):
        self._pending: deque[Proposal] = deque(maxlen=max_size)
        self._approved_patterns: list[ApprovalPattern] = []
        self._lock = threading.Lock()
        self._expiry_seconds = expiry_seconds

    def submit(self, proposal: Proposal) -> None:
        """Queue proposal for human review."""
        with self._lock:
            self._prune_expired()
            self._pending.append(proposal)

    def get_pending(self) -> list[Proposal]:
        """Get all pending proposals."""
        with self._lock:
            self._prune_expired()
            return [p for p in self._pending if p.status == "pending"]

    def approve(self, proposal_id: str, approved_by: str = "human") -> bool:
        """Approve a proposal."""
        with self._lock:
            for proposal in self._pending:
                if proposal.id == proposal_id and proposal.status == "pending":
                    proposal.status = "approved"
                    proposal.approved_by = approved_by
                    return True
            return False

    def reject(self, proposal_id: str, reason: str = "") -> bool:
        """Reject a proposal."""
        with self._lock:
            for proposal in self._pending:
                if proposal.id == proposal_id and proposal.status == "pending":
                    proposal.status = "rejected"
                    proposal.rejected_reason = reason
                    return True
            return False

    def get_approved(self) -> list[Proposal]:
        """Get approved proposals ready for execution."""
        with self._lock:
            approved = [p for p in self._pending if p.status == "approved"]
            # Remove approved from queue
            for p in approved:
                self._pending.remove(p)
            return approved

    def add_approval_pattern(self, pattern: ApprovalPattern) -> None:
        """Add a pattern for auto-approval in SUPERVISED mode."""
        with self._lock:
            self._approved_patterns.append(pattern)

    def matches_approved_pattern(self, proposal: Proposal) -> bool:
        """Check if proposal matches previously approved patterns."""
        with self._lock:
            for pattern in self._approved_patterns:
                if pattern.matches(proposal):
                    pattern.approved_count += 1
                    return True
            return False

    def _prune_expired(self) -> None:
        """Remove expired proposals (must hold lock)."""
        now = time.time()
        expired = [p for p in self._pending if p.status == "pending" and (now - p.timestamp) > self._expiry_seconds]
        for p in expired:
            p.status = "expired"


# ─────────────────────────────────────────────────────────────────────────────
# Autonomy Controller
# ─────────────────────────────────────────────────────────────────────────────


class AutonomyController:
    """Manages transitions between autonomy levels."""

    def __init__(
        self,
        initial_level: AutonomyLevel = AutonomyLevel.PLANNING,
        safety_constraints: SafetyConstraints | None = None,
        supervision_policy: SupervisionPolicy | None = None,
        on_level_change: Callable[[AutonomyLevel, AutonomyLevel, str], None] | None = None,
    ):
        self._current_level = initial_level
        self._level_history: list[tuple[float, AutonomyLevel, str]] = [(time.time(), initial_level, "initialization")]
        self._autonomous_until: float | None = None
        self._paused = False
        self._lock = threading.Lock()

        self.safety_constraints = safety_constraints or SafetyConstraints()
        self.supervision_policy = supervision_policy or SupervisionPolicy()
        self.proposal_queue = ProposalQueue()
        self.audit_log: deque[AuditEntry] = deque(maxlen=1000)

        self._on_level_change = on_level_change
        self._pending_requests: list[AutonomyRequest] = []

    @property
    def current_level(self) -> AutonomyLevel:
        """Get current autonomy level."""
        with self._lock:
            # Check if timed autonomy has expired
            if self._autonomous_until is not None and time.time() > self._autonomous_until:
                self._autonomous_until = None
                if self._current_level == AutonomyLevel.AUTONOMOUS:
                    self._transition_to(AutonomyLevel.SUPERVISED, "timed autonomy expired")
            return self._current_level

    @property
    def is_paused(self) -> bool:
        """Check if execution is paused."""
        with self._lock:
            return self._paused

    def escalate(self, reason: str) -> None:
        """Move to more restrictive level."""
        with self._lock:
            if self._current_level == AutonomyLevel.AUTONOMOUS:
                self._transition_to(AutonomyLevel.SUPERVISED, reason)
            elif self._current_level == AutonomyLevel.SUPERVISED:
                self._transition_to(AutonomyLevel.PLANNING, reason)

    def emergency_halt(self, reason: str) -> None:
        """Immediately drop to PLANNING and pause execution."""
        with self._lock:
            self._transition_to(AutonomyLevel.PLANNING, f"EMERGENCY: {reason}")
            self._paused = True
            self._autonomous_until = None

    def resume(self) -> None:
        """Resume from paused state."""
        with self._lock:
            self._paused = False
            logger.info("Autonomy controller resumed")

    def request_autonomy(
        self,
        target_level: AutonomyLevel,
        duration_seconds: float | None = None,
        justification: str = "",
    ) -> AutonomyRequest:
        """Agent requests higher autonomy level (requires human approval)."""
        request = AutonomyRequest(
            current=self._current_level,
            requested=target_level,
            duration=duration_seconds,
            justification=justification,
        )
        with self._lock:
            self._pending_requests.append(request)
        return request

    def approve_autonomy_request(self, request: AutonomyRequest) -> bool:
        """Human approves autonomy request."""
        with self._lock:
            if request not in self._pending_requests:
                return False

            request.status = "approved"
            self._pending_requests.remove(request)

            self._transition_to(request.requested, f"approved: {request.justification}")

            if request.duration is not None:
                self._autonomous_until = time.time() + request.duration

            return True

    def reject_autonomy_request(self, request: AutonomyRequest, reason: str = "") -> bool:
        """Human rejects autonomy request."""
        with self._lock:
            if request not in self._pending_requests:
                return False

            request.status = "rejected"
            self._pending_requests.remove(request)
            logger.info(f"Autonomy request rejected: {reason}")
            return True

    def set_level(
        self,
        level: AutonomyLevel,
        reason: str,
        duration_seconds: float | None = None,
    ) -> None:
        """Directly set autonomy level (human command)."""
        with self._lock:
            self._transition_to(level, reason)
            if duration_seconds is not None and level == AutonomyLevel.AUTONOMOUS:
                self._autonomous_until = time.time() + duration_seconds
            else:
                self._autonomous_until = None

    # Tools that bypass all autonomy checks (including PLANNING mode)
    # These are tools deemed safe for immediate execution without approval
    ALWAYS_ALLOWED_TOOLS: frozenset[str] = frozenset(
        {
            # Visual control (responsive, no side effects)
            "move",  # Direct head movement
            "track_target",  # Visual tracking
            "focus_interests",  # Vision focus
            "focus_on_sound",  # Closed-loop audio orienting (head pose only)
            # Read-only operations (cannot modify state)
            "glob",  # File pattern search
            "read_file",  # Read file contents
            "list_directory",  # List directory contents
            "search_code",  # Regex search file contents (read-only)
            "git_diff",  # Show git differences (read-only)
            "respond",  # Send text response to user
        }
    )

    def can_execute_action(self, action: dict[str, Any], confidence: float | None = None) -> tuple[bool, str | None]:
        """Check if an action can be executed at current autonomy level."""
        level = self.current_level
        tool_name = str(action.get("tool_name", "") or "")

        if self.is_paused:
            return False, "Execution is paused"

        # Check for always-allowed tools FIRST (bypass all autonomy checks)
        # These tools are safe for immediate execution without approval
        if tool_name in self.ALWAYS_ALLOWED_TOOLS:
            return True, None

        # Build runtime context for safety check
        context = RuntimeContext(
            autonomy_level=level,
            pending_tool=tool_name,
            actions_last_second=self._get_recent_action_rate(),
            consecutive_failures=self._get_consecutive_failures(),
            mode_switches_last_hour=self._get_mode_switches_last_hour(),
        )

        # Check safety constraints first (applies to all levels)
        violations = self.safety_constraints.check_constraints(context)
        critical_violations = [v for v in violations if v.severity == "critical"]
        if critical_violations:
            return False, critical_violations[0].description

        # Level-specific checks
        if level == AutonomyLevel.PLANNING:
            return False, "PLANNING mode requires human approval for all actions"

        if level == AutonomyLevel.SUPERVISED:
            return self.supervision_policy.can_execute(action, confidence=confidence)

        # AUTONOMOUS - only safety constraints apply
        return True, None

    def log_action(
        self,
        action_type: str,
        action: dict[str, Any] | None,
        reasoning: str,
        mode: str,
        confidence: float = 0.0,
        citations: list[dict[str, str]] | None = None,
        outcome: str | None = None,
        human_involved: bool = False,
        error: str | None = None,
    ) -> None:
        """Log an action to the audit trail."""
        entry = AuditEntry(
            timestamp=time.time(),
            autonomy_level=self.current_level,
            mode=mode,
            action_type=action_type,
            action=action,
            reasoning=reasoning,
            confidence=confidence,
            citations=citations or [],
            outcome=outcome,
            human_involved=human_involved,
            error=error,
        )
        with self._lock:
            self.audit_log.append(entry)

    def get_audit_log(self, limit: int = 100) -> list[AuditEntry]:
        """Get recent audit log entries."""
        with self._lock:
            return list(self.audit_log)[-limit:]

    def get_level_history(self, limit: int = 50) -> list[tuple[float, AutonomyLevel, str]]:
        """Get autonomy level transition history."""
        with self._lock:
            return self._level_history[-limit:]

    def get_pending_requests(self) -> list[AutonomyRequest]:
        """Get pending autonomy requests."""
        with self._lock:
            return list(self._pending_requests)

    def _transition_to(self, level: AutonomyLevel, reason: str) -> None:
        """Internal: transition to new level (must hold lock)."""
        old_level = self._current_level
        if old_level == level:
            return

        self._current_level = level
        self._level_history.append((time.time(), level, reason))

        logger.info(f"Autonomy level: {old_level.value} -> {level.value} ({reason})")

        if self._on_level_change:
            try:
                self._on_level_change(old_level, level, reason)
            except Exception as e:
                logger.error(f"Error in level change callback: {e}")

    def _get_recent_action_rate(self) -> float:
        """Get actions per second in the last second."""
        now = time.time()
        with self._lock:
            recent = [e for e in self.audit_log if e.action_type == "executed" and (now - e.timestamp) < 1.0]
            return len(recent)

    def _get_consecutive_failures(self) -> int:
        """Get count of consecutive failures."""
        with self._lock:
            count = 0
            for entry in reversed(list(self.audit_log)):
                if entry.action_type == "executed":
                    if entry.error:
                        count += 1
                    else:
                        break
            return count

    def _get_mode_switches_last_hour(self) -> int:
        """Get mode switch count in the last hour."""
        one_hour_ago = time.time() - 3600
        with self._lock:
            return len(
                [
                    e
                    for e in self.audit_log
                    if e.action_type == "executed"
                    and e.action
                    and e.action.get("tool_name") == "mode_switch"
                    and e.timestamp > one_hour_ago
                ]
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize state for persistence."""
        with self._lock:
            return {
                "current_level": self._current_level.value,
                "paused": self._paused,
                "autonomous_until": self._autonomous_until,
                "level_history": [
                    {"timestamp": ts, "level": lvl.value, "reason": reason}
                    for ts, lvl, reason in self._level_history[-50:]
                ],
            }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        safety_constraints: SafetyConstraints | None = None,
        supervision_policy: SupervisionPolicy | None = None,
    ) -> AutonomyController:
        """Restore from persisted state."""
        level = AutonomyLevel(data.get("current_level", "planning"))
        controller = cls(
            initial_level=level,
            safety_constraints=safety_constraints,
            supervision_policy=supervision_policy,
        )
        controller._paused = data.get("paused", False)
        controller._autonomous_until = data.get("autonomous_until")

        # Restore history
        for entry in data.get("level_history", []):
            controller._level_history.append(
                (
                    entry["timestamp"],
                    AutonomyLevel(entry["level"]),
                    entry["reason"],
                )
            )

        return controller


# ─────────────────────────────────────────────────────────────────────────────
# Hard Stops (Always Enforced)
# ─────────────────────────────────────────────────────────────────────────────


HARD_STOP_TRIGGERS: dict[str, str] = {
    # Verbal triggers
    "stop": "User said stop",
    "halt": "User said halt",
    "maxim stop": "Direct stop command",
    "maxim halt": "Direct halt command",
    "emergency stop": "Emergency stop command",
    # Physical triggers (detected by perception)
    "emergency_button": "Physical emergency button",
}


def check_hard_stop(transcript: str | None, percept_override: str | None) -> str | None:
    """Check for hard stop triggers in transcript or percept.

    Returns the reason if a hard stop is triggered, None otherwise.
    """
    if percept_override in HARD_STOP_TRIGGERS:
        return HARD_STOP_TRIGGERS[percept_override]

    if transcript:
        transcript_lower = transcript.lower().strip()
        for trigger, reason in HARD_STOP_TRIGGERS.items():
            if trigger in transcript_lower:
                return reason

    return None
