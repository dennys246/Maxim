"""Preemption circuit — generalized interrupt system.

Any bio-subsystem can trigger preemption to interrupt normal pipeline
processing and force an immediate response. Pain is the first registered
source, but the abstraction supports any future system via register_source().

Components:
- PreemptionCircuit: Evaluates and dispatches preemption signals
- ExecutionTracker: Maintains rolling window of pre-execution snapshots
- Tool reversal registry: Declares reversal strategy per tool
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from maxim.agents.bus import AgentBus

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Tool Reversal Registry
# ─────────────────────────────────────────────────────────────────────────────


class ReversalType(Enum):
    """How a tool's effects can be reversed."""

    PHYSICAL = "physical"  # Can reverse the physical action (movement)
    DECISION = "decision"  # Can only reverse at decision level (abandon goal)


# Registry of reversal strategies per tool
TOOL_REVERSALS: dict[str, ReversalType] = {
    # Movement tools — physical reversal
    "move": ReversalType.PHYSICAL,
    "track_target": ReversalType.PHYSICAL,
    # State-changing tools — decision reversal
    "write_file": ReversalType.DECISION,
    "maxim_command": ReversalType.DECISION,
    "send_message": ReversalType.DECISION,
    # Read-only tools — decision reversal (no side effects)
    "focus_interests": ReversalType.DECISION,
    "read_file": ReversalType.DECISION,
    "glob": ReversalType.DECISION,
    "respond": ReversalType.DECISION,
    "speak": ReversalType.DECISION,
}


# ─────────────────────────────────────────────────────────────────────────────
# Data Classes
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class RobotState:
    """Physical state before action — for movement reversal."""

    head_yaw: float
    head_pitch: float
    timestamp: float


@dataclass
class ExecutionSnapshot:
    """Complete decision context before tool execution.

    Captures everything needed to reverse the decision that led here.
    """

    timestamp: float
    goal_description: str
    tool_name: str
    tool_params: dict[str, Any]
    robot_state: RobotState | None = None
    recent_outcomes: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class SourceConfig:
    """Configuration for a preemption source."""

    threshold: float  # Minimum intensity to trigger
    cooldown: float  # Seconds between triggers from this source
    priority: int  # Higher = takes precedence


@dataclass
class WithdrawalParams:
    """Parameters for graded physical withdrawal."""

    reversal_fraction: float = 0.4
    speed_damping: float = 0.6
    hold_after: float = 0.5


@dataclass
class PreemptionResponse:
    """What to do when preemption fires."""

    action: str  # "halt", "withdraw", "reverse_decision", "override_goal"
    goal_override: Any = None  # ProposedGoal if action == "override_goal"
    duration: float | None = None
    withdrawal: WithdrawalParams | None = None


@dataclass
class PreemptionSignal:
    """Signal from a bio-subsystem requesting preemption."""

    source: str  # "pain", "energy", "startle", "user_comms"
    intensity: float  # 0.0 - 1.0
    response: PreemptionResponse
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class PreemptionEvent:
    """Published on bus when preemption is accepted."""

    signal: PreemptionSignal
    timestamp: float = field(default_factory=time.time)


@dataclass
class PreemptionCleared:
    """Published on bus when preemption hold ends."""

    source: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class GoalDiscarded:
    """Published when a goal is discarded due to preemption reversal."""

    goal_description: str
    reason: str
    preemption_source: str
    timestamp: float = field(default_factory=time.time)


class PreemptionState(Enum):
    """State of the preemption hold in the goal agent."""

    NONE = "none"
    REVERSING = "reversing"
    HOLDING = "holding"
    CLEARING = "clearing"


# ─────────────────────────────────────────────────────────────────────────────
# Execution Tracker
# ─────────────────────────────────────────────────────────────────────────────


class ExecutionTracker:
    """Maintains a rolling window of snapshots for preemption reversal."""

    def __init__(self, max_snapshots: int = 20) -> None:
        self._snapshots: deque[ExecutionSnapshot] = deque(maxlen=max_snapshots)

    def capture_before(
        self,
        goal_description: str,
        tool_name: str,
        tool_params: dict[str, Any],
        robot: Any = None,
    ) -> None:
        """Called by agent_loop.py just before executor.execute()."""
        robot_state = None
        if robot is not None:
            try:
                robot_state = RobotState(
                    head_yaw=robot.get_yaw(),
                    head_pitch=robot.get_pitch(),
                    timestamp=time.time(),
                )
            except (AttributeError, Exception):
                pass  # Robot may not have these methods

        self._snapshots.append(
            ExecutionSnapshot(
                timestamp=time.time(),
                goal_description=goal_description,
                tool_name=tool_name,
                tool_params=tool_params,
                robot_state=robot_state,
            )
        )

    def capture_batch(
        self,
        actions: list[dict[str, Any]],
        goal_description: str,
        robot: Any = None,
    ) -> None:
        """Capture snapshot for an entire parallel batch."""
        for action in actions:
            self.capture_before(
                goal_description=goal_description,
                tool_name=action.get("tool_name", ""),
                tool_params=action.get("params", {}),
                robot=robot,
            )

    @property
    def latest(self) -> ExecutionSnapshot | None:
        return self._snapshots[-1] if self._snapshots else None

    @property
    def active_batch(self) -> list[ExecutionSnapshot]:
        """All snapshots from the current batch (for preemption reversal)."""
        if not self._snapshots:
            return []
        latest_ts = self._snapshots[-1].timestamp
        # Snapshots within 0.1s are considered same batch
        return [s for s in self._snapshots if abs(s.timestamp - latest_ts) < 0.1]


# ─────────────────────────────────────────────────────────────────────────────
# Preemption Circuit
# ─────────────────────────────────────────────────────────────────────────────


class PreemptionCircuit:
    """Generalized interrupt system.

    Any bio-subsystem can trigger preemption. Pain is the first registered
    source, but the abstraction supports any future system via
    register_source().
    """

    def __init__(self, bus: "AgentBus") -> None:
        self.bus = bus
        self.active_preemption: PreemptionSignal | None = None
        self.cooldowns: dict[str, float] = {}
        self.source_config: dict[str, SourceConfig] = {}
        bus.subscribe(PreemptionSignal, self._evaluate)

    def register_source(self, name: str, config: SourceConfig) -> None:
        """Register a preemption source with its configuration."""
        self.source_config[name] = config

    def _evaluate(self, signal: PreemptionSignal) -> None:
        """Evaluate whether a preemption signal should be accepted."""
        cfg = self.source_config.get(signal.source)
        if not cfg:
            logger.debug("Preemption signal from unregistered source: %s", signal.source)
            return
        if signal.intensity < cfg.threshold:
            return
        if self._in_cooldown(signal.source):
            return
        if self.active_preemption and self._priority(self.active_preemption) >= self._priority(signal):
            return

        self.active_preemption = signal
        self.cooldowns[signal.source] = time.time() + cfg.cooldown
        logger.info(
            "Preemption accepted: source=%s intensity=%.2f action=%s",
            signal.source,
            signal.intensity,
            signal.response.action,
        )
        self.bus.publish(PreemptionEvent(signal=signal))

    def clear(self) -> None:
        """Clear the active preemption and publish cleared event."""
        suspended = self.active_preemption
        self.active_preemption = None
        if suspended:
            self.bus.publish(PreemptionCleared(source=suspended.source))

    def _in_cooldown(self, source: str) -> bool:
        expires = self.cooldowns.get(source, 0.0)
        return time.time() < expires

    def _priority(self, signal: PreemptionSignal) -> int:
        cfg = self.source_config.get(signal.source)
        return cfg.priority if cfg else 0
