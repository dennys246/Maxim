"""Pain detection module - Detects aversive proprioceptive events.

Monitors movement metrics and generates pain signals when thresholds
are exceeded, enabling learning to avoid harmful movement patterns.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable

from maxim.decisions.causal_link import Valence
from maxim.reactions.types import Reaction, ReactionContext, TraceSnapshot

if TYPE_CHECKING:
    from maxim.agents.bus import ToolErrorKind
    from maxim.proprioception.movement_tracker import MovementMetrics, MovementTracker

logger = logging.getLogger(__name__)


class PainType(Enum):
    """Categories of pain signals."""

    EXCESSIVE_VELOCITY = "excessive_velocity"  # Too fast movement
    DIRECTION_THRASHING = "direction_thrashing"  # Rapid back-and-forth
    SUSTAINED_STRAIN = "sustained_strain"  # Prolonged near-limit position
    EXCESSIVE_ACCELERATION = "excessive_acceleration"  # Too rapid speed change
    MOVEMENT_FAILURE = "movement_failure"  # Commanded but didn't reach target
    TOOL_FAILURE = "tool_failure"
    TOOL_TIMEOUT = "tool_timeout"
    TOOL_INVALID_INPUT = "tool_invalid_input"
    TOOL_SUSTAINED = "tool_sustained"

    # Non-motor sources (via PainBus)
    RESOURCE_EXHAUSTION = "resource_exhaustion"  # Energy budget depleted
    COGNITIVE_OVERLOAD = "cognitive_overload"  # Repeated failures, context loss
    EXTERNAL_SIGNAL = "external_signal"  # From percept/simulation/other robot
    SAFETY_VIOLATION = "safety_violation"  # FearAgent-detected threat

    # Predictive / felt signal, fired BEFORE an action executes. Combines
    # NAc's learned prediction with innate priors (e.g. known-sensitive
    # paths). Biological analog: ACC/vmPFC anticipatory aversion.
    ANTICIPATED = "anticipated"  # Predicted likelihood of pain before action


@dataclass(frozen=True)
class PainConfig:
    """Configuration for pain detection thresholds.

    Attributes:
        angular_velocity_pain: Angular velocity (deg/sec) that triggers pain.
        translation_velocity_pain: Translation velocity (mm/sec) that triggers pain.
        angular_acceleration_pain: Angular acceleration (deg/sec²) that triggers pain.
        reversal_pain_threshold: Number of direction reversals in window that triggers pain.
        pain_scaling_factor: Factor to convert excess velocity to intensity (0-1).
        pain_cooldown_seconds: Minimum time between pain signals of same type.
        movement_failure_angular_threshold: Angular error (degrees) to trigger movement failure.
        movement_failure_translation_threshold: Translation error (mm) to trigger movement failure.
        movement_failure_timeout: Seconds after command to check for failure.
    """

    # Velocity thresholds
    angular_velocity_pain: float = 100.0  # deg/sec triggers pain
    translation_velocity_pain: float = 50.0  # mm/sec triggers pain

    # Acceleration threshold
    angular_acceleration_pain: float = 200.0  # deg/sec² triggers pain

    # Thrashing detection
    reversal_pain_threshold: int = 3  # Reversals in window triggers pain

    # Pain intensity scaling
    pain_scaling_factor: float = 0.01  # Converts excess velocity to 0-1 intensity

    # Cooldown to prevent spam
    pain_cooldown_seconds: float = 0.5

    # Movement failure detection - when commanded position isn't reached
    movement_failure_angular_threshold: float = 10.0  # degrees - error to trigger pain
    movement_failure_translation_threshold: float = 5.0  # mm - error to trigger pain
    movement_failure_timeout: float = 2.0  # seconds after command to check


@dataclass
class PainSignal:
    """A detected pain event."""

    pain_type: PainType
    intensity: float  # 0-1 scale (0 = threshold, 1 = severe)
    timestamp: float
    angular_velocity: float = 0.0  # The velocity that triggered this
    translation_velocity: float = 0.0
    direction_reversals: int = 0
    context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "pain_type": self.pain_type.value,
            "intensity": round(self.intensity, 3),
            "timestamp": self.timestamp,
            "angular_velocity": round(self.angular_velocity, 2),
            "translation_velocity": round(self.translation_velocity, 2),
            "direction_reversals": self.direction_reversals,
            "context": self.context,
        }


class PainDetector:
    """Detects pain signals from movement metrics.

    Monitors movement velocity, acceleration, and direction changes,
    generating pain signals when thresholds are exceeded.

    Example:
        detector = PainDetector(PainConfig(angular_velocity_pain=100.0))

        # Register callback for pain events
        detector.add_pain_callback(lambda signal: print(f"Pain: {signal}"))

        # Record positions (usually from sync_head_position)
        signal = detector.record_position(
            yaw=45.0, pitch=10.0,
            x=5.0, y=3.0, z=2.0, roll=0.0,
        )

        if signal:
            print(f"Pain detected: {signal.pain_type}")
    """

    def __init__(
        self,
        config: PainConfig | None = None,
        movement_tracker: "MovementTracker | None" = None,
        pain_bus: Any | None = None,
    ) -> None:
        """Initialize the pain detector.

        Args:
            config: Pain detection configuration. Uses defaults if None.
            movement_tracker: Optional pre-existing tracker. Creates one if None.
            pain_bus: Optional PainBus for centralized pain dispatch. When
                provided, _emit_pain() publishes to the bus instead of
                invoking internal callbacks. Consumers should subscribe to
                the bus directly rather than using add_pain_callback().
        """
        self.config = config or PainConfig()
        self._lock = threading.Lock()
        self._pain_bus = pain_bus

        # Create tracker if not provided
        if movement_tracker is None:
            from maxim.proprioception.movement_tracker import MovementTracker

            self._tracker = MovementTracker()
        else:
            self._tracker = movement_tracker

        # Cooldown tracking
        self._last_pain_time: dict[PainType, float] = {}

        # Callbacks for pain events (legacy — prefer pain_bus.subscribe())
        self._callbacks: list[Callable[[PainSignal], None]] = []

        # Pending movement tracking for failure detection
        self._pending_target: dict[str, float] | None = None  # yaw, pitch, x, y, z, roll
        self._pending_target_time: float = 0.0
        self._pending_action_sig: str = ""

        # Statistics
        self._total_pain_signals = 0
        self._pain_counts: dict[PainType, int] = {t: 0 for t in PainType}

    def record_position(
        self,
        yaw: float,
        pitch: float,
        x: float,
        y: float,
        z: float,
        roll: float,
    ) -> PainSignal | None:
        """Record position and check for pain.

        Args:
            yaw, pitch, roll: Angular positions (degrees)
            x, y, z: Translation positions (mm)

        Returns:
            PainSignal if pain detected, None otherwise.
        """
        metrics = self._tracker.record_position(yaw, pitch, x, y, z, roll)

        if metrics is None:
            return None

        # Check for movement velocity pain
        velocity_pain = self._check_for_pain(metrics)
        if velocity_pain:
            return velocity_pain

        # Check for movement failure pain (if we have a pending target)
        failure_pain = self._check_movement_failure(yaw, pitch, x, y, z, roll)
        if failure_pain:
            return failure_pain

        return None

    def set_movement_target(
        self,
        yaw: float | None = None,
        pitch: float | None = None,
        x: float | None = None,
        y: float | None = None,
        z: float | None = None,
        roll: float | None = None,
        action_signature: str = "",
    ) -> None:
        """Set expected target position for movement failure detection.

        Call this BEFORE issuing a movement command. After movement completes,
        record_position() will check if actual position matches target.

        Args:
            yaw, pitch, roll: Target angular positions (degrees). None = don't check.
            x, y, z: Target translation positions (mm). None = don't check.
            action_signature: Identifier for the action (for learning).
        """
        self._pending_target = {
            "yaw": yaw,
            "pitch": pitch,
            "x": x,
            "y": y,
            "z": z,
            "roll": roll,
        }
        self._pending_target_time = time.time()
        self._pending_action_sig = action_signature
        logger.debug(
            "Movement target set: yaw=%s, pitch=%s (sig=%s)",
            yaw,
            pitch,
            action_signature,
        )

    def clear_movement_target(self) -> None:
        """Clear pending movement target (call after successful movement)."""
        self._pending_target = None
        self._pending_target_time = 0.0
        self._pending_action_sig = ""

    def _check_movement_failure(
        self,
        yaw: float,
        pitch: float,
        x: float,
        y: float,
        z: float,
        roll: float,
    ) -> PainSignal | None:
        """Check if current position differs from pending target.

        Only triggers after timeout has elapsed (allowing movement to complete).

        Returns:
            PainSignal if significant position error detected, None otherwise.
        """
        if self._pending_target is None:
            return None

        now = time.time()
        elapsed = now - self._pending_target_time

        # Only check after movement should have completed
        if elapsed < self.config.movement_failure_timeout:
            return None

        # Check if we can emit this pain type (cooldown)
        if not self._can_emit_pain(PainType.MOVEMENT_FAILURE, now):
            self.clear_movement_target()
            return None

        # Calculate position errors
        target = self._pending_target
        errors = []

        # Angular error (yaw + pitch combined)
        angular_error = 0.0
        if target.get("yaw") is not None:
            angular_error += abs(yaw - target["yaw"])
        if target.get("pitch") is not None:
            angular_error += abs(pitch - target["pitch"])

        # Translation error (euclidean distance)
        translation_error = 0.0
        dx = (x - target["x"]) if target.get("x") is not None else 0.0
        dy = (y - target["y"]) if target.get("y") is not None else 0.0
        dz = (z - target["z"]) if target.get("z") is not None else 0.0
        translation_error = (dx**2 + dy**2 + dz**2) ** 0.5

        # Check thresholds
        angular_threshold = self.config.movement_failure_angular_threshold
        translation_threshold = self.config.movement_failure_translation_threshold

        # Determine if movement failed and calculate intensity
        failed = False
        intensity = 0.0

        if angular_error > angular_threshold:
            failed = True
            # Intensity scales with how far off we are
            excess = angular_error - angular_threshold
            intensity = max(intensity, min(1.0, 0.3 + excess * 0.02))
            errors.append(f"angular_error={angular_error:.1f}°")

        if translation_error > translation_threshold:
            failed = True
            excess = translation_error - translation_threshold
            intensity = max(intensity, min(1.0, 0.3 + excess * 0.05))
            errors.append(f"translation_error={translation_error:.1f}mm")

        # Clear pending target regardless of outcome
        action_sig = self._pending_action_sig
        self.clear_movement_target()

        if not failed:
            return None

        # Generate pain signal for movement failure
        signal = PainSignal(
            pain_type=PainType.MOVEMENT_FAILURE,
            intensity=intensity,
            timestamp=now,
            angular_velocity=0.0,  # Not velocity-related
            translation_velocity=0.0,
            direction_reversals=0,
            context={
                "angular_error": angular_error,
                "translation_error": translation_error,
                "target_yaw": target.get("yaw"),
                "target_pitch": target.get("pitch"),
                "actual_yaw": yaw,
                "actual_pitch": pitch,
                "action_signature": action_sig,
                "errors": errors,
            },
        )
        return self._emit_pain(signal)

    def _check_for_pain(self, metrics: "MovementMetrics") -> PainSignal | None:
        """Check metrics against thresholds and generate pain if exceeded."""
        now = time.time()

        # Check angular velocity
        if metrics.angular_velocity > self.config.angular_velocity_pain:
            if self._can_emit_pain(PainType.EXCESSIVE_VELOCITY, now):
                excess = metrics.angular_velocity - self.config.angular_velocity_pain
                intensity = min(1.0, excess * self.config.pain_scaling_factor)

                signal = PainSignal(
                    pain_type=PainType.EXCESSIVE_VELOCITY,
                    intensity=intensity,
                    timestamp=now,
                    angular_velocity=metrics.angular_velocity,
                    translation_velocity=metrics.translation_velocity,
                    direction_reversals=metrics.direction_reversals,
                )
                return self._emit_pain(signal)

        # Check translation velocity
        if metrics.translation_velocity > self.config.translation_velocity_pain:
            if self._can_emit_pain(PainType.EXCESSIVE_VELOCITY, now):
                excess = metrics.translation_velocity - self.config.translation_velocity_pain
                intensity = min(1.0, excess * self.config.pain_scaling_factor)

                signal = PainSignal(
                    pain_type=PainType.EXCESSIVE_VELOCITY,
                    intensity=intensity,
                    timestamp=now,
                    angular_velocity=metrics.angular_velocity,
                    translation_velocity=metrics.translation_velocity,
                    direction_reversals=metrics.direction_reversals,
                )
                return self._emit_pain(signal)

        # Check direction thrashing
        if metrics.direction_reversals >= self.config.reversal_pain_threshold:
            if self._can_emit_pain(PainType.DIRECTION_THRASHING, now):
                # Intensity scales with number of reversals
                excess_reversals = metrics.direction_reversals - self.config.reversal_pain_threshold
                intensity = min(1.0, 0.5 + excess_reversals * 0.2)

                signal = PainSignal(
                    pain_type=PainType.DIRECTION_THRASHING,
                    intensity=intensity,
                    timestamp=now,
                    angular_velocity=metrics.angular_velocity,
                    translation_velocity=metrics.translation_velocity,
                    direction_reversals=metrics.direction_reversals,
                )
                return self._emit_pain(signal)

        # Check angular acceleration
        if abs(metrics.angular_acceleration) > self.config.angular_acceleration_pain:
            if self._can_emit_pain(PainType.EXCESSIVE_ACCELERATION, now):
                excess = abs(metrics.angular_acceleration) - self.config.angular_acceleration_pain
                intensity = min(1.0, excess * self.config.pain_scaling_factor * 0.5)

                signal = PainSignal(
                    pain_type=PainType.EXCESSIVE_ACCELERATION,
                    intensity=intensity,
                    timestamp=now,
                    angular_velocity=metrics.angular_velocity,
                    translation_velocity=metrics.translation_velocity,
                    direction_reversals=metrics.direction_reversals,
                )
                return self._emit_pain(signal)

        return None

    def _can_emit_pain(self, pain_type: PainType, now: float) -> bool:
        """Check if enough time has passed since last pain of this type."""
        last_time = self._last_pain_time.get(pain_type, 0.0)
        return (now - last_time) >= self.config.pain_cooldown_seconds

    def _emit_pain(self, signal: PainSignal) -> PainSignal:
        """Emit a pain signal and update tracking.

        If a PainBus is configured, publishes to the bus (which notifies
        all bus subscribers). Otherwise falls back to internal callbacks.
        """
        with self._lock:
            self._last_pain_time[signal.pain_type] = signal.timestamp
            self._total_pain_signals += 1
            self._pain_counts[signal.pain_type] = self._pain_counts.get(signal.pain_type, 0) + 1

        # Log at WARNING for high intensity pain, INFO for moderate
        log_msg = "Pain detected: %s (intensity=%.2f, angular_vel=%.1f deg/s, translation_vel=%.1f mm/s, reversals=%d)"
        log_args = (
            signal.pain_type.value,
            signal.intensity,
            signal.angular_velocity,
            signal.translation_velocity,
            signal.direction_reversals,
        )

        if signal.intensity >= 0.7:
            logger.warning(log_msg, *log_args)
        else:
            logger.info(log_msg, *log_args)

        # Dispatch: prefer ReactionBus (via PainBus wrapper), fall back to internal callbacks
        if self._pain_bus is not None:
            entity_path = signal.context.get("entity_path", "")
            reaction = Reaction(
                kind="pain",
                intensity=signal.intensity,
                valence=Valence.NEGATIVE,
                timestamp=signal.timestamp,
                source=f"pain_detector:{signal.pain_type.value}",
                context=ReactionContext(
                    bindings={"entity_path": TraceSnapshot(percept_id=entity_path)} if entity_path else {},
                ),
            )
            self._pain_bus.reaction_bus.publish(reaction)
        else:
            for callback in self._callbacks:
                try:
                    callback(signal)
                except Exception as e:
                    logger.warning("Pain callback error: %s", e)

        return signal

    def add_pain_callback(self, callback: Callable[[PainSignal], None]) -> None:
        """Register a callback for pain events.

        When a PainBus is configured, prefer pain_bus.subscribe() instead.
        This method still works but callbacks registered here are only
        invoked when no PainBus is present (fallback path).

        Args:
            callback: Function to call when pain is detected.
        """
        if self._pain_bus is not None:
            logger.debug("add_pain_callback called with PainBus active; consider using pain_bus.subscribe() instead")
        self._callbacks.append(callback)

    def remove_pain_callback(self, callback: Callable[[PainSignal], None]) -> None:
        """Remove a previously registered callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def record_tool_error(
        self,
        tool_name: str,
        error: str,
        error_kind: "ToolErrorKind",
        context: dict[str, Any] | None = None,
    ) -> PainSignal | None:
        """Record a tool error as a pain signal.

        Args:
            tool_name: Name of the tool that failed.
            error: Error message.
            error_kind: Classification of the error.
            context: Optional additional context.

        Returns:
            PainSignal if emitted (not in cooldown), None otherwise.
        """
        from maxim.agents.bus import ToolErrorKind

        pain_type = {
            ToolErrorKind.TIMEOUT: PainType.TOOL_TIMEOUT,
            ToolErrorKind.INVALID_INPUT: PainType.TOOL_INVALID_INPUT,
        }.get(error_kind, PainType.TOOL_FAILURE)

        now = time.time()
        if now - self._last_pain_time.get(pain_type, 0) < self.config.pain_cooldown_seconds:
            return None

        # Escalating intensity based on repeated failures
        key = f"tool:{tool_name}"
        if not hasattr(self, "_tool_pain_counts"):
            self._tool_pain_counts: dict[str, int] = {}
        self._tool_pain_counts[key] = self._tool_pain_counts.get(key, 0) + 1
        count = self._tool_pain_counts[key]
        intensity = min(1.0, 0.3 + 0.17 * math.log(max(1, count)))

        signal = PainSignal(
            pain_type=pain_type,
            intensity=intensity,
            timestamp=now,
            context={
                "tool_name": tool_name,
                "error": error,
                "error_kind": error_kind.value,
                **(context or {}),
            },
        )
        self._emit_pain(signal)
        return signal

    def record_tool_running(
        self,
        tool_name: str,
        elapsed: float,
        expected: float,
    ) -> PainSignal | None:
        """Record a tool running longer than expected as sustained pain.

        Args:
            tool_name: Name of the long-running tool.
            elapsed: Seconds elapsed so far.
            expected: Expected duration in seconds.

        Returns:
            PainSignal if the tool is over its expected time, None otherwise.
        """
        ratio = elapsed / expected if expected > 0 else elapsed / 30.0
        if ratio < 1.0:
            return None
        intensity = min(1.0, 0.3 * ratio)
        signal = PainSignal(
            pain_type=PainType.TOOL_SUSTAINED,
            intensity=intensity,
            timestamp=time.time(),
            context={
                "tool_name": tool_name,
                "elapsed": elapsed,
                "expected": expected,
                "ratio": ratio,
            },
        )
        self._emit_pain(signal)
        return signal

    def get_stats(self) -> dict[str, Any]:
        """Get pain detection statistics."""
        return {
            "total_pain_signals": self._total_pain_signals,
            "pain_counts": {t.value: c for t, c in self._pain_counts.items()},
            "sample_count": self._tracker.get_sample_count(),
        }

    def clear(self) -> None:
        """Clear all state and reset tracking."""
        self._tracker.clear()
        self._last_pain_time.clear()
        self._total_pain_signals = 0
        self._pain_counts = {t: 0 for t in PainType}
        self.clear_movement_target()

    @property
    def tracker(self) -> "MovementTracker":
        """Access the underlying movement tracker."""
        return self._tracker


__all__ = ["PainDetector", "PainConfig", "PainSignal", "PainType"]
