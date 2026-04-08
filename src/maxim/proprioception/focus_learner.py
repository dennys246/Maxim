"""FocusLearner - Dynamic movement correction through focus feedback.

Learns the relationship between commanded movements and actual focus results,
adaptively adjusting movement gain to achieve accurate targeting.

Uses Rescorla-Wagner learning (ΔV = α(λ - V)) for stable convergence:
- V = current gain estimate (our prediction of optimal gain)
- λ = optimal gain computed from each trial's outcome
- α = learning rate controlling adaptation speed

Instead of a fixed dampening factor, this learns the optimal correction by:
1. Recording where we intended to focus (target pixel position)
2. Observing where the target ended up after movement
3. Computing optimal gain from overshoot ratio (optimal = current/overshoot)
4. Updating gain via prediction error (optimal - current) × learning_rate

This creates a closed-loop system that naturally adapts to:
- Camera latency
- Mechanical overshoot
- Head/neck dynamics
- Target motion prediction errors

The R-W algorithm provides:
- Smooth asymptotic convergence (no oscillation)
- Faster learning when error is large
- Self-limiting as gain approaches optimal
"""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from maxim.decisions.nac import NAc

logger = logging.getLogger(__name__)


@dataclass
class FocusLearnerConfig:
    """Configuration for focus learning.

    Attributes:
        initial_gain: Starting movement gain before learning (0-1).
        min_gain: Minimum allowed gain to prevent under-movement.
        max_gain: Maximum allowed gain to prevent overshoot.
        learning_rate: How fast to adapt gain (0-1). Higher = faster but noisier.
        error_threshold: Pixel error below which movement is "successful".
        sample_window: Number of recent samples to use for gain estimation.
        oscillation_threshold: Number of sign changes to detect oscillation.
        use_nac: Whether to use NAc for pattern learning (vs simple averaging).
        gain_decay_rate: Per-second decay toward initial_gain when not learning.
        min_result_delay: Minimum seconds between intent and result to ensure movement completed.
        max_correction: Maximum correction factor per learning step (prevents runaway).
        persist_path: Path to save/load learned gains (None to disable persistence).
        auto_save_interval: Seconds between auto-saves (0 to disable).
    """

    initial_gain: float = 0.7  # Higher initial - we want to actually reach targets
    min_gain: float = 0.4
    max_gain: float = 1.0
    learning_rate: float = 0.2  # Faster learning
    error_threshold: float = 30.0  # pixels
    sample_window: int = 20
    oscillation_threshold: int = 3
    use_nac: bool = True
    gain_decay_rate: float = 0.01  # per second
    min_result_delay: float = 0.3  # Wait at least 300ms for movement to complete
    max_correction: float = 1.5  # Don't increase gain by more than 50% per step
    persist_path: str = ""  # resolved via maxim.utils.paths at runtime
    auto_save_interval: float = 60.0  # Save every 60 seconds

    def __post_init__(self) -> None:
        if not self.persist_path:
            from maxim.utils.paths import resolve_user_state
            self.persist_path = str(resolve_user_state("util/focus_learner.json"))


@dataclass
class FocusSample:
    """A single focus attempt and its outcome."""

    timestamp: float
    commanded_du: float  # Commanded horizontal pixel delta
    commanded_dv: float  # Commanded vertical pixel delta
    actual_du: float | None = None  # Where target actually ended up (horizontal)
    actual_dv: float | None = None  # Where target actually ended up (vertical)
    error_u: float | None = None  # Difference from center after movement
    error_v: float | None = None
    gain_used: float = 1.0  # Gain that was applied


@dataclass
class FocusError:
    """Computed focus error for learning."""

    overshoot_ratio_u: float  # >1 = overshoot, <1 = undershoot
    overshoot_ratio_v: float
    direction_u: int  # Sign of commanded movement: -1, 0, or 1
    direction_v: int
    magnitude_u: float  # Absolute commanded pixels
    magnitude_v: float


class FocusLearner:
    """Learns optimal movement gain through focus feedback.

    This creates a closed-loop control system that adapts movement
    based on observed results, rather than using fixed dampening.

    Example:
        learner = FocusLearner(nac=nac)

        # Before movement
        gain = learner.get_gain(du=100, dv=50)
        adjusted_du = du * gain
        adjusted_dv = dv * gain

        # Record what we're about to do
        learner.record_intent(du=100, dv=50, gain=gain)

        # After movement completes, provide feedback
        learner.record_result(target_u=340, target_v=260)  # Where target is now

        # Over time, gain adapts to achieve center targeting
    """

    def __init__(
        self,
        config: FocusLearnerConfig | None = None,
        nac: "NAc | None" = None,
        image_center: tuple[float, float] = (320.0, 240.0),
    ) -> None:
        """Initialize focus learner.

        Args:
            config: Learning configuration.
            nac: Optional NAc for pattern-based learning.
            image_center: (u, v) center of image frame.
        """
        self.config = config or FocusLearnerConfig()
        self._nac = nac
        self._image_center = image_center

        # Current learned gains (separate for horizontal/vertical)
        self._gain_h: float = self.config.initial_gain
        self._gain_v: float = self.config.initial_gain

        # Direction-aware gains: positive vs negative movements may differ
        self._gain_h_pos: float = self.config.initial_gain
        self._gain_h_neg: float = self.config.initial_gain
        self._gain_v_pos: float = self.config.initial_gain
        self._gain_v_neg: float = self.config.initial_gain

        # Sample history for learning
        self._samples: deque[FocusSample] = deque(maxlen=self.config.sample_window)

        # Pending intent (waiting for result)
        self._pending_sample: FocusSample | None = None

        # Oscillation detection
        self._recent_directions: deque[int] = deque(maxlen=10)

        # Statistics
        self._total_samples = 0
        self._successful_focuses = 0
        self._oscillation_count = 0
        self._last_update_time = time.time()
        self._last_save_time = time.time()

        # Auto-load persisted state
        if self.config.persist_path:
            self.load(self.config.persist_path)

    def get_gain(
        self,
        du: float,
        dv: float,
    ) -> tuple[float, float]:
        """Get the current learned gain for a movement.

        Returns separate gains for horizontal and vertical axes,
        taking movement direction into account.

        Args:
            du: Commanded horizontal pixel delta.
            dv: Commanded vertical pixel delta.

        Returns:
            (gain_h, gain_v) tuple of gains to apply.
        """
        # Apply time-based decay toward initial values
        self._apply_decay()

        # Select direction-aware gain
        if du >= 0:
            gain_h = self._gain_h_pos
        else:
            gain_h = self._gain_h_neg

        if dv >= 0:
            gain_v = self._gain_v_pos
        else:
            gain_v = self._gain_v_neg

        return gain_h, gain_v

    def get_average_gain(self) -> float:
        """Get overall average gain (for logging/debugging)."""
        return (self._gain_h_pos + self._gain_h_neg + self._gain_v_pos + self._gain_v_neg) / 4

    def record_intent(
        self,
        du: float,
        dv: float,
        gain_h: float | None = None,
        gain_v: float | None = None,
    ) -> None:
        """Record the intent to make a movement.

        Call this BEFORE executing the movement. Then call record_result()
        after the movement completes to provide feedback.

        Args:
            du: Commanded horizontal pixel delta (before gain).
            dv: Commanded vertical pixel delta (before gain).
            gain_h: Horizontal gain that will be applied (if different from get_gain).
            gain_v: Vertical gain that will be applied.
        """
        # Discard previous pending sample if not completed
        if self._pending_sample is not None:
            logger.debug("Discarding incomplete focus sample")

        actual_gain_h = gain_h if gain_h is not None else self._gain_h
        actual_gain_v = gain_v if gain_v is not None else self._gain_v

        self._pending_sample = FocusSample(
            timestamp=time.time(),
            commanded_du=du,
            commanded_dv=dv,
            gain_used=(actual_gain_h + actual_gain_v) / 2,
        )

        # Track direction for oscillation detection
        direction = 1 if du >= 0 else -1
        self._recent_directions.append(direction)

    def record_result(
        self,
        target_u: float,
        target_v: float,
    ) -> FocusError | None:
        """Record where the target ended up after movement.

        Call this after movement completes to provide feedback for learning.

        Args:
            target_u: Current horizontal position of target in frame.
            target_v: Current vertical position of target in frame.

        Returns:
            FocusError with computed error metrics, or None if no pending intent.
        """
        if self._pending_sample is None:
            return None  # Silently ignore - this is normal during rapid updates

        sample = self._pending_sample

        # Check if enough time has passed for movement to complete
        elapsed = time.time() - sample.timestamp
        if elapsed < self.config.min_result_delay:
            # Movement probably not finished yet - keep pending sample for later
            return None

        self._pending_sample = None

        # Compute error from center
        error_u = target_u - self._image_center[0]
        error_v = target_v - self._image_center[1]

        sample.actual_du = target_u
        sample.actual_dv = target_v
        sample.error_u = error_u
        sample.error_v = error_v

        self._samples.append(sample)
        self._total_samples += 1

        # Compute focus error for learning
        focus_error = self._compute_error(sample)

        # Check for success
        error_magnitude = (error_u**2 + error_v**2) ** 0.5
        if error_magnitude < self.config.error_threshold:
            self._successful_focuses += 1

        # Check for oscillation
        if self._detect_oscillation():
            self._oscillation_count += 1
            logger.debug(
                "Oscillation detected (count=%d), reducing gains",
                self._oscillation_count,
            )
            # Reduce all gains on oscillation
            reduction = 0.9
            self._gain_h_pos *= reduction
            self._gain_h_neg *= reduction
            self._gain_v_pos *= reduction
            self._gain_v_neg *= reduction
            self._clamp_gains()

        # Learn from this sample
        self._learn_from_sample(sample, focus_error)

        # Report to NAc if available
        if self._nac is not None and self.config.use_nac:
            self._report_to_nac(sample, focus_error)

        # Auto-save periodically
        self._maybe_save()

        return focus_error

    def _compute_error(self, sample: FocusSample) -> FocusError:
        """Compute error metrics from a sample.

        The overshoot ratio tells us how much of the intended movement we achieved:
        - 1.0 = perfect (target is now centered)
        - <1.0 = undershoot (target still offset in same direction)
        - >1.0 = overshoot (target passed center, now on other side)
        """
        # Commanded movement (before gain) - this is the raw offset from center
        cmd_u = sample.commanded_du
        cmd_v = sample.commanded_dv

        # Error from center after movement
        err_u = sample.error_u or 0.0
        err_v = sample.error_v or 0.0

        # Account for the gain that was actually used
        # If we used gain=0.5, we only commanded half the movement
        gain_used = sample.gain_used if sample.gain_used > 0 else 0.5

        overshoot_u = 1.0
        overshoot_v = 1.0

        min_cmd = 20.0  # Minimum command to consider for learning

        if abs(cmd_u) > min_cmd:
            # What we expected: target moves from cmd_u to 0 (centered)
            # With gain applied, we expected: target moves from cmd_u to cmd_u * (1 - gain)
            # Actual: target is at err_u

            # Expected remaining error after a perfect gain-dampened move:
            cmd_u * (1.0 - gain_used)

            # If actual == expected, gain is calibrated correctly
            # If actual > expected (same sign as cmd), we undershot -> need more gain
            # If actual < expected or opposite sign, we overshot -> need less gain

            # Compute what fraction of the intended movement we achieved
            intended_movement = cmd_u * gain_used  # What we intended to move
            actual_movement = cmd_u - err_u  # What we actually moved

            if abs(intended_movement) > 1.0:
                # Overshoot ratio: actual / intended
                # >1 means we moved too much, <1 means we moved too little
                ratio = actual_movement / intended_movement
                # Clamp to reasonable range to avoid extreme values
                overshoot_u = max(0.1, min(3.0, ratio))

        if abs(cmd_v) > min_cmd:
            intended_movement = cmd_v * gain_used
            actual_movement = cmd_v - err_v

            if abs(intended_movement) > 1.0:
                ratio = actual_movement / intended_movement
                overshoot_v = max(0.1, min(3.0, ratio))

        return FocusError(
            overshoot_ratio_u=overshoot_u,
            overshoot_ratio_v=overshoot_v,
            direction_u=1 if cmd_u >= 0 else -1,
            direction_v=1 if cmd_v >= 0 else -1,
            magnitude_u=abs(cmd_u),
            magnitude_v=abs(cmd_v),
        )

    def _learn_from_sample(self, sample: FocusSample, error: FocusError) -> None:
        """Update gains based on observed error using Rescorla-Wagner learning.

        Rescorla-Wagner: ΔV = α(λ - V)
        - V = current gain estimate (prediction)
        - λ = optimal gain for this trial (outcome)
        - α = learning rate

        This provides stable, asymptotic learning that:
        - Converges smoothly without oscillation
        - Learns faster when error is large
        - Self-limits as gain approaches optimal
        """
        lr = self.config.learning_rate
        min_g = self.config.min_gain
        max_g = self.config.max_gain

        # Only learn from significant movements
        min_magnitude = 20.0  # pixels

        # Learn horizontal gain
        if error.magnitude_u > min_magnitude:
            # Compute optimal gain: the gain that WOULD have achieved centering
            # optimal_gain = current_gain / overshoot_ratio
            # - overshoot=1.2 -> we moved 20% too much -> optimal = current/1.2
            # - overshoot=0.8 -> we moved 20% too little -> optimal = current/0.8
            current_gain = self._gain_h_pos if error.direction_u >= 0 else self._gain_h_neg

            optimal_gain = current_gain / error.overshoot_ratio_u
            optimal_gain = max(min_g, min(max_g, optimal_gain))

            # Rescorla-Wagner: prediction error × learning rate
            prediction_error = optimal_gain - current_gain
            delta = lr * prediction_error
            new_gain = current_gain + delta
            new_gain = max(min_g, min(max_g, new_gain))

            if error.direction_u >= 0:
                self._gain_h_pos = new_gain
            else:
                self._gain_h_neg = new_gain

            logger.info(
                "FocusLearner R-W: H overshoot=%.2f, optimal=%.3f, δ=%.3f, gain %.3f->%.3f (err=%.0fpx)",
                error.overshoot_ratio_u,
                optimal_gain,
                prediction_error,
                current_gain,
                new_gain,
                sample.error_u or 0,
            )

        # Learn vertical gain
        if error.magnitude_v > min_magnitude:
            current_gain = self._gain_v_pos if error.direction_v >= 0 else self._gain_v_neg

            optimal_gain = current_gain / error.overshoot_ratio_v
            optimal_gain = max(min_g, min(max_g, optimal_gain))

            prediction_error = optimal_gain - current_gain
            delta = lr * prediction_error
            new_gain = current_gain + delta
            new_gain = max(min_g, min(max_g, new_gain))

            if error.direction_v >= 0:
                self._gain_v_pos = new_gain
            else:
                self._gain_v_neg = new_gain

            logger.info(
                "FocusLearner R-W: V overshoot=%.2f, optimal=%.3f, δ=%.3f, gain %.3f->%.3f (err=%.0fpx)",
                error.overshoot_ratio_v,
                optimal_gain,
                prediction_error,
                current_gain,
                new_gain,
                sample.error_v or 0,
            )

        # Update average gains for backward compatibility
        self._gain_h = (self._gain_h_pos + self._gain_h_neg) / 2
        self._gain_v = (self._gain_v_pos + self._gain_v_neg) / 2

    def _clamp_gains(self) -> None:
        """Clamp all gains to valid range."""
        min_g = self.config.min_gain
        max_g = self.config.max_gain

        self._gain_h_pos = max(min_g, min(max_g, self._gain_h_pos))
        self._gain_h_neg = max(min_g, min(max_g, self._gain_h_neg))
        self._gain_v_pos = max(min_g, min(max_g, self._gain_v_pos))
        self._gain_v_neg = max(min_g, min(max_g, self._gain_v_neg))

    def _apply_decay(self) -> None:
        """Apply time-based decay toward initial gain."""
        now = time.time()
        dt = now - self._last_update_time
        self._last_update_time = now

        if dt > 10.0:
            dt = 10.0  # Clamp to prevent huge jumps

        decay = self.config.gain_decay_rate * dt
        initial = self.config.initial_gain

        # Blend toward initial
        self._gain_h_pos = self._gain_h_pos * (1 - decay) + initial * decay
        self._gain_h_neg = self._gain_h_neg * (1 - decay) + initial * decay
        self._gain_v_pos = self._gain_v_pos * (1 - decay) + initial * decay
        self._gain_v_neg = self._gain_v_neg * (1 - decay) + initial * decay

    def _detect_oscillation(self) -> bool:
        """Detect oscillation from direction history."""
        if len(self._recent_directions) < 4:
            return False

        # Count sign changes
        changes = 0
        prev = self._recent_directions[0]
        for d in list(self._recent_directions)[1:]:
            if d != prev and d != 0 and prev != 0:
                changes += 1
            prev = d

        return changes >= self.config.oscillation_threshold

    def _report_to_nac(self, sample: FocusSample, error: FocusError) -> None:
        """Report focus outcome to NAc for pattern learning."""
        if self._nac is None:
            return

        from maxim.decisions.causal_link import Valence

        # Create signature based on movement magnitude
        magnitude = (error.magnitude_u + error.magnitude_v) / 2
        if magnitude < 50:
            mag_bucket = "small"
        elif magnitude < 150:
            mag_bucket = "medium"
        else:
            mag_bucket = "large"

        action_sig = f"focus_move:{mag_bucket}"

        # Determine valence based on error
        error_magnitude = ((sample.error_u or 0) ** 2 + (sample.error_v or 0) ** 2) ** 0.5

        if error_magnitude < self.config.error_threshold:
            valence = Valence.POSITIVE
        elif error_magnitude < self.config.error_threshold * 2:
            valence = Valence.NEUTRAL
        else:
            valence = Valence.NEGATIVE

        # Record the observation
        self._nac.observe(
            event_type="focus",
            event_signature=action_sig,
            outcome_type="focus_result",
            outcome_signature=f"error:{error_magnitude:.0f}px",
            outcome_valence=valence,
            delta_seconds=time.time() - sample.timestamp,
            context={
                "magnitude_u": error.magnitude_u,
                "magnitude_v": error.magnitude_v,
                "overshoot_u": error.overshoot_ratio_u,
                "overshoot_v": error.overshoot_ratio_v,
                "gain_used": sample.gain_used,
            },
        )

    def get_stats(self) -> dict[str, Any]:
        """Get learning statistics."""
        return {
            "gain_h_pos": round(self._gain_h_pos, 3),
            "gain_h_neg": round(self._gain_h_neg, 3),
            "gain_v_pos": round(self._gain_v_pos, 3),
            "gain_v_neg": round(self._gain_v_neg, 3),
            "average_gain": round(self.get_average_gain(), 3),
            "total_samples": self._total_samples,
            "successful_focuses": self._successful_focuses,
            "success_rate": round(self._successful_focuses / max(1, self._total_samples), 3),
            "oscillation_count": self._oscillation_count,
        }

    def reset(self) -> None:
        """Reset learning state."""
        initial = self.config.initial_gain
        self._gain_h = initial
        self._gain_v = initial
        self._gain_h_pos = initial
        self._gain_h_neg = initial
        self._gain_v_pos = initial
        self._gain_v_neg = initial
        self._samples.clear()
        self._pending_sample = None
        self._recent_directions.clear()
        self._total_samples = 0
        self._successful_focuses = 0
        self._oscillation_count = 0

    def _maybe_save(self) -> None:
        """Auto-save if enough time has passed since last save."""
        if not self.config.persist_path or self.config.auto_save_interval <= 0:
            return

        now = time.time()
        if now - self._last_save_time >= self.config.auto_save_interval:
            try:
                self.save(self.config.persist_path)
                self._last_save_time = now
            except Exception as e:
                logger.debug("Auto-save failed: %s", e)

    def save(self, path: str | Path) -> None:
        """Save learned gains to a JSON file.

        Args:
            path: File path to save to.
        """
        state = {
            "version": 1,
            "gains": {
                "h_pos": self._gain_h_pos,
                "h_neg": self._gain_h_neg,
                "v_pos": self._gain_v_pos,
                "v_neg": self._gain_v_neg,
            },
            "stats": {
                "total_samples": self._total_samples,
                "successful_focuses": self._successful_focuses,
                "oscillation_count": self._oscillation_count,
            },
        }

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(state, f, indent=2)

        logger.info(
            "FocusLearner saved to %s (gains: h+%.3f/h-%.3f, v+%.3f/v-%.3f)",
            path,
            self._gain_h_pos,
            self._gain_h_neg,
            self._gain_v_pos,
            self._gain_v_neg,
        )

    def load(self, path: str | Path) -> bool:
        """Load learned gains from a JSON file.

        Args:
            path: File path to load from.

        Returns:
            True if loaded successfully, False otherwise.
        """
        path = Path(path)
        if not path.exists():
            logger.debug("FocusLearner state file not found: %s", path)
            return False

        try:
            with open(path) as f:
                state = json.load(f)

            # Load gains
            gains = state.get("gains", {})
            self._gain_h_pos = gains.get("h_pos", self.config.initial_gain)
            self._gain_h_neg = gains.get("h_neg", self.config.initial_gain)
            self._gain_v_pos = gains.get("v_pos", self.config.initial_gain)
            self._gain_v_neg = gains.get("v_neg", self.config.initial_gain)

            # Update derived values
            self._gain_h = (self._gain_h_pos + self._gain_h_neg) / 2
            self._gain_v = (self._gain_v_pos + self._gain_v_neg) / 2

            # Load stats
            stats = state.get("stats", {})
            self._total_samples = stats.get("total_samples", 0)
            self._successful_focuses = stats.get("successful_focuses", 0)
            self._oscillation_count = stats.get("oscillation_count", 0)

            # Clamp loaded gains to valid range
            self._clamp_gains()

            logger.info(
                "FocusLearner loaded from %s (gains: h+%.3f/h-%.3f, v+%.3f/v-%.3f, samples=%d)",
                path,
                self._gain_h_pos,
                self._gain_h_neg,
                self._gain_v_pos,
                self._gain_v_neg,
                self._total_samples,
            )
            return True

        except Exception as e:
            logger.warning("Failed to load FocusLearner state from %s: %s", path, e)
            return False


__all__ = ["FocusLearner", "FocusLearnerConfig", "FocusSample", "FocusError"]
