"""Robot controller abstraction.

The :class:`RobotController` ABC defines the interface for controlling
any robot hardware. Implementations wrap specific SDKs (Reachy, Atlas,
Spot, simulation, etc.) and live in their own subpackages under
:mod:`maxim.hardware`. The agent loop, default network, and decision
engine consume this abstract interface — they have no SDK assumptions.

Adding a new robot is a 3-step process:

1. Implement :class:`RobotController` for your robot's SDK in a new
   subpackage (e.g. ``maxim/hardware/atlas/controller.py``).
2. Register it via the ``maxim.robots`` entry-point group in
   ``pyproject.toml`` so :class:`RobotRegistry` auto-discovers it::

       [project.entry-points."maxim.robots"]
       atlas = "maxim_atlas.controller:AtlasController"

3. (Optional) Ship robot-specific tools in ``maxim/tools/atlas.py``.

Robot-specific joints (Reachy antennas, Atlas arms, Spot legs) are
expressed via the generic :attr:`MotionTarget.extras` field — no
subclassing required. See ``hardware/reachy/controller.py`` for the
canonical reference implementation.
"""

from __future__ import annotations

import logging
import math
import numbers
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from maxim.exceptions import MotionRefusalKind, MotionRefused
from maxim.hardware.capabilities import (
    RobotCapabilities,
    RobotConnectionState,
    RobotState,
)
from maxim.hardware.streams import AudioStream, VideoStream

if TYPE_CHECKING:
    from maxim.embodiment.audio_localization import DoAReader

logger = logging.getLogger(__name__)

__all__ = [
    "MOTION_DURATION_MAX_S",
    "MOTION_DURATION_MIN_S",
    "DurationPolicy",
    "MotionOutcome",
    "HEAD_TRANSLATION_EXTRAS",
    "MotionOutcomeKind",
    "MotionRefused",
    "MotionTarget",
    "PixelTarget",
    "RobotController",
    "as_outcome",
    "clamp_finite",
    "floor_duration",
    "normalize_method",
    "require_duration",
    "require_finite",
]

# ── Motion validation (#1496): no non-finite, out-of-range-duration or instant
# command reaches a motor. One copy, used at entry (MotionTarget / PixelTarget,
# Selfy, the tools), at every clamp, and at the last mile before each SDK call.
# docs/plans/motion_finite_1496.md.

MOTION_DURATION_MIN_S = 0.1
MOTION_DURATION_MAX_S = 10.0

# Canonical interpolation names. ``cartoon`` is refused: it overshoots its
# target by 10%, past any clamp. Aliases keep their historical meanings.
_METHOD_ALIASES = {
    "minimum_jerk": "minimum_jerk",
    "minjerk": "minimum_jerk",
    "min_jerk": "minimum_jerk",
    "linear": "linear",
    "ease_in_out": "ease_in_out",
    "ease": "ease_in_out",
}


def require_finite(field_name: str, value: Any) -> float:
    """Return ``value`` as a plain finite float, or raise ``MotionRefused(kind="non_finite")``.

    Rejects bools and strings (internal entries carry numbers; the tool boundary
    converts LLM strings itself). The result is a plain ``float``, so Decimal and
    numpy scalars do not survive.
    """
    if isinstance(value, (bool, str, np.bool_)):
        raise MotionRefused("invalid", f"{field_name} must be a number, got {value!r}")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as e:
        raise MotionRefused("invalid", f"{field_name} is not a number: {value!r}") from e
    if not math.isfinite(number):
        raise MotionRefused("non_finite", f"{field_name} is not finite: {value!r}")
    return number


def require_duration(value: Any) -> float:
    """An explicit duration: finite and within [0.1 s, 10 s] inclusive, else refused."""
    duration = require_finite("duration", value)
    if not MOTION_DURATION_MIN_S <= duration <= MOTION_DURATION_MAX_S:
        raise MotionRefused(
            "duration",
            f"duration {duration!r} s is outside [{MOTION_DURATION_MIN_S}, {MOTION_DURATION_MAX_S}] s",
        )
    return duration


def floor_duration(value: Any) -> float:
    """A default or derived duration: refused if non-finite, otherwise lifted into range.

    Only for durations the caller did not choose explicitly (decision 5).
    """
    duration = require_finite("duration", value)
    return _bounded(duration, MOTION_DURATION_MIN_S, MOTION_DURATION_MAX_S)


def clamp_finite(field_name: str, value: Any, lo: float, hi: float) -> float:
    """Refuse a non-finite value (or bound), then clamp. For COMMAND clamps only.

    Replaces the ``max(-L, min(L, x))`` idiom, which turns NaN into a finite
    limit. Measurement clamps discard a non-finite value instead of refusing.
    """
    lo_f = require_finite(f"{field_name} lower bound", lo)
    hi_f = require_finite(f"{field_name} upper bound", hi)
    return _bounded(require_finite(field_name, value), lo_f, hi_f)


def _bounded(value: float, lo: float, hi: float) -> float:
    """``value`` limited to [lo, hi]; callers have already proven all three finite."""
    if value < lo:
        return lo
    return hi if value > hi else value


def normalize_method(method: Any) -> str:
    """Canonical interpolation name, or ``MotionRefused(kind="method")``."""
    key = method.strip().lower().replace("-", "_") if isinstance(method, str) else None
    canonical = _METHOD_ALIASES.get(key) if key is not None else None
    if canonical is None:
        raise MotionRefused("method", f"interpolation method {method!r} is not one of {sorted(set(_METHOD_ALIASES))}")
    return canonical


# How a requested duration is treated (#1499 R4, O1). ``"exact"``: the caller
# chose it (a model-supplied duration), so a move that cannot be done safely in
# that time is refused (``rate``/``path``). ``"at_least"``: a default or derived
# duration, stretched to the rate cap and split along the path as needed.
DurationPolicy = Literal["exact", "at_least"]

MotionOutcomeKind = MotionRefusalKind | Literal["superseded", "cancelled", "partial", "error"]

# ``MotionTarget.extras`` keys for a head translation, body frame, metres: written by
# ``motion/movement.py::move_head`` when it delegates, read by ``ReachyMiniController`` (#1499 5.4).
HEAD_TRANSLATION_EXTRAS = ("head_x", "head_y", "head_z")


@dataclass(frozen=True)
class MotionOutcome:
    """What became of one motion command (#1499 O4, O7). Truthy iff ``ok``.

    Never persisted and never sent over a wire: it lives for one dispatch and
    its callers (the tools, the motor worker, DN), so it carries no
    forward-compat path (CC3 scope: persisted or wire-crossing only).

    ``kind`` is ``None`` when ``ok``. Otherwise it is a refusal kind
    (``MotionRefusalKind``), or ``"cancelled"`` (aborted, or the motor was
    busy), ``"partial"`` (a split stopped after ``chunks_done`` chunks),
    ``"superseded"`` (replaced before it ran) or ``"error"`` (the controller
    failed, or a plugin returned ``False``). No non-ok outcome feeds the gaze or
    workspace-bounds learners (#1499 O4). ``"superseded"`` has no producer until
    PR 2's motor queue.
    """

    ok: bool
    kind: MotionOutcomeKind | None = None
    reason: str = ""
    duration_s: float | None = None
    chunks_planned: int = 0
    chunks_done: int = 0

    def __bool__(self) -> bool:
        return self.ok

    @classmethod
    def refused(cls, kind: MotionOutcomeKind, reason: str, *, chunks_planned: int = 0) -> MotionOutcome:
        return cls(ok=False, kind=kind, reason=reason, chunks_planned=chunks_planned)


def as_outcome(result: bool | MotionOutcome | None) -> MotionOutcome:
    """A controller result as a ``MotionOutcome`` (#1499 O7).

    Plugins and older fakes return ``bool``; ``True`` is ok with no reason,
    ``False`` is ``kind="error"``. Every consumer that reads a field goes
    through this.
    """
    if isinstance(result, MotionOutcome):
        return result
    if result:
        return MotionOutcome(ok=True)
    if result is None:
        return MotionOutcome(ok=False, kind="error", reason="the controller returned no outcome")
    return MotionOutcome(ok=False, kind="error", reason="the controller returned False")


@dataclass
class MotionTarget:
    """Target pose for robot motion.

    All angles are in radians. ``None`` values mean "don't change".

    Robot-specific joints (Reachy antennas, Atlas arms, Spot legs) go
    in the generic :attr:`extras` dict so the canonical motion command
    stays SDK-agnostic. A controller implementation reads its own keys
    out of ``extras`` and ignores keys it doesn't recognize. Example::

        # Reachy antenna positions
        target = MotionTarget(
            head_pitch=0.2,
            extras={"antenna_left": 0.5, "antenna_right": -0.5},
        )

        # Future Atlas arm posture
        target = MotionTarget(
            extras={"left_arm_yaw": 1.0, "right_arm_pitch": -0.3},
        )
    """

    # Head pose (3-DOF) — common to humanoid robots.
    #
    # FRAME (explicit since 2026-07-16 — it was ambiguous before, and ambiguity
    # about frames is what cost us a full session; see CLAUDE.md's Reachy
    # head-frame invariant): head_roll/pitch/yaw are **BODY-RELATIVE** — "turn my
    # head 30 deg left" means 30 deg left OF THE BODY, which is what a caller
    # asking a robot to turn its head means. Controllers are responsible for
    # converting into whatever frame their SDK wants (the Reachy SDK's head pose
    # is WORLD-referenced, so ReachyMiniController adds the body yaw). Before this
    # was pinned, `head_yaw=0.5` with the body at 0.9 rad put the head 0.4 rad to
    # the RIGHT of the body — the opposite of the request.
    head_roll: float | None = None
    head_pitch: float | None = None
    head_yaw: float | None = None

    # Body rotation — common to most robots
    body_yaw: float | None = None

    # Robot-specific joints — keys are controller-defined.
    # Examples: ``"antenna_left"`` / ``"antenna_right"`` (Reachy),
    # ``"left_arm_yaw"`` / ``"gripper_pos"`` (Atlas).
    extras: dict[str, float] = field(default_factory=dict)

    # Motion duration in seconds
    duration: float = 1.0

    # Motion method (e.g., "linear", "minimum_jerk"); normalised to the canonical name.
    method: str = "minimum_jerk"

    # "exact" when the caller chose ``duration`` (refuse what cannot be done in
    # it); "at_least" for a default or derived one (stretch and split) (#1499).
    duration_policy: DurationPolicy = "at_least"

    def __post_init__(self) -> None:
        # #1496: refuse at construction. The dataclass is mutable, so the
        # controller re-checks every field at the last mile as well.
        for name in ("head_roll", "head_pitch", "head_yaw", "body_yaw"):
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, require_finite(name, value))
        self.extras = {key: require_finite(f"extras[{key!r}]", val) for key, val in self.extras.items()}
        self.duration = require_duration(self.duration)
        self.method = normalize_method(self.method)
        self.duration_policy = _require_policy(self.duration_policy)


@dataclass(slots=True)
class PixelTarget:
    """Target pixel coordinates for look_at operations."""

    u: int  # X coordinate (0 = left)
    v: int  # Y coordinate (0 = top)
    duration: float = 0.5
    duration_policy: DurationPolicy = "at_least"  # see MotionTarget.duration_policy

    def __post_init__(self) -> None:
        self.u = _require_pixel("u", self.u)
        self.v = _require_pixel("v", self.v)
        self.duration = require_duration(self.duration)
        self.duration_policy = _require_policy(self.duration_policy)


def _require_policy(value: Any) -> DurationPolicy:
    if value == "exact":
        return "exact"
    if value == "at_least":
        return "at_least"
    raise MotionRefused("invalid", f"duration_policy must be 'exact' or 'at_least', got {value!r}")


def _require_pixel(field_name: str, value: Any) -> int:
    """A finite integral pixel coordinate (an int, or a float with no fraction)."""
    if isinstance(value, numbers.Integral) and not isinstance(value, bool):
        return int(value)
    number = require_finite(field_name, value)
    if not number.is_integer():
        raise MotionRefused("invalid", f"{field_name} must be a whole pixel, got {value!r}")
    return int(number)


class RobotController(ABC):
    """Abstract base class for robot controllers.

    Implementations wrap specific robot SDKs and provide a
    unified interface for:
    - Connection management
    - Motion control
    - Lifecycle (wake/sleep)
    - Data stream access

    The controller is responsible for maintaining connection
    state and providing access to video/audio streams.

    Usage:
        controller = ReachyMiniController(robot_name="reachy_mini")
        if controller.connect(timeout=30.0):
            controller.wake_up()
            video = controller.get_video_stream()
            frame = video.get_frame()
            controller.goto_target(MotionTarget(head_pitch=0.2))
            controller.goto_sleep()
            controller.disconnect()
    """

    def __init__(self, robot_id: str) -> None:
        """Initialize controller.

        Args:
            robot_id: Unique identifier for this robot instance.
        """
        self._robot_id = robot_id
        self._state = RobotState()
        self._capabilities: RobotCapabilities | None = None

    @property
    def robot_id(self) -> str:
        """Get the robot's unique identifier."""
        return self._robot_id

    @property
    @abstractmethod
    def robot_type(self) -> str:
        """Get the robot type identifier (e.g., 'reachy_mini', 'simulated')."""
        ...

    @property
    def capabilities(self) -> RobotCapabilities | None:
        """Get the robot's capabilities (None if not connected)."""
        return self._capabilities

    @property
    def state(self) -> RobotState:
        """Get the robot's current state."""
        return self._state

    def _update_state(self, **kwargs: Any) -> None:
        """Update state fields."""
        for key, value in kwargs.items():
            if hasattr(self._state, key):
                setattr(self._state, key, value)

    # ─────────────────────────────────────────────────────────────────────────
    # Connection Management
    # ─────────────────────────────────────────────────────────────────────────

    @abstractmethod
    def connect(self, timeout: float = 30.0) -> bool:
        """Connect to the robot.

        Args:
            timeout: Connection timeout in seconds.

        Returns:
            True if connection successful, False otherwise.
        """
        ...

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the robot."""
        ...

    def is_connected(self) -> bool:
        """Check if robot is connected."""
        return self._state.is_connected

    def reconnect(self, timeout: float = 30.0, max_attempts: int = 3) -> bool:
        """Attempt to reconnect to the robot.

        Args:
            timeout: Connection timeout per attempt.
            max_attempts: Maximum number of reconnection attempts.

        Returns:
            True if reconnection successful.
        """
        self._update_state(connection_state=RobotConnectionState.RECONNECTING)

        for attempt in range(max_attempts):
            self._update_state(connection_attempts=attempt + 1)
            logger.info(
                "Reconnection attempt %d/%d for %s",
                attempt + 1,
                max_attempts,
                self._robot_id,
            )

            self.disconnect()
            if self.connect(timeout=timeout):
                return True

        self._update_state(
            connection_state=RobotConnectionState.ERROR,
            error_message=f"Failed to reconnect after {max_attempts} attempts",
        )
        return False

    # ─────────────────────────────────────────────────────────────────────────
    # Motion Control
    # ─────────────────────────────────────────────────────────────────────────

    @abstractmethod
    def goto_target(self, target: MotionTarget) -> bool | MotionOutcome:
        """Move robot to target pose.

        Args:
            target: Target pose specification.

        Returns:
            A ``MotionOutcome`` (preferred; truthy iff ok) or a ``bool``, which
            stays accepted for 1.x plugins. Consumers read it through
            ``as_outcome`` (#1499 O7).
        """
        ...

    @abstractmethod
    def look_at_pixel(self, target: PixelTarget) -> bool | MotionOutcome:
        """Move head to look at a pixel coordinate.

        Uses the robot's camera model to convert pixel
        coordinates to head angles.

        Args:
            target: Target pixel coordinates.

        Returns:
            True if motion command accepted.
        """
        ...

    @abstractmethod
    def get_current_pose(self) -> dict[str, float]:
        """Get current joint positions.

        Returns:
            Dict mapping joint names to current angles (radians).
        """
        ...

    def center_vision(self, duration: float = 0.5) -> bool | MotionOutcome:
        """Center the head to look straight ahead.

        Returns:
            True if motion command accepted; False when it was refused, like
            ``goto_target`` (a bad duration is logged, not raised; #1496).
        """
        try:
            target = MotionTarget(head_roll=0.0, head_pitch=0.0, head_yaw=0.0, body_yaw=0.0, duration=duration)
        except MotionRefused as e:
            logger.error("center_vision refused: %s", e)
            return False
        return self.goto_target(target)

    # ─────────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────────────────────────────────────

    @abstractmethod
    def wake_up(self) -> bool | MotionOutcome:
        """Wake up the robot (enable motors, start streaming).

        Returns:
            True if wake-up successful.
        """
        ...

    @abstractmethod
    def goto_sleep(self) -> bool | MotionOutcome:
        """Put the robot to sleep (disable motors, stop streaming).

        Returns:
            True if sleep command successful.
        """
        ...

    @abstractmethod
    def start_recording(self) -> bool:
        """Start media recording/streaming.

        Returns:
            True if recording started successfully.
        """
        ...

    @abstractmethod
    def stop_recording(self) -> bool:
        """Stop media recording/streaming.

        Returns:
            True if recording stopped successfully.
        """
        ...

    # ─────────────────────────────────────────────────────────────────────────
    # Data Streams
    # ─────────────────────────────────────────────────────────────────────────

    @abstractmethod
    def get_video_stream(self) -> VideoStream | None:
        """Get the video stream for this robot.

        Returns:
            VideoStream implementation, or None if not available.
        """
        ...

    @abstractmethod
    def get_audio_stream(self) -> AudioStream | None:
        """Get the audio stream for this robot.

        Returns:
            AudioStream implementation, or None if not available.
        """
        ...

    def get_doa_reader(self) -> DoAReader | None:
        """Get a Direction-of-Arrival reader for this robot, if it has one.

        Optional capability seam (live_audio_orient_wiring.md Stage 1a):
        a robot with sound-localization hardware returns a
        :data:`maxim.embodiment.audio_localization.DoAReader` — a zero-arg
        callable yielding ``(doa_radians, is_speech_detected) | None`` —
        which the runtime's audio-orient feed polls. ``None`` (the default)
        means the capability is absent; the caller gets no audio-orient
        wiring, exactly as a robot without a mic array would.

        Deliberately NON-abstract: the ABC is contract-frozen at 12 abstract
        methods (a 13th breaks every third-party ``maxim.robots`` plugin).
        Default-``None`` is the established convenience shape
        (``center_vision``, ``reconnect``). Call sites probe via
        ``getattr(controller, "get_doa_reader", None)`` so plugins compiled
        against an older ABC keep working. Implementations advertise the
        capability via ``RobotCapabilities.custom`` (``"audio_localization"``).
        NOT a StreamCapability/SensorStream: DoA is a stateless pull, not a
        stream.
        """
        return None

    # ─────────────────────────────────────────────────────────────────────────
    # Context Manager
    # ─────────────────────────────────────────────────────────────────────────

    def __enter__(self) -> RobotController:
        """Context manager entry - connect and wake up."""
        if self.connect():
            self.wake_up()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit - sleep and disconnect."""
        try:
            self.goto_sleep()
        finally:
            self.disconnect()

    # ─────────────────────────────────────────────────────────────────────────
    # String Representation
    # ─────────────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"robot_id={self._robot_id!r}, "
            f"robot_type={self.robot_type!r}, "
            f"connected={self.is_connected()})"
        )
