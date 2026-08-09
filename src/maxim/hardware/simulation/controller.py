"""Simulated robot controller for testing.

Provides a fully functional RobotController that simulates
robot behavior for testing without hardware.
"""

from __future__ import annotations

import dataclasses
import logging
import math
import random
import time

from maxim.hardware.capabilities import (
    SIMULATED_CAPABILITIES,
    RobotConnectionState,
)
from maxim.hardware.controller import MotionTarget, PixelTarget, RobotController
from maxim.hardware.simulation.streams import MockAudioStream, MockVideoStream
from maxim.hardware.streams import AudioStream, VideoStream

logger = logging.getLogger(__name__)

# Structured trace channel for the DoA scenario (Exp 49). Events ride the
# standard logging pipeline so MAXIM_LOG_FILE's StructuredFormatter
# serializes them (`event` + `data` extras — the pain_chain pattern);
# plain messages would be invisible in the JSONL.
_scenario_log = logging.getLogger("maxim.sim_doa")


def _normalize_angle(theta: float) -> float:
    """Normalize an angle to ``(-pi, pi]``."""
    while theta > math.pi:
        theta -= 2.0 * math.pi
    while theta <= -math.pi:
        theta += 2.0 * math.pi
    return theta


class SimulatedDoAScenario:
    """Honest-physics DoA source at a FIXED world bearing (Exp 49 harness).

    Produces ``(doa_radians, is_speech)`` readings the way the real
    XVF3800 + linear mic array would, for a sound source pinned at one
    world bearing for the whole trial (it never teleports — the
    fixed-source discipline from the Exp 45 sim-vs-hardware lesson):

    1. The head-relative offset is computed from the controller's CURRENT
       world head yaw (the mics live in the HEAD — the pose the scenario
       reads is the pose the controller's own motion commands produced,
       including the body ride-along).
    2. The LINEAR-ARRAY FOLD is applied honestly: a source behind the
       head reads as its front mirror (same left/right side, ``sin``
       preserved) — front/back is not recoverable from this hardware.
    3. Gaussian noise (sigma in azimuth units; live characterization
       ~0.03) is added, then the value is INVERTED through the exact
       inverse of :func:`maxim.embodiment.audio_localization.doa_to_azimuth`
       (``doa = pi/2 + az * pi/2``) so the PRODUCTION mapping code runs on
       the consumer side — the sim never bypasses the mapping under test.
    4. ``speech_density`` gates ``is_speech`` per read (1.0 = dense —
       every read localizable; the Exp 49 main arms deliberately remove
       speech sparsity as a confound).

    Sign conventions (the load-bearing part — see the head-frame
    invariant): pose yaw is radians with **+yaw = LEFT**; azimuth is
    **-1 = left, +1 = right**; ``bearing_deg`` is the source's world
    bearing in the pose-yaw convention (**positive = LEFT** of world
    forward). A source 90° left of the head therefore reads az = -1.0
    (DoA 0), matching the XVF3800 convention (0 = left, pi/2 = front,
    pi = right).

    Thread note: ``read()`` runs on the DoA feed thread while motion
    commands mutate the pose on the executor thread. Sim motion is
    instantaneous single-key assignments; a read interleaving a motion
    can at worst produce one stale sample, which the feed's median-of-k
    absorbs — same class of transient the real chip has mid-rotation.
    """

    def __init__(
        self,
        controller: "SimulatedController",
        *,
        bearing_deg: float,
        noise_sigma: float = 0.03,
        speech_density: float = 1.0,
        seed: int | None = None,
    ) -> None:
        self._controller = controller
        self._bearing_rad = math.radians(float(bearing_deg))
        self._noise_sigma = max(0.0, float(noise_sigma))
        self._speech_density = max(0.0, min(1.0, float(speech_density)))
        self._rng = random.Random(seed)

    def read(self) -> "tuple[float, bool] | None":
        """``DoAReader``-shaped: ``(doa_radians, is_speech)``, never None
        while the controller is connected (the chip always has an angle
        estimate; the speech flag is the localizability gate)."""
        pose = self._controller.get_current_pose()
        head_world = float(pose.get("yaw", 0.0) or 0.0)
        body = float(pose.get("body_yaw", 0.0) or 0.0)
        # Head-relative offset, + = source to the LEFT of head forward.
        theta = _normalize_angle(self._bearing_rad - head_world)
        # Linear-array fold: reflect across the interaural (left-right)
        # axis — same side, front/back collapsed.
        theta_folded = theta
        if theta > math.pi / 2.0:
            theta_folded = math.pi - theta
        elif theta < -math.pi / 2.0:
            theta_folded = -math.pi - theta
        # +theta (left) → negative azimuth (-1 = left).
        az_true = -theta_folded / (math.pi / 2.0)
        az_read = az_true
        if self._noise_sigma > 0.0:
            az_read += self._rng.gauss(0.0, self._noise_sigma)
        az_read = max(-1.0, min(1.0, az_read))
        # Exact inverse of doa_to_azimuth — the production mapping
        # reconstructs az_read on the consumer side.
        doa = math.pi / 2.0 + az_read * (math.pi / 2.0)
        is_speech = self._rng.random() < self._speech_density
        _scenario_log.debug(
            "sim_doa.read",
            extra={
                "event": "sim_doa.read",
                "data": {
                    "bearing_deg": round(math.degrees(self._bearing_rad), 2),
                    "head_world_deg": round(math.degrees(head_world), 2),
                    "body_yaw_deg": round(math.degrees(body), 2),
                    # UNFOLDED head-relative offset — the honest centering
                    # criterion (the folded az reads ~0 when facing exactly
                    # AWAY from the source; a harness gating on folded az
                    # would score a 180°-wrong head as centered).
                    "theta_deg": round(math.degrees(theta), 2),
                    "az_true": round(az_true, 4),
                    "az_read": round(az_read, 4),
                    "speech": is_speech,
                },
            },
        )
        return (doa, is_speech)


class SimulatedController(RobotController):
    """Simulated robot controller for testing.

    Provides a complete RobotController implementation that
    simulates robot behavior without requiring hardware.
    Useful for:
    - Unit testing
    - Development without hardware
    - CI/CD pipelines
    - Demonstration

    All operations succeed instantly (simulated).
    Motion commands update the internal pose state.
    Streams provide synthetic data.
    """

    def __init__(
        self,
        robot_id: str | None = None,
        *,
        video_resolution: tuple[int, int] = (640, 480),
        video_fps: float = 10.0,
        audio_sample_rate: int = 16000,
        simulate_delays: bool = False,
        doa_reader=None,
        doa_source_bearing_deg: float | None = None,
        doa_noise_sigma: float = 0.03,
        doa_speech_density: float = 1.0,
        doa_seed: int | None = None,
        head_yaw_limit_deg: float | None = None,
        body_yaw_limit_deg: float | None = None,
        video_enabled: bool = True,
        audio_enabled: bool = True,
    ) -> None:
        """Initialize simulated controller.

        Args:
            robot_id: Unique identifier (defaults to "simulated").
            video_resolution: Resolution for mock video stream.
            video_fps: FPS for mock video stream.
            audio_sample_rate: Sample rate for mock audio stream.
            simulate_delays: If True, add realistic delays to operations.
            doa_reader: Optional scripted DoA reader (a zero-arg callable
                yielding ``(doa_radians, is_speech) | None``). Lets tests
                exercise the ``get_doa_reader`` capability seam end-to-end
                without hardware (live_audio_orient_wiring.md Stage 1).
                ``None`` (default) = capability absent, matching a robot
                without sound localization. An injected reader WINS over
                the scenario config below.
            doa_source_bearing_deg: When set, build a
                :class:`SimulatedDoAScenario` — an honest-physics DoA
                source at this FIXED world bearing (degrees, positive =
                LEFT, the pose-yaw convention), served via
                ``get_doa_reader``. All four ``doa_*`` keys are plain
                constructor kwargs so ``robots.yaml``'s ``config:`` dict
                declares a scenario declaratively (the registry's
                signature filter threads them through). ``None``
                (default) = no scenario, byte-identical to before.
            doa_noise_sigma: Gaussian noise sigma in azimuth units
                (live XVF3800 characterization ~0.03).
            doa_speech_density: Probability per read that ``is_speech``
                is True (1.0 = dense speech, the Exp 49 main arms).
            doa_seed: RNG seed for noise + speech gating (per-trial
                determinism of the draw sequence).
            head_yaw_limit_deg: When set, clamp the head's BODY-RELATIVE
                yaw to ±this on every motion command — the sim analog of
                the daemon's neck envelope (Exp 49 uses the measured
                ~±22°, not the optimistic ±45). ``None`` = unclamped
                (legacy behavior).
            body_yaw_limit_deg: When set, clamp body yaw to ±this
                (the real daemon clamps at ±160°). ``None`` = unclamped.
            video_enabled: When False, ``connect()`` creates no video
                stream and ``get_video_stream()`` returns ``None`` — the
                sim analog of a robot without a camera (keeps DN idle
                visual exploration honest during audio-only scenarios).
            audio_enabled: Same for the mock audio stream. (The DoA
                scenario is independent of this — the XVF3800 DoA value
                is served even when no raw audio stream is consumed.)
        """
        super().__init__(robot_id or "simulated")
        self._video_resolution = video_resolution
        self._video_fps = video_fps
        self._audio_sample_rate = audio_sample_rate
        self._simulate_delays = simulate_delays
        self._video_enabled = bool(video_enabled)
        self._audio_enabled = bool(audio_enabled)
        self._doa_scenario: SimulatedDoAScenario | None = None
        if doa_reader is None and doa_source_bearing_deg is not None:
            self._doa_scenario = SimulatedDoAScenario(
                self,
                bearing_deg=float(doa_source_bearing_deg),
                noise_sigma=doa_noise_sigma,
                speech_density=doa_speech_density,
                seed=doa_seed,
            )
            doa_reader = self._doa_scenario.read
        self._doa_reader = doa_reader
        # Sim-hardware parity (2026-08-07 safety fold): the real controller
        # now clamps unconditionally, so the sim's DEFAULT must match it —
        # a sim accepting poses the hardware would alter is silent
        # sim/hardware divergence in the exact axis the motor-destruction
        # incident lived in. Explicit tighter limits (Exp 49's measured
        # ~±22°) still win; None now means "the real controller's limit",
        # not "unclamped".
        from maxim.hardware.reachy.controller import ReachyMiniController as _Real

        self._head_yaw_limit_rad: float = (
            math.radians(abs(float(head_yaw_limit_deg)))
            if head_yaw_limit_deg is not None
            else _Real._MAX_HEAD_REL_YAW_RAD
        )
        self._body_yaw_limit_rad: float = (
            math.radians(abs(float(body_yaw_limit_deg))) if body_yaw_limit_deg is not None else _Real._MAX_BODY_YAW_RAD
        )
        self._head_roll_limit_rad: float = _Real._MAX_HEAD_ROLL_RAD
        self._head_pitch_limit_rad: float = _Real._MAX_HEAD_PITCH_RAD

        self._video_stream: MockVideoStream | None = None
        self._audio_stream: MockAudioStream | None = None

        # Simulated pose state
        self._current_pose: dict[str, float] = {
            "roll": 0.0,
            "pitch": 0.0,
            "yaw": 0.0,
            "body_yaw": 0.0,
        }

    @property
    def robot_type(self) -> str:
        """Get the robot type identifier."""
        return "simulated"

    # ─────────────────────────────────────────────────────────────────────────
    # Connection Management
    # ─────────────────────────────────────────────────────────────────────────

    def connect(self, timeout: float = 30.0) -> bool:
        """Connect to the simulated robot (always succeeds).

        Args:
            timeout: Ignored in simulation.

        Returns:
            True.
        """
        self._update_state(connection_state=RobotConnectionState.CONNECTING)

        if self._simulate_delays:
            time.sleep(0.1)

        # Create stream wrappers (a disabled stream stays None — the sim
        # analog of an absent device; get_*_stream() returning None is the
        # positive evidence derive_media_capabilities downgrades on).
        if self._video_enabled:
            self._video_stream = MockVideoStream(
                resolution=self._video_resolution,
                fps=self._video_fps,
            )
        if self._audio_enabled:
            self._audio_stream = MockAudioStream(
                input_sample_rate=self._audio_sample_rate,
                output_sample_rate=self._audio_sample_rate,
            )

        # Build capabilities
        self._capabilities = dataclasses.replace(
            SIMULATED_CAPABILITIES,
            robot_id=self._robot_id,
            video_resolution=self._video_resolution,
            audio_input_rate=self._audio_sample_rate,
            audio_output_rate=self._audio_sample_rate,
        )

        self._update_state(
            connection_state=RobotConnectionState.CONNECTED,
            last_heartbeat=time.time(),
        )

        logger.info("Simulated robot connected: %s", self._robot_id)
        return True

    def disconnect(self) -> None:
        """Disconnect from the simulated robot."""
        if self._video_stream:
            self._video_stream.stop()
        if self._audio_stream:
            self._audio_stream.stop()

        self._video_stream = None
        self._audio_stream = None

        self._update_state(
            connection_state=RobotConnectionState.DISCONNECTED,
            is_awake=False,
            is_recording=False,
        )

        logger.info("Simulated robot disconnected: %s", self._robot_id)

    # ─────────────────────────────────────────────────────────────────────────
    # Motion Control
    # ─────────────────────────────────────────────────────────────────────────

    def goto_target(self, target: MotionTarget) -> bool:
        """Move robot to target pose (simulated).

        Updates internal pose state to reflect the motion.

        Args:
            target: Target pose specification.

        Returns:
            True.
        """
        if not self.is_connected():
            return False

        # RETAINED-AXES TRIPWIRE (F1 fold, 2026-08-09): this sim fills
        # unspecified axes from its own stored pose, which is fine ONLY
        # because the sim achieves exactly what is commanded (achieved ==
        # commanded, so readback-fill is mathematically identical to the
        # real controller's last-commanded stash). If actuation bias or
        # noise is ever modeled here (the Exp 49 act-and-compare direction),
        # mirror ReachyMiniController._last_commanded or the F1 readback
        # ratchet returns in sim (tests/unit/test_reachy_retained_axes.py).

        # Update pose state. FRAME CONTRACT (matches the real controller,
        # review fold 2026-08-04): MotionTarget.head_yaw is BODY-RELATIVE;
        # the stored/reported pose "yaw" is WORLD-frame (relative + body),
        # exactly like ReachyMiniController.get_current_pose (the SDK's fk
        # folds body_yaw in). Storing the relative value verbatim made the
        # sim disagree with hardware by the body angle for any consumer
        # doing world − body — e.g. focus_on_sound's honest readback.
        #
        # BODY RIDE-ALONG (Exp 49 physics fix, 2026-08-04): a body-only
        # command (head_yaw=None) preserves the head's BODY-RELATIVE yaw —
        # the head (and its mounted mics/camera) rides along with the body,
        # exactly what ReachyMiniController.goto_target composes (it ships
        # a head matrix with `relative_yaw + target_body` for EVERY motion
        # touching yaw). The pre-fix sim left world yaw untouched on
        # body-only moves — the head=None counter-rotation pathology the
        # head-frame invariant documents, replicated in miniature: body
        # turns would not have moved the simulated mic array at all.
        old_body = float(self._current_pose.get("body_yaw", 0.0) or 0.0)
        old_world_yaw = float(self._current_pose.get("yaw", 0.0) or 0.0)
        new_body = old_body if target.body_yaw is None else float(target.body_yaw)
        new_body = max(-self._body_yaw_limit_rad, min(self._body_yaw_limit_rad, new_body))
        if target.head_yaw is not None:
            relative_yaw = float(target.head_yaw)
        else:
            relative_yaw = old_world_yaw - old_body
        # The daemon-analog neck envelope: the head cannot exceed this
        # BODY-RELATIVE yaw (Exp 49 pins the measured ~±22°; the default is
        # the real controller's ±65° capability limit). Clamped on every
        # motion so a body turn cannot smuggle the head past the envelope.
        relative_yaw = max(-self._head_yaw_limit_rad, min(self._head_yaw_limit_rad, relative_yaw))

        if target.head_roll is not None:
            self._current_pose["roll"] = max(
                -self._head_roll_limit_rad, min(self._head_roll_limit_rad, float(target.head_roll))
            )
        if target.head_pitch is not None:
            self._current_pose["pitch"] = max(
                -self._head_pitch_limit_rad, min(self._head_pitch_limit_rad, float(target.head_pitch))
            )
        self._current_pose["body_yaw"] = new_body
        self._current_pose["yaw"] = relative_yaw + new_body

        if self._simulate_delays:
            time.sleep(target.duration * 0.1)  # 10% of actual duration

        self._update_state(current_pose=self._current_pose.copy())

        if self._doa_scenario is not None:
            _scenario_log.info(
                "sim_doa.motion",
                extra={
                    "event": "sim_doa.motion",
                    "data": {
                        "body_yaw_deg_before": round(math.degrees(old_body), 2),
                        "body_yaw_deg_after": round(math.degrees(new_body), 2),
                        "head_world_deg_before": round(math.degrees(old_world_yaw), 2),
                        "head_world_deg_after": round(math.degrees(relative_yaw + new_body), 2),
                        "head_relative_deg_after": round(math.degrees(relative_yaw), 2),
                        "commanded_body": target.body_yaw is not None,
                        "commanded_head": target.head_yaw is not None,
                    },
                },
            )

        logger.debug("Simulated motion to: %s", self._current_pose)
        return True

    def look_at_pixel(self, target: PixelTarget) -> bool:
        """Move head to look at a pixel coordinate (simulated).

        Converts pixel to approximate head angles.

        Args:
            target: Target pixel coordinates.

        Returns:
            True.
        """
        if not self.is_connected():
            return False

        # Simple conversion: center of frame = (0, 0)
        # Assume 640x480 resolution, 60 degree FOV
        width, height = self._video_resolution
        center_x, center_y = width / 2, height / 2

        # Convert to approximate angles (radians)
        fov_h = 1.0  # ~60 degrees horizontal
        fov_v = 0.75  # ~45 degrees vertical

        yaw = -((target.u - center_x) / center_x) * (fov_h / 2)
        pitch = ((target.v - center_y) / center_y) * (fov_v / 2)

        # A pixel gaze is a HEAD-RELATIVE command (camera frame): route it
        # through the same relative-clamp + body composition as goto_target
        # so the envelope cannot be smuggled past and the stored yaw stays
        # world-frame (review fold — the old direct write treated a
        # camera-relative yaw as world and bypassed head_yaw_limit_deg).
        if self._head_yaw_limit_rad is not None:
            yaw = max(-self._head_yaw_limit_rad, min(self._head_yaw_limit_rad, yaw))
        body = float(self._current_pose.get("body_yaw", 0.0) or 0.0)
        self._current_pose["yaw"] = yaw + body
        self._current_pose["pitch"] = pitch

        if self._simulate_delays:
            time.sleep(target.duration * 0.1)

        self._update_state(current_pose=self._current_pose.copy())

        logger.debug("Simulated look at pixel (%d, %d) -> pitch=%.2f, yaw=%.2f", target.u, target.v, pitch, yaw)
        return True

    def get_current_pose(self) -> dict[str, float]:
        """Get current joint positions.

        Returns:
            Dict with simulated pose values.
        """
        return self._current_pose.copy()

    # ─────────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────────────────────────────────────

    def wake_up(self) -> bool:
        """Wake up the robot (simulated).

        Returns:
            True.
        """
        if not self.is_connected():
            return False

        if self._simulate_delays:
            time.sleep(0.1)

        self._update_state(is_awake=True)
        logger.info("Simulated robot awake: %s", self._robot_id)
        return True

    def goto_sleep(self) -> bool:
        """Put the robot to sleep (simulated).

        Returns:
            True.
        """
        if self._simulate_delays:
            time.sleep(0.1)

        self._update_state(is_awake=False)
        logger.info("Simulated robot asleep: %s", self._robot_id)
        return True

    def start_recording(self) -> bool:
        """Start media recording/streaming.

        Returns:
            True.
        """
        if not self.is_connected():
            return False

        if self._video_stream:
            self._video_stream.start()
        if self._audio_stream:
            self._audio_stream.start()

        self._update_state(is_recording=True)
        logger.info("Simulated recording started: %s", self._robot_id)
        return True

    def stop_recording(self) -> bool:
        """Stop media recording/streaming.

        Returns:
            True.
        """
        if self._video_stream:
            self._video_stream.stop()
        if self._audio_stream:
            self._audio_stream.stop()

        self._update_state(is_recording=False)
        logger.info("Simulated recording stopped: %s", self._robot_id)
        return True

    # ─────────────────────────────────────────────────────────────────────────
    # Data Streams
    # ─────────────────────────────────────────────────────────────────────────

    def get_video_stream(self) -> VideoStream | None:
        """Get the mock video stream.

        Returns:
            MockVideoStream, or None if not connected.
        """
        return self._video_stream

    def get_audio_stream(self) -> AudioStream | None:
        """Get the mock audio stream.

        Returns:
            MockAudioStream, or None if not connected.
        """
        return self._audio_stream

    def get_doa_reader(self):
        """Get the DoA reader — injected, or the configured scenario's.

        Pins the Stage-1 capability seam without hardware: the reader is
        returned only while connected (mirroring the Reachy
        implementation's connected gate); otherwise ``None``. An injected
        ``doa_reader`` wins over a ``doa_source_bearing_deg`` scenario;
        with neither, the capability is absent (a robot without a mic).
        """
        if not self.is_connected():
            return None
        return self._doa_reader
