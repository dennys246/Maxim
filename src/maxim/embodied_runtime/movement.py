"""Movement & kinematics mixin for the Selfy body controller.

Provides all head-tracking, gaze, workspace-clamping, and body-rotation
methods that are mixed into the main Selfy class.
"""

from __future__ import annotations

import math
import time
import uuid
from typing import Any, Optional

from maxim.motion.movement import (
    move_antenna,
    move_head,
)
from maxim.utils.logging import warn


class MovementMixin:
    """Mixin that supplies movement / kinematics behaviour to Selfy."""

    # Safe pixel bounds for look_at_image to prevent IK failures.
    # With translation-first approach, we can use more of the frame.
    _LOOK_AT_PIXEL_BOUNDS = {
        "u_min": 64,
        "u_max": 576,
        "v_min": 48,
        "v_max": 432,
    }

    # ═══════════════════════════════════════════════════════════════════════════
    # BLENDED 6-DOF WORKSPACE LIMITS
    # ═══════════════════════════════════════════════════════════════════════════
    # We use BOTH translation AND rotation in a balanced way:
    # - Translation provides mechanical stability for the Stewart platform
    # - Rotation provides natural head-movement feel
    #
    # Coordinate system (from SDK docs):
    #   X = forward (positive = toward what you're looking at)
    #   Y = left (positive = left, negative = right)
    #   Z = upward (positive = up, negative = down)
    #
    # SDK hint: "1 degree ≈ 1 mm" as rough equivalence
    # ═══════════════════════════════════════════════════════════════════════════

    # Translation limits (mm) - moderate range for stability
    _SAFE_X_LIMIT = 15.0  # Forward/backward (mm) - rarely used
    _SAFE_Y_LIMIT = 18.0  # Left/right (mm)
    _SAFE_Z_LIMIT = 15.0  # Up/down (mm)

    # Rotation limits (degrees) - based on Reachy Mini SDK documentation
    # SDK limits: Roll/Pitch [-40°, +40°], Yaw delta max 65° (head relative to body)
    # Using slightly conservative values for safety margin
    _SAFE_ROLL_LIMIT = 35.0  # Head tilt (degrees) - SDK allows ±40°
    _SAFE_PITCH_LIMIT = 35.0  # Look up/down (degrees) - SDK allows ±40°
    _SAFE_YAW_LIMIT = 55.0  # Look left/right (degrees) - SDK allows ±65° (relative to body)

    # Turn-around trigger settings
    # When head is at horizontal limit and target is beyond, rotate body instead
    # Note: Threshold is now LEARNED via pain detection - no hardcoded _TURN_AROUND_THRESHOLD
    _TURN_AROUND_MIN_PIXEL_OFFSET = 60  # Min pixel offset to trigger (avoid small targets)
    _TURN_AROUND_COOLDOWN = 12.0  # Seconds between turn_around triggers
    _TURN_AROUND_ANGLE = 45.0  # Degrees to rotate body (matches yaw limit for smooth transition)
    _TURN_AROUND_DURATION = 6.0  # Seconds for body rotation (slow and deliberate)
    _TURN_AROUND_PAIN_THRESHOLD = 0.3  # Pain risk threshold to trigger turn-around
    _TURN_AROUND_BOUNDS_THRESHOLD = 0.75  # Fallback: bounds usage threshold if pain not available

    # Camera parameters for converting pixels to movement
    # Internal coordinate system (matches behavior/detection pipeline)
    _IMAGE_CENTER_U = 320.0
    _IMAGE_CENTER_V = 240.0
    _IMAGE_WIDTH = 640.0
    _IMAGE_HEIGHT = 480.0

    # SDK's expected resolution (camera native resolution)
    # The SDK's look_at_image expects coordinates in the camera's native resolution
    _SDK_IMAGE_WIDTH = 1920.0
    _SDK_IMAGE_HEIGHT = 1080.0
    _SDK_IMAGE_CENTER_U = 960.0  # 1920 / 2
    _SDK_IMAGE_CENTER_V = 540.0  # 1080 / 2

    # Camera FOV (approximate)
    _HORIZONTAL_FOV_DEG = 70.0  # degrees
    _VERTICAL_FOV_DEG = 50.0  # degrees

    # ═══════════════════════════════════════════════════════════════════════════
    # BLENDED MOVEMENT PER PIXEL
    # ═══════════════════════════════════════════════════════════════════════════
    # Both translation and rotation contribute to movement.
    # Rotation is now more prominent for natural head movement,
    # while translation provides the mechanical stability base.
    # For a 100px offset: ~5mm translation + ~10° rotation
    _MM_PER_PIXEL_H = 0.05  # mm translation per pixel horizontally
    _MM_PER_PIXEL_V = 0.04  # mm translation per pixel vertically
    _DEG_PER_PIXEL_H = 0.10  # degrees rotation per pixel horizontally
    _DEG_PER_PIXEL_V = 0.08  # degrees rotation per pixel vertically

    def center_vision(self, *, duration: Optional[float] = None) -> None:
        return self.goto_pose("centered", duration=duration)

    def mark_trainable_moment(self) -> None:
        sample = getattr(self, "_last_motor_sample", None)
        training_logger = getattr(self, "_training_logger", None)
        if training_logger is None:
            warn("Training sample logger is not running.", logger=self.log)
            return
        if not isinstance(sample, dict) or not sample:
            warn("No recent motor sample to mark yet.", logger=self.log)
            return

        record = dict(sample)
        record["user_marked"] = True
        record["mark_time"] = time.time()
        record["mark_id"] = uuid.uuid4().hex
        record["marked_from_sample_id"] = record.get("sample_id")

        try:
            training_logger.log_motor_sample(record, flush=True)
        except Exception as e:
            warn("Failed to mark trainable moment: %s", e, logger=self.log)

    def goto_pose(self, name: str = "centered", *, duration: Optional[float] = None) -> None:
        pose = None
        try:
            pose = getattr(self, "poses", {}).get(name)
        except Exception:
            pose = None

        # Collect into locals and dispatch through self.move() — NOT a direct
        # _enqueue_motor(move_head, ...) (2026-08-07 safety fold): move() owns
        # the workspace clamp, so a config-authored pose beyond limits cannot
        # ship raw. (Setting self.x..yaw first and then calling move() would
        # trip move()'s no-op check — collect first, assign nothing.)
        vals: tuple[float, float, float, float, float, float] | None = None
        if isinstance(pose, (list, tuple)) and len(pose) >= 6:
            try:
                vals = tuple(float(pose[i]) for i in range(6))  # type: ignore[assignment]
                if duration is None and len(pose) >= 7:
                    duration = float(pose[6])
            except Exception:
                vals = None

        if vals is None:
            fallback = getattr(self, "_default_head_pose", None)
            if not isinstance(fallback, dict):
                fallback = {}
            vals = (
                float(fallback.get("x", 0.0) or 0.0),
                float(fallback.get("y", 0.0) or 0.0),
                float(fallback.get("z", 0.0) or 0.0),
                float(fallback.get("roll", 0.0) or 0.0),
                float(fallback.get("pitch", 0.0) or 0.0),
                float(fallback.get("yaw", 0.0) or 0.0),
            )

        if duration is None:
            duration = float(getattr(self, "duration", 0.5) or 0.5)

        try:
            self.move(
                x=vals[0],
                y=vals[1],
                z=vals[2],
                roll=vals[3],
                pitch=vals[4],
                yaw=vals[5],
                duration=float(duration),
            )
        except Exception as e:
            warn("Failed to center vision: %s", e, logger=self.log)

        try:
            time.sleep(float(duration))
        except Exception:
            pass

    def sync_head_position(self) -> bool:
        """Sync internal position tracking with actual hardware position.

        Reads the current head pose from the reachy_mini SDK and updates
        the internal yaw/pitch tracking to match. This helps correct drift
        between software tracking and actual hardware position.

        Controllers WITHOUT an SDK handle (``self.mini`` is None — e.g.
        ``SimulatedController``) sync from the RobotController abstraction
        instead: ``get_current_pose()`` reports world-frame head yaw +
        body yaw in radians (the same frame contract as the SDK path), so
        the mirrors (``self.yaw`` body-relative degrees, ``self.body_yaw``
        degrees) stay truthful. Pre-fix the SDK read raised into the
        except and the mirrors silently froze — every consumer of the
        frame (DoA capture stamps, focus_on_sound aim, the motor
        backend's post-turn sync) then reasoned in a stale frame on any
        non-SDK controller (Exp 49 seam, 2026-08-04).

        Returns:
            True if sync was successful, False otherwise.
        """
        if getattr(self, "mini", None) is None:
            return self._sync_head_position_from_controller()
        try:
            import math

            # Get current joint positions - this gives us body_yaw directly.
            # SDK >= 1.5 joint vector is [body_yaw, *6 stewart_legs] — its own
            # kinematics reads INDEX 0 as body_yaw (analytical_kinematics.fk:
            # `body_yaw = joint_angles[0]`; ik returns `[body_yaw] + stewart`).
            # The previous index-6 read (zenoh-era ordering, never migrated
            # across the v1.5 pivot) returned a STEWART LEG angle, so every
            # body-relative yaw derived here folded a leg-motor angle in —
            # measured live 2026-08-04: maxim.yaw reported 26.0° while the
            # daemon's ground truth (/api/state/full) showed world yaw −0.4°,
            # body −0.3°, with a leg at ≈ −0.45 rad ≈ −26°. That phantom
            # frame corrupted DoA capture yaw, focus_on_sound aim math, and
            # the bounds learner's coordinates.
            head_joints, antenna_joints = self.mini.get_current_joint_positions()

            if head_joints is not None and len(head_joints) >= 1:
                body_yaw_rad = head_joints[0]
                body_yaw_deg = math.degrees(body_yaw_rad)
                self.body_yaw = body_yaw_deg
            else:
                body_yaw_deg = float(getattr(self, "body_yaw", 0.0) or 0.0)

            # Get current pose matrix from SDK
            pose_matrix = self.mini.get_current_head_pose()

            # Extract rotation matrix (upper-left 3x3)
            R = pose_matrix[:3, :3]

            # Extract Euler angles from rotation matrix
            # Using ZYX convention (yaw, pitch, roll)
            # Check for gimbal lock
            if abs(R[2, 0]) < 0.9999:
                pitch = -math.asin(R[2, 0])
                yaw_world = math.atan2(R[1, 0], R[0, 0])
                roll = math.atan2(R[2, 1], R[2, 2])
            else:
                # Gimbal lock - pitch is ±90°
                pitch = math.copysign(math.pi / 2, -R[2, 0])
                yaw_world = math.atan2(-R[0, 1], R[1, 1])
                roll = 0.0

            # Convert to degrees
            yaw_world_deg = math.degrees(yaw_world)
            pitch_deg = math.degrees(pitch)
            roll_deg = math.degrees(roll)

            # The pose matrix from get_current_head_pose() is in WORLD frame.
            # We need to convert to HEAD-relative yaw by accounting for body rotation.
            #
            # SDK sign convention:
            # - Negative body_yaw = body has turned LEFT
            # - Positive body_yaw = body has turned RIGHT
            #
            # Formula: head_yaw = world_yaw - body_yaw
            #
            # Example: body turned left 77° (body_yaw=-77), head at world_yaw=-65
            # - head_yaw = -65 - (-77) = -65 + 77 = +12° (head is 12° right of body forward)
            yaw_deg = yaw_world_deg - body_yaw_deg  # SUBTRACT to get head-relative

            # Normalize to [-180, 180]
            while yaw_deg > 180:
                yaw_deg -= 360
            while yaw_deg < -180:
                yaw_deg += 360

            # DEBUG: Log body_yaw and world_yaw to diagnose frame issues
            if abs(yaw_deg) > 55.0:
                self.log.warning(
                    "SYNC_DEBUG: yaw=%.1f EXCEEDS physical limit | world_yaw=%.1f body_yaw=%.1f",
                    yaw_deg,
                    yaw_world_deg,
                    body_yaw_deg,
                )

            # SANITY CHECK: The head physically cannot rotate more than ±55° relative
            # to the body. If calculated yaw exceeds this, clamp it to prevent bad behavior.
            # Previous ±90° threshold was too permissive and allowed invalid values through.
            # Use workspace limits (respects protocol overrides)
            limits = self._get_workspace_limits()
            HEAD_YAW_PHYSICAL_LIMIT = limits.get("yaw", 55.0)
            HEAD_PITCH_PHYSICAL_LIMIT = limits.get("pitch", 35.0)
            HEAD_ROLL_PHYSICAL_LIMIT = limits.get("roll", 35.0)

            if abs(yaw_deg) > HEAD_YAW_PHYSICAL_LIMIT:
                old_yaw = float(getattr(self, "yaw", 0.0) or 0.0)
                clamped_yaw = max(-HEAD_YAW_PHYSICAL_LIMIT, min(HEAD_YAW_PHYSICAL_LIMIT, yaw_deg))
                self.log.warning(
                    "Sync: clamping invalid yaw=%.1f -> %.1f (world=%.1f - body=%.1f, old=%.1f)",
                    yaw_deg,
                    clamped_yaw,
                    yaw_world_deg,
                    body_yaw_deg,
                    old_yaw,
                )
                yaw_deg = clamped_yaw

            if abs(pitch_deg) > HEAD_PITCH_PHYSICAL_LIMIT:
                old_pitch = float(getattr(self, "pitch", 0.0) or 0.0)
                clamped_pitch = max(-HEAD_PITCH_PHYSICAL_LIMIT, min(HEAD_PITCH_PHYSICAL_LIMIT, pitch_deg))
                self.log.warning(
                    "Sync: clamping invalid pitch=%.1f -> %.1f (old=%.1f)", pitch_deg, clamped_pitch, old_pitch
                )
                pitch_deg = clamped_pitch

            if abs(roll_deg) > HEAD_ROLL_PHYSICAL_LIMIT:
                old_roll = float(getattr(self, "roll", 0.0) or 0.0)
                clamped_roll = max(-HEAD_ROLL_PHYSICAL_LIMIT, min(HEAD_ROLL_PHYSICAL_LIMIT, roll_deg))
                self.log.warning(
                    "Sync: clamping invalid roll=%.1f -> %.1f (old=%.1f)", roll_deg, clamped_roll, old_roll
                )
                roll_deg = clamped_roll

            # Extract translation (convert meters to mm)
            x_mm = pose_matrix[0, 3] * 1000
            y_mm = pose_matrix[1, 3] * 1000
            z_mm = pose_matrix[2, 3] * 1000

            # Update internal tracking
            old_yaw = float(getattr(self, "yaw", 0.0) or 0.0)
            old_pitch = float(getattr(self, "pitch", 0.0) or 0.0)
            old_y = float(getattr(self, "y", 0.0) or 0.0)
            old_z = float(getattr(self, "z", 0.0) or 0.0)

            self.x = x_mm
            self.y = y_mm
            self.z = z_mm
            self.roll = roll_deg
            self.pitch = pitch_deg
            self.yaw = yaw_deg

            # Only log if there's a significant difference (>2° or >2mm)
            yaw_diff = abs(old_yaw - yaw_deg)
            pitch_diff = abs(old_pitch - pitch_deg)
            y_diff = abs(old_y - y_mm)
            z_diff = abs(old_z - z_mm)

            if yaw_diff > 2.0 or pitch_diff > 2.0 or y_diff > 2.0 or z_diff > 2.0:
                self.log.debug(
                    "Sync: yaw=%.1f (body=%.1f), pitch=%.1f, y=%.1f, z=%.1f",
                    yaw_deg,
                    body_yaw_deg,
                    pitch_deg,
                    y_mm,
                    z_mm,
                )

            # Record position for pain detection (if enabled)
            pain_bridge = getattr(getattr(self, "_default_network", None), "_pain_bridge", None)
            if pain_bridge is not None:
                try:
                    pain_bridge.detector.record_position(
                        yaw=yaw_deg,
                        pitch=pitch_deg,
                        x=x_mm,
                        y=y_mm,
                        z=z_mm,
                        roll=roll_deg,
                    )
                except Exception as pain_e:
                    self.log.debug("Pain recording failed: %s", pain_e)

            return True

        except Exception as e:
            self.log.warning("Failed to sync head position: %s", e)
            return False

    def _sync_head_position_from_controller(self) -> bool:
        """Sync the frame mirrors from the RobotController abstraction.

        The non-SDK arm of :meth:`sync_head_position` (see its docstring).
        ``get_current_pose()`` reports radians with pose ``yaw`` in the
        WORLD frame and ``body_yaw`` separate — the same contract as the
        SDK's fk — so the conversion mirrors the SDK path exactly:
        ``self.yaw`` = (world − body) in degrees (body-relative),
        ``self.body_yaw`` = body degrees. Translation (x/y/z) is left
        untouched — the sim controller does not model it, and a fabricated
        zero would overwrite the operator's centered-pose defaults.
        """
        try:
            import math

            robot = getattr(self, "_robot", None)
            if robot is None or not robot.is_connected():
                return False
            pose = robot.get_current_pose() or {}
            if "yaw" not in pose:
                return False
            body_yaw_deg = math.degrees(float(pose.get("body_yaw", 0.0) or 0.0))
            yaw_deg = math.degrees(float(pose["yaw"])) - body_yaw_deg
            while yaw_deg > 180:
                yaw_deg -= 360
            while yaw_deg < -180:
                yaw_deg += 360
            self.body_yaw = body_yaw_deg
            self.yaw = yaw_deg
            if "pitch" in pose:
                self.pitch = math.degrees(float(pose["pitch"]))
            if "roll" in pose:
                self.roll = math.degrees(float(pose["roll"]))
            return True
        except Exception as e:
            self.log.debug("Controller-abstraction head sync failed: %s", e)
            return False

    # Protocol workspace override — set by ProtocolRegistry, read by
    # _get_workspace_limits(). Can only tighten, never widen limits.
    _workspace_limit_override: dict[str, float] | None = None

    def _get_workspace_limits(self) -> dict[str, float]:
        """Get workspace limits: protocol override > learned bounds > hardcoded.

        Returns:
            Dict with x, y, z, roll, pitch, yaw limits (all positive values).
        """
        # 1. Start with hardcoded defaults
        limits = {
            "x": self._SAFE_X_LIMIT,
            "y": self._SAFE_Y_LIMIT,
            "z": self._SAFE_Z_LIMIT,
            "roll": self._SAFE_ROLL_LIMIT,
            "pitch": self._SAFE_PITCH_LIMIT,
            "yaw": self._SAFE_YAW_LIMIT,
        }

        # 2. Override with learned bounds if available — TIGHTEN-ONLY
        # (2026-08-07 safety fold): the learner exists to shrink limits that
        # caused pain; a config-inflated learner ceiling must never WIDEN
        # what is now a safety clamp past the hardcoded safe limits.
        default_network = getattr(self, "_default_network", None)
        if default_network is not None:
            bounds_learner = getattr(default_network, "_bounds_learner", None)
            if bounds_learner is not None:
                for axis in limits:
                    limits[axis] = min(limits[axis], bounds_learner.get_bound(axis))

        # 3. Apply protocol override (can only tighten, never widen)
        override = self._workspace_limit_override
        if override is not None:
            for axis, val in override.items():
                if val is not None:
                    limits[axis] = min(limits[axis], val)

        return limits

    def _clamp_to_workspace_6d(
        self,
        x: float,
        y: float,
        z: float,
        roll: float,
        pitch: float,
        yaw: float,
    ) -> tuple[float, float, float, float, float, float]:
        """Clamp 6-DOF pose to the reachable Stewart platform workspace.

        The Stewart platform has a complex 6D workspace. We use a simplified
        model with separate ellipsoids for translation and rotation, plus
        a combined constraint to prevent extreme combinations.

        Translation (x, y, z) is the PRIMARY movement mechanism.
        Rotation (roll, pitch, yaw) is SECONDARY for personality/fine-tuning.

        Uses learned workspace bounds when available, falling back to hardcoded limits.

        Args:
            x, y, z: Translation in mm.
            roll, pitch, yaw: Rotation in degrees.

        Returns:
            Clamped (x, y, z, roll, pitch, yaw) tuple.
        """

        # Get limits (learned or hardcoded)
        limits = self._get_workspace_limits()
        x_limit = limits["x"]
        y_limit = limits["y"]
        z_limit = limits["z"]
        roll_limit = limits["roll"]
        pitch_limit = limits["pitch"]
        yaw_limit = limits["yaw"]

        # 1. Clamp translation to rectangular bounds first
        x = max(-x_limit, min(x_limit, x))
        y = max(-y_limit, min(y_limit, y))
        z = max(-z_limit, min(z_limit, z))

        # 2. Clamp rotation to rectangular bounds
        roll = max(-roll_limit, min(roll_limit, roll))
        pitch = max(-pitch_limit, min(pitch_limit, pitch))
        yaw = max(-yaw_limit, min(yaw_limit, yaw))

        # 3. Check translation ellipsoid: (x/max_x)² + (y/max_y)² + (z/max_z)² <= 1
        if x_limit > 0 and y_limit > 0 and z_limit > 0:
            norm_x = x / x_limit
            norm_y = y / y_limit
            norm_z = z / z_limit
            trans_dist = norm_x**2 + norm_y**2 + norm_z**2

            if trans_dist > 1.0:
                scale = 1.0 / math.sqrt(trans_dist)
                x, y, z = x * scale, y * scale, z * scale
                self.log.debug("Clamped translation to ellipsoid")

        # 4. Check rotation ellipsoid: (roll/max)² + (pitch/max)² + (yaw/max)² <= 1
        if roll_limit > 0 and pitch_limit > 0 and yaw_limit > 0:
            norm_roll = roll / roll_limit
            norm_pitch = pitch / pitch_limit
            norm_yaw = yaw / yaw_limit
            rot_dist = norm_roll**2 + norm_pitch**2 + norm_yaw**2

            if rot_dist > 1.0:
                scale = 1.0 / math.sqrt(rot_dist)
                roll, pitch, yaw = roll * scale, pitch * scale, yaw * scale
                self.log.debug("Clamped rotation to ellipsoid")

        # 5. Combined constraint: only reduce rotation when translation is VERY high
        # The blended approach allows both translation and rotation to work together,
        # but we still need to prevent extreme combined positions that cause IK failure.
        trans_usage = math.sqrt((x / x_limit) ** 2 + (y / y_limit) ** 2 + (z / z_limit) ** 2) if x_limit > 0 else 0
        rot_usage = (
            math.sqrt((roll / roll_limit) ** 2 + (pitch / pitch_limit) ** 2 + (yaw / yaw_limit) ** 2)
            if yaw_limit > 0
            else 0
        )

        # Only apply combined constraint when BOTH are high
        combined = trans_usage * 0.5 + rot_usage * 0.5  # Equal weighting
        if combined > 0.85:  # If combined usage exceeds 85%
            # Scale both down proportionally to stay within workspace
            scale = 0.85 / combined
            x, y, z = x * scale, y * scale, z * scale
            roll, pitch, yaw = roll * scale, pitch * scale, yaw * scale
            self.log.debug("Scaled both trans/rot due to combined limit: %.2f -> %.2f", combined, 0.85)

        return (x, y, z, roll, pitch, yaw)

    def _clamp_to_workspace(self, yaw: float, pitch: float) -> tuple[float, float]:
        """Legacy 2D clamp for backward compatibility.

        Delegates to 6D clamping with zero translation.
        """
        _, _, _, _, pitch_out, yaw_out = self._clamp_to_workspace_6d(0, 0, 0, 0, pitch, yaw)
        return (yaw_out, pitch_out)

    def _calculate_movement_for_pixel(self, du: float, dv: float) -> tuple[float, float, float, float, float, float]:
        """Calculate 6-DOF movement needed to center a pixel offset.

        Uses TRANSLATION as the primary movement mechanism:
        - Y translation for horizontal movement (left/right)
        - Z translation for vertical movement (up/down)

        Uses ROTATION as secondary for personality and efficiency:
        - Yaw for horizontal fine-tuning
        - Pitch for vertical fine-tuning

        Args:
            du: Horizontal pixel offset from center (positive = right).
            dv: Vertical pixel offset from center (positive = down).

        Returns:
            (dx, dy, dz, droll, dpitch, dyaw) deltas to apply.
        """
        # Translation deltas (primary movement)
        # Note: positive du (right of center) requires negative Y (move head right)
        # Note: positive dv (below center) requires negative Z (move head down)
        dy = -du * self._MM_PER_PIXEL_H  # Y = left, so negative for right
        dz = -dv * self._MM_PER_PIXEL_V  # Z = up, so negative for down
        dx = 0.0  # X (forward) not typically needed for looking at pixels

        # Rotation deltas (secondary/supplementary)
        # Same sign convention as translation
        dyaw = -du * self._DEG_PER_PIXEL_H  # Turn right for right-of-center
        dpitch = dv * self._DEG_PER_PIXEL_V  # Look down for below-center
        droll = 0.0  # Roll not used for pixel tracking

        return (dx, dy, dz, droll, dpitch, dyaw)

    def look_at_image(
        self,
        u: int,
        v: int,
        *,
        duration: Optional[float] = None,
        perform_movement: bool = True,
        min_reachability: float = 0.2,
        clamp_to_bounds: bool = True,
    ) -> None:
        """Look at a position in image coordinates.

        Args:
            u: Horizontal pixel coordinate.
            v: Vertical pixel coordinate.
            duration: Movement duration in seconds.
            perform_movement: Whether to actually move (vs just calculate).
            min_reachability: Minimum reachability score to attempt movement (0-1).
                              Set to 0 to disable reachability gating.
            clamp_to_bounds: Whether to clamp coordinates to safe pixel bounds.
                             This prevents the robot from drifting outside servo limits.
        """
        if duration is None:
            duration = getattr(self, "duration", 0.5)

        u, v = self._clamp_pixel_to_safe_bounds(u, v, clamp_to_bounds)
        position = (int(u), int(v))

        if min_reachability > 0 and not self._is_pixel_reachable(position, min_reachability):
            return

        if not perform_movement:
            self._enqueue_sdk_look_at(u, v, position, duration, perform_movement=False)
            return

        # Sync to get accurate current position before computing offsets
        self.sync_head_position()

        cur_yaw = float(getattr(self, "yaw", 0.0) or 0.0)
        cur_pitch = float(getattr(self, "pitch", 0.0) or 0.0)
        cur_y = float(getattr(self, "y", 0.0) or 0.0)
        cur_z = float(getattr(self, "z", 0.0) or 0.0)

        # Pixel offset from center
        du = float(u) - self._IMAGE_CENTER_U
        dv = float(v) - self._IMAGE_CENTER_V

        # Apply learned/fallback gain
        du, dv = self._apply_focus_gain(du, dv)

        # Compute usage of horizontal/vertical workspace + lookup current direction
        limits = self._get_workspace_limits()
        h_usage, looking_left, looking_right = self._horizontal_workspace_state(cur_yaw, cur_y, limits)

        pain_bridge = getattr(getattr(self, "_default_network", None), "_pain_bridge", None)

        # If at a limit and target is further into it, rotate body instead of pushing servo
        if self._maybe_turn_around(
            du=du,
            looking_left=looking_left,
            looking_right=looking_right,
            h_usage=h_usage,
            pain_bridge=pain_bridge,
        ):
            return  # turn_around handled the movement

        # Restrict du/dv based on pain risk or workspace usage near limits
        du = self._restrict_horizontal(du, looking_left, looking_right, h_usage, pain_bridge)
        dv = self._restrict_vertical(dv, cur_pitch, cur_z, limits, pain_bridge)

        # Compute final pixel coords (with hard bounds as the last safety net)
        final_u, final_v = self._final_pixel_with_bounds(du, dv)
        self._log_look_at_debug(u, v, du, dv, cur_yaw, cur_pitch, final_u, final_v)

        commanded_6d = self._build_commanded_6d(du, dv, cur_yaw, cur_pitch, cur_y, cur_z)
        self._record_pain_action_start(commanded_6d, position)

        self._enqueue_sdk_look_at(
            final_u,
            final_v,
            position,
            duration,
            perform_movement=True,
            commanded_6d=commanded_6d,
        )

    # ── look_at_image helpers ──────────────────────────────────────────────

    def _clamp_pixel_to_safe_bounds(self, u: int, v: int, clamp_to_bounds: bool) -> tuple[int, int]:
        """Clamp ``(u, v)`` to ``_LOOK_AT_PIXEL_BOUNDS`` to prevent servo drift."""
        if not clamp_to_bounds:
            return int(u), int(v)
        bounds = self._LOOK_AT_PIXEL_BOUNDS
        u_clamped = max(bounds["u_min"], min(bounds["u_max"], int(u)))
        v_clamped = max(bounds["v_min"], min(bounds["v_max"], int(v)))
        if u_clamped != int(u) or v_clamped != int(v):
            self.log.debug("Clamped look_at_image coordinates: (%d, %d) -> (%d, %d)", u, v, u_clamped, v_clamped)
        return u_clamped, v_clamped

    def _is_pixel_reachable(self, position: tuple[int, int], min_reachability: float) -> bool:
        """Check the AttentionNetwork's learned reachability for ``position``."""
        default_network = getattr(self, "_default_network", None)
        if default_network is None:
            return True
        attention = getattr(default_network, "_attention_network", None)
        if attention is None:
            return True
        try:
            reachability = attention.get_reachability(position)
        except Exception as e:
            self.log.debug("Reachability check failed for %s: %s", position, e)
            return True  # be permissive on lookup failure
        if reachability < min_reachability:
            self.log.debug(
                "Skipping unreachable position %s — reachability %.2f < %.2f",
                position,
                reachability,
                min_reachability,
            )
            return False
        return True

    def _apply_focus_gain(self, du: float, dv: float) -> tuple[float, float]:
        """Apply ``FocusLearner`` adaptive gain (or fallback dampening)."""
        default_network = getattr(self, "_default_network", None)
        focus_learner = getattr(default_network, "_focus_learner", None) if default_network else None

        if focus_learner is None:
            if default_network is None:
                self.log.debug("FocusLearner unavailable: _default_network is None")
            else:
                self.log.debug("FocusLearner unavailable: _focus_learner is None in DefaultNetwork")
            DAMPENING_FACTOR = 0.5
            self.log.warning("Using fallback DAMPENING_FACTOR=0.5 (FocusLearner unavailable)")
            return du * DAMPENING_FACTOR, dv * DAMPENING_FACTOR

        gain_h, gain_v = focus_learner.get_gain(du, dv)
        focus_learner.record_intent(du, dv, gain_h, gain_v)
        self.log.info(
            "FocusLearner: raw_du=%.1f raw_dv=%.1f gain_h=%.3f gain_v=%.3f",
            du,
            dv,
            gain_h,
            gain_v,
        )
        return du * gain_h, dv * gain_v

    def _horizontal_workspace_state(
        self,
        cur_yaw: float,
        cur_y: float,
        limits: dict[str, float],
    ) -> tuple[float, bool, bool]:
        """Return ``(h_usage, looking_left, looking_right)`` for the current pose."""
        yaw_limit = limits["yaw"]
        y_limit = limits["y"]
        yaw_usage = abs(cur_yaw) / yaw_limit if yaw_limit > 0 else 0
        y_usage = abs(cur_y) / y_limit if y_limit > 0 else 0
        h_usage = max(yaw_usage, y_usage)  # Use the more restrictive
        looking_left = cur_yaw > 0 or cur_y > 0
        looking_right = cur_yaw < 0 or cur_y < 0
        return h_usage, looking_left, looking_right

    def _maybe_turn_around(
        self,
        *,
        du: float,
        looking_left: bool,
        looking_right: bool,
        h_usage: float,
        pain_bridge: Any,
    ) -> bool:
        """If movement would push past a limit, rotate the body instead.

        Returns ``True`` if a turn was executed (caller should early-return).
        """
        import time as time_module

        now = time_module.time()
        can_turn = (now - self._last_turn_around_time) > self._TURN_AROUND_COOLDOWN

        target_is_beyond = (looking_left and du < -self._TURN_AROUND_MIN_PIXEL_OFFSET) or (
            looking_right and du > self._TURN_AROUND_MIN_PIXEL_OFFSET
        )
        if not (can_turn and target_is_beyond):
            return False

        should_turn = False
        turn_reason = ""
        if pain_bridge is not None:
            dyaw = abs(du * self._DEG_PER_PIXEL_H)
            action_sig = f"look_at:dy={dyaw:.0f}:dp=0"
            pain_risk = pain_bridge.get_pain_risk(action_sig)
            if pain_risk >= self._TURN_AROUND_PAIN_THRESHOLD:
                should_turn = True
                turn_reason = f"pain_risk={pain_risk:.2f}"
        elif h_usage > self._TURN_AROUND_BOUNDS_THRESHOLD:
            should_turn = True
            turn_reason = f"bounds_usage={h_usage:.2f}"

        if not should_turn:
            return False

        turn_angle = self._TURN_AROUND_ANGLE if looking_left else -self._TURN_AROUND_ANGLE
        self.log.info(
            "look_at_image triggering turn_around: %s, du=%.1f, turning %.0f°",
            turn_reason,
            du,
            turn_angle,
        )
        self._last_turn_around_time = now

        if pain_bridge is not None:
            try:
                pain_bridge.record_action_start(
                    "turn_around",
                    context={"reason": turn_reason, "angle": turn_angle},
                )
            except Exception as e:
                self.log.debug("Pain bridge record_action_start failed: %s", e)

        self.turn_around(turn_angle, duration=self._TURN_AROUND_DURATION, recenter_head=True)

        if pain_bridge is not None:
            try:
                pain_bridge.record_action_complete(success=True)
            except Exception as e:
                self.log.debug("Pain bridge record_action_complete failed: %s", e)

        return True

    def _restrict_horizontal(
        self,
        du: float,
        looking_left: bool,
        looking_right: bool,
        h_usage: float,
        pain_bridge: Any,
    ) -> float:
        """Reduce horizontal pixel delta when near a limit (pain-aware)."""
        h_restrict = 0.0
        if pain_bridge is not None:
            dyaw_h = abs(du * self._DEG_PER_PIXEL_H)
            action_sig_h = f"look_at:dy={dyaw_h:.0f}:dp=0"
            h_pain_risk = pain_bridge.get_pain_risk(action_sig_h)
            h_restrict = min(1.0, h_pain_risk * 2.0)  # Scale up for faster response
        elif h_usage > 0.5:
            h_restrict = min(1.0, (h_usage - 0.5) * 2)

        if h_restrict > 0.1:
            if looking_left and du < 0:
                du = du * (1.0 - h_restrict * 0.8)
            elif looking_right and du > 0:
                du = du * (1.0 - h_restrict * 0.8)
        return du

    def _restrict_vertical(
        self,
        dv: float,
        cur_pitch: float,
        cur_z: float,
        limits: dict[str, float],
        pain_bridge: Any,
    ) -> float:
        """Reduce vertical pixel delta when near a limit (pain-aware)."""
        pitch_limit = limits["pitch"]
        z_limit = limits["z"]
        pitch_usage = abs(cur_pitch) / pitch_limit if pitch_limit > 0 else 0
        z_usage = abs(cur_z) / z_limit if z_limit > 0 else 0
        v_usage = max(pitch_usage, z_usage)

        v_restrict = 0.0
        if pain_bridge is not None:
            dpitch_v = abs(dv * self._DEG_PER_PIXEL_V)
            action_sig_v = f"look_at:dy=0:dp={dpitch_v:.0f}"
            v_pain_risk = pain_bridge.get_pain_risk(action_sig_v)
            v_restrict = min(1.0, v_pain_risk * 2.0)
        elif v_usage > 0.5:
            v_restrict = min(1.0, (v_usage - 0.5) * 2)

        if v_restrict > 0.1:
            looking_down = cur_pitch > 0 or cur_z > 0
            looking_up = cur_pitch < 0 or cur_z < 0
            if looking_down and dv > 0:
                dv = dv * (1.0 - v_restrict * 0.8)
            elif looking_up and dv < 0:
                dv = dv * (1.0 - v_restrict * 0.8)
        return dv

    def _final_pixel_with_bounds(self, du: float, dv: float) -> tuple[int, int]:
        """Compute the final integer pixel target with hard bounds applied."""
        final_u = int(self._IMAGE_CENTER_U + du)
        final_v = int(self._IMAGE_CENTER_V + dv)
        px_bounds = self._LOOK_AT_PIXEL_BOUNDS
        final_u = max(px_bounds["u_min"], min(px_bounds["u_max"], final_u))
        final_v = max(px_bounds["v_min"], min(px_bounds["v_max"], final_v))
        return final_u, final_v

    def _log_look_at_debug(
        self,
        u: int,
        v: int,
        du: float,
        dv: float,
        cur_yaw: float,
        cur_pitch: float,
        final_u: int,
        final_v: int,
    ) -> None:
        """Trace pixel-to-motor direction for debug visibility."""
        du_direction = "RIGHT" if du > 0 else "LEFT" if du < 0 else "CENTER"
        expected_yaw_change = "decrease" if du > 0 else "increase" if du < 0 else "none"
        self.log.warning(
            "LOOK_AT_DEBUG: input_pixel=(%d,%d) center=(320,240) du=%.1f (%s) dv=%.1f "
            "-> expect_yaw_to_%s | cur_yaw=%.1f cur_pitch=%.1f | final_pixel=(%d,%d)",
            u,
            v,
            du,
            du_direction,
            dv,
            expected_yaw_change,
            cur_yaw,
            cur_pitch,
            final_u,
            final_v,
        )

    def _build_commanded_6d(
        self,
        du: float,
        dv: float,
        cur_yaw: float,
        cur_pitch: float,
        cur_y: float,
        cur_z: float,
    ) -> dict[str, float]:
        """Estimate the commanded 6D pose from pixel deltas (for bounds learning)."""
        delta = self._calculate_movement_for_pixel(du, dv)
        return {
            "yaw": cur_yaw + delta[5],
            "pitch": cur_pitch + delta[4],
            "y": cur_y + delta[1],
            "z": cur_z + delta[2],
            "roll": float(getattr(self, "roll", 0.0) or 0.0) + delta[3],
            "x": float(getattr(self, "x", 0.0) or 0.0) + delta[0],
        }

    def _record_pain_action_start(self, commanded_6d: dict[str, float], position: tuple[int, int]) -> None:
        """Record action start with the pain bridge before issuing the move."""
        pain_bridge = getattr(getattr(self, "_default_network", None), "_pain_bridge", None)
        if pain_bridge is None:
            return
        try:
            dyaw = abs(commanded_6d["yaw"])
            dpitch = abs(commanded_6d["pitch"])
            action_sig = f"look_at:dy={dyaw:.0f}:dp={dpitch:.0f}"
            pain_bridge.record_action_start(
                action_sig,
                context={"position": position, "commanded_6d": commanded_6d},
                target_yaw=commanded_6d.get("yaw"),
                target_pitch=commanded_6d.get("pitch"),
                target_y=commanded_6d.get("y"),
                target_z=commanded_6d.get("z"),
            )
        except Exception as e:
            self.log.debug("Pain action start recording failed: %s", e)

    def _enqueue_sdk_look_at(
        self,
        u: int,
        v: int,
        position: tuple[int, int],
        duration: float,
        *,
        perform_movement: bool,
        commanded_6d: dict[str, float] | None = None,
    ) -> None:
        """Scale internal pixel coords to the SDK resolution and enqueue the call."""
        sdk_scale_x = self._SDK_IMAGE_WIDTH / self._IMAGE_WIDTH
        sdk_scale_y = self._SDK_IMAGE_HEIGHT / self._IMAGE_HEIGHT
        sdk_u = int(u * sdk_scale_x)
        sdk_v = int(v * sdk_scale_y)

        if perform_movement:
            self.log.warning(
                "LOOK_AT_SDK: internal_pixel=(%d,%d) -> sdk_pixel=(%d,%d) scale=(%.2f,%.2f)",
                u,
                v,
                sdk_u,
                sdk_v,
                sdk_scale_x,
                sdk_scale_y,
            )

        kwargs: dict[str, Any] = {
            "duration": float(duration),
            "perform_movement": perform_movement,
            "_position_info": position,
        }
        if commanded_6d is not None:
            kwargs["_commanded_6d"] = commanded_6d

        # F1 retained-axes fold: this raw look_at_image bypasses
        # ReachyMiniController.look_at_pixel, so the controller's
        # last-commanded head stash never sees the motion. Wrap the
        # dispatch to invalidate the stash (duck-typed — SimulatedController
        # has none) so the controller's next goto_target re-seeds the head
        # axes from the readback instead of snapping back to a stale pose.
        # Invalidate BEFORE dispatch (conservative direction: a spurious
        # clear costs one readback seed; a missed clear leaves stale
        # re-commands) and only when the head will actually move.
        sdk_look_at = self.mini.look_at_image
        invalidate = getattr(getattr(self, "_robot", None), "note_external_head_motion", None)
        if perform_movement and invalidate is not None:

            def _look_at_with_invalidate(*args: Any, **kw: Any) -> Any:
                invalidate()
                return sdk_look_at(*args, **kw)

            self._enqueue_motor(_look_at_with_invalidate, sdk_u, sdk_v, **kwargs)
        else:
            self._enqueue_motor(sdk_look_at, sdk_u, sdk_v, **kwargs)

    def move(
        self,
        x: Optional[float] = None,
        y: Optional[float] = None,
        z: Optional[float] = None,
        roll: Optional[float] = None,
        pitch: Optional[float] = None,
        yaw: Optional[float] = None,
        duration: Optional[float] = None,
    ) -> None:
        """
        Docstring for move

        :param self: Description
        :param x: Description
        :type x: Optional[float]
        :param y: Description
        :type y: Optional[float]
        :param z: Description
        :type z: Optional[float]
        :param roll: Description
        :type roll: Optional[float]
        :param pitch: Description
        :type pitch: Optional[float]
        :param yaw: Description
        :type yaw: Optional[float]
        :param duration: Description
        :type duration: Optional[float]
        """

        # Update duration if specified
        if duration is not None:
            self.duration = duration

        # Execute head movement
        cur_x = float(getattr(self, "x", 0.0) or 0.0)
        cur_y = float(getattr(self, "y", 0.0) or 0.0)
        cur_z = float(getattr(self, "z", 0.0) or 0.0)
        cur_roll = float(getattr(self, "roll", 0.0) or 0.0)
        cur_pitch = float(getattr(self, "pitch", 0.0) or 0.0)
        cur_yaw = float(getattr(self, "yaw", 0.0) or 0.0)

        next_x = cur_x if x is None else float(x)
        next_y = cur_y if y is None else float(y)
        next_z = cur_z if z is None else float(z)
        next_roll = cur_roll if roll is None else float(roll)
        next_pitch = cur_pitch if pitch is None else float(pitch)
        next_yaw = cur_yaw if yaw is None else float(yaw)

        max_step = getattr(self, "_head_max_step", None)
        if isinstance(max_step, dict) and max_step:
            try:
                step = float(max_step.get("x", 0.0) or 0.0)
            except Exception:
                step = 0.0
            if step > 0:
                dx = next_x - cur_x
                if abs(dx) > step:
                    next_x = cur_x + (step if dx > 0 else -step)

            try:
                step = float(max_step.get("y", 0.0) or 0.0)
            except Exception:
                step = 0.0
            if step > 0:
                dy = next_y - cur_y
                if abs(dy) > step:
                    next_y = cur_y + (step if dy > 0 else -step)

            try:
                step = float(max_step.get("z", 0.0) or 0.0)
            except Exception:
                step = 0.0
            if step > 0:
                dz = next_z - cur_z
                if abs(dz) > step:
                    next_z = cur_z + (step if dz > 0 else -step)

            try:
                step = float(max_step.get("roll", 0.0) or 0.0)
            except Exception:
                step = 0.0
            if step > 0:
                droll = next_roll - cur_roll
                if abs(droll) > step:
                    next_roll = cur_roll + (step if droll > 0 else -step)

            try:
                step = float(max_step.get("pitch", 0.0) or 0.0)
            except Exception:
                step = 0.0
            if step > 0:
                dpitch = next_pitch - cur_pitch
                if abs(dpitch) > step:
                    next_pitch = cur_pitch + (step if dpitch > 0 else -step)

            try:
                step = float(max_step.get("yaw", 0.0) or 0.0)
            except Exception:
                step = 0.0
            if step > 0:
                dyaw = next_yaw - cur_yaw
                if abs(dyaw) > step:
                    next_yaw = cur_yaw + (step if dyaw > 0 else -step)

        # WORKSPACE CLAMP (2026-08-07 safety fold). This was the missing
        # guard on the MoveTool no-robot_id fallback: move() applied only the
        # per-call max-step limiter above — whose defaults are all 0.0, i.e.
        # DISABLED — and then enqueued the ABSOLUTE pose raw to the SDK. An
        # LLM-passed yaw of any magnitude went straight to the motors: the
        # "commanded a pose beyond its physical capability" failure class
        # from the 2026-08-07 hardware note (motors 2+3 destroyed).
        # move_relative() already clamps; move() must too.
        next_x, next_y, next_z, next_roll, next_pitch, next_yaw = self._clamp_to_workspace_6d(
            next_x, next_y, next_z, next_roll, next_pitch, next_yaw
        )

        if (
            next_x == cur_x
            and next_y == cur_y
            and next_z == cur_z
            and next_roll == cur_roll
            and next_pitch == cur_pitch
            and next_yaw == cur_yaw
        ):
            return

        self.x = float(next_x)
        self.y = float(next_y)
        self.z = float(next_z)
        self.roll = float(next_roll)
        self.pitch = float(next_pitch)
        self.yaw = float(next_yaw)

        # Track commanded 6D pose for bounds learning
        commanded_6d = {
            "yaw": self.yaw,
            "pitch": self.pitch,
            "y": self.y,
            "z": self.z,
            "roll": self.roll,
            "x": self.x,
        }

        self._enqueue_motor(
            move_head,
            self.mini,
            self.x,
            self.y,
            self.z,
            self.roll,
            self.pitch,
            self.yaw,
            self.duration,
            # Serialize against controller-routed callers (executor-lens
            # review fold): move_head's body-yaw read + compose + dispatch
            # runs under the same lock as the controller's goto_target.
            motion_lock=getattr(getattr(self, "_robot", None), "_motion_lock", None),
            # F1 retained-axes fold: move_head moves the head outside the
            # controller's last-commanded stash — thread the controller so
            # move_head can invalidate its head axes after dispatch.
            controller=getattr(self, "_robot", None),
            _commanded_6d=commanded_6d,
        )

    def move_relative(
        self,
        delta: tuple[float, float],
        duration: Optional[float] = None,
    ) -> None:
        """Move the head by a relative amount (used by DefaultNetwork scan actions).

        This uses TRANSLATION-FIRST approach: the delta primarily affects
        translation (Y for horizontal, Z for vertical), with a smaller
        component affecting rotation for personality.

        Args:
            delta: (dx, dy) tuple where dx affects horizontal and dy affects vertical.
                   Values are in "scaled units" - divide by 10 to get actual movement.
            duration: Movement duration in seconds.
        """
        if duration is None:
            duration = getattr(self, "duration", 0.5)

        dx, dy = delta

        # Convert from scaled units to actual movement amounts
        # Split between translation (primary) and rotation (secondary)
        scale = 1.0 / 10.0

        # Translation deltas (primary) - mm
        delta_y = float(dx) * scale * 2.0  # Horizontal: Y translation (2mm per unit)
        delta_z = float(dy) * scale * 2.0  # Vertical: Z translation

        # Rotation deltas (secondary) - degrees (smaller contribution)
        delta_yaw = float(dx) * scale * 0.5  # Horizontal: yaw rotation (0.5° per unit)
        delta_pitch = float(dy) * scale * 0.5  # Vertical: pitch rotation

        # Get current 6D position
        cur_x = float(getattr(self, "x", 0.0) or 0.0)
        cur_y = float(getattr(self, "y", 0.0) or 0.0)
        cur_z = float(getattr(self, "z", 0.0) or 0.0)
        cur_roll = float(getattr(self, "roll", 0.0) or 0.0)
        cur_pitch = float(getattr(self, "pitch", 0.0) or 0.0)
        cur_yaw = float(getattr(self, "yaw", 0.0) or 0.0)

        # Calculate new 6D position
        new_x = cur_x  # X (forward) unchanged
        new_y = cur_y + delta_y
        new_z = cur_z + delta_z
        new_roll = cur_roll  # Roll unchanged
        new_pitch = cur_pitch + delta_pitch
        new_yaw = cur_yaw + delta_yaw

        # Clamp to 6D workspace
        new_x, new_y, new_z, new_roll, new_pitch, new_yaw = self._clamp_to_workspace_6d(
            new_x, new_y, new_z, new_roll, new_pitch, new_yaw
        )

        self.log.debug(
            "move_relative 6D: delta=(%.1f, %.1f) -> trans(y=%.1f,z=%.1f) rot(yaw=%.1f,pitch=%.1f)",
            dx,
            dy,
            new_y,
            new_z,
            new_yaw,
            new_pitch,
        )

        # Execute the 6D movement
        self.move(x=new_x, y=new_y, z=new_z, roll=new_roll, pitch=new_pitch, yaw=new_yaw, duration=duration)

    def turn_around(
        self,
        angle: float,
        *,
        duration: float = 5.0,
        recenter_head: bool = True,
    ) -> None:
        """Rotate the body to bring a new area into view.

        This is used when the head is at its yaw limit and there's something
        interesting beyond what the head can see. The body rotates to bring
        that area into the camera's view.

        Args:
            angle: Degrees to rotate body. Positive = counterclockwise (left),
                   negative = clockwise (right).
            duration: Time for the rotation in seconds (default 5.0).
            recenter_head: If True, return head toward center during rotation.
        """

        self.log.info("turn_around: rotating body %.0f° over %.1fs (recenter_head=%s)", angle, duration, recenter_head)

        # Clear any pending movements from the queue
        # This prevents old look_at_image commands from executing after the turn
        cleared = self._clear_motor_queue()
        if cleared > 0:
            self.log.debug("Cleared %d queued movements before turn_around", cleared)

        # Inhibit DefaultNetwork to prevent new tracking commands during turn
        default_network = getattr(self, "_default_network", None)
        if default_network is not None:
            try:
                # Inhibit for the duration of the turn + buffer
                default_network.inhibit(duration=duration + 2.0)
                self.log.debug("Inhibited DefaultNetwork for %.1fs during turn_around", duration + 2.0)
            except Exception as inh_e:
                self.log.debug("Failed to inhibit DefaultNetwork: %s", inh_e)

        import time as time_mod

        from maxim.hardware.controller import MotionTarget

        try:
            # THE DISPATCH PATH IS THE FIX (2026-08-07 safety fold). This
            # method previously hand-rolled its 3-step sequence against the
            # raw SDK (`self.mini.goto_target(...)`), bypassing
            # ReachyMiniController.goto_target — and with it the head-frame
            # composition, the workspace clamps, and the motion lock. The
            # STEP-2 body rotation shipped `head=None`, so the daemon held
            # the head at its RETAINED world pose (centered at world 0 from
            # STEP 1) while the body rotated up to ±160° beneath it —
            # commanding a head-relative angle far past the ±65° neck
            # capability. That is the motor-destruction failure class from
            # the 2026-08-07 hardware note. Routing through the controller
            # gives: head rides along with the body (head-frame invariant),
            # every axis clamped to physical limits, and the read→compose→
            # dispatch TOCTOU closed by the controller's motion lock.
            robot = getattr(self, "_robot", None)
            if robot is None:
                self.log.warning("turn_around: no robot controller — skipping body rotation")
                return

            # Get ACTUAL current body yaw from the controller's pose read.
            try:
                pose = robot.get_current_pose() or {}
                current_body_yaw = math.degrees(float(pose["body_yaw"])) if "body_yaw" in pose else None
            except Exception:
                current_body_yaw = None
            if current_body_yaw is None:
                current_body_yaw = float(getattr(self, "body_yaw", 0.0) or 0.0)

            target_body_yaw = current_body_yaw + angle
            target_body_yaw = max(-160.0, min(160.0, target_body_yaw))

            self.log.info(
                "turn_around: 2-step sequence starting (body %.1f -> %.1f)", current_body_yaw, target_body_yaw
            )

            # ═══════════════════════════════════════════════════════════════════
            # STEP 1: Center the head RELATIVE TO THE BODY (prevents IK issues
            # during the turn). head_yaw on MotionTarget is body-relative, so
            # 0.0 means "straight ahead of the body", not world zero.
            # ═══════════════════════════════════════════════════════════════════
            step1_duration = min(2.0, duration * 0.3)
            self.log.info("turn_around STEP 1: centering head (%.1fs)", step1_duration)
            try:
                ok1 = robot.goto_target(
                    MotionTarget(head_roll=0.0, head_pitch=0.0, head_yaw=0.0, duration=step1_duration)
                )
                if not ok1:
                    self.log.warning("turn_around STEP 1: motion command rejected")
                time_mod.sleep(step1_duration + 0.3)  # Wait for completion + buffer
            except Exception as e1:
                self.log.warning("turn_around STEP 1 failed: %s", e1)

            # ═══════════════════════════════════════════════════════════════════
            # STEP 2: Rotate body. The controller composes a head matrix with
            # the body delta added (head-frame invariant), so the head — and
            # the camera/mics in it — ride along instead of counter-rotating.
            # ═══════════════════════════════════════════════════════════════════
            step2_duration = max(3.0, duration * 0.5)
            body_yaw_rad = math.radians(target_body_yaw)

            self.log.info("turn_around STEP 2: rotating body to %.1f° (%.1fs)", target_body_yaw, step2_duration)
            ok2 = False
            try:
                ok2 = robot.goto_target(MotionTarget(body_yaw=body_yaw_rad, duration=step2_duration))
                if not ok2:
                    self.log.warning("turn_around STEP 2: motion command rejected")
                time_mod.sleep(step2_duration + 0.3)  # Wait for completion + buffer
            except Exception as e2:
                self.log.warning("turn_around STEP 2 failed: %s", e2)

            # ═══════════════════════════════════════════════════════════════════
            # STEP 3: Settle and sync position
            # ═══════════════════════════════════════════════════════════════════
            self.log.info("turn_around STEP 3: settling and syncing")
            time_mod.sleep(0.5)  # Let everything settle

            # Sync with hardware to get accurate position
            self.sync_head_position()

            # Update internal tracking ONLY for a motion the controller
            # accepted (executor-lens review fold): a rejected/failed turn
            # must not leave the mirror asserting a rotation that never
            # happened — sync_head_position above already restored the
            # hardware truth, and overwriting it with the TARGET would
            # corrupt every frame consumer until the next sync.
            if ok2:
                self.body_yaw = target_body_yaw
                if recenter_head:
                    self.yaw = 0.0
                    self.pitch = 0.0
                    self.y = 0.0
                    self.z = 0.0

            self.log.info(
                "turn_around complete: body_yaw %.1f -> %.1f (total %.1fs)",
                current_body_yaw,
                target_body_yaw,
                step1_duration + step2_duration + 0.8,
            )

        except Exception as e:
            self.log.warning("turn_around failed: %s", e)
            import traceback

            self.log.debug("turn_around traceback: %s", traceback.format_exc())

        finally:
            # Release DefaultNetwork inhibition
            if default_network is not None:
                try:
                    default_network.release()
                    self.log.debug("Released DefaultNetwork inhibition after turn_around")
                except Exception:
                    pass

    def move_antenna(
        self,
        right: Optional[float] = None,
        left: Optional[float] = None,
        angle: Optional[float] = None,
        duration: Optional[float] = None,
        method: str = "minjerk",
        degrees: bool = True,
        relative: bool = False,
    ) -> None:
        if angle is not None:
            right = angle
            left = angle
        if duration is None:
            duration = self.duration

        self._enqueue_motor(
            move_antenna,
            self.mini,
            right=right,
            left=left,
            duration=duration,
            method=method,
            degrees=degrees,
            relative=relative,
        )
