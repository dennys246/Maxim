"""Reachy orient motor backend — SEM affordances become real body turns.

sem_motor_binding.md Phase 1: the ``orient`` modulator's spec-declared
affordances (``turn_left`` / ``turn_right`` / ``_big`` in
``bodies/reachy_mini.yaml``) dispatch a REAL body rotation on live via
``ReachyMiniController.goto_target`` — the head rides along automatically
(the controller composes the retained body-relative head yaw onto the
TARGET body yaw; the head-frame invariant machinery). The affordance's own
declared ``self_effect["head_yaw"]`` IS the motor command (radians,
+yaw = LEFT), so retuning the YAML retunes model and motor together.

Honesty contract (verify-actuation discipline):
- the pose is read back after the blocking goto; ``success`` reflects
  dispatch-accepted AND not-confirmed-short (unknown readback stays
  optimistic — unknown != failed);
- a rejected motion is a REAL failure (``ModulatorResult(success=False)``
  → ``ToolOutput(success=False)``), ending the stub path's success lie;
- the backend world-owns ``head_yaw`` (declared via
  :attr:`world_owned_sensors`, unioned into
  ``Embodiment.live_world_set_sensors`` by ``build_executor``) and writes
  the MEASURED body-relative head yaw back into the entity's
  ``vital_metrics`` — the modeled self_effect write is filtered, killing
  the SEM sensor drift that shipped with the virtual-turn era.

Credit contract (Phase 2, measured): the ``azimuth`` claim stays with
the DoAFeed (single writer); relief credit comes from a MEASURED
before/after pair — the feed's reading at execute entry vs the first
reading stamped after the motion settles. Timeout or staleness → no
credit (never the modeled delta). The backend only MEASURES; the credit
formula (drive_comfort_progress) and emission stay in the bio layer
(tool_bridge reads ``metadata["measured_drive_transitions"]``). See
docs/plans/sem_motor_binding.md Phase 2.

DN contention: the DefaultNetwork runs on its own thread and its gaze
behaviors would fight a 1.5-3 s body turn — the backend inhibits the DN
and clears the motor queue around the goto (the ``turn_around`` defense),
then syncs ``maxim.yaw``/``maxim.body_yaw`` immediately (the periodic 3 s
sync is too slow for consumers reading the frame right after the turn).
"""

from __future__ import annotations

import logging
import math
from typing import Any

logger = logging.getLogger(__name__)

# Daemon-side body yaw clamp (matches embodied_runtime/movement.py's
# turn_around clamp; the daemon's inverse_kinematics_safe uses ±160°).
_MAX_BODY_YAW_RAD = math.radians(160.0)
# Post-motion readback tolerance: |achieved − target| within this counts
# as reached (minjerk settling + pose-estimate noise).
_REACH_TOLERANCE_RAD = math.radians(5.0)
# Phase 2 measured relief credit (sem_motor_binding.md): the "before"
# azimuth reading must be recent enough to still describe the sound the
# turn is answering (the feed holds the last value through silence).
_MEASURE_BEFORE_MAX_AGE_S = 10.0
# The chip converges in ~0.23 s after motion; readings stamped earlier
# than t_end + settle are mid-rotation garbage.
_MEASURE_SETTLE_S = 0.3
# How long to wait for a post-motion reading before giving up. A silent
# room yields NO credit (sparsity is acceptable; fabricated sign is not —
# the timeout NEVER falls back to the modeled delta). Must exceed the
# feed's worst case for a fully-post-settle SAMPLE WINDOW: the in-flight
# gated_azimuth cycle (<= sample_timeout_s 2.0 s) + one clean cycle
# (<= 2.0 s) + settle (review F3 — 2.0 s systematically missed under
# sparse speech once the window gate landed).
_MEASURE_TIMEOUT_S = 4.5


class ReachyOrientMotorBackend:
    """Modulator backend mapping orient affordances to real body turns."""

    # Sensors this backend world-owns on live: the modeled self_effect on
    # these keys must not write (the backend's measured readback is the
    # single writer). build_executor unions these into
    # Embodiment.live_world_set_sensors at construction.
    world_owned_sensors: tuple[str, ...] = ("head_yaw", "body_yaw")

    def __init__(
        self,
        *,
        robot: Any,
        maxim: Any = None,
        entity: Any = None,
        deltas: dict[str, float],
        modulator_name: str,
        entity_name: str,
    ) -> None:
        self._robot = robot
        self._maxim = maxim
        self._entity = entity
        self._embodiment: Any = None  # bound by build_executor post-construction
        # Monotonic end-time of this backend's LAST motion — a before-
        # reading whose sample window overlaps it is frame-mixed and
        # uncorrectable (review F2's repeat-turn wrong-sign path).
        self._last_motion_t_end: float = 0.0
        self._deltas = dict(deltas)  # affordance -> signed body-yaw delta (rad)
        self._modulator_name = modulator_name
        self._entity_name = entity_name

    def bind_embodiment(self, embodiment: Any) -> None:
        """Called by ``build_executor`` after the Embodiment wrapper exists —
        the backend attaches to the raw entity BEFORE the wrapper is
        constructed, but the measured world-set goes through the canonical
        ``world_set_axis(embodiment, ...)`` API."""
        self._embodiment = embodiment

    # ── helpers ──────────────────────────────────────────────────────────

    def _result(self, affordance: str, params: dict[str, Any], **kw: Any) -> Any:
        from maxim.embodiment.sem import ModulatorResult

        return ModulatorResult(
            modulator_name=self._modulator_name,
            entity_name=self._entity_name,
            affordance=affordance,
            params=params,
            **kw,
        )

    def _inhibit_dn(self, duration_s: float) -> None:
        """Best-effort: stop DN gaze behaviors from fighting the turn."""
        maxim = self._maxim
        if maxim is None:
            return
        try:
            clear = getattr(maxim, "_clear_motor_queue", None)
            if callable(clear):
                clear()
        except Exception:
            logger.debug("motor backend: motor-queue clear failed", exc_info=True)
        try:
            dn = getattr(maxim, "_default_network", None)
            if dn is not None and hasattr(dn, "inhibit"):
                dn.inhibit(duration=duration_s + 2.0)
        except Exception:
            logger.debug("motor backend: DN inhibit failed", exc_info=True)

    def _world_set_measured(self, pose: dict[str, float]) -> None:
        """Write the MEASURED body-relative head yaw into the entity sensor
        via the canonical ``world_set_axis`` API (clamps to the sensor's
        declared range; no-op for a body without the sensor)."""
        if self._embodiment is None or "yaw" not in pose:
            return
        try:
            from maxim.embodiment.audio_localization import world_set_axis

            body = float(pose.get("body_yaw", 0.0) or 0.0)
            head_rel = float(pose["yaw"]) - body
            world_set_axis(self._embodiment, "head_yaw", head_rel)
            if "body_yaw" in pose:
                world_set_axis(self._embodiment, "body_yaw", body)
        except Exception:
            logger.debug("motor backend: measured world-set failed", exc_info=True)

    def _read_azimuth(self) -> "tuple[float, float, float | None, float | None, float | None] | None":
        """(az, ts, capture_head_yaw_deg, capture_body_yaw_deg, window_start) or None.

        The capture-frame stamps and sample-window start are the honesty
        payload (review F1/F2): the STAMP trails the samples by up to a
        full gated_azimuth cycle, and the azimuth is head-relative in its
        CAPTURE frame — both must travel with the value or the gates
        downstream test the wrong thing.
        """
        try:
            feed = getattr(self._maxim, "_doa_feed", None)
            latest = getattr(feed, "latest", None) if feed is not None else None
            if latest is None:
                return None
            cap_head = latest[2] if len(latest) > 2 and latest[2] is not None else None
            cap_body = latest[3] if len(latest) > 3 and latest[3] is not None else None
            win = float(latest[4]) if len(latest) > 4 and latest[4] is not None else None
            return (
                float(latest[0]),
                float(latest[1]),
                float(cap_head) if cap_head is not None else None,
                float(cap_body) if cap_body is not None else None,
                win,
            )
        except Exception:
            logger.debug("motor backend: azimuth read failed", exc_info=True)
            return None

    def _frame_corrected_before(
        self,
        reading: "tuple[float, float, float | None, float | None, float | None] | None",
        entry_world_yaw_deg: float,
    ) -> "float | None":
        """The before-azimuth expressed in the TURN-ENTRY head frame, or None.

        Azimuth is head-relative in its CAPTURE frame (review F2: an
        age-only gate let a pre-previous-turn reading serve as the
        baseline — the frame had rotated, minting wrong-SIGN credit on the
        direction-bearing cluster). Correction: +yaw = LEFT, azimuth
        +1 = RIGHT, so a head that rotated left by d° since capture sees
        the same fixed source d/90 further right:
        ``az_entry = az_capture + (entry_world − capture_world) / 90``.
        Discards (None) when: no reading; older than the staleness gate;
        capture stamps missing (uncorrectable); or the sample window
        overlaps this backend's own previous motion (frame-mixed median —
        uncorrectable). Saturation clamp attenuates, never flips sign.
        """
        if reading is None:
            return None
        az_cap, ts, cap_head, cap_body, win_start = reading
        import time as _time

        if (_time.monotonic() - ts) > _MEASURE_BEFORE_MAX_AGE_S:
            return None
        if cap_head is None or cap_body is None:
            return None
        if win_start is None or win_start <= self._last_motion_t_end:
            return None
        cap_world = cap_head + cap_body
        corrected = az_cap + (entry_world_yaw_deg - cap_world) / 90.0
        return max(-1.0, min(1.0, corrected))

    def _measure_azimuth_after(self, t_end: float) -> "float | None":
        """First reading whose SAMPLE WINDOW began past ``t_end + settle``.

        Gating on the stamp is wrong (review F1): the feed stamps AFTER
        gated_azimuth returns, up to a full cycle later than the samples —
        stamp-gating systematically preferred the mid-rotation cycle. A
        silent room times out → None; credit SPARSITY is acceptable, a
        fabricated or modeled sign is not. The executor serializes actions
        and the DN is inhibited, so no other motion contaminates a window
        that begins post-settle.
        """
        import time as _time

        deadline = t_end + _MEASURE_TIMEOUT_S
        while _time.monotonic() < deadline:
            after = self._read_azimuth()
            if after is not None and after[4] is not None and after[4] > t_end + _MEASURE_SETTLE_S:
                return after[0]
            _time.sleep(0.1)
        return None

    # ── the contract surface ─────────────────────────────────────────────

    def execute(self, affordance: str, params: dict[str, Any]) -> Any:
        delta = self._deltas.get(affordance)
        if delta is None:
            return self._result(
                affordance, params, success=False, error=f"No motor mapping for affordance: {affordance}"
            )

        robot = self._robot
        if robot is None or not getattr(robot, "is_connected", lambda: False)():
            return self._result(affordance, params, success=False, error="Robot controller not connected")

        try:
            from maxim.hardware import MotionTarget

            pose = robot.get_current_pose() or {}
            if "body_yaw" not in pose:
                # Refusing to guess (review fold E2): the controller's joint
                # read is best-effort — assuming body=0 when it fails would
                # command a swing of up to the full body angle at a duration
                # computed for a 17-52 deg step. An unverifiable PRE-state is
                # as disqualifying as an unverified post-state.
                return self._result(
                    affordance,
                    params,
                    success=False,
                    error="Body pose unreadable — refusing to guess the current body angle",
                )
            current_body = float(pose["body_yaw"])
            target_body = max(-_MAX_BODY_YAW_RAD, min(_MAX_BODY_YAW_RAD, current_body + delta))
            clamped = abs(current_body + delta) > _MAX_BODY_YAW_RAD
            # Duration scales with the swing (~1.5 s for a normal 17°
            # step, ~2.6 s for a big 52° step). The SDK goto blocks.
            duration_s = min(3.0, max(1.0, 1.0 + abs(math.degrees(delta)) / 30.0))

            # Phase 2 measured credit: the BEFORE reading is the sound this
            # turn answers, expressed in the TURN-ENTRY frame (frame-
            # corrected via the capture stamps; discarded when stale,
            # uncorrectable, or frame-mixed with our own previous motion).
            import time as _time

            entry_world_deg = math.degrees(float(pose["yaw"])) if "yaw" in pose else None
            az_before = (
                self._frame_corrected_before(self._read_azimuth(), entry_world_deg)
                if entry_world_deg is not None
                else None
            )

            self._inhibit_dn(duration_s)
            ok = robot.goto_target(MotionTarget(body_yaw=target_body, duration=duration_s))
            t_end = _time.monotonic()
            if not ok:
                return self._result(affordance, params, success=False, error="Motion command rejected by controller")

            # Post-motion: sync the runtime mirror NOW (the periodic 3 s
            # DN sync is too slow for the DoA stamp / focus_on_sound), then
            # read the frame back.
            maxim = self._maxim
            try:
                sync = getattr(maxim, "sync_head_position", None)
                if callable(sync):
                    sync()
            except Exception:
                logger.debug("motor backend: post-motion sync failed", exc_info=True)

            achieved_body: float | None = None
            reached: bool | None = None
            try:
                pose2 = robot.get_current_pose() or {}
                if "body_yaw" in pose2:
                    achieved_body = float(pose2["body_yaw"])
                    reached = abs(achieved_body - target_body) <= _REACH_TOLERANCE_RAD
                self._world_set_measured(pose2)
            except Exception:
                logger.debug("motor backend: pose readback failed", exc_info=True)

            logger.info(
                "motor turn %s: body %.1f° → %.1f°%s%s",
                affordance,
                math.degrees(current_body),
                math.degrees(target_body),
                " [clamped]" if clamped else "",
                (
                    f" achieved {math.degrees(achieved_body):.1f}°" + ("" if reached else " [FELL SHORT]")
                    if achieved_body is not None
                    else ""
                ),
            )

            # Phase 2: measured azimuth transition (None on timeout/
            # staleness — never fabricated). Only meaningful when the
            # motion actually happened; a confirmed-short turn still
            # rotated PART way, so the measurement stands for it too.
            transition: "tuple[float, float] | None" = None
            if az_before is not None:
                az_after = self._measure_azimuth_after(t_end)
                if az_after is not None:
                    transition = (az_before, az_after)
            self._last_motion_t_end = t_end

            # Unknown readback stays optimistic (unknown != failed);
            # a CONFIRMED shortfall is a real failure the learning
            # chain should see.
            success = reached is not False
            metadata: dict[str, Any] = {
                "commanded_body_yaw_deg": round(math.degrees(target_body), 1),
                "achieved_body_yaw_deg": (round(math.degrees(achieved_body), 1) if achieved_body is not None else None),
                "reached": reached,
                "clamped_to_body_limit": clamped,
            }
            if transition is not None:
                # Consumed by ModulatorAffordanceTool.execute: the bio
                # layer owns the credit formula (drive_comfort_progress);
                # the backend only measures.
                metadata["measured_drive_transitions"] = {"azimuth": transition}
            return self._result(
                affordance,
                params,
                success=success,
                error=None if success else "Body turn fell short (see metadata)",
                metadata=metadata,
            )
        except Exception as e:
            logger.warning("motor turn %s failed: %s", affordance, e)
            return self._result(affordance, params, success=False, error=str(e))


def make_reachy_orient_factory(robot: Any, maxim: Any = None) -> Any:
    """``attach_backends``-shaped factory binding ONLY the ``orient`` modulator.

    ``(entity, mod_name, spec_modulator) -> backend | None`` — every other
    modulator keeps stub semantics. The body-yaw delta per affordance is
    read from the affordance's declared ``self_effect["head_yaw"]`` so the
    YAML stays the single source of magnitude truth.
    """

    def factory(entity: Any, mod_name: str, spec_modulator: Any) -> Any:
        if mod_name != "orient":
            return None
        deltas: dict[str, float] = {}
        for aff_name, schema in (getattr(spec_modulator, "_affordances", None) or {}).items():
            se = getattr(schema, "self_effect", None) or {}
            if "head_yaw" in se:
                try:
                    deltas[aff_name] = float(se["head_yaw"])
                except (TypeError, ValueError):
                    continue
        if not deltas:
            return None
        return ReachyOrientMotorBackend(
            robot=robot,
            maxim=maxim,
            entity=entity,
            deltas=deltas,
            modulator_name=mod_name,
            entity_name=getattr(entity, "name", "") or "",
        )

    return factory
