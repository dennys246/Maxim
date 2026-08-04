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

Credit contract (Phase 1): NONE. The ``azimuth`` claim stays with the
DoAFeed; no relief credit is booked for real turns until the Phase 2
measured-credit slice (post-motion DoA re-read) ships with its own review
round. See docs/plans/sem_motor_binding.md.

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

            self._inhibit_dn(duration_s)
            ok = robot.goto_target(MotionTarget(body_yaw=target_body, duration=duration_s))
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

            # Unknown readback stays optimistic (unknown != failed);
            # a CONFIRMED shortfall is a real failure the learning
            # chain should see.
            success = reached is not False
            return self._result(
                affordance,
                params,
                success=success,
                error=None if success else "Body turn fell short (see metadata)",
                metadata={
                    "commanded_body_yaw_deg": round(math.degrees(target_body), 1),
                    "achieved_body_yaw_deg": (
                        round(math.degrees(achieved_body), 1) if achieved_body is not None else None
                    ),
                    "reached": reached,
                    "clamped_to_body_limit": clamped,
                },
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
