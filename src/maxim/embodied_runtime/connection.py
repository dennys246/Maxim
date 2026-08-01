"""Reachy Mini connection management with automatic reconnection.

Handles robot connection lifecycle, failure tracking, graceful reconnection,
and runtime capability degradation/restoration via ConnectionState callbacks.
"""

from __future__ import annotations

import logging
import queue
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from maxim.utils.gpu_compat import is_connection_error
from maxim.utils.logging import warn

if TYPE_CHECKING:
    pass  # ReachyMini imported dynamically

logger = logging.getLogger(__name__)


class ConnectionState(Enum):
    """Robot connection lifecycle states."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"


@dataclass
class ConnectionConfig:
    """Connection configuration for Reachy Mini."""

    robot_name: str = "reachy_mini"
    timeout: float = 30.0
    media_backend: str = "default"
    localhost_only: bool = False  # Deprecated: use connection_mode instead
    connection_mode: str = "network"  # "auto", "localhost_only", or "network"
    spawn_daemon: bool = False
    use_sim: bool = False

    # Reconnection settings
    reconnect_cooldown_s: float = 20.0
    reconnect_window_s: float = 5.0

    # Failure thresholds before reconnect
    motor_failure_threshold: int = 3
    video_failure_threshold: int = 5
    audio_failure_threshold: int = 5


@dataclass
class FailureState:
    """Tracks connection failures for a subsystem."""

    count: int = 0
    last_ts: float = 0.0


@dataclass
class FailureTracker:
    """Track connection failures across subsystems.

    Triggers reconnection when failures exceed thresholds within
    a time window.
    """

    motor: FailureState = field(default_factory=FailureState)
    video: FailureState = field(default_factory=FailureState)
    audio: FailureState = field(default_factory=FailureState)

    thresholds: dict[str, int] = field(default_factory=lambda: {"motor": 3, "video": 5, "audio": 5})
    window_s: float = 5.0

    def record_failure(self, subsystem: str) -> bool:
        """Record a connection failure.

        Args:
            subsystem: One of "motor", "video", "audio".

        Returns:
            True if failure count reached threshold.
        """
        state = getattr(self, subsystem, None)
        if not isinstance(state, FailureState):
            return False

        now = time.time()

        # Reset counter if outside time window
        if now - state.last_ts > self.window_s:
            state.count = 0

        state.count += 1
        state.last_ts = now

        threshold = self.thresholds.get(subsystem, 3)
        return state.count >= threshold

    def reset(self) -> None:
        """Reset all failure counters."""
        self.motor = FailureState()
        self.video = FailureState()
        self.audio = FailureState()

    def reset_subsystem(self, subsystem: str) -> None:
        """Reset failure counter for a specific subsystem."""
        state = getattr(self, subsystem, None)
        if isinstance(state, FailureState):
            state.count = 0
            state.last_ts = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# ConnectionMixin — extracted from selfy.py for compartmentalization
# ─────────────────────────────────────────────────────────────────────────────


def _import_cv2():
    try:
        import cv2

        return cv2
    except ImportError:
        raise ImportError(
            "OpenCV is required for display features. Install with: pip install 'pymaxim[vision]'"
        ) from None


class ConnectionMixin:
    """Mixin providing connection management methods for the Maxim class.

    Handles CV2 resource release, media cleanup, connection failure tracking,
    and soft reconnection logic. Self-contained reconnection state machine.
    """

    def _release_cv2(self) -> None:
        try:
            cv2 = _import_cv2()
            cv2.destroyAllWindows()
            cv2.waitKey(1)
        except Exception:
            pass

    def _release_media(self) -> None:
        mini = getattr(self, "mini", None)
        if mini is None:
            return

        try:
            mini.media.close()
        except Exception as e:
            warn("Failed to close media: %s", e, logger=getattr(self, "log", None))

        self._release_cv2()

    def _reset_connection_failures(self) -> None:
        for state in self._connection_failures.values():
            try:
                state["count"] = 0
                state["last_ts"] = 0.0
            except Exception:
                continue

    def _note_connection_failure(self, kind: str, error: object) -> None:
        if not is_connection_error(error):
            if int(getattr(self, "verbosity", 0) or 0) >= 2:
                try:
                    self.log.debug("Ignoring non-connection error (%s): %s", kind, error)
                except Exception:
                    pass
            return
        stop_event = getattr(self, "_live_stop_event", None)
        if stop_event is not None and stop_event.is_set():
            if int(getattr(self, "verbosity", 0) or 0) >= 2:
                try:
                    self.log.debug("Ignoring connection failure during shutdown (%s): %s", kind, error)
                except Exception:
                    pass
            return
        requested_mode = str(getattr(self, "requested_mode", "") or "").strip().lower()
        if requested_mode:
            if int(getattr(self, "verbosity", 0) or 0) >= 2:
                try:
                    self.log.debug(
                        "Ignoring connection failure during mode switch (%s -> %s): %s",
                        str(getattr(self, "mode", "") or "").strip().lower(),
                        requested_mode,
                        error,
                    )
                except Exception:
                    pass
            return

        state = self._connection_failures.get(kind)
        if not isinstance(state, dict):
            state = {"count": 0, "last_ts": 0.0}
            self._connection_failures[kind] = state

        now = time.time()
        last_ts = float(state.get("last_ts") or 0.0)
        if now - last_ts > float(getattr(self, "_reconnect_window_s", 5.0) or 5.0):
            state["count"] = 0

        state["count"] = int(state.get("count", 0) or 0) + 1
        state["last_ts"] = now

        threshold = int(self._reconnect_thresholds.get(kind, 3) or 3)
        if int(getattr(self, "verbosity", 0) or 0) >= 2:
            try:
                self.log.debug(
                    "Connection failure (%s): count=%d threshold=%d window_s=%.1f error=%s",
                    kind,
                    int(state["count"]),
                    threshold,
                    float(getattr(self, "_reconnect_window_s", 5.0) or 5.0),
                    error,
                )
            except Exception:
                pass
        if state["count"] >= threshold:
            self._soft_reconnect(reason=f"{kind}_connection_failed", error=error)

    def _degrade_capabilities(self) -> None:
        """Mark robot capabilities as unavailable during connection outage.

        Called when reconnection fails.  The existing capability gates
        (CaptureManager, DefaultNetwork, tool registry, _compute_target_hz)
        will automatically adapt to headless behavior.
        """
        caps = getattr(self, "_capabilities", None)
        if caps is None:
            return
        caps.has_robot = False
        caps.has_motor = False
        caps.has_vision = False
        caps.has_audio = False
        try:
            self.log.warning("Capabilities degraded — robot unavailable")
        except Exception:
            pass

    def _restore_capabilities(self) -> None:
        """Restore robot capabilities after successful reconnection.

        Vision/audio are RE-DERIVED from the robot's actual media state
        (2026-08-01 fold) — the pre-fix unconditional ``True`` silently
        undid the no_media capability truth on every soft reconnect,
        re-enabling the capture loops against absent devices.
        """
        caps = getattr(self, "_capabilities", None)
        if caps is None:
            return
        from maxim.runtime.capabilities import derive_media_capabilities

        caps.has_robot = True
        caps.has_motor = True
        caps.has_vision, caps.has_audio = derive_media_capabilities(getattr(self, "_robot", None))
        try:
            self.log.info("Capabilities restored — robot reconnected")
        except Exception:
            pass

    def _soft_reconnect(self, *, reason: str, error: object | None = None) -> bool:
        if getattr(self, "_closed", False):
            return False
        stop_event = getattr(self, "_live_stop_event", None)
        if stop_event is not None and stop_event.is_set():
            return False

        now = time.time()
        last_ts = float(getattr(self, "_last_reconnect_ts", 0.0) or 0.0)
        cooldown_s = float(getattr(self, "_reconnect_cooldown_s", 20.0) or 20.0)
        if now - last_ts < cooldown_s:
            if int(getattr(self, "verbosity", 0) or 0) >= 2:
                try:
                    remaining = max(0.0, cooldown_s - (now - last_ts))
                    self.log.debug(
                        "Soft reconnect suppressed (cooldown %.1fs remaining): %s",
                        remaining,
                        reason,
                    )
                except Exception:
                    pass
            return False
        if not self._reconnect_lock.acquire(blocking=False):
            if int(getattr(self, "verbosity", 0) or 0) >= 2:
                try:
                    self.log.debug("Soft reconnect suppressed (lock busy): %s", reason)
                except Exception:
                    pass
            return False

        self._last_reconnect_ts = now
        try:
            if int(getattr(self, "verbosity", 0) or 0) >= 2:
                try:
                    self.log.debug("Soft reconnect begin: %s", reason)
                except Exception:
                    pass
            warn(
                "Soft reconnect requested (%s): %s",
                reason,
                error if error is not None else "unknown error",
                logger=self.log,
            )

            old_mini = getattr(self, "mini", None)
            if old_mini is not None:
                try:
                    old_mini.stop_recording()
                except Exception:
                    pass

                try:
                    media = getattr(old_mini, "media", None)
                    if media is not None:
                        media_lock = getattr(self, "_media_lock", None)
                        if media_lock is not None:
                            with media_lock:
                                media.close()
                        else:
                            media.close()
                except Exception as e:
                    warn("Failed to close media during reconnect: %s", e, logger=self.log)

                for attr in ("disconnect", "close", "shutdown"):
                    fn = getattr(old_mini, attr, None)
                    if callable(fn):
                        try:
                            fn()
                        except Exception:
                            pass

            # Use the RobotController's reconnect method
            if self._robot is None:
                warn("Soft reconnect failed (no robot controller)", logger=self.log)
                self._degrade_capabilities()
                return False

            # For simulation mode, just reset state
            if self._simulation:
                self.log.info("Simulation mode - skipping reconnect")
                return True

            # Attempt reconnection via the controller
            if not self._robot.reconnect(timeout=30.0, max_attempts=1):
                warn("Soft reconnect failed (controller reconnect failed)", logger=self.log)
                self._degrade_capabilities()
                return False

            try:
                self._robot.start_recording()
            except Exception as e:
                warn("Failed to start recording after reconnect: %s", e, logger=self.log)

            if getattr(self, "_woke_up", False):
                mode = str(getattr(self, "mode", "") or "").strip().lower()
                if mode != "sleep":
                    try:
                        self._robot.wake_up()
                    except Exception as e:
                        warn("Failed to wake Reachy after reconnect: %s", e, logger=self.log)

            motor_queue = getattr(self, "_motor_queue", None)
            if motor_queue is not None:
                try:
                    while True:
                        motor_queue.get_nowait()
                except queue.Empty:
                    pass

            self._reset_connection_failures()
            self._restore_capabilities()
            self.log.info("Soft reconnect complete.")
            return True
        finally:
            self._reconnect_lock.release()


__all__ = [
    "ConnectionConfig",
    "ConnectionMixin",
    "ConnectionState",
    "FailureState",
    "FailureTracker",
]
