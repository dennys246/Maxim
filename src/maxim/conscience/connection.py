"""Reachy Mini connection management with automatic reconnection.

Handles robot connection lifecycle, failure tracking, and graceful reconnection.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from maxim.utils.gpu_compat import is_connection_error
from maxim.utils.logging import warn

if TYPE_CHECKING:
    pass  # ReachyMini imported dynamically

logger = logging.getLogger(__name__)


@dataclass
class ConnectionConfig:
    """Connection configuration for Reachy Mini."""

    robot_name: str = "reachy_mini"
    timeout: float = 30.0
    media_backend: str = "default"
    localhost_only: bool = False
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

    thresholds: dict[str, int] = field(
        default_factory=lambda: {"motor": 3, "video": 5, "audio": 5}
    )
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


class ReachyConnection:
    """Manages ReachyMini connection with automatic reconnection.

    Tracks connection failures and automatically reconnects when
    failure thresholds are exceeded.

    Example:
        conn = ReachyConnection(config)
        mini = conn.connect()

        # In error handler
        if conn.should_reconnect("motor", error):
            conn.reconnect()
    """

    def __init__(
        self,
        config: ConnectionConfig | None = None,
        log: logging.Logger | None = None,
    ) -> None:
        """Initialize connection manager.

        Args:
            config: Connection configuration.
            log: Logger instance.
        """
        self.config = config or ConnectionConfig()
        self._log = log or logger
        self._mini: Any = None
        self._media_lock: threading.Lock | None = None
        self._reconnect_lock = threading.Lock()
        self._last_reconnect_ts = 0.0
        self._woke_up = False
        self._closed = False

        # Build thresholds from config
        self._failures = FailureTracker(
            thresholds={
                "motor": self.config.motor_failure_threshold,
                "video": self.config.video_failure_threshold,
                "audio": self.config.audio_failure_threshold,
            },
            window_s=self.config.reconnect_window_s,
        )

        # Callbacks
        self._on_reconnect: list[Callable[[], None]] = []

    @property
    def mini(self) -> Any:
        """The current ReachyMini instance."""
        return self._mini

    @property
    def is_connected(self) -> bool:
        """Check if connected to robot."""
        return self._mini is not None

    def set_media_lock(self, lock: threading.Lock) -> None:
        """Set the media lock for thread-safe media access."""
        self._media_lock = lock

    def on_reconnect(self, callback: Callable[[], None]) -> None:
        """Register a callback to be called after reconnection."""
        self._on_reconnect.append(callback)

    def connect(self, start_recording: bool = True) -> Any:
        """Establish connection to Reachy Mini.

        Args:
            start_recording: Whether to start recording after connect.

        Returns:
            ReachyMini instance.
        """
        from reachy_mini import ReachyMini

        self._mini = ReachyMini(
            robot_name=self.config.robot_name,
            localhost_only=self.config.localhost_only,
            spawn_daemon=self.config.spawn_daemon,
            use_sim=self.config.use_sim,
            timeout=self.config.timeout,
            media_backend=self.config.media_backend,
        )

        if start_recording:
            try:
                self._mini.start_recording()
            except Exception as e:
                self._log.warning("Failed to start recording: %s", e)

        return self._mini

    def disconnect(self) -> None:
        """Disconnect from robot."""
        self._closed = True
        if self._mini is None:
            return

        try:
            self._mini.stop_recording()
        except Exception:
            pass

        # Close media
        try:
            media = getattr(self._mini, "media", None)
            if media is not None:
                if self._media_lock is not None:
                    with self._media_lock:
                        media.close()
                else:
                    media.close()
        except Exception as e:
            warn("Failed to close media: %s", e, logger=self._log)

        # Try various disconnect methods
        for attr in ("disconnect", "close", "shutdown"):
            fn = getattr(self._mini, attr, None)
            if callable(fn):
                try:
                    fn()
                except Exception:
                    pass

        self._mini = None

    def note_failure(
        self,
        subsystem: str,
        error: object,
        *,
        stop_event: threading.Event | None = None,
    ) -> bool:
        """Note a connection failure, potentially triggering reconnect.

        Args:
            subsystem: One of "motor", "video", "audio".
            error: The error that occurred.
            stop_event: If set, suppresses reconnection during shutdown.

        Returns:
            True if reconnection was triggered.
        """
        # Ignore non-connection errors
        if not is_connection_error(error):
            return False

        # Ignore during shutdown
        if stop_event is not None and stop_event.is_set():
            return False

        if self._closed:
            return False

        # Record failure
        if self._failures.record_failure(subsystem):
            return self.reconnect(
                reason=f"{subsystem}_connection_failed",
                error=error,
            )

        return False

    def reconnect(
        self,
        *,
        reason: str = "manual",
        error: object | None = None,
        motor_queue: "queue.Queue[Any] | None" = None,
    ) -> bool:
        """Attempt reconnection to robot.

        Args:
            reason: Reason for reconnection (for logging).
            error: Error that triggered reconnection.
            motor_queue: If provided, clear pending motor commands.

        Returns:
            True if reconnection succeeded.
        """
        if self._closed:
            return False

        now = time.time()
        if now - self._last_reconnect_ts < self.config.reconnect_cooldown_s:
            return False

        if not self._reconnect_lock.acquire(blocking=False):
            return False

        self._last_reconnect_ts = now
        try:
            warn(
                "Reconnecting (%s): %s",
                reason,
                error if error is not None else "unknown error",
                logger=self._log,
            )

            # Close old connection
            old_mini = self._mini
            if old_mini is not None:
                try:
                    old_mini.stop_recording()
                except Exception:
                    pass

                try:
                    media = getattr(old_mini, "media", None)
                    if media is not None:
                        if self._media_lock is not None:
                            with self._media_lock:
                                media.close()
                        else:
                            media.close()
                except Exception as e:
                    warn("Failed to close media during reconnect: %s", e, logger=self._log)

                for attr in ("disconnect", "close", "shutdown"):
                    fn = getattr(old_mini, attr, None)
                    if callable(fn):
                        try:
                            fn()
                        except Exception:
                            pass

            # Connect new
            try:
                from reachy_mini import ReachyMini

                self._mini = ReachyMini(
                    robot_name=self.config.robot_name,
                    localhost_only=self.config.localhost_only,
                    spawn_daemon=self.config.spawn_daemon,
                    use_sim=self.config.use_sim,
                    timeout=self.config.timeout,
                    media_backend=self.config.media_backend,
                )
            except Exception as e:
                warn("Reconnect failed: %s", e, logger=self._log)
                return False

            try:
                self._mini.start_recording()
            except Exception as e:
                warn("Failed to start recording after reconnect: %s", e, logger=self._log)

            # Re-wake if needed
            if self._woke_up:
                try:
                    self._mini.wake_up()
                except Exception as e:
                    warn("Failed to wake after reconnect: %s", e, logger=self._log)

            # Clear motor queue
            if motor_queue is not None:
                try:
                    while True:
                        motor_queue.get_nowait()
                except queue.Empty:
                    pass

            # Reset failure counters
            self._failures.reset()

            # Notify callbacks
            for callback in self._on_reconnect:
                try:
                    callback()
                except Exception:
                    pass

            self._log.info("Reconnect complete.")
            return True

        finally:
            self._reconnect_lock.release()

    def mark_woke_up(self) -> None:
        """Mark that the robot has been woken up (for reconnect)."""
        self._woke_up = True

    def mark_sleeping(self) -> None:
        """Mark that the robot is sleeping."""
        self._woke_up = False


__all__ = [
    "ConnectionConfig",
    "FailureState",
    "FailureTracker",
    "ReachyConnection",
]
