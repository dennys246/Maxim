"""Centralized registry for periodic signal monitors."""

from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod
from typing import Callable

from maxim.proprioception.pain import PainSignal

logger = logging.getLogger(__name__)


class SignalMonitor(ABC):
    """Abstract base class for periodic signal monitors.

    Subclasses implement ``check()`` which is polled at a fixed interval.
    If a monitor returns a PainSignal, it is forwarded to all registered
    callbacks (typically the PainDetector or ToolPainBridge).
    """

    name: str = "base"

    @abstractmethod
    def check(self) -> PainSignal | None:
        """Check for a signal condition.

        Returns:
            PainSignal if the condition is detected, None otherwise.
        """
        ...


class MonitorRegistry:
    """Polls registered SignalMonitors on a background thread.

    Example:
        registry = MonitorRegistry(poll_interval=0.5)
        registry.register(ToolTimeoutMonitor(executor))
        registry.add_signal_callback(pain_detector._emit_pain)
        registry.start()
    """

    def __init__(self, poll_interval: float = 0.5) -> None:
        self._monitors: list[SignalMonitor] = []
        self._poll_interval = poll_interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._callbacks: list[Callable[[PainSignal], None]] = []

    def register(self, monitor: SignalMonitor) -> None:
        """Register a signal monitor for polling."""
        self._monitors.append(monitor)

    def add_signal_callback(self, callback: Callable[[PainSignal], None]) -> None:
        """Register a callback invoked when any monitor fires."""
        self._callbacks.append(callback)

    def start(self) -> None:
        """Start the background polling thread."""
        self._stop.clear()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True, name="monitor_registry")
        self._thread.start()

    def stop(self) -> None:
        """Stop the background polling thread."""
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def _poll_loop(self) -> None:
        """Internal polling loop executed on the background thread."""
        while not self._stop.is_set():
            for monitor in self._monitors:
                try:
                    signal = monitor.check()
                    if signal is not None:
                        for cb in self._callbacks:
                            cb(signal)
                except Exception:
                    logger.debug("Monitor %s check failed", monitor.name, exc_info=True)
            self._stop.wait(timeout=self._poll_interval)


__all__ = ["MonitorRegistry", "SignalMonitor"]
