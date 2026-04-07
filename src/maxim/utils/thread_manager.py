"""Thread registry for coordinated shutdown.

Provides a central registry for tracking threads created by Maxim,
enabling graceful shutdown with force-kill fallback for unresponsive threads.
"""

from __future__ import annotations

import ctypes
import logging
import threading
from typing import Callable

logger = logging.getLogger(__name__)


class ThreadRegistry:
    """Central registry for thread lifecycle management.

    Tracks all threads created by a component, enabling:
    - Coordinated shutdown with configurable timeout
    - Force termination of unresponsive threads
    - Thread status monitoring

    Example:
        registry = ThreadRegistry()

        # Register threads
        registry.register("worker", worker_thread)

        # On shutdown
        failed = registry.stop_all(timeout=5.0)
        if failed:
            logger.warning("Threads failed to stop: %s", failed)
    """

    def __init__(self, log: logging.Logger | None = None) -> None:
        """Initialize thread registry.

        Args:
            log: Logger instance. Defaults to module logger.
        """
        self._registry: dict[str, threading.Thread] = {}
        self._lock = threading.Lock()
        self._log = log or logger

    def register(self, name: str, thread: threading.Thread) -> None:
        """Register a thread for coordinated shutdown tracking.

        Args:
            name: Unique identifier for this thread.
            thread: The thread to register.
        """
        with self._lock:
            self._registry[name] = thread

    def unregister(self, name: str) -> None:
        """Remove a thread from the registry.

        Args:
            name: The thread identifier to remove.
        """
        with self._lock:
            self._registry.pop(name, None)

    def get(self, name: str) -> threading.Thread | None:
        """Get a thread by name.

        Args:
            name: Thread identifier.

        Returns:
            Thread if found, None otherwise.
        """
        with self._lock:
            return self._registry.get(name)

    def is_alive(self, name: str) -> bool:
        """Check if a thread is alive.

        Args:
            name: Thread identifier.

        Returns:
            True if thread exists and is alive.
        """
        thread = self.get(name)
        return thread is not None and thread.is_alive()

    def list_threads(self) -> list[str]:
        """List all registered thread names.

        Returns:
            List of thread names.
        """
        with self._lock:
            return list(self._registry.keys())

    def list_alive(self) -> list[str]:
        """List names of threads that are currently alive.

        Returns:
            List of alive thread names.
        """
        with self._lock:
            return [name for name, thread in self._registry.items() if thread is not None and thread.is_alive()]

    def stop_all(self, timeout: float = 10.0) -> list[str]:
        """Stop all registered threads with force-kill fallback.

        Attempts graceful join first, then uses ctypes to force-terminate
        threads that don't respond within timeout.

        Args:
            timeout: Total timeout for stopping all threads.

        Returns:
            List of thread names that could not be stopped.
        """
        with self._lock:
            threads = list(self._registry.items())

        if not threads:
            return []

        # Calculate per-thread timeout
        per_thread_timeout = timeout / max(len(threads), 1)
        failed_threads = []

        for name, thread in threads:
            if thread is None or not thread.is_alive():
                continue

            # Try graceful join
            try:
                thread.join(timeout=per_thread_timeout)
            except Exception:
                pass

            if thread.is_alive():
                # Try force termination
                if self._force_terminate(thread, name):
                    continue

                failed_threads.append(name)
                self._log.warning("Could not stop registered thread '%s'", name)

        # Clear registry
        with self._lock:
            self._registry.clear()

        return failed_threads

    def _force_terminate(self, thread: threading.Thread, name: str) -> bool:
        """Force-terminate a thread using ctypes.

        Args:
            thread: Thread to terminate.
            name: Thread name for logging.

        Returns:
            True if successfully terminated.
        """
        try:
            thread_id = thread.ident
            if thread_id is None:
                return False

            # Raise SystemExit in the target thread
            res = ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(thread_id), ctypes.py_object(SystemExit))

            if res == 1:
                thread.join(timeout=0.5)
                if not thread.is_alive():
                    self._log.info("Force-terminated thread '%s'", name)
                    return True
        except Exception:
            pass

        return False

    def clear(self) -> None:
        """Clear the registry without stopping threads."""
        with self._lock:
            self._registry.clear()

    def __len__(self) -> int:
        """Number of registered threads."""
        with self._lock:
            return len(self._registry)

    def __contains__(self, name: str) -> bool:
        """Check if a thread is registered."""
        with self._lock:
            return name in self._registry


def create_daemon_thread(
    target: Callable[..., None],
    name: str | None = None,
    args: tuple = (),
    kwargs: dict | None = None,
) -> threading.Thread:
    """Create a daemon thread with standard configuration.

    Daemon threads are automatically terminated when the main process exits.

    Args:
        target: Callable to run in thread.
        name: Thread name.
        args: Positional arguments for target.
        kwargs: Keyword arguments for target.

    Returns:
        Configured (but not started) daemon thread.
    """
    return threading.Thread(
        target=target,
        name=name,
        args=args,
        kwargs=kwargs or {},
        daemon=True,
    )


__all__ = [
    "ThreadRegistry",
    "create_daemon_thread",
]
