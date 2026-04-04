"""Simple terminal spinner for simulation progress feedback."""

from __future__ import annotations

import sys
import threading
import time


_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]


class Spinner:
    """Non-blocking terminal spinner with status message.

    Usage:
        spinner = Spinner()
        spinner.start("Orchestrator thinking...")
        # ... long operation ...
        spinner.update("AUT processing probe...")
        # ... more work ...
        spinner.stop("Done — 3 actions recorded")
    """

    def __init__(self, prefix: str = "") -> None:
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._message = ""
        self._lock = threading.Lock()
        self._prefix = prefix

    def start(self, message: str = "") -> None:
        """Start the spinner with an initial message."""
        self._stop_event.clear()
        self._message = message
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def update(self, message: str) -> None:
        """Update the spinner message without stopping."""
        with self._lock:
            self._message = message

    def stop(self, final_message: str | None = None) -> None:
        """Stop the spinner and optionally print a final message."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None
        # Clear the spinner line
        sys.stderr.write("\r\033[K")
        sys.stderr.flush()
        if final_message:
            sys.stderr.write(f"  {self._prefix}{final_message}\n")
            sys.stderr.flush()

    def _spin(self) -> None:
        idx = 0
        start = time.time()
        while not self._stop_event.is_set():
            with self._lock:
                msg = self._message
            frame = _FRAMES[idx % len(_FRAMES)]
            elapsed = int(time.time() - start)
            sys.stderr.write(f"\r\033[K  {self._prefix}{frame} {msg} ({elapsed}s)")
            sys.stderr.flush()
            idx += 1
            self._stop_event.wait(0.1)
