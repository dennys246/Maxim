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
        self._reset_time = False

    def start(self, message: str = "") -> None:
        """Start the spinner with an initial message.

        If the spinner is already running, just updates the message
        and resets the elapsed timer instead of spawning a duplicate thread.
        """
        if self._thread is not None and self._thread.is_alive():
            # Already running — update message, reset timer
            with self._lock:
                self._message = message
                self._reset_time = True
            return
        # Stop any lingering thread before clearing the event
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._stop_event.clear()
        self._message = message
        self._reset_time = False
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
        last_elapsed = -1
        last_msg = ""
        while not self._stop_event.is_set():
            with self._lock:
                msg = self._message
                if self._reset_time:
                    start = time.time()
                    last_elapsed = -1
                    self._reset_time = False
            elapsed = int(time.time() - start)
            # Only redraw when seconds tick up or message changes
            if elapsed > last_elapsed or msg != last_msg:
                frame = _FRAMES[idx % len(_FRAMES)]
                sys.stderr.write(f"\r\033[K  {self._prefix}{frame} {msg} ({elapsed}s)")
                sys.stderr.flush()
                last_elapsed = elapsed
                last_msg = msg
                idx += 1
            self._stop_event.wait(0.1)
