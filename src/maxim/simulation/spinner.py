"""Simple terminal spinner for simulation progress feedback.

When a ``MaximDisplay`` is active, the spinner routes status through
``display.set_status()`` instead of raw ANSI cursor writes to stderr.
Turn-summary lines route through ``sim_logger._emit()`` so the display
can render them in its log panel.
"""

from __future__ import annotations

import sys
import threading
import time
from typing import Any


_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]


def _get_display() -> Any | None:
    """Return the active MaximDisplay, or None. Import deferred to avoid cycles."""
    from maxim.simulation.sim_logger import get_active_display

    return get_active_display()


def spinner_truth_message(
    *,
    between_turns: bool,
    in_flight_and_silent: bool,
    stall_duration_s: float,
    threshold_s: float,
    nudge_count: int,
    byte_silence_s: float | None,
    byte_silence_threshold_s: float,
) -> str | None:
    """Status-line truth for the between-turns spinner (bugs ledger D14).

    The bridge sets "Orchestrator planning next probe..." when a turn ENDS
    and nothing updates it on any failure path, so during a dropped planning
    turn the display asserts work that py-spy proves is not happening — the
    defect that steered hours of diagnosis toward server/network theories.
    A status display must report OBSERVED state, never intent.

    Returns the corrected spinner text, or ``None`` when the default text is
    truthful (the inter-call gap is still within the stall threshold) — the
    caller leaves the spinner alone.

    ``in_flight_and_silent`` is named for what the ONE production caller can
    actually observe at this point: the stall detector has already
    ``continue``d on a healthy in-flight call, so reaching here with a live
    call means its bytes have gone silent past the keepalive threshold. A
    plain ``in_flight`` would assert more than the caller knows — the same
    display-lies class this function exists to fix (pre-merge review,
    architecture lens S6).
    ``between_turns=False`` always returns ``None``: mid-turn the spinner
    belongs to the AUT exchange, not the orchestrator.

    Pure function — fully unit-testable; the stall detector supplies the
    same registry-derived state it already polls.
    """
    if not between_turns:
        return None
    if in_flight_and_silent:
        if byte_silence_s is not None and byte_silence_s >= byte_silence_threshold_s:
            return f"⚠ planning call in flight but silent {int(byte_silence_s)}s (connection may be wedged)"
        return None
    if stall_duration_s >= threshold_s:
        nudge_note = f", {nudge_count} nudge(s) sent" if nudge_count > 0 else ""
        return (
            f"⚠ no LLM call in flight — planning turn may be lost "
            f"({int(stall_duration_s)}s since last turn{nudge_note})"
        )
    return None


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
        # D14: True exactly while this spinner shows the between-turns
        # "Orchestrator planning next probe..." text. Owned by the spinner
        # (not the transport) and read/written under ``_lock`` so the stall
        # detector's correction cannot race a turn that just started —
        # a plain flag + separate update would let the detector clobber
        # fresh turn text with a "may be lost" warning while the AUT is
        # actively working (pre-merge review, executor lens F7 /
        # architecture lens N5).
        self._planning_window = False

    def set_planning_window(self, active: bool) -> None:
        """Mark (or clear) the between-turns planning window."""
        with self._lock:
            self._planning_window = active

    @property
    def planning_window(self) -> bool:
        with self._lock:
            return self._planning_window

    def update_if_planning(self, message: str) -> bool:
        """Atomically replace the message ONLY while the planning window is
        open. Returns True if the message was applied.

        The test-and-set happens under one lock acquisition, so a
        ``send_and_wait`` that opens a new turn between a caller's check and
        its write cannot have its fresh text overwritten.
        """
        with self._lock:
            if not self._planning_window:
                return False
            self._message = message
            return True

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

        display = _get_display()
        if display is not None:
            # Clear status and route the summary through the display log
            display.set_status(status="")
            if final_message:
                from maxim.simulation.sim_logger import _emit

                _emit(f"  {self._prefix}{final_message}", "summary")
        else:
            # Raw terminal — clear the spinner line
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
                display = _get_display()
                if display is not None:
                    display.set_status(status=f"{msg} ({elapsed}s)")
                else:
                    frame = _FRAMES[idx % len(_FRAMES)]
                    sys.stderr.write(f"\r\033[K  {self._prefix}{frame} {msg} ({elapsed}s)")
                    sys.stderr.flush()
                last_elapsed = elapsed
                last_msg = msg
                idx += 1
            self._stop_event.wait(0.1)
