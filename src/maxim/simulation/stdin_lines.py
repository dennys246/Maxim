"""The ``sim.stdin`` thread's operator-line reader, which the run's stop ends (#1541).

The run's line reader (``orchestrator.py``'s ``_stdin_reader_line``: the non-TTY reader, and the TTY reader's fallback
in ``tty_stdin.stdin_reader_raw``) used to block in ``input()``. It exited only on ``/cancel``, EOF or an interrupt, so
a run that ended any other way (the narrator's ``finish_simulation``, a campaign finish, ``max_turns``, the D12 hard
abort, the grace) returned with its reader alive. That reader then took the next stdin line meant for a later run in
the same process, or for the caller.

``OperatorLines`` reads lines so that the run's ``stop_event`` ends the read:

  - a stdin with a selectable file descriptor (a TTY, a pipe or a file, on POSIX) is polled with ``select`` every
    ``POLL_S`` between checks of the stop event, and read with ``os.read`` into this reader's own line buffer;
  - any other stdin (an object with no ``fileno``, or one ``select`` refuses, as a Windows console) is read with its
    blocking ``readline``. Nothing interrupts that read: a line it returns after the stop is dropped, not processed,
    and ``stop_reader`` reports a reader it could not join.

The fd path reads beneath ``sys.stdin``'s own buffer: bytes Python had buffered before the run's reader started are not
seen, and bytes after the last line the run processed are dropped with the reader. That loses lines only on a piped or
file stdin, and the design rests on an assumption, true when this was written (2026-10-30, no ``stdin=PIPE`` driver of a
sim in ``src/`` or ``scripts/``): nothing pipes operator lines into a sim. A terminal delivers a line per read, so on a
TTY nothing is lost.
"""

from __future__ import annotations

import logging
import os
import select
import sys
import threading
from typing import Any, Callable

logger = logging.getLogger(__name__)

POLL_S = 0.1  # how often a read on a selectable stdin looks at the stop event
STOP_JOIN_S = 1.0  # how long teardown waits for the reader after the stop: many polls


def _selectable_fd(stdin: Any) -> int | None:
    """``stdin``'s file descriptor when ``select`` accepts it, else None (the blocking fallback)."""
    if stdin is None or sys.platform == "win32":  # select takes only sockets there
        return None
    try:
        fd = stdin.fileno()
        select.select([fd], [], [], 0)
    except (AttributeError, OSError, ValueError) as e:  # io.UnsupportedOperation is both of the last two
        logger.debug("stdin has no selectable fd (%s): the operator-line reader blocks, and a stop cannot end it", e)
        return None
    return fd


class OperatorLines:
    """One run's operator lines from ``stdin``, ended by the run's ``stop_event`` (module docstring)."""

    def __init__(self, stdin: Any, stop_event: threading.Event) -> None:
        self._stdin = stdin
        self._stop = stop_event
        self._fd = _selectable_fd(stdin)
        self._encoding = getattr(stdin, "encoding", None) or "utf-8"
        self._errors = getattr(stdin, "errors", None) or "replace"
        self._buf = b""
        self._eof = False

    @property
    def interruptible(self) -> bool:
        """Whether the stop event ends a read in progress (a selectable stdin)."""
        return self._fd is not None

    def next_line(self) -> str | None:
        """The next line without its newline, as ``input()`` returns it; None once the stop event is set; raises
        ``EOFError`` at the end of input."""
        if self._fd is None:
            return self._blocking_line()
        while True:
            nl = self._buf.find(b"\n")
            if nl >= 0:
                line, self._buf = self._buf[:nl], self._buf[nl + 1 :]
                return line.decode(self._encoding, self._errors)
            if self._eof:
                if not self._buf:
                    raise EOFError
                line, self._buf = self._buf, b""
                return line.decode(self._encoding, self._errors)
            if self._stop.is_set():
                return None
            ready, _, _ = select.select([self._fd], [], [], POLL_S)
            if ready:
                chunk = os.read(self._fd, 4096)
                self._eof = not chunk
                self._buf += chunk

    def _blocking_line(self) -> str | None:
        line = self._stdin.readline()
        if not line:
            raise EOFError
        if self._stop.is_set():
            return None  # the run ended while this read blocked: the line was not the run's to process
        return line[:-1] if line.endswith("\n") else line


def read_operator_lines(
    *,
    lines: OperatorLines,
    stop_event: threading.Event,
    operator_stop: threading.Event,
    sim_prompt_handler: Any,
    process_line: Callable[[str], bool],
    stop_by_operator: Callable[[threading.Event, threading.Event], None],
) -> None:
    """The line reader's loop. A line goes to the agent's pending prompt when one was waiting as the read began
    (``SimPromptHandler.deliver_response``), else to the run's command parser, which returns True to end the reader.
    End of input ends the reader, and also stops the run when a prompt was waiting on it. A ``KeyboardInterrupt`` stops
    the run too, kept from the ``input()`` loop this replaced; Python delivers SIGINT only to the main thread, which this
    reader never is, so a real Ctrl+C reaches the run's own handler instead."""
    while not stop_event.is_set():
        pending = sim_prompt_handler is not None and sim_prompt_handler.has_pending_prompt
        try:
            line = lines.next_line()
        except EOFError:
            if pending:
                stop_by_operator(stop_event, operator_stop)
            return
        except KeyboardInterrupt:
            stop_by_operator(stop_event, operator_stop)
            return
        if line is None:
            return
        if pending:
            sim_prompt_handler.deliver_response(line.strip())
            continue
        if process_line(line):
            return


def stop_reader(thread: threading.Thread, *, timeout_s: float = STOP_JOIN_S) -> bool:
    """Join the run's ``sim.stdin`` reader once the run's stop event is set; False, with a warning, if it is still
    running after ``timeout_s``: between two polls (it ends within ``POLL_S``; a stopped run past its bound passes a
    cap of 0), inside a slow command, or blocked in a read on a stdin ``select`` cannot poll (it takes, and drops, one
    more line). A WARNING, as the shutdown raises ``maxim``'s level to WARNING before the tail waits."""
    thread.join(timeout=timeout_s)
    if not thread.is_alive():
        return True
    logger.warning(
        "the sim.stdin reader did not stop within %.1fs of the run's stop: it ends at its next poll (within %.1fs), "
        "unless it is still in a command, or blocked on a stdin that cannot be polled (then it takes, and drops, the "
        "next line typed)",
        timeout_s,
        POLL_S,
    )
    return False
