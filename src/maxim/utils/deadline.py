"""A point in monotonic time a bounded operation must finish by (#1438; docs/plans/bounded_stop.md §5).

In-process only, never persisted (it is a ``time.monotonic()`` reading, meaningless in another process). Sub-budgets
are carved from a parent and never exceed it, so a chain of waits cannot add up past the caller's bound.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class Deadline:
    """``at`` is a ``time.monotonic()`` instant, or ``math.inf`` for no bound (:meth:`none`).

    SHAPE-FROZEN at 1.0 (CC3): never persisted or sent over a wire; a monotonic instant only means anything in the
    process that took it.
    """

    at: float

    @classmethod
    def after(cls, seconds: float) -> Deadline:
        return cls(time.monotonic() + max(0.0, seconds))

    @classmethod
    def none(cls) -> Deadline:
        return cls(math.inf)

    def remaining(self) -> float:
        """Seconds left, never negative (``math.inf`` for no bound)."""
        return max(0.0, self.at - time.monotonic())

    def expired(self) -> bool:
        return self.remaining() <= 0.0

    def sub(self, seconds: float) -> Deadline:
        """At most ``seconds`` from now, clamped to this deadline."""
        return Deadline(min(self.at, time.monotonic() + max(0.0, seconds)))

    def ending_early(self, seconds: float) -> Deadline:
        """This deadline brought forward by ``seconds`` (never before now): a reserve kept for whoever waits on the
        work this bound limits. Not ``sub``, which counts from NOW. ``none()`` stays unbounded (and an infinite
        ``seconds`` never makes it NaN)."""
        if math.isinf(self.at):
            return self
        return Deadline(max(time.monotonic(), self.at - max(0.0, seconds)))

    def share(self, fraction: float) -> Deadline:
        """``fraction`` of the remaining budget, clamped to this deadline. No bound stays no bound."""
        if math.isinf(self.at):
            return self
        return Deadline(time.monotonic() + self.remaining() * min(1.0, max(0.0, fraction)))

    def cap(self, seconds: float) -> float:
        """The wait to pass a blocking call: ``min(remaining, seconds)``."""
        return min(self.remaining(), seconds)

    @staticmethod
    def earliest(*deadlines: Deadline | None) -> Deadline | None:
        """The earliest bounded deadline of ``deadlines``; ``None`` and :meth:`none` are no bound and never win, so
        ``None`` when none is bounded."""
        bounded = [d for d in deadlines if d is not None and not math.isinf(d.at)]
        return min(bounded, key=lambda d: d.at) if bounded else None


class EndBound:
    """A bound handed to ONE loop after it started (#1536): the handle that built a Talk loop sets it when it stops
    the loop, and the loop's session end reads it. ``set`` only narrows (an earlier deadline wins, a later one is
    ignored), so two stops can never widen it. In-process only, never persisted."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._deadline: Deadline | None = None
        self._remaining_at_set = 0.0

    def set(self, deadline: Deadline | None) -> None:
        """Narrow to ``deadline``, recording the time it had left NOW. ``None`` is no bound and changes nothing:
        ``rest()`` and a config rebuild stop a loop through the same path without bounding it."""
        if deadline is None:
            return
        with self._lock:
            if self._deadline is None or deadline.at < self._deadline.at:
                self._deadline = deadline
                self._remaining_at_set = deadline.remaining()

    def get(self) -> Deadline | None:
        """The bound, or ``None`` when no stop set one."""
        with self._lock:
            return self._deadline

    def snapshot(self) -> tuple[Deadline, float] | None:
        """The bound with the time it had left when it was set, so a reserve carved from it is sized at the stop,
        not by however much the end had already spent before reading it; ``None`` when unset."""
        with self._lock:
            return None if self._deadline is None else (self._deadline, self._remaining_at_set)
