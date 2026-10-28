"""Who owns a memory session, and who may end it (#1438 PR 0; docs/plans/bounded_stop.md §4).

The hub's session moves ``CLOSED -> OPEN(generation, owner) -> ENDING(generation) -> CLOSED``, every transition
under one lock. A claim returns a :class:`SessionTicket`; only the CURRENT ticket ends its generation, so a closer
that does not hold it (``AgentInstance.shutdown``, a drain, the orchestrator's ``/new``) cannot end a session a
running loop still holds. That is the defect the old test-and-clear bool allowed.

Claim rules:

====================  ==================================  =====================================================
State                 Claimer                             Result
====================  ==================================  =====================================================
CLOSED                anyone                              a new generation
OPEN, same owner      that owner (identity, ``is``)       idempotent: the same ticket
OPEN, no live loop    a loop (``live=True``)              handover: a new generation; the old ticket goes stale
OPEN, a live loop     anyone else                         :class:`~maxim.exceptions.SessionHeld`
ENDING                anyone                              waits for ``end_done`` within the deadline, else
                                                          :class:`~maxim.exceptions.SessionStillEnding`
====================  ==================================  =====================================================

"Live" is hub state: a claim made with ``live=True`` that its context manager has not yet released
(:meth:`SessionOwnership.release_live`). It is never inferred from the owner's kind.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Literal

from maxim.exceptions import SessionHeld, SessionStillEnding

# A claimer that passes no deadline still never waits forever on an ending session.
ENDING_WAIT_CAP_S = 60.0

State = Literal["CLOSED", "OPEN", "ENDING"]


@dataclass(eq=False)
class SessionTicket:
    """One claim on one session generation. ``end_done`` is set once that generation has ended.

    In-process only (never persisted, never sent): identity is the ticket object itself.
    """

    generation: int
    owner: Any
    end_done: threading.Event = field(default_factory=threading.Event, repr=False)
    # True when this claim moved CLOSED -> OPEN (the hub restores its stores); False for a handover, whose session
    # data is continuous: a restore would clear and reload what was stored since the open.
    fresh: bool = False
    # What the hub restored at a fresh open (counts per store); empty for a handover or an idempotent claim.
    start_stats: dict = field(default_factory=dict, repr=False)


class SessionOwnership:
    """The three-state session ledger a :class:`~maxim.integration.memory_hub.MemoryHub` holds."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state: State = "CLOSED"
        self._generation = 0
        self._ticket: SessionTicket | None = None
        self._live = False

    @property
    def state(self) -> State:
        return self._state

    @property
    def current(self) -> SessionTicket | None:
        return self._ticket

    def is_current(self, ticket: SessionTicket | None) -> bool:
        return ticket is not None and ticket is self._ticket

    def claim(self, *, owner: Any, live: bool = False, deadline_s: float | None = None) -> SessionTicket:
        """Claim the session for ``owner`` by the rules in the module docstring."""
        return self.claim_reporting(owner=owner, live=live, deadline_s=deadline_s)[0]

    def claim_reporting(
        self, *, owner: Any, live: bool = False, deadline_s: float | None = None
    ) -> tuple[SessionTicket, bool]:
        """:meth:`claim`, also reporting whether THIS call opened the generation it returns.

        An idempotent re-claim returns the existing ticket (``opened`` False) even though that ticket's ``fresh`` is
        still True from its own open; the hub restores its stores only when ``opened and ticket.fresh``.
        """
        wait_until = time.monotonic() + (
            ENDING_WAIT_CAP_S if deadline_s is None else min(deadline_s, ENDING_WAIT_CAP_S)
        )
        while True:
            with self._lock:
                if self._state == "ENDING":
                    ending = self._ticket
                elif self._state == "OPEN" and self._ticket is not None:
                    if self._ticket.owner is owner:
                        self._live = self._live or live
                        return self._ticket, False
                    if self._live:
                        raise SessionHeld(
                            f"the memory session (generation {self._generation}) is held by a live loop; "
                            "it ends when that loop exits"
                        )
                    if not live:
                        # Only a loop takes over an owner with no live loop; any other claimer shares it.
                        return self._ticket, False
                    return self._open(owner, live, fresh=False), True
                else:
                    return self._open(owner, live, fresh=True), True
            remaining = wait_until - time.monotonic()
            if ending is None or remaining <= 0 or not ending.end_done.wait(remaining):
                with self._lock:
                    if self._state != "ENDING" or self._ticket is not ending:
                        continue  # it ended (or moved on) just as we gave up: look again
                raise SessionStillEnding(
                    f"the memory session (generation {ending.generation if ending else '?'}) is still ending after "
                    f"{ENDING_WAIT_CAP_S if deadline_s is None else deadline_s:g} s"
                )

    def _open(self, owner: Any, live: bool, *, fresh: bool) -> SessionTicket:
        self._generation += 1
        self._ticket = SessionTicket(generation=self._generation, owner=owner, fresh=fresh)
        self._state = "OPEN"
        self._live = live
        return self._ticket

    def abandon(self, ticket: SessionTicket) -> None:
        """Close a generation whose open failed half-way, so no orphaned OPEN session outlives a raising opener."""
        if self.begin_end(ticket):
            self.finish_end(ticket)

    def release_live(self, ticket: SessionTicket) -> None:
        """The loop's context manager let go of its claim; a stale ticket changes nothing."""
        with self._lock:
            if ticket is self._ticket:
                self._live = False

    def begin_end(self, ticket: SessionTicket | None) -> bool:
        """OPEN -> ENDING for the current ticket only. False (and nothing changes) for any other ticket."""
        with self._lock:
            if ticket is None or ticket is not self._ticket or self._state != "OPEN":
                return False
            self._state = "ENDING"
            return True

    def finish_end(self, ticket: SessionTicket) -> None:
        """ENDING -> CLOSED, then wake every waiter on ``end_done``."""
        with self._lock:
            if ticket is not self._ticket or self._state != "ENDING":
                return
            self._state = "CLOSED"
            self._ticket = None
            self._live = False
        ticket.end_done.set()

    def wait_ended(self, *, deadline_s: float | None = None) -> bool:
        """For a non-owner closer: wait until no session is open or ending, within the deadline (else the cap)."""
        with self._lock:
            ticket = self._ticket
        if ticket is None:
            return True
        return ticket.end_done.wait(ENDING_WAIT_CAP_S if deadline_s is None else min(deadline_s, ENDING_WAIT_CAP_S))
