"""Session ownership (#1438 PR 0; docs/plans/bounded_stop.md §4): the hub's session is CLOSED -> OPEN(gen, owner) ->
ENDING(gen) -> CLOSED, and only the ticket that opened a generation ends it.

Today ownership is a test-and-clear bool (`MemoryHub._session_active`): any caller of `on_session_end()` ends whatever
is open, so `AgentInstance.shutdown` can end a session a running loop still holds, and two closers race.
"""

from __future__ import annotations

import threading
import time

import pytest



def _ownership():
    from maxim.integration.session_ownership import SessionOwnership

    return SessionOwnership()


def test_owner_is_required_keyword_only() -> None:
    own = _ownership()
    with pytest.raises(TypeError):
        own.claim()  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        own.claim(object())  # type: ignore[misc]


def test_a_claim_on_a_closed_session_opens_a_new_generation() -> None:
    own = _ownership()
    instance = object()
    ticket = own.claim(owner=instance)
    assert ticket.owner is instance and ticket.generation == 1
    assert own.state == "OPEN"


def test_the_same_owner_claiming_again_gets_the_same_ticket() -> None:
    own = _ownership()
    instance = object()
    first = own.claim(owner=instance)
    assert own.claim(owner=instance) is first


def test_a_loop_takes_over_from_an_owner_with_no_live_loop() -> None:
    own = _ownership()
    instance, loop = object(), object()
    held = own.claim(owner=instance)
    taken = own.claim(owner=loop, live=True)
    assert taken.generation == held.generation + 1 and taken.owner is loop
    assert not own.is_current(held)  # the old ticket is invalidated: ending it ends nothing


def test_anyone_else_claiming_from_a_live_loop_is_refused() -> None:
    from maxim.exceptions import SessionHeld

    own = _ownership()
    loop = object()
    own.claim(owner=loop, live=True)
    with pytest.raises(SessionHeld):
        own.claim(owner=object(), live=True)
    with pytest.raises(SessionHeld):
        own.claim(owner=object())


def test_only_the_current_ticket_ends_its_generation() -> None:
    own = _ownership()
    instance, loop = object(), object()
    stale = own.claim(owner=instance)
    current = own.claim(owner=loop, live=True)
    assert own.begin_end(stale) is False  # a stale ticket after a handover ends nothing
    assert own.state == "OPEN"
    assert own.begin_end(current) is True
    assert own.state == "ENDING"
    own.finish_end(current)
    assert own.state == "CLOSED" and current.end_done.is_set()


def test_a_claim_during_ending_waits_for_end_done() -> None:
    own = _ownership()
    loop = object()
    ticket = own.claim(owner=loop, live=True)
    own.begin_end(ticket)
    threading.Timer(0.2, own.finish_end, args=(ticket,)).start()
    started = time.monotonic()
    nxt = own.claim(owner=object(), deadline_s=5.0)
    assert time.monotonic() - started >= 0.15
    assert nxt.generation == ticket.generation + 1


def test_a_claim_during_ending_past_its_deadline_raises_still_ending() -> None:
    from maxim.exceptions import SessionStillEnding

    own = _ownership()
    ticket = own.claim(owner=object(), live=True)
    own.begin_end(ticket)
    with pytest.raises(SessionStillEnding):
        own.claim(owner=object(), deadline_s=0.05)


def test_a_released_loop_claim_is_no_longer_live() -> None:
    own = _ownership()
    loop = object()
    ticket = own.claim(owner=loop, live=True)
    own.release_live(ticket)
    taken = own.claim(owner=object(), live=True)  # no SessionHeld: the loop let go
    assert taken.generation == ticket.generation + 1


def test_a_non_owner_waits_for_end_done_within_its_deadline() -> None:
    own = _ownership()
    ticket = own.claim(owner=object(), live=True)
    assert own.wait_ended(deadline_s=0.05) is False  # the owner has not ended it
    own.begin_end(ticket)
    own.finish_end(ticket)
    assert own.wait_ended(deadline_s=0.05) is True


def test_the_hub_open_requires_an_owner_and_a_ticketless_close_raises() -> None:
    from maxim.integration.memory_hub import MemoryHub

    with pytest.raises(TypeError):
        MemoryHub.on_session_start(MemoryHub.__new__(MemoryHub))  # type: ignore[call-arg]
    for closer in ("on_session_end", "on_session_end_lightweight"):
        with pytest.raises(TypeError):
            getattr(MemoryHub, closer)(MemoryHub.__new__(MemoryHub))  # type: ignore[call-arg]
