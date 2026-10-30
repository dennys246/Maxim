"""Owed after-steps on the session ledger (#1520, #1438 PR 1a; ``integration/session_ownership.py``).

A closer that must wait for a running loop's end (``AgentInstance.shutdown``) hands the steps it may not run before
that end to the generation it waits for. They attach only to a generation that WILL end, run once in the ender's
thread while the state is still ENDING, never cost the close, and an abandon drops them loudly.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from maxim.integration.session_ownership import SessionOwnership
from maxim.utils.deadline import Deadline


def _ending(ledger: SessionOwnership) -> Any:
    ticket = ledger.claim(owner=object(), live=True)
    assert ledger.begin_end(ticket)
    return ticket


# ── 1. which generations take a step ─────────────────────────────────────────


def test_defer_attaches_only_to_a_generation_that_will_end() -> None:
    ledger = SessionOwnership()
    assert ledger.defer_until_ended("s", lambda: None) is False  # CLOSED: nothing will end

    opener = ledger.claim(owner=object())  # OPEN, no live loop: it may be handed over, its ticket would go stale
    assert ledger.defer_until_ended("s", lambda: None) is False
    assert opener.after_end == {}

    loop = ledger.claim(owner=object(), live=True)  # OPEN under a live loop
    assert ledger.defer_until_ended("s", lambda: None) is True

    assert ledger.begin_end(loop)  # ENDING
    assert ledger.defer_until_ended("t", lambda: None) is True
    assert set(loop.after_end) == {"s", "t"}


def test_defer_is_refused_once_the_after_steps_started() -> None:
    ledger = SessionOwnership()
    ticket = _ending(ledger)
    answers: list[bool] = []
    ran: list[str] = []
    ticket.after_end["first"] = lambda: answers.append(ledger.defer_until_ended("late", lambda: ran.append("late")))

    ledger.finish_end(ticket)

    assert answers == [False]  # the caller runs it itself ...
    assert ran == [] and "late" not in ticket.after_end  # ... it is never silently queued on a finished list


# ── 2. one step per name ─────────────────────────────────────────────────────


def test_the_same_name_twice_runs_one_step() -> None:
    ledger = SessionOwnership()
    ticket = _ending(ledger)
    ran: list[str] = []
    assert ledger.defer_until_ended("agent-shutdown", lambda: ran.append("first"))
    assert ledger.defer_until_ended("agent-shutdown", lambda: ran.append("second"))  # a second shutdown adds nothing

    ledger.finish_end(ticket)

    assert ran == ["first"]


# ── 3. where and when they run ───────────────────────────────────────────────


def test_finish_end_runs_the_steps_once_in_the_enders_thread_while_still_ending() -> None:
    ledger = SessionOwnership()
    ticket = _ending(ledger)
    seen: list[tuple[str, Any, bool]] = []
    ledger.defer_until_ended(
        "step", lambda: seen.append((ledger.state, threading.current_thread(), ticket.end_done.is_set()))
    )

    ledger.finish_end(ticket)
    ledger.finish_end(ticket)  # a second finish on a closed generation runs nothing

    assert seen == [("ENDING", threading.current_thread(), False)]
    assert ledger.state == "CLOSED" and ledger.current is None
    assert ticket.end_done.is_set()


# ── 4. a failing step never costs the close ──────────────────────────────────


def test_a_raising_step_costs_neither_the_others_nor_the_close(caplog: Any) -> None:
    ledger = SessionOwnership()
    ticket = _ending(ledger)
    ran: list[str] = []

    def _boom() -> None:
        raise RuntimeError("cerebellum disk full")

    ledger.defer_until_ended("a", lambda: ran.append("a"))
    ledger.defer_until_ended("boom", _boom)
    ledger.defer_until_ended("c", lambda: ran.append("c"))

    with caplog.at_level(logging.WARNING, logger="maxim.integration.session_ownership"):
        ledger.finish_end(ticket)

    assert ran == ["a", "c"]
    assert "'boom' failed" in caplog.text
    assert ledger.state == "CLOSED" and ticket.end_done.is_set()


def test_a_base_exception_in_a_step_still_closes_the_session_and_propagates() -> None:
    ledger = SessionOwnership()
    ticket = _ending(ledger)

    def _interrupt() -> None:
        raise KeyboardInterrupt

    ledger.defer_until_ended("interrupt", _interrupt)

    with pytest.raises(KeyboardInterrupt):
        ledger.finish_end(ticket)

    assert ledger.state == "CLOSED" and ledger.current is None
    assert ticket.end_done.is_set()


# ── 5. abandon drops them, loudly ────────────────────────────────────────────


def test_abandon_drops_the_owed_steps_and_names_each_at_error(caplog: Any) -> None:
    ledger = SessionOwnership()
    ticket = ledger.claim(owner=object(), live=True)
    ran: list[str] = []
    ledger.defer_until_ended("hub-workers", lambda: ran.append("hub-workers"))
    ledger.defer_until_ended("cerebellum", lambda: ran.append("cerebellum"))

    with caplog.at_level(logging.ERROR, logger="maxim.integration.session_ownership"):
        ledger.abandon(ticket)

    assert ran == []
    errors = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
    assert any("'hub-workers'" in m for m in errors) and any("'cerebellum'" in m for m in errors)
    assert ledger.state == "CLOSED" and ticket.end_done.is_set()


# ── 6. AgentInstance / AgentPool ─────────────────────────────────────────────


def _agent(tmp_path: Path) -> Any:
    from maxim.runtime.agent_factory import AgentConfig, AgentFactory
    from maxim.tools.registry import ToolRegistry

    config = AgentConfig(
        agent_id="ledger_agent",
        role="pc",
        persistence_dir=str(tmp_path / "ledger_agent"),
        with_bio_stack=True,
        with_executor=False,
        with_pain_bridge=False,
        with_fear_gate=False,
    )
    return AgentFactory(base_data_dir=tmp_path).create_full_agent(config, tool_registry=ToolRegistry(), auto_load=True)


def test_shutdown_bounds_only_its_wait_for_a_running_loop(tmp_path: Path, caplog: Any) -> None:
    """``loop_wait`` alone (the console and the pool pass no ``deadline``): the wait for the loop is bounded."""
    inst = _agent(tmp_path)
    hub = inst.memory_hub
    loop_ticket = hub.on_session_start(owner=object(), live=True)  # a loop takes the session over

    started = time.monotonic()
    with caplog.at_level(logging.WARNING, logger="maxim.runtime.agent_factory"):
        inst.shutdown(loop_wait=Deadline.after(0.05))
    elapsed = time.monotonic() - started

    assert elapsed < 1.0, f"shutdown waited {elapsed:.2f}s for a running loop despite loop_wait=0.05 s"
    assert "session_end_not_reached" in caplog.text
    assert f"agent-shutdown:{id(inst)}" in loop_ticket.after_end  # handed to the loop's end, not dropped
    hub.on_session_end_lightweight(loop_ticket)  # the loop ends; its after-steps run there
    assert loop_ticket.end_done.is_set()


def test_the_pool_bounds_only_the_wait_never_the_agents_own_end() -> None:
    from maxim.runtime.agent_pool import AgentPool

    calls: list[dict[str, Any]] = []

    class _Instance:
        agent_id = "pooled"
        role = "pc"

        def shutdown(self, **kwargs: Any) -> None:
            calls.append(kwargs)

    pool = AgentPool()
    pool.add(_Instance())
    bound = Deadline.after(0.5)

    pool.remove("pooled", deadline=bound)

    assert calls == [{"loop_wait": bound}]  # no ``deadline=``: the agent's own end stays unbounded until 1b
