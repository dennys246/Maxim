"""Substrate-primary loop wake source — the loop must tick from its OWN cadence, not from text events.

Exp 60 (2026-09-16): every live probe window read one substrate tick, then silence. The idle
gate in ``run_agentic_loop`` wakes on pending input/work, a sim percept, a carried percept,
the first step, or an awaited LLM. In substrate-primary mode there is no LLM worker, and the
Minecraft percept source's ``has_pending`` is the EVENT queue (chat/death), not state — so a
live bridge that emits no events leaves the loop idling forever after step 0. The fake bridge's
periodic "wind shifts" event masked it offline (and produced the "5 percepts per tick" cadence).

Guard: against a fake bridge with events OFF (the live condition) the loop must still reach
``propose_via_substrate`` repeatedly at its submit cadence. Verified to FAIL on the pre-fix loop
(one call in 2.5 s) and PASS after the wake term.
"""

from __future__ import annotations

import threading
import time

import pytest


def _ticks_with_events(events: bool, tmp_path, *, want: int = 3, deadline_s: float = 10.0) -> list[float]:
    from maxim.runtime import agent_loop as AL
    from maxim.simulation.minecraft_harness import FakeBridgeServer, build_minecraft_aut, run_minecraft_aut

    srv = FakeBridgeServer(state_interval_s=0.1, events=events)
    aut = build_minecraft_aut(
        agent_id="wake_probe",
        bridge_port=srv.port,
        persistence_dir=str(tmp_path / ("events" if events else "noevents")),
        entity_ref="bodies/minecraft_player",
    )
    calls: list[float] = []
    orig = AL.propose_via_substrate

    def spy(**kw):
        calls.append(time.monotonic())
        return orig(**kw)

    AL.propose_via_substrate = spy
    stop = threading.Event()
    th = threading.Thread(
        target=run_minecraft_aut,
        args=(aut,),
        kwargs={"max_steps": 100_000, "target_hz": 4.0, "stop_event": stop},
        daemon=True,
    )
    try:
        th.start()
        # Poll for the wanted tick count instead of a fixed wall window: loop startup
        # (agent, memory, decision engine, session start) is inside the thread and is
        # unmeasured on a slow runner. The pre-fix loop produces exactly ONE call ever
        # (step 0), so a 10 s deadline still fails RED; green runs finish in ~1 s.
        deadline = time.monotonic() + deadline_s
        while len(calls) < want and time.monotonic() < deadline:
            time.sleep(0.05)
    finally:
        stop.set()
        th.join(timeout=30.0)
        AL.propose_via_substrate = orig
        assert not th.is_alive(), "loop thread outlived its stop event"
        try:
            aut.client.close()
        except Exception:
            pass
        srv.close()
    return calls


@pytest.mark.timeout(90)
def test_substrate_primary_ticks_without_any_text_events(tmp_path):
    """The live condition: state snapshots only, no events — the loop must keep ticking."""
    calls = _ticks_with_events(False, tmp_path)
    assert len(calls) >= 3, f"substrate branch reached {len(calls)}x within 10 s with no events — the loop idled"


@pytest.mark.timeout(90)
def test_substrate_primary_ticks_with_events_too(tmp_path):
    """Sanity: events still wake it (the pre-fix offline behaviour, kept)."""
    assert len(_ticks_with_events(True, tmp_path)) >= 3
