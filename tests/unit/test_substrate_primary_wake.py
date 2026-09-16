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


@pytest.mark.timeout(120)
def test_substrate_proposal_is_executed_on_the_harness_loop(tmp_path):
    """A substrate proposal must reach the EXECUTOR through `run_minecraft_aut`'s loop.

    The proposal here is fear-driven (fear booked on the active cluster), but the defect
    under test is the AUTONOMY level, so the assertion is on ANY body affordance reaching
    the executor — the fake world re-rolls, and a hunger-driven `eat` is equally valid
    evidence that the loop can act at all.

    Exp 60 (2026-09-16): with fear at −1.0 the loop proposed `flee` every tick, and
    `calls=[]` — no proposal was ever executed. `run_minecraft_aut` passed no autonomy
    controller, so the loop ran at the default PLANNING level ("requires human approval
    for all actions"); the orchestrator hands its sim AUTs an AUTONOMOUS controller. On
    this harness path no substrate proposal had ever executed. Verified RED on the
    pre-fix runner (0 executor calls in 10 s), green after.
    """
    from maxim.runtime.agent_loop import _encode_current_clusters
    from maxim.simulation.minecraft_harness import FakeBridgeServer, build_minecraft_aut, run_minecraft_aut

    srv = FakeBridgeServer(state_interval_s=0.1)  # events off: the live condition
    agent_id = "exec_probe"
    aut = build_minecraft_aut(
        agent_id=agent_id,
        bridge_port=srv.port,
        persistence_dir=str(tmp_path / "exec"),
        entity_ref="bodies/minecraft_player",
    )
    from maxim.similarity.encoder import SensorEncoder, SensorEncoderConfig

    enc = SensorEncoder(ec=aut.bio.ec, atl=aut.bio.atl, nac=aut.bio.nac, config=SensorEncoderConfig())
    time.sleep(0.5)
    aut.backend.sync_world_sensors()
    wc = _encode_current_clusters(enc, agent_id, aut.executor).get("world")
    assert wc, "no world cluster encoded from the fake bridge"
    for _ in range(4):
        aut.bio.nac.record_cluster_fear(agent_id, wc, "drive:oxygen", 1.0)
    assert aut.bio.nac.anticipatory_threat_need(agent_id, {"world": wc}) > 0.5

    executed: list[str] = []
    orig_execute = aut.executor.execute

    def spy(action):
        executed.append((action or {}).get("tool_name"))
        return orig_execute(action)

    aut.executor.execute = spy
    stop = threading.Event()
    th = threading.Thread(
        target=run_minecraft_aut,
        args=(aut,),
        kwargs={"max_steps": 100_000, "target_hz": 4.0, "stop_event": stop},
        daemon=True,
    )
    try:
        th.start()
        deadline = time.monotonic() + 10.0
        while not executed and time.monotonic() < deadline:
            time.sleep(0.05)
    finally:
        stop.set()
        th.join(timeout=30.0)
        try:
            aut.client.close()
        except Exception:
            pass
        srv.close()
    assert executed, "fear proposed but NOTHING reached the executor in 10 s — the harness loop cannot act"
    assert any(str(t).startswith("minecraft_player_") for t in executed), executed


@pytest.mark.timeout(120)
def test_planning_level_never_executes_a_body_affordance(tmp_path):
    """The mechanised RED arm: the same loop at the default PLANNING level, same fear, same fake
    bridge — a body affordance must NOT reach the executor. This is the pre-fix runner's condition
    (`run_agentic_loop` builds `AutonomyController()` = PLANNING when none is passed)."""
    from maxim.agents.autonomy import AutonomyController, AutonomyLevel
    from maxim.runtime.agent_loop import _encode_current_clusters, run_agentic_loop
    from maxim.simulation.minecraft_harness import FakeBridgeServer, _loop_kwargs, build_minecraft_aut

    srv = FakeBridgeServer(state_interval_s=0.1)
    agent_id = "planning_probe"
    aut = build_minecraft_aut(
        agent_id=agent_id,
        bridge_port=srv.port,
        persistence_dir=str(tmp_path / "planning"),
        entity_ref="bodies/minecraft_player",
    )
    from maxim.similarity.encoder import SensorEncoder, SensorEncoderConfig

    enc = SensorEncoder(ec=aut.bio.ec, atl=aut.bio.atl, nac=aut.bio.nac, config=SensorEncoderConfig())
    time.sleep(0.5)
    aut.backend.sync_world_sensors()
    wc = _encode_current_clusters(enc, agent_id, aut.executor).get("world")
    for _ in range(4):
        aut.bio.nac.record_cluster_fear(agent_id, wc, "drive:oxygen", 1.0)
    executed: list[str] = []
    orig_execute = aut.executor.execute
    aut.executor.execute = lambda action: (executed.append((action or {}).get("tool_name")), orig_execute(action))[1]
    stop = threading.Event()
    kwargs = _loop_kwargs(aut, max_steps=100_000, stop_event=stop, target_hz=4.0)
    assert kwargs["autonomy_controller"].current_level == AutonomyLevel.AUTONOMOUS
    kwargs["autonomy_controller"] = AutonomyController()  # the pre-fix condition: PLANNING
    from maxim.agents.maxim_agent import MaximAgent
    from maxim.environment.filesystem_env import FileSystemEnv
    from maxim.runtime.bootstrap import build_decision_engine, build_memory
    from maxim.runtime.state import RuntimeState

    state = RuntimeState()
    state.data["mode"] = "active"
    th = threading.Thread(
        target=run_agentic_loop,
        args=(
            MaximAgent(),
            FileSystemEnv(str(tmp_path / "ws")),
            state,
            build_memory(),
            build_decision_engine(),
            aut.executor,
        ),
        kwargs=kwargs,
        daemon=True,
    )
    try:
        th.start()
        time.sleep(3.0)
    finally:
        stop.set()
        th.join(timeout=30.0)
        try:
            aut.client.close()
        except Exception:
            pass
        srv.close()
    assert not any(str(t).startswith("minecraft_player_") for t in executed), executed
