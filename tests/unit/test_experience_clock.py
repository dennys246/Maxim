"""The experience clock and what advances it (memory-strength plan, Phase 2 decision 1)."""

from __future__ import annotations

import ast
import inspect
import logging
import threading
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from maxim.memory.experience_clock import UNIT, ExperienceClock
from maxim.memory.hippocampus import Hippocampus, HippocampusConfig
from maxim.runtime.agent_loop import _loop_bio_handles, _loop_live_tick
from maxim.runtime.experience_time import REALTIME_PASS_CAP_US, ExperienceClockDriver
from maxim.simulation.composite_source import CompositePerceptSource, CompositeRoutingError
from maxim.simulation.conversational_source import ConversationalSource


def _hippo() -> Hippocampus:
    return Hippocampus(HippocampusConfig(persistence_path=None))


class _FakeMonotonic:
    def __init__(self) -> None:
        self.t = 100.0

    def __call__(self) -> float:
        return self.t


# ── the clock ────────────────────────────────────────────────────────────────


def test_clock_is_monotonic_and_reads_no_clock():
    clock = ExperienceClock()
    assert clock.advance(250_000) == 250_000
    assert clock.advance(0) == 250_000
    with pytest.raises(ValueError):
        clock.advance(-1)
    with pytest.raises(ValueError):
        ExperienceClock(-5)
    import maxim.memory.experience_clock as module

    tree = ast.parse(inspect.getsource(module))
    imported = {a.name for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom)) for a in n.names}
    imported |= {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    assert not imported & {"time", "datetime"}


def test_concurrent_advances_lose_nothing():
    clock = ExperienceClock()
    threads = [threading.Thread(target=lambda: [clock.advance(1) for _ in range(5000)]) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert clock.now_us() == 40_000


@pytest.mark.parametrize(
    "record",
    [{"us": 10, "unit": "nominal_active_ms"}, {"us": 1.5, "unit": UNIT}, {"us": "10", "unit": UNIT}, [1, 2], {}],
)
def test_a_malformed_record_is_refused_not_misread(record):
    with pytest.raises(ValueError):
        ExperienceClock.from_dict(record)


# ── persistence ──────────────────────────────────────────────────────────────


def test_the_hippocampus_persists_its_clock_and_restores_it_in_place(tmp_path):
    path = str(tmp_path / "hippocampus.json")
    hippo = _hippo()
    hippo.experience_clock.advance(1_234_567)
    hippo.save(path)

    restored = _hippo()
    held = restored.experience_clock  # a reference taken BEFORE the load (e.g. by ATL via the hub)
    restored.load(path)
    assert restored.experience_clock is held
    assert held.now_us() == 1_234_567
    assert restored.dump()["experience_clock"] == {"us": 1_234_567, "unit": UNIT}


def test_a_pre_clock_snapshot_loads_at_zero_with_one_warning(caplog, monkeypatch):
    import maxim.memory.hippocampus_persistence as hp

    monkeypatch.setattr(hp, "_missing_clock_warned", False)
    state = _hippo().dump()
    del state["experience_clock"]
    with caplog.at_level(logging.WARNING, logger=hp.logger.name):
        for _ in range(3):
            hippo = _hippo()
            hippo.experience_clock.advance(7)
            hippo.load_state(state)
            assert hippo.experience_clock.now_us() == 0
    assert sum("experience_clock" in r.getMessage() for r in caplog.records) == 1


def test_a_corrupt_clock_never_costs_the_memories(caplog, hippocampus, complete_memory_args):
    mid = hippocampus.capture(**complete_memory_args)
    state = hippocampus.dump()
    state["experience_clock"] = {"us": "garbage", "unit": UNIT}
    restored = _hippo()
    with caplog.at_level(logging.WARNING):
        restored.load_state(state)
    assert restored.recall_by_ids([mid])  # the memories loaded
    assert restored.experience_clock.now_us() == 0
    assert any("bad experience_clock" in r.getMessage() for r in caplog.records)


# ── the driver: world time, per world ────────────────────────────────────────


def test_a_realtime_world_advances_by_elapsed_world_time_idle_included():
    clock, mono = ExperienceClock(), _FakeMonotonic()
    driver = ExperienceClockDriver(clock, percept_source=None, monotonic=mono)
    assert driver.kind == "realtime"
    assert driver.on_live_pass() == 0  # the first pass only starts the count
    for dt in (0.25, 0.25, 0.1, 0.4):  # 4 Hz ticks and idle sleeps alike: the world kept going
        mono.t += dt
        driver.on_live_pass()
    assert clock.now_us() == 1_000_000


def test_paused_time_is_subtracted_exactly():
    clock, mono = ExperienceClock(), _FakeMonotonic()
    paused = {"s": 0.0}
    driver = ExperienceClockDriver(clock, percept_source=None, paused_seconds=lambda: paused["s"], monotonic=mono)
    driver.on_live_pass()
    mono.t += 600.5  # ten minutes paused, then half a second lived after resume
    paused["s"] += 600.0
    assert driver.on_live_pass() == 500_000


def test_a_long_blocking_pass_counts_in_full_up_to_the_stall_bound():
    clock, mono = ExperienceClock(), _FakeMonotonic()
    driver = ExperienceClockDriver(clock, percept_source=None, monotonic=mono)
    driver.on_live_pass()
    mono.t += 12.0  # a multi-cycle deliberation or a robot motion: the world went on meanwhile
    assert driver.on_live_pass() == 12_000_000
    mono.t += 3600.0  # a genuine stall is bounded
    assert driver.on_live_pass() == REALTIME_PASS_CAP_US


def test_the_autonomy_controller_accounts_its_paused_time(monkeypatch):
    import maxim.agents.autonomy as autonomy

    mono = _FakeMonotonic()
    monkeypatch.setattr(autonomy.time, "monotonic", mono)
    controller = autonomy.AutonomyController()
    controller.emergency_halt("test")
    mono.t += 30.0
    assert controller.paused_seconds_total() == 30.0  # the open pause counts
    controller.resume()
    mono.t += 10.0
    assert controller.paused_seconds_total() == 30.0


def test_pain_inside_a_turn_is_not_a_new_turn():
    source = ConversationalSource()
    start = source.experience_turns()
    source.inject_pain(pain_type="burn", intensity=0.9)
    assert source.experience_turns() == start


@pytest.mark.parametrize("aut_mode", ["llm-primary", "substrate-primary"])
def test_every_bridge_turn_moves_the_world(aut_mode):
    # substrate-primary injects no text, but the world still took its turn (round-2 review S1)
    from maxim.simulation.bridge import SimulationBridge

    bridge = SimulationBridge(response_timeout=0.05, settle_s=0.01, aut_mode=aut_mode)
    before = bridge.percept_source.experience_turns()
    bridge.send_and_wait("the tide comes in", timeout=0.05, settle_s=0.01)
    assert bridge.percept_source.experience_turns() == before + 1


def test_a_misdeclared_turn_world_fails_at_construction_not_on_the_loop_thread():
    class Bad(_Live):
        experience_us_per_turn = 1.5

        def experience_turns(self):
            return 0

    with pytest.raises(TypeError):
        ExperienceClockDriver(ExperienceClock(), percept_source=Bad())
    with pytest.raises(TypeError):
        ExperienceClockDriver(ExperienceClock(), percept_source=MagicMock())


def test_a_turn_based_world_advances_per_turn_never_by_wall_time():
    clock, mono = ExperienceClock(), _FakeMonotonic()
    source = ConversationalSource()
    driver = ExperienceClockDriver(clock, percept_source=source, monotonic=mono)
    assert driver.kind == "turns"
    mono.t += 90.0  # a slow LLM: wall time passes, no world turn does
    driver.on_live_pass()
    assert clock.now_us() == 0
    source.inject_cli("the door creaks open")
    source.inject_cli("a draft blows in")
    driver.on_live_pass()  # two world turns, whether or not the agent has read them yet
    assert clock.now_us() == 2 * source.experience_us_per_turn
    driver.on_live_pass()  # no new turns
    assert clock.now_us() == 2 * source.experience_us_per_turn


def test_turns_taken_before_the_run_do_not_count():
    source = ConversationalSource()
    source.inject_cli("before")
    clock = ExperienceClock()
    ExperienceClockDriver(clock, percept_source=source).on_live_pass()
    assert clock.now_us() == 0


def test_no_clock_means_an_inert_driver():
    assert ExperienceClockDriver(None, percept_source=None).on_live_pass() == 0


class _Live:
    name = "sensor"

    def next_percept(self):
        return None

    def is_exhausted(self):
        return False


class _TurnOnly(_Live):
    name = "turns"
    experience_us_per_turn = 1

    def experience_turns(self):
        return 0


def test_a_composite_is_turn_based_only_through_its_one_turn_source():
    composite = CompositePerceptSource([ConversationalSource(), _Live()], ambient=[])
    assert ExperienceClockDriver(ExperienceClock(), percept_source=composite).kind == "turns"
    realtime = CompositePerceptSource([_Live()])
    assert ExperienceClockDriver(ExperienceClock(), percept_source=realtime).kind == "realtime"
    with pytest.raises(CompositeRoutingError, match="turn-based"):
        CompositePerceptSource([_TurnOnly(), _TurnOnly()])


# ── the loop's wiring ────────────────────────────────────────────────────────


def test_bio_handles_follow_the_hippocampus_the_loop_captures_into():
    hippo = _hippo()
    nac, driver = _loop_bio_handles(None, hippo, None, None)  # a Hippocampus without a hub still moves
    assert nac is None and driver.clock is hippo.experience_clock
    hub = MagicMock(hippocampus=_hippo(), nac="NAC")
    nac, driver = _loop_bio_handles(hub, None, None, None)
    assert nac == "NAC" and driver.clock is hub.hippocampus.experience_clock
    assert _loop_bio_handles(None, None, None, None)[1].clock is None


def test_the_live_tick_drifts_the_body_and_advances_the_world_clock(monkeypatch):
    import maxim.runtime.agent_loop as agent_loop

    calls = []
    monkeypatch.setattr(agent_loop, "tick_embodiment_drift", lambda e, m: calls.append((e, m)))
    driver = MagicMock()
    _loop_live_tick("EXEC", "llm-primary", driver)
    assert calls == [("EXEC", "llm-primary")]
    driver.on_live_pass.assert_called_once_with()


def _loop_calls(name: str) -> list[ast.Call]:
    # Parse the FILE, not the live attribute: other tests replace ``agent_loop.run_agentic_loop``.
    import maxim.runtime.agent_loop as agent_loop

    module = ast.parse(Path(agent_loop.__file__).read_text())
    [tree] = [n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == "run_agentic_loop"]
    return [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == name]


def test_run_agentic_loop_advances_the_clock_on_every_live_pass():
    # One driver, built once; one live tick carrying it, and no bare drift call that would bypass it.
    assert len(_loop_calls("_loop_bio_handles")) == 1
    [live] = _loop_calls("_loop_live_tick")
    assert isinstance(live.args[-1], ast.Name) and live.args[-1].id == "_loop_xclock"
    assert _loop_calls("tick_embodiment_drift") == []


# ── owed work, held by strict red gates ──────────────────────────────────────


def test_a_session_that_used_memory_on_a_frozen_clock_is_loud():
    """Phase 2c-3 FLIPPED this gate (it was the strict red xfail Phase 2a landed).

    Rewritten as the BEHAVIOUR rather than unmarked: the old gate only imported the exception name,
    which a stub would have satisfied. The failure it guards is silent -- with dt frozen at 0 every
    trace sits at R = 1 and nothing is ever forgotten -- so what has to be pinned is that a real
    session end raises. The negative arms (default strategy, advanced clock, idle session) live in
    ``test_memory_strength_strategy.py``, beside the rest of the model.
    """
    from maxim.decisions.nac import NAc
    from maxim.integration.memory_hub import MemoryHub
    from maxim.memory.encoding import EncodingSignals
    from maxim.memory.experience_clock import ExperienceClockStalled
    from maxim.similarity.ec import EntorhinalCortex
    from maxim.time.scn import SCN

    hub = MemoryHub(
        hippocampus=Hippocampus(HippocampusConfig(persistence_path=None, memory_strategy="strength")),
        scn=SCN(),
        nac=NAc(),
        ec=EntorhinalCortex(),
        _allow_raw=True,
    )
    hub.on_session_start()
    hub.hippocampus.capture(encoding=EncodingSignals.unmeasured("loop"))
    with pytest.raises(ExperienceClockStalled):
        hub.on_session_end_lightweight()


_BYPASS_HARNESSES = [
    "scripts/survival_world/water_trial.py",
    "scripts/survival_world/exp58_run.py",
    "scripts/survival_world/exp58_offline_gates.py",
    "scripts/orient_backbone/exp53_cross_context_readout.py",
]


def test_the_bypass_harness_list_names_real_files():
    root = Path(__file__).resolve().parents[2]
    assert all((root / p).is_file() for p in _BYPASS_HARNESSES)  # so the gate below can never pass vacuously


@pytest.mark.xfail(
    strict=True,
    reason="Phase 2S (#848): the scripted harnesses that call propose_via_substrate without the loop "
    "must advance the experience clock themselves (an ExperienceClockDriver per run). Flips when all do.",
)
def test_phase2s_bypass_harnesses_advance_the_clock():
    root = Path(__file__).resolve().parents[2]
    assert all("ExperienceClockDriver" in (root / p).read_text() for p in _BYPASS_HARNESSES)
