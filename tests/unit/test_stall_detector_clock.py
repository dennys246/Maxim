"""The stall detector's injected clock (orchestrator §11 slice; ``simulation/stall_detector.py``).

Every "now" the detector reads goes through ``StallDetector.CLOCK``: its idle timer AND the ages of the in-flight LLM
calls it asks the registry about. Advancing that one clock therefore reaches the idle threshold, and judges a held
call wedged by age, without waiting the wall-clock time. These drive ``_poll`` directly, one cycle at a time, on
fakes; the orchestrator characterization harness uses the same seam for the stall arm (#1259).
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace
from typing import Any

import pytest

from maxim.runtime import llm_call_registry as reg
from maxim.simulation.stall_detector import StallDetector

_THRESHOLD_S = 30.0  # stall_threshold.DEFAULT_STALL_FLOOR_S, with no lane timeout configured


@pytest.fixture(autouse=True)
def _clean(monkeypatch: pytest.MonkeyPatch) -> Any:
    for name in (
        "MAXIM_STALL_FLOOR_S",
        "MAXIM_SIM_STALL_THRESHOLD_S",
        "MAXIM_STALL_MARGIN_S",
        "MAXIM_LANE_LARGE_TIMEOUT_S",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MAXIM_SIM_HARD_ABORT", "0")  # never arm the forced-exit backstop in a test process
    reg._reset_for_tests()
    yield
    reg._reset_for_tests()


class _Clock:
    """The wall clock plus an offset the test advances: registry entries are stamped on the wall clock, so their ages
    read on this clock grow with the offset."""

    def __init__(self) -> None:
        self.offset = 0.0

    def __call__(self) -> float:
        return time.time() + self.offset


def _detector(monkeypatch: pytest.MonkeyPatch, clock: _Clock) -> tuple[StallDetector, list[str]]:
    monkeypatch.setattr(StallDetector, "CLOCK", staticmethod(clock))
    injected: list[str] = []
    spinner = SimpleNamespace(planning_window=False, update_if_planning=lambda _t: False, update=lambda _t: None)
    bridge = SimpleNamespace(
        turn_count=1, turn_in_progress=False, get_all_actions=lambda: [], _spinner=spinner, finish_context={}
    )
    detector = StallDetector(
        stop_event=threading.Event(),
        bridge=bridge,
        orchestrator_source=SimpleNamespace(inject_cli=lambda text, **_k: injected.append(text)),
        orch_executor=SimpleNamespace(_tools_attempted=[]),
        goal="check the clock",
        max_turns=0,
        is_interactive=False,
        is_observe_only=False,
        paused=[False],
        sim_prompt_handler=None,
        display_status=lambda _m: None,
        emit=lambda *_a, **_k: None,
        get_active_display=lambda: None,
    )
    detector._configure()
    detector._poll()  # the first cycle sees turn 1 advance: every idle tracker starts from now
    return detector, injected


def test_the_idle_nudge_fires_on_the_injected_clock_without_waiting(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = _Clock()
    detector, injected = _detector(monkeypatch, clock)

    clock.offset = _THRESHOLD_S - 1
    assert detector._poll() is False and injected == [], "under the threshold: no nudge"
    clock.offset = _THRESHOLD_S + 1
    assert detector._poll() is False
    assert len(injected) == 1 and injected[0].startswith(f"SYSTEM: Stall detected ({int(_THRESHOLD_S + 1)}s idle,")


def test_a_held_call_is_aged_on_the_detectors_clock_busy_then_wedged(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-streaming planning call keeps the narrator BUSY until it has outlived the time it was allowed plus the
    margin, both read on the detector's clock; then it is wedged and the idle nudge fires (#1042, D12/D14)."""
    from maxim.agents.llm_types import PLANNING_LANE

    clock = _Clock()
    detector, injected = _detector(monkeypatch, clock)
    call = reg.register_call_start(tier=PLANNING_LANE, streams=False, allowed_s=300.0)
    try:
        clock.offset = _THRESHOLD_S + 1  # idle past the threshold, but the call is young: busy, suppressed
        assert detector._poll() is False and injected == []
        clock.offset = 300.0 + 10.0 + 1  # allowed_s + DEFAULT_STALL_MARGIN_S, passed: wedged by age
        assert detector._poll() is False
        assert len(injected) == 1 and injected[0].startswith("SYSTEM: Stall detected (")
        assert detector._wedged_byte_silence is not None and detector._wedged_byte_silence > 310.0
    finally:
        reg.register_call_end(call)


def test_the_default_clock_is_the_wall_clock() -> None:
    assert StallDetector.CLOCK is time.time
