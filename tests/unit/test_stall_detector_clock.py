"""The stall detector's injected clock (orchestrator §11 slice; ``simulation/stall_detector.py``).

Every "now" the detector reads goes through ``StallDetector.CLOCK``: its idle timer, the nudge cooldown, the hard
abort's stall duration AND the ages of the in-flight LLM calls it asks the registry about. Advancing that one clock
therefore reaches each threshold, and judges a held call wedged by age, without waiting the wall-clock time. These
drive ``_poll`` directly, one cycle at a time, on fakes; #1259 re-homes the orchestrator characterization harness's
stall arm onto the same seam. Each branch ``_poll`` maps from the old closure's ``continue``/``break`` runs here.
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace
from typing import Any

import pytest

from maxim.runtime import llm_call_registry as reg
from maxim.runtime.stall_threshold import DEFAULT_STALL_FLOOR_S, DEFAULT_STALL_MARGIN_S
from maxim.simulation import stall_detector as SD
from maxim.simulation.stall_detector import StallDetector

_THRESHOLD_S = DEFAULT_STALL_FLOOR_S  # no lane timeout configured: the threshold is the floor


@pytest.fixture(autouse=True)
def _clean(monkeypatch: pytest.MonkeyPatch) -> Any:
    for name in (
        "MAXIM_STALL_FLOOR_S",
        "MAXIM_SIM_STALL_THRESHOLD_S",
        "MAXIM_STALL_MARGIN_S",
        "MAXIM_LANE_LARGE_TIMEOUT_S",
        "MAXIM_SIM_PING_PONG_BUDGET",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MAXIM_SIM_HARD_ABORT", "0")  # never arm the forced-exit backstop unless a test fakes it
    reg._reset_for_tests()
    yield
    reg._reset_for_tests()


class _Clock:
    """The wall clock plus an offset the test advances: the clock's contract (module docstring). Registry entries are
    stamped on the wall clock, so their ages read on this clock grow with the offset."""

    def __init__(self) -> None:
        self.offset = 0.0

    def __call__(self) -> float:
        return time.time() + self.offset


def _detector(
    monkeypatch: pytest.MonkeyPatch, clock: _Clock, *, max_turns: int = 0, paused: list[bool] | None = None
) -> tuple[StallDetector, list[str], SimpleNamespace]:
    monkeypatch.setattr(StallDetector, "CLOCK", staticmethod(clock))
    injected: list[str] = []
    spinner = SimpleNamespace(planning_window=False, texts=[], updates=[])
    spinner.update_if_planning = lambda t: spinner.texts.append(t) or spinner.planning_window
    spinner.update = spinner.updates.append
    bridge = SimpleNamespace(
        turn_count=1,
        turn_in_progress=False,
        get_all_actions=lambda: [],
        _spinner=spinner,
        finish_context={},
        finished=[],
    )
    bridge.finish = lambda: bridge.finished.append(True)
    detector = StallDetector(
        stop_event=threading.Event(),
        bridge=bridge,
        orchestrator_source=SimpleNamespace(inject_cli=lambda text, **_k: injected.append(text)),
        orch_executor=SimpleNamespace(_tools_attempted=[]),
        goal="check the clock",
        max_turns=max_turns,
        is_interactive=False,
        is_observe_only=False,
        paused=paused if paused is not None else [False],
        sim_prompt_handler=None,
        display_status=lambda _m: None,
        emit=lambda *_a, **_k: None,
        get_active_display=lambda: None,
    )
    detector._configure()
    detector._poll()  # the first cycle sees turn 1 advance: every idle tracker starts from now
    return detector, injected, bridge


def test_the_idle_nudge_fires_on_the_injected_clock_without_waiting(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = _Clock()
    detector, injected, _ = _detector(monkeypatch, clock)

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
    detector, injected, _ = _detector(monkeypatch, clock)
    call = reg.register_call_start(tier=PLANNING_LANE, streams=False, allowed_s=300.0)
    try:
        clock.offset = _THRESHOLD_S + 1  # idle past the threshold, but the call is young: busy, suppressed
        assert detector._poll() is False and injected == []
        clock.offset = 300.0 + DEFAULT_STALL_MARGIN_S + 1  # its allowed time plus the margin, passed: wedged by age
        assert detector._poll() is False
        assert len(injected) == 1 and injected[0].startswith("SYSTEM: Stall detected (")
        assert detector._wedged_byte_silence is not None and detector._wedged_byte_silence > 310.0
    finally:
        reg.register_call_end(call)


def test_a_turn_in_progress_restarts_the_idle_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = _Clock()
    detector, injected, bridge = _detector(monkeypatch, clock)
    bridge.turn_in_progress = True
    clock.offset = _THRESHOLD_S + 1
    assert detector._poll() is False and injected == [], "waiting on the agent's turn is activity"
    bridge.turn_in_progress = False
    assert detector._poll() is False and injected == [], "the idle clock restarted at that poll, on the clock"


def test_a_paused_run_is_never_nudged_and_resumes_with_a_fresh_idle_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = _Clock()
    paused = [True]
    detector, injected, _ = _detector(monkeypatch, clock, paused=paused)
    clock.offset = _THRESHOLD_S * 3
    assert detector._poll() is False and injected == []
    paused[0] = False
    assert detector._poll() is False and injected == [], "the pause reset the idle tracker"
    clock.offset = _THRESHOLD_S * 4 + 1
    assert detector._poll() is False and len(injected) == 1


def test_max_turns_ends_the_run(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = _Clock()
    detector, injected, bridge = _detector(monkeypatch, clock, max_turns=1)
    # _detector's first poll already met the cap (turn_count 1 >= max_turns 1): the run ended there.
    assert detector._stop_event.is_set() and bridge.finished == [True]
    assert bridge.finish_context["status"] == "max_turns" and injected == []
    assert detector._poll() is False, "warned once: a later cycle does not end it again"


def test_ping_pong_nudges_and_the_cooldown_runs_on_the_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = _Clock()
    detector, injected, _ = _detector(monkeypatch, clock)
    detector._orch_executor._tools_attempted = ["observe_actions"] * 6  # the budget (6) of tool calls, no turn
    assert detector._poll() is False
    assert len(injected) == 1 and injected[0].startswith("SYSTEM: Ping-pong detected — you have called 6 tools")
    detector._orch_executor._tools_attempted *= 2  # six more: a second ping-pong
    clock.offset = SD._NUDGE_COOLDOWN_S - 1
    assert detector._poll() is False and len(injected) == 1, "inside the cooldown, read on the clock"
    clock.offset = SD._NUDGE_COOLDOWN_S + 1
    assert detector._poll() is False and len(injected) == 2


def test_a_failing_suppression_decision_is_reported_and_does_not_suppress(monkeypatch: pytest.MonkeyPatch) -> None:
    from maxim.runtime import stall_threshold

    def boom(**_kwargs: Any) -> Any:
        raise RuntimeError("registry down")

    reported: list[str] = []
    monkeypatch.setattr(stall_threshold, "stall_suppression", boom)
    monkeypatch.setattr("maxim.utils.logging.log_swallowed_exception", lambda **k: reported.append(k["site"]))
    clock = _Clock()
    detector, injected, _ = _detector(monkeypatch, clock)
    clock.offset = _THRESHOLD_S + 1
    assert detector._poll() is False and len(injected) == 1
    assert reported == ["stall_detector.py:StallDetector._suppressed:stall_suppression"]


def test_the_spinner_tells_the_truth_then_the_hard_abort_ends_the_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """Three idle nudges with no call in flight, then the D12 persistent-stall abort, every duration on the clock:
    the spinner first reports the lost planning turn, the third nudge is the REPEATED STALL text, and the abort
    records ``llm_wedged``. The forced-exit backstop's thread is faked: it would end the test process."""
    started: list[str] = []

    class _NoThread:
        def __init__(self, *, target: Any, name: str, daemon: bool) -> None:
            self.name = name

        def start(self) -> None:
            started.append(self.name)

    clock = _Clock()
    detector, injected, bridge = _detector(monkeypatch, clock)
    bridge._spinner.planning_window = True
    monkeypatch.setenv("MAXIM_SIM_HARD_ABORT", "1")
    monkeypatch.setattr(SD.threading, "Thread", _NoThread)
    for nudge in (1, 2, 3):
        clock.offset = nudge * (_THRESHOLD_S + 1)
        assert detector._poll() is False and len(injected) == nudge
    assert injected[2].startswith("SYSTEM: REPEATED STALL (#3).")
    assert bridge._spinner.texts[0].startswith("⚠ no LLM call in flight — planning turn may be lost")
    assert detector._spinner_truth_overridden
    clock.offset = max(3 * _THRESHOLD_S, _THRESHOLD_S + 120.0) + 1  # should_hard_abort's persistent-stall route
    assert detector._poll() is True
    assert detector._stop_event.is_set() and bridge.finished == [True] and started == ["sim.hard_abort"]
    assert (
        bridge.finish_context["status"] == "llm_wedged" and bridge.finish_context["initiated_by"] == "stall_hard_abort"
    )
    assert bridge._spinner.updates[-1].startswith("🛑 llm_wedged — aborting sim (3 nudges unconsumed)")


def test_the_default_clock_is_the_wall_clock() -> None:
    assert StallDetector.CLOCK is time.time
