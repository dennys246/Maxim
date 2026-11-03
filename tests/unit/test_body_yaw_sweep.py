"""The body-yaw sweep script (#1499 PR 3): its start refusals and its verdicts, against the shared fake."""

from __future__ import annotations

import importlib.util
import math
import threading
from pathlib import Path

import pytest

from tests.unit._reachy_fakes import FakeDaemon, controller, head_pose, install

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def sweep(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "_body_yaw_sweep", REPO / "scripts" / "orient_backbone" / "body_yaw_sweep.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.time, "sleep", lambda _s: None)  # the fake moves at once: no pacing
    # The fake's REST read is local; the 100 ms deadline is the robot's, and a loaded CI box can miss it.
    monkeypatch.setattr("maxim.hardware.reachy.controller.FRESH_READ_DEADLINE_S", 5.0)
    monkeypatch.setattr(module, "HOLD_S", 0.05)
    return module


@pytest.fixture
def daemon(monkeypatch):
    d = FakeDaemon()
    install(monkeypatch, d)
    return d


def test_a_full_sweep_reaches_the_clamp_on_both_sides_and_passes(sweep, daemon):
    assert sweep._run(controller(daemon), (640, 480)) == 0
    looks = [c for c in daemon.gotos() if c["body_yaw"] is None]
    assert looks, "the sweep drives head-only look-ats"
    assert all(c["body_yaw"] is None for c in daemon.gotos()), "the body is never commanded"


def test_torque_off_refuses_to_start(sweep, daemon):
    daemon.control_mode = "disabled"
    assert sweep._run(controller(daemon), (640, 480)) == 2
    assert daemon.gotos() == []


def test_a_turned_body_refuses_to_start(sweep, daemon):
    daemon.body = math.radians(10)
    daemon.head = head_pose(yaw=daemon.body)
    assert sweep._run(controller(daemon), (640, 480)) == 2
    assert daemon.gotos() == []


def test_a_body_that_turns_during_the_sweep_is_reported(sweep, daemon, capsys):
    """The auto-yaw symptom: the daemon turns the body behind a head-only move."""
    real = daemon.goto_target

    def _goto_and_drift(**kw):
        real(**kw)
        if len(daemon.gotos()) <= 5:
            daemon.body += math.radians(1)  # the first moves drag the body a degree each

    daemon.goto_target = _goto_and_drift  # type: ignore[method-assign]
    assert sweep._run(controller(daemon), (640, 480)) == 1
    assert "BODY MOVED" in capsys.readouterr().out


def test_a_sweep_that_stops_short_of_55_degrees_is_inconclusive(sweep, daemon, monkeypatch, capsys):
    ctl = controller(daemon)
    monkeypatch.setattr(ctl, "_MAX_HEAD_REL_YAW_RAD", math.radians(40))
    assert sweep._run(ctl, (640, 480)) == 1
    assert "INCONCLUSIVE" in capsys.readouterr().out


def test_hold_s_sets_the_hold_at_each_clamp(sweep, daemon):
    assert sweep._run(controller(daemon), (640, 480), hold_s=0.0) == 0
    no_hold = len(daemon.gotos())
    daemon.calls.clear()
    assert sweep._run(controller(daemon), (640, 480), hold_s=0.2) == 0
    assert len(daemon.gotos()) > no_hold


def test_no_camera_refuses_the_sweep(sweep, daemon):
    assert sweep._run(controller(daemon), None) == 2
    assert daemon.gotos() == []


def _lagging(daemon, gap_deg: float) -> None:
    """While a goto runs, the daemon's target leads the present body by *gap_deg* (servo lag)."""
    real = daemon.goto_target

    def _goto(**kw):
        daemon.target_gap = math.radians(gap_deg)
        threading.Event().wait(0.3)  # the sampler reads the gap meanwhile (time.sleep is stubbed)
        real(**kw)

    daemon.goto_target = _goto  # type: ignore[method-assign]


def test_the_lag_check_passes_on_a_gap_that_closes(sweep, daemon, capsys):
    _lagging(daemon, 3.0)
    assert sweep._lag_check(controller(daemon)) == 0
    out = capsys.readouterr().out
    assert "largest |gap| during the turns: 3.0°" in out
    bodies = [c["body_yaw"] for c in daemon.gotos()]
    assert bodies[0] == pytest.approx(math.radians(sweep.LAG_TURN_DEG)) and bodies[-1] == pytest.approx(0.0)


def test_the_lag_check_reports_no_gap_as_inconclusive(sweep, daemon, capsys):
    assert sweep._lag_check(controller(daemon)) == 1
    assert "NO GAP SEEN" in capsys.readouterr().out


def test_the_lag_check_stops_on_a_gap_that_stays(sweep, daemon, capsys):
    daemon.target_gap_after_goto = math.radians(5)
    assert sweep._lag_check(controller(daemon)) == 1
    assert "GAP STAYED" in capsys.readouterr().out
    assert len(daemon.gotos()) == 1, "no return move is attempted while the gap is open"


def test_the_lag_check_refuses_to_start_with_torque_off(sweep, daemon):
    daemon.control_mode = "disabled"
    assert sweep._lag_check(controller(daemon)) == 2
    assert daemon.gotos() == []


def test_a_missed_read_during_the_lag_check_does_not_end_it(sweep, daemon, monkeypatch, capsys):
    from maxim.exceptions import MotionRefused

    _lagging(daemon, 3.0)
    real = sweep._read_gap
    calls = {"n": 0}

    def _first_misses(ctl):
        calls["n"] += 1
        if calls["n"] == 1:
            raise MotionRefused("reference", "fresh readback took 408 ms (deadline 100 ms)")
        return real(ctl)

    monkeypatch.setattr(sweep, "_read_gap", _first_misses)
    assert sweep._lag_check(controller(daemon)) == 0
    assert "read missed" in capsys.readouterr().out
