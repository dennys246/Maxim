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
    monkeypatch.setattr(module, "_sleep", lambda _s: None)  # the fake moves at once: no pacing
    monkeypatch.setattr(module, "HOLD_S", 0.05)
    monkeypatch.setattr(module, "SETTLE_S", 0.05)
    # The fake's REST read is local; the 100 ms deadline is the robot's, and a loaded CI box can miss it.
    monkeypatch.setattr("maxim.hardware.reachy.controller.FRESH_READ_DEADLINE_S", 5.0)
    return module


@pytest.fixture
def daemon(monkeypatch):
    d = FakeDaemon()
    install(monkeypatch, d)
    return d


# ── the sweep ──────────────────────────────────────────────────────────────


def test_a_full_sweep_reaches_the_clamp_on_both_sides_and_passes(sweep, daemon):
    assert sweep._run(controller(daemon), (640, 480)) == 0
    assert daemon.gotos(), "the sweep drives head-only look-ats"
    assert all(c["body_yaw"] is None for c in daemon.gotos()), "the body is never commanded"


def test_torque_off_refuses_to_start(sweep, daemon):
    daemon.control_mode = "disabled"
    assert sweep._run(controller(daemon), (640, 480)) == 2
    assert daemon.gotos() == []


@pytest.mark.parametrize("body_deg", [10.0, 3.0])
def test_a_turned_body_refuses_to_start(sweep, daemon, body_deg):
    """3° too: on 1.8.3 the watcher's gap is |body|, and it passes only at ≤ 2°."""
    daemon.body = math.radians(body_deg)
    daemon.head = head_pose(yaw=daemon.body)
    assert sweep._run(controller(daemon), (640, 480)) == 2
    assert daemon.gotos() == []


def test_a_failed_first_read_refuses_to_start(sweep, daemon):
    daemon.rest_mode = "status500"
    assert sweep._run(controller(daemon), (640, 480)) == 2
    assert daemon.gotos() == []


def test_no_camera_refuses_the_sweep(sweep, daemon):
    assert sweep._run(controller(daemon), None) == 2
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


def test_a_missed_read_mid_sweep_does_not_end_it(sweep, daemon, monkeypatch, capsys):
    real = sweep._read
    calls = {"n": 0}

    def _third_misses(ctl):
        calls["n"] += 1
        if calls["n"] == 3:
            print("  read missed: fresh readback took 408 ms (deadline 100 ms)")
            return None
        return real(ctl)

    monkeypatch.setattr(sweep, "_read", _third_misses)
    assert sweep._run(controller(daemon), (640, 480)) == 0
    assert "read missed" in capsys.readouterr().out


def test_ctrl_c_in_the_sweep_stops_and_still_summarises(sweep, daemon, monkeypatch, capsys):
    ctl = controller(daemon)
    real = ctl.look_at_pixel
    calls = {"n": 0}

    def _interrupt_on_the_third(target):
        calls["n"] += 1
        if calls["n"] == 3:
            raise KeyboardInterrupt
        return real(target)

    monkeypatch.setattr(ctl, "look_at_pixel", _interrupt_on_the_third)
    aborted: list[bool] = []
    monkeypatch.setattr(ctl, "request_abort", lambda: aborted.append(True))
    assert sweep._run(ctl, (640, 480)) == 1, "stopped short of the clamp: INCONCLUSIVE"
    out = capsys.readouterr().out
    assert "[stopped]" in out and "[summary]" in out
    assert aborted == [True]


# ── the turn check (the watcher's control) ─────────────────────────────────


def test_the_turn_check_turns_the_body_and_back(sweep, daemon, capsys):
    assert sweep._turn_check(controller(daemon)) == 0
    bodies = [c["body_yaw"] for c in daemon.gotos()]
    assert bodies[0] == pytest.approx(math.radians(sweep.TURN_DEG)) and bodies[-1] == pytest.approx(0.0)
    assert "the body peaked at 30.0°" in capsys.readouterr().out


def test_the_turn_check_samples_the_body_while_it_turns(sweep, daemon, capsys):
    real = daemon.goto_target

    def _slow(**kw):
        threading.Event().wait(0.2)  # the sampler reads meanwhile (the module's _sleep is stubbed)
        real(**kw)

    daemon.goto_target = _slow  # type: ignore[method-assign]
    assert sweep._turn_check(controller(daemon)) == 0
    assert capsys.readouterr().out.count("  turn      body=") >= 1


def test_a_body_that_does_not_arrive_fails_the_turn_check(sweep, daemon, capsys):
    real = daemon.goto_target

    def _stall(**kw):
        real(**kw)
        daemon.body -= math.radians(5)  # the body stops 5° short

    daemon.goto_target = _stall  # type: ignore[method-assign]
    ctl = controller(daemon)
    ctl._pose_distance = lambda a, b: 0.0  # the controller's own final check would refuse first; this is the script's
    assert sweep._turn_check(ctl) == 1
    assert "NOT ARRIVED" in capsys.readouterr().out
    assert len(daemon.gotos()) == 1, "no return move after a turn that did not arrive"


def test_a_refused_turn_ends_the_turn_check(sweep, daemon, monkeypatch, capsys):
    from maxim.hardware.controller import MotionOutcome

    ctl = controller(daemon)
    monkeypatch.setattr(
        ctl, "goto_target", lambda t: MotionOutcome.refused("reference", "fresh readback took 300 ms (deadline 100 ms)")
    )
    assert sweep._turn_check(ctl) == 2
    out = capsys.readouterr().out
    assert "reference" in out and "100 ms deadline" in out


def test_no_reads_after_a_turn_ends_the_turn_check(sweep, daemon, monkeypatch, capsys):
    real = sweep._read
    state = {"turned": False}

    def _blind_after_the_turn(ctl):
        return None if state["turned"] else real(ctl)

    real_goto = daemon.goto_target

    def _goto(**kw):
        real_goto(**kw)
        state["turned"] = True

    daemon.goto_target = _goto  # type: ignore[method-assign]
    monkeypatch.setattr(sweep, "_read", _blind_after_the_turn)
    assert sweep._turn_check(controller(daemon)) == 2
    assert "NO READS" in capsys.readouterr().out


def test_ctrl_c_during_a_turn_aborts_before_waiting_for_it(sweep, daemon, monkeypatch, capsys):
    ctl = controller(daemon)
    order: list[str] = []
    real_goto = daemon.goto_target
    release = threading.Event()

    def _goto(**kw):
        release.wait(2.0)
        order.append("move ended")
        real_goto(**kw)

    daemon.goto_target = _goto  # type: ignore[method-assign]

    def _abort():
        order.append("abort")
        release.set()

    monkeypatch.setattr(ctl, "request_abort", _abort)

    def _interrupt(_ctl, _phase, _t0):
        raise KeyboardInterrupt

    monkeypatch.setattr(sweep, "_sample", _interrupt)
    assert sweep._turn_check(ctl) == 2
    assert order[:2] == ["abort", "move ended"]
    assert "[stopped]" in capsys.readouterr().out


def test_the_turn_check_refuses_to_start_with_torque_off(sweep, daemon):
    daemon.control_mode = "disabled"
    assert sweep._turn_check(controller(daemon)) == 2
    assert daemon.gotos() == []


def test_the_dry_run_entry_point_runs_both_modes(sweep, monkeypatch):
    for argv in (["--dry-run", "--hold-s", "0"], ["--dry-run", "--turn-check"]):
        monkeypatch.setattr("sys.argv", ["body_yaw_sweep.py", *argv])
        assert sweep.main() == 0


def test_the_fake_starts_where_a_run_needs_it():
    d = FakeDaemon()
    assert d.control_mode == "enabled" and d.body == 0.0
