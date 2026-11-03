#!/usr/bin/env python
"""The body-yaw hardware test: drive the look-at tracking path to the ±65° neck clamp (#1499 PR 3, plan §13).

Auto-yaw acts past 65° of head-to-body yaw, and the controller's clamp is exactly ±65°, so head tracking can reach
the daemon's threshold. A conversational "look far left" may stop short of it, and no other rig script drives the
tracking path (``ReachyMiniController.look_at_pixel``). This one does, the way Default Network tracking does: a
look-at at the image's left edge every 0.3 s (``at_least``), so the head walks to the clamp; about 5 s of edge
look-ats there; the same to the right; then back to centre. The head moves; the body is never commanded.

Run it with pulse's watcher (REST only, 5 Hz; a REST reader cannot flip the setting). The daemon (SDK 1.8.3) does
not serve ``target_body_yaw`` (its ``FullState`` response model drops it), so the watcher's "told" column reads 0
and its "gap" is the body's yaw itself: in a head-only run that starts at body 0, that is exactly the auto-yaw
symptom. Pulse grades the run (hold gap ≤ ~2° in a run that reached ≥ 55° relative). This script prints the
relative yaw it reached and whether the body moved.

``--turn-check`` is the watcher's control, run first: it proves the watcher sees a body that moves. It turns the
body 30° over 3 s through the controller and back, printing its own body-yaw trace at 10 Hz. The control PASSES
when the watcher's whole-run gap peaks at the script's peak (within ~2°) and returns to ~0. A watcher that stays
flat while the body turned is blind, and the control FAILS. The watcher's own verdict line reads INCONCLUSIVE in
this run by design: the head rides along with the body, far below 55° relative.

    ~/Scripts/Maxim/.venv/bin/python scripts/orient_backbone/body_yaw_sweep.py --host <ip> --turn-check
    ~/Scripts/Maxim/.venv/bin/python scripts/orient_backbone/body_yaw_sweep.py --host <ip> [--hold-s 5]
    python scripts/orient_backbone/body_yaw_sweep.py --dry-run [--turn-check]   # offline, the shared test fake

It refuses to start unless motor torque is on and the body is within 5° of 0. Ctrl+C stops the motion at the
current chunk and leaves the robot where it is (the head always inside the clamp).

Exit 0: the sweep reached ≥ 55° relative on both sides and the body held (turn check: the body reached the turn
and came back). 1: INCONCLUSIVE (a side never reached 55°) or the body moved (turn check: the body did not reach
the turn or did not come back). 2: it could not start, a read failed, or a move was refused.
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import math
import sys
import threading
import time
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
TRACK_PERIOD_S = 0.3  # the DN tracking cadence (plan §13)
HOLD_S = 5.0  # edge look-ats at the clamp (--hold-s)
WALK_CAP_S = 20.0  # a walk to one side ends here even if the yaw is still rising
PLATEAU_STEPS = 3  # a walk ends after this many steps without 0.5° more yaw
REACHED_DEG = 55.0  # pulse's bar: a run below this is INCONCLUSIVE
BODY_START_DEG = 5.0  # the body must start this close to 0
BODY_HELD_DEG = 2.0  # a head-only sweep that moves the body further than this is the auto-yaw symptom
TURN_DEG = 30.0  # the turn check's body turn, over TURN_S, then back
TURN_S = 3.0
SAMPLE_S = 0.1  # the turn check prints the body at 10 Hz (the watcher reads at 5 Hz)
SETTLE_S = 1.0  # after each turn, the body must end within ARRIVED_DEG of where it was sent
ARRIVED_DEG = 2.0
_DEADLINE_HINT = (
    "  (the controller's fresh read has a 100 ms deadline; other REST pollers, the watcher included, compete for "
    "it: re-run, or lower their rate)"
)

_sleep = time.sleep  # the module's pacing; tests replace this, not the process-wide time.sleep


@contextlib.contextmanager
def _dry_run_controller() -> Iterator[tuple[Any, tuple[int, int] | None]]:
    """A controller over the shared test fake (tests/unit/_reachy_fakes.py): no robot, no network."""
    import pytest

    sys.path.insert(0, str(REPO))
    from tests.unit._reachy_fakes import FakeDaemon, controller, install

    mp = pytest.MonkeyPatch()
    try:
        daemon = FakeDaemon()
        install(mp, daemon)
        yield controller(daemon), (640, 480)
    finally:
        mp.undo()


@contextlib.contextmanager
def _robot_controller(host: str, media_backend: str) -> Iterator[tuple[Any, tuple[int, int] | None]]:
    from maxim.hardware.reachy.controller import ReachyMiniController

    # The sweep needs a media backend with the camera: look_at_image needs its intrinsics. The turn check does not.
    ctl = ReachyMiniController(host=host, connection_mode="network", media_backend=media_backend)
    if not ctl.connect(timeout=15.0):
        raise SystemExit(
            f"[FAIL] ReachyMiniController.connect() returned False (curl http://{host}:8000/api/daemon/status)"
        )
    try:
        camera = getattr(getattr(ctl.mini, "media_manager", None), "camera", None)
        yield ctl, (tuple(camera.resolution) if camera is not None else None)
    finally:
        ctl.disconnect()


def _read(ctl: Any) -> tuple[float, float, str] | None:
    """(relative head yaw °, body yaw °, control_mode) from the daemon's REST state, or None when the read missed."""
    from maxim.exceptions import MotionRefused

    import maxim.hardware.reachy.motion_path as mp

    try:
        pose, control_mode = mp.parse_full_state(ctl._get_state_full())  # noqa: SLF001 - the controller's own read
    except MotionRefused as e:
        print(f"  read missed: {e.reason}")
        return None
    return math.degrees(mp.neck_twist(pose.head, pose.body)), math.degrees(pose.body), control_mode


def _ready(ctl: Any, resolution: tuple[int, int] | None) -> float | None:
    """The start body yaw (°) when torque is on and the body is near 0, else None (printed)."""
    read = _read(ctl)
    if read is None:
        print("[FAIL] the first read of the robot failed")
        return None
    rel, body, control_mode = read
    print(f"[start] torque={control_mode}  body={body:+.1f}°  relative head yaw={rel:+.1f}°  image={resolution}")
    if control_mode != "enabled":
        print("[FAIL] motor torque is not on: wake the robot first (this script never wakes it)")
        return None
    if abs(body) > BODY_START_DEG:
        print(f"[FAIL] the body is at {body:+.1f}°, not within {BODY_START_DEG:.0f}° of 0: re-centre it first")
        return None
    return body


def _report_refusal(what: str, outcome: Any) -> None:
    reason = outcome.reason if outcome is not None else "no outcome"
    print(f"  {what}: {outcome.kind if outcome is not None else 'error'}: {reason}")
    if "deadline" in str(reason):
        print(_DEADLINE_HINT)


class _Sweep:
    def __init__(self, ctl: Any, resolution: tuple[int, int], hold_s: float = HOLD_S) -> None:
        self.ctl = ctl
        self.hold_s = hold_s
        self.width, self.height = int(resolution[0]), int(resolution[1])
        self.outcomes: Counter[str] = Counter()
        self.rel_max = -math.inf
        self.rel_min = math.inf
        self.body_max = 0.0
        self.t0 = time.monotonic()

    def look(self, phase: str, u: int) -> float | None:
        """One tracking look-at at column *u*, paced to the cadence; the relative yaw reached (°), None if unread."""
        from maxim.hardware.controller import PixelTarget, as_outcome

        started = time.monotonic()
        outcome = as_outcome(
            self.ctl.look_at_pixel(
                PixelTarget(u=u, v=self.height // 2, duration=TRACK_PERIOD_S, duration_policy="at_least")
            )
        )
        self.outcomes["ok" if outcome.ok else (outcome.kind or "refused")] += 1
        verdict = "ok" if outcome.ok else f"{outcome.kind}: {outcome.reason}"
        read = _read(self.ctl)
        if read is not None:
            rel, body, _ = read
            self.rel_max, self.rel_min = max(self.rel_max, rel), min(self.rel_min, rel)
            self.body_max = max(self.body_max, abs(body))
            print(
                f"  {time.monotonic() - self.t0:6.1f}s  {phase:<11}  u={u:<4}  "
                f"rel={rel:+6.1f}°  body={body:+5.1f}°  {verdict}"
            )
        _sleep(max(0.0, TRACK_PERIOD_S - (time.monotonic() - started)))
        return None if read is None else read[0]

    def walk(self, phase: str, u: int, sign: int) -> None:
        """Look at column *u* until the yaw (in direction *sign*) stops rising, or the cap."""
        best, flat, deadline = -math.inf, 0, time.monotonic() + WALK_CAP_S
        while flat < PLATEAU_STEPS and time.monotonic() < deadline:
            rel = self.look(phase, u)
            if rel is None:
                continue
            flat = 0 if sign * rel > best + 0.5 else flat + 1
            best = max(best, sign * rel)

    def hold(self, phase: str, u: int) -> None:
        end = time.monotonic() + self.hold_s
        while time.monotonic() < end:
            self.look(phase, u)


def _sample(ctl: Any, phase: str, t0: float) -> float | None:
    """One printed body-yaw sample (°), or None when the read missed (a slow read does not end the run)."""
    read = _read(ctl)
    if read is None:
        return None
    print(f"  {time.monotonic() - t0:6.1f}s  {phase:<8}  body={read[1]:+6.1f}°")
    return read[1]


def _turn_and_sample(ctl: Any, phase: str, to_deg: float, t0: float) -> tuple[bool, list[float]]:
    """Turn the body to *to_deg* through the controller while sampling it; (move ok, body yaws seen meanwhile)."""
    from maxim.hardware.controller import MotionTarget, as_outcome

    result: list[Any] = []
    mover = threading.Thread(
        target=lambda: result.append(
            as_outcome(ctl.goto_target(MotionTarget(body_yaw=math.radians(to_deg), duration=TURN_S)))
        ),
        daemon=True,
    )
    bodies: list[float] = []
    mover.start()
    try:
        while mover.is_alive():
            body = _sample(ctl, phase, t0)
            if body is not None:
                bodies.append(body)
            _sleep(SAMPLE_S)
    except KeyboardInterrupt:
        ctl.request_abort()  # before the join: the move ends at its next chunk instead of running its 3 s
        mover.join()
        raise
    mover.join()
    outcome = result[0] if result else None
    if outcome is None or not outcome.ok:
        _report_refusal(phase, outcome)
        return False, bodies
    return True, bodies


def _settled_body(ctl: Any, phase: str, t0: float) -> float | None:
    """The last body yaw read over SETTLE_S after a turn, or None when every read missed."""
    last = None
    end = time.monotonic() + SETTLE_S
    while time.monotonic() < end:
        body = _sample(ctl, phase, t0)
        last = body if body is not None else last
        _sleep(SAMPLE_S)
    return last


def _turn_check(ctl: Any) -> int:
    """The watcher's control: a body turn and back that the watcher must see (plan §13, owner 2026-11-03)."""
    if _ready(ctl, None) is None:
        return 2
    print(f"[turn-check] body to {TURN_DEG:+.0f}° over {TURN_S:.0f} s and back")
    t0 = time.monotonic()
    seen: list[float] = []
    try:
        for phase, to_deg in (("turn", TURN_DEG), ("return", 0.0)):
            ok, bodies = _turn_and_sample(ctl, phase, to_deg, t0)
            seen += bodies
            if not ok:
                return 2
            ended = _settled_body(ctl, "settle", t0)
            if ended is None:
                print(f"\n  NO READS after the {phase}: where the body ended could not be checked")
                return 2
            seen.append(ended)
            if abs(ended - to_deg) > ARRIVED_DEG:
                print(f"\n  NOT ARRIVED: the body ended at {ended:+.1f}° after the {phase} to {to_deg:+.0f}°")
                return 1
    except KeyboardInterrupt:
        print("\n[stopped] Ctrl+C: motion stopped at the current chunk")
        return 2
    peak = max((abs(b) for b in seen), default=0.0)
    print("\n[summary]")
    print(f"  the body peaked at {peak:.1f}° and came back to within {ARRIVED_DEG:.0f}° of 0 ({len(seen)} samples)")
    print(f"  Compare with the watcher's whole-run gap: it should peak at ~{peak:.0f}° (within ~2°) and return to ~0,")
    print("  which is the control PASSING; a watcher that stayed flat while the body turned is blind (FAILS).")
    return 0


def _run(ctl: Any, resolution: tuple[int, int] | None, hold_s: float = HOLD_S) -> int:
    from maxim.hardware.controller import MotionTarget, as_outcome

    if resolution is None:
        print("[FAIL] no camera on this connection: look_at_pixel needs the camera's intrinsics. Is another app")
        print("       holding media, or was --media-backend no_media passed? (the turn check runs without one)")
        return 2
    body = _ready(ctl, resolution)
    if body is None:
        return 2
    sweep = _Sweep(ctl, resolution, hold_s)
    left, right = 1, sweep.width - 2  # look_at_image asserts 0 < u < width
    try:
        sweep.walk("walk-left", left, +1)
        sweep.hold("hold-left", left)
        sweep.walk("walk-right", right, -1)
        sweep.hold("hold-right", right)
        centre = as_outcome(ctl.goto_target(MotionTarget(head_roll=0.0, head_pitch=0.0, head_yaw=0.0, duration=1.5)))
        if centre.ok:
            print("  back to centre: ok")
        else:
            _report_refusal("back to centre", centre)
    except KeyboardInterrupt:
        ctl.request_abort()
        print("\n[stopped] Ctrl+C: motion stopped at the current chunk; the head stays where it is")
    end = _read(ctl)
    if end is not None:
        sweep.body_max = max(sweep.body_max, abs(end[1]))

    print("\n[summary]")
    print(f"  relative yaw reached: left {sweep.rel_max:+.1f}°, right {sweep.rel_min:+.1f}°  (clamp ±65°)")
    print(f"  largest |body yaw| seen: {sweep.body_max:.1f}°  (it started at {body:+.1f}°)")
    print(f"  look-at outcomes: {dict(sweep.outcomes)}")
    if sweep.body_max - abs(body) > BODY_HELD_DEG:
        print("  BODY MOVED: the body turned during a head-only sweep, the auto-yaw symptom. Check pulse's watcher.")
        return 1
    if min(sweep.rel_max, -sweep.rel_min) < REACHED_DEG:
        print(
            f"  INCONCLUSIVE: a side never reached {REACHED_DEG:.0f}° relative, so auto-yaw's threshold was not tested"
        )
        return 1
    print(f"  REACHED: ≥ {REACHED_DEG:.0f}° on both sides with the body held; pulse's watcher grades the hold")
    return 0


def _quiet_the_clamp_warning() -> None:
    """At the clamp every look-at WARNs that it clamped; the table's ``rel=±65.0°`` already says so."""
    logging.getLogger("maxim.hardware.reachy.controller").addFilter(
        lambda record: not str(record.msg).startswith("goto_target: %s %.3f rad exceeds physical limit")
    )


def main() -> int:
    _quiet_the_clamp_warning()
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--host", default=None, help="Reachy daemon IP (or $MAXIM_REACHY_HOST)")
    ap.add_argument("--dry-run", action="store_true", help="run offline against the shared test fake")
    ap.add_argument("--turn-check", action="store_true", help="the watcher's control: a body turn and back")
    ap.add_argument("--hold-s", type=float, default=HOLD_S, help="seconds of edge look-ats at each clamp")
    ap.add_argument(
        "--media-backend", default="default", help="the SDK media backend (no_media works for --turn-check only)"
    )
    args = ap.parse_args()

    def _go(ctl: Any, resolution: tuple[int, int] | None) -> int:
        return _turn_check(ctl) if args.turn_check else _run(ctl, resolution, args.hold_s)

    if args.dry_run:
        print("[dry-run] the shared fake (tests/unit/_reachy_fakes.py): no robot is moved")
        with _dry_run_controller() as (ctl, resolution):
            return _go(ctl, resolution)

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import live_1_smoke as smoke

    host, source = smoke.resolve_host(args.host)
    if host is None:
        print("[FAIL] no robot address: --host <ip> or export MAXIM_REACHY_HOST=<ip>")
        return 2
    print(f"[host] using {host} (source: {source})")
    smoke.preflight(host)
    with _robot_controller(host, args.media_backend) as (ctl, resolution):
        return _go(ctl, resolution)


if __name__ == "__main__":
    sys.exit(main())
