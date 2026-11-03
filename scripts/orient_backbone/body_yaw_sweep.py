#!/usr/bin/env python
"""The body-yaw hardware test: drive the look-at tracking path to the ±65° neck clamp (#1499 PR 3, plan §13).

Auto-yaw acts past 65° of head-to-body yaw, and the controller's clamp is exactly ±65°, so head tracking can reach
the daemon's threshold. A conversational "look far left" may stop short of it, and no other rig script drives the
tracking path (``ReachyMiniController.look_at_pixel``). This one does, the way Default Network tracking does: a
look-at at the image's left edge every 0.3 s (``at_least``), so the head walks to the clamp; about 5 s of edge
look-ats there; the same to the right; then back to centre. The head moves; the body is never commanded.

Run it with pulse's watcher (``GET :8000/api/state/full?...&with_target_body_yaw=true`` at 5 Hz; a REST reader
cannot flip the setting). Pulse grades the run: the present body yaw tracks the target body yaw (gap ≤ ~2°) in a
run that reached ≥ 55° relative. This script prints the relative yaw it reached and whether the body moved.

``--lag-check`` is the watcher's control, run first. It turns the body 30° over 3 s through the controller and
back, and samples the same REST field the watcher reads (``target_body_yaw − body_yaw``) at 10 Hz itself. The
daemon's goto ramps its body target along the move, so the gap is only servo lag (a few degrees), not a jump. The
control PASSES when the watcher's largest gap during the turns agrees with this script's within ~1° and both are
non-zero. A watcher that reads ~0 while this script reads a gap is blind, and the control FAILS. The watcher's own
verdict line reads INCONCLUSIVE in this run by design: the head stays centred, far below 55° relative.

    ~/Envs/maxim-env/bin/python scripts/orient_backbone/body_yaw_sweep.py --host <ip> --lag-check
    ~/Envs/maxim-env/bin/python scripts/orient_backbone/body_yaw_sweep.py --host <ip> [--hold-s 5]
    python scripts/orient_backbone/body_yaw_sweep.py --dry-run [--lag-check]   # offline, the shared test fake

It refuses to start unless motor torque is on and the body is within 5° of 0. Ctrl+C stops the motion at the
current chunk and leaves the head where it is (always inside the clamp).

Exit 0: the sweep reached ≥ 55° relative on both sides and the body held (lag check: both turns completed, a
non-zero gap was seen and it returned to ≤ 2°). 1: INCONCLUSIVE (a side never reached 55°; no gap seen) or the
body moved (lag check: the gap did not return). 2: it could not start or a move failed.
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
LAG_TURN_DEG = 30.0  # the lag check's body turn, over LAG_TURN_S, then back
LAG_TURN_S = 3.0
LAG_SAMPLE_S = 0.1  # the lag check samples the gap at 10 Hz (the watcher: 5 Hz)
LAG_SETTLE_S = 1.0  # after each turn: the gap must be back within SETTLED_GAP_DEG
SETTLED_GAP_DEG = 2.0


@contextlib.contextmanager
def _dry_run_controller() -> Iterator[tuple[Any, Any]]:
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

    # The sweep needs a media backend with the camera: look_at_image needs its intrinsics. The lag check does not.
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


def _read(ctl: Any) -> tuple[float, float, str]:
    """(relative head yaw °, body yaw °, control_mode) from the daemon's REST state, never the WS mirror."""
    import maxim.hardware.reachy.motion_path as mp

    pose, control_mode = mp.parse_full_state(ctl._get_state_full())  # noqa: SLF001 - the controller's own read
    return math.degrees(mp.neck_twist(pose.head, pose.body)), math.degrees(pose.body), control_mode


def _read_gap(ctl: Any) -> tuple[float, float]:
    """(body yaw °, target − body °) from one REST read: the field pulse's watcher grades."""
    import maxim.hardware.reachy.motion_path as mp

    data = ctl._get_state_full()  # noqa: SLF001 - the controller's own read
    pose, _ = mp.parse_full_state(data)
    return math.degrees(pose.body), math.degrees(mp.parse_target_body(data) - pose.body)


def _sample(ctl: Any, phase: str, t0: float) -> float | None:
    """One printed gap sample (°), or None when the read missed (a slow or failed read does not end the run)."""
    from maxim.exceptions import MotionRefused

    try:
        body, gap = _read_gap(ctl)
    except MotionRefused as e:
        print(f"  {time.monotonic() - t0:6.1f}s  {phase:<8}  read missed: {e.reason}")
        return None
    print(f"  {time.monotonic() - t0:6.1f}s  {phase:<8}  body={body:+6.1f}°  gap={gap:+5.1f}°")
    return gap


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

    def look(self, phase: str, u: int) -> float:
        """One tracking look-at at column *u*, paced to the cadence; returns the relative yaw reached (°)."""
        from maxim.hardware.controller import PixelTarget, as_outcome

        started = time.monotonic()
        outcome = as_outcome(
            self.ctl.look_at_pixel(
                PixelTarget(u=u, v=self.height // 2, duration=TRACK_PERIOD_S, duration_policy="at_least")
            )
        )
        self.outcomes["ok" if outcome.ok else (outcome.kind or "refused")] += 1
        rel, body, _ = _read(self.ctl)
        self.rel_max, self.rel_min = max(self.rel_max, rel), min(self.rel_min, rel)
        self.body_max = max(self.body_max, abs(body))
        verdict = "ok" if outcome.ok else f"{outcome.kind}: {outcome.reason}"
        print(
            f"  {time.monotonic() - self.t0:6.1f}s  {phase:<11}  u={u:<4}  rel={rel:+6.1f}°  body={body:+5.1f}°  {verdict}"
        )
        time.sleep(max(0.0, TRACK_PERIOD_S - (time.monotonic() - started)))
        return rel

    def walk(self, phase: str, u: int, sign: int) -> None:
        """Look at column *u* until the yaw (in direction *sign*) stops rising, or the cap."""
        best, flat, deadline = -math.inf, 0, time.monotonic() + WALK_CAP_S
        while flat < PLATEAU_STEPS and time.monotonic() < deadline:
            rel = sign * self.look(phase, u)
            flat = 0 if rel > best + 0.5 else flat + 1
            best = max(best, rel)

    def hold(self, phase: str, u: int) -> None:
        end = time.monotonic() + self.hold_s
        while time.monotonic() < end:
            self.look(phase, u)


def _ready(ctl: Any, resolution: tuple[int, int] | None) -> float | None:
    """The start body yaw (°) when torque is on and the body is near 0, else None (printed)."""
    rel, body, control_mode = _read(ctl)
    print(f"[start] torque={control_mode}  body={body:+.1f}°  relative head yaw={rel:+.1f}°  image={resolution}")
    if control_mode != "enabled":
        print("[FAIL] motor torque is not on: wake the robot first (this script never wakes it)")
        return None
    if abs(body) > BODY_START_DEG:
        print(f"[FAIL] the body is at {body:+.1f}°, not within {BODY_START_DEG:.0f}° of 0: re-centre it first")
        return None
    return body


def _turn_and_sample(ctl: Any, phase: str, to_deg: float, t0: float) -> tuple[bool, list[float]]:
    """Turn the body to *to_deg* through the controller while sampling the gap; (move ok, gaps seen during it)."""
    from maxim.hardware.controller import MotionTarget, as_outcome

    result: list[Any] = []
    mover = threading.Thread(
        target=lambda: result.append(
            as_outcome(ctl.goto_target(MotionTarget(body_yaw=math.radians(to_deg), duration=LAG_TURN_S)))
        ),
        daemon=True,
    )
    gaps: list[float] = []
    mover.start()
    try:
        while mover.is_alive():
            gap = _sample(ctl, phase, t0)
            if gap is not None:
                gaps.append(gap)
            time.sleep(LAG_SAMPLE_S)
    finally:
        mover.join()
    outcome = result[0] if result else None
    if outcome is None or not outcome.ok:
        print(f"  {phase}: {outcome.kind if outcome else 'error'}: {outcome.reason if outcome else 'no outcome'}")
        return False, gaps
    return True, gaps


def _settled_gap(ctl: Any, phase: str, t0: float) -> float | None:
    """The largest |gap| over LAG_SETTLE_S after a turn, or None when every read missed."""
    seen: list[float] = []
    end = time.monotonic() + LAG_SETTLE_S
    while time.monotonic() < end:
        gap = _sample(ctl, phase, t0)
        if gap is not None:
            seen.append(abs(gap))
        time.sleep(LAG_SAMPLE_S)
    return max(seen) if seen else None


def _lag_check(ctl: Any) -> int:
    """The watcher's control: a slow body turn and back, with this script's own gap trace (plan §13)."""
    if _ready(ctl, None) is None:
        return 2
    print(f"[lag-check] body to {LAG_TURN_DEG:+.0f}° over {LAG_TURN_S:.0f} s and back; gap = target − present body")
    t0 = time.monotonic()
    gaps: list[float] = []
    settled = 0.0
    try:
        for phase, to_deg in (("turn", LAG_TURN_DEG), ("return", 0.0)):
            ok, seen = _turn_and_sample(ctl, phase, to_deg, t0)
            gaps += seen
            if not ok:
                return 2
            after = _settled_gap(ctl, "settle", t0)
            if after is None:
                print(f"\n  NO READS after the {phase}: the gap's return could not be checked")
                return 2
            settled = max(settled, after)
            if settled > SETTLED_GAP_DEG:
                print(
                    f"\n  GAP STAYED: {settled:.1f}° after the {phase}; the target and the present body did not converge."
                )
                print(
                    f"  The body is left at {to_deg:+.0f}°: the controller refuses any move while the gap is over 2° (#1563)."
                )
                return 1
    except KeyboardInterrupt:
        ctl.request_abort()
        print("\n[stopped] Ctrl+C: motion stopped at the current chunk")
        return 2
    largest = max((abs(g) for g in gaps), default=0.0)
    print("\n[summary]")
    print(f"  largest |gap| during the turns: {largest:.1f}° ({len(gaps)} samples)")
    print(f"  largest |gap| after them:       {settled:.1f}°  (want ≤ {SETTLED_GAP_DEG:.0f}°)")
    print("  Compare with the watcher's largest gap during the turns: within ~1° and non-zero = the control PASSES;")
    print("  the watcher at ~0 while this reads a gap = a blind watcher, the control FAILS.")
    if largest < 0.1:
        print("  NO GAP SEEN: this reader saw no lag either, so the watcher cannot be checked against it")
        return 1
    return 0


def _run(ctl: Any, resolution: tuple[int, int] | None, hold_s: float = HOLD_S) -> int:
    from maxim.hardware.controller import MotionTarget, as_outcome

    if resolution is None:
        print("[FAIL] no camera on this connection: look_at_pixel needs the camera's intrinsics. Is another app")
        print("       holding media, or was --media-backend no_media passed? (the lag check runs without one)")
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
        print(f"  back to centre: {'ok' if centre.ok else f'{centre.kind}: {centre.reason}'}")
    except KeyboardInterrupt:
        ctl.request_abort()
        print("\n[stopped] Ctrl+C: motion stopped at the current chunk; the head stays where it is")
    _, body_end, _ = _read(ctl)
    sweep.body_max = max(sweep.body_max, abs(body_end))

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
    print(f"  REACHED: ≥ {REACHED_DEG:.0f}° on both sides with the body held; pulse's watcher grades the target gap")
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
    ap.add_argument("--lag-check", action="store_true", help="the watcher's control: a slow body turn and back")
    ap.add_argument("--hold-s", type=float, default=HOLD_S, help="seconds of edge look-ats at each clamp")
    ap.add_argument(
        "--media-backend", default="default", help="the SDK media backend (no_media works for --lag-check only)"
    )
    args = ap.parse_args()

    def _go(ctl: Any, resolution: tuple[int, int] | None) -> int:
        return _lag_check(ctl) if args.lag_check else _run(ctl, resolution, args.hold_s)

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
