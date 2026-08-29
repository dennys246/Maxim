"""Shared glue for the Phase 1 live orient scripts (Steps 2-3).

docs/plans/reachy_orient_live.md. Reuses the Step-1 hardware-verified
primitives from live_1_smoke.py (REST DoA read, host resolution, preflight)
and adds the pieces Steps 2-3 share:

  - JSONL trial logging (the agent reads this to iterate device-in-loop),
  - speech-gated median azimuth sampling (never fabricate a direction),
  - WS-era connect + enable_motors + wake_up (torque gate first),
  - body-yaw motion (the COARSE orient axis — head yaw clamps ~±15-18°,
    azimuth spans ±90°, so the discrete orient step drives body_yaw;
    goto_target(body_yaw=...) alone leaves the head untouched),
  - a dry-run world so each script's loop logic is verifiable OFFLINE
    before burning robot time (verify-each-layer discipline).

Coordinate/sign convention under calibration (Step 2's whole purpose):
the body YAML's sim convention is turn_left = +yaw -> azimuth grows
toward +1 (source appears more to the right). DEFAULT here matches that:
``az_increases_with_positive_yaw = True``; ``--flip-sign`` toggles it.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from collections.abc import Callable

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import live_1_smoke as smoke  # noqa: E402  (Step-1 verified primitives)

# scripts/_provenance.py, imported by path so it comes from the same tree as
# this file (the harness-provenance discipline; stdlib-only, never imports maxim).
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(_REPO_ROOT, "scripts"))
import _provenance  # noqa: E402

doa_to_azimuth = smoke.doa_to_azimuth
resolve_host = smoke.resolve_host
preflight = smoke.preflight

# A reader returns (doa_radians, is_speech_detected) or None (no reading).
Reader = Callable[[], "tuple[float, bool] | None"]


class JsonlLog:
    """Append-only JSONL event log (one dict per line, ts stamped).

    This is the in-process family's ONE record writer, so the gated-record
    refusal lives here rather than in every harness (roadmap 1.1.x item 16.7;
    docs/lessons/experiment-prereg-precedes-data.md): opening a path under
    ``docs/experiments/data/`` from a dirty ``src``/``scripts`` tree exits 3
    unless the harness passed ``allow_dirty=True`` (its ``--allow-dirty``),
    and an allowed dirty write stamps ``allow_dirty: true`` into EVERY record
    so the write-up cannot silently omit it. Non-gated paths (``/tmp`` logs)
    are never refused. Exp 53/53b stamped the dirty flag and kept going —
    stamping is detection, refusing is enforcement.
    """

    def __init__(self, path: str, *, allow_dirty: bool = False, mode: str = "a") -> None:
        if mode not in ("a", "w"):
            raise ValueError(f"mode must be 'a' (append, default) or 'w' (truncate), got {mode!r}")
        self.path = path
        gate = _provenance.preflight_gated_record_or_exit(_REPO_ROOT, path, allow_dirty=allow_dirty)
        self.gated = gate["gated"]
        self._stamp = {"allow_dirty": True} if gate["allow_dirty"] else {}
        self._f = open(path, mode, encoding="utf-8")  # noqa: SIM115 - long-lived handle

    def write(self, event: str, **fields: object) -> None:
        rec = {"ts": round(time.time(), 3), "event": event, **self._stamp, **fields}
        self._f.write(json.dumps(rec) + "\n")
        self._f.flush()

    def close(self) -> None:
        try:
            self._f.close()
        except Exception:  # noqa: BLE001
            pass


def make_rest_reader(host: str) -> Reader:
    """DoA reader over the daemon's REST endpoint (the network path, Step-1 verified).

    Delegates to the LIBRARY reader (`maxim.embodiment.audio_localization`), which
    is the canonical off-robot DoA path and the one Landing 1 wires into the
    runtime — so the scripts and the runtime read DoA through the exact same code.
    (`live_1_smoke.py` keeps its own inline copy on purpose: it must run standalone
    onboard with only the Reachy SDK, no `maxim` install.)
    """
    from maxim.embodiment.audio_localization import make_reachy_rest_doa_reader

    return make_reachy_rest_doa_reader(host)


# Promoted into the library (live_audio_orient_wiring.md Stage 1d) so the
# runtime's Stage-2 poll thread and these scripts share the exact same
# speech-gate + median-of-k code. Re-exported here so script callers keep
# their import path.
from maxim.embodiment.audio_localization import gated_azimuth  # noqa: E402,F401


def connect_and_wake(host: str):
    """WS-era connect + torque + wake. Returns (mini, create_head_pose).

    Era-gated: these scripts REQUIRE SDK >= 1.5 (WS transport). Torque
    first — wake_up() no longer enables it (daemon boots
    --no-wake-up-on-start; commands to limp motors are silently ignored).

    DELIBERATE BYPASS of src/maxim/hardware/reachy/controller.py
    (ReachyMiniController) and AzimuthDoASource: these runbook scripts stay
    on the Step-1-verified raw primitives so device-in-loop iteration has
    zero moving parts between the operator and the SDK. The production
    integration (controller + PerceptSource wiring) is the 1.1
    --embodiment-hardware work; the porting doc's "do not duplicate" table
    addresses NEW consumers, not this bring-up harness.
    """
    from importlib.metadata import version as _pkg_version

    try:
        sdk_ver = _pkg_version("reachy-mini")
        vt = tuple(int(x) for x in sdk_ver.split(".")[:2])
    except Exception:  # noqa: BLE001 - unknown/dev install: assume current era, but SAY so
        print("[warn] reachy-mini SDK version unreadable — assuming WS era (>=1.5). A zenoh-era")
        print("       (<=1.4) install would fail to connect; check `pip show reachy-mini`.")
        sdk_ver, vt = "?", (99, 0)
    if vt < (1, 5):
        raise RuntimeError(
            f"reachy-mini {sdk_ver} is the legacy zenoh era; Steps 2-3 require the WS-era "
            f"SDK (>= 1.5). Use ~/Envs/maxim-env (1.8.3) — the repo .venv is the stale one."
        )
    from reachy_mini import ReachyMini
    from reachy_mini.utils import create_head_pose

    mini = ReachyMini(host=host, port=8000, connection_mode="network", timeout=10.0, media_backend="no_media")
    print(f"[ok] connected (reachy-mini {sdk_ver}, ws://{host}:8000/ws/sdk)")
    mini.enable_motors()
    print("[motors] enable_motors() sent (torque on)")
    time.sleep(0.5)
    mini.wake_up()
    time.sleep(1.0)
    return mini, create_head_pose


class LiveRig:
    """Real-hardware rig: REST DoA + body-yaw motion, tracked commanded yaw."""

    def __init__(self, host: str) -> None:
        self.host = host
        self.mini, self._head_pose = connect_and_wake(host)
        self.reader: Reader = make_rest_reader(host)
        self.body_yaw = 0.0
        # The daemon may modulate a commanded body_yaw when automatic body yaw is
        # on. The orient loop owns the yaw axis outright — turn it off so what we
        # command is what happens.
        try:
            self.mini.set_automatic_body_yaw(False)
            print("[motion] automatic_body_yaw disabled (the orient loop owns the yaw axis)")
        except Exception as e:  # noqa: BLE001 - older daemons may not expose it
            print(f"[motion] set_automatic_body_yaw unavailable ({e}) — proceeding")

    def _goto(self, **kwargs) -> None:
        """goto_target with recovery. TWO distinct transient failures seen live:

        - ``TimeoutError`` — a dropped task-completion ack (killed s2 at trial
          36). The command itself probably landed; our targets are ABSOLUTE, so
          re-issuing is idempotent either way.
        - ``ConnectionError`` — the WS session itself died (killed mag1 at trial
          27, ~300 motion commands in). Re-issuing on a dead socket cannot work:
          rebuild the session first. The REST DoA reader is independent and
          keeps working across this.

        A SECOND failure of either kind is a real fault (daemon down, motors
        wedged or thermally throttled) and propagates — the loop's `finally`
        still saves the NAc, and checkpoints every 5 trials bound the loss.
        """
        for attempt in range(2):
            try:
                self.mini.goto_target(**kwargs)
                return
            except TimeoutError:
                if attempt:
                    raise
                print("      (goto_target ack timed out — re-issuing once)")
                time.sleep(0.5)
            except ConnectionError:
                if attempt:
                    raise
                print("      (WS connection lost — reconnecting, then re-issuing once)")
                self._reconnect()

    def _reconnect(self) -> None:
        """Rebuild the WS session — LIGHTWEIGHT: connect + torque, NO wake_up.

        The first version of this called connect_and_wake(), whose wake_up()
        issues its own goto_target — so a reconnect onto a still-sick daemon
        threw from INSIDE the recovery path and killed the run (mag2, trial 53).
        Recovery must not itself command motion. If the daemon's backend is
        down (``/api/daemon/status`` -> ``backend_status.ready: false``) no
        amount of reconnecting helps: it needs
        ``sudo systemctl restart reachy-mini-daemon`` on the robot.
        """
        from reachy_mini import ReachyMini

        time.sleep(2.0)
        self.mini = ReachyMini(
            host=self.host, port=8000, connection_mode="network", timeout=10.0, media_backend="no_media"
        )
        try:
            self.mini.enable_motors()
            self.mini.set_automatic_body_yaw(False)
        except Exception as e:  # noqa: BLE001 - torque/config best-effort on a shaky daemon
            print(f"      (post-reconnect setup partial: {e})")
        print("      (reconnected)")

    # Walk increment. RATIONALE RETRACTED (2026-07-16): introduced for "DoA lock
    # safety" per an Exp 45 finding that the chip loses lock on large jumps — that
    # finding was an artifact of the head=None counter-rotation bug (the mics
    # weren't turning at all). Post-fix the DoA is fast and stable (0.562 az/rad,
    # 0.23s convergence) and probably needs no walking. Kept because it is harmless
    # and cheap; drop it only after a clean post-headfix re-sweep says so.
    _TRACK_STEP = 0.3

    def recenter(self, duration: float = 1.0) -> None:
        self.goto_body_yaw(0.0, duration=duration)  # walked, head-aligned
        time.sleep(duration + 0.3)

    def goto_body_yaw(self, yaw: float, duration: float = 0.6) -> None:
        """Absolute body-yaw target (rad), head RIDING ALONG, walked in increments.

        THE BUG THIS FIXES (2026-07-16, found via Pollen's own AGENTS.md after
        six wrong hypotheses of mine): the head 4x4 pose is in the **world
        frame**, and it sits ABOVE body_yaw in the kinematic chain. Passing
        ``head=None`` does NOT mean "leave the head alone" — the daemon
        re-solves IK against the RETAINED world-frame head target
        (``backend/abstract.py::update_target_head_joints_from_ik``:
        ``if pose is None: pose = self.target_head_pose``), so the Stewart
        platform COUNTER-ROTATES to hold the head's absolute orientation and the
        body pivots underneath it. **The microphone array lives in the head.**
        Every DoA reading taken while commanding body_yaw alone was therefore of
        an array that barely turned (measured: mics rotated 0.32 rad for a 0.9
        rad body command) — which is the whole Exp 45b gain mystery.

        Pollen's prescription, verbatim: "to make the head follow the body, ship
        a `head` matrix in the same call with the body delta added to the head
        yaw." Commanding head world-yaw == body_yaw keeps the platform's
        head-vs-body angle at ~0, so the head is rigid to the body and the mics
        rotate exactly as commanded. (Head yaw reads [-180, +180] precisely
        because body_yaw supplies most of it; the platform's own ~±15-18° is not
        the limit here.)

        Still walked in <=_TRACK_STEP increments — cheap, and the tracking-
        estimator caution stands until re-measured on a correctly-rotating array.
        """
        target = float(yaw)
        while abs(target - self.body_yaw) > self._TRACK_STEP + 1e-9:
            nxt = self.body_yaw + self._TRACK_STEP * (1.0 if target > self.body_yaw else -1.0)
            self._goto_aligned(nxt, duration=min(duration, 0.4))
            time.sleep(0.5)
        self._goto_aligned(target, duration=duration)

    def _goto_aligned(self, yaw: float, duration: float) -> None:
        """Command body_yaw with the head yawed to match (world frame) — mics ride along."""
        self._goto(
            head=self._head_pose(yaw=math.degrees(yaw), degrees=True),
            body_yaw=float(yaw),
            duration=duration,
        )
        self.body_yaw = float(yaw)


class DryRig:
    """Offline stand-in: a speech source at a fixed world bearing.

    World model: body yaw psi (rad, CCW-positive = left); source at world
    bearing theta_src (same sense). Head-relative azimuth follows the
    DEFAULT sign convention az = (psi - theta_src)/(pi/2)  (turning left
    makes the source appear more to the right, az -> +1), unless
    ``world_flipped`` — which simulates the OPPOSITE hardware convention so
    the Step-2 flip-detection path is testable offline.
    """

    def __init__(
        self,
        theta_src: float = -0.7,
        *,
        world_flipped: bool = False,
        seed: int = 0,
        jump_prob: float = 0.04,
        gain: float = 2.0 / math.pi,
    ) -> None:
        import random

        self._rng = random.Random(seed)
        self.theta_src = theta_src
        self.body_yaw = 0.0
        self.world_flipped = world_flipped
        self.jump_prob = jump_prob  # 0.0 = stationary source (sweep/characterization runs)
        # az per radian of body yaw. Geometric default 2/pi=0.637; the REAL robot
        # measured 0.58 (sweep) then 0.39 (next session) — pass the measured value
        # to make the sim faithful to the hardware you are about to run on.
        self.gain = gain
        self.reader: Reader = self._read

    def _read(self) -> tuple[float, bool] | None:
        # The dry "operator" occasionally moves the source (as the live
        # protocol asks) so centered holds don't stall the loop forever.
        if self._rng.random() < self.jump_prob:
            self.theta_src = self._rng.uniform(-1.2, 1.2)
        rel = self.body_yaw - self.theta_src
        if self.world_flipped:
            rel = -rel
        az = max(-1.0, min(1.0, rel * self.gain))
        az = max(-1.0, min(1.0, az + self._rng.gauss(0.0, 0.02)))
        doa = az * math.pi / 2.0 + math.pi / 2.0  # invert doa_to_azimuth
        return (doa, self._rng.random() >= 0.1)  # occasional no-speech tick

    def recenter(self, duration: float = 1.0) -> None:  # noqa: ARG002 - parity with LiveRig
        self.body_yaw = 0.0

    def goto_body_yaw(self, yaw: float, duration: float = 0.6) -> None:  # noqa: ARG002
        self.body_yaw = float(yaw)


def _meta_path(nac_path: str) -> str:
    base = os.path.expanduser(nac_path)
    return (base[:-5] if base.endswith(".json") else base) + ".meta.json"


def save_policy_meta(nac_path: str, **meta: object) -> None:
    """Persist the STATE-SPACE definition next to the policy that learned in it.

    A cluster_reward_bias table is keyed on bin NAMES ("near_left"), but a bin
    name means nothing without the boundary that produced it. A consumer that
    bins at 0.5 while the policy learned at 0.33 looks up the wrong bin for every
    |az| in between — silently, no error. That is the third instance of this
    exact failure class in one day (head frame; learner-vs-metric az_bin; this).
    A flag cannot fix it: "remember how this policy was trained" is the
    assumption that keeps breaking. So the boundary travels WITH the policy.

    (This is also the local instance of the cross-robot bundle caveat in
    docs/embodiment/porting_orient_loop.md — substrate bundles need the same
    thing in their manifest before they can safely cross robots.)
    """
    with open(_meta_path(nac_path), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)


def load_policy_meta(nac_path: str) -> dict | None:
    """Read a policy's state-space definition, or None if it predates the sidecar."""
    p = _meta_path(nac_path)
    if not os.path.exists(p):
        return None
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def az_bin(az: float, band: float, boundary: float = 0.5) -> str:
    """State discretization (5 bins). ``boundary`` splits near from far.

    The default 0.5 is the legacy Exp 45/45b value (arbitrary). Exp 45c passes
    the DERIVED decision boundary — see ``decision_boundary`` — so that a bin
    never straddles the point where the correct action magnitude changes.
    """
    if abs(az) <= band:
        return "center"
    side = "left" if az < 0 else "right"
    return f"{'far' if abs(az) > boundary else 'near'}_{side}"


def decision_boundary(action_deltas: dict[str, float], gain: float) -> float:
    """|az| at which the BIG step overtakes the NORMAL step. Derived, not guessed.

    A correct-direction step of shift ``S = |delta| * gain`` takes ``|az|`` to
    ``|az - S|``, so its relief is ``az - |az - S|``. Step A therefore beats step
    B exactly when ``|az - S_A| < |az - S_B|`` — i.e. when az is NEARER to A's
    shift. The boundary is the MIDPOINT of the two shifts:

        boundary = gain * (|delta_big| + |delta_normal|) / 2

    NOT ``gain * |delta_big| / 2`` (an earlier error of mine): that is merely
    where big's relief crosses zero, which decides nothing on its own.

    Generalizes: with N distinct magnitudes there are N-1 such boundaries, and
    the optimal policy is nearest-neighbour quantization of |az| onto the
    available shifts. Reachy (gain 0.546, deltas 0.3/0.9): boundary = 0.328.
    """
    mags = sorted({abs(d) for d in action_deltas.values()})
    if len(mags) < 2:
        return 0.5  # single magnitude: nothing to decide
    return (mags[0] + mags[1]) * gain / 2.0


def placement_ranges(band: float, boundary: float, *, az_max: float = 0.80, margin: float = 0.06) -> dict:
    """Per-bin placement targets that do NOT straddle the decision boundary.

    az_max 0.80: the post-headfix sweep is monotonic to |az| ~0.87 (R^2=0.998),
    so the old conservative 0.65 endfire cap — which was chasing an artifact of
    the head-frame bug — can widen.
    """
    return {"near": (band + margin, boundary - margin), "far": (boundary + margin, az_max)}


def population_readout(bias_fn, state, action_deltas, direction_sign):
    """S4 population-vector MAGNITUDE readout (orient_magnitude_learning.md S4).

    The starvation Exp 45d measured is a *magnitude* readout problem: the tabular
    biases are hard-argmax'd over a hard bin, so a far bin whose big-turn cell was
    never positively sampled mis-picks the small turn even though its MIRROR bin
    learned "far -> big" decisively. This is the superior-colliculus population
    vector — discrete substrate (unchanged; still merges + transfers), continuous
    output — reading out a **magnitude** that pools the magnitude-class evidence
    across the SAME-ECCENTRICITY bins of BOTH sides. So far_left borrows
    far_right's learned "big" without ever sampling far_left x big itself. It does
    NOT hard-code "far -> big": the substrate still LEARNED that at far_right from
    relief; the readout only lets it generalise across the amplitude-map
    neighbourhood, the way a real population code does. Only positive biases vote
    (ReLU) — a punished action does not pull the average.

    **DIRECTION IS NOT DECIDED HERE.** ``direction_sign`` is passed in from the
    argmax action. Hardware (seed-3, near_right) showed why: a deterministic
    direction vote traps a *cold* bin (0-vs-0 tie -> defaults one way -> keeps
    turning wrong -> punished -> never escapes), whereas argmax's own None ->
    random fallback escapes cold bins. Direction learns fine on its own; only
    magnitude needed the sharing — so the two concerns are split, and this owns
    magnitude only. The caller uses argmax for direction (and its cold-bin random
    fallback) and calls this only once a direction is confidently chosen.

    Args:
        bias_fn: ``(bin_name, action_name) -> float`` — the learned
            ``cluster_reward_bias`` (a closure over the live NAc).
        state: the active bin (``near_left`` / ``far_right`` / ...).
        action_deltas: ``{action_name: yaml_delta_rad}`` — sign is direction
            (+ left / - right), ``abs`` is magnitude.
        direction_sign: +1 (left) / -1 (right), from the argmax action's sign.

    Returns:
        ``(continuous_delta_rad, credit_action_name)`` in the YAML sign convention
        (before ``step_scale`` / ``sign_mult``). ``credit_action_name`` is the
        chosen-direction action with the nearest magnitude class, so the greedy
        trial still credits the tabular table and progressively fills the starved
        cell. ``(None, None)`` for the center bin (caller keeps its argmax action).
    """
    if "_" not in state:  # center — no directional turn to make
        return None, None
    ecc = state.split("_", 1)[0]

    mags = sorted({abs(d) for d in action_deltas.values()})
    small_m, big_m = mags[0], mags[-1]

    def relu(x):
        return x if x > 0.0 else 0.0

    # Magnitude: pool the magnitude-class evidence across same-eccentricity bins
    # of BOTH sides (direction-agnostic amplitude map).
    ecc_bins = [f"{ecc}_left", f"{ecc}_right"]

    def class_score(target_mag):
        s = 0.0
        for b in ecc_bins:
            for a, d in action_deltas.items():
                if abs(abs(d) - target_mag) < 1e-9:
                    s += relu(bias_fn(b, a))
        return s

    small_score = class_score(small_m)
    big_score = class_score(big_m)
    total = small_score + big_score
    if total <= 1e-12:
        magnitude = small_m  # no positive evidence at this eccentricity -> gentlest step
    else:
        magnitude = (small_score * small_m + big_score * big_m) / total

    cont_delta = direction_sign * magnitude

    # Nearest discrete action (chosen direction, snapped magnitude) for crediting.
    nearest_m = small_m if abs(magnitude - small_m) <= abs(magnitude - big_m) else big_m
    dir_acts = {a: d for a, d in action_deltas.items() if (d > 0.0) == (direction_sign > 0.0)}
    credit_action = next((a for a, d in dir_acts.items() if abs(abs(d) - nearest_m) < 1e-9), None)
    return cont_delta, credit_action
