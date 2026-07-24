#!/usr/bin/env python
"""Phase 1, Step 3 — the live learning orient loop (substrate-primary, no LLM).

docs/plans/reachy_orient_live.md Step 3 + the substrate_native_orienting.md
rigor bar. Each speech-gated trial:

    az_before (live DoA) -> state = az_bin -> epsilon-greedy
    NAc.recommend_action over the body YAML's orient affordances
    (turn_left / turn_right) -> dispatch as a discrete BODY-YAW step
    (sign calibrated by Step 2) -> re-read DoA ->
    potential_diff = |az_before| - |az_after| (relief credit, Phase 0b)
    -> NAc.update_cluster_reward -> persist NAc for cross-session.

TWO TRIAL-GENERATION MODES:
  --perturb (RECOMMENDED): the source sits STILL in front of the robot; the
    APPARATUS generates each trial by rotating the base a commanded offset
    (known ground truth), then the learner — which only sees DoA — must
    turn back the right way. Balanced az-bin schedule by construction,
    intended-vs-measured azimuth logged per trial (continuous sign/gain
    validation), and a two-sided sign self-check at session start that
    aborts BEFORE learning if --flip-sign is wrong. Between trials the
    apparatus re-centers on the source (logged as `apparatus` events).
  (default, manual): the operator physically moves the source between
    trials — kept for A/B against the perturb protocol.

CONTAMINATION GUARD (--perturb): the commanded offset is EXPERIMENTER
APPARATUS ONLY — it sets up and labels trials. The learner's inputs are
unchanged: DoA-derived az_bin state in, potential_diff from re-measured
live DoA as credit. Nothing from the apparatus (ground truth, gain
estimate, servo direction) reaches NAc. Apparatus moves are logged as
`apparatus_*` events, learner moves as `trial` events.

LEARNED-vs-SERVO SEPARABILITY (the make-or-break rigor caveat): a hard-coded
servo would be 100% correct from trial 1. This run instead shows LEARNING:
  1. NAc starts empty (or --fresh) -> executed greedy choices start ~chance
     (2 actions => 0.5) and converge as cluster biases accumulate.
  2. Every --probe-every trials, a FROZEN-POLICY PROBE (pure NAc lookup, no
     motion, no exploration) logs the argmax action per off-center bin +
     the raw cluster_reward_bias table -> an epsilon-free learning curve.
  3. NAc persists to --nac-path; a SECOND session starts with the loaded
     biases -> its trial-0 probe is already correct = cross-session transfer.
The JSONL (--log) carries all three, pre-registered-metric style.

Interpretation guards:
  - Probe "correct" labels are the YAML action signs (with a correctly
    calibrated --flip-sign each action's az-effect IS its YAML sign; the
    calibration multiplier and world convention cancel). A WRONG flag makes
    probe correctness trend to 0, not 0.5 — in --perturb mode the start-up
    sign self-check catches this before any learning happens.
  - Credit only flows on trials with a valid speech-gated az_after (never
    fabricate). The source must be sustained speech, FRONT hemisphere
    (linear array = front/back ambiguity; perturbations are capped so the
    source stays in front).

OPERATOR PROTOCOL (--perturb):
    ~/Envs/maxim-env/bin/python scripts/orient_backbone/live_3_learn.py \
        --host <robot-ip> --perturb --session s1 --fresh
  Place a sustained speech source (phone playing a podcast) ~1-2 m away,
  roughly in front of the base's NEUTRAL heading, and don't touch it.
  Default 40 trials, hands-off. Second session (cross-session arm): same
  command minus --fresh, --session s2.
  Offline logic check: --dry-run (add --dry-flip for a flipped world).

SUCCESS: probe correctness rises from 0.00 (an empty NAc probes None per
bin — recommend_action admits only positive-scored tools) toward 1.00;
session 2's trial-0 probe starts correct. STOP-IF: the sign self-check
aborts (wrong --flip-sign), or probe correctness PINS AT 0.00 after 30+
trials (credit never reaching NAc, or an inverted flag forced past the
self-check), or hovers ~0.5 (credit flows but is uninformative — check
potential_diff vs DoA noise in the JSONL and the perturb records' gain).
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time

from live_common import (
    DryRig,
    JsonlLog,
    LiveRig,
    az_bin,
    decision_boundary,
    gated_azimuth,
    placement_ranges,
    population_readout,
    preflight,
    resolve_host,
    save_policy_meta,
)

if os.environ.pop("MAXIM_NAC_REWARD_BIAS_DISABLED", None) is not None:
    print("[env] MAXIM_NAC_REWARD_BIAS_DISABLED was set — cleared for this run (ablation flag does not apply here)")
from maxim.decisions.nac import NAc, NACConfig  # noqa: E402
from maxim.embodiment.component_registry import ComponentRegistry  # noqa: E402

AGENT_ID = "reachy"
ARGMAX = -1.0e9  # min_confidence sentinel: always return argmax (Phase 0b)
OFF_CENTER_BINS = ("far_left", "near_left", "near_right", "far_right")
GEOMETRIC_GAIN = 2.0 / math.pi  # d(normalized az)/d(yaw rad) if DoA is ideal
# Measured az-per-radian gain, POST-HEADFIX (2026-07-16). Four independent
# measurements agree within 0.03: sweep central 0.546/0.548, full-range fit
# 0.571/0.578, settle traces 0.562, mag2 sign-check 0.58. The earlier 0.39 was
# contaminated by the head=None counter-rotation bug (the mics barely turned).
# Still not an authority — rooms/mounts/sources differ, so --az-gain overrides
# and --perturb runs prefer the apparatus's own live EMA.
MEASURED_GAIN = 0.55
# Legacy placement targets (Exp 45/45b): near STRADDLES the derived decision
# boundary (0.328), which is exactly why 45b scored magnitude 0.75. --flip-bins
# replaces these with placement_ranges(band, decision_boundary(...)).
LEGACY_PLACEMENT_RANGES = {"near": (0.18, 0.42), "far": (0.55, 0.65)}


def sign(x: float) -> float:
    return 1.0 if x > 0 else -1.0


def load_orient_actions() -> tuple[dict[str, float], float]:
    """Affordance names -> head_yaw DELTAS (rad) from the real body YAML.

    Exp 45b (magnitude): the YAML's self_effect magnitudes ARE the orient
    step sizes (0.3 normal / 0.9 big) — the body declares what it can
    express; the script no longer imposes one --step. On hardware these
    drive BODY yaw (the orient axis — the head clamps ±15-18°);
    --step-scale rescales them for a different robot/room without editing
    shipped data.
    """
    ent = ComponentRegistry().instantiate("bodies/reachy_mini")
    orient = ent.modulators["orient"]
    actions = {an: float(sch.self_effect["head_yaw"]) for an, sch in orient.affordances.items()}
    band = ent.drive_specs["azimuth"].comfort_band
    return actions, band


def _mean_relief(abs_delta_rad: float, gain: float, lo: float, hi: float, n: int = 41) -> float:
    """Mean per-step |az| reduction for a correct-direction step of this size.

    Integrated over the bin's PLACEMENT range (what the policy actually meets).
    A step shifts |az| by ``abs_delta_rad * gain``; overshoot is the ``abs()`` —
    past center, further travel ADDS error back. That single term is the whole
    reason magnitude is (or isn't) learnable from relief alone.
    """
    shift = abs_delta_rad * gain
    return sum(m - abs(m - shift) for m in (lo + (hi - lo) * i / (n - 1) for i in range(n))) / n


def optimal_action_for_bin(
    bin_name: str,
    action_deltas: dict[str, float],
    *,
    gain: float,
    ranges: dict,
    effort_lambda: float = 0.0,
) -> str | None:
    """The utility-optimal action for a bin GIVEN THE MEASURED PHYSICS.

    utility = mean relief over the bin's placements − effort_lambda * |Δyaw|.
    Wrong-direction actions always earn negative relief, so they never win —
    the argmax picks direction, then magnitude by the relief/effort trade.

    Deriving this from the run's own gain (rather than hard-coding "near→small")
    is Exp 45b's metric correction: at gain 0.39 the big step does NOT overshoot
    across most of the near bin, so "near→big" is genuinely relief-optimal and a
    hard-coded expectation would score a correct policy as wrong.
    """
    side = -1.0 if "left" in bin_name else 1.0
    lo, hi = ranges["far" if bin_name.startswith("far") else "near"]
    best, best_u = None, -1e9
    for name, d in action_deltas.items():
        if sign(d) != -side:  # wrong direction — never optimal
            continue
        u = _mean_relief(abs(d), gain, lo, hi) - effort_lambda * abs(d)
        if u > best_u:
            best_u, best = u, name
    return best


def probe_policy(
    nac: NAc,
    names: list[str],
    action_deltas: dict[str, float],
    *,
    gain: float = MEASURED_GAIN,
    ranges: dict | None = None,
    effort_lambda: float = 0.0,
) -> dict:
    """Frozen-policy probe: argmax per off-center bin, no motion, no epsilon.

    TWO pre-registered metrics (Exp 45 + 45b):
      correctness            — does the argmax turn toward center? (direction)
      magnitude_appropriateness — is it the right SIZE? far bins should prefer
        the big step (it centers them: huge relief); near bins the normal step
        (big overshoots there -> negative relief). Bin-averaged, and the near
        bin spans the flip point — which is itself the S1 (Weber-bins)
        motivation, not a metric defect.

    correct := the argmax action drives azimuth toward 0 for that bin.
    With a CORRECTLY calibrated --flip-sign, each named action's az-effect
    is always its YAML sign (the calibration multiplier and the world's
    yaw->az convention cancel — that is what Step 2 calibrates), so the
    label is simply yaml_sign == -side and sign_mult does NOT appear here.
    A WRONG flag makes the world teach the reversed action per bin, so
    correctness trends to 0 — the docstring's diagnostic signature. Also
    snapshots the raw biases so convergence is visible before argmax flips.
    """
    ranges = ranges if ranges is not None else LEGACY_PLACEMENT_RANGES
    out: dict[str, dict] = {}
    n_correct = 0
    n_mag = 0
    for b in OFF_CENTER_BINS:
        rec = nac.recommend_action(
            agent_id=AGENT_ID,
            available_tools=names,
            current_drives=None,
            current_cluster_id=b,
            min_confidence=ARGMAX,
        )
        # rec is None on an empty bin AND on a bin holding only negative
        # biases (only the wrong action tried so far) — recommend_action
        # admits positive scores only. Both probe as incorrect here, which
        # undercounts learned avoidance slightly; acceptable for a
        # 2-action policy where convergence fills the positive side fast.
        chosen = rec["tool_name"] if rec else None
        side = -1.0 if "left" in b else 1.0  # az sign for this bin
        correct = bool(chosen) and sign(action_deltas[chosen]) == -side
        opt = optimal_action_for_bin(b, action_deltas, gain=gain, ranges=ranges, effort_lambda=effort_lambda)
        mag_ok = bool(chosen) and opt is not None and abs(action_deltas[chosen]) == abs(action_deltas[opt])
        n_correct += int(correct)
        n_mag += int(mag_ok)
        out[b] = {
            "argmax": chosen,
            "optimal": opt,
            "correct": correct,
            "magnitude_ok": mag_ok,
            "biases": {a: round(nac.cluster_reward_bias(AGENT_ID, b, f"tool:{a}"), 4) for a in names},
        }
    return {
        "bins": out,
        "correctness": n_correct / len(OFF_CENTER_BINS),
        "magnitude_appropriateness": n_mag / len(OFF_CENTER_BINS),
        "gain_used": round(gain, 3),
        "effort_lambda": effort_lambda,
    }


class Apparatus:
    """Experimenter rig for --perturb mode (NOT the learner).

    Knows the ground truth (commanded offsets, calibrated servo direction,
    yaw->az gain estimate). The learner never reads any of it — its state
    is DoA-derived az_bin and its credit is potential_diff on re-measured
    DoA. Everything here is logged under `apparatus_*` / `perturb` events
    so the record cleanly separates rig motion from learner motion.

    The gain estimate exists because Step 2 measured |az| changes 2-3x
    larger than the geometric prediction (0.637/rad) — chip smoothing or
    base gearing. It's EMA-updated from every commanded move with valid
    before/after readings and used only to size apparatus moves.
    """

    def __init__(
        self,
        rig,
        log,
        pace,
        poll_s: float,
        band: float,
        sign_mult: float,
        max_yaw: float,
        duration: float,
        settle: float,
    ) -> None:
        self.rig = rig
        self.log = log
        self.pace = pace
        self.poll_s = poll_s
        self.band = band
        self.sign_mult = sign_mult
        self.max_yaw = max_yaw
        self.duration = duration
        self.settle = settle
        # |d(az)/d(yaw)| prior = the sweep-measured TRACKED gain (0.58/rad,
        # baseline sweep 2026-07-16), not the geometric 0.64; EMA-updated live.
        self.gain = 0.58

    def read(self) -> float | None:
        return gated_azimuth(self.rig.reader, k=3, timeout_s=6.0, poll_s=self.poll_s)

    # Sweep finding (2026-07-16, /tmp/doa_sweep.jsonl label=baseline): the
    # XVF3800 DoA is a TRACKING estimator, not memoryless — it follows small
    # incremental pose changes cleanly (measured tracked gain ~0.58/rad,
    # near-geometric) but LOSES LOCK on large jumps (a single 1.4 rad jump
    # left it pinned near the stale estimate for an entire half-sweep).
    # Every apparatus move is therefore WALKED in <= _TRACK_STEP increments
    # with settle between, preserving the chip's track.
    _TRACK_STEP = 0.3

    def _move(self, delta_psi: float) -> float:
        """Clamped relative base move, walked in tracked increments."""
        target = max(-self.max_yaw, min(self.max_yaw, self.rig.body_yaw + delta_psi))
        total = target - self.rig.body_yaw
        if total == 0.0:
            return 0.0
        n_inc = max(1, math.ceil(abs(total) / self._TRACK_STEP))
        inc = total / n_inc
        for _ in range(n_inc):
            self.rig.goto_body_yaw(self.rig.body_yaw + inc, duration=self.duration)
            self.pace(self.duration + self.settle)
        return total

    def _observe_gain(self, delta_psi: float, az0: float | None, az1: float | None) -> None:
        if az0 is None or az1 is None or abs(delta_psi) < 0.15:
            return
        if max(abs(az0), abs(az1)) > 0.95:  # saturated reading — uninformative
            return
        g = abs((az1 - az0) / delta_psi)
        if 0.1 < g < 4.0:
            self.gain = max(0.3, min(2.5, 0.7 * self.gain + 0.3 * g))

    def center(self, max_steps: int = 12) -> float | None:
        """Servo the base back onto the source (trial reset). Final az or None."""
        for _ in range(max_steps):
            az = self.read()
            if az is None:
                return None
            if abs(az) <= self.band:
                return az
            step = -sign(az) * min(0.35, abs(az) / self.gain) * self.sign_mult
            actual = self._move(step)
            if actual == 0.0:
                # pinned at the yaw limit: hard-reset the base and retry
                self.log.write("apparatus_recenter_limit", body_yaw=round(self.rig.body_yaw, 3))
                self.rig.recenter()
                continue
            az1 = self.read()
            self._observe_gain(actual, az, az1)
            self.log.write(
                "apparatus_center_step",
                az_before=round(az, 3),
                delta_psi=round(actual, 3),
                az_after=None if az1 is None else round(az1, 3),
                gain=round(self.gain, 3),
            )
        az = self.read()
        return az if az is not None and abs(az) <= self.band else None

    def perturb(self, target_az: float) -> tuple[float | None, float]:
        """From centered, place the source at ~target_az (head-relative).

        Returns (measured azimuth after the offset, delta_psi commanded).
        The measured value — never the intent — is what the learner sees.
        """
        # Moving the base by +delta shifts az by sign_mult*gain*delta
        # (sign_mult = the Step-2-calibrated world convention).
        delta_psi = self.sign_mult * target_az / self.gain
        az0 = self.rig.body_yaw  # noqa: F841 - kept for debugging breadcrumbs
        actual = self._move(delta_psi)
        az1 = self.read()
        self._observe_gain(actual, 0.0, az1)  # perturb starts from centered (az~0)
        return az1, actual

    def sign_self_check(self) -> bool:
        """Two-sided perturbation check BEFORE learning: +0.45 then -0.45.

        Catches a wrong --flip-sign at t=0 (measured az sign must match the
        intended side on both probes) — upgrading the wrong-flag failure
        mode from 'diagnose after 30 trials' to 'abort at start'. Also
        seeds the gain estimate. Replaces the manual left-side Step 2 rerun.
        """
        for intent in (0.45, -0.45):
            az_c = self.center()
            if az_c is None:  # one retry — a transient gate drop is not a verdict
                az_c = self.center()
            if az_c is None:
                print("[FAIL] sign self-check: could not center on the source (speech gate?)")
                return False
            measured, delta_psi = self.perturb(intent)
            self.log.write(
                "apparatus_sign_check",
                intended_az=intent,
                measured_az=None if measured is None else round(measured, 3),
                delta_psi=round(delta_psi, 3),
                gain=round(self.gain, 3),
            )
            m = "?" if measured is None else f"{measured:+.2f}"
            print(f"[sign-check] intended az {intent:+.2f} -> measured {m}   (gain est {self.gain:.2f}/rad)")
            if measured is not None and abs(measured) > self.band and sign(measured) != sign(intent):
                # Fail FAST on the first decisive wrong-side landing: with an
                # inverted flag the centering servo also diverges, so waiting
                # for the second probe just thrashes the base at the yaw limit.
                flag = "WITHOUT --flip-sign" if self.sign_mult < 0 else "WITH --flip-sign"
                print(f"[FAIL] perturbation landed on the WRONG side ({intent:+.2f} -> {m}) — rerun {flag}.")
                return False
            if measured is None or abs(measured) <= self.band:
                print(f"[warn] sign-check probe inconclusive ({intent:+.2f} -> {m}) — proceeding; check JSONL.")
        self.center()
        return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=None, help="Reachy daemon IP (or $MAXIM_REACHY_HOST)")
    ap.add_argument(
        "--perturb", action="store_true", help="apparatus-generated trials (fixed source, commanded base offsets)"
    )
    ap.add_argument("--trials", type=int, default=40, help="valid credited trials this session")
    ap.add_argument(
        "--step-scale", type=float, default=1.0, help="rescale the YAML orient magnitudes (1.0 = as declared)"
    )
    ap.add_argument(
        "--effort-lambda",
        type=float,
        default=0.0,
        help="Exp 45c: effort cost per radian turned (credit = relief - lambda*|dyaw|). "
        "0.0 = Exp 45b pure-relief credit. Derived window at gain 0.39: 0.18 < lambda < 0.39.",
    )
    ap.add_argument(
        "--az-gain",
        type=float,
        default=None,
        help="az-per-radian gain the METRIC assumes (default: the apparatus's live EMA in "
        f"--perturb mode, else {MEASURED_GAIN}). Does not affect the robot — only what "
        "counts as the magnitude-optimal action.",
    )
    ap.add_argument("--epsilon", type=float, default=0.25, help="exploration rate (executed choices)")
    ap.add_argument(
        "--readout",
        choices=["argmax", "population"],
        default="argmax",
        help="S4 (orient_magnitude_learning.md): 'population' reads out a CONTINUOUS "
        "bias-weighted turn, magnitude pooled across same-eccentricity bins so a starved "
        "far cell borrows its mirror's learned 'big' (fixes the Exp 45d 0.75 residual). "
        "'argmax' (default) = the discrete tabular readout, byte-identical to before. "
        "Learning is unchanged in both; only the greedy readout differs.",
    )
    ap.add_argument("--gain", type=float, default=1.0, help="scale on potential_diff credit")
    ap.add_argument("--flip-sign", action="store_true", help="Step-2-calibrated sign flag")
    ap.add_argument("--probe-every", type=int, default=5, help="frozen-policy probe cadence (trials)")
    ap.add_argument("--nac-path", default=os.path.expanduser("~/.maxim/orient_live/nac_reachy.json"))
    ap.add_argument("--fresh", action="store_true", help="ignore any existing NAc file (start untrained)")
    ap.add_argument("--session", default="s1", help="session label for the JSONL (cross-session evidence)")
    ap.add_argument("--max-yaw", type=float, default=1.4, help="clamp cumulative body yaw (rad)")
    ap.add_argument("--duration", type=float, default=0.6)
    ap.add_argument("--settle", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0, help="epsilon-exploration RNG seed")
    ap.add_argument("--log", default="/tmp/orient_step3.jsonl")
    ap.add_argument("--dry-run", action="store_true", help="offline logic check (no robot)")
    ap.add_argument("--dry-flip", action="store_true", help="dry-run world uses the OPPOSITE sign convention")
    ap.add_argument(
        "--flip-bins",
        action="store_true",
        help="Exp 45c: put the near/far boundary at the DERIVED decision boundary "
        "(gain*(|d_big|+|d_normal|)/2) instead of the legacy arbitrary 0.5, so no bin "
        "straddles the point where the correct magnitude changes. Widens placements to "
        "the post-headfix reliable range (|az| <= 0.80).",
    )
    ap.add_argument(
        "--dry-gain",
        type=float,
        default=MEASURED_GAIN,
        help=f"dry-run world's az-per-radian gain (default {MEASURED_GAIN} = the Exp 45b hardware "
        "measurement, so sim mirrors the robot you are about to run on)",
    )
    args = ap.parse_args()

    rng = random.Random(args.seed)
    log = JsonlLog(args.log)
    sign_mult = -1.0 if args.flip_sign else 1.0
    # Dry-run: reads are synthetic and instant — skip real-time pacing.
    poll_s = 0.0 if args.dry_run else 0.15

    def pace(seconds: float) -> None:
        if not args.dry_run:
            time.sleep(seconds)

    action_deltas, band = load_orient_actions()
    names = list(action_deltas)
    print(f"[body] orient affordances (yaml rad): {action_deltas}   comfort_band={band}")

    def metric_gain() -> float:
        """Gain the METRIC assumes: CLI > apparatus live EMA > last measurement."""
        if args.az_gain is not None:
            return args.az_gain
        if app is not None:
            return app.gain
        return MEASURED_GAIN

    # Exp 45c: the state boundary is DERIVED from this robot's own measured gain and
    # its declared action magnitudes — not a constant. Frozen at start (a boundary
    # that moved mid-run would silently redefine the state space under the learner).
    boundary_gain = args.az_gain if args.az_gain is not None else MEASURED_GAIN
    if args.flip_bins:
        bin_boundary = decision_boundary(action_deltas, boundary_gain)
        ranges = placement_ranges(band, bin_boundary)
        print(
            f"[bins] --flip-bins: boundary DERIVED at |az|={bin_boundary:.3f} "
            f"(gain {boundary_gain} x (|{max(map(abs, action_deltas.values())):.1f}|+"
            f"|{min(map(abs, action_deltas.values())):.1f}|)/2)   placements={ranges}"
        )
    else:
        bin_boundary = 0.5
        ranges = LEGACY_PLACEMENT_RANGES
        print(
            f"[bins] legacy boundary |az|=0.5 (straddles the derived {decision_boundary(action_deltas, boundary_gain):.3f})"
        )

    def probe() -> dict:
        return probe_policy(
            nac, names, action_deltas, gain=metric_gain(), ranges=ranges, effort_lambda=args.effort_lambda
        )

    def save_policy() -> None:
        """NAc + the state-space definition that gives its bin names meaning."""
        nac.save(args.nac_path)
        save_policy_meta(
            args.nac_path,
            bin_boundary=round(bin_boundary, 4),
            band=band,
            gain=round(boundary_gain, 4),
            action_deltas=action_deltas,
            placements={k: list(v) for k, v in ranges.items()},
            agent_id=AGENT_ID,
            session=args.session,
            flip_bins=bool(args.flip_bins),
        )

    os.makedirs(os.path.dirname(args.nac_path) or ".", exist_ok=True)
    nac = NAc(NACConfig(persistence_path=args.nac_path))
    if args.fresh:
        print("[nac] --fresh: starting untrained")
    else:
        ok, err = nac.load_safe(args.nac_path)
        print(f"[nac] load_safe({args.nac_path}) -> ok={ok}" + (f" ({err})" if err else ""))

    if args.dry_run:
        # jump_prob=0 in --perturb: the apparatus moves the ROBOT and the source is
        # fixed (that is the whole point of the protocol). The 0.04 default models an
        # operator relocating the source between trials — correct for manual mode,
        # a pure noise source in perturb mode. It was silently degrading every sim
        # sweep today: ~10 reads/trial => ~1-in-3 trials had the source teleport
        # mid-trial, which corrupts potential_diff and hits narrow (low-relief) bins
        # hardest — exactly the bins Exp 45c narrows.
        rig = DryRig(
            theta_src=-0.7,
            world_flipped=args.dry_flip,
            seed=args.seed,
            gain=args.dry_gain,
            jump_prob=0.0 if args.perturb else 0.04,
        )
        print(
            f"[dry] source at world bearing {rig.theta_src:+.2f} rad, "
            f"world_flipped={rig.world_flipped}, gain={args.dry_gain}"
        )
    else:
        host, source = resolve_host(args.host)
        if host is None:
            print("[FAIL] no robot address: --host <ip> or export MAXIM_REACHY_HOST=<ip>")
            log.close()
            return 2
        print(f"[host] using {host} (source: {source})")
        preflight(host)
        rig = LiveRig(host)
        rig.recenter()

    log.write(
        "start",
        session=args.session,
        mode="perturb" if args.perturb else "manual",
        trials=args.trials,
        step_scale=args.step_scale,
        effort_lambda=args.effort_lambda,
        epsilon=args.epsilon,
        flip_sign=args.flip_sign,
        flip_bins=args.flip_bins,
        bin_boundary=round(bin_boundary, 4),
        fresh=args.fresh,
        nac_path=args.nac_path,
        dry_run=args.dry_run,
    )

    app: Apparatus | None = None
    if args.perturb:
        app = Apparatus(rig, log, pace, poll_s, band, sign_mult, args.max_yaw, args.duration, args.settle)
        print("[apparatus] sign self-check (two-sided, before any learning)...")
        if not app.sign_self_check():
            log.write("abort", reason="sign_self_check_failed")
            log.close()
            return 1
        log.write("apparatus_sign_check_ok", gain=round(app.gain, 3))

    # Write the state space BEFORE any trial: an interrupted run must never leave a
    # policy whose sidecar describes a different state space than it learned in.
    save_policy()

    # Trial-0 probe: the cross-session headline number (session 2 should start correct).
    p0 = probe()
    print(
        f"[probe 0] correctness={p0['correctness']:.2f} magnitude={p0['magnitude_appropriateness']:.2f}  "
        + str({b: v["argmax"] for b, v in p0["bins"].items()})
    )
    log.write("probe", trial=0, **p0)

    valid = 0
    no_reading_streak = 0
    schedule: list[str] = []
    greedy_correct_first10: list[int] = []
    greedy_correct_last10: list[int] = []
    # S4 metric: continuous residual |az_after| on GREEDY trials — the yardstick
    # discrete argmax-correctness can't see. Lower = better centering. Comparable
    # across --readout argmax vs population (both log it).
    greedy_residuals: list[float] = []
    aborted: str | None = None
    while valid < args.trials:
        try:
            # --- acquire az_before (trial setup differs per mode; learner step is shared) ---
            intended_bin: str | None = None
            if app is not None:
                if not schedule:
                    schedule = list(OFF_CENTER_BINS)
                    rng.shuffle(schedule)
                intended_bin = schedule.pop()
                if app.center() is None:
                    no_reading_streak += 1
                    if no_reading_streak >= 10:
                        print("[STOP] speech gate not passing for ~1 min — check the source, then rerun.")
                        break
                    log.write("no_reading_center")
                    continue
                side = -1.0 if "left" in intended_bin else 1.0
                # PLACEMENT_RANGES: far ceiling 0.65 (the baseline sweep showed reliable
                # tracked readings to ~|az| 0.69, with a bimodal endfire-degeneracy zone
                # beyond — s1's poisonous trials). Shared with the metric, which must
                # integrate expected relief over the SAME distribution the policy meets.
                lo, hi = ranges["far" if intended_bin.startswith("far") else "near"]
                target_az = side * rng.uniform(lo, hi)
                az_before, delta_psi = app.perturb(target_az)
                if az_before is None:
                    no_reading_streak += 1
                    if no_reading_streak >= 10:
                        print("[STOP] speech gate not passing for ~1 min — check the source, then rerun.")
                        break
                    log.write("no_reading_perturb", intended_bin=intended_bin)
                    continue
                no_reading_streak = 0
                measured_bin = az_bin(az_before, band, bin_boundary)
                log.write(
                    "perturb",
                    intended_bin=intended_bin,
                    target_az=round(target_az, 3),
                    delta_psi=round(delta_psi, 3),
                    az_measured=round(az_before, 3),
                    bin_measured=measured_bin,
                    gain=round(app.gain, 3),
                )
                if measured_bin == "center":
                    print(f"      perturb undershoot (intended {intended_bin}, az={az_before:+.2f}) — retrying")
                    schedule.append(intended_bin)  # keep the schedule balanced
                    continue
            else:
                az_before = gated_azimuth(rig.reader, k=3, timeout_s=6.0, poll_s=poll_s)
                if az_before is None:
                    no_reading_streak += 1
                    if no_reading_streak >= 10:
                        print("[STOP] speech gate not passing for ~1 min — check the source, then rerun.")
                        break
                    log.write("no_reading_before")
                    continue
                no_reading_streak = 0
                if az_bin(az_before, band, bin_boundary) == "center":
                    print(f"      az={az_before:+.2f} centered — holding (move the source to continue)")
                    log.write("centered", az=round(az_before, 3))
                    pace(0.8)
                    continue

            # --- the LEARNER's step (identical in both modes: DoA in, relief credit out) ---
            state = az_bin(az_before, band, bin_boundary)
            explore = rng.random() < args.epsilon
            if explore:
                # Exploration is always DISCRETE (both readouts) so the tabular
                # bias table — the thing that merges + transfers — is built
                # identically; only the GREEDY readout differs.
                action = rng.choice(names)
                delta = action_deltas[action] * args.step_scale * sign_mult
            elif args.readout == "population":
                # S4: continuous bias-weighted delta, magnitude pooled across the
                # same-eccentricity bins so a starved far cell borrows its mirror
                # (orient_magnitude_learning.md S4). ``action`` is the nearest
                # discrete action, so the greedy trial still credits the tabular
                # table and progressively fills the starved cell.
                cont_delta, action = population_readout(
                    lambda b, a: nac.cluster_reward_bias(AGENT_ID, b, f"tool:{a}"),
                    state,
                    action_deltas,
                )
                if action is None:  # center / no positive evidence — argmax fallback
                    rec = nac.recommend_action(
                        agent_id=AGENT_ID,
                        available_tools=names,
                        current_drives=None,
                        current_cluster_id=state,
                        min_confidence=ARGMAX,
                    )
                    action = rec["tool_name"] if rec else rng.choice(names)
                    delta = action_deltas[action] * args.step_scale * sign_mult
                else:
                    delta = cont_delta * args.step_scale * sign_mult
            else:
                rec = nac.recommend_action(
                    agent_id=AGENT_ID,
                    available_tools=names,
                    current_drives=None,
                    current_cluster_id=state,
                    min_confidence=ARGMAX,
                )
                action = rec["tool_name"] if rec else rng.choice(names)
                delta = action_deltas[action] * args.step_scale * sign_mult
            target_yaw = max(-args.max_yaw, min(args.max_yaw, rig.body_yaw + delta))
            if target_yaw == rig.body_yaw:
                print(f"      at yaw limit ({rig.body_yaw:+.2f}) — recentering (no credit)")
                log.write("yaw_limit", body_yaw=round(rig.body_yaw, 3))
                rig.recenter()
                continue

            rig.goto_body_yaw(target_yaw, duration=args.duration)
            pace(args.duration + args.settle)
            az_after = gated_azimuth(rig.reader, k=3, timeout_s=6.0, poll_s=poll_s)
            if az_after is None:
                print("      (no gated reading after turn — trial invalid, no credit)")
                log.write("no_reading_after", state=state, action=action)
                continue

            potential_diff = abs(az_before) - abs(az_after)
            # Exp 45c: effort cost. Pure relief cannot teach magnitude on this hardware —
            # overshoot needs |delta|*gain > 2*|az|, i.e. > 2.1 rad at the measured 0.39
            # gain, beyond the 1.4 rad yaw limit. Organisms pay for movement; so does this.
            credit = args.gain * potential_diff - args.effort_lambda * abs(delta)
            nac.update_cluster_reward(AGENT_ID, state, f"tool:{action}", credit)
            valid += 1
            turned_toward = potential_diff > 0.02
            if not explore:
                greedy_residuals.append(abs(az_after))
                bucket = greedy_correct_first10 if valid <= 10 else greedy_correct_last10
                if valid <= 10 or valid > args.trials - 10:
                    bucket.append(int(turned_toward))
            print(
                f"      [{valid:2d}/{args.trials}] {state:<11} {action:<11} ({'explore' if explore else 'greedy'})  "
                f"az {az_before:+.2f} -> {az_after:+.2f}  relief={potential_diff:+.3f}"
            )
            log.write(
                "trial",
                session=args.session,
                n=valid,
                state=state,
                intended_bin=intended_bin,
                action=action,
                explore=explore,
                az_before=round(az_before, 3),
                az_after=round(az_after, 3),
                potential_diff=round(potential_diff, 4),
                credit=round(credit, 4),
                body_yaw=round(rig.body_yaw, 3),
            )

            if valid % args.probe_every == 0:
                p = probe()
                print(
                    f"[probe {valid}] correctness={p['correctness']:.2f} "
                    f"magnitude={p['magnitude_appropriateness']:.2f}  "
                    + str({b: v["argmax"] for b, v in p["bins"].items()})
                )
                log.write("probe", trial=valid, **p)
            if valid % 5 == 0:
                # Checkpoint often — a transient SDK ack-timeout killed s2 at trial
                # 36 with the last save at 30; the file is ~1 KB, saving is free.
                # MUST be save_policy(), not nac.save(): a bare save leaves the sidecar
                # STALE, and Ctrl+C (the normal way to stop a hands-off run) would then
                # leave a 0.33-trained policy advertising boundary 0.5 — which the demo
                # trusts CONFIDENTLY, no warning. (An earlier patch of mine missed this
                # line because its text anchor silently didn't match after a reindent.)
                save_policy()
        except (ConnectionError, TimeoutError) as e:
            # The robot went away and recovery could not bring it back (mag2: the
            # daemon's backend_status.ready went false). Bank what we have and
            # report honestly rather than tracebacking over a real result.
            aborted = f"{type(e).__name__}: {e}"
            print(f"\n[STOP] lost the robot at trial {valid}: {aborted}")
            print("       Check: curl http://<host>:8000/api/daemon/status -> backend_status.ready")
            print("       If false: sudo systemctl restart reachy-mini-daemon on the robot.")
            log.write("abort", reason=aborted, trials_completed=valid)
            break

    if not args.dry_run and aborted is None:
        rig.recenter()
    save_policy()

    pf = probe()

    def _rate(bucket: list[int]) -> float | None:
        # None (not NaN) when empty — json.dumps(nan) emits a bare `NaN`
        # token that breaks strict-JSON consumers of the summary record.
        return round(sum(bucket) / len(bucket), 3) if bucket else None

    g1 = _rate(greedy_correct_first10)
    g2 = _rate(greedy_correct_last10)
    print(
        f"\n[final probe] correctness={pf['correctness']:.2f} "
        f"magnitude_appropriateness={pf['magnitude_appropriateness']:.2f} "
        f"(metric gain={pf['gain_used']}, effort_lambda={pf['effort_lambda']})"
    )
    for b, v in pf["bins"].items():
        print(
            f"      {b:<11} argmax={str(v['argmax']):<15} optimal={str(v['optimal']):<15} "
            f"dir={v['correct']!s:<5} mag={v['magnitude_ok']!s:<5} biases={v['biases']}"
        )
    print(f"[learning] greedy turned-toward rate: first-10 trials {g1} -> last-10 trials {g2}")
    resid_all = round(sum(greedy_residuals) / len(greedy_residuals), 4) if greedy_residuals else None
    resid_last = (
        round(sum(greedy_residuals[-10:]) / len(greedy_residuals[-10:]), 4) if greedy_residuals else None
    )
    print(
        f"[S4 metric] greedy residual |az_after| (readout={args.readout}): "
        f"overall {resid_all} -> settled last-10 {resid_last}   (lower = better centering)"
    )
    if app is not None:
        print(f"[apparatus] final yaw->az gain estimate: {app.gain:.2f}/rad (geometric would be {GEOMETRIC_GAIN:.2f})")
    print(f"[nac] persisted to {args.nac_path} — rerun with --session s2 for the cross-session arm")
    log.write(
        "summary",
        session=args.session,
        readout=args.readout,
        final_probe=pf,
        greedy_first10=g1,
        greedy_last10=g2,
        greedy_residual_overall=resid_all,
        greedy_residual_last10=resid_last,
        gain=None if app is None else round(app.gain, 3),
    )
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
