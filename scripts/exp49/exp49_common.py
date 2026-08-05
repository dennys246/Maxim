"""Shared Exp 49 harness pieces: config writer, JSONL parsing, metrics.

Pre-registration: docs/experiments/49_two_joint_centering.md. The trial
driver (run_trials.py) and the offline scripted smoke both route through
this module so the metric definitions exist in exactly one place — the
instrument being verified once, then reused (verify-the-instrument).

JSONL event shapes (StructuredFormatter compact form — ``e`` is the
event name, data keys flattened at top level, None values dropped):

- ``sim_doa.read``   — the scenario's per-read truth channel:
  ``theta_deg`` (UNFOLDED head-relative offset, + = left), ``az_true``
  (folded, noiseless), ``az_read`` (folded + noise), ``speech``,
  ``head_world_deg``, ``body_yaw_deg``.
- ``sim_doa.motion`` — every controller motion: ``body_yaw_deg_before/
  after``, ``head_world_deg_before/after``, ``commanded_body``,
  ``commanded_head``.
- ``motor_credit.measured`` — Phase-2 measured relief credit decisions
  (MAXIM_MOTOR_CREDIT_TRACE=1): ``transitions`` {sensor: [before,
  after]}, ``potential_diff`` (signed; its SIGN is what record_outcome
  books; absent = nothing credited).

Pre-registered trial-end criteria: CENTERED = |theta_deg| < 9.9° (i.e.
|az| < 0.11 in the unfolded frame) sustained 2 consecutive reads; or the
action cap; or the wall-clock cap. The centered criterion gates on the
UNFOLDED theta — the folded az reads ~0 when facing exactly AWAY from
the source, and a harness gating on it would score a 180°-wrong head as
centered.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Pre-registered scenario constants (49_two_joint_centering.md).
BEARINGS_DEG: tuple[float, ...] = (40.0, 60.0, 90.0, 120.0, 160.0)  # ± counterbalanced
CENTERED_THETA_DEG = 9.9  # |az| < 0.11 → 0.11 * 90
CENTERED_SUSTAIN_READS = 2
MAX_ACTIONS = 12
MAX_TRIAL_SECONDS = 180.0
DOA_NOISE_SIGMA = 0.03  # live XVF3800 characterization
HEAD_YAW_LIMIT_DEG = 22.0  # measured neck envelope, not the optimistic 45
BODY_YAW_LIMIT_DEG = 160.0  # daemon clamp


def robots_yaml_text(
    *,
    arm: str,
    bearing_deg: float,
    seed: int,
    speech_density: float = 1.0,
) -> str:
    """The per-trial robots.yaml — the declarative scenario config.

    Arm differences are EXACTLY the pre-registered ones: arm A adds
    ``motor_binding: false`` (head-only — same prompt path, the turn
    tools simply never bind); arm C adds ``aut_mode: substrate-primary``.
    Everything else is byte-identical across arms.
    """
    lines = [
        "robots:",
        "  exp49_sim:",
        "    type: simulated",
        "    primary: true",
        "    config:",
        "      body: bodies/reachy_mini",
        "      video_enabled: false",
        f"      doa_source_bearing_deg: {bearing_deg}",
        f"      doa_noise_sigma: {DOA_NOISE_SIGMA}",
        f"      doa_speech_density: {speech_density}",
        f"      doa_seed: {seed}",
        f"      head_yaw_limit_deg: {HEAD_YAW_LIMIT_DEG}",
        f"      body_yaw_limit_deg: {BODY_YAW_LIMIT_DEG}",
        "      audio_salience: 0.9",
        "      audio_novelty: 0.6",
    ]
    if arm == "A":
        lines.append("      motor_binding: false")
    if arm == "C":
        lines.append("      aut_mode: substrate-primary")
    return "\n".join(lines) + "\n"


def iter_jsonl_events(path: Path) -> "list[dict[str, Any]]":
    """Parse a MAXIM_LOG_FILE JSONL into event dicts (compact form)."""
    events: list[dict[str, Any]] = []
    if not path.exists():
        return events
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(rec, dict) and "e" in rec:
                events.append(rec)
    return events


@dataclass
class TrialMetrics:
    """Per-trial pre-registered metrics (49_two_joint_centering.md)."""

    centered: bool = False
    time_to_center_s: float | None = None
    actions_total: int = 0
    actions_to_center: int | None = None
    body_turns: int = 0
    first_body_turn_correct: bool | None = None  # None = no body turn happened
    head_only_moves: int = 0
    final_abs_theta_deg: float | None = None
    min_abs_theta_deg: float | None = None
    plateaued_at_neck_limit: bool = False
    credited_turns: int = 0
    credited_sign_matches: int = 0
    credited_sign_accuracy: float | None = None
    end_reason: str = "unknown"
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def _nearest_read_before(reads: list[dict], t: float) -> dict | None:
    # STRICTLY before: JSONL timestamps are rounded to 10 ms, so a
    # post-motion read can TIE the motion's timestamp — `<=` would adopt
    # the post-turn frame as the "before" reference and invert the
    # direction-correctness verdict (caught by the scripted smoke).
    prior = [r for r in reads if r["t"] < t]
    return prior[-1] if prior else None


def _nearest_read_after(reads: list[dict], t: float) -> dict | None:
    later = [r for r in reads if r["t"] > t]
    return later[0] if later else None


def compute_trial_metrics(events: list[dict[str, Any]]) -> TrialMetrics:
    """Compute the pre-registered metrics from a trial's JSONL events.

    Direction-correctness convention: pose yaw is +LEFT; ``theta_deg``
    is + when the source is LEFT of the head. A correct body turn
    therefore moves body yaw in the SAME sign as theta at the most
    recent read before the turn.

    H3 (credited-turn sign accuracy): the credited sign is
    ``sign(potential_diff)``; the TRUE progress sign is the change in
    |theta| across the motion window (nearest truth reads before/after
    the credit event). A credit whose sign disagrees with the true
    |theta| trajectory is a leak in the measurement honesty gates.
    """
    m = TrialMetrics()
    reads = sorted((e for e in events if e.get("e") == "sim_doa.read"), key=lambda r: r["t"])
    motions = sorted((e for e in events if e.get("e") == "sim_doa.motion"), key=lambda r: r["t"])
    credits = sorted((e for e in events if e.get("e") == "motor_credit.measured"), key=lambda r: r["t"])

    if not reads:
        m.end_reason = "no_reads"
        m.notes.append("no sim_doa.read events — scenario never produced readings")
        return m

    t0 = reads[0]["t"]
    thetas = [(r["t"], float(r["theta_deg"])) for r in reads if "theta_deg" in r]
    if thetas:
        m.final_abs_theta_deg = abs(thetas[-1][1])
        m.min_abs_theta_deg = min(abs(th) for _, th in thetas)

    # Centered: N consecutive reads inside the band.
    consecutive = 0
    for t, th in thetas:
        consecutive = consecutive + 1 if abs(th) < CENTERED_THETA_DEG else 0
        if consecutive >= CENTERED_SUSTAIN_READS:
            m.centered = True
            m.time_to_center_s = round(t - t0, 3)
            m.actions_to_center = sum(1 for mo in motions if mo["t"] <= t)
            break

    # Actions.
    m.actions_total = len(motions)
    body_turns = [
        mo
        for mo in motions
        if mo.get("commanded_body") and abs(float(mo["body_yaw_deg_after"]) - float(mo["body_yaw_deg_before"])) > 0.5
    ]
    m.body_turns = len(body_turns)
    m.head_only_moves = sum(1 for mo in motions if mo.get("commanded_head") and not mo.get("commanded_body"))

    if body_turns:
        first = body_turns[0]
        delta = float(first["body_yaw_deg_after"]) - float(first["body_yaw_deg_before"])
        ref = _nearest_read_before(reads, first["t"])
        if ref is not None and "theta_deg" in ref:
            m.first_body_turn_correct = (delta > 0) == (float(ref["theta_deg"]) > 0)

    # Head-only plateau (arm A signature): never centered, no body turns,
    # and the head parked at the neck envelope while theta stayed out.
    if not m.centered and not body_turns and reads:
        last = reads[-1]
        head_rel = abs(float(last.get("head_world_deg", 0.0)) - float(last.get("body_yaw_deg", 0.0)))
        m.plateaued_at_neck_limit = head_rel >= HEAD_YAW_LIMIT_DEG - 2.0

    # H3: credited-turn sign accuracy against the truth channel.
    for c in credits:
        pd = c.get("potential_diff")
        if pd is None or abs(float(pd)) <= 1e-9:
            continue
        before = _nearest_read_before(reads, c["t"] - 0.0)
        # The credit event fires after the post-motion measurement; the
        # motion preceding it is the credited turn. Find it.
        turn = None
        for mo in reversed(motions):
            if mo["t"] <= c["t"]:
                turn = mo
                break
        if turn is None or before is None:
            m.notes.append("credit event without a matching motion/read — skipped")
            continue
        pre = _nearest_read_before(reads, turn["t"])
        post = _nearest_read_after(reads, turn["t"])
        if pre is None or post is None or "theta_deg" not in pre or "theta_deg" not in post:
            m.notes.append("credit event lacked surrounding truth reads — skipped")
            continue
        true_progress = abs(float(pre["theta_deg"])) - abs(float(post["theta_deg"]))
        if abs(true_progress) <= 1e-9:
            m.notes.append("credited turn with ~zero true progress — skipped for sign accuracy")
            continue
        m.credited_turns += 1
        if (float(pd) > 0) == (true_progress > 0):
            m.credited_sign_matches += 1
    if m.credited_turns:
        m.credited_sign_accuracy = m.credited_sign_matches / m.credited_turns

    return m


def check_trial_end(
    events: list[dict[str, Any]],
    *,
    started_at: float,
    now: float,
) -> str | None:
    """The driver's live end-condition check. Returns the end reason or None."""
    reads = [e for e in events if e.get("e") == "sim_doa.read" and "theta_deg" in e]
    consecutive = 0
    for r in sorted(reads, key=lambda r: r["t"]):
        consecutive = consecutive + 1 if abs(float(r["theta_deg"])) < CENTERED_THETA_DEG else 0
        if consecutive >= CENTERED_SUSTAIN_READS:
            return "centered"
    motions = [e for e in events if e.get("e") == "sim_doa.motion"]
    if len(motions) >= MAX_ACTIONS:
        return "action_cap"
    if now - started_at > MAX_TRIAL_SECONDS:
        return "timeout"
    return None


def summarize_arm(trial_records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate an arm's pre-registered metrics across trials."""
    n = len(trial_records)
    if n == 0:
        return {"trials": 0}
    ms = [t["metrics"] for t in trial_records]
    centered = [m for m in ms if m.get("centered")]
    first_turns = [m["first_body_turn_correct"] for m in ms if m.get("first_body_turn_correct") is not None]
    credited = sum(m.get("credited_turns", 0) for m in ms)
    matches = sum(m.get("credited_sign_matches", 0) for m in ms)
    times = [m["time_to_center_s"] for m in centered if m.get("time_to_center_s") is not None]
    acts = [m["actions_to_center"] for m in centered if m.get("actions_to_center") is not None]
    return {
        "trials": n,
        "centering_rate": round(len(centered) / n, 3),
        "mean_time_to_center_s": round(sum(times) / len(times), 2) if times else None,
        "mean_actions_to_center": round(sum(acts) / len(acts), 2) if acts else None,
        "first_body_turn_correct_rate": (round(sum(first_turns) / len(first_turns), 3) if first_turns else None),
        "body_turn_usage_rate": round(sum(1 for m in ms if m.get("body_turns", 0) > 0) / n, 3),
        "credited_turns": credited,
        "credited_sign_accuracy": round(matches / credited, 4) if credited else None,
        "plateau_rate": round(sum(1 for m in ms if m.get("plateaued_at_neck_limit")) / n, 3),
    }


def scripted_policy_action(az_read: float) -> tuple[str, float]:
    """The scripted proposer: which turn answers this (folded) reading.

    Mirrors the Exp 45c derived decision boundary (~0.33): below it the
    normal ±17° step wins, above it the big ±52° step. Returns
    ``(affordance_name, body_yaw_delta_rad)``. Convention: az < 0 =
    LEFT → +yaw (LEFT) body turn — the same sign map as the YAML
    self_effect deltas the motor backend dispatches.
    """
    big = abs(az_read) >= 0.33
    magnitude = math.radians(52.0) if big else math.radians(17.0)
    if az_read < 0:
        return ("turn_left_big" if big else "turn_left", magnitude)
    return ("turn_right_big" if big else "turn_right", -magnitude)
