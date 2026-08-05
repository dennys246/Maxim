#!/usr/bin/env python3
"""Exp 49 trial driver — two-joint centering (docs/experiments/49_two_joint_centering.md).

Arms (pre-registered):

- ``scripted`` — OFFLINE smoke: no ``maxim`` spawn, no LLM. An
  in-process closed loop drives the SimulatedController's honest-physics
  DoA scenario with the derived-boundary turn policy, and the SAME
  metric extraction that scores the real arms scores the smoke — the
  instrument is verified before it measures anything (harness physics,
  JSONL emission, centered detection, direction-correctness).
- ``A`` — head-only: full live stack against the ``reachy_mini_headonly``
  body variant (orient modulator removed — the turn tools are genuinely
  absent, not placebo-stubbed; amendment 1) + ``motor_binding: false``
  belt-and-suspenders. focus_on_sound is the whole orient repertoire.
- ``B`` — full: live stack with motor binding, LLM proposer.
- ``C`` — substrate: live stack with motor binding,
  ``aut_mode: substrate-primary`` (no LLM in the action path).

Every spawned arm passes the provenance gate (scripts/_provenance.py)
BEFORE the first trial and stamps ``executed_code_provenance`` into every
trial record (the Exp 42b rule: a result whose code-under-test cannot be
established is not a validation). Each trial runs in its own sandbox with
``MAXIM_DATA_HOME`` isolation — robots.yaml resolution is data-home-aware
as of this branch, so the operator's real ``~/.maxim/robots.yaml`` (and
persisted NAc state) is never touched.

Usage:
    export PYTHONPATH="$PWD/src"     # absolute, own line (Exp 42b lesson)
    python scripts/exp49/run_trials.py --arm scripted --out /tmp/exp49_smoke
    python scripts/exp49/run_trials.py --arm C --out ~/exp49/armC
    python scripts/exp49/run_trials.py --arm B --out ~/exp49/armB --limit-trials 1
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent))

import _provenance  # noqa: E402  (scripts/_provenance.py — by path, same tree as this harness)
import exp49_common as common  # noqa: E402


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _assert_in_process_repo() -> str:
    """The scripted smoke imports maxim IN-PROCESS — assert it is THIS repo.

    The installed-package-shadows-worktree trap: a stale editable install
    can shadow the checkout even in-process. Same standard as the spawn
    gate, applied to our own interpreter.
    """
    import maxim

    imported_root = Path(maxim.__file__).resolve().parent.parent
    src = (REPO_ROOT / "src").resolve()
    if imported_root != src:
        print(
            f"PROVENANCE FAILURE: in-process `maxim` is {imported_root}, not {src}.\n"
            f'  Fix: export PYTHONPATH="{src}" (absolute, own line).',
            file=sys.stderr,
        )
        sys.exit(3)
    return str(maxim.__file__)


# ─────────────────────────────────────────────────────────────────────────────
# Scripted offline smoke (in-process)
# ─────────────────────────────────────────────────────────────────────────────


def run_scripted_trial(bearing_deg: float, seed: int, sandbox: Path, *, speech_density: float = 1.0) -> dict:
    """One in-process closed-loop trial with the scripted proposer.

    Exercises: scenario physics (fold, inverse mapping, noise, pose
    tracking), the production ``gated_azimuth`` sampling helper, body
    turns with head ride-along, the neck envelope, JSONL event emission,
    and the metric extractor — everything except the agent runtime.
    """
    from maxim.embodiment.audio_localization import gated_azimuth
    from maxim.hardware.controller import MotionTarget
    from maxim.hardware.simulation.controller import SimulatedController
    from maxim.utils.structured_logging import StructuredFormatter

    sandbox.mkdir(parents=True, exist_ok=True)
    jsonl = sandbox / "maxim.jsonl"

    handler = logging.FileHandler(jsonl)
    handler.setFormatter(StructuredFormatter())
    handler.setLevel(logging.DEBUG)
    scenario_logger = logging.getLogger("maxim.sim_doa")
    prev_level = scenario_logger.level
    scenario_logger.setLevel(logging.DEBUG)
    scenario_logger.addHandler(handler)

    controller = SimulatedController(
        doa_source_bearing_deg=bearing_deg,
        doa_noise_sigma=common.DOA_NOISE_SIGMA,
        doa_speech_density=speech_density,
        doa_seed=seed,
        head_yaw_limit_deg=common.HEAD_YAW_LIMIT_DEG,
        body_yaw_limit_deg=common.BODY_YAW_LIMIT_DEG,
        video_enabled=False,
    )
    assert controller.connect()
    reader = controller.get_doa_reader()
    assert reader is not None

    expected_converged = False
    try:
        for _action in range(common.MAX_ACTIONS):
            az = gated_azimuth(reader, k=3, timeout_s=1.0, poll_s=0.01)
            if az is None:
                break
            # In-process truth check, independent of the JSONL channel:
            # unfolded offset from bearing + pose (the same quantities the
            # scenario logs — computed here separately so a JSONL bug
            # cannot hide from the smoke).
            pose = controller.get_current_pose()
            theta_deg = math.degrees(math.radians(bearing_deg) - float(pose["yaw"]))
            while theta_deg > 180:
                theta_deg -= 360
            while theta_deg <= -180:
                theta_deg += 360
            if abs(theta_deg) < common.CENTERED_THETA_DEG:
                # Two confirmation reads inside the band (the sustain rule).
                a1 = gated_azimuth(reader, k=3, timeout_s=1.0, poll_s=0.01)
                a2 = gated_azimuth(reader, k=3, timeout_s=1.0, poll_s=0.01)
                if a1 is not None and a2 is not None:
                    expected_converged = True
                break
            _name, delta = common.scripted_policy_action(az)
            body = float(pose["body_yaw"])
            controller.goto_target(MotionTarget(body_yaw=body + delta, duration=0.1))
        # Final reads so the extractor sees the sustained-centered tail.
        for _ in range(2):
            gated_azimuth(reader, k=3, timeout_s=1.0, poll_s=0.01)
    finally:
        scenario_logger.removeHandler(handler)
        scenario_logger.setLevel(prev_level)
        handler.close()
        controller.disconnect()

    events = common.iter_jsonl_events(jsonl)
    metrics = common.compute_trial_metrics(events)
    metrics.end_reason = "centered" if metrics.centered else "action_cap"
    # Instrument cross-check: the extractor and the in-process truth must
    # agree on convergence, or the JSONL channel is lying.
    if expected_converged != metrics.centered:
        metrics.notes.append(
            f"INSTRUMENT MISMATCH: in-process converged={expected_converged} but extractor centered={metrics.centered}"
        )
    return {
        "arm": "scripted",
        "bearing_deg": bearing_deg,
        "seed": seed,
        "started_at": _now_iso(),
        "jsonl": str(jsonl),
        "instrument_ok": expected_converged == metrics.centered,
        "metrics": metrics.to_dict(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Spawned live-stack trials (arms A / B / C)
# ─────────────────────────────────────────────────────────────────────────────


def _trial_env(arm: str, home: Path, jsonl: Path, *, min_confidence: float | None) -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = str((REPO_ROOT / "src").resolve())
    env["MAXIM_DATA_HOME"] = str(home)
    env["MAXIM_LOG_FILE"] = str(jsonl)
    env["MAXIM_DISABLE_IMAGINATION"] = "1"
    env["MAXIM_MOTOR_CREDIT_TRACE"] = "1"
    env["MAXIM_ROBOT_TIMEOUT"] = "10"
    if arm == "C":
        env["MAXIM_SUBSTRATE_PATH"] = "1"
        env["MAXIM_LLM_ENABLED"] = "0"
        # Restrict substrate-primary to the orient repertoire so
        # always-succeed generic tools don't out-compete the affordances
        # under test (the probe-3 floor lesson; substring match).
        env["MAXIM_SUBSTRATE_TOOL_WHITELIST"] = "turn_left,turn_right"
    if min_confidence is not None:
        # Documented experiment knob (CLAUDE.md env table): bypasses
        # recommend_action's cold-start gate. Needed for arm C with an
        # imported live policy — cluster_reward_bias is CAPPED (~0.2)
        # below the default 0.3 threshold, so a real learned bias can
        # never act through the default gate (the known clamp-vs-
        # threshold asymmetry). Recorded in the trial record.
        env["MAXIM_NAC_MIN_CONFIDENCE"] = str(min_confidence)
    return env


def _import_substrate(home: Path, source_dir: Path) -> dict:
    """Copy a trained nac.json + ec.json PAIR into the trial's data home.

    The pair rule (CLAUDE.md nac_cross_session_persistence invariant):
    NAc biases key on EC node ids — restoring either without the other
    leaves biases dangling on nodes a fresh EC never re-allocates. Refuses
    a half-pair. build_bio_stack restores both on startup because the
    sandbox home now "has prior files". Returns provenance for the record.
    """
    import hashlib

    nac_src = source_dir / "nac.json"
    ec_src = source_dir / "ec.json"
    if not nac_src.exists() or not ec_src.exists():
        raise SystemExit(f"--import-substrate needs BOTH nac.json and ec.json in {source_dir} (the pair rule)")
    mem = home / "memory"
    mem.mkdir(parents=True, exist_ok=True)
    prov = {}
    for src in (nac_src, ec_src):
        data = src.read_bytes()
        (mem / src.name).write_bytes(data)
        prov[src.name] = {"source": str(src), "sha256": hashlib.sha256(data).hexdigest()[:16]}
    return prov


def run_spawned_trial(
    arm: str,
    bearing_deg: float,
    seed: int,
    sandbox: Path,
    maxim_bin: str,
    *,
    import_substrate: Path | None = None,
    min_confidence: float | None = None,
    speech_density: float = 1.0,
) -> dict:
    sandbox.mkdir(parents=True, exist_ok=True)
    home = sandbox / "home"
    home.mkdir(exist_ok=True)
    jsonl = sandbox / "maxim.jsonl"
    (home / "robots.yaml").write_text(
        common.robots_yaml_text(arm=arm, bearing_deg=bearing_deg, seed=seed, speech_density=speech_density)
    )
    if arm == "A":
        # Head-only body variant into the sandbox's user-components layer
        # (data_home()/components — MAXIM_DATA_HOME-scoped, shadows nothing
        # outside this trial). See headonly_body_yaml for the review fold.
        bundled = REPO_ROOT / "src" / "maxim" / "_data" / "components" / "bodies" / "reachy_mini.yaml"
        variant_dir = home / "components" / "bodies"
        variant_dir.mkdir(parents=True, exist_ok=True)
        (variant_dir / "reachy_mini_headonly.yaml").write_text(common.headonly_body_yaml(bundled))
    substrate_prov = None
    if import_substrate is not None:
        substrate_prov = _import_substrate(home, import_substrate)

    cmd = [
        maxim_bin,
        "--mode",
        "live",
        "--interactive",
        "false",
        "--audio",
        "false",
        "--home-dir",
        str(sandbox / "rundata"),
        "--verbosity",
        "1",
    ]
    env = _trial_env(arm, home, jsonl, min_confidence=min_confidence)
    started = time.time()
    record: dict = {
        "arm": arm,
        "bearing_deg": bearing_deg,
        "seed": seed,
        "started_at": _now_iso(),
        "cmd": cmd,
        "jsonl": str(jsonl),
        "min_confidence_env": min_confidence,
        "imported_substrate": substrate_prov,
        "provenance": _provenance.executed_code_provenance(REPO_ROOT, maxim_bin),
    }
    stdout_log = open(sandbox / "stdout.log", "w")
    proc = subprocess.Popen(cmd, env=env, stdout=stdout_log, stderr=subprocess.STDOUT, cwd=str(sandbox))
    end_reason = "timeout"
    try:
        while True:
            time.sleep(2.0)
            if proc.poll() is not None:
                end_reason = f"process_exited_{proc.returncode}"
                break
            events = common.iter_jsonl_events(jsonl)
            reason = common.check_trial_end(events, started_at=started, now=time.time())
            if reason is not None:
                end_reason = reason
                break
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:
                    proc.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    proc.kill()
        stdout_log.close()

    events = common.iter_jsonl_events(jsonl)
    metrics = common.compute_trial_metrics(events)
    metrics.end_reason = end_reason
    record["metrics"] = metrics.to_dict()
    record["wall_seconds"] = round(time.time() - started, 1)
    return record


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True, choices=["scripted", "A", "B", "C"])
    ap.add_argument("--out", required=True, help="Output directory (sandboxes + trials JSONL)")
    ap.add_argument(
        "--bearings",
        default=None,
        help="Comma-separated bearings in degrees (+=LEFT). Default: pre-registered ±{40,60,90,120,160}.",
    )
    ap.add_argument("--seed-base", type=int, default=4900)
    ap.add_argument("--limit-trials", type=int, default=0, help="Run only the first N trials (0 = all)")
    ap.add_argument(
        "--import-substrate",
        default=None,
        help="Directory holding a trained nac.json + ec.json PAIR to seed each trial's "
        "data home (arm C 'trained policy import'). Both files required.",
    )
    ap.add_argument(
        "--min-confidence",
        type=float,
        default=None,
        help="Sets MAXIM_NAC_MIN_CONFIDENCE for spawned trials (arm C: 0.0 lets a "
        "capped cluster bias act through the 0.3 default gate).",
    )
    ap.add_argument(
        "--speech-density",
        type=float,
        default=1.0,
        help="Probability per DoA read that is_speech is True (default 1.0 = the "
        "dense main arms; ~0.3-0.5 = the pre-registered sparse follow-on arm, "
        "the live-conversation condition Exp 49's main arms deliberately removed).",
    )
    args = ap.parse_args()

    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.bearings:
        bearings = [float(b) for b in args.bearings.split(",")]
    else:
        bearings = [b for mag in common.BEARINGS_DEG for b in (mag, -mag)]

    maxim_bin = None
    if args.arm == "scripted":
        maxim_file = _assert_in_process_repo()
        print(f"provenance OK (in-process): {maxim_file}")
    else:
        maxim_bin = shutil.which("maxim")
        if not maxim_bin:
            print("`maxim` console script not found on PATH — activate the venv first.", file=sys.stderr)
            return 2
        try:
            resolved = _provenance.assert_repo_interpreter(REPO_ROOT, maxim_bin)
        except _provenance.ProvenanceError as e:
            print(f"PROVENANCE FAILURE:\n{e}", file=sys.stderr)
            return 3
        print(f"provenance OK (spawn): {resolved}")

    trials_path = out_dir / f"trials_{args.arm}.jsonl"
    records: list[dict] = []
    for i, bearing in enumerate(bearings):
        if args.limit_trials and i >= args.limit_trials:
            break
        seed = args.seed_base + i
        sandbox = out_dir / f"trial_{args.arm}_{i:02d}_b{bearing:+.0f}_s{seed}"
        print(f"[{i + 1}/{len(bearings)}] arm={args.arm} bearing={bearing:+.0f}° seed={seed} ... ", end="", flush=True)
        if args.arm == "scripted":
            rec = run_scripted_trial(bearing, seed, sandbox, speech_density=args.speech_density)
        else:
            rec = run_spawned_trial(
                args.arm,
                bearing,
                seed,
                sandbox,
                maxim_bin,
                import_substrate=(Path(args.import_substrate).expanduser() if args.import_substrate else None),
                min_confidence=args.min_confidence,
                speech_density=args.speech_density,
            )
        rec["speech_density"] = args.speech_density
        records.append(rec)
        m = rec["metrics"]
        print(
            f"{m['end_reason']} centered={m['centered']} actions={m['actions_total']} "
            f"body_turns={m['body_turns']} first_correct={m['first_body_turn_correct']} "
            f"h3={m['credited_sign_accuracy']}"
        )
        with open(trials_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    summary = common.summarize_arm(records)
    print(f"\narm {args.arm} summary: {json.dumps(summary, indent=2)}")
    (out_dir / f"summary_{args.arm}.json").write_text(json.dumps({"arm": args.arm, **summary}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
