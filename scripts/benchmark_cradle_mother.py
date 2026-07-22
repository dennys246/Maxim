#!/usr/bin/env python
"""Cradle-mother OPERANT orient harness — does a mother TEACH orienting?

Runs the ``cradle_mother`` arc (substrate-primary, no LLM in the action path) on
``bodies/infant_operant`` (hunger drive + a DRIVELESS azimuth sensor — no
intrinsic reason to orient) across 2 arms × N seeds, and extracts the LEARNING
CURVE: per time-bin ("act"), the infant's DIRECTEDNESS — the fraction of turns
its own turn moved it TOWARD the sound. The question: does the infant learn to
orient PURELY from the mother's contingent feeding (operant credit), with no
innate orient drive?

Arms (same arc; the mother tick reads env toggles). BOTH set
MAXIM_OPERANT_ONLY_CREDIT=1 so the tool-success floor can't drown the signal:
  taught  : the mother shapes — feeds + credits the infant's own action when it
            turned toward the sound (NAc.credit_operant_reward, the sole teacher).
  no_feed : the mother still PLACES the sound but never feeds/credits
            (MAXIM_CRADLE_MOTHER_DISABLE_CARE=1). With no intrinsic drive the
            infant has NO teacher — the CONTROL that isolates the mother.

Read (analyze_cradle_mother.py): `taught` directedness RISES across the bins and
ends high (learned to self-orient); `no_feed` stays at chance (~0.5). The gap is
the mother's operant teaching. directedness is logged in BOTH arms (the mother
computes progress even when she doesn't feed), so the control is measurable.

Metric source: the per-turn ``mother`` telemetry (``sim_log``), captured via
MAXIM_LOG_FILE per sub-sim. Fail-soft, deterministic per seed, ``--mock`` for CI.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

ARC = "cradle_mother"
EMBODIMENT = "bodies/infant_operant"  # hunger drive + DRIVELESS azimuth (no intrinsic orient)
ARMS = ("taught", "no_feed")

# Shared conditions for BOTH arms:
#   MAXIM_OPERANT_ONLY_CREDIT — the tool-success floor never drowns the operant
#     signal (probe 3 tool_floor arm).
#   MAXIM_SIM_SUBSTRATE_EXPLORE_BONUS_WEIGHT — the infant must EXPLORE turns to
#     bootstrap (no intrinsic drive, cold-start bias). Without it it never turns
#     and stays inert (the original review's B1, confirmed in the operant smoke).
_SHARED_ENV = {"MAXIM_OPERANT_ONLY_CREDIT": "1", "MAXIM_SIM_SUBSTRATE_EXPLORE_BONUS_WEIGHT": "1.5"}
# Per-arm environment (the arc is identical; the arm is selected by env toggles).
_ARM_ENV: dict[str, dict[str, str]] = {
    # Full operant teaching: mother shapes (feeds + credits toward-turns).
    "taught": {**_SHARED_ENV},
    # Control: mother still PLACES the sound but never feeds/credits — with no
    # intrinsic orient drive the infant has NO teacher and stays at chance.
    "no_feed": {**_SHARED_ENV, "MAXIM_CRADLE_MOTHER_DISABLE_CARE": "1"},
}


def _resolve_maxim() -> list[str]:
    """Invoke maxim as a module so the caller's PYTHONPATH=src worktree wins."""
    return [sys.executable, "-m", "maxim"]


def _git_hash() -> str:
    """Short git HEAD, recorded per result so a stale checkout is detectable
    (verify the results' git_hash before trusting them — the hard-won lesson)."""
    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=Path(__file__).parent
            ).stdout.strip()
            or "unknown"
        )
    except Exception:
        return "unknown"


# ── mother-telemetry parsing (the fade metric) ──────────────────────────────


def _parse_mother_record(msg: str) -> dict[str, Any] | None:
    """Parse a ``mother`` sim_log message: 'act=<a> fed=<b> guided=<b> az_prior=..
    az_stimulus=.. az_guided=..' → dict. None if not a mother record."""
    if not msg.startswith("act="):
        return None
    out: dict[str, Any] = {}
    for tok in msg.split():
        if "=" not in tok:
            continue
        k, v = tok.split("=", 1)
        if k in ("fed", "guided", "credited"):
            out[k] = v == "True"
        elif k == "act":
            out[k] = v
        else:
            try:
                out[k] = None if v in ("None", "") else float(v)
            except ValueError:
                out[k] = v
    return out if "act" in out else None


def _extract_fade(log_path: Path) -> dict[str, dict[str, float]]:
    """Read the sub-sim log and return per-act OPERANT learning-curve metrics:
    {act: {turns, directedness, fed_rate, credited_rate}}.

    ``directedness`` = fraction of measured turns where the infant's own turn
    moved it TOWARD the sound (``progress > 0``). This is the arm-independent
    learning signal: it is logged in BOTH arms (the mother computes progress even
    when she doesn't feed), so ``taught`` (directedness rises across acts) can be
    compared against ``no_feed`` (stays ~chance). ``fed_rate``/``credited_rate``
    are diagnostics for the taught arm."""
    per_act: dict[str, list[dict[str, Any]]] = {}
    if not log_path.exists():
        return {}
    for line in log_path.read_text().splitlines():
        line = line.strip()
        if not line or "act=" not in line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        m = _parse_mother_record(str(rec.get("message", "")))
        if m is None:
            continue
        per_act.setdefault(m["act"], []).append(m)

    result: dict[str, dict[str, float]] = {}
    for act, recs in per_act.items():
        n = len(recs)
        fed = sum(1 for r in recs if r.get("fed"))
        credited = sum(1 for r in recs if r.get("credited"))
        # directedness: of the turns where progress was measurable, how many moved
        # the infant toward the sound (progress > 0)? Arm-independent.
        measured = [r for r in recs if isinstance(r.get("progress"), (int, float))]
        directed = sum(1 for r in measured if r["progress"] > 0)
        result[act] = {
            "turns": n,
            "directedness": directed / len(measured) if measured else 0.0,
            "measured": len(measured),
            "fed_rate": fed / n if n else 0.0,
            "credited_rate": credited / n if n else 0.0,
        }
    return result


# ── one sub-sim run ─────────────────────────────────────────────────────────


def _run_one(arm: str, seed: int, *, model: str, max_turns: int, timeout_s: int, workdir: Path) -> dict[str, Any]:
    data_home = workdir / f"{arm}_seed{seed}"
    data_home.mkdir(parents=True, exist_ok=True)
    src_models = Path(os.path.expanduser("~/.maxim/models"))
    link = data_home / "models"
    if src_models.exists() and not link.exists():
        try:
            link.symlink_to(src_models)
        except OSError:
            pass

    log_path = data_home / "mother_log.jsonl"
    env = os.environ.copy()
    env["MAXIM_DATA_HOME"] = str(data_home)
    env["MAXIM_LLM_PROFILE"] = model
    env["MAXIM_AUTO_SPAWN_LLM_SERVER"] = "0"
    env["MAXIM_LLM_CLOUD_ENABLED"] = "0"
    env["MAXIM_ROLE"] = "solo"
    env["MAXIM_LOG_FILE"] = str(log_path)
    env.update(_ARM_ENV[arm])

    cmd = _resolve_maxim() + [
        "--sim",
        ARC,
        "--aut-mode",
        "substrate-primary",
        "--embodiment",
        EMBODIMENT,
        "--interactive",
        "false",
        "--sim-max-turns",
        str(max_turns),
        "--seed",
        str(seed),
    ]
    logdir = data_home / "harness_logs"
    logdir.mkdir(parents=True, exist_ok=True)
    try:
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        (logdir / "timeout.log").write_text(f"TIMEOUT {timeout_s}s\n{exc.stderr or ''}")
        raise RuntimeError(f"{arm} seed={seed}: sub-sim timed out after {timeout_s}s") from exc
    (logdir / "run.log").write_text((proc.stdout or "") + "\n---STDERR---\n" + (proc.stderr or ""))
    return _extract_fade(log_path)


def _mock_fade(arm: str, seed: int) -> dict[str, dict[str, float]]:
    """Deterministic synthetic operant learning curve per arm (CI smoke — no
    subprocess/LLM). taught: directedness RISES from chance to learned across the
    4 time-bins; no_feed: stays ~chance (no teacher)."""
    acts = ["act1_early", "act2_warming", "act3_consolidating", "act4_autonomous"]
    if arm == "taught":
        di = [0.5, 0.68, 0.82, 0.9]
    else:  # no_feed — chance floor
        di = [0.5, 0.5, 0.5, 0.5]
    j = (seed % 3) * 0.02
    return {
        a: {
            "turns": 11,
            "measured": 10,
            "directedness": min(1.0, di[i] + j),
            "fed_rate": min(1.0, (di[i] if arm == "taught" else 0.0)),
            "credited_rate": min(1.0, (di[i] if arm == "taught" else 0.0)),
        }
        for i, a in enumerate(acts)
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Cradle-mother fade-curve harness")
    p.add_argument("--arms", default=",".join(ARMS), help="comma-separated: taught,no_feed")
    p.add_argument("--trials", type=int, default=6)
    p.add_argument("--seed-base", type=int, default=42)
    p.add_argument(
        "--sim-max-turns", type=int, default=56, help="4 acts × 12 + margin (payoff act must get its full sample)"
    )
    p.add_argument("--model", default="mistral-7b", help="narrator profile (prose-less arc → light use)")
    p.add_argument("--timeout-s", type=int, default=1800)
    p.add_argument("--out", required=True)
    p.add_argument("--workdir", default="/tmp/cradle_mother_runs")
    p.add_argument("--mock", action="store_true", help="synthetic fade (CI smoke, no subprocess)")
    p.add_argument("--resume", action="store_true", help="skip (arm,seed) already in --out")
    args = p.parse_args()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    for a in arms:
        if a not in _ARM_ENV:
            print(f"unknown arm {a!r}; valid: {list(_ARM_ENV)}", file=sys.stderr)
            return 2

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done: set[tuple[str, int]] = set()
    if args.resume and out_path.exists():
        for line in out_path.read_text().splitlines():
            try:
                r = json.loads(line)
                done.add((r["arm"], r["seed"]))
            except (json.JSONDecodeError, KeyError):
                pass

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    n_ok = n_fail = 0
    import time as _t

    with out_path.open("a") as fh:
        for arm in arms:
            for trial in range(args.trials):
                seed = args.seed_base + trial
                if (arm, seed) in done:
                    continue
                t0 = _t.monotonic() if hasattr(_t, "monotonic") else 0
                try:
                    fade = (
                        _mock_fade(arm, seed)
                        if args.mock
                        else _run_one(
                            arm,
                            seed,
                            model=args.model,
                            max_turns=args.sim_max_turns,
                            timeout_s=args.timeout_s,
                            workdir=workdir,
                        )
                    )
                    rec = {
                        "experiment": "cradle_mother",
                        "arm": arm,
                        "seed": seed,
                        "mock": args.mock,
                        "git_hash": _git_hash(),
                        "fade": fade,
                    }
                    fh.write(json.dumps(rec) + "\n")
                    fh.flush()
                    n_ok += 1
                    acts_summary = " ".join(
                        f"{a.split('_')[0]}:{v['directedness']:.2f}" for a, v in sorted(fade.items())
                    )
                    dt = (_t.monotonic() - t0) if hasattr(_t, "monotonic") else 0
                    print(f"ok {arm} seed={seed} directed[{acts_summary}] ({dt:.0f}s)")
                except Exception as e:  # noqa: BLE001 — one run's failure must not kill the sweep
                    n_fail += 1
                    print(f"FAIL {arm} seed={seed}: {e}", file=sys.stderr)

    print(f"\ndone: {n_ok} runs recorded, {n_fail} failed → {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
