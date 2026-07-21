#!/usr/bin/env python
"""Cradle-mother fade-curve harness — the mother-scaffolded orient experiment.

Runs the ``cradle_mother`` 4-act arc (substrate-primary, no LLM in the action
path) across 3 ablation arms × N seeds, and extracts the FADE CURVE: per act, how
often the infant is oriented (fed) — which in the AUTONOMOUS acts (no mother
guide) means it oriented ITSELF. The question: as the guide fades (Act 1 full →
Act 3/4 none), does the infant keep getting fed because it learned to orient
toward the mother's voice?

Arms (same arc; the mother tick reads env toggles):
  taught       : full scenario — fading guide + feed + stimulus + speak.
  drive_only   : the mother still places the sound (stimulus) but does NOT guide
                 or feed — the infant must orient from its centeredness drive
                 alone (MAXIM_CRADLE_MOTHER_DISABLE_CARE=1). Isolates the mother's
                 feed-scaffold contribution.
  scaffold_only: full scenario BUT bio-learning disabled
                 (MAXIM_NAC_REWARD_BIAS_DISABLED=1) — guided+fed but cannot LEARN
                 from it. Isolates whether the fade curve is learned.

Read (analyze_cradle_mother.py): `taught` keeps a high fed-rate through the
autonomous acts (learned to self-orient) while `scaffold_only` drops (guided-fed
but never learned) and `drive_only` is the built-in-drive floor.

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
EMBODIMENT = "bodies/infant_humanoid"
ARMS = ("taught", "drive_only", "scaffold_only")

# Per-arm environment (the arc is identical; the arm is selected by env toggles).
_ARM_ENV: dict[str, dict[str, str]] = {
    "taught": {},
    "drive_only": {"MAXIM_CRADLE_MOTHER_DISABLE_CARE": "1"},
    "scaffold_only": {"MAXIM_NAC_REWARD_BIAS_DISABLED": "1"},
}


def _resolve_maxim() -> list[str]:
    """Invoke maxim as a module so the caller's PYTHONPATH=src worktree wins."""
    return [sys.executable, "-m", "maxim"]


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
        if k in ("fed", "guided"):
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
    """Read the sub-sim log and return per-act fade metrics:
    {act: {turns, fed_rate, guided_rate, self_orient_rate}}.

    ``fed_rate`` = fraction of mother-ticks where the infant was oriented (fed).
    In autonomous acts (guided_rate == 0) fed_rate IS the self-orient rate — the
    infant produced the orient itself. ``self_orient_rate`` reports fed-when-not-
    guided-this-turn for every act (the direct self-orient signal)."""
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
        guided = sum(1 for r in recs if r.get("guided"))
        # self-orient: fed on a turn the mother did NOT guide (infant's own orient)
        self_orient = sum(1 for r in recs if r.get("fed") and not r.get("guided"))
        result[act] = {
            "turns": n,
            "fed_rate": fed / n if n else 0.0,
            "guided_rate": guided / n if n else 0.0,
            "self_orient_rate": self_orient / n if n else 0.0,
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
    """Deterministic synthetic fade curve per arm (CI smoke — no subprocess/LLM).
    taught: self-orient rises across acts; scaffold_only: fed while guided, drops
    when the guide fades; drive_only: low floor."""
    acts = ["act1_fully_guided", "act2_co_active", "act3_autonomous", "act4_autonomous_voice"]
    if arm == "taught":
        so = [0.0, 0.4, 0.85, 0.9]
    elif arm == "scaffold_only":
        so = [0.0, 0.1, 0.15, 0.1]
    else:  # drive_only
        so = [0.0, 0.2, 0.35, 0.3]
    j = (seed % 3) * 0.02
    return {
        a: {
            "turns": 10,
            "fed_rate": min(1.0, (1.0 if i < 1 else so[i]) + j),
            "guided_rate": 1.0 if i == 0 else (0.5 if i == 1 else 0.0),
            "self_orient_rate": min(1.0, so[i] + j),
        }
        for i, a in enumerate(acts)
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Cradle-mother fade-curve harness")
    p.add_argument("--arms", default=",".join(ARMS), help="comma-separated: taught,drive_only,scaffold_only")
    p.add_argument("--trials", type=int, default=6)
    p.add_argument("--seed-base", type=int, default=42)
    p.add_argument("--sim-max-turns", type=int, default=44, help="~11 per act × 4 acts")
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
                    rec = {"experiment": "cradle_mother", "arm": arm, "seed": seed, "mock": args.mock, "fade": fade}
                    fh.write(json.dumps(rec) + "\n")
                    fh.flush()
                    n_ok += 1
                    acts_summary = " ".join(
                        f"{a.split('_')[0]}:{v['self_orient_rate']:.2f}" for a, v in sorted(fade.items())
                    )
                    dt = (_t.monotonic() - t0) if hasattr(_t, "monotonic") else 0
                    print(f"ok {arm} seed={seed} self_orient[{acts_summary}] ({dt:.0f}s)")
                except Exception as e:  # noqa: BLE001 — one run's failure must not kill the sweep
                    n_fail += 1
                    print(f"FAIL {arm} seed={seed}: {e}", file=sys.stderr)

    print(f"\ndone: {n_ok} runs recorded, {n_fail} failed → {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
