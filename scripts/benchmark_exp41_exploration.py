#!/usr/bin/env python3
"""Harness for Exp 41 — Substrate-Primary Exploration (counter-prior 2×2).

Runs the four arms of docs/experiments/41_substrate_primary_exploration.md §4
under ``--aut-mode substrate-primary`` (no LLM in the action path → ``cost=$0``):

    | arm    | arc                            | exploration |
    |--------|--------------------------------|-------------|
    | A_cons | cradle_prelinguistic           | OFF         |
    | B_cons | cradle_prelinguistic           | ON          |
    | A_dec  | cradle_prelinguistic_deceptive | OFF         |
    | B_dec  | cradle_prelinguistic_deceptive | ON          |

Exploration is toggled via ``MAXIM_SIM_SUBSTRATE_EXPLORE_BONUS_WEIGHT`` (0.0 vs
``--explore-weight``). Each run uses the COLD infant body so warmth-seeking is a
sustained, drive-relevant temptation (see substrate_primary_cradle_readiness.md).

This is a DEDICATED harness, not part of benchmark_cross_session.py (Exp 37):
the substrate-primary 2×2 has none of Exp 37's cross-session / cost / cloud /
ablation-arm / FAILURE_CLASS machinery, and bolting it on would risk that
shipped harness. It reuses the same actions.jsonl contract (header line skipped;
per-action ``tool`` field) and ``MAXIM_DATA_HOME/sim_reports/<id>/`` layout.

Output: append-only per-run JSONL consumed by ``scripts/analyze_exp41_exploration.py``.

Smoke test (CI-safe, no subprocess / no LLM):
    python scripts/benchmark_exp41_exploration.py --mock --trials 3 --out /tmp/x.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ARMS: tuple[str, ...] = ("A_cons", "B_cons", "A_dec", "B_dec")

# arm → (arc goal, exploration on?)
_ARM_ARC: dict[str, str] = {
    "A_cons": "cradle_prelinguistic",
    "B_cons": "cradle_prelinguistic",
    "A_dec": "cradle_prelinguistic_deceptive",
    "B_dec": "cradle_prelinguistic_deceptive",
}
_ARM_EXPLORE: dict[str, bool] = {"A_cons": False, "B_cons": True, "A_dec": False, "B_dec": True}
_ARM_DECEPTIVE: dict[str, bool] = {"A_cons": False, "B_cons": False, "A_dec": True, "B_dec": True}

DEFAULT_EXPLORE_WEIGHT = 1.5
DEFAULT_MODEL = "smollm-1.7b-instruct"  # generative-narrator LLM; AUT is LLM-free
DEFAULT_EMBODIMENT = "bodies/infant_humanoid_cold"
DEFAULT_MAX_TURNS = 18


# ── metric extraction (the harness owns rates; the analyzer owns hypotheses) ──


def _is_warm_self(tool: str) -> bool:
    return tool.endswith("_warm_self")


def _is_harm(tool: str) -> bool:
    # On the deceptive arc, warm_self AND touch on the hearth breach the arms'
    # thermal comfort band (pain). Both count as harmful engagement. NOTE: this
    # measures *engagement* (tool-name suffix), not the actual comfort-band
    # *failure* the doc §4 references — equivalent on the deceptive arc (every
    # warm_self/touch on the hearth harms), but a future non-deceptive harm
    # variant would need to read the executed-action outcome, not the name.
    return tool.endswith("_warm_self") or tool.endswith("_touch")


def _thirds_rates(tools: list[str], predicate) -> list[float]:
    """Bin the executed-action sequence into 3 equal thirds (by count) and
    return the predicate-rate in each third. Empty thirds → 0.0."""
    n = len(tools)
    if n == 0:
        return [0.0, 0.0, 0.0]
    t = n // 3
    # Distribute remainder into the last third so all actions are counted.
    bounds = [(0, t), (t, 2 * t), (2 * t, n)]
    rates: list[float] = []
    for lo, hi in bounds:
        chunk = tools[lo:hi]
        if not chunk:
            rates.append(0.0)
            continue
        rates.append(sum(1 for x in chunk if predicate(x)) / len(chunk))
    return rates


def compute_run_metrics(tools: list[str]) -> dict[str, Any]:
    return {
        "n_actions": len(tools),
        "harm_rate_thirds": _thirds_rates(tools, _is_harm),
        "warm_self_rate_thirds": _thirds_rates(tools, _is_warm_self),
    }


# ── sub-sim execution ────────────────────────────────────────────────────


def _resolve_maxim_binary() -> str:
    found = shutil.which("maxim")
    if found:
        return found
    # Fall back to the venv next to this checkout.
    here = Path(__file__).resolve().parent.parent
    cand = here / ".venv" / "bin" / "maxim"
    if cand.exists():
        return str(cand)
    return "maxim"


def _load_action_tools(data_home: Path) -> list[str]:
    """Read the newest session's actions.jsonl and return executed tool names.

    Mirrors benchmark_cross_session._load_latest_session: skip the Stage-0b
    header line, then collect per-action ``tool`` fields in order.
    """
    reports = data_home / "sim_reports"
    if not reports.exists():
        raise RuntimeError(f"sim_reports/ missing under {data_home}")
    sessions = sorted(reports.glob("*/actions.jsonl"), key=lambda p: p.stat().st_mtime)
    if not sessions:
        raise RuntimeError(f"no actions.jsonl under {reports}")
    tools: list[str] = []
    for line in sessions[-1].read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(rec, dict) or rec.get("_record_kind") == "header":
            continue
        tool = rec.get("tool") or (rec.get("action") or {}).get("tool_name")
        if tool:
            tools.append(str(tool))
    return tools


def _run_real(
    arm: str,
    seed: int,
    *,
    model: str,
    embodiment: str,
    max_turns: int,
    explore_weight: float,
    timeout_s: int,
    workdir: Path,
) -> list[str]:
    data_home = workdir / f"{arm}_seed{seed}"
    data_home.mkdir(parents=True, exist_ok=True)
    # Share the model cache so we don't re-download the small narrator GGUF.
    src_models = Path(os.path.expanduser("~/.maxim/models"))
    link = data_home / "models"
    if src_models.exists() and not link.exists():
        try:
            link.symlink_to(src_models)
        except OSError:
            pass

    env = os.environ.copy()
    env["MAXIM_DATA_HOME"] = str(data_home)
    env["MAXIM_LLM_PROFILE"] = model
    env["MAXIM_AUTO_SPAWN_LLM_SERVER"] = "0"
    env["MAXIM_LLM_CLOUD_ENABLED"] = "0"
    env["MAXIM_ROLE"] = "solo"
    env["MAXIM_SIM_SUBSTRATE_EXPLORE_BONUS_WEIGHT"] = str(explore_weight if _ARM_EXPLORE[arm] else 0.0)

    cmd = [
        _resolve_maxim_binary(),
        "--sim",
        _ARM_ARC[arm],
        "--aut-mode",
        "substrate-primary",
        "--embodiment",
        embodiment,
        "--interactive",
        "false",
        "--sim-max-turns",
        str(max_turns),
        "--seed",
        str(seed),
        "--research",
    ]
    log_dir = data_home / "harness_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    try:
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        (log_dir / "timeout.log").write_text(f"TIMEOUT {timeout_s}s\n{exc.stdout or ''}\n{exc.stderr or ''}")
        raise RuntimeError(f"{arm} seed={seed}: sub-sim timed out after {timeout_s}s") from exc
    (log_dir / "run.log").write_text((proc.stdout or "") + "\n---STDERR---\n" + (proc.stderr or ""))
    return _load_action_tools(data_home)


def _mock_tools(arm: str, seed: int) -> list[str]:
    """Synthesize a plausible action sequence per arm for CI smoke tests.

    Encodes the EXPECTED pattern so a mock fire yields a GRADUATE verdict:
      * A_dec: fixates on the harmful warm_self all session (no learning).
      * B_dec: tries warm_self early, then avoids it (switches to blanket).
      * A_cons / B_cons: keep doing the safe warm_self (correct prior held).
    A small seed-dependent jitter creates non-zero cross-seed SD.
    """
    j = seed % 3  # 0..2 jitter
    if arm == "A_dec":
        # harmful throughout
        return ["hearth_warm_self"] * 18
    if arm == "B_dec":
        # early harm, then safe blanket — within-session learning
        early = ["hearth_warm_self"] * (2 + j) + ["hearth_observe"] * (4 - j)
        late = ["blanket_wrap", "sense_hearth"] * 6
        return (early + late)[:18]
    if arm == "A_cons":
        return ["fire_pit_warm_self"] * 18
    # B_cons: explore once then settle on the (correct) safe warm
    return (["sense_fire_pit"] * (1 + j) + ["fire_pit_warm_self"] * (17 - j))[:18]


def _record(arm: str, seed: int, tools: list[str], *, mock: bool, git_hash: str) -> dict[str, Any]:
    rec = {
        "experiment": "exp41",
        "arm": arm,
        "arc": _ARM_ARC[arm],
        "deceptive": _ARM_DECEPTIVE[arm],
        "exploration": _ARM_EXPLORE[arm],
        "seed": seed,
        "git_hash": git_hash,
        "mock": mock,
    }
    rec.update(compute_run_metrics(tools))
    return rec


def _git_hash() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parent.parent,
            timeout=5,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _existing_keys(out_path: Path) -> set[tuple[str, int]]:
    if not out_path.exists():
        return set()
    keys: set[tuple[str, int]] = set()
    for line in out_path.read_text().splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(rec, dict) and "arm" in rec and "seed" in rec:
            keys.add((rec["arm"], int(rec["seed"])))
    return keys


def run_benchmark(
    *,
    arms: tuple[str, ...],
    trials: int,
    seed_base: int,
    out_path: Path,
    mock: bool,
    model: str,
    embodiment: str,
    max_turns: int,
    explore_weight: float,
    timeout_s: int,
    resume: bool,
) -> int:
    git_hash = _git_hash()
    done = _existing_keys(out_path) if resume else set()
    workdir = Path("data/sim_sandbox/exp41_runs")
    workdir.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_done = 0
    n_fail = 0
    with out_path.open("a") as out:
        for arm in arms:
            for trial in range(trials):
                seed = seed_base + trial
                if (arm, seed) in done:
                    print(f"skip {arm} seed={seed} (already recorded)")
                    continue
                t0 = time.time()
                try:
                    if mock:
                        tools = _mock_tools(arm, seed)
                    else:
                        tools = _run_real(
                            arm,
                            seed,
                            model=model,
                            embodiment=embodiment,
                            max_turns=max_turns,
                            explore_weight=explore_weight,
                            timeout_s=timeout_s,
                            workdir=workdir,
                        )
                except Exception as exc:  # noqa: BLE001 - log + continue per run
                    n_fail += 1
                    print(f"FAIL {arm} seed={seed}: {exc}", file=sys.stderr)
                    continue
                rec = _record(arm, seed, tools, mock=mock, git_hash=git_hash)
                out.write(json.dumps(rec) + "\n")
                out.flush()
                n_done += 1
                print(
                    f"ok {arm} seed={seed} n_actions={rec['n_actions']} "
                    f"harm_thirds={[round(x, 2) for x in rec['harm_rate_thirds']]} "
                    f"({time.time() - t0:.1f}s)"
                )

    print(f"\ndone: {n_done} runs recorded, {n_fail} failed → {out_path}")
    return 0 if n_fail == 0 else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Exp 41 substrate-primary exploration harness (2×2)")
    p.add_argument("--arms", default=",".join(ARMS), help="comma-separated subset of A_cons,B_cons,A_dec,B_dec")
    p.add_argument("--trials", type=int, default=10, help="seeds per arm")
    p.add_argument("--seed-base", type=int, default=42)
    p.add_argument("--out", type=Path, required=True, help="append-only per-run JSONL")
    p.add_argument(
        "--allow-dirty",
        action="store_true",
        help="write a GATED record (docs/experiments/data/) from a dirty src/scripts tree; stamps allow_dirty: true "
        "into every record (default: refuse, exit 3 — docs/lessons/experiment-prereg-precedes-data.md)",
    )
    p.add_argument("--mock", action="store_true", help="synthesize runs (CI-safe; no subprocess/LLM)")
    p.add_argument("--model", default=DEFAULT_MODEL, help="generative-narrator LLM profile (AUT is LLM-free)")
    p.add_argument("--embodiment", default=DEFAULT_EMBODIMENT)
    p.add_argument("--sim-max-turns", type=int, default=DEFAULT_MAX_TURNS)
    p.add_argument("--explore-weight", type=float, default=DEFAULT_EXPLORE_WEIGHT)
    p.add_argument("--timeout-s", type=int, default=1800)
    p.add_argument("--resume", action="store_true", help="skip (arm, seed) pairs already in --out")
    args = p.parse_args(argv)

    # Provenance guard — refuse to run if the sub-sims would import a `maxim`
    # from outside this repo (scripts/_provenance.py; Exp 42b post-mortem).
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _provenance import ProvenanceError, assert_repo_interpreter, preflight_gated_record_or_exit

    try:
        assert_repo_interpreter(Path(__file__).resolve().parent.parent, _resolve_maxim_binary(), exempt=args.mock)
    except ProvenanceError as exc:
        print(f"PREFLIGHT FAIL: {exc}", file=sys.stderr)
        return 3
    preflight_gated_record_or_exit(Path(__file__).resolve().parent.parent, args.out, allow_dirty=args.allow_dirty)

    arms = tuple(a.strip() for a in args.arms.split(",") if a.strip())
    bad = [a for a in arms if a not in ARMS]
    if bad:
        print(f"error: unknown arms {bad}; valid: {ARMS}", file=sys.stderr)
        return 2

    return run_benchmark(
        arms=arms,
        trials=args.trials,
        seed_base=args.seed_base,
        out_path=args.out,
        mock=args.mock,
        model=args.model,
        embodiment=args.embodiment,
        max_turns=args.sim_max_turns,
        explore_weight=args.explore_weight,
        timeout_s=args.timeout_s,
        resume=args.resume,
    )


if __name__ == "__main__":
    raise SystemExit(main())
