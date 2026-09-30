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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _provenance  # noqa: E402  (scripts/_provenance.py, loaded from THIS tree)

# Row schema. 2: rows carry record_kind/status, the nested `provenance` block, and `sims` (the echoed
# evidence of the sim report the run wrote); failed runs are rows with status "failed" (M1b, #1003).
HARNESS_SCHEMA = 2

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


def _load_action_tools(session_dir: Path) -> list[str]:
    """Read THE session's actions.jsonl (the one this run's spawn wrote, found by run id) and return
    executed tool names: skip the Stage-0b header line, then collect per-action ``tool`` fields in order."""
    actions = session_dir / "actions.jsonl"
    if not actions.exists():
        raise RuntimeError(f"no actions.jsonl in {session_dir}")
    tools: list[str] = []
    for line in actions.read_text().splitlines():
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
) -> tuple[list[str], dict[str, Any]]:
    """One sub-sim in a FRESH home. Returns the executed tools and the echoed evidence of the report it
    wrote; a run that cannot be established raises (``_provenance.SimRunFailed``), and becomes a failed row."""
    # Fresh per run (#1003; the Exp 42 / cradle rule): MAXIM_DATA_HOME persists the substrate (#446), so a
    # reused home would resume an earlier launch's NAc under possibly different code.
    data_home = workdir / f"{arm}_seed{seed}"
    if data_home.exists():
        shutil.rmtree(data_home)
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
    run_id = _provenance.harness_run_id()
    env["MAXIM_HARNESS_RUN_ID"] = run_id

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
    before = _provenance.list_sessions(data_home)
    if before:  # the home was just wiped: an inherited session would be state this run did not declare
        raise _provenance.SimRunFailed(f"fresh home {data_home} already holds sessions {sorted(before)}", sims=[])
    try:
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        (log_dir / "timeout.log").write_text(f"TIMEOUT {timeout_s}s\n{exc.stdout or ''}\n{exc.stderr or ''}")
        raise RuntimeError(f"{arm} seed={seed}: sub-sim timed out after {timeout_s}s") from exc
    (log_dir / "run.log").write_text((proc.stdout or "") + "\n---STDERR---\n" + (proc.stderr or ""))
    session_dir, report = _provenance.spawn_evidence(data_home, run_id, before, returncode=proc.returncode)
    evidence = _provenance.sim_evidence(session_dir, report)
    try:
        return _load_action_tools(session_dir), evidence
    except Exception as exc:  # the report was found: the failed row names the session that ran
        raise _provenance.SimRunFailed(f"{type(exc).__name__}: {exc}", sims=[evidence]) from exc


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


def _base(arm: str, seed: int, *, mock: bool, git_hash: str) -> dict[str, Any]:
    return {
        "record_kind": "harness_row",
        "harness_schema": HARNESS_SCHEMA,
        "experiment": "exp41",
        "arm": arm,
        "arc": _ARM_ARC[arm],
        "deceptive": _ARM_DECEPTIVE[arm],
        "exploration": _ARM_EXPLORE[arm],
        "seed": seed,
        "git_hash": git_hash,
        "mock": mock,
        "provenance": dict(_PROVENANCE),
    }


def _record(
    arm: str, seed: int, tools: list[str], *, mock: bool, git_hash: str, sims: list[dict[str, Any]]
) -> dict[str, Any]:
    rec = {**_base(arm, seed, mock=mock, git_hash=git_hash), "status": "ok", "sims": sims, "depends_on": []}
    rec.update(compute_run_metrics(tools))
    return rec


# The harness's own provenance block (executed_code_provenance: code, dirty flag, allowance, run id), set in
# main() and stamped into every row.
_PROVENANCE: dict[str, Any] = {}


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
        # A failed row is not done: a resume re-runs it (legacy rows carry no status and count as done).
        if isinstance(rec, dict) and "arm" in rec and "seed" in rec and not _provenance.is_failed_row(rec):
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
                        tools, sims = _mock_tools(arm, seed), []
                    else:
                        tools, evidence = _run_real(
                            arm,
                            seed,
                            model=model,
                            embodiment=embodiment,
                            max_turns=max_turns,
                            explore_weight=explore_weight,
                            timeout_s=timeout_s,
                            workdir=workdir,
                        )
                        sims = [evidence]
                except Exception as exc:  # noqa: BLE001 - recorded as a failed row, then continue
                    n_fail += 1
                    print(f"FAIL {arm} seed={seed}: {exc}", file=sys.stderr)
                    failed = {**_base(arm, seed, mock=mock, git_hash=git_hash), **_provenance.failed_row(exc)}
                    out.write(json.dumps(failed) + "\n")
                    out.flush()
                    continue
                rec = _record(arm, seed, tools, mock=mock, git_hash=git_hash, sims=sims)
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
    repo_root = Path(__file__).resolve().parent.parent
    try:
        _provenance.assert_repo_interpreter(repo_root, _resolve_maxim_binary(), exempt=args.mock)
    except _provenance.ProvenanceError as exc:
        print(f"PREFLIGHT FAIL: {exc}", file=sys.stderr)
        return 3
    _provenance.harness_run_id()  # minted at start, mock runs included
    try:
        # Refuses a dirty tree for a gated --out unless --allow-dirty, which it then stamps (#1003: before, the
        # preflight's result was discarded and no row said which code ran).
        _PROVENANCE.update(
            _provenance.executed_code_provenance(
                repo_root, _resolve_maxim_binary(), out_path=args.out, allow_dirty=args.allow_dirty
            )
        )
    except _provenance.DirtyTreeError as exc:
        print(f"PREFLIGHT FAIL: {exc}", file=sys.stderr)
        return 3

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
