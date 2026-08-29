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
import time
from pathlib import Path
from typing import Any

ARC = "cradle_mother"
EMBODIMENT = "bodies/infant_operant"  # hunger drive + DRIVELESS azimuth (no intrinsic orient)
ARMS = ("taught", "no_feed", "satiated")
# Exp 52 (Nurture): the satiated arm is a BODY variant, not an env flag — never
# hungry, so the mother's contingent feed relieves nothing and (under
# relief-sourced credit) mints no reward. Visible in provenance as the body ref.
# Keyed on the CHOSEN --embodiment (Exp 54 runs the same three arms on the
# robot's own nursery body, bodies/reachy_mini_infant); --satiated-embodiment
# overrides, and an embodiment with no mapping refuses the satiated arm loudly
# rather than guessing a body name.
_SATIATED_EMBODIMENT: dict[str, str] = {
    "bodies/infant_operant": "bodies/infant_operant_satiated",
    "bodies/reachy_mini_infant": "bodies/reachy_mini_infant_satiated",
}


def _arm_embodiment(arm: str, embodiment: str, satiated_embodiment: str | None = None) -> str:
    """The body ref an arm runs on: the satiated arm's never-hungry variant of the
    chosen embodiment; every other arm the embodiment itself."""
    if arm != "satiated":
        return embodiment
    body = satiated_embodiment or _SATIATED_EMBODIMENT.get(embodiment)
    if not body:
        raise ValueError(
            f"no satiated body known for {embodiment!r}; pass --satiated-embodiment <ref> "
            f"(known: {sorted(_SATIATED_EMBODIMENT)})"
        )
    return body


# Shared conditions for ALL arms:
#   MAXIM_OPERANT_ONLY_CREDIT — the tool-success floor never drowns the operant
#     signal (probe 3 tool_floor arm).
#   MAXIM_SIM_SUBSTRATE_EXPLORE_BONUS_WEIGHT — the infant must EXPLORE turns to
#     bootstrap (no intrinsic drive, cold-start bias). Without it it never turns
#     and stays inert (the original review's B1, confirmed in the operant smoke).
#   MAXIM_SUBSTRATE_TOOL_WHITELIST — restrict the infant's repertoire to the two
#     turn actions. Without it, generic always-succeed tools (sense_presence,
#     causal_pos 0.99) out-compete the turns and the infant never orients (found
#     in the mac-mini sweep). ``listen`` is deliberately EXCLUDED: in substrate-
#     primary the infant already perceives azimuth via the sensor encoding, so
#     listen is a no-op that scores progress=0 and only dilutes directedness.
#     The task is a clean 2-alternative motor choice (turn toward the sound).
_SHARED_ENV = {
    "MAXIM_OPERANT_ONLY_CREDIT": "1",
    "MAXIM_SIM_SUBSTRATE_EXPLORE_BONUS_WEIGHT": "1.5",
    "MAXIM_SUBSTRATE_TOOL_WHITELIST": "turn_left,turn_right",
}
# Per-arm environment (the arc is identical; the arm is selected by env toggles).
_ARM_ENV: dict[str, dict[str, str]] = {
    # Full operant teaching: mother shapes (feeds + credits toward-turns).
    "taught": {**_SHARED_ENV},
    # Control: mother still PLACES the sound but never feeds/credits — with no
    # intrinsic orient drive the infant has NO teacher and stays at chance.
    "no_feed": {**_SHARED_ENV, "MAXIM_CRADLE_MOTHER_DISABLE_CARE": "1"},
    # Exp 52: same contingency and feed as taught; the BODY is never hungry.
    "satiated": {**_SHARED_ENV},
}


def _acquire_single_runner_lock(lock_path: Path) -> bool:
    """Take the single-runner lock, or refuse loudly. Ported from
    scripts/exp44/campaign.py::acquire_campaign_lock (#494).

    The 2026-08-13 Exp 48 re-baseline was contaminated by concurrent
    harness instances sharing one workdir + out file: duplicate
    (arm, seed) rows, per-act `turns` inflating 12 -> 24 -> 36 (each
    instance parsing the MERGED MAXIM_LOG_FILE of the shared run dirs),
    and byte-identical curves across "different" seeds. The launch
    ergonomics practically guarantee the double-launch (nohup +
    block-buffered stdout makes a fresh launch LOOK dead, so the
    operator relaunches), so structure has to make it safe — the same
    push-into-structure rule as the silent-no-op invariants.

    O_CREAT|O_EXCL is the atomic take; the lock records the holder pid.
    A lock whose holder is dead is stale (crash/SIGKILL) and is retaken
    automatically. Released via atexit.
    """
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            break
        except FileExistsError:
            try:
                holder = int(lock_path.read_text().strip() or "0")
            except (OSError, ValueError):
                holder = 0
            alive = False
            if holder > 0:
                try:
                    os.kill(holder, 0)
                    alive = True
                except ProcessLookupError:
                    alive = False
                except PermissionError:
                    alive = True  # pid exists, owned by someone else
            if alive:
                print(
                    f"ERROR: another harness (pid {holder}) already holds {lock_path} — "
                    "exactly one runner per workdir/out. To watch progress, tail the "
                    "log; do NOT relaunch.",
                    file=sys.stderr,
                )
                return False
            lock_path.unlink(missing_ok=True)  # stale (holder dead) — retake

    import atexit

    atexit.register(lambda: lock_path.unlink(missing_ok=True))
    return True


def _resolve_maxim() -> list[str]:
    """Invoke maxim as a module so the caller's PYTHONPATH=src worktree wins."""
    return [sys.executable, "-m", "maxim"]


def _narrator_preflight(model: str) -> str | None:
    """Verify the narrator profile resolves to a loadable local model — with
    the SUB-SIM's env view — before spawning run 1. Returns an error string
    or None.

    S3 (assert your own health), earned 2026-08-13: THREE campaigns completed
    'ok' with a dead narrator — the in-process load failed (absent/mis-resolved
    GGUF, config-layer profile override), the router fell back to
    _llm_unavailable, and every run produced plausible-looking metronomic
    ~712s results at \\$0. The probe runs in a subprocess with the same env
    the sub-sims inherit, resolves the profile through the REAL
    load_llm_config path, and requires the resolved GGUF to exist and
    llama_cpp to import. It prints the resolved profile + path either way,
    so a config-layer hijack (env says mistral, sub-sim loads qwen) is
    visible at launch, not at 3am.
    """
    probe = (
        "import os, sys\n"
        "from maxim.models.language.config import load_llm_config\n"
        "cfg = load_llm_config()\n"
        "sys.stderr.write(f'narrator resolves to profile={cfg.profile} path={cfg.model_path}\\n')\n"
        "if cfg.backend != 'llama_cpp':\n"
        "    sys.stderr.write(f'ERROR: backend={cfg.backend} — this harness expects a local llama_cpp narrator\\n')\n"
        "    sys.exit(3)\n"
        "if not cfg.model_path or not os.path.isfile(cfg.model_path):\n"
        "    sys.stderr.write(f'ERROR: resolved model file does not exist: {cfg.model_path!r}\\n')\n"
        "    sys.exit(3)\n"
        "try:\n"
        "    import llama_cpp  # noqa: F401\n"
        "except Exception as e:\n"
        "    sys.stderr.write(f'ERROR: llama_cpp import failed: {e}\\n')\n"
        "    sys.exit(3)\n"
    )
    env = os.environ.copy()
    env["MAXIM_LLM_PROFILE"] = model
    env["MAXIM_ROLE"] = "solo"
    try:
        r = subprocess.run([sys.executable, "-c", probe], env=env, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return "narrator preflight timed out (120s)"
    sys.stderr.write(r.stderr)
    if r.returncode != 0:
        return f"narrator preflight failed (exit {r.returncode}) — fix the resolution above before burning a campaign"
    return None


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
        # Exp 52 S3 in-sim assertions (analyzer --gate v3 reads these):
        neg_reward = sum(1 for r in recs if isinstance(r.get("reward"), (int, float)) and r["reward"] < 0)
        credited_no_relief = sum(
            1
            for r in recs
            if r.get("credited") and (not isinstance(r.get("relief"), (int, float)) or abs(r["relief"]) <= 1e-9)
        )
        # directedness: of the turns where progress was measurable, how many moved
        # the infant toward the sound (progress > 0)? Arm-independent.
        measured = [r for r in recs if isinstance(r.get("progress"), (int, float))]
        directed = sum(1 for r in measured if r["progress"] > 0)
        # Observed (not intended) apparatus stamp: the credit mode the sub-sim
        # actually logged this act — what ran, not what the CLI asked for.
        modes = [str(r["credit"]) for r in recs if r.get("credit") not in (None, "")]
        credit_observed = max(set(modes), key=modes.count) if modes else None
        result[act] = {
            "turns": n,
            "credit_observed": credit_observed,
            "directedness": directed / len(measured) if measured else 0.0,
            "measured": len(measured),
            "fed_rate": fed / n if n else 0.0,
            "credited_rate": credited / n if n else 0.0,
            "neg_reward": neg_reward,
            "credited_no_relief": credited_no_relief,
        }
    return result


# ── one sub-sim run ─────────────────────────────────────────────────────────


def _run_one(
    arm: str,
    seed: int,
    *,
    model: str,
    max_turns: int,
    timeout_s: int,
    workdir: Path,
    explore_weight: float,
    stimulus_order: str = "cycle",
    credit: str = "relief",
    embodiment: str = EMBODIMENT,
) -> dict[str, Any]:
    data_home = workdir / f"{arm}_seed{seed}_ew{explore_weight}"
    # ALWAYS a fresh sandbox (2026-08-13 contamination post-mortem): reusing a
    # prior attempt's dir poisons the run through TWO channels — MAXIM_LOG_FILE
    # appends, so the fade parse reads the MERGED telemetry of every attempt
    # (turns 12 -> 24 -> 36); and MAXIM_DATA_HOME persists the substrate, so a
    # re-run RESUMES the prior attempt's NAc (#446 cross-session persistence)
    # and the "infant" starts pre-trained. Reaching here with an existing dir
    # means a prior attempt never recorded its row (crash/timeout/kill) — a
    # clean retry is the only valid semantics.
    if data_home.exists():
        import shutil

        shutil.rmtree(data_home)
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
    # CLI override wins over the arm default (explore-weight sweep).
    env["MAXIM_SIM_SUBSTRATE_EXPLORE_BONUS_WEIGHT"] = str(explore_weight)
    env["MAXIM_CRADLE_MOTHER_STIMULUS_ORDER"] = stimulus_order
    # Exp 52 credit-value source (S6 fidelity toggle; identical across arms).
    env["MAXIM_CRADLE_MOTHER_CREDIT"] = credit

    cmd = _resolve_maxim() + [
        "--sim",
        ARC,
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
    p.add_argument("--arms", default=",".join(ARMS), help="comma-separated: taught,no_feed,satiated")
    p.add_argument("--trials", type=int, default=6)
    p.add_argument("--seed-base", type=int, default=42)
    p.add_argument(
        "--sim-max-turns", type=int, default=56, help="4 acts × 12 + margin (payoff act must get its full sample)"
    )
    p.add_argument("--model", default="mistral-7b", help="narrator profile (prose-less arc → light use)")
    p.add_argument("--timeout-s", type=int, default=1800)
    p.add_argument("--out", required=True)
    p.add_argument(
        "--allow-dirty",
        action="store_true",
        help="write a GATED record (docs/experiments/data/) from a dirty src/scripts tree; stamps allow_dirty: true "
        "into every record (default: refuse, exit 3 — docs/lessons/experiment-prereg-precedes-data.md)",
    )
    # Durable default per apparatus standard S4: Exp 48's graduation
    # originals lived in /tmp and are permanently gone (macOS cleared it
    # again mid-investigation). Never default an experiment workdir there.
    p.add_argument("--workdir", default=os.path.expanduser("~/.maxim/experiments/cradle_mother_runs"))
    p.add_argument(
        "--explore-weight",
        type=float,
        default=1.5,
        help="MAXIM_SIM_SUBSTRATE_EXPLORE_BONUS_WEIGHT (sweepable). 1.5 bootstraps but may cap the ceiling; "
        "lower exploits the (small ~2-cluster) orient policy sooner.",
    )
    p.add_argument(
        "--stimulus-order",
        choices=("cycle", "shuffled"),
        default="cycle",
        help="Mother stimulus order. 'cycle' = the v1/v2 apparatus (deterministic replay — "
        "phase-locks against a greedy agent; Exp 48 sweep finding). 'shuffled' = seeded "
        "per-block permutation (exposure-balanced, phase-lock broken) — the apparatus-v3 "
        "setting. S6: an apparatus change, stamped per record.",
    )
    p.add_argument(
        "--credit",
        default="relief",
        choices=["relief", "constant"],
        help="operant credit VALUE source (Exp 52): relief = sign of the infant's drive relief (default); constant = pre-Exp-52 by-fiat feed_reward (A/B)",
    )
    p.add_argument(
        "--embodiment",
        default=EMBODIMENT,
        help="body ref for the taught/no_feed arms (Exp 52: bodies/infant_operant; Exp 54: bodies/reachy_mini_infant). "
        "The satiated arm runs the never-hungry variant keyed on this ref (see --satiated-embodiment).",
    )
    p.add_argument(
        "--satiated-embodiment",
        default=None,
        help="body ref for the satiated arm (default: the never-hungry variant of --embodiment; "
        "required when --embodiment has no known variant)",
    )
    p.add_argument("--mock", action="store_true", help="synthetic fade (CI smoke, no subprocess)")
    p.add_argument("--resume", action="store_true", help="skip (arm,seed,explore_weight) already in --out")
    args = p.parse_args()
    # Assert the apparatus before measuring (review fold, BLOCKER): _run_one
    # overwrites the sub-sim env from the CLI flags, so an operator who exported
    # MAXIM_CRADLE_MOTHER_STIMULUS_ORDER=shuffled (or _CREDIT) and forgot the
    # flag would run 12 h on the wrong apparatus while believing otherwise.
    # A disagreement between the ambient env and the flag is refused loudly.
    for _var, _flag, _val in (
        ("MAXIM_CRADLE_MOTHER_STIMULUS_ORDER", "--stimulus-order", args.stimulus_order),
        ("MAXIM_CRADLE_MOTHER_CREDIT", "--credit", args.credit),
    ):
        _amb = os.environ.get(_var, "").strip().lower()
        if _amb and _amb != str(_val).lower():
            print(
                f"[FAIL] ambient {_var}={_amb!r} disagrees with {_flag} {_val!r}; the flag is what the "
                "sub-sims will see. Unset the variable or pass the matching flag (exit 3).",
                file=sys.stderr,
            )
            return 3

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    for a in arms:
        if a not in _ARM_ENV:
            print(f"unknown arm {a!r}; valid: {list(_ARM_ENV)}", file=sys.stderr)
            return 2
    # Resolve every arm's body BEFORE the campaign and instantiate each once: a
    # typo or a missing satiated variant fails in seconds here, not after the
    # first 12-minute sub-sim (and the body ref is stamped into every row).
    try:
        arm_bodies = {a: _arm_embodiment(a, args.embodiment, args.satiated_embodiment) for a in arms}
    except ValueError as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 2
    try:
        from maxim.embodiment.component_registry import ComponentRegistry

        _registry = ComponentRegistry()
        for _ref in sorted(set(arm_bodies.values())):
            _registry.instantiate(_ref)
    except Exception as exc:  # noqa: BLE001 — any body that cannot build is a preflight failure
        print(f"[FAIL] embodiment preflight: {exc}", file=sys.stderr)
        return 2

    # Provenance preflight (Exp 42b lesson — MANDATORY for any harness that
    # spawns sub-sims): the `maxim` the sub-sims import must be THIS repo.
    # This harness spawns `[sys.executable, "-m", "maxim"]`, so the probe
    # interpreter is sys.executable itself.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _provenance import (
        ProvenanceError,
        assert_repo_interpreter,
        executed_code_provenance,
        preflight_gated_record_or_exit,
    )

    repo_root = Path(__file__).resolve().parent.parent
    provenance: dict[str, str] = {}
    try:
        assert_repo_interpreter(repo_root, sys.executable, exempt=args.mock)
    except ProvenanceError as exc:
        print(f"PREFLIGHT FAIL: {exc}", file=sys.stderr)
        return 3
    # Gated-record refusal (roadmap 1.1.x item 16.7, both harness families): a record
    # under docs/experiments/data/ from a dirty src/scripts tree exits 3 unless
    # --allow-dirty, which stamps allow_dirty: true into every record.
    preflight_gated_record_or_exit(repo_root, args.out, allow_dirty=args.allow_dirty)
    if not args.mock:
        provenance = executed_code_provenance(
            repo_root, sys.executable, out_path=args.out, allow_dirty=args.allow_dirty
        )
        err = _narrator_preflight(args.model)
        if err is not None:
            print(f"PREFLIGHT FAIL: {err}", file=sys.stderr)
            return 3

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # A second campaign appending to an existing out file contaminates it
    # (duplicate (arm, seed) rows the analyzer pools as extra trials).
    # Continuing a campaign is what --resume is for.
    if out_path.exists() and out_path.stat().st_size > 0 and not args.resume:
        print(
            f"ERROR: {out_path} already has rows — appending a second campaign "
            "contaminates it. Use --resume to continue it, or a new --out path.",
            file=sys.stderr,
        )
        return 6
    done: set[tuple[str, int, float, str, str]] = set()
    if args.resume and out_path.exists():
        for line in out_path.read_text().splitlines():
            try:
                r = json.loads(line)
                # Rows without a credit stamp predate Exp 52 = constant credit.
                # Rows without an embodiment stamp predate the flag = the infant body.
                done.add(
                    (
                        r["arm"],
                        r["seed"],
                        float(r.get("explore_weight", 1.5)),
                        r.get("credit", "constant"),
                        r.get("embodiment", EMBODIMENT),
                    )
                )
            except (json.JSONDecodeError, KeyError):
                pass

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    # Single-runner locks on BOTH shared surfaces (S8): the workdir (shared
    # run sandboxes/logs) and the out file (the append surface). A relaunch
    # becomes a loud refusal naming the holder pid instead of a silent
    # contamination.
    if not _acquire_single_runner_lock(workdir / "harness.lock"):
        return 6
    if not _acquire_single_runner_lock(Path(str(out_path) + ".lock")):
        return 6
    # Flushed immediately: under nohup with redirected stdout, Python
    # block-buffers and the log looks dead for the first ~25-min run —
    # which is precisely what tempts the operator into relaunching.
    print(
        f"harness pid {os.getpid()} holding locks — arms={arms} bodies={arm_bodies} trials={args.trials} "
        f"ew={args.explore_weight} -> {out_path} (first per-run line in ~20-45 min)",
        flush=True,
    )
    n_ok = n_fail = 0
    import time as _t

    with out_path.open("a") as fh:
        for arm in arms:
            for trial in range(args.trials):
                seed = args.seed_base + trial
                if (arm, seed, float(args.explore_weight), args.credit, arm_bodies[arm]) in done:
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
                            explore_weight=args.explore_weight,
                            stimulus_order=args.stimulus_order,
                            credit=args.credit,
                            embodiment=arm_bodies[arm],
                        )
                    )
                    # S3 (assert your own health): a truncated run (sub-sim
                    # died early) must be a FAIL, not an ok row with missing
                    # acts — the analyzer pools per act, so a partial row
                    # silently skews the bins it does have.
                    _acts = ("act1_early", "act2_warming", "act3_consolidating", "act4_autonomous")
                    _missing = [a for a in _acts if a not in fade]
                    if _missing:
                        raise RuntimeError(
                            f"incomplete fade — sub-sim ended early, missing acts {_missing}; not recording a partial row"
                        )
                    rec = {
                        "experiment": "cradle_mother",
                        "arm": arm,
                        "seed": seed,
                        "explore_weight": args.explore_weight,
                        # S6 apparatus stamp (review fold, BLOCKING): the
                        # substrate action budget the sub-sim SAW — _run_one
                        # copies os.environ, so an operator-shell value flows
                        # into every sub-sim; without this stamp two sweeps
                        # under different regimes produce indistinguishable
                        # JSONLs (the Exp 42b self-auditing-artifact rule).
                        # Raw env string (or null): records what the child
                        # inherited, not an interpretation of it.
                        "substrate_actions_per_turn_env": os.environ.get("MAXIM_SUBSTRATE_ACTIONS_PER_TURN"),
                        # S6 stamp: which stimulus-order apparatus produced this
                        # row ("cycle" = phase-lockable v1/v2; "shuffled" = v3).
                        "stimulus_order": args.stimulus_order,
                        # S6 stamp (Exp 52): where the operant credit's VALUE came
                        # from — "relief" (sign of the infant's drive relief) or
                        # "constant" (the pre-Exp-52 by-fiat feed_reward).
                        "credit": args.credit,
                        "embodiment": arm_bodies[arm],
                        "mock": args.mock,
                        "ts": round(time.time(), 3),  # first-write time, for lint_prereg_precedes_data
                        "git_hash": _git_hash(),
                        # Exp 42b self-auditing-artifact rule: harness hash
                        # describes where the harness LIVES; these describe
                        # the code the sub-sims IMPORTED.
                        **provenance,
                        "fade": fade,
                    }
                    fh.write(json.dumps(rec) + "\n")
                    fh.flush()
                    n_ok += 1
                    acts_summary = " ".join(
                        f"{a.split('_')[0]}:{v['directedness']:.2f}" for a, v in sorted(fade.items())
                    )
                    dt = (_t.monotonic() - t0) if hasattr(_t, "monotonic") else 0
                    print(f"ok {arm} seed={seed} directed[{acts_summary}] ({dt:.0f}s)", flush=True)
                except Exception as e:  # noqa: BLE001 — one run's failure must not kill the sweep
                    n_fail += 1
                    print(f"FAIL {arm} seed={seed}: {e}", file=sys.stderr)

    print(f"\ndone: {n_ok} runs recorded, {n_fail} failed → {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
