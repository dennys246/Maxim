"""Exp 44b campaign runner — grow N on the substrate-counterfactual with cached stages.

Orchestrates the full Exp 44 pipeline per (arm × seed), sequentially:

  1. LEARN    substrate-primary run on the arm's arc (no LLM in the action path)
              → verified learned substrate (aut_nac.json, min-bias gate).
  2. CAPTURE  llm-primary run resuming the learned substrate with the decay-tau
              hold + MAXIM_EXP44_CAPTURE_LOG → paired full/ablated prompts.
  3. REQUERY  offline temp-0 re-query of both prompt variants per decision
              (rerun_ablated_offline.py), once per requery model. Cached by
              (capture-content-hash, model) — re-analysis and cross-model
              sweeps never re-pay compute for unchanged captures.
  4. STATS    stats_counterfactual.py over the whole campaign directory.

Every stage is resumable: outputs on disk (with verification markers) are
skipped, so a killed campaign continues where it stopped. Every sub-sim runs in
its own MAXIM_DATA_HOME under the campaign directory (the Exp 42 harness
isolation pattern), so nothing touches the operator's ~/.maxim and campaigns
can run on a second machine while other work continues.

Provenance (Exp 42b lesson — MANDATORY): assert_repo_interpreter() runs before
the first sub-sim (exit 3 on mismatch) and executed_code_provenance() is
stamped into the campaign manifest and every stage record.

Usage::

    export PYTHONPATH="$PWD/src"          # absolute, own line (Exp 42b lesson)
    python scripts/exp44/campaign.py --config scripts/exp44/campaign_44b.json \
        --workdir data/exp44b/run1 [--dry-run] [--arms A,B] [--seeds 1,2,3]

Config schema — see scripts/exp44/campaign_44b.json (the pre-registered default).

Arm ``substrate`` values:
  "learn"            — stage 1 runs and its session is resumed by stage 2.
  "transplant:<arm>" — no learn stage; the SOURCE arm's same-seed learned
                       session is copied in (wrong-content control: a substrate
                       whose content mismatches this arm's world).
  "none"             — no substrate; capture runs without --resume-sim.
                       (Baseline color preference is ALSO measurable for free
                       from any arm's ablated re-queries; a "none" arm is only
                       needed if you want it without any resume machinery.)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[2]
sys.path.insert(0, str(_HERE.parent))  # sibling scripts
sys.path.insert(0, str(_REPO / "scripts"))  # _provenance
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from _provenance import assert_repo_interpreter, executed_code_provenance  # noqa: E402

MIN_BIAS_DEFAULT = 0.9
MIN_CAPTURE_PAIRS = 5
LEARN_TIMEOUT_S = 7200  # runbook lesson: 1800s default killed healthy 56-turn runs
CAPTURE_TIMEOUT_S = 7200


# ── small utilities ──────────────────────────────────────────────────────────


def _sha16(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _append_manifest(campaign_dir: Path, record: dict[str, Any]) -> None:
    record = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), **record}
    with open(campaign_dir / "manifest.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def _resolve_maxim_binary() -> str:
    found = shutil.which("maxim")
    if not found:
        print("ERROR: `maxim` not on PATH (activate the venv with the editable install)", file=sys.stderr)
        raise SystemExit(2)
    return found


def _max_abs_cluster_bias(nac_json: Path) -> float:
    """Best-effort max |value| under any cluster_reward_bias subtree.

    Matches NAc.dump()'s flat ``"cluster_reward_bias": {key: float}`` map. The
    recursive substring scan tolerates shape evolution, but note the failure
    DIRECTION is asymmetric: a renamed field degrades loudly (0.0 → gate
    fails), while a persisted numeric CONFIG field whose name contains the
    substring (e.g. a future ``cluster_reward_bias_decay_tau``) would pass the
    gate silently. NAc.dump() persists no config today; if that changes,
    tighten this to exact-key matching. The gate is deliberately sign- and
    cluster-blind (anti-forking: no direction-conditioned exclusions).
    """
    try:
        data = json.loads(nac_json.read_text())
    except (OSError, json.JSONDecodeError):
        return 0.0

    best = 0.0

    def walk(node: Any, under_bias: bool) -> None:
        nonlocal best
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, under_bias or "cluster_reward_bias" in str(k))
        elif isinstance(node, list):
            for v in node:
                walk(v, under_bias)
        elif under_bias and isinstance(node, (int, float)) and not isinstance(node, bool):
            best = max(best, abs(float(node)))

    walk(data, False)
    return best


# Experiment toggles that must NEVER leak from the operator's shell into a
# campaign sub-sim (two-lens fold, cross-confirmed): one stray
# MAXIM_DISABLE_CLUSTER_BIAS_ANNOTATION=1 silently produces full==ablated
# prompts -> a clean-looking confirmatory NULL. Every toggle a stage needs is
# set EXPLICITLY in that stage's `extra` dict; everything else is scrubbed.
_SCRUBBED_ENV_VARS = (
    "MAXIM_DISABLE_CLUSTER_BIAS_ANNOTATION",
    "MAXIM_DISABLE_VARIANCE_ANNOTATION",
    "MAXIM_NAC_REWARD_BIAS_DISABLED",
    "MAXIM_ENABLE_BODY_STATE_PROMPT",
    "MAXIM_DISABLE_COACH_BODY_LAYERS",
    "MAXIM_SUBSTRATE_TOOL_WHITELIST",
    "MAXIM_OPERANT_ONLY_CREDIT",
    "MAXIM_CRADLE_MOTHER_DISABLE_CARE",
    "MAXIM_NAC_MIN_CONFIDENCE",
    "MAXIM_EXP44_CAPTURE_LOG",
    "MAXIM_DETERMINISTIC_SCENE_EMBODIMENT",
    "MAXIM_NAC_CLUSTER_REWARD_BIAS_DECAY_TAU",
    "MAXIM_SIM_SUBSTRATE_EXPLORE_BONUS_WEIGHT",
    "MAXIM_SIM_DRIVE_GATE_ENABLED",
    "MAXIM_DISABLE_IMAGINATION",
    "MAXIM_DISABLE_IMAGINATION_SUBSTRATE_SIGNAL",
)


def _sub_env(data_home: Path, *, profile: str | None, extra: dict[str, str]) -> dict[str, str]:
    env = os.environ.copy()
    for var in _SCRUBBED_ENV_VARS:
        env.pop(var, None)
    env["MAXIM_DATA_HOME"] = str(data_home)
    env["MAXIM_ROLE"] = "solo"
    env["MAXIM_LLM_CLOUD_ENABLED"] = "0"
    # Controlled arcs must present ONLY their declared entities (Exp 44 fix).
    env["MAXIM_DISABLE_IMAGINATION"] = "1"
    env["PYTHONPATH"] = str(_REPO / "src")
    if profile:
        env["MAXIM_LLM_PROFILE"] = profile
    env.update(extra)
    # Share the model cache so sub-sims don't re-download GGUFs.
    src_models = Path(os.path.expanduser("~/.maxim/models"))
    link = data_home / "models"
    if src_models.exists() and not link.exists():
        try:
            link.symlink_to(src_models)
        except OSError:
            pass
    return env


def _run_sim(cmd: list[str], env: dict[str, str], log_path: Path, timeout_s: int) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as log:
        log.write(f"# cmd: {' '.join(cmd)}\n")
        log.flush()
        try:
            proc = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=timeout_s)
            return proc.returncode
        except subprocess.TimeoutExpired:
            log.write(f"\n# TIMEOUT after {timeout_s}s\n")
            return -9


def _list_sessions(data_home: Path) -> set[str]:
    reports = data_home / "sim_reports"
    if not reports.is_dir():
        return set()
    return {d.name for d in reports.iterdir() if d.is_dir()}


def _find_new_session(data_home: Path, before: set[str]) -> Path | None:
    """Newest session dir CREATED by the run we just launched (snapshot diff).

    Newest-mtime-overall is wrong here: learn and capture sessions share one
    sim_reports/, so after a deleted learn marker the newest dir could be a
    CAPTURE session whose resumed, tau-held aut_nac.json would pass the bias
    gate spuriously (executor-lens finding).
    """
    new = _list_sessions(data_home) - before
    if not new:
        return None
    reports = data_home / "sim_reports"
    return max((reports / n for n in new), key=lambda d: d.stat().st_mtime)


def _fingerprint(d: dict[str, Any]) -> str:
    """Stable hash of the stage-relevant config so edited parameters invalidate
    cached stage markers instead of silently reusing stale outputs."""
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]


# ── stages ───────────────────────────────────────────────────────────────────


def stage_learn(
    arm: dict[str, Any],
    seed: int,
    arm_dir: Path,
    cfg: dict[str, Any],
    campaign_dir: Path,
    prov: dict[str, str],
    dry: bool,
) -> Path | None:
    """Run the substrate-primary learn pass; return the verified session dir."""
    data_home = arm_dir / f"seed{seed}"
    marker = data_home / "learn_verified.json"
    learn = arm.get("learn", {})
    min_bias = float(learn.get("min_bias", MIN_BIAS_DEFAULT))
    fp = _fingerprint(
        {
            "arc": arm["arc"],
            "max_turns": learn.get("max_turns", 56),
            "min_bias": min_bias,
            "seed": seed,
            "embodiment": cfg.get("embodiment", "bodies/infant_humanoid"),
            "narrator_profile": cfg.get("narrator_profile"),
        }
    )
    if marker.exists():
        rec = json.loads(marker.read_text())
        if rec.get("fp") == fp and Path(rec["session_dir"]).is_dir():
            return Path(rec["session_dir"])
        print(f"  learn marker stale (config changed or session gone) arm={arm['name']} seed={seed} — re-running")
        marker.unlink()
    cmd = [
        _resolve_maxim_binary(),
        "--sim",
        arm["arc"],
        "--aut-mode",
        "substrate-primary",
        "--embodiment",
        cfg.get("embodiment", "bodies/infant_humanoid"),
        "--interactive",
        "false",
        "--sim-max-turns",
        str(learn.get("max_turns", 56)),
        "--seed",
        str(seed),
    ]
    if dry:
        print(f"  [dry] LEARN  {' '.join(cmd)}")
        return None
    env = _sub_env(
        data_home,
        profile=cfg.get("narrator_profile"),
        extra={
            # Learn is substrate-primary (no LLM in the action path) — no server
            # spawn. Capture deliberately DOES allow auto-spawn (llm-primary
            # needs a backend).
            "MAXIM_AUTO_SPAWN_LLM_SERVER": "0",
            # Exploration bonus is REQUIRED (executor-lens blocker): config
            # default is 0.0 and without it an untried affordance never clears
            # the NAc min-confidence gate — every learn seed floors at the bias
            # gate. 1.5 is the Exp 42 graduated value.
            "MAXIM_SIM_SUBSTRATE_EXPLORE_BONUS_WEIGHT": str(learn.get("explore_weight", 1.5)),
            # Drive-gating ON — the Exp 42 enabling mechanism. Pinned (no
            # parent-env override): 44b runs no gating ablation.
            "MAXIM_SIM_DRIVE_GATE_ENABLED": "1",
        },
    )
    before = _list_sessions(data_home)
    rc = _run_sim(cmd, env, data_home / "logs" / "learn.log", int(learn.get("timeout_s", LEARN_TIMEOUT_S)))
    session = _find_new_session(data_home, before)
    nac = session / "aut_nac.json" if session else None
    bias = _max_abs_cluster_bias(nac) if nac and nac.exists() else 0.0
    ok = rc == 0 and session is not None and bias >= min_bias
    _append_manifest(
        campaign_dir,
        {
            "stage": "learn",
            "arm": arm["name"],
            "seed": seed,
            "rc": rc,
            "session": session.name if session else None,
            "max_abs_cluster_bias": round(bias, 4),
            "min_bias": min_bias,
            "ok": ok,
            **prov,
        },
    )
    if not ok:
        print(
            f"  LEARN FAILED arm={arm['name']} seed={seed} rc={rc} bias={bias:.3f} "
            f"(need >= {min_bias}) — see {data_home / 'logs' / 'learn.log'}",
            file=sys.stderr,
        )
        return None
    marker.write_text(json.dumps({"session_dir": str(session), "bias": bias, "fp": fp}))
    return session


def stage_capture(
    arm: dict[str, Any],
    seed: int,
    arm_dir: Path,
    session: Path | None,
    cfg: dict[str, Any],
    campaign_dir: Path,
    prov: dict[str, str],
    dry: bool,
) -> Path | None:
    """Run the llm-primary capture pass; return the verified paired-prompt JSONL."""
    data_home = arm_dir / f"seed{seed}"
    capture_path = data_home / "capture.jsonl"
    marker = data_home / "capture_verified.json"
    cap = arm.get("capture", {})
    substrate = arm.get("substrate", "learn")
    fp = _fingerprint(
        {
            "arc": arm["arc"],
            "max_turns": cap.get("max_turns", 40),
            "decay_tau": cap.get("decay_tau", 1000),
            "model": cap.get("model") or cfg.get("capture_profile"),
            "substrate": substrate,
            "seed": seed,
        }
    )
    if marker.exists():
        rec = json.loads(marker.read_text())
        if rec.get("fp") == fp and capture_path.exists():
            return capture_path
        print(f"  capture marker stale (config changed or file gone) arm={arm['name']} seed={seed} — re-running")
        marker.unlink()
    # The capture hook opens the JSONL in APPEND mode — a failed attempt's
    # partial pairs must not mix with the retry's (executor-lens blocker).
    if capture_path.exists():
        capture_path.unlink()

    # Transplant: copy the source arm's same-seed learned session into OUR
    # data home so --resume-sim resolves it (wrong-content control).
    if substrate.startswith("transplant:") and not dry:
        src_arm = substrate.split(":", 1)[1]
        src_marker = campaign_dir / "arms" / src_arm / f"seed{seed}" / "learn_verified.json"
        if not src_marker.exists():
            print(
                f"  TRANSPLANT BLOCKED arm={arm['name']} seed={seed}: source arm "
                f"'{src_arm}' has no verified learn for this seed",
                file=sys.stderr,
            )
            return None
        src_session = Path(json.loads(src_marker.read_text())["session_dir"])
        dest = data_home / "sim_reports" / src_session.name
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(src_session, dest)
        session = dest

    cmd = [
        _resolve_maxim_binary(),
        "--sim",
        arm["arc"],
        "--aut-mode",
        "llm-primary",
        "--embodiment",
        cfg.get("embodiment", "bodies/infant_humanoid"),
        "--interactive",
        "false",
        "--sim-max-turns",
        str(cap.get("max_turns", 40)),
        "--seed",
        str(seed + 1000),  # decorrelate world noise from the learn pass
    ]
    if cap.get("model"):
        cmd += ["--aut-model", cap["model"]]
    if substrate != "none":
        if session is None and not dry:
            return None
        cmd += ["--resume-sim", session.name if session else "<learn-session>"]

    if dry:
        print(f"  [dry] CAPTURE {' '.join(cmd)}")
        return None
    env = _sub_env(
        data_home,
        profile=cap.get("model") or cfg.get("capture_profile"),
        extra={
            "MAXIM_EXP44_CAPTURE_LOG": str(capture_path),
            "MAXIM_NAC_CLUSTER_REWARD_BIAS_DECAY_TAU": str(cap.get("decay_tau", 1000)),
            # Deterministic scene self_effect in llm-primary (Exp 44 G1).
            "MAXIM_DETERMINISTIC_SCENE_EMBODIMENT": "1",
        },
    )
    rc = _run_sim(cmd, env, data_home / "logs" / "capture.log", int(cap.get("timeout_s", CAPTURE_TIMEOUT_S)))
    n_pairs = 0
    n_with_annotation = 0
    if capture_path.exists():
        with open(capture_path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                n_pairs += 1
                try:
                    if json.loads(line).get("has_cluster_bias"):
                        n_with_annotation += 1
                except json.JSONDecodeError:
                    pass
    annotation_fraction = (n_with_annotation / n_pairs) if n_pairs else 0.0

    # Annotation-presence gates (two-lens fold, frozen in the pre-registration):
    # a substrate-carrying capture whose prompts DON'T carry the annotation is a
    # broken instrument, not a null result. Confirmatory ("learn") arms FAIL the
    # seed; transplant arms proceed but are marked VOID (visible in stats), since
    # cross-arc bias surfacing across the _b name suffix is the unverified thing
    # the control gate exists to check.
    void_marker = data_home / "control_void.json"
    annotation_ok = True
    if substrate == "learn":
        annotation_ok = annotation_fraction >= 0.5
    elif substrate.startswith("transplant:"):
        if annotation_fraction < 0.5:
            void_marker.write_text(
                json.dumps(
                    {
                        "annotation_fraction": annotation_fraction,
                        "reason": "transplanted substrate did not surface in capture prompts",
                    }
                )
            )
            print(
                f"  CONTROL VOID arm={arm['name']} seed={seed}: annotation_fraction={annotation_fraction:.2f} < 0.5",
                file=sys.stderr,
            )
        elif void_marker.exists():
            void_marker.unlink()

    ok = rc == 0 and n_pairs >= int(cap.get("min_pairs", MIN_CAPTURE_PAIRS)) and annotation_ok
    _append_manifest(
        campaign_dir,
        {
            "stage": "capture",
            "arm": arm["name"],
            "seed": seed,
            "rc": rc,
            "n_pairs": n_pairs,
            "annotation_fraction": round(annotation_fraction, 3),
            "substrate": substrate,
            "resumed_session": session.name if session else None,
            "capture_sha16": _sha16(capture_path) if capture_path.exists() else None,
            "ok": ok,
            **prov,
        },
    )
    if not ok:
        print(
            f"  CAPTURE FAILED arm={arm['name']} seed={seed} rc={rc} pairs={n_pairs} "
            f"annotation_fraction={annotation_fraction:.2f} — see {data_home / 'logs' / 'capture.log'}",
            file=sys.stderr,
        )
        return None
    # A fresh verified capture invalidates every requery of the old capture —
    # prune so stale results can't be picked up by the stats walk (cross-
    # confirmed two-lens finding).
    shutil.rmtree(data_home / "requery", ignore_errors=True)
    marker.write_text(json.dumps({"n_pairs": n_pairs, "sha16": _sha16(capture_path), "fp": fp}))
    return capture_path


def stage_requery(
    arm: dict[str, Any],
    seed: int,
    arm_dir: Path,
    capture: Path,
    model: str,
    cfg: dict[str, Any],
    campaign_dir: Path,
    prov: dict[str, str],
    dry: bool,
) -> Path | None:
    """Offline temp-0 re-query, cached by (capture hash, model, entropy params)."""
    ent = cfg.get("entropy", {})
    if dry:
        print(f"  [dry] REQUERY[{model}] arm={arm['name']} seed={seed} (cached by capture hash)")
        return None
    if not capture.exists():
        print(f"  REQUERY SKIPPED arm={arm['name']} seed={seed}: capture file missing", file=sys.stderr)
        return None
    # "__" is the stats-side field separator — sanitize it out of the model key
    # so a profile name containing "__" can't be mislabeled downstream.
    model_key = model.replace("__", "-")
    key = f"{model_key}__{_sha16(capture)}__e{ent.get('samples', 8)}t{ent.get('temp', 0.7)}"
    out = arm_dir / f"seed{seed}" / "requery" / f"{key}.jsonl"
    if out.exists() and out.stat().st_size > 0:
        return out  # cache hit — cross-model sweeps and re-runs are free
    # Write to a .partial and rename only on success: the re-query flushes per
    # record, so a SIGKILL mid-run would otherwise leave a truncated-but-valid-
    # looking JSONL that is cached forever (executor-lens blocker).
    tmp = out.with_suffix(".partial")
    cmd = [
        sys.executable,
        str(_HERE.parent / "rerun_ablated_offline.py"),
        "--log",
        str(capture),
        "--out",
        str(tmp),
        "--entropy-samples",
        str(ent.get("samples", 8)),
        "--entropy-temp",
        str(ent.get("temp", 0.7)),
        "--entropy-hi",
        str(ent.get("hi", 0.5)),
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["MAXIM_LLM_PROFILE"] = model
    env["MAXIM_LLM_CLOUD_ENABLED"] = "0"  # a cloud-mapped model must not silently spend
    env["MAXIM_DATA_HOME"] = str(arm_dir / f"seed{seed}")
    env["PYTHONPATH"] = str(_REPO / "src")
    try:
        rc = subprocess.run(cmd, env=env, timeout=int(cfg.get("requery_timeout_s", 14400))).returncode
    except subprocess.TimeoutExpired:
        rc = -9
    ok = rc == 0 and tmp.exists() and tmp.stat().st_size > 0
    if ok:
        tmp.rename(out)
    elif tmp.exists():
        tmp.unlink()
    _append_manifest(
        campaign_dir,
        {
            "stage": "requery",
            "arm": arm["name"],
            "seed": seed,
            "model": model,
            "rc": rc,
            "out": str(out),
            "ok": ok,
            **prov,
        },
    )
    if not ok:
        print(f"  REQUERY FAILED arm={arm['name']} seed={seed} model={model}", file=sys.stderr)
        if out.exists():
            out.unlink()  # never leave a half-written cache entry
        return None
    return out


# ── driver ───────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="campaign JSON (see campaign_44b.json)")
    ap.add_argument("--workdir", required=True, help="campaign output directory")
    ap.add_argument("--arms", default="", help="comma-separated arm-name filter")
    ap.add_argument("--seeds", default="", help="comma-separated seed filter")
    ap.add_argument("--requery-models", default="", help="override config requery_models")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-stats", action="store_true")
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    campaign_dir = Path(args.workdir).resolve()
    campaign_dir.mkdir(parents=True, exist_ok=True)

    # Provenance preflight (Exp 42b): the maxim the sub-sims import must be
    # THIS repo. Exits 3 on mismatch before any compute is spent (the contract
    # the CLAUDE.md lesson + the sibling Exp 42 harness pin verbatim).
    binary = _resolve_maxim_binary()
    if not args.dry_run:
        from _provenance import ProvenanceError

        try:
            assert_repo_interpreter(_REPO, binary)
        except ProvenanceError as e:
            print(f"PROVENANCE MISMATCH: {e}", file=sys.stderr)
            return 3
    prov = executed_code_provenance(_REPO, binary)
    _append_manifest(campaign_dir, {"stage": "campaign_start", "config": cfg, **prov})

    arm_filter = {a for a in args.arms.split(",") if a}
    seed_filter = {int(s) for s in args.seeds.split(",") if s}
    requery_models = [
        m
        for m in (
            [m for m in args.requery_models.split(",") if m]
            or cfg.get("requery_models")
            or [cfg.get("capture_profile")]
        )
        if m
    ]
    if not requery_models:
        print("ERROR: no requery model — set requery_models or capture_profile in the config", file=sys.stderr)
        return 2

    # Learn-before-transplant ordering: all "learn" arms first.
    arms = [a for a in cfg["arms"] if not arm_filter or a["name"] in arm_filter]
    arms.sort(key=lambda a: 0 if a.get("substrate", "learn") == "learn" else 1)

    failed_cells = 0
    for arm in arms:
        arm_dir = campaign_dir / "arms" / arm["name"]
        seeds = [s for s in arm["seeds"] if not seed_filter or s in seed_filter]
        print(f"== arm {arm['name']} ({arm['arc']}, substrate={arm.get('substrate', 'learn')}) seeds={seeds}")
        for seed in seeds:
            session = None
            if arm.get("substrate", "learn") == "learn":
                session = stage_learn(arm, seed, arm_dir, cfg, campaign_dir, prov, args.dry_run)
                if session is None and not args.dry_run:
                    failed_cells += 1
                    continue
            capture = stage_capture(arm, seed, arm_dir, session, cfg, campaign_dir, prov, args.dry_run)
            if capture is None and not args.dry_run:
                failed_cells += 1
                continue
            for model in requery_models:
                if capture is not None or args.dry_run:
                    got = stage_requery(
                        arm, seed, arm_dir, capture or Path("dry"), model, cfg, campaign_dir, prov, args.dry_run
                    )
                    if got is None and not args.dry_run:
                        failed_cells += 1

    if failed_cells:
        print(f"\n{failed_cells} cell(s) FAILED — see manifest.jsonl and per-seed logs", file=sys.stderr)

    if not args.dry_run and not args.skip_stats:
        rc = subprocess.run(
            [
                sys.executable,
                str(_HERE.parent / "stats_counterfactual.py"),
                "--campaign",
                str(campaign_dir),
                "--config",
                args.config,
            ]
        ).returncode
        return rc or (1 if failed_cells else 0)
    return 1 if failed_cells else 0


if __name__ == "__main__":
    raise SystemExit(main())
