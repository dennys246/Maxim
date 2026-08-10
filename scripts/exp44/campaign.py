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

    Recursive scan rather than a hardcoded shape so NAc persistence-format
    evolution degrades this gate loudly (0.0 → gate fails) instead of silently.
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


def _sub_env(data_home: Path, *, profile: str | None, extra: dict[str, str]) -> dict[str, str]:
    env = os.environ.copy()
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


def _find_session(data_home: Path) -> Path | None:
    reports = data_home / "sim_reports"
    if not reports.is_dir():
        return None
    dirs = sorted((d for d in reports.iterdir() if d.is_dir()), key=lambda d: d.stat().st_mtime)
    return dirs[-1] if dirs else None


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
    if marker.exists():
        return Path(json.loads(marker.read_text())["session_dir"])

    learn = arm.get("learn", {})
    min_bias = float(learn.get("min_bias", MIN_BIAS_DEFAULT))
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
            "MAXIM_AUTO_SPAWN_LLM_SERVER": "0",
            "MAXIM_SIM_DRIVE_GATE_ENABLED": os.environ.get("MAXIM_SIM_DRIVE_GATE_ENABLED", "1"),
        },
    )
    rc = _run_sim(cmd, env, data_home / "logs" / "learn.log", int(learn.get("timeout_s", LEARN_TIMEOUT_S)))
    session = _find_session(data_home)
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
    marker.write_text(json.dumps({"session_dir": str(session), "bias": bias}))
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
    if marker.exists():
        return capture_path

    cap = arm.get("capture", {})
    substrate = arm.get("substrate", "learn")

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
    if capture_path.exists():
        with open(capture_path, encoding="utf-8") as f:
            n_pairs = sum(1 for line in f if line.strip())
    ok = rc == 0 and n_pairs >= int(cap.get("min_pairs", MIN_CAPTURE_PAIRS))
    _append_manifest(
        campaign_dir,
        {
            "stage": "capture",
            "arm": arm["name"],
            "seed": seed,
            "rc": rc,
            "n_pairs": n_pairs,
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
            f"— see {data_home / 'logs' / 'capture.log'}",
            file=sys.stderr,
        )
        return None
    marker.write_text(json.dumps({"n_pairs": n_pairs, "sha16": _sha16(capture_path)}))
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
    key = f"{model}__{_sha16(capture)}__e{ent.get('samples', 8)}t{ent.get('temp', 0.7)}"
    out = arm_dir / f"seed{seed}" / "requery" / f"{key}.jsonl"
    if out.exists() and out.stat().st_size > 0:
        return out  # cache hit — cross-model sweeps and re-runs are free
    cmd = [
        sys.executable,
        str(_HERE.parent / "rerun_ablated_offline.py"),
        "--log",
        str(capture),
        "--out",
        str(out),
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
    env["PYTHONPATH"] = str(_REPO / "src")
    rc = subprocess.run(cmd, env=env).returncode
    ok = rc == 0 and out.exists() and out.stat().st_size > 0
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
    # THIS repo. Exits 3 on mismatch before any compute is spent.
    binary = _resolve_maxim_binary()
    if not args.dry_run:
        assert_repo_interpreter(_REPO, binary)
    prov = executed_code_provenance(_REPO, binary)
    _append_manifest(campaign_dir, {"stage": "campaign_start", "config": cfg, **prov})

    arm_filter = {a for a in args.arms.split(",") if a}
    seed_filter = {int(s) for s in args.seeds.split(",") if s}
    requery_models = (
        [m for m in args.requery_models.split(",") if m] or cfg.get("requery_models") or [cfg.get("capture_profile")]
    )

    # Learn-before-transplant ordering: all "learn" arms first.
    arms = [a for a in cfg["arms"] if not arm_filter or a["name"] in arm_filter]
    arms.sort(key=lambda a: 0 if a.get("substrate", "learn") == "learn" else 1)

    for arm in arms:
        arm_dir = campaign_dir / "arms" / arm["name"]
        seeds = [s for s in arm["seeds"] if not seed_filter or s in seed_filter]
        print(f"== arm {arm['name']} ({arm['arc']}, substrate={arm.get('substrate', 'learn')}) seeds={seeds}")
        for seed in seeds:
            session = None
            if arm.get("substrate", "learn") == "learn":
                session = stage_learn(arm, seed, arm_dir, cfg, campaign_dir, prov, args.dry_run)
                if session is None and not args.dry_run:
                    continue
            capture = stage_capture(arm, seed, arm_dir, session, cfg, campaign_dir, prov, args.dry_run)
            if capture is None and not args.dry_run:
                continue
            for model in requery_models:
                if capture is not None or args.dry_run:
                    stage_requery(
                        arm, seed, arm_dir, capture or Path("dry"), model, cfg, campaign_dir, prov, args.dry_run
                    )

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
        return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
