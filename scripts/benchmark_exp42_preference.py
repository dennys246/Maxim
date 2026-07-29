#!/usr/bin/env python3
"""Harness for Exp 42 — Substrate-Primary Preference (counterbalanced A/B).

Runs the two counterbalanced arms of
docs/experiments/42_substrate_primary_preference.md §4 under
``--aut-mode substrate-primary`` (no LLM in the action path → ``cost=$0``):

    | arm           | arc           | safe identity | harmful identity |
    |---------------|---------------|---------------|------------------|
    | cradle_pref_a | cradle_pref_a | beta          | alpha            |
    | cradle_pref_b | cradle_pref_b | alpha         | beta             |

Exploration is held ON in BOTH arms (Exp 41 settled it as the enabling
condition, not a differentiator) via ``MAXIM_SIM_SUBSTRATE_EXPLORE_BONUS_WEIGHT``.
Each run uses the entropic-cold infant body (``bodies/infant_humanoid_chilled``)
so warming is a sustained, recurring drive-relevant choice — the dense
exploitation-phase sampling the validation spike confirmed (avoids the Exp 41
flooring trap; see the doc §2a).

The single primary test (the agent prefers the SAFE source in BOTH arms) is
bias-proof: the safe identity flips between arms, so a fixed identity/name/
position bias cannot satisfy both. The harness owns the per-run rate extraction
(``safe_pref``, ``id_pref_a``); the analyzer owns the hypotheses (§4/§5).

This is a DEDICATED harness (sibling of benchmark_exp41_exploration.py), not part
of benchmark_cross_session.py (Exp 37): the substrate-primary preference run has
none of Exp 37's cross-session / cost / cloud / ablation machinery. It reuses the
same actions.jsonl contract (header line skipped; per-action ``tool`` field) and
``MAXIM_DATA_HOME/sim_reports/<id>/`` layout, plus a best-effort read of the
CWD-relative ``data/sim_sandbox/substrate_telemetry_*.jsonl`` for the C2
per-source learning signal.

Output: append-only per-run JSONL consumed by ``scripts/analyze_exp42_preference.py``.

Smoke test (CI-safe, no subprocess / no LLM):
    python scripts/benchmark_exp42_preference.py --mock --trials 3 --out /tmp/x.jsonl
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

ARMS: tuple[str, ...] = ("cradle_pref_a", "cradle_pref_b")

# arm → arc goal (1:1 here; kept as a map to mirror the Exp 41 shape and to keep
# the arm label decoupled from the arc string if a variant is added later).
_ARM_ARC: dict[str, str] = {a: a for a in ARMS}

DEFAULT_EXPLORE_WEIGHT = 1.5
DEFAULT_MODEL = "smollm-1.7b-instruct"  # generative-narrator LLM; AUT is LLM-free
DEFAULT_EMBODIMENT = "bodies/infant_humanoid_chilled"
DEFAULT_MAX_TURNS = 40  # validation spike: K=7 at 24 turns → ~40 for K≥10


# ── metric extraction (the harness owns rates; the analyzer owns hypotheses) ──


def _is_contact(tool: str) -> bool:
    """A warming contact affordance on a warmth source. warm_self and touch share
    an identical self_effect (both relieve cold; harmful variants both breach the
    thermal band) — aggregated for exploitation-phase sample density per the
    validation spike (doc §2a / §4)."""
    return "warmth_" in tool and (tool.endswith("_warm_self") or tool.endswith("_touch"))


def _safety(tool: str) -> str:
    """'safe' | 'harm' from the entity-name suffix. The substrate never reads
    names semantically (no LLM; the affinity heuristic only substring-matches
    'warm'), so the suffix is an analysis label, not a leaked cue."""
    if "_safe_" in tool:
        return "safe"
    if "_harm_" in tool:
        return "harm"
    return "?"


def _identity(tool: str) -> str:
    """'alpha' | 'beta' from the entity-name prefix — the identity channel for
    the corroborating flip (C1)."""
    if "_alpha_" in tool:
        return "alpha"
    if "_beta_" in tool:
        return "beta"
    return "?"


def compute_run_metrics(tools: list[str]) -> dict[str, Any]:
    """Exploitation-phase preference from the executed-action sequence.

    Exploitation phase (doc §4): all contacts AFTER the tick by which BOTH
    warmth sources (the safe entity AND the harmful entity) have each been
    engaged via a contact affordance at least once (the explore-first discovery
    point). Choices before that point are forced first trials, not preference.
    """
    contacts = [(i, t) for i, t in enumerate(tools) if _is_contact(t)]
    n_contact = len(contacts)

    # discovery point: the contact index by which both safety classes have appeared.
    seen: set[str] = set()
    discovery_pos: int | None = None  # position WITHIN the contact list
    for pos, (_idx, tool) in enumerate(contacts):
        s = _safety(tool)
        if s in ("safe", "harm"):
            seen.add(s)
        if {"safe", "harm"} <= seen:
            discovery_pos = pos
            break

    if discovery_pos is None:
        # never discovered both sources → no exploitation phase
        return {
            "n_actions": len(tools),
            "n_contact": n_contact,
            "n_exploit": 0,
            "safe_pref": None,
            "id_pref_a": None,
            "discovery_pos": None,
        }

    exploit = [t for _i, t in contacts[discovery_pos + 1 :]]
    n_exploit = len(exploit)
    if n_exploit == 0:
        return {
            "n_actions": len(tools),
            "n_contact": n_contact,
            "n_exploit": 0,
            "safe_pref": None,
            "id_pref_a": None,
            "discovery_pos": discovery_pos,
        }

    n_safe = sum(1 for t in exploit if _safety(t) == "safe")
    n_alpha = sum(1 for t in exploit if _identity(t) == "alpha")
    return {
        "n_actions": len(tools),
        "n_contact": n_contact,
        "n_exploit": n_exploit,
        "safe_pref": n_safe / n_exploit,
        "id_pref_a": n_alpha / n_exploit,
        "discovery_pos": discovery_pos,
    }


def extract_learning_nets(causal_links: dict[str, Any]) -> dict[str, Any]:
    """Best-effort C2 signal: max net over harmful vs safe contact links.

    ``causal_links`` is the per-event ``{tool:NAME: {pos, neg, net}}`` map from
    the last substrate_telemetry line. Relative, not absolute (doc §2a): the
    harmful source takes a fresh breach every warm so it ends MORE NEGATIVE than
    the safe source, which only inherits lingering noise.
    """
    harm_nets: list[float] = []
    safe_nets: list[float] = []
    for key, v in causal_links.items():
        if not isinstance(v, dict) or "net" not in v:
            continue
        name = key[len("tool:") :] if key.startswith("tool:") else key
        if not _is_contact(name):
            continue
        try:
            net = float(v["net"])
        except (TypeError, ValueError):
            continue
        if _safety(name) == "harm":
            harm_nets.append(net)
        elif _safety(name) == "safe":
            safe_nets.append(net)
    return {
        "harm_net": max(harm_nets) if harm_nets else None,
        "safe_net": max(safe_nets) if safe_nets else None,
    }


# ── sub-sim execution ────────────────────────────────────────────────────


def _resolve_maxim_binary() -> str:
    found = shutil.which("maxim")
    if found:
        return found
    here = Path(__file__).resolve().parent.parent
    cand = here / ".venv" / "bin" / "maxim"
    if cand.exists():
        return str(cand)
    return "maxim"


def _load_action_tools(data_home: Path) -> list[str]:
    """Read the newest session's actions.jsonl and return executed tool names.

    Mirrors benchmark_exp41_exploration._load_action_tools: skip the header line,
    then collect per-action ``tool`` fields in order.
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


_TELEM_DIR = Path("data") / "sim_sandbox"


def _telemetry_files() -> set[Path]:
    if not _TELEM_DIR.exists():
        return set()
    return set(_TELEM_DIR.glob("substrate_telemetry_*.jsonl"))


def _read_learning_nets(new_files: set[Path]) -> dict[str, Any]:
    """Best-effort: from the telemetry file(s) created during this run, read the
    last line's per-event causal_links and reduce to harm/safe nets. Returns
    ``{harm_net: None, safe_net: None}`` on any miss (C2 is corroborating, not a
    gate — a telemetry miss must not fail the run)."""
    if not new_files:
        return {"harm_net": None, "safe_net": None}
    newest = max(new_files, key=lambda p: p.stat().st_mtime)
    try:
        lines = [ln for ln in newest.read_text().splitlines() if ln.strip()]
        if not lines:
            return {"harm_net": None, "safe_net": None}
        last = json.loads(lines[-1])
        cl = (last.get("nac") or {}).get("causal_links") or {}
        return extract_learning_nets(cl)
    except Exception:
        return {"harm_net": None, "safe_net": None}


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
    aut_mode: str = "substrate-primary",
    aut_model: str | None = None,
) -> tuple[list[str], dict[str, Any]]:
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
    if aut_mode == "substrate-primary":
        # Exploration ON in BOTH arms (Exp 42 design).
        env["MAXIM_SIM_SUBSTRATE_EXPLORE_BONUS_WEIGHT"] = str(explore_weight)
        # Drive-gating ON (motivated attention) — opt-in, the Exp 42 enabling
        # mechanism (without it the agent fixates on always-succeeding zero-stakes
        # tools instead of warming; see the doc §3a / the B7 triage history).
        # Respect a parent-env override so the gating-OFF ABLATION arm runs cleanly:
        #   MAXIM_SIM_DRIVE_GATE_ENABLED=0 python scripts/benchmark_exp42_preference.py ...
        env["MAXIM_SIM_DRIVE_GATE_ENABLED"] = os.environ.get("MAXIM_SIM_DRIVE_GATE_ENABLED", "1")
    # Exp 44 (LLM-primary body_state ablation): arm env vars —
    # MAXIM_ENABLE_BODY_STATE_PROMPT / MAXIM_DISABLE_COACH_BODY_LAYERS —
    # flow through os.environ.copy() above; export them in the launching
    # shell per arm. They are recorded per-run for provenance (_record).

    cmd = [
        _resolve_maxim_binary(),
        "--sim",
        _ARM_ARC[arm],
        "--aut-mode",
        aut_mode,
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
    if aut_model:
        cmd += ["--aut-model", aut_model]
    log_dir = data_home / "harness_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    telem_before = _telemetry_files()
    try:
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        (log_dir / "timeout.log").write_text(f"TIMEOUT {timeout_s}s\n{exc.stdout or ''}\n{exc.stderr or ''}")
        raise RuntimeError(f"{arm} seed={seed}: sub-sim timed out after {timeout_s}s") from exc
    (log_dir / "run.log").write_text((proc.stdout or "") + "\n---STDERR---\n" + (proc.stderr or ""))
    nets = _read_learning_nets(_telemetry_files() - telem_before)
    return _load_action_tools(data_home), nets


def _mock_tools(arm: str, seed: int) -> list[str]:
    """Synthesize a plausible action sequence per arm for CI smoke tests.

    Encodes the EXPECTED pattern so a mock fire yields a GRADUATE verdict — the
    agent prefers the SAFE source in BOTH arms (the safe identity flips):
      * cradle_pref_a: safe = beta  → exploit warmth_beta_safe.
      * cradle_pref_b: safe = alpha → exploit warmth_alpha_safe.
    Both explore each source once first (so the discovery point exists), then a
    couple of harmful slips, then converge on the safe source. A small
    seed-dependent jitter creates non-zero cross-seed SD.
    """
    j = seed % 3  # 0..2 jitter
    if arm == "cradle_pref_a":
        safe, harm = "warmth_beta_safe", "warmth_alpha_harm"
    else:  # cradle_pref_b
        safe, harm = "warmth_alpha_safe", "warmth_beta_harm"
    # explore both (discovery), a few harmful slips, then exploit safe.
    explore = [f"{harm}_warm_self", f"{safe}_warm_self"]
    slips = [f"{harm}_warm_self"] * (1 + j)
    exploit = [f"{safe}_warm_self", f"{safe}_touch"] * (8 - j)
    # interleave some non-contact sensing so n_actions > n_contact (realistic).
    seq = explore + ["sense_scene"] + slips + ["observe"] + exploit
    return seq


# Same six-value truthy set as the runtime parsers
# (prompts/cluster_bias_annotation.py::TRUTHY_DISABLE_VALUES, replicated in
# integration/memory_hub.py::body_state_prompt_enabled). Replicated here
# because this script runs standalone; tests/behavioral/test_exp42_pipeline.py
# pins arm derivation so divergence fails loudly.
_TRUTHY = frozenset({"1", "true", "t", "yes", "y", "on"})


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUTHY


def _ablation_arm() -> str:
    """Normalize the Exp 44 arm from the two env vars: A (unwired status
    quo), B (body_state + coach layers off), C (body_state + coach on),
    or 'inconsistent' (layers-disable set while body_state is unwired —
    the disable is a no-op; flag it rather than mislabel)."""
    body = _env_truthy("MAXIM_ENABLE_BODY_STATE_PROMPT")
    layers_off = _env_truthy("MAXIM_DISABLE_COACH_BODY_LAYERS")
    if not body:
        return "inconsistent" if layers_off else "A"
    return "B" if layers_off else "C"


def _record(
    arm: str,
    seed: int,
    tools: list[str],
    nets: dict[str, Any],
    *,
    mock: bool,
    git_hash: str,
    aut_mode: str = "substrate-primary",
) -> dict[str, Any]:
    rec = {
        "experiment": "exp42",
        "arm": arm,
        "arc": _ARM_ARC[arm],
        "seed": seed,
        "git_hash": git_hash,
        "mock": mock,
        # Exp 44 provenance: which AUT mode ran, the NORMALIZED ablation arm
        # (derived with the same truthy parsing the runtime uses — group on
        # this, not the raw echoes), plus the raw env echoes as corroboration.
        "aut_mode": aut_mode,
        "ablation_arm": _ablation_arm(),
        "env_body_state_prompt": os.environ.get("MAXIM_ENABLE_BODY_STATE_PROMPT", ""),
        "env_coach_body_layers_disabled": os.environ.get("MAXIM_DISABLE_COACH_BODY_LAYERS", ""),
        # Drive-gating (B7) arm marker. Records the value ACTUALLY propagated to
        # the sub-sims (same `os.environ.get(..., "1")` default the launcher
        # applies), so a run file is self-describing about which arm it is.
        # Added 2026-07-28 after the Exp 42b re-validation: the treatment and
        # gating-OFF files were behaviourally indistinguishable and nothing in
        # the record could confirm the toggle had fired — "did the actuation
        # actually happen?" must be answerable from the data, not the shell
        # history.
        "env_drive_gate_enabled": os.environ.get("MAXIM_SIM_DRIVE_GATE_ENABLED", "1"),
    }
    rec.update(compute_run_metrics(tools))
    rec.update(nets)
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
    aut_mode: str = "substrate-primary",
    aut_model: str | None = None,
) -> int:
    git_hash = _git_hash()
    if aut_mode == "substrate-primary" and _env_truthy("MAXIM_ENABLE_BODY_STATE_PROMPT"):
        # body_state wiring is prompt-side (LLM-primary). In substrate-primary
        # it contaminates an Exp 42 replay for no effect on the action path.
        print(
            "WARNING: MAXIM_ENABLE_BODY_STATE_PROMPT is set but --aut-mode is "
            "substrate-primary — this contaminates an Exp 42 replay; unset it "
            "or pass --aut-mode llm-primary (Exp 44).",
            file=sys.stderr,
        )
    done = _existing_keys(out_path) if resume else set()
    workdir = Path("data/sim_sandbox/exp42_runs")
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
                        nets = {"harm_net": -0.25, "safe_net": 0.14}
                    else:
                        tools, nets = _run_real(
                            arm,
                            seed,
                            model=model,
                            embodiment=embodiment,
                            max_turns=max_turns,
                            explore_weight=explore_weight,
                            timeout_s=timeout_s,
                            workdir=workdir,
                            aut_mode=aut_mode,
                            aut_model=aut_model,
                        )
                except Exception as exc:  # noqa: BLE001 - log + continue per run
                    n_fail += 1
                    print(f"FAIL {arm} seed={seed}: {exc}", file=sys.stderr)
                    continue
                rec = _record(arm, seed, tools, nets, mock=mock, git_hash=git_hash, aut_mode=aut_mode)
                out.write(json.dumps(rec) + "\n")
                out.flush()
                n_done += 1
                sp = rec["safe_pref"]
                print(
                    f"ok {arm} seed={seed} n_contact={rec['n_contact']} n_exploit={rec['n_exploit']} "
                    f"safe_pref={'—' if sp is None else round(sp, 2)} ({time.time() - t0:.1f}s)"
                )

    print(f"\ndone: {n_done} runs recorded, {n_fail} failed → {out_path}")
    return 0 if n_fail == 0 else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Exp 42 substrate-primary preference harness (counterbalanced A/B)")
    p.add_argument("--arms", default=",".join(ARMS), help="comma-separated subset of cradle_pref_a,cradle_pref_b")
    p.add_argument("--trials", type=int, default=10, help="seeds per arm")
    p.add_argument("--seed-base", type=int, default=42)
    p.add_argument("--out", type=Path, required=True, help="append-only per-run JSONL")
    p.add_argument("--mock", action="store_true", help="synthesize runs (CI-safe; no subprocess/LLM)")
    p.add_argument("--model", default=DEFAULT_MODEL, help="generative-narrator LLM profile (AUT is LLM-free)")
    p.add_argument("--embodiment", default=DEFAULT_EMBODIMENT)
    p.add_argument("--sim-max-turns", type=int, default=DEFAULT_MAX_TURNS)
    p.add_argument("--explore-weight", type=float, default=DEFAULT_EXPLORE_WEIGHT)
    p.add_argument("--timeout-s", type=int, default=2400)
    p.add_argument("--resume", action="store_true", help="skip (arm, seed) pairs already in --out")
    p.add_argument(
        "--aut-mode",
        default="substrate-primary",
        choices=["substrate-primary", "llm-primary"],
        help="AUT action-selection mode passed to the sub-sim. substrate-primary "
        "(default) preserves Exp 42 byte-for-byte. llm-primary is the Exp 44 "
        "body_state ablation mode (docs/plans/acting_coach_body_state_ablation.md) — "
        "export MAXIM_ENABLE_BODY_STATE_PROMPT / MAXIM_DISABLE_COACH_BODY_LAYERS "
        "per arm in the launching shell; both are recorded per-run for provenance. "
        "Raise --timeout-s: LLM-primary turns are order-of-magnitude slower.",
    )
    p.add_argument(
        "--aut-model",
        default=None,
        help="Separate LLM profile for the agent-under-test (passed through as "
        "the sub-sim's --aut-model). Only meaningful with --aut-mode llm-primary.",
    )
    args = p.parse_args(argv)

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
        aut_mode=args.aut_mode,
        aut_model=args.aut_model,
    )


if __name__ == "__main__":
    raise SystemExit(main())
