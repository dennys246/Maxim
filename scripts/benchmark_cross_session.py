#!/usr/bin/env python3
"""Exp 37 benchmark harness — cross-session graduation.

Drives the paired-trials measurement defined in
[docs/experiments/37_cross_session_graduation.md] across six arms and two
scenarios, with per-trial sandbox isolation, ablation env vars, an
aggregate cost cap, and a mock-LLM mode for smoke testing.

Arm structure (per scenario × per trial pair):
  - A             — fresh agent on the failure scenario (also = prior for B-family).
  - B             — resume from Arm A; primary cross-session measurement.
  - B-wire-a-off  — resume from Arm A + MAXIM_DISABLE_CLUSTER_BIAS_ANNOTATION=1.
  - B-wire-1-off  — resume from Arm A + MAXIM_DISABLE_VARIANCE_ANNOTATION=1.
  - B-nac-bias-off — resume from Arm A + MAXIM_NAC_REWARD_BIAS_DISABLED=1.
  - C             — resume from a peaceful-prior session (isolation control).

Arm C priors are shared per trial-pair across scenarios (a single peaceful
prior feeds both fire_pit-C and sharp_rock-C). Arm A target runs are copied
(shutil.copytree) before each B-family resume so each ablation arm starts
from an identical post-A substrate snapshot.

Per-trial JSONL records are emitted to ``--out`` (append-only). The schema
is locked at :data:`SCHEMA_VERSION` and documented in
``docs/experiments/protocols/37_cross_session_graduation_reproduction.md``.

USAGE — DO NOT FIRE TRIALS UNPROMPTED. Real-LLM execution costs ~$0.20/run.
A 5-trial fire_pit + sharp_rock run on Claude Sonnet projects ~65 runs /
~$14. The harness aborts with a clear error if cumulative cost exceeds
``--cost-cap`` (default $20).

CI MUST NOT run this script — see tests/behavioral/test_exp37_harness_smoke.py
for the mock-backend coverage. The smoke test invokes this module's
functions directly with ``mock=True``; no subprocess, no LLM, no cost.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

# Make `maxim` importable when run from the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

# ─── Constants (frozen at v1) ────────────────────────────────────────────

HARNESS_VERSION = "1.0"
SCHEMA_VERSION = "1.0"
EXPERIMENT_ID = "exp37_cross_session_graduation"

SCENARIOS: tuple[str, ...] = ("fire_pit", "sharp_rock")
ARMS: tuple[str, ...] = ("A", "B", "C", "B-wire-a-off", "B-wire-1-off", "B-nac-bias-off")

# Per-scenario failure-class action detection (per pre-reg §"Failed action class").
# Entity affordance tools are named ``{entity_name}_{affordance}`` per
# embodiment/tool_bridge.py::_resolve_tool_name. Body-level tools (pick_up)
# carry the entity in ``params.object``.
FAILURE_CLASS: dict[str, dict[str, Any]] = {
    "fire_pit": {
        "direct_failure_tools": frozenset({"fire_pit_touch"}),
        "body_failure_rules": (("infant_humanoid_pick_up", "object", "fire_pit"),),
        "direct_safe_tools": frozenset({"fire_pit_observe", "fire_pit_warm_self"}),
        # NEW (cradle_activation_fixes.md P2): positive-approach affordances
        # — distinct from the broader ``direct_safe_tools`` (which lumps
        # ``observe`` + ``warm_self`` together as non-failure choices).
        # These are the affordances that produce warming on fire_pit
        # WITHOUT triggering thermal_contact failure. Drives the
        # ``fire_approach_action_count`` descriptive corroborating metric
        # (NOT pre-reg gated) — the substrate hypothesis is that Arm B's
        # transferred positive edge ("fire = warm") yields APPROACH counts
        # at or above Arm A while the failure_class_action_count drops.
        "direct_approach_tools": frozenset({"fire_pit_warm_self"}),
    },
    "sharp_rock": {
        "direct_failure_tools": frozenset({"sharp_rock_touch"}),
        "body_failure_rules": (("infant_humanoid_pick_up", "object", "sharp_rock"),),
        "direct_safe_tools": frozenset({"sharp_rock_examine"}),
        # No proximity-approach analog for sharp_rock — the metric is 0
        # by construction for this scenario, but the field MUST exist so
        # compute_metrics doesn't KeyError. Asymmetric metric is by
        # design (scenario-specific positive edges).
        "direct_approach_tools": frozenset(),
    },
}

# Ablation env vars (resumed run only — prior runs always have all bio
# mechanisms ON so substrate accumulates normally).
ARM_ENV: dict[str, dict[str, str]] = {
    "A": {},
    "B": {},
    "C": {},
    "B-wire-a-off": {"MAXIM_DISABLE_CLUSTER_BIAS_ANNOTATION": "1"},
    "B-wire-1-off": {"MAXIM_DISABLE_VARIANCE_ANNOTATION": "1"},
    "B-nac-bias-off": {"MAXIM_NAC_REWARD_BIAS_DISABLED": "1"},
}

# Goal strings push the narrator toward the target entity. The cradle arc's
# keyword scorer still routes to ``cradle``; the entity manifest at the
# matching phase is what surfaces the right items. See
# ``docs/experiments/protocols/37_cross_session_graduation_reproduction.md``
# for the fixture-vs-narrator decision.
SCENARIO_GOAL: dict[str, str] = {
    "fire_pit": "cradle infant explores the warm room with the fire pit",
    "sharp_rock": "cradle infant explores the play area with sharp rock and blanket",
}
PEACEFUL_GOAL = "cradle infant explores the puzzle door and button"

# Turn-boundary tools — actions before each say/respond are grouped into one
# turn. Documented in the protocol as the implementation operationalization
# of the pre-reg's "per-turn" language (the raw actions.jsonl has no turn
# field; report.json's ``turns`` counts deliberation cycles, not actions).
TURN_BOUNDARY_TOOLS = frozenset({"say", "respond"})


# ─── Sim result + invocation ─────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class SimResult:
    """Parsed output of one sim run."""

    session_id: str
    data_home: Path
    report: dict[str, Any]
    actions: list[dict[str, Any]]


class CostCapExceeded(RuntimeError):
    """Raised when cumulative cost exceeds ``--cost-cap``."""


def run_one_sim(
    *,
    data_home: Path,
    goal: str,
    model: str,
    max_turns: int,
    resume_session: str | None = None,
    extra_env: dict[str, str] | None = None,
    mock: bool = False,
    mock_failure_count: int = 0,
    timeout_s: int = 1800,
) -> SimResult:
    """Run one sim and return the parsed report + actions.

    Real mode: subprocess ``maxim --sim "<goal>" ...``.
    Mock mode: writes synthetic report.json + actions.jsonl into ``data_home``.

    ``data_home`` must already exist. For resumed runs, the caller is
    responsible for seeding ``data_home`` with the prior session's state
    (typically via shutil.copytree from the prior arm's home).
    """
    if mock:
        return _mock_sim(data_home, goal, model, max_turns, resume_session, mock_failure_count)
    return _real_sim(data_home, goal, model, max_turns, resume_session, extra_env or {}, timeout_s)


_MAXIM_BIN_CACHE: str | None = None


def _resolve_maxim_binary() -> str:
    """Resolve the `maxim` CLI binary once per process so all subprocess
    calls hit the same executable (avoids the silent-version-drift trap
    where a globally-installed `maxim` shadows the dev venv)."""
    global _MAXIM_BIN_CACHE
    if _MAXIM_BIN_CACHE is None:
        resolved = shutil.which("maxim")
        if resolved is None:
            raise RuntimeError("Could not locate `maxim` on PATH. Activate the venv or pip-install.")
        _MAXIM_BIN_CACHE = resolved
    return _MAXIM_BIN_CACHE


def _real_sim(
    data_home: Path,
    goal: str,
    model: str,
    max_turns: int,
    resume_session: str | None,
    extra_env: dict[str, str],
    timeout_s: int,
) -> SimResult:
    """Subprocess maxim --sim, parse the resulting report.json + actions.jsonl."""
    env = os.environ.copy()
    env["MAXIM_DATA_HOME"] = str(data_home)
    env.update(extra_env)

    # Symlink the models cache so resumed runs don't re-download GGUFs.
    # Per CLAUDE.md "Parallel sessions use worktrees" — sharing model weights
    # across MAXIM_DATA_HOMEs is safe (binary cache, not substrate state).
    src_models = Path(os.path.expanduser("~/.maxim/models"))
    models_link = data_home / "models"
    if src_models.exists() and not models_link.exists():
        try:
            models_link.symlink_to(src_models)
        except OSError:
            pass

    cmd = [
        _resolve_maxim_binary(),
        "--sim",
        goal,
        "--embodiment",
        "bodies/infant_humanoid",
        "--interactive",
        "false",
        "--sim-max-turns",
        str(max_turns),
    ]
    # ``--language-model`` is INTENTIONALLY omitted: passing it forces the
    # sub-sim into local-llama-cpp mode (spawn-the-profile-locally),
    # overriding peer.yml routing even when role detects as ``peer``. For
    # Exp 37 we want every sub-sim to call the shared leader (so the
    # post-A substrate snapshot resumes against the SAME model and the
    # variance-survival rule isn't polluted by per-sandbox model
    # downloads). The harness's ``--model`` flag is preserved as a
    # JSONL metadata field; the served model name is discovered by
    # ``_MaximPeerBackend.warmup()`` and surfaced in the report via
    # PR #325's served-model substitution.
    if resume_session is not None:
        cmd.extend(["--resume-sim", resume_session])

    log_dir = data_home / "harness_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"sim_{int(time.time() * 1000)}.log"

    try:
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        log_path.write_text(f"TIMEOUT after {timeout_s}s\nstdout:\n{exc.stdout or ''}\nstderr:\n{exc.stderr or ''}")
        raise RuntimeError(
            f"maxim --sim timed out after {timeout_s}s "
            f"(resume={resume_session!r}, home={data_home}). Full log: {log_path}"
        ) from exc

    # Persist full logs so an exit-nonzero error message doesn't truncate the root cause.
    log_path.write_text(f"exit={proc.returncode}\ncmd={cmd!r}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "")[-1500:]
        raise RuntimeError(
            f"maxim --sim exited {proc.returncode} (resume={resume_session!r}, home={data_home}). "
            f"Full log: {log_path}\n{tail}"
        )

    return _load_latest_session(data_home, exclude=resume_session)


def _load_latest_session(data_home: Path, *, exclude: str | None) -> SimResult:
    """Pick the newest report.json under ``data_home/sim_reports/`` that isn't
    the excluded (resumed-prior) session.

    Exclude match is on the exact session-id directory name, NOT a
    substring of the full path — substring matching is fragile when a
    session_id is itself a substring of another. See
    ``simulation/report.py::save_action_log`` for the actions.jsonl
    Stage-0b header contract: the first line is
    ``{"_record_kind": "header", "_format_version": ...}`` and consumers
    MUST skip it before treating per-line records as actions.
    """
    reports_dir = data_home / "sim_reports"
    if not reports_dir.exists():
        raise RuntimeError(f"sim_reports/ missing under {data_home}")
    candidates = sorted(
        (p for p in reports_dir.glob("*/report.json") if exclude is None or p.parent.name != exclude),
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        raise RuntimeError(f"No new report.json under {reports_dir} (exclude={exclude!r})")
    report_path = candidates[-1]
    session_id = report_path.parent.name
    report = json.loads(report_path.read_text())
    actions_path = report_path.parent / "actions.jsonl"
    actions: list[dict[str, Any]] = []
    if actions_path.exists():
        for line in actions_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                # Tolerate trailing partial writes; surface count mismatch later.
                continue
            # Stage-0b header skip — must precede per-action interpretation.
            if isinstance(parsed, dict) and parsed.get("_record_kind") == "header":
                continue
            actions.append(parsed)
    return SimResult(
        session_id=session_id,
        data_home=data_home,
        report=report,
        actions=actions,
    )


def _mock_sim(
    data_home: Path,
    goal: str,
    model: str,
    max_turns: int,
    resume_session: str | None,
    mock_failure_count: int,
) -> SimResult:
    """Write synthetic artifacts for smoke testing — no subprocess, no LLM."""
    session_id = f"mock_{int(time.time() * 1000)}_{os.urandom(2).hex()}"
    session_dir = data_home / "sim_reports" / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    if "fire pit" in goal:
        scenario: str | None = "fire_pit"
        failure_tool = "fire_pit_touch"
        safe_tool = "fire_pit_observe"
    elif "sharp rock" in goal:
        scenario = "sharp_rock"
        failure_tool = "sharp_rock_touch"
        safe_tool = "sharp_rock_examine"
    else:
        scenario = None
        failure_tool = "infant_humanoid_use"
        safe_tool = "examine"

    # Interleave failure-class + safe + a respond() per turn so the harness
    # exercises both the per-action detector and the turn-boundary binner.
    actions: list[dict[str, Any]] = []
    base_ts = time.time()
    for turn in range(max_turns):
        if scenario is not None and turn < mock_failure_count:
            tool, params = failure_tool, {}
        elif scenario is not None:
            tool, params = safe_tool, {}
        else:
            tool, params = safe_tool, {}
        actions.append(
            {
                "timestamp": base_ts + turn * 2.0,
                "tool": tool,
                "params": params,
                "success": True,
                "output": "{}",
                "error": None,
                "blocked": False,
                "block_reason": None,
            }
        )
        actions.append(
            {
                "timestamp": base_ts + turn * 2.0 + 1.0,
                "tool": "respond",
                "params": {"message": f"turn {turn} done"},
                "success": True,
                "output": "{}",
                "error": None,
                "blocked": False,
                "block_reason": None,
            }
        )

    tool_usage: dict[str, int] = {}
    for a in actions:
        tool_usage[a["tool"]] = tool_usage.get(a["tool"], 0) + 1

    report = {
        "session_id": session_id,
        "goal": goal,
        "language_model": model,
        "duration_s": 1.0,
        "turns": max_turns,
        "finish_reason": "max_turns",
        "total_actions": len(actions),
        "tool_usage": tool_usage,
        "aut_memories_formed": 12,
        "aut_causal_links": 4,
        "cost_usd": 0.02,
        "total_input_tokens": 1000,
        "total_output_tokens": 200,
    }
    (session_dir / "report.json").write_text(json.dumps(report))
    with (session_dir / "actions.jsonl").open("w") as f:
        # Mirror simulation/report.py::save_action_log Stage-0b contract:
        # the first line is a header carrying _format_version + _record_kind.
        # Without this, the mock diverges from real and would mask the
        # header-skip bug the harness was patched for. See B2 in the
        # architecture review fold.
        f.write(
            json.dumps(
                {
                    "_format_version": "1.1",
                    "_record_kind": "header",
                    "session_id": session_id,
                }
            )
            + "\n"
        )
        for a in actions:
            f.write(json.dumps(a) + "\n")
    return SimResult(session_id=session_id, data_home=data_home, report=report, actions=actions)


# ─── Metrics ─────────────────────────────────────────────────────────────


def compute_metrics(actions: list[dict[str, Any]], scenario: str) -> dict[str, Any]:
    """Compute primary + corroborating metrics from one session's actions.

    Per pre-reg §"Failed action class": a turn counts 1 if ANY tool call in
    that turn matches the failure-class; 0 otherwise. We bin actions into
    turns by splitting on ``say``/``respond`` boundaries (each is the
    agent's textual close-out for a turn). Documented in the protocol as
    the operationalization of the pre-reg's "per-turn" language.

    Also returns a per-action rate as a robustness check independent of
    binning, plus the corroborating tool-class-diversity, affordance-
    preference, and time-to-safe-steady-state metrics.
    """
    rules = FAILURE_CLASS[scenario]
    direct_failure: frozenset[str] = rules["direct_failure_tools"]
    body_rules: tuple[tuple[str, str, str], ...] = rules["body_failure_rules"]
    direct_safe: frozenset[str] = rules["direct_safe_tools"]
    direct_approach: frozenset[str] = rules.get("direct_approach_tools", frozenset())

    def _is_failure(action: dict[str, Any]) -> bool:
        tool = action.get("tool")
        if tool in direct_failure:
            return True
        params = action.get("params") or {}
        for body_tool, key, val in body_rules:
            if tool == body_tool and params.get(key) == val:
                return True
        return False

    def _is_safe_on_target(action: dict[str, Any]) -> bool:
        return action.get("tool") in direct_safe

    def _is_approach(action: dict[str, Any]) -> bool:
        return action.get("tool") in direct_approach

    # Per-action counts (robustness check).
    failure_actions = sum(1 for a in actions if _is_failure(a))
    safe_target_actions = sum(1 for a in actions if _is_safe_on_target(a))
    approach_actions = sum(1 for a in actions if _is_approach(a))
    total_actions = len(actions) or 1
    per_action_rate = failure_actions / total_actions

    # Per-turn binning: split on say/respond. Each bucket is one turn.
    per_turn_failure: list[int] = []
    per_turn_safe: list[int] = []
    bucket_failure = 0
    bucket_safe = 0
    saw_any = False
    for action in actions:
        if _is_failure(action):
            bucket_failure = 1
        if _is_safe_on_target(action):
            bucket_safe = 1
        saw_any = True
        if action.get("tool") in TURN_BOUNDARY_TOOLS:
            per_turn_failure.append(bucket_failure)
            per_turn_safe.append(bucket_safe)
            bucket_failure = 0
            bucket_safe = 0
            saw_any = False
    # Tail bucket if the session ended without a final say/respond.
    if saw_any:
        per_turn_failure.append(bucket_failure)
        per_turn_safe.append(bucket_safe)

    total_turns = len(per_turn_failure) or 1
    per_turn_rate = sum(per_turn_failure) / total_turns

    # Time-to-safe-steady-state: first turn at which 3 consecutive
    # failure-class-free turns begin.
    steady_state_turn: int | None = None
    run_len = 0
    for i, v in enumerate(per_turn_failure):
        if v == 0:
            run_len += 1
            if run_len >= 3:
                steady_state_turn = i - 2
                break
        else:
            run_len = 0

    # Corroborating: affordance preference (safe vs failed on-target).
    on_target_total = safe_target_actions + failure_actions
    safe_fraction = safe_target_actions / on_target_total if on_target_total else 0.0

    return {
        "primary_metric_repeat_failure_action_rate": per_turn_rate,
        "per_action_failure_rate": per_action_rate,
        "failure_class_action_count": failure_actions,
        "failure_class_actions_per_turn": per_turn_failure,
        "turn_count_binned": len(per_turn_failure),
        "affordance_preference_safe_count": safe_target_actions,
        "affordance_preference_failed_count": failure_actions,
        "affordance_preference_safe_fraction": safe_fraction,
        "time_to_safe_steady_state_turns": steady_state_turn,
        # NEW (cradle_activation_fixes.md P2): positive-approach corroborating
        # metric — DESCRIPTIVE only, NOT pre-reg gated. Pairs with
        # ``failure_class_action_count`` to test the substrate-transfer
        # claim's POSITIVE edge (B should approach as often as A while
        # touching less). For sharp_rock the value is structurally 0
        # (no approach affordance in that scenario by design).
        "fire_approach_action_count": approach_actions,
    }


def build_record(
    sim: SimResult,
    *,
    trial_pair_id: int,
    arm: str,
    scenario: str | None,
    prior_session: str | None,
    seed: int,
    model: str,
    versions: dict[str, str],
) -> dict[str, Any]:
    """Build one JSONL record. Schema frozen at SCHEMA_VERSION."""
    metrics: dict[str, Any] = compute_metrics(sim.actions, scenario) if scenario else {}
    return {
        # Project-wide persistence-format envelope (CLAUDE.md _format_version invariant).
        # ``_format_version`` is the canonical key; ``schema_version`` mirrors it as
        # the experiment-scoped semantic equivalent for analyzer-side branching.
        # Both stay in lockstep — bump together; ``_format_version`` is what
        # downstream re-runners + universal tooling reads.
        "_format_version": SCHEMA_VERSION,
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT_ID,
        "harness_version": HARNESS_VERSION,
        "trial_pair_id": trial_pair_id,
        "arm": arm,
        "scenario": scenario,
        "session_id": sim.session_id,
        "prior_session_id": prior_session,
        "data_home": str(sim.data_home),
        "seed": seed,
        "model": model,
        "version_info": versions,
        "turns": sim.report.get("turns", 0),
        "finish_reason": sim.report.get("finish_reason", ""),
        "duration_s": sim.report.get("duration_s", 0.0),
        "cost_usd": float(sim.report.get("cost_usd", 0.0)),
        "total_input_tokens": sim.report.get("total_input_tokens", 0),
        "total_output_tokens": sim.report.get("total_output_tokens", 0),
        "tool_usage": sim.report.get("tool_usage", {}),
        "tool_class_diversity": len(sim.report.get("tool_usage", {})),
        "aut_memories_formed": sim.report.get("aut_memories_formed", 0),
        "aut_causal_links": sim.report.get("aut_causal_links", 0),
        "wall_clock_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **metrics,
    }


# ─── Versioning + cost guard ─────────────────────────────────────────────


def _capture_versions() -> dict[str, str]:
    """Snapshot reproducibility-pin fields. Embedded in every record."""
    out: dict[str, str] = {"harness_version": HARNESS_VERSION}
    try:
        from maxim import get_version_info  # type: ignore[import-untyped]

        out.update(get_version_info())
    except Exception as exc:  # pragma: no cover — defensive
        out["version_info_error"] = repr(exc)
    try:
        from importlib.metadata import PackageNotFoundError, version as _pkg_version

        try:
            out["sentence_transformers_version"] = _pkg_version("sentence-transformers")
        except PackageNotFoundError:
            out["sentence_transformers_version"] = "not-installed"
    except Exception as exc:  # pragma: no cover — defensive
        out["sentence_transformers_error"] = repr(exc)
    return out


def _check_cost(cumulative: float, cap: float) -> None:
    """Per-record cap check — catches single-arm cost balloons that exceed
    the cap mid-trial-pair. The pair-boundary check in ``run_benchmark``
    is the primary defense; this is the safety net for variance shocks."""
    if cumulative > cap:
        raise CostCapExceeded(f"Cumulative cost ${cumulative:.2f} exceeds cap ${cap:.2f}; aborting.")


def _load_existing_record_keys(out_path: Path) -> set[tuple[str, int, str, str | None]]:
    """Build the set of (experiment, trial_pair_id, arm, scenario) tuples
    already in ``out_path`` so the orchestrator can refuse to re-emit."""
    if not out_path.exists():
        return set()
    keys: set[tuple[str, int, str, str | None]] = set()
    for line in out_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        keys.add(
            (
                rec.get("experiment", ""),
                int(rec.get("trial_pair_id", -1)),
                rec.get("arm", ""),
                rec.get("scenario"),
            )
        )
    return keys


def _assert_failure_class_matches_yaml(scenario: str) -> None:
    """Cross-check FAILURE_CLASS rules against the live cradle YAMLs at
    startup so a YAML affordance rename (touch → poke) loudly fails the
    harness instead of silently zeroing the primary metric.

    Best-effort: if the YAML loader isn't importable, log a warning but
    don't block the harness — the mock smoke path doesn't need bio import.
    """
    try:
        from maxim.embodiment.component_registry import ComponentRegistry  # type: ignore[import-untyped]
    except Exception:
        return  # Smoke-test contexts may not have full bio stack.

    rules = FAILURE_CLASS[scenario]
    expected_affordances: set[str] = set()
    # Include direct_approach_tools so a rename of ``warm_self`` in the
    # YAML fails loudly instead of silently zeroing the corroborating
    # metric (cradle_activation_fixes.md P2 invariant).
    declared = (
        rules["direct_failure_tools"] | rules["direct_safe_tools"] | rules.get("direct_approach_tools", frozenset())
    )
    for tname in declared:
        # tool names are ``{entity_name}_{affordance}``; entity_name is the
        # scenario itself for these direct-affordance tools.
        prefix = f"{scenario}_"
        if tname.startswith(prefix):
            expected_affordances.add(tname[len(prefix) :])

    try:
        info = ComponentRegistry().get_info(f"items/cradle_{scenario}")
    except Exception:
        return  # Component not registered in this environment.
    source_path = getattr(info, "source_path", None)
    if not source_path:
        return

    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError:
        return  # pyyaml not installed in this env.

    try:
        with open(source_path) as f:
            spec = yaml.safe_load(f) or {}
    except OSError:
        return

    entity_spec = (spec or {}).get("entity", {}) if isinstance(spec, dict) else {}
    actual_affordances: set[str] = set()
    for mod in (entity_spec.get("modulators") or {}).values():
        for aff_name in mod.get("affordances") or {}:
            actual_affordances.add(aff_name)

    missing = expected_affordances - actual_affordances
    if missing:
        raise RuntimeError(
            f"FAILURE_CLASS[{scenario!r}] references affordance(s) {sorted(missing)} "
            f"not present in {source_path} (declared: {sorted(actual_affordances)}). "
            f"Either the YAML was renamed or the harness rules are stale. "
            f"Update both in lockstep."
        )


# ─── Orchestration ───────────────────────────────────────────────────────


def run_benchmark(
    *,
    out_path: Path,
    workdir: Path,
    arms: tuple[str, ...],
    scenarios: tuple[str, ...],
    trials: int,
    model: str,
    cost_cap: float,
    max_turns: int,
    seed_base: int,
    mock: bool,
    cleanup_after_trial: bool = False,
) -> dict[str, Any]:
    """Drive paired trials across arms × scenarios. Returns a summary dict.

    ``cleanup_after_trial=True`` removes per-trial sandboxes once the
    trial's last record is committed to the JSONL. The default preserves
    sandboxes (matches the ``run_v1_phases.sh`` precedent) so a failed
    run is post-mortem-able; pass True for long batched runs where disk
    space matters (65 runs × ~200 MB ≈ 13 GB).
    """
    workdir.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    arms_set = set(arms)
    unknown = arms_set - set(ARMS)
    if unknown:
        raise ValueError(f"Unknown arm(s): {sorted(unknown)}; valid: {ARMS}")

    has_b_family = any(a == "B" or a.startswith("B-") for a in arms)
    needs_arm_a = "A" in arms_set or has_b_family

    # Pre-flight: assert FAILURE_CLASS rules still match the YAMLs (skipped
    # in smoke-test contexts where the bio stack isn't loadable).
    if not mock:
        for sc in scenarios:
            _assert_failure_class_matches_yaml(sc)

    # I1 (Exec) idempotency: refuse to re-emit a record whose key is
    # already present in the append-only JSONL. The pre-reg locks N=5
    # paired trials; silent duplication of trial_pair_id ∈ {1..5} from
    # a second invocation against the same --out would poison the
    # variance-survival analysis.
    existing_keys = _load_existing_record_keys(out_path)

    versions = _capture_versions()
    cumulative_cost = 0.0
    records_written = 0
    # Track per-record cost for worst-case projection at pair boundaries.
    # Seed at 0.0 so the FIRST trial pair always passes the projection check
    # (no data yet to predict from); the per-record _check_cost safety net
    # still catches a single-arm cost balloon mid-pair. After pair 1 runs,
    # observed_max_record_cost ratchets up from actual data and the
    # projection becomes meaningful for subsequent pairs.
    observed_max_record_cost = 0.0

    # Skip patterns for substrate copy: omit live filelock files that
    # may carry stale PIDs from the prior run. Models cache symlink is
    # preserved by symlinks=True at copytree level.
    copy_ignore = shutil.ignore_patterns("*.lock", "*.lock.tmp")

    def _emit(rec: dict[str, Any]) -> None:
        nonlocal cumulative_cost, records_written, observed_max_record_cost
        key = (rec["experiment"], rec["trial_pair_id"], rec["arm"], rec.get("scenario"))
        if key in existing_keys:
            raise RuntimeError(
                f"Refusing to re-emit existing record key {key} into {out_path}. "
                "Re-runs MUST use a different --out file or clear the existing one."
            )
        existing_keys.add(key)
        out_f.write(json.dumps(rec) + "\n")
        out_f.flush()
        records_written += 1
        cumulative_cost += rec["cost_usd"]
        observed_max_record_cost = max(observed_max_record_cost, rec["cost_usd"])
        _check_cost(cumulative_cost, cost_cap)

    def _project_pair_cost(needs_arm_c_prior: bool) -> float:
        """Worst-case cost of the next trial pair before it starts."""
        target_arm_count = sum(1 for a in arms if a == "A" or a == "B" or a.startswith("B-") or a == "C")
        target_cost = observed_max_record_cost * target_arm_count
        prior_cost = observed_max_record_cost if needs_arm_c_prior else 0.0
        return target_cost + prior_cost

    with out_path.open("a") as out_f:
        for trial_id in range(1, trials + 1):
            seed = seed_base + trial_id
            arm_c_prior: SimResult | None = None  # shared across scenarios within a trial

            for scenario in scenarios:
                # C3 fold: pre-pair cost projection — abort BETWEEN pairs
                # rather than mid-pair so the analyzer never sees a half-
                # written trial. The per-record _check_cost is the safety
                # net for single-arm cost balloons.
                needs_c_prior_this_pair = "C" in arms_set and arm_c_prior is None
                projected = cumulative_cost + _project_pair_cost(needs_c_prior_this_pair)
                if projected > cost_cap:
                    raise CostCapExceeded(
                        f"Projected cost ${projected:.2f} (cumulative ${cumulative_cost:.2f} + "
                        f"projected pair ${projected - cumulative_cost:.2f}) would exceed cap "
                        f"${cost_cap:.2f}. Aborting cleanly between trial pairs — "
                        f"trial {trial_id} scenario {scenario!r} NOT started."
                    )

                arm_a_result: SimResult | None = None

                # Arm A target (also = prior for B-family).
                if needs_arm_a:
                    a_home = workdir / f"trial{trial_id:03d}_{scenario}_A"
                    a_home.mkdir(parents=True, exist_ok=True)
                    arm_a_result = run_one_sim(
                        data_home=a_home,
                        goal=SCENARIO_GOAL[scenario],
                        model=model,
                        max_turns=max_turns,
                        resume_session=None,
                        mock=mock,
                        mock_failure_count=4,
                    )
                    if "A" in arms_set:
                        _emit(
                            build_record(
                                arm_a_result,
                                trial_pair_id=trial_id,
                                arm="A",
                                scenario=scenario,
                                prior_session=None,
                                seed=seed,
                                model=model,
                                versions=versions,
                            )
                        )

                # B-family: copy Arm A's data_home, resume, apply env var.
                for arm in [a for a in arms if a == "B" or a.startswith("B-")]:
                    if arm_a_result is None:  # pragma: no cover — guarded above
                        raise RuntimeError(f"{arm} requires Arm A as prior")
                    resume_home = workdir / f"trial{trial_id:03d}_{scenario}_{arm}"
                    if resume_home.exists():
                        shutil.rmtree(resume_home)
                    shutil.copytree(
                        arm_a_result.data_home,
                        resume_home,
                        symlinks=True,
                        ignore=copy_ignore,
                    )
                    sim = run_one_sim(
                        data_home=resume_home,
                        goal=SCENARIO_GOAL[scenario],
                        model=model,
                        max_turns=max_turns,
                        resume_session=arm_a_result.session_id,
                        extra_env=ARM_ENV[arm],
                        mock=mock,
                        mock_failure_count=(1 if arm == "B" else 2),
                    )
                    _emit(
                        build_record(
                            sim,
                            trial_pair_id=trial_id,
                            arm=arm,
                            scenario=scenario,
                            prior_session=arm_a_result.session_id,
                            seed=seed,
                            model=model,
                            versions=versions,
                        )
                    )

                # Arm C: build peaceful prior (shared across scenarios), then resume.
                if "C" in arms_set:
                    if arm_c_prior is None:
                        prior_home = workdir / f"trial{trial_id:03d}_C_prior"
                        prior_home.mkdir(parents=True, exist_ok=True)
                        arm_c_prior = run_one_sim(
                            data_home=prior_home,
                            goal=PEACEFUL_GOAL,
                            model=model,
                            max_turns=max_turns,
                            resume_session=None,
                            mock=mock,
                            mock_failure_count=0,
                        )
                        prior_cost = float(arm_c_prior.report.get("cost_usd", 0.0))
                        cumulative_cost += prior_cost
                        observed_max_record_cost = max(observed_max_record_cost, prior_cost)
                        _check_cost(cumulative_cost, cost_cap)
                    c_home = workdir / f"trial{trial_id:03d}_{scenario}_C"
                    if c_home.exists():
                        shutil.rmtree(c_home)
                    shutil.copytree(
                        arm_c_prior.data_home,
                        c_home,
                        symlinks=True,
                        ignore=copy_ignore,
                    )
                    sim = run_one_sim(
                        data_home=c_home,
                        goal=SCENARIO_GOAL[scenario],
                        model=model,
                        max_turns=max_turns,
                        resume_session=arm_c_prior.session_id,
                        mock=mock,
                        mock_failure_count=3,
                    )
                    _emit(
                        build_record(
                            sim,
                            trial_pair_id=trial_id,
                            arm="C",
                            scenario=scenario,
                            prior_session=arm_c_prior.session_id,
                            seed=seed,
                            model=model,
                            versions=versions,
                        )
                    )

            # End-of-trial sandbox cleanup (optional — opt-in to preserve
            # debuggability by default).
            if cleanup_after_trial:
                for path in workdir.glob(f"trial{trial_id:03d}_*"):
                    if path.is_dir():
                        shutil.rmtree(path, ignore_errors=True)

    return {
        "records_written": records_written,
        "cumulative_cost_usd": cumulative_cost,
        "out_path": str(out_path),
        "workdir": str(workdir),
    }


# ─── CLI ─────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Exp 37 cross-session graduation harness")
    parser.add_argument("--scenario", choices=["fire_pit", "sharp_rock", "both"], default="both")
    parser.add_argument(
        "--arms", default=",".join(ARMS), help=f"Comma-separated arms ({'|'.join(ARMS)}); default: all six"
    )
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--model", default="claude-sonnet")
    parser.add_argument("--out", type=Path, required=True, help="Append-only JSONL output path for per-trial records")
    parser.add_argument(
        "--cost-cap", type=float, default=20.0, help="Hard ceiling on cumulative cost (USD); default $20"
    )
    parser.add_argument("--sim-max-turns", type=int, default=12)
    parser.add_argument("--seed-base", type=int, default=42)
    parser.add_argument(
        "--workdir", type=Path, default=None, help="Parent dir for per-trial sandboxes (default: tempfile.mkdtemp)"
    )
    parser.add_argument(
        "--cleanup-after-trial",
        action="store_true",
        help="Delete per-trial sandboxes after each trial's records commit. "
        "Saves disk (65 runs × ~200 MB ≈ 13 GB at default) but kills post-mortem.",
    )
    parser.add_argument(
        "--mock-llm",
        action="store_true",
        help="Synthetic backend — for smoke tests only. NEVER use for graduation runs.",
    )
    args = parser.parse_args(argv)

    arms = tuple(a.strip() for a in args.arms.split(",") if a.strip())
    scenarios = SCENARIOS if args.scenario == "both" else (args.scenario,)
    workdir = args.workdir or Path(tempfile.mkdtemp(prefix="exp37_"))

    try:
        summary = run_benchmark(
            out_path=args.out,
            workdir=workdir,
            arms=arms,
            scenarios=scenarios,
            trials=args.trials,
            model=args.model,
            cost_cap=args.cost_cap,
            max_turns=args.sim_max_turns,
            seed_base=args.seed_base,
            mock=args.mock_llm,
            cleanup_after_trial=args.cleanup_after_trial,
        )
    except CostCapExceeded as exc:
        print(f"ABORT: {exc}", file=sys.stderr)
        return 3
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(
        f"Wrote {summary['records_written']} records to {summary['out_path']} "
        f"(workdir={summary['workdir']}, cost=${summary['cumulative_cost_usd']:.2f})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
