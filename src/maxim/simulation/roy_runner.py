"""Roy three-arm iteration runner — orchestrate the persona convergence crucible.

R3 of the Roy long-horizon harness build (see
``docs/plans/persona_convergence_crucible.md`` "Three-arm comparison"
and "What we record per Roy iteration"). Turns one Roy iteration —
priming + held-out test across three arms + pairwise substrate
divergence — into a single operator command.

Arm contract::

    | Arm | Substrate priming     | System prompt at test |
    |-----|-----------------------|-----------------------|
    | A   | Full Roy lifetime     | neutral               |
    | B   | Blank                 | persona-injected      |
    | C   | Blank                 | neutral               |

The runner:

1. Runs the priming curriculum (via R1's ``run_curriculum``) to grow
   arm A's substrate.
2. Runs the same held-out test scenario three times — one per arm —
   with arm A resumed from priming's final substrate, arms B and C
   starting blank.
3. Calls R2's :func:`maxim.analysis.substrate_diff.substrate_diff`
   for all three pairwise comparisons (A vs B, A vs C, B vs C).
4. Persists a :class:`RoyIterationResult` to
   ``~/.maxim/roy/<iteration_name>/result.json`` plus a
   ``summary.md`` so the operator can read the iteration outcome at
   a glance.

Iteration spec YAML (deliberately tiny)::

    name: roy-0-smoke
    description: methodology smoke test, not a real persona run
    embodiment: bodies/infant_humanoid     # default for every sim
    aut_mode: substrate-primary            # default; "llm-primary" also OK

    # Inline curriculum spec OR a path string pointing at a curriculum
    # yaml (relative to this spec's directory unless absolute).
    priming:
      name: roy-0-priming
      stages:
        - name: act1_warmup
          fixture: scenarios/cradle/warmup.yaml
          turns: 4

    # Held-out test scenario — same fixture for every arm.
    test_scenario:
      fixture: scenarios/cradle/warmup.yaml
      turns: 5

    # Per-arm overrides — substrate source + system_prompt slug.
    # ``substrate: from_priming`` resumes the priming curriculum's final
    # session id; ``substrate: blank`` starts fresh. ``system_prompt``
    # maps to the orchestrator ``persona`` field on
    # ``start_simulation_mode`` — "neutral" is the canonical no-injection
    # value; arm B passes a registered persona slug ("adversarial",
    # "cooperative", ...) to drive the prompt-injected baseline.
    arms:
      a: { substrate: from_priming, system_prompt: neutral }
      b: { substrate: blank,        system_prompt: adversarial }
      c: { substrate: blank,        system_prompt: neutral }

Fail-fast semantics: spec problems raise ``ValueError`` before any sim
runs. Priming failure aborts the iteration before any arm. Per-arm
failures are surfaced on the corresponding :class:`ArmResult`; pairwise
diffs are only computed between arms that produced a usable
``session_dir`` (others are marked ``available=false``).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class ArmResult:
    """One arm's outcome — successful or failed."""

    name: str  # "a" | "b" | "c"
    substrate: str  # "from_priming" | "blank"
    system_prompt: str
    resumed_from: str = ""  # session_id A resumed from (empty for B/C)
    session_id: str = ""
    session_dir: str = ""
    turns_run: int = 0
    finish_reason: str = ""
    duration_s: float = 0.0
    error: str | None = None


@dataclass
class RoyIterationResult:
    """Result of a full three-arm Roy iteration.

    ``pairwise_diffs`` keys are ``"a_vs_b"``, ``"a_vs_c"``, ``"b_vs_c"``
    and values are :func:`substrate_diff_to_json`-shaped dicts. Each
    pair entry carries an ``available`` flag inside its sub-dicts
    (NAc / EC / Hippocampus / ATL) so a partial-success iteration
    surfaces honestly rather than crashing on missing snapshots.
    """

    name: str
    spec_path: str
    description: str = ""
    priming: dict[str, Any] = field(default_factory=dict)  # CurriculumResult-shaped
    arms: dict[str, ArmResult] = field(default_factory=dict)
    pairwise_diffs: dict[str, dict[str, Any]] = field(default_factory=dict)
    total_duration_s: float = 0.0
    artifact_dir: str = ""
    aborted_at: str | None = None  # "priming" | "arm_a" | None


# ---------------------------------------------------------------------------
# Spec parsing
# ---------------------------------------------------------------------------


@dataclass
class _ArmSpec:
    name: str
    substrate: str  # "from_priming" | "blank"
    system_prompt: str


@dataclass
class _TestScenarioSpec:
    fixture: str
    turns: int


@dataclass
class _IterationSpec:
    name: str
    spec_path: Path
    description: str
    embodiment: str | None
    aut_mode: str
    priming_path: Path  # always a path on disk (inline specs are materialised to a tmp file)
    priming_inline: dict[str, Any] | None  # raw inline curriculum dict, if any
    test_scenario: _TestScenarioSpec
    arms: dict[str, _ArmSpec]


_DEFAULT_AUT_MODE = "llm-primary"
_VALID_AUT_MODES = ("llm-primary", "substrate-primary")
_VALID_SUBSTRATE_SOURCES = ("from_priming", "blank")
_ARM_KEYS = ("a", "b", "c")


def _parse_arm(key: str, raw: Any) -> _ArmSpec:
    if not isinstance(raw, dict):
        raise ValueError(f"Arm {key!r} must be a mapping, got {type(raw).__name__}")
    substrate = raw.get("substrate")
    if substrate not in _VALID_SUBSTRATE_SOURCES:
        raise ValueError(f"Arm {key!r} 'substrate' must be one of {_VALID_SUBSTRATE_SOURCES!r}, got {substrate!r}")
    sp = raw.get("system_prompt")
    if not isinstance(sp, str) or not sp.strip():
        raise ValueError(f"Arm {key!r} 'system_prompt' must be a non-empty string")
    return _ArmSpec(name=key, substrate=substrate, system_prompt=sp.strip())


def _parse_spec(spec_path: Path) -> _IterationSpec:
    """Load and validate an iteration YAML.

    Fails fast on any schema problem so the operator gets a precise
    message instead of a half-completed iteration.
    """
    import yaml

    spec_path = Path(spec_path).resolve()
    if not spec_path.exists():
        raise FileNotFoundError(f"Iteration spec not found: {spec_path}")

    with open(spec_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Iteration spec must be a YAML mapping, got {type(raw).__name__}")

    name = raw.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Iteration spec missing required string field 'name'")

    description = raw.get("description", "")
    if not isinstance(description, str):
        raise ValueError("'description' must be a string if set")

    embodiment = raw.get("embodiment")
    if embodiment is not None and not isinstance(embodiment, str):
        raise ValueError("'embodiment' must be a string if set")

    aut_mode = raw.get("aut_mode", _DEFAULT_AUT_MODE)
    if aut_mode not in _VALID_AUT_MODES:
        raise ValueError(f"'aut_mode' must be one of {_VALID_AUT_MODES!r}, got {aut_mode!r}")

    spec_dir = spec_path.parent

    # priming — either a path string or an inline curriculum dict.
    priming_raw = raw.get("priming")
    priming_inline: dict[str, Any] | None = None
    if isinstance(priming_raw, str):
        pp = Path(priming_raw)
        if not pp.is_absolute():
            pp = (spec_dir / pp).resolve()
        if not pp.exists():
            raise ValueError(f"Priming curriculum not found: {pp} (resolved relative to {spec_dir})")
        priming_path = pp
    elif isinstance(priming_raw, dict):
        # Validate the inline curriculum has the minimum shape; the
        # full validation runs when run_curriculum loads the
        # materialised file.
        if not isinstance(priming_raw.get("stages"), list) or not priming_raw["stages"]:
            raise ValueError("Inline 'priming.stages' must be a non-empty list")
        priming_inline = dict(priming_raw)
        # If the inline spec doesn't carry a name, inherit from iteration
        # so the resulting CurriculumResult.name is meaningful.
        priming_inline.setdefault("name", f"{name.strip()}-priming")
        # Inline specs are materialised next to the iteration spec so
        # relative fixture paths resolve the same way they would for an
        # external curriculum.
        priming_path = spec_dir / f".roy_inline_priming_{name.strip()}.yaml"
    else:
        raise ValueError("'priming' must be a path string or an inline curriculum mapping")

    test_raw = raw.get("test_scenario")
    if not isinstance(test_raw, dict):
        raise ValueError("'test_scenario' must be a mapping")
    fixture_raw = test_raw.get("fixture")
    if not isinstance(fixture_raw, str) or not fixture_raw.strip():
        raise ValueError("'test_scenario.fixture' must be a non-empty string")
    fixture_path = Path(fixture_raw)
    if not fixture_path.is_absolute():
        fixture_path = (spec_dir / fixture_path).resolve()
    if not fixture_path.exists():
        raise ValueError(f"test_scenario fixture not found: {fixture_path} (resolved relative to {spec_dir})")
    turns_raw = test_raw.get("turns")
    if not isinstance(turns_raw, int) or turns_raw <= 0:
        raise ValueError(f"'test_scenario.turns' must be a positive int, got {turns_raw!r}")
    test_scenario = _TestScenarioSpec(fixture=str(fixture_path), turns=turns_raw)

    arms_raw = raw.get("arms")
    if not isinstance(arms_raw, dict):
        raise ValueError("'arms' must be a mapping with keys 'a', 'b', 'c'")
    missing = [k for k in _ARM_KEYS if k not in arms_raw]
    if missing:
        raise ValueError(f"'arms' missing required key(s): {missing}")
    extra = [k for k in arms_raw if k not in _ARM_KEYS]
    if extra:
        raise ValueError(f"'arms' has unknown key(s): {extra} (expected only {list(_ARM_KEYS)})")
    arms = {k: _parse_arm(k, arms_raw[k]) for k in _ARM_KEYS}

    # Per the methodology, arm A is the only one that resumes priming;
    # B and C must be blank. Catching this here keeps misconfigured
    # specs from silently producing meaningless diffs.
    if arms["a"].substrate != "from_priming":
        raise ValueError("arms.a.substrate must be 'from_priming' (the lived-Roy arm)")
    if arms["b"].substrate != "blank":
        raise ValueError("arms.b.substrate must be 'blank' (the prompt-injected baseline)")
    if arms["c"].substrate != "blank":
        raise ValueError("arms.c.substrate must be 'blank' (the neutral baseline)")

    return _IterationSpec(
        name=name.strip(),
        spec_path=spec_path,
        description=description,
        embodiment=embodiment,
        aut_mode=aut_mode,
        priming_path=priming_path,
        priming_inline=priming_inline,
        test_scenario=test_scenario,
        arms=arms,
    )


# ---------------------------------------------------------------------------
# Stage helpers
# ---------------------------------------------------------------------------


def _materialise_inline_priming(spec: _IterationSpec) -> Path:
    """Write the inline priming curriculum to a sibling YAML so
    ``run_curriculum`` can load it the same way it loads external files.

    Returns the materialised path. Caller is responsible for deleting it
    after the curriculum finishes — we keep it on disk during the run
    so any error trace points at a real file the operator can inspect.
    """
    if spec.priming_inline is None:
        return spec.priming_path
    import yaml

    spec.priming_path.write_text(yaml.safe_dump(spec.priming_inline, sort_keys=False))
    return spec.priming_path


def _build_test_kwargs(
    spec: _IterationSpec,
    arm: _ArmSpec,
    resume_from: str | None,
) -> dict[str, Any]:
    """Build the kwargs for one arm's test-scenario sim.

    Same fixture for every arm; persona varies (system_prompt slug);
    resume_session set only when ``arm.substrate == "from_priming"``.
    """
    kwargs: dict[str, Any] = {
        "goal": f"roy:{spec.name}:arm_{arm.name}",
        "persona": arm.system_prompt,
        "max_turns": spec.test_scenario.turns,
        "aut_mode": spec.aut_mode,
        "entity_ref": spec.embodiment,
        "fixture_path": spec.test_scenario.fixture,
    }
    if arm.substrate == "from_priming" and resume_from:
        kwargs["resume_session"] = resume_from
    return kwargs


def _run_arm(
    arm: _ArmSpec,
    kwargs: dict[str, Any],
    sim_runner: Callable[..., Any],
) -> ArmResult:
    record = ArmResult(
        name=arm.name,
        substrate=arm.substrate,
        system_prompt=arm.system_prompt,
        resumed_from=kwargs.get("resume_session", "") or "",
    )
    start = time.time()
    try:
        sim_result = sim_runner(**kwargs)
    except Exception as e:
        record.duration_s = round(time.time() - start, 2)
        record.error = f"{type(e).__name__}: {e}"
        record.finish_reason = "error"
        logger.error("Roy arm %r raised: %s", arm.name, record.error, exc_info=True)
        return record

    record.session_id = getattr(sim_result, "session_id", "") or ""
    record.session_dir = getattr(sim_result, "session_dir", "") or ""
    record.turns_run = int(getattr(sim_result, "turns", 0) or 0)
    record.finish_reason = getattr(sim_result, "finish_reason", "") or ""
    record.duration_s = round(time.time() - start, 2)
    if not record.session_id:
        record.error = "arm returned empty session_id; diff against this arm will be unavailable"
        logger.warning("Roy arm %r: %s", arm.name, record.error)
    return record


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_roy_iteration(
    spec_path: Path | str,
    *,
    sim_runner: Callable[..., Any] | None = None,
    curriculum_fn: Callable[..., Any] | None = None,
    artifact_root: Path | str | None = None,
) -> RoyIterationResult:
    """Run one Roy three-arm iteration.

    Parameters
    ----------
    spec_path:
        Path to the iteration YAML.
    sim_runner:
        Injection seam for tests — defaults to
        :func:`maxim.simulation.orchestrator.start_simulation_mode`.
        Must accept the same kwargs and return a
        :class:`SimulationResult`-shaped object.
    curriculum_fn:
        Injection seam for tests — defaults to
        :func:`maxim.simulation.curriculum_runner.run_curriculum`.
        Receives ``(priming_path, sim_runner=sim_runner)`` and must
        return a ``CurriculumResult``-shaped object.
    artifact_root:
        Override for the iteration artifact directory parent. Defaults
        to ``~/.maxim/roy``. Tests use this to point at a tmp dir.
    """
    from maxim.analysis.substrate_diff import substrate_diff, substrate_diff_to_json
    from maxim.utils.atomic_io import atomic_write_text, atomic_write_json
    from maxim.utils.format_version import with_format_version
    from maxim.utils.paths import data_home

    spec = _parse_spec(Path(spec_path))

    if sim_runner is None:
        from maxim.simulation.orchestrator import start_simulation_mode

        sim_runner = start_simulation_mode
    if curriculum_fn is None:
        from maxim.simulation.curriculum_runner import run_curriculum

        curriculum_fn = run_curriculum

    if artifact_root is None:
        artifact_dir = data_home() / "roy" / spec.name
    else:
        artifact_dir = Path(artifact_root) / spec.name
    artifact_dir.mkdir(parents=True, exist_ok=True)

    result = RoyIterationResult(
        name=spec.name,
        spec_path=str(spec.spec_path),
        description=spec.description,
        artifact_dir=str(artifact_dir),
    )
    overall_start = time.time()

    # ── Priming ────────────────────────────────────────────────────────
    priming_path = _materialise_inline_priming(spec)
    try:
        priming_result = curriculum_fn(priming_path, sim_runner=sim_runner)
    except Exception as e:
        result.aborted_at = "priming"
        result.priming = {"error": f"{type(e).__name__}: {e}"}
        result.total_duration_s = round(time.time() - overall_start, 2)
        logger.error("Roy iteration %r aborted during priming: %s", spec.name, e, exc_info=True)
        _persist_result(result, artifact_dir, atomic_write_json, atomic_write_text, with_format_version)
        return result
    finally:
        # Clean up the materialised inline file so the operator's working
        # tree doesn't accumulate transient artifacts. External
        # priming_path files are left alone.
        if spec.priming_inline is not None:
            try:
                priming_path.unlink(missing_ok=True)
            except OSError:
                pass

    result.priming = _curriculum_result_to_dict(priming_result)
    if getattr(priming_result, "aborted_at", None) is not None:
        result.aborted_at = "priming"
        result.total_duration_s = round(time.time() - overall_start, 2)
        logger.error(
            "Roy iteration %r: priming aborted at stage %r; skipping arms",
            spec.name,
            priming_result.aborted_at,
        )
        _persist_result(result, artifact_dir, atomic_write_json, atomic_write_text, with_format_version)
        return result

    priming_session_id = getattr(priming_result, "final_session_id", "") or ""
    if not priming_session_id:
        result.aborted_at = "priming"
        result.total_duration_s = round(time.time() - overall_start, 2)
        logger.error("Roy iteration %r: priming produced no usable session_id", spec.name)
        _persist_result(result, artifact_dir, atomic_write_json, atomic_write_text, with_format_version)
        return result

    # ── Arms ────────────────────────────────────────────────────────────
    for arm_key in _ARM_KEYS:
        arm_spec = spec.arms[arm_key]
        kwargs = _build_test_kwargs(spec, arm_spec, resume_from=priming_session_id)
        logger.info(
            "Roy %r arm %s: substrate=%s system_prompt=%s resume=%s turns=%d",
            spec.name,
            arm_key,
            arm_spec.substrate,
            arm_spec.system_prompt,
            kwargs.get("resume_session", "<none>"),
            spec.test_scenario.turns,
        )
        result.arms[arm_key] = _run_arm(arm_spec, kwargs, sim_runner)

    # ── Pairwise diffs ─────────────────────────────────────────────────
    pairs = (("a", "b"), ("a", "c"), ("b", "c"))
    for left, right in pairs:
        key = f"{left}_vs_{right}"
        dir_l = result.arms[left].session_dir
        dir_r = result.arms[right].session_dir
        if not dir_l or not dir_r:
            result.pairwise_diffs[key] = {
                "available": False,
                "reason": f"missing session_dir for arm(s): "
                f"{','.join(side for side, d in ((left, dir_l), (right, dir_r)) if not d)}",
            }
            continue
        try:
            diff = substrate_diff(dir_l, dir_r)
            payload = substrate_diff_to_json(diff)
            payload["available"] = True
            result.pairwise_diffs[key] = payload
        except Exception as e:
            result.pairwise_diffs[key] = {
                "available": False,
                "reason": f"substrate_diff raised: {type(e).__name__}: {e}",
            }
            logger.warning("Roy %r pair %s: substrate_diff raised: %s", spec.name, key, e)

    result.total_duration_s = round(time.time() - overall_start, 2)

    _persist_result(result, artifact_dir, atomic_write_json, atomic_write_text, with_format_version)

    logger.info(
        "Roy iteration %r complete: %d/3 arms produced sessions, %d/3 pairwise diffs available, duration=%.1fs",
        spec.name,
        sum(1 for a in result.arms.values() if a.session_id),
        sum(1 for d in result.pairwise_diffs.values() if d.get("available")),
        result.total_duration_s,
    )
    return result


# ---------------------------------------------------------------------------
# Persistence + summary
# ---------------------------------------------------------------------------


def _curriculum_result_to_dict(cr: Any) -> dict[str, Any]:
    """Serialize a ``CurriculumResult`` (R1) to a JSON-ready dict.

    We accept duck-typed inputs because the test seam injects a fake
    curriculum_fn that may return a stand-in dataclass.
    """
    stages_out: list[dict[str, Any]] = []
    for s in getattr(cr, "stages", []) or []:
        stages_out.append(
            {
                "name": getattr(s, "name", ""),
                "source_kind": getattr(s, "source_kind", ""),
                "source_value": getattr(s, "source_value", ""),
                "turns_requested": getattr(s, "turns_requested", 0),
                "session_id": getattr(s, "session_id", ""),
                "session_dir": getattr(s, "session_dir", ""),
                "turns_run": getattr(s, "turns_run", 0),
                "finish_reason": getattr(s, "finish_reason", ""),
                "duration_s": getattr(s, "duration_s", 0.0),
                "error": getattr(s, "error", None),
            }
        )
    return {
        "name": getattr(cr, "name", ""),
        "spec_path": getattr(cr, "spec_path", ""),
        "stages": stages_out,
        "final_session_id": getattr(cr, "final_session_id", ""),
        "final_session_dir": getattr(cr, "final_session_dir", ""),
        "total_duration_s": getattr(cr, "total_duration_s", 0.0),
        "aborted_at": getattr(cr, "aborted_at", None),
    }


def _arm_to_dict(a: ArmResult) -> dict[str, Any]:
    return {
        "name": a.name,
        "substrate": a.substrate,
        "system_prompt": a.system_prompt,
        "resumed_from": a.resumed_from,
        "session_id": a.session_id,
        "session_dir": a.session_dir,
        "turns_run": a.turns_run,
        "finish_reason": a.finish_reason,
        "duration_s": a.duration_s,
        "error": a.error,
    }


def _result_to_dict(r: RoyIterationResult) -> dict[str, Any]:
    return {
        "name": r.name,
        "spec_path": r.spec_path,
        "description": r.description,
        "priming": r.priming,
        "arms": {k: _arm_to_dict(v) for k, v in r.arms.items()},
        "pairwise_diffs": r.pairwise_diffs,
        "total_duration_s": r.total_duration_s,
        "artifact_dir": r.artifact_dir,
        "aborted_at": r.aborted_at,
    }


def _persist_result(
    result: RoyIterationResult,
    artifact_dir: Path,
    atomic_write_json: Callable[..., None],
    atomic_write_text: Callable[..., None],
    with_format_version: Callable[..., dict[str, Any]],
) -> None:
    payload = with_format_version(_result_to_dict(result))
    atomic_write_json(str(artifact_dir / "result.json"), payload)
    atomic_write_text(str(artifact_dir / "summary.md"), _format_summary(result))


def _format_summary(r: RoyIterationResult) -> str:
    """Render a one-screen markdown summary of the iteration."""
    lines: list[str] = []
    lines.append(f"# Roy iteration `{r.name}`")
    if r.description:
        lines.append("")
        lines.append(r.description)
    lines.append("")
    lines.append(f"- spec: `{r.spec_path}`")
    lines.append(f"- artifact_dir: `{r.artifact_dir}`")
    lines.append(f"- total_duration_s: {r.total_duration_s}")
    if r.aborted_at:
        lines.append(f"- **aborted_at:** `{r.aborted_at}`")
    lines.append("")

    # Priming
    lines.append("## Priming")
    priming = r.priming or {}
    if "error" in priming:
        lines.append(f"- error: `{priming['error']}`")
    else:
        lines.append(f"- name: `{priming.get('name', '')}`")
        lines.append(f"- final_session_id: `{priming.get('final_session_id', '')}`")
        lines.append(f"- stages: {len(priming.get('stages', []) or [])}")
        lines.append(f"- aborted_at: `{priming.get('aborted_at') or '<none>'}`")

    # Arms
    lines.append("")
    lines.append("## Arms")
    for key in _ARM_KEYS:
        arm = r.arms.get(key)
        if arm is None:
            lines.append(f"- **{key}:** not run")
            continue
        lines.append(
            f"- **{key}** (substrate={arm.substrate}, system_prompt={arm.system_prompt}): "
            f"session_id=`{arm.session_id or '<none>'}` "
            f"turns={arm.turns_run} finish_reason={arm.finish_reason or '<none>'} "
            f"duration={arm.duration_s}s" + (f"  ⚠ error: {arm.error}" if arm.error else "")
        )

    # Diffs
    lines.append("")
    lines.append("## Pairwise substrate divergence")
    for key in ("a_vs_b", "a_vs_c", "b_vs_c"):
        diff = r.pairwise_diffs.get(key)
        if not diff:
            lines.append(f"- **{key}:** not computed")
            continue
        if not diff.get("available", True):
            lines.append(f"- **{key}:** unavailable — {diff.get('reason', 'unknown reason')}")
            continue
        nac = diff.get("nac", {})
        ec = diff.get("ec", {})
        hipp = diff.get("hippocampus", {})
        atl = diff.get("atl", {})
        lines.append(f"- **{key}:**")
        if nac.get("available"):
            lines.append(
                f"  - NAc: reward_bias L2={nac.get('reward_bias_l2', 0.0):.4f}  "
                f"causal_link Δ={nac.get('causal_link_count_delta', 0):+d}"
            )
        if ec.get("available"):
            lines.append(f"  - EC: nodes Δ={ec.get('node_count_delta', 0):+d}")
        if hipp.get("available"):
            v = hipp.get("valence", {})
            lines.append(
                f"  - Hippocampus: episodes Δ={hipp.get('episode_count_delta', 0):+d}  "
                f"valence KS={v.get('ks_statistic', 0.0):.3f} (p={v.get('ks_pvalue', 1.0):.3f})"
            )
        if atl.get("available"):
            lines.append(
                f"  - ATL: concepts Δ={atl.get('concept_count_delta', 0):+d}  jaccard={atl.get('jaccard', 0.0):.3f}"
            )

    return "\n".join(lines) + "\n"


def validate_spec(spec_path: Path | str) -> _IterationSpec:
    """Public spec-validation entry point for ``--dry-run`` CLI flag."""
    return _parse_spec(Path(spec_path))


__all__ = [
    "ArmResult",
    "RoyIterationResult",
    "run_roy_iteration",
    "validate_spec",
]
