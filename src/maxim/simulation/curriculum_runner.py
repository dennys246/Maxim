"""Roy curriculum runner — chain substrate-only priming sessions.

R1 of the Roy long-horizon harness build (see
``docs/plans/grounded_language_acquisition.md`` "Roy long-horizon harness"
and ``docs/plans/deferred/persona_convergence_crucible.md`` "Substrate-only
priming").

Reads a curriculum YAML — a sequence of fixture / arc / goal stages —
and runs each stage as one ``start_simulation_mode`` call, threading
``resume_session`` from the previous stage so the substrate
(hippocampus + NAc + ATL + EC) chains across the whole curriculum.
That makes a Roy iteration a single operator command instead of a
hand-rolled shell loop over a hundred sims.

Curriculum YAML schema (deliberately tiny)::

    name: roy-0-smoke
    embodiment: bodies/infant_humanoid     # default for all stages
    aut_mode: substrate-primary            # default; "llm-primary" also OK
    persona: neutral                       # default
    stages:
      - name: act1_warmup
        fixture: scenarios/cradle/warmup.yaml   # OR
        # arc: cradle_prelinguistic              # OR
        # goal: "test memory recall"             # plain goal string
        turns: 50
      - name: act2_main
        arc: cradle_prelinguistic
        turns: 200

Top-level ``embodiment`` / ``aut_mode`` / ``persona`` apply to every
stage unless the stage overrides them. Each stage picks exactly one of
``fixture`` / ``arc`` / ``goal``; passing more than one is rejected.

Fixture paths are resolved relative to the curriculum YAML's directory
(unless absolute).

The runner returns a :class:`CurriculumResult` with per-stage session
ids, the final substrate path (the last successful stage's session
directory), total wall-clock duration, and ``aborted_at`` set to the
stage name where the chain failed (or ``None`` on full success).
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
class StageResult:
    """One stage's outcome — successful or failed.

    ``session_id`` / ``session_dir`` are populated when the underlying
    sim ran far enough to write a report; on early failure they may be
    empty. ``error`` is set iff the stage raised.
    """

    name: str
    source_kind: str  # "fixture" | "arc" | "goal"
    source_value: str  # the path / arc-name / goal text
    turns_requested: int
    session_id: str = ""
    session_dir: str = ""
    turns_run: int = 0
    finish_reason: str = ""
    duration_s: float = 0.0
    error: str | None = None


@dataclass
class CurriculumResult:
    """Result of a chained-stage curriculum run.

    ``final_session_id`` / ``final_session_dir`` point at the **last
    successful** stage — that's where the most-evolved substrate
    snapshots live. On full-curriculum success this is the last stage;
    on mid-curriculum failure it's the last stage that completed before
    ``aborted_at``.
    """

    name: str
    spec_path: str
    stages: list[StageResult] = field(default_factory=list)
    final_session_id: str = ""
    final_session_dir: str = ""
    total_duration_s: float = 0.0
    aborted_at: str | None = None  # name of the stage where the chain stopped


# ---------------------------------------------------------------------------
# Spec parsing
# ---------------------------------------------------------------------------


@dataclass
class _StageSpec:
    name: str
    source_kind: str  # "fixture" | "arc" | "goal"
    source_value: str
    turns: int
    embodiment: str | None
    aut_mode: str
    persona: str


@dataclass
class _CurriculumSpec:
    name: str
    spec_path: Path
    stages: list[_StageSpec]


_DEFAULT_AUT_MODE = "llm-primary"
_DEFAULT_PERSONA = "neutral"
_VALID_AUT_MODES = ("llm-primary", "substrate-primary")


def _parse_spec(spec_path: Path) -> _CurriculumSpec:
    """Load and validate a curriculum YAML.

    Raises ``ValueError`` with a precise message on any schema problem
    so the operator gets fail-fast feedback instead of a silent half-run.
    """
    import yaml

    spec_path = Path(spec_path).resolve()
    if not spec_path.exists():
        raise FileNotFoundError(f"Curriculum spec not found: {spec_path}")

    with open(spec_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Curriculum spec must be a YAML mapping, got {type(raw).__name__}")

    name = raw.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Curriculum spec missing required string field 'name'")

    stages_raw = raw.get("stages")
    if not isinstance(stages_raw, list) or not stages_raw:
        raise ValueError("Curriculum spec 'stages' must be a non-empty list")

    top_embodiment = raw.get("embodiment")
    if top_embodiment is not None and not isinstance(top_embodiment, str):
        raise ValueError("Top-level 'embodiment' must be a string if set")

    top_aut_mode = raw.get("aut_mode", _DEFAULT_AUT_MODE)
    if top_aut_mode not in _VALID_AUT_MODES:
        raise ValueError(f"Top-level 'aut_mode' must be one of {_VALID_AUT_MODES!r}, got {top_aut_mode!r}")

    top_persona = raw.get("persona", _DEFAULT_PERSONA)
    if not isinstance(top_persona, str) or not top_persona.strip():
        raise ValueError("Top-level 'persona' must be a non-empty string")

    spec_dir = spec_path.parent
    seen_names: set[str] = set()
    stages: list[_StageSpec] = []
    for i, stage_raw in enumerate(stages_raw):
        if not isinstance(stage_raw, dict):
            raise ValueError(f"Stage #{i} must be a mapping, got {type(stage_raw).__name__}")

        sname = stage_raw.get("name")
        if not isinstance(sname, str) or not sname.strip():
            raise ValueError(f"Stage #{i} missing required string field 'name'")
        if sname in seen_names:
            raise ValueError(f"Duplicate stage name: {sname!r}")
        seen_names.add(sname)

        sources = [k for k in ("fixture", "arc", "goal") if k in stage_raw]
        if len(sources) != 1:
            raise ValueError(
                f"Stage {sname!r} must specify exactly one of 'fixture', 'arc', 'goal' (got: {sources or 'none'})"
            )
        source_kind = sources[0]
        source_value_raw = stage_raw[source_kind]
        if not isinstance(source_value_raw, str) or not source_value_raw.strip():
            raise ValueError(f"Stage {sname!r} field {source_kind!r} must be a non-empty string")

        if source_kind == "fixture":
            fixture_path = Path(source_value_raw)
            if not fixture_path.is_absolute():
                fixture_path = (spec_dir / fixture_path).resolve()
            if not fixture_path.exists():
                raise ValueError(f"Stage {sname!r} fixture not found: {fixture_path} (resolved relative to {spec_dir})")
            source_value = str(fixture_path)
        else:
            source_value = source_value_raw.strip()

        turns_raw = stage_raw.get("turns")
        if not isinstance(turns_raw, int) or turns_raw <= 0:
            raise ValueError(f"Stage {sname!r} 'turns' must be a positive int, got {turns_raw!r}")

        stage_embodiment = stage_raw.get("embodiment", top_embodiment)
        if stage_embodiment is not None and not isinstance(stage_embodiment, str):
            raise ValueError(f"Stage {sname!r} 'embodiment' must be a string if set")

        stage_aut_mode = stage_raw.get("aut_mode", top_aut_mode)
        if stage_aut_mode not in _VALID_AUT_MODES:
            raise ValueError(f"Stage {sname!r} 'aut_mode' must be one of {_VALID_AUT_MODES!r}, got {stage_aut_mode!r}")

        stage_persona = stage_raw.get("persona", top_persona)
        if not isinstance(stage_persona, str) or not stage_persona.strip():
            raise ValueError(f"Stage {sname!r} 'persona' must be a non-empty string")

        stages.append(
            _StageSpec(
                name=sname,
                source_kind=source_kind,
                source_value=source_value,
                turns=turns_raw,
                embodiment=stage_embodiment,
                aut_mode=stage_aut_mode,
                persona=stage_persona,
            )
        )

    return _CurriculumSpec(name=name.strip(), spec_path=spec_path, stages=stages)


# ---------------------------------------------------------------------------
# Stage execution
# ---------------------------------------------------------------------------


def _build_stage_kwargs(stage: _StageSpec, prev_session_id: str | None) -> dict[str, Any]:
    """Translate one stage spec into kwargs for ``start_simulation_mode``.

    ``fixture`` stages route through ``fixture_path`` (no narrator LLM).
    ``arc`` stages set ``goal=arc_name`` so the orchestrator's existing
    ``select_arc_for_goal`` picks them up and runs the generative
    campaign runner. ``goal`` stages pass the goal verbatim.
    """
    kwargs: dict[str, Any] = {
        "persona": stage.persona,
        "max_turns": stage.turns,
        "aut_mode": stage.aut_mode,
        "entity_ref": stage.embodiment,
    }
    if prev_session_id:
        kwargs["resume_session"] = prev_session_id

    if stage.source_kind == "fixture":
        # fixture-driven sims supply their own percepts; the goal string
        # is just a label that ends up in the report.
        kwargs["goal"] = f"curriculum:{stage.name}"
        kwargs["fixture_path"] = stage.source_value
    elif stage.source_kind == "arc":
        # generative=True so the campaign runner picks up arc_yaml /
        # builtin arc lookup. select_arc_for_goal in orchestrator.py
        # already maps a builtin arc name to the right arc.
        kwargs["goal"] = stage.source_value
        kwargs["generative"] = True
    else:  # goal
        kwargs["goal"] = stage.source_value
        # Mirror cli.py: auto-route through the generative campaign
        # runner if the goal matches a builtin arc.
        try:
            from maxim.simulation.arcs import select_arc_for_goal

            if select_arc_for_goal(stage.source_value) is not None:
                kwargs["generative"] = True
        except Exception as e:
            logger.debug("select_arc_for_goal lookup failed for %r: %s", stage.source_value, e)

    return kwargs


def run_curriculum(
    spec_path: Path | str,
    *,
    sim_runner: Callable[..., Any] | None = None,
) -> CurriculumResult:
    """Run a chained-stage Roy curriculum.

    Parameters
    ----------
    spec_path:
        Path to the curriculum YAML.
    sim_runner:
        Injection seam for tests — defaults to
        :func:`maxim.simulation.orchestrator.start_simulation_mode`.
        Must accept the same kwargs and return an object with
        ``session_id`` / ``session_dir`` / ``turns`` / ``finish_reason``
        attributes (i.e. a :class:`SimulationResult`).
    """
    spec = _parse_spec(Path(spec_path))

    if sim_runner is None:
        from maxim.simulation.orchestrator import start_simulation_mode

        sim_runner = start_simulation_mode

    result = CurriculumResult(name=spec.name, spec_path=str(spec.spec_path))
    overall_start = time.time()
    prev_session_id: str | None = None

    for stage in spec.stages:
        stage_record = StageResult(
            name=stage.name,
            source_kind=stage.source_kind,
            source_value=stage.source_value,
            turns_requested=stage.turns,
        )
        stage_start = time.time()
        kwargs = _build_stage_kwargs(stage, prev_session_id)

        logger.info(
            "Curriculum %r stage %r: %s=%r turns=%d aut_mode=%s resume=%s",
            spec.name,
            stage.name,
            stage.source_kind,
            stage.source_value,
            stage.turns,
            stage.aut_mode,
            prev_session_id or "<fresh>",
        )

        try:
            sim_result = sim_runner(**kwargs)
        except Exception as e:
            stage_record.duration_s = round(time.time() - stage_start, 2)
            stage_record.error = f"{type(e).__name__}: {e}"
            stage_record.finish_reason = "error"
            result.stages.append(stage_record)
            result.aborted_at = stage.name
            logger.error(
                "Curriculum %r aborted at stage %r: %s",
                spec.name,
                stage.name,
                stage_record.error,
                exc_info=True,
            )
            break

        stage_record.session_id = getattr(sim_result, "session_id", "") or ""
        stage_record.session_dir = getattr(sim_result, "session_dir", "") or ""
        stage_record.turns_run = int(getattr(sim_result, "turns", 0) or 0)
        stage_record.finish_reason = getattr(sim_result, "finish_reason", "") or ""
        stage_record.duration_s = round(time.time() - stage_start, 2)
        result.stages.append(stage_record)

        if stage_record.finish_reason == "error":
            result.aborted_at = stage.name
            logger.error(
                "Curriculum %r aborted at stage %r: sim finished with finish_reason=error",
                spec.name,
                stage.name,
            )
            break

        # Only update the chain pointers when the stage produced a usable
        # session id; otherwise stage N+1 would resume from "" which the
        # orchestrator would just treat as a fresh start AND silently break
        # the chain. Better to fail loudly.
        if not stage_record.session_id:
            stage_record.error = "stage returned empty session_id; cannot chain to next stage"
            result.aborted_at = stage.name
            logger.error(
                "Curriculum %r aborted at stage %r: %s",
                spec.name,
                stage.name,
                stage_record.error,
            )
            break

        result.final_session_id = stage_record.session_id
        result.final_session_dir = stage_record.session_dir
        prev_session_id = stage_record.session_id

    result.total_duration_s = round(time.time() - overall_start, 2)

    logger.info(
        "Curriculum %r complete: %d/%d stages, final_session=%s, aborted_at=%s, duration=%.1fs",
        spec.name,
        sum(1 for s in result.stages if s.error is None and s.finish_reason != "error"),
        len(spec.stages),
        result.final_session_id or "<none>",
        result.aborted_at or "<none>",
        result.total_duration_s,
    )

    return result


__all__ = ["CurriculumResult", "StageResult", "run_curriculum"]
