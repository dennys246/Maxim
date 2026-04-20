"""Asset Foundry — autonomous SEM component generation, validation, and curation.

Pipeline: Generate → Validate → SEM Protocol Tests → Gauntlet → Score → Curate

Usage::

    runner = FoundryRunner(theme="cyberpunk weapons", genre="cyberpunk", category="weapons")
    result = runner.run(count=10)
    # result.promoted contains high-scoring components ready for human review

Each foundry run persists to ~/.maxim/foundry/{run_id}/ with candidates,
results, scores, and a human-readable report.

See docs/plans/deferred/asset_foundry_plan.md for the full design.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class CandidateSpec:
    """A generated SEM component candidate."""

    name: str
    spec: dict[str, Any]
    category: str
    genre: str
    source: str = "llm"  # "llm" or "fallback"


@dataclass
class ValidationResult:
    """Result of validating a candidate spec."""

    candidate_name: str
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class GauntletResult:
    """Result of running a candidate through the test gauntlet."""

    candidate_name: str
    encounters_completed: int = 0
    encounters_total: int = 3
    hippocampal_captures: int = 0
    nac_observations: int = 0
    pain_signals: int = 0
    affordances_used: set[str] = field(default_factory=set)
    status: str = "pending"  # pending, pass, partial, infra_error
    error: str | None = None


@dataclass
class CandidateScore:
    """Scored + curated candidate."""

    candidate_name: str
    total_score: float
    dimension_scores: dict[str, float] = field(default_factory=dict)
    bucket: str = "reject"  # promote, review, reject
    highlights: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)


@dataclass
class ScoringConfig:
    """Configuration for the scoring rubric."""

    dimensions: dict[str, float] = field(
        default_factory=lambda: {
            "hippocampal_engagement": 0.30,
            "causal_learning": 0.30,
            "pain_failure": 0.20,
            "affordance_usage": 0.20,
        }
    )
    promote_threshold: float = 0.7
    reject_threshold: float = 0.4


@dataclass
class FoundryResult:
    """Complete result of a foundry run."""

    run_id: str
    theme: str
    genre: str
    category: str | None
    generated: int = 0
    validated: int = 0
    tested: int = 0
    promoted: list[CandidateScore] = field(default_factory=list)
    review: list[CandidateScore] = field(default_factory=list)
    rejected: list[CandidateScore] = field(default_factory=list)
    generation_failures: int = 0
    infra_errors: int = 0
    output_dir: str = ""
    dedup_skipped: list[tuple[str, str, float]] = field(default_factory=list)
    """(candidate_name, existing_ref, cosine_score) for near-duplicates."""
    promoted_paths: list[str] = field(default_factory=list)
    """Absolute paths of files promoted to ~/.maxim/components/."""


@dataclass
class CurationReport:
    """Summary of an auto-curation run before a sim."""

    genre: str
    categories_checked: int = 0
    categories_below_threshold: int = 0
    foundry_results: list[FoundryResult] = field(default_factory=list)
    total_promoted: int = 0
    total_skipped_dedup: int = 0
    total_generated: int = 0


# ---------------------------------------------------------------------------
# F-0: Generation
# ---------------------------------------------------------------------------

# Sub-theme seeds per category to avoid repetitive generation
_CATEGORY_SUBTHEMES: dict[str, list[str]] = {
    "weapons": [
        "a close-range melee weapon with an energy mechanic",
        "a long-range weapon with overheating",
        "a stealth weapon with limited charges",
        "a heavy weapon with area-of-effect and recoil",
        "a hybrid weapon that switches between two modes",
        "a primitive weapon with high durability",
        "a fragile but devastating weapon",
        "a defensive weapon that can also counter-attack",
    ],
    "creatures": [
        "a predator with ambush tactics",
        "a defensive creature with thick armor",
        "a swarm creature that relies on numbers",
        "a flying creature with ranged attacks",
        "a burrowing creature with terrain control",
        "an intelligent creature that uses tools",
        "a venomous creature with delayed effects",
        "a pack animal with loyalty mechanics",
    ],
    "npcs": [
        "a merchant with dynamic pricing",
        "a guard with patrol behavior",
        "a healer with limited resources",
        "a scholar with knowledge to share",
        "a thief with stealth capabilities",
        "a leader who commands followers",
        "a craftsperson who can repair items",
        "a mysterious stranger with hidden motives",
    ],
    "items": [
        "a consumable with immediate effect",
        "a key item that unlocks progress",
        "a tool with multiple uses",
        "a container that stores resources",
        "a trap that can be deployed",
        "a healing item with side effects",
        "a light source with limited fuel",
        "a communication device",
    ],
    "environments": [
        "a hazardous area with environmental damage",
        "a dark area requiring light sources",
        "an underwater area with pressure mechanics",
        "a defensible position with chokepoints",
        "a resource-rich area with competition",
        "a shifting environment that changes over time",
        "an area with weather effects",
        "a vertical area requiring climbing",
    ],
    "vehicles": [
        "a fast vehicle with poor handling",
        "an armored vehicle with slow speed",
        "a flying vehicle with fuel management",
        "a water vehicle with weather sensitivity",
        "a mount with loyalty and fatigue",
        "a cargo vehicle with capacity limits",
    ],
    "bodies": [
        "an augmented limb with power management",
        "a sensory organ with calibration needs",
        "a defensive body part with regeneration",
        "a locomotion system with terrain adaptation",
    ],
}


def generate_candidates(
    theme: str,
    count: int,
    genre: str,
    category: str | None = None,
    llm_router: Any | None = None,
) -> list[CandidateSpec]:
    """Generate candidate SEM component specs from a theme prompt.

    Uses EntityDesigner for LLM-driven generation with fallback to
    template-based generation when no LLM is available.

    Args:
        theme: Theme description (e.g., "cyberpunk weapons").
        count: Number of candidates to generate.
        genre: Genre tag for the components.
        category: Specific category (weapons, creatures, etc.).
            If None, distributes across categories.
        llm_router: LLM router for generation. None = fallback mode.

    Returns:
        List of CandidateSpec instances.
    """
    from maxim.simulation.entity_designer import EntityDesigner

    designer = EntityDesigner(llm_router=llm_router)
    candidates: list[CandidateSpec] = []

    # Determine categories to distribute across
    if category:
        categories = [category] * count
    else:
        all_cats = list(_CATEGORY_SUBTHEMES.keys())
        categories = [all_cats[i % len(all_cats)] for i in range(count)]

    for i in range(count):
        cat = categories[i]
        subthemes = _CATEGORY_SUBTHEMES.get(cat, ["a unique entity"])
        subtheme = subthemes[i % len(subthemes)]

        description = f"{theme} — {subtheme}. Genre: {genre}."
        entity_type = _category_to_entity_type(cat)

        try:
            spec = designer.design(
                description=description,
                entity_type=entity_type,
                genre=genre,
            )

            # Ensure name uniqueness
            base_name = spec.get("name", f"{cat}_{i}")
            spec["name"] = _ensure_unique_name(base_name, [c.name for c in candidates])

            # Add genre tag to metadata
            spec.setdefault("metadata", {})
            spec["metadata"]["genre"] = genre
            spec["metadata"]["foundry_theme"] = theme

            source = "llm" if llm_router else "fallback"
            candidates.append(
                CandidateSpec(
                    name=spec["name"],
                    spec=spec,
                    category=cat,
                    genre=genre,
                    source=source,
                )
            )
            log.info("Generated candidate %d/%d: %s (%s)", i + 1, count, spec["name"], cat)

        except Exception as e:
            log.warning("Generation failed for candidate %d/%d: %s", i + 1, count, e)

    return candidates


def _category_to_entity_type(category: str) -> str:
    """Map category to SEM entity_type."""
    mapping = {
        "weapons": "weapon",
        "creatures": "creature",
        "npcs": "npc",
        "items": "item",
        "environments": "environment",
        "vehicles": "vehicle",
        "bodies": "body_part",
    }
    return mapping.get(category, "generic")


def _ensure_unique_name(name: str, existing: list[str]) -> str:
    """Ensure name is unique among existing names."""
    if name not in existing:
        return name
    for suffix in range(2, 100):
        candidate = f"{name}_{suffix}"
        if candidate not in existing:
            return candidate
    return f"{name}_{int(time.time())}"


# ---------------------------------------------------------------------------
# F-1: Validation
# ---------------------------------------------------------------------------


def validate_candidate(spec: dict[str, Any]) -> ValidationResult:
    """Validate a candidate spec against SEM schema rules.

    Runs checks in order of cost (cheapest first):
    1. Schema — required fields exist
    2. Sensor sanity — ranges ordered, initial within range
    3. Affordance sanity — at least one affordance per modulator
    4. Failure mode sanity — triggers reference valid sensors
    5. Semantic sanity — ranges reasonable, failure modes triggerable

    Returns:
        ValidationResult with errors (hard reject) and warnings (test anyway).
    """
    from maxim.simulation.entity_designer import validate_entity_spec

    name = spec.get("name", "<unnamed>")
    errors: list[str] = []
    warnings: list[str] = []

    # 1. Schema validation (reuse existing)
    schema_errors = validate_entity_spec(spec)
    errors.extend(schema_errors)

    # 2. Additional sensor sanity
    sensors = spec.get("sensors", {})
    if not sensors:
        errors.append("No sensors defined")

    for sname, sspec in sensors.items():
        if not isinstance(sspec, dict):
            continue
        rng = sspec.get("range")
        if isinstance(rng, (list, tuple)) and len(rng) == 2:
            span = abs(rng[1] - rng[0])
            if span > 10_000:
                warnings.append(f"Sensor '{sname}': very large range span ({span})")
            if span == 0:
                errors.append(f"Sensor '{sname}': zero-width range")

    # 3. Affordance sanity
    modulators = spec.get("modulators", {})
    if not modulators:
        errors.append("No modulators defined")

    total_affordances = 0
    for mname, mspec in modulators.items():
        if not isinstance(mspec, dict):
            continue
        affordances = mspec.get("affordances", {})
        if not affordances:
            warnings.append(f"Modulator '{mname}': no affordances defined")
        total_affordances += len(affordances)

    if total_affordances == 0 and not errors:
        errors.append("No affordances defined in any modulator")

    # 4. Failure mode sanity
    failure_modes = spec.get("failure_modes", [])
    sensor_names = set(sensors.keys())
    for fm in failure_modes:
        if not isinstance(fm, dict):
            errors.append(f"Failure mode must be a dict, got {type(fm).__name__}")
            continue
        trigger = fm.get("trigger", {})
        if isinstance(trigger, dict):
            field = trigger.get("field")
            if field and field not in sensor_names:
                errors.append(
                    f"Failure mode '{fm.get('name', '?')}': trigger field "
                    f"'{field}' not in sensors ({sorted(sensor_names)})"
                )
            pain = trigger.get("pain", 0)
            if not (0.0 <= pain <= 1.0):
                errors.append(f"Failure mode '{fm.get('name', '?')}': pain {pain} not in [0, 1]")

    # 5. Semantic sanity — check failure modes are triggerable
    if not failure_modes:
        warnings.append("No failure modes defined — entity has no risk/learning potential")

    for fm in failure_modes:
        if not isinstance(fm, dict):
            continue
        trigger = fm.get("trigger", {})
        if isinstance(trigger, dict):
            field = trigger.get("field")
            op = trigger.get("op")
            value = trigger.get("value")
            if field in sensors and isinstance(sensors[field], dict):
                rng = sensors[field].get("range")
                initial = sensors[field].get("initial")
                if rng and initial is not None and value is not None:
                    # Check if failure is triggerable from initial state
                    if op in ("<", "<=") and initial <= value:
                        pass  # Already below threshold at start — always triggered
                    elif op in (">", ">=") and initial >= value:
                        pass  # Already above threshold at start
                    # If the trigger threshold is outside the sensor range, warn
                    if isinstance(rng, (list, tuple)) and len(rng) == 2:
                        if value < rng[0] or value > rng[1]:
                            warnings.append(
                                f"Failure mode '{fm.get('name', '?')}': trigger "
                                f"value {value} outside sensor range {rng}"
                            )

    return ValidationResult(
        candidate_name=name,
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# F-2: SEM Protocol Tests
# ---------------------------------------------------------------------------


def run_sem_protocol_tests(spec: dict[str, Any]) -> ValidationResult:
    """Run 8 structural SEM protocol tests on a candidate spec.

    These are fast (no LLM) and verify the spec can be instantiated
    and exercised through the full SEM pipeline.

    Tests:
    1. Instantiation — _parse_entity() succeeds
    2. Sensor R/W — sensors initialize within range
    3. Affordance enumeration — all affordances have descriptions
    4. Tool generation — tool_bridge generates tools without collision
    5. Failure mode triggers — triggers reference valid sensors
    6. Composition — entity can walk() its tree
    7. Vital metrics — vital_metrics populated from sensor ranges
    8. Embodiment wrapping — Embodiment() accepts the entity
    """
    name = spec.get("name", "<unnamed>")
    errors: list[str] = []
    warnings: list[str] = []

    # Test 1: Instantiation
    try:
        from maxim.embodiment.spec import _parse_entity

        entity = _parse_entity(spec)
    except Exception as e:
        errors.append(f"T1 Instantiation failed: {e}")
        return ValidationResult(candidate_name=name, valid=False, errors=errors)

    # Test 2: Sensor R/W
    try:
        readings = entity.read_all_sensors()
        for sname, reading in readings.items():
            sensor_spec = spec.get("sensors", {}).get(sname, {})
            rng = sensor_spec.get("range")
            if rng and isinstance(reading.value, (int, float)):
                if not (rng[0] <= reading.value <= rng[1]):
                    errors.append(f"T2 Sensor '{sname}': initial {reading.value} outside range {rng}")
    except Exception as e:
        errors.append(f"T2 Sensor R/W failed: {e}")

    # Test 3: Affordance enumeration
    try:
        for ent in entity.walk():
            for mod in ent.modulators.values():
                for aff_name, aff_schema in mod.affordances.items():
                    if not aff_schema.description:
                        warnings.append(f"T3 Affordance '{aff_name}': no description")
    except Exception as e:
        errors.append(f"T3 Affordance enumeration failed: {e}")

    # Test 4: Tool generation
    try:
        from maxim.embodiment.tool_bridge import generate_tools_for_entity
        from maxim.tools.registry import ToolRegistry

        test_registry = ToolRegistry()
        tools = generate_tools_for_entity(entity, test_registry)
        if not tools:
            warnings.append("T4 No tools generated from entity")
    except Exception as e:
        errors.append(f"T4 Tool generation failed: {e}")

    # Test 5: Failure mode triggers (already checked in validate_candidate,
    # but here we verify they work on the live entity)
    try:
        for fm in entity.failure_modes:
            if not fm.triggers:
                warnings.append(f"T5 Failure mode '{fm.name}': no triggers")
    except Exception as e:
        errors.append(f"T5 Failure mode check failed: {e}")

    # Test 6: Composition — entity tree walkable
    try:
        all_ents = list(entity.walk())
        if not all_ents:
            errors.append("T6 Entity tree walk returned empty")
    except Exception as e:
        errors.append(f"T6 Composition walk failed: {e}")

    # Test 7: Vital metrics
    try:
        if not entity.vital_metrics:
            warnings.append("T7 No vital_metrics populated")
    except Exception as e:
        errors.append(f"T7 Vital metrics check failed: {e}")

    # Test 8: Embodiment wrapping
    try:
        from maxim.embodiment.body import Embodiment

        emb = Embodiment(entity)
        if emb.root.name != entity.name:
            errors.append("T8 Embodiment root name mismatch")
    except Exception as e:
        errors.append(f"T8 Embodiment wrapping failed: {e}")

    return ValidationResult(
        candidate_name=name,
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# F-2: Gauntlet
# ---------------------------------------------------------------------------


def run_gauntlet(
    candidate: CandidateSpec,
    *,
    model: str | None = None,
) -> GauntletResult:
    """Run a candidate through the 3-encounter test gauntlet.

    Uses build_executor with entity_ref-like construction to exercise
    the full pain cascade pipeline. No LLM needed — the gauntlet
    directly invokes affordance tools and checks bio-system responses.

    Args:
        candidate: The candidate spec to test.
        model: LLM model for the gauntlet (unused in structural mode).

    Returns:
        GauntletResult with bio-system metrics.
    """
    result = GauntletResult(candidate_name=candidate.name, encounters_total=3)

    try:
        from maxim.decisions.nac import NAc, NACConfig
        from maxim.embodiment.body import Embodiment
        from maxim.embodiment.spec import _parse_entity
        from maxim.embodiment.tool_bridge import generate_tools_for_entity
        from maxim.memory.hippocampus import Hippocampus
        from maxim.proprioception.pain_bus import build_pain_bus
        from maxim.runtime.bootstrap import build_executor
        from maxim.tools.registry import ToolRegistry

        # Fresh bio-stack per candidate — uses build_pain_bus to auto-subscribe
        # hippocampus + NAc learners (per CLAUDE.md invariant).
        nac = NAc(NACConfig(temporal_window_seconds=60.0))
        hippocampus = Hippocampus()
        pain_bus = build_pain_bus(hippocampus=hippocampus, nac=nac)

        # Parse entity, create embodiment, generate tools, then build executor.
        # Tools must be in the registry BEFORE build_executor so the bridge
        # can track them. Embodiment is attached to the executor after
        # construction (declared field, not a stash).
        entity = _parse_entity(candidate.spec)
        embodiment = Embodiment(entity, pain_bus=pain_bus)
        registry = ToolRegistry()
        tools = generate_tools_for_entity(entity, registry, embodiment=embodiment)

        executor = build_executor(
            registry,
            pain_bus=pain_bus,
            nac=nac,
            hippocampus=hippocampus,
        )
        executor.embodiment = embodiment

        # Collect affordance tool names for testing
        affordance_tools = [
            t.name
            for t in tools
            if hasattr(t, "_affordance_name")  # ModulatorAffordanceTool
        ]

        if not affordance_tools:
            result.status = "partial"
            result.error = "No affordance tools generated"
            return result

        # --- Encounter 1: Discovery (observe sensors) ---
        try:
            sense_tools = [t.name for t in tools if "sense" in t.name.lower()]
            if sense_tools:
                executor.execute({"tool_name": sense_tools[0], "params": {}})
                result.encounters_completed += 1
        except Exception as e:
            log.debug("Encounter 1 failed for %s: %s", candidate.name, e)

        # --- Encounter 2: Interaction (use affordances) ---
        try:
            for tool_name in affordance_tools[:3]:
                try:
                    executor.execute({"tool_name": tool_name, "params": {"target": "test_dummy", "force": 0.5}})
                    result.affordances_used.add(tool_name)
                except Exception:
                    # Some affordances may need different params — try minimal
                    try:
                        executor.execute({"tool_name": tool_name, "params": {}})
                        result.affordances_used.add(tool_name)
                    except Exception:
                        pass
            result.encounters_completed += 1
        except Exception as e:
            log.debug("Encounter 2 failed for %s: %s", candidate.name, e)

        # --- Encounter 3: Stress (push toward failure modes) ---
        try:
            # Set vital metrics to near-failure thresholds
            for fm in entity.failure_modes:
                for trigger in fm.triggers:
                    if trigger.field in entity.vital_metrics:
                        if trigger.op in ("<", "<="):
                            entity.vital_metrics[trigger.field] = max(0.0, trigger.value - 0.01)
                        elif trigger.op in (">", ">="):
                            entity.vital_metrics[trigger.field] = trigger.value + 0.01

            # Evaluate failures at stressed state
            failures = embodiment.evaluate_failures()
            result.pain_signals = len(failures)

            # Try using affordances at stressed state
            if affordance_tools:
                try:
                    executor.execute(
                        {"tool_name": affordance_tools[0], "params": {"target": "stress_test", "force": 0.9}}
                    )
                except Exception:
                    try:
                        executor.execute({"tool_name": affordance_tools[0], "params": {}})
                    except Exception:
                        pass

            result.encounters_completed += 1
        except Exception as e:
            log.debug("Encounter 3 failed for %s: %s", candidate.name, e)

        # Measure real bio-system state (not manual counters)
        result.nac_observations = len(nac)  # public __len__
        result.hippocampal_captures = len(hippocampus)

        result.status = "pass" if result.encounters_completed == 3 else "partial"

    except Exception as e:
        result.status = "infra_error"
        result.error = str(e)
        log.error("Gauntlet infra error for %s: %s", candidate.name, e)

    return result


# ---------------------------------------------------------------------------
# F-3: Scoring + Curation
# ---------------------------------------------------------------------------


def score_result(
    gauntlet: GauntletResult,
    candidate: CandidateSpec,
    config: ScoringConfig | None = None,
) -> CandidateScore:
    """Score a gauntlet result using the 4-dimension rubric.

    Dimensions:
    - hippocampal_engagement: captures / encounters
    - causal_learning: NAc observations normalized
    - pain_failure: pain signals / failure modes defined
    - affordance_usage: distinct affordances used / total available
    """
    config = config or ScoringConfig()
    scores: dict[str, float] = {}
    highlights: list[str] = []
    issues: list[str] = []

    total_affordances = sum(
        len(m.get("affordances", {})) for m in candidate.spec.get("modulators", {}).values() if isinstance(m, dict)
    )
    total_failure_modes = len(candidate.spec.get("failure_modes", []))

    # Hippocampal engagement (0-1)
    if gauntlet.encounters_total > 0:
        scores["hippocampal_engagement"] = min(1.0, gauntlet.hippocampal_captures / gauntlet.encounters_total)
    else:
        scores["hippocampal_engagement"] = 0.0

    # Causal learning (0-1)
    # Normalize: 3+ observations = full score
    scores["causal_learning"] = min(1.0, gauntlet.nac_observations / 3.0)

    # Pain/failure activation (0-1)
    if total_failure_modes > 0:
        scores["pain_failure"] = min(1.0, gauntlet.pain_signals / total_failure_modes)
    else:
        scores["pain_failure"] = 0.0
        issues.append("No failure modes defined")

    # Affordance usage (0-1)
    if total_affordances > 0:
        scores["affordance_usage"] = len(gauntlet.affordances_used) / total_affordances
    else:
        scores["affordance_usage"] = 0.0
        issues.append("No affordances to use")

    # Compute total
    total = 0.0
    for dim_name, weight in config.dimensions.items():
        dim_score = scores.get(dim_name, 0.0)
        total += dim_score * weight

    # Generate highlights
    if scores.get("causal_learning", 0) >= 0.8:
        highlights.append(f"Strong causal learning ({gauntlet.nac_observations} NAc links)")
    if scores.get("pain_failure", 0) >= 0.8:
        highlights.append(f"Pain cascade activated ({gauntlet.pain_signals} signals)")
    if len(gauntlet.affordances_used) >= 3:
        highlights.append(f"{len(gauntlet.affordances_used)} affordances exercised")

    if gauntlet.status == "infra_error":
        issues.append(f"Infrastructure error: {gauntlet.error}")
        total = 0.0

    # Bucket
    if total >= config.promote_threshold:
        bucket = "promote"
    elif total >= config.reject_threshold:
        bucket = "review"
    else:
        bucket = "reject"

    return CandidateScore(
        candidate_name=candidate.name,
        total_score=round(total, 3),
        dimension_scores={k: round(v, 3) for k, v in scores.items()},
        bucket=bucket,
        highlights=highlights,
        issues=issues,
    )


def generate_report(result: FoundryResult) -> str:
    """Generate a human-readable markdown report."""
    lines = [
        f"# Foundry Report — {result.theme}",
        "",
        f"**Run ID:** {result.run_id}",
        f"**Genre:** {result.genre} | **Category:** {result.category or 'mixed'}",
        f"**Generated:** {result.generated} | **Validated:** {result.validated} | **Tested:** {result.tested}",
        f"**Promoted:** {len(result.promoted)} | **Review:** {len(result.review)} "
        f"| **Rejected:** {len(result.rejected)}",
        "",
    ]

    if result.promoted:
        lines.append("## Promoted")
        lines.append("")
        lines.append("| Component | Score | Highlights |")
        lines.append("|-----------|-------|-----------|")
        for s in result.promoted:
            hl = "; ".join(s.highlights) if s.highlights else "—"
            lines.append(f"| {s.candidate_name} | {s.total_score:.2f} | {hl} |")
        lines.append("")

    if result.review:
        lines.append("## Flagged for Review")
        lines.append("")
        lines.append("| Component | Score | Issue |")
        lines.append("|-----------|-------|-------|")
        for s in result.review:
            iss = "; ".join(s.issues) if s.issues else "—"
            lines.append(f"| {s.candidate_name} | {s.total_score:.2f} | {iss} |")
        lines.append("")

    if result.rejected:
        lines.append("## Rejected")
        lines.append("")
        lines.append("| Component | Score | Reason |")
        lines.append("|-----------|-------|--------|")
        for s in result.rejected:
            reason = "; ".join(s.issues) if s.issues else "Low engagement"
            lines.append(f"| {s.candidate_name} | {s.total_score:.2f} | {reason} |")
        lines.append("")

    if result.dedup_skipped:
        lines.append("## Skipped (Near-Duplicates)")
        lines.append("")
        lines.append("| Candidate | Similar To | Cosine |")
        lines.append("|-----------|-----------|--------|")
        for name, existing, score in result.dedup_skipped:
            lines.append(f"| {name} | {existing} | {score:.2f} |")
        lines.append("")

    if result.generation_failures > 0:
        lines.append(f"**Generation failures:** {result.generation_failures}")
    if result.infra_errors > 0:
        lines.append(f"**Infrastructure errors:** {result.infra_errors}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def _get_foundry_dir(run_id: str) -> Path:
    """Get the foundry output directory."""
    from maxim.utils.paths import data_home

    return data_home() / "foundry" / run_id


def _save_candidate_yaml(candidate: CandidateSpec, output_dir: Path) -> Path:
    """Save a candidate spec as YAML."""
    from maxim.utils.atomic_io import atomic_write_text

    candidates_dir = output_dir / "candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)

    path = candidates_dir / f"{candidate.name}.yaml"
    content = {
        "component": {
            "name": candidate.name,
            "tags": [candidate.genre, candidate.category],
            "category": candidate.category,
        },
        "entity": candidate.spec,
    }
    atomic_write_text(str(path), yaml.dump(content, default_flow_style=False, sort_keys=False))
    return path


def _save_result_json(score: CandidateScore, gauntlet: GauntletResult, output_dir: Path) -> Path:
    """Save gauntlet result + score as JSON."""
    from maxim.utils.atomic_io import atomic_write_json

    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    path = results_dir / f"{score.candidate_name}.json"
    data = {
        "candidate": score.candidate_name,
        "score": score.total_score,
        "bucket": score.bucket,
        "dimensions": score.dimension_scores,
        "highlights": score.highlights,
        "issues": score.issues,
        "gauntlet": {
            "encounters_completed": gauntlet.encounters_completed,
            "encounters_total": gauntlet.encounters_total,
            "hippocampal_captures": gauntlet.hippocampal_captures,
            "nac_observations": gauntlet.nac_observations,
            "pain_signals": gauntlet.pain_signals,
            "affordances_used": sorted(gauntlet.affordances_used),
            "status": gauntlet.status,
            "error": gauntlet.error,
        },
    }
    atomic_write_json(str(path), data)
    return path


def _promote_candidate(candidate: CandidateSpec, output_dir: Path) -> Path:
    """Copy a promoted candidate to the promoted/ directory."""
    from maxim.utils.atomic_io import atomic_write_text

    promoted_dir = output_dir / "promoted"
    promoted_dir.mkdir(parents=True, exist_ok=True)

    path = promoted_dir / f"{candidate.name}.yaml"
    content = _candidate_to_component_dict(candidate)
    atomic_write_text(str(path), yaml.dump(content, default_flow_style=False, sort_keys=False))
    return path


def _candidate_to_component_dict(candidate: CandidateSpec) -> dict:
    """Build the standard component YAML dict from a CandidateSpec."""
    synonyms = candidate.spec.get("metadata", {}).get("synonyms", [])
    component_header: dict[str, Any] = {
        "name": candidate.name,
        "tags": [candidate.genre, candidate.category],
        "category": candidate.category,
    }
    if synonyms:
        component_header["synonyms"] = synonyms
    return {
        "component": component_header,
        "entity": candidate.spec,
    }


def _promote_to_user_components(candidate: CandidateSpec) -> Path | None:
    """Promote a candidate to ~/.maxim/components/ for cross-session persistence.

    INVARIANT: Never writes to bundled _data/components/.
    Writes to ``data_home() / "components" / category / name.yaml``.
    """
    try:
        from maxim.utils.atomic_io import atomic_write_text
        from maxim.utils.paths import data_home

        user_dir = data_home() / "components" / candidate.category
        user_dir.mkdir(parents=True, exist_ok=True)

        path = user_dir / f"{candidate.name}.yaml"
        content = _candidate_to_component_dict(candidate)
        atomic_write_text(str(path), yaml.dump(content, default_flow_style=False, sort_keys=False))
        log.info("Promoted to user components: %s", path)
        return path
    except Exception as e:
        log.warning("Failed to promote %s to user components: %s", candidate.name, e)
        return None


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class FoundryRunner:
    """Orchestrates the full foundry pipeline.

    Usage::

        runner = FoundryRunner(
            theme="cyberpunk weapons",
            genre="cyberpunk",
            category="weapons",
        )
        result = runner.run(count=10)

    With ComponentIndex for dedup::

        from maxim.embodiment.component_index import ComponentIndex
        runner = FoundryRunner(..., component_index=index)
        # Candidates are checked against the index before promotion.
        # Near-duplicates (cosine >= 0.80) are skipped.
    """

    def __init__(
        self,
        theme: str,
        genre: str,
        category: str | None = None,
        llm_router: Any | None = None,
        scoring_config: ScoringConfig | None = None,
        dry_run: bool = False,
        component_index: Any | None = None,
        promote_to_user_dir: bool = False,
    ) -> None:
        self.theme = theme
        self.genre = genre
        self.category = category
        self._llm = llm_router
        self._scoring = scoring_config or ScoringConfig()
        self._dry_run = dry_run
        self._component_index = component_index
        self._promote_to_user_dir = promote_to_user_dir
        self.run_id = time.strftime("%Y%m%d_%H%M%S")

    def run(self, count: int = 10) -> FoundryResult:
        """Execute the full foundry pipeline."""
        output_dir = _get_foundry_dir(self.run_id)
        output_dir.mkdir(parents=True, exist_ok=True)

        result = FoundryResult(
            run_id=self.run_id,
            theme=self.theme,
            genre=self.genre,
            category=self.category,
            output_dir=str(output_dir),
        )

        # F-0: Generate candidates
        log.info("=== F-0: Generating %d candidates ===", count)
        candidates = generate_candidates(
            theme=self.theme,
            count=count,
            genre=self.genre,
            category=self.category,
            llm_router=self._llm,
        )
        result.generated = len(candidates)
        result.generation_failures = count - len(candidates)

        # Save raw candidates
        for c in candidates:
            _save_candidate_yaml(c, output_dir)

        # F-1: Validate
        log.info("=== F-1: Validating %d candidates ===", len(candidates))
        validated: list[tuple[CandidateSpec, ValidationResult]] = []
        for c in candidates:
            v = validate_candidate(c.spec)
            if v.valid:
                validated.append((c, v))
                log.info("  VALID: %s", c.name)
            else:
                log.info("  REJECT (schema): %s — %s", c.name, "; ".join(v.errors))
                # Save to rejected/
                rejected_dir = output_dir / "rejected"
                rejected_dir.mkdir(parents=True, exist_ok=True)
                from maxim.utils.atomic_io import atomic_write_text as _awt

                _awt(
                    str(rejected_dir / f"{c.name}.yaml"),
                    yaml.dump({"errors": v.errors, "spec": c.spec}, default_flow_style=False),
                )

        # SEM protocol tests
        sem_passed: list[tuple[CandidateSpec, ValidationResult]] = []
        for c, v in validated:
            sem = run_sem_protocol_tests(c.spec)
            if sem.valid:
                sem_passed.append((c, v))
                log.info("  SEM PASS: %s", c.name)
            else:
                log.info("  REJECT (SEM): %s — %s", c.name, "; ".join(sem.errors))

        result.validated = len(sem_passed)

        if self._dry_run:
            log.info("=== Dry run — skipping gauntlet + scoring ===")
            from maxim.utils.atomic_io import atomic_write_text

            atomic_write_text(
                str(output_dir / "config.yaml"),
                yaml.dump(
                    {
                        "theme": self.theme,
                        "genre": self.genre,
                        "category": self.category,
                        "count": count,
                        "dry_run": True,
                    },
                    default_flow_style=False,
                ),
            )
            return result

        # F-2: Gauntlet
        log.info("=== F-2: Running gauntlet on %d candidates ===", len(sem_passed))
        gauntlet_results: list[tuple[CandidateSpec, GauntletResult]] = []
        for c, _v in sem_passed:
            log.info("  Gauntlet: %s", c.name)
            gr = run_gauntlet(c)
            gauntlet_results.append((c, gr))
            result.tested += 1
            if gr.status == "infra_error":
                result.infra_errors += 1
                log.warning("  INFRA ERROR: %s — %s", c.name, gr.error)
            else:
                log.info(
                    "  %s: enc=%d/%d aff=%d nac=%d pain=%d",
                    c.name,
                    gr.encounters_completed,
                    gr.encounters_total,
                    len(gr.affordances_used),
                    gr.nac_observations,
                    gr.pain_signals,
                )

        # F-3: Score + curate
        log.info("=== F-3: Scoring + curation ===")
        for c, gr in gauntlet_results:
            score = score_result(gr, c, self._scoring)
            _save_result_json(score, gr, output_dir)

            if score.bucket == "promote":
                # Dedup check: skip near-duplicates of existing components
                if self._component_index is not None:
                    match = self._component_index.dedup_check(c.spec, threshold=0.80)
                    if match is not None:
                        result.dedup_skipped.append((c.name, match.ref, match.score))
                        log.info(
                            "  SKIP (dedup): %s — similar to existing %s (cosine %.2f)",
                            c.name,
                            match.ref,
                            match.score,
                        )
                        score.bucket = "review"
                        result.review.append(score)
                        continue

                result.promoted.append(score)
                _promote_candidate(c, output_dir)

                # Promote to ~/.maxim/components/ for persistence across sessions
                if self._promote_to_user_dir:
                    path = _promote_to_user_components(c)
                    if path is not None:
                        result.promoted_paths.append(str(path))

                # Add to index so future dedup checks see this entity
                if self._component_index is not None:
                    ref = f"{c.category}/{c.name}"
                    synonyms = c.spec.get("metadata", {}).get("synonyms", [])
                    self._component_index.add(ref, c.spec, synonyms=synonyms)

                log.info("  PROMOTE: %s (%.2f)", c.name, score.total_score)
            elif score.bucket == "review":
                result.review.append(score)
                log.info("  REVIEW: %s (%.2f)", c.name, score.total_score)
            else:
                result.rejected.append(score)
                log.info("  REJECT: %s (%.2f)", c.name, score.total_score)

        # Generate + save report
        from maxim.utils.atomic_io import atomic_write_json, atomic_write_text

        report = generate_report(result)
        atomic_write_text(str(output_dir / "report.md"), report)

        # Save scores summary
        scores_data = {
            "run_id": self.run_id,
            "theme": self.theme,
            "genre": self.genre,
            "promoted": [s.candidate_name for s in result.promoted],
            "review": [s.candidate_name for s in result.review],
            "rejected": [s.candidate_name for s in result.rejected],
        }
        atomic_write_json(str(output_dir / "scores.json"), scores_data)

        # Save config
        atomic_write_text(
            str(output_dir / "config.yaml"),
            yaml.dump(
                {
                    "theme": self.theme,
                    "genre": self.genre,
                    "category": self.category,
                    "count": count,
                    "run_id": self.run_id,
                },
                default_flow_style=False,
            ),
        )

        log.info("=== Foundry complete: %s ===", output_dir)
        log.info(
            "Promoted: %d | Review: %d | Rejected: %d | Dedup-skipped: %d",
            len(result.promoted),
            len(result.review),
            len(result.rejected),
            len(result.dedup_skipped),
        )

        return result


# ---------------------------------------------------------------------------
# Auto-Curation — pre-sim coverage gap analysis + foundry fill
# ---------------------------------------------------------------------------

# Categories that make sense for auto-curation gap filling.
_CURATION_CATEGORIES = ("weapons", "creatures", "npcs", "items", "environments")


def analyze_coverage(
    genre: str,
    threshold: int = 5,
    registry: Any | None = None,
) -> dict[str, int]:
    """Check per-category component coverage for a genre.

    Returns a dict of ``{category: count}`` for categories that are
    **below** *threshold*. Empty dict means full coverage.
    """
    if registry is None:
        from maxim.embodiment.component_registry import ComponentRegistry

        registry = ComponentRegistry()

    gaps: dict[str, int] = {}
    for cat in _CURATION_CATEGORIES:
        components = registry.query(genre=genre, category=cat)
        if len(components) < threshold:
            gaps[cat] = len(components)
    return gaps


def auto_curate(
    genre: str,
    *,
    threshold: int = 5,
    registry: Any | None = None,
    component_index: Any | None = None,
    llm_router: Any | None = None,
    dry_run: bool = False,
) -> CurationReport:
    """Run auto-curation: analyze coverage gaps and fill them via foundry.

    For each category below *threshold*, runs a FoundryRunner to generate
    enough components to fill the gap. Promoted candidates are written to
    ``~/.maxim/components/`` and registered in the ComponentIndex.

    Args:
        genre: Target genre (e.g., ``"fantasy"``, ``"cyberpunk"``).
        threshold: Minimum components per category (default: 5).
        registry: ComponentRegistry to check. Built if None.
        component_index: ComponentIndex for dedup. Skips dedup if None.
        llm_router: LLM router for generation. None = template fallback.
        dry_run: If True, generate + validate only (no gauntlet/promotion).

    Returns:
        CurationReport summarizing what was generated/promoted/skipped.
    """
    if registry is None:
        from maxim.embodiment.component_registry import ComponentRegistry

        registry = ComponentRegistry()

    report = CurationReport(genre=genre)
    gaps = analyze_coverage(genre, threshold=threshold, registry=registry)
    report.categories_checked = len(_CURATION_CATEGORIES)
    report.categories_below_threshold = len(gaps)

    if not gaps:
        log.info("Auto-curation: genre '%s' has full coverage (>= %d per category)", genre, threshold)
        return report

    log.info(
        "Auto-curation: genre '%s' has %d under-covered categor%s: %s",
        genre,
        len(gaps),
        "y" if len(gaps) == 1 else "ies",
        ", ".join(f"{cat}={count}" for cat, count in gaps.items()),
    )

    for cat, existing_count in gaps.items():
        needed = threshold - existing_count
        theme = f"{genre} {cat}"

        runner = FoundryRunner(
            theme=theme,
            genre=genre,
            category=cat,
            llm_router=llm_router,
            dry_run=dry_run,
            component_index=component_index,
            promote_to_user_dir=True,
        )
        result = runner.run(count=needed)

        report.foundry_results.append(result)
        report.total_promoted += len(result.promoted)
        report.total_skipped_dedup += len(result.dedup_skipped)
        report.total_generated += result.generated

        # Register promoted candidates in the registry so they're
        # available for the current session immediately.
        for promoted_path in result.promoted_paths:
            try:
                with open(promoted_path) as f:
                    spec = yaml.safe_load(f)
                if spec:
                    header = spec.get("component", {})
                    ref = f"{header.get('category', cat)}/{header.get('name', 'unknown')}"
                    registry.register(ref, spec, source_path=promoted_path)
                    log.info("Registered promoted component: %s", ref)
            except Exception as e:
                log.warning("Failed to register promoted component %s: %s", promoted_path, e)

    log.info(
        "Auto-curation complete: generated=%d promoted=%d dedup_skipped=%d",
        report.total_generated,
        report.total_promoted,
        report.total_skipped_dedup,
    )
    return report


def generate_curation_report_section(report: CurationReport) -> str:
    """Generate a markdown section for a curation report (appended to sim report)."""
    lines = [
        "",
        "## Auto-Curation Summary",
        "",
        f"**Genre:** {report.genre}",
        f"**Categories checked:** {report.categories_checked} "
        f"| **Below threshold:** {report.categories_below_threshold}",
        f"**Generated:** {report.total_generated} "
        f"| **Promoted:** {report.total_promoted} "
        f"| **Dedup-skipped:** {report.total_skipped_dedup}",
        "",
    ]

    for fr in report.foundry_results:
        lines.append(f"### {fr.theme}")
        lines.append(f"Generated: {fr.generated} | Validated: {fr.validated} | Tested: {fr.tested}")
        if fr.promoted:
            for s in fr.promoted:
                lines.append(f"  - PROMOTED: {s.candidate_name} ({s.total_score:.2f})")
        if fr.dedup_skipped:
            for name, existing, score in fr.dedup_skipped:
                lines.append(f"  - SKIPPED (dedup): {name} — similar to {existing} ({score:.2f})")
        lines.append("")

    return "\n".join(lines)
