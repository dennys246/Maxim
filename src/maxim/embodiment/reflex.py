"""Percept Reflex System — innate automatic body responses.

Reflexes are the body's fastest response layer: pattern-match keywords
in percept text → invoke a tool (damage_component, set_entity_sensor)
before the LLM deliberates.  They are INNATE (declared in the entity
spec / archetype YAML), not learned — the learned counterpart is
pre-emption via NAc predictions (Layer 2).

Architecture::

    Percept text → BioEnrichmentPipeline.enrich()
      ├── ... existing bio-system queries ...
      └── ReflexRegistry.evaluate(text, predictions)
            ├── keyword detection
            ├── cooldown check
            ├── habituation decay
            ├── sensitization boost
            ├── pre-emption suppression
            └── tool dispatch → pain → NAc learning

Key invariants:

- **Pain pipeline is the ONLY NAc learning path.**  Reflexes cause damage →
  pain → NAc.  No separate Reaction is emitted to NAc.  Prevents double-counting.

- **Pre-emption suppresses reflex intensity.**
  ``effective = base * habituation * sensitization * (1 - preemption)``.
  Learning to anticipate REDUCES net pain — correct biological gradient.

- **Responses are tool invocation specs, not enum dispatch.**
  YAML declares ``response: {tool: damage_component, params: {...}}``.
  Reuses existing tool infrastructure.

- **Clock injection for deterministic testing.**
  ``ReflexRegistry(clock=...)`` avoids ``time.monotonic()`` in tests.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from maxim.integration.bio_enrichment import CausalPrediction

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ReflexResponse:
    """Tool invocation spec for a reflex response."""

    tool: str  # e.g., "damage_component", "set_entity_sensor"
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ReflexSpec:
    """A single percept reflex defined in the entity spec.

    Detects keyword patterns in percept text and triggers an automatic
    body response.  Reflexes fire during bio-enrichment processing,
    BEFORE the LLM deliberates — the body responds before the mind
    decides.
    """

    name: str  # e.g., "attack_flinch"
    detect_keywords: tuple[str, ...]  # keyword patterns to detect
    response: ReflexResponse  # tool invocation spec
    base_intensity: float = 0.15  # default intensity
    intensity_scale: dict[str, float] = field(default_factory=dict)
    cooldown_s: float = 2.0  # min seconds between firings
    suppressible: bool = True  # pre-emption can reduce intensity


@dataclass(frozen=True, slots=True)
class ReflexFiring:
    """Record of a single reflex firing."""

    reflex_name: str
    tool: str
    params: dict[str, Any]
    effective_intensity: float
    raw_intensity: float
    habituation_factor: float
    sensitization_factor: float
    preemption_factor: float


# ---------------------------------------------------------------------------
# Registry + evaluation
# ---------------------------------------------------------------------------

# Habituation decay: intensity *= 1 / (1 + k * exposure_count)
_HABITUATION_K = 0.3

# Sensitization: intensity *= 1 + s * (1 - component_integrity)
_SENSITIZATION_S = 0.5


class ReflexRegistry:
    """Holds reflex specs for one agent and evaluates them against percept text.

    One registry per agent instance.  Tracks cooldowns, exposure counts
    (habituation), and provides deterministic clock injection for tests.

    Args:
        reflexes: Sequence of ReflexSpec to register.
        clock: Callable returning current time in seconds.  Defaults to
            ``time.monotonic``.  Inject a deterministic counter for tests.
        get_component_integrity: Callable that takes a component name and
            returns its current integrity (0–1).  Used for sensitization.
            Returns 1.0 (no sensitization) if not provided.
    """

    def __init__(
        self,
        reflexes: tuple[ReflexSpec, ...] = (),
        *,
        clock: Callable[[], float] = time.monotonic,
        get_component_integrity: Callable[[str], float] | None = None,
    ) -> None:
        self._reflexes = list(reflexes)
        self._clock = clock
        self._get_integrity = get_component_integrity or (lambda _name: 1.0)

        # Cooldown tracking: reflex_name → last fire timestamp
        self._last_fired: dict[str, float] = {}

        # Habituation: (reflex_name, context_hash) → exposure count
        self._exposure_counts: dict[tuple[str, str], int] = defaultdict(int)

    @property
    def reflexes(self) -> tuple[ReflexSpec, ...]:
        return tuple(self._reflexes)

    def add(self, spec: ReflexSpec) -> None:
        """Register a new reflex spec."""
        self._reflexes.append(spec)

    def reset_state(self) -> None:
        """Clear all mutable state (cooldowns, habituation).

        Useful for test isolation and session boundaries.
        """
        self._last_fired.clear()
        self._exposure_counts.clear()

    def evaluate(
        self,
        text: str,
        *,
        predictions: tuple[CausalPrediction, ...] = (),
        context_key: str = "",
        execute_tool: Callable[..., Any] | None = None,
    ) -> tuple[ReflexFiring, ...]:
        """Evaluate all reflexes against percept text.

        Args:
            text: Percept text to match against.
            predictions: NAc predictions already computed during enrichment.
                Used for pre-emption suppression — NOT re-queried.
            context_key: Context identifier for habituation tracking.
                Different contexts (different attacker, new environment)
                reset habituation.  Empty string = single global context.
            execute_tool: Callable to dispatch tool invocations.  Signature:
                ``execute_tool(tool_name, **params) -> Any``.
                If None, reflexes are evaluated but NOT executed (dry run).

        Returns:
            Tuple of ReflexFiring records for all reflexes that fired.
        """
        if not text:
            return ()

        lower = text.lower()
        now = self._clock()
        firings: list[ReflexFiring] = []

        for spec in self._reflexes:
            # 1. Keyword detection
            matched_keyword = self._match_keywords(lower, spec.detect_keywords)
            if matched_keyword is None:
                continue

            # 2. Cooldown check (sentinel -inf ensures first firing always passes)
            last = self._last_fired.get(spec.name, float("-inf"))
            if now - last < spec.cooldown_s:
                continue

            # 3. Compute raw intensity (base + keyword scale)
            raw_intensity = self._compute_raw_intensity(lower, spec)

            # 4. Habituation decay
            habit_key = (spec.name, context_key)
            exposure = self._exposure_counts[habit_key]
            habituation_factor = 1.0 / (1.0 + _HABITUATION_K * exposure)

            # 5. Sensitization boost (damaged parts feel more pain)
            component_name = spec.response.params.get("component", "")
            integrity = self._get_integrity(component_name) if component_name else 1.0
            sensitization_factor = 1.0 + _SENSITIZATION_S * (1.0 - integrity)

            # 6. Pre-emption suppression
            preemption_factor = 0.0
            if spec.suppressible and predictions:
                preemption_factor = self._compute_preemption(spec, predictions)

            # 7. Effective intensity
            effective = raw_intensity * habituation_factor * sensitization_factor * (1.0 - preemption_factor)
            effective = max(0.0, min(1.0, effective))

            if effective < 0.01:
                continue  # suppressed to nothing

            # 8. Build tool params with intensity
            params = dict(spec.response.params)
            if spec.response.tool == "damage_component":
                params["amount"] = round(effective, 3)
            elif spec.response.tool == "set_entity_sensor":
                # Scale the value adjustment by intensity ratio
                if "value" in params and spec.base_intensity > 0:
                    params["value"] = round(float(params["value"]) * effective / spec.base_intensity, 3)

            # 9. Execute tool (if dispatcher provided)
            dispatch_ok = True
            if execute_tool is not None:
                try:
                    execute_tool(spec.response.tool, **params)
                except Exception as e:
                    log.debug("Reflex %s tool dispatch failed: %s", spec.name, e)
                    dispatch_ok = False

            # 10. Update state — only on successful dispatch (or dry run).
            # Failed dispatch should not consume cooldown or increment
            # habituation, so the reflex can retry next tick.
            if dispatch_ok:
                self._last_fired[spec.name] = now
                self._exposure_counts[habit_key] += 1

            firings.append(
                ReflexFiring(
                    reflex_name=spec.name,
                    tool=spec.response.tool,
                    params=params,
                    effective_intensity=round(effective, 4),
                    raw_intensity=round(raw_intensity, 4),
                    habituation_factor=round(habituation_factor, 4),
                    sensitization_factor=round(sensitization_factor, 4),
                    preemption_factor=round(preemption_factor, 4),
                )
            )

        return tuple(firings)

    @staticmethod
    def _match_keywords(lower_text: str, keywords: tuple[str, ...]) -> str | None:
        """Return first matching keyword, or None."""
        for kw in keywords:
            if kw in lower_text:
                return kw
        return None

    @staticmethod
    def _compute_raw_intensity(lower_text: str, spec: ReflexSpec) -> float:
        """Compute raw intensity from base + intensity_scale keywords."""
        for keyword, scale in sorted(spec.intensity_scale.items(), key=lambda x: -x[1]):
            if keyword in lower_text:
                return scale
        return spec.base_intensity

    @staticmethod
    def _compute_preemption(spec: ReflexSpec, predictions: tuple[CausalPrediction, ...]) -> float:
        """Compute pre-emption suppression from NAc predictions.

        Checks if any prediction matches the reflex's keywords (the agent
        already anticipated this kind of stimulus).  Returns 0–1 intensity
        of pre-emption (how much to suppress the reflex).
        """
        max_preemption = 0.0
        for pred in predictions:
            if pred.valence != "negative":
                continue
            # Check if the prediction's event matches any of the reflex's keywords
            pred_lower = pred.event.lower()
            for kw in spec.detect_keywords:
                if kw in pred_lower or pred_lower in kw:
                    # Pre-emption intensity = prediction confidence
                    max_preemption = max(max_preemption, pred.confidence)
                    break
        return min(1.0, max_preemption)


# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------


def load_reflex_specs(path: Path) -> tuple[ReflexSpec, ...]:
    """Load reflex specs from a YAML file.

    Expected format::

        reflexes:
          attack_flinch:
            detect_keywords: [attack, strikes, hits, ...]
            response: {tool: damage_component, params: {component: torso}}
            base_intensity: 0.15
            intensity_scale:
              devastating: 0.30
              powerful: 0.20
            cooldown_s: 1.0
            suppressible: true

    Returns:
        Tuple of parsed ReflexSpec objects.

    Raises:
        FileNotFoundError: If the path doesn't exist.
        ValueError: If the YAML is malformed or missing required fields.
    """
    import yaml

    if not path.exists():
        raise FileNotFoundError(f"Reflex file not found: {path}")

    with open(path) as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict) or "reflexes" not in data:
        raise ValueError(f"Reflex file must contain a 'reflexes' key: {path}")

    specs: list[ReflexSpec] = []
    for name, defn in data["reflexes"].items():
        if not isinstance(defn, dict):
            raise ValueError(f"Reflex '{name}' must be a mapping, got {type(defn).__name__}")

        # Required fields
        keywords = defn.get("detect_keywords")
        if not keywords or not isinstance(keywords, list):
            raise ValueError(f"Reflex '{name}' requires 'detect_keywords' (list of strings)")

        response_raw = defn.get("response")
        if not response_raw or not isinstance(response_raw, dict):
            raise ValueError(f"Reflex '{name}' requires 'response' (mapping with 'tool' key)")

        tool = response_raw.get("tool")
        if not tool:
            raise ValueError(f"Reflex '{name}'.response requires 'tool' key")

        response = ReflexResponse(
            tool=str(tool),
            params=dict(response_raw.get("params", {})),
        )

        specs.append(
            ReflexSpec(
                name=str(name),
                detect_keywords=tuple(str(k).lower() for k in keywords),
                response=response,
                base_intensity=float(defn.get("base_intensity", 0.15)),
                intensity_scale={str(k).lower(): float(v) for k, v in defn.get("intensity_scale", {}).items()},
                cooldown_s=float(defn.get("cooldown_s", 2.0)),
                suppressible=bool(defn.get("suppressible", True)),
            )
        )

    return tuple(specs)


def load_archetype_reflexes(archetype: str) -> tuple[ReflexSpec, ...]:
    """Load reflex specs for an archetype from bundled data.

    Looks for ``_data/reflexes/{archetype}.yaml``.  Returns empty tuple
    if no reflex file exists for this archetype (not an error — some
    archetypes genuinely have no reflexes).
    """
    from maxim.utils.paths import bundled_data

    path = bundled_data() / "reflexes" / f"{archetype}.yaml"
    if not path.exists():
        return ()

    try:
        return load_reflex_specs(path)
    except Exception as e:
        log.warning("Failed to load reflexes for archetype '%s': %s", archetype, e)
        return ()


# ---------------------------------------------------------------------------
# Builder (canonical construction site)
# ---------------------------------------------------------------------------


def build_reflex_registry(
    *,
    entity_spec: Any,
    get_component_integrity: Callable[[str], float] | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> ReflexRegistry | None:
    """Canonical builder for ReflexRegistry.

    Loads archetype-level reflexes from bundled data.  Returns None if
    the entity has no archetype or no reflexes are found.

    Follows the ``build_pain_bus`` / ``build_executor`` pattern:
    required keyword-only args prevent silent wiring failures.

    Args:
        entity_spec: Parsed entity spec (needs ``archetype`` attribute).
        get_component_integrity: Returns component integrity by name.
        clock: Time source for cooldown tracking.

    Returns:
        ReflexRegistry with loaded reflexes, or None.
    """
    archetype = getattr(entity_spec, "archetype", None)
    if not archetype:
        # Check component-level archetype
        component = getattr(entity_spec, "component", None)
        if component is not None:
            archetype = getattr(component, "archetype", None)

    if not archetype:
        return None

    specs = load_archetype_reflexes(archetype)
    if not specs:
        return None

    return ReflexRegistry(
        specs,
        clock=clock,
        get_component_integrity=get_component_integrity,
    )
