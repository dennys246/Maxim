"""Function-to-tier routing with fallback chains (Lane Tier Architecture Phase 1).

Maps named LLM functions (agent_inference, concept_extraction, etc.) to
capability-tier lanes (large, medium, small) with configurable fallback
chains and hard restrictions.

The tier system replaces the old function-based lanes (infer/review/record)
with size-based tiers. Functions declare what capability level they need;
the router handles placement, fallback, and restrictions.

Mesh-ready: ``available_tiers`` accepts either a static ``set[str]`` or a
``Callable[[], set[str]]``. A future mesh layer can pass a callable that
queries local tiers + peer-advertised tiers. When mesh is off, a plain set
is wrapped in a lambda — zero overhead. See
``docs/plans/deferred/llm_path_multi_peer_dispatch.md`` for the planned
peer-advertised-tier integration if/when it revives.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Callable

logger = logging.getLogger(__name__)


# ─── Exceptions ──────────────────────────────────────────────────────────────


class TierRoutingError(Exception):
    """Base for tier routing failures."""


class TierRestrictionError(TierRoutingError):
    """Function is restricted to a tier that isn't available (no fallback)."""


class TierUnavailableError(TierRoutingError):
    """No tier in the fallback chain is available."""


# ─── FunctionSpec ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class FunctionSpec:
    """How a named function maps to the tier system.

    Attributes:
        name: Function identifier (e.g. ``"agent_inference"``).
        tier: Primary tier this function targets.
        fallback: Ordered fallback tiers. Empty tuple = hard-restricted to
            primary tier (never degrade).
        priority_boost: Added to job priority for restricted functions so they
            don't starve behind degradable ones in the same tier queue.
        description: Human-readable purpose (for doctor output / logging).
    """

    name: str
    tier: str
    fallback: tuple[str, ...] = ()
    priority_boost: int = 0
    description: str = ""


# ─── Default function table ─────────────────────────────────────────────────

# Configurable via llm.json "functions" key or MAXIM_FUNCTION_*_TIER env vars.
# Keep this table minimal (~13 entries). Custom functions go in config.

DEFAULT_FUNCTIONS: dict[str, FunctionSpec] = {
    # ── Core agent operations — need strong reasoning ──
    "agent_inference": FunctionSpec(
        name="agent_inference",
        tier="large",
        fallback=(),
        priority_boost=2,
        description="Main agent loop LLM calls",
    ),
    "fear_review": FunctionSpec(
        name="fear_review",
        tier="large",
        fallback=("medium",),
        description="FearAgent safety review",
    ),
    # ── Analysis and writing — need decent reasoning ──
    "paper_writing": FunctionSpec(
        name="paper_writing",
        tier="large",
        fallback=("medium",),
        description="Research protocol Writer agent",
    ),
    "sim_orchestrator": FunctionSpec(
        name="sim_orchestrator",
        tier="large",
        fallback=("medium",),
        description="Simulation orchestrator LLM (drives sim agent loop)",
    ),
    "campaign_summary": FunctionSpec(
        name="campaign_summary",
        tier="medium",
        fallback=("small",),
        description="Post-campaign analysis summaries",
    ),
    "sim_report_analysis": FunctionSpec(
        name="sim_report_analysis",
        tier="medium",
        fallback=("small",),
        description="SimulationReport LLM analysis",
    ),
    # ── Structured extraction — small model sufficient ──
    "concept_extraction": FunctionSpec(
        name="concept_extraction",
        tier="small",
        fallback=("medium",),
        description="ATL concept extraction from memories",
    ),
    "concept_grounding": FunctionSpec(
        name="concept_grounding",
        tier="small",
        fallback=("medium",),
        description="ATL concept grounding verification",
    ),
    "relationship_formation": FunctionSpec(
        name="relationship_formation",
        tier="small",
        fallback=("medium",),
        description="Concept extractor inline relationship formation",
    ),
    # ── Utility tasks — lightweight, structured output ──
    "narrative_transcription": FunctionSpec(
        name="narrative_transcription",
        tier="small",
        fallback=(),
        description="Parse narrative text into structured detections",
    ),
    "entity_naming": FunctionSpec(
        name="entity_naming",
        tier="small",
        fallback=(),
        description="Generate character/entity display names",
    ),
    "choice_classification": FunctionSpec(
        name="choice_classification",
        tier="small",
        fallback=("medium",),
        description="DM choice classifier (map AUT response to encounter choices)",
    ),
    "json_repair": FunctionSpec(
        name="json_repair",
        tier="small",
        fallback=(),
        description="Fix malformed JSON from other LLM calls",
    ),
}


# ─── FunctionRouter ──────────────────────────────────────────────────────────

# Default tier ordering — largest capability first. Configurable via
# tier_order param for deployments with custom tiers.
DEFAULT_TIER_ORDER: list[str] = ["large", "medium", "small"]


class FunctionRouter:
    """Routes named functions to tier-based lanes with fallback.

    Phase 1: static availability (tier exists in deployment).
    Phase 2 (mesh): dynamic availability via health_check + callable tiers.

    Mesh-ready: ``available_tiers`` accepts either a ``set[str]`` or a
    ``Callable[[], set[str]]``. A future mesh layer can pass a callable that
    queries local tiers + peer-advertised tiers. When mesh is off, a plain
    set is wrapped in a lambda — zero overhead.
    """

    def __init__(
        self,
        functions: dict[str, FunctionSpec] | None = None,
        available_tiers: set[str] | Callable[[], set[str]] = frozenset(),
        *,
        tier_order: list[str] | None = None,
        health_check: Callable[[str], bool] | None = None,
    ) -> None:
        self._functions = functions if functions is not None else dict(DEFAULT_FUNCTIONS)
        # Callable or static set — normalize to callable for uniform access
        self._get_available: Callable[[], set[str]] = (
            available_tiers if callable(available_tiers) else lambda _tiers=available_tiers: _tiers  # type: ignore[misc]
        )
        self._tier_order = tier_order or DEFAULT_TIER_ORDER
        self._health_check = health_check
        self._fallback_lock = threading.Lock()
        self._fallback_counts: dict[str, int] = {}

    # ── Public API ───────────────────────────────────────────────────────

    def resolve(self, function: str) -> tuple[str, int]:
        """Return ``(tier_name, priority_boost)`` for a function.

        Tries primary tier, then fallbacks in order. Raises
        :class:`TierRestrictionError` if the function is restricted and its
        tier is unavailable, or :class:`TierUnavailableError` if all
        fallback tiers are also unavailable.
        """
        spec = self._functions.get(function)
        if spec is None:
            logger.debug("Unknown function %r — routing to largest available tier", function)
            return self._largest_available(), 0

        if self._tier_available(spec.tier):
            return spec.tier, spec.priority_boost

        for fallback_tier in spec.fallback:
            if self._tier_available(fallback_tier):
                with self._fallback_lock:
                    self._fallback_counts[function] = self._fallback_counts.get(function, 0) + 1
                logger.info(
                    "Function %s: primary tier %s unavailable, falling back to %s",
                    function,
                    spec.tier,
                    fallback_tier,
                )
                return fallback_tier, spec.priority_boost

        try:
            available = self._get_available()
        except Exception:
            available = set()  # Best-effort for error message
        if not spec.fallback:
            raise TierRestrictionError(
                f"Function {function!r} requires tier {spec.tier!r} "
                f"(restricted, no fallback). Available tiers: {available}"
            )
        raise TierUnavailableError(
            f"No tier available for function {function!r}. "
            f"Tried: {spec.tier} → {spec.fallback}. Available tiers: {available}"
        )

    def get_function(self, name: str) -> FunctionSpec | None:
        """Look up a function spec by name."""
        return self._functions.get(name)

    def list_functions(self) -> dict[str, FunctionSpec]:
        """Return a copy of the function table."""
        return dict(self._functions)

    def available_tiers(self) -> set[str]:
        """Current set of available tiers (snapshot)."""
        return set(self._get_available())

    def fallback_stats(self) -> dict[str, int]:
        """Fallback frequency per function — feeds LaneMetrics / diagnostics."""
        with self._fallback_lock:
            return dict(self._fallback_counts)

    # ── Private ──────────────────────────────────────────────────────────

    def _tier_available(self, tier: str) -> bool:
        """Check if a tier is available (static + optional dynamic health).

        Exceptions from the callable ``available_tiers`` or ``health_check``
        are caught and treated as "tier unavailable" — this prevents mesh
        failures from crashing the routing path.
        """
        try:
            if tier not in self._get_available():
                return False
        except Exception:
            logger.warning("available_tiers callable failed — treating tier %r as unavailable", tier)
            return False
        if self._health_check is not None:
            try:
                return self._health_check(tier)
            except Exception:
                logger.warning("health_check(%r) failed — treating as unavailable", tier)
                return False
        return True

    def _largest_available(self) -> str:
        """Return the largest available tier per ``tier_order``."""
        available = self._get_available()
        for tier in self._tier_order:
            if tier in available:
                return tier
        raise TierUnavailableError(f"No tiers available. Checked order: {self._tier_order}")


__all__ = [
    "DEFAULT_FUNCTIONS",
    "DEFAULT_TIER_ORDER",
    "FunctionRouter",
    "FunctionSpec",
    "TierRestrictionError",
    "TierRoutingError",
    "TierUnavailableError",
]
