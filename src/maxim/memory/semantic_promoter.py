"""SemanticPromoter — promotes patterns from NAc/StatisticianAgent to ATL.

Multi-source promotion pipeline: NAc reward signals + StatisticianAgent
confirmed patterns + IPS randomness gating determine what becomes
semantic knowledge in the ATL.

Brain mapping: Repeated experiences (hippocampus) and confirmed causal
patterns (NAc) are gradually abstracted into semantic concepts (ATL).
The IPS randomness gate ensures only genuinely non-random patterns are
promoted, filtering out coincidental correlations.

This module defines:
- PromotionCandidate — source-agnostic pattern worth promoting
- PromotionSource — protocol for any system that produces candidates
- SemanticPromoter — orchestrates the full promotion pipeline
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@dataclass
class PromotionCandidate:
    """A pattern worth promoting to semantic memory.

    Source-agnostic — any causal system can produce these.
    """

    pattern_name: str               # Human-readable: "grasp → success"
    category: str                   # "causal_pattern", "operational_pattern", etc.
    confidence: float               # Source system's confidence in this pattern
    source_memory_ids: list[str]    # Episodic memory IDs where this was observed
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class PromotionSource(Protocol):
    """Any system that can produce promotion candidates.

    NAc implements this. StatisticianAgent implements this.
    Future causal systems implement this too.
    """

    def get_promotion_candidates(
        self,
        min_confidence: float = 0.6,
        min_observations: int = 3,
    ) -> list[PromotionCandidate]: ...


@dataclass
class PromotionConfig:
    """Configuration for the SemanticPromoter."""

    min_confidence: float = 0.6
    min_observations: int = 3
    randomness_gate_enabled: bool = True
    randomness_threshold: float = 0.4
    randomness_min_samples: int = 8


class SemanticPromoter:
    """Orchestrates promotion of patterns to ATL semantic memory.

    Collects candidates from all PromotionSource instances, applies
    the IPS randomness quality gate, then promotes qualifying patterns
    as ATL concepts with cross-layer edges linking back to source records.
    """

    def __init__(
        self,
        hippocampus: Any,
        atl: Any,
        sources: list[PromotionSource],
        cross_layer: Any,
        ips: Any | None = None,
        config: PromotionConfig | None = None,
    ) -> None:
        self._hippocampus = hippocampus
        self._atl = atl
        self._sources = sources
        self._cross_layer = cross_layer
        self._ips = ips
        self._config = config or PromotionConfig()

    def scan_for_promotions(self) -> list[str]:
        """Scan all sources, promote qualifying patterns to ATL.

        IPS randomness gate filters noise before expensive promotion.

        Returns:
            List of ATL concept IDs that were created or reinforced.
        """
        promoted_ids: list[str] = []

        for source in self._sources:
            try:
                candidates = source.get_promotion_candidates(
                    self._config.min_confidence,
                    self._config.min_observations,
                )
            except Exception as e:
                logger.warning("PromotionSource failed: %s", e)
                continue

            for candidate in candidates:
                # IPS randomness quality gate
                if self._ips and self._config.randomness_gate_enabled:
                    if not self._passes_randomness_gate(candidate):
                        continue

                concept_id = self._promote_candidate(candidate)
                if concept_id:
                    promoted_ids.append(concept_id)

        return promoted_ids

    def _passes_randomness_gate(self, candidate: PromotionCandidate) -> bool:
        """IPS fast-check: is this candidate's pattern non-random?

        For StatisticianAgent candidates: trust the already-computed confidence
        (StatAgent already ran IPS → AG escalation, no need to re-assess).

        For NAc candidates: assess the observation history if enough samples.
        """
        # StatisticianAgent candidates already passed IPS assessment
        if candidate.category == "operational_pattern":
            return candidate.confidence >= self._config.randomness_threshold

        # NAc candidates — run IPS assessment on observation timing
        obs_count = candidate.metadata.get("observation_count", 0)
        if obs_count < self._config.randomness_min_samples:
            return True  # Too few observations to assess — allow through

        # If NAc candidate has memory IDs, assess randomness of timing
        memory_ids = candidate.source_memory_ids
        if len(memory_ids) >= self._config.randomness_min_samples:
            timestamps: list[float] = []
            for mid in memory_ids:
                mem = self._hippocampus.get(mid)
                if mem:
                    timestamps.append(mem.timestamp)

            if len(timestamps) >= self._config.randomness_min_samples:
                timestamps.sort()
                intervals = [
                    timestamps[i + 1] - timestamps[i]
                    for i in range(len(timestamps) - 1)
                ]
                result = self._ips.assess_randomness(intervals)
                # Random timing suggests coincidence, not a real pattern
                if result.is_random and result.pattern_confidence < self._config.randomness_threshold:
                    return False

        return True  # Default: allow through

    def _promote_candidate(self, candidate: PromotionCandidate) -> str | None:
        """Promote a single candidate to an ATL concept.

        Returns concept_id if promoted, None otherwise.
        """
        from maxim.memory.cross_layer import CrossLayerEdgeType
        from maxim.memory.semantic_types import ConceptProvenance

        try:
            # Determine provenance
            if candidate.category == "operational_pattern":
                provenance = ConceptProvenance.AGENT_INFERENCE
            else:
                provenance = ConceptProvenance.EPISODIC_CONSOLIDATION

            # Find or create concept
            source_id = candidate.source_memory_ids[0] if candidate.source_memory_ids else None
            concept_id, was_created = self._atl.find_or_create(
                name=candidate.pattern_name,
                category=candidate.category,
                definition=self._build_definition(candidate),
                provenance=provenance,
                source_episode_id=source_id,
            )

            # If found existing, reinforce with additional episodes
            if not was_created and candidate.source_memory_ids:
                concept = self._atl.get(concept_id)
                if concept and hasattr(concept, "reinforce"):
                    for mid in candidate.source_memory_ids[1:]:
                        concept.reinforce(mid)

            # Create cross-layer edges
            self._create_cross_layer_edges(concept_id, candidate)

            logger.debug(
                "Promoted %s → ATL concept %s (new=%s)",
                candidate.pattern_name,
                concept_id,
                was_created,
            )
            return concept_id

        except Exception as e:
            logger.warning("Failed to promote %s: %s", candidate.pattern_name, e)
            return None

    def _build_definition(self, candidate: PromotionCandidate) -> str:
        """Build a definition string for the ATL concept."""
        parts = [f"Pattern: {candidate.pattern_name}"]

        if candidate.category == "causal_pattern":
            event_sig = candidate.metadata.get("event_signature", "")
            outcome_sig = candidate.metadata.get("outcome_signature", "")
            if event_sig and outcome_sig:
                parts.append(f"Causal: {event_sig} leads to {outcome_sig}")
            obs = candidate.metadata.get("observation_count", 0)
            if obs:
                parts.append(f"Observed {obs} times")

        elif candidate.category == "operational_pattern":
            pattern_type = candidate.metadata.get("pattern_type", "")
            r_squared = candidate.metadata.get("r_squared")
            if pattern_type:
                parts.append(f"Type: {pattern_type}")
            if r_squared is not None:
                parts.append(f"R²={r_squared:.2f}")

        return ". ".join(parts)

    def _create_cross_layer_edges(
        self,
        concept_id: str,
        candidate: PromotionCandidate,
    ) -> None:
        """Create cross-layer edges linking ATL concept to source records."""
        from maxim.memory.cross_layer import CrossLayerEdgeType

        # Link to source episodes (NAc candidates)
        for mid in candidate.source_memory_ids:
            self._cross_layer.add_edge(
                source_layer="atl",
                source_id=concept_id,
                target_layer="hippocampus",
                target_id=mid,
                edge_type=CrossLayerEdgeType.DERIVED_FROM,
                weight=candidate.confidence,
            )

        # Link to AG PATTERN record (StatisticianAgent candidates)
        ag_memory_id = candidate.metadata.get("ag_memory_id")
        if ag_memory_id:
            self._cross_layer.add_edge(
                source_layer="angular_gyrus",
                source_id=ag_memory_id,
                target_layer="atl",
                target_id=concept_id,
                edge_type=CrossLayerEdgeType.STATISTICALLY_CONFIRMS,
                weight=candidate.confidence,
            )

        # Temporal correlation edge
        phase_coherence = candidate.metadata.get("phase_coherence")
        if phase_coherence is not None and phase_coherence > 0.7:
            self._cross_layer.add_edge(
                source_layer="atl",
                source_id=concept_id,
                target_layer="atl",
                target_id=concept_id,
                edge_type=CrossLayerEdgeType.TEMPORALLY_CORRELATES,
                weight=phase_coherence,
                metadata={"phase_coherence": phase_coherence},
            )


__all__ = [
    "PromotionCandidate",
    "PromotionConfig",
    "PromotionSource",
    "SemanticPromoter",
]
