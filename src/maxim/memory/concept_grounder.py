"""ConceptGrounder — grounds ATL concept relationships with AG numerical analysis.

Piggybacks on concept recall: when episodes linked to a concept are loaded,
extracts numerical fields and runs IPS/AG analysis. Results become QUANTIFIES
edges and modulate existing relationship confidence via Jaccard co-occurrence.

Brain mapping: Angular gyrus provides numerical/spatial cognition that grounds
the ATL's semantic representations in quantitative evidence. IPS handles fast
approximate assessment; AG handles precise analysis when needed.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from maxim.memory.semantic_types import Concept
from maxim.memory.types import CompressedMemory, EpisodicMemory

if TYPE_CHECKING:
    from maxim.math.angular_gyrus import AngularGyrus
    from maxim.math.ips import IPS
    from maxim.memory.atl import ATL
    from maxim.memory.cross_layer import CrossLayerGraph

logger = logging.getLogger(__name__)


class ConceptGrounder:
    """Grounds ATL concept relationships with AG numerical analysis.

    Piggybacks on concept recall: when episodes linked to a concept are
    loaded, extracts numerical fields and runs IPS/AG analysis. Results
    become QUANTIFIES edges and modulate existing relationship confidence.

    Brain mapping: Angular gyrus provides numerical/spatial cognition
    that grounds the ATL's semantic representations in quantitative
    evidence. IPS handles fast approximate assessment; AG handles
    precise analysis when needed.
    """

    # Jaccard weight scaling factor. Maps Jaccard similarity [0, 1] to
    # edge weight [0, 1]. 2.0 means Jaccard >= 0.5 saturates at weight 1.0.
    JACCARD_WEIGHT_SCALE: float = 2.0

    def __init__(
        self,
        atl: ATL,
        angular_gyrus: AngularGyrus,
        ips: IPS,
        cross_layer: CrossLayerGraph,
        cache_ttl: float = 300.0,
        jaccard_weight_scale: float | None = None,
    ) -> None:
        self._atl = atl
        self._ag = angular_gyrus
        self._ips = ips
        self._cross_layer = cross_layer
        self._cache_ttl = cache_ttl
        if jaccard_weight_scale is not None:
            self.JACCARD_WEIGHT_SCALE = jaccard_weight_scale
        # concept_id -> (timestamp, stats_dict)
        self._stats_cache: dict[str, tuple[float, dict]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ground_concept(
        self,
        concept: Concept,
        episodes: list[EpisodicMemory | CompressedMemory],
    ) -> dict[str, Any]:
        """Compute numerical properties for a concept from its linked episodes.

        Called during concept recall when episodes are already loaded.
        Returns stats dict for inclusion in LLM context.

        Uses IPS (fast path) for basic stats. Escalates to AG (slow path)
        for regression/trend analysis when enough data points exist.
        """
        if not episodes:
            return {}

        # Check cache
        cached = self._stats_cache.get(concept.id)
        if cached:
            cache_time, cache_stats = cached
            if time.time() - cache_time < self._cache_ttl:
                if concept.ref_count("hippocampus") <= cache_stats.get("_ref_count", 0):
                    return cache_stats

        # Extract numerical fields from episodes
        numerics = self._extract_numerics(episodes)
        if not numerics:
            return {}

        stats = self._compute_stats(concept, numerics)

        # Cache results
        stats["_ref_count"] = concept.ref_count("hippocampus")
        self._stats_cache[concept.id] = (time.time(), stats)

        # Update relationship confidence based on numerical evidence
        self._modulate_relationships(concept)

        # Store AG MathMemory with QUANTIFIES edge if significant
        self._store_quantifications(concept, stats)

        return stats

    # ------------------------------------------------------------------
    # Numeric extraction
    # ------------------------------------------------------------------

    def _extract_numerics(
        self, episodes: list[EpisodicMemory | CompressedMemory]
    ) -> dict[str, list[float]]:
        """Extract numerical fields from loaded episodes.

        Returns field_name -> list of values across episodes.
        Only includes fields present in at least 2 episodes.

        Handles CompressedMemory gracefully: compressed records have
        novelty, salience, and success_rate but lack timing, confidence,
        and fear_level.
        """
        collectors: dict[str, list[float]] = defaultdict(list)

        for ep in episodes:
            if isinstance(ep, CompressedMemory):
                if ep.salience is not None:
                    collectors["salience"].append(ep.salience)
                if ep.novelty is not None:
                    collectors["novelty"].append(ep.novelty)
                if ep.success is not None:
                    collectors["success_rate"].append(1.0 if ep.success else 0.0)
                continue

            # Action timing
            exec_ms = ep.action.execution_time_ms
            if exec_ms and exec_ms > 0:
                collectors["execution_time_ms"].append(exec_ms)

            # Perception scores
            if ep.perception.salience is not None:
                collectors["salience"].append(ep.perception.salience)
            if ep.perception.novelty is not None:
                collectors["novelty"].append(ep.perception.novelty)

            # Decision confidence
            if ep.decision.confidence is not None:
                collectors["decision_confidence"].append(ep.decision.confidence)

            # Context
            if ep.context.fear_level is not None and ep.context.fear_level > 0:
                collectors["fear_level"].append(ep.context.fear_level)

            # Outcome
            if ep.outcome.success is not None:
                collectors["success_rate"].append(1.0 if ep.outcome.success else 0.0)

        # Filter to fields with enough data points
        return {k: v for k, v in collectors.items() if len(v) >= 2}

    # ------------------------------------------------------------------
    # Statistics computation
    # ------------------------------------------------------------------

    def _compute_stats(
        self, concept: Concept, numerics: dict[str, list[float]]
    ) -> dict[str, Any]:
        """Compute statistics using IPS (fast) with AG escalation (precise).

        IPS handles: mean (ApproximateResult), trend (TrendResult)
        AG escalation: regression analysis when 8+ data points and IPS uncertain
        """
        stats: dict[str, Any] = {}

        for field_name, values in numerics.items():
            # IPS fast path: basic stats
            approx = self._ips.estimate_mean(values)
            trend = self._ips.detect_trend(values) if len(values) >= 3 else None

            field_stats: dict[str, Any] = {
                "mean": approx.value,
                "n": len(values),
                "min": min(values),
                "max": max(values),
            }
            if trend and trend.direction.name != "STABLE":
                field_stats["trend"] = trend.direction.name.lower()

            # AG escalation: precise analysis when IPS trend is uncertain
            if (
                len(values) >= 8
                and trend
                and 0.3 < trend.confidence < 0.65
            ):
                try:
                    analysis = self._ag.analyze(values, method="linear")
                    if analysis and analysis.confidence > 0.5:
                        field_stats["r_squared"] = analysis.confidence
                        slope = analysis.parameters.get("slope")
                        if slope is not None:
                            field_stats["slope"] = slope
                        field_stats["ag_analysis"] = True
                except Exception as e:
                    logger.debug("AG escalation failed for %s: %s", field_name, e)

            stats[field_name] = field_stats

        return stats

    # ------------------------------------------------------------------
    # Relationship modulation
    # ------------------------------------------------------------------

    def _modulate_relationships(self, concept: Concept) -> None:
        """Strengthen or weaken concept relationships based on Jaccard
        co-occurrence similarity.

        Uses Jaccard index (|A ∩ B| / |A ∪ B|) — symmetric and handles
        size imbalance naturally.
        """
        relationships = self._atl.find_by_relationship(
            concept.id, direction="both", limit=50
        )
        if not relationships:
            return

        concept_refs = set(concept.memory_refs.get("hippocampus", {}))

        for other_id, rel in relationships:
            other = self._atl.get(other_id)
            if not isinstance(other, Concept):
                continue

            other_refs = set(other.memory_refs.get("hippocampus", {}))
            shared = len(concept_refs & other_refs)
            union = len(concept_refs | other_refs)

            if union < 3:
                continue

            jaccard = shared / union

            # Strengthen: high co-occurrence with enough shared evidence
            if jaccard > 0.3 and shared >= 3:
                self._atl.semantics.update_edge(
                    concept.id, other_id, rel.relationship_type,
                    weight=min(1.0, jaccard * self.JACCARD_WEIGHT_SCALE),
                    confidence_delta=0.05,
                )
            # Weaken: low co-occurrence despite many total observations
            elif jaccard < 0.05 and union >= 10:
                self._atl.semantics.update_edge(
                    concept.id, other_id, rel.relationship_type,
                    confidence_delta=-0.1,
                )

    # ------------------------------------------------------------------
    # AG quantification storage
    # ------------------------------------------------------------------

    def _store_quantifications(
        self, concept: Concept, stats: dict[str, Any]
    ) -> None:
        """Store significant numerical properties as AG MathMemory records
        linked to the concept via QUANTIFIES edges.

        Only creates/updates records for fields with enough data (n >= 5).
        """
        from maxim.math.math_types import MathCategory, MathMemory
        from maxim.memory.cross_layer import CrossLayerEdgeType

        for field_name, field_stats in stats.items():
            if field_name.startswith("_"):
                continue
            n = field_stats.get("n", 0)
            if n < 5:
                continue

            existing_name = f"{concept.name}:{field_name}"
            existing = self._ag.recall(limit=1, name=existing_name)

            if existing:
                # Update existing record in-place (it's in-memory)
                record = existing[0]
                if isinstance(record, MathMemory):
                    record.verbal = (
                        f"{concept.name} {field_name}: "
                        f"mean={field_stats['mean']:.2f} (n={n})"
                    )
                    record.observation_count = n
                    record.confidence = min(0.9, 0.3 + 0.05 * n)
                    record.touch()
            else:
                # Create new MathMemory
                record = MathMemory(
                    id=str(uuid4()),
                    timestamp=time.time(),
                    name=existing_name,
                    category=MathCategory.PATTERN,
                    domain="concept_property",
                    verbal=(
                        f"{concept.name} {field_name}: "
                        f"mean={field_stats['mean']:.2f} (n={n})"
                    ),
                    code="",
                    inputs=[concept.name],
                    outputs=[field_name],
                    source="derived",
                    confidence=min(0.9, 0.3 + 0.05 * n),
                    observation_count=n,
                )
                record_id = self._ag.store(record)

                # QUANTIFIES edge: AG record -> ATL concept
                self._cross_layer.add_edge(
                    source_layer="angular_gyrus",
                    source_id=record_id,
                    target_layer="atl",
                    target_id=concept.id,
                    edge_type=CrossLayerEdgeType.QUANTIFIES,
                    weight=1.0,
                    metadata={"field": field_name},
                )

                # Track ref in concept
                concept.add_ref("angular_gyrus", record_id)


__all__ = ["ConceptGrounder"]