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
import threading
import time
from collections import defaultdict
from functools import partial
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from maxim.memory.semantic_types import Concept
from maxim.memory.types import CompressedMemory, EpisodicMemory

if TYPE_CHECKING:
    from maxim.math.angular_gyrus import AngularGyrus
    from maxim.math.ips import IPS
    from maxim.memory.atl import ATL
    from maxim.memory.cross_layer import CrossLayerGraph
    from maxim.runtime.worker_pool import WorkerPool

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

    # Dedup cooldown: don't re-review a concept within this window
    REVIEW_COOLDOWN_S: float = 2.0

    def __init__(
        self,
        atl: ATL,
        angular_gyrus: AngularGyrus,
        ips: IPS,
        cross_layer: CrossLayerGraph,
        cache_ttl: float = 300.0,
        jaccard_weight_scale: float | None = None,
        worker_pool: WorkerPool | None = None,
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
        # WorkerPool for async relationship modulation + quantification
        self._pool = worker_pool
        # Dedup tracking: concept_id -> last enqueue monotonic time
        self._pending_reviews: dict[str, float] = {}
        self._pending_lock = threading.Lock()

        # Provenance collector (wired by MaximAgent.wire_provenance)
        self._collector: Any = None

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

        # Log grounding activity (P3e — Tier 2)
        if self._collector and self._collector.verbosity >= 1:
            from maxim.provenance.types import PipelineStage, ProvenanceRef

            prop_count = sum(1 for v in stats.values() if isinstance(v, dict))
            self._collector.log_activity(
                PipelineStage.ENRICHMENT,
                "concept_grounder",
                f"Grounded atl:{concept.id[:8]} ({concept.name}) with {prop_count} AG properties",
                sources=[ProvenanceRef("atl", concept.id, concept.name)],
            )

        # Cache results
        stats["_ref_count"] = concept.ref_count("hippocampus")
        self._stats_cache[concept.id] = (time.time(), stats)

        # Enqueue heavy work to background lanes if pool available
        if self._pool is not None:
            self._enqueue_background_work(concept.id, stats)
        else:
            # Fallback: sync (preserves existing behavior without pool)
            self._modulate_relationships(concept)
            self._store_quantifications(concept, stats)

        return stats

    # ------------------------------------------------------------------
    # Background async pipeline (review → record lanes)
    # ------------------------------------------------------------------

    def _enqueue_background_work(self, concept_id: str, stats: dict[str, Any]) -> None:
        """Submit relationship modulation + quantification to review lane.

        Includes dedup cooldown to avoid re-reviewing the same concept
        within REVIEW_COOLDOWN_S seconds.
        """
        with self._pending_lock:
            last = self._pending_reviews.get(concept_id, 0.0)
            if time.monotonic() - last < self.REVIEW_COOLDOWN_S:
                return  # Already enqueued recently
            self._pending_reviews[concept_id] = time.monotonic()

        try:
            self._pool.submit(
                lane="review",
                job_id=f"concept-review-{concept_id}-{time.monotonic_ns()}",
                fn=partial(self._review_concept, concept_id, dict(stats)),
                priority=5,
            )
        except Exception as e:
            logger.debug("Failed to enqueue concept review for %s: %s", concept_id, e)
            # Clear pending flag so next call can retry
            with self._pending_lock:
                self._pending_reviews.pop(concept_id, None)

    def _review_concept(self, concept_id: str, stats: dict[str, Any], prefetched: Any = None) -> None:
        """REVIEW LANE: Compute relationship changes + quantification proposals.

        Read-heavy, no writes. Hands off proposed changes to record lane.
        """
        # Clear pending flag after execution starts
        with self._pending_lock:
            self._pending_reviews.pop(concept_id, None)

        concept = self._atl.get(concept_id)
        if concept is None or not isinstance(concept, Concept):
            return  # Concept evicted between enqueue and execution

        # Compute proposed relationship edge updates (read-heavy)
        edge_updates = self._compute_relationship_updates(concept)

        # Compute quantification proposals (pure computation on stats)
        quant_proposals = self._compute_quantification_proposals(concept, stats)

        if not edge_updates and not quant_proposals:
            return

        # Hand off writes to record lane
        try:
            self._pool.submit(
                lane="record",
                job_id=f"concept-record-{concept_id}-{time.monotonic_ns()}",
                fn=partial(self._apply_updates, concept_id, edge_updates, quant_proposals),
                priority=7,
            )
        except Exception as e:
            logger.debug("Failed to enqueue concept record for %s: %s", concept_id, e)

    def _compute_relationship_updates(self, concept: Concept) -> list[tuple[str, str, str, float | None, float | None]]:
        """Compute proposed edge changes from Jaccard co-occurrence.

        Returns list of (source_id, target_id, rel_type, weight_or_None, confidence_delta_or_None).
        """
        relationships = self._atl.find_by_relationship(concept.id, direction="both", limit=50)
        if not relationships:
            return []

        concept_refs = set(concept.memory_refs.get("hippocampus", {}))
        updates: list[tuple[str, str, str, float | None, float | None]] = []

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

            if jaccard > 0.3 and shared >= 3:
                updates.append(
                    (
                        concept.id,
                        other_id,
                        rel.relationship_type,
                        min(1.0, jaccard * self.JACCARD_WEIGHT_SCALE),
                        0.05,
                    )
                )
            elif jaccard < 0.05 and union >= 10:
                updates.append(
                    (
                        concept.id,
                        other_id,
                        rel.relationship_type,
                        None,
                        -0.1,
                    )
                )

        return updates

    def _compute_quantification_proposals(self, concept: Concept, stats: dict[str, Any]) -> list[dict[str, Any]]:
        """Compute proposed AG quantification records.

        Returns list of proposal dicts with field_name, concept info, and stats.
        """
        proposals: list[dict[str, Any]] = []
        for field_name, field_stats in stats.items():
            if field_name.startswith("_"):
                continue
            n = field_stats.get("n", 0)
            if n < 5:
                continue
            proposals.append(
                {
                    "field_name": field_name,
                    "concept_name": concept.name,
                    "concept_id": concept.id,
                    "mean": field_stats["mean"],
                    "n": n,
                }
            )
        return proposals

    def _apply_updates(
        self,
        concept_id: str,
        edge_updates: list[tuple[str, str, str, float | None, float | None]],
        quant_proposals: list[dict[str, Any]],
        prefetched: Any = None,
    ) -> None:
        """RECORD LANE: Apply computed changes under write locks.

        Handles edge updates and AG quantification storage.
        """
        concept = self._atl.get(concept_id)
        if concept is None or not isinstance(concept, Concept):
            return  # Concept evicted

        # Apply edge updates
        for source_id, target_id, rel_type, weight, confidence_delta in edge_updates:
            try:
                kwargs: dict[str, Any] = {}
                if weight is not None:
                    kwargs["weight"] = weight
                if confidence_delta is not None:
                    kwargs["confidence_delta"] = confidence_delta
                self._atl.semantics.update_edge(source_id, target_id, rel_type, **kwargs)
            except Exception as e:
                logger.debug("Failed to update edge %s→%s: %s", source_id, target_id, e)

        # Apply quantification proposals
        for proposal in quant_proposals:
            try:
                self._store_single_quantification(concept, proposal)
            except Exception as e:
                logger.debug("Failed to store quantification for %s: %s", proposal.get("field_name"), e)

    def _store_single_quantification(self, concept: Concept, proposal: dict[str, Any]) -> None:
        """Store a single AG MathMemory record from a quantification proposal."""
        from maxim.math.math_types import MathCategory, MathMemory
        from maxim.memory.cross_layer import CrossLayerEdgeType

        field_name = proposal["field_name"]
        concept_name = proposal["concept_name"]
        mean = proposal["mean"]
        n = proposal["n"]

        existing_name = f"{concept_name}:{field_name}"
        existing = self._ag.recall(limit=1, name=existing_name)

        if existing:
            record = existing[0]
            if isinstance(record, MathMemory):
                record.verbal = f"{concept_name} {field_name}: mean={mean:.2f} (n={n})"
                record.observation_count = n
                record.confidence = min(0.9, 0.3 + 0.05 * n)
                record.touch()
        else:
            record = MathMemory(
                id=str(uuid4()),
                timestamp=time.time(),
                name=existing_name,
                category=MathCategory.PATTERN,
                domain="concept_property",
                verbal=f"{concept_name} {field_name}: mean={mean:.2f} (n={n})",
                code="",
                inputs=[concept_name],
                outputs=[field_name],
                source="derived",
                confidence=min(0.9, 0.3 + 0.05 * n),
                observation_count=n,
            )
            record_id = self._ag.store(record)

            self._cross_layer.add_edge(
                source_layer="angular_gyrus",
                source_id=record_id,
                target_layer="atl",
                target_id=concept.id,
                edge_type=CrossLayerEdgeType.QUANTIFIES,
                weight=1.0,
                metadata={"field": field_name},
            )

            concept.add_ref("angular_gyrus", record_id)

    # ------------------------------------------------------------------
    # Numeric extraction
    # ------------------------------------------------------------------

    def _extract_numerics(self, episodes: list[EpisodicMemory | CompressedMemory]) -> dict[str, list[float]]:
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

    def _compute_stats(self, concept: Concept, numerics: dict[str, list[float]]) -> dict[str, Any]:
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
            if len(values) >= 8 and trend and 0.3 < trend.confidence < 0.65:
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
        relationships = self._atl.find_by_relationship(concept.id, direction="both", limit=50)
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
                    concept.id,
                    other_id,
                    rel.relationship_type,
                    weight=min(1.0, jaccard * self.JACCARD_WEIGHT_SCALE),
                    confidence_delta=0.05,
                )
            # Weaken: low co-occurrence despite many total observations
            elif jaccard < 0.05 and union >= 10:
                self._atl.semantics.update_edge(
                    concept.id,
                    other_id,
                    rel.relationship_type,
                    confidence_delta=-0.1,
                )

    # ------------------------------------------------------------------
    # AG quantification storage
    # ------------------------------------------------------------------

    def _store_quantifications(self, concept: Concept, stats: dict[str, Any]) -> None:
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
                    record.verbal = f"{concept.name} {field_name}: mean={field_stats['mean']:.2f} (n={n})"
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
                    verbal=(f"{concept.name} {field_name}: mean={field_stats['mean']:.2f} (n={n})"),
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
