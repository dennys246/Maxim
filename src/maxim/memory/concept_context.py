"""ConceptContextBuilder — builds LLM context entries from ATL concepts.

Standalone collaborator in the integration layer. For each concept matching
the current percept, iterates registered memory layers to collect enrichment
context (AG stats, relationships). Same pattern as ConceptExtractor and
ConceptGrounder.

Brain mapping: Concept-aware recall. When building context for the LLM,
looks up concepts matching the current percept and iterates registered layers
to collect context for each concept. Currently AG stats, but extensible to
future layer types (social, emotional, spatial).
"""

from __future__ import annotations

import logging
import time as _time
from typing import TYPE_CHECKING, Any

from maxim.memory.semantic_types import Concept
from maxim.memory.text import normalize_tokens

if TYPE_CHECKING:
    from maxim.memory.atl import ATL
    from maxim.memory.concept_grounder import ConceptGrounder
    from maxim.memory.layer import MemoryLayer

logger = logging.getLogger(__name__)


class ConceptContextBuilder:
    """Builds LLM context entries from ATL concepts and registered layers.

    Standalone collaborator in the integration layer. Iterates registered
    memory layers to collect enrichment context for each concept matching
    the current percept. Same pattern as ConceptExtractor and ConceptGrounder.

    Wired in: MemoryHub._wire_multi_layer()
    Called by: MemoryAgent.build_context() via MemoryHub
    """

    # Default time budget for concept grounding on the recall hot path.
    GROUNDING_BUDGET_MS: float = 50.0

    def __init__(
        self,
        atl: ATL,
        layers: dict[str, MemoryLayer],
        concept_grounder: ConceptGrounder | None = None,
    ) -> None:
        self._atl = atl
        self._layers = layers  # "hippocampus" -> Hippocampus, "angular_gyrus" -> AG, etc.
        self._concept_grounder = concept_grounder

    def build(
        self,
        detected_objects: list[str] | None = None,
        detected_people: list[str] | None = None,
        active_goal: str | None = None,
        limit: int = 5,
        budget_ms: float | None = None,
    ) -> list[dict]:
        """Build concept context entries for the current percept.

        Synchronous — AG grounding is arithmetic on in-memory arrays, not
        blocking I/O. Budget bounds total grounding work across all concepts.

        For each matching concept:
        1. Look up concept in ATL by percept names and goal tokens
        2. Load linked episodes from hippocampus layer
        3. Run ConceptGrounder (synchronous, budget-bounded) for AG stats
        4. Iterate all non-hippocampus layers for additional enrichment
        5. Collect relationships (always — cheap graph lookup)

        If the budget is exhausted, remaining concepts get relationship
        context only (no numerical stats). Graceful degradation.

        Returns list of context dicts for StructuredContext.concept_context.
        """
        if budget_ms is None:
            budget_ms = self.GROUNDING_BUDGET_MS

        concepts = self._find_matching_concepts(
            detected_objects or [],
            detected_people or [],
            active_goal,
        )
        if not concepts:
            return []

        # Rank by confidence, take top `limit`
        concepts.sort(key=lambda c: c.confidence, reverse=True)
        concepts = concepts[:limit]

        context_entries: list[dict] = []
        budget_start = _time.monotonic()
        budget_exhausted = False

        hippocampus = self._layers.get("hippocampus")

        for concept in concepts:
            # Load linked episodes (for AG grounding)
            episodes: list[Any] = []
            if hippocampus:
                episode_ids = list(concept.memory_refs.get("hippocampus", {}))
                episodes = hippocampus.recall_by_ids(episode_ids[:20])

            # AG grounding — synchronous with time budget
            stats: dict[str, Any] = {}
            if self._concept_grounder and episodes and not budget_exhausted:
                elapsed_ms = (_time.monotonic() - budget_start) * 1000
                if elapsed_ms < budget_ms:
                    stats = self._concept_grounder.ground_concept(
                        concept, episodes
                    )
                else:
                    budget_exhausted = True
                    logger.debug(
                        "Concept grounding budget exhausted (%.1fms), "
                        "skipping AG stats for remaining %d concepts",
                        elapsed_ms,
                        len(concepts) - len(context_entries),
                    )

            # Iterate layers for additional enrichment context
            layer_enrichment = self._collect_layer_enrichment(concept)

            # Get relationships (always — cheap O(edges) lookup)
            rel_summaries = self._collect_relationships(concept)

            # Format AG quantifications
            ag_props: dict[str, Any] = {}
            for field_name, field_stats in stats.items():
                if field_name.startswith("_"):
                    continue
                if not isinstance(field_stats, dict):
                    continue
                ag_props[field_name] = {
                    "mean": field_stats.get("mean"),
                    "n": field_stats.get("n"),
                    "trend": field_stats.get("trend"),
                }

            entry: dict[str, Any] = {
                "name": concept.name,
                "category": concept.category,
                "confidence": concept.confidence,
                "episode_count": concept.ref_count("hippocampus"),
                "relationships": rel_summaries,
                "properties": ag_props,
            }
            if layer_enrichment:
                entry["layer_context"] = layer_enrichment

            context_entries.append(entry)

        return context_entries

    def _find_matching_concepts(
        self,
        detected_objects: list[str],
        detected_people: list[str],
        active_goal: str | None,
    ) -> list[Concept]:
        """Find ATL concepts matching the percept's objects, people, and goal.

        Objects and people are single-word concept names, so direct lookup
        works. Goals are free-form phrases, so we tokenize via
        normalize_tokens() — same approach as ConceptExtractor.
        """
        matches: list[Concept] = []
        seen: set[str] = set()
        search_terms: list[str] = list(detected_objects) + list(detected_people)

        # Goal: tokenize, filter stop words, lemmatize
        if active_goal:
            search_terms.extend(normalize_tokens(active_goal))

        for term in search_terms:
            results = self._atl.recall(limit=1, name=term.lower())
            for concept in results:
                if isinstance(concept, Concept) and concept.id not in seen:
                    matches.append(concept)
                    seen.add(concept.id)

        return matches

    def _collect_layer_enrichment(self, concept: Concept) -> list[dict]:
        """Iterate registered layers for enrichment context.

        Skips hippocampus (provides episodes, not enrichment). For each
        layer, loads records referenced by the concept's memory_refs and
        formats them.
        """
        from maxim.math.math_types import MathMemory

        entries: list[dict] = []
        for layer_name, layer in self._layers.items():
            if layer_name == "hippocampus":
                continue
            ref_ids = concept.memory_refs.get(layer_name, {})
            if not ref_ids:
                continue
            records = layer.recall_by_ids(list(ref_ids)[:5])
            for record in records:
                if isinstance(record, MathMemory):
                    entries.append({
                        "layer": layer_name,
                        "name": record.name,
                        "verbal": record.verbal,
                        "confidence": record.confidence,
                    })
        return entries

    def _collect_relationships(self, concept: Concept) -> list[dict]:
        """Collect relationship summaries for a concept."""
        relationships = self._atl.find_by_relationship(
            concept.id, direction="both", limit=10
        )
        summaries: list[dict] = []
        for other_id, rel in relationships:
            other = self._atl.get(other_id)
            if other:
                summaries.append({
                    "type": rel.relationship_type,
                    "target": other.name,
                    "confidence": rel.confidence,
                })
        return summaries


__all__ = ["ConceptContextBuilder"]
