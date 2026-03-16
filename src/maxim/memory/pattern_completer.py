"""PatternCompleter — predicts outcomes via concept graph chaining.

Traverses ATL concepts → linked episodes → past decisions/actions/outcomes,
enriched with per-concept math context from registered layers.

Separated from ATL to avoid god-object accumulation. ATL stores concepts;
PatternCompleter queries them. Same pattern as ConceptGrounder and
ConceptExtractor.

Brain mapping: Pattern completion. When a new episode is forming, the brain
predicts likely outcomes by activating concept representations and retrieving
associated experiences. This module implements that retrieval chain.

Wired into MemoryAgent via set_pattern_completion_fn(completer.complete).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from maxim.memory.semantic_types import Concept
from maxim.memory.text import normalize_tokens
from maxim.memory.types import (
    CompressedMemory,
    EpisodicMemory,
    MathContextEntry,
    PredictedOutcome,
)

if TYPE_CHECKING:
    from maxim.memory.atl import ATL
    from maxim.memory.layer import MemoryLayer

logger = logging.getLogger(__name__)


class PatternCompleter:
    """Predicts outcomes for partially-formed episodes via concept graph chaining.

    Traverses ATL concepts → linked episodes → past decisions/actions/outcomes,
    enriched with per-concept math context from registered layers.

    Separated from ATL to avoid god-object accumulation. ATL stores concepts;
    PatternCompleter queries them. Same separation as ConceptGrounder.

    Wired into MemoryAgent via set_pattern_completion_fn(completer.complete).
    """

    MAX_EPISODES: int = 20

    def __init__(
        self,
        atl: ATL,
        layers: dict[str, MemoryLayer],
    ) -> None:
        self._atl = atl
        self._layers = layers

    def complete(self, episodic: EpisodicMemory) -> list[PredictedOutcome]:
        """Pattern completion function wired into MemoryAgent.

        Called during FORMING stage with partial EpisodicMemory
        (has Perception+Context, lacks Decision/Action/Outcome).
        Returns predicted outcomes from similar past experiences.
        """
        # 1. Find matching concepts from percept
        concepts = self._find_matching_concepts(episodic)
        if not concepts:
            return []

        # 2. Collect episode IDs from concept refs (deduplicated)
        hippocampus = self._layers.get("hippocampus")
        if not hippocampus:
            return []

        episode_ids: set[str] = set()
        for concept in concepts:
            episode_ids.update(concept.memory_refs.get("hippocampus", {}))

        if not episode_ids:
            return []

        # 3. Load ALL matched episodes, sort by recency, then cap.
        # Loading first ensures we get the most recent ones, not an
        # arbitrary subset from set iteration order. recall_by_ids is
        # an in-memory dict lookup so loading all is cheap.
        all_episodes = hippocampus.recall_by_ids(list(episode_ids))
        episodes = sorted(
            all_episodes, key=lambda ep: ep.timestamp, reverse=True
        )[:self.MAX_EPISODES]

        # 4. Extract predictions from past outcomes
        predictions: list[PredictedOutcome] = []
        for ep in episodes:
            if isinstance(ep, CompressedMemory):
                predictions.append(PredictedOutcome(
                    tool=ep.tool_name,
                    success=ep.success,
                    goal=ep.goal,
                    confidence=0.3,  # No decision.confidence on compressed
                    source_episode_id=ep.id,
                ))
            elif isinstance(ep, EpisodicMemory):
                goal = None
                if isinstance(ep.decision.intent, dict):
                    goal = ep.decision.intent.get("goal")
                predictions.append(PredictedOutcome(
                    tool=ep.action.tool_name,
                    success=ep.outcome.success,
                    goal=goal,
                    confidence=ep.decision.confidence,
                    source_episode_id=ep.id,
                ))

        # 5. Enrich with per-concept math context using memory_refs
        # intersection. A prediction matches a concept if the prediction's
        # source episode is in the concept's memory_refs.
        for concept in concepts:
            layer_context = self._get_concept_layer_context(concept)
            if not layer_context:
                continue
            concept_episode_ids = set(
                concept.memory_refs.get("hippocampus", {})
            )
            for pred in predictions:
                if pred.source_episode_id in concept_episode_ids:
                    pred.math_context = layer_context

        return predictions

    def _find_matching_concepts(
        self, episodic: EpisodicMemory
    ) -> list[Concept]:
        """Find concepts matching the percept's objects, people, and goal.

        Objects and people are single-word concept names, so direct lookup
        works. Goals are tokenized via normalize_tokens().
        """
        matches: list[Concept] = []
        seen: set[str] = set()

        search_terms: list[str] = list(
            episodic.perception.detected_objects
            + episodic.perception.detected_people
        )

        if episodic.context.active_goal:
            search_terms.extend(
                normalize_tokens(episodic.context.active_goal)
            )

        for term in search_terms:
            results = self._atl.recall(limit=1, name=term.lower())
            for concept in results:
                if isinstance(concept, Concept) and concept.id not in seen:
                    matches.append(concept)
                    seen.add(concept.id)

        return matches

    def _get_concept_layer_context(
        self, concept: Concept
    ) -> list[MathContextEntry] | None:
        """Get enrichment context from registered layers for a concept.

        Skips hippocampus (provides episodes, not enrichment). Uses ID-based
        lookup from concept.memory_refs. Currently only AG produces
        MathContextEntry.
        """
        from maxim.math.math_types import MathMemory

        entries: list[MathContextEntry] = []

        for layer_name, layer in self._layers.items():
            if layer_name == "hippocampus":
                continue

            ref_ids = concept.memory_refs.get(layer_name, {})
            if not ref_ids:
                continue

            records = layer.recall_by_ids(list(ref_ids)[:5])
            for record in records:
                if isinstance(record, MathMemory):
                    entries.append(MathContextEntry(
                        name=record.name,
                        verbal=record.verbal,
                        confidence=record.confidence,
                        domain=record.domain,
                    ))

        return entries if entries else None


__all__ = ["PatternCompleter"]
