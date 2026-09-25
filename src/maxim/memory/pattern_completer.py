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
import math
import threading
from typing import TYPE_CHECKING, Any

from maxim.memory.semantic_types import Concept
from maxim.memory.text import normalize_tokens
from maxim.memory.types import (
    CompressedMemory,
    EpisodicMemory,
    MathContextEntry,
    PredictedOutcome,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from maxim.memory.atl import ATL
    from maxim.memory.layer import MemoryLayer

logger = logging.getLogger(__name__)

# 2S-d: a situation cue qualifies AND ranks a memory only by its shared EXTEROCEPTIVE clusters, in this
# order (the place outranks the sound). Interoception plays no part: its cluster is broad (cosine
# separates only a neutral->extreme swing: docs/wiring/cosine-separation-is-directional.md), so as a
# qualifier it would recall the latest memories from anywhere, and as a ranker it would put safe past
# visits (full air, matching the cue) above the drownings (extreme cluster, not matching) at the very
# moment the drownings are the ones to recall -- the Exp 60 failure (owner decision (b), 2026-09-25).
SITUATION_RANK_ORDER: tuple[str, ...] = ("world", "audio")
SITUATION_QUALIFYING_MODALITIES: frozenset[str] = frozenset(SITUATION_RANK_ORDER)


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
        # 2S-d: the last situation cued per agent (in memory only; reset at each session start).
        self._situation_lock = threading.Lock()
        self._last_situation: dict[str, dict[str, str]] = {}
        self._situation_stats = {"cues": 0, "changes": 0, "with_matches": 0, "matched": 0}

    # ── 2S-d: the situation cue ───────────────────────────────────────────

    def cue_situation(self, agent_id: str, situation: "Mapping[str, str] | None") -> tuple[str, ...]:
        """Recall the memories formed in this situation, when the situation CHANGES (memory 2S-d).

        ``situation`` is the tick's ``{modality: cluster_id}`` (the ids are ATL concept ids). On a
        change from this agent's last cue, the concepts with those ids nominate the memories linked
        to them (2S-b's ``memory_refs['hippocampus']``, a lossy index); each candidate is judged on
        its OWN recorded ``situation``: it qualifies only through a shared world/audio cluster, and
        the TIER is the highest-ranked modality in ``SITUATION_RANK_ORDER`` it shares (any same-place
        memory; same-sound only when no place matches; interoception never ranks). Within the tier the
        most SALIENT memories come first (``max(encoding_tag, retro_tag)``: pain, surprise, novelty,
        drive, or a strong moment just after), then those also sharing a lower-ranked modality, then
        the newest by experience time,
        capped at ``MAX_EPISODES`` -- so a long run of uneventful visits cannot crowd out the one that
        hurt. The same situation again returns ``()``.

        **Recall only (owner decision, 2026-09-25):** nothing is activated -- no counter, no strength
        credit. ``MemoryLayer.activate`` is for a CONSUMPTION point, and nothing consumes these ids
        until 2S-e, where the outcome is known and can gate the credit. Every read here is a
        non-touching bulk read. Returns the recalled record ids.
        """
        cue = {m: c for m, c in (situation or {}).items() if isinstance(m, str) and isinstance(c, str)}
        with self._situation_lock:
            self._situation_stats["cues"] += 1
            previous = self._last_situation.get(agent_id)
            if previous == cue:
                return ()
            self._last_situation[agent_id] = cue
            self._situation_stats["changes"] += 1
        try:
            recalled = self.recall_situation(cue)
        except Exception:
            # A failed recall must not mark this situation as already cued: roll back, so the next
            # tick in the same situation tries again instead of staying silent until it changes.
            with self._situation_lock:
                if self._last_situation.get(agent_id) == cue:
                    if previous is None:
                        self._last_situation.pop(agent_id, None)
                    else:
                        self._last_situation[agent_id] = previous
            raise
        with self._situation_lock:
            if recalled:
                self._situation_stats["with_matches"] += 1
                self._situation_stats["matched"] += len(recalled)
        return recalled

    def recall_situation(self, cue: "Mapping[str, str]") -> tuple[str, ...]:
        """The recall itself, with NO change detection and no state: the seam 2S-e calls with a cue it
        completed to a neighbouring situation (going through ``cue_situation`` would overwrite the
        agent's last situation and corrupt change detection). Same selection rule as ``cue_situation``.
        """
        cue = dict(cue)
        hippocampus = self._layers.get("hippocampus")
        qualifying = {m: c for m, c in cue.items() if m in SITUATION_QUALIFYING_MODALITIES}
        if hippocampus is None or self._atl is None or not qualifying:
            return ()
        candidate_ids: set[str] = set()
        for concept in self._atl.recall_by_ids(list(qualifying.values())):  # no touch
            refs = getattr(concept, "memory_refs", None)
            if refs:
                # Copied before scoring, but NOT under the ATL lock: the extractor thread can resize
                # this dict mid-copy ("dictionary changed size"). The caller's fail-soft wrapper
                # absorbs that, and ``cue_situation`` rolls back, so the next tick retries.
                candidate_ids.update(tuple(refs.get("hippocampus", {})))
        if not candidate_ids:
            return ()
        # (tier, record, how many lower-ranked modalities it also shares). A CompressedMemory carries no
        # ``situation``, so it never matches and never reaches the sort.
        matched: list[tuple[int, Any, int]] = []
        for record in hippocampus.recall_by_ids(list(candidate_ids)):  # no touch
            own = getattr(record, "situation", None) or {}
            hits = [m in qualifying and own.get(m) == qualifying[m] for m in SITUATION_RANK_ORDER]
            if any(hits):
                first = hits.index(True)  # the tier: the HIGHEST-ranked modality it shares
                matched.append((first, record, sum(hits[first + 1 :])))
        if not matched:
            return ()
        # Lower index = higher rank. Only the tier is exclusive: a same-place memory whose SOUND differs
        # is still in the place tier (the drowning's hurt sound must not drop it below safe swims that
        # share today's splash); the lower-ranked matches only order memories inside the tier.
        best = min(first for first, _, _ in matched)
        tier = [(r, extra) for first, r, extra in matched if first == best]

        def salience(r: Any) -> float:
            # The same measure the strength floor reads (2d-2): the stamped tag or a retro tag, whichever
            # is higher -- the moments just before a drowning are retro-tagged, not uneventful.
            values = [getattr(r, "encoding_tag", None), getattr(r, "retro_tag", None)]
            finite = [float(v) for v in values if v is not None and math.isfinite(float(v))]
            return max(finite, default=0.0)

        def order(item: tuple[Any, int]) -> tuple[float, int, int, int]:
            r, extra = item
            at, seq = getattr(r, "encoded_at_us", None), getattr(r, "capture_seq", None)
            return (salience(r), extra, -1 if at is None else at, -1 if seq is None else seq)

        tier.sort(key=order, reverse=True)
        return tuple(r.id for r, _ in tier[: self.MAX_EPISODES])

    def reset_situations(self) -> None:
        """Forget every agent's last situation, so the next cue is an ENTRY (a new session)."""
        with self._situation_lock:
            self._last_situation.clear()

    def situation_cue_stats(self) -> dict[str, int]:
        """``cues`` (calls), ``changes`` (situation changed), ``with_matches``, ``matched`` (ids).

        Cumulative for this completer's life (repeated sessions add up). A recall that raised is rolled
        back and retried, so its ``cues``/``changes`` count again on the retry.
        """
        with self._situation_lock:
            return dict(self._situation_stats)

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

        # 1b. Discover skill concepts via EXECUTES_WITH edges (A7.4b)
        if self._atl is not None:
            try:
                skill_concepts = self._atl.recall(
                    limit=10,
                    category="skill_execution",
                )
                episode_concept_ids = {c.id for c in concepts}
                for sc in skill_concepts:
                    if isinstance(sc, Concept):
                        rels = self._atl.find_by_relationship(
                            sc.id,
                            rel_type="EXECUTES_WITH",
                            direction="outgoing",
                            limit=20,
                        )
                        if any(oid in episode_concept_ids for oid, _ in rels):
                            concepts.append(sc)
            except Exception as e:
                logger.debug("Skill concept discovery failed: %s", e)

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
        episodes = sorted(all_episodes, key=lambda ep: ep.timestamp, reverse=True)[: self.MAX_EPISODES]

        # 4. Extract predictions from past outcomes
        predictions: list[PredictedOutcome] = []
        for ep in episodes:
            if isinstance(ep, CompressedMemory):
                predictions.append(
                    PredictedOutcome(
                        tool=ep.tool_name,
                        success=ep.success,
                        goal=ep.goal,
                        confidence=0.3,  # No decision.confidence on compressed
                        source_episode_id=ep.id,
                    )
                )
            elif isinstance(ep, EpisodicMemory):
                goal = None
                if isinstance(ep.decision.intent, dict):
                    goal = ep.decision.intent.get("goal")
                predictions.append(
                    PredictedOutcome(
                        tool=ep.action.tool_name,
                        success=ep.outcome.success,
                        goal=goal,
                        confidence=ep.decision.confidence,
                        source_episode_id=ep.id,
                    )
                )

        # 5. Enrich with per-concept math context using memory_refs
        # intersection. A prediction matches a concept if the prediction's
        # source episode is in the concept's memory_refs.
        for concept in concepts:
            layer_context = self._get_concept_layer_context(concept)
            if not layer_context:
                continue
            concept_episode_ids = set(concept.memory_refs.get("hippocampus", {}))
            for pred in predictions:
                if pred.source_episode_id in concept_episode_ids:
                    pred.math_context = layer_context

        # The episodes completed into predictions were reactivated (memory-strength Phase 1); the
        # cue concepts that led to them were not -- seeds are excluded, as in spreading activation.
        from maxim.memory.layer import activate_after_use

        activate_after_use(hippocampus, (p.source_episode_id for p in predictions), source="prediction")
        return predictions

    def _find_matching_concepts(self, episodic: EpisodicMemory) -> list[Concept]:
        """Find concepts matching the percept's objects, people, and goal.

        Objects and people are single-word concept names, so direct lookup
        works. Goals are tokenized via normalize_tokens().
        """
        matches: list[Concept] = []
        seen: set[str] = set()

        search_terms: list[str] = list(episodic.perception.detected_objects + episodic.perception.detected_people)

        if episodic.context.active_goal:
            search_terms.extend(normalize_tokens(episodic.context.active_goal))

        for term in search_terms:
            results = self._atl.recall(limit=1, name=term.lower())
            for concept in results:
                if isinstance(concept, Concept) and concept.id not in seen:
                    matches.append(concept)
                    seen.add(concept.id)

        return matches

    def _get_concept_layer_context(self, concept: Concept) -> list[MathContextEntry] | None:
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
                    entries.append(
                        MathContextEntry(
                            name=record.name,
                            verbal=record.verbal,
                            confidence=record.confidence,
                            domain=record.domain,
                        )
                    )

        return entries if entries else None


__all__ = ["PatternCompleter"]
