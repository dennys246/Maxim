"""Nucleus Accumbens (NAc) - Causal inference and reward prediction.

Learns event → outcome relationships through temporal difference learning,
enabling prediction of outcomes before taking actions.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from maxim.decisions.causal_link import (
    CausalLink,
    OutcomePrediction,
    TemporalDelta,
    Valence,
)

if TYPE_CHECKING:
    from maxim.memory.semantic_promoter import PromotionCandidate

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NACConfig:
    """Configuration for Nucleus Accumbens."""

    max_links: int = 10000  # Maximum causal links to track
    min_confidence_threshold: float = 0.3  # Min confidence to use for predictions
    decay_interval_hours: float = 24.0  # How often to decay unused links
    context_similarity_threshold: float = 0.5  # Min context match for retrieval
    temporal_window_seconds: float = 300.0  # Max time between event and outcome
    max_pending_events: int = 100  # Hard cap on awaiting-outcome event buffer
    enable_hippocampus_queries: bool = True  # Query Hippocampus for similar episodes
    base_learning_rate: float = 0.2  # Rescorla-Wagner base learning rate
    use_ec_similarity: bool = False  # Phase 3 flag, default OFF
    persistence_path: str | None = None  # Path for save/load (set by AgentFactory)

    # P2: Reward modulation of recognition
    reward_bias_alpha: float = 0.15  # Threshold modulation strength
    reward_bias_decay_tau: float = 50.0  # Decay timescale (ticks) for reward bias
    max_reward_bias: float = 0.20  # Cap on how much bias can lower EC threshold


class NAc:
    """Nucleus Accumbens - Causal inference and reward prediction engine.

    Learns event → outcome relationships through observation, enabling
    prediction of outcomes before taking actions.

    Integration points:
    - SCN: Temporal context for when causal patterns apply
    - Hippocampus: Query similar episodes for causal inference

    Example:
        nac = NAc()

        # Record an observation
        nac.observe(
            event_type="tool",
            event_signature="internet_search",
            outcome_type="tool_result",
            outcome_signature="success_with_results",
            outcome_valence=Valence.POSITIVE,
            delta_seconds=2.3,
            context={"mode": "exploration", "query_type": "factual"},
        )

        # Predict outcome for a potential action
        prediction = nac.predict(
            event_type="tool",
            event_signature="internet_search",
            context={"mode": "exploration", "query_type": "factual"},
        )
        if prediction:
            print(f"Expected: {prediction.predicted_outcome}")
            print(f"Confidence: {prediction.confidence:.2f}")
    """

    def __init__(self, config: NACConfig | None = None, ec: Any = None):
        self.config = config or NACConfig()

        # Thread safety: RLock for concurrent access from multi-agent party mode
        # and Mother Maxim's contribution processing. RLock (not Lock) because
        # record_outcome() calls _find_matching_link() which also reads _links.
        self._lock = threading.RLock()

        # Primary storage: event_signature → list of CausalLinks
        self._links: dict[str, list[CausalLink]] = {}

        # EC reference for causal pattern similarity registration (Phase 2)
        self._ec = ec

        # Provenance collector (wired by MaximAgent.wire_provenance)
        self._collector: Any = None

        # Index by outcome for reverse lookups
        self._outcome_index: dict[str, set[str]] = {}  # outcome_sig → link_ids

        # Pending events awaiting outcome attribution
        self._pending_events: list[dict[str, Any]] = []

        # Cold start priors: event_sig → (predicted_value, confidence)
        self._priors: dict[str, tuple[float, float]] = {}

        # P2: Per-node reward bias keyed by (agent_id, node_id).
        # Positive bias = node has been rewarded → EC should widen recognition radius.
        # Decays toward 0 over time when reinforcement stops.
        self._reward_bias: dict[tuple[str, str], float] = {}

        # P2: Eligibility traces — nodes that were recently active and
        # should receive credit when a reward arrives. Maps
        # (agent_id, node_id) → activation strength from PerceptTraceBuffer.
        self._eligibility: dict[tuple[str, str], float] = {}

        # Stats
        self._total_observations = 0
        self._last_decay_time = time.time()

    def _generate_link_id(self, event_sig: str, outcome_sig: str, context_hash: str) -> str:
        """Generate unique ID for a causal link."""
        combined = f"{event_sig}:{outcome_sig}:{context_hash}"
        return hashlib.sha256(combined.encode()).hexdigest()[:16]

    def _hash_context(self, context: dict[str, Any]) -> str:
        """Create a hashable representation of context."""
        sorted_items = sorted(context.items())
        return hashlib.sha256(str(sorted_items).encode()).hexdigest()[:8]

    def _context_similarity(self, ctx1: dict[str, Any], ctx2: dict[str, Any]) -> float:
        """Calculate similarity between two contexts (0.0-1.0)."""
        if not ctx1 or not ctx2:
            return 0.5  # Neutral if no context

        keys = set(ctx1.keys()) | set(ctx2.keys())
        if not keys:
            return 1.0

        matches = 0.0
        for key in keys:
            v1 = ctx1.get(key)
            v2 = ctx2.get(key)
            if v1 == v2:
                matches += 1
            elif v1 is not None and v2 is not None:
                if isinstance(v1, str) and isinstance(v2, str):
                    if v1.lower() == v2.lower():
                        matches += 0.8

        return matches / len(keys)

    # ─────────────────────────────────────────────────────────────────────────
    # COLD START PRIORS
    # ─────────────────────────────────────────────────────────────────────────

    def set_prior(
        self,
        event_type: str,
        event_id: str,
        predicted_value: float,
        confidence: float,
    ) -> None:
        """Set a prior prediction for cold start.

        Args:
            event_type: Type of event (e.g., "tool")
            event_id: Identifier (e.g., "internet_search")
            predicted_value: Expected outcome (0-1)
            confidence: Confidence in prior (typically low, ~0.3)
        """
        self._priors[f"{event_type}:{event_id}"] = (predicted_value, confidence)

    # ─────────────────────────────────────────────────────────────────────────
    # EVENT RECORDING
    # ─────────────────────────────────────────────────────────────────────────

    def record_event(
        self,
        event_type: str,
        event_signature: str,
        context: dict[str, Any] | None = None,
        memory_id: str | None = None,
    ) -> str:
        """Record an event that may later be linked to an outcome.

        Call this when an action is taken. Later, when an outcome occurs,
        call record_outcome() to create the causal link.

        Returns:
            Event ID for later outcome attribution.
        """
        with self._lock:
            now = time.time()
            event_id = f"{event_signature}:{time.time_ns()}"

            # Age-prune stale events (no outcome ever arrived within 2× the
            # temporal window) so the buffer doesn't leak in failure-heavy runs.
            stale_cutoff = now - (self.config.temporal_window_seconds * 2)
            if self._pending_events and self._pending_events[0]["timestamp"] < stale_cutoff:
                self._pending_events = [e for e in self._pending_events if e["timestamp"] >= stale_cutoff]

            self._pending_events.append(
                {
                    "id": event_id,
                    "type": event_type,
                    "signature": event_signature,
                    "context": context or {},
                    "memory_id": memory_id,
                    "timestamp": now,
                }
            )

            # Hard cap in case of pathological bursts (keep most recent).
            if len(self._pending_events) > self.config.max_pending_events:
                self._pending_events = self._pending_events[-self.config.max_pending_events :]

            return event_id

    def record_outcome(
        self,
        event_type: str,
        event_id: str,
        outcome_valence: Valence,
        context: dict[str, Any] | None = None,
        memory_id: str | None = None,
    ) -> list[CausalLink]:
        """Record an outcome and attribute it to recent events.

        Simplified API for common case where we know the event.

        Args:
            event_type: Type of event
            event_id: Event signature (e.g., tool name)
            outcome_valence: Quality of outcome
            context: Current context
            memory_id: Hippocampus memory ID if available

        Returns:
            List of CausalLinks that were updated.
        """
        return self.record_outcome_full(
            outcome_type="result",
            outcome_signature=f"{event_id}:{outcome_valence.value}",
            outcome_valence=outcome_valence,
            context=context,
            memory_id=memory_id,
            attributed_event_signature=event_id,
        )

    def record_outcome_full(
        self,
        outcome_type: str,
        outcome_signature: str,
        outcome_valence: Valence,
        context: dict[str, Any] | None = None,
        memory_id: str | None = None,
        attributed_event_id: str | None = None,
        attributed_event_signature: str | None = None,
    ) -> list[CausalLink]:
        """Record an outcome and attribute it to recent events (full API).

        Args:
            outcome_type: Category of outcome
            outcome_signature: Specific outcome identifier
            outcome_valence: Quality of outcome
            context: Current context for context-sensitive learning
            memory_id: Hippocampus memory ID if available
            attributed_event_id: If known, the specific event ID
            attributed_event_signature: If known, the event signature

        Returns:
            List of CausalLinks that were created or updated.
        """
        with self._lock:
            return self._record_outcome_impl(
                outcome_type,
                outcome_signature,
                outcome_valence,
                context,
                memory_id,
                attributed_event_id,
                attributed_event_signature,
            )

    def _record_outcome_impl(
        self,
        outcome_type: str,
        outcome_signature: str,
        outcome_valence: Valence,
        context: dict[str, Any] | None = None,
        memory_id: str | None = None,
        attributed_event_id: str | None = None,
        attributed_event_signature: str | None = None,
    ) -> list[CausalLink]:
        """Internal implementation — called under self._lock."""
        now = time.time()
        context = context or {}
        updated_links: list[CausalLink] = []

        # Find events within temporal window
        events_to_link = []
        remaining_events = []

        for event in self._pending_events:
            delta = now - event["timestamp"]

            if delta <= self.config.temporal_window_seconds:
                # Check if this is the attributed event
                if attributed_event_id and event["id"] == attributed_event_id:
                    events_to_link.append((event, delta))
                elif attributed_event_signature and event["signature"] == attributed_event_signature:
                    events_to_link.append((event, delta))
                elif not attributed_event_id and not attributed_event_signature:
                    # No specific attribution, check context similarity
                    ctx_sim = self._context_similarity(event["context"], context)
                    if ctx_sim >= self.config.context_similarity_threshold:
                        events_to_link.append((event, delta))
                    else:
                        remaining_events.append(event)
                else:
                    remaining_events.append(event)
            elif delta > self.config.temporal_window_seconds * 2:
                pass  # Event too old, discard
            else:
                remaining_events.append(event)

        self._pending_events = remaining_events

        # Create or update causal links
        for event, delta in events_to_link:
            ctx_hash = self._hash_context(event["context"])
            link_id = self._generate_link_id(event["signature"], outcome_signature, ctx_hash)

            # Find existing link or create new one
            event_links = self._links.setdefault(event["signature"], [])
            existing_link = None
            for link in event_links:
                if link.id == link_id:
                    existing_link = link
                    break

            if existing_link:
                existing_link.record_observation(
                    delta_seconds=delta,
                    valence=outcome_valence,
                    memory_id=memory_id or event.get("memory_id"),
                    context=context,
                )
                existing_link.update_prediction_rw(outcome_valence, learning_rate=self.config.base_learning_rate)
                # Register established causal patterns in EC similarity space
                if self._ec is not None and existing_link.observation_count >= 3:
                    self._register_causal_in_ec(existing_link)
                updated_links.append(existing_link)
            else:
                new_link = CausalLink(
                    id=link_id,
                    event_type=event["type"],
                    event_signature=event["signature"],
                    event_context=event["context"],
                    outcome_type=outcome_type,
                    outcome_signature=outcome_signature,
                    outcome_valence=outcome_valence,
                    temporal_delta=TemporalDelta(observed_deltas=(delta,)),
                    observation_count=1,
                    confidence=0.5,
                    memory_ids=[memory_id] if memory_id else [],
                )
                # Bootstrap RPE on first observation so callers
                # can gauge surprise even for novel events.
                new_link.update_prediction_rw(
                    outcome_valence,
                    learning_rate=self.config.base_learning_rate,
                )
                event_links.append(new_link)
                updated_links.append(new_link)

                # Update outcome index
                self._outcome_index.setdefault(outcome_signature, set()).add(link_id)

            self._total_observations += 1

        self._enforce_limits()

        # Log causal learning activity (P3g — Tier 2)
        if hasattr(self, "_collector") and self._collector and self._collector.verbosity >= 1:
            from maxim.provenance.types import PipelineStage, ProvenanceRef

            for link in updated_links:
                self._collector.log_activity(
                    PipelineStage.LEARNING,
                    "nac",
                    f"Causal: {link.event_signature} → {link.outcome_signature} "
                    f"(V={link.predicted_value:.2f}, n={link.observation_count})",
                    sources=[ProvenanceRef("nac", link.id, link.event_signature)],
                )

        # Simulation verbosity
        for link in updated_links:
            try:
                from maxim.simulation.sim_logger import sim_log

                sim_log(
                    "NAc",
                    f"Causal link: {link.event_signature} -> {link.outcome_valence.value} "
                    f"(RPE={link.last_rpe:.2f}, confidence={link.confidence:.2f})",
                )
            except Exception:
                pass

        return updated_links

    def observe(
        self,
        event_type: str,
        event_signature: str,
        outcome_type: str,
        outcome_signature: str,
        outcome_valence: Valence,
        delta_seconds: float,
        context: dict[str, Any] | None = None,
        memory_id: str | None = None,
    ) -> CausalLink:
        """Directly observe a complete causal relationship.

        Use this when you have both the event and outcome together — for
        example, when recording a plan outcome where you already know the
        tool that was called and whether it succeeded. Unlike
        :meth:`record_outcome`, this does NOT require the event to have
        been previously enqueued in ``_pending_events``; it creates the
        causal link directly.
        """
        context = context or {}
        ctx_hash = self._hash_context(context)
        link_id = self._generate_link_id(event_signature, outcome_signature, ctx_hash)

        with self._lock:
            event_links = self._links.setdefault(event_signature, [])

            # Find or create link
            existing_link = None
            for link in event_links:
                if link.id == link_id:
                    existing_link = link
                    break

            if existing_link:
                existing_link.record_observation(
                    delta_seconds=delta_seconds,
                    valence=outcome_valence,
                    memory_id=memory_id,
                    context=context,
                )
                existing_link.update_prediction_rw(outcome_valence, learning_rate=self.config.base_learning_rate)
                self._total_observations += 1
                return existing_link

            new_link = CausalLink(
                id=link_id,
                event_type=event_type,
                event_signature=event_signature,
                event_context=context,
                outcome_type=outcome_type,
                outcome_signature=outcome_signature,
                outcome_valence=outcome_valence,
                temporal_delta=TemporalDelta(observed_deltas=(delta_seconds,)),
                observation_count=1,
                confidence=0.5,
                memory_ids=[memory_id] if memory_id else [],
            )
            # Bootstrap RPE on first observation so callers
            # can gauge surprise even for novel events.
            new_link.update_prediction_rw(
                outcome_valence,
                learning_rate=self.config.base_learning_rate,
            )
            event_links.append(new_link)
            self._outcome_index.setdefault(outcome_signature, set()).add(link_id)
            self._total_observations += 1
            self._enforce_limits()
            return new_link

    # ─────────────────────────────────────────────────────────────────────────
    # PREDICTION
    # ─────────────────────────────────────────────────────────────────────────

    def predict(
        self,
        event_type: str,
        event_signature: str,
        context: dict[str, Any] | None = None,
    ) -> OutcomePrediction | None:
        """Predict the outcome of a potential event/action.

        Uses learned causal links to predict:
        - What outcome is likely
        - How long until the outcome
        - Confidence in the prediction

        Args:
            event_type: Type of event being considered
            event_signature: Signature of the potential event
            context: Current context for context-sensitive prediction

        Returns:
            OutcomePrediction if any relevant links exist, None otherwise.
        """
        with self._lock:
            return self._predict_impl(event_type, event_signature, context)

    def _predict_impl(
        self,
        event_type: str,
        event_signature: str,
        context: dict[str, Any] | None = None,
    ) -> OutcomePrediction | None:
        """Internal implementation — called under self._lock."""
        context = context or {}
        event_links = self._links.get(event_signature, [])

        # Phase 3: augment with EC-similar causal patterns
        if self._ec is not None and self.config.use_ec_similarity:
            try:
                from maxim.similarity.signature import SituationSignature

                sig = SituationSignature(
                    structural_hash=hash(event_signature),
                    temporal_hash=(0, 0, 0, 0),
                    tool_name=(event_signature.split(":")[-1] if ":" in event_signature else event_signature),
                    outcome_type="",
                    mode="",
                    goal_keywords=tuple((context or {}).get("goal", "").split()[:3]),
                    context_hash=(hash(frozenset(sorted((context or {}).items()))) if context else 0),
                    semantic_hash=(),
                )
                similar = self._ec.find_similar(sig, k=10, min_similarity=0.5)
                existing_ids = {id(lnk) for lnk in event_links}
                for causal_id, score in similar:
                    if causal_id.startswith("causal:"):
                        link_id = causal_id[7:]
                        link = self._get_link_by_id(link_id)
                        if link and id(link) not in existing_ids:
                            event_links = list(event_links) + [link]
                            existing_ids.add(id(link))
            except Exception:
                pass  # EC query is best-effort

        if not event_links:
            # Check priors
            prior_key = f"{event_type}:{event_signature}"
            if prior_key in self._priors:
                pred_val, conf = self._priors[prior_key]
                return OutcomePrediction(
                    event_signature=event_signature,
                    predicted_outcome="unknown",
                    predicted_valence=(Valence.POSITIVE if pred_val > 0.6 else Valence.NEUTRAL),
                    predicted_value=pred_val,
                    predicted_delay=0.0,
                    delay_bounds=(0.0, 0.0),
                    confidence=conf,
                    contributing_links=[],
                    context_match=0.0,
                )
            return None

        # Filter to high-confidence links
        valid_links = [link for link in event_links if link.confidence >= self.config.min_confidence_threshold]

        if not valid_links:
            return None

        # Score links by context similarity and confidence
        scored_links: list[tuple[CausalLink, float]] = []
        for link in valid_links:
            ctx_sim = self._context_similarity(link.event_context, context)
            score = link.confidence * (0.5 + 0.5 * ctx_sim)
            scored_links.append((link, score))

        scored_links.sort(key=lambda x: x[1], reverse=True)
        best_link, best_score = scored_links[0]

        # Context match floor — don't surface predictions from completely wrong contexts
        best_ctx_match = self._context_similarity(best_link.event_context, context)
        if best_ctx_match < 0.2:
            return None

        contributing_ids = [link.id for link, _ in scored_links[:5]]
        expected_delay, lower, upper = best_link.temporal_delta.predict_delay()

        return OutcomePrediction(
            event_signature=event_signature,
            predicted_outcome=best_link.outcome_signature,
            predicted_valence=best_link.outcome_valence,
            predicted_value=best_link.predicted_value,
            predicted_delay=expected_delay,
            delay_bounds=(lower, upper),
            confidence=best_score,
            contributing_links=contributing_ids,
            context_match=best_ctx_match,
        )

    def predict_all_outcomes(
        self,
        event_type: str,
        event_signature: str,
        context: dict[str, Any] | None = None,
    ) -> list[OutcomePrediction]:
        """Predict all possible outcomes for an event."""
        context = context or {}
        event_links = self._links.get(event_signature, [])

        predictions = []
        seen_outcomes: set[str] = set()

        for link in event_links:
            if link.confidence < self.config.min_confidence_threshold:
                continue
            if link.outcome_signature in seen_outcomes:
                continue

            ctx_sim = self._context_similarity(link.event_context, context)
            score = link.confidence * (0.5 + 0.5 * ctx_sim)
            expected_delay, lower, upper = link.temporal_delta.predict_delay()

            predictions.append(
                OutcomePrediction(
                    event_signature=event_signature,
                    predicted_outcome=link.outcome_signature,
                    predicted_valence=link.outcome_valence,
                    predicted_value=link.predicted_value,
                    predicted_delay=expected_delay,
                    delay_bounds=(lower, upper),
                    confidence=score,
                    contributing_links=[link.id],
                    context_match=ctx_sim,
                )
            )
            seen_outcomes.add(link.outcome_signature)

        predictions.sort(key=lambda p: p.confidence, reverse=True)
        return predictions

    # ─────────────────────────────────────────────────────────────────────────
    # QUERIES
    # ─────────────────────────────────────────────────────────────────────────

    def get_links_for_event(self, event_signature: str) -> list[CausalLink]:
        """Get all causal links originating from an event type."""
        return self._links.get(event_signature, [])

    def get_links_for_outcome(self, outcome_signature: str) -> list[CausalLink]:
        """Get all causal links leading to an outcome type."""
        link_ids = self._outcome_index.get(outcome_signature, set())
        result = []
        for links in self._links.values():
            for link in links:
                if link.id in link_ids:
                    result.append(link)
        return result

    def get_links_for(self, memory_id: str) -> list[CausalLink]:
        """Get all causal links referencing a memory ID."""
        result = []
        for links in self._links.values():
            for link in links:
                if memory_id in link.memory_ids:
                    result.append(link)
        return result

    def get_positive_outcomes(self, event_signature: str) -> list[CausalLink]:
        """Get links where this event led to positive outcomes."""
        return [link for link in self._links.get(event_signature, []) if link.outcome_valence == Valence.POSITIVE]

    def get_negative_outcomes(self, event_signature: str) -> list[CausalLink]:
        """Get links where this event led to negative outcomes."""
        return [link for link in self._links.get(event_signature, []) if link.outcome_valence == Valence.NEGATIVE]

    def get_promotion_candidates(
        self,
        min_confidence: float = 0.6,
        min_observations: int = 3,
    ) -> list["PromotionCandidate"]:
        """Scan all CausalLinks for promotion-worthy patterns.

        Implements PromotionSource protocol for the SemanticPromoter.
        Returns high-confidence positive causal links as candidates
        for ATL semantic memory promotion.
        """
        # Lazy import to break the circular dependency with maxim.memory
        # (NAc → memory.semantic_promoter → memory.__init__ → ... → nac).
        from maxim.memory.semantic_promoter import PromotionCandidate

        candidates: list[PromotionCandidate] = []
        for event_sig, links in self._links.items():
            for link in links:
                if (
                    link.confidence >= min_confidence
                    and link.observation_count >= min_observations
                    and link.outcome_valence == Valence.POSITIVE
                ):
                    candidates.append(
                        PromotionCandidate(
                            pattern_name=f"{link.event_signature} → {link.outcome_signature}",
                            category="causal_pattern",
                            confidence=link.confidence,
                            source_memory_ids=list(link.memory_ids),
                            metadata={
                                "event_signature": link.event_signature,
                                "outcome_signature": link.outcome_signature,
                                "valence": link.outcome_valence.value,
                                "observation_count": link.observation_count,
                            },
                        )
                    )
        return candidates

    def remove_memory(self, memory_id: str) -> None:
        """Remove a memory reference from all causal links."""
        for links in self._links.values():
            for link in links:
                if memory_id in link.memory_ids:
                    link.memory_ids.remove(memory_id)

    # ─────────────────────────────────────────────────────────────────────────
    # MAINTENANCE
    # ─────────────────────────────────────────────────────────────────────────

    def _get_link_by_id(self, link_id: str) -> CausalLink | None:
        """Find a CausalLink by its ID across all event signatures."""
        for links in self._links.values():
            for link in links:
                if link.id == link_id:
                    return link
        return None

    def _register_causal_in_ec(self, link: CausalLink) -> None:
        """Register an established causal pattern in EC for similarity queries."""
        try:
            from maxim.similarity.signature import SituationSignature

            sig = SituationSignature(
                structural_hash=hash(f"{link.event_signature}:{link.outcome_signature}"),
                temporal_hash=(0, 0, 0, 0),
                tool_name=(
                    link.event_signature.split(":")[-1] if ":" in link.event_signature else link.event_signature
                ),
                outcome_type=link.outcome_valence.value,
                mode="",
                goal_keywords=(
                    tuple(link.event_context.get("goal", "").split()[:3]) if link.event_context.get("goal") else ()
                ),
                context_hash=(hash(frozenset(sorted(link.event_context.items()))) if link.event_context else 0),
                semantic_hash=(),
            )
            self._ec.register(f"causal:{link.id}", sig)
        except Exception:
            pass  # EC registration is best-effort

    def _register_imported_link(self, link: CausalLink) -> None:
        """Register an externally-imported CausalLink.

        The link should already have transfer discount applied to confidence,
        predicted_value reset, and provenance tagged in event_context.
        """
        event_links = self._links.setdefault(link.event_signature, [])
        event_links.append(link)
        self._outcome_index.setdefault(link.outcome_signature, set()).add(link.id)
        self._register_causal_in_ec(link)

    def _deregister_causal_from_ec(self, link_id: str) -> None:
        """Remove a causal pattern from EC when link is evicted."""
        if self._ec is not None:
            try:
                self._ec.remove_signature(f"causal:{link_id}")
            except Exception:
                pass

    def _enforce_limits(self) -> None:
        """Enforce max_links limit by removing lowest-confidence links."""
        total_links = sum(len(links) for links in self._links.values())

        if total_links <= self.config.max_links:
            return

        # Collect all links with confidence
        all_links: list[tuple[str, CausalLink]] = []
        for event_sig, links in self._links.items():
            for link in links:
                all_links.append((event_sig, link))

        # Sort by confidence (lowest first)
        all_links.sort(key=lambda x: x[1].confidence)

        # Remove lowest until under limit
        to_remove = total_links - self.config.max_links
        for event_sig, link in all_links[:to_remove]:
            self._links[event_sig].remove(link)
            # Clean up EC registration
            self._deregister_causal_from_ec(link.id)
            # Clean up outcome index
            for outcome_sig, link_ids in list(self._outcome_index.items()):
                link_ids.discard(link.id)
                if not link_ids:
                    del self._outcome_index[outcome_sig]

    def decay_all(self, factor: float = 0.99) -> None:
        """Apply decay to all links."""
        for links in self._links.values():
            for link in links:
                link.decay(factor)

    # ─────────────────────────────────────────────────────────────────────────
    # P2: Reward Bias — per-node recognition modulation
    # ─────────────────────────────────────────────────────────────────────────

    def reward_bias(self, agent_id: str, node_id: str) -> float:
        """Get the current reward bias for a substrate node.

        Returns a value in [0, max_reward_bias]. Positive means the node
        has been rewarded and EC should lower its threshold for this node.
        """
        return self._reward_bias.get((agent_id, node_id), 0.0)

    def credit_node(
        self,
        agent_id: str,
        node_id: str,
        reward: float,
    ) -> None:
        """Credit a substrate node with reward, strengthening its recognition bias.

        Called when a Reaction (positive or negative) is attributed to a
        node via eligibility traces. Positive reward increases bias
        (widens recognition radius), negative reward decreases it.

        Args:
            agent_id: Agent whose recognition should be modulated.
            node_id: ATL node to credit.
            reward: Reward magnitude. Positive = reinforce, negative = weaken.
        """
        with self._lock:
            key = (agent_id, node_id)
            current = self._reward_bias.get(key, 0.0)
            updated = current + self.config.reward_bias_alpha * reward
            # Clamp to [0, max_reward_bias] — bias only widens, never inverts
            self._reward_bias[key] = max(0.0, min(updated, self.config.max_reward_bias))
            logger.debug(
                "NAc credit_node(%s, %s): %.4f → %.4f (reward=%.2f)",
                agent_id[:8] if agent_id else "?",
                node_id[:8],
                current,
                self._reward_bias[key],
                reward,
            )

    def update_eligibility(
        self,
        agent_id: str,
        node_id: str,
        activation: float,
    ) -> None:
        """Update the eligibility trace for a node.

        Called when a percept activates (completes to) a node. The
        activation strength determines how much credit the node receives
        when a reward arrives later.

        Args:
            agent_id: Agent context.
            node_id: ATL node that was activated.
            activation: Activation strength (typically from PerceptTraceBuffer).
        """
        with self._lock:
            self._eligibility[(agent_id, node_id)] = activation

    def distribute_reward(
        self,
        agent_id: str,
        reward: float,
    ) -> list[tuple[str, float]]:
        """Distribute reward to all eligible nodes for an agent.

        Credits each node proportional to its eligibility trace strength.
        Returns list of (node_id, credit_applied) for logging.
        """
        credited: list[tuple[str, float]] = []
        with self._lock:
            eligible = {
                (aid, nid): strength
                for (aid, nid), strength in self._eligibility.items()
                if aid == agent_id and strength > 0.01
            }
            if not eligible:
                return credited

            # Normalize eligibility so total credit = reward
            total_strength = sum(eligible.values())
            for (aid, nid), strength in eligible.items():
                proportion = strength / total_strength
                credit = reward * proportion
                self.credit_node(aid, nid, credit)
                credited.append((nid, credit))

        return credited

    def decay_reward_biases(self) -> int:
        """Decay all reward biases toward zero.

        Called periodically (e.g., on each tick). Uses exponential decay
        with timescale tau from config. Returns count of biases pruned.
        """
        if not self._reward_bias:
            return 0

        decay_factor = 1.0 / self.config.reward_bias_decay_tau
        pruned = 0
        with self._lock:
            to_remove = []
            for key, bias in self._reward_bias.items():
                new_bias = bias * (1.0 - decay_factor)
                if new_bias < 0.001:
                    to_remove.append(key)
                    pruned += 1
                else:
                    self._reward_bias[key] = new_bias

            for key in to_remove:
                del self._reward_bias[key]

        return pruned

    def decay_eligibility(self, factor: float = 0.9) -> None:
        """Decay all eligibility traces. Called on each tick."""
        with self._lock:
            to_remove = []
            for key, strength in self._eligibility.items():
                new_strength = strength * factor
                if new_strength < 0.01:
                    to_remove.append(key)
                else:
                    self._eligibility[key] = new_strength
            for key in to_remove:
                del self._eligibility[key]

    def get_threshold_overrides(self, agent_id: str) -> dict[str, float]:
        """Build EC threshold overrides from reward biases for an agent.

        Returns a dict of node_id → adjusted_threshold suitable for
        EC.pattern_complete_or_separate(threshold_override=...).

        Formula: threshold_override = base - α × reward_bias(agent_id, node)
        Clamped to [0.1, base] to prevent degenerate matching.
        """
        overrides: dict[str, float] = {}
        base = 0.40  # matches ECConfig.pattern_complete_threshold default
        with self._lock:
            for (aid, nid), bias in self._reward_bias.items():
                if aid != agent_id or bias < 0.001:
                    continue
                adjusted = base - bias
                overrides[nid] = max(0.10, adjusted)
        return overrides

    def stats(self) -> dict[str, Any]:
        """Return NAc statistics."""
        total_links = sum(len(links) for links in self._links.values())
        return {
            "total_links": total_links,
            "event_signatures": len(self._links),
            "outcome_signatures": len(self._outcome_index),
            "total_observations": self._total_observations,
            "pending_events": len(self._pending_events),
            "priors": len(self._priors),
            "reward_biases": len(self._reward_bias),
            "eligibility_traces": len(self._eligibility),
        }

    def __len__(self) -> int:
        """Total number of causal links."""
        return sum(len(links) for links in self._links.values())

    # ─────────────────────────────────────────────────────────────────────────
    # Persistence
    # ─────────────────────────────────────────────────────────────────────────

    def save(self, path: str | None = None) -> None:
        """Save NAc state to JSON file.

        If ``path`` is omitted, falls back to ``self.config.persistence_path``.
        Raises ``ValueError`` if neither is set.
        """
        path = path or self.config.persistence_path
        if path is None:
            raise ValueError("NAc.save() requires a path or NACConfig.persistence_path to be set")
        with self._lock:
            data = {
                "version": "1.0",
                "links": {event_sig: [link.to_dict() for link in links] for event_sig, links in self._links.items()},
                "outcome_index": {k: list(v) for k, v in self._outcome_index.items()},
                "priors": self._priors,
                "total_observations": self._total_observations,
                "reward_bias": {f"{aid}:{nid}": bias for (aid, nid), bias in self._reward_bias.items()},
            }

        from maxim.utils.atomic_io import atomic_write_json

        atomic_write_json(path, data)

        logger.info("Saved NAc to %s (%d links)", path, len(self))

    def load(self, path: str | None = None) -> None:
        """Load NAc state from JSON file.

        If ``path`` is omitted, falls back to ``self.config.persistence_path``.
        Raises ``ValueError`` if neither is set.
        """
        path = path or self.config.persistence_path
        if path is None:
            raise ValueError("NAc.load() requires a path or NACConfig.persistence_path to be set")
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        version = data.get("version", "0.0")
        if version != "1.0":
            raise ValueError(f"Unsupported NAc version: {version}")

        self._links = {
            event_sig: [CausalLink.from_dict(link_data) for link_data in links]
            for event_sig, links in data.get("links", {}).items()
        }
        self._outcome_index = {k: set(v) for k, v in data.get("outcome_index", {}).items()}
        self._priors = data.get("priors", {})
        self._total_observations = data.get("total_observations", 0)

        # P2: Load reward biases
        self._reward_bias = {}
        for key_str, bias in data.get("reward_bias", {}).items():
            parts = key_str.split(":", 1)
            if len(parts) == 2:
                self._reward_bias[(parts[0], parts[1])] = bias

        logger.info("Loaded NAc from %s (%d links, %d biases)", path, len(self), len(self._reward_bias))

    def load_safe(self, path: str | None = None) -> tuple[bool, str | None]:
        """Load with recovery on failure. Returns (success, error_message)."""
        path = path or self.config.persistence_path
        if path is None:
            raise ValueError("NAc.load_safe() requires a path or NACConfig.persistence_path to be set")
        if not os.path.exists(path):
            logger.info("No existing NAc file at %s, starting fresh", path)
            return True, None
        try:
            self.load(path)
            return True, None
        except (json.JSONDecodeError, ValueError, KeyError, TypeError) as e:
            error_msg = f"Corrupt NAc file ({type(e).__name__}): {e}"
            logger.warning("%s — starting with empty causal model", error_msg)
            self._links = {}
            self._outcome_index = {}
            self._priors = {}
            self._total_observations = 0
            return False, error_msg

    def get_version(self) -> str:
        """Return data format version."""
        return "1.0"


__all__ = ["NAc", "NACConfig"]
