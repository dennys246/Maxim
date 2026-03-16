"""Nucleus Accumbens (NAc) - Causal inference and reward prediction.

Learns event → outcome relationships through temporal difference learning,
enabling prediction of outcomes before taking actions.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from typing import Any

from maxim.decisions.causal_link import (
    CausalLink,
    OutcomePrediction,
    TemporalDelta,
    Valence,
)
from maxim.memory.semantic_promoter import PromotionCandidate

logger = logging.getLogger(__name__)


@dataclass
class NACConfig:
    """Configuration for Nucleus Accumbens."""

    max_links: int = 10000  # Maximum causal links to track
    min_confidence_threshold: float = 0.3  # Min confidence to use for predictions
    decay_interval_hours: float = 24.0  # How often to decay unused links
    context_similarity_threshold: float = 0.5  # Min context match for retrieval
    temporal_window_seconds: float = 300.0  # Max time between event and outcome
    enable_hippocampus_queries: bool = True  # Query Hippocampus for similar episodes
    base_learning_rate: float = 0.2  # Rescorla-Wagner base learning rate


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

    def __init__(self, config: NACConfig | None = None):
        self.config = config or NACConfig()

        # Primary storage: event_signature → list of CausalLinks
        self._links: dict[str, list[CausalLink]] = {}

        # Index by outcome for reverse lookups
        self._outcome_index: dict[str, set[str]] = {}  # outcome_sig → link_ids

        # Pending events awaiting outcome attribution
        self._pending_events: list[dict[str, Any]] = []

        # Cold start priors: event_sig → (predicted_value, confidence)
        self._priors: dict[str, tuple[float, float]] = {}

        # Stats
        self._total_observations = 0
        self._last_decay_time = time.time()

    def _generate_link_id(
        self, event_sig: str, outcome_sig: str, context_hash: str
    ) -> str:
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
        event_id = f"{event_signature}:{time.time_ns()}"
        self._pending_events.append(
            {
                "id": event_id,
                "type": event_type,
                "signature": event_signature,
                "context": context or {},
                "memory_id": memory_id,
                "timestamp": time.time(),
            }
        )

        # Limit pending events
        if len(self._pending_events) > 100:
            self._pending_events = self._pending_events[-100:]

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
                elif (
                    attributed_event_signature
                    and event["signature"] == attributed_event_signature
                ):
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
            link_id = self._generate_link_id(
                event["signature"], outcome_signature, ctx_hash
            )

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
                existing_link.update_prediction_rw(
                    outcome_valence, learning_rate=self.config.base_learning_rate
                )
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
                event_links.append(new_link)
                updated_links.append(new_link)

                # Update outcome index
                self._outcome_index.setdefault(outcome_signature, set()).add(link_id)

            self._total_observations += 1

        self._enforce_limits()
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

        Use this when you have both the event and outcome together.
        """
        context = context or {}
        ctx_hash = self._hash_context(context)
        link_id = self._generate_link_id(event_signature, outcome_signature, ctx_hash)

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
            existing_link.update_prediction_rw(
                outcome_valence, learning_rate=self.config.base_learning_rate
            )
            return existing_link
        else:
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
        context = context or {}
        event_links = self._links.get(event_signature, [])

        if not event_links:
            # Check priors
            prior_key = f"{event_type}:{event_signature}"
            if prior_key in self._priors:
                pred_val, conf = self._priors[prior_key]
                return OutcomePrediction(
                    event_signature=event_signature,
                    predicted_outcome="unknown",
                    predicted_valence=(
                        Valence.POSITIVE if pred_val > 0.6 else Valence.NEUTRAL
                    ),
                    predicted_value=pred_val,
                    predicted_delay=0.0,
                    delay_bounds=(0.0, 0.0),
                    confidence=conf,
                    contributing_links=[],
                    context_match=0.0,
                )
            return None

        # Filter to high-confidence links
        valid_links = [
            link
            for link in event_links
            if link.confidence >= self.config.min_confidence_threshold
        ]

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
            context_match=self._context_similarity(best_link.event_context, context),
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
        return [
            link
            for link in self._links.get(event_signature, [])
            if link.outcome_valence == Valence.POSITIVE
        ]

    def get_negative_outcomes(self, event_signature: str) -> list[CausalLink]:
        """Get links where this event led to negative outcomes."""
        return [
            link
            for link in self._links.get(event_signature, [])
            if link.outcome_valence == Valence.NEGATIVE
        ]

    def get_promotion_candidates(
        self,
        min_confidence: float = 0.6,
        min_observations: int = 3,
    ) -> list[PromotionCandidate]:
        """Scan all CausalLinks for promotion-worthy patterns.

        Implements PromotionSource protocol for the SemanticPromoter.
        Returns high-confidence positive causal links as candidates
        for ATL semantic memory promotion.
        """
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
        }

    def __len__(self) -> int:
        """Total number of causal links."""
        return sum(len(links) for links in self._links.values())

    # ─────────────────────────────────────────────────────────────────────────
    # Persistence
    # ─────────────────────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save NAc state to JSON file."""
        data = {
            "version": "1.0",
            "links": {
                event_sig: [link.to_dict() for link in links]
                for event_sig, links in self._links.items()
            },
            "outcome_index": {k: list(v) for k, v in self._outcome_index.items()},
            "priors": self._priors,
            "total_observations": self._total_observations,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        logger.info("Saved NAc to %s (%d links)", path, len(self))

    def load(self, path: str) -> None:
        """Load NAc state from JSON file."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        version = data.get("version", "0.0")
        if version != "1.0":
            raise ValueError(f"Unsupported NAc version: {version}")

        self._links = {
            event_sig: [CausalLink.from_dict(link_data) for link_data in links]
            for event_sig, links in data.get("links", {}).items()
        }
        self._outcome_index = {
            k: set(v) for k, v in data.get("outcome_index", {}).items()
        }
        self._priors = data.get("priors", {})
        self._total_observations = data.get("total_observations", 0)

        logger.info("Loaded NAc from %s (%d links)", path, len(self))

    def get_version(self) -> str:
        """Return data format version."""
        return "1.0"


__all__ = ["NAc", "NACConfig"]