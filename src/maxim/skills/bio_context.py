"""BioContext — skill-friendly facade over MemoryHub bio memory systems.

Wraps MemoryHub's internal components with methods that skills can call
without needing to know the internal wiring. Skills opt in by declaring
``bio_dependencies(["memory_hub"])`` and receive a BioContext instance
in ``self._bio["memory_hub"]``.

All ID-based methods expect concept/memory IDs from ``to_context_dict()``
or ``ConceptContextBuilder.build()`` outputs (added by A7.0a-b).

Brain mapping: This is the skill-facing API for bio memory access —
analogous to how cortical regions access hippocampal and semantic
memory through standardised pathways rather than direct wiring.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from maxim.integration.memory_hub import MemoryHub
    from maxim.math.ips import AnomalyResult, ApproximateResult, TrendResult
    from maxim.memory.hippocampus import Hippocampus
    from maxim.memory.types import (
        CompressedMemory,
        EpisodicMemory,
        PredictedOutcome,
    )

logger = logging.getLogger(__name__)


class BioContext:
    """Bio memory facade injected into skills via _bio["memory_hub"].

    Wraps MemoryHub's internal components with skill-friendly methods.
    Skills that need advanced access can reach ``bio.atl`` or
    ``bio.hippocampus`` directly (both are public MemoryHub fields).
    """

    def __init__(self, hub: MemoryHub) -> None:
        self._hub = hub

    # --- Text-to-concept resolution ---

    def resolve_concepts(
        self,
        detected_objects: list[str] | None = None,
        detected_people: list[str] | None = None,
        active_goal: str | None = None,
        limit: int = 5,
    ) -> list[dict]:
        """Resolve current percept tokens into ATL concepts.

        Delegates to ConceptContextBuilder.build(). Returns list[dict]
        with keys: id, name, category, confidence, episode_count,
        relationships, properties, (optional) layer_context.

        The ``id`` field is what you pass to ground_concept() and other
        ID-based methods.
        """
        builder = self._hub._concept_context_builder
        if builder is None:
            return []
        try:
            return builder.build(
                detected_objects=detected_objects or [],
                detected_people=detected_people or [],
                active_goal=active_goal or "",
                limit=limit,
            )
        except Exception as e:
            logger.warning("BioContext.resolve_concepts failed: %s", e)
            return []

    # --- Episodic recall ---

    def recall_associated(
        self,
        seed_ids: list[str],
        limit: int = 10,
        include_compressed: bool = True,
    ) -> list[tuple[EpisodicMemory | CompressedMemory, float]]:
        """Retrieve related memories via associative graph spreading activation.

        Returns list of (memory, activation_score) tuples sorted
        highest-first. Seed memories are excluded from results.
        """
        return self._hub.hippocampus.recall_associated(
            seed_ids=seed_ids,
            limit=limit,
            include_compressed=include_compressed,
        )

    # --- Numerical grounding ---

    def ground_concept(self, concept_id: str) -> dict[str, Any] | None:
        """Ground a concept with numerical statistics from linked episodes.

        Takes a concept ID (from resolve_concepts()[i]["id"]).
        Gathers linked episodes via concept.memory_refs, passes
        concept + episodes to ConceptGrounder.

        Returns dict of {field_name: {mean, n, min, max, trend, ...}}
        or None if grounder/ATL unavailable.
        """
        grounder = self._hub._concept_grounder
        if grounder is None or self._hub.atl is None:
            return None
        concept = self._hub.atl.get(concept_id)
        if concept is None:
            return None
        ep_ids = list(concept.memory_refs.get("hippocampus", {}).keys())
        episodes = self._hub.hippocampus.recall_by_ids(ep_ids) if ep_ids else []
        return grounder.ground_concept(concept, episodes)

    # --- Pattern completion ---

    def complete_pattern(
        self,
        episodic: EpisodicMemory,
    ) -> list[PredictedOutcome]:
        """Predict likely outcomes for an episode via concept graph chaining.

        Returns list[PredictedOutcome] with fields: tool, success,
        goal, confidence, math_context, source_episode_id.
        """
        completer = self._hub._pattern_completer
        if completer is None:
            return []
        return completer.complete(episodic)

    # --- IPS numerical analysis (stateless) ---

    def detect_trend(self, values: list[float]) -> TrendResult:
        """Sliding-window trend detection.

        Returns TrendResult: direction, slope_estimate, confidence, magnitude.
        """
        from maxim.math.ips import IPS

        return IPS().detect_trend(values)

    def detect_anomaly(self, value: float, history: list[float]) -> AnomalyResult:
        """Robust outlier detection using median and MAD.

        Returns AnomalyResult: is_anomalous, deviation, magnitude, iqr.
        """
        from maxim.math.ips import IPS

        return IPS().detect_anomaly(value, history)

    def estimate_mean(self, values: list[float]) -> ApproximateResult:
        """Fast approximate mean.

        Returns ApproximateResult: value, magnitude, confidence.
        """
        from maxim.math.ips import IPS

        return IPS().estimate_mean(values)

    # --- Direct system access ---

    @property
    def atl(self) -> Any:
        """Direct access to ATL (Anterior Temporal Lobe) concept store."""
        return self._hub.atl

    @property
    def hippocampus(self) -> Hippocampus:
        """Direct access to Hippocampus episodic memory."""
        return self._hub.hippocampus

    @property
    def angular_gyrus(self) -> Any:
        """Direct access to Angular Gyrus numerical quantification."""
        return self._hub.angular_gyrus


__all__ = ["BioContext"]
