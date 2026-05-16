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
from typing import TYPE_CHECKING, Any, ClassVar

from maxim.decisions.causal_link import (
    CausalLink,
    OutcomePrediction,
    TemporalDelta,
    Valence,
)

if TYPE_CHECKING:
    from maxim.memory.semantic_promoter import PromotionCandidate
    from maxim.models.bio_context import PredictionContext

logger = logging.getLogger(__name__)


def _emit_recommend_action_event(
    *,
    agent_id: str,
    current_cluster_id: str | None,
    cluster_reward_bias_consulted: float | None,
    best_tool: str | None,
    best_score: float,
    min_confidence: float,
    passed_gate: bool,
) -> None:
    """Emit a ``sim_recommend_action`` event for Stage 0c telemetry.

    Per release_0_9_1.md Stage 0c, every ``recommend_action`` call MUST
    emit exactly one event — even the early-return paths (empty
    available_tools, empty scores, sub-threshold) — so Roy-3 measurement
    can distinguish "gate fired but consumer did nothing" from
    "consumer ran and proposed nothing."

    The event lands on the ``sim_log("NAc_RECOMMEND", ...)`` channel,
    which routes through the standard sim_log JSONL writer + the
    MAXIM_LOG_FILE bridge.

    **Tick alignment with Stage 0d (CRITICAL):** the ``tick`` field
    matches Stage 0d's ``sim_ec_activation`` tick space —
    ``int(time.time() - sim_logger._sim_start)``, NOT raw epoch seconds.
    Without this alignment Roy-3 cannot left-join the two channels
    on tick (a 1e9 offset returns zero matches every time). For
    sub-second ordering use the sim_log JSONL's top-level ``t`` field,
    which sim_log auto-attaches with millisecond resolution from the
    same ``_sim_start`` reference.

    The emission is fail-soft: ``ImportError`` (non-sim runtime where
    sim_logger isn't importable at all) is swallowed silently. Any
    other exception propagates — a real sim_logger bug should surface
    rather than masquerade as silent annotation-off.
    """
    try:
        from maxim.simulation import sim_logger as _sl

        tick = int(time.time() - _sl._sim_start) if _sl._sim_start > 0.0 else 0
        _sl.sim_log(
            "NAc_RECOMMEND",
            f"recommend_action: passed_gate={passed_gate}",
            {
                "tick": tick,
                "current_cluster_id": current_cluster_id,
                "cluster_reward_bias_consulted": cluster_reward_bias_consulted,
                "best_tool": best_tool,
                "best_score": round(best_score, 4),
                "min_confidence": min_confidence,
                "passed_gate": passed_gate,
            },
            agent_id=agent_id,
        )
    except ImportError:
        # Non-sim runtime: sim_logger isn't importable at all (e.g.,
        # headless API without the simulation extras). Stage 0c is
        # observability only, not load-bearing for correctness —
        # swallow silently. Any OTHER exception (a real sim_logger
        # bug, an attribute error from a broken refactor) propagates
        # so we don't silently disable telemetry the Roy-3 measurement
        # arm depends on.
        pass


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

    # Track 2 of grounded_language_acquisition.md Phase 0+: cluster-keyed
    # tool reward bias for substrate-primary action selection. Range
    # [-max_cluster_reward_bias, +max_cluster_reward_bias]. Larger than
    # max_reward_bias because cluster bias is a primary action-selection
    # score (it competes with causal_pos and the cold-start drive-affinity
    # heuristic, both of which contribute up to ~1.0), not a recognition
    # modulator like _reward_bias.
    max_cluster_reward_bias: float = 1.0

    # Temporal credit weight for SCN-coupled eligibility (affordance transfer).
    # When fast-decay traces expire, nodes with temporal anchors still receive
    # credit at this fraction of the temporal similarity score.
    # Override: MAXIM_NAC_TEMPORAL_CREDIT_WEIGHT (clamped 0.05-1.0)
    temporal_credit_weight: float = 0.3


# Phase -1 prototype — drive→tool affinity table for cold-start action
# proposal. Stand-in for proper EC embedding similarity (Phase 0+ replaces).
# See NAc.recommend_action(). Keys are lowercase drive names; values are
# substrings to match in tool names (also lowercase).
_DRIVE_TOOL_AFFINITIES: dict[str, tuple[str, ...]] = {
    "hunger": ("eat", "pick_up", "food", "consume", "feed"),
    "thirst": ("drink", "water", "consume"),
    "fatigue": ("rest", "sleep", "sit", "lie"),
    "stamina": ("rest", "sleep", "sit", "lie"),
    "cold": ("warm", "fire", "blanket", "huddle"),
    "thermal": ("warm", "fire", "blanket", "huddle"),
    "fear": ("flee", "hide", "retreat", "escape"),
    "curiosity": ("examine", "look", "sense", "inspect"),
    "pain": ("rest", "heal", "tend", "withdraw"),
}


class NAc:
    """Nucleus Accumbens - Causal inference and reward prediction engine.

    Learns event → outcome relationships through observation, enabling
    prediction of outcomes before taking actions.

    Integration points:
    - SCN: Temporal eligibility credit via ``_temporal_anchors``.
      First closed loop (2026-04-24): affordance concept nodes get
      ``TemporalSignature`` anchors at encoding time; ``distribute_reward``
      uses ``TemporalSignature.similarity()`` to credit nodes whose
      fast-decay traces expired but whose temporal phase matches the
      reward. Broader temporal pattern learning (tool reliability by
      time-of-day, oscillator-driven prediction) remains unbuilt.
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

    # P3.5 Stage 1 — BioSystemSnapshot Protocol envelope version.
    # Payload-layer legacy version string "1.0" is tombstoned; all future
    # migrations land at the envelope layer. See memory/snapshot.py docstring.
    schema_version: ClassVar[int] = 1

    def __init__(self, config: NACConfig | None = None, ec: Any = None):
        config = config or NACConfig()
        # Apply env-var override for temporal_credit_weight if set
        tcw_env = os.environ.get("MAXIM_NAC_TEMPORAL_CREDIT_WEIGHT")
        if tcw_env is not None:
            try:
                from dataclasses import replace

                tcw = max(0.05, min(1.0, float(tcw_env)))
                config = replace(config, temporal_credit_weight=tcw)
            except (ValueError, TypeError):
                pass
        self.config = config

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

        # SCN temporal anchors for eligibility credit (affordance transfer).
        # When fast-decay traces expire, temporal anchors let distribute_reward
        # credit nodes that were activated in the same temporal phase as the
        # reward, via TemporalSignature.similarity(). Session-scoped — NOT
        # persisted (wall-clock timestamps go stale across sessions).
        self._temporal_anchors: dict[tuple[str, str], tuple[float, Any]] = {}
        # Maps (agent_id, node_id) → (original_activation, TemporalSignature)

        # Track 2 of grounded_language_acquisition.md Phase 0+: cluster-keyed
        # reward bias for substrate-primary action selection. Keyed by
        # (agent_id, cluster_id, tool_signature) → reward in
        # [-max_cluster_reward_bias, +max_cluster_reward_bias]. Bidirectional
        # like _goal_reward_bias (positive = "this tool worked here", negative
        # = "this tool failed here"). Wired by SensorEncoder via
        # propose_via_substrate, which captures the EC interoception cluster
        # id once per tick and passes it to recommend_action.
        self._cluster_reward_bias: dict[tuple[str, str, str], float] = {}

        # Goal-level reward bias: tracks whether deliberation under a goal
        # type historically produces good outcomes.  Keyed by goal string.
        # Range: [-max_reward_bias, +max_reward_bias] — UNLIKE _reward_bias
        # which clamps to [0, max].  Positive = direct pathway "go" (lower
        # ThoughtGate threshold), negative = indirect pathway "no-go" (raise
        # threshold, skip deliberation).  Persisted across sessions.
        self._goal_reward_bias: dict[str, float] = {}

        # Reserved for 1.1 (bio_emergent_persona_foundations Wire 2): per-agent
        # percept-level valence keyed by (agent_id, entity_class, failure_mode)
        # → float in [-1.0, +1.0].  Lives on NAc because the *learning source*
        # is NAc-adjacent: the PainBus subscriber feeding it would mirror
        # `create_pain_nac_subscriber`, and the per-tick decay would extend the
        # existing `decay_reward_biases` / `decay_goal_reward_biases` cycle
        # in agent_loop.py section 8.5.  (The cross-bio-system valence scalar
        # — Episode.valence, ValenceSignal — justifies the *type*, not the
        # *placement*; placement is justified by learning-source proximity.)
        # Flat-tuple keying matches `_reward_bias`'s shape so persistence
        # (1.1: ``f"{aid}:{ec}:{fm}"`` join, mirroring `_reward_bias`'s
        # ``f"{aid}:{nid}"``) reuses the same dump/load idiom.  1.0 reserves
        # the attribute name + placement + shape only; no methods, no
        # persistence, no read sites.  Consumer-side concerns (per-agent
        # scoping on the GatingContext path, JSON-key encoding for keys
        # containing ':') are deferred to 1.1 design with Crucible data.
        # See docs/plans/bio_emergent_persona_foundations.md and
        # docs/experiments/12_v1_phased_attribution.md (Phase A clean pass —
        # substrate sufficient for V1 cross-session recall; persona-divergence
        # wires deferred pending Crucible findings).
        self._percept_valences: dict[tuple[str, str, str], float] = {}

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
        """How well does ctx1 appear inside ctx2? (0.0-1.0)

        Directional: the denominator is ``len(ctx1)``, not the union of
        keys. Semantics: "how much of the pending event's context is
        matched by the outcome context (or: how much of a stored link's
        event context is matched by the current query context)?"

        This convention is load-bearing: every caller in this file
        passes the pending-event / stored-link side as ``ctx1`` and the
        outcome / current-query side as ``ctx2``. The outcome /
        query side is allowed to carry EXTRA keys that ctx1 doesn't
        have — those extras DO NOT dilute the similarity score. Keys
        present in ctx1 but absent from ctx2 count as misses because
        they're the things the event was conditioned on and the
        outcome couldn't confirm.

        Pre-Stage-2 this function used ``len(keys_union)`` as the
        denominator, which meant any caller passing a richer context
        on the outcome side (e.g., ``ToolPainBridge._on_embodiment_pain``
        passing 7 cause-description keys) would dilute legitimate
        2-key pending-event matches below the
        ``context_similarity_threshold`` (0.5 default). This was a
        silent learning failure: pending events never linked to
        pain outcomes, causal-link buffers filled with unlinked
        events, and nac.predict() returned None for actions that
        had clearly produced pain in the past. The pre-merge review
        round for Substrate P2 Stage 2 caught this.

        Callers must pass arguments in (event_or_stored, outcome_or_query)
        order. See ``_record_outcome_impl`` line ~322 and ``predict``
        line ~581 for the canonical call sites.
        """
        if not ctx1 or not ctx2:
            return 0.5  # Neutral if no context

        matches = 0.0
        for key in ctx1:
            v1 = ctx1[key]
            v2 = ctx2.get(key)
            if v1 == v2:
                matches += 1
            elif v1 is not None and v2 is not None:
                if isinstance(v1, str) and isinstance(v2, str):
                    if v1.lower() == v2.lower():
                        matches += 0.8

        return matches / len(ctx1)

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
        prediction_context: PredictionContext | None = None,
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
        prediction_context: "PredictionContext | None" = None,
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
            updated_links = self._record_outcome_impl(
                outcome_type,
                outcome_signature,
                outcome_valence,
                context,
                memory_id,
                attributed_event_id,
                attributed_event_signature,
            )

        # Surface NOVEL causal-link formations as LEARN headlines OUTSIDE
        # the lock — first-time observation of an (event_signature →
        # outcome_signature) pair is a "the agent just learned something
        # new" moment that belongs in the headline channel.  Subsequent
        # observations of the same pair stay at BIO tier via the existing
        # ``sim_nac_learn`` emission below.
        novel = [link for link in updated_links if link.observation_count == 1]
        if novel:
            try:
                from maxim.simulation.sim_logger import sim_learn

                for link in novel:
                    valence_str = getattr(link.outcome_valence, "value", str(link.outcome_valence))
                    sim_learn(
                        f"new causal link: {link.event_signature[:24]} → {link.outcome_signature[:24]}",
                        detail=f"valence={valence_str}, RPE={link.last_rpe:+.2f}",
                        source="NAc",
                        event=link.event_signature,
                        outcome=link.outcome_signature,
                        valence=valence_str,
                    )
            except Exception:
                pass

        return updated_links

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

        # Simulation verbosity — use structured sim_nac_learn for display + JSONL
        for link in updated_links:
            try:
                from maxim.simulation.sim_logger import sim_nac_learn

                sim_nac_learn(
                    event=link.event_signature,
                    outcome=link.outcome_signature,
                    confidence=link.confidence,
                    rpe=link.last_rpe,
                    link_type="updated" if link.observation_count > 1 else "new",
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
                try:
                    from maxim.simulation.sim_logger import sim_nac_learn

                    sim_nac_learn(
                        event=event_signature,
                        outcome=outcome_signature,
                        confidence=existing_link.confidence,
                        rpe=existing_link.last_rpe,
                        link_type="observed",
                    )
                except Exception:
                    pass
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
            try:
                from maxim.simulation.sim_logger import sim_nac_learn

                sim_nac_learn(
                    event=event_signature,
                    outcome=outcome_signature,
                    confidence=new_link.confidence,
                    rpe=new_link.last_rpe,
                    link_type="new_observed",
                )
            except Exception:
                pass
            # Capture new-link payload for the OUTSIDE-the-lock LEARN
            # headline emission below.  ``observe`` is the direct-observation
            # API (caller already attributed event → outcome), so any new
            # link here is a first-time observation that warrants a
            # CLEAN-tier headline.  ``outcome_valence`` is the ``Valence``
            # enum — capture its ``.value`` string for display.
            _new_link_summary: tuple[str, str, str, float] | None = (
                new_link.event_signature,
                new_link.outcome_signature,
                getattr(new_link.outcome_valence, "value", str(new_link.outcome_valence)),
                new_link.last_rpe,
            )

        # Outside the lock — surface novel causal links as LEARN headlines.
        if _new_link_summary is not None:
            try:
                from maxim.simulation.sim_logger import sim_learn

                event_sig, outcome_sig, valence_str, rpe = _new_link_summary
                sim_learn(
                    f"new causal link: {event_sig[:24]} → {outcome_sig[:24]}",
                    detail=f"valence={valence_str}, RPE={rpe:+.2f}",
                    source="NAc",
                    event=event_sig,
                    outcome=outcome_sig,
                    valence=valence_str,
                )
            except Exception:
                pass
        return new_link

    # ─────────────────────────────────────────────────────────────────────────
    # PREDICTION
    # ─────────────────────────────────────────────────────────────────────────

    def predict(
        self,
        event_type: str,
        event_signature: str,
        context: dict[str, Any] | None = None,
        prediction_context: "PredictionContext | None" = None,
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
            result = self._predict_impl(event_type, event_signature, context)
        # Log prediction outside lock to avoid import-under-lock
        try:
            from maxim.simulation.sim_logger import sim_nac_predict

            if result is not None:
                outcomes = [(result.outcome_signature, result.confidence)]
            else:
                outcomes = []
            sim_nac_predict(event=event_signature, outcomes=outcomes)
        except Exception:
            pass
        return result

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
        prediction_context: "PredictionContext | None" = None,
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

    def get_links_for_event(
        self,
        event_signature: str,
        prediction_context: "PredictionContext | None" = None,
    ) -> list[CausalLink]:
        """Get all causal links originating from an event type."""
        return self._links.get(event_signature, [])

    def scan_links_for_keywords(
        self,
        keywords: list[str],
        *,
        min_keyword_length: int = 3,
        min_confidence: float = 0.3,
        max_matches: int = 10,
    ) -> list[CausalLink]:
        """Find causal links whose event_signature contains any keyword.

        Companion to ``get_links_for_event`` for narrative-keyword queries
        where the caller doesn't know the exact stored signature. Tool
        events are stored under canonical signatures like
        ``tool:rusty_sword_slash``; this method lets a query keyword
        ``rusty`` or ``slash`` find those links via case-insensitive
        substring containment.

        Returns links sorted by confidence descending, deduplicated by
        link id, capped at ``max_matches``. Short keywords (below
        ``min_keyword_length``) are dropped to avoid matching everything
        — narrative stop-words like "to"/"of" would otherwise hit nearly
        every signature.
        """
        if not keywords:
            return []
        kws_lower = [kw.lower() for kw in keywords if kw and len(kw) >= min_keyword_length]
        if not kws_lower:
            return []
        matched: list[CausalLink] = []
        seen_ids: set[str] = set()
        with self._lock:
            for sig, links in self._links.items():
                sig_lower = sig.lower()
                if not any(kw in sig_lower for kw in kws_lower):
                    continue
                for link in links:
                    if link.confidence < min_confidence:
                        continue
                    if link.id in seen_ids:
                        continue
                    seen_ids.add(link.id)
                    matched.append(link)
        matched.sort(key=lambda lk: lk.confidence, reverse=True)
        return matched[:max_matches]

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

    def tag_imagined_links(self, entity_refs: frozenset[str]) -> int:
        """Retroactively tag causal links involving imagined entities.

        Called at session end with the set of imagined entity refs.
        Matches links whose ``event_signature`` contains any of the
        entity ref basenames (e.g., ``"crystal_dragon"`` matches
        ``"crystal_dragon_bite"``).

        Returns the number of links tagged.
        """
        if not entity_refs:
            return 0

        # Extract basenames from refs: "imagined/crystal_dragon" → "crystal_dragon"
        basenames = {ref.rsplit("/", 1)[-1] for ref in entity_refs}

        count = 0
        with self._lock:
            for links in self._links.values():
                for link in links:
                    if link.imagined:
                        continue  # Already tagged
                    sig = link.event_signature
                    if any(base in sig for base in basenames):
                        link.imagined = True
                        count += 1
        if count:
            logger.info("NAc: tagged %d links as imagined (from %d entity refs)", count, len(entity_refs))
        return count

    def decay_imagined_links(self, factor: float = 0.5) -> int:
        """Decay confidence of all links with ``imagined=True`` provenance.

        Called at session end when ephemeral (imagined) entities are
        discarded. Reduces confidence by *factor* (default 50%) rather
        than deleting — partial learning from imagined experiences is
        useful at reduced confidence.

        Returns the number of links decayed.
        """
        count = 0
        with self._lock:
            for links in self._links.values():
                for link in links:
                    if link.imagined:
                        link.decay(factor)
                        count += 1
        if count:
            logger.debug("NAc: decayed %d imagined links by factor %.2f", count, factor)
        return count

    # ─────────────────────────────────────────────────────────────────────────
    # Phase -1 prototype — substrate-driven action proposal
    # ─────────────────────────────────────────────────────────────────────────
    #
    # See docs/plans/grounded_language_acquisition.md Phase -1 for the gating
    # Boolean experiment this method enables: can the substrate propose even
    # one non-reflex action without LLM mediation? The answer determines
    # whether substrate-primary AUT mode is feasible at all.
    #
    # This is a PROTOTYPE. The drive-relevance heuristic is a temporary
    # stand-in for proper EC embedding similarity (Phase 0+ will replace it).
    # The method exists to answer the Boolean, not to ship as final API.

    def recommend_action(
        self,
        *,
        agent_id: str,
        available_tools: "list[str] | set[str] | tuple[str, ...]",
        current_drives: dict[str, float] | None = None,
        current_cluster_id: str | None = None,
        min_confidence: float = 0.3,
    ) -> dict[str, Any] | None:
        """Substrate-driven action proposal — Phase -1 prototype.

        Returns an action dict (compatible with ``Proposal.action``) or
        ``None`` if no tool meets the confidence threshold. Random
        selection is explicitly NOT a fallback — substrate must have an
        opinion (learned bias OR coherent drive signal) to act.

        Algorithm:
          1. For each available tool, score by learned ``reward_bias``
             (positive value means the agent has been rewarded for this
             tool; zero for new agents).
          2. Add ``cluster_reward_bias`` when ``current_cluster_id`` is
             set — the per-(agent, cluster, tool) signal accumulated by
             :meth:`update_cluster_reward`. Replaces the drive-affinity
             substring heuristic as the substrate's primary action-
             selection signal once any history exists for the active
             interoception cluster.
          3. Augment with drive-relevance heuristic: active drives
             (value > 0.5) bias scoring toward semantically-related
             tools via name substring match + a small affinity table.
             This is the Phase -1 cold-start fallback — when no cluster
             history exists for any tool, drive affinity carries the
             selection.
          4. Return the highest-scoring tool above ``min_confidence``,
             else ``None``.

        Args:
            agent_id: Per-agent scoping for ``reward_bias`` lookup.
                Empty string is rejected (per the multi-agent isolation
                invariant in CLAUDE.md "Per-agent stash dicts" lesson).
            available_tools: Tool names the substrate may propose.
                Order is irrelevant; ties resolved by name sort.
            current_drives: Optional ``{drive_name: value in [0, 1]}``.
                Drives with value > 0.5 contribute drive-relevance
                scoring. ``None`` skips the cold-start heuristic.
            current_cluster_id: Optional EC node id for the active
                interoception cluster (from
                ``SensorEncoder.encode_sensors``). When set, tools with
                positive ``cluster_reward_bias`` in this cluster
                outscore drive affinity. ``None`` skips cluster-keyed
                scoring (only path before Track 2 wired this in).
            min_confidence: Threshold below which we return ``None``
                instead of proposing a low-confidence action. Default
                ``0.3`` matches ``NACConfig.min_confidence_threshold``.

        Returns:
            ``{"tool_name", "params", "confidence", "source", "reasoning"}``
            on success, or ``None`` when nothing scores high enough.
            ``source`` is always ``"substrate-primary"`` so the executor
            can distinguish substrate-proposed actions from LLM-proposed.
        """
        if not agent_id:
            raise ValueError("recommend_action requires non-empty agent_id")
        if not available_tools:
            # Stage 0c: empty available_tools is a legitimate early return
            # (e.g., the scene_actor filter trimmed the executor's tool set
            # to nothing). Still emit so Roy-3 can distinguish "no tools
            # available" from "no tools scored above gate."
            _emit_recommend_action_event(
                agent_id=agent_id,
                current_cluster_id=current_cluster_id,
                cluster_reward_bias_consulted=None,
                best_tool=None,
                best_score=0.0,
                min_confidence=min_confidence,
                passed_gate=False,
            )
            return None

        drives = current_drives or {}
        tool_list = sorted(available_tools)

        scores: dict[str, float] = {}
        reasoning_parts: dict[str, list[str]] = {}

        for tool_name in tool_list:
            score = 0.0
            parts: list[str] = []
            event_sig = f"tool:{tool_name}"

            # Component 1 (primary learned signal): causal-link confidence.
            # Positive outcomes contribute their best confidence; negative
            # outcomes subtract (weighted lower so a single bad outcome
            # doesn't permanently block exploration).
            pos_links = self.get_positive_outcomes(event_sig)
            if pos_links:
                best_pos = max(link.confidence for link in pos_links)
                score += best_pos
                parts.append(f"causal_pos={best_pos:.2f}")
            neg_links = self.get_negative_outcomes(event_sig)
            if neg_links:
                best_neg = max(link.confidence for link in neg_links)
                score -= best_neg * 0.5
                parts.append(f"causal_neg={best_neg:.2f}")

            # Component 2 (secondary learned signal): reward bias. Capped at
            # max_reward_bias (default 0.20) by design — it's a recognition
            # modulator, not a primary action-selection score. Adds a small
            # positive nudge when present.
            bias = self.reward_bias(agent_id, event_sig)
            if bias > 0:
                score += bias
                parts.append(f"reward_bias={bias:.2f}")

            # Component 2b (Track 2 of grounded_language_acquisition.md):
            # cluster-keyed reward bias. Positive value = this tool worked
            # in this interoception cluster; negative = it failed here.
            # Added directly to score (range [-1, +1]) so it competes
            # with causal_pos and dominates cold-start drive affinity
            # once any history exists for the active cluster.
            if current_cluster_id:
                cluster_bias = self.cluster_reward_bias(agent_id, current_cluster_id, event_sig)
                if cluster_bias != 0.0:
                    score += cluster_bias
                    parts.append(f"cluster_bias={cluster_bias:+.2f}")

            # Component 3: drive-relevance (cold-start heuristic)
            tool_lower = tool_name.lower()
            for drive_name, drive_value in drives.items():
                if drive_value <= 0.5:
                    continue
                drive_lower = drive_name.lower()

                # Direct name substring match
                if drive_lower in tool_lower:
                    score += drive_value
                    parts.append(f"drive:{drive_name}({drive_value:.2f}) name-match")
                    continue

                # Affinity table — semantic stand-in until EC integration
                affinities = _DRIVE_TOOL_AFFINITIES.get(drive_lower, ())
                for keyword in affinities:
                    if keyword in tool_lower:
                        score += drive_value * 0.7
                        parts.append(f"drive:{drive_name}({drive_value:.2f}) →{keyword}")
                        break

            if score > 0:
                scores[tool_name] = score
                reasoning_parts[tool_name] = parts

        # Stage 0c (release_0_9_1.md): emit `sim_recommend_action` for
        # post-hoc Roy-3 measurement. Every recommend_action call emits
        # exactly one event — even on the early-return paths (no scores,
        # sub-threshold) — so Roy iterations can distinguish "gate fired
        # but consumer didn't run" from "consumer ran and proposed
        # nothing." Per the plan: "the event MUST emit even when
        # recommend_action returns None."
        if not scores:
            # Bio-fidelity review fold: distinguish "cluster known, no
            # tool scored" (0.0 sentinel — agent had context but nothing
            # rewarded) from "cluster unknown" (None — no
            # current_cluster_id at all). Roy-3 needs this distinction
            # to expose the Wire-A vs recommend_action gap; collapsing
            # both into None would elide the H1 signal.
            _consulted_on_empty: float | None = 0.0 if current_cluster_id else None
            _emit_recommend_action_event(
                agent_id=agent_id,
                current_cluster_id=current_cluster_id,
                cluster_reward_bias_consulted=_consulted_on_empty,
                best_tool=None,
                best_score=0.0,
                min_confidence=min_confidence,
                passed_gate=False,
            )
            return None

        best_tool = max(scores, key=lambda t: (scores[t], t))
        best_score = scores[best_tool]

        # Record the cluster_reward_bias consulted for the best tool —
        # informative for Roy-3 because Wire-A renders aggregate biases
        # across all clusters, but recommend_action only consults the
        # active-cluster value. Mismatch between rendered Wire-A signal
        # and consulted recommend_action signal is the failure mode the
        # H1 sub-hypothesis branches (cross_modal_substrate_binding.md /
        # jepa_cross_modal_alignment.md) eventually address.
        consulted_bias: float | None = None
        if current_cluster_id:
            consulted_bias = self.cluster_reward_bias(agent_id, current_cluster_id, f"tool:{best_tool}")

        if best_score < min_confidence:
            _emit_recommend_action_event(
                agent_id=agent_id,
                current_cluster_id=current_cluster_id,
                cluster_reward_bias_consulted=consulted_bias,
                best_tool=best_tool,
                best_score=best_score,
                min_confidence=min_confidence,
                passed_gate=False,
            )
            return None

        _emit_recommend_action_event(
            agent_id=agent_id,
            current_cluster_id=current_cluster_id,
            cluster_reward_bias_consulted=consulted_bias,
            best_tool=best_tool,
            best_score=best_score,
            min_confidence=min_confidence,
            passed_gate=True,
        )
        return {
            "tool_name": best_tool,
            "params": {},
            "confidence": min(best_score, 1.0),
            "source": "substrate-primary",
            "reasoning": "; ".join(reasoning_parts[best_tool]),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # P2: Reward Bias — per-node recognition modulation
    # ─────────────────────────────────────────────────────────────────────────

    def reward_bias(self, agent_id: str, node_id: str) -> float:
        """Get the current reward bias for a substrate node.

        Returns a value in [0, max_reward_bias]. Positive means the node
        has been rewarded and EC should lower its threshold for this node.
        """
        return self._reward_bias.get((agent_id, node_id), 0.0)

    def get_temporal_anchors(self, agent_id: str) -> dict[str, tuple[float, Any]]:
        """Return temporal anchors for an agent, keyed by node_id.

        The distributor reads these to provide phase-similarity credit
        after fast-decay traces expire.  NAc owns the anchors — the
        distributor does NOT mutate them.

        Returns:
            ``{node_id: (original_activation, TemporalSignature)}``
        """
        with self._lock:
            return {nid: anchor for (aid, nid), anchor in self._temporal_anchors.items() if aid == agent_id}

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
            new_bias = self._reward_bias[key]
            logger.debug(
                "NAc credit_node(%s, %s): %.4f → %.4f (reward=%.2f)",
                agent_id[:8] if agent_id else "?",
                node_id[:8],
                current,
                new_bias,
                reward,
            )
        # NOTE: per-node LEARN headlines are emitted by the *batch*
        # caller (NAc.distribute_reward / TemporalCreditDistributor.
        # distribute), aggregating across nodes instead of one line
        # per credit_node call.  A 20-node distribution would otherwise
        # produce 20 LEARN lines from a single reward arrival, flooding
        # the sparse-headline channel — pre-merge review caught this.

    # -- Cluster-keyed tool reward bias (Track 2, Phase 0+) ---------------

    def update_cluster_reward(
        self,
        agent_id: str,
        cluster_id: str | None,
        tool_signature: str,
        reward: float,
    ) -> None:
        """Update cluster-keyed reward bias for a tool.

        Used by substrate-primary action selection (Track 2 of
        grounded_language_acquisition.md Phase 0+) to record which
        tools work in which EC interoception clusters. The cluster id
        comes from SensorEncoder.encode_sensors().

        Args:
            agent_id: Per-agent scoping (CLAUDE.md per-agent stash
                invariant). Empty string is rejected.
            cluster_id: EC node id for the active interoception cluster.
                ``None`` or empty string is a no-op — without cluster
                context the substrate cannot key the learning.
            tool_signature: From ``build_tool_signature(tool_name, params)``.
                Same shape as the event_signature used by reward_bias.
            reward: Positive reinforces, negative punishes. Accumulated
                via the same ``reward_bias_alpha`` as ``_reward_bias`` and
                clamped to ``[-max_cluster_reward_bias, +max_cluster_reward_bias]``.
        """
        if not agent_id:
            raise ValueError("update_cluster_reward requires non-empty agent_id")
        if not cluster_id:
            return
        with self._lock:
            key = (agent_id, cluster_id, tool_signature)
            current = self._cluster_reward_bias.get(key, 0.0)
            updated = current + self.config.reward_bias_alpha * reward
            cap = self.config.max_cluster_reward_bias
            self._cluster_reward_bias[key] = max(-cap, min(updated, cap))

    def cluster_reward_bias(
        self,
        agent_id: str,
        cluster_id: str | None,
        tool_signature: str,
    ) -> float:
        """Read cluster-keyed reward bias.

        Returns 0.0 when ``cluster_id`` is missing/empty (no cluster
        context to key against) or when no learning exists for the
        ``(agent_id, cluster_id, tool_signature)`` triple.
        """
        if not cluster_id:
            return 0.0
        return self._cluster_reward_bias.get((agent_id, cluster_id, tool_signature), 0.0)

    def get_agent_tool_biases(
        self,
        *,
        agent_id: str,
        top_n: int = 5,
    ) -> list[tuple[str, float]]:
        """Aggregate per-tool reward bias across all clusters for one agent.

        Used by Wire-A (release_0_9_1.md Stage 2) to surface substrate-
        acquired tool-level reward signal to the LLM prompt. The
        aggregation is **agent-wide** (no active-cluster filter) because
        Roy-2c confirmed the encoder-alignment gap makes priming clusters
        structurally disjoint from test-fixture clusters — restricting
        rendering to active-cluster intersection reproduces the exact
        bug Wire-A exists to fix.

        For each unique ``tool_signature`` under ``agent_id``, takes the
        bias whose absolute value is largest across all clusters. This
        treats a strong negative (avoidance) and a strong positive
        (attraction) as equally diagnostic of substrate-acquired signal,
        and surfaces whichever is stronger. The sign is preserved in the
        returned bias so the caller can render aversion vs reward
        distinctly.

        Args:
            agent_id: Per-agent scoping (CLAUDE.md per-agent stash
                invariant). Empty string is rejected.
            top_n: Maximum number of (tool, bias) pairs to return.
                Sorted by ``abs(bias)`` descending; ties broken by
                ``tool_signature`` ascending for stable output.

        Returns:
            List of ``(tool_signature, bias)`` tuples. Empty list when
            ``agent_id`` has no entries in ``_cluster_reward_bias`` (a
            cold-start agent OR an agent that has never run a
            substrate-primary tick).
        """
        if not agent_id:
            raise ValueError("get_agent_tool_biases requires non-empty agent_id")
        # Aggregate per tool_signature: keep the bias with the largest
        # |bias| seen across all (agent_id, cluster_id, tool_signature)
        # entries matching agent_id.
        per_tool: dict[str, float] = {}
        with self._lock:
            for (aid, _cid, tool_sig), bias in self._cluster_reward_bias.items():
                if aid != agent_id:
                    continue
                existing = per_tool.get(tool_sig)
                if existing is None or abs(bias) > abs(existing):
                    per_tool[tool_sig] = bias
        if not per_tool:
            return []
        # Sort by |bias| desc, tool_signature asc (stable tiebreaker).
        items = sorted(per_tool.items(), key=lambda kv: (-abs(kv[1]), kv[0]))
        return items[: max(0, top_n)]

    # -- Goal-level reward bias (bidirectional, for ThoughtGate) ----------

    def credit_goal(self, goal_tag: str | None, reward: float) -> None:
        """Update goal-level reward bias.

        Unlike credit_node (unidirectional [0, max]), goal bias allows
        negative values for indirect pathway "no-go" suppression.

        Args:
            goal_tag: Active goal string.  None is a no-op (guard against
                phantom None key in _goal_reward_bias).
            reward: Positive = deliberation helped, negative = wasted time.
        """
        if goal_tag is None:
            return
        with self._lock:
            current = self._goal_reward_bias.get(goal_tag, 0.0)
            updated = current + self.config.reward_bias_alpha * reward
            cap = self.config.max_reward_bias
            self._goal_reward_bias[goal_tag] = max(-cap, min(updated, cap))

    def get_goal_reward_bias(self, goal_tag: str | None) -> float:
        """Return goal-level reward bias for ThoughtGate modulation.

        Returns 0.0 for None goal (no modulation when goalless).
        Positive = lower threshold (deliberate more).
        Negative = raise threshold (skip deliberation).
        """
        if goal_tag is None:
            return 0.0
        return self._goal_reward_bias.get(goal_tag, 0.0)

    def decay_goal_reward_biases(self) -> int:
        """Decay goal-level biases toward zero.  Called alongside decay_reward_biases().

        Uses same decay tau as node biases.  Returns count pruned.
        """
        if not self._goal_reward_bias:
            return 0
        decay_factor = 1.0 / self.config.reward_bias_decay_tau
        pruned = 0
        with self._lock:
            to_remove = []
            for goal, bias in self._goal_reward_bias.items():
                new_bias = bias * (1.0 - decay_factor)
                if abs(new_bias) < 0.001:
                    to_remove.append(goal)
                    pruned += 1
                else:
                    self._goal_reward_bias[goal] = new_bias
            for goal in to_remove:
                del self._goal_reward_bias[goal]
        return pruned

    def decay_cluster_reward_biases(self) -> int:
        """Decay cluster-keyed reward biases toward zero.

        Called per-tick alongside ``decay_reward_biases()`` and
        ``decay_goal_reward_biases()``. Without per-tick decay the
        cluster-bias map accumulates indefinitely; Wire-A
        (release_0_9_1.md Stage 2) reads this map at every LLM
        submission and renders it as the substrate's "felt familiarity"
        annotation, so stale-forever biases would silently lie about
        being "from prior experience" when they're actually "from
        forever ago."

        Uses same decay tau as node and goal biases (bidirectional —
        absolute-value prune below 0.001, mirroring
        ``decay_goal_reward_biases``). Returns count pruned.
        """
        if not self._cluster_reward_bias:
            return 0
        decay_factor = 1.0 / self.config.reward_bias_decay_tau
        pruned = 0
        with self._lock:
            to_remove = []
            for key, bias in self._cluster_reward_bias.items():
                new_bias = bias * (1.0 - decay_factor)
                if abs(new_bias) < 0.001:
                    to_remove.append(key)
                    pruned += 1
                else:
                    self._cluster_reward_bias[key] = new_bias
            for key in to_remove:
                del self._cluster_reward_bias[key]
        return pruned

    def update_eligibility(
        self,
        agent_id: str,
        node_id: str,
        activation: float,
        temporal_sig: Any = None,
    ) -> None:
        """Update the eligibility trace for a node.

        Called when a percept activates (completes to) a node. The
        activation strength determines how much credit the node receives
        when a reward arrives later.

        When ``temporal_sig`` (a ``TemporalSignature``) is provided, a
        temporal anchor is also stored. If the fast-decay trace expires
        before a reward arrives, ``distribute_reward`` can still credit
        the node via SCN temporal similarity — at reduced weight
        (``NACConfig.temporal_credit_weight``).

        Args:
            agent_id: Agent context.
            node_id: ATL node that was activated.
            activation: Activation strength (typically from PerceptTraceBuffer).
            temporal_sig: Optional TemporalSignature for SCN-coupled credit.
        """
        with self._lock:
            self._eligibility[(agent_id, node_id)] = activation
            if temporal_sig is not None:
                self._temporal_anchors[(agent_id, node_id)] = (activation, temporal_sig)

    def distribute_reward(
        self,
        agent_id: str,
        reward: float,
    ) -> list[tuple[str, float]]:
        """Distribute reward to all eligible nodes for an agent.

        Two-path credit assignment:

        1. **Fast-decay path** (existing): nodes with active eligibility
           traces (> 0.01) receive credit proportional to trace strength.
        2. **Temporal fallback** (SCN coupling): nodes whose fast-decay
           trace has expired but that have a temporal anchor receive credit
           proportional to ``TemporalSignature.similarity(anchor, now)``
           scaled by ``NACConfig.temporal_credit_weight`` (default 0.3x).

        The temporal path only fires for nodes NOT already credited by
        the fast-decay path, preventing double-counting.

        Returns list of (node_id, credit_applied) for logging.
        """
        credited: list[tuple[str, float]] = []
        with self._lock:
            # 1. Fast-decay path
            eligible = {
                (aid, nid): strength
                for (aid, nid), strength in self._eligibility.items()
                if aid == agent_id and strength > 0.01
            }

            # 2. Temporal fallback — nodes with expired traces but valid anchors
            temporal_eligible: dict[tuple[str, str], float] = {}
            if self._temporal_anchors:
                try:
                    from maxim.time.temporal_signature import TemporalSignature

                    now_sig = TemporalSignature.now()
                    tcw = self.config.temporal_credit_weight
                    for (aid, nid), (orig_activation, anchor_sig) in self._temporal_anchors.items():
                        if aid != agent_id:
                            continue
                        if (aid, nid) in eligible:
                            continue  # Already credited by fast-decay
                        sim = anchor_sig.similarity(now_sig)
                        temporal_strength = tcw * sim * orig_activation
                        if temporal_strength > 0.01:
                            temporal_eligible[(aid, nid)] = temporal_strength
                except Exception:
                    pass  # TemporalSignature not available — skip fallback

            all_eligible = {**eligible, **temporal_eligible}
            if not all_eligible:
                return credited

            # Normalize so total credit = reward
            total_strength = sum(all_eligible.values())
            for (aid, nid), strength in all_eligible.items():
                proportion = strength / total_strength
                credit = reward * proportion
                self.credit_node(aid, nid, credit)
                credited.append((nid, credit))

        # Emit ONE LEARN headline aggregating the per-node updates,
        # outside the lock so disk I/O does not block the next reward
        # arrival.  ``credited`` was populated inside the lock; reading
        # it here is safe because the loop has exited.
        if credited:
            try:
                from maxim.simulation.sim_logger import sim_learn

                top_nid, top_credit = max(credited, key=lambda kv: abs(kv[1]))
                more = f" +{len(credited) - 1} more" if len(credited) > 1 else ""
                sim_learn(
                    f"reward distributed (reward={reward:+.2f}, {len(credited)} nodes)",
                    detail=f"top: {top_nid[:16]} credit={top_credit:+.3f}{more}",
                    source="NAc",
                    agent_id=agent_id or None,
                    reward=reward,
                    nodes_credited=len(credited),
                )
            except Exception:
                pass

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
        """Decay all eligibility traces. Called on each tick.

        Also prunes temporal anchors whose fast-decay trace has expired
        AND whose temporal signature is older than ``temporal_window_seconds``.
        This prevents ``_temporal_anchors`` from growing unboundedly.
        """
        now = time.time()
        temporal_window = self.config.temporal_window_seconds
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
                # Prune temporal anchor if the fast-decay trace expired
                # AND the anchor is old enough (beyond temporal window)
                anchor = self._temporal_anchors.get(key)
                if anchor is not None:
                    _, sig = anchor
                    age = now - getattr(sig, "timestamp", 0.0)
                    if age > temporal_window:
                        del self._temporal_anchors[key]

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

    def dump(self) -> dict[str, Any]:
        """Return NAc state as a JSON-serializable dict.

        P3.5 Stage 1 — BioSystemSnapshot Protocol conformance. Acquires
        the same mutex the pre-refactor save() body held.
        """
        with self._lock:
            return {
                "version": "1.0",  # legacy payload string — tombstoned, do not bump
                "links": {event_sig: [link.to_dict() for link in links] for event_sig, links in self._links.items()},
                "outcome_index": {k: list(v) for k, v in self._outcome_index.items()},
                "priors": self._priors,
                "total_observations": self._total_observations,
                "reward_bias": {f"{aid}:{nid}": bias for (aid, nid), bias in self._reward_bias.items()},
                "goal_reward_bias": dict(self._goal_reward_bias),
                # G4: cluster-keyed tool reward bias (Track 2 of
                # grounded_language_acquisition.md Phase 0+). Keys join
                # the (agent_id, cluster_id, tool_signature) triple with
                # ``\x1f`` (ASCII unit separator) so they can be split
                # back unambiguously even when tool_signature contains
                # ``:`` (which ``build_tool_signature`` does for the
                # ``use`` tool — ``tool:use:dodge``).
                "cluster_reward_bias": {
                    f"{aid}\x1f{cid}\x1f{tsig}": bias for (aid, cid, tsig), bias in self._cluster_reward_bias.items()
                },
            }

    def load_state(self, state: dict[str, Any]) -> None:
        """Mutate self in place from a state dict.

        P3.5 Stage 1 — BioSystemSnapshot Protocol conformance. Preserves
        runtime wires (self.config, self.ec, whatever was wired at
        construction). Does NOT acquire the NAc mutex because callers
        expect load-time quiescence; acquiring self._lock here would
        deadlock with any concurrent observe() call in a long-running
        sim.

        Round 2 fold: the payload-layer ``version`` check is removed
        for consistency with the envelope-authoritative versioning
        tombstone documented in ``memory/snapshot.py``.
        """
        self._links = {
            event_sig: [CausalLink.from_dict(link_data) for link_data in links]
            for event_sig, links in state.get("links", {}).items()
        }
        self._outcome_index = {k: set(v) for k, v in state.get("outcome_index", {}).items()}
        self._priors = state.get("priors", {})
        self._total_observations = state.get("total_observations", 0)

        # P2: Load reward biases
        self._reward_bias = {}
        for key_str, bias in state.get("reward_bias", {}).items():
            parts = key_str.split(":", 1)
            if len(parts) == 2:
                self._reward_bias[(parts[0], parts[1])] = bias

        # Goal-level reward biases (backward-compatible: missing → empty)
        self._goal_reward_bias = dict(state.get("goal_reward_bias", {}))

        # G4: cluster-keyed reward biases. Missing field → empty dict
        # (backward-compatible: every aut_nac.json written before G4
        # closure lacks this key, and substrate-primary still works —
        # ``cluster_reward_bias()`` returns 0.0 for unknown triples).
        self._cluster_reward_bias = {}
        for key_str, bias in state.get("cluster_reward_bias", {}).items():
            parts = key_str.split("\x1f", 2)
            if len(parts) == 3:
                self._cluster_reward_bias[(parts[0], parts[1], parts[2])] = bias

    def save(self, path: str | None = None) -> None:
        """Save NAc state to JSON file.

        If ``path`` is omitted, falls back to ``self.config.persistence_path``.
        Raises ``ValueError`` if neither is set.
        """
        path = path or self.config.persistence_path
        if path is None:
            raise ValueError("NAc.save() requires a path or NACConfig.persistence_path to be set")

        from maxim.utils.atomic_io import atomic_write_json
        from maxim.utils.format_version import with_format_version

        atomic_write_json(path, with_format_version(self.dump()))
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
            state = json.load(f)

        from maxim.utils.format_version import check_format_version

        check_format_version(state, "nac", log=logger)
        self.load_state(state)
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
