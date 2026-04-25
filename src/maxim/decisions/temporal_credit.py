"""TemporalCreditDistributor — centralized temporal credit attribution.

Composes NAc + SCN for temporal-phase-aware reward distribution.  NOT a
bio-system — software wiring convenience.  Lives in ``decisions/`` as
plumbing, not a biological analog.

Ownership invariant: ``_temporal_anchors`` stays on NAc.  The distributor
READS NAc's anchors via ``nac.get_temporal_anchors(agent_id)`` during
``distribute()`` but does NOT own them.  ``LinguisticEncoder.encode_decomposed()``
continues calling ``nac.update_eligibility(temporal_sig=...)`` directly.

Double-attribution prevention: the distributor REPLACES
``_distribute_reward_from_reaction`` in ``build_bio_stack`` — they must NOT
coexist for the same agent.  ``NAc.distribute_reward()`` stays public for
tests/scripts, but production reward flows go through the distributor.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from maxim.decisions.valence_signal import ValenceSignal
from maxim.time.temporal_event import TemporalEvent
from maxim.time.temporal_signature import TemporalSignature

logger = logging.getLogger(__name__)


class TemporalCreditDistributor:
    """Centralized temporal credit attribution.

    Composes NAc + SCN for temporal-phase-aware reward distribution.
    Three public methods: ``record_event``, ``distribute``, ``cleanup_session``.

    Thread safety: own ``threading.RLock``.  Lock ordering: distributor → NAc
    (never reverse).
    """

    def __init__(
        self,
        nac: Any,
        scn: Any,
        *,
        temporal_credit_weight: float | None = None,
    ) -> None:
        """Construct a distributor.

        Args:
            nac: NAc instance (required).  The distributor delegates to
                ``nac.credit_node()``, ``nac.credit_goal()``, reads
                ``nac.get_temporal_anchors()``.
            scn: SCN instance (required, keyword-only).  The distributor's
                primary value IS temporal credit — making SCN optional would
                degrade it to a passthrough with no warning.
            temporal_credit_weight: Override for phase-similarity credit
                weight.  When None, reads from ``nac.config.temporal_credit_weight``.
        """
        self._nac = nac
        self._scn = scn
        self._lock = threading.RLock()

        # Phase-similarity credit weight (cross-session generalization signal).
        self._tcw = (
            temporal_credit_weight
            if temporal_credit_weight is not None
            else getattr(getattr(nac, "config", None), "temporal_credit_weight", 0.3)
        )

        # Session-scoped event_ids registered with SCN for cleanup.
        self._session_event_ids: list[str] = []

        # Deliberation events stored internally for the agent loop to
        # query via get_pending_deliberation_event (if needed later).
        # Currently unused — Phase 3 credit_goal fires directly from
        # tool_dispatch.py.  Kept for future Phase 4+ wiring.
        self._deliberation_events: dict[str, TemporalEvent] = {}

        # Last emitted ValenceSignal (for callers that need to forward
        # to WMS after distribute()).  None until first distribute().
        self._last_valence_signal: ValenceSignal | None = None

    def record_event(self, event: TemporalEvent) -> None:
        """Record a temporal event for future credit attribution.

        Calls ``nac.update_eligibility()`` for the fast-decay trace,
        registers with SCN for bin indexing.  For ``"deliberation"``
        events, registers with LOW significance (0.1) so thoughts lose
        eviction battles to real memories.
        """
        with self._lock:
            # Fast-decay eligibility trace on NAc
            self._nac.update_eligibility(
                event.agent_id,
                event.event_signature,
                event.activation,
                temporal_sig=event.temporal_sig,
            )

            # SCN bin registration
            significance = 0.1 if event.event_type == "deliberation" else 0.3
            try:
                self._scn.register(
                    event.event_id,
                    event.temporal_sig,
                    significance=significance,
                )
            except Exception as e:
                logger.debug("SCN registration failed for %s: %s", event.event_id, e)

            # Track for session-end cleanup
            self._session_event_ids.append(event.event_id)

            # Store deliberation events by goal for potential future lookup
            if event.event_type == "deliberation":
                goal = event.context.get("goal")
                if goal is not None:
                    self._deliberation_events[goal] = event

    def distribute(
        self,
        agent_id: str,
        reward: float,
        goal_tag: str | None = None,
    ) -> list[tuple[str, float]]:
        """Distribute reward with temporal credit fallback.

        Two-path credit assignment:
        1. **Fast-decay path**: nodes with active eligibility traces (> 0.01)
           receive credit proportional to trace strength.
        2. **Phase-similarity fallback**: nodes whose fast-decay trace has
           expired but that have temporal anchors receive credit proportional
           to ``TemporalSignature.similarity(anchor, now)`` scaled by
           ``temporal_credit_weight``.

        When ``goal_tag`` is not None, also calls ``nac.credit_goal()`` for
        ThoughtGate modulation.

        Returns list of ``(node_id, credit_applied)`` for logging.
        """
        with self._lock:
            # Read eligibility traces from NAc
            eligible: dict[str, float] = {}
            for (aid, nid), strength in self._nac._eligibility.items():
                if aid == agent_id and strength > 0.01:
                    eligible[nid] = strength

            # Phase-similarity fallback from temporal anchors
            temporal_eligible: dict[str, float] = {}
            anchors = self._nac.get_temporal_anchors(agent_id)
            if anchors:
                now_sig = TemporalSignature.now()
                for nid, (orig_activation, anchor_sig) in anchors.items():
                    if nid in eligible:
                        continue  # Already credited by fast-decay
                    sim = anchor_sig.similarity(now_sig)
                    temporal_strength = self._tcw * sim * orig_activation
                    if temporal_strength > 0.01:
                        temporal_eligible[nid] = temporal_strength

            all_eligible = {**eligible, **temporal_eligible}
            if not all_eligible:
                # Still credit goal even if no substrate nodes are eligible
                if goal_tag is not None:
                    self._nac.credit_goal(goal_tag, reward)
                return []

            # Normalize so total credit = reward
            total_strength = sum(all_eligible.values())
            credited: list[tuple[str, float]] = []
            for nid, strength in all_eligible.items():
                proportion = strength / total_strength
                credit = reward * proportion
                self._nac.credit_node(agent_id, nid, credit)
                credited.append((nid, credit))

            # Goal-level credit (orthogonal to substrate node credit)
            if goal_tag is not None:
                self._nac.credit_goal(goal_tag, reward)

            # Emit ValenceSignal for WMS salience modulation.
            # Stored on self for callers that need it (agent loop → WMS).
            self._last_valence_signal = ValenceSignal.from_credit(
                value=reward, goal_tag=goal_tag, source="temporal_credit"
            )

            return credited

    @property
    def last_valence_signal(self) -> ValenceSignal | None:
        """The most recently emitted ValenceSignal from distribute().

        Returns None if distribute() hasn't been called yet.
        Callers can forward this to ``wms.receive_valence(tick, signal.value)``.
        """
        return self._last_valence_signal

    def cleanup_session(self) -> None:
        """Unregister session-scoped events from SCN.

        Called from ``BioStack.on_session_end()``.  Idempotent — safe to
        call multiple times (second call is a no-op on empty list).
        """
        with self._lock:
            for event_id in self._session_event_ids:
                try:
                    self._scn.unregister(event_id)
                except Exception:
                    pass  # Best-effort cleanup
            self._session_event_ids.clear()
            self._deliberation_events.clear()
