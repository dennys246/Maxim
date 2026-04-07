"""CommunicationBridge — NAc reward attribution for communication tools.

Bridges communication tools (send_message, call_user) to NAc for reward
attribution. Watches for outbound messages and inbound responses, linking
them causally so NAc can learn when messaging is rewarding.

Implements PromotionSource protocol — confirmed communication patterns
(e.g., "morning status texts get positive replies") are surfaced as
PromotionCandidates for SemanticPromoter → ATL promotion.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from datetime import datetime
from typing import TYPE_CHECKING, Any

from maxim.agents.bus import Percept, ToolResult
from maxim.decisions.causal_link import Valence

if TYPE_CHECKING:
    from maxim.agents.bus import AgentBus
    from maxim.decisions.nac import NAc
    from maxim.memory.semantic_promoter import PromotionCandidate

logger = logging.getLogger(__name__)


class CommunicationBridge:
    """Bridges communication tools to NAc for reward attribution.

    Watches for outbound messages and inbound responses, linking them
    causally so NAc can learn when messaging is rewarding.

    Also implements PromotionSource protocol — confirmed communication
    patterns (e.g., "morning status texts get positive replies") are
    surfaced as PromotionCandidates for SemanticPromoter → ATL promotion.
    """

    def __init__(self, bus: "AgentBus", nac: "NAc") -> None:
        self.bus = bus
        self.nac = nac
        self.pending_outbound: deque[dict[str, Any]] = deque(maxlen=50)
        self._response_history: list[dict[str, Any]] = []

        bus.subscribe(ToolResult, self._on_tool_result)
        bus.subscribe(Percept, self._on_percept)

    def _on_tool_result(self, result: ToolResult) -> None:
        """Track outbound communication tool results."""
        if result.tool_name not in ("send_message", "call_user"):
            return

        params = result.params

        # Record outbound message as NAc event
        event_id = self.nac.record_event(
            event_type="communication",
            event_signature=result.tool_name,
            context={
                "channel": params.get("channel"),
                "recipient": params.get("recipient"),
                "body_length": len(params.get("body", params.get("message", ""))),
                "time_of_day": datetime.now().hour,
            },
        )
        self.pending_outbound.append(
            {
                "event_id": event_id,
                "tool": result.tool_name,
                "timestamp": time.time(),
                "recipient": params.get("recipient"),
            }
        )

    def _on_percept(self, percept: Percept) -> None:
        """Handle inbound communication — attribute to recent outbound."""
        if not percept.source.startswith("comms:"):
            return

        sender = (percept.metadata or {}).get("sender")
        recent = self._find_recent_outbound(sender, max_age=300)

        if recent:
            valence = self._classify_response_valence(percept.content or "")
            self.nac.record_outcome_full(
                outcome_type="user_response",
                outcome_signature="user_replied",
                outcome_valence=valence,
                context={
                    "response_content": (percept.content or "")[:100],
                    "time_of_day": datetime.now().hour,
                    "delay_seconds": time.time() - recent["timestamp"],
                },
                attributed_event_signature=recent["tool"],
            )
            self._response_history.append(
                {
                    "tool": recent["tool"],
                    "valence": valence.value if hasattr(valence, "value") else str(valence),
                    "delay": time.time() - recent["timestamp"],
                    "time_of_day": datetime.now().hour,
                    "timestamp": time.time(),
                }
            )
            self.pending_outbound.remove(recent)

    def _find_recent_outbound(self, sender: str | None, max_age: float = 300) -> dict[str, Any] | None:
        """Find most recent outbound message to this sender."""
        now = time.time()
        for outbound in reversed(self.pending_outbound):
            if now - outbound["timestamp"] > max_age:
                continue
            # Match by recipient if sender is known
            if sender and outbound.get("recipient") and sender != outbound["recipient"]:
                continue
            return outbound
        return None

    def _classify_response_valence(self, text: str) -> Valence:
        """Simple heuristic for response sentiment."""
        positive = {"thanks", "great", "perfect", "helpful", "awesome", "good", "nice", "cool", "yes"}
        negative = {"stop", "don't", "annoying", "spam", "quiet", "shut", "no", "wrong"}
        words = set(text.lower().split())

        if words & positive:
            return Valence.POSITIVE
        elif words & negative:
            return Valence.NEGATIVE
        else:
            return Valence.NEUTRAL

    def timeout_unanswered(self) -> None:
        """Called by agent loop's periodic maintenance.

        Messages older than 10 minutes with no response get NEGATIVE
        attribution.
        """
        now = time.time()
        for outbound in list(self.pending_outbound):
            if now - outbound["timestamp"] > 600:  # 10 min timeout
                self.nac.record_outcome_full(
                    outcome_type="user_response",
                    outcome_signature="user_ignored",
                    outcome_valence=Valence.NEGATIVE,
                    context={"timeout": True},
                    attributed_event_signature=outbound["tool"],
                )
                self.pending_outbound.remove(outbound)

    # ── PromotionSource protocol (for SemanticPromoter → ATL) ──────────

    def get_promotion_candidates(
        self,
        min_confidence: float = 0.6,
        min_observations: int = 3,
    ) -> list["PromotionCandidate"]:
        """Surface confirmed communication patterns for ATL promotion.

        Analyzes response history to find patterns like:
        - "morning status messages → positive reply" (temporal pattern)
        - "goal completion notifications → user thanks" (event pattern)

        These become ATL semantic concepts (e.g., "user prefers morning updates")
        linked back to source episodes via CrossLayerGraph.
        """
        candidates = []
        # NAc already tracks causal links — delegate and filter for comms.
        for candidate in self.nac.get_promotion_candidates(min_confidence, min_observations):
            if candidate.pattern_name.startswith("send_message →") or candidate.pattern_name.startswith("call_user →"):
                candidates.append(candidate)
        return candidates
