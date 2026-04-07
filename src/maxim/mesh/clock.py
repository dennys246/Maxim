"""PeerClockEstimator — learns clock offsets between mesh peers.

Uses heartbeat round-trip times (NTP-lite) to estimate per-peer clock
offsets. Corrects incoming TemporalSignatures before SCN registration
so temporal bins align across agents with different system clocks.

The SCN's phase-normalized indexing (0.0-1.0 per rhythm) is naturally
resistant to small offsets — skew only matters when it shifts a memory
into the wrong hour bin (>30 minutes). This estimator handles larger
drift gracefully.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class PeerClockEstimate:
    """Estimated clock offset for a single peer."""

    peer_id: str
    offset_s: float = 0.0  # Estimated: peer_time - local_time
    confidence: float = 0.0  # 0.0 = no data, 1.0 = many sync points
    sync_points: int = 0  # Number of round-trips used to estimate
    last_sync: float = 0.0  # Local monotonic time of last sync

    def to_dict(self) -> dict[str, Any]:
        return {
            "peer_id": self.peer_id,
            "offset_s": round(self.offset_s, 4),
            "confidence": round(self.confidence, 3),
            "sync_points": self.sync_points,
            "last_sync": self.last_sync,
        }


class PeerClockEstimator:
    """Learns clock offsets between local SCN and peer agents.

    Each heartbeat exchange includes the sender's wall-clock time. The
    receiver measures RTT and estimates the offset using NTP-lite:
        offset = (peer_timestamp + rtt/2) - local_time

    Offsets are smoothed with an EMA (alpha=0.3) and stabilize after
    ~10 sync points (confidence reaches 1.0).
    """

    # EMA smoothing factor — ~3 sync points to stabilize
    ALPHA = 0.3
    # Full confidence after this many sync points
    FULL_CONFIDENCE_SYNCS = 10
    # Minimum confidence before corrections are applied
    MIN_CORRECTION_CONFIDENCE = 0.1

    def __init__(self) -> None:
        self._estimates: dict[str, PeerClockEstimate] = {}

    def record_sync_point(
        self,
        peer_id: str,
        peer_timestamp: float,
        rtt_s: float,
    ) -> None:
        """Update clock estimate from a heartbeat round-trip.

        Args:
            peer_id: The peer's node_id
            peer_timestamp: The peer's reported wall-clock time (from HEARTBEAT payload)
            rtt_s: Measured round-trip time for the heartbeat exchange
        """
        local_time = time.time()
        # NTP-lite: estimate when the peer sent its timestamp
        estimated_peer_now = peer_timestamp + rtt_s / 2
        offset = estimated_peer_now - local_time

        est = self._estimates.get(peer_id)
        if est is None:
            est = PeerClockEstimate(peer_id=peer_id)
            self._estimates[peer_id] = est

        # EMA smoothing
        if est.sync_points == 0:
            est.offset_s = offset
        else:
            est.offset_s = self.ALPHA * offset + (1 - self.ALPHA) * est.offset_s

        est.sync_points += 1
        est.confidence = min(1.0, est.sync_points / self.FULL_CONFIDENCE_SYNCS)
        est.last_sync = time.monotonic()

        if est.sync_points <= 3 or est.sync_points % 10 == 0:
            logger.debug(
                "clock: peer %s offset=%.3fs confidence=%.2f (sync #%d)",
                peer_id,
                est.offset_s,
                est.confidence,
                est.sync_points,
            )

    def correct_timestamp(self, peer_id: str, peer_timestamp: float) -> float:
        """Convert a peer's timestamp to local-clock equivalent.

        Returns the original timestamp if no correction is possible yet.
        """
        est = self._estimates.get(peer_id)
        if est is None or est.confidence < self.MIN_CORRECTION_CONFIDENCE:
            return peer_timestamp
        return peer_timestamp - est.offset_s

    def correct_signature(
        self,
        peer_id: str,
        signature: Any,
    ) -> Any:
        """Correct a peer's TemporalSignature for clock skew.

        Re-derives phases from the corrected timestamp so SCN bins
        align with local time.
        """
        from maxim.time.temporal_signature import TemporalSignature

        corrected_ts = self.correct_timestamp(peer_id, signature.timestamp)
        return TemporalSignature.from_timestamp(corrected_ts)

    def get_estimate(self, peer_id: str) -> PeerClockEstimate | None:
        return self._estimates.get(peer_id)

    def get_all_estimates(self) -> list[dict[str, Any]]:
        return [est.to_dict() for est in self._estimates.values()]

    def peer_count(self) -> int:
        return len(self._estimates)
