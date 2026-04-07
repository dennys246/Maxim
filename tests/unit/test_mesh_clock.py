"""Tests for mesh clock synchronization (Phase 7).

Covers PeerClockEstimator and SCN.register_external().
"""

from __future__ import annotations

import time

from maxim.mesh.clock import PeerClockEstimate, PeerClockEstimator
from maxim.time.scn import SCN
from maxim.time.temporal_signature import TemporalSignature


# ─────────────────────────────────────────────────────────────────────────────
# PeerClockEstimate
# ─────────────────────────────────────────────────────────────────────────────


class TestPeerClockEstimate:
    def test_to_dict(self):
        est = PeerClockEstimate(
            peer_id="node1",
            offset_s=1.234567,
            confidence=0.567,
            sync_points=5,
        )
        d = est.to_dict()
        assert d["peer_id"] == "node1"
        assert d["offset_s"] == 1.2346  # Rounded to 4 decimals
        assert d["confidence"] == 0.567


# ─────────────────────────────────────────────────────────────────────────────
# PeerClockEstimator
# ─────────────────────────────────────────────────────────────────────────────


class TestPeerClockEstimator:
    def test_empty_estimator(self):
        est = PeerClockEstimator()
        assert est.peer_count() == 0
        assert est.get_estimate("nobody") is None
        assert est.get_all_estimates() == []

    def test_first_sync_point_sets_offset(self):
        est = PeerClockEstimator()
        local_now = time.time()
        # Peer clock is 2 seconds ahead, RTT is 0.1s
        peer_ts = local_now + 2.0
        est.record_sync_point("peer1", peer_ts, rtt_s=0.1)

        result = est.get_estimate("peer1")
        assert result is not None
        assert result.sync_points == 1
        # offset ≈ 2.05 (peer ahead by 2s, plus half RTT)
        assert abs(result.offset_s - 2.05) < 0.2  # Allow for timing jitter

    def test_confidence_increases_with_syncs(self):
        est = PeerClockEstimator()
        local_now = time.time()

        for i in range(10):
            est.record_sync_point("peer1", local_now + 1.0, rtt_s=0.05)

        result = est.get_estimate("peer1")
        assert result is not None
        assert result.confidence == 1.0
        assert result.sync_points == 10

    def test_ema_smoothing(self):
        est = PeerClockEstimator()
        local_now = time.time()

        # First sync: offset ~1.0
        est.record_sync_point("peer1", local_now + 1.0, rtt_s=0.0)
        first_offset = est.get_estimate("peer1").offset_s

        # Second sync with wildly different offset — EMA dampens it
        est.record_sync_point("peer1", local_now + 5.0, rtt_s=0.0)
        second_offset = est.get_estimate("peer1").offset_s

        # Should be between 1.0 and 5.0 (EMA, not jump to 5.0)
        assert second_offset > first_offset
        assert second_offset < 5.0

    def test_correct_timestamp_no_estimate(self):
        est = PeerClockEstimator()
        ts = 1000.0
        assert est.correct_timestamp("unknown", ts) == ts  # No correction

    def test_correct_timestamp_low_confidence(self):
        est = PeerClockEstimator()
        est._estimates["peer1"] = PeerClockEstimate(
            peer_id="peer1",
            offset_s=5.0,
            confidence=0.05,
        )
        ts = 1000.0
        assert est.correct_timestamp("peer1", ts) == ts  # Too low confidence

    def test_correct_timestamp_applies_offset(self):
        est = PeerClockEstimator()
        est._estimates["peer1"] = PeerClockEstimate(
            peer_id="peer1",
            offset_s=2.0,
            confidence=0.5,
            sync_points=5,
        )
        ts = 1002.0
        corrected = est.correct_timestamp("peer1", ts)
        assert abs(corrected - 1000.0) < 0.01

    def test_correct_signature(self):
        est = PeerClockEstimator()
        est._estimates["peer1"] = PeerClockEstimate(
            peer_id="peer1",
            offset_s=3600.0,
            confidence=0.8,
            sync_points=8,
        )
        # Create a signature 1 hour ahead of "real" time
        real_ts = time.time()
        peer_sig = TemporalSignature.from_timestamp(real_ts + 3600.0)

        corrected = est.correct_signature("peer1", peer_sig)
        # The corrected timestamp should be close to real_ts
        assert abs(corrected.timestamp - real_ts) < 1.0

    def test_get_all_estimates(self):
        est = PeerClockEstimator()
        local_now = time.time()
        est.record_sync_point("peer1", local_now, rtt_s=0.01)
        est.record_sync_point("peer2", local_now + 1.0, rtt_s=0.02)

        all_est = est.get_all_estimates()
        assert len(all_est) == 2
        ids = {e["peer_id"] for e in all_est}
        assert ids == {"peer1", "peer2"}


# ─────────────────────────────────────────────────────────────────────────────
# SCN.register_external
# ─────────────────────────────────────────────────────────────────────────────


class TestSCNRegisterExternal:
    def test_registers_with_corrected_signature(self):
        scn = SCN()
        estimator = PeerClockEstimator()
        # Set a known offset
        estimator._estimates["peer1"] = PeerClockEstimate(
            peer_id="peer1",
            offset_s=3600.0,
            confidence=0.9,
            sync_points=9,
        )

        # Peer's timestamp is 1 hour ahead
        real_ts = time.time()
        peer_sig = TemporalSignature.from_timestamp(real_ts + 3600.0)

        scn.register_external("peer1:mem_001", peer_sig, "peer1", estimator)

        # The memory should be registered
        stored_sig = scn.get_signature("peer1:mem_001")
        assert stored_sig is not None
        # Corrected timestamp should be close to real_ts
        assert abs(stored_sig.timestamp - real_ts) < 1.0

    def test_registers_without_estimator_data(self):
        """When no clock estimate exists, registers with uncorrected signature."""
        scn = SCN()
        estimator = PeerClockEstimator()  # Empty — no sync points

        sig = TemporalSignature.from_timestamp(time.time())
        scn.register_external("peer1:mem_002", sig, "peer1", estimator)

        stored = scn.get_signature("peer1:mem_002")
        assert stored is not None
        # Should use the original timestamp (no correction)
        assert abs(stored.timestamp - sig.timestamp) < 0.01

    def test_significance_passed_through(self):
        scn = SCN()
        estimator = PeerClockEstimator()
        sig = TemporalSignature.from_timestamp(time.time())

        scn.register_external("peer1:mem_003", sig, "peer1", estimator, significance=0.9)

        # Verify it's in the bin (significance affects eviction, not query)
        stored = scn.get_signature("peer1:mem_003")
        assert stored is not None

    def test_queryable_after_external_registration(self):
        """Externally registered memories show up in temporal queries."""
        scn = SCN()
        estimator = PeerClockEstimator()
        sig = TemporalSignature.from_timestamp(time.time())
        hour_bin = sig.to_bins()[0]

        scn.register_external("peer1:mem_004", sig, "peer1", estimator)

        results = scn.query_hour(hour_bin)
        assert "peer1:mem_004" in results
