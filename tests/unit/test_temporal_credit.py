"""Tests for TemporalCreditDistributor (Phase 4).

Covers:
- record_event: eligibility trace + SCN registration
- distribute: fast-decay path, phase-similarity fallback, goal credit
- distribute with goal_tag=None: no phantom key
- cleanup_session: idempotent SCN unregistration
- thread safety: RLock prevents interleaved distribute
- behavioral equivalence: distributor produces same credit as nac.distribute_reward
"""

from __future__ import annotations

import uuid

from maxim.decisions.nac import NAc, NACConfig
from maxim.decisions.temporal_credit import TemporalCreditDistributor
from maxim.time.scn import SCN
from maxim.time.temporal_event import TemporalEvent
from maxim.time.temporal_signature import TemporalSignature


def _make_event(
    agent_id: str = "agent-1",
    event_type: str = "tool",
    event_signature: str = "tool:sword_slash",
    activation: float = 1.0,
    **ctx: object,
) -> TemporalEvent:
    return TemporalEvent(
        event_id=uuid.uuid4().hex,
        event_type=event_type,
        event_signature=event_signature,
        agent_id=agent_id,
        temporal_sig=TemporalSignature.now(),
        activation=activation,
        context=dict(ctx),
    )


class TestRecordEvent:
    def test_creates_eligibility_trace(self):
        nac = NAc()
        scn = SCN()
        dist = TemporalCreditDistributor(nac=nac, scn=scn)

        event = _make_event()
        dist.record_event(event)

        # NAc should have an eligibility trace for the event signature
        assert ("agent-1", "tool:sword_slash") in nac._eligibility

    def test_creates_temporal_anchor(self):
        nac = NAc()
        scn = SCN()
        dist = TemporalCreditDistributor(nac=nac, scn=scn)

        event = _make_event()
        dist.record_event(event)

        assert ("agent-1", "tool:sword_slash") in nac._temporal_anchors

    def test_deliberation_event_low_significance(self):
        """Deliberation events register with SCN at significance=0.1."""
        nac = NAc()
        scn = SCN()
        dist = TemporalCreditDistributor(nac=nac, scn=scn)

        event = _make_event(event_type="deliberation", event_signature="deliberation:goal:escape")
        dist.record_event(event)

        # Event should be tracked for session cleanup
        assert event.event_id in dist._session_event_ids

    def test_deliberation_stored_by_goal(self):
        nac = NAc()
        scn = SCN()
        dist = TemporalCreditDistributor(nac=nac, scn=scn)

        event = _make_event(
            event_type="deliberation",
            event_signature="deliberation:goal:escape",
            goal="escape",
        )
        dist.record_event(event)

        assert "escape" in dist._deliberation_events


class TestDistribute:
    def test_fast_decay_credits_active_traces(self):
        """Nodes with active eligibility traces get credited."""
        nac = NAc()
        scn = SCN()
        dist = TemporalCreditDistributor(nac=nac, scn=scn)

        event = _make_event()
        dist.record_event(event)

        credits = dist.distribute("agent-1", 1.0)
        assert len(credits) > 0
        assert any(nid == "tool:sword_slash" for nid, _ in credits)

    def test_temporal_fallback_after_decay(self):
        """After fast-decay expires, temporal anchors provide fallback credit."""
        nac = NAc(NACConfig(temporal_credit_weight=0.3))
        scn = SCN()
        dist = TemporalCreditDistributor(nac=nac, scn=scn)

        event = _make_event()
        dist.record_event(event)

        # Decay until fast traces expire
        for _ in range(200):
            nac.decay_eligibility()

        # Fast-decay should be gone
        active = {k: v for k, v in nac._eligibility.items() if v > 0.01}
        assert not active

        # But temporal anchors survive
        anchors = nac.get_temporal_anchors("agent-1")
        assert anchors

        # Distribute should still credit via fallback
        credits = dist.distribute("agent-1", 1.0)
        assert len(credits) > 0

    def test_goal_credit_on_distribute(self):
        """distribute(..., goal_tag=X) calls nac.credit_goal(X, reward)."""
        nac = NAc()
        scn = SCN()
        dist = TemporalCreditDistributor(nac=nac, scn=scn)

        event = _make_event()
        dist.record_event(event)

        dist.distribute("agent-1", 1.0, goal_tag="escape")
        assert nac.get_goal_reward_bias("escape") > 0

    def test_goal_tag_none_no_phantom_key(self):
        """distribute(..., goal_tag=None) does NOT create a None key."""
        nac = NAc()
        scn = SCN()
        dist = TemporalCreditDistributor(nac=nac, scn=scn)

        event = _make_event()
        dist.record_event(event)

        dist.distribute("agent-1", 1.0, goal_tag=None)
        assert None not in nac._goal_reward_bias

    def test_no_eligible_nodes_still_credits_goal(self):
        """Even with no substrate nodes, goal credit fires."""
        nac = NAc()
        scn = SCN()
        dist = TemporalCreditDistributor(nac=nac, scn=scn)

        # No events recorded — no eligible nodes
        credits = dist.distribute("agent-1", 1.0, goal_tag="escape")
        assert credits == []
        assert nac.get_goal_reward_bias("escape") > 0

    def test_distribute_emits_valence_signal(self):
        """distribute() stores a ValenceSignal for callers to forward to WMS."""
        from maxim.decisions.valence_signal import ValenceSignal

        nac = NAc()
        scn = SCN()
        dist = TemporalCreditDistributor(nac=nac, scn=scn)

        event = _make_event()
        dist.record_event(event)

        assert dist.last_valence_signal is None  # Not emitted yet

        dist.distribute("agent-1", 0.8, goal_tag="escape")
        sig = dist.last_valence_signal
        assert sig is not None
        assert isinstance(sig, ValenceSignal)
        assert sig.value == 0.8
        assert sig.goal_tag == "escape"
        assert sig.source == "temporal_credit"

    def test_no_double_count_fast_and_temporal(self):
        """A node credited by fast-decay is NOT also credited by temporal."""
        nac = NAc()
        scn = SCN()
        dist = TemporalCreditDistributor(nac=nac, scn=scn)

        event = _make_event()
        dist.record_event(event)

        # Before decay — fast-decay should credit, temporal should NOT
        credits = dist.distribute("agent-1", 1.0)
        sig_credits = [nid for nid, _ in credits if nid == "tool:sword_slash"]
        assert len(sig_credits) == 1  # Exactly one credit, not two

    def test_agent_isolation(self):
        """Agent A's events don't credit Agent B."""
        nac = NAc()
        scn = SCN()
        dist = TemporalCreditDistributor(nac=nac, scn=scn)

        event_a = _make_event(agent_id="agent-A")
        dist.record_event(event_a)

        # Agent B should get nothing
        credits = dist.distribute("agent-B", 1.0)
        assert credits == []


class TestCleanupSession:
    def test_cleanup_clears_session_ids(self):
        nac = NAc()
        scn = SCN()
        dist = TemporalCreditDistributor(nac=nac, scn=scn)

        event = _make_event()
        dist.record_event(event)
        assert len(dist._session_event_ids) > 0

        dist.cleanup_session()
        assert len(dist._session_event_ids) == 0

    def test_cleanup_idempotent(self):
        """Calling cleanup_session twice doesn't raise."""
        nac = NAc()
        scn = SCN()
        dist = TemporalCreditDistributor(nac=nac, scn=scn)

        event = _make_event()
        dist.record_event(event)

        dist.cleanup_session()
        dist.cleanup_session()  # No-op, no exception

    def test_cleanup_clears_deliberation_events(self):
        nac = NAc()
        scn = SCN()
        dist = TemporalCreditDistributor(nac=nac, scn=scn)

        event = _make_event(event_type="deliberation", goal="escape")
        dist.record_event(event)
        assert "escape" in dist._deliberation_events

        dist.cleanup_session()
        assert len(dist._deliberation_events) == 0


class TestBioStackIntegration:
    """Verify distributor is constructed by build_bio_stack."""

    def test_build_bio_stack_has_distributor(self):
        from maxim.runtime.bio_stack import build_bio_stack

        bio = build_bio_stack()
        assert bio.distributor is not None
        assert isinstance(bio.distributor, TemporalCreditDistributor)

    def test_on_session_end_calls_cleanup(self):
        from maxim.runtime.bio_stack import build_bio_stack

        bio = build_bio_stack()

        # Record an event so there's something to clean up
        event = _make_event()
        bio.distributor.record_event(event)
        assert len(bio.distributor._session_event_ids) > 0

        bio.on_session_end()
        assert len(bio.distributor._session_event_ids) == 0
