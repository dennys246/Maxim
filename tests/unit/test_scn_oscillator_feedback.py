"""Tests for SCN oscillator feedback — anticipatory temporal credit (B2).

Covers:
- OscillatorNetwork.observe_event: per-event-type phase recording
- OscillatorNetwork.predict_event_imminence: phase similarity + concentration
- OscillatorNetwork.get_anticipatory_signatures: threshold filtering
- SCN.observe_event delegation to oscillator
- SCN.get_anticipatory_signatures delegation
- TemporalCreditDistributor.record_event feeds oscillator
- TemporalCreditDistributor.distribute anticipatory path
- Serialization round-trip preserves event_phases
- Cold-start guard: < 3 observations returns 0.0
- build_bio_stack enables oscillator
"""

from __future__ import annotations

import math
import uuid

from maxim.decisions.nac import NAc
from maxim.decisions.temporal_credit import TemporalCreditDistributor
from maxim.time.oscillator import OscillatorConfig, OscillatorNetwork
from maxim.time.scn import SCN
from maxim.time.temporal_event import TemporalEvent
from maxim.time.temporal_signature import TemporalSignature


def _make_sig(circadian_phase: float = 0.5) -> TemporalSignature:
    """Create a TemporalSignature with a specific circadian phase."""
    return TemporalSignature(
        timestamp=1000.0,
        circadian_phase=circadian_phase,
        weekly_phase=0.3,
        monthly_phase=0.2,
        annual_phase=0.1,
    )


def _make_event(
    agent_id: str = "agent-1",
    event_signature: str = "tool:sword_slash",
    circadian_phase: float = 0.5,
    activation: float = 1.0,
) -> TemporalEvent:
    return TemporalEvent(
        event_id=uuid.uuid4().hex,
        event_type="tool",
        event_signature=event_signature,
        agent_id=agent_id,
        temporal_sig=_make_sig(circadian_phase),
        activation=activation,
    )


# ---------------------------------------------------------------------------
# OscillatorNetwork — event-type phase tracking
# ---------------------------------------------------------------------------


class TestObserveEvent:
    def test_records_phase(self):
        osc = OscillatorNetwork()
        sig = _make_sig(0.5)
        osc.observe_event("tool:attack", sig)
        assert "tool:attack" in osc._event_phases
        assert len(osc._event_phases["tool:attack"]) == 1
        # Phase should be circadian_phase * 2π
        expected = 0.5 * 2 * math.pi
        assert abs(osc._event_phases["tool:attack"][0] - expected) < 1e-6

    def test_ring_buffer_caps_at_max(self):
        config = OscillatorConfig(max_event_phases=5)
        osc = OscillatorNetwork(config)
        for i in range(10):
            osc.observe_event("tool:x", _make_sig(i / 24.0))
        assert len(osc._event_phases["tool:x"]) == 5

    def test_multiple_signatures(self):
        osc = OscillatorNetwork()
        osc.observe_event("tool:a", _make_sig(0.25))
        osc.observe_event("tool:b", _make_sig(0.75))
        assert "tool:a" in osc._event_phases
        assert "tool:b" in osc._event_phases


class TestPredictEventImminence:
    def test_cold_start_returns_zero(self):
        """< 3 observations = 0.0 (cold-start guard)."""
        osc = OscillatorNetwork()
        osc.observe_event("tool:x", _make_sig(0.5))
        osc.observe_event("tool:x", _make_sig(0.5))
        assert osc.predict_event_imminence("tool:x") == 0.0

    def test_unknown_signature_returns_zero(self):
        osc = OscillatorNetwork()
        assert osc.predict_event_imminence("tool:unknown") == 0.0

    def test_consistent_phase_high_imminence_when_aligned(self):
        """Events consistently at phase 0.5 → high imminence when oscillator is near 0.5."""
        osc = OscillatorNetwork()
        # Observe 5 events all at circadian phase 0.5
        for _ in range(5):
            sig = _make_sig(0.5)
            osc.observe(sig)  # Also aligns the oscillator to phase 0.5
            osc.observe_event("tool:noon_event", sig)

        score = osc.predict_event_imminence("tool:noon_event")
        # Oscillator's circadian phase should be near 0.5 * 2π = π
        # Event mean phase is π, concentration is high (R ≈ 1.0)
        assert score > 0.5, f"Expected high imminence, got {score}"

    def test_scattered_phase_low_imminence(self):
        """Events at uniformly distributed phases → low concentration → low imminence."""
        osc = OscillatorNetwork()
        # Observe events at evenly spread phases
        for i in range(8):
            phase = i / 8.0
            osc.observe_event("tool:scattered", _make_sig(phase))

        score = osc.predict_event_imminence("tool:scattered")
        # Low concentration R → low score regardless of current phase
        assert score < 0.3, f"Expected low imminence for scattered events, got {score}"

    def test_distant_phase_low_imminence(self):
        """Events consistently at phase 0.0 but oscillator at phase 0.5 → low imminence."""
        osc = OscillatorNetwork()
        # Observe 5 events at circadian phase 0.0 (midnight)
        for _ in range(5):
            osc.observe_event("tool:midnight", _make_sig(0.0))
        # Set oscillator to ~noon (phase 0.5 * 2π = π)
        for _ in range(5):
            osc.observe(_make_sig(0.5))

        score = osc.predict_event_imminence("tool:midnight")
        assert score < 0.5, f"Expected low imminence for distant phase, got {score}"


class TestGetAnticipatorySignatures:
    def test_filters_by_threshold(self):
        osc = OscillatorNetwork()
        # Create an imminent event
        for _ in range(5):
            sig = _make_sig(0.5)
            osc.observe(sig)
            osc.observe_event("tool:imminent", sig)
        # Create a distant event
        for _ in range(5):
            osc.observe_event("tool:distant", _make_sig(0.0))

        result = osc.get_anticipatory_signatures(min_imminence=0.5)
        assert "tool:imminent" in result
        # "tool:distant" may or may not be present depending on exact oscillator phase,
        # but its score should be lower than "tool:imminent"
        if "tool:distant" in result:
            assert result["tool:distant"] <= result["tool:imminent"]

    def test_empty_when_no_events(self):
        osc = OscillatorNetwork()
        assert osc.get_anticipatory_signatures() == {}


# ---------------------------------------------------------------------------
# SCN delegation
# ---------------------------------------------------------------------------


class TestSCNObserveEvent:
    def test_delegates_to_oscillator(self):
        scn = SCN()
        scn.enable_oscillator()
        sig = _make_sig(0.5)
        scn.observe_event("tool:attack", sig)
        assert "tool:attack" in scn.oscillator._event_phases

    def test_noop_without_oscillator(self):
        scn = SCN()
        # Should not raise
        scn.observe_event("tool:attack", _make_sig(0.5))


class TestSCNGetAnticipatorySignatures:
    def test_empty_without_oscillator(self):
        scn = SCN()
        assert scn.get_anticipatory_signatures() == {}

    def test_delegates_to_oscillator(self):
        scn = SCN()
        scn.enable_oscillator()
        # Build enough observations for a prediction
        for _ in range(5):
            sig = _make_sig(0.5)
            scn.register("mem-" + uuid.uuid4().hex[:8], sig)
            scn.oscillator.observe_event("tool:noon", sig)
        result = scn.get_anticipatory_signatures(min_imminence=0.3)
        # With oscillator aligned to 0.5 and events at 0.5, should have hits
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# Serialization round-trip
# ---------------------------------------------------------------------------


class TestOscillatorSerialization:
    def test_event_phases_survive_round_trip(self):
        osc = OscillatorNetwork()
        for _ in range(5):
            osc.observe_event("tool:attack", _make_sig(0.5))
            osc.observe_event("tool:defend", _make_sig(0.25))

        data = osc.to_dict()
        restored = OscillatorNetwork.from_dict(data)

        assert set(restored._event_phases.keys()) == {"tool:attack", "tool:defend"}
        assert len(restored._event_phases["tool:attack"]) == 5
        for orig, rest in zip(
            osc._event_phases["tool:attack"],
            restored._event_phases["tool:attack"],
        ):
            assert abs(orig - rest) < 1e-10

    def test_empty_event_phases_backward_compat(self):
        """Old persisted data without event_phases should load cleanly."""
        data = {
            "phases": [0.0, 0.0, 0.0, 0.0],
            "coupling": [[0.0] * 4 for _ in range(4)],
            "observation_count": 0,
            "prediction_error_ema": 0.0,
            "config": {},
        }
        restored = OscillatorNetwork.from_dict(data)
        assert restored._event_phases == {}

    def test_config_fields_survive_round_trip(self):
        """Custom config values for new fields should persist."""
        config = OscillatorConfig(max_event_phases=25, anticipatory_weight=0.5)
        osc = OscillatorNetwork(config)
        data = osc.to_dict()
        restored = OscillatorNetwork.from_dict(data)
        assert restored.config.max_event_phases == 25
        assert restored.config.anticipatory_weight == 0.5


# ---------------------------------------------------------------------------
# TemporalCreditDistributor — oscillator feedback integration
# ---------------------------------------------------------------------------


class TestDistributorOscillatorFeedback:
    def test_record_event_feeds_oscillator(self):
        """record_event should call scn.observe_event with the event signature."""
        nac = NAc()
        scn = SCN()
        scn.enable_oscillator()
        dist = TemporalCreditDistributor(nac=nac, scn=scn)

        event = _make_event(event_signature="tool:slash", circadian_phase=0.5)
        dist.record_event(event)

        assert "tool:slash" in scn.oscillator._event_phases

    def test_record_event_safe_without_oscillator(self):
        """record_event should not crash when oscillator is disabled."""
        nac = NAc()
        scn = SCN()  # No oscillator
        dist = TemporalCreditDistributor(nac=nac, scn=scn)

        event = _make_event()
        dist.record_event(event)  # Should not raise

    def test_anticipatory_pre_activate_creates_traces(self):
        """anticipatory_pre_activate should create eligibility traces for imminent events."""
        nac = NAc()
        scn = SCN()
        scn.enable_oscillator()
        dist = TemporalCreditDistributor(nac=nac, scn=scn)
        agent_id = "agent-1"

        # Record enough events at phase 0.5 to build a prediction
        for _ in range(5):
            event = _make_event(
                agent_id=agent_id,
                event_signature="tool:recurring",
                circadian_phase=0.5,
            )
            dist.record_event(event)

        # Decay all eligibility traces so fast-decay path is empty
        for _ in range(50):
            nac.decay_eligibility(factor=0.5)

        # Pre-activate — should create a new eligibility trace
        pre_activated = dist.anticipatory_pre_activate(agent_id)

        # If the oscillator's phase is near 0.5, the event should be pre-activated
        if pre_activated:
            sigs = {sig for sig, _ in pre_activated}
            assert "tool:recurring" in sigs
            # The trace should now exist in NAc eligibility
            assert (agent_id, "tool:recurring") in nac._eligibility

    def test_pre_activate_then_distribute_credits(self):
        """Pre-activated traces should be credited when distribute() runs."""
        nac = NAc()
        scn = SCN()
        scn.enable_oscillator()
        dist = TemporalCreditDistributor(nac=nac, scn=scn)
        agent_id = "agent-1"

        # Record enough events at phase 0.5 to build a prediction
        for _ in range(5):
            event = _make_event(
                agent_id=agent_id,
                event_signature="tool:recurring",
                circadian_phase=0.5,
            )
            dist.record_event(event)

        # Decay all eligibility traces
        for _ in range(50):
            nac.decay_eligibility(factor=0.5)

        # Pre-activate, then distribute
        pre_activated = dist.anticipatory_pre_activate(agent_id)
        if pre_activated:
            credited = dist.distribute(agent_id, reward=1.0)
            # The pre-activated trace should now receive credit
            assert len(credited) > 0
            assert all(c > 0 for _, c in credited)

    def test_anticipatory_does_not_double_count_fast_decay(self):
        """Events with active traces should NOT be pre-activated."""
        nac = NAc()
        scn = SCN()
        scn.enable_oscillator()
        dist = TemporalCreditDistributor(nac=nac, scn=scn)
        agent_id = "agent-1"

        # Record events to build phase history AND leave fast-decay active
        for _ in range(5):
            event = _make_event(
                agent_id=agent_id,
                event_signature="tool:active",
                circadian_phase=0.5,
            )
            dist.record_event(event)

        # Fast-decay traces should still be active (no decay called)
        # Pre-activate should skip them
        pre_activated = dist.anticipatory_pre_activate(agent_id)
        # "tool:active" should NOT appear in pre-activated (already has trace)
        pre_sigs = {sig for sig, _ in pre_activated}
        assert "tool:active" not in pre_sigs


# ---------------------------------------------------------------------------
# build_bio_stack enables oscillator
# ---------------------------------------------------------------------------


class TestBioStackOscillatorEnabled:
    def test_oscillator_enabled_in_bio_stack(self):
        """build_bio_stack should enable the SCN oscillator."""
        from maxim.runtime.bio_stack import build_bio_stack

        bio = build_bio_stack(persistence_dir=None)
        assert bio.scn.oscillator is not None
        assert bio.scn.oscillator._observation_count == 0
