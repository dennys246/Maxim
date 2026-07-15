"""Tests for build_default_network Layer 5 upgrade (Wave 2, biosystem_unification).

Regression guards for:
- nac is required keyword-only (Gap A: Layer 4 → 5 upgrade)
- pain_bus injection (Gap B: split-subscriber-ownership fix)
- maxim=None no longer bails early (headless/sim mode)
- narrow exception handling (import failure → None, not broad swallow)

See ``docs/plans/archive/default_network_unification.md`` for the audit.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from maxim.runtime.bootstrap import build_default_network


class TestBuildDefaultNetworkLayerUpgrade:
    """Gap A: nac is now required keyword-only. Forgetting it is a TypeError."""

    def test_missing_nac_raises_type_error(self):
        """Forgetting nac= is a TypeError, not a silent None."""
        with pytest.raises(TypeError):
            build_default_network()  # type: ignore[call-arg]

    def test_nac_none_explicit_opt_out(self):
        """Passing nac=None is a legitimate explicit opt-out."""
        dn = build_default_network(nac=None)
        # DN constructs but pain circuit is skipped (nac is None)
        if dn is not None:
            assert dn._pain_bridge is None
            assert dn._pain_bus is None

    def test_nac_positional_rejected(self):
        """nac is keyword-only — positional args are a TypeError."""
        with pytest.raises(TypeError):
            build_default_network(None)  # type: ignore[misc]


class TestBuildDefaultNetworkHeadless:
    """maxim=None no longer bails early — DN constructs for headless/sim."""

    def test_maxim_none_returns_dn_not_none(self):
        """Headless mode: DN still provides pain detection + novelty tracking."""
        nac = MagicMock()
        dn = build_default_network(nac=nac, maxim=None)
        # Should NOT return None just because maxim is None.
        # May return None if default_network package isn't installed,
        # but on this codebase it should be available.
        assert dn is not None

    def test_maxim_none_pain_circuit_still_wires(self):
        """With nac provided and maxim=None, pain detection still wires."""
        nac = MagicMock()
        dn = build_default_network(nac=nac, maxim=None)
        if dn is not None:
            # Pain circuit should be active (nac provided, pain_detection_enabled=True by default)
            assert dn._pain_bus is not None


class TestBuildDefaultNetworkPainBusInjection:
    """Gap B: injected PainBus replaces DN-internal construction."""

    def test_injected_pain_bus_used(self):
        """When pain_bus= is provided, DN uses it instead of constructing its own."""
        from maxim.proprioception.pain_bus import PainBus

        nac = MagicMock()
        injected_bus = PainBus(_allow_raw=True)
        dn = build_default_network(nac=nac, pain_bus=injected_bus)
        if dn is not None:
            assert dn._pain_bus is injected_bus

    def test_no_pain_bus_constructs_internal(self):
        """When pain_bus=None, DN constructs its own (backward compat)."""
        from maxim.proprioception.pain_bus import PainBus

        nac = MagicMock()
        dn = build_default_network(nac=nac, pain_bus=None)
        if dn is not None and dn._pain_bus is not None:
            assert isinstance(dn._pain_bus, PainBus)

    def test_injected_bus_with_build_pain_bus(self):
        """Full integration: build_pain_bus → inject into build_default_network.

        This is the production shape that closes Gap B — the bus arrives
        with hippocampus + NAc subscribers already wired, so DN doesn't
        need external hippocampus subscription at agentic_runtime.py:719.
        """
        from maxim.proprioception.pain_bus import PainBus, build_pain_bus

        hippo = MagicMock()
        hippo.capture = MagicMock()
        nac = MagicMock()

        # Build a pain bus with both subscribers
        bus = build_pain_bus(hippocampus=hippo, nac=nac)
        assert isinstance(bus, PainBus)

        # Inject into DN
        dn = build_default_network(nac=nac, pain_bus=bus)
        if dn is not None:
            assert dn._pain_bus is bus
            # DN's pain_bus is the SAME object — subscribers wired by
            # build_pain_bus are inherited, not duplicated.
            stats = bus.get_stats()
            # At minimum: 2 from build_pain_bus + PainCircuitBridge._on_pain
            assert stats["direct_pain_subscribers"] >= 2


class TestBuildDefaultNetworkConfig:
    """Config precedence: explicit config > config_path > default."""

    def test_explicit_config_takes_precedence(self):
        """When config= is passed, config_path is ignored."""
        from maxim.default_network import DefaultNetworkConfig

        custom_config = DefaultNetworkConfig(
            enabled=True,
            publish_actions=False,
            fear_gate_enabled=False,
        )
        nac = MagicMock()
        dn = build_default_network(nac=nac, config=custom_config)
        if dn is not None:
            assert dn._config.publish_actions is False
            assert dn._config.fear_gate_enabled is False


class TestBuildDefaultNetworkImportFailure:
    """Narrow exception handling: import failure → None + warning."""

    def test_import_failure_returns_none(self):
        """If default_network package is missing, returns None gracefully."""
        with patch.dict("sys.modules", {"maxim.default_network": None}):
            dn = build_default_network(nac=MagicMock())
            assert dn is None


class TestPainCircuitBridgeSubscriberTripwire:
    """Tripwire for the latent PainCircuitBridge × NAc subscriber double-attribution risk.

    Same shape as ``test_pain_bus.py::test_subscriber_does_not_link_pending_tool_event``
    (the ToolPainBridge tripwire from Wave 1). With an injected PainBus from
    ``build_pain_bus``, the bus has ``create_pain_nac_subscriber`` AND
    ``PainCircuitBridge._on_pain`` both subscribed. The subscriber's
    ``record_outcome_full`` walks ``_pending_events`` — if it finds the
    bridge's pending movement event, it double-counts.

    **Today** the pending movement event context is ``{"reason": ..., "angle": ...}``
    (from ``movement.py::record_action_start``) while the pain signal context
    is ``{"source": "embodiment", "entity": ..., ...}``. Zero key overlap →
    ``_context_similarity = 0/N = 0.0 < 0.5 threshold``. No double-counting.

    **If this test fails:** someone enriched ``record_action_start``'s context
    to include keys that overlap with pain signal context. Open
    ``docs/plans/deferred/pain_bus_bridge_subscriber_unification.md`` — the deeper
    bridge-aware subscriber fix is now needed for BOTH the ToolPainBridge
    and PainCircuitBridge paths.
    """

    def test_subscriber_does_not_link_pending_movement_event(self):
        """create_pain_nac_subscriber does NOT link PainCircuitBridge's pending events."""
        import time

        from maxim.decisions.nac import NAc, NACConfig
        from maxim.proprioception.pain import PainSignal, PainType
        from maxim.proprioception.pain_bus import build_pain_bus

        nac = NAc(NACConfig(temporal_window_seconds=60.0))

        # Simulate PainCircuitBridge.record_action_start: pending movement
        # event with the context shape from movement.py ({"reason", "angle"}).
        nac.record_event(
            event_type="movement",
            event_signature="look_at:dy=90:dp=10",
            context={"reason": "orienting_response", "angle": 90.0},
        )
        assert len(nac._pending_events) == 1

        # Build a bus with NAc subscriber (the production shape post-Wave-2).
        bus = build_pain_bus(hippocampus=None, nac=nac)

        # Publish a rich-context pain signal (what body._publish_pain fires).
        signal = PainSignal(
            pain_type=PainType.EXTERNAL_SIGNAL,
            intensity=0.7,
            timestamp=time.time(),
            context={
                "source": "embodiment",
                "entity": "body.arm.rusty_sword",
                "entity_type": "weapon",
                "failure_mode": "shatter",
                "composes": [],
                "sensor_readings": {"durability": 0.05},
            },
        )
        bus.publish(signal)

        # NO link should form for the pending movement event because the
        # context-similarity mismatch ({"reason","angle"} vs 7-key body
        # context) yields 0/2 = 0.0 < 0.5 threshold.
        assert "look_at:dy=90:dp=10" not in nac._links or len(nac._links.get("look_at:dy=90:dp=10", [])) == 0
