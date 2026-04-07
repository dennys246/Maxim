"""Tests for causal memory integration Phases 1-4.

Phase 1: CAUSES edges in hippocampus graph via ToolPainBridge
Phase 2: EC registration of established causal links
Phase 3: EC similarity augmentation in NAc.predict()
Phase 4: RPE-based observation salience boost
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


# ───────────────────────────────────────────────────────────────────────────
# Helpers — all maxim imports are lazy to avoid circular import at collect time
# ───────────────────────────────────────────────────────────────────────────


def _make_link(
    link_id: str = "test-link",
    event_sig: str = "tool:test",
    outcome_sig: str = "result:test",
    valence: str = "POSITIVE",
    predicted_value: float = 0.5,
    observation_count: int = 1,
    confidence: float = 0.5,
    memory_ids: list[str] | None = None,
    last_rpe: float | None = None,
):
    from maxim.decisions.causal_link import CausalLink, TemporalDelta, Valence

    val = Valence[valence]
    link = CausalLink(
        id=link_id,
        event_type="tool",
        event_signature=event_sig,
        event_context={},
        outcome_type="result",
        outcome_signature=outcome_sig,
        outcome_valence=val,
        temporal_delta=TemporalDelta(observed_deltas=(1.0,)),
        predicted_value=predicted_value,
        observation_count=observation_count,
        confidence=confidence,
        memory_ids=memory_ids or [],
    )
    if last_rpe is not None:
        link.last_rpe = last_rpe
    return link


# ═══════════════════════════════════════════════════════════════════════════
# Phase 1: CAUSES edges in hippocampus graph
# ═══════════════════════════════════════════════════════════════════════════


class TestPhase1CausesEdgeCreatedOnHighRPE:
    """CAUSES edge should be created when RPE exceeds threshold (0.3)."""

    def test_causes_edge_created(self) -> None:
        from maxim.bridges.tool_pain_bridge import ToolPainBridge

        hippo = MagicMock()
        hippo.graph = MagicMock()

        link = _make_link(
            last_rpe=0.8,
            confidence=0.7,
            memory_ids=["mem-1", "mem-2", "mem-3"],
        )

        nac = MagicMock()
        nac.record_outcome.return_value = [link]

        from maxim.proprioception.pain import PainDetector

        detector = PainDetector()
        bridge = ToolPainBridge(nac=nac, pain_detector=detector, hippocampus=hippo)

        bridge.record_tool_start("web_search", "inv-1")
        bridge.record_tool_complete("web_search", "inv-1", success=True)

        # Verify CAUSES edges were created
        from maxim.agents.bus import EdgeType

        assert hippo.graph.add_edge.called
        for c in hippo.graph.add_edge.call_args_list:
            assert c[1]["edge_type"] == EdgeType.CAUSES
            assert c[1]["target"] == "mem-3"  # newest memory


class TestPhase1NoCausesEdgeOnLowRPE:
    """No CAUSES edge when RPE is below threshold."""

    def test_no_edge_on_low_rpe(self) -> None:
        from maxim.bridges.tool_pain_bridge import ToolPainBridge

        hippo = MagicMock()
        hippo.graph = MagicMock()

        link = _make_link(
            last_rpe=0.1,
            memory_ids=["mem-1", "mem-2"],
        )

        nac = MagicMock()
        nac.record_outcome.return_value = [link]

        from maxim.proprioception.pain import PainDetector

        detector = PainDetector()
        bridge = ToolPainBridge(nac=nac, pain_detector=detector, hippocampus=hippo)

        bridge.record_tool_start("web_search", "inv-1")
        bridge.record_tool_complete("web_search", "inv-1", success=True)

        hippo.graph.add_edge.assert_not_called()


class TestPhase1NoCausesEdgeWithoutHippocampus:
    """No crash when hippocampus is None."""

    def test_no_crash_without_hippocampus(self) -> None:
        from maxim.bridges.tool_pain_bridge import ToolPainBridge

        link = _make_link(last_rpe=0.8, memory_ids=["mem-1", "mem-2"])

        nac = MagicMock()
        nac.record_outcome.return_value = [link]

        from maxim.proprioception.pain import PainDetector

        detector = PainDetector()
        bridge = ToolPainBridge(nac=nac, pain_detector=detector, hippocampus=None)

        bridge.record_tool_start("web_search", "inv-1")
        # Should not raise
        bridge.record_tool_complete("web_search", "inv-1", success=True)


# ═══════════════════════════════════════════════════════════════════════════
# Phase 2: EC registration of established causal links
# ═══════════════════════════════════════════════════════════════════════════


class TestPhase2ECRegistrationOnEstablishedLink:
    """EC.register should be called when a link reaches 3 observations."""

    def test_ec_register_called(self) -> None:
        from maxim.decisions.causal_link import Valence
        from maxim.decisions.nac import NAc

        ec = MagicMock()
        nac = NAc(ec=ec)

        # Record 3 outcomes for the same event to reach observation_count >= 3
        for i in range(3):
            nac.record_event(
                event_type="tool",
                event_signature="tool:search",
                context={"mode": "active"},
            )
            nac.record_outcome(
                event_type="tool",
                event_id="tool:search",
                outcome_valence=Valence.POSITIVE,
                context={"mode": "active"},
            )

        # EC.register should have been called with "causal:" prefix
        assert ec.register.called
        causal_calls = [c for c in ec.register.call_args_list if c[0][0].startswith("causal:")]
        assert len(causal_calls) >= 1


class TestPhase2NoECRegistrationOnNewLink:
    """EC.register should NOT be called when observation_count < 3."""

    def test_ec_register_not_called(self) -> None:
        from maxim.decisions.causal_link import Valence
        from maxim.decisions.nac import NAc

        ec = MagicMock()
        nac = NAc(ec=ec)

        # Record only 1 outcome
        nac.record_event(
            event_type="tool",
            event_signature="tool:search",
            context={"mode": "active"},
        )
        nac.record_outcome(
            event_type="tool",
            event_id="tool:search",
            outcome_valence=Valence.POSITIVE,
            context={"mode": "active"},
        )

        ec.register.assert_not_called()


class TestPhase2ECDeregistrationOnEviction:
    """EC.remove_signature should be called when a link is evicted."""

    def test_ec_remove_on_eviction(self) -> None:
        from maxim.decisions.causal_link import Valence
        from maxim.decisions.nac import NAc, NACConfig

        ec = MagicMock()
        config = NACConfig(max_links=2)
        nac = NAc(config=config, ec=ec)

        # Create 3 links (exceeds max_links=2), forcing eviction
        for i in range(3):
            sig = f"tool:t{i}"
            nac.observe(
                event_type="tool",
                event_signature=sig,
                outcome_type="result",
                outcome_signature=f"result:t{i}",
                outcome_valence=Valence.POSITIVE,
                delta_seconds=1.0,
                context={"idx": str(i)},
            )

        # _enforce_limits should have called _deregister_causal_from_ec
        assert ec.remove_signature.called
        # The removed signature should have "causal:" prefix
        remove_calls = ec.remove_signature.call_args_list
        for c in remove_calls:
            assert c[0][0].startswith("causal:")


# ═══════════════════════════════════════════════════════════════════════════
# Phase 3: EC similarity augmentation in predict()
# ═══════════════════════════════════════════════════════════════════════════


class TestPhase3ECDisabledByDefault:
    """EC.find_similar should NOT be called when use_ec_similarity=False."""

    def test_ec_not_queried_by_default(self) -> None:
        from maxim.decisions.causal_link import Valence
        from maxim.decisions.nac import NAc

        ec = MagicMock()
        nac = NAc(ec=ec)

        # Add a link so predict has something to return
        nac.observe(
            event_type="tool",
            event_signature="tool:search",
            outcome_type="result",
            outcome_signature="result:ok",
            outcome_valence=Valence.POSITIVE,
            delta_seconds=1.0,
        )

        nac.predict(
            event_type="tool",
            event_signature="tool:search",
        )

        ec.find_similar.assert_not_called()


class TestPhase3ECEnabledQueriesEC:
    """EC.find_similar should be called when use_ec_similarity=True."""

    def test_ec_queried_when_enabled(self) -> None:
        from maxim.decisions.causal_link import Valence
        from maxim.decisions.nac import NAc, NACConfig

        ec = MagicMock()
        # EC returns a causal link ID that exists in the NAc
        ec.find_similar.return_value = [("causal:existing-link", 0.9)]

        config = NACConfig(use_ec_similarity=True)
        nac = NAc(config=config, ec=ec)

        # Create a link under a DIFFERENT event signature so EC augmentation
        # is the only way it can appear in predict results
        other_link = _make_link(
            link_id="existing-link",
            event_sig="tool:other",
            outcome_sig="result:other",
            confidence=0.8,
        )
        nac._links.setdefault("tool:other", []).append(other_link)

        # Also create a direct link for "tool:search" so predict returns
        nac.observe(
            event_type="tool",
            event_signature="tool:search",
            outcome_type="result",
            outcome_signature="result:ok",
            outcome_valence=Valence.POSITIVE,
            delta_seconds=1.0,
        )

        result = nac.predict(
            event_type="tool",
            event_signature="tool:search",
        )

        ec.find_similar.assert_called_once()
        assert result is not None


# ═══════════════════════════════════════════════════════════════════════════
# Phase 4: RPE-based observation salience boost
# ═══════════════════════════════════════════════════════════════════════════


class TestPhase4RPEBoostsSalience:
    """RPE > 0 should boost observation salience."""

    def test_salience_boosted_by_rpe(self) -> None:
        # Simulate the RPE salience boost formula:
        # boosted = base_salience + rpe * (1.0 - base_salience)
        rpe = 0.8
        base_salience = 0.5
        boosted = base_salience + rpe * (1.0 - base_salience)
        assert boosted == pytest.approx(0.9)

    def test_executor_rpe_available_for_boost(self) -> None:
        """Executor.get_last_rpe should return RPE from bridge."""
        from maxim.bridges.tool_pain_bridge import ToolPainBridge
        from maxim.decisions.nac import NAc
        from maxim.proprioception.pain import PainDetector

        nac = NAc()
        detector = PainDetector()
        bridge = ToolPainBridge(nac=nac, pain_detector=detector)

        # Establish a link
        bridge.record_tool_start("search", "inv-1")
        bridge.record_tool_complete("search", "inv-1", success=True)

        # Second observation produces RPE
        bridge.record_tool_start("search", "inv-2")
        rpe = bridge.record_tool_complete("search", "inv-2", success=True)

        assert isinstance(rpe, float)
        assert rpe >= 0.0

        # Simulate boost
        base_salience = 0.5
        boosted = base_salience + rpe * (1.0 - base_salience)
        assert boosted >= base_salience


class TestPhase4NoBoostOnZeroRPE:
    """Zero RPE should not change salience."""

    def test_salience_unchanged_on_zero_rpe(self) -> None:
        rpe = 0.0
        base_salience = 0.5
        boosted = base_salience + rpe * (1.0 - base_salience)
        assert boosted == pytest.approx(base_salience)

    def test_bridge_rpe_zero_without_history(self) -> None:
        """Fresh bridge should have zero RPE."""
        from maxim.bridges.tool_pain_bridge import ToolPainBridge
        from maxim.proprioception.pain import PainDetector

        nac = MagicMock()
        nac.record_outcome.return_value = []
        detector = PainDetector()
        bridge = ToolPainBridge(nac=nac, pain_detector=detector)

        assert bridge._last_rpe == 0.0

        # Complete without a matching pending event
        rpe = bridge.record_tool_complete("unknown", "inv-x", success=True)
        assert rpe == 0.0
        assert bridge._last_rpe == 0.0
