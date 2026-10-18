"""Guards for the NAc credit-path fixes (A2).

1. A tool outcome feeds the SCN's coupled oscillator ONCE. The bridge's
   persistent signature-keyed bin entry and the distributor's per-event entry
   both stay (they serve different readers); only the second observation goes.
2. ``_cluster_reward_source`` never outlives its ``_cluster_reward_bias`` entry
   (per-tick decay prune, wall-clock decay prune, load of an orphaned source).
3. ``MAXIM_NAC_REWARD_BIAS_DISABLED`` keeps its documented ``_reward_bias``-only
   scope: goal bias is pinned as NOT gated (a deliberate decision, not drift).
4. ``_event_outcome_welford`` is bounded by the links: when ``_enforce_limits``
   evicts the last link of an event signature, that signature's Welford state
   (and its empty ``_links`` slot) goes with it.
"""

from __future__ import annotations

import logging
import pytest

from maxim.bridges.tool_pain_bridge import ToolPainBridge
from maxim.decisions.causal_link import Valence
from maxim.decisions.nac import NAc, NACConfig
from maxim.decisions.temporal_credit import TemporalCreditDistributor
from maxim.time.scn import SCN


# ---------------------------------------------------------------------------
# 1. One SCN registration per tool outcome
# ---------------------------------------------------------------------------


class TestOscillatorObservedOncePerOutcome:
    """The persistent signature-keyed SCN entry stays; the oscillator sees each outcome ONCE."""

    def _bridge(self, *, with_distributor: bool) -> tuple[ToolPainBridge, SCN, TemporalCreditDistributor | None]:
        nac = NAc(NACConfig())
        scn = SCN()
        scn.enable_oscillator()
        distributor = TemporalCreditDistributor(nac=nac, scn=scn) if with_distributor else None
        bridge = ToolPainBridge(nac=nac, scn=scn, distributor=distributor, agent_id="a1")
        return bridge, scn, distributor

    def test_success_observes_oscillator_once_with_distributor(self) -> None:
        bridge, scn, _ = self._bridge(with_distributor=True)
        bridge.record_tool_start("web_search", "inv-1")
        bridge.record_tool_complete("web_search", "inv-1", success=True)
        assert scn._oscillator._observation_count == 1
        # Both registrations still exist: the persistent signature entry and the per-event id.
        assert scn.get_signature("tool:web_search") is not None
        assert len(scn._signatures) == 2

    def test_embodiment_failure_observes_oscillator_once_with_distributor(self) -> None:
        bridge, scn, _ = self._bridge(with_distributor=True)
        bridge.record_tool_start("fire_breath", "inv-2")
        bridge.record_tool_embodiment_failure(
            "fire_breath", "inv-2", failures=[{"name": "burn", "pain": 0.8, "entity": "dragon"}]
        )
        assert scn._oscillator._observation_count == 1

    def test_without_distributor_direct_entry_observes_oscillator(self) -> None:
        bridge, scn, _ = self._bridge(with_distributor=False)
        bridge.record_tool_start("web_search", "inv-3")
        bridge.record_tool_complete("web_search", "inv-3", success=True)
        assert scn._oscillator._observation_count == 1
        assert set(scn._signatures) == {"tool:web_search"}

    def test_signature_entry_survives_cleanup_session_and_save_load(self, tmp_path) -> None:
        bridge, scn, distributor = self._bridge(with_distributor=True)
        bridge.record_tool_start("web_search", "inv-4")
        bridge.record_tool_complete("web_search", "inv-4", success=True)
        assert distributor is not None
        distributor.cleanup_session()
        # The per-event id is session-scoped; the signature entry is not.
        assert set(scn._signatures) == {"tool:web_search"}
        path = tmp_path / "scn.json"
        scn.save(str(path))
        reloaded = SCN()
        reloaded.load(str(path))
        assert reloaded.get_signature("tool:web_search") is not None
        assert "tool:web_search" in reloaded.query_hour(scn.get_signature("tool:web_search").to_bins()[0])


# ---------------------------------------------------------------------------
# 2. _cluster_reward_source never outlives its bias
# ---------------------------------------------------------------------------


KEY = ("a1", "c1", "tool:x")


class TestClusterRewardSourceFollowsBias:
    def test_tick_decay_prune_drops_source(self) -> None:
        """The reviewer's probe: drive_relief credit -> decay to zero -> a fresh
        tool_success credit must report 'tool_success', not 'mixed'."""
        nac = NAc(NACConfig())
        nac.update_cluster_reward(*KEY, reward=0.01, source="drive_relief")
        for _ in range(10_000):
            if KEY not in nac._cluster_reward_bias:
                break
            nac.decay_cluster_reward_biases()
        assert KEY not in nac._cluster_reward_bias
        assert KEY not in nac._cluster_reward_source
        nac.update_cluster_reward(*KEY, reward=1.0, source="tool_success")
        assert nac.get_cluster_reward_sources(agent_id="a1") == {"tool:x": "tool_success"}

    def test_tick_decay_keeps_source_of_surviving_bias(self) -> None:
        nac = NAc(NACConfig())
        nac.update_cluster_reward(*KEY, reward=5.0, source="drive_relief")
        nac.decay_cluster_reward_biases()
        assert nac._cluster_reward_source[KEY] == "drive_relief"

    def test_wall_clock_decay_prune_drops_source(self) -> None:
        nac = NAc(NACConfig())
        nac.update_cluster_reward(*KEY, reward=0.05, source="drive_relief")
        nac.apply_wall_clock_decay(elapsed_s=10 * 365 * 86400.0)
        assert KEY not in nac._cluster_reward_bias
        assert KEY not in nac._cluster_reward_source

    def test_load_drops_orphaned_source(self, caplog: pytest.LogCaptureFixture) -> None:
        nac = NAc(NACConfig())
        nac.update_cluster_reward(*KEY, reward=1.0, source="drive_relief")
        state = nac.dump()
        state["cluster_reward_source"]["a1\x1fc_gone\x1ftool:y"] = "operant"
        fresh = NAc(NACConfig())
        with caplog.at_level(logging.WARNING):
            fresh.load_state(state)
        assert fresh._cluster_reward_source == {KEY: "drive_relief"}
        assert "orphan" in caplog.text


# ---------------------------------------------------------------------------
# 3. MAXIM_NAC_REWARD_BIAS_DISABLED scope: _reward_bias only (goal bias NOT gated)
# ---------------------------------------------------------------------------


def test_reward_bias_ablation_leaves_goal_bias_by_documented_scope(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAXIM_NAC_REWARD_BIAS_DISABLED", "1")
    nac = NAc(NACConfig())
    nac.credit_node("a1", "node", 1.0)
    nac.credit_goal("goal", 1.0)
    assert nac.reward_bias("a1", "node") == 0.0
    assert nac.get_goal_reward_bias("goal") > 0.0


# ---------------------------------------------------------------------------
# 4. _event_outcome_welford bounded by the links
# ---------------------------------------------------------------------------


def _booked(nac: NAc, sig: str, valence: Valence) -> None:
    event_id = nac.record_event(event_type="tool", event_signature=sig, context={"agent_id": "a1"})
    links = nac.record_outcome_full(
        outcome_type="result",
        outcome_signature=f"{sig}:{valence.value}",
        outcome_valence=valence,
        context={"agent_id": "a1"},
        attributed_event_id=event_id,
    )
    assert links


class TestWelfordBoundedByLinks:
    def test_evicting_last_link_drops_welford_and_empty_slot(self) -> None:
        nac = NAc(NACConfig(max_links=1))
        _booked(nac, "tool:old", Valence.POSITIVE)
        assert ("a1", "tool:old") in nac._event_outcome_welford
        # Make the old link the eviction candidate, then book a second signature.
        for link in nac._links["tool:old"]:
            link.confidence = 0.0
        _booked(nac, "tool:new", Valence.POSITIVE)
        assert "tool:old" not in nac._links
        assert ("a1", "tool:old") not in nac._event_outcome_welford
        assert ("a1", "tool:new") in nac._event_outcome_welford

    def test_welford_with_surviving_links_kept(self) -> None:
        nac = NAc(NACConfig(max_links=10))
        _booked(nac, "tool:a", Valence.POSITIVE)
        _booked(nac, "tool:b", Valence.NEGATIVE)
        assert {k[1] for k in nac._event_outcome_welford} == {"tool:a", "tool:b"}
