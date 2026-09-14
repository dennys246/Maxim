"""Wire 4 (Exp 58, 1.3 Phase 1): situation-keyed fear — pain→cluster negative valence.

Covers the mechanism the four-lens design review shaped
(docs/experiments/exp58_survival_wants_prereg.md §Mechanism; rationale in
docs/experiments/rationale/exp58_survival_wants/):

- the NAc fear store (allowlist, clamp, per-agent keying, persistence, wall-decay class,
  NO per-tick decay — pinned with an anti-vacuity arm);
- the auto-wired PainBus subscriber booking fear on the NOTED world cluster only;
- the READ through the real consumer: an anticipatory threat need that makes
  ``recommend_action`` select the ``flee`` affordance (the triple-confirmed
  DO-NOT-BUILD was this read path being dead on the minecraft body).
"""

from __future__ import annotations

import time

import pytest

from maxim.decisions.nac import NAc, NACConfig
from maxim.proprioception.pain import PainSignal, PainType
from maxim.proprioception.pain_bus import build_pain_bus

AGENT = "fear_test_agent"
DARK = "cluster-dark-uuid"
LIT = "cluster-lit-uuid"

MINECRAFT_TOOLS = [
    "minecraft_player_move_to",
    "minecraft_player_turn",
    "minecraft_player_mine_block",
    "minecraft_player_place_block",
    "minecraft_player_eat",
    "minecraft_player_attack_nearest",
    "minecraft_player_flee",
]


def _nac(**overrides) -> NAc:
    return NAc(NACConfig(**overrides))


def _health_pain(intensity: float = 1.0, **ctx) -> PainSignal:
    """A drive:health PainSignal shaped like body._publish_drive_pain's rich context."""
    base = {
        "source": "drive:health",
        "entity": "minecraft_player",
        "entity_type": "body",
        "entity_name": "minecraft_player",
        "failure_mode": "drive:health",
        "sensor_readings": {"health": 8.0},
        "entity_path": "minecraft_player",
        "agent_id": AGENT,
    }
    base.update(ctx)
    return PainSignal(
        pain_type=PainType.EXTERNAL_SIGNAL,
        intensity=intensity,
        timestamp=time.time(),
        context=base,
    )


class TestFearStore:
    def test_health_pain_books_negative_and_clamps_at_cap(self):
        nac = _nac(cluster_fear_alpha=0.5, max_cluster_fear=1.0)
        for _ in range(5):  # 5 × 0.5 = 2.5 → clamps at -1.0
            nac.record_cluster_fear(AGENT, DARK, "drive:health", 1.0)
        assert nac.cluster_fear(AGENT, DARK) == -1.0
        assert nac.cluster_fear(AGENT, LIT) == 0.0

    def test_allowlist_rejects_hunger_pain_silently(self):
        """W-5: hunger pain must not write fear (lit/dining contamination)."""
        nac = _nac()
        nac.record_cluster_fear(AGENT, DARK, "drive:food", 1.0)
        assert nac.cluster_fear(AGENT, DARK) == 0.0

    def test_empty_agent_raises_empty_cluster_noops(self):
        nac = _nac()
        with pytest.raises(ValueError):
            nac.record_cluster_fear("", DARK, "drive:health", 1.0)
        nac.record_cluster_fear(AGENT, "", "drive:health", 1.0)  # no situation → no-op
        assert nac.cluster_fear(AGENT, "") == 0.0

    def test_per_agent_keying(self):
        nac = _nac()
        nac.record_cluster_fear(AGENT, DARK, "drive:health", 1.0)
        assert nac.cluster_fear("other_agent", DARK) == 0.0

    def test_note_active_clusters_stash_and_clear(self):
        nac = _nac()
        nac.note_active_clusters(AGENT, {"world": DARK, "interoception": "i-1"})
        assert nac.active_clusters(AGENT) == {"world": DARK, "interoception": "i-1"}
        nac.note_active_clusters(AGENT, None)
        assert nac.active_clusters(AGENT) == {}

    def test_persistence_round_trip_and_positive_reclamp(self, tmp_path):
        nac = _nac()
        nac.record_cluster_fear(AGENT, DARK, "drive:health", 0.8)
        value = nac.cluster_fear(AGENT, DARK)
        assert value < 0.0
        path = str(tmp_path / "nac.json")
        nac.save(path)
        loaded = _nac()
        loaded.load(path)
        assert loaded.cluster_fear(AGENT, DARK) == pytest.approx(value)
        # A hand-edited positive "fear" must reclamp to 0 on load.
        import json

        state = json.loads(open(path).read())
        key = next(iter(state["cluster_fear"]))
        state["cluster_fear"][key] = 0.7
        open(path, "w").write(json.dumps(state))
        reloaded = _nac()
        reloaded.load(path)
        assert reloaded.cluster_fear(AGENT, DARK) == 0.0

    def test_no_per_tick_decay_but_slow_wall_decay(self):
        """Extinction is re-learning, not a timer: the per-tick decayers must
        not touch fear (anti-vacuity: they DO run and DO decay a sibling),
        while apply_wall_decay decays it in the slow class."""
        nac = _nac()
        nac.record_cluster_fear(AGENT, DARK, "drive:health", 1.0)
        before = nac.cluster_fear(AGENT, DARK)
        # Sibling store that per-tick decay DOES touch (anti-vacuity arm):
        nac.update_cluster_reward(AGENT, DARK, "tool:x", 1.0)
        sibling_before = nac.cluster_reward_bias(AGENT, DARK, "tool:x")
        for _ in range(50):
            nac.decay_reward_biases()
            nac.decay_cluster_reward_biases()
            nac.decay_eligibility()
        assert nac.cluster_fear(AGENT, DARK) == before, "per-tick decay must not touch fear"
        assert nac.cluster_reward_bias(AGENT, DARK, "tool:x") != sibling_before, (
            "anti-vacuity: the per-tick decayers did not run at all"
        )
        # Wall decay (cross-session) DOES reach it, in the slow class:
        decayed = nac.apply_wall_clock_decay(elapsed_s=7 * 86400.0)  # one slow half-life
        assert nac.cluster_fear(AGENT, DARK) == pytest.approx(before * 0.5, rel=0.01)
        assert "cluster_fear_pruned" not in decayed or decayed["cluster_fear_pruned"] == 0


class TestFearSubscriber:
    def test_pain_books_fear_on_noted_world_cluster_only(self):
        nac = _nac()
        bus = build_pain_bus(hippocampus=None, nac=nac)
        nac.note_active_clusters(AGENT, {"world": DARK, "interoception": "i-1"})
        bus.publish(_health_pain(1.0))
        assert nac.cluster_fear(AGENT, DARK) < 0.0
        assert nac.cluster_fear(AGENT, "i-1") == 0.0, "interoception fear is tautological"
        assert nac.cluster_fear(AGENT, LIT) == 0.0

    def test_no_noted_clusters_books_nothing(self):
        nac = _nac()
        bus = build_pain_bus(hippocampus=None, nac=nac)
        bus.publish(_health_pain(1.0))
        assert all(v == 0.0 for v in [nac.cluster_fear(AGENT, DARK), nac.cluster_fear(AGENT, LIT)])

    def test_low_intensity_and_disallowed_failure_mode_ignored(self):
        nac = _nac()
        bus = build_pain_bus(hippocampus=None, nac=nac)
        nac.note_active_clusters(AGENT, {"world": DARK})
        bus.publish(_health_pain(0.1))  # below the 0.3 subscriber threshold
        bus.publish(_health_pain(1.0, failure_mode="drive:food", source="drive:food"))
        assert nac.cluster_fear(AGENT, DARK) == 0.0


class TestFearRead:
    def test_anticipatory_need_thresholded_and_normalized(self):
        nac = _nac(cluster_fear_alpha=0.5, cluster_fear_threshold=0.3)
        nac.record_cluster_fear(AGENT, DARK, "drive:health", 0.4)  # -0.2, below θ
        assert nac.anticipatory_threat_need(AGENT, {"world": DARK}) == 0.0
        nac.record_cluster_fear(AGENT, DARK, "drive:health", 0.4)  # -0.4, above θ
        need = nac.anticipatory_threat_need(AGENT, {"world": DARK})
        assert need == pytest.approx(0.4)
        assert nac.anticipatory_threat_need(AGENT, {"world": LIT}) == 0.0
        assert nac.anticipatory_threat_need(AGENT, None) == 0.0

    def test_fear_selects_flee_through_recommend_action(self):
        """The load-bearing read (triple-confirmed DNB): a feared situation
        must surface as a threat need that selects the flee affordance on
        the REAL minecraft tool roster, through the real consumer."""
        nac = _nac()
        for _ in range(3):
            nac.record_cluster_fear(AGENT, DARK, "drive:health", 1.0)
        need = nac.anticipatory_threat_need(AGENT, {"world": DARK})
        assert need >= 0.9
        rec = nac.recommend_action(
            agent_id=AGENT,
            available_tools=MINECRAFT_TOOLS,
            current_drives={"threat": need},
            current_clusters=None,
            min_confidence=0.0,
        )
        assert rec is not None
        assert rec["tool_name"] == "minecraft_player_flee"

    def test_no_fear_no_flee_bias(self):
        """Specificity through the same consumer: without fear the threat
        need is absent and flee gets no drive boost."""
        nac = _nac()
        rec = nac.recommend_action(
            agent_id=AGENT,
            available_tools=MINECRAFT_TOOLS,
            current_drives={"threat": nac.anticipatory_threat_need(AGENT, {"world": LIT})},
            current_clusters=None,
            min_confidence=0.0,
        )
        assert rec is None or rec["tool_name"] != "minecraft_player_flee"


class TestFearHivemindPosture:
    """Phase-2 deferral enforced at all three sites (Exp 58 wiring W-6 fold):
    scrub excludes, ingest strips, merge min-folds (commutative — receiver-
    preserving given the scrub). A partial fold here reproduces the D43
    delete-state class the merge's own comments memorialize."""

    def _state_with_fear(self, value: float = -0.6) -> dict:
        nac = _nac()
        nac.record_cluster_fear(AGENT, DARK, "drive:health", -value / 0.5)
        state = nac.dump()
        assert "cluster_fear" in state and state["cluster_fear"]
        return state

    def test_bundle_scrub_excludes_fear(self):
        from maxim.hivemind.bundle import scrub_nac_state_for_bundle

        scrubbed = scrub_nac_state_for_bundle(self._state_with_fear())
        assert "cluster_fear" not in scrubbed, "fear must not travel in bundles (Phase-2 deferral)"

    def test_merge_preserves_receiver_fear_and_commutes(self):
        from maxim.hivemind.merge import nac_merge

        with_fear = self._state_with_fear()
        without = _nac().dump()
        a = nac_merge(with_fear, without, left_source="local", right_source="donor")
        b = nac_merge(without, with_fear, left_source="donor", right_source="local")
        assert a["cluster_fear"] == with_fear["cluster_fear"], "merge deleted the receiver's fear (D43 class)"
        assert a["cluster_fear"] == b["cluster_fear"], "fear fold must stay commutative"

    def test_merge_clamps_malformed_fear(self):
        from maxim.hivemind.merge import nac_merge

        bad = _nac().dump()
        bad["cluster_fear"] = {f"{AGENT}\x1f{DARK}\x1fdrive:health": -999.0, f"{AGENT}\x1f{LIT}\x1fdrive:health": 0.7}
        merged = nac_merge(bad, _nac().dump(), left_source="a", right_source="b")
        values = merged["cluster_fear"].values()
        assert all(-1.0 <= v <= 0.0 for v in values), "fold must clamp fear to [-1, 0]"
