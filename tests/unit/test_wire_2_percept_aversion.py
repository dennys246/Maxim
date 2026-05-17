"""Tests for Wire 2 (release_0_9_1.md Stage 3 — Pavlovian percept aversion).

Layers:

1. ``NAc.record_percept_valence`` / ``get_percept_valence`` — accumulation +
   clamping + per-agent isolation + agent_id validation.
2. ``NAc.get_percept_aversions`` — aggregate-by-entity_class, aversion-side
   only, floor pruning, cold-start.
3. ``NAc.decay_percept_valences`` — per-tick decay shape + prune.
4. NAc persistence round-trip — ``percept_valences`` round-trips through
   ``dump`` / ``load_state`` AND ``save`` / ``load`` (with
   ``_format_version=1.1``).  Backward-compat: 1.0 payloads (no
   ``percept_valences`` field) load cleanly to empty dict.
5. ``create_percept_valence_subscriber`` — intensity gating + agent_id
   from signal.context + interactive-mode gate + empty-field silent
   no-op + exception surfacing.
6. ``build_pain_bus`` auto-wires the subscriber when ``nac is not None``.
7. **Latent-bridge × subscriber trap regression** — Wire-A's
   ``_cluster_reward_bias`` write path AND Wire 2's
   ``_percept_valences`` write path are disjoint by key shape and by
   producer surface.  A single embodiment pain signal MUST NOT
   write to both maps for the same outcome twice.
8. ``GatingContext.learned_aversions`` + ``TextSalienceScorer`` —
   the aversion-side modulation lifts salience proportionally; the
   word-fragment matching rule fires on underscore-split entity_class
   keys; opt-out (None / empty dict) is a no-op.
9. ``BioEnrichmentPipeline._snapshot_learned_aversions`` — pipeline
   reads NAc, falls back to None on no-NAc / no-agent_id / NAc-raises.
10. Multi-agent isolation — two agents sharing one NAc instance
    attribute valence to distinct ``agent_id`` keys (CC4 rule).

Tests are env-var-free; the only env vars Wire 2 touches are the ones
introduced by upstream stages.  The latent-bridge trap test in §7 is
the load-bearing assertion the plan flagged in advance.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from maxim.decisions.nac import NAc, NACConfig, _NAC_FORMAT_VERSION
from maxim.proprioception.pain import PainSignal, PainType
from maxim.proprioception.pain_bus import (
    build_pain_bus,
    create_percept_valence_subscriber,
)
from maxim.runtime.gating import (
    GatingContext,
    TextSalienceScorer,
    _match_learned_aversion,
)


def _fresh_nac() -> NAc:
    return NAc(config=NACConfig(temporal_window_seconds=60.0))


def _make_pain_signal(
    *,
    intensity: float,
    agent_id: str = "agent_1",
    entity_type: str = "dragon",
    failure_mode: str = "burn",
    source: str = "embodiment",
) -> PainSignal:
    return PainSignal(
        pain_type=PainType.EXTERNAL_SIGNAL,
        intensity=intensity,
        timestamp=0.0,
        context={
            "source": source,
            "entity": "scene.dragon_1",
            "entity_type": entity_type,
            "failure_mode": failure_mode,
            "agent_id": agent_id,
        },
    )


# ─────────────────────────────────────────────────────────────────────
# Layer 1: record/get_percept_valence
# ─────────────────────────────────────────────────────────────────────


class TestRecordPerceptValence:
    """Accumulation + clamping + per-agent isolation + agent_id validation."""

    def test_initial_record_writes_alpha_scaled_value(self) -> None:
        nac = _fresh_nac()
        nac.record_percept_valence("dragon", "burn", -1.0, agent_id="a")
        # alpha = 0.20 → -0.20
        assert nac.get_percept_valence("dragon", "burn", agent_id="a") == pytest.approx(-0.20)

    def test_repeated_records_accumulate(self) -> None:
        nac = _fresh_nac()
        nac.record_percept_valence("dragon", "burn", -1.0, agent_id="a")
        nac.record_percept_valence("dragon", "burn", -1.0, agent_id="a")
        nac.record_percept_valence("dragon", "burn", -1.0, agent_id="a")
        # 3 × 0.20 × -1.0 = -0.60
        assert nac.get_percept_valence("dragon", "burn", agent_id="a") == pytest.approx(-0.60)

    def test_clamp_at_max_percept_valence(self) -> None:
        """Cap at ``-max_percept_valence`` no matter how many signals fire."""
        nac = _fresh_nac()
        for _ in range(50):
            nac.record_percept_valence("dragon", "burn", -1.0, agent_id="a")
        assert nac.get_percept_valence("dragon", "burn", agent_id="a") == pytest.approx(-nac.config.max_percept_valence)

    def test_clamp_at_positive_cap(self) -> None:
        """Positive valences also clamp symmetrically."""
        nac = _fresh_nac()
        for _ in range(50):
            nac.record_percept_valence("flower", "bloom", 1.0, agent_id="a")
        assert nac.get_percept_valence("flower", "bloom", agent_id="a") == pytest.approx(nac.config.max_percept_valence)

    def test_distinct_keys_dont_collide(self) -> None:
        nac = _fresh_nac()
        nac.record_percept_valence("dragon", "burn", -1.0, agent_id="a")
        nac.record_percept_valence("dragon", "crush", -1.0, agent_id="a")
        nac.record_percept_valence("rusty_sword", "burn", -1.0, agent_id="a")
        assert nac.get_percept_valence("dragon", "burn", agent_id="a") == pytest.approx(-0.20)
        assert nac.get_percept_valence("dragon", "crush", agent_id="a") == pytest.approx(-0.20)
        assert nac.get_percept_valence("rusty_sword", "burn", agent_id="a") == pytest.approx(-0.20)

    def test_per_agent_isolation(self) -> None:
        """Two agents on ONE NAc instance get separate keys (CC4 rule)."""
        nac = _fresh_nac()
        nac.record_percept_valence("dragon", "burn", -1.0, agent_id="a")
        nac.record_percept_valence("dragon", "burn", -1.0, agent_id="b")
        nac.record_percept_valence("dragon", "burn", -1.0, agent_id="b")
        assert nac.get_percept_valence("dragon", "burn", agent_id="a") == pytest.approx(-0.20)
        assert nac.get_percept_valence("dragon", "burn", agent_id="b") == pytest.approx(-0.40)

    def test_empty_agent_id_raises(self) -> None:
        nac = _fresh_nac()
        with pytest.raises(ValueError, match="agent_id"):
            nac.record_percept_valence("dragon", "burn", -1.0, agent_id="")

    def test_empty_entity_class_raises(self) -> None:
        nac = _fresh_nac()
        with pytest.raises(ValueError, match="entity_class"):
            nac.record_percept_valence("", "burn", -1.0, agent_id="a")

    def test_empty_failure_mode_raises(self) -> None:
        nac = _fresh_nac()
        with pytest.raises(ValueError, match="failure_mode"):
            nac.record_percept_valence("dragon", "", -1.0, agent_id="a")

    def test_get_returns_zero_for_unknown_triple(self) -> None:
        nac = _fresh_nac()
        assert nac.get_percept_valence("dragon", "burn", agent_id="a") == 0.0

    def test_get_returns_zero_for_empty_args(self) -> None:
        nac = _fresh_nac()
        nac.record_percept_valence("dragon", "burn", -1.0, agent_id="a")
        assert nac.get_percept_valence("", "burn", agent_id="a") == 0.0
        assert nac.get_percept_valence("dragon", "", agent_id="a") == 0.0
        assert nac.get_percept_valence("dragon", "burn", agent_id="") == 0.0


# ─────────────────────────────────────────────────────────────────────
# Layer 2: get_percept_aversions
# ─────────────────────────────────────────────────────────────────────


class TestGetPerceptAversions:
    """Aggregate-by-entity_class, aversion-side only, floor pruning."""

    def test_cold_start_returns_empty(self) -> None:
        nac = _fresh_nac()
        assert nac.get_percept_aversions(agent_id="a") == {}

    def test_aggregates_across_failure_modes_per_entity_class(self) -> None:
        """Most-negative valence wins; absolute value returned."""
        nac = _fresh_nac()
        # 1 hit (alpha=0.2): magnitude=0.20
        nac.record_percept_valence("dragon", "burn", -1.0, agent_id="a")
        # 3 hits: magnitude=0.60
        nac.record_percept_valence("dragon", "crush", -1.0, agent_id="a")
        nac.record_percept_valence("dragon", "crush", -1.0, agent_id="a")
        nac.record_percept_valence("dragon", "crush", -1.0, agent_id="a")
        result = nac.get_percept_aversions(agent_id="a")
        assert "dragon" in result
        assert result["dragon"] == pytest.approx(0.60)

    def test_positive_valences_excluded_from_aversion_read(self) -> None:
        """Only negative-valence entries contribute to the aversion map."""
        nac = _fresh_nac()
        nac.record_percept_valence("flower", "bloom", 1.0, agent_id="a")
        nac.record_percept_valence("dragon", "burn", -1.0, agent_id="a")
        result = nac.get_percept_aversions(agent_id="a")
        assert "flower" not in result
        assert "dragon" in result

    def test_below_floor_pruned(self) -> None:
        """Aversion magnitudes below the floor are dropped from the read."""
        nac = _fresh_nac()
        # Single small hit: magnitude=0.04 (alpha=0.20, valence=-0.20)
        nac.record_percept_valence("dragon", "burn", -0.20, agent_id="a")
        # 0.04 < default floor 0.05 → pruned
        assert nac.get_percept_aversions(agent_id="a") == {}

    def test_per_agent_isolation_in_aversion_read(self) -> None:
        nac = _fresh_nac()
        nac.record_percept_valence("dragon", "burn", -1.0, agent_id="a")
        nac.record_percept_valence("fire", "burn", -1.0, agent_id="b")
        a_result = nac.get_percept_aversions(agent_id="a")
        b_result = nac.get_percept_aversions(agent_id="b")
        assert "dragon" in a_result
        assert "dragon" not in b_result
        assert "fire" in b_result
        assert "fire" not in a_result

    def test_empty_agent_id_raises(self) -> None:
        nac = _fresh_nac()
        with pytest.raises(ValueError, match="agent_id"):
            nac.get_percept_aversions(agent_id="")


# ─────────────────────────────────────────────────────────────────────
# Layer 3: decay_percept_valences
# ─────────────────────────────────────────────────────────────────────


class TestDecayPerceptValences:
    """Per-tick decay + prune semantics."""

    def test_cold_start_returns_zero(self) -> None:
        nac = _fresh_nac()
        assert nac.decay_percept_valences() == 0

    def test_single_tick_decays_toward_zero(self) -> None:
        nac = _fresh_nac()
        nac.record_percept_valence("dragon", "burn", -1.0, agent_id="a")
        before = nac.get_percept_valence("dragon", "burn", agent_id="a")
        nac.decay_percept_valences()
        after = nac.get_percept_valence("dragon", "burn", agent_id="a")
        assert abs(after) < abs(before)
        # Magnitude reduced by 1/reward_bias_decay_tau (default 50.0).
        assert after == pytest.approx(before * (1.0 - 1.0 / 50.0))

    def test_many_ticks_prune_below_threshold(self) -> None:
        nac = _fresh_nac()
        nac.record_percept_valence("dragon", "burn", -0.10, agent_id="a")
        # Starting magnitude=0.02; decay tau=50; needs many ticks to fall below 0.001
        pruned_total = 0
        for _ in range(2000):
            pruned_total += nac.decay_percept_valences()
            if pruned_total > 0:
                break
        assert pruned_total == 1
        assert nac.get_percept_valence("dragon", "burn", agent_id="a") == 0.0


# ─────────────────────────────────────────────────────────────────────
# Layer 4: persistence round-trip + backward compat
# ─────────────────────────────────────────────────────────────────────


class TestNacPersistence:
    """``_format_version`` bump + percept_valences round-trip + back-compat."""

    def test_format_version_is_1_1(self) -> None:
        """Plan release_0_9_1.md: NAc dump bumps to 1.1."""
        assert _NAC_FORMAT_VERSION == "1.1"

    def test_dump_includes_percept_valences_key(self) -> None:
        nac = _fresh_nac()
        nac.record_percept_valence("dragon", "burn", -1.0, agent_id="a")
        state = nac.dump()
        assert "percept_valences" in state
        # Key shape: "{aid}\x1f{ec}\x1f{fm}"
        keys = list(state["percept_valences"].keys())
        assert any("\x1f" in k for k in keys)

    def test_round_trip_via_load_state(self) -> None:
        nac1 = _fresh_nac()
        nac1.record_percept_valence("dragon", "burn", -1.0, agent_id="a")
        nac1.record_percept_valence("dragon", "crush", -0.5, agent_id="a")
        nac1.record_percept_valence("rusty_sword", "burn", -0.5, agent_id="b")

        state = nac1.dump()
        nac2 = _fresh_nac()
        nac2.load_state(state)

        assert nac2.get_percept_valence("dragon", "burn", agent_id="a") == pytest.approx(
            nac1.get_percept_valence("dragon", "burn", agent_id="a")
        )
        assert nac2.get_percept_valence("dragon", "crush", agent_id="a") == pytest.approx(
            nac1.get_percept_valence("dragon", "crush", agent_id="a")
        )
        assert nac2.get_percept_valence("rusty_sword", "burn", agent_id="b") == pytest.approx(
            nac1.get_percept_valence("rusty_sword", "burn", agent_id="b")
        )

    def test_save_then_load_file(self, tmp_path: Path) -> None:
        nac1 = _fresh_nac()
        nac1.record_percept_valence("dragon", "burn", -1.0, agent_id="a")

        save_path = tmp_path / "nac.json"
        nac1.save(str(save_path))

        # File on disk carries _format_version=1.1
        data = json.loads(save_path.read_text())
        assert data["_format_version"] == "1.1"
        assert "percept_valences" in data

        nac2 = _fresh_nac()
        nac2.load(str(save_path))
        assert nac2.get_percept_valence("dragon", "burn", agent_id="a") == pytest.approx(-0.20)

    def test_backward_compat_load_1_0_payload(self) -> None:
        """1.0 payloads (no ``percept_valences`` field) load to empty dict."""
        nac = _fresh_nac()
        legacy_state = {
            "version": "1.0",
            "links": {},
            "outcome_index": {},
            "priors": {},
            "total_observations": 0,
            "reward_bias": {},
            "goal_reward_bias": {},
            "cluster_reward_bias": {},
            # NO percept_valences key
        }
        nac.load_state(legacy_state)
        # Must NOT raise; map is empty.
        assert nac.get_percept_aversions(agent_id="a") == {}

    def test_entity_class_with_colon_round_trips(self) -> None:
        """Drive-spec entity classes (``drive:hunger``) survive the
        ``\\x1f`` separator encoding even though they contain ``:``."""
        nac1 = _fresh_nac()
        nac1.record_percept_valence("infant_humanoid", "drive:hunger", -1.0, agent_id="a")

        state = nac1.dump()
        nac2 = _fresh_nac()
        nac2.load_state(state)
        assert nac2.get_percept_valence("infant_humanoid", "drive:hunger", agent_id="a") == pytest.approx(-0.20)


# ─────────────────────────────────────────────────────────────────────
# Layer 5: create_percept_valence_subscriber
# ─────────────────────────────────────────────────────────────────────


class TestPerceptValenceSubscriber:
    """Subscriber input handling + interactive-mode gate + agent_id derivation."""

    def test_intensity_below_threshold_no_op(self) -> None:
        nac = _fresh_nac()
        sub = create_percept_valence_subscriber(nac, intensity_threshold=0.3)
        sub(_make_pain_signal(intensity=0.1))
        assert nac.get_percept_aversions(agent_id="agent_1") == {}

    def test_intensity_above_threshold_writes_negative_valence(self) -> None:
        nac = _fresh_nac()
        sub = create_percept_valence_subscriber(nac)
        sub(_make_pain_signal(intensity=0.7))
        # Negative valence = pain. alpha=0.20 × -0.7 = -0.14.
        assert nac.get_percept_valence("dragon", "burn", agent_id="agent_1") == pytest.approx(-0.14)

    def test_missing_agent_id_silent_no_op(self) -> None:
        nac = _fresh_nac()
        sub = create_percept_valence_subscriber(nac)
        sig = PainSignal(
            pain_type=PainType.EXTERNAL_SIGNAL,
            intensity=0.7,
            timestamp=0.0,
            context={
                "source": "embodiment",
                "entity_type": "dragon",
                "failure_mode": "burn",
                # NO agent_id
            },
        )
        sub(sig)  # MUST NOT raise
        # No writes happened.
        assert not nac._percept_valences

    def test_missing_entity_type_silent_no_op(self) -> None:
        nac = _fresh_nac()
        sub = create_percept_valence_subscriber(nac)
        sig = PainSignal(
            pain_type=PainType.EXTERNAL_SIGNAL,
            intensity=0.7,
            timestamp=0.0,
            context={
                "source": "embodiment",
                "agent_id": "agent_1",
                "failure_mode": "burn",
                # NO entity_type
            },
        )
        sub(sig)
        assert not nac._percept_valences

    def test_interactive_mode_suppresses_writes(self, monkeypatch) -> None:
        """Mirror create_pain_nac_subscriber's interactive-mode gate."""
        from maxim.simulation import sim_logger as _sl

        monkeypatch.setattr(_sl, "_interactive_mode", _sl.InteractiveMode.ON)
        try:
            nac = _fresh_nac()
            sub = create_percept_valence_subscriber(nac)
            sub(_make_pain_signal(intensity=0.7))
            assert not nac._percept_valences
        finally:
            monkeypatch.setattr(_sl, "_interactive_mode", _sl.InteractiveMode.OFF)

    def test_nac_raise_logs_and_continues(self, caplog) -> None:
        """Subscriber catches all exceptions from NAc and logs (doesn't raise)."""
        nac_mock = MagicMock()
        nac_mock.record_percept_valence.side_effect = RuntimeError("boom")
        sub = create_percept_valence_subscriber(nac_mock)
        # Must NOT raise.
        sub(_make_pain_signal(intensity=0.7))
        assert "percept_valence recording failed" in caplog.text


# ─────────────────────────────────────────────────────────────────────
# Layer 6: build_pain_bus auto-wires the subscriber
# ─────────────────────────────────────────────────────────────────────


class TestBuildPainBusAutoWires:
    """The plan's "auto-wired via build_pain_bus" invariant."""

    def test_nac_provided_subscribes_percept_valence(self) -> None:
        """When nac is wired, build_pain_bus subscribes the new
        Wire 2 subscriber alongside ``create_pain_nac_subscriber``.

        Three direct subscribers expected:
        - create_pain_memory_subscriber (hippocampus)
        - create_pain_nac_subscriber (nac causal learning)
        - create_percept_valence_subscriber (nac Pavlovian learning)
        """
        hippo = MagicMock()
        hippo.capture = MagicMock()
        nac = _fresh_nac()
        bus = build_pain_bus(hippocampus=hippo, nac=nac)
        assert bus.get_stats()["direct_pain_subscribers"] == 3

    def test_nac_none_skips_percept_valence_subscriber(self) -> None:
        """Opt-out: nac=None → no Wire 2 subscriber."""
        hippo = MagicMock()
        bus = build_pain_bus(hippocampus=hippo, nac=None)
        # 1 subscriber (memory only); no percept-valence subscriber.
        assert bus.get_stats()["direct_pain_subscribers"] == 1

    def test_pain_signal_writes_percept_valence_via_bus(self) -> None:
        """End-to-end: bus publish → subscriber → NAc map updated."""
        nac = _fresh_nac()
        bus = build_pain_bus(hippocampus=None, nac=nac)
        bus.publish(_make_pain_signal(intensity=0.7))
        # alpha=0.20 × -0.7 = -0.14
        assert nac.get_percept_valence("dragon", "burn", agent_id="agent_1") == pytest.approx(-0.14)


# ─────────────────────────────────────────────────────────────────────
# Layer 7: LATENT-BRIDGE × SUBSCRIBER TRAP — the plan-flagged invariant
# ─────────────────────────────────────────────────────────────────────


class TestLatentBridgeSubscriberTrap:
    """Wire-A's read of ``_cluster_reward_bias`` and Wire 2's write of
    ``_percept_valences`` MUST NOT double-attribute the same outcome.

    Per release_0_9_1.md risk register + plan-flagged review concern:
    Wire-A writes via ``update_cluster_reward`` (called from
    ``record_outcome`` / ``record_outcome_full``) on the
    ``(agent_id, cluster_id, tool_signature)`` triple.  Wire 2 writes
    via ``record_percept_valence`` (called from
    ``create_percept_valence_subscriber``) on the
    ``(agent_id, entity_class, failure_mode)`` triple.  Different
    maps, different keys — no double-attribution by construction.
    """

    def test_pain_signal_writes_percept_valences_not_cluster_bias(self) -> None:
        """A pain signal flowing through the Wire 2 subscriber writes
        ONLY ``_percept_valences``; ``_cluster_reward_bias`` stays empty
        (Wire-A's write path is record_outcome-driven, not subscriber-
        driven)."""
        nac = _fresh_nac()
        bus = build_pain_bus(hippocampus=None, nac=nac)
        bus.publish(_make_pain_signal(intensity=0.7))
        assert len(nac._percept_valences) == 1
        assert len(nac._cluster_reward_bias) == 0

    def test_cluster_reward_write_does_not_pollute_percept_valences(self) -> None:
        """Wire-A's update_cluster_reward writes ONLY
        ``_cluster_reward_bias``; ``_percept_valences`` stays untouched."""
        nac = _fresh_nac()
        nac.update_cluster_reward(
            agent_id="a",
            cluster_id="cluster_42",
            tool_signature="tool:sense_food_source",
            reward=1.0,
        )
        assert len(nac._cluster_reward_bias) == 1
        assert len(nac._percept_valences) == 0

    def test_no_double_attribution_under_combined_path(self) -> None:
        """Even when BOTH paths fire (a record_outcome AND a pain
        publish), the two maps stay disjoint — keys differ in shape
        AND a single outcome doesn't accidentally hit both maps."""
        nac = _fresh_nac()
        bus = build_pain_bus(hippocampus=None, nac=nac)

        # Wire-A path: a tool succeeded, cluster bias accumulates.
        nac.update_cluster_reward(
            agent_id="agent_1",
            cluster_id="cluster_42",
            tool_signature="tool:slash",
            reward=1.0,
        )
        # Wire 2 path: a tool *outcome* produces embodiment pain;
        # the pain subscriber writes percept_valences.
        bus.publish(_make_pain_signal(intensity=0.7))

        # Each map has exactly its own entry — neither bled into the
        # other.
        assert len(nac._cluster_reward_bias) == 1
        assert len(nac._percept_valences) == 1
        # Wire-A's key shape: (agent_id, cluster_id, tool_signature)
        cluster_key = next(iter(nac._cluster_reward_bias.keys()))
        assert cluster_key == ("agent_1", "cluster_42", "tool:slash")
        # Wire 2's key shape: (agent_id, entity_class, failure_mode)
        valence_key = next(iter(nac._percept_valences.keys()))
        assert valence_key == ("agent_1", "dragon", "burn")


# ─────────────────────────────────────────────────────────────────────
# Layer 8: GatingContext + TextSalienceScorer
# ─────────────────────────────────────────────────────────────────────


class TestGatingContextLearnedAversions:
    """``learned_aversions`` field + scorer modulation + word-fragment match."""

    def test_gating_context_default_none(self) -> None:
        """Frozen dataclass additive field default is None (CC3 audit)."""
        ctx = GatingContext()
        assert ctx.learned_aversions is None

    def test_no_aversions_returns_zero_match(self) -> None:
        assert _match_learned_aversion({"dragon"}, None) == 0.0
        assert _match_learned_aversion({"dragon"}, {}) == 0.0

    def test_simple_match(self) -> None:
        aversions = {"dragon": 0.5}
        assert _match_learned_aversion({"dragon", "roars"}, aversions) == pytest.approx(0.5)

    def test_underscore_split_match(self) -> None:
        """``rusty_sword`` should split into {rusty, sword} and match
        a percept containing either word."""
        aversions = {"rusty_sword": 0.6}
        assert _match_learned_aversion({"the", "rusty", "blade"}, aversions) == pytest.approx(0.6)
        assert _match_learned_aversion({"the", "sword", "swings"}, aversions) == pytest.approx(0.6)

    def test_drive_colon_split_match(self) -> None:
        """``drive:hunger`` (as part of an entity_class) splits on
        ``:`` for fragment matching."""
        aversions = {"drive:hunger": 0.4}
        assert _match_learned_aversion({"hunger", "pangs"}, aversions) == pytest.approx(0.4)

    def test_strongest_aversion_wins(self) -> None:
        """When multiple keys match, the strongest magnitude wins."""
        aversions = {"dragon": 0.3, "fire": 0.7}
        assert _match_learned_aversion({"dragon", "fire", "breath"}, aversions) == pytest.approx(0.7)

    def test_zero_magnitude_ignored(self) -> None:
        aversions = {"dragon": 0.0}
        assert _match_learned_aversion({"dragon"}, aversions) == 0.0

    def test_no_word_overlap_returns_zero(self) -> None:
        aversions = {"dragon": 0.5}
        assert _match_learned_aversion({"flower", "blooms"}, aversions) == 0.0


class TestTextSalienceScorerAversion:
    """End-to-end: scorer raises salience on aversive percepts."""

    def test_no_aversions_baseline_unchanged(self) -> None:
        scorer = TextSalienceScorer(ec=None, nac=None)
        ctx = GatingContext()
        score_no_aversion = scorer.score("a dragon roars", ctx).salience
        scorer.reset()
        ctx_empty = GatingContext(learned_aversions={})
        score_empty = scorer.score("a dragon roars", ctx_empty).salience
        assert score_no_aversion == pytest.approx(score_empty)

    def test_aversion_lifts_salience(self) -> None:
        """Aversive percepts get raised salience.

        Baseline (no NAc) = 0.5.  Saturating mix with aversion=0.6:
        0.5 + (1-0.5)*0.6 = 0.80.
        """
        scorer = TextSalienceScorer(ec=None, nac=None)
        ctx = GatingContext(learned_aversions={"dragon": 0.6})
        score = scorer.score("a dragon roars in the distance", ctx)
        assert score.salience == pytest.approx(0.80)

    def test_aversion_does_not_match_unrelated_percept(self) -> None:
        scorer = TextSalienceScorer(ec=None, nac=None)
        ctx = GatingContext(learned_aversions={"dragon": 0.6})
        score = scorer.score("a flower blooms", ctx)
        # No overlap → no boost.
        assert score.salience == pytest.approx(0.5)

    def test_aversion_modulation_clamped_to_one(self) -> None:
        scorer = TextSalienceScorer(ec=None, nac=None)
        ctx = GatingContext(learned_aversions={"dragon": 5.0})  # > 1, clamped
        score = scorer.score("a dragon roars", ctx)
        # min(1, aversion) → boost = 0.5 → 1.0
        assert score.salience == pytest.approx(1.0)


# ─────────────────────────────────────────────────────────────────────
# Layer 9: BioEnrichmentPipeline._snapshot_learned_aversions
# ─────────────────────────────────────────────────────────────────────


class TestBioEnrichmentSnapshot:
    """The pipeline reads NAc once per ``enrich`` call and injects into
    ``GatingContext.learned_aversions``."""

    def _make_pipeline(self, *, nac, agent_id: str):
        from maxim.integration.bio_enrichment import BioEnrichmentPipeline

        scorer = TextSalienceScorer(ec=None, nac=nac)
        return BioEnrichmentPipeline(
            scorer=scorer,
            hippocampus=None,
            nac=nac,
            atl=None,
            ec=None,
            agent_id=agent_id,
            novelty_threshold=0.0,
        )

    def test_no_nac_returns_none(self) -> None:
        from maxim.integration.bio_enrichment import BioEnrichmentPipeline

        pipeline = BioEnrichmentPipeline(scorer=None, hippocampus=None, nac=None)
        assert pipeline._snapshot_learned_aversions() is None

    def test_no_agent_id_returns_none(self) -> None:
        from maxim.integration.bio_enrichment import BioEnrichmentPipeline

        nac = _fresh_nac()
        pipeline = BioEnrichmentPipeline(scorer=None, hippocampus=None, nac=nac, agent_id="")
        assert pipeline._snapshot_learned_aversions() is None

    def test_nac_raise_returns_none(self) -> None:
        from maxim.integration.bio_enrichment import BioEnrichmentPipeline

        nac_mock = MagicMock()
        nac_mock.get_percept_aversions.side_effect = RuntimeError("boom")
        pipeline = BioEnrichmentPipeline(scorer=None, hippocampus=None, nac=nac_mock, agent_id="a")
        assert pipeline._snapshot_learned_aversions() is None

    def test_nac_returns_map(self) -> None:
        nac = _fresh_nac()
        nac.record_percept_valence("dragon", "burn", -1.0, agent_id="a")
        nac.record_percept_valence("dragon", "burn", -1.0, agent_id="a")
        nac.record_percept_valence("dragon", "burn", -1.0, agent_id="a")
        pipeline = self._make_pipeline(nac=nac, agent_id="a")
        snap = pipeline._snapshot_learned_aversions()
        assert snap == {"dragon": pytest.approx(0.60)}


# ─────────────────────────────────────────────────────────────────────
# Layer 10: Multi-agent NAc isolation (CC4 rule)
# ─────────────────────────────────────────────────────────────────────


class TestMultiAgentIsolation:
    """Two agents sharing one NAc instance MUST attribute valence to
    distinct ``agent_id`` keys per CLAUDE.md per-agent stash rule.
    """

    def test_subscribers_with_distinct_agent_ids_dont_collide(self) -> None:
        """Two distinct embodiment pain events (different entity_paths
        so the PainBus (entity, failure_mode) refractory gate doesn't
        collapse them) MUST stay isolated by agent_id.
        """
        nac = _fresh_nac()
        bus = build_pain_bus(hippocampus=None, nac=nac)

        # Agent A's pain event.
        sig_a = PainSignal(
            pain_type=PainType.EXTERNAL_SIGNAL,
            intensity=0.7,
            timestamp=0.0,
            context={
                "source": "embodiment",
                "entity": "scene.agent_a.body.arm",  # different entity_path
                "entity_type": "dragon",
                "failure_mode": "burn",
                "agent_id": "agent_a",
            },
        )
        bus.publish(sig_a)

        # Agent B's pain event (different entity_path → no refractory
        # collision; different agent_id → isolated bucket).
        sig_b = PainSignal(
            pain_type=PainType.EXTERNAL_SIGNAL,
            intensity=0.7,
            timestamp=0.0,
            context={
                "source": "embodiment",
                "entity": "scene.agent_b.body.arm",
                "entity_type": "fire",
                "failure_mode": "burn",
                "agent_id": "agent_b",
            },
        )
        bus.publish(sig_b)

        a_aversions = nac.get_percept_aversions(agent_id="agent_a")
        b_aversions = nac.get_percept_aversions(agent_id="agent_b")
        assert "dragon" in a_aversions
        assert "fire" in b_aversions
        # Neither agent sees the other's aversion.
        assert "fire" not in a_aversions
        assert "dragon" not in b_aversions
