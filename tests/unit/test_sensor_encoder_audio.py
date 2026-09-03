"""Tests for the exteroceptive 'audio' substrate modality (commit 3).

``SensorEncoder.encode_sensors(modality=...)`` routes exteroceptive
sensors (sound localization) into their own EC modality space with
frozen-centroid semantics, while the ``interoception`` default stays
byte-identical for existing callers.

Regression guard for perception_pipeline_placement.md (Q5 + commit 3).
"""

from __future__ import annotations

from maxim.similarity.ec import ECConfig, EntorhinalCortex
from maxim.similarity.encoder import SensorEncoder


def _ec() -> EntorhinalCortex:
    return EntorhinalCortex(ECConfig(pattern_complete_threshold=0.40))


class TestModalityRouting:
    def test_default_modality_is_interoception(self):
        # Existing callers (no modality arg) are byte-identical.
        ec = _ec()
        enc = SensorEncoder(ec=ec, atl=None, nac=None)
        node = enc.encode_sensors(agent_id="a", sensors={"hunger": 0.2})
        assert ec._substrate_nodes[node][1] == "interoception"

    def test_audio_modality_tags_node_audio(self):
        ec = _ec()
        enc = SensorEncoder(ec=ec, atl=None, nac=None)
        node = enc.encode_sensors(agent_id="a", sensors={"azimuth": 0.5}, modality="audio")
        assert ec._substrate_nodes[node][1] == "audio"

    def test_audio_and_interoception_are_separate_node_spaces(self):
        # EC compares within-modality, so an audio reading never completes
        # onto an interoception node even with an identical sensor dict.
        ec = _ec()
        enc = SensorEncoder(ec=ec, atl=None, nac=None)
        intero = enc.encode_sensors(agent_id="a", sensors={"x": 0.5})
        audio = enc.encode_sensors(agent_id="a", sensors={"x": 0.5}, modality="audio")
        assert intero != audio
        assert ec._substrate_nodes[intero][1] == "interoception"
        assert ec._substrate_nodes[audio][1] == "audio"


class TestPerAgentModalityStashIsolation:
    def test_delta_gate_does_not_conflate_modalities(self):
        # The delta gate is keyed per (agent_id, modality). Encoding a
        # second modality for an agent must not return that agent's
        # other-modality node via the delta short-circuit, even when the
        # sensor dicts are identical.
        ec = _ec()
        enc = SensorEncoder(ec=ec, atl=None, nac=None)
        intero = enc.encode_sensors(agent_id="a", sensors={"hunger": 0.5})
        audio = enc.encode_sensors(agent_id="a", sensors={"hunger": 0.5}, modality="audio")
        assert audio != intero
        assert ec._substrate_nodes[audio][1] == "audio"

    def test_delta_gate_still_short_circuits_within_a_modality(self):
        # Same (agent, modality) + sub-delta change still returns the cached
        # node (the gate works; we only re-keyed it, didn't disable it).
        ec = _ec()
        enc = SensorEncoder(ec=ec, atl=None, nac=None)
        first = enc.encode_sensors(agent_id="a", sensors={"azimuth": 0.50}, modality="audio")
        # A change below min_delta returns the same node without a new EC scan.
        again = enc.encode_sensors(agent_id="a", sensors={"azimuth": 0.50}, modality="audio")
        assert again == first


class TestAudioFrozenCentroid:
    def test_audio_in_default_frozen_set(self):
        assert "audio" in ECConfig().frozen_centroid_modalities

    def test_audio_node_centroid_frozen_on_completion(self):
        # Register an audio node, then complete a near (above-threshold)
        # reading onto it. Frozen => the stored embedding does NOT move.
        ec = _ec()
        e0 = [1.0, 0.0, 0.0]
        ec.register_substrate_node("n_audio", e0, "audio")
        e1 = [1.0, 0.5, 0.0]  # cos(e0, e1) ≈ 0.894 > 0.40 → completes
        res = ec.pattern_complete_or_separate(e1, "audio", threshold=0.40, geometry=None)
        assert not res.is_new
        assert res.node_id == "n_audio"
        assert ec._substrate_nodes["n_audio"][0] == e0  # unchanged (frozen)

    def test_non_frozen_modality_still_drifts(self):
        # Contrast: "vision" is NOT frozen, so the centroid running-mean
        # path still updates (confirms commit 3 didn't break drifting).
        ec = _ec()
        e0 = [1.0, 0.0, 0.0]
        ec.register_substrate_node("n_vis", e0, "vision")
        e1 = [1.0, 0.5, 0.0]
        res = ec.pattern_complete_or_separate(e1, "vision", threshold=0.40, geometry=None)
        assert not res.is_new
        stored = ec._substrate_nodes["n_vis"][0]
        assert stored != e0
        assert stored[1] == 0.25  # running mean (e0*1 + e1)/2
