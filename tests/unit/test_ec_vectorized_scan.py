"""Guards for the 1.1.4 PR 1 vectorized substrate scan + A4 per-modality gain.

The vectorized scan (`ec.py::_ModalityMatrix`) is the measured remedy for the
scan-cost index-prerequisite verdict (`docs/plans/world_seam_1_1_4.md` §PR 0
result). It must make the SAME DECISIONS as the per-node reference loop it
replaced — kept verbatim as `EntorhinalCortex._scan_substrate_reference` for
exactly this comparison. Decision equivalence, not bit equality: the BLAS
path sums in a different order, so similarities may differ in the last ulp.

The A4 gain tests pin the plan's kickoff decisions: D1 (the measured
equation, literal 0.5 neutral), D2 (a gained body at rest encodes nothing),
D3 resolved to per-modality membership (world gained; interoception and
audio ungained — measured worse at their scales), and the geometry-tag rule
that an UNGAINED modality's tag must stay byte-identical to pre-A4 so no
existing node re-stales.
"""

from __future__ import annotations

import random

import pytest

from maxim.similarity.ec import ECConfig, EntorhinalCortex, _cosine_similarity
from maxim.similarity.encoder import (
    SensorEncoder,
    SensorEncoderConfig,
    _sensor_embed,
    _stable_basis,
    encoding_geometry_tag,
)

DIM = 16  # small dim keeps the randomized sweep fast; nothing scales with it


def _vec(rng: random.Random, dim: int = DIM) -> list[float]:
    return [rng.uniform(-1.0, 1.0) for _ in range(dim)]


class TestVectorizedScanEquivalence:
    """The vectorized scan and the reference loop must agree on every decision."""

    def _assert_same_decision(self, ec, embedding, modality, threshold, overrides, geometry):
        ref_node, ref_sim = ec._scan_substrate_reference(embedding, modality, threshold, overrides, geometry)
        result = ec.pattern_complete_or_separate(
            embedding,
            modality=modality,
            threshold=threshold,
            threshold_override=overrides or None,
            geometry=geometry,
        )
        if ref_node is None:
            assert result.is_new, f"reference separated but vectorized completed onto {result.node_id}"
        else:
            assert not result.is_new, "reference completed but vectorized separated"
            assert result.node_id == ref_node
            assert result.similarity == pytest.approx(ref_sim, abs=1e-9)
        return result

    def test_randomized_operation_stream_agrees_with_reference(self):
        """Registers, removes, re-registers, centroid updates, geometry filters,
        threshold overrides, mixed dims, frozen + non-frozen modalities — after
        every mutation the two scans must make the same completion decision."""
        rng = random.Random(1104)
        ec = EntorhinalCortex(ECConfig(frozen_centroid_modalities=frozenset({"frozen_mod"})))
        modalities = ["world", "frozen_mod"]
        geometries = [None, "gA", "gB"]
        registered: list[str] = []

        for step in range(400):
            op = rng.random()
            modality = rng.choice(modalities)
            geometry = rng.choice(geometries)
            # Occasional 24-dim nodes exercise the per-dim matrix split.
            dim = 24 if rng.random() < 0.1 else DIM
            emb = _vec(rng, dim)
            threshold = rng.choice([0.44, 0.85])
            overrides = (
                {nid: rng.uniform(0.2, 0.99) for nid in rng.sample(registered, min(3, len(registered)))}
                if registered and rng.random() < 0.3
                else {}
            )

            if op < 0.15 and registered:
                victim = rng.choice(registered)
                registered.remove(victim)
                ec.remove_substrate_node(victim)
            elif op < 0.2 and registered:
                # re-register an existing id with fresh values ("register or update")
                victim = rng.choice(registered)
                ec.register_substrate_node(victim, _vec(rng, dim), modality, geometry=geometry)

            result = self._assert_same_decision(ec, emb, modality, threshold, overrides, geometry)
            if result.is_new:
                ec.register_substrate_node(result.node_id, emb, modality, geometry=geometry)
                registered.append(result.node_id)

        assert len(registered) > 50, "sweep degenerated — too few nodes to be a real test"

    def test_centroid_update_reaches_the_matrix(self):
        """A non-frozen completion updates the running-mean centroid; the NEXT
        scan must see the moved centroid, not a stale cached row."""
        ec = EntorhinalCortex(ECConfig())
        base = [1.0] + [0.0] * (DIM - 1)
        near = [0.9, 0.1] + [0.0] * (DIM - 2)
        first = ec.pattern_complete_or_separate(base, modality="world", threshold=0.85, geometry=None)
        assert first.is_new
        ec.register_substrate_node(first.node_id, base, "world")
        second = ec.pattern_complete_or_separate(near, modality="world", threshold=0.85, geometry=None)
        assert not second.is_new
        stored, _ = ec._substrate_nodes[first.node_id]
        expected = [(b + n) / 2 for b, n in zip(base, near)]
        assert stored == pytest.approx(expected)
        # and the matrix row moved with it: a query at the NEW centroid scores ~1.0
        again = ec.pattern_complete_or_separate(expected, modality="world", threshold=0.85, geometry=None)
        assert not again.is_new and again.node_id == first.node_id
        assert again.similarity == pytest.approx(
            _cosine_similarity(expected, ec._substrate_nodes[first.node_id][0]), abs=1e-9
        )

    def test_geometry_stamp_on_first_touch_reaches_the_matrix(self):
        """A legacy (unstamped) node adopts the live tag on first match; a
        DIFFERENT tag afterwards must then be filtered by the cached matrix."""
        ec = EntorhinalCortex(ECConfig(frozen_centroid_modalities=frozenset({"world"})))
        emb = [1.0] + [0.0] * (DIM - 1)
        ec.register_substrate_node("legacy", emb, "world", geometry=None)
        touched = ec.pattern_complete_or_separate(emb, modality="world", threshold=0.85, geometry="gA")
        assert not touched.is_new and touched.node_id == "legacy"
        foreign = ec.pattern_complete_or_separate(emb, modality="world", threshold=0.85, geometry="gB")
        assert foreign.is_new, "stamped node must be invisible to a different geometry"

    def test_save_load_round_trip_scans_identically(self, tmp_path):
        rng = random.Random(7)
        ec = EntorhinalCortex(ECConfig())
        embs = {}
        for i in range(20):
            emb = _vec(rng)
            r = ec.pattern_complete_or_separate(emb, modality="world", threshold=0.85, geometry="gA")
            if r.is_new:
                ec.register_substrate_node(r.node_id, emb, "world", geometry="gA")
            embs[i] = emb
        path = str(tmp_path / "ec.json")
        ec.save(path)
        ec2 = EntorhinalCortex(ECConfig())
        ec2.load(path)
        for emb in embs.values():
            a = ec._scan_substrate_reference(emb, "world", 0.85, {}, "gA")
            b = ec2._scan_substrate_reference(emb, "world", 0.85, {}, "gA")
            r2 = ec2.pattern_complete_or_separate(emb, modality="world", threshold=0.85, geometry="gA")
            assert a[0] == b[0]
            assert (r2.node_id == b[0]) == (not r2.is_new) or b[0] is None


class TestA4GainEncoding:
    """Plan decisions D1–D3 pinned in code."""

    def test_gain_none_is_byte_identical_to_pre_a4(self):
        sensors = {"a": 0.3, "b": 0.9, "c": 0.5}
        got = _sensor_embed(sensors, dim=DIM)  # default gain_exponent=None
        expected = [0.0] * DIM
        for name in sorted(sensors):
            v = sensors[name]  # already in [0,1]; range-blind map is identity here
            lo = _stable_basis(name, DIM, salt="low")
            hi = _stable_basis(name, DIM, salt="high")
            for i in range(DIM):
                expected[i] += (1.0 - v) * lo[i] + v * hi[i]
        assert got == pytest.approx(expected)

    def test_gain_silences_a_sensor_at_neutral_and_weights_the_moved_one(self):
        moved_only = _sensor_embed({"moved": 1.0}, dim=DIM, gain_exponent=3.0)
        with_resting = _sensor_embed({"moved": 1.0, "resting": 0.5}, dim=DIM, gain_exponent=3.0)
        assert with_resting == pytest.approx(moved_only), (
            "a sensor at its neutral point must contribute nothing under gain"
        )

    def test_gained_body_at_rest_encodes_nothing(self):
        """D2: all-neutral world reading → zero vector → encode returns None,
        and the delta gate keeps returning None while nothing moves."""
        ec = EntorhinalCortex(ECConfig())
        enc = SensorEncoder(ec=ec)
        at_rest = {f"s{i}": 0.5 for i in range(6)}
        assert enc.encode_sensors(agent_id="a", sensors=at_rest, modality="world") is None
        assert enc.encode_sensors(agent_id="a", sensors=at_rest, modality="world") is None
        assert len(ec._substrate_nodes) == 0
        moved = dict(at_rest, s0=1.0)
        assert enc.encode_sensors(agent_id="a", sensors=moved, modality="world") is not None

    def test_gain_membership_is_world_only_by_default(self):
        cfg = SensorEncoderConfig()
        assert cfg.gain_modalities == frozenset({"world"})
        assert cfg.gain_exponent == 3.0
        assert cfg.pattern_threshold == 0.85, "A5 lesson: gain ships at the UNCHANGED threshold"

    def test_ungained_modality_geometry_tag_is_byte_identical_to_pre_a4(self):
        """An interoception (ungained) encode must produce EXACTLY the tag the
        pre-A4 code produced — no `gain` field — or every existing node
        re-stales for a space that did not change."""
        ec = EntorhinalCortex(ECConfig())
        enc = SensorEncoder(ec=ec)
        ranges = {"hunger": (0.0, 1.0)}
        node = enc.encode_sensors(agent_id="a", sensors={"hunger": 0.9}, modality="interoception", ranges=ranges)
        assert node is not None
        pre_a4_tag = encoding_geometry_tag(
            encoder="sensor",
            modality="interoception",
            declared_sensors=["hunger"],
            normalization="range-aware",
            embedding_dim=384,
        )
        assert ec._substrate_node_geometries[node] == pre_a4_tag

    def test_gained_modality_geometry_tag_moves(self):
        ec = EntorhinalCortex(ECConfig())
        enc = SensorEncoder(ec=ec)
        ranges = {"light": (0.0, 1.0)}
        node = enc.encode_sensors(agent_id="a", sensors={"light": 0.95}, modality="world", ranges=ranges)
        assert node is not None
        ungained_tag = encoding_geometry_tag(
            encoder="sensor",
            modality="world",
            declared_sensors=["light"],
            normalization="range-aware",
            embedding_dim=384,
        )
        assert ec._substrate_node_geometries[node] != ungained_tag
        assert ec._substrate_node_geometries[node] == encoding_geometry_tag(
            encoder="sensor",
            modality="world",
            declared_sensors=["light"],
            normalization="range-aware",
            embedding_dim=384,
            gain="p3.0",
        )

    def test_gain_exponent_recorded_in_provenance(self):
        ec = EntorhinalCortex(ECConfig())
        enc = SensorEncoder(ec=ec)
        enc.encode_sensors(agent_id="a", sensors={"light": 0.95}, modality="world")
        enc.encode_sensors(agent_id="a", sensors={"hunger": 0.9}, modality="interoception")
        assert ec.encoder_provenance["sensor:world"]["gain_exponent"] == 3.0
        assert ec.encoder_provenance["sensor:interoception"]["gain_exponent"] is None
