"""Gate 2 (1.1.3) — D3 threshold pin + D4 same-dimension geometry guard.

> *D3/D4 threshold and same-dimension geometry compatibility must be explicit
> and tested, not inferred from vector length.* — roadmap, "Gates before 1.2"

Both defects are the same species: **a compatibility check that cannot fire.**

**D3** — `merge.py` deliberately refuses internal imports, so its thresholds are
hardcoded duplicates of the encoder/EC defaults. The frozen-modality set had
already diverged this way once (`"audio"` was in `ECConfig` and not in
`merge.py`, so audio centroids drifted across contributors) and is now pinned;
the thresholds were not. **D3's trigger is literally "per-modality thresholds",
which D43 shipped** — `SENSOR_MODALITY_THRESHOLDS` is a NEW duplicate of 0.85
that arrived unpinned, so this file is owed by that change.

**D4** — `_cosine` refuses a dimension mismatch, which catches a 384-vs-768
encoder swap. It cannot catch a **same-dimension** geometry change: a place code
adds sensor names, so `_sensor_embed` sums a different basis set while keeping
`dim=384` and the same `"audio"` modality tag. Old- and new-geometry nodes then
fold whenever the partial cosine clears the threshold — and because `audio` is a
frozen-centroid modality the centroid never moves, so the only symptom is
inflated counts and contributors. Nothing is observably wrong.
"""

from __future__ import annotations

from maxim.hivemind.merge import (
    DEFAULT_FROZEN_CENTROID_MODALITIES,
    SENSOR_MODALITY_THRESHOLDS,
    ec_merge,
    ec_merge_aligned,
)
from maxim.similarity.ec import ECConfig, EntorhinalCortex
from maxim.similarity.encoder import SensorEncoderConfig, encoding_geometry_tag


class TestD3ThresholdPin:
    """The duplicated literals must track their canonical sources."""

    def test_text_default_matches_ec(self):
        """`ec_merge`'s default is the EC pattern-complete threshold."""
        import inspect

        default = inspect.signature(ec_merge).parameters["cosine_threshold"].default
        assert default == ECConfig().pattern_complete_threshold, (
            "ec_merge's cosine_threshold drifted from ECConfig.pattern_complete_threshold"
        )

    def test_sensor_thresholds_match_the_sensor_encoder(self):
        """THE D43 CASE. Sensor clusters align at the threshold they FORMED at.

        If `SensorEncoderConfig.pattern_threshold` moves and this map does not,
        the merge silently goes back to aligning interoception clusters at the
        text default — a confidently wrong alignment, which D43 established is
        strictly worse than an honestly missing one.
        """
        expected = SensorEncoderConfig().pattern_threshold
        assert SENSOR_MODALITY_THRESHOLDS, "the map must not be empty — that silently restores the text default"
        for modality, value in SENSOR_MODALITY_THRESHOLDS.items():
            assert value == expected, f"{modality} pinned at {value}, SensorEncoder forms clusters at {expected}"

    def test_every_frozen_centroid_modality_has_a_threshold(self):
        """The two sets must not drift apart.

        A frozen-centroid modality is one whose centroid must never be moved by
        a merge. Such a modality aligning at the *text* threshold is the exact
        pairing that produced D43: it folds too eagerly AND cannot self-correct,
        because the centroid is pinned.
        """
        missing = set(DEFAULT_FROZEN_CENTROID_MODALITIES) - set(SENSOR_MODALITY_THRESHOLDS)
        assert not missing, f"frozen-centroid modalities with no pinned threshold: {sorted(missing)}"


class TestD4GeometryGuard:
    """Same dimension, same modality, different encoding space — must not fold."""

    @staticmethod
    def _node(vec, geometry=None, modality="audio"):
        n = {"embedding": vec, "modality": modality, "count": 1, "source": "s"}
        if geometry is not None:
            n["geometry"] = geometry
        return n

    def test_identical_vectors_in_different_geometries_do_not_fold(self):
        """The defect, stated at its sharpest.

        Two nodes with the SAME dimension, SAME modality and a cosine of
        exactly 1.0 must still not merge if they declare different encoding
        spaces. Similarity is meaningless across spaces — that is the whole
        point, and it is why this cannot be expressed as a threshold.
        """
        left = {"L": self._node([1.0, 0.0], geometry="gAAAA")}
        right = {"R": self._node([1.0, 0.0], geometry="gBBBB")}

        res = ec_merge_aligned(left, right, left_source="A", right_source="B")
        assert len(res.nodes) == 2, "nodes from different geometries folded"
        assert res.id_map["R"] != "L"

    def test_same_geometry_still_folds(self):
        """The guard must not simply disable merging."""
        left = {"L": self._node([1.0, 0.0], geometry="gSAME")}
        right = {"R": self._node([1.0, 0.0], geometry="gSAME")}

        res = ec_merge_aligned(left, right, left_source="A", right_source="B")
        assert len(res.nodes) == 1
        assert res.id_map["R"] == "L"

    def test_unstamped_legacy_nodes_still_fold_by_default(self):
        """Absence is unverifiable, not a mismatch — old files must still load.

        Mirrors gate 7's `body_ref` precedent: a declared mismatch is refused,
        an undeclared one is admitted unless the caller asks for strictness.
        """
        left = {"L": self._node([1.0, 0.0])}
        right = {"R": self._node([1.0, 0.0], geometry="gAAAA")}

        res = ec_merge_aligned(left, right, left_source="A", right_source="B")
        assert len(res.nodes) == 1, "an unstamped legacy node should still merge by default"

    def test_strict_geometry_refuses_unstamped(self):
        """The opt-in that a shared Oasis should run with."""
        left = {"L": self._node([1.0, 0.0])}
        right = {"R": self._node([1.0, 0.0], geometry="gAAAA")}

        res = ec_merge_aligned(left, right, left_source="A", right_source="B", strict_geometry=True)
        assert len(res.nodes) == 2

    def test_geometry_survives_the_merge(self):
        """If the tag were dropped, the NEXT merge would see two unstamped
        nodes and the guard would silently switch itself off."""
        left = {"L": self._node([1.0, 0.0], geometry="gSAME")}
        right = {"R": self._node([0.0, 1.0], geometry="gSAME")}

        merged = ec_merge(left, right, left_source="A", right_source="B")
        assert all(n.get("geometry") == "gSAME" for n in merged.values())


class TestGeometryTagIsMeaningful:
    """The tag must actually change when the encoding space changes."""

    def test_added_sensor_changes_the_tag(self):
        """A place code adds sensor names at identical dim — D4's own example."""
        before = encoding_geometry_tag(encoder="sensor", modality="audio", sensor_names=["azimuth"], embedding_dim=384)
        after = encoding_geometry_tag(
            encoder="sensor", modality="audio", sensor_names=["azimuth", "place_0"], embedding_dim=384
        )
        assert before != after

    def test_normalization_mode_changes_the_tag(self):
        a = encoding_geometry_tag(encoder="sensor", sensor_names=["x"], normalization="range")
        b = encoding_geometry_tag(encoder="sensor", sensor_names=["x"], normalization="raw")
        assert a != b

    def test_tag_is_order_independent_and_stable(self):
        """Same space must give the same tag, or every merge sees a mismatch."""
        a = encoding_geometry_tag(encoder="sensor", sensor_names=["b", "a"], embedding_dim=384)
        b = encoding_geometry_tag(embedding_dim=384, sensor_names=["a", "b"], encoder="sensor")
        assert a == b

    def test_tag_is_process_stable(self):
        """Persisted and compared across processes — builtin hash() would make
        it permanently unmatchable under PYTHONHASHSEED randomization."""
        import subprocess
        import sys

        code = (
            "import sys; sys.path.insert(0,'src');"
            "from maxim.similarity.encoder import encoding_geometry_tag;"
            "print(encoding_geometry_tag(encoder='sensor', sensor_names=['a','b'], embedding_dim=384))"
        )
        outs = set()
        for seed in ("0", "12345"):
            r = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True,
                text=True,
                env={"PYTHONHASHSEED": seed, "PATH": "/usr/bin:/bin"},
            )
            # Without these two the test passes on a BROKEN subprocess: an
            # import failure under the wiped env yields two empty strings,
            # `len({""}) == 1`, and a green result. A two-process determinism
            # check that cannot tell determinism from "neither process ran" is
            # the vacuous shape this file exists to refuse. (Review round.)
            assert r.returncode == 0, f"subprocess failed (seed {seed}): {r.stderr[-400:]}"
            assert r.stdout.strip(), f"subprocess produced no output (seed {seed}): {r.stderr[-400:]}"
            # Without these two the test passes on a BROKEN subprocess: an
            # import failure under the wiped env yields two empty strings,
            # `len({""}) == 1`, and green. A determinism check that cannot tell
            # determinism from "neither process ran" is the vacuous shape this
            # file exists to refuse. (Review round.)
            assert r.returncode == 0, f"subprocess failed (seed {seed}): {r.stderr[-400:]}"
            assert r.stdout.strip(), f"subprocess produced no output (seed {seed}): {r.stderr[-400:]}"
            outs.add(r.stdout.strip())
        assert len(outs) == 1, f"geometry tag differs across PYTHONHASHSEED: {outs}"


class TestGeometryIsStampedInProduction:
    """A guard nothing stamps is a guard that never fires (this session's lesson)."""

    def test_sensor_encoder_stamps_geometry_on_registered_nodes(self):
        from maxim.similarity.encoder import SensorEncoder

        ec = EntorhinalCortex(ECConfig(frozen_centroid_modalities=frozenset({"interoception"})))
        enc = SensorEncoder(ec=ec, atl=None, nac=None)
        enc.encode_sensors(
            agent_id="a", sensors={"hunger": 0.2}, modality="interoception", ranges={"hunger": (0.0, 1.0)}
        )
        assert ec._substrate_node_geometries, "no node carried a geometry tag — the D4 guard cannot fire"
        assert all(g for g in ec._substrate_node_geometries.values())

    def test_geometry_survives_save_load_round_trip(self, tmp_path):
        ec = EntorhinalCortex(ECConfig())
        ec.register_substrate_node("n1", [1.0, 0.0], "audio", geometry="gXYZ")
        path = tmp_path / "ec.json"
        ec.save(str(path))

        ec2 = EntorhinalCortex(ECConfig())
        ec2.load(str(path))
        assert ec2._substrate_node_geometries.get("n1") == "gXYZ"

    def test_geometry_survives_ingest_substrate_nodes(self):
        ec = EntorhinalCortex(ECConfig())
        ec.ingest_substrate_nodes({"n1": {"embedding": [1.0, 0.0], "modality": "audio", "geometry": "gXYZ"}})
        assert ec._substrate_node_geometries.get("n1") == "gXYZ"
