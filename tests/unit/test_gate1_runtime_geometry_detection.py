"""Gate 1 (1.1.3) — D1: `encoder_provenance` must detect a geometry change at RUNTIME.

> *D1 live encoder-provenance validation must reject or migrate incompatible
> state.* — roadmap, "Gates before 1.2"

D1's complaint was not that the stamp was wrong. It was that **nothing read
it**: `encoder_provenance` was recorded, persisted and reloaded, its only
readers were the hivemind bundle/CLI export, and `record_encoder_provenance`
*merges* divergence ("'mixed' is a finding, not an error"). So a geometry change
loaded old-geometry centroids and cosine-scanned them against new embeddings,
silently completing onto them.

This is gate 2's guard on the **live recall path** rather than the merge path.
`_cosine_similarity` already refuses a DIMENSION mismatch, which covers the
768-vs-384 `sentence-transformers` fallback across a load boundary. It cannot
see a **same-dimension** change — a place code adds sensor names, so the basis
set changes and the length does not.

The chosen remedy is SKIP-AND-WARN rather than raise: an incomparable node is
pattern-SEPARATED from (a new node is allocated) rather than completed onto,
which is the same failure mode `_cosine_similarity` already chose for a
dimension mismatch, and it keeps old files loadable. What was silent is now a
one-per-triple WARNING that says recall against those nodes is lost.
"""

from __future__ import annotations

import logging

import pytest

from maxim.similarity.ec import ECConfig, EntorhinalCortex
from maxim.similarity.encoder import SensorEncoder, encoding_geometry_tag


class TestRuntimeGeometryDetection:
    def _ec(self):
        return EntorhinalCortex(ECConfig(frozen_centroid_modalities=frozenset({"audio"})))

    def test_old_geometry_node_is_not_completed_onto(self):
        """THE D1 CASE. Same dim, same modality, cosine 1.0, different space.

        Before this gate the scan filtered on modality alone, so the stored
        node was a perfect match and the encode completed onto it.
        """
        ec = self._ec()
        ec.register_substrate_node("old", [1.0, 0.0], "audio", geometry="gOLD")

        res = ec.pattern_complete_or_separate([1.0, 0.0], "audio", threshold=0.5, geometry="gNEW")
        assert res.is_new, "completed onto a node from a different encoding space"
        assert res.node_id != "old"

    def test_same_geometry_still_completes(self):
        """The guard must not disable recall."""
        ec = self._ec()
        ec.register_substrate_node("n", [1.0, 0.0], "audio", geometry="gSAME")

        res = ec.pattern_complete_or_separate([1.0, 0.0], "audio", threshold=0.5, geometry="gSAME")
        assert not res.is_new
        assert res.node_id == "n"

    def test_unstamped_nodes_still_complete(self):
        """Legacy files must keep working — absence is unverifiable, not a mismatch."""
        ec = self._ec()
        ec.register_substrate_node("legacy", [1.0, 0.0], "audio")  # no geometry

        res = ec.pattern_complete_or_separate([1.0, 0.0], "audio", threshold=0.5, geometry="gNEW")
        assert not res.is_new

    def test_explicit_none_geometry_disables_the_check(self):
        """A caller that genuinely does not know its space can say so — but it
        must say so EXPLICITLY."""
        ec = self._ec()
        ec.register_substrate_node("old", [1.0, 0.0], "audio", geometry="gOLD")

        res = ec.pattern_complete_or_separate([1.0, 0.0], "audio", threshold=0.5, geometry=None)
        assert not res.is_new

    def test_omitting_geometry_is_a_TypeError_not_a_silent_bypass(self):
        """The structural half, and the reason the kwarg is required.

        `geometry: str | None = None` was optional, and its omission silently
        turned the guard off — CLAUDE.md's "silent no-op" shape. The review
        round found a live caller that had ALREADY omitted it
        (`bio_enrichment`'s text recall); because `"text"` is not a
        frozen-centroid modality the running-mean update fired and was actively
        CORRUPTING old centroids with incomparable vectors. Forgetting must
        fail loudly, per the `build_executor(pain_bus=..., permissions=None)` precedent.
        """
        ec = self._ec()
        with pytest.raises(TypeError, match="geometry"):
            ec.pattern_complete_or_separate([1.0, 0.0], "audio", threshold=0.5)  # type: ignore[call-arg]

    def test_the_mismatch_is_reported_and_deduped(self, caplog):
        """D1's actual complaint: nothing reported this at all.

        Deduped because the scan runs per node per encode — an undeduped
        warning is filtered out by whoever reads the logs, which is the same as
        not warning.
        """
        ec = self._ec()
        for i in range(5):
            ec.register_substrate_node(f"old{i}", [1.0, 0.0], "audio", geometry="gOLD")

        with caplog.at_level(logging.WARNING):
            for _ in range(3):
                ec.pattern_complete_or_separate([1.0, 0.0], "audio", threshold=0.5, geometry="gNEW")

        hits = [r for r in caplog.records if "geometry mismatch" in r.getMessage()]
        assert len(hits) == 1, f"expected exactly one deduped warning, got {len(hits)}"
        assert "gOLD" in hits[0].getMessage() and "gNEW" in hits[0].getMessage()

    def test_a_second_distinct_mismatch_is_reported_separately(self):
        """Dedup must be per-triple, not a single global latch that hides the
        second, different problem."""
        ec = self._ec()
        ec.register_substrate_node("a", [1.0, 0.0], "audio", geometry="gA")
        ec.register_substrate_node("b", [0.0, 1.0], "audio", geometry="gB")

        ec.pattern_complete_or_separate([1.0, 0.0], "audio", threshold=0.5, geometry="gNEW")
        assert len(ec._geometry_mismatch_seen) == 2


class TestProductionPathDetectsIt:
    """A guard nothing calls is a guard that never fires."""

    def test_sensor_encoder_passes_its_geometry_into_the_scan(self):
        """End to end: a body that GAINS a sensor must not recall onto the
        old-geometry nodes. This is D4's place-code scenario on the live path,
        and the dimension guard cannot see it — both encodings are dim 384.
        """
        ec = EntorhinalCortex(ECConfig(frozen_centroid_modalities=frozenset({"interoception"})))
        enc = SensorEncoder(ec=ec, atl=None, nac=None)

        enc.encode_sensors(
            agent_id="a", sensors={"hunger": 0.5}, modality="interoception", ranges={"hunger": (0.0, 1.0)}
        )
        before = dict(ec._substrate_nodes)
        assert len(before) == 1

        # Same agent, same modality — but the body now has a second sensor, so
        # `_sensor_embed` sums a different basis set at the SAME dimension.
        enc.encode_sensors(
            agent_id="a",
            sensors={"hunger": 0.5, "place_0": 0.0},
            modality="interoception",
            ranges={"hunger": (0.0, 1.0), "place_0": (0.0, 1.0)},
        )
        assert len(ec._substrate_nodes) == 2, "the new-geometry encode completed onto an old-geometry node"
        assert ec._geometry_mismatch_seen, "the divergence was not reported"

    def test_geometry_tags_differ_for_the_two_bodies(self):
        """Pins why the test above is a real scenario and not a fixture trick."""
        a = encoding_geometry_tag(
            encoder="sensor",
            modality="interoception",
            sensor_names=["hunger"],
            normalization="range",
            embedding_dim=384,
        )
        b = encoding_geometry_tag(
            encoder="sensor",
            modality="interoception",
            sensor_names=["hunger", "place_0"],
            normalization="range",
            embedding_dim=384,
        )
        assert a != b


class TestGeometryIsAPropertyOfTheSpaceNotTheReading:
    """The review round's blocker — found independently by TWO lenses.

    The tag was first keyed on `sorted(sensors.keys())`: the READING. But
    `agent_loop._read_drive_states` emits `cold` only while a thermal drive is
    outside its comfort band (`drives.setdefault("cold", cold_need)`), and
    `place_code` drops cells below an activation floor. So the key set is
    state-dependent, and a warm infant and a cold one hashed to DIFFERENT
    geometries: their interoception clusters became mutually unreachable, a
    contingency learned while warm was invisible the moment the body got cold —
    exactly when the corrective affordance should be salient — and the
    "encoder space changed" warning fired in BOTH directions on routine
    thermoregulation, training operators to filter the one message gate 1
    exists to deliver.

    The tag now names the DECLARED set (`ranges`, from the body walk), so a
    body change moves it and a state change does not.
    """

    def _enc(self):
        ec = EntorhinalCortex(ECConfig(frozen_centroid_modalities=frozenset({"interoception"})))
        return ec, SensorEncoder(ec=ec, atl=None, nac=None)

    def test_a_derived_need_appearing_does_not_change_the_geometry(self):
        ec, enc = self._enc()
        ranges = {"hunger": (0.0, 1.0)}  # the body declares hunger; `cold` is derived

        enc.encode_sensors(agent_id="a", sensors={"hunger": 0.8}, modality="interoception", ranges=ranges)
        enc.encode_sensors(agent_id="a", sensors={"hunger": 0.8, "cold": 0.4}, modality="interoception", ranges=ranges)

        tags = set(ec._substrate_node_geometries.values())
        assert len(tags) == 1, f"a state change forked the encoding space: {tags}"
        assert not ec._geometry_mismatch_seen, "routine thermoregulation reported as encoder-space corruption"

    def test_separation_still_works_across_that_boundary(self):
        """The guard must not have been bought by disabling pattern separation."""
        ec, enc = self._enc()
        ranges = {"hunger": (0.0, 1.0)}
        warm = enc.encode_sensors(agent_id="a", sensors={"hunger": 0.8}, modality="interoception", ranges=ranges)
        cold = enc.encode_sensors(
            agent_id="a", sensors={"hunger": 0.8, "cold": 0.4}, modality="interoception", ranges=ranges
        )
        assert warm != cold, "warm and cold must still be different CLUSTERS — just not different SPACES"

    def test_a_declared_sensor_appearing_DOES_change_the_geometry(self):
        """D4's real case must survive the fix: a place code adds declared sensors."""
        a = encoding_geometry_tag(
            encoder="sensor",
            modality="interoception",
            declared_sensors=["hunger"],
            normalization="range-aware",
            embedding_dim=384,
        )
        b = encoding_geometry_tag(
            encoder="sensor",
            modality="interoception",
            declared_sensors=["hunger", "place_0"],
            normalization="range-aware",
            embedding_dim=384,
        )
        assert a != b


class TestStampOnFirstTouch:
    """Without this, gate 1 is INERT for every installation that already exists.

    Every `ec.json` written before the geometry field is entirely unstamped,
    and completion never stamped — only `register_substrate_node`, only when
    `is_new`. So legacy nodes matched everything forever and never acquired an
    identity: the guard was permanently off while the operator believed it was
    running. The same hole let an unstamped node act as a permanent BRIDGE,
    uniting two geometries the direct gate refuses.
    """

    def test_a_legacy_node_adopts_the_live_tag_on_first_match(self):
        ec = EntorhinalCortex(ECConfig(frozen_centroid_modalities=frozenset({"audio"})))
        ec.register_substrate_node("legacy", [1.0, 0.0], "audio")  # unstamped

        res = ec.pattern_complete_or_separate([1.0, 0.0], "audio", threshold=0.5, geometry="gNEW")
        assert not res.is_new, "old files must keep working"
        assert ec._substrate_node_geometries["legacy"] == "gNEW", "the permissive branch never expired"

    def test_and_then_a_different_geometry_is_refused(self):
        ec = EntorhinalCortex(ECConfig(frozen_centroid_modalities=frozenset({"audio"})))
        ec.register_substrate_node("legacy", [1.0, 0.0], "audio")

        ec.pattern_complete_or_separate([1.0, 0.0], "audio", threshold=0.5, geometry="gNEW")
        res = ec.pattern_complete_or_separate([1.0, 0.0], "audio", threshold=0.5, geometry="gOTHER")
        assert res.is_new, "the guard still has no teeth after the stamp"

    def test_an_unstamped_node_cannot_bridge_two_geometries_in_a_merge(self):
        from maxim.hivemind.merge import ec_merge_aligned

        def node(vec, geom=None):
            n = {"embedding": vec, "modality": "audio", "count": 1, "source": "s"}
            if geom:
                n["geometry"] = geom
            return n

        left = {"U": node([1.0, 0.0])}
        right = {"A": node([1.0, 0.0], "gAAAA"), "B": node([1.0, 0.0], "gBBBB")}

        res = ec_merge_aligned(left, right, left_source="L", right_source="R")
        assert res.id_map["A"] != res.id_map["B"], "two refused geometries united through an unstamped node"
        assert res.nodes[res.id_map["A"]].get("geometry") == "gAAAA", "the survivor stayed a universal donor"


class TestD66LoadTimeMigration:
    """Gate 1's "reject **or migrate**" — the migrate half.

    1.1.3 shipped only reject, and the mismatch warning promised a remedy
    ("until they are re-encoded or migrated") that did not exist. This is that
    remedy, and it is deliberately PARTIAL — stamping only where the tag is
    genuinely recoverable, because **a wrong geometry tag is worse than a
    missing one**: a missing one is permissive and visible, a wrong one
    silently refuses matches that should succeed.
    """

    @staticmethod
    def _write_legacy(tmp_path, strip_declared: bool):
        """A saved EC with every geometry stamp stripped, as an old file has."""
        import json

        ec = EntorhinalCortex(ECConfig(frozen_centroid_modalities=frozenset({"interoception"})))
        enc = SensorEncoder(ec=ec, atl=None, nac=None)
        enc.encode_sensors(
            agent_id="a", sensors={"hunger": 0.8}, modality="interoception", ranges={"hunger": (0.0, 1.0)}
        )
        original = next(iter(ec._substrate_node_geometries.values()))
        path = tmp_path / "ec.json"
        ec.save(str(path))
        raw = json.loads(path.read_text())
        for n in raw["substrate_nodes"].values():
            n["geometry"] = None
        if strip_declared:
            for v in raw.get("encoder_provenance", {}).values():
                v.pop("declared_sensors", None)
        path.write_text(json.dumps(raw))
        return path, original

    def test_a_file_written_by_this_release_migrates_exactly(self, tmp_path):
        path, original = self._write_legacy(tmp_path, strip_declared=False)

        ec = EntorhinalCortex(ECConfig())
        ec.load(str(path))
        got = next(iter(ec._substrate_node_geometries.values()))
        assert got == original, "migrated tag must reconstruct EXACTLY, or it silently refuses real matches"

    def test_a_file_without_declared_sensors_is_left_alone(self, tmp_path):
        """The honest limit. The tag hashes the DECLARED set; older files
        recorded `sensor_names` — the READING, accumulated as a union across a
        session — which is a different quantity that no care recovers. A file
        that saw `cold` appear once has it in the union forever.
        """
        path, _ = self._write_legacy(tmp_path, strip_declared=True)

        ec = EntorhinalCortex(ECConfig())
        ec.load(str(path))
        assert all(g is None for g in ec._substrate_node_geometries.values()), (
            "guessed a geometry tag it could not derive — a wrong tag is worse than none"
        )

    def test_a_mixed_normalization_session_is_not_guessed(self, tmp_path):
        """A file whose session mixed range-aware and range-blind calls is
        itself the finding: its nodes are not all in one space, so there is no
        single correct tag."""
        import json

        path, _ = self._write_legacy(tmp_path, strip_declared=False)
        raw = json.loads(path.read_text())
        for v in raw["encoder_provenance"].values():
            v["normalization_modes"] = ["range-aware", "range-blind"]
        path.write_text(json.dumps(raw))

        ec = EntorhinalCortex(ECConfig())
        ec.load(str(path))
        assert all(g is None for g in ec._substrate_node_geometries.values())

    def test_migration_runs_after_provenance_loads(self, tmp_path):
        """Ordering guard. The migration DERIVES tags from
        `encoder_provenance`; running it before that dict is populated
        migrates nothing while reporting that nothing could be migrated — a
        failure that looks exactly like the honest refusal above. This was a
        real bug during implementation.
        """
        path, original = self._write_legacy(tmp_path, strip_declared=False)
        ec = EntorhinalCortex(ECConfig())
        ec.load(str(path))
        assert ec._encoder_provenance, "provenance must be loaded before the migration reads it"
        assert next(iter(ec._substrate_node_geometries.values())) == original
