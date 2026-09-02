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

    def test_no_geometry_on_the_query_disables_the_check(self):
        """Callers that do not know their space are not punished for it."""
        ec = self._ec()
        ec.register_substrate_node("old", [1.0, 0.0], "audio", geometry="gOLD")

        res = ec.pattern_complete_or_separate([1.0, 0.0], "audio", threshold=0.5)
        assert not res.is_new

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
