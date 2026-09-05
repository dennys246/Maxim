"""D8 separation (1.2 gate 3): recall goes through pattern_complete_readonly.

The pre-registered D8 measurement (docs/experiments/protocols/
d8_read_mutation_preregistration.md, gated record 2026-09-05) returned
`separate-required`: one session-scale recall workload through the mutating
`pattern_complete_or_separate` moved centroids past the frozen 0.98 cosine
bound (min_cos 0.9521) and incremented member counts (+192), conflating query
traffic with observation evidence in merge weighting.

These tests pin the separation: `pattern_complete_readonly` matches like the
mutating path but is structurally incapable of writing (no centroid update,
no count increment, no first-touch geometry stamp), and `bio_enrichment`'s
recall path — the single production recall caller — routes through it and
leaves the EC bit-identical.
"""

from __future__ import annotations

from typing import Any

from maxim.integration.bio_enrichment import BioEnrichmentPipeline
from maxim.similarity.ec import EntorhinalCortex

GEOM = "linguistic:test:v1"


def _store() -> EntorhinalCortex:
    ec = EntorhinalCortex()
    ec.register_substrate_node("n-a", [1.0, 0.0, 0.0], "text", geometry=GEOM)
    ec.register_substrate_node("n-b", [0.0, 1.0, 0.0], "text", geometry=GEOM)
    return ec


def _snapshot(ec: EntorhinalCortex) -> dict[str, dict[str, Any]]:
    return {
        nid: {
            "embedding": list(ec.substrate_node_metadata(nid)["embedding"]),
            "member_count": ec.substrate_node_metadata(nid)["member_count"],
            "geometry": ec.substrate_node_metadata(nid)["geometry"],
        }
        for nid in list(ec._substrate_nodes.keys())  # noqa: SLF001 — id listing; values via public accessor
    }


class TestPatternCompleteReadonly:
    def test_matches_like_the_mutating_path(self):
        ec = _store()
        ro = ec.pattern_complete_readonly([0.9, 0.1, 0.0], "text", geometry=GEOM)
        mut = _store().pattern_complete_or_separate([0.9, 0.1, 0.0], "text", geometry=GEOM)
        assert not ro.is_new and not mut.is_new
        assert ro.node_id == mut.node_id == "n-a"
        assert ro.similarity == mut.similarity

    def test_repeated_readonly_recall_leaves_the_store_bit_identical(self):
        ec = _store()
        before = _snapshot(ec)
        for _ in range(50):
            r = ec.pattern_complete_readonly([0.9, 0.1, 0.0], "text", geometry=GEOM)
            assert r.node_id == "n-a"
        assert _snapshot(ec) == before  # embeddings, counts, geometry — all bit-identical

    def test_the_mutating_path_still_mutates(self):
        """Anti-vacuity: the same workload through the ENCODE path must move
        the store, or the readonly test above proves nothing."""
        ec = _store()
        before = _snapshot(ec)
        for _ in range(50):
            ec.pattern_complete_or_separate([0.9, 0.1, 0.0], "text", geometry=GEOM)
        after = _snapshot(ec)
        assert after["n-a"]["member_count"] > before["n-a"]["member_count"]
        assert after["n-a"]["embedding"] != before["n-a"]["embedding"]

    def test_no_match_registers_nothing(self):
        ec = _store()
        before = _snapshot(ec)
        r = ec.pattern_complete_readonly([0.0, 0.0, 1.0], "text", geometry=GEOM)
        assert r.is_new is True
        assert r.node_id not in before
        assert _snapshot(ec) == before

    def test_geometry_mask_respected(self):
        """A stored node in a different declared space never matches — it is
        incomparable, not less similar (gate 1 / D1 semantics, same as the
        mutating path)."""
        ec = _store()
        r = ec.pattern_complete_readonly([1.0, 0.0, 0.0], "text", geometry="linguistic:other:v2")
        assert r.is_new is True

    def test_no_first_touch_geometry_stamp(self):
        """The mutating path adopts the live tag on first match (gate 1's
        legacy-migration behavior). The READ path must not: stamping is a
        write."""
        ec = EntorhinalCortex()
        ec.register_substrate_node("legacy", [1.0, 0.0, 0.0], "text", geometry=None)
        r = ec.pattern_complete_readonly([1.0, 0.0, 0.0], "text", geometry=GEOM)
        assert r.node_id == "legacy"  # unstamped nodes are permissive, still match
        assert ec.substrate_node_metadata("legacy")["geometry"] is None


class _StubEncoder:
    """Deterministic embed + geometry, no model download."""

    def embed(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0]

    def geometry_for(self, embedding: list[float], modality: str) -> str:
        return GEOM


class _StubHippocampus:
    def retrieve_on_cue(self, node_id: str, limit: int = 5, multi_hop: bool = False) -> list:
        return []

    def recall(self, **kwargs: Any) -> list:
        return []


class TestBioEnrichmentRecallIsReadOnly:
    """The caller half: the single production recall caller leaves EC untouched."""

    def test_query_hippocampus_leaves_the_ec_bit_identical(self):
        ec = _store()
        # Spy so a silent early exit in the recall path cannot make the
        # "unchanged" assertion pass vacuously: the read-only completion must
        # actually be REACHED, once per query.
        calls = {"readonly": 0, "mutating": 0}
        real_readonly = ec.pattern_complete_readonly
        real_mutating = ec.pattern_complete_or_separate

        def spy_readonly(*args: Any, **kwargs: Any):
            calls["readonly"] += 1
            return real_readonly(*args, **kwargs)

        def spy_mutating(*args: Any, **kwargs: Any):
            calls["mutating"] += 1
            return real_mutating(*args, **kwargs)

        ec.pattern_complete_readonly = spy_readonly  # type: ignore[method-assign]
        ec.pattern_complete_or_separate = spy_mutating  # type: ignore[method-assign]

        pipeline = BioEnrichmentPipeline(
            hippocampus=_StubHippocampus(),
            ec=ec,
            encoder=_StubEncoder(),
        )
        before = _snapshot(ec)
        for _ in range(20):
            pipeline._query_hippocampus("the kettle is whistling", ["kettle"])
        assert calls["readonly"] == 20  # the recall path genuinely ran, via the READ method
        assert calls["mutating"] == 0  # and never through the ENCODE method
        assert _snapshot(ec) == before
