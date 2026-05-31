"""Provenance + substrate domain on EC substrate nodes (Hivemind shareability, PR A).

v1_refinement.md §B5 adds parallel ``_substrate_node_sources`` and
``_substrate_node_domains`` dicts to ``EntorhinalCortex`` so each
substrate node carries shareability metadata. Stored alongside the
``(embedding, modality)`` tuple rather than extending the tuple shape
so existing tuple-unpacking call sites (encoder.py + sim_telemetry +
internal centroid-update path) stay stable.

These tests pin:

1. ``register_substrate_node`` defaults to ``source="local"`` /
   ``domain=None`` when called without the keyword-only kwargs (existing
   3 production callers in ``maxim.similarity.encoder`` continue to
   work unchanged).
2. Custom values via ``register_substrate_node(..., source=..., domain=...)``
   are stored and queryable via ``substrate_node_metadata``.
3. ``remove_substrate_node`` clears both parallel dicts so removed
   nodes don't leak source/domain entries.
4. ``save`` → ``load`` round-trip preserves source + domain per node.
5. Pre-B5 dumps that lack ``source``/``domain`` per-node keys load
   with the defaults applied (backward-compat).
"""

from __future__ import annotations

import json
from pathlib import Path

from maxim.similarity.ec import EntorhinalCortex


def _embedding(value: float = 1.0) -> list[float]:
    """Tiny stand-in embedding for tests — exact value irrelevant."""
    return [value, value, value]


# ─────────────────────────────────────────────────────────────────────────
# register_substrate_node — defaults
# ─────────────────────────────────────────────────────────────────────────


def test_register_defaults_source_local_domain_none() -> None:
    """Calling ``register_substrate_node`` without kwargs preserves the
    existing 3 production-call-site signatures and uses the new defaults.
    """
    ec = EntorhinalCortex()
    ec.register_substrate_node("node-1", _embedding(), "text")
    meta = ec.substrate_node_metadata("node-1")
    assert meta is not None
    assert meta["source"] == "local"
    assert meta["domain"] is None


# ─────────────────────────────────────────────────────────────────────────
# register_substrate_node — custom values
# ─────────────────────────────────────────────────────────────────────────


def test_register_with_custom_source_and_domain() -> None:
    """Keyword-only ``source`` + ``domain`` are stored on the node."""
    ec = EntorhinalCortex()
    ec.register_substrate_node(
        "node-1",
        _embedding(),
        "text",
        source="oasis-abc123",
        domain="combat",
    )
    meta = ec.substrate_node_metadata("node-1")
    assert meta is not None
    assert meta["source"] == "oasis-abc123"
    assert meta["domain"] == "combat"
    # Embedding + modality + count still readable (parallel-dict layout
    # didn't break the original surface).
    assert meta["embedding"] == _embedding()
    assert meta["modality"] == "text"
    assert meta["member_count"] == 1
    assert meta["node_id"] == "node-1"


def test_substrate_node_metadata_returns_none_for_unknown() -> None:
    """Querying an unregistered node returns ``None`` (not an empty dict)."""
    ec = EntorhinalCortex()
    assert ec.substrate_node_metadata("never-registered") is None


# ─────────────────────────────────────────────────────────────────────────
# remove_substrate_node — cleans both parallel dicts
# ─────────────────────────────────────────────────────────────────────────


def test_remove_substrate_node_clears_source_and_domain() -> None:
    """Removing a node clears ``_substrate_node_sources`` and
    ``_substrate_node_domains`` entries (no orphan metadata leaks).
    """
    ec = EntorhinalCortex()
    ec.register_substrate_node(
        "node-1",
        _embedding(),
        "text",
        source="oasis-1",
        domain="combat",
    )
    ec.remove_substrate_node("node-1")
    assert ec.substrate_node_metadata("node-1") is None
    # Parallel dicts emptied (no orphan keys).
    assert "node-1" not in ec._substrate_node_sources
    assert "node-1" not in ec._substrate_node_domains


# ─────────────────────────────────────────────────────────────────────────
# Persistence round-trip
# ─────────────────────────────────────────────────────────────────────────


def test_save_load_preserves_source_and_domain(tmp_path: Path) -> None:
    """A node saved with custom source+domain round-trips through save/load."""
    ec = EntorhinalCortex()
    ec.register_substrate_node(
        "node-1",
        _embedding(0.5),
        "text",
        source="oasis-xyz",
        domain="medical",
    )
    ec.register_substrate_node(
        "node-2",
        _embedding(0.7),
        "text",
        source="local",
        domain=None,
    )

    path = str(tmp_path / "ec.json")
    ec.save(path)

    ec2 = EntorhinalCortex()
    ec2.load(path)

    meta_1 = ec2.substrate_node_metadata("node-1")
    assert meta_1 is not None
    assert meta_1["source"] == "oasis-xyz"
    assert meta_1["domain"] == "medical"

    meta_2 = ec2.substrate_node_metadata("node-2")
    assert meta_2 is not None
    assert meta_2["source"] == "local"
    assert meta_2["domain"] is None


# ─────────────────────────────────────────────────────────────────────────
# Backward compatibility — pre-B5 dumps lack the new per-node keys
# ─────────────────────────────────────────────────────────────────────────


def test_load_pre_b5_dump_defaults_source_and_domain(tmp_path: Path) -> None:
    """A pre-B5 ``ec.json`` lacking per-node ``source``/``domain`` keys
    loads with ``source="local"`` and ``domain=None`` applied.

    The on-disk shape simulated here matches what every user's existing
    ``aut_ec.json`` looks like today: ``substrate_nodes[nid]`` only has
    ``embedding`` / ``modality`` / ``count``. The loader MUST fill in
    defaults rather than KeyError.
    """
    legacy_payload = {
        "_format_version": "1.0",
        "version": "1.0",
        "config": {
            "num_lsh_tables": 4,
            "bits_per_table": 8,
            "default_k": 10,
            "min_similarity": 0.3,
            "enable_semantic": False,
        },
        "lsh": {},
        "inverted": {},
        "signatures": {},
        "substrate_nodes": {
            "legacy-node": {
                "embedding": _embedding(),
                "modality": "text",
                "count": 3,
                # Note: no "source" or "domain" keys
            }
        },
    }
    path = tmp_path / "ec.json"
    path.write_text(json.dumps(legacy_payload))

    ec = EntorhinalCortex()
    ec.load(str(path))

    meta = ec.substrate_node_metadata("legacy-node")
    assert meta is not None
    assert meta["source"] == "local"
    assert meta["domain"] is None
    # Existing fields are still loaded correctly.
    assert meta["modality"] == "text"
    assert meta["member_count"] == 3


# ─────────────────────────────────────────────────────────────────────────
# Pattern-completion preserves provenance on an existing node
# ─────────────────────────────────────────────────────────────────────────


def test_pattern_completion_preserves_source_and_domain() -> None:
    """An oasis-sourced node receiving a local pattern-completion match keeps
    its origin attribution.

    Pre-merge review (Executor-lens NICE) flagged that the centroid-update
    path in ``pattern_complete_or_separate`` does not touch the parallel
    source/domain dicts. That is the intended semantics — a node imported
    from a contributor and subsequently reinforced by local embeddings
    stays attributed to its origin so the audit trail survives. This
    test pins the behavior before PR B starts depending on it.
    """
    ec = EntorhinalCortex()
    # Register at a non-default threshold-friendly axis so the next
    # embedding is similar enough to pattern-complete (text is not in
    # frozen_centroid_modalities — centroid would update).
    ec.register_substrate_node(
        "node-1",
        _embedding(1.0),
        "text",
        source="oasis-xyz",
        domain="combat",
    )
    # An embedding identical in direction guarantees cosine=1.0, which
    # is well above the 0.44 default threshold.
    result = ec.pattern_complete_or_separate(_embedding(0.9), "text")
    assert not result.is_new
    assert result.node_id == "node-1"

    meta = ec.substrate_node_metadata("node-1")
    assert meta is not None
    assert meta["source"] == "oasis-xyz"
    assert meta["domain"] == "combat"
    # Centroid update DID happen (member_count incremented) — confirms
    # we're exercising the running-mean code path, not the
    # frozen-prototype branch.
    assert meta["member_count"] == 2
