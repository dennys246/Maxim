"""Probe callable for the P4 mug test subprocess round-trip.

Used by ``test_p4_mug_test_roundtrip.py`` via the P3.5
``run_session_round_trip`` harness. The harness resolves this as
``"tests.substrate.p4_mug_probe:retrieve_cross_modal_snapshot"`` and
runs it BOTH pre-shutdown (in the parent) AND post-reload (in the
child subprocess after ``SessionSnapshot.restore_into``). The harness
compares the two results with exact equality — if the P4 sidecar or
episode binding graph fails to round-trip, the retrieval results
diverge and the test fails loudly.

The probe is intentionally simple: it runs one forward retrieval from
every text node in ``_node_modality`` against the vision bucket,
returns a sorted dict keyed by text node id with the result node_ids
sorted lexicographically. We deliberately DO NOT include float weights
(they can drift across platforms) AND we SORT the node-id list
(not ranked by weight) so the equality check isn't sensitive to
weight tie-break order under dict iteration.

**Why the list is sorted, not weight-ranked** (Round 2 Exec-lens #3
fold): retrieve_cross_modal ranks results by activation descending.
When multiple nodes tie at the same activation (e.g., 5 class-correct
vision nodes all reached through identical-weight 1-hop edges), the
tie-break order is determined by the ``DependencyGraph``'s internal
dict iteration order under ``activations.items()``. After a P3.5
``restore_into``, the binding graph is rebuilt from loaded episodes
via ``apply_hebbian_on_close``, which may insert edges in a different
order than the original build — so the iteration order can differ
even though the graph is semantically identical. Weight-ranking the
probe's output would flake on that tie-break order without
indicating a real correctness regression. Sorting by node id
instead captures "the binding graph round-trips" (the actual claim)
without false flakes from insertion-order drift.
"""

from __future__ import annotations

from typing import Any


def retrieve_cross_modal_snapshot(systems: dict[str, Any]) -> dict[str, list[str]]:
    """Run one retrieve_cross_modal call per text-tagged node.

    Returns a dict keyed by text node id, where each value is a
    lexicographically sorted list of the vision node ids retrieved.
    Deterministic given the hippocampus state — byte-identical
    pre-shutdown and post-reload when the round-trip works AND
    robust against weight tie-break order drift (see module docstring).
    """
    hippocampus = systems["hippocampus"]

    # Collect text node ids under the episode lock — mirrors the
    # snapshot-pattern filter in retrieve_cross_modal itself.
    with hippocampus._episode_lock:
        text_node_ids = sorted(
            node_id for node_id, modality in hippocampus._node_modality.items() if modality == "text"
        )

    result: dict[str, list[str]] = {}
    for text_id in text_node_ids:
        hits = hippocampus.retrieve_cross_modal(text_id, target_modality="vision", limit=50)
        # Sort node ids lexicographically (NOT by weight) to make the
        # equality check insensitive to tie-break order. See module
        # docstring for the rationale.
        result[text_id] = sorted(node_id for node_id, _weight in hits)
    return result
