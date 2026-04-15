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
(not weights — float weights can drift across platforms and we want
exact comparison). This captures the structural correctness of the
binding graph + sidecar without false negatives from floating-point
noise.
"""

from __future__ import annotations

from typing import Any


def retrieve_cross_modal_snapshot(systems: dict[str, Any]) -> dict[str, list[str]]:
    """Run one retrieve_cross_modal call per text-tagged node.

    Returns a sorted dict keyed by text node id, where each value is
    the list of vision node ids retrieved in order. Deterministic given
    the hippocampus state — byte-identical pre-shutdown and post-reload
    when the round-trip works.
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
        result[text_id] = [node_id for node_id, _weight in hits]
    return result
