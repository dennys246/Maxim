"""Calibration sweep for ``semantic_shift_rule`` (v1_refinement P2).

Runs the rule across a small bundled conversational fixture at four
candidate thresholds {0.30, 0.40, 0.50, 0.60} and reports:

- false-positive count on a same-topic stream (expected: zero shifts —
  the conversation never leaves its topic, so the rule should not fire)
- true-positive count on a cross-topic stream (expected: one shift at
  each topic boundary, so 2 shifts across 3 topics)

The headline numbers from this script populate the calibration table in
``semantic_shift_rule``'s docstring. Re-run when the embedding model or
fixture changes.

Two execution surfaces:

1. ``python -m tests.calibration.p2_episode_boundary_calibration``
   (or ``pytest -s`` on this file) — prints the full sweep table to
   stdout for human review.
2. ``pytest tests/calibration/p2_episode_boundary_calibration.py`` —
   collects ``test_default_threshold_passes_calibration`` as a
   regression guard that asserts the published 0.40 default still
   passes the criteria used to choose it.

The fixture is hand-written and intentionally small (20 messages) so
the sweep is fast (~3s including model load) and the boundaries are
unambiguous to a human reviewer. A larger fixture would not change the
qualitative outcome — the threshold is chosen as the lowest value with
zero false positives on coherent conversation, and that property is
robust across fixture sizes.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np
import pytest


# ─────────────────────────────────────────────────────────────────────────
# Fixture: hand-crafted conversational data
# ─────────────────────────────────────────────────────────────────────────


SAME_TOPIC: list[str] = [
    "I'm trying to track down a flaky test in the Python codebase.",
    "The test passes locally but fails on CI about one in five runs.",
    "It looks like a race condition in the threading code.",
    "Adding a lock around the shared dict didn't help.",
    "Maybe the issue is in the test fixture rather than the code under test.",
    "I think pytest might be reusing state across tests in some way.",
    "Could be the autouse fixture not resetting properly.",
    "Let me add a print statement to see what state leaks across tests.",
]


CROSS_TOPIC: list[tuple[str, str]] = [
    ("python", "I'm trying to track down a flaky test in the Python codebase."),
    ("python", "The test passes locally but fails on CI about one in five runs."),
    ("python", "It looks like a race condition in the threading code."),
    ("python", "Adding a lock around the shared dict didn't help."),
    ("cooking", "What's the best way to cook pasta al dente?"),
    ("cooking", "I usually salt the water generously and time it carefully."),
    ("cooking", "Fresh pasta cooks much faster than dried — about three minutes."),
    ("cooking", "The trick is reserving some pasta water before draining."),
    ("hiking", "Have you done the trail up to Mount Tamalpais?"),
    ("hiking", "The west peak loop is about ten miles with some elevation gain."),
    ("hiking", "Bring layers — it can fog up suddenly in the afternoon."),
    ("hiking", "There's a great viewpoint about two miles in from the trailhead."),
]


# Boundary indices in CROSS_TOPIC where a topic shift occurs (0-indexed).
# Index i means: the embedding at position i belongs to a different topic
# than the embedding at position i-1.
CROSS_TOPIC_BOUNDARIES: list[int] = [4, 8]  # python→cooking at 4, cooking→hiking at 8


@dataclass(frozen=True)
class SweepResult:
    threshold: float
    same_topic_shifts: int  # false positives — should be 0
    cross_topic_shifts: int  # true positives — should be len(CROSS_TOPIC_BOUNDARIES) = 2
    cross_topic_shift_indices: tuple[int, ...]


# ─────────────────────────────────────────────────────────────────────────
# Sweep machinery
# ─────────────────────────────────────────────────────────────────────────


def _embed_corpus(texts: list[str]) -> list[np.ndarray]:
    """Embed each text with all-MiniLM-L6-v2 (the model the substrate
    path uses). Returns L2-normalised float32 vectors."""
    from sentence_transformers import SentenceTransformer  # local import — heavy dep

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return [np.asarray(e, dtype=np.float32) for e in embeddings]


def _replay_through_detector(
    embeddings: list[np.ndarray],
    threshold: float,
) -> list[int]:
    """Replay the embedding sequence through ``semantic_shift_rule`` with
    a fresh ``PendingEpisodeState`` and return the indices at which the
    rule fired.

    Each fire is treated as a "close + reopen" event: the centroid resets
    to the firing event's embedding (matching how Hippocampus would open
    a fresh pending episode after closing the old one). This mirrors the
    real wiring without requiring a full Hippocampus instance.
    """
    from maxim.memory.episode import (
        CaptureEvent,
        PendingEpisodeState,
        semantic_shift_rule,
    )

    rule = semantic_shift_rule(threshold=threshold)
    pending = PendingEpisodeState(id="cal_1", start_tick=0, last_tick=0, channel="text")
    fires: list[int] = []

    for i, emb in enumerate(embeddings):
        event = CaptureEvent(tick=i, channel="text", embedding=emb)
        if rule(pending, event):
            fires.append(i)
            # Simulate close + reopen: fresh pending state seeded by the
            # firing event.
            pending = PendingEpisodeState(
                id=f"cal_{len(fires) + 1}",
                start_tick=i,
                last_tick=i,
                channel="text",
            )
        pending.update_centroid(emb)
        pending.last_tick = i

    return fires


def run_sweep(
    thresholds: tuple[float, ...] = (0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90),
) -> list[SweepResult]:
    """Run the full sweep across both fixtures at every threshold."""
    same_embs = _embed_corpus(SAME_TOPIC)
    cross_embs = _embed_corpus([msg for _, msg in CROSS_TOPIC])

    results: list[SweepResult] = []
    for t in thresholds:
        same_fires = _replay_through_detector(same_embs, threshold=t)
        cross_fires = _replay_through_detector(cross_embs, threshold=t)
        results.append(
            SweepResult(
                threshold=t,
                same_topic_shifts=len(same_fires),
                cross_topic_shifts=len(cross_fires),
                cross_topic_shift_indices=tuple(cross_fires),
            )
        )
    return results


def _format_table(results: list[SweepResult]) -> str:
    expected_boundaries = tuple(CROSS_TOPIC_BOUNDARIES)
    lines = [
        "semantic_shift_rule calibration sweep",
        "=" * 78,
        f"Same-topic fixture: {len(SAME_TOPIC)} messages, no boundaries (expect 0 fires).",
        f"Cross-topic fixture: {len(CROSS_TOPIC)} messages, expected boundaries at {expected_boundaries}.",
        "",
        f"{'threshold':>10}  {'same(FP)':>10}  {'cross(TP)':>10}  cross_indices",
        "-" * 78,
    ]
    for r in results:
        lines.append(
            f"{r.threshold:>10.2f}  {r.same_topic_shifts:>10}  "
            f"{r.cross_topic_shifts:>10}  {list(r.cross_topic_shift_indices)}"
        )
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────
# Pytest regression guard for the published default
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.slow
def test_default_threshold_passes_calibration() -> None:
    """Regression guard: the published default must produce zero false
    positives on the same-topic fixture and fire at every expected
    cross-topic boundary.

    The published default lives in
    ``maxim.memory.episode.semantic_shift_rule``'s default argument; we
    pull it dynamically so the test breaks if someone changes the
    default without re-running the sweep.

    Marked ``slow`` because it loads sentence-transformers (~50 MB model
    download on first run). Skipped in the fast test suite; run via
    ``pytest -m slow tests/calibration/`` or as part of the periodic
    calibration sweep.
    """
    import inspect

    from maxim.memory.episode import semantic_shift_rule

    sig = inspect.signature(semantic_shift_rule)
    published_default = sig.parameters["threshold"].default
    assert isinstance(published_default, float), (
        f"semantic_shift_rule.threshold default must be a float, got {type(published_default).__name__}"
    )

    results = run_sweep(thresholds=(published_default,))
    assert len(results) == 1
    r = results[0]
    assert r.same_topic_shifts == 0, (
        f"Published default threshold {published_default} produced "
        f"{r.same_topic_shifts} false positives on the same-topic fixture. "
        f"Re-run the sweep to pick a higher threshold and update the docstring."
    )
    detected_expected = set(r.cross_topic_shift_indices) >= set(CROSS_TOPIC_BOUNDARIES)
    assert detected_expected, (
        f"Published default threshold {published_default} missed expected "
        f"boundaries {CROSS_TOPIC_BOUNDARIES} (fired at "
        f"{r.cross_topic_shift_indices}). Re-run the sweep to pick a lower "
        f"threshold and update the docstring."
    )


# ─────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    results = run_sweep()
    print(_format_table(results))
    sys.exit(0)
