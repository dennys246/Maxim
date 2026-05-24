"""Regression guard for the EC text-modality centroid drift fix.

Phase 3 of docs/plans/ec_centroid_drift_fix.md bumped
``ECConfig.pattern_complete_threshold`` from 0.40 to 0.44 to close the
sequential text-modality centroid drift surfaced by the 24 paraphrase-
collapse diagnostic. The 0.01 fine sweep in Phase 2
(docs/experiments/26_ec_drift_phase_2_regression.md) named 0.44 as the
strictly-dominant cell on both the P1 fixture and the Roy paraphrase
fixture.

These tests pin the structural invariants the fix depends on:

1. The new default value (0.44) — direct guard against drift in the
   default that the rest of the codebase depends on.
2. The coupled NAc constant — ``get_threshold_overrides`` has a
   hardcoded ``base = 0.44`` that should track the EC default per the
   documented "matches ECConfig.pattern_complete_threshold default"
   invariant. The test inspects the actual override math, not just
   the constant.
3. The frozen-centroid set — Phase 2 ruled out frozen-text-centroid as
   a fix variant (d0_f1_* cells hurt P1 paraphrase recall). The set
   should still contain only "interoception".

Production-realistic behavioral verification of the fix lives in:
  - tests/substrate/test_p1_recognition.py (P1 sweep — sentence-
    transformers paraphrase fixture, 10 shuffled seeds)
  - docs/experiments/results/ec_drift_phase_2_fine_sweep.json (the
    captured 0.01-granularity sweep on both P1 and Roy fixtures)

Synthetic drift sequences cannot reproduce the bug cleanly — the Roy
mega-collapse depends on real sentence embeddings sharing a common
"second-person body sensation" direction that's hard to construct from
random unit vectors. So this unit test pins config; behavioral tests
pin behavior.
"""

from __future__ import annotations

import math

from maxim.similarity.ec import ECConfig


# ─────────────────────────────────────────────────────────────────────────
# Default-value regression guards
# ─────────────────────────────────────────────────────────────────────────


def test_pattern_complete_threshold_default_is_0_44() -> None:
    """Pin the post-Phase-3 default.

    If this fails, someone changed the default without updating Phase 3's
    diagnostic numbers, the NAc coupled constant, and the lessons in
    CLAUDE.md. Re-read docs/plans/ec_centroid_drift_fix.md + the 24-26
    experiment docs before bumping.
    """
    assert ECConfig().pattern_complete_threshold == 0.44


def test_nac_threshold_override_base_tracks_ec_default() -> None:
    """The NAc ``get_threshold_overrides`` ``base`` constant has a
    hardcoded copy of the EC default at
    src/maxim/decisions/nac.py:get_threshold_overrides. The comment
    there says "matches ECConfig.pattern_complete_threshold default" —
    this test inspects the actual override math to verify the constants
    track. When a rewarded node has reward bias ``b``, the override
    should equal ``ECConfig().pattern_complete_threshold - b``.
    """
    from maxim.decisions.nac import NAc

    nac = NAc()
    agent_id = "test-agent"
    node_id = "node-1"
    # Bump the reward bias for one node above the 0.001 threshold so it
    # appears in the override dict.
    nac._reward_bias[(agent_id, node_id)] = 0.05  # type: ignore[attr-defined]

    overrides = nac.get_threshold_overrides(agent_id)
    assert node_id in overrides, "override missing for rewarded node"
    expected = ECConfig().pattern_complete_threshold - 0.05
    assert math.isclose(overrides[node_id], expected, abs_tol=1e-9), (
        f"NAc override base mismatched EC default: got {overrides[node_id]}, "
        f"expected ECConfig.pattern_complete_threshold "
        f"({ECConfig().pattern_complete_threshold}) - 0.05"
    )


def test_frozen_centroid_modalities_default_unchanged() -> None:
    """Phase 3 changed the threshold, NOT the frozen-centroid set.

    Phase 2 ruled out frozen-text-centroid as a fix variant
    (d0_f1_* cells hurt P1 paraphrase recall). The frozen set should
    still contain only "interoception".
    """
    assert ECConfig().frozen_centroid_modalities == frozenset({"interoception"})
