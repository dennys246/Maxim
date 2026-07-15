"""Regression guard for the EC text-modality centroid drift fix.

Phase 3 of docs/plans/archive/ec_centroid_drift_fix.md bumped
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
    CLAUDE.md. Re-read docs/plans/archive/ec_centroid_drift_fix.md + the 24-26
    experiment docs before bumping.
    """
    assert ECConfig().pattern_complete_threshold == 0.44


def test_nac_threshold_override_base_tracks_ec_default() -> None:
    """The NAc ``get_threshold_overrides`` fallback constant (used
    when ``base_threshold`` is not passed) has a hardcoded copy of the
    EC default at src/maxim/decisions/nac.py:get_threshold_overrides.
    Phase 3.5 parameterized the method — production callers
    (LinguisticEncoder) pass ``ec.config.pattern_complete_threshold``
    so the override always tracks the LIVE EC threshold, but the
    fallback is kept for legacy callers that lack an EC reference.
    This test verifies the fallback (None → 0.44) still matches the
    EC default. When a rewarded node has reward bias ``b``, the
    fallback override should equal ``ECConfig().pattern_complete_threshold - b``.
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
        f"NAc override fallback base mismatched EC default: got {overrides[node_id]}, "
        f"expected ECConfig.pattern_complete_threshold "
        f"({ECConfig().pattern_complete_threshold}) - 0.05"
    )


def test_nac_threshold_override_accepts_base_threshold_parameter() -> None:
    """Phase 3.5: ``get_threshold_overrides`` accepts a
    ``base_threshold`` keyword argument. When passed, the override
    formula is ``base_threshold - reward_bias`` instead of the
    hardcoded fallback. Production callers (LinguisticEncoder at
    src/maxim/similarity/encoder.py:_get_reward_overrides + line 262)
    pass ``self.ec.config.pattern_complete_threshold`` so the override
    tracks the live EC threshold, not the hardcoded fallback. This
    matters at non-default EC thresholds — e.g., P2 validation sweep
    uses 0.80 and would otherwise get a 0.46-cosine-units recognition
    radius widening artifact from the pre-Phase-3.5 hardcoded base.
    """
    from maxim.decisions.nac import NAc

    nac = NAc()
    agent_id = "test-agent"
    node_id = "node-1"
    nac._reward_bias[(agent_id, node_id)] = 0.05  # type: ignore[attr-defined]

    # Pass a non-default base_threshold — override should use it.
    overrides = nac.get_threshold_overrides(agent_id, base_threshold=0.80)
    assert math.isclose(overrides[node_id], 0.75, abs_tol=1e-9), (
        f"base_threshold=0.80, bias=0.05 → expected 0.75, got {overrides[node_id]}"
    )

    # Pass base_threshold equal to EC default — same as fallback path.
    ec_default = ECConfig().pattern_complete_threshold
    overrides_default = nac.get_threshold_overrides(agent_id, base_threshold=ec_default)
    assert math.isclose(overrides_default[node_id], ec_default - 0.05, abs_tol=1e-9)


def test_nac_threshold_override_clamps_to_floor_at_high_bias() -> None:
    """Clamp floor of 0.10 prevents degenerate matching even at extreme
    bias. Verified independently from the base parameterization
    because the clamp is a separate invariant in the override formula.
    """
    from maxim.decisions.nac import NAc

    nac = NAc()
    agent_id = "test-agent"
    node_id = "node-1"
    # Bias of 0.40 with default base 0.44 would yield 0.04 — should clamp.
    nac._reward_bias[(agent_id, node_id)] = 0.40  # type: ignore[attr-defined]
    overrides = nac.get_threshold_overrides(agent_id)
    assert math.isclose(overrides[node_id], 0.10, abs_tol=1e-9), (
        f"high-bias override should clamp to floor 0.10, got {overrides[node_id]}"
    )


def test_frozen_centroid_modalities_default() -> None:
    """Frozen-centroid set = interoception + audio.

    Phase 2 ruled out frozen-text-centroid as a fix variant
    (d0_f1_* cells hurt P1 paraphrase recall), so "text"/"vision" stay
    drifting. "audio" was added (perception_pipeline_placement.md Q5,
    2026-06-27) because a densely-streamed continuous sound-localization
    signal would walk a running-mean centroid into the same collapse
    interoception suffers — exteroceptive localization nodes are
    frozen-prototype for stable per-direction clusters.
    """
    assert ECConfig().frozen_centroid_modalities == frozenset({"interoception", "audio"})
