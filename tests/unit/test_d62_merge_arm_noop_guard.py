"""D62 — Exp 45's merge/fleet arm was a VACUOUS GUARD on an EARNED row.

`orient_merge_arm.py` is cited as the regression guard for Exp 45 arm 3 under a
`Re-run on: nac_merge semantics change` trigger. Measured 2026-09-01 on the real
recorded NACs, that trigger **could not fire**: the gauntlet passed with
`nac_merge` replaced by `return left` AND by `return right` — only `return {}`
failed — and the real merge was argmax-identical to `return left` in all four
bins. You could gut `nac_merge` and the guard stayed green.

Three vacuities compounded, and they need different fixes:

1. The gate read `correctness` ONLY. `magnitude_appropriateness` was computed
   and printed on every run and never consulted, which is why `return right`
   (direction-correct everywhere, argmax-wrong in both far bins) passed.
   **Fixed by gating both axes.**
2. Both parents already probe correctness 1.00, so `merged >= max(parents)` was
   evaluated at ceiling and carried zero information. No threshold fixes this:
   a parent that is already correct is, by itself, a passing policy.
   **Fixed by changing the INPUTS** — `--complementary-split`.
3. Both parents share `agent_id` and a hardcoded symbolic bin space, so D43
   cannot structurally fire. Out of scope here by design: this arm tests the
   FOLD. The alignment arm is `test_d44_merge_behavioural_delta.py`.

These tests pin (1) and (2) in CI, since the script itself needs recorded NAc
files that are not in the repo.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ARM_DIR = Path(__file__).resolve().parents[2] / "scripts" / "orient_backbone"
if str(_ARM_DIR) not in sys.path:
    sys.path.insert(0, str(_ARM_DIR))

orient_merge_arm = pytest.importorskip("orient_merge_arm")

NAC_KEY_SEP = orient_merge_arm.NAC_KEY_SEP
verdict = orient_merge_arm.verdict
split_complementary = orient_merge_arm.split_complementary
NOOP_MERGES = orient_merge_arm.NOOP_MERGES


def _probe(correctness: float, magnitude: float) -> dict:
    return {"correctness": correctness, "magnitude_appropriateness": magnitude}


class TestGateReadsBothAxes:
    """Vacuity 1: `magnitude_appropriateness` was printed but never gated."""

    def test_a_magnitude_regression_now_fails(self):
        """THE D62 CASE. `return right` on the recorded parents: direction
        correct everywhere (1.00) but magnitude 0.50 against a parent at 1.00.
        Under the old correctness-only gate this PASSED."""
        passed, why = verdict(_probe(1.0, 0.5), _probe(1.0, 1.0), _probe(1.0, 0.5), 1.0)
        assert not passed
        assert "magnitude" in why

    def test_a_correctness_regression_still_fails(self):
        passed, _ = verdict(_probe(0.5, 1.0), _probe(1.0, 1.0), _probe(1.0, 1.0), 1.0)
        assert not passed

    def test_a_genuine_improvement_on_both_axes_passes(self):
        passed, _ = verdict(_probe(1.0, 1.0), _probe(0.5, 0.5), _probe(0.5, 0.5), 1.0)
        assert passed

    def test_the_gate_cannot_be_satisfied_by_matching_only_the_weaker_parent(self):
        """`max(parents)` per axis, not per parent — a merge may not take the
        best correctness from one and the worst magnitude from the other."""
        passed, _ = verdict(_probe(1.0, 0.5), _probe(1.0, 0.5), _probe(0.5, 1.0), 1.0)
        assert not passed


class TestComplementarySplit:
    """Vacuity 2: the INPUTS, which no threshold can fix."""

    AGENT = "reachy"
    BINS = ("far_left", "near_left", "near_right", "far_right")

    def _policy(self) -> dict:
        correct = {
            "far_left": "tool:turn_left_big",
            "near_left": "tool:turn_left",
            "near_right": "tool:turn_right",
            "far_right": "tool:turn_right_big",
        }
        return {
            "cluster_reward_bias": {
                NAC_KEY_SEP.join((self.AGENT, b, tool)): (1.0 if tool == correct[b] else -0.3)
                for b in self.BINS
                for tool in ("tool:turn_left", "tool:turn_right", "tool:turn_left_big", "tool:turn_right_big")
            }
        }

    def test_each_half_is_disjoint_and_together_they_are_the_whole(self):
        whole = self._policy()
        a, b = split_complementary(whole, list(self.BINS))
        ka, kb = set(a["cluster_reward_bias"]), set(b["cluster_reward_bias"])

        assert ka and kb, "a split that empties a side cannot test a fold"
        assert not (ka & kb), "the halves must be disjoint or the merge has nothing to contribute"
        assert ka | kb == set(whole["cluster_reward_bias"]), "the union must be the original policy"

    def test_each_half_covers_only_its_own_bins(self):
        a, b = split_complementary(self._policy(), list(self.BINS))

        def bins_of(state):
            return {k.split(NAC_KEY_SEP)[1] for k in state["cluster_reward_bias"]}

        assert bins_of(a) == {"far_left", "near_left"}
        assert bins_of(b) == {"near_right", "far_right"}

    def test_no_op_merges_lose_half_the_policy(self):
        """The property that makes the split non-vacuous: every stub returns at
        most one half, so none of them can reconstruct the whole."""
        whole = self._policy()
        a, b = split_complementary(whole, list(self.BINS))
        full = set(whole["cluster_reward_bias"])

        for name, stub in NOOP_MERGES.items():
            got = set(stub(a, b, left_source="l", right_source="r").get("cluster_reward_bias", {}))
            assert got != full, f"no-op {name!r} reconstructed the whole policy — the split is vacuous"

    def test_the_real_merge_does_reconstruct_the_whole(self):
        """The other half of the same claim: if the real fold ALSO failed to
        recover the policy, the arm would fail for the wrong reason."""
        from maxim.hivemind.merge import nac_merge

        whole = self._policy()
        a, b = split_complementary(whole, list(self.BINS))
        merged = nac_merge(a, b, left_source="half1", right_source="half2")
        assert set(merged["cluster_reward_bias"]) == set(whole["cluster_reward_bias"])
