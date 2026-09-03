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

from maxim.hivemind.merge import nac_merge  # noqa: E402

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

    def test_the_halves_OVERLAP_so_the_fold_is_load_bearing(self):
        """The correction the first D62 pass needed.

        A DISJOINT split makes the union the whole policy, so a plain
        `{**left, **right}` reproduces `nac_merge` bit-identically and the
        mean-fold on colliding keys — the semantics the `Re-run on: nac_merge
        semantics change` trigger is meant to watch — is never exercised.
        Overlapping halves make the fold decide the answer.
        """
        whole = self._policy()
        a, b = split_complementary(whole, sorted(self.BINS))
        ka, kb = set(a["cluster_reward_bias"]), set(b["cluster_reward_bias"])

        assert ka == kb == set(whole["cluster_reward_bias"]), "both halves must hold every key"
        differing = [k for k in ka if a["cluster_reward_bias"][k] != b["cluster_reward_bias"][k]]
        assert differing, "the halves must DISAGREE somewhere or the fold has nothing to resolve"

    def test_each_half_keeps_only_its_own_bins_learned(self):
        """Uses `sorted(bins)`, which is what `main()` actually passes — the
        previous version pinned an ordering production never produces."""
        a, b = split_complementary(self._policy(), sorted(self.BINS))

        def learned(state):
            return {k.split(NAC_KEY_SEP)[1] for k, v in state["cluster_reward_bias"].items() if v != 0.0}

        assert learned(a) == {"far_left", "far_right"}
        assert learned(b) == {"near_left", "near_right"}

    def test_the_naive_dict_update_does_NOT_reconstruct_the_policy(self):
        """The stub that matters, and the one omitted first time round.

        With overlap, a dict update lets the second half's 0.0 clobber the
        first half's learned value, so it cannot recover the argmax — whereas
        the real mean-fold averages 0.0 against +1.0 to +0.5, which still wins.
        """
        whole = self._policy()
        a, b = split_complementary(whole, sorted(self.BINS))
        naive = NOOP_MERGES["naive dict update"](a, b, left_source="l", right_source="r")
        real = nac_merge(a, b, left_source="l", right_source="r")

        assert naive["cluster_reward_bias"] != real["cluster_reward_bias"], (
            "a plain dict update reproduced the fold — the split is vacuous again"
        )

    def test_no_op_merges_cannot_recover_the_learned_values(self):
        """Every stub must lose information the real fold keeps."""
        whole = self._policy()
        a, b = split_complementary(whole, sorted(self.BINS))
        real = nac_merge(a, b, left_source="l", right_source="r")["cluster_reward_bias"]

        for name, stub in NOOP_MERGES.items():
            got = stub(a, b, left_source="l", right_source="r").get("cluster_reward_bias", {})
            assert got != real, f"no-op {name!r} reproduced the real fold — the split is vacuous"

    def test_the_real_merge_recovers_every_learned_argmax(self):
        """If the real fold ALSO failed, the arm would fail for the wrong reason."""
        whole = self._policy()
        a, b = split_complementary(whole, sorted(self.BINS))
        merged = nac_merge(a, b, left_source="half1", right_source="half2")["cluster_reward_bias"]

        for b_name in self.BINS:
            per_tool = {k.split(NAC_KEY_SEP)[2]: v for k, v in merged.items() if k.split(NAC_KEY_SEP)[1] == b_name}
            best = max(per_tool, key=lambda t: per_tool[t])
            expected = {
                "far_left": "tool:turn_left_big",
                "near_left": "tool:turn_left",
                "near_right": "tool:turn_right",
                "far_right": "tool:turn_right_big",
            }[b_name]
            assert best == expected, f"{b_name}: fold picked {best}, expected {expected}"
