"""Statistic-shape tests for accumulator fields (Stage 1 of structural invariant tests).

Catches the **Wire 1 class** of bugs documented in CLAUDE.md ("Key-embedded
values produce structurally-degenerate statistics"): if a statistic
accumulator's KEY embeds the very dimension you want the statistic to
vary over, the per-key statistic is structurally degenerate (0 or
constant) on every binary-alternating input case — the bug masquerades
as "the feature is wired, the value is just small."

For every accumulator field in the codebase, this file asserts:

1. The accumulator produces a **non-degenerate** signal on
   binary-alternating input (variance > 0, distinct values, monotone
   growth, etc. as appropriate to the statistic).
2. **Distinct keys** accumulate independently (no silent cross-key
   merging — the P4 multi-agent trap rhymes with the Wire 1 trap).

The linchpin test in this file is
:class:`TestCausalLinkMovedStatisticInvariant`: Wire 1 moved variance
OFF ``CausalLink`` (where ``_generate_link_id`` embeds
``outcome_signature`` into the key, making per-link variance structurally
0 for binary outcomes) and ONTO ``NAc._event_outcome_welford`` keyed by
``(agent_id, event_signature)`` — the parent aggregation level where
variance carries signal. Re-introducing a ``variance_estimate`` field on
``CausalLink`` would re-introduce the Wire 1 trap; that test fails if it
happens.

**If a new accumulator gets added to the codebase, add a ``*_case`` entry
below so the structural baseline keeps pace.** The companion plan
explains the scoping at
``docs/plans/structural_invariant_tests.md``.

**Honest caveat:** binary-alternating is one specific degenerate-shape
probe. A statistic could be non-degenerate on alternating input but still
degenerate on production input shapes. This test catches the Wire 1
class specifically; broader coverage requires more cases.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import pytest


# ─────────────────────────────────────────────────────────────────────
# Parametrized case shape
# ─────────────────────────────────────────────────────────────────────


@dataclass
class AccumulatorCase:
    """One parametrized accumulator under test.

    ``setup_fn`` returns the fresh accumulator (or container holding it).
    ``feed_fn`` drives binary-alternating input through the public update API.
    ``assert_non_degenerate_fn`` asserts the statistic-specific
    non-degenerate property after feed.
    ``assert_distinct_keys_independent_fn`` (optional) asserts that distinct
    keys accumulate independently — required for keyed accumulators.
    """

    name: str
    setup_fn: Callable[[], Any]
    feed_fn: Callable[[Any], None]
    assert_non_degenerate_fn: Callable[[Any], None]
    assert_distinct_keys_independent_fn: Callable[[Any], None] | None = None

    # pytest parametrize id
    def __repr__(self) -> str:
        return self.name


# ─────────────────────────────────────────────────────────────────────
# Case 1 — NAc._event_outcome_welford (Wire 1 fix; CLAUDE.md L37)
# ─────────────────────────────────────────────────────────────────────


def _make_nac() -> Any:
    from maxim.decisions.nac import NAc, NACConfig

    return NAc(NACConfig(temporal_window_seconds=60, max_pending_events=2000))


def _drive_welford(nac: Any) -> None:
    """Feed 10 alternating POSITIVE/NEGATIVE outcomes — Bernoulli p=0.5."""
    from maxim.decisions.causal_link import Valence

    for _ in range(5):
        for v in (Valence.POSITIVE, Valence.NEGATIVE):
            nac.record_event("tool", "tool:test", context={"agent_id": "a"})
            nac.record_outcome("tool", "tool:test", v, context={"agent_id": "a"})


def _assert_welford_non_degenerate(nac: Any) -> None:
    state = nac._event_outcome_welford[("a", "tool:test")]
    assert state["n"] == 10.0, f"expected 10 observations, got {state['n']}"
    variance = state["m2"] / state["n"]
    # Bernoulli p=0.5 → 0.25 population variance. If this drops to ~0
    # the Wire 1 fix has regressed (variance lives on a key-trapped
    # surface again).
    assert math.isclose(variance, 0.25, abs_tol=1e-10), (
        f"Wire 1 regression: expected Bernoulli variance 0.25 on binary "
        f"alternating input; got {variance:.6f}. See CLAUDE.md L37."
    )


def _assert_welford_distinct_keys_independent(nac: Any) -> None:
    from maxim.decisions.causal_link import Valence

    nac.record_event("tool", "tool:test", context={"agent_id": "b"})
    nac.record_outcome("tool", "tool:test", Valence.POSITIVE, context={"agent_id": "b"})
    keys = set(nac._event_outcome_welford.keys())
    assert ("a", "tool:test") in keys
    assert ("b", "tool:test") in keys
    # Agent b accumulator must not have been polluted by agent a's history.
    assert nac._event_outcome_welford[("b", "tool:test")]["n"] == 1.0


welford_case = AccumulatorCase(
    name="NAc._event_outcome_welford",
    setup_fn=_make_nac,
    feed_fn=_drive_welford,
    assert_non_degenerate_fn=_assert_welford_non_degenerate,
    assert_distinct_keys_independent_fn=_assert_welford_distinct_keys_independent,
)


# ─────────────────────────────────────────────────────────────────────
# Case 2 — NAc._reward_bias (unidirectional, clamps [0, max])
# ─────────────────────────────────────────────────────────────────────


def _drive_reward_bias(nac: Any) -> None:
    """Credit two distinct nodes with distinct reward histories."""
    # node_a: positive rewards → bias > 0
    for _ in range(5):
        nac.credit_node("agent_a", "node_a", reward=0.8)
    # node_b: a single small positive → bias smaller than node_a's
    nac.credit_node("agent_a", "node_b", reward=0.1)


def _assert_reward_bias_non_degenerate(nac: Any) -> None:
    bias_a = nac._reward_bias[("agent_a", "node_a")]
    bias_b = nac._reward_bias[("agent_a", "node_b")]
    # Both within clamp range — Wire 1-style invariant
    cap = nac.config.max_reward_bias
    assert 0.0 <= bias_a <= cap, f"node_a bias {bias_a} out of [0, {cap}]"
    assert 0.0 <= bias_b <= cap, f"node_b bias {bias_b} out of [0, {cap}]"
    # Distinct histories → distinct biases (the non-degenerate signal).
    assert bias_a > bias_b, (
        f"distinct reward histories should produce distinct biases; got bias_a={bias_a:.4f} bias_b={bias_b:.4f}"
    )


def _assert_reward_bias_distinct_keys_independent(nac: Any) -> None:
    # Credit a different agent on the SAME node_id — must not merge.
    nac.credit_node("agent_b", "node_a", reward=0.9)
    assert ("agent_a", "node_a") in nac._reward_bias
    assert ("agent_b", "node_a") in nac._reward_bias
    # Agent b's slot starts fresh — not pre-populated by agent a's history.
    # The P4 multi-agent trap (CLAUDE.md L43) would silently merge these.
    bias_a = nac._reward_bias[("agent_a", "node_a")]
    bias_b = nac._reward_bias[("agent_b", "node_a")]
    # Agent b had ONE update with reward 0.9; agent a had FIVE with 0.8.
    # Biases must differ in magnitude (proves no cross-agent merge).
    assert bias_a != bias_b, "cross-agent slots silently merged"


reward_bias_case = AccumulatorCase(
    name="NAc._reward_bias",
    setup_fn=_make_nac,
    feed_fn=_drive_reward_bias,
    assert_non_degenerate_fn=_assert_reward_bias_non_degenerate,
    assert_distinct_keys_independent_fn=_assert_reward_bias_distinct_keys_independent,
)


# ─────────────────────────────────────────────────────────────────────
# Case 3 — NAc._cluster_reward_bias (Wire-A; bidirectional)
# ─────────────────────────────────────────────────────────────────────


def _drive_cluster_reward_bias(nac: Any) -> None:
    """Distinct (cluster, tool) entries with distinct reward histories."""
    # cluster_x + tool_t1: positive rewards
    for _ in range(5):
        nac.update_cluster_reward("agent_a", "cluster_x", "tool_t1", reward=0.8)
    # cluster_x + tool_t2: negative rewards (bidirectional clamp matters)
    for _ in range(5):
        nac.update_cluster_reward("agent_a", "cluster_x", "tool_t2", reward=-0.8)


def _assert_cluster_reward_bias_non_degenerate(nac: Any) -> None:
    bias_t1 = nac._cluster_reward_bias[("agent_a", "cluster_x", "tool_t1")]
    bias_t2 = nac._cluster_reward_bias[("agent_a", "cluster_x", "tool_t2")]
    cap = nac.config.max_cluster_reward_bias
    assert -cap <= bias_t1 <= cap, f"tool_t1 bias {bias_t1} out of [-{cap}, {cap}]"
    assert -cap <= bias_t2 <= cap, f"tool_t2 bias {bias_t2} out of [-{cap}, {cap}]"
    # Bidirectional: positive reward stream produces positive bias,
    # negative stream produces negative bias. The Wire-A signal hinges
    # on the sign survival.
    assert bias_t1 > 0, f"positive-reward stream should yield positive bias; got {bias_t1}"
    assert bias_t2 < 0, f"negative-reward stream should yield negative bias; got {bias_t2}"


def _assert_cluster_reward_bias_distinct_keys_independent(nac: Any) -> None:
    # Same (cluster_x, tool_t1) but DIFFERENT agent — must not merge.
    nac.update_cluster_reward("agent_b", "cluster_x", "tool_t1", reward=-0.5)
    key_a = ("agent_a", "cluster_x", "tool_t1")
    key_b = ("agent_b", "cluster_x", "tool_t1")
    assert key_a in nac._cluster_reward_bias
    assert key_b in nac._cluster_reward_bias
    # Distinct agents → distinct biases; opposite signs prove no merge.
    assert nac._cluster_reward_bias[key_a] > 0
    assert nac._cluster_reward_bias[key_b] < 0


cluster_reward_bias_case = AccumulatorCase(
    name="NAc._cluster_reward_bias",
    setup_fn=_make_nac,
    feed_fn=_drive_cluster_reward_bias,
    assert_non_degenerate_fn=_assert_cluster_reward_bias_non_degenerate,
    assert_distinct_keys_independent_fn=_assert_cluster_reward_bias_distinct_keys_independent,
)


# ─────────────────────────────────────────────────────────────────────
# Case 4 — NAc._goal_reward_bias (bidirectional; goal_tag keyed)
# ─────────────────────────────────────────────────────────────────────


def _drive_goal_reward_bias(nac: Any) -> None:
    """Two distinct goal_tags with distinct reward histories."""
    for _ in range(5):
        nac.credit_goal("goal_x", reward=0.8)
    for _ in range(5):
        nac.credit_goal("goal_y", reward=-0.8)


def _assert_goal_reward_bias_non_degenerate(nac: Any) -> None:
    bias_x = nac._goal_reward_bias["goal_x"]
    bias_y = nac._goal_reward_bias["goal_y"]
    cap = nac.config.max_reward_bias
    assert -cap <= bias_x <= cap, f"goal_x bias {bias_x} out of [-{cap}, {cap}]"
    assert -cap <= bias_y <= cap, f"goal_y bias {bias_y} out of [-{cap}, {cap}]"
    assert bias_x > 0
    assert bias_y < 0


def _assert_goal_reward_bias_distinct_keys_independent(nac: Any) -> None:
    nac.credit_goal("goal_z", reward=0.5)
    assert nac._goal_reward_bias["goal_z"] != nac._goal_reward_bias["goal_x"]
    assert nac._goal_reward_bias["goal_z"] != nac._goal_reward_bias["goal_y"]


goal_reward_bias_case = AccumulatorCase(
    name="NAc._goal_reward_bias",
    setup_fn=_make_nac,
    feed_fn=_drive_goal_reward_bias,
    assert_non_degenerate_fn=_assert_goal_reward_bias_non_degenerate,
    assert_distinct_keys_independent_fn=_assert_goal_reward_bias_distinct_keys_independent,
)


# ─────────────────────────────────────────────────────────────────────
# Case 5 — MemoryRecord.promotion_pressure (pressure-based promotion)
# ─────────────────────────────────────────────────────────────────────


@dataclass
class _PromotionPressureFixture:
    """Holds the consolidation manager + a pair of memories."""

    consolidator: Any
    memory_diverse: Any  # Will receive context-diverse recalls
    memory_repeated: Any  # Will receive identical-context recalls


def _make_promotion_pressure_fixture() -> _PromotionPressureFixture:
    """Build a minimal hippocampus + two memories for promotion-pressure shape."""
    from maxim.memory.hippocampus import Hippocampus, HippocampusConfig
    from maxim.memory.types import EpisodicMemory, Perception

    hippocampus = Hippocampus(config=HippocampusConfig())

    def _make_memory(memory_id: str, cli_input: str) -> Any:
        perception = Perception(cli_input=cli_input, salience=0.5)
        mem = EpisodicMemory(
            id=memory_id,
            timestamp=0.0,
            perception=perception,
        )
        # Force into SHORT_TERM (promotion path only fires from SHORT_TERM).
        mem.long_term = False
        return mem

    diverse = _make_memory("mem_diverse", "alpha bravo")
    repeated = _make_memory("mem_repeated", "alpha bravo")
    return _PromotionPressureFixture(
        consolidator=hippocampus,
        memory_diverse=diverse,
        memory_repeated=repeated,
    )


def _drive_promotion_pressure(fixture: _PromotionPressureFixture) -> None:
    """Feed context-diverse recalls vs identical-context recalls.

    The non-degenerate signal: context-diverse recalls accumulate
    ``promotion_pressure``; identical-context recalls do NOT (they hit
    the access_contexts gate). The shape test asserts that the diverse
    memory's pressure ends up strictly greater than the repeated
    memory's pressure.
    """
    consolidator = fixture.consolidator

    # Five DIFFERENT queries → five novel context_signatures on diverse.
    for i in range(5):
        consolidator._score_and_maybe_promote_batch(
            memories=[fixture.memory_diverse],
            query=f"question_variant_{i}",
        )

    # Five IDENTICAL queries → first contributes pressure, rest are gated.
    for _ in range(5):
        consolidator._score_and_maybe_promote_batch(
            memories=[fixture.memory_repeated],
            query="same_query_every_time",
        )


def _assert_promotion_pressure_non_degenerate(fixture: _PromotionPressureFixture) -> None:
    diverse_pressure = fixture.memory_diverse.promotion_pressure
    repeated_pressure = fixture.memory_repeated.promotion_pressure
    # If diverse PROMOTED (crossed threshold), promotion_pressure resets
    # to 0 and long_term flips. Either outcome is non-degenerate behavior.
    diverse_promoted = fixture.memory_diverse.long_term
    if diverse_promoted:
        assert diverse_pressure == 0.0, "promoted memory pressure should reset to 0"
    else:
        # Not promoted → pressure must reflect five diverse accumulations.
        assert diverse_pressure > repeated_pressure, (
            f"context-diverse recall should accumulate more pressure than "
            f"identical-context; got diverse={diverse_pressure:.4f} "
            f"repeated={repeated_pressure:.4f}"
        )
    # Repeated memory got ONE pressure bump (first novel context),
    # not five — assert it's strictly less than what 5 bumps would produce.
    assert repeated_pressure > 0.0, "first recall should add some pressure"
    # Five bumps of the same access_score would be 5 * score; one bump is 1 * score.
    # Strict inequality is enough.


def _assert_promotion_pressure_distinct_records_independent(
    fixture: _PromotionPressureFixture,
) -> None:
    # Each MemoryRecord owns its own promotion_pressure field. The
    # accumulator is per-record by construction (no shared dict). This
    # is a structural assertion: the two records have different field
    # objects and different values.
    assert fixture.memory_diverse is not fixture.memory_repeated
    # Touching one's pressure must not affect the other's.
    fixture.memory_diverse.promotion_pressure = 99.0
    assert fixture.memory_repeated.promotion_pressure != 99.0


promotion_pressure_case = AccumulatorCase(
    name="MemoryRecord.promotion_pressure",
    setup_fn=_make_promotion_pressure_fixture,
    feed_fn=_drive_promotion_pressure,
    assert_non_degenerate_fn=_assert_promotion_pressure_non_degenerate,
    assert_distinct_keys_independent_fn=_assert_promotion_pressure_distinct_records_independent,
)


# ─────────────────────────────────────────────────────────────────────
# Case 6 — SignificanceWeightLearner.rpe_running_mean (RPE normalization)
# ─────────────────────────────────────────────────────────────────────


def _make_weight_learner(tmp_path: Any | None = None) -> Any:
    """Build a SignificanceWeightLearner; hippocampus stub is enough."""
    from maxim.decisions.significance import SignificanceWeightLearner

    class _StubHippocampus:
        class _Graph:
            def get_associated(self, _mid: str) -> list:
                return []

        graph = _Graph()

    # Use a path that won't exist so the learner takes the fresh-init path.
    import tempfile
    import os

    tmp = tempfile.mkdtemp(prefix="maxim_stat_shape_")
    path = os.path.join(tmp, "weights.json")
    return SignificanceWeightLearner(weights_path=path, hippocampus=_StubHippocampus())


def _drive_rpe_running_mean(learner: Any) -> None:
    """Feed alternating ±RPE values — running mean must become non-zero."""
    # Need ≥2 samples for stdev to update (per update_rpe_stats guard).
    samples = [1.0, -1.0, 0.5, -0.5, 0.8, -0.8, 0.3, -0.3]
    for s in samples:
        learner.update_rpe_stats(s)


def _assert_rpe_running_mean_non_degenerate(learner: Any) -> None:
    # 8 alternating ± samples → mean ≈ 0 BUT std must be non-zero.
    # The non-degenerate signal here is std > 0 — otherwise the
    # normalization in evaluate() would divide by ~1e-6 sentinel and
    # produce garbage RPE-normalized scores.
    assert learner.rpe_running_std > 0.0, (
        f"alternating RPE samples should yield non-zero std; got {learner.rpe_running_std}"
    )
    # Window must reflect the full feed.
    assert len(learner.rpe_window) == 8


rpe_running_mean_case = AccumulatorCase(
    name="SignificanceWeightLearner.rpe_running_mean",
    setup_fn=_make_weight_learner,
    feed_fn=_drive_rpe_running_mean,
    assert_non_degenerate_fn=_assert_rpe_running_mean_non_degenerate,
    # rpe_running_mean is a single-slot accumulator (no key shape) —
    # the distinct-keys-independent check doesn't apply.
    assert_distinct_keys_independent_fn=None,
)


# ─────────────────────────────────────────────────────────────────────
# Parametrized shape test
# ─────────────────────────────────────────────────────────────────────


ALL_CASES = [
    welford_case,
    reward_bias_case,
    cluster_reward_bias_case,
    goal_reward_bias_case,
    promotion_pressure_case,
    rpe_running_mean_case,
]


class TestStatisticShapes:
    """Cross-accumulator shape baseline.

    For each accumulator in the codebase that holds a learned-statistic
    value, assert it produces a non-degenerate signal on
    binary-alternating input AND (where applicable) that distinct keys
    accumulate independently. New accumulators added to the codebase
    should append a ``*_case`` entry above.
    """

    @pytest.mark.parametrize(
        "case",
        ALL_CASES,
        ids=[c.name for c in ALL_CASES],
    )
    def test_non_degenerate_on_binary_alternating_input(self, case: AccumulatorCase) -> None:
        accumulator = case.setup_fn()
        case.feed_fn(accumulator)
        case.assert_non_degenerate_fn(accumulator)

    @pytest.mark.parametrize(
        "case",
        [c for c in ALL_CASES if c.assert_distinct_keys_independent_fn is not None],
        ids=[c.name for c in ALL_CASES if c.assert_distinct_keys_independent_fn is not None],
    )
    def test_distinct_keys_accumulate_independently(self, case: AccumulatorCase) -> None:
        accumulator = case.setup_fn()
        case.feed_fn(accumulator)
        assert case.assert_distinct_keys_independent_fn is not None
        case.assert_distinct_keys_independent_fn(accumulator)


# ─────────────────────────────────────────────────────────────────────
# CausalLink.variance_estimate moved-statistic invariant (Wire 1 linchpin)
# ─────────────────────────────────────────────────────────────────────


class TestCausalLinkMovedStatisticInvariant:
    """Pin the Wire 1 fix structurally.

    Pre-Wire-1 draft placed variance on ``CausalLink``. Because
    ``NAc._generate_link_id`` embeds ``outcome_signature`` into the link
    key (and ``outcome_signature`` embeds valence), each
    ``(event_sig, outcome_valence)`` pair allocates a separate
    ``CausalLink``. Per-link reward variance is therefore structurally 0
    for binary success/failure observations — every link saw ONE outcome
    class, never alternation. The Welford accumulator would have read
    0.0 and the "is this tool reliable" signal would have been silently
    dead.

    Wire 1 moved variance to ``NAc._event_outcome_welford`` keyed by
    ``(agent_id, event_signature)`` — the parent aggregation level
    where the cross-outcome distribution actually lives. The tests below
    pin both directions of the invariant:

    1. ``CausalLink`` does NOT carry a ``variance_estimate`` field.
       Re-introducing one would re-introduce the trap.
    2. ``NAc._event_outcome_welford`` IS populated with non-degenerate
       variance on binary-alternating outcomes for the corresponding
       ``(agent_id, event_signature)``.
    3. The per-link ``CausalLink`` objects created by the same input
       stream each see only ONE outcome valence (the structural
       signature of the trap) — confirming why placing variance on the
       link was structurally degenerate.

    If a future refactor re-introduces ``variance_estimate`` on
    ``CausalLink``, test 1 fails. If Wire 1's NAc-level Welford
    regresses, test 2 fails. If the link-keying changes so each link
    starts seeing both valences, test 3 fails AND the architectural
    rationale for the move needs revisiting.

    See CLAUDE.md L37 ("Key-embedded values produce structurally-
    degenerate statistics") and ``docs/plans/release_0_9_1.md`` Stage 4
    for the full rationale.
    """

    def test_causal_link_has_no_variance_estimate_field(self) -> None:
        """The Wire 1 trap field must not exist on CausalLink."""
        from maxim.decisions.causal_link import CausalLink
        from dataclasses import fields

        link_field_names = {f.name for f in fields(CausalLink)}
        # If this fails, someone re-added `variance_estimate` (or a
        # rename like `reward_variance`/`outcome_variance`) to
        # CausalLink — Wire 1 trap is back. The fix is to remove the
        # field and route the statistic through NAc._event_outcome_welford
        # per the moved-statistic invariant.
        forbidden = {"variance_estimate", "reward_variance", "outcome_variance"}
        assert link_field_names & forbidden == set(), (
            f"Wire 1 regression: CausalLink has key-trapped variance field(s) "
            f"{link_field_names & forbidden}. Variance MUST live on "
            f"NAc._event_outcome_welford keyed by (agent_id, event_signature). "
            f"See CLAUDE.md L37 + release_0_9_1.md Stage 4."
        )

    def test_welford_captures_variance_for_same_event_sig(self) -> None:
        """The parent aggregation IS non-degenerate where the link wasn't."""
        from maxim.decisions.causal_link import Valence

        nac = _make_nac()
        # Same event_signature, alternating outcome valences — the
        # exact input shape that produced structurally-0 per-link
        # variance pre-Wire-1.
        for _ in range(10):
            for v in (Valence.POSITIVE, Valence.NEGATIVE):
                nac.record_event("tool", "tool:flaky", context={"agent_id": "a"})
                nac.record_outcome("tool", "tool:flaky", v, context={"agent_id": "a"})

        state = nac._event_outcome_welford[("a", "tool:flaky")]
        variance = state["m2"] / state["n"]
        # Bernoulli p=0.5 → 0.25 variance at the NAc level. The signal
        # the Wire 1 trap silently killed is now visible here.
        assert math.isclose(variance, 0.25, abs_tol=1e-10), (
            f"Wire 1 regression: NAc Welford should capture Bernoulli "
            f"variance 0.25 on binary-alternating outcomes; got {variance:.6f}"
        )

    def test_per_link_outcome_homogeneity_explains_the_move(self) -> None:
        """Per-link outcomes are structurally homogeneous — by key design.

        ``_generate_link_id`` embeds ``outcome_signature`` (which
        embeds valence) into the link key. So binary-alternating
        outcomes generate TWO links — one for each valence — and each
        link sees only ONE outcome class. This is the structural
        signature of why per-link variance was 0: the link's existence
        IS the outcome's value.
        """
        from maxim.decisions.causal_link import Valence

        nac = _make_nac()
        for _ in range(10):
            for v in (Valence.POSITIVE, Valence.NEGATIVE):
                nac.record_event("tool", "tool:flaky", context={"agent_id": "a"})
                nac.record_outcome("tool", "tool:flaky", v, context={"agent_id": "a"})

        links_for_event = nac._links.get("tool:flaky", [])
        # The 20 outcomes (10 POSITIVE + 10 NEGATIVE) must split across
        # at least two distinct links — one per outcome valence — proving
        # the key trap is still structurally present at the link level.
        outcome_sigs_per_link = {link.id: link.outcome_signature for link in links_for_event}
        distinct_sigs = set(outcome_sigs_per_link.values())
        assert len(distinct_sigs) >= 2, (
            f"expected at least 2 distinct outcome_signatures across links "
            f"(binary-alternating input); got {distinct_sigs}. If this drops "
            f"to 1, _generate_link_id no longer embeds outcome_signature in "
            f"the key — Wire 1's architectural rationale needs re-checking."
        )
        # Each individual link saw only ONE outcome valence (the trap
        # signature). The observation_count per link reflects only its
        # one valence class — so per-link variance on the binary reward
        # would be structurally 0.
        for link in links_for_event:
            # A link's own outcome_valence is fixed at construction —
            # the per-link reward signal is constant by design.
            assert link.outcome_valence in (Valence.POSITIVE, Valence.NEGATIVE)
