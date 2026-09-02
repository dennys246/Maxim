"""D44 — a merge must change BEHAVIOUR, between genuinely independent agents.

This is 1.1.3's **ship gate**, and it is written to fail. Per
`docs/plans/d43_merge_correctness.md` §6: *a D44 test that is green before D43
is fixed is by definition testing the wrong thing.* Re-keying does not exist on
`main` — `nac_merge` folds `cluster_reward_bias` by exact key match, and
`ec_merge` computes its right→left node alignment and **discards the id map** —
so the behavioural arms below are `xfail(strict=True)` until D43 lands. The
strictness matters: when the fix works, an XPASS turns the suite red and forces
someone to remove the marker deliberately.

**What "independent" means here, mechanically** (all three D43 barriers):

1. **Distinct `agent_id`.** `NAc._cluster_reward_bias` is keyed
   `(agent_id, cluster_id, tool_signature)` and `get_agent_tool_biases` filters
   on it.
2. **A separate `EntorhinalCortex` + `SensorEncoder` object per agent.** This is
   the load-bearing construction step: `pattern_complete_or_separate` allocates
   `str(uuid4())` on separation, so identical sensor input yields *disjoint*
   cluster ids. Sharing the encoder — or building one and copying it — is what
   makes the existing federation probes unable to observe D43 at all
   (`5_operant_creche_federation.py` shares one encoder by construction and says
   so in its own docstring).
3. Optionally distinct bodies, which gate 7 now refuses rather than silently
   zeroes. Not exercised here — the tool-signature barrier does **not** fire for
   two agents on one body, which is exactly this configuration.

**Why the existing tests do not cover this.** No test in the repo calls
`recommend_action` after a merge. `test_hivemind_bundle.py::test_end_to_end_hivemind_round_trip`
runs the full pipeline but asserts dict equality on hand-set, already-matching
keys; every value-level merge test uses keys that match by construction. They
correctly pin `_merge_mean_clamped`'s arithmetic and say nothing about behaviour.
"""

from __future__ import annotations

import pytest

from maxim.decisions.nac import NAc, NACConfig
from maxim.hivemind.merge import nac_merge
from maxim.similarity.ec import ECConfig, EntorhinalCortex
from maxim.similarity.encoder import SensorEncoder

# The contingency A learns and B has never seen: in the "left" sensory state the
# correct action is `turn_left`; in the "right" state, `turn_right`.
TOOLS = ("turn_left", "turn_right")
STATES = {"left": {"azimuth": 0.10}, "right": {"azimuth": 0.90}}
CORRECT = {"left": "tool:turn_left", "right": "tool:turn_right"}
RANGES = {"azimuth": (0.0, 1.0)}


class Agent:
    """An agent with its OWN encoder, EC and NAc — no shared objects."""

    def __init__(self, agent_id: str) -> None:
        self.agent_id = agent_id
        self.nac = NAc(NACConfig())
        self.ec = EntorhinalCortex(ECConfig(frozen_centroid_modalities=frozenset({"interoception"})))
        self.encoder = SensorEncoder(ec=self.ec, atl=None, nac=self.nac)

    def cluster_for(self, state_name: str) -> str:
        return str(
            self.encoder.encode_sensors(
                agent_id=self.agent_id,
                sensors=STATES[state_name],
                modality="interoception",
                ranges=RANGES,
            )
        )

    def teach(self, reps: int = 12) -> None:
        """Credit the correct tool in each state — the contingency."""
        for _ in range(reps):
            for name in STATES:
                cid = self.cluster_for(name)
                for tool in TOOLS:
                    sig = f"tool:{tool}"
                    self.nac.update_cluster_reward(
                        agent_id=self.agent_id,
                        cluster_id=cid,
                        tool_signature=sig,
                        reward=1.0 if sig == CORRECT[name] else -1.0,
                        source="operant",
                    )

    def score(self) -> float:
        """Fraction of states where recommend_action picks the correct tool."""
        hits = 0
        for name in STATES:
            rec = self.nac.recommend_action(
                agent_id=self.agent_id,
                available_tools=list(TOOLS),
                current_cluster_id=self.cluster_for(name),
                min_confidence=0.0,
            )
            if rec and f"tool:{rec['tool_name']}" == CORRECT[name]:
                hits += 1
        return hits / len(STATES)


def _merged_into(receiver: Agent, donor: Agent, merge_fn=nac_merge) -> None:
    receiver.nac.load_state(
        merge_fn(receiver.nac.dump(), donor.nac.dump(), left_source="receiver", right_source="donor")
    )


def test_independence_is_real_not_assumed() -> None:
    """The construction check. If this fails, every arm below is vacuous.

    Two agents shown the SAME sensory state must land on DIFFERENT cluster ids —
    that is what `uuid4()` on separation guarantees and what a shared encoder
    would destroy.
    """
    a, b = Agent("A"), Agent("B")
    assert a.cluster_for("left") != b.cluster_for("left")
    assert a.cluster_for("right") != b.cluster_for("right")
    assert a.agent_id != b.agent_id


def test_the_teacher_actually_learned() -> None:
    """Guards against a null that is really a broken fixture."""
    a = Agent("A")
    assert a.score() == 0.0 or a.score() < 1.0
    a.teach()
    assert a.score() == 1.0, "the donor never learned the contingency — fixture broken, not a merge finding"


@pytest.mark.xfail(strict=True, reason="D43: re-keying does not exist; ec_merge discards the id map")
def test_merge_transfers_the_want_behaviourally() -> None:
    """THE SHIP GATE. B never saw the contingency; after the merge it acts on it."""
    a, b = Agent("A"), Agent("B")
    a.teach()

    before = b.score()
    _merged_into(b, a)
    after = b.score()

    assert before <= 0.5, f"B already knew the contingency before the merge ({before})"
    assert after - before >= 0.20, (
        f"merge produced no behavioural delta: {before} -> {after}. "
        "nac_merge folds cluster_reward_bias on exact key match and ec_merge "
        "discards its id map, so A's biases land under cluster ids that are not "
        "nodes in B's EC — structurally unreachable at readout (D43)."
    )


@pytest.mark.xfail(strict=True, reason="D43: same as above")
def test_negative_control_a_bundle_that_learned_nothing_changes_nothing() -> None:
    """Separates 'a bundle arrived' from 'a want arrived'.

    Marked xfail with the gate: it is meaningless to assert this arm passes
    while the positive arm cannot. Both flip together when D43 lands.
    """
    a = Agent("A")
    a.teach()
    naive = Agent("N")  # never taught

    b_taught, b_naive = Agent("B"), Agent("B2")
    _merged_into(b_taught, a)
    _merged_into(b_naive, naive)

    assert b_taught.score() - b_naive.score() >= 0.20


def test_dangling_half_reproduces_the_silent_zero() -> None:
    """Merging NAc state without aligning EC must contribute NOTHING.

    This arm passes TODAY and must keep passing after D43 — it pins the failure
    mode shut. `nac_merge` alone is exactly what `hivemind/cli.py`'s `merge-nac`
    verb does, so this is the shipped default path, not a hypothetical.
    """
    a, b = Agent("A"), Agent("B")
    a.teach()

    before = b.score()
    _merged_into(b, a)  # NAc only — no ec_merge, no re-keying
    after = b.score()

    assert after == before, (
        f"a NAc-only merge changed behaviour ({before} -> {after}) — if D43 is fixed, "
        "this arm must be re-pointed at the aligned path rather than deleted"
    )


def test_the_merge_reports_success_while_contributing_nothing() -> None:
    """The property that makes D43 dangerous, pinned directly.

    The receiver's bias dict GROWS — `len()` is `|left ∪ right|`, maximal exactly
    when nothing aligns — while behaviour is unchanged. The success indicator is
    inversely correlated with success.
    """
    a, b = Agent("A"), Agent("B")
    a.teach()

    keys_before = len(b.nac.dump().get("cluster_reward_bias", {}))
    score_before = b.score()
    _merged_into(b, a)
    keys_after = len(b.nac.dump().get("cluster_reward_bias", {}))

    assert keys_after > keys_before, "expected the union to grow"
    assert b.score() == score_before, "expected zero behavioural change"


def test_anti_vacuity_a_noop_merge_must_not_pass_the_gate() -> None:
    """The guard on the guard.

    If the ship gate can be satisfied by a merge that folds nothing, it is not a
    gate. This arm asserts the delta collapses when `nac_merge` is replaced by
    `return left` — the check Exp 45's merge arm lacks, which is why that arm
    passes against a no-op today (D62).
    """
    a, b = Agent("A"), Agent("B")
    a.teach()

    before = b.score()
    _merged_into(b, a, merge_fn=lambda left, right, **kw: left)
    assert b.score() == before, "a no-op merge changed behaviour — the fixture is not measuring the merge"


@pytest.mark.xfail(strict=True, reason="D43: ec_merge discards the id map, so no key is re-keyed")
def test_every_surviving_bias_key_names_a_reachable_cluster() -> None:
    """The brief's §4 guard, as a test — the cheapest possible D43 detector.

    > Every `cluster_reward_bias` key surviving a merge must name a cluster id
    > present in the merged EC.

    Measured today: **0 of 4** merged keys name a cluster the receiver can
    reach. That is D43's mechanism stated as an invariant rather than as a
    behavioural consequence, and it is mechanical enough that it would have
    caught the defect the day `ec_merge` shipped.

    It is deliberately separate from the behavioural gate above: this one says
    *why* the delta is zero, so a future failure is diagnosable without
    re-deriving the cause.
    """
    a, b = Agent("A"), Agent("B")
    a.teach()
    _merged_into(b, a)

    reachable = {b.cluster_for(name) for name in STATES}
    merged_keys = list(b.nac.dump().get("cluster_reward_bias", {}))
    assert merged_keys, "nothing merged at all — wrong failure"

    unreachable = [k for k in merged_keys if k.split("\x1f")[1] not in reachable]
    assert not unreachable, (
        f"{len(unreachable)} of {len(merged_keys)} merged bias keys name cluster ids that are "
        "not nodes the receiver can reach — the foreign want is structurally unreadable (D43)"
    )
