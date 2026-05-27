"""Scripted-action trajectory invariants (Stage 2 of structural invariant tests).

Catches **sequential-drift** bugs: the interaction between operations
over time, where individual operations are already unit-tested but their
combination across N ticks isn't. Companion to
``test_statistic_shapes.py`` (Stage 1) and to the multi-agent marker
discipline (Stage 3).

**Design choice — scripted actions, not mocked LLM.** A mocked LLM
sounds simple but the agent loop has many call sites (action selection,
planning, narrator, AUT, deliberation, Acting Coach). A mock returning
garbage exercises error paths rather than happy paths. Instead, drive
the bio-pipeline with **pre-baked action sequences** that bypass LLM
calls entirely. Lower complexity, no mock-quality false-negative risk.
The scripted-action harness mirrors agent_loop.py section 8.5's
per-tick bio-system maintenance — see CLAUDE.md L159.

**Honest caveat:** scripted actions skip the real selection logic. This
is fine for mechanical-invariant tests but means trajectory tests do NOT
validate that bio-systems produce the right *actions* — that stays
Roy's job (per ``docs/plans/behavioral_graduation_candidates.md``).

Invariants asserted across at least three scripted scenarios (smoke,
drift, failure-recovery):

1. Memory tier progression is one-way (FORMING → SHORT_TERM → LONG_TERM)
   — enforced structurally by ``TierTransitionError`` in
   ``agents/bus.py`` (CLAUDE.md L115). Trajectory test confirms no
   illegal transition ever raises in production-shape code paths.
2. ``NAc._reward_bias ∈ [0, max_reward_bias]`` at every tick
   (CLAUDE.md L157).
3. **Decay non-negativity:** after ``decay_eligibility`` +
   ``decay_reward_biases``, no existing-trace value exceeds its
   pre-decay value (CLAUDE.md L159).
4. **No orphan eligibility traces:** every trace at tick N has either
   been touched in the last ``temporal_window_seconds`` OR has been
   pruned (CLAUDE.md L160).
5. **Promotion pressure monotonicity:** ``promotion_pressure`` only
   grows (on novel-context recall) or decays (with wall-clock time);
   no spontaneous jumps (CLAUDE.md L153).

Lives alongside ``test_sem_pain_cascade.py`` per the plan doc.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from maxim.agents.bus import MemoryTier, TierTransitionError, _assert_tier_transition
from maxim.decisions.causal_link import Valence
from maxim.decisions.nac import NAc, NACConfig


# ─────────────────────────────────────────────────────────────────────
# Scripted action types — minimal vocabulary for trajectory scripts
# ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RecordEvent:
    event_sig: str
    agent_id: str = "a"


@dataclass(frozen=True)
class RecordOutcome:
    event_sig: str
    valence: Valence
    agent_id: str = "a"


@dataclass(frozen=True)
class CreditNode:
    node_id: str
    reward: float
    agent_id: str = "a"


@dataclass(frozen=True)
class UpdateEligibility:
    node_id: str
    activation: float
    agent_id: str = "a"


@dataclass(frozen=True)
class DistributeReward:
    reward: float
    agent_id: str = "a"


@dataclass(frozen=True)
class NoOp:
    """Pure decay tick — no scripted operation, just per-tick maintenance."""


ScriptedAction = RecordEvent | RecordOutcome | CreditNode | UpdateEligibility | DistributeReward | NoOp


# ─────────────────────────────────────────────────────────────────────
# Per-tick snapshot of bio-state
# ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class BioSnapshot:
    """Frozen snapshot of NAc bio-state at one tick.

    Fields mirror the per-agent stash dicts at the moment the snapshot
    was taken. Pre-decay snapshots are taken AFTER the scripted action
    fires and BEFORE the per-tick decay phase runs; post-decay snapshots
    are taken AFTER the decay phase.
    """

    tick: int
    phase: str  # "pre_decay" or "post_decay"
    reward_bias: dict[tuple[str, str], float]
    eligibility: dict[tuple[str, str], float]
    cluster_reward_bias: dict[tuple[str, str, str], float]
    goal_reward_bias: dict[str, float]


@dataclass
class TrajectoryRecorder:
    """Collects ordered (pre_decay, post_decay) snapshot pairs per tick."""

    snapshots: list[BioSnapshot] = field(default_factory=list)

    def append(self, snap: BioSnapshot) -> None:
        self.snapshots.append(snap)

    def pre_decay_snapshots(self) -> list[BioSnapshot]:
        return [s for s in self.snapshots if s.phase == "pre_decay"]

    def post_decay_snapshots(self) -> list[BioSnapshot]:
        return [s for s in self.snapshots if s.phase == "post_decay"]


# ─────────────────────────────────────────────────────────────────────
# Harness — scripted action queue + per-tick decay (mirrors agent_loop 8.5)
# ─────────────────────────────────────────────────────────────────────


def _snapshot(nac: NAc, tick: int, phase: str) -> BioSnapshot:
    """Take an immutable snapshot of NAc bio-state."""
    return BioSnapshot(
        tick=tick,
        phase=phase,
        reward_bias=dict(nac._reward_bias),
        eligibility=dict(nac._eligibility),
        cluster_reward_bias=dict(nac._cluster_reward_bias),
        goal_reward_bias=dict(nac._goal_reward_bias),
    )


def _execute_action(nac: NAc, action: ScriptedAction) -> None:
    """Dispatch a scripted action to NAc — happy path only."""
    if isinstance(action, RecordEvent):
        nac.record_event("tool", action.event_sig, context={"agent_id": action.agent_id})
    elif isinstance(action, RecordOutcome):
        nac.record_outcome("tool", action.event_sig, action.valence, context={"agent_id": action.agent_id})
    elif isinstance(action, CreditNode):
        nac.credit_node(action.agent_id, action.node_id, action.reward)
    elif isinstance(action, UpdateEligibility):
        nac.update_eligibility(action.agent_id, action.node_id, action.activation)
    elif isinstance(action, DistributeReward):
        nac.distribute_reward(action.agent_id, action.reward)
    elif isinstance(action, NoOp):
        pass
    else:
        raise TypeError(f"unknown scripted action: {action!r}")


def _per_tick_decay(nac: NAc) -> None:
    """Mirror agent_loop.py section 8.5 per-tick decay phase."""
    nac.decay_eligibility()
    nac.decay_reward_biases()
    nac.decay_goal_reward_biases()
    nac.decay_cluster_reward_biases()


@pytest.fixture
def scripted_agent():
    """Build a minimal NAc-driven agent + scripted-action runner.

    Returns (run_scenario, nac, recorder). ``run_scenario(actions)``
    drives the given action sequence through the per-tick cycle,
    snapshotting before and after the decay phase for each tick.
    """
    nac = NAc(NACConfig(temporal_window_seconds=60, max_pending_events=2000))
    recorder = TrajectoryRecorder()

    def run_scenario(actions: list[ScriptedAction]) -> TrajectoryRecorder:
        for tick, action in enumerate(actions):
            _execute_action(nac, action)
            recorder.append(_snapshot(nac, tick, "pre_decay"))
            _per_tick_decay(nac)
            recorder.append(_snapshot(nac, tick, "post_decay"))
        return recorder

    return run_scenario, nac, recorder


# ─────────────────────────────────────────────────────────────────────
# Scenarios — smoke / drift / failure-recovery
# ─────────────────────────────────────────────────────────────────────


def _smoke_scenario() -> list[ScriptedAction]:
    """Short happy-path: one event → outcome → some decay ticks."""
    return [
        RecordEvent("tool:search"),
        UpdateEligibility("node_x", activation=0.8),
        RecordOutcome("tool:search", Valence.POSITIVE),
        DistributeReward(reward=1.0),
        NoOp(),
        NoOp(),
    ]


def _drift_scenario() -> list[ScriptedAction]:
    """50-tick mixed run — exercises decay across many ticks."""
    actions: list[ScriptedAction] = []
    # Seed some traces and credit-bearing nodes
    for i in range(5):
        actions.append(UpdateEligibility(f"node_{i}", activation=0.9))
        actions.append(CreditNode(f"node_{i}", reward=0.1 * (i + 1)))
    # Long decay phase
    actions.extend([NoOp()] * 40)
    return actions


def _failure_recovery_scenario() -> list[ScriptedAction]:
    """Alternating positive/negative — exercises bidirectional updates."""
    actions: list[ScriptedAction] = []
    for i in range(10):
        v = Valence.POSITIVE if i % 2 == 0 else Valence.NEGATIVE
        actions.append(RecordEvent("tool:flaky"))
        actions.append(UpdateEligibility(f"node_alt_{i % 3}", activation=0.7))
        actions.append(RecordOutcome("tool:flaky", v))
        actions.append(DistributeReward(reward=1.0 if v is Valence.POSITIVE else -1.0))
    return actions


ALL_SCENARIOS: list[tuple[str, callable]] = [
    ("smoke", _smoke_scenario),
    ("drift", _drift_scenario),
    ("failure_recovery", _failure_recovery_scenario),
]


# ─────────────────────────────────────────────────────────────────────
# Invariant 1 — Memory tier progression is one-way
# ─────────────────────────────────────────────────────────────────────


class TestTierProgressionInvariant:
    """``MemoryTier`` lifecycle is strictly one-way.

    The trajectory test confirms no production code path silently
    bypasses the ``TierTransitionError`` guard. Direct test of the
    enforcement function: every legal transition succeeds; every
    illegal one raises. (Scripted-action trajectories don't drive
    memory tier transitions directly — that's covered structurally by
    the bus-level enforcement these tests pin.)
    """

    def test_legal_progressions_do_not_raise(self) -> None:
        _assert_tier_transition(MemoryTier.FORMING, MemoryTier.FORMING)  # no-op
        _assert_tier_transition(MemoryTier.FORMING, MemoryTier.SHORT_TERM)
        _assert_tier_transition(MemoryTier.SHORT_TERM, MemoryTier.SHORT_TERM)
        _assert_tier_transition(MemoryTier.SHORT_TERM, MemoryTier.LONG_TERM)
        _assert_tier_transition(MemoryTier.LONG_TERM, MemoryTier.LONG_TERM)

    @pytest.mark.parametrize(
        "old,new",
        [
            (MemoryTier.FORMING, MemoryTier.LONG_TERM),  # skip SHORT_TERM
            (MemoryTier.SHORT_TERM, MemoryTier.FORMING),  # reverse
            (MemoryTier.LONG_TERM, MemoryTier.SHORT_TERM),  # reverse
            (MemoryTier.LONG_TERM, MemoryTier.FORMING),  # reverse skip
        ],
    )
    def test_illegal_transitions_raise(self, old: MemoryTier, new: MemoryTier) -> None:
        with pytest.raises(TierTransitionError):
            _assert_tier_transition(old, new)


# ─────────────────────────────────────────────────────────────────────
# Invariant 2 — NAc._reward_bias bounds across every tick
# ─────────────────────────────────────────────────────────────────────


class TestRewardBiasBoundsInvariant:
    @pytest.mark.parametrize("name,scenario_fn", ALL_SCENARIOS, ids=[n for n, _ in ALL_SCENARIOS])
    def test_reward_bias_stays_in_bounds_at_every_tick(self, scripted_agent, name: str, scenario_fn) -> None:
        run_scenario, nac, recorder = scripted_agent
        run_scenario(scenario_fn())
        cap = nac.config.max_reward_bias
        for snap in recorder.snapshots:
            for key, bias in snap.reward_bias.items():
                assert 0.0 <= bias <= cap, (
                    f"[{name}] tick {snap.tick} {snap.phase}: reward_bias for {key} = {bias} out of [0, {cap}]"
                )


# ─────────────────────────────────────────────────────────────────────
# Invariant 3 — Decay non-negativity (post-decay <= pre-decay for existing)
# ─────────────────────────────────────────────────────────────────────


class TestDecayNonNegativityInvariant:
    @pytest.mark.parametrize("name,scenario_fn", ALL_SCENARIOS, ids=[n for n, _ in ALL_SCENARIOS])
    def test_eligibility_post_decay_does_not_exceed_pre_decay(self, scripted_agent, name: str, scenario_fn) -> None:
        """For each tick, post-decay eligibility values must be ≤ pre-decay."""
        run_scenario, _nac, recorder = scripted_agent
        run_scenario(scenario_fn())
        snaps = recorder.snapshots
        # snapshots are interleaved [pre0, post0, pre1, post1, ...]
        for i in range(0, len(snaps), 2):
            pre = snaps[i]
            post = snaps[i + 1]
            for key, post_val in post.eligibility.items():
                pre_val = pre.eligibility.get(key, 0.0)
                # Existing trace cannot grow during the decay phase.
                # (A new trace can ONLY appear if the scripted action at
                # this tick added it BEFORE the pre-decay snapshot, so
                # `pre_val` already reflects it.)
                assert post_val <= pre_val + 1e-9, (
                    f"[{name}] tick {pre.tick}: eligibility[{key}] grew during decay phase ({pre_val} → {post_val})"
                )

    @pytest.mark.parametrize("name,scenario_fn", ALL_SCENARIOS, ids=[n for n, _ in ALL_SCENARIOS])
    def test_reward_bias_post_decay_does_not_exceed_pre_decay(self, scripted_agent, name: str, scenario_fn) -> None:
        """For each tick, post-decay reward_bias values must be ≤ pre-decay."""
        run_scenario, _nac, recorder = scripted_agent
        run_scenario(scenario_fn())
        snaps = recorder.snapshots
        for i in range(0, len(snaps), 2):
            pre = snaps[i]
            post = snaps[i + 1]
            for key, post_val in post.reward_bias.items():
                pre_val = pre.reward_bias.get(key, 0.0)
                assert post_val <= pre_val + 1e-9, (
                    f"[{name}] tick {pre.tick}: reward_bias[{key}] grew during decay phase ({pre_val} → {post_val})"
                )


# ─────────────────────────────────────────────────────────────────────
# Invariant 4 — No orphan eligibility traces (everything decays or stays touched)
# ─────────────────────────────────────────────────────────────────────


class TestNoOrphanEligibilityTracesInvariant:
    """Eligibility traces must decay below threshold AND get pruned, given
    enough idle ticks."""

    def test_orphan_trace_eventually_pruned(self, scripted_agent) -> None:
        """A trace that never gets re-touched must disappear after enough decay ticks."""
        run_scenario, nac, recorder = scripted_agent
        # Seed one trace, then run pure-decay ticks.
        # decay_eligibility factor=0.9; threshold for pruning = 0.01.
        # 0.8 * 0.9^N < 0.01 → N > log(0.0125)/log(0.9) ≈ 41.6 → 42 ticks.
        actions: list[ScriptedAction] = [UpdateEligibility("orphan_node", activation=0.8)]
        actions.extend([NoOp()] * 50)
        run_scenario(actions)
        # After all decay ticks, the orphan trace must be pruned.
        final_snap = recorder.snapshots[-1]
        assert ("a", "orphan_node") not in final_snap.eligibility, (
            f"orphan eligibility trace survived 50 idle decay ticks; final eligibility map = {final_snap.eligibility}"
        )


# ─────────────────────────────────────────────────────────────────────
# Invariant 5 — Promotion pressure monotonicity
# ─────────────────────────────────────────────────────────────────────


class TestPromotionPressureMonotonicityInvariant:
    """``promotion_pressure`` either grows (on novel-context recall) or
    decays smoothly (with wall-clock time); never spontaneous jumps.

    Driven directly on the consolidation surface — scripted-action
    harness focuses on NAc, but the pressure invariant is checked here
    so the trajectory file covers all 5 plan invariants.
    """

    def test_pressure_only_grows_on_novel_context(self) -> None:
        from maxim.memory.hippocampus import Hippocampus, HippocampusConfig
        from maxim.memory.types import EpisodicMemory, Perception

        hippocampus = Hippocampus(config=HippocampusConfig())
        mem = EpisodicMemory(
            id="mem_x",
            timestamp=0.0,
            perception=Perception(cli_input="alpha", salience=0.5),
        )

        # Five novel-context recalls → pressure strictly grows each time
        # (until either the threshold promotes or all five complete).
        pressures: list[float] = [mem.promotion_pressure]
        for i in range(5):
            hippocampus._score_and_maybe_promote_batch([mem], query=f"q_{i}")
            pressures.append(mem.promotion_pressure)

        # Either: monotone non-decreasing across the run (no promotion),
        # OR: reached threshold and reset to 0.0 (promotion fired).
        promoted = mem.long_term
        if promoted:
            # Last value reset to 0; assert at least one earlier value
            # exceeded the threshold trigger.
            assert any(p > 0 for p in pressures[:-1])
            assert pressures[-1] == 0.0
        else:
            # Monotone non-decreasing — no spontaneous jumps down.
            for prev, curr in zip(pressures, pressures[1:]):
                assert curr >= prev, (
                    f"promotion_pressure decreased without promotion: {prev} → {curr} (sequence: {pressures})"
                )

    def test_pressure_does_not_grow_on_identical_context(self) -> None:
        from maxim.memory.hippocampus import Hippocampus, HippocampusConfig
        from maxim.memory.types import EpisodicMemory, Perception

        hippocampus = Hippocampus(config=HippocampusConfig())
        mem = EpisodicMemory(
            id="mem_y",
            timestamp=0.0,
            perception=Perception(cli_input="alpha", salience=0.5),
        )

        # First recall bumps pressure once; subsequent identical-context
        # recalls hit the access_contexts gate and do NOT grow pressure.
        hippocampus._score_and_maybe_promote_batch([mem], query="same_query")
        first_pressure = mem.promotion_pressure
        for _ in range(5):
            hippocampus._score_and_maybe_promote_batch([mem], query="same_query")
        # Pressure must not have grown beyond the first bump.
        assert mem.promotion_pressure <= first_pressure + 1e-9, (
            f"identical-context recall grew pressure from {first_pressure} to {mem.promotion_pressure}"
        )


# ─────────────────────────────────────────────────────────────────────
# Cross-cutting — trajectory smoke (the harness itself produces records)
# ─────────────────────────────────────────────────────────────────────


class TestHarnessSmoke:
    """The harness machinery itself does what it claims."""

    def test_scenario_produces_paired_snapshots_per_tick(self, scripted_agent) -> None:
        run_scenario, _nac, recorder = scripted_agent
        actions = [RecordEvent("tool:t"), NoOp(), NoOp()]
        run_scenario(actions)
        # Two snapshots per tick (pre_decay + post_decay)
        assert len(recorder.snapshots) == 2 * len(actions)
        for i, action in enumerate(actions):
            assert recorder.snapshots[2 * i].phase == "pre_decay"
            assert recorder.snapshots[2 * i].tick == i
            assert recorder.snapshots[2 * i + 1].phase == "post_decay"
            assert recorder.snapshots[2 * i + 1].tick == i

    def test_all_scenarios_run_without_error(self, scripted_agent) -> None:
        run_scenario, _nac, recorder = scripted_agent
        # Smoke-run just the smallest scenario via the fixture; the
        # parametrized invariant tests above cover all three across all
        # invariants.
        run_scenario(_smoke_scenario())
        assert len(recorder.snapshots) > 0
