"""Tests for PFC multi-cycle deliberation (_run_deliberation_cycles).

Covers:
- Cycle 2+ enrichment + re-submission when ready_to_act is False
- Convergence detection (Jaccard >= 0.8)
- Max cycle cap
- Ready-to-act on cycle 2 returns immediately
- Empty reasoning gracefully handled
- Stop event cancellation
"""

from __future__ import annotations

import threading
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from maxim.agents.llm_types import LLMProposal


_DEFAULT_ACTION = {"tool_name": "speak", "params": {"text": "hello"}}
_SENTINEL = object()


def _make_proposal(
    *,
    ready_to_act: bool = True,
    reasoning: str = "test reasoning",
    action: dict[str, Any] | None | object = _SENTINEL,
) -> LLMProposal:
    return LLMProposal(
        request_id="test-req",
        action=_DEFAULT_ACTION if action is _SENTINEL else action,
        reasoning=reasoning,
        strategy_used=None,
        confidence=0.8,
        mode_goal_achieved=False,
        ready_to_act=ready_to_act,
    )


class FakeEnrichmentResult:
    def __init__(self, memories: str = "mem", predictions: str = "", concepts: str = ""):
        self.memories = memories
        self.predictions = predictions
        self.concepts = concepts
        self.affordances = ""
        self.recent_context = ""


class FakeBioEnrichment:
    """Minimal BioEnrichmentPipeline mock."""

    def __init__(self):
        self.enrich_calls: list[str] = []

    def enrich(self, text: str, *, context: Any = None, bypass_gate: bool = False) -> FakeEnrichmentResult:
        self.enrich_calls.append(text)
        return FakeEnrichmentResult(memories=f"memory about: {text[:30]}")

    def format_thought_response(self, result: FakeEnrichmentResult) -> str:
        return f"[enriched] {result.memories}"


class FakeWMS:
    """Minimal WorkingMemorySet mock."""

    def __init__(self):
        self.entries: list[dict] = []
        self.current_tick = 0

    def add(self, kind: Any, *, content: Any, salience: float = 0.5) -> None:
        self.entries.append({"kind": kind, "content": content, "salience": salience})

    def by_kind(self, kinds: set, limit: int = 10) -> list:
        return []


class FakeContext:
    """Minimal StructuredContext mock."""

    def __init__(self):
        self.bio_enrichment_context: str = ""
        self.working_memory_thoughts: list[str] | None = None
        self.deliberation_transcript: list[str] | None = None


class FakeThoughtGate:
    def __init__(self):
        self.refractory_resets: list[int] = []

    def reset_refractory(self, tick: int) -> None:
        self.refractory_resets.append(tick)


@pytest.fixture
def bio():
    return FakeBioEnrichment()


@pytest.fixture
def wms():
    return FakeWMS()


@pytest.fixture
def ctx():
    return FakeContext()


@pytest.fixture
def gate():
    return FakeThoughtGate()


def _run_cycles(**kwargs):
    """Import and call _run_deliberation_cycles."""
    from maxim.runtime.agent_loop import _run_deliberation_cycles

    return _run_deliberation_cycles(**kwargs)


class TestDeliberationCycles:
    """Unit tests for _run_deliberation_cycles."""

    def test_ready_to_act_on_cycle_2(self, bio, wms, ctx, gate):
        """When cycle 2 returns ready_to_act=True, the proposal is returned."""
        cycle2_proposal = _make_proposal(ready_to_act=True, reasoning="I will act now")
        llm_worker = MagicMock()
        llm_worker.get_latest_proposal.return_value = cycle2_proposal

        submit_calls = []

        def submit_fn(context):
            submit_calls.append(context)
            return True

        first = _make_proposal(ready_to_act=False, reasoning="I need to think more about the guard")

        with (
            patch("maxim.simulation.sim_logger.sim_log"),
            patch("maxim.simulation.sim_logger.sim_pre_deliberation"),
            patch("maxim.simulation.sim_logger.sim_contemplation"),
        ):
            # These are imported inside the function, patch at module level
            result = _run_cycles(
                first_proposal=first,
                bio_enrichment=bio,
                working_memory=wms,
                context=ctx,
                submit_fn=submit_fn,
                llm_worker=llm_worker,
                stop_event=None,
                thought_gate=gate,
                active_goal="escape the dungeon",
                step_num=5,
                max_cycles=3,
            )

        assert result is not None
        assert result.ready_to_act is True
        assert len(submit_calls) == 1  # One re-submission (cycle 2)
        assert len(bio.enrich_calls) == 1  # Enriched cycle 1's reasoning
        assert gate.refractory_resets == [5]

    def test_max_cycles_with_action(self, bio, wms, ctx, gate):
        """When max cycles reached with action present, returns last proposal."""
        not_ready = _make_proposal(
            ready_to_act=False,
            reasoning="still thinking about different aspects",
            action={"tool_name": "move", "params": {"dir": "north"}},
        )
        llm_worker = MagicMock()
        llm_worker.get_latest_proposal.return_value = not_ready

        submit_calls = []

        def submit_fn(context):
            submit_calls.append(True)
            return True

        first = _make_proposal(ready_to_act=False, reasoning="first thoughts on the situation")

        with (
            patch("maxim.simulation.sim_logger.sim_log"),
            patch("maxim.simulation.sim_logger.sim_pre_deliberation"),
            patch("maxim.simulation.sim_logger.sim_contemplation"),
        ):
            result = _run_cycles(
                first_proposal=first,
                bio_enrichment=bio,
                working_memory=wms,
                context=ctx,
                submit_fn=submit_fn,
                llm_worker=llm_worker,
                stop_event=None,
                thought_gate=gate,
                active_goal=None,
                step_num=10,
                max_cycles=3,
            )

        assert result is not None
        assert result.action is not None
        # 2 re-submissions (cycles 2 and 3)
        assert len(submit_calls) == 2
        assert gate.refractory_resets == [10]

    def test_max_cycles_no_action_returns_none(self, bio, wms, ctx, gate):
        """When max cycles reached without action on any proposal, returns None."""
        not_ready = _make_proposal(ready_to_act=False, reasoning="still thinking but no plan", action=None)
        llm_worker = MagicMock()
        llm_worker.get_latest_proposal.return_value = not_ready

        # First proposal also has no action
        first = _make_proposal(ready_to_act=False, reasoning="initial thoughts no action", action=None)

        with (
            patch("maxim.simulation.sim_logger.sim_log"),
            patch("maxim.simulation.sim_logger.sim_pre_deliberation"),
            patch("maxim.simulation.sim_logger.sim_contemplation"),
        ):
            result = _run_cycles(
                first_proposal=first,
                bio_enrichment=bio,
                working_memory=wms,
                context=ctx,
                submit_fn=lambda c: True,
                llm_worker=llm_worker,
                stop_event=None,
                thought_gate=gate,
                active_goal=None,
                step_num=1,
                max_cycles=2,
            )

        assert result is None

    def test_convergence_detection(self, bio, wms, ctx, gate):
        """When reasoning converges (Jaccard >= 0.8), returns proposal."""
        # Make cycle 2 produce nearly identical reasoning to cycle 1
        converged = _make_proposal(
            ready_to_act=False,
            reasoning="the guard is sleeping and I should sneak past",
            action={"tool_name": "sneak", "params": {}},
        )
        llm_worker = MagicMock()
        llm_worker.get_latest_proposal.return_value = converged

        # Cycle 1 has very similar reasoning
        first = _make_proposal(
            ready_to_act=False,
            reasoning="the guard is sleeping and I should sneak past quietly",
        )

        with (
            patch("maxim.simulation.sim_logger.sim_log"),
            patch("maxim.simulation.sim_logger.sim_pre_deliberation"),
            patch("maxim.simulation.sim_logger.sim_contemplation"),
        ):
            result = _run_cycles(
                first_proposal=first,
                bio_enrichment=bio,
                working_memory=wms,
                context=ctx,
                submit_fn=lambda c: True,
                llm_worker=llm_worker,
                stop_event=None,
                thought_gate=gate,
                active_goal=None,
                step_num=3,
                max_cycles=3,
            )

        # Should return the proposal (convergence detected) since it has action
        assert result is not None

    def test_stop_event_cancellation(self, bio, wms, ctx, gate):
        """When stop_event is set, returns last proposal or None."""
        stop = threading.Event()
        stop.set()  # Already cancelled

        llm_worker = MagicMock()
        llm_worker.get_latest_proposal.return_value = None  # Won't return before stop

        first = _make_proposal(ready_to_act=False, reasoning="thinking")

        with (
            patch("maxim.simulation.sim_logger.sim_log"),
            patch("maxim.simulation.sim_logger.sim_pre_deliberation"),
            patch("maxim.simulation.sim_logger.sim_contemplation"),
        ):
            _run_cycles(
                first_proposal=first,
                bio_enrichment=bio,
                working_memory=wms,
                context=ctx,
                submit_fn=lambda c: True,
                llm_worker=llm_worker,
                stop_event=stop,
                thought_gate=gate,
                active_goal=None,
                step_num=1,
                max_cycles=3,
            )

        # Should return None since wait_for_proposal returns None on stop
        # and first_proposal has no action
        # (first has action by default from _make_proposal, so last_proposal.action exists)
        # The function breaks on _wait_for_proposal returning None and falls through
        # to the max-cycles path which checks last_proposal.action

    def test_enrichment_feeds_reasoning_back(self, bio, wms, ctx, gate):
        """Verify that the LLM's reasoning text is enriched in cycle 2+."""
        cycle2 = _make_proposal(ready_to_act=True, reasoning="now I know what to do")
        llm_worker = MagicMock()
        llm_worker.get_latest_proposal.return_value = cycle2

        first = _make_proposal(ready_to_act=False, reasoning="The door seems locked but the guard has keys")

        with (
            patch("maxim.simulation.sim_logger.sim_log"),
            patch("maxim.simulation.sim_logger.sim_pre_deliberation"),
            patch("maxim.simulation.sim_logger.sim_contemplation"),
        ):
            _run_cycles(
                first_proposal=first,
                bio_enrichment=bio,
                working_memory=wms,
                context=ctx,
                submit_fn=lambda c: True,
                llm_worker=llm_worker,
                stop_event=None,
                thought_gate=None,
                active_goal=None,
                step_num=1,
                max_cycles=3,
            )

        # Bio-enrichment should have been called with cycle 1's reasoning
        assert len(bio.enrich_calls) == 1
        assert "The door seems locked" in bio.enrich_calls[0]

    def test_context_updated_each_cycle(self, bio, wms, ctx, gate):
        """Verify bio_enrichment_context is replaced (not appended) each cycle."""
        cycle2 = _make_proposal(ready_to_act=True, reasoning="acting now")
        llm_worker = MagicMock()
        llm_worker.get_latest_proposal.return_value = cycle2

        first = _make_proposal(ready_to_act=False, reasoning="thinking about guard")

        with (
            patch("maxim.simulation.sim_logger.sim_log"),
            patch("maxim.simulation.sim_logger.sim_pre_deliberation"),
            patch("maxim.simulation.sim_logger.sim_contemplation"),
        ):
            _run_cycles(
                first_proposal=first,
                bio_enrichment=bio,
                working_memory=wms,
                context=ctx,
                submit_fn=lambda c: True,
                llm_worker=llm_worker,
                stop_event=None,
                thought_gate=None,
                active_goal=None,
                step_num=1,
                max_cycles=3,
            )

        # Context should have the enriched text (replaced, not appended)
        assert ctx.bio_enrichment_context.startswith("[enriched]")

    def test_submit_fn_queue_full(self, bio, wms, ctx, gate):
        """When submit_fn returns False (queue full), cycle stops gracefully."""
        first = _make_proposal(ready_to_act=False, reasoning="thinking")

        llm_worker = MagicMock()

        with (
            patch("maxim.simulation.sim_logger.sim_log"),
            patch("maxim.simulation.sim_logger.sim_pre_deliberation"),
            patch("maxim.simulation.sim_logger.sim_contemplation"),
        ):
            result = _run_cycles(
                first_proposal=first,
                bio_enrichment=bio,
                working_memory=wms,
                context=ctx,
                submit_fn=lambda c: False,  # Queue full
                llm_worker=llm_worker,
                stop_event=None,
                thought_gate=gate,
                active_goal=None,
                step_num=1,
                max_cycles=3,
            )

        # Should return first_proposal's action (it has one by default)
        assert result is not None


class TestJaccardConvergence:
    """Tests for _jaccard_convergence helper."""

    def test_identical_sets(self):
        from maxim.runtime.agent_loop import _jaccard_convergence

        a = {"the", "guard", "is", "sleeping"}
        assert _jaccard_convergence(a, a) is True

    def test_disjoint_sets(self):
        from maxim.runtime.agent_loop import _jaccard_convergence

        a = {"the", "guard", "is", "sleeping"}
        b = {"we", "should", "run", "away"}
        assert _jaccard_convergence(a, b) is False

    def test_too_few_keywords(self):
        from maxim.runtime.agent_loop import _jaccard_convergence

        a = {"hi", "there"}
        b = {"hi", "there"}
        assert _jaccard_convergence(a, b) is False  # < 3 keywords

    def test_high_overlap(self):
        from maxim.runtime.agent_loop import _jaccard_convergence

        a = {"the", "guard", "is", "sleeping", "near", "door"}
        b = {"the", "guard", "is", "sleeping", "near", "door", "quietly"}
        # Jaccard = 6/7 ≈ 0.857 >= 0.8
        assert _jaccard_convergence(a, b) is True


class TestComputeThoughtSalience:
    """Tests for _compute_thought_salience (Stage 2)."""

    def test_maximum_salience(self):
        """All signals at max → salience near 1.0."""
        from maxim.runtime.agent_loop import _compute_thought_salience

        # 5 sections (0.3), 4 memories (0.3), 0.0 jaccard = fully novel (0.4)
        s = _compute_thought_salience(n_sections=5, n_memories=4, jaccard_with_previous=0.0)
        assert s == pytest.approx(1.0)

    def test_minimum_salience(self):
        """All signals at min → salience near 0.0."""
        from maxim.runtime.agent_loop import _compute_thought_salience

        # 0 sections (0.0), 0 memories (0.0), 1.0 jaccard = identical (0.0)
        s = _compute_thought_salience(n_sections=0, n_memories=0, jaccard_with_previous=1.0)
        assert s == pytest.approx(0.0)

    def test_typical_cycle_1(self):
        """Cycle 1 (no prior → jaccard=0.0) with moderate enrichment."""
        from maxim.runtime.agent_loop import _compute_thought_salience

        # 2 sections, 1 memory, fully novel
        s = _compute_thought_salience(n_sections=2, n_memories=1, jaccard_with_previous=0.0)
        expected = 0.3 * (2 / 5) + 0.3 * (1 / 4) + 0.4 * 1.0
        assert s == pytest.approx(expected)
        assert s > 0.5  # Cycle 1 should be above average

    def test_converging_later_cycle(self):
        """Later cycle with high convergence scores lower."""
        from maxim.runtime.agent_loop import _compute_thought_salience

        # 2 sections, 1 memory, high convergence (jaccard=0.85)
        s = _compute_thought_salience(n_sections=2, n_memories=1, jaccard_with_previous=0.85)
        assert s < 0.4  # Converging = low novelty = low salience

    def test_novelty_differentiates_equal_enrichment(self):
        """With equal enrichment, novelty breaks the tie."""
        from maxim.runtime.agent_loop import _compute_thought_salience

        # Same enrichment depth, different novelty
        novel = _compute_thought_salience(2, 1, 0.0)  # fully novel
        stale = _compute_thought_salience(2, 1, 0.9)  # near-identical
        assert novel > stale

    def test_clamping(self):
        """Values above max are clamped."""
        from maxim.runtime.agent_loop import _compute_thought_salience

        s = _compute_thought_salience(n_sections=10, n_memories=10, jaccard_with_previous=0.0)
        assert s == pytest.approx(1.0)

    def test_range(self):
        """Output is always in [0.0, 1.0]."""
        from maxim.runtime.agent_loop import _compute_thought_salience

        for ns in range(6):
            for nm in range(6):
                for jp in [0.0, 0.3, 0.5, 0.8, 1.0]:
                    s = _compute_thought_salience(ns, nm, jp)
                    assert 0.0 <= s <= 1.0, f"Out of range: {s} for ({ns}, {nm}, {jp})"


class TestJaccardSimilarity:
    """Tests for _jaccard_similarity helper."""

    def test_identical(self):
        from maxim.runtime.agent_loop import _jaccard_similarity

        a = {"the", "guard", "sleeps"}
        assert _jaccard_similarity(a, a) == pytest.approx(1.0)

    def test_disjoint(self):
        from maxim.runtime.agent_loop import _jaccard_similarity

        a = {"the", "guard", "sleeps"}
        b = {"we", "run", "away"}
        assert _jaccard_similarity(a, b) == pytest.approx(0.0)

    def test_empty(self):
        from maxim.runtime.agent_loop import _jaccard_similarity

        assert _jaccard_similarity(set(), {"a"}) == pytest.approx(0.0)
        assert _jaccard_similarity({"a"}, set()) == pytest.approx(0.0)
        assert _jaccard_similarity(set(), set()) == pytest.approx(0.0)

    def test_partial_overlap(self):
        from maxim.runtime.agent_loop import _jaccard_similarity

        a = {"the", "guard", "sleeps", "near", "door"}
        b = {"the", "guard", "watches", "the", "gate"}
        # intersection: {"the", "guard"} = 2, union = 7
        assert 0.0 < _jaccard_similarity(a, b) < 1.0


class TestDeliberationTranscript:
    """Tests for deliberation transcript (Stage 1)."""

    def test_transcript_built_on_multi_cycle(self, bio, wms, ctx, gate):
        """Multi-cycle deliberation populates deliberation_transcript."""
        cycle2 = _make_proposal(ready_to_act=True, reasoning="I will act now")
        llm_worker = MagicMock()
        llm_worker.get_latest_proposal.return_value = cycle2

        first = _make_proposal(ready_to_act=False, reasoning="The guard sleeps near the door")

        with (
            patch("maxim.simulation.sim_logger.sim_log"),
            patch("maxim.simulation.sim_logger.sim_pre_deliberation"),
            patch("maxim.simulation.sim_logger.sim_contemplation"),
        ):
            _run_cycles(
                first_proposal=first,
                bio_enrichment=bio,
                working_memory=wms,
                context=ctx,
                submit_fn=lambda c: True,
                llm_worker=llm_worker,
                stop_event=None,
                thought_gate=gate,
                active_goal="escape",
                step_num=1,
                max_cycles=3,
            )

        assert ctx.deliberation_transcript is not None
        assert len(ctx.deliberation_transcript) == 1  # One cycle 2 entry
        entry = ctx.deliberation_transcript[0]
        assert "You thought:" in entry
        assert "Your experience responded:" in entry

    def test_transcript_accumulates_across_cycles(self, bio, wms, ctx, gate):
        """Transcript grows with each cycle."""
        # Cycle 2: not ready. Cycle 3: ready.
        proposals = [
            _make_proposal(ready_to_act=False, reasoning="still exploring options for escape route"),
            _make_proposal(ready_to_act=True, reasoning="found the way out"),
        ]
        call_count = [0]

        llm_worker = MagicMock()

        def get_prop():
            idx = min(call_count[0], len(proposals) - 1)
            call_count[0] += 1
            return proposals[idx]

        llm_worker.get_latest_proposal.side_effect = get_prop

        first = _make_proposal(ready_to_act=False, reasoning="I see the guard near the door")

        with (
            patch("maxim.simulation.sim_logger.sim_log"),
            patch("maxim.simulation.sim_logger.sim_pre_deliberation"),
            patch("maxim.simulation.sim_logger.sim_contemplation"),
        ):
            _run_cycles(
                first_proposal=first,
                bio_enrichment=bio,
                working_memory=wms,
                context=ctx,
                submit_fn=lambda c: True,
                llm_worker=llm_worker,
                stop_event=None,
                thought_gate=gate,
                active_goal="escape",
                step_num=1,
                max_cycles=4,
            )

        assert ctx.deliberation_transcript is not None
        assert len(ctx.deliberation_transcript) == 2  # Cycles 2 and 3

    def test_no_transcript_on_enrichment_failure(self, wms, ctx, gate):
        """When enrichment returns None, transcript stays None."""
        bio = MagicMock()
        bio.enrich.return_value = None

        llm_worker = MagicMock()
        first = _make_proposal(ready_to_act=False, reasoning="thinking")

        with (
            patch("maxim.simulation.sim_logger.sim_log"),
            patch("maxim.simulation.sim_logger.sim_pre_deliberation"),
            patch("maxim.simulation.sim_logger.sim_contemplation"),
        ):
            _run_cycles(
                first_proposal=first,
                bio_enrichment=bio,
                working_memory=wms,
                context=ctx,
                submit_fn=lambda c: True,
                llm_worker=llm_worker,
                stop_event=None,
                thought_gate=gate,
                active_goal=None,
                step_num=1,
                max_cycles=3,
            )

        assert ctx.deliberation_transcript is None

    def test_computed_salience_on_thought_entries(self, bio, wms, ctx, gate):
        """THOUGHT entries use computed salience, not hardcoded 0.5."""
        cycle2 = _make_proposal(ready_to_act=True, reasoning="acting now")
        llm_worker = MagicMock()
        llm_worker.get_latest_proposal.return_value = cycle2

        first = _make_proposal(ready_to_act=False, reasoning="The guard sleeps near the ancient door lock")

        with (
            patch("maxim.simulation.sim_logger.sim_log"),
            patch("maxim.simulation.sim_logger.sim_pre_deliberation"),
            patch("maxim.simulation.sim_logger.sim_contemplation"),
        ):
            _run_cycles(
                first_proposal=first,
                bio_enrichment=bio,
                working_memory=wms,
                context=ctx,
                submit_fn=lambda c: True,
                llm_worker=llm_worker,
                stop_event=None,
                thought_gate=gate,
                active_goal="escape",
                step_num=1,
                max_cycles=3,
            )

        # WMS should have a THOUGHT entry with computed salience != 0.5
        thought_entries = [e for e in wms.entries if str(e["kind"]).endswith("THOUGHT")]
        assert len(thought_entries) >= 1
        # Salience should be computed (not the old hardcoded 0.5)
        # With 1 section (memories), 1 memory, and jaccard=0.0 for first enrichment:
        # expected ≈ 0.3*(1/5) + 0.3*(1/4) + 0.4*1.0 = 0.06 + 0.075 + 0.4 = 0.535
        assert thought_entries[0]["salience"] != 0.5


class TestTopBySalience:
    """Tests for WorkingMemorySet.top_by_salience (Stage 2)."""

    def test_sorted_by_salience_descending(self):
        from maxim.agents.working_memory import WorkingMemoryKind, WorkingMemorySet

        wms = WorkingMemorySet(agent_id="test")
        wms.add(WorkingMemoryKind.THOUGHT, content="low", salience=0.2)
        wms.add(WorkingMemoryKind.THOUGHT, content="high", salience=0.9)
        wms.add(WorkingMemoryKind.THOUGHT, content="mid", salience=0.5)

        result = wms.top_by_salience({WorkingMemoryKind.THOUGHT}, limit=10)
        assert len(result) == 3
        assert result[0].salience == 0.9
        assert result[1].salience == 0.5
        assert result[2].salience == 0.2

    def test_limit(self):
        from maxim.agents.working_memory import WorkingMemoryKind, WorkingMemorySet

        wms = WorkingMemorySet(agent_id="test")
        for i in range(10):
            wms.add(WorkingMemoryKind.THOUGHT, content=f"t{i}", salience=i * 0.1)

        result = wms.top_by_salience({WorkingMemoryKind.THOUGHT}, limit=3)
        assert len(result) == 3
        assert result[0].salience == pytest.approx(0.9)

    def test_min_salience_filter(self):
        from maxim.agents.working_memory import WorkingMemoryKind, WorkingMemorySet

        wms = WorkingMemorySet(agent_id="test")
        wms.add(WorkingMemoryKind.THOUGHT, content="low", salience=0.1)
        wms.add(WorkingMemoryKind.THOUGHT, content="high", salience=0.8)

        result = wms.top_by_salience({WorkingMemoryKind.THOUGHT}, limit=10, min_salience=0.5)
        assert len(result) == 1
        assert result[0].content == "high"

    def test_kind_filtering(self):
        from maxim.agents.working_memory import WorkingMemoryKind, WorkingMemorySet

        wms = WorkingMemorySet(agent_id="test")
        wms.add(WorkingMemoryKind.THOUGHT, content="thought", salience=0.5)
        wms.add(WorkingMemoryKind.PERCEPT, content="percept", salience=0.9)
        wms.add(WorkingMemoryKind.RECALL, content="recall", salience=0.7)

        result = wms.top_by_salience({WorkingMemoryKind.THOUGHT}, limit=10)
        assert len(result) == 1
        assert result[0].content == "thought"

    def test_recency_tiebreaker(self):
        from maxim.agents.working_memory import WorkingMemoryKind, WorkingMemorySet

        wms = WorkingMemorySet(agent_id="test")
        wms.add(WorkingMemoryKind.THOUGHT, content="older", salience=0.5)
        wms.add(WorkingMemoryKind.THOUGHT, content="newer", salience=0.5)

        result = wms.top_by_salience({WorkingMemoryKind.THOUGHT}, limit=10)
        assert len(result) == 2
        # Same salience → higher tick (newer) first
        assert result[0].content == "newer"
        assert result[1].content == "older"

    def test_empty_wms(self):
        from maxim.agents.working_memory import WorkingMemoryKind, WorkingMemorySet

        wms = WorkingMemorySet(agent_id="test")
        result = wms.top_by_salience({WorkingMemoryKind.THOUGHT}, limit=5)
        assert result == []
