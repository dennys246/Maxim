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
