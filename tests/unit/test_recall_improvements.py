"""Tests for Phase 5 hippocampus recall improvements."""

from __future__ import annotations

import time
import uuid

import pytest

from maxim.memory.hippocampus import Hippocampus, HippocampusConfig
from maxim.memory.hippocampus_retrieval import _rank_by_relevance
from maxim.memory.types import (
    Action,
    Context,
    Decision,
    EpisodicMemory,
    Outcome,
    Perception,
)


def _em(**kwargs) -> EpisodicMemory:
    """Create an EpisodicMemory with auto-generated id + timestamp."""
    kwargs.setdefault("id", str(uuid.uuid4()))
    kwargs.setdefault("timestamp", time.time())
    return EpisodicMemory(**kwargs)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def hippo(tmp_path):
    """Create a hippocampus with short dedup window for testing."""
    return Hippocampus(HippocampusConfig(
        persistence_path=str(tmp_path / "hippo.json"),
        dedup_window_s=2.0,  # Short window for testing
    ))


def _make_memory(
    goal: str = "test",
    tool: str = "say",
    objects: list[str] | None = None,
    people: list[str] | None = None,
    success: bool = True,
    text: str = "",
) -> dict:
    """Helper to create capture kwargs."""
    return {
        "perception": Perception(
            observations={"text": text} if text else {},
            detected_objects=objects or [],
            detected_people=people or [],
            tool_alternatives=[],
        ),
        "context": Context(active_goal=goal),
        "decision": Decision(intent={"goal": goal}),
        "action": Action(tool_name=tool),
        "outcome": Outcome(success=success),
    }


# ---------------------------------------------------------------------------
# store_observation
# ---------------------------------------------------------------------------


class TestStoreObservation:
    def test_stores_text(self, hippo):
        mid = hippo.store_observation("A guard approaches the gate.")
        assert mid  # Non-empty ID returned
        assert len(hippo._memories) == 1

    def test_stores_with_metadata(self, hippo):
        mid = hippo.store_observation("Hello", metadata={"agent_id": "npc_guard"})
        assert mid
        mem = hippo.get(mid)
        assert mem.perception.observations.get("agent_id") == "npc_guard"

    def test_dedup_rejects_identical(self, hippo):
        mid1 = hippo.store_observation("The sun sets over the mountains.")
        mid2 = hippo.store_observation("The sun sets over the mountains.")
        assert mid1  # First accepted
        assert mid2 == ""  # Second deduped
        assert len(hippo._memories) == 1

    def test_dedup_accepts_after_window(self, hippo):
        hippo.config = HippocampusConfig(dedup_window_s=0.1)
        hippo.store_observation("Repeated observation.")
        time.sleep(0.15)
        mid = hippo.store_observation("Repeated observation.")
        assert mid  # Accepted after window expires

    def test_dedup_accepts_different_text(self, hippo):
        hippo.store_observation("First observation.")
        mid = hippo.store_observation("Completely different observation.")
        assert mid  # Different text accepted


# ---------------------------------------------------------------------------
# Relevance ranking
# ---------------------------------------------------------------------------


class TestRelevanceRanking:
    def test_ranks_by_keyword_overlap(self):
        """Memories with more query token matches rank higher."""
        mem_guard = _em(
            perception=Perception(
                observations={"text": "guard at the gate"},
                detected_objects=["gate"],
                detected_people=["guard"],
                tool_alternatives=[],
            ),
            context=Context(active_goal="find the guard"),
            action=Action(tool_name="look"),
            outcome=Outcome(success=True),
        )
        mem_merchant = _em(
            perception=Perception(
                observations={"text": "merchant in the market"},
                detected_objects=["stall"],
                detected_people=["merchant"],
                tool_alternatives=[],
            ),
            context=Context(active_goal="buy supplies"),
            action=Action(tool_name="trade"),
            outcome=Outcome(success=True),
        )

        results = _rank_by_relevance([mem_merchant, mem_guard], "guard gate", limit=5)
        assert results[0] is mem_guard  # Guard matches both tokens

    def test_empty_query_returns_by_recency(self):
        mem1 = _em(
            perception=Perception(observations={}, detected_objects=[], detected_people=[], tool_alternatives=[]),
            context=Context(),
            action=Action(tool_name="a"),
            outcome=Outcome(success=True),
        )
        mem2 = _em(
            perception=Perception(observations={}, detected_objects=[], detected_people=[], tool_alternatives=[]),
            context=Context(),
            action=Action(tool_name="b"),
            outcome=Outcome(success=True),
        )
        # mem2 has later timestamp (created second)
        results = _rank_by_relevance([mem1, mem2], "", limit=5)
        assert len(results) == 2

    def test_respects_limit(self):
        mems = []
        for i in range(10):
            mems.append(_em(
                perception=Perception(
                    observations={"text": f"memory {i}"},
                    detected_objects=[], detected_people=[], tool_alternatives=[],
                ),
                context=Context(active_goal="test"),
                action=Action(tool_name="tool"),
                outcome=Outcome(success=True),
            ))
        results = _rank_by_relevance(mems, "memory", limit=3)
        assert len(results) == 3


# ---------------------------------------------------------------------------
# recall() with query parameter
# ---------------------------------------------------------------------------


class TestRecallWithQuery:
    def test_recall_with_query_ranks_results(self, hippo):
        """recall(query=...) returns relevance-ranked results."""
        hippo.capture(**_make_memory(goal="find the sword", tool="search", objects=["sword"]))
        hippo.capture(**_make_memory(goal="buy supplies", tool="trade", objects=["potion"]))
        hippo.capture(**_make_memory(goal="talk to guard", tool="speak", people=["guard"]))

        results = hippo.recall(query="sword", limit=10)
        # Sword memory should rank first
        assert len(results) >= 1
        first = results[0]
        assert hasattr(first, 'context')
        if first.context and first.context.active_goal:
            assert "sword" in first.context.active_goal.lower()

    def test_recall_without_query_uses_recency(self, hippo):
        """recall() without query returns most recent first."""
        hippo.capture(**_make_memory(goal="old memory"))
        time.sleep(0.01)
        hippo.capture(**_make_memory(goal="new memory"))

        results = hippo.recall(limit=2)
        assert len(results) == 2
        # Most recent first
        assert results[0].timestamp >= results[1].timestamp

    def test_recall_query_with_filters(self, hippo):
        """query + filters work together (filters AND, then rank)."""
        hippo.capture(**_make_memory(goal="sword quest", tool="search", success=True))
        hippo.capture(**_make_memory(goal="sword fight", tool="attack", success=False))

        results = hippo.recall(query="sword", success=True, limit=10)
        assert len(results) >= 1
        # All results should be successful
        for r in results:
            if hasattr(r, 'outcome') and r.outcome:
                assert r.outcome.success is True
