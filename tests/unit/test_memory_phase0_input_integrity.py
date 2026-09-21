"""Memory Phase 0 — the capture inputs the strength model will read are stored correctly.

docs/plans/memory_strength_and_forgetting.md Phase 0 (the line's entry condition). Each test drives
the REAL MemoryAgent into a REAL Hippocampus, not a hand-built call, so a fix that only works on a
recipe cannot pass:

- #813 MemoryAgent passed salience/novelty in `state`, but capture_from_loop reads the observation:
  every capture was stored at 0.5/0.5.
- #814 a dict result read as success (hasattr(dict, "success") is False): failures stored as successes.
- #815 `action` was passed as a bare tool name: `.get` on a str raised, so failed tool results never
  reached the hippocampus at all.
- #816 reinforcing a compressed ATL concept raised AttributeError and aborted the extraction.
- #817 the forming pool grew without bound (staged formation is Dormant).
"""

from __future__ import annotations

import pytest

from maxim.agents.bus import AgentBus, ToolResult
from maxim.agents.memory_agent import MemoryAgent
from maxim.memory.hippocampus import Hippocampus


@pytest.fixture
def agent_and_hippocampus():
    agent = MemoryAgent(AgentBus())
    hippocampus = Hippocampus.empty()
    agent.connect_hippocampus(hippocampus)
    return agent, hippocampus


def _last_memory(agent, hippocampus):
    assert agent._recent_ids, "nothing was captured"
    return hippocampus.get(agent._recent_ids[-1])


def test_failed_tool_result_is_captured_as_a_failure_with_its_salience(agent_and_hippocampus) -> None:
    agent, hippocampus = agent_and_hippocampus
    agent._on_tool_result(ToolResult(tool_call_id="1", tool_name="http_fetch", success=False, error="boom"))

    memory = _last_memory(agent, hippocampus)  # #815: this used to raise before storing anything
    assert memory.action.tool_name == "http_fetch"
    assert memory.outcome.success is False  # #814
    assert memory.outcome.error == "boom"
    assert memory.perception.salience == pytest.approx(0.8)  # #813: was stored as 0.5


def test_capture_from_loop_rejects_a_non_mapping_action_loudly() -> None:
    hippocampus = Hippocampus.empty()
    with pytest.raises(TypeError, match="action must be a mapping"):
        hippocampus.capture_from_loop(
            observation={}, state=None, intent={}, decision={}, action="http_fetch", result={}
        )


def test_capture_from_loop_reads_success_from_a_mapping_result() -> None:
    hippocampus = Hippocampus.empty()
    mid = hippocampus.capture_from_loop(
        observation={"salience": 0.6},
        state=None,
        intent={},
        decision={},
        action={"tool": "t"},
        result={"success": False, "error": "nope"},
    )
    memory = hippocampus.get(mid)
    assert memory.outcome.success is False and memory.outcome.error == "nope"


def test_the_forming_pool_is_bounded(agent_and_hippocampus) -> None:
    from maxim.agents.bus import MemoryTier, WorkingMemoryEntry

    agent, _ = agent_and_hippocampus
    for i in range(agent._MAX_FORMING * 3):
        agent._forming_pool[f"run-{i}"] = WorkingMemoryEntry(record=None, tier=MemoryTier.FORMING)
    agent._flush_completed_from_pool()
    assert len(agent._forming_pool) == agent._MAX_FORMING
    assert f"run-{agent._MAX_FORMING * 3 - 1}" in agent._forming_pool  # the NEWEST are kept


def test_reinforcing_a_compressed_concept_no_longer_aborts_extraction() -> None:
    """Through a REAL ATL: compress a concept exactly as ATL.consolidate does, then re-register it."""
    from maxim.memory.atl import ATL
    from maxim.memory.concept_extractor import ConceptExtractor
    from maxim.memory.cross_layer import CrossLayerGraph
    from maxim.memory.semantic_types import CompressedSemantic

    atl = ATL()
    cross_layer = CrossLayerGraph()
    extractor = ConceptExtractor(atl, cross_layer, start_worker=False)
    extractor._register_concept("water", "object", "m1", record=None)
    [cid] = [c for c in atl._concepts]
    edges_before = cross_layer.stats()["total_edges"]

    # ATL.consolidate's compression step, verbatim in effect.
    atl._concepts[cid] = CompressedSemantic.from_semantic(atl._concepts[cid], 0)
    assert not hasattr(atl._concepts[cid], "reinforce")

    # A later episode mentioning the same concept: find_or_create returns the compressed hit.
    # Before #816's crash fix this raised AttributeError and aborted the episode's extraction.
    extractor._register_concept("water", "object", "m2", record=None)
    assert isinstance(atl._concepts[cid], CompressedSemantic)  # still compressed, not strengthened
    # The extraction ran through to the cross-layer step: m2's edge to the concept was recorded.
    assert cross_layer.stats()["total_edges"] == edges_before + 1


def test_percept_novelty_is_stored_not_hardcoded(agent_and_hippocampus) -> None:
    from maxim.agents.bus import Percept

    agent, hippocampus = agent_and_hippocampus
    percept = Percept(timestamp=1.0, source="test", transcript_chunk="a new thing", novelty=0.83)
    mid = agent._capture_to_hippocampus({"note": "x"}, 0.6, 0.05, "percept", percept)
    assert hippocampus.get(mid).perception.novelty == pytest.approx(0.83)  # was hard-coded 0.5


def test_the_forming_pool_bound_holds_through_the_real_insertion_path(agent_and_hippocampus) -> None:
    from maxim.agents.bus import Percept

    agent, _ = agent_and_hippocampus
    for i in range(agent._MAX_FORMING + 10):
        percept = Percept(timestamp=float(i), source="test", transcript_chunk=f"event {i}")
        agent._begin_memory_formation(percept, f"run-{i}")
    assert len(agent._forming_pool) == agent._MAX_FORMING  # not MAX + 1 (the flush runs before the insert)
    assert f"run-{agent._MAX_FORMING + 9}" in agent._forming_pool
