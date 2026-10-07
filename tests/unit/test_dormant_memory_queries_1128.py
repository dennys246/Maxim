"""#1128: ``MemoryAgent``'s Dormant context queries keep running but no longer count accesses.

They fill ``relevant_memories``, ``concept_context``, ``knowledge_context``, ``valence_context`` and
``causal_context``, which no production path reads (#845), on every tick through
``ExecAgent.propose_intent``. They still run, because ``_build_concept_context`` is concept grounding's only
production caller outside ``agentic_runtime``; but their reads of records nothing uses touched them
(``Hippocampus.get``, ``AngularGyrus.recall``, the concept builder's ``ATL.get``), moving default retention
every tick (owner decision 2026-10-07: keep running, read uncounted).
"""

from __future__ import annotations

import time

import pytest

from maxim.agents.bus import AgentBus
from maxim.agents.memory_agent import MemoryAgent
from maxim.memory.hippocampus import Hippocampus


@pytest.mark.xfail(strict=True, reason="#1128: building the Dormant context touches a Hippocampus record")
def test_building_context_does_not_count_a_hippocampus_access():
    agent = MemoryAgent(AgentBus())
    hippocampus = Hippocampus.empty()
    agent.connect_hippocampus(hippocampus)
    memory_id = agent._add_memory({"note": "the kettle whistled"}, 0.7, 0.05, "percept")
    [record] = hippocampus.recall_by_ids([memory_id])
    before = record.access_count
    agent.build_context()
    assert record.access_count == before


@pytest.mark.xfail(strict=True, reason="#1128: AngularGyrus.recall has no uncounted read")
def test_an_angular_gyrus_recall_can_read_without_touching():
    from maxim.math.angular_gyrus import AngularGyrus
    from maxim.math.math_types import MathMemory
    from maxim.math.types import MathCategory

    ag = AngularGyrus()
    ag.store(MathMemory(id="p1", timestamp=time.time(), name="p", category=MathCategory.PATTERN, confidence=0.9))
    [record] = ag.recall_by_ids(["p1"])
    before = record.access_count  # storing counts one access
    assert [r.id for r in ag.recall(limit=10, category="PATTERN", touch=False) if r.id == "p1"] == ["p1"]
    assert record.access_count == before


@pytest.mark.xfail(strict=True, reason="#1128: the concept builder's relationship lookup always touches")
def test_concept_relationships_can_be_read_without_touching():
    from maxim.memory.atl import ATL
    from maxim.memory.concept_context import ConceptContextBuilder
    from maxim.memory.semantic_types import SemanticMemory

    atl = ATL()
    atl.store(SemanticMemory(id="a", timestamp=time.time(), name="kettle", category="object"))
    atl.store(SemanticMemory(id="b", timestamp=time.time(), name="stove", category="object"))
    assert atl.define_relationship("a", "b", "RELATED_TO")
    [other] = atl.recall_by_ids(["b"])
    before = other.access_count
    summaries = ConceptContextBuilder(atl=atl, layers={})._collect_relationships(
        atl.recall_by_ids(["a"])[0], count_access=False
    )
    assert {s["target"] for s in summaries} == {"stove"}
    assert other.access_count == before
