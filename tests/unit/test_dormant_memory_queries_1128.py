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

from maxim.agents.bus import AgentBus
from maxim.agents.memory_agent import MemoryAgent
from maxim.memory.hippocampus import Hippocampus


def test_building_context_does_not_count_a_hippocampus_access():
    agent = MemoryAgent(AgentBus())
    hippocampus = Hippocampus.empty()
    agent.connect_hippocampus(hippocampus)
    memory_id = agent._add_memory({"note": "the kettle whistled"}, 0.7, 0.05, "percept")
    [record] = hippocampus.recall_by_ids([memory_id])
    before = record.access_count
    agent.build_context()
    assert record.access_count == before


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


# -- the production wiring (review fold): a dropped kwarg would silently restore counting ----------


def test_memory_agent_asks_the_hub_for_uncounted_concept_reads():
    from types import SimpleNamespace

    seen: dict = {}

    def _build(**kwargs):
        seen.update(kwargs)
        return []

    agent = MemoryAgent(AgentBus())
    agent._memory_hub = SimpleNamespace(build_concept_context=_build)
    agent._build_concept_context([{"label": "kettle"}], [])
    assert seen.get("count_access") is False


def test_the_hub_passes_count_access_through_to_the_builder():
    from types import SimpleNamespace

    from maxim.integration.memory_hub import MemoryHub

    seen: dict = {}

    def _build(**kwargs):
        seen.update(kwargs)
        return []

    hub = SimpleNamespace(_concept_context_builder=SimpleNamespace(build=_build))
    MemoryHub.build_concept_context(hub, detected_objects=["kettle"], count_access=False)
    assert seen.get("count_access") is False
