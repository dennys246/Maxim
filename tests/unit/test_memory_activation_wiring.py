"""Honest activation reaches the real consumption points (memory-strength plan Phase 1).

Each test drives a REAL store through the site's own code and reads the counters back: a site that
stops calling ``activate`` -- or starts counting what it did not render -- fails here. What each
site counts is its render cap, not its retrieval limit: surfaced-but-unshown is not a use.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from uuid import uuid4

import pytest

from maxim.memory.atl import ATL, ATLConfig
from maxim.memory.hippocampus import Hippocampus, HippocampusConfig
from maxim.memory.semantic_types import Concept
from maxim.memory.types import Action, Context, Decision, EpisodicMemory, Outcome, Perception


@pytest.fixture
def hippo():
    return Hippocampus(HippocampusConfig(persistence_path=None))


@pytest.fixture
def atl():
    return ATL(ATLConfig(persistence_path=None))


def _episode(hippo, *, goal="get water", success=True, reflection=None, metadata=None) -> EpisodicMemory:
    result = {"reflection": reflection} if reflection else None
    ep = EpisodicMemory(
        id=str(uuid4()),
        timestamp=time.time(),
        perception=Perception(detected_objects=["mug"]),
        context=Context(active_goal=goal),
        decision=Decision(intent={"goal": goal}),
        action=Action(tool_name="grasp"),
        outcome=Outcome(success=success, result=result),
        metadata=metadata or {},
    )
    hippo.capture(record=ep)
    return ep


def _counts(store, ids):
    return [r.activation_count for r in store.recall_by_ids(ids)]


def test_enrichment_counts_exactly_what_the_formatter_renders(hippo, atl):
    from maxim.integration.bio_enrichment import (
        BioEnrichmentPipeline,
        ConceptLink,
        EnrichmentResult,
        EpisodicSummary,
    )

    eps = [_episode(hippo) for _ in range(4)]
    cid, _ = atl.find_or_create(name="mug", category="object")
    pipeline = BioEnrichmentPipeline(hippocampus=hippo, atl=atl)
    result = EnrichmentResult(
        memories=tuple(EpisodicSummary(memory_id=e.id, summary="s", valence=0.0, relevance=1.0) for e in eps),
        concepts=(ConceptLink(concept="mug", category="object", activation=0.7, concept_id=cid),),
    )
    text = pipeline.format_thought_response(result)
    assert text.count("[~]") == 3  # the render cap
    assert _counts(hippo, [e.id for e in eps]) == [1, 1, 1, 0]
    assert atl.recall_by_ids([cid])[0].activation_sources == {"enrichment": 1}


def test_enrichment_atl_query_carries_the_concept_id(atl):
    from maxim.integration.bio_enrichment import BioEnrichmentPipeline

    cid, _ = atl.find_or_create(name="mug", category="object")
    links = BioEnrichmentPipeline(atl=atl)._query_atl(["mug"])
    assert [link.concept_id for link in links] == [cid]


def test_empty_render_counts_nothing(hippo):
    from maxim.integration.bio_enrichment import BioEnrichmentPipeline, EnrichmentResult

    assert BioEnrichmentPipeline(hippocampus=hippo).format_thought_response(EnrichmentResult()) == ""


def test_memory_recall_tool_counts_what_it_returns(hippo):
    from maxim.tools.introspection import MemoryRecallTool

    eps = [_episode(hippo) for _ in range(3)]
    out = MemoryRecallTool(hippocampus=hippo).execute(limit=2)
    assert out.output["count"] == 2
    assert sorted(_counts(hippo, [e.id for e in eps])) == [0, 1, 1]


def test_concept_query_tool_counts_what_it_returns(atl):
    from maxim.tools.introspection import ConceptQueryTool

    cid, _ = atl.find_or_create(name="mug", category="object")
    out = ConceptQueryTool(atl=atl).execute(name="mug")
    assert out.output["count"] == 1
    assert atl.recall_by_ids([cid])[0].activation_sources == {"tool": 1}


def test_replan_counts_only_episodes_that_reach_the_prompt(hippo):
    from maxim.runtime.loop_state import _build_replan_context

    @dataclass
    class FakeResult:
        success: bool = False
        error: str = "failed"
        error_kind: str = "tool_error"
        output: str = ""

    rendered = _episode(hippo, metadata={"plan_actions": [{"tool_name": "grasp", "params": {}}]})
    skipped = _episode(hippo)  # no extractable actions -> not in the prompt
    ctx = _build_replan_context({"goal": "get water"}, {"tool_name": "grasp"}, FakeResult(), None, hippocampus=hippo)
    assert len(ctx.prior_attempt_actions) == 1
    assert _counts(hippo, [rendered.id, skipped.id]) == [1, 0]


def test_pattern_completion_counts_the_completed_episodes_not_the_cue(hippo, atl):
    from maxim.memory.pattern_completer import PatternCompleter

    cid, _ = atl.find_or_create(name="mug", category="object")
    concept = atl.get(cid)
    assert isinstance(concept, Concept)
    ep = _episode(hippo)
    concept.add_ref("hippocampus", ep.id)
    forming = EpisodicMemory(id="forming", timestamp=time.time(), perception=Perception(detected_objects=["mug"]))

    predictions = PatternCompleter(atl=atl, layers={"hippocampus": hippo}).complete(forming)
    assert [p.source_episode_id for p in predictions] == [ep.id]
    assert hippo.recall_by_ids([ep.id])[0].activation_sources == {"prediction": 1}
    assert atl.recall_by_ids([cid])[0].activation_count == 0  # the cue is not a use


class _StubLLM:
    def __init__(self):
        self.prompts = []

    def generate_structured(self, prompt, schema):
        self.prompts.append(prompt)
        return [{"tool_name": "grasp", "params": {}}]


def test_adaptive_planner_gathering_is_not_a_use(hippo, monkeypatch):
    from maxim.planning.adaptive_planner import AdaptivePlanner

    eps = [_episode(hippo) for _ in range(4)]
    monkeypatch.setattr(hippo, "recall_associated", lambda seed_ids, limit: [(e, 1.0) for e in eps])
    pctx = AdaptivePlanner(hippocampus=hippo)._gather_context({"description": "get water"}, "grasp", {}, None)
    assert len(pctx.successful_strategies) == 4
    assert _counts(hippo, [e.id for e in eps]) == [0, 0, 0, 0]  # gathered, never rendered


def test_adaptive_planner_counts_the_three_its_decomposition_prompt_renders(hippo, monkeypatch):
    from maxim.planning.adaptive_planner import AdaptivePlanner

    eps = [_episode(hippo) for _ in range(4)]
    monkeypatch.setattr(hippo, "recall_associated", lambda seed_ids, limit: [(e, 1.0) for e in eps])
    llm = _StubLLM()
    plan = AdaptivePlanner(hippocampus=hippo, llm=llm).decompose(
        {"description": "get water", "tool_name": "grasp"}, None, 0
    )
    assert plan is not None and len(llm.prompts) == 1
    assert _counts(hippo, [e.id for e in eps]) == [1, 1, 1, 0]
    assert hippo.recall_by_ids([eps[0].id])[0].activation_sources == {"planner": 1}


def test_counting_can_never_cost_the_consumer_its_content(hippo, monkeypatch):
    from maxim.integration.bio_enrichment import BioEnrichmentPipeline, EnrichmentResult, EpisodicSummary

    ep = _episode(hippo)

    def boom(ids, *, source):
        raise RuntimeError("store down")

    monkeypatch.setattr(hippo, "activate", boom)
    result = EnrichmentResult(memories=(EpisodicSummary(memory_id=ep.id, summary="s", valence=0.0, relevance=1.0),))
    assert "Your experience suggests:" in BioEnrichmentPipeline(hippocampus=hippo).format_thought_response(result)
