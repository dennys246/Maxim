"""#845: memory consumers that never delivered, against REAL stores.

The old tests passed only because their fakes had fields the real types lack
(``FakeMemory._Context.goal``; the real ``Context`` has ``active_goal``), and the
ExamineTool test asserted only ``success``, which holds whether or not memory
contributed. Every test here uses a real ``Hippocampus`` / ``ATL``.
"""

from __future__ import annotations

import pytest

from maxim.memory.hippocampus import Hippocampus, HippocampusConfig
from maxim.tools.narrative import ExamineTool


def _hippocampus() -> Hippocampus:
    return Hippocampus(HippocampusConfig(auto_save_after_sleep=False))


# -- ExamineTool: wired (owner decision 2026-10-06) ---------------------------


@pytest.mark.xfail(strict=True, reason="#845 item 2: Stage 2 reads context.goal, which Context does not have")
def test_examine_recalls_what_the_matching_memory_holds():
    h = _hippocampus()
    assert h.store_observation("The brass key lies under the loose floorboard.")
    out = ExamineTool(hippocampus=h).execute(target="brass key")
    assert out.success
    assert "You recall: The brass key lies under the loose floorboard." in out.output["observation"]


@pytest.mark.xfail(strict=True, reason="#845 item 2: nothing reaches the LLM, so nothing is counted as used")
def test_a_recalled_memory_is_counted_as_used_by_a_tool():
    h = _hippocampus()
    mid = h.store_observation("The brass key lies under the loose floorboard.")
    ExamineTool(hippocampus=h).execute(target="brass key")
    [record] = h.recall_by_ids([mid])
    assert record.activation_sources.get("tool") == 1


def test_examine_without_a_matching_memory_still_reports_nothing_notable():
    h = _hippocampus()
    assert h.store_observation("A quiet meadow at dusk.")
    out = ExamineTool(hippocampus=h).execute(target="dragon")
    assert out.success
    assert "don't see anything notable" in out.output["observation"]


# -- Knowledge context: lookup fixed, consumer Dormant (owner decision 2026-10-06) ----------


def _knowledge_context_with_hit(atl):
    """Run ``_build_knowledge_context`` with a hub whose cross-layer activation returns ``atl``'s id."""
    import time
    from types import SimpleNamespace

    from maxim.agents.memory_agent import MemoryAgent
    from maxim.memory.semantic_types import SemanticMemory

    # Confidence BELOW the fallback's min_confidence=0.5, so only the cross-layer path can find it.
    concept = SemanticMemory(
        id="c-brass-key", timestamp=time.time(), name="brass_key", category="object", confidence=0.3
    )
    atl.store(concept)
    stored_count = concept.access_count  # store() itself counts one access

    class _Hub:
        angular_gyrus = None

        def __init__(self) -> None:
            self.atl = atl

        def recall_with_knowledge(self, **_kwargs):
            return {"atl": [("c-brass-key", 0.9)]}

    agent = SimpleNamespace(
        _memory_hub=_Hub(),
        _forming_pool={"x": SimpleNamespace(_hippocampus_id="ep-1")},
        _get_concept_relationships=MemoryAgent._get_concept_relationships,
    )
    return MemoryAgent._build_knowledge_context(agent), concept, stored_count


@pytest.mark.xfail(strict=True, reason="#845 item 3: atl.recall(name=<id>) looks an id up as a NAME")
def test_a_cross_layer_atl_hit_is_found_by_its_id():
    from maxim.memory.atl import ATL

    entries, _, _ = _knowledge_context_with_hit(ATL())
    assert [e["concept_name"] for e in entries] == ["brass_key"]


def test_the_lookup_does_not_count_as_an_access():
    """The result is discarded by its only caller, so the lookup must not move retention."""
    from maxim.memory.atl import ATL

    _, concept, stored_count = _knowledge_context_with_hit(ATL())
    assert concept.access_count == stored_count


# -- recall_deep: Dormant, its text fallback fixed (owner decision 2026-10-06) ----------------


@pytest.mark.xfail(strict=True, reason="#845 item 4: recall_similar(<str>) raised and was swallowed")
def test_recall_deep_text_fallback_finds_a_matching_memory():
    from types import SimpleNamespace

    from maxim.agents.exec_agent import ExecAgent

    h = _hippocampus()
    mid = h.store_observation("The brass key lies under the loose floorboard.")
    agent = SimpleNamespace(_hippocampus=h)
    assert [m.id for m in ExecAgent.recall_deep(agent, "brass key")] == [mid]
