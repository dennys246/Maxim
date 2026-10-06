"""#845: memory consumers that never delivered, against REAL stores.

The old tests passed only because their fakes had fields the real types lack
(``FakeMemory._Context.goal``; the real ``Context`` has ``active_goal``), and the
ExamineTool test asserted only ``success``, which holds whether or not memory
contributed. Every test here uses a real ``Hippocampus`` / ``ATL``.
"""

from __future__ import annotations

from types import SimpleNamespace

from maxim.memory.hippocampus import Hippocampus, HippocampusConfig
from maxim.tools.narrative import ExamineTool


def _hippocampus() -> Hippocampus:
    return Hippocampus(HippocampusConfig(auto_save_after_sleep=False))


# -- ExamineTool: wired (owner decision 2026-10-06) ---------------------------


def test_examine_recalls_what_the_matching_memory_holds():
    h = _hippocampus()
    assert h.store_observation("The brass key lies under the loose floorboard.")
    out = ExamineTool(hippocampus=h).execute(target="brass key")
    assert out.success
    assert "You recall: The brass key lies under the loose floorboard." in out.output["observation"]


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


def _bridge(*scene: str):
    """A sim bridge whose transcript window holds ``scene`` (the percepts Stage 1 reads)."""
    return SimpleNamespace(percept_source=SimpleNamespace(_transcript_percepts=[{"cli_input": t} for t in scene]))


def test_examine_does_not_recall_the_current_scene_as_memory():
    """A sim captures each percept into the hippocampus: the newest match is the present, not memory."""
    scene = "The brass key lies under the loose floorboard."
    h = _hippocampus()
    mid = h.store_observation(scene)
    out = ExamineTool(bridge=_bridge(scene), hippocampus=h).execute(target="brass key")
    assert "You recall" not in out.output["observation"]
    [record] = h.recall_by_ids([mid])
    assert record.activation_sources.get("tool") is None


def test_the_current_scenes_capture_is_not_recalled_through_its_other_text():
    """The capture of a percept in the window is the present even where only its metadata names the target."""
    scene = "A wooden table stands by the window."
    h = _hippocampus()
    mid = h.store_observation(scene, {"note": "The brass key sits in its drawer."})
    out = ExamineTool(bridge=_bridge(scene), hippocampus=h).execute(target="brass key")
    assert "You recall" not in out.output["observation"]
    [record] = h.recall_by_ids([mid])
    assert record.activation_sources.get("tool") is None


def test_examine_recalls_the_past_beside_a_matching_scene():
    h = _hippocampus()
    past = h.store_observation("Yesterday the brass key opened the cellar door.")
    scene = "A brass key glints on the table."
    h.store_observation(scene)  # this scene's own capture: not recalled
    out = ExamineTool(bridge=_bridge(scene), hippocampus=h).execute(target="brass key")
    text = out.output["observation"]
    assert "A brass key glints on the table." in text
    assert "You recall: Yesterday the brass key opened the cellar door." in text
    assert text.count("You recall") == 1
    [record] = h.recall_by_ids([past])
    assert record.activation_sources.get("tool") == 1


def test_only_memories_shown_under_the_five_line_cap_count_as_used():
    scene = "A brass key here. Another brass key there. A third brass key below. A fourth brass key above."
    h = _hippocampus()
    ids = [h.store_observation(f"Long ago brass key number {n} was lost.") for n in range(3)]
    out = ExamineTool(bridge=_bridge(scene), hippocampus=h).execute(target="brass key")
    shown = out.output["observation"]
    used = [i for i in ids if h.recall_by_ids([i])[0].activation_sources.get("tool")]
    assert shown.count("You recall") == 1  # 4 scene sentences + 1 recalled = the 5-line cap
    assert len(used) == 1


def test_a_past_memory_does_not_repeat_a_sentence_the_scene_shows():
    """A different, older memory whose matching sentence is already on screen adds nothing."""
    h = _hippocampus()
    h.store_observation("The brass key glints. The door is shut.")
    out = ExamineTool(bridge=_bridge("A note lies here. The brass key glints"), hippocampus=h).execute(
        target="brass key"
    )
    assert "You recall" not in out.output["observation"]


def test_at_most_three_past_memories_are_shown_and_counted():
    h = _hippocampus()
    ids = [h.store_observation(f"Long ago brass key number {n} was lost.") for n in range(5)]
    out = ExamineTool(hippocampus=h).execute(target="brass key")
    assert out.output["observation"].count("You recall") == 3
    used = [i for i in ids if h.recall_by_ids([i])[0].activation_sources.get("tool")]
    assert len(used) == 3


def test_two_memories_with_the_same_sentence_in_different_case_show_once():
    h = _hippocampus()
    assert h.store_observation("The brass key is old. It was in the hall.")
    assert h.store_observation("the brass key is old. Someone left it.")  # not deduped by the store
    out = ExamineTool(hippocampus=h).execute(target="brass key")
    assert out.output["observation"].lower().count("you recall") == 1


def test_a_long_sentence_is_cut_around_the_match():
    h = _hippocampus()
    h.store_observation("x" * 300 + " the brass key " + "y" * 300 + ".")
    [m] = h.search_by_content("brass key")
    text = Hippocampus.matching_sentence(m, "brass key")
    assert "brass key" in text and len(text) <= 201


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


def test_recall_deep_text_fallback_finds_a_matching_memory():
    from types import SimpleNamespace

    from maxim.agents.exec_agent import ExecAgent

    h = _hippocampus()
    mid = h.store_observation("The brass key lies under the loose floorboard.")
    agent = SimpleNamespace(_hippocampus=h)
    assert [m.id for m in ExecAgent.recall_deep(agent, "brass key")] == [mid]


def test_a_cross_layer_angular_gyrus_hit_is_found_by_its_id():
    import time

    from maxim.agents.memory_agent import MemoryAgent
    from maxim.math.angular_gyrus import AngularGyrus
    from maxim.math.math_types import MathMemory

    ag = AngularGyrus()
    ag.store(MathMemory(id="m-sum", timestamp=time.time(), name="sum_rule", confidence=0.3))

    class _Hub:
        atl = None

        def __init__(self) -> None:
            self.angular_gyrus = ag

        def recall_with_knowledge(self, **_kwargs):
            return {"angular_gyrus": [("m-sum", 0.9)]}

    agent = SimpleNamespace(
        _memory_hub=_Hub(),
        _forming_pool={"x": SimpleNamespace(_hippocampus_id="ep-1")},
        _get_concept_relationships=MemoryAgent._get_concept_relationships,
    )
    entries = MemoryAgent._build_knowledge_context(agent)
    assert [e["concept_name"] for e in entries] == ["sum_rule"]
