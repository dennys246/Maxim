"""#993/#994/#995/#1129: a memory's tool and goal are read the same way for every record kind.

An ``EpisodicMemory`` keeps its tool on ``action`` and its goal on ``decision.intent`` / ``context``;
a ``CompressedMemory`` carries ``tool_name`` and ``goal`` directly. Readers probed one shape with
``getattr(..., default)`` (sometimes a field neither kind has: ``Context.goal``, ``Action.tool_used``,
``Outcome.valence``) and silently got the default. Every test here uses real records.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from maxim.memory.types import (
    Action,
    CompressedMemory,
    Context,
    Decision,
    EpisodicMemory,
    Outcome,
    Perception,
)


def _episode(*, intent_goal: str | None = "fetch water", active_goal: str | None = "stay warm") -> EpisodicMemory:
    return EpisodicMemory(
        id="e1",
        timestamp=1.0,
        perception=Perception(cli_input="a well stands in the yard", salience=0.7),
        context=Context(active_goal=active_goal),
        decision=Decision(intent={"goal": intent_goal} if intent_goal else {}),
        action=Action(tool_name="draw_water"),
        outcome=Outcome(success=True),
    )


# -- #995: one answer per fact on the type ------------------------------------------------------


@pytest.mark.xfail(strict=True, reason="#995: EpisodicMemory has no tool_name / goal")
@pytest.mark.parametrize(
    ("intent_goal", "active_goal", "expected"),
    [("fetch water", "stay warm", "fetch water"), (None, "stay warm", "stay warm"), (None, None, None)],
)
def test_an_episode_answers_tool_and_goal_as_its_compressed_form_does(intent_goal, active_goal, expected):
    ep = _episode(intent_goal=intent_goal, active_goal=active_goal)
    compressed = CompressedMemory.from_episodic(ep)
    assert (ep.tool_name, ep.goal) == ("draw_water", expected)
    assert (ep.tool_name, ep.goal) == (compressed.tool_name, compressed.goal)


@pytest.mark.xfail(strict=True, reason="#995: a compressed record hashes ':<outcome>' with no tool")
def test_a_compressed_records_signature_keeps_its_tool():
    from maxim.similarity.signature import SituationSignature

    ep = _episode()
    assert SituationSignature.from_memory(CompressedMemory.from_episodic(ep)).tool_name == "draw_water"


@pytest.mark.xfail(strict=True, reason="#995: a compressed context item carries no goal or tool")
def test_a_compressed_records_context_item_keeps_goal_and_tool():
    from maxim.agents.memory_agent import MemoryAgent

    item = MemoryAgent._memory_to_context_item(CompressedMemory.from_episodic(_episode()), 0.5)
    assert (item["content"].get("goal"), item["content"].get("action")) == ("fetch water", "draw_water")


# -- #993: summaries read real fields ------------------------------------------------------------


@pytest.mark.xfail(strict=True, reason="#993: Observer reads Context.goal and Action.tool_used, which do not exist")
def test_the_observers_memory_list_shows_goal_and_tool_for_both_kinds():
    from maxim.simulation.introspection import Observer

    ep = _episode()

    class _H:
        def recall(self, **_kw):
            return [ep, CompressedMemory.from_episodic(ep)]

        def __len__(self):
            return 2

    out = Observer(hippocampus=_H()).memory_recall()
    assert [(m["goal"], m["tool"]) for m in out["memories"]] == [("fetch water", "draw_water")] * 2


@pytest.mark.xfail(strict=True, reason="#993: the summary's valence reads Outcome.valence, which does not exist")
def test_an_agent_export_labels_each_memorys_outcome():
    from maxim.runtime.agent_factory import AgentInstance

    ep = _episode()
    failed = EpisodicMemory(id="e2", timestamp=2.0, outcome=Outcome(success=False))
    unknown = CompressedMemory(id="c3", timestamp=3.0, tool_name="wait", success=None)
    agent = SimpleNamespace(agent_id="a", role="aut", hippocampus=[ep, failed, unknown], nac=None)
    out = AgentInstance.export_memories(agent)["memory_summaries"]
    assert [(s["tool"], s["outcome"]) for s in out] == [
        ("draw_water", "success"),
        ("", "failure"),
        ("wait", "unknown"),
    ]


# -- #1129: the public recall source reads the real record ---------------------------------------


@pytest.mark.xfail(strict=True, reason="#1129: _summarize reads cli_input/transcript on the record, not perception")
def test_public_recall_returns_an_episode_from_a_real_record():
    from maxim.integration.recall import EpisodicRecallSource

    ep = _episode()
    compressed = CompressedMemory.from_episodic(_episode(intent_goal="find the cellar"))
    records = {"e1": ep, "c1": compressed}
    episodes = [
        SimpleNamespace(imagined=False, valence=0.0, activated_nodes=("e1",)),
        SimpleNamespace(imagined=False, valence=0.0, activated_nodes=("c1",)),
    ]
    hippo = SimpleNamespace(get=records.get, _episode_store=SimpleNamespace(all_episodes=lambda: episodes))
    items = EpisodicRecallSource(hippo).recalled_items(limit=8)
    assert sorted(i.text for i in items) == ["a well stands in the yard", "find the cellar"]
    assert max(i.salience for i in items) == pytest.approx(0.7)
