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


@pytest.mark.parametrize(
    ("intent_goal", "active_goal", "expected"),
    [("fetch water", "stay warm", "fetch water"), (None, "stay warm", "stay warm"), (None, None, None)],
)
def test_an_episode_answers_tool_and_goal_as_its_compressed_form_does(intent_goal, active_goal, expected):
    ep = _episode(intent_goal=intent_goal, active_goal=active_goal)
    compressed = CompressedMemory.from_episodic(ep)
    assert (ep.tool_name, ep.goal) == ("draw_water", expected)
    assert (ep.tool_name, ep.goal) == (compressed.tool_name, compressed.goal)


def test_a_compressed_records_signature_keeps_its_tool():
    from maxim.similarity.signature import SituationSignature

    ep = _episode()
    assert SituationSignature.from_memory(CompressedMemory.from_episodic(ep)).tool_name == "draw_water"


def test_a_compressed_records_context_item_keeps_goal_and_tool():
    from maxim.agents.memory_agent import MemoryAgent

    item = MemoryAgent._memory_to_context_item(CompressedMemory.from_episodic(_episode()), 0.5)
    assert (item["content"].get("goal"), item["content"].get("action")) == ("fetch water", "draw_water")


# -- #993: summaries read real fields ------------------------------------------------------------


def test_the_observers_memory_list_shows_goal_and_tool_for_both_kinds():
    from maxim.simulation.introspection import Observer

    ep = _episode()

    class _H:
        def recall(self, **_kw):
            return [ep, CompressedMemory.from_episodic(ep)]

        def __len__(self):
            return 2

    out = Observer(hippocampus=_H()).memory_recall()
    # An episode shows its ACTIVE goal; the compressed record only kept the intent goal (#1137).
    assert [(m["goal"], m["tool"]) for m in out["memories"]] == [
        ("stay warm", "draw_water"),
        ("fetch water", "draw_water"),
    ]


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


def test_public_recall_reads_a_real_records_fields_when_an_id_resolves():
    """Hand-built episodes carrying Hippocampus ids: production episodes carry none, so the source is
    Dormant (#1138, redesign #1144). This pins only that the reader uses the real fields."""
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


def test_no_reader_probes_a_field_no_record_has():
    """``Action.tool_used`` and ``Context.goal`` never existed; ``Outcome.valence`` neither. A probe of one
    always got its default (#993, #1129). Scan ``src/`` and ``scripts/`` for the probes. Limits: a blocklist
    of these three names in their one-line forms; a two-step probe (``ctx = m.context; getattr(ctx, "goal")``)
    or a new misspelling passes. Typed parameters checked by mypy would be the structural guard."""
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    bad = re.compile(
        r'"tool_used"|getattr\(\s*getattr\([^)]*"context"[^)]*\)\s*,\s*"goal"|\.context\.goal\b|getattr\(\s*getattr\([^)]*"outcome"[^)]*\)\s*,\s*"valence"'
    )
    hits = [
        f"{path.relative_to(root)}:{n}"
        for top in ("src/maxim", "scripts")
        for path in (root / top).rglob("*.py")
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if bad.search(line) and 'exp.type == "tool_used"' not in line  # an expectation type, not a field
    ]
    assert hits == []


# -- each migrated reader, pinned for both kinds (executor review) --------------------------------


def test_an_episodes_context_item_keeps_its_active_goal():
    from maxim.agents.memory_agent import MemoryAgent

    item = MemoryAgent._memory_to_context_item(_episode(), 0.5)
    assert (item["content"]["goal"], item["content"]["action"]) == ("stay warm", "draw_water")


def test_an_enrichment_summary_names_tool_and_intent_goal_for_both_kinds():
    from maxim.integration.bio_enrichment import BioEnrichmentPipeline

    for record in (_episode(), CompressedMemory.from_episodic(_episode())):
        text = BioEnrichmentPipeline._summarize_episode(record)
        assert "draw_water" in text and "(goal: fetch water)" in text, (type(record).__name__, text)


def test_public_recall_ranks_each_item_by_its_own_records_salience():
    from maxim.integration.recall import EpisodicRecallSource

    compressed = CompressedMemory(id="c1", timestamp=1.0, goal="find the cellar", salience=0.2)
    records = {"e1": _episode(), "c1": compressed}
    episodes = [SimpleNamespace(imagined=False, valence=0.0, activated_nodes=(n,)) for n in ("e1", "c1")]
    hippo = SimpleNamespace(get=records.get, _episode_store=SimpleNamespace(all_episodes=lambda: episodes))
    by_text = {i.text: i.salience for i in EpisodicRecallSource(hippo).recalled_items(limit=8)}
    assert by_text == {"a well stands in the yard": pytest.approx(0.7), "find the cellar": pytest.approx(0.2)}


def test_public_recall_falls_back_to_an_episodes_active_goal():
    from maxim.integration.recall import EpisodicRecallSource

    quiet = _episode()
    quiet.perception = Perception(salience=0.4)  # no text
    hippo = SimpleNamespace(
        get={"e1": quiet}.get,
        _episode_store=SimpleNamespace(
            all_episodes=lambda: [SimpleNamespace(imagined=False, valence=0.0, activated_nodes=("e1",))]
        ),
    )
    assert [i.text for i in EpisodicRecallSource(hippo).recalled_items(limit=8)] == ["stay warm"]


def test_the_memory_tool_formats_tool_and_goal_for_both_kinds():
    from maxim.tools.introspection import _format_episodic_memory

    assert [
        (_format_episodic_memory(r)["tool"], _format_episodic_memory(r)["goal"])
        for r in (_episode(), CompressedMemory.from_episodic(_episode()))
    ] == [("draw_water", "stay warm"), ("draw_water", "fetch water")]
