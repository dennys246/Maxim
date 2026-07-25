"""Curated recall abstraction — the three rules + the episodic join.

Pins that ``curate_recall`` applies provenance-filter + salience-rank + under-claim
uniformly across ANY ``RecallSource``, and that ``EpisodicRecallSource`` honors
provenance (drops imagined) and joins episodes to readable record text.
"""

from __future__ import annotations

from types import SimpleNamespace

from maxim.integration.recall import (
    CuratedRecall,
    EpisodicRecallSource,
    RecalledItem,
    curate_recall,
)


class _StaticSource:
    def __init__(self, items):
        self._items = items

    def recalled_items(self, *, limit):
        return list(self._items)


class _BoomSource:
    def recalled_items(self, *, limit):
        raise RuntimeError("source exploded")


def test_curate_applies_the_three_rules():
    src = _StaticSource(
        [
            RecalledItem(text="real, high", kind="story", salience=0.9, real=True),
            RecalledItem(text="real, low", kind="story", salience=0.1, real=True),
            RecalledItem(text="IMAGINED", kind="story", salience=1.0, real=False),  # dropped
            RecalledItem(text="", kind="story", salience=1.0, real=True),  # no text → dropped
            RecalledItem(text="a trait", kind="trait", salience=0.5, real=True),
        ]
    )
    out = curate_recall([src], per_kind_limit=8)
    assert isinstance(out, CuratedRecall)
    # provenance: imagined excluded; under-claim: empty-text excluded
    summaries = [m.text for m in out.story_memories]
    assert "IMAGINED" not in summaries and "" not in summaries
    # rank by salience, not order
    assert summaries == ["real, high", "real, low"]
    # kind routing
    assert out.player_model == ["a trait"]


def test_a_broken_source_does_not_sink_recall():
    good = _StaticSource([RecalledItem(text="survived", kind="story", salience=0.5)])
    out = curate_recall([_BoomSource(), good])
    assert [m.text for m in out.story_memories] == ["survived"]


def test_per_kind_cap():
    items = [RecalledItem(text=f"m{i}", kind="story", salience=i / 10) for i in range(20)]
    out = curate_recall([_StaticSource(items)], per_kind_limit=3)
    assert len(out.story_memories) == 3
    # highest-salience three
    assert [m.text for m in out.story_memories] == ["m19", "m18", "m17"]


def test_episodic_source_filters_imagined_and_joins_record_text():
    """The episodic join: non-imagined episode → readable record text + salience;
    imagined episode is never yielded (provenance at the source)."""
    records = {
        "m_real": SimpleNamespace(
            cli_input="your rogue betrayed the party", salience=0.9, transcript=None, context=None
        ),
        "m_dull": SimpleNamespace(cli_input="you entered a room", salience=0.2, transcript=None, context=None),
        "m_fic": SimpleNamespace(cli_input="the dragon spoke", salience=0.95, transcript=None, context=None),
    }
    episodes = [
        SimpleNamespace(imagined=False, valence=-0.9, activated_nodes=("m_real",)),
        SimpleNamespace(imagined=False, valence=0.1, activated_nodes=("m_dull",)),
        SimpleNamespace(imagined=True, valence=-0.95, activated_nodes=("m_fic",)),  # in-fiction → hidden
        SimpleNamespace(imagined=False, valence=0.0, activated_nodes=("m_missing",)),  # unresolvable → skipped
    ]
    hippo = SimpleNamespace(
        _episode_store=SimpleNamespace(all_episodes=lambda: episodes),
        get=lambda node: records.get(node),
    )
    out = curate_recall([EpisodicRecallSource(hippo)], per_kind_limit=8)
    texts = [m.text for m in out.story_memories]
    assert "your rogue betrayed the party" in texts
    assert "you entered a room" in texts
    assert "the dragon spoke" not in texts  # imagined dropped
    assert len(texts) == 2  # the unresolvable one is skipped (under-claim)
    # the betrayal (|valence| 0.9 / salience 0.9) outranks the room
    assert texts[0] == "your rogue betrayed the party"


def test_recall_verb_empty_when_no_state(tmp_path):
    import maxim

    r = maxim.recall(home_dir=str(tmp_path))
    assert isinstance(r.story_memories, list) and r.story_memories == []
    assert r.name is None and r.player_model == []
