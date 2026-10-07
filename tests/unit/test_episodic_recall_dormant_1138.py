"""#1138: ``EpisodicRecallSource`` returns no story memory on the production path, so it is Dormant.

An episode's ``activated_nodes`` are ATL substrate node ids (``MemoryHub.on_percept_received`` ->
``bio_integration.record_substrate_nodes``), empty unless the substrate path is on. The source joined them
against the Hippocampus, where no such id lives. These tests pin that, on a real Hippocampus and real
episodes, so a revival has to change what they assert (owner decision 2026-10-06). The positive control
shows the source itself works when an episode does carry a Hippocampus id: the defect is the link, not the
reader (redesign: #1144).
"""

from __future__ import annotations

from maxim.integration.recall import EpisodicRecallSource
from maxim.memory.episode import CaptureEvent
from maxim.memory.hippocampus import Hippocampus, HippocampusConfig

TEXT = "Your rogue betrayed the party at the bridge."


def _hippocampus() -> tuple[Hippocampus, str]:
    h = Hippocampus(HippocampusConfig(auto_save_after_sleep=False))
    memory_id = h.store_observation(TEXT)
    assert memory_id
    return h, memory_id


def _close(h: Hippocampus) -> None:
    assert h.finalize_pending_episode() is not None
    assert len(h._episode_store.all_episodes()) == 1  # the episode is there; only the join can fail


def test_the_production_producer_links_no_record_with_the_substrate_path_off():
    """The agent loop's call shape: ``observe_episode(activated_nodes=())`` with an empty substrate stash."""
    from maxim.runtime import bio_integration

    h, _ = _hippocampus()
    bio_integration.observe_episode(hippocampus=h, agent_id="aut-1138", channel="text", activated_nodes=())
    _close(h)
    [episode] = h._episode_store.all_episodes()
    assert episode.activated_nodes == ()
    assert EpisodicRecallSource(h).recalled_items(limit=8) == []


def test_a_substrate_node_id_does_not_resolve_in_the_hippocampus():
    """With the substrate path on, the episode carries ATL substrate ids, which no Hippocampus record has."""
    h, _ = _hippocampus()
    h.observe_episode_event(CaptureEvent(tick=1, channel="text", activated_nodes=("substrate-text-7f3a",)))
    _close(h)
    assert h.get("substrate-text-7f3a") is None
    assert EpisodicRecallSource(h).recalled_items(limit=8) == []


def test_positive_control_an_episode_carrying_a_hippocampus_id_is_recalled():
    """Not the production path: it shows the source works when the link exists, so the pins above are
    specific to the id mismatch rather than to a source that is always empty."""
    from maxim.memory.encoding import EncodingSignals
    from maxim.memory.types import Perception

    h = Hippocampus(HippocampusConfig(auto_save_after_sleep=False))
    # A loop-style record (``capture_from_loop`` sets ``perception.cli_input``); ``store_observation`` keeps
    # its text only in ``observations``, which this reader does not read (noted on #1144).
    memory_id = h.capture(
        perception=Perception(cli_input=TEXT, salience=0.6), encoding=EncodingSignals.unmeasured("api")
    )
    assert h.get(memory_id) is not None
    h.observe_episode_event(CaptureEvent(tick=1, channel="text", activated_nodes=(memory_id,)))
    _close(h)
    assert [i.text for i in EpisodicRecallSource(h).recalled_items(limit=8)] == [TEXT]
