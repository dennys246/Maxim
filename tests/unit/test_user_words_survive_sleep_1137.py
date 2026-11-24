"""#1137's flake, both halves of its fix: user words are compressed before any sleep removes them, and a text
percept's novelty is judged by its words.

The red gate ``test_compressed_recall_1137.py`` flaked on main (2026-11-23 at 266aa5c7, 2026-11-24 at f145e27b):

- the talk loop could tick once before the person's words arrived, and that idle percept made the words read as
  novelty 0 (``PerceptionAgent._compute_novelty`` compared vision tracks, and two percepts with none look the same);
- so they were not promoted to long-term at birth, and an aged, unpromoted record scored below
  ``retention_threshold`` and was deleted outright instead of compressed.

These tests pin each half on its own: a user-words record that sleep would remove is compressed (on both sleep
paths), and only the compressed record faces removal later; text novelty ignores idle percepts and recognizes a
repeated sentence.
"""

from __future__ import annotations

from typing import Any

import pytest

from maxim.memory.encoding import EncodingSignals

_DAYS = 24 * 3600


def _hippo() -> Any:
    from maxim.memory.hippocampus import Hippocampus, HippocampusConfig

    return Hippocampus(
        HippocampusConfig(persistence_path=None, enable_sleep_consolidation=True, auto_save_after_sleep=False)
    )


def _capture(hippo: Any, *, cli_input: str | None) -> str:
    """A low-salience, non-novel record (never promoted at birth), with or without words a person typed."""
    from maxim.memory.types import Action, Context, Decision, Outcome, Perception

    return hippo.capture(
        perception=Perception(cli_input=cli_input, salience=0.3, novelty=0.3),
        context=Context(),
        decision=Decision(),
        action=Action(tool_name="respond" if cli_input is None else ""),
        outcome=Outcome(success=None),
        encoding=EncodingSignals.unmeasured("api"),
    )


def _age(hippo: Any, seconds: float) -> None:
    for record in hippo._memories.values():
        record.created_at -= seconds
        record.timestamp -= seconds
        record.accessed_at -= seconds


def _sleep(hippo: Any, path: str) -> None:
    if path == "clustering":
        from maxim.time.scn import SCN

        hippo.connect_scn(SCN())
        hippo.sleep_with_clustering()
    else:
        hippo.sleep()


@pytest.mark.parametrize("path", ["sleep", "clustering"])
def test_an_aged_user_words_record_is_compressed_not_removed(path: str) -> None:
    from maxim.memory.types import CompressedMemory

    hippo = _hippo()
    said = _capture(hippo, cli_input="my favourite colour is green")
    other = _capture(hippo, cli_input=None)
    assert not hippo._memories[said].long_term, "the record was promoted at birth: the test would not exercise sleep"
    _age(hippo, 10 * _DAYS)

    _sleep(hippo, path)

    kept = hippo._memories.get(said)
    assert isinstance(kept, CompressedMemory), f"user words were not compressed by {path}: {kept!r}"
    assert "green" in (kept.heard or ""), "the compressed record lost what the person said"
    assert other not in hippo._memories, "a record with no user words was kept: the retention score did not apply"


def test_a_compressed_user_words_record_still_faces_removal() -> None:
    """Compress-first is not a retention floor (#843): the compressed record is removed when its score falls."""
    hippo = _hippo()
    said = _capture(hippo, cli_input="my favourite colour is green")
    _age(hippo, 10 * _DAYS)
    hippo.sleep()
    assert said in hippo._memories
    _age(hippo, 30 * _DAYS)
    hippo.sleep()
    assert said not in hippo._memories, "a compressed user-words record was kept forever"


# ── text novelty ──────────────────────────────────────────────────────────


def _perception() -> Any:
    from maxim.agents.bus import AgentBus
    from maxim.agents.perception_agent import PerceptionAgent

    return PerceptionAgent(AgentBus())


def test_an_idle_tick_does_not_make_the_next_words_familiar() -> None:
    agent = _perception()
    agent.process_observation({})  # the loop ticked before the person spoke
    percept = agent.process_observation({"cli_input": "My name is Denny and my favourite colour is green."})
    assert percept.novelty == 1.0


def test_a_repeated_sentence_reads_as_familiar_among_others() -> None:
    agent = _perception()
    agent.process_observation({"cli_input": "The kettle is on the stove."})
    agent.process_observation({"cli_input": "What time does the shop open?"})
    repeat = agent.process_observation({"cli_input": "The kettle is on the stove."})
    fresh = agent.process_observation({"cli_input": "Penguins cannot fly."})
    assert repeat.novelty == 0.0
    assert fresh.novelty == 1.0


def test_vision_novelty_is_unchanged_for_percepts_without_text() -> None:
    agent = _perception()
    agent.process_observation({})
    assert agent.process_observation({}).novelty == 0.0, "two idle ticks look the same on the track path"
