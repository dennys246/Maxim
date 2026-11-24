"""#1710 (#1137's flake): sleep compresses a record of what a person said before it can remove it.

The red gate ``test_compressed_recall_1137.py`` flaked on main and on PRs (2026-11-19..24) with an empty store:

- the talk loop could tick once before the person's words arrived, and that idle percept made the words read as
  novelty 0 (``PerceptionAgent._compute_novelty`` compares vision tracks; two percepts with none look the same);
- so they were not promoted to long-term at birth, and an aged, unpromoted record scored below
  ``retention_threshold`` and was deleted outright instead of compressed.

The fix (owner decision 2026-11-24) is compress-first, on both sleep paths: such a record is compressed, which keeps
the words (``CompressedMemory.heard``), and the compressed record then follows ordinary retention: it is removed at a
later sleep unless something used it in between. It is not a retention floor (#843). Text novelty itself is a
separate design pass, #1790 (it would change promotion at birth for every utterance).
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


def _capture(hippo: Any, *, cli_input: str | None, transcript: str | None = None) -> str:
    """A low-salience, non-novel record (never promoted at birth), with or without words a person typed or said."""
    from maxim.memory.types import Action, Context, Decision, Outcome, Perception

    return hippo.capture(
        perception=Perception(cli_input=cli_input, transcript=transcript, salience=0.3, novelty=0.3),
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


def test_spoken_words_are_compressed_too() -> None:
    from maxim.memory.types import CompressedMemory

    hippo = _hippo()
    said = _capture(hippo, cli_input=None, transcript="my favourite colour is green")
    _age(hippo, 10 * _DAYS)
    hippo.sleep()
    assert isinstance(hippo._memories.get(said), CompressedMemory), "a voice transcript's record was removed"


def test_the_compressed_record_gets_one_more_sleep_not_a_floor() -> None:
    """Compression does not raise the retention score: unused, the compressed record goes at the NEXT sleep (#843:
    no floor), even with no further ageing. This pins the real rule, one more stage."""
    hippo = _hippo()
    said = _capture(hippo, cli_input="my favourite colour is green")
    _age(hippo, 10 * _DAYS)
    hippo.sleep()
    assert said in hippo._memories
    hippo.sleep()
    assert said not in hippo._memories, "a compressed user-words record outlived an unused sleep: that is a floor"


def test_the_compressed_record_is_kept_while_recall_uses_it() -> None:
    """Recall finds the compressed record by what it heard and touches it (bio-memory: ``recall()`` always touches),
    which resets its recency, so the next sleep keeps it."""
    from maxim.memory.types import CompressedMemory

    hippo = _hippo()
    said = _capture(hippo, cli_input="my favourite colour is green")
    _age(hippo, 10 * _DAYS)
    hippo.sleep()
    found = hippo.recall(query="what colour is green")
    assert said in [m.id for m in found], "recall did not reach the compressed record through its words"
    hippo.sleep()
    assert isinstance(hippo._memories.get(said), CompressedMemory), "a used compressed record was removed"
