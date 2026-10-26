"""#1137's red gate: a fact the user stated still reaches the prompt after its memory is COMPRESSED.

#1465 made "a fact said in session N is answerable in session N+1" true, but only until compression. A
``CompressedMemory`` kept ``had_user_input`` as a bool and nothing it heard, and ``goal`` intent-first only, so
once a record aged past its compression threshold the words were gone. Sleep consolidation compresses a record
holding user words after 3 x ``compression_age``, about 3 days under ``access_based``. The owner decided
(2026-10-25) that #1137 ships in 1.3.4, as a release condition for the recall claim.

Same shipped composition as ``test_cross_session_recall_1444.py``: a real ``MaximHandle``, its talk loop, the real
``LLMWorker`` and ``PromptBuilder``, and a recording fake model. Session N states the fact. Its records are aged past
the compression threshold, the only test-side manipulation, standing in for days of wall time. ``stop()`` runs the
full consolidation, which compresses them, and saves. A new handle on the same home asks. The fact's record must be a
``CompressedMemory`` on reload (so the gate cannot pass on an uncompressed store), and "green" must reach the prompt
inside the untrusted-content frame.

Landed as a strict xfail (``c79ef22b``) and flipped in place by #1137's shape: ``CompressedMemory`` keeps the
episode's active goal and, bounded, what it heard; ``input_text`` answers it, so recall renders it framed.
"""

from __future__ import annotations

import json
import time
from typing import Any

import pytest

from tests.unit.test_cross_session_recall_1444 import (
    _FACT,
    _QUESTION,
    _RecordingModel,
    _ask,
    _framed_bodies,
    _handle,
)

_DAYS = 5 * 24 * 3600  # past 3 x compression_age (1 day): user-input records compress at 3 days (access_based)


def _age_every_record(handle: Any, seconds: float) -> None:
    """Days of wall time, compressed: every record of session N was made ``seconds`` ago."""
    hippocampus = handle.instance.hippocampus
    if hasattr(hippocampus, "flush"):
        hippocampus.flush()
    for record in list(hippocampus._memories.values()):
        record.created_at -= seconds
        record.timestamp -= seconds
        record.accessed_at -= seconds


@pytest.fixture(scope="module")
def compressed_session(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    from maxim.simulation.sim_logger import disable_sim_logging

    home = tmp_path_factory.mktemp("compressed_gate") / "home"
    model = _RecordingModel()
    with pytest.MonkeyPatch.context() as mp:
        import maxim.runtime.lane_backends as lane_backends
        from maxim.simulation.bridge import SimulationBridge

        mp.setattr(lane_backends, "build_primary_router", lambda *a, **k: (model, None))
        mp.setenv("MAXIM_SKIP_REMOTE_PROBE", "1")
        real_init = SimulationBridge.__init__

        def _quick_settle(self: Any, *a: Any, **k: Any) -> None:  # timing only: Talk settles 2 s after each reply
            real_init(self, *a, **{**k, "settle_s": 0.2})

        mp.setattr(SimulationBridge, "__init__", _quick_settle)

        session_n = _handle(home)
        try:
            _ask(session_n, model, _FACT)
            time.sleep(0.5)
            _age_every_record(session_n, _DAYS)
        finally:
            session_n.stop()

        stored = json.loads((home / "hippocampus.json").read_text())
        session_n1 = _handle(home)
        try:
            prompts = _ask(session_n1, model, _QUESTION)
        finally:
            session_n1.stop()
            disable_sim_logging()
    return {"stored": stored, "prompts": prompts}


def test_the_facts_memory_was_compressed(compressed_session):
    """Non-vacuity: the session-end consolidation did compress the record holding the user's words."""
    assert any(m.get("_compressed") and m.get("had_user_input") for m in compressed_session["stored"]["memories"]), [
        (m.get("_compressed"), m.get("had_user_input")) for m in compressed_session["stored"]["memories"]
    ]


def test_a_compressed_fact_still_reaches_the_prompt_framed(compressed_session):
    prompts = compressed_session["prompts"]
    assert any("green" in body.lower() for p in prompts for body in _framed_bodies(p)), prompts[-1]


# ── review folds (Executor + Architecture + Wire, 2026-10-25) ───────────────────────────────────────────────────


def _episode(*, intent_goal: str = "fetch water", active_goal: str | None = "stay warm") -> Any:
    from maxim.memory.types import Action, Context, Decision, EpisodicMemory, Outcome, Perception

    return EpisodicMemory(
        id="e1",
        timestamp=1.0,
        perception=Perception(cli_input="the secret word is zanzibar", salience=0.7),
        context=Context(active_goal=active_goal),
        decision=Decision(intent={"goal": intent_goal}),
        action=Action(tool_name="draw_water"),
        outcome=Outcome(success=True),
    )


def test_a_1_2_round_trip_keeps_the_words_and_the_active_goal() -> None:
    from maxim.memory.types import CompressedMemory

    back = CompressedMemory.from_dict(json.loads(json.dumps(CompressedMemory.from_episodic(_episode()).to_dict())))
    assert (back.heard, back.active_goal, back.goal) == ("the secret word is zanzibar", "stay warm", "fetch water")


def test_an_episode_with_no_active_goal_keeps_none_through_compression() -> None:
    """ "" is "kept: none"; only a pre-1.2 record (None) falls back to the intent goal (review SF-2/S3)."""
    from maxim.memory.types import CompressedMemory, record_active_goal

    ep = _episode(intent_goal="learn_from_failure", active_goal=None)
    compressed = CompressedMemory.from_episodic(ep)
    assert record_active_goal(ep) is None and record_active_goal(compressed) is None
    assert record_active_goal(CompressedMemory(id="old", timestamp=1.0, goal="pain_response")) == "pain_response"


def test_compression_keeps_what_the_render_shows_bounded_after_normalization() -> None:
    from maxim.integration.bio_enrichment import BioEnrichmentPipeline
    from maxim.memory.text import RECALLED_WORDS_MAX_CHARS, bounded_words
    from maxim.memory.types import CompressedMemory

    long_words = "ﷺ" * 40 + " the colour is green"  # NFKC expands each ~18x: bounded AFTER normalization
    compressed = CompressedMemory.from_episodic(_episode().__class__(id="e2", timestamp=1.0))
    compressed.heard = bounded_words(long_words)
    assert len(compressed.heard) <= RECALLED_WORDS_MAX_CHARS and bounded_words(compressed.heard) == compressed.heard
    import unicodedata

    rendered = BioEnrichmentPipeline()._summarize_episode(compressed)
    assert unicodedata.normalize("NFKC", compressed.heard) in rendered  # the frame NFKCs its body: "…" shows as "..."


def test_substring_search_and_matching_sentence_find_a_compressed_records_words() -> None:
    from maxim.memory.hippocampus import Hippocampus
    from maxim.memory.types import CompressedMemory

    compressed = CompressedMemory.from_episodic(_episode())
    assert Hippocampus._memory_matches_query(compressed, "zanzibar")
    assert Hippocampus.matching_sentence(compressed, "zanzibar") == "the secret word is zanzibar."


def test_a_compressed_records_summary_and_association_weight_read_its_active_goal() -> None:
    from maxim.memory.hippocampus import Hippocampus, HippocampusConfig
    from maxim.memory.hippocampus_retrieval import _memory_summary
    from maxim.memory.types import CompressedMemory

    compressed = CompressedMemory.from_episodic(_episode())
    assert "stay warm" in _memory_summary(compressed) and "fetch water" not in _memory_summary(compressed)
    h = Hippocampus(HippocampusConfig(auto_save_after_sleep=False))
    same, other = _episode(intent_goal="x", active_goal="stay warm"), _episode(intent_goal="x", active_goal="dig")
    assert h._compute_association_weight(same, compressed, 0.5) > h._compute_association_weight(other, compressed, 0.5)


def test_a_compressed_records_keys_are_cached_and_follow_its_words() -> None:
    from maxim.memory.types import CompressedMemory

    compressed = CompressedMemory.from_episodic(_episode())
    first = compressed.content_tokens()
    assert compressed.content_tokens() is first and "zanzibar" in first
    compressed.heard = "a silver spoon"
    assert "silver" in compressed.content_tokens() and "zanzibar" not in compressed.content_tokens()
    assert "zanzibar" not in repr(compressed)  # raw stored words stay out of the repr
