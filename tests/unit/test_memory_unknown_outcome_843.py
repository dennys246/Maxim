"""#843 -- a memory that is not an action outcome records its success as UNKNOWN (``None``).

1. A percept is neither a success nor a failure. ``capture_from_loop`` defaulted it to ``success=True``
   (the 0.7 "successful interaction" retention floor, and promotion to long-term at sleep), while
   ``store_observation`` / ``capture()`` with no outcome stored it as a FAILURE. Owner decisions
   2026-09-29: unknown in both directions; ``ImportanceBasedStrategy`` scores unknown 0.65, the midpoint
   of success (1.0) and failure (0.3); files already on disk are not migrated.
2. A plain ``SemanticMemory`` concept was reinforced twice per episode.
3. Two percepts in the same second shared one forming-pool key and overwrote each other.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from maxim.memory.encoding import EncodingSignals
from maxim.memory.hippocampus import Hippocampus, HippocampusConfig
from maxim.memory.types import Outcome, Perception

ENC = EncodingSignals.unmeasured("api")


def _hippo(**kw) -> Hippocampus:
    kw.setdefault("auto_save_after_sleep", False)
    return Hippocampus(HippocampusConfig(**kw))


def _percept(h: Hippocampus, text: str = "the kettle is whistling") -> str:
    """A MemoryAgent percept capture: no tool, an empty result, a transcript."""
    return h.capture_from_loop(
        observation={"transcript": text, "salience": 0.8, "novelty": 0.5},
        state=None,
        intent={},
        decision={},
        action={},
        result={},
        situation=None,
        encoding=ENC,
    )


# ── 1. capture ───────────────────────────────────────────────────────────


def test_a_percept_captured_from_the_loop_has_no_outcome() -> None:
    h = _hippo()
    assert h.get(_percept(h)).outcome.success is None


@pytest.mark.parametrize(
    "result, expected",
    [({"success": False}, False), ({"success": True}, True), (SimpleNamespace(success=False, error="x"), False)],
)
def test_a_stated_outcome_is_still_recorded(result, expected) -> None:
    h = _hippo()
    mid = h.capture_from_loop(
        observation={},
        state=None,
        intent={},
        decision={},
        action={"tool": "t"},
        result=result,
        situation=None,
        encoding=ENC,
    )
    assert h.get(mid).outcome.success is expected


def test_an_observation_is_not_stored_as_a_failure() -> None:
    h = _hippo()
    assert h.get(h.store_observation("a door slammed")).outcome.success is None


def test_capture_without_an_outcome_is_unknown_and_an_explicit_one_is_kept() -> None:
    h = _hippo()
    assert h.get(h.capture(perception=Perception(salience=0.5), encoding=ENC)).outcome.success is None
    assert (
        h.get(
            h.capture(perception=Perception(salience=0.5), outcome=Outcome(success=False), encoding=ENC)
        ).outcome.success
        is False
    )


def test_an_outcome_nobody_stated_is_unknown_whichever_door_built_it() -> None:
    """The type's own default, not only the capture doors: a pre-built record stored as is (#843)."""
    from maxim.memory.types import CompressedMemory, EpisodicMemory

    assert Outcome().success is None
    assert Outcome.from_dict({}).success is None
    h = _hippo()
    mid = h.store(EpisodicMemory(id="ep-prebuilt", timestamp=time.time()), encoding=ENC)
    assert h.get(mid).outcome.success is None
    assert CompressedMemory(id="c", timestamp=1.0).success is None
    assert CompressedMemory.from_dict({"id": "c", "timestamp": 1.0}).success is None


@pytest.mark.parametrize("value", [1, 0, "true", 0.5])
def test_both_stored_copies_refuse_anything_but_a_bool_or_none(value) -> None:
    from maxim.memory.types import CompressedMemory

    with pytest.raises(TypeError):
        Outcome(success=value)
    with pytest.raises(TypeError):
        CompressedMemory(id="c", timestamp=1.0, success=value)


@pytest.mark.parametrize("result", [{"success": None}, SimpleNamespace(success=None, error=None)])
def test_a_result_that_states_none_stays_unknown(result) -> None:
    h = _hippo()
    mid = h.capture_from_loop(
        observation={},
        state=None,
        intent={},
        decision={},
        action={"tool": "t"},
        result=result,
        situation=None,
        encoding=ENC,
    )
    assert h.get(mid).outcome.success is None


def test_a_stored_non_bool_makes_the_file_unreadable_and_names_the_record(tmp_path) -> None:
    """Owner's #971 policy for unreadable data: keep a copy, start fresh. The log line names the episode."""
    import json

    path = tmp_path / "hippocampus.json"
    h = _hippo(persistence_path=str(path))
    mid = _percept(h)
    h.save()
    data = json.loads(path.read_text())
    episodes = [m for m in data["memories"] if m.get("id") == mid]
    assert episodes, "fixture: the saved file holds the percept"
    episodes[0]["outcome"]["success"] = 1
    path.write_text(json.dumps(data))

    fresh = _hippo(persistence_path=str(path))
    ok, message = fresh.load_with_recovery()
    assert ok and message and repr(mid) in message
    assert fresh.get(mid) is None
    assert list(tmp_path.glob("hippocampus.json.corrupt-*"))


def test_stats_count_unknown_outcomes_in_their_own_bucket() -> None:
    h = _hippo()
    _percept(h)
    h.capture(perception=Perception(salience=0.5), outcome=Outcome(success=True), encoding=ENC)
    h.capture(perception=Perception(salience=0.5), outcome=Outcome(success=False), encoding=ENC)
    stats = h.stats()
    assert (stats.get("successful"), stats.get("failed"), stats.get("unknown_outcome")) == (1, 1, 1)


def test_unknown_survives_a_save_and_load(tmp_path) -> None:
    path = tmp_path / "hippocampus.json"
    h = _hippo(persistence_path=str(path))
    mid = _percept(h)
    h.save()
    reloaded = _hippo(persistence_path=str(path))
    reloaded.load()
    assert reloaded.get(mid).outcome.success is None


# ── 1. readers ───────────────────────────────────────────────────────────


def _episode(success, *, transcript: str = "hi"):
    h = _hippo()
    mid = h.capture(
        perception=Perception(salience=0.5, transcript=transcript),
        outcome=Outcome(success=success),
        encoding=ENC,
    )
    return h.get(mid)


def test_an_unknown_outcome_earns_no_successful_interaction_floor() -> None:
    from maxim.memory.strategies import AccessBasedStrategy

    strategy = AccessBasedStrategy()
    success, unknown = _episode(True), _episode(None)
    # Long unaccessed, so recency and access give no score of their own: only the floor is left.
    later = time.time() + 365 * 86400
    assert strategy.score_for_retention(success, later) >= 0.7
    assert strategy.score_for_retention(unknown, later) < 0.7


def test_the_importance_strategy_scores_unknown_at_the_midpoint() -> None:
    """Owner decision: 0.65, halfway between success (1.0) and failure (0.3)."""
    from maxim.memory.strategies import ImportanceBasedStrategy

    strategy = ImportanceBasedStrategy(novelty_weight=0.0)
    now = time.time()
    # No transcript (no user bonus) keeps the score under its 1.0 cap, where it is linear in success.
    scores = {s: strategy.score_for_retention(_episode(s, transcript=""), now) for s in (True, None, False)}
    assert scores[None] == pytest.approx((scores[True] + scores[False]) / 2)
    assert scores[False] < scores[None] < scores[True]


def test_a_memory_summary_says_unknown_not_failure() -> None:
    from maxim.memory.hippocampus_retrieval import _memory_summary

    assert "unknown" in _memory_summary(_episode(None))
    assert "failure" in _memory_summary(_episode(False))


def test_a_situation_signature_does_not_call_unknown_a_failure() -> None:
    from maxim.similarity.signature import SituationSignature

    assert SituationSignature.from_memory(_episode(None)).outcome_type == "unknown"
    assert SituationSignature.from_memory(_episode(False)).outcome_type == "failure"


def test_prediction_confidence_ignores_unknown_outcomes() -> None:
    """An unknown outcome is not evidence either way; it used to count as a failure in the rate."""
    from maxim.agents.bus import AgentBus
    from maxim.agents.memory_agent import MemoryAgent
    from maxim.memory.types import PredictedOutcome

    ma = MemoryAgent(AgentBus())

    def pred(success):
        return PredictedOutcome(tool="t", success=success)

    def percept():  # what a percept episode predicts: no tool, no outcome
        return PredictedOutcome(tool="", success=None)

    # One known success is weak evidence (sample factor 1/5); four unknown percepts must neither
    # inflate the sample size nor, with their different "tool", dilute the consistency.
    assert ma._compute_prediction_confidence([pred(True)]) == pytest.approx(0.2)
    assert ma._compute_prediction_confidence([pred(True)] + [percept()] * 4) == pytest.approx(0.2)
    assert ma._compute_prediction_confidence([pred(True)] * 5 + [percept()] * 5) == pytest.approx(1.0)
    assert ma._compute_prediction_confidence([percept()] * 5) == 0.0


# ── 2. one reinforcement per episode ─────────────────────────────────────


def test_a_plain_semantic_memory_is_reinforced_once_per_episode() -> None:
    from maxim.memory.atl import ATL, ATLConfig
    from maxim.memory.concept_extractor import ConceptExtractor
    from maxim.memory.semantic_types import SemanticMemory

    atl = ATL(ATLConfig(persistence_path=None))
    plain = SemanticMemory(
        id="sem-kettle", timestamp=time.time(), name="kettle", category="object", definition="object: kettle"
    )
    atl.store(plain)
    before = plain.reinforcement_count
    from maxim.memory.cross_layer import CrossLayerGraph

    extractor = ConceptExtractor(atl=atl, cross_layer=CrossLayerGraph(layers={"atl": atl}))
    extractor._register_concept("kettle", "object", "ep-1", _episode(None))
    assert plain.reinforcement_count == before + 1


# ── 3. a forming entry per percept ───────────────────────────────────────


def test_two_percepts_in_one_second_keep_two_forming_entries_in_order() -> None:
    from maxim.agents.bus import AgentBus, Percept
    from maxim.agents.memory_agent import MemoryAgent

    ma = MemoryAgent(AgentBus())
    for text in ("first", "second"):
        ma._on_percept(Percept(timestamp=1000.0, source="test", salience=0.9, transcript_chunk=text))
    keys = list(ma._forming_pool)
    assert len(keys) == 2 and keys[0] != keys[1]
    assert keys[-1].endswith("-2")  # the newest really is last


def test_a_percept_with_user_text_is_not_promoted_as_a_successful_interaction() -> None:
    """Criterion 3 promoted every percept with a transcript to long-term (never evicted) as a success."""
    h = _hippo()
    now = time.time()
    assert h._should_promote(_episode(True), now)
    assert not h._should_promote(_episode(None), now)
