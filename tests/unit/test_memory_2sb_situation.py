"""Memory-strength Phase 2S-b (#848): a loop capture records the situation it happened in.

The situation is the loop's substrate clusters at capture, ``{modality: EC cluster id}``. Those
ids ARE ATL concept ids, so the ConceptExtractor links the trace to them -- the substrate-native
cue survival memory lacked. The end-to-end composition (a real loop capture linking the water
cluster) is pinned in ``test_water_trial_smoke.py``; this module pins the contract pieces.
"""

from __future__ import annotations

import logging

import pytest

from maxim.memory.encoding import EncodingSignals
from maxim.memory.types import EpisodicMemory

SITUATION = {"interoception": "c-intero", "world": "c-world"}


def _hippo(tmp_path):
    from maxim.memory.hippocampus import Hippocampus, HippocampusConfig

    return Hippocampus(HippocampusConfig(persistence_path=str(tmp_path / "hippo.json")))


def _loop_kwargs(**over):
    kw = dict(observation={}, state=None, intent={}, decision={}, action={}, result=None)
    kw.update(over)
    return kw


# ── the record ───────────────────────────────────────────────────────────────


def test_the_situation_round_trips_through_persistence():
    rec = EpisodicMemory(id="m1", timestamp=1.0, situation=dict(SITUATION))
    assert EpisodicMemory.from_dict(rec.to_dict()).situation == SITUATION


def test_a_file_written_before_2s_b_loads_as_no_situation():
    data = EpisodicMemory(id="m1", timestamp=1.0).to_dict()
    data.pop("situation")
    assert EpisodicMemory.from_dict(data).situation is None


@pytest.mark.parametrize("bad", [["c-world"], {"world": 3}, {"": "c"}, "c-world"])
def test_a_malformed_situation_loads_as_not_recorded_and_warns(bad, caplog):
    data = EpisodicMemory(id="m1", timestamp=1.0).to_dict()
    data["situation"] = bad
    with caplog.at_level(logging.WARNING):
        assert EpisodicMemory.from_dict(data).situation is None
    assert any("malformed situation" in r.getMessage() for r in caplog.records)


# ── the capture doors ────────────────────────────────────────────────────────


def test_capture_from_loop_requires_the_situation(tmp_path):
    with pytest.raises(TypeError, match="situation"):
        _hippo(tmp_path).capture_from_loop(**_loop_kwargs(), encoding=EncodingSignals.unmeasured("loop"))


def test_capture_from_loop_stores_the_situation(tmp_path):
    hippo = _hippo(tmp_path)
    mid = hippo.capture_from_loop(**_loop_kwargs(), encoding=EncodingSignals.unmeasured("loop"), situation=SITUATION)
    assert hippo.get(mid).situation == SITUATION


@pytest.mark.parametrize("empty", [None, {}])
def test_no_clusters_is_no_situation(tmp_path, empty):
    hippo = _hippo(tmp_path)
    mid = hippo.capture_from_loop(**_loop_kwargs(), encoding=EncodingSignals.unmeasured("loop"), situation=empty)
    assert hippo.get(mid).situation is None


@pytest.mark.parametrize("bad", [["c"], {"world": ""}, {"world": 3}, {"": "c"}])
def test_a_bad_situation_fails_on_the_callers_thread_before_it_is_queued(tmp_path, bad):
    from maxim.memory.encoding import SituationContractError

    hippo = _hippo(tmp_path)
    with pytest.raises(SituationContractError):
        hippo.capture_from_loop_async(**_loop_kwargs(), encoding=EncodingSignals.unmeasured("loop"), situation=bad)
    assert hippo._capture_queue.qsize() == 0


def test_a_malformed_situation_escapes_the_loops_capture_instead_of_dropping_the_trace(tmp_path):
    """The loop's capture swallows ordinary failures at DEBUG; a contract break must not vanish."""
    from maxim.memory.encoding import EncodingContractError, SituationContractError
    from maxim.runtime.bio_integration import capture_episodic_memory

    assert issubclass(SituationContractError, EncodingContractError)
    with pytest.raises(SituationContractError):
        capture_episodic_memory(
            hippocampus=_hippo(tmp_path),
            executor=None,
            observation={},
            state=None,
            intent={},
            action={"tool_name": "t"},
            result=None,
            run_id="r",
            situation={"world": 3},
        )


def test_a_prebuilt_record_keeps_its_own_situation(tmp_path):
    hippo = _hippo(tmp_path)
    rec = EpisodicMemory(id="pre", timestamp=1.0, situation=dict(SITUATION))
    mid = hippo.capture(record=rec, encoding=EncodingSignals.unmeasured("loop"))
    assert hippo.get(mid).situation == SITUATION


def test_the_async_path_carries_the_situation_to_the_trace(tmp_path):
    hippo = _hippo(tmp_path)
    hippo.capture_from_loop_async(**_loop_kwargs(), encoding=EncodingSignals.unmeasured("loop"), situation=SITUATION)
    hippo._process_capture(hippo._capture_queue.get_nowait())
    [trace] = list(hippo)
    assert trace.situation == SITUATION


# ── the link ─────────────────────────────────────────────────────────────────


def _extractor_with_atl():
    from unittest.mock import MagicMock

    from maxim.memory.atl import ATL
    from maxim.memory.concept_extractor import ConceptExtractor

    atl = ATL()
    extractor = ConceptExtractor(atl=atl, cross_layer=MagicMock())
    return atl, extractor


def test_the_extractor_links_existing_situation_concepts_and_creates_none(tmp_path):
    atl, extractor = _extractor_with_atl()
    try:
        world_id, _ = atl.find_or_create(name="sensors:is_in_water=1.00", category="sensor", definition="x")
        before = len(list(atl))
        rec = EpisodicMemory(id="m1", timestamp=1.0, situation={"world": world_id})
        extractor._process_capture("m1", rec)
        assert "m1" in atl.get(world_id).memory_refs.get("hippocampus", {})
        assert world_id in extractor._reverse_index["m1"]
        assert len(list(atl)) == before  # linked the existing concept, invented nothing
    finally:
        extractor.shutdown()


def test_a_compressed_situation_concept_is_reported_not_silently_skipped(caplog):
    from maxim.memory.semantic_types import CompressedSemantic

    atl, extractor = _extractor_with_atl()
    try:
        atl._concepts["c-comp"] = CompressedSemantic(id="c-comp", timestamp=0.0, name="sensors:x")
        with caplog.at_level(logging.WARNING):
            extractor._process_capture("m1", EpisodicMemory(id="m1", timestamp=1.0, situation={"world": "c-comp"}))
        assert any("compressed ATL concept" in r.getMessage() for r in caplog.records)
    finally:
        extractor.shutdown()


def test_situation_links_survive_a_reload_and_are_removed_with_the_trace():
    atl, extractor = _extractor_with_atl()
    try:
        world_id, _ = atl.find_or_create(name="sensors:is_in_water=1.00", category="sensor", definition="x")
        extractor._process_capture("m1", EpisodicMemory(id="m1", timestamp=1.0, situation={"world": world_id}))
        extractor._reverse_index.clear()  # a fresh process: the reverse index is rebuilt from the refs
        extractor.rebuild_reverse_index()
        assert world_id in extractor._reverse_index["m1"]
        extractor.on_memory_deleted("m1")
        assert "m1" not in atl.get(world_id).memory_refs.get("hippocampus", {})
    finally:
        extractor.shutdown()


def test_a_cluster_missing_from_the_atl_warns_once_per_modality(caplog):
    atl, extractor = _extractor_with_atl()
    try:
        with caplog.at_level(logging.WARNING):
            for i in range(3):
                extractor._process_capture(
                    f"m{i}", EpisodicMemory(id=f"m{i}", timestamp=1.0, situation={"world": "absent"})
                )
        warnings = [r for r in caplog.records if "not an ATL concept" in r.getMessage()]
        assert len(warnings) == 1
    finally:
        extractor.shutdown()
