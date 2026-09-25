"""Memory-strength Phase 2d-1: every trace records WHEN it happened, in experience µs.

``encoded_at_us`` is immutable (unlike ``retrievability_anchor_us``, which a credited retrieval
moves) and ``capture_seq`` orders captures that share one loop pass's timestamp. The async loop path
stamps both at ENQUEUE, so the worker's lag is not the trace's; every other door captures on the
moment's own thread and takes the defaults. Recording only -- the look-back that reads them is 2d-2.
"""

from __future__ import annotations

import logging

import pytest

from maxim.memory.encoding import EncodingSignals
from maxim.memory.types import CompressedMemory, EpisodicMemory


def _hippo(tmp_path):
    from maxim.memory.hippocampus import Hippocampus, HippocampusConfig

    return Hippocampus(HippocampusConfig(persistence_path=str(tmp_path / "hippo.json")))


def _loop_kwargs():
    return dict(observation={}, state=None, intent={}, decision={}, action={}, result=None)


def test_a_synchronous_capture_is_stamped_now_and_in_sequence(tmp_path):
    hippo = _hippo(tmp_path)
    hippo.experience_clock.advance(5_000_000)
    a = hippo.get(hippo.store_observation("first"))
    b = hippo.get(hippo.store_observation("second"))  # same loop pass: same experience time
    assert a.encoded_at_us == b.encoded_at_us == 5_000_000
    assert b.capture_seq == a.capture_seq + 1  # order survives the shared timestamp
    assert a.retrievability_anchor_us == a.encoded_at_us  # R decays from the same moment


def test_the_async_path_records_when_it_happened_not_when_the_worker_got_to_it(tmp_path):
    hippo = _hippo(tmp_path)
    hippo.experience_clock.advance(1_000_000)
    hippo.capture_from_loop_async(**_loop_kwargs(), encoding=EncodingSignals.unmeasured("loop"), situation=None)
    hippo.experience_clock.advance(3_000_000)  # the queue lags three seconds of experience
    sync_id = hippo.store_observation("captured after the queued one, processed before it")
    hippo._process_capture(hippo._capture_queue.get_nowait())
    queued = next(m for m in hippo if m.id != sync_id)
    later = hippo.get(sync_id)
    assert queued.encoded_at_us == 1_000_000 and later.encoded_at_us == 4_000_000
    assert queued.capture_seq < later.capture_seq  # sequence reserved at enqueue, not on the worker
    assert queued.retrievability_anchor_us == 1_000_000  # R ages from the moment too, not the worker's lag


def test_encoded_at_is_not_moved_by_a_credited_retrieval(tmp_path):
    from maxim.memory.types import update_strength_atomically

    hippo = _hippo(tmp_path)
    rec = hippo.get(hippo.store_observation("x"))
    encoded = rec.encoded_at_us
    update_strength_atomically(rec, lambda s, anchor: (s or 1.0, 9_999_999))  # a retrieval re-anchors
    assert rec.retrievability_anchor_us == 9_999_999 and rec.encoded_at_us == encoded


def test_both_fields_round_trip_and_survive_compression():
    rec = EpisodicMemory(id="m1", timestamp=1.0)
    rec.encoded_at_us, rec.capture_seq = 7_000_000, 42
    back = EpisodicMemory.from_dict(rec.to_dict())
    assert (back.encoded_at_us, back.capture_seq) == (7_000_000, 42)
    comp = CompressedMemory.from_episodic(back)
    assert (comp.encoded_at_us, comp.capture_seq) == (7_000_000, 42)
    comp_back = CompressedMemory.from_dict(comp.to_dict())
    assert (comp_back.encoded_at_us, comp_back.capture_seq) == (7_000_000, 42)


def test_a_file_written_before_2d1_loads_as_not_recorded():
    data = EpisodicMemory(id="m1", timestamp=1.0).to_dict()
    data.pop("encoded_at_us")
    data.pop("capture_seq")
    back = EpisodicMemory.from_dict(data)
    assert back.encoded_at_us is None and back.capture_seq is None


@pytest.mark.parametrize("bad", [-1, 1.5, "7", True])
def test_a_malformed_value_loads_as_not_recorded_and_warns(bad, caplog):
    data = EpisodicMemory(id="m1", timestamp=1.0).to_dict()
    data["encoded_at_us"] = bad
    with caplog.at_level(logging.WARNING):
        assert EpisodicMemory.from_dict(data).encoded_at_us is None
    assert any("encoded_at_us" in r.getMessage() for r in caplog.records)


def test_the_sequence_resumes_past_the_saved_maximum_after_a_load(tmp_path):
    hippo = _hippo(tmp_path)
    for i in range(3):
        hippo.store_observation(f"m{i}")
    saved_max = max(m.capture_seq for m in hippo)
    hippo.save()
    reloaded = _hippo(tmp_path)
    reloaded.load()
    new = reloaded.get(reloaded.store_observation("after restart"))
    assert new.capture_seq == saved_max + 1


@pytest.mark.parametrize(
    ("name", "bad"), [("experience_us", 1.7e9), ("experience_us", True), ("experience_us", -1), ("capture_seq", 2.0)]
)
def test_the_capture_door_refuses_a_value_the_loader_would_refuse(tmp_path, name, bad):
    hippo = _hippo(tmp_path)
    with pytest.raises(ValueError, match=name):
        hippo.capture(encoding=EncodingSignals.unmeasured("observation"), **{name: bad})
    assert len(list(hippo)) == 0  # refused before anything was stored
