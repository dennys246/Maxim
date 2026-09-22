"""Honest activation on MemoryRecord (memory-strength plan Phase 1).

The subclass list is DISCOVERED, not written out: a new MemoryRecord subclass whose own
``to_dict``/``from_dict`` forgets the activation helpers fails here instead of silently
resetting every memory's use history to zero on the next reload.
"""

from __future__ import annotations

import inspect
import threading

import pytest

import maxim.math.math_types  # noqa: F401  (registers MathMemory subclasses)
import maxim.memory.semantic_types  # noqa: F401  (registers SemanticMemory subclasses)
from maxim.math.math_types import CompressedMathMemory, MathMemory
from maxim.memory.semantic_types import CompressedSemantic, SemanticMemory
from maxim.memory.types import CompressedMemory, EpisodicMemory, MemoryRecord


def _concrete_subclasses(cls: type) -> list[type]:
    out = []
    for sub in cls.__subclasses__():
        if not inspect.isabstract(sub):
            out.append(sub)
        out.extend(_concrete_subclasses(sub))
    return out


CONCRETE = _concrete_subclasses(MemoryRecord)


def _activated(cls: type) -> MemoryRecord:
    rec = cls(id="m1", timestamp=1.0)
    rec.activate("enrichment")
    rec.activate("enrichment")
    rec.activate("prediction")
    return rec


def test_discovery_found_every_known_record_type():
    names = {c.__name__ for c in CONCRETE}
    assert {
        "EpisodicMemory",
        "CompressedMemory",
        "SemanticMemory",
        "Concept",
        "CompressedSemantic",
        "MathMemory",
        "CompressedMathMemory",
    } <= names


def test_activate_counts_use_and_leaves_access_tracking_alone():
    rec = EpisodicMemory(id="m1", timestamp=1.0)
    before = (rec.access_count, rec.accessed_at)
    rec.activate("enrichment")
    rec.activate("enrichment")
    rec.activate("prediction")
    assert rec.activation_count == 3
    assert rec.activation_sources == {"enrichment": 2, "prediction": 1}
    assert (rec.access_count, rec.accessed_at) == before


def test_touch_does_not_count_as_activation():
    rec = EpisodicMemory(id="m1", timestamp=1.0)
    rec.touch()
    assert rec.activation_count == 0


@pytest.mark.parametrize("cls", CONCRETE, ids=lambda c: c.__name__)
def test_every_record_type_round_trips_activation(cls):
    restored = cls.from_dict(_activated(cls).to_dict())
    assert restored.activation_count == 3
    assert restored.activation_sources == {"enrichment": 2, "prediction": 1}


@pytest.mark.parametrize("cls", CONCRETE, ids=lambda c: c.__name__)
def test_pre_phase1_files_load_as_never_activated(cls):
    data = cls(id="m1", timestamp=1.0).to_dict()
    for key in ("activation_count", "activation_sources"):
        data.pop(key)
    restored = cls.from_dict(data)
    assert restored.activation_count == 0
    assert restored.activation_sources == {}


@pytest.mark.parametrize(
    ("full_cls", "convert"),
    [
        (EpisodicMemory, CompressedMemory.from_episodic),
        (SemanticMemory, CompressedSemantic.from_semantic),
        (MathMemory, CompressedMathMemory.from_math_record),
    ],
    ids=["episodic", "semantic", "math"],
)
def test_compression_carries_activation(full_cls, convert):
    full = _activated(full_cls)
    compressed = convert(full)
    assert compressed.activation_count == 3
    assert compressed.activation_sources == {"enrichment": 2, "prediction": 1}
    compressed.activate("enrichment")  # a copy, not an alias of the source's dict
    assert full.activation_sources == {"enrichment": 2, "prediction": 1}


def test_concurrent_activation_loses_no_counts():
    rec = EpisodicMemory(id="m1", timestamp=1.0)

    def hammer():
        for _ in range(2000):
            rec.activate("enrichment")

    threads = [threading.Thread(target=hammer) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert rec.activation_count == 16000
    assert rec.activation_sources == {"enrichment": 16000}


# ── store-level: MemoryLayer.activate, the one path per store ──────────────────


def _atl_with(record_id: str):
    from maxim.memory.atl import ATL

    atl = ATL()
    atl.store(SemanticMemory(id=record_id, timestamp=1.0))
    return atl


def _ag_with(record_id: str):
    from maxim.math.angular_gyrus import AngularGyrus

    ag = AngularGyrus()
    ag.store(MathMemory(id=record_id, timestamp=1.0))
    return ag


def test_hippocampus_activate_counts_only_known_ids_once_each(hippocampus, complete_memory_args):
    mid = hippocampus.capture(**complete_memory_args)
    rec = hippocampus.recall_by_ids([mid])[0]
    before = (rec.access_count, rec.accessed_at)
    assert hippocampus.activate([mid, mid, "nope"], source="enrichment") == 1
    assert rec.activation_count == 1  # duplicate ids in one call are ONE use
    assert rec.activation_sources == {"enrichment": 1}
    assert (rec.access_count, rec.accessed_at) == before  # the retention path is not moved


@pytest.mark.parametrize("build", [_atl_with, _ag_with], ids=["atl", "angular_gyrus"])
def test_semantic_and_math_stores_share_the_path(build):
    store = build("r1")
    assert store.activate(["r1"], source="tool") == 1
    assert store.recall_by_ids(["r1"])[0].activation_sources == {"tool": 1}


def test_unknown_source_is_rejected_before_anything_counts(hippocampus, complete_memory_args):
    mid = hippocampus.capture(**complete_memory_args)
    with pytest.raises(ValueError, match="unknown activation source"):
        hippocampus.activate([mid], source="promt")
    assert hippocampus.recall_by_ids([mid])[0].activation_count == 0


def test_bookkeeping_reads_do_not_activate(hippocampus, complete_memory_args):
    mid = hippocampus.capture(**complete_memory_args)
    hippocampus.get(mid)
    hippocampus.recall_by_ids([mid])
    assert hippocampus.recall_by_ids([mid])[0].activation_count == 0


# ── red gate: the counters have no READER until the Phase 2 strength strategy ──


def test_record_level_activate_enforces_the_closed_vocabulary():
    rec = EpisodicMemory(id="m1", timestamp=1.0)
    with pytest.raises(ValueError, match="unknown activation source"):
        rec.activate("prompt")
    assert rec.activation_count == 0 and rec.activation_sources == {}


def test_activate_after_use_raises_on_a_bad_source_and_swallows_a_store_failure(caplog):
    from maxim.memory.layer import activate_after_use

    class Broken:
        def activate(self, ids, *, source):
            raise RuntimeError("store down")

    with pytest.raises(ValueError):
        activate_after_use(Broken(), ["x"], source="promt")
    activate_after_use(Broken(), ["x"], source="tool")  # logged, not raised
    activate_after_use(None, ["x"], source="tool")  # no store wired -> no-op


def _strategy_reads_activation() -> bool:
    from pathlib import Path

    import maxim

    text = (Path(maxim.__file__).resolve().parent / "memory" / "strategies.py").read_text()
    # The bare names, so getattr / dict / attribute access all match.
    return "activation_count" in text or "activation_sources" in text


@pytest.mark.xfail(
    strict=True,
    reason="Phase 1 records honest activation; the retention strategies (memory/strategies.py, the "
    "plan's owned file) read none of it until Phase 2 -- and Phase 2 hooks MemoryLayer.activate "
    "EVENTS, since this massed count cannot seed S. Flips to XPASS when a strategy names the "
    "fields: then replace this with a behavioural test that the strategy's score moves with use.",
)
def test_phase2_strength_strategy_reads_activation():
    assert _strategy_reads_activation()
