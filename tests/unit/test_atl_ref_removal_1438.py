"""#1438: the ATL's memory refs change only under the ATL's write lock, and the loader can prune dangling ones.

``ConceptExtractor._remove_refs_for`` runs as a Hippocampus deletion callback (under the Hippocampus write lock)
while the session end may be saving the ATL. It edited ``Concept.memory_refs`` outside the ATL's lock, racing
``ATL.dump``'s iteration of the same dicts. It now goes through ``ATL.remove_memory_refs``. On CPython the GIL makes
``dump``'s inner ``sorted(ids.keys())`` atomic, so the stress test rarely tears even without the lock; the
deterministic guard is ``test_deletion_ref_removal_waits_for_the_atl_lock`` (proven by deleting the lock).
``ATL.prune_refs`` is the post-load pass's ref pruning.
"""

from __future__ import annotations

import sys
import threading
import time
import traceback

import pytest

from maxim.memory.atl import ATL
from maxim.memory.concept_extractor import ConceptExtractor
from maxim.memory.cross_layer import CrossLayerGraph
from maxim.memory.semantic_types import CompressedSemantic, Concept, SemanticMemory


def _concept(cid: str, refs: dict[str, list[str]] | None = None, sources: list[str] | None = None) -> Concept:
    c = Concept(id=cid, timestamp=time.time(), name=cid, category="object", source_episode_ids=list(sources or []))
    for layer, ids in (refs or {}).items():
        for mid in ids:
            c.add_ref(layer, mid)
    return c


@pytest.fixture
def forced_switching():
    old = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        yield
    finally:
        sys.setswitchinterval(old)


def test_deletion_ref_removal_races_dump_under_forced_switching(forced_switching) -> None:
    errors: list[str] = []
    for _trial in range(3):
        atl = ATL()
        mids = [f"m{i}" for i in range(Concept.MAX_REFS_PER_LAYER)]
        for c in range(4):
            atl.store(_concept(f"c{c}", {"hippocampus": mids}))
        ext = ConceptExtractor(atl=atl, cross_layer=CrossLayerGraph(layers={"atl": atl}), scn=None, start_worker=False)
        ext.rebuild_reverse_index()
        stop = threading.Event()

        def deleter(ext: ConceptExtractor = ext) -> None:
            try:
                for mid in mids:
                    ext.on_memory_deleted(mid)
            except Exception:
                errors.append(traceback.format_exc())

        def dumper(atl: ATL = atl, stop: threading.Event = stop) -> None:
            try:
                while not stop.is_set():
                    atl.dump()
            except Exception:
                errors.append(traceback.format_exc())

        readers = [threading.Thread(target=dumper) for _ in range(2)]
        writer = threading.Thread(target=deleter)
        for t in [*readers, writer]:
            t.start()
        writer.join(60)
        stop.set()
        for t in readers:
            t.join(60)
        assert not any(t.is_alive() for t in [*readers, writer]), "a thread is still running (deadlock?)"
        assert not errors, errors[0]
        assert all(c.ref_count("hippocampus") == 0 for c in atl if isinstance(c, Concept))


def test_remove_memory_refs_counts_and_does_not_touch() -> None:
    atl = ATL()
    atl.store(_concept("a", {"hippocampus": ["m1", "m2"], "angular_gyrus": ["m1"]}))
    atl.store(_concept("b", {"hippocampus": ["m2"]}))
    before = {c.id: c.access_count for c in atl}
    assert atl.remove_memory_refs("hippocampus", "m1", ["a", "b", "missing"]) == 1
    a, b = atl.recall_by_ids(["a", "b"])
    assert dict(a.memory_refs["hippocampus"]) == {"m2": None}
    assert dict(a.memory_refs["angular_gyrus"]) == {"m1": None}  # another layer is untouched
    assert dict(b.memory_refs["hippocampus"]) == {"m2": None}
    assert {c.id: c.access_count for c in atl} == before  # deletion bookkeeping is not a use


def test_prune_refs_drops_dangling_keeps_valid_and_leaves_other_kinds() -> None:
    atl = ATL()
    atl.store(_concept("a", {"hippocampus": ["keep", "gone1"], "angular_gyrus": ["gone1"]}, sources=["keep", "gone2"]))
    plain = SemanticMemory(id="s", timestamp=time.time(), name="s", category="object", source_episode_ids=["gone3"])
    atl.store(plain)
    atl.store(CompressedSemantic(id="z", timestamp=time.time(), name="z", category="object"))

    # memory_refs gone1 + source ids gone2 and gone3
    assert atl.prune_refs("hippocampus", {"keep"}) == 3
    a, s, z = atl.recall_by_ids(["a", "s", "z"])
    assert dict(a.memory_refs["hippocampus"]) == {"keep": None}
    assert dict(a.memory_refs["angular_gyrus"]) == {"gone1": None}  # only the named layer
    assert a.source_episode_ids == ["keep"]
    assert s.source_episode_ids == []
    assert isinstance(z, CompressedSemantic)

    # Another layer prunes only its refs, never source_episode_ids.
    atl.store(_concept("b", {"angular_gyrus": ["x"]}, sources=["y"]))
    assert atl.prune_refs("angular_gyrus", set()) == 2  # a's gone1 + b's x
    (b,) = atl.recall_by_ids(["b"])
    assert b.source_episode_ids == ["y"]
    assert atl.prune_refs("hippocampus", {"keep", "y"}) == 0


def test_deletion_ref_removal_waits_for_the_atl_lock() -> None:
    """Deterministic half of the guard: on CPython the GIL makes ``sorted(dict.keys())`` atomic, so the stress test
    above rarely tears even without the lock. A removal that goes through the ATL's write lock must WAIT while a
    reader (a ``dump`` in flight) holds the read lock; the pre-#1438 code edited the concept immediately."""
    atl = ATL()
    atl.store(_concept("a", {"hippocampus": ["m1"]}))
    ext = ConceptExtractor(atl=atl, cross_layer=CrossLayerGraph(layers={"atl": atl}), scn=None, start_worker=False)
    ext.rebuild_reverse_index()
    done = threading.Event()
    with atl._rwlock.read():
        t = threading.Thread(target=lambda: (ext.on_memory_deleted("m1"), done.set()), daemon=True)
        t.start()
        assert not done.wait(0.3), "the ref removal did not wait for the ATL lock"
        assert "m1" in atl._concepts["a"].memory_refs["hippocampus"]  # no lock taken: a read lock is held here
    assert done.wait(5)
    assert "m1" not in atl._concepts["a"].memory_refs["hippocampus"]


def test_ref_add_waits_for_the_atl_lock() -> None:
    """The ADD side (#1438 follow-up): ``Concept.add_ref`` can create a layer key in ``memory_refs``, which ``dump``
    iterates. ``ConceptExtractor._link_situation`` must wait while a reader holds the read lock."""
    from types import SimpleNamespace

    atl = ATL()
    atl.store(_concept("c"))
    ext = ConceptExtractor(atl=atl, cross_layer=CrossLayerGraph(layers={"atl": atl}), scn=None, start_worker=False)
    linked = threading.Event()
    with atl._rwlock.read():
        record = SimpleNamespace(situation={"world": "c"})
        threading.Thread(target=lambda: (ext._link_situation("m1", record), linked.set()), daemon=True).start()
        assert not linked.wait(0.3), "add_ref did not wait for the ATL lock"
        assert "hippocampus" not in atl._concepts["c"].memory_refs  # no lock taken: a read lock is held here
    assert linked.wait(5)
    assert dict(atl._concepts["c"].memory_refs["hippocampus"]) == {"m1": None}
    assert ext._reverse_index["m1"] == {"c"}


def test_reinforce_waits_for_the_atl_lock() -> None:
    """``reinforce`` appends to ``source_episode_ids``, which ``dump`` serializes: ``find_or_create``'s
    reinforcement of an existing concept must wait while a reader holds the read lock."""
    atl = ATL()
    atl.store(_concept("c"))
    reinforced = threading.Event()
    with atl._rwlock.read():
        threading.Thread(
            target=lambda: (atl.find_or_create("c", "object", source_episode_id="e1"), reinforced.set()), daemon=True
        ).start()
        assert not reinforced.wait(0.3), "reinforce did not wait for the ATL lock"
        assert atl._concepts["c"].source_episode_ids == []
    assert reinforced.wait(5)
    assert atl._concepts["c"].source_episode_ids == ["e1"]


def test_to_dict_shares_no_live_container() -> None:
    """``ATL.save`` serializes ``dump()`` after releasing its lock: the payload must not be the live lists."""
    c = _concept("c", sources=["e1"])
    c.properties["k"] = "v"
    d = c.to_dict()
    assert d["source_episode_ids"] == ["e1"] and d["source_episode_ids"] is not c.source_episode_ids
    assert d["properties"] == {"k": "v"} and d["properties"] is not c.properties


def test_concept_refs_are_written_only_through_the_atl() -> None:
    """Static half: the extractor and grounder never call ``add_ref``/``remove_ref``/``reinforce`` directly."""
    from pathlib import Path

    import maxim.memory.concept_extractor as ce
    import maxim.memory.concept_grounder as cg

    for mod in (ce, cg):
        src = Path(mod.__file__).read_text()
        for call in (".add_ref(", ".remove_ref(", ".reinforce("):
            assert call not in src, f"{mod.__name__} calls {call} outside the ATL's lock"


def test_iter_and_ref_index_are_snapshots() -> None:
    atl = ATL()
    atl.store(_concept("a", {"hippocampus": ["m1", "m2"]}))
    atl.store(_concept("b", {"hippocampus": ["m2"], "angular_gyrus": ["r1"]}))
    seen = []
    for concept in atl:
        seen.append(concept.id)
        atl.store(_concept(f"new-{concept.id}"))  # would raise on the live dict, and must not deadlock
    assert sorted(seen) == ["a", "b"]
    assert atl.ref_index("hippocampus") == {"m1": {"a"}, "m2": {"a", "b"}}
    assert atl.ref_index("angular_gyrus") == {"r1": {"b"}}
