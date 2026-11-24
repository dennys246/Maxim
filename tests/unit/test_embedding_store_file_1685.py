"""#1685 + #1531: the semantic embedding file loads without pickle, saves atomically, and an emptied store saves.

#1685: `EmbeddingStore.load` called `np.load(path, allow_pickle=True)`, because `save` wrote the memory ids and the
hashes as object arrays, which numpy can only read back through pickle. A crafted `.npz` ran arbitrary code on load.
Version 2 stores both as plain string arrays and loads with pickle off. A version 1 file is refused with one WARNING
and the store starts empty (owner decision 2026-11-23): the embeddings are computed from memory text at capture, and
nothing re-embeds an old memory, so the memories it held drop out of semantic search. Semantic similarity is off by
default (`enable_semantic=False`).

#1531: `save` wrote straight to the target (a kill mid-write tore the file) and returned without writing when the
store was empty, so embeddings removed during a session came back from the old file next session.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest

from maxim.similarity.semantic import EmbeddingStore


class _WritesMarker:
    """Unpickling this object creates ``marker``: the visible side effect of a pickle payload."""

    def __init__(self, marker: str) -> None:
        self.marker = marker

    def __reduce__(self):
        return (open, (self.marker, "w"))


def _store(**embeddings: list[float]) -> EmbeddingStore:
    store = EmbeddingStore(embedder=None)  # type: ignore[arg-type]  # save and load never touch the embedder
    for mid, vec in embeddings.items():
        store.set(mid, np.array(vec, dtype=np.float32), (1, 0, 1))
    return store


def _crafted(path: Path, marker: Path, version: int) -> None:
    np.savez_compressed(
        path,
        memory_ids=np.array([_WritesMarker(str(marker))], dtype=object),
        embeddings=np.zeros((1, 2), dtype=np.float32),
        hashes_json=np.array(["{}"], dtype=object),
        version=np.array([version]),
    )


@pytest.mark.parametrize("version", [1, 2])
def test_a_crafted_file_runs_no_code(tmp_path: Path, version: int) -> None:
    path, marker = tmp_path / "emb.npz", tmp_path / "PWNED"
    _crafted(path, marker, version)
    store = _store()
    assert store.load(str(path)) == 0
    assert not marker.exists()
    assert len(store) == 0


def test_a_version_1_file_is_refused_with_one_warning(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    path = tmp_path / "emb.npz"
    np.savez_compressed(
        path,
        memory_ids=np.array(["m1"], dtype=object),
        embeddings=np.ones((1, 2), dtype=np.float32),
        hashes_json=np.array(['{"m1": [1]}'], dtype=object),
        version=np.array([1]),
    )
    store = _store()
    with caplog.at_level(logging.WARNING, logger="maxim.similarity.semantic"):
        assert store.load(str(path)) == 0
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "pickle" in warnings[0].getMessage()
    assert len(store) == 0


def test_save_then_load_round_trips_without_pickle(tmp_path: Path) -> None:
    path = tmp_path / "emb.npz"
    _store(m1=[1.0, 2.0], m2=[3.0, 4.0]).save(str(path))
    with np.load(path, allow_pickle=False) as data:  # every array in the file reads with pickle off
        assert {k: data[k].dtype.kind for k in data.files}["memory_ids"] == "U"
    loaded = _store()
    assert loaded.load(str(path)) == 2
    assert loaded.get("m2").tolist() == [3.0, 4.0]
    assert loaded.get_hash("m1") == (1, 0, 1)


def test_an_emptied_store_saves_so_removed_embeddings_stay_removed(tmp_path: Path) -> None:
    path = tmp_path / "emb.npz"
    store = _store(m1=[1.0, 2.0])
    store.save(str(path))
    store.remove("m1")
    store.save(str(path))
    assert _store().load(str(path)) == 0


def test_save_goes_through_atomic_write_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "emb.npz"
    _store(m1=[1.0, 2.0]).save(str(path))

    def _killed(*_a, **_k):
        raise OSError("killed mid-write")

    monkeypatch.setattr("maxim.utils.atomic_io.atomic_write_bytes", _killed)
    with pytest.raises(OSError):
        _store(m2=[5.0, 6.0]).save(str(path))
    loaded = _store()
    assert loaded.load(str(path)) == 1  # the earlier file is whole
    assert loaded.get("m1").tolist() == [1.0, 2.0]


def _write_version(path: Path, version: int | None) -> None:
    arrays = {
        "memory_ids": np.array(["m1"]),
        "embeddings": np.ones((1, 2), dtype=np.float32),
        "hashes_json": np.array(['{"m1": [1]}']),
    }
    if version is not None:
        arrays["version"] = np.array([version])
    np.savez_compressed(path, **arrays)


@pytest.mark.parametrize("version", [3, None], ids=["newer_version", "no_version_key"])
def test_an_unreadable_file_is_refused_and_left_alone(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, version: int | None
) -> None:
    """A file this code cannot read (a newer Maxim's, or one with no version) is refused, and an empty save keeps it."""
    path = tmp_path / "emb.npz"
    _write_version(path, version)
    before = path.read_bytes()
    store = _store()
    with caplog.at_level(logging.WARNING, logger="maxim.similarity.semantic"):
        assert store.load(str(path)) == 0
    assert [r.levelno for r in caplog.records if r.levelno >= logging.WARNING] == [logging.WARNING]
    store.save(str(path))
    assert path.read_bytes() == before


def test_a_session_that_embedded_nothing_leaves_another_agents_file(tmp_path: Path) -> None:
    """The path is machine-wide (#984): agent B, which never loaded the file and embedded nothing, must not wipe A's."""
    path = tmp_path / "emb.npz"
    agent_b = _store()  # started before A's file existed
    _store(m1=[1.0, 2.0]).save(str(path))  # agent A ends first
    agent_b.save(str(path))  # agent B ends second, empty
    assert _store().load(str(path)) == 1


def test_a_refused_version_1_file_is_replaced_by_the_next_save(tmp_path: Path) -> None:
    """The owner decision: the refused old-format file is superseded, even by an empty store."""
    path = tmp_path / "emb.npz"
    _write_version(path, 1)
    store = _store()
    assert store.load(str(path)) == 0
    store.save(str(path))
    with np.load(path, allow_pickle=False) as data:
        assert int(data["version"][0]) == 2


def test_clear_lets_an_empty_store_save(tmp_path: Path) -> None:
    path = tmp_path / "emb.npz"
    _store(m1=[1.0, 2.0]).save(str(path))
    store = _store()
    store.load(str(path))
    store.clear()
    store.save(str(path))
    assert _store().load(str(path)) == 0
