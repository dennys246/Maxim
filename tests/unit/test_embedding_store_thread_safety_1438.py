"""#1438: the EC's semantic ``EmbeddingStore`` is safe to save while the embedder worker and deletions still write it.

The embedder's worker thread ``set``s embeddings (the hub's callback) and Hippocampus deletion callbacks ``remove``
them under the Hippocampus write lock, while the session end saves the store. Before #1438 the store had no lock:
``save`` iterated the live dict (``dictionary changed size during iteration``, or a ``KeyError`` from
``self._embeddings[mid]`` after a concurrent remove) and the embeddings were not saved.
"""

from __future__ import annotations

import sys
import threading
import traceback
from pathlib import Path

import numpy as np
import pytest

from maxim.similarity.semantic import EmbeddingStore


class _StubEmbedder:
    """Stands in for ``NeuralSemanticLSH``: healthy, deterministic, no model."""

    is_healthy = True

    def embed(self, text: str) -> np.ndarray:
        v = np.zeros(8)
        v[hash(text) % 8] = 1.0
        return v

    def hash(self, text: str) -> tuple[int, ...]:
        return tuple(int(x) for x in self.embed(text))


@pytest.fixture
def forced_switching():
    old = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        yield
    finally:
        sys.setswitchinterval(old)


def test_save_and_readers_race_set_and_remove_under_forced_switching(tmp_path: Path, forced_switching) -> None:
    errors: list[str] = []
    for trial in range(3):
        store = EmbeddingStore(embedder=_StubEmbedder())
        store.set("seed", np.ones(8), (1,) * 8)
        path = str(tmp_path / f"emb{trial}.npz")
        stop = threading.Event()

        def writer(t: int, store: EmbeddingStore = store) -> None:
            try:
                for i in range(400):
                    store.set(f"m{t}-{i}", np.full(8, float(i)), (i % 2,) * 8)
                    if i % 3 == 0:
                        store.remove(f"m{t}-{i - 1}")
            except Exception:
                errors.append(traceback.format_exc())

        def reader(read, store: EmbeddingStore = store, stop: threading.Event = stop) -> None:
            try:
                while not stop.is_set():
                    read(store)
            except Exception:
                errors.append(traceback.format_exc())

        reads = [
            lambda s: s.save(path),
            lambda s: s.find_similar("hello", k=3, threshold=0.0),
            lambda s: (len(s), s.get("seed"), s.get_hash("seed")),
            lambda s: s.embed_missing({"x": "a", "seed": "b"}),
        ]
        readers = [threading.Thread(target=reader, args=(r,)) for r in reads]
        writers = [threading.Thread(target=writer, args=(t,)) for t in range(3)]
        for t in readers + writers:
            t.start()
        for t in writers:
            t.join(timeout=60)
        stop.set()
        for t in readers:
            t.join(timeout=60)
        assert not any(t.is_alive() for t in readers + writers), "a thread is still running (deadlock?)"
        assert not errors, errors[0]

        store.save(path)
        loaded = EmbeddingStore(embedder=_StubEmbedder())
        assert loaded.load(path) == len(store)


def test_save_writes_the_file_outside_the_lock(tmp_path: Path, monkeypatch) -> None:
    store = EmbeddingStore(embedder=_StubEmbedder())
    store.set("a", np.ones(8))
    in_write = threading.Event()
    release = threading.Event()
    real = np.savez_compressed

    def slow(*a, **kw):
        in_write.set()
        assert release.wait(10)
        return real(*a, **kw)

    monkeypatch.setattr(np, "savez_compressed", slow)
    saver = threading.Thread(target=store.save, args=(str(tmp_path / "e.npz"),))
    saver.start()
    assert in_write.wait(10)
    done = threading.Event()
    threading.Thread(target=lambda: (store.set("b", np.ones(8)), done.set()), daemon=True).start()
    assert done.wait(5), "set blocked behind save's file write: the write holds the store lock"
    release.set()
    saver.join(10)
    loaded = EmbeddingStore(embedder=_StubEmbedder())
    assert loaded.load(str(tmp_path / "e.npz")) == 1  # the snapshot predates "b"


def test_find_similar_embeds_the_query_with_the_lock_free() -> None:
    """The model forward pass must not run under the store's lock (a slow encode would stall every deletion)."""
    store = EmbeddingStore(embedder=_StubEmbedder())
    store.set("a", np.ones(8))
    free: list[bool] = []

    class _Probe(_StubEmbedder):
        def embed(self, text: str) -> np.ndarray:
            got: list[bool] = []

            def try_lock() -> None:
                ok = store._lock.acquire(timeout=2)
                if ok:
                    store._lock.release()  # released by the thread that took it (an RLock requires it)
                got.append(ok)

            t = threading.Thread(target=try_lock)
            t.start()
            t.join()
            free.append(bool(got and got[0]))
            return super().embed(text)

    store.embedder = _Probe()
    store.find_similar("q", threshold=-1.0)
    assert free == [True]
