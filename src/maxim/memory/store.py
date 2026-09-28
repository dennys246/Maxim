"""Split persistence protocols for Maxim memory subsystems.

Three protocols — one per subsystem — because each has fundamentally
different query patterns:
- Episodic (Hippocampus): similarity search, time-range queries
- Causal (NAc): event→outcome lookups
- Semantic (ATL): concept-type filtering

Default implementations (``File*Store``) wrap the current JSON
persistence behavior.  Database implementations (PostgreSQL +
pgvector) are provided by the ``[database]`` extra for Mother Maxim.

Example::

    from maxim.memory.store import FileEpisodicStore

    store = FileEpisodicStore("~/.maxim/memory/hippocampus.json")
    store.save(memories)
    loaded = store.load()
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# File ownership: a store never writes over a file it did not read (#939)
# ---------------------------------------------------------------------------


# What a store's loader raises for a file it cannot parse or rebuild (JSONDecodeError is a ValueError).
# ONE definition, used by every "start fresh from an unreadable file" path and by maxim.load.* (#939);
# anything else (OSError: permissions, EIO) is not corruption, and never licenses replacing the file.
UNREADABLE_STORE_ERRORS: tuple[type[Exception], ...] = (ValueError, KeyError, TypeError, AttributeError)

# Copies this process made, so identical bytes met by two loaders in one construction are kept once,
# while a corruption that RECURS across runs still leaves one dated copy per run.
_COPIES_MADE_THIS_PROCESS: set[str] = set()


class StoreFileOwnership:
    """Mixin for a memory store that persists to one JSON file (Hippocampus, ATL).

    A store may write a file it READ, a file that did not exist when it first wrote it (it created
    it), or a file it was explicitly told to overwrite (``save(overwrite=True)`` /
    ``allow_overwrite()``). Anything else is an existing file whose memories this instance never
    loaded, and saving would destroy them, so ``save`` raises ``StoreOverwriteRefused`` (#939: a
    ``create.*`` store, the plain constructor and ``from_config`` with the path on the config all used
    to start empty and clobber the file on the next save). Enforced at the write, so every path that
    builds a store is covered, including ones added later.

    Ownership is per instance and never released: it is not a lock. Two live stores that both read
    one file can still overwrite each other's saves, and a file deleted and recreated by another writer
    is still "owned" here. The guard stops the empty-store clobber, not concurrent writers.

    The class that mixes this in names itself in ``_store_name`` and exposes ``config.persistence_path``.
    """

    _store_name: str = "store"

    def _owned_store_files(self) -> set[str]:
        return self.__dict__.setdefault("_store_owned_files", set())

    @staticmethod
    def _store_file_key(path: str) -> str:
        import os

        from maxim.utils.paths import store_file_path

        return os.path.realpath(store_file_path(path))

    def _default_store_path(self, path: str | None) -> str:
        chosen = path or getattr(getattr(self, "config", None), "persistence_path", None)
        if not chosen:
            raise ValueError(f"{self._store_name}: no path given and no persistence_path configured")
        from maxim.utils.paths import store_file_path

        return store_file_path(chosen)

    def may_write(self, path: str | None = None) -> bool:
        """Whether ``save`` would write ``path`` (default: the configured one) without refusing."""
        import os

        target = self._default_store_path(path)
        return not os.path.exists(target) or self._store_file_key(target) in self._owned_store_files()

    def allow_overwrite(self, path: str | None = None) -> None:
        """Declare that this store may replace ``path`` (default: the configured one) without reading it.

        For a deliberately write-but-don't-read store (the sim NPC, a fresh start over a file the
        operator chose to discard). The file's current contents are lost on the next save.
        """
        import os

        target = self._default_store_path(path)
        self._owned_store_files().add(self._store_file_key(target))
        if os.path.exists(target):
            log.warning("%s: will overwrite %s without reading it (declared)", self._store_name, target)

    def set_aside_unreadable_file(self, path: str | None = None) -> str | None:
        """Keep an unreadable store file as evidence, then let this store save in its place.

        Copies ``path`` (default: the configured one) to ``<name>.corrupt-<UTC timestamp>`` beside
        it, logs where it went, and claims the original, so a store that started empty after a
        failed load persists normally instead of being refused forever (owner decision 2026-09-28,
        #939). Returns the copy's path, or None when there is no file.
        """
        import shutil
        import time as _time
        from pathlib import Path

        import filecmp

        target = Path(self._default_store_path(path))
        if not target.exists():
            return None
        # A failed load can leave PART of the file in the store; "starting empty" must be true before
        # the store is allowed to save over the file.
        self._reset_store_state()
        # Two loaders can meet the same corrupt file in one construction (create_full_agent's factory,
        # then its bio stack): keep ONE copy of identical bytes -- among copies made by THIS process.
        for existing in sorted(_COPIES_MADE_THIS_PROCESS):
            existing_path = Path(existing)
            if existing_path.parent == target.parent and existing_path.name.startswith(f"{target.name}.corrupt-"):
                if existing_path.exists() and filecmp.cmp(existing_path, target, shallow=False):
                    self._claim_store_file(str(target))
                    log.warning(
                        "%s: %s could not be read; its copy is already at %s", self._store_name, target, existing
                    )
                    return existing
        stamp = _time.strftime("%Y%m%dT%H%M%SZ", _time.gmtime())
        copy = target.with_name(f"{target.name}.corrupt-{stamp}")
        n = 1
        while copy.exists():
            copy = target.with_name(f"{target.name}.corrupt-{stamp}-{n}")
            n += 1
        shutil.copy2(target, copy)
        _COPIES_MADE_THIS_PROCESS.add(str(copy))
        self._claim_store_file(str(target))
        log.warning(
            "%s: %s could not be read; kept a copy at %s and starting empty -- the next save replaces it",
            self._store_name,
            target,
            copy,
        )
        return str(copy)

    def _reset_store_state(self) -> None:
        """Empty this store completely, through its own snapshot contract: restore the dump of a fresh
        store with the same config (every surface ``load_state`` owns -- graph, indexes, stats)."""
        fresh = type(self)(getattr(self, "config", None))  # type: ignore[call-arg]
        self.load_state(fresh.dump())  # type: ignore[attr-defined]

    def _claim_store_file(self, path: str) -> None:
        """Record that this store read, created or was told to replace ``path``."""
        self._owned_store_files().add(self._store_file_key(path))

    def _check_store_write(self, path: str, *, overwrite: bool) -> None:
        """Raise ``StoreOverwriteRefused`` unless this store may write ``path`` (expanded already)."""
        if overwrite:
            self._claim_store_file(path)
            return
        if not self.may_write(path):
            from maxim.exceptions import StoreOverwriteRefused

            raise StoreOverwriteRefused(
                f"{self._store_name}: refusing to save over {path}, which this instance never read -- "
                f"it would replace the memories stored there. Open it with maxim.load.{self._store_name}() "
                "(or the store's load()) to keep them, save to another path, or pass overwrite=True / call "
                "allow_overwrite() to replace them deliberately.",
                path=path,
                store=self._store_name,
            )


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class EpisodicStore(Protocol):
    """Persistence protocol for Hippocampus episodic memories."""

    def save(self, memories: list[dict], *, namespace: str = "default") -> None: ...
    def load(self, *, namespace: str = "default") -> list[dict]: ...
    def query_similar(self, embedding: list[float], *, top_k: int = 5, namespace: str = "default") -> list[dict]: ...
    def query_by_time(self, start: float, end: float, *, namespace: str = "default") -> list[dict]: ...


@runtime_checkable
class CausalStore(Protocol):
    """Persistence protocol for NAc causal links."""

    def save(self, links: list[dict], *, namespace: str = "default") -> None: ...
    def load(self, *, namespace: str = "default") -> list[dict]: ...
    def query_by_event(self, event_sig: str, *, namespace: str = "default") -> list[dict]: ...


@runtime_checkable
class SemanticStore(Protocol):
    """Persistence protocol for ATL semantic concepts."""

    def save(self, concepts: list[dict], *, namespace: str = "default") -> None: ...
    def load(self, *, namespace: str = "default") -> list[dict]: ...
    def query_by_type(self, concept_type: str, *, namespace: str = "default") -> list[dict]: ...


# ---------------------------------------------------------------------------
# File-based implementations (current behavior, default)
# ---------------------------------------------------------------------------


def _wrap_items(kind: str, items: list[dict]) -> dict[str, Any]:
    """Wrap a list of items in the v1.0 file-format envelope.

    File*Store implementations historically wrote a bare JSON array at
    root, which has no slot for ``_format_version``. v1.0 wraps items
    in a ``{kind, items}`` dict so the version field can sit at root
    alongside the payload. ``_unwrap_items`` accepts both shapes for
    backwards compatibility.
    """
    from maxim.utils.format_version import with_format_version

    return with_format_version({"kind": kind, "items": items})


def _unwrap_items(data: Any, kind: str) -> list[dict]:
    """Read a File*Store payload, accepting both v1.0 dict and pre-1.0 list.

    A bare list at root indicates a pre-1.0 file. The version helper's
    ``check_format_version`` only handles dicts; we emit the warning
    by passing a synthetic empty dict so the helper's dedupe logic
    keys correctly on ``kind``.

    Loud-fail on truly unrecognized roots (CC1 review fold, arch I3):
    a JSON value that's neither list nor dict is corruption, not a
    valid file shape — return [] but log a warning so the caller sees
    silent data loss before it cascades.
    """
    from maxim.utils.format_version import check_format_version

    if isinstance(data, list):
        check_format_version({}, kind, log=log)
        return data

    if isinstance(data, dict):
        check_format_version(data, kind, log=log)
        items = data.get("items")
        if isinstance(items, list):
            return items
        return []

    log.warning(
        "%s payload has unexpected JSON root type %s (expected dict or list); treating as empty store",
        kind,
        type(data).__name__,
    )
    return []


class FileEpisodicStore:
    """JSON file persistence for Hippocampus (wraps current save/load)."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    def save(self, memories: list[dict], *, namespace: str = "default") -> None:
        from maxim.utils.atomic_io import atomic_write_json

        self._path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(str(self._path), _wrap_items("file_episodic_store", memories))

    def load(self, *, namespace: str = "default") -> list[dict]:
        if not self._path.exists():
            return []
        with open(self._path) as f:
            data = json.load(f)
        return _unwrap_items(data, "file_episodic_store")

    def query_similar(self, embedding: list[float], *, top_k: int = 5, namespace: str = "default") -> list[dict]:
        # File-based store doesn't support vector search — return empty
        return []

    def query_by_time(self, start: float, end: float, *, namespace: str = "default") -> list[dict]:
        memories = self.load(namespace=namespace)
        return [m for m in memories if start <= m.get("timestamp", 0) <= end]


class FileCausalStore:
    """JSON file persistence for NAc causal links (wraps current save/load)."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    def save(self, links: list[dict], *, namespace: str = "default") -> None:
        from maxim.utils.atomic_io import atomic_write_json

        self._path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(str(self._path), _wrap_items("file_causal_store", links))

    def load(self, *, namespace: str = "default") -> list[dict]:
        if not self._path.exists():
            return []
        with open(self._path) as f:
            data = json.load(f)
        return _unwrap_items(data, "file_causal_store")

    def query_by_event(self, event_sig: str, *, namespace: str = "default") -> list[dict]:
        links = self.load(namespace=namespace)
        return [link for link in links if link.get("event_signature") == event_sig]


class FileSemanticStore:
    """JSON file persistence for ATL semantic concepts (wraps current save/load)."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    def save(self, concepts: list[dict], *, namespace: str = "default") -> None:
        from maxim.utils.atomic_io import atomic_write_json

        self._path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(str(self._path), _wrap_items("file_semantic_store", concepts))

    def load(self, *, namespace: str = "default") -> list[dict]:
        if not self._path.exists():
            return []
        with open(self._path) as f:
            data = json.load(f)
        return _unwrap_items(data, "file_semantic_store")

    def query_by_type(self, concept_type: str, *, namespace: str = "default") -> list[dict]:
        concepts = self.load(namespace=namespace)
        return [c for c in concepts if c.get("category") == concept_type]
