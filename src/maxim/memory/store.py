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
        "%s payload has unexpected JSON root type %s (expected dict or list); "
        "treating as empty store",
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
