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
from typing import Protocol, runtime_checkable

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


class FileEpisodicStore:
    """JSON file persistence for Hippocampus (wraps current save/load)."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    def save(self, memories: list[dict], *, namespace: str = "default") -> None:
        from maxim.utils.atomic_io import atomic_write_json
        self._path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(str(self._path), memories)

    def load(self, *, namespace: str = "default") -> list[dict]:
        if not self._path.exists():
            return []
        with open(self._path) as f:
            data = json.load(f)
        return data if isinstance(data, list) else []

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
        atomic_write_json(str(self._path), links)

    def load(self, *, namespace: str = "default") -> list[dict]:
        if not self._path.exists():
            return []
        with open(self._path) as f:
            data = json.load(f)
        return data if isinstance(data, list) else []

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
        atomic_write_json(str(self._path), concepts)

    def load(self, *, namespace: str = "default") -> list[dict]:
        if not self._path.exists():
            return []
        with open(self._path) as f:
            data = json.load(f)
        return data if isinstance(data, list) else []

    def query_by_type(self, concept_type: str, *, namespace: str = "default") -> list[dict]:
        concepts = self.load(namespace=namespace)
        return [c for c in concepts if c.get("category") == concept_type]
