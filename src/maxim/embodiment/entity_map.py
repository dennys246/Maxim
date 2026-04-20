"""EntityMap — name-to-live-Entity resolution.

Standalone object decoupled from ToolRegistry so any layer (tools,
prompt builder, memory, Reachy runtime) can resolve entity names
without depending on the tools layer.

Thread-safe via RLock.  Populated by ``generate_tools_for_entity``
and ``ImaginationTrigger``.  Read by ``UniversalSenseTool`` and
``DiscoverToolsTool``.

Name collisions (e.g., two entities both named ``"guard"``) are
disambiguated by storing both under their ``full_path`` instead.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from maxim.embodiment.sem import Entity


class EntityMap:
    """Maps entity names/paths to live Entity objects.

    Example::

        emap = EntityMap()
        emap.register(rusty_sword_entity)   # walks descendants
        entity = emap.resolve("rusty_sword")
        names = emap.list_names()
    """

    __slots__ = ("_entities", "_lock")

    def __init__(self) -> None:
        self._entities: dict[str, Entity] = {}
        self._lock = threading.RLock()

    def register(self, entity: Entity) -> None:
        """Register an entity tree.  Walks all descendants.

        On name collision, both the existing and new entity are stored
        under their ``full_path`` to disambiguate.
        """
        with self._lock:
            for ent in entity.walk():
                if ent.name in self._entities:
                    existing = self._entities.pop(ent.name)
                    # Only re-store under full_path if not already there
                    if existing.full_path not in self._entities:
                        self._entities[existing.full_path] = existing
                    self._entities[ent.full_path] = ent
                else:
                    self._entities[ent.name] = ent

    def resolve(self, name: str) -> Entity | None:
        """Resolve by name, then by full_path.  Returns None if not found."""
        with self._lock:
            entity = self._entities.get(name)
            if entity is not None:
                return entity
            # Try case-insensitive + underscore/space normalization
            normalized = name.strip().lower().replace(" ", "_")
            for key, ent in self._entities.items():
                if key.lower().replace(" ", "_") == normalized:
                    return ent
            return None

    def list_names(self) -> list[str]:
        """Return all known entity keys (for error messages and discovery)."""
        with self._lock:
            return list(self._entities.keys())

    def list_entities(self) -> list[Entity]:
        """Return all unique registered entities."""
        with self._lock:
            # Deduplicate — an entity may appear under name AND full_path
            seen_ids: set[int] = set()
            result: list[Entity] = []
            for ent in self._entities.values():
                if id(ent) not in seen_ids:
                    seen_ids.add(id(ent))
                    result.append(ent)
            return result

    def __len__(self) -> int:
        with self._lock:
            return len(set(id(e) for e in self._entities.values()))

    def __contains__(self, name: str) -> bool:
        return self.resolve(name) is not None
