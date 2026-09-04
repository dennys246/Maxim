"""EntityMap — name-to-live-Entity resolution with ownership tracking.

Standalone object decoupled from ToolRegistry so any layer (tools,
prompt builder, memory, Reachy runtime) can resolve entity names
without depending on the tools layer.

Thread-safe via RLock.  Populated by ``generate_tools_for_entity``
and ``ImaginationTrigger``.  Read by ``UniversalSenseTool`` and
``SenseToolsTool``.

Name collisions (e.g., two entities both named ``"guard"``) are
disambiguated by storing both under their ``full_path`` instead.

Ownership: entities are either **self** (the agent's body — tools
are callable) or **scene** (observed entities — no callable tools).
``register_self`` marks an entity tree as agent-controlled;
``register_scene`` (and the backward-compat ``register``) marks as
scene-only.  ``list_self_entities`` / ``list_scene_entities`` filter
by ownership.
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
        emap.register_self(humanoid_entity)    # agent body — tools callable
        emap.register_scene(dragon_entity)     # observed — no tools
        entity = emap.resolve("dragon")
        names = emap.list_names()
    """

    __slots__ = ("_entities", "_lock", "_self_ids")

    def __init__(self) -> None:
        self._entities: dict[str, Entity] = {}
        self._self_ids: set[int] = set()
        self._lock = threading.RLock()

    # -- Registration -------------------------------------------------------

    def register_self(self, entity: Entity) -> None:
        """Register the agent's own entity tree (body + carried items).

        All descendants are marked as self-owned.  Tools generated for
        self entities are callable by the agent.
        """
        with self._lock:
            for ent in entity.walk():
                self._self_ids.add(id(ent))
            self._register_locked(entity)

    def register_scene(self, entity: Entity) -> None:
        """Register a scene entity (observable, not controllable).

        The agent can sense scene entities but cannot call their
        affordances as tools.
        """
        with self._lock:
            self._register_locked(entity)

    def register(self, entity: Entity) -> None:
        """Backward-compat: defaults to ``register_scene``."""
        self.register_scene(entity)

    def _register_locked(self, entity: Entity) -> None:
        """Core registration logic.  Caller MUST hold ``self._lock``.

        Walks all descendants.  On name collision, both the existing
        and new entity are stored under their ``full_path``.
        """
        for ent in entity.walk():
            if self._entities.get(ent.name) is ent:
                # Re-registration of the SAME object is a no-op — the
                # collision branch below would pop the name key and strand
                # the entity under its full_path, silently killing every
                # name-based resolve afterwards (found via D77: acquisition
                # regenerates tools, which re-registers the item, which
                # evicted "minecraft_bread" and no-opped its target_effect).
                continue
            if ent.name in self._entities:
                existing = self._entities.pop(ent.name)
                # Only re-store under full_path if not already there
                if existing.full_path not in self._entities:
                    self._entities[existing.full_path] = existing
                self._entities[ent.full_path] = ent
            else:
                self._entities[ent.name] = ent

    # -- Resolution ---------------------------------------------------------

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

    # -- Ownership queries --------------------------------------------------

    def is_self(self, entity: Entity) -> bool:
        """Check whether an entity belongs to the agent's body."""
        with self._lock:
            return id(entity) in self._self_ids

    def list_self_entities(self) -> list[Entity]:
        """Return entities belonging to the agent (self-owned)."""
        with self._lock:
            return [e for e in self._deduplicated() if id(e) in self._self_ids]

    def list_scene_entities(self) -> list[Entity]:
        """Return scene entities (observable, not controllable)."""
        with self._lock:
            return [e for e in self._deduplicated() if id(e) not in self._self_ids]

    # -- Ownership transfer ---------------------------------------------------

    def transfer_to_self(self, entity: Entity) -> None:
        """Transfer a scene entity to self-owned (agent picked it up).

        Adds the entity and all descendants to self-ownership.
        Does NOT reparent — caller must call entity.reparent() first.
        """
        with self._lock:
            self._self_ids.add(id(entity))
            for child in entity.children:
                self._self_ids.add(id(child))

    def transfer_to_scene(self, entity: Entity) -> None:
        """Transfer a self-owned entity back to scene (agent dropped it).

        Removes the entity and all descendants from self-ownership.
        Does NOT reparent — caller must call entity.reparent() first.
        """
        with self._lock:
            self._self_ids.discard(id(entity))
            for child in entity.children:
                self._self_ids.discard(id(child))

    # -- Listing ------------------------------------------------------------

    def list_names(self) -> list[str]:
        """Return all known entity keys (for error messages and discovery)."""
        with self._lock:
            return list(self._entities.keys())

    def list_entities(self) -> list[Entity]:
        """Return all unique registered entities (self + scene)."""
        with self._lock:
            return list(self._deduplicated())

    def _deduplicated(self) -> list[Entity]:
        """Return unique entities.  Caller MUST hold ``self._lock``."""
        seen_ids: set[int] = set()
        result: list[Entity] = []
        for ent in self._entities.values():
            if id(ent) not in seen_ids:
                seen_ids.add(id(ent))
                result.append(ent)
        return result

    # -- Dunder -------------------------------------------------------------

    def __len__(self) -> int:
        with self._lock:
            return len(set(id(e) for e in self._entities.values()))

    def __contains__(self, name: str) -> bool:
        return self.resolve(name) is not None
