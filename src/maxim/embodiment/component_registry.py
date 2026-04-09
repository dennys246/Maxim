"""SEM Component Registry — template catalog for reusable entity specs.

Discovers, indexes, and resolves SEM component YAML files from multiple
search paths.  Components are stored as raw spec dicts (templates), NOT
live Entity objects.  Callers use :meth:`instantiate` to get a fresh,
independent Entity instance from a template.

Search path priority (highest to lowest):
1. Campaign-local directory (same folder as the campaign YAML)
2. User components (``~/.maxim/components/``)
3. Bundled components (``src/maxim/_data/components/``)
4. Legacy path (``scenarios/embodiment/``, backward compat)

Example::

    registry = ComponentRegistry()
    guard_spec = registry.get("npcs/guard")   # resolved dict (with extends)
    guard = registry.instantiate("npcs/guard") # fresh Entity instance

    # With per-instance overrides:
    captain = registry.instantiate(
        "npcs/guard",
        name="captain_aldric",
        sensors={"hp": {"initial": 25}},
    )
"""

from __future__ import annotations

import copy
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ComponentInfo:
    """Lightweight metadata returned by :meth:`ComponentRegistry.query`.

    Built from the ``component:`` header without loading the full spec.
    """

    ref: str  # "weapons/rusty_sword"
    name: str  # "rusty_sword"
    category: str  # "weapons"
    tags: tuple[str, ...]
    extends: str | None  # Parent ref or None
    source_path: str  # Absolute path to the YAML file


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base*.

    - Dict values are merged recursively (leaf-level override).
    - All other values (lists, scalars) are replaced entirely.
    - Neither input is mutated; returns a new dict.
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _read_component_header(path: Path) -> dict[str, Any] | None:
    """Read only the ``component:`` header from a YAML file.

    Returns the header dict, or None if the file doesn't have one.
    Falls back to auto-detection for legacy files (``body:`` or
    ``world_entities:`` without a ``component:`` header).
    """
    try:
        with open(path) as f:
            raw = yaml.safe_load(f)
    except Exception:
        return None

    if not isinstance(raw, dict):
        return None

    # Explicit component header
    if "component" in raw and isinstance(raw["component"], dict):
        return raw["component"]

    # Legacy auto-detection: files with body: or world_entities: but no component:
    if "body" in raw or "world_entities" in raw:
        name = raw.get("name", path.stem)
        category = path.parent.name if path.parent.name not in (".", "") else "bodies"
        return {
            "name": name,
            "category": category,
            "tags": [],
            "extends": None,
            "_legacy": True,
        }

    return None


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class ComponentRegistry:
    """Discovers, indexes, and resolves SEM component templates.

    Stores raw YAML spec dicts — NOT Entity objects.  Callers instantiate
    entities via :meth:`instantiate` to get independent instances with
    separate sensor state.
    """

    def __init__(
        self,
        search_paths: list[Path] | None = None,
        campaign_dir: Path | None = None,
        include_defaults: bool = True,
    ) -> None:
        self._index: dict[str, ComponentInfo] = {}  # ref -> info
        self._file_map: dict[str, Path] = {}  # ref -> YAML path
        self._spec_cache: dict[str, dict] = {}  # ref -> resolved spec
        self._lock = threading.Lock()

        if search_paths and not include_defaults:
            paths = [Path(p) for p in search_paths]
        elif search_paths:
            paths = [Path(p) for p in search_paths] + self._default_search_paths(campaign_dir)
        else:
            paths = self._default_search_paths(campaign_dir)

        self._search_paths = paths
        self._scan_all()

    # -- public API ---------------------------------------------------------

    def get(self, ref: str) -> dict:
        """Return the resolved entity spec dict for a component ref.

        If the component has ``extends``, deep-merges the parent spec
        first.  Result is cached after first resolution.

        Returns a **deep copy** of the cached dict so callers can
        mutate freely (e.g., override ``name`` for a specific instance).

        Raises
        ------
        KeyError
            If *ref* is not found in any search path.
        """
        if ref not in self._index:
            raise KeyError(f"Component not found: '{ref}'. Available: {sorted(self._index)}")

        with self._lock:
            if ref not in self._spec_cache:
                self._spec_cache[ref] = self._resolve(ref, _visited=set())

        return copy.deepcopy(self._spec_cache[ref])

    def instantiate(self, ref: str, **overrides: Any) -> Any:
        """Convenience: ``get()`` + ``_parse_entity()`` in one call.

        Creates a fresh, independent Entity instance from the template.
        Optional *overrides* are deep-merged into the entity spec before
        parsing (e.g., ``name="captain"``, ``sensors={"hp": {"initial": 25}}``).

        Returns
        -------
        Entity
            A fully parsed Entity with SpecSensor/SpecModulator stubs.
        """
        spec = self.get(ref)
        entity_spec = spec.get("entity", spec)

        if overrides:
            entity_spec = deep_merge(entity_spec, overrides)

        from maxim.embodiment.spec import _parse_entity
        return _parse_entity(entity_spec)

    def query(
        self,
        *,
        tags: list[str] | None = None,
        category: str | None = None,
        genre: str | None = None,
        has_sensor: str | None = None,
        has_affordance: str | None = None,
    ) -> list[ComponentInfo]:
        """Search components by metadata.  All filters are AND-combined.

        Args:
            tags: Require all of these tags.
            category: Require this category.
            genre: If set, only return components whose tags include this
                genre (e.g., ``"fantasy"``, ``"cyberpunk"``).  Components
                without any genre tag are considered genre-neutral and
                always included.
            has_sensor: Require a sensor with this name.
            has_affordance: Require an affordance with this name.
        """
        # Known genre tags — used to determine if a component is genre-tagged.
        _GENRE_TAGS = {"fantasy", "cyberpunk", "scifi", "modern", "devops", "horror", "historical"}

        results = []
        for info in self._index.values():
            if category and info.category != category:
                continue
            if tags and not all(t in info.tags for t in tags):
                continue
            # Genre filter: if a genre is requested, exclude components that
            # carry a *different* genre tag.  Components with no genre tag
            # at all (genre-neutral) always pass.
            if genre:
                component_genres = set(info.tags) & _GENRE_TAGS
                if component_genres and genre not in component_genres:
                    continue
            # Sensor/affordance filters require loading the full spec
            if has_sensor or has_affordance:
                try:
                    spec = self.get(info.ref)
                except KeyError:
                    continue
                entity_spec = spec.get("entity", spec)
                if has_sensor and has_sensor not in entity_spec.get("sensors", {}):
                    continue
                if has_affordance:
                    found = False
                    for mod in entity_spec.get("modulators", {}).values():
                        if has_affordance in mod.get("affordances", {}):
                            found = True
                            break
                    if not found:
                        continue
            results.append(info)
        return sorted(results, key=lambda i: i.ref)

    def list_categories(self) -> list[str]:
        """Return sorted list of available categories."""
        return sorted({info.category for info in self._index.values()})

    def list_refs(self, category: str | None = None) -> list[str]:
        """Return all known refs, optionally filtered by category."""
        refs = [
            info.ref for info in self._index.values()
            if category is None or info.category == category
        ]
        return sorted(refs)

    def has(self, ref: str) -> bool:
        """Check if a ref exists without loading the full spec."""
        return ref in self._index

    def register(self, ref: str, spec: dict, source_path: str = "<inline>") -> None:
        """Manually register a component spec (e.g., from campaign YAML)."""
        header = spec.get("component", {})
        info = ComponentInfo(
            ref=ref,
            name=header.get("name", ref.rsplit("/", 1)[-1]),
            category=header.get("category", ref.rsplit("/", 1)[0] if "/" in ref else "misc"),
            tags=tuple(header.get("tags", ())),
            extends=header.get("extends"),
            source_path=source_path,
        )
        with self._lock:
            self._index[ref] = info
            self._spec_cache[ref] = spec
            self._file_map[ref] = Path(source_path)

    # -- private ------------------------------------------------------------

    @staticmethod
    def _default_search_paths(campaign_dir: Path | None = None) -> list[Path]:
        """Build the default search path list (highest priority first)."""
        paths: list[Path] = []

        # 1. Campaign-local
        if campaign_dir is not None:
            paths.append(Path(campaign_dir))

        # 2. User components
        try:
            from maxim.utils.paths import data_home
            user_components = data_home() / "components"
            if user_components.is_dir():
                paths.append(user_components)
        except Exception:
            pass

        # 3. Bundled components
        try:
            from maxim.utils.paths import bundled_data
            bundled = bundled_data() / "components"
            if bundled.is_dir():
                paths.append(bundled)
        except Exception:
            pass

        # 4. Legacy paths (repo checkout)
        legacy = Path("scenarios") / "embodiment"
        if legacy.is_dir():
            paths.append(legacy)

        return paths

    def _scan_all(self) -> None:
        """Scan all search paths and build the component index.

        Higher-priority paths shadow lower-priority ones (same ref).
        """
        # Scan in reverse order so higher-priority paths overwrite
        for search_dir in reversed(self._search_paths):
            if not search_dir.is_dir():
                continue
            for yaml_path in sorted(search_dir.rglob("*.yaml")):
                header = _read_component_header(yaml_path)
                if header is None:
                    continue

                name = header.get("name", yaml_path.stem)
                category = header.get("category", "")

                # Derive category from directory structure if not explicit
                if not category:
                    rel = yaml_path.relative_to(search_dir)
                    if len(rel.parts) > 1:
                        category = rel.parts[0]
                    else:
                        category = "misc"

                ref = f"{category}/{name}"
                tags = tuple(header.get("tags", ()))
                extends = header.get("extends")

                info = ComponentInfo(
                    ref=ref,
                    name=name,
                    category=category,
                    tags=tags,
                    extends=extends,
                    source_path=str(yaml_path.resolve()),
                )
                self._index[ref] = info
                self._file_map[ref] = yaml_path

        log.debug("ComponentRegistry indexed %d components from %d paths",
                  len(self._index), len(self._search_paths))

    def _load_full_spec(self, ref: str) -> dict:
        """Load and return the complete YAML spec for a ref."""
        path = self._file_map.get(ref)
        if path is None:
            raise KeyError(f"No file mapped for component '{ref}'")

        with open(path) as f:
            raw = yaml.safe_load(f)

        return raw

    def _resolve(self, ref: str, *, _visited: set[str]) -> dict:
        """Resolve a component ref, applying ``extends`` inheritance.

        Uses *_visited* to detect circular inheritance chains.
        """
        if ref in _visited:
            chain = " -> ".join(_visited) + f" -> {ref}"
            raise ValueError(f"Circular inheritance detected: {chain}")
        _visited.add(ref)

        spec = self._load_full_spec(ref)
        info = self._index[ref]

        if info.extends:
            if info.extends not in self._index:
                raise KeyError(
                    f"Component '{ref}' extends '{info.extends}' which was not found. "
                    f"Available: {sorted(self._index)}"
                )
            parent_spec = self._resolve(info.extends, _visited=_visited)

            # Deep-merge: parent entity spec + child entity spec
            parent_entity = parent_spec.get("entity", parent_spec)
            child_entity = spec.get("entity", spec)
            merged_entity = deep_merge(parent_entity, child_entity)

            # Preserve the component header from the child
            result = dict(spec)
            result["entity"] = merged_entity
            return result

        return spec
