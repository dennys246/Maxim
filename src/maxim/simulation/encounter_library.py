"""DM Encounter Library — reusable encounter templates for campaigns.

Discovers, indexes, and serves encounter templates from multiple search
paths.  Templates store the campaign-independent parts of an encounter
(scene prose, choices, dice mechanics).  Campaign YAML references templates
via ``template: "combat/forest_ambush"`` and adds campaign-specific wiring
(active_npcs, branches, on_choice, dialogue_hints) inline.

Search path priority (same as ComponentRegistry):
1. Campaign-local directory
2. User encounters (``~/.maxim/encounters/``)
3. Bundled encounters (``src/maxim/_data/encounters/``)

Example::

    library = EncounterLibrary()
    info_list = library.query(tags=["combat"], difficulty=3)
    template = library.get("combat/forest_ambush")
    # template = {"encounter": {"scene": "...", "choices": [...], ...}}
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
class EncounterInfo:
    """Lightweight metadata returned by :meth:`EncounterLibrary.query`."""

    ref: str  # "combat/forest_ambush"
    name: str  # "forest_ambush"
    category: str  # "combat"
    tags: tuple[str, ...]
    difficulty_range: tuple[int, int] | None  # (min, max) or None
    narrative_role: str | None  # "rising_action", "climax", etc.
    suggested_npcs: int  # Hint for architect
    source_path: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_encounter_header(path: Path) -> dict[str, Any] | None:
    """Read the ``encounter:`` header from a YAML file.

    Returns the header dict, or None if the file isn't an encounter template.
    """
    try:
        with open(path) as f:
            raw = yaml.safe_load(f)
    except Exception:
        return None

    if not isinstance(raw, dict):
        return None

    enc = raw.get("encounter")
    if not isinstance(enc, dict):
        return None

    # Must have at least a name or scene to be considered a template
    if "name" not in enc and "scene" not in enc:
        return None

    return enc


# ---------------------------------------------------------------------------
# Library
# ---------------------------------------------------------------------------


class EncounterLibrary:
    """Discovers and indexes reusable encounter templates.

    Templates store campaign-independent encounter parts (scene, choices,
    dice).  Campaigns reference them via ``template:`` key and add
    campaign-specific wiring on top.
    """

    def __init__(
        self,
        search_paths: list[Path] | None = None,
        campaign_dir: Path | None = None,
        include_defaults: bool = True,
    ) -> None:
        self._index: dict[str, EncounterInfo] = {}
        self._file_map: dict[str, Path] = {}
        self._cache: dict[str, dict] = {}
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
        """Return the full encounter template dict (deep copy).

        The returned dict has an ``encounter`` key containing the template
        fields (scene, choices, dice, tags, etc.).

        Raises KeyError if *ref* is not found.
        """
        if ref not in self._index:
            raise KeyError(
                f"Encounter template not found: '{ref}'. "
                f"Available: {sorted(self._index)}"
            )

        with self._lock:
            if ref not in self._cache:
                self._cache[ref] = self._load_full(ref)

        return copy.deepcopy(self._cache[ref])

    def query(
        self,
        *,
        tags: list[str] | None = None,
        category: str | None = None,
        difficulty: int | None = None,
        narrative_role: str | None = None,
    ) -> list[EncounterInfo]:
        """Search encounters by metadata.  All filters are AND-combined.

        If *difficulty* is provided, matches encounters whose
        ``difficulty_range`` contains the value.
        """
        results = []
        for info in self._index.values():
            if category and info.category != category:
                continue
            if tags and not all(t in info.tags for t in tags):
                continue
            if narrative_role and info.narrative_role != narrative_role:
                continue
            if difficulty is not None and info.difficulty_range:
                lo, hi = info.difficulty_range
                if not (lo <= difficulty <= hi):
                    continue
            results.append(info)
        return sorted(results, key=lambda i: i.ref)

    def list_categories(self) -> list[str]:
        """Return sorted list of available categories."""
        return sorted({info.category for info in self._index.values()})

    def list_refs(self, category: str | None = None) -> list[str]:
        """Return all known refs, optionally filtered by category."""
        return sorted(
            info.ref for info in self._index.values()
            if category is None or info.category == category
        )

    def has(self, ref: str) -> bool:
        """Check if a ref exists without loading the full template."""
        return ref in self._index

    # -- private ------------------------------------------------------------

    @staticmethod
    def _default_search_paths(campaign_dir: Path | None = None) -> list[Path]:
        paths: list[Path] = []

        if campaign_dir is not None:
            paths.append(Path(campaign_dir))

        try:
            from maxim.utils.paths import data_home
            user_encounters = data_home() / "encounters"
            if user_encounters.is_dir():
                paths.append(user_encounters)
        except Exception:
            pass

        try:
            from maxim.utils.paths import bundled_data
            bundled = bundled_data() / "encounters"
            if bundled.is_dir():
                paths.append(bundled)
        except Exception:
            pass

        return paths

    def _scan_all(self) -> None:
        """Scan search paths and build the encounter index."""
        for search_dir in reversed(self._search_paths):
            if not search_dir.is_dir():
                continue
            for yaml_path in sorted(search_dir.rglob("*.yaml")):
                header = _read_encounter_header(yaml_path)
                if header is None:
                    continue

                name = header.get("name", yaml_path.stem)
                category = header.get("category", "")

                if not category:
                    rel = yaml_path.relative_to(search_dir)
                    if len(rel.parts) > 1:
                        category = rel.parts[0]
                    else:
                        category = "misc"

                ref = f"{category}/{name}"
                tags = tuple(header.get("tags", ()))

                diff_raw = header.get("difficulty_range")
                difficulty_range = tuple(diff_raw) if diff_raw and len(diff_raw) == 2 else None

                info = EncounterInfo(
                    ref=ref,
                    name=name,
                    category=category,
                    tags=tags,
                    difficulty_range=difficulty_range,
                    narrative_role=header.get("narrative_role"),
                    suggested_npcs=header.get("suggested_npcs", 0),
                    source_path=str(yaml_path.resolve()),
                )
                self._index[ref] = info
                self._file_map[ref] = yaml_path

        log.debug("EncounterLibrary indexed %d encounters from %d paths",
                  len(self._index), len(self._search_paths))

    def _load_full(self, ref: str) -> dict:
        path = self._file_map[ref]
        with open(path) as f:
            return yaml.safe_load(f)
