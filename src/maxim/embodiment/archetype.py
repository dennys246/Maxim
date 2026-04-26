"""Body archetype loading and validation.

Archetypes are reusable body-structure templates stored in
``_data/components/archetypes/*.yaml``.  They define the vocabulary
of body parts (required + optional) that a component of that archetype
type should use.

The module is read-only — no runtime mutation, no state.  Used by:

- ``ImaginationDesigner`` to scaffold generated entity specs
- Validation (CI / pytest) to check seed component modulators
- ``format_archetype_prompt()`` to embed archetype descriptions in
  LLM design prompts

Archetype YAML format::

    archetype:
      name: quadruped
      description: "Four-legged creature..."
      body_parts:
        head:
          sensors: [skull_integrity, jaw_strength]
          affordances: [bite, roar]
        ...
      optional_parts:
        wings:
          sensors: [membrane_integrity]
          affordances: [take_flight, dive_attack]
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BodyPartTemplate:
    """Template for a single body part (modulator)."""

    sensors: tuple[str, ...]
    affordances: tuple[str, ...]
    abstract: bool = False


@dataclass(frozen=True)
class Archetype:
    """A body archetype — named template with required + optional parts."""

    name: str
    description: str
    body_parts: dict[str, BodyPartTemplate]  # required parts
    optional_parts: dict[str, BodyPartTemplate] = field(default_factory=dict)

    @property
    def all_part_names(self) -> frozenset[str]:
        """All valid modulator names (required + optional)."""
        return frozenset(self.body_parts) | frozenset(self.optional_parts)

    @property
    def all_affordance_names(self) -> frozenset[str]:
        """All affordances across all parts (required + optional)."""
        names: set[str] = set()
        for part in (*self.body_parts.values(), *self.optional_parts.values()):
            names.update(part.affordances)
        return frozenset(names)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

_ARCHETYPES_DIR = Path(__file__).resolve().parent.parent / "_data" / "components" / "archetypes"

# Module-level cache — loaded once, never mutated after.
_cache: dict[str, Archetype] | None = None


def _parse_body_part(name: str, raw: dict[str, Any]) -> BodyPartTemplate:
    """Parse a body part definition from YAML."""
    sensors = tuple(raw.get("sensors", []))
    affordances = tuple(raw.get("affordances", []))
    abstract = bool(raw.get("abstract", False))
    return BodyPartTemplate(sensors=sensors, affordances=affordances, abstract=abstract)


def _parse_archetype(raw: dict[str, Any]) -> Archetype:
    """Parse an archetype definition from YAML."""
    arch = raw.get("archetype", raw)
    name = arch["name"]
    description = arch.get("description", "").strip()

    body_parts: dict[str, BodyPartTemplate] = {}
    for part_name, part_raw in arch.get("body_parts", {}).items():
        body_parts[part_name] = _parse_body_part(part_name, part_raw)

    optional_parts: dict[str, BodyPartTemplate] = {}
    for part_name, part_raw in arch.get("optional_parts", {}).items():
        optional_parts[part_name] = _parse_body_part(part_name, part_raw)

    return Archetype(
        name=name,
        description=description,
        body_parts=body_parts,
        optional_parts=optional_parts,
    )


def load_archetypes(directory: Path | str | None = None) -> dict[str, Archetype]:
    """Load all archetype definitions from YAML files.

    Returns a dict mapping archetype name to Archetype.
    Cached after first call (module-level, read-only).
    """
    global _cache
    if _cache is not None:
        return _cache

    d = Path(directory) if directory is not None else _ARCHETYPES_DIR
    archetypes: dict[str, Archetype] = {}

    if not d.exists():
        log.debug("Archetypes directory not found: %s", d)
        _cache = archetypes
        return archetypes

    for yaml_file in sorted(d.glob("*.yaml")):
        try:
            with open(yaml_file) as f:
                raw = yaml.safe_load(f)
            if raw is None:
                continue
            arch = _parse_archetype(raw)
            archetypes[arch.name] = arch
        except Exception as e:
            log.warning("Failed to load archetype from %s: %s", yaml_file, e)

    _cache = archetypes
    log.debug("Loaded %d archetypes: %s", len(archetypes), list(archetypes.keys()))
    return archetypes


def get_archetype(name: str) -> Archetype | None:
    """Get a specific archetype by name. Returns None if not found."""
    return load_archetypes().get(name)


def validate_component_archetype(
    component_spec: dict[str, Any],
    entity_spec: dict[str, Any],
) -> list[str]:
    """Validate that a component's modulators match its archetype.

    Returns list of warning strings (empty = valid).
    Non-fatal: components without an archetype field always pass.
    """
    archetype_name = component_spec.get("archetype")
    if not archetype_name:
        return []  # No archetype declared — no validation

    arch = get_archetype(archetype_name)
    if arch is None:
        return [f"Unknown archetype '{archetype_name}'"]

    warnings: list[str] = []
    modulators = entity_spec.get("modulators", {})
    valid_names = arch.all_part_names

    for mod_name in modulators:
        if mod_name not in valid_names:
            # Check if it's a reasonable custom part (not in vocabulary)
            warnings.append(
                f"Modulator '{mod_name}' not in archetype '{archetype_name}' vocabulary (valid: {sorted(valid_names)})"
            )

    return warnings


# ---------------------------------------------------------------------------
# LLM prompt formatting
# ---------------------------------------------------------------------------


def format_archetype_prompt() -> str:
    """Format all archetypes as a prompt section for the entity designer LLM.

    Returns a text block listing each archetype with its description,
    required body parts, and optional parts. Used in the LLM prompt
    to guide generated entity structure.
    """
    archetypes = load_archetypes()
    if not archetypes:
        return ""

    lines = ["BODY ARCHETYPES — select one for this entity:\n"]
    for arch in archetypes.values():
        lines.append(f"  {arch.name}: {arch.description}")
        parts_str = ", ".join(arch.body_parts.keys())
        lines.append(f"    Required parts: {parts_str}")
        if arch.optional_parts:
            opt_str = ", ".join(arch.optional_parts.keys())
            lines.append(f"    Optional parts: {opt_str}")
        lines.append("")

    return "\n".join(lines)


def archetype_scaffold(archetype_name: str) -> dict[str, Any] | None:
    """Generate a scaffold entity spec from an archetype.

    Returns a minimal entity spec dict with all required body parts
    as modulators (empty affordances/sensors), ready for the LLM to
    customize. Returns None if archetype not found.
    """
    arch = get_archetype(archetype_name)
    if arch is None:
        return None

    modulators: dict[str, Any] = {}
    for part_name, part in arch.body_parts.items():
        mod: dict[str, Any] = {}
        if part.abstract:
            mod["abstract"] = True
        if part.sensors:
            sensors: dict[str, Any] = {}
            for s in part.sensors:
                sensors[s] = {"unit": "ratio", "range": [0, 1], "initial": 1.0}
            mod["sensors"] = sensors
        if part.affordances:
            affs: dict[str, Any] = {}
            for a in part.affordances:
                affs[a] = {"params": {}, "description": ""}
            mod["affordances"] = affs
        else:
            mod["affordances"] = {}
        modulators[part_name] = mod

    return {
        "name": "",
        "entity_type": "",
        "sensors": {},
        "modulators": modulators,
    }
