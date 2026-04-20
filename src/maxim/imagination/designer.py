"""ImaginationDesigner — wraps EntityDesigner for real-time entity design.

Lighter than FoundryRunner: schema + sensor validation only, NO gauntlet.
Energy-gated via the global EnergyRegistry.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from maxim.embodiment.component_index import ComponentIndex
    from maxim.simulation.entity_designer import EntityDesigner

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class DesignResult:
    """Output of a successful imagination design.

    Attributes:
        ref: Generated component ref (e.g., "creatures/giant_spider").
        spec: Valid entity spec dict.
        synonyms: Alternative names for ComponentIndex aliases.
        validation_warnings: Non-fatal issues found during quick validation.
    """

    ref: str
    spec: dict[str, Any]
    synonyms: list[str]
    validation_warnings: tuple[str, ...]


class ImaginationDesigner:
    """Wraps EntityDesigner for real-time (non-foundry) entity design.

    Quick validation only — no gauntlet (too slow for real-time).
    Reuses ``validate_entity_spec()`` from entity_designer.py for
    schema + sensor sanity checks.
    """

    def __init__(
        self,
        entity_designer: EntityDesigner,
        component_index: ComponentIndex | None = None,
    ) -> None:
        self._designer = entity_designer
        self._index = component_index

    def imagine(
        self,
        description: str,
        scene_context: dict[str, Any] | None = None,
    ) -> DesignResult | None:
        """Design a new entity from a natural language description.

        Args:
            description: Entity phrase (e.g., "rusty iron gate").
            scene_context: Scene/campaign context for genre inference.

        Returns:
            DesignResult on success, None if design or validation fails.
        """
        # Infer genre from scene context if available
        genre = None
        if scene_context:
            genre = scene_context.get("genre") or scene_context.get("campaign_genre")

        # Infer entity_type from description
        entity_type = self._infer_entity_type(description)

        # Call EntityDesigner
        try:
            spec = self._designer.design(
                description=description,
                entity_type=entity_type,
                genre=genre,
            )
        except Exception as e:
            log.warning("ImaginationDesigner: design() failed for '%s': %s", description, e)
            return None

        if not spec:
            log.debug("ImaginationDesigner: design() returned empty spec for '%s'", description)
            return None

        # Quick validation — schema + sensor sanity (no gauntlet)
        from maxim.simulation.entity_designer import validate_entity_spec

        entity_spec = spec.get("entity", spec)
        errors = validate_entity_spec(entity_spec)
        if errors:
            log.warning(
                "ImaginationDesigner: validation failed for '%s': %s",
                description,
                "; ".join(errors),
            )
            return None

        # Additional sensor sanity (from foundry validation)
        warnings = self._sensor_sanity(entity_spec)

        # Extract synonyms from the spec (EntityDesigner prompt generates them)
        synonyms = self._extract_synonyms(spec, description)

        # Build ref from entity name + type
        name = entity_spec.get("name", self._slugify(description))
        category = entity_type or "misc"
        ref = f"{category}/{name}"

        # Dedup check via index if available
        if self._index is not None:
            existing = self._index.dedup_check(spec, threshold=0.80)
            if existing is not None:
                log.debug(
                    "ImaginationDesigner: '%s' is near-duplicate of '%s' (score=%.2f), skipping",
                    description,
                    existing.ref,
                    existing.score,
                )
                return None

        return DesignResult(
            ref=ref,
            spec=spec,
            synonyms=synonyms,
            validation_warnings=tuple(warnings),
        )

    @staticmethod
    def _infer_entity_type(description: str) -> str | None:
        """Infer SEM entity type from description keywords."""
        desc_lower = description.lower()
        creature_words = {
            "wolf",
            "dragon",
            "spider",
            "snake",
            "bear",
            "goblin",
            "zombie",
            "skeleton",
            "ghost",
            "demon",
            "beast",
            "creature",
            "monster",
            "drone",
            "robot",
            "dog",
            "hound",
            "guard",
        }
        weapon_words = {
            "sword",
            "axe",
            "bow",
            "staff",
            "dagger",
            "spear",
            "mace",
            "hammer",
            "blade",
            "gun",
            "rifle",
            "pistol",
            "laser",
            "cannon",
        }
        item_words = {
            "potion",
            "scroll",
            "amulet",
            "ring",
            "gem",
            "crystal",
            "key",
            "flask",
            "vial",
            "lantern",
            "torch",
            "device",
            "gadget",
            "chip",
            "terminal",
            "console",
        }
        vehicle_words = {"cart", "wagon", "ship", "boat", "speeder", "hover", "vehicle", "bike", "motorcycle", "car"}
        env_words = {
            "door",
            "gate",
            "bridge",
            "lever",
            "switch",
            "trap",
            "pit",
            "altar",
            "shrine",
            "fountain",
            "well",
            "chest",
            "pillar",
            "statue",
            "portal",
            "pedestal",
            "mechanism",
        }
        npc_words = {
            "merchant",
            "blacksmith",
            "healer",
            "priest",
            "mage",
            "wizard",
            "warrior",
            "knight",
            "thief",
            "rogue",
            "assassin",
            "archer",
            "captain",
            "commander",
            "villager",
            "innkeeper",
        }

        words = desc_lower.split()
        for word in words:
            stem = word.rstrip("s")
            if word in creature_words or stem in creature_words:
                return "creatures"
            if word in weapon_words or stem in weapon_words:
                return "weapons"
            if word in item_words or stem in item_words:
                return "items"
            if word in vehicle_words or stem in vehicle_words:
                return "vehicles"
            if word in npc_words or stem in npc_words:
                return "npcs"
            if word in env_words or stem in env_words:
                return "environments"
        return None

    @staticmethod
    def _extract_synonyms(spec: dict, description: str) -> list[str]:
        """Extract synonyms from spec + add the original description."""
        synonyms: list[str] = []

        # From spec (EntityDesigner generates these)
        entity_spec = spec.get("entity", spec)
        component = spec.get("component", {})

        for source in [entity_spec.get("synonyms", []), component.get("synonyms", [])]:
            if isinstance(source, list):
                synonyms.extend(str(s) for s in source if s)

        # Always include the original description as an alias
        if description.lower() not in [s.lower() for s in synonyms]:
            synonyms.append(description.lower())

        return synonyms

    @staticmethod
    def _sensor_sanity(entity_spec: dict) -> list[str]:
        """Quick sensor sanity checks (subset of foundry validation)."""
        warnings: list[str] = []
        sensors = entity_spec.get("sensors", {})
        for name, sensor in sensors.items():
            if not isinstance(sensor, dict):
                continue
            rng = sensor.get("range", [0, 1])
            if isinstance(rng, (list, tuple)) and len(rng) == 2:
                lo, hi = rng
                if isinstance(lo, (int, float)) and isinstance(hi, (int, float)):
                    if hi - lo == 0:
                        warnings.append(f"sensor '{name}' has zero-width range")
                    elif hi - lo > 10000:
                        warnings.append(f"sensor '{name}' has very large range span ({hi - lo})")
            initial = sensor.get("initial")
            if initial is not None and isinstance(rng, (list, tuple)) and len(rng) == 2:
                lo, hi = rng
                if isinstance(initial, (int, float)) and isinstance(lo, (int, float)) and isinstance(hi, (int, float)):
                    if initial < lo or initial > hi:
                        warnings.append(f"sensor '{name}' initial value {initial} outside range [{lo}, {hi}]")
        return warnings

    @staticmethod
    def _slugify(text: str) -> str:
        """Convert text to a valid component name slug."""
        slug = re.sub(r"[^a-z0-9]+", "_", text.lower().strip())
        slug = slug.strip("_")
        return slug or "unknown_entity"
