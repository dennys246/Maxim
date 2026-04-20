"""Entity Designer — LLM-driven SEM spec generation from natural language.

Converts descriptions like "a suspicious guard captain" into valid SEM
entity specs with sensors, modulators, failure modes, and affordances.
Uses ComponentRegistry templates as bases when available, generating
only the delta.

The designer's system prompt includes the SEM schema spec so the LLM
knows what valid output looks like.  Medium-tier LLM preferred for
quality; small-tier as fallback.

Example::

    designer = EntityDesigner(component_registry=registry)
    spec = designer.design(
        description="A suspicious guard captain, battle-hardened",
        base_ref="npcs/guard",
    )
    # spec is a valid entity dict ready for _parse_entity()
"""

from __future__ import annotations

import json
import logging
from typing import Any

log = logging.getLogger(__name__)

# System prompt teaching the LLM the SEM entity schema
_SEM_SCHEMA_PROMPT = """\
You are an Entity Designer for a bio-inspired cognitive architecture.
You generate valid SEM (Sensor-Entity-Modulator) entity specifications
from natural language descriptions.

OUTPUT FORMAT — return ONLY a JSON object with this structure:
{
  "name": "entity_name",
  "entity_type": "npc|weapon|creature|environment|item|vehicle",
  "metadata": {
    "persona_prompt": "personality description for NPCs",
    "race": "human|elf|dwarf|etc (for NPCs)",
    "class": "warrior|mage|rogue|etc (for NPCs)"
  },
  "sensors": {
    "sensor_name": {
      "unit": "points|ratio|count|kg|degrees",
      "range": [min, max],
      "initial": starting_value
    }
  },
  "modulators": {
    "modulator_group_name": {
      "affordances": {
        "action_name": {
          "params": {"param_name": "type (str|float|int|bool)"},
          "description": "what this action does"
        }
      }
    }
  },
  "failure_modes": [
    {
      "name": "failure_name",
      "trigger": {"field": "sensor_name", "op": "<|>|<=|>=|==", "value": threshold, "pain": 0.0-1.0}
    }
  ]
}

RULES:
- Every sensor MUST have unit, range, and initial
- Sensor ranges must be [min, max] with min < max
- initial must be within range
- Affordance params: use str, float, int, or bool types
- failure_modes are optional but recommended for interesting entities
- pain values: 0.0 = none, 0.5 = moderate, 1.0 = severe
- For NPCs: always include hp, and at least one social sensor (trust, mood, etc.)
- For weapons: always include durability
- Include a "synonyms" list of 5-10 alternative names or short phrases a user
  might use to refer to this entity (e.g. "rusty blade", "old sword", "dull knife")
- Return ONLY the JSON object, no markdown, no explanation
"""


class EntityDesigner:
    """Generates valid SEM entity specs from natural language descriptions.

    Uses an LLM to interpret descriptions and produce structured YAML-compatible
    entity specs.  When a ``component_registry`` is provided, the designer
    can start from a base template and generate only the overrides.

    Attributes:
        component_registry: Optional ComponentRegistry for base template lookup.
    """

    def __init__(
        self,
        component_registry: Any | None = None,
        llm_router: Any | None = None,
    ) -> None:
        self._registry = component_registry
        self._llm = llm_router

    def design(
        self,
        description: str,
        base_ref: str | None = None,
        entity_type: str | None = None,
        genre: str | None = None,
    ) -> dict[str, Any]:
        """Design an entity from a natural language description.

        Args:
            description: Natural language description (e.g., "a suspicious guard").
            base_ref: Optional ComponentRegistry ref to use as template.
            entity_type: Optional type hint (npc, weapon, creature, environment, item, vehicle).
            genre: Optional genre constraint. When set, only suggests base_refs
                whose tags include this genre (or have no genre tag at all).
                Prevents cross-genre contamination (e.g., cyberpunk drone in
                a fantasy campaign).

        Returns:
            Valid entity spec dict ready for ``_parse_entity()``.
            If LLM is available, uses it for generation.
            Otherwise, falls back to template + basic overrides.
        """
        # Start from base template if available
        base_spec: dict[str, Any] = {}
        if base_ref and self._registry:
            # Validate genre compatibility when both genre and base_ref are set
            if genre:
                try:
                    info = self._registry._index.get(base_ref)
                    if info is not None:
                        _genre_tags = {"fantasy", "cyberpunk", "scifi", "modern", "devops", "horror", "historical"}
                        component_genres = set(info.tags) & _genre_tags
                        if component_genres and genre not in component_genres:
                            log.warning(
                                "EntityDesigner: base ref '%s' is %s but campaign genre is '%s' — skipping",
                                base_ref,
                                "/".join(component_genres),
                                genre,
                            )
                            base_ref = None  # Fall through to fallback
                except Exception:
                    pass  # Best-effort genre check

            if base_ref:
                try:
                    full = self._registry.get(base_ref)
                    base_spec = full.get("entity", full)
                    log.debug("EntityDesigner: using base '%s'", base_ref)
                except KeyError:
                    log.warning("EntityDesigner: base ref '%s' not found, generating from scratch", base_ref)

        # Try LLM generation
        if self._llm:
            try:
                return self._design_with_llm(description, base_spec, entity_type)
            except Exception as e:
                log.warning("EntityDesigner: LLM generation failed: %s, using fallback", e)

        # Fallback: template-based generation
        return self._design_fallback(description, base_spec, entity_type)

    def design_npc(
        self,
        name: str,
        personality: str = "",
        race: str = "human",
        archetype: str = "commoner",
        base_ref: str | None = "npcs/base_humanoid",
    ) -> dict[str, Any]:
        """Convenience: design an NPC with common defaults.

        Args:
            name: NPC name.
            personality: Personality description for persona_prompt.
            race: Race (human, elf, dwarf, etc.).
            archetype: Class archetype (warrior, mage, rogue, etc.).
            base_ref: Component base to extend.

        Returns:
            Valid NPC entity spec dict.
        """
        description = f"A {race} {archetype} named {name}"
        if personality:
            description += f". Personality: {personality}"

        spec = self.design(description, base_ref=base_ref, entity_type="npc")

        # Ensure name and metadata are set
        spec["name"] = name
        spec.setdefault("metadata", {})
        spec["metadata"]["race"] = race
        spec["metadata"]["class"] = archetype
        if personality:
            spec["metadata"]["persona_prompt"] = personality

        return spec

    def design_item(
        self,
        name: str,
        description: str = "",
        base_ref: str | None = None,
    ) -> dict[str, Any]:
        """Convenience: design an item/weapon/equipment.

        Args:
            name: Item name.
            description: What the item is and does.
            base_ref: Optional component base (e.g., "weapons/rusty_sword").

        Returns:
            Valid item entity spec dict.
        """
        full_desc = f"An item called {name}"
        if description:
            full_desc += f": {description}"

        spec = self.design(full_desc, base_ref=base_ref, entity_type="weapon")
        spec["name"] = name
        return spec

    # -- LLM generation ----------------------------------------------------

    def _design_with_llm(
        self,
        description: str,
        base_spec: dict[str, Any],
        entity_type: str | None,
    ) -> dict[str, Any]:
        """Use LLM to generate an entity spec from description."""
        prompt_parts = [_SEM_SCHEMA_PROMPT]

        if base_spec:
            prompt_parts.append(f"\nBASE TEMPLATE (modify/extend this):\n{json.dumps(base_spec, indent=2)}")

        prompt_parts.append(f"\nDESCRIPTION: {description}")
        if entity_type:
            prompt_parts.append(f"ENTITY TYPE: {entity_type}")
        prompt_parts.append("\nGenerate the entity spec JSON:")

        system_prompt = "\n".join(prompt_parts)

        # Call LLM — prefer medium tier for quality
        result = self._llm.generate_json(
            system_prompt,
            max_tokens=1000,
        )

        if isinstance(result, dict):
            # Validate basic structure
            if "name" not in result and "sensors" not in result:
                raise ValueError("LLM output missing required fields (name or sensors)")

            # Merge with base if provided
            if base_spec:
                from maxim.embodiment.component_registry import deep_merge

                result = deep_merge(base_spec, result)

            return result
        raise ValueError(f"LLM returned non-dict: {type(result)}")

    # -- Fallback generation -----------------------------------------------

    def _design_fallback(
        self,
        description: str,
        base_spec: dict[str, Any],
        entity_type: str | None,
    ) -> dict[str, Any]:
        """Template-based fallback when LLM is unavailable.

        Starts from base_spec (if provided) and adds reasonable defaults
        based on entity_type and description keywords.
        """
        spec = dict(base_spec) if base_spec else {}

        # Extract name from description
        words = description.lower().split()
        name = spec.get("name", "_".join(words[:3]))
        spec["name"] = name
        spec["entity_type"] = entity_type or spec.get("entity_type", "generic")

        # Add default sensors based on entity type
        etype = spec["entity_type"]
        spec.setdefault("sensors", {})
        spec.setdefault("modulators", {})
        spec.setdefault("metadata", {})

        if etype in ("npc", "character", "humanoid"):
            spec["sensors"].setdefault("hp", {"unit": "points", "range": [0, 20], "initial": 20})
            spec["sensors"].setdefault("mood", {"unit": "ratio", "range": [-1, 1], "initial": 0.0})
            spec["modulators"].setdefault(
                "social",
                {
                    "affordances": {
                        "speak": {"params": {"message": "str"}, "description": "Say something"},
                    }
                },
            )

        elif etype in ("weapon", "item", "equipment"):
            spec["sensors"].setdefault("durability", {"unit": "ratio", "range": [0, 1], "initial": 0.8})
            spec["modulators"].setdefault(
                "use",
                {
                    "affordances": {
                        "use": {"params": {"target": "str"}, "description": "Use this item"},
                    }
                },
            )

        elif etype in ("creature", "animal", "monster"):
            spec["sensors"].setdefault("hp", {"unit": "points", "range": [0, 15], "initial": 15})
            spec["sensors"].setdefault("aggression", {"unit": "ratio", "range": [0, 1], "initial": 0.5})
            spec["modulators"].setdefault(
                "combat",
                {
                    "affordances": {
                        "attack": {"params": {"target": "str"}, "description": "Attack a target"},
                    }
                },
            )

        elif etype in ("environment", "location", "room"):
            spec["sensors"].setdefault("visibility", {"unit": "ratio", "range": [0, 1], "initial": 0.7})
            spec["modulators"].setdefault(
                "interaction",
                {
                    "affordances": {
                        "survey": {"params": {}, "description": "Survey the area"},
                    }
                },
            )

        elif etype in ("vehicle", "mount", "transport"):
            spec["sensors"].setdefault("speed", {"unit": "ratio", "range": [0, 1], "initial": 0.0})
            spec["sensors"].setdefault("fuel", {"unit": "ratio", "range": [0, 1], "initial": 0.8})
            spec["modulators"].setdefault(
                "movement",
                {
                    "affordances": {
                        "move": {"params": {"destination": "str"}, "description": "Travel toward a destination"},
                    }
                },
            )

        # Extract personality keywords from description
        if any(kw in description.lower() for kw in ("suspicious", "wary", "distrustful")):
            spec["sensors"].setdefault("trust", {"unit": "ratio", "range": [0, 1], "initial": 0.2})
        if any(kw in description.lower() for kw in ("friendly", "warm", "welcoming")):
            spec["sensors"].setdefault("trust", {"unit": "ratio", "range": [0, 1], "initial": 0.8})
        if any(kw in description.lower() for kw in ("battle", "hardened", "veteran", "warrior")):
            if "hp" in spec["sensors"]:
                # Boost both range max and initial for battle-hardened entities
                spec["sensors"]["hp"]["range"] = [
                    spec["sensors"]["hp"]["range"][0],
                    spec["sensors"]["hp"]["range"][1] + 5,
                ]
                spec["sensors"]["hp"]["initial"] = spec["sensors"]["hp"].get("initial", 20) + 5

        return spec


def validate_entity_spec(spec: dict[str, Any]) -> list[str]:
    """Validate an entity spec dict against SEM schema rules.

    Returns list of error messages (empty = valid).
    """
    errors: list[str] = []

    if "name" not in spec:
        errors.append("Missing 'name' field")

    for sensor_name, sensor_spec in spec.get("sensors", {}).items():
        if not isinstance(sensor_spec, dict):
            errors.append(f"Sensor '{sensor_name}': must be a dict")
            continue
        if "unit" not in sensor_spec:
            errors.append(f"Sensor '{sensor_name}': missing 'unit'")
        if "range" not in sensor_spec:
            errors.append(f"Sensor '{sensor_name}': missing 'range'")
        elif not isinstance(sensor_spec["range"], (list, tuple)) or len(sensor_spec["range"]) != 2:
            errors.append(f"Sensor '{sensor_name}': 'range' must be [min, max]")
        elif sensor_spec["range"][0] >= sensor_spec["range"][1]:
            errors.append(f"Sensor '{sensor_name}': range min must be < max")

        initial = sensor_spec.get("initial")
        if initial is not None and "range" in sensor_spec:
            lo, hi = sensor_spec["range"]
            if not (lo <= initial <= hi):
                errors.append(f"Sensor '{sensor_name}': initial {initial} outside range [{lo}, {hi}]")

    for mod_name, mod_spec in spec.get("modulators", {}).items():
        if not isinstance(mod_spec, dict):
            errors.append(f"Modulator '{mod_name}': must be a dict")
            continue
        for aff_name, aff_spec in mod_spec.get("affordances", {}).items():
            if "description" not in aff_spec:
                errors.append(f"Affordance '{mod_name}.{aff_name}': missing 'description'")

    return errors
