"""DM-specific tools for campaign encounters.

The ChooseTool is dynamically updated per encounter to list valid choices.
It gives the AUT a structured way to select from offered options instead
of relying on free-text classification.
"""

from __future__ import annotations

import logging
from typing import Any

from maxim.tools.base import Tool, ToolOutput

log = logging.getLogger(__name__)


class ChooseTool(Tool):
    """Let the AUT pick from the current encounter's choices.

    The DM runtime updates ``valid_choices`` before each encounter.
    If the AUT calls choose with an invalid option, it gets an error
    listing the valid ones.

    Example:
        AUT calls: choose(option="accept_job")
        Returns: {"chosen": "accept_job", "valid_choices": ["accept_job", "decline", "negotiate_pay"]}
    """

    name = "choose"
    description = "Pick one of the available choices for the current encounter."
    input_schema: dict[str, Any] = {"option": str}
    timeout = 10.0

    def __init__(self) -> None:
        self.valid_choices: list[str] = []
        self._last_choice: str | None = None
        super().__init__()

    def set_choices(self, choices: list[str]) -> None:
        """Update valid choices for the current encounter."""
        self.valid_choices = [c.lower() for c in choices]
        self._last_choice = None
        # Update description to show current choices
        if self.valid_choices:
            opts = ", ".join(self.valid_choices)
            self.description = f"Pick one of the available choices: {opts}"
        else:
            self.description = "No choices available in current encounter."

    @property
    def last_choice(self) -> str | None:
        """The last valid choice made by the AUT."""
        return self._last_choice

    def execute(self, **kwargs: Any) -> Any:
        option = kwargs.get("option", "")
        if isinstance(option, str):
            option = option.strip().lower()

        if not self.valid_choices:
            return ToolOutput(
                success=False,
                error="No choices available in the current encounter.",
            )

        # Exact match
        if option in self.valid_choices:
            self._last_choice = option
            return {
                "chosen": option,
                "valid_choices": self.valid_choices,
                "success": True,
            }

        # Fuzzy match — underscore/space normalization
        option_normalized = option.replace(" ", "_")
        for choice in self.valid_choices:
            if choice.replace(" ", "_") == option_normalized:
                self._last_choice = choice
                return {
                    "chosen": choice,
                    "valid_choices": self.valid_choices,
                    "success": True,
                }

        # Partial match — choice keyword in option or vice versa
        for choice in self.valid_choices:
            if choice in option or option in choice:
                self._last_choice = choice
                return {
                    "chosen": choice,
                    "valid_choices": self.valid_choices,
                    "success": True,
                }

        # No match
        opts = ", ".join(self.valid_choices)
        return ToolOutput(
            success=False,
            error=f"Invalid choice '{option}'. Valid choices are: {opts}",
        )


# ---------------------------------------------------------------------------
# Architect tools — used by adventure_architect persona
# ---------------------------------------------------------------------------


class BrowseComponentsTool(Tool):
    """Browse the SEM Component Registry for entity templates."""

    name = "browse_components"
    description = "Search the component library for entity templates (NPCs, weapons, creatures, environments)."
    input_schema: dict[str, Any] = {
        "category": (str, None),
        "tags": (str, None),
    }

    def __init__(self, component_registry: Any = None) -> None:
        self._registry = component_registry
        super().__init__()

    def execute(self, **kwargs: Any) -> Any:
        if self._registry is None:
            return ToolOutput(success=False, error="No ComponentRegistry available")

        category = kwargs.get("category")
        tags_str = kwargs.get("tags", "")
        tags = [t.strip() for t in tags_str.split(",") if t.strip()] if tags_str else None

        results = self._registry.query(category=category, tags=tags)
        return {
            "count": len(results),
            "components": [
                {"ref": r.ref, "name": r.name, "category": r.category, "tags": list(r.tags)}
                for r in results[:20]
            ],
        }


class BrowseEncountersTool(Tool):
    """Browse the Encounter Library for scene templates."""

    name = "browse_encounters"
    description = "Search the encounter library for reusable scene templates."
    input_schema: dict[str, Any] = {
        "category": (str, None),
        "tags": (str, None),
        "difficulty": (int, None),
        "narrative_role": (str, None),
    }

    def __init__(self, encounter_library: Any = None) -> None:
        self._library = encounter_library
        super().__init__()

    def execute(self, **kwargs: Any) -> Any:
        if self._library is None:
            return ToolOutput(success=False, error="No EncounterLibrary available")

        category = kwargs.get("category")
        tags_str = kwargs.get("tags", "")
        tags = [t.strip() for t in tags_str.split(",") if t.strip()] if tags_str else None
        difficulty = kwargs.get("difficulty")
        narrative_role = kwargs.get("narrative_role")

        results = self._library.query(
            category=category, tags=tags,
            difficulty=difficulty, narrative_role=narrative_role,
        )
        return {
            "count": len(results),
            "encounters": [
                {
                    "ref": r.ref, "name": r.name, "category": r.category,
                    "tags": list(r.tags), "difficulty_range": r.difficulty_range,
                    "narrative_role": r.narrative_role,
                }
                for r in results[:20]
            ],
        }


class DesignEntityTool(Tool):
    """Generate an NPC/item/creature from a natural language description."""

    name = "design_entity"
    description = (
        "Design an entity from a description. Returns a valid SEM entity spec. "
        "Use browse_components first to check if a base template exists."
    )
    input_schema: dict[str, Any] = {
        "description": str,
        "base_ref": (str, None),
        "entity_type": (str, None),
    }

    def __init__(self, entity_designer: Any = None) -> None:
        self._designer = entity_designer
        super().__init__()

    def execute(self, **kwargs: Any) -> Any:
        if self._designer is None:
            return ToolOutput(success=False, error="No EntityDesigner available")

        description = kwargs.get("description", "")
        base_ref = kwargs.get("base_ref")
        entity_type = kwargs.get("entity_type")

        try:
            spec = self._designer.design(
                description=description,
                base_ref=base_ref,
                entity_type=entity_type,
            )

            # Validate
            from maxim.simulation.entity_designer import validate_entity_spec
            errors = validate_entity_spec(spec)
            if errors:
                return {
                    "spec": spec,
                    "valid": False,
                    "errors": errors,
                    "message": "Entity generated but has validation issues",
                }

            return {
                "spec": spec,
                "valid": True,
                "message": f"Entity '{spec.get('name', '?')}' designed successfully",
            }
        except Exception as e:
            return ToolOutput(success=False, error=f"Entity design failed: {e}")


class EmitCampaignTool(Tool):
    """Output the final campaign YAML for the architect to emit."""

    name = "emit_campaign"
    description = (
        "Emit a complete campaign YAML. Provide the full campaign structure "
        "including acts, encounters, NPCs, and player character."
    )
    input_schema: dict[str, Any] = {
        "campaign_yaml": str,
    }

    def __init__(self, output_dir: str | None = None) -> None:
        self._output_dir = output_dir
        self._emitted: str | None = None
        super().__init__()

    @property
    def last_emitted(self) -> str | None:
        """The last emitted campaign YAML string."""
        return self._emitted

    def execute(self, **kwargs: Any) -> Any:
        yaml_content = kwargs.get("campaign_yaml", "")
        if not yaml_content:
            return ToolOutput(success=False, error="No campaign_yaml provided")

        self._emitted = yaml_content

        # Optionally save to file
        if self._output_dir:
            import time
            from pathlib import Path
            output_path = Path(self._output_dir) / f"generated_campaign_{int(time.time())}.yaml"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(yaml_content)
            log.info("Architect: campaign saved to %s", output_path)
            return {
                "success": True,
                "saved_to": str(output_path),
                "message": "Campaign YAML emitted and saved",
            }

        return {
            "success": True,
            "message": "Campaign YAML emitted (not saved to disk)",
        }
