"""DM Campaign Display Extension — character sheet, inventory, encounter info.

Adds toggleable game-state panels to MaximDisplay during DM campaigns.
Reads live sensor values from SEM entities via SceneState, so HP,
trust, and equipment reflect the current game state.

Example::

    from maxim.interactive.display import MaximDisplay
    from maxim.interactive.dm_display import CampaignDisplay

    display = MaximDisplay()
    dm_ext = CampaignDisplay(campaign_state, scene_state)
    display.add_extension(dm_ext)
"""

from __future__ import annotations

import logging
from typing import Any, Callable

log = logging.getLogger(__name__)

try:
    from rich.table import Table  # noqa: F401
    from rich.panel import Panel  # noqa: F401
    from rich.text import Text  # noqa: F401

    _RICH_AVAILABLE = True
except ImportError:
    _RICH_AVAILABLE = False


class CampaignDisplay:
    """DM campaign display extension for MaximDisplay.

    Shows game state panels: character sheet, inventory, encounter info,
    NPC relationships, choice history, and user notes.

    Toggle panels via key bindings:
        [c] Character    [i] Inventory    [e] Encounter
        [r] Relationships    [h] History    [n] Notes
    """

    def __init__(
        self,
        campaign_state: Any = None,
        scene_state: Any = None,
        pc_entity: Any = None,
    ) -> None:
        self._state = campaign_state
        self._scene = scene_state
        self._pc = pc_entity
        self._active_panel: str = "encounter"
        self._notes: list[str] = []
        self._show_side: bool = True

    def panel_name(self) -> str:
        return f"Campaign — {self._active_panel.title()}"

    def render(self) -> Any:
        if not _RICH_AVAILABLE:
            return ""

        renderers = {
            "character": self._render_character,
            "inventory": self._render_inventory,
            "encounter": self._render_encounter,
            "relationships": self._render_relationships,
            "history": self._render_history,
            "notes": self._render_notes,
        }
        renderer = renderers.get(self._active_panel, self._render_encounter)
        try:
            return renderer()
        except Exception as e:
            return Text(f"(render error: {e})", style="red")

    def key_bindings(self) -> dict[str, Callable]:
        return {
            "c": lambda: self._set_panel("character"),
            "i": lambda: self._set_panel("inventory"),
            "e": lambda: self._set_panel("encounter"),
            "r": lambda: self._set_panel("relationships"),
            "h": lambda: self._set_panel("history"),
            "n": lambda: self._set_panel("notes"),
        }

    def bind_campaign_state(self, state: Any) -> None:
        """Update campaign state reference."""
        self._state = state

    def bind_scene_state(self, scene: Any) -> None:
        """Update scene state reference (for live entity sensor reads)."""
        self._scene = scene

    def bind_pc_entity(self, entity: Any) -> None:
        """Update PC entity reference."""
        self._pc = entity

    def add_note(self, text: str) -> None:
        """Add a user note (persisted in session JSONL)."""
        self._notes.append(text)

    # -- Panel renderers ---------------------------------------------------

    def _render_character(self) -> Any:
        """Character sheet — name, HP, attributes from PC entity sensors."""
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("", style="bold cyan", width=12)
        table.add_column("")

        if self._pc is not None:
            table.add_row("Name", getattr(self._pc, "name", "Unknown"))
            table.add_row("Type", getattr(self._pc, "entity_type", "character"))

            # Read sensor values from entity
            for sensor_name, sensor in getattr(self._pc, "sensors", {}).items():
                try:
                    reading = sensor.read()
                    val = reading.value if hasattr(reading, "value") else str(reading)
                    table.add_row(sensor_name.title(), str(val))
                except Exception:
                    table.add_row(sensor_name.title(), "?")
        else:
            table.add_row("", "(no character loaded)")

        return table

    def _render_inventory(self) -> Any:
        """Bag contents from PC entity children with entity_type == item."""
        table = Table(show_header=True, box=None, padding=(0, 1))
        table.add_column("Item", style="bold")
        table.add_column("Status", style="dim")

        if self._pc is not None:
            items = [c for c in getattr(self._pc, "children", [])
                     if getattr(c, "entity_type", "") in ("item", "weapon", "armor", "consumable")]
            if items:
                for item in items:
                    status_parts = []
                    for sn, sv in getattr(item, "sensors", {}).items():
                        try:
                            reading = sv.read()
                            val = reading.value if hasattr(reading, "value") else reading
                            status_parts.append(f"{sn}={val}")
                        except Exception:
                            pass
                    table.add_row(item.name, ", ".join(status_parts) or "-")
            else:
                table.add_row("(empty)", "")
        else:
            table.add_row("(no character)", "")

        return table

    def _render_encounter(self) -> Any:
        """Current encounter: scene summary, NPCs, choices."""
        lines = []

        if self._state is not None:
            lines.append(f"[bold]Encounter:[/bold] {self._state.current_encounter}")
            lines.append(f"[bold]Turn:[/bold] {self._state.turn_count}")
            lines.append(f"[bold]Flags:[/bold] {', '.join(sorted(self._state.flags)) or 'none'}")

            if self._state.choices_made:
                last = self._state.choices_made[-1]
                lines.append(f"[bold]Last choice:[/bold] {last.get('choice', '?')} @ {last.get('encounter', '?')}")
        else:
            lines.append("(no campaign state)")

        return Text.from_markup("\n".join(lines))

    def _render_relationships(self) -> Any:
        """NPC trust/mood/hostility from entity sensors."""
        table = Table(show_header=True, box=None, padding=(0, 1))
        table.add_column("NPC", style="bold")
        table.add_column("Trust", style="green")
        table.add_column("Mood", style="yellow")

        if self._scene is not None and hasattr(self._scene, "_entity_registry"):
            for name, entity in self._scene._entity_registry.items():
                if getattr(entity, "entity_type", "") == "npc":
                    trust = "?"
                    mood = "?"
                    for sn, sv in getattr(entity, "sensors", {}).items():
                        try:
                            val = sv.read().value
                            if sn == "trust":
                                trust = f"{val:.1f}"
                            elif sn == "mood":
                                mood = f"{val:.1f}"
                        except Exception:
                            pass
                    table.add_row(name, trust, mood)

        if not table.rows:
            table.add_row("(no NPCs)", "", "")

        return table

    def _render_history(self) -> Any:
        """Choices made + dice rolls."""
        table = Table(show_header=True, box=None, padding=(0, 1))
        table.add_column("#", style="dim", width=3)
        table.add_column("Encounter")
        table.add_column("Choice", style="bold")

        if self._state is not None:
            for i, choice in enumerate(self._state.choices_made, 1):
                table.add_row(
                    str(i),
                    choice.get("encounter", "?"),
                    choice.get("choice", "?"),
                )

        if not table.rows:
            table.add_row("", "(no choices yet)", "")

        return table

    def _render_notes(self) -> Any:
        """User's personal notes."""
        if self._notes:
            return Text("\n".join(f"• {n}" for n in self._notes))
        return Text("(no notes — press [n] to add)", style="dim")

    # -- internals ---------------------------------------------------------

    def _set_panel(self, name: str) -> None:
        self._active_panel = name
