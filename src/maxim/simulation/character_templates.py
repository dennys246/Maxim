"""Character templates — class archetypes and NPC role templates.

Provides fallback templates when LLM is unavailable for the Generative
Architect's character creation flow.  Each archetype defines attribute
arrays, ability pools, and starting inventory patterns.

Used by:
- EntityDesigner (fallback mode)
- Generative Architect (propose_character tool)
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# PC Class Archetypes
# ---------------------------------------------------------------------------

PC_ARCHETYPES: dict[str, dict[str, Any]] = {
    "warrior": {
        "attributes": {"str": 16, "dex": 12, "con": 14, "int": 8, "wis": 10, "cha": 10},
        "abilities": ["Power Strike", "Shield Bash", "Battle Cry"],
        "starting_inventory": ["longsword", "shield", "chainmail", "healing_potion"],
        "suggested_sensors": {
            "hp": {"unit": "points", "range": [0, 25], "initial": 25},
            "stamina": {"unit": "points", "range": [0, 12], "initial": 12},
            "armor": {"unit": "points", "range": [0, 18], "initial": 14},
        },
    },
    "mage": {
        "attributes": {"str": 8, "dex": 10, "con": 10, "int": 16, "wis": 14, "cha": 12},
        "abilities": ["Fireball", "Shield Spell", "Arcane Insight"],
        "starting_inventory": ["staff", "spellbook", "robes", "mana_potion"],
        "suggested_sensors": {
            "hp": {"unit": "points", "range": [0, 16], "initial": 16},
            "mana": {"unit": "points", "range": [0, 20], "initial": 20},
            "spell_slots": {"unit": "count", "range": [0, 5], "initial": 5},
        },
    },
    "rogue": {
        "attributes": {"str": 10, "dex": 16, "con": 10, "int": 14, "wis": 10, "cha": 12},
        "abilities": ["Sneak Attack", "Lockpick", "Evasion"],
        "starting_inventory": ["daggers", "lockpicks", "leather_armor", "smoke_bomb"],
        "suggested_sensors": {
            "hp": {"unit": "points", "range": [0, 18], "initial": 18},
            "stamina": {"unit": "points", "range": [0, 15], "initial": 15},
            "stealth": {"unit": "ratio", "range": [0, 1], "initial": 0.8},
        },
    },
    "diplomat": {
        "attributes": {"str": 8, "dex": 10, "con": 10, "int": 14, "wis": 12, "cha": 16},
        "abilities": ["Persuade", "Calm Emotions", "Detect Lies"],
        "starting_inventory": ["fine_clothes", "signet_ring", "letter_of_introduction", "coin_purse"],
        "suggested_sensors": {
            "hp": {"unit": "points", "range": [0, 14], "initial": 14},
            "influence": {"unit": "ratio", "range": [0, 1], "initial": 0.6},
            "reputation": {"unit": "ratio", "range": [0, 1], "initial": 0.5},
        },
    },
    "healer": {
        "attributes": {"str": 10, "dex": 10, "con": 12, "int": 12, "wis": 16, "cha": 14},
        "abilities": ["Healing Word", "Purify", "Sanctuary"],
        "starting_inventory": ["mace", "holy_symbol", "healer_kit", "blessed_water"],
        "suggested_sensors": {
            "hp": {"unit": "points", "range": [0, 18], "initial": 18},
            "faith": {"unit": "ratio", "range": [0, 1], "initial": 0.8},
            "spell_slots": {"unit": "count", "range": [0, 4], "initial": 4},
        },
    },
}

# ---------------------------------------------------------------------------
# NPC Role Templates
# ---------------------------------------------------------------------------

NPC_ROLES: dict[str, dict[str, Any]] = {
    "guard": {
        "entity_type": "npc",
        "suggested_base": "npcs/guard",
        "default_personality": "Stern and duty-bound, follows orders.",
        "sensors": {"trust": 0.4, "suspicion": 0.3},
    },
    "merchant": {
        "entity_type": "npc",
        "suggested_base": "npcs/merchant",
        "default_personality": "Shrewd but fair, always looking for a deal.",
        "sensors": {"trust": 0.5, "gold": 200},
    },
    "innkeeper": {
        "entity_type": "npc",
        "suggested_base": "npcs/base_humanoid",
        "default_personality": "Warm and welcoming, knows all the local gossip.",
        "sensors": {"trust": 0.7, "mood": 0.3},
    },
    "villain": {
        "entity_type": "npc",
        "suggested_base": "npcs/base_humanoid",
        "default_personality": "Cunning and manipulative, hides true intentions.",
        "sensors": {"trust": 0.1, "mood": -0.3},
    },
    "companion": {
        "entity_type": "npc",
        "suggested_base": "npcs/base_humanoid",
        "default_personality": "Loyal and supportive, follows the player's lead.",
        "sensors": {"trust": 0.9, "mood": 0.5},
    },
    "mysterious_stranger": {
        "entity_type": "npc",
        "suggested_base": "npcs/base_humanoid",
        "default_personality": "Enigmatic, speaks in riddles, motives unclear.",
        "sensors": {"trust": 0.3, "mood": 0.0},
    },
}


def get_archetype(name: str) -> dict[str, Any] | None:
    """Look up a PC class archetype by name."""
    return PC_ARCHETYPES.get(name.lower())


def get_npc_role(name: str) -> dict[str, Any] | None:
    """Look up an NPC role template by name."""
    return NPC_ROLES.get(name.lower())


def list_archetypes() -> list[str]:
    """Return available PC class archetype names."""
    return sorted(PC_ARCHETYPES.keys())


def list_npc_roles() -> list[str]:
    """Return available NPC role template names."""
    return sorted(NPC_ROLES.keys())
