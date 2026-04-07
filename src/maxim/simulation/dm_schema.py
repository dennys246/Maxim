"""DM Campaign schema — dataclasses + YAML loader + validator.

Campaign YAML has two parts:
1. Entity definitions (PCs, NPCs, objects) using standard SEM spec format
2. Campaign structure (acts, encounters, branches, dice checks)

Example:
    campaign = load_campaign("scenarios/campaigns/heist_v1.yaml")
    errors = validate_campaign(campaign)
    if errors:
        raise ValueError(f"Campaign validation failed: {errors}")
"""

from __future__ import annotations

import logging
import random
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class DiceCheck:
    """A dice roll required by an encounter choice."""

    roll: str  # e.g. "1d20", "2d6"
    dc: int  # Difficulty class
    attr_mod: str = ""  # Attribute modifier (e.g. "dexterity")
    success_flag: str = ""  # Flag set on success


@dataclass
class EncounterDef:
    """A single encounter within an act."""

    name: str
    scene: str  # Narrative text delivered to AUT
    active_npcs: list[str] = field(default_factory=list)
    world_objects: list[str] = field(default_factory=list)
    choices: list[str] = field(default_factory=list)
    branches: dict[str, str] = field(default_factory=dict)  # choice → encounter_name or __END__
    on_choice: dict[str, dict[str, Any]] = field(default_factory=dict)  # choice → {flags, loot, ...}
    dice: dict[str, dict[str, Any]] = field(default_factory=dict)  # choice → DiceCheck params
    dialogue_hints: dict[str, str] = field(default_factory=dict)  # flag → hint text (context seeds)


@dataclass
class Act:
    """A named group of encounters forming a narrative arc."""

    name: str
    encounters: list[str]  # Encounter names in order


@dataclass
class CampaignDef:
    """Top-level campaign definition loaded from YAML."""

    name: str
    goal: str
    seed: int
    acts: list[Act]
    encounters: dict[str, EncounterDef]
    pc_spec: dict[str, Any]  # Raw SEM entity YAML for player character
    npc_specs: dict[str, dict[str, Any]]  # name → raw SEM entity YAML
    object_specs: dict[str, dict[str, Any]] = field(default_factory=dict)
    expectations: dict[str, Any] = field(default_factory=dict)  # Bio-system expectations

    @property
    def encounter_order(self) -> list[str]:
        """Flat list of encounter names in act order."""
        result = []
        for act in self.acts:
            result.extend(act.encounters)
        return result

    @property
    def first_encounter(self) -> str:
        """Name of the first encounter."""
        return self.acts[0].encounters[0]


# ---------------------------------------------------------------------------
# YAML Loader
# ---------------------------------------------------------------------------


def load_campaign(path: str | Path) -> CampaignDef:
    """Load a campaign from a YAML file.

    Args:
        path: Path to the campaign YAML file.

    Returns:
        Parsed CampaignDef.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If required fields are missing.
    """
    path = Path(path)
    with open(path) as f:
        raw = yaml.safe_load(f)

    campaign = raw.get("campaign", {})
    name = campaign.get("name", path.stem)
    goal = campaign.get("goal", "")
    seed = campaign.get("seed", 42)

    # Parse acts
    acts = []
    for act_raw in raw.get("acts", []):
        acts.append(
            Act(
                name=act_raw["name"],
                encounters=act_raw["encounters"],
            )
        )

    # Parse encounters (case-normalize keys)
    encounters = {}
    for enc_name, enc_raw in raw.get("encounters", {}).items():
        enc_name_lower = enc_name.lower() if isinstance(enc_name, str) else enc_name
        # Case-normalize dialogue hints and on_choice keys
        dialogue_hints = {}
        for k, v in (enc_raw.get("dialogue_hints") or {}).items():
            dialogue_hints[k.lower()] = v
        on_choice = {}
        for k, v in (enc_raw.get("on_choice") or {}).items():
            on_choice[k.lower()] = v

        encounters[enc_name_lower] = EncounterDef(
            name=enc_name_lower,
            scene=enc_raw.get("scene", ""),
            active_npcs=enc_raw.get("active_npcs", []),
            world_objects=enc_raw.get("world_objects", []),
            choices=enc_raw.get("choices", []),
            branches={k.lower(): v for k, v in (enc_raw.get("branches") or {}).items()},
            on_choice=on_choice,
            dice=enc_raw.get("dice", {}),
            dialogue_hints=dialogue_hints,
        )

    # Entity specs (raw dicts — loaded via load_spec() at runtime)
    pc_spec = raw.get("player_character", {})
    npc_specs = raw.get("npcs", {})
    object_specs = raw.get("world_objects", {}) if isinstance(raw.get("world_objects"), dict) else {}
    expectations = raw.get("expectations", {})

    return CampaignDef(
        name=name,
        goal=goal,
        seed=seed,
        acts=acts,
        encounters=encounters,
        pc_spec=pc_spec,
        npc_specs=npc_specs,
        object_specs=object_specs,
        expectations=expectations,
    )


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


def validate_campaign(campaign: CampaignDef) -> list[str]:
    """Validate a campaign for structural correctness.

    Returns a list of error strings. Empty list = valid.

    Checks:
    1. Reachability — all encounters reachable from first encounter
    2. Termination — all paths can reach __END__
    3. Dangling branches — targets must be defined encounters or __END__
    4. NPC refs — active_npcs must reference defined NPCs
    5. Object refs — world_objects must reference defined objects
    6. Unknown choice keys — on_choice keys must be in choices list
    """
    errors = []
    enc_names = set(campaign.encounters.keys())

    # 1. Dangling branches
    for enc_name, enc in campaign.encounters.items():
        for choice, target in enc.branches.items():
            if target != "__END__" and target.lower() not in enc_names:
                errors.append(f"Encounter '{enc_name}': branch '{choice}' targets unknown encounter '{target}'")

    # 2. Reachability from first encounter (BFS)
    if campaign.acts:
        start = campaign.first_encounter.lower()
        reachable = set()
        queue = deque([start])
        while queue:
            current = queue.popleft()
            if current in reachable or current == "__END__":
                continue
            reachable.add(current)
            enc = campaign.encounters.get(current)
            if enc:
                for target in enc.branches.values():
                    t = target.lower() if target != "__END__" else "__END__"
                    if t not in reachable:
                        queue.append(t)

        unreachable = enc_names - reachable
        for name in unreachable:
            errors.append(f"Encounter '{name}' is unreachable from start")

    # 3. Termination — all encounters must be able to reach __END__
    # Build reverse graph and check __END__ reachability
    can_reach_end: set[str] = set()
    # Direct __END__ connections
    for enc_name, enc in campaign.encounters.items():
        if "__END__" in enc.branches.values():
            can_reach_end.add(enc_name)
    # Propagate backward
    changed = True
    while changed:
        changed = False
        for enc_name, enc in campaign.encounters.items():
            if enc_name in can_reach_end:
                continue
            for target in enc.branches.values():
                if target.lower() in can_reach_end:
                    can_reach_end.add(enc_name)
                    changed = True
                    break
    no_end = enc_names - can_reach_end
    for name in no_end:
        if campaign.encounters[name].branches:  # Only flag if it has branches at all
            errors.append(f"Encounter '{name}' cannot reach __END__ through any path")

    # 4. NPC refs
    npc_names = set(campaign.npc_specs.keys())
    for enc_name, enc in campaign.encounters.items():
        for npc in enc.active_npcs:
            if npc not in npc_names:
                errors.append(f"Encounter '{enc_name}': references undefined NPC '{npc}'")

    # 5. Object refs
    obj_names = set(campaign.object_specs.keys())
    for enc_name, enc in campaign.encounters.items():
        for obj in enc.world_objects:
            if obj not in obj_names:
                errors.append(f"Encounter '{enc_name}': references undefined object '{obj}'")

    # 6. Unknown choice keys in on_choice
    for enc_name, enc in campaign.encounters.items():
        valid_choices = set(c.lower() for c in enc.choices)
        for key in enc.on_choice:
            if key not in valid_choices:
                errors.append(f"Encounter '{enc_name}': on_choice key '{key}' not in choices {enc.choices}")

    return errors


# ---------------------------------------------------------------------------
# Dice resolution
# ---------------------------------------------------------------------------


def roll_dice(spec: str, rng: random.Random) -> int:
    """Roll dice from a spec string like '1d20', '2d6+3'.

    Args:
        spec: Dice notation (NdM or NdM+K).
        rng: Seeded random instance for reproducibility.

    Returns:
        Total roll result.
    """
    # Parse NdM+K
    spec = spec.strip().lower()
    bonus = 0
    if "+" in spec:
        dice_part, bonus_str = spec.split("+", 1)
        bonus = int(bonus_str)
    elif "-" in spec and not spec.startswith("-"):
        dice_part, penalty_str = spec.split("-", 1)
        bonus = -int(penalty_str)
    else:
        dice_part = spec

    num_str, sides_str = dice_part.split("d")
    num = int(num_str) if num_str else 1
    sides = int(sides_str)

    total = sum(rng.randint(1, sides) for _ in range(num)) + bonus
    return total
