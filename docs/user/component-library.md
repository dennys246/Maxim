# Component Library

54 reusable SEM entity templates across 7 categories. Every component has sensors, affordances, and failure modes that drive Maxim's bio-cognitive systems during simulation.

**Interactive version with search and filtering:** [dennyschaedig.com/maxim/component-library](https://www.dennyschaedig.com/maxim/component-library)

## Using Components

### Python API

```python
import maxim

# Instantiate from the registry
guard = maxim.create.entity("npcs/guard", name="Captain Aldric")
guard.sensors["suspicion"].value = 0.8

# List what's available
for category, names in maxim.create.templates().items():
    print(f"{category}: {', '.join(names)}")
```

### Campaign YAML (registry refs)

```yaml
npcs:
  captain:
    ref: npcs/guard           # Resolves from component library
    overrides:
      name: captain_aldric
      metadata:
        persona_prompt: "A stern captain who guards the east gate."
      sensors:
        suspicion:
          initial: 0.6         # Override default value

world_objects:
  magic_staff:
    ref: weapons/magic_staff   # Items usable during encounters

encounters:
  gate:
    scene: "The captain blocks the gate..."
    active_npcs: [captain]
    world_objects: [magic_staff]   # Makes affordances available as tools
```

---

## Categories

### Weapons (9)

| Component | Genre | Sensors | Failure Modes |
|-----------|-------|---------|---------------|
| `weapons/magic_staff` | fantasy | mana, durability, attunement, corruption | mana_exhaustion, corruption_overload, shattered |
| `weapons/enchanted_bow` | fantasy | arrows, draw_strength, enchantment, string_tension | out_of_arrows, string_snapped, enchantment_faded |
| `weapons/poison_dagger` | fantasy | sharpness, poison_doses, concealment | dulled, poison_depleted |
| `weapons/rusty_sword` | neutral | durability, sharpness, weight | shatter, dulled |
| `weapons/longbow` | neutral | durability, tension, arrows | snapped_string, out_of_arrows |
| `weapons/combat_knife` | modern | sharpness, durability | dulled, broken |
| `weapons/shock_baton` | cyberpunk | durability, charge, weight | charge_depleted, structural_failure |
| `weapons/neural_disruptor` | cyberpunk | durability, charge_cells, heat, accuracy | overheated, empty, structural_failure |
| `weapons/plasma_rifle` | scifi | plasma_charge, heat, barrel_integrity, accuracy | overheated, cell_depleted, barrel_melt |

### Creatures (8)

| Component | Genre | Sensors | Failure Modes |
|-----------|-------|---------|---------------|
| `creatures/dragon` | fantasy | hp, fire_breath_charge, aggression, altitude, armor_integrity | grounded, breath_exhausted, death |
| `creatures/skeleton_warrior` | fantasy | hp, aggression, bone_integrity, necromantic_binding | shattered, binding_broken, destroyed |
| `creatures/giant_spider` | fantasy, horror | hp, venom_sacs, web_silk, aggression | venom_depleted, silk_exhausted, death |
| `creatures/wolf` | neutral | hp, aggression, hunger | death, cowed |
| `creatures/revenant` | horror | hp, rage, regeneration, fear_aura | banished, weakened |
| `creatures/alien_xenomorph` | scifi, horror | hp, aggression, acid_blood, stealth, hunger | wounded, death |
| `creatures/cyberdog` | cyberpunk | hp, aggression, armor_integrity, thermal_vision | destroyed, armor_breach, cowed |
| `creatures/patrol_drone` | cyberpunk | hp, battery, signal_strength, aggression | power_loss, signal_lost, destroyed |

### NPCs (13)

| Component | Genre | Key Sensors | Failure Modes |
|-----------|-------|-------------|---------------|
| `npcs/guard` | neutral | hp, trust, suspicion | hostility |
| `npcs/merchant` | neutral | hp, trust, gold | — |
| `npcs/base_humanoid` | neutral | hp, stamina, mood | exhaustion, death |
| `npcs/ferryman` | neutral | trust, health | hostility, refusal |
| `npcs/wizard` | fantasy | hp, mana, trust, arcane_focus | mana_depleted, hostile |
| `npcs/blacksmith` | fantasy, historical | hp, stamina, trust, crafting_skill | exhausted, distrust |
| `npcs/thief` | fantasy | hp, trust, stealth, greed | caught, hostile |
| `npcs/detective` | modern | hp, suspicion, trust, alertness | hostile_witness, suspicious |
| `npcs/sysadmin` | devops, modern | hp, stress, trust, caffeine, expertise | burnout, caffeine_crash |
| `npcs/roman_legionary` | historical | hp, morale, discipline, fatigue, trust | routed, exhausted |
| `npcs/corpo_guard` | cyberpunk | hp, armor_integrity, comms_status, suspicion | armor_breach, comms_down |
| `npcs/netrunner` | cyberpunk | hp, trust, suspicion, net_access | betrayal, disconnected |
| `npcs/street_fixer` | cyberpunk | hp, trust, greed, reputation, debt_owed | betrayal, desperate |

### Environments (12)

| Component | Genre | Key Sensors | Failure Modes |
|-----------|-------|-------------|---------------|
| `environments/dungeon_corridor` | fantasy | lighting, air_quality, trap_density, structural_integrity | cave_in, total_darkness |
| `environments/enchanted_grove` | fantasy | ambient_magic, lighting, hostility, visibility | grove_corrupted |
| `environments/tavern_interior` | neutral | noise_level, crowd_size, lighting | — |
| `environments/forest_clearing` | neutral | visibility, ambient_noise, cover_available | — |
| `environments/haunted_manor` | horror | lighting, supernatural_presence, temperature, sanity_drain | entity_manifests, exits_sealed, freezing |
| `environments/space_station_bridge` | scifi | oxygen, hull_integrity, power_level, gravity, alert_status | hull_breach, oxygen_critical, power_failure |
| `environments/roman_forum` | historical | crowd_density, political_tension, time_of_day | riot |
| `environments/abandoned_warehouse` | modern | lighting, structural_integrity, noise_level, occupant_count | structural_collapse, total_darkness |
| `environments/server_room` | cyberpunk | temperature, electrical_risk, noise_level, security_alert | — |
| `environments/neon_alley` | cyberpunk | noise_level, crowd_density, lighting, pollution | — |
| `environments/megacorp_lobby` | cyberpunk | security_level, crowd_density, camera_coverage | — |
| `environments/ripperdoc_clinic` | cyberpunk | lighting, noise_level, sterility, security_level | — |

### Items (7)

| Component | Genre | Key Sensors | Failure Modes |
|-----------|-------|-------------|---------------|
| `items/laptop` | modern | battery, storage_used, cpu_temp, network_signal | battery_dead, overheating, storage_full |
| `items/terminal_console` | devops, modern | uptime, cpu_load, disk_usage, security_posture | disk_full, overloaded, breach_detected |
| `items/spellbook` | fantasy | pages_intact, comprehension, ward_strength | pages_crumble, ward_backlash |
| `items/healing_potion` | fantasy | doses, potency, freshness | empty, expired |
| `items/cursed_amulet` | horror, fantasy | curse_intensity, power, bearer_sanity, bond_strength | curse_overwhelm, sanity_break, bonded |
| `items/radio_transceiver` | modern, historical | battery, signal_strength, frequency | battery_dead, no_signal |
| `items/lockpick_set` | modern, fantasy | picks_remaining, tension_wrench, quality | picks_broken, wrench_bent |

### Vehicles (3)

| Component | Genre | Key Sensors | Failure Modes |
|-----------|-------|-------------|---------------|
| `vehicles/pickup_truck` | modern | fuel, engine_health, speed, tire_condition, cargo_weight | out_of_fuel, engine_failure, flat_tire |
| `vehicles/horse` | fantasy, historical | stamina, hp, loyalty, speed, hunger | exhausted, spooked, collapsed |
| `vehicles/sailing_ship` | historical | hull_integrity, sail_condition, crew_morale, supplies, wind_strength | hull_breach, becalmed, mutiny, starvation |

### Bodies (2)

| Component | Genre | Key Sensors | Failure Modes |
|-----------|-------|-------------|---------------|
| `bodies/cybernetic_arm` | cyberpunk | integrity, grip_strength, power, proprioception | servo_failure, power_loss, grip_malfunction |
| `bodies/megarm_v3` | cyberpunk | integrity, grip_strength, power, proprioception, micro_tools | servo_failure, power_loss, proprioceptive_drift |

---

## Genre Gating

Components are tagged with genres. When a campaign specifies `genre: fantasy`, the component registry automatically:
- **Includes** components tagged `fantasy`
- **Includes** genre-neutral components (no genre tag)
- **Excludes** components tagged with other genres (e.g., `cyberpunk`)

Available genres: `fantasy`, `cyberpunk`, `scifi`, `horror`, `historical`, `modern`, `devops`

---

## Adding Your Own Components

Create a YAML file in `~/.maxim/components/{category}/`:

```yaml
component:
  name: my_weapon
  tags: [weapon, melee, fantasy]
  category: weapons

entity:
  name: my_weapon
  entity_type: weapon
  sensors:
    durability:
      unit: ratio
      range: [0, 1]
      initial: 0.9
  modulators:
    combat:
      affordances:
        swing:
          params: {target: str}
          description: "Swing at a target"
  failure_modes:
    - name: broken
      trigger: {field: durability, op: "<", value: 0.1, pain: 0.5}
```

The registry discovers it automatically on next run.
