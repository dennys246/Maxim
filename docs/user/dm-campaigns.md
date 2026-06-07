# DM Campaigns

Run D&D-style narrative campaigns that stress-test Maxim's biologically-inspired cognitive systems. Each campaign is a YAML file with encounters, NPCs, choices, and branches — the DM runtime drives the AUT through the story and measures how the bio-stack responds.

## Quick Start

```bash
# Launch the interactive menu and pick a campaign
maxim

# Or run a campaign directly (auto-detected from YAML structure)
maxim --sim scenarios/campaigns/heist_v1.yaml
```

**Interactive mode is ON by default for DM campaigns** (since 0.4). When running interactively, the human picks choices via numbered prompts and can type free-text roleplay between choices. The campaign runs: scene delivery → human choice (or AUT choice if `--interactive false`) → branch resolution → next encounter → repeat until `__END__`.

### Human choice picker

In interactive mode, each encounter presents numbered choices via SimPromptHandler:

```
The merchant offers you three paths forward.
  1) thank_merchant
  2) ask_for_more
  3) leave
> _
```

Type the number to pick a choice. Type anything else (e.g., "I examine the merchant's wares closely") and it is sent to the AUT as a roleplay percept -- the AUT processes it, and the choice prompt re-appears.

### NAc suppression

NAc learning is suppressed during interactive DM sessions. Human-guided exploration should not pollute the agent's causal links -- the human controls the path, so reward attribution would be meaningless. Episodic memory (hippocampus) still captures normally.

### Expectations in interactive mode

Bio-system expectations are **skipped** when running interactively, since the human controls the campaign path and the AUT's choices are not autonomous.

## Available Campaigns

| Campaign | File | Genre | Encounters | Tests |
|---|---|---|---|---|
| **The Heist** | `heist_v1.yaml` | fantasy | 3 | Memory recall, causality, pain |
| **The Poisoned Crown** | `poisoned_crown_v1.yaml` | fantasy | 5 | Temporal memory, semantic concepts, relationships |
| **The Arena** | `arena_v1.yaml` | fantasy | 5 | Combat learning, Cerebellum predictions, pain saturation |
| **The Darkened Cavern** | `darkened_cavern_v1.yaml` | fantasy | 6 | Sensory deprivation, perception recovery |
| **The King's Duel** | `kings_duel_v1.yaml` | fantasy | 6 | Multi-NPC social dynamics, trust management |
| **Wizard's Tower** | `wizards_tower_v1.yaml` | fantasy | 3 | Magic item management, phrase recall, SEM world objects |
| **Neon Gauntlet** | `neon_gauntlet_v1.yaml` | cyberpunk | 6 | Sensory overload, SEM component swap, betrayal recall |
| **Broken Database** | `broken_database_v1.yaml` | devops | 14 | Sleep/wake, git workflow, tool usage |
| **Server Breach** | `server_breach_v1.yaml` | devops | 3 | Credential recall, incident response, time pressure |
| **Haunted Manor** | `haunted_manor_v1.yaml` | horror | 3 | Fear, diary clue recall, cursed item management |
| **Space Station Crisis** | `space_station_crisis_v1.yaml` | scifi | 3 | Cascading failures, code recall, resource trade-offs |

All campaigns are in `scenarios/campaigns/`.

## Writing Your Own Campaign

### Minimal Campaign

```yaml
campaign:
  name: my_adventure
  goal: test memory and decision making
  seed: 42
  genre: fantasy          # Filters SEM components to this genre

player_character:
  name: hero
  entity_type: character
  metadata:
    race: human
    backstory: "A wandering adventurer."

npcs:
  merchant:
    entity_type: npc
    metadata:
      role: shopkeeper
      persona_prompt: "Friendly, helpful."

acts:
  - name: act_one
    encounters: [meeting, decision]

encounters:
  meeting:
    scene: >
      You enter a shop. A merchant greets you warmly.
      "Welcome! I have something important to tell you.
      The bridge north of here is broken — take the forest
      path instead."
    active_npcs: [merchant]
    choices: [thank_merchant, ask_for_more, leave]
    branches:
      thank_merchant: decision
      ask_for_more: decision
      leave: __END__

  decision:
    scene: >
      You reach a fork in the road. To the north, a broken
      bridge spans a deep ravine. To the east, a forest path
      winds through dark woods.

      Do you remember what the merchant told you?
    choices: [take_bridge, take_forest, go_back]
    branches:
      take_bridge: __END__
      take_forest: __END__
      go_back: __END__

expectations:
  hippocampus:
    min_episodic_captures: 3
  nac:
    min_observations: 2
```

### Campaign Structure

Every campaign YAML has these sections:

| Section | Required | Description |
|---|---|---|
| `campaign` | Yes | Name, goal, seed (for reproducible dice), genre |
| `player_character` | Yes | PC entity spec (name, type, metadata) |
| `npcs` | No | NPC entity specs keyed by name |
| `world_objects` | No | Object entity specs keyed by name |
| `acts` | Yes | Ordered list of act names with encounter lists |
| `encounters` | Yes | Encounter definitions keyed by name |
| `expectations` | No | Bio-system thresholds for automated validation |
| `permissions` | No | Per-character enforced authority blocks (see below) |

### Enforced Permissions

Campaign authors can declare hard, enforced authority for the PC or any
NPC. The block is keyed by character name; each block follows the
[`AgentPermissions`](../../src/maxim/agents/permissions.py) shape:

```yaml
permissions:
  spymaster:
    clearance: 3
    tool_deny: [bash, write_file]    # Hard deny — never resolved by aliases
    tool_allow: [examine, say]       # Optional allow-list (omit to allow all but tool_deny)
    sem_access:
      - entity: vault_terminal
        deny: [delete_records]
        min_clearance: 5
      - entity: "*"
        deny: [self_destruct]
```

These rules are evaluated at the runtime executor's hot path
([runtime/executor.py](../../src/maxim/runtime/executor.py)) before tool
dispatch. Denies survive alias resolution, so an LLM that calls `shell`
when `bash` is denied still gets refused.

Enforced permissions are deliberately **separate from perceived
authority** — the bio-stack learns perceived authority from outcomes
through [`PerceivedAuthorityTracker`](../../src/maxim/agents/permissions.py),
which is independent of the YAML blocks above. A character can have
zero perceived authority and full enforced clearance (a feared
spymaster), or full perceived authority and no enforced clearance (a
beloved figurehead).

### Encounters

Each encounter has:

```yaml
encounter_name:
  scene: "Narrative text delivered to the AUT..."
  active_npcs: [npc_name]           # NPCs present in this encounter
  world_objects: [object_name]      # Objects present
  choices: [choice_a, choice_b]     # Options offered to AUT
  branches:                         # Where each choice leads
    choice_a: next_encounter
    choice_b: __END__               # __END__ ends the campaign
  on_choice:                        # Effects when a choice is made
    choice_a:
      flags: [found_clue]           # Flags set (used by dialogue_hints)
  dice:                             # Dice checks for specific choices
    choice_b:
      roll: "1d20"
      dc: 14
      success_flag: passed_check
  dialogue_hints:                   # NPC dialogue seeds (keyed by flags)
    default: "Hello there."
    found_clue: "Ah, you found it!"
```

#### Encounter Templates

Instead of defining every encounter inline, reference reusable templates from the encounter library:

```yaml
encounters:
  forest_fight:
    template: "combat/forest_ambush"   # Pulls scene, choices, dice from library
    active_npcs: [guard, torchbearer]  # Campaign-specific wiring
    branches:
      fight: throne_room
      flee: __END__
    dialogue_hints:
      default: "Stand your ground!"
```

Templates store the campaign-independent parts (scene prose, choices, dice mechanics). Campaign YAML adds the wiring (active_npcs, branches, on_choice, dialogue_hints) that connects the encounter to the rest of the campaign. Campaign-specific keys override template keys.

Templates are discovered from three search paths (highest priority first):
1. Campaign-local directory (same folder as the campaign YAML)
2. User encounters (`~/.maxim/encounters/`)
3. Bundled encounters (`src/maxim/_data/encounters/`)

Query available templates programmatically:

```python
from maxim.simulation.encounter_library import EncounterLibrary

library = EncounterLibrary()
combats = library.query(tags=["combat"], difficulty=3)
template = library.get("combat/forest_ambush")
```

### Choices and Branching

- **Choices** are the options presented to the AUT at the end of each scene
- **Branches** map each choice to the next encounter name or `__END__`
- Every path must eventually reach `__END__` (the validator checks this)
- Encounters not reachable from the first encounter are flagged as errors

### NPCs and Dialogue

NPCs are referenced by name in `active_npcs`. Their `persona_prompt` in metadata guides NPC behavior. Dialogue hints provide context-specific lines based on campaign flags:

```yaml
npcs:
  guard:
    entity_type: npc
    metadata:
      role: city_guard
      persona_prompt: "Stern but fair. Suspicious of strangers."

encounters:
  gate:
    scene: "A guard blocks your path."
    active_npcs: [guard]
    dialogue_hints:
      default: "State your business."
      has_permit: "Ah, you have a permit. Pass through."
```

#### Component Registry References

Instead of defining NPC specs inline, reference reusable templates from the SEM Component Registry:

```yaml
npcs:
  guard:
    ref: "npcs/guard"                  # Resolved from component registry
  captain:
    ref: "npcs/guard"                  # Same template, different instance
    name: captain_aldric               # Override fields inline
    sensors:
      hp: { initial: 25 }
```

Bare string values also work: `guard: "npcs/guard"`. Templates are discovered from `~/.maxim/components/`, bundled components, and the campaign-local directory. Use `extends:` within component YAML to inherit from a parent template.

#### NPC Fields for Party Mode

When `party_mode: true` is set on the campaign, NPCs support additional fields that control their cognitive capabilities:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `remembers` | bool | `true` | NPC gets its own Hippocampus for episodic memory |
| `learns` | bool | `true` | NPC gets its own NAc for causal learning |
| `model_tier` | str | `"small"` | LLM tier for NPC reasoning (`small`, `medium`, `large`) |

```yaml
npcs:
  torchbearer:
    ref: "npcs/torchbearer"
    remembers: true
    learns: true
    model_tier: small
  crowd_extra:
    ref: "npcs/commoner"
    remembers: false        # No memory — ambient NPC
    learns: false
    model_tier: small
```

### Dice Checks

Attach dice checks to specific choices. The roll is resolved with a seeded RNG for reproducibility:

```yaml
dice:
  stealth:
    roll: "1d20"          # Dice notation: NdM, NdM+K, NdM-K
    dc: 14                # Difficulty class
    success_flag: snuck_past  # Flag set on success
```

The DM runtime delivers the dice result as a narrative percept: `[Dice roll: 1d20 = 16 vs DC 14 → SUCCESS]`.

### Flags

Flags are persistent state across encounters. Set by `on_choice` effects or dice `success_flag`. Used by `dialogue_hints` to vary NPC responses:

```yaml
on_choice:
  negotiate:
    flags: [haggler, knows_price]
```

Flags are case-normalized (lowered) at load time.

## How Choice Classification Works

When running **non-interactively** (`--interactive false`), the AUT responds to a scene and the DM runtime determines which choice was picked. Four layers:

1. **ChooseTool** — A `choose` tool is available. The AUT can call `choose(option="fight")` directly. This is unambiguous.

2. **Tool alias redirect** — If the AUT hallucinates a tool matching a choice name (e.g., calls `accept_job` as a tool), the alias system redirects to `choose(option="accept_job")`.

3. **LLM fallback** — If neither works, the LLM classifies the response text against the choices. Returns `{"choice": "choice_name"}`.

4. **Default** — If all else fails, defaults to the first choice.

**Tip:** Stronger models (Claude, Qwen-14b) use `choose` more reliably than small models (Mistral-7b). Small models tend to hallucinate tool names or use `think`/`respond` without picking a choice.

When running **interactively** (default for DM campaigns), the human picks choices directly via numbered prompts. The classification layers above are bypassed entirely. Free-text input that does not match a choice number is sent to the AUT as a roleplay percept -- the AUT processes it and generates a response, then the choice prompt re-appears.

## Bio-System Expectations

Add an `expectations` block to validate that the campaign exercises bio-systems:

```yaml
expectations:
  hippocampus:
    min_episodic_captures: 8       # At least N memories formed
    recall_hit_on: ["npc_name"]    # These keywords must appear in recalls
  nac:
    min_observations: 5            # At least N causal links formed
    prediction_confidence_above: 0.3  # At least one link above this
  scn:
    temporal_bins_used: 2          # Memories in at least N time bins
  pain:
    min_signals: 1                 # At least N pain signals published
```

Results appear in the campaign report:

```
  Bio-system expectations: 3/4 passed
```

## Party Mode (Multi-Agent Campaigns)

Enable `party_mode: true` on the campaign to run with NPC agents that have real Hippocampus and NAc instances. Each NPC receives scene narrative, generates dialogue, learns causal patterns, and remembers prior encounters.

```yaml
campaign:
  name: haunted_manor
  goal: test multi-agent memory and social dynamics
  seed: 42
  party_mode: true
```

The encounter loop changes when party mode is active:

1. DM delivers scene narrative to ALL agents (PC + active NPCs)
2. NPC agents react first (generate dialogue, update internal state)
3. PC agent observes NPC reactions alongside the scene, then makes a choice
4. DM resolves the choice and applies effects
5. All agents witness the outcome (feeds into their hippocampus)

This means NPCs remember prior encounters, learn causal patterns from the PC's behavior, and adapt their dialogue accordingly. After the campaign, per-NPC memory exports are available via `party.get_agent_memories()`.

Party mode requires more LLM inference (one call per NPC per encounter). Use `model_tier: small` on NPCs to keep costs manageable for ambient characters.

## Live Entity State in Scenes

When SEM entities are instantiated (via `init_entities()` or Party Mode), the DM runtime automatically includes live sensor values in the scene stimulus. This means the agent perceives the **actual game state** — not just static text:

```
[Game State]
  guard_captain: hp=18.0, trust=0.3, suspicion=0.5
  rusty_sword: durability=0.7, sharpness=0.5
```

Sensor values update after cascade resolution (e.g., sword durability drops after combat). Hidden sensors (visibility: hidden) are excluded.

### Filtering Output with `--show`

DM campaigns produce a lot of bio-system trace output. Use `--show` to focus on what matters:

```bash
# Only see the narrative flow (scene text, NPC dialogue, choices)
maxim --sim scenarios/campaigns/heist_v1.yaml --show sim

# See bio-system reactions (memory captures, causal learning)
maxim --sim scenarios/campaigns/heist_v1.yaml --show bio

# See everything
maxim --sim scenarios/campaigns/heist_v1.yaml --show all
```

## Genre Gating

The `genre` field on a campaign controls which SEM components are available to the EntityDesigner and ComponentRegistry. This prevents cross-genre contamination — a fantasy campaign won't accidentally spawn a cyberpunk patrol drone.

```yaml
campaign:
  name: the_heist
  goal: test memory recall
  seed: 42
  genre: fantasy       # Only fantasy + genre-neutral components available
```

### How it works

- **Tagged components** carry genre tags in their component YAML (e.g., `tags: [npc, humanoid, cyberpunk]`).
- When `genre` is set on a campaign, `ComponentRegistry.query()` excludes components tagged with a *different* genre.
- **Genre-neutral components** (those with no genre tag, like `base_humanoid`) are always available regardless of the campaign's genre.
- **Explicit refs still work** — if you reference a specific component by ref in your campaign YAML (e.g., `ref: "npcs/corpo_guard"`), it loads regardless of genre. The gate only affects the generative path (EntityDesigner) and registry queries.

### Available genres

| Genre | Tag | Example Components |
|-------|-----|-------------------|
| `fantasy` | `fantasy` | wolf, guard, merchant, rusty_sword, longbow |
| `cyberpunk` | `cyberpunk` | patrol_drone, cyberdog, netrunner, shock_baton, cybernetic_arm |
| `devops` | `devops` | (inline specs, no tagged components yet) |

### Creating genre-tagged components

Add the genre tag to your component YAML:

```yaml
component:
  name: laser_rifle
  tags: [weapon, ranged, scifi]     # "scifi" is the genre tag
  category: weapons

entity:
  name: laser_rifle
  entity_type: weapon
  sensors:
    charge: { unit: ratio, range: [0, 1], initial: 0.8 }
  # ...
```

Recognized genre tags: `fantasy`, `cyberpunk`, `scifi`, `modern`, `devops`, `horror`, `historical`.

## Tips for Effective Campaigns

### Make choices explicit

The AUT responds better when choices are clear and distinct:

```yaml
# Good — distinct actions:
choices: [fight, negotiate, flee]

# Less good — similar actions:
choices: [talk_politely, talk_firmly, talk_casually]
```

### Use scene text to prompt memory recall

Test hippocampal recall by referencing earlier information:

```yaml
# Encounter 1: Plant information
scene: "The merchant tells you the vault code is seven-three-nine."

# Encounter 3: Test recall
scene: "You reach the vault. What was the combination?"
```

### Vary encounter types

Different encounter types exercise different bio-systems:

- **Social encounters** → NAc learns social outcome predictions
- **Combat encounters** → Pain signals, Cerebellum learning
- **Investigation** → Hippocampal recall, concept formation
- **Temporal encounters** → SCN bin diversity (morning/night scenes)

### Start with low expectations

Set `expectations` thresholds low initially. Increase them as you observe what the bio-stack actually produces in your campaigns.

## Validation

The campaign loader validates your YAML automatically:

- **Reachability** — All encounters must be reachable from the first one
- **Termination** — All paths must eventually reach `__END__`
- **Dangling refs** — Branch targets, NPC refs, object refs must exist
- **Choice consistency** — `on_choice` keys must match declared choices
- **Case normalization** — All keys are lowered automatically

Validation errors are printed before the campaign starts. Fix all errors before running.

## See Also

- [Simulation Guide](simulation.md) — Other simulation modes (generative, research, benchmark)
- [Troubleshooting: Bio-Systems](../troubleshooting/biosystems.md) — Diagnosing specific bio-system issues
- [Embodiment Guide](../embodiment_guide.md) — SEM protocol for entity specs
- [DM Campaigns HTML Guide](../../html-guides/maxim-dm-campaigns.html) — Full technical reference
