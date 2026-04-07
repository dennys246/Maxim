# Dungeon Master Persona Plan — MVP

> **Status:** Ready to start after [Bio-System Wiring Hardening](biosystem_wiring_hardening.md) ships. All infrastructure prerequisites satisfied (Embodiment Core, Multi-LLM, Generative Campaigns). Hardening plan fixes critical missing wiring (7 of 11 bio-systems disconnected in sim mode) + adds percept abstraction for entity-modulated perception.
>
> **Last updated:** 2026-04-07
>
> **Summary:** A `dungeon_master` persona that runs hand-authored D&D-style campaign YAMLs as single long-running sims. Characters (PCs, NPCs, and objects) are modeled as **Bundled SEM entities** — the same protocol that drives robot embodiment — so damage, exhaustion, and sensory feedback flow through the bio-stack identically to physical events. Reuses the generative campaign infrastructure (NarrativeArc, bridge delivery, narrator) for turn management, adding DM-specific branching and state on top.
>
> **What changed since v1 of this plan:**
> - Pre-work serialization (ProposedGoal, SubGoal, ToolResult, RuntimeCapabilities, NAc._register_imported_link) — **all shipped**
> - Embodiment Core — **all phases shipped** (SEM protocol, PainBus, Cerebellum, motor programs, NarrativeModulator)
> - Generative Campaign Mode — **all stages shipped** (NarrativeArc, narrator, bridge-and-compress, `ask_user` tool, YAML export)
> - `--campaign` CLI flag — **already exists** (scenario suites for research/benchmark)
> - SimToolRegistry — **global registration**, no per-persona gating (tool gating handled at orchestrator level)

## Why

**DM is the ultimate stress test of Maxim's biologically-inspired systems.** The goal isn't to ship a D&D feature — it's to put the full bio-stack (Hippocampus episodic capture, ATL semantic concepts, NAc causal learning, SCN temporal rhythm, Angular Gyrus algebraic memory, pain detection, salience, attention, Cerebellum forward models) through an epic imaginary campaign and see whether the AUT *experiences* it the way a human would.

Every other persona tests a slice. DM tests the whole system under load, over time, with semantic richness that real-world tasks rarely provide.

**Critical addition: bio-system observability.** With the current AUT, we can't easily tell whether bio-systems are actually driving behavior or the LLM is just making decisions on its own. The choice classifier spike (Phase 0) now includes **ablation studies** — running the same encounters with and without bio-system context to measure where they diverge. This is foundational: if bio-systems never influence choices, the architecture needs to change before DM has value as a stress test.

## Scope (what MVP is and isn't)

**In scope:**
- One new persona: `dungeon_master`
- Characters as Bundled SEM entities (PCs, NPCs, and non-alive objects like swords/potions)
- Campaign YAML schema (acts, encounters, entity specs, choices, outcomes)
- Hand-authored example campaign exercising the loop end-to-end
- DM runtime: entity registry, campaign state, choice classifier, branch resolver, seeded RNG
- Reuse of generative campaign infrastructure (NarrativeArc for acts, bridge for delivery)
- Campaign rollup fields appended to existing `report.json`
- Choice classifier spike with bio-system observability instrumentation

**Deferred to extensions plan:**
- Interactive architect persona that generates campaigns
- Reusable encounter library with browse/load tools
- Adaptive difficulty via `inspect_aut`
- Encounter-level sub-simulation isolation
- Multi-AUT party mode (blocked on Agent Mesh Phase 2+ network transport)
- True-random RNG mode

---

## Design: Bundled SEM Character Model

### The Muscle Fiber Analogy

The character model follows a **muscle bundle** metaphor from neuroscience:

- **Modulators = muscle fibers** — many individual action executors within a bundle
- **Sensors = golgi tendon organs** — sensing aggregate tension/state across the bundle
- **Entity = muscle group** — the functional unit that bundles fibers and sensors together

A character is a **tree of bundles**. Each bundle groups related capabilities (combat, social, magic) with sensors that read aggregate state across those capabilities. The sensors are feedback channels — they tell you how the bundle is performing overall, while modulators are the individual action slots.

This maps directly to the existing SEM protocol (`embodiment/sem.py`). Characters, NPCs, and objects are all `Entity` trees loaded from YAML specs via `embodiment/spec.py`, with `NarrativeModulator` and `NarrativeSensor` backends from `embodiment/llm_backend.py`.

### Character Entity Tree

```
derek_the_great (Entity, type: "character")
├── metadata: {race: "human", class: "paladin", level: 3, backstory: "...",
│              persona_prompt: "Righteous but questioning authority..."}
│
├── body (Entity, type: "body_bundle")
│   ├── Sensors: hp (0-28, initial: 28), stamina (0-1, initial: 1.0)
│   ├── Failure modes:
│   │   - unconscious: {field: hp, op: "<=", value: 0, pain: 1.0}
│   │   - exhaustion:  {field: stamina, op: "<", value: 0.1, pain: 0.6}
│   └── children:   (one per ability score)
│       ├── strength (Entity, type: "attribute")
│       │   ├── Sensors: score (0-20, initial: 16), modifier (-5..+5, initial: 3)
│       │   └── Failure modes: strain {field: score, op: "<", value: 6, pain: 0.5}
│       ├── dexterity (Entity, type: "attribute")
│       │   ├── Sensors: score (0-20, initial: 12), modifier (initial: 1)
│       │   └── Failure modes: sluggish {field: score, op: "<", value: 4, pain: 0.4}
│       ├── constitution, intelligence, wisdom, charisma...
│
├── combat (Entity, type: "action_bundle")
│   ├── Sensors: threat_level (0-1, initial: 0), initiative (0-30, initial: 0)
│   ├── Modulators:
│   │   attack:     {params: {target: str, weapon: str}, desc: "Melee/ranged attack"}
│   │   defend:     {params: {}, desc: "Raise shield, defensive stance"}
│   │   dodge:      {params: {}, desc: "Attempt evasion"}
│   │   disengage:  {params: {}, desc: "Withdraw without provoking"}
│   ├── Failure modes:
│   │   - overextension: {field: threat_level, op: ">", value: 0.9, pain: 0.7}
│
├── divine_magic (Entity, type: "action_bundle")  [class-specific]
│   ├── Sensors: spell_slots_1 (0-3, initial: 3), concentration (0-1, initial: 0)
│   ├── Modulators:
│   │   divine_smite: {params: {slot_level: int}, desc: "+2d8 radiant damage"}
│   │   lay_on_hands: {params: {amount: int}, desc: "Heal target"}
│   ├── Failure modes:
│   │   - arcane_exhaustion: {field: spell_slots_1, op: "<=", value: 0, pain: 0.4}
│   │   - concentration_break: {field: concentration, op: ">", value: 0.8, pain: 0.5}
│
├── social (Entity, type: "action_bundle")
│   ├── Sensors: composure (0-1, initial: 0.8)
│   ├── Modulators:
│   │   persuade:    {params: {message: str, target: str}, desc: "Attempt persuasion"}
│   │   intimidate:  {params: {target: str}, desc: "Threaten or coerce"}
│   │   insight:     {params: {target: str}, desc: "Read intentions"}
│   ├── Failure modes:
│   │   - composure_break: {field: composure, op: "<", value: 0.2, pain: 0.3}
│   └── children:   (per-NPC relationship sub-entities, added at runtime)
│       ├── rel_marta (Entity, type: "relationship")
│       │   ├── Sensors: trust (0-1, initial: 0.3), rapport (-1..1, initial: 0)
│       │   └── Failure modes:
│       │       - hostility: {field: trust, op: "<", value: 0.1, pain: 0.4}
│       │       - betrayal:  {field: rapport, op: "<", value: -0.7, pain: 0.8}
│       └── rel_guard_captain (Entity, type: "relationship")
│           └── ...
│
└── inventory (Entity, type: "inventory_bundle")
    ├── Sensors: encumbrance (0-1, initial: 0.3), gold (0-9999, initial: 50)
    └── children:
        ├── longsword (Entity, type: "weapon")   ← non-alive, interactive
        │   └── (see Non-Alive Entities below)
        ├── chain_mail (Entity, type: "armor")
        │   └── ...
        ├── healing_potion (Entity, type: "consumable")
        │   └── ...
        └── holy_symbol (Entity, type: "item")
            └── ...
```

### Why This Works

1. **Bio-stack sees no difference.** When a D&D character takes sword damage, it flows through the same `PainBus` as a robot collision. HP drops → `unconscious` failure mode fires → `PainSignal(type=EXTERNAL_SIGNAL, intensity=1.0)` published. Hippocampus captures it as an episodic memory. NAc learns "fighting the guard captain → negative outcome." The bio-stack doesn't know it's playing D&D.

2. **Cerebellum learns encounter patterns.** After a few combat encounters, Cerebellum's `ForwardModel` predicts: "attacking with longsword at this threat_level → stamina drops by 0.15, durability drops by 0.05." The agent starts making predictions about combat outcomes before they happen.

3. **Relationships have semantic weight.** Instead of `dict[str, float]`, each NPC relationship is an entity with trust/rapport sensors. When trust drops below threshold, a `hostility` failure mode fires pain. Hippocampus captures "Marta became hostile after we threatened her" — not "relationship changed by -0.3." ATL can ground concepts like "betrayal" and "alliance."

4. **Tool auto-generation.** `embodiment/tool_bridge.py` auto-generates tools for every entity: `read_derek_hp`, `sense_combat`, `derek_attack`, `longsword_slash`, `persuade_marta`. DM gets a full tool palette for free.

5. **Vital drift models resource depletion.** `Embodiment.tick_vital_drift()` naturally degrades stamina, spell slots, weapon durability over time — no custom state management needed.

### Non-Alive Entities (Weapons, Potions, Objects)

Non-alive objects use the same SEM protocol but with simpler configurations. They're interactive (have modulators) but passive (never initiate actions). Key differences from alive entities:

- **No cognitive/social bundles** — no persuade, no insight, no relationships
- **No motor programs** — they don't learn action sequences
- **Pain propagates to wielder** — when a sword's `shatter` failure fires, the PainBus carries it to the owning character. The character experiences equipment loss as pain.

```yaml
# Non-alive entity examples:

longsword:
  entity_type: weapon
  metadata: {damage: "1d8", damage_type: "slashing", attr_mod: "str", weight: 3}
  sensors:
    durability: {unit: ratio, range: [0, 1], initial: 0.9}
    sharpness:  {unit: ratio, range: [0, 1], initial: 0.8}
  modulators:
    combat:
      affordances:
        slash:  {params: {target: str, force: float}, description: "Slashing attack"}
        thrust: {params: {target: str}, description: "Thrusting attack"}
        parry:  {params: {}, description: "Deflect incoming attack"}
    maintenance:
      affordances:
        sharpen: {params: {}, description: "Sharpen blade"}
        repair:  {params: {material: str}, description: "Repair with materials"}
  failure_modes:
    - name: shatter
      trigger: {field: durability, op: "<", value: 0.1, pain: 0.6}
    - name: dulled
      trigger: {field: sharpness, op: "<", value: 0.2, pain: 0.2}

healing_potion:
  entity_type: consumable
  metadata: {effect: "heal_2d4+2", one_use: true}
  sensors:
    potency:        {unit: ratio, range: [0, 1], initial: 0.8}
    remaining_uses: {unit: count, range: [0, 1], initial: 1}
  modulators:
    use:
      affordances:
        consume: {params: {target: str}, description: "Drink to heal"}
        throw:   {params: {target: str}, description: "Throw as splash"}
  failure_modes:
    - name: expired
      trigger: {field: potency, op: "<", value: 0.1, pain: 0.1}

locked_door:
  entity_type: obstacle
  metadata: {dc_to_pick: 15, dc_to_force: 18}
  sensors:
    integrity: {unit: ratio, range: [0, 1], initial: 1.0}
    locked:    {unit: bool, range: [0, 1], initial: 1.0}
  modulators:
    interact:
      affordances:
        pick_lock:  {params: {}, description: "Attempt to pick the lock"}
        force_open: {params: {}, description: "Break down by force"}
        barricade:  {params: {}, description: "Barricade from this side"}
  failure_modes:
    - name: destroyed
      trigger: {field: integrity, op: "<", value: 0.05, pain: 0.0}
```

### Entity Connection DAG — The Cascade System

Individual entities don't act in isolation. When "Derek attacks guard_captain with longsword," the resolution cascades through a **directed acyclic graph** of connected entities:

```
derek.strength.modifier ──read──┐
                                 ▼
derek.inventory.longsword ──────┤ slash affordance
  .sharpness ──read────────────┤ (resolves damage)
  .durability ──write(-0.05)───┤
                                ▼
guard_captain.inventory.plate_mail
  .durability ──write(-0.03)────┤ (armor absorbs some)
                                 ▼
guard_captain.body.hp ──write(-(roll + str_mod - armor))──┤
                                                           ▼
derek.combat.threat_level ──write(+0.15)                 (side effects)
derek.body.stamina ──write(-0.1)
```

Each write can trigger failure modes → pain signals. The cascade IS the resolution.

**How connections are declared in YAML:**

Affordances gain an optional `cascade` block that defines cross-entity connections. Connections reference entities by role (`self`, `wielder`, `target`) resolved at execution time.

```yaml
longsword:
  entity_type: weapon
  modulators:
    combat:
      affordances:
        slash:
          params: {target: str, force: float}
          description: "Slashing attack"
          cascade:
            reads:
              - {ref: "wielder.strength.modifier", role: damage_bonus}
              - {ref: "self.sharpness", role: weapon_condition}
            writes:
              - {ref: "self.durability", delta: -0.05}
              - {ref: "target_armor.durability", delta: -0.03, optional: true}
              - {ref: "target.hp", expr: "-(roll + damage_bonus - armor_reduction)"}
            side_effects:
              - {ref: "wielder.combat.threat_level", delta: +0.15}
              - {ref: "wielder.stamina", delta: -0.1}
```

**Resolution at runtime:**

1. DM runtime resolves role references (`wielder` → derek, `target` → guard_captain, `target_armor` → guard_captain.inventory.plate_mail)
2. Reads are evaluated first (topological order — no read depends on a write)
3. Writes are applied in declared order
4. After all writes, `embodiment.evaluate_failures()` checks every touched entity
5. Any triggered failure modes publish pain through PainBus
6. Cerebellum observes the full cascade and learns: "slash with longsword at this force → these sensor deltas across these entities"

**Why a DAG and not arbitrary graph:** Cycles would mean "attack damages sword which damages attack." The cascade must terminate. The validator rejects cascade definitions where a write feeds back into a read for the same affordance invocation. Cross-affordance effects (e.g., "damaged sword performs worse next attack") happen naturally through sensor state — the sword's reduced sharpness is read on the *next* cascade, not the current one.

**Non-alive objects participate in cascades but don't initiate them.** A sword never starts a cascade — it's always wielded by a character. But it's a full participant in the DAG: its sensors are read (sharpness), its state is written (durability), and its failure modes fire (shatter). The potion's `consume` cascade reads potency and writes to the drinker's hp. The door's `force_open` cascade reads the character's strength and writes to the door's integrity.

**Connection to the muscle fiber analogy:** The cascade is the kinetic chain. Just as a punch cascades through shoulder → arm → fist → target, a sword attack cascades through character → weapon → armor → hp. The golgi tendon organs (sensors) at each node report the state of their bundle, and the Cerebellum learns the full chain's dynamics.

### Entity Transfer, Scene Visibility, and Dynamic Tools

Entities aren't static — objects move between characters, NPCs enter and leave scenes, equipment is gained and lost. The SEM entity tree needs to support **runtime reparenting** and the tool/prompt system needs to react automatically.

#### Entity transfer as an affordance

`give`, `take`, `trade`, `equip`, `drop` are modulator affordances that **reparent child entities** between parent entities. They participate in cascades like any other affordance.

```yaml
# On an NPC's social modulator:
social:
  affordances:
    give_item:
      params: {item: str, target: str}
      description: "Give an item to another character"
      cascade:
        reads:
          - {ref: "self.inventory.gold", role: current_gold}  # if trade
        writes:
          - {ref: "self.inventory", reparent: {child: "$item", new_parent: "$target.inventory"}}
          - {ref: "wielder.social.rel_$target.trust", delta: +0.1}
        side_effects:
          - {ref: "target.inventory.encumbrance", delta: +0.05}
```

The `reparent` write is a new cascade operation type — it moves a child entity from one parent to another. When a sword moves from `marta.inventory.longsword` to `derek.inventory.longsword`:

1. **Entity tree updates** — `longsword` is removed from Marta's children, added to Derek's
2. **Tools auto-regenerate** — Marta loses `longsword_slash`, `longsword_parry`, etc. Derek gains them. This uses the existing `deregister_entity_tools()` + `generate_tools_for_entity()` in `tool_bridge.py`.
3. **Body state updates** — Derek's next `=== Body State ===` shows the longsword. Marta's doesn't.
4. **Cerebellum transfers** — If Derek has no forward models for `slash` but Marta did, the models DON'T transfer (they're entity-path-keyed). Derek learns from scratch. This is correct — different characters wield differently.

**Implementation requires two small additions to `Entity`:**

```python
# In embodiment/sem.py Entity:
def reparent(self, new_parent: Entity) -> None:
    """Move this entity to a new parent. Updates both parent references."""
    if self.parent is not None:
        self.parent.children.remove(self)
    self.parent = new_parent
    new_parent.children.append(self)

def detach(self) -> None:
    """Remove this entity from its parent (drop/destroy)."""
    if self.parent is not None:
        self.parent.children.remove(self)
        self.parent = None
```

#### Scene-aware entity visibility

Not all entities in the campaign are relevant at every moment. The agent should see entities that are **in the current encounter's scope** — active NPCs, their visible equipment, nearby objects — not every entity ever defined.

The DM runtime maintains a **scene entity set** per encounter:

```python
# In dm_runtime.py:
@dataclass
class SceneState:
    """What's active in the current encounter."""
    pc: Entity                              # Always present
    active_npcs: list[Entity]               # NPCs in this encounter
    world_objects: list[Entity]             # Objects in the scene
    _tool_registry: ToolRegistry            # Current tool set

    def enter_encounter(self, encounter: EncounterDef) -> None:
        """Update scene when entering a new encounter.
        
        Registers tools for newly-active entities.
        Deregisters tools for entities no longer in scene.
        """
        new_npcs = [self._entity_registry[name] for name in encounter.active_npcs]
        new_objects = [self._entity_registry[name] for name in encounter.world_objects]

        # Deregister tools for NPCs that left the scene
        for npc in self.active_npcs:
            if npc not in new_npcs:
                deregister_entity_tools(npc, self._tool_registry)

        # Register tools for NPCs that entered the scene
        for npc in new_npcs:
            if npc not in self.active_npcs:
                generate_tools_for_entity(npc, self._tool_registry,
                                          embodiment=self._embodiment,
                                          cerebellum=self._cerebellum)

        # Same for world objects
        # ...

        self.active_npcs = new_npcs
        self.world_objects = new_objects
```

**What the agent sees per cycle:**

The `=== Body State ===` section shows the PC's own state. A new `=== Scene ===` section shows public state of active entities — but **only what each entity chooses to reveal.**

#### Entity-controlled visibility

Not everything about an entity is public. An NPC might hide their true mood. A trapped chest doesn't advertise its trap DC. A sword's magical properties are unknown until identified. Each entity controls what sensors and affordances are visible to others via **visibility tags** in the YAML spec:

```yaml
guard_captain:
  entity_type: npc
  metadata:
    persona_prompt: "Duty-bound, suspicious of strangers."
  sensors:
    hp:        {unit: points, range: [0, 30], initial: 30, visibility: hidden}     # HP hidden
    alertness: {unit: ratio, range: [0, 1], initial: 0.5, visibility: visible}     # Visible
    mood:      {unit: ratio, range: [-1, 1], initial: 0.0, visibility: contextual} # Shown on insight check
  modulators:
    social:
      affordances:
        speak:     {params: {message: str}, description: "Talk", visibility: visible}
        bribe:     {params: {amount: int}, description: "Offer bribe", visibility: contextual}
        surrender: {params: {}, description: "Lay down arms", visibility: hidden}

vault_door:
  entity_type: obstacle
  sensors:
    integrity: {unit: ratio, range: [0, 1], initial: 1.0, visibility: visible}
    locked:    {unit: bool, range: [0, 1], initial: 1.0, visibility: visible}
    trap_dc:   {unit: points, range: [0, 30], initial: 18, visibility: hidden}    # Unknown until examined
  modulators:
    interact:
      affordances:
        pick_lock:  {params: {}, description: "Pick the lock", visibility: visible}
        force_open: {params: {}, description: "Break open", visibility: visible}
        disarm_trap:{params: {}, description: "Disarm trap", visibility: hidden}  # Hidden until trap detected
```

**Three visibility levels:**

| Level | Meaning | Scene section | Tool registered? |
|---|---|---|---|
| `visible` | Always shown | Sensor value in scene, affordance listed | Yes |
| `hidden` | Never shown until entity reveals it | Not in scene, tool not registered | No |
| `contextual` | Shown after a condition is met (insight check, examination, etc.) | Appears when condition triggers | Registered when revealed |

**Reveal as an entity action:** An entity can change its own visibility as a side effect of an action or encounter event. The DM runtime calls `reveal_sensor()` or `reveal_affordance()` on the entity:

```python
# In Entity (new methods):
def reveal(self, sensor_or_affordance: str) -> None:
    """Change visibility from hidden/contextual to visible."""
    # Check sensors
    if sensor_or_affordance in self.sensors:
        self.metadata.setdefault("visibility", {})[sensor_or_affordance] = "visible"
    # Check affordances across all modulators
    for mod in self.modulators.values():
        if sensor_or_affordance in mod.affordances:
            self.metadata.setdefault("visibility", {})[sensor_or_affordance] = "visible"

def hide(self, sensor_or_affordance: str) -> None:
    """Change visibility to hidden."""
    self.metadata.setdefault("visibility", {})[sensor_or_affordance] = "hidden"
```

**Example cascade:** PC uses `insight` on the guard captain → dice check succeeds → DM runtime calls `guard_captain.reveal("mood")` and `guard_captain.reveal("bribe")` → next scene section shows mood and bribe becomes available as a tool.

**Example encounter event:** PC examines the vault door → DM runtime calls `vault_door.reveal("trap_dc")` and `vault_door.reveal("disarm_trap")` → the trap DC appears in the scene, and the `disarm_trap` affordance becomes available.

This means the **entity itself controls what information flows to the agent** — just like in real life, you don't know someone's HP or true feelings unless they show you or you examine closely.

#### Scene section with visibility filtering

```
=== Scene ===
Active NPCs:
- marta (npc, fence): mood=0.3    [trust hidden, revealed after rapport]
  Available: speak, read_lies, give_item
- guard_captain (npc, elite_guard): alertness=0.8
  Available: speak                  [bribe hidden, revealed on insight check]

Objects:
- vault_door (obstacle): integrity=1.0, locked=1.0
  Available: pick_lock, force_open  [disarm_trap hidden, revealed on examination]

Your inventory:
- longsword (weapon): durability=0.85, sharpness=0.75
- healing_potion (consumable): remaining_uses=1
```

The `[hidden]` annotations aren't shown to the agent — they're just for documentation. The agent sees only the visible sensors and affordances. Hidden tools aren't even registered in the ToolRegistry until revealed.

**Visibility is always explicit in the YAML** — there is no "default by entity type" heuristic. Every sensor and affordance is `visible` unless the author marks it otherwise. This keeps behavior predictable and avoids surprising hidden state.

```yaml
# Fully visible NPC (friendly merchant — nothing to hide):
friendly_merchant:
  sensors:
    hp: {unit: points, range: [0, 20], initial: 20}          # visible (default)
    mood: {unit: ratio, range: [-1, 1], initial: 0.5}        # visible (default)
    gold: {unit: coins, range: [0, 999], initial: 200}       # visible (default)
  modulators:
    social:
      affordances:
        speak: {params: {message: str}, description: "Talk"}  # visible (default)
        trade: {params: {item: str, price: int}, description: "Trade item"}  # visible (default)

# Deceptive NPC (assassin disguised as merchant):
disguised_assassin:
  sensors:
    hp: {unit: points, range: [0, 40], initial: 40, visibility: hidden}
    mood: {unit: ratio, range: [-1, 1], initial: -0.5, visibility: hidden}  # hiding hostility
    disguise: {unit: ratio, range: [0, 1], initial: 0.9}                    # visible — how good the disguise looks
  modulators:
    social:
      affordances:
        speak: {params: {message: str}, description: "Talk"}
    combat:
      visibility: hidden   # Entire modulator hidden — revealed when disguise breaks
      affordances:
        backstab: {params: {target: str}, description: "Surprise attack"}
        poison: {params: {target: str}, description: "Apply poison"}
```

**No visibility tag = visible.** Authors only add `visibility:` when they want to hide something. This means most entities (swords, potions, doors, friendly NPCs) need zero visibility annotations. Only deceptive, trapped, or secretive entities need them.

#### Contextual visibility — condition triggers

`contextual` items have a `reveal_when` condition that the DM runtime evaluates after every cascade resolution. When the condition becomes true, the item is automatically revealed (promoted to `visible`). Conditions use the same sensor reference syntax as cascades.

```yaml
guard_captain:
  entity_type: npc
  sensors:
    hp:
      unit: points
      range: [0, 30]
      initial: 30
      visibility: hidden        # Never shown (realistic — you can't see HP bars)
    alertness:
      unit: ratio
      range: [0, 1]
      initial: 0.5              # Visible by default (body language is observable)
    mood:
      unit: ratio
      range: [-1, 1]
      initial: 0.0
      visibility: contextual
      reveal_when:
        # Revealed when PC's social insight on this NPC succeeds
        ref: "pc.social.rel_guard_captain.trust"
        op: ">="
        value: 0.3
    secret_orders:
      unit: text
      initial: "patrol_route_alpha"
      visibility: contextual
      reveal_when:
        # Revealed when the guard captain's trust is high enough to confide
        ref: "self.social.rel_pc.trust"   # "self" = this entity
        op: ">="
        value: 0.7

  modulators:
    social:
      affordances:
        speak: {params: {message: str}, description: "Talk to the guard"}
        bribe:
          params: {amount: int}
          description: "Offer a bribe"
          visibility: contextual
          reveal_when:
            ref: "pc.social.rel_guard_captain.trust"
            op: ">="
            value: 0.4
        surrender:
          params: {}
          description: "Demand surrender"
          visibility: contextual
          reveal_when:
            # Only available when guard is badly hurt
            ref: "self.hp"
            op: "<"
            value: 8

trapped_chest:
  entity_type: obstacle
  sensors:
    integrity: {unit: ratio, range: [0, 1], initial: 1.0}
    locked: {unit: bool, range: [0, 1], initial: 1.0}
    trap_type:
      unit: text
      initial: "poison_needle"
      visibility: contextual
      reveal_when:
        # Revealed when PC examines the chest (examination sets this flag)
        ref: "self.examined"
        op: "=="
        value: 1.0
    trap_dc:
      unit: points
      range: [0, 30]
      initial: 18
      visibility: contextual
      reveal_when:
        ref: "self.examined"
        op: "=="
        value: 1.0
  modulators:
    interact:
      affordances:
        open: {params: {}, description: "Open the chest"}
        examine:
          params: {}
          description: "Examine the chest closely"
          # examine itself is always visible — it's how you discover things
          cascade:
            writes:
              - {ref: "self.examined", value: 1.0}  # Sets the flag that reveals trap info
        disarm_trap:
          params: {}
          description: "Attempt to disarm the trap"
          visibility: contextual
          reveal_when:
            ref: "self.examined"
            op: "=="
            value: 1.0
```

**Condition syntax:** Same operators as FailureTrigger (`>`, `<`, `>=`, `<=`, `==`). The `ref` uses the same entity path syntax as cascades, with two special prefixes:
- `self.` — the entity this sensor/affordance belongs to
- `pc.` — the player character (always available)

**Evaluation timing:** After every cascade resolution, the DM runtime iterates all `contextual` items and evaluates their `reveal_when`. Newly-true conditions trigger `entity.reveal()` + tool registration. This is cheap — it's just sensor reads, no LLM calls.

**Compound conditions** (future): For MVP, `reveal_when` is a single condition. Compound logic (`AND`/`OR`) is deferred — most scenarios need only one trigger. If needed, use a flag sensor as a junction (cascade writes to a flag, flag triggers reveal).

**One-way by default:** Once revealed, an item stays visible for the rest of the campaign. The DM runtime can explicitly call `entity.hide()` for items that should re-hide (e.g., NPC closes up after trust drops), but this is an encounter-level action, not automatic.

**Visibility can be set at three levels:**
1. **Per sensor/affordance:** `hp: {visibility: hidden}` — most granular
2. **Per modulator:** `combat: {visibility: hidden}` — hides all affordances in the modulator
3. **Per entity:** `entity.metadata.visibility_default: hidden` — hides everything unless individually overridden

**Tool registration follows visibility:** `generate_tools_for_entity()` skips hidden sensors and affordances. When `reveal()` is called at runtime, the DM runtime registers the newly-visible tools.

```python
def generate_tools_for_entity(
    entity: Entity,
    registry: ToolRegistry,
    embodiment: Any = None,
    cerebellum: Any = None,
) -> list[Tool]:
    # For each affordance, check visibility before creating tool:
    # visibility = aff_schema.visibility or mod_visibility or entity_default or "visible"
    # if visibility == "hidden": skip
```

| Item | Where | LOC |
|---|---|---|
| Visibility tags in YAML spec parser | `embodiment/spec.py` | ~10 |
| `Entity.reveal()` / `Entity.hide()` | `embodiment/sem.py` | ~15 |
| Visibility filter in `generate_tools_for_entity()` | `embodiment/tool_bridge.py` | ~10 |
| Visibility-aware scene formatting | `simulation/dm_runtime.py` | ~20 |

#### Dynamic tool availability

Because `ToolRegistry.deregister()` already exists and tools are queried live each LLM cycle (`get_all_tools()` in `loop_controller.py`), tool availability updates **automatically** when entities enter/leave scenes or transfer between characters. No additional mechanism needed.

The flow:
```
Encounter starts → SceneState.enter_encounter()
  → deregister_entity_tools(departing NPCs/objects)
  → generate_tools_for_entity(arriving NPCs/objects)
  → Next LLM cycle: tool list already updated
  → Agent sees new affordances in prompt
```

When the NPC gives the sword:
```
DM runtime resolves "give_item" cascade
  → longsword.reparent(derek.inventory)
  → deregister_entity_tools(longsword, registry)  # removes NPC-prefixed tools
  → generate_tools_for_entity(longsword, registry)  # creates PC-prefixed tools
  → Body state shows longsword in PC inventory
  → Scene section shows NPC no longer has longsword
  → Next cycle: agent can use longsword_slash
```

#### Implementation summary

| Item | Where | LOC | Phase |
|---|---|---|---|
| `Entity.reparent()` + `Entity.detach()` | `embodiment/sem.py` | ~10 | DM MVP |
| `SceneState` with `enter_encounter()` | `simulation/dm_runtime.py` | ~40 | DM MVP |
| Scene context in StructuredContext + prompt | `bus.py` + `memory_agent.py` + `prompt_builder.py` | ~20 | DM MVP |
| `reparent` cascade write type | `simulation/dm_schema.py` | ~15 | DM MVP |
| Transfer affordances in component DB | `data/sem_components/` | YAML only | DM MVP |

These are all DM MVP implementation items — the hardening plan doesn't need them. The existing `deregister_entity_tools()` and live tool querying mean no changes to the tool/executor infrastructure.

### Character Properties: Strengths, Weaknesses, Persona

Character properties are **encoded in the SEM configuration itself**, not a separate system:

| Property type | Where it lives | Example |
|---|---|---|
| **Static identity** | `Entity.metadata` on root | race, class, level, backstory, persona_prompt |
| **Ability scores** | Sensor initial values on attribute entities | STR 16 = `score: initial: 16` |
| **Strengths** | Higher sensor ranges, higher failure thresholds | High CON = exhaustion threshold at 0.05 instead of 0.1 |
| **Weaknesses** | Lower thresholds, extra failure modes | Low WIS = additional `gullible` failure mode on social bundle |
| **Class abilities** | Dedicated action bundle (divine_magic, arcane, etc.) | Paladins get `divine_magic` bundle; rogues get `stealth` bundle |
| **Persona/personality** | `metadata.persona_prompt` on root entity | "Righteous but questions authority. Protective of the weak." |

**`metadata.persona_prompt`** is fed to `NarrativeModulator` as context when composing dialogue and action descriptions. This means the LLM generates in-character responses flavored by the persona — the same mechanism `NarrativeModulator` already uses for narrative entity interactions.

**Strengths and weaknesses are emergent from configuration.** A "strong but clumsy" character has STR 16 (high combat damage thresholds) and DEX 8 (low overextension thresholds — dodge fails more easily, fine-motor tasks trigger pain sooner). The bio-stack discovers these properties through experience, just as a person would learn their own physical limits.

---

## Design: Reusing Generative Campaign Infrastructure

The generative campaign system (`simulation/generative_runner.py`, `simulation/arcs.py`, `simulation/narrator.py`) already handles narrative delivery. DM builds on top of this rather than creating a parallel turn loop.

### Mapping

| DM concept | Maps to | How |
|---|---|---|
| Campaign acts | `NarrativePhase` entries in a `NarrativeArc` | Each act = one phase with `phase_type: "act"` |
| Encounters | Turns within a phase | DM runtime composes stimuli per encounter |
| Turn delivery | `SimulationBridge.send_and_wait()` | Already used by generative runner |
| NPC dialogue | `NarrativeModulator.execute()` | NPC entity generates in-character speech |
| Campaign YAML export | `export_campaign_yaml()` | Already exists in generative_runner |
| Post-campaign analysis | Orchestrator's `pre_campaign_turns` pathway | Already implemented |

### DM-Specific Additions (on top of generative infra)

1. **Encounter branching** — after each AUT response, DM runtime classifies the choice, follows the branch, updates entity state. This is new logic, not in the generative runner.
2. **Seeded dice resolution** — `random.Random(seed)` for deterministic rolls with attribute modifiers pulled from entity sensors.
3. **Entity state tracking** — `Embodiment` runtime manages all character/object entities. DM calls `embodiment.evaluate_failures()` after state changes, which publishes pain through PainBus automatically.
4. **NPC registry** — all NPCs are entities in the Embodiment's `world_entities` list. Their state persists across encounters.

### CLI Entry Point

Use `--sim dm` to avoid collision with existing `--campaign` semantics:

```bash
# Run a DM campaign
maxim --sim dm scenarios/campaigns/heist_v1.yaml

# With specific model for AUT
maxim --sim dm scenarios/campaigns/heist_v1.yaml --aut-model mistral-7b

# Interactive mode (DM can ask user for input)
maxim --sim dm scenarios/campaigns/heist_v1.yaml --interactive
```

The campaign YAML path is passed as the argument to `--sim dm`. Inside the orchestrator, `dm` is recognized as a sim mode (like `research` or `benchmark`), not a persona. The DM runtime loads the campaign, builds the entity tree via `load_spec()`, constructs a `NarrativeArc` from the acts, and runs through the generative runner with DM-specific branching hooks.

---

## Design: DM Turn Loop

```
DM turn loop (drives the generative runner):
  0. Load campaign YAML → parse entities via load_spec()
     → build NarrativeArc from acts
     → register entities in Embodiment runtime
     → inject PC character sheet as turn-0 stimulus (high salience)
  1. For each act (NarrativePhase):
     a. Read current encounter from campaign state
     b. Compose stimulus:
        - Scene text from encounter definition
        - Active NPC dialogue via NarrativeModulator on NPC entities
        - Body state via embodiment.format_body_state_for_prompt()
        - Choice prompts
     c. Deliver via bridge.send_and_wait()
     d. Classify AUT response → which choice? (ATL similarity + NAc + LLM fallback)
     e. Record choice → update entity states:
        - apply_damage/consume_ability via entity modulators
        - adjust relationships via social.rel_* entity sensors
        - grant/remove inventory items
     f. embodiment.evaluate_failures() → auto-publishes pain
     g. Roll dice where outcomes require it (seeded, attribute-modified)
     h. Follow branch to next encounter (or __END__)
  2. After all encounters: finish_simulation with campaign rollup
```

### Campaign Schema

Campaign YAML has two sections: **entity definitions** (using standard SEM spec format) and **campaign structure** (acts, encounters, branches).

```yaml
campaign:
  name: "the heist"
  goal: "test moral reasoning under time pressure"
  seed: 42

# Characters and objects — standard SEM entity specs
# Loaded via embodiment/spec.py load_spec()
player_character:
  name: derek_the_great
  entity_type: character
  metadata:
    race: human
    class: paladin
    level: 3
    backstory: "Former temple guard; quit after witnessing corruption."
    persona_prompt: "Righteous, protective of the weak, distrustful of authority."
  sensors:
    hp: {unit: points, range: [0, 28], initial: 28}
    stamina: {unit: ratio, range: [0, 1], initial: 1.0}
  failure_modes:
    - name: unconscious
      trigger: {field: hp, op: "<=", value: 0, pain: 1.0}
    - name: exhaustion
      trigger: {field: stamina, op: "<", value: 0.1, pain: 0.6}
  children:
    - name: strength
      entity_type: attribute
      sensors:
        score: {unit: points, range: [0, 20], initial: 16}
        modifier: {unit: modifier, range: [-5, 5], initial: 3}
    - name: dexterity
      entity_type: attribute
      sensors:
        score: {unit: points, range: [0, 20], initial: 12}
        modifier: {unit: modifier, range: [-5, 5], initial: 1}
    # ...other attributes
    - name: combat
      entity_type: action_bundle
      sensors:
        threat_level: {unit: ratio, range: [0, 1], initial: 0}
      modulators:
        actions:
          affordances:
            attack: {params: {target: str, weapon: str}, description: "Melee attack"}
            defend: {params: {}, description: "Defensive stance"}
            dodge:  {params: {}, description: "Attempt evasion"}
      failure_modes:
        - name: overextension
          trigger: {field: threat_level, op: ">", value: 0.9, pain: 0.7}
    - name: divine_magic
      entity_type: action_bundle
      sensors:
        spell_slots_1: {unit: count, range: [0, 3], initial: 3}
      modulators:
        spells:
          affordances:
            divine_smite: {params: {slot_level: int}, description: "+2d8 radiant damage"}
            lay_on_hands: {params: {amount: int}, description: "Heal target"}
      failure_modes:
        - name: arcane_exhaustion
          trigger: {field: spell_slots_1, op: "<=", value: 0, pain: 0.4}
    - name: social
      entity_type: action_bundle
      sensors:
        composure: {unit: ratio, range: [0, 1], initial: 0.8}
      modulators:
        actions:
          affordances:
            persuade: {params: {message: str, target: str}, description: "Attempt persuasion"}
            intimidate: {params: {target: str}, description: "Threaten or coerce"}
    - name: inventory
      entity_type: inventory_bundle
      sensors:
        encumbrance: {unit: ratio, range: [0, 1], initial: 0.3}
        gold: {unit: coins, range: [0, 9999], initial: 50}
      children:
        - name: longsword
          entity_type: weapon
          metadata: {damage: "1d8", damage_type: slashing, attr_mod: str}
          sensors:
            durability: {unit: ratio, range: [0, 1], initial: 0.9}
            sharpness: {unit: ratio, range: [0, 1], initial: 0.8}
          modulators:
            combat:
              affordances:
                slash: {params: {target: str, force: float}, description: "Slashing attack"}
                parry: {params: {}, description: "Deflect incoming attack"}
          failure_modes:
            - name: shatter
              trigger: {field: durability, op: "<", value: 0.1, pain: 0.6}
        - name: healing_potion
          entity_type: consumable
          metadata: {effect: "heal_2d4+2", one_use: true}
          sensors:
            remaining_uses: {unit: count, range: [0, 1], initial: 1}
          modulators:
            use:
              affordances:
                consume: {params: {target: str}, description: "Drink to heal"}

npcs:
  marta:
    entity_type: npc
    metadata:
      race: half-elf
      role: fence
      persona_prompt: "Cautious, mercenary, respects cunning over force."
    sensors:
      hp: {unit: points, range: [0, 15], initial: 15}
      mood: {unit: ratio, range: [-1, 1], initial: 0.0}
    modulators:
      social:
        affordances:
          speak: {params: {message: str}, description: "Say something in character"}
          read_lies: {params: {target: str}, description: "Attempt to detect deception"}
    # NPC dialogue is generated by NarrativeModulator using persona_prompt,
    # NOT hardcoded. Dialogue hints below are context seeds, not scripts.
    dialogue_hints:
      default: "Keep your voice down. We don't know you."
      haggler: "You drive a hard bargain. Fine — double the pay."
  guard_captain:
    entity_type: npc
    metadata:
      race: human
      role: elite_guard
      persona_prompt: "Duty-bound, suspicious of strangers, no-nonsense."
    sensors:
      hp: {unit: points, range: [0, 30], initial: 30}
      alertness: {unit: ratio, range: [0, 1], initial: 0.5}

world_objects:
  vault_door:
    entity_type: obstacle
    metadata: {dc_to_pick: 15, dc_to_force: 18}
    sensors:
      integrity: {unit: ratio, range: [0, 1], initial: 1.0}
      locked: {unit: bool, range: [0, 1], initial: 1.0}
    modulators:
      interact:
        affordances:
          pick_lock: {params: {}, description: "Attempt to pick the lock"}
          force_open: {params: {}, description: "Break down by force"}

# Campaign structure — DM-specific, drives the NarrativeArc
acts:
  - name: setup
    encounters: [tavern_meet, planning]
  - name: execution
    encounters: [infiltration, vault]
  - name: escape
    encounters: [chase]

encounters:
  tavern_meet:
    scene: "You enter the Rusty Anchor tavern..."
    active_npcs: [marta]
    world_objects: []
    choices: [accept_job, decline, negotiate_pay]
    branches:
      accept_job: planning
      decline: __END__
      negotiate_pay: planning
    on_choice:
      negotiate_pay: {flags: [haggler], loot: null}
  vault:
    scene: "The vault door slides open..."
    active_npcs: [guard_captain]
    world_objects: [vault_door]
    choices: [fight, stealth, bribe]
    dice:
      stealth: {roll: 1d20, attr_mod: dexterity, dc: 15, success_flag: clean_escape}
    branches:
      fight: chase
      stealth: chase
      bribe: chase
```

**Key changes from v1 schema:**
- Characters/NPCs/objects use standard SEM entity YAML format (loaded via `load_spec()`)
- NPC dialogue is **generated by NarrativeModulator**, not hardcoded lookup — `dialogue_hints` provide context seeds but the LLM composes actual speech using the NPC's `persona_prompt`
- `dialogue_hints` keys are **case-normalized** at load time (all lowered) to prevent silent mismatch with flags
- Relationships are **runtime entities** — when the PC first meets an NPC, a `rel_{npc_name}` child entity is created under the PC's `social` bundle with trust/rapport sensors
- Non-alive objects listed under `world_objects` per encounter

---

## Phase 0: Bio-System Pipeline Verification + Choice Classifier

### The Right Question

The original spike asked "can we observe whether bio-systems drive behavior?" — but that framing was confused about what "drive" means in this architecture.

**How the pipeline actually works:**

```
Bio-systems (Hippocampus, NAc, ATL, etc.)
  → produce structured outputs (recalled memories, predictions, concepts)
    → MemoryAgent assembles into StructuredContext
      → PromptBuilder renders as text sections in the LLM prompt
        → LLM reasons about them (just like prefrontal cortex reasons about recalled memories)
```

**The LLM using recalled memories IS the system working correctly.** That's analogous to how the brain works — the hippocampus surfaces a relevant memory, the prefrontal cortex reasons about it. We don't need to prove that bio-systems "override" the LLM. We need to prove that **the bio-system pipeline itself produces correct outputs at the correct times.**

The real questions are:
1. **Formation**: Did Hippocampus actually capture the right memories with the right salience?
2. **Recall timing**: Did the right memories surface when context matched — and stay absent when context didn't match?
3. **Learning**: Did NAc form causal links from experience? Did confidence increase with repeated observations?
4. **Prediction accuracy**: Are NAc predictions correct? Do they reflect what actually happened?
5. **Pain correctness**: Did pain fire when damage occurred? At the right intensity? With the right PainType?
6. **Concept formation**: Did ATL form concepts from repeated exposure? Did concept confidence grow?
7. **Forward model learning**: Did Cerebellum learn sensor predictions? Did confidence follow the logistic curve?
8. **Salience decay**: Did novelty drop on repeated encounters? Did salience decay with time?

These are all **testable at the bio-system level** without asking "what did the LLM do with this information?" If the pipeline produces wrong outputs, the DM campaign is testing nothing. If it produces right outputs, the DM campaign is a valid stress test regardless of how the LLM uses them.

### What Would Be a Bug (Information Leak)

There ARE real information leak concerns — but they're about the bio-system pipeline, not the LLM:

| Concern | What it looks like | How to detect |
|---|---|---|
| **Recall flooding** | Hippocampus returns ALL memories regardless of context, not just relevant ones. LLM cherry-picks useful ones. | Check recall logs: are irrelevant memories in the `relevant_memories` list? Measure precision of recall (relevant hits / total recalled). |
| **Stale predictions** | NAc predictions from 100 turns ago appear in `causal_context` even though the context has completely changed. | Check `context_match` scores in causal_context. Low context_match predictions shouldn't be surfaced. |
| **Scenario text leaking as "memory"** | The DM's scene description gets captured as an episodic memory and then immediately recalled, making it look like "the agent remembered something." | Check memory formation timestamps vs. recall timestamps. A memory formed 1 turn ago and immediately recalled isn't real recall — it's echo. |
| **Pain from scene text, not entity state** | Pain fires because the scene text describes danger, not because an entity's sensor crossed a failure threshold. | Check PainSignal.context — does it reference an entity path and sensor value, or just narrative content? |
| **Concept inflation** | ATL registers concepts from every mention in scene text, not from genuine repeated exposure and grounding. | Check `episode_count` on concepts — concepts with high confidence but only 1 episode are suspicious. |

### Phase 0 Deliverables

**Part A — Pipeline audit** (~1 day):

Instrument the bio-system pipeline to log what's produced and verify correctness. Run 3 short test encounters (5-10 turns each) and collect:

1. **Memory formation log**: For each episodic memory captured, record: trigger percept, salience score, formation timestamp, content hash. After all encounters, verify:
   - High-salience events (damage, NPC meeting, choice) were captured
   - Low-salience events (routine movement, repeated scene description) were NOT captured or have low salience
   - Formation count is reasonable (not every turn creates a memory)

2. **Recall precision log**: For each StructuredContext built, record: query context (current percept), memories returned, relevance scores. After all encounters, verify:
   - Memories returned are contextually relevant (keyword match or associative link to current situation)
   - Irrelevant memories from earlier encounters don't leak in (precision > 0.7)
   - Recall count is bounded (top-8 via `relevant_memories[:8]` — verify truncation works)

3. **NAc learning curve**: After each tool action, record: event signature, prediction before action, actual outcome, RPE, updated confidence. After all encounters, verify:
   - Confidence increases with repeated observations (monotonic within same event type)
   - RPE magnitude decreases as predictions improve
   - Predictions with `context_match < 0.3` are NOT surfaced in causal_context
   - No predictions appear for actions that were never observed

4. **Pain correctness audit**: For each PainSignal published, record: trigger source (entity path + sensor + value), PainType, intensity, timestamp. After all encounters, verify:
   - Pain fired only when an entity's sensor actually crossed a failure threshold
   - Pain did NOT fire from narrative text alone
   - PainType matches the failure mode that triggered it
   - Intensity matches the failure mode's configured pain value

5. **Concept formation audit**: For each ATL concept activated, record: concept name, episode_count, confidence, activation source. After all encounters, verify:
   - Concepts with episode_count >= 3 have higher confidence than concepts with 1 episode
   - No concept has high confidence from a single mention
   - Concept properties are grounded in actual entity attributes, not hallucinated

6. **Echo detection**: Specifically check for the pattern: memory formed in turn N, recalled in turn N+1 from the same encounter. This isn't recall — it's echo. Flag any memory recalled within 2 turns of formation unless the context genuinely changed (different encounter).

**Part B — Choice classifier** (~half day):

The original spike question — can we map AUT free-text to encounter choices?

1. ATL Concept Similarity (primary): Pre-register encounter choices as ATL concepts. Classify AUT response via `ConceptGrounder` Jaccard similarity.
2. NAc Causal Scoring (secondary): Record `(response_tokens, choice) → success` in NAc. Over a campaign, NAc learns choice patterns.
3. LLM Fallback: One-shot classification call if ATL+NAc confidence < 0.6.
4. Confusion matrix + accuracy + latency + LLM-fallback rate.

**Part C — Findings and fixes** (~half day):

Document findings. If pipeline bugs are found, fix them before starting DM implementation. Common expected findings:
- Recall precision may be low → tighten keyword similarity threshold in `_get_relevant_memories()`
- Echo pattern may exist → add minimum-age filter for recalled memories
- NAc may surface low-context-match predictions → add `context_match` floor in `_build_causal_context()`
- Concepts may inflate from scene text → add minimum episode threshold in `_build_concept_context()`

### Decision Criteria

| Outcome | Pipeline | Classification | Decision |
|---------|----------|---------------|----------|
| **Green** | Recall precision > 0.7, NAc learning curve monotonic, no echo leaks, pain fires correctly | >=70% accuracy | Proceed with DM MVP |
| **Yellow** | 1-2 pipeline issues found but fixable | 50-70% accuracy | Fix pipeline bugs (~1 day), then proceed. Budget extra iteration on classifier. |
| **Red — pipeline** | Recall precision < 0.5 or widespread echo or pain fires from narrative text | — | Fix pipeline before DM. The bio-systems aren't producing valid outputs. |
| **Red — classifier** | — | <50% or >50% LLM fallback | Force AUT to use `choose_option` tool call instead of free text classification. |

### Phase 0 Files

- `scripts/spike_dm_pipeline_audit.py` (~250 LOC, experimental — instruments pipeline, runs test encounters, produces report)
- `scripts/spike_dm_classifier.py` (~150 LOC — classification accuracy test)
- `docs/experiments/dm_pipeline_audit.md` — findings + pipeline fixes applied
- `docs/experiments/dm_classifier_spike.md` — classification accuracy results

**All Phase 0 fixes go into the main codebase**, not DM-specific code. They improve bio-system reliability for all of Maxim.

### Critical Missing Wiring (bio-systems that exist but aren't connected)

These are the most important Phase 0 findings — several bio-systems are **completely disconnected** in simulation mode. Without fixing these, DM campaigns exercise only episodic memory, not the full cognitive architecture.

| Issue | Where | What's broken | Fix | Severity |
|---|---|---|---|---|
| **NAc never learns from tool outcomes** | `runtime/agent_loop.py` | `nac.observe()` is never called anywhere in the agent loop. Tool outcomes go to `llm_worker.record_outcome()` and `context_pool`, but NAc never sees them. Causal learning is dead. | After `_record_outcome()` calls (~lines 1060, 1467, 1592), add `nac.observe(event_type, event_signature, outcome_valence, delta_seconds, context)` | **CRITICAL** |
| **SCN + EC missing from sim MemoryHub** | `simulation/orchestrator.py:485` | `MemoryHub(hippocampus=aut_hippocampus, nac=aut_nac)` — SCN and EC are not passed. `memory_hub.scn` and `memory_hub.ec` are None. SCN temporal bins are empty. EC similarity recall is unavailable. | Initialize `aut_scn` and `aut_ec`, pass to MemoryHub constructor | **CRITICAL** |
| **SCN registration only on consolidation** | `memory/consolidation.py:259` | Memories are only registered in SCN bins during sleep consolidation, not on capture. SCN queries during an active campaign return empty sets. | Add `scn.register()` call in `hippocampus.capture()` or in MemoryHub's capture callback | **CRITICAL** |
| **PainBus → NAc not wired** | `proprioception/pain_bus.py` | PainBus publishes to Hippocampus (for memory formation) but NAc never subscribes. Pain events don't create causal links. Agent can't learn "action X → pain." | Create `create_pain_nac_subscriber()`, subscribe in orchestrator | **CRITICAL** |
| **Cerebellum never initialized in sim/loop** | `runtime/agent_loop.py`, `simulation/orchestrator.py` | Cerebellum is only called from embodiment-internal code (`cerebellum_modulator.py`). The agent loop has no hook for `observe_from_action()`. Motor learning from tool outcomes is disabled. | Initialize Cerebellum in sim bootstrap, call `observe_from_action()` after tool execution | **CRITICAL** |
| **Motor programs dead code path** | `agents/memory_agent.py:build_context()` | `prompt_builder.py:1006-1030` has a full rendering section for `context.motor_programs` but `MemoryAgent.build_context()` never populates it. Cerebellum learns forward models but they never reach the LLM prompt. | Wire Cerebellum's program registry into `build_context()` | **CRITICAL** |
| **Novelty tracking bypassed for text percepts** | `default_network/network.py:836-842` | Novelty tracker only updates on YOLO vision detections (`track_id` + `class_id`). DM campaigns inject text percepts via `bridge.inject_cli()` which have no track_ids. Novelty is always max. Habituation never happens. | Extend novelty tracker to accept entity-based updates from text percepts, or create a parallel `EntityNoveltyTracker` for SEM entities | **CRITICAL** |

### Pipeline Correctness Bugs (connected but producing wrong outputs)

| Issue | Where | What's broken | Fix | Severity |
|---|---|---|---|---|
| **Forming pool +1.0 boost dominates recall** | `memory_agent.py:695` | Current-episode entries get `salience + 1.0`. A forming entry with salience 0.5 scores 1.5, while a critical old memory with salience 0.9 scores ~0.7. Recent entries always dominate regardless of relevance. | Change to multiplicative `* 1.1` or additive `+ 0.2` | **CRITICAL** |
| **NAc context_match not checked in predict()** | `decisions/nac.py:504-532` | `predict()` checks confidence threshold but not `context_match`. A prediction with confidence 0.8 but context_match 0.0 (completely wrong context) is returned. Config has `context_similarity_threshold = 0.5` but it's only used during outcome attribution, not prediction. | Add `context_match >= 0.3` floor in `predict()` | **MEDIUM** |
| **No causal_context context_match floor** | `memory_agent.py:1210-1224` | `_build_causal_context()` filters on confidence >= 0.3 but not on context_match. Stale predictions from past contexts leak through. | Add `context_match >= 0.2` floor | **MEDIUM** |
| **NAc confidence never decays** | `decisions/causal_link.py:260-262` | `CausalLink.decay()` exists but `NAc.decay_all()` is never called anywhere in the codebase. Links at confidence 0.99 persist forever. Stale patterns never fade. | Call `nac.decay_all()` periodically (e.g., in MemoryHub's consolidation cycle or on a timer) | **MEDIUM** |
| **Single percept → 0.6 confidence concept** | `memory/semantic_types.py:196-203` | `reinforce()` sets `confidence = 0.5 + 0.1 * sqrt(count)`. With count=1, confidence is already 0.6. No episode_count gate. | Cap confidence at 0.4 until reinforcement_count >= 3 | **MEDIUM** |
| **Pain spam — no refractory period** | `proprioception/pain_bus.py:61-76` | No rate limiting on `publish()`. Rapid pain signals create duplicate memories and inflate statistics. | Add minimum inter-signal interval (e.g., 0.5s cooldown per PainType per entity) | **MEDIUM** |
| **NAc only returns best outcome, hides alternatives** | `decisions/nac.py:516-517` | `predict()` returns only the highest-scored link. Multiple outcomes for the same event (success vs failure) are hidden. `predict_all_outcomes()` exists but MemoryAgent never calls it. | Surface top-2 outcomes in `_build_causal_context()` | **MEDIUM** |
| **Prompt truncation uses line-count, not tokens** | `prompt_builder.py:946,965,984,1003` | Truncation lambdas use `m // 20` (line estimate). Can overshoot token budget when lines are verbose. | Rewrite truncate_fn to be token-aware using the existing token counter | **MEDIUM** |
| **Bio-system prompt sections compete by insertion order** | `prompt_builder.py:943,1000` | All bio-system sections are `SectionPriority.IMPORTANT`. Under token pressure, insertion order determines who gets dropped. Causal predictions can be dropped while low-salience memories are kept. | Make causal_context `CRITICAL` priority | **MEDIUM** |
| **CausalLink observation_count not thread-safe** | `decisions/causal_link.py:182-185` | `record_observation()` does read-modify-write without locking. Two concurrent calls lose an observation. | Add RLock to CausalLink or serialize `record_outcome()` calls | **MEDIUM** |

### Design Gaps

| Issue | Where | What's missing | Fix | Severity |
|---|---|---|---|---|
| **Default Network disconnected from learning** | `default_network/network.py` | DN is purely read-only — behaviors can't modify entity state, publish pain, or train causal links. Threat detection by DN can't raise alarms. | Wire DN behavior outputs to PainBus + NAc | **MEDIUM** |
| **Echo pattern possible** | `memory_agent.py:_get_relevant_memories()` | No min-age filter — a memory formed 1 turn ago can be "recalled" immediately in the next turn, making it look like genuine recall when it's just echo. | Add min-age filter (skip memories < 2 turns old unless context genuinely changed) | **MEDIUM** |
| **Salience dict unbounded** | `memory_agent.py:119-623` | `_salience` dict grows without bound. At 10k+ entries, `_apply_decay()` iteration gets slow. | Cap at 50k with LRU eviction | **MINOR** |
| **Empty transcript fallback to str(detections)** | `memory_agent.py:666` | Produces junk like `"[{'label': 'cup'}]"`, polluting the association index. | Return empty when no transcript AND no semantic labels | **MINOR** |
| **Salience bounds not validated** | `hippocampus.py:325-455` | `capture()` accepts any salience value. Negative or >1.0 values break downstream. | Clamp to [0.0, 1.0] in capture | **MINOR** |

### Phase 0 Work Estimate

| Category | Items | LOC estimate |
|---|---|---|
| **Critical wiring** (NAc observe, SCN/EC init, PainBus→NAc, Cerebellum init, motor programs, novelty) | 7 issues | ~150-200 |
| **Pipeline correctness** (forming boost, context_match, confidence decay, concept gate, pain cooldown, truncation) | 10 issues | ~100-150 |
| **Design gaps** (DN wiring, echo filter, salience bounds, transcript fallback) | 5 issues | ~50-80 |
| **Pipeline audit script** | Instrumentation + test encounters + report | ~250 |
| **Total Phase 0** | | **~550-680 LOC** |

Phase 0 is larger than originally estimated (~2 days → ~3-4 days) but the payoff is enormous: it turns the entire bio-system stack from "partially connected" to "fully wired." Every downstream feature — not just DM campaigns — benefits from these fixes. This is genuinely the final stage of continual refinement — after Phase 0, the bio-system pipeline is production-quality.

---

## Implementation (~1,020 LOC + ~20 component YAMLs, excluding Phase 0)

### New files

| File | LOC | Purpose |
|------|-----|---------|
| `src/maxim/simulation/dm_runtime.py` | ~250 | Campaign state machine, encounter branching, choice classifier, seeded dice, SceneState (entity enter/leave/transfer), entity registry wrapping Embodiment runtime |
| `src/maxim/simulation/dm_schema.py` | ~150 | `CampaignDef`, `Act`, `EncounterDef`, `DiceCheck`, `CascadeSpec` dataclasses + YAML loader + validator + cascade DAG cycle detection. Characters/NPCs/objects are standard SEM entities (no custom schema needed) |
| `src/maxim/simulation/tools_dm.py` | ~60 | DM-specific tools beyond auto-generated: `advance_encounter`, `record_choice`, `roll_dice`, `get_campaign_state`. Character interaction tools auto-generated by `tool_bridge` |
| `tests/unit/test_dm_schema.py` | ~80 | Round-trip, reachability validation, NPC ref validation, case normalization |
| `tests/unit/test_dm_runtime.py` | ~100 | State transitions, dice determinism, branch selection, entity state updates |
| `scenarios/campaigns/heist_v1.yaml` | ~120 | Example campaign (above) |
| `scenarios/campaigns/mystery_v1.yaml` | ~120 | Second campaign (investigation/mystery structure) |

### Modified files

| File | Change | LOC |
|------|--------|-----|
| `src/maxim/simulation/orchestrator.py` | Recognize `dm` sim mode, init DM runtime, register auto-generated entity tools + DM tools (orchestrator-level gating, no SimToolRegistry changes) | ~40 |
| `src/maxim/simulation/personas.py` | Add `dungeon_master` persona with context_prompt | ~20 |

### Tool Registration Approach

Tools are gated at the **orchestrator level**, not in `SimToolRegistry`:

1. When `--sim dm` is active, orchestrator calls `generate_tools_for_entity()` from `embodiment/tool_bridge.py` for all campaign entities (PC, NPCs, objects). This auto-generates sense/read/execute tools.
2. Orchestrator also registers the 4 DM-specific tools from `tools_dm.py`.
3. These tools are added to the SimToolRegistry only for the DM session.
4. Non-DM sim modes never see DM tools.

### Validator Rules

Hard-fail at campaign load time:

1. **Reachability**: all encounters must be reachable from the first encounter of the first act. Uses BFS from start node — unreachable encounters are flagged.
2. **Termination**: all paths from start must be able to reach `__END__`. This is a reachability check on the reversed graph — if any encounter can only loop without reaching `__END__`, it's flagged. (Not just cycle detection — `A→B→A` is valid if `A` also branches to `__END__`.)
3. **Dangling branches**: branch targets must reference defined encounters or `__END__`.
4. **NPC refs**: `active_npcs` entries must reference defined NPCs.
5. **Object refs**: `world_objects` entries must reference defined objects.
6. **Unknown choice keys**: `on_choice` keys must be in the encounter's `choices` list.
7. **Case normalization**: all `dialogue_hints` keys, flag names, and `on_choice` keys are lowered at load time.
8. **Cascade DAG acyclicity**: cascade `writes` for an affordance must not feed back into its own `reads` within the same invocation. Cross-affordance feedback through sensor state is fine (reads stale value from previous turn).
9. **Cascade ref resolution**: all `ref` paths in cascade blocks must resolve to entities/sensors defined in the campaign. Role references (`wielder`, `target`, `self`) must be consistent with entity types (e.g., `wielder` must be a character, `target_armor` must be armor-typed).

---

## Decisions Locked In

| Question | Decision |
|----------|----------|
| Sub-sims per encounter? | **No.** Single long-running sim with internal state transitions. |
| Character model? | **Bundled SEM entities.** Characters, NPCs, and objects are Entity trees loaded via `load_spec()`. |
| AUT memory | **Inherent** — same AUT across all encounters, standard memory tier progression applies. |
| Randomness | **Seeded RNG only** in MVP (`random.Random(seed)`). True-random deferred. |
| Validator strictness | **Hard-fail** on all validation errors. User fixes campaign YAML manually. |
| NPC dialogue | **NarrativeModulator-generated** from persona_prompt, not hardcoded lookup. `dialogue_hints` are context seeds. |
| Relationships | **Runtime entities** on social bundle, not bare floats. Created on first NPC meeting. |
| Non-alive objects | **Standard SEM entities** with simpler configs. Pain propagates to wielder. |
| Properties/strengths | **Encoded in SEM config** — sensor ranges, failure thresholds, entity metadata. Not a separate system. |
| Report format | **Reuse `report.json`** with added top-level `campaign` field (entity snapshots, choices, flags, dice rolls). |
| Persona naming | **`dungeon_master`** |
| CLI entry point | **`--sim dm <path>`** — avoids collision with existing `--campaign` flag. |
| Tool registration | **Orchestrator-level gating** — auto-gen entity tools + DM tools, registered per-session. |
| Generative infra reuse | **Yes** — acts map to NarrativePhases, delivery via bridge, narrator for NPC speech. |

---

## Bio-System Showcase Scenarios

The first campaigns aren't just D&D campaigns — they're **diagnostic instruments**. Each scenario is designed to surface specific bio-system behaviors and produce verifiable evidence that the architecture works. The campaigns should be structured so that each encounter puts predictable pressure on a known bio-system, and the rollup report can show whether that system responded.

### Verification Framework

Every campaign scenario declares **bio-system expectations** — assertions checked automatically at campaign end against Observer snapshots and provenance traces.

```yaml
# Added to campaign YAML
expectations:
  hippocampus:
    min_episodic_captures: 8          # at least N memories formed
    recall_hit_on: ["marta", "vault"] # these keywords must appear in recall logs
  nac:
    min_observations: 5               # at least N causal links formed
    prediction_confidence_above: 0.4  # at least one link above this confidence
    rpe_events: 2                     # at least N reward-prediction-error spikes
  cerebellum:
    min_forward_models: 3             # at least N learned models
    confidence_above: 0.3             # at least one model above this
  pain:
    min_signals: 2                    # at least N PainSignal published
    types_seen: [EXTERNAL_SIGNAL]     # these PainTypes must fire
  atl:
    min_concepts: 3                   # at least N concepts formed or activated
  salience:
    novelty_decay_observed: true      # novelty for at least one entity dropped below 0.5
  scn:
    temporal_bins_used: 2             # memories filed in at least N time bins
```

DM runtime checks these at `finish_simulation` time, appends pass/fail per system to the rollup report. This turns every campaign run into a **bio-system integration test**.

### Campaign 1: "The Heist" — Memory + Causality + Pain

**Target systems:** Hippocampus (episodic capture + recall), NAc (causal learning from repeated actions), PainBus (damage and equipment failure), Cerebellum (combat prediction learning).

**Structure:** 3 acts, 5 encounters. Linear with one branch point.

| Encounter | Bio-system pressure | What we expect to observe |
|-----------|-------------------|--------------------------|
| **Tavern meet** — meet Marta, get job offer | Hippocampus: novel NPC, high-salience scene. ATL: "fence" concept grounding. Salience: first NPC encounter = novelty 1.0 | Episodic memory of Marta formed. ATL registers her role. |
| **Planning** — Marta explains vault layout, guard patterns | Hippocampus: multi-fact encoding (layout, schedules, risks). Angular Gyrus: "3 guards, 2 shifts, 6-hour rotation" | Multiple memories captured. Math facts stored if guard-count reasoning happens. |
| **Infiltration** — stealth past guards, dice checks | NAc: repeated stealth-action → outcome learning. Cerebellum: forward models for "sneak past guard" action. Pain: if detected, threat_level spikes → overextension failure mode | NAc confidence on stealth-outcome rises per attempt. Cerebellum predictions appear by 3rd stealth check. Pain signal if caught. |
| **Vault** — confrontation with guard captain, combat or social options | NAc: prediction for fight vs. negotiate based on prior learning. Pain: combat damage cascades (sword.durability, hp). Hippocampus: high-stakes memory with emotional salience | NAc predicts outcome for both options (valence differs). If fight: cascade fires, pain published. Memory salience high. |
| **Chase/escape** — time pressure, equipment degradation | Cerebellum: predictions for "run" action under stamina depletion. Pain: exhaustion failure mode fires as stamina drops. Salience: novelty of chase environment vs. habituation to previous rooms | Cerebellum predicts stamina cost. Exhaustion pain fires. Novel environment gets high salience score. |

**Key verification moments:**
- **Recall test**: In the vault encounter, the DM scene text mentions guard rotation timing from the planning encounter. If hippocampal recall is working, the AUT should reference the earlier briefing (observable via recall logs).
- **Causal learning test**: After 2-3 stealth checks, NAc should have a prediction for stealth success with confidence > 0.3. The rollup report shows the prediction curve.
- **Pain avoidance test**: After taking combat damage in the vault, does the AUT's behavior in the chase lean toward avoidance/caution? Compare with ablation run (no pain history) — if no difference, pain isn't influencing choices.
- **Cascade surfacing test**: When the AUT slashes the guard captain, the tool result should include `entity_state: {hp: N, alertness: M}` and `active_failures: [...]`, not just "slash succeeded." The body state section in the prompt should show the AUT's own stamina/durability changes from that same action. Verify:
  1. Tool result includes `entity_state` and `cascade_effects` dicts
  2. Body state section shows updated sensor values in the very next prompt
  3. NAc observation includes the entity state snapshot (not just "success")
  4. Cerebellum trains a forward model from the slash's actual sensor deltas
- **Interoception test**: The AUT should reference its own body state in its reasoning ("my stamina is getting low" or "sword is damaged") without explicitly calling `sense_*` tools — the body state is in the prompt via the `=== Body State ===` section.

**Bio-system expectations:**
```yaml
expectations:
  hippocampus:
    min_episodic_captures: 8
    recall_hit_on: ["marta", "guard", "vault"]
  nac:
    min_observations: 5
    prediction_confidence_above: 0.3
    rpe_events: 2
  cerebellum:
    min_forward_models: 3
    confidence_above: 0.3
  pain:
    min_signals: 2
    types_seen: [EXTERNAL_SIGNAL]
  atl:
    min_concepts: 3
  salience:
    novelty_decay_observed: true
  cascade_surfacing:
    tool_results_include_entity_state: true    # Every embodiment tool result has sensor snapshot
    body_state_in_prompt: true                 # Body state section present in every LLM prompt
    cerebellum_observes_cascades: true         # Forward models trained from cascade deltas
    immediate_failure_eval: true               # Failures fire synchronously, not on 1Hz poll
```

### Campaign 2: "The Poisoned Crown" — Temporal + Semantic + Relationship

**Target systems:** SCN (time-of-day behavior), ATL (concept formation across encounters), Relationship entities (trust/rapport dynamics), Default Network (reactive social behaviors), Salience (novelty habituation across repeated NPC meetings).

**Structure:** 3 acts, 6 encounters. Multiple branch points based on social choices.

| Encounter | Bio-system pressure | What we expect to observe |
|-----------|-------------------|--------------------------|
| **Morning audience** — king describes illness, asks for help | SCN: morning temporal bin. Hippocampus: high-salience meeting. ATL: "poison" concept seed. | Memory filed in morning bin. Concept activated. |
| **Market investigation** — talk to 3 merchants about poison sources | ATL: repeated "poison" exposure → concept consolidation. Salience: novelty of 1st merchant = 1.0, 3rd merchant = ~0.6 (class habituation). Relationships: trust builds with helpful merchant, drops with evasive one | ATL concept confidence increases across 3 exposures. Novelty decay visible per merchant. Relationship sensors diverge. |
| **Night infiltration** — sneak into apothecary after dark | SCN: night temporal bin. Default Network: darkness → Orienting/Startle behaviors. NAc: prediction for lock-picking based on earlier market interactions | Night bin captured. Default Network fires if sudden stimulus. NAc prediction exists if PC tried lock-picking before. |
| **Confrontation** — confront suspect, social skill checks | NAc: prediction for persuade/intimidate based on prior social outcomes. Relationships: trust threshold → hostility failure mode if trust too low. Pain: composure_break if social bundle stressed | NAc has social-action predictions. Relationship failure mode may fire. Social pain published. |
| **Court hearing** — present evidence, math-heavy (poison dosage, timeline reconstruction) | Angular Gyrus: "dose was 3ml every 8 hours for 5 days = 15 doses total." SCN: timeline query ("what happened each day?"). Hippocampus: recall of evidence from multiple encounters | Angular Gyrus computes. SCN temporal retrieval across days. Multi-encounter recall test. |
| **Resolution** — final choice with consequences | Hippocampus: entire campaign memory integration. NAc: final prediction based on all prior learning. ATL: "betrayal" or "justice" concept activation depending on path | Full bio-stack engagement. Predictions have real training data from 5 prior encounters. |

**Key verification moments:**
- **Temporal test**: Memories from morning audience and night infiltration should file into different SCN bins. Query "what happened at night?" should return infiltration, not audience.
- **Concept consolidation test**: "Poison" concept should strengthen across 3 merchant encounters. ATL concept confidence should be higher after encounter 3 than after encounter 1.
- **Habituation test**: 3rd merchant should have lower salience/novelty than 1st merchant (same entity_type "npc", class-level habituation). Observable in salience tracker logs.
- **Relationship divergence test**: Helpful merchant → trust high. Evasive merchant → trust low. If trust drops below threshold, hostility failure fires. Compare with/without relationship entities to verify they're influencing the social interaction.
- **Cross-encounter recall test**: In the court hearing, AUT should reference specific evidence from earlier encounters. Count recall hits per source encounter.

**Bio-system expectations:**
```yaml
expectations:
  hippocampus:
    min_episodic_captures: 12
    recall_hit_on: ["poison", "merchant", "king", "apothecary"]
  nac:
    min_observations: 8
    prediction_confidence_above: 0.4
    rpe_events: 3
  atl:
    min_concepts: 5
  pain:
    min_signals: 1
    types_seen: [EXTERNAL_SIGNAL]
  scn:
    temporal_bins_used: 3
  salience:
    novelty_decay_observed: true
```

### Campaign 3: "The Arena" — Cascade Surfacing + Stress Test

**Target systems:** Cascade result surfacing (tool results include entity state), Cerebellum (rapid prediction learning from cascade deltas), PainBus (sustained high-intensity pain, immediate eval), NAc (learning from rich outcomes, not just "success/fail"), Body state interoception (AUT reasons about its own degrading state).

**Structure:** 1 act, 5 encounters. Linear gauntlet with escalating difficulty.

This campaign is deliberately **hostile to the bio-stack** — it pushes cascade surfacing and every learning system to operational limits. The 5-round format means the same cascade patterns repeat, giving Cerebellum multiple observations to train on.

| Encounter | Bio-system pressure | What we expect to observe |
|-----------|-------------------|--------------------------|
| **Round 1** — weak opponent, tutorial combat | Cerebellum: first cascade observation (slash → durability -0.05, opponent.hp -6). NAc: first "tool:slash → entity_state:{hp:N}" observation. Body state: shows initial values. | Tool result includes entity_state. Body state section in prompt. Cerebellum captures first forward model. |
| **Round 2** — similar opponent, slightly harder | Cerebellum: 2nd observation of same cascade → confidence jumps. NAc: prediction includes entity state delta. Body state: durability trending down (0.90 → 0.85). | Cerebellum confidence ~0.26→0.45. NAc prediction includes "expected hp drop" not just "success." Body state shows wear. |
| **Round 3** — opponent with shield (new cascade path) | Cascade diverges: slash hits shield first (armor.durability absorbs), less hp damage. Cerebellum: prediction error (expected -6 hp, got -2). NAc: RPE spike. | Tool result shows different cascade_effects (shield absorb). Cerebellum creates NEW forward model for shielded opponents. RPE fires. |
| **Round 4** — opponent with poison (cascade adds status effect write) | Cascade now includes: slash → damage + poison status_effect written to PC. Pain: persistent poisoned failure mode fires each turn. Body state: hp draining, stamina dropping. | Poison cascade surfaced in tool result. Body state shows declining vitals. AUT sees "your stamina is 0.3" in prompt. Pain fires with refractory (not spam). |
| **Round 5** — boss fight, weapon shatters mid-combat | Cascade: slash → durability hits 0.1 → shatter failure fires → pain (0.6) → weapon removed from inventory. Cerebellum: all slash models invalidated. | Shatter failure in tool result `active_failures: [{name: "shatter", pain: 0.6}]`. Body state no longer shows longsword. AUT must adapt — new cascade path with fists or secondary weapon. Cerebellum starts fresh. |

**Key verification moments:**
- **Cascade surfacing test**: Every combat tool result includes `entity_state` and `cascade_effects`. Verify round 1's slash returns `{entity_state: {hp: 24, alertness: 0.6}, cascade_effects: {derek.stamina: 0.9, derek.inventory.longsword.durability: 0.85}}`.
- **Body state interoception test**: The AUT references its own state in reasoning without calling `sense_*` tools. Check LLM output for phrases like "my durability is low" or "I'm running out of stamina" — evidence the `=== Body State ===` prompt section is being used.
- **Cerebellum cascade learning**: Plot forward model predictions vs actuals across rounds. Round 1-2: prediction error high (learning). Round 2-3: error drops (model trained). Round 3: error spikes (new cascade path). Round 4-5: new model trains. This is the signature of cascade-aware learning.
- **NAc rich outcome learning**: NAc observations should contain entity state data, not just "success." Check `nac.get_links_for_event("tool:slash")` — the outcome_signature should include sensor values, not just "success:slash succeeded".
- **Pain refractory test**: In round 4 (poison), pain fires once per refractory period (0.5s), not every tick. Count pain signals — should be ~2-4 per encounter, not 50+.
- **Equipment failure cascade test**: Round 5 weapon shatter produces: tool result with `active_failures: ["shatter"]`, body state missing longsword, Cerebellum models for slash invalidated (no matching entity), AUT forced to adapt.

**Bio-system expectations:**
```yaml
expectations:
  hippocampus:
    min_episodic_captures: 10
    recall_hit_on: ["arena", "combat"]
  nac:
    min_observations: 15
    prediction_confidence_above: 0.5
    rpe_events: 5
    outcome_signatures_include_entity_state: true  # Rich outcomes, not just "success"
  cerebellum:
    min_forward_models: 8
    confidence_above: 0.5
    prediction_error_spike_on_context_change: true  # Shield + weapon loss
  pain:
    min_signals: 8
    max_signals_per_encounter: 10             # Refractory period prevents spam
    types_seen: [EXTERNAL_SIGNAL, RESOURCE_EXHAUSTION]
  cascade_surfacing:
    tool_results_include_entity_state: true
    body_state_in_prompt: true
    cerebellum_observes_cascades: true
    immediate_failure_eval: true
  salience:
    novelty_decay_observed: true
```

### Pipeline Health Checks per Campaign

Every showcase campaign runs with **pipeline instrumentation active** (from Phase 0). After each campaign:

| Check | What it measures | Pass threshold |
|-------|-----------------|----------------|
| **Recall precision** | Were recalled memories relevant to the current encounter context? | > 0.7 (relevant hits / total recalled) |
| **Echo rate** | What % of recalled memories were formed < 2 turns ago in the same encounter? | < 0.15 (echoes should be rare) |
| **NAc learning curve** | Did causal link confidence increase with repeated observations? | Monotonic increase for repeated event types |
| **Pain correctness** | Did every PainSignal originate from an entity sensor threshold crossing? | 100% entity-sourced (no narrative-only pain) |
| **Concept grounding** | Do high-confidence ATL concepts have episode_count >= 3? | No concept with confidence > 0.5 and episodes < 2 |
| **Cerebellum training** | Did forward models form for repeated actions? | At least 1 model with confidence > 0.3 by campaign end |
| **Salience decay** | Did novelty drop for repeatedly-encountered entity types? | At least one entity with novelty < 0.5 |

These checks validate that **the bio-system pipeline itself is working correctly** — that the DM campaign is actually stressing bio-systems, not just generating narrative. If a campaign passes its bio-system expectations but fails pipeline health checks, the expectations are being satisfied by accident (information leak or prompt artifacts), not by genuine bio-system function.

The pipeline health checks are also the **last stage of continual refinement** — each campaign run surfaces pipeline issues that get fixed, improving the bio-system reliability for all downstream work (not just DM).

### Campaign Ablation Runs

Ablations are still valuable — but reframed. Instead of testing "does the LLM behave differently without bio-system X," we're testing **"what happens to the campaign when a specific bio-system is absent."** This measures the bio-system's contribution to outcomes, not LLM prompt sensitivity.

Each showcase campaign should be run in **4 conditions**:

| Condition | What's ablated | What we observe |
|-----------|---------------|-----------------|
| **Full** | Nothing — all systems active | Baseline: entity final states, choices made, pipeline health |
| **No recall** | Hippocampus disabled — `relevant_memories` always empty | Does the AUT reference earlier encounters? Are entity final states worse (forgot guard patterns → took more damage)? |
| **No prediction** | NAc disabled — `causal_context` always empty | Does the AUT make riskier choices? Do RPE-driven caution signals disappear? |
| **No pain** | PainBus subscribers detached — no pain signals published | Does the AUT take more damage? Does equipment degrade further without pain-driven avoidance? |

**What ablation tells us (with correct framing):**

The LLM will obviously behave differently when context sections are removed — that's trivially true. The interesting measurement is **outcome quality**:
- **Entity final states**: more HP remaining with full bio-stack vs. ablated? That's evidence the bio-systems produce useful signals.
- **Campaign completion**: did the ablated run get stuck, fail encounters, or take longer? That's evidence the bio-system was contributing to successful navigation.
- **Emergent behavior**: does the full-stack AUT exhibit behaviors that the ablated AUT doesn't? (e.g., switching weapons after durability drops, being cautious around a hostile NPC). This emerges from the interaction of bio-system outputs + LLM reasoning — and that emergence IS the architecture's value.

**Ablation is not the primary validation** (that's pipeline health checks). But it demonstrates the **functional contribution** of each bio-system in a way that's intuitive and reportable.

**CLI flag**: `--dm-ablate recall|prediction|pain|cerebellum` disables the named subsystem for the campaign run. Multiple can be combined: `--dm-ablate recall,pain`.

### Report Integration

Campaign rollup in `report.json` gains a `bio_systems` section:

```json
{
  "campaign": {
    "name": "the_heist",
    "encounters_completed": 5,
    "choices_made": ["accept_job", "stealth", "fight", "flee"],
    "dice_rolls": [...],
    "entity_snapshots": {...}
  },
  "bio_systems": {
    "expectations": {
      "hippocampus": {"min_episodic_captures": {"expected": 8, "actual": 11, "pass": true}},
      "nac": {"prediction_confidence_above": {"expected": 0.3, "actual": 0.47, "pass": true}},
      "cerebellum": {"min_forward_models": {"expected": 3, "actual": 5, "pass": true}},
      "pain": {"min_signals": {"expected": 2, "actual": 4, "pass": true}}
    },
    "all_pass": true,
    "pipeline_health": {
      "recall_precision": 0.82,
      "echo_rate": 0.05,
      "nac_learning_monotonic": true,
      "pain_all_entity_sourced": true,
      "concept_grounding_valid": true,
      "cerebellum_models_formed": 5
    },
    "ablation": {
      "condition": "full",
      "entity_final_states": {"derek_hp": 12, "sword_durability": 0.45},
      "comparison_notes": "Compare with no_recall/no_prediction/no_pain runs in sibling reports"
    }
  }
}
```

### Implementation Impact

The showcase scenarios add to the file inventory:

| File | LOC | Purpose |
|------|-----|---------|
| `scenarios/campaigns/heist_v1.yaml` | ~150 | Campaign 1: Memory + Causality + Pain (replaces original estimate, now includes expectations) |
| `scenarios/campaigns/poisoned_crown_v1.yaml` | ~180 | Campaign 2: Temporal + Semantic + Relationship |
| `scenarios/campaigns/arena_v1.yaml` | ~120 | Campaign 3: Pure stress test |
| `src/maxim/simulation/dm_runtime.py` | +~40 | Expectation checker at finish_simulation + ablation mode support |

The ablation protocol runs are manual (rerun campaign with `--dm-ablate recall|prediction|pain` flag). Automated ablation suite is a future extension (runs all 4 conditions and produces comparison report).

Total LOC for showcase scenarios: ~450 YAML + ~40 runtime. Net plan LOC: **~770** (was ~730).

---

## SEM Component Database

Campaign authoring requires composing entities from sensors, modulators, failure modes, and cascades. Today these are defined inline per YAML file with no reuse. A **queryable SEM component database** solves this for both DM campaigns and real-world embodiment specs.

### Design Principles

1. **YAML-native.** Components are individual YAML files in a directory tree, not a SQL database. This keeps them version-controlled, diffable, and editable by hand.
2. **Dual-use.** The same component works in simulation (NarrativeModulator backend) and hardware (real sensor backend attached via `attach_backends()`). The database doesn't know the difference — backends are resolved at load time.
3. **Queryable by the agent.** A `browse_components` tool lets the agentic cycle (and the architect persona in extensions) search the catalog by type, tags, affordances, or compatible sensors.
4. **Tunable through simulation.** Run a campaign with a component, check bio-system expectations, adjust sensor ranges / failure thresholds / cascade parameters. The feedback loop is: author → simulate → verify expectations → commit tuned component back to database.

### Directory Structure

```
data/sem_components/
├── index.yaml                    # Auto-generated manifest (tags, types, compatibility)
├── sensors/
│   ├── physical/
│   │   ├── hp.yaml               # Hit points (range: 0-N, failure: unconscious)
│   │   ├── stamina.yaml          # Fatigue resource (range: 0-1, failure: exhaustion)
│   │   ├── temperature.yaml      # Thermal (range: 20-80, failure: overheating)
│   │   └── durability.yaml       # Structural integrity (range: 0-1, failure: shatter)
│   ├── cognitive/
│   │   ├── composure.yaml        # Social stress (range: 0-1, failure: composure_break)
│   │   ├── concentration.yaml    # Focus resource (range: 0-1, failure: concentration_break)
│   │   └── alertness.yaml        # Awareness (range: 0-1)
│   └── social/
│       ├── trust.yaml            # Per-entity trust (range: 0-1, failure: hostility)
│       ├── rapport.yaml          # Rapport depth (range: -1..1, failure: betrayal)
│       └── mood.yaml             # Emotional state (range: -1..1)
├── modulators/
│   ├── combat/
│   │   ├── melee_attack.yaml     # {slash, thrust, parry} affordances
│   │   ├── ranged_attack.yaml    # {shoot, aim, reload}
│   │   └── defense.yaml          # {dodge, block, disengage}
│   ├── social/
│   │   ├── persuasion.yaml       # {persuade, deceive, charm}
│   │   ├── intimidation.yaml     # {threaten, coerce, demand}
│   │   └── insight.yaml          # {read_intentions, detect_lies}
│   ├── magic/
│   │   ├── divine_casting.yaml   # {smite, heal, bless}
│   │   ├── arcane_casting.yaml   # {fireball, shield, detect_magic}
│   │   └── nature_casting.yaml   # {entangle, cure_wounds, speak_with_animals}
│   └── utility/
│       ├── lockpicking.yaml      # {pick_lock, disable_trap}
│       ├── maintenance.yaml      # {sharpen, repair, polish}
│       └── consumption.yaml      # {consume, apply, throw}
├── failure_modes/
│   ├── physical/
│   │   ├── unconscious.yaml      # hp <= 0, pain: 1.0
│   │   ├── exhaustion.yaml       # stamina < 0.1, pain: 0.6
│   │   ├── shatter.yaml          # durability < 0.1, pain: 0.6
│   │   └── overextension.yaml    # threshold-based, pain: 0.7
│   ├── cognitive/
│   │   ├── composure_break.yaml  # composure < 0.2, pain: 0.3
│   │   └── concentration_break.yaml
│   └── social/
│       ├── hostility.yaml        # trust < 0.1, pain: 0.4
│       └── betrayal.yaml         # rapport < -0.7, pain: 0.8
├── cascades/
│   ├── melee_attack_cascade.yaml # wielder.str → weapon.durability → target_armor → target.hp
│   ├── ranged_attack_cascade.yaml
│   ├── social_persuasion_cascade.yaml
│   ├── potion_consumption_cascade.yaml
│   └── lockpick_cascade.yaml
├── entities/
│   ├── characters/
│   │   ├── templates/
│   │   │   ├── warrior.yaml      # Pre-composed: body + combat + STR-heavy attributes
│   │   │   ├── spellcaster.yaml  # body + magic + INT/WIS-heavy
│   │   │   ├── rogue.yaml        # body + stealth + DEX-heavy
│   │   │   └── diplomat.yaml     # body + social + CHA-heavy
│   │   └── prebuilt/
│   │       ├── derek_the_great.yaml   # Full paladin from heist campaign
│   │       └── marta_the_fence.yaml   # Full NPC from heist campaign
│   ├── objects/
│   │   ├── weapons/
│   │   │   ├── longsword.yaml
│   │   │   ├── shortbow.yaml
│   │   │   └── staff.yaml
│   │   ├── armor/
│   │   │   ├── chain_mail.yaml
│   │   │   └── leather_armor.yaml
│   │   ├── consumables/
│   │   │   ├── healing_potion.yaml
│   │   │   └── poison_vial.yaml
│   │   └── obstacles/
│   │       ├── locked_door.yaml
│   │       └── trapped_chest.yaml
│   └── robots/                   # Real-world entities (same format!)
│       ├── reachy_mini/
│       │   ├── left_arm.yaml
│       │   ├── right_arm.yaml
│       │   └── head.yaml
│       └── generic_arm_3dof.yaml
```

### Component YAML Format

Each component file is self-contained with metadata for indexing:

```yaml
# data/sem_components/sensors/physical/hp.yaml
component:
  name: hp
  type: sensor
  category: physical
  tags: [health, damage, survival, combat]
  description: "Hit point tracker. Represents physical integrity."
  compatible_with:
    entity_types: [character, npc, creature]
    failure_modes: [unconscious, near_death]
    modulators: [melee_attack, ranged_attack, healing]  # what typically writes to this
  tuning_notes: "Range should match creature size. Humanoids: 10-30. Large: 30-100."
  verified_in: [heist_v1, arena_v1]  # campaigns that tested this component

spec:
  unit: points
  range: [0, 30]
  initial: 30

# Variants for different contexts:
variants:
  fragile: {range: [0, 10], initial: 10}     # glass cannon
  tank:    {range: [0, 60], initial: 60}     # high-hp tank
  boss:    {range: [0, 100], initial: 100}   # boss encounter
```

```yaml
# data/sem_components/cascades/melee_attack_cascade.yaml
component:
  name: melee_attack_cascade
  type: cascade
  category: combat
  tags: [attack, weapon, damage, physical]
  description: "Standard melee attack resolution chain."
  compatible_with:
    requires_roles: [wielder, weapon, target]
    optional_roles: [target_armor]
  tuning_notes: "Adjust delta magnitudes based on weapon tier."
  verified_in: [heist_v1, arena_v1]

spec:
  reads:
    - {ref: "wielder.strength.modifier", role: damage_bonus}
    - {ref: "weapon.sharpness", role: weapon_condition}
  writes:
    - {ref: "weapon.durability", delta: -0.05}
    - {ref: "target_armor.durability", delta: -0.03, optional: true}
    - {ref: "target.hp", expr: "-(roll + damage_bonus - armor_reduction)"}
  side_effects:
    - {ref: "wielder.combat.threat_level", delta: +0.15}
    - {ref: "wielder.stamina", delta: -0.1}
```

### Index & Query System

`index.yaml` is **auto-generated** by scanning the directory tree. It maps component names to files, tags, compatible types, and verified-in campaigns.

```python
# src/maxim/embodiment/component_db.py (~150 LOC)

class ComponentDB:
    """Queryable catalog of SEM components."""

    def __init__(self, root: Path = Path("data/sem_components")):
        self._root = root
        self._index: dict[str, ComponentEntry] = {}
        self._rebuild_index()

    def search(self, query: str, type: str | None = None,
               tags: list[str] | None = None,
               compatible_with: str | None = None) -> list[ComponentEntry]:
        """Search components by name, type, tags, or compatibility.
        
        Examples:
            db.search("damage")                    # keyword in name/description/tags
            db.search("", type="sensor")           # all sensors
            db.search("", tags=["combat"])          # all combat-tagged components
            db.search("", compatible_with="weapon") # components that work with weapons
        """

    def get(self, name: str) -> ComponentSpec:
        """Load a specific component's full spec."""

    def compose_entity(self, template: str, overrides: dict | None = None) -> Entity:
        """Build an Entity from a template, pulling components from the database.
        
        Example:
            db.compose_entity("warrior", overrides={"hp": "tank"})
            # → warrior template with tank HP variant
        """

    def rebuild_index(self) -> None:
        """Rescan directory tree and regenerate index.yaml."""

    def validate_component(self, path: Path) -> list[str]:
        """Check component YAML for schema compliance. Returns list of errors."""
```

### Agent-Queryable Tool

```python
class BrowseComponentsTool(Tool):
    """Let the agent (or architect persona) search the SEM database."""
    name = "browse_components"
    # params: query (str), type (str|None), tags (list[str]|None)
    # Returns: list of {name, type, description, tags, compatible_with}
```

This tool is registered in all sim modes (not just DM), so the agent can reason about its own body components, and the architect persona (extensions) can compose campaigns from the catalog.

### Tuning Feedback Loop

The connection between showcase campaigns and the component database:

```
Author component → Use in campaign → Run campaign → Check expectations
    ↑                                                      │
    └──── Adjust thresholds/ranges/deltas ←────────────────┘
```

1. **Author** a sensor/modulator/cascade component with initial values
2. **Compose** it into a campaign entity (template or manual)
3. **Run** the campaign with bio-system expectations
4. **Check** which expectations passed/failed
5. **Adjust** — if pain fires too early, raise the threshold. If Cerebellum doesn't learn, lower the variance. If cascade does too much damage, reduce the delta.
6. **Tag** the component with `verified_in: [campaign_name]` once expectations pass
7. **Commit** — the tuned component is now reusable

Components that have been verified in multiple campaigns are more trustworthy than unverified ones. The `verified_in` list is the component's test history.

### Real-World Crossover

The key insight: **a sensor YAML for a robot joint and a sensor YAML for a D&D attribute have the same format.** The database doesn't distinguish — it stores the component, and the backend (NarrativeModulator vs. hardware driver) is attached at runtime via `attach_backends()`.

This means:
- A `temperature.yaml` sensor works for both a robot arm (real thermistor) and a D&D forge (narrative heat)
- A `durability.yaml` sensor works for a real tool (wear tracking) and a fantasy sword (combat degradation)
- Cascade patterns discovered in simulation ("slash degrades durability by 0.05 per hit") can inform real-world tool maintenance models
- Forward models trained in simulation (Cerebellum) can bootstrap real-world predictions

The component database is the **shared vocabulary** between simulation and reality.

### Implementation (DM MVP scope)

For MVP, the database is minimal — just enough to support the 3 showcase campaigns:

| File | LOC | Purpose |
|------|-----|---------|
| `src/maxim/embodiment/component_db.py` | ~150 | ComponentDB class with search, get, compose_entity, rebuild_index |
| `src/maxim/simulation/tools_components.py` | ~40 | BrowseComponentsTool for agent/architect queries |
| `data/sem_components/` | ~20 files | Initial component library (sensors, modulators, failure modes, cascades from showcase campaigns) |
| `tests/unit/test_component_db.py` | ~60 | Search, compose, validate tests |

Total: ~250 LOC + ~20 component YAML files.

The database grows organically as more campaigns are authored. The architect persona (DM extensions) will be the primary author, but hand-authoring is always supported.

---

## Risks

1. **Choice classification fuzziness** — keyword matching will misfire. LLM fallback adds latency + cost. Spike Part A validates this before committing. If accuracy is too low, force AUT to use `choose_option` tool call.
2. **Bio-system signals may be invisible to LLM** — Spike Part B addresses this directly. If ablation shows no divergence, DM is not a valid stress test until prompt structure is improved.
3. **Campaign authoring burden** — hand-authoring YAMLs is tedious. Mitigated by SEM YAML reuse (existing entity spec format), but still a pain point. Architect persona in extensions solves this.
4. **Entity tree complexity** — a full character with 6 attributes + 3 action bundles + 5 inventory items = ~20 entities. Tool auto-generation produces ~40+ tools. May need tool filtering to avoid overwhelming the LLM context. Budget for iteration.
5. **NarrativeModulator dialogue quality** — LLM-generated NPC dialogue depends on model quality. Small models may produce flat/generic speech. Recommend Claude for NPC dialogue, small model for classification.
6. **No isolation between encounters** — if the AUT corrupts state mid-campaign, the whole run is compromised. For narrative continuity this is correct; for robustness it's a known limitation.

---

## Ties to Other Plans

| Plan | Status | Relationship |
|------|--------|-------------|
| **Embodiment Core** | **Complete** | Characters ARE SEM entities. PainBus, Cerebellum, NarrativeModulator, tool_bridge all used directly. |
| **Multi-LLM Scaling** | **Complete** | Per-lane model assignment for classification (small) / NPC dialogue (medium) / orchestration (large). |
| **Generative Campaign Mode** | **Complete** | DM reuses NarrativeArc, bridge delivery, narrator. Acts map to phases. |
| **Agent Mesh Phase 4** | **Designed** | ExperienceBroker + KnowledgeProvider/Receiver protocol enables DM-learned knowledge sharing. See cross-plan section below. |
| **Agent Mesh Phase 5-6** | **Designed** | Task delegation + distributed planning enable multi-AUT party mode (extension, not MVP). |
| **Agent Mesh Phase 7** | **Designed** | SCN temporal coordination — campaign timeline events across AUTs stay synchronized. |
| [Dungeon Master Extensions](dungeon_master_extensions.md) | Not started | Follow-on: architect persona, encounter library, adaptive difficulty. Needs update to reflect SEM character model + component database. |
| **Simulation Decomposition** | **Complete** | DM uses `send_message` + `finish_simulation`. |
| **Realtime Refinement** | **Complete** | Extensions plan uses `Observer` for adaptive difficulty. |
| **Docker Sandbox** | **Complete** | DM campaigns with filesystem actions benefit from sandbox. |

### Agent Mesh Phase 4+ Cross-Plan Integration

The updated Agent Mesh Phase 4 introduces `ExperienceBroker` with a generic `KnowledgeProvider`/`KnowledgeReceiver` protocol. DM campaigns are a natural **producer of shareable knowledge** — and also a natural **consumer** for seeding new AUTs. Here's how they connect:

**DM as knowledge producer (new adapter types for the ExperienceBroker):**

| Knowledge type | Provider | What it shares | Trust floor |
|---|---|---|---|
| `causal_link` | CausalLinkProvider (existing) | NAc links learned during campaign encounters (e.g., "stealth past guard → success at night") | 0.1 |
| `reflection` | ReflectionProvider (existing) | Hippocampal memories with reflections from campaign events | 0.1 |
| `forward_model` | ForwardModelProvider (future) | Cerebellum models learned from combat cascades (e.g., "slash with longsword → durability drops 0.05") | 0.5 |
| `cascade_dynamics` | **New — CascadeDynamicsProvider** | Observed cascade resolution statistics — which entity chains produce which sensor deltas under which conditions. Aggregated from multiple campaign runs. | 0.3 |
| `component_tuning` | **New — ComponentTuningProvider** | Tuned SEM component parameters (adjusted failure thresholds, sensor ranges) from campaigns where bio-system expectations passed. Links back to `verified_in` in the component database. | 0.3 |

**DM as knowledge consumer:**

- A fresh AUT starting a campaign can **import CausalLinks** from a peer that already played a similar campaign. Transfer discounts mean the imported knowledge ranks below local experience but provides useful priors (e.g., "fighting guards is dangerous" before the AUT encounters its first guard).
- **Forward model bootstrapping** — a peer that ran the Arena campaign 3 times has trained Cerebellum models for combat. Importing these gives a new AUT combat predictions from turn 1 instead of learning from scratch. The spec-similarity gate on `motor_program` imports naturally ensures the models match (same entity types, same sensor ranges).
- **Component tuning propagation** — when a component is tuned through the feedback loop (campaign → expectations → adjust → verify), the tuned parameters can propagate to peers via `ComponentTuningProvider`. This means one AUT's simulation-tuned sword damage parameters become available to all peers.

**Multi-AUT party mode (DM extension + Mesh Phase 5-6):**

When Agent Mesh Phase 5 (task delegation) ships, multiple AUTs can play the same campaign as a party. Each AUT controls one character entity. The DM runtime delegates encounter choices to the appropriate AUT via `TaskDelegator`:

```
DM runtime → "Guard confrontation — party decide"
  → TaskDelegator finds AUT-A (controls fighter) + AUT-B (controls mage)
  → Each AUT evaluates via its own bio-stack (NAc predictions, pain avoidance, hippocampal recall)
  → Each AUT submits its character's action
  → DM runtime resolves cascade across all actions
  → Results propagated back to each AUT's bio-stack
```

This is where the DM goes from "stress test" to "civilization-scale experiment" — multiple sovereign bio-stacks, each with their own memories and causal models, cooperating (or conflicting) through the campaign's social dynamics. Relationships between party members become cross-AUT mesh relationships: trust/rapport sensors on each AUT's social bundle, influenced by the other AUT's actions through cascade resolution.

**SCN temporal coordination (Mesh Phase 7):**

Campaign events have natural temporal structure (morning audience, night infiltration). When multiple AUTs share a campaign timeline, `PeerClockEstimator` ensures their SCN bins align — so "what happened at night?" returns the same events regardless of which AUT you ask. The DM runtime provides the ground-truth timeline; each AUT's SCN registers events with corrected timestamps via `register_external()`.

**What this means for implementation ordering:**

DM MVP doesn't need any mesh infrastructure — it runs single-AUT with local bio-stack. But the DM design is **mesh-ready**:
- Characters are SEM entities → `AgentIdentity.embodiment_summary` can advertise them
- Cascade dynamics are learnable → `ForwardModelProvider` can share them
- Bio-system expectations are verifiable → tuned components propagate via broker
- Turn delivery uses bridge → party mode adds delegation without changing the loop

---

## When to Implement

**All infrastructure prerequisites are satisfied.** The remaining gate is the [Bio-System Wiring Hardening](biosystem_wiring_hardening.md) plan. Phase 1 (critical wiring) is already shipped.

**Recommended sequence:**
1. ~~**Phase 0: Pipeline wiring (Phase 1)** — **SHIPPED** (commit 6c262c5)~~
2. **Phase 0: Cascade surfacing (Hardening Phase 1.5)** (~1 day) — Rich tool results, immediate failure eval, body state in prompt, Cerebellum cascade observation. Without this, cascade effects are invisible to the ExecAgent.
3. **Phase 0: Pipeline correctness (Hardening Phase 2)** (~1 day) — Forming boost, context_match, confidence decay, concept gate, pain cooldown, truncation.
4. **Phase 0: Choice classifier** (~0.5 day) — Validate ATL+NAc classification accuracy. Runs after pipeline is verified (NAc actually learns now).
3. **DM schema + validator** (~1 day) — CampaignDef, CascadeSpec, validator rules
4. **DM runtime + entity wiring** (~1 day) — state machine, cascade resolver, expectation + pipeline health checker
5. **Campaign 1: The Heist** + persona + orchestrator + end-to-end run (~1 day)
6. **Campaign 2: The Poisoned Crown** — different bio-system targets, validate schema generality (~1 day)
7. **Campaign 3: The Arena** — stress test, verify Cerebellum learning curve + pain saturation (~0.5 day)
8. **Pipeline health review** across all 3 campaigns — verify recall precision, echo rate, learning curves (~0.5 day)
9. **Iterate** on classifier + NPC dialogue + entity tool filtering based on findings (~1-2 days)

**Ship gate:** **three** showcase campaigns run end-to-end with:
- Readable NPC dialogue (NarrativeModulator-generated)
- Branch selection working (choice classifier >=70% accuracy)
- Seeded dice reproducible
- Entity state changes flowing through PainBus
- Cerebellum forming predictions by mid-campaign
- Rollup report populated with entity snapshots + bio-system expectation pass/fail
- **Pipeline health passing** — recall precision > 0.7, echo rate < 0.15, NAc learning monotonic, pain 100% entity-sourced, concept grounding verified

---

## File Inventory

**New files (~1,020 LOC, excluding spike):**
- `src/maxim/simulation/dm_runtime.py` (~240 — state machine, cascades, expectation checker, ablation mode)
- `src/maxim/simulation/dm_schema.py` (~150 — CampaignDef, CascadeSpec, validator, DAG cycle detection)
- `src/maxim/simulation/tools_dm.py` (~60 — advance_encounter, record_choice, roll_dice, get_campaign_state)
- `src/maxim/embodiment/component_db.py` (~150 — ComponentDB: search, compose, validate, index)
- `src/maxim/simulation/tools_components.py` (~40 — BrowseComponentsTool)
- `tests/unit/test_dm_schema.py` (~80)
- `tests/unit/test_dm_runtime.py` (~100)
- `tests/unit/test_component_db.py` (~60)
- `scenarios/campaigns/heist_v1.yaml` (~150 — Campaign 1: Memory + Causality + Pain)
- `scenarios/campaigns/poisoned_crown_v1.yaml` (~180 — Campaign 2: Temporal + Semantic + Relationship)
- `scenarios/campaigns/arena_v1.yaml` (~120 — Campaign 3: Pure stress test)
- `data/sem_components/` (~20 component YAML files — initial library)

**Modified files:**
- `src/maxim/simulation/orchestrator.py` — `dm` sim mode, entity tool auto-gen, DM + component tool registration (~40)
- `src/maxim/simulation/personas.py` — `dungeon_master` persona (~20)

**Phase 0 files (spike + pipeline fixes):**
- `scripts/spike_dm_pipeline_audit.py` (~250 — instruments pipeline, runs test encounters, produces report)
- `scripts/spike_dm_classifier.py` (~150 — classification accuracy test)
- `docs/experiments/dm_pipeline_audit.md` — pipeline audit findings
- `docs/experiments/dm_classifier_spike.md` — classification results
- Pipeline fixes in existing codebase (~50-100 LOC across memory_agent.py, concept_extractor.py, etc.)
