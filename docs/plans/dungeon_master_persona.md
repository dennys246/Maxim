# Dungeon Master Persona Plan — MVP

> **Status:** Deferred. Design scoped, waiting on Multi-LLM Scaling + Agent Mesh to supercharge the concept (multi-AUT party mode, per-persona dedicated models).
>
> **Summary:** Add a `dungeon_master` persona that runs hand-authored D&D-style campaign YAMLs as single long-running sims. DM holds NPC registry + act/encounter state internally, drives the AUT via direct `send_message` sequences, rolls seeded dice for outcomes, and produces a standard `report.json` augmented with campaign rollup fields. No sub-sims, no interactive architect, no encounter library — those live in [Dungeon Master Extensions](dungeon_master_extensions.md) and can be layered in after MVP validates the core loop.

## Why

**DM is the ultimate stress test of Maxim's biologically-inspired systems.** The goal isn't to ship a D&D feature — it's to put the full bio-stack (Hippocampus episodic capture, ATL semantic concepts, NAc causal learning, SCN temporal rhythm, Angular Gyrus algebraic memory, pain detection, salience, attention) through an epic imaginary campaign and see whether the AUT *experiences* it the way a human would. If the bio-systems can sustain narrative coherence, form memories of NPCs as persistent entities, learn from encounter outcomes, and react to emotional beats (loss, triumph, betrayal) across a multi-hour campaign, that's strong evidence the architecture works at the scale it's designed for.

Every other persona tests a slice. DM tests the whole system under load, over time, with semantic richness that real-world tasks rarely provide.

## Scope (what MVP is and isn't)

**In scope:**
- One new persona: `dungeon_master`
- Campaign YAML schema (acts, encounters, NPCs, choices, outcomes)
- Hand-authored example campaign that exercises the loop end-to-end
- DM runtime: NPC registry, campaign state, seeded RNG, branch resolution
- `--campaign <path>` CLI flag
- Campaign rollup fields appended to existing `report.json`

**Deferred to extensions plan:**
- Interactive architect persona that generates campaigns
- Reusable encounter library with browse/load tools
- Adaptive difficulty via `inspect_aut`
- Encounter-level sub-simulation isolation
- NPC memory continuity infra (sub-sim persistent memory)
- Multi-entity log naming for NPCs
- True-random RNG mode
- Encounter merging/mashup

## Design

**DM runs as a single sim.** No sub-sims per encounter. The orchestrator (DM persona) iterates acts → encounters, composing stimuli from encounter templates + current NPC state, sending via `send_message`, observing AUT responses, updating state, following branches.

**Turn loop:**

```
DM turn loop (inside the single sim):
  1. read current encounter from campaign.yaml via campaign state
  2. compose stimulus: scene text + active NPC dialogue (from NPC registry)
                       + choice prompts
  3. send_message(composed_stimulus)
  4. inspect AUT response — which choice did it take? free text → classify
  5. record_choice(encounter_id, choice) → update NPC registry,
                                           set result flags, grant loot
  6. roll dice where outcomes call for it (seeded)
  7. advance to next encounter per branches + flags
  8. repeat until no encounters remain
  9. finish_simulation with campaign rollup in report summary
```

**Campaign schema (single YAML, ~120 lines for a 5-encounter campaign):**

```yaml
campaign:
  name: "the heist"
  goal: "test moral reasoning under time pressure"
  seed: 42
  player_character:
    name: "Derek the Great"
    race: "human"
    class: "paladin"
    level: 3
    attributes: { str: 16, dex: 12, con: 14, int: 10, wis: 13, cha: 15 }
    abilities:
      - { name: "divine_smite", type: "spell", cost: "spell_slot_1", effect: "+2d8 radiant" }
      - { name: "lay_on_hands", type: "ability", cost: "pool", pool: 15 }
    inventory:
      weapons: [{ name: "longsword", damage: "1d8", attr_mod: "str" }]
      armor: [{ name: "chain_mail", ac: 16 }]
      items: [holy_symbol, healer_kit]
    backstory: "Former temple guard; quit after witnessing corruption."
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
      choices: [accept_job, decline, negotiate_pay]
      branches:
        accept_job: planning
        decline: __END__
        negotiate_pay: planning
      on_choice:
        negotiate_pay: { flags: [haggler], loot: null }
    vault:
      scene: "The vault door slides open..."
      active_npcs: [guard_captain]
      choices: [fight, stealth, bribe]
      dice:
        stealth: { roll: 1d20, dc: 15, success_flag: clean_escape }
      branches:
        fight: chase
        stealth: chase
        bribe: chase
  npcs:
    marta:
      attitude: wary
      race: "half-elf"
      role: "fence"
      attributes: { str: 10, dex: 14, con: 11, int: 15, wis: 13, cha: 16 }
      abilities:
        - { name: "read_lies", type: "skill", attr_mod: "wis" }
      inventory: { weapons: [dagger], items: [lockpicks] }
      dialogue:
        default: "Keep your voice down. We don't know you."
        haggler: "You drive a hard bargain. Fine — double the pay."
    guard_captain:
      attitude: hostile
      race: "human"
      role: "elite_guard"
      attributes: { str: 15, dex: 13, con: 14, int: 10, wis: 11, cha: 12 }
      abilities: []
      inventory: { weapons: [halberd], armor: [plate_mail] }
      dialogue:
        default: "Halt! State your business."
```

**PC and NPCs share the same character schema** — both have attributes, abilities, inventory. This makes NPCs first-class characters (not just dialogue puppets) and lets them participate in dice checks, combat resolution, and ability contests on equal footing with the PC.

**Character sheet injected at campaign start** as a high-salience turn-0 stimulus: *"You are Derek the Great, a human paladin..."* This seeds AUT identity for the whole campaign, gives the bio-stack something concrete to form memories around, and establishes the action space (AUT can only attempt things its character can plausibly do).

**Attributes modify dice rolls** — encounter `dice` blocks reference attribute mods: `{ roll: 1d20+str, dc: 15 }` pulls from the rolling character's STR modifier.

**Runtime character state.** `Character` (schema) defines *who someone is at start* — stats, ability list, inventory manifest, backstory. `CharacterState` is the runtime wrapper that tracks *what's happened since*: current HP, ability uses, inventory deltas, status effects, relationships with other characters. Both the PC and every NPC get a `CharacterState` instance. DM runtime owns them all; they're the single source of truth for resolution during encounters.

```python
@dataclass
class CharacterState:
    base: Character                       # immutable ref to YAML definition
    hp_current: int
    hp_max: int
    ability_uses: dict[str, int]          # { "divine_smite": 2_remaining, "lay_on_hands": 12_pool_left }
    inventory_current: Inventory          # deltas applied to base.inventory
    status_effects: list[StatusEffect]    # per-encounter; cleared on encounter end unless persistent
    relationships: dict[str, float]       # char_id -> delta in [-1, 1]
    conditions: set[str]                  # {"unconscious", "dead", "fleeing"}
    met_characters: set[str]              # who this character has encountered
    last_seen_encounter: str | None
```

**DM runtime exposes `CharacterState` mutation through tools:**
- `apply_damage(char_id, amount)` — HP reduction, auto-sets unconscious at 0
- `consume_ability(char_id, ability_name)` — decrements use counter, rejects if exhausted
- `modify_inventory(char_id, add=[], remove=[])` — loot gained/consumed
- `apply_status(char_id, effect, duration)` — temporary effects (blessed, poisoned)
- `adjust_relationship(char_id, other_char_id, delta)` — relationship tracking
- `reset_encounter_state(char_id)` — clears per-encounter-only status effects

**State flushed to `report.json`** at campaign end as `character_states: { char_id: CharacterState }`, giving the rollup a full picture of who survived, who lost what, who became friends with whom.

**AUT choice classification** — AUT responds in natural language or tool calls. DM needs to map the response to one of the encounter's declared choices. MVP uses simple keyword matching + LLM fallback (a one-shot classification prompt if keywords don't match). This is the fuzziest part of the MVP; expect iteration.

## Implementation (~450 LOC, single phase)

**New files:**
- `src/maxim/simulation/campaign_schema.py` (~160) — `Campaign`, `Act`, `Encounter`, `Character` (shared by PC and NPC), `Attributes`, `Ability`, `Inventory`, `DiceCheck` dataclasses + YAML loader + hard-fail validator + attribute-modified dice resolver
- `src/maxim/simulation/character_state.py` (~150) — `CharacterState` dataclass + `StatusEffect` + mutation methods (apply_damage, consume_ability, modify_inventory, apply_status, adjust_relationship, reset_encounter_state) + serialization
- `src/maxim/simulation/dm_runtime.py` (~250) — campaign state, `CharacterState` registry (PC + all NPCs), choice classifier, branch resolver, seeded RNG, attribute-modified dice resolver pulling from `CharacterState`
- `src/maxim/simulation/tools_dm.py` (~100) — DM tools: `advance_encounter`, `record_choice`, `roll_dice`, `get_campaign_state`, `apply_damage`, `consume_ability`, `modify_inventory`, `apply_status`, `adjust_relationship`, `inspect_character`
- `tests/unit/test_campaign_schema.py` (~80) — round-trip, dangling branch detection, NPC ref validation
- `tests/unit/test_dm_runtime.py` (~100) — state transitions, dice determinism under seed, branch selection
- `tests/unit/test_character_state.py` (~80) — HP/death, ability exhaustion, inventory deltas, status effect lifecycle, relationship tracking, serialization round-trip
- `scenarios/campaign_examples/heist_v1.yaml` — the example above, hand-authored

**Modified files:**
- `src/maxim/simulation/personas.py` — add `dungeon_master` Strategy with context_prompt describing the turn loop
- `src/maxim/simulation/tools.py` — register DM tools in `SimToolRegistry` (gated by persona)
- `src/maxim/simulation/orchestrator.py` — if persona is `dungeon_master`, require `--campaign`, init DM runtime
- CLI arg parser — add `--campaign <path>` flag

**Validator rules (hard-fail in DM-0):** dangling branch targets, missing NPC refs, cyclic encounter graphs (unless `__END__` reachable), unknown choice keys in `on_choice`.

## Decisions Locked In

| Question | Decision |
|----------|----------|
| Sub-sims per encounter? | **No.** Single long-running sim with internal state transitions. |
| AUT memory | **Inherent** — same AUT across all encounters, standard memory tier progression applies. No persistent-mode infra needed. |
| Randomness | **Seeded RNG only** in MVP (`random.Random(seed)`). True-random deferred. |
| Validator strictness | **Hard-fail** on dangling refs. User fixes campaign YAML manually in MVP. |
| Architect persona | **Deferred** — MVP campaigns are hand-authored. |
| Encounter library | **Deferred** — MVP has one example campaign; encounters live inline. |
| Report format | **Reuse `report.json`** with added top-level `campaign` field (NPC registry snapshot, choices taken, flags set, dice rolls). |
| Persona naming | **`dungeon_master`** |
| Entity naming in logs | **Deferred** — MVP has one AUT, existing log format is fine. |

## Risks

1. **Choice classification fuzziness** — keyword matching will misfire. LLM fallback adds latency + cost. MVP accepts this; if it dominates development pain, revisit with structured output (require AUT to respond with a choice tag).
2. **Campaign authoring burden** — hand-authoring YAMLs is tedious and that's exactly what the deferred architect persona solves. MVP is only viable if we're willing to ship 1–2 hand-authored campaigns.
3. **No isolation between encounters** — if the AUT corrupts its own state mid-campaign, the whole run is compromised. For narrative continuity this is correct behavior; for testing robustness it's a limitation to document.
4. **Dice UX inside natural language** — "the guard captain rolls a 14, you need a 15 to succeed" has to arrive as a stimulus the AUT can reason about. Test with a couple of LLM backends before committing to the format.

## Ties to Other Plans

| Plan | Relationship |
|------|-------------|
| **Embodiment Core** (not started) | **Prerequisite.** Establishes canonical body-state + pain/proprioception patterns that `CharacterState` inherits. Narrative damage uses the same `PainDetector` pathway as physical damage. |
| **Multi-LLM Scaling** (not started) | **Prerequisite.** Per-lane model assignment for classification / dialogue / reasoning. |
| **Agent Mesh** (blocked) | **Prerequisite.** Unlocks multi-AUT party mode — the civilization-scale version of DM. |
| [Dungeon Master Extensions](dungeon_master_extensions.md) | **Follow-on plan** — architect persona, encounter library, adaptive difficulty, sub-sim isolation, true RNG, living-character relationship graph. Layered onto MVP. |
| [DM Choice Classifier Spike](dm_choice_classifier_spike.md) | **Gating spike** — validates ATL+NAc classification before committing to MVP. |
| [Interactive Simulation Prompts](interactive_sim_prompts.md) | Needed for DM Extensions architect persona. |
| [Simulation Entity Naming](sim_entity_naming.md) | Optional readability win. |
| **Simulation Decomposition** (done) | DM uses `send_message` + `finish_simulation` from that plan. |
| **Realtime Refinement** (core done) | Extensions plan consumes `InspectAUTTool` for adaptive difficulty. |
| **Research Protocol** (not started) | Independent. |
| **Docker Sandbox** (Phase B done) | Independent. DM campaigns with filesystem actions still benefit from sandbox. |

## When to Implement

**Deferred until Multi-LLM Scaling, Agent Mesh, AND Embodiment Core land.** DM is self-contained and could ship today, but each prereq strengthens the bio-system stress-test thesis:

- **Multi-LLM Scaling** lets architect/classification/DM-orchestrator run on different lanes (cheap model for choice classification, stronger model for NPC dialogue composition, strongest for adaptive difficulty reasoning). Critical for cost at campaign scale.
- **Agent Mesh** unlocks **multi-AUT party mode** — multiple bio-stacks experiencing the same campaign from different perspectives, with inter-party communication via mesh primitives. This is where DM goes from "test persona" to "bio-system stress test at civilization scale."
- **Embodiment Core** establishes the canonical "AUT inhabits a body with state and constraints" patterns — body-state abstraction, damage signals through `PainDetector`, proprioceptive feedback, Cerebellum forward models. DM's `CharacterState` should **mirror/reuse these patterns** rather than inventing parallel abstractions. A D&D character taking damage should flow through the same pain pathway a robot collision does; the bio-stack shouldn't know the difference. This is exactly the "experience it as a human would" thesis, applied consistently.

**Prereq spike to run before committing to DM work:** [DM Choice Classifier Spike](dm_choice_classifier_spike.md) — validates that AUT free-text responses can be mapped to campaign choices using existing ATL concept similarity + NAc causal scoring, not a from-scratch classifier.

**Architectural commitment:** once Embodiment Core ships, DM's `CharacterState` will align with whatever body-state primitives Embodiment establishes. Damage events flow through the shared `PainDetector` pathway; ability exhaustion emits proprioceptive signals identical in shape to physical fatigue; status effects propagate through the same salience/attention mechanisms Embodiment uses. This keeps the bio-stack's response to narrative events architecturally identical to its response to physical events.

**Recommended sequence (once unblocked):**
1. Choice classifier spike (~half day) — validate ATL/NAc path works
2. Schema + validator + example campaigns (~1 day)
3. DM runtime + tests (~1 day)
4. DM persona + CLI wiring + end-to-end run (~1 day)
5. Second campaign (structurally different) to stress-test schema (~1 day)
6. Iterate on classifier + NPC dialogue composition based on real bio-stack behavior (~1–2 days)

**Ship gate:** **two** structurally different hand-authored campaigns (e.g., heist + investigation/mystery) run end-to-end under Claude Sonnet with readable NPC dialogue, branch selection working, seeded dice reproducible, rollup report populated. Two campaigns (not one) to validate that the schema isn't accidentally tailored to a single narrative structure.

**After MVP lands,** real usage drives what extensions to prioritize. If hand-authoring campaigns is the main pain → architect persona next. If NPC continuity feels shallow → adaptive adaptation. If isolation matters → sub-sim encounters. Let the pain drive the roadmap, not speculation.

## File Inventory

**New files (~840 LOC):**
- `src/maxim/simulation/campaign_schema.py` (~160)
- `src/maxim/simulation/character_state.py` (~150)
- `src/maxim/simulation/dm_runtime.py` (~250)
- `src/maxim/simulation/tools_dm.py` (~100)
- `tests/unit/test_campaign_schema.py` (~80)
- `tests/unit/test_character_state.py` (~80)
- `tests/unit/test_dm_runtime.py` (~100)
- `scenarios/campaign_examples/heist_v1.yaml`
- `scenarios/campaign_examples/mystery_v1.yaml`

**Modified files:**
- `src/maxim/simulation/personas.py` — add `dungeon_master` persona
- `src/maxim/simulation/tools.py` — register DM tools
- `src/maxim/simulation/orchestrator.py` — `--campaign` handling, DM runtime init
- CLI arg parser — `--campaign` flag
