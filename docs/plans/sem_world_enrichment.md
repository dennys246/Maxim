# SEM World Enrichment — Rich Entity Environments

**Status:** PHASES 1+2 SHIPPED. Phase 3 (composable body archetypes) PARTIAL — archetype YAMLs exist in `_data/components/archetypes/`, avatar migration not yet done. 1.0 vs 1.1 scope decision pending.
**Depends on:** PR #190 (sem-damage-autosense) merged

## Motivation

The imagination pipeline resolves individual entity phrases reactively — one at a time, mid-conversation, after 2+ mentions. This means:

1. **Entities arrive late.** The AUT's first 2-3 turns in a sim with a dragon have no dragon SEM — no sensors, no affordances, no pain cascade. By the time imagination fires, the narrative has moved on.
2. **The orchestrator generates rich entity descriptions that go to waste.** The narrator's opening scene mentions creatures, weapons, and environments, but that text only reaches the AUT as a percept — the SEM layer never sees it until the imagination trigger post-processes.
3. **Entity generation is narrow.** Only things the narrator explicitly describes get imagined. A dungeon scene should have doors, torches, traps, rats — a *world* — not just the one creature the narrator happened to name.

The goal: make every simulation start with a rich, pre-instantiated SEM world where the agent can immediately sense, interact with, and learn from its environment.

## Prior art / what exists

- `ImaginationTrigger.process_percept()` — reactive entity extraction + resolution
- `_ensure_entity_live()` — instantiates seed components as scene entities
- `_encode_entity_affordances()` — substrate path for affordance concept transfer
- `Narrator` — LLM-driven scene generation (two-call: decide + generate)
- `_load_world_entities()` — loads SEM entities from arc metadata (YAML-declared)
- `BioEnrichmentPipeline` — thalamic relay that queries hippocampus/NAc/ATL/ComponentIndex per percept
- `ComponentIndex` — two-layer discovery (alias + embedding), 65 seed components
- `EntityMap` — self/scene ownership model
- `bodies/base_humanoid.yaml` — existing humanoid body template for sim agents

## Phase 1: Scene Manifest Pre-trigger

**Goal:** All major entities exist as live SEMs before the first AUT turn.

### Design

Add a `process_manifest()` method to `ImaginationTrigger` that:
- Takes raw text describing a scene's entity inventory
- Bypasses arousal gate, energy gate, and mention threshold (these are reactive safeguards; pre-triggering is a deliberate act by the orchestrator)
- Runs the same extraction → index → design pipeline
- Returns all resolved entities

Add a `generate_scene_manifest()` method to `Narrator` that:
- Takes the simulation goal
- Makes a single focused LLM call: "List the key physical entities (creatures, weapons, items, NPCs, environmental features) that should appear in this scenario. One per line, with a brief physical description."
- Returns the raw text for `process_manifest()`

Wire in `orchestrator.py` between ImaginationTrigger wiring (~line 1325) and the AUT thread launch (~line 1391). This is after EntityMap, ComponentRegistry, and ImaginationDesigner are all constructed, so all dependencies are satisfied.

### Key decisions

- **Manifest is text, not structured JSON.** The same `extract_entity_phrases()` pipeline parses it, which means the manifest benefits from the same indicator/stop word filters. The LLM doesn't need to produce valid JSON — it just describes a scene.
- **process_manifest() is a separate method, not a config toggle on process_percept().** This makes the pre-trigger contract explicit and prevents gate-ordering drift if DN initialization moves in a future refactor (architecture review finding #2).
- **Dedup with arc world_entities:** If the arc already declares `world_entities` in metadata (loaded by `_load_world_entities`), those are already live. `_ensure_entity_live` checks EntityMap before instantiating, so duplicates are harmless. The manifest adds entities the arc didn't declare.
- **Self-entity guard:** `process_manifest()` must skip any entity whose ref or name matches the AUT's self-entity. The self-entity is registered via `entity_ref` / `register_self()` before the manifest runs. Calling `register_scene()` for the same entity would collide with the self-entity ownership invariant (executor review finding #1).
- **Hard cap: 8 entities total, 5 LLM designs max.** Process ComponentIndex hits first (instant, no cap). Then budget at most 5 LLM design calls for truly novel entities (~2-4s each = 10-20s max). Log and skip any beyond the cap (executor review finding #2).
- **Failure is non-fatal.** If the manifest LLM call fails, log a warning and proceed — the reactive pipeline still works.

### Files touched

| File | Change |
|------|--------|
| `imagination/trigger.py` | Add `process_manifest()` method |
| `simulation/narrator.py` | Add `generate_scene_manifest()` method |
| `simulation/orchestrator.py` | Wire pre-trigger between imagination wiring and thread launch |
| `tests/unit/test_imagination.py` | Test process_manifest bypasses gates, caps entities, deduplicates |
| `tests/unit/test_narrator.py` | Test manifest generation prompt + fallback on LLM failure |

### Risks

- **Token cost:** One extra LLM call per sim start (~200 output tokens). Negligible vs the narrator's per-turn calls.
- **Ordering:** Must run after ImaginationTrigger._entity_map is assigned (line ~1299). Placement in orchestrator.py is straightforward — this is a linear setup sequence.


## Phase 2: Bio-Enrichment Entity Detection

**Goal:** Nouns the bio-pipeline recognizes as entity-like automatically trigger entity resolution.

### Design (refined by architecture review)

The architecture review flagged a **critical layering violation** if bio_enrichment directly mutates EntityMap. The fix: bio_enrichment *detects* entity candidates but does not *instantiate* them. The caller (agent_loop) routes candidates to ImaginationTrigger.

Concretely:

1. Add `entity_candidates: tuple[str, ...] = ()` field to `EnrichmentResult`.
2. In `BioEnrichmentPipeline._extract_keywords()` (which already tokenizes the text), check each keyword against `_ENTITY_INDICATORS` from trigger.py. Collect matching phrases.
3. With `ComponentIndex` available (already wired at `self._component_index`), do **Layer 1 alias lookup only** (O(1), no embedding) for each candidate. If found, include in `entity_candidates`.
4. In `agent_loop.py`, after bio_enrichment returns, forward `result.entity_candidates` to `imagination_trigger.process_percept()` if the trigger is available.

### Key decisions

- **Alias-only lookup in the hot path.** The architecture review flagged the 50ms latency budget. Embedding computation (~5ms per query) on 10 nouns = 50ms, blowing the budget. Layer 1 alias lookup is O(1) — effectively free.
- **Detection, not instantiation.** BioEnrichmentPipeline returns candidates; agent_loop forwards them. This preserves the unidirectional flow: bio-enrichment reads the world, ImaginationTrigger mutates it. No layering violation.
- **ThinkTool bypass_gate=True paths don't spawn entities from deliberation.** The architecture review caught this: if bio_enrichment directly spawned entities, internal thoughts would literally imagine things into existence. The routing-only approach means agent_loop can gate entity forwarding on `not bypass_gate` — only external percepts trigger entity resolution.
- **Dedup with reactive trigger.** The reactive ImaginationTrigger already runs on the same percept text. The bio-enrichment path adds value by catching nouns the extraction regex misses (e.g., nouns in the middle of sentences that don't match intro patterns). The ImaginationCache deduplicates resolution.

### Files touched

| File | Change |
|------|--------|
| `integration/bio_enrichment.py` | Accept `resolved_entities` from EnrichmentContext, surface richer affordances |
| `embodiment/component_index.py` | Add `find_alias_only()` method (Layer 1 + 1b only, no embedding) |
| `runtime/agent_loop.py` | Pass imagination results into enrichment context |
| `tests/unit/test_bio_enrichment.py` | Test enriched affordance output from resolved entities |

### Risks

- **Minimal.** This phase adds no new entity detection path — it enriches bio_enrichment output using entities the ImaginationTrigger already resolved. No double-counting, no hot-path regression, no layering violation.


## Phase 3: Composable Body Templates

**Goal:** Every entity has a composable body structure. Maxim has a body in every context.

### Design

This is the largest phase and should be broken into sub-stages.

#### Stage 3a: Body archetype taxonomy

Define body archetypes as YAML templates in `_data/components/archetypes/`:

```yaml
# archetypes/quadruped.yaml
archetype:
  name: quadruped
  description: Four-legged creature (wolves, horses, dragons, dogs)
  body_parts:
    head:
      sensors: [awareness]
      affordances: [bite, roar]
    torso:
      sensors: [hp, armor_integrity]
      affordances: []
    front_legs:
      sensors: [mobility]
      affordances: [claw_strike, pounce]
    hind_legs:
      sensors: [mobility]
      affordances: [kick, leap]
    tail:
      sensors: []
      affordances: [tail_sweep]
  # Optional body parts (not all quadrupeds have these)
  optional_parts:
    wings:
      sensors: [altitude]
      affordances: [take_flight, dive_attack, land, circle]
    breath_weapon:
      sensors: [charge]
      affordances: [fire_breath, frost_breath, acid_spray]
```

Archetypes: `humanoid`, `quadruped`, `serpentine`, `avian`, `vehicle`, `machine`, `environmental`.

#### Stage 3b: Upgrade seed components

Add an `archetype` field to existing component YAMLs. This is **additive and backward-compatible** — components without `archetype` work exactly as before. The field is advisory metadata for the designer and display, not a structural requirement.

```yaml
# creatures/dragon.yaml (upgraded)
component:
  name: dragon
  tags: [creature, predator, hostile, flying, fantasy]
  synonyms: [wyrm, drake, fire breather, winged serpent, ...]
  category: creatures
  archetype: quadruped  # NEW — advisory, used by designer and display

entity:
  # ... existing spec unchanged ...
```

Validation: a CI script checks that every seed component's `archetype` field (if present) names a valid archetype YAML. User components in `~/.maxim/components/` are not validated — they're opt-in.

#### Stage 3c: Designer archetype selection

When `ImaginationDesigner` generates a new entity, the LLM prompt includes archetype descriptions. The designer selects one and the generated spec inherits the archetype's body-part structure as a starting scaffold.

The LLM can customize (add/remove optional parts, adjust sensors) but cannot invent body parts outside the archetype's vocabulary. This bounds the generation space and prevents nonsensical anatomy (humanoid dragon → caught because dragon description says quadruped).

Validation: `ImaginationDesigner._quick_validate()` checks that the generated spec's modulator names are a subset of the archetype's `body_parts` + `optional_parts` keys. Mismatch = regenerate or fall back to template.

#### Stage 3d: Maxim's own body

Two new seed components:

- `bodies/maxim_sim_avatar` — humanoid archetype, default embodiment in sim mode when no `--embodiment` is specified. Sensors: health, stamina, awareness. Affordances: move, look, listen, speak, pick_up, use. This is `base_humanoid` with a name that clarifies it's Maxim's body, not a generic template.

- `bodies/host_machine` — machine archetype, scene entity (not self) in production mode. Sensors: cpu_usage, memory_usage, disk_usage, gpu_usage, network_latency (fed by psutil/nvidia-smi). No callable affordances — the agent observes its host but doesn't control hardware via SEM. Hardware-control actions (restart service, clear cache) stay in the tool registry where autonomy gating is mature.

**Why scene, not self for host_machine:** The architecture review caught this. Self-entities get callable affordance tools. Giving an AI agent `shutdown` or `kill_process` affordances through SEM bypasses the autonomy/permission model. Model the host as a scene entity with read-only sensors. If production monitoring needs actions, those are standard tools with proper gating, not SEM affordances.

#### Stage 3e: Default embodiment in sim

When `--embodiment` is not specified, the orchestrator auto-selects `bodies/maxim_sim_avatar` as the AUT's self-entity. This means every sim has a body — the agent can always `sense` itself, and pain cascades from combat apply to its own health sensor.

### Files touched (across all sub-stages)

| File | Change |
|------|--------|
| `_data/components/archetypes/*.yaml` | NEW: 7 archetype templates |
| `_data/components/creatures/*.yaml` | Add `archetype` field to existing seeds |
| `_data/components/bodies/maxim_sim_avatar.yaml` | NEW: Maxim's sim body |
| `_data/components/bodies/host_machine.yaml` | NEW: production host sensors |
| `embodiment/archetype.py` | NEW: archetype loading + validation |
| `simulation/entity_designer.py` | Archetype selection in LLM prompt |
| `imagination/designer.py` | Archetype scaffold in generated specs |
| `simulation/orchestrator.py` | Default embodiment when --embodiment absent |
| `embodiment/spec.py` | Parse archetype field (optional, backward-compat) |

### Risks

- **Migration surface:** 65 seed YAMLs need `archetype` added. Automated via script + CI validation. User components in `~/.maxim/` are unaffected (field is optional).
- **LLM archetype selection accuracy:** If the LLM picks "humanoid" for a dragon, the body structure is wrong. Mitigation: the archetype descriptions include explicit examples ("quadruped: wolves, horses, **dragons**, dogs"). Validation catches mismatches.
- **Default embodiment changes sim behavior.** Currently, sims without `--embodiment` have no body. Adding one means the AUT can sense itself and takes damage. This is intentional — it's the "every sim has a rich SEM world" goal — but it changes baseline behavior. Gate behind `--no-embodiment` for opt-out.


## Implementation Order

```
Phase 1 (scene manifest)     ← smallest, highest impact, unblocks rich worlds
  └─ Phase 2 (bio-enrichment routing)  ← leverages existing infrastructure
       └─ Phase 3a (archetypes)
            └─ Phase 3b (seed upgrades)
                 └─ Phase 3c (designer integration)
                      └─ Phase 3d (Maxim's body)
                           └─ Phase 3e (default embodiment)
```

Phase 1 can ship independently. Phase 2 can ship independently. Phase 3 sub-stages are sequential within the phase but the whole phase can be deferred without blocking 1 or 2.


## Review Findings (Architecture + Executor lenses)

### Architecture Review

1. **[Critical] Phase 2 layering violation.** Bio-enrichment triggering EntityMap mutations would couple substrate encoding to embodiment. **Resolution:** routing-only approach — bio_enrichment returns candidates, agent_loop forwards to ImaginationTrigger. Adopted in Phase 2 design above.

2. **[Important] Phase 1 bypasses arousal/energy gates.** Pre-trigger must not depend on DN state which may not be initialized. **Resolution:** dedicated `process_manifest()` method with explicit bypass. Adopted.

3. **[Important] Phase 1 orchestrator ordering.** Pre-trigger must run after EntityMap + ImaginationTrigger wiring but before thread launch. **Resolution:** placed between lines ~1325 and ~1391. Linear dependency chain, no circularity.

4. **[Important] Phase 2 hot-path performance.** Embedding lookups (~5ms each) on 10 nouns would blow the 50ms budget. **Resolution:** alias-only Layer 1 lookup in bio_enrichment. Embedding reserved for ImaginationTrigger's dedicated path.

5. **[Minor] Phase 3 host-machine is scene, not self.** SEM self-entities get callable tools; giving hardware-control affordances to an AI agent bypasses autonomy gating. **Resolution:** host_machine is a scene entity with read-only sensors. Actions stay in the tool registry.

6. **[Minor] ComponentRegistry read-only invariant preserved.** All imagination uses `register_ephemeral()` (session-scoped overlay). Confirmed no violation.

### Executor Review

1. **[Critical] Phase 1: Self-entity collision.** If the manifest lists the same entity as the AUT's `--embodiment` (e.g., "a warrior with a rusty sword"), `_ensure_entity_live` calls `register_scene()` for something that should be self. **Resolution:** `process_manifest()` must check `_entity_map.resolve(name)` before instantiating AND skip any entity ref matching the self-entity ref. Added explicit guard to Phase 1 design.

2. **[Critical] Phase 1: Unbounded LLM design calls.** If the manifest lists 20 novel entities not in ComponentIndex, each triggers a sequential LLM design call (~2-4s each = 40-80s blocking before sim starts). **Resolution:** Hard-cap at 8 entities. Process ComponentIndex hits first (instant), then budget max 5 LLM design calls. Log and skip the rest. Adopted in Phase 1 design.

3. **[Critical] Phase 2: Double resolution.** ImaginationTrigger.process_percept() already runs on percept text in agent_loop. If bio_enrichment also extracts entities from the same text, mention counts increment twice per phrase per percept — a single-mention entity reaches threshold=2 and triggers design prematurely. **Resolution:** Reconsider Phase 2 design — instead of bio_enrichment doing independent entity detection, have agent_loop pass ImaginationTrigger results (already computed) into the enrichment context. Bio_enrichment surfaces affordances from *already-resolved* entities rather than detecting new ones. This avoids the double-counting footgun entirely. Phase 2 design updated below.

4. **[Important] Phase 1: Vague goals waste LLM call.** "test memory" produces zero entity phrases — one wasted call. **Resolution:** Accept as negligible cost (~200 tokens). The fallback is an empty manifest which is handled gracefully. Not worth adding goal-complexity heuristics.

5. **[Important] Phase 2: Enforce alias-only in API.** Callers must not accidentally use embedding lookup in the hot path. **Resolution:** Add `ComponentIndex.find_alias_only(query)` method that only checks Layer 1 + Layer 1b. Adopted.

6. **[Important] Phase 3: Seed YAML migration validation.** 65 YAMLs updated, risk of parse failures. **Resolution:** Parametrized pytest that loads every YAML in `_data/components/` through `_parse_entity`. User components must treat archetype as optional.

7. **[Important] Phase 3: Body template selection correctness.** LLM might assign "humanoid" to a dragon. **Resolution:** Post-design validation with keyword→archetype map override (dragon/serpent→quadruped, wolf/horse→quadruped). Log warning on override.

8. **[Minor] Phase 1: Testing strategy.** Minimum: (a) unit test for `process_manifest` with mock index, entity cap, self-entity guard; (b) unit test for `Narrator.generate_scene_manifest` with mock LLM; (c) integration test verifying EntityMap state after manifest. Existing test_imagination.py fixtures reusable.


## Phase 2 Design Revision (post executor review)

The executor review's finding #3 changes Phase 2 fundamentally. The original design had bio_enrichment independently detecting entities — but this double-counts with the existing ImaginationTrigger path on the same text.

**Revised design:** Bio_enrichment does NOT detect entities. Instead, the agent_loop passes already-resolved ImaginationTrigger results into the enrichment context. Bio_enrichment then surfaces affordances from those entities (via ComponentIndex) to enrich the ThinkTool response.

Concretely:
1. Add `resolved_entities: tuple[str, ...] = ()` to `EnrichmentContext` (not `EnrichmentResult`).
2. In agent_loop, after `imagination_trigger.process_percept()` returns results, populate `EnrichmentContext.resolved_entities` with the resolved entity refs.
3. In `BioEnrichmentPipeline._query_component_index()`, use `resolved_entities` to surface richer affordance information — the entity's full capability set, not just keyword-matched affordance names.
4. No new entity detection path. No double-counting. ImaginationTrigger remains the single entity resolution path.

This is simpler, avoids the layering violation AND the double-counting bug, and still achieves the goal: bio_enrichment produces richer context because it knows which entities are live in the scene.
