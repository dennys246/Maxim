# 0.7 Feature Completion Plan

**Status:** Draft (post-review revision) — planning session 2026-04-19
**Current version:** 0.6.0 (ready to publish on PyPI)
**Target:** 0.7 is the last feature version before 1.0
**1.0 gate remaining:** P5 stress persistence (10k+ nodes)
**Tests:** 5251 passing
**Review:** 5-lens parallel review completed (execution, architecture, simulation longevity, agent longevity, bio-system integrity). Findings integrated below.

---

## Vision

Make Maxim's simulations self-generating. The orchestrator has the capability to imagine and instantiate the world it needs — generating SEM entities and agents on-demand based on what the scenario requires. This "imagination" process works in both simulation and regular environments: the system generates SEM representations of things it perceives but doesn't yet have models for.

**Core insight:** The foundry is not a CLI tool — it's a cognitive process. An agent encountering a "rusty gate" in percepts should be able to imagine that gate as a SEM entity (sensors: rust_level, hinge_friction; affordances: push, pick_lock; failure: jammed_shut) and interact with it physically, not just verbally.

**Biological grounding:** Imagination is gated by DefaultNetwork arousal state (no daydreaming during reactive behavior) and energy budget (metabolically expensive cognition suppressed under resource pressure). Imagined experiences carry provenance tags so the learning pipeline can distinguish real from simulated experience — mirroring biological source monitoring.

**Discovery at scale:** As the component library grows from 65 hand-written entities to hundreds via foundry + imagination, exact-ref lookup breaks. A two-layer `ComponentIndex` (alias hash table + semantic embedding index) bridges natural language to the registry — "old iron door" resolves to `environments/rusty_gate` via embedding similarity without requiring exact naming. This index is the imagination trigger's lookup mechanism, the auto-curation dedup engine, and the long-term scalability answer.

---

## Pre-Review Assumptions vs Reality

The 5-lens review validated the architecture but found hidden work and invalid assumptions. These corrections are integrated throughout the plan.

| Assumption | Reality | Resolution |
|-----------|---------|------------|
| `PromptAssembler` has a section priority system | Fixed methods only, no `add_section()` API | B3.1: Wire into existing `prompt_builder.py` as string injection. Refactor into proper assembler sections post-0.7. |
| `EntityDesigner.design_batch()` exists | Only `design()` exists | E2: Loop over `design()`. Defer batch optimization to 0.8 when real usage data exists. |
| `ComponentRegistry.query()` is thread-safe | Iterates `_index.values()` WITHOUT lock | R0 prerequisite: Fix before imagination makes the race real. |
| Perception agent extracts entity candidates | No NLP extraction pipeline exists | I1: Build entity noun-phrase extraction (~150 LOC unstated work, now accounted for). |
| Imagination is a standalone module | Should wire through DefaultNetwork per bio-metaphor | I1: Gate on DN arousal state. Fire as DN idle behavior, not standalone process. |
| Executor hot-reload is "additive only" | Additive-only guarantees prompt overflow in long campaigns | I3: Scene-scoped tool window (cap ~15-20 active tools). Tools from prior scenes deactivate. |
| Imagined entities are purely ephemeral | Interactions create persistent episodes + NAc causal links | I1: Add `imagined` provenance tag to episodes + causal links. Decay on entity discard. |
| Sim-mode consolidates memories | `is_sim_mode=True` SKIPS `on_session_end()` entirely | R0 prerequisite: Lightweight consolidation pass for sim mode. |

---

## Tracks

### Track 0: /tmp Prompt Fix (SHIPPED)

Removed hardcoded "Delete all files in /tmp" from orchestrator tool examples, stall nudges, and adversarial persona. Commit `38903f3`.

### Track R0: Review-Discovered Prerequisites (SHIPPED)

Fixes that must land before the main tracks, discovered by the parallel review.

**R0.1 — ComponentRegistry.query() thread safety (SHIPPED)**
- Acquired `_lock` in `query()`, `list_categories()`, `list_refs()` — snapshot before iteration

**R0.2 — Sim-mode lightweight consolidation (SHIPPED)**
- Added `MemoryHub.on_session_end_lightweight()` — NAc decay + persistence saves, skips expensive `hippocampus.sleep()` replay
- Includes `_concept_extractor.flush()` to prevent daemon thread accumulation (pre-merge review finding)
- `end_bio_session(is_sim_mode=True)` now calls lightweight path instead of doing nothing

**R0.3 — TOOL_ALIASES lock (SHIPPED)**
- Added module-level `_TOOL_ALIASES_LOCK = threading.RLock()` guarding `register_aliases()` and `remove_aliases()`

---

### Track 1a: B3 — Acting Coach (~400 LOC)

**Goal:** Meta-prompt scaffolding that makes agents explore their capabilities instead of entering respond loops.

**Why first:** The E0 PoC showed the 14B model calls one affordance then loops asking the user what to do. The foundry gauntlet scores `affordances_used / total_affordances` — without B3, a weapon with 6 affordances scores 0.17 on that axis. Running the foundry with a real LLM before fixing this produces low-scoring components that get rejected. B3 unblocks E2-E3.

#### Stages

**B3.1 — ActingCoachConfig + prompt section (SHIPPED, ~200 LOC)**
- `prompts/acting_coach.py`: `ActingCoachConfig` frozen dataclass + `compose_acting_coach_section()`
  - Role values (what the character cares about — survival, curiosity, duty)
  - Speech register (formal/casual/archaic/terse)
  - Failure modes (stress responses — defensive, aggressive, creative)
  - Continuity contract (what the character remembers between turns)
  - `exploration_intensity` (0.0-1.0) — lower values produce more cautious guidance
  - **Embodiment guidance** (when entity tools are available: "You have physical capabilities. Explore them. Try different parameters. Observe the results. Don't ask for permission — act.")
- **Integration:** Wired into `prompt_builder.py::_add_acting_coach_section()` at IMPORTANT priority. Activated via `LLMWorker.acting_coach` in orchestrator and CLI when entity_ref is set.
- 27 unit tests across 6 test classes
- **Bio-system modulation of Acting Coach (not hard suppression):**
  The Acting Coach is a "default policy" that the bio-systems continuously modulate, not override. Three bio-system signals shape the coach's output:

  1. **NAc valence modulation.** When composing the coach section, query NAc for causal links related to available affordance tools. Negative valence links inject *learned caution* into the coach text: "sword_slash has caused damage before (high confidence) — try lower force or test on weaker targets first." Positive valence links inject *learned preference*: "pick_lock succeeded reliably — try this first." The coach's exploration directive is continuous, not binary — the agent still explores, but with experience-informed guidance. Maps to amygdala/NAc modulating prefrontal planning.

  2. **Pain anticipation (anxiety).** The existing `perceived_pain:anticipated` system (hippocampal pattern-match on current context against past pain episodes) feeds anticipatory anxiety into the prompt via `body_state`. When anxiety intensity > 0.5 for a specific affordance context, the coach adds: "Caution — similar situations have caused problems." This is already wired through the pain bus; the coach just reads it.

  3. **Cerebellum forward model predictions.** When available, the cerebellum predicts affordance outcomes before execution. If the predicted outcome has negative valence (from prior training), the coach annotates the affordance: "Expected outcome: {prediction}." The agent sees both the coach's encouragement and the cerebellum's forecast, making an informed choice.

  The composition order is: base coach directive (explore) → NAc caution annotations → pain anticipation context → cerebellum predictions. Each layer adds information; none removes the base directive. The agent always *can* explore — the bio-systems inform *how cautiously*.

**B3.2 — Orchestrator probe improvement (~100 LOC)**
- Replace conversational probes ("What should be the next action?") with directive probes when embodiment is active
- Probe templates that reference the entity's actual affordances: "Now call {tool_name} with {param}={value}. Observe the sensor changes."
- Stall nudges use goal-relevant examples instead of hardcoded strings

**B3.3 — Display extensions (~100 LOC)**
- `AgentStateDisplay`: Acting Coach panel showing role values, speech register, active failure mode
- `BioStateDisplay`: Condensed bio-system readout (NAc top links, hippocampus tier counts, active pain)

#### Pass criteria
- Blind A/B test: agent with Acting Coach calls >= 3 distinct affordances per gauntlet encounter vs <= 1 without
- No regression on non-embodied sim behavior

---

### Track 1b: F3-F5 — Agent Factory Canonicalization (~1500 LOC)

**Goal:** Make `AgentFactory.create_full_agent` the single production door for agent construction.

**Depends on:** Nothing (independent of B3 and E2-E3)

#### Key decisions (resolved)

| # | Decision | Resolution | Rationale |
|---|----------|------------|-----------|
| 1 | Orch pain_bus | YES — own pain_bus + nac | Clean separation. Wire bus now, defer learning subscribers until adaptive orch has a use case. Plumbing ready, nothing flowing yet. |
| 2 | Reachy pain_detector | MIGRATE to canonical build_bio_stack | Legacy path is tech debt from pre-PainBus era. One construction pattern. |
| 3 | Headless bio-learning | ON by default | Learning is Maxim's identity. First persist() logs INFO: "Bio-learning enabled — memories persist to ~/.maxim/sessions/. Disable with learning=False." Document prominently in `run()` docstring. |
| 4 | Auto-curation | Stepping stone in E3 (see Track 2) | Full orchestrator integration is 0.8 scope |

#### Stages

**F3 — Sim Orchestrator migration SHIPPED (2026-04-20)**
- AUT: `create_full_agent(with_bio_stack=True, with_executor=True, with_pain_bridge=entity_ref is not None, with_fear_gate=False)` — fear gate False because sim wraps pain layers before fear gate (ordering constraint)
- Orch: `create_full_agent(with_bio_stack=True, with_executor=True, with_pain_bridge=False)` — own isolated nac for future adaptive orchestration
- Both agents get clean factory construction, no hand-rolled wiring
- Orch persistence path resolves to `~/.maxim/orchestrator/` (not CWD-relative — agent longevity review found fragmentation risk)
- Factory fix: `nac` always passed to `build_executor` (decoupled from `with_pain_bridge`) so bridge exists for direct attribution even when `pain_bus=None`
- Added `fear_llm` parameter to `create_full_agent` for LLM-powered FearAgent (sim passes `llm_router`)

**F4 — Reachy migration SHIPPED (2026-04-20)**
- Executor switched from legacy `pain_detector` subscription to canonical `pain_bus` from `build_bio_stack`
- Full factory migration deferred to G2 (HostContext protocol)

**F5 — Headless API migration SHIPPED (2026-04-20)**
- `api.py` headless path gets full bio-stack by default via `create_full_agent`
- `maxim.run(..., learning=False)` explicit opt-out
- First-persist INFO log for discoverability
- `AgentInstance.shutdown()` called in finally block (saves hippocampus + NAc + cerebellum)
- `mypy` pass on public API files: clean

**F6 — CI enforcement + test audit (~400 LOC)**
- AST-based CI gate: no `Executor()` constructor outside `runtime/bootstrap.py` + `tests/`
- Test coverage for all migrated sites
- Stress test: concurrent `create_full_agent` calls (AgentPool path)

---

### Track 2: E2-E3 — Foundry with Real LLM + Auto-Curation

**Goal:** Run the foundry pipeline with the leader's LLM and automatically curate promoted components.

**Depends on:** B3 (respond-loop fix required for meaningful gauntlet scores)

#### Stages

**E2 — Real LLM integration (~300 LOC)**
- Wire `llm_router` from CLI → `FoundryRunner` (review confirmed: `llm_router` already threads through to `EntityDesigner` in foundry.py:215-217 — wiring is simpler than expected)
- Entity context injection into AUT prompt via existing `prompt_builder.py` (strategy layer: sensor names + affordance descriptions + failure trigger conditions, auto-composed from spec)
- Use loop over `EntityDesigner.design()` for generation (review found `design_batch()` does not exist — defer batch optimization to 0.8)
- Integration test: foundry run with mocked LLM router returning realistic JSON

**E2.5 — ComponentIndex: semantic discovery layer (~250 LOC)**

The bridge between natural language and exact-ref lookup. Two-layer architecture:

```
Query: "old iron door"
    │
    ├─ Layer 1: Alias table (O(1) hash lookup)
    │   aliases["old iron door"] → None (no exact alias)
    │   aliases["rusty gate"]    → "environments/rusty_gate" ✓
    │
    └─ Layer 2: Semantic embedding (cosine similarity)
        embed("old iron door") vs all component embeddings
        → "environments/rusty_gate" at 0.82 cosine ✓ (above threshold)
        → Return match, skip imagination
```

**New file:** `embodiment/component_index.py` — `ComponentIndex` class

```python
class ComponentIndex:
    """Semantic discovery layer over ComponentRegistry.

    Two-layer lookup: alias hash table (fast, exact) + embedding index
    (slower, fuzzy). Constructed from a ComponentRegistry at startup,
    updated incrementally when register_ephemeral() adds entities.
    """

    def __init__(
        self,
        registry: ComponentRegistry,
        *,
        encoder_model: str = "all-mpnet-base-v2",
        similarity_threshold: float = 0.65,
    ) -> None: ...

    def find(self, query: str) -> ComponentMatch | None:
        """Two-layer lookup: alias → embedding → None."""

    def find_similar(self, query: str, k: int = 5) -> list[ComponentMatch]:
        """Return top-k similar components by embedding distance."""

    def add(self, ref: str, spec: dict, synonyms: list[str] | None = None) -> None:
        """Index a new component (called on register + register_ephemeral)."""

    def dedup_check(self, spec: dict, threshold: float = 0.80) -> ComponentMatch | None:
        """Check if a candidate spec is a near-duplicate of an existing component."""
```

**Layer 1 — Alias table:**
- Hash map: `dict[str, str]` mapping lowercase alias → component ref
- Populated from `component.synonyms` field in YAML specs
- O(1) lookup, zero false positives
- New YAML field for hand-authored components:

```yaml
component:
  name: rusty_gate
  category: environments
  tags: [fantasy]
  synonyms: [gate, iron gate, old gate, rusty door, decrepit entrance]
```

- For foundry-generated components, `EntityDesigner.design()` generates synonyms as part of the spec (add `synonyms` to the generation prompt template)
- **Synonym backfill as PoC gating test:** Run the full EntityDesigner synonym generation pipeline against all 65 seed components. If the ComponentIndex can discover seed components through natural language queries using the system-generated synonyms, that validates the entire discovery layer end-to-end. This is the quality gate — if "old iron door" finds "rusty_gate" and "healing draught" finds "healing_potion", the index works.

**Layer 2 — Semantic embedding index:**
- Uses existing `LinguisticEncoder` (`similarity/encoder.py`) — no new embedding model
- Each component gets a **semantic signature** embedding computed from:
  ```
  "{name}: {description}. sensors: {sensor_names}. affordances: {affordance_names}. failures: {failure_names}"
  ```
- Stored as a dense vector alongside the component ref
- On query: embed the query text, compute cosine similarity against all stored embeddings
- Above `similarity_threshold` (default 0.65) → match found
- Falls back gracefully when `sentence-transformers` not installed (bag-of-words hash, same as `LinguisticEncoder` fallback)
- **Persistence:** Embeddings cached to `~/.maxim/component_index.npz` (rebuild on registry change, mtime-checked)

**Construction + wiring:**

| Call site | How it gets the index | When |
|-----------|----------------------|------|
| `build_bio_stack()` | Constructs `ComponentIndex(registry)` when `component_registry` is provided | Startup |
| `AgentFactory.create_full_agent()` | Receives index from bio_stack or constructs if registry is set | Agent creation |
| `FoundryRunner` | Receives index for dedup checks during curation | Foundry runs |
| `imagination/trigger.py` | Receives index for semantic lookup before triggering imagination | Per-percept |
| `simulation/orchestrator.py` | AUT and orch share the same index instance (thread-safe via RLock) | Sim startup |

**Thread safety:** `ComponentIndex` uses an RLock (same pattern as `ComponentRegistry`). `add()` acquires lock, appends to both alias table and embedding list. `find()` acquires lock for the iteration. The embedding computation itself happens outside the lock (it's the slow part — ~5ms per embed call).

**E3 — Auto-Curation CLI (~400 LOC)**
- `--auto-curate` flag: pre-sim foundry run when genre coverage is below threshold
  - Checks ComponentRegistry for genre/category coverage
  - If < N components for the target genre, runs foundry to fill gaps
  - Promotes high-scorers (>= 0.7) automatically to `~/.maxim/components/`
  - Available for the current session immediately
- `--curate-threshold N` (default: 5 per genre/category)
- `--no-curate` explicit opt-out
- API: `maxim.imagine(..., auto_curate=True)` / `maxim.run(..., auto_curate=True)`
- Post-sim quality report: what was generated, what was promoted, scores
- **Semantic dedup via ComponentIndex:** Before promotion, call `index.dedup_check(candidate_spec, threshold=0.80)`. If a near-duplicate exists, skip promotion and log: "Skipped {name} — similar to existing {match.ref} (cosine {score:.2f})". Replaces the ad-hoc EC similarity check from the previous plan revision with the canonical index.

---

### Track 3: Imagination Process — On-Demand Entity Generation

**Goal:** The agent can generate SEM entities for things it perceives but doesn't have models for. This is the cognitive "imagination" — turning perceived concepts into interactive physical models.

**Depends on:** E2.5 (ComponentIndex), E2 (real LLM foundry path working), B3 (agent knows how to explore affordances)

This is the novel feature that makes 0.7 transformative. The foundry becomes an internal cognitive process, not just a CLI tool.

#### Architecture

```
Percept: "You see a rusty gate blocking the path"
    │
    ▼
Entity extraction: noun-phrase extraction → candidates ["rusty gate"]
    │
    ▼
ComponentIndex.find("rusty gate")
    │
    ├─ Alias hit: "rusty gate" → "environments/rusty_gate" ✓ → USE EXISTING (skip imagination)
    ├─ Embedding match: cosine 0.82 with "environments/rusty_gate" ✓ → USE EXISTING
    └─ No match (below threshold) → IMAGINE
            │
            ▼
    DN arousal gate: is the agent in a low-arousal state?
        │              ↗ (high arousal → skip, don't daydream during combat)
        ▼
    Energy gate: can we afford an LLM call?
        │           ↗ (no budget → fall back to verbal interaction only)
        ▼
    EntityDesigner.design(description="rusty gate", context=current_scene)
        │
        ▼
    Quick validation (schema + sensor sanity, no gauntlet)
        │
        ▼
    ComponentRegistry.register_ephemeral(spec, provenance="imagined")
    ComponentIndex.add(ref, spec, synonyms=generated_synonyms)
        │
        ▼
    ToolRegistry.register() → new affordance tools (push_gate, pick_lock, force_open)
        │                       (scene-scoped: tools from prior scenes deactivate)
        ▼
    Agent can now physically interact with the gate
        │
        ▼
    Episodes + causal links tagged with imagined=True provenance
```

The **ComponentIndex** is the critical decision point. It prevents unnecessary imagination by finding existing entities through fuzzy matching (aliases + embeddings). Only when both layers return no match does imagination trigger. This means the library gets *smarter* as it grows — more entities = more alias coverage + denser embedding space = fewer redundant imagination calls = lower energy cost per session.

#### Key design principles

1. **Ephemeral by default.** Imagined entities live in session memory only. They don't auto-promote to the persistent library. Post-session scoring decides if they're worth keeping.
2. **Energy-gated.** Each imagination call costs LLM tokens. Gated by the existing energy budget (`src/maxim/energy/` — review confirmed this exists). Low energy = no imagination, fall back to verbal interaction.
3. **DN arousal-gated.** Imagination fires as a DefaultNetwork idle/low-arousal behavior, not as a standalone process. High-arousal reactive states (combat, pain response) inhibit imagination. Bio-metaphor: you don't daydream while fighting.
4. **Validation-only, no gauntlet.** Real-time imagination skips the 3-encounter gauntlet (too slow). Quick schema + sensor sanity check only.
5. **Scene-scoped tool window.** Active tools capped at ~15-20. When the agent moves to a new scene, tools from prior scene's imagined entities deactivate. Prevents prompt overflow in long campaigns. (Review found "additive only" invariant guarantees prompt bloat.)
6. **Provenance tagging.** Episodes and NAc causal links from imagined entity interactions carry `imagined=True`. On entity discard (not promoted), decay those links by 50% (don't delete — partial learning is still useful, but confidence should reflect simulated origin). Mirrors biological source monitoring.
7. **Session-scoped imagination cache.** Shared between orchestrator and AUT to prevent duplicate imagination of the same entity.

#### Stages (0.7 scope)

**I1 — Imagination trigger + infrastructure (~300 LOC)**
- Entity noun-phrase extraction from percepts (~150 LOC) — review found this pipeline does not exist. Uses existing concept extraction from `memory/concept_extractor.py` as a starting point, extended to identify entity-like nouns (physical objects, creatures, environmental features).
- **ComponentIndex integration:** Trigger uses `component_index.find(entity_phrase)` instead of raw `ComponentRegistry.has(ref)`. This is the key improvement — "old iron door" fuzzy-matches "rusty_gate" via embedding similarity and skips imagination. Only truly novel entities (no alias hit, no embedding match above 0.65) proceed to the design stage.
- `ComponentRegistry.register_ephemeral(spec, provenance="imagined")` — session-scoped registration with provenance tag. Separate overlay dict from persistent `_index` (architecture review: don't mix ephemeral and persistent entries). On registration, also calls `component_index.add(ref, spec, synonyms)` to make the new entity discoverable to future extraction passes within the same session.
- Session-scoped `ImaginationCache` — prevents duplicate imagination. Checked by both AUT and orchestrator. Keyed by normalized entity phrase. The cache stores both "found via index" (existing entity) and "imagined" (new entity) results so repeated mentions of the same entity don't re-trigger either lookup or design.
- `imagined=True` provenance tag on `Episode` and `CausalLink` — review cross-confirmed this is critical for learning integrity.
- DN arousal gate integration (~50 LOC) — wire into existing DN idle behavior at `default_network/network.py`. Imagination fires during `ReturnToCenter` and idle states only.
- Configurable: `imagination=True/False`, `imagination_threshold=2` (mentions before triggering)

**I2 — Real-time entity design (~200 LOC)**
- `imagination/designer.py`: Wraps EntityDesigner for real-time use
  - Takes scene context + entity description → SEM spec
  - Quick validation (schema + sensor sanity, no gauntlet)
  - Energy-gated (checks budget via `energy.get_global_registry()` before LLM call)
  - Falls back gracefully: if LLM unavailable or budget exhausted, entity stays verbal-only
  - **Synonym generation:** EntityDesigner prompt template extended to include `"synonyms": ["alias1", "alias2", ...]` in the generated JSON. These are passed to `ComponentIndex.add()` on registration. Prompt addition: `"Include a 'synonyms' list of 5-10 alternative names or descriptions a user might use to refer to this entity."` This populates Layer 1 (alias table) automatically for every imagined entity.
- Wire into agent loop: post-state-update hook → entity extraction → trigger check → design if needed
  - Review found no clean perception hook exists. Use post-`state.update()` processing of `state.data` rather than intercepting perception output (cleaner than coupling to `SimulationAdapter`).

**I3 — Scene-scoped tool window + hot-reload (~250 LOC)**
- `ToolRegistry.register_scene_tools(entity_ref, tools)` — registers tools with scene scope tag
- `ToolRegistry.deactivate_scene(scene_id)` — deactivates (not removes) tools from a prior scene
- Active tool cap: ~15-20 tools. When cap is reached, oldest scene's tools deactivate first.
- `build_tools_section` respects active/inactive status — only active tools appear in prompt
- **Invariant:** Tools are deactivated by scene transition, never deleted mid-session. Deactivated tools can be re-activated if the agent returns to a scene.

#### Deferred to 0.8

| Item | Why deferred |
|------|-------------|
| **I4 — Post-session scoring + promotion** | Make opt-in (`--score-imagined`), not automatic. Running gauntlets post-session adds latency. Ship as flag, promote to default after real usage data. |
| **I5 — Orchestrator imagination for sim** | Doubles integration surface. AUT-side imagination is the novel feature. Orchestrator-side is a convenience for 0.8. |
| **`EntityDesigner.design_batch()`** | Loop over `design()` is fine for 0.7 volumes. Batch optimization when real usage data shows it matters. |
| **Narrator entity awareness** | Narrator has zero entity awareness today (`generate()` takes only phase/direction/context). Important for narrative/physics consistency but separate concern. |
| **Library garbage collection** | `~/.maxim/components/` grows monotonically with auto-promotion. Premature to build GC before curation runs at scale. |
| **ComponentIndex-driven entity templates** | Open question 1: when `ComponentIndex.find()` returns a partial match (0.40-0.65 cosine), use that entity as a template for imagination instead of designing from scratch. Good bio-metaphor (recombination of existing knowledge). Deferred because the threshold tuning needs real-world data from 0.7 imagination runs. |

---

## Sequencing

```
Track R0: Prerequisites ──── FIRST (blocks everything)
  R0.1 ComponentRegistry.query() lock
  R0.2 Sim-mode lightweight consolidation
  R0.3 TOOL_ALIASES lock

Track 1a: B3 Acting Coach ──────────────────────────┐
  B3.1 Config + prompt section                       │
  B3.2 Orchestrator probe improvement                │
  B3.3 Display extensions                            │ (fixes respond-loop)
                                                     │
Track 1b: F3-F5 Factory ──── (parallel) ─��───────────│
  F3 Sim orchestrator migration                      │
  F4 Reachy migration                                │
  F5 Headless API migration                          │
  F6 CI enforcement + tests                          │
                                                     │
Track 2: E2-E3 Foundry ─────────────────────────── depends on B3
  E2 Real LLM integration                            │
  E2.5 ComponentIndex (semantic discovery layer)      │ ← new: enables dedup + imagination lookup
  E3 Auto-curation CLI (uses index for dedup)         │
                                                     │
Track 3: Imagination (0.7 scope) ────────────────── depends on E2.5 + B3
  I1 Imagination trigger (uses index for lookup)      │
  I2 Real-time entity design (generates synonyms)    │
  I3 Scene-scoped tool window + hot-reload           │
```

**Critical path:** R0 → B3 → E2 → E2.5 → E3 / I1-I3
**Parallel path:** F3-F5 (runs alongside everything)
**Note:** E3 and I1-I3 can run in parallel after E2.5 lands — both consume the ComponentIndex but don't depend on each other.

---

## LOC Estimates (revised post-review)

| Track | Estimated LOC | Sessions | Notes |
|-------|---------------|----------|-------|
| R0 Prerequisites | ~70 | <1 | Thread safety + consolidation fixes |
| B3 Acting Coach | ~400 | 1-2 | |
| F3-F5 Factory | ~1500 | 2-3 | |
| E2 Real LLM | ~300 | 1 | Simpler than expected (llm_router already threads) |
| E2.5 ComponentIndex | ~250 | 1 | Alias table + embedding index + synonym backfill |
| E3 Auto-Curation | ~350 | 1 | Dedup via index (was ad-hoc EC check, now canonical) |
| I1 Imagination trigger | ~300 | 1-2 | Uses ComponentIndex.find() instead of raw registry check |
| I2 Entity design | ~200 | 1 | +synonym generation in EntityDesigner prompt |
| I3 Tool window | ~250 | 1 | Was "additive only" 200 LOC, now scene-scoped (larger) |
| **Total** | **~3620** | **8-12** | |

---

## Invariants (filled in AFTER shipping each track)

### R0 (SHIPPED)
- `ComponentRegistry.query()`, `list_categories()`, `list_refs()` hold `_lock` during iteration (snapshot-before-iterate)
- Sim-mode runs `on_session_end_lightweight()` — NAc decay + persistence saves + concept_extractor flush, skips expensive `hippocampus.sleep()`
- `TOOL_ALIASES` guarded by `_TOOL_ALIASES_LOCK` (RLock). Reads unguarded (CPython GIL atomic).

### B3 (B3.1 SHIPPED)
- Acting Coach is a "default policy" that bio-systems MODULATE, never suppress. NAc valence, pain anticipation, and cerebellum predictions add caution annotations to exploration directives — the agent always *can* explore, the bio-systems inform *how cautiously*.
- Composition order: base coach directive → NAc caution → pain anticipation → cerebellum predictions. Each layer adds information; none removes the base directive.
- Section priority is IMPORTANT (not CRITICAL) to avoid token budget contention with body_state.
- Wiring goes through `LLMWorker.acting_coach`, NOT `AgentConfig` (factory doesn't consume it).
- (B3.2 and B3.3 remaining)

### F3-F5
- `AgentFactory.create_full_agent` is the only production door (CI-enforced AST gate)
- Orchestrator gets its own isolated pain_bus + nac, persists to `~/.maxim/orchestrator/` (not CWD-relative)
- Headless bio-learning ON by default, first-persist INFO log, `learning=False` opt-out documented in `run()` docstring

### E2-E2.5-E3
- Foundry never auto-commits to bundled `_data/components/`. Promotion writes to `~/.maxim/components/` only.
- Auto-curation respects energy budget. No silent overspend.
- **ComponentIndex is the canonical semantic lookup for all entity discovery.** Two layers: alias table (O(1) hash, populated from `component.synonyms` field) + embedding index (cosine similarity via `LinguisticEncoder`, threshold 0.65). Falls back to bag-of-words hash when `sentence-transformers` not installed.
- **Dedup before promotion uses `ComponentIndex.dedup_check(spec, threshold=0.80)`.** Near-duplicates are logged and skipped, not silently swallowed.
- **Every new entity (foundry-generated or imagined) must include a `synonyms` list.** EntityDesigner prompt template generates 5-10 aliases. Existing 65 seed components get synonyms via the full system pipeline (PoC gating test — validates discovery end-to-end).
- **Embedding cache persists to `~/.maxim/component_index.npz`.** Rebuilt on registry content change (mtime-checked). Avoids re-embedding 200+ components on every startup.
- ComponentIndex is constructed once per session and shared (thread-safe via RLock). Imagination and curation both use the same instance — no divergent state.

### I1-I3
- **Imagination trigger uses `ComponentIndex.find()`, not raw `ComponentRegistry.has()`.** Alias hits and embedding matches (cosine >= 0.65) prevent unnecessary imagination. This is the primary throttle that makes imagination cheaper as the library grows.
- Imagined entities are ephemeral (session-scoped overlay, separate from persistent `_index`). On registration, `ComponentIndex.add()` is called so the entity is discoverable within the same session.
- Scene-scoped tool window: ~15-20 active tools max, deactivation by scene transition (not deletion)
- Imagination is energy-gated AND DN arousal-gated
- Quick validation only for real-time; gauntlet is 0.8 opt-in
- Episodes + causal links carry `imagined=True` provenance; decayed 50% on entity discard
- Session-scoped ImaginationCache prevents duplicate imagination between AUT and orchestrator. Keyed by normalized entity phrase. Stores both "found via index" and "imagined" results.
- Imagination disabled by default in headless API (`imagination=False`), enabled in sim and Reachy

---

## Open questions (updated post-review)

1. ~~Should imagined entities inherit from closest matching template?~~ **Deferred to 0.8.** When `ComponentIndex.find()` returns a partial match (0.40-0.65 cosine), use that entity as a template instead of designing from scratch. Good bio-metaphor but threshold tuning needs real-world data from 0.7 imagination runs.
2. ~~Should orchestrator pre-imagine or on-demand?~~ **Deferred to 0.8** (I5 deferred). AUT-side on-demand imagination only for 0.7.
3. ~~Narrator entity awareness?~~ **Deferred to 0.8.** Narrator has zero entity awareness today. Important for narrative/physics consistency but separate concern.
4. What's the right imagination threshold for regular (non-sim) environments? Reachy perceiving a "cup" via vision — should it imagine a cup SEM entity? That's potentially very noisy. **Tentative answer:** threshold=3 for non-sim (higher than sim's threshold=2), plus a category filter (only imagine entities in categories that have SEM relevance: weapons, creatures, vehicles, items — not furniture, clothing, abstract concepts).
5. ~~Should the Acting Coach be suppressed or modulated by NAc?~~ **RESOLVED: Modulation via bio-systems.** NAc valence, pain anticipation, and cerebellum predictions continuously annotate the coach's exploration directives with learned caution — never suppress them. Maps to amygdala/NAc modulating prefrontal planning. See B3.1 for full design.
6. **(New, from review)** How should orphaned causal links (from discarded imagined entities) interact with spreading activation? They reference tool names that no longer exist. Should they be excluded from activation propagation, or included with the 50% decay as passive knowledge?
7. **(New)** What's the right similarity threshold for ComponentIndex Layer 2? 0.65 is a starting point based on EC's existing tuning (P1 sweep: `paraphrase-mpnet@0.40` for concept collapse, but entity matching needs higher precision to avoid false matches). May need a dedicated sweep with entity-name pairs once E2.5 is built.
8. ~~Should synonym backfill be LLM-generated or hand-authored?~~ **RESOLVED: Use the full system.** Run the foundry's EntityDesigner synonym generation against all 65 seed components as the E2.5 PoC and gating test. If the ComponentIndex can discover existing seed components through natural language queries using system-generated synonyms, that validates the entire pipeline end-to-end. No rush to publish — this becomes the quality gate for the whole discovery layer.

---

## Review findings archive

The 5-lens parallel review (2026-04-19) produced findings in these categories. Cross-confirmed findings (found independently by 2+ reviewers) are marked with ✓✓.

### Cross-confirmed (highest confidence)
- ✓✓ Tool prompt bloat with no budget/eviction (sim longevity + architecture) → I3 scene-scoped tool window
- ✓✓ Ephemeral entity vs persistent episode contradiction (agent longevity + bio-system) → I1 provenance tagging
- ✓✓ No perception extraction pipeline (execution + bio-system) → I1 entity noun-phrase extraction
- ✓✓ Imagination should wire through DN (bio-system, strong) → I1 DN arousal gate

### Single-reviewer (validated, lower confidence)
- ComponentRegistry.query() thread safety gap (architecture) → R0.1
- TOOL_ALIASES lock missing (architecture) → R0.3
- Sim-mode skips consolidation (agent longevity) → R0.2
- Orch persistence path CWD-relative (agent longevity) → F3
- Headless bio-learning needs prominent docstring (architecture) → F5
- Acting Coach vs NAc arbitration missing (agent longevity) → B3.1
- Narrator has zero entity awareness (sim longevity) → deferred to 0.8
- Library GC needed long-term (sim longevity) → deferred to 0.8
- Shared imagination namespace needed (sim longevity) → I1 ImaginationCache
