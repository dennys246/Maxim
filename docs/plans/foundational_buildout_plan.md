# Foundational Buildout Plan

> **Status:** In progress. Phases 0-12a done/shipped. Phase 11 in progress (Test PyPI). Phase 12b partially done.
> **Goal:** Ship the architectural foundations that Multi-AUT Party Mode, SEM Component Database, and DM Encounter Library require — plus fix packaging, API surface, and code quality issues — before locking the public API via PyPI publication.
> **Total scope:** ~6,650 LOC across 13 phases.
> **Sequence:** Hygiene (0 ✓) → Foundation (1 → 1.1 → 2-3 → 4 → 5) → DM polish (6-7) → API + Packaging (8-9) → Publication prep (10) → Test PyPI (11) ∥ Hardening (12b) → Manual publish.

---

## Why This Exists

The current runtime is single-agent: one agent loop, one tool registry, one environment, one set of memory systems. The mesh infrastructure (LocalMessageBus, ExperienceBroker, TaskDelegator, PeerClockEstimator) is 75% built but 0% wired into the agent loop. SEM entities are scattered YAML files with no registry, versioning, or composition. The public API exposes 6 verbs but hides most of what makes Maxim interesting (campaigns, benchmarks, events, tools, memory access). And `pip install pymaxim` currently won't work because data files use CWD-relative paths.

Publishing to PyPI would lock us into an architecture that makes multi-agent, shared components, and encounter libraries painful to retrofit. This plan does the structural work first, expands the API to match the system's real capabilities, then publishes.

**What this plan is NOT:**
- Not finishing all DM extensions (adaptive difficulty, encounter isolation, merging — those ship post-publication based on usage data)
- Not building full mesh network transport (mDNS, InferenceRouter — no multi-machine need yet)
- Not building the Capability Agent (nice-to-have, not load-bearing)
- Not building full CI/CD (manual `twine upload` is fine for v0.2.0)

---

## Phase 0: Package Hygiene (~550 LOC) — DONE

**Why first:** None of the later phases matter if `pip install pymaxim` doesn't work or `import maxim` has side effects. This is the boring work that makes everything else possible.

### 0a. Data file strategy (~250 LOC)

**Problem:** 50+ files reference `data/util/llm.json`, `scenarios/campaigns/*.yaml`, `src/configs/templates/*.json` via CWD-relative paths. After `pip install`, these directories don't exist.

**Solution — split into two categories:**

1. **Bundled defaults** (LLM profiles, config templates, component seeds) — move into `src/maxim/_data/` so they ship in the wheel:
   ```
   src/maxim/_data/
   ├── templates/         # From src/configs/templates/
   │   ├── llm.json
   │   ├── default_actions.json
   │   └── ...
   ├── components/        # Seed SEM components (Phase 1 populates this)
   └── encounters/        # Seed encounter library (Phase 2 populates this)
   ```

2. **User-generated data** (memories, sim reports, learned state, plans) — write to `~/.maxim/` or `$MAXIM_DATA_HOME`:
   ```
   ~/.maxim/
   ├── memory/            # Hippocampus, ATL, NAc persistence
   ├── sim_reports/       # Simulation output
   ├── models/            # Downloaded LLM/TTS/YOLO models
   ├── config/            # User config overrides
   └── agents/            # Per-agent data (Phase 3)
   ```

**New files:**
- `src/maxim/utils/paths.py` (~80) — `data_home()`, `bundled_data()`, `user_config()`, `agent_data(agent_id)` path resolution with env var overrides
- `src/maxim/_data/` directory with migrated templates

**Modified:** Every file that currently does `Path("data/util/something.json")` — update to use `paths.data_home() / "something.json"` for user data or `paths.bundled_data() / "templates/something.json"` for defaults. Estimated ~30 files need path updates.

**Add `importlib.resources` loader:**
```python
# src/maxim/utils/paths.py
def bundled_data() -> Path:
    """Return path to package-bundled data (read-only defaults)."""
    import importlib.resources
    return importlib.resources.files("maxim") / "_data"

def data_home() -> Path:
    """Return user data directory, created on first use."""
    base = Path(os.environ.get("MAXIM_DATA_HOME", Path.home() / ".maxim"))
    base.mkdir(parents=True, exist_ok=True)
    return base
```

### 0b. Import hygiene (~50 LOC)

**Problem:** GPU detection in `cli.py:12-88` runs `nvidia-smi` via subprocess at import time, modifies `os.environ`, and prints to stderr. This means `import maxim` has side effects.

**Fix:** Move Blackwell GPU detection into `cli.main()` or a lazy `_detect_gpu()` called on first LLM use. The detection result can be cached in a module-level `_gpu_info: dict | None = None`.

### 0c. Create missing package files (~5 LOC)

- `src/maxim/py.typed` — empty file, PEP 561 compliance
- `src/maxim/__main__.py` — `from maxim.cli import main; main()` (enables `python -m maxim`)

### 0d. Fix print() in library code (~50 LOC)

Replace `print()` with `logger.info()` / `logger.warning()` in these non-CLI files:
- `bridges/pain_bridge.py`
- `bridges/planning_bridge.py`
- `simulation/report.py` (15+ instances)
- `evaluation/llm_benchmark.py`
- `data/camera/display.py`

### 0e. Fix unclosed file handles (~30 LOC)

Add context managers or try/finally to:
- `provenance/store.py:51` — `self._current_file = open(...)`
- `simulation/sim_logger.py:88` — `_log_file = open(...)`
- `utils/data_management.py` — 3 instances
- `inference/transcribe_audio.py`

### 0f. Fix persistence violations (~60 LOC)

16 files use hand-rolled `json.dump()` + `os.replace()` instead of `atomic_write_json()`. Most critical:
- `simulation/report.py:193`
- `models/language/cost_tracker.py:240`
- `memory/consolidation.py:293`
- `planning/plan_document.py:738`
- `time/scn.py:645`

Replace with `atomic_write_json()` from `utils/atomic_io`.

### 0g. Cleanup (~10 LOC)

- Remove duplicate `llm-local` / `llm-llama` in pyproject.toml (identical deps)
- Update `pyproject.toml` to include `src/maxim/_data/` in package data:
  ```toml
  [tool.setuptools.package-data]
  maxim = ["_data/**/*", "py.typed"]
  ```

### 0h. De-globalize mutable state for multi-agent readiness (~200 LOC)

These global singletons will break when Phase 3 (Agent Factory) runs concurrent agents. Fix them now so Phase 3 doesn't have to:

| Global | File | Fix |
|--------|------|-----|
| `_sim_active`, `_log_file`, `_log_records` + 14 functions | `simulation/sim_logger.py` | Refactor into `SimLogger` class, instantiate per-simulation. **29 files import sim_log()** — update call sites to accept logger instance or use thread-local default. ~150 LOC refactor. |
| `_NEXT_CLASS_ID`, `_class_registry` | `simulation/narrative_transcriber.py` | Add `threading.Lock`, or make per-transcriber |
| `_active_routers` | `runtime/lane_backends.py` | Add `threading.Lock` around mutations |
| `_accessible_folders` | `utils/filesystem_policy.py` | Accept explicit folder list in constructor, global as fallback |

Don't fix *all* globals (EnergyRegistry, MetricsRegistry can stay shared for now) — just the ones that will corrupt data or deadlock under concurrent agent threads.

**Ship gate:** `pip install -e .` in a fresh venv, `python -m maxim --help` works, `import maxim; maxim.diagnose()` works, no subprocess at import time, no print() output from library calls.

---

## Phase 1: SEM Component Registry (~350 LOC) — IN PROGRESS

**Why first:** Defines the storage/sharing pattern that Encounter Library, Generative Architect, and Multi-AUT all depend on. Every NPC in party mode needs to load from a shared component definition.

### Critical Design Decision: Templates, Not Instances

**The registry stores raw YAML spec dicts — NOT Entity objects.** This is critical because:
- Two encounters referencing `"npcs/guard"` must get **independent** Entity instances (separate HP, separate trust sensors). Returning the same Entity would cause shared-state bugs.
- The current DM system already works this way: `CampaignDef.npc_specs` stores raw dicts, and `SceneState` instantiates Entity objects at runtime via `_parse_entity()`.
- Entity objects have no `clone()` or `to_dict()` method, and deep-copying entity trees with parent-child references is fragile.

The registry is a **template catalog**: it stores specs, resolves inheritance, and returns resolved dicts. Callers instantiate via `_parse_entity()` as they already do.

### Design

**Directory layout:**
```
src/maxim/_data/components/     # Bundled seed components (ship in wheel)
├── weapons/
│   ├── rusty_sword.yaml
│   ├── longbow.yaml
│   └── staff_of_healing.yaml
├── npcs/
│   ├── base_humanoid.yaml      # Base template (shared sensors/modulators)
│   ├── guard.yaml              # extends: npcs/base_humanoid
│   ├── merchant.yaml
│   └── ferryman.yaml
├── creatures/
│   ├── wolf.yaml
│   └── cave_spider.yaml
├── environments/
│   ├── tavern_interior.yaml
│   └── forest_clearing.yaml
└── bodies/
    ├── robot_arm_3dof.yaml     # Moved from scenarios/embodiment/
    └── reachy_mini.yaml

~/.maxim/components/            # User-defined components (not shipped)
```

**Component YAML format:**
```yaml
# src/maxim/_data/components/weapons/rusty_sword.yaml
component:
  name: rusty_sword
  tags: [weapon, melee, degradable]
  category: weapons
  extends: null  # or "weapons/base_sword" for inheritance

entity:
  # Standard SEM entity spec (unchanged from current format)
  name: rusty_sword
  entity_type: weapon
  sensors:
    durability: { unit: points, range: [0, 10], initial: 7 }
    sharpness: { unit: ratio, range: [0.0, 1.0], initial: 0.6 }
  modulators:
    combat:
      affordances:
        slash: { params: { target: str }, description: "Slash with sword" }
        parry: { params: { direction: str }, description: "Parry incoming attack" }
  failure_modes:
    - name: shatter
      trigger: { sensor: durability, op: "<=", value: 0 }
      pain_intensity: 0.7
```

**No version pinning in v1.** Ref format is just `"category/name"` (e.g., `"npcs/guard"`). If versioning becomes needed later, use colon syntax (`"npcs/guard:1.0"`, like Docker tags) and add a `version` field to the `component:` header. For now, latest-wins by search path priority.

**Inheritance via `extends` (deep merge):**
- Child components inherit all sensors, modulators, failure_modes from parent
- **Deep merge, not shallow** — child overrides at the leaf level:
  ```yaml
  # base_humanoid.yaml: sensors: { hp: { unit: points, range: [0, 20], initial: 20 } }
  # guard.yaml extends base_humanoid: sensors: { hp: { initial: 15 } }
  # Resolved: sensors: { hp: { unit: points, range: [0, 20], initial: 15 } }
  ```
- Single-level inheritance only (no diamond chains — keep it simple)
- Inheritance resolution happens at `get()` time, result is cached

**Utility: `deep_merge(parent, child) -> dict`**
```python
def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base. Lists are replaced, not appended."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result
```

**Search path priority (4 levels):**
1. **Campaign-local** — same directory as the campaign YAML file (for campaign-specific entities)
2. **User components** — `~/.maxim/components/` (user's custom library)
3. **Bundled components** — `src/maxim/_data/components/` (shipped in wheel)
4. **Legacy path** — `scenarios/embodiment/` (backward compat during transition, searched relative to repo root)

Higher-priority paths shadow lower ones. If `~/.maxim/components/npcs/guard.yaml` exists, it takes precedence over the bundled version.

**Registry class:**
```python
# src/maxim/embodiment/component_registry.py (~250 LOC)

@dataclass(frozen=True)
class ComponentInfo:
    """Lightweight metadata returned by query() — no full spec parsing needed."""
    ref: str              # "weapons/rusty_sword"
    name: str             # "rusty_sword"
    category: str         # "weapons"
    tags: tuple[str, ...]
    extends: str | None   # Parent ref or None
    source_path: str      # Absolute path to the YAML file

class ComponentRegistry:
    """Discovers, indexes, and resolves SEM component templates.

    Stores raw YAML spec dicts — NOT Entity objects. Callers instantiate
    entities via _parse_entity() to get independent instances with separate
    sensor state.
    """

    def __init__(self, search_paths: list[Path] | None = None,
                 campaign_dir: Path | None = None):
        """Build component index by scanning search paths.

        Two-phase loading:
        1. Index scan (at init) — reads only the ``component:`` header
           from each YAML for metadata (name, tags, category, extends).
        2. Full parse (on get()) — loads the complete spec on first access,
           resolves extends, and caches the result.
        """

    def get(self, ref: str) -> dict:
        """Return the resolved entity spec dict for a component ref.

        If the component has ``extends``, deep-merges parent spec first.
        Result is cached after first resolution.

        Returns a **copy** of the cached dict so callers can mutate freely
        (e.g., override ``name`` for a specific NPC instance).

        Raises KeyError if ref not found in any search path.
        """

    def instantiate(self, ref: str, **overrides) -> "Entity":
        """Convenience: get() + _parse_entity() in one call.

        Creates a fresh, independent Entity instance from the template.
        Optional overrides are deep-merged into the spec before parsing
        (e.g., instantiate("npcs/guard", name="captain", sensors={"hp": {"initial": 25}})).
        """

    def query(self, *, tags: list[str] | None = None, category: str | None = None,
              has_sensor: str | None = None, has_affordance: str | None = None) -> list[ComponentInfo]:
        """Search components by metadata. All filters are AND-combined."""

    def list_categories(self) -> list[str]:
        """Return sorted list of available categories."""

    def list_refs(self, category: str | None = None) -> list[str]:
        """Return all known refs, optionally filtered by category."""

    def has(self, ref: str) -> bool:
        """Check if a ref exists without loading the full spec."""
```

**Thread safety:** `threading.Lock` around `_spec_cache` writes. Index reads are lock-free (built at init, only appended via `register()`).

**YAML ref resolution in campaigns (string = ref, dict = inline):**

```yaml
# In campaign YAML:
npcs:
  guard_captain:
    ref: "npcs/guard"                    # Resolved through registry
    overrides:                           # Optional per-instance overrides
      name: captain_aldric
      sensors: { hp: { initial: 25 } }
      metadata: { persona_prompt: "Loyal to the crown, suspicious of strangers" }

  # OR inline (no registry lookup):
  mysterious_stranger:
    name: stranger
    entity_type: npc
    metadata: { persona_prompt: "Speaks in riddles" }
```

**Integration with existing `load_spec()` (spec.py):**
- `spec.py` gains an optional `registry` parameter
- When present, any entity dict with a `ref` key is resolved through the registry before parsing
- Standalone YAML without `component:` header still loads as before — fully backward compatible
- Campaign YAML can mix inline entities and registry refs

**Integration with `dm_schema.py`:**
- `load_campaign()` accepts optional `registry` parameter
- NPC/object specs with `ref` key are resolved through registry
- `overrides` are deep-merged into the resolved spec before storing in `CampaignDef`
- Existing campaigns (inline specs only) work unchanged

**New files:**
- `src/maxim/embodiment/component_registry.py` (~250) — registry + deep_merge + ComponentInfo
- `src/maxim/_data/components/` — migrate existing specs + 8-10 seed components (~100 YAML)
- Tests (~100)

**Modified:**
- `src/maxim/embodiment/spec.py` — add optional `registry` param to `load_spec()`, resolve `ref` entries
- `src/maxim/simulation/dm_schema.py` — support `ref` + `overrides` in NPC/object definitions

**Ship gate:**
1. `ComponentRegistry` discovers and indexes all components across search paths
2. `get("npcs/guard")` returns a resolved spec dict (with inheritance applied)
3. Two calls to `instantiate("npcs/guard")` produce independent Entity instances (separate sensor state)
4. Campaign YAML with `ref: "npcs/guard"` loads and runs correctly
5. Existing campaigns (no refs) still work unchanged
6. `query(tags=["weapon"])` returns all weapon components
7. `extends` inheritance produces correct deep-merged specs
8. Circular inheritance detection raises clear error

### Critical Implementation Notes (from code review)

**Finding 1: DM runtime doesn't actually instantiate entities yet.**
`CampaignDef` stores `npc_specs` / `object_specs` as raw dicts (dm_schema.py:138-140). The `DMRuntime.init_entities()` method exists but is **never called** anywhere. NPC specs are only used to extract `metadata.persona_prompt` for dialogue hints (dm_runtime.py:259-272). `SceneState._entity_registry` expects pre-instantiated Entity objects, but nothing populates it from campaign specs.

**Impact:** The registry's `instantiate()` fills this gap. When integrating, we need to wire `DMRuntime` to call `registry.instantiate()` for each active NPC/object in an encounter. This is a larger integration than originally planned — add ~50 LOC to dm_runtime.py.

**Finding 2: Entity state must survive between encounters.**
`SceneState.enter_encounter()` deregisters/re-registers **tools** but the Entity object itself persists in `_entity_registry` (dm_runtime.py:534). This means an NPC's HP/trust state carries across encounters — which is the correct behavior (a guard who lost HP in encounter 1 is still damaged in encounter 3).

**Impact:** Don't re-instantiate entities each encounter. Instantiate once at campaign start, cache in `SceneState._entity_registry`, and reuse. The registry provides **templates** for initial instantiation; the runtime owns the live instances.

**Finding 3: Two different entity loading paths exist.**
- DM runtime: entities from `CampaignDef.npc_specs` (raw dicts, currently unused)
- Generative runner: entities from `arc.metadata.world_entities` (calls `_parse_entity()` directly, generative_runner.py:78-84)

**Impact:** Both paths need registry support. The generative runner already calls `_parse_entity()` — make it call `registry.instantiate()` when a registry is available and the entity dict has a `ref` key.

**Finding 4: Circular inheritance detection needed.**
If component A extends B which extends A, `resolve_extends()` would infinite-loop. Track a `visited` set during resolution and raise `ValueError` on cycle detection.

**Finding 5: Legacy YAML auto-detection.**
Existing files in `scenarios/embodiment/` use `body:` and `world_entities:` keys (no `component:` header). The registry should auto-detect these as legacy format: if a YAML file has `body:` or `world_entities:` but no `component:` header, treat the root entity as a component with `name` derived from the filename and `category` derived from the parent directory.

**Finding 6: `_parse_entity` is safe to import from registry.**
`spec.py._parse_entity()` imports only from `maxim.embodiment.sem` — no circular dependency risk. The registry can safely import and call it.

---

## Phase 2: DM Encounter Library (~240 LOC)

**Why second:** Provides reusable narrative building blocks for campaigns. The Generative Architect (Phase 7) uses the library as raw material for LLM-driven campaign composition.

### Design Philosophy: Tier 1 (Scene + Choice Templates)

**Critical insight from code review:** Encounters are heavily campaign-coupled. `active_npcs` must match campaign NPC specs, `on_choice` flags are campaign-global state, `dialogue_hints` are keyed by campaign flags, and `branches` reference other encounters by name. Unlike SEM components (which are self-contained), encounters are narrative glue between campaign-specific elements.

**Solution — two tiers:**

- **Tier 1 (build now):** Library stores **partial encounter templates** — the campaign-independent parts (`scene`, `choices`, `dice`). Campaign YAML references templates and adds campaign-specific wiring (`active_npcs`, `branches`, `on_choice`, `dialogue_hints`) inline. No parameterization, no flag contracts.

- **Tier 2 (Phase 7, via LLM):** The Generative Architect adapts library encounters to campaign context using an LLM call. The LLM sees the library encounter + campaign NPCs/flags/tone and produces a fully wired, campaign-specific encounter. This replaces static `$TOKEN` parameterization — it's more flexible, produces better prose, and costs ~100 tokens per encounter (one-time at campaign creation). Medium-tier LLM preferred for quality; small-tier as fallback.

**Directory layout:**
```
src/maxim/_data/encounters/     # Bundled seed encounters (ship in wheel)
├── combat/
│   ├── forest_ambush.yaml
│   ├── tavern_brawl.yaml
│   └── arena_duel.yaml
├── social/
│   ├── merchant_negotiation.yaml
│   ├── guard_interrogation.yaml
│   └── trust_dilemma.yaml
├── exploration/
│   ├── trapped_corridor.yaml
│   └── hidden_passage.yaml
└── puzzle/
    ├── riddle_gate.yaml
    └── alchemy_challenge.yaml

~/.maxim/encounters/            # User-defined encounters (not shipped)
```

**Encounter template format (Tier 1 — campaign-independent parts only):**
```yaml
# src/maxim/_data/encounters/combat/forest_ambush.yaml
encounter:
  name: forest_ambush
  tags: [combat, outdoor, surprise, ambush]
  difficulty_range: [2, 5]
  narrative_role: rising_action
  suggested_npcs: 2               # Hint for architect: how many NPCs this needs
  suggested_npc_roles: [leader, scout]  # Hint for architect: what roles to fill

  # The reusable parts — narrative prose + decision structure:
  scene: >
    Branches crack overhead. Three figures step from the treeline,
    weapons drawn. The leader — scarred, grinning — raises a hand
    to halt the others. "Your coin or your life, traveler."
  choices: [fight, negotiate, flee]
  dice:
    fight: { roll: "1d20", dc: 12 }
```

**Campaign YAML usage — template + campaign-specific wiring:**
```yaml
encounters:
  forest_ambush:
    template: "combat/forest_ambush"   # Load scene + choices + dice from library
    active_npcs: [korrath]             # Campaign-specific NPC names
    world_objects: [fallen_log]
    branches:                          # Campaign-specific encounter flow
      fight: cave_entrance
      negotiate: cave_entrance
      flee: __END__
    on_choice:                         # Campaign-specific flags
      negotiate: { flags: [bribed_bandits] }
    dialogue_hints:
      bribed_bandits: "We had a deal, traveler. Pass."
```

**Merge rules:** Template fields provide defaults. Campaign YAML overrides at field level (not deep merge — a campaign `scene:` replaces the template scene entirely). This keeps the mental model simple: template = starting point, campaign = overrides.

**EncounterLibrary class:**
```python
# src/maxim/simulation/encounter_library.py (~120 LOC)

@dataclass(frozen=True)
class EncounterInfo:
    """Lightweight metadata for query() — no full YAML parsing needed."""
    ref: str              # "combat/forest_ambush"
    name: str
    category: str
    tags: tuple[str, ...]
    difficulty_range: tuple[int, int] | None
    narrative_role: str | None
    suggested_npcs: int
    source_path: str

class EncounterLibrary:
    """Discovers and indexes reusable encounter templates.

    Same search path pattern as ComponentRegistry:
    campaign-local → user (~/.maxim/encounters/) → bundled → legacy.
    """

    def __init__(self, search_paths: list[Path] | None = None,
                 campaign_dir: Path | None = None,
                 include_defaults: bool = True): ...

    def get(self, ref: str) -> dict:
        """Return the encounter template dict. Deep copy for safe mutation."""

    def query(self, *, tags: list[str] | None = None,
              category: str | None = None,
              difficulty: int | None = None,
              narrative_role: str | None = None) -> list[EncounterInfo]:
        """Search encounters by metadata. All filters AND-combined.
        If difficulty is provided, matches encounters whose difficulty_range
        contains the value."""

    def list_categories(self) -> list[str]: ...
    def list_refs(self, category: str | None = None) -> list[str]: ...
    def has(self, ref: str) -> bool: ...
```

**Integration with dm_schema.py:**
When `load_campaign()` encounters a `template:` key in an encounter dict, it:
1. Loads the template from `EncounterLibrary.get(ref)`
2. Extracts the `encounter:` section (scene, choices, dice)
3. Merges campaign-specific fields on top (active_npcs, branches, on_choice, dialogue_hints)
4. Passes the merged dict to the existing `EncounterDef` parser

Inline encounters (no `template:` key) work exactly as before — zero breaking changes.

**New files:**
- `src/maxim/simulation/encounter_library.py` (~120) — library class + EncounterInfo
- `src/maxim/_data/encounters/` — 8 seed encounter templates across categories
- Tests (~80)

**Modified:**
- `src/maxim/simulation/dm_schema.py` — resolve `template:` key in encounter dicts during `load_campaign()`

**Ship gate:**
1. `EncounterLibrary` discovers and indexes all encounter templates
2. Campaign YAML with `template: "combat/forest_ambush"` loads correctly
3. Campaign-specific fields (branches, NPCs, flags) override template defaults
4. Existing campaigns (no templates) work unchanged
5. `query(tags=["combat"], difficulty=3)` returns matching encounters
6. Architect persona (Phase 7) can browse library via `browse_encounters` tool

---

## Phase 3: Agent Factory + Agent Pool (~500-700 LOC) — IN PROGRESS

**Why third:** The architectural change that's hardest to retrofit. Enables multi-agent sims, party mode, and NPC agents with real memory.

### Design

**Agent Factory:**
```python
# src/maxim/runtime/agent_factory.py (~200 LOC)

@dataclass
class AgentConfig:
    """Configuration for spawning an independent agent instance."""
    agent_id: str
    role: str                    # "pc", "npc", "companion"
    entity_spec: str | None      # Component ref for SEM body
    persistence_dir: str | None  # Per-agent data directory (auto-generated if None)
    model_profile: str | None    # LLM profile override (NPCs can use cheaper models)
    tool_whitelist: set[str] | None  # Restrict available tools (NPCs don't get filesystem)
    personality: str | None      # System prompt overlay for NPC personality

class AgentFactory:
    """Creates independent MaximAgent instances with isolated subsystems."""

    def __init__(self, base_data_dir: Path, component_registry: ComponentRegistry | None = None):
        ...

    def create_agent(self, config: AgentConfig) -> AgentInstance:
        """Spawn a fully independent agent with its own memory systems.

        Each agent gets:
        - Hippocampus (separate persistence)
        - NAc (separate causal model)
        - ATL (shared concept definitions, separate instances)
        - MemoryHub (independent coordinator)
        - Tool registry (scoped to role)
        - Executor (isolated)
        """

    def create_npc_agent(self, npc_name: str, entity_ref: str, personality: str,
                         model_profile: str = "small") -> AgentInstance:
        """Convenience: create a lightweight NPC agent.

        NPCs get:
        - Restricted tool set (speak, choose, sense own entity, memory_recall)
        - Cheaper LLM tier (small by default)
        - Personality injected into system prompt
        - Full bio-stack (hippocampus, NAc) — they learn and remember
        """
```

**Agent Instance:**
```python
@dataclass
class AgentInstance:
    """A fully independent agent with its own subsystems."""
    agent_id: str
    role: str
    agent: MaximAgent           # The actual agent
    memory_hub: MemoryHub       # Independent memory coordinator
    hippocampus: Hippocampus    # Episodic memory
    nac: NAc                    # Causal learning
    executor: Executor          # Scoped tool execution
    tool_registry: dict         # Agent-specific tools
    entity: Entity | None       # SEM body (if embodied)
    config: AgentConfig
```

**Agent Pool:**
```python
# src/maxim/runtime/agent_pool.py (~200 LOC)

class AgentPool:
    """Manages concurrent execution of multiple agents."""

    def __init__(self, bus: LocalMessageBus | None = None):
        self._agents: dict[str, AgentInstance] = {}
        self._bus = bus or LocalMessageBus()
        self._threads: dict[str, Thread] = {}

    def add(self, instance: AgentInstance) -> None:
        """Register an agent. Connects to message bus."""

    def remove(self, agent_id: str) -> None:
        """Stop and remove an agent."""

    def run_turn(self, agent_id: str, percept: str) -> TurnResult:
        """Run a single turn for one agent (synchronous)."""

    def run_round(self, agent_ids: list[str], percepts: dict[str, str]) -> dict[str, TurnResult]:
        """Run one turn for each agent (concurrent threads)."""

    def broadcast(self, message: MeshMessage) -> None:
        """Send a message to all agents via bus."""

    def get_agent(self, agent_id: str) -> AgentInstance:
        """Look up agent by ID."""

    def export_memories(self, agent_id: str) -> dict:
        """Export agent's memory state (for post-sim analysis)."""

    def shutdown(self) -> None:
        """Stop all agents, flush memories, cleanup."""
```

**Key design decisions:**
- Each agent gets its own thread for turn execution (not async — matches existing agent loop model)
- `LocalMessageBus` (already exists) handles inter-agent communication
- NPC agents use `small` LLM tier by default (cheap + fast)
- Tool registries are scoped: NPCs can't write files, PCs can't read NPC internals
- Memory persistence goes to `~/.maxim/agents/{agent_id}/` subdirectories
- `ExperienceBroker` (already exists) handles knowledge sharing between agents
- LLM Router is **shared** across agents (expensive resource), but session cost tracking is per-agent

**New files:**
- `src/maxim/runtime/agent_factory.py` (~200)
- `src/maxim/runtime/agent_pool.py` (~200)
- Tests (~200)

**Modified:**
- `src/maxim/runtime/agent_loop.py` — extract single-turn execution into `run_single_turn()` callable from pool. **Note:** `run_agentic_loop()` is ~1,700 lines with heavy instrumentation (autonomy checks, LLM worker async, context pool, sim logging). The loop is a `for step_num in step_iter:` iterator, not a while-loop, so extraction is feasible but requires careful stripping of per-session setup from per-turn logic. Expect ~200 LOC for the extraction itself.
- `src/maxim/mesh/bus.py` — `LocalMessageBus` currently routes by **nickname string**, not agent_id. Add `agent_id`-based registration so agents can subscribe with their unique ID. The existing nickname routing stays for backward compat.
- `src/maxim/tools/registry.py` — add `threading.RLock` around `_tools` dict mutations
- `src/maxim/simulation/sim_logger.py` — de-globalize into `SimLogger` class with thread-local default (~200 LOC, 29 call sites). Per-agent loggers become possible once agents own their own SimLogger instance.

**Ship gate:** Can spawn 3 independent agents, run concurrent turns, verify each has separate hippocampus memories. Knowledge sharing via ExperienceBroker works between agents.

---

## Phase 4: Party DM Runtime (~400 LOC)

**Why fourth:** Proves multi-agent works in the most demanding scenario (D&D campaigns with NPC agents that learn and remember).

### Design

**Party DM Runtime:**
```python
# src/maxim/simulation/dm_party.py (~300 LOC)

class PartyDMRuntime:
    """DM runtime for multi-agent campaigns.

    Extends DMRuntime with:
    - Turn order management (initiative system)
    - NPC agents with real memory + learning
    - Party choice resolution (consensus, majority, first-to-act)
    - Inter-agent observation (agents witness each other's actions)
    - Post-campaign memory export per agent
    """

    def __init__(self, campaign: CampaignDef, bridge: SimulationBridge,
                 agent_pool: AgentPool, agent_factory: AgentFactory, ...):
        ...

    def _setup_npc_agents(self) -> None:
        """For each NPC in campaign, spawn an NPC agent via AgentFactory.

        NPC agents get:
        - Entity loaded from ComponentRegistry
        - Personality from campaign NPC definition
        - Small LLM tier
        - Restricted tool set
        """

    def _run_encounter_round(self, encounter: EncounterDef) -> RoundResult:
        """Execute one encounter for all active agents.

        Turn order:
        1. DM delivers scene narrative to all agents
        2. NPC agents react first (generate dialogue, update internal state)
        3. PC agent observes NPC reactions + scene, makes choice
        4. DM resolves choice, applies effects
        5. All agents witness outcome (feeds into their hippocampus)
        """

    def _resolve_party_choice(self, choices: dict[str, str]) -> str:
        """Resolve when multiple agents make different choices.

        Modes:
        - 'pc_decides': PC choice wins (default for single-PC parties)
        - 'majority': Most common choice wins
        - 'first': First agent to respond wins
        - 'negotiation': Agents discuss before choosing (future)
        """

    def get_agent_memories(self) -> dict[str, dict]:
        """Export all agent memories for post-campaign analysis."""
```

**Campaign schema additions:**
```yaml
campaign:
  name: Heist of the Golden Crown
  party_mode: true                    # Enables PartyDMRuntime
  choice_resolution: pc_decides       # How conflicting choices resolve
  npcs:
    guard_captain:
      ref: "npcs/guard"            # Load from registry
      personality: "Loyal to the crown, suspicious of strangers"
      model_tier: small               # LLM tier for this NPC
      remembers: true                 # Enable hippocampus (default true)
      learns: true                    # Enable NAc (default true)
```

**What this enables (the compelling part):**
- Guard NPC *remembers* that the PC lied to it 3 encounters ago
- Merchant NPC *learns* that the PC tends to threaten → becomes fearful over time
- Companion NPC *shares knowledge* with PC via ExperienceBroker when they witness the same event
- Post-campaign reports include per-NPC memory dumps, causal links learned, pain events experienced

**New files:**
- `src/maxim/simulation/dm_party.py` (~300)
- Tests (~100)

**Modified:**
- `src/maxim/simulation/dm_schema.py` — add `party_mode`, `choice_resolution`, `model_tier`, `remembers`, `learns` fields
- `src/maxim/simulation/orchestrator.py` — route to `PartyDMRuntime` when `party_mode: true`

**Campaign save/load for resume (~150 LOC, folded in from Phase 0.1):**
- `save_campaign_state(session_id)` — persists campaign YAML path + `CampaignState` dict + per-agent memory snapshots + entity sensor states to `~/.maxim/sim_reports/{session_id}/campaign_checkpoint.json`
- `load_campaign_state(session_id)` — reconstructs `AgentPool` from checkpoint, restores each agent's Hippocampus/NAc/ATL state, resumes from last completed encounter
- CLI: `maxim --sim campaign.yaml --resume-campaign <session_id>`
- Uses `atomic_write_json` for checkpoint persistence

**Ship gate:** Run a 3-encounter campaign with 1 PC + 2 NPC agents. NPCs produce independent memories. At least one NPC demonstrates learned behavior (changes response based on prior PC actions). Campaign can be interrupted and resumed from checkpoint.

---

## Phase 5: Hippocampus Recall Refinement (~400 LOC)

**Why after Phase 4:** NPC agents need good memory to be interesting. Also the #1 priority from experiments — behavioral recall failed at the door challenge. Moved after Party DM (Phase 4) because running real multi-agent campaigns will reveal whether recall is actually the bottleneck vs. other issues. The experiment failure at the door challenge might not reproduce with the new multi-agent setup.

This was already Priority #1 in future_plans.md. Pulling it into the buildout because multi-agent NPCs amplify the problem: if one agent's memory is weak, it's a curiosity; if every NPC has weak memory, the whole party mode feels broken.

### Design

Improve `memory_recall` tool and hippocampus query pipeline:

1. **Semantic relevance ranking** — Score recall results by cosine similarity to query, not just recency. Use EC embeddings if available, fall back to keyword overlap.

2. **Modality-aware recall** — `memory_recall("what did I hear?")` filters to episodes tagged with `SensoryTag.AUDITORY`. Uses existing `SensoryTag` metadata that hippocampus already captures.

3. **Decision rationale search** — `memory_recall("why did I choose X?")` searches `decision_rationale` field in episodic memories. Already captured by provenance system, just not queryable.

4. **Spam reduction** — Rate-limit observation capture (already partially done). Add deduplication window: don't capture near-identical observations within 30s.

5. **Cross-encounter recall** — When DM delivers a new encounter that references a prior NPC name, automatically prime recall with that NPC's memory cluster. Uses hippocampus associative graph (already implemented, spreading activation with decay=0.5).

**Modified:**
- `src/maxim/memory/hippocampus.py` — relevance ranking, modality filter, dedup window
- `src/maxim/tools/introspection.py` — `memory_recall` tool gains `modality` and `rationale` params
- `src/maxim/integration/memory_hub.py` — cross-encounter priming hook
- Tests (~100)

**Ship gate:** Re-run hippocampal recall experiment. Behavioral recall at door challenge succeeds (AUT references Verath from indirect cues, not just direct "what do you remember?" prompts).

---

## Phase 6: Interactive Runtime + Rich Display (~500 LOC) — IN PROGRESS

**Why sixth:** The user currently has no clean way to interact with a running agent. `AskUserTool` already exists (`tools_user.py`, ~420 LOC) and handles structured prompts with timeout/replay/audit. But it uses bare `print()` + `input()`, and there's no generalized prompt protocol, no display framework, and no DM-specific UI. This phase builds three layers: a universal prompt protocol, a `rich`-based display, and a DM campaign extension.

### The Problem

Right now during a running simulation:
- Agent logs scroll past at speed, burying any prompt
- `AskUserTool` prints a bare `> ` that's visually indistinguishable from log output
- Users can't see campaign state (HP, inventory, encounter progress) without reading logs
- There's no way to do freeform input (user wants to say something unprompted)
- `print()` throughout library code means output is unstructured and unfilterable

### Activation Rules

The interactive display + prompt system is **opt-in, not force-on**. It only activates when user input is relevant to the mode:

| Mode | `--interactive` default | Rich display | Why |
|------|------------------------|-------------|-----|
| `maxim` (agentic) | `False` | Off | Agent loop runs autonomously |
| `maxim --sim agent` | `False` | Off | Orchestrator LLM drives sim |
| `maxim --sim "goal string"` | `False` | Off | Narrator drives generative campaign |
| `maxim --sim campaign.yaml` | `False` | Off | Direct injection, deterministic |
| `maxim --sim campaign.yaml --dm` | **`True`** | **On** | User IS the player — input is the point |
| `maxim --sim "goal" --dm` | **`True`** | **On** | Architect interviews user, then DM runs |
| Python API `maxim.campaign(...)` | `False` | Off | Programmatic default |
| Python API `maxim.campaign(..., interactive=True)` | `True` | On | Explicit opt-in |
| Python API `maxim.imagine(...)` | `False` | Off | Headless default |

**Override rules:**
- `--interactive` / `interactive=True` — forces display + prompts on in any mode
- `--interactive False` / `interactive=False` — forces everything off, even in `--dm` mode (runs campaign with defaults, no user input, useful for automated testing of campaigns)
- `--non-interactive` — alias for `--interactive False` (existing flag, preserved for compat)
- `--replay-from <session>` — implies non-interactive, reads recorded responses from JSONL

**Rich display specifically** only activates when ALL of: (1) interactive is True, (2) `rich` is installed, (3) stdout is a TTY. Otherwise it falls back to `PlainPromptHandler` (bare print/input, current behavior). This means:
- Piped output (`maxim --dm ... | tee log.txt`) → plain mode
- CI environments → plain mode
- `pip install pymaxim` without `[ui]` extra → plain mode
- SSH sessions without TTY → plain mode

### Layer 1: Universal Prompt Protocol (~150 LOC)

A generalized prompt system that abstracts *what* the system is asking from *how* it's rendered. Every interaction — DM choices, architect interviews, agent questions, freeform input — goes through this protocol.

```python
# src/maxim/interactive/prompts.py (~150 LOC)

class PromptType(Enum):
    """Universal prompt types — cover every interactive use case."""
    SINGLE_CHOICE = "single_choice"     # Pick one from a list
    MULTI_CHOICE = "multi_choice"       # Pick N from a list (checkboxes)
    CONFIRM = "confirm"                 # Yes/No
    SHORT_TEXT = "short_text"           # One-line freeform (name, keyword)
    LONG_TEXT = "long_text"             # Multi-line freeform (backstory, notes)
    FREEFORM = "freeform"              # Unprompted user input (agent chat)
    NUMERIC = "numeric"                 # Number input with range validation
    RATING = "rating"                   # 1-5 or 1-10 scale

@dataclass(frozen=True)
class PromptRequest:
    """What the system wants from the user.

    Note: frozen=True for hashability/safety. Mutable fields use
    immutable alternatives (tuple instead of dict for context).
    """
    prompt_type: PromptType
    question: str
    options: tuple[str, ...] | None = None  # For SINGLE_CHOICE, MULTI_CHOICE (tuple, not list)
    default: str | None = None
    timeout_sec: float = 300.0
    min_selections: int = 1            # For MULTI_CHOICE
    max_selections: int | None = None  # For MULTI_CHOICE (None = unlimited)
    value_range: tuple[float, float] | None = None  # For NUMERIC (renamed: 'range' shadows builtin)
    context: tuple[tuple[str, Any], ...] = ()  # Metadata as frozen key-value pairs
    # context examples (constructed via helper):
    #   PromptRequest(..., context=freeze_context(category="character_creation", phase="attributes"))
    #   Helper: freeze_context(**kw) -> tuple[tuple[str, Any], ...] = tuple(kw.items())

@dataclass(frozen=True)
class PromptResponse:
    """What the user responded."""
    value: str | list[str]             # Single value or list for MULTI_CHOICE
    timed_out: bool = False
    was_default: bool = False
    elapsed_s: float = 0.0

class PromptHandler(ABC):
    """How prompts are delivered and responses collected."""

    @abstractmethod
    def prompt(self, request: PromptRequest) -> PromptResponse:
        """Present prompt to user and collect response."""

    def supports_freeform(self) -> bool:
        """Whether this handler supports unprompted user input."""
        return False

    def poll_freeform(self) -> str | None:
        """Check for unprompted user input (non-blocking). Returns None if nothing."""
        return None
```

**Built-in handlers:**

| Handler | Where it runs | What it does |
|---------|---------------|-------------|
| `RichPromptHandler` | CLI with rich installed | Renders prompts as rich panels, options as tables |
| `PlainPromptHandler` | CLI without rich, or dumb terminal | Current print() + input() behavior (backwards compat) |
| `CallbackPromptHandler` | Python API | Delegates to user-provided callback function |
| `ReplayPromptHandler` | `--replay-from` mode | Reads from JSONL audit log (already exists) |
| `NonInteractiveHandler` | `--non-interactive` or CI | Returns defaults immediately |

**Integration with existing `AskUserTool`:**
- `AskUserTool._execute_interactive()` currently does its own print/input (lines 358-421)
- Refactor: `AskUserTool` constructs a `PromptRequest`, passes it to the active `PromptHandler`
- The handler decides rendering + input collection
- Response flows back as `PromptResponse` → same audit/escalation logic

**Integration with Python API (Phase 8):**
```python
# User provides a callback — receives PromptRequest, returns value
def my_handler(request):
    if request.prompt_type == PromptType.SINGLE_CHOICE:
        return request.options[0]  # Always pick first
    return request.default

result = maxim.imagine(goal="test", prompt_handler=my_handler)

# Or use the event system
maxim.on("prompt", lambda req: "yes" if req.prompt_type == PromptType.CONFIRM else None)
```

### Layer 2: Rich Display Framework (~200 LOC)

A `rich`-based live terminal display that replaces the current raw ANSI output. Provides structured panels with a persistent input area.

```python
# src/maxim/interactive/display.py (~200 LOC)

class MaximDisplay:
    """Rich-based live terminal display with structured panels.

    Layout:
    ┌─ Status ─────────────────────────────────────────┐
    │ Mode: simulation  Goal: test memory  Turn: 7/∞   │
    ├─ Agent Log ──────────────────────────────────────┤
    │ [hippo] Captured: "merchant offered healing pot…  │
    │ [nac]   Link: threaten → hostility (conf 0.82)   │
    │ [exec]  Tool: memory_recall("healing") → 2 hits  │
    │ [pain]  0.3 — overextension warning              │
    ├──────────────────────────────────────────────────┤
    │ > What do you do? [fight / negotiate / flee]     │
    │ > _                                              │
    └──────────────────────────────────────────────────┘
    """

    def __init__(self, title: str = "Maxim"):
        self._live: Live | None = None     # rich.live.Live context
        self._log_lines: deque[str] = deque(maxlen=200)
        self._status: dict[str, str] = {}
        self._current_prompt: PromptRequest | None = None
        self._extensions: list[DisplayExtension] = []

    def start(self) -> None:
        """Begin live display. Call from main thread."""

    def stop(self) -> None:
        """End live display, restore terminal."""

    def log(self, subsystem: str, message: str, level: str = "info") -> None:
        """Add a line to the scrolling log panel."""

    def set_status(self, **fields) -> None:
        """Update status bar fields (mode, goal, turn, model, etc.)."""

    def show_prompt(self, request: PromptRequest) -> None:
        """Render a prompt in the input panel."""

    def clear_prompt(self) -> None:
        """Clear the input panel after response collected."""

    def add_extension(self, ext: DisplayExtension) -> None:
        """Register a display extension (adds panels)."""

    def _build_layout(self) -> Layout:
        """Compose all panels into rich Layout."""


class DisplayExtension(ABC):
    """Protocol for adding custom panels to the display."""

    @abstractmethod
    def panel_name(self) -> str:
        """Name shown in panel header."""

    @abstractmethod
    def render(self) -> RenderableType:
        """Return a rich renderable for this panel."""

    def key_bindings(self) -> dict[str, Callable]:
        """Optional key bindings this extension handles (e.g., 'i' for inventory)."""
        return {}
```

**Integration with `sim_logger`:**
- `sim_log()` currently writes ANSI to stdout
- When `MaximDisplay` is active, `sim_log()` routes to `display.log()` instead
- Subsystem labels (HIPPO, NAc, FEAR, etc.) become filterable tags
- When display is NOT active (headless, piped, no rich), falls back to current behavior

**Graceful degradation:**
```python
try:
    from rich.live import Live
    from rich.layout import Layout
    _RICH_AVAILABLE = True
except ImportError:
    _RICH_AVAILABLE = False

def create_display(mode: str = "auto") -> MaximDisplay | None:
    """Create display if rich is available and stdout is a TTY."""
    if mode == "off" or not _RICH_AVAILABLE or not sys.stdout.isatty():
        return None
    return MaximDisplay()
```

### Layer 3: DM Campaign Display Extension (~150 LOC)

A `DisplayExtension` that adds game-specific panels for DM campaigns. Activated automatically when `DMRuntime` or `PartyDMRuntime` starts.

```python
# src/maxim/interactive/dm_display.py (~150 LOC)

class CampaignDisplay(DisplayExtension):
    """DM campaign display extension — character sheet, inventory, encounter state.

    Adds toggleable panels via key bindings:
      [c] Character sheet    [i] Inventory / bag    [s] Spells / abilities
      [n] Notes (editable)   [e] Encounter info     [r] NPC relationships
      [m] Map / scene        [h] History / choices

    Layout when panels active:
    ┌─ Agent Log ────────────┬─ Character ──────────────┐
    │ [scene] The guard      │ Aldric (Human Warrior)   │
    │   blocks your path...  │ HP: 15/20  Stam: 8/10   │
    │ [npc] Guard: "Halt!"   │ STR: 16  DEX: 12        │
    │ [choice] fight/talk/run│ CON: 14  WIS: 10        │
    ├────────────────────────┼──────────────────────────┤
    │ > _                    │ [i] Inventory            │
    │                        │  * Rusty Sword (equipped) │
    │                        │  * Healing Potion (x2)   │
    │                        │  * Lockpick Set          │
    └────────────────────────┴──────────────────────────┘
    """

    def __init__(self, campaign_state: CampaignState, pc_entity: Entity | None = None):
        self._state = campaign_state
        self._pc = pc_entity
        self._active_panel: str = "character"  # Which side panel is shown
        self._notes: list[str] = []            # User's personal notes
        self._show_side: bool = True           # Toggle side panel visibility

    # --- Renderable panels ---

    def render_character(self) -> Table:
        """Character sheet — name, race, class, attributes, HP, stamina."""
        # Reads from PC entity sensors: hp, stamina, str, dex, con, int, wis, cha

    def render_inventory(self) -> Table:
        """Bag contents + equipped items. Groups by: equipped, consumables, quest, misc."""
        # Reads from PC entity children with entity_type == "item"
        # Equipped items marked via metadata or sensor value

    def render_spells(self) -> Table:
        """Known spells/abilities with slot tracking."""
        # Reads from PC entity children with entity_type == "ability" or "spell"
        # Slot usage from sensor values

    def render_encounter(self) -> Panel:
        """Current encounter: scene text, active NPCs, available choices."""

    def render_relationships(self) -> Table:
        """NPC relationship sensors: trust, mood, hostility, fear."""
        # Reads NPC entity sensors for relationship-type values

    def render_notes(self) -> Panel:
        """User's personal notes (editable via [n] key)."""

    def render_history(self) -> Table:
        """Choices made, encounters visited, dice rolls."""

    # --- Key bindings ---

    def key_bindings(self) -> dict[str, Callable]:
        return {
            "c": lambda: self._set_panel("character"),
            "i": lambda: self._set_panel("inventory"),
            "s": lambda: self._set_panel("spells"),
            "e": lambda: self._set_panel("encounter"),
            "r": lambda: self._set_panel("relationships"),
            "h": lambda: self._set_panel("history"),
            "n": self._edit_notes,        # Opens LONG_TEXT prompt for adding a note
            "tab": self._toggle_side,     # Show/hide side panel entirely
        }

    def _edit_notes(self) -> None:
        """Prompt user to add a note via PromptHandler."""
        # Uses PromptRequest(LONG_TEXT, "Add a note:")
        # Appends to self._notes
        # Notes persist to session JSONL for replay

    # --- Data binding ---

    def bind_campaign_state(self, state: CampaignState) -> None:
        """Update reference when campaign state changes."""

    def bind_pc_entity(self, entity: Entity) -> None:
        """Update reference when PC entity changes (equipment, HP, etc.)."""
```

**How the DM display reads entity state:**

**Important implementation detail:** In the current DM system, the PC and NPCs are stored as raw YAML dicts in `CampaignDef` (`pc_spec`, `npc_specs`). They become instantiated `Entity` objects at runtime when `SceneState` calls `load_spec()` to register tools. `CampaignState` only tracks *narrative* state (encounters, choices, flags, dice rolls) — NOT character HP, inventory, or spells.

The display therefore needs **two data sources**:
- `CampaignState` — for encounter progress, choices made, flags, dice rolls
- `SceneState`'s instantiated entities — for HP, inventory, spells, NPC relationships

Once the entities are instantiated, SEM sensors provide everything:
- `pc_entity.find("body").sensors["hp"].read()` → current hit points
- `pc_entity.find("combat").sensors["equipped"].read()` → what's in hand
- PC entity children with `entity_type == "item"` → bag contents
- `npc_entity.find("social").sensors["trust"].read()` → NPC relationship
- `pc_entity.find("divine_magic").sensors["slots"].read()` → spell availability

The display just calls `sensor.read()` on the runtime entities — it doesn't need special game logic. SEM already models all of this. The `CampaignDisplay` constructor takes both `campaign_state` and a reference to the `SceneState` (or the instantiated PC entity directly).

**Notes persistence:**
- Notes stored in `UserInteraction` records with `question="[user_note]"` convention
- Replayed via `ReplayPromptHandler` like any other interaction
- Exported in campaign rollup JSON for post-analysis

### Integration Points

**With existing systems:**

| System | Integration |
|--------|-------------|
| `AskUserTool` (tools_user.py) | Refactor to construct `PromptRequest`, delegate to active `PromptHandler` |
| `sim_logger` (sim_logger.py) | Route `sim_log()` to `MaximDisplay.log()` when display active |
| `DMRuntime` (dm_runtime.py) | Auto-create `CampaignDisplay` extension on campaign start |
| `PartyDMRuntime` (Phase 4) | Extend `CampaignDisplay` with party member tabs |
| `bridge.py` spinner | Replace with display status update |
| `orchestrator.py` | Pass `PromptHandler` to tools, create display on sim start |
| `formatting.py` | Keep for non-display contexts; display uses rich natively |
| Python API (Phase 8) | `CallbackPromptHandler` wraps user callback |

**With DM campaigns specifically:**

```python
# In DMRuntime.run():
if display:
    dm_display = CampaignDisplay(self._state, self._pc_entity)
    display.add_extension(dm_display)

# In _run_encounter():
dm_display.bind_campaign_state(self._state)
# Entity sensors auto-update on cascade resolution — display reads current values

# Encounter choice becomes a PromptRequest:
request = PromptRequest(
    prompt_type=PromptType.SINGLE_CHOICE,
    question=f"The guard blocks your path. What do you do?",
    options=["fight", "negotiate", "flee"],
    context={"encounter": encounter.name, "act": act.name},
)
response = handler.prompt(request)
```

**With Generative Architect (Phase 7):**

```python
# Character creation uses the prompt protocol:
name = handler.prompt(PromptRequest(
    prompt_type=PromptType.SHORT_TEXT,
    question="What's your character's name?",
    context={"category": "character_creation", "step": 1},
))

race = handler.prompt(PromptRequest(
    prompt_type=PromptType.SINGLE_CHOICE,
    question="Race?",
    options=["human", "elf", "dwarf", "halfling", "half-elf", "other"],
))

abilities = handler.prompt(PromptRequest(
    prompt_type=PromptType.MULTI_CHOICE,
    question="Choose starting abilities (pick up to 3):",
    options=["Power Strike", "Shield Bash", "Healing Word", "Sneak Attack", "Fireball"],
    max_selections=3,
))

backstory = handler.prompt(PromptRequest(
    prompt_type=PromptType.LONG_TEXT,
    question="One-line backstory? (or leave blank to generate)",
    default="generate",
))
```

### New files
- `src/maxim/interactive/__init__.py`
- `src/maxim/interactive/prompts.py` (~150) — protocol, request/response, handler ABC + built-in handlers
- `src/maxim/interactive/display.py` (~200) — `MaximDisplay`, `DisplayExtension`, layout composition
- `src/maxim/interactive/dm_display.py` (~150) — `CampaignDisplay` extension
- Tests (~100)

### Modified
- `src/maxim/simulation/tools_user.py` — refactor `AskUserTool` to use `PromptHandler` instead of bare print/input
- `src/maxim/simulation/sim_logger.py` — route to display when active
- `src/maxim/simulation/dm_runtime.py` — create `CampaignDisplay` on campaign start, pass `SceneState` reference
- `src/maxim/simulation/orchestrator.py` — create display + pass handler. **Note:** `AskUserTool` exists but is NOT currently registered in orchestrator.py — must add registration here.
- `pyproject.toml` — add `rich>=13.0.0` to new `ui` optional extra

### Dependency
```toml
[project.optional-dependencies]
ui = ["rich>=13.0.0"]
```

### Ship gate
1. `pip install pymaxim[ui]` → rich display works with scrolling log + input panel <--- probably should ship with ui always and just be used if user requests --interactive, should be noted somewhere else, ask if confused. --dm should turn --interactive True by default but allow the user to turn off with --interactive False or whatever to run the dm automatically
2. `pip install pymaxim` (no ui) → falls back to plain print/input (current behavior)
3. DM campaign shows character sheet, inventory, encounter info in side panels
4. Generative Architect interview uses SINGLE_CHOICE, MULTI_CHOICE, SHORT_TEXT, LONG_TEXT prompts
5. `--non-interactive` and `--replay-from` still work (handlers are swappable)
6. Python API users can pass `prompt_handler=callback` to `imagine()`/`campaign()`

---

## Phase 7: Generative Architect Persona (~600 LOC)

**Why seventh:** Makes the system usable without hand-authoring YAML. Uses component registry (Phase 1), encounter library (Phase 2), and ask_user (Phase 6).

This is Extension B from the DM extensions plan, pulled forward. Implementation as designed there — multi-phase interview, character creation sub-flow, campaign generation.

**Key change from original plan:** Architect now composes from ComponentRegistry and EncounterLibrary instead of generating everything from scratch. Three internal LLM sub-personas handle the heavy lifting: Entity Designer, Encounter Adapter, and the Architect orchestrator itself.

### The Architect's Campaign Creation Flow

```
User: "maxim --sim 'run a dark fantasy heist' --dm"

1. INTERVIEW (ask_user, Phase 6)
   → Theme, setting, tone, PC concept, desired difficulty

2. ENTITY DESIGNER (medium-tier LLM)
   → Generate PC from user description → valid SEM spec
   → Generate 3-5 NPCs from roles → valid SEM specs (using ComponentRegistry bases)
   → Generate items/objects → valid SEM specs (cursed sword, healing potion, etc.)

3. ENCOUNTER ADAPTER (medium-tier LLM)
   → Browse EncounterLibrary for narrative arc matches
   → Adapt 3-5 library encounters to campaign NPCs/flags/tone
   → Generate 1-2 original encounters for campaign-specific moments

4. ARCHITECT ORCHESTRATOR (medium-tier LLM)
   → Assemble acts + encounter order + branch graph
   → Wire flags across encounters (flag set in enc A read in enc B)
   → Generate bio-system expectations
   → Emit complete campaign YAML

Total: ~8-10 LLM calls × ~200 tokens = ~2,000 tokens. Under $0.02.
```

### Entity Designer — LLM-driven SEM spec generation

The primary friction in campaign creation is authoring valid SEM entity YAML (sensors with ranges, modulators with affordances, failure modes with thresholds). The Entity Designer eliminates this:

**Input:** Natural language description + optional ComponentRegistry base
**Output:** Valid SEM entity spec dict ready for `_parse_entity()`

```
User: "A guard captain, suspicious and battle-hardened"

Entity Designer:
  1. Selects base: "npcs/guard" from ComponentRegistry
  2. Generates overrides from description:
     - name: captain_aldric
     - Adds sensor: battle_scars (flavor, ratio 0-1, initial 0.8)
     - Adjusts: hp.initial=25, trust.initial=0.2
     - Adds combat affordance: rally_troops
     - Adds failure_mode: paranoia (suspicion > 0.9, pain 0.3)
  3. Validates against SEM schema (sensors have unit+range, affordances have params)
  4. Returns valid spec dict
```

**Same pattern for items:**
```
"A cursed sword that grows stronger as it drinks blood"
  → Base: weapons/rusty_sword
  → Adds sensor: blood_charge (count 0-10, initial 0)
  → Modifies slash description to mention scaling damage
  → Adds failure_mode: curse_takeover (blood_charge > 8, pain 0.9)
```

**And environments:**
```
"A dimly lit underground marketplace, crowded and dangerous"
  → Base: environments/tavern_interior
  → Adjusts: lighting.initial=0.2, noise_level.initial=0.7
  → Adds sensor: danger_level (ratio 0-1, initial 0.6)
  → Adds affordance: blend_in_crowd
```

**LLM tier:** Medium by default (needs to understand SEM schema semantics). Small as fallback with quality warning. The designer's system prompt includes the SEM schema spec (sensor format, modulator/affordance format, failure_mode format) so the LLM knows what valid output looks like.

**Validation:** Designer output is validated against the SEM schema before use. If validation fails, the designer retries once with the error message. If retry fails, falls back to the base template without overrides.

### Encounter Adaptation via LLM (Tier 2 parameterization)

Instead of static `$TOKEN` parameterization in encounter templates, the architect uses an LLM to adapt library encounters to campaign context at creation time:

**Flow:**
1. Architect browses `EncounterLibrary.query(tags=["combat"], narrative_role="rising_action")`
2. Selects `"combat/forest_ambush"` as a match for the campaign's narrative arc
3. Calls the **Encounter Adapter** LLM with:
   - Library encounter template (scene, choices, dice)
   - Campaign NPCs (names, personas, relationships — from Entity Designer output)
   - Campaign tone/setting ("dark fantasy, morally gray")
   - Current narrative state (flags set by prior encounters, arc phase)
   - Branch targets (what encounters come next in this campaign)
4. LLM returns a fully wired encounter: adapted scene text (references campaign NPCs by name), `dialogue_hints` (contextual to campaign flags), `on_choice` effects (appropriate flag names), NPC assignment
5. Architect emits the adapted encounter into campaign YAML

**LLM tier:** Medium by default. Small as fallback. Cost: ~100-200 tokens per encounter × 5-8 encounters = ~1,500 tokens total, once at campaign creation.

**Why LLM adaptation beats static parameterization:**
- No `$TOKEN` syntax for encounter authors to learn — templates are just good prose
- Context-aware: adapts dialogue tone based on setting, references NPCs naturally
- Handles complexity gracefully: conditional reveals, multi-flag dialogue variants
- Falls back cleanly: if no library match, architect generates encounter from scratch

### Dynamic NPC Dialogue (replaces static dialogue_hints)

Currently, `dialogue_hints` in campaigns are static text keyed by flags. With Party Mode (Phase 4) giving NPCs real memory and learning, static hints become a bottleneck.

**Enhancement:** When delivering dialogue to the AUT, the DM runtime can optionally pass the hint through a **small-tier LLM** that enriches it with:
- NPC's personality (from entity metadata.persona_prompt)
- NPC's current sensor state (trust, mood, hp)
- Campaign flag context
- NPC's hippocampus memories (if Party Mode is active — the NPC *remembers* prior interactions)

Static hints remain the fallback when LLM is unavailable or `--non-interactive` is set.

### New files
- Entry in `src/maxim/simulation/personas.py` for `adventure_architect`
- `src/maxim/simulation/entity_designer.py` (~150) — Entity Designer LLM sub-persona with SEM schema prompt + validation
- `src/maxim/simulation/character_templates.py` (~120) — class archetypes, NPC role templates (fallback when LLM unavailable)
- Tool additions in `src/maxim/simulation/tools_dm.py` — `emit_campaign`, `browse_encounters`, `browse_components`, `design_entity`, `adapt_encounter`
- Tests (~100)

**Ship gate:**
1. Architect produces a runnable campaign with PC + 3+ NPCs in < 8 minutes
2. Entity Designer generates valid SEM specs from natural language descriptions
3. At least 2 encounters sourced from library and adapted via LLM
4. Campaign runs end-to-end without manual YAML edits
5. Works with `party_mode: true`
6. Fallback: works with small-tier LLM (lower quality but functional)

---

## Phase 8: API Surface Expansion (~400 LOC)

**Why now:** The current public API (6 verbs) hides most of what makes Maxim interesting. Researchers and developers can't programmatically run campaigns, benchmarks, register tools, subscribe to events, or access memory systems. The CLI can do significantly more than the API. This gap must close before publication — adding verbs later is additive, but the *patterns* (how sessions connect to observation, how tools register, how events flow) are architectural and should be right from v0.2.0.

### 8a. New API verbs (~200 LOC in api.py)

```python
# Campaign execution (programmatic access to DM system)
result = maxim.campaign("scenarios/campaigns/heist_v1.yaml", model="mistral-7b")
result = maxim.campaign(
    "scenarios/campaigns/heist_v1.yaml",
    party_mode=True,
    model="claude-sonnet",
    npc_model="mistral-7b",
)
# Returns: CampaignResult (choices_made, agent_memories, expectations, encounter_log)

# Benchmark execution (programmatic multi-model comparison)
report = maxim.benchmark(
    models=["mistral-7b", "qwen2.5-14b"],
    suite="cognitive",          # or path to custom suite YAML
    runs=3,
)
# Returns: BenchmarkReport (per_model_scores, comparison_table, expectations)

# Research protocol (experiment → paper)
result = maxim.research(
    goal="hippocampal recall under interference",
    campaign="scenarios/experiments/hippocampal_recall_short.yaml",
    model="claude-sonnet",
    aut_model="mistral-7b",
)
# Returns: ResearchResult (paper_draft, review, experiment_data)
```

### 8b. Session-linked observation (~50 LOC)

**Problem:** `observe()` reads from persisted state but has no concept of "which run." After `imagine()` returns, you can't query the memories *from that specific run*.

**Fix:** `imagine()`, `campaign()`, and `research()` all return result objects with a `session_id`. `observe()` accepts an optional `session` param:

```python
result = maxim.imagine(goal="test memory", persona="cooperative")
memories = maxim.observe("memory", session=result.session_id)
causal = maxim.observe("causal", session=result.session_id)
```

### 8c. Tool registration helper (~30 LOC)

```python
# Register a custom tool available to all agents
@maxim.tool
def my_analysis(data: str, depth: int = 3) -> str:
    """Analyze data at specified depth."""
    return f"Analysis of {data} at depth {depth}"

# Or class-based
maxim.register_tool(MyCustomTool())
```

Internally, this stores tools in a pending registry that gets injected at `run()`/`imagine()`/`campaign()` time.

### 8d. Event subscription helpers (~50 LOC)

```python
# Subscribe to agent events (returned handle for cleanup)
handle = maxim.on("tool_call", lambda event: print(f"Tool: {event.tool_name}"))
handle = maxim.on("memory_capture", lambda event: print(f"Memory: {event.content[:50]}"))
handle = maxim.on("pain_signal", lambda event: print(f"Pain: {event.intensity}"))

# Cleanup
handle.unsubscribe()
```

Events bridge to the existing `AgentBus` and `PainBus` internally. The `maxim.on()` pattern is additive — doesn't change how buses work, just exposes subscription.

### 8e. Configuration expansion (~30 LOC)

```python
# LLM model selection (currently only via CLI or env vars)
maxim.configure(model="mistral-7b")
maxim.configure(model="claude-sonnet", cloud_budget=2.00)

# Autonomy level (currently only via CLI)
maxim.configure(autonomy="supervised")  # planning | supervised | autonomous
```

### 8f. Custom persona registration (~40 LOC)

```python
maxim.register_persona(
    name="medical_tester",
    description="Tests medical knowledge and safety boundaries",
    focus="healthcare decision-making and drug interactions",
    context_prompt="You are testing a medical AI assistant...",
    max_initiative=0.8,
)
```

### New/Modified files:
- `src/maxim/api.py` — new verbs + helpers (~300)
- `src/maxim/__init__.py` — export new verbs + helpers (~30)
- `src/maxim/utils/event_bridge.py` — thin bridge from `maxim.on()` to internal buses (~70)
- Tests (~100)

**Ship gate:** All new verbs work. `campaign()` runs a DM campaign and returns structured results. `observe()` with `session` param returns memories from a specific run. `register_tool()` makes tools available to the agent. `on()` delivers events to callbacks.

---

## Phase 9: PyPI Dependency Restructuring + Docs + Pre-Pub Hardening (~900 LOC)

### 9a. Dependency audit (~100 LOC)

Phase 1 from the existing PyPI publication plan. Gate optional imports, verify `pip install pymaxim` works with base deps (numpy, scipy, pyyaml, json-repair, rich).

**Base dependency changes (already applied to pyproject.toml):**
- `rich` moved from optional `[ui]` extra to base — pure Python, 3 tiny deps, used throughout interactive codepaths
- `dateparser` stays optional `[temporal]` — pulls `regex` (C extension) which can fail on systems without a compiler
- `[ui]` extra removed (rich is always available)

**Audit sweep:**
- Verify every non-core import is inside function body or `try/except ImportError`
- Add helpful error messages: `raise ImportError("Install pymaxim[llm-anthropic] for Claude support")`
- Test clean install in fresh venv: `pip install pymaxim && python -c "import maxim; maxim.diagnose()"`
- Relax `requires-python` to `>=3.10` (no 3.12-exclusive features used)
- Relax `numpy>=2.2` to `>=1.26,<3.0` (no numpy 2.x-specific APIs)
- Relax `reachy-mini==1.2.6` to `>=1.2.6,<1.3`
- Add upper bounds: `torch>=2.7,<3.0`, `tensorflow>=2.20,<3.0`

### 9b. Cloud provider builtin profiles (~50 LOC)

The existing `openai_compatible` backend type already supports custom `base_url` and `api_key_env`. Add builtin profiles for popular OpenAI-compatible cloud providers — **zero new backend code needed**, just config entries in `_BUILTIN_PROFILES`:

| Provider | Profile Name | API Key Env | Notes |
|----------|-------------|-------------|-------|
| Google Gemini | `gemini-2.5-flash`, `gemini-2.5-pro` | `GOOGLE_API_KEY` | OpenAI-compatible via `generativelanguage.googleapis.com` |
| Groq | `groq-llama3-70b`, `groq-mixtral` | `GROQ_API_KEY` | Ultra-fast inference |
| Together.ai | `together-llama3-70b`, `together-qwen` | `TOGETHER_API_KEY` | Wide model selection |
| Fireworks | `fireworks-llama3-70b` | `FIREWORKS_API_KEY` | Low-latency |
| Mistral API | `mistral-large`, `mistral-small` | `MISTRAL_API_KEY` | Official Mistral cloud |
| DeepSeek | `deepseek-chat`, `deepseek-reasoner` | `DEEPSEEK_API_KEY` | Strong reasoning models |

Each profile is ~5 lines in config.py. No new dependencies. Users just set the API key env var and `--language-model groq-llama3-70b`.

**Modified:** `src/maxim/models/language/config.py` — add profiles to `_BUILTIN_PROFILES`

### 9c. Installation & setup guide (docs + HTML)

**New files:**
- `docs/user/installation.md` — comprehensive installation guide covering:
  - Base install (`pip install pymaxim`)
  - Optional extras breakdown (what each extra provides, install size, platform requirements)
  - Cloud LLM setup: step-by-step API key configuration for each provider (Anthropic, OpenAI, Gemini, Groq, Together, Fireworks, Mistral, DeepSeek)
  - Local LLM setup: llama.cpp backend, model download, GPU configuration
  - Platform-specific notes (macOS, Linux, Windows/WSL, Docker)
  - Troubleshooting common install failures (missing compiler for C extensions, CUDA version mismatches)
  - `maxim doctor` as post-install verification

- `htmls-guides/maxim-installation.html` — Jinja2 template for dennyschaedig.com:
  - Same content as docs/user/installation.md, formatted as HTML guide page
  - Sections: Quick Start, Base Install, Cloud Providers (with provider logos/cards), Local Models, Optional Packages, Platform Notes, Troubleshooting, Verifying Your Setup
  - Extends `base.html`, Tailwind dark slate/indigo theme, `btn-link` footer nav
  - Route: `/maxim-installation`

- `docs/user/python-api.md` — Full API reference with examples for all verbs, event subscription, tool registration, session observation
- `docs/user/extending-maxim.md` — How to add custom tools, personas, SEM components, encounters without modifying source

### 9d. Example scripts (~50 LOC)

```
examples/
├── quickstart.py              # configure + run in 5 lines
├── run_campaign.py            # campaign() + observe memories
├── custom_tool.py             # register_tool + run
├── event_monitoring.py        # on() + imagine
├── benchmark_comparison.py    # benchmark() + print table
└── post_hoc_analysis.py       # observe() with session linking
```

### 9e. Mother Maxim pre-publication prep (~100 LOC)

Items from [Mother Maxim Plan](mother_maxim_plan.md) M-0 that MUST ship before publication because they define interfaces that users and Mother will depend on. Changing them post-pub would be a breaking change.

**M-0a: Split persistence protocols (~80 LOC)**

Extract save/load into three protocols — one per subsystem. Each has different query patterns (similarity search vs event→outcome lookup vs concept type filtering).

```python
# src/maxim/memory/store.py (~80 LOC)

class EpisodicStore(Protocol):
    """Persistence for Hippocampus episodic memories."""
    def save(self, memories: list[dict], *, namespace: str = "default") -> None: ...
    def load(self, *, namespace: str = "default") -> list[dict]: ...
    def query_similar(self, embedding: list[float], *, top_k: int = 5, namespace: str = "default") -> list[dict]: ...
    def query_by_time(self, start: float, end: float, *, namespace: str = "default") -> list[dict]: ...

class CausalStore(Protocol):
    """Persistence for NAc causal links."""
    def save(self, links: list[dict], *, namespace: str = "default") -> None: ...
    def load(self, *, namespace: str = "default") -> list[dict]: ...
    def query_by_event(self, event_sig: str, *, namespace: str = "default") -> list[dict]: ...

class SemanticStore(Protocol):
    """Persistence for ATL semantic concepts."""
    def save(self, concepts: list[dict], *, namespace: str = "default") -> None: ...
    def load(self, *, namespace: str = "default") -> list[dict]: ...
    def query_by_type(self, concept_type: str, *, namespace: str = "default") -> list[dict]: ...
```

Plus `FileEpisodicStore`, `FileCausalStore`, `FileSemanticStore` wrapping current JSON behavior. Hippocampus/NAc/ATL gain optional `store` constructor parameter (defaults to File variant for backward compat).

**M-0b: NAc thread safety (~30 LOC)**

Add `threading.RLock` around `_links`, `_pending_events`, `_priors` mutations in NAc. Required for multi-agent party mode (already shipped) and Mother's concurrent processing.

**M-0c: Metadata field on EpisodicMemory + SemanticMemory (~20 LOC)**

Add `metadata: dict[str, Any] = field(default_factory=dict)` to both dataclasses + update `to_dict()`/`from_dict()`. Pre-pub this is trivial. Post-pub it requires migration for every user's persisted memories. Mother uses this for: `domain_tags`, `contribution_source`, `witness_count`, `tenant_id`, `deidentification_model`.

---

## Phase 10: Publication Prep (2 files + version bump)

**CHANGELOG.md** — Version history covering 0.1.0 → 0.2.0 changes:
- Package hygiene (data paths, import cleanup, `__main__`)
- Multi-agent runtime (AgentFactory, AgentPool)
- SEM Component Registry with inheritance
- DM Encounter Library with versioned templates
- Party DM Mode (NPC agents with real memory + learning)
- Hippocampus recall improvements
- Generative Architect persona
- ask_user interactive tool
- Expanded Python API (campaign, benchmark, research, tool registration, events)
- Session-linked observation

**CONTRIBUTING.md** — Developer guidelines:
- Code style (ruff, line-length=120)
- Architecture rules (layer dependency graph from ARCHITECTURE.md)
- Testing requirements (pytest markers, conftest fixtures)
- Commit message format
- PR process
- How to add: tools, personas, SEM components, encounters

**SECURITY.md** — Already exists (11.2 KB). Review for completeness, no rewrite needed.

**Version bump:** 0.1.0 → 0.2.0 in `pyproject.toml` + `src/maxim/__init__.py`

---

## Phase 11: Test PyPI (dry-run validation) — IN PROGRESS

1. `python -m build && twine check dist/*`
2. `twine upload --repository testpypi dist/*`
3. Test install in clean venv: `pip install --index-url https://test.pypi.org/simple/ pymaxim`
4. Verify examples work: run each script in `examples/`
5. Verify headless: `python -c "import maxim; maxim.diagnose()"`

**Publication delayed.** Real PyPI upload (`twine upload dist/*`) blocked on Phase 11 completion. Phases 12a (security hardening) and 12b (pre-publication hardening) are done. Test PyPI validation (steps 1-5) proceeds.

---

## Phase 12a: Security Hardening (~200 LOC)

Security fixes identified via pre-publication audit. All are in existing files — no new modules.

**P0 — Must fix before publish (CRITICAL/HIGH):**

| Fix | File | LOC | Issue |
|-----|------|-----|-------|
| Remove `shell=True` from BashTool | `tools/filesystem.py:949` | ~20 | Shell injection via LLM-generated commands. Switch to `subprocess.run(["/bin/bash", "-c", cmd], shell=False)` or restricted shell. Harden `_is_command_safe()` with whitelist, not regex blacklist. |
| Fix path traversal in `sandbox._resolve()` | `simulation/sandbox.py:191` | ~10 | `lstrip("/")` doesn't prevent `../../` escape. Use `os.path.realpath()` + verify result starts with sandbox root. |
| Add tool parameter schema validation | `runtime/executor.py` | ~40 | Validate LLM-generated tool params against each tool's `input_schema` before execution. Currently only FearAgent reviews — add structural validation. |
| Require auth when admin endpoints enabled | `runtime/leader_proxy.py:235` | ~10 | `if not self.api_key: return True` bypasses ALL auth. If admin endpoints are active AND no key is set, refuse to start. |
| Authenticate debug endpoints | `runtime/leader_proxy.py:910` | ~15 | Debug routes execute before auth check. Move auth before debug dispatch, or restrict debug to localhost. |
| Restrict CORS origins | `runtime/leader_proxy.py:946` | ~10 | Replace `Access-Control-Allow-Origin: *` with configurable origin or `null`. |
| Sanitize error responses | `runtime/leader_proxy.py` (multiple) | ~30 | Replace `f"error: {e}"` patterns with generic messages. Log full exception server-side only. ~8 locations. |
| Whitelist subprocess env vars | `utils/sandbox_executor.py:625` | ~15 | Flip blacklist to whitelist: `PATH`, `HOME`, `TMPDIR`, `LANG`, `LC_ALL`, `MAXIM_*` only. |

**P1 — Fix before Mother goes public:**

| Fix | File | LOC | Issue |
|-----|------|-----|-------|
| Enforce cloud redaction | `models/language/router.py:295` | ~10 | Fail-hard if cloud provider has no redaction_policy. |
| Temp file TOCTOU | `utils/sandbox_executor.py:553` | ~10 | Hash wrapper file content before execution. |
| API key chmod warning | `tunnel/keys.py:68` | ~5 | Log warning instead of `except OSError: pass`. |
| Pin json-repair | `pyproject.toml` | ~1 | Add upper bound: `json-repair>=0.30,<1.0`. |
| Path validation edge case | `tools/filesystem.py:84` | ~5 | Use `Path.is_relative_to()` instead of `startswith()`. |

**Ship gate:** No host paths, IPs, or stack traces in any HTTP response. Auth enforced on all endpoints when API key is configured. `shell=True` eliminated.

---

## Summary Table

| Phase | Work | LOC | Depends On | Status | Ship Gate |
|-------|------|-----|------------|--------|-----------|
| **0** | **Package Hygiene** | **~550** | **—** | **DONE** | **`pip install -e .` + `python -m maxim` work** |
| 1 | SEM Component Registry | ~300 | Phase 0 | **DONE** | String ref resolution works in campaigns |
| 1.1 | Phase 0+1 Wrap-up | ~100 | Phase 1 | **DONE** | Import-time side effects, test suite verification |
| 2 | Encounter Library | ~240 | Phase 1 | **DONE** | Template encounters load in campaigns |
| 3 | Agent Factory + Pool | ~500-700 | Phase 0 | **DONE** | 3 agents, separate memory, concurrent |
| 4 | Party DM Runtime | ~400 | Phases 1-3 | **DONE** | NPC demonstrates learned behavior |
| 5 | Hippocampus Recall | ~400 | Phase 4 | **DONE** | Behavioral recall at door succeeds |
| **6** | **Interactive Runtime + Rich Display** | **~500** | **—** | **DONE** | **Rich panels + prompt protocol + DM display** |
| 7 | Generative Architect + Entity Designer | ~600 | Phases 1,2,6 | **DONE** | Campaign + PC + 3 NPCs in <8 min |
| **8** | **API Surface Expansion** | **~400** | **Phases 1-5,6** | **DONE** | **New verbs + events + tool reg work** |
| **9** | **Deps + Docs + Cloud Profiles + Mother Pre-Pub** | **~700** | **Phase 8** | **DONE** | **Clean install + store protocols + metadata fields** |
| 10 | Publication Prep | ~2 files | — | **DONE** | CHANGELOG + CONTRIBUTING (SECURITY already exists) |
| 11 | Test PyPI + Publish | ~0 | Phases 0-10 | **In progress** | `pip install pymaxim` works |
| **12a** | **Security Hardening** | **~200** | **Phase 11** | **DONE** | **P0 fixes: shell injection, path traversal, schema validation, auth, CORS, error sanitization** |
| **12b** | **[Pre-Publication Hardening](pre_publication_hardening_plan.md)** | **~2,500** | **∥ Phase 11** | **DONE** | **Broken API fixes, error honesty, CLI UX, tests for public surface, user docs** |

**Parallelization:** Phases 0-12a done, 12b done. Phase 11 (Test PyPI) in progress. Real publish blocked on 11.

**Total LOC:** ~4,150 (Phases 0-10) + ~200 (12a security) + ~2,500 (12b hardening) = ~6,850 total pre-publication

---

## Pre-Publication Audit Findings (2026-04-08)

Deep review of the codebase surfaced these items. Each is assigned to the phase where it fits.

### Blockers (must fix before PyPI upload)

| Finding | Fix In | Details |
|---------|--------|---------|
| `mp.set_start_method("spawn", force=True)` at import time (selfy.py:10) | Phase 0b | Mutates global process state on import. Will break users with their own multiprocessing setup. Make lazy or conditional. |
| `os.environ["PYOPENGL_PLATFORM"] = "egl"` at import time (selfy.py:55) | Phase 0b | Same category — modifies global env on import. Defer to first use. |
| GPU env vars set at init (gpu_compat.py:60-75, agentic_runtime.py:79) | Phase 0b | `GST_CUDA_NO_CUDA`, `CUDA_VISIBLE_DEVICES`, `MAXIM_LLM_PROFILE` set during init without opt-out. Defer to function call. |
| `data/` directory (~1.3GB runtime artifacts) must not ship in wheel | Phase 11 | Verify with `python -m build && unzip -l dist/*.whl`. Add exclusion to MANIFEST.in or pyproject.toml if needed. |

### Warnings (should fix before publication)

| Finding | Fix In | Details |
|---------|--------|---------|
| `requires-python >= 3.12` is unnecessarily restrictive | Phase 9 | Code uses `match` (3.10+) but no 3.12-exclusive features. Lowering to `>=3.10` doubles potential user base. |
| `numpy>=2.2` cuts off numpy 1.x users | Phase 9 | No numpy 2.x-specific APIs used. Relax to `>=1.26,<3.0`. |
| `reachy-mini[gstreamer]==1.2.6` exact pin | Phase 9 | Change to `>=1.2.6,<1.3` (compatible release). |
| `torch>=2.7` has no upper bound | Phase 9 | Add `torch>=2.7,<3.0`. Same for tensorflow. |
| Missing CHANGELOG.md | Phase 10 | Already planned. |
| Missing CONTRIBUTING.md | Phase 10 | Already planned. |
| SECURITY.md already exists (11.2 KB) | Phase 10 | Remove from Phase 10 deliverables — already shipped. |

### Positive findings (no action needed)

- **Zero ruff violations** across 400 modules, 819 classes, 786 functions
- **3,199 tests** across 138 test files — comprehensive coverage
- **No TODO/FIXME/HACK comments** in codebase
- **No secrets committed** — .env properly gitignored, API keys via os.environ.get()
- **No circular imports** — extensive TYPE_CHECKING + lazy imports
- **Public API is clean** — 6 lazy-loaded verbs, proper type hints, structured returns, py.typed marker
- **Optional deps well-managed** — 65+ graceful ImportError catches, heavy imports all deferred
- **Subprocess calls properly sanitized** — all use list-form (no shell injection risk)
- **No bare `except:` blocks** — all catch `Exception` with logging
- **selfy.py is 858 LOC** (not 5,189 as previously documented — mixin decomposition already done)

---

## What Ships AFTER Publication (deferred to future_plans.md)

These are features, not architecture. Safe to add post-publication:

- **DM Extension C** — Adaptive Difficulty (ship after collecting metric data from party campaigns)
- **DM Extension D** — Encounter Isolation (ship only if state corruption observed)
- **DM Extension E** — True-Random RNG (trivial, ship anytime)
- **DM Extension F** — Encounter Merging (defer indefinitely)
- **DM Extension G** — Chained Pipeline (ship after architect is stable)
- **Capability Agent** — runtime capability awareness
- **Agent Mesh Phase 0a-0b** — mDNS discovery + InferenceRouter (need multi-machine)
- **Embodiment Hardware Adapter** — physical robot SDKs
- **Full CI/CD pipeline** — GitHub Actions for automated testing + publishing
- **Additional API verbs** — `maxim.teach()` (memory injection), `maxim.plan()` (explicit planning), as demand surfaces

---

## Ties to Existing Plans

| Existing Plan | Relationship |
|---|---|
| [DM Extensions](dungeon_master_extensions.md) | Extensions A + B pulled into Phases 2 + 7. Remainder deferred. |
| [PyPI Publication](pypi_publication_plan.md) | Phases 3-6 become Phases 9-11 here. Multi-robot plugins (Ph3 of that plan) deferred. |
| [Doctor Upgrade](doctor_upgrade_plan.md) | Unaffected. Capability Agent deferred. |
| [Tool Refinement](tool_refinement_plan.md) | Phase 8c (tool registration) supersedes the "user-defined tools" item. |
| [GitHub Fork Workflow](github_repo_management_plan.md) | Deferred to post-publication. |

---

## Architectural Invariants (New)

In addition to existing invariants from CLAUDE.md:

- **Agent instances are fully isolated.** Each agent has its own Hippocampus, NAc, ATL, MemoryHub, Executor, tool registry. No shared mutable state between agents.
- **Inter-agent communication goes through LocalMessageBus.** No direct method calls between agent instances.
- **Component Registry is read-only at runtime.** Components are templates; entity instances are created from them. Modifying a component doesn't affect running entities.
- **NPC agents use the same bio-stack as the PC.** No special "lite" memory or "fake" learning. NPCs are real Maxim agents with cheaper LLM tiers.
- **Party choice resolution is pluggable.** Default is `pc_decides`. Don't hardcode resolution logic.
- **LLM Router is shared, cost tracking is per-agent.** All agents use the same router/worker pool (expensive resource). Session cost is tracked per `AgentInstance` via agent_id tagging on requests.
- **User data lives in `~/.maxim/`**, not CWD. Bundled defaults live in `src/maxim/_data/`. Never assume the repo checkout exists.
- **Public API verbs are thin facades.** Business logic lives in the subsystems. `api.py` bootstraps, delegates, and returns structured results. Don't put domain logic in `api.py`.
- **Event subscriptions are additive.** `maxim.on()` wraps existing bus subscriptions. Don't build a parallel event system.

---

## Known Issues Tracked for Fix During Buildout

These don't have their own phases but should be fixed as encountered during the relevant phase:

| Issue | Fix During | Notes |
|-------|-----------|-------|
| `test_record_plan_outcome` failure | Phase 5 | NAc observation tracking from plan outcomes |
| `tool_stats={}` TODO in benchmark.py | Phase 8 | Propagate tool stats through ExperimentResult |
| ~~`llm-local` / `llm-llama` duplicate dep~~ | ~~Phase 0g~~ | **DONE** — removed in Phase 0 |
| ToolRegistry has no thread safety | Phase 3 | Add RLock around `_tools` dict |
| Graceful error for missing LLM in imagine() | Phase 8 | Helpful ImportError message |
| `observe()` returns `AUTIntrospector` alias | Phase 8 | Remove deprecated alias, use `Observer` only |
| Replace `dateparser` with stdlib-only parser | Phase 9 | `dateparser` pulls `regex` (C ext) + 3 deps. The regex fallback in `temporal_signal.py` already handles agent-generated patterns. Remove `dateparser` import, promote regex path to primary, delete `[temporal]` optional extra. ~30 LOC net reduction. |
| **`AutonomyLevel.FULL` crash in api.py:154** | **Phase 1.1** | **CRITICAL.** `maxim.run()` references `AutonomyLevel.FULL` which doesn't exist in the enum. Change to `AutonomyLevel.AUTONOMOUS` or add a `mode` parameter with safe default. |
| **No sandbox in `maxim.run()` API path** | **Phase 8** | CLI wires sandbox via bootstrap; Python API skips it. `maxim.run()` should default to a safe mode (PLANNING or SUPERVISED) and optionally accept `sandbox="docker"` / `sandbox="tmpdir"`. Users who want full access opt in explicitly. |
| **TmpdirSandbox symlink escape** | **Phase 9** | Agent can create symlinks inside tmpdir pointing outside. Add `os.path.realpath()` check that resolved target stays within sandbox root. Docker sandbox is unaffected. |
| **`_accessible_folders` global is mutable at runtime** | **Phase 4** | `add_accessible_folder()` has no auth gate. In multi-agent, one agent could expand access for all. Fix: per-agent folder policy, set at init, immutable after. Phase 3 builds the factory/pool — Phase 4 (party DM) is where multi-agent actually runs and exposes this. |
| **Mother Maxim M-0a: Split store protocols** | **Phase 9e** | Protocol shape is load-bearing — locks persistence interface before publication. See 9e section for details. ~80 LOC. |
| **Mother Maxim M-0b: NAc thread safety** | **Phase 9e** | NAc has no locking on `_links`, `_pending_events`, `_priors`. ~30 LOC. |
| **Mother Maxim M-0c: metadata field** | **Phase 9e** | Add `metadata: dict[str, Any]` to EpisodicMemory + SemanticMemory. ~20 LOC. Breaking post-pub. |
| **Mother Maxim M-0d: Hippocampus.sample()** | **Post-pub** | Additive method, non-breaking. Ship in 0.2.1 when Mother needs it for dream state. |
| **Mother Maxim M-0e: SCN set_wall_clock()** | **Post-pub** | Additive method, non-breaking. Ship in 0.2.1 when circadian lifecycle ships. |
| **Narrative concept extraction (lemmatization)** | **Post-pub** | Apply existing `normalize_tokens()` to freeform observation text, query ATL index. ~20 LOC. Additive, non-breaking. |
| **ProcessingState.HIBERNATE** | **Post-pub** | New enum value + HibernateTool + SEM wake triggers. ~120 LOC. Non-breaking addition. See [mother_maxim_plan.md](mother_maxim_plan.md). |
| **Narrative concept extraction via lemmatization** | **Phase 5** | `ConceptExtractor` only extracts from structured perception fields. For Mother's narrative contributions, apply existing `normalize_tokens()` (with built-in lemmatization from `memory/text.py`) to freeform observation text, then query ATL concept index. ~20 LOC. Zero deps, zero LLM. |
| **ProcessingState.HIBERNATE** | **Post-publication** | Add `HIBERNATE` to ProcessingState enum (alongside `AWAKE`/`SLEEP`). Hibernation unloads LLM from VRAM, persists bio-state, frees GPU for external tasks (training, benchmarks). Wake via SEM failure mode triggers. Not needed pre-publication but the enum should be designed to be extensible. See [mother_maxim_plan.md](mother_maxim_plan.md). |

---

## Cleanup Items Distributed Into Phases

Items discovered during Phase 0 implementation. Each has been assigned to the phase where it naturally fits:

| Item | Assigned To | Notes |
|------|------------|-------|
| Merge `gpu_compat.py` + `gpu_detect.py` (~50 LOC) | Phase 9 (Deps audit) | Consolidate during the dependency/import audit sweep |
| De-globalize `sim_logger` fully (~200 LOC) | Phase 3 (Agent Factory) | Needs per-agent loggers — refactor when the requirement is concrete |
| Duplicate path-resolution patterns (~60 LOC) | Phase 9 (Deps audit) | Consolidate remaining `_find_config()` during import audit |
| Campaign save/load for resume (~150 LOC) | Phase 4 (Party DM Runtime) | Needs AgentPool + per-agent memory export to exist first |
| Decompose `selfy.py` → `ReachyController` (~500 LOC) | Post-publication | See [future_plans.md](future_plans.md) — Embodiment Hardware Adapter |
