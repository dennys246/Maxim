# Asset Foundry Plan

**Goal:** An autonomous pipeline that generates, validates, tests, and curates SEM components — expanding the component library without manual YAML authoring while stress-testing the bio-stack against novel entity designs.

**Motivation:** Hand-writing SEM components is slow and biased toward what the author thinks of. An LLM-driven foundry generates creative sensor combinations, failure modes, and affordance patterns that humans wouldn't consider — then the simulation system ruthlessly filters for the ones that actually produce interesting cognitive behavior.

**Depends on:** EntityDesigner (shipped), ComponentRegistry + genre gating (shipped), DM campaign system (shipped), BenchmarkRunner (shipped).

**Trigger:** Post-publication. The foundry consumes LLM tokens for both generation and testing — ship the stable package first, then use the foundry to grow the library.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  maxim --foundry "cyberpunk weapons" --count 10                  │
│                                                                  │
│  ┌────────────┐   ┌────────────┐   ┌────────────┐   ┌────────┐  │
│  │  GENERATE  │──→│  VALIDATE  │──→│    TEST    │──→│ CURATE │  │
│  │            │   │            │   │            │   │        │  │
│  │ Theme +    │   │ Schema     │   │ Gauntlet   │   │ Score  │  │
│  │ genre +    │   │ Sensor     │   │ campaign   │   │ Rank   │  │
│  │ category   │   │ ranges     │   │ per entity │   │ Report │  │
│  │ → N specs  │   │ Genre tag  │   │ Bio-system │   │ Promote│  │
│  │            │   │ Affordance │   │ expectations│  │ Flag   │  │
│  └────────────┘   └────────────┘   └────────────┘   └────────┘  │
│       │                │                │                │       │
│       ▼                ▼                ▼                ▼       │
│  candidates/       rejected/        results/         report.md  │
│  *.yaml            *.yaml           *.json           scores.json │
│                                                      promoted/   │
│                                                      *.yaml      │
└──────────────────────────────────────────────────────────────────┘

Output: ~/.maxim/foundry/{run_id}/
```

---

## Phase F-0: Generation Engine (~200 LOC)

**Goal:** Batch-generate SEM component YAML specs from a theme prompt.

**What it does:**
1. Takes a theme prompt, genre, category, and count
2. Uses the EntityDesigner's LLM path to generate `N` candidate entity specs
3. Each candidate gets a unique name, proper genre tags, and category placement
4. Writes raw candidate YAMLs to `~/.maxim/foundry/{run_id}/candidates/`

**Theme prompt design:**
```
Generate a {category} component for a {genre} setting.
Theme: {theme_prompt}

Requirements:
- 3-6 sensors with realistic ranges and units
- 2-4 modulators with 2-5 affordances each
- 1-3 failure modes with sensor-based triggers
- Tags must include the genre: {genre}

Vary each component — don't repeat the same pattern.
Component {n} of {total}:
```

The prompt rotates through sub-themes to avoid repetition. For "cyberpunk weapons":
- Iteration 1: "a close-range melee weapon with an energy mechanic"
- Iteration 2: "a long-range weapon with overheating"
- Iteration 3: "a stealth weapon with limited charges"
- etc.

Sub-themes are either LLM-generated (medium tier, one call) or drawn from a seed list per genre.

**Batch efficiency:** Group generation into chunks of 3-5 per LLM call (one call produces multiple component specs in a JSON array) to reduce round-trips.

**Files:**
- **New:** `src/maxim/simulation/foundry.py` — `FoundryRunner` class, `generate_candidates()`
- **Touch:** `src/maxim/simulation/entity_designer.py` — add `design_batch()` for multi-entity generation in one call

### CLI

```bash
# Generate 10 cyberpunk weapons
maxim --foundry "cyberpunk weapons" --count 10 --genre cyberpunk --category weapons

# Generate across all categories for a theme
maxim --foundry "underwater civilization" --count 20 --genre fantasy

# Use a specific model for generation
maxim --foundry "horror creatures" --count 5 --genre horror --model claude-sonnet
```

---

## Phase F-1: Validation Pipeline (~150 LOC)

**Goal:** Reject malformed or nonsensical specs before wasting simulation cost on them.

**Validation checks (ordered by cost, cheapest first):**

| Check | What | Rejects |
|-------|------|---------|
| **Schema** | Valid YAML, has required fields (entity.name, entity_type, sensors, modulators) | Malformed specs |
| **Sensor sanity** | Ranges are ordered (min < max), initial within range, units are strings | `range: [100, 0]`, `initial: 999` |
| **Affordance sanity** | At least one affordance per modulator, params are typed | Empty modulators |
| **Failure mode sanity** | Trigger field exists in sensors, op is valid, pain in [0, 1] | Triggers referencing nonexistent sensors |
| **Genre compliance** | Genre tag present and matches the requested genre | Cross-genre leaks |
| **Duplicate detection** | Name not already in the component registry | Collisions with existing components |
| **Diversity check** | Sensor/affordance names not too similar to other candidates in this batch | 5 variants of the same weapon |

**Validation result per candidate:**
```python
@dataclass
class ValidationResult:
    candidate_path: str
    valid: bool
    errors: list[str]      # Hard failures (reject)
    warnings: list[str]    # Soft issues (test anyway)
    diversity_score: float  # 0-1, how different from other candidates
```

**Files:**
- **New:** `src/maxim/simulation/foundry_validation.py` — `validate_candidate()`, `validate_batch()`
- **Touch:** `src/maxim/embodiment/spec.py` — extract schema validation into a reusable `validate_entity_spec()` function

---

## Phase F-2: Test Gauntlet (~250 LOC)

**Goal:** Run each validated candidate through a campaign that exercises the bio-stack and measures engagement.

### Gauntlet Campaign Design

The foundry generates a **per-candidate micro-campaign** (3 encounters, ~30s each) that specifically targets the candidate's capabilities:

**Encounter 1 — Discovery:**
Scene introduces the entity. Agent observes its sensors and affordances.
Tests: Hippocampus captures the entity. Novelty is high. SalienceNetwork tracks it.

**Encounter 2 — Interaction:**
Agent must use the entity's affordances in a goal-directed context.
Tests: Cerebellum forms forward models. NAc learns action→outcome links. Motor programs may crystallize.

**Encounter 3 — Stress:**
Push the entity toward its failure modes. Sensor values shift toward trigger thresholds.
Tests: Pain signals fire. Failure modes activate. Agent adapts behavior in response.

### Gauntlet generation

The micro-campaign is auto-generated from the candidate spec:

```python
def generate_gauntlet(candidate: dict) -> CampaignDef:
    """Generate a 3-encounter test campaign for a single SEM component.
    
    Reads the candidate's sensors, affordances, and failure modes to
    create scenes that exercise each one. No LLM call needed — the
    campaign is structural, derived directly from the spec.
    """
```

Scene text is template-based with entity-specific fill:
- "You discover a {entity.name}. It has {sensor_list}. You can {affordance_list}."
- "Use the {entity.name} to {goal}. Watch how it responds."
- "The {entity.name} is degrading. {failure_approaching}. What do you do?"

### Bio-system expectations

Auto-generated from the spec:
- Hippocampus: `min_episodic_captures: 3` (one per encounter)
- NAc: `min_observations: 2` (interaction + stress outcomes)
- Cerebellum: `min_forward_models: 1` (learned from interaction)
- Pain: `min_signals: 1` if the entity has failure modes

### Execution

```python
for candidate in validated_candidates:
    gauntlet = generate_gauntlet(candidate)
    result = run_campaign(gauntlet, model=test_model)
    scores[candidate.name] = score_result(result, candidate)
```

Uses the cheapest available LLM (Mistral-7b or smollm) for testing — the point isn't quality reasoning, it's whether the entity spec produces bio-system engagement.

**Files:**
- **New:** `src/maxim/simulation/foundry_gauntlet.py` — `generate_gauntlet()`, `run_gauntlet()`, `score_result()`
- **New:** `src/maxim/simulation/foundry_sem_tests.py` — `run_sem_protocol_tests()` (see below)
- **Touch:** `src/maxim/simulation/dm_schema.py` — ensure programmatic CampaignDef construction works (no YAML file needed)

### SEM Protocol Test Suite

Before the gauntlet campaign runs (which tests bio-system engagement), every candidate goes through a **structural SEM protocol test** that verifies the component actually works as an entity in the runtime. These are fast (no LLM, no simulation) and catch spec issues the schema validator can't.

**Test 1 — Instantiation:**
```python
entity = registry.instantiate(candidate_ref)
assert entity is not None
assert entity.name == expected_name
assert entity.entity_type in {"npc", "creature", "weapon", "body_part", "environment"}
```
Verifies `_parse_entity()` succeeds and produces a valid Entity with the right type.

**Test 2 — Sensor Read/Write Cycle:**
```python
for sensor_name, sensor in entity.sensors.items():
    reading = sensor.read()
    assert reading.value is not None
    # Value is within declared range
    spec_range = candidate_spec["sensors"][sensor_name].get("range", [0, 1])
    assert spec_range[0] <= reading.value <= spec_range[1]
    # Write a new value and verify
    entity.vital_metrics[sensor_name] = (spec_range[0] + spec_range[1]) / 2
    new_reading = sensor.read()
    assert new_reading.value == pytest.approx((spec_range[0] + spec_range[1]) / 2)
```
Verifies every sensor initializes within range, reads correctly, and responds to vital_metrics writes (which is how cascades modify sensors).

**Test 3 — Modulator Affordance Enumeration:**
```python
affordances = {}
for mod_name, mod in entity_spec.get("modulators", {}).items():
    for aff_name, aff in mod.get("affordances", {}).items():
        affordances[aff_name] = aff
        # Verify params are typed
        for param_name, param_type in aff.get("params", {}).items():
            assert param_type in {"str", "float", "int", "bool"}
        # Verify description exists
        assert aff.get("description", "").strip() != ""
assert len(affordances) >= 1  # At least one affordance
```
Verifies all affordances have descriptions (LLM needs these) and typed parameters.

**Test 4 — Tool Generation:**
```python
from maxim.embodiment.tool_bridge import generate_tools_for_entity
tools = generate_tools_for_entity(entity, tool_registry=None)
assert len(tools) >= 1
# Verify no name collisions
tool_names = [t.name for t in tools]
assert len(tool_names) == len(set(tool_names))
```
Verifies `tool_bridge` can auto-generate tools from the entity spec without errors or name collisions.

**Test 5 — Failure Mode Triggers:**
```python
for failure in candidate_spec.get("failure_modes", []):
    trigger = failure["trigger"]
    field = trigger["field"]
    assert field in entity.sensors, f"Failure trigger references missing sensor: {field}"
    op = trigger["op"]
    assert op in ("<", "<=", ">", ">=", "==", "!=")
    pain = trigger.get("pain", 0)
    assert 0 <= pain <= 1
    # Simulate the trigger condition
    entity.vital_metrics[field] = trigger["value"]
    # Verify the entity can evaluate this (structural, not runtime)
```
Verifies failure mode triggers reference real sensors with valid operators and pain intensities.

**Test 6 — Composition (parent-child):**
```python
# Only for body_part and weapon types — they should be attachable
if entity.entity_type in ("body_part", "weapon"):
    parent = Entity(name="test_host", entity_type="character")
    entity.reparent(parent)
    assert entity.parent is parent
    assert entity in parent.children
    entity.detach()
    assert entity.parent is None
```
Verifies body parts and weapons can attach/detach from parent entities (needed for the `swap_entity` system).

**Test 7 — Cascade Compatibility:**
```python
from maxim.simulation.dm_runtime import CascadeResolver
resolver = CascadeResolver({"test_entity": entity})
# Verify sensors are readable via the resolver's ref path system
for sensor_name in entity.sensors:
    val = resolver._read_sensor(f"test_entity.{sensor_name}", {})
    assert val is not None
```
Verifies the entity's sensors are accessible through the CascadeResolver's path system (used by DM runtime for damage, healing, state changes).

**Test 8 — Inheritance (if extends is set):**
```python
if candidate_component.get("extends"):
    parent_ref = candidate_component["extends"]
    parent_spec = registry.get(parent_ref)
    # Verify child has all parent sensors (at minimum)
    parent_sensors = set(parent_spec.get("entity", {}).get("sensors", {}).keys())
    child_sensors = set(candidate_spec.get("sensors", {}).keys())
    assert parent_sensors.issubset(child_sensors), (
        f"Child missing parent sensors: {parent_sensors - child_sensors}"
    )
```
Verifies components that use `extends` actually inherit all parent sensors.

**Protocol test result:**
```python
@dataclass
class SEMProtocolResult:
    candidate_name: str
    tests_passed: int
    tests_total: int
    failures: list[str]    # Which tests failed and why
    warnings: list[str]    # Non-fatal issues
    
    @property
    def passed(self) -> bool:
        return self.tests_passed == self.tests_total
```

**Integration with the pipeline:** SEM protocol tests run between validation (F-1) and the gauntlet campaign (F-2). Candidates that fail protocol tests are rejected without spending simulation cost. The gauntlet then only runs against structurally sound entities.

```
Generate → Validate (schema) → SEM Protocol Tests → Gauntlet (bio-system) → Score → Curate
                                      ↓
                              Reject structurally
                              broken components
                              before sim cost
```

---

## Phase F-3: Scoring + Curation (~200 LOC)

**Goal:** Rank candidates by bio-system engagement and sort into promote/review/reject buckets.

### Scoring Rubric

| Dimension | Weight | How Measured |
|-----------|--------|-------------|
| **Schema quality** | 10% | Validation pass, no warnings |
| **Hippocampal engagement** | 20% | Episodic captures / encounters |
| **Causal learning** | 20% | NAc observations + link confidence |
| **Cerebellum learning** | 15% | Forward models formed, prediction accuracy |
| **Pain/failure activation** | 15% | Pain signals published, failure modes triggered |
| **Novelty impact** | 10% | Salience decay observed (confirms entity is tracked) |
| **Diversity** | 10% | How different from existing library components |

**Total score in [0, 1]. Thresholds:**
- **Promote** (> 0.7): Auto-copied to `promoted/` directory, ready for human review and commit
- **Review** (0.4 - 0.7): Interesting but flawed — flagged with specific issues in the report
- **Reject** (< 0.4): Inert or broken — logged but not kept

### Interesting Failures

Some candidates will fail expectations in novel ways — these are potentially more valuable than clean passes:

- Entity that causes unexpected NAc causal links → might reveal bio-stack edge case
- Entity whose failure mode triggers a cascade the agent can't recover from → stress test for resilience
- Entity whose affordances are used in unexpected combinations → emergent behavior

Flag these separately in the report as "interesting failures" with the specific unexpected behavior noted.

### Report

```markdown
# Foundry Report — cyberpunk weapons
Generated: 10 | Validated: 8 | Tested: 8 | Promoted: 5 | Review: 2 | Rejected: 1

## Promoted
| Component | Score | Highlights |
|-----------|-------|-----------|
| plasma_cutter | 0.85 | 3 forward models, pain cascade from overheat |
| mono_wire | 0.78 | NAc learned stealth→success, novel failure mode |
| ...

## Flagged for Review
| Component | Score | Issue |
|-----------|-------|-------|
| gravity_hammer | 0.55 | Cerebellum never formed predictions (affordances too abstract?) |
| ...

## Interesting Failures
- sonic_disruptor: NAc formed a link between "use at close range" and "self-damage"
  that doesn't match the spec's failure modes — possible bio-stack edge case

## Rejected
| Component | Reason |
|-----------|--------|
| quantum_blade | Zero bio-system engagement — all sensors static |
```

**Files:**
- **New:** `src/maxim/simulation/foundry_scoring.py` — `score_result()`, `curate_batch()`, `generate_report()`
- Output format: `report.md` (human-readable) + `scores.json` (machine-readable)

---

## Phase F-4: Thematic Campaign Templates (~150 LOC)

**Goal:** Pre-built theme configurations so you can run `maxim --foundry "cyberpunk"` without specifying every parameter.

### Theme Templates

```yaml
# ~/.maxim/foundry/themes/cyberpunk.yaml (or bundled in _data/foundry/)
theme:
  name: cyberpunk
  genre: cyberpunk
  description: "Neon-soaked urban dystopia with augmented humans and autonomous machines"
  
  categories:
    weapons:
      count: 5
      sub_themes:
        - "close-range melee with energy mechanic"
        - "long-range with overheating or charge cells"
        - "stealth weapon with limited uses"
        - "non-lethal crowd control"
        - "experimental military prototype"
    creatures:
      count: 3
      sub_themes:
        - "autonomous security machine"
        - "augmented animal"
        - "rogue AI construct"
    npcs:
      count: 4
      sub_themes:
        - "street-level criminal or fixer"
        - "corporate employee or executive"
        - "underground hacker or netrunner"
        - "augmented mercenary"
    environments:
      count: 3
      sub_themes:
        - "dangerous urban exterior"
        - "high-security interior"
        - "underground or hidden location"
    bodies:
      count: 2
      sub_themes:
        - "combat augmentation"
        - "stealth or infiltration augmentation"

  test_model: mistral-7b    # Cheapest model for gauntlet testing
  generate_model: null       # Use default (medium tier)
```

### CLI with themes

```bash
# Run a full themed foundry pass
maxim --foundry cyberpunk              # Uses theme template
maxim --foundry cyberpunk --count 30   # Override total count (distributed across categories)

# Custom theme from YAML
maxim --foundry themes/my_underwater_civ.yaml

# Just generate + validate (skip expensive testing)
maxim --foundry cyberpunk --dry-run

# Re-test previously generated candidates
maxim --foundry --retest ~/.maxim/foundry/20260409_cyberpunk/candidates/
```

**Files:**
- **New:** `src/maxim/_data/foundry/` — bundled theme templates (cyberpunk.yaml, fantasy.yaml, scifi.yaml, horror.yaml)
- **Touch:** `src/maxim/simulation/foundry.py` — add theme loading, CLI integration

---

## Phase F-5: Integration + Polish (~150 LOC)

**Goal:** Wire the foundry into the CLI, add session persistence, and enable incremental runs.

### Session Persistence

Each foundry run produces:
```
~/.maxim/foundry/{run_id}/
├── config.yaml          # Theme, params, model, timestamp
├── candidates/          # Raw generated YAML specs
│   ├── plasma_cutter.yaml
│   ├── mono_wire.yaml
│   └── ...
├── rejected/            # Failed validation
│   └── quantum_blade.yaml
├── results/             # Per-candidate gauntlet results
│   ├── plasma_cutter.json
│   └── ...
├── promoted/            # Top scorers (ready for review + commit)
│   ├── plasma_cutter.yaml
│   └── ...
├── scores.json          # Machine-readable scoring data
└── report.md            # Human-readable summary
```

### Incremental Runs

```bash
# Resume a previous run (re-test failed candidates, test new ones)
maxim --foundry --resume 20260409_cyberpunk

# Add more candidates to an existing run
maxim --foundry cyberpunk --resume 20260409_cyberpunk --count 5

# Promote a candidate from review to the component library
maxim --foundry --promote 20260409_cyberpunk/candidates/gravity_hammer.yaml
```

`--promote` copies the YAML to `src/maxim/_data/components/{category}/` with proper formatting and prints a reminder to commit.

### CLI Help

```
maxim --foundry <theme|yaml> [options]

Options:
  --count N         Total components to generate (default: from theme)
  --genre GENRE     Override genre tag
  --category CAT    Generate only this category
  --model MODEL     LLM for generation (default: medium tier)
  --test-model M    LLM for gauntlet testing (default: small tier)
  --dry-run         Generate + validate only, skip testing
  --resume RUN_ID   Continue a previous foundry run
  --retest DIR      Re-run gauntlet on existing candidates
  --promote PATH    Copy a candidate to the component library
```

**Files:**
- **Touch:** `src/maxim/cli.py` — add `--foundry` argument handling
- **Touch:** `src/maxim/simulation/foundry.py` — session persistence, resume logic

---

## Cost Model

| Operation | Model | Tokens/call | Calls/component | Cost/component |
|-----------|-------|------------|-----------------|---------------|
| Generate | Medium (Mistral-7b) | ~500 | 0.3 (batched 3/call) | ~$0.00 (local) |
| Generate | Medium (Claude) | ~500 | 0.3 | ~$0.005 |
| Gauntlet (3 encounters) | Small (smollm) | ~300/turn × 6 turns | 1 | ~$0.00 (local) |
| Gauntlet (3 encounters) | Small (Mistral-7b) | ~300/turn × 6 turns | 1 | ~$0.00 (local) |
| Gauntlet (3 encounters) | Medium (Claude) | ~300/turn × 6 turns | 1 | ~$0.01 |

**Typical run (local models):** 20 components × (generate + validate + test) ≈ 2-5 minutes, $0.00
**Typical run (Claude):** 20 components ≈ 5-10 minutes, ~$0.30

Local models are the default for foundry work — it's about volume and diversity, not reasoning quality.

---

## Invariants

- **Foundry never auto-commits to the component library.** `--promote` copies files, but the human must `git add` and commit.
- **Generated components must pass the same validation as hand-written ones.** No special treatment.
- **The gauntlet campaign is deterministic for a given spec.** Same candidate → same campaign → reproducible results (seeded RNG).
- **Genre gating applies during generation.** The theme's genre tag is enforced at every step.
- **Foundry runs don't require the full Maxim runtime.** Generation needs only the LLM router. Testing needs the sim orchestrator but not the interactive display.
- **No new dependencies.** Foundry uses existing EntityDesigner, ComponentRegistry, DM runtime, and BenchmarkRunner.

---

## Phase Summary

| Phase | Work | LOC | What it enables |
|-------|------|-----|----------------|
| F-0 | Generation engine + batch design | ~200 | `maxim --foundry "theme" --count N` produces candidate YAMLs |
| F-1 | Validation pipeline | ~150 | Rejects malformed/nonsensical/cross-genre specs before testing |
| F-2 | Test gauntlet (auto-generated micro-campaigns) | ~250 | Each candidate gets a 3-encounter bio-stack stress test |
| F-3 | Scoring + curation + reports | ~200 | Rank, promote, flag interesting failures |
| F-4 | Theme templates + theme CLI | ~150 | `maxim --foundry cyberpunk` with pre-built category distributions |
| F-5 | CLI integration + session persistence + incremental runs | ~150 | Resume, re-test, promote workflow |
| **Total** | | **~1,100** | Autonomous SEM component generation pipeline |

## Testing Strategy

- **F-0:** Test batch generation with mocked LLM. Verify output YAML is parseable.
- **F-1:** Test each validation check in isolation with crafted good/bad specs.
- **F-2:** Test gauntlet generation from a known spec. Verify campaign structure (3 encounters, expectations present). Run gauntlet with mock LLM and verify scoring inputs.
- **F-3:** Test scoring rubric with crafted results (perfect score, zero score, edge cases). Test promote/review/reject bucketing.
- **F-4:** Test theme loading from YAML. Test category distribution.
- **F-5:** Test session persistence (write + resume). Test `--promote` file copy.
