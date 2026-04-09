# Asset Foundry Plan

**Goal:** An autonomous pipeline that generates, validates, tests, and curates SEM components — expanding the component library without manual YAML authoring while stress-testing the bio-stack against novel entity designs.

**Motivation:** Hand-writing SEM components is slow and biased toward what the author thinks of. An LLM-driven foundry generates creative sensor combinations, failure modes, and affordance patterns that humans wouldn't consider — then the simulation system ruthlessly filters for the ones that actually produce interesting cognitive behavior.

**Depends on:** EntityDesigner (shipped), ComponentRegistry + genre gating (shipped), DM campaign system (shipped), BenchmarkRunner (shipped).

**Trigger:** Post-publication. The foundry consumes LLM tokens for both generation and testing — ship the stable package first, then use the foundry to grow the library.

---

## Architecture

```
┌────────────────────────────────────────────────────────────────────────────┐
│  maxim --foundry "cyberpunk weapons" --count 10                            │
│                                                                            │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────────┐  │
│  │ GENERATE │→ │ VALIDATE │→ │SEM PROTO │→ │ GAUNTLET │→ │SCORE+CURATE │  │
│  │          │  │          │  │  TESTS   │  │          │  │             │  │
│  │ Theme +  │  │ Schema   │  │ 8 struct │  │ 3-enc    │  │ 11-dim      │  │
│  │ genre +  │  │ Semantic │  │ tests per│  │ campaign │  │ rubric      │  │
│  │ batch    │  │ EC dedup │  │ candidate│  │ per cand │  │ promote/    │  │
│  │ JSON fix │  │ Genre    │  │ (no LLM) │  │ fresh    │  │ review/     │  │
│  │ → N YAML │  │          │  │          │  │ MemoryHub│  │ reject      │  │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘  └─────────────┘  │
│       │              │             │              │              │          │
│       ▼              ▼             ▼              ▼              ▼          │
│  candidates/     rejected/    rejected/      results/       report.md      │
│  *.yaml          *.yaml       *.yaml         *.json         scores.json    │
│                  (schema)     (structural)                  promoted/       │
│                                                             *.yaml         │
│                                                                            │
│  Run-level state (persists across candidates):                             │
│  ├── ec_index.json  — cross-candidate EC similarity for diversity checks   │
│  └── tool_names.json — accumulated tool names for collision detection      │
└────────────────────────────────────────────────────────────────────────────┘

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

**JSON repair:** Small LLMs (Mistral-7b) produce malformed JSON ~5% of the time (missing closing braces, trailing commas). Every batch response goes through the existing `json-repair` library pipeline (`models/language/json_parser.py`) before schema validation. If repair fails, the batch is re-requested with a simpler prompt (one spec at a time). If the single-spec call also fails repair after 2 retries, that candidate slot is logged as `generation_failure` and skipped — no infinite retry.

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
| **Semantic sanity** | Sensor ranges within reasonable bounds (warn if range span > 10,000), at least one failure mode triggerable within nominal operation, no contradictory failure modes | `range: [0, 1000000]`, untriggerable failures |
| **Diversity check** | EC similarity distance from existing library + other candidates in batch | 5 variants of the same weapon, near-duplicates of existing components |

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

## Phase F-2: Test Gauntlet (~500 LOC)

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

Uses the cheapest available LLM (Mistral-7b or smollm) for testing — the point isn't quality reasoning, it's whether the entity spec produces bio-system engagement. See the isolated execution loop in "Gauntlet stochasticity and isolation" below.

### Gauntlet stochasticity and isolation

**Gauntlets are stochastic, not deterministic.** The campaign seed controls dice rolls and encounter order, but LLM responses vary between runs. The same entity spec can produce different bio-system engagement on different runs. To handle this:

1. **Adaptive rerun policy (default):** Run 1 gauntlet. If the score falls within a threshold band (0.55-0.85 — near promote/review boundary), automatically trigger a second run. If the two runs diverge significantly (stddev > 0.25 on any stochastic dimension), trigger a third run. This minimizes cost for clear promotes and clear rejects.
2. **Forced multi-run:** `--runs 3` overrides adaptive policy and always runs 3 gauntlets per candidate.
3. **Deterministic fallback:** `--deterministic` flag uses a mock LLM that always picks the first choice and calls the first affordance. Useful for structural validation (does the entity produce any engagement at all?) but misses emergent behavior.

**Stochastic vs deterministic scoring dimensions:**
Not all scoring dimensions benefit from multi-run averaging. Some are structural and produce identical results on every run:

| Dimension | Type | Why |
|-----------|------|-----|
| Schema quality | Deterministic | Same spec → same validation result |
| EC indexing | Deterministic | Structural signature, not LLM-dependent |
| Temporal coverage | Deterministic | SCN bins from fixed encounter times |
| Diversity | Deterministic | EC distance from library, not LLM-dependent |
| Hippocampal engagement | **Stochastic** | LLM response affects what gets captured |
| Causal learning | **Stochastic** | LLM choices affect NAc observation count |
| Cerebellum learning | **Stochastic** | Forward models depend on action sequences |
| Motor program discovery | **Stochastic** | Crystallization depends on repeated use |
| Pain/failure activation | **Stochastic** | Whether agent triggers failure modes |
| Salience tracking | **Stochastic** | Novelty decay depends on LLM attention |
| ATL concept grounding | **Stochastic** | Concept extraction depends on captured memories |

Deterministic dimensions are measured once (first run only). Stochastic dimensions are averaged across runs. The final score formula weights both categories: `score = Σ(deterministic_i × weight_i) + Σ(mean(stochastic_i) × weight_i)`. Variance constant: `VARIANCE_THRESHOLD = 0.25` (stddev across runs for any stochastic dimension).

**MemoryHub state isolation:** Each gauntlet gets a **fresh MemoryHub** instance (new Hippocampus, ATL, EC, NAc, SCN, AngularGyrus, Cerebellum). This prevents cross-contamination — memories from gauntlet 1 must not appear in gauntlet 2's recall.

**Run-level EC for cross-candidate diversity:** A separate `EntorhinalCortex` instance persists across the entire foundry run. After each gauntlet scores, the candidate's `SituationSignature` is copied into the run-level EC. The F-1 diversity check queries this run-level EC (not the per-gauntlet EC) to detect near-duplicates. Persisted to `~/.maxim/foundry/{run_id}/ec_index.json`.

```python
# Run-level state (persists across all candidates)
run_ec = EntorhinalCortex()       # Cross-candidate diversity tracking
batch_tool_names: set[str] = set() # Cross-entity tool collision tracking

for candidate in validated_candidates:
    try:
        # Fresh bio-stack per candidate — zero state leakage
        hub = MemoryHub(
            hippocampus=Hippocampus(),
            nac=NAc(),
            scn=SCN(),
            ec=EntorhinalCortex(),   # Per-gauntlet EC (isolated)
            atl=ATL(),
            angular_gyrus=AngularGyrus(),  # Needed for concept grounding
            cerebellum=Cerebellum(),       # Needed for motor program scoring
        )
        # __post_init__ auto-calls _wire_multi_layer() when atl is not None

        gauntlet = generate_gauntlet(candidate)
        result = run_gauntlet(gauntlet, hub=hub, model=test_model)

        # Copy signature to run-level EC for cross-candidate diversity
        if result.situation_signature:
            run_ec.register(candidate.name, result.situation_signature)

        hub.on_session_end()
        scores[candidate.name] = score_result(result, candidate)

    except Exception as e:
        # Distinguish infra failures from candidate failures
        if is_infra_failure(e):  # Setup/teardown crash, import error, OOM
            log.error("Infra failure for %s: %s — retrying once", candidate.name, e)
            # Retry once; if it fails again, mark as infra_error (not scored)
            scores[candidate.name] = InfraError(candidate.name, str(e))
        else:
            # Candidate-specific failure (encounter crash, assertion, timeout)
            scores[candidate.name] = score_partial(result_so_far, candidate)
```

**Concurrency:** Gauntlets run **sequentially by default** — DMRuntime and MemoryHub are not thread-safe. For parallel execution, use `--parallel N` which uses `concurrent.futures.ProcessPoolExecutor`:

- F-1 validation (including EC diversity check) runs **sequentially first** — it needs the run-level EC
- Gauntlet processes are then spawned in parallel, each with its own isolated MemoryHub
- Results are collected via the shared `results/` filesystem directory
- If a process crashes, others continue; the failed candidate is logged as `infra_error`
- After all processes complete, run-level EC is updated with all signatures
- `--resume` works with `--parallel` — it scans `results/` for missing final JSONs and re-spawns only those

### Error recovery and partial results

Crashes happen — LLM timeouts, OOM, specs that trigger unhandled edge cases. The foundry distinguishes **infra failures** (not the candidate's fault) from **candidate failures** (the spec is broken/adversarial):

**Infra failures** (MemoryHub setup crash, import error, OOM, network timeout):
1. Retry once with a fresh MemoryHub
2. If retry fails, mark as `"status": "infra_error"` — not scored, not counted against the candidate
3. Reported separately in the foundry report as infrastructure issues

**Candidate failures** (encounter crash, assertion error in bio-stack, spec-triggered edge case):
1. **Per-encounter checkpointing:** After each encounter completes, save a checkpoint to `results/{name}.partial.json` with encounters completed, bio-system state, and partial scores
2. Save the partial result (encounters completed so far)
3. Score the partial result with a penalty (incomplete encounters score 0 for their stochastic dimensions)
4. Move to the next candidate (don't abort the foundry run)

**SEM protocol test crashes:** Each of the 8 protocol tests is wrapped in its own try/except. A crash (as opposed to an assertion failure) counts as a test failure for that test. If tests 1-3 all crash (instantiation, sensor R/W, affordance enumeration), the candidate is rejected — it's fundamentally broken. If only tests 6-8 crash (composition, cascade, inheritance), the candidate proceeds to the gauntlet with the crash noted as a warning.

**Resume:** `--resume` re-runs only candidates without a final `results/{name}.json`. Candidates with partial results or infra errors get re-run from scratch. `--resume` works with `--parallel`.

### Full-stack integration during gauntlet

The gauntlet doesn't just run a campaign — it exercises every system that a real campaign would touch. The following are **required integration points** in the gauntlet runner, not optional:

**Narrative Transcriber → Salience pipeline:**
Gauntlet scene text is routed through `NarrativeTranscriber.transcribe_to_items()` to produce `SalienceItem` objects with `NarrativeWhere` coordinates. These feed into `SalienceNetwork.update()` so the salience system tracks the generated entity properly. The gauntlet verifies:
- Entity appears in `SalienceNetwork.get_top_salient()` after Encounter 1
- Novelty decays between Encounter 1 and Encounter 3 (same entity seen multiple times)
- `to_context_str()` uses `NarrativeWhere.region()` not pixel coordinates

**ATL concept extraction:**
The gauntlet runs with a full MemoryHub (Hippocampus + ATL + EC + NAc + SCN wired). When Hippocampus captures a memory containing the generated entity, ConceptExtractor fires and should:
- Create (or find) an ATL concept for the entity's label and category
- Form `INSTANCE_OF` edges between the episodic memory and the concept
- If the entity has `extends` (e.g., extends `base_humanoid`), verify the parent concept also exists

Gauntlet scoring tracks: `atl_concepts_created: int` — how many new concepts this entity introduced.

**EC similarity indexing:**
Gauntlet results are indexed in the EntorhinalCortex so that:
- Future foundry runs can query EC for "have we already tested something similar?"
- The diversity check in F-1 can use EC similarity (not just string matching) to detect near-duplicates
- `SituationSignature` is computed from the gauntlet's structural/temporal/semantic context

This prevents the foundry from generating 50 slight variations of the same weapon.

**Motor program discovery:**
For entity types with multiple affordances (weapons, body_parts), the gauntlet's Encounter 2 should present a sequence of related affordance uses. The scoring rubric tracks:
- `motor_programs_formed: int` — did the Cerebellum crystallize any motor programs from repeated affordance sequences?
- `motor_program_steps: int` — how many steps in the longest program?

Motor programs indicate the entity's affordances compose well — a sign of good design.

**SCN temporal context:**
Each gauntlet encounter runs at a different simulated time (morning, afternoon, night) so that temporal bins are populated. This validates that generated entities don't break SCN indexing and provides richer temporal context for EC similarity queries.

**Reveal conditions:**
If the generated entity has `metadata.visibility` or `metadata.reveal_when` fields (from LLM generation), the gauntlet should test whether `evaluate_reveal_conditions()` correctly reveals hidden sensors/affordances when conditions are met. This validates the full contextual visibility system.

**DM runtime entity lifecycle:**
The gauntlet exercises the complete entity lifecycle:
1. Entity registered in `_entity_registry` at campaign start
2. `SceneState.enter_encounter()` registers/deregisters entity tools per encounter
3. Live sensor state appears in `_format_entity_state()` stimulus
4. If entity has failure modes, `CascadeResolver` drives sensor values toward triggers
5. `swap_entity` tested for body_part types (detach old, instantiate new from same spec)

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

**Test 4 — Tool Generation + Cross-Entity Collision Check:**
```python
from maxim.embodiment.tool_bridge import generate_tools_for_entity
tools = generate_tools_for_entity(entity, tool_registry=None)
assert len(tools) >= 1
# Verify no name collisions within this entity
tool_names = [t.name for t in tools]
assert len(tool_names) == len(set(tool_names))
# Cross-entity collision check (accumulated across batch):
# All tool names from this entity must not collide with tools from
# previously tested entities in this foundry run.
for name in tool_names:
    assert name not in _batch_tool_names, f"Tool name collision: {name}"
_batch_tool_names.update(tool_names)
```
Verifies `tool_bridge` can auto-generate tools from the entity spec without errors or name collisions — both within the entity and across all entities in the foundry batch.

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

| Dimension | Weight | Averaging | How Measured |
|-----------|--------|-----------|-------------|
| **Schema quality** | 5% | Single | Validation pass, no warnings |
| **Hippocampal engagement** | 15% | Per-run avg | Episodic captures / encounters |
| **Causal learning** | 15% | Per-run avg | NAc observations + link confidence |
| **Cerebellum learning** | 10% | Per-run avg | Forward models formed, prediction accuracy |
| **Motor program discovery** | 5% | Per-run avg | Motor programs crystallized from affordance sequences |
| **Pain/failure activation** | 10% | Per-run avg | Pain signals published, failure modes triggered |
| **Salience tracking** | 10% | Per-run avg | Entity tracked in SalienceNetwork, novelty decay observed, NarrativeWhere region populated |
| **ATL concept grounding** | 10% | Per-run avg | Concepts created in ATL, INSTANCE_OF edges formed, parent concept inheritance |
| **EC indexing** | 5% | Single | SituationSignature computed and indexed, no duplicate collision |
| **Temporal coverage** | 5% | Single | SCN bins populated across encounters (morning/afternoon/night) |
| **Diversity** | 10% | Single | EC similarity distance from run-level EC + existing library |

**Score formula:** `score = Σ(single_i × weight_i) + Σ(mean(per_run_i) × weight_i)`. Single dimensions (25% total weight) measured on first run only. Per-run dimensions (75% total weight) averaged across all gauntlet runs.

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

`--promote` copies the YAML to `~/.maxim/components/{category}/` (user search path, always writable). The ComponentRegistry already discovers components from this path. For package maintainers working in the source tree, `--promote --dev` copies to `src/maxim/_data/components/{category}/` instead and prints a reminder to commit.

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

## Phase F-6: Downstream Integration (~200 LOC)

**Goal:** Feed foundry outputs back into the broader system — encounter library, generative campaigns, and interactive curation.

### Encounter Library Archival

Each gauntlet micro-campaign is a valid 3-encounter campaign. The best ones (from promoted components) should be archived as reusable encounter templates:

```python
def archive_gauntlet_encounters(
    candidate: dict,
    gauntlet: CampaignDef,
    score: float,
) -> None:
    """Archive a successful gauntlet as encounter templates.
    
    Writes to ~/.maxim/encounters/foundry/{genre}/{category}/
    with metadata linking back to the source component.
    """
```

This means a foundry run producing 5 promoted weapons also produces 5 reusable "weapon discovery" encounter templates that other campaigns can reference via the EncounterLibrary.

### Generative Campaign Feeding

Promoted components should be discoverable by the generative campaign system (narrator + arcs). When the narrator generates encounters for a genre, it should:

1. Query `ComponentRegistry` for genre-matching components
2. Use promoted foundry components as entity templates for generated NPCs/objects
3. Reference the component's gauntlet results to inform encounter difficulty (a weapon with high pain activation = dangerous encounter)

**Integration point:** `simulation/narrator.py` — add optional `registry: ComponentRegistry` parameter. When generating an entity description, check if a matching component exists and use it as a base.

### Interactive Curation Mode

For the "review" bucket (score 0.4-0.7), provide an interactive curation workflow:

```bash
maxim --foundry --curate 20260409_cyberpunk
```

This launches a PromptRequest-based review session:

1. For each "review" candidate, show: component YAML, gauntlet results, scoring breakdown
2. Ask: **Promote** (copy to library), **Edit** (open YAML for manual fixes, re-test), **Reject** (discard)
3. Edited components get re-run through the SEM protocol tests + gauntlet
4. Uses the existing `PromptHandler` protocol — works in terminal (rich display) or programmatically

**Files:**
- **Touch:** `src/maxim/simulation/foundry.py` — `curate_interactive()`
- **Touch:** `src/maxim/simulation/encounter_library.py` — `register_from_foundry()`
- **Touch:** `src/maxim/simulation/narrator.py` — component-aware entity generation

### Benchmark Scenario Generation

Promoted components can auto-generate benchmark scenarios. A foundry run that produces 5 cyberpunk weapons and 3 cyberpunk NPCs can generate a `cyberpunk_combat_suite.yaml` benchmark:

```yaml
# Auto-generated from foundry run 20260409_cyberpunk
suite:
  name: cyberpunk_combat
  genre: cyberpunk
  scenarios:
    - name: plasma_cutter_stress
      source_component: weapons/plasma_cutter
      encounters: 3
      expectations:
        cerebellum: { min_forward_models: 2 }
        pain: { min_signals: 1 }
    # ...
```

This closes the loop: foundry generates components → components become benchmark scenarios → benchmarks validate model performance with those components.

---

## Cost Model

| Operation | Model | Tokens/call | Calls/component | Cost/component |
|-----------|-------|------------|-----------------|---------------|
| Generate | Medium (Mistral-7b) | ~500 | 0.3 (batched 3/call) | ~$0.00 (local) |
| Generate | Medium (Claude) | ~500 | 0.3 | ~$0.005 |
| Gauntlet (3 encounters, 2-3 runs) | Small (smollm) | ~300 tok × 3-6 calls × 2-3 runs | 1 | ~$0.00 (local) |
| Gauntlet (3 encounters, 2-3 runs) | Small (Mistral-7b) | ~300 tok × 3-6 calls × 2-3 runs | 1 | ~$0.00 (local) |
| Gauntlet (3 encounters, 2-3 runs) | Medium (Claude) | ~300 tok × 3-6 calls × 2-3 runs | 1 | ~$0.02-0.03 |

Per encounter: 1 LLM call for AUT response + 0-1 for choice classification fallback = 1-2 calls. 3 encounters × 1-2 calls = 3-6 calls per gauntlet run. With 2-3 runs for averaging: 6-18 calls per candidate.

**Typical run (local models):** 20 components × (generate + validate + 2 gauntlet runs) ≈ 5-10 minutes, $0.00
**Typical run (Claude):** 20 components ≈ 10-20 minutes, ~$0.50-0.70

Local models are the default for foundry work — it's about volume and diversity, not reasoning quality.

---

## Invariants

- **Foundry never auto-commits to the component library.** `--promote` copies to `~/.maxim/components/` (user path). `--promote --dev` copies to source tree for maintainers. Human must commit.
- **Generated components must pass the same validation as hand-written ones.** No special treatment.
- **Gauntlet campaign structure is deterministic; bio-system outcomes are stochastic.** Same spec → same encounters/choices/expectations. LLM responses vary. Adaptive rerun policy handles borderline candidates.
- **Each gauntlet gets a fresh MemoryHub** (Hippocampus, ATL, EC, NAc, SCN, AngularGyrus, Cerebellum). Zero state leakage. `__post_init__` handles wiring — don't call `_wire_multi_layer()` explicitly.
- **Run-level EC persists across candidates** for cross-candidate diversity. Per-gauntlet EC is isolated. Signatures copied to run-level EC after scoring.
- **Infra failures ≠ candidate failures.** Setup crashes get one retry, then `infra_error` (not scored). Encounter crashes score as partial with penalty.
- **SEM protocol test crashes are isolated.** Each test in its own try/except. Tests 1-3 crashing = reject. Tests 6-8 crashing = warn + proceed.
- **Genre gating applies during generation.** Theme genre tag enforced at every step.
- **Sequential by default, parallel by opt-in.** F-1 validation runs sequentially (needs run-level EC). Gauntlets parallelize via `ProcessPoolExecutor`. Results collected via filesystem.
- **JSON repair before validation.** LLM output → `json-repair` → schema check. Malformed batches fall back to single-spec. Max 2 retries per candidate, then `generation_failure`.
- **Scoring formula is explicit.** Single-measurement dimensions (25% weight) from first run. Per-run dimensions (75% weight) averaged. `VARIANCE_THRESHOLD = 0.25` triggers extra runs.
- **No new dependencies.** Foundry uses existing EntityDesigner, ComponentRegistry, DM runtime, BenchmarkRunner, json-repair, and concurrent.futures.

---

## Phase Summary

| Phase | Work | LOC | What it enables |
|-------|------|-----|----------------|
| F-0 | Generation engine + batch design + JSON repair | ~220 | `maxim --foundry "theme" --count N` produces candidate YAMLs |
| F-1 | Validation pipeline + semantic sanity + EC diversity | ~180 | Rejects malformed/nonsensical/cross-genre/duplicate specs before testing |
| F-2 | Test gauntlet + SEM protocol tests + full-stack integration + isolation + error recovery | ~500 | Fresh MemoryHub per candidate, 8 structural tests, 3-encounter campaign with all 22 systems, checkpoint/crash handling, multi-run averaging |
| F-3 | Scoring + curation + reports (11 dimensions, stochastic averaging) | ~250 | Rank by schema + bio-system + salience + ATL + EC + motor + temporal + diversity |
| F-4 | Theme templates + theme CLI | ~150 | `maxim --foundry cyberpunk` with pre-built category distributions |
| F-5 | CLI integration + session persistence + incremental runs + parallel mode | ~200 | Resume, re-test, promote, `--parallel N` for multi-process execution |
| F-6 | Downstream integration — encounter library, narrator enhancement, interactive curation, benchmark generation | ~250 | Foundry outputs feed back into the broader system |
| **Total** | | **~1,750** | Full-stack autonomous SEM component generation pipeline |

## Systems Exercised

Every system the foundry touches, and where in the pipeline:

| System | Module | Exercised In | How |
|--------|--------|-------------|-----|
| SEM Entity tree | `embodiment/sem.py` | F-2 protocol tests | Instantiate, reparent, detach, find |
| Spec parser | `embodiment/spec.py` | F-2 protocol test 1 | `_parse_entity()` from YAML |
| Tool bridge | `embodiment/tool_bridge.py` | F-2 protocol test 4 | `generate_tools_for_entity()` + collision check |
| Component Registry | `embodiment/component_registry.py` | F-0 through F-6 | Discovery, instantiate, extends, genre filter |
| Cerebellum | `embodiment/cerebellum.py` | F-2 gauntlet enc. 2 | Forward model formation, prediction accuracy |
| Motor programs | `embodiment/motor.py` | F-2 gauntlet enc. 2 | Program crystallization from repeated sequences |
| PainBus | `embodiment/body.py` | F-2 gauntlet enc. 3 | Failure mode → pain signal routing |
| Body runtime | `embodiment/body.py` | F-2 gauntlet | Failure eval, vital drift, prompt state |
| Cascade resolver | `simulation/dm_runtime.py` | F-2 protocol test 7 + gauntlet | Reads, writes, safe_eval_expr |
| NarrativeTranscriber | `simulation/narrative_transcriber.py` | F-2 gauntlet | `transcribe_to_items()` → SalienceItem + NarrativeWhere |
| Salience protocols | `salience/protocols.py` + `where.py` | F-2 gauntlet | SalienceItem, NarrativeWhere, SalienceNetwork.update() |
| SalienceNetwork | `salience/salience_network.py` | F-2 gauntlet | Entity tracking, novelty decay, to_context_str() |
| ATL concepts | `memory/concept_extractor.py` | F-2 gauntlet | Concept creation, INSTANCE_OF edges, inheritance |
| NAc causal learning | `decisions/nac.py` | F-2 gauntlet enc. 2-3 | Observation, link formation, confidence |
| Hippocampus | `memory/hippocampus.py` | F-2 gauntlet all enc. | Capture, consolidation candidacy, recall |
| SCN temporal | `time/scn.py` | F-2 gauntlet | Temporal bin indexing across encounters |
| EC similarity | `similarity/ec.py` | F-1 diversity + F-2 | Signature indexing, duplicate detection |
| DM runtime | `simulation/dm_runtime.py` | F-2 gauntlet | SceneState, entity lifecycle, swap_entity |
| Reveal conditions | `simulation/dm_runtime.py` | F-2 gauntlet | `evaluate_reveal_conditions()` for hidden sensors |
| Encounter library | `simulation/encounter_library.py` | F-6 | Archive promoted gauntlet encounters |
| Generative campaigns | `simulation/narrator.py` | F-6 | Component-aware entity generation |
| Benchmark system | `simulation/benchmark.py` | F-6 | Auto-generated benchmark suites from promoted components |
| Interactive prompts | `interactive/prompts.py` | F-6 | `--curate` mode for human review |

## Testing Strategy

- **F-0:** Test batch generation with mocked LLM. Verify output YAML is parseable.
- **F-1:** Test each validation check in isolation with crafted good/bad specs. Test EC-based diversity scoring with mock signatures.
- **F-2:** Test SEM protocol tests (8 tests) with crafted good/bad specs. Test gauntlet generation from a known spec — verify campaign structure (3 encounters, expectations present, NarrativeTranscriber wiring, MemoryHub initialization). Run gauntlet with mock LLM and verify: SalienceNetwork tracks entity, ATL concept created, NAc observations recorded, SCN bins populated, EC signature indexed.
- **F-3:** Test scoring rubric with crafted results (perfect score, zero score, edge cases across all 11 dimensions). Test promote/review/reject bucketing. Test report generation.
- **F-4:** Test theme loading from YAML. Test category distribution.
- **F-5:** Test session persistence (write + resume). Test `--promote` file copy.
- **F-6:** Test encounter archival. Test generative campaign component discovery. Test benchmark suite generation from promoted components. Test interactive curation flow with mock PromptHandler.
