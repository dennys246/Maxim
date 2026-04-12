# Asset Foundry Plan

> **Status:** DEFERRED (post-1.0). Design complete, not started.
>
> **Revive when:** Manual SEM component authoring becomes a demonstrable bottleneck — i.e., someone is waiting on new components to unblock a sim or experiment. Current library (54 components across 7 categories) is sufficient for 1.0 demonstration scenarios. Do not build this preemptively.

**Goal:** An autonomous pipeline that generates, validates, tests, and curates SEM components — expanding the component library without manual YAML authoring while stress-testing the bio-stack against novel entity designs.

**Motivation:** Hand-writing SEM components is slow and biased toward what the author thinks of. An LLM-driven foundry generates creative sensor combinations, failure modes, and affordance patterns that humans wouldn't consider — then the simulation system ruthlessly filters for the ones that actually produce interesting cognitive behavior.

**Depends on:** EntityDesigner (shipped), ComponentRegistry + genre gating (shipped), DM campaign system (shipped), BenchmarkRunner (shipped).

**Trigger:** Post-publication. The foundry consumes LLM tokens for both generation and testing — ship the stable package first, then use the foundry to grow the library.

**Success criteria:** Run a simulation where the foundry generates an entity on-demand that is (1) **usable** — passes SEM protocol tests, (2) **useful** — scores > 0.7 on the core rubric, and (3) **used** — the agent called the entity's affordances during the sim (affordance_usage_count > 0).

---

## Architecture

```
┌────────────────────────────────────────────────────────────────────────────┐
│  maxim --foundry "cyberpunk weapons" --count 10                            │
│                                                                            │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────────┐  │
│  │ GENERATE │→ │ VALIDATE │→ │SEM PROTO │→ │ GAUNTLET │→ │SCORE+CURATE │  │
│  │          │  │          │  │  TESTS   │  │          │  │             │  │
│  │ Theme +  │  │ Schema   │  │ 8 struct │  │ 3-enc    │  │ 4-dim core  │  │
│  │ genre +  │  │ Semantic │  │ tests per│  │ campaign │  │ rubric      │  │
│  │ batch    │  │ Genre    │  │ candidate│  │ per cand │  │ promote/    │  │
│  │ JSON fix │  │          │  │ (no LLM) │  │ fresh    │  │ review/     │  │
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

## Phase F-0: Generation Engine (~220 LOC)

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

**Energy gating:** Before calling the EntityDesigner, check the session's energy budget via the energy tracker. If the budget can't afford a ~500 token generation call, skip generation and log `energy_insufficient`. This applies to both standalone foundry runs (where it caps total generation) and demand-driven generation (F-7, where it naturally limits on-demand entities per session based on available budget).

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

## Phase F-1: Validation Pipeline (~180 LOC)

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
| **Semantic sanity** | Sensor ranges within reasonable bounds (warn if range span > 10,000), at least one failure mode triggerable within nominal operation | `range: [0, 1000000]`, untriggerable failures |

**Validation result per candidate:**
```python
@dataclass
class ValidationResult:
    candidate_path: str
    valid: bool
    errors: list[str]      # Hard failures (reject)
    warnings: list[str]    # Soft issues (test anyway)
```

**Files:**
- **New:** `src/maxim/simulation/foundry_validation.py` — `validate_candidate()`, `validate_batch()`
- **Touch:** `src/maxim/embodiment/spec.py` — extract schema validation into a reusable `validate_entity_spec()` function

---

## Phase F-2: Test Gauntlet (~400 LOC)

**Goal:** Run each validated candidate through a campaign that exercises the bio-stack and measures engagement.

### Gauntlet Campaign Design

The foundry generates a **per-candidate micro-campaign** (3 encounters, ~30s each) that specifically targets the candidate's capabilities:

**Encounter 1 — Discovery:**
Scene introduces the entity. Agent observes its sensors and affordances.
Tests: Hippocampus captures the entity. SalienceNetwork tracks it.

**Encounter 2 — Interaction:**
Agent must use the entity's affordances in a goal-directed context.
Tests: NAc learns action→outcome links. Affordances are actually called.

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
- Pain: `min_signals: 1` if the entity has failure modes

### Execution and isolation

Uses the cheapest available LLM (Mistral-7b or smollm) for testing — the point isn't quality reasoning, it's whether the entity spec produces bio-system engagement.

**MemoryHub state isolation:** Each gauntlet gets a **fresh MemoryHub** instance (new Hippocampus, ATL, EC, NAc, SCN, AngularGyrus, Cerebellum) and a **fresh ReactionBus** (or its predecessor PainBus, depending on whether [reaction_abstraction_plan Phase 2](../reaction_abstraction_plan.md) has landed). This prevents cross-contamination — memories from gauntlet 1 must not appear in gauntlet 2's recall, and reactive signals must not leak across candidates.

**Run-level EC for cross-candidate diversity:** A separate `EntorhinalCortex` instance persists across the entire foundry run. After each gauntlet scores, the candidate's `SituationSignature` is copied into the run-level EC for future diversity checks. Persisted to `~/.maxim/foundry/{run_id}/ec_index.json`.

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
            angular_gyrus=AngularGyrus(),
            cerebellum=Cerebellum(),
        )

        gauntlet = generate_gauntlet(candidate)
        result = run_gauntlet(gauntlet, hub=hub, model=test_model)

        # Copy signature to run-level EC for cross-candidate diversity
        if result.situation_signature:
            run_ec.register(candidate.name, result.situation_signature)

        hub.on_session_end()
        scores[candidate.name] = score_result(result, candidate)

    except Exception as e:
        if is_infra_failure(e):
            log.error("Infra failure for %s: %s — retrying once", candidate.name, e)
            scores[candidate.name] = InfraError(candidate.name, str(e))
        else:
            scores[candidate.name] = score_partial(result_so_far, candidate)
```

**Concurrency:** Gauntlets run sequentially. The existing `WorkerPool` handles LLM inference concurrency across lanes — the foundry doesn't need its own parallel orchestration.

### Error recovery

**Infra failures** (MemoryHub setup crash, import error, OOM):
1. Retry once with a fresh MemoryHub
2. If retry fails, mark as `"status": "infra_error"` — not scored
3. Reported separately in the foundry report

**Candidate failures** (encounter crash, spec-triggered edge case):
1. Save partial result (encounters completed so far)
2. Score with penalty (incomplete encounters score 0)
3. Move to next candidate

**SEM protocol test crashes:** Each of the 8 tests wrapped in try/except. Tests 1-3 crashing = reject. Tests 6-8 crashing = warn + proceed.

**Resume:** `--resume` re-runs only candidates without a final `results/{name}.json`.

### Entity context injection

When a generated entity is instantiated, the foundry composes a **strategy description** from the spec and injects it into the prompt via the existing `PromptBudgeter`:

```python
budgeter.add("entity_context", entity_context_str, SectionPriority.IMPORTANT)
```

This bridges the gap between individual tool descriptions and strategic usage. The LLM sees tools ("shock_strike", "blunt_strike") and sensor state ("charge: 0.9") — the entity context adds the strategic layer:

```
=== Entity: shock_baton (weapon) ===
A melee weapon with two attack modes.
- shock_strike: high damage, drains charge. Use when charge is available.
- blunt_strike: lower damage, preserves charge. Use when charge is low.
- Failure risk: charge < 0.05 disables electric mode. Durability < 0.1 causes structural failure.
- Recharge from power sources when possible.
```

This is auto-composed from the entity spec (sensor names + ranges + affordance descriptions + failure trigger conditions). No LLM call needed — it's template-based. The `body_state` section (already at CRITICAL priority) shows current sensor values; the entity context section (at IMPORTANT priority) shows how to use them.

**Enhancement opportunity (post [reaction_abstraction Phase 3](../reaction_abstraction_plan.md)):** Once `EmbodimentPerceptSource` is wired into the agent loop via the `PerceptProducer` protocol, sensor readings flow as typed Percepts instead of only through `format_body_state_for_prompt()`. The gauntlet's Hippocampal engagement scoring becomes richer because the bio-stack sees typed percepts with modality/context metadata, not just prompt text. Not required for the foundry to work — the string-based path is sufficient — but it's a free upgrade if Phase 3 has landed by revive time.

**Files:**
- **New:** `src/maxim/simulation/foundry_gauntlet.py` — `generate_gauntlet()`, `run_gauntlet()`
- **New:** `src/maxim/simulation/foundry_sem_tests.py` — `run_sem_protocol_tests()`
- **New:** `src/maxim/embodiment/entity_context.py` — `compose_entity_context()` for prompt injection
- **Touch:** `src/maxim/agents/prompt_builder.py` — add entity_context section
- **Touch:** `src/maxim/simulation/dm_schema.py` — ensure programmatic CampaignDef construction works

### SEM Protocol Test Suite

Before the gauntlet campaign runs, every candidate goes through 8 structural tests (fast, no LLM):

1. **Instantiation** — `_parse_entity()` succeeds, valid entity_type
2. **Sensor R/W** — every sensor initializes within range, responds to vital_metrics writes
3. **Affordance enumeration** — all affordances have descriptions + typed params
4. **Tool generation + collision check** — `tool_bridge` generates tools, no collisions within entity or across batch
5. **Failure mode triggers** — trigger fields reference real sensors, valid ops, pain in [0, 1]
6. **Composition** — body_part/weapon types can reparent/detach
7. **Cascade compatibility** — CascadeResolver can read all sensors via ref paths
8. **Inheritance** — children of `extends` parents have all parent sensors

```
Generate → Validate (schema) → SEM Protocol Tests → Gauntlet (bio-system) → Score → Curate
                                      ↓
                              Reject structurally
                              broken components
                              before sim cost
```

---

## Phase F-3: Scoring + Curation (~250 LOC)

**Goal:** Rank candidates by bio-system engagement and sort into promote/review/reject buckets.

### Core Scoring Rubric (4 dimensions)

| Dimension | Weight | How Measured |
|-----------|--------|-------------|
| **Hippocampal engagement** | 30% | Episodic captures / encounters (did the agent remember this entity?) |
| **Causal learning** | 30% | NAc observations + link confidence (did the agent learn cause-effect from this entity?) |
| **Pain/failure activation** | 20% | Pain signals published, failure modes triggered (did the entity's failure modes fire?). Post [reaction_abstraction Phase 2](../reaction_abstraction_plan.md): measures `ReactionBus.history(kind="pain")` instead of `PainBus.history`. |
| **Affordance usage** | 20% | Number of distinct affordances the agent actually called (was the entity used, not just observed?) |

**Score formula:** `score = Σ(dimension_i × weight_i)`, normalized to [0, 1].

**Extensible foundation:** The scoring system accepts a `ScoringConfig` with a `dimensions: dict[str, float]` weights map. The 4 core dimensions ship in v1. Additional dimensions (cerebellum forward models, motor programs, salience tracking, ATL concepts, EC indexing, temporal coverage, diversity) can be added by extending the config without changing the scoring engine. Each dimension is a function `(GauntletResult, CandidateSpec) → float` — adding a new one is ~20 LOC.

```python
@dataclass
class ScoringConfig:
    dimensions: dict[str, float] = field(default_factory=lambda: {
        "hippocampal_engagement": 0.30,
        "causal_learning": 0.30,
        "pain_failure": 0.20,
        "affordance_usage": 0.20,
    })
    promote_threshold: float = 0.7
    reject_threshold: float = 0.4
```

**Thresholds:**
- **Promote** (> 0.7): Auto-copied to `promoted/` directory, ready for human review
- **Review** (0.4 - 0.7): Interesting but flawed — flagged with specific issues
- **Reject** (< 0.4): Inert or broken — logged but not kept

### Report

```markdown
# Foundry Report — cyberpunk weapons
Generated: 10 | Validated: 8 | Tested: 8 | Promoted: 5 | Review: 2 | Rejected: 1

## Promoted
| Component | Score | Highlights |
|-----------|-------|-----------|
| plasma_cutter | 0.85 | 3 affordances used, pain cascade from overheat |
| mono_wire | 0.78 | NAc learned stealth→success, novel failure mode |

## Flagged for Review
| Component | Score | Issue |
|-----------|-------|-------|
| gravity_hammer | 0.55 | Agent never used affordances (too abstract?) |

## Rejected
| Component | Reason |
|-----------|--------|
| quantum_blade | Zero bio-system engagement — all sensors static |
```

**Files:**
- **New:** `src/maxim/simulation/foundry_scoring.py` — `score_result()`, `curate_batch()`, `generate_report()`
- Output format: `report.md` (human-readable) + `scores.json` (machine-readable)

---

## Session Persistence + CLI (~150 LOC)

Each foundry run produces:
```
~/.maxim/foundry/{run_id}/
├── config.yaml          # Theme, params, model, timestamp
├── candidates/          # Raw generated YAML specs
├── rejected/            # Failed validation or protocol tests
├── results/             # Per-candidate gauntlet results
├── promoted/            # Top scorers (ready for review + commit)
├── scores.json          # Machine-readable scoring data
└── report.md            # Human-readable summary
```

### CLI

```
maxim --foundry <theme> [options]

Options:
  --count N         Total components to generate (default: 10)
  --genre GENRE     Genre tag for components
  --category CAT    Generate only this category
  --model MODEL     LLM for generation (default: medium tier)
  --test-model M    LLM for gauntlet testing (default: small tier)
  --dry-run         Generate + validate only, skip testing
  --resume RUN_ID   Continue a previous foundry run
  --promote PATH    Copy a candidate to ~/.maxim/components/
```

`--promote` copies to `~/.maxim/components/{category}/` (user search path, always writable). `--promote --dev` copies to `src/maxim/_data/components/` for maintainers.

---

## Cost Model

| Operation | Model | Tokens | Cost |
|-----------|-------|--------|------|
| Generate (per entity) | Mistral-7b (local) | ~500 | $0.00 |
| Generate (per entity) | Claude | ~500 | ~$0.005 |
| Gauntlet (3 encounters) | Mistral-7b (local) | ~1,500 | $0.00 |
| Gauntlet (3 encounters) | Claude | ~1,500 | ~$0.01 |
| Fast validate / SEM tests | N/A | 0 | $0.00 |

**Typical run (local):** 10 components × (generate + validate + gauntlet) ≈ 3-5 minutes, $0.00
**Typical run (Claude):** 10 components ≈ 5-10 minutes, ~$0.15

---

## Invariants

- **Foundry never auto-commits to the component library.** Human must review and commit.
- **Generated components pass the same validation as hand-written ones.** No special treatment.
- **Each gauntlet gets a fresh MemoryHub.** Zero state leakage between candidates.
- **Run-level EC persists across candidates** for cross-candidate diversity.
- **Infra failures ≠ candidate failures.** Setup crashes get one retry, then `infra_error` (not scored).
- **Energy gating.** Generation only proceeds if energy budget allows. No silent overspend.
- **Scoring is extensible.** 4 core dimensions in v1. New dimensions are added via `ScoringConfig` without engine changes.
- **Entity context injection uses existing PromptBudgeter.** No new prompt infrastructure needed.
- **No new dependencies.** Foundry uses existing EntityDesigner, ComponentRegistry, DM runtime, energy tracker, json-repair, and PromptBudgeter.

---

## Phase Summary

| Phase | Work | LOC | What it enables |
|-------|------|-----|----------------|
| F-0 | Generation engine + batch design + JSON repair + energy gate | ~220 | `maxim --foundry "theme" --count N` produces candidate YAMLs |
| F-1 | Validation pipeline + semantic sanity | ~180 | Rejects malformed/nonsensical specs before testing |
| F-2 | Gauntlet + SEM protocol tests + entity context injection + isolation + error recovery | ~400 | 8 structural tests, 3-encounter campaign, entity strategy in prompts |
| F-3 | Scoring (4 core dimensions, extensible) + curation + reports | ~250 | Rank, promote, flag, report. Foundation for adding dimensions |
| — | Session persistence + CLI | ~150 | Resume, promote workflow |
| **Total** | | **~1,200** | Core foundry pipeline |

---

## Future Extensions (deferred — implement when core proves out)

These are tracked in `future_plans.md`, not implemented as part of the foundry core:

| Extension | Trigger | Notes |
|-----------|---------|-------|
| **Theme templates** | Running foundry across multiple genres | Pre-built YAML configs with category distributions and sub-themes |
| **Additional scoring dimensions** | Core 4 dimensions prove insufficient | Cerebellum, motor programs, salience, ATL, EC, temporal, diversity — each ~20 LOC |
| **Demand-driven generation (F-7)** | Library too small for generative campaigns | Energy-gated on-demand entity creation during sim, post-sim quality report |
| **`generate_entity` tool** | Agent needs to model novel entities | Requires salience refactor (S-0–S-5) for "perceived but unmodeled" detection. Table until stress-tested. |
| **Encounter library archival** | Promoted gauntlets should be reusable | Archive gauntlet micro-campaigns as encounter templates |
| **Narrator entity awareness** | Narrator should use registry components | Component-aware scene generation in generative campaigns |
| **Interactive curation** | Review bucket needs human-in-the-loop | PromptHandler-based review workflow for borderline candidates |
| **Benchmark suite generation** | Promoted components should become benchmarks | Auto-generate benchmark scenarios from promoted entities |
| **Iterative spec refinement** | Single-shot specs are insufficient quality | Energy-aware multi-pass refinement ("rough spec if low energy, polished if budget available") |
| **PerceptProducer/ReactionProducer protocol tests** | [reaction_abstraction Phase 3](../reaction_abstraction_plan.md) lands | Test 9: generated sensors satisfy `PerceptProducer` (EmbodimentPerceptSource can wrap them). Test 10: generated failure modes emit `Reaction` through CerebellumModulator. ~40 LOC addition to F-2's test suite. |

---

## Testing Strategy

- **F-0:** Test batch generation with mocked LLM. Verify output YAML is parseable. Test energy gate (mock tracker at zero budget → no generation).
- **F-1:** Test each validation check in isolation with crafted good/bad specs.
- **F-2:** Test SEM protocol tests (8 tests) with crafted good/bad specs. Test gauntlet generation from a known spec — verify campaign structure. Test entity context composition from spec. Run gauntlet with mock LLM and verify bio-system metrics are collected. Test MemoryHub isolation between sequential gauntlets.
- **F-3:** Test scoring rubric with crafted results (perfect score, zero score, edge cases). Test promote/review/reject bucketing. Test extensible dimension registration. Test report generation.
