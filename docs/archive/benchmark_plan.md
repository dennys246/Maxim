# Simulation Benchmark Plan — Multi-Model Comparative Testing

## Context

The tool refactoring work (2026-04-06) revealed that model choice dramatically affects AUT behavior. Mistral-7B discovered `think` and attempted `remember`; Qwen 14B got stuck in a `respond` loop. Tool hallucination rates, recall fidelity, and narrative engagement all vary per model. Currently, comparing models requires manually re-running sims and reading through logs.

This plan adds a `maxim --sim benchmark` subcommand that automates multi-model comparison, computes standardized metrics across three tiers (LLM behavior, cognitive architecture, and embodiment), and outputs a comparative report. It begins with **Phase 0: sim improvements** that benefit all sim modes, not just benchmarks.

## Three-Tier Metric Architecture

Maxim's value is the bio-inspired cognitive architecture, not just the LLM. Benchmarks must measure all three layers:

```
Tier 1: LLM Behavior
  "Is this model good at being an agent?"
  Tool usage, hallucination rates, JSON compliance, latency, cost

Tier 2: Cognitive Architecture
  "Does this model drive the bio-systems effectively?"
  Memory formation + consolidation, causal learning curves,
  semantic promotion, pain/aversion learning, temporal patterns

Tier 3: Embodiment (future — hooks designed now, details in embodiment_core_plan.md)
  "Does this model inhabit a body believably?"
  Percept consistency, forward model accuracy, pain sigma,
  body-state coherence
```

Tier 1 validates the LLM. Tier 2 validates the architecture. Tier 3 validates the embodiment thesis. A model could ace Tier 1 (perfect tool usage) while failing Tier 2 (never forming causal links) — that's a model that games the tools but doesn't learn. The benchmark must distinguish these cases.

## CLI Interface

```bash
# Run a benchmark suite against multiple models
maxim --sim benchmark \
  --models mistral-7b,qwen2.5-14b,llama-3-8b \
  --campaign scenarios/benchmarks/cognitive_suite.yaml \
  --runs 3                    # repeat each model N times for variance
  --output data/benchmarks/   # output directory

# Run against a single new model (quick smoke test)
maxim --sim benchmark \
  --models phi-3-mini \
  --campaign scenarios/benchmarks/quick_check.yaml

# Compare against a previous benchmark baseline
maxim --sim benchmark \
  --models qwen2.5-14b \
  --campaign scenarios/benchmarks/cognitive_suite.yaml \
  --baseline data/benchmarks/baseline_20260406.json

# Filter by tag
maxim --sim benchmark --models mistral-7b --tag memory

# List available scenarios
maxim sim list
maxim sim list --tag safety
```

---

## Phase 0: Simulation Infrastructure Improvements

> These changes benefit **all sim modes** (agent, research, benchmark, interactive), not just benchmarks. They're prerequisites for Phase 1 but independently valuable.

### 0a. Enrich SimulationResult (~30 LOC in orchestrator.py)

Currently `SimulationResult` returns a thin summary while detailed data is only persisted to session files. Make it carry everything:

```python
@dataclass
class SimulationResult:
    # ... existing fields ...
    introspector: Any = None                          # already added

    # NEW — captured at orchestrator shutdown
    tool_stats: dict[str, Any] = field(...)           # executor.tool_usage_stats()
    actions: list[dict[str, Any]] = field(...)        # serialized ActionRecords
    pain_events: list[dict[str, Any]] = field(...)    # pain signals fired during run
    subsystem_snapshot: dict[str, Any] = field(...)   # introspector.benchmark_snapshot()
    expectations_results: list[Any] = field(...)      # validation results (if expectations defined)
    router_stats: dict[str, Any] = field(...)         # JSON parse counters, token counts per tier
```

Populated at shutdown from objects already in scope:
```python
result.tool_stats = aut_executor.tool_usage_stats()
result.actions = [
    {"timestamp": a.timestamp, "tool_name": a.tool_name, "tool_args": a.tool_args,
     "result_success": a.result_success, "result_output": str(a.result_output)[:200],
     "blocked": a.blocked, "block_reason": a.block_reason}
    for a in bridge.get_all_actions()
]
result.subsystem_snapshot = aut_introspector.benchmark_snapshot()
```

**Why this matters:** `run_campaign()` and `BenchmarkRunner` both read `result.tool_stats` directly — no file parsing. Regular users can inspect `result.actions` in scripts. The sim report persists the same data to disk; the result object carries it in memory.

### 0b. Activate full bio-stack in sim mode (~50 LOC in orchestrator.py)

Currently missing from AUT in sim mode: `PainDetector`, `DefaultNetwork` (SalienceNetwork, NoveltyTracker, AttentionNetwork). These need activation for Tier 2 engagement metrics and for narrative transcription to work.

```python
# In start_simulation_mode(), after AUT memory subsystem setup:

# Wire PainDetector for aversion learning metrics
from maxim.proprioception.pain import PainDetector
aut_pain_detector = PainDetector()
# ... wire to executor's ToolPainBridge ...

# Wire DefaultNetwork for salience/novelty/attention
from maxim.default_network.network import DefaultNetwork
aut_default_network = DefaultNetwork(
    salience_network=SalienceNetwork(),
    novelty_tracker=NoveltyTracker(),
    attention_network=AttentionNetwork(),
)
# ... subscribe to AUT's percept bus ...
```

Pass `pain_detector` to `AUTIntrospector` so `pain_stats()` returns real PainDetector counters instead of the hippocampus word-search proxy.

### 0c. Extend expectations with bio-system types (~80 LOC in validation.py)

The existing expectation DSL has 7 types. Add bio-system types so **scenarios carry their own cognitive assertions** — benchmarks just aggregate pass rates:

```yaml
expectations:
  # Memory system
  - type: memory_count_range
    params: { min: 5, max: 50 }
    description: "Formed 5-50 episodic memories"

  - type: concept_formed
    concept_name: "Verath"
    description: "ATL formed a semantic concept for Verath"

  - type: graph_density_above
    params: { min_density: 1.0 }
    description: "Associative memory graph is well-connected"

  # Causal learning
  - type: causal_link_formed
    event_contains: "say"
    description: "NAc learned from say actions"

  - type: prediction_valence
    tool: "delete_file"
    expected_valence: "negative"
    description: "NAc predicts delete_file has negative outcome"

  # Tool behavior
  - type: hallucination_rate_below
    params: { max_rate: 0.3 }
    description: "Less than 30% tool hallucination"

  - type: tool_used
    tool: "examine"
    description: "AUT used examine tool at least once"

  # Pain / aversion
  - type: pain_signal_count
    params: { min: 1 }
    description: "At least one pain signal fired"
```

These become **the same language** for scenarios, benchmarks, and regression tests. A benchmark scenario is a regular scenario with more expectations + a `benchmark:` section.

Validation entry point stays the same — `validate_expectations()` gains new type handlers that read from `SimulationResult.subsystem_snapshot` and `SimulationResult.tool_stats`.

### 0d. Add scenario metadata for discoverability (~20 LOC in scenario_source.py)

Extend `ScenarioDefinition` with optional metadata:

```yaml
name: hippocampal_recall_short
description: ...
tags: [memory, recall, interference, hippocampus, narrative]
difficulty: medium
estimated_duration_s: 45
subsystems_tested: [hippocampus, nac, atl]
tools_tested: [memory_recall, say, think, examine]
```

Enables:
- `maxim --sim benchmark --tag memory` — run only memory-related scenarios
- `maxim sim list --tag safety` — browse available scenarios
- Auto-categorization in benchmark reports by tag/subsystem

### 0e. Add JSON compliance counter (~10 LOC in json_parser.py)

`json_parser.py` has a 4-stage repair pipeline but no success counter. Add:
- `_json_first_try: int` — incremented when `json.loads()` succeeds in Stage 1
- `_json_total: int` — incremented on every JSON parse attempt

Include in `SimulationResult.router_stats`.

### 0f. Add AUTIntrospector.benchmark_snapshot() (~30 LOC in introspection.py)

Collects all data a benchmark needs in one call:

```python
def benchmark_snapshot(self, seed_keywords: list[str] | None = None) -> dict:
    snapshot = self.full_analysis(seed_keywords=seed_keywords)

    if self._hippocampus is not None:
        snapshot["hippocampus_stats"] = self._hippocampus.stats()
    if self._nac is not None:
        snapshot["nac_stats"] = self._nac.stats()
    if self._pain_detector is not None:
        snapshot["pain_stats"] = self._pain_detector.get_stats()

    return snapshot
```

### Phase 0 summary

| Sub-phase | LOC | What it enables |
|-----------|-----|-----------------|
| 0a: Enrich SimulationResult | ~30 | Programmatic access to all run data |
| 0b: Activate full bio-stack in sim | ~50 | Salience, novelty, attention, pain in sims |
| 0c: Bio-system expectation types | ~80 | Cognitive assertions in scenario YAML |
| 0d: Scenario metadata | ~20 | Tags, difficulty, subsystem discovery |
| 0e: JSON compliance counter | ~10 | First-try parse success tracking |
| 0f: benchmark_snapshot() | ~30 | Single-call subsystem data collection |
| **Total Phase 0** | **~220** | |

---

## Tier 1: LLM Behavior Metrics

| Metric | How to compute |
|--------|----------------|
| **Tool hallucination rate** | `result.tool_stats["hallucination_rate"]` |
| **Alias redirect rate** | `len(tool_stats["alias_redirects"]) / tool_stats["total_attempts"]` |
| **Correct tool usage rate** | `1 - hallucination_rate` |
| **JSON compliance** | `result.router_stats["json_first_try"] / result.router_stats["json_total"]` |
| **Think-before-act rate** | Count `think → X` sequences in `result.actions` |
| **Reasoning depth** | Max chain length within a single turn |
| **Action latency (p50, p95)** | `ActionRecord.timestamp` deltas |
| **Token efficiency** | `total_actions / (total_tokens / 1000)` |
| **Cost per turn** | `report.cost_usd / report.turns` |

---

## Tier 2: Cognitive Architecture Metrics

### Memory System

| Metric | How to compute |
|--------|----------------|
| **Memory formation rate** | `hippo_stats["total_memories"] / turns` |
| **Memory recall success** | `memory_recall(keyword)["count"] > 0` |
| **Behavioral recall** | Scan `result.actions` for `say("Verath")` vs `respond("Verath")` |
| **Interference resistance** | Seed content in hippocampus after interference turns |
| **Associative graph density** | `hippo_stats["graph_edges"] / hippo_stats["graph_nodes"]` |
| **Concept formation rate** | `system_stats["atl_concepts"] / turns` |
| **Concept avg confidence** | Mean confidence across `concept_query(limit=100)` |

### Causal Learning

| Metric | How to compute |
|--------|----------------|
| **Causal link count** | `nac_stats["total_links"]` |
| **Learning efficiency** | `total_links / total_actions` |
| **Causal diversity** | `event_signatures / total_links` |
| **Observation density** | `total_observations / total_links` |
| **Prediction accuracy** | `predict_outcome("delete_file")` → negative valence? |

### Aversive Learning

| Metric | How to compute |
|--------|----------------|
| **Pain signal count** | `pain_stats["total_pain_signals"]` (via PainDetector, activated in Phase 0b) |
| **Pain avoidance rate** | After pain, does AUT avoid repeating painful tool? Scan action history |

### Engagement

| Metric | How to compute |
|--------|----------------|
| **Engagement cascade depth** | Bio-driven: salience → novelty → capture → action → learning (see below) |
| **Type-token ratio** | Unique tokens / total tokens across responses (loop detector) |
| **Scene-element reference rate** | AUT references `scene_elements` from percept metadata |

---

## Narrative Percept Transcriber

> **Depends on:** [Lane tier architecture](../archive/lane_tier_plan.md) (small tier) + DefaultNetwork activation (Phase 0b).

Narrative text currently bypasses the bio-stack. The transcriber converts text into structured detections for the full pipeline:

```
"A massive silver elm with a stone door"
    ↓
NarrativeTranscriber.transcribe(text)  [small tier — smollm 1.7B on CPU]
    ↓
[{track_id: "elm_1", label: "silver_elm", conf: 0.9, bbox: [...]}, ...]
    ↓
Percept(source="narrative", cli_input=text, detections=[...])
    ↓
SalienceNetwork → NoveltyTracker → AttentionNetwork → Hippocampus → LLM → NAc
```

Maintains **stable entity IDs** across turns — `NoveltyTracker` correctly computes novelty decay when "silver elm" reappears.

### Implementation (~100 LOC)

**File:** `src/maxim/simulation/narrative_transcriber.py`

- LLM-powered entity extraction on `small` tier (smollm 1.7B, CPU, ~1s per turn)
- Produces `SalienceNetwork.update_from_detections()`-compatible dicts
- Narrative class IDs start at 900 (avoid COCO collision)
- Stable track_id per entity label across turns

### Engagement cascade (bio-driven measurement)

With the transcriber active, engagement is measured as propagation through the bio-stack:

```
Level 0: Percept delivered
Level 1: SalienceNetwork scored entities above threshold
Level 2: NoveltyTracker flagged novel entities
Level 3: Hippocampus captured scene content
Level 4: AUT action referenced a detected entity
Level 5: NAc formed causal link from scene interaction
```

**Engagement depth** = highest level per turn. **Engagement rate** = mean depth / 5 across turns.

### Fallback without transcriber

When small tier is unavailable:
1. Type-token ratio (always available)
2. Scene-element metadata annotations in YAML (deterministic)

---

## Unified Scenario YAML Format

One schema for all scenario types. Optional sections add capability when present — the loader ignores what it doesn't need:

```yaml
# ── Required ─────────────────────────────────────────────
name: hippocampal_recall_short
description: |
  Tests episodic recall under narrative interference.

# ── Metadata (optional — discoverability) ────────────────
tags: [memory, recall, interference, hippocampus]
difficulty: medium
estimated_duration_s: 45
subsystems_tested: [hippocampus, nac, atl]
tools_tested: [memory_recall, say, examine]

# ── Percepts ─────────────────────────────────────────────
timing: step_based
percepts:
  - at: 0
    source: cli
    cli_input: |
      You arrive at the village of Thornhaven...
    salience: 1.0
    novelty: 1.0
    metadata:
      scenario_tag: seed_password
      phase: "act1_warning"
      experiment_role: seed
      critical_detail: "Verath"
      scene_elements: [silver elm, stone door, carved face]
      expected_tool: say

  - at: 1
    source: proprioception
    content: pain_signal
    salience: 0.7
    metadata:
      pain_type: external_signal
      intensity: 0.8

# ── Expectations (optional — post-run assertions) ────────
expectations:
  # Existing types
  - type: memory_formed
    memory_contains: "Verath"
  - type: action_count_range
    params: { min: 5, max: 30 }
  # New bio-system types (Phase 0c)
  - type: hallucination_rate_below
    params: { max_rate: 0.3 }
  - type: causal_link_formed
    event_contains: "say"

# ── Benchmark (optional — tier + scoring config) ─────────
benchmark:
  category: memory
  tier: [1, 2]
  weight: 2.0
  seed_keywords: ["Verath"]
  metrics:
    - memory_recall_success
    - behavioral_recall
    - interference_resistance

# ── Suite (optional — meta-file referencing scenarios) ────
suite:
  default_models: [mistral-7b, qwen2.5-14b]
  scenarios:
    - path: scenarios/experiments/hippocampal_recall_short.yaml
      weight: 2.0
    - path: scenarios/benchmarks/causal_learning.yaml
      weight: 1.5
  scoring:
    memory_recall_success: { pass: 1.0 }
    hallucination_rate: { pass_below: 0.3 }

# ── Run config (optional — execution defaults) ───────────
config:
  persona: campaign
  max_turns: 50
  response_timeout: 60.0
  sandbox: tmpdir
```

### What this unifies

| Scenario type | Optional sections present |
|---------------|--------------------------|
| Safety test (`malware_with_pain.yaml`) | none (just percepts + expectations) |
| Refinement baseline | none |
| Research campaign | benchmark (when used in a suite) |
| Benchmark scenario | benchmark + metadata |
| Benchmark suite | suite + scoring (no percepts) |
| DM campaign (future) | dm section |

---

## Architecture

### BenchmarkRunner wraps run_campaign()

The standalone runner (`experiment.py`) handles campaign loading, simulation execution, and introspection. The benchmark runner wraps it rather than duplicating the orchestrator call:

```python
class BenchmarkRunner:
    def _run_single(self, scenario_path: str, model: str) -> RunResult:
        """Run one scenario on one model, return structured results."""
        result = run_campaign(
            campaign=scenario_path,
            aut_model=model,
            seed_keywords=self._get_seed_keywords(scenario_path),
            persona="campaign",
        )
        # result has .tool_stats, .actions, .subsystem_snapshot, .introspector
        metrics = self._compute_metrics(result, scenario)
        exp_results = validate_expectations(
            scenario.expectations,
            actions=result.actions,
            snapshot=result.subsystem_snapshot,
            tool_stats=result.tool_stats,
        )
        return RunResult(metrics=metrics, expectations=exp_results)
```

**`run_campaign()`** is the shared engine for:
- Standalone experiments (direct call)
- Benchmarks (`BenchmarkRunner` wraps it)
- Research protocol (`start_research_mode` could also wrap it)
- Future: generative campaigns

### Full runner flow

```
BenchmarkRunner
  ├── run() → BenchmarkReport
  │     ├── load suite YAML → scenario paths + weights + scoring
  │     ├── for each model:
  │     │     ├── (self-hosted) peer llm swap
  │     │     ├── for each run (1..N):
  │     │     │     ├── for each scenario:
  │     │     │     │     ├── run_campaign(scenario, model)
  │     │     │     │     ├── compute metrics from result
  │     │     │     │     ├── validate expectations
  │     │     │     │     └── print live progress (turn/model/scenario)
  │     │     │     └── aggregate across scenarios
  │     │     └── aggregate across runs (mean, stddev)
  │     ├── score against thresholds
  │     └── build BenchmarkReport
  ├── _compute_metrics(result, scenario) → dict
  └── _score(metrics, thresholds) → ModelScore
```

### Live progress during benchmark runs

For multi-model, multi-run benchmarks, users need intermediate feedback:

```
  ═══ cognitive_suite_v1 ═══  mistral-7b  run 1/3
    hippocampal_recall_short    [4/7 turns]  mem: 8  links: 3  pain: 0  $0.02
    ✓ hippocampal_recall_short  7 turns  4.2s  recall_Verath: 1.0
    tool_discovery              [2/5 turns]  halluc: 0.20  alias: 0.10
    ...

  ═══ cognitive_suite_v1 ═══  qwen2.5-14b  run 1/3
    ...
```

Reuses existing `sim_logger` formatting with per-turn bio-system snapshots. Added in Phase 4 (output).

### Metric computation

```python
def _compute_metrics(self, result, scenario) -> dict:
    snapshot = result.subsystem_snapshot  # captured at shutdown (Phase 0a)
    hippo = snapshot.get("hippocampus_stats", {})
    nac = snapshot.get("nac_stats", {})
    pain = snapshot.get("pain_stats", {})
    stats = snapshot.get("system_stats", {})
    turns = max(result.turns, 1)

    metrics = {}

    # Tier 1
    ts = result.tool_stats
    if ts:
        metrics["hallucination_rate"] = ts.get("hallucination_rate", 0)
        metrics["correct_tool_usage_rate"] = 1 - metrics["hallucination_rate"]
    if result.actions:
        metrics["think_before_act_rate"] = self._count_think_chains(result.actions) / turns

    # Tier 2
    metrics["memory_formation_rate"] = hippo.get("total_memories", 0) / turns
    metrics["associative_graph_density"] = (
        hippo.get("graph_edges", 0) / max(hippo.get("graph_nodes", 1), 1)
    )
    metrics["causal_link_count"] = nac.get("total_links", 0)
    metrics["learning_efficiency"] = nac.get("total_links", 0) / max(result.total_actions, 1)
    metrics["pain_signal_count"] = pain.get("total_pain_signals", 0)

    # Keyword recall
    for kw, data in snapshot.get("memory_recall", {}).items():
        metrics[f"recall_{kw}"] = 1.0 if data.get("count", 0) > 0 else 0.0

    # Engagement (transcriber-powered if available, fallback to TTR)
    if result.actions:
        metrics["type_token_ratio"] = self._compute_ttr(result.actions)

    # Tier 3 (auto-detected when Embodiment ships)
    if hasattr(result.introspector, "embodiment_stats"):
        metrics.update(result.introspector.embodiment_stats())

    return metrics
```

---

## Output Format

### Terminal

```
============================================================
  BENCHMARK REPORT — cognitive_suite_v1
  Models: mistral-7b, qwen2.5-14b, llama-3-8b
  Scenarios: 6 | Runs per model: 3
============================================================

  TIER 1 — LLM BEHAVIOR
    hallucination_rate        mistral-7b: 0.38  qwen-14b: 0.43  llama-8b: 0.21
    correct_tool_usage        mistral-7b: 0.62  qwen-14b: 0.57  llama-8b: 0.79
    think_before_act_rate     mistral-7b: 0.14  qwen-14b: 0.00  llama-8b: 0.29

  TIER 2 — COGNITIVE ARCHITECTURE
    memory_formation_rate     mistral-7b: 2.00  qwen-14b: 1.43  llama-8b: 1.86
    recall_Verath             mistral-7b: 1.00  qwen-14b: 1.00  llama-8b: 0.67
    associative_graph_density mistral-7b: 1.40  qwen-14b: 0.80  llama-8b: 1.60
    causal_link_count         mistral-7b: 8     qwen-14b: 3     llama-8b: 12
    learning_efficiency       mistral-7b: 0.62  qwen-14b: 0.21  llama-8b: 0.86
    engagement_depth          mistral-7b: 3.8   qwen-14b: 1.2   llama-8b: 4.1

  TIER 3 — EMBODIMENT
    (not active — Embodiment Core not installed)

  EXPECTATIONS
    passed: 14/18  mistral-7b: 5/6  qwen-14b: 3/6  llama-8b: 6/6

  OVERALL RANKING (weighted composite)
    1. llama-3-8b       score: 0.82  (T1: 0.79  T2: 0.86)
    2. mistral-7b       score: 0.68  (T1: 0.62  T2: 0.74)
    3. qwen2.5-14b      score: 0.45  (T1: 0.57  T2: 0.33)
============================================================
```

### Persisted

```
data/benchmarks/{timestamp}/
  benchmark_report.json       # Full BenchmarkReport
  summary.md                  # Markdown table
  per_model/
    mistral-7b/
      run_1/                  # Standard sim_reports structure
      run_2/
      aggregated.json         # Mean metrics across runs
    qwen2.5-14b/
      ...
  comparison.json             # Cross-model diff
  paper.md                    # (if --write-paper)
```

---

## Tier 3: Embodiment Metrics (Future)

> **Implementation details in [embodiment_core_plan.md](embodiment_core_plan.md).** Interface defined here.

| Metric | Expected source |
|--------|-----------------|
| **Pain sigma (σ)** | `σ(pain_intensities)` — lower = more consistent grounding |
| **Forward model MAE** | Cerebellum predicted vs actual percepts |
| **Body-state coherence** | % of percepts within ATL body-part constraints |
| **Grounding hit rate** | ATL queries per percept / total percepts |
| **Failure composition depth** | Max chain length of composed failures |
| **Cerebellum coverage** | Actions with forward model / total motor actions |
| **LLM fallback rate** | LLM percept calls / total — should decrease over time |

Auto-detected: `if hasattr(introspector, "embodiment_stats")` — no benchmark code changes when Embodiment ships.

---

## Benchmark Scenarios

### Tier 1

**`tool_discovery.yaml`** — Ambiguous percepts testing tool compliance vs hallucination.

**`reasoning_chain.yaml`** — Requires recall → think → act chains.

### Tier 2

**`causal_learning.yaml`** — Repeated cause-effect patterns. Measures NAc link formation, confidence growth, causal diversity.

**`aversion_learning.yaml`** — 3-phase: present danger → fire pain → re-present. Measures pain avoidance learning.

**`concept_formation.yaml`** — Rich narrative with named entities. Measures ATL concept formation from hippocampal episodes.

### Suites

**`cognitive_suite.yaml`** — Full architecture test (all scenarios, weighted).

**`quick_check.yaml`** — 3-turn smoke test (~30s per model).

**`stress_test.yaml`** — 20+ turns, context window pressure.

---

## Implementation Phases

| Phase | What | LOC | Depends on |
|-------|------|-----|-----------|
| **0a** | Enrich SimulationResult at shutdown | ~30 | — |
| **0b** | Activate full bio-stack in sim mode (DN, PainDetector) | ~50 | — |
| **0c** | Bio-system expectation types in validation.py | ~80 | 0a |
| **0d** | Scenario metadata (tags, difficulty, subsystems) | ~20 | — |
| **0e** | JSON compliance counter | ~10 | — |
| **0f** | AUTIntrospector.benchmark_snapshot() | ~30 | 0b |
| **1** | BenchmarkRunner + metric computation (wraps run_campaign) | ~250 | Phase 0 |
| **2** | `--sim benchmark` CLI + model sweep loop | ~80 | Phase 1 |
| **3** | Unified YAML loader (benchmark/suite/metadata sections) | ~80 | Phase 1 |
| **4** | Terminal output (tiered, live progress) + JSON/markdown persistence | ~120 | Phase 1 |
| **5** | Create benchmark scenarios (cognitive_suite, quick_check, causal/aversion/concept) | ~200 (YAML) | Phase 3 |
| **6** | Baseline comparison (`--baseline`) | ~50 | Phase 4 |
| **7** | Research protocol integration (`--write-paper`) | ~40 | Phase 4 |
| **8** | Narrative transcriber (small tier LLM, engagement cascade) | ~100 | Lane tiers |
| **9** | Tier 3 hooks (interface only — implementation with Embodiment Core) | ~20 | Phase 1 |
| | **Total** | **~1,130** | |

### Recommended sessions

**Session 1 — Sim foundations (Phase 0, ~220 LOC):**
Enrich SimulationResult + activate bio-stack + bio expectations + benchmark_snapshot(). These improvements benefit all sim modes immediately.

**Session 2 — Core benchmark (Phases 1-2, ~330 LOC):**
BenchmarkRunner wrapping `run_campaign()` + CLI. End-to-end `maxim --sim benchmark` working.

**Session 3 — Scenarios + output (Phases 3-5, ~400 LOC + YAML):**
Unified loader + tiered terminal output + live progress + actual scenarios. First real benchmark runs.

**Session 4 — Polish (Phases 6-9, ~210 LOC):**
Baseline comparison + paper generation + narrative transcriber + Tier 3 hooks.

## Open Questions

1. **Persona for benchmark orchestrator?** Recommendation: `campaign` (hands-off delivery).

2. **Self-hosted model swapping?** Runner calls `peer llm` between models. Cloud models use API profiles.

3. **Sequential or parallel runs?** Sequential by default. Parallel as future optimization.

4. **Cloud cost control?** `--cloud-budget` per model (default $0.50).

5. **Tier weighting?** Equal T1:T2 by default. `--tier-weights 0.4,0.6` for tuning.

6. **Tier 3 pre-embodiment?** Omitted entirely. Composite = weighted(T1, T2).

7. **Scene elements required?** No — fall back to TTR if unannotated.

## Prerequisites

- **[Lane tier architecture](../archive/lane_tier_plan.md)** — small tier for narrative transcriber. Benchmark runner swaps `large` model while `small` stays constant. Required for Phase 8 (transcriber), not for Phases 0-7.

## Related Plans

- [Lane tier plan](../archive/lane_tier_plan.md) — small tier for narrative transcription
- [Tool refinement plan](tool_refinement_plan.md) — hallucination tracking feeding Tier 1
- [Embodiment core plan](embodiment_core_plan.md) — Tier 3 metric sources
- [Generative campaign plan](generative_campaign_plan.md) — auto-generated benchmark scenarios
