# Simulation Benchmark Plan — Multi-Model Comparative Testing

## Context

The tool refactoring work (2026-04-06) revealed that model choice dramatically affects AUT behavior. Mistral-7B discovered `think` and attempted `remember`; Qwen 14B got stuck in a `respond` loop. Tool hallucination rates, recall fidelity, and narrative engagement all vary per model. Currently, comparing models requires manually re-running sims and reading through logs.

This plan adds a `maxim --sim benchmark` subcommand that automates multi-model comparison, computes standardized metrics across three tiers (LLM behavior, cognitive architecture, and embodiment), and outputs a comparative report — reusing the existing research protocol, campaign YAML, `AUTIntrospector`, and experiment recording infrastructure.

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

## What Already Exists

| Component | Reuse for benchmarks |
|-----------|---------------------|
| `--aut-model` flag | Per-run model selection (creates separate AUT router) |
| Campaign YAML + expectations | Standardized test scenarios with pass/fail criteria |
| `SimulationReport` | Tool usage, success rates, cost, timing, AUT cognitive state |
| `ExperimentLog` | UMR-tracked experiment recording with metrics dict |
| `AUTIntrospector` | Programmatic access to all cognitive subsystems (just shipped) |
| `Executor.tool_usage_stats()` | Hallucination rate, alias redirects, tools attempted/succeeded |
| `run_campaign()` | Standalone experiment runner with structured results (just shipped) |
| `validation.py` expectations | action_count_range, tool_success_rate, response_latency_ms |
| `TOOL_ALIASES` + `alias_redirects` | Hallucination rate tracking per model |
| Research protocol (Writer + Reviewer) | Auto-generate comparative paper from results |

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
```

---

## Tier 1: LLM Behavior Metrics

These measure the LLM's capability as an agent — tool compliance, reasoning, efficiency. Applicable to any agent system.

| Metric | What it measures | How to compute |
|--------|-----------------|----------------|
| **Tool hallucination rate** | % of tool calls to unregistered names | `executor.tool_usage_stats()["hallucination_rate"]` |
| **Alias redirect rate** | % of calls that needed alias resolution | `len(alias_redirects) / total_calls` |
| **Correct tool usage rate** | % of calls from the available list | `1 - hallucination_rate` |
| **Instruction following** | Did the model use tools from the list? | Same as correct tool usage |
| **JSON compliance** | % of LLM responses that parsed as valid JSON on first try | Track parse success in router |
| **Think-before-act rate** | % of turns where `think` preceded another action | Count `think → X` sequences in action history |
| **Reasoning depth** | Does the model chain actions (think → recall → act)? | Detect multi-step chains within a turn |
| **Action latency** | Time from percept to action (p50, p95) | Timestamps on bridge send → action record |
| **Token efficiency** | Actions per 1K tokens consumed | `total_actions / (total_tokens / 1000)` |
| **Cost per turn** | USD per simulation turn | `cost_usd / turns` |

**Data sources:** `Executor.tool_usage_stats()`, `SimulationReport`, `ActionRecord` timestamps.

---

## Tier 2: Cognitive Architecture Metrics

These measure whether the bio-inspired subsystems are functioning correctly and producing emergent cognitive behavior. A model that aces Tier 1 but fails Tier 2 is gaming the tools without learning.

### Memory System (Hippocampus + ATL + EC)

| Metric | Subsystem | How to compute |
|--------|-----------|----------------|
| **Memory formation rate** | Hippocampus | `introspector.system_stats()["hippocampus_memories"] / turns` |
| **Memory recall success** | Hippocampus | `introspector.memory_recall(keyword)["count"] > 0` |
| **Behavioral recall** | Hippocampus | Did AUT `say("Verath")` (not `respond("Verath")`) at the door? Check action history |
| **Interference resistance** | Hippocampus | Seed content survives after interference turns |
| **Associative graph density** | Hippocampus | `graph_edges / graph_nodes` from hippocampus stats — denser = more interconnected memories |
| **Concept formation rate** | ATL | `introspector.system_stats()["atl_concepts"] / turns` |
| **Concept avg confidence** | ATL | Mean confidence across ATL concepts via `introspector.concept_query(limit=100)` |
| **Episodic→semantic promotion** | MemoryHub | Count of ATL concepts with hippocampal source — measures whether episodic memories consolidate into knowledge |

### Causal Learning (NAc)

| Metric | Subsystem | How to compute |
|--------|-----------|----------------|
| **Causal link count** | NAc | `introspector.system_stats()["nac_causal_links"]` |
| **Learning efficiency** | NAc | `causal_links / total_actions` — links formed per action |
| **Causal diversity** | NAc | Unique event_signatures / total links — breadth of learning |
| **Confidence growth** | NAc | Mean link.confidence at campaign end (higher = stronger learning) |
| **Prediction accuracy** | NAc | For known dangerous tools: does `predict_outcome("delete_file")` return negative valence? |

### Aversive Learning (Pain + NAc)

| Metric | Subsystem | How to compute |
|--------|-----------|----------------|
| **Pain signal count** | PainDetector | `pain_history()["pain_memory_count"]` |
| **Aversion learning speed** | NAc + Pain | After N pain signals from the same source, does NAc predict negative outcome? Measure N |
| **Pain avoidance rate** | Actions + Pain | % of turns where AUT avoids previously painful actions |

### Temporal & Engagement

| Metric | Subsystem | How to compute |
|--------|-----------|----------------|
| **Temporal pattern count** | SCN | `len(scn.find_rhythmic_patterns())` via introspector |
| **Narrative engagement** | Actions | Semantic diversity of responses — does the AUT respond to scene content or repeat instructions? |

**Data sources:** `AUTIntrospector.full_analysis()`, `AUTIntrospector.system_stats()`, subsystem `.stats()` methods. All accessible programmatically post-run — no new instrumentation needed.

---

## Tier 3: Embodiment Metrics (Future)

> **Implementation details live in [embodiment_core_plan.md](embodiment_core_plan.md).** This section defines the benchmark interface — what we'll measure and how the runner will collect it. The actual metric computation will be added when Embodiment Core ships.

Embodiment metrics test whether the LLM produces physically plausible percepts and whether the bio-systems ground body-state consistently. The Embodiment Core plan's success criterion (pain intensity σ ≥50% lower with ATL grounding) is itself a benchmark metric.

### Planned metrics (interface only — computed when Embodiment ships)

| Metric | What it tells us | Expected source |
|--------|-----------------|-----------------|
| **Pain sigma (σ)** | Consistency of pain signals with ATL grounding | `σ(pain_intensities)` — lower = more consistent |
| **Forward model MAE** | Cerebellum prediction accuracy over time | Mean absolute error: predicted vs actual percepts |
| **Body-state coherence** | Are LLM-generated percepts physically plausible? | % of percepts within ATL body-part constraints (joint ranges, torque curves) |
| **Grounding hit rate** | How often does ATL grounding override raw LLM output? | ATL queries per percept / total percepts |
| **Failure composition depth** | Do structured failures build on each other? | Max chain length of composed failure events |
| **Cerebellum coverage** | What % of motor actions have learned forward models? | Actions with Cerebellum prediction / total motor actions |
| **LLM fallback rate** | How often does the system need the LLM because no forward model exists? | LLM percept calls / total percept calls — should decrease over time |

### How the runner will integrate

```python
# In BenchmarkRunner._compute_metrics():
if hasattr(introspector, 'embodiment_stats'):
    tier3 = introspector.embodiment_stats()
    # pain_sigma, forward_model_mae, body_state_coherence, etc.
else:
    tier3 = {}  # Pre-embodiment: Tier 3 is empty, not failed
```

The benchmark report will show Tier 3 metrics only when Embodiment is active. Pre-embodiment runs will simply omit the section rather than reporting zeroes, so baselines remain clean.

### Embodiment-specific scenarios (to be created with Embodiment Core)

These will live in `scenarios/benchmarks/` alongside Tier 1-2 scenarios:

- **Grounding consistency**: repeated motor commands, measuring pain sigma reduction
- **Forward model learning curve**: 50+ motor actions, tracking Cerebellum MAE over time
- **Body-state violation**: LLM-generated percepts that violate joint limits, measuring correction rate
- **Aversion transfer**: pain learned in narrative context (DM campaign) transfers to embodied context (robot)

---

## Benchmark Campaign Format

Extends the existing campaign YAML with benchmark-specific metadata and tier indicators:

```yaml
name: cognitive_suite_v1
type: benchmark
description: |
  Comprehensive cognitive architecture benchmark.
  Tests LLM behavior (Tier 1) and bio-system dynamics (Tier 2).

# Models to test (can be overridden by --models CLI flag)
default_models:
  - mistral-7b
  - qwen2.5-14b

# Scenarios to run (in order)
scenarios:
  - path: scenarios/experiments/hippocampal_recall_short.yaml
    weight: 2.0
    category: memory
    tier: [1, 2]
    metrics:
      # Tier 1
      - behavioral_recall
      # Tier 2
      - memory_recall_success
      - interference_resistance
      - memory_formation_rate
      - associative_graph_density

  - path: scenarios/benchmarks/tool_discovery.yaml
    weight: 1.0
    category: tool_usage
    tier: [1]
    metrics:
      - tool_hallucination_rate
      - alias_redirect_rate
      - correct_tool_usage_rate

  - path: scenarios/benchmarks/causal_learning.yaml
    weight: 1.5
    category: learning
    tier: [2]
    metrics:
      - causal_link_count
      - learning_efficiency
      - causal_diversity
      - confidence_growth

  - path: scenarios/benchmarks/aversion_learning.yaml
    weight: 1.5
    category: safety
    tier: [2]
    metrics:
      - pain_signal_count
      - aversion_learning_speed
      - pain_avoidance_rate

  - path: scenarios/benchmarks/reasoning_chain.yaml
    weight: 1.5
    category: reasoning
    tier: [1]
    metrics:
      - think_before_act_rate
      - reasoning_depth

  - path: scenarios/benchmarks/narrative_engagement.yaml
    weight: 1.0
    category: engagement
    tier: [1, 2]
    metrics:
      - narrative_engagement
      - concept_formation_rate

# Scoring thresholds for pass/fail
scoring:
  # Tier 1
  tool_hallucination_rate: { pass_below: 0.3 }
  correct_tool_usage_rate: { pass_above: 0.7 }
  think_before_act_rate: { pass_above: 0.2 }
  behavioral_recall: { pass: 1.0 }
  # Tier 2
  memory_recall_success: { pass: 1.0 }
  interference_resistance: { pass: 1.0 }
  learning_efficiency: { pass_above: 0.1 }
  concept_formation_rate: { pass_above: 0.0 }  # any concept formation = pass
```

## Architecture

### BenchmarkRunner class

```
src/maxim/simulation/benchmark.py (new)

BenchmarkRunner
  ├── __init__(models, campaign_path, runs, output_dir, baseline)
  ├── run() → BenchmarkReport
  │     ├── for each model:
  │     │     ├── for each run (1..N):
  │     │     │     ├── start_simulation_mode(aut_model=model, campaign=scenario)
  │     │     │     ├── collect introspector + executor stats
  │     │     │     └── compute per-run metrics (all tiers)
  │     │     └── aggregate across runs (mean, stddev)
  │     ├── compute comparative metrics
  │     ├── score against thresholds
  │     └── build BenchmarkReport
  ├── _compute_metrics(introspector, executor) → ModelMetrics
  │     ├── _tier1_metrics(executor) → dict
  │     ├── _tier2_metrics(introspector) → dict
  │     └── _tier3_metrics(introspector) → dict  # empty pre-embodiment
  ├── _score(metrics, thresholds) → ModelScore
  └── _compare(scores, baseline) → ComparisonTable
```

### Metric computation (core logic)

```python
def _compute_metrics(self, introspector, executor) -> ModelMetrics:
    # Tier 1: LLM behavior
    tool_stats = executor.tool_usage_stats()
    tier1 = {
        "hallucination_rate": tool_stats["hallucination_rate"],
        "alias_redirect_rate": ...,
        "correct_tool_usage_rate": 1 - tool_stats["hallucination_rate"],
        "think_before_act_rate": self._count_think_chains(executor),
        "cost_per_turn": ...,
        "token_efficiency": ...,
    }

    # Tier 2: Cognitive architecture
    analysis = introspector.full_analysis(seed_keywords=self._seed_keywords)
    stats = introspector.system_stats()
    tier2 = {
        "memory_formation_rate": stats.get("hippocampus_memories", 0) / max(turns, 1),
        "concept_formation_rate": stats.get("atl_concepts", 0) / max(turns, 1),
        "causal_link_count": stats.get("nac_causal_links", 0),
        "learning_efficiency": stats.get("nac_causal_links", 0) / max(total_actions, 1),
        "pain_signal_count": introspector.pain_history(limit=100)["pain_memory_count"],
        # ... keyword recall, interference resistance from analysis
    }

    # Tier 3: Embodiment (placeholder — filled when embodiment ships)
    tier3 = {}
    if hasattr(introspector, 'embodiment_stats'):
        tier3 = introspector.embodiment_stats()

    return ModelMetrics(tier1=tier1, tier2=tier2, tier3=tier3)
```

### BenchmarkReport

```python
@dataclass
class BenchmarkReport:
    timestamp: str
    campaign: str
    models: list[str]
    runs_per_model: int

    # Per-model results
    results: dict[str, ModelResult]  # model_name → ModelResult

    # Comparative
    rankings: dict[str, list[str]]   # metric_name → [models ranked]
    overall_ranking: list[str]        # weighted composite score

@dataclass
class ModelResult:
    model: str
    runs: list[RunResult]            # individual run data
    tier1: dict[str, float]          # LLM behavior (aggregated mean)
    tier2: dict[str, float]          # Cognitive architecture
    tier3: dict[str, float]          # Embodiment (empty pre-embodiment)
    metrics_stddev: dict[str, float] # variance across runs
    score: float                     # weighted composite
    passed: bool                     # met all pass thresholds
    expectations_met: int
    expectations_total: int
```

### Integration with Research Protocol

```bash
# Benchmark only (fast, metrics + table)
maxim --sim benchmark --models mistral-7b,qwen2.5-14b --campaign ...

# Benchmark + paper (slower, includes analysis)
maxim --sim benchmark --models mistral-7b,qwen2.5-14b --campaign ... --write-paper
```

With `--write-paper`, the benchmark feeds `BenchmarkReport` into the Writer agent as experiment data, producing a comparative research paper with the Reviewer validating claims against the metrics.

## Output Format

### Terminal Output

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
    cost_per_turn             mistral-7b: $0.00  qwen-14b: $0.00  llama-8b: $0.00
    latency_p50_ms            mistral-7b: 2800   qwen-14b: 2500   llama-8b: 3100

  TIER 2 — COGNITIVE ARCHITECTURE
    memory_formation_rate     mistral-7b: 2.00  qwen-14b: 1.43  llama-8b: 1.86
    memory_recall_success     mistral-7b: 1.00  qwen-14b: 1.00  llama-8b: 0.67
    interference_resistance   mistral-7b: 1.00  qwen-14b: 1.00  llama-8b: 1.00
    concept_formation_rate    mistral-7b: 0.29  qwen-14b: 0.00  llama-8b: 0.14
    causal_link_count         mistral-7b: 8     qwen-14b: 3     llama-8b: 12
    learning_efficiency       mistral-7b: 0.62  qwen-14b: 0.21  llama-8b: 0.86
    pain_avoidance_rate       mistral-7b: 0.50  qwen-14b: 0.00  llama-8b: 0.75

  TIER 3 — EMBODIMENT
    (not active — Embodiment Core not installed)

  OVERALL RANKING (weighted composite)
    1. llama-3-8b       score: 0.82  (T1: 0.79  T2: 0.86)
    2. mistral-7b       score: 0.68  (T1: 0.62  T2: 0.74)
    3. qwen2.5-14b      score: 0.45  (T1: 0.57  T2: 0.33)
============================================================
```

### Persisted Files

```
data/benchmarks/{timestamp}/
  benchmark_report.json     # Full BenchmarkReport (all tiers)
  summary.md                # Human-readable markdown table
  per_model/
    mistral-7b/
      run_1/                # Standard sim_reports structure
      run_2/
      run_3/
      aggregated.json       # Mean metrics across runs
    qwen2.5-14b/
      ...
  comparison.json           # Cross-model comparison data
  paper.md                  # (if --write-paper) Comparative analysis
```

## Benchmark Scenarios

### Tier 1 scenarios

#### `tool_discovery.yaml` — Tool compliance
Deliberately ambiguous percepts that could go to any tool. Measures whether the model hallucinates or picks from the available list.

#### `reasoning_chain.yaml` — Multi-step reasoning
Scenario requiring recall → think → act chains. Tests whether the model uses `think` before acting.

### Tier 2 scenarios

#### `causal_learning.yaml` — NAc learning curve
Repeated exposure to cause-effect patterns (touch fire → pain, help NPC → reward). Measures causal link formation rate, confidence growth, and diversity of learned associations.

#### `aversion_learning.yaml` — Pain avoidance
3-phase design: (1) present dangerous actions, (2) fire pain signals when taken, (3) re-present same actions. Measures whether NAc learns to predict negative outcomes and whether the AUT avoids repeating painful actions.

#### `concept_formation.yaml` — Semantic memory
Rich narrative with named entities (characters, places, objects) repeated across turns. Measures whether hippocampal episodes consolidate into ATL semantic concepts.

### Combined scenarios

#### `cognitive_suite.yaml` — Full architecture test
Combines all scenarios into a weighted benchmark. Each scenario targets specific tiers and metrics.

#### `quick_check.yaml` — Fast smoke test
Minimal 3-turn scenario: seed → interference → recall. ~30s per model. Good for quick validation when a new model drops.

#### `stress_test.yaml` — Context window pressure
Long campaign (20+ turns) testing context retention, memory formation under load, and cost efficiency. Useful for comparing 7B vs 14B vs 70B on the same narrative.

## Implementation Phases

| Phase | What | LOC | Depends on |
|-------|------|-----|-----------|
| 1 | `BenchmarkRunner` class + Tier 1-2 metric computation | ~250 | `AUTIntrospector`, `Executor.tool_usage_stats()` |
| 2 | `--sim benchmark` CLI integration + model sweep loop | ~80 | Phase 1 |
| 3 | Benchmark YAML format + scenario loader (tier-aware) | ~60 | Phase 1 |
| 4 | Terminal output (tiered) + JSON/markdown persistence | ~120 | Phase 1 |
| 5 | Create benchmark scenarios (cognitive_suite, quick_check, causal_learning, aversion_learning) | ~200 (YAML) | Phase 3 |
| 6 | Baseline comparison (`--baseline`) | ~50 | Phase 4 |
| 7 | Research protocol integration (`--write-paper`) | ~40 | Phase 4 |
| 8 | Tier 3 hooks (interface only — implementation with Embodiment Core) | ~30 | Phase 1 |
| **Total** | | **~830** | |

**Recommended first session:** Phases 1-2 (~330 LOC) — the runner + CLI with Tier 1-2 metrics. This gives you `maxim --sim benchmark --models X,Y --campaign Z` working end-to-end with cognitive architecture visibility.

**Second session:** Phases 3-5 (~380 LOC) — tiered YAML format + output + actual scenarios.

**Polish:** Phases 6-8 (~120 LOC) — baseline comparison, paper generation, Tier 3 interface.

## Open Questions

1. **Should benchmark runs use the `researcher` or `sweep` persona for the orchestrator?**
   - Recommendation: `campaign` persona (no orchestrator probing, just deliver and measure)

2. **How to handle model loading for self-hosted models?**
   - Benchmark runner calls `peer llm` between self-hosted model runs
   - Cloud models use different API profiles (no swap needed)

3. **Should multiple runs be sequential or parallel?**
   - Recommendation: sequential by default, parallel as future optimization

4. **How to handle cloud model costs?**
   - `--cloud-budget` cap applies per model. Default $0.50 per benchmark model.

5. **Should the benchmark report feed back into the tool alias map?**
   - Recommendation: report new hallucinations with suggested aliases, require manual review

6. **How should Tier 2 metrics be weighted relative to Tier 1?**
   - Recommendation: equal weight by default. A model that's great at tool usage but doesn't drive learning is not a good cognitive architecture model. Override with `--tier-weights 0.4,0.6` for tuning.

7. **Should Tier 3 metrics affect the overall score pre-embodiment?**
   - No. Tier 3 is omitted entirely pre-embodiment. Overall score = weighted(Tier1, Tier2) only. Once Embodiment ships, Tier 3 joins the composite: weighted(T1, T2, T3).

## Related Plans

- [Tool refinement plan](tool_refinement_plan.md) — tool aliases and hallucination tracking that feed into Tier 1 metrics
- Tool refactoring (done, archived) — `Executor.tool_usage_stats()` provides Tier 1 data
- Introspection API (done, archived) — `AUTIntrospector` provides Tier 2 data
- [Embodiment core plan](embodiment_core_plan.md) — defines Tier 3 metric sources and success criteria
- [Generative campaign plan](generative_campaign_plan.md) — LLM-generated campaigns could auto-create benchmark scenarios
- Research protocol (done, archived) — Writer + Reviewer pipeline for `--write-paper`
