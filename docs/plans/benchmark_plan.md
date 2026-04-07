# Simulation Benchmark Plan — Multi-Model Comparative Testing

## Context

The tool refactoring work (2026-04-06) revealed that model choice dramatically affects AUT behavior. Mistral-7B discovered `think` and attempted `remember`; Qwen 14B got stuck in a `respond` loop. Tool hallucination rates, recall fidelity, and narrative engagement all vary per model. Currently, comparing models requires manually re-running sims and reading through logs.

This plan adds a `maxim --sim benchmark` subcommand that automates multi-model comparison, computes standardized metrics, and outputs a comparative report — reusing the existing research protocol, campaign YAML, and experiment recording infrastructure.

## What Already Exists

| Component | Reuse for benchmarks |
|-----------|---------------------|
| `--aut-model` flag | Per-run model selection (creates separate AUT router) |
| Campaign YAML + expectations | Standardized test scenarios with pass/fail criteria |
| `SimulationReport` | Tool usage, success rates, cost, timing, AUT cognitive state |
| `ExperimentLog` | UMR-tracked experiment recording with metrics dict |
| `validation.py` expectations | action_count_range, tool_success_rate, response_latency_ms |
| `TOOL_ALIASES` + `alias_redirects` | Hallucination rate tracking per model |
| `sweep` persona | Systematic boundary exploration (could drive benchmark probes) |
| `researcher` persona | Evidence-based experiment flow |
| Research protocol (Writer + Reviewer) | Auto-generate comparative paper from results |
| NAc causal links | Learning efficiency metric |
| Hippocampus memory count | Memory formation metric |

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

## Benchmark Metrics

### Repo-Specific (Cognitive Architecture)

These test whether the AUT's biological subsystems work correctly with a given model.

| Metric | What it measures | How to compute |
|--------|-----------------|----------------|
| **Memory recall success** | Did the AUT recall the seed detail when prompted? | Check hippocampus for expected content at recall turn |
| **Behavioral recall** | Did the AUT *act* on the recalled memory (not just report it)? | Check for `say("Verath")` (not `respond("Verath")`) at the door |
| **Tool hallucination rate** | % of tool calls that were unregistered names | `alias_redirects + failed_unregistered / total_calls` |
| **Alias redirect rate** | % of calls that needed alias resolution | `len(alias_redirects) / total_calls` |
| **Correct tool usage rate** | % of calls to tools the model chose from the available list | `1 - hallucination_rate` |
| **NAc learning efficiency** | How many causal links formed per action | `causal_links / total_actions` |
| **Think-before-act rate** | % of turns where `think` preceded another action | Count `think → X` sequences |
| **Memory formation rate** | Episodic memories per turn | `aut_memories_formed / turns` |
| **Narrative engagement** | Did the AUT respond to scene content (not just repeat instructions)? | Semantic diversity of responses across turns |
| **Interference resistance** | Did the seed memory survive interference turns? | Check hippocampus for seed content after interference phase |

### Industry-Standard

These are model capability metrics that apply to any agent system.

| Metric | What it measures | How to compute |
|--------|-----------------|----------------|
| **Instruction following** | Did the model use tools from the available list? | `correct_tool_usage_rate` (inverse of hallucination) |
| **JSON compliance** | % of LLM responses that parsed as valid JSON on first try | Track parse success in router |
| **Context retention** | Does the model retain information across turns? | Verath recall at turn 6 (after 3 interference turns) |
| **Action latency** | Time from percept to action (p50, p95) | Timestamps on bridge send → action record |
| **Token efficiency** | Actions per 1K tokens consumed | `total_actions / (total_tokens / 1000)` |
| **Cost per turn** | USD per simulation turn | `cost_usd / turns` |
| **Reasoning depth** | Does the model chain actions (think → recall → act)? | Detect multi-step action chains within a turn |

## Benchmark Campaign Format

Extends the existing campaign YAML with benchmark-specific metadata:

```yaml
name: cognitive_suite_v1
type: benchmark
description: |
  Comprehensive cognitive architecture benchmark.
  Tests memory recall, tool usage, reasoning, and narrative engagement.

# Models to test (can be overridden by --models CLI flag)
default_models:
  - mistral-7b
  - qwen2.5-14b

# Scenarios to run (in order)
scenarios:
  - path: scenarios/experiments/hippocampal_recall_short.yaml
    weight: 2.0  # counts double in overall score
    category: memory
    metrics:
      - memory_recall_success  # did Verath survive?
      - behavioral_recall      # did AUT say("Verath") at the door?
      - interference_resistance

  - path: scenarios/benchmarks/tool_discovery.yaml
    weight: 1.0
    category: tool_usage
    metrics:
      - tool_hallucination_rate
      - alias_redirect_rate
      - correct_tool_usage_rate

  - path: scenarios/benchmarks/reasoning_chain.yaml
    weight: 1.5
    category: reasoning
    metrics:
      - think_before_act_rate
      - reasoning_depth

  - path: scenarios/benchmarks/narrative_engagement.yaml
    weight: 1.0
    category: engagement
    metrics:
      - narrative_engagement
      - memory_formation_rate

# Scoring thresholds for pass/fail
scoring:
  memory_recall_success: { pass: 1.0 }           # binary: recalled or not
  behavioral_recall: { pass: 1.0 }                # binary: said it or not
  tool_hallucination_rate: { pass_below: 0.3 }    # <30% hallucinated
  correct_tool_usage_rate: { pass_above: 0.7 }    # >70% correct
  think_before_act_rate: { pass_above: 0.2 }      # >20% of turns
  interference_resistance: { pass: 1.0 }          # binary
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
  │     │     │     ├── collect SimulationReport + executor.alias_redirects
  │     │     │     └── compute per-run metrics
  │     │     └── aggregate across runs (mean, stddev)
  │     ├── compute comparative metrics
  │     ├── score against thresholds
  │     └── build BenchmarkReport
  ├── _compute_metrics(report, executor, hippocampus) → ModelMetrics
  ├── _score(metrics, thresholds) → ModelScore
  └── _compare(scores, baseline) → ComparisonTable
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
    metrics: dict[str, float]        # aggregated (mean)
    metrics_stddev: dict[str, float] # variance across runs
    score: float                     # weighted composite
    passed: bool                     # met all pass thresholds
    expectations_met: int
    expectations_total: int
```

### Integration with Research Protocol

The benchmark runner can optionally feed results into the research protocol's Writer + Reviewer:

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
  Scenarios: 4 | Runs per model: 3
============================================================

  MEMORY
    memory_recall_success     mistral-7b: 1.00  qwen-14b: 1.00  llama-8b: 0.67
    behavioral_recall         mistral-7b: 0.33  qwen-14b: 0.00  llama-8b: 0.33
    interference_resistance   mistral-7b: 1.00  qwen-14b: 1.00  llama-8b: 1.00

  TOOL USAGE
    hallucination_rate        mistral-7b: 0.38  qwen-14b: 0.43  llama-8b: 0.21
    correct_tool_usage        mistral-7b: 0.62  qwen-14b: 0.57  llama-8b: 0.79
    alias_redirect_rate       mistral-7b: 0.25  qwen-14b: 0.29  llama-8b: 0.14

  REASONING
    think_before_act_rate     mistral-7b: 0.14  qwen-14b: 0.00  llama-8b: 0.29

  EFFICIENCY
    cost_per_turn             mistral-7b: $0.00  qwen-14b: $0.00  llama-8b: $0.00
    actions_per_turn          mistral-7b: 1.86  qwen-14b: 2.00  llama-8b: 1.57
    latency_p50_ms            mistral-7b: 2800   qwen-14b: 2500   llama-8b: 3100

  OVERALL RANKING
    1. llama-3-8b       score: 0.78
    2. mistral-7b       score: 0.65
    3. qwen2.5-14b      score: 0.52
============================================================
```

### Persisted Files

```
data/benchmarks/{timestamp}/
  benchmark_report.json     # Full BenchmarkReport
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

## Benchmark Scenarios to Create

### 1. `cognitive_suite.yaml` — Full cognitive architecture test

Combines all existing experiment scenarios into a single benchmark:
- Hippocampal recall (short) — memory formation + recall under interference
- Tool discovery — are narrative/introspection tools used correctly?
- Reasoning chain — does think→recall→act chaining work?
- Narrative engagement — does the AUT respond to scene content?

### 2. `quick_check.yaml` — Fast smoke test for new models

Minimal 3-turn scenario: seed → interference → recall. Takes ~30s per model. Good for quick validation when a new model drops.

### 3. `instruction_following.yaml` — Industry-standard tool compliance

Tests whether the model reads and follows the available tool list. Deliberately ambiguous percepts that could go to any tool — measures whether the model hallucinates or picks from the list.

### 4. `stress_test.yaml` — Context window pressure

Long campaign (20+ turns) that tests context retention, memory formation under load, and cost efficiency. Useful for comparing 7B vs 14B vs 70B models on the same narrative.

## Implementation Phases

| Phase | What | LOC | Depends on |
|-------|------|-----|-----------|
| 1 | `BenchmarkRunner` class + `ModelMetrics` computation | ~200 | Existing sim infrastructure |
| 2 | `--sim benchmark` CLI integration + model sweep loop | ~80 | Phase 1 |
| 3 | Benchmark YAML format + scenario loader | ~60 | Phase 1 |
| 4 | Terminal output + JSON/markdown persistence | ~100 | Phase 1 |
| 5 | Create benchmark scenarios (cognitive_suite, quick_check) | ~150 (YAML) | Phase 3 |
| 6 | Baseline comparison (`--baseline`) | ~50 | Phase 4 |
| 7 | Research protocol integration (`--write-paper`) | ~40 | Phase 4 |
| **Total** | | **~680** | |

**Recommended first session:** Phases 1-2 (~280 LOC) — the runner + CLI. This gives you `maxim --sim benchmark --models X,Y --campaign Z` working end-to-end. Phases 3-7 add polish, scenarios, and paper generation.

## Open Questions

1. **Should benchmark runs use the `researcher` or `sweep` persona for the orchestrator?**
   - `researcher` follows a structured hypothesis→experiment→conclusion flow
   - `sweep` does systematic boundary exploration
   - For benchmarks, the orchestrator isn't probing — it's just delivering campaigns. The `campaign` persona (which stays hands-off) may be best.
   - Recommendation: `campaign` persona (no orchestrator probing, just deliver and measure)

2. **How to handle model loading for self-hosted models?**
   - `--aut-model mistral-7b` requires the leader to have that model available
   - `maxim peer llm mistral-7b` hot-swaps the leader's model, but only one at a time
   - Benchmark would need to swap models between runs: `peer llm model_A → run → peer llm model_B → run`
   - For cloud models (Claude, GPT-4), no swap needed — just different API profiles
   - Recommendation: benchmark runner calls `peer llm` between self-hosted model runs, uses profiles for cloud models

3. **Should multiple runs be sequential or parallel?**
   - Sequential: simpler, one GPU at a time, deterministic ordering
   - Parallel: faster but needs multiple GPU slots or cloud dispatch
   - Recommendation: sequential by default, parallel as future optimization

4. **How to handle cloud model costs?**
   - Claude-sonnet at ~$3/Mtok could get expensive with 3 runs × 4 scenarios
   - Recommendation: `--cloud-budget` cap applies per model. Default budget of $0.50 per benchmark model. Skip remaining scenarios if budget exceeded.

5. **Should the benchmark report feed back into the tool alias map?**
   - New hallucinated tool names discovered during benchmarks could auto-update TOOL_ALIASES
   - Risk: auto-updating could add bad mappings
   - Recommendation: report new hallucinations in the benchmark output with suggested aliases, but require manual review before adding to the map

## Related Plans

- [Tool refactoring plan](tool_refactoring_plan.md) — tool aliases and hallucination tracking that feed into benchmark metrics
- [Realtime refinement plan](realtime_refinement_plan.md) — refinement persona and metric expectations that benchmarks reuse
- [Research protocol plan](research_protocol_plan.md) — Writer + Reviewer pipeline for benchmark papers
- [Generative campaign plan](generative_campaign_plan.md) — LLM-generated campaigns could auto-create benchmark scenarios
