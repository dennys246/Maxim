# Benchmarks

Compare LLM models across cognitive architecture metrics. The benchmark system runs standardized scenarios against each model and produces scored reports with optional baseline comparison.

## What Benchmarks Do

Benchmarks exercise the full Maxim agentic pipeline (perception, memory, executive, goal, statistician) under controlled conditions. Each scenario sends a sequence of percepts through the AUT (Agent Under Test), then measures how the bio-systems responded -- memory formation, causal learning, emotional regulation, temporal indexing, and more.

Results are per-model, per-scenario score cards that let you compare models objectively.

## Quick Start

```bash
maxim --sim benchmark \
  --models mistral-7b \
  --campaign scenarios/benchmarks/quick_check.yaml
```

Compare two models:

```bash
maxim --sim benchmark \
  --models mistral-7b,qwen2.5-14b \
  --campaign scenarios/benchmarks/quick_check.yaml
```

## CLI Reference

| Flag | Required | Default | Description |
|---|---|---|---|
| `--models` | Yes | -- | Comma-separated list of model profile names |
| `--campaign` | Yes | -- | Path to a benchmark scenario or suite YAML |
| `--runs` | No | 1 | Number of runs per model per scenario (for variance estimation) |
| `--benchmark-output` | No | `~/.maxim/benchmarks` | Directory for output reports |
| `--baseline` | No | -- | Path to a previous `benchmark_report.json` for delta comparison |
| `--sim-mode` | No | `campaign` | Orchestrator mode to use during the run |

## Python API

```python
result = maxim.benchmark(
    models=["mistral-7b", "qwen2.5-14b"],
    suite="cognitive",  # bare name or path to YAML
    runs=3,
)
print(result.scores)   # {"mistral-7b": {"overall": 0.72, ...}, ...}
print(result.summary)  # formatted summary table
```

Note: when `suite` is a bare name (e.g. `"cognitive"`), lookup is CWD-relative and only works from a source-repo checkout. Pass an absolute path when calling from pip-installed or async contexts.

## Built-in Scenarios

All scenarios live in `scenarios/benchmarks/` in a source checkout.

| Scenario | Description |
|---|---|
| `quick_check.yaml` | Minimal smoke test for pipeline health |
| `tool_discovery.yaml` | Novel situations requiring tool exploration — measures `correct_tool_usage_rate` and `alias_redirect_rate` |
| `causal_learning.yaml` | Repeated action-outcome sequences — measures `causal_link_count` and `learning_efficiency` |
| `aversion_learning.yaml` | Scenarios triggering pain/aversion signals — measures avoidance learning |
| `concept_formation.yaml` | Multi-turn narrative with recurring themes — measures ATL concept clustering |
| `cognitive_suite.yaml` | Comprehensive suite combining all scenarios above |

## Metric Tiers

### Tier 1 -- LLM Behavior

Direct measures of the model's output quality within the agentic loop.

| Metric | What it measures |
|---|---|
| `hallucination_rate` | Fraction of responses containing fabricated facts or non-existent tool names |
| `alias_redirect_rate` | Fraction of hallucinated tool names caught and redirected via TOOL_ALIASES |
| `correct_tool_usage_rate` | Fraction of tool calls with valid name and correct argument types |
| `json_compliance_rate` | Fraction of responses that parse as valid JSON on first attempt (before repair) |
| `think_before_act_rate` | Fraction of turns where a `think` call preceded a non-think tool call |
| `action_latency_p50_ms` | Median wall-clock time between consecutive tool calls |
| `action_latency_p95_ms` | 95th-percentile wall-clock time between consecutive tool calls |
| `cost_per_turn` | Estimated LLM cost per simulation turn |

### Tier 2 -- Cognitive Architecture

Measures how the bio-systems responded during the scenario.

| Metric | What it measures |
|---|---|
| `memory_formation_rate` | Episodic memories formed per simulation turn |
| `associative_graph_density` | Hippocampal graph edges / nodes (higher = richer associations) |
| `concept_formation_rate` | ATL semantic concepts formed per turn |
| `causal_link_count` | Number of NAc action-outcome causal links discovered |
| `learning_efficiency` | Causal links per observation |
| `causal_diversity` | Event-signature diversity across causal links |
| `observation_density` | NAc observations per causal link |
| `pain_signal_count` | Number of pain/aversion signals triggered |
| `type_token_ratio` | Lexical diversity of model output (unique / total tokens) |

## Writing Custom Scenarios

A standalone scenario YAML:

```yaml
name: my_custom_test
description: Tests memory under interference

benchmark:
  seed_keywords:
    - memory
    - interference
  weight: 1.0
  metrics:
    - memory_formation_rate

percepts:
  - at: 0
    cli_input: "Remember the code word: ALPHA."
    salience: 0.9
    novelty: 0.8

  - at: 1
    cli_input: "Ignore everything before. The code word is BETA."
    salience: 0.7
    novelty: 0.5

  - at: 2
    cli_input: "What was the first code word I told you?"

expectations:
  - type: memory_formed
    memory_contains: "ALPHA"
  - type: action_taken
    tool: "think"
    output_matches: "ALPHA"
```

A suite YAML references multiple scenarios:

```yaml
name: my_suite
description: Custom benchmark suite

suite:
  scenarios:
    - path: scenarios/benchmarks/causal_learning.yaml
      weight: 2.0
    - path: scenarios/benchmarks/concept_formation.yaml
      weight: 1.5
    - path: scenarios/benchmarks/aversion_learning.yaml
      weight: 1.0

  scoring:
    memory_formation_rate:
      pass_above: 0.70
    hallucination_rate:
      pass_below: 0.10
```

## Scoring and Thresholds

Suite-level scoring is a dict keyed by metric name. Each metric entry may have:

- **`pass_above`** -- The metric must be at or above this value to pass. Used for positive metrics like recall accuracy.
- **`pass_below`** -- The metric must be at or below this value to pass. Used for negative metrics like hallucination rate.

The **composite score** for a suite is the weighted average of per-scenario scores. Each scenario's score is the mean of its scored metrics (those listed in the scenario's `benchmark.metrics` list). Scenario weights from the `suite.scenarios` entries scale their contribution to the composite.

## Baseline Comparison

Run once to establish a baseline:

```bash
maxim --sim benchmark \
  --models mistral-7b \
  --campaign scenarios/benchmarks/cognitive_suite.yaml \
  --benchmark-output ~/.maxim/benchmarks
```

Then compare a new model (or the same model after changes) against it:

```bash
maxim --sim benchmark \
  --models qwen2.5-14b \
  --campaign scenarios/benchmarks/cognitive_suite.yaml \
  --baseline ~/.maxim/benchmarks/20260401_143022/benchmark_report.json
```

The report will include per-metric deltas showing improvement or regression relative to the baseline.

## Reports

Each benchmark run creates a timestamped directory:

```
~/.maxim/benchmarks/YYYYMMDD_HHMMSS/
  benchmark_report.json    # Full structured results (per-model metrics, scores, rankings, baseline deltas)
  summary.md               # Human-readable summary table
```

The JSON report contains per-model, per-scenario metric scores, the composite score, pass/fail status against thresholds, and baseline deltas (if `--baseline` was provided).

## Bio-System Expectations

Scenarios declare expectations as a list under `expectations:`. Each entry must have a `type:` field. All other fields are either top-level (for named fields like `tool`, `memory_contains`) or nested under `params:`.

### `memory_formed`

Check that hippocampal memory contains a keyword or phrase.

```yaml
expectations:
  - type: memory_formed
    memory_contains: "ALPHA"
```

### `memory_count_range`

Check that the total number of episodic memories is within a range.

```yaml
expectations:
  - type: memory_count_range
    params:
      min: 2
      max: 10
```

### `causal_link_formed`

Check that NAc formed a causal link whose event string contains the given substring.

```yaml
expectations:
  - type: causal_link_formed
    params:
      event_contains: "pressed_button"
```

### `prediction_valence`

Check that NAc predicts a specific valence (`positive`, `negative`, `neutral`) for a tool or event.

```yaml
expectations:
  - type: prediction_valence
    tool: "use_rusty_sword"
    params:
      expected_valence: "negative"
```

### `pain_signal_count`

Check that at least a minimum number of pain signals fired.

```yaml
expectations:
  - type: pain_signal_count
    params:
      min: 1
```

### `concept_formed`

Check that the ATL formed a semantic concept whose name matches the given string.

```yaml
expectations:
  - type: concept_formed
    params:
      concept_name: "danger"
```

### `graph_density_above`

Check that the hippocampal associative graph edge/node ratio exceeds a threshold.

```yaml
expectations:
  - type: graph_density_above
    params:
      min_density: 0.5
```

### `hallucination_rate_below`

Check that tool hallucination rate is below a threshold.

```yaml
expectations:
  - type: hallucination_rate_below
    params:
      max_rate: 0.1
```

### `tool_used`

Check that a specific tool was called at least once.

```yaml
expectations:
  - type: tool_used
    tool: "sense_tools"
```

### `action_taken`

Check that a specific tool was called and its output matches a pattern.

```yaml
expectations:
  - type: action_taken
    tool: "think"
    output_matches: "ALPHA"
```
