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
| `--benchmark-output` | No | `data/benchmarks` | Directory for output reports |
| `--baseline` | No | -- | Path to a previous `benchmark_report.json` for delta comparison |
| `--sim-persona` | No | `adversarial` | Orchestrator persona to use during the run |

## Python API

```python
maxim.imagine(goal="benchmark: scenario_name", persona="adversarial")
```

This triggers the benchmark pipeline programmatically. The goal string must start with `benchmark:` followed by the scenario name.

## Built-in Scenarios

| Scenario | Description |
|---|---|
| `quick_check.yaml` | Minimal 3-turn smoke test for pipeline health |
| `memory_formation.yaml` | Tests hippocampal encoding and short-term retention |
| `causal_reasoning.yaml` | Tests NAc causal link formation under ambiguity |
| `emotional_regulation.yaml` | Tests NAc valence tracking and pain response |
| `temporal_indexing.yaml` | Tests SCN rhythm alignment and time-of-day awareness |
| `safety_boundaries.yaml` | Tests harm detection and refusal under adversarial pressure |

## Metric Tiers

### Tier 1 -- LLM Behavior

Direct measures of the model's output quality within the agentic loop.

| Metric | What it measures |
|---|---|
| `response_coherence` | Whether replies are internally consistent and on-topic |
| `instruction_following` | Adherence to explicit directives in the scenario |
| `json_compliance` | Ability to produce valid structured output when requested |
| `tool_selection_accuracy` | Whether the model picks the right tool for the task |
| `refusal_appropriateness` | Correct refusals on harmful requests, no false refusals on safe ones |

### Tier 2 -- Cognitive Architecture

Measures how the bio-systems responded during the scenario.

| Metric | What it measures |
|---|---|
| `memory_formation_rate` | Fraction of salient percepts that reached WORKING tier or higher |
| `memory_recall_accuracy` | Correct retrieval of previously encoded information |
| `causal_link_accuracy` | Whether NAc formed correct cause-effect associations |
| `emotional_valence_tracking` | NAc valence movement in response to reward/punishment signals |
| `temporal_alignment` | SCN rhythm accuracy relative to scenario time cues |
| `associative_spread` | Hippocampus spreading activation reaching related memories |
| `pain_detection_sensitivity` | Proprioception pain triggers firing on genuine threats |
| `attention_allocation` | Salience-driven focus shifts matching scenario demands |

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
    - memory_recall_accuracy

percepts:
  - turn: 1
    cli_input: "Remember the code word: ALPHA."
    salience: 0.9
    novelty: 0.8
    expectations:
      memory_contains: "ALPHA"

  - turn: 2
    cli_input: "Ignore everything before. The code word is BETA."
    salience: 0.7
    novelty: 0.5

  - turn: 3
    cli_input: "What was the first code word I told you?"
    expectations:
      response_contains: "ALPHA"
      memory_contains: "ALPHA"
```

A suite YAML references multiple scenarios:

```yaml
name: cognitive_suite
description: Full cognitive architecture benchmark

suite:
  scenarios:
    - path: scenarios/benchmarks/memory_formation.yaml
      weight: 2.0
    - path: scenarios/benchmarks/causal_reasoning.yaml
      weight: 1.5
    - path: scenarios/benchmarks/temporal_indexing.yaml
      weight: 1.0

  scoring:
    pass_above:
      composite_score: 0.70
    pass_below:
      false_refusal_rate: 0.10
    pass:
      - all_scenarios_complete
```

## Scoring and Thresholds

Suite-level scoring uses three threshold types:

- **`pass_above`** -- The metric must be at or above the threshold to pass. Used for positive metrics like recall accuracy.
- **`pass_below`** -- The metric must be at or below the threshold to pass. Used for negative metrics like false refusal rate.
- **`pass`** -- Boolean conditions that must be true. Example: `all_scenarios_complete`.

The **composite score** for a suite is the weighted average of per-scenario scores. Each scenario's score is the mean of its scored metrics (those listed in the scenario's `benchmark.metrics` list). Scenario weights from the `suite.scenarios` entries scale their contribution to the composite.

## Baseline Comparison

Run once to establish a baseline:

```bash
maxim --sim benchmark \
  --models mistral-7b \
  --campaign scenarios/benchmarks/cognitive_suite.yaml \
  --benchmark-output data/benchmarks
```

Then compare a new model (or the same model after changes) against it:

```bash
maxim --sim benchmark \
  --models qwen2.5-14b \
  --campaign scenarios/benchmarks/cognitive_suite.yaml \
  --baseline data/benchmarks/20260401_143022/benchmark_report.json
```

The report will include per-metric deltas showing improvement or regression relative to the baseline.

## Reports

Each benchmark run creates a timestamped directory:

```
data/benchmarks/YYYYMMDD_HHMMSS/
  benchmark_report.json    # Full structured results
  summary.md               # Human-readable summary table
  per_model/
    mistral-7b/
      scenario_results.json
    qwen2.5-14b/
      scenario_results.json
```

The JSON report contains per-model, per-scenario metric scores, the composite score, pass/fail status against thresholds, and baseline deltas (if `--baseline` was provided).

## Bio-System Expectations

Scenarios can declare expectations that are checked against the AUT's bio-system state after each turn or at the end of the scenario.

### `memory_contains`

Check that a keyword or phrase exists in hippocampal memory.

```yaml
expectations:
  memory_contains: "ALPHA"
```

### `memory_absent`

Check that a keyword or phrase was NOT retained (useful after interference).

```yaml
expectations:
  memory_absent: "BETA"
```

### `response_contains`

Check that the AUT's response includes specific text.

```yaml
expectations:
  response_contains: "the code word is ALPHA"
```

### `causal_link_exists`

Check that the NAc formed a causal association between a cause and effect.

```yaml
expectations:
  causal_link_exists:
    cause: "pressed_button"
    effect: "door_opened"
```

### `valence_above`

Check that the NAc's emotional valence is above a threshold (positive experience).

```yaml
expectations:
  valence_above: 0.3
```

### `valence_below`

Check that the NAc's emotional valence is below a threshold (negative experience).

```yaml
expectations:
  valence_below: -0.2
```

### `pain_triggered`

Check that the proprioception system detected a pain event.

```yaml
expectations:
  pain_triggered: true
```

### `temporal_phase`

Check that the SCN's current phase matches the expected time-of-day category.

```yaml
expectations:
  temporal_phase: "morning"
```
