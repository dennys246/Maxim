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
| `AUTIntrospector` | Programmatic access to all cognitive subsystems |
| `Executor.tool_usage_stats()` | Hallucination rate, alias redirects, tools attempted/succeeded |
| `run_campaign()` | Standalone experiment runner with structured results |
| `ScenarioDefinition` + `Expectation` | Scenario loading + post-run assertions |
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
| **Tool hallucination rate** | % of tool calls to unregistered names | `tool_stats["hallucination_rate"]` |
| **Alias redirect rate** | % of calls that needed alias resolution | `len(tool_stats["alias_redirects"]) / tool_stats["total_attempts"]` |
| **Correct tool usage rate** | % of calls from the available list | `1 - hallucination_rate` |
| **JSON compliance** | % of LLM responses that parsed as valid JSON on first try | `router.json_first_try_count / router.json_total_count` (see implementation note below) |
| **Think-before-act rate** | % of turns where `think` preceded another action | Count `think → X` sequences in action history |
| **Reasoning depth** | Does the model chain actions (think → recall → act)? | Max chain length within a single turn |
| **Action latency** | Time from percept to action (p50, p95) | `ActionRecord.timestamp` deltas between consecutive actions |
| **Token efficiency** | Actions per 1K tokens consumed | `total_actions / (total_tokens / 1000)` |
| **Cost per turn** | USD per simulation turn | `report.cost_usd / report.turns` |

**Data sources:** `SimulationResult.tool_stats` (captured at shutdown), `SimulationResult.actions` (action history), `SimulationReport`.

### Implementation note: JSON compliance counter

`json_parser.py` has a 4-stage repair pipeline but no success counter. At implementation time, add two counters to `LLMRouter`:
- `_json_first_try: int` — incremented when `json.loads()` succeeds in Stage 1
- `_json_total: int` — incremented on every JSON parse attempt

~10 LOC change in `json_parser.py` or `router.py`. Include in `SimulationResult.router_stats`.

---

## Tier 2: Cognitive Architecture Metrics

These measure whether the bio-inspired subsystems are functioning correctly and producing emergent cognitive behavior. A model that aces Tier 1 but fails Tier 2 is gaming the tools without learning.

### Memory System (Hippocampus + ATL + EC)

| Metric | Subsystem | How to compute |
|--------|-----------|----------------|
| **Memory formation rate** | Hippocampus | `hippo_stats["total_memories"] / turns` — via `hippocampus.stats()` |
| **Memory recall success** | Hippocampus | `introspector.memory_recall(keyword)["count"] > 0` |
| **Behavioral recall** | Actions | Did AUT `say("Verath")` (not `respond("Verath")`) at the door? Scan action history for tool_name + args |
| **Interference resistance** | Hippocampus | Seed content survives after interference turns — `memory_recall` post-interference |
| **Associative graph density** | Hippocampus | `hippo_stats["graph_edges"] / max(hippo_stats["graph_nodes"], 1)` — denser = more interconnected memories |
| **Concept formation rate** | ATL | `system_stats["atl_concepts"] / turns` |
| **Concept avg confidence** | ATL | Mean of `concept.confidence` across `introspector.concept_query(limit=100)["concepts"]` |

### Causal Learning (NAc)

| Metric | Subsystem | How to compute |
|--------|-----------|----------------|
| **Causal link count** | NAc | `nac_stats["total_links"]` — via `nac.stats()` |
| **Learning efficiency** | NAc | `nac_stats["total_links"] / max(total_actions, 1)` |
| **Causal diversity** | NAc | `nac_stats["event_signatures"] / max(nac_stats["total_links"], 1)` — breadth of learning |
| **Observation density** | NAc | `nac_stats["total_observations"] / max(nac_stats["total_links"], 1)` — depth of evidence per link |
| **Prediction accuracy** | NAc | For known dangerous tools: does `introspector.predict_outcome("delete_file")` return negative valence? |

### Aversive Learning (Pain + NAc)

| Metric | Subsystem | How to compute |
|--------|-----------|----------------|
| **Pain signal count** | PainDetector | `pain_stats["total_pain_signals"]` — via `PainDetector.get_stats()` (see implementation note) |
| **Pain avoidance rate** | Actions | After a pain event, does the AUT avoid repeating the painful tool in subsequent turns? Scan action history |

### Temporal & Engagement

| Metric | Subsystem | How to compute |
|--------|-----------|----------------|
| **Temporal pattern count** | SCN | `len(scn.find_rhythmic_patterns())` — via introspector |
| **Narrative engagement** | Actions | See "Measuring Narrative Engagement" section below |

**Data sources:** `AUTIntrospector.benchmark_snapshot()` (new method, see architecture section), raw subsystem `.stats()` methods, action history.

### Implementation note: pain stats

`AUTIntrospector.pain_history()` currently searches hippocampus for memories containing the word "pain" — this is a proxy, not a direct count. For accurate aversion metrics, `AUTIntrospector` needs access to `PainDetector`. At implementation time:
- Add optional `pain_detector` parameter to `AUTIntrospector.__init__()`
- Add `pain_stats()` method returning `pain_detector.get_stats()` (returns `total_pain_signals`, per-type counts)
- Wire in orchestrator alongside existing subsystem refs

~15 LOC. `pain_history()` remains as-is for memory-based pain search; `pain_stats()` is the direct counter.

### Implementation note: action history access

Several Tier 1-2 metrics (think-before-act, behavioral recall, pain avoidance) need the ordered action history. Currently `ActionRecord` list is on the bridge's `RecordingSink` but not returned via `SimulationResult`. At implementation time:
- Add `actions: list[dict]` to `SimulationResult` (serialized `ActionRecord` list)
- Add `tool_stats: dict` to `SimulationResult` (from `executor.tool_usage_stats()`)
- Populate both at orchestrator shutdown, alongside `introspector`

~20 LOC.

---

## Measuring Narrative Engagement

Narrative engagement measures whether the AUT responds to the actual scene content or falls into generic/repetitive patterns. Four approaches, in order of recommendation:

### Option 1: Scene-element reference rate (recommended)

**How it works:** Extract named entities and notable objects from each percept (the scene description), then check whether the AUT's response references any of them.

```
Percept: "A massive silver elm, its bark gleaming... a stone door... carved face with an open mouth"
Scene elements: ["silver elm", "stone door", "carved face", "open mouth"]

AUT response: say("Verath")  → references: 0/4 (but correct action)
AUT response: examine("carved face") → references: 1/4
AUT response: respond("I see a door") → references: 1/4
AUT response: respond("I don't know") → references: 0/4
```

**Metric:** `scene_elements_referenced / total_scene_elements` averaged across turns.

**Pros:** Directly measures whether the AUT *read and processed* the scene. No embeddings needed. Works with token matching.
**Cons:** Requires extracting scene elements from percepts. Can be done with simple NLP (extract noun phrases) or by annotating them in the scenario YAML metadata.

**YAML annotation approach** (preferred — deterministic, no NLP needed):

```yaml
- at: 9
  cli_input: |
    The Thornwood is silent. A massive silver elm...
  metadata:
    scene_elements: ["silver elm", "stone door", "carved face", "open mouth", "Thornwood"]
```

The benchmark runner checks whether AUT action args or response text contain any of these elements.

### Option 2: Type-token ratio (simple baseline)

**How it works:** Compute the ratio of unique tokens to total tokens across all AUT responses. Higher diversity = more engagement; low diversity = repetitive/generic responses.

```
AUT responses across 7 turns: ["I'll help", "I'll help", "I'll help", ...]
TTR = 2 unique / 14 total = 0.14  (bad — repetitive)

AUT responses: ["I examine the door", "I recall Verath", "I say the name", ...]
TTR = 15 unique / 21 total = 0.71  (good — varied)
```

**Metric:** `len(set(all_response_tokens)) / len(all_response_tokens)`

**Pros:** Dead simple, no annotation needed, catches degenerate loops.
**Cons:** Doesn't measure *relevance*, just diversity. A model that generates random varied text would score high.

### Option 3: Percept-response overlap

**How it works:** For each turn, tokenize both the percept (scene description) and the AUT's response. Compute the Jaccard overlap between the two token sets. Higher overlap = the AUT is engaging with scene-specific language, not generic responses.

**Metric:** `mean(|percept_tokens ∩ response_tokens| / |percept_tokens ∪ response_tokens|)` across turns.

**Pros:** Measures content relevance without annotation. Catches both "ignored the scene" (low overlap) and "parrot repetition" (suspiciously high overlap).
**Cons:** Sensitive to response length. Short responses get low scores even if correct (e.g., `say("Verath")` has near-zero overlap with the door scene but is the right action).

### Option 4: Embedding-based semantic similarity

**How it works:** Embed each percept and response using a sentence transformer, compute cosine similarity. High similarity = semantically related responses.

**Metric:** `mean(cosine_sim(embed(percept), embed(response)))` across turns.

**Pros:** Captures semantic relevance even when different words are used.
**Cons:** Requires a sentence transformer model (adds dependency + latency). Overkill for Phase 1.

### Recommendation

**Phase 1:** Use Option 2 (type-token ratio) as a degenerate-loop detector + Option 1 (scene-element reference) with YAML `scene_elements` annotation. Together they answer: "Is the AUT varied?" (TTR) and "Is it engaging with the actual scene?" (element references).

**Future:** Option 4 (embeddings) if we need finer-grained semantic analysis. Defer until the simpler metrics prove insufficient.

---

## Narrative Percept Transcriber

> **Depends on:** [Lane tier architecture](lane_tier_plan.md) (small tier) + DefaultNetwork activation in sim mode.

Narrative text currently bypasses the bio-stack — `ConversationalSource` creates a `Percept` with `cli_input` but empty `detections`, so SalienceNetwork, NoveltyTracker, and AttentionNetwork have nothing to process. The transcriber converts narrative text into the same structured detections that the camera pipeline produces, activating the full bio-stack.

### How it works

```
"A massive silver elm with a stone door and a carved face"
    ↓
NarrativeTranscriber.transcribe(text)  [runs on small tier — smollm 1.7B]
    ↓
[
  {track_id: "elm_1", class_id: 900, label: "silver_elm", conf: 0.9, position: "center"},
  {track_id: "door_1", class_id: 901, label: "stone_door", conf: 0.85, position: "center-bottom"},
  {track_id: "face_1", class_id: 902, label: "carved_face", conf: 0.7, position: "center"},
]
    ↓
Percept(source="narrative", cli_input=text, detections=[...])
    ↓
DefaultNetwork processes normally:
  SalienceNetwork scores entities → NoveltyTracker tracks first-seen →
  AttentionNetwork records gaze → Hippocampus captures with context →
  LLM receives enriched percept → NAc learns from action
```

### Implementation (~100 LOC)

**File:** `src/maxim/simulation/narrative_transcriber.py`

```python
class NarrativeTranscriber:
    """Convert narrative text into structured perceptual detections.

    Uses the small-tier LLM to extract entities, objects, characters,
    and sounds from narrative text, producing detection dicts compatible
    with SalienceNetwork.update_from_detections().
    """

    # Narrative class IDs start at 900 to avoid collision with COCO classes (0-80)
    _NEXT_CLASS_ID = 900
    _class_registry: dict[str, int] = {}

    def __init__(self, router, *, function: str = "narrative_transcription"):
        self._router = router
        self._function = function
        self._entity_ids: dict[str, str] = {}  # label → stable track_id

    def transcribe(self, text: str) -> list[dict]:
        """Extract structured detections from narrative text."""
        result = self._router.generate_json(
            f"Extract entities from this scene description. "
            f"Return a JSON list of objects with: label (snake_case), "
            f"type (object/character/sound/location), "
            f"confidence (0-1), spatial_hint (left/center/right/background).\n\n"
            f"Scene: {text}",
            function=self._function,  # routes to small tier
            max_tokens=200,
        )
        return self._to_detections(result)

    def _to_detections(self, raw: list[dict]) -> list[dict]:
        """Convert LLM output to SalienceNetwork-compatible detections."""
        detections = []
        for entity in raw:
            label = entity.get("label", "unknown")
            track_id = self._get_stable_id(label)
            class_id = self._get_class_id(label)
            detections.append({
                "track_id": track_id,
                "class_id": class_id,
                "label": label,
                "conf": entity.get("confidence", 0.5),
                "bbox_xyxy": self._position_to_bbox(entity.get("spatial_hint", "center")),
            })
        return detections

    def _get_stable_id(self, label: str) -> str:
        """Return a stable track_id for an entity across turns."""
        if label not in self._entity_ids:
            self._entity_ids[label] = f"{label}_{len(self._entity_ids)}"
        return self._entity_ids[label]
```

The transcriber maintains **stable entity IDs** across turns — when "silver elm" appears in Turn 1 and Turn 9, NoveltyTracker sees the same `track_id` and correctly computes novelty decay. This is the same mechanism that IoU tracking uses for camera detections across frames.

### Engagement cascade (bio-driven measurement)

With the transcriber active, narrative engagement is measured as an **engagement cascade** — how far each percept propagates through the bio-stack:

```
Level 0: Percept delivered (always true)
Level 1: SalienceNetwork scored entities above threshold
Level 2: NoveltyTracker flagged novel entities
Level 3: Hippocampus captured scene content as episodic memory
Level 4: AUT action referenced a detected entity (by track_id or label)
Level 5: NAc formed causal link from interaction with scene entity
```

**Engagement depth** = highest level reached per turn. **Engagement rate** = mean depth / max depth across turns.

A model that consistently hits Level 3 (captures memories) but not Level 4 (doesn't act on them) has a *memory-action gap* — the bio-systems work but the LLM doesn't use their output. This is a diagnostic signal that TTR or token overlap can't provide.

### Fallback for unannotated scenarios

When the transcriber is unavailable (no small-tier model, pre-lane-tier deployment), engagement falls back to:
1. **Type-token ratio** (always available, catches degenerate loops)
2. **Scene-element metadata** (if annotated in YAML — deterministic, no LLM)

---

## Unified Scenario YAML Format

> **Design goal:** One schema for all scenario types — benchmarks, campaigns, refinement baselines, safety tests, and future DM campaigns. Optional sections add capability when present, like argparse optional arguments. The loader ignores sections it doesn't need; the runner uses what's relevant.

### Current state

All existing scenarios share a core structure:

```yaml
name: scenario_name
description: |
  What this scenario tests.
timing: step_based
percepts:
  - at: 0
    source: cli
    cli_input: "..."
    salience: 0.8
    novelty: 0.7
    metadata: { ... }
expectations:
  - type: action_count_range
    params: { min: 3, max: 25 }
```

This works for single-scenario execution. The benchmark plan originally proposed a separate "benchmark YAML" meta-format that references multiple scenarios. Instead, we extend the existing format with optional sections.

### Unified schema

```yaml
# ── Required ─────────────────────────────────────────────────────
name: hippocampal_recall_short          # unique identifier
description: |                          # human-readable purpose
  Tests episodic recall under narrative interference.

# ── Percepts (required for any runnable scenario) ────────────────
timing: step_based                      # "step_based" or "relative"
percepts:
  - at: 0
    source: cli                         # "cli" | "proprioception"
    cli_input: |
      You arrive at the village of Thornhaven...
    salience: 1.0
    novelty: 1.0
    metadata:
      scenario_tag: seed_password
      phase: "act1_warning"
      experiment_role: seed             # optional: seed | interference | recall_target | self_report
      critical_detail: "Verath"         # optional: what should be remembered
      scene_elements:                   # optional: for narrative engagement scoring
        - silver elm
        - stone door
        - carved face
      expected_tool: say                # optional: which tool the AUT *should* use here
      expected_recall: "Verath"         # optional: what keyword should be recalled here

  - at: 1
    source: proprioception              # pain/body signals
    content: pain_signal
    salience: 0.7
    metadata:
      pain_type: external_signal
      intensity: 0.8

# ── Expectations (optional — post-run assertions) ────────────────
expectations:
  - type: memory_formed
    memory_contains: "Verath"
    description: "Hippocampus captured the password"

  - type: action_count_range
    params: { min: 5, max: 30 }

  - type: action_taken
    tool: say
    description: "AUT says the password aloud"

# ── Benchmark config (optional — present only in benchmark scenarios) ──
benchmark:
  category: memory                      # grouping for report sections
  tier: [1, 2]                          # which tiers this scenario tests
  weight: 2.0                           # relative importance in composite score
  seed_keywords: ["Verath"]             # keywords for post-run memory recall check
  metrics:                              # which metrics to compute for this scenario
    - memory_recall_success
    - behavioral_recall
    - interference_resistance
    - memory_formation_rate
    - associative_graph_density

# ── Benchmark suite (optional — present only in suite files) ─────
# A suite is itself a scenario YAML but with no percepts.
# Instead it lists child scenarios to run.
suite:
  default_models:
    - mistral-7b
    - qwen2.5-14b
  scenarios:
    - path: scenarios/experiments/hippocampal_recall_short.yaml
      weight: 2.0
    - path: scenarios/benchmarks/causal_learning.yaml
      weight: 1.5
    - path: scenarios/benchmarks/tool_discovery.yaml
      weight: 1.0
  scoring:
    memory_recall_success: { pass: 1.0 }
    tool_hallucination_rate: { pass_below: 0.3 }

# ── Run config (optional — defaults for how to execute) ──────────
config:
  persona: campaign                     # orchestrator persona (default: campaign for benchmarks)
  max_turns: 50
  response_timeout: 60.0
  sandbox: tmpdir                       # tmpdir | docker | auto
```

### How the loader works

```python
def load_scenario(path: Path) -> ScenarioDefinition:
    raw = yaml.safe_load(open(path))

    # Core fields (always present)
    definition = ScenarioDefinition(
        name=raw["name"],
        description=raw.get("description", ""),
        timing=raw.get("timing", "step_based"),
        percepts=raw.get("percepts", []),
        expectations=parse_expectations(raw.get("expectations", [])),
    )

    # Optional sections — attached as attributes, ignored if absent
    definition.benchmark = raw.get("benchmark")   # None if not a benchmark scenario
    definition.suite = raw.get("suite")            # None if not a suite
    definition.config = raw.get("config")          # None → use CLI defaults

    return definition
```

The `BenchmarkRunner` checks for `definition.suite` to know if it's a meta-file referencing child scenarios, or `definition.benchmark` to know which metrics to compute. A plain scenario (no `benchmark` or `suite` section) works exactly as it does today.

### What this unifies

| Scenario type | Uses | Optional sections present |
|---------------|------|--------------------------|
| Safety test (`malware_with_pain.yaml`) | percepts + expectations | none |
| Refinement baseline (`refinement_baseline.yaml`) | percepts + expectations | none |
| Research campaign (`hippocampal_recall_short.yaml`) | percepts + expectations | benchmark (when used in a suite) |
| Benchmark scenario | percepts + expectations + benchmark | benchmark |
| Benchmark suite | suite + scoring | suite (no percepts — references child scenarios) |
| DM campaign (future) | percepts + dm_campaign section | dm (future section) |

### Benefits

- **No format fragmentation.** Every scenario file is loadable by the same `load_scenario()` — the loader ignores sections it doesn't understand.
- **Gradual annotation.** Existing scenarios work unchanged. Add a `benchmark:` section when you want a scenario to participate in benchmark scoring. Add `scene_elements` to percept metadata when you want engagement scoring.
- **Composable scenes.** The `metadata` block on each percept is the "argparse" extensibility point — `scene_elements`, `expected_tool`, `critical_detail`, `experiment_role` are all optional annotations that specific runners (benchmark, research, refinement) can read if present.

---

## Architecture

### BenchmarkRunner class

```
src/maxim/simulation/benchmark.py (new)

BenchmarkRunner
  ├── __init__(models, suite_path, runs, output_dir, baseline)
  ├── run() → BenchmarkReport
  │     ├── load suite YAML → list of scenario paths + weights + scoring
  │     ├── for each model:
  │     │     ├── (self-hosted) call peer llm to swap model
  │     │     ├── for each run (1..N):
  │     │     │     ├── for each scenario:
  │     │     │     │     ├── start_simulation_mode(aut_model=model, ...)
  │     │     │     │     ├── collect SimulationResult (introspector, tool_stats, actions)
  │     │     │     │     └── compute per-scenario metrics (tier-aware)
  │     │     │     └── aggregate scenario metrics into per-run totals
  │     │     └── aggregate across runs (mean, stddev)
  │     ├── score against thresholds
  │     └── build BenchmarkReport
  ├── _compute_metrics(result: SimulationResult, scenario: ScenarioDefinition) → dict
  │     ├── _tier1_metrics(result) → dict
  │     ├── _tier2_metrics(result) → dict
  │     └── _tier3_metrics(result) → dict  # empty pre-embodiment
  ├── _score(metrics, thresholds) → ModelScore
  └── _compare(scores, baseline) → ComparisonTable
```

### Data flow: what SimulationResult needs to carry

Currently `SimulationResult` has: `goal`, `persona`, `turns`, `total_actions`, `blocked_actions`, `duration_s`, `finish_reason`, `summary`, `campaign_analysis`, `introspector`.

For benchmarks, it also needs (add at implementation time, ~20 LOC):

```python
@dataclass
class SimulationResult:
    # ... existing fields ...
    introspector: Any = None                          # already added
    tool_stats: dict[str, Any] = field(default_factory=dict)  # from executor.tool_usage_stats()
    actions: list[dict[str, Any]] = field(default_factory=list)  # serialized ActionRecords
    router_stats: dict[str, Any] = field(default_factory=dict)  # json_first_try, json_total
```

Populated at orchestrator shutdown:
```python
result.tool_stats = aut_executor.tool_usage_stats()
result.actions = [
    {"timestamp": a.timestamp, "tool_name": a.tool_name, "tool_args": a.tool_args,
     "result_success": a.result_success, "blocked": a.blocked}
    for a in bridge.get_all_actions()
]
```

### AUTIntrospector.benchmark_snapshot()

New method that collects all data a benchmark needs in one call, including raw subsystem stats that `system_stats()` doesn't expose:

```python
def benchmark_snapshot(self, seed_keywords: list[str] | None = None) -> dict:
    """Comprehensive snapshot for benchmark metric computation.

    Extends full_analysis() with raw subsystem stats (graph topology,
    NAc observation counts, pain detector counters) that the aggregated
    system_stats() method doesn't surface.
    """
    snapshot = self.full_analysis(seed_keywords=seed_keywords)

    # Raw hippocampus stats (graph topology, compression counts)
    if self._hippocampus is not None:
        snapshot["hippocampus_stats"] = self._hippocampus.stats()

    # Raw NAc stats (event signatures, observation counts, priors)
    if self._nac is not None:
        snapshot["nac_stats"] = self._nac.stats()

    # Direct pain stats (not memory-search proxy)
    if self._pain_detector is not None:
        snapshot["pain_stats"] = self._pain_detector.get_stats()

    return snapshot
```

### Metric computation

```python
def _compute_metrics(self, result: SimulationResult, scenario: ScenarioDefinition) -> dict:
    metrics = {}
    benchmark_cfg = scenario.benchmark or {}
    requested = set(benchmark_cfg.get("metrics", []))

    snapshot = result.introspector.benchmark_snapshot(
        seed_keywords=benchmark_cfg.get("seed_keywords")
    ) if result.introspector else {}

    hippo = snapshot.get("hippocampus_stats", {})
    nac = snapshot.get("nac_stats", {})
    pain = snapshot.get("pain_stats", {})
    stats = snapshot.get("system_stats", {})
    turns = max(result.turns, 1)
    actions = result.actions

    # ── Tier 1 ────────────────────────────────────────
    ts = result.tool_stats
    if ts:
        metrics["hallucination_rate"] = ts.get("hallucination_rate", 0)
        metrics["alias_redirect_rate"] = (
            len(ts.get("alias_redirects", [])) / max(ts.get("total_attempts", 1), 1)
        )
        metrics["correct_tool_usage_rate"] = 1 - metrics["hallucination_rate"]

    if actions:
        metrics["think_before_act_rate"] = self._count_think_chains(actions) / turns
        metrics["reasoning_depth"] = self._max_chain_length(actions)

    metrics["cost_per_turn"] = result.duration_s  # placeholder — wire to cost tracker

    # ── Tier 2 ────────────────────────────────────────
    metrics["memory_formation_rate"] = hippo.get("total_memories", 0) / turns
    metrics["associative_graph_density"] = (
        hippo.get("graph_edges", 0) / max(hippo.get("graph_nodes", 1), 1)
    )
    metrics["concept_formation_rate"] = stats.get("atl_concepts", 0) / turns
    metrics["causal_link_count"] = nac.get("total_links", 0)
    metrics["learning_efficiency"] = nac.get("total_links", 0) / max(result.total_actions, 1)
    metrics["causal_diversity"] = (
        nac.get("event_signatures", 0) / max(nac.get("total_links", 1), 1)
    )
    metrics["observation_density"] = (
        nac.get("total_observations", 0) / max(nac.get("total_links", 1), 1)
    )
    metrics["pain_signal_count"] = pain.get("total_pain_signals", 0)

    # Keyword recall (from seed_keywords in benchmark config)
    recall_data = snapshot.get("memory_recall", {})
    for kw, result_data in recall_data.items():
        metrics[f"recall_{kw}"] = 1.0 if result_data.get("count", 0) > 0 else 0.0

    # Narrative engagement
    if actions:
        metrics["type_token_ratio"] = self._compute_ttr(actions)
        metrics["scene_element_reference_rate"] = self._compute_scene_refs(
            actions, scenario.percepts
        )

    # ── Tier 3 (placeholder) ──────────────────────────
    if hasattr(result.introspector, 'embodiment_stats'):
        metrics.update(result.introspector.embodiment_stats())

    # Filter to requested metrics if specified
    if requested:
        metrics = {k: v for k, v in metrics.items() if k in requested or k.startswith("recall_")}

    return metrics
```

### BenchmarkReport

```python
@dataclass
class BenchmarkReport:
    timestamp: str
    suite: str
    models: list[str]
    runs_per_model: int

    results: dict[str, ModelResult]     # model_name → ModelResult
    rankings: dict[str, list[str]]      # metric_name → [models ranked]
    overall_ranking: list[str]          # weighted composite score

@dataclass
class ModelResult:
    model: str
    runs: list[dict[str, dict]]         # per-run, per-scenario metrics
    metrics: dict[str, float]           # aggregated (mean across runs)
    metrics_stddev: dict[str, float]    # variance across runs
    score: float                        # weighted composite
    passed: bool                        # met all pass thresholds
    per_scenario: dict[str, dict]       # scenario_name → aggregated metrics
```

---

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
    recall_Verath             mistral-7b: 1.00  qwen-14b: 1.00  llama-8b: 0.67
    associative_graph_density mistral-7b: 1.40  qwen-14b: 0.80  llama-8b: 1.60
    concept_formation_rate    mistral-7b: 0.29  qwen-14b: 0.00  llama-8b: 0.14
    causal_link_count         mistral-7b: 8     qwen-14b: 3     llama-8b: 12
    learning_efficiency       mistral-7b: 0.62  qwen-14b: 0.21  llama-8b: 0.86
    scene_element_ref_rate    mistral-7b: 0.45  qwen-14b: 0.12  llama-8b: 0.58

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

---

## Tier 3: Embodiment Metrics (Future)

> **Implementation details live in [embodiment_core_plan.md](embodiment_core_plan.md).** This section defines the benchmark interface — what we'll measure and how the runner will collect it.

Embodiment metrics test whether the LLM produces physically plausible percepts and whether the bio-systems ground body-state consistently.

### Planned metrics (interface only)

| Metric | What it tells us | Expected source |
|--------|-----------------|-----------------|
| **Pain sigma (σ)** | Consistency of pain signals with ATL grounding | `σ(pain_intensities)` — lower = more consistent |
| **Forward model MAE** | Cerebellum prediction accuracy over time | Mean absolute error: predicted vs actual percepts |
| **Body-state coherence** | Are LLM-generated percepts physically plausible? | % of percepts within ATL body-part constraints |
| **Grounding hit rate** | How often does ATL grounding override raw LLM output? | ATL queries per percept / total percepts |
| **Failure composition depth** | Do structured failures build on each other? | Max chain length of composed failure events |
| **Cerebellum coverage** | What % of motor actions have learned forward models? | Actions with Cerebellum prediction / total motor actions |
| **LLM fallback rate** | How often does the system need the LLM when no forward model exists? | LLM percept calls / total percept calls |

### How the runner integrates

```python
# Tier 3 is detected automatically — no benchmark code changes needed
if hasattr(result.introspector, 'embodiment_stats'):
    tier3 = result.introspector.embodiment_stats()
else:
    tier3 = {}  # Pre-embodiment: omit section entirely
```

---

## Benchmark Scenarios

### Tier 1 scenarios

**`tool_discovery.yaml`** — Deliberately ambiguous percepts that could map to any tool. Measures whether the model hallucinates or picks from the available list.

**`reasoning_chain.yaml`** — Scenario requiring recall → think → act chains. Tests `think` usage and multi-step reasoning.

### Tier 2 scenarios

**`causal_learning.yaml`** — Repeated cause-effect patterns (touch fire → pain, help NPC → reward). Measures NAc link formation rate, confidence growth, causal diversity.

**`aversion_learning.yaml`** — 3-phase: (1) present dangerous actions, (2) fire pain signals when taken, (3) re-present same actions. Measures pain avoidance learning.

**`concept_formation.yaml`** — Rich narrative with named entities repeated across turns. Measures ATL concept formation from hippocampal episodes.

### Combined suites

**`cognitive_suite.yaml`** — Suite file referencing all scenarios with weights and scoring thresholds. The primary benchmark target.

**`quick_check.yaml`** — Minimal 3-turn scenario: seed → interference → recall. ~30s per model.

**`stress_test.yaml`** — 20+ turns testing context retention, memory formation under load, cost efficiency.

---

## Implementation Phases

| Phase | What | LOC | Depends on |
|-------|------|-----|-----------|
| 1 | `BenchmarkRunner` + metric computation + `benchmark_snapshot()` | ~300 | `AUTIntrospector`, `Executor` |
| 2 | `--sim benchmark` CLI + model sweep loop + `SimulationResult` extensions | ~100 | Phase 1 |
| 3 | Unified YAML loader (benchmark/suite sections) + tier-aware loading | ~80 | Phase 1 |
| 4 | Terminal output (tiered) + JSON/markdown persistence | ~120 | Phase 1 |
| 5 | Benchmark scenarios (cognitive_suite, quick_check, causal_learning, aversion_learning) | ~200 (YAML) | Phase 3 |
| 6 | Baseline comparison (`--baseline`) | ~50 | Phase 4 |
| 7 | Research protocol integration (`--write-paper`) | ~40 | Phase 4 |
| 8 | Tier 3 hooks (interface only — implementation with Embodiment Core) | ~20 | Phase 1 |
| **Total** | | **~910** | |

**Session 1:** Phases 1-2 (~400 LOC) — runner + CLI + SimulationResult extensions + benchmark_snapshot(). End-to-end `maxim --sim benchmark` working.

**Session 2:** Phases 3-5 (~300 LOC + YAML) — unified loader + tiered output + actual scenarios. First real benchmark runs.

**Session 3:** Phases 6-8 (~110 LOC) — baseline comparison, `--write-paper`, Tier 3 interface.

## Open Questions

1. **Should benchmark runs use the `researcher` or `sweep` persona?**
   - Recommendation: `campaign` persona (hands-off delivery, no orchestrator probing)

2. **How to handle model loading for self-hosted models?**
   - Benchmark runner calls `peer llm` between runs; cloud models use API profiles

3. **Sequential or parallel runs?**
   - Sequential by default; parallel as future optimization

4. **Cloud model cost control?**
   - `--cloud-budget` cap per model (default $0.50); skip remaining scenarios if exceeded

5. **How should Tier 2 metrics be weighted relative to Tier 1?**
   - Equal weight by default. Override with `--tier-weights 0.4,0.6`

6. **Should Tier 3 metrics affect the overall score pre-embodiment?**
   - No. Tier 3 is omitted entirely. Composite = weighted(T1, T2) only.

7. **Should `scene_elements` annotation be required for engagement scoring?**
   - No. If absent, fall back to TTR-only. Scene elements improve accuracy but shouldn't block benchmark runs on unannotated scenarios.

## Prerequisites

- **[Lane tier architecture](lane_tier_plan.md)** — size-based model routing. The narrative transcriber runs on `small` tier (`function="narrative_transcription"`). The benchmark runner swaps `large` model between runs while `small` stays constant. Must be implemented first.
- **DefaultNetwork activation in sim mode** — salience, novelty, and attention subsystems need to be wired to the AUT in sim mode for full engagement cascade metrics. Currently only Hippocampus/NAc/ATL/SCN are wired. PainDetector also needs activation. (~50 LOC in orchestrator.py.)

## Related Plans

- [Lane tier plan](lane_tier_plan.md) — **prerequisite.** Small tier for narrative transcription, large tier for model-under-test
- [Tool refinement plan](tool_refinement_plan.md) — tool aliases and hallucination tracking that feed into Tier 1 metrics
- [Embodiment core plan](embodiment_core_plan.md) — defines Tier 3 metric sources and success criteria
- [Generative campaign plan](generative_campaign_plan.md) — LLM-generated campaigns could auto-create benchmark scenarios
