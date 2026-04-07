# Lane Tier Architecture — Size-Based Model Routing

> **Status:** Not started. Prerequisite for benchmark plan (narrative transcriber needs small-tier routing).
> **Scope:** ~500 LOC refactor across 4 phases.
> **Depends on:** Multi-LLM Scaling (done), WorkerPool + LaneConfig + LaneBackendManager (all exist).

---

## Problem

The current lane system maps **functions to lanes**: `infer`, `review`, `record`. Each lane gets one model. This worked for 3 lanes but doesn't scale:

- Adding a 4th function (narrative transcription) means: new lane? Reuse `record`? The naming breaks down.
- `record` (concept extraction) and a proposed `utility` lane would run the same model — why separate?
- No fallback logic: if `infer` is GPU-busy, `review` can't help even though it runs the same model.
- No way to restrict functions from small models (e.g., "never run agent inference on 1.7B") without hardcoding lane checks in every caller.
- Every new consumer (benchmarks, DM choice classification, entity naming, JSON repair) needs a lane decision that's really a model-size decision.

## Solution

Replace function-based lanes with **size-tier lanes**. Functions are **routed to tiers** via a configurable mapping with fallback chains and restrictions.

### Current vs proposed

```
Current (function = lane):
  infer  lane → mistral-7b GPU  → agent loop inference
  review lane → mistral-7b GPU  → fear review
  record lane → smollm-1.7b CPU → concept extraction

Proposed (tier = capability level):
  large tier → 14B+ GPU          → inference, fear review, paper writing
  medium tier → 7B GPU/CPU       → analysis, campaign summary, review fallback
  small tier → 1.7B-3B CPU       → transcription, classification, extraction, JSON repair
```

Functions declare what tier they need. The router handles placement, fallback, and restrictions.

### Function routing table

```yaml
# Default function → tier mapping (configurable via llm_lanes.json or env)
functions:
  # Core agent operations — need strong reasoning
  agent_inference:
    tier: large
    fallback: []                    # RESTRICTED — never fall back to weaker model
    description: "Main agent loop LLM calls"

  fear_review:
    tier: large
    fallback: [medium]              # Can degrade to medium if large is saturated
    description: "FearAgent safety review"

  # Analysis and writing — need decent reasoning
  paper_writing:
    tier: large
    fallback: [medium]
    description: "Research protocol Writer agent"

  campaign_summary:
    tier: medium
    fallback: [small]
    description: "Post-campaign analysis summaries"

  sim_report_analysis:
    tier: medium
    fallback: [small]
    description: "SimulationReport LLM analysis"

  # Structured extraction — structured output, small model sufficient
  concept_extraction:
    tier: small
    fallback: [medium]              # Can upgrade if small is unavailable
    description: "ATL concept extraction from memories"

  concept_grounding:
    tier: small
    fallback: [medium]
    description: "ATL concept grounding verification"

  # Utility tasks — lightweight, structured output
  narrative_transcription:
    tier: small
    fallback: []                    # Fast enough on small, no fallback needed
    description: "Parse narrative text into structured detections"

  entity_naming:
    tier: small
    fallback: []
    description: "Generate character/entity display names"

  choice_classification:
    tier: small
    fallback: [medium]
    description: "DM choice classifier (map AUT response to encounter choices)"

  json_repair:
    tier: small
    fallback: []
    description: "Fix malformed JSON from other LLM calls"
```

### Fallback behavior

When a function's primary tier is unavailable (queue full, no model loaded, backend error):

1. Check fallback chain in order: `[medium, small]`
2. Skip tiers that are also unavailable
3. If all fallbacks exhausted and function is restricted (`fallback: []`), **fail with clear error** — don't silently degrade
4. Track fallback frequency in `LaneMetrics` for diagnostics

### Restrictions

Functions with `fallback: []` are **hard-restricted** to their declared tier. This prevents:
- Agent inference on a 1.7B model (would produce garbage)
- Narrative transcription bouncing to large (wastes GPU for a trivial task)

Functions with fallback chains degrade gracefully under load.

---

## Architecture

### Tier configuration (replaces DEFAULT_LANES)

```python
# In worker_pool.py
DEFAULT_TIERS = {
    "large": LaneConfig(
        name="large",
        max_workers=1,
        requires_gpu=True,
        model_profile=None,         # Set by --language-model or peer config
        device="gpu",
    ),
    "medium": LaneConfig(
        name="medium",
        max_workers=1,
        requires_gpu=False,
        model_profile=None,         # Optional — omit if only 2 tiers needed
        device="auto",
    ),
    "small": LaneConfig(
        name="small",
        max_workers=1,
        requires_gpu=False,
        model_profile="smollm-1.7b",
        device="cpu",
        n_gpu_layers=0,
    ),
}
```

**Flexible tier count:** Not every deployment needs 3 tiers. A peer machine with no GPU might have only `medium` + `small`. The leader with an RTX 5080 might have `large` + `small` (skipping medium since large handles both). The function router adapts — if a function's tier doesn't exist, it follows the fallback chain.

### FunctionRouter (new, ~150 LOC)

```python
@dataclass
class FunctionSpec:
    """How a named function maps to the tier system."""
    name: str
    tier: str                           # primary tier
    fallback: list[str]                 # ordered fallback tiers (empty = restricted)
    description: str = ""

class FunctionRouter:
    """Routes named functions to tier-based lanes with fallback."""

    def __init__(
        self,
        functions: dict[str, FunctionSpec],
        available_tiers: set[str],
    ) -> None:
        self._functions = functions
        self._available = available_tiers

    def resolve(self, function: str) -> str:
        """Return the tier name to use for this function.

        Tries primary tier, then fallbacks. Raises if no tier available
        and function has no fallback (restricted).
        """
        spec = self._functions.get(function)
        if spec is None:
            # Unknown function — default to largest available tier
            return self._largest_available()

        if spec.tier in self._available:
            return spec.tier

        for fallback_tier in spec.fallback:
            if fallback_tier in self._available:
                logger.info("Function %s: primary tier %s unavailable, falling back to %s",
                           function, spec.tier, fallback_tier)
                return fallback_tier

        if not spec.fallback:
            raise TierRestrictionError(
                f"Function '{function}' requires tier '{spec.tier}' "
                f"(restricted, no fallback). Available: {self._available}"
            )
        raise TierUnavailableError(
            f"No tier available for function '{function}'. "
            f"Tried: {spec.tier} → {spec.fallback}. Available: {self._available}"
        )

    def _largest_available(self) -> str:
        for tier in ["large", "medium", "small"]:
            if tier in self._available:
                return tier
        raise TierUnavailableError("No tiers available")
```

### WorkerPool changes (~60 LOC)

```python
class WorkerPool:
    def submit_function(
        self,
        function: str,
        fn: Callable,
        *,
        priority: int = 5,
        deps: DependencySpec | None = None,
    ) -> str:
        """Submit a job by function name. FunctionRouter resolves the tier."""
        tier = self._function_router.resolve(function)
        return self.submit(fn, lane=tier, priority=priority, deps=deps)
```

Existing `submit(..., lane="infer")` still works — it maps directly to a tier name. New callers use `submit_function("agent_inference", ...)`.

### LLMRouter changes (~50 LOC)

```python
class LLMRouter:
    def generate_json(self, prompt, *, function: str = "agent_inference", **kw):
        """Generate JSON using the appropriate tier for this function."""
        tier = self._function_router.resolve(function)
        backend = self._backend_manager.get_backend(tier)
        return backend.generate_json(prompt, **kw)
```

Callers that currently pass `lane="review"` switch to `function="fear_review"`. Backward compat: if `lane=` is passed, it's treated as a direct tier name.

---

## Backward Compatibility

The refactor must not break existing code. Strategy:

1. **`lane=` parameter still works everywhere.** If a caller passes `lane="infer"`, it maps to `"large"`. A backward-compat alias dict handles this:
   ```python
   _LEGACY_LANE_ALIASES = {
       "infer": "large",
       "review": "large",    # or medium, configurable
       "record": "small",
   }
   ```

2. **`DEFAULT_LANES` still works.** Code that constructs `WorkerPool(lane_configs=DEFAULT_LANES)` keeps working because `DEFAULT_LANES` becomes `DEFAULT_TIERS` with an alias export.

3. **Gradual migration.** Callers can switch from `lane="review"` to `function="fear_review"` incrementally. Both work simultaneously.

---

## Configuration

### Default: auto-detect from hardware

```python
def detect_tiers() -> dict[str, LaneConfig]:
    """Auto-detect available tiers based on hardware."""
    tiers = {}

    # Always available: small (CPU, ~2GB RAM)
    tiers["small"] = LaneConfig(
        name="small", max_workers=1,
        model_profile="smollm-1.7b", device="cpu", n_gpu_layers=0,
    )

    # GPU available? Add large tier
    if gpu_available():
        tiers["large"] = LaneConfig(
            name="large", max_workers=1,
            requires_gpu=True, device="gpu",
        )

    # Enough RAM for a medium CPU model? Add medium tier
    if available_ram_gb() > 8:
        tiers["medium"] = LaneConfig(
            name="medium", max_workers=1,
            device="cpu",
        )

    return tiers
```

### Override: CLI flags + env vars

```bash
# Existing flags (unchanged semantics)
maxim --language-model mistral-7b         # Sets large tier model
maxim --aut-model qwen2.5-14b             # Sets AUT's large tier in sim mode

# New flag: assign model to specific tier
maxim --tier-model small=smollm-1.7b      # Explicit small tier model
maxim --tier-model medium=mistral-7b      # Explicit medium tier model

# Env vars for function routing overrides
MAXIM_FUNCTION_FEAR_REVIEW_TIER=medium    # Override fear_review to medium
MAXIM_FUNCTION_CONCEPT_EXTRACTION_TIER=medium  # Upgrade extraction to medium
```

### Config file: `data/util/llm_lanes.json`

```json
{
  "tiers": {
    "large": {"model_profile": "qwen2.5-14b", "device": "gpu"},
    "small": {"model_profile": "smollm-1.7b", "device": "cpu"}
  },
  "functions": {
    "fear_review": {"tier": "medium", "fallback": ["large"]},
    "concept_extraction": {"tier": "small", "fallback": ["medium"]}
  }
}
```

---

## Hardware Scenarios

### Leader (RTX 5080 + 16GB CPU RAM)

```
large:  qwen2.5-14b on GPU (inference, fear review, paper writing)
small:  smollm-1.7b on CPU (transcription, classification, extraction)
```

Two tiers — `medium` not needed because `large` handles its functions directly.

### Mac Peer (24GB unified memory, no discrete GPU)

```
medium: mistral-7b on CPU (inference, review — no GPU available)
small:  smollm-1.7b on CPU (utility tasks)
```

Two tiers — `large` unavailable (no GPU). Agent inference falls back from `large` to `medium` via fallback chain. Functions restricted to `large` with no fallback would error with a clear message.

### Dual-GPU server (future)

```
large:  qwen2.5-14b on GPU:0 (inference)
medium: mistral-7b on GPU:1  (review, analysis)
small:  smollm-1.7b on CPU   (utility)
```

Three tiers — full pipeline. Each GPU handles a different capability level.

### Cloud-only (no local hardware)

```
large:  claude-sonnet via API (inference, review)
small:  claude-haiku via API  (utility — cheap, fast)
```

Two tiers — cloud profiles substitute for local models. Cost gates from `LaneBackendManager` still apply.

---

## What This Unlocks

### For benchmarks (immediate)
- Narrative transcriber runs on `small` tier, always available
- Benchmark runner swaps `large` model between runs while `small` stays constant
- `function="narrative_transcription"` in the transcriber code — clean, self-documenting

### For generative campaigns
- Narrator LLM runs on `medium` or `large`, entity naming on `small`
- Dynamic tier assignment based on narrative complexity

### For DM campaigns
- Choice classification on `small` (fast, cheap)
- NPC dialogue generation on `large` (quality)
- Dice resolution narration on `medium`

### For embodiment
- Cerebellum forward model queries on `small` (fast, frequent)
- Novel percept generation (no forward model) on `large`

### For general runtime
- JSON repair on `small` instead of burning `large` capacity
- Concept extraction stays on `small` (no change from current `record` lane)
- Fear review on `large` with `medium` fallback under load

---

## Implementation Phases

| Phase | What | LOC | Touches |
|-------|------|-----|---------|
| 1 | `FunctionSpec` + `FunctionRouter` + default function table | ~150 | New file: `runtime/function_router.py` |
| 2 | `DEFAULT_TIERS` + backward-compat aliases + `detect_tiers()` | ~80 | `runtime/worker_pool.py` |
| 3 | `WorkerPool.submit_function()` + `LLMRouter` function routing | ~100 | `runtime/worker_pool.py`, `models/language/router.py` |
| 4 | Migrate callers: `lane="infer"` → `function="agent_inference"` etc. | ~30 | `agents/llm_worker.py`, `memory/concept_*.py`, `agents/fear_agent.py` |
| 5 | Config: `llm_lanes.json` loader + CLI `--tier-model` flag | ~40 | `runtime/lane_backends.py`, `cli_parser.py` |
| 6 | Tests: fallback chains, restrictions, backward compat, multi-tier | ~150 | New: `tests/unit/test_function_router.py` |
| **Total** | | **~550** | |

**Session 1:** Phases 1-3 (~330 LOC) — FunctionRouter + tier config + routing wired. End-to-end `function="agent_inference"` working. Backward compat for `lane="infer"`.

**Session 2:** Phases 4-6 (~220 LOC) — caller migration + config file + tests. Clean cutover.

## Risks

1. **WorkerPool thread model.** Each tier has its own thread pool. Adding `small` tier adds one thread. Acceptable — it's CPU-only, not a GPU resource.

2. **SmolLM startup latency.** First `small` tier call triggers model load (~5s for 1.7B). Mitigation: lazy load is fine — utility tasks aren't latency-critical on the first call.

3. **Fallback cascades under load.** If `large` is saturated and many functions fall back to `medium`, `medium` can also saturate. Mitigation: queue depth limits per tier + clear error messages. The system should fail fast rather than cascade into a traffic jam.

4. **Function name proliferation.** As more consumers add functions, the routing table grows. Mitigation: keep the default table minimal (~12 functions). Custom functions can be added via config file.

5. **Backward compat edge cases.** Code that checks `lane.name == "infer"` breaks. Mitigation: grep for string comparisons against lane names and update. Limited blast radius (~4 callsites).

## Ties to Other Plans

| Plan | Relationship |
|------|-------------|
| [Benchmark plan](benchmark_plan.md) | **Primary consumer.** Narrative transcriber uses `small` tier. Benchmark runner swaps `large` model per run. |
| [Generative campaign plan](generative_campaign_plan.md) | Narrator on `large`/`medium`, entity naming on `small` |
| [Embodiment core plan](embodiment_core_plan.md) | Cerebellum queries on `small`, novel percept generation on `large` |
| [DM persona plan](dungeon_master_persona.md) | Choice classification on `small`, NPC dialogue on `large` |
| [Agent mesh plan](agent_mesh.md) | Per-peer tier availability affects function routing in mesh context |

## When to Implement

**Before benchmarks.** The benchmark plan's narrative transcriber is the first consumer of the `small` tier. Building the tier system first means benchmarks get clean function-based routing from day one instead of ad-hoc lane assignment.

Recommended order:
1. Lane tier refactor (this plan, ~550 LOC)
2. Benchmark Phases 1-2 (~400 LOC) — includes narrative transcriber using `function="narrative_transcription"`
3. Benchmark Phases 3-5 (~300 LOC + YAML) — scenarios + output
