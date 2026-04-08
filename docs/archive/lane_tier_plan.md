# Lane Tier Architecture — Size-Based Model Routing

> **Status:** Archived — complete. All 8 phases shipped across 2 sessions (2026-04-06).
> **Scope:** ~820 LOC across 8 phases.
> **Depends on:** Multi-LLM Scaling (done), WorkerPool + LaneConfig + LaneBackendManager (all exist).

---

## Problem

The current lane system maps **functions to lanes**: `infer`, `review`, `record`. Each lane gets one model. This worked for 3 lanes but doesn't scale:

- Adding a 4th function (narrative transcription) means: new lane? Reuse `record`? The naming breaks down.
- `record` (concept extraction) and a proposed `utility` lane would run the same model — why separate?
- No fallback logic: if `infer` is GPU-busy, `review` can't help even though it runs the same model.
- No way to restrict functions from small models (e.g., "never run agent inference on 1.7B") without hardcoding lane checks in every caller.
- Every new consumer (benchmarks, DM choice classification, entity naming, JSON repair) needs a lane decision that's really a model-size decision.
- The `infer`/`infer_net` split in `LLMWorker` bakes cloud vs. local routing into the caller instead of the infrastructure.

## Solution

Replace function-based lanes with **size-tier lanes**. Functions are **routed to tiers** via a configurable mapping with fallback chains and restrictions. Cloud/local dispatch moves inside the tier backend (handled by `LaneBackendManager`), removing the `infer_net` workaround from `LLMWorker`.

### Current vs proposed

```
Current (function = lane):
  infer     lane → mistral-7b GPU  → agent loop inference (local)
  infer_net lane → claude-sonnet   → agent loop inference (cloud)
  review    lane → smollm-1.7b CPU → concept grounding, relationship formation
  record    lane → (shared)        → concept extraction, DB writes

Proposed (tier = capability level):
  large tier → 14B+ GPU/cloud      → inference, fear review, paper writing
  medium tier → 7B GPU/CPU         → analysis, campaign summary, review fallback
  small tier → 1.7B-3B CPU         → transcription, classification, extraction, JSON repair
```

Each tier's backend can dispatch to local, peer, or cloud providers internally — the `infer_net` split is absorbed into the tier's backend manager. Functions declare what tier they need. The router handles placement, fallback, and restrictions.

### Function routing table

```yaml
# Default function → tier mapping (configurable via llm.json "tiers" key or env)
functions:
  # Core agent operations — need strong reasoning
  agent_inference:
    tier: large
    fallback: []                    # RESTRICTED — never fall back to weaker model
    priority_boost: 2               # Higher priority within tier queue
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

  sim_orchestrator:
    tier: large
    fallback: [medium]
    description: "Simulation orchestrator LLM (drives sim agent loop)"

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

  relationship_formation:
    tier: small
    fallback: [medium]
    description: "Concept extractor inline relationship formation"

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
4. Track fallback frequency in `LaneMetrics` (uses existing `failover_count` field)

**Static vs. dynamic fallback** (phased):
- **Phase 1 (this plan):** Static availability — `resolve()` checks whether a tier exists in the deployment. Covers the 90% case: "this machine has no GPU, so `large` falls back to `medium`."
- **Phase 2 (future, mesh-aligned):** Dynamic availability — `resolve()` queries queue depth + backend health at call time. Covers: "GPU is loaded, fall back to peer or cloud." This integrates with `InferenceRouter` from the agent mesh plan (Phase 0b).

### Restrictions and priority

Functions with `fallback: []` are **hard-restricted** to their declared tier. This prevents:
- Agent inference on a 1.7B model (would produce garbage)
- Narrative transcription bouncing to large (wastes GPU for a trivial task)

**Priority within a tier:** Functions with `fallback: []` (restricted) get a priority boost of +2 within their tier's queue. This prevents restricted functions from starving behind degradable ones that could move to another tier. The `priority_boost` field is configurable per function.

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
        max_workers=2,              # Multiple concurrent consumers (extraction + transcription + etc.)
        requires_gpu=False,
        model_profile="smollm-1.7b",
        device="cpu",
        n_gpu_layers=0,
    ),
}
```

**Flexible tier count:** Not every deployment needs 3 tiers. A peer machine with no GPU might have only `medium` + `small`. The leader with an RTX 5080 might have `large` + `small` (skipping medium since large handles both). The function router adapts — if a function's tier doesn't exist, it follows the fallback chain.

**Note:** `small` gets `max_workers=2` (matching current `record` lane) because many functions target it concurrently (narrative transcription, JSON repair, concept extraction, entity naming). A single worker would serialize all of these, creating a bottleneck.

### Absorbing `infer_net` into tier backends

Currently `LLMWorker` creates two lanes (`infer` + `infer_net`) and routes cloud calls to `infer_net` based on provider type. This logic moves into `LaneBackendManager`:

```python
# In lane_backends.py — each tier's backend can dispatch to local or cloud
class LaneBackendManager:
    def get_backend(self, tier: str) -> LLMRouter:
        """Get the backend for a tier.

        If the tier has both a local model and cloud providers configured,
        the returned LLMRouter handles provider selection internally
        (via its existing RoutingPolicy fallback chains).
        """
        ...
```

`LLMWorker` no longer needs `_INFER_LANES`, `infer_net`, or the `_has_cloud_providers()` check. It submits to a tier name; the backend decides local vs. cloud. This is a net simplification of ~30 LOC in `llm_worker.py`.

### FunctionRouter (new, ~180 LOC)

```python
@dataclass
class FunctionSpec:
    """How a named function maps to the tier system."""
    name: str
    tier: str                           # primary tier
    fallback: list[str]                 # ordered fallback tiers (empty = restricted)
    priority_boost: int = 0             # added to job priority for restricted functions
    description: str = ""

class FunctionRouter:
    """Routes named functions to tier-based lanes with fallback.

    Phase 1: static availability (tier exists in deployment).
    Phase 2 (mesh): dynamic availability via health_check + callable tiers.

    Mesh-ready: available_tiers accepts either a static set or a callable
    that returns the current set. When PeerRegistry is active, the callable
    queries local tiers + peer-advertised tiers. When mesh is off, a plain
    set is wrapped in a lambda — zero overhead.
    """

    def __init__(
        self,
        functions: dict[str, FunctionSpec],
        available_tiers: set[str] | Callable[[], set[str]],
        *,
        tier_order: list[str] | None = None,
        health_check: Callable[[str], bool] | None = None,
    ) -> None:
        self._functions = functions
        # Callable or static set — normalize to callable for uniform access
        self._get_available: Callable[[], set[str]] = (
            available_tiers if callable(available_tiers)
            else lambda _tiers=available_tiers: _tiers
        )
        # Configurable ordering — supports adding tiers (e.g., "large_plus")
        # without code changes. Three tiers is the default; the ordering is
        # a capability class, not a model size (a strong 10B is "large").
        self._tier_order = tier_order or ["large", "medium", "small"]
        self._health_check = health_check  # Phase 2: dynamic check
        self._fallback_counts: dict[str, int] = {}  # function → count

    def resolve(self, function: str) -> tuple[str, int]:
        """Return (tier_name, priority_boost) for this function.

        Tries primary tier, then fallbacks. Raises if no tier available
        and function has no fallback (restricted).
        """
        spec = self._functions.get(function)
        if spec is None:
            # Unknown function — default to largest available tier
            return self._largest_available(), 0

        if self._tier_available(spec.tier):
            return spec.tier, spec.priority_boost

        for fallback_tier in spec.fallback:
            if self._tier_available(fallback_tier):
                self._fallback_counts[function] = self._fallback_counts.get(function, 0) + 1
                logger.info("Function %s: primary tier %s unavailable, falling back to %s",
                           function, spec.tier, fallback_tier)
                return fallback_tier, spec.priority_boost

        if not spec.fallback:
            raise TierRestrictionError(
                f"Function '{function}' requires tier '{spec.tier}' "
                f"(restricted, no fallback). Available: {self._get_available()}"
            )
        raise TierUnavailableError(
            f"No tier available for function '{function}'. "
            f"Tried: {spec.tier} → {spec.fallback}. Available: {self._get_available()}"
        )

    def _tier_available(self, tier: str) -> bool:
        """Check if a tier is available (static + optional dynamic health)."""
        if tier not in self._get_available():
            return False
        if self._health_check is not None:
            return self._health_check(tier)
        return True

    def _largest_available(self) -> str:
        available = self._get_available()
        for tier in self._tier_order:
            if tier in available:
                return tier
        raise TierUnavailableError("No tiers available")

    def fallback_stats(self) -> dict[str, int]:
        """Fallback frequency per function — feeds LaneMetrics/diagnostics."""
        return dict(self._fallback_counts)
```

**Mesh integration:** `available_tiers` as a callable is the zero-cost integration point for the agent mesh. When `PeerRegistry` is active:
- The callable queries local tiers + peer-advertised tiers (via `AgentIdentity.inference_models`)
- `health_check` queries `InferenceRouter` for per-tier backend health (latency, failure rate)
- When mesh is off, the plain set wrapped in a lambda has no runtime cost

**Tier ordering extensibility:** `tier_order` is configurable at construction. Three named tiers (`large`, `medium`, `small`) are the default — these are capability *classes*, not model sizes. A strong 10B model is `large`; a mediocre 10B is `medium`. If we later need a 4th tier (e.g., `"large_plus"` for 70B+ models), it's a config change, not a code change.

**Locality vs. capability:** The tier system handles *capability* routing; *locality* routing (local → peer → tunnel) is handled by `InferenceRouter` (mesh Phase 0b) at the backend level. The tier router picks the right capability class, then the backend decides whether to serve it locally or remotely. This separation means `fear_review` always gets a `large`-capable model, and `InferenceRouter` decides whether that's the local GPU or a peer's GPU.

### WorkerPool changes (~60 LOC)

```python
class WorkerPool:
    def submit_function(
        self,
        function: str,
        fn: Callable,
        *,
        job_id: str | None = None,
        priority: int = 5,
        deps: DependencySpec | None = None,
    ) -> str:
        """Submit a job by function name. FunctionRouter resolves the tier."""
        tier, priority_boost = self._function_router.resolve(function)
        return self.submit(fn, lane=tier, priority=priority + priority_boost, deps=deps)
```

Existing `submit(..., lane="infer")` still works — it maps directly to a tier name via legacy aliases. New callers use `submit_function("agent_inference", ...)`.

### LLMRouter changes (~50 LOC)

```python
class LLMRouter:
    def generate_json(self, prompt, *, function: str = "agent_inference", **kw):
        """Generate JSON using the appropriate tier for this function."""
        tier, _ = self._function_router.resolve(function)
        backend = self._backend_manager.get_backend(tier)
        return backend.generate_json(prompt, **kw)
```

Callers that currently pass `lane="review"` switch to `function="fear_review"`. Backward compat: if `lane=` is passed, it's treated as a direct tier name (via alias).

---

## Backward Compatibility

The refactor must not break existing code. Strategy:

1. **`lane=` parameter still works everywhere.** If a caller passes `lane="infer"`, it maps to `"large"`. A backward-compat alias dict handles this:
   ```python
   _LEGACY_LANE_ALIASES = {
       "infer": "large",
       "infer_net": "large",   # Cloud dispatch now internal to tier backend
       "review": "small",      # Current review callers do concept grounding (small-tier work)
       "record": "small",      # Current record callers do concept extraction (small-tier work)
   }
   ```

   **Rationale for `review → small`:** The current `review` lane callers are `concept_grounder.py` (concept grounding) and `concept_extractor.py` (relationship formation) — both structured-output tasks that the function table assigns to `small` tier. Mapping `review → large` would waste GPU on tasks that should run on CPU.

   **Rationale for `infer_net → large`:** Cloud inference is now handled *within* the `large` tier's backend. The `infer_net` lane is absorbed, not renamed.

2. **`DEFAULT_LANES` still works.** Code that constructs `WorkerPool(lane_configs=DEFAULT_LANES)` keeps working because `DEFAULT_LANES` becomes an alias for `DEFAULT_TIERS`:
   ```python
   DEFAULT_TIERS = { ... }
   DEFAULT_LANES = DEFAULT_TIERS  # Backward compat alias
   ```

3. **Gradual migration.** Callers can switch from `lane="review"` to `function="concept_grounding"` incrementally. Both work simultaneously.

4. **`_INFER_LANES` in LLMWorker.** Currently `("infer", "infer_net")` — both poll for completed proposals. After migration, only the `"large"` tier needs polling. The `_INFER_LANES` tuple becomes `("large",)` and the `infer_net` branch is removed.

---

## Configuration

### Default: auto-detect from hardware (builds on existing infrastructure)

Tier detection reuses the existing capability stack — no new hardware probing:

- **`RuntimeCapabilities`** (`runtime/capabilities.py`) — `has_gpu`, `gpu_type`, `vram_gb`, `ram_gb`, MPS detection for Mac
- **`PlatformInfo`** (`doctor/platform_detect.py`) — OS, runtime (native/WSL/docker), arch
- **`_pick_infer_profile()`** (`runtime/lane_models.py`) — VRAM-tier → model profile mapping
- **`detect_compute_resources()`** (`runtime/capabilities.py`) — GPU probe respecting `CUDA_VISIBLE_DEVICES`

```python
def detect_tiers(caps: RuntimeCapabilities | None = None) -> dict[str, LaneConfig]:
    """Auto-detect available tiers based on hardware.

    Delegates VRAM-based profile selection to _pick_infer_profile() in
    lane_models.py (reuses existing _INFER_VRAM_TIERS table). Does NOT
    duplicate hardware detection — RuntimeCapabilities is the single
    source of truth.

    Called at startup by LaneBackendManager. Also called by `maxim doctor`
    to report tier availability (check_tier_detection).
    """
    if caps is None:
        from maxim.runtime.capabilities import detect_compute_resources, RuntimeCapabilities
        has_gpu, gpu_type, vram_gb, ram_gb = detect_compute_resources()
        caps = RuntimeCapabilities(
            has_gpu=has_gpu, gpu_type=gpu_type, vram_gb=vram_gb, ram_gb=ram_gb,
        )

    tiers = {}

    # Always available: small (CPU, ~2GB RAM)
    tiers["small"] = LaneConfig(
        name="small", max_workers=2,
        model_profile="smollm-1.7b", device="cpu", n_gpu_layers=0,
    )

    # GPU available? Add large tier (profile selected by VRAM)
    if caps.has_gpu and caps.vram_gb >= 4.0:
        from maxim.runtime.lane_models import _pick_infer_profile
        profile = _pick_infer_profile(caps.vram_gb)
        tiers["large"] = LaneConfig(
            name="large", max_workers=1,
            requires_gpu=True, model_profile=profile, device="gpu",
        )
    elif caps.has_gpu and caps.gpu_type == "mps":
        # Mac with MPS: unified memory GPU — treat as medium-capable
        tiers["medium"] = LaneConfig(
            name="medium", max_workers=1,
            model_profile="mistral-7b-instruct-v0.2", device="auto",
        )

    # No GPU? Use CPU if enough RAM for a 7B model
    if not caps.has_gpu and caps.ram_gb > 8:
        tiers["medium"] = LaneConfig(
            name="medium", max_workers=1,
            model_profile="mistral-7b-instruct-v0.2", device="cpu",
        )

    # No GPU and not enough RAM for medium? Only small exists
    if len(tiers) == 1:
        # agent_inference requires large (restricted, no fallback) — warn user
        logger.warning(
            "Only 'small' tier detected (no GPU, low RAM). "
            "Agent inference requires --language-model or --cloud-fallback."
        )

    return tiers
```

**Platform handling:**
- **RTX 5080 leader:** `has_gpu=True`, `vram_gb≈16` → `large` (qwen2.5-14b via `_pick_infer_profile`) + `small`
- **Mac peer (24GB unified):** `gpu_type="mps"` → `medium` (mistral-7b, device=auto uses MPS) + `small`
- **CPU-only Linux (16GB RAM):** `has_gpu=False`, `ram_gb=16` → `medium` (mistral-7b CPU) + `small`
- **Raspberry Pi (4GB RAM):** only `small` — needs `--cloud-fallback` for agent inference

### Doctor integration: `check_tier_detection`

`maxim doctor` gains a new check that reports tier availability, validating the auto-detection logic and giving users a clear picture of what their hardware supports:

```python
# In doctor/checks.py
def check_tier_detection() -> CheckResult:
    """Report which tiers are available on this hardware."""
    from maxim.runtime.lane_models import detect_tiers
    from maxim.runtime.capabilities import detect_compute_resources, RuntimeCapabilities

    has_gpu, gpu_type, vram_gb, ram_gb = detect_compute_resources()
    caps = RuntimeCapabilities(
        has_gpu=has_gpu, gpu_type=gpu_type, vram_gb=vram_gb, ram_gb=ram_gb,
    )
    tiers = detect_tiers(caps)

    tier_names = sorted(tiers.keys())
    profiles = {name: cfg.model_profile for name, cfg in tiers.items()}

    if "large" in tiers or "medium" in tiers:
        return CheckResult(
            name="LLM Tiers",
            status="ok",
            message=f"Tiers: {', '.join(tier_names)}. Profiles: {profiles}",
        )
    else:
        return CheckResult(
            name="LLM Tiers",
            status="warn",
            message=f"Only 'small' tier detected ({ram_gb:.0f}GB RAM, GPU: {gpu_type or 'none'})",
            fix=(
                "Agent inference needs a large or medium tier. Options:\n"
                "  --language-model mistral-7b          # if you have 8+ GB RAM\n"
                "  --cloud-fallback claude-sonnet       # use cloud for inference\n"
                "  --tier-model large=<remote-url>      # point to a remote leader"
            ),
            retry_id="tier_detection",
        )
```

This check fits into the existing "GPU / CUDA" section of `run_all_checks()`. It runs after `check_gpu()` and before the server checks, so users see the full picture: GPU → tiers → server → network.

**Future doctor expansions tied to tiers** (from [future_plans.md](../plans/future_plans.md) "Doctor Enhancements"):
- **Inference coherence** (§5): `maxim doctor benchmark` runs a fixed prompt on each tier, reports tokens/sec + coherence per tier
- **Peer capability audit** (§9): compare tier availability across mesh peers, flag weak links
- **Sim pre-flight** (cross-cutting): before launching a sim, verify the tiers needed by the sim's functions are available

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

### Config file: extend existing `data/util/llm.json`

Tier configuration lives in the existing `llm.json` under a new `"tiers"` key — no new config file. This keeps all LLM config in one place.

```json
{
  "default_profile": "mistral-7b-instruct-v0.2",
  "tiers": {
    "large": {"model_profile": "qwen2.5-14b", "device": "gpu"},
    "small": {"model_profile": "smollm-1.7b", "device": "cpu", "max_workers": 2}
  },
  "functions": {
    "fear_review": {"tier": "medium", "fallback": ["large"]},
    "concept_extraction": {"tier": "small", "fallback": ["medium"]}
  }
}
```

---

## LaneMetrics Migration

`LaneMetrics` are keyed by lane name. Renaming lanes changes the keys that `heartbeat.py`, `leader_proxy.py`, `doctor/checks.py`, and `lane_backends.py` query. Migration strategy:

1. **MetricsRegistry keys become tier names.** `get_metrics_registry().get("large")` replaces `get("infer")`.
2. **Alias lookup in MetricsRegistry.** For backward compat during migration, `get("infer")` returns the `"large"` metrics (using `_LEGACY_LANE_ALIASES`). Logged as deprecation warning.
3. **Update consumers (Phase 4).** Heartbeat, LeaderProxy admission control, `maxim doctor` output — all switch to tier names. Limited blast radius (~6 callsites across 3 files).
4. **Historical metrics.** Counters reset on restart anyway (they're in-memory). No persistence migration needed.

---

## Hardware Scenarios

### Leader (RTX 5080 + 16GB VRAM + 16GB CPU RAM)

```
large:  qwen2.5-14b on GPU (inference, fear review, paper writing)
small:  smollm-1.7b on CPU, 2 workers (transcription, classification, extraction)
```

Two tiers — `medium` not needed because `large` handles its functions directly.

### Mac Peer (24GB unified memory, no discrete GPU)

```
medium: mistral-7b on CPU (inference via fallback, review — no GPU available)
small:  smollm-1.7b on CPU, 2 workers (utility tasks)
```

Two tiers — `large` unavailable (no GPU). Agent inference falls back from `large` to `medium` via fallback chain. Functions restricted to `large` with no fallback would error with a clear message — user should configure `--cloud-fallback claude-sonnet` to provide a remote `large` tier.

### Dual-GPU server (future)

```
large:  qwen2.5-14b on GPU:0, 1 worker (inference)
medium: mistral-7b on GPU:1, 1 worker (review, analysis)
small:  smollm-1.7b on CPU, 2 workers (utility)
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

### For agent mesh
- Per-peer tier availability advertised in `AgentIdentity.inference_models`
- `FunctionRouter(available_tiers=callable)` dynamically includes peer-advertised tiers
- `FunctionRouter.health_check` callback integrates with `InferenceRouter` (mesh Phase 0b)
- Mesh-aware routing: "peer-B has a `large` tier I don't" → callable returns `{"large", "small"}`, functions route to peer-B's large tier
- Locality routing (local vs. peer vs. tunnel) handled by `InferenceRouter`, not the tier system

### For general runtime
- JSON repair on `small` instead of burning `large` capacity
- Concept extraction stays on `small` (no change from current `record` lane)
- Fear review on `large` with `medium` fallback under load
- `infer_net` eliminated — cloud dispatch is internal to tier backend

---

## Implementation Phases

| Phase | What | LOC | Touches |
|-------|------|-----|---------|
| 1 | `FunctionSpec` + `FunctionRouter` + default function table + exceptions | ~180 | New file: `runtime/function_router.py` |
| 2 | `DEFAULT_TIERS` + `_LEGACY_LANE_ALIASES` + backward-compat alias resolution in `WorkerPool.submit()` | ~80 | `runtime/worker_pool.py` |
| 3 | `detect_tiers()` delegating to `lane_models.py` + `RuntimeCapabilities` + MPS handling | ~80 | `runtime/lane_models.py` |
| 4 | `WorkerPool.submit_function()` + `LLMRouter` function routing | ~100 | `runtime/worker_pool.py`, `models/language/router.py` |
| 5 | Migrate callers + remove `infer_net` from LLMWorker + update `_INFER_LANES` | ~120 | `agents/llm_worker.py`, `memory/concept_extractor.py`, `memory/concept_grounder.py` |
| 6 | Config: `llm.json` tier loader + CLI `--tier-model` flag + LaneMetrics key migration | ~60 | `runtime/lane_backends.py`, `cli_parser.py`, `models/language/lane_metrics.py` |
| 7 | Doctor: `check_tier_detection` + add to `run_all_checks()` | ~40 | `doctor/checks.py`, `doctor/cli.py` |
| 8 | Tests: fallback chains, restrictions, priority boost, backward compat, multi-tier, `infer_net` removal, doctor check | ~160 | New: `tests/unit/test_function_router.py`, updates to `tests/unit/test_worker_pool.py`, `tests/unit/test_doctor.py` |
| **Total** | | **~820** | |

### Session plan

**Session 1:** Phases 1-4 (~440 LOC) — FunctionRouter + tier config + routing wired. End-to-end `function="agent_inference"` working. Backward compat for `lane="infer"`. `detect_tiers()` uses existing hardware detection.

**Session 2:** Phases 5-8 (~380 LOC) — caller migration, `infer_net` removal, config file, LaneMetrics migration, doctor check, full test suite.

## Risks

1. **`infer_net` removal in LLMWorker.** The `infer`/`infer_net` split currently serves a real purpose: cloud calls go through a separate semaphore-gated lane so a slow Anthropic API call doesn't block the local GPU queue. **Mitigation:** The `large` tier's `LLMRouter` already has provider-level routing with its own rate limiting. `LaneBackendManager` applies the existing `MAXIM_MAX_CLOUD_LANES` gate. The semaphore-per-provider pattern in `LLMWorker._init_provider_semaphores()` can stay — it just operates within the tier backend rather than as a separate lane.

2. **WorkerPool thread model.** Each tier has its own thread pool. Adding `small` tier adds ≤2 threads. Acceptable — they're CPU-only, not GPU resources.

3. **SmolLM startup latency.** First `small` tier call triggers model load (~5s for 1.7B). Mitigation: lazy load is fine — utility tasks aren't latency-critical on the first call.

4. **Fallback cascades under load.** If `large` is saturated and many functions fall back to `medium`, `medium` can also saturate. Mitigation: queue depth limits per tier + clear error messages + priority boost for restricted functions. The system should fail fast rather than cascade into a traffic jam. Dynamic fallback (Phase 2, future) will add queue-depth awareness.

5. **Function name proliferation.** As more consumers add functions, the routing table grows. Mitigation: keep the default table minimal (~13 functions). Custom functions can be added via config file. Unknown functions default to largest available tier with a logged warning.

6. **Backward compat edge cases.** Code that checks `lane.name == "infer"` breaks. Mitigation: grep for string comparisons against lane names and update. Limited blast radius (~8 callsites across 4 files, including tests).

7. **LaneMetrics key migration.** Heartbeat, LeaderProxy, and doctor all query metrics by lane name. Mitigation: Phase 6 includes alias lookup in MetricsRegistry + consumer updates. Counters are in-memory only (no persistence migration).

---

## Design Decisions

Resolved architectural choices and their rationale.

### Mesh-aware tier routing: callable `available_tiers`

`FunctionRouter` accepts `available_tiers` as either a `set[str]` or a `Callable[[], set[str]]`. When the agent mesh is active, `PeerRegistry` provides a callable that returns local tiers + peer-advertised tiers. When mesh is off, a plain set is wrapped in a lambda — zero overhead.

This is ~10 LOC in Phase 1 and avoids rebuilding the `FunctionRouter` when peers join/leave. The callable approach was chosen over static-set-with-rebuild because it's simpler, thread-safe (the callable can hold its own lock), and doesn't require coordinating router reconstruction across the startup path.

### Locality routing deferred to `InferenceRouter`

The tier system handles *capability* routing (which tier can serve this function). *Locality* routing (prefer local GPU over remote peer over tunnel) is handled by `InferenceRouter` (mesh Phase 0b) at the backend level, below the tier router.

This separation means:
- `fear_review` always gets a `large`-capable model (tier router decides)
- Whether that's the local GPU or peer-B's GPU is decided by `InferenceRouter`
- Functions restricted to `small` with `fallback: []` (like `json_repair`) never bounce to a remote `large` — the tier router prevents it before locality even comes into play

No `prefer_local` field on `FunctionSpec` is needed. The `InferenceRouter`'s routing chain (local → LAN peer → tunnel → cloud) already encodes the locality preference at the right layer.

### Three named tiers, extensible via `tier_order`

Tiers are capability *classes*, not model sizes. A strong 10B model is `large`; a mediocre 10B is `medium`. Three tiers (`large`, `medium`, `small`) cover current deployments. If 4+ tiers are needed (e.g., `"large_plus"` for 70B+ models on a multi-GPU server), `tier_order` is configurable at `FunctionRouter` construction — adding a tier is a config change, not a code change.

Numeric capability scores (`large=3`, `medium=2`, `small=1`) were considered and rejected — they're more flexible but harder to read in configs, logs, and doctor output. Named tiers map directly to the mental model ("this function needs a big model").

### Cost gating at the backend level, not the tier level

The tier router picks the right capability class. Cost gating is handled below that by existing infrastructure:
- `LaneBackendManager`: `MAXIM_MAX_CLOUD_LANES`, `MAXIM_MAX_CONCURRENT_BACKENDS`
- `LLMRouter`: `CostTracker`, `MAXIM_CLOUD_SESSION_BUDGET`

These gates already prevent runaway cloud spending. Adding per-function cost caps (e.g., `max_cost_per_call: float` on `FunctionSpec`) is a future option if cloud-heavy deployments show cost issues, but is not needed for Phase 1. The tier router's job is capability matching; the backend's job is cost enforcement.

---

## Ties to Other Plans

| Plan | Relationship |
|------|-------------|
| [Benchmark plan](benchmark_plan.md) | **Primary consumer.** Narrative transcriber uses `small` tier. Benchmark runner swaps `large` model per run. |
| [Generative campaign plan](generative_campaign_plan.md) | Narrator on `large`/`medium`, entity naming on `small` |
| [Embodiment core plan](embodiment_core_plan.md) | Cerebellum queries on `small`, novel percept generation on `large` |
| [DM persona plan](dungeon_master_persona.md) | Choice classification on `small`, NPC dialogue on `large` |
| [Agent mesh plan](agent_mesh.md) | Per-peer tier availability via callable `available_tiers`. `FunctionRouter.health_check` integrates with `InferenceRouter`. `AgentIdentity.inference_models` advertises available tiers. Locality routing handled by `InferenceRouter`, not the tier system. |

## When to Implement

**Before benchmarks.** The benchmark plan's narrative transcriber is the first consumer of the `small` tier. Building the tier system first means benchmarks get clean function-based routing from day one instead of ad-hoc lane assignment.

Recommended order:
1. Lane tier refactor (this plan, ~820 LOC)
2. Benchmark Phases 1-2 (~400 LOC) — includes narrative transcriber using `function="narrative_transcription"`
3. Benchmark Phases 3-5 (~300 LOC + YAML) — scenarios + output
