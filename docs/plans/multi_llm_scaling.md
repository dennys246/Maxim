# Multi-LLM Scaling Plan

> **Status:** Not started. Depends on RuntimeCapabilities (implemented) and WorkerPool lane system (implemented).

Run heterogeneous LLM backends within the same WorkerPool: a primary model on GPU for main inference, and a smaller model on CPU for background evaluation/review tasks. RuntimeCapabilities detection determines what's available; lane configuration adapts automatically.

---

## Motivation

Currently all WorkerPool lanes share a single LLM backend. This means:
- GPU inference blocks CPU-appropriate review tasks
- No way to run a tiny model for low-priority background work while the main model handles inference
- CPU-only machines use the same model for everything despite having no VRAM constraints

With per-lane model assignment, the infer lane can run a larger quantized model on GPU while the review lane runs a tiny model on CPU — no VRAM contention, parallel execution.

---

## Implementation

### Phase 1: LaneConfig Gains Model Fields

Add `model_profile`, `device`, and `n_gpu_layers` to `LaneConfig`:

```python
@dataclass
class LaneConfig:
    name: str
    max_workers: int
    queue_size: int = 10
    requires_gpu: bool = False
    # Per-lane model assignment
    model_profile: str | None = None   # LLM profile name
    device: str = "auto"               # "gpu", "cpu", "auto"
    n_gpu_layers: int = -1             # -1 = all on GPU, 0 = CPU only
```

### Phase 2: Capability-Driven Model Assignment

```python
@dataclass(frozen=True)
class LaneModelConfig:
    profile: str
    quantization: str = "Q4_K_M"
    device: str = "auto"
    n_gpu_layers: int = -1


def build_lane_model_config(caps: RuntimeCapabilities) -> dict[str, LaneModelConfig]:
    """Map WorkerPool lanes to model profiles based on hardware."""
    if caps.has_gpu:
        return {
            "infer": LaneModelConfig(
                profile="phi-3-mini-4k-instruct",
                quantization="Q4_K_M",
                device="gpu",
                n_gpu_layers=-1,
            ),
            "review": LaneModelConfig(
                profile="smollm-1.7b-instruct",
                quantization="Q4_K_M",
                device="cpu",
                n_gpu_layers=0,
            ),
        }
    else:
        return {
            "infer": LaneModelConfig(
                profile="smollm-1.7b-instruct",
                quantization="Q4_K_M",
                device="cpu",
                n_gpu_layers=0,
            ),
            "review": LaneModelConfig(
                profile="smollm-1.7b-instruct",
                quantization="Q4_K_M",
                device="cpu",
                n_gpu_layers=0,
            ),
        }
```

**Memory warning:** Loading 2 LLM models simultaneously (GPU + CPU) may exhaust system RAM even if VRAM is managed. Monitor total memory usage during dual-model operation. Consider lazy unloading of CPU model when not actively processing review jobs.

### Phase 3: LaneBackendManager

Per-lane LLM backend creation with lazy loading and thread-safe caching:

```python
class LaneBackendManager:
    """Manages per-lane LLM backends based on LaneConfig.

    Each lane with a model_profile gets a dedicated backend instance.
    Backends are lazy-loaded on first use. CPU backends can coexist
    with GPU backends since they don't compete for VRAM.
    """

    def __init__(self, lane_configs: dict[str, LaneConfig]) -> None:
        self._backends: dict[str, LLMBackend] = {}
        self._configs = lane_configs
        self._lock = threading.Lock()

    def get_backend(self, lane: str) -> LLMBackend | None:
        """Get or create the backend for a lane."""
        config = self._configs.get(lane)
        if not config or not config.model_profile:
            return None
        with self._lock:
            if lane not in self._backends:
                self._backends[lane] = self._create_backend(config)
            return self._backends[lane]

    def _create_backend(self, config: LaneConfig) -> LLMBackend:
        from maxim.models.language.router import load_llm_config, LLMRouter
        llm_config = load_llm_config(profile_override=config.model_profile)
        llm_config = dataclasses.replace(llm_config, n_gpu_layers=config.n_gpu_layers)
        return LLMRouter(cfg=llm_config)

    def unload_all(self) -> None:
        with self._lock:
            for backend in self._backends.values():
                try:
                    backend.unload()
                except Exception:
                    pass
            self._backends.clear()
```

**Thread-local device isolation:** For llama-cpp backends, GPU layer count is set at model load time (not per-thread). CPU-only models naturally don't touch VRAM. For PyTorch backends, device placement is set via the `device` parameter. No `CUDA_VISIBLE_DEVICES` manipulation needed for single-GPU setups — just `n_gpu_layers=0` for CPU lanes.

### Phase 4: Wire into Agentic Runtime + LLMWorker

```python
# In agentic_runtime.py, after RuntimeCapabilities detection:
lane_model_configs = build_lane_model_config(self._capabilities)

from maxim.runtime.worker_pool import LaneConfig, DEFAULT_LANES
lane_configs = dict(DEFAULT_LANES)
for lane_name, model_cfg in lane_model_configs.items():
    if lane_name in lane_configs:
        lane_configs[lane_name] = dataclasses.replace(
            lane_configs[lane_name],
            model_profile=model_cfg.profile,
            device=model_cfg.device,
            n_gpu_layers=model_cfg.n_gpu_layers,
        )

pool = WorkerPool(lane_configs=lane_configs)
backend_manager = LaneBackendManager(lane_configs)
# LLMWorker uses backend_manager.get_backend(lane) for each job
```

### Phase 5: Per-Lane Metrics

```python
@dataclass
class LaneMetrics:
    """Per-lane performance counters."""
    jobs_completed: int = 0
    jobs_dropped: int = 0
    total_latency_ms: float = 0.0
    peak_queue_depth: int = 0

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / max(self.jobs_completed, 1)
```

Exposed via enriched `WorkerPool.status()`.

### Phase 6: Environment Variable / Config Support

```bash
MAXIM_INFER_PROFILE=phi-3-mini-4k-instruct
MAXIM_REVIEW_PROFILE=smollm-1.7b-instruct
MAXIM_INFER_DEVICE=gpu
MAXIM_REVIEW_DEVICE=cpu
```

Or via `llm.json`:
```json
{
  "lane_models": {
    "infer": {"profile": "phi-3-mini-4k-instruct", "device": "gpu"},
    "review": {"profile": "smollm-1.7b-instruct", "device": "cpu"}
  }
}
```

---

## Implementation Sequencing

| Phase | What | Effort |
|-------|------|--------|
| **1** | Add `model_profile`, `device`, `n_gpu_layers` to `LaneConfig` | Small |
| **2** | `LaneModelConfig` + `build_lane_model_config()` capability mapping | Small |
| **3** | `LaneBackendManager` — per-lane backend creation/caching | Medium |
| **4** | Wire into agentic_runtime + LLMWorker | Medium |
| **5** | `LaneMetrics` + `WorkerPool.status()` enrichment | Small |
| **6** | Environment variable / llm.json config support | Small |
| **7** | Tests: multi-model smoke, CPU+GPU coexistence, metrics | Medium |

Phases 1-2 are independent. Phase 3 depends on Phase 1. Phase 4 depends on Phases 2-3. Phases 5-6 are independent.

---

## Verified API Signatures (from code audit)

> **Note:** Line numbers verified as of 2026-04-02. The modularization plan (Phase 1C) proposes splitting `router.py` into 5 files. After that split, `_LlamaCppBackend` moves to `llama_backend.py`, token counters move to `token_counter.py`, etc. **This plan must be sequenced AFTER modularization Phase 1C**, and these references updated to the post-split module structure.

| Method | File:Line | Notes |
|--------|-----------|-------|
| `LaneConfig` | worker_pool.py:72 | Currently has: name, max_workers, queue_size, requires_gpu |
| `DEFAULT_LANES` | worker_pool.py:426 | infer (gpu), review, record |
| `WorkerPool.submit()` | worker_pool.py:479 | Takes lane name, job_id, fn, priority, deps |
| `WorkerPool.status()` | worker_pool.py:519 | Returns queue_size, max_workers per lane |
| `LLMRouter.__init__()` | router.py:1102 | Takes LLMConfig, manages multi-backend cache |
| `LLMConfig.n_gpu_layers` | router.py:363 | int, default -1 (all on GPU) |
| `_LlamaCppBackend._ensure()` | router.py:960 | Sets n_gpu_layers on Llama() init (post-split: `llama_backend.py`) |
| `_PyTorchTransformersBackend._ensure()` | transformers_backend.py:224 | Device detection + model.to(device) |
| `load_llm_config()` | router.py:444 | Profile-based config loading |
| `BUILTIN_PROFILES` | router.py:81 | Includes smollm-1.7b-instruct (tiny) |

---

## Dependencies

**Modularization conflict:** The modularization plan (Phase 1C) splits `router.py` into 5 files:
- `_LlamaCppBackend` → `models/language/llama_backend.py`
- Token counters → `models/language/token_counter.py`
- Prompt formats → `models/language/prompt_formats.py`
- JSON parsing → `models/language/json_parser.py`
- `LLMConfig`, `LLMRouter`, profiles → `models/language/router.py` (reduced)

**Sequencing:** Implement this plan AFTER modularization Phase 1C completes. The `LaneBackendManager` (Phase 3) imports from `llama_backend.py` and `router.py` — these must be in their post-split locations. Update import paths at implementation time.

---

## Risks

1. **RAM exhaustion with dual models.** Two LLM models (even one quantized) can consume 4-8 GB combined. **Mitigation:** `LaneBackendManager` lazy-loads backends; add memory monitoring and auto-unload idle CPU backends.

2. **llama-cpp thread contention.** Two llama-cpp instances may compete for CPU threads. **Mitigation:** Set `n_threads` explicitly per backend to avoid over-subscription.

3. **Profile mismatch.** Environment variable overrides may specify non-existent profiles. **Mitigation:** Validate against `BUILTIN_PROFILES` at startup, fall back to defaults.
