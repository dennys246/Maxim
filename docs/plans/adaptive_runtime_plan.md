# Adaptive Runtime Plan

> **Status:** Not started. Pre-requisite: claw-code upgrade (items 1-10).
>
> **Includes:** Multi-LLM scaling (merged from `multi-llm-scaling.md`)
> as Phase 5c — per-lane model assignment with CPU+GPU coexistence.

Make Maxim self-metering: detect available hardware, connected devices, and data sources at startup and continuously, then adapt loop frequency, subsystem initialization, and behavior accordingly. The system should work as a full robotics agent with a Reachy, as a headless GPU agent without any robot, and everything in between — without code changes or explicit mode flags.

---

## What the Repo Does Now

### The Good: Strong Foundations

Maxim already has several pieces that support adaptive operation:

**Hardware abstraction layer** (`hardware/controller.py`, `hardware/registry.py`):
- `RobotController` ABC with `connect()`, `disconnect()`, `wake_up()`, `goto_sleep()`, streams
- `RobotRegistry` manages multiple controllers, tracks connection state lifecycle (CONNECTING → CONNECTED → DISCONNECTED → RECONNECTING → ERROR)
- Automatic reconnection with failure thresholds and cooldown
- Multi-robot coordination (`wake_all()`, `sleep_all()`, `move_all()`)

**Simulation controller** (`hardware/simulation/controller.py`):
- Full `RobotController` implementation with MockVideoStream and MockAudioStream
- `simulate_delays: bool` for realistic timing in tests
- Registered via `_robot_registry.register_controller_type("simulated", SimulatedController)`

**mDNS discovery** (`hardware/reachy/controller.py:74-95`):
- Resolves robot hostname → IP before SDK connection
- 5s timeout, non-blocking — failure falls back to SDK discovery
- 16 test cases in `test_mdns_discovery.py`

**Connection mode refactor** (commit `fc84bb1`):
- `connection_mode: str = "network"` replaces deprecated `localhost_only: bool`
- Supports `"auto"`, `"localhost_only"`, `"network"`

**GPU detection** (`agentic_runtime.py:34-66`):
- Detects GPU absence, falls back to smaller LLM (`smollm-1.7b-instruct`)
- Detects Blackwell GPU, avoids GStreamer incompatibility
- Sets `MAXIM_LLM_N_GPU_LAYERS=0` for CPU-only

**Headless display** (`data/camera/display.py`):
- `MAXIM_HEADLESS=1` env var disables OpenCV display
- Detects missing `DISPLAY`/`WAYLAND_DISPLAY` on Linux/WSL

**Agentic mode** (CLI `--mode agentic`):
- Creates `MaximAgent` directly without a `Maxim` instance
- The closest thing to headless today — but undocumented and limited

### The Problem: Crash Instead of Adapt

Despite the good abstractions, the system **crashes rather than adapts** when hardware isn't available:

```
CLI main()
  └─ Maxim.__init__()
       └─ _robot_registry.connect_robot(...)
            └─ if self._robot is None:
                 raise RuntimeError(...)  ← HARD CRASH, line 228 of selfy.py
```

**CORRECTION (verified against code):** Most subsystems downstream already have guards for `self._robot = None`. The actual crash sites are fewer than initially assumed:

| Subsystem | What Happens Without Robot | Guard Status |
|-----------|---------------------------|-------------|
| `selfy.py:228` | **RuntimeError** — hard crash in `__init__()` | **NO GUARD** — this is THE blocker |
| CaptureManager (`agentic_runtime.py:94`) | Passes `maxim=self` without checking capabilities | **NO GUARD** |
| `media_loop.py:137` | Calls `self.awaken()` unconditionally | **NO GUARD** (but awaken() has internal guards) |
| `awaken()` wake_up call | `self._robot.wake_up()` | **GUARDED** — line 643 checks `self._robot is not None` |
| `live()` get_video_stream | `self._robot.get_video_stream()` | **GUARDED** — line 367 checks `self._robot is None` |
| `live()` get_audio_stream | `self._robot.get_audio_stream()` | **GUARDED** — line 392 checks `self._robot is None` |
| Recording | `self._robot.start_recording()` | **GUARDED** — wrapped in try/except |
| DefaultNetwork | Needs `maxim._robot` for motor control | **GUARDED** — builder returns None if `maxim=None` |

**Only 3 sites need new guards** (selfy.py:228, agentic_runtime.py:94, media_loop.py:137). The rest already handle `self._robot = None` gracefully.

The loop frequency is hardcoded: `target_hz=30.0` is the default parameter on `run_agentic_loop()` (agent_loop.py:358) and `DefaultNetworkConfig.update_hz` (network.py:78). Callers may override these but the defaults assume motor control. A headless coding agent doesn't need 30Hz polling — it needs event-driven execution with ~2s cycles during LLM inference.

### What Partially Works

Some subsystems already handle missing hardware gracefully:

| Component | Handles None? | How |
|-----------|--------------|-----|
| `DefaultNetwork` builder | **Yes** | Returns `None` if `maxim=None` |
| CaptureManager | **Partially** | Falls back to JSONL vision events if not created |
| Tool registry | **Partially** | Registers no-op stubs for robot tools when `maxim=None` |
| LLM worker | **Yes** | Fully robot-agnostic |
| Memory subsystems | **Yes** | Hippocampus, NAc, SCN, EC, ATL are all robot-agnostic |
| Shutdown | **Yes** | Guards robot calls with `if self._robot is not None` |

The agentic loop itself (`run_agentic_loop()`) is robot-agnostic — it accepts `default_network=None` and skips the reactive layer. The problem is getting TO the loop without crashing.

---

## How This Plan Improves It

### Design Principle: Capability Detection, Not Mode Flags

An animal that loses a sense doesn't need a `--no-eyes` flag. Its nervous system detects the absence of visual input and adapts — other senses sharpen, behaviors change, but the organism keeps functioning. Maxim should work the same way.

Instead of `--headless` or `--no-robot`, detect capabilities at startup and continuously:

```python
@dataclass
class RuntimeCapabilities:
    """What's available right now. Can change during runtime."""
    has_robot: bool = False
    has_gpu: bool = False
    has_vision: bool = False        # Camera stream available
    has_audio: bool = False         # Microphone available
    has_motor: bool = False         # Motor control available
    has_display: bool = False       # Screen for cv2.imshow
    has_network: bool = False       # Internet connectivity
    robot_type: str | None = None   # "reachy_mini", "simulated", None
    gpu_type: str | None = None     # "cuda", "metal", "cpu"
    connected_devices: list[str] = field(default_factory=list)
```

This is populated once at startup, then updated when connections change (robot disconnects, GPU runs out of memory, network drops). Subsystems read capabilities instead of checking `self._robot is not None` scattered throughout the code.

### Phase 1: Graceful Degradation (Don't Crash)

**Goal:** Maxim starts and runs regardless of what's connected. Missing hardware means missing capabilities, not crashes.

#### 1a. Allow `self._robot = None` in selfy.py

```python
# selfy.py line 225-230 (CURRENT):
self._robot = _robot_registry.connect_robot(...)
if self._robot is None:
    raise RuntimeError(f"Failed to connect to robot: {effective_robot_id}")

# CHANGED TO:
self._robot = _robot_registry.connect_robot(...)
if self._robot is None:
    self.log.warning("No robot connected — running in headless mode")
    self._capabilities.has_robot = False
    self._capabilities.has_vision = False
    self._capabilities.has_audio = False
    self._capabilities.has_motor = False
```

#### 1b. Guard the 3 unguarded crash sites

Most robot call sites already have guards (wake_up, get_video_stream, get_audio_stream, start_recording are all guarded). Only 3 sites need new guards:

| File | Line | Call | Guard |
|------|------|------|-------|
| media_loop.py | 137 | `self.awaken(...)` | `if self._capabilities.has_robot:` (awaken has internal guards but shouldn't be called at all headless) |
| agentic_runtime.py | ~94 | `CaptureManager(maxim=self)` | `if self._capabilities.has_vision:` |
| live() | main loop | Frame/audio capture loop | Route to `_run_headless_loop()` instead (see 1d) |

#### 1c. Skip DefaultNetwork when no robot

Already partially done — `build_default_network()` returns `None` if `maxim=None`. But currently `maxim` is never None (it's `self`). Change to check capabilities:

```python
default_network = build_default_network(maxim=self, ...) if self._capabilities.has_motor else None
```

#### 1d. Headless media loop alternative

The current `live()` method is a tight media capture loop that requires frames and audio. For headless mode, replace with an event-driven loop:

```python
def _run_headless_loop(self, stop_event: threading.Event):
    """Headless alternative to live(). Event-driven, no media capture."""
    while not stop_event.is_set():
        # Process CLI input
        if self._cli_input_queue and not self._cli_input_queue.empty():
            self._handle_cli_input(self._cli_input_queue.get())
        # Process incoming comms (SMS, webhook)
        if self._gateway:
            self._gateway.poll()
        # Sleep until next event (not 30Hz polling)
        stop_event.wait(timeout=0.5)
```

Wire into `live()`:
```python
def live(self, ...):
    # ... existing file path setup (robot-agnostic) ...
    if self._capabilities.has_robot:
        self._run_media_loop(...)  # Existing 30Hz frame capture loop
    else:
        self._run_headless_loop(stop_event)  # Event-driven, no media
```

### Phase 2: Adaptive Loop Frequency

**Goal:** Loop frequency matches workload, not hardcoded assumptions.

#### 2a. Capability-aware target_hz

```python
def _compute_target_hz(capabilities: RuntimeCapabilities) -> float:
    """Adapt loop frequency to available hardware."""
    if capabilities.has_motor:
        return 30.0   # Motor control needs real-time updates
    if capabilities.has_vision:
        return 10.0   # Vision processing without motors
    return 2.0        # Headless: LLM inference cycles, event-driven
```

Pass to `run_agentic_loop(target_hz=_compute_target_hz(self._capabilities))`.

#### 2b. DefaultNetwork frequency adaptation

`DefaultNetworkConfig.update_hz` (currently hardcoded 30.0) should match motor availability:

```python
dn_config = DefaultNetworkConfig(
    update_hz=30.0 if capabilities.has_motor else 5.0,
    # Lower rate for vision-only monitoring (no motor commands)
)
```

#### 2c. LLM submit interval adaptation

Agent loop's `llm_submit_interval` (currently 0.5s) should adapt:
- With robot: 0.5s (real-time responsiveness)
- Headless: 0.1s (faster LLM cycles, no motor latency to hide)

### Phase 3: Conditional Subsystem Initialization

**Goal:** Only create subsystems that have the hardware they need.

#### 3a. Capability-gated initialization in agentic_runtime.py

Replace the current monolithic `_start_agentic_runtime()` with capability-aware initialization:

```python
def _start_agentic_runtime(self, ...):
    capabilities = self._capabilities

    # Always initialize (robot-agnostic):
    nac = NAc(...)
    memory_hub = MemoryHub(hippocampus=..., scn=..., nac=nac, ec=..., atl=..., ag=...)
    llm_worker = LLMWorker(...)
    tool_registry = build_tool_registry(
        maxim=self if capabilities.has_robot else None,  # No-op stubs if no robot
        ...
    )
    executor = Executor(tool_registry, ...)
    decision_engine = DecisionEngine(...)

    # Vision subsystems (only if camera available):
    if capabilities.has_vision:
        capture_manager = CaptureManager(maxim=self, ...)
        capture_manager.start()
    else:
        capture_manager = None
        # JSONL fallback already exists

    # Motor subsystems (only if motors available):
    if capabilities.has_motor:
        default_network = build_default_network(maxim=self, ...)
        default_network.start()
    else:
        default_network = None

    # Spawn agentic loop (always):
    target_hz = _compute_target_hz(capabilities)
    run_agentic_loop(
        ...,
        default_network=default_network,  # None in headless → skips reactive layer
        target_hz=target_hz,
    )
```

#### 3b. Tool registry capability awareness

`build_tool_registry()` already conditionally registers tools when `maxim=None`. Extend to use capabilities:

```python
def build_tool_registry(*, capabilities: RuntimeCapabilities, ...):
    registry = ToolRegistry()

    # Always register (robot-agnostic):
    registry.register(ReadFileTool(allowed_dirs=allowed_dirs))
    registry.register(WriteFileTool(allowed_dirs=allowed_dirs))
    registry.register(EditFileTool(allowed_dirs=allowed_dirs))    # From claw-code plan
    registry.register(CodeSearchTool(allowed_dirs=allowed_dirs))  # From claw-code plan
    registry.register(GlobTool(allowed_dirs=allowed_dirs))
    registry.register(RespondTool(...))

    # Robot tools (only if motor available):
    if capabilities.has_motor:
        registry.register(MoveTool(maxim))
        registry.register(TrackTargetTool(maxim))
        registry.register(FocusInterestsTool(maxim))

    # Vision tools (only if camera available):
    if capabilities.has_vision:
        registry.register(NoveltyTrackTool(maxim))

    # Network tools (only if internet available):
    if capabilities.has_network:
        registry.register(InternetSearchTool(...))
        registry.register(HttpFetchTool(...))

    # Coding tools (always — these are the headless workload):
    if os.environ.get("MAXIM_ALLOW_EXECUTE_FILE") == "1":
        registry.register(RunTestsTool())
        registry.register(BuildTool())
    registry.register(GitDiffTool())
    registry.register(GitCommitTool())

    return registry
```

### Phase 4: Runtime Capability Changes

**Goal:** Adapt when things connect or disconnect mid-session.

#### 4a. Create ConnectionState enum (doesn't exist yet)

**CORRECTION:** The plan originally assumed a `ConnectionState` enum exists. It does not. `connection.py` has `ReachyConnection` with `is_connected` property and `FailureTracker` dataclass, but no state machine enum. Create it:

```python
# In connection.py:
class ConnectionState(Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"
```

Add a callback mechanism to `ReachyConnection`:
```python
class ReachyConnection:
    def __init__(self, ...):
        self._state = ConnectionState.DISCONNECTED
        self._state_callbacks: list[Callable[[ConnectionState], None]] = []

    def add_state_callback(self, cb: Callable[[ConnectionState], None]) -> None:
        self._state_callbacks.append(cb)

    def _set_state(self, state: ConnectionState) -> None:
        old = self._state
        self._state = state
        if old != state:
            for cb in self._state_callbacks:
                cb(state)
```

#### 4b. Robot disconnect → graceful degradation

Hook into the new ConnectionState callbacks:

```python
# In agentic_runtime.py:
def _on_connection_state_changed(self, state: ConnectionState):
    if state == ConnectionState.DISCONNECTED:
        self._capabilities.has_robot = False
        self._capabilities.has_motor = False
        self._capabilities.has_vision = False  # If vision comes from robot camera
        self.log.warning("Robot disconnected — degrading to headless mode")
        # DefaultNetwork stops (it checks _running flag)
        # Agent loop continues without reactive layer
        # Tools that need robot return ToolOutput(success=False, error_kind=EXTERNAL_FAILURE)
        # Pain system records disconnection as cognitive pain

    elif state == ConnectionState.CONNECTED:
        self._capabilities.has_robot = True
        self._capabilities.has_motor = True
        self._capabilities.has_vision = True
        self.log.info("Robot reconnected — restoring full capabilities")
        # Restart DefaultNetwork
        # Re-register robot tools
```

#### 4b. Capability-aware Percept

The current `Percept` dataclass has vision-centric fields (`detections`, `velocity`, `primary_track_id`) that are empty noise in headless mode. Rather than changing `Percept` (which would break the vision pipeline), let the perception agent skip vision fields when no camera:

```python
# In perception_agent.py:
def build_percept(self, capabilities: RuntimeCapabilities) -> Percept:
    if capabilities.has_vision:
        # Full percept with detections, tracking, velocity
        return Percept(source="vision", detections=..., ...)
    else:
        # Headless percept: CLI input, file changes, comms only
        return Percept(source="cli", cli_input=..., content=..., ...)
```

### Phase 5: Integration with Existing Plans

#### 5a. MonitorRegistry (from claw-code upgrade plan)

The `MonitorRegistry` is already designed to work standalone — independent of `ThalamicGate` and `DefaultNetwork`. In headless mode:
- MonitorRegistry runs with its own callback list
- ToolDurationMonitor works (monitors Executor, which is robot-agnostic)
- No ThalamicGate wiring needed (gate doesn't exist without robot)

In robot mode:
- MonitorRegistry signals additionally flow through ThalamicGate
- Gate's adaptive thresholds manage escalation sensitivity

#### 5b. Cognitive Pain (#15 from claw-code upgrade plan)

The pain system is already robot-agnostic by design:
- `PainDetector.record_tool_error()` doesn't touch motors
- `ToolPainBridge` uses NAc for learning (robot-agnostic)
- `ToolHarmPredictor` queries NAc predictions (robot-agnostic)
- `FearAgent` gates tool calls (robot-agnostic)

Movement pain types (`EXCESSIVE_VELOCITY`, `DIRECTION_THRASHING`) simply don't fire in headless mode — no position updates means no movement pain signals. Tool pain types fire normally.

#### 5c. Multi-LLM Scaling (merged from multi-llm-scaling.md)

> **Merged from:** `docs/plans/multi-llm-scaling.md` (WorkerPool Phase 5).
> That plan assumed multi-GPU hardware. This section adapts it for
> real-world single-GPU or CPU-only setups where multiple small models
> run in parallel across CPU and GPU.

##### Goal

Run heterogeneous LLM backends within the same WorkerPool: a primary
model on GPU for main inference, and a smaller model on CPU for
background evaluation/review tasks. Capabilities detection (Phase 1)
determines what's available; lane configuration adapts automatically.

##### Capability-Driven Model Assignment

`RuntimeCapabilities` from Phase 1 drives model selection:

```python
def build_lane_model_config(caps: RuntimeCapabilities) -> dict[str, LaneModelConfig]:
    """Map WorkerPool lanes to model profiles based on hardware."""
    if caps.has_gpu:
        return {
            # Primary inference: quantized model on GPU
            "infer": LaneModelConfig(
                profile="phi-3-mini-4k-instruct",
                quantization="Q4_K_M",
                device="gpu",
                n_gpu_layers=-1,  # All layers on GPU
            ),
            # Background review: tiny model on CPU (doesn't compete for GPU)
            "review": LaneModelConfig(
                profile="smollm-1.7b-instruct",
                quantization="Q4_K_M",
                device="cpu",
                n_gpu_layers=0,
            ),
        }
    else:
        # No GPU: both lanes use smallest model on CPU
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

##### LaneConfig changes

Add `model_profile` and `device` to `LaneConfig`:

```python
@dataclass
class LaneConfig:
    name: str
    max_workers: int
    queue_size: int = 10
    requires_gpu: bool = False
    # NEW: per-lane model assignment (Phase 5c)
    model_profile: str | None = None   # LLM profile name
    device: str = "auto"               # "gpu", "cpu", "auto"
    n_gpu_layers: int = -1             # -1 = all on GPU, 0 = CPU only
```

##### Per-Lane LLM Backend

Each lane that needs LLM access gets its own backend instance:

```python
# In LLMWorker or a new LaneBackendManager:
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
        """Create a backend from lane config."""
        # Uses existing LLMRouter profile system
        from maxim.models.language.router import load_llm_config, LLMRouter
        llm_config = load_llm_config(profile_override=config.model_profile)
        # Override GPU layers for device placement
        llm_config = dataclasses.replace(
            llm_config,
            n_gpu_layers=config.n_gpu_layers,
        )
        return LLMRouter(cfg=llm_config)

    def unload_all(self) -> None:
        """Unload all backends (session end)."""
        with self._lock:
            for backend in self._backends.values():
                try:
                    backend.unload()
                except Exception:
                    pass
            self._backends.clear()
```

##### Thread-Local Device Isolation

For llama-cpp backends, GPU layer count is set at model load time
(not per-thread). CPU-only models naturally don't touch VRAM. For
PyTorch backends, device placement is set in `_ensure()` via the
`device` parameter. No `CUDA_VISIBLE_DEVICES` manipulation needed
for single-GPU setups — just `n_gpu_layers=0` for CPU lanes.

##### Configuration via Environment

```bash
# Override default lane model assignments
MAXIM_INFER_PROFILE=phi-3-mini-4k-instruct    # Main inference model
MAXIM_REVIEW_PROFILE=smollm-1.7b-instruct     # Background review model
MAXIM_INFER_DEVICE=gpu                          # gpu, cpu, auto
MAXIM_REVIEW_DEVICE=cpu                         # Force CPU for review
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

##### Per-Lane Metrics

Add lightweight counters to each lane for observability:

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

Exposed via `WorkerPool.status()`:
```python
def status(self) -> dict[str, Any]:
    info = {}
    for name, lane in self._lanes.items():
        info[name] = {
            "queue_size": lane._queue.qsize(),
            "max_workers": lane._config.max_workers,
            "model_profile": lane._config.model_profile,
            "device": lane._config.device,
            "metrics": lane._metrics.to_dict() if lane._metrics else None,
        }
    return info
```

##### Wiring in agentic_runtime.py

```python
# After RuntimeCapabilities detection (Phase 1):
lane_model_configs = build_lane_model_config(self._capabilities)

# Apply to WorkerPool lane configs
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

# Create per-lane backend manager
backend_manager = LaneBackendManager(lane_configs)
# LLMWorker uses backend_manager.get_backend(lane) for each job
```

##### Implementation Order (within Phase 5)

| Step | What | Effort |
|------|------|--------|
| 5c.1 | Add `model_profile`, `device`, `n_gpu_layers` to `LaneConfig` | Small |
| 5c.2 | `LaneModelConfig` + `build_lane_model_config()` capability mapping | Small |
| 5c.3 | `LaneBackendManager` — per-lane backend creation/caching | Medium |
| 5c.4 | Wire into agentic_runtime + LLMWorker | Medium |
| 5c.5 | `LaneMetrics` + `WorkerPool.status()` enrichment | Small |
| 5c.6 | Environment variable / llm.json config support | Small |
| 5c.7 | Tests: multi-model smoke, CPU+GPU coexistence, metrics | Medium |

##### Verified API Signatures (from code audit)

| Method | File:Line | Notes |
|--------|-----------|-------|
| `LaneConfig` | worker_pool.py:72 | Currently has: name, max_workers, queue_size, requires_gpu |
| `DEFAULT_LANES` | worker_pool.py:421 | infer (gpu), review, record |
| `WorkerPool.submit()` | worker_pool.py:479 | Takes lane name, job_id, fn, priority, deps |
| `WorkerPool.status()` | worker_pool.py:519 | Returns queue_size, max_workers per lane |
| `LLMRouter.__init__()` | router.py:1091 | Takes LLMConfig, manages multi-backend cache |
| `LLMConfig.n_gpu_layers` | router.py:370 | int, default -1 (all on GPU) |
| `_LlamaCppBackend._ensure()` | router.py:951 | Sets n_gpu_layers on Llama() init |
| `_PyTorchTransformersBackend._ensure()` | transformers_backend.py:224 | Device detection + model.to(device) |
| `load_llm_config()` | router.py:444 | Profile-based config loading |
| `BUILTIN_PROFILES` | router.py:82 | Includes smollm-1.7b-instruct (tiny) |

---

## Implementation Sequencing

| Phase | Item | Effort | Impact | Notes |
|-------|------|--------|--------|-------|
| **1** | `RuntimeCapabilities` dataclass + detection | Small | High | Foundation for everything |
| **1** | Remove RuntimeError on no robot (selfy.py:228) | Small | Critical | Unblocks headless mode |
| **1** | Guard 3 unguarded crash sites | Small | Critical | media_loop:137, agentic_runtime:94, live() routing |
| **1** | Headless media loop alternative | Medium | High | Event-driven loop for no-robot |
| **2** | Adaptive target_hz | Small | Medium | 30Hz→2Hz based on capabilities |
| **2** | Conditional DefaultNetwork init | Small | Medium | Skip if no motors |
| **2** | Conditional CaptureManager init | Small | Medium | Skip if no camera |
| **3** | Capability-aware tool registration | Small | Medium | Robot tools gated by capabilities |
| **3** | Capability-aware Percept building | Small | Medium | CLI/comms percepts in headless |
| **4** | Create ConnectionState enum + callbacks | Small | Medium | Doesn't exist yet — must create |
| **4** | Runtime connection state → capabilities | Medium | Medium | Live adapt on disconnect/reconnect |
| **5c.1** | `LaneConfig` gains model_profile, device, n_gpu_layers | Small | High | Foundation for per-lane models |
| **5c.2** | `build_lane_model_config()` capability-driven assignment | Small | High | Maps hardware → model profiles |
| **5c.3** | `LaneBackendManager` per-lane backend creation | Medium | High | Lazy-loaded, thread-safe |
| **5c.4** | Wire into agentic_runtime + LLMWorker | Medium | High | End-to-end multi-model pipeline |
| **5c.5** | `LaneMetrics` + enriched status() | Small | Medium | Per-lane observability |
| **5c.6** | Env var / llm.json config support | Small | Medium | MAXIM_INFER_PROFILE etc. |

### Phase 6: Smoke Tests

**Goal:** Validate the headless path end-to-end. One regression test per capability configuration prevents future changes from silently re-breaking headless mode.

**New file:** `tests/integration/test_headless_smoke.py`

These tests run WITHOUT a robot, WITHOUT a GPU requirement, and WITHOUT network. They validate that Maxim boots, runs tools, learns, and shuts down cleanly in every degraded configuration.

```python
@pytest.mark.integration
class TestHeadlessSmoke:
    """End-to-end smoke tests for headless operation."""

    def test_boot_no_robot(self):
        """Maxim.__init__() completes without a robot (no RuntimeError)."""
        maxim = Maxim(simulation=False, robot_name="nonexistent")
        assert maxim._robot is None
        assert maxim._capabilities.has_robot is False
        assert maxim._capabilities.has_motor is False
        maxim.shutdown()

    def test_capability_detection(self):
        """RuntimeCapabilities correctly reflects available hardware."""
        caps = detect_capabilities(robot=None, gpu_available=False)
        assert caps.has_robot is False
        assert caps.has_gpu is False
        assert caps.has_vision is False
        assert caps.has_motor is False
        # Display detection depends on environment
        # Network detection depends on environment

    def test_tool_execution_headless(self, tmp_path):
        """Core tools work without a robot: read, write, glob."""
        registry = build_tool_registry(
            maxim=None,  # No robot → no-op stubs for robot tools
            operational_mode="active",
        )
        # Write a file
        write_tool = registry.get("write_file")
        result = write_tool.run(path=str(tmp_path / "test.txt"), content="hello", overwrite=True)
        assert result.success

        # Read it back
        read_tool = registry.get("read_file")
        result = read_tool.run(path=str(tmp_path / "test.txt"))
        assert result.success
        assert "hello" in result.output

        # Glob for it
        glob_tool = registry.get("glob")
        result = glob_tool.run(pattern="*.txt", path=str(tmp_path))
        assert result.success

    def test_robot_tools_noop_headless(self):
        """Robot tools return graceful no-op results, not crashes."""
        registry = build_tool_registry(maxim=None, operational_mode="active")
        # These should be no-op stubs, not missing
        for tool_name in ["focus_interests", "move", "track_target"]:
            tool = registry.get(tool_name)
            result = tool.run()
            # No-op stubs should succeed or return informative error, not crash
            assert isinstance(result, ToolOutput)

    def test_memory_systems_no_robot(self):
        """Hippocampus, NAc, SCN initialize and work without robot."""
        from maxim.memory.hippocampus import Hippocampus, HippocampusConfig
        from maxim.decisions.nac import NAc

        hippo = Hippocampus(config=HippocampusConfig())
        nac = NAc()

        # Record and recall without robot
        nac.record_event(event_type="tool", event_signature="test_tool")
        nac.record_outcome(
            event_type="tool", event_id="test_tool",
            outcome_valence=Valence.POSITIVE,
        )
        links = nac.get_positive_outcomes("test_tool")
        assert len(links) > 0

    def test_decision_engine_no_robot(self):
        """DecisionEngine produces decisions without robot context."""
        from maxim.planning.decision_engine import DecisionEngine
        # Minimal wiring — no robot, no DefaultNetwork
        engine = DecisionEngine(planner=..., policy=..., constraints=[])
        # Should not crash, may produce no-action decisions
        # (full test would need a mock planner)

    def test_default_network_none_when_headless(self):
        """build_default_network returns None when maxim=None."""
        from maxim.runtime.bootstrap import build_default_network
        dn = build_default_network(maxim=None)
        assert dn is None

    def test_agentic_loop_runs_without_default_network(self):
        """Agent loop starts and stops cleanly with default_network=None."""
        stop = threading.Event()
        stop.set()  # Stop immediately after first iteration
        # Should not crash with default_network=None
        run_agentic_loop(
            agent=mock_agent,
            environment=mock_env,
            state=mock_state,
            memory=mock_memory,
            decision_engine=mock_engine,
            executor=mock_executor,
            default_network=None,  # Headless
            stop_event=stop,
            target_hz=2.0,
        )

    def test_adaptive_target_hz(self):
        """Loop frequency adapts to capabilities."""
        full = RuntimeCapabilities(has_robot=True, has_motor=True, has_vision=True)
        vision_only = RuntimeCapabilities(has_vision=True)
        headless = RuntimeCapabilities()

        assert _compute_target_hz(full) == 30.0
        assert _compute_target_hz(vision_only) == 10.0
        assert _compute_target_hz(headless) == 2.0

    def test_headless_shutdown_clean(self):
        """Shutdown completes without errors when no robot connected."""
        maxim = Maxim(simulation=False, robot_name="nonexistent")
        # Should not raise on any shutdown step
        maxim.shutdown()
        # Verify no threads leaked
        assert maxim._agentic_thread is None or not maxim._agentic_thread.is_alive()

    def test_cognitive_pain_no_robot(self):
        """Tool pain system works without movement pain."""
        from maxim.proprioception.pain import PainDetector, PainType

        detector = PainDetector()
        signals_received = []
        detector.add_pain_callback(lambda s: signals_received.append(s))

        # Tool error emits cognitive pain (no robot needed)
        detector.record_tool_error(
            tool_name="test_tool",
            error="connection refused",
            error_kind=ToolErrorKind.EXTERNAL_FAILURE,
        )
        assert len(signals_received) == 1
        assert signals_received[0].pain_type == PainType.TOOL_FAILURE

        # Movement pain does NOT fire (no position updates)
        # This is correct — no robot means no movement pain
```

**Additional test files:**

`tests/integration/test_capability_transitions.py` — Tests runtime capability changes (robot disconnect/reconnect):

```python
@pytest.mark.integration
class TestCapabilityTransitions:

    def test_disconnect_degrades_capabilities(self):
        """Robot disconnect updates capabilities and doesn't crash running systems."""
        caps = RuntimeCapabilities(has_robot=True, has_motor=True, has_vision=True)
        # Simulate disconnect
        caps.has_robot = False
        caps.has_motor = False
        caps.has_vision = False
        # Systems that read capabilities should adapt
        assert _compute_target_hz(caps) == 2.0

    def test_tool_failure_on_disconnect(self):
        """Robot tools return failure after disconnect, feeding pain system."""
        # After disconnect, robot tools should return ToolOutput(success=False)
        # not crash with AttributeError

    def test_monitor_registry_standalone(self):
        """MonitorRegistry runs without ThalamicGate (headless mode)."""
        registry = MonitorRegistry(poll_interval=0.1)
        signals = []
        registry.add_signal_callback(lambda s: signals.append(s))

        # Mock monitor that fires once
        mock_monitor = MockSignalMonitor(fire_once=True)
        registry.register(mock_monitor)
        registry.start()
        time.sleep(0.3)  # Let it poll a few times
        registry.stop()

        assert len(signals) >= 1
```

| Phase | Item | Effort | Notes |
|-------|------|--------|-------|
| **1** | `test_headless_smoke.py` | Medium | 11 tests covering boot, tools, memory, loop, shutdown, pain |
| **4** | `test_capability_transitions.py` | Small | 3 tests for disconnect/reconnect and MonitorRegistry standalone |

### Dependencies

- **Claw-code upgrade (items 1-10) should come FIRST** — establishes
  frozen dataclasses, factory methods, and code quality patterns that
  adaptive runtime code should follow from the start
- **Claw-code item 11 (coding tools) comes AFTER** adaptive Phase 1 —
  coding tools need headless mode working to run without a robot
- Phases 1-3 of this plan are independent of claw-code
- MonitorRegistry and Cognitive Pain from the claw-code plan work in both modes by design
- Phase 5c (multi-LLM scaling) uses `RuntimeCapabilities.gpu_type` from Phase 1

### What NOT to Change

- `RobotController` ABC — hardware abstraction is correct
- `RobotRegistry` — multi-robot management is correct
- `SimulatedController` — useful for testing, not for headless production
- `mDNS discovery` — connection infrastructure is correct
- `DefaultNetwork` behaviors — they correctly assume motors, just shouldn't be initialized without them
- `connection_mode` values — `"network"`, `"auto"`, `"localhost_only"` are for robot discovery, not headless mode

### Cross-References with Other Plans

| Plan | Relationship | Conflicts? |
|------|-------------|-----------|
| [claw-code-upgrade-plan.md](claw-code-upgrade-plan.md) | Coding tools, cognitive pain, session persistence — all robot-agnostic by design | None — but adaptive Phase 1 should precede claw-code Phase 2 |
| [safe_freezing_plan.md](safe_freezing_plan.md) | Frozen configs — orthogonal to runtime adaptation | None |
| ~~multi-llm-scaling.md~~ | **MERGED** into this plan as Phase 5c | N/A — delete original |
| [provenance_plan.md](provenance_plan.md) | Decision tracing — **IMPLEMENTED**, orthogonal | None |
| ~~bio-skill-integration.md~~ (ATL A7) | Bio-skill integration — **IMPLEMENTED**, bio systems are robot-agnostic | None |

### Review Corrections

| Claim | Original | Correction |
|-------|----------|------------|
| Crash sites | "6 sites need guards" | **3 sites need guards** — wake_up, get_video_stream, get_audio_stream, start_recording already have None checks |
| ConnectionState enum | "Already tracks state (CONNECTED → DISCONNECTED → RECONNECTING)" | **Doesn't exist** — connection.py has `ReachyConnection.is_connected` property and `FailureTracker`, but no state machine enum. Must create in Phase 4. |
| target_hz | "10.0 for agentic loop" | **Default is 30.0** (agent_loop.py:358). Callers may override. DefaultNetworkConfig.update_hz is also 30.0. |
| Agentic mode | "Bypasses Maxim.__init__()" | **Correct** — `--mode agentic` (cli.py:673-907) creates MaximAgent directly. But `--mode exploration` still creates Maxim. |
