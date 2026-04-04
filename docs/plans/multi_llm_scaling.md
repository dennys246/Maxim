# Multi-LLM Scaling Plan

> **Status:** Not started — but all prerequisites are complete. Depends on RuntimeCapabilities (implemented), WorkerPool lane system (implemented), shared LLMRouter pattern (implemented via Agentic Loop Modularization), and **Router Modularization (done — router.py split into config/types/token_counter/prompt_formats/json_parser modules)**. Ready to begin Phase 1.
>
> **Scope:** Local multi-model inference, remote model serving via home server + Cloudflare tunnel, dynamic backend spawning, and peer-to-peer inference mesh across Maxim instances.

Turn any machine running Maxim into a node in a distributed inference mesh. Each node contributes whatever compute it has — a 5080 GPU at home, a laptop CPU on the go, a Reachy with an edge GPU — and any node can route inference requests to the best available backend. Models stay loaded and warm. Access from anywhere via Cloudflare tunnel.

---

## Motivation

Currently all WorkerPool lanes share a single local LLM backend. This means:
- GPU inference blocks CPU-appropriate review tasks
- The Reachy's limited compute can't offload to a more powerful machine
- No way to keep models warm on a home server for instant inference from anywhere
- Each Maxim instance is an island — no shared compute

**Target setup:**

```
┌─────────────────────────┐     Cloudflare Tunnel      ┌──────────────────┐
│   Home PC (RTX 5080)    │◄────────────────────────────│  Laptop (CPU)    │
│                         │     maxim-llm.example.com   │  Maxim instance  │
│  vLLM / llama.cpp srv   │                             │  uses remote     │
│  ├─ 24B Q4 (GPU, infer) │◄──── LAN mDNS ────────────│  backend         │
│  └─ 7B Q4 (CPU, review) │                             └──────────────────┘
│                         │
│  Maxim instance         │◄──── LAN mDNS ────────────┌──────────────────┐
│  (also runs agentic)    │                             │  Reachy (edge)   │
└─────────────────────────┘                             │  Maxim instance  │
                                                        │  offloads to     │
                                                        │  home server     │
                                                        └──────────────────┘
```

---

## Architecture: Four Layers

### Layer 1: Local Multi-Model (per-machine)

Run multiple LLM backends on a single machine, each assigned to a WorkerPool lane.

### Layer 2: Remote Model Server

A dedicated model-serving process (vLLM, llama.cpp server, or Ollama) on a powerful machine, exposed via an OpenAI-compatible API.

### Layer 3: Cloudflare Tunnel

Zero-config remote access. `cloudflared` daemon on the model server, points a subdomain at the API. Works from any network without VPN or port forwarding.

### Layer 4: Peer Discovery + Inference Mesh

Maxim instances discover each other via mDNS on LAN and advertise their available models. Any instance can route inference to the best available backend: local → peer → remote.

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

Per-lane LLM backend creation with lazy loading and thread-safe caching. Supports both local and remote backends.

```python
class LaneBackendManager:
    """Manages per-lane LLM backends — local, remote, or peer.

    Each lane with a model_profile gets a dedicated backend instance.
    Backends are lazy-loaded on first use. Supports three backend types:
    - local: llama-cpp or transformers, loaded in-process
    - remote: HTTP client to a model server (OpenAI-compatible API)
    - peer: HTTP client to another Maxim instance discovered via mDNS
    """

    def __init__(self, lane_configs: dict[str, LaneConfig]) -> None:
        self._backends: dict[str, Any] = {}
        self._configs = lane_configs
        self._lock = threading.Lock()
        self._peer_registry: PeerRegistry | None = None  # Set by Phase 7

    def get_backend(self, lane: str) -> Any | None:
        """Get or create the backend for a lane.

        Resolution order: local config → peer → remote fallback.
        """
        config = self._configs.get(lane)
        if not config or not config.model_profile:
            return None
        with self._lock:
            if lane not in self._backends:
                self._backends[lane] = self._create_backend(config)
            return self._backends[lane]

    def _create_backend(self, config: LaneConfig) -> Any:
        if config.remote_url:
            return self._create_remote_backend(config)
        return self._create_local_backend(config)

    def _create_local_backend(self, config: LaneConfig) -> Any:
        from maxim.models.language.router import load_llm_config, LLMRouter
        import dataclasses
        llm_config = load_llm_config(profile_override=config.model_profile)
        llm_config = dataclasses.replace(llm_config, n_gpu_layers=config.n_gpu_layers)
        return LLMRouter(cfg=llm_config)

    def _create_remote_backend(self, config: LaneConfig) -> Any:
        # _OpenAIBackend already exists in models/language/openai_backend.py
        from maxim.models.language.openai_backend import _OpenAIBackend
        return _OpenAIBackend(
            base_url=config.remote_url,
            model=config.model_profile,
            api_key=config.remote_api_key or "not-needed",  # Local servers don't need keys
        )

    def unload_all(self) -> None:
        with self._lock:
            for backend in self._backends.values():
                try:
                    if hasattr(backend, 'unload'):
                        backend.unload()
                except Exception:
                    pass
            self._backends.clear()
```

### Phase 4: Remote Model Server Setup

Run a model server on the home PC (RTX 5080, 16GB VRAM) that serves models via an OpenAI-compatible API.

#### 4a. Server options (pick one)

| Server | Pros | Cons |
|--------|------|------|
| **llama-cpp-python server** | Already a dependency, minimal setup, `--n-gpu-layers -1` | Single model per process, no batching |
| **Ollama** | Multi-model, auto-download, simple API | Extra dependency, less control |
| **vLLM** | Production-grade, continuous batching, multi-model | Heavy, needs CUDA, overkill for single-user |

**Recommendation:** Start with `llama-cpp-python` server (already installed). Upgrade to Ollama or vLLM when you need multi-model concurrency.

#### 4b. Server launch

```bash
# On home PC — serve a 24B model on GPU
python -m llama_cpp.server \
    --model ~/models/qwen2.5-coder-14b-instruct-Q4_K_M.gguf \
    --n_gpu_layers -1 \
    --host 0.0.0.0 \
    --port 8000 \
    --chat_format chatml

# Optional: second server on CPU for review lane
python -m llama_cpp.server \
    --model ~/models/smollm-1.7b-instruct-Q4_K_M.gguf \
    --n_gpu_layers 0 \
    --host 0.0.0.0 \
    --port 8001 \
    --chat_format chatml
```

#### 4c. LaneConfig gains remote fields

```python
@dataclass
class LaneConfig:
    name: str
    max_workers: int
    queue_size: int = 10
    requires_gpu: bool = False
    # Local model assignment
    model_profile: str | None = None
    device: str = "auto"
    n_gpu_layers: int = -1
    # Remote model server
    remote_url: str | None = None       # e.g., "http://homepc:8000/v1"
    remote_api_key: str | None = None   # For authenticated endpoints
    remote_model: str | None = None     # Model name on the remote server
```

### Phase 5: Cloudflare Tunnel

Zero-config remote access to the model server from any network.

#### 5a. Tunnel setup (one-time)

```bash
# Install cloudflared
brew install cloudflare/cloudflare/cloudflared  # macOS
# or: sudo apt install cloudflared              # Linux

# Authenticate
cloudflared tunnel login

# Create tunnel
cloudflared tunnel create maxim-llm

# Route DNS
cloudflared tunnel route dns maxim-llm maxim-llm.yourdomain.com

# Config file (~/.cloudflared/config.yml)
tunnel: <tunnel-id>
credentials-file: ~/.cloudflared/<tunnel-id>.json
ingress:
  - hostname: maxim-llm.yourdomain.com
    service: http://localhost:8000
  - service: http_status:404
```

#### 5b. Run as system service

```bash
# Install as system service (starts on boot)
sudo cloudflared service install

# Or run manually
cloudflared tunnel run maxim-llm
```

#### 5c. Maxim configuration

```bash
# Environment variable
MAXIM_REMOTE_LLM_URL=https://maxim-llm.yourdomain.com/v1
MAXIM_REMOTE_LLM_KEY=optional-bearer-token
```

Or in `llm.json`:
```json
{
  "lane_models": {
    "infer": {
      "remote_url": "https://maxim-llm.yourdomain.com/v1",
      "model": "qwen2.5-coder-14b-instruct"
    },
    "review": {
      "profile": "smollm-1.7b-instruct",
      "device": "cpu"
    }
  }
}
```

#### 5d. Latency considerations

| Path | Latency | Use for |
|------|---------|---------|
| Local llama-cpp | ~50-200ms | Real-time robot control (30Hz) |
| LAN remote | ~5-20ms + inference | Planning, reflection, goal proposal |
| Cloudflare tunnel | ~20-50ms + inference | Same as LAN but from any network |

The AdaptivePlanner's `context_budget_ms` (currently 50ms) applies to memory gathering, not LLM inference. LLM calls are async via the WorkerPool. Remote latency is invisible to the agentic loop — it just means results arrive one cycle later.

### Phase 6: Dynamic Local Backend Spawning

Automatically detect available hardware at startup and spawn model server processes for models that should be locally available.

#### 6a. Backend spawner

```python
class LocalBackendSpawner:
    """Spawns llama-cpp-server processes based on available hardware.

    Detects GPU VRAM and system RAM, selects appropriate models,
    and spawns server processes on available ports. Each spawned
    server is registered as a local remote endpoint.
    """

    def __init__(self, model_dir: str = "~/models") -> None:
        self._model_dir = os.path.expanduser(model_dir)
        self._processes: dict[int, subprocess.Popen] = {}  # port → process
        self._next_port = 8100  # Dynamic ports start here

    def spawn_for_capabilities(self, caps: RuntimeCapabilities) -> list[dict]:
        """Spawn model servers based on detected hardware.

        Returns list of {"port": int, "model": str, "device": str}
        for each spawned server.
        """
        spawned = []

        if caps.has_gpu:
            vram_gb = self._detect_vram_gb()
            # Pick the largest model that fits
            gpu_model = self._select_gpu_model(vram_gb)
            if gpu_model:
                port = self._spawn_server(gpu_model, n_gpu_layers=-1)
                if port:
                    spawned.append({"port": port, "model": gpu_model, "device": "gpu"})

        # Always try to spawn a small CPU model for background tasks
        ram_gb = self._detect_ram_gb()
        if ram_gb > 4:  # Need at least 4GB free for a small model
            cpu_model = self._select_cpu_model()
            if cpu_model:
                port = self._spawn_server(cpu_model, n_gpu_layers=0)
                if port:
                    spawned.append({"port": port, "model": cpu_model, "device": "cpu"})

        return spawned

    def _spawn_server(self, model_path: str, n_gpu_layers: int) -> int | None:
        """Spawn a llama-cpp-server on the next available port."""
        port = self._next_port
        self._next_port += 1
        try:
            proc = subprocess.Popen(
                [
                    sys.executable, "-m", "llama_cpp.server",
                    "--model", model_path,
                    "--n_gpu_layers", str(n_gpu_layers),
                    "--host", "127.0.0.1",
                    "--port", str(port),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            # Wait briefly and check it started
            time.sleep(2.0)
            if proc.poll() is not None:
                return None  # Failed to start
            self._processes[port] = proc
            return port
        except Exception:
            return None

    def shutdown_all(self) -> None:
        for port, proc in self._processes.items():
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                proc.kill()
        self._processes.clear()

    @staticmethod
    def _detect_vram_gb() -> float:
        """Detect available GPU VRAM in GB."""
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.get_device_properties(0).total_mem / (1024**3)
        except ImportError:
            pass
        return 0.0

    @staticmethod
    def _detect_ram_gb() -> float:
        """Detect available system RAM in GB."""
        try:
            import psutil
            return psutil.virtual_memory().available / (1024**3)
        except ImportError:
            # Fallback: /proc/meminfo on Linux
            try:
                with open("/proc/meminfo") as f:
                    for line in f:
                        if line.startswith("MemAvailable:"):
                            return int(line.split()[1]) / (1024**2)
            except Exception:
                pass
        return 8.0  # Conservative default
```

#### 6b. Model selection strategy

| Available VRAM | GPU Model | CPU Model |
|----------------|-----------|-----------|
| 16GB+ (5080) | 24B Q4_K_M (~14GB) | 3B Q4 (~2GB RAM) |
| 8-16GB | 13B Q4_K_M (~8GB) | 3B Q4 |
| 4-8GB | 7B Q4_K_M (~4.5GB) | — (not enough RAM) |
| 0 (CPU only) | — | 7B Q4 (~4.5GB RAM) |

#### 6c. Integration with LaneBackendManager

```python
# In agentic_runtime.py startup:
spawner = LocalBackendSpawner(model_dir=data_dir + "/models")
spawned = spawner.spawn_for_capabilities(self._capabilities)

# Register spawned servers as lane backends
for info in spawned:
    url = f"http://127.0.0.1:{info['port']}/v1"
    if info["device"] == "gpu":
        lane_configs["infer"].remote_url = url
    else:
        lane_configs["review"].remote_url = url
```

### Phase 7: Peer Discovery + Inference Mesh

Maxim instances discover each other on LAN and share compute. Any instance can offload inference to a peer with better hardware.

#### 7a. Peer advertisement via mDNS

Each Maxim instance advertises its available models via mDNS (same mechanism used for robot discovery):

```python
class PeerRegistry:
    """Discover and track peer Maxim instances on the LAN.

    Uses mDNS (zeroconf) to advertise available models and discover
    peers. Each peer exposes an OpenAI-compatible API on a local port.

    Service type: _maxim-llm._tcp.local.
    TXT records: models=<comma-separated>, vram=<GB>, device=<gpu|cpu>
    """

    SERVICE_TYPE = "_maxim-llm._tcp.local."

    def __init__(self, node_id: str, port: int) -> None:
        self._node_id = node_id
        self._port = port
        self._peers: dict[str, PeerInfo] = {}  # node_id → info
        self._zeroconf = None
        self._browser = None
        self._lock = threading.Lock()

    def start(self, available_models: list[str], device: str, vram_gb: float) -> None:
        """Start advertising this node and browsing for peers."""
        from zeroconf import Zeroconf, ServiceBrowser, ServiceInfo
        import socket

        self._zeroconf = Zeroconf()

        # Advertise this node
        info = ServiceInfo(
            self.SERVICE_TYPE,
            f"maxim-{self._node_id}.{self.SERVICE_TYPE}",
            addresses=[socket.inet_aton(self._get_local_ip())],
            port=self._port,
            properties={
                b"models": ",".join(available_models).encode(),
                b"device": device.encode(),
                b"vram": str(vram_gb).encode(),
                b"node_id": self._node_id.encode(),
            },
        )
        self._zeroconf.register_service(info)

        # Browse for peers
        self._browser = ServiceBrowser(
            self._zeroconf,
            self.SERVICE_TYPE,
            handlers=[self._on_service_change],
        )

    def get_peer_for_model(self, model_name: str) -> PeerInfo | None:
        """Find the best peer that serves a given model.

        Prefers: highest VRAM → GPU over CPU → lowest latency.
        """
        with self._lock:
            candidates = [
                p for p in self._peers.values()
                if model_name in p.models and p.is_alive
            ]
        if not candidates:
            return None
        # Sort: GPU first, then by VRAM descending
        candidates.sort(key=lambda p: (p.device != "gpu", -p.vram_gb))
        return candidates[0]

    def stop(self) -> None:
        if self._zeroconf:
            self._zeroconf.unregister_all_services()
            self._zeroconf.close()


@dataclass
class PeerInfo:
    node_id: str
    host: str
    port: int
    models: list[str]
    device: str
    vram_gb: float
    last_seen: float = 0.0

    @property
    def is_alive(self) -> bool:
        return (time.time() - self.last_seen) < 30.0  # 30s heartbeat timeout

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"
```

#### 7b. Inference routing with fallback chain

```python
# Resolution order for each inference request:
#
# 1. Local backend (if available for this lane)
#    └─ Fastest, no network hop
#
# 2. LAN peer (discovered via mDNS)
#    └─ ~5-20ms latency, may have better GPU
#
# 3. Remote server (Cloudflare tunnel)
#    └─ ~20-50ms + inference, always available
#
# 4. Fallback: smallest local CPU model
#    └─ Slow but always works

class InferenceRouter:
    """Routes inference requests to the best available backend.

    Combines local, peer, and remote backends with automatic failover.
    """

    def __init__(
        self,
        lane_manager: LaneBackendManager,
        peer_registry: PeerRegistry | None = None,
        remote_url: str | None = None,
    ) -> None:
        self._lane_manager = lane_manager
        self._peer_registry = peer_registry
        self._remote_url = remote_url

    def get_backend(self, lane: str, model_hint: str | None = None) -> Any:
        """Get the best available backend for a lane.

        Tries: local → peer → remote → fallback.
        """
        # 1. Local
        backend = self._lane_manager.get_backend(lane)
        if backend is not None:
            return backend

        # 2. Peer (LAN)
        if self._peer_registry and model_hint:
            peer = self._peer_registry.get_peer_for_model(model_hint)
            if peer:
                from maxim.models.language.openai_backend import _OpenAIBackend
                return _OpenAIBackend(
                    base_url=peer.base_url,
                    model=model_hint,
                    api_key="not-needed",
                )

        # 3. Remote (Cloudflare tunnel)
        if self._remote_url:
            from maxim.models.language.openai_backend import _OpenAIBackend
            return _OpenAIBackend(
                base_url=self._remote_url,
                model=model_hint or "default",
                api_key=os.environ.get("MAXIM_REMOTE_LLM_KEY", "not-needed"),
            )

        # 4. Fallback: None (caller handles gracefully)
        return None
```

#### 7c. Cluster status view

```python
def cluster_status(peer_registry: PeerRegistry, lane_manager: LaneBackendManager) -> dict:
    """Get a snapshot of the full inference mesh."""
    return {
        "local": {
            lane: {
                "model": config.model_profile,
                "device": config.device,
                "status": "loaded" if lane in lane_manager._backends else "idle",
            }
            for lane, config in lane_manager._configs.items()
        },
        "peers": {
            peer.node_id: {
                "host": peer.host,
                "models": peer.models,
                "device": peer.device,
                "vram_gb": peer.vram_gb,
                "alive": peer.is_alive,
            }
            for peer in peer_registry._peers.values()
        },
    }
```

### Phase 8: Per-Lane Metrics

```python
@dataclass
class LaneMetrics:
    """Per-lane performance counters."""
    jobs_completed: int = 0
    jobs_dropped: int = 0
    total_latency_ms: float = 0.0
    peak_queue_depth: int = 0
    remote_calls: int = 0      # Calls routed to remote/peer
    local_calls: int = 0       # Calls served locally
    failover_count: int = 0    # Times primary backend was unavailable

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / max(self.jobs_completed, 1)

    @property
    def remote_ratio(self) -> float:
        total = self.remote_calls + self.local_calls
        return self.remote_calls / max(total, 1)
```

### Phase 9: Environment Variable / Config Support

```bash
# Local model override
MAXIM_INFER_PROFILE=phi-3-mini-4k-instruct
MAXIM_REVIEW_PROFILE=smollm-1.7b-instruct
MAXIM_INFER_DEVICE=gpu
MAXIM_REVIEW_DEVICE=cpu

# Remote model server
MAXIM_REMOTE_LLM_URL=https://maxim-llm.yourdomain.com/v1
MAXIM_REMOTE_LLM_KEY=optional-bearer-token

# Peer mesh
MAXIM_PEER_ENABLED=1              # Enable mDNS peer discovery
MAXIM_PEER_PORT=8100              # Port for this node's model API
MAXIM_PEER_ADVERTISE_MODELS=1     # Advertise local models to peers

# Dynamic spawning
MAXIM_AUTO_SPAWN_MODELS=1         # Auto-spawn model servers at startup
MAXIM_MODEL_DIR=~/models          # Where to find GGUF files
```

Or via `llm.json`:
```json
{
  "lane_models": {
    "infer": {
      "remote_url": "https://maxim-llm.yourdomain.com/v1",
      "model": "qwen2.5-coder-14b-instruct"
    },
    "review": {
      "profile": "smollm-1.7b-instruct",
      "device": "cpu"
    }
  },
  "peer": {
    "enabled": true,
    "port": 8100,
    "advertise": true
  },
  "auto_spawn": {
    "enabled": true,
    "model_dir": "~/models"
  }
}
```

---

## Implementation Sequencing

| Phase | What | Effort | Dependencies |
|-------|------|--------|-------------|
| **1** | `LaneConfig` gains model fields | Small | None |
| **2** | `LaneModelConfig` + capability-driven assignment | Small | Phase 1 |
| **3** | `LaneBackendManager` (local + remote backends) | Medium | Phase 1 |
| **4** | Remote model server setup (home PC) | Small (config) | None (external) |
| **5** | Cloudflare tunnel setup | Small (config) | Phase 4 |
| **6** | `LocalBackendSpawner` (dynamic model servers) | Medium | Phase 3 |
| **7** | `PeerRegistry` + `InferenceRouter` (mesh) | Large | Phases 3, 6 |
| **8** | `LaneMetrics` enrichment | Small | Phase 3 |
| **9** | Environment variable / config support | Small | All phases |

**Recommended order:**
1. Phases 1-3 (local multi-model) — gets dual-model working on one machine
2. Phases 4-5 (remote + tunnel) — offload from laptop/Reachy to home PC
3. Phase 6 (auto-spawn) — convenience, no more manual server launches
4. Phases 7-9 (mesh + metrics + config) — multi-node cluster

Phases 4-5 are external configuration, not code changes. They can be done in parallel with Phase 3.

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
| `_OpenAIBackend` | openai_backend.py:56 | HTTP client, OpenAI-compatible API — **reuse for remote** |
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

**New dependency for Phase 7:** `zeroconf` package for mDNS peer discovery. Already commonly used in Python networking. Add as optional: `pip install "maxim[mesh]"`.

---

## Risks

1. **RAM exhaustion with dual models.** Two LLM models (even one quantized) can consume 4-8 GB combined. **Mitigation:** `LaneBackendManager` lazy-loads backends; `LocalBackendSpawner` checks available RAM/VRAM before spawning.

2. **llama-cpp thread contention.** Two llama-cpp instances may compete for CPU threads. **Mitigation:** Set `n_threads` explicitly per backend to avoid over-subscription.

3. **Profile mismatch.** Environment variable overrides may specify non-existent profiles. **Mitigation:** Validate against `BUILTIN_PROFILES` at startup, fall back to defaults.

4. **Cloudflare tunnel latency spikes.** Tunnel adds variable latency (20-200ms depending on Cloudflare edge proximity). **Mitigation:** Keep time-critical motor control on local backends; only offload planning/reflection to remote. The agentic loop's async WorkerPool hides inference latency.

5. **Peer discovery noise.** On busy LANs, stale mDNS entries may point to dead nodes. **Mitigation:** 30s heartbeat timeout on `PeerInfo.is_alive`; health check before routing first request to a new peer.

6. **Security: open model API on LAN.** Spawned llama-cpp servers listen on 127.0.0.1 by default (local only). Peer mesh requires binding to 0.0.0.0 for LAN access. **Mitigation:** Bind to LAN interface only (not public), optional bearer token auth, Cloudflare tunnel handles remote auth via zero-trust policies.

7. **Model file management.** `LocalBackendSpawner` assumes GGUF files exist in `model_dir`. **Mitigation:** Existing `python -m maxim.models.download` already handles model downloads; extend it to download models needed by lane configs.

---

## RTX 5080 Capacity Planning

| Configuration | VRAM Usage | RAM Usage | Concurrent |
|--------------|-----------|-----------|------------|
| 1x 24B Q4_K_M (GPU) + 1x 3B Q4 (CPU) | ~14 GB | ~2 GB | Yes |
| 1x 14B Q4_K_M (GPU) + 1x 7B Q4 (CPU) | ~8 GB | ~4.5 GB | Yes |
| 2x 7B Q4_K_M (GPU) | ~9 GB | — | Yes |
| 1x 24B Q4_K_M (GPU) only | ~14 GB | — | Single model, shared across lanes |

**Recommended for home server:** 1x Qwen2.5-Coder-14B Q4_K_M on GPU (~8GB) + 1x SmolLM-1.7B on CPU (~1.5GB). Leaves headroom for VRAM spikes and system overhead. Upgrade to 24B when you confirm stability.
