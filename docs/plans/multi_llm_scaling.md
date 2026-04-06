# Multi-LLM Scaling Plan

> **Status:** Phases 1–6 live (local multi-model, gates, remote lanes, tunnel, auto-spawn, leader mode). Phase 8 (LaneMetrics) is next. Phase 7 restructured into sub-phases 7a-7e covering leader-proxy + shared queue + mDNS + routing + multi-front input; see [peer_leader_debug_plan.md](peer_leader_debug_plan.md) for the architectural analysis that motivated the restructure.
>
> **Scope:** Local multi-model inference, remote model serving via home server + Cloudflare tunnel, dynamic backend spawning, peer-to-peer inference mesh, and multi-frontend input for shared-consciousness deployments.

Turn any machine running Maxim into a node in a distributed inference mesh. Each node contributes whatever compute it has — a 5080 GPU at home, a laptop CPU on the go, a Reachy with an edge GPU — and any node can route inference requests to the best available backend. Models stay loaded and warm. Access from anywhere via Cloudflare tunnel. A leader machine can expose its agent loop to multiple input frontends (local CLI, remote CLI, SMS) so a whole household talks to one Maxim mind.

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

## Observed Performance & Gotchas

Measured against a **self-hosted llama-cpp-server (mistral-7b-instruct-v0.2 Q4_K_M, `--n_ctx 8192`, CUDA, RTX 5080)** via HTTP loopback, Maxim process separate. Use these as baseline numbers when sizing mesh topologies or budgeting token counts; measure again on each new hardware / model combo.

### Latency baseline (2026-04-04)

| Scenario | Latency |
|---|---|
| Short completion (≤10 output tokens) | **~44 ms mean** (42–46 ms range over 10 sequential calls) |
| 4 parallel short completions (wall-time) | **~0.72 s** for 4 requests — server batches well, ~4× faster than sequential |
| Long-context completion (~4k prompt tokens) | ~1 s |
| Direct `_OpenAIBackend.complete_with_usage()` one-shot | ~8.7 s (cold start; includes KV warmup) |

**Implications for mesh routing:**
- Peer hops over LAN add ~5–20 ms on top of these numbers. At 44 ms baseline, a LAN hop is ~11–45% overhead — acceptable for planning/reflection lanes, tight for real-time motor lanes.
- Cloudflare tunnel hops add 20–50 ms + inference time (~65–94 ms total for a short call). Still async-friendly; only a concern for <100 ms response budgets.
- Concurrency scales well: 2–4 Maxim instances sharing one server is realistic without per-request latency degradation.

### Context window gotcha

**Mistral tokens are denser than English-word count suggests.** A prompt that "looks like 5000 tokens" by word count may actually be 12,500. Measured via the server's own tokenizer:

| Text | Word count | Token count | Ratio |
|---|---|---|---|
| "The word banana. " × 800 | ~2400 words | **4055 tokens** | 1.69× |
| "The word banana. " × 2500 | ~7500 words | **12,552 tokens** | 1.67× |

**Don't estimate prompt tokens from word counts** — use the actual tokenizer. The sim orchestrator's cooperative persona prompt is already ~2900 tokens on a simple goal; adversarial + history-heavy sims will push much higher. **`MAXIM_AUTO_SPAWN_N_CTX=8192` is acceptable for short sims but tight for long ones.** Mistral-7B-v0.2 trained at 32768 — raise if VRAM permits (≈doubling ctx roughly doubles KV-cache VRAM, ≈300 MB per doubling for 7B Q4).

### Resilience behavior (verified)

- **Auto-discovery**: a fresh `build_primary_router()` call correctly detects an existing server on the auto-spawn port and reuses it instead of spawning a duplicate.
- **Stale-URL recovery**: if `MAXIM_LANE_*_REMOTE_URL` points at a dead server, the private-IP URL is probed at startup, dropped with a warning, and auto-spawn takes over. Public/cloud URLs are trusted (not probed — would add latency + cost).
- **Signal isolation**: spawned server runs in its own process group (`start_new_session=True`), so Ctrl+C on the Maxim CLI doesn't kill it mid-shutdown. Cleanup paths (sim report LLM roundup) can make final inference calls before atexit terminates the subprocess.

### Per-process VRAM accounting (5080, 16 GB VRAM)

| Setup | VRAM | Notes |
|---|---|---|
| 1× mistral-7b Q4 (server) + client process | ~5 GB | Leaves 11 GB headroom |
| 2× mistral-7b Q4 (one user ran a manual server + auto-spawn) | ~9 GB | Wasteful — auto-discovery now prevents this |
| 1× llama-2-13b Q4 on server + client process | ~8 GB | Headroom for 16 GB card |
| Peer mesh: 2 Maxim clients sharing 1 server | ~5 GB | Single model copy, N independent minds |

---

## Deployment Topologies

One code path, four real-world shapes. All driven by the **same `build_primary_router()` factory** + env/config overrides:

### 1. Solo laptop (no multi-LLM)
Nothing to configure. `build_primary_router()` detects local capabilities, picks a profile, loads in-process. What most users see on day one.

### 2. Solo desktop with a dedicated model server
User manually runs `llama-cpp-server` in a second terminal; sets `MAXIM_LANE_INFER_REMOTE_URL=http://127.0.0.1:8000/v1`. No tunnel, no peers. Useful for keeping a model hot across Maxim restarts. **This is what Phase 4 already supports today.**

### 3. Home leader + follower(s)
One machine (typically the one with the best GPU) is the **leader**: it auto-spawns llama-cpp-server, exposes it via a Cloudflare tunnel or LAN IP, and still runs its own agentic loop locally. Followers (laptop, Reachy, second desktop) run Maxim with `MAXIM_LANE_INFER_REMOTE_URL` pointed at the leader. Leader is detected via `~/.cloudflared/config.yml` or `MAXIM_ROLE=leader`. **Phase 6 work.**

### 4. LAN mesh with shared consciousness
Multiple Maxim instances discover each other via mDNS. Inference is routed to whichever peer has the best hardware for each request. A designated leader optionally exposes its agent loop to remote CLI frontends so the whole household talks to one Maxim mind. **Phase 7 work.**

The topologies are **additive** — a user can move from solo-laptop → home-leader → LAN mesh without rewriting configs, just by adding env vars / config entries as they scale up.

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

### Phase 6: Dynamic Local Backend Spawning + Leader Mode

Automatically detect available hardware at startup and spawn model server processes for models that should be locally available. On machines configured for remote access, promote this instance to a **leader** that serves inference to peers — same process also runs the normal agentic loop, so the leader machine stays fully usable.

#### Role detection

At startup, classify this instance:

| Signal | Role | Behavior |
|---|---|---|
| `MAXIM_ROLE=leader` env set | explicit leader | Auto-spawn server + expose via tunnel |
| `~/.cloudflared/config.yml` exists with `service: http://localhost:8000` | implicit leader | Same as above |
| GPU present AND `MAXIM_ROLE` unset AND no cloudflared config | local-only | Spawn server, only `127.0.0.1` binding |
| No GPU / `MAXIM_ROLE=client` | follower | Skip spawner; require `MAXIM_LANE_INFER_REMOTE_URL` or fall back to tiny CPU model |

The leader distinction matters for:
- **Bind address**: local-only uses `127.0.0.1`, leader binds `0.0.0.0` so tunnel/LAN can reach it.
- **Peer advertisement**: leader registers itself via mDNS (Phase 7).
- **Lifecycle**: leader's spawned servers outlive the Maxim process (optional `--persist` flag) so a peer isn't cut off when the leader restarts its agentic loop.

The leader still runs its own agentic loop + CLI locally — **it doesn't become "just a server."** It uses its own spawned backends via `127.0.0.1`, identical to how peers reach it. One code path, many deployment topologies.


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

### Phase 7: Peer Mesh — leader proxy, shared queue, discovery, routing

**Prerequisite**: [peer_leader_debug_plan.md](peer_leader_debug_plan.md) Stage A (observability) should land first or in lockstep with 7a. The debug plan's §1 analysis is load-bearing for this phase — it identified the architectural gap that 7a closes.
>
> **Before building Phase 7a**, validate the current peer path is end-to-end functional using [peer_diagnosis_runbook.md](peer_diagnosis_runbook.md). The runbook's 6-rung bisection ladder covers every hop (DNS → edge → tunnel → auth → outbound client → GPU).

**The current architecture** (after Phases 1-6): two sovereign Maxim minds share a GPU via a dumb HTTP proxy. Peers and the leader are independent clients of the same llama-cpp-server, contending for GPU time with no coordination and no leader-side visibility. Phase 7 turns that into a real mesh where:

- The leader's runtime **sees** every peer request (logging, auth, routing decisions).
- Leader + peers share a queue on the leader's `WorkerPool` instead of racing at the inference server.
- Instances discover each other without manually shared URLs.
- Per-request routing picks the best available backend (local / peer / remote) based on measured latency + queue depth.

Phase 7 splits into five sub-phases. Each is landable independently, each closes a specific gap.

#### 7a. LeaderProxy — sidecar reverse proxy + observability hook

A thin HTTP service on `:8099` that sits **in front of** llama-cpp-server (`:8100`). Tunnel ingress flips from `localhost:8100` → `localhost:8099`; llama-cpp-server becomes private-only.

**Responsibilities:**
- **Authoritative auth**: enforce Bearer check BEFORE the request reaches llama-cpp-server. Closes the "solo→leader transition leaves the server unauthenticated" gap identified in the debug plan.
- **Request-id propagation**: generate/preserve `X-Maxim-Request-Id` (UUID4) on every call, pass through to the backend. Peer logs ID on send → leader proxy logs on receipt → trivial correlation across machines.
- **Structured logging**: every inbound request → one log line with `{request_id, source_ip, auth_status, model, latency_ms, status}` via a dedicated `maxim.mesh.trace` logger.
- **`/debug/last-requests`** (localhost-only, auth-gated): ring buffer of last 100 calls for post-hoc inspection.
- **Future hook point**: where 7b plugs in to route requests through the `WorkerPool`.

**Scope**: ~250 LOC stdlib-only HTTP server in `runtime/leader_proxy.py`. Adds ~1-2ms per request. Negligible vs. 44ms inference baseline.

**Status (2026-04-05)**: IMPLEMENTED. `leader_proxy.py` handles auth, structured logging with request-id/peer-IP/latency/tokens/GPU metrics, `/v1/debug/status` (GPU util/VRAM/temp), `/v1/debug/last-requests` (ring buffer, localhost-only), and injects `X-Maxim-GPU-Util`, `X-Maxim-GPU-VRAM`, `X-Maxim-GPU-Temp`, `X-Maxim-Proxy-Ms`, `X-Maxim-Server-Ms` response headers. Auto-started in leader mode. Peer-side `mesh_trace.py` reads GPU metrics from response headers (zero-cost), falling back to `/v1/debug/status` poll for pre-7a leaders.

#### 7a-ext. Remote self-update via LeaderProxy

Allow a peer (or the user from any machine) to trigger `git pull + pip install + restart` on the leader without SSH access. Builds on LeaderProxy's existing auth.

**Endpoint**: `POST /v1/admin/update` (Bearer auth required)

**Flow:**
1. Peer/user sends `POST /v1/admin/update` with optional `{"branch": "main", "dry_run": true}`
2. LeaderProxy runs `git fetch && git log HEAD..origin/main --oneline` to preview changes
3. If `dry_run=false` (default): `git pull origin main && pip install -e .`
4. Health-check: verify llama-cpp-server + LeaderProxy come back on their ports
5. If health check fails: `git checkout HEAD~1 && pip install -e .` and report failure
6. Restart the Maxim process (graceful: finish current inference, then `os.execv`)

**Safety:**
- Only pull from the configured remote + branch (default `origin/main`)
- Refuse if working tree is dirty (`git status --porcelain` non-empty)
- Log every update attempt to the request ring buffer
- Optional `MAXIM_ALLOW_REMOTE_UPDATE=1` env flag (off by default) — leader must explicitly opt in
- `dry_run=true` returns the pending commits without applying, for review

**CLI companion**: `maxim peer update <url>` — calls the endpoint, prints the diff preview, asks for confirmation before sending `dry_run=false`.

**Scope**: ~100 LOC in `leader_proxy.py` + ~30 LOC CLI command. No new dependencies.

#### 7b. Route peer jobs through the leader's WorkerPool

With 7a's proxy in place, parse inbound OpenAI-protocol requests and enqueue them on the leader's `WorkerPool` infer lane instead of forwarding straight to llama-cpp-server. The existing backend inside the lane still hits llama-cpp-server, but now peer jobs share the same queue as the leader's own agent loop.

**What this unlocks:**
- **Fair scheduling** between leader's agentic loop and N peers (no more GPU contention races).
- **Per-peer attribution** via API key header or source IP; requests tagged in logs + metrics.
- **Optional per-peer rate limits** (N jobs/min) — per-device keys from the long-parked security conversation become a prerequisite for meaningful rate-limiting.
- **Cancellation + priority**: the WorkerPool already supports both; peer requests get surfaced to the same machinery that throttles the leader's own work.

**Scope**: ~250 LOC (protocol parsing, lane enqueue, response streaming back through the proxy, SSE passthrough for streaming completions).

**Trade-off**: now peer requests go through two layers (proxy → WorkerPool → backend). If the WorkerPool is saturated, peer latency jumps. Phase 8 metrics make this visible; Phase 7d's routing uses the signal.

#### 7c. PeerRegistry + mDNS discovery

Remove the need for manually shared URLs on LAN. Each Maxim instance advertises itself via mDNS (same mechanism the robot stack already uses for Reachy discovery):

```python
class PeerRegistry:
    """Discover and track peer Maxim instances on the LAN.

    Uses mDNS (zeroconf) to advertise available models and discover peers.
    Each peer exposes its LeaderProxy on a local port.

    Service type: _maxim-llm._tcp.local.
    TXT records: models, vram_gb, device (gpu|cpu), node_id, proxy_port
    """

    SERVICE_TYPE = "_maxim-llm._tcp.local."

    def start(self, *, node_id: str, port: int, advertise: PeerAdvertisement) -> None: ...
    def peers(self) -> list[PeerInfo]: ...
    def get_peer_for_model(self, model_name: str) -> PeerInfo | None: ...
    def stop(self) -> None: ...
```

`PeerInfo` carries `{node_id, host, port, models, device, vram_gb, last_seen}`; `is_alive` uses a 30s heartbeat timeout.

**Opt-in via `[mesh]` extra**: `zeroconf` is an optional dependency (`pip install -e '.[mesh]'`). Non-mesh users don't pay the dep cost.

**Opt-in via env var**: `MAXIM_PEER_ENABLED=1` starts the registry. Off by default so solo/tunnel setups aren't affected.

**Scope**: ~150 LOC + the `zeroconf` dep. Firewalls often block mDNS — `maxim doctor` gets a new "mDNS broadcast reachable" check ([doctor_upgrade_plan.md §6](doctor_upgrade_plan.md)).

#### 7d. InferenceRouter — per-request backend selection

The actual "local vs. peer vs. remote" decision, finally. Each inference call consults the router:

```
Per-request routing chain (first success wins):
  1. Local lane backend if configured          — lowest latency, no network hop
  2. Best LAN peer via PeerRegistry            — 5-20ms extra, may have better GPU
  3. Remote tunnel backend if configured       — 20-50ms extra, always available (if tunnel up)
  4. Fallback: smallest local CPU model        — slow but always works
  5. None (caller handles gracefully)
```

**Routing inputs** (sourced from Phase 8's LaneMetrics):
- Per-backend **queue depth** + current utilization
- Recent **p50/p99 latency** per backend
- Per-backend **failure rate** (backoff on consistently-failing backends)
- **Context window fit** — skip a backend whose `n_ctx` can't hold the request
- **Peer VRAM + advertised device** (GPU over CPU, higher VRAM wins ties)

**Backoff on failure**: consistent 5xx or timeout from a backend → exponential backoff (30s, 60s, 120s, cap at 10min). Doesn't retry the same dead peer every call.

**Scope**: ~200 LOC + wiring into `LaneBackendManager`. The router *augments* the manager — today's single-backend-per-lane behavior is the degenerate case (no peers, no remote, just the local backend).

**Decision logging**: at `MAXIM_PROVENANCE_VERBOSITY=2`, log the full routing tree per request (candidates considered, what was skipped, why). Pairs with Phase 10.

#### 7e. Multi-front input — "one mind, many channels" (split-candidate)

A leader instance exposes its `ConversationalSource` to multiple I/O frontends concurrently. **This is architecturally orthogonal to 7a-7d** — it changes the agent's input boundary, not its compute. Strong candidate to split into a separate "Phase 11" plan.

| Frontend | Transport | Exists? |
|---|---|---|
| Local stdin | Terminal | ✓ (current behavior) |
| Remote CLI | WebSocket over tunnel | new |
| SMS / voice | Twilio webhook | ✓ ([comms/](../../src/maxim/comms/)) |
| Another Maxim instance | mDNS peer over HTTP | new |

**Design constraints**:
- **Input merging** — two humans giving opposing commands: turn-based queue with attribution, or latest-wins with provenance?
- **Auth for remote CLI** — shared-secret token to hit the agent bus, reusing leader's API key or a separate channel key?
- **Percept fan-out** — every connected frontend sees the same percept stream, not just the one driving the current turn.
- **Graceful degradation** — leader disappears → remote frontends fail visibly, not silently hang.

**What stays the same**: agent loop, memory, LLM backends. Only the input/output boundary changes. The comms gateway (Twilio SMS) already proves the pattern; formalizing it means lifting `ConversationalSource` behind a pub/sub so N callers publish/subscribe.

**Recommendation**: ship 7a-7d first. Revisit 7e when you have a concrete use case (shared household Maxim? remote operator watching a Reachy sim?).


#### 7a. Peer advertisement via mDNS (legacy section — superseded by 7c above)

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

### Phase 8: Per-Lane Metrics (prerequisite for Phase 7d routing)

Phase 8 is the signal source for Phase 7d's routing decisions. It also feeds `maxim doctor` so users can answer "is my infer lane actually fast?" empirically. Landable standalone; small, additive, no behavior change.

**Data model:**

```python
from collections import deque
from dataclasses import dataclass, field

@dataclass
class LaneMetrics:
    """Per-lane performance counters with reservoir-sampled latency."""
    # Monotonic counters
    jobs_submitted: int = 0
    jobs_completed: int = 0
    jobs_dropped: int = 0        # stale before execution reached
    jobs_failed: int = 0         # backend returned error
    failover_count: int = 0      # routing chain hit a fallback (Phase 7d)

    # Backend attribution (set by the router, read in logs + doctor)
    local_calls: int = 0         # in-process or loopback
    remote_calls: int = 0        # self-hosted HTTP (any private IP)
    peer_calls: int = 0          # LAN peer via mDNS (Phase 7c-d)
    cloud_calls: int = 0         # public HTTPS endpoints

    # Queue pressure (sampled at each submit)
    current_queue_depth: int = 0
    peak_queue_depth: int = 0

    # Latency reservoir (last 100 samples, thread-safe)
    latencies_ms: deque[float] = field(default_factory=lambda: deque(maxlen=100))

    # Token + cost (from LLMResponse.complete_with_usage)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0

    @property
    def avg_latency_ms(self) -> float: ...
    @property
    def p50_latency_ms(self) -> float: ...
    @property
    def p99_latency_ms(self) -> float: ...
    @property
    def remote_ratio(self) -> float: ...
    @property
    def failure_rate(self) -> float: ...
```

**Integration points:**

- **`LaneBackendManager` owns a `dict[str, LaneMetrics]`** keyed by lane name.
- **Record hooks** wrap `backend.complete_with_usage()`: start-timestamp → completion delta, token counts, backend attribution via the `kind` field already exposed in `describe()`.
- **Per-call metadata**: every submission gets `X-Maxim-Request-Id` (UUID4) attached; the ID appears in metrics records and in the `maxim.mesh.trace` logger (shared with Phase 7a).
- **Exposure**:
  - `manager.metrics_snapshot()` → dict for programmatic access
  - `maxim doctor` gains a new "Lane metrics" section (per-lane p50/p99/counts)
  - `maxim doctor --json` (from [doctor_upgrade_plan.md](doctor_upgrade_plan.md)) includes the snapshot
  - `MAXIM_METRICS_INTERVAL_S=30` (default off) emits a periodic log line

**Overlaps with the debug plan:** Phase 8 absorbs [peer_leader_debug_plan.md](peer_leader_debug_plan.md)'s `MAXIM_LANE_TRACE=1` flag — instead of a separate trace mechanism, metrics are always recorded and trace mode just prints each record at INFO rather than accumulating. One mechanism, two verbosities.

**Scope**: ~150 LOC + ~50 LOC tests. Additive. No existing-behavior change.

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

### Phase 10: Observability & Verbose Tracing

**Timing:** Wait until at least Phases 3 + 7 are live. Observability designed on a single-backend world will miss what actually matters in a mesh (routing decisions, peer failovers, remote-vs-local pressure). Building it too early means rewriting it.

**Goal:** Every LLM call in the logs should answer three questions at a glance:

1. **Where did this response come from?** — lane, model profile, backend type (`local-llama`/`local-torch`/`remote-http`/`peer`), host identity if remote.
2. **What memory pressure was the backend under?** — VRAM/RAM usage for local backends, queue depth for remote, backpressure signals.
3. **What compute pressure?** — tokens/sec, p50/p99 latency, active workers in the lane, failover count since start.

**Implementation sketch:**

- **Structured log records per LLM call.** Extend `LLMResponse` (or add a `LLMCallTrace` alongside) with `lane`, `backend_kind`, `backend_host`, `model_profile`, `tokens_in`, `tokens_out`, `latency_ms`, `vram_used_mb`, `queue_depth_at_submit`. Log one JSON line per call at DEBUG, plus a compact one-liner at INFO.
- **Periodic pressure snapshots.** A lightweight background task (~every 5s, gated on `MAXIM_PROVENANCE_VERBOSITY>=1`) samples `torch.cuda.memory_allocated()`, `psutil.virtual_memory()`, and per-lane `WorkerPool.status()`, and emits one structured line. Makes it easy to correlate latency spikes with memory/compute events.
- **Routing-decision traces.** When `InferenceRouter` picks a backend (local → peer → remote fallback chain), log *why* — which candidates were considered, which were skipped, why. At `MAXIM_PROVENANCE_VERBOSITY=2`, include the full decision tree.
- **Reuse existing provenance infrastructure.** `src/maxim/provenance/` already has 2-tier tracing (cycle traces + activity log). LLM-call traces should plug into the same system so they show up in the existing cycle trace viewer, not a separate log stream.
- **Env toggles** (align with existing convention):
  - `MAXIM_LLM_TRACE=1` — enable per-call structured trace
  - `MAXIM_LLM_PRESSURE_INTERVAL_S=5` — snapshot cadence
  - Tied into `MAXIM_PROVENANCE_VERBOSITY` for the broader verbosity knob

**What this unlocks:** diagnosing "why is this sim slow?" without attaching a profiler — you see at a glance that `infer` lane spent 60% of the session on a peer because the local GPU OOM'd, or that the review lane queue stayed at depth 8 for 20s because a 7B CPU model was thrashing.

**Non-goals:** Prometheus/Grafana export, long-term metric storage, dashboards. Those can layer on top of structured log lines later.

---

## Implementation Sequencing

| Phase | What | Status | Dependencies |
|-------|------|--------|-------------|
| **1** | `LaneConfig` gains model fields | ✅ done | None |
| **2** | `LaneModelConfig` + capability-driven assignment | ✅ done | Phase 1 |
| **3** | `LaneBackendManager` + gates (`MAXIM_MAX_CONCURRENT_BACKENDS`, `MAXIM_MAX_CLOUD_LANES`) | ✅ done | Phase 1 |
| **4** | Remote model server (`_build_remote_backend` + llama-cpp-server docs) | ✅ done | Phase 3 |
| **5** | Cloudflare tunnel setup (`maxim tunnel` subcommand) | ✅ done | Phase 4 |
| **6a** | `LocalServerSpawner` (auto-spawn llama-cpp-server) | ✅ done | Phase 3 |
| **6b** | Leader-mode detection + `TunnelDaemonSpawner` | ✅ done | Phase 6a |
| **8** | `LaneMetrics` (prerequisite for 7d routing) | **next** | Phase 3 |
| **7a** | `LeaderProxy` reverse-proxy + request-id + auth + `/debug/last-requests` | after 8 | Phase 6 + debug-plan Stage A |
| **7b** | Route peer jobs through leader's `WorkerPool` | after 7a | Phase 7a |
| **7c** | `PeerRegistry` + mDNS discovery (opt-in `[mesh]` extra) | after 7a | Phase 7a |
| **7d** | `InferenceRouter` (per-request local→peer→remote fallback) | after 7b+7c | Phases 7b, 7c, 8 |
| **7e** | Multi-front input — **split-candidate for a Phase 11** | deferred | Phase 7a |
| **9** | Environment variable / config support | ✅ mostly done (absorbed into 3+4+6) | — |
| **10** | Observability & verbose tracing (structured `maxim.mesh.trace`) | after 7d | Phases 7d + 8 |

**Cross-plan references:**
- [peer_leader_debug_plan.md](peer_leader_debug_plan.md) — Stage A (observability foundations) is a prerequisite for Phase 7a. Its §1 architectural analysis motivated the Phase 7 restructure. Stage D items are now folded into Phases 7a-7d.
- [doctor_upgrade_plan.md](doctor_upgrade_plan.md) — `maxim doctor --json` + mDNS check feed into Phase 7 operability.

**Also landed outside the plan** (discovered during implementation):
- `build_primary_router()` factory unifying agentic_runtime + sim orchestrator LLM construction — prevents double-model loading.
- `MAXIM_LANE_{NAME}_REMOTE_URL/_MODEL/_API_KEY` env overrides.
- `_validate_base_url` permits `http://` for private-IP endpoints.
- LLMRouter respects `allow_local_endpoints` — self-hosted providers bypass cloud gate, cost check, and PII redaction.

**Path forward from the current state (Phases 1–6 done):**

1. **[peer_leader_debug_plan.md](peer_leader_debug_plan.md) Stage A** (~1 session) — request-id propagation, `MAXIM_LANE_TRACE`, `MAXIM_PEER_LOG_REQUESTS`, `maxim tunnel tail`. No behavior change; turns opacity into traceable logs. Prerequisite for Phase 7a.
2. **Phase 8** (~1 session) — `LaneMetrics`. Small, landable standalone, feeds `maxim doctor`. Its data model is what Phase 7d routes against. Absorbs the debug plan's `MAXIM_LANE_TRACE` flag.
3. **Phase 7a** (~1-2 sessions) — `LeaderProxy` reverse-proxy with request-id + auth + structured logging. Closes the "leader runtime never sees peer requests" gap identified in the debug plan §1.
4. **Phase 7b** (~2 sessions) — peer jobs enqueue on the leader's `WorkerPool`. Fair scheduling between leader's own agent loop and N peers.
5. **Phase 7c** (~1-2 sessions) — `PeerRegistry` + mDNS discovery. Can land in parallel with 7b since 7a already created the proxy identity.
6. **Phase 7d** (~2 sessions) — `InferenceRouter` with fallback chain. Uses Phase 8 metrics + 7c peer list.
7. **Phase 10** — structured `maxim.mesh.trace` + verbose tracing across the whole mesh. Natural follow-on to 7d.
8. **Phase 7e / Phase 11 (multi-front input)** — defer until there's a concrete use case. Architecturally orthogonal to 7a-7d; doesn't block anything else.

Phases 4-5 were external configuration + docs and are both done. Phase 9 (env vars) was largely absorbed into Phases 3+4+6.

**Why 8 before 7a**: Phase 8 is smaller, self-contained, and its request-id + latency recording are exactly the observability primitives Phase 7a wants to emit. Landing metrics first means 7a can reuse them rather than building parallel trace plumbing.

**Why the debug plan ships first**: it identified a real architectural gap (§1) — without the LeaderProxy, leader-side code never sees peer requests, so there's nothing to log or route. Stage A's flags + request-id propagation give us the observability we need to *validate* 7a when we build it.

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
