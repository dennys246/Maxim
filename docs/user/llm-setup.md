# LLM Setup

## Overview

Maxim uses local LLM inference by default via llama.cpp. Cloud backends (Anthropic, OpenAI) are optional and opt-in. Local inference keeps everything on your machine -- no data leaves your network.

## Quick Start

```bash
pip install pymaxim[llm-llama]
maxim --list-models                    # see available local models
maxim --llm smollm-1.7b                # auto-downloads on first run
```

## Local Models

### Available Profiles

| Profile | Model | Size | Context | Best For |
|---------|-------|------|---------|----------|
| `smollm-1.7b` | SmolLM 1.7B Instruct | ~1.1 GB | 2048 | CPU-only, low RAM, fast iteration |
| `mistral-7b` | Mistral 7B Instruct v0.2 | ~4.4 GB | 8192 | Best balance of quality and speed |
| `phi3-mini` | Phi-3 Mini 4K Instruct | ~2.3 GB | 4096 | Good quality, moderate size |
| `llama3-8b` | Llama 3 8B Instruct | ~4.9 GB | 8192 | Highest quality local |
| `qwen2-7b` | Qwen2 7B Instruct | ~4.4 GB | 8192 | Strong multilingual support |
| `qwen2.5-14b` | Qwen2.5 14B Instruct | ~8.4 GB | 32768 | Strong large-context model, fits on 16 GB+ VRAM |
| `qwen2.5-32b` | Qwen2.5 32B Instruct | ~19.9 GB | 32768 | Leader-grade default for 48 GB+ Apple Silicon or 24 GB+ VRAM |
| `mixtral-8x7b` | Mixtral 8x7B Instruct v0.1 | ~26.4 GB | 32768 | MoE, all 8 experts in memory; great on 48 GB+ |
| `llama-3.1-70b` | Llama 3.1 70B Instruct | ~42.5 GB | 32768 | True large-model territory; needs 64 GB+ unified memory |

### Downloading Models

```bash
maxim --list-models                    # see what's available + download status
maxim --llm mistral-7b                 # auto-downloads on first run
```

The first launch with a given `--llm` profile prompts to download the GGUF (~4 GB for Mistral 7B Q4_K_M). Set `MAXIM_AUTO_DOWNLOAD_MODELS=1` to skip the prompt in CI.

### Quantization

Quantization controls the quality vs. memory tradeoff for local models:

| Level | Quality | Memory | Use Case |
|-------|---------|--------|----------|
| `Q3_K_M` | Fair | Lowest | Very memory constrained |
| `Q4_K_M` | Good | Low | Default, recommended |
| `Q5_K_M` | Better | Medium | When quality matters more |
| `Q8_0` | Excellent | High | Maximum quality |

Set the quantization level with an environment variable:

```bash
export MAXIM_LLM_QUANTIZATION=Q4_K_M
```

### Adding Custom Profiles

The bundled profiles cover the common cases, but any GGUF on HuggingFace (or any local GGUF file) can become a first-class Maxim profile via `maxim model add`. The CLI writes to `~/.config/maxim/profiles.yml`, which sits in the same declarative-config directory as `peer.yml` and `mesh.yml`.

**From a HuggingFace repo:**

```bash
maxim model add my-qwen-32b-q5 \
    --hf bartowski/Qwen2.5-32B-Instruct-GGUF:Qwen2.5-32B-Instruct-Q5_K_M.gguf \
    --n-ctx 32768

# Then use it like any bundled profile:
maxim --llm my-qwen-32b-q5 --auto-download
```

The `--hf` argument is `REPO:FILE` — the repo (which may contain `/`) and the GGUF filename, separated by the first `:`.

**From a local file:**

```bash
maxim model add my-local --local ~/models/custom.gguf --chat-format llama3_instruct
maxim --llm my-local
```

**Chat-format auto-inference.** When `--chat-format` is omitted, Maxim substring-matches the profile name (and HF repo basename) against `qwen`, `llama-3`, `llama-2`, `mixtral`, `mistral`, `phi-3`, `phi-2`, `gemma` to pick a default. If no rule matches, you must supply `--chat-format` explicitly (one of `chatml`, `mistral_instruct`, `llama3_instruct`, `llama2_chat`, `phi3`, `phi`, `gemma`) — Maxim refuses to guess in the no-match case so a wrong template doesn't silently mis-render every prompt.

**Manage profiles:**

```bash
maxim model list                # show user profiles
maxim model remove my-qwen-32b-q5
```

**Hand-editing `~/.config/maxim/profiles.yml`** works too — useful when you want to preserve comments (which the CLI verbs strip on round-trip) or supply the optional `arch:` block for accurate VRAM estimation:

```yaml
profiles:
  my-qwen-32b-q5:
    backend: llama_cpp                  # llama_cpp or pytorch
    prompt_style: chatml                # chatml | mistral_instruct |
                                        # llama3_instruct | llama2_chat |
                                        # phi3 | phi | gemma
    download:
      hf_repo: bartowski/Qwen2.5-32B-Instruct-GGUF
      hf_file: Qwen2.5-32B-Instruct-Q5_K_M.gguf
    n_ctx: 32768
    aliases: [qwen32q5]
    arch:                               # optional — enables fine VRAM estimation
      n_layers: 64
      n_kv_heads: 8
      head_dim: 128
      kv_type_bytes: 2
      weights_gb: 23.0
```

User profiles WIN over built-ins on name or alias collision (a WARNING is logged), so you can override a bundled profile's quantization or context window without touching source. `maxim doctor` surfaces user-profile counts in its Environment section and flags YAML syntax errors before they block startup.

### Per-Mode Response Configuration

LLM context windows and response lengths adapt automatically to the current operational mode. Configuration lives in `~/.maxim/config/llm.json` under `mode_response_config`:

| Mode | Response Tokens | Context Window | Format |
|------|----------------|----------------|--------|
| sleep | 64 | 256 | minimal |
| observe | 128 | 512 | minimal |
| exploration | 256 | 1,024 | brief |
| live / train | 512 | 2,048 | conversational |
| active-assistance | 768 | 2,048 | detailed |
| reflection | 1,024 | 3,072 | detailed |
| research | 2,048 | 4,096 | academic |

Lower modes save tokens and latency; higher modes give the LLM more room to reason. The mode is set via `--mode` or switches automatically based on context.

## Cloud Backends (Optional)

Cloud backends provide faster inference and higher quality reasoning than local models. They're especially useful for simulation agent mode where both the orchestrator and AUT need fast LLM access.

Cloud calls are budgeted (token and cost limits enforced), audit-logged (every call recorded in `~/.maxim/util/cost_state.json`), and persist cost data across sessions.

### Anthropic (Claude)

**1. Get an API key:**
- Go to [console.anthropic.com](https://console.anthropic.com)
- Sign up or log in
- Navigate to **API Keys** in the left sidebar
- Click **Create Key**, give it a name, and copy the key (starts with `sk-ant-`)

**2. Install the SDK:**
```bash
pip install -e '.[llm-anthropic]'
```

**3. Set the environment variable:**
```bash
export ANTHROPIC_API_KEY="sk-ant-api03-..."
```

To make it permanent, add it to your shell profile (`~/.bashrc`, `~/.zshrc`, or `~/.bash_profile`):
```bash
echo 'export ANTHROPIC_API_KEY="sk-ant-api03-..."' >> ~/.zshrc
source ~/.zshrc
```

**4. Add a Claude profile to `~/.maxim/config/llm.json`** (under the `"profiles"` section):
```json
"claude-sonnet": {
  "backend": "anthropic",
  "model": "claude-sonnet-4-6",
  "n_ctx": 65536,
  "max_tokens": 4096
},
"claude-haiku": {
  "backend": "anthropic",
  "model": "claude-haiku-4-5-20251001",
  "n_ctx": 65536,
  "max_tokens": 4096
}
```

**5. Run with Claude:**
```bash
# Normal agentic mode
maxim --language-model claude-sonnet

# Simulation agent mode (recommended — fast turns)
maxim --sim "test safety" --language-model claude-sonnet
```

### Available Claude Models

| Profile | Model | Speed | Cost | Best For |
|---------|-------|-------|------|----------|
| `claude-haiku` | Claude Haiku 4.5 | Fastest | $0.80/1M in | Quick sim runs, high-volume testing |
| `claude-sonnet` | Claude Sonnet 4.5 | Fast | $3.00/1M in | Best balance for sim + refinement |

### OpenAI

```bash
pip install -e '.[llm-openai]'
export OPENAI_API_KEY="sk-..."
```

Add a profile to `~/.maxim/config/llm.json`:
```json
"gpt-4o": {
  "backend": "openai",
  "model": "gpt-4o",
  "n_ctx": 128000,
  "max_tokens": 4096
}
```

### Cost Tracking & Enforcement

All cloud API calls are automatically tracked in `~/.maxim/util/cost_state.json` with:
- Per-model pricing (input, output, cached tokens)
- Rolling windows (hourly, daily, monthly)
- Spend rate estimates (3h, 24h, 7d EMAs)
- Per-provider breakdowns

Use the `energy_status` introspection tool or `inspect_aut(energy_status)` in simulation mode to check token usage and budget projections in real time.

### Cost Limits

The router enforces budget limits at multiple levels:

| Limit | Default | Behavior When Hit |
|-------|---------|-------------------|
| Per-request | $0.50 | Skips expensive provider, tries cheaper |
| Hourly | $1.00 | Downgrades model (Opus -> Sonnet -> Haiku) |
| Daily | $10.00 | Downgrades model, falls back to local |
| Monthly | $100.00 | Downgrades model, falls back to local |
| **Session ceiling** | **$5.00** | **Hard reject -- ALL requests blocked** |

The session ceiling is the only hard stop. All other limits degrade gracefully (cheaper model or local fallback). Configure in `~/.maxim/config/llm.json`:

```json
{
  "routing": {
    "max_session_cost": 20.00,
    "max_cost_per_hour": 5.00,
    "max_cost_per_day": 50.00,
    "fallback_on_budget_exceeded": "local"
  }
}
```

Set `fallback_on_budget_exceeded` to `"reject"` for hard enforcement on all limits (not just the session ceiling).

## Configuration File

Edit `~/.maxim/config/llm.json` for fine-grained control over LLM behavior:

```json
{
  "enabled": true,
  "profile": "mistral-7b-instruct-v0.2",
  "max_tokens": 512,
  "temperature": 0.0,
  "quantization": "Q4_K_M"
}
```

### Per-Mode Response Tuning

The configuration includes per-mode token budgets that control how much output the LLM generates in each operating context:

| Mode | Max Response Tokens | Purpose |
|------|-------------------|---------|
| `observe` | 128 | Minimal -- quick perception summaries |
| `sleep` | 64 | Minimal -- background consolidation |
| `exploration` | 256 | Brief -- exploratory reasoning |
| `live` | 512 | Conversational -- interactive responses |
| `reflection` | 1024 | Detailed -- self-assessment and review |
| `research` | 2048 | Academic -- thorough analysis |

## GPU Acceleration

### NVIDIA (CUDA)

llama.cpp auto-detects CUDA when available. To force CPU-only inference:

```bash
CUDA_VISIBLE_DEVICES="" maxim
```

### Apple Silicon (Metal)

The `llama-cpp-python` package builds with Metal support on macOS by default. No additional configuration is needed.

**Tier behavior on Apple Silicon (P2 + P3):** Maxim now reports the Mac's effective unified-memory budget as `vram_gb` and admits MPS into the **large** tier. Previously MPS was hard-excluded — a 24 GB Mac was capped at `mistral-7b` on the medium tier even though it could comfortably run a 14B-class model. With P4a's tier-table expansion, a 24 GB Mac now defaults to `qwen2.5-14b-instruct` on the large tier, and dynamic n_ctx sizing (P4c) picks an appropriate context window for the available memory (typically 8K-16K depending on headroom). Intel Macs (no MPS) and Apple Silicon Macs whose effective VRAM falls below 4 GB still route to the medium tier.

### Blackwell GPUs (RTX 5080/5090)

Blackwell (sm_120) GPUs require special handling because:

1. **GStreamer/CUDA conflict**: reachy-mini's WebRTC media pipeline segfaults when CUDA is active on Blackwell. Maxim sets `GST_CUDA_NO_CUDA=1` + `REACHY_MEDIA_BACKEND=default` to neutralize this.
2. **CUDA 12.8 requirement**: sm_120 needs CUDA toolkit ≥ 12.8 and wheels built against it. PyPI's default `llama-cpp-python` and `torch` wheels don't target sm_120.

**Default behavior:** On Blackwell detection Maxim sets `GST_CUDA_NO_CUDA=1` + `REACHY_MEDIA_BACKEND=default` to neutralize GStreamer, but keeps CUDA visible for `llama-cpp-python` / `torch`. This requires Blackwell-compatible builds of both (see below).

**Opt-out (CPU-only):** Set `MAXIM_BLACKWELL_HIDE_CUDA=1` to hide CUDA from the entire process. Use this if you still hit a GStreamer crash despite the GST guard, or if you haven't installed Blackwell-compatible wheels yet.

#### Install Blackwell-compatible wheels

**PyTorch (cu128):**

```bash
pip install "torch>=2.7" --index-url https://download.pytorch.org/whl/cu128
pip install -e '.[llm-torch]' --no-deps  # installs the rest of the extra
```

**llama-cpp-python with CUDA for sm_120** (source build required, PyPI ships CPU-only; abetlen's cu124 wheel index is stuck at v0.2.67):

```bash
# Requires CUDA 12.8 toolkit installed (e.g., /usr/local/cuda-12.8)
bash scripts/build_llama_cpp_blackwell.sh
```

Or manually:

```bash
export CUDACXX=/usr/local/cuda-12.8/bin/nvcc
export CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=120"
export FORCE_CMAKE=1
pip install --upgrade --force-reinstall --no-cache-dir \
    --no-binary=llama-cpp-python llama-cpp-python==0.3.20
```

Build takes 5–15 min. Verify success:

```bash
python scripts/smoke_test_blackwell_gpu.py
# expect: gpu_offload_supported: True, backend_info: ... CUDA ...
```

Then run:

```bash
maxim --language-model mistral-7b
```

## Remote & Distributed LLMs

Maxim can serve LLM inference from a remote machine over HTTP. Typical use cases:

- Offload heavy inference from a laptop or Reachy to a home PC with a GPU.
- Keep one hot model resident on a home server; attach multiple Maxim instances.
- Use a cloud provider (Anthropic, OpenAI) via its OpenAI-compatible API.

All remote inference goes through **lane backends** — each WorkerPool lane can
independently point at a different local model, a local LAN server, or a cloud API.

### Safety gates

Two environment variables control how many backends can be created. Defaults are
conservative; raise them only as needed:

| Env var | Default | Effect |
|---|---|---|
| `MAXIM_MAX_CONCURRENT_BACKENDS` | `2` | Hard cap on cached backends. Protects VRAM/RAM. |
| `MAXIM_MAX_CLOUD_LANES` | `0` | Hard cap on lanes targeting a cloud endpoint. **Must be raised to use Claude, OpenAI, etc.** |

Self-hosted servers (localhost / private-IP endpoints) are **not** counted as
cloud, so they bypass `MAXIM_MAX_CLOUD_LANES`. The session-cost ceiling in
`~/.maxim/config/llm.json` (`max_session_cost`, default $5.00) provides a second layer.

### Auto-spawn (Phase 6) — zero-terminal setup

If you have a GPU and the `[llm-server]` extra installed, Maxim will **automatically launch a llama-cpp-server subprocess** at startup, wire the large lane to it, and shut it down when Maxim exits. No second terminal needed.

```bash
pip install -e '.[llm-server]'  # one-time: bundles sse-starlette, openai SDK, etc.
maxim                           # that's it — server spawns, lane wired, GPU used
```

On startup you'll see a banner like:

```
 ──────────────────────────────────────────────────────────────
  Maxim LLM lanes
  infer   self-hosted http://127.0.0.1:8100/v1
  review  local   smollm-1.7b-instruct (cpu)
  record  (no LLM)
 ──────────────────────────────────────────────────────────────
```

**Auto-spawn is skipped if** any of:

- `MAXIM_AUTO_SPAWN_LLM_SERVER=0` (opt-out)
- No GPU detected
- `MAXIM_LANE_LARGE_REMOTE_URL` already set (user pointed at a different server)
- The profile's GGUF file isn't present locally
- `llama_cpp.server` isn't installed (missing `[llm-server]` extra)

**Config knobs:**

| Env var | Default | Effect |
|---|---|---|
| `MAXIM_AUTO_SPAWN_LLM_SERVER` | `1` | Set `0`/`false` to disable auto-spawn |
| `MAXIM_AUTO_SPAWN_PORT` | `8100` | Port for the spawned server (avoid 8000 if you run your own) |
| `MAXIM_LLM_N_CTX` | _(unset)_ | Force a specific n_ctx for the spawned server. Same effect as `--llm-n-ctx N`. |
| `MAXIM_AUTO_SPAWN_N_CTX` | _(unset)_ | Legacy alias for `MAXIM_LLM_N_CTX`. Kept for in-place upgrades; new installs should prefer `MAXIM_LLM_N_CTX`. |
| `MAXIM_AUTO_DOWNLOAD_MODELS` | _(unset)_ | Set to `1` to skip the interactive 'download model? [y/N]' prompt and proceed with any missing GGUF download. Same effect as `--auto-download`. |
| `MAXIM_DATA_BUDGET_GB` | _(unset)_ | Optional soft cap on Maxim's total disk usage. When set, the auto-download preflight refuses if the new download would push `~/.maxim/` past this budget. |

**Auto-download (P5):** when the active LLM profile's GGUF is not on disk, Maxim's tier-detection layer calls `ensure_available()` before auto-spawn. If you're at a tty it prompts (`Maxim wants to download 'X' (~Y GB)... Proceed? [y/N]`) with a 30-second timeout; if you pass `--auto-download` (or set the env var) it proceeds without prompting; if stdin isn't a tty and neither flag is set, it fails fast with an actionable error and the exact `python -m maxim.models.download --llm <profile>` command. Concurrent `maxim` invocations on the same machine serialize via an advisory file lock at `~/.maxim/util/download.lock` so two processes can't race on the same target file.

**Dynamic n_ctx (P4c):** when neither env var is set, Maxim sizes the spawned server's context window from the active profile's architecture metadata + your detected VRAM. A 16 GB CUDA card running `qwen2.5-14b` lands at ~4096; a 24 GB Apple Silicon mac at ~16384; the profile's declared 32K is used as a hard ceiling so we never request more than the model was trained for. Override with `--llm-n-ctx N` if you need a specific value (e.g. tuning against a tight VRAM budget).

**Auto-discovery:** if a llama-cpp-server is already answering on the auto-spawn port, Maxim will **reuse it** rather than spawning a duplicate. Running two `maxim` terminals from the same shell is transparent — first spawns, second detects the existing server and wires its own lane to it. Two Maxim "minds" share one model copy in VRAM.

**Signal isolation:** the spawned server runs in its own process group so Ctrl+C on the Maxim CLI doesn't kill it mid-shutdown. The server stays alive through Maxim's cleanup (e.g., sim-report LLM roundup) and is stopped explicitly via the atexit handler.

**Stale remote URL recovery (P6):** every configured `remote_url` is probed at startup using a structured two-attempt probe. Outcomes:
- `ok` — lane stays wired.
- `auth_rejected` (HTTP 401) — leader is alive but rejected the API key. Lane STAYS wired (because the leader is the right target — rotate the key with `maxim peer key`, don't fall back locally).
- `dns_fail` / `tls_error` / `connection_refused` / `timeout` / `http_5xx` / `other` — lane is dropped, auto-spawn or local fallback takes over, and the warning carries an outcome-specific fix hint.

Probe results are cached at `~/.maxim/util/last_probe_status.json` for 60 seconds (configurable via `MAXIM_REMOTE_PROBE_CACHE_TTL_S`) so repeated startups within the window pay zero probe cost. `maxim peer connect/forget/restart/update/llm` clear the cache so the next startup re-probes the freshly-changed leader. CI bypass: `MAXIM_SKIP_REMOTE_PROBE=1`.

### Leader mode (Phase 6b) — be the cluster's inference host

Leader mode changes the bind address of the auto-spawned server from `127.0.0.1`
(loopback only) to `0.0.0.0` (LAN + tunnel reachable). The Maxim instance keeps
running its own CLI + agent loop — it just also serves peers.

**You become a leader automatically if:**

- `~/.cloudflared/config.yml` or `/etc/cloudflared/config.yml` exists, OR
- You set `MAXIM_ROLE=leader`

**Auto-spawn of cloudflared daemon:** when leader mode fires on `maxim` startup AND a tunnel config is present AND no cloudflared daemon is already running, Maxim launches the daemon as a managed subprocess alongside the llama-cpp-server. Respects existing systemd services (checks `pgrep` before spawning). Opt out with `MAXIM_AUTO_SPAWN_TUNNEL=0`.

Other valid roles:

| `MAXIM_ROLE` | Bind | Use |
|---|---|---|
| `leader` | `0.0.0.0` | Host for peers (home PC, desktop with GPU) |
| `client` | `127.0.0.1` | Follower — pairs with `MAXIM_LANE_LARGE_REMOTE_URL` |
| `solo` | `127.0.0.1` | Single-machine default (no peers) |

**Setting up a home leader:**

1. Install cloudflared + configure tunnel (see next section) — or skip and just use LAN IP
2. Run `maxim` — auto-spawn detects cloudflared, promotes to leader, binds `0.0.0.0`
3. On a peer machine: `MAXIM_LANE_LARGE_REMOTE_URL=https://maxim-llm.yourdomain.com/v1 maxim`

The leader's Maxim still runs locally too — both your home CLI and the laptop CLI hit the same GPU.

### Running a local llama-cpp model server (Phase 4)

On a machine with a capable GPU (e.g., your home PC), serve a GGUF model via
llama-cpp-python's built-in HTTP server:

```bash
# Install server dependencies
pip install 'llama-cpp-python[server]'

# Serve a model — adjust --model path and --n_gpu_layers to your hardware
python -m llama_cpp.server \
    --model ~/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf \
    --n_gpu_layers -1 \
    --host 0.0.0.0 \
    --port 8000 \
    --chat_format chatml
```

Keep that terminal running. The server exposes an OpenAI-compatible API at
`http://<host>:8000/v1`.

On the client machine, point a lane at the server via environment overrides
(temporary) or `~/.maxim/config/llm.json` (persistent):

```bash
# Quick test — one-shot remote lane
MAXIM_LANE_LARGE_REMOTE_URL=http://192.168.1.10:8000/v1 \
MAXIM_LANE_LARGE_REMOTE_MODEL=mistral-7b-instruct-v0.2 \
maxim --language-model mistral-7b
```

Or pin it in `~/.maxim/config/llm.json`:

```json
{
  "lane_models": {
    "large": {
      "remote_url": "http://192.168.1.10:8000/v1",
      "model": "mistral-7b-instruct-v0.2"
    }
  }
}
```

**Latency:** LAN ~5-20 ms + inference. The agentic loop is async (WorkerPool),
so network hops are absorbed by the next cycle — you'll notice it only in
reaction-time budgets. For real-time motor control, keep a local backend on
the lane that drives motion.

### `maxim doctor` — environment diagnostics

Run anytime to see what's configured and what's missing:

```bash
maxim doctor           # print a one-shot report
maxim doctor --retry   # walk through failing checks, re-test after each fix
```

Checks:
- Platform (OS, runtime — WSL1/WSL2/native/docker, Linux distro, arch)
- GPU / CUDA availability and VRAM
- `llama_cpp.server` installed (the `[llm-server]` extra)
- Auto-spawn server reachable on port 8100
- Leader mode / bind address
- **LAN access with platform-specific fix commands** (WSL2 netsh, Linux ufw/firewalld, macOS settings hint, Windows `New-NetFirewallRule`)
- `cloudflared` installed (with OS-specific install command)
- Tunnel config file + API key

The fix hints include **your actual IP addresses** detected from the system — for WSL2, both the WSL IP (from `hostname -I`) and the Windows host's LAN IP (parsed from `ipconfig.exe`).

### `maxim peer connect <url>` — one-time peer setup

Symmetric to `maxim tunnel setup` on the leader side. Configures this machine to route inference to a leader's URL, persists the config, so subsequent `maxim` runs just work.

```bash
# Interactive (prompts for key)
maxim peer connect https://maxim.yourdomain.com/v1

# Non-interactive
maxim peer connect https://maxim.yourdomain.com/v1 --key "paste-from-leader" --model mistral-7b
```

Stores `{url, api_key, model?, is_cloud}` at `~/.config/maxim/peer.yml` (or `%APPDATA%\maxim\peer.yml` on Windows), mode 0600 on POSIX. By default runs `maxim peer test` first and refuses to save if connectivity fails — pass `--skip-test` to save unconditionally.

**Inspect / reset:**
```bash
maxim peer show      # print current peer config (key truncated)
maxim peer forget    # remove the peer config file
```

**How it's used:** on every `maxim` startup, if `peer.yml` exists the URL + key + optional model are set as defaults for `MAXIM_LANE_LARGE_*` env vars. **Env vars still win** — you can do per-session overrides without editing the file.

For public URLs (tunnel hostnames), `is_cloud: true` is detected automatically and raises `MAXIM_MAX_CLOUD_LANES` to 1 on startup.

### `maxim peer test <url>` — verify peer connectivity

Run from a peer machine to verify the leader is reachable and correctly authenticated:

```bash
maxim peer test http://192.168.1.47:8100/v1
maxim peer test https://maxim.yourdomain.com/v1 --key sk-ant-xxx
```

Runs four checks:
1. DNS resolution
2. HTTP(S) handshake
3. `GET /v1/models` (401 → tells you the key is wrong)
4. Chat completion round-trip with latency timing

Uses `MAXIM_LANE_LARGE_REMOTE_API_KEY` from env if `--key` isn't passed. Exit code 0 = fully working, 1 = any failure.

#### Interpreting peer-test failures

| Result | What it means | Where to fix |
|---|---|---|
| `URL has no host` | Missing scheme (no `https://`) or malformed URL | Prepend `https://`; don't paste angle-bracketed placeholders (`<hostname>`) literally — zsh will parse them as redirection syntax |
| `DNS fails` | Hostname doesn't resolve | Verify the tunnel DNS record on the leader's Cloudflare dashboard; confirm the exact hostname with `maxim tunnel status` on the leader |
| `SSL: CERTIFICATE_VERIFY_FAILED` | You're reaching a different server than you think — wrong hostname, parked domain, or a stale tunnel with an expired cert | Re-verify the hostname on the leader; don't bypass TLS verification |
| `HTTP 401` | Reached the leader; key rejected | Re-copy the key from `maxim tunnel key export` on the leader; if uncertain, rotate with `maxim tunnel key rotate` and export again |
| `HTTP 403` | Request blocked before reaching the model — either the leader's auth layer or a Cloudflare Access / WAF policy in front of the tunnel | Check Cloudflare Zero Trust → Access → Applications for a policy on the tunnel hostname, and verify the leader's key. See diagnostic below |
| `HTTP 502` | Cloudflare reached the tunnel edge but the origin didn't answer — leader process isn't running, or cloudflared can't reach the local server | On the leader: confirm `MAXIM_ROLE=leader maxim` is running and `cloudflared` is connected (`maxim tunnel status` should show an active tunnel) |

**Diagnosing 401 vs 403 vs 502:** run `curl -v -H "Authorization: Bearer <key>" https://<hostname>/v1/models` and look at the response headers:

- Headers contain only `server: cloudflare` + `cf-ray` with a small body → Cloudflare edge is answering (403 = Access/WAF block, 502 = tunnel origin unreachable, 521/522 = origin down/timeout). The request never reached your server.
- Headers come from your actual backend (e.g., `server: uvicorn`, `llama-cpp-server`) → the leader itself is responding. A 401/403 here means the key is wrong or the server's auth middleware rejected it.

**Don't** use `--skip-test` to paper over a failing connection test. The test is what catches wrong hostnames, wrong keys, and misconfigured tunnels *before* they get written into a config file that persists across sessions. `--skip-test` is for deliberately pre-staging a config for a leader that isn't online yet — not for auth failures.

**Don't** disable TLS verification or ignore cert errors to "make it work." An invalid cert almost always means you're reaching the wrong host; fix the hostname instead.

### `maxim tunnel` — guided Cloudflare tunnel setup

Maxim ships a `maxim tunnel` subcommand that wraps cloudflared's CLI. Once
configured, leader-mode detection auto-fires on startup — no env vars needed.

**Actions:**

```bash
maxim tunnel              # show usage
maxim tunnel setup        # interactive setup (login, create, DNS, config.yml)
maxim tunnel status       # show what's currently configured
maxim tunnel start        # run cloudflared daemon in foreground (for testing)
```

**Prerequisites:**
1. `cloudflared` binary installed on the machine (system package, not pip). `maxim tunnel status` prints the right install command for your OS.
2. A domain on Cloudflare's nameservers — the tunnel needs a hostname (e.g., `maxim.yourdomain.com`).

**Guided setup flow:**

```bash
maxim tunnel setup
```

Walks you through:
1. Cloudflare authentication (opens browser)
2. Tunnel name (default `maxim-llm`)
3. DNS routing (you provide the hostname)
4. Local port (default `8099`, routes through LeaderProxy for auth + logging)
5. Writes `~/.cloudflared/config.yml`

After setup, keep the tunnel daemon running:

```bash
# Foreground (test it first)
cloudflared tunnel run maxim-llm

# Or install as a system service (starts on boot)
sudo cloudflared service install
```

Then on peers:

```bash
export MAXIM_LANE_LARGE_REMOTE_URL=https://maxim.yourdomain.com/v1
export MAXIM_MAX_CLOUD_LANES=1   # cloud-lane gate opt-in (public URL)
maxim
```

#### Peer authentication — API key

Once your tunnel is public, anyone who knows the hostname could hit your LLM.
Maxim generates a **256-bit API key** during `tunnel setup` and wires it into
the spawned llama-cpp-server via `--api_key`. Peers must present the key as a
Bearer token or get 401.

**Key file locations:**

| OS | Path |
|---|---|
| Linux / macOS / WSL | `~/.config/maxim/api_key` (or `$XDG_CONFIG_HOME/maxim/api_key`) |
| Windows | `%APPDATA%\maxim\api_key` |

POSIX files are created with mode 0600 (owner read/write only).

**Commands:**

```bash
maxim tunnel key show       # print the full key (for secure sharing)
maxim tunnel key rotate     # generate a new key (invalidates peers)
maxim tunnel key export     # copy-paste snippets for all shells
```

**Sharing with a peer — cross-platform snippets:**

`maxim tunnel key export` prints ready-to-paste snippets for **bash/zsh,
fish, PowerShell, and Windows cmd** — the peer picks theirs:

```bash
# bash / zsh (Linux, macOS, WSL)
export MAXIM_LANE_LARGE_REMOTE_API_KEY="…"
echo 'export MAXIM_LANE_LARGE_REMOTE_API_KEY="…"' >> ~/.bashrc

# fish
set -Ux MAXIM_LANE_LARGE_REMOTE_API_KEY "…"

# PowerShell
$env:MAXIM_LANE_LARGE_REMOTE_API_KEY = "…"
Add-Content $PROFILE "`n$env:MAXIM_LANE_LARGE_REMOTE_API_KEY = `"…`""

# Windows cmd
setx MAXIM_LANE_LARGE_REMOTE_API_KEY "…"
```

**Rotation:** `maxim tunnel key rotate` generates a new key. Peers using the old
key will get 401 until they update. Restart `maxim` on the leader after rotating
so the spawned server picks up the new key.

**Leader's own client:** auto-reads the key file — no shell config needed on
the leader machine.

**Solo mode (no tunnel):** localhost bind + no `MAXIM_ROLE=leader` = no auth
required. The API key is only enforced when the spawner runs in leader mode
(binds `0.0.0.0`).

The `maxim tunnel status` command shows what's configured at any time:

```
──────────────────────────────────────────────────────────────
  Maxim tunnel status
──────────────────────────────────────────────────────────────
  cloudflared: ✓ /usr/local/bin/cloudflared — cloudflared 2024.12.0
  config.yml:  ✓ /home/dennys/.cloudflared/config.yml
    tunnel: a1b2c3d4-...
    credentials_file: /home/dennys/.cloudflared/a1b2c3d4-...json
    hostname: maxim.example.com
    service: http://localhost:8099
  daemon:      ✓ running
──────────────────────────────────────────────────────────────
```

### Remote access via Cloudflare tunnel (Phase 5, optional)

**Optional.** Only needed if you want to reach your home server from
**outside your LAN** without VPN or port-forwarding. For same-network setups,
LAN direct IP or mDNS is enough.

One-time setup on the machine running the model server:

```bash
# Install cloudflared (Linux example)
sudo apt install cloudflared

# Authenticate
cloudflared tunnel login

# Create and route tunnel
cloudflared tunnel create maxim-llm
cloudflared tunnel route dns maxim-llm maxim-llm.yourdomain.com
```

Config file (`~/.cloudflared/config.yml`):

```yaml
tunnel: <tunnel-id-from-create>
credentials-file: ~/.cloudflared/<tunnel-id>.json
ingress:
  - hostname: maxim-llm.yourdomain.com
    service: http://localhost:8000
  - service: http_status:404
```

Run as a service (starts on boot):

```bash
sudo cloudflared service install
```

On the client, point a lane at the tunnel URL:

```bash
MAXIM_LANE_LARGE_REMOTE_URL=https://maxim-llm.yourdomain.com/v1 \
MAXIM_LANE_LARGE_REMOTE_MODEL=mistral-7b-instruct-v0.2 \
maxim
```

Because the tunnel URL is HTTPS + public DNS, it hits the cloud-lane gate:
raise `MAXIM_MAX_CLOUD_LANES=1` to allow it. Cloudflare's zero-trust policies
can handle auth if you need it (no Maxim-side API key required for tunnel access).

### Cloud providers (Anthropic, OpenAI)

Cloud providers need **three** things enabled explicitly:

1. `cloud_enabled: true` in `~/.maxim/config/llm.json` (or `MAXIM_LLM_CLOUD_ENABLED=1`)
2. `MAXIM_MAX_CLOUD_LANES=1` (or higher) — gate on the number of cloud lanes
3. API key in env — `ANTHROPIC_API_KEY` or `OPENAI_API_KEY`

Example `~/.maxim/config/llm.json` for a Claude cloud lane:

```json
{
  "cloud_enabled": true,
  "lane_models": {
    "large": {
      "remote_url": "https://api.anthropic.com/v1",
      "model": "claude-3-5-sonnet-20241022",
      "api_key_env": "ANTHROPIC_API_KEY"
    }
  }
}
```

Then:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export MAXIM_MAX_CLOUD_LANES=1
maxim
```

The session cost ceiling (`max_session_cost`, default $5.00 in the routing
policy) is enforced **per LLMRouter** — if you run multiple cloud lanes, each
has its own ceiling. Keep an eye on cost in sim reports.

## Python API

```python
from maxim.agents import LLMAgent, ChatLLMAgent

# Single-turn generation
agent = LLMAgent(profile="mistral-7b")
response = agent.generate("What is Python?")

# Multi-turn chat (maintains conversation history)
chat = ChatLLMAgent(profile="llama3-8b", temperature=0.7)
chat.generate("Hi! My name is Alex.")
response = chat.generate("What's my name?")

# Structured JSON output
result = agent.generate_json("Extract name and age from: 'John is 25'")
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Model not found | Run `maxim --list-models` to see download status |
| Out of memory | Use a smaller model or lower quantization level |
| Slow inference | Use smaller model (`smollm-1.7b`) or lower quantization (`Q3_K_M`) |
| Gibberish output | Check that `prompt_style` matches the model family in `~/.maxim/config/llm.json` |
