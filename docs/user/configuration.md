# Configuration

## Overview

Maxim is configured through three mechanisms: CLI flags, environment variables, and JSON config files. CLI flags override environment variables, which override config file defaults.

## Public env var contract (CC4)

The variables below are **public**: removal or rename is a breaking change at a major-version bump (1.x → 2.0). Behavior may evolve (smarter defaults, better validation) but the names and the contract these variables provide will not.

Environment variables not on this list are **debug / experimental** — see the [Debug / experimental env vars](#debug--experimental-env-vars-may-change-without-notice) section. They may change without notice in any minor release.

### Public — LLM + model selection

| Variable | Description | Default |
|----------|-------------|---------|
| `MAXIM_LLM_ENABLED` | Enable LLM inference (1/true). | 0 |
| `MAXIM_LLM_PROFILE` | Model profile name. | None |
| `MAXIM_LLM_QUANTIZATION` | Quantization level (Q3_K_M, Q4_K_M, Q5_K_M, Q8_0). | Q4_K_M |
| `MAXIM_LLM_N_CTX` | Override auto-computed llama.cpp context window. Same as `--llm-n-ctx`. | (formula) |
| `MAXIM_AUTO_DOWNLOAD_MODELS` | Set to `1` to auto-download missing GGUFs. Same as `--auto-download`. | off |
| `MAXIM_DATA_BUDGET_GB` | Soft cap on `~/.maxim/` disk usage. Auto-download preflight refuses if it would exceed the cap. | (unset) |
| `MAXIM_DATA_HOME` | Override the base data directory (default `~/.maxim`). | `~/.maxim` |
| `MAXIM_LLM_CALL_TIMEOUT_S` | LLMWorker agent-level call timeout (clamped 10-1800). | 300 |
| `MAXIM_PROVENANCE_VERBOSITY` | Provenance tracing (0=off, 1=compact, 2=verbose). | 0 |
| `MAXIM_LOG_FILE` | Path to JSONL log file. Dual-format: stdout stays human-readable, file is machine-parseable. | (unset) |

### Public — peer / leader / role

| Variable | Description | Default |
|----------|-------------|---------|
| `MAXIM_ROLE` | Explicit role: `leader`, `peer`, or `solo`. Set automatically by `cli.py::main` at startup. | (auto) |
| `MAXIM_LANE_{TIER}_REMOTE_URL` | Override the named tier to use a remote peer/leader URL. `{TIER}` is one of `LARGE`, `MEDIUM`, `SMALL`. | (unset) |
| `MAXIM_LANE_{TIER}_REMOTE_MODEL` | Model name to request from the remote server for the named tier. | (unset) |
| `MAXIM_LANE_{TIER}_REMOTE_API_KEY` | Auth token for the remote server for the named tier. | (unset) |

### Public — cloud providers

| Variable | Description | Default |
|----------|-------------|---------|
| `ANTHROPIC_API_KEY` | Required for Claude backend. | (unset) |
| `OPENAI_API_KEY` | Required for OpenAI backend. | (unset) |
| `GOOGLE_API_KEY` | Required for Gemini backend. | (unset) |
| `GROQ_API_KEY` | Required for Groq backend. | (unset) |
| `TOGETHER_API_KEY` | Required for Together backend. | (unset) |
| `FIREWORKS_API_KEY` | Required for Fireworks backend. | (unset) |
| `MISTRAL_API_KEY` | Required for Mistral API backend. | (unset) |
| `DEEPSEEK_API_KEY` | Required for DeepSeek backend. | (unset) |
| `MAXIM_LLM_CLOUD_ENABLED` | Enable cloud dispatch (required for `--cloud-*` flags). | 0 |
| `MAXIM_MAX_CLOUD_LANES` | Max lanes using cloud providers. | 0 |
| `MAXIM_LLM_REDACTION_POLICY` | Redaction policy for cloud dispatch (standard/relaxed/strict). | standard |
| `MAXIM_CLOUD_SESSION_BUDGET` | Hard ceiling on cloud spending per session (USD). | 5.00 |

### Public — embodiment + hardware

| Variable | Description | Default |
|----------|-------------|---------|
| `MAXIM_ROBOT_NAME` | Robot identifier (Reachy daemon `robot_name` / zenoh namespace). | reachy_mini |
| `MAXIM_COMMS_ENABLED` | Enable SMS/Voice (1/true). | 0 |
| `MAXIM_WHISPER_COMPUTE_TYPE` | Whisper compute type (int8/float16/float32). | int8 |
| `MAXIM_DISABLE_IMSHOW` | Disable OpenCV window display. | 0 |
| `TWILIO_ACCOUNT_SID` | Twilio Account SID. | None |
| `TWILIO_AUTH_TOKEN` | Twilio Auth Token. | None |
| `TWILIO_FROM_NUMBER` | Twilio phone number. | None |
| `CUDA_VISIBLE_DEVICES` | GPU selection (empty string = CPU only). | auto |

## Debug / experimental env vars (may change without notice)

These variables are **debug / experimental**: useful for diagnostics or workarounds, but their names, default values, and behavior may change in any minor release. **Do not depend on them in scripts or shell aliases that need to survive Maxim upgrades.**

### Debug — tracing + logging

| Variable | Description | Default |
|----------|-------------|---------|
| `MAXIM_HTTP_TRACE` | Bumps `http_request` events from DEBUG to INFO (every outbound call logged). | 0 |
| `MAXIM_BACKEND_TRACE` | Bumps `_MaximPeerBackend` `peer_backend_call` events from DEBUG to INFO. Pair with `MAXIM_LOG_FILE` for per-call JSONL. | 0 |
| `MAXIM_HEARTBEAT` | System health heartbeat every 10s (GPU/CPU/RAM/disk/WiFi + stall detection). | 0 |
| `MAXIM_HEARTBEAT_INTERVAL_S` | Heartbeat sample interval. | 10 |
| `MAXIM_HEARTBEAT_STALL_S` | Warn after this many seconds with no LLM calls. | 30 |
| `MAXIM_LANE_TRACE` | Per-request LLM trace logs (also enables heartbeat). | 0 |
| `MAXIM_PEER_LOG_REQUESTS` | JSON log per outbound peer call. | 0 |
| `MAXIM_HIPPO_TRACE` / `MAXIM_NAC_TRACE` / `MAXIM_ATL_TRACE` / `MAXIM_EC_TRACE` / `MAXIM_SCN_TRACE` / `MAXIM_PAIN_TRACE` / `MAXIM_FEAR_TRACE` / `MAXIM_DEFAULT_NET_TRACE` | Enable bio-subsystem traces. Set by the `--trace` CLI flag. | 0 |

### Debug — substrate + decision-system experiments

| Variable | Description | Default |
|----------|-------------|---------|
| `MAXIM_SUBSTRATE_PATH` | Enable substrate encoding path (LinguisticEncoder → EC → ATL dual-write). | 0 |
| `MAXIM_CONCEPT_DECOMPOSITION` | Enable concept decomposition (noun-phrase extraction before EC). Requires spaCy + en_core_web_sm. | 0 |
| `MAXIM_NAC_TEMPORAL_CREDIT_WEIGHT` | Temporal credit weight for SCN-substrate eligibility traces. | 0.3 |
| `MAXIM_AUTO_SPAWN_N_CTX` | Legacy alias for `MAXIM_LLM_N_CTX`. Kept for in-place upgrades. | (unset) |

### Debug — peer/probe internals

| Variable | Description | Default |
|----------|-------------|---------|
| `MAXIM_SKIP_REMOTE_PROBE` | Bypass the remote-URL probe. CI/test escape hatch. | 0 |
| `MAXIM_REMOTE_PROBE_FIRST_TIMEOUT_S` | First-attempt probe timeout (clamped 0.2-5.0). | 0.8 |
| `MAXIM_REMOTE_PROBE_RETRY_TIMEOUT_S` | Retry probe timeout (clamped 0.5-10.0). | 2.5 |
| `MAXIM_REMOTE_PROBE_CACHE_TTL_S` | Probe cache freshness window (clamped 0-600). | 60 |
| `MAXIM_DRAIN_CACHE_TTL_S` | DrainConstraint mtime cache freshness (clamped 0-60). | 1.0 |
| `MAXIM_AUTO_DRAIN_THRESHOLD` | Transient failure count before auto-drain (clamped 2-20). | 5 |
| `MAXIM_AUTO_UNDRAIN_PROBE_INTERVAL_S` | Auto-undrain probe cycle interval (clamped 30-600). | 90 |
| `MAXIM_PROXY_MAX_CONCURRENT` | Max in-flight requests to upstream (0 = unlimited). | 4 |
| `MAXIM_PROXY_RATE_LIMIT_RPM` | Per-peer requests/minute (0 = unlimited). | 0 |

### Debug — embodiment

| Variable | Description | Default |
|----------|-------------|---------|
| `MAXIM_DEEP_EMBODIMENT` | Enable level-3 deep embodiment: sub-sensor exposure + per-sub-sensor damage routing. Same as `--deep-embodiment`. Level-3 semantics are post-1.0 work; the toggle is debug-only until the broader feature stabilises. | 0 |

### Debug — sim safety + deprecated

| Variable | Description | Default |
|----------|-------------|---------|
| `MAXIM_REAP_ORPHANS` | Kill stale `maxim sim` processes detected at startup. | 0 |
| `MAXIM_SHOW_CHANNELS` | Channel filter for simulation output (legacy — `--display` is preferred). | (unset) |
| `MAXIM_PROMPT_PROFILE` | Prompt optimization (deprecated — not read by current code). | standard |

## CLI flag stability (CC4)

CLI flags are **stable** unless their `--help` text carries the `[experimental]` suffix. Stable flags' names and default values are part of the contract; experimental flags may be renamed, retyped, or removed in any minor release.

Currently flagged as `[experimental]`:

- `--research` — research protocol; matches the experimental status of `maxim.research()`
- `--deep-embodiment` — level-3 deep embodiment; matches `MAXIM_DEEP_EMBODIMENT` debug status
- `--auto-curate`, `--curate-threshold`, `--no-curate` — pre-sim auto-curation surface (E3, late 0.7)
- `--foundry`, `--foundry-count`, `--foundry-genre`, `--foundry-category`, `--foundry-dry-run` — Asset Foundry surface
- `--reap-orphans` — sim safety net; behavior may evolve
- `--audit-architecture` — internal audit verb
- `--generate-simulation` — scenario generation utility

## Token telemetry contract (CC12)

Per-call LLM token telemetry is exposed under these field names — frozen at 1.0:

| Field | Meaning |
|---|---|
| `input_tokens` | Total prompt tokens (cached + uncached). |
| `output_tokens` | Generated tokens. |
| `cached_tokens` | Cached portion of the input. Read from prompt cache, charged at the cached rate (or free, depending on provider). |

Where these fields appear:

| Surface | Contract |
|---|---|
| `LLMResponse.input_tokens` / `.output_tokens` / `.cached_tokens` | `cached_tokens` is a property alias for the legacy `cached_input_tokens` field. |
| `LLMRouter.generate(...)` `usage` dict | All three fields present. `cached_input_tokens` retained as legacy alias. |
| JSONL events `peer_backend_call`, `peer_stream_complete` | Emitted under `MAXIM_LOG_FILE`. All three fields present. |
| Leader proxy per-request log entry | `cached_tokens` parsed from upstream `usage.prompt_tokens_details.cached_tokens` when present. |
| `CostTracker.get_session_tokens()` | Exposes `input_tokens`, `output_tokens`, `cached_tokens`, `total_tokens`. |

Legacy field names — `cached_input_tokens`, `uncached_input_tokens` (Maxim-internal cost-calculation detail), `prompt_tokens`, `completion_tokens` (OpenAI/llama-cpp wire-format compatibility) — are kept as **permanent legacy aliases**. Removing them is a major-version-bump change. **External callers should prefer the standard names** (`input_tokens`, `output_tokens`, `cached_tokens`) — those are the only token field names this page commits to.

## Data Directory

All runtime data lives under `~/.maxim/` by default. Override the base path by setting the `MAXIM_DATA_HOME` environment variable:

```bash
export MAXIM_DATA_HOME=/path/to/custom/maxim-data
```

When set, all subdirectories (`config/`, `util/`, `memory/`, `models/`, `sim_reports/`, `benchmarks/`, etc.) are resolved relative to `$MAXIM_DATA_HOME` instead of `~/.maxim/`.

## Config Files

User-modifiable config files live under `~/.maxim/`:

### ~/.maxim/config/llm.json -- LLM Configuration

Controls which model runs, how it behaves per mode, token limits.

Key fields:
- `enabled` (bool) -- master switch
- `profile` (str) -- active model profile name
- `max_tokens` (int) -- max response tokens (default: 512)
- `temperature` (float) -- sampling temperature (default: 0.0 = deterministic)
- `quantization` (str) -- weight quantization level
- `profiles` (dict) -- model definitions with backend, model_path, prompt_style, stop tokens, n_ctx
- `mode_response_config` (dict) -- per-mode token budgets and response formats

Three profiles ship by default:
- phi-3-mini-4k-instruct (ChatML, 4096 ctx)
- mistral-7b-instruct-v0.2 (Mistral instruct, 8192 ctx)
- smollm-1.7b-instruct (ChatML, 2048 ctx)

### ~/.maxim/util/whisper.json -- Audio Transcription

Controls Whisper model, device, VAD settings.

Key fields:
- `model` -- Whisper model size (tiny through large-v3, distil-large-v3 recommended)
- `device` -- auto, cpu, or cuda
- `compute_type` -- int8 (fast), float16 (GPU), float32 (compatible)
- `language` -- language code or "auto"
- `vad_filter` -- enable voice activity detection
- `vad_threshold` -- 0.0-1.0, lower = more sensitive (default: 0.25)

### ~/.maxim/util/phrase_responses.json -- Voice Commands

Maps spoken phrases to actions. Format:
```json
{
  "maxim shutdown": { "call": "request_shutdown", "cooldown_s": 2.0 },
  "maxim sleep": { "call": "request_sleep", "cooldown_s": 2.0 },
  "maxim": { "call": "wake_up_agentic", "wake_word": true, "cooldown_s": 2.0 }
}
```
Users can add custom voice commands by adding entries.

### ~/.maxim/util/key_responses.json -- Keyboard Shortcuts

Maps key presses to actions:
- `c` -- center vision
- `u` -- mark trainable moment
- `0-9` -- label outcome (for training mode)

## Cloud Provider Profiles

Maxim ships with built-in profiles for 8 cloud LLM providers (Anthropic, OpenAI, Google Gemini, Groq, Together, Fireworks, Mistral, DeepSeek) across 15 cloud profiles. Most use the OpenAI-compatible backend (so no extra dependencies are needed beyond `pip install -e ".[llm-openai]"`); Anthropic uses the native SDK via `pip install -e ".[llm-anthropic]"`. Set the corresponding API key environment variable to enable a profile.

| Profile | Provider | Model | API Key Env Var |
|---------|----------|-------|-----------------|
| `claude-sonnet` | Anthropic | claude-sonnet-4-20250514 | `ANTHROPIC_API_KEY` |
| `claude-haiku` | Anthropic | claude-haiku | `ANTHROPIC_API_KEY` |
| `gpt-4o` | OpenAI | gpt-4o | `OPENAI_API_KEY` |
| `gemini-2.5-flash` | Google | gemini-2.5-flash-preview-05-20 | `GOOGLE_API_KEY` |
| `gemini-2.5-pro` | Google | gemini-2.5-pro-preview-05-06 | `GOOGLE_API_KEY` |
| `groq-llama3-70b` | Groq | llama-3.3-70b-versatile | `GROQ_API_KEY` |
| `groq-mixtral` | Groq | mixtral-8x7b-32768 | `GROQ_API_KEY` |
| `together-llama3-70b` | Together | Llama-3.3-70B-Instruct-Turbo | `TOGETHER_API_KEY` |
| `fireworks-llama3-70b` | Fireworks | llama-v3p3-70b-instruct | `FIREWORKS_API_KEY` |
| `mistral-large` | Mistral | mistral-large-latest | `MISTRAL_API_KEY` |
| `mistral-small` | Mistral | mistral-small-latest | `MISTRAL_API_KEY` |
| `deepseek-chat` | DeepSeek | deepseek-chat | `DEEPSEEK_API_KEY` |
| `deepseek-reasoner` | DeepSeek | deepseek-reasoner | `DEEPSEEK_API_KEY` |

Use any profile with `--language-model`:

```bash
maxim --sim "test safety" --language-model gemini-2.5-flash
maxim --sim "test memory" --language-model groq-llama3-70b
```

Cloud providers can also be used as fallback or dedicated lane backends:

```bash
maxim --cloud-fallback claude-sonnet         # Fallback when self-hosted fails
maxim --cloud-lane small gemini-2.5-flash    # Dedicated cloud model for small tier
maxim --cloud-budget 2.00                    # Max session cost for cloud providers
```

Cloud dispatch requires `MAXIM_LLM_CLOUD_ENABLED=1`. See the environment variables table for related settings.

## Auto-Generated Files (Do Not Edit)

- `~/.maxim/util/adaptive_thresholds.json` -- auto-tuned novelty/salience thresholds
- `~/.maxim/util/focus_learner.json` -- motor gain learning state
- `~/.maxim/util/learned_bounds.json` -- workspace safety bounds
- `~/.maxim/util/cost_state.json` -- resource usage tracking

## Directory Structure

```
~/.maxim/
├── config/         -- LLM config (llm.json)
├── util/           -- Runtime config files (whisper.json, phrase_responses.json, etc.)
├── memory/         -- Episodic memories (persistent)
├── models/
│   ├── LLM/        -- Downloaded GGUF model files
│   ├── tts/        -- Text-to-speech models
│   └── YOLO/       -- YOLO vision models
├── sim_reports/    -- Simulation session reports
├── benchmarks/     -- Benchmark output reports
├── audio/          -- WAV recordings
├── videos/         -- MP4 recordings
├── transcript/     -- JSONL transcripts with timestamps
├── logs/           -- Run logs
└── plans/
    ├── checkpoints/ -- Goal tree snapshots
    └── exports/     -- Exported plan files
```
