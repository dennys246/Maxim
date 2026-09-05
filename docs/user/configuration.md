# Configuration

## Quick start: `maxim config`

> **As of 1.0, the canonical way to configure a Maxim instance is the `maxim config` CLI verbs + `~/.config/maxim/config.json`.**
> Environment variables still work as a per-session override, but they are no longer the recommended primary surface.

```bash
maxim config get                     # show every effective field + source
maxim config get llm.profile         # show one field with source marker
maxim config set role leader         # write to ~/.config/maxim/config.json
maxim config set llm.profile qwen2.5-32b-instruct
maxim config set lanes.large.remote_url https://leader.example.com/v1
maxim config edit                    # open $EDITOR on the file
maxim config path                    # print the resolved file path
```

The `maxim doctor` command shows a **Resolved Config** section that surfaces every absorbed field with its source — the single answer to "what does this instance think it's configured as?"

### Precedence chain

For every absorbed field:

**CLI args > env vars > `~/.config/maxim/config.json` > builtin defaults**

This mirrors `kubeconfig`, `gh`, `npm`, and `pyproject.toml`. Mismatches between layers are logged at WARNING (env shadows config with a different value), and convergence is logged at INFO (env and config agree — operator's two-sources-of-truth confusion class). Run `maxim doctor` to see the resolved value + source for every absorbed field in one place.

**Empty-string env vars are treated as unset.** `export MAXIM_LANE_LARGE_REMOTE_URL=` (a common bash-rc leak) falls through to `config.json` per the C-1 fold.

### Absorbed fields (~25)

| Field path | Type | Default | Replaces env var |
|---|---|---|---|
| `role` | leader / peer / solo | (computed) | `MAXIM_ROLE` |
| `llm.enabled` | bool | true | `MAXIM_LLM_ENABLED` |
| `llm.profile` | string | none | `MAXIM_LLM_PROFILE` |
| `llm.n_ctx` | int ≥ 256 | 8192 | `MAXIM_LLM_N_CTX` |
| `llm.backend` | llama_cpp / pytorch | llama_cpp | `MAXIM_LLM_BACKEND` |
| `llm.auto_download` | bool | false | `MAXIM_AUTO_DOWNLOAD_MODELS` |
| `llm.max_response_tokens` | int ≥ 1 \| null | null (the mode's own reserve; 512 in the agent loop) | `MAXIM_LLM_MAX_RESPONSE_TOKENS` |
| `llm.deliberation_max_cycles` | int ≥ 1 \| null | null (3 in sim, 2 live) | `MAXIM_LLM_DELIBERATION_MAX_CYCLES` |

`llm.max_response_tokens` is the agent loop's per-call `max_tokens` and the prompt budgeter's response reserve in one field (a value at or above `llm.n_ctx` collapses the prompt budget and logs a WARNING); it does not touch the router's direct completion calls, which read the legacy `MAXIM_LLM_MAX_TOKENS`. `llm.deliberation_max_cycles` caps the PFC deliberation cycles per turn (`1` = one LLM call per turn). Both are read when an agent loop starts, so `maxim serve` needs a restart to pick up a change — the same holds for `tools.allow` / `tools.deny`.
| `lanes.<tier>.remote_url` | string \| null | null | `MAXIM_LANE_<TIER>_REMOTE_URL` |
| `lanes.<tier>.remote_model` | string \| null | null | `MAXIM_LANE_<TIER>_REMOTE_MODEL` |
| `lanes.<tier>.remote_api_key_ref` | path or `keyring:<service>:<account>` | null | `MAXIM_LANE_<TIER>_REMOTE_API_KEY` |
| `lanes.<tier>.timeout_s` | float > 0 \| null | null (backend default) | `MAXIM_LANE_<TIER>_TIMEOUT_S` |
| `cloud.enabled` | bool | false | `MAXIM_LLM_CLOUD_ENABLED` |
| `cloud.max_lanes` | int ≥ 0 | 0 | `MAXIM_MAX_CLOUD_LANES` |
| `cloud.fallback_model` | string \| null | null | `MAXIM_CLOUD_FALLBACK_MODEL` |
| `cloud.session_budget_usd` | float ≥ 0 | 5.0 | `MAXIM_CLOUD_SESSION_BUDGET` |
| `cloud.redaction_policy` | standard / relaxed / strict | standard | `MAXIM_LLM_REDACTION_POLICY` |
| `proxy.max_concurrent` | int ≥ 0 | 4 | `MAXIM_PROXY_MAX_CONCURRENT` |
| `proxy.rate_limit_rpm` | int ≥ 0 | 0 | `MAXIM_PROXY_RATE_LIMIT_RPM` |
| `auto_spawn.llm_server` | bool | true | `MAXIM_AUTO_SPAWN_LLM_SERVER` |
| `auto_spawn.tunnel` | bool | true | `MAXIM_AUTO_SPAWN_TUNNEL` |
| `auto_spawn.port` | int 1..65535 | 8100 | `MAXIM_AUTO_SPAWN_PORT` |
| `auto_spawn.timeout_s` | int ≥ 1 | 120 | `MAXIM_AUTO_SPAWN_TIMEOUT_S` |
| `data.home` | string \| null | null | `MAXIM_DATA_HOME` |
| `data.budget_gb` | float ≥ 0 \| null | null | `MAXIM_DATA_BUDGET_GB` |

Tier names (`large`, `medium`, `small`) are FROZEN at 1.0 per the lane-tier-names invariant.

### API key references — file path or keyring URI only

`lanes.<tier>.remote_api_key_ref` accepts **two forms**:

- **File path** (starts with `/` or `~`): the file is read at lane backend construction time. Must be mode 0600. The canonical leader-key file lives at `~/.config/maxim/api_key`.
- **Keyring URI** (`keyring:<service>:<account>`): resolved via the system keychain (macOS Keychain, Linux Secret Service). Requires `pip install keyring`.

**Inline plaintext keys are rejected** at `config.json` load time. The cross-confirmed cross-confirmed I-3/IM3 fold from the pre-implementation review: `maxim config set lanes.large.remote_api_key_ref sk-abc123` would cheerfully write mode-0644 plaintext keys to disk. Use a file-path reference instead:

```bash
echo "$LEADER_API_KEY" > ~/.config/maxim/api_key
chmod 0600 ~/.config/maxim/api_key
maxim config set lanes.large.remote_api_key_ref ~/.config/maxim/api_key
```

The legacy `MAXIM_LANE_<TIER>_REMOTE_API_KEY` env var still works (it directly holds the inline key — that's the pre-1.0 semantics). `maxim doctor` flags it with a migration fix-hint.

### What `config.json` does NOT replace

- **`~/.config/maxim/peer.yml`** — kept as a back-compat reader through 1.x. New `peer connect` invocations dual-write both files. Retired in 2.0.
- **`~/.config/maxim/mesh.yml`** — multi-node topology (Plan 4). Per-cluster, not per-instance.
- **`~/.config/maxim/profiles.yml`** — custom profile catalog. Hand-edit-canonical.
- **`~/.cloudflared/config.{yml,yaml}`** — cloudflared's own config, not Maxim's. Detected as a leader signal (extension widened to accept either form).

The four declarative-config files coexist by design.

### Auto-migration from peer.yml

On first startup when `config.json` is absent **AND** `peer.yml` is present **AND** `~/.cloudflared/config.{yml,yaml}` is absent (preserves the legitimate leader-case), the loader auto-writes a minimal `config.json` populated from peer.yml fields. peer.yml is left in place — never deleted by the shim. Subsequent startups read `config.json` directly. The migration logs INFO once.

If cloudflared is present (i.e., this machine is provisioned as a tunneled leader), the migration is skipped so a stale peer.yml from a previous peer setup doesn't silently flip the role to peer. This was Mac Mini Trigger #3 from the 2026-06-01 incident that motivated this entire plan.

---

## Overview

Maxim is configured through three mechanisms: CLI flags, environment variables, and JSON config files. CLI flags override environment variables, which override config file defaults.

## Public env var contract (CC4)

The variables below are **public**: removal or rename is a breaking change at a major-version bump (1.x → 2.0). Behavior may evolve (smarter defaults, better validation) but the names and the contract these variables provide will not.

Environment variables not on this list are **debug / experimental** — see the [Debug / experimental env vars](#debug--experimental-env-vars-may-change-without-notice) section. They may change without notice in any minor release.

### Public — LLM + model selection

| Variable | Description | Default |
|----------|-------------|---------|
| `MAXIM_LLM_ENABLED` | Enable LLM inference (1/true). | 1 |
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
| `MAXIM_NAC_MIN_CONFIDENCE` | Minimum confidence threshold for `propose_via_substrate` (substrate-primary action selection). Set to `0.0` to bypass the cold-start gate entirely. Invalid values fall back to default with a WARNING. | 0.3 |
| `MAXIM_NAC_CLUSTER_REWARD_BIAS_DECAY_TAU` | Decay timescale (ticks) for the Wire-A cluster-reward bias. Higher values make learned substrate-voice annotations persist across more turns. Clamped 50–1000. | 300.0 |
| `MAXIM_NAC_REWARD_BIAS_DISABLED` | Gates NAc reward-bias surfaces (`distribute_reward`, `decay_reward_biases`, `get_agent_tool_biases`) as no-ops. Truthy values: `1`, `true`, `yes`, `on` (case-insensitive). Eligibility traces and causal links are unaffected. Read once at NAc construction. | off |
| `MAXIM_EC_TRACE_ACTIVATIONS` | Emit per-tick `sim_ec_activation` JSONL events from `EntorhinalCortex.pattern_complete_or_separate`. Truthy values: `1`, `true`, `yes`, `on`. Used for co-activation analysis (e.g. `scripts/analyze_roy_4_coactivation.py`). | off |
| `MAXIM_AUTO_SPAWN_N_CTX` | Legacy alias for `MAXIM_LLM_N_CTX`. Kept for in-place upgrades. | (unset) |

### Debug — peer/probe internals

| Variable | Description | Default |
|----------|-------------|---------|
| `MAXIM_SKIP_REMOTE_PROBE` | Bypass the remote-URL probe. CI/test escape hatch. | 0 |
| `MAXIM_REMOTE_PROBE_FIRST_TIMEOUT_S` | First-attempt probe timeout (clamped 0.2-5.0). | 1.5 |
| `MAXIM_REMOTE_PROBE_RETRY_TIMEOUT_S` | Retry probe timeout (clamped 0.5-10.0). | 8.0 |
| `MAXIM_REMOTE_PROBE_CACHE_TTL_S` | Probe cache freshness window (clamped 0-600). | 60 |
| `MAXIM_DRAIN_CACHE_TTL_S` | DrainConstraint mtime cache freshness (clamped 0-60). | 1.0 |
| `MAXIM_AUTO_DRAIN_THRESHOLD` | Transient failure count before auto-drain (clamped 2-20). | 5 |
| `MAXIM_AUTO_UNDRAIN_PROBE_INTERVAL_S` | Auto-undrain probe cycle interval (clamped 30-600). | 90 |
| `MAXIM_LEADER_PROXY_PORT` | Port the leader proxy listens on. Peers point `MAXIM_LANE_LARGE_REMOTE_URL` at this port. | 8099 |
| `MAXIM_PROXY_MAX_CONCURRENT` | Max in-flight requests to upstream (0 = unlimited). | 4 |
| `MAXIM_PROXY_RATE_LIMIT_RPM` | Per-peer requests/minute (0 = unlimited). | 0 |
| `MAXIM_PROXY_KEEPALIVE_INTERVAL_S` | SSE keepalive cadence (seconds) during TTFT on streaming responses (clamped 5-90). Prevents cloudflared's ~100s idle timeout from closing the connection on slow 30B+ models. | 30 |
| `MAXIM_PROXY_CONTEXT_ADMISSION` | Enable the proxy-side context-overflow admission gate (rejects requests whose estimated prompt exceeds `MAXIM_LLM_N_CTX`). `0`/`false`/`no`/`off` to disable. | on when `MAXIM_LLM_N_CTX` resolvable |
| `MAXIM_PROXY_CONTEXT_OVERHEAD_TOKENS` | Safety margin (tokens) for the char-based token estimator in the admission gate (clamped 0-4096). | 256 |

### Debug — context pool

The context pool (`agents/context_pool.py`) manages the rolling LLM context window, automatically summarizing when the pool exceeds a configured token budget.

| Variable | Description | Default |
|----------|-------------|---------|
| `MAXIM_CONTEXT_POOL_MAX_TOKENS` | Maximum token budget before the pool triggers summarization. | 2000 |
| `MAXIM_CONTEXT_POOL_SUMMARY_TOKENS` | Target token count for each generated summary entry. | 500 |
| `MAXIM_CONTEXT_POOL_MAX_ENTRIES` | Maximum number of entries before forced summarization. | 50 |
| `MAXIM_CONTEXT_POOL_KEEP_RECENT` | Number of most-recent entries always kept unsummarized. | 5 |
| `MAXIM_CONTEXT_POOL_AGENT_STATES` | Include agent bio-state snapshots in the pool. `true`/`false`. | true |
| `MAXIM_CONTEXT_POOL_OUTCOMES` | Include tool outcome entries in the pool. `true`/`false`. | true |
| `MAXIM_CONTEXT_POOL_ABSTRACTIONS` | Include the abstraction stream in the pool. `true`/`false`. | true |
| `MAXIM_CONTEXT_POOL_PATH` | File path for pool persistence across restarts. Unset = no persistence. | (unset) |

### Debug — bio-system feature gates

These three switches disable specific LLM-context annotation sections injected by the substrate pipeline. They accept truthy values `1`, `true`, `yes`, `on` (case-insensitive) to disable the feature.

| Variable | Description | Default |
|----------|-------------|---------|
| `MAXIM_DISABLE_CLUSTER_BIAS_ANNOTATION` | Disable Wire-A's cluster-bias annotation section in the LLM prompt (NAc learned substrate voice). Off = annotation active. | off |
| `MAXIM_DISABLE_VARIANCE_ANNOTATION` | Disable Wire 1's variance-band felt-sensation annotation on tool descriptions. Off = annotation active. | off |
| `MAXIM_DISABLE_IMAGINATION_SUBSTRATE_SIGNAL` | Disable W2's substrate-aware scene-manifest enrichment (NAc tool biases fed into the imagination scene manifest). Off = enrichment active. | off |

These are operator toggles as well as research ablation arms — setting one lets you measure the behavioral contribution of each annotation layer independently.

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

### ~/.config/maxim/config.json — Primary operator config (1.0+)

As of 1.0, the canonical operator-config file is `~/.config/maxim/config.json`. It absorbs the ~23 fields listed in the [Absorbed fields](#absorbed-fields-23) table above. Use the `maxim config` CLI to read and write it:

```bash
maxim config get                        # show all effective fields + source
maxim config get llm.profile            # get one field
maxim config set llm.profile qwen2.5-32b-instruct
maxim config set lanes.large.remote_url https://leader.example.com/v1
```

The legacy `~/.maxim/config/llm.json` file was the pre-0.9 LLM config store. Its role has been absorbed by `config.json`. If you have a hand-edited `llm.json`, migrate the relevant fields via `maxim config set` and remove the old file — it is no longer read by the current runtime.

> **Checking what the runtime actually sees:** `maxim config get` shows each field annotated with its source (`cli`, `env`, `config.json`, or `default`). `maxim doctor` shows the full Resolved Config section and flags any env/config mismatches at WARNING.

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
~/.config/maxim/    -- Primary operator config (1.0+)
├── config.json     -- Absorbed fields (llm, lanes, cloud, proxy, auto_spawn, data)
├── api_key         -- Leader API key file (mode 0600, referenced by remote_api_key_ref)
├── peer.yml        -- Back-compat peer config (read-only; new setups write config.json)
├── mesh.yml        -- Multi-node topology (Plan 4)
└── profiles.yml    -- Custom LLM profile catalog

~/.maxim/
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
