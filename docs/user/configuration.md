# Configuration

## Overview

Maxim is configured through three mechanisms: CLI flags, environment variables, and JSON config files. CLI flags override environment variables, which override config file defaults.

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MAXIM_LLM_ENABLED` | Enable LLM inference (1/true) | 0 |
| `MAXIM_LLM_PROFILE` | Model profile name | None |
| `MAXIM_LLM_QUANTIZATION` | Quantization level (Q3_K_M, Q4_K_M, Q5_K_M, Q8_0) | Q4_K_M |
| `MAXIM_LLM_N_CTX` | Override auto-computed llama.cpp context window (P4c). Same as `--llm-n-ctx`. | (formula) |
| `MAXIM_AUTO_SPAWN_N_CTX` | Legacy alias for `MAXIM_LLM_N_CTX`. Kept for in-place upgrades. | (unset) |
| `MAXIM_AUTO_DOWNLOAD_MODELS` | Set to `1` to skip the interactive download prompt and auto-download missing GGUFs (P5). Same as `--auto-download`. | off |
| `MAXIM_DATA_BUDGET_GB` | Optional soft cap on `~/.maxim/` disk usage. The auto-download preflight refuses if the new download would exceed it. | (unset) |
| `MAXIM_SKIP_REMOTE_PROBE` | Set to `1` to bypass the P6 remote-URL probe. CI/test escape hatch. | off |
| `MAXIM_REMOTE_PROBE_FIRST_TIMEOUT_S` | First-attempt probe timeout (clamped 0.2-5.0). | 0.8 |
| `MAXIM_REMOTE_PROBE_RETRY_TIMEOUT_S` | Retry probe timeout (clamped 0.5-10.0). | 2.5 |
| `MAXIM_REMOTE_PROBE_CACHE_TTL_S` | Probe cache freshness window (clamped 0-600). | 60 |
| `MAXIM_PROMPT_PROFILE` | Prompt optimization (deprecated — not read by current code; use per-mode config in llm.json) | standard |
| `MAXIM_ROBOT_NAME` | Robot identifier | reachy_mini |
| `MAXIM_COMMS_ENABLED` | Enable SMS/Voice (1/true) | 0 |
| `MAXIM_WHISPER_COMPUTE_TYPE` | Whisper compute type (int8/float16/float32) | int8 |
| `MAXIM_DISABLE_IMSHOW` | Disable OpenCV window display | 0 |
| `MAXIM_PROVENANCE_VERBOSITY` | Provenance tracing (0=off, 1=compact, 2=verbose) | 0 |
| `TWILIO_ACCOUNT_SID` | Twilio Account SID (for comms) | None |
| `TWILIO_AUTH_TOKEN` | Twilio Auth Token (for comms) | None |
| `TWILIO_FROM_NUMBER` | Twilio phone number (for comms) | None |
| `CUDA_VISIBLE_DEVICES` | GPU selection (empty string = CPU only) | auto |

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
