# Configuration

## Overview

Maxim is configured through three mechanisms: CLI flags, environment variables, and JSON config files. CLI flags override environment variables, which override config file defaults.

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MAXIM_LLM_ENABLED` | Enable LLM inference (1/true) | 0 |
| `MAXIM_LLM_PROFILE` | Model profile name | None |
| `MAXIM_LLM_QUANTIZATION` | Quantization level (Q3_K_M, Q4_K_M, Q5_K_M, Q8_0) | Q4_K_M |
| `MAXIM_PROMPT_PROFILE` | Prompt optimization (legacy; per-mode config in llm.json is preferred) | standard |
| `MAXIM_ROBOT_NAME` | Robot identifier | reachy_mini |
| `MAXIM_COMMS_ENABLED` | Enable SMS/Voice (1/true) | 0 |
| `MAXIM_WHISPER_COMPUTE_TYPE` | Whisper compute type (int8/float16/float32) | int8 |
| `MAXIM_DISABLE_IMSHOW` | Disable OpenCV window display | 0 |
| `MAXIM_PROVENANCE_VERBOSITY` | Provenance tracing (0=off, 1=compact, 2=verbose) | 0 |
| `TWILIO_ACCOUNT_SID` | Twilio Account SID (for comms) | None |
| `TWILIO_AUTH_TOKEN` | Twilio Auth Token (for comms) | None |
| `TWILIO_FROM_NUMBER` | Twilio phone number (for comms) | None |
| `CUDA_VISIBLE_DEVICES` | GPU selection (empty string = CPU only) | auto |

## Config Files

All config files are in `data/util/`. User-modifiable files:

### llm.json -- LLM Configuration

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

### whisper.json -- Audio Transcription

Controls Whisper model, device, VAD settings.

Key fields:
- `model` -- Whisper model size (tiny through large-v3, distil-large-v3 recommended)
- `device` -- auto, cpu, or cuda
- `compute_type` -- int8 (fast), float16 (GPU), float32 (compatible)
- `language` -- language code or "auto"
- `vad_filter` -- enable voice activity detection
- `vad_threshold` -- 0.0-1.0, lower = more sensitive (default: 0.25)

### phrase_responses.json -- Voice Commands

Maps spoken phrases to actions. Format:
```json
{
  "maxim shutdown": { "call": "request_shutdown", "cooldown_s": 2.0 },
  "maxim sleep": { "call": "request_sleep", "cooldown_s": 2.0 },
  "maxim": { "call": "wake_up_agentic", "wake_word": true, "cooldown_s": 2.0 }
}
```
Users can add custom voice commands by adding entries.

### key_responses.json -- Keyboard Shortcuts

Maps key presses to actions:
- `c` -- center vision
- `u` -- mark trainable moment
- `0-9` -- label outcome (for training mode)

## Auto-Generated Files (Do Not Edit)

- `adaptive_thresholds.json` -- auto-tuned novelty/salience thresholds
- `focus_learner.json` -- motor gain learning state
- `learned_bounds.json` -- workspace safety bounds
- `cost_state.json` -- resource usage tracking

## Outputs Directory Structure

```
data/
├── audio/          -- WAV recordings
├── videos/         -- MP4 recordings
├── transcript/     -- JSONL transcripts with timestamps
├── memory/         -- Episodic memories (persistent)
├── logs/           -- Run logs
├── plans/
│   ├── checkpoints/ -- Goal tree snapshots
│   └── exports/     -- Exported plan files
└── util/           -- Config files (see above)
```
