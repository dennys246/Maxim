# ------- | Maxim | -------

A Reachy Mini repo for orchestrating data streaming to and from a PC and Reachy Mini to orchestrate agents and models.

## - Getting Started with Maxim

Run the Reachy Mini daemon on the robot, then run `maxim` from any computer on the same LAN/Wi‑Fi (Zenoh peer discovery).

```bash
ssh pollen@<INSERT YOUR REACHY IP>
```

Then enter the default password 'root' if first logging on or the unique password you reset it too.

Stop the process if something is using it.

```bash
sudo systemctl stop reachy-mini-daemon
```

Check to see if you can start a new daemon process

```bash
source /venvs/mini_daemon/bin/activate
python -m reachy_mini.daemon.app.main --wireless-version --no-localhost-only
```

On your controller computer clone this repo into a folder of your choosing

```bash
git clone https://github.com/dennys246/Maxim.git
```

Before installing the `maxim` library, follow Pollen Robotics' Reachy Mini SDK installation guide for your OS: https://github.com/pollen-robotics/reachy_mini/blob/develop/docs/SDK/installation.md.

Prepare a computing environment for running Maxim by creating a new python virtual environment. Avoid installing requirements into a virtual environment you typically use for machine learning as it may mess up your tensorflow or pytorch dependencies and how your GPU is handled.

```bash
cd Maxim
python -m venv maxim-env
source maxim-env/bin/activate
pip install -e .
```

If you previously installed an older version, re-run `pip install -e .` to refresh the `maxim` command.

After `pip install -e .`, run the `maxim` command (from anywhere in that environment) to initiate basic observation using Ultralytics incredibly efficient YOLO8 model. This dynamically find objects of interest and center the Reach Mini vision on them. Audio is recorded is transcribed when enabled.

```bash
maxim
```

NOTE: You can also set `MAXIM_ROBOT_NAME=reachy_mini` and run `maxim`.
By default Maxim runs indefinitely; use `--epochs N` to stop after N cycles.

Legacy entrypoint (still supported when running from a cloned checkout):

```bash
python scripts/main.py
```

You can also run Maxim straight from a python shell or your own script by importing it (package name: `maxim`)

```python
from maxim.conscience.selfy import Maxim

maxim = Maxim()

# Starts the live loop (capture → inference/control → record artifacts)
maxim.live()

# Stop after N epochs (observation cycles)
# maxim.live(epochs=100)

# General movement wrapper to the Reachy SDK
maxim.move(y = 10, yaw = 3)

```

Of course extensions of the Maxim class using the datastreams set up are more than welcomed!

## Outputs (Default)

Each run writes a timestamped set of artifacts under `data/`:
- `videos/reachy_video_<YYYY-MM-DD_HHMMSS>.mp4`
- `audio/reachy_audio_<YYYY-MM-DD_HHMMSS>.wav`
- `transcript/reachy_transcript_<YYYY-MM-DD_HHMMSS>.jsonl` (when `--audio true` and Whisper is available)
- `logs/reachy_log_<YYYY-MM-DD_HHMMSS>.log`
- `training/motor_training_set.jsonl` (append-only log of trainable vision+movement samples)

Shared model artifacts and weights live under `data/models/` (e.g., `MotorCortex/`, `YOLO/`).

## CLI Flags

- `--mode`: `passive-interaction` (default), `live`, `train`, `sleep` (audio-only; no `wake_up()`), `agentic` (agentic runtime loop)
- `--agent`: agent name for `--mode agentic` (default: `reachy_mini`; options: `reachy_mini`, `goal`)
- `--verbosity`: `0`, `1`, `2`
- `--audio`: `True/False` (enables audio recording + transcription)
- `--audio_len`: seconds per transcription chunk (default `5.0`)
- `--language-model`: LLM profile name (e.g., `mistral-7b-instruct-v0.2`, `smollm-1.7b-instruct`; lists available on unknown)
- `--segmentation-model`: vision segmentation model (default `YOLO8`; lists available on unknown)
- `--goal`: goal for `--mode agentic` when using `--agent goal` (e.g., `read_readme` or JSON `{"tool_name":"read_file","params":{"path":"README.md"}}`)

## Keyboard Shortcuts

While `maxim` is running in a terminal, it listens for single-key presses configured in `data/util/key_responses.json` (or `$MAXIM_KEY_RESPONSES`).

Default:
- `c`: center vision (pauses training briefly in `--mode train`)
- `u`: mark the most recent trainable moment (writes a `user_marked=true` entry to `data/training/motor_training_set.jsonl`)
- `0`: label outcome as “no errors”
- `1`–`9`: label outcome as a generic “error/bug/odd behavior” code (metadata can be added later)

Voice triggers are configured in `data/util/phrase_responses.json` (or `$MAXIM_PHRASE_RESPONSES`) and are driven by new transcript lines.

Default:
- saying `Maxim` (or `Reachy`) wakes the robot (`wake_up()`), starts the agentic runtime loop, and enables voice-triggered actions
- Voice matching normalizes punctuation/possessives and treats common transcription `maximum` as `maxim`
- When `maxim` appears in a transcript line, Maxim prefers a more specific command match (e.g., sleep/observe/shutdown) before falling back to the wake word
- saying `Maxim shutdown` requests a clean shutdown (same as Ctrl+C cleanup)
- saying `Maxim sleep` (or `sleep maxim`) switches to `--mode sleep` (audio-only)
- saying `Maxim observe` (or `observe maxim`) switches to `--mode passive-interaction`

## LLM Integration (Local Language Models)

Maxim supports local LLM inference via **llama.cpp** for voice-controlled actions, chat, and agentic task execution. Models run entirely on your machine with no cloud dependencies.

### Quick Start

1. Install the LLM dependencies:

```bash
pip install -e '.[llm]'
```

2. Download a GGUF model (Q4_K_M quantization recommended):

```bash
# Example: Download Mistral 7B (Q4_K_M ~4GB)
mkdir -p data/models/LLM
# Place your .gguf file in data/models/LLM/
# Expected naming: <model-base>.Q4_K_M.gguf
```

3. Enable and run:

```bash
export MAXIM_LLM_ENABLED=1
maxim --language-model mistral-7b
```

### Supported Models

| Profile | Model | Context | Prompt Style |
|---------|-------|---------|--------------|
| `mistral-7b` | Mistral 7B Instruct v0.2 | 4096 | Mistral |
| `smollm-1.7b` | SmolLM 1.7B Instruct | 4096 | ChatML |
| `llama2-7b` / `llama2-13b` | Llama 2 Chat | 4096 | Llama 2 |
| `llama3-8b` | Llama 3 8B Instruct | 8192 | Llama 3 |
| `phi2` / `phi3-mini` | Microsoft Phi | 2048/4096 | Phi |
| `qwen2-7b` | Qwen2 7B Instruct | 8192 | ChatML |
| `gemma-2b` / `gemma-7b` | Google Gemma IT | 8192 | Gemma |

### Quantization Options

Models can be quantized to reduce size and memory usage. **Q4_K_M is the default** and recommended for most use cases:

| Level | Bits | Size | Quality | Use Case |
|-------|------|------|---------|----------|
| Q2_K | 2 | Tiny | Low | Embedded/testing |
| Q3_K_M | 3 | Small | Fair | Memory constrained |
| **Q4_K_M** | 4 | Medium | **Good (default)** | **Recommended** |
| Q5_K_M | 5 | Large | Better | Quality priority |
| Q6_K | 6 | Larger | High | Near-original |
| Q8_0 | 8 | Largest | Excellent | Maximum quality |

Set quantization via environment variable:

```bash
export MAXIM_LLM_QUANTIZATION=Q4_K_M  # Default
export MAXIM_LLM_QUANTIZATION=Q5_K_M  # Higher quality
export MAXIM_LLM_QUANTIZATION=Q3_K_M  # Smaller/faster
```

### CLI Usage

```bash
# Use default profile (Mistral 7B, Q4_K_M)
maxim --language-model mistral-7b

# Specify a different model
maxim --language-model llama3-8b

# Override model path directly
export MAXIM_LLM_MODEL_PATH='data/models/LLM/custom-model.Q4_K_M.gguf'
maxim --language-model mistral-7b
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MAXIM_LLM_ENABLED` | Enable LLM (`1`/`true`) | `false` |
| `MAXIM_LLM_PROFILE` | Model profile name | `mistral-7b-instruct-v0.2` |
| `MAXIM_LLM_QUANTIZATION` | Quantization level | `Q4_K_M` |
| `MAXIM_LLM_MODEL_PATH` | Override model file path | Auto-generated |
| `MAXIM_LLM_N_CTX` | Context window size | Profile default |
| `MAXIM_LLM_MAX_TOKENS` | Max generation tokens | `128` |
| `MAXIM_LLM_TEMPERATURE` | Sampling temperature | `0.0` |
| `MAXIM_LLM_TOP_P` | Top-p sampling | `0.95` |
| `MAXIM_LLM_TOP_K` | Top-k sampling | `40` |
| `MAXIM_LLM_N_GPU_LAYERS` | GPU layers (`-1` = all) | `-1` |
| `MAXIM_LLM_N_THREADS` | CPU threads (auto if unset) | Auto |

### Programmatic Usage (Python API)

#### Basic LLM Agent

```python
from maxim.agents import LLMAgent, LLMAgentConfig

# Simple usage with defaults (Mistral 7B, Q4_K_M)
agent = LLMAgent()
response = agent.generate("What is Python?")
print(response)

# Use a different model
agent = LLMAgent(profile="llama3-8b")

# Custom quantization
agent = LLMAgent(profile="mistral-7b", quantization="Q5_K_M")

# Full custom configuration
config = LLMAgentConfig(
    profile="phi3-mini",
    quantization="Q4_K_M",
    temperature=0.8,
    max_tokens=1024,
    system_prompt="You are a helpful coding assistant.",
    n_gpu_layers=-1,  # Use all GPU layers
)
agent = LLMAgent(config=config)
response = agent.generate("Write a Python function to sort a list")
```

#### Chat Agent (Multi-turn Conversations)

```python
from maxim.agents import ChatLLMAgent

chat = ChatLLMAgent(profile="llama3-8b", temperature=0.7)

# Conversations maintain history
chat.generate("Hi! My name is Alex.")
response = chat.generate("What's my name?")  # Has context
print(response)  # "Your name is Alex"

# Clear history when needed
chat.clear_history()
```

#### Task Agent (Structured JSON Outputs)

```python
from maxim.agents import TaskLLMAgent

task = TaskLLMAgent(
    profile="mistral-7b",
    allowed_tools={"read_file", "write_file", "search"}
)

# Returns structured intent from state
intent = task.propose_intent(state, memory)
# {"goal": {"tool_name": "read_file", "params": {...}}, "confidence": 0.9}
```

#### JSON Mode

```python
from maxim.agents import LLMAgent

agent = LLMAgent(profile="mistral-7b")

# Generate structured JSON responses
result = agent.generate_json(
    "Extract the person's name and age from: 'John is 25 years old'"
)
print(result)  # {"name": "John", "age": 25}
```

#### Model Switching at Runtime

```python
from maxim.agents import LLMAgent

agent = LLMAgent(profile="mistral-7b")
response = agent.generate("Hello!")

# Switch to a different model
agent.switch_model(profile="phi3-mini", quantization="Q4_K_S")
response = agent.generate("Hello again!")

# List available options
print(LLMAgent.list_available_profiles())
print(LLMAgent.list_quantization_levels())
print(LLMAgent.get_quantization_info("Q4_K_M"))
```

#### Using the Low-Level Router

```python
from maxim.models.language import LLMRouter, load_llm_config

# Load config from environment/files
config = load_llm_config()

# Create router
router = LLMRouter(config)

if router.enabled():
    action = router.route(
        "Maxim, read the readme file",
        allowed_tools={"read_file", "write_file"},
        allowed_commands={"center_vision", "request_sleep"},
    )
    print(action)  # {"tool_name": "read_file", "params": {"path": "README.md"}}
```

### Voice-Controlled Actions (Agentic Runtime)

When running in agentic mode, the LLM routes transcript lines containing the wake word (`maxim`) into actions:

```bash
maxim --mode agentic --language-model mistral-7b
```

Hard keyword commands always override the LLM:
- `sleep maxim` / `maxim sleep` → Switch to sleep mode
- `observe maxim` / `maxim observe` → Switch to passive mode
- `shutdown maxim` / `maxim shutdown` → Clean shutdown

### Configuration File

Create `data/util/llm.json` for persistent configuration:

```json
{
  "enabled": true,
  "profile": "mistral-7b-instruct-v0.2",
  "quantization": "Q4_K_M",
  "temperature": 0.0,
  "max_tokens": 128,
  "n_gpu_layers": -1,
  "profiles": {
    "my-custom-model": {
      "backend": "llama_cpp",
      "model_base": "my-model-name",
      "prompt_style": "chatml",
      "stop": ["<|im_end|>"],
      "n_ctx": 4096
    }
  }
}
```

### Model File Naming Convention

Place GGUF files in `data/models/LLM/` with this naming pattern:

```
<model-base>.<quantization>.gguf
```

Examples:
- `mistral-7b-instruct-v0.2.Q4_K_M.gguf`
- `Meta-Llama-3-8B-Instruct.Q5_K_M.gguf`
- `Phi-3-mini-4k-instruct.Q4_K_M.gguf`

### Benchmarking

Run against recorded transcripts:

```bash
python -m maxim.evaluation.llm_benchmark --transcript-dir data/transcript --limit 25
```

Default movement presets are defined in `data/motion/default_actions.json`.
Default head poses (including `centered`) are defined in `data/motion/default_poses.json`.
Per-call head movement step limits are defined in `data/motion/movement_thresholds.json`.

## Smoke Tests

Quick local checks live under `src/tests/`:
- `bash src/tests/basic_vision.sh`
- `bash src/tests/basic_audio.sh` (set `MAXIM_TEST_REAL_WHISPER=1` to attempt real transcription)
- `bash src/tests/basic_learn.sh` (skips if `tensorflow/keras` not installed)
- `bash src/tests/basic_move.sh --require-robot` (requires a Reachy daemon on the network)

For easy future use consider editing your Reachy's .bashrc...

```bash
nano ~/.bashrc
```

and adding aliases so you can run simple commands to start processes

```bash
alias mini-env='source /venvs/mini_daemon/bin/activate'

alias list-daemon='ss -lntp | grep 8000'
alias clear-daemon='sudo systemctl stop reachy-mini-daemon'
alias start-daemon='python -m reachy_mini.daemon.app.main --wireless-version --no-localhost-only'

alias list-zenoh='ss -lntp | grep 7447'

MAXIM_ROBOT_NAME=reachy_mini
REACHY_IP=<INSERT YOUR REACHY IP>
```
Then you can simply type commands like list-daemon, clear-daemon or start-daemon.

## Networking

Make sure you are on the same network as your Reachy Mini with no VPN. With a VPN you may be able to do simple things like start the daemon but the python SDK will struggle to connect to the Reachy.

## Troubleshooting

1. Reachy Mini immendiately closing down on running or not running at all.

Check if the reachy mini port 8000 is occupied by ssh into you Reachy Mini then
checking if the port is occupied...

```bash
ss -lntp | grep 8000
```

Stop the process if something is using it.

```bash
sudo systemctl stop reachy-mini-daemon
```

Check to see if you can start a new process

```bash
python -m reachy_mini.daemon.app.main --wireless-version --no-localhost-only
```

2. Matplotlib font cache crash on Linux/WSL (ft2font / "Can not load face" / core dump)

- Clear Matplotlib's cache: `rm -rf ~/.cache/matplotlib`
- Rebuild the system font cache: `fc-cache -f`
- If you have custom fonts under `~/.local/share/fonts`, temporarily move them out and retry.
- Run with a clean cache dir: `MPLCONFIGDIR=./data/matplotlib maxim`
- To bypass Maxim's Matplotlib preflight (not recommended): `MAXIM_SKIP_MPL_PREFLIGHT=1 maxim`
- To bypass Maxim's early Matplotlib preload (not recommended): `MAXIM_SKIP_MPL_PRELOAD=1 maxim`

3. onnxruntime VAD segfaults during audio transcription

- Temporarily disable the VAD filter to confirm the crash source: `MAXIM_VAD_FILTER=0 maxim`
- If VAD is the culprit, prefer CPU-only `onnxruntime` and avoid `onnxruntime-gpu` in the same environment.

4. faster-whisper / CTranslate2 segfaults after the first chunk (Linux/WSL)

- Force a safer compute type: `MAXIM_WHISPER_COMPUTE_TYPE=float32 maxim`
- If stable, try `int8_float32` or revert to `int8` to regain speed.

5. OpenCV imshow / Qt thread warnings (WSL/headless)

- Disable OpenCV display: `MAXIM_DISABLE_IMSHOW=1 maxim`
- Or run headless explicitly: `MAXIM_HEADLESS=1 maxim`
- If you only need logging, run with `--verbosity 0` to skip on-screen display.
- On Linux/WSL, Maxim defaults to a display subprocess for thread safety. Force main-thread imshow with `MAXIM_IMSHOW_MODE=direct`.
- Run imshow in a dedicated process: `MAXIM_IMSHOW_MODE=process maxim`
