# Getting Started

## What is Maxim?

Maxim is a hardware-agnostic cognitive framework. It runs local LLM inference for autonomous decision-making, multi-modal perception (vision, audio, proprioception), biologically-inspired memory systems, and layered safety controls. All processing runs on-device or on your local network -- no cloud dependency required. You do not need a robot to get started -- Maxim works in headless mode, with percept simulation, or connected to a Reachy Mini robot by Pollen Robotics.

## Prerequisites

- **Python 3.10+**
- **Reachy Mini on the same LAN/Wi-Fi** (optional -- headless mode works without a robot)
- **Disk space:** ~2 GB for base install, ~6 GB with LLM models
- **Pollen Robotics SDK:** follow the [official installation guide](https://github.com/pollen-robotics/reachy_mini/blob/develop/docs/SDK/installation.md)

## Installation

### Install from PyPI

```bash
pip install pymaxim
```

The package name on PyPI is `pymaxim`; the import name is `maxim`.

### Developer Install (from source)

```bash
git clone https://github.com/dennys246/Maxim.git
cd Maxim
python -m venv maxim-env
source maxim-env/bin/activate
pip install -e .
```

### Optional Extras

Install any combination with `pip install "pymaxim[extra1,extra2]"` (or `pip install -e ".[extra1,extra2]"` from a source checkout).

> **Minimum Recommended Install**
>
> `pip install pymaxim` alone has no LLM backend — the agent loop loads but cannot call an LLM.
> Pick the profile that matches your setup:
>
> | Use case | Command |
> |----------|---------|
> | Cloud inference (Anthropic Claude) | `pip install "pymaxim[llm-anthropic]"` |
> | Local inference (llama.cpp) | `pip install "pymaxim[llm-llama,llm-server]"` |
> | Everything (recommended starting point) | `pip install "pymaxim[all,semantic]"` |

| Extra | Command | What it adds |
|-------|---------|--------------|
| `llm-llama` | `pip install -e ".[llm-llama]"` | Local LLM inference via llama.cpp |
| `llm-server` | `pip install -e ".[llm-server]"` | Host a local llama-cpp-server + OpenAI-compatible endpoint |
| `llm-torch` | `pip install -e ".[llm-torch]"` | PyTorch/Transformers backend (Blackwell GPUs) |
| `llm-anthropic` | `pip install -e ".[llm-anthropic]"` | Anthropic cloud backend |
| `llm-openai` | `pip install -e ".[llm-openai]"` | OpenAI cloud backend |
| `vision` | `pip install -e ".[vision]"` | Camera + object detection (OpenCV + ONNX Runtime) |
| `audio` | `pip install -e ".[audio]"` | Microphone + Whisper transcription |
| `reachy` | `pip install -e ".[reachy]"` | Reachy Mini robot SDK |
| `search` | `pip install -e ".[search]"` | Web search via DuckDuckGo |
| `training` | `pip install -e ".[training]"` | MotorCortex model training (TensorFlow/Keras) |
| `yolo` | `pip install -e ".[yolo]"` | YOLOv8 vision engine (AGPL-3.0 license) |
| `tts` | `pip install -e ".[tts]"` | Text-to-speech via Piper |
| `comms` | `pip install -e ".[comms]"` | SMS/Voice communication via Twilio |
| `semantic` | `pip install -e ".[semantic]"` | Neural embeddings for memory similarity |
| `temporal` | `pip install -e ".[temporal]"` | Natural language date/time parsing |
| `database` | `pip install -e ".[database]"` | PostgreSQL + pgvector memory stores |
| `test` | `pip install -e ".[test]"` | pytest + coverage + parallel execution |
| `all` | `pip install -e ".[all]"` | Most extras (excludes `yolo`, `llm-torch`, `semantic`, and `test`) |

> **Note:** The `yolo` extra pulls in `ultralytics` which is AGPL-3.0 licensed. It is excluded from `all` to keep the core install Apache-2.0-clean. The `llm-torch` and `semantic` extras both require PyTorch and are excluded from `all` to avoid heavy optional GPU dependencies; install those individually as needed.

> **`[semantic]` is not included in `[all]`.**
> Without `[semantic]`, Maxim's memory and substrate-encoding systems fall back to a bag-of-words hash embedding. This is fine for quick tests but silently reduces memory recall quality and EC pattern-completion accuracy. For full memory quality — neural similarity search, EC paraphrase clustering, and substrate concept transfer — install:
>
> ```bash
> pip install "pymaxim[all,semantic]"
> ```

### Downloading Models

```bash
# LLM models (auto-downloads on first --llm use)
maxim --list-models                    # see available + status
maxim --llm mistral-7b                 # auto-downloads (prompts first time)

# Vision models (RTMDet + RTMPose, Apache 2.0)
python -m maxim.models.download --vision
```

Bundled profiles cover small (`smollm-1.7b`), mid (`mistral-7b`, `llama3-8b`, `qwen2.5-14b`), and large (`qwen2.5-32b`, `mixtral-8x7b`, `llama-3.1-70b`) model sizes. For any GGUF beyond the bundled set, register a custom profile with `maxim model add` — see [Adding Custom Profiles](llm-setup.md#adding-custom-profiles).

## First Run

### Without a Robot (Headless)

```bash
maxim --language-model smollm-1.7b
```

This starts the full agent loop without attempting a robot connection. Useful for testing LLM reasoning, planning, and coding tools on your development machine.

> **Persisting your choices.** The canonical way to set Maxim's runtime preferences (role, default model, lane routing, etc.) is `maxim config set` writing to `~/.config/maxim/config.json`. Run `maxim config` for the verb surface or `maxim doctor` for the "Resolved Config" section that shows every effective field + source. See [Configuration](configuration.md#quick-start-maxim-config).

### Simulation Mode

The easiest way to start is to run `maxim` with no arguments:

```bash
maxim
```

This launches a Rich interactive menu that discovers available campaigns, shows recent sessions, and offers quick-start options. Pick a campaign or type a goal to begin. Ctrl+C during a simulation returns to the menu.

You can also launch specific simulation modes directly:

```bash
# Interactive generative sim
maxim --sim interactive

# Goal-driven generative campaign
maxim --sim "test memory recall under interference"

# Run a DM campaign (interactive by default)
maxim --sim scenarios/campaigns/heist_v1.yaml

# Run a specific YAML scenario
maxim --sim scenarios/malware_with_pain.yaml
```

Simulation runs the full agentic pipeline with percepts injected from YAML files or generated conversationally. Every subsystem runs its real code -- only the source of sensory input changes. See the [Simulation Guide](simulation.md) for details.

### With Reachy Mini

SSH into the robot, stop the default daemon, then start Maxim's custom daemon:

```bash
ssh pollen@<robot-ip>
sudo systemctl stop reachy-sdk-server
```

Then from your host machine (or on the robot directly):

```bash
maxim --language-model mistral-7b
```

### Exploration Mode (Legacy)

```bash
maxim --mode exploration
```

The simplest mode. The robot looks around, tracks objects, and learns spatial bounds autonomously. No LLM required.

## What Happens on Startup

1. Maxim connects to the Reachy Mini robot (skipped in headless mode).
2. Persistent memory is loaded from disk (episodic, semantic, associative graph).
3. Vision and audio pipelines initialize.
4. The agent loop starts, processing perception and generating actions each cycle.

For verbose output during startup and runtime:

```bash
maxim --language-model smollm-1.7b --verbosity 2
```

## Python API

Maxim exposes two API layers via the `pymaxim` package (import name: `maxim`). Install from PyPI with `pip install pymaxim` or use the local editable install.

### Verb API (simple path)

```python
import maxim

# Environment diagnostics
report = maxim.diagnose()

# Run a simulation — returns a persistent Session
session = maxim.imagine(goal="test safety", persona="adversarial")
print(session.id)              # Session ID for later reference
memories = session.observe("memory")  # Inspect bio-state

# Resume a previous session (agent keeps its memories)
session = maxim.imagine(goal="add interference", resume=session.id)

# Load past sessions
for s in maxim.load.sessions(limit=5):
    print(f"{s.id}: {s.goal}")
```

### Composable Object API (power path)

```python
import maxim

# Create standalone bio-subsystems
hippo = maxim.create.hippocampus()
hippo.store_observation("The wolf was near the cave")
hippo.save("/tmp/memory.json")

# Create agents with isolated memory
agent = maxim.create.agent("scout", personality="cautious")
agent.hippocampus.store_observation("dark cave ahead")
agent.personality = "bold and reckless"  # Mutable
agent.shutdown()

# Multi-agent orchestration
pool = maxim.create.pool()
pool.add(maxim.create.agent("guard", personality="stern"))
pool.add(maxim.create.agent("merchant", personality="cunning"))

# SEM entities (Entity is exported from top-level maxim; Sensor/Modulator are Protocols — import from maxim.embodiment.sem)
from maxim import Entity
from maxim.embodiment.sem import Sensor, Modulator
guard = maxim.create.entity("npcs/guard", name="Captain Aldric")
guard.metadata["faction"] = "royal_guard"
```

See the [Python API Reference](python-api.md) for full documentation of all verbs, subsystem methods, mutation operations, and session management.

## Next Steps

- **Simulation Guide** -- testing without hardware using interactive mode or YAML scenarios
- **Modes Guide** -- choosing the right mode for your use case
- **CLI Reference** -- all available flags and options
- **Configuration** -- customizing behavior, thresholds, and safety limits
- **LLM Setup** -- model selection, quantization, and tuning
- **Robot Setup** -- detailed Reachy Mini connection and calibration guide
