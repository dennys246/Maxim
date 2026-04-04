# Getting Started

## What is Maxim?

Maxim is a hardware-agnostic cognitive framework. It runs local LLM inference for autonomous decision-making, multi-modal perception (vision, audio, proprioception), biologically-inspired memory systems, and layered safety controls. All processing runs on-device or on your local network -- no cloud dependency required. You do not need a robot to get started -- Maxim works in headless mode, with percept simulation, or connected to a Reachy Mini robot by Pollen Robotics.

## Prerequisites

- **Python 3.12+**
- **Reachy Mini on the same LAN/Wi-Fi** (optional -- headless mode works without a robot)
- **Disk space:** ~2 GB for base install, ~6 GB with LLM models
- **Pollen Robotics SDK:** follow the [official installation guide](https://github.com/pollen-robotics/reachy_mini/blob/develop/docs/SDK/installation.md)

## Installation

### Base Install

```bash
git clone https://github.com/dennys246/Maxim.git
cd Maxim
python -m venv maxim-env
source maxim-env/bin/activate
pip install -e .
```

### Optional Extras

Install any combination with `pip install -e ".[extra1,extra2]"`.

| Extra | Command | What it adds |
|-------|---------|--------------|
| `llm` | `pip install -e ".[llm]"` | Local LLM inference via llama.cpp |
| `llm-torch` | `pip install -e ".[llm-torch]"` | PyTorch/Transformers backend (Blackwell GPUs) |
| `llm-anthropic` | `pip install -e ".[llm-anthropic]"` | Anthropic cloud backend |
| `llm-openai` | `pip install -e ".[llm-openai]"` | OpenAI cloud backend |
| `yolo` | `pip install -e ".[yolo]"` | YOLOv8 vision engine (AGPL-3.0 license) |
| `tts` | `pip install -e ".[tts]"` | Text-to-speech via Piper |
| `comms` | `pip install -e ".[comms]"` | SMS/Voice communication via Twilio |
| `semantic` | `pip install -e ".[semantic]"` | Neural embeddings for memory similarity |
| `temporal` | `pip install -e ".[temporal]"` | Natural language date/time parsing |
| `test` | `pip install -e ".[test]"` | pytest + coverage + parallel execution |
| `all` | `pip install -e ".[all]"` | Everything above (except `yolo` and `test`) |

> **Note:** The `yolo` extra pulls in `ultralytics` which is AGPL-3.0 licensed. It is excluded from `all` to keep the core install MIT-clean.

### Downloading Models

```bash
# LLM models (pick one)
./scripts/download_models.sh --llm --enable

# Vision models (RTMDet + RTMPose, Apache 2.0)
python -m maxim.models.download --vision
```

## First Run

### Without a Robot (Headless)

```bash
maxim --mode agentic --language-model smollm-1.7b
```

This starts the full agent loop without attempting a robot connection. Useful for testing LLM reasoning, planning, and coding tools on your development machine.

### Simulation Mode

```bash
# Interactive REPL -- type scenarios, get bio-subsystem traces
maxim --sim

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
maxim --mode agentic --language-model mistral-7b
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
maxim --mode agentic --language-model smollm-1.7b --verbosity 2
```

## Next Steps

- **Simulation Guide** -- testing without hardware using interactive mode or YAML scenarios
- **Modes Guide** -- choosing the right mode for your use case
- **CLI Reference** -- all available flags and options
- **Configuration** -- customizing behavior, thresholds, and safety limits
- **LLM Setup** -- model selection, quantization, and tuning
- **Robot Setup** -- detailed Reachy Mini connection and calibration guide
