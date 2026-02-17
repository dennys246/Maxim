# Maxim

An agentic robotics framework for orchestrating Reachy Mini with multi-level goal decomposition, local LLM inference, and adaptive planning.

## Overview

Maxim provides:
- **Robotic control** via Pollen Robotics' Reachy Mini SDK (vision, audio, motor control)
- **Agentic runtime** with recursive goal decomposition and reflection loops
- **Local LLM inference** via llama.cpp (no cloud dependencies)
- **Multi-modal perception** using YOLO vision and Whisper transcription
- **Low-compute optimization** with prompt profiles for CPU-only and GPU systems

## Features

| Feature | Description |
|---------|-------------|
| **Recursive Planning** | Multi-level goal decomposition with dynamic re-planning |
| **Parallel Execution** | Independent sub-goals run concurrently with thread pools |
| **Reflection Loops** | Post-execution evaluation with adaptive course correction |
| **Prompt Profiles** | `minimal`, `standard`, `rich` profiles for different hardware |
| **Checkpointing** | Persist and resume goal trees across restarts |
| **FearAgent** | Safety review for sensitive operations before execution |
| **Voice Control** | Wake-word activation and voice-triggered actions |
| **Pain Detection** | Proprioceptive monitoring for aversive movement patterns |
| **Harm Prediction** | Zero-latency prediction of harmful outcomes before execution |
| **Energy Tracking** | Resource expenditure monitoring for tokens, compute, and movement |

---

## Getting Started

### Prerequisites

1. **Reachy Mini** on the same LAN/Wi-Fi (Zenoh peer discovery)
2. **Python 3.12+** with virtual environment
3. Follow Pollen Robotics' [SDK installation guide](https://github.com/pollen-robotics/reachy_mini/blob/develop/docs/SDK/installation.md)

### Installation

```bash
git clone https://github.com/dennys246/Maxim.git
cd Maxim
python -m venv maxim-env
source maxim-env/bin/activate
pip install -e .
```

For LLM features:
```bash
pip install -e '.[llm]'
./scripts/download_models.sh --llm --enable
```

### Running Maxim

```bash
# Default exploration mode
maxim

# With agentic runtime and LLM
maxim --mode agentic --language-model mistral-7b

# Specify prompt profile for low-compute systems
maxim --mode agentic --prompt-profile minimal
```

### Reachy Mini Setup

SSH into your Reachy and start the daemon:

```bash
ssh pollen@<REACHY_IP>
sudo systemctl stop reachy-mini-daemon
source /venvs/mini_daemon/bin/activate
python -m reachy_mini.daemon.app.main --wireless-version --no-localhost-only
```

---

## Architecture

Maxim follows a strict layered architecture with one-way dependencies:

```
Agents → Planning → Decision Engine → Runtime → Executor → Tools → Environment → State → Memory
```

### Core Modules

| Module | Responsibility |
|--------|---------------|
| `src/maxim/agents/` | Goal reasoning, intent generation (no side effects) |
| `src/maxim/planning/` | Plan generation, refinement, and decision engine |
| `src/maxim/tools/` | Tool implementations (side effects) |
| `src/maxim/environment/` | World observation (no side effects) |
| `src/maxim/memory/` | Storage and retrieval |
| `src/maxim/runtime/` | Agentic orchestration loop |
| `src/maxim/conscience/` | Reachy capture/inference/control loop |
| `src/maxim/modes/` | Operating mode definitions and strategies |
| `src/maxim/proprioception/` | Movement tracking and pain detection |
| `src/maxim/harm/` | Predictive harm detection (velocity, joint limits) |
| `src/maxim/energy/` | Resource expenditure tracking (tokens, compute, movement) |
| `src/maxim/bridges/` | Cross-system integration (pain, energy, memory) |

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed design rules.

---

## Operating Modes

| Mode | Description |
|------|-------------|
| `exploration` | Novelty-driven active discovery (default) |
| `live` | Real-time vision and motor control |
| `agentic` | Full agentic runtime with goal decomposition |
| `sleep` | Audio-only mode (no wake_up()) |
| `reflection` | Introspection and memory consolidation |
| `train` | Model training mode |

```bash
maxim --mode agentic
maxim --mode exploration
maxim --mode sleep
```

---

## Planning System

Maxim uses a multi-layer planning architecture:

### Key Components

| Component | Location | Description |
|-----------|----------|-------------|
| `PlanManager` | `planning/plan_manager.py` | Plan lifecycle management |
| `DecisionEngine` | `planning/decision_engine.py` | Single point of action selection |
| `Policy` | `planning/policy.py` | Constraints and guardrails |
| `PlanDocument` | `planning/plan_document.py` | Structured plan representation |

### Decision Flow

```
Observe state → Agents propose intents → Planners propose plans
    → Policies constrain plans → Decision engine selects action
    → Runtime executes
```

Action selection happens in exactly one place: `DecisionEngine.decide()`

---

## Prompt Profiles

Optimize for your hardware with prompt profiles:

| Profile | Max Depth | LLM Calls | Parallel | Use Case |
|---------|-----------|-----------|----------|----------|
| `minimal` | 2 | 8 | No | CPU-only, low RAM |
| `standard` | 5 | 20 | Yes (4 workers) | GPU or fast CPU |
| `rich` | 7 | 50 | Yes (8 workers) | High-end GPU |

```bash
# Set via CLI
maxim --prompt-profile minimal

# Or environment variable
export MAXIM_PROMPT_PROFILE=minimal
```

### Profile Features

**Minimal Profile**:
- Shallow decomposition (max 2 levels)
- Reflection only on failure
- No parallel execution
- Fast retry backoff

**Standard Profile**:
- Balanced depth (5 levels)
- Parallel sibling execution
- Exponential retry backoff
- Reflection on failures

**Rich Profile**:
- Deep decomposition (7 levels)
- Always reflect
- Plan validation enabled
- Maximum parallelism

---

## LLM Integration

Maxim supports local LLM inference via llama.cpp with no cloud dependencies.

### Quick Start

```bash
# Install LLM dependencies
pip install -e '.[llm]'

# Download default model (SmolLM 1.7B, ~1.1GB)
./scripts/download_models.sh --llm --enable

# Run with LLM
export MAXIM_LLM_ENABLED=1
maxim --mode agentic --language-model smollm-1.7b
```

### Supported Models

| Profile | Model | Size | Context |
|---------|-------|------|---------|
| `smollm-1.7b` | SmolLM 1.7B Instruct | ~1.1 GB | 4096 |
| `mistral-7b` | Mistral 7B Instruct v0.2 | ~4.4 GB | 4096 |
| `llama3-8b` | Llama 3 8B Instruct | ~4.9 GB | 8192 |
| `phi3-mini` | Microsoft Phi-3 Mini | ~2.3 GB | 4096 |
| `qwen2-7b` | Qwen2 7B Instruct | ~4.4 GB | 8192 |

### Python API

```python
from maxim.agents import LLMAgent, ChatLLMAgent

# Simple generation
agent = LLMAgent(profile="mistral-7b")
response = agent.generate("What is Python?")

# Multi-turn chat
chat = ChatLLMAgent(profile="llama3-8b", temperature=0.7)
chat.generate("Hi! My name is Alex.")
response = chat.generate("What's my name?")  # Has context

# JSON mode
result = agent.generate_json(
    "Extract name and age from: 'John is 25 years old'"
)
# {"name": "John", "age": 25}
```

### Quantization

| Level | Quality | Use Case |
|-------|---------|----------|
| `Q3_K_M` | Fair | Memory constrained |
| `Q4_K_M` | Good (default) | Recommended |
| `Q5_K_M` | Better | Quality priority |
| `Q8_0` | Excellent | Maximum quality |

```bash
export MAXIM_LLM_QUANTIZATION=Q4_K_M
```

---

## CLI Reference

### Main Command

```bash
maxim [OPTIONS]
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--mode` | Operating mode | `exploration` |
| `--verbosity` | Log level (0, 1, 2) | 1 |
| `--audio` | Enable audio recording | True |
| `--audio_len` | Transcription chunk seconds | 5.0 |
| `--language-model` | LLM profile name | None |
| `--prompt-profile` | Prompt optimization profile | `standard` |
| `--interactive` | Enable terminal prompt | True |
| `--memory-path` | Memory persistence path | `~/memory/memories.json` |
| `--reset` | Reset memory on startup | False |
| `--epochs` | Stop after N cycles | None (infinite) |
| `--clear-memory` | Clear persistent memory and exit | None |
| `--clear-cache` | Clear Python bytecode cache | False |

### Clearing Persistent Memory

Maxim learns over time and persists state across sessions. To reset learning:

```bash
# Clear all persistent memory
maxim --clear-memory

# Clear specific types (comma-separated)
maxim --clear-memory focus           # Movement gain learning
maxim --clear-memory bounds          # Workspace bounds
maxim --clear-memory fear,escalation # Safety and escalation thresholds
maxim --clear-memory all             # Everything
```

**Memory types:** `focus`, `bounds`, `escalation`, `fear`, `threshold`, `nac`, `scn`, `hippo`, `pain`, `semantic`

### Voice Commands

When wake word ("Maxim") is detected:
- `Maxim shutdown` - Clean shutdown
- `Maxim sleep` - Switch to sleep mode
- `Maxim observe` - Switch to reflection mode

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `c` | Center vision |
| `u` | Mark trainable moment |
| `0` | Label "no errors" |
| `1-9` | Label error codes |

---

## Configuration

### Environment Variables

| Variable | Description |
|----------|-------------|
| `MAXIM_LLM_ENABLED` | Enable LLM (1/true) |
| `MAXIM_LLM_PROFILE` | Model profile name |
| `MAXIM_LLM_QUANTIZATION` | Quantization level |
| `MAXIM_PROMPT_PROFILE` | Prompt optimization profile |
| `MAXIM_ROBOT_NAME` | Robot identifier |
| `CUDA_VISIBLE_DEVICES` | GPU selection (empty for CPU) |

### Config Files

| File | Purpose |
|------|---------|
| `data/util/llm.json` | LLM configuration |
| `data/util/phrase_responses.json` | Voice trigger mappings |
| `data/util/key_responses.json` | Keyboard shortcuts |
| `data/motion/default_actions.json` | Movement presets |
| `data/prompts/planning/` | Recursive planning prompts |

---

## Outputs

Each run creates timestamped artifacts under `data/`:

| Path | Content |
|------|---------|
| `videos/` | MP4 video recordings |
| `audio/` | WAV audio recordings |
| `transcript/` | JSONL transcripts |
| `training/` | Trainable motor samples |
| `vision/` | Vision event stream |
| `logs/` | Run logs |
| `plans/checkpoints/` | Goal tree checkpoints |
| `plans/exports/` | Exported plan files |

---

## GPU Acceleration

Maxim auto-detects GPUs for vision, motor cortex, and LLM inference.

```bash
# Force CPU-only mode
CUDA_VISIBLE_DEVICES="" maxim

# Or use helper script
./scripts/run_maxim_cpu.sh
```

### RTX 5080 / Blackwell Support

Use `tensorflow[and-cuda]` for Blackwell-architecture GPUs:

```toml
# In pyproject.toml
"tensorflow[and-cuda]>=2.15"
```

---

## Troubleshooting

### Connection Issues

Run diagnostics:
```bash
python scripts/check_reachy_connection.py --host <REACHY_IP>
# Or: maxim-diagnostics --host <REACHY_IP>
```

### Common Issues

| Issue | Solution |
|-------|----------|
| Port 8443 refused | Restart Reachy or run `reachyminios_check` |
| Port 7447 refused | Check daemon: `systemctl status reachy-mini-daemon` |
| Matplotlib crash | `rm -rf ~/.cache/matplotlib && fc-cache -f` |
| Whisper segfaults | `MAXIM_WHISPER_COMPUTE_TYPE=float32 maxim` |
| OpenCV Qt warnings | `MAXIM_DISABLE_IMSHOW=1 maxim` |

### Debug Logging

```bash
maxim --verbosity 2
```

---

## Development

### Running Tests

```bash
# Smoke tests
bash src/tests/basic_vision.sh
bash src/tests/basic_audio.sh
bash src/tests/basic_move.sh --require-robot

# Planning system tests
python -c "
import sys
sys.path.insert(0, 'src')
from maxim.planning import *
# ... test code
"
```

### Project Structure

```
Maxim/
├── src/maxim/           # Main package
│   ├── agents/          # Agent implementations
│   ├── planning/        # Planning, decision engine, and policy
│   ├── tools/           # Tool implementations
│   ├── modes/           # Operating modes
│   ├── models/          # ML models (vision, audio, language)
│   └── runtime/         # Agentic orchestration
├── data/                # Runtime data and configs
│   ├── prompts/         # LLM prompts
│   ├── models/          # Downloaded model weights
│   └── util/            # Configuration files
├── scripts/             # Utility scripts
└── tests/               # Test suite
```

---

## Roadmap

### Completed
- Long-horizon planning with PlanDocument (phases, sub-goals, energy budgets)
- LLM-driven goal decomposition via ExecAgent
- Worker pool with typed lanes (infer, review, record)
- Concurrent tool execution with conflict detection
- Plan checkpointing and session persistence
- Preemption circuit with soft preemption and rollback
- FearAgent safety gating (code review, action review, pain prediction)
- Memory consolidation and associative graph
- Nine cross-system bridges (Spatial, Salience, Planning, Escalation, Fear, Pain, Energy, Communication, Math)

### In Progress
- Enhanced parallel execution via WorkerPool
- Re-planning with failure context and alternative approaches
- Energy-aware planning with per-phase budgets

### Planned
- Execution tracing and observability
- Memory-based pattern learning for plan optimization

---

## License

See [LICENSE](LICENSE) for details.

## Contributing

Issues and PRs welcome at [github.com/dennys246/Maxim](https://github.com/dennys246/Maxim).
