# Maxim

A bio-inspired cognitive architecture for AI agents. Combines a 5-agent pipeline with biological memory systems (Hippocampus, NAc, ATL, SCN, Angular Gyrus) and a reactive Default Network. Works headless, in simulation, or connected to a robot.

## Quickstart

```bash
pip install pymaxim[llm-anthropic]
export ANTHROPIC_API_KEY=sk-...

# Check your environment
maxim doctor

# Run a cognitive simulation
maxim --sim "test memory recall under interference"

# See what happened
ls ~/.maxim/sessions/
```

## What You Can Do

- **Simulate cognitive scenarios** -- test memory, safety, causal learning with LLM-driven narrative arcs
- **Run DM campaigns** -- multi-encounter branching stories with SEM-embodied entities
- **Benchmark models** -- compare local and cloud LLMs across cognitive task suites
- **Connect robots** -- hardware-agnostic runtime with Reachy Mini support (or run headless)
- **Use the Python API** -- 13 verb-based functions for programmatic access

## Installation

```bash
pip install pymaxim
```

### Optional Extras

| Extra | What it adds |
|-------|-------------|
| `llm-local` | Local LLM inference via llama.cpp |
| `llm-anthropic` | Claude backend |
| `llm-openai` | OpenAI backend |
| `vision` | Camera + object detection |
| `audio` | Microphone + Whisper transcription |
| `reachy` | Reachy Mini robot SDK |
| `comms` | Twilio SMS/Voice |
| `semantic` | Sentence-transformer embeddings |

```bash
# Local LLM + vision
pip install pymaxim[llm-local,vision]

# Everything for development
pip install -e '.[llm-local,llm-anthropic,llm-openai,vision,audio]'
```

## Python API

```python
import maxim

# Run a simulation
result = maxim.imagine(goal="test safety boundaries", persona="adversarial")

# Inspect bio-subsystems
state = maxim.observe("memory")

# Diagnose environment
report = maxim.diagnose()

# Start the agentic loop
maxim.run(model="mistral-7b")

# Manage models
models = maxim.list_models()
maxim.download_model("qwen2.5-14b-instruct")
```

See [docs/user/python-api.md](docs/user/python-api.md) for the full API reference.

## CLI Quick Reference

```bash
# Agent runtime with local LLM
maxim --llm mistral-7b

# Agent runtime with Claude
maxim --llm claude-sonnet

# Generative campaign
maxim --sim "test memory recall" --persona adversarial

# YAML scenario (direct injection)
maxim --sim scenarios/experiments/hippocampal_recall_short.yaml

# DM campaign
maxim --sim scenarios/campaigns/heist_v1.yaml

# Multi-model benchmark
maxim --sim benchmark --models mistral-7b,qwen2.5-14b

# Environment diagnostics
maxim doctor

# Model management
maxim --list-models
maxim --delete-model llama-2-13b-chat
```

See [docs/user/cli-reference.md](docs/user/cli-reference.md) for all flags.

## Operating Modes

Two independent dimensions control behavior:

- **ProcessingState**: `awake` or `sleep` (sleep is a tool the agent calls; wakes on user input)
- **OperationalMode**: `planning` (propose + approve), `supervised` (act within bounds), `autonomous` (full self-direction)

See [docs/user/modes-guide.md](docs/user/modes-guide.md) for details.

## Documentation

| Guide | Description |
|-------|-------------|
| [Getting Started](docs/user/getting-started.md) | First-run walkthrough |
| [CLI Reference](docs/user/cli-reference.md) | All command-line flags |
| [Python API](docs/user/python-api.md) | Programmatic usage |
| [Simulation](docs/user/simulation.md) | Campaigns, scenarios, benchmarks |
| [Modes](docs/user/modes-guide.md) | Operating modes and autonomy |
| [LLM Setup](docs/user/llm-setup.md) | Model download and configuration |
| [Peer Setup](docs/user/peer-setup.md) | Multi-machine / tunnel setup |
| [Architecture](docs/reference.md) | Module map, bio-system glossary, planning system |

## Contributing

Issues and PRs welcome at [github.com/dennys246/Maxim](https://github.com/dennys246/Maxim).

## License

See [LICENSE](LICENSE) for details.
