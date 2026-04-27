# Maxim

Bio-inspired cognitive architecture for LLM agents — embodied sensation, homeostatic drives, and brain-modeled persistent memory enable cross-session learning without fine-tuning.

Maxim gives an LLM agent a **body** (sensors, modulators, pain), **drives** (hunger, temperature, fatigue that drift and compete), and **biological memory systems** (Hippocampus, NAc, ATL, SCN, Angular Gyrus) that learn from experience. The agent doesn't "know" fire is dangerous because GPT said so — it knows because touching fire triggered pain in its thermal sensors, NAc formed a causal link, and the enrichment pipeline surfaces "fire = negative" next session.

**Website:** [dennyschaedig.com/maxim](https://www.dennyschaedig.com/maxim)

## What Makes This Different

| Traditional LLM Agent | Maxim Agent |
|---|---|
| Stateless between sessions | Cross-session memory via hippocampal recall + NAc causal links |
| Text in, text out | Embodied: sensors, pain, homeostatic drives, reflexes |
| Learns via fine-tuning | Learns via bio-pipeline: sensation → pain/reward → causal links → enrichment |
| Flat tool list | Three interaction levels: observe, touch, acquire |
| No internal state | Hunger drifts, temperature self-regulates, fatigue accumulates |
| Prompt engineering for behavior | Behavior emerges from learned experience |

## Quickstart

```bash
# With Claude (fastest way to start)
pip install pymaxim[llm-anthropic]
export ANTHROPIC_API_KEY=sk-...
maxim --sim "test memory recall under interference"

# Or with a local model (no API key needed)
pip install pymaxim[llm-llama]
maxim --list-models                        # see available models
maxim --sim "test memory recall" --llm mistral-7b   # auto-downloads on first run

# Cradle sensorimotor development (infant agent learns from sensation)
maxim --sim cradle --embodiment bodies/infant_humanoid --sim-max-turns 25
```

Check your setup with `maxim doctor`, and find session results in `~/.maxim/sessions/`.

## Bio-Systems

Maxim's cognitive architecture is modeled after brain systems, not software patterns:

| System | Biological Analog | What It Does |
|---|---|---|
| **Hippocampus** | Episodic memory | Captures experiences, recalls by context, promotes across tiers (FORMING → SHORT_TERM → LONG_TERM) |
| **NAc** (Nucleus Accumbens) | Reward/punishment learning | Forms causal links from actions to outcomes, eligibility traces, reward bias |
| **SCN** (Suprachiasmatic Nucleus) | Circadian clock | Temporal phase tracking, oscillator predicts event imminence, anticipatory credit |
| **ATL** (Anterior Temporal Lobe) | Semantic concepts | Forms and reinforces concept categories from experience |
| **EC** (Entorhinal Cortex) | Pattern separation/completion | Substrate encoding, centroid clustering, spreading activation |
| **Angular Gyrus** | Cross-modal binding | Hebbian binding across episodes, associative retrieval |
| **PainBus** | Nociceptive system | Rich-context pain signals from embodiment failures, drives NAc learning |
| **Default Network** | Resting-state network | Novelty detection, arousal tracking, reactive behaviors |

## Embodiment & Drives

Agents have bodies with sensors, modulators, and failure modes declared in YAML:

```yaml
# Homeostatic drive — body self-regulates toward set_point
core_temperature:
  drive:
    drift_mode: homeostatic
    set_point: 0.0
    drift_rate: 0.001        # body recovers at this rate
    comfort_band: 0.4        # no discomfort within +/-0.4
    pain_scale: 0.5          # pain intensity per unit outside band

# Entropic drive — drifts away, requires external action
hunger:
  drive:
    drift_mode: entropic
    drift_direction: up
    drift_rate: 0.006
    deprivation_threshold: 0.7
    deprivation_pain: 0.3
```

Three sensation layers converge on the same pipeline:
- **Contact** (entity acquisition): pick up a rock → its sensors join your body → damage model evaluates
- **Touch** (self_effect): touch fire → one-time thermal spike on arms
- **Narrative** (keyword reflexes): narrator describes flames → reflex fires → damage → pain

All produce: sensor change → `evaluate_failures()` → PainBus → NAc learning.

## What You Can Do

- **Cradle sensorimotor development** — infant agent learns fire avoidance, drive satisfaction, and texture discrimination through structured developmental acts
- **Simulate cognitive scenarios** — test memory, safety, causal learning with LLM-driven narrative arcs
- **Run DM campaigns** — multi-encounter branching stories with SEM-embodied entities
- **Benchmark models** — compare local and cloud LLMs across cognitive task suites
- **Connect robots** — hardware-agnostic runtime; Reachy Mini ships in-tree, third-party robots plug in via `maxim.robots` entry-point group
- **Use the Python API** — 17 verb-based functions for programmatic access

## Installation

```bash
pip install pymaxim
```

### Optional Extras

| Extra | What it adds |
|-------|-------------|
| `llm-llama` | Local LLM inference via llama.cpp |
| `llm-torch` | PyTorch/Transformers backend |
| `llm-anthropic` | Claude backend |
| `llm-openai` | OpenAI backend |
| `vision` | Camera + object detection |
| `audio` | Microphone + Whisper transcription |
| `reachy` | Reachy Mini robot SDK |
| `comms` | Twilio SMS/Voice |
| `semantic` | Sentence-transformer embeddings |
| `tts` | Text-to-speech via Piper |
| `database` | PostgreSQL + pgvector memory stores |

See [getting-started.md](docs/user/getting-started.md) for the full list of 16 extras.

```bash
# Local LLM + vision
pip install pymaxim[llm-llama,vision]

# Everything for development
pip install -e '.[llm-llama,llm-anthropic,llm-openai,vision,audio]'
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
# Agent runtime
maxim --llm mistral-7b                    # local LLM
maxim --llm claude-sonnet                 # Claude

# Simulations
maxim --sim "test memory recall"          # generative campaign
maxim --sim cradle --embodiment bodies/infant_humanoid  # sensorimotor development
maxim --sim scenarios/campaigns/heist_v1.yaml           # DM campaign
maxim --sim benchmark --models mistral-7b,qwen2.5-14b   # benchmark

# Diagnostics
maxim doctor                              # environment check
maxim --list-models                       # available models
```

See [docs/user/cli-reference.md](docs/user/cli-reference.md) for all flags.

## Documentation

| Guide | Description |
|-------|-------------|
| [Getting Started](docs/user/getting-started.md) | First-run walkthrough |
| [CLI Reference](docs/user/cli-reference.md) | All command-line flags |
| [Python API](docs/user/python-api.md) | Programmatic usage |
| [Simulation](docs/user/simulation.md) | Campaigns, scenarios, cradle, benchmarks |
| [Architecture](docs/reference.md) | Module map, bio-system glossary |
| [LLM Setup](docs/user/llm-setup.md) | Model download and configuration |
| [Peer Setup](docs/user/peer-setup.md) | Multi-machine / tunnel setup |

## Contributing

Issues and PRs welcome at [github.com/dennys246/Maxim](https://github.com/dennys246/Maxim).

## License

See [LICENSE](LICENSE) for details.
