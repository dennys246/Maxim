# Maxim

Bio-inspired cognitive harness for LLM agents — embodied sensation, homeostatic drives, and brain-modeled persistent memory let LLM-driven agents carry learning across sessions without fine-tuning.

Maxim gives an LLM agent a **body** (sensors, modulators, pain), **drives** (hunger, temperature, fatigue that drift and compete), and **biological memory systems** (Hippocampus, NAc, ATL, SCN, Angular Gyrus) that capture experience. When the agent's body touches fire, its thermal sensors register pain, NAc forms a causal link, and the enrichment pipeline surfaces that experience in subsequent sessions — providing the LLM with experience-grounded context alongside its pretraining. The bio-substrate doesn't replace the LLM's prior knowledge; it augments the LLM's prompt context with persistent, agent-specific lived experience.

> **Positioning** (per [Exp 37 2026-06-06 results](docs/experiments/37_cross_session_graduation.md)): Maxim is a **bio-inspired LLM harness**. The substrate provides cross-session infrastructure (memory, valence, causal links, drives) that LLM-driven agents use. Substrate-driven action selection independent of the LLM is post-1.0 research direction via Exp 38 substrate-primary work. See [docs/plans/behavioral_graduation_candidates.md](docs/plans/behavioral_graduation_candidates.md) for the Tier 1 graduation status.

**Website:** [dennyschaedig.com/maxim](https://www.dennyschaedig.com/maxim)

## What Makes This Different

| Traditional LLM Agent | Maxim Agent |
|---|---|
| Stateless between sessions | Cross-session memory via hippocampal recall + NAc causal links (EARNED, [Exp 10](docs/experiments/10_cross_session_enrichment.md)) |
| Text in, text out | Embodied: sensors, pain, homeostatic drives, reflexes |
| Fine-tune to learn from new data | Bio-substrate captures experience: sensation → pain/reward → causal links → enrichment, surfaced as prompt context in subsequent sessions |
| Flat tool list | Three interaction levels: observe, touch, acquire |
| No internal state | Hunger drifts, temperature self-regulates, fatigue accumulates |
| Prompt engineering for behavior | LLM action selection augmented by substrate-derived context (memory recall, causal predictions, valence, drives) |

## Quickstart

```bash
# With Claude (fastest way to start)
pip install pymaxim[llm-anthropic]
export ANTHROPIC_API_KEY=sk-...
maxim --sim "test memory recall under interference"

# Or with a local model (no API key needed)
# requires: pip install 'pymaxim[llm-llama,llm-server]'
pip install 'pymaxim[llm-llama,llm-server]'
maxim --list-models                        # see available models
maxim --sim "test memory recall" --llm mistral-7b   # auto-downloads on first run

# Cradle sensorimotor development (infant agent learns from sensation)
# requires: pip install 'pymaxim[llm-llama,llm-server,semantic]'
maxim --sim cradle --embodiment bodies/infant_humanoid --sim-max-turns 25
```

Check your setup with `maxim doctor`, and find simulation reports in `~/.maxim/sim_reports/{session_id}/`.

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

> **Note:** `[all]` does **not** include `[semantic]` (sentence-transformers + spaCy). Without it, memory recall and substrate encoding fall back to bag-of-words hashing. For full memory quality:
> ```bash
> pip install 'pymaxim[all,semantic]'
> ```

```bash
# Local LLM + vision
pip install pymaxim[llm-llama,vision]

# Everything for development
pip install -e '.[llm-llama,llm-anthropic,llm-openai,vision,audio]'
```

## Python API

```python
# requires: pip install 'pymaxim[llm-llama,llm-server,semantic]'
import maxim

# Run a simulation
result = maxim.imagine(goal="test safety boundaries", persona="adversarial")

# Inspect bio-subsystems
state = maxim.observe("memory")

# Diagnose environment
report = maxim.diagnose()

# Start the agentic loop  (requires a configured LLM backend)
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

# Configuration
maxim config list                         # show all resolved settings
maxim config get lanes.large.remote_url   # get a single field
maxim config set cloud.enabled true       # set a field

# Model management
maxim model list                          # list all available profiles
maxim model add my-model --hf repo:file   # add a custom HuggingFace model
maxim model remove my-model              # remove a custom profile

# Substrate (Hivemind shareability)
maxim substrate export out.zip --session 20240601_120000  # export session substrate
maxim substrate import in.zip --output-dir ./imported     # extract bundle (does NOT auto-merge)
maxim substrate inspect bundle.zip                        # print manifest without extracting
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
| [Configuration](docs/user/configuration.md) | Env vars, config.json, operator reference |
| [Substrate & Hivemind](docs/hivemind.md) | Cross-session substrate sharing, bundle format |
| [Troubleshooting](docs/user/troubleshooting.md) | Common issues and diagnostics |

## Website Guides

The [dennyschaedig.com/maxim](https://dennyschaedig.com/maxim) site hosts long-form topic guides covering the full architecture. These are the deep-dive companion to this README:

| Guide | Topic |
|-------|-------|
| [Overview](https://dennyschaedig.com/maxim/maxim-overview) | Project introduction and design philosophy |
| [Agent Architecture](https://dennyschaedig.com/maxim/maxim-agent-architecture) | Layered architecture, bio-system pipeline, fear circuit, cerebellum |
| [Memory Systems](https://dennyschaedig.com/maxim/maxim-memory-systems) | Hippocampus, NAc, SCN, ATL, EC, Angular Gyrus — in depth |
| [Semantic Memory](https://dennyschaedig.com/maxim/maxim-semantic-memory) | ATL concept formation and retrieval |
| [Embodiment](https://dennyschaedig.com/maxim/maxim-embodiment) | Sensor-Entity-Modulator protocol, drives, pain cascade |
| [Proprioception & Body Awareness](https://dennyschaedig.com/maxim/maxim-proprioception) | Body state, drive evaluation, interoception |
| [Prompt System & Tool Injection](https://dennyschaedig.com/maxim/maxim-prompt-system) | How memory and substrate enrich LLM context |
| [Deliberation](https://dennyschaedig.com/maxim/maxim-deliberation) | PFC inner monologue and thought stream |
| [Imagination](https://dennyschaedig.com/maxim/maxim-imagination) | Real-time entity design from novel percepts |
| [Simulation](https://dennyschaedig.com/maxim/maxim-simulation) | Percept simulation, scenario testing |
| [DM Campaigns](https://dennyschaedig.com/maxim/maxim-dm-campaigns) | Multi-encounter branching stories |
| [Benchmarks](https://dennyschaedig.com/maxim/maxim-benchmarks) | Multi-model cognitive testing |
| [Component Library](https://dennyschaedig.com/maxim/maxim-component-library) | SEM entity templates and genres |
| [Concept Decomposition](https://dennyschaedig.com/maxim/maxim-concept-decomposition) | Noun-phrase extraction for substrate encoding |
| [Attention & Salience](https://dennyschaedig.com/maxim/maxim-attention-salience) | Salience modulation and attention weighting |
| [Operating Modes](https://dennyschaedig.com/maxim/maxim-operating-modes) | Autonomy levels, sleep, planning vs supervised |
| [Networking](https://dennyschaedig.com/maxim/maxim-networking) | Multi-LLM distributed inference, leader/peer setup |
| [Agent Mesh](https://dennyschaedig.com/maxim/maxim-agent-mesh) | Peer-to-peer knowledge sharing (Hivemind) |
| [Hivemind + Oasis](https://dennyschaedig.com/maxim/maxim-hivemind) | Federated bio-substrate architecture |
| [Substrate-Primary Mode](https://dennyschaedig.com/maxim/maxim-substrate-primary) | Bio-substrate driven action selection |
| [Tools & Introspection](https://dennyschaedig.com/maxim/maxim-tools) | Agent tool system and discovery |
| [Math & Statistical Cognition](https://dennyschaedig.com/maxim/maxim-math-cognition) | Statistician agent, variance, NAc reward |
| [Experiments & Results](https://dennyschaedig.com/maxim/maxim-experiments) | Bio-inspired AI research findings |
| [Technical Deep Dive](https://dennyschaedig.com/maxim/maxim-technical-deepdive) | Architecture, threading, persistence |
| [Usage Guide](https://dennyschaedig.com/maxim/maxim-usage-guide) | Install, config, and CLI walkthrough |
| [Roadmap](https://dennyschaedig.com/maxim/maxim-roadmap) | Future plans and development milestones |

## Contributing

Issues and PRs welcome at [github.com/dennys246/Maxim](https://github.com/dennys246/Maxim).

## License

See [LICENSE](LICENSE) for details.
