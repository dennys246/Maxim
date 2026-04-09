# Maxim Reference

Detailed reference material for Maxim's architecture, bio-system mappings, and configuration. For getting started, see [README.md](../README.md).

---

## Architecture

Maxim follows a strict layered architecture with one-way dependencies:

```
Agents -> Planning -> Decision Engine -> Runtime -> Executor -> Tools -> Environment -> State -> Memory
```

### Core Modules

| Module | Responsibility |
|--------|---------------|
| `src/maxim/agents/` | Goal reasoning, intent generation (no side effects) |
| `src/maxim/planning/` | Plan generation, refinement, and decision engine |
| `src/maxim/tools/` | Tool implementations (side effects) |
| `src/maxim/environment/` | World observation (no side effects) |
| `src/maxim/memory/` | Storage and retrieval |
| `src/maxim/runtime/` | Agent orchestration loop |
| `src/maxim/conscience/` | Reachy capture/inference/control loop |
| `src/maxim/modes/` | Operating mode definitions |
| `src/maxim/proprioception/` | Movement tracking and pain detection |
| `src/maxim/harm/` | Predictive harm detection (velocity, joint limits) |
| `src/maxim/energy/` | Resource expenditure tracking (tokens, compute, movement) |
| `src/maxim/bridges/` | Cross-system integration (pain, energy, memory) |
| `src/maxim/embodiment/` | SEM protocol (Sensor-Entity-Modulator), Cerebellum, motor programs |
| `src/maxim/mesh/` | Agent mesh: identity, protocol, transport, knowledge sharing, delegation |
| `src/maxim/simulation/` | Simulation modes, generative campaigns, research protocol, benchmarks |
| `src/maxim/integration/` | MemoryHub cross-system coordinator (11 bio-systems) |
| `src/maxim/decisions/` | NAc causal learning, adaptive planner |
| `src/maxim/time/` | SCN temporal rhythm indexing |
| `src/maxim/similarity/` | Entorhinal Cortex similarity matching |
| `src/maxim/math/` | Angular Gyrus mathematical cognition, IPS fast stats |
| `src/maxim/default_network/` | Reactive behavior layer (thalamic gate, arbiter) |
| `src/maxim/salience/` | Novelty tracking, interest matching |
| `src/maxim/interactive/` | Interactive runtime: universal prompt protocol, rich terminal display, DM extensions |
| `src/maxim/cli_utils.py` | CLI helper utilities (extracted from main CLI module) |
| `src/maxim/runtime/tool_dispatch.py` | Tool execution dispatch (extracted from executor) |
| `src/maxim/runtime/bio_integration.py` | Bio-system wiring for the agent loop |
| `src/maxim/runtime/llm_server.py` | Local llama-cpp server lifecycle management |
| `src/maxim/simulation/sim_types.py` | Shared type definitions for simulation subsystem |
| `src/maxim/simulation/campaign_runner.py` | Campaign execution engine (generative + DM) |
| `src/maxim/models/language/cloud_dispatch.py` | Cloud provider request routing and redaction |
| `src/maxim/memory/store.py` | Split persistence protocols: EpisodicStore, CausalStore, SemanticStore |
| `src/maxim/memory/atl.py` | ATL semantic concept memory (concepts, relationships, grounding) |
| `src/maxim/memory/concept_extractor.py` | Concept extraction from episodic memories |
| `src/maxim/memory/concept_grounder.py` | Concept grounding against perceptual evidence |
| `src/maxim/memory/pattern_completer.py` | Pattern completion from partial cues |
| `src/maxim/memory/semantic_promoter.py` | Promotion of recurring patterns to semantic concepts |
| `src/maxim/memory/cross_layer.py` | CrossLayerGraph: associative edges between memory systems |
| `src/maxim/memory/consolidation.py` | ConsolidationOrchestrator: wave-based sleep consolidation |

See [ARCHITECTURE.md](../ARCHITECTURE.md) for detailed design rules.

---

## Bio-System Glossary

Maxim uses neuroscience-inspired names. Here is the translation:

| Bio Name | Plain English | Module | What It Does |
|----------|--------------|--------|--------------|
| Hippocampus | Episodic memory | `memory/` | Stores and recalls experiences (events, conversations) |
| ATL | Semantic memory | `memory/` | Extracts concepts, categories, and generalizations |
| NAc | Reward / causal learning | `decisions/` | Learns cause-and-effect relationships ("what leads to what") |
| SCN | Internal clock | `time/` | Tracks circadian-like temporal patterns and rhythms |
| EC | Memory indexing | `similarity/` | Routes queries to the right memory store via similarity |
| Angular Gyrus | Cross-modal algebra | `math/` | Combines memories across different modalities |
| Cerebellum | Motor prediction | `embodiment/` | Predicts outcomes of physical actions, learns motor programs |
| Amygdala / Fear | Threat detection | `proprioception/` | Detects harm, triggers pain signals, gates risky actions |
| Default Network | Reactive behavior | `default_network/` | Background processing, idle behaviors, spontaneous thoughts |

---

## Operating Modes

Maxim's mode system combines two dimensions:

1. **ProcessingState** -- `awake` or `sleep`. Determines whether the agent loop is running.
2. **OperationalMode** -- `planning`, `supervised`, or `autonomous`. Controls permissions and initiative level.

Sleep is not a mode -- it is a processing state the agent enters by calling the `sleep` tool and exits automatically when user input arrives.

See [docs/user/modes-guide.md](user/modes-guide.md) for the full guide.

---

## Planning System

| Component | Location | Description |
|-----------|----------|-------------|
| `PlanManager` | `planning/plan_manager.py` | Plan lifecycle management |
| `DecisionEngine` | `planning/decision_engine.py` | Single point of action selection |
| `Policy` | `planning/policy.py` | Constraints and guardrails |
| `PlanDocument` | `planning/plan_document.py` | Structured plan representation |

Decision flow:
```
Observe state -> Agents propose intents -> Planners propose plans
    -> Policies constrain plans -> Decision engine selects action
    -> Runtime executes
```

---

## Feature Reference

| Feature | Description |
|---------|-------------|
| Adaptive Planning | ADaPT + Reflexion hybrid: direct execution first, recursive decomposition on failure |
| Multi-Signal Scoring | Plans scored across 6 dimensions: NAc value, EC familiarity, concept relevance, delay efficiency, depth penalty, action cost |
| Parallel Execution | Independent sub-goals run concurrently with thread pools |
| Reflection Loops | Surprising failures (RPE > 0.3) generate verbal self-critiques stored as episodic memories |
| Prompt Profiles | `minimal`, `standard`, `rich` profiles for different hardware |
| Checkpointing | Persist and resume goal trees across restarts |
| FearAgent | Safety review for sensitive operations before execution |
| Pain Detection | Proprioceptive monitoring for aversive movement patterns |
| Harm Prediction | Zero-latency prediction of harmful outcomes before execution |
| Contemplation | Local chain-of-thought: multi-pass critique+refine for complex plans |
| Energy Tracking | Resource expenditure monitoring for tokens, compute, and movement |
| Introspection Tools | 10 read-only tools exposing biological subsystems to the LLM |
| Learned Tool Index | Keyword-weighted hashtable learns which tools match which goals; saves ~74% of tool-context tokens |
| Coding Tools | Edit files, search code, run tests, git diff/commit with structured error reporting |
| Skills & Protocols | Composable capabilities with lifecycle states and workspace constraints |
| SMS/Voice Comms | Send and receive texts/calls via Twilio |
| Generative Campaigns | LLM-driven narrative arcs, bridge-and-compress for long campaigns |
| Research Protocol | Multi-agent research: Researcher + Writer + Reviewer agents, dual-LLM, experiment tracking |
| Embodiment | SEM protocol (Sensor-Entity-Modulator) for body definition, Cerebellum forward models, motor programs with engrams |
| Agent Mesh | Cooperative peer-to-peer network: knowledge sharing, task delegation, distributed planning, SCN clock sync |
| Multi-LLM Scaling | Local + remote + cloud LLM backends, Cloudflare tunnel, per-tier model routing, hot-swap |
| Benchmarks | Multi-model comparative testing with bio-system expectations and scenario suites |

---

## LLM Models

### Local Models

| Profile | Model | Size | Context |
|---------|-------|------|---------|
| `smollm-1.7b` | SmolLM 1.7B Instruct | ~1.1 GB | 4096 |
| `mistral-7b` | Mistral 7B Instruct v0.2 | ~4.4 GB | 4096 |
| `llama3-8b` | Llama 3 8B Instruct | ~4.9 GB | 8192 |
| `phi3-mini` | Microsoft Phi-3 Mini | ~2.3 GB | 4096 |
| `qwen2-7b` | Qwen2 7B Instruct | ~4.4 GB | 8192 |
| `qwen2.5-14b-instruct` | Qwen2.5 14B Instruct | ~9.0 GB | 8192 |

### Cloud Providers

10 cloud provider profiles are supported: Anthropic (`claude-sonnet`, `claude-haiku`), OpenAI, Gemini, Groq, Together, Fireworks, Mistral, DeepSeek. Cloud dispatch is opt-in via `MAXIM_LLM_CLOUD_ENABLED=1`.

### Quantization

| Level | Quality | Use Case |
|-------|---------|----------|
| `Q3_K_M` | Fair | Memory constrained |
| `Q4_K_M` | Good (default) | Recommended |
| `Q5_K_M` | Better | Quality priority |
| `Q8_0` | Excellent | Maximum quality |

---

## Configuration

### Environment Variables

| Variable | Description |
|----------|-------------|
| `ANTHROPIC_API_KEY` | Required for Claude backend |
| `OPENAI_API_KEY` | Required for OpenAI backend |
| `GOOGLE_API_KEY` | Required for Gemini backend |
| `GROQ_API_KEY` | Required for Groq backend |
| `TOGETHER_API_KEY` | Required for Together backend |
| `FIREWORKS_API_KEY` | Required for Fireworks backend |
| `MISTRAL_API_KEY` | Required for Mistral API backend |
| `DEEPSEEK_API_KEY` | Required for DeepSeek backend |
| `MAXIM_LLM_ENABLED` | Enable LLM (1/true) |
| `MAXIM_LLM_PROFILE` | Model profile name |
| `MAXIM_LLM_QUANTIZATION` | Quantization level |
| `MAXIM_COMMS_ENABLED` | Enable SMS/Voice comms (1/true) |
| `MAXIM_PROVENANCE_VERBOSITY` | 0=off, 1=compact, 2=verbose |
| `CUDA_VISIBLE_DEVICES` | GPU selection (empty for CPU) |

See [CLAUDE.md](../CLAUDE.md) for the full environment variable reference including heartbeat, cloud, and peer variables.

---

## Persistence

Components that persist state to `~/.maxim/`:

| Component | CLI Clear |
|-----------|-----------|
| FocusLearner | `--clear-memory focus` |
| WorkspaceBoundsLearner | `--clear-memory bounds` |
| EscalationLearningBridge | `--clear-memory escalation` |
| FearCircuitBridge | `--clear-memory fear` |
| AdaptiveThresholdController | `--clear-memory threshold` |
| NAc | `--clear-memory nac` |
| SCN | `--clear-memory scn` |
| Hippocampus | `--clear-memory hippo` |
| ATL | `--clear-memory atl` |
| AngularGyrus | `--clear-memory angular` |
| Cerebellum | `--clear-memory cerebellum` |
| PainDetector | `--clear-memory pain` |
| SemanticEmbeddings | `--clear-memory semantic` |

Clear all: `maxim --clear-memory all`

---

## Reachy Mini Robot Setup

### Prerequisites

1. Reachy Mini on the same LAN/Wi-Fi
2. Pollen Robotics SDK installed (see [SDK guide](https://github.com/pollen-robotics/reachy_mini/blob/develop/docs/SDK/installation.md))

### Starting the Daemon

```bash
ssh pollen@<REACHY_IP>
sudo systemctl stop reachy-mini-daemon
source /venvs/mini_daemon/bin/activate
python -m reachy_mini.daemon.app.main --wireless-version --no-localhost-only
```

### Troubleshooting

```bash
python scripts/check_reachy_connection.py --host <REACHY_IP>
```

| Issue | Solution |
|-------|----------|
| Port 8443 refused | Restart Reachy or run `reachyminios_check` |
| Port 7447 refused | Check daemon: `systemctl status reachy-mini-daemon` |
| Matplotlib crash | `rm -rf ~/.cache/matplotlib && fc-cache -f` |
| Whisper segfaults | `MAXIM_WHISPER_COMPUTE_TYPE=float32 maxim` |
| OpenCV Qt warnings | `MAXIM_DISABLE_IMSHOW=1 maxim` |

---

## GPU Acceleration

Maxim auto-detects GPUs for vision, motor cortex, and LLM inference.

```bash
# Force CPU-only mode
CUDA_VISIBLE_DEVICES="" maxim
```

---

## Outputs

Each run creates timestamped artifacts under `~/.maxim/`:

| Path | Content |
|------|---------|
| `sessions/` | Session data, sim reports |
| `memory/` | Persisted memory state |
| `benchmarks/` | Benchmark results |

---

## See Also

- [README.md](../README.md) -- Getting started
- [ARCHITECTURE.md](../ARCHITECTURE.md) -- Design rules
- [CLI Reference](user/cli-reference.md) -- All CLI flags
- [Modes Guide](user/modes-guide.md) -- Operating modes
- [Python API](user/python-api.md) -- Programmatic usage
- [Simulation Guide](user/simulation.md) -- Simulation framework
