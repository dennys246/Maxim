# Maxim Documentation

Comprehensive documentation for Maxim's systems and subsystems.

**Version:** 1.0.0 | **Last updated:** 2026-04-09

## Quick Links

- [README](../README.md) — Getting started guide
- [ARCHITECTURE.md](../ARCHITECTURE.md) — High-level architecture
- [AGENTS.md](../AGENTS.md) — Agent system documentation
- [User Guides](user/index.md) — End-user documentation (install, CLI, API, modes)
- [Troubleshooting](troubleshooting/index.md) — Runbooks and diagnostics
- [Plans](plans/README.md) — Active roadmap and future plans

---

## System Documentation

### Core Systems

| Document | Description |
|----------|-------------|
| [Default Network](default_network.md) | Reactive behavior layer, behaviors, thalamic gating |
| [Memory](memory.md) | Hippocampus, ATL, Angular Gyrus, MemoryHub, store protocols |
| [Decisions](decisions.md) | NAc causal inference, outcome prediction |
| [Time](time.md) | SCN temporal indexing, rhythmic patterns |
| [Embodiment](embodiment_guide.md) | SEM protocol, entity specs, pain, motor learning |
| [Simulation](simulation.md) | Simulation modes, scenarios, campaigns, benchmarks |
| [Memory Layer Lifecycle](memory-layer-lifecycle.md) | Tier progression: FORMING -> WORKING -> SHORT_TERM -> LONG_TERM |
| [Memory System Interactions](memory-system-interactions.md) | Threading model, RWLock, ContextPool, MemoryHub coordination |

### Perception & Attention

| Document | Description |
|----------|-------------|
| [Semantic Similarity](semantic_similarity_analysis.md) | Phase 4 neural embeddings, NeuralSemanticLSH |
| Attention (archived) | Vision-centric spatial attention — being rebuilt as part of the [substrate plans](plans/README.md). Recognition substrate closed 2026-04-14; next is [substrate_binding_persistence](plans/substrate_binding_persistence.md). |
| Salience (archived) | Vision-centric salience — being rebuilt as part of the [substrate plans](plans/README.md). Recognition substrate closed 2026-04-14; next is [substrate_binding_persistence](plans/substrate_binding_persistence.md). |

### Safety & Learning

| Document | Description |
|----------|-------------|
| [Proprioception](proprioception.md) | Body awareness, pain detection, focus learning |
| [Harm](harm.md) | Predictive harm detection, risk assessment |
| [Energy](energy.md) | Resource tracking, energy budgeting |

### Embodiment & Campaigns

| Document | Description |
|----------|-------------|
| [Embodiment Guide](embodiment_guide.md) | SEM protocol, entity specs, sensors, modulators, failure modes |
| [Embodiment YAML Reference](embodiment_yaml_reference.md) | YAML format for body/entity definitions |
| [Generative Campaigns](generative_campaigns_guide.md) | Narrative arc system, narrator, campaign modes |
| [DM Campaigns](user/dm-campaigns.md) | Bundled SEM characters, encounter choices, cascade DAG, 11 campaigns |

### Communication & Networking

| Document | Description |
|----------|-------------|
| Communication | SMS/Voice via Twilio, webhook setup. See `src/maxim/comms/` |
| [Peer Setup](user/peer-setup.md) | Leader/peer networking, Cloudflare Tunnel, remote LLM |
| [LLM Setup](user/llm-setup.md) | Local models, 8 cloud providers, tunnels, LeaderProxy |
| [MediaMTX](mediaMTX.md) | RTSP relay: auto-start, network topology, deployment scenarios |

### Integration & Mesh

| Document | Description |
|----------|-------------|
| [Bridges](archive/bridges.md) | Cross-system integration, memory bridges |
| Agent Mesh Guide ([HTML](../htmls-guides/maxim-agent-mesh.html)) | Identity, protocol, transport, knowledge sharing, delegation |

### Publication & Development

| Document | Description |
|----------|-------------|
| [Publication Guide](publication_guide.md) | PyPI publication checklist for pymaxim v1.0.0 |
| [Reference](reference.md) | Module layout, bio-system mappings, configuration |
| [Skills (tombstone)](skills.md) | Removed module — replaced by Cerebellum/motor programs |

### Experiments

| Document | Description |
|----------|-------------|
| [Hippocampal Recall Experiment](experiments/hippocampal_recall_experiment.md) | Design and infrastructure for memory interference testing |
| [Run Notes](experiments/hippocampal_recall_run_notes.md) | Results from 2026-04-06: Verath recall experiment |

---

## Architecture Overview

```
┌──────────────┐
│  AGENT MESH  │  (identity, protocol, transport, delegation)
└──────┬───────┘
       ↓
┌─────────────────────────────────────────────────────────────────┐
│                        AGENTS LAYER                             │
│  ExecAgent (+ Contemplation), FearAgent, PerceptionAgent, etc. │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                     PLANNING LAYER                              │
│  RecursivePlannerAgent, GoalTree, RecursiveGoalExecutor        │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                   DECISION ENGINE                               │
│  NAc (Nucleus Accumbens), CausalLinks, Valence                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      RUNTIME                                    │
│  AgentLoop, Bootstrap, Capture, Prefetch                       │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    DEFAULT NETWORK                              │
│  Behaviors, ThalamicGate, PriorityArbiter, AttentionNetwork    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                       TOOLS                                     │
│  ReachyControl, Filesystem, HTTP, InternetSearch, Sandbox      │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                     EMBODIMENT                                  │
│  SEM Protocol, Cerebellum, MotorPrograms, Engrams, PainBus     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                     ENVIRONMENT                                 │
│  Reachy Mini SDK, Cameras, Microphones, Speakers               │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                       MEMORY                                    │
│  Hippocampus, ATL, EC, AngularGyrus, StateStore, SCN, Bridges  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Data Flow

### Perception → Memory

```
Camera Frame
     ↓
Vision Detection → SalienceNetwork → ThalamicGate
     ↓                                  ↓
AttentionNetwork                   Escalate?
     ↓                                  ↓
GazeController                    LLMAgent
     ↓                                  ↓
Motor Commands                  Goal Decomposition
     ↓                                  ↓
Hippocampus.capture()          Tool Execution
```

### Learning Flow

```
Action Executed
     ↓
┌────┴────┐
│ Outcome │
└────┬────┘
     ↓
┌─────────────────────────────┐
│ NAc.observe(event, outcome) │
│ ↓                           │
│ CausalLink updated          │
│ (Rescorla-Wagner learning)  │
└─────────────────────────────┘
     ↓
Future Actions
     ↓
┌──────────────────────────────┐
│ NAc.predict(event)           │
│ ↓                            │
│ Expected outcome             │
│ ↓                            │
│ FearAgent.review_action()    │
│ ↓                            │
│ Gate / Modify / Allow        │
└──────────────────────────────┘
```

---

## Module Map

```
src/maxim/
├── agents/          → AGENTS.md (+ modality.py SensoryModality/SensoryTag, sensory_gate.py SensoryGate)
├── attention/       → archive/attention.md
├── bridges/         → archive/bridges.md
├── comms/           → (SMS/Voice communication; see user/peer-setup.md for tunnel)
├── embodied_runtime/ → ARCHITECTURE.md (embodied runtime section — robot mixin stack, formerly `conscience/`)
├── data/            → (camera/audio data utilities)
├── decisions/       → decisions.md (NAc causal learning, AdaptivePlanner, significance)
├── default_network/ → default_network.md
├── embodiment/      → embodiment_guide.md (SEM protocol, Cerebellum, motor programs)
├── energy/          → energy.md
├── environment/     → README.md (environment section)
├── evaluation/      → (lightweight evaluators/metrics)
├── hardware/        → (Reachy hardware + simulation backends)
├── harm/            → harm.md
├── inference/       → (observation/control functions)
├── integration/     → archive/bridges.md (MemoryHub coordinator)
├── math/            → (IPS, AngularGyrus, linalg)
├── memory/          → memory.md (Hippocampus, ATL semantic memory, EC, consolidation)
├── mesh/            → archive/agent_mesh.md (identity, protocol, transport, knowledge sharing, delegation)
├── models/          → (vision, audio, language, movement models)
├── modes/           → user/modes-guide.md (ProcessingState × OperationalMode)
├── motion/          → (motion presets and actions)
├── planning/        → ARCHITECTURE.md (planning section, workspace dashboard/logger)
├── proprioception/  → proprioception.md
├── retrieval/       → (retrieval utilities)
├── runtime/         → ARCHITECTURE.md (runtime section)
├── salience/        → archive/salience.md
├── similarity/      → semantic_similarity_analysis.md (Phase 4 implemented)
├── simulation/      → simulation.md (orchestrator, bridge, personas, campaigns, benchmarks)
├── spatial/         → archive/bridges.md (SpatialMemoryBridge)
├── time/            → time.md (+ BoundedBin, significance-based eviction)
├── tools/           → ARCHITECTURE.md (tools section)
└── utils/           → (config, logging, plotting, filesystem helpers)
```

---

## Biological Mappings

Maxim's architecture draws inspiration from neuroscience:

| Brain Region | Maxim Component |
|--------------|-----------------|
| Prefrontal Cortex | LLMAgent, Planning, Contemplation (System 2 deliberation) |
| Hippocampus | Hippocampus memory |
| Entorhinal Cortex | EC similarity + Phase 4 semantic embeddings |
| Nucleus Accumbens | NAc decision system |
| Suprachiasmatic Nucleus | SCN temporal indexing |
| Amygdala | FearAgent, PainDetector |
| Thalamus | ThalamicGate |
| Superior Colliculus | AttentionNetwork |
| Cerebellum | Cerebellum forward models, motor programs, engrams |
| Default Mode Network | DefaultNetwork behaviors |
| Anterior Temporal Lobe | ATL semantic memory, concept extraction, grounding |
| Angular Gyrus | AngularGyrus mathematical cognition |
| Basal Ganglia | PriorityArbiter |

---

## Persistence Overview

All user data persists under `~/.maxim/` (configurable via `MAXIM_DATA_HOME`). Bundled seed data lives in `src/maxim/_data/`.

| Component | File (under `~/.maxim/`) | CLI Clear |
|-----------|--------------------------|-----------|
| FocusLearner | `util/focus_learner.json` | `--clear-memory focus` |
| WorkspaceBoundsLearner | `util/workspace_bounds.json` | `--clear-memory bounds` |
| EscalationLearningBridge | `util/escalation_learning.json` | `--clear-memory escalation` |
| FearCircuitBridge | `util/fear_learning.json` | `--clear-memory fear` |
| AdaptiveThresholdController | `util/adaptive_thresholds.json` | `--clear-memory threshold` |
| NAc | `util/nac_state.json` | `--clear-memory nac` |
| SCN | `util/scn_state.json` | `--clear-memory scn` |
| Hippocampus | `util/hippocampus.json` | `--clear-memory hippo` |
| ATL | `util/atl_state.json` | `--clear-memory atl` |
| PainDetector | `util/pain_detector.json` | `--clear-memory pain` |
| SemanticEmbeddings | `util/semantic_embeddings.npz` | `--clear-memory semantic` |
| Statistician | `util/statistician_state.json` | `--clear-memory statistician` |
| Active LLM Model | `util/active_llm_model.txt` | (manual / hot-swap) |
| Node ID | `util/node_id.txt` | (persistent mesh identity) |
| Simulation Reports | `sessions/{session_id}/` | (per-session, not auto-cleared) |

Clear all: `maxim --clear-memory all`

---

## See Also

- [ARCHITECTURE.md](../ARCHITECTURE.md) - Detailed architecture rules
- [AGENTS.md](../AGENTS.md) - Agent implementations
- [README.md](../README.md) - Getting started
