# Maxim Documentation

Comprehensive documentation for Maxim's systems and subsystems.

## Quick Links

- [README](../README.md) - Getting started guide
- [ARCHITECTURE.md](../ARCHITECTURE.md) - High-level architecture
- [AGENTS.md](../AGENTS.md) - Agent system documentation

---

## System Documentation

### Core Systems

| Document | Description |
|----------|-------------|
| [Default Network](default_network.md) | Reactive behavior layer, behaviors, thalamic gating |
| [Memory](memory.md) | Hippocampus, episodic memory, state storage |
| [Decisions](decisions.md) | NAc causal inference, outcome prediction |
| [Time](time.md) | SCN temporal indexing, rhythmic patterns |

### Perception & Attention

| Document | Description |
|----------|-------------|
| [Attention](attention.md) | Spatial attention, gaze control, scene context |
| [Salience](salience.md) | Object-level salience, novelty tracking |

### Safety & Learning

| Document | Description |
|----------|-------------|
| [Proprioception](proprioception.md) | Body awareness, pain detection, focus learning |
| [Harm](harm.md) | Predictive harm detection, risk assessment |
| [Energy](energy.md) | Resource tracking, energy budgeting |

### Communication

| Document | Description |
|----------|-------------|
| Communication | SMS/Voice via Twilio, webhook setup, Cloudflare Tunnel (doc not yet written; see `src/maxim/comms/`) |

### Planning & Workspace

| Document | Description |
|----------|-------------|
| [ARCHITECTURE.md](../ARCHITECTURE.md#workspace-maxim_workspace) | Workspace structure, working notes, plan dashboard |

### Integration

| Document | Description |
|----------|-------------|
| [Bridges](bridges.md) | Cross-system integration, memory bridges |

---

## Architecture Overview

```
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
│                     ENVIRONMENT                                 │
│  Reachy Mini SDK, Cameras, Microphones, Speakers               │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                       MEMORY                                    │
│  Hippocampus, StateStore, SCN, Bridges                         │
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
├── agents/          → AGENTS.md
├── attention/       → attention.md
├── bridges/         → bridges.md
├── comms/           → (SMS/Voice communication; doc not yet written)
├── conscience/      → ARCHITECTURE.md (conscience section)
├── data/            → (camera/audio data utilities)
├── decisions/       → decisions.md (+ significance.py: learnable heuristics)
├── default_network/ → default_network.md
├── energy/          → energy.md
├── environment/     → README.md (environment section)
├── evaluation/      → (lightweight evaluators/metrics)
├── hardware/        → (Reachy hardware + simulation backends)
├── harm/            → harm.md
├── inference/       → (observation/control functions)
├── integration/     → bridges.md (MemoryHub coordinator)
├── math/            → (IPS, AngularGyrus, linalg)
├── memory/          → memory.md (+ consolidation.py, context_index.py)
├── models/          → (vision, audio, language, movement models)
├── modes/           → README.md (modes section)
├── motion/          → (motion presets and actions)
├── planning/        → ARCHITECTURE.md (planning section, workspace dashboard/logger)
├── proprioception/  → proprioception.md
├── retrieval/       → (retrieval utilities)
├── runtime/         → ARCHITECTURE.md (runtime section)
├── salience/        → salience.md
├── similarity/      → semantic_similarity_analysis.md (Phase 4 implemented)
├── spatial/         → bridges.md (SpatialMemoryBridge)
├── time/            → time.md (+ BoundedBin, significance-based eviction)
├── tools/           → ARCHITECTURE.md (tools section)
├── training/        → (training pipelines)
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
| Cerebellum | FocusLearner, motor adaptation |
| Default Mode Network | DefaultNetwork behaviors |
| Basal Ganglia | PriorityArbiter |

---

## Persistence Overview

Components that persist state:

| Component | File | CLI Clear |
|-----------|------|-----------|
| FocusLearner | `data/util/focus_learner.json` | `--clear-memory focus` |
| WorkspaceBoundsLearner | `data/util/workspace_bounds.json` | `--clear-memory bounds` |
| EscalationLearningBridge | `data/util/escalation_learning.json` | `--clear-memory escalation` |
| FearCircuitBridge | `data/util/fear_learning.json` | `--clear-memory fear` |
| AdaptiveThresholdController | `data/util/adaptive_thresholds.json` | `--clear-memory threshold` |
| NAc | `data/util/nac_state.json` | `--clear-memory nac` |
| SCN | `data/util/scn_state.json` | `--clear-memory scn` |
| Hippocampus | `data/util/hippocampus.json` | `--clear-memory hippo` |
| PainDetector | `data/util/pain_detector.json` | `--clear-memory pain` |
| SemanticEmbeddings | `data/util/semantic_embeddings.npz` | `--clear-memory semantic` |
| SignificanceWeights | `data/util/significance_weights.json` | `--clear-memory significance` |
| SimilarityIndex (context) | `data/util/context_index.json` | `--clear-memory context_index` |
| SimilarityIndex (percept) | `data/util/percept_index.json` | `--clear-memory percept_index` |
| Staged Sidecars | `data/short_term_memory/*.json` | `--clear-memory staging` |
| PlanDashboard | `.maxim_workspace/plans/ACTIVE_PLAN.md` | (auto-cleared on plan completion) |
| PlanLogger | `.maxim_workspace/plans/history.md` | (manual delete) |

Clear all: `maxim --clear-memory all`

---

## See Also

- [ARCHITECTURE.md](../ARCHITECTURE.md) - Detailed architecture rules
- [AGENTS.md](../AGENTS.md) - Agent implementations
- [README.md](../README.md) - Getting started
