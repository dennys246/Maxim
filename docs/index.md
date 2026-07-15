# Maxim Documentation

Comprehensive documentation for Maxim's systems and subsystems.

**Version:** 1.0.0 | **Last updated:** 2026-06-15

> **2026-05-09 architectural pivot** — Maxim is moving toward a **parallel-mode architecture** where the bio-substrate (NAc + EC + ATL + Hippocampus + Default Network + reflexes) can drive action selection directly, with LLMs demoted to supporting roles (orchestrator, NPCs, optional AUT). The existing **LLM-AUT mode** remains the user-facing default; the new **substrate-primary AUT mode** ships in parallel as opt-in via `--aut-mode substrate-primary`. The **Maxim Hivemind** shareability layer (infrastructure shipped in 0.9.x; `maxim substrate export|import|inspect` CLI available; Oasis persistent-substrate-primary instances target 1.1+) lets multiple Maxims share distilled bio-substrate across instances. See [Substrate-Primary Mode](substrate_primary.md) and [Hivemind + Oasis](hivemind.md) for the new docs.

## Quick Links

- [README](../README.md) — Getting started guide
- [ARCHITECTURE.md](../ARCHITECTURE.md) — High-level architecture
- [AGENTS.md](../AGENTS.md) — Agent system documentation
- [User Guides](user/index.md) — End-user documentation (install, CLI, API, modes)
- [Troubleshooting](troubleshooting/index.md) — Runbooks and diagnostics
- [Plans](plans/README.md) — Active roadmap and future plans
- **[Substrate-Primary Mode](substrate_primary.md)** — NEW: parallel architecture where the substrate carries action selection without LLM mediation
- **[Maxim Hivemind + Oasis](hivemind.md)** — NEW: federated bio-substrate layer for cross-instance learning

---

## System Documentation

### Core Systems

| Document | Description |
|----------|-------------|
| [Default Network](default_network.md) | Reactive behavior layer, behaviors, thalamic gating |
| [Memory](memory.md) | Hippocampus, ATL, Angular Gyrus, MemoryHub, store protocols |
| [Decisions](decisions.md) | NAc causal inference, outcome prediction, **substrate-primary action recommendation** |
| [Time](time.md) | SCN temporal indexing, rhythmic patterns |
| [Embodiment](embodiment_guide.md) | SEM protocol, entity specs, pain, motor learning |
| [Simulation](simulation.md) | Simulation modes, scenarios, campaigns, benchmarks |
| [Memory Layer Lifecycle](memory-layer-lifecycle.md) | Tier progression: FORMING -> SHORT_TERM -> LONG_TERM + WorkingMemorySet |
| [Memory System Interactions](memory-system-interactions.md) | Threading model, RWLock, ContextPool, MemoryHub coordination |

### Architecture & Modes

| Document | Description |
|----------|-------------|
| [Substrate-Primary Mode](substrate_primary.md) | NEW (2026-05-09) — parallel AUT mode where bio-substrate selects actions without LLM. Phase -1 prototype shipped; Phase 0 harness in 1.0 (B5). |
| [Maxim Hivemind + Oasis](hivemind.md) | (2026-05-09) — federated peer-to-peer substrate-sharing layer. Hivemind shareability infrastructure (merge, bundle, identity, `maxim substrate` CLI) shipped in 0.9.x; Oasis persistent-substrate-primary instances target 1.1+. |
| [Roy Harness — Persona Convergence Crucible](plans/deferred/persona_convergence_crucible.md) | NEW (2026-05-11) — long-horizon three-arm iteration runner: prime substrate via curriculum, run same held-out test across substrate-primed / persona-injected / neutral arms, report pairwise substrate divergence (`reward_bias_l2`, **`cluster_reward_bias_l2`**, episode + concept deltas). G3 fail-fast preflight + G4 cluster_id reward wire shipped 2026-05-11. CLI: `maxim roy run <spec.yaml>`. |

### Perception & Attention

| Document | Description |
|----------|-------------|
| [Semantic Similarity](semantic_similarity_analysis.md) | Phase 4 neural embeddings, NeuralSemanticLSH |
| Attention (archived) | Vision-centric spatial attention — being rebuilt as part of the [substrate plans](plans/README.md). Recognition substrate closed 2026-04-14; next is [substrate_binding_persistence](plans/archive/substrate_binding_persistence.md). |
| Salience (archived) | Vision-centric salience — being rebuilt as part of the [substrate plans](plans/README.md). Recognition substrate closed 2026-04-14; next is [substrate_binding_persistence](plans/archive/substrate_binding_persistence.md). |

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
| [Hardware Platform Guides](embodiment/README.md) | Per-platform embodiment: capabilities, limits, body-YAML — incl. [Reachy Mini](embodiment/reachy_mini/README.md) audio/sound-localization deep-dive |
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
| Agent Mesh Guide ([HTML](../html-guides/maxim-agent-mesh.html)) | Identity, protocol, transport, knowledge sharing, delegation |

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

## Website Guides

Long-form topic guides are published at **[dennyschaedig.com/maxim](https://dennyschaedig.com/maxim)**.
These HTML pages provide deep narrative coverage of each subsystem — architecture rationale, design decisions, and worked examples.

| Guide | Topic |
|-------|-------|
| [**Maxim 1.0 — The Honest Benchmark**](https://dennyschaedig.com/maxim/maxim-1-0-release.html) | The 1.0 release writeup: pre-registered cross-session experiments + where biology matters vs the LLM prior |
| [Overview](https://dennyschaedig.com/maxim/maxim-overview.html) | Project overview and philosophy |
| [Agent Architecture](https://dennyschaedig.com/maxim/maxim-agent-architecture.html) | The agent brain: ExecAgent, FearAgent, PerceptionAgent, agent loop |
| [Memory Systems](https://dennyschaedig.com/maxim/maxim-memory-systems.html) | Hippocampus, ATL, EC, AngularGyrus, tier progression, consolidation |
| [Semantic Memory](https://dennyschaedig.com/maxim/maxim-semantic-memory.html) | ATL concepts, embeddings, concept decomposition |
| [Concept Decomposition](https://dennyschaedig.com/maxim/maxim-concept-decomposition.html) | Noun-phrase extraction, EC pattern completion, substrate encoding |
| [Embodiment](https://dennyschaedig.com/maxim/maxim-embodiment.html) | SEM protocol, drives, sensors, modulators, pain cascade |
| [Proprioception](https://dennyschaedig.com/maxim/maxim-proprioception.html) | Body awareness, pain detection, interoception |
| [Prompt System](https://dennyschaedig.com/maxim/maxim-prompt-system.html) | Prompt composition, Acting Coach, tool injection, bio-modulation |
| [Deliberation](https://dennyschaedig.com/maxim/maxim-deliberation.html) | ThinkTool, ThoughtGate, working memory, System 2 deliberation |
| [Simulation](https://dennyschaedig.com/maxim/maxim-simulation.html) | Percept simulation, narrative arcs, campaign modes |
| [DM Campaigns](https://dennyschaedig.com/maxim/maxim-dm-campaigns.html) | Dungeon-master campaign format, encounter choices, cascade DAG |
| [Imagination](https://dennyschaedig.com/maxim/maxim-imagination.html) | Real-time entity design from novel percept mentions |
| [Component Library](https://dennyschaedig.com/maxim/maxim-component-library.html) | SEM component specs, foundry, auto-curation |
| [Operating Modes](https://dennyschaedig.com/maxim/maxim-operating-modes.html) | ProcessingState × OperationalMode, sleep/wake, supervised/autonomous |
| [Tools & Introspection](https://dennyschaedig.com/maxim/maxim-tools.html) | Tool registry, executor, side effects, introspection |
| [Attention & Salience](https://dennyschaedig.com/maxim/maxim-attention-salience.html) | Spatial attention, salience network, thalamic gating |
| [Math & Statistical Cognition](https://dennyschaedig.com/maxim/maxim-math-cognition.html) | Angular Gyrus, IPS fast stats |
| [Networking](https://dennyschaedig.com/maxim/maxim-networking.html) | Multi-LLM networking, peer mesh, leader/peer topology, Cloudflare Tunnel |
| [Communication](https://dennyschaedig.com/maxim/maxim-communication.html) | SMS/Voice safety, Twilio integration |
| [Agent Mesh](https://dennyschaedig.com/maxim/maxim-agent-mesh.html) | Identity, protocol, knowledge sharing, cooperative intelligence |
| [Hivemind + Oasis](https://dennyschaedig.com/maxim/maxim-hivemind.html) | Federated bio-substrate sharing, bundle format, Oasis instances |
| [Substrate-Primary Mode](https://dennyschaedig.com/maxim/maxim-substrate-primary.html) | Parallel architecture: bio-substrate drives action selection |
| [Benchmarks](https://dennyschaedig.com/maxim/maxim-benchmarks.html) | Benchmark harnesses, recovery time, performance measurement |
| [Experiments & Results](https://dennyschaedig.com/maxim/maxim-experiments.html) | Roy iterations, substrate convergence, experimental results |
| [Roadmap](https://dennyschaedig.com/maxim/maxim-roadmap.html) | 1.0 release plan, feature tracks, behavioral graduation candidates |
| [Technical Deep Dive](https://dennyschaedig.com/maxim/maxim-technical-deepdive.html) | Implementation details, architecture decisions, engineering notes |
| [Usage Guide](https://dennyschaedig.com/maxim/maxim-usage-guide.html) | Installation, CLI reference, configuration, day-to-day usage |

---

## Architecture Overview

```
┌──────────────┐
│  AGENT MESH  │  (identity, protocol, transport, delegation)
└──────┬───────┘
       ↓
┌─────────────────────────────────────────────────────────────────┐
│                        AGENTS LAYER                             │
│  ExecAgent, FearAgent, PerceptionAgent, AgenticGoalAgent       │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                     PLANNING LAYER                              │
│  AdaptivePlanner, PlanManager, DecisionEngine                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                   DECISION ENGINE                               │
│  NAc (Nucleus Accumbens), CausalLinks, Valence                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      RUNTIME                                    │
│  run_agent_loop, build_executor, AgentFactory, LLMRouter       │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    DEFAULT NETWORK                              │
│  DefaultNetwork, ThalamicGate, AttentionNetwork                │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                       TOOLS                                     │
│  ReachyControl, Filesystem, HTTP, InternetSearch, Sandbox      │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                     EMBODIMENT                                  │
│  SEM Protocol, Cerebellum, MotorPrograms, PainBus              │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                     ENVIRONMENT                                 │
│  Reachy Mini SDK, Cameras, Microphones, Speakers               │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                       MEMORY                                    │
│  Hippocampus, ATL, EC, AngularGyrus, SCN, CrossLayerGraph      │
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

For the authoritative module-level reference, see [reference.md](reference.md). The table below maps each package to its canonical doc; packages added since the last index update are marked with *.

```
src/maxim/
├── agents/          → AGENTS.md (+ modality.py SensoryModality/SensoryTag, sensory_gate.py SensoryGate)
├── analysis/      * → reference.md (cross-session substrate diff: NacDiff, EcDiff, HippocampusDiff, AtlDiff; Roy log generation)
├── attention/       → archive/attention.md
├── bench/         * → reference.md (LLM-path benchmark harnesses; maxim bench subcommand)
├── bridges/         → archive/bridges.md
├── comms/           → (SMS/Voice communication; see user/peer-setup.md for tunnel)
├── decisions/       → decisions.md (NAc causal learning, AdaptivePlanner, significance)
├── default_network/ → default_network.md
├── doctor/        * → reference.md (maxim doctor diagnostics; checks.py, platform_detect.py, cli.py)
├── embodied_runtime/ → ARCHITECTURE.md (embodied runtime section — robot mixin stack, formerly `conscience/`)
├── data/            → (camera/audio data utilities)
├── embodiment/      → embodiment_guide.md (SEM protocol, Cerebellum, motor programs)
├── energy/          → energy.md
├── environment/     → README.md (environment section)
├── evaluation/      → (lightweight evaluators/metrics)
├── hardware/        → (Reachy hardware + simulation backends)
├── harm/            → harm.md
├── hivemind/      * → hivemind.md (merge, bundle, identity; maxim substrate CLI)
├── imagination/   * → reference.md (real-time entity design; ImaginationCache, ImaginationDesigner, trigger)
├── inference/       → (observation/control functions)
├── integration/     → reference.md (MemoryHub cross-system coordinator; build_memory_hub)
├── interactive/   * → reference.md (MaximDisplay, PromptHandler, prompt protocol)
├── math/            → (IPS, AngularGyrus, linalg)
├── memory/          → memory.md (Hippocampus, ATL semantic memory, EC, consolidation)
├── mesh/            → archive/agent_mesh.md (simulation-only bus, identity, message, naming)
├── models/          → (vision, audio, language, movement models)
├── modes/           → user/modes-guide.md (ProcessingState × OperationalMode)
├── motion/          → (motion presets and actions)
├── peer/          * → user/peer-setup.md (mesh config, drain routing, admin; MeshNode, MeshConfig)
├── planning/        → ARCHITECTURE.md (planning section; AdaptivePlanner, PlanManager, PlanDashboard)
├── prompts/       * → reference.md (Acting Coach B3, cluster-bias annotation, prompt profiles)
├── proprioception/  → proprioception.md
├── provenance/    * → reference.md (ProvenanceTrace, ProvenanceEntry, opt-in traceability pipeline)
├── reactions/     * → reference.md (Reaction, ReactionBus, PerceptProducer/ReactionProducer protocols)
├── retrieval/       → (retrieval utilities)
├── roy/           * → reference.md (maxim roy CLI: run / diff / log subcommands)
├── runtime/         → ARCHITECTURE.md (runtime section)
├── salience/        → archive/salience.md
├── similarity/      → semantic_similarity_analysis.md (Phase 4 implemented)
├── simulation/      → simulation.md (orchestrator, bridge, personas, campaigns, benchmarks)
├── spatial/         → archive/bridges.md (SpatialMemoryBridge)
├── time/            → time.md (+ BoundedBin, significance-based eviction)
├── tools/           → ARCHITECTURE.md (tools section)
├── tunnel/        * → user/peer-setup.md (Cloudflare tunnel lifecycle; run_tunnel_subcommand)
└── utils/           → (config, logging, plotting, filesystem helpers)
```

---

## Biological Mappings

Maxim's architecture draws inspiration from neuroscience:

| Brain Region | Maxim Component |
|--------------|-----------------|
| Prefrontal Cortex | LLMAgent, Planning, deliberation (System 2) |
| Hippocampus | Hippocampus memory |
| Entorhinal Cortex | EC similarity + Phase 4 semantic embeddings |
| Nucleus Accumbens | NAc decision system |
| Suprachiasmatic Nucleus | SCN temporal indexing |
| Amygdala | FearAgent, PainDetector |
| Thalamus | ThalamicGate (DefaultNetwork) |
| Superior Colliculus | AttentionNetwork |
| Cerebellum | Cerebellum forward models, motor programs |
| Default Mode Network | DefaultNetwork behaviors |
| Anterior Temporal Lobe | ATL semantic memory, concept extraction, grounding |
| Angular Gyrus | AngularGyrus mathematical cognition |
| Basal Ganglia | DecisionEngine (planning) |

---

## Persistence Overview

All user data persists under `~/.maxim/` (configurable via `MAXIM_DATA_HOME`). Bundled seed data lives in `src/maxim/_data/`. For the full authoritative reference see [reference.md](reference.md).

| Component | File (under `~/.maxim/`) | `--clear-memory` key |
|-----------|--------------------------|----------------------|
| FocusLearner | `util/focus_learner.json` | `focus` |
| WorkspaceBoundsLearner | `util/workspace_bounds.json` | `bounds` |
| EscalationLearningBridge | `util/escalation_learning.json` | `escalation` |
| FearCircuitBridge | `util/fear_learning.json` | `fear` |
| AdaptiveThresholdController | `util/adaptive_thresholds.json` | `threshold` |
| NAc | `util/nac_state.json` | `nac` |
| SCN | `util/scn_state.json` | `scn` |
| Hippocampus | `util/hippocampus.json` | `hippo` |
| ATL | `util/atl_state.json` | `atl` |
| PainDetector | `util/pain_detector.json` | `pain` |
| EC semantic embeddings | `util/semantic_embeddings.npz` | `semantic` |
| StatisticianAgent | `util/statistician_state.json` | `statistician` |
| CrossLayerGraph | `util/cross_layer_graph.json` | `cross_layer` |
| PlanManager | `planning/` (directory) | `planning` |
| Active LLM Model | `util/active_llm_model.{role}.txt` | (manual / hot-swap) |
| Node ID | `util/node_id.txt` | (persistent mesh identity) |
| Simulation Reports | `sim_reports/{session_id}/` | (per-session, not auto-cleared) |
| Live Session Recordings | `sessions/{session_id}/` | (per-session, not auto-cleared) |

Clear all clearable entries: `maxim --clear-memory all`

---

## See Also

- [ARCHITECTURE.md](../ARCHITECTURE.md) - Detailed architecture rules
- [AGENTS.md](../AGENTS.md) - Agent implementations
- [README.md](../README.md) - Getting started
