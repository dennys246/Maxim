# Maxim Documentation

Comprehensive documentation for Maxim's systems and subsystems.

**Package version:** 1.1.4 (`pyproject.toml`; PyPI: https://pypi.org/project/pymaxim/) | **Release notes:** [release_1_1_4.md](announcements/release_1_1_4.md) | **Last updated:** 2026-09-05

> **2026-05-09 architectural pivot** — Maxim is moving toward a **parallel-mode architecture** where the bio-substrate (NAc + EC + ATL + Hippocampus + Default Network + reflexes) can drive action selection directly, with LLMs demoted to supporting roles (orchestrator, NPCs, optional AUT). The existing **LLM-AUT mode** remains the user-facing default; the new **substrate-primary AUT mode** ships in parallel as opt-in via `--aut-mode substrate-primary`. The **Maxim Hivemind** shareability layer (infrastructure shipped in 0.9.x; `maxim substrate export|import|inspect` CLI available) is the foundation for Oasis/Hivemind in 1.2 after the provenance, compatibility, and threat-model gates. See [Substrate-Primary Mode](substrate_primary.md) and [Hivemind + Oasis](hivemind.md) for the architecture docs.

## Quick Links

- [README](../README.md) — Getting started guide
- [ARCHITECTURE.md](../ARCHITECTURE.md) — High-level architecture
- [AGENTS.md](../AGENTS.md) — Agent system documentation
- [User Guides](user/index.md) — End-user documentation (install, CLI, API, modes)
- [Troubleshooting](troubleshooting/index.md) — Runbooks and diagnostics
- [Plans](plans/README.md) — Active roadmap and future plans
- [Known defects](bugs/README.md) — Verified defects with evidence and dispositions
- [Repository scorecards](limits/score_cards/) — Engineering grades and improvement criteria (dual-assessor: Codex + Claude, same axes, independent evidence)
- [Measurement limits](limits/README.md) — Characterized apparatus limits experiment designs must respect
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
| [Substrate-Primary Mode](substrate_primary.md) | NEW (2026-05-09) — parallel AUT mode where bio-substrate selects actions without LLM. Shipped as an experimental opt-in in 1.1.0; carries Exp 42 / 52 / 53b (see Experiments). |
| [Maxim Hivemind + Oasis](hivemind.md) | (2026-05-09) — federated peer-to-peer substrate-sharing layer. Hivemind shareability infrastructure (merge, bundle, identity, `maxim substrate` CLI) shipped in 0.9.x; Oasis/Hivemind target 1.2 after the provenance, compatibility, and threat-model gates. |
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
| [Agent Mesh (archived plan)](archive/agent_mesh.md) | Identity, protocol, transport, knowledge sharing, delegation — status banner inside; the legacy HTML guide is in [archive/html-guides/](archive/html-guides/README.md) |

### Publication & Development

| Document | Description |
|----------|-------------|
| [Publication Guide](publication_guide.md) | PyPI release checklist, including the canonical website and package-link audit |
| [Reference](reference.md) | Module layout, bio-system mappings, configuration |
| [Skills (tombstone)](skills.md) | Removed module — replaced by Cerebellum/motor programs |
| [Legacy HTML guides (archived)](archive/html-guides/README.md) | The pre-pymaxim.bio documentation set, frozen 2026-06-17; 11 guides still have no Markdown/site equivalent |

### Experiments

| Document | Description |
|----------|-------------|
| [Experiments lab notebook](experiments/README.md) | Every experiment with status — the 1.1.0 headline rows: [Exp 52 Nurture](experiments/52_nurture.md) (the want is learned; EARNED) and [Exp 53/53b](experiments/53_cross_context_readout.md) (it reads out on the physical robot; EARNED), with [limits](limits/README.md) and the [graduation ledger](plans/behavioral_graduation_candidates.md) |
| [Hippocampal Recall Experiment](experiments/hippocampal_recall_experiment.md) | Design and infrastructure for memory interference testing |
| [Run Notes](experiments/hippocampal_recall_run_notes.md) | Results from 2026-04-06: Verath recall experiment |

---

## Design essays

[dennyschaedig.com/maxim](https://www.dennyschaedig.com/maxim) hosts Denny's **design essays** — the *why*
behind Maxim's architecture. They are opinion and rationale, not reference: the canonical
reference and evidence site is [pymaxim.bio](https://pymaxim.bio/getting-started/), which wins
wherever the two disagree, and the repository's experiment, defect, limits, and graduation
ledgers win over both.

| Essay | Topic |
|---|---|
| [Maxim 1.0 — The Honest Benchmark](https://www.dennyschaedig.com/maxim/release-1-0) | The 1.0 release: what shipped, and the pre-registered experiments that mapped where the bio-substrate helps and where the LLM prior dominates |
| [Sound orientation](https://www.dennyschaedig.com/maxim/sound-orientation) | The Reachy Mini sound-orient case study — real-hardware sensorimotor learning, including the actuation bug |
| [Substrate-primary mode](https://www.dennyschaedig.com/maxim/substrate-primary) | Why the bio-substrate should drive action selection, and the phased plan for it |
| [Hivemind + Oasis](https://www.dennyschaedig.com/maxim/hivemind) | Federated bio-substrate sharing — the design, not a shipped service |
| [Agent architecture](https://www.dennyschaedig.com/maxim/agent-architecture) | Layered architecture, the bio-system pipeline, fear circuit, cerebellum |
| [Math & statistical cognition](https://www.dennyschaedig.com/maxim/math-cognition) | Statistician agent, variance, NAc reward, Angular Gyrus |
| [Memory systems](https://www.dennyschaedig.com/maxim/memory-systems) | Hippocampus, NAc, SCN, ATL, EC, Angular Gyrus in depth; semantic memory at `#semantic` |
| [Embodiment](https://www.dennyschaedig.com/maxim/embodiment) | Sensor-Entity-Modulator protocol, drives, pain cascade |
| [Imagination](https://www.dennyschaedig.com/maxim/imagination) | Real-time entity design from novel percepts |
| [Proprioception & body awareness](https://www.dennyschaedig.com/maxim/proprioception) | Body state, drive evaluation, interoception |
| [Attention & salience](https://www.dennyschaedig.com/maxim/attention-salience) | Salience modulation and attention weighting |
| [Deliberation](https://www.dennyschaedig.com/maxim/deliberation) | PFC inner monologue and the thought stream |

The reference pages that used to live beside the essays have moved to pymaxim.bio (the old
URLs redirect):

| Was | Now |
|---|---|
| Usage guide | [pymaxim.bio/installation/](https://pymaxim.bio/installation/) |
| Tools & introspection | [pymaxim.bio/reference/tools/](https://pymaxim.bio/reference/tools/) |
| Simulation | [pymaxim.bio/guides/simulation/](https://pymaxim.bio/guides/simulation/) |
| Networking / Agent mesh | [pymaxim.bio/guides/networking/](https://pymaxim.bio/guides/networking/) |
| Operating modes | [pymaxim.bio/concepts/operating-modes/](https://pymaxim.bio/concepts/operating-modes/) |
| Communication & safety | [pymaxim.bio/concepts/communication/](https://pymaxim.bio/concepts/communication/) |
| Technical deep dive | [pymaxim.bio/concepts/architecture/](https://pymaxim.bio/concepts/architecture/) |
| Experiments & results | [pymaxim.bio/research/experiments/](https://pymaxim.bio/research/experiments/) |
| Overview | [pymaxim.bio/getting-started/](https://pymaxim.bio/getting-started/) |

Five reference-flavoured pages are still served on dennyschaedig.com only until their
pymaxim.bio equivalents deploy; delete a row here when the page is retired:

| Held page | Retires to |
|---|---|
| [DM campaigns](https://www.dennyschaedig.com/maxim/dm-campaigns) | [pymaxim.bio/guides/dm-campaigns/](https://pymaxim.bio/guides/dm-campaigns/) |
| [Benchmarks](https://www.dennyschaedig.com/maxim/benchmarks) | [pymaxim.bio/guides/benchmarks/](https://pymaxim.bio/guides/benchmarks/) |
| [Prompt system & tool injection](https://www.dennyschaedig.com/maxim/prompt-system) | [pymaxim.bio/concepts/prompt-system/](https://pymaxim.bio/concepts/prompt-system/) |
| [Concept decomposition](https://www.dennyschaedig.com/maxim/concept-decomposition) | [pymaxim.bio/systems/concept-decomposition/](https://pymaxim.bio/systems/concept-decomposition/) |
| [Component library (interactive catalog)](https://www.dennyschaedig.com/maxim/component-library) | [pymaxim.bio/reference/components/](https://pymaxim.bio/reference/components/) |

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
