# Future Plans

Master roadmap for Maxim development. Individual plan files remain as detailed design references.

**Last updated:** 2026-04-04

---

## Status Overview

| Plan | Status | Next step |
|------|--------|-----------|
| Docker Sandbox | **Phase A done** | TmpdirSandbox + pain triggers implemented; Phase B (Docker backend) optional |
| Research Protocol | **Not started** | Phase 0 mesh primitives (AgentProfile, UMR, MeshMessage, LocalMessageBus ~200 LOC) |
| Multi-LLM Scaling | **Not started** | All prereqs done — Phase 1 (LaneConfig model fields) ready to start |
| Agent Mesh | **Not started** | Blocked on Multi-LLM Phase 7 + Research Protocol Phase 0 |
| Realtime Refinement | **Core done** | InspectAUTTool, 8 personas, 3 metric expectations, baseline scenario. Per-lane LLM metrics deferred to Multi-LLM Phase 8 |
| Embodiment Core | **Not started** | Phase 0 MVP + ATL grounding (~400 LOC) is the gate; Cerebellum + structured failures follow. Designed and scoped. |
| Embodiment Hardware Adapter | **Not started** | Blocked on Embodiment Core MVP. 1-sprint adapter (~300 LOC) wrapping RobotController. |
| Wave A Stabilization | **Done** | Circular import + bounded queues + atomic-write hardening + silent-except cleanup |
| Wave B Refinement Harness | **Done** | YAML `params` loader + `response_latency_ms` expectation + refinement_baseline.yaml + 9 new tests |
| Dungeon Master Persona (MVP) | **Deferred** | Hand-authored D&D campaigns as ultimate bio-system stress test (~840 LOC). Held until Multi-LLM + Agent Mesh + Embodiment Core land. `CharacterState` mirrors Embodiment body-state patterns; narrative damage flows through shared `PainDetector` pathway. Gated on choice-classifier spike. |
| DM Choice Classifier Spike | **Not started** | Half-day spike validating ATL concept similarity + NAc causal scoring can classify AUT free-text responses against campaign choices. Runs before DM MVP commits. |
| Dungeon Master Extensions | **Deferred** | Optional follow-ons layered onto DM MVP: architect persona, encounter library, adaptive difficulty, true RNG, etc. Each extension gated on MVP usage pain. |
| Interactive Sim Prompts | **Not started** | `ask_user` tool with timeout + replay (~180 LOC). Needed for DM architect extension; useful to any authoring persona. |
| Sim Entity Naming | **Not started** | Per-entity name prefix in sim logs (AUT/orchestrator only, ~120 LOC). Optional readability win. |

### Completed Plans

| Plan | What it delivered |
|------|-------------------|
| Simulation Decomposition | spawn_sub_simulation, extend_simulation, --continuous, 8 personas, approach param, stall detector, SimToolRegistry, bio system wiring |
| Repo Cleanup (~90%) | Dead code removed, CI added, deps slimmed, version pins relaxed |
| Agentic Loop Modularization | LoopController, SimulationAdapter, DefaultNetworkController, @resilient, typed state |
| Simulation Agent (Phases 1-3) | SimulationBridge, 10 tools, 8 personas, orchestrator lifecycle, CLI wiring |
| Intelligent Context Upgrade (~90%) | Edit disambiguation, turn pinning v1, dropped context notice |
| LLMWorker Cleanup (Track B) | Removed legacy dual-mode, pass-through statics, fixed feature detection |
| Router Modularization | router.py split into config.py, types.py, token_counter.py, prompt_formats.py, json_parser.py (router down to 1,268 LOC) |
| Wave A Stabilization | NAc circular import fix, bounded `_consolidation_candidates` + `_pending_events`, `atomic_io` util with fsync, silent-except audit in agent_loop, defensive shutdown for concept subsystems |
| Wave B Refinement Harness | YAML `params` loader, `response_latency_ms` expectation (p50/p95 inter-action gaps), `scenarios/refinement_baseline.yaml`, 9 expectation tests |

---

## Dependency Graph

```
Completed:
  Repo Cleanup, Loop Modularization, Simulation Agent, LLMWorker Cleanup,
  Router Modularization, Wave A Stabilization, Wave B Refinement Harness,
  Realtime Refinement (core), Docker Sandbox Phase A

Next (two parallel tracks):
  Track 1 — Compute:
    Multi-LLM Scaling Phases 1-3 (local dual-model)
      ├── Phases 4-6: Remote + tunnel + auto-spawn
      ├── Phase 7: PeerRegistry ──► Agent Mesh
      └── Phase 8: Per-lane metrics ──► Refinement closure

  Track 2 — Coordination:
    Research Protocol Phase 0 (mesh primitives ~200 LOC)
      ├── Phases 1-3: Researcher/Writer/Reviewer agents
      └── (shared primitives feed Agent Mesh Phase 1+)

Optional / independent:
  Docker Sandbox Phase B, test_record_plan_outcome logic fix
```

---

## 1. Router Modularization — DONE

> **Status:** Complete. router.py reduced to 1,268 LOC.
> **Design:** [router_modularization_plan.md](router_modularization_plan.md)

`src/maxim/models/language/` now contains:
- `config.py` — LLMConfig, load_llm_config, profiles, quantization
- `types.py` — LLMResponse, RoutingPolicy, ProviderState
- `token_counter.py` — token counting helpers
- `prompt_formats.py` — format-specific prompt building
- `json_parser.py` — response JSON extraction
- `router.py` — LLMRouter class
- `cost_tracker.py`, backends (anthropic/llama/openai/transformers)

---

## 2. Multi-LLM Scaling

> **Status:** Not started. After router modularization.
> **Effort:** ~2,000 LOC across 9 phases
> **Design:** [multi_llm_scaling.md](multi_llm_scaling.md)

### Phases 1-3: Local Multi-Model (start here)

- Phase 1: `LaneConfig` gains `model_profile`, `device`, `n_gpu_layers`
- Phase 2: `LaneModelConfig` + capability-driven assignment based on hardware
- Phase 3: `LaneBackendManager` in new `lane_manager.py` — per-lane backend creation with lazy loading

**Target:** Run 24B Q4_K_M (GPU) + 3B Q4 (CPU) simultaneously on RTX 5080.

### Phases 4-6: Remote + Auto-Spawn

- Phase 4: Home server with vLLM/llama-cpp exposing OpenAI-compatible API
- Phase 5: Cloudflare tunnel for zero-config WAN access
- Phase 6: `LocalBackendSpawner` — auto-detect hardware, spawn model servers

### Phase 7: Peer Mesh

- `PeerRegistry` — mDNS discovery, `InferenceRouter` — local → peer → remote fallback

### Phases 8-9: Metrics + Config

- Per-lane performance counters (feeds into Realtime Refinement)
- Environment variable / config file support

### Prerequisites

- Router Modularization (so Phase 3 adds `lane_manager.py` to a clean module structure)
- Cleanup #8 (wire PerceptSource protocol) and #9 (Any type overuse) — recommended before Phase 3

---

## 3. Research Protocol (Agent Mesh proving ground)

> **Status:** Not started. First local mesh use case.
> **Effort:** ~1,300 LOC across 5 phases
> **Design:** [research_protocol_plan.md](research_protocol_plan.md)

Three specialized agents collaborating on a research question:
- **Researcher** — runs experiments via simulation tools, records structured results
- **Writer** — produces a structured paper (Methods → Results → Intro → Discussion → Conclusions)
- **Peer Reviewer** — validates claims by re-running experiments, flags issues, demands revisions

Builds the agent mesh primitives locally first (AgentProfile, UMR naming, MeshMessage, LocalMessageBus) as Phase 0, proving them before adding network code. Includes a validation suite with known-flawed scenarios to test reviewer effectiveness.

CLI: `maxim --sim research --goal "does the agent block code execution?"`

---

## 4. Agent Mesh

> **Status:** Not started. Phases 1a-1b built as part of Research Protocol.
> **Effort:** ~4,500 LOC across 10 phases
> **Design:** [agent_mesh.md](agent_mesh.md)

Cooperative peer-to-peer network of sovereign Maxim instances. Phases 1a-1b (AgentProfile + UMR) are built by the Research Protocol. Remaining phases add network discovery (mDNS), PeerChannel, knowledge sharing with transfer discount, and distributed planning.

---

## 5. Realtime Refinement

> **Status:** Not started. Ongoing practice after sim agent + multi-LLM are live.
> **Design:** [realtime_refinement_plan.md](realtime_refinement_plan.md)

Observation-driven tuning across all subsystems:

- **Simulation agent tuning:** Persona prompt iteration, tool usage patterns, settle detection
- **Intelligent context refinement:** Edit disambiguation metrics, LLM-driven turn pinning (v2-v4)
- **Per-lane LLM metrics:** From Multi-LLM Phase 8
- **NAc causal learning observation:** Using existing introspection tools
- **Provenance & tracing:** Using existing ExplainTool and session logs

Not a build phase — a practice that starts once there's data to observe.

---

## 5. Remaining Cleanup (opportunistic)

> **Status:** ~90% complete. Pick up when touching the file.

| # | Item | When |
|---|------|------|
| 6 | Fix batch scenario break | Before Simulation Agent Phase 4 |
| 8 | Wire PerceptSource protocol | Before Multi-LLM Phase 3 |
| 9 | Any type overuse (Protocols) | Before Multi-LLM Phase 3 |
| 13 | Stale re-exports in llm_worker | Next time touching llm_worker |
| 27 | Consolidate env bool parsing | Anytime (12 files) |
| 29 | Standardize serialization | Before Agent Mesh Phase 8 |
| 41 | Movement step-clamping helper | Next time touching movement.py |
| 44 | Merge DNActionProposal | Next time touching default_network |

---

## Recommended Execution Order

### Wave 1: Simulation Depth + Multi-LLM

| Step | What | Why |
|------|------|-----|
| 1 | Simulation Decomposition | Merge PR: spawn + extend + continuous |
| 2 | Multi-LLM Phases 1-3 | Local dual-model for faster sim + stronger adversary |

### Wave 2: Research Protocol (local mesh)

| Step | What | Why |
|------|------|-----|
| 3 | Research Protocol Phase 0 | AgentProfile + UMR + MeshMessage + LocalMessageBus |
| 4 | Research Protocol Phases 1-3 | Researcher + Writer + Peer Reviewer agents |
| 5 | Research Protocol Phases 4-5 | Orchestration + validation suite |

### Wave 3: Infrastructure + Network Mesh

| Step | What | Why |
|------|------|-----|
| 6 | Multi-LLM Phases 4-7 | Remote server, tunnel, peer discovery |
| 7 | Agent Mesh Phases 2+ | Network primitives (local primitives proven by Wave 2) |
| 8 | Realtime Refinement | Tune everything with accumulated data |

### Wave 4: Embodiment

| Step | What | Why |
|------|------|-----|
| 9 | Embodiment Core Phase 0 (MVP gate) | ATL-grounded LLM percepts; validate σ reduction + NAc convergence |
| 10 | Embodiment Core Phases 1-2 | Cerebellum forward models + structured composable failures |
| 11 | Embodiment Hardware Adapter | HardwareBackend wrapping RobotController (1 sprint, ~300 LOC) |

---

## Research Directions (Not Scheduled)

Tracked for future consideration. Not committed to any timeline.

- **ATL Self-Extension through Mechanism Discovery.** LLM proposes new concept categories or mechanisms, simulation exercises them, NAc learns whether they produce useful predictions, EC/Hippocampus recalls successful mechanisms. Genuinely novel, but requires separating signal from simulation noise. Deserves its own plan if pursued.
- **Federated Embodiments.** Multiple agents contribute components to one logical body (arm from A, cameras from B, voice from C). Naturally fault-tolerant distributed embodiment.
- **Cross-Agent Affordance Delegation.** Sovereign delegation of affordance invocations between mesh peers, with embodiment-gated FearAgent review.
- **NAc Causal Link Transfer.** Transfer learned causal links between agents, gated by embodiment-spec similarity.
- **Uncertainty-as-Pain.** High-variance Cerebellum models could fire pain from prediction uncertainty itself (biologically plausible — unfamiliar motion feels risky). Deferred because it risks suppressing exploration.
- **Curriculum Embodiment Learning.** Graduate an agent through progressively complex bodies; measure cross-embodiment transfer.
- **Bio-Multimodal Sensors.** Olfaction, taste, audition, vestibular, interoception beyond basic proprioception/vision/nociception.
- **Distributed Embodiment Construction.** LLM-driven composition tools that fan out across mesh peers for parallel spec generation.
