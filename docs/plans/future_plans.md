# Future Plans

Master roadmap for Maxim development. Individual plan files remain as detailed design references.

**Last updated:** 2026-04-03

---

## Status Overview

| Plan | Status | Next step |
|------|--------|-----------|
| Simulation Decomposition | **In PR** | spawn_sub_simulation, extend_simulation, --continuous |
| Research Protocol | **Not started** | Local mesh proving ground (3 agents + paper) |
| Multi-LLM Scaling | **Not started** | Phases 1-3 ready (router modularization done) |
| Agent Mesh | **Not started** | Phases 1a-1b built by Research Protocol |
| Realtime Refinement | **Not started** | After running sim agent + multi-LLM |

### Completed Plans

| Plan | What it delivered |
|------|-------------------|
| Repo Cleanup (~90%) | Dead code removed, CI added, deps slimmed, version pins relaxed |
| Agentic Loop Modularization | LoopController, SimulationAdapter, DefaultNetworkController, @resilient, typed state |
| Simulation Agent (Phases 1-3) | SimulationBridge, 7 tools, 5 personas, orchestrator lifecycle, CLI wiring |
| Intelligent Context Upgrade (~90%) | Edit disambiguation, turn pinning v1, dropped context notice |
| LLMWorker Cleanup (Track B) | Removed legacy dual-mode, pass-through statics, fixed feature detection |

---

## Dependency Graph

```
Completed:
  Repo Cleanup, Loop Modularization, Simulation Agent, LLMWorker Cleanup

Next:
  Router Modularization ──► Multi-LLM Scaling Phases 1-3
                               ├── Phases 4-6: Remote + tunnel + auto-spawn
                               └── Phase 7: PeerRegistry ──► Agent Mesh

Ongoing:
  Realtime Refinement (observation-driven, after sim agent + multi-LLM are live)
```

---

## 1. Router Modularization

> **Status:** Not started. Prerequisite for Multi-LLM Phase 3.
> **Effort:** ~1-2 hours
> **Design:** [router_modularization_plan.md](router_modularization_plan.md)

Split `router.py` (1,721 LOC) into focused modules before adding multi-LLM code:

| New file | LOC | Contents |
|----------|-----|----------|
| `config.py` | ~400 | LLMConfig, load_llm_config, profiles, quantization |
| `types.py` | ~80 | LLMResponse, RoutingPolicy, ProviderState |
| `router.py` | ~1,120 | LLMRouter class (stays, reduced) |

Re-exports in router.py preserve backward compat. Backends already split (llama_backend.py, openai_backend.py, transformers_backend.py).

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
