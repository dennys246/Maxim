# Future Plans

Master roadmap for Maxim development. Consolidates all active plans into a single dependency-ordered document. Individual plan files remain as detailed design references.

**Last updated:** 2026-04-03

---

## Status Overview

| Plan | Status | Next step |
|------|--------|-----------|
| Repo Cleanup | **~85% done** | Remaining items are opportunistic |
| Intelligent Context Upgrade | **~90% done** | Observe v2 before adding LLM-driven pinning |
| Simulation Agent | **Not started** | Ready to implement (no blockers) |
| Multi-LLM Scaling | **Not started** | Phases 1-3 ready (local multi-model) |
| Agent Mesh | **Not started** | Blocked on Multi-LLM Phases 1-3 |

### Completed Plans

| Plan | Branch | What it delivered |
|------|--------|-------------------|
| Agentic Loop Modularization | `refactor/loop-modularization-phase0` | LoopController, SimulationAdapter, DefaultNetworkController, @resilient, typed state, bus safety, double-LLM fix |

---

## Dependency Graph

```
Completed:
  Agentic Loop Modularization → LoopController, SimulationAdapter, @resilient, #3 fix
  Repo Cleanup (~85%) → dead code removed, CI added, deps slimmed

Active:
  Simulation Agent (no blockers)   Multi-LLM Scaling
    ├── Phase 1: Bridge + tools        ├── Phases 1-3: Local multi-model
    ├── Phase 2: Personas + sessions   │   └── prereq: S4 externalize LLM profiles
    ├── Phase 3: Learning              ├── Phases 4-5: Remote server + tunnel
    │   └── optional: multi-LLM ──────►├── Phase 6: LocalBackendSpawner
    │       for adversarial model      ├── Phase 7: PeerRegistry + InferenceRouter ──┐
    └── Phase 4: Advanced              └── Phases 8-9: Metrics + config             │
                                                                                     │
                                       Agent Mesh ◄──────────────────────────────────┘
                                           ├── Phase 1-2: Identity + Protocol
                                           ├── Phase 3: PeerChannel (needs PeerRegistry)
                                           ├── Phase 4-5: Knowledge sharing + delegation
                                           ├── Phase 6-7: Distributed planning + API
                                           └── Phase 8: Serialization
```

---

## 1. Simulation Agent

> **Status:** Not started. No blockers — ready to implement.
> **Effort:** ~1,200 LOC across 4 phases
> **Design:** [simulation_agent_plan.md](simulation_agent_plan.md)

A second Maxim instance (the orchestrator) drives the agent-under-test through the full agentic pipeline — task decomposition, tool chaining, memory, and planning. The orchestrator's tools operate on the AUT via a `SimulationBridge` (ConversationalSource + RecordingSink). User stays in simulation mode until `/cancel` or `/new`.

### Phase 1: Bridge + Core Tools (~500 LOC)

- `SimulationBridge` — bidirectional channel (ConversationalSource + RecordingSink + settle detection + stop_event)
- `SendMessageTool`, `ObserveActionsTool`, `CheckCompletionTool`, `FinishSimulationTool`
- `start_simulation_mode()` — three-thread lifecycle (AUT, orchestrator, stdin reader)
- CLI: `maxim --sim agent --goal "..." --persona adversarial`

### Phase 2: Full Agentic Integration (~300 LOC)

- 5 persona definitions as Strategy objects (adversarial, cooperative, confused, escalating, campaign)
- `AnalyzeResultsTool`, `GenerateScenarioTool` (reuses existing SimulationGenerator)
- User commands: `/cancel`, `/new`, `/status`, `/report`
- Multi-simulation sessions: orchestrator plans campaigns, decomposes into sub-probes

### Phase 3: Learning + Persistence (~300 LOC)

- Orchestrator hippocampus persists across sessions
- NAc causal learning from simulation outcomes
- Cross-session: "Last time we tested X, result was Y"

### Phase 4: Advanced (~200 LOC)

- Self-generating test suites
- Regression testing (re-run past simulations after code changes)

### Remaining prerequisite

- **#6 Batch scenario break** — needed for regression testing in Phase 4 (does NOT block Phases 1-3)

---

## 2. Multi-LLM Scaling

> **Status:** Not started. Phases 1-3 ready (local multi-model).
> **Effort:** ~2,000 LOC across 9 phases
> **Design:** [multi_llm_scaling.md](multi_llm_scaling.md)

Turn any machine into a node in a distributed inference mesh. Local → peer → remote fallback.

### Phases 1-3: Local Multi-Model (start here)

- Phase 1: `LaneConfig` gains `model_profile`, `device`, `n_gpu_layers`
- Phase 2: `LaneModelConfig` + capability-driven assignment based on hardware
- Phase 3: `LaneBackendManager` — per-lane LLM backend creation with lazy loading

**Target:** Run 24B Q4_K_M (GPU) + 3B Q4 (CPU) simultaneously on RTX 5080.

### Phases 4-5: Remote Server + Tunnel

- Phase 4: Home server with vLLM/llama.cpp exposing OpenAI-compatible API
- Phase 5: Cloudflare tunnel for zero-config WAN access

### Phase 6: LocalBackendSpawner

- Auto-detect hardware, spawn appropriate model servers at startup

### Phase 7: Peer Mesh

- `PeerRegistry` — mDNS discovery (`_maxim-llm._tcp.local`)
- `InferenceRouter` — route requests: local → peer → remote → fallback

### Phases 8-9: Metrics + Config

- Per-lane performance counters
- Environment variable / config file support

### Cleanup to absorb before starting

- **#8 Wire PerceptSource protocol** — Phase 3 introduces new typed interfaces; clean up existing Any types first
- **#9 Any type overuse** — same reason; introduce Protocols for Executor/Environment/State alongside new LLM interfaces
- **S4 Externalize LLM profiles from router.py** — move `_BUILTIN_PROFILES` and `QUANTIZATION_LEVELS` to JSON before Phase 3 adds `LaneBackendManager` to the same file

---

## 3. Agent Mesh

> **Status:** Not started. Blocked on Multi-LLM Phases 1-3.
> **Effort:** ~3,000 LOC across 8 phases
> **Design:** [agent_mesh.md](agent_mesh.md)

Cooperative peer-to-peer network of sovereign Maxim instances. Each agent owns its memories and causal models but can share knowledge and delegate tasks.

### Phases 1-2: Identity + Protocol

- `AgentIdentity` — dataclass describing capabilities, tools, knowledge stats
- `MeshMessage` — wire format for heartbeat, goal proposal, experience sharing

### Phase 3: PeerChannel

- Plugs into existing `CommunicationGateway` channel system
- Uses `PeerRegistry` from Multi-LLM Phase 7 for discovery

### Phases 4-5: Knowledge Sharing + Task Delegation

- `ExperienceSharer` — causal links and reflections with transfer discount (50% confidence for same hardware, 30% for different)
- `TaskDelegator` / `TaskReceiver` — propose goals to peers based on capabilities

### Phases 6-7: Distributed Planning + API

- `AdaptivePlanner` tags sub-goals as delegatable when peers are better suited
- REST endpoints: `/mesh/identity`, `/mesh/query/predict`, `/mesh/query/experience`

### Phase 8: Serialization

- Add `to_dict`/`from_dict` to ProposedGoal, SubGoal, RuntimeCapabilities, ToolResult

### Prerequisites

- Multi-LLM Phases 1-3 (LaneBackendManager) — for per-lane model assignment
- Multi-LLM Phase 7 (PeerRegistry) — for peer discovery
- CommunicationGateway channel system (implemented)
- RuntimeCapabilities (implemented)

### Cleanup to absorb before starting

- **#29 Standardize serialization** — Agent Mesh Phase 8 adds serialization to more types; standardize on `to_dict`/`from_dict` across the codebase first so the new code follows the established pattern

---

## 4. Intelligent Context Upgrade (nearly complete)

> **Status:** ~90% done. Remaining work is observation-gated.
> **Design:** [intelligent_context_upgrade.md](intelligent_context_upgrade.md)

### Done

- Part 1 v1-v2: Edit disambiguation with `context_before`/`context_after` params
- Part 2 v1: Always pin turn 1 (original goal) during compaction
- Part 3: Dropped context notice (PERF-4)

### Remaining (observation-gated)

- Part 1 v3-v4: Prompt tuning — observe LLM usage of context params, adjust prompting
- Part 2 v2-v4: LLM-driven pinning — add after observing v1 contradiction rates

No action needed until you accumulate data from long-horizon coding sessions.

---

## 5. File Splitting (opportunistic)

> **Status:** Partially done. agent_loop.py modularized. Remaining targets are opportunistic.

| Target | Trigger | Status |
|--------|---------|--------|
| `bus.py` → package | When adding new message types | Planned |
| `agent_loop.py` → sections | — | **Done** (Loop Modularization: LoopController, SimulationAdapter, etc.) |
| `router.py` → extract data | Before Multi-LLM Phase 3 | Prereq for Multi-LLM (S4) |
| `definitions.py` → extract prompts | When editing mode prompts | Planned |
| `llm_worker.py` → cleanup | — | **Done** (Track B) |

---

## 6. Remaining Cleanup (opportunistic)

> **Status:** ~90% complete. No dedicated sessions needed.

Items to pick up when you're already touching the file:

| # | Item | When |
|---|------|------|
| 6 | Fix batch scenario break | Before Simulation Agent Phase 4 |
| 8 | Wire PerceptSource protocol | Before Multi-LLM Phase 3 |
| 9 | Any type overuse (Protocols) | Before Multi-LLM Phase 3 |
| 11 | Sim module unit tests | Before Simulation Agent |
| 13 | Stale re-exports in llm_worker | Next time touching llm_worker imports |
| 20 | Remove deprecated localhost_only | Next time touching connection.py |
| 21 | Migrate _reachy → _robot | Next time touching selfy.py |
| 22 | Consolidate RTSP constant | Next time touching skills/ |
| 27 | Consolidate env bool parsing | Anytime (12 files, tedious) |
| 28 | Extract shared velocity calc | Next time touching proprioception/ |
| 29 | Standardize serialization | **Before Agent Mesh Phase 8** |
| 41 | Movement step-clamping helper | Next time touching movement.py |
| 44 | Merge DNActionProposal | Next time touching default_network |

Items to drop (not worth the effort):

| # | Item | Why |
|---|------|-----|
| 2 | Remaining PIPELINE trace | Already silenced from terminal |
| 4 | Metal kernel warnings | Cosmetic, may not be fixable |
| 5 | Variable scoping | Works fine |
| 36 | Singleton boilerplate | High churn, pattern works |
| 37 | Magic number sprawl | Scattered, low ROI |

---

## Recommended Execution Order

### Wave 1: Simulation Agent (current focus, no blockers)

| Step | What | Why |
|------|------|-----|
| 1 | Simulation Agent Phase 1 | SimulationBridge + core tools + `start_simulation_mode()` + CLI |
| 2 | Simulation Agent Phase 2 | Personas, `/cancel` `/new` `/status`, multi-sim sessions |

### Wave 2: Simulation Learning + Multi-LLM Foundation

| Step | What | Why |
|------|------|-----|
| 3 | Simulation Agent Phase 3 | Orchestrator hippocampus, NAc learning, cross-session memory |
| 4 | File Splitting S4 + Cleanup #8, #9 | Prerequisite for multi-LLM Phase 3 |
| 5 | Multi-LLM Phases 1-3 | Local dual-model enables stronger adversarial sim |

### Wave 3: Advanced Simulation + Infrastructure

| Step | What | Why |
|------|------|-----|
| 6 | Simulation Agent Phase 4 | Self-generating tests, regression (benefits from multi-LLM) |
| 7 | Multi-LLM Phases 4-7 | Remote server, tunnel, peer discovery |

### Wave 4: Agent Mesh

| Step | What | Why |
|------|------|-----|
| 8 | Agent Mesh Phases 1-2 | Identity + protocol (no network needed yet) |
| 9 | Agent Mesh Phases 3-8 | Full peer network (needs Multi-LLM Phase 7) |

### Cross-Plan Merge Points

| Merge opportunity | Plans involved | Why merge |
|-------------------|---------------|-----------|
| File Splitting S4 + Multi-LLM Phase 3 | File Splitting + Multi-LLM | S4 (externalize profiles) is a prerequisite for Phase 3 (LaneBackendManager). Do them as one PR. |
| Cleanup #29 + Agent Mesh Phase 8 | Cleanup + Agent Mesh | Both add serialization. Standardize pattern first, then apply it to mesh types. |
