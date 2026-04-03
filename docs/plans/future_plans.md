# Future Plans

Master roadmap for Maxim development. Consolidates all active plans into a single dependency-ordered document. Individual plan files remain as detailed design references.

**Last updated:** 2026-04-03

---

## Status Overview

| Plan | Status | Next step |
|------|--------|-----------|
| Repo Cleanup | **~85% done** | Remaining items are opportunistic |
| Intelligent Context Upgrade | **~90% done** | Observe v2 before adding LLM-driven pinning |
| File Splitting | **Not started** | Do when modifying target files |
| Simulation Agent | **Not started** | Ready to implement (no blockers) |
| Multi-LLM Scaling | **Not started** | Phases 1-3 ready (local multi-model) |
| Agent Mesh | **Not started** | Blocked on Multi-LLM Phases 1-3 |

---

## Dependency Graph

```
Repo Cleanup (done)
    │
    ├── #3 Fix double LLM load ──► Multi-LLM Scaling (prevents pattern duplication)
    │
Intelligent Context (done)
    │
File Splitting (opportunistic)
    ├── Split router.py ──────────► Multi-LLM Scaling Phase 3 (cleaner backend mgmt)
    ├── Split agent_loop.py ──────► Simulation Agent Phase 2 (new execution modes)
    │
Simulation Agent (independent)     Multi-LLM Scaling
    ├── Phase 1: SimulationAgent       ├── Phases 1-3: Local multi-model
    ├── Phase 2: Sim-as-tools          ├── Phases 4-5: Remote server + tunnel
    │   └── optional: multi-LLM ──────►├── Phase 6: LocalBackendSpawner
    │       for adversarial model      ├── Phase 7: PeerRegistry + InferenceRouter ──┐
    └── Future: sleep/dream integration│                                             │
                                       └── Phases 8-9: Metrics + config             │
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
> **Effort:** ~1,500 LOC across 2 phases
> **Design:** [simulation_agent_plan.md](simulation_agent_plan.md)

Replace fixed YAML percept sequences with an LLM-driven adversary/collaborator that adapts to Maxim's responses in real time.

### Phase 1: SimulationAgent Framework (~1,000 LOC)

- `SimulationAgent` class — wraps LLM with persona system prompts
- `DynamicPerceptSource` — implements existing `PerceptSource` protocol, calls SimulationAgent per turn
- `CompletionDetector` — goal achieved / max turns / stalemate / safety violation
- 4 personas: adversarial, cooperative, confused, escalating
- CLI: `maxim --sim agent --persona adversarial --goal "..."`

### Phase 2: Simulation as Agentic Tools (~500 LOC)

- `RunSimulationTool` — agent runs scenario YAML to verify behavior before acting
- `GenerateSimulationTool` — agent generates + runs simulations from descriptions
- `SimulationReflectionTool` — analyze past simulation logs for patterns
- Sleep/dream integration: review sim logs during consolidation, generate edge cases

### Prerequisites (all met)

- PerceptSource protocol, ActionSink, RecordingSink, ScenarioSource
- FearGatedExecutor for safety gating
- LLMRouter.wait_ready() for model readiness
- Simulation logging framework

### Cleanup to absorb before starting

- **#3 Double LLM load** — SimulationAgent needs its own LLM instance; fix the current double-load pattern first so the sim generator shares the main backend rather than duplicating it
- **#6 Batch scenario break** — fix the `break` in cli.py so `--sim scenarios/` processes all files, which the sim agent will need for batch testing
- **#11 Missing sim tests** — write unit tests for sim modules before adding more sim code on top

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

- **#3 Double LLM load** — must fix before Phase 3 (LaneBackendManager replaces the pattern that caused this)
- **#8 Wire PerceptSource protocol** — Phase 3 introduces new typed interfaces; clean up existing Any types first
- **#9 Any type overuse** — same reason; introduce Protocols for Executor/Environment/State alongside new LLM interfaces

### File splitting to absorb

- **S4 Externalize LLM profiles from router.py** — move `_BUILTIN_PROFILES` and `QUANTIZATION_LEVELS` to JSON before Phase 3 adds `LaneBackendManager` to the same file. This is a prerequisite, not optional.

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

> **Status:** Planned. Do when modifying target files, not as standalone work.
> **Design:** [file_splitting_plan.md](file_splitting_plan.md)

| Target | Trigger | Absorbed by |
|--------|---------|-------------|
| `bus.py` → package | When adding new message types | Standalone (low risk) |
| `agent_loop.py` → sections | When adding execution modes | Simulation Agent Phase 2 |
| `router.py` → extract data | Before Multi-LLM Phase 3 | **Multi-LLM Scaling** |
| `definitions.py` → extract prompts | When editing mode prompts | Standalone |
| `llm_worker.py` → cleanup | Done (Track B) | ~~Completed~~ |

---

## 6. Remaining Cleanup (opportunistic)

> **Status:** ~85% complete. No dedicated sessions needed.

Items to pick up when you're already touching the file:

| # | Item | When |
|---|------|------|
| 3 | Share LLM backend in sim | **Before Simulation Agent or Multi-LLM** |
| 6 | Fix batch scenario break | **Before Simulation Agent** |
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

**Option A: Build simulation capabilities first**
1. Fix #3 (double LLM load) + #6 (batch scenario break) + #11 (sim tests)
2. Simulation Agent Phase 1
3. Simulation Agent Phase 2
4. Multi-LLM Phases 1-3 (enables stronger adversarial model for sim)
5. Agent Mesh when multi-agent becomes relevant

**Option B: Build infrastructure first**
1. Fix #3 + S4 (externalize LLM profiles)
2. Multi-LLM Phases 1-3 (local dual-model)
3. Multi-LLM Phases 4-6 (remote server)
4. Simulation Agent (uses multi-LLM for adversarial testing)
5. Multi-LLM Phase 7 → Agent Mesh

**Option C: Breadth-first (sample everything)**
1. Simulation Agent Phase 1 (dynamic sims)
2. Multi-LLM Phases 1-2 (per-lane config)
3. Agent Mesh Phases 1-2 (identity + protocol)
4. Deepen whichever proves most valuable

Option A is best if you're actively developing and testing Maxim's behavior. Option B is best if you're preparing for multi-machine deployment. Option C is best for exploration and design validation.
