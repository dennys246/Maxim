# Future Plans

Master roadmap for Maxim development. Consolidates all active plans into a single dependency-ordered document. Individual plan files remain as detailed design references.

**Last updated:** 2026-04-03

---

## Status Overview

| Plan | Status | Next step |
|------|--------|-----------|
| Agentic Loop Modularization | **Not started** | Phase 0 ready (bug fix + helpers) |
| Repo Cleanup | **~85% done** | Remaining items are opportunistic |
| Intelligent Context Upgrade | **~90% done** | Observe v2 before adding LLM-driven pinning |
| File Splitting | **Not started** | Do when modifying target files |
| Simulation Agent | **Not started** | Ready to implement (no blockers) |
| Multi-LLM Scaling | **Not started** | Phases 1-3 ready (local multi-model) |
| Agent Mesh | **Not started** | Blocked on Multi-LLM Phases 1-3 |

---

## Dependency Graph

```
Agentic Loop Modularization (new)
    ├── Phase 0: Extract helpers + fix bug ◄── start here (safe, no behavior change)
    ├── Phase 1: Type the state bag
    ├── Phase 2: Extract phase methods ◄── critical path
    ├── Phases 3-5: Consolidate loops, isolate sim, fix followup
    └── Phases 6-9: Freeze context, error handling, bus safety, DN decoupling
        │
        ├── Phase 4 (isolate sim) ────► Simulation Agent Phase 2 (cleaner integration)
        │
Repo Cleanup (done)
    │
    ├── #3 Fix double LLM load ──► Multi-LLM Scaling (prevents pattern duplication)
    │
Intelligent Context (done)
    │
File Splitting (opportunistic)
    ├── Split router.py ──────────► Multi-LLM Scaling Phase 3 (cleaner backend mgmt)
    ├── Split agent_loop.py ──────► Superseded by Loop Modularization Phase 2
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

## 1. Agentic Loop Modularization

> **Status:** Not started. Phase 0 ready (safe, no behavior change).
> **Effort:** ~1,500 LOC refactoring across 10 phases
> **Design:** [agentic_loop_modularization_plan.md](agentic_loop_modularization_plan.md)

Refactor `run_agentic_loop()` from a 2,300-line monolithic function into a testable `LoopController` class. Fixes a correctness bug (`set.pop()` evicting arbitrary elements), eliminates 7x outcome-recording duplication, types the state bag, isolates simulation concerns, and improves error handling and bus safety.

### Phase 0: Extract helpers + fix bug (start here)
- `_record_outcome()` helper (consolidate 7 copy-pasted blocks)
- Fix `processed_cli_inputs` from `set` to `deque(maxlen=20)` (correctness bug)
- `_execute_and_record()` helper (unify agent fallback and LLM proposal paths)
- Cache `_get_all_tools()` per iteration

### Phases 1-2: Type state + extract phase methods (critical path)
- `LoopState` dataclass replaces stringly-typed `state.data["pending_*"]` keys
- `LoopController` class with `observe()`, `parse_input()`, `check_proposals()`, `execute_proposal()`, `submit_to_llm()` etc.
- Main loop body drops from ~2,300 lines to ~40 lines

### Phases 3-5: Consolidate, isolate, improve (independent)
- Phase 3: Make `run_agent_loop` a thin wrapper over `LoopController(sync_mode=True)`
- Phase 4: `SimulationAdapter` / `NullSimulationAdapter` replaces ~20 inline `if percept_source` guards
- Phase 5: `ActionFollowup` dataclass, first-class followup in LLMRequest, true parallel execution

### Phases 6-9: Polish (independent)
- Phase 6: `StructuredContext` frozen dataclass with builder pattern
- Phase 7: Error severity tiers + `@resilient` decorator replacing blanket `except: pass`
- Phase 8: Bus handler timeout warnings + optional async delivery for slow subscribers
- Phase 9: `DefaultNetworkController` extracts DN lifecycle from loop body

### Relationship to File Splitting plan
This supersedes the "Split agent_loop.py" item from the File Splitting plan. The modularization achieves the same goal (smaller, testable units) with a cleaner class-based approach.

---

## 2. Simulation Agent

> **Status:** Not started. No blockers — ready to implement.
> **Effort:** ~1,200 LOC across 4 phases
> **Design:** [simulation_agent_plan.md](simulation_agent_plan.md)

A second Maxim instance (the orchestrator) drives the agent-under-test through the full agentic pipeline — task decomposition, tool chaining, memory, and planning. The orchestrator's tools operate on the AUT via a `SimulationBridge` (ConversationalSource + RecordingSink). User stays in simulation mode until `/cancel` or `/new`.

### Phase 1: Bridge + Core Tools (~400 LOC)

- `SimulationBridge` — bidirectional channel wrapping existing ConversationalSource + RecordingSink
- `InjectPerceptTool`, `ObserveActionsTool`, `WaitForResponseTool` — orchestrator's tools
- `CheckCompletionTool` — LLM-based evaluation of whether simulation goal is met
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

### Cleanup to absorb before starting

- **#3 Double LLM load** — orchestrator and AUT each need an LLM instance; fix the current pattern first
- **#6 Batch scenario break** — needed for regression testing in Phase 4
- **#11 Missing sim tests** — write unit tests for sim modules before adding more code on top

---

## 3. Multi-LLM Scaling

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

## 4. Agent Mesh

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

## 5. Intelligent Context Upgrade (nearly complete)

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

## 6. File Splitting (opportunistic)

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

## 7. Remaining Cleanup (opportunistic)

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

Given simulation agent is the active focus, here's the optimal merged sequence:

### Wave 1: Foundations (unblocks everything)

| Step | What | Why |
|------|------|-----|
| 1a | Cleanup #3 (double LLM load) | Blocks sim agent shared backend + multi-LLM |
| 1b | Loop Modularization Phase 0 | Fixes correctness bug, extracts helpers. Safe, no behavior change. |

These are independent — do in parallel or either order.

### Wave 2: Simulation Agent (current focus)

| Step | What | Why |
|------|------|-----|
| 2 | Simulation Agent Phase 1 | SimulationBridge + core tools + `start_simulation_mode()` + CLI |
| 3 | Simulation Agent Phase 2 | Personas, `/cancel` `/new` `/status`, multi-sim sessions |

The sim plan is already well-structured and doesn't need the loop modularization as a prerequisite — it uses the existing `percept_source`/`action_sink` interface directly.

### Wave 3: Loop Cleanup (interleave with sim stabilization)

| Step | What | Why |
|------|------|-----|
| 4 | Loop Modularization Phases 1-2 | Type state bag, extract `LoopController`. Critical path for all later work. |
| 5 | Loop Modularization Phase 4 | `SimulationAdapter` replaces ~20 inline guards. Cleaner AUT integration. |

### Wave 4: Deepen Both (sim learning + infrastructure)

| Step | What | Why |
|------|------|-----|
| 6 | Simulation Agent Phase 3 | Orchestrator hippocampus, NAc learning, cross-session memory |
| 7 | File Splitting S4 + Cleanup #8, #9 | Prerequisite for multi-LLM Phase 3 |
| 8 | Multi-LLM Phases 1-3 | Local dual-model enables stronger adversarial sim |
| 9 | Simulation Agent Phase 4 | Self-generating tests, regression (benefits from multi-LLM) |

### Wave 5: Future

| Step | What | Why |
|------|------|-----|
| 10 | Loop Modularization Phases 3, 5-9 | Polish: consolidate loops, freeze context, bus safety, etc. |
| 11 | Multi-LLM Phases 4-7 | Remote server, peer discovery |
| 12 | Agent Mesh | Blocked on multi-LLM Phase 7 |

### Cross-Plan Merge Points

These stages from different plans can be combined because they touch the same code:

| Merge opportunity | Plans involved | Why merge |
|-------------------|---------------|-----------|
| Loop Mod Phase 4 + Sim Agent Phase 2 | Loop Mod + Sim Agent | Both touch agent_loop.py's sim integration. Extract `SimulationAdapter` while building orchestrator AUT wiring. |
| Loop Mod Phase 9 + Cleanup #44 | Loop Mod + Cleanup | Both touch DefaultNetwork. Extract `DefaultNetworkController` and merge `DNActionProposal` in one pass. |
| File Splitting S4 + Multi-LLM Phase 3 | File Splitting + Multi-LLM | S4 (externalize profiles) is a prerequisite for Phase 3 (LaneBackendManager). Do them as one PR. |
| Loop Mod Phase 7 + Cleanup #27 | Loop Mod + Cleanup | Both address code quality in the same area. `@resilient` decorator + env bool consolidation. |
| Cleanup #29 + Agent Mesh Phase 8 | Cleanup + Agent Mesh | Both add serialization. Standardize pattern first, then apply it to mesh types. |
