# Architecture (Maxim)

> Last reviewed 2026-08-26 against the 1.1.0 tree. The layer-ownership rules below are what `maxim --audit-architecture` enforces (baseline `src/maxim/utils/architecture_baseline.json`, CI-gated). **Known gap:** the Key Modules section predates several live subsystems — `comms/`, `doctor/`, `hivemind/`, `imagination/`, `motion/`, `reactions/`, `roy/`, `tunnel/`, `default_network/`, `console/` — see `docs/reference.md` for the current inventory until this section is refreshed.

Maxim is a bio-inspired cognitive architecture for AI agents. It combines a 5-agent pipeline (Perception, Memory, Exec, Goal, Statistician) with biological memory systems (Hippocampus, ATL, Angular Gyrus, SCN, NAc) and a reactive Default Network. Works headless, in simulation, or connected to a robot.

## High-Level Flow

```
Percepts → 5-Agent Pipeline → Tool Execution → Memory Capture → Bio-System Learning
              ↑ LLM Router (8 cloud providers / 15 cloud profiles + local backends)
```

**Modes of operation:**
- **Headless** — `maxim.run(headless=True)` — pure cognitive loop, no hardware
- **Simulation** — `maxim --sim “goal”` — generative campaigns with orchestrator + AUT
- **DM Campaigns** — `maxim --sim campaign.yaml` — structured multi-agent scenarios
- **Robot** — `maxim --robot reachy_mini` — full hardware I/O (camera, audio, motors)

## Threading / Process Model
- Main agent loop at 2-30Hz + WorkerPool (tier-based lanes: large/medium/small, owned by LLMWorker)
- Hippocampus capture thread (owned + shut down by MemoryHub.on_session_end)
- When connected to hardware:
  - Video capture thread pulls frames from robot and feeds bounded save/latest queues
  - Video writer thread writes a single MP4 per run
  - Audio capture/writer threads for WAV recording + optional chunk transcription
  - Transcription runs in a separate process (faster-whisper)
  - Motor commands funneled through a single executor thread

## Agentic Architecture (Enforcement Rules)

These are **hard architectural rules** for the agentic subsystems in this repo. Violations are architectural bugs and should be caught in docs, code review, and (where possible) CI.

### Layer Ownership (Exclusive)

Paths refer to the `src/maxim/` package layout.

- `src/maxim/agents/`: owns goals, role-specific reasoning, intent generation, and contemplation (local chain-of-thought for plan quality); must **not** execute tools, mutate state, or inspect environments.
- `src/maxim/planning/`: owns plan generation/refinement; must **not** execute actions, select final actions, or mutate state.
  - `adaptive_planner.py`: ADaPT-style lazy planner with deep memory integration. Queries EC (situational similarity), NAc (causal prediction), Hippocampus (spreading activation), ConceptContextBuilder (semantic context + skill discovery), and RetrievalOrchestrator (multi-signal fusion) before proposing plans. Decomposes goals via LLM only on failure or when memory signals caution. Includes `PlanCandidate` and `PlanningContext` dataclasses.
  - `adaptive_policy.py`: 6-dimension scoring policy. Scores plans by NAc value, EC familiarity, concept relevance, delay efficiency, depth penalty, and action cost. Includes `explain_score()` for provenance.
- `src/maxim/planning/decision_engine.py`: owns action selection/arbitration/control flow; must **not** generate plans, execute tools, store memory, or inspect environment internals. Supports both `PlanCandidate` (from `AdaptivePlanner`) and raw `list[dict]` plans (from `TaskPlanner`).
- `src/maxim/planning/policy.py`: owns constraints/guardrails/safety rules; must **not** perform planning, execution, or goal reasoning.
- `src/maxim/tools/`: owns side effects (I/O, network, filesystem, APIs); must **not** do control flow, reasoning, or decision making. Tools have a `timeout` class variable for per-tool execution limits. Coding tools: `EditFileTool` (text-anchor edits with `context_before`/`context_after` disambiguation), `CodeSearchTool` (regex search), `RunTestsTool` (structured test results), `GitDiffTool`, `GitCommitTool`.
  - `introspection.py`: 10 read-only tools exposing biological subsystems to the LLM: `memory_recall` (hippocampus with spreading activation), `predict_outcome` (NAc causal predictions), `causal_links` (cause-effect inspection), `pain_history` (pain signals + fear gate check), `temporal_patterns` (SCN circadian discovery), `energy_status` (resource consumption), `concept_query` (ATL semantic knowledge), `scene_summary` (salience + attention), `similarity_search` (EC multi-modal matching), `system_stats` (aggregate health check).
  - `learned_index.py`: `LearnedToolIndex` — keyword-weighted hashtable for tool relevance scoring. Auto-extracts keywords from tool metadata; learns from execution outcomes via Rescorla-Wagner updates. Keyword discovery on success, surfaced-but-unused decay on rejection. Persisted across sessions. Integrated into `PromptBuilder` to partition tools into CRITICAL (full schema) vs NICE_TO_HAVE (name only) prompt sections.
- `src/maxim/environment/`: owns observation of the world; must **not** perform side effects or execute tools.
- `src/maxim/memory/`: owns storage/retrieval/compression/forgetting; must **not** do decision making or action selection.
  - `hippocampus.py`: Associative memory graph storing complete agentic loops with selective capture, compression, and sleep-based consolidation.
  - `types.py`: EpisodicMemory and CompressedMemory dataclasses.
  - `strategies.py`: Pluggable memory management strategies (AccessBased, ImportanceBased, TemporalAware).
  - `hippocampus_consolidation.py`: `ConsolidationMixin` — wave-based sleep consolidation with path-dependent thresholds (acute/chronic).
  - `context_index.py`: SimilarityIndex - MinHash + LSH for O(1) context/percept similarity lookup.
  - `store.py`: Split persistence protocols (`EpisodicStore`, `CausalStore`, `SemanticStore`) with `File*Store` defaults. Database backends via `[database]` extra.
- `src/maxim/time/`: owns temporal indexing and rhythm tracking.
  - `scn.py`: Suprachiasmatic Nucleus - temporal bin indexing for circadian/weekly/monthly patterns. BoundedBin for capacity-managed bins with significance-based eviction.
  - `temporal_signature.py`: Phase-based temporal fingerprinting.
- `src/maxim/decisions/`: owns causal inference and prediction. Includes `StopReason` enum (10 loop termination reasons) and `ToolErrorKind` enum (7 error classifications on `ToolOutput`) for structured error vocabulary. `CodingReplanContext` captures structured test/build failures for test-driven replanning. Context compaction uses a sliding window with first-turn pinning for long-horizon plans. All config dataclasses are `frozen=True` for thread safety (16 configs frozen including `HippocampusConfig`, `NACConfig`, `DefaultNetworkConfig`, `PainConfig`, `LLMAgentConfig`, etc.). Mutation sites use `dataclasses.replace()`.
  - `nac.py`: Nucleus Accumbens - learns event→outcome relationships via temporal difference learning.
  - `significance.py`: SignificanceWeightLearner - learnable heuristics for memory staging (RPE, novelty, user interaction, etc.).
- `src/maxim/similarity/`: owns multi-modal similarity queries.
  - `ec.py`: Entorhinal Cortex - two distinct query surfaces, easily conflated. **Situation matching** (`find_similar` over `SituationSignature`) is LSH-based approximate nearest neighbour. **Substrate pattern routing** (`pattern_complete_or_separate`, the path the substrate results rest on) is an **exact same-modality centroid scan, O(Nd)** — no LSH, no approximation. Bio-mapping is FUNCTIONAL only: in the brain, pattern separation is a **dentate gyrus** function and pattern completion a **CA3** attractor function; entorhinal cortex is the interface, not the separator. The names here describe what the code decides, not a claimed isomorphism.
  - `semantic.py`: Phase 4 neural semantic embeddings (SentenceTransformer) for deep similarity ("cup" ≈ "mug").
- `src/maxim/proprioception/`: owns body awareness and pain detection.
  - `focus_learner.py`: Rescorla-Wagner learning for movement gain adaptation. Learns optimal gain from overshoot feedback.
  - `movement_tracker.py`: Tracks position history, computes velocity/acceleration metrics.
  - `pain.py`: Detects aversive movement patterns (excessive velocity, thrashing, strain).
- `src/maxim/harm/`: owns predictive harm detection (zero-latency, before execution).
  - `predictor.py`: Abstract HarmPredictor protocol and HarmPrediction dataclass.
  - `registry.py`: HarmRegistry aggregates predictions from multiple domain predictors.
  - `movement.py`: MovementHarmPredictor - predicts velocity harm from action signatures.
  - `joint_limit.py`: JointLimitHarmPredictor - predicts motor stall from workspace limits.
- `src/maxim/energy/`: owns resource expenditure tracking and budgets.
  - `signal.py`: EnergyType enum, EnergySignal dataclass, EnergyBudget.
  - `tracker.py`: Abstract EnergyTracker base class.
  - `llm_tracker.py`: Token-based LLM energy (input/output tokens, latency, model multipliers).
  - ~~`movement_tracker.py`~~: Deleted in the cradle sensorimotor update. `MovementEnergyTracker` was removed; interoceptive drive signals are now handled by `embodiment/sem.py` (`HomeostaticDriveSpec` / `EntropicDriveSpec`).
  - `registry.py`: EnergyRegistry with domain budgets and aggregation.
- `src/maxim/bridges/`: owns cross-system integration between memory and external systems.
  - `spatial_bridge.py`: Location priors from historical object positions.
  - `salience_bridge.py`: Interaction history boosts for salience scoring.
  - `planning_bridge.py`: Plan template retrieval from successful memories.
  - `escalation_bridge.py`: Learned thresholds for when to escalate to human.
  - `pain_bridge.py`: Connects pain detection to NAc for causal learning of aversive patterns.
  - `tool_pain_bridge.py`: Routes tool errors → NAc + SCN, creates CAUSES edges in hippocampus for surprising outcomes (RPE > 0.3), generates Reflexion-style verbal self-critiques stored as episodic memories, and updates `LearnedToolIndex` keyword weights.
- `src/maxim/embodiment/`: owns body definition and motor learning via the SEM (Sensor-Entity-Modulator) protocol. Entities form composable trees; each entity owns sensors (readings), modulators (actions via affordances), and failure modes (pain triggers). Virtual entities use NarrativeModulator (LLM-backed) while hardware entities use real sensor backends.
  - `sem.py`: Core protocol — Entity, Sensor, Modulator, Affordance, FailureMode, FailureTrigger.
  - `spec.py`: YAML loader — parse entity specs, attach backends via `attach_backends()`.
  - `body.py`: Embodiment runtime — failure evaluation, vital drift, body state for prompts.
  - `tool_bridge.py`: Auto-generate tools from entity sensors/modulators (SensorReadTool, ModulatorAffordanceTool, EntitySenseTool).
  - `cerebellum.py`: Forward models (Rescorla-Wagner per-action prediction), motor programs, ProgramRegistry.
  - `motor.py`: MotorProgram, MotorStep, sequence crystallization.
  - `engrams.py`: MotorEngram — contextual links between programs and episodic memories.
  - `llm_backend.py`: LLMSensor, LLMModulator, NarrativeSensor, NarrativeModulator.
  - `program_executor.py`: Step-by-step motor program execution with pain gates.
  - `component_registry.py`: ComponentRegistry — template catalog for reusable SEM entity specs. Multi-path discovery (campaign-local → `~/.maxim/components/` → `_data/components/`). 73 seed components across 7 categories (bodies, creatures, environments, items, npcs, vehicles, weapons). Genre-gated: fantasy, cyberpunk, scifi, horror, historical, modern, devops.
- `src/maxim/mesh/`: owns cooperative peer-to-peer agent networking. Each Maxim instance is sovereign (owns its memories, causal models, behaviors) but can share cooperatively.
  - `identity.py`: AgentProfile — lightweight identity for local multi-agent coordination.
  - `agent_identity.py`: AgentIdentity — extends AgentProfile with hardware capabilities and knowledge statistics for network-level coordination.
  - `message.py`: MeshMessage (24 typed message types), protocol versioning.
  - `peer_channel.py`: PeerChannel — HTTP transport with async send queue, retry with backoff.
  - `peer_registry.py`: PeerRegistry — thread-safe registry, bootstrapped from peer config.
  - `admission.py`: MeshAdmissionControl — per-peer rate limiting, burst detection, escalating gate durations.
  - `knowledge.py`: ExperienceBroker + KnowledgeProvider/Receiver protocol. Built-in adapters: CausalLink (NAc), Reflection (Hippocampus), MotorProgram (Cerebellum).
  - `task_delegation.py`: TaskDelegator + TaskReceiver — goal delegation with loop detection and queue depth checks.
  - `clock.py`: PeerClockEstimator — NTP-lite clock offset estimation for cross-agent temporal coordination.
- `src/maxim/integration/`: owns central coordination.
  - `memory_hub.py`: MemoryHub coordinates all bridges, manages session lifecycle, and wires multi-layer memory (ATL concept extraction, grounding, promotion). Connects 11 bio-systems in production; now also fully wired in simulation mode.
- `src/maxim/state/` (reserved): owns authoritative runtime truth; must **not** contain long-term storage logic or planning.
- `src/maxim/runtime/`: owns agentic orchestration/main execution loop; must **not** do domain reasoning. Includes `RuntimeCapabilities` for hardware detection and graceful degradation (headless mode without robot), and `StreamEvent`/`on_event` callback for fine-grained streaming events from the agent loop. ADaPT-style replan loop: `FailureStrategy.REPLAN` triggers `planner.decompose()` at depth+1. `AgentFactory` creates independent agent instances with isolated subsystems (Hippocampus, NAc, ATL, MemoryHub). `AgentPool` orchestrates concurrent multi-agent execution with `LocalMessageBus` for inter-agent communication.
- `src/maxim/embodied_runtime/`: owns robot orchestration/main loop (Reachy capture/inference/control); must **not** do agentic decision making. `ConnectionState` enum with callback system for runtime capability degradation/restoration on robot disconnect/reconnect. `_run_headless_loop()` for event-driven operation without media capture. (Renamed from `conscience/` in v1.0.0 to better describe contents — robot mixin stack, not safety enforcement.)

### Absolute Separation Rules
- Agents never call tools directly.
- Environments never cause side effects.
- Memory never selects actions.
- Planning never mutates state.
- Action selection happens in one place only.
- State is the single source of truth.
- No component may bypass state.

### One-Way Dependency Graph

Dependencies must flow strictly downward (reverse imports are forbidden):

Agents → Planning → Decision Engine → Runtime → Executor → Tools → Environment → State → Memory

### Testability Rule

Each layer must be independently mockable:
- Tools can be no-op or simulated.
- Environments can be simulated.
- Memory can be in-memory.
- Agents can run without side effects.

If a component cannot be tested in isolation, the architecture is violated.

## Key Modules
- `src/maxim/cli.py`: primary CLI entrypoint (`maxim` console script).
- `scripts/main.py`: legacy checkout entrypoint (delegates to `maxim.cli`).
- `src/configs/`: version-controlled config templates and notes.
- `src/maxim/embodied_runtime/selfy.py`: `Maxim` robot orchestrator class, composed from six mixins:
  - `connection.py` (`ConnectionMixin`): Reachy SDK connection lifecycle
  - `vision_stream.py` (`VisionStreamMixin`): vision capture and segmentation pipeline
  - `agentic_runtime.py` (`AgenticRuntimeMixin`): agentic runtime bootstrap and lifecycle
  - `movement.py` (`MovementMixin`): motor command helpers
  - `input_handlers.py` (`InputHandlerMixin`): CLI/keyboard/voice input routing
  - `media_loop.py` (`MediaLoopMixin`): video/audio recording and display loop
  - `workers.py`: module-level worker functions (video writer, audio writer, transcription)
- `src/maxim/agents/`: agent interfaces + implementations (reasoning/intent, no side effects).
  - `modality.py`: `SensoryModality` enum, `SensoryTag` dataclass — typed percept classification.
  - `sensory_gate.py`: `SensoryGate` — entity-modulated filtering of sensory input before pipeline processing.
  - Extracted modules (re-exported from `llm_worker.py` for backward compatibility): `llm_types.py`, `llm_context.py`, `prompt_budgeter.py`, `llm_fallback.py`, `prompt_builder.py`.
- `src/maxim/planning/`: planning + policy + decision engine (agentic action selection).
  - `plan_dashboard.py`: Bus-driven `ACTIVE_PLAN.md` writer for workspace visibility.
  - `plan_logger.py`: Append-only `history.md` plan event log with async write queue.
- `src/maxim/tools/`: tool implementations (side effects).
- `src/maxim/environment/`: environment interfaces/implementations (observations, no side effects).
- `src/maxim/memory/`: memory interfaces/implementations (storage/retrieval, no decisions).
  - `hippocampus.py`: Hash index for O(1) context lookup + associative memory of agentic loops.
  - `types.py`: EpisodicMemory and CompressedMemory dataclasses.
  - `strategies.py`: Pluggable memory management strategies.
- `src/maxim/bridges/`: cross-system integration bridges.
- `src/maxim/integration/`: MemoryHub coordinator.
- `src/maxim/evaluation/`: lightweight evaluators/metrics for tools, plans, and agent intents.
- `src/maxim/runtime/`: agentic runtime loop + bootstrap wiring (decision engine → executor → tools).
- `src/maxim/inference/`: observation/control functions (vision target selection, motor control, etc.).
- `src/maxim/models/vision/`: perception models (Vision engine: RTMDet-m + RTMPose-m by default, YOLOv8 optional via `[yolo]` extra).
- `src/maxim/models/movement/`: MotorCortex model (ConvNeXt-Tiny head-movement prediction).
- `src/maxim/models/audio/`: Whisper wrapper (transcription backend).
- `src/maxim/models/language/`: optional local LLM routing (transcript → agentic action).
- `src/maxim/interactive/`: universal prompt protocol and rich terminal display.
  - `prompts.py`: PromptRequest, PromptHandler ABC, PromptType enum. Every user interaction flows through this protocol (DM choices, architect interviews, freeform chat, confirmations).
  - `display.py`: Rich-based split-panel terminal UI with scrolling agent log, status bar, and input area. Graceful degradation without `rich`.
  - `dm_display.py`: DM-specific display extensions (encounter info panels, character sheet).
- `src/maxim/simulation/dm_runtime.py`: DM campaign runtime — multi-agent campaign execution. NPC agents have real memory (Hippocampus, NAc) via AgentFactory. NPCs witness outcomes, remember encounters, and adapt dialogue.
- `src/maxim/_data/encounters/`: Bundled encounter templates — 8 seed YAML files across 4 categories (combat, exploration, puzzle, social). Accessible via the `browse_encounters` tool (`simulation/tools_dm.py::BrowseEncountersTool`) in DM campaign mode.
- `src/maxim/simulation/entity_designer.py`: EntityDesigner — LLM-driven SEM spec generation from natural language descriptions. Uses ComponentRegistry templates as bases; generates only the delta.
- `src/maxim/memory/store.py`: Split persistence protocols — `EpisodicStore`, `CausalStore`, `SemanticStore`. Each protocol matches its subsystem's query patterns. `File*Store` defaults wrap current JSON persistence. Database implementations (PostgreSQL + pgvector) provided by `[database]` extra.
- `src/maxim/_data/`: Bundled seed data shipped with the package.
  - `components/`: SEM entity templates (bodies, creatures, environments, npcs, weapons).
  - `encounters/`: Encounter templates (combat, exploration, puzzle, social).
  - `prompts/`: System prompt templates.
  - `templates/`: Report/output templates.
- `src/maxim/data/`: camera/audio utilities and file outputs.
- `src/maxim/utils/`: config, logging, plotting, filesystem helpers (and reusable small helpers).
  - `paths.py`: Data path resolution — bundled `_data/` for read-only seed data, `~/.maxim/` for user-generated data (memory, sessions, benchmarks, config).

## Output Layout (Default)

User data now lives at `~/.maxim/` by default (resolved via `utils/paths.py`). Legacy `data/` paths still work but are deprecated.

- `~/.maxim/memory/`: Persistent memory files (hippocampus, nac, scn, atl)
- `~/.maxim/sessions/`: Per-session recordings and replays
- `~/.maxim/benchmarks/`: Benchmark run reports
- `~/.maxim/components/`: User-created SEM entity templates
- `~/.maxim/encounters/`: User-created encounter templates
- `~/.maxim/config/`: User config overrides
- `~/.maxim/models/`: Downloaded LLM/TTS/vision models
- `data/videos/`: `reachy_video_<YYYY-MM-DD_HHMMSS>.mp4`
- `data/audio/`: `reachy_audio_<YYYY-MM-DD_HHMMSS>.wav` and optional `audio/chunks/*.wav`
- `data/transcript/`: `reachy_transcript_<YYYY-MM-DD_HHMMSS>.jsonl`
- `data/training/`: `motor_training_set.jsonl` (trainable samples + user marks)
- `data/agents/<STATE_NAME>/runtime/`: `state_<run_id>.json` (agentic runtime state snapshots; defaults to `agent_name` unless an agent sets `state_name`)
- `data/models/MotorCortex/`: MotorCortex checkpoint + training artifacts
- `data/planning/`: Plan JSONs, active plan pointer (internal state, configurable via `LongHorizonConfig.plan_persistence_path`)
- `data/util/llm.json`: local LLM config (created on install with SmolLM 1.7B as default)

## Workspace (`.maxim_workspace/`)

The workspace is the user-facing surface for Maxim's planning and note-taking. It lives in the CWD (gitignored) and is always writable regardless of operational mode. `data/` (aka `--home-dir`) owns internal state; `.maxim_workspace/` gives visibility.

### Directory Structure

```
.maxim_workspace/
├── drafts/     - Code drafts, proposed edits, work-in-progress
├── notes/      - LLM working notes, observations
│   └── context.md   - Persistent scratchpad (always in prompt)
├── plans/      - Plan visibility
│   ├── ACTIVE_PLAN.md   - Live dashboard (auto-updated by PlanDashboard)
│   ├── history.md       - Append-only plan log (auto-updated by PlanLogger)
│   └── history_archive/ - Rolled history (when history.md exceeds 500 lines)
└── scratch/    - Temporary working files
```

### Working Notes (`notes/context.md`)

A persistent LLM scratchpad — always read into the prompt via `StructuredContext.working_notes`. The LLM writes to it via `write_file` and sees it every cycle. Used for pinning important context that similarity-based recall might miss.

- Size cap: 2000 chars. Truncation warning injected when exceeded.
- User-visible and editable — the user can correct or clear notes at any time.
- Not committed to Hippocampus on write — consequential notes graduate via short-term memory staging.

### Plan Dashboard (`plans/ACTIVE_PLAN.md`)

Auto-generated by `PlanDashboard` from bus events (`PlanCreated`, `PhaseStarted`, `PhaseCompleted`, `PlanCompleted`, `PlanReplanRequested`, `PlanRestored`). Shows current plan status, phase checklist, and timing. Cleared on plan completion. Background thread with event coalescing prevents blocking `PlanManager`'s RLock.

### Plan History (`plans/history.md`)

Append-only log maintained by `PlanLogger`. One-line timestamped entries for all plan lifecycle events. Background write queue for async safety. Rolls at 500 lines to `history_archive/`.

### Workspace Context in Prompt

`StructuredContext.workspace_files` contains a scan of user-facing artifacts (up to 10 most recently modified). Plan system files (`ACTIVE_PLAN.md`, `history.md`) and `notes/context.md` are filtered out (already represented elsewhere in the prompt). The LLM can reference workspace artifacts during plan decomposition.

### Filesystem Permissions

`ModeFilesystemConfig.workspace_write_always = True` — all operational modes can write to the workspace without approval. Read access is unrestricted. There is no `delete_file` tool (by design); the LLM overwrites files with empty content instead.

## LLM Configuration

The LLM subsystem uses a JSON config file that persists user preferences across reinstalls.

### Config File Priority

The system searches for config in this order (first found wins; one shared
list in `models/language/config.py::_llm_config_candidates`):
1. `MAXIM_LLM_CONFIG` environment variable path
2. `~/.maxim/config/llm.json` (the documented user location — the CLI,
   downloader, and docs all point here)
3. `./data/util/llm.json` (current working directory)
4. `./llm.json` (current working directory)
5. Repo root `data/util/llm.json`
6. Repo root `llm.json`

When the winning file shadows a lower-priority one that also exists, an
INFO line says which file won (once per process). The downloader's
`enable_llm_config` writes to the same first-existing candidate, so the
file a session reads is always the file the downloader updated.

### Default Configuration

On first install, `data/util/llm.json` is created with:
- **Default model**: `smollm-1.7b-instruct` (~1.1GB, smallest available)
- **Enabled**: `true` (ready to use immediately)
- **Preserved on reinstall**: Existing config is not overwritten

### Switching Models

```bash
# List available models
python -m maxim.models.download --list

# Download a different model
python -m maxim.models.download --llm mistral-7b-instruct-v0.2

# Edit data/util/llm.json and change "profile" to the new model name
```

### Key Config Fields

| Field | Description |
|-------|-------------|
| `enabled` | `true`/`false` - whether LLM is active |
| `profile` | Active model profile name |
| `profiles` | Dict of available model configurations |
| `max_tokens` | Default max response tokens |
| `temperature` | Sampling temperature (0.0 = deterministic) |
| `mode_response_config` | Per-mode token limits and response formats |

### Environment Variable Overrides

All config values can be overridden via environment variables:
- `MAXIM_LLM_ENABLED`, `MAXIM_LLM_PROFILE`, `MAXIM_LLM_BACKEND`
- `MAXIM_LLM_MODEL_PATH`, `MAXIM_LLM_N_CTX`, `MAXIM_LLM_MAX_TOKENS`
- `MAXIM_LLM_TEMPERATURE`, `MAXIM_LLM_N_GPU_LAYERS`, etc.

## Cognitive Memory Systems

Six biologically-inspired systems collaborate to give Maxim memory, temporal awareness, reward prediction, similarity matching, semantic concepts, and algebraic reasoning. In the brain, these are anatomically distinct regions that communicate via neural pathways; in Maxim, each lives in its own package and they coordinate through the MemoryHub.

### System Architecture

```
  ┌─────────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌──────────────┐
  │ Hippocampus │  │    SCN    │  │    NAc    │  │    EC     │  │    ATL    │  │AngularGyrus  │
  │  (memory/)  │  │  (time/)  │  │(decisions/)│  │(similarity/)│ │  (memory/)│  │   (math/)    │
  │  Episodic   │  │ Temporal  │  │  Reward   │  │ Similarity│  │ Semantic  │  │ Algebraic    │
  │  Memory     │  │  Rhythm   │  │ Prediction│  │  Matching │  │ Concepts  │  │  Memory      │
  └──────┬──────┘  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘  └──────┬───────┘
         │               │              │               │              │               │
         └───────────────┼──────────────┼───────────────┼──────────────┼───────────────┘
                         │              │               │              │
                ┌────────┴──────────────┴───────────────┴──────────────┘
                │                    MEMORY HUB
                │             (integration/memory_hub.py)
                └────────────────┬───────────────┘
                                 │
     ┌─────────┬─────────┬──────┼──────┬─────────┬─────────┐
     ▼         ▼         ▼      ▼      ▼         ▼         ▼
  Spatial  Salience  Planning  Escal  Fear     Pain     ToolPain
   Bridge   Bridge    Bridge  Bridge Bridge   Bridge    Bridge
```

### Coordinated Systems

| System | Brain Region | Purpose | Key Features |
|--------|-------------|---------|--------------|
| **Hippocampus** | Medial temporal lobe | Episodic memory | Selective capture, associative graph, sleep consolidation |
| **ATL** (Anterior Temporal Lobe) | Temporal pole | Semantic concepts | Concept extraction, grounding, pattern completion, promotion |
| **AngularGyrus** | Parietal-temporal junction | Algebraic memory | Mathematical reasoning, quantity tracking |
| **SCN** (Suprachiasmatic Nucleus) | Hypothalamus | Temporal rhythm indexing | 24h/7d/monthly bins, coupled oscillator, pattern detection |
| **NAc** (Nucleus Accumbens) | Ventral striatum | Causal inference | Event→outcome learning, reward prediction |
| **EC** (Entorhinal Cortex) | Medial temporal lobe (adjacent to hippocampus) | Similarity queries | LSH + neural semantic embeddings for situation matching; **exact O(Nd) centroid scan** for substrate pattern routing. Separation/completion are DG/CA3 functions biologically — FUNCTIONAL naming, not isomorphism |

### Bridges

| Bridge | Connects | Before | After |
|--------|----------|--------|-------|
| **SpatialMemoryBridge** | Hippocampus ↔ SpatialMap | Workspace rebuilt each session | Multi-session object location priors |
| **SalienceMemoryBridge** | Hippocampus ↔ SalienceNetwork | Pure novelty/recency | Interaction history boosts |
| **PlanHistoryBridge** | Hippocampus ↔ NAc | Plans from scratch | Successful template retrieval |
| **EscalationLearningBridge** | Hippocampus ↔ SCN/NAc | Fixed thresholds | Per-goal, per-time learned thresholds |
| **FearCircuitBridge** | Hippocampus ↔ FearAgent ↔ NAc | No learned risk patterns | Memory-informed risk assessment (also queries EC via associative graph for contextual history) |
| **PainCircuitBridge** | PainDetector ↔ NAc | No movement-pain learning | Learned action→pain associations |
| **ToolPainBridge** | Tool errors ↔ NAc ↔ FearAgent | No tool-failure learning | Cognitive pain from tool errors via Rescorla-Wagner |

### Selective Capture

Not every loop is captured - only "interesting" ones:
- User input (CLI or speech)
- High novelty (> 0.7 threshold)
- High salience (> 0.7 threshold)
- Goal changes
- Failures (for learning)
- Periodic checkpoints

### Sleep Consolidation

A periodic process (call `hippocampus.sleep()` or `hub.on_session_end()`) manages memory. Note: `sleep()` is the top-level method that handles compression, removal, and preservation; it internally calls `consolidate()` for long-term promotion. `consolidate()` can also be called standalone to promote specific memories without running full sleep consolidation:
1. **Long-Term Promotion**: Important memories marked for preservation
2. **Compression**: Old EpisodicMemory → CompressedMemory (reduces ~2.5KB → ~200 bytes)
3. **Removal**: Memories not accessed in 1 week (configurable) are removed
4. **Preservation**: High-access, high-centrality, or user-interaction memories are protected
5. **Temporal Clustering**: SCN-aware consolidation keeps temporal coverage

### Memory Strategies

The strategy pattern allows flexible memory management:
- `AccessBasedStrategy`: Recency + frequency of access
- `ImportanceBasedStrategy`: Novelty + success + user interaction
- `CompositeStrategy`: Weighted combination
- `TemporalAwareStrategy`: SCN-integrated with sole-representative and rhythmic-pattern boosts

Custom strategies can be implemented by subclassing `MemoryStrategy`.

### Access Tracking

Every memory tracks:
- `created_at`: Original capture time
- `accessed_at`: Last retrieval time (updated by `recall()`, `get()`, etc.)
- `access_count`: Total retrieval count
- `long_term`: Boolean flag for long-term promotion
- `consolidated_at`: When promoted to long-term (if applicable)

This enables biological-like memory decay and reinforcement.

### Session Lifecycle

```python
# Create core systems (use canonical builder — raw MemoryHub() raises TypeError)
hub = build_memory_hub(
    hippocampus=hippocampus, scn=scn, nac=nac, ec=ec,
    spatial=spatial_map, salience=salience_network,
    agent_id="my_agent",
)

# Start session (restores priors from memory)
hub.on_session_start()

# ... agent loop runs, capturing memories ...

# End session (runs sleep consolidation)
hub.on_session_end()
```

## Pain Detection and Harm Prediction

A two-tier system for detecting and preventing harmful robot behaviors.

### Tier 1: Predictive Harm (Zero Latency)

Analyzes action parameters BEFORE execution to predict harmful outcomes:

```
Action Request → HarmRegistry.predict_all() → Gate Decision
                         ↓
    ┌─────────────────────────────────────────────┐
    │  MovementHarmPredictor (velocity analysis)  │
    │  JointLimitHarmPredictor (workspace bounds) │
    │  (Future: LLMTimeoutPredictor, etc.)        │
    └─────────────────────────────────────────────┘
```

**Key components:**
- `HarmPredictor`: Abstract protocol for domain-specific predictors
- `HarmRegistry`: Central aggregator, returns worst-case prediction
- `HarmPrediction`: Contains category, intensity, confidence, mitigation

### Tier 2: Reactive Pain (Learned)

Detects aversive patterns from proprioceptive signals and learns to avoid them:

```
Position Updates → MovementTracker → PainDetector → PainCircuitBridge → NAc Learning
                                          ↓
                   FearAgent.review_action() ← NAc.predict()
```

**Pain types detected:**
- `EXCESSIVE_VELOCITY`: Movement too fast (> 100 deg/sec default)
- `DIRECTION_THRASHING`: Rapid back-and-forth reversals
- `EXCESSIVE_ACCELERATION`: Sudden speed changes
- `SUSTAINED_STRAIN`: Prolonged near-limit positions
- `TOOL_FAILURE`: Tool execution returned an error
- `TOOL_TIMEOUT`: Tool exceeded its execution timeout
- `TOOL_INVALID_INPUT`: Tool received malformed parameters
- `TOOL_SUSTAINED`: Repeated failures from the same tool

ToolPainBridge routes tool errors through PainDetector → NAc, creating cognitive pain signals that the FearAgent uses to learn which tools fail in which contexts.

### Integration with FearAgent

```python
# In FearAgent.review_action():
harm_prediction = harm_registry.predict_worst(action_type, action_params)
if harm_prediction and harm_prediction.risk_score >= 0.7:
    findings.append(Finding(
        category=DangerCategory.RESOURCE_EXHAUSTION,
        description=f"Predicted harm: {harm_prediction.reason}",
        severity=RiskLevel.MEDIUM,
    ))
```

## Energy Tracking System

Monitors resource expenditure across subsystems to enable energy-aware decisions.

### Energy Types

| Type | Description | Source |
|------|-------------|--------|
| `LLM_TOKENS` | Token-based energy (input + output) | LLMEnergyTracker |
| `LLM_LATENCY` | Time waiting for LLM response | LLMEnergyTracker |
| `MOTOR_COMMAND` | Energy for movement execution | (EnergyRegistry / not actively sourced; `MovementEnergyTracker` was deleted) |
| `VISION_INFERENCE` | Vision model inference | (Future) |
| `AUDIO_PROCESSING` | Audio transcription/TTS | (Future) |

### Model-Specific Multipliers

```python
model_multipliers = {
    "claude-3-haiku": 0.5,      # Efficient
    "claude-3-sonnet": 1.0,     # Baseline
    "claude-3-opus": 2.0,       # Expensive
    "claude-opus-4-5": 2.5,     # Most expensive
    "local": 0.2,               # Local inference is cheap
}
```

### Energy Budgets

Each domain has a budget with capacity and recharge rate:

```python
budget_configs = {
    "llm": {"capacity": 1000.0, "recharge_rate": 10.0},
    "movement": {"capacity": 500.0, "recharge_rate": 5.0},
}
```

### NAc Integration

Energy tracking now wires directly into NAc for metabolic cost learning via the agent loop (`runtime/agent_loop.py`):
- High energy expenditure → NEGATIVE valence → NAc learns to predict
- Low energy expenditure → POSITIVE valence → Efficient actions preferred
- Future actions can be gated based on predicted energy cost

## Persistence System

Many learning components persist their state across sessions, enabling continuous improvement over time.

### Persistent Components

| Component | File | Persists | Auto-Save Interval |
|-----------|------|----------|-------------------|
| **FocusLearner** | `data/util/focus_learner.json` | Directional gains (h+/h-/v+/v-), sample stats | 60s |
| **WorkspaceBoundsLearner** | `data/util/learned_bounds.json` | Learned workspace limits | 60s |
| **EscalationLearningBridge** | `data/util/escalation_learning.json` | Per-goal/hour thresholds, escalation records | 60s |
| **FearCircuitBridge** | `data/util/fear_learning.json` | Risk adjustments, events, category stats | 60s |
| **AdaptiveThresholdController** | `data/util/adaptive_thresholds.json` | Novelty/salience thresholds, history | 60s |
| **NAc** | `data/util/nac_state.json` | Causal links, event outcomes | 60s |
| **SCN** | `data/util/scn_state.json` | Temporal bins, rhythm patterns | 60s |
| **Hippocampus** | `data/util/hippocampus.json` | Episodic memories | On session end |
| **PainDetector** | `data/util/pain_detector.json` | Pain event history | 60s |
| **PlanDashboard** | `.maxim_workspace/plans/ACTIVE_PLAN.md` | Active plan status | On bus event |
| **PlanLogger** | `.maxim_workspace/plans/history.md` | Plan lifecycle log | On bus event |

### Persistence Pattern

All persistent components follow a consistent pattern:

```python
class LearnableComponent:
    def __init__(self, persist_path: str = "data/util/component.json"):
        self.persist_path = persist_path
        self.auto_save_interval = 60.0
        self._last_save_time = time.time()

        # Auto-load on init
        if os.path.exists(persist_path):
            self.load(persist_path)

    def save(self, path: str | None = None) -> bool:
        """Save state to JSON file."""
        ...

    def load(self, path: str | None = None) -> int:
        """Load state from JSON file. Returns count of items loaded."""
        ...

    def _maybe_auto_save(self) -> None:
        """Auto-save if interval has elapsed."""
        if time.time() - self._last_save_time >= self.auto_save_interval:
            self.save()
```

### Clearing Persistent Memory

Use the `--clear-memory` CLI flag to reset learning:

```bash
# Clear all persistent memory
maxim --clear-memory

# Clear specific types (comma-separated)
maxim --clear-memory focus
maxim --clear-memory focus,bounds,fear
maxim --clear-memory escalation,threshold
```

**Available memory types:**
- `focus` - FocusLearner gains
- `bounds` - Workspace bounds
- `escalation` - Escalation thresholds
- `fear` - Fear/risk adjustments
- `threshold` - Adaptive thresholds
- `nac` - NAc causal links
- `scn` - SCN temporal patterns
- `hippo` - Hippocampus memories
- `pain` - Pain detector history
- `semantic` - Semantic embeddings (Phase 4)
- `all` - Clear everything

### File Format

All persistence files use JSON with a version field for forward compatibility:

```json
{
  "version": 1,
  "saved_at": 1707235200.0,
  "data": { ... },
  "config": { ... }
}
```

## Contemplation (Local Chain-of-Thought)

ExecAgent includes a contemplation loop that improves plan quality for complex goals when native extended thinking (Anthropic) is unavailable:

```
_propose_goal()
  ├── LLM draft (any provider)
  ├── Complexity gate: 2+ sub_goals or HIGH/CRITICAL priority
  ├── Contemplation (standard: critique → refine, or fast: combined)
  │   ├── Preemption: only urgent percepts interrupt
  │   └── Fallback: any failure returns original draft
  ├── NAc outcome tracking: learns when contemplation helps
  └── Adaptive thresholds: auto-tunes gates from NAc data
```

Config: `data/util/llm.json` → `contemplation` key. See [AGENTS.md](AGENTS.md#contemplation-system-execagent-local-chain-of-thought) for full documentation.

## Architecture Audit

The `--audit-architecture` CLI flag runs an AST-based import validator that checks all source files against the layer dependency rules above. It reports any reverse imports (e.g., tools importing from agents) and exits with a non-zero code on violations. Useful for CI and pre-commit validation.

## Invariants
- Control loop must not perform heavy disk I/O.
- Recording uses backpressure (bounded queues) rather than intentional dropping when "record everything" is requested.
- Public import paths should remain stable, or be preserved via re-exports when refactoring.
- Reusable helpers should live at module scope (prefer `src/maxim/utils/`) instead of being defined inside hot-loop functions.
- Memory management (sleep consolidation) should run outside the hot loop (during idle time or scheduled intervals).
