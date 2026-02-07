# Architecture (Maxim)

Maxim orchestrates Reachy Mini data capture (camera + mic), perception/inference, optional learning, and motor control.

## High-Level Flow
Capture (Reachy) → Writers (video/audio) → Inference (vision/audio) → Control (motor) → Persist (models/history)

## Threading / Process Model
- Video capture thread pulls frames from Reachy and feeds:
  - a bounded “save” queue for the video writer (records everything; blocks when backpressured)
  - a “latest” queue for observation/control (keeps the loop responsive)
- Video writer thread writes a single MP4 for the run.
- Audio capture thread pulls samples and feeds a bounded “save” queue.
- Audio writer thread appends to a single WAV and (optionally) cuts chunk WAVs for transcription.
- Transcription runs in a separate process consuming chunk paths and appending JSONL transcripts.
- Motor commands are funneled through a single executor thread to avoid unsafe concurrent SDK calls.

## Agentic Architecture (Enforcement Rules)

These are **hard architectural rules** for the agentic subsystems in this repo. Violations are architectural bugs and should be caught in docs, code review, and (where possible) CI.

### Layer Ownership (Exclusive)

Paths refer to the `src/maxim/` package layout.

- `src/maxim/agents/`: owns goals, role-specific reasoning, intent generation; must **not** execute tools, mutate state, or inspect environments.
- `src/maxim/planning/`: owns plan generation/refinement; must **not** execute actions, select final actions, or mutate state.
- `src/maxim/planning/decision_engine.py`: owns action selection/arbitration/control flow; must **not** generate plans, execute tools, store memory, or inspect environment internals.
- `src/maxim/planning/policy.py`: owns constraints/guardrails/safety rules; must **not** perform planning, execution, or goal reasoning.
- `src/maxim/tools/`: owns side effects (I/O, network, filesystem, APIs); must **not** do control flow, reasoning, or decision making.
- `src/maxim/environment/`: owns observation of the world; must **not** perform side effects or execute tools.
- `src/maxim/memory/`: owns storage/retrieval/compression/forgetting; must **not** do decision making or action selection.
  - `hippocampus.py`: Associative memory graph storing complete agentic loops with selective capture, compression, and sleep-based consolidation.
  - `types.py`: EpisodicMemory and CompressedMemory dataclasses.
  - `strategies.py`: Pluggable memory management strategies (AccessBased, ImportanceBased, TemporalAware).
- `src/maxim/time/`: owns temporal indexing and rhythm tracking.
  - `scn.py`: Suprachiasmatic Nucleus - temporal bin indexing for circadian/weekly/monthly patterns.
  - `temporal_signature.py`: Phase-based temporal fingerprinting.
- `src/maxim/decisions/`: owns causal inference and prediction.
  - `nac.py`: Nucleus Accumbens - learns event→outcome relationships via temporal difference learning.
- `src/maxim/similarity/`: owns multi-modal similarity queries.
  - `ec.py`: Entorhinal Cortex - LSH-based approximate nearest neighbor for situation matching.
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
  - `movement_tracker.py`: Physics-based movement energy estimation.
  - `registry.py`: EnergyRegistry with domain budgets and aggregation.
- `src/maxim/bridges/`: owns cross-system integration between memory and external systems.
  - `spatial_bridge.py`: Location priors from historical object positions.
  - `salience_bridge.py`: Interaction history boosts for salience scoring.
  - `planning_bridge.py`: Plan template retrieval from successful memories.
  - `escalation_bridge.py`: Learned thresholds for when to escalate to human.
  - `pain_bridge.py`: Connects pain detection to NAc for causal learning of aversive patterns.
  - `energy_bridge.py`: Connects energy tracking to NAc for learning action→energy associations.
- `src/maxim/integration/`: owns central coordination.
  - `memory_hub.py`: MemoryHub coordinates all bridges and manages session lifecycle.
- `src/maxim/state/` (reserved): owns authoritative runtime truth; must **not** contain long-term storage logic or planning.
- `src/maxim/runtime/`: owns agentic orchestration/main execution loop; must **not** do domain reasoning.
- `src/maxim/conscience/`: owns robot orchestration/main loop (Reachy capture/inference/control); must **not** do agentic decision making.

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
- `src/maxim/conscience/selfy.py`: `Maxim` orchestrator (capture loop, lifecycle, logging, key responses).
- `src/maxim/agents/`: agent interfaces + implementations (reasoning/intent, no side effects).
- `src/maxim/planning/`: planning + policy + decision engine (agentic action selection).
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
- `src/maxim/models/vision/`: perception models (YOLO segmentation/pose).
- `src/maxim/models/movement/`: MotorCortex model (ConvNeXt-Tiny head-movement prediction).
- `src/maxim/models/audio/`: Whisper wrapper (transcription backend).
- `src/maxim/models/language/`: optional local LLM routing (transcript → agentic action).
- `src/maxim/data/`: camera/audio utilities and file outputs.
- `src/maxim/utils/`: config, logging, plotting, filesystem helpers (and reusable small helpers).

## Output Layout (Default)
- `data/videos/`: `reachy_video_<YYYY-MM-DD_HHMMSS>.mp4`
- `data/audio/`: `reachy_audio_<YYYY-MM-DD_HHMMSS>.wav` and optional `audio/chunks/*.wav`
- `data/transcript/`: `reachy_transcript_<YYYY-MM-DD_HHMMSS>.jsonl`
- `data/training/`: `motor_training_set.jsonl` (trainable samples + user marks)
- `data/agents/<STATE_NAME>/runtime/`: `state_<run_id>.json` (agentic runtime state snapshots; defaults to `agent_name` unless an agent sets `state_name`)
- `data/models/MotorCortex/`: MotorCortex checkpoint + training artifacts
- `data/util/llm.json`: local LLM config (created on install with SmolLM 1.7B as default)

## LLM Configuration

The LLM subsystem uses a JSON config file that persists user preferences across reinstalls.

### Config File Priority

The system searches for config in this order (first found wins):
1. `MAXIM_LLM_CONFIG` environment variable path
2. `./data/util/llm.json` (current working directory)
3. `./llm.json` (current working directory)
4. Repo root `data/util/llm.json`
5. Repo root `llm.json`

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

## Memory Management (Hippocampus + MemoryHub)

The Hippocampus provides episodic memory for the agentic system, storing complete loop cycles. The MemoryHub coordinates cross-system integration via bridges.

### System Architecture

```
                      ┌─────────────────────────────────────────┐
                      │            HIPPOCAMPUS                  │
                      │  ┌───────┐ ┌───────┐ ┌───────┐         │
                      │  │  SCN  │ │  NAc  │ │  EC   │         │
                      │  └───┬───┘ └───┬───┘ └───┬───┘         │
                      └──────┼─────────┼─────────┼──────────────┘
                             │         │         │
                             └─────────┼─────────┘
                                       │
                              ┌────────┴────────┐
                              │   MEMORY HUB    │
                              └────────┬────────┘
                                       │
     ┌──────────────┬──────────────────┼──────────────────┬──────────────┐
     ▼              ▼                  ▼                  ▼              ▼
  Spatial       Salience           Planning          Escalation      (Future)
   Bridge        Bridge             Bridge            Bridge          Fear
```

### Subsystems

| Subsystem | Purpose | Key Features |
|-----------|---------|--------------|
| **SCN** (Suprachiasmatic Nucleus) | Temporal rhythm indexing | 24h/7d/monthly bins, pattern detection |
| **NAc** (Nucleus Accumbens) | Causal inference | Event→outcome learning, reward prediction |
| **EC** (Entorhinal Cortex) | Similarity queries | LSH + neural semantic embeddings (Phase 4) |

### Bridges

| Bridge | Connects | Before | After |
|--------|----------|--------|-------|
| **SpatialMemoryBridge** | Hippocampus ↔ SpatialMap | Workspace rebuilt each session | Multi-session object location priors |
| **SalienceMemoryBridge** | Hippocampus ↔ SalienceNetwork | Pure novelty/recency | Interaction history boosts |
| **PlanHistoryBridge** | Hippocampus ↔ NAc | Plans from scratch | Successful template retrieval |
| **EscalationLearningBridge** | Hippocampus ↔ SCN/NAc | Fixed thresholds | Per-goal, per-time learned thresholds |

### Selective Capture

Not every loop is captured - only "interesting" ones:
- User input (CLI or speech)
- High novelty (> 0.7 threshold)
- High salience (> 0.7 threshold)
- Goal changes
- Failures (for learning)
- Periodic checkpoints

### Sleep Consolidation

A periodic process (call `hippocampus.sleep()` or `hub.on_session_end()`) manages memory:
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
# Create core systems
hub = MemoryHub(hippocampus=hippocampus, scn=scn, nac=nac, ec=ec)
hub.connect(spatial=spatial_map, salience=salience_network)

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
| `MOTOR_COMMAND` | Energy for movement execution | MovementEnergyTracker |
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

The EnergyCircuitBridge enables learning action→energy associations:
- High energy expenditure → NEGATIVE valence → NAc learns to predict
- Low energy expenditure → POSITIVE valence → Efficient actions preferred
- Future actions can be gated based on predicted energy cost

### LLM Context Injection

Energy state can be injected into LLM prompts for energy-aware decisions:

```python
energy_context = bridge.get_energy_context_for_llm()
# Returns: "[Energy Status]\n- llm: 45% energy remaining\n..."
```

## Persistence System

Many learning components persist their state across sessions, enabling continuous improvement over time.

### Persistent Components

| Component | File | Persists | Auto-Save Interval |
|-----------|------|----------|-------------------|
| **FocusLearner** | `data/util/focus_learner.json` | Directional gains (h+/h-/v+/v-), sample stats | 60s |
| **WorkspaceBoundsLearner** | `data/util/workspace_bounds.json` | Learned workspace limits | 60s |
| **EscalationLearningBridge** | `data/util/escalation_learning.json` | Per-goal/hour thresholds, escalation records | 60s |
| **FearCircuitBridge** | `data/util/fear_learning.json` | Risk adjustments, events, category stats | 60s |
| **AdaptiveThresholdController** | `data/util/adaptive_thresholds.json` | Novelty/salience thresholds, history | 60s |
| **NAc** | `data/util/nac_state.json` | Causal links, event outcomes | 60s |
| **SCN** | `data/util/scn_state.json` | Temporal bins, rhythm patterns | 60s |
| **Hippocampus** | `data/util/hippocampus.json` | Episodic memories | On session end |
| **PainDetector** | `data/util/pain_detector.json` | Pain event history | 60s |

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

## Invariants
- Control loop must not perform heavy disk I/O.
- Recording uses backpressure (bounded queues) rather than intentional dropping when "record everything" is requested.
- Public import paths should remain stable, or be preserved via re-exports when refactoring.
- Reusable helpers should live at module scope (prefer `src/maxim/utils/`) instead of being defined inside hot-loop functions.
- Memory management (sleep consolidation) should run outside the hot loop (during idle time or scheduled intervals).
