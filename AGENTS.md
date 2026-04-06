# Agent Rules (Maxim)

Maxim is a Reachy Mini project for capturing audio/video, running perception + motor learning, and controlling the robot in real time.

## Standards (Project Defaults)
- Prefer small, surgical changes; minimize refactors unless explicitly requested.
- When possible reduce the size of code to simplified systems.
- Preserve behavior unless the user asks for functional changes.
- Keep runtime loops responsive: push heavy I/O and compute off the control loop (threads/processes + queues).
- Make logging human-friendly; use `src/maxim/utils/logging.py` and respect `--verbosity`.
- Keep importable Python code under `src/maxim/` (avoid new top-level packages under `src/` unless packaging is updated).
- Avoid reusable nested functions: if a helper could be reused, define it at module scope (or under `src/maxim/utils/`) instead of inside another function/method. Nested defs are OK when they must capture closure state (e.g., worker threads) or are truly one-off.
- Any public API or CLI change must be reflected in `DECISIONS.md` and (when user-facing) `README.md`.
- **Versioning:** When making changes that affect runtime behavior, CLI interface, or peer/leader protocol, bump the version in **both** `pyproject.toml` (`version = "X.Y.Z"`) and `src/maxim/__init__.py` (`__version__ = "X.Y.Z"`). Use `maxim peer version` to verify local/leader sync after deployment.
- Add concise comments about important code functionality or nuanced behavior.
- Build modules for scalability to be applied to multiple sensory modalities (e.g., diffusion models applied to both images and audio).
- Run additional analysis on the security of code. If an insecurity is identified within the repo, analyze the bug for potential fixes and notify the user alongside the fix.
- If files or data are created in the `sandbox/` or `.maxim_workspace/`, delete old and un-necessary files and data if no longer being used.
- Build code to handle both CPU and GPU execution paths; agentic runtime should only run when a GPU is available.

## Naming Conventions (Cross-System Consistency)

When multiple systems share the same functional role, they **must** use the same method and property names. This ensures the codebase reads uniformly and that abstract protocols (ABCs) map cleanly to concrete implementations.

**Rule:** If two or more modules perform the same operation, name that operation identically across all of them — don't invent synonyms.

| Operation | Canonical Name | DO NOT use |
|-----------|---------------|------------|
| Full sleep-cycle management (compress + remove + preserve) | `sleep()` | `cleanup()`, `gc()` |
| Promote important memories to long-term | `consolidate()` | `promote()`, `commit()` |
| Store a new record | `capture()` / `store()` | `add()`, `insert()`, `create()` |
| Retrieve by ID | `get()` | `fetch()`, `find_by_id()` |
| Query by filters | `recall()` | `search()`, `query()`, `find()` |
| Persist to disk | `save()` / `load()` | `dump()`, `serialize()`, `write()` |
| Internal association graph | `graph` (property) | `_graph`, `get_graph()` |
| Layer identifier | `layer_name` (property) | `name`, `type`, `kind` |
| Record access tracking | `touch()` | `update_access()`, `mark_read()` |

**Why:** When adding new memory layers (ATL, future layers) or subsystems, they implement the same `MemoryLayer` protocol. Consistent naming means:
- Protocol compliance is obvious (same names = same interface)
- Cross-system code (MemoryHub, bridges, promoters) works uniformly
- New contributors immediately understand the API from any one implementation

**Extending this table:** When introducing a new shared operation, add the canonical name here before implementing. If an existing operation has an inconsistent name in one system, rename it during the next change to that system.

## Allowed Actions
- Modify code under `src/` with user requests.
- Add/modify smoke tests under `src/tests/` (offline-by-default; provide explicit opt-in for robot/network).
- Update documentation (`README.md`, `DECISIONS.md`, `ARCHITECTURE.md`).
- Creating new file within a `src/` folder if another file would better seperate module functionality, always request approval first.
- Create files and data in the `sandbox/` folder for creating experimental functionality and to be added directly into the repo if useful.

## Forbidden / Avoid
- Avoid excessively long additions or extensive refactoring without explicit requests.
- Large refactors without explicit task instruction.
- Breaking public imports/entrypoints without providing a compatibility layer.
- Adding network-requiring steps to default tests (e.g., model downloads) without an opt-in flag.
- Adding code outside of pre-existing architectures when a pre-existing build component could be
- Adding in functions within functions that could be used in other functions or libraries. Attempt to add to their respective src/maxim/ folders and scripts.
- Using insecure libraries, configurations, or practices even with explicit permission from the user if forbidden.

## Coding Guidelines
- Python `>=3.12` (see `pyproject.toml`).
- Follow existing repo style; keep code straightforward and readable.
- Add type hints where they improve clarity; prioritize stable interfaces for cross-module use.

## Environment Diagnostics (`maxim doctor`)

`maxim doctor` lives in [src/maxim/doctor/](src/maxim/doctor/) and runs platform-aware checks (OS, runtime — native/WSL2/docker/etc., GPU, local LLM server, leader role, LAN access, cloudflared, tunnel config, API key) with platform-specific fix hints. Companion subcommand `maxim peer test <url>` verifies peer↔leader connectivity. User-facing docs: [docs/user/llm-setup.md](docs/user/llm-setup.md). Forward roadmap: [docs/plans/doctor_upgrade_plan.md](docs/plans/doctor_upgrade_plan.md).

**When adding a new check:**
1. Pure function in `doctor/checks.py` → returns `CheckResult(name, status, message, fix?, retry_id?)`.
2. Register it in `run_all_checks()` under the right section.
3. Branch fix hints on `PlatformInfo.runtime` / `.os` / `.distro` with copy-pasteable, runnable commands. Fill in **real detected values** (IPs via `detect_wsl_ip()` / `detect_lan_ip()` / `info.windows_host_ip`) rather than placeholders.
4. Add unit tests in `tests/unit/test_doctor.py`. Mock network + subprocess calls so tests run offline.
5. Keep checks fast (< 1s). Long benchmarks belong in a future `maxim benchmark` subcommand.

**Module layout rules:**
- Lazy-import deps inside check functions (keeps `doctor` startup fast when optional features aren't installed). Tests must patch the **original** module path (e.g., `maxim.tunnel.cloudflared.find_cloudflared`), not the check module.
- `maxim peer test` stays self-contained (no imports from the agent runtime) — peers may not have the full dep tree installed.

**Don'ts:**
- Don't auto-execute fixes without explicit user opt-in (the `--fix` flag is intentionally opt-in; see the upgrade plan).
- Don't add a failing check without a user-actionable `fix` string.
- Don't rename the subcommand modules (`doctor/`, `tunnel/`) — they're referenced by CLI entrypoints and user docs.

## When Uncertain
- Ask for clarification about desired runtime behavior (e.g., “record everything” vs “latest snapshot”).
- Don’t guess domain logic (robot kinematics, label semantics, training targets); prefer small instrumentation/logging to validate assumptions.


## AGENT DEFINITION

An agent is a goal-oriented reasoning module.
Agents THINK but do not ACT.


## ALLOWED AGENT ACTIONS

Agents MAY:
- Read from state
- Query memory
- Propose intents
- Evaluate outcomes
- Request plans

Agents MAY NOT:
- Execute tools
- Mutate state
- Control execution loops
- Inspect environment internals
- Coordinate directly with other agents


## AGENT OUTPUT CONTRACT

Agents must emit STRUCTURED INTENT, never imperative commands.

Valid:
- Intent objects
- High-level goals
- Action proposals

Invalid:
- Tool calls
- HTTP requests
- File operations


## MULTI-AGENT RULE

Agents do not call or import other agents.
All coordination happens through:
- State
- Decision engine
- Explicit message passing


## AGENT PURITY RULE

Given the same state, memory, and policy constraints,
agent output must be deterministic.
Any randomness must be explicit and injected.


## HIPPOCAMPUS MEMORY SYSTEM

The Hippocampus is an associative memory substrate that stores complete agentic loop cycles
(perception → decision → action → outcome) as episodic memories with rich contextual indexing.

### Architecture Overview

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
     ┌─────────┬─────────┬──────────┼──────────┬─────────┬─────────┬─────────┬─────────┐
     ▼         ▼         ▼          ▼          ▼         ▼         ▼         ▼         ▼
  Spatial  Salience  Planning   Escalation   Fear      Pain     Energy    Comms     Math
   Bridge   Bridge    Bridge     Bridge     Bridge    Bridge    Bridge   Bridge    Bridge
```

### Core Components

| Component | Location | Purpose |
|-----------|----------|---------|
| **Hippocampus** | `memory/hippocampus.py` | Episodic memory storage, capture, recall, consolidation |
| **SCN** | `time/scn.py` | Temporal rhythm indexing (circadian, weekly, monthly patterns) |
| **NAc** | `decisions/nac.py` | Causal inference and reward prediction |
| **EC** | `similarity/ec.py` | Multi-modal similarity engine (LSH-based) |
| **MemoryHub** | `integration/memory_hub.py` | Central coordinator for all bridges |
| **PainDetector** | `proprioception/pain.py` | Detects aversive movement patterns |
| **MovementTracker** | `proprioception/movement_tracker.py` | Tracks position history, computes velocity |
| **HarmRegistry** | `harm/registry.py` | Aggregates harm predictions from all predictors |
| **EnergyRegistry** | `energy/registry.py` | Aggregates energy tracking from all domains |

### Bridges (Cross-System Integration)

Bridges connect the memory system to external perception/decision/action systems:

| Bridge | Connects | Purpose | Persists To |
|--------|----------|---------|-------------|
| **SpatialMemoryBridge** | Hippocampus ↔ SpatialMap | Location priors for object finding | - |
| **SalienceMemoryBridge** | Hippocampus ↔ SalienceNetwork | Interaction history boosts | - |
| **PlanHistoryBridge** | Hippocampus ↔ NAc | Successful plan template retrieval | - |
| **EscalationLearningBridge** | Hippocampus ↔ SCN/NAc | Learned escalation thresholds | `escalation_learning.json` |
| **FearCircuitBridge** | Hippocampus ↔ FearAgent ↔ NAc (+ EC via associative graph) | Memory-informed risk assessment | `fear_learning.json` |
| **PainCircuitBridge** | PainDetector ↔ NAc | Learns action→pain associations | *(via NAc persistence)* |
| **CommunicationBridge** | Comms ↔ Hippocampus | Communication-aware memory | - |

### Memory Types

- **EpisodicMemory**: Complete capture of one agentic loop (~2.5KB)
  - Perception, Context, Decision, Action, Outcome
  - Access tracking (created_at, accessed_at, access_count)
  - Long-term promotion status

- **CompressedMemory**: Lightweight summary for long-term storage (~200 bytes)
  - Essential fields only (goal, tool, success, novelty, salience)
  - Preserves enough for pattern matching and learning

### Selective Capture

Only "interesting" loops are captured:
- User input (CLI or speech)
- High novelty (> 0.7 threshold)
- High salience (> 0.7 threshold)
- Goal changes
- Failures (for learning)
- Periodic checkpoints

### Consolidation

A periodic process (call `hippocampus.sleep()` or `hub.on_session_end()`) manages memory. `sleep()` is the top-level method that runs compression, removal, and preservation, and internally calls `consolidate()` for long-term promotion. `consolidate()` can also be called standalone to promote specific memories without full sleep processing:
1. **Long-Term Promotion**: Important memories marked for preservation
2. **Compression**: Old EpisodicMemory → CompressedMemory
3. **Removal**: Stale memories not accessed in 1 week (configurable)
4. **Temporal Clustering**: SCN-aware consolidation keeps temporal coverage

### Memory Strategies (Extensible)

Memory management uses a strategy pattern for flexibility:
- `AccessBasedStrategy`: Recency + frequency of access
- `ImportanceBasedStrategy`: Novelty + success + user interaction
- `CompositeStrategy`: Weighted combination
- `TemporalAwareStrategy`: SCN-integrated with temporal boosts

New strategies can be implemented by subclassing `MemoryStrategy`.

### Access Tracking

Every memory tracks:
- `created_at`: Original capture time
- `accessed_at`: Last retrieval time (updated by `recall()`, `get()`, etc.)
- `access_count`: Total retrieval count
- `long_term`: Boolean flag for long-term promotion
- `consolidated_at`: When promoted to long-term (if applicable)

This enables biological-like memory decay and reinforcement.

### Usage Example

```python
from maxim.memory.hippocampus import Hippocampus
from maxim.integration import MemoryHub
from maxim.time.scn import SCN
from maxim.decisions.nac import NAc
from maxim.similarity.ec import EntorhinalCortex

# Create core systems
hippocampus = Hippocampus()
scn = SCN()
nac = NAc()
ec = EntorhinalCortex()

# Create hub
hub = MemoryHub(hippocampus=hippocampus, scn=scn, nac=nac, ec=ec)
hub.connect()

# Session lifecycle
hub.on_session_start()

# ... capture memories via hippocampus.capture() or capture_from_loop() ...

# Escalation decision using learned thresholds
should, reason = hub.should_escalate("find mug", novelty=0.7, salience=0.8)

# Plan success prediction
success = hub.get_predicted_success("find mug", ["look_around", "grasp"])

# End session (runs mandatory consolidation cycle)
hub.on_session_end()
```

### Location

- Core: `src/maxim/memory/hippocampus.py`
- Subsystems: `src/maxim/time/`, `src/maxim/decisions/`, `src/maxim/similarity/`
- Bridges: `src/maxim/bridges/`
- Integration: `src/maxim/integration/memory_hub.py`


## PROPRIOCEPTION SYSTEM

The proprioception module provides body awareness through movement tracking, pain detection, and adaptive focus learning.

### FocusLearner (Rescorla-Wagner Movement Correction)

The FocusLearner adapts movement gain through closed-loop feedback using Rescorla-Wagner learning:

```
ΔV = α(λ - V)

Where:
- V = current gain estimate (prediction)
- λ = optimal gain computed from overshoot ratio
- α = learning rate (0.2 default)
```

**Learning Flow:**
1. `record_intent(du, dv)` - Record intended pixel movement
2. Robot executes movement with current gain
3. `record_result(target_u, target_v)` - Record where target ended up
4. Compute overshoot ratio: `actual_movement / intended_movement`
5. Update gain: `new_gain = current + lr × (optimal - current)`

**Directional Gains:**
Separate gains for each direction (handles mechanical asymmetry):
- `_gain_h_pos` / `_gain_h_neg` - Horizontal positive/negative
- `_gain_v_pos` / `_gain_v_neg` - Vertical positive/negative

**Persistence:**
Learned gains persist to `data/util/focus_learner.json`:
```json
{
  "version": 1,
  "gains": {
    "h_pos": 0.72,
    "h_neg": 0.68,
    "v_pos": 0.65,
    "v_neg": 0.71
  },
  "stats": {
    "total_samples": 1523,
    "successful_focuses": 1102
  }
}
```

### AdaptiveThresholdController

Dynamically adjusts escalation thresholds based on feedback:
- Too many escalations → raise threshold
- Escalations ignored by LLM → raise threshold
- LLM queue busy → raise threshold
- High fear/risk → lower threshold

Persists to `data/util/adaptive_thresholds.json`.

### Components

| Component | Purpose | Persists |
|-----------|---------|----------|
| `FocusLearner` | Adaptive movement gain via R-W learning | Yes |
| `MovementTracker` | Position history, velocity computation | No |
| `PainDetector` | Aversive pattern detection | Yes |
| `AdaptiveThresholdController` | Dynamic escalation thresholds | Yes |

### Location

- `src/maxim/proprioception/focus_learner.py`
- `src/maxim/proprioception/movement_tracker.py`
- `src/maxim/proprioception/pain.py`
- `src/maxim/default_network/gate.py` (AdaptiveThresholdController)


## FEAR AGENT (Safety Gating)

The FearAgent reviews actions before execution, integrating with both predictive harm detection
and learned pain associations to prevent harmful robot behaviors.

### Two-Tier Harm Prevention

```
Action Proposal
      │
      ├──→ Tier 1: HarmRegistry.predict_all()  ← Zero latency (physics-based)
      │         ├── MovementHarmPredictor (velocity analysis)
      │         └── JointLimitHarmPredictor (workspace bounds)
      │
      └──→ Tier 2: NAc.predict()  ← Learned associations
                └── Pain patterns learned from past movements
```

### FearAgent.review_action() Flow

1. **Predictive Check**: Query HarmRegistry for physics-based harm predictions
2. **Learned Check**: Query PainCircuitBridge for NAc-based pain predictions
3. **Risk Scoring**: Calculate risk_score = intensity × confidence
4. **Gating Decision**: Gate if risk_score ≥ 0.7, warn if ≥ 0.4

### Example Integration

```python
def review_action(
    self,
    action_type: str,
    action_params: dict[str, Any],
    harm_registry: HarmRegistry | None = None,
    pain_bridge: PainCircuitBridge | None = None,
) -> ReviewResult:
    findings = []

    # Tier 1: Predictive harm
    if harm_registry:
        prediction = harm_registry.predict_worst(action_type, action_params)
        if prediction and prediction.risk_score >= 0.4:
            findings.append(Finding(
                category=DangerCategory.RESOURCE_EXHAUSTION,
                description=f"Predicted: {prediction.reason}",
                severity=RiskLevel.MEDIUM if prediction.risk_score >= 0.7 else RiskLevel.LOW,
            ))

    # Tier 2: Learned pain prediction
    if pain_bridge and action_type == "movement":
        should_gate, reason = pain_bridge.should_gate_action(
            action_params.get("action_signature", "")
        )
        if should_gate:
            findings.append(Finding(
                category=DangerCategory.RESOURCE_EXHAUSTION,
                description=f"Learned pain risk: {reason}",
                severity=RiskLevel.MEDIUM,
            ))

    return ReviewResult(findings=findings)
```

### Harm Categories

| Category | Description | Predictors |
|----------|-------------|------------|
| `MOVEMENT_VELOCITY` | Excessive movement speed | MovementHarmPredictor |
| `JOINT_LIMIT` | Near workspace boundaries | JointLimitHarmPredictor |
| `MOTOR_STALL` | Position unreachable | JointLimitHarmPredictor |
| `LLM_TIMEOUT` | Predicted slow LLM response | (Future) |
| `RESOURCE_EXHAUSTION` | Energy budget exceeded | EnergyCircuitBridge |


## CONTEMPLATION SYSTEM (ExecAgent Local Chain-of-Thought)

When running on local LLMs (or any provider without native extended thinking), ExecAgent uses a multi-pass contemplation loop to improve plan quality for complex goals. This replicates the effect of Anthropic's extended thinking across multiple LLM calls.

### Architecture

```
_propose_goal(ctx)
  │
  ├── Pass 1: DRAFT — Generate initial plan (existing code path)
  │
  ├── Complexity gate: _should_contemplate()
  │   └── Triggers on: 2+ sub_goals OR HIGH/CRITICAL priority
  │   └── Skips: IDLE, simple plans, extended thinking active
  │
  ├── Mode dispatch: _contemplate()
  │   ├── Standard mode (default): critique → confidence gate → optional refine
  │   └── Fast mode: single combined critique+refine call
  │
  ├── Preemption: only urgent percepts (CLI, voice, comms) interrupt
  │   └── Normal vision percepts queue until contemplation finishes
  │
  └── Fallback: any failure returns original draft unchanged
```

### Modes

| Mode | Passes | Description |
|------|--------|-------------|
| `standard` | 3 max | Separate critique and refine calls. Explicit confidence gate between. |
| `fast` | 2 max | Combined critique+refine in one call. Confidence gate preserved in response. |

### Quality Metrics (Phase 2)

ExecAgent subscribes to `GoalCompleted` on the AgentBus and correlates outcomes with contemplation metadata:
- **`_contemplation_log`**: Maps `goal_id → {contemplated, refined, timestamp}`
- **`_contemplation_stats`**: Running counters for contemplated/uncontemplated success rates
- **`contemplation_improvement_rate()`**: Returns success rate delta

### NAc Integration

Contemplation outcomes feed to NAc via `nac.observe()`:
- `event_type="contemplation"`, `event_signature="contemplation:refined"` or `"contemplation:draft"`
- `outcome_valence=POSITIVE/NEGATIVE` based on goal success
- NAc learns when contemplation helps vs wastes energy

### Adaptive Thresholds (Phase 3)

`_adaptive_thresholds()` queries NAc for learned contemplation outcomes and adjusts:
- **`confidence_threshold`**: Lower when contemplation helps (contemplate more), raise when it hurts
- **`min_sub_goals_to_trigger`**: Loosen gate when contemplation helps, tighten when it doesn't
- Bounded by configurable floor/ceiling values
- Requires minimum observations before adapting (default: 10)

### Smart Preemption (Phase 5)

Contemplation checks `_urgent_work_available` (not `_work_available`):
- **Urgent** (interrupts contemplation): CLI input, comms, voice keywords, high-urgency filtered percepts (≥ 0.7)
- **Non-urgent** (queues until done): vision percepts, low-urgency filtered percepts

### Configuration

In `data/util/llm.json` under the `contemplation` key:

```json
{
  "contemplation": {
    "enabled": true,
    "mode": "standard",
    "confidence_threshold": 0.7,
    "min_sub_goals_to_trigger": 2,
    "trigger_on_high_priority": true,
    "critique_max_tokens": 384,
    "refine_max_tokens": 512,
    "fast_max_tokens": 640,
    "adaptive_enabled": true,
    "adaptive_min_observations": 10
  }
}
```

### Biological Analogy

| Biological | Maxim Equivalent |
|------------|------------------|
| System 1 (fast path) | Simple percept → immediate action, no contemplation |
| System 2 (slow path) | Complex goal → draft → critique → refine → commit |
| Feeling of knowing | Confidence threshold — high confidence skips deliberation |
| Rumination cap | 3-pass maximum prevents infinite deliberation loops |
| Attentional interrupt | Urgent preemption aborts contemplation for salient stimuli |

### Location

- Core implementation: `src/maxim/agents/exec_agent.py` (methods: `_contemplate`, `_contemplate_standard`, `_contemplate_fast`, `_critique_plan`, `_refine_plan`, `_should_contemplate`, `_contemplation_config`, `_adaptive_thresholds`, `_on_goal_completed`, `contemplation_improvement_rate`)
- Config field: `src/maxim/models/language/router.py` (`LLMConfig.contemplation`)
- NAc wiring: `src/maxim/agents/maxim_agent.py` (`wire_memory_hub`)
- Tests: `tests/unit/test_contemplation.py` (93 tests)


## ENERGY TRACKING SYSTEM

The energy system monitors resource expenditure to enable energy-aware decision making.

### Energy Types

| Type | Tracker | Metrics |
|------|---------|---------|
| `LLM_TOKENS` | LLMEnergyTracker | input_tokens, output_tokens, model multiplier |
| `LLM_LATENCY` | LLMEnergyTracker | latency_ms, opportunity cost |
| `MOTOR_COMMAND` | MovementEnergyTracker | angular_distance, velocity, duration |

### LLM Energy Tracking

The LLMWorker automatically records energy for each LLM call:

```python
# In LLMWorker._process_request():
if self._energy_tracker is not None:
    self._energy_tracker.record(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        model=model_name,
        latency_ms=latency_ms,
        context={"request_id": request.request_id},
    )
```

### Model Multipliers (Energy Cost)

| Model | Multiplier | Notes |
|-------|------------|-------|
| `claude-3-haiku` | 0.5x | Most efficient |
| `claude-3-sonnet` | 1.0x | Baseline |
| `claude-3-opus` | 2.0x | High quality, high cost |
| `claude-opus-4-5` | 2.5x | Latest flagship |
| `local` | 0.2x | Local inference |

### Energy Budgets

Domains have capacity and recharge:

```python
# Check budget before expensive action
if registry.is_low_energy("llm"):
    # Consider using smaller model
    model = "claude-3-haiku"
elif registry.is_critical_energy("llm"):
    # Skip non-essential LLM calls
    return fallback_response()
```

### NAc Integration for Learning

The EnergyCircuitBridge teaches the system which actions are "expensive":

1. **Record Start**: `bridge.record_action_start("llm:planning:complex")`
2. **Energy Accumulates**: LLM tracker records tokens, latency
3. **Record End**: `bridge.record_action_end(event_id)` → Reports to NAc
4. **Future Prediction**: `bridge.predict_energy("llm:planning:complex")` → Learned cost

High-energy actions get NEGATIVE valence, enabling the system to prefer efficient alternatives.

### Location

- Energy Types: `src/maxim/energy/signal.py`
- Trackers: `src/maxim/energy/llm_tracker.py`, `src/maxim/energy/movement_tracker.py`
- Registry: `src/maxim/energy/registry.py`
- NAc Bridge: `src/maxim/bridges/energy_bridge.py`
