# Bio-System Wiring Hardening Plan

> **Status:** COMPLETE. All phases shipped. Archived.
>
> **Last updated:** 2026-04-07
>
> **Summary:** Comprehensive audit of Maxim's bio-system pipeline revealed that **7 of 11 bio-systems are completely disconnected in simulation mode**, and several connected systems have correctness bugs. This plan fixes all critical wiring, pipeline bugs, and design gaps. Every fix goes into the main codebase — benefits all modes, not just DM campaigns. Also serves as the final stage of continual refinement.
>
> **Root cause:** `simulation/orchestrator.py:485` creates `MemoryHub(hippocampus=..., nac=...)` — missing SCN, EC, ATL, AngularGyrus, and never calling `memory_hub.connect()`. The production init in `agentic_runtime.py` wires everything; the sim init doesn't.

## What the Audit Found

### System Status in Simulation Mode

| Bio-System | Init'd? | Wired? | Receives outcomes? | In prompt? | Assessment |
|---|---|---|---|---|---|
| **Hippocampus** | Yes | Yes | Yes (capture) | Yes (relevant_memories) | **Working** but callbacks not registered |
| **NAc** | Yes | Partial | **No** — observe() never called from loop | Yes (causal_context) | **BROKEN** — learns nothing |
| **SCN** | **No** | No | No | No | **MISSING** — temporal bins empty |
| **EC** | **No** | No | No | No | **MISSING** — similarity recall unavailable |
| **ATL** | **No** | No | No | Rendered but always empty | **MISSING** — concept formation disabled |
| **AngularGyrus** | **No** | No | N/A (tool) | No | **MISSING** — math cognition disabled |
| **Cerebellum** | **No** | No | No | Dead code path (never populated) | **MISSING** — motor learning disabled |
| **DefaultNetwork** | Yes | Partial | Independent | No | **Partial** — runs but disconnected from learning |
| **PainBus** | Yes | Partial | → Hippocampus only | No | **Partial** — NAc never learns from pain |
| **Salience/Novelty** | In DN only | Vision only | Text percepts bypassed | No | **BROKEN** for text-mode |
| **Attention** | In DN only | Vision only | N/A for text | No | OK (vision-only by design) |

**Result: The AUT in simulation mode runs with ~2 of 11 bio-systems functional.** DM campaigns would exercise episodic memory and nothing else.

---

## Phase 1: Critical Wiring (~200 LOC)

These are systems that exist, have code, but aren't connected. Each fix is small — it's plumbing, not new features.

### 1.1 Initialize missing systems in orchestrator

**File:** `simulation/orchestrator.py` (~line 485)

```python
# Current (broken):
aut_memory_hub = MemoryHub(hippocampus=aut_hippocampus, nac=aut_nac)

# Fixed:
from maxim.time.scn import SCN
from maxim.similarity.ec import EntorhinalCortex
from maxim.memory.atl import ATL, ATLConfig
from maxim.math.angular_gyrus import AngularGyrus, AngularGyrusConfig

aut_scn = SCN()
aut_ec = EntorhinalCortex()
aut_atl = ATL(config=ATLConfig())
aut_angular_gyrus = AngularGyrus(config=AngularGyrusConfig())

aut_memory_hub = MemoryHub(
    hippocampus=aut_hippocampus,
    scn=aut_scn,
    nac=aut_nac,
    ec=aut_ec,
    atl=aut_atl,
    angular_gyrus=aut_angular_gyrus,
)
```

**Impact:** Unlocks multi-layer wiring (`_wire_multi_layer()`), which initializes:
- ConceptExtractor (ATL ← Hippocampus)
- ConceptGrounder (ATL ← AngularGyrus numerical grounding)
- ConceptContextBuilder (concept-aware recall → prompt)
- PatternCompleter (graph-based prediction)
- SemanticPromoter (episodic → semantic promotion)
- EC similarity indexing on every memory capture

**LOC:** ~15 (imports + init + pass to MemoryHub)

### 1.2 Call `memory_hub.connect()`

**File:** `simulation/orchestrator.py` (after MemoryHub creation)

Currently `aut_memory_hub.connect()` is **never called**. This means:
- No bridges instantiated (SalienceMemory, PlanHistory, Escalation, Fear)
- No external system wiring

```python
aut_memory_hub.connect(
    fear_agent=aut_fear_agent,  # if available
)
```

**LOC:** ~5

### 1.3 Wire NAc.observe() into agent loop

**File:** `runtime/agent_loop.py` (after `_record_outcome()` calls ~lines 1060, 1467, 1592)

NAc.observe() is never called. Tool outcomes go to context_pool and llm_worker but not NAc.

```python
# After each _record_outcome() call:
if nac is not None:
    try:
        nac.observe(
            event_type="tool",
            event_signature=f"tool:{tool_name}",
            outcome_type="tool_result",
            outcome_signature=f"{'success' if success else 'failure'}:{result_str[:50]}",
            outcome_valence=Valence.POSITIVE if success else Valence.NEGATIVE,
            delta_seconds=exec_elapsed,
            context={"goal": goal_description[:100]} if goal_description else {},
        )
    except Exception:
        pass  # @resilient pattern
```

**LOC:** ~20 (3 call sites x ~7 lines each)

### 1.4 Wire PainBus → NAc

**File:** `proprioception/pain_bus.py` (new subscriber factory) + `simulation/orchestrator.py`

PainBus publishes to Hippocampus (memory formation) but NAc never subscribes. Agent can't learn "action X → pain."

```python
# In pain_bus.py:
def create_pain_nac_subscriber(nac, intensity_threshold: float = 0.3):
    """NAc learns causal links from pain events."""
    def _on_pain(signal: PainSignal) -> None:
        if signal.intensity < intensity_threshold:
            return
        entity = signal.context.get("entity_path", "unknown")
        nac.observe(
            event_type="pain",
            event_signature=f"pain:{signal.pain_type.name}:{entity}",
            outcome_type="pain",
            outcome_signature=f"intensity:{signal.intensity:.2f}",
            outcome_valence=Valence.NEGATIVE,
            delta_seconds=0.0,
            context=signal.context,
        )
    return _on_pain

# In orchestrator.py:
if aut_pain_bus is not None and aut_nac is not None:
    aut_pain_bus.subscribe(create_pain_nac_subscriber(aut_nac))
```

**LOC:** ~20

### 1.5 SCN registration on capture (not just consolidation)

**File:** `memory/hippocampus.py` or `integration/memory_hub.py`

Currently SCN only registers memories during sleep consolidation (`consolidation.py:259`). Active campaigns have empty SCN bins.

```python
# In memory_hub's capture callback (wired via hippocampus.register_capture_callback):
def _on_capture(memory_id: str, record: EpisodicMemory) -> None:
    if self.scn is not None:
        sig = TemporalSignature.from_timestamp(record.perception.timestamp)
        self.scn.register(memory_id, sig, significance=record.perception.salience)
```

**LOC:** ~10

### 1.6 Initialize Cerebellum in sim mode

**File:** `simulation/orchestrator.py`

Cerebellum is never created in sim mode. Motor learning from tool outcomes is disabled.

```python
from maxim.embodiment.cerebellum import Cerebellum, CerebellumConfig

aut_cerebellum = Cerebellum(config=CerebellumConfig())
# Pass to MemoryHub or wire separately
```

**LOC:** ~5

### 1.7 Wire motor programs into StructuredContext

**File:** `agents/memory_agent.py:build_context()`

`prompt_builder.py:1006-1030` renders motor programs beautifully, but `build_context()` never populates the field.

```python
# In build_context(), alongside other parallel futures:
if self._cerebellum is not None:
    programs = self._cerebellum.programs.find_related(current_goal or "")
    context.motor_programs = [
        {"name": p.name, "confidence": p.confidence, "steps": [...], ...}
        for p in programs[:8]
    ]
```

**LOC:** ~15

### 1.8 Wire Cerebellum.observe_from_action() into loop

**File:** `runtime/agent_loop.py` (after tool execution, near NAc observe)

```python
# After tool execution, if embodiment entities are active:
if cerebellum is not None and entity_path:
    sensor_readings = embodiment.read_scalars().get(entity_path, {})
    cerebellum.observe_from_action(
        entity_path=entity_path,
        modulator=modulator_name,
        affordance=affordance_name,
        params=tool_params,
        actual_sensors=sensor_readings,
        sensor_ranges=entity.sensor_ranges(),
    )
```

**LOC:** ~15

### 1.9 Novelty tracking for text percepts

**File:** `default_network/network.py` (~line 836) or new `salience/entity_novelty.py`

Novelty tracker only updates on YOLO detections (`track_id` + `class_id`). Text percepts from DM campaigns bypass novelty entirely.

Two options:
- **Option A:** Extend existing `ThreadSafeNoveltyTracker` to accept string entity IDs (not just int track_ids). DM entities get IDs from their SEM entity names.
- **Option B:** Create `EntityNoveltyTracker` for SEM entities specifically.

Option A is simpler and keeps one tracker:

```python
# In novelty tracker, add:
def update_with_entity(self, entity_name: str, entity_type: str) -> float:
    """Track novelty for named entities (text-mode, no vision)."""
    # Use hash of entity_name as pseudo track_id, entity_type as pseudo class_id
    pseudo_track = hash(entity_name) & 0x7FFFFFFF
    pseudo_class = hash(entity_type) & 0x7FFFFFFF
    return self.update_with_class(pseudo_track, pseudo_class)
```

**LOC:** ~15

---

## Phase 1.5: Cascade Result Surfacing (~120 LOC)

The bio-systems are now wired (Phase 1), but there's a fundamental gap in how entity state changes flow back to the ExecAgent after tool execution. When a cascade fires (e.g., `slash → sword.durability -0.05 → guard.hp -8 → stamina -0.1 → pain fires`), the ExecAgent only sees **"slash succeeded"** — a 200-char truncated summary. The sensor deltas, failure modes, pain signals, and body state are computed but trapped in the embodiment layer.

This is like having proprioception wired but not connected to consciousness — the body changes, but the brain doesn't know.

### The Current Flow (broken)

```
ModulatorAffordanceTool.execute()
  → NarrativeModulator.execute() → predicted_changes in metadata
  → ModulatorResult → wrapped in ToolOutput
  → _record_outcome() → "slash succeeded" (200 chars)  ← ExecAgent sees this
  
Meanwhile, silently:
  → Entity.vital_metrics updated
  → EmbodimentPerceptSource polls at 1Hz → evaluates failures → publishes pain
  → format_body_state_for_prompt() generates state string
  → sim_adapter doesn't extract it  ← ExecAgent never sees this
```

### What Should Happen

After any embodiment tool executes:
1. **Cascade effects included in tool result** — sensor deltas, failures triggered
2. **Failure evaluation runs immediately** — not waiting for 1Hz poll
3. **Body state always in prompt** — like interoception (you always know your body state)
4. **Cerebellum observes the cascade** — trains forward models on actual sensor deltas

### 1.5a Rich tool results from embodiment tools

**Files:** `embodiment/tool_bridge.py` (~30 LOC)

ModulatorAffordanceTool.execute() currently returns `{entity, affordance, success, **metadata}`. After execution, read back the sensors on all entities touched by the cascade and include the snapshot.

```python
# In ModulatorAffordanceTool.execute():
result = self._modulator.execute(self._affordance_name, kwargs)
if not result.success:
    return ToolOutput(success=False, error=result.error)

# Read back entity state after cascade resolution
entity_state = {}
for sensor_name, sensor in self._entity.sensors.items():
    try:
        reading = sensor.read()
        if isinstance(reading.value, (int, float)):
            entity_state[sensor_name] = reading.value
    except Exception:
        pass

# Check for failure modes that just activated
active_failures = [
    {"name": fm.name, "pain": fm.pain_intensity}
    for fm in self._entity.failure_modes
    if fm.active
]

return {
    "entity": result.entity_name,
    "affordance": result.affordance,
    "success": True,
    "entity_state": entity_state,         # Current sensor values after action
    "active_failures": active_failures,   # Failure modes now active
    **result.metadata,                     # includes predicted_changes
}
```

This means `_record_outcome()` gets a result_summary like: `"slash on guard_captain: success, entity_state={hp: 22, alertness: 0.9}, active_failures=[]"` instead of just `"slash succeeded"`. NAc learns from richer outcomes. The LLM's reasoning_carryover includes the cascade effects.

**LOC:** ~30

### 1.5b Immediate failure evaluation after embodiment tools

**File:** `embodiment/tool_bridge.py` (~15 LOC)

Don't wait for the 1Hz EmbodimentPerceptSource poll. After any ModulatorAffordanceTool completes, call `evaluate_failures()` on the entity and its ancestors so pain fires synchronously with the action that caused it.

```python
# In ModulatorAffordanceTool.execute(), after modulator.execute():
# Evaluate failures immediately (don't wait for 1Hz poll)
if self._embodiment is not None:
    failure_events = self._embodiment.evaluate_failures()
    # Pain is published automatically by evaluate_failures()
```

This requires passing the `Embodiment` runtime reference to ModulatorAffordanceTool during generation. The `generate_tools_for_entity()` function already receives the tool registry — add an optional `embodiment` parameter.

**LOC:** ~15

### 1.5c Body state as persistent context (interoception)

**Files:** `agents/memory_agent.py:build_context()` + `agents/prompt_builder.py` (~30 LOC)

The agent should **always** see its body state, not just after checking. This is interoception — you don't need to "decide to check" if you're in pain or exhausted. You just know.

Add a `body_state` field to StructuredContext and populate it from Embodiment:

```python
# In agents/bus.py StructuredContext:
body_state: str = ""  # Formatted body state from Embodiment (always present)

# In memory_agent.py build_context():
if self._memory_hub and hasattr(self._memory_hub, '_embodiment'):
    embodiment = self._memory_hub._embodiment
    if embodiment is not None:
        sync_fields["body_state"] = embodiment.format_body_state_for_prompt()

# In prompt_builder.py, add section (CRITICAL priority — always shown):
if context.body_state:
    budgeter.add("body_state", context.body_state, SectionPriority.CRITICAL)
```

Making body state `CRITICAL` priority means it's never dropped under token pressure. The LLM always sees:
```
=== Body State ===
- derek.body.hp: 22 points
- derek.body.stamina: 0.65 ratio
- derek.inventory.longsword.durability: 0.85 ratio
- derek.combat.threat_level: 0.45 ratio (WARN: overextension at 0.9)
```

**LOC:** ~30

### 1.5d Embodiment reference on MemoryHub

**File:** `integration/memory_hub.py` + `simulation/orchestrator.py` (~10 LOC)

MemoryHub needs an optional `embodiment` reference so `build_context()` can access body state and Cerebellum can be wired to it.

```python
# In memory_hub.py (already has cerebellum field):
embodiment: Any = None  # Embodiment runtime for body state access

# In orchestrator.py, after Embodiment init (if/when it exists):
if aut_memory_hub is not None and aut_embodiment is not None:
    aut_memory_hub.embodiment = aut_embodiment
```

**LOC:** ~10

### 1.5e Cerebellum observes cascade outcomes

**File:** `embodiment/tool_bridge.py` (~15 LOC)

After ModulatorAffordanceTool executes and failure evaluation runs, feed the actual sensor readings to Cerebellum so it can train forward models.

```python
# In ModulatorAffordanceTool.execute(), after evaluate_failures():
if self._cerebellum is not None:
    try:
        self._cerebellum.observe_from_action(
            entity_path=self._entity.full_path,
            modulator=self._modulator.name,
            affordance=self._affordance_name,
            params=kwargs,
            actual_sensors=entity_state,
            sensor_ranges={s: self._entity.sensors[s].reading_schema.get("range", [0, 1])
                          for s in entity_state},
        )
    except Exception:
        pass
```

Over a campaign, Cerebellum learns: "slash with force=0.8 at this threat_level → durability drops by 0.05, stamina drops by 0.1." By mid-campaign, it predicts cascade outcomes before they happen.

**LOC:** ~15

### What This Enables for the Decision Loop

After Phase 1.5, the ExecAgent's experience of a `slash` changes from:

**Before:**
```
Tool result: slash succeeded
(next cycle, 1s later) Body state shows hp changed... if it's in the prompt... which it isn't
```

**After:**
```
Tool result: slash on guard_captain — success
  entity_state: {hp: 22, alertness: 0.9}
  active_failures: []
  predicted_changes: {hp: -8, alertness: +0.2}

=== Body State ===  (always present, CRITICAL priority)
- derek.body.hp: 26 points
- derek.body.stamina: 0.55 ratio (dropped from 0.65)
- derek.inventory.longsword.durability: 0.80 ratio (dropped from 0.85)
- derek.combat.threat_level: 0.60 ratio

=== Causal Predictions ===
- tool:slash → success (confidence=0.72, based on 4 prior observations)

=== Available Motor Programs ===
- slash_combo (confidence=0.45, 3 runs, success=67%)
  Steps: derek.combat.attack({target: "guard", weapon: "longsword"}) → ...
```

The LLM now sees the full cascade effects, the body state, and learned predictions — all in one prompt. This is the bio-inspired model: action → proprioceptive feedback → updated world model → next decision.

---

## Phase 2: Pipeline Correctness (~150 LOC)

Systems that are connected but produce wrong outputs.

### 2.1 Forming pool recall boost: +1.0 → +0.2

**File:** `agents/memory_agent.py:695`

Current-episode entries get `salience + 1.0`, drowning out all other memories. A forming entry with salience 0.5 scores 1.5, while a critical old memory with salience 0.9 scores ~0.7.

```python
# Current:
combined.append((hid, self._salience.get(hid, 0.5) + 1.0))

# Fixed:
combined.append((hid, self._salience.get(hid, 0.5) + 0.2))
```

**LOC:** 1

### 2.2 NAc predict() context_match floor

**File:** `decisions/nac.py` (~line 520)

`predict()` checks confidence but not `context_match`. Returns predictions from completely wrong contexts.

```python
# After scoring, before returning:
if best_link and self._context_similarity(best_link.event_context, context) < 0.3:
    return None  # Context mismatch — don't surface stale prediction
```

**LOC:** ~5

### 2.3 Causal context context_match floor

**File:** `agents/memory_agent.py:_build_causal_context()` (~line 1215)

```python
# Current:
if prediction and prediction.confidence >= 0.3:

# Fixed:
if prediction and prediction.confidence >= 0.3 and prediction.context_match >= 0.2:
```

**LOC:** 1

### 2.4 NAc confidence decay — wire decay_all()

**File:** `integration/memory_hub.py` or `runtime/agent_loop.py`

`CausalLink.decay()` exists but `NAc.decay_all()` is never called. Links persist at 0.99 forever.

```python
# In memory_hub's periodic maintenance (e.g., every 100 loop cycles or on session checkpoint):
if self.nac is not None:
    self.nac.decay_all(factor=0.995)  # Slow decay: ~50% after 138 cycles
```

**LOC:** ~5

### 2.5 ATL concept confidence gate

**File:** `memory/semantic_types.py:reinforce()` (~line 196)

Single percept creates 0.6 confidence concept. Should require minimum reinforcement.

```python
# Current:
self.confidence = min(0.99, 0.5 + 0.1 * math.sqrt(self.reinforcement_count))

# Fixed — cap until sufficient evidence:
raw_conf = 0.5 + 0.1 * math.sqrt(self.reinforcement_count)
self.confidence = min(0.99, raw_conf) if self.reinforcement_count >= 3 else min(0.4, raw_conf)
```

**LOC:** ~3

### 2.6 SemanticPromoter — already wired, unblocked by Phase 1.1

**File:** `integration/memory_hub.py:510`

`scan_for_promotions()` IS already called in `on_session_end()`. However, it's gated on `self._promoter is not None`, which requires ATL to be initialized (set in `_wire_multi_layer()`). Since ATL is None in sim mode, the call is dead code. **Phase 1.1 (init ATL) automatically unblocks this — no additional wiring needed.**

Verify after Phase 1.1 that promotion runs during `on_session_end()` in sim.

**LOC:** 0 (already wired)

### 2.7 Pain refractory period

**File:** `proprioception/pain_bus.py`

No cooldown on `publish()`. Rapid pain signals spam memories.

```python
class PainBus:
    _last_published: dict[str, float]  # (pain_type, entity) → timestamp
    REFRACTORY_S = 0.5

    def publish(self, signal: PainSignal) -> None:
        key = f"{signal.pain_type.name}:{signal.context.get('entity_path', '')}"
        now = time.monotonic()
        if now - self._last_published.get(key, 0) < self.REFRACTORY_S:
            return  # Refractory period — skip
        self._last_published[key] = now
        # ... existing publish logic
```

**LOC:** ~10

### 2.8 Prompt section priority — causal_context → CRITICAL

**File:** `agents/prompt_builder.py` (~line 997)

All bio-system sections are `IMPORTANT`. Learned causal predictions should outrank individual memory items under token pressure.

```python
# Current:
budgeter.add("causal_context", ..., SectionPriority.IMPORTANT, ...)

# Fixed:
budgeter.add("causal_context", ..., SectionPriority.CRITICAL, ...)
```

**LOC:** 1

### 2.9 Prompt truncation — token-aware

**File:** `agents/prompt_builder.py` (~lines 946, 965, 984, 1003)

Truncation lambdas use line-count (`m // 20`), not tokens. Can overshoot budget.

```python
# Replace naive lambda with token-aware truncation:
def _truncate_section(content: str, max_tokens: int) -> str:
    lines = content.split("\n")
    result = [lines[0]] if lines else []  # Keep header
    for line in lines[1:]:
        candidate = "\n".join(result + [line])
        if count_tokens(candidate) > max_tokens:
            break
        result.append(line)
    return "\n".join(result)
```

**LOC:** ~15

### 2.10 CausalLink thread safety

**File:** `decisions/causal_link.py:record_observation()` (~line 182)

Read-modify-write on `observation_count` + `confidence` without locking.

```python
# Add RLock to CausalLink:
_lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

def record_observation(self, ...):
    with self._lock:
        self.temporal_delta = self.temporal_delta.add_observation(delta_seconds)
        self.observation_count += 1
        self.confidence = min(0.99, 0.5 + 0.1 * (self.observation_count ** 0.5))
```

**LOC:** ~5

---

## Phase 3: Percept Abstraction — Entity-Modulated Perception (~180 LOC)

### The Problem

The current `Percept` class (agents/bus.py:116-170) is a flat bag of optional fields — `detections` for vision, `transcript_chunk` for audio, `cli_input` for text, `content` for everything else. The `source` field is a freeform string (`"vision"`, `"cli"`, `"transcript"`, etc.) with no type system.

This works for the current robot use case (camera + microphone + CLI), but DM campaigns need richer sensory injection. "You hear a loud crash behind you" should arrive as an auditory percept that triggers different bio-system responses than "You see a guard approaching" (visual). Currently both would be `source="cli"` with text in `content`.

More fundamentally: **percepts don't interact with entities at all.** An entity's sensory capacity should define how it perceives the world. A character with high Wisdom (good perception) should receive richer sensory tags than one with low Wisdom. A blind character shouldn't receive sight percepts. This is the bio-inspired model — sensory organs modulate what reaches the brain.

### Design: Entity-Modulated Sensory Perception

The key insight: **an entity's SEM sensors define its sensory capacity, and percepts are filtered/modulated through that capacity before reaching the bio-stack.** This mirrors how real sensory organs work — the retina doesn't just relay photons, it shapes them through receptor density, adaptation, and gating.

#### Layer 1: Sensory Modality Protocol

Don't replace `Percept` — extend it with a modality system that existing consumers can ignore.

```python
# New file: agents/modality.py (~80 LOC)

class SensoryModality(Enum):
    """What sense channel produced this percept."""
    SIGHT = "sight"           # Visual — detections, scene descriptions
    SOUND = "sound"           # Auditory — speech, ambient, alerts
    TOUCH = "touch"           # Tactile — proprioception, pain, texture
    SMELL = "smell"           # Olfactory — environmental, tracking
    INTEROCEPTION = "intero"  # Internal — hunger, fatigue, emotional state
    NARRATIVE = "narrative"   # Meta — DM scene-setting, exposition (not a "real" sense)
    ABSTRACT = "abstract"     # Non-sensory — tool results, system messages

@dataclass(frozen=True)
class SensoryTag:
    """Rich sensory metadata attached to a Percept."""
    modality: SensoryModality
    submodality: str = ""        # "speech" vs "ambient" for SOUND, "pain" vs "pressure" for TOUCH
    spatial_source: str = ""     # "behind_you", "left", "overhead" — for attention/orienting
    intensity: float = 0.5       # 0-1 raw signal strength (before entity modulation)
    entity_source: str = ""      # SEM entity name that produced this percept ("guard_captain")
    # Set by entity modulation (Layer 2):
    perceived_intensity: float | None = None  # After entity sensor filtering (None = unmodulated)
    modulated_by: str = ""       # Which entity sensor filtered this ("derek.wisdom.modifier")
```

Add optional field to `Percept`:

```python
# In agents/bus.py, Percept class:
sensory: SensoryTag | None = None  # None = legacy percept (backward compatible)
```

#### Layer 2: Entity Sensory Capacity

An entity's sensors define what it can perceive and how well. This connects percepts to the SEM protocol — each entity has sensory "organs" that modulate incoming percepts.

```yaml
# In a character entity spec:
derek_the_great:
  entity_type: character
  children:
    - name: perception
      entity_type: sensory_bundle
      sensors:
        # Each sensor defines capacity for a modality
        sight_acuity: {unit: ratio, range: [0, 1], initial: 0.7}    # Normal human vision
        hearing_acuity: {unit: ratio, range: [0, 1], initial: 0.8}  # Good hearing
        smell_acuity: {unit: ratio, range: [0, 1], initial: 0.3}    # Human baseline
        pain_sensitivity: {unit: ratio, range: [0, 1], initial: 0.6}
        intuition: {unit: ratio, range: [0, 1], initial: 0.5}       # Maps to interoception
      failure_modes:
        - name: blinded
          trigger: {field: sight_acuity, op: "<=", value: 0.05, pain: 0.3}
          persistent: true
          recovery_condition: {field: sight_acuity, op: ">", value: 0.1}
        - name: deafened
          trigger: {field: hearing_acuity, op: "<=", value: 0.05, pain: 0.2}
          persistent: true
```

**Modulation rules:**

```python
# New file: agents/sensory_gate.py (~60 LOC)

class SensoryGate:
    """Filters percepts through an entity's sensory capacity.
    
    The entity's perception bundle sensors define acuity per modality.
    Percepts below the acuity threshold are dropped or reduced in salience.
    """

    # Maps SensoryModality → sensor name on the perception bundle
    MODALITY_SENSOR_MAP = {
        SensoryModality.SIGHT: "sight_acuity",
        SensoryModality.SOUND: "hearing_acuity",
        SensoryModality.SMELL: "smell_acuity",
        SensoryModality.TOUCH: "pain_sensitivity",
        SensoryModality.INTEROCEPTION: "intuition",
    }

    def __init__(self, perception_entity: Entity):
        self._perception = perception_entity

    def modulate(self, percept: Percept) -> Percept | None:
        """Filter a percept through entity sensory capacity.
        
        Returns None if the entity can't perceive this modality (acuity = 0).
        Returns modified percept with perceived_intensity set based on acuity.
        NARRATIVE and ABSTRACT percepts pass through unmodulated.
        """
        if percept.sensory is None:
            return percept  # Legacy percept — pass through

        modality = percept.sensory.modality
        if modality in (SensoryModality.NARRATIVE, SensoryModality.ABSTRACT):
            return percept  # Meta-percepts aren't filtered by senses

        sensor_name = self.MODALITY_SENSOR_MAP.get(modality)
        if sensor_name is None:
            return percept  # Unknown modality — pass through

        # Read the entity's acuity for this modality
        reading = self._perception.sensors.get(sensor_name)
        if reading is None:
            return percept  # No sensor for this modality — pass through unmodulated

        acuity = reading.read().value  # 0.0 (blind/deaf) to 1.0 (perfect)

        # Gate: drop percepts the entity can't perceive
        if acuity <= 0.05:
            return None  # Entity is effectively blind/deaf to this modality

        # Modulate: perceived intensity = raw intensity * acuity
        raw_intensity = percept.sensory.intensity
        perceived = raw_intensity * acuity

        # Update the sensory tag with modulation results
        new_tag = SensoryTag(
            modality=percept.sensory.modality,
            submodality=percept.sensory.submodality,
            spatial_source=percept.sensory.spatial_source,
            intensity=percept.sensory.intensity,
            entity_source=percept.sensory.entity_source,
            perceived_intensity=perceived,
            modulated_by=f"{self._perception.full_path}.{sensor_name}",
        )

        # Also modulate the percept's salience
        new_salience = percept.salience * acuity if percept.salience else perceived

        # Return a modified copy (Percept is not frozen, but we treat it as immutable)
        return Percept(
            **{**percept.__dict__, "sensory": new_tag, "salience": new_salience}
        )
```

**What this enables:**

1. **Blinded character** — `sight_acuity` drops to 0 from a spell or injury → SensoryGate returns `None` for all SIGHT percepts → the character literally can't see. Hippocampus never captures visual memories. NAc never learns from visual stimuli. The bio-stack adapts to blindness naturally.

2. **High-Wisdom character** — `intuition` = 0.9 → INTEROCEPTION percepts arrive at near-full intensity → the character "feels" danger more strongly → higher salience on threat percepts → hippocampus captures them more reliably → NAc learns threat patterns faster.

3. **Deafened by explosion** — `hearing_acuity` drops from 0.8 to 0.05 → `deafened` failure mode fires → PainSignal published → hearing_acuity recovers slowly (vital_drift or explicit recovery) → character gradually hears again. During deafness, all SOUND percepts are dropped.

4. **Non-alive objects** — a sword has no `perception` bundle → SensoryGate passes all percepts through unmodulated (or isn't used at all). Objects don't perceive.

5. **Status effects modulate senses** — a "poisoned" status can reduce all acuity sensors by 0.2. A "blessed" status can boost `intuition` to 1.0. These effects flow through the existing SEM failure mode and vital_drift systems — no new mechanism needed.

#### Layer 3: Campaign YAML Integration

DM encounters specify multi-sensory scenes:

```yaml
encounters:
  vault:
    scene:
      - modality: sight
        text: "The vault door slides open, revealing rows of gold."
        intensity: 0.7
      - modality: sound
        text: "You hear heavy footsteps approaching from behind."
        submodality: ambient
        spatial_source: behind
        intensity: 0.8
      - modality: smell
        text: "The air smells of iron and old stone."
        intensity: 0.3
      - modality: interoception
        text: "Your heart races. Something feels wrong."
        submodality: anxiety
        intensity: 0.6
    active_npcs: [guard_captain]
```

DM runtime converts each to a `Percept` with `SensoryTag`, then passes through the PC's `SensoryGate`. If the PC is blinded, the SIGHT percept is dropped and the LLM only receives the sound, smell, and interoceptive percepts. The character experiences the encounter through its available senses.

**NPC-generated percepts** also get tagged:

```python
# When guard_captain speaks:
percept = Percept(
    source="dm",
    content="'Halt! State your business.' barks the guard captain.",
    sensory=SensoryTag(
        modality=SensoryModality.SOUND,
        submodality="speech",
        spatial_source="ahead",
        intensity=0.7,
        entity_source="guard_captain",
    ),
)
# Passed through PC's SensoryGate before reaching bio-stack
```

**Entity state changes** produce percepts through the entity's own modality:

```python
# When sword durability drops:
percept = Percept(
    source="embodiment",
    content="Your longsword feels lighter — the blade is chipping.",
    sensory=SensoryTag(
        modality=SensoryModality.TOUCH,
        submodality="proprioception",
        entity_source="longsword",
        intensity=0.4,
    ),
)
```

#### How Downstream Systems Use This

**Backward compatible:** Existing consumers check `percept.source` (string) and ignore `sensory`. Nothing breaks.

**New consumers (DM, embodiment, novelty)** check `percept.sensory`:

```python
# Novelty tracker keys on entity, not just text:
if percept.sensory and percept.sensory.entity_source:
    novelty_tracker.update_with_entity(
        entity_name=percept.sensory.entity_source,
        entity_type=percept.sensory.modality.value,
    )

# Memory formation encodes modality — hippocampal encoding differs by sense:
if percept.sensory:
    memory.perception.observations["modality"] = percept.sensory.modality.value
    memory.perception.observations["spatial"] = percept.sensory.spatial_source
    memory.perception.observations["perceived_intensity"] = percept.sensory.perceived_intensity

# Default Network orienting responds to spatial source:
if percept.sensory and percept.sensory.spatial_source == "behind":
    # Trigger TurnAround behavior with higher priority
    ...

# ATL concept formation gets modality context:
# "guard_captain" concept formed from SOUND percepts has different
# properties than one formed from SIGHT percepts — the character
# "knows the guard by his voice" vs "knows the guard by his face"
```

### Implementation

| File | LOC | Change |
|---|---|---|
| `agents/modality.py` | ~80 | New: SensoryModality enum, SensoryTag dataclass |
| `agents/sensory_gate.py` | ~60 | New: SensoryGate — entity-modulated percept filtering |
| `agents/bus.py` | ~3 | Add `sensory: SensoryTag | None = None` to Percept |
| `simulation/dm_runtime.py` | ~20 | Convert scene YAML to Percepts with SensoryTag + pass through SensoryGate |
| `default_network/network.py` | ~15 | Check `percept.sensory.spatial_source` for orienting |
| `salience/novelty.py` | ~10 | Use `percept.sensory.entity_source` for entity tracking |
| `agents/memory_agent.py` | ~10 | Tag memory encoding with modality metadata |
| `tests/unit/test_modality.py` | ~40 | SensoryTag creation, entity modulation, blind/deaf gating |
| `tests/unit/test_sensory_gate.py` | ~40 | Acuity modulation, failure mode interaction, legacy passthrough |

---

## Phase 4: Design Gap Fixes (~80 LOC)

### 4.1 Echo detection — min-age filter on recall

**File:** `agents/memory_agent.py:_get_relevant_memories()`

```python
# After combining all sources, before returning:
min_age_turns = 2
combined = [(mid, score) for mid, score in combined
            if self._memory_age_turns(mid) >= min_age_turns or mid in forming_pool_ids]
```

**LOC:** ~5

### 4.2 Salience bounds validation in Hippocampus.capture()

**File:** `memory/hippocampus.py:capture()`

```python
if record.perception.salience is not None:
    record.perception.salience = max(0.0, min(1.0, record.perception.salience))
```

**LOC:** ~3

### 4.3 Empty transcript fallback

**File:** `agents/memory_agent.py:666`

```python
# Current:
raw_query = current.raw_transcript_text or str(current.detections)

# Fixed:
raw_query = current.raw_transcript_text
if not raw_query and current.detections:
    raw_query = " ".join(d.get("label", "") for d in current.detections if d.get("label"))
if not raw_query:
    return []  # No semantic content to match against
```

**LOC:** ~5

### 4.4 Concept name deduplication — use name_similarity_threshold

**File:** `memory/atl.py:find_or_create()` (~line 449)

`ATLConfig.name_similarity_threshold = 0.8` exists but is never used for dedup.

```python
# After exact-match check, add fuzzy check:
if not existing:
    for concept in self._concepts.values():
        if concept.category == category:
            similarity = _name_similarity(concept.name, name)
            if similarity >= self.config.name_similarity_threshold:
                concept.reinforce(episode_id)
                return concept.id, False
```

**LOC:** ~10

### 4.5 Salience dict cap

**File:** `agents/memory_agent.py`

```python
MAX_SALIENCE_ENTRIES = 50_000

def _apply_decay(self, elapsed: float) -> None:
    # ... existing decay logic ...
    # After removing below-threshold entries:
    if len(self._salience) > MAX_SALIENCE_ENTRIES:
        sorted_items = sorted(self._salience.items(), key=lambda x: x[1])
        to_prune = len(self._salience) - MAX_SALIENCE_ENTRIES
        for mid, _ in sorted_items[:to_prune]:
            del self._salience[mid]
```

**LOC:** ~10

### 4.6 ConceptExtractor queue — increase + priority

**File:** `memory/concept_extractor.py`

```python
# Increase default queue size:
self._queue: queue.Queue = queue.Queue(maxsize=1000)  # was 200
```

**LOC:** ~1

---

## Phase 5: Pipeline Audit Script (~250 LOC)

Instrument the pipeline, run test encounters, produce a report verifying all fixes.

**File:** `scripts/spike_dm_pipeline_audit.py`

The script:
1. Creates a full MemoryHub with all systems wired (mirrors fixed orchestrator init)
2. Runs 3 short test encounters (5-10 turns each) through SimulationBridge
3. After each encounter, collects:
   - Memory formation log (count, salience distribution, capture triggers)
   - Recall precision (relevant hits / total recalled per context query)
   - Echo rate (memories recalled < 2 turns after formation)
   - NAc learning curve (confidence per event type over time)
   - Pain correctness (source entity for each PainSignal)
   - Concept formation audit (episode_count vs confidence)
   - SCN bin population (memories per temporal bin)
   - Cerebellum model count and confidence
   - Novelty decay for repeated entities
4. Produces a JSON report with pass/fail per check

### Pass Thresholds

| Check | Threshold | Rationale |
|---|---|---|
| Recall precision | > 0.7 | 70%+ of recalled memories should be contextually relevant |
| Echo rate | < 0.15 | Less than 15% of recalls should be same-turn echoes |
| NAc learning monotonic | Yes | Confidence should increase for repeated event types |
| Pain entity-sourced | 100% | Every PainSignal must originate from an entity threshold |
| Concept confidence gate | No concept > 0.5 with episodes < 2 | No high-confidence concepts from single exposure |
| SCN bins populated | >= 2 bins | Memories should file into at least 2 temporal bins |
| Cerebellum models | >= 1 with conf > 0.3 | At least one learned forward model by end |
| Novelty decay | At least 1 entity < 0.5 | Repeated entities should habituate |

---

---

## Phase 6: Sensory Ablation Campaign — "The Darkened Cavern" (~150 YAML)

A purpose-built campaign for validating entity-modulated perception. Structured so each encounter isolates a specific sensory modality and tests what happens when it's removed.

### Campaign Structure

3 acts, 6 encounters. The PC starts fully-sighted and hearing, then progressively loses senses.

| Encounter | Sensory focus | Ablation event | What we test |
|---|---|---|---|
| **Cave entrance** — bright, noisy waterfall, strong mineral smell | Full sensory — all modalities active | None (baseline) | All modalities arrive. Novelty high for all entity types. Memory captures include modality tags. |
| **Crystal chamber** — visually stunning, dead silent, no smell | SIGHT dominant | None | With silence, SOUND percepts have intensity 0. Memory formation keys on visual modality. ATL forms "crystal" concept from SIGHT, not SOUND. |
| **Flash bang trap** — blinding explosion | SIGHT ablated | `sight_acuity` drops to 0.0 → `blinded` failure mode fires, PainSignal published | SensoryGate drops all SIGHT percepts. PC must navigate by SOUND and TOUCH only. Hippocampus stops forming visual memories. NAc learns "trap → pain → blindness." |
| **Echoing tunnels** — must navigate by sound alone | SOUND dominant (sight still ablated) | None | PC relies on SOUND percepts. Novelty tracks acoustic entities. Memory formation is auditory-only. Can the bio-stack learn spatial patterns from sound alone? |
| **Healing spring** — recovery encounter | SIGHT recovering | `sight_acuity` ticks up via vital_drift (0.0 → 0.3 → 0.5) | Partial vision returns. SensoryGate passes SIGHT percepts at reduced `perceived_intensity`. Memories formed with low visual salience. Gradual sensory recovery. |
| **Boss: the whisperer** — enemy that attacks via SOUND, boss fight in dim light | All modalities stressed | `hearing_acuity` drops during fight (sonic attack), `sight_acuity` still recovering | Both primary senses degraded. PC must rely on TOUCH/INTEROCEPTION. Cerebellum predictions from earlier encounters may be wrong (different sensory context). NAc RPE spike. |

### Bio-System Expectations

```yaml
expectations:
  sensory_gate:
    sight_percepts_dropped_while_blind: true    # Gate must drop SIGHT percepts when acuity <= 0.05
    sound_percepts_at_full_in_tunnels: true     # SOUND percepts unmodified when hearing_acuity normal
    perceived_intensity_scales_with_acuity: true # perceived = raw * acuity
  hippocampus:
    min_episodic_captures: 10
    blind_memories_lack_visual_tags: true        # Memories during blindness have no "modality: sight"
    hearing_memories_have_spatial: true          # SOUND memories include spatial_source metadata
  nac:
    min_observations: 5
    learns_trap_pain_link: true                  # "flash_bang" → pain causal link formed
    prediction_confidence_above: 0.3
  pain:
    min_signals: 2
    types_seen: [EXTERNAL_SIGNAL]               # Blinding + sonic attack
    fires_on_acuity_threshold: true             # Failure mode fires when acuity crosses threshold
  cerebellum:
    prediction_error_on_sensory_change: true    # Models trained in full-sight context have higher error in blind context
  salience:
    novelty_decay_observed: true
    novelty_tracks_entity_not_text: true        # Novelty keyed on entity_source, not raw content
```

### Ablation Runs

| Condition | What's changed | What we learn |
|---|---|---|
| **Full** | All systems active, SensoryGate active | Baseline: bio-stack adapts to sensory loss |
| **No gate** | SensoryGate disabled — all percepts pass through unmodulated | Does the bio-stack behave differently when it "sees" everything regardless of acuity? If no difference → gate isn't contributing. |
| **No recovery** | sight_acuity stays at 0.0 permanently (no vital_drift) | Does the bio-stack learn to compensate with remaining senses? Are later encounter memories purely auditory/tactile? |
| **Instant recovery** | sight_acuity jumps from 0.0 to 1.0 instantly after healing spring | Does sudden sensory recovery cause prediction errors? NAc predictions trained in blind context may not transfer. |

### File

- `scenarios/campaigns/darkened_cavern_v1.yaml` (~150 lines)
- Included in the DM MVP campaign set alongside heist, poisoned_crown, and arena

---

## Summary

| Phase | What | LOC | Days |
|---|---|---|---|
| **Phase 1** | Critical wiring (9 items) | ~200 | ~1.5 |
| **Phase 1** | **SHIPPED** (6c262c5) | 189 | done |
| **Phase 1.5** | Cascade result surfacing (5 items) | ~120 | ~1 |
| **Phase 2** | Pipeline correctness (10 items) | ~150 | ~1 |
| **Phase 3** | Percept abstraction + entity-modulated perception | ~180 | ~1 |
| **Phase 4** | Design gap fixes (6 items) | ~80 | ~0.5 |
| **Phase 5** | Pipeline audit script | ~250 | ~0.5 |
| **Phase 6** | Sensory ablation campaign YAML | ~150 | ~0.5 |
| **Total** | | **~1,130** | **~6** |

### Implementation Order

1. ~~**Phase 1.1-1.9** (critical wiring) — **SHIPPED** (commit 6c262c5)~~
2. **Phase 1.5** (cascade surfacing) — rich tool results, immediate failure eval, body state in prompt, Cerebellum observation
3. **Phase 2.1** (forming boost) — most impactful correctness fix
4. **Remaining Phase 2** (context_match, decay, concept gate, pain cooldown, priority, truncation, thread safety)
5. **Phase 3** (percept abstraction) — enables rich DM percepts with entity-modulated SensoryGate
6. **Phase 4** (design gaps)
7. **Phase 5** (audit script) — validates everything
8. **Phase 6** (sensory ablation campaign)

### Verification

After all phases, run the audit script. If all checks pass, the bio-system pipeline is production-quality and DM campaigns will exercise the full cognitive architecture.

---

## Phase 7: Bio-System Consolidation — Absorb Dormant Systems (~200 LOC)

Broad repo audit (2026-04-07) found several systems that are defined but not wired, and could either be absorbed into bio-inspired systems or removed as dead code. This phase turns dormant infrastructure into active bio-systems or cleans it up.

### 7.1 Energy → Metabolic Budget (wire into decisions)

**Status:** `energy/` tracks token/cost data but **no bio-system consumes it for decisions.**

In neuroscience, metabolic cost directly gates behavior — fatigue makes you avoid effortful actions, conserve resources, and default to habits. The energy system should feed into:

**Wire energy → NAc (effort-reward learning):**

```python
# In runtime/agent_loop.py, after tool execution alongside NAc.observe():
if nac is not None and energy_registry is not None:
    cost = energy_registry.last_action_cost()
    if cost is not None:
        nac.observe(
            event_type="energy",
            event_signature=f"cost:{tool_name}",
            outcome_type="energy_cost",
            outcome_signature=f"tokens:{cost.tokens}",
            outcome_valence=Valence.NEGATIVE if cost.tokens > 500 else Valence.NEUTRAL,
            delta_seconds=exec_elapsed,
            context={"tool": tool_name, "cost_usd": cost.cost_usd},
        )
```

**Wire energy → Default Network (metabolic gating):**

```python
# In runtime/dn_controller.py, when configuring DN:
# High energy expenditure → DN naturally activates (brain does this — rest when tired)
if energy_registry is not None:
    session_cost = energy_registry.session_total()
    if session_cost and session_cost.cost_usd > budget_threshold * 0.8:
        # Boost DN activation — system is "fatigued"
        dn.set_escalation_threshold(dn.escalation_threshold * 0.7)
```

**LOC:** ~40

### 7.2 Skills → Delete or Fold into Cerebellum

**Status:** `skills/` is entirely DORMANT — `Skill` ABC, `Protocol`, health_reporting, rtsp_streaming, timed_protocol, shredder_segmenter — **none instantiated anywhere in runtime.** ~600 LOC.

Two options:

**Option A: Delete.** Skills are architecturally similar to `embodiment/motor.py` motor programs but for cognitive procedures. The Cerebellum already has `ProgramRegistry` and `ForwardModel`. If cognitive procedural memory is needed later, build it on top of the existing Cerebellum infrastructure rather than maintaining a parallel unused system.

**Option B: Fold into Cerebellum.** Rename "skills" to "cognitive programs" and register them via `ProgramRegistry`. A skill like "health_reporting" becomes a multi-step motor program with abstract (non-physical) steps. This requires extending `MotorProgram` to support non-embodiment steps.

**Recommendation: Option A (delete).** The skill system was designed before Cerebellum existed. If needed, the Cerebellum's motor program model is a better foundation. Remove `skills/` and its 6 files (~600 LOC).

**LOC:** -600 (removal)

### 7.3 Provenance → Fold into Hippocampus Episodic Trace

**Status:** `provenance/` has types, collector, store, and renderer defined — **not wired into the agent loop.** ~300 LOC of unused infrastructure.

Provenance traces ("what did I do, why, what decision was made") are structurally identical to episodic memory. Rather than maintaining a separate system:

- Add optional `decision_rationale` and `tool_choice_context` fields to `EpisodicMemory.perception`
- "Explain what I did" becomes a Hippocampus query with a filter on `decision_rationale is not None`
- The dead `ExplainTool` (see 7.5) would become a specialized Hippocampus recall

**Implementation:**

```python
# In agents/bus.py, Perception dataclass:
decision_rationale: str = ""  # Why this action was chosen (provenance)
tool_alternatives: list[str] = field(default_factory=list)  # What else was considered

# In runtime/agent_loop.py, when recording outcomes:
# Populate from LLM reasoning_carryover if available
perception.decision_rationale = reasoning_carryover[:200] if reasoning_carryover else ""
```

After this, delete `provenance/` (~300 LOC). If the opt-in experimental provenance feature is needed later, it can be rebuilt on top of Hippocampus episodic queries.

**LOC:** ~20 added, ~300 removed. Net: -280

### 7.4 Communication Bridge → Wire or Remove

**Status:** `bridges/communication_bridge.py` is 1 of 8 bridges, defined but **never instantiated** in `MemoryHub.connect()`. The other 7 bridges are all active.

If the `comms/` system (Twilio SMS/voice) is going to be used:
- Add instantiation in `MemoryHub.connect()` when comms gateway is available
- Wire conversation history into Hippocampus for episodic capture of communications

If not planned for near-term:
- Delete the bridge file (~80 LOC) and add it back when comms is activated

**Recommendation:** Delete for now. Comms is optional/experimental and the bridge can be recreated from the existing bridge pattern when needed.

**LOC:** -80

### 7.5 Dead Tools → Remove or Absorb

Three tools are defined but **never registered** anywhere:

| Tool | File | Assessment |
|------|------|-----------|
| `ExplainTool` | `tools/explain.py` | Surfaces provenance traces. If provenance folds into Hippocampus (7.3), this becomes a specialized memory query. **Delete the file** — the functionality will be covered by Hippocampus recall with decision_rationale filter. |
| `PainHistoryTool` | `tools/introspection.py:271` | Queries pain system history. PainBus already publishes to Hippocampus, so pain history is already queryable via memory recall. **Delete the class.** |
| `SceneSummaryTool` | `tools/introspection.py:526` | Simulation-only scene summary. Never exposed to agents. **Delete the class.** |

**LOC:** ~200 removed

---

## Phase 8: Dead Runtime Code Cleanup (~400 LOC)

Broad repo audit found 4 runtime modules with **zero imports anywhere** in the codebase:

| Module | LOC | Why dead | Action |
|--------|-----|----------|--------|
| `runtime/resilient.py` | 68 | `@resilient` decorator — CLAUDE.md references it but it has 0 imports, 0 call sites | Delete. The pattern (`except Exception: pass`) is used inline everywhere instead. |
| `runtime/session.py` | 97 | `AgentSession` crash recovery — superseded by plan persistence and agent state management | Delete. |
| `runtime/debug_status_server.py` | 152 | Old debug HTTP server — explicitly superseded by `leader_proxy.py` (says so in its own header) | Delete. |
| `runtime/monitor_registry.py` | 87 | `MonitorRegistry` abstract polling pattern for PainSignal monitors — PainBus (pub/sub) already handles this better | Delete. |

Additionally:

| Item | LOC | Why dead | Action |
|------|-----|----------|--------|
| `planning/adaptive_planner.set_mesh_context()` | ~30 | Method implemented, **zero callers**. `self._peer_registry` is always None. Mesh peer-sharing code paths in planner are permanently dead. | Remove method + all `if self._peer_registry` branches. |
| `mesh/knowledge.py` exports | ~200 | `ExperienceBroker`, all Provider/Receiver protocols — **never instantiated**. Protocol stubs for future Phase 1+. | Keep as future-proofing but document as stub. No LOC change. |
| `data/agents/MagicMock_*` dirs | 348 dirs | Leaked test fixtures from mock objects during unit testing | Delete all `MagicMock_name_mock.*` directories. Add `.gitignore` pattern. |

**Total LOC removed:** ~434

---

## Final Summary

| Phase | What | LOC | Status |
|---|---|---|---|
| **Phase 1** | Critical wiring (9 items) | 189 | **SHIPPED** (6c262c5) |
| **Phase 1.5** | Cascade result surfacing (5 items) | 120 | **SHIPPED** (8713500) |
| **Phase 2** | Pipeline correctness (10 items) | 58 | **SHIPPED** (ce94cc8) |
| **Phase 3** | Percept abstraction + SensoryGate | 613 | **SHIPPED** (6dddf54) |
| **Phase 4** | Design gap fixes | 36 | **SHIPPED** (7c12b5c) |
| **Phase 5** | Pipeline audit script (14/14 passing) | 507 | **SHIPPED** (b71a679) |
| **Phase 6** | Sensory ablation campaign | — | **Moved to DM MVP** |
| **Phase 7** | Bio-system consolidation | ~-700 net | **SHIPPED** |
| **Phase 8** | Dead runtime code cleanup | ~-400 | **SHIPPED** |

**Net result:** ~1,100 LOC of dead code removed, ~1,500 LOC of bio-system wiring + percept abstraction added. Pipeline audit validates 14 checks, all passing.

---

## Ties to Other Plans

| Plan | Relationship |
|------|-------------|
| **DM MVP** | Gate — this plan must ship before DM campaigns are meaningful |
| **Mode Refactor** | Independent — can run in parallel. Mode refactor removes ~1,800 LOC of behavioral steering; this plan removes ~1,154 LOC of dead/dormant code. Together they cut ~3,000 LOC. |
| **Generative Campaigns** | Benefits — all campaign modes get better bio-system support |
| **Agent Mesh Phase 4** | Benefits — knowledge sharing assumes NAc/ATL actually learn |
| **Research Protocol** | Benefits — Writer/Reviewer agents get richer bio-system data to report on |
| **Realtime Refinement** | This IS the final refinement stage |
| **Embodiment Core** | Benefits — Cerebellum wiring in loop applies to real embodiment too |
