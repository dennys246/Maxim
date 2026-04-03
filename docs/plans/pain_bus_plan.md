# Pain Bus Plan

Extract a standalone `PainBus` from PainDetector's internal callback mechanism, enabling pain signals from any source — motor, tool, simulation, energy, cognitive — to flow through a single pub/sub channel to all consumers.

---

## Motivation

Pain today is a closed loop tied to physical movement:

```
MovementTracker → PainDetector._check_for_pain() → _emit_pain() → callbacks
Tool executor   → PainDetector.record_tool_error() ─────────────┘
MonitorRegistry → PainDetector._emit_pain() ─────────────────────┘
```

Three problems:

1. **Simulated percepts can't inject pain.** There's no entry point that accepts a Percept with `source: proprioception`. The malware+pain scenario in the percept simulation plan has no path to trigger pain memory formation.

2. **Non-motor pain has no clean entry.** Energy exhaustion, cognitive overload, and repeated LLM failures have no way to emit pain signals except the `record_tool_error()` backdoor, which was designed for tool failures and carries tool-specific fields.

3. **Pain doesn't reach Hippocampus directly.** Pain → memory today is indirect: through NAc causal links (learned association) or tool reflexion (LLM-generated self-critique). There's no direct episodic memory capture of "pain happened here, this is what it felt like." The malware+pain scenario expects `memory_formed` with pain content — that path doesn't exist.

---

## Current Architecture (Before)

### Pain Sources (3 entry points)

| Source | Entry Point | How it works |
|--------|------------|--------------|
| Motor movement | `PainDetector.record_position()` | MovementTracker computes metrics → `_check_for_pain()` → `_emit_pain()` |
| Tool execution | `PainDetector.record_tool_error()` | Executor reports failure → PainDetector maps to PainType → `_emit_pain()` |
| Monitor registry | `MonitorRegistry` polling → `PainDetector._emit_pain()` | Background thread polls SignalMonitors, forwards PainSignals directly |

### Pain Consumers (3 callback subscribers)

| Consumer | Registration Site | What it does |
|----------|------------------|--------------|
| `PainCircuitBridge._on_pain()` | `pain_bridge.py:125` | Reports to NAc as negative outcome, builds causal links |
| `ToolPainBridge._on_pain()` | `tool_pain_bridge.py:74` | Records tool-specific context, triggers reflexion, stores in Hippocampus via reflection |
| `MonitorRegistry` callbacks | `monitor_registry.py:41` | Forwards SignalMonitor outputs to `_emit_pain()` |

### Pain Consumers (indirect, via bridge queries)

| Consumer | How it queries | What it does |
|----------|---------------|--------------|
| `FearAgent` | `pain_bridge.should_gate_action()` | Blocks movement if predicted pain risk > threshold |
| Motor control (`movement.py`) | `pain_bridge.get_pain_risk()` | Clamps movement amplitude by up to 80%, triggers turn-around |

### The bottleneck

Everything flows through `PainDetector._emit_pain()`:

```python
# pain.py:416-448 — the actual emission logic
def _emit_pain(self, signal: PainSignal) -> PainSignal:
    with self._lock:
        self._last_pain_time[signal.pain_type] = signal.timestamp
        self._total_pain_signals += 1
        self._pain_counts[signal.pain_type] = ...
    # Log
    # Notify callbacks (outside lock)
    for callback in self._callbacks:
        callback(signal)
    return signal
```

This is already a pub/sub pattern — it's just trapped inside PainDetector. `_emit_pain` is even called directly by MonitorRegistry as a callback. The extraction is mechanical.

---

## Proposed Architecture (After)

### New: PainBus (extracted from PainDetector)

```
SOURCES                              CONSUMERS
─────────                            ──────────
PainDetector (motor)  ──┐
Tool executor         ──┤            ┌──→ PainCircuitBridge._on_pain()  [NAc learning]
MonitorRegistry       ──┤            ├──→ ToolPainBridge._on_pain()     [tool learning]
Percept pipeline  NEW ──┼── PainBus ─┼──→ Hippocampus capture      NEW [episodic memory]
Energy registry   NEW ──┤            ├──→ FearAgent (via bridge queries)
Cognitive signals NEW ──┤            └──→ Motor control (via bridge queries)
Other robot       NEW ──┘
```

### What changes

| Component | Before | After |
|-----------|--------|-------|
| `PainDetector._emit_pain()` | Owns callback list, does stats + logging + dispatch | Delegates to `pain_bus.publish()` after stats + logging |
| `PainDetector.add_pain_callback()` | Manages its own callback list | Removed — consumers subscribe to PainBus directly |
| `PainCircuitBridge.__init__` | Calls `detector.add_pain_callback(self._on_pain)` | Calls `pain_bus.subscribe(self._on_pain)` |
| `ToolPainBridge.__init__` | Calls `pain_detector.add_pain_callback(self._on_pain)` | Calls `pain_bus.subscribe(self._on_pain)` |
| `MonitorRegistry` | Calls `pain_detector._emit_pain` as callback | Calls `pain_bus.publish()` as callback |
| Agent loop | No pain awareness | Routes `source: proprioception` Percepts → `pain_bus.publish()` |
| Hippocampus | No direct pain subscription | Subscribes to PainBus, captures high-intensity pain as episodes |
| EnergyRegistry | No pain integration | Publishes `RESOURCE_EXHAUSTION` pain on budget depletion |

---

## Implementation

### Step 1: PainBus (~60 lines)

**New file:** `src/maxim/proprioception/pain_bus.py`

```python
"""Central publish/subscribe for pain signals from any source.

Extracted from PainDetector's internal callback mechanism to allow
pain signals from motor, tool, simulation, energy, and cognitive
sources to reach all consumers through a single channel.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Callable

from maxim.proprioception.pain import PainSignal

logger = logging.getLogger(__name__)


class PainBus:
    """Central publish/subscribe for pain signals.

    Any system can publish a PainSignal. All subscribers are notified
    synchronously (consistent with existing PainDetector behavior).

    Example:
        bus = PainBus()
        bus.subscribe(lambda sig: print(f"Pain: {sig.pain_type}"))
        bus.publish(PainSignal(
            pain_type=PainType.EXTERNAL_SIGNAL,
            intensity=0.7,
            timestamp=time.time(),
        ))
    """

    def __init__(self, history_size: int = 200) -> None:
        self._subscribers: list[Callable[[PainSignal], None]] = []
        self._lock = threading.Lock()
        self._history: deque[PainSignal] = deque(maxlen=history_size)
        self._total_published: int = 0

    def subscribe(self, callback: Callable[[PainSignal], None]) -> None:
        """Register a pain signal consumer."""
        with self._lock:
            self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[PainSignal], None]) -> None:
        """Remove a previously registered consumer."""
        with self._lock:
            if callback in self._subscribers:
                self._subscribers.remove(callback)

    def publish(self, signal: PainSignal) -> None:
        """Publish a pain signal to all subscribers.

        Callbacks are invoked outside the lock to prevent deadlocks
        with subscribers that query the bus.
        """
        with self._lock:
            self._history.append(signal)
            self._total_published += 1
            subscribers = list(self._subscribers)

        for callback in subscribers:
            try:
                callback(signal)
            except Exception as e:
                logger.warning("PainBus subscriber error: %s", e)

    @property
    def recent(self) -> list[PainSignal]:
        """Recent pain signals (newest last)."""
        with self._lock:
            return list(self._history)

    def recent_by_type(self, pain_type: "PainType") -> list[PainSignal]:
        """Recent signals filtered by type."""
        with self._lock:
            return [s for s in self._history if s.pain_type == pain_type]

    def get_stats(self) -> dict[str, int]:
        """Bus-level statistics."""
        with self._lock:
            return {
                "total_published": self._total_published,
                "subscriber_count": len(self._subscribers),
                "history_size": len(self._history),
            }
```

### Step 2: New PainTypes (~10 lines)

**Edit:** `src/maxim/proprioception/pain.py`

Add to `PainType` enum:

```python
class PainType(Enum):
    # Existing motor types (unchanged)
    EXCESSIVE_VELOCITY = "excessive_velocity"
    DIRECTION_THRASHING = "direction_thrashing"
    SUSTAINED_STRAIN = "sustained_strain"
    EXCESSIVE_ACCELERATION = "excessive_acceleration"
    MOVEMENT_FAILURE = "movement_failure"

    # Existing tool types (unchanged)
    TOOL_FAILURE = "tool_failure"
    TOOL_TIMEOUT = "tool_timeout"
    TOOL_INVALID_INPUT = "tool_invalid_input"
    TOOL_SUSTAINED = "tool_sustained"

    # NEW: non-motor sources
    RESOURCE_EXHAUSTION = "resource_exhaustion"    # Energy budget depleted
    COGNITIVE_OVERLOAD = "cognitive_overload"       # Repeated failures, context loss
    EXTERNAL_SIGNAL = "external_signal"             # From percept/simulation/other robot
    SAFETY_VIOLATION = "safety_violation"            # FearAgent-detected threat
```

### Step 3: PainDetector delegates to PainBus (~30 lines changed)

**Edit:** `src/maxim/proprioception/pain.py`

PainDetector gains an optional `pain_bus` parameter. If provided, `_emit_pain()` publishes to the bus after doing its own stats/logging. If not provided, it falls back to its internal callback list (backward compat).

```python
class PainDetector:
    def __init__(
        self,
        config: PainConfig | None = None,
        movement_tracker: MovementTracker | None = None,
        pain_bus: PainBus | None = None,           # NEW
    ) -> None:
        ...
        self._pain_bus = pain_bus
        self._callbacks: list[...] = []  # Kept for backward compat

    def _emit_pain(self, signal: PainSignal) -> PainSignal:
        # Stats + logging (unchanged)
        with self._lock:
            self._last_pain_time[signal.pain_type] = signal.timestamp
            self._total_pain_signals += 1
            self._pain_counts[signal.pain_type] = ...

        # Logging (unchanged)
        ...

        # Dispatch: prefer PainBus, fall back to internal callbacks
        if self._pain_bus is not None:
            self._pain_bus.publish(signal)
        else:
            for callback in self._callbacks:
                try:
                    callback(signal)
                except Exception as e:
                    logger.warning("Pain callback error: %s", e)

        return signal
```

`add_pain_callback()` and `remove_pain_callback()` remain for backward compat but log a deprecation warning if a PainBus is configured.

### Step 4: Rewire consumers to PainBus (~25 lines changed across 3 files)

**`bridges/pain_bridge.py`** — PainCircuitBridge:

PainCircuitBridge has 10+ references to `self._detector` beyond the callback — it calls `set_movement_target()`, `clear_movement_target()`, and `get_stats()` for movement tracking and harm prediction. The bridge needs PainDetector for movement tracking AND PainBus for pain callbacks. The fix is to add `pain_bus` as a *new* constructor parameter, not a replacement:

```python
# Before:
def __init__(self, nac, pain_detector, config=None):
    self._detector = pain_detector
    self._detector.add_pain_callback(self._on_pain)

# After:
def __init__(self, nac, pain_detector, pain_bus, config=None):
    self._detector = pain_detector  # KEPT — needed for movement tracking
    pain_bus.subscribe(self._on_pain)  # pain callbacks via bus now
```

All `self._detector.set_movement_target()` / `clear_movement_target()` / `get_stats()` calls remain unchanged — they still go through PainDetector. Only the pain callback subscription moves to the bus.

**`bridges/tool_pain_bridge.py`** — ToolPainBridge:

Clean swap — ToolPainBridge has no references to `pain_detector` beyond the callback registration at line 74.

```python
# Before:
pain_detector.add_pain_callback(self._on_pain)

# After:
pain_bus.subscribe(self._on_pain)
```

**Note:** ToolPainBridge instantiation location is not in DefaultNetwork — search for `ToolPainBridge(` across the codebase to locate the actual init site before implementing.

**`runtime/monitor_registry.py`** — MonitorRegistry:
```python
# Before:
registry.add_signal_callback(pain_detector._emit_pain)

# After:
registry.add_signal_callback(pain_bus.publish)
```

### Step 5: Percept → PainBus routing (~30 lines)

**Edit:** `src/maxim/runtime/agent_loop.py` (or a new `simulation/pain_routing.py` helper)

When a Percept arrives with pain content, convert to PainSignal and publish:

```python
def route_pain_percept(percept: Percept, pain_bus: PainBus) -> None:
    """Convert proprioception Percepts to PainSignals on the bus."""
    if percept.source != "proprioception" or percept.content != "pain_signal":
        return

    meta = percept.metadata or {}
    try:
        pain_type = PainType(meta.get("pain_type", "external_signal"))
    except ValueError:
        pain_type = PainType.EXTERNAL_SIGNAL

    signal = PainSignal(
        pain_type=pain_type,
        intensity=meta.get("intensity", 0.5),
        timestamp=percept.timestamp,
        angular_velocity=meta.get("angular_velocity", 0.0),
        translation_velocity=meta.get("translation_velocity", 0.0),
        direction_reversals=meta.get("direction_reversals", 0),
        context={k: v for k, v in meta.items() if k not in {
            "pain_type", "intensity", "angular_velocity",
            "translation_velocity", "direction_reversals",
        }},
    )
    pain_bus.publish(signal)
```

Called in the agent loop's percept processing path. This is the bridge that makes simulated percepts work.

### Step 6: PainBus → Hippocampus subscriber (~40 lines)

**New function** wired during `MemoryHub` or agent loop initialization:

```python
def create_pain_memory_subscriber(
    hippocampus: Hippocampus,
    intensity_threshold: float = 0.4,
) -> Callable[[PainSignal], None]:
    """Create a PainBus subscriber that captures pain as episodic memory.

    Args:
        hippocampus: Hippocampus instance for memory capture.
        intensity_threshold: Minimum pain intensity to trigger memory formation.
    """
    def _on_pain(signal: PainSignal) -> None:
        if signal.intensity < intensity_threshold:
            return

        hippocampus.capture(
            perception=Perception(
                observations={
                    "pain_type": signal.pain_type.value,
                    "intensity": signal.intensity,
                    **signal.context,
                },
                salience=min(signal.intensity + 0.2, 1.0),
                novelty=0.6,
            ),
            decision=Decision(
                intent={"goal": "pain_response"},
                reasoning=f"Pain detected: {signal.pain_type.value} "
                          f"(intensity={signal.intensity:.2f})",
            ),
            outcome=Outcome(
                success=False,
                result={
                    "pain_type": signal.pain_type.value,
                    "intensity": signal.intensity,
                    "context": signal.context,
                },
            ),
        )

    return _on_pain
```

Registration:
```python
pain_bus.subscribe(create_pain_memory_subscriber(hippocampus))
```

### Step 7: Energy → PainBus integration (~35 lines)

**Issue:** EnergyRegistry does not have an `add_depletion_callback()` method. It only has a generic `add_callback()` that fires on every `record_signal()` call. The energy publisher must filter signals manually.

**Edit:** Energy registry callback (wired during initialization):

```python
def create_energy_pain_publisher(
    pain_bus: PainBus,
    depletion_intensity: float = 0.6,
    depletion_threshold: float = 0.05,
) -> Callable:
    """Publish RESOURCE_EXHAUSTION pain when an energy budget is nearly empty.

    Subscribes to EnergyRegistry's generic add_callback(). Filters
    incoming signals to only fire pain when remaining fraction is
    below depletion_threshold.
    """
    _fired: set[str] = set()  # Track which resources already fired (prevent spam)

    def _on_energy_signal(signal) -> None:
        resource = signal.resource
        remaining = signal.remaining_fraction
        if remaining > depletion_threshold:
            _fired.discard(resource)  # Reset if resource recovers
            return
        if resource in _fired:
            return  # Already fired for this resource
        _fired.add(resource)
        pain_bus.publish(PainSignal(
            pain_type=PainType.RESOURCE_EXHAUSTION,
            intensity=depletion_intensity,
            timestamp=time.time(),
            context={"resource": resource, "remaining": remaining},
        ))
    return _on_energy_signal
```

**Registration** uses the generic callback API:
```python
energy_registry.add_callback(create_energy_pain_publisher(pain_bus))
```

This means energy depletion triggers the same learning loop as physical pain. NAc learns to avoid actions that exhaust budgets. Hippocampus records "we ran out of tokens during phase 3."

### Step 8: Wiring (~30 lines)

**Where PainBus is created and wired.** The natural place is wherever PainDetector is currently created — in `DefaultNetwork.__init__()` or the agentic runtime initialization.

```python
# Create the bus (single instance, shared)
pain_bus = PainBus()

# Create detector with bus
pain_detector = PainDetector(config=pain_config, pain_bus=pain_bus)

# PainCircuitBridge keeps detector reference for movement tracking,
# subscribes to bus for pain callbacks
pain_bridge = PainCircuitBridge(
    nac=nac,
    pain_detector=pain_detector,  # kept for set_movement_target(), get_stats()
    pain_bus=pain_bus,             # new: pain callbacks via bus
    ...
)

# ToolPainBridge only needs the bus (no detector coupling)
tool_pain_bridge = ToolPainBridge(pain_bus=pain_bus, ...)

# NEW consumers
pain_bus.subscribe(create_pain_memory_subscriber(hippocampus))

# Energy integration (uses generic add_callback with filtering)
energy_registry.add_callback(create_energy_pain_publisher(pain_bus))

# Monitor registry
monitor_registry.add_signal_callback(pain_bus.publish)
```

**Pre-implementation task:** Before starting Step 8, grep for `ToolPainBridge(` and `MonitorRegistry(` across the codebase to locate all actual initialization sites. ToolPainBridge is NOT wired in DefaultNetwork — it's instantiated elsewhere (likely agentic_runtime.py or agent_loop.py).

---

## Data Flow: Before vs After

### Before (current)

```
Motor position ──→ PainDetector ──→ _emit_pain() ──→ [internal callbacks]
                                                       ├─→ PainCircuitBridge (NAc)
Tool error     ──→ PainDetector ──→ _emit_pain() ──→  ├─→ ToolPainBridge (learning)
                                                       └─→ (nothing else)
MonitorRegistry ─→ PainDetector._emit_pain() ──────→

Percept(source=proprioception)  →  ??? (no path)
Energy depletion                →  ??? (no path)
Hippocampus                     ←  ??? (no direct subscription)
```

### After (with PainBus)

```
Motor position ──→ PainDetector ──→ pain_bus.publish() ──→ [subscribers]
Tool error     ──→ PainDetector ──→ pain_bus.publish()      ├─→ PainCircuitBridge (NAc)
MonitorRegistry ─→ pain_bus.publish() ───────────────────→  ├─→ ToolPainBridge (learning)
Percept pipeline → route_pain_percept() → pain_bus.publish()├─→ Hippocampus (episodic) NEW
Energy registry  → pain_bus.publish() ───────────────────→  └─→ (future: any subscriber)
```

---

## What This Enables for Percept Simulation

The malware+pain scenario from the percept simulation plan now works:

```yaml
percepts:
  - at: 0.5
    source: proprioception
    content: pain_signal
    metadata:
      pain_type: joint_strain        # → PainType.EXTERNAL_SIGNAL (or map to existing type)
      joint: head_pitch
      intensity: 0.8
      velocity: 2.1
```

**Flow:** ScenarioSource emits Percept → agent loop calls `route_pain_percept()` → PainBus.publish() → all subscribers fire:
- PainCircuitBridge records NAc negative outcome
- ToolPainBridge records context (if tool was active)
- **Hippocampus captures episodic memory with pain content** (validates `memory_formed` expectation)
- FearAgent updates risk predictions

The `action_blocked` expectation (FearAgent blocks malware) works through the existing ExecAgent → FearAgent path. The `memory_formed` expectation now works through PainBus → Hippocampus.

---

## Files Changed

| File | Change | Lines |
|------|--------|-------|
| `proprioception/pain_bus.py` | **NEW** — PainBus class | ~60 |
| `proprioception/pain.py` | Add 4 PainTypes, add `pain_bus` param to PainDetector, delegate `_emit_pain()` | ~30 changed |
| `bridges/pain_bridge.py` | Add `pain_bus` constructor param, subscribe to bus for callbacks (keep `_detector` for movement tracking) | ~10 changed |
| `bridges/tool_pain_bridge.py` | Replace `pain_detector.add_pain_callback` with `pain_bus.subscribe` | ~5 changed |
| `runtime/monitor_registry.py` | Forward to `pain_bus.publish` instead of `_emit_pain` | ~3 changed |
| `runtime/agent_loop.py` | Add `route_pain_percept()` call in percept processing | ~30 added |
| `integration/memory_hub.py` or init site | Wire `create_pain_memory_subscriber()` | ~20 added |
| Energy init site | Wire `create_energy_pain_publisher()` (generic callback with depletion filter) | ~25 added |
| Wiring site (DefaultNetwork + agentic_runtime) | Create PainBus, pass to constructors, locate ToolPainBridge init site | ~20 changed |

**Total: ~205 lines new/changed, 1 new file**

---

## Implementation Order

| Step | What | Risk | Size |
|------|------|------|------|
| 1 | `pain_bus.py` — standalone, no dependencies | None | ~60 lines |
| 2 | Add new PainTypes to enum | None (additive) | ~10 lines |
| 3 | PainDetector delegates to PainBus | Low — backward compat via fallback | ~30 lines |
| 4 | Rewire 3 consumers to PainBus | Low — same callback signature | ~15 lines |
| 5 | Percept → PainBus routing | Low — additive to agent loop | ~30 lines |
| 6 | PainBus → Hippocampus subscriber | Low — additive | ~40 lines |
| 7 | Energy → PainBus integration | Low — additive | ~25 lines |
| 8 | Wiring at init sites | Medium — touches initialization order | ~30 lines |

Steps 1-4 are the mechanical extraction — zero behavior change, just rewiring. Steps 5-6 are what make the percept simulation plan work. Steps 7-8 are progressive enhancements.

**Validate after Step 4:** All existing tests pass. Pain detection, NAc learning, motor gating, and tool pain all work identically — just routed through PainBus instead of PainDetector's internal list.

**Validate after Step 6:** Run the malware+pain scenario. Verify:
- FearAgent blocks the malware request
- Pain signal appears in Hippocampus as episodic memory
- `pain_bus.recent` contains the signal
- NAc records negative outcome

---

## Dependencies

| This plan | Depends on | Reason |
|-----------|-----------|--------|
| Steps 1-4 | Nothing | Pure extraction, can do today |
| Steps 5-6 | Percept Simulation Plan (Phase 1) | Needs `PerceptSource` wired into agent loop |
| Step 7 | Energy registry depletion callback | May need small addition to EnergyRegistry |
| Malware+pain scenario | This plan Steps 1-6 + Percept Sim Steps 1-7 | Both needed for end-to-end validation |

**Recommended sequence across all plans:**

```
Modularization Phase 0 ✓ (done)
Modularization Phase 1      ← split monolithic files
Pain Bus Steps 1-4           ← extract PainBus (can overlap with modularization)
Percept Simulation Steps 1-4 ← protocols + agent loop wiring
Pain Bus Steps 5-6           ← percept→pain routing + hippocampus subscriber
Percept Simulation Steps 5-7 ← runner + first scenario + CLI
Pain Bus Steps 7-8           ← energy integration + full wiring
Run malware+pain scenario    ← first proof of concept
Write more scenarios         ← proof of concept portfolio
Modularization Phase 2       ← medium splits
Simulation docs              ← document what works
Context upgrade Part 2 v2+   ← measure and improve LLM decisions
Multi-LLM scaling            ← only if profiling shows need
```
