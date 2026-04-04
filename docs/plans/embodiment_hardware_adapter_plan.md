# Embodiment Hardware Adapter Plan

> **Status:** Not started. **Blocked on:** `embodiment_core_plan.md` (must ship and validate MVP first).
> **Scope:** one sprint, ~300 LOC. Adapter-only — no refactor of existing hardware code.

---

## Goal

Bridge the **Embodiment layer** (delivered by Embodiment Core) to real hardware (`RobotController` / `RobotState`) via an adapter pattern. Enables the Cerebellum to learn forward models against real sensors and existing robot control to appear in the unified Embodiment representation.

**Explicitly not a refactor.** We do not rewrite `RobotController`, touch `RobotState.current_pose`, or modify `MovementTracker`. We wrap, not replace.

---

## Why Adapter, Not Refactor

A full refactor (migrating all hardware code paths through the Embodiment layer) is a 3+ week project with high breakage risk across existing tests and live-robot deployments. The adapter pattern delivers ~90% of the value (unified representation for new code) at ~10% of the cost, with a clean rollback path if it doesn't pan out.

If the adapter proves itself, a full refactor becomes its own separate plan informed by real usage.

---

## Deliverables

- `src/maxim/embodiment/backends/hardware.py` — `HardwareBackend` wrapping `RobotController`
- `Embodiment.sync_from_robot_state(state: RobotState)` — pulls current pose into component state
- `MovementTracker` optionally publishes to embodiment observers (new subscription, old API untouched)
- `scenarios/embodiment/hardware_live_baseline.yaml` — validation scenario requiring live robot
- `tests/integration/test_embodiment_hardware.py` — smoke tests

---

## Design

```
┌────────────────────────┐      ┌──────────────────────┐
│  Existing hardware     │      │  Existing tests      │
│  code paths            │      │  (unchanged)         │
│  (unchanged)           │      │                      │
└──────────┬─────────────┘      └──────────────────────┘
           │
           │ reads/writes
           ↓
┌──────────────────────────────────────────────────────┐
│ RobotController / RobotState / MovementTracker       │
│ (existing, untouched)                                 │
└──────────┬───────────────────────────────────────────┘
           │
           │ observed via adapter
           ↓
┌──────────────────────────────────────────────────────┐
│ HardwareBackend (new adapter)                         │
│   - reads RobotState.current_pose                    │
│   - translates affordances → MotionTarget             │
│   - publishes sensor values to Embodiment             │
└──────────┬───────────────────────────────────────────┘
           │
           │ feeds
           ↓
┌──────────────────────────────────────────────────────┐
│ Embodiment (from Core plan)                           │
│   - Cerebellum learns from real sensors               │
│   - New code reads via embodiment API                 │
└──────────────────────────────────────────────────────┘
```

**Key insight:** Embodiment becomes an *observer* of hardware state, not a *replacement*. Existing callers keep working. New callers use the unified API.

---

## Integration Points

1. **Component state sync.** `Embodiment.sync_from_robot_state()` maps `RobotState.current_pose` keys to component state entries (`head_yaw` → `head.yaw` component).
2. **Affordance dispatch.** When embodiment executes an affordance on a hardware-backed component, `HardwareBackend` translates it into a `MotionTarget` and calls `RobotController.goto_target()`.
3. **Sensor readings.** Real sensor values (pose, battery, temperature) flow into embodiment as percepts via the normal `EmbodimentPerceptSource` path.
4. **Movement tracking integration.** `MovementTracker` gains an optional `on_metrics` callback that pushes metrics to embodiment observers. Existing pain-detection path remains intact and independent.

---

## Success Criteria

1. **Live Reachy Mini test:** embodiment pose readings match `RobotState.current_pose` within 1 tick.
2. **Cerebellum learning on real hardware:** after 50 motor commands, Cerebellum forward models predict head position with MAE < 5° on held-out commands.
3. **Zero regression:** all existing hardware integration tests pass unchanged. No performance delta > 5% on live-robot tick rate.

---

## Future (Not In Scope)

If adapter proves itself and shows pain points over time:
- **Full refactor** — migrate all hardware code paths through Embodiment as the canonical representation
- **Inverse models** — Embodiment plans motor commands from sensory goals (biological cerebellum does this)
- **Cross-modal learning** — Cerebellum models that predict vision percepts from motor commands

These become their own plans only after the adapter is validated in practice.

---

## Risks

1. **Sync lag.** If `sync_from_robot_state()` lags behind the control loop, Cerebellum trains on stale state. **Mitigation:** sync is called in the same tick as state read; benchmark and enforce latency budget.
2. **Affordance translation ambiguity.** Mapping Embodiment affordances to `MotionTarget` may be lossy. **Mitigation:** keep mapping deterministic; add test coverage for each affordance type.
3. **Double pain fire.** Both `PainDetector` (via MovementTracker) and Embodiment failure modes could fire pain for the same event. **Mitigation:** Embodiment defers to existing PainDetector for motor-derived pain; only fires for failure modes PainDetector doesn't cover (e.g., fatigue accumulation).

---

## No Blockers Once Core Ships

Depends only on:
- Embodiment Core (MVP + Cerebellum + Failures) shipped and validated ✓ (blocking)
- Existing `RobotController` ABC + implementations ✓
- Existing `RobotState` dataclass ✓
- Existing `MovementTracker` ✓
- Existing `PainDetector` + `PainBus` ✓
