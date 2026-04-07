# Embodiment Core Plan

> **Status:** Not started. MVP-first — validates core loop before committing to further phases.
> **Depends on:** ATL semantic memory (exists), PainBus (exists), Hippocampus (exists), NAc (exists), ToolPainBridge (exists), PerceptSource protocol (exists), LLMRouter (exists), EnergySignal (exists).
> **Related plans:** `dungeon_master_persona.md` (downstream consumer — DM's `CharacterState` mirrors body-state primitives), `agent_mesh.md` (adds EmbodimentCapability), `benchmark_plan.md` (Tier 3 metrics defined there, implementation details here).
> **Includes:** Hardware adapter (formerly `embodiment_hardware_adapter_plan.md`) merged as Phase 3.
>
> **Benchmark integration:** When each phase ships, add the corresponding Tier 3 metric computation to `AUTIntrospector.embodiment_stats()` and create benchmark scenarios in `scenarios/benchmarks/`. See `benchmark_plan.md` Tier 3 section for the metric interface.

---

## Core Insight

Maxim's ATL and proprioception subsystems were built for grounded math about a body. ATL holds canonical knowledge (joint ranges, torque curves, pain thresholds) as semantic concepts with IPS statistics and Angular Gyrus geometry. A new **Cerebellum** module stores lightweight forward models that predict sensory consequences deterministically. The LLM is consulted **only when no forward model exists** — it teaches the cerebellum, then fades out.

This fixes the biggest risk of LLM-imagined percepts (inconsistent physics at runtime) by making the LLM a **teacher**, not a per-tick oracle.

---

## Architecture

```
             Agent proposes action
                      ↓
              ┌───────────────────┐
              │   Cerebellum      │  ← forward models (fast, deterministic)
              │   (predict)       │
              └─────────┬─────────┘
                        │
           ┌────────────┴────────────┐
           │                         │
    Model exists?              No model?
           │                         │
           ↓                         ↓
   Predict percepts         Consult LLM (ATL-grounded)
           │                         │
           │                         └──→ train Cerebellum on result
           ↓                         ↓
           └─────────────┬───────────┘
                         ↓
                  Predicted percepts
                         ↓
               Apply affordance in backend
                         ↓
                  Actual percepts
                         ↓
              Error = predicted - actual
                         ↓
              Cerebellum learns (R-W update)
              PainBus fires via ToolPainBridge (if threshold)
                         ↓
              Percepts → MemoryAgent → Hippocampus → NAc
```

### Layer Responsibilities

| Layer | Role | Backed By |
|-------|------|-----------|
| **EmbodimentSpec** | Declarative body (components, sensors, affordances, failures) | YAML files |
| **ATL Body Concepts** | Canonical physical knowledge (ranges, torque, wear, pain) | New `body_part` concept category |
| **Cerebellum** | Learned forward models per (component, affordance, param_bucket) | New module, Rescorla-Wagner-style predictors |
| **LLMBackend** | Generates percepts for novel situations; teaches Cerebellum | Existing LLMRouter |
| **RuleBackend** | Deterministic physics for simple cases (joint limits, damping) | Pure Python |
| **Embodiment** | Runtime: holds component state, dispatches to backends, tracks failures | New class |

### Hardware Backend: Phase 3

Hardware integration (`HardwareBackend` wrapping `RobotController`) is **deferred to Phase 3** (see below). Phases 0-2 ship and validate the software stack first.

---

## Phase 0 — MVP with ATL Grounding (Gate) (~400 LOC)

**Goal:** prove the core loop produces stable, learnable (action → pain) pairs with ATL grounding active from day one.

**Why merged with ATL grounding:** the original plan split MVP (ungrounded) from Phase 1 (grounded). But we can only know if the architecture works *once grounded* — the ungrounded MVP has no meaningful success criterion. Ship grounded or don't ship.

### Deliverables

- `src/maxim/embodiment/spec.py` — `SensorSpec`, `AffordanceSpec`, `ComponentSpec`, `EmbodimentSpec`, `FailureModeSpec` dataclasses
- `src/maxim/embodiment/body.py` — `Embodiment` runtime with structured state, component tree traversal, R-W-style vital-metric drift
- `src/maxim/embodiment/percepts.py` — `EmbodimentPerceptSource(PerceptSource)` adapter
- `src/maxim/embodiment/llm_backend.py` — LLM percept generation with ATL-injected context
- `src/maxim/embodiment/atl_integration.py` — new `body_part` ATL concept category + `BioContext.lookup_body_concept(name)`
- `scenarios/embodiment/robot_arm_3dof.yaml` — demo spec
- `scenarios/embodiment/embodiment_baseline.yaml` — regression test scenario (reuses refinement metric types)
- `tests/unit/test_embodiment_mvp.py`

### Hard Constraints

- **Fixed failure vocabulary.** 6 base modes only: `overextension`, `overheating`, `strain`, `fatigue`, `impact`, `exhaustion`.
- **Structured trigger format, no eval.** `{field: "angle", op: ">", value: 175, pain: 0.8}`.
- **No homeostasis.** Vital metrics drift linearly only.
- **LLM called once per action**, not per tick.
- **Failures route through existing `ToolPainBridge`** (extended to accept embodiment-sourced failures — ~20 LOC change).

### Success Criteria (Relative, Not Arbitrary)

1. **Grounding A/B comparison:** pain intensity σ across 10 repetitions of same action must be **≥50% lower with ATL grounding enabled** compared to ungrounded baseline. Measured on `embodiment_baseline.yaml`.
2. **NAc learning:** after running `embodiment_baseline.yaml`'s forced-bounds-violation sequence, NAc must learn (action → pain) link with confidence > 0.5 within 3 repetitions.
3. **Latency budget:** percept generation p95 < 2s per action (verified via `response_latency_ms` expectation type from refinement harness).

### Validation Approach

Use the existing `scenarios/refinement_baseline.yaml` pattern. Add embodiment-specific scenarios that use the **same metric expectation types** (`action_count_range`, `tool_success_rate`, `response_latency_ms`) already wired in `validation.py`. Add one new expectation type: `nac_convergence` (asserts causal link confidence ≥ threshold within N repetitions).

**If MVP fails** (σ reduction < 50% OR NAc doesn't converge), stop and revisit architecture. The rest of the plan is contingent on this working.

---

## Phase 1 — Cerebellum (Forward Models) (~400 LOC)

**Goal:** replace LLM percept calls with learned deterministic predictors wherever possible.

### Design

```python
class Cerebellum:
    """Forward models for predicting sensory consequences of actions.

    Stores lightweight predictors per (component, affordance, param_bucket).
    Each predictor learns via prediction-error feedback (Rescorla-Wagner style,
    mirroring NAc but for sensory prediction instead of reward).
    """
    def predict(
        self,
        component: str,
        affordance: str,
        params: dict,
    ) -> list[Percept] | None:
        """Return predicted percepts or None if no model exists."""

    def observe(
        self,
        key: ModelKey,
        predicted: list[Percept],
        actual: list[Percept],
    ) -> None:
        """Update forward model from prediction error."""

    def has_model(self, component: str, affordance: str, params: dict) -> bool

    def get_confidence(self, component: str, affordance: str, params: dict) -> float
    def get_variance(self, component: str, affordance: str, params: dict) -> dict
    def prune_stale_models(self, max_age_s: float) -> int
    def export_state(self) -> dict      # for persistence
    def import_state(self, data: dict) -> None
```

### Model Structure (per key)

- Expected sensor values (mean, variance) per sensor
- Expected failure probabilities per failure mode
- Confidence (grows with observations)
- Last-observed timestamp (for pruning)

### Prediction Policy

- Confidence < 0.3 → use LLM, observe, train Cerebellum
- Confidence ≥ 0.3 → use Cerebellum prediction
- High-variance models → fall back to LLM (uncertain predictions need grounding)

### Bucket Granularity Decision

**Highly specific (component, affordance, param_bucket) keys.** No generalization at the cerebellum layer. Generalization happens at ATL: it clusters specific cerebellum models into concepts like "fast-elbow-flex" that span multiple buckets. Clean separation:
- **Cerebellum:** specific, deterministic, fast
- **ATL:** general, symbolic, slow

When Cerebellum has no model for the current bucket, it falls back to ATL concept prediction *before* calling the LLM.

### Thread Safety

Cerebellum is accessed from both agent loop (read during predict) and training path (write during observe). Use **per-key locks** (read-heavy pattern; per-key granularity avoids global contention). Document this in the class docstring.

### Success Criteria

1. **Reproducible LLM-skip rate:** over a fixed replay of 20 action patterns × 5 reps = 100 actions, LLM calls drop from 100 to ≤40.
2. **Prediction accuracy:** MAE on held-out actions < 20% of full-scale sensor range.
3. **Persistence:** Cerebellum state serializes to `data/embodiment/cerebellum.json` and reloads correctly after restart.

### Why Biologically

Biological cerebellum does exactly this — forward models predict sensory consequences of motor commands; climbing-fiber complex spikes carry prediction error; massive microcircuit specialization. This is a crude but functional version for synthetic sensors.

---

## Phase 2 — Structured Composable Failures (~150 LOC)

**Goal:** embodiment failures persist and learn exactly like tool failures, using existing bridge.

### Fixed Vocabulary + Composition

Base modes (6): `overextension`, `overheating`, `strain`, `fatigue`, `impact`, `exhaustion`.

Compositions allow specific failures without taxonomy explosion:

```yaml
failure_modes:
  - name: tennis_elbow
    composes: [strain, fatigue]
    component: left_elbow
    trigger:
      all:
        - {field: strain, op: ">", value: 0.6}
        - {field: fatigue, op: ">", value: 0.5}
    pain_intensity: 0.5
    persistent: true
    recovery_condition: {field: fatigue, op: "<", value: 0.2}
```

### Persistence Pattern (Reuses ToolPainBridge)

Existing `ToolPainBridge` already persists tool failures through NAc/Hippocampus. Widen its input to accept embodiment-sourced failures (~20 LOC change):

- Failure fires → `PainSignal` published with `source="embodiment"` metadata
- Hippocampus captures episodic memory with `failure_mode` tag
- NAc learns `(action, component_state) → failure` causal link
- EC's associative graph links failures to triggering actions
- AdaptivePlanner consults NAc before re-proposing same action on same body state

### Success Criteria

1. After overextension failure, NAc predicts pain for same action within 3 repetitions.
2. EC associative recall returns prior failure when agent plans similar movement.
3. Failure modes serialize to `data/embodiment/failures.json` and reload correctly.

### Explicit Non-Goal

No LLM-invented failure modes at runtime. Users compose from base vocabulary in YAML; no runtime taxonomy extension.

---

## LLM Fallback Cost Management (Spans Phases 0-2)

**Decision:** EnergySignal-based budgeting with Rescorla-Wagner-learned costs.

- Each LLM fallback emits an `EnergySignal` with tokens, latency, cost.
- **Per-context budgets** (not global): "novel action percept generation" and "failure narration" have separate learned costs.
- R-W updates expected cost per context bucket from actual observations.
- When cumulative energy on LLM fallbacks exceeds budget, cancel the current action and replan (consistent with existing resource-exhaustion handling).
- Budget is itself learned: initial guess → updated from observations → converges.

Reuses existing EnergySignal infrastructure. No new plumbing.

---

## Dynamic Sensor Rate-of-Change Bounds (Phase 1 polish)

**Decision:** bounds are *informative*, not *prescriptive*, and they adapt to body state.

- Each sensor has an `expected_rate_of_change` vital metric tracking the body's current state.
- Bounds widen with observed wear, injury, or malfunction — a damaged joint moves slower, its bound drops accordingly.
- Anomalies beyond bounds are still recorded (novelty signals or pain triggers) — they indicate "unexpected state change, investigate," not "reject."
- Cerebellum uses bounds to gate training weight: observations wildly outside bounds get reduced learning weight (avoid overfitting to sensor glitches).

---

## Simulation-Driven Development Process

Each phase validates through Maxim's own simulation harness:

1. Implement phase deliverable
2. Run `maxim --sim agent --goal "validate embodiment phase N" --persona researcher` with new capability
3. Measure success criteria from simulation reports (uses existing `refinement_baseline.yaml` infrastructure)
4. If criteria fail, diagnose and iterate before proceeding
5. If criteria pass, ship the phase and move to next

This plan dogfoods Maxim's testing harness for Maxim's own cognitive-architecture development.

---

## What's Out of Scope (Tracked Elsewhere)

| Deferred To | Reason |
|-------------|--------|
| Phase 3 (below) | HardwareBackend adapter — merged from former standalone plan |
| `agent_mesh.md` (Phase 1 bullet) | `EmbodimentCapability` in AgentIdentity |
| `future_plans.md` research directions | ATL Self-Extension, federated embodiments, uncertainty-as-pain, curriculum learning, bio-multimodal sensors |

---

## Scope Summary

| Phase | LOC | Sprints | Required? |
|-------|-----|---------|-----------|
| 0 (MVP + ATL grounding) | 400 | 2 | **Yes — gate** |
| 1 (Cerebellum) | 400 | 2 | **Yes** |
| 2 (Structured failures) | 150 | 1 | **Yes** |
| 3 (Hardware adapter) | 300 | 1 | After 0-2 validated |
| **Total** | **1,250** | **6** | |

Three phases, one hypothesis: **ATL-grounded + Cerebellum-cached LLM percepts produce consistent-enough signals for NAc to learn stable causal links.**

---

## No Blockers

Everything depends on existing infrastructure:
- `PerceptSource` protocol ✓
- `PainBus` + `ToolPainBridge` ✓
- `Hippocampus` episodic memory ✓
- `NAc` causal learning ✓
- `ATL` semantic memory ✓ (new `body_part` category added in Phase 0)
- `LLMRouter` ✓
- `EnergySignal` + R-W engine ✓
- `refinement_baseline.yaml` metric expectation infrastructure ✓

Phase 0 can start today.

---

## Phase 3: Hardware Adapter (~300 LOC)

> Formerly standalone plan `embodiment_hardware_adapter_plan.md`. Merged here as the natural follow-on once Phases 0-2 are validated.

**Goal:** Bridge the Embodiment layer to real hardware (`RobotController` / `RobotState`) via an adapter pattern. Enables the Cerebellum to learn forward models against real sensors.

**Explicitly not a refactor.** We wrap, not replace. Existing callers keep working. New callers use the unified API. ~10% cost for ~90% value, with clean rollback.

**Deliverables:**
- `src/maxim/embodiment/backends/hardware.py` (~150) — `HardwareBackend` wrapping `RobotController`
- `Embodiment.sync_from_robot_state()` — pulls current pose into component state
- `MovementTracker` optional `on_metrics` callback for embodiment observers
- `scenarios/embodiment/hardware_live_baseline.yaml`
- `tests/integration/test_embodiment_hardware.py`

**Design:**
```
RobotController / RobotState (existing, untouched)
        │ observed via adapter
        ↓
HardwareBackend (new adapter)
  - reads RobotState.current_pose
  - translates affordances → MotionTarget
  - publishes sensor values to Embodiment
        │ feeds
        ↓
Embodiment (from Phases 0-2)
  - Cerebellum learns from real sensors
  - New code reads via embodiment API
```

**Success criteria:**
1. Live Reachy Mini test: pose readings match within 1 tick
2. Cerebellum forward models predict head position with MAE < 5° after 50 motor commands
3. Zero regression: all existing hardware tests pass unchanged

**Risks:**
- Sync lag (mitigation: sync in same tick as state read)
- Double pain fire from PainDetector + Embodiment (mitigation: Embodiment defers to existing PainDetector for motor-derived pain)
- Affordance translation ambiguity (mitigation: deterministic mapping + tests per affordance type)
