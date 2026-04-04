# Embodiment Layer Plan

> **Status:** Not started. MVP-first — validates core loop before expanding.
> **Depends on:** ATL semantic memory (exists), PainBus (exists), Hippocampus (exists), NAc (exists), PerceptSource protocol (exists)
> **Integrates with:** Simulation Agent, Agent Mesh (later phases)

---

## Core Insight

Maxim's ATL and proprioception subsystems were built for exactly this: **grounded math about a body**. The ATL holds canonical knowledge (joint ranges, torque curves, pain thresholds) as semantic concepts with IPS statistics and Angular Gyrus geometry. The Cerebellum (new) stores lightweight forward models that predict sensory consequences before an action commits. The LLM is consulted **only when no forward model exists yet** — it teaches the cerebellum, then fades out.

This fixes the original plan's biggest problem (LLM at 2Hz hallucinates inconsistent physics) by making the LLM a **teacher**, not a **runtime oracle**.

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
              PainBus fires if error-pain threshold
                         ↓
              Percepts → MemoryAgent → Hippocampus → NAc
```

### Layer Responsibilities

| Layer | Role | Backed By |
|-------|------|-----------|
| **EmbodimentSpec** | Declarative body description (components, sensors, affordances, failures) | YAML/JSON files |
| **ATL Body Concepts** | Canonical physical knowledge: joint ranges, torque, wear curves, pain calibration | Semantic concepts with IPS stats + AG geometry |
| **Cerebellum** | Learned forward models per component+action context | Lightweight Rescorla-Wagner-like predictors |
| **LLMBackend** | Generates percepts for novel situations; teaches Cerebellum | Existing LLMRouter |
| **HardwareBackend** | Reads real sensors, executes real actuators | Wraps `RobotController` |
| **RuleBackend** | Deterministic physics for simple cases (joint limits, linear damping) | Pure Python |
| **Embodiment** | Runtime: holds component state, dispatches to backends, tracks failures | New class |

### Why This Scopes Down

- **No LLM-at-2Hz.** Cerebellum runs at tick rate; LLM runs on novelty.
- **ATL already exists.** We don't build a new math engine; we add body-concept grounding to ATL.
- **No hardware refactor required for MVP.** Embodiment can coexist with existing `RobotController` code. Refactoring is its own separate decision.

---

## MVP (Phase 0) — Single File, Single Demo

**Goal:** prove the core loop produces stable, learnable (action → pain) pairs.

**Deliverables:**
- `src/maxim/embodiment/mvp.py` (~300 LOC)
  - `SensorSpec`, `AffordanceSpec`, `ComponentSpec`, `EmbodimentSpec` dataclasses
  - `Embodiment` runtime with structured state
  - `LLMBackend.generate_percepts(embodiment, action) → list[Percept]`
  - `EmbodimentPerceptSource(PerceptSource)` adapter
- `scenarios/embodiment/robot_arm_3dof.yaml` demo spec
- `tests/unit/test_embodiment_mvp.py`

**Success criteria (measurable, pre-commit):**
1. Run same action 10 times → observe pain intensity variance. **Target: σ < 0.2.** If higher, LLM is too inconsistent and MVP fails.
2. Run simulation for 100 actions → NAc builds ≥3 distinct causal links with confidence > 0.5.
3. Percept generation latency p95 < 2s per action.

**Hard constraints for MVP:**
- **Fixed failure vocabulary.** 6 base modes only: `overextension`, `overheating`, `strain`, `fatigue`, `impact`, `exhaustion`.
- **Structured trigger format, no eval.** `{field: "angle", op: ">", value: 175, pain: 0.8}`.
- **No homeostasis yet.** Vital metrics drift linearly only.
- **No user embodiment yet.** Just one AUT body.
- **LLM called once per action**, not per tick.

**If MVP fails** (σ > 0.2 or no stable NAc learning), we stop and revisit. The rest of the plan is contingent on this working.

---

## Phase 1 — ATL Body Concepts (~250 LOC)

**Goal:** ground embodiment percepts in ATL semantic memory instead of LLM improvisation.

**What this means:** each component type (joint, sensor, limb) gets an ATL concept with IPS statistics capturing typical ranges, AG geometry for spatial math, and semantic links ("shoulder IS-A joint", "elbow FLEXES_WITH shoulder"). When the LLM generates a percept, it queries ATL for canonical ranges; when the Cerebellum predicts, it uses ATL's geometry.

**Deliverables:**
- New concept category in ATL: `body_part`
- `BioContext` (existing skill facade) extended with `lookup_body_concept(name)`
- Grounding pipeline: first LLM call on each embodiment seeds ATL concepts from the spec
- Percept generation prompt now injects ATL-grounded ranges: "elbow typical angle range: 10-170deg"
- Demo: run identical action on grounded vs. ungrounded embodiment, measure consistency delta

**Success criteria:**
- Pain intensity σ drops from MVP baseline by ≥50% with ATL grounding enabled.
- ATL body concepts persist across sessions (reload from `data/atl/concepts.json`).

**Honest risk:** ATL may not have enough existing machinery for this. This phase exposes that. If ATL needs significant extensions, it becomes its own subtask.

---

## Phase 2 — Cerebellum (Forward Models) (~400 LOC)

**Goal:** replace LLM percept calls with learned deterministic predictors wherever possible.

**What this means:** the Cerebellum observes (action context → actual percept) pairs and learns small lightweight models per `(component, affordance, action_params_bucket)` key. When the same action is proposed again, the Cerebellum predicts the percept directly, without calling the LLM. The LLM is only invoked when no model exists for the current context.

**Design:**

```python
class Cerebellum:
    """Forward models for predicting sensory consequences of actions.
    
    Stores lightweight predictors per (component, affordance, param_bucket).
    Each predictor learns via prediction-error feedback (Rescorla-Wagner style,
    like NAc but for sensory prediction instead of reward).
    """
    def predict(self, component: str, affordance: str, params: dict) -> list[Percept] | None:
        """Return predicted percepts or None if no model exists."""
    
    def observe(self, predicted: list[Percept], actual: list[Percept]) -> None:
        """Update forward model from prediction error."""
    
    def has_model(self, component: str, affordance: str, params: dict) -> bool
```

**Model structure (per key):**
- Expected sensor values (mean, variance) per sensor
- Expected failure probabilities per failure mode
- Confidence (grows with observations)

**Prediction policy:**
- Confidence < 0.3 → use LLM, observe, train Cerebellum
- Confidence ≥ 0.3 → use Cerebellum prediction
- High-variance models → emit pain signal from *uncertainty itself* (matches biology: unfamiliar motion feels risky)

**Success criteria:**
- After 50 training actions, ≥60% of subsequent actions skip the LLM.
- LLM-call rate drops by >10x over a 200-action session.
- Prediction MAE on held-out actions < 20% of full-scale sensor range.

**Why this matters biologically:** this is exactly what the cerebellum does — it lets the agent *simulate actions before doing them* (used by motor planning), and its learning signal is prediction error (climbing-fiber complex spikes). We're implementing a crude but functional version of this for synthetic sensors.

---

## Phase 3 — Structured Failures + NAc/Hippocampus Integration (~200 LOC)

**Goal:** embodiment failures persist and learn exactly like tool failures.

**Design:**

**Fixed failure vocabulary** (extensible by composition, not by LLM invention):
```
Base modes (6): overextension, overheating, strain, fatigue, impact, exhaustion
```

**Compositions** allow specific failures without taxonomy explosion:
```yaml
failure_modes:
  - name: tennis_elbow
    composes: [strain, fatigue]
    component: left_elbow
    trigger: {all: [
      {field: strain, op: ">", value: 0.6},
      {field: fatigue, op: ">", value: 0.5}
    ]}
    pain_intensity: 0.5
    persistent: true
    recovery_condition: {field: fatigue, op: "<", value: 0.2}
```

**Persistence pattern** (mirrors existing tool-failure flow):
- Failure fires → PainSignal published
- Hippocampus captures episodic memory with `failure_mode` tag
- NAc learns `(action, component_state) → failure` causal link
- EC's associative graph links failures to triggering actions
- AdaptivePlanner consults NAc before re-proposing same action on same body state

**Success criteria:**
- After an overextension failure, NAc predicts pain for same action within 3 repetitions.
- EC associative recall returns prior failure when agent plans similar movement.
- Failure modes serialize to `data/embodiment/failures.json` and reload correctly.

**Explicit non-goal:** no LLM-invented failure modes. If users want custom failures, they edit YAML; no runtime taxonomy extension.

---

## Phase 4 — User Embodiment (Minimal) (~150 LOC)

**Goal:** theory-of-mind surface via a sensed-only user body.

**Minimal user spec:**
```yaml
name: user_default
components:
  - name: voice
    sensors: [speech]
    affordances: [speak]
  - name: attention
    vital_metrics:
      engagement: {initial: 0.7, drift_rate: -0.001, recovery_rate: 0.005}
      patience: {initial: 0.8, drift_rate: -0.0005}
    failure_modes:
      - {name: frustrated, composes: [fatigue], trigger: {field: patience, op: "<", value: 0.3}, persistent: true}
```

**Update rules (deterministic, not LLM-driven):**
- User sends short message → patience -= 0.05
- User sends "nvm"/"stop"/"never mind" → patience -= 0.3
- User thanks Maxim → engagement += 0.1, patience += 0.1
- Time passing without interaction → engagement drifts down

**What Maxim gets:** percepts like `{source: "user_state", content: "user frustration rising"}` fed into its normal context. Over time, NAc learns "long-winded responses → user patience drops → frustration." This is theory-of-mind via the existing causal-learning machinery.

**Custom user specs** (future, not MVP): orchestrator can override the default spec (`--user-context "driving"` → richer spec with divided attention).

**Scope check:** no conversational analysis, no emotion inference, no LLM-based user modeling. Deterministic heuristics only.

---

## Phase 5 — Architectural Integration (Opt-in, Not Refactor) (~300 LOC)

**Goal:** embodiment becomes first-class, coexisting with existing hardware code.

**Explicitly NOT a refactor.** We don't rewrite `RobotController` or touch `RobotState.current_pose`. Instead:

- `HardwareBackend` wraps existing `RobotController` as an adapter
- `Embodiment.sync_from_robot_state(state)` pulls current pose into component state
- `MovementTracker` optionally publishes to embodiment observers (new subscription, old API intact)
- New agents can use embodiment-first; old code keeps working unchanged

**Why not refactor:** the refactor is a 3+ week project with high breakage risk. This phase delivers 90% of the value (unified representation) with 10% of the cost.

**Success criteria:**
- Run live Reachy Mini with embodiment enabled; verify pose readings match `RobotState`.
- Cerebellum learns motor-command → head-position mapping on real hardware.
- No regressions in existing hardware tests.

**Full refactor** becomes its own plan, executed only if the adapter approach shows pain points.

---

## Phase 6 — Mesh Integration (Minimal) (~200 LOC)

**Goal:** embodiments show up in mesh capability broadcasts.

**Trimmed scope:** advertise capabilities only. No federation, no delegation, no distributed construction yet.

- `RuntimeCapabilities` adds `embodiment_summary: dict` (name, modalities, affordances, hardware_backed bool)
- Peers can query "who has a body that can grasp?"
- That's it.

**Deferred to future plans** (noted as speculative, not scheduled):
- Cross-agent affordance invocation (sovereign delegation)
- Federated embodiments (arm from A, cameras from B)
- NAc transfer gated by spec similarity
- Shared user embodiment across co-present agents

These are interesting but not validated by current needs.

---

## What I Explicitly Removed from the Original Plan

- **8 phases → 7 phases** (Phases 0-6, where 0 is the MVP gate)
- **~2450 LOC → ~1800 LOC** (and Phases 3-6 are optional expansions)
- **Architectural refactor (Phase 5 original)** → downgraded to adapter-based integration
- **Decompositional failure construction** → fixed base vocabulary + composition
- **Bio-inspired multi-modal sensors (olfaction, taste, vestibular)** → not needed for MVP, add if demand
- **Curriculum learning** → future plan
- **Federated embodiments** → future plan
- **Distributed mesh construction** → future plan
- **Unsafe expression-language triggers** → structured dicts, no eval
- **LLM-at-2Hz percept generation** → Cerebellum predicts, LLM teaches

---

## Honest Open Questions (Unresolved)

These are things I still don't have good answers for:

1. **Does ATL have enough machinery for body concepts?** Phase 1 might discover ATL needs extension. If it's significant, that becomes blocking work on ATL itself.

2. **Will the Cerebellum's action-param-bucket key generalize?** If angular motion needs fine-grained buckets (every 5°), model count explodes. Coarse buckets lose predictive power. This is an empirical tuning problem.

3. **What's the LLM fallback cost ceiling?** If Cerebellum never gets confident enough, we're paying LLM costs forever. Need a monitoring dashboard + threshold alerts.

4. **How do we validate LLM-generated percepts are internally consistent?** A sensor reading at t+1 that contradicts t (joint angle "teleports") breaks the Cerebellum's learning. Need a sanity check layer: bound changes by max rate-of-change per sensor.

5. **Does imagined pain really produce the same NAc learning as real pain?** MVP success criteria checks this loosely (σ < 0.2, ≥3 causal links). A rigorous answer requires comparing with hardware-backed pain on same action sequence.

---

## Simulation-Driven Development Process

This plan is built to validate through Maxim's own simulation harness:

**Development loop:**
1. Implement phase deliverable
2. Run `maxim --sim agent --goal "explore embodiment" --persona researcher` with the new capability
3. Measure success criteria from simulation reports
4. If criteria fail, diagnose and iterate before proceeding
5. If criteria pass, ship the phase and move to next

This means each phase produces measurable simulation evidence before we commit to the next. It also means the simulation framework itself is a development dogfood — we use our own testing agent to validate our own subsystems.

---

## Scope Summary

| Phase | LOC | Time Estimate (rough) | Required? |
|-------|-----|----------------------|-----------|
| 0 (MVP) | 300 | 1 sprint | **Yes — gate** |
| 1 (ATL grounding) | 250 | 1 sprint | **Yes** |
| 2 (Cerebellum) | 400 | 2 sprints | **Yes** |
| 3 (Structured failures) | 200 | 1 sprint | **Yes** |
| 4 (User embodiment) | 150 | <1 sprint | Recommended |
| 5 (Hardware adapter) | 300 | 1 sprint | Recommended |
| 6 (Mesh capability) | 200 | <1 sprint | Optional |
| **Core (0-3):** | **1150** | **~5 sprints** | Validates core value |
| **Extended (0-6):** | **1800** | **~7 sprints** | Full vision |

MVP alone proves the concept. Phases 1-3 deliver the complete cognitive loop. Phases 4-6 are polish.

---

## No Blockers

Everything depends on existing infrastructure:
- `PerceptSource` protocol ✓
- `PainBus` + routing ✓
- `Hippocampus` episodic memory ✓
- `NAc` causal learning ✓
- `ATL` semantic memory ✓ (extensions needed in Phase 1)
- `LLMRouter` ✓
- `EnergySignal` tracking ✓

Phase 0 can start today.
