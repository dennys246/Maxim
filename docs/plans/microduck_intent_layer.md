# The microduck — intent vocabulary, pluggable valence, and the throughput problem (1.3)

**Status:** DESIGN DRAFT, rev 1 (2026-08-31). Zero code. Written from **design constraints
supplied by the operator on 2026-08-31**, which close the "unknown SDK" and "unknown kinematics"
hedges that [roadmap_1_1_to_1_3.md](roadmap_1_1_to_1_3.md) §"The microduck" was forced to carry
when the duck was slotted to 1.3 on 2026-08-30 — but **not** the sensing one, which is still the
decisive unknown (§1.1).

**Target version:** 1.3 (design may start now; nothing here is on the 1.1.x or 1.2 critical
path). **The slot does not change** — see §7, where the constraints turn out to *harden* the
1.3 decision rather than argue against it.

**Owns (proposed):** the backend-independent intent vocabulary, the valence-source plug
contract, and the headless episode-loop harness shape for a many-trial robot experiment.

**Companion plans:** [sem_motor_binding.md](sem_motor_binding.md) (the motor path any robot
readout goes through) · [hybrid_substrate_reflex_runtime.md](hybrid_substrate_reflex_runtime.md)
(the 1.3 reflex tier — the duck relocates part of it into firmware, see §4) ·
[maxim_hivemind.md](maxim_hivemind.md) (gate 7, the bundle action namespace — §2.4 shows the
duck is that gate's portability-side second consumer) ·
[deferred/jepa_cross_modal_alignment.md](deferred/jepa_cross_modal_alignment.md) (the
paired-data hypothesis, unchanged and still a hypothesis) ·
[cross_modal_perception_fabric.md](cross_modal_perception_fabric.md) (1.3 sibling).

---

## 0. The constraints, as supplied

Recorded verbatim-in-substance and attributed, so later readers can separate *what we were
told* from *what we derived*. Everything in §§1–8 is derivation and is arguable; this section
is the input and is not.

1. **Intent vocabulary is its own layer.** The action set is defined once, independent of any
   backend. Two backends implement it: **sim** (`infer_policy.py`, CPU MuJoCo) now, and
   **`robotd`** (NDJSON JSON-RPC over wifi) when hardware arrives. The substrate binds only to
   the vocabulary. Signatures follow the existing structural-key convention —
   `skill:locomotion:velocity`, `skill:recovery:standup`, `skill:manipulation:groundpick`,
   `action:kick:ball`.
2. **Arbitration granularity is the policy, not the joint.** All Microduck ONNX policies share
   a **61-dim observation contract** (48 proprioception + twist / head-pose / body-pose
   commands), so the runtime hot-swaps any of them at any moment. No handoff logic between
   skills is needed — emit one intent, the 50 Hz loop handles the rest.
3. **Timing split.** Policies run onboard at **50 Hz (20 ms budget)**. The substrate arbitrates
   off-board at **1–5 Hz**. **Nothing in the substrate goes inside the control loop.**
4. **Reward sources are pluggable, not hard-coded.** Two distinct sources feed one valence
   mapper. *Sim:* MuJoCo ground truth — exact pose, contacts, fall state. *Hardware:*
   `robotctl monitor` stream — fall verdict from projected gravity, per-joint commanded vs.
   measured, servo temperature, battery. **Plus commanded-vs-applied divergence with a named
   reason**, which is a *labeled unexecutable-intent signal* and **does not exist in sim**.
5. **Aversive channels are rich; positive valence is thin** and must be synthesized from task
   completion.
6. **Throughput.** Rescorla-Wagner needs repeated trials. The keyboard demo runs at real time —
   a **headless episode-loop variant with resets** must exist before anything is collected.

---

## 1. What the constraints resolve, and the one they do not

The 2026-08-30 scoping recorded three unknowns. Two are now closed:

| 2026-08-30 hedge | Status after the constraints |
|---|---|
| "its SDK … unknown here" | **Closed.** Two backends, both specified: `infer_policy.py` over CPU MuJoCo, and `robotd` speaking NDJSON JSON-RPC over wifi. |
| "its kinematics … unknown here" | **Closed enough to design against.** We never need joint-level kinematics: arbitration is at policy granularity, and all policies share one 61-dim observation contract. This is a *smaller* integration surface than the Reachy's, not a larger one. |
| "The decisive unknown is **directional audio**" | **NOT closed — and the silence is informative.** §1.1. |

### 1.1 Audio is still the decisive unknown, and the constraint list argues against it

The constraints enumerate the duck's sensing twice — once as the observation contract (48
proprioception + command channels) and once as the reward sources (pose, contacts, fall state,
projected gravity, per-joint commanded-vs-measured, servo temperature, battery). **No
microphone appears in either list.** That is not proof of absence — the lists are scoped to
control and reward, and a mic array would legitimately appear in neither — but it is the second
independent enumeration of the duck's sensors that does not mention audio, and the design
should not assume one arrives.

The consequence, unchanged from the 2026-08-30 record and now more firmly grounded: **every
EARNED behavioural result this project has is sound-orienting** (Exp 45, 52, 53b, 54). A duck
without a mic array inherits *no analogue of the only validated behaviour*. It is therefore not
a port of the orient result onto a second body; it is a **new behaviour class on a new
modality** — proprioceptive/locomotor rather than auditory — and it needs its own
pre-registered experiment to earn anything.

**Decision owed (operator):** does the duck have a microphone array, and if so does it give
*direction* or only presence? The answer selects between two very different first experiments,
and it is the single highest-value fact to establish before any code is written. See §8.

---

## 2. The intent vocabulary — the premise needs correcting first, then it mostly rides

Per CLAUDE.md's design-time scope pressure, a proposed layer must first answer *"does this need
to be its own mechanism, or can it ride existing infrastructure?"* Answering it here required
mapping what the codebase actually has, and that map **contradicts one premise in constraint 1**.
Correcting it changes the design, so it comes first.

### 2.1 Correction: there is no "existing structural-key convention" of that shape

Constraint 1 says the proposed signatures — `skill:locomotion:velocity`, `skill:recovery:standup`
— "follow the existing structural-key convention." **They do not, because that convention does
not exist.** Verified against `main`:

- **Colon-delimited, body-prefixed action keys have zero occurrences** in `src/`, `docs/`,
  `scenarios/` or `tests/`. Nothing is keyed `body:<name>:<verb>`.
- The **real** tool signature is built in
  `embodiment/tool_bridge.py::generate_tools_for_entity` as `f"{ent.name}_{aff_name}"` — flat,
  underscore-joined, entity-prefixed, and **the modulator name is dropped**. Real strings in the
  tree: `reachy_mini_turn_left_big`, `infant_operant_turn_left`.
- The only thing actually called a structural key is
  `similarity/signature.py`'s `structural_str = f"{tool_name}:{outcome_type}"` — and its
  `tool_name` half is that entity-prefixed flat string.
- `affordance_namespace` and a manifest `body_ref` **exist only in plan docs** (roadmap gate 7,
  the Oasis case study). `hivemind/bundle.py`'s manifest carries `_format_version`,
  `contributor_id`, `domain`, `signature`, `contents`, `schema_version` — no body, no namespace.

Three consequences follow, and they are the substance of this section:

1. Adopting `skill:locomotion:velocity` is **introducing** a convention, not following one. That
   is defensible — see §2.3 — but it must be costed as new, and the plan must say what happens
   to the existing flat names.
2. Because NAc keys on that flat string (`decisions/nac.py::record_outcome` takes the tool name
   as the event signature), **learned substrate today is keyed to backend-specific body names**.
   That is the same finding the roadmap calls "a bias-key identity namespace — the undocumented
   design gap," reached independently from the portability side.
3. Consumers **reconstruct** these names by string concatenation. `tools/reachy.py` builds
   `f"{_ent_name}_turn_{'left' if az < 0 else 'right'}_big"` at the call site, and substrate
   action-restriction is done by **substring match** on `MAXIM_SUBSTRATE_TOOL_WHITELIST`
   (`runtime/agent_loop.py`; used as `turn_left,turn_right` by `scripts/exp49/run_trials.py` and
   `scripts/benchmark_cradle_mother.py`) — the code's own surrounding comment calls the mechanism
   a "BAND-AID (tracked)". A vocabulary layer is the principled fix for a thing already known to
   be held together with string matching.

### 2.2 What already exists and must not be rebuilt

- **A real, complete backend-independent motion/lifecycle abstraction.**
  `hardware/controller.py::RobotController` is an ABC with **exactly 12 abstract methods**, and
  it is **contract-frozen at 12** by its own docstring ("a 13th breaks every third-party
  `maxim.robots` plugin"). Robot-specific joints ride `MotionTarget.extras: dict[str, float]` —
  no subclassing. Optional capabilities are probed with `getattr` (e.g. `get_doa_reader`).
- **Plugin discovery already exists.** `hardware/registry.py::RobotRegistry` discovers the
  `maxim.robots` entry-point group, so a duck backend can ship as its own package without a core
  edit. `~/.maxim/robots.yaml` names the type; the body rides the free-form `config: {body: ...}`
  dict (`hardware/config.py::resolve_body_ref`).
- **A per-body action-set declaration.** Body YAML at
  `src/maxim/_data/components/bodies/*.yaml` declares `modulators:` → `affordances:` with
  `params`, `description`, `self_effect`, `always_active`. This is the capability-driven
  principle in [docs/embodiment/README.md](../embodiment/README.md): declare what the hardware
  supports, cognition code unchanged.
- **A clean binding socket.** `embodiment/spec.py::attach_backends(entity, modulator_factory=…)`
  takes a factory of shape `(entity, mod_name, spec_modulator) -> backend | None`. The only
  production implementation is `hardware/reachy/motor_backend.py::make_reachy_orient_factory`,
  hard-gated by `if mod_name != "orient": return None`. **This socket is where a duck's skill
  backend plugs in, and it is already generic.**
- **A three-part action identity, already in the tree.** `embodiment/motor.py::MotorStep.sem_key`
  is the triple `(entity_path, modulator, affordance)` — the one place the modulator namespace
  survives past declaration.

### 2.3 The recommendation: ride `sem_key`, do not invent a parallel string format

`skill:locomotion:velocity` is a three-part key: namespace, group, verb. `MotorStep.sem_key` is
already a three-part key: entity, modulator, affordance. **These are the same structure**, and
the repo's own precedent (`similarity/signature.py` composing a structural string from parts)
shows the intended direction of travel.

So the proposal should be framed not as a new string format but as **promoting `sem_key` from a
motor-program-internal identity to the general action identity**, with two changes:

1. The first element becomes a **capability namespace** (`skill`, `action`) rather than an
   entity path, with the body carried as an attribute instead of a prefix.
2. Tool-name generation and NAc keying derive from the triple, instead of the triple being
   discarded at `tool_bridge.py` and reconstructed by concatenation downstream.

That is a real piece of work and it is **not** free — but it is work gate 7 already owes.

### 2.4 The duck and gate 7 are the same question from two sides

- Gate 7 asks *"can agent A's learned want be read by agent B on a different body?"* — a
  **sharing** question, blocking 1.2, landing in **1.1.3**.
- The duck asks *"can one vocabulary bind to two backends?"* — a **portability** question.

One capability-scoped namespace answers both; a body-scoped one answers neither.

> **Recommendation, and the most actionable thing in this document.** When gate 7 is designed in
> 1.1.3, design it as a **capability namespace with the body as an attribute**, not as a body
> namespace with a compatibility shim, and take `MotorStep.sem_key`'s triple as the starting
> shape. Cite this section as the second consumer that justifies the abstraction. If gate 7 ships
> body-scoped, the duck's vocabulary becomes a genuine new mechanism plus a translation table —
> the outcome to avoid. **None of this requires the duck to exist.**

### 2.5 The shape mismatch: the frozen ABC is pose-shaped, the duck is policy-shaped

Worth naming before someone tries the obvious thing. `RobotController.goto_target(MotionTarget)`
takes a **pose** (`head_roll`, `head_pitch`, `head_yaw`, `body_yaw`, plus `extras`). The duck does
not want a pose — it wants "run policy P, with twist command C," and the 61-dim observation
contract means the policy consumes commands, not joint targets.

Do **not** solve this by widening the ABC: it is frozen at 12 for a stated reason, and a 13th
method breaks every third-party plugin. The clean decomposition is:

| Seam | Duck's use |
|---|---|
| `RobotController` (12 methods) | Transport + lifecycle only — `connect`/`disconnect` over NDJSON JSON-RPC to `robotd`, `wake_up`/`goto_sleep`, and the stream getters. |
| `attach_backends` modulator factory | **The intent seam.** A `make_microduck_skill_factory` gated on the duck's modulator names, exactly as the Reachy factory is gated on `"orient"`. |
| `MotionTarget.extras` | Twist/head-pose/body-pose command channels, if a pose-shaped call is ever genuinely needed. Probably it is not. |

This also keeps the sim and hardware backends symmetric: both implement the same modulator
factory; only the transport underneath differs (`infer_policy.py` in-process vs. `robotd` over
wifi).

### 2.6 The second-consumer test applies per layer, not per robot

[porting_orient_loop.md](../embodiment/porting_orient_loop.md) §"When robot #2 arrives"
pre-commits to a three-part extraction, deferred "until robot #2 exists (second-consumer test)":
an `OrientRig` protocol (`read_azimuth()`, `turn(delta)`, `recenter()`, `sensor_frame()`); moving
the learning loop into `src/maxim/embodiment/orient_loop.py`; and wiring affordance dispatch
through the executor/tool_bridge.

It is tempting to call the duck robot #2 and fire all three. **That is wrong for the first two
and right for the third.** If the duck has no mic array (§1.1) it is not a consumer of the orient
rig at all, and extracting `OrientRig` against a non-consumer is precisely what the
second-consumer test exists to prevent — abstracting from one example while claiming two. The
duck *is* a second consumer of the `RobotController` factory, the body-YAML declaration pattern,
the `attach_backends` socket, the valence intake (§5), and item 3's executor/tool_bridge dispatch
(which is body-agnostic and should proceed on its own merits regardless).

Apply the test per layer, and record which layers the duck actually consumes before extracting
anything.

---

## 3. Arbitration at policy granularity — this one fits cleanly

Constraint 2 is the constraint that costs least, and it is worth saying so plainly because the
rest of this document is mostly caveats.

The substrate's action-selection surface is `decisions/nac.py::NAc.recommend_action`, which
chooses among **named, discrete tool signatures** given a state-bin string. It carries no
inter-action state and no notion of a trajectory. That is a poor fit for joint-level control and
an excellent fit for "pick one of N policies, hot-swappable at any tick" — which is exactly what
the shared 61-dim observation contract buys. Policy-granularity arbitration is the granularity
the learning core already speaks.

Two cautions:

1. **Do not build motor programs for skill chaining.** `embodiment/motor.py::MotorProgram` /
   `MotorStep` exist and sequence affordances. Constraint 2 says no handoff logic is needed
   between skills, so the first duck integration should declare skills as flat affordances and
   leave `MotorProgram` alone. If sequencing later turns out to be necessary, that is a finding
   worth recording, not a default to reach for.
2. **Tool names are not stable under scene composition.** `tool_bridge.py::_resolve_tool_name`
   resolves collisions by progressively prepending parent entity names, raising `ValueError` if
   it runs out. Two bodies in one scene — precisely the duck-and-a-Reachy configuration the JEPA
   paired-data hypothesis wants (§8, and
   [deferred/jepa_cross_modal_alignment.md](deferred/jepa_cross_modal_alignment.md)) — can
   silently rename the first-registered body's tools, and NAc keys on those names. **Check this
   before ever running two bodies against one substrate.**

---

## 4. The timing split, and what it does to the 1.3 reflex tier

Constraint 3 — 50 Hz onboard, 1–5 Hz off-board, nothing from the substrate inside the control
loop — is not merely an implementation note. It is the same architecture
[hybrid_substrate_reflex_runtime.md](hybrid_substrate_reflex_runtime.md) designs in software,
except the duck ships half of it in firmware.

That plan's thesis is that the learned orient policy runs as a Default Network `Behavior` on
DN's own ~30 Hz thread, arbitrated by `PriorityArbiter` *within* DN, while the agent loop
deliberates slowly alongside it. The duck splits the same two tiers across a **hardware**
boundary instead of a thread boundary:

| Tier | Reachy (as designed in the reflex plan) | Microduck |
|---|---|---|
| Fast, action-below-deliberation | DN thread, ~30 Hz, in-process | The ONNX policy, 50 Hz, **onboard, in firmware** |
| Slow, deliberative/selective | The agent loop | The substrate, 1–5 Hz, **off-board** |

Three consequences worth writing down before anyone implements:

1. **The duck does not need BL-1..BL-5 for balance or gait.** Those five DN additions exist
   because DN's action loop is built around visual detections and cannot be evaluated on an
   audio-only tick. The duck's balance/gait reflex never enters DN at all — it is below the
   integration seam entirely. This is a genuine simplification, and it is the strongest
   architectural argument in the duck's favour.
2. **The converse: DN cannot arbitrate at the joint level on the duck, ever.** Whatever
   arbitration the substrate performs happens *between intents*, at 1–5 Hz, and the firmware
   resolves everything below that. Any future design that wants per-joint influence on the duck
   is asking for something the timing split forbids. Say no early.
3. **The 20 ms budget is a hard invariant, not a target.** The rule "nothing in the substrate
   goes inside the control loop" should be recorded as an `[engineering]` invariant in the
   embodiment brief when the first duck code lands, with a regression guard, because it is
   exactly the kind of rule that erodes one convenient call at a time. Note the Reachy already
   has the analogous rule in a different shape (`ReachyMiniController.goto_target` as the single
   clamped dispatch point) and it needed a CI grep to hold.

---

## 5. Valence — the constraint that needs the most rework

Constraints 4 and 5 assume a valence mapper that accepts graded, multi-source signals. What
exists is narrower, in three specific ways. This section is the main engineering content of the
document.

### 5.1 The action-selection surface is sign-only, so "rich aversive channels" mostly collapse

`runtime/tool_dispatch.py::record_outcome` is the per-action credit router. Its cluster-reward
assignment is, verbatim:

```python
cluster_reward: float | None = 1.0 if drive_potential_diff > 0.0 else -1.0
```

`drive_potential_diff` is a **graded** float at the producer
(`embodiment/sem.py::drive_comfort_progress`) and only its **sign** survives to the surface that
selects actions. The in-tree rationale is sound — a graded 0.15 would lose the argmax to the flat
`±1` tool-success floor — but the consequence for the duck is direct:

> **Servo temperature, battery level, per-joint commanded-vs-measured magnitude, and depth of a
> fall are all graded signals, and at the action-selection surface they are worth exactly the
> same as each other: `-1`.**

The graded paths that do exist, and where they go:

| Path | Accepts | Reaches |
|---|---|---|
| `NAc.credit_operant_reward(reward: float)` | arbitrary float | **the motor-selection surface** — the rich channel |
| `NAc.update_cluster_reward(reward=…)` | arbitrary float | cluster bias (clamped ±1.0) |
| `NAc.record_percept_valence` | signed `[-1, 1]` | salience gating only, **and has no positive producer in-tree** |
| `CausalLink.update_prediction_rw` | the 4-member `Valence` enum, via `_VALENCE_TO_REWARD` = `{POSITIVE: 1.0, NEGATIVE: 0.0, NEUTRAL/UNKNOWN: 0.5}` | link value `[0,1]`, 0.5 = neutral prior |

Note the last row: the RW target lives on `[0,1]` with **0.5 as neutral**, not on `[-1,+1]`. A
"0.3 good" outcome has no lossless intake through the enum path at all.

**Design consequence.** The duck's valence mapper should target
`NAc.credit_operant_reward` with an episode-scoped float, not the per-action enum path. Route
graded body state there, and accept that anything routed through `record_outcome`'s cluster term
is a sign.

### 5.2 "Commanded vs. applied with a named reason" is a NEW channel, not a wiring job

This is the finding that most changes the work estimate, and it is good news disguised as bad.

**The Reachy already computes this signal and then throws it away.**
`hardware/reachy/controller.py` records `last_clamped_axes` after clamping, and its **only**
consumer in `src/` is `tools/reachy.py`, which renders it as English prose for the LLM. There is
no path from `last_clamped_axes` to the `PainBus`, to the NAc, or to any modality channel. A
refused or clamped command is **invisible to the substrate**.

It is worse than invisible in one configuration. `tools/reachy.py::focus_on_sound` returns
`ToolResult(success=True, …)` even when `clamped` is true or `reached` is false — deliberately,
and its payload is scrupulously honest (`faced_sound`, `clamped_to_head_limit`, an explanatory
`note`). But `tool_dispatch.py` computes `learn_success = success and not embodiment_failed`, so
the substrate books a **POSITIVE** outcome for a motion the payload says did not happen.

**Bound this honestly:** in the graduated orient experiments the credit comes through
`drive_relief` / `orient_relief`, and the tool-success floor is explicitly suppressed there
(`drive_relief_only`, `drive_credit_withheld`, `MAXIM_OPERANT_ONLY_CREDIT`). So this does not
retroactively touch Exp 45/52/53b/54. It bites where the tool-success floor is live — the
LLM-primary runtime driving a real robot. **It is still a defect and deserves a bugs-ledger row**
(see §8; the ledger is mid-renumber in PR #577, so file after that lands).

**The socket already exists.** `tool_dispatch.py` has a precedent for exactly this shape: an
action that mechanically succeeded but harmed the body is flipped to negative via
`ToolOutput.side_effects["embodiment_failures"]`, with the comment that without it "a
harmful-but-mechanically-successful affordance books a POSITIVE causal link." A clamped or
diverged motion is the same class of event and nothing puts it there.

> **Recommendation.** Wire `last_clamped_axes` (and any commanded-vs-achieved divergence) into
> `side_effects["embodiment_failures"]` with a named reason. This is **the microduck's
> divergence channel, prototyped on the Reachy, buildable today, with no duck.** It is the second
> most actionable item in this document after §2.4, and it turns constraint 4's
> hardware-only signal into something we can design against a year early.

Nearest existing analogues, both real and both stranded on the legacy path:
`proprioception/pain.py::PainDetector.set_movement_target` / `_check_movement_failure` is a
genuine commanded-vs-achieved comparator emitting graded `PainType.MOVEMENT_FAILURE`, and
`harm/joint_limit.py::JointLimitHarmPredictor` predicts unreachability before the move. Both are
reachable only through `bridges/pain_bridge.py::PainCircuitBridge` from the legacy Selfy runtime
— **not wired in the substrate-primary / `agent_loop` path**. Prefer reviving these over writing
new comparators.

### 5.3 Positive valence: constraint 5 is correct, and the repo's existing answer is the weak one

Confirmed by enumeration. Of ten `Valence.POSITIVE` producers in `src/`, **seven mean literally
"the function returned without raising."** There is no reward bus symmetric to the `PainBus`
(which has a required-kwarg builder, `proprioception/pain_bus.py::build_pain_bus`, and three
auto-subscribed learners); `ReactionKind` has a `"reward"` member whose only publisher is the
dormant cerebellum modulator and which nothing subscribes to; and `record_percept_valence`'s own
docstring admits positive conditioning has no producer.

The one rich positive channel, `NAc.credit_operant_reward`, **requires an external teacher** —
in-tree its producers are experiment harnesses (`scripts/orient_substrate/9_hunger_relief_orient.py`,
`simulation/cradle_mother.py`), never the runtime.

So constraint 5's "synthesize from task completion" is *already* what the codebase does, through
the path it independently found to be too thin to learn from — which is why three separate flags
exist to suppress it. **The duck must not inherit the tool-success floor.** Its positive valence
should come from an episode-scoped task-completion teacher calling `credit_operant_reward`, with
the floor suppressed (`MAXIM_OPERANT_ONLY_CREDIT`), exactly as the nursery experiments do. In
other words: **the episode loop of §6 is the duck's caregiver**, and that is a design commitment,
not an implementation detail.

### 5.4 Proprioception is not a perceptual modality here, and 48 dims would dilute

Three facts, each with a consequence:

1. **`proprioception/` is an aversion source only.** `MovementSample` is a fixed 6-field head
   pose (`yaw, pitch, x, y, z, roll`) hardcoded to Reachy geometry; its only downstream is
   `PainDetector` → `PainSignal` → `PainBus`. It never becomes a `Percept`, never reaches
   `SensorEncoder.encode_sensors`, never gets an EC cluster. **A 48-dim joint vector has nowhere
   to land today.**
2. **Exactly two modality channels ship.** `runtime/agent_loop.py`'s `_SUBSTRATE_CHANNELS` is
   `(interoception, audio)`. Adding a third is genuinely one tuple entry — the seam is
   declarative by design — **but it is a selection-dynamics recalibration, not a free
   extension.** The registry's own comment warns that `max_cluster_reward_bias` caps *per
   cluster*, so the summed term in `recommend_action` scales with channel count: 2 → 3 channels
   moves the range from ±2 to ±3 against a `causal_pos` term topping out near 1.0, and gate
   calibration (`min_confidence`) must be re-checked. **This is the same warning the Minecraft
   world-modality channel carries** ([minecraft_benchmark.md](minecraft_benchmark.md)); if both
   land, they compound, and the recalibration is one job, not two.
3. **48 raw dims is the dilution failure at 48×.** The seam exists because *one* azimuth scalar
   merged among a handful of drives collapsed left/right onto one EC node and put the orient sim
   at chance ([archive/exteroception_interoception_seam.md](archive/exteroception_interoception_seam.md)).
   The in-tree mitigation is `similarity/place_code.py::place_code` — population coding that
   **replaces** the scalar rather than augmenting it. Design a **reduced proprioceptive summary**
   (gait phase, CoM offset, per-limb load bands), not 48 raw joints, and declare `(lo, hi)`
   ranges for every dimension or signed values fold non-monotonically.

Two smaller traps worth writing down now:

- **Operant-credit tag selection.** `tool_dispatch.py` prefers `AUDIO_TAG` and otherwise takes
  `sorted(extero_tags)[0]`. On a mic-less duck there is no `AUDIO_TAG`, so a `"proprioception"`
  channel becomes the operant channel **by alphabetical fallback**. That is probably the right
  outcome, arrived at by accident — pin it deliberately rather than relying on the sort order.
- **Pain refractory is 0.5 s, keyed `(entity, failure_mode)`.** A duck emitting per-joint pain at
  gait frequency will have almost all of it silently dropped. Aggregate before publishing.

---

## 6. Throughput — the headless episode loop

Constraint 6 is right, and the repo has a well-worn template for it. Three concrete answers:

### 6.1 Copy the in-process family, and copy it from the newest one

Harnesses here come in three families: sub-sim spawners (`scripts/exp49/run_trials.py`,
`scripts/exp44/campaign.py`), **in-process scripted** (`scripts/orient_substrate/*.py`), and
analysis-only. A many-trial duck experiment wants the in-process family — no subprocess, no LLM,
no per-trial `maxim` spawn — and the best template is
**`scripts/orient_substrate/9_hunger_relief_orient.py`**: it is the newest, it exercises the
drive-relief credit path end to end, and it has the cleanest arms/seeds/bins structure.

The idioms to copy exactly:

- **Reset is construct-per-trial.** There is no `.reset()` on `NAc` or `Embodiment` anywhere. Each
  trial builds a fresh `NAc(NACConfig())`, a fresh encoder (new `EntorhinalCortex`), and a fresh
  `Embodiment`. Do not invent a reset method for the duck; MuJoCo's own reset covers the body,
  and the substrate is rebuilt.
- **Independent RNG streams per arm.** `9_hunger_relief_orient.py` uses
  `np.random.default_rng(seed)` with a `YOKED_SEED_OFFSET`, added after a dry run found shared
  seeds made the control arm "learn" by construction. A duck's control arms need the same
  treatment.
- **Apparatus-sanity telemetry that can void the run.** The harness records counters (`fed`,
  `credits`, `credit_rewards`, `relief_min/max`) and a frozen "mechanism sanity" gate fails the
  run if they are wrong. This is what catches a harness that ran beautifully and measured
  nothing.
- **Incremental append**, from the spawner family: `scripts/exp49/run_trials.py` appends each
  record inside the loop so a crash at trial 40 keeps trials 1–39. At the trial counts §6.3
  implies, this matters.

Mirror `scripts/orient_backbone/live_common.py`'s **`LiveRig` / `DryRig`** pair for the duck's
sim/hardware split — it is the same shape as constraint 1's two backends, already proven.

### 6.2 Provenance is mandatory and there is a one-line way to get it right

Any harness writing under `docs/experiments/data/` is gated. For the in-process family the call
is `scripts/_provenance.py::in_process_code_provenance(...)`, and any gated write needs
`::preflight_gated_record_or_exit(...)`. **The shortcut:**
`scripts/orient_backbone/live_common.py::JsonlLog` runs the preflight in its `__init__` and
auto-stamps `allow_dirty` — using it as the writer satisfies
`scripts/lint_harness_provenance.py` and the clean-tree rule in one move.

This is not optional bookkeeping: Exp 53/53b's originals ran from a dirty tree, which cost a
replication round to repair
([experiment-prereg-precedes-data](../lessons/experiment-prereg-precedes-data.md)). Build it in
from trial 1.

### 6.3 The trial budget, and why real time is disqualifying

Concrete arithmetic, so the throughput requirement is a number rather than an intuition. The RW
update is `ΔV = α(R − V)` with `base_learning_rate = 0.2` and default novelty 0.5, giving an
effective **α ≈ 0.14**. Reaching 90% of asymptote needs `0.86ⁿ ≤ 0.1`, i.e. **n ≈ 16 credited
trials per (state-bin, action) cell**. Multiply by cells, arms, and seeds and a modest duck
design — say 4 skills × 4 state bins × 4 arms × 6 seeds — lands in the **thousands of episodes**.
(Exp 45's direction learning moved 0.00 → 1.00 in ~10 trials, which is the right order and
confirms the arithmetic is not pessimistic.)

At real time, with resets, that is not a run — it is a season. Hence constraint 6. **Build the
headless MuJoCo episode loop before collecting anything**, and treat wall-clock per episode as a
gating measurement in its own right: if the sim cannot deliver order-1000 episodes in a working
day, the experiment design has to shrink before it is pre-registered, not after.

---

## 7. Why this stays in 1.3 — the constraints harden the slot, they do not soften it

It would be easy to read §1's two closed unknowns as an argument for pulling the duck earlier.
They are not, and the reasoning should be explicit so it is not re-litigated:

1. **The engineering got easier; the science did not.** Policy-granularity arbitration and a
   fixed 61-dim observation contract make the *integration* smaller than the Reachy's. But the
   thing that earns a release claim is a pre-registered behavioural result, and §1.1 leaves the
   duck with no analogue of the only behaviour this project has ever earned. A cheap port of an
   un-validated behaviour is not a shipping item.
2. **Positive valence must be synthesized (constraint 5), which is a design question with no
   current answer.** See §5. That is research, and research belongs far, per the roadmap's
   standing ordering principle (predictable deliverable near, may-fail research far).
3. **A release cannot be gated on hardware with an uncertain arrival date.** Unchanged from
   2026-08-30. The sim backend removes the *blocking* dependency — design and even the episode
   harness can proceed on MuJoCo before any duck exists — but it does not make a hardware
   result schedulable.
4. **1.1.3 owes the duck something first.** Gate 7 (§2) and the `Maxim.mini` break-out are both
   prerequisites that land on the 1.1.x ladder for their own reasons. Doing them well is the
   highest-value duck work available right now, and none of it requires the duck.

**What changes, and it is worth something:** the duck is no longer a hardware bet with unknown
sensing. It is a **specified sim target available today** plus a hardware backend later. The sim
backend can be built and the episode loop exercised without owning the robot, which converts the
duck from "revisit when it arrives" to "the sim half is startable whenever it earns priority
against the 1.3 fabric."

---

## 8. Open questions and decisions owed

### Owed by the operator (blocking design, not implementation)

1. **Does the duck have a microphone array, and does it give direction or only presence?**
   The single highest-value fact. It selects between "the duck replicates the orient line on a
   second body" and "the duck is a new proprioceptive/locomotor behaviour class needing its own
   pre-registration." §1.1. Everything downstream of the experiment design waits on this.
2. **What is the first duck claim?** No experiment number is allocated here deliberately (the
   highest in `docs/experiments/` is 54 as of 2026-08-31). Until §8.1 is answered there is no
   honest way to write the pre-registration, and the house rule is that gates are frozen before
   data.

### Owed by 1.1.3, where the duck is the second consumer but does not need to exist

3. **Does gate 7 ship as a capability namespace or a body namespace?** §2.4. The duck is the
   portability-side second consumer of the same abstraction D43's third barrier names from the
   sharing side. This is the most actionable item in the document and it is due months before
   any duck.
4. **Is `side_effects["embodiment_failures"]` the right socket for commanded-vs-applied
   divergence?** §5.2. If yes, wiring the Reachy's `last_clamped_axes` into it prototypes the
   duck's divergence channel today, on hardware we own.

### Owed by this plan before any code

5. **What is the duck's reduced proprioceptive summary?** Not 48 raw dims (§5.4). Candidates:
   gait phase, CoM offset, per-limb load bands. Needs `(lo, hi)` ranges per dimension and a
   decision on `place_code` population coding vs. scalars.
6. **What is the task-completion teacher?** §5.3 commits to episode-scoped
   `credit_operant_reward` with the tool-success floor suppressed, which makes the episode loop
   the caregiver — but *what counts as task completion* for a locomotion skill is unwritten.
7. **Which layers does the duck actually consume?** §2.6. Record the per-layer answer before any
   extraction, and specifically do **not** fire `porting_orient_loop.md`'s `OrientRig` extraction
   unless §8.1 comes back with a mic array.
8. **Does the sim backend get built ahead of hardware, and at what priority against the 1.3
   fabric?** The MuJoCo half has no hardware dependency (§7). It is startable; it is not
   scheduled.

### A defect this investigation surfaced, owed to the bugs ledger

9. **A clamped or unreached motion is credited as a success.** `tools/reachy.py::focus_on_sound`
   returns `ToolResult(success=True, …)` when `clamped` or `reached is False`, and
   `tool_dispatch.py` computes `learn_success = success and not embodiment_failed` — so the
   substrate books POSITIVE for a motion the tool's own payload says did not happen. Bounded: the
   graduated orient experiments suppress the tool-success floor, so no EARNED row is touched.
   Not filed as a row here because the ledger is mid-renumber in PR #577 (duplicate D43/D44
   ids); **file it after that lands**, with §5.2's `embodiment_failures` wiring as the fix.

### What this document deliberately does not commit to

- **Not** an experiment, a number, or a gate — §8.1 blocks all three.
- **Not** the JEPA paired-data pairing. That remains what the 2026-08-30 scoping called it: a
  hypothesis that *raises* JEPA's bar rather than lowering it, revisited when the duck's sensing
  is known. §3's tool-name-collision caution is a prerequisite for even trying it.
- **Not** a schedule change. The duck stays in 1.3 (§7).
