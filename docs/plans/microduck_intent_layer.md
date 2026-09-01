# The microduck — intent vocabulary, pluggable valence, and the throughput problem (1.3)

**Status:** DESIGN DRAFT, **rev 2** (2026-08-31), after a two-lens pre-merge review round
([reviews/microduck_intent_layer_two_lens_review.md](reviews/microduck_intent_layer_two_lens_review.md)).
Zero code. Written from **design constraints supplied by the operator on 2026-08-31**, which close
the "unknown SDK" and "unknown kinematics" hedges that
[roadmap_1_1_to_1_3.md](roadmap_1_1_to_1_3.md) §"The microduck" was forced to carry when the duck
was slotted to 1.3 on 2026-08-30 — but **not** the sensing one, which is still the decisive
unknown (§1.1).

> **What changed in rev 2.** The Architecture lens returned two BLOCKING findings and both were
> confirmed against `main`; rev 1's two headline recommendations are **withdrawn as
> recommendations and demoted to options**.
> **(1)** Rev 1 said gate 7 "should" be a capability namespace. But
> [oasis_case_study_taught_orient.md](oasis_case_study_taught_orient.md) §1 had already
> front-gated exactly that choice and picked the other option for a stated reason, 1.1.3's ship
> gate (D44) does not need it, and `hivemind/bundle.py::register_bundle_migration` makes the
> choice reversible — so the urgency was false and the roadmap fold was creating a live
> contradiction between two active plans. Now §2.4 asks the design pass to *cost* it. **(2)** Rev
> 1 said "promote `MotorStep.sem_key`" and called it work gate 7 already owed. `sem_key` has
> exactly **two** references in the tree, so that meant *building* an identity, not promoting one
> — and §2.3 now carries the blast radius (20 persisted `53_agents` agent files, hivemind's
> signature scrub, substring-matching harnesses behind EARNED rows).
> Also revised: §5.2 no longer routes the divergence signal into the *harm* channel (a clamp is a
> refusal, not harm, and the rev 1 fix would have inverted the bug it fixed); the header's "Owns"
> line no longer claims three mechanisms the body concludes are ridden; a front-gate table and a
> schedule trigger were added; and the defect in §8 is now filed rather than deferred.

**Target version:** 1.3 (design may start now; nothing here is on the 1.1.x or 1.2 critical
path). **The slot does not change** — see §7, where the constraints turn out to *harden* the
1.3 decision rather than argue against it.

**Schedule trigger:** §8 item 1 answered (does the duck have a mic array?) → write the
pre-registration. The MuJoCo sim backend is unblocked independently of hardware but is **not
scheduled**; it competes for priority against the 1.3 fabric. No trigger fires from this document
alone.

**Owns (genuinely new, after the front-gate below):** the duck's **reduced proprioceptive
summary** and the state-bin design that goes with it (§5.4, §8 item 5). That is all. Rev 1's
header claimed ownership of the intent vocabulary, the valence-source plug contract and the
harness shape; the body of this document concludes all three **ride existing seams**, so claiming
them was a contradiction the review round caught.

## Front-gate scope pressure

CLAUDE.md requires every proposed mechanism to answer *"does this need to be its own mechanism,
or can it ride existing infrastructure?"* before a plan is drafted. Per constraint:

| Proposed | Verdict | Rides |
|---|---|---|
| Intent vocabulary as its own layer (c1) | **Rides** | Body-YAML `modulators:`/`affordances:` declaration + `embodiment/spec.py::attach_backends`'s `modulator_factory` socket (§2.2, §2.5). The open question is the *namespace*, which is gate 7's, not this doc's (§2.4). |
| Two backends, sim + `robotd` (c1) | **Rides** | `hardware/controller.py::RobotController` + `RobotRegistry`'s `maxim.robots` entry-point group — a duck backend ships as its own package with no core edit (§2.2). |
| Policy-granularity arbitration (c2) | **Rides** | `decisions/nac.py::NAc.recommend_action` already selects among named discrete signatures (§3). |
| Timing split (c3) | **Rides** — it is a constraint, not a mechanism | Enforced as an invariant when duck code lands, not built (§4). |
| Pluggable valence sources (c4) | **Rides, with one revival** | `NAc.credit_operant_reward` for graded reward; the divergence channel revives `proprioception/pain.py::PainDetector` on the `agent_loop` path rather than adding a comparator (§5.2). |
| Positive valence (c5) | **Rides** | Episode-scoped `credit_operant_reward` with the tool-success floor suppressed — the nursery pattern (§5.3). |
| Headless episode loop (c6) | **Rides** | Copy `scripts/orient_substrate/9_hunger_relief_orient.py`; mirror `live_common.py`'s `LiveRig`/`DryRig` (§6.1). |
| A proprioceptive modality channel | **NEW** | A `ModalityChannel` entry **plus a reader pair** (and a `_EXTEROCEPTIVE_ROOT_SENSORS` edit if it rides the existing extero readers), and it is a selection-dynamics recalibration — the reduced summary behind it has no existing design (§5.4). |

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
   `action:kick:ball`. *[operator premise; the "existing structural-key convention" clause is
   **corrected in §2.1** — no such convention exists in the tree. The verbatim record is kept
   here; do not cite this line as evidence that one does.]*
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
  `scenarios/` or `tests/`. Nothing is keyed `body:<name>:<verb>`. *(Precision, added in rev 2:
  the claim is about the **action namespace**. A colon-delimited `skill:`-prefixed key does exist
  elsewhere — `memory/concept_extractor.py` builds `f"skill:{skill_name}"`, consumed by
  `memory/concept_context.py` — so the `skill:` prefix is not unprecedented in the tree, it is
  simply not an action-signature convention. That is two-thirds of the proposed shape living in
  the concept graph, which is worth knowing before reusing the prefix for a different purpose.)*
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
  it is **contract-frozen at 12** — the statement lives in
  `hardware/controller.py::RobotController.get_doa_reader`'s docstring, not the class docstring:
  "a 13th breaks every third-party `maxim.robots` plugin". Robot-specific joints ride `MotionTarget.extras: dict[str, float]` —
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
  production implementation is `hardware/reachy/motor_backend.py::make_reachy_orient_factory`
  (a second factory of the same shape,
  `embodiment/backends/cerebellum_modulator.py::cerebellum_modulator_factory`, exists but is
  dormant with no callers outside its own docstring),
  hard-gated by `if mod_name != "orient": return None`. **This socket is where a duck's skill
  backend plugs in, and it is already generic.**
- **A three-part action identity, already in the tree.** `embodiment/motor.py::MotorStep.sem_key`
  is the triple `(entity_path, modulator, affordance)` — the one place the modulator namespace
  survives past declaration.

### 2.3 A capability-scoped action identity — the option, and its real blast radius

> **Revised after the two-lens review round (2026-08-31).** Rev 1 recommended "promote
> `MotorStep.sem_key` from a motor-program-internal identity to the general action identity" and
> called it "work gate 7 already owes." The Architecture lens refuted both halves and it was
> right. `sem_key` has **exactly two references in the entire tree**, both inside
> `embodiment/motor.py` (the property and `MotorProgram.structure_key`) — it is not a load-bearing
> identity today, so "promoting" it means **building a new one**. And gate 7 owes *one of two*
> options, of which this is the other; justifying it by asserting the thing under debate was
> circular. What follows is the corrected version: this is an **option to cost**, not a
> recommendation.

`skill:locomotion:velocity` is a three-part key: namespace, group, verb. `MotorStep.sem_key` is
a three-part key of the same shape: `(entity_path, modulator, affordance)`. If a capability-scoped
identity is adopted, that triple is the right starting shape — but the work is building it, and
the correct chokepoint is not `similarity/signature.py` (rev 1's citation) but
**`runtime/tool_dispatch.py::build_tool_signature`**, whose own docstring declares it "the single
source of truth for tool→NAc event signature format. All code that records or queries tool
signatures MUST use this function." Any identity change lands there first.

**The blast radius, which rev 1 did not cost.** The flat string is not merely generated — it is
*persisted, parsed, and matched on*:

- **Persisted agent state embeds it.** `docs/experiments/data/53_agents/…/aut_nac.json` carries
  `links` keyed `tool:infant_operant_turn_left`, an `outcome_index` built from those keys, and
  `cluster_reward_bias` / `cluster_reward_source` keyed on unit-separator triples
  `sim_aut\x1f<uuid>\x1ftool:infant_operant_turn_left`. There are **20 such files**, and
  [porting_orient_loop.md](../embodiment/porting_orient_loop.md) calls that directory the shipped
  example bundle-in-waiting. Changing the action identity migrates or invalidates the repo's only
  shipped evidence bundle.
- **Hivemind parses the string.** `hivemind/bundle.py::_scrub_event_signature` special-cases the
  `tool:use:<free text>` prefix and re-keys `event_outcome_welford`; its composition scrub is
  documented as keeping identifier-shaped signatures so the learned `tool:infant_operant_turn_left`
  keys ship intact.
- **Harnesses behind EARNED rows match it by substring.** `MAXIM_SUBSTRATE_TOOL_WHITELIST` is an
  `any(term in t …)` filter, used as `turn_left,turn_right` by `scripts/exp49/run_trials.py` and
  `scripts/benchmark_cradle_mother.py`. A triple-derived name silently changes which tools those
  harnesses see.
- **The cross-robot porting contract pins it.** `porting_orient_loop.md` states the policy is
  keyed on azimuth bins and **YAML action names**, which is what lets a different robot consume
  the same substrate.

So the honest statement is: a capability-scoped identity is a **migration of persisted learned
state**, not a rename. That does not make it wrong — it makes it something that must be costed
before it is chosen, which is §2.4's whole point.

### 2.4 The duck is gate 7's portability-side second consumer — an input, not a decision

- Gate 7 asks *"can agent A's learned want be read by agent B on a different body?"* — a
  **sharing** question, blocking 1.2, landing in **1.1.3**.
- The duck asks *"can one vocabulary bind to two backends?"* — a **portability** question.

These are the same question, and that is worth recording. What rev 1 got wrong was concluding
which way to answer it.

> **Revised after the review round.** Rev 1 said gate 7 "should" ship as a capability namespace
> and called the body-scoped option "the outcome to avoid." That overreached in three ways the
> Architecture lens caught, all confirmed:
>
> 1. **The question was already decided, the other way, with a stated reason.**
>    [oasis_case_study_taught_orient.md](oasis_case_study_taught_orient.md) §1 front-gated exactly
>    this, offered both options — (a) typed bundles `manifest.body_ref` +
>    `manifest.affordance_namespace`, (b) body-agnostic keys on the SEM modulator/affordance — and
>    chose (a), because "**(b) is the better long-term key but changes what every existing NAc file
>    means; (a) is the honest first step and is what Exp 53b actually did.**" §2.3's blast radius
>    is that sentence, restated from the portability side. Gate 7 records the decision as made in
>    the case study's design pass.
> 2. **It is not on 1.1.3's critical path.** 1.1.3 ships on **D44** — a behavioural delta across a
>    merge between two genuinely independent agents. D43 names the two axes that actually miss,
>    `agent_id` and `cluster_id`, and calls `tool_signature` the *third* barrier. Two independent
>    agents **on the same body** produce identical tool signatures, so D44 is clearable by option
>    (a) plus D43's two halves. Adding a cross-body identity redesign expands a release whose gate
>    does not need it.
> 3. **The choice is reversible, so the urgency was false.** `hivemind/bundle.py` already ships
>    `register_bundle_migration` / `migrate_bundle_envelope` as reserved hooks, documented as
>    "bumping `BUNDLE_SCHEMA_VERSION` to 2 + registering a v1 migration is the only change 1.1
>    needs to make for older bundles to load." Shipping (a) now and migrating to (b) later is a
>    **supported path**, not a shim to fear.

**What this document actually asks of gate 7's design pass**, which is all it is entitled to ask:

> Cost the capability-scoped option (b) against typed bundles (a) explicitly, and state the
> migration cost of the 20 persisted `53_agents` files when you do. The microduck is the
> portability-side second consumer, and it is the reason (b) has a constituency beyond sharing —
> but the case study's (a)-first reasoning stands unless that costing overturns it, and nothing
> here overturns it. **None of this requires the duck to exist, and none of it changes 1.1.3's
> ship gate.**

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
   [deferred/jepa_cross_modal_alignment.md](deferred/jepa_cross_modal_alignment.md)) — silently
   renames one body's tools, and NAc keys on those names. **Direction matters and rev 1 had it
   backwards** (caught by the Executor lens): the function starts at `candidate = base_name` and
   only prepends `while candidate in existing_names`, so the **first** registrant keeps the plain
   name and the **later** one is renamed. The hazard is therefore that a body which learned its
   keys in a solo run finds them changed when it registers *second* in a two-body scene — and
   registration order is not something an experiment currently pins. **Pin it before ever running
   two bodies against one substrate.**

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

1. **The duck does not need BL-1..BL-5 for balance or gait.** Most of those DN additions exist
   because DN's action loop is built around visual detections (BL-1 is a body-rotation/head-matrix
   defect rather than a detection one) and cannot be evaluated on an
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

`runtime/tool_dispatch.py::record_outcome` is the per-action credit router, and it assigns the
cluster reward as **the sign of `drive_potential_diff` only** — `+1.0` when the diff is positive,
`-1.0` otherwise. (Cited as a symbol rather than quoted: verbatim code in a design doc rots. Read
the function; if the sign collapse is gone, this whole subsection is stale.)

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
| `NAc.credit_operant_reward(reward: float)` **→ a thin wrapper over `update_cluster_reward`** | arbitrary float | **the motor-selection surface** — the least lossy channel available, but see the attenuation note below |
| `NAc.update_cluster_reward(reward=…)` | arbitrary float | cluster bias, `+= reward_bias_alpha (0.15) × reward`, clamped ±1.0 |
| `NAc.record_percept_valence` | an unvalidated float (`[-1, 1]` is the *stored* range — the accumulator clamps) | salience gating only, **and has no positive producer in-tree** |
| `CausalLink.update_prediction_rw` | the 4-member `Valence` enum, via `_VALENCE_TO_REWARD` = `{POSITIVE: 1.0, NEGATIVE: 0.0, NEUTRAL/UNKNOWN: 0.5}` | link value `[0,1]`, 0.5 = neutral prior |

Note the last row: the RW target lives on `[0,1]` with **0.5 as neutral**, not on `[-1,+1]`. A
"0.3 good" outcome has no lossless intake through the enum path at all.

**And note rows 1 and 2 are the same surface.** Rev 1 called `credit_operant_reward` "the rich
channel" beside a `update_cluster_reward` row marked "clamped ±1.0", implying two different
mechanisms; the Executor lens found it is a wrapper that looks up the pending action and calls
`update_cluster_reward(..., source="operant")`. It therefore inherits the identical
`reward_bias_alpha = 0.15` attenuation and the identical ±1.0 cap. **This is load-bearing for the
duck**: it is exactly why a graded servo-temperature or battery reading is less influential than
"accepts an arbitrary float" suggests — a reward of 0.02 moves the bias by 0.003 per trial against
a `causal_pos` term topping out near 1.0. Graded intake is necessary but not sufficient; magnitude
still has to clear the competition.

**Design consequence.** The duck's valence mapper should target
`NAc.credit_operant_reward` with an episode-scoped float, not the per-action enum path. Route
graded body state there, and accept that anything routed through `record_outcome`'s cluster term
is a sign.

### 5.2 "Commanded vs. applied with a named reason" is a NEW channel, not a wiring job

This is the finding that most changes the work estimate, and it is good news disguised as bad.

**The Reachy already computes this signal and then throws it away.**
`hardware/reachy/controller.py` records `last_clamped_axes` after clamping, and its **only**
consumer in `src/` is `tools/reachy.py::MoveTool`, which renders it as English prose for the LLM
(note that is `MoveTool` — not the `FocusOnSoundTool` discussed next; different tools, same file).
There is
no path from `last_clamped_axes` to the `PainBus`, to the NAc, or to any modality channel. A
refused or clamped command is **invisible to the substrate**.

It is worse than invisible in one configuration. `tools/reachy.py::focus_on_sound` returns
`ToolResult(success=True, …)` even when `clamped` is true or `reached` is false — deliberately,
and its payload is scrupulously honest (`faced_sound`, `clamped_to_head_limit`, an explanatory
`note`). But `tool_dispatch.py` computes `learn_success = success and not embodiment_failed`, so
the substrate books a **POSITIVE** outcome for a motion the payload says did not happen.

**Bound this honestly — and rev 1's stated reason was wrong even though its conclusion was
right.** Rev 1 said the graduated experiments suppress the tool-success floor via
`drive_relief_only` / `drive_credit_withheld` / `MAXIM_OPERANT_ONLY_CREDIT`. The Executor lens
found that **none of those flags appear in the Exp 45/53/53b/54 harnesses at all.** The real
reason is stronger:

| Experiment | Harness | Why the defect cannot fire |
|---|---|---|
| 45 | `scripts/orient_backbone/live_3_learn.py` and siblings | Call `NAc.update_cluster_reward` **directly**; `record_outcome` is never invoked and `focus_on_sound` never runs |
| 52 Phase A | `scripts/orient_substrate/9_hunger_relief_orient.py` | In-process, calls `NAc.credit_operant_reward` directly; no `record_outcome` |
| 52 Phase B | `scripts/benchmark_cradle_mother.py` | Genuinely sets `MAXIM_OPERANT_ONLY_CREDIT=1` — the only row rev 1's reason actually covered |
| 53 / 53b / 54 | `scripts/orient_backbone/exp53_cross_context_readout.py` | **Readout-only by its own module docstring**: "nothing calls `record_outcome` / `credit_operant_reward`" |

**And the flag is narrower than rev 1 implied.** `MAXIM_OPERANT_ONLY_CREDIT` reaches only the
`cluster_reward = None` branch in `record_outcome`. Upstream and unaffected by all three flags:
`learn_success`, the resulting `Valence.POSITIVE` causal link via `nac.observe`, and
`credit_goal(+1.0)`. So a clamped `focus_on_sound` still books a POSITIVE **causal link** under
every flag — the bound is over the **cluster-bias surface** the graduated readouts measure, not
over the substrate as a whole. State it that way or the bound is quietly overclaimed.

It bites where the full dispatch path is live — the LLM-primary runtime driving a real robot.
**Filed as D53** in [../bugs/README.md](../bugs/README.md) (see §8 item 9).

**Partial credit where it is due:** the divergence is not entirely unrecorded. **D35** (FIXED
2026-08-29, PR #569) routes the controller's achieved-vs-commanded divergence *warnings* into the
experiment JSONL as `controller_warning` records. Note the scope precisely: the handler is
`scripts/orient_backbone/exp53_cross_context_readout.py::_WarningsToJsonl`, attached by that
**harness** for the duration of a run — it is instrumentation, not a production path. So "dies at
the controller" is exact about the **substrate**, and must not be read as "nothing was done": an
experimenter running Exp 53's harness can see the divergence; the learner never can, and neither
can the runtime.

> **Recommendation, revised after the two-lens review round (2026-08-31).** Rev 1 recommended
> wiring `last_clamped_axes` into `ToolOutput.side_effects["embodiment_failures"]`. **Do not do
> that**, for two reasons the Architecture lens raised and I accept:
>
> 1. **Wrong semantics.** `embodiment_failures` means *the entity's own components failed or were
>    harmed* — it is populated from `active_failures` and read as harm by
>    `bridges/tool_pain_bridge.py`. A clamp is not harm; it is the controller **correctly
>    refusing** an out-of-workspace command. Constraint 4 names the signal exactly — a *labeled
>    unexecutable-intent* signal — and routing it to the harm channel would teach the substrate
>    "this action damages my body" for an event that means "this command was not executable from
>    here."
> 2. **It inverts the bug rather than fixing it.** `learn_success = success and not
>    embodiment_failed` is unconditional, so *every* clamped motion would book NEGATIVE —
>    including a turn that clamped at 40° of a requested 60° and **nonetheless centred the
>    sound**. In precisely the configuration where the defect bites, that is the mirror image of
>    the defect.
>
> **The right home is the comparator that already exists.**
> `proprioception/pain.py::PainDetector.set_movement_target` / `::_check_movement_failure` is a
> genuine commanded-vs-achieved comparator emitting a **graded** `PainType.MOVEMENT_FAILURE`, and
> `harm/joint_limit.py::JointLimitHarmPredictor` is its predictive half. Both are real, both are
> reachable only through `bridges/pain_bridge.py::PainCircuitBridge` on the legacy Selfy runtime,
> and **neither is wired into the substrate-primary / `agent_loop` path**. Reviving them there is
> the work — graded, correctly named, and no new comparator.
>
> **And the credit rule must key on the outcome, not on clamp-occurrence.** Whether a clamped
> action was good or bad is answered by "did azimuth improve?", which the orient path already
> computes as `potential_diff`. Clamping is information about *executability*, and belongs in the
> percept/pain stream; it is not by itself a valence.
>
> With that correction this remains **the duck's commanded-vs-applied divergence channel,
> prototyped on the Reachy, buildable today, with no duck** — which is the point worth keeping
> from rev 1.

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

1. **`proprioception/` is an aversion source only.** `MovementSample` is a fixed 7-field record — a
   `timestamp` plus a 6-DOF head pose (`yaw, pitch, x, y, z, roll`) — hardcoded to Reachy geometry; its only downstream is
   `PainDetector` → `PainSignal` → `PainBus`. It never becomes a `Percept`, never reaches
   `SensorEncoder.encode_sensors`, never gets an EC cluster. **A 48-dim joint vector has nowhere
   to land today.**
2. **Exactly two modality channels ship.** `runtime/agent_loop.py`'s `_SUBSTRATE_CHANNELS` is
   `(interoception, audio)`. Adding a third is **one `ModalityChannel` tuple entry plus a
   `read_values`/`read_ranges` reader pair — and, if it rides the existing exteroceptive readers,
   an edit to the hardcoded `_EXTEROCEPTIVE_ROOT_SENSORS = ("azimuth",)` tuple** (rev 1 said "one
   tuple entry", which was the one place this section understated the work; the Minecraft plan
   names both obstacles and rev 1 carried only the second). The seam is genuinely declarative,
   **but it is also a selection-dynamics recalibration, not a free extension.** The registry's own comment warns that `max_cluster_reward_bias` caps *per
   cluster*, so the summed term in `recommend_action` scales with channel count: 2 → 3 channels
   moves the range from ±2 to ±3 against a `causal_pos` term topping out near 1.0, and gate
   calibration (`min_confidence`) must be re-checked. **This is the same warning the Minecraft
   world-modality channel carries** ([minecraft_benchmark.md](minecraft_benchmark.md)); if both
   land, they compound, and the recalibration is one job, not two. **Owner:** the roadmap's
   **1.1.4** row, which already carries "the world modality channel plus its selection-dynamics
   re-baseline" — that re-baseline is hereby scoped as *one job for any additional channel*, so
   this document observes the shared cost rather than owning it. (Rev 1 spotted the sharing and
   named no owner, which would have left the job described in two plans and owned by neither.)
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

Concrete arithmetic, so the throughput requirement is a number rather than an intuition.

> **Corrected in rev 2.** Rev 1 derived the budget from Rescorla-Wagner. The Executor lens
> pointed out that this is the wrong rule for the duck: §5.1 and §5.3 both commit the duck's
> valence to `NAc.credit_operant_reward`, which is a **wrapper over
> `NAc.update_cluster_reward`** — and that rule is *not* RW. Both are given below, because the
> duck touches both surfaces.

**The surface the duck actually learns on** (`decisions/nac.py::NAc.update_cluster_reward`) is a
**linear accumulate-and-clamp**, not an exponential approach: the bias moves by
`NACConfig.reward_bias_alpha` (0.15) × reward per credited trial and clamps at
`max_cluster_reward_bias` (±1.0). At `reward = 1.0` that saturates in ~7 trials. But saturation is
not the behavioural quantity — **what matters is when the correct action's bias separates from a
competitor's**, and any pre-registration must state that separation criterion rather than a
convergence threshold. Note the direct consequence for graded reward: a signal of magnitude 0.02
moves the bias by 0.003 per trial against a `causal_pos` term topping out near 1.0 — the same
drowning failure mode §5.1 diagnoses for the sign collapse, one layer down. **This is why "graded"
is not the same as "influential" here.**

**The causal-link surface** (`decisions/nac.py::CausalLink.update_prediction_rw`) *is* RW:
`ΔV = α(R − V)`, with α derived from `NACConfig.base_learning_rate` (0.2) and a default novelty of
0.5 that no call site overrides, giving **α ≈ 0.14**; 90% of asymptote needs `0.86ⁿ ≤ 0.1`, i.e.
**n ≈ 16**. **Re-derive both figures if `reward_bias_alpha`, `base_learning_rate` or the novelty
default changes.**

Either way the order is the same: multiply ~10–16 credited trials per cell by cells, arms and
seeds, and a modest duck design — say 4 skills × 4 state bins × 4 arms × 6 seeds — lands in the
**thousands of episodes**. (Exp 45's direction learning moved 0.00 → 1.00 in ~10 trials — the
right order, but note it is *argmax-correctness across bins*, not 90% of an asymptotic link value,
so treat it as an order-of-magnitude sanity check and not as confirmation of either derivation.)

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
   highest in `docs/experiments/` is 54 as of 2026-08-31). Until the mic-array question (§8, item 1) is answered there is no
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
   unless the mic-array question (§8, item 1) comes back yes.
8. **Does the sim backend get built ahead of hardware, and at what priority against the 1.3
   fabric?** The MuJoCo half has no hardware dependency (§7). It is startable; it is not
   scheduled.

### A defect this investigation surfaced — FILED

9. **A clamped or unreached motion is credited as a success.** Filed as **D53** in
   [../bugs/README.md](../bugs/README.md). `tools/reachy.py::focus_on_sound` returns
   `ToolResult(success=True, …)` when `clamped` or `reached is False`, and `tool_dispatch.py`
   computes `learn_success = success and not embodiment_failed` — so the substrate books POSITIVE
   for a motion the tool's own payload says did not happen. Bounded: the graduated orient
   experiments suppress the tool-success floor, so no EARNED row is touched.

   > **Revised after the review round.** Rev 1 deferred filing until PR #577's ledger renumber
   > landed. The Architecture lens was right that this was wrong: the defect is *verified* against
   > two named symbols, and ledger rule 1 puts verified defects in the ledger while reserving the
   > plan's open-questions section for suspicions — so the plan doc was the wrong home by the
   > ledger's own rule, and "file it later" named no trigger and no owner. It is numbered **D53**,
   > deliberately skipping D49/D50 so #577's renumber has room. Rev 1 also proposed the
   > `embodiment_failures` wiring as the fix; §5.2 explains why that was rejected in review, and
   > the filed row records the correct fix instead.

### What this document deliberately does not commit to

- **Not** an experiment, a number, or a gate — the mic-array question (§8, item 1) blocks all three.
- **Not** the JEPA paired-data pairing. That remains what the 2026-08-30 scoping called it: a
  hypothesis that *raises* JEPA's bar rather than lowering it, revisited when the duck's sensing
  is known. §3's tool-name-collision caution is a prerequisite for even trying it.
- **Not** a schedule change. The duck stays in 1.3 (§7).
