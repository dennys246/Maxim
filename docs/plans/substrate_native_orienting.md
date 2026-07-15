# Substrate-Native Orienting — embodied sensorimotor learning as the substrate's first real-hardware policy

**Status:** ACTIVE umbrella plan (consolidated 2026-07-15). This is the authoritative
plan for the orient-to-center line. It **absorbs** the former
`audiovisual_orienting.md` (cross-track coordination) and sits above two execution
layers it does NOT re-do:
- [`reachy_orient_live.md`](reachy_orient_live.md) — the live hardware runbook (Phase 1; Step 1 **PASSED 2026-07-15**).
- [`perception_pipeline_placement.md`](perception_pipeline_placement.md) — the 1.1 placement/substrate-modality **infrastructure** the orient loop rides on (broader than orienting; stays separate).

**One-line thesis:** the orient-to-center loop — turn the head/body to drive a sensed
bearing error to zero, credited by drive-pain *reduction* through NAc — is the
cheapest **real closed-loop sensorimotor policy** the substrate can learn on hardware
that already works, and it is the natural first rung of a ladder that ends at
**spatial co-location as the grounding substrate for language and vision**.

**Why this plan exists (the strategic frame):** Exp 44 (the LLM-primary body_state
arms) empirically confirmed that LLM-primary embodied action is a scaffolding sink —
each fix revealed the next LLM-usability gap (model → n_ctx → scene → imagination →
harm → salience + tool-naming). See memory `project_exp44_llm_primary_divergence`.
Orienting is the opposite kind of task: **continuous, closed-loop, intrinsically and
deterministically rewarded, on real hardware, substrate-primary by nature.** It
sidesteps everything that made Exp 44 hard, and it is squarely on the substrate-native
side of the line the project drew ([`grounded_language_acquisition.md`](grounded_language_acquisition.md)).

---

## Why orienting is the right substrate-native task (honest case)

Everything that made Exp 44 a scaffolding sink is *absent* here — that absence is the tell:

- **Continuous, closed-loop, tight.** Sense bearing → turn → re-sense → reward. That's
  the shape NAc/substrate reward-shaping is actually good at, vs discrete-affordance +
  LLM-tool-calling.
- **Intrinsically and deterministically rewarded.** The reward *is* the sensor reading
  (bearing error). No narrator mediating harm, no `MAXIM_DETERMINISTIC_SCENE_EMBODIMENT`
  non-determinism, no tool-naming to fumble. Clean, attributable by construction.
- **Real hardware, already de-risked.** Reachy Mini Step 1 passed on-device (connect +
  wake + DoA-via-REST + head-yaw tracking); the DoA front-end, `doa_to_azimuth`, and
  `AzimuthDoASource` all work.
- **It's the exact claim the Reachy embodiment was scoped for.** Localization is
  on-chip (the XVF3800 DSP gives azimuth), so Maxim is **not learning to localize** —
  it's learning the **sensorimotor orient policy** (turn which way, how much, to drive
  azimuth-error → 0), credited by drive-pain reduction through NAc. Documented in
  [`docs/embodiment/reachy_mini/audio_localization.md`](../embodiment/reachy_mini/audio_localization.md).
- **Substrate-primary by nature.** The action path selects via `NAc.recommend_action`
  (no LLM in the loop). That is what makes it the substrate-native line and not more
  LLM-behavior-patching.

**Is there a better first task?** No — the orient loop's properties (real HW +
continuous reward + intrinsic/deterministic reward + a single clean 1-D axis) make it
unusually well-suited. Azimuth-only DoA (no elevation, front/back ambiguity) is a
*feature* for a first result: it makes the task a clean 1-D servo-to-center.

---

## The SEM mapping (the user's framing, sharpened)

The user's entity/sensor/modulator framing is right, with one sharpening about *what
is learned*:

| Role | Concretely |
|---|---|
| **Entity** | Reachy (the body) |
| **Sensor** | azimuth (DoA-derived; `"audio"` modality — frozen-centroid EC, `[-1,1]`, 384-d) |
| **Drive** | **centeredness** — a `HomeostaticDriveSpec`, set-point = azimuth 0; drive-pain ∝ `\|azimuth\|` |
| **Affordance/modulator** | head-yaw orient (`turn_left`/`turn_right`); body-yaw is the coarse axis (head clamps ~±15-18°) |
| **What is LEARNED** | the **state→action policy** — given azimuth-region (EC node), which orient action reduces error — as NAc `cluster_reward` keyed on that node |

**The refinement:** "orientation is the modulator to learn" is *nearly* right — the head
motor is the modulator, but the modulator itself is a fixed capability; the *policy*
(which orient action, given the sensed state, reduces the error) is what the substrate
learns, in NAc, not in the SEM modulator. This distinction is load-bearing for the
claim (below).

**Reward is potential-based (RESOLVED — Phase 0b).** Credit the *reduction* in
`\|azimuth\|` per action (`potential_diff = \|az_before\| − \|az_after\|`), not the
absolute post-state. This is drive-pain **reduction** = relief = negative-reinforcement,
which is bio-faithful AND mechanically selectable. It reconciles the audio (drive-pain)
and visual (reward) credit approaches into one signal. See the resolved-decisions table
below; ties to [`transition_based_drive_pain.md`](transition_based_drive_pain.md). The
codebase already learned this the hard way (memory `recommend_action reward-driven`:
"pain per-tool state-blind → use potential_diff").

---

## The two layers — be disciplined about them

The user's proposal contains a near-term achievable core and an ambitious research arc.
Keeping them separate is how this does **not** become another divergence spiral (the
"don't stack 4 unverifiable layers" cradle-cascade lesson).

### Layer 1 — the orient policy (the achievable, hardware-validated core)

Reward = azimuth-error reduction. The first **real-hardware substrate sensorimotor
policy** in the project. This is what the execution layers below already build:
`audiovisual_orienting`'s Phase 0/1 + the Reachy runbook's Steps 1-3. Machinery mostly
exists (DoA→azimuth→`SensorEncoder`→EC→centeredness drive→NAc→orient affordance).

### Layer 2 — spatial co-location as the grounding substrate (the north star)

The user's real excitement, and the deeper of the two: **sounds are co-located in space
and time with more than just head orientation — space binds sound, vision, and words.**

When the head centers on a sound at a stable pose, that pose **is** a spatial index. A
visual (camera) percept and a spoken **word** co-occurring at that pose/time bind to it
via EC/ATL co-activation: *"the word 'ball,' from the left, where I saw the ball."*
**Space is the shared grounding index** across audio + vision + language.

This connects three of the project's deepest threads at once:
- [`grounded_language_acquisition.md`](grounded_language_acquisition.md) — "language is
  I/O from substrate"; here the grounding is *spatial* (a word grounded in a
  sound-source location + a seen object is a spatially-grounded symbol). This is a
  concrete substrate-native curriculum for Phase 1's `token_id → ec_node_id` binding
  registry — space supplies the co-occurrence ground truth for free.
- [`cross_modal_substrate_binding.md`](cross_modal_substrate_binding.md) /
  [`jepa_cross_modal_alignment.md`](jepa_cross_modal_alignment.md) — the binding path.
  **Crucially, spatial AV has a structural reason to co-fire that the cancelled text
  case lacked:** a simultaneously-seen-and-heard target produces both percepts in the
  *same tick*. Roy-4 cancelled text-binding because priming/test clusters never
  co-fired in time; spatial co-location is the tractable case — **but measure before
  building** (a Roy-4-style co-activation pass, `MAXIM_EC_TRACE_ACTIVATIONS=1`). Do not
  resurrect the cancelled mechanism; a fresh plan if it passes.

**Discipline:** nail Layer 1 first. Layer 2 also needs vision wired (the Reachy camera +
the deferred GStreamer battle) and word input — real prerequisites. Layer 1 is the next
*step*; Layer 2 is the *arc* Layer 1 unlocks.

---

## The rigor caveat that makes-or-breaks the claim

**Distinguish a *learned* orient policy from a hard-coded servo.** A fixed "turn toward
the larger-azimuth side" controller would *also* drive error → 0 — so a working orient
loop is **not by itself** evidence of substrate learning. The claim needs the same rigor
the Roy experiments applied:

- start from untuned/near-random action selection and show NAc reward-bias **converges**
  to the correct turn-direction policy over trials (a learning curve), and/or
- show **transfer** (a policy learned for sounds-on-the-right generalizes; centeredness
  learned in one context primes another).

Design the experiment so "it learned" is separable from "it's a servo," or the result
will not earn the substrate-learning claim. The pre-registered measurement lives in a
future `docs/experiments/NN_reachy_sound_localization.md`.

---

## Central clarification: TWO learning signals, not one

The tensions between the audio and visual tracks dissolve once you see two distinct,
non-conflicting signals:

1. **Motor centering (SHARED, dim-trivial).** Orient to reduce bearing error. One
   centeredness `HomeostaticDriveSpec` (set_point 0) → drive-pain → NAc credit on
   discrete `turn_left`/`turn_right`. **Both audio and visual feed their scalar
   azimuth/bearing into this ONE drive** (scalar → 384-d `SensorEncoder` space →
   dim-consistent across modalities).
2. **Target valuation (MODALITY-SPECIFIC).** "Is this worth orienting to?" Audio: the
   hardware `is_speech_detected` gate (no learning). Visual: EC **category recognition**
   (person vs distractor), `cluster_reward` on orient/ignore — the
   substrate-generalization payoff that transfers to *novel* individuals
   ([Exp 43 Probe 4](../experiments/43_gaze_operant_substrate.md); high-dim `"vision"`
   modality).

Keeping these separate is the load-bearing design move: the shared motor loop is scalar
and bio-faithful; the valuation is visual-specific and high-dimensional. They never
compete.

---

## Resolved decisions (carried forward from the merged coordination plan)

- **Action path = substrate-primary, NOT LLM-gated. [CONFIRMED 2026-06-28]** The orient
  loop selects via `recommend_action` (no LLM in the action path), per the Exp 43
  probes. This **removes the old audio "sustained-sound / slow-convergence"
  constraint**, which assumed one LLM cognition cycle per orient step — each step is now
  a millisecond-scale NAc lookup at the perception/motor tick rate.
- **Motor credit [RESOLVED 2026-06-28, Phase 0b]: credit drive-pain REDUCTION per step
  (relief), as a state-conditioned positive (`potential_diff` on `cluster_reward`) — NOT
  the off-center pain state.** Phase 0b
  ([`scripts/orient_backbone/phase0_affect_study.py`](../../scripts/orient_backbone/phase0_affect_study.py))
  found **pain alone cannot drive `recommend_action`**: it is positive-gated
  (negative-only pain has nothing to select on → defaults to `stay`), the
  causal/`record_outcome` surface is per-tool **state-blind**, and off-center pain
  misinforms mid-approach (it punishes correct-but-incomplete actions). Reward
  (state-conditioned `cluster_reward`) drives a perfect orient policy; `potential_diff`
  (dense per-step `\|az\|`-reduction credit) additionally closes the far-state gap
  (far-dir 1.0 vs terminal-reward 0.94). Consistent with the NAc invariant (reward_bias
  clamps `[0,max]`; pain is a modulator handled by edge valence, not a selection driver).
- **EC modalities stay separate:** `"audio"` (frozen, `[-1,1]`, 384-d) and `"vision"`
  (frozen, vision-encoder dim) for category. The **spatial azimuth** from both is scalar
  → same 384-d space → dim-consistent (relevant only to binding, Phase 3b).

---

## Phased sequence (cross-track)

**Phase 0 — sim, no hardware: build the shared backbone + characterize the credit signals.**
- **0a — backbone [built]:** a modality-agnostic centeredness drive + discrete orient
  affordances + substrate-primary NAc credit, consuming an azimuth-error from *any*
  `PerceptSource`. This is Exp 43 Probe 1/2 generalized AND the audio "commit 5"
  generalized — both tracks collapse into this. Validated in sim (world-coupling rule
  for the sim; on hardware the mic/camera physically re-measure).
- **0b — affect-signal study [DONE 2026-06-28]:** ran pain / positive / both /
  `potential_diff` on the 0a backbone. Verdict above: orienting is **reward-driven**;
  credit drive-pain **reduction** (`potential_diff`, state-conditioned); **pain-state
  alone does not orient**.

**Phase 1 — hardware, AUDIO first.** Wire `AzimuthDoASource` → shared backbone, live.
Audio leads because its perception is *solved* → the cleanest path to a working live
learning loop, and it **validates the shared motor/drive/credit substrate on-device**
(which visual then inherits for free). Deliverable: *Reachy turns toward sounds and
learns.* **Execution runbook + live status: [`reachy_orient_live.md`](reachy_orient_live.md)
(Step 1 PASSED 2026-07-15; Steps 2-3 next).**

**Phase 2 — sim → hardware, VISUAL.** First clear **P1** (vision-encoder category
clustering on real skier-vs-object *images* — Exp 43 prerequisite, no robot; the
make-or-break for the substrate payoff). If it holds, wire the visual `PerceptSource`
(bearing → motor loop; category → valuation) into the *same* backbone. Deliverable:
*Reachy turns toward people and generalizes to novel people.*

**Phase 3 — hardware, fusion (two tiers).**
- **3a — drive-level azimuth fusion (cheap, no binding).** Both modalities' azimuth →
  one centeredness error; arbitration by confidence/recency; vision resolves audio
  front/back; audio extends vision FOV. Dim-agnostic. **This is the practical
  co-localization win** (the superior-colliculus AV orienting map) and needs none of the
  binding machinery.
- **3b — substrate cross-modal binding (GATED research).** Audio-azimuth node ↔ visual
  node binding into a multimodal object. **Gated on a Roy-4-style co-activation
  measurement** (`MAXIM_EC_TRACE_ACTIVATIONS`). Spatial AV has a structural reason to
  co-fire (same-tick see-and-hear) that the cancelled text case lacked — the tractable
  case, but **measure before building.** Fresh plan if it passes.

**Phase 4 (Layer 2 north star) — spatial-grounding curriculum (research direction).**
Once 3a/3b give a stable spatial index, the pose becomes the co-occurrence ground truth
for grounded language: a spoken word co-located with a seen+heard object feeds the
[`grounded_language_acquisition.md`](grounded_language_acquisition.md) Phase 1/2
binding registry with *spatially-grounded* labels. This is the "connecting words and
visuals to places in space" arc. Deliberately unscheduled — it is the destination, not
a next step, and depends on Phase 2 vision + word input + the Phase 3b co-activation
verdict.

---

## Complementarity (why co-running audio+visual is not redundant)

- **Audio:** 360° coverage, densely/continuously sensed, but coarse and
  front/back-ambiguous (linear array).
- **Vision:** precise and identity-bearing ("it's a *person*"), but FOV-limited.

Co-localization = each covers the other's blind spot: **vision resolves audio's
front/back ambiguity; audio pulls out-of-FOV targets into view for vision to confirm.**
This is the easiest instance of cross-modal binding because both modalities collapse to
one shared spatial variable (head-relative azimuth).

---

## Front-gate + consolidation note (Principle 3 + "merge before multiplying")

**Does the orient line need its own umbrella plan, or does it ride existing docs?** The
execution was already well-factored across three docs — a coordination plan
(`audiovisual_orienting.md`), a hardware runbook (`reachy_orient_live.md`), and an infra
abstraction (`perception_pipeline_placement.md`). What was **missing** was a single
authoritative statement of the *substrate-native learning thesis* + the *spatial-grounding
arc* + the *learned-vs-servo rigor*. This plan is that statement. It **merges** the
coordination plan up (archived) and **cross-links** the runbook + infra (kept separate —
the runbook is actively used on-device; the infra is broader than orienting).

**What each remaining doc owns (no overlap):**
- **This plan** — thesis, two-layer framing, rigor bar, cross-track phasing, the Layer-2
  north star.
- [`reachy_orient_live.md`](reachy_orient_live.md) — the live hardware bring-up runbook
  (SDK surface, per-step gates, calibration unknowns).
- [`perception_pipeline_placement.md`](perception_pipeline_placement.md) — the 1.1
  placement type/config abstraction + exteroceptive substrate-modality work (the `"audio"`
  EC modality, normalization, frozen-centroid decision). Broader than orienting.

---

## Open questions (pre-implementation review)

1. **Empty-state arbitration** — audio-empty (no sound → wait) vs vision-empty (no
   person → Layer-0 search) differ; the fused drive needs a policy for "audio says turn,
   vision sees nothing yet."
2. **Shared coordinate frame** — reconcile audio `(doa−π/2)/(π/2)` sign against the
   camera bearing sign *explicitly*, or the two modalities pull the head opposite
   directions. (Runbook Step 2 calibrates the audio sign on-device.)
3. **Shared body YAML** —
   [`reachy_mini.yaml`](../../src/maxim/_data/components/bodies/reachy_mini.yaml) already
   has head_yaw + the azimuth sensor + orient affordances (PR #387); add the
   camera/vision sensor there when Phase 2 lands (one body, both tracks).
4. **Learned-vs-servo measurement design** (the rigor caveat) — pre-register how
   convergence/transfer is shown before the live learning run.

---

## Pointers

- Execution runbook: [`reachy_orient_live.md`](reachy_orient_live.md)
- Infra: [`perception_pipeline_placement.md`](perception_pipeline_placement.md),
  [`audio_localization.py`](../../src/maxim/embodiment/audio_localization.py),
  [`docs/embodiment/reachy_mini/`](../embodiment/reachy_mini/README.md)
- Visual track: [`docs/experiments/43_gaze_operant_substrate.md`](../experiments/43_gaze_operant_substrate.md),
  [`scripts/gaze_substrate/`](../../scripts/gaze_substrate/)
- Backbone scripts: [`scripts/orient_backbone/`](../../scripts/orient_backbone/)
- Layer-2 / grounding: [`grounded_language_acquisition.md`](grounded_language_acquisition.md)
- Binding (cancelled / gated): [`cross_modal_substrate_binding.md`](cross_modal_substrate_binding.md),
  [`jepa_cross_modal_alignment.md`](jepa_cross_modal_alignment.md)
- Drive-pain refinement: [`transition_based_drive_pain.md`](transition_based_drive_pain.md)
- Interoceptive-drive sibling (different axis): [`sem_environmental_proximity_sensing.md`](sem_environmental_proximity_sensing.md)

## History

- **2026-06-28** — `audiovisual_orienting.md` drafted as a coordination plan unifying the
  audio (DoA) + visual (gaze) tracks; Phase 0a backbone built, Phase 0b affect study
  resolved the credit signal (`potential_diff`).
- **2026-07-09** — `reachy_orient_live.md` hardware runbook started.
- **2026-07-15** — Reachy Step 1 PASSED on-device (WS-era SDK ≥1.5 transport). This plan
  created: consolidates the coordination content, adds the Layer-1/Layer-2 framing + the
  spatial-grounding north star + the learned-vs-servo rigor bar. `audiovisual_orienting.md`
  archived (merged up).
