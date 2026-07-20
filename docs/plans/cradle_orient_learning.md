# Cradle: learning to orient — orienting-to-sound as a *taught* developmental behavior

**Status:** Design draft (2026-07-19). The developmental reframing of
[substrate_primary_orient_learning.md](substrate_primary_orient_learning.md), prompted by the
observation that **a baby doesn't innately know that turning toward a sound localizes its source — it's
learned, as a caregiver guides them successfully to things.** This draft asks whether Maxim can *learn
to orient* the way an infant does: an innate reflex + a mild tendency as scaffolding, and the calibrated,
*useful* orienting learned via caregiver + cross-modal feedback.

Still `[engineering]`. If it works, it's a strong `[behavioral]` claim: **cross-session developmental
acquisition of a sensorimotor skill, no LLM in the action path, no fine-tuning** — learned, not built in.

---

## The insight — separate what's innate from what's learned

The productive-orienting work (PR #403) and the drive-relief plan (#404) quietly assume the *value* of
orienting is innate: the centeredness drive makes off-center aversive, so nulling it is intrinsically
rewarding. The developmental reality is subtler, and splits cleanly:

**Innate (the scaffold — small, seeds behavior, doesn't dictate it):**
- The **orienting reflex** — a loud + sudden sound triggers an automatic head-turn (the reflex tier we
  built; superior-colliculus, subcortical). Present at birth.
- A **mild** attend-toward tendency — a *weak* centeredness drive (small `pain_scale`), enough to bias
  exploration toward stimuli, not enough to be the whole reward.
- The **motor + sensory hardware** — the ability to turn (turn affordances) and to hear a bearing (the
  `azimuth` sensor).

**Learned (the developmental target):**
- **Audio→spatial calibration** — *which* turn (direction, magnitude) actually reduces `|azimuth|`. This
  is exactly the Exp-45 magnitude-learning, but acquired developmentally rather than pre-calibrated.
- **The *value* of orienting** — that turning toward a sound *reveals its source* / earns caregiver
  approval. This is the part #404 hardcoded and the part an infant actually learns.

So the experiment is not "does the agent null azimuth because it's uncomfortable" — it's "**does the
agent learn that orienting is worth doing, and how to do it well, from feedback.**"

---

## The teaching loop — where the reward *comes from*

Instead of (only) the built-in drive relief, orienting is rewarded by two developmentally-honest signals:

1. **Caregiver guidance** — when the infant orients correctly (reduces `|azimuth|` toward a sounding
   source), a **caregiver reacts positively** ("there it is!" / a warm response). This is the parent
   guiding the child to things. *First version:* the **narrator** emits the approval (available now);
   *richer version:* the deferred [mother_npc_stimulus_plan.md](deferred/mother_npc_stimulus_plan.md)
   Mother NPC.
2. **Cross-modal confirmation** — turning toward the sound makes the source *resolve* into something
   perceivable/nameable (the sound becomes a recognized entity — ties into the existing
   `infant_humanoid_naming` cradle line). Discovering the source is intrinsically rewarding; the audio
   bearing gets *bound* to a visual/entity identity, which is how the audio-spatial map calibrates.

Both flow through the same substrate path as #404 — `NAc` reward → `cluster_reward_bias` on the turn
actions, keyed on the `"audio"` EC cluster — so the mechanics are shared; only the **reward source**
changes from "innate drive relief" to "taught + discovered."

---

## The developmental arc (a cradle variant)

Rides the existing 4-act cradle machinery (`BUILTIN_ARCS["cradle"]`, `infant_humanoid` body):

- **Act 1 — reflexive / random.** Sounds occur; the infant reflex-turns to the loudest and otherwise
  turns near-randomly. No calibration, no learned value. Baseline orient-correctness ≈ chance.
- **Act 2 — guided.** The caregiver rewards correct orients; correct turns also reveal the source
  (cross-modal). The infant begins associating *turn-toward-sound → good*. `cluster_reward_bias` starts
  favoring the correct-direction turn.
- **Act 3 — calibration.** With direction learned, the infant learns *magnitude* — which step size best
  reduces `|azimuth|` (the Exp-45 boundary, but discovered from reward, not pre-set).
- **Act 4 — mastery.** Reliable, calibrated, *self-motivated* orienting — it orients even without the
  caregiver present, because the value has been internalized (persisted `cluster_reward_bias`).

---

## Measurement — and the ablation that proves it was *taught*

Port the orient_backbone / M3 metrics into the orchestrator (shared with #404):
direction-correctness, `|az|` reduction per event, latency-to-center, and the load-bearing
**cross-act / cross-session improvement** (persisted `aut_nac.json`).

**The decisive ablation — is orienting *learned* or merely *driven*?**
- **Arm A (taught):** caregiver + cross-modal reward, mild innate drive.
- **Arm B (drive-only, = #404):** the centeredness drive alone, no caregiver, no cross-modal reward.
- **Arm C (scaffold-only):** reflex + mild drive, *no* learning reward at all.

If Arm A reaches reliable calibrated orienting and Arm C does not, orienting is **learned**, not innate.
If Arm A markedly outperforms Arm B (faster, more robust, generalizes to novel azimuths/sources), the
**developmental/taught signal carries more than the built-in drive** — which is the whole thesis of this
reframing, and a stronger claim than #404's drive-relief in isolation.

**Graduation:** Arm A cross-act improvement, absent in Arm C, superior to Arm B → a `[behavioral]` entry:
*developmental acquisition of a sensorimotor skill from caregiver + cross-modal feedback, cross-session,
no LLM in the action path.*

---

## Prerequisites (what has to land first)

1. **The `always_active` affordance fix** (shipped, PR #403) — the infant must be able to *use* its
   `listen`/`turn` affordances; the goal-top-k was deactivating them.
2. **Give `infant_humanoid` the orient capability** — add an `azimuth` root sensor + `turn_left/turn_right`
   affordances (+ a *mild* centeredness drive, `pain_scale` ~0.15, smaller than base_humanoid's 0.3 —
   the *scaffold*, not the reward). Capability-driven, same declaration pattern.
3. **The Decision-4 substrate fixes (from #404)** — P1 `_normalize_value` range-aware, P2 azimuth
   de-bundled to the `"audio"` modality, P3 world-set azimuth in the substrate-primary tick. The cradle
   run is substrate-primary, so it depends on all three.
4. **A caregiver-reward hook** — narrator-emitted approval on a correct orient (first version); the
   Mother NPC (deferred) later.
5. **Cross-modal source-binding** — turning toward the sound resolves the source into a nameable entity
   (rides `infant_humanoid_naming` + the imagination/entity system). This is the richest prerequisite
   and can be staged: start with caregiver-reward-only (Arm A′), add cross-modal in a second pass.

---

## Why this is the stronger experiment

#404 asks *does the substrate learn the orient policy given a built-in reward.* This asks *does the agent
learn the orient behavior — policy AND value — the way a developing organism does.* The second is closer
to the project's actual thesis (learning without fine-tuning), harder to fake (the Arm C control has no
reward to exploit), and it retires an assumption we'd otherwise smuggle in (that centering is innately
rewarding). It also unifies three existing lines — the reflex tier, the Exp-45 magnitude learning, and
the `infant_humanoid_naming` cross-modal binding — into one developmental story.

## Sequencing relative to #404

Not either/or. **#404's mechanics (P1–P3) are shared prerequisites** — do them first. Then #404's
drive-relief run *is* Arm B of this experiment. So the cradle experiment **supersets** #404: run P1–P3,
then the three-arm cradle study, with #404's drive-only run as the built-in-reward control. One build,
the stronger claim.

## Connections to the broader research program (why this is worth revisiting)

The cradle orient loop looks small, but it is a **minimal sensorimotor primitive** that sits at the
intersection of several larger lines. That is why it is worth treating as a nucleus, not a one-off: it is
the cheapest concrete task that exercises action-conditioned prediction, cross-modal binding, grounded
symbols, and reward-driven policy *all at once*, developmentally. The connections, made explicit:

### JEPA cross-modal alignment — the cradle is a candidate *revival trigger* and *paired-data source*
[deferred/jepa_cross_modal_alignment.md](deferred/jepa_cross_modal_alignment.md) is a learned projection
that bridges the `SensorEncoder` (384-dim) and `LinguisticEncoder` (768-dim) spaces — cross-modal cosine
is *mathematically undefined* across different-dimensional encoders, and JEPA is the smallest new
mechanism that makes cross-modal binding defined. It is **deferred until** a problem appears that is
"structurally cross-modal AND unsolvable by threshold tuning, AND the cradle arc yields sufficient
training pairs." **The cross-modal step of *this* experiment is exactly that scenario.** When the infant
turns toward a sound and the source *resolves into a named/visible entity*, the audio bearing (sensor
space) must bind to the entity/word (language space) — a cross-dimensional alignment that threshold
tuning cannot close. And the *successful orient* is the event that **produces the paired data**
(azimuth-at-orient ↔ resolved-source) that JEPA consumes. So the cradle orient study is a leading
candidate to both **fire JEPA's revival condition** and **generate its training set** — worth flagging so
a future JEPA revisit starts here rather than from scratch.

### Cross-modal substrate binding (cancelled by Roy-4) — same wall, and the evidence for the projection
[archive/cross_modal_substrate_binding.md](archive/cross_modal_substrate_binding.md) tried Hebbian
audio↔visual binding edges and was cancelled because raw cross-modal cosine is undefined. The cradle
cross-modal step hits the *same wall* — which is not a dead end but the concrete demonstration that *some*
projection (JEPA) is required. If the cradle run needs cross-modal binding and can't get it in raw encoder
space, that is the resurrection evidence the binding plan's Stage-4a conditions asked for.

### Grounded language acquisition — orienting is how a symbol gets its referent
[grounded_language_acquisition.md](grounded_language_acquisition.md) Phase 2's "symbol-binding layer" is
structurally a JEPA. The cradle's cross-modal step — turn → the sound *becomes a nameable thing* — ties
into the existing `infant_humanoid_naming` line: **orienting is the act that binds an audio percept to a
referent.** A word/entity acquires its meaning partly *because* orienting to the sound reliably reveals
the same thing. So learning-to-orient is upstream of grounded naming, not parallel to it.

### Forward models / world-model prediction — the orient loop *is* action-conditioned prediction
The `Cerebellum` already learns "forward models for predicting sensory consequences" of actions
([src/maxim/embodiment/cerebellum.py](../../src/maxim/embodiment/cerebellum.py)). The orient loop is the
minimal such model: *"if I turn_left, azimuth will increase by ~δ."* Learning the turn→azimuth mapping
(Act 3 calibration) is forward-model learning; JEPA generalizes it from a scalar sensor to a *latent*
predictive model. So the cradle orient study is a concrete, measurable instance of the same predictive-
world-model thesis JEPA pursues at scale — a place to validate the primitive before the general machinery.

### The orient-specific substrate this rides
[substrate_native_orienting.md](substrate_native_orienting.md) (the azimuth "two learning signals" —
signed EC state + folded drive reward) and [orient_magnitude_learning.md](orient_magnitude_learning.md)
(Exp 45's magnitude calibration = Act 3) are the substrate + calibration mechanics; the innate reflex
scaffold is [hybrid_substrate_reflex_runtime.md](hybrid_substrate_reflex_runtime.md) (Track 2); imagined/
resolved sources ride [deferred/imagination_substrate_signals.md](deferred/imagination_substrate_signals.md).

**Net:** revisit this as the *entry point* for the cross-modal / world-model program. A future plan that
wants to fire JEPA, resurrect cross-modal binding, or ground the first symbol should start from the
cradle orient loop — it is the smallest task where all four threads are simultaneously live and
measurable, and it produces the paired data the rest of the program needs.

## Related

- [substrate_primary_orient_learning.md](substrate_primary_orient_learning.md) — the substrate mechanics (P1–P3) + the drive-relief arm (= Arm B here).
- [productive_orienting_affordance.md](productive_orienting_affordance.md) — the orient action + tiers + the `always_active` fix this depends on.
- [deferred/mother_npc_stimulus_plan.md](deferred/mother_npc_stimulus_plan.md) — the caregiver mechanism (richer version of the reward source).
- `BUILTIN_ARCS["cradle"]` + `infant_humanoid*` bodies + `infant_humanoid_naming` — the cradle developmental machinery this rides.
