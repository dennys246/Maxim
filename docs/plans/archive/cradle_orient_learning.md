# Cradle: learning to orient — a hungry infant, a mother who feeds, and the scaffold that fades

> **ADOPTED INTO THE ROADMAP 2026-08-13 (historical design record).** Written 2026-07-19 on
> `docs/substrate-primary-orient-plan`, never landed; the scenario it designs SHIPPED as
> `simulation/cradle_mother.py` + the `cradle_mother` arc, validated scripted by **Exp 46**
> (taught 0.90 vs yoked 0.36 / none 0.50) and embodied by **Exp 48** (GRADUATE 2026-07-23 →
> CONTESTED 2026-08-11; re-baseline pending under apparatus standard S7). The decisive
> taught-vs-no_feed ablation designed here IS Exp 46/48's arm structure. The
> JEPA-revival-trigger and grounded-language connections in §Connections remain live
> pointers into `deferred/jepa_cross_modal_alignment.md` and
> `grounded_language_acquisition.md`. Archived as the design record; the standing
> measurement environment ask ("cradle formalized") is a 1.2 roadmap item.

**Status:** Design (2026-07-19, rewritten around the mother-scaffolded feeding scenario). The
developmental grounding of [substrate_primary_orient_learning.md](substrate_primary_orient_learning.md).
The productive-orient work (PR #403, merged) shipped the orient *action* in llm-primary — but a live run
confirmed the LLM won't reliably sequence `listen → turn`, which is the LLM-scaffolding treadmill the
project decided to step off. This plan is the substrate-primary answer: **the agent learns to orient
because orienting is *taught and rewarded*, the way an infant learns to turn to its mother for food.**

Still `[engineering]`. If it works it is a strong `[behavioral]` claim: **developmental acquisition of a
sensorimotor skill from a caregiver, cross-session, no LLM in the action path, no fine-tuning.**

---

## The scenario (the core)

A **hungry infant** and a **mother** who speaks, feeds, and gently turns the infant's head toward her.
The loop, one feeding episode:

1. The infant's **hunger** drive rises (interoceptive discomfort). It looks around, not knowing why its
   stomach hurts — it cannot yet reliably orient itself.
2. The **mother guides its head toward her** — a SEM `target_effect` writing the infant's `azimuth` toward
   center (facing mom). This is the *scaffold*: the caregiver produces the oriented state the infant
   can't yet produce.
3. Now oriented, the infant **sees and hears her** — her face (visual) and her voice ("here comes the
   choo-choo train," rich infant-directed speech). Audio + visual co-occur, bound to what follows.
4. The mother **feeds it** — a `target_effect` reducing the infant's `hunger`. The discomfort resolves.
5. The substrate credits the *oriented state* with the hunger relief → **NAc reward on the orient
   action**, keyed to the audio-visual "mom" cluster.
6. Across episodes, the **scaffold fades**: the mother guides less, the infant completes the turn itself,
   and eventually orients to her voice/face *autonomously* — because it has learned that orienting toward
   mom ends the hunger.

**The whole thesis in one sentence:** the infant doesn't discover orienting on its own — it is *guided*
into the rewarded state, then *learns to reproduce it*, then generalizes to orienting on mom's voice
alone. Taught, not innate; and not hand-prompted either.

---

## Why this scenario (the design rationale)

- **It's the most fundamental learned orienting behavior there is** — rooting/orienting toward the
  caregiver for food is *the* first thing an infant learns to orient for. Not an abstract sound source.
- **The reward is concrete and strong, not abstract.** Hunger → oriented → fed → relief. No "caregiver
  approval" hand-waving; the reward is a real interoceptive drive resolving.
- **The mother's guidance *solves cold-start.*** The hard problem with "learn to orient" is: with only a
  weak innate seed, why would the agent *ever try* orienting, so what does reward reinforce? Answer: it
  doesn't have to try — the **mother guides it into the oriented state (`target_effect`), so it
  experiences "oriented → fed" immediately**, and learns to reproduce it. This is Vygotsky's
  scaffolding / zone-of-proximal-development, and it is far more bio-honest than "infant learns to orient
  from scratch." The scaffold *is* the cold-start solution.
- **It is inherently audio-visual.** The mother speaks (audio) and is seen (visual) at the moment of
  feeding, so cross-modal binding falls out of the scenario instead of being contrived.
- **It grounds the first symbols.** Her words ("choo-choo," "here comes") acquire meaning by being tied
  to a referent (her) and an outcome (food) — the grounded-language / `infant_humanoid_naming` line, now
  with a *reason* the symbol matters.
- **Cleaner substrate path than pure audio-orient.** The reward is *hunger relief* — **interoception**,
  already correctly encoded — so this **sidesteps the Decision-4 P2 de-bundle** that the exteroceptive
  azimuth path needs. Fewer prerequisites, less risk.
- **Mostly-existing machinery.** The gaze stack (`attention/gaze_controller.py`,
  `default_network/gaze_manager.py`) + the `scripts/gaze_substrate/` probes already showed **operant
  gaze redirection DECISIVE and cross-session learning STRONG** (2026-06-28, scratchpad). `infant_humanoid`
  already has a `hunger` drive. `target_effect` (caregiver acts on infant) and `speak` already exist. This
  is mostly *grounding proven machinery in a bio-honest scenario*, not building from zero — and it
  **promotes that scratchpad gaze result to a graduated developmental claim.**

---

## Innate vs learned (the split, refined by the scaffold)

- **Innate (weak seed):** the rooting/orienting reflex (loud/sudden → auto-turn, a DN reflex arc — Track 2),
  the hunger drive, and the *ability* to gaze/turn. Kept **weak** so learning does the work.
- **Learned (the target):** that orienting toward mom's voice/face *ends the hunger* — the **value** — and
  *which* turn achieves it — the **calibration**.
- **The scaffold bridges the two:** the mother produces the oriented+rewarded experience *before* the
  infant can, so there is a reward signal to learn from on day one. The learning is the transfer of the
  orienting from the mother's `target_effect` to the infant's own `turn`/gaze.

---

## The developmental arc (mother-scaffolded, the scaffold fades)

Rides the existing 4-act cradle machinery (`BUILTIN_ARCS["cradle"]`, `infant_humanoid`):

- **Act 1 — fully guided.** Mother guides the head all the way (`target_effect` → azimuth ≈ 0) + speaks +
  feeds. Infant passive; it *experiences* oriented-paired-with-relief. Baseline: infant produces ~0% of
  the orient itself.
- **Act 2 — co-active.** Mother guides *partway*; the infant must complete the turn to face her and be
  fed. Reward now credits the infant's *own* orient action. `cluster_reward_bias` on `turn` rises.
- **Act 3 — autonomous (visual).** Mother stops guiding; the infant orients to her *face* itself to be
  fed. Learned visual orienting.
- **Act 4 — autonomous (cross-modal).** The infant orients to mom's *voice* before seeing her — because
  the voice has been bound (audio ↔ visual ↔ food) — i.e. it has learned *why a sound is worth turning
  toward.* This is where the audio-orient work we built (PR #403) becomes meaningful: "orient to mom's
  voice," grounded in feeding.

**The fade schedule is the experiment's primary knob:** too slow and the infant never has to learn; too
fast and it cold-starts anyway. The fade *curve* — how quickly guidance can be withdrawn while the infant
keeps getting fed — is itself the measured learning signal.

---

## The concrete build (scope)

**New — the mother component** (`_data/components/npcs/mother.yaml` or similar):
- `speak` (exists) — rich, scripted infant-directed lines (motherese), emitted each episode.
- `feed` — a `target_effect` reducing the infant's `hunger` (caregiver acts on infant; `target_effect`
  already resolves a target body and applies deltas — confirmed in `tool_bridge`).
- `guide_head` — a `target_effect` moving the infant's `azimuth` toward center, **sign-aware** (a fixed
  delta won't null a signed value — reuse the `reflex_oriented_azimuth`-style toward-center logic, applied
  to the target). The *amount* it guides per episode is the fade knob.

**Mother behavior — reactive/scripted v1** (NOT a full NPC agent to start): narrator- or reaction-driven —
*infant hungry AND not-yet-oriented → guide (per fade schedule) + speak + feed*. The full
[deferred/mother_npc_stimulus_plan.md](deferred/mother_npc_stimulus_plan.md) agent is the richer later
version; the reactive script is enough for the experiment.

**Infant** — `infant_humanoid` + an `azimuth` root sensor + `turn`/gaze affordances + the existing `hunger`
drive (mostly there; add azimuth + turns, `always_active`, mild centeredness seed).

**Substrate wiring** — the oriented-state + feeding reward → `NAc` `cluster_reward_bias` on the infant's
orient action, keyed to the audio-visual "mom" cluster, so it learns to *reproduce* the orient. Hunger
reward is interoception (already clean); the audio-orient cross-modal step (Act 4) is where the Decision-4
fixes (P1–P3) and eventually JEPA come in.

**Measurement** (port orient_backbone / M3 metrics + gaze metrics into the orchestrator):
- **fraction of the orient the infant produces itself** (vs mother-guided) per act — the fade curve, the
  headline signal;
- latency-to-fed, hunger-relief per episode;
- **cross-session persistence** (`aut_nac.json`) — does an infant that learned in session N start session
  N+1 already orienting?

---

## The decisive ablation (is it *taught*, or just *driven*?)

- **Arm A (taught):** full scenario — mother guides (fading) + speaks + feeds.
- **Arm B (drive-only, = #404):** the centeredness/hunger drive alone, no mother, no guidance.
- **Arm C (scaffold-only):** mother guides + feeds but the *learning reward is disabled*
  (`MAXIM_NAC_REWARD_BIAS_DISABLED`) — the infant is guided-and-fed but cannot *learn* from it.

If **A** reaches autonomous orienting (the scaffold successfully fades) and **C** does not (guided-and-fed
but never learns to do it itself), orienting is **learned, not innate or hand-fed**. If **A** markedly
beats **B** (the drive alone is a weak, slow teacher vs. the caregiver-scaffolded one), the **taught
signal carries more than the built-in drive** — the whole point of the reframing.

**Graduation:** Arm A's fade curve (guided → autonomous), absent in Arm C, cross-session-persistent → a
`[behavioral]` entry: *developmental acquisition of a caregiver-scaffolded sensorimotor skill, no LLM in
the action path, no fine-tuning.*

---

## Mode split: scaffolded in llm-primary, learned in substrate-primary

Orienting is *available and scaffolded* in llm-primary and *inherently learned* in substrate-primary — the
clean separation between the product mode and the thesis mode. Two things that are easy to conflate:
**hearing** (the audio percept reaching cognition) is ON in **both** modes (you cannot learn to orient to
a sound you cannot hear); the orienting **behavior** is what differs.

- **llm-primary — scaffolded (product default).** Reflex tier + the LLM choosing `listen`/`turn`.
  Controlled by `--audio-orient` / `MAXIM_SIM_AUDIO_ORIENT`; `--audio-orient false` opts the whole channel
  out. **Keep opt-in (default OFF) until validated; flip to default-ON-for-llm-primary only once proven.**
- **substrate-primary — learned (thesis).** Hearing ON (percept → EC via P3), **no scaffolded orient** —
  value + calibration learned from reward (this plan). Keep a **weak innate seed** (mild drive ~0.15 + the
  DN reflex arc, Track 2 — NOT the llm-primary §1.16 sim-model) for bio-honesty (newborns *have* the
  reflex) and cold-start; the mother's guidance is the *external* scaffold that seeds the reward. So
  "substrate-primary disables audio-orient" means: **disables the scaffolded/LLM orient, keeps hearing +
  a weak seed + the caregiver scaffold, and learns the rest.**

Already aligns with the code: §1.16 (the llm-primary scaffold) is gated `aut_mode != "substrate-primary"`;
P3 (route azimuth to the substrate) is the distinct learned path.

---

## Connections to the broader research program (why this is worth revisiting)

The mother-feeding cradle is a **minimal sensorimotor primitive** that sits at the intersection of several
larger lines — the cheapest concrete task that exercises action-conditioned prediction, cross-modal
binding, grounded symbols, and reward-driven policy *all at once*, developmentally, on a real reward.

### JEPA cross-modal alignment — the cradle is a *revival trigger* and *paired-data source*
[deferred/jepa_cross_modal_alignment.md](deferred/jepa_cross_modal_alignment.md) is a learned projection
bridging the `SensorEncoder` (384-dim) and `LinguisticEncoder` (768-dim) spaces — cross-modal cosine is
mathematically undefined across different-dimensional encoders. It is **deferred until** a problem appears
that is "structurally cross-modal, unsolvable by threshold tuning, and comes with sufficient paired data."
**Act 4 of this scenario is exactly that:** the mother's *voice* (sensor/audio) must bind to her *face*
(visual) and her *words* (language) — a cross-dimensional alignment — and every feeding episode
**produces the paired data** (voice ↔ face ↔ hunger-relief). Grounded in a real reward, not an abstract
source. This scenario is a leading candidate to both **fire JEPA's revival condition** and **generate its
training set**.

### Cross-modal substrate binding (cancelled by Roy-4) — same wall, and the evidence for the projection
[archive/cross_modal_substrate_binding.md](archive/cross_modal_substrate_binding.md) tried Hebbian
audio↔visual binding and was cancelled because raw cross-modal cosine is undefined. Act 4 hits the same
wall — which is the concrete demonstration that a projection (JEPA) is required.

### Grounded language acquisition — the mother's words are the first grounded symbols
[grounded_language_acquisition.md](grounded_language_acquisition.md) Phase 2's "symbol-binding layer" is
structurally a JEPA. Motherese ("here comes the choo-choo train") tied to her presence and to feeding is
the most concrete symbol-grounding setup there is — ties into `infant_humanoid_naming`.

### The gaze machinery — proven, and this promotes it
`attention/gaze_controller.py` + `default_network/gaze_manager.py` + the `scripts/gaze_substrate/` probes
(operant redirection, category transfer) already demonstrated substrate gaze learning. This scenario
grounds that proven mechanism in a bio-honest developmental task and promotes it to a graduated claim.

### Forward models / world-model — the orient loop is action-conditioned prediction
The `Cerebellum` learns "forward models for predicting sensory consequences"
([src/maxim/embodiment/cerebellum.py](../../src/maxim/embodiment/cerebellum.py)). The
guide→see→feed→relief loop is a minimal predictive model (orient predicts food); JEPA generalizes it from
scalar sensors to a latent. This scenario validates the primitive before the general machinery.

**Net:** revisit this as the *entry point* for the cross-modal / world-model program — the smallest task
where all four threads are simultaneously live and measurable, on a real reward, and it produces the
paired data the rest of the program needs.

---

## Prerequisites & sequencing

1. **`always_active` affordance fix** — SHIPPED (PR #403), so the infant can use its own `listen`/`turn`.
2. **Mother component + 3 affordances** (`speak`, `feed` via `target_effect` hunger, `guide_head` via
   sign-aware `target_effect` azimuth) + the reactive behavior script + the **fade schedule**.
3. **`infant_humanoid` prep** — azimuth sensor + turn/gaze affordances (`always_active`) + mild
   centeredness seed; hunger drive exists.
4. **Decision-4 P1–P3** (from #404) — needed for the *audio-orient* cross-modal step (Act 4). The visual
   gaze + hunger-reward core (Acts 1–3) largely sidesteps them (hunger is interoception).
5. **transition-based drive-pain** (near-path) — clean the hunger/centeredness reward signal.
6. **M3 + gaze telemetry** in the orchestrator so the fade curve is measurable.

**Order:** the visual gaze + hunger + mother-scaffold core (Acts 1–3) is the first, cleanest slice and does
*not* depend on the Decision-4 fixes; the audio-voice cross-modal step (Act 4) folds in P1–P3 and sets up
JEPA. So: build the mother + infant + fade + reward-wiring → run Acts 1–3 (visual) → add the voice (Act 4).

## Related

- [substrate_primary_orient_learning.md](substrate_primary_orient_learning.md) — the substrate mechanics (P1–P3); Arm B (drive-only) is defined here as the built-in-reward control.
- [productive_orienting_affordance.md](../deferred/productive_orienting_affordance.md) — the orient action + tiers + the `always_active` fix (shipped) this builds on; the 2-D elevation extension path.
- [deferred/mother_npc_stimulus_plan.md](deferred/mother_npc_stimulus_plan.md) — the full caregiver-NPC mechanism (the reactive script is the v1 stand-in).
- [deferred/jepa_cross_modal_alignment.md](deferred/jepa_cross_modal_alignment.md) — the cross-modal projection Act 4 sets up + feeds.
- [grounded_language_acquisition.md](grounded_language_acquisition.md) — motherese → grounded symbols.
- `BUILTIN_ARCS["cradle"]` + `infant_humanoid*` + `attention/gaze_*` + `scripts/gaze_substrate/` — the developmental + gaze machinery this rides.
