# Reflex layering: separating what hit the body, what it felt, and what it did

> **DEFERRED 2026-09-24 (owner decision).** Route 1 of this finding shipped as a fix: a
> `ReflexFiring` now records its `outcome` (`acted` / `failed` / `suppressed` / `dry_run`). Routes 2
> and 3 below are recorded here and not built.
> **Revive when**, whichever comes first:
> (a) anticipation is meant to be measured in the narrative world, not only the survival rig;
> (b) [adaptive_nociception.md](adaptive_nociception.md) revives, because its producer-side pain
> adaptation and this plan's step 3a are the same change seen from two sides;
> (c) a behavioural-graduation row 9 re-run is scheduled for a reason other than the minor-version
> heartbeat. That re-run is the **baseline taken before** route 2 or 3, not the place to build them,
> because both change the modulators row 9 measures;
> (d) the narrative reflex path gains a consumer that acts on `ReflexFiring` values.
> Tracking issue: [#870](https://github.com/dennys246/Maxim/issues/870).

## The finding (verified on `main`, 2026-09-24)

`embodiment/reflex.py` is described as a reflex arc (keyword → cooldown → habituation →
sensitization → pre-emption → tool). But every shipped response in `_data/reflexes/*.yaml` is
`damage_component` or `set_entity_sensor`. `attack_flinch` does not flinch: it **damages the torso**,
and `impact_brace` does not brace: it damages the legs.
The embodiment brief confirms the module's actual role. It is the third sensation layer ("Narrative
(Layer 3, keyword reflex fallback)"), the way narrated events reach the body when nothing physical
does.

Biology separates five stages that this code collapses into one number:

| Stage | Biology | Today |
|---|---|---|
| 1. Stimulus | The blow lands; tissue damage depends on the blow | keyword match |
| 2. Transduction | Damage becomes nociception | `damage_component` |
| 3. Perception | Pain *felt*: amplified by sensitization, reduced by expectation | PainBus → NAc (fixed gain) |
| 4. Reflex response | Withdraw or flinch; **this is what habituates** | only offered to the LLM as latent affordances (dodge/block/brace); never executed |
| 5. Anticipatory act | Bracing before impact, the only thing that genuinely reduces damage | not an act |

The result is that the three modulators land on the wrong stage:

- **Habituation** reduces the **damage the world inflicts**: repeated attacks wound less. In biology
  the flinch habituates, not the wound.
- **Sensitization** is implemented as "damaged parts take more damage". It should act on what the
  body feels and does: damaged parts *feel* more pain (perception), and, as in the canonical Aplysia
  model, the reflex *response* is amplified too.
- **Pre-emption** means that because the agent predicted the blow, it takes less damage. But no
  bracing act happens: the prediction alone reduces physical harm. The archived plan's rationale ("I
  was braced, the blow hurts less") assumes an act the code never performs.

This also breaks a rule the owner already set. [adaptive_nociception.md](adaptive_nociception.md)
"What it must not do": *"Adapt a copy of pain inside one consumer, such as memory, NAc or a reflex.
Only the producer adapts."* The reflex is the producer of this damage. But it adapts the stimulus
itself, which is worse than adapting a copy.

Two more gaps in the same module:

- **`context_key` is never passed.** `evaluate` accepts it, meaning "a new attacker resets
  habituation", but its only caller (`BioEnrichmentPipeline._evaluate_reflexes`) never supplies one.
  All habituation shares one global bucket, and dishabituation cannot happen. This is the D43 shape:
  the capability shipped without its composition.
- **Habituation never recovers.** Exposure counts only grow, and `reset_state()` has no production
  caller.
- **The sensor reflexes modulated nothing (a live defect, verified 2026-09-24).** The YAML wrote
  `set_entity_sensor` values as DELTAS, but `simulation/tools.py::SetEntitySensorTool` SETS the value
  and clamps it to `[0, 1]`, so the intensity-scaled (negative) delta wrote `0`: `environment_cold`
  zeroed `stamina`, and `startle` — whose bare `awareness` is not a root sensor on any shipped body
  (it is the `head` modulator's sub-sensor) — wrote an orphan root key and never touched the head.
  Habituation, sensitization and pre-emption scaled nothing on them. A discrete defect, not part of
  the routes below: **fixed** in [#871](https://github.com/dennys246/Maxim/issues/871).
- **`_data/reflexes/infant.yaml` is never loaded.** Reflexes load by the body's `archetype`, and no
  shipped body declares `infant`: the infant bodies are `archetype: humanoid`, so they get the
  humanoid set, and `infant.yaml`'s `thermal_contact` (which targets `arms.thermal`, a sensor the
  infant bodies DO have) never fires. Whether infants should load it (an archetype mapping, or
  reflex-set composition) is its own decision.

## Behaviour tiers (declared per [behavior_tiers.md](behavior_tiers.md))

| Behaviour | Tier today | Tier after route 3 |
|---|---|---|
| Narrated damage from a narrated blow | tier 2 innate prior, **wrongly modulated** | **tier 1 invariant**: the world's effect on the body is not learned |
| Felt pain gain (sensitization, expectation) | absent (fixed gain) | tier 2 with learned gain, **owned by [adaptive_nociception.md](adaptive_nociception.md)** |
| Flinch / withdrawal response | not executed | tier 2 innate prior, habituating |
| Bracing motor program | not an act | tier 2 innate prior (the brace itself is not learned) |
| Stimulus → brace trigger | not an act | learned (tier 3): a conditioned anticipatory response selected by NAc prediction; must happen **before** impact |

## Route 2: finish the half-shipped habituation (reframed)

As first proposed, route 2 would wire `context_key` and add recovery on the experience clock
(dishabituation after a gap). Doing that **before** route 3 would polish habituation that sits on the
wrong stage: it would make damage habituate more convincingly. So route 2 is recorded as a
**step of route 3**, and it applies to the *response* (stage 4), not to damage.

- **`context_key`:** derive it from what the percept says is the source (the attacker and the scene),
  and pass it from `_evaluate_reflexes`.
- **Recovery:** exposure decays on the experience clock (`memory/experience_clock.py`), not wall
  time, the same as the memory line and adaptive nociception.

## Route 3: re-layer the narrative sensation path

Steps, in dependency order:

1. **3a: unmodulated damage.** The narrated blow's damage becomes a pure function of the stimulus
   (`raw_intensity`). Sensitization and expectation move to the felt-pain producer, which is
   [adaptive_nociception.md](adaptive_nociception.md)'s producer-side adaptation. Do not build a
   second one here. Expectation's sign is not fixed: anticipation can reduce felt pain, and dread can
   increase it.
2. **3b: the reflex response becomes an act.** A flinch or withdrawal response executes (a real
   affordance, tier 2), and **its** intensity habituates (with route 2's `context_key` and recovery)
   and sensitizes.
3. **3c: bracing is an act.** Pre-emption stops scaling damage. Instead, a confident NAc prediction
   of the stimulus lets the body **brace** before impact, and bracing, if it happened in time,
   reduces the damage. That makes anticipation causal and measurable: *did the body act on the
   prediction before the blow?* For a release called "Anticipation", this is the readout.
4. **3d: consumers.** `EnrichmentResult.reflexes_fired` has no reader today. Either wire it (tell the
   agent it flinched, and put the efference copy into the deliberative context) or mark it dormant.

## What it must not do

- Change what the body **experiences** in any shipped scenario before row 9 is re-run. Route 1 was
  scoped to change nothing for a reflex that acts.
- Build a pain-gain mechanism here. That belongs to adaptive nociception.
- Let pre-emption reduce damage without an act that happened before impact.
- Touch 1.3's survival reflex or the Minecraft path. This registry is built only in the narrative
  sim (`simulation/orchestrator.py`, where the AUT bio-enrichment pipeline is wired).

## What gates it

- **Four-lens design review** before any build (it changes a mechanism an EARNED row cites; see
  [../../experiments/DESIGN_REVIEW.md](../../experiments/DESIGN_REVIEW.md)).
- **Behavioural-graduation row 9** (reflex system, Exp 09) was EARNED on habituation and
  sensitization trajectories. Steps 3a–3c move those modulators, which fires its **Re-run on:**
  trigger. The original Exp 09 raw log is lost; the 2026-08-18 heartbeat re-run is the surviving
  baseline, and a re-run after #870/#871 is planned.
- **Row 9's own hypotheses encode the wrong layering.** Exp 09's H4 is "habituation reduces
  *damage*" and its H5 is "sensitization amplifies *damage*". Route 3 deliberately breaks both as
  written, so building it means re-registering row 9's metric (habituation of the *response*,
  sensitization of *felt pain*), not just re-running it. A re-run that fails H4/H5 after route 3 is
  the intended outcome, not a regression.
- Frozen during experiment campaigns, like adaptive nociception.

## Links

- [adaptive_nociception.md](adaptive_nociception.md): owns felt-pain adaptation (step 3a's other half).
- [behavior_tiers.md](behavior_tiers.md) M4 (innate reflex gains) and M6 (pain intensity).
- [archive/percept_reflex_system.md](../archive/percept_reflex_system.md): the original design, whose
  items 6 (habituation reset on context change) and 8 (bracing reduces damage) this plan revisits.
- [../behavioral_graduation_candidates.md](../behavioral_graduation_candidates.md) row 9.
