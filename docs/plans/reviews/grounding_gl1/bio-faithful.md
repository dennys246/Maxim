# GL1 four-lens design review: BIO-FAITHFUL lens

Reviewed: `docs/plans/grounding.md` (umbrella) and `docs/plans/autonomic_layer.md` (GL2), worktree
`.worktrees/gl1` (origin/main with #1167 and GL0 #1186). This lens asks one question: does each mechanism do
its biological job, or a caricature of it? I do not re-open owner decisions G1–G8 or the 2026-10-09 GL2a
decisions (tool-path record only; measured drop-oldest bound; unweighted `valence = relief − harm −
nociception`; urgency = pressure only). Where a finding touches one of those, it changes the **inputs** a
decided formula consumes, never the formula.

**Verdict: 1 DO-NOT-BUILD / 9 SHOULD-FIX / 5 NIT.**

---

## DO-NOT-BUILD

### DNB-1 (GL2a): `relief` and `harm` ignore internal state (no alliesthesia), and only the core is persisted, so the error cannot be undone

**Issue.** §3.1 defines `drive_delta = drive_comfort_progress / drive_span`, `relief = max positive
drive_delta` and `harm = max |negative drive_delta|`. The guard `relief == max(drive_relief)` ties `relief` to
the existing `relief_fraction_from_progress`. The helper's own docstring says what that means
(`embodiment/sem.py::drive_comfort_progress`): it "credits movement toward the set point even *inside* the
comfort band (where `drive_pain_for_value` is 0)". For entropic drives it credits any movement in the
comfort direction, including movement below `satisfaction_threshold`. So the record scores the same movement
the same way whatever state the body is in. Moving away from the set point inside the comfort band counts as
**harm**, and eating when sated counts as **relief**.

Worked through the shipped YAML (`infant_humanoid.yaml` `arms.thermal`: set_point 0, band 0.5, range [−1,1],
span 1; `core_temperature`: band 0.25):
- `cradle_fire_pit.yaml::warm_self` writes `arms.thermal +0.2`. Its YAML comment calls it "the positive
  substrate edge ('fire = warm')" and notes it "stays within the arms' comfort_band". The record scores harm
  0.2. In GL2a's own gate sequence (`cool_air` ×2 then `warm_self` ×2), each `warm_self` relieves core
  temperature by about 0.2/span and harms the arms by 0.2. Net valence is about 0 for the action the cradle
  was designed to make feel good.
- `cradle_blanket.yaml::touch` (`arms.thermal +0.1`) gives harm 0.1 and valence −0.1. `fire_pit.touch` gives
  harm 0.6. Both are harm-only vectors in the 6-d core, so they point the same way and differ only in
  magnitude, which cosine cannot see (`docs/wiring/cosine-separation-is-directional.md`). That breaks the
  dissociation pair `latent_forward_model.md` §6 lists as "opposite sign by consequence".
- On `minecraft_player`, `food` is entropic down with satisfaction 16. Eating at food 18 counts as relief.

Biology: thermal and gustatory pleasure depend on internal state. This is **alliesthesia** (Cabanac 1971):
warmth on the skin is pleasant when the core is cool, neutral when the body is comfortable, and unpleasant
when it is hot. Food is rewarding when hungry and neutral or aversive when sated. A body-consequence code
that scores the same physical change the same way regardless of need is the textbook caricature. It measures
"distance moved from the set point", not "what this did for my body".

**Why it blocks GL2a, not just a later stage.** §3.1.4 persists only "the core plus `str(pid)` and
`provenance`" into `EncodingSignals.extra["interoception"]`. Those Hippocampus `"loop"` traces are the replay
buffer GL4 S3 refits from, and the fresh capture GL4 S0b audits. The per-drive block (`pressure_before`,
`deviation_after`, `drive_delta`) is not persisted, so a corrected `ans-v2` cannot be recomputed from data
captured under `ans-v1`. In addition, the GL2a falsifiable gate commits a hand-computed table, which would
lock the mis-scoring in as the expected answer.

**Concrete fix (two plan edits; the decided formula is unchanged):**
1. Define the core `relief` and `harm` from the **change in `drive_pressure`** per drive:
   `relief = max_d (p_before − p_after)⁺` and `harm = max_d (p_after − p_before)⁺`.
   `sem.py::drive_pressure` already exists, already sits in the record (`pressure_before`; urgency is
   `max pressure_after`), is 0 inside the homeostatic comfort band and on the satisfied side of an entropic
   drive, and is normalised per drive. This is first-order alliesthesia with no new formula. Keep `drive_delta`
   (raw signed progress / span) in the per-body block as the physical description.
   - Recomputed: `fire_pit.warm_self` after 2× `cool_air` gives harm 0 and relief ≈ 0.27, so valence is
     positive. `blanket.touch` at rest gives 0. `fire_pit.touch` gives harm 0.2 plus nociception once GL2b
     lands. Sated eating gives 0.
   - Replace the guard `relief == max(drive_relief)` with "positive part of `drive_delta` == `drive_relief`".
     That still pins one derivation per formula. The existing `ToolOutput.drive_relief` trio stays as it is,
     because it is a memory-strength record with its own consumers.
2. Persist the **per-drive block** (`pressure_before`, `pressure_after` or `deviation_after`, `drive_delta`,
   and the caused/felt flag from SF-2) in `extra["interoception"]` beside the core. Then any later `ans-vN`
   projection can be recomputed from the replay buffer, and `schema_id` versioning works as intended rather
   than in name only.

---

## SHOULD-FIX

### SF-1 (GL2a): `drive:health` is counted twice, and its "nociception" measures accumulated damage, not this event's injury

**Issue.** The record carries `drive:health` as nociception (`proprioception/pain.py::TISSUE_DAMAGE_DRIVES`).
Its intensity comes from the drive pain, `sem.py::drive_pain_for_value`, which is a **level** function. On
`minecraft_player` health (set 20, band 6, pain_scale 0.5), losing 1 hp at 11 hp gives nociception
min(1, (9−6)·0.5) = 1.0. Losing 1 hp at 15 hp gives 0. Both events also put the same Δhealth into `harm`. So:
- one physical change is counted twice in valence (harm and nociception). That is a hidden ×2 weight on
  health relative to oxygen, which the owner's *unweighted* decision did not choose;
- the "pain of this event" depends on how damaged the body already was, not on what the event did. Real
  nociceptors code the noxious stimulus's intensity, not the cumulative deficit;
- drowning damage (hypoxic, nearly painless, as the plan itself concedes) lands in the nociception dimension
  at 1.0. That makes "a Minecraft burn and a cradle burn land near each other" (§3.1.1) partly an artifact:
  Minecraft *drowning* lands there too.

**Fix.** For `TISSUE_DAMAGE_DRIVES`, compute the record's nociception from **this event's injury**
(normalised Δhealth loss, or the PainSignal published on entry or re-injury only), and exclude that drive
from `harm`, so each transduction counts once. If the bridge exposes a damage source, put it in `extra`
(UNVERIFIED that it does). Otherwise flag `injury_cause_unknown` so GL4 can separate drowning from contact
damage.

### SF-2 (GL2a): the tool-path nociception mixes pain the action *caused* with pain the body merely *felt* while it ran

**Issue.** `bridges/tool_pain_bridge.py::ToolPainBridge.pop_invocation_pain` returns "caused if caused is not
None else felt": the peak NOCICEPTIVE pain published while the invocation was pending. The record's
`nociception` (§3.1: "max PainKind.NOCICEPTIVE intensity") inherits this without marking which one it is.

Biology: a forward model's job is to predict **reafference**, the sensory consequence of one's own action,
and to keep it apart from exafference (efference-copy cancellation; the reason you cannot tickle yourself).
Felt-but-not-caused pain is contiguity without contingency. Training the predictor on it teaches "this action
hurts" from coincidence.

**Fix.** Record which it is: `nociception_caused` and `nociception_felt`, or a flag in `extra`. Persist it with
the per-drive block (DNB-1). GL4 decides which one trains the target, and S0b counts both.

### SF-3 (§3.1.1 / GL4 target): the core mixes phasic consequences with tonic state levels, so a predictor can score by copying context

**Issue.** The 6-d core combines **changes** (relief, harm, caused nociception) with **levels** (urgency =
max pressure *after*; `drive_pain` is a level; `deviation_after` in the per-body block). The bio job of the
forward model the plan names (cerebellum plus a slow cortical learner) is to predict the *change* an action
causes. The level terms are largely predictable from `ActionContext`, which already carries the sensed
readings. The predictor can then pass S2's gates by reproducing state, and "similar by consequence" turns
into "similar by the situation where it was used".

**Fix.** Make the forward-model target's core the phasic subset (or the Δ of each level term), and pass
levels in as context. S2 and GL5 should add a **context-copy baseline**: a model that sees context but not the
action must lose on the phasic dimensions. The record itself can keep every field. This changes only the
`as_vector` projection.

### SF-4 (GL2c): satiation intensity ignores how deprived the body was, and its credit has no prediction error

**Issue.** Option A emits `intensity=<relief of the satiated drive>`, which is the delta of the step that
crossed the threshold. That step can be tiny, for example a last drift across the hysteresis band. Biology
(alliesthesia, again): the hedonic value of a correction scales with how deep the deficit was, and the latch
already holds that depth (`Entity.drive_breach_severity`). Separately, `_distribute_reward_from_reaction` →
`NAc.credit_node(+)` accumulates on every satiation up to the clamp. Dopaminergic reward signals are
prediction errors: a fully predicted reward produces none (Schultz 1997). On Exp 60, surfacing repeats every
trial, so the positive bias would saturate and end up measuring how often the agent surfaced, not how
valuable surfacing was.

**Fix (for GL2c's joint review with the relief store).**
- Set intensity from the episode's latched severity (deprivation depth), not the last step.
- Deliver the signal as a prediction error against the relief store's expectation for that cluster. The
  store is the natural home for the expectation, which is a bio argument for option B or C over A. It does
  not change G7's finding that every routing fires the rows.

### SF-5 (GL2c): "satiation" on passive recovery is the wrong event; relief from an aversive state belongs at its offset

**Issue.** §3.5 emits satiation whenever a latch clears, including `minecraft_player` `health` regeneration
back past 14 hp. That is slow, passive and not consummatory, and the latch can clear long after the harm
stopped. Biology separates three things:
- **consummatory satiation**: a corrective act ends a drive, as with food or warmth;
- **relief at the offset of an aversive stimulus**: negative reinforcement. Pain-relief learning (Tanimoto,
  Heisenberg & Gerber 2004; Gerber et al. 2014) gives cues present *at pain offset* a positive value;
- **passive recovery**, which is neither.

Routed through the temporal-credit distributor, a regeneration crossing would credit whatever action
happened to be eligible at that moment. Calling oxygen recovery "satiation" is a decorative label: air hunger
is relieved, not sated. Biological satiety is also partly pre-absorptive (eating stops before the deficit is
corrected); the crossing rule is a post-absorptive simplification.

**Fix.** Type the positive event by kind:
- `satiation`: a consummatory or corrective act with a tool cause (`cause` not None);
- `relief`: the offset of nociception or air hunger, at the event where the aversive input stops;
- `recovery`: a passive or world crossing (`cause` None). Record it, but deliver no credit unless an arm
  declares it.

The "satiation" `ReactionKind` should be used only for the first. State the crossing rule as an innate-prior
simplification.

### SF-6 (GL2b(iii)): cause attribution is handed to the learner as ground truth, and the learning rule has no prediction error

**Issue.** §3.4 takes the cause from the producer (`CauseRef.entity` is the YAML noun) and tiers it as an
**invariant** ("who did it is a fact the producer knows"). Biology: Pavlovian learning attaches value to the
*perceived* cue by contingency, with cues competing through prediction error (Rescorla–Wagner; Kamin
blocking; overshadowing). The existing accumulator, `decisions/nac.py::NAc.record_percept_valence`
(`current + α·v`, clamped), has no prediction error and no extinction. With an oracle cause, only one cue can
ever be credited, so cue competition can never be tested. "Keyed on the cause" is then an engineering prior,
not learned stimulus valence.

**Fix.**
- Re-tier cause attribution as an "engineering prior (oracle attribution)", not an invariant.
- State that cause rows cannot be cited as learned stimulus valence in GL5 or in any claim.
- File a follow-up with a trigger for RW-style updating (prediction error, extinction) on the cause
  namespace.
- For GL2c's positive cause rows, the plan names no `failure_mode` key. Give appetitive rows their own key so
  appetitive and aversive values are not netted into one scalar. Biology keeps them in partly separate
  populations (basolateral amygdala positive and negative neurons; Namburi et al. 2015).

### SF-7 (grounding.md track table / GL3): Minecraft world sensors are mapped to the touch-and-proprioception track

**Issue.** The table puts "Minecraft world sensors via the pump" on `mechano_proprio` (dorsal column /
medial lemniscus). `minecraft_player.yaml` declares:
- `health`, `food`, `saturation`, `oxygen`: interoceptive;
- `light_level`, `nearest_hostile_dist`, `hostile_count`, `nearest_player_dist`, `is_raining`: exteroceptive
  distance senses;
- `time_of_day`: circadian, retinal to SCN;
- `speed`, `on_ground`, `look_pitch`: vestibular and proprioceptive;
- `xp_level`: abstract.

Only the last group belongs on a dorsal-column track. Because a `ReceptorSpec` declares its tracks, this
"innate" mapping would put air hunger on a touch channel.

**Fix.** Map per sensor in GL3.B0's census:
- vitals → `affective_slow` (interoceptive);
- distance senses → `extero_detail`;
- `time_of_day` → its own clock input;
- `speed`, `on_ground`, `look_pitch` → `mechano_proprio`.

The EC `world` modality stays byte-identical. This is track membership only.

### SF-8 (GL5 / T9 claim): name the phenomenon and use its standard design: acquired equivalence

**Issue.** "Concepts become similar by what they do to the body" is a known, measured phenomenon:
**acquired equivalence and distinctiveness** (Honey & Hall 1989; in humans, Myers/Shohamy et al. 2003).
Its standard design isolates similarity that comes from consequences:
1. Stage 1: A→O1, B→O1, C→O2.
2. Stage 2: retrain A→O3.
3. Test: B should inherit O3 more than C does.

GL5's jet triad trains `flame_jet` only and probes `water_jet` with no experience of it. Any inversion must
then come from the target entity's sensed readings in `ActionContext`, which is perceptual-feature
generalization, not consequence similarity. Semantic generalization of fear is also the *default* in humans
(Dunsmoor & Murphy 2015, category-based fear generalization). So "`water_jet` must not inherit the aversion"
without differential experience is not what biology predicts.

**Fix.**
- Add an acquired-equivalence arm to the GL5 prereg, with Stage-1 experience on every item.
- For the jet triad, either give `water_jet` differential experience, or relabel its result as
  perceptual-feature generalization.

### SF-9 (thesis framing): "word embedding = innate prior" is the wrong bio analog

**Issue.** mpnet's geometry is learned from human text. For the agent it is an **inherited cultural prior**,
not an innate one. The bio comparator is **instructed or social learning versus experienced learning**:
instructed fear (Phelps et al. 2001) and observational fear (Olsson & Phelps 2007) both reach the amygdala
and compete with first-hand learning. Truly innate priors behave differently: prepared associations such as
Garcia taste aversion resist being overridden. Calling the word prior "innate" therefore predicts the wrong
α dynamics, and it mixes up "hard-coded by us" (the behaviour-tier sense) with biological innateness.

**Fix.** Keep the behaviour-tier label, which is an engineering category. In the thesis, the insula/LFM
prose and the GL5 prereg, call it an "inherited (linguistic/cultural) prior". Frame α's fall as precision
weighting of instruction against experience. GL5 then has a real biological comparison to make.

---

## NIT

- **N1 (grounding.md insula row).** The insula represents interoceptive state and computes **interoceptive
  prediction errors** (Barrett & Simmons 2015; Seth 2013). "Was it good" is integrated in the anterior insula
  together with the OFC, amygdala and vmPFC. Suggest: "posterior insula → the outcome record; the
  predicted-vs-actual comparison (anterior insula) is GL4". Otherwise the row reads as decoration.
- **N2 (autonomic §3.2, grounding track table).** The justification that DRIVE breaches ride `affective_slow`
  under "Craig's lamina-I pathway" fits thermal, pain and muscle afferents. Air hunger is chemoreceptive
  (carotid and medullary chemoreceptors → NTS). Hunger is mostly humoral and vagal, acting on the
  hypothalamus. Say "homeostatic afferents (lamina-I spinal + vagal/NTS cranial)", and note that hunger is
  humoral state, not an afferent event.
- **N3 (GL2b(ii)).** `NociceptorSpec.modality` (heat | cold | mechanical | chemical) has no reader, so it is
  decorative. Either validate it (heat⇒`above`, cold⇒`below`; the modality must match the sensor's unit) and
  use it for track routing at GL3.B3, or drop it.
- **N4 (GL3.B3).** First and second pain come from two **fibre populations** with different thresholds and
  kinetics (Aδ and C), not one receptor with one threshold fanned onto two tracks. Engine ≠ track (G3) is
  decided; just add "FUNCTIONAL: one threshold for both fibre classes" to the fan-out's docstring.
- **N5 (latent_forward_model.md §6).** "`blanket.touch` vs `fire_pit.touch` … opposite sign by
  consequence" is false under the plan's own definition: both are harm-only (DNB-1). Under DNB-1's fix it is
  neutral versus negative, not opposite. Correct the pair list. A `blanket.wrap` that relieves core
  temperature would be a genuinely opposite-sign partner.

---

## What I verified (in code, at this worktree)

- `embodiment/sem.py::drive_comfort_progress`: the docstring states in-band crediting. For homeostatic
  drives it computes `|before−sp| − |after−sp|`; for entropic drives, the raw difference.
  `relief_fraction_from_progress` takes the positive part over `drive_span`. `drive_pressure` is 0 inside
  the band (homeostatic) and on the satisfied side (entropic). `drive_pain_for_value` is a level function.
- `_data/components/bodies/infant_humanoid.yaml`: `arms.thermal` (sp 0, band 0.5, pain_scale 0.4, range
  [−1,1]); `core_temperature` (band 0.25, pain_scale 1.5, initial −0.15).
- `items/cradle_fire_pit.yaml`: `warm_self` arms +0.2, with the comment "positive substrate edge"; `touch`
  +0.6. `items/cradle_blanket.yaml::touch`: arms +0.1. `items/warmth_alpha_harm.yaml` /
  `warmth_alpha_safe.yaml` / `warmth_beta_safe.yaml`: cold −0.3, arms +0.6 / +0.05.
- `bodies/minecraft_player.yaml`: `health` homeostatic (sp 20, band 6, pain_scale 0.5, range [0,40]);
  `food` entropic down (satisfaction 16, deprivation 6); the sensor list used for SF-7.
- `proprioception/pain.py`: `TISSUE_DAMAGE_DRIVES = {"drive:health"}`, `classify_pain`, `PainKind`
  (ANTICIPATORY is never pain, which matches the plan's exclusion).
- `bridges/tool_pain_bridge.py::pop_invocation_pain`: falls back from caused to felt;
  `_note_caused_pain`: NOCICEPTIVE only.
- `proprioception/pain_bus.py::create_percept_valence_subscriber` and
  `decisions/nac.py::NAc.record_percept_valence`: an accumulator with a clamp, no prediction error. The
  docstring reserves positive valence for future work.
- `embodiment/backends/minecraft.py`: world-owned sensors are synced after every action. Whether oxygen has
  recovered past 14 inside `escape_water`'s post-action evaluation window is **UNVERIFIED**. That decides
  whether the tool-path record or only the (post-fence) out-of-band record sees the surfacing relief.
- **UNVERIFIED:** whether the Minecraft bridge exposes a damage *source* (SF-1). The literature citations
  are from my own knowledge and were not checked in this session.

Plan sections read: grounding.md (all); autonomic_layer.md (all); latent_forward_model.md §4.1, §6;
DECISIONS.md 2026-10-07 entry; DESIGN_REVIEW.md; `docs/wiring/cosine-separation-is-directional.md`.
