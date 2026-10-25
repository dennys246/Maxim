# GL4–GL5 design review — BIO-FAITHFUL lens (latent_forward_model.md)

I read all of `docs/plans/latent_forward_model.md` (1,214 lines) in worktree `.worktrees/e-t8b` at origin/main
`d9dffe6e`. Beside it I read `docs/experiments/DESIGN_REVIEW.md`, `docs/plans/reviews/grounding_gl1/bio-faithful.md`
(for format, and so as not to repeat it), `docs/plans/autonomic_layer.md` §3.1–§4 (the record, alliesthesia,
G11, the §3.1.1 projection, the tier table), the G13 row and the GL5 row of `grounding.md`, the
Cerebellum/SCN sections of `docs/agents/bio-memory.md` and `docs/agents/embodiment.md`, and
`docs/wiring/body-and-word-worlds.md` §4–§5. In code I read `src/maxim/embodiment/cerebellum.py` (all of it),
`embodiment/tool_bridge.py::ModulatorAffordanceTool.execute` (the observe block),
`similarity/encoder.py::_sensor_embed` / `_normalize_value`, `decisions/temporal_credit.py::anticipatory_pre_activate`
and `docs/limits/l11_sensor_dilution.md` (the A4 small-N rows). I ran one read-only encoder probe and
`tests/unit/test_cerebellum_wiring.py` (13 passed). Owner decisions G1–G20 and the §10 decided rows are not
re-opened. Where a finding touches one, it names what the decided item consumes or omits, and puts the
option to the owner.

**Verdict: 2 DO-NOT-BUILD / 8 SHOULD-FIX / 4 NIT.**

Reading order: **D1** first. It decides whether S4 and S5 test acquired equivalence at all. Then **D2**,
which decides what S1 persists. Then SF-1 (a measured encoder defect in S4), SF-2 and SF-3 (the S5
design). The rest can be read in any order.

---

## DO-NOT-BUILD

### D1 (S4, S5): no part of the design can produce acquired equivalence. Every readout the plan names predicts C ≥ B, which is the sensed-similarity baseline the gate must beat

**Issue.** G13 makes the GL5 test acquired equivalence: Stage 1 A→O1, B→O1, C→O2, with C reading like A;
Stage 2 A→O3; the test is B inheriting O3 more than C. The biology of that effect is **outcome-mediated**.
The stimuli that predict the same outcome come to share a representation, either by mediated association
(A evokes O1 during Stage 2, so O1's representation acquires O3) or by hippocampal-region representational
compression (Gluck & Myers 1993). A **fast associative learner reads that shared representation**, so Stage-2
learning on A reaches B through it. Lesion and patient work places the transfer phase in the hippocampal
region and the entorhinal cortex (Coutureau et al. 2002; Myers, Shohamy et al. 2003: initial acquisition
spared, transfer impaired). The mechanism needs (1) a representation that becomes shared because the
outcomes are shared, (2) a representation that stays stable while Stage 2 runs, and (3) a learner keyed on
it. The plan supplies none of the three.

- **The predictor generalises in input space, not output space.** `ConsequencePredictor` is a dual-form
  kernel ridge over the context blocks (identity, object, situation, params; §4.2 item 4). When A is
  retrained, a prediction moves in proportion to that item's kernel value k(·, A) over the *inputs*. By
  G13's construction, B is sensed-unlike A and C is sensed-like A, and the names point neither way. So the
  refit moves C's prediction toward O3 more than B's, and B is anchored by its own Stage-1 pairs. That is
  the no-learning sensed-similarity baseline's ordering, C > B. §6 says only that "whether the shipped model
  can is the experiment's question". On the design as written, the answer is already fixed: it cannot.
- **The α-blended similarity (§4.2 item 6) runs the wrong way.**
  `sim(x, y) = α·cos_id + (1 − α)·cos(pred(x), pred(y))`. After the Stage-2 refit, pred(A) ≈ O3 and pred(B)
  ≈ O1, so sim(A, B) *falls*. Retraining A dissolves the equivalence instead of carrying value through it.
- **The S4 consequence node is re-derived from the same predictions.** The node is not a stable
  representation. A's predicted vector moves to O3, so A leaves the node it shared with B. Nothing in S4
  freezes Stage-1 membership, and nothing learns *on* a node. §8 lists `credit_node` as "not touched by any
  stage", and the NAc tables are keyed on (situation cluster, tool) and tool signatures (§3.4), not on
  consequence nodes.
- **S5's selection seam excludes the probe items.** The strict seam (Q5) "re-orders or annotates the
  **untried** candidates", and the "untried-tool prior applies only to untried tools" (§7 S5 blast radius).
  In acquired equivalence B and C are experienced in Stage 1 by design, so the arm's only selection effect
  never touches them. "First-contact choice after Stage 2" is not first contact for B or C.

Running S5 as written would spend a four-lens review, a harness and a cradle campaign to confirm the
baseline. A null would then be read as "consequence similarity does not transfer", when the architecture
never contained a path for it. That would be a false negative on T9's claim.

**Proposed fold. S4 must not be built until this is chosen; the owner picks, strict option first.**
- **(a) Strict: design the mediating path in S4, and front-gate it.** S4 specifies three things before it
  is built. First, the consequence representation is formed from each item's **experienced** Stage-1
  outcomes, and its membership is frozen per item at the end of Stage 1. The EC frozen-centroid mechanism
  already exists for this (§4.2 item 5). Second, a **fast outcome/value learner keyed on the consequence
  node** is updated when a member item is acted on in Stage 2. Candidates are an NAc table keyed on the
  node id, or a node→outcome running mean on the Cerebellum's delta rule. The front-gate section (§5) has
  to say why `cluster_reward_bias` keyed by (cluster, tool) cannot carry it: it is tool-specific, so B
  cannot inherit A's entry. Third, B's Stage-2 readout is read through its node membership. The
  ConsequencePredictor then keeps the job it can do: zero-shot generalisation to untried items. This
  touches the `credit_node` exclusion in §8, so §8 and S5's blast radius change with it.
- **(b) Shrink the claim.** T9 becomes "an action-conditioned predictor generalises bodily consequence to
  untried items better than the word prior and the sensed baseline". That is S2's held-out-key gate made
  behavioural, with acquired equivalence dropped from S5. This changes G13's design, so it needs the owner.
- **(c) Not recommended: run as written,** with the prereg stating that the architecture predicts C ≥ B.
  That makes S5 a test of the baseline, not of the claim.

### D2 (S1): the Cerebellum's "corrected" target is an evaluative, need-gated code that includes level terms, learned under a key that carries no internal state. That reproduces the state-mixture defect the plan itself lists, and it gives the cerebellum a job it does not do

**Issue.** S1 sets `actual =` "the consequence dims (the 6-d core plus the body's per-drive block)".
- **It includes levels.** The 6-d core is `[valence, nociception, drive_pain, relief, harm, urgency]`
  (autonomic_layer.md §3.1.1). `drive_pain` and `urgency` are levels, valence is a derived summary, and the
  per-drive block is `deviation_after ‖ drive_delta`, where `deviation_after` is a level. §4.1's own rule
  (bio SF-3 from GL1) is that the forward-model target is the **change-only** projection, with levels moved
  to context. The ConsequencePredictor is held to that rule, and the Cerebellum target in S1 breaks it. The
  Cerebellum has no context to move levels into, because its key is (entity, modulator, affordance,
  param_bucket).
- **It is need-dependent by construction.** Relief and harm are the change in `drive_pressure`. That is
  first-order alliesthesia (G11, GL1 DNB-1), which is correct for the *record*. It means the same act gives
  different values in different body states: `fire_pit.warm_self` gives relief ≈ 0.27 after two `cool_air`
  and 0 at rest. `ForwardModel.update` is a running mean (lr 0.2) per key, and Q6 recommends no situation
  in the key. The per-key mean therefore tracks **the recent history of the body's state**, not what the act
  does. That is the defect §3.1's last row names ("`escape_water` in water and on land share one mean"),
  reintroduced on purpose.
- **Biology.** A cerebellar forward model predicts the **sensory** consequence of a motor command, the
  reafference: limb state, contact, the change in a sensor (Wolpert, Miall & Kawato 1998). It is used for
  fast correction and for reafference cancellation. It does not compute how good the outcome is for the
  body's current need. That is the job of orbitofrontal, amygdalar and insular outcome expectations, and of
  devaluation-sensitive goal-directed control. Putting valence and alliesthesia-gated relief and harm into
  the Cerebellum is the caricature this lens exists to catch. The plan's own §4.2 item 3 says the
  Cerebellum is "a separate, error-driven forward-model learner", and the target S1 gives it is not one.
- **It persists.** Payload `"1.2"` writes these means under a new key. If S0a's branch (i) later resurrects
  the read side for E3, it will read state-mixed evaluative means, and fixing them needs a further version
  bump.
- **Overstated framing (see N1).** Today's target, the acted-on entity's post-action readings, is a
  legitimate forward model of how an *object* responds to manipulation. Internal models of tools and
  objects are a well-documented cerebellar job (Imamizu et al. 2000). The defect is that the *body* change
  never reaches the Cerebellum, not that the object model is wrong. S1 **replaces** the object model when
  it should **add** the body model.

**Proposed fold (plan edit to S1, §4.2 item 2 and Q6).**
- **(a) Strict.** The Cerebellum learns the **signed sensory reafference on the sufferer body**: the
  per-sensor change of the invoked affordance's declared body sensors, net of declared drift, on caused
  entries only. That is the record's `drive_delta` (the "physical description"), plus the raw sensor deltas
  for any sensors without a declared drive. To first order it is state-independent physics: touching the
  fire pit raises `arms.thermal` by 0.6 whatever the core temperature is. Clamping at range edges is the one
  remaining state dependence, and S0b already reports range-edge records separately. Keep the
  object-readings model beside it under its existing `models` key instead of deleting it. **Evaluation
  happens at read time**: apply the current `drive_pressure` to the predicted Δ (predicted relief = drop in
  pressure from now to now + Δ). That is alliesthesia applied in the right place, and it gives the
  Cerebellum one meaning across states and across worlds. With this target, Q6's "situation not in the key"
  is safe.
- **(b)** Keep the evaluative target but put the pre-state band in the key. That **reverses Q6's
  recommendation**, and Q6 is an open item, so the owner can choose it. It multiplies the keys and still
  leaves the Cerebellum valuing outcomes.
- **Either way:** drop `valence`, `drive_pain`, `urgency` and `deviation_after` from the Cerebellum target,
  and update `cerebellum.py`'s module "Bio-mapping" docstring to say what the target is now (FUNCTIONAL:
  body reafference Δ; NOT valuation). **Owner flag:** Q6's recommendation is only safe under (a).

---

## SHOULD-FIX

### SF-1 (S4): "nothing happens maps to the zero vector" is false for [0,1]-ranged dimensions. Measured: no consequence, mild harm and mild relief fall into one node at 0.85

**Issue.** §4.2 item 5 encodes the predicted core with `_sensor_embed` and the A4 gain, and claims that a
null consequence is the zero vector. The A4 weight is `(|v − 0.5|·2)^g`, where the neutral point is the
**range midpoint** (`similarity/encoder.py::_sensor_embed` docstring: "no set-point plumbing").
Relief, harm and nociception are all in [0,1], so 0 is an *extreme*, and it is encoded at full weight.
Probe (read-only, `gain_exponent=1.0`):

| encoded vector | [0,1] ranges | [−1,1] ranges |
|---|---|---|
| ‖nothing‖ | 20.2 (not zero) | 0.0 |
| cos(nothing, harm 0.2) | **0.959** | — (zero vector) |
| cos(nothing, relief 0.3) | **0.902** | — |
| cos(relief 0.3, harm 0.2) | — | −0.04 |
| cos(harm 0.2, harm 0.9) | 0.897 | 0.866 |

At the pinned 0.85, a null consequence, a mild burn and a mild relief share a node under [0,1] ranges. That
is exactly the blanket-vs-fire-pit collision S4 claims to separate. The claim holds only if the dimensions
are declared on a centred range. The plan never states the range, and "predicted core" is ambiguous between
the 6-d core (which includes the level terms D2 removes) and the change-only projection. Separately, the A4
gain is documented as a many-sensor tool that is "WRONG for a six-drive body": stability 0.62 at N=6 against
A1's 0.88 (`l11_sensor_dilution.md` §Bake-off N=6/8 rows). The consequence vector has 3 to 6 dimensions.

**Fold.** S4 names the encoded vector exactly: the change-only projection, plus the per-drive signed
`drive_delta` (see SF-2). It declares the ranges centred ([−1,1], 0 = nothing happened). It adds a guard
that a null consequence is the zero vector and that relief and harm are near-orthogonal. It either
justifies A4 at this N with a measurement, or uses the ungained embed. The 0.85 threshold was calibrated on
the world modality, so it gets its own pinned, measured value with a follow-up trigger (behaviour-tiers
rule).

### SF-2 (S4, S5): the consequence code keeps the valence class and drops the outcome's identity, and its nodes are formed at whatever need state the agent is in

**Issue.** The phasic core is three non-negative magnitudes (relief, harm, caused nociception). Clustered
by direction at 0.85, it gives roughly four classes: relief, harm, pain, and mixes. A cold-core relief and a
hunger relief land together, and so do an arm burn and a chill. That is **general affective
equivalence** (Pavlovian, valence-only). Acquired equivalence and the OFC/hippocampal outcome
representations are **outcome-specific**: they code which outcome happened, not only its sign. It also caps
G13's design at three distinguishable outcomes, O1, O2 and O3, one per axis. Separately, S4 encodes the
*predicted* code "for each (entity, affordance) the agent can act on", and under alliesthesia the
prediction depends on the current state. The same item then forms a different node when cold than when
comfortable, and the nodes multiply with the state history.

**Fold.** Encode outcome identity together with sign. Use the per-drive signed `drive_delta` in the body's
drive order (per body, as §4.1 already allows), and leave the alliesthesia-gated values out of node
formation. Value is applied at read time (D2a). That makes the representation state-independent, so one
item gets one node. If cross-body nodes are wanted, the core stays as an additional block, and the S5
prereg says which block the acquired-equivalence outcomes O1, O2 and O3 differ on.

### SF-3 (S5): "Stage 2: A alone is retrained to O3" cannot be implemented under the plan's own authoring rule or under alliesthesia

**Issue.** In authored physics an item's consequence *is* its declared `self_effect`. Retraining A to O3
means A's declaration changes between stages. The S5 authoring rule says every consequence "must follow
from each object's own `self_effect`/`target_effect` declaration, with no per-experiment tuning", and a
mid-run YAML swap is per-experiment tuning. The other lever is the body's need state, which changes
relief and harm under alliesthesia. That lever re-values *every* item acting on the same drive, so it is
outcome devaluation (a different phenomenon, read through a different structure), not A-specific
retraining.

**Fold.** The S5 prereg names the Stage-2 manipulation. The strict form is a contingency switch declared
in A's own YAML before the run, for example a declared staged effect authored blind with the rest, so that
it is part of the object's physics and not tuning. A drive-state lever is acceptable only on a drive that
A alone affects. The prereg also confirms that B's and C's declarations, and the body states at test, are
identical across Stage 1 and Stage 2.

### SF-4 (S4): predicted consequences are written into the EC as nodes, so imagined content enters memory under a label that has no definition

**Issue.** S4 forms consequence nodes from **predictions** for items the agent can act on, untried ones
included. The plan's own neighbour holds that "a prediction is not an outcome" (nociception_layer
principle 4, cited in autonomic_layer.md §3.1), and the provenance kinds include `imagined`. The S4 gate
says "consequence nodes form only from experienced-provenance predictions", but a prediction from a model
trained on experienced pairs plus discounted narrated pairs has no experienced provenance. Biologically,
writing expectations into the store that holds experienced episodes is a reality-monitoring failure: the
agent cannot later tell remembered from imagined (Johnson & Raye 1981).

**Fold.** Either form nodes only from **experienced outcome records**, as in D1(a), and keep predictions
read-time only, or stamp prediction-formed nodes `imagined` (Q3 applies) and exclude them from every
memory consumer, retro-tagging and the hivemind. Define the S4 gate wording in those terms.

### SF-5 (§4.2 item 6): α is called "precision weighting of instruction against experience" but is computed from a count

**Issue.** `α = 1 − conf(n_eff)` reuses `ForwardModel.confidence`, `1 − exp(−0.3·n)`, which does not
depend on outcome variance. A key whose consequences are inconsistent drives α to 0.05 as fast as a
consistent one. Precision weighting, which is how §1 frames α's fall, weights each source by its inverse
variance (Bayesian cue combination). The precision the plan needs is already in its model: dual-form
kernel ridge is the posterior mean of a Gaussian process, and its predictive variance
`k(x,x) − k_xᵀ(K+λI)⁻¹k_x` comes at no extra cost.

**Fold.** Set α from the ridge's predictive variance, combined with the consequence residual variance
near *x*, pinned as an innate-prior formula. Otherwise drop "precision weighting" from §1 and call α
"experience-count weighting".

### SF-6 (§4.2 item 3, S3): "complementary learning systems" is decorative. A dual-form kernel ridge refit in batch over a permanent training set is an exemplar store, closer to the hippocampus than to slow cortical learning

**Issue.** In CLS (McClelland, McNaughton & O'Reilly 1995) the neocortical learner is **parametric and
slow**, learns by interleaved replay so that new learning does not interfere with old, and lets the
episodic store go once it has consolidated. The plan's predictor keeps every training pair forever (the
persisted training set and `pid` list) and re-solves exactly at each session end. It predicts as a
kernel-weighted sum over stored exemplars. There is nothing slow about it, no interference to avoid, and
no consolidation. The CLS label implies a behavioural property (gradual, interference-resistant
generalisation) that nothing earns.

**Fold.** In §4.2 item 3 and in the S3 module docstring, describe it as "an offline batch refit over the
replay buffer; FUNCTIONAL analogy only, NOT CLS interleaved learning". Keep the session-end seam. Drop
"cortex-like" and "slow generalising" from the tier table (§4.3), or say that "slow" means *offline*.

### SF-7 (S0a): give the audit a bio criterion for the division of labour, and make Q6 wait for it

**Issue.** S0a asks only whether a shipped piece *can* carry "how far is pain" (graded Δoxygen/Δhealth by
situation). Biology splits that question:
- the cerebellum predicts short-latency sensory consequences, and the precisely timed aversive conditioned
  response at sub-second-to-about-a-second intervals (eyeblink conditioning, interpositus);
- interval timing over seconds to tens of seconds, which is what drowning in Minecraft spans, is
  cortico-striatal (Buhusi & Meck 2005), and trace or long-interval aversive anticipation depends on the
  hippocampus;
- `anticipatory_pre_activate` is oscillator and SCN-bin based, and circadian machinery does not time
  seconds-scale threat.

Branch (i) (the Cerebellum's read side becomes E3's graded predictor) and branch (iii) (the timed predictor
carries the *when*) can each be engineered, and each would put a bio name on the wrong job. Separately,
branch (i) allows "the situation cluster in its key", but Q6's recommendation ("situation: no") is asked at
S1 start, after S0a, and does not depend on the verdict.

**Fold.** Add a fourth line to S0a's gate: for whichever branch is chosen, the verdict states the bio
mapping it implies (MECHANISM / FUNCTIONAL / none) and confirms that no bio-named class is given a job
outside its mapping. A seconds-scale graded predictor is named functionally, not "cerebellar" or "SCN".
Rewrite Q6's recommendation as "per S0a's verdict and D2's target choice".

### SF-8 (§4.3): the tier table omits the oracle attributions the learner is handed

**Issue.** Three of the predictor's inputs are facts the *producer* supplies and the agent could not
perceive: which entity was acted on (`ActionContext.entity_path`, the YAML target), whether a pain was
caused or merely felt (the per-drive `caused` flag comes from the affordance's declaration, while in
biology reafference is identified by comparing against an efference copy), and the provenance label
(experienced / narrated). GL1 SF-6 re-tiered cause attribution as "engineering prior (oracle)" for exactly
this reason, and this plan's table does not carry that over. The 0.44 identity threshold and the
reference state at which predictions are encoded (SF-2) are also pinned constants with no row.

**Fold.** Add rows to §4.3: target-entity attribution, the caused/felt split and the provenance labels are
each an "engineering prior (oracle attribution); never cited as learned reafference identification or
source monitoring", each with a follow-up trigger. Add the 0.44 identity threshold and the encode
reference state as innate priors.

---

## NIT

- **N1 (§3.1, `docs/wiring/body-and-word-worlds.md` row 5).** "Learns the wrong target" should read "learns
  the object's response, never the body's". A forward model of an object's state after manipulation is a
  real cerebellar function (tool and object internal models). The correction is to *add* the body
  reafference (D2), not to discard the object model.
- **N2 (§1, §6).** Cite the mechanism as well as the paradigm: Gluck & Myers 1993 (representational
  compression of co-predictive stimuli) and Coutureau et al. 2002 (entorhinal involvement in acquired
  equivalence). They support placing the representation in the **EC**, which S4 does, and is the most
  bio-faithful choice in the plan. They also make D1's missing fast learner visible.
- **N3 (`embodiment/cerebellum.py::ForwardModel.update`).** On the first observation the mean is set to
  `actual` and the error is reported as 0, which is a one-shot copy (lr = 1). Rescorla–Wagner starts at
  V₀ = 0, so the first trial carries the largest error. The docstring's "IS the Rescorla-Wagner delta rule"
  is wrong on trial 1, and first-contact surprise, the strongest teaching signal, is lost. If S0a or E3
  ever reads the error, initialise at 0 (no consequence), and note this in the bio-mapping.
- **N4 (§4.1 "Caused, not felt").** In biology, predicting reafference serves **sensory attenuation** of
  self-caused input (Blakemore, Wolpert & Frith 1998). The plan predicts reafference and never attenuates
  anything with the prediction. That is fine for scope, but the bio-mapping should say "reafference
  prediction without attenuation", so the name claims no cancellation.

---

## Verified

- `embodiment/cerebellum.py`: `ModelKey` has four fields. `bucket_params` uses 10% of range or step 5.0.
  `ForwardModel.update` is single-cue RW at lr 0.2, and its first observation sets mean = actual with
  error 0 (N3). `ForwardModel.confidence` = `min(0.95, 1 − exp(−0.3·n))`, so α ≥ 0.05 holds.
  `Cerebellum.predict`, `observe_action_sequence`, `query_engrams` and `cleanup_program` carry Dormant
  docstrings (#909). `export_state` writes `"version": "1.1"`. `import_state` warns on an unknown version
  and loads anyway (the §3.1 claim is confirmed). The module docstring tags it "MECHANISM (single-cue delta
  rule) … NOT … cerebellar microcircuitry".
- `embodiment/tool_bridge.py::ModulatorAffordanceTool.execute`: `entity_state` is read from
  `self._entity.sensors` right after `modulator.execute`. The self_effect block comes after it.
  `observe_from_action(entity=self._entity.full_path, …, actual=entity_state, sensor_ranges=…)` builds its
  ranges from the owner's sensors, and `get_confidence` is read on the same owner key. All §3.1 claims are
  confirmed.
- `tests/unit/test_cerebellum_dormant_909.py` scans `src/` and `scripts/` (confirmed).
  `tests/unit/test_cerebellum_wiring.py`: 13 passed at this commit.
- `similarity/encoder.py::_sensor_embed`: the A4 weight is `(|v−0.5|·2)^g` with the neutral point at the
  range midpoint. `_normalize_value` maps [0,1] as the identity and [−1,1] to (v+1)/2. The SF-1 table was
  measured by direct calls (`PYTHONPATH=src`, `gain_exponent=1.0`, dim 384).
- `docs/limits/l11_sensor_dilution.md`: A4 stability 0.62 at N=6 against A1 0.88 (2026-09-03 row).
- `decisions/temporal_credit.py::TemporalCreditDistributor.anticipatory_pre_activate`: "Dormant since
  2026-05-26"; it primes eligibility for events the oscillator predicts, and takes no action input
  (confirmed).
- `autonomic_layer.md` §3.1: relief and harm = Δ`drive_pressure` (alliesthesia); the core
  `[valence, nociception, drive_pain, relief, harm, urgency]`; the per-drive `caused` flag is "this
  action's declared effect" (SF-8); the change-only projection (§3.1.1).
- `latent_forward_model.md` §8 lists `credit_node` as untouched by every stage. S5's strict seam acts on
  untried candidates only (D1).
- **UNVERIFIED:** every literature citation (Honey & Hall 1989; Gluck & Myers 1993; Coutureau et al. 2002;
  Myers, Shohamy et al. 2003; Wolpert, Miall & Kawato 1998; Imamizu et al. 2000; Buhusi & Meck 2005;
  Blakemore, Wolpert & Frith 1998; Johnson & Raye 1981; McClelland, McNaughton & O'Reilly 1995) is from my
  own knowledge and was not checked in this session. D1's "the refit moves C more than B" is an argument
  from the kernel-ridge form under G13's item construction, not a run. The fold's (a) option should carry a
  synthetic known-answer check that the shipped ridge alone gives C ≥ B, which would turn the argument into
  a measurement.
