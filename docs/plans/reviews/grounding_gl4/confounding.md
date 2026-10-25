# GL4–GL5 design review — CONFOUNDING lens (latent_forward_model.md)

Reviewed: `docs/plans/latent_forward_model.md` in full (1,214 lines) at the `e-t8b` worktree, origin/main
`d9dffe6e`. Read alongside: `docs/experiments/DESIGN_REVIEW.md`, `docs/plans/grounding.md` (§Owner decisions
G1–G20 and the GL4/GL5 rows), the GL1 confounding review (`reviews/grounding_gl1/confounding.md`, for format and
for what DNB-2/DNB-3 asked), `docs/plans/baseline_floors.md` (BF2, the B1 lessons, "a contingency a reasonable floor
can lose"), `docs/experiments/grounding_census.md`, `docs/wiring/` (substrate-learning-channels,
cosine-separation-is-directional, instrument-band-statistic-matches-window), and `roadmap_1_4.md`'s GL5/T9 rows.
Code claims were checked at `file::symbol` (§Verified). I ran one numpy toy of the plan's predictor
(`/private/tmp/claude-501/lfm_toy/toy.py`): fixed cosine-kernel ridge in dual form, three blocks, the §6 A/B/C
design. G1–G20 and the §10 decided items are not re-opened; where a finding touches one (G13 above all), it is
about whether the design as written can deliver what the decision says, and the fold is plan text or an owner
choice.

The question for every stage: could a positive, or a null, come from something other than the claimed cause, and
can the mechanism lose to a floor that could reasonably win?

**Verdict: 3 DO-NOT-BUILD / 9 SHOULD-FIX / 6 NIT.**

Reading order: D1 → D2 (both S5, the claim) → D3 (S2, which can stop the line) → S1 (half of T9 has no test) →
S2/S3/S4 (S0b, the stop gate) → the rest.

- DO-NOT-BUILD **S5**: the shipped model cannot produce the acquired-equivalence ordering through either readout the
  plan names. A fixed-kernel ridge gives C ≥ B by linear algebra (toy confirms). So a positive can only come from an
  artifact: α's experience-density asymmetry, situation-block leakage from the trial schedule, or refit timing.
- DO-NOT-BUILD **S5**: there is no floor that could reasonably win. In the AE test every item is experienced, so
  generalisation is never exercised. A per-key observed-outcome table (the S1 Cerebellum plus outcome clustering)
  is the reasonable floor and is not an arm. BF2's `tabular_q`/`random` are missing too. And "beats the sensed
  baseline by a margin" is a strawman statistic, because a learner that transfers nothing beats a baseline that
  predicts C > B.
- DO-NOT-BUILD **S2**: the held-out collision/convergence gate has no admissible way to pass. 24 of the 34 cradle
  keys come from items with no sensors (object block = zero vector), and `false_hearth`'s sensors are byte-identical
  to `fire_pit`'s. Zero-shot prediction on a held-out collision key can only come from sensed features, and the
  sensed-baseline clause forbids exactly that route. The gate either fails by construction, and a FAIL "stops" the
  line, or passes on ridge-vs-1-NN model-class differences.

---

## DO-NOT-BUILD

### D1 (S5, §6, §4.2 items 5–6): the named readouts cannot produce B > C, so any positive is an artifact

**Issue.** §6: "A fixed-kernel ridge over sensed features cannot produce this ordering by itself; whether the
shipped model can is the experiment's question." For the readouts the plan names, that is not an empirical
question. It can be computed on paper:

1. **Predictor output (§4.2 item 4).** Kernel ridge with fixed per-block cosine kernels predicts
   f(x) = k(x, X)(K + λI)⁻¹Y. Retraining A to O3 changes f(B) only through B's kernel row to A's contexts. By
   design, B is sensed-unlike A and the names are neutral, while C is sensed-like A. So Δf(C) > Δf(B), which is the
   same ordering as the no-learning sensed baseline. Toy (neutral situation block): Δf(B)[O3] = −0.005,
   Δf(C)[O3] = +0.297. The forward model has no learned representation in which A and B become alike because they
   share O1. That shared representation is what acquired equivalence requires (mediated generalisation through the
   common outcome; Honey & Hall 1989).
2. **The S4 consequence node over the predicted core.** After the Stage-2 refit, pred(A) ≈ O3 and pred(B) ≈ O1,
   so A leaves the O1 node. B can inherit only if Stage-2 value is booked on the node A occupied **before** the
   refit. That is a credit write onto a consequence node. The plan never specifies it, and §8 lists `credit_node` as
   untouched. Since the refit runs only at `BioStack.on_session_end` (§4.2 item 3), whether the effect exists
   depends on where the session boundary falls relative to Stage 2 and the test. That is an apparatus variable, not
   the mechanism.
3. **The α-blended similarity (§4.2 item 6).** sim(x, y) = α·cos_id + (1 − α)·cos(pred x, pred y), with
   α = 1 − conf(n_eff) and n_eff "the kernel-weighted count of training contexts near x and y". C is sensed-like
   A by design, so Stage 2's extra A trials raise n_eff near C more than near B. α(A,C) therefore falls further
   than α(A,B). With neutral names, cos_id is high and roughly equal for both pairs, and after the refit
   cos(pred A, pred B) ≈ cos(pred A, pred C) ≈ 0 (O3 vs O1, O3 vs O2). So sim(A,B) > sim(A,C) **purely because
   B kept more of the identity prior**. That is "B inherits more" produced by experience density, with zero
   consequence equivalence.
4. **Situation-block leakage from the schedule.** The context carries a `situation` block, the body state at
   decision time, which reflects the previous action's consequence. In a blocked Stage-1 schedule, items that share
   O1 are acted on from "after-O1" states. When the probe for each item is read in its habitual state, the toy
   flips the ordering: with block weights (1, 1, 3) and λ = 1, Δf(B) = +0.198 and Δf(C) = +0.102. With equal
   weights it does not flip (+0.033 / +0.041). So a "pass" is available from schedule × block weight × λ, all
   pinned "innate prior" constants, through co-occurrence in a shared situation. That is the co-activation binding
   §2 disowns.

The plan therefore has no stated route by which the mechanism under test produces the predicted result, and three
unstated routes by which artifacts do.

**Why it matters.** A T9 positive would be uninterpretable: it would need to be traced to α density, schedule
leakage or session-boundary timing before it could be credited. A null would be uninformative, because it was
predictable from the algebra. This is the "toy model before rig time" lesson from B1
(`docs/lessons/baseline-floors-b1.md`, cited in baseline_floors.md), and the E1/B4 rule that toys must be the
**composed** loop.

**Fold (plan text in §6/§7 S5, before S5's prereg; asked at S5 start).**
- **Name the mediating mechanism in the plan, not the prereg.** State which component carries Stage-2 value from A
  to B: for example, credit booked on the consequence node that A's **pre-refit** prediction occupies, with the
  refit schedule pinned. If that component is a credit write, it is a new selection or credit surface and goes
  through Q5's seam analysis. If the plan cannot name one, say plainly that the shipped predictor predicts no
  acquired equivalence and re-scope S5 (owner options, strict first: (a) a design whose prediction the mechanism
  actually makes, with the AE test kept as a control the model is expected to fail; (b) add a learned
  representation, e.g. a learned target encoder or MLP hidden layer, as a declared mechanism with its own
  front-gate answer; (c) shrink T9).
- **A composed-loop toy before the prereg freezes:** the S3 refit, the S4 node, the α read and the S5 selection
  seam on a scripted A/B/C world, every arm of D2. It must show that the mechanism predicts B > C and that every
  floor does not. Commit it as `docs/plans/rationale/latent-forward-model/s5_toy.md`.
- **Remove the artifact routes:** reset the body to a frozen pre-state band before every Stage-1 and Stage-2
  trial, and probe B and C in the same state. Interleave and counterbalance the trial order. Use equal Stage-1
  trial counts per item. Pin the session and refit boundaries identically across arms. Do not use α as a gated
  readout: either drop it, or equalise n_eff for B and C and report α per pair. Report a situation-block ablation
  beside the gate.

### D2 (S5, BF2): no reasonable floor that could win; the gate statistic is set against a baseline that is wrong by construction

**Issue.** Three parts.

1. **The AE test never exercises the forward model's distinguishing capacity.** §7 S5: "In the acquired-equivalence
   test every item is experienced by design." Each probe item's prediction is then close to its own observed
   outcome mean. A **per-key observed-outcome table**, which is exactly what S1 makes the Cerebellum hold ("its
   per-key consequence means", §4.3), clustered by the same S4 consequence modality and carrying the same Stage-2
   value booking as D1's mediating path, gives the same consequence representation for A, B and C with no
   generalising model. If it matches the full arm, T9's result belongs to the per-key table plus outcome clustering,
   not to `ConsequencePredictor`. This is baseline_floors.md's mechanism twin ("the mechanism twin … is RUN as an
   explicit arm"), and it is the floor that can reasonably win. It is absent from the comparator list (sensed
   baseline, identity-only, yoked no-predictor).
2. **BF2 is not mentioned.** `grep -n "BF2\|tabular_q\|floor"` over the plan finds only S0b's data floors. T9 is a
   NEW `[behavioral]` claim on the substrate path, so under BF2 it "cannot be EARNED unless it beats `tabular_q`
   and `random` on the same state keys". Otherwise it takes `PARTIAL (narrow: floor matched)`. S5 declares neither.
   Note that `tabular_q` on (EC cluster × tool) predicts **no** inheritance (B = C, because the tools are distinct),
   which is the natural null floor for the B − C statistic.
3. **"Beats the sensed baseline by a margin" is a strawman statistic.** §6: the sensed baseline "predicts C > B. The
   model passes only by reversing it." S2's wording, carried into S5, is "beats the sensed-similarity baseline by a
   pre-registered margin". A learner that transfers nothing (B = C, e.g. `tabular_q` or the yoked no-predictor arm)
   beats a baseline that predicts C − B = d > 0 by d. The baseline's error is in the direction against the
   hypothesis, so "beating" it is cheap. The statistic has to be matched to the null floor: (B − C) > 0 with its
   interval excluding zero, against the twin and `tabular_q`, not a difference from a signed-wrong baseline.
   The sensed baseline is also partly degenerate. `similarity/encoder.py::_sensor_embed` sums SHA bases **per
   sensor name**, so two items with disjoint sensor names are near-orthogonal whatever their values. "Sensed-unlike B"
   can then be achieved by schema alone, and the baseline spreads exactly zero to B. An empty sensor dict returns the
   zero vector, which has undefined cosine.

**Why it matters.** As designed, an S5 PASS can be earned by beating a baseline that cannot win, with the floor that
could win never run. That is the strawman-in-our-favour that baseline_floors.md tells this lens to refuse.

**Fold (plan §6, §7 S5 arms and gate, §10 table; asked at S5 start).**
- Add arms: **the per-key-outcome twin** (S1 Cerebellum means + S4 clustering + the same Stage-2 booking, no
  `ConsequencePredictor`), **`tabular_q`** and **`random`** on the same state keys, with exploration declared in
  the prereg. Keep the sensed baseline and identity-only arm as reported comparators.
- Gate statistic: per dimension, Δ_B − Δ_C (Stage-1 end → Stage-2 end) > a pre-registered margin with its interval
  excluding 0, **and** greater than the twin's Δ_B − Δ_C. A twin match is recorded as
  `PARTIAL (narrow: floor matched)`, the BF2 outcome, and the prereg must say so in advance.
- The AE item set shares **one sensor schema** (same sensor names, values differ), so object-block cosine measures
  readings, not schema. No item has an empty object block. The prereg tabulates cos_obj(A,B), cos_obj(A,C) and
  cos_id for every triple.
- Owner decision (strict first): (a) T9 requires beating the twin, so the claim is about the generalising predictor;
  (b) T9 is re-worded as a claim about the consequence representation (per-key outcomes + clustering), the twin
  becomes the mechanism, and `ConsequencePredictor` is not credited by S5 (T4a Dormant marking applies).

### D3 (S2): the held-out dissociation gate has no admissible route to PASS on the shipped cradle data

**Issue.** S2 gates on "the held-out collision and convergence keys" with held-out splits "by key and by entity":
full-model sign accuracy ≥ 0.8, ≥ 0.2 above identity-only, **and** above the no-learning sensed baseline. It also
refuses the result if the sensed baseline itself reaches ≥ 0.8. The shipped data cannot support this:

- **Most consequence-bearing keys have no object block.** The twelve items `green_/purple_hearth`,
  `green_/purple_flame`, their `_b` twins and `warmth_alpha/beta_safe/harm` have no `sensors:` at all (checked over
  `_data/components/items/*.yaml`). Their modulator is `abstract: true # capability-only — no sensors (C4)`. That is
  12 items × (`warm_self`, `touch`) = **24 of S0b's 34 keys**, and their object block is the zero vector
  (`_sensor_embed`: "Empty dict short-circuits to the zero vector"). For these keys the full model has only the
  identity block, the situation (body state, shared across items) and empty params. So full ≈ identity-only by
  construction, and the 0.2 margin over identity-only cannot be reached. The sensed baseline is undefined there
  (cosine of zero vectors).
- **The one shipped hidden-property pair has identical sensors.** `cradle_false_hearth.yaml` is "modelled
  byte-for-byte on cradle_fire_pit.yaml EXCEPT warm_self.self_effect" (`heat_output` 0.9, `fuel` 0.8 in both), yet
  `warm_self` is safe on the fire pit (`arms.thermal` +0.2) and harmful on the false hearth (+0.6). Held out, neither
  the full model nor the sensed baseline can predict it.
- **The general point.** A held-out collision key is by definition one whose name misleads. A zero-shot prediction
  for it can come only from pre-experience features, meaning the object block. G13 says sensed-feature
  generalisation is not the claim, and S2 adds a clause forbidding it. So there is no admissible route to PASS. The
  gate FAILs because of how the YAML was authored. "FAIL means ridge cannot ground … try the MLP once, or stop" then
  stops or diverts the line for a reason unrelated to ridge's capacity. Where it does pass, the margin over the
  sensed baseline is a model-class difference (multi-neighbour regularised regression vs nearest-neighbour spread on
  the same features), not information about consequence. That is a statistic mismatched to its baseline.

**Why it matters.** S2 is the gate between S0b and the `src/` stages. A gate that cannot pass for the claimed reason,
and can pass only for an unclaimed one, is in the vacuous-guard class (here, unfailable for the wrong reason).

**Fold (plan §7 S2; S2 start).**
- Re-scope S2 as **capability**, stated as such. The gates become: (i) within-key held-out **invocations** (split by
  invocation *blocks* within a key and band, with the target standardisation and λ fitted on training folds only)
  beat the context-copy baseline on phasic dimensions. This answers "does it learn per-key consequences and not copy
  state". (ii) Across keys, the **matched-model-class** comparison: the full ridge against identity-only ridge and
  object-only ridge, using the same λ procedure. Gate only on keys with a non-empty object block. Report the
  sensorless keys separately.
- The sensed-baseline comparison on shipped keys is **reported, not gated**. Shipped YAML carries one
  hidden-property pair, so it cannot test G13. The G13 test is S5's authored set (D2).
- A S2 FAIL on the dissociation keys does not stop the line by itself. It stops it only if (i) fails. Record this in
  §9's divergence row.

---

## SHOULD-FIX

### S1 (S5 / T9 wording): the "blocks it where names collide" half of T9 has no gated test

**Issue.** The T9 claim (grounding.md GL5 row; roadmap row 498): "consequence similarity transfers learned value
between affordances whose names differ **and blocks it between affordances whose names collide**." The AE design
deliberately uses names "that point neither way". C is sensed-like, not name-like, so the test covers the transfer
half and a "blocks where *sensed* alike" half. The name-collision half appears only in the jet triad (demoted to an
ungated control) and in S4's offline separation check ("reported, not claimed"). G13 says "T9 keeps its claim";
under this design half the claim has no measurement that could earn it.

**Fold (owner, S5 start; strict first).** (a) Add an **acquired-distinctiveness arm** with colliding names: two
items with the same affordance name, near-identical identity blocks, equal sensed readings, and differing
consequences. Gate: after experience, value learned on one does not transfer to the other, beyond identity-only
and the D2 twin. The shipped `false_hearth`/`fire_pit` `warm_self` pair and the census's same-name flips are
templates. (b) Re-word T9 to the half the design tests.

### S2 (S0b): under the plan's own non-null, per-dimension rules, the shipped collision count is ~0, so the stop floor fails by definition

**Issue.** Collisions require "opposite non-null sign in the same band", read per dimension (`nociception`, `harm`,
`relief`), with null-consequence keys excluded. The census (`grounding_census.md` headline) finds that **every**
primary-walk harm flip is a same-name **neutral-vs-harmful** pair, and the cross-name collisions are all azimuth
turns that are "opposite signs, both harmful". Per target dimension:
- `blanket.touch` (`arms.thermal` +0.1, inside the band) is null on every phasic dimension at rest, so it is
  excluded.
- `warm_self` safe vs harm: same sign on `relief` (both relieve `cold`), harm 0 vs harm > 0. That is null vs
  non-null on `harm`, not opposite signs.
- `turn_left`/`turn_right`: same sign on `harm`. The opposite sign is on the azimuth *sensor*, which is not a
  target dimension.

So the count is plausibly **zero** before any capture, and the plan hands the definition to the prereg ("decides …
whether a neutral-versus-negative pair counts"), **after** the census has shown which definition yields a non-zero
count. A definition chosen after seeing its count is the post-hoc class (`feedback_literal_vs_structural_pre_registration`).
**Statistic mismatch, too:** the census measures affordance-name decomposition cosine (`touch↔touch` = 1.000), while
S0b counts "identity-block" cosine of `"<affordance> <entity name>"` (§4.1). The floors are "set against GL1's
census prediction" of a different statistic.

**Fold (plan §7 S0b, now).** Freeze the collision definition in the plan: per dimension, a pair is a collision when
one key is non-null and harmful, and the other is null or of opposite sign, **on that dimension**, provided the
union is non-null. Report "opposite non-null sign" as a sub-count. Have the census (or the prereg's own
re-computation) predict both counts **on the identity-block statistic S0b uses**, before the capture.

### S3 (S0b): pair counts are not independent; one prolific key can meet a floor

**Issue.** Collisions and convergences are counted as **pairs** of keys. k mutually convergent keys yield
k(k−1)/2 pairs. The 12 warmth/hearth/flame items share one effect template (`cold −0.3`, `arms.thermal` +0.05 or
+0.6), so a handful of templates can produce dozens of "convergences" whose shared consequence is also shared
`warm_self`/`touch` affordance names, which pushes identity cosine up. The floor measures template multiplicity,
not grounding evidence.

**Fold.** Count and gate on the **distinct keys participating**, and on the number of connected components of the
dissociation graph. Report pair counts beside them. Collapse effect-template duplicates (the `_b` and α/β twins)
into one unit for counting.

### S4 (S0b, S2 data): who chooses the actions in the "scripted-substrate" capture is unspecified, and learned avoidance confounds coverage

**Issue.** If substrate-primary selection picks the actions, the agent avoids harmful keys after the first burn
(fear, causal links). Per-key n and pre-state-band coverage then correlate with each key's consequence. Harmful keys
are sampled rarely and in early bands, and repeated safe keys stack (§7 S0b already notes `warm_self` stacking).
"Key order … counterbalanced" implies a forced schedule. The capture runtime is a substrate-primary AUT
(`--aut-mode substrate-primary`), and whether the fixture orchestrator can force a schedule is UNVERIFIED.

**Fold.** The prereg specifies a **forced, counterbalanced schedule** (Latin square over keys), with the body reset
to a frozen pre-state band before each invocation. Agent-chosen invocations are captured but reported separately and
never counted toward the floors. If the named runtime cannot force a schedule, say so and count per-key n and band
coverage stratified by first-burn time.

### S5 (S2): the context-copy baseline is ill-defined, and standardisation leaks the held-out keys

**Issue.** (i) "A model that sees the context but not the action": the context's `identity` block **is** the action
(`"<affordance> <entity name>"`), and the `object` block identifies the entity. A context-copy model that keeps them
nearly knows the key and will match the full model, so it is not the level-copy baseline bio SF-3 asked for.
(ii) The target standardisation is "computed once from S0b's capture and pinned", i.e. over held-out keys too.

**Fold.** Define context-copy as **situation block only** (the levels), and add a **key-mean** baseline (the training
mean per key, which is what the Cerebellum would say). Fit the standardisation, like λ, on training folds only, and
pin the procedure, not the numbers, before S2 reads data.

### S6 (S5): no control for drift, decay or refit between Stage 1 and test; no counterbalancing; pseudo-replication

**Issue.** The readout is each probe item's change from Stage-1 end to Stage-2 end. Over that interval other
channels move without any Stage-2 learning: causal-link decay, consolidation and session-end refits, `cleanup_session`
(which S3 newly runs on `--sim`). These may move B and C differently because O1 and O2 differ in valence and
magnitude. Item identity (B's and C's particular readings) is confounded with role unless roles rotate. The predictor
is closed-form deterministic and the schedule is scripted, so repeated "seeds" on one authored triple are not
independent replicates.

**Fold.** Add a **no-revaluation control** (Stage 2 retrains A to its own O1, or no Stage 2, with the same elapsed
passes and session boundaries). Gate on (Δ_B − Δ_C)_revaluation − (Δ_B − Δ_C)_control. Rotate each authored item
through the A/B/C roles across replicates, and counterbalance which outcome is O1 vs O2. The **unit of replication is
the authored triple**, and n is stated in triples in the prereg. Choose O3 on a dimension where B and C start at the
same level (no floor or ceiling for C if O2 is already harmful).

### S7 (S4, S5): the standardised target turns "nothing happened" into a shared non-zero direction

**Issue.** §4.1: the target encoder is the change-only vector "plus a fixed per-dimension standardisation". §4.2
item 5: "nothing happens" maps to the zero vector and forms no node. After standardisation, a zero change maps to
−μ/σ, which is non-zero whenever the capture's mean change is non-zero (it will be: most invocations relieve
`cold`). All null-consequence items then share one direction and form one consequence node, so null items acquire
"equivalence" with each other: the null/null pairing GL1 DNB-3 excluded, reintroduced in the representation.
Separately, cosine is directional (`docs/wiring/cosine-separation-is-directional.md`), so O1 and O3 that differ in
magnitude only would share a node, and A would never leave it.

**Fold.** The S4 modality encodes the **raw** (unstandardised) predicted change, or standardises without centring.
State that a zero change is the zero vector after any transform. In S5, O1, O2 and O3 differ in **direction** (sign
pattern across dimensions) and none is null. Record cos(O1,O3), cos(O1,O2) and cos(O2,O3) in the prereg.

### S8 (S5): the name-neutrality and keyword controls need a numeric criterion and must cover the affordance part of the tool name

**Issue.** "Names are chosen so that neither the identity block nor the drive-keyword path points to the shared
consequence" has no numeric test. A blind author naming items by their sensed properties (a "warm rug", a "fire
stone") makes the names track the readings. Two A/B names sharing a 0.44 text node let #1181's widening spread
credit along the word path. Tool names are `<entity>_<affordance>` (`embodiment/tool_bridge.py`), and
`NAc.recommend_action`'s Component 3 matches keywords anywhere in the tool name (`decisions/nac.py`,
`_DRIVE_TOOL_AFFINITIES`: `cold`/`thermal` → `warm`, `fire`, …). So a `warm_self` affordance on an AE item fires
the bonus whatever the entity is called.

**Fold.** Freeze per triple: cos_id(B,A) ≤ cos_id(C,A) (conservative: the word prior favours C), no A/B shared text
node at 0.44 on the production encoder (checked under the `MAXIM_SUBSTRATE_PATH=1` encode), and option (a)'s keyword
test run over the **full tool name** (which rules out `warm_self` as an AE affordance).

### S9 (S0b, Minecraft): the action contrast cannot fail for any reason the line cares about

**Issue.** Δoxygen(`escape_water`) − Δoxygen(no-op) within a band, measured over a window that includes the refill,
is guaranteed non-zero by game physics: surfacing refills and idling underwater drains. "A non-zero, pre-registered
difference" therefore tests only that the producer and the join ran. That is fine for a data-existence gate, but the
plan lets this half stand in as "the bridge to E3" under S0a branch (ii).

**Fold.** Label it a data-existence check. Pre-register its sign and a minimum magnitude per band, and state that
the graded "how far is pain" readout is a predictor property, tested in S5's Minecraft arm against its own floor
(the B1 mechanism twin: TD state value on the cluster + the same prior), not inferred from S0b's contrast.

---

## NIT

- **N1 (S4).** "`turn_left`/`turn_right` and the `touch` ×16 collisions separate by consequence on the S0b data" is
  an in-sample check on trained keys. Separation there follows from the per-key means, so it is a tautology. Say so,
  or run it on held-out invocation blocks.
- **N2 (S5).** "First-contact choice after Stage 2" contradicts "every item is experienced". Rename it (e.g. "first
  B-vs-C choice after Stage 2") and define it relative to the Stage-1 choice baseline.
- **N3 (S5).** Retraining A to O3 swaps A's declared physics mid-experiment. That is a harness intervention, and the
  blind-author rule ("no per-experiment tuning") must cover it: O3 is authored with the item set, before the design
  is known, and A's sensors and name are unchanged across stages.
- **N4 (§4.3).** Block weights and λ are pinned as innate priors. The toy shows the sign of Δ_B − Δ_C under schedule
  leakage depends on them. Pin them before S0b's data is read, and report a sensitivity row beside the S5 gate.
- **N5 (S2).** Hold out by **twin group** (`green_hearth` + `green_hearth_b`, `warmth_alpha_*` + `warmth_beta_*`),
  not only by entity. A twin in training makes a same-effect held-out key trivially predictable and inflates the
  "no worse than identity-only elsewhere" row.
- **N6 (§4.1).** "Felt-only pain is reported, never trained on" leaves the felt-only invocations in the training set
  with nociception 0. That labels them "no pain", which biases the predictor toward under-predicting pain in
  situations where pain co-occurs. Either drop felt-only invocations from the nociception target (mask the
  dimension) or say that the 0 label is intended.

---

## Verified

- `similarity/encoder.py::_sensor_embed`: per-sensor-name SHA bases (`_stable_basis(name, …)`), and an empty dict
  returns the zero vector (docstring and code). This is the basis for D2 (schema-driven object cosine) and D3
  (sensorless object blocks).
- `_data/components/items/*.yaml`: `green_/purple_hearth`, `green_/purple_flame`, their `_b` twins and
  `warmth_alpha/beta_safe/harm` have no `sensors:` (grep count 0). `cradle_false_hearth.yaml` declares sensors
  identical to `cradle_fire_pit.yaml` (`heat_output` 0.9, `fuel` 0.8). Fire pit `warm_self` is `arms.thermal` +0.2
  and false hearth `warm_self` is +0.6. `cradle_blanket.yaml` `touch` is `arms.thermal` +0.1 and `wrap` is
  `core_temperature` +0.1. `warmth_alpha_safe`/`_harm` differ only in `arms.thermal` (0.05 vs 0.6).
- `decisions/nac.py::_DRIVE_TOOL_AFFINITIES` (`cold`/`thermal` → `warm`, `fire`, `blanket`, `huddle`; `thirst` →
  `water`) and `NAc.recommend_action`'s substring match (`drive_lower in tool_lower`; `keyword in tool_lower` →
  `drive_value * 0.7`). `runtime/substrate_proposal.py::_DRIVE_CORRECTIVE_NEEDS` maps `temp`/`thermal` → `cold`.
  Tool names are `f"{ent.name}_{aff_name}"` (`embodiment/tool_bridge.py`).
- `embodiment/cerebellum.py::ForwardModel.confidence`: `1 − exp(−0.3·n)`, capped at 0.95 (the α curve).
- `bridges/tool_pain_bridge.py::ToolPainBridge.pop_invocation_pain`: returns caused pain, else felt pain.
- `runtime/executor.py::Executor._drive_relief`: a drive moved away from comfort records 0.0 (docstring).
- `embodiment/tool_bridge.py::ModulatorAffordanceTool.execute`: `entity_state` is built before the
  `_apply_sensor_deltas` calls, and `observe_from_action(… actual=entity_state)` is called (the §3.1 defect stands).
- `similarity/ec.py::EntorhinalCortex._matrix_for` exists (per-modality matrices).
- `docs/experiments/grounding_census.md` headline: 10/10 harm flips are same-name; the cross-name collisions are the
  azimuth turns, "opposite signs, both harmful". `scripts/grounding_census.py` encodes affordance names through
  `_make_aff_encoder(...).encode_decomposed`, not the `"<affordance> <entity name>"` identity block (S2).
- Toy (`/private/tmp/claude-501/lfm_toy/toy.py`, numpy, fixed cosine-kernel ridge, dual form): with a neutral
  situation, Δf(B)[O3] = −0.005 and Δf(C)[O3] = +0.297. With a blocked schedule and habitual-state probes, the sign
  depends on block weights and λ: (1,1,1), λ = 0.1 gives +0.033 / +0.041; (1,1,3), λ = 1 gives +0.198 / +0.102. A
  toy, not the composed loop. It shows only that the ordering is fixed by the algebra and by the schedule, not by
  consequence learning.
- UNVERIFIED: whether the fixture orchestrator can force an invocation schedule for a substrate-primary AUT (S4);
  whether `StrengthStrategy`-type retention or causal-link decay moves B and C differently over Stage 2 (S6, argued
  structurally, not measured); the census's counts under the S2 definition proposed here.
