# E1 v3 design review: bio-faithful lens

Reviewed: e1_conflict_prereg.md DRAFT v1 (2026-10-18)

The question: does E1 v3 test what an approach–avoidance mechanism is for, or a caricature of it?

Short answer: the AVOID half tests the real thing. The WANT half, as written, does not. It credits eating at any
hunger level, and its mechanism twin is not a TD learner in an executed-choice design. Both are fixable on paper.
Once fixed, the predicted `AVOID_ONLY` becomes a sharper test than the draft claims: it tests the no-extinction
decision (M73) directly.

## What I verified

- `decisions/nac.py::record_cluster_fear`: `V ← clamp(V − 0.5·intensity, [−1, 0])`. It is an accumulator, not a
  prediction error, and nothing un-learns it (`NACConfig` comment; brief invariant "Wire 4 … (a)").
- `nac.py::anticipatory_threat_need` returns the fear when it is ≥ θ = 0.5. `recommend_action` then skips any
  drive ≤ 0.5 (`if drive_value <= 0.5: continue`). So a fear of exactly 0.5 does nothing, and the threshold that
  actually applies is > 0.5. The NACConfig comment and the bio-memory brief both say "keep θ ABOVE that floor", but
  the default makes θ equal to the floor.
- `recommend_action` scoring: the prior adds `need × 0.7` per affinity hit. The causal term is `+best_pos −
  0.5·best_neg`, and it is state-blind. `learned_bias` is the summed `cluster_reward_bias`. Ties go to the higher
  name (`max(scores, key=(score, t))`). On a fresh NAc with hunger 1.0 and threat 1.0, `eat`, `flee` and
  `escape_water` all score 0.70 and the pick is `minecraft_player_flee` (measured).
- `runtime/substrate_proposal.py::_read_drive_states`:
  - hunger need = `(16 − food)/10`, clipped to [0, 1]. Food 14 gives 0.2, food 12 gives 0.4, food 8 gives 0.8,
    food ≤ 6 gives 1.0.
  - `health→threat` is `min(1, |dev|)` in raw hp, so it jumps to 1.0 below 14 hp.
  - Oxygen has no innate need, by design (body YAML comment on `escape_water`).
  - The fear need is max-combined into `threat` after the encode.
- `proprioception/pain_bus.py::create_pain_cluster_fear_subscriber` writes to the WORLD cluster only, for
  intensity ≥ 0.3.
- Body `minecraft_player.yaml`:
  - Oxygen pain is `min(1, (|oxy − 20| − 6)·0.5)`: 0.5 at 13 bubbles, 1.0 at 12. Exp 60 recorded 20 `drive:oxygen`
    signals per 10 episodes, so each pained episode delivers a 0.5 pain and then a 1.0 pain.
  - `saturation` has no drive. It lives in the world modality only.
  - `food` has a drive and is in the world modality, so it is in the world key and also in interoception.
- `runtime/tool_dispatch.py::record_outcome`: a successful `eat` with a measured food rise books
  `update_cluster_reward(+1, source="drive_relief")` on the interoception cluster. Every other success books the
  tool-success floor there (α 0.15, cap ±1). The measured-relief path is gated on `modality: world` (brief,
  measured-relief invariant).
- `embodiment/sem.py::drive_comfort_progress` for an entropic "down" drive is `after − before`. Any food rise counts
  as relief at any hunger, so the body has no satiety term.
- The toy (`learner_sim.py`):
  - Its MAXIM passes `current_clusters={"world": …}` only, and calls `record_outcome` without `clusters` or
    `drive_potential_diff`. So the interoception `learned_bias` channel is absent.
  - Each pained episode writes one 1.0 pain.
  - The twin updates only on pain.
- I re-ran the toy with variants. Scratch scripts are in
  `/private/tmp/claude-501/-Users-dennyschaedig-Scripts-Maxim/15239c88-ef6e-47b7-8925-0c2ae097dbd4/scratchpad/`:
  `bio_probe.py`, `bio_probe2.py` and `bio_probe3.py`. They import the committed toy unchanged. Results are cited
  inline below.
- Read: DESIGN_REVIEW.md, the bio-memory brief (Wire 4, measured-relief and drive-prior invariants),
  baseline_floors.md §Why, baseline-floors-b1.md, the B1 prereg §mechanism, roadmap_1_4 §Phase 2/5/§Pressure,
  substrate-learning-channels.md, and fear_learning.md (deferred; §Extinction).

## DO-NOT-BUILD

### D1. The "TD twin" is not TD in this design, so H-TWIN's "tie expected" is an artifact of the toy

**The claim attacked:** "TWIN: … TD(0) state value … (α 0.5, target −intensity per pain)" and "H-TWIN: a tie is
the expected result (the twin is Wire 4 with a TD write)".

**Why this is wrong for E1:**

- Under B1's propose-only training every episode ended in the pain. There, a TD state value on the cluster
  really was Wire 4 (baseline_floors.md §Why).
- E1 executes its choices, so the agent has pain-free surfacings: the water state ends at a terminal value of 0.
  A TD(0) or Rescorla–Wagner critic updates on every visit, so a pain-free surfacing moves V toward 0. That is
  extinction, and it is the one property that separates a prediction-error learner from Wire 4's accumulator.
- The twin as drafted updates only on pain. That is Wire 4 with a different asymptote, not TD.
- The bio standard here is the actor–critic account of avoidance (Maia 2010, *Cogn Affect Behav Neurosci*). Fear
  (the critic) declines as avoidance (the actor) becomes reliable, while the avoidance persists. This matches the
  data showing fear falls in well-trained avoidance (Kamin, Brimer & Black 1963; Mineka 1979).

**Measured (toy, deterministic):**

| Twin variant | Probes at food 14 / 8 / 2 | WANT | AVOID |
|---|---|---|---|
| As drafted | 0 eats, no pain | 0/6 | 0/3 |
| + `V ← V + 0.5(0 − V)` on a pain-free surfaced terminal | 0 / 1 / 2 eats, no pain | 3/6 | 0/3 |

- With the extinction update, avoidance passes to the state-blind `escape_water` link: the actor. Eating becomes
  graded by hunger. That is the textbook resolution, and it beats Q_EXPLORE on both DVs in the toy (Q_EXPLORE takes
  the pain on 29–58 % of probes and eats 1.8 bread in total).
- Under the live pain schedule (a 0.5 pain then a 1.0 pain per pained episode), the drafted pain-only twin no
  longer ties Wire 4. It eats 2/2/2 and takes the pain on 3/3 probes, because its value tends toward a mix of the
  two intensities and never reaches Wire 4's cap. MAXIM stays at 0/0 (`bio_probe3.py`).
- So the drafted twin's result depends on the pain schedule, not on the mechanism.

**Fix (prereg text):** "TWIN: a TD(0) critic on the same world cluster. At each episode terminal on that cluster,
`V ← V + 0.5(target − V)`, where target = −(the episode's saturating pain intensity) if the pain landed, else 0
(surfaced). The need is −V when it is > 0.5, and the actor (causal links, prior, cluster bias) is MAXIM's
unchanged. This is the actor–critic account of avoidance. H-TWIN (reported): MAXIM vs TWIN isolates EXTINCTION,
the one thing Wire 4 lacks by design (M73). The predicted direction, from the toy, is TWIN ≥ MAXIM on WANT and
TWIN = MAXIM on AVOID." Also freeze the twin's update unit (per episode or per pain signal). The toy shows the
choice decides the outcome.

### D2. WANT credits eating whether or not the agent is hungry, but a need's job in a conflict is to make the choice depend on hunger

**The claim attacked:** "WANT per seed = bread eaten across the 3 probes (0–6)", with H-WANT gated against
Q_EXPLORE and RANDOM. Also "the agent … expresses the want … at hunger levels it was never trained at".

**Why:** This is the charter's own named failure ("the credit's job is state-conditioned selection, not
eat-sooner").

- In an approach–avoidance conflict, the drive's signature is a dose–response: a hungrier animal accepts more
  risk to get the food. Miller's raised approach gradient; food-under-threat tasks (Choi & Kim 2010); hunger
  (AgRP) neurons suppressing competing fear and anxiety states (Padilla et al. 2016; Burnett et al. 2016).
- substrate-learning-channels.md says the same: measure "P(eat | hungry) vs P(eat | satiated)".
- The body declares the probe at food 14 as need 0.2, below the > 0.5 activation floor, and the train band
  [11, 13] as need 0.3–0.5, at or below the floor. MAXIM correctly has no want there.
- The floors' +1 per bread does not depend on hunger. So eating at food 14 scores for a floor (Q_EXPLORE 0.41,
  Q_RAW 0.50 in the toy) and counts as "want" against MAXIM.

**Fix (prereg text):**

- "WANT (gated) = bread eaten on the two HUNGRY probes (food 8 and 2; need 0.8 and 1.0), 0–4."
- "WANT_CONTINGENCY (gated, or at least reported with a pre-stated direction) = eaten(food 2) − eaten(food 14).
  Eating is credited as the want only where the body declares a need."
- "Food 14 is the SATED control. Eating there is reported, never credited."
- In §Protocol, state that train band [11, 13] is below MAXIM's activation floor. Half of MAXIM's training
  episodes are sated by the body's own declaration, and the floors' reward does not know that.

## SHOULD-FIX

### S1. "No relief store" is wrong as written

**The claim attacked:** "the want's suppression is the shipped wiring's (… no relief store)" and, in the task
framing, "MAXIM gets a causal link on eat success, no relief magnitude".

**Evidence:** `tool_dispatch.record_outcome` books a signed relief credit, `update_cluster_reward(+1,
source="drive_relief")`, on the interoception cluster from the measured food delta (break 2). It also books the
tool-success floor for `escape_water`. The toy drops this channel: no `clusters` reach `record_outcome`, and only
`world` reaches `recommend_action`.

I added it back, with an interoception stand-in of hunger band × oxygen bin and `drive_potential_diff` on a
successful `eat`. It changed 0 probe cells (`bio_probe.py` variant B). That is consistent with the R2 finding that
this credit is a messenger, not a cause.

**On fairness:** the floors' +1 per bread and MAXIM's credit are the same hunger-blind sign, because
`drive_comfort_progress` for "down" drives has no satiety. So the relief comparison is at parity. Both sides
disagree with alliesthesia (Cabanac 1971) and with homeostatic RL's drive-reduction reward (Keramati & Gutkin
2014). That is a shared limit, so disclose it rather than fix it.

The real asymmetry is elsewhere. The floors also get B1's graded per-step oxygen cost, which is a "how far is the
pain" signal that MAXIM's fear does not have (fear is constant across the dive).

**Fix:**

1. The toy passes `clusters` and `drive_potential_diff` exactly as the live path does.
2. Prereg text: "MAXIM's relief credit exists, but it is sign-only, keyed to interoception, ≤ 0.15 per booking and
   a messenger (substrate-learning-channels.md). The missing piece is a world-keyed relief store (roadmap Phase 5).
   The floors' oxygen cost is a graded pre-pain signal MAXIM lacks (Phase 5 graded predictor). Both are disclosed
   as the floors' structural advantages."

### S2. The predicted failure names its mechanism vaguely, and the AVOID claim text overstates what the fear does

**What the wiring can express.** A trade-off inside one dive is possible: eat while the need beats the fear, and
leave once a bread lowers the need. At fear 0.75 and food ≤ 6, eat scores 0.70 against escape 0.525. After one
bread (+5 food) the need is 0.5–0.7, so eat drops to 0.35–0.49 and escape wins. But this only works while the fear
sits in (0.5, 1.0).

- The accumulator crosses that window in one or two pained episodes and never comes back, because nothing
  extinguishes it.
- At the cap the readout ties: a full need and a capped fear both score 0.70, and the alphabet picks `flee`
  (measured).
- After training, eat's causal links cancel (pos 0.641, neg 0.641) and `escape_water` holds pos 0.816 (`bio_probe2.py`).

**So the missing mechanisms, in order of the evidence:**

1. Extinction (M73; the safety map in fear_learning.md §Extinction, option 2b). The D1 twin shows that adding it
   alone flips WANT.
2. A readout where hunger can scale down how strongly the fear acts (the AgRP evidence), instead of a fixed
   additive sum with equal 0.7 weights and a name tie-break.
3. State-blind causal links.

The relief store is not what blocks WANT here (S1).

**Overstated:** "the fear … generalizes across hunger; once it reaches the cap it wins every conflict". With fear
removed at the probes (the counterfactual), `escape_water` still wins at food 14 on `causal_pos` 0.82 alone. The
fear decides the choice only at food 8 and food 2. That is B1's OD4 finding (habit after the first escape) again.

**Fix:** make the `fear_decisive` and `causal_decisive` counterfactuals at each probe's first selection REQUIRED
reported fields. Word the AVOID_ONLY claim per probe: "fear-decisive at need ≥ 0.8; the escape habit carries the
sated probe". Replace "no graded trade-off, no extinction and no relief store" with the ordered list above, and
cite fear_learning.md as the named Phase 5 consumer.

### S3. The toy's fear dynamics do not match the live pain schedule

**Evidence:** live, a pained episode delivers 0.5 then 1.0 (body pain formula; Exp 60's 20 signals per 10
episodes). So Wire 4 is at 0.75 after ONE pained episode, which is active, and at the cap after two. The toy writes
one 1.0 pain per episode, so after one episode the fear is 0.5, which is inert.

This leaves MAXIM's probes unchanged (0/0). It changes the learning curve (MAXIM eats 1 bread at T3 under the pair)
and flips the twin (D1).

**Fix:** the toy models the pain pair, or whatever the pilot measures, and the prereg freezes the per-signal
intensities used for the twin and for any fear-strength reporting. The MAXIM sanity gate should count pain
signals, not pained episodes.

### S4. OE2 is faithful on the memory side, but it is a half-measure forced by a code conflation, and the reason should be on record

**What is right:** keying the fear of a place on a hunger-blind situation is right. Internal state changes how
strongly a fear acts, not which place it is about. The missing modulation belongs at readout (S2 item 2).

**What OE2 actually does:**

- It deletes `saturation` from the substrate entirely. It has no drive, so it was never in interoception. It is not
  moved anywhere.
- `food` stays in the world key (0.915 at food 2; food 0 misses at 0.842). The faithful geometry ("food and
  saturation both off": 1.000 hold, 0.787 separation) was not taken, presumably because the measured-relief credit
  is gated on `modality: world`. Moving `food` off the world modality would silently turn off MAXIM's relief credit.
- For B1's tuple key, the interoception entry is unchanged: it carries food, hunger and oxygen. So the floors' key
  can split by hunger band AND by oxygen level within a dive. That gives the floors the time-in-dive axis that
  Wire 4's world-only fear lacks (S1).

**Fix (prereg text):** "OE2 drops saturation from the world modality only. Food stays there because the
measured-relief gate keys on `modality: world`. The modality tag doing two jobs is recorded as a body artifact,
with a follow-up issue." Add a refusal: read-back food < 1 refuses the episode, which guards the [1, 3] band edge
against the food-0 miss.

### S5. `flee`: keep it, and disclose that the alphabet decides every capped tie

**Why keep it:** dropping `flee` from the variant's roster would fit the roster to the task in MAXIM's favour.
Bolles's SSDR hierarchy (a defense that fails is suppressed and the next one emerges) is roughly what the negative
`flee` link does.

**What to disclose:** the first act under a capped fear, and the result of the full-hunger-vs-capped-fear conflict,
are both decided by name order, not by any mechanism. The replay's "renaming changes nothing" holds only because a
shared prefix keeps that order.

**Fix:** keep `flee` (answering OE-open 4). Report per decision whether the winning score was tied, and count how
many probe decisions ties settled.

## NIT

- **N1. θ equals the activation floor.** The bio-memory brief's Wire 4 invariant says θ "must stay ABOVE" the
  0.5 floor; the default is equal to it. In the prereg, write "the fear acts only when it is > 0.5". The
  code-versus-invariant mismatch belongs in a follow-up issue, not in E1.
- **N2. `health→threat` is a step to 1.0 below 14 hp** (raw-hp `min(1, |dev|)`). It never acts in a scored E1
  episode, because any `drive:health` pain refuses the arm-seed. Say so in the prereg. Minecraft's starvation
  damage (food 0) is outside every band; the S4 guard keeps it there.
- **N3. Bio naming.** The fear store sits on NAc in code. Biologically, contextual fear runs hippocampus → BLA,
  and arbitration under reward-and-threat runs through NAc shell and mPFC (Burgos-Robles et al. 2017). The prereg
  makes no amygdala claim and should not add one. No rename (CLAUDE.md).
- **N4. Feasibility budget.** Two breads × 1.6 s plus the 2.6 s escape is 5.8 s, against a saturating pain at
  about 6 s in the toy, with the 0.5 pain earlier. At 2 breads, "both" is a 0.2 s race rather than a trade-off.
  The environment lens should measure it. The prereg should keep an eat-one-then-leave region reachable, which
  the D1 twin shows (0/1/2 graded by hunger).
