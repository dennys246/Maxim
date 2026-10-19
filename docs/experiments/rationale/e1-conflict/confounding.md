# E1 v3 design review: confounding lens

Reviewed: e1_conflict_prereg.md DRAFT v1 (2026-10-18)

The question: does the metric isolate the claimed cause? Could a positive or a null on AVOID or WANT arise for a
reason other than "a learned situation fear and an innate need resolve a conflict"?

Method: I re-ran `learner_sim.py` unchanged and reproduced every cell of the prereg's toy table. I then extended it in
scratch copies that subclass its arms and reuse its `episode` timings, the real `NAc` and the real
`tool_dispatch.record_outcome`. The scripts are in the session scratchpad (`variants.py`, `exp_*.py`) and are not
committed. Cells are per seed: AVOID is the number of probe pains (0–3) and WANT is bread eaten (0–6). Stochastic arms
use 200–400 seeds. MAXIM and the twin are deterministic.

## What I verified (numbers)

- **The toy reproduces.** MAXIM 0/0, Q_EXPLORE AVOID 1.31 / WANT 1.79, Q_PRIOR 0.19 / 1.98, RANDOM 2.32 / 0.96 (per
  seed). The per-probe cells match the prereg table.
- **MAXIM's trajectory, traced:**
  - ep 1 (food 12): nothing scores, it idles, pain. Fear 0.5 is inert (`drive_value <= 0.5`).
  - ep 2 (food 4): eat, eat, then eat again at food 14 (need 0.2), driven only by the +0.64 causal link, which beats
    `min_confidence` 0.3. That eat fails with no bread, then the pain. Fear goes to 1.0; eat holds a positive link
    of 0.64 and a negative link of 0.64.
  - ep 3 (food 12): `flee` (wins the name-sort tie), it fails, then `escape_water` succeeds.
  - From then on, `escape_water` first in every episode. In ep 4 the scores are eat 0.7 + 0.64 − 0.32 = 1.02 against
    escape 0.7 + 0.64 = 1.34.
- **The relief weight does not decide either direction.** Over weights 0 / 0.25 / 0.5 / 1 / 2 / 5, Q_EXPLORE AVOID runs
  1.09 → 1.58 and WANT 0.35 → 2.35. MAXIM's WANT is 0 at every weight, so at weight 0 the floor still out-eats MAXIM.
- **Q₀:** Q_EXPLORE AVOID is 2.92 at Q₀ 0, 2.17 at −1, 1.31 at −2 (frozen) and 1.10 at −5. B1's frozen −2 is not a
  strawman in either direction.
- **K:** Q_EXPLORE AVOID is 1.24 / 1.31 / 1.21 / 1.05 / 0.75 at K 6 / 12 / 24 / 48 / 96.
- **ε:** AVOID 1.12 at ε 0.1, 1.39 at ε 0.3.
- **2 Hz tick quantization and a 2.0 s eat:** Q_EXPLORE AVOID 1.37–1.38. MAXIM is unchanged at 0/0.
- **A factored floor still loses AVOID.** Linear Q over [1, hunger need] on the world cluster (alpha 0.25): AVOID
  0.86–0.94, WANT 1.52–1.54 at K 12–48; probe-8 pain 0.28. The AVOID loss is not only a cold-key artifact.
- **Power under the frozen statistic** (toy per-seed distributions, MAXIM all zeros, 150 simulated 8v8 draws):
  - AVOID vs RANDOM: p ≤ 0.0125 in 100 % of draws.
  - AVOID vs Q_EXPLORE: p ≤ 0.0125 in 52 % and p ≤ 0.05 in 93 %.
  - Both AVOID tests pass Holm-over-4 in **78 %** of draws.
  - AVOID vs Q_PRIOR: 1.3 %.
  - Reverse WANT test (Q_EXPLORE eats more than MAXIM): 100 %.

## DO-NOT-BUILD

### D1. Neither DV isolates the fear: both of MAXIM's outcomes are set by the causal channel and the name sort

**The claim under attack:** "The fear is learned in two pains and generalizes across hunger; once it reaches the cap
it wins every conflict" (§The claim, stated per outcome, `AVOID_ONLY`). The prereg also says "the want's suppression
is the shipped wiring's (a fear at the cap…)".

**The alternative explanation.** In `NAc.recommend_action` the cap fear does not win anything by itself:

- `threat` × 0.7 = 0.7 exactly ties maximal `hunger` × 0.7 = 0.7 (food ≤ 6; the need clamps at 1.0).
- Among `{eat, escape_water, flee}`, `max(scores, key=(score, name))` gives every exact tie to `flee`, then
  `escape_water`. `eat` loses every exact tie.

What MAXIM does in the probes comes from the causal links that `record_outcome` books:

- **AVOID.** Turn causal booking off and keep everything else: MAXIM takes the pain on **3/3 probes**. The fear sends
  it to `flee` (name sort), `flee` fails underwater, and with no negative link it picks `flee` again until the pain.
  Drop `flee` from the roster with causal still off, and AVOID is 0 from the fear alone. So in the roster E1 plans to
  run, the avoidance is the `flee` negative link plus the `escape_water` positive link. Both links are global
  (`event_sig = tool:…`), not situation-keyed and not need-keyed.
- **WANT.** Remove every eat-failure link (eat booked only when it succeeds). MAXIM then does eat in training (ep 3:
  eat, then escape, no pain), yet still eats **0** at the probes. Escape's confidence grows with every episode
  (0.5 → 0.64 → 0.67 …) while eat's grows only in hungry episodes. The want is lost to a **repetition race in the
  causal confidence**, not to the fear's magnitude.
- **A constant policy ties MAXIM.** "`escape_water` at t = 0, always" matches MAXIM action for action on every probe:
  AVOID 0, WANT 0. That is B1's finding again ("an always-`escape_water` policy would score the same", b1 prereg §A
  finding about Exp 60 itself). H-AVOID vs Q_EXPLORE would pass for a policy with no learning and no fear.

**Fix (text for the prereg):**

- **(a) Make `causal_decisive` a freeze requirement**, not a "should weigh". Item 1a logs the argmax with
  `comp["causal"]` removed for every decision. Pre-register this rule: the AVOID result is credited to the fear only
  on probes where the first in-water pick is not `causal_decisive`. Otherwise it is recorded as "fear-gated trial
  plus a need-blind escape habit".
- **(b) Add a reported counterfactual MAXIM_NO_CAUSAL**, run offline on the logged `score_components`, or as an arm
  with `record_outcome` booking detached.
- **(c) Add two scripted reference rows, reported and not gated.** ALWAYS_ESCAPE (escape at t = 0) and
  EAT_ONCE_THEN_ESCAPE (the "resolved" anchor: 1.6 + 2.6 = 4.2 s < 6 s). If MAXIM's probe picks equal
  ALWAYS_ESCAPE's, record it as such.
- **(d) Rewrite the `AVOID_ONLY` sentence:** "the avoidance is the fear-gated defensive prior plus the escape causal
  link, which is need-blind and global; the want is lost to the causal confidence race (escape is booked every
  episode, eat only in hungry ones) and to the name-sort tie at maximal need". Delete "once it reaches the cap it
  wins every conflict".
- **(e) Carry over B1's shore check.** Count MAXIM's `escape_water` calls outside water. B1 logged 13–14 per FEAR
  seed.

### D2. MAXIM's predicted AVOID pass needs a sated training band; the claim hides that

**The claim under attack:** "avoids the pain … at hunger levels it was never trained at".

**The alternative explanation.** `escape_water` books its first success only in a post-fear episode where hunger is
inert (need ≤ 0.5, the activation floor), so `eat` cannot compete. The schedule map (MAXIM probe AVOID / WANT, same
toy):

| Training bands | MAXIM probe AVOID / WANT |
|---|---|
| (12, 4), (4, 12), (11, 3), (13, 5), (10, 4), (9, 4), (8, 4), (12, 12) | 0 / 0 |
| (4, 4), (4, 2): both hungry | **3 / 6**: MAXIM eats twice and takes the pain on every probe, losing AVOID to every floor |

The prereg's sated band [11, 13] has its lower edge exactly at need 0.5. That edge is inert only because the test is
`<=`, so food 10 inside a training episode activates hunger.

What the probes measure is therefore "the agent was once afraid while not hungry, so escape won the causal race
first". That is an order-of-first-success effect, not resolution of a conflict.

**Fix:**

- Add the schedule sweep to `learner_sim.py`, committed.
- State the dependence in the claim: "after training that includes a sated band in which the need is inert".
- Freeze which band comes first.
- Either move the sated band away from the 0.5 knife-edge, e.g. [12, 14], or pre-register the knife-edge and refuse
  any training episode whose food crosses below 11 while armed.
- Report a hungry-only training contrast as a reported arm. It is the cheapest test of whether the "resolution" is
  the fear's or the schedule's.

## SHOULD-FIX

### S1. The held-out hunger bands do not test generalization for either arm

- **MAXIM.** Its fear key is need-blind by construction: OE2 puts one world cluster at every hunger level, Wire 4
  writes the world cluster only, and the causal links are global. A held-out hunger level is the same input to the
  fear and to the links as a trained one.
- **Q_EXPLORE.** Under the toy's band key, probes [13, 15] and [1, 3] fall in trained keys ([11, 16) and [0, 6)). Only
  [7, 9] is a new key, and training partly visits it: eating at food 4 moves the agent to 9.
  - Q_EXPLORE's pain at the new key is 0.58, and 0.73 with the key frozen at the arming hunger.
  - In the trained keys it is 0.45 and 0.29.
- **The bands overlap at their edges:** train [11, 13] with probe [13, 15], and train [3, 5] with probe [1, 3].

**Fix:**

- Make the bands disjoint, e.g. probes [14, 15], [7, 9] and [1, 2].
- Drop "never trained at" and "generalizes" from the claim, or pre-register an interaction: MAXIM's AVOID advantage
  at the new-key band minus its advantage at the in-key bands, one-sided. Only that earns "generalizes".
- Report the factored linear-Q floor (it removes the cold-key part: probe-8 pain 0.28, yet AVOID 0.92 overall), so
  the AVOID win cannot be read as the floor's key choice.

### S2. The binding floor's key is unresolved (open Q3), and it decides the verdict

If the live cluster tuple does not split by hunger, Q_EXPLORE becomes Q_WORLD. Then AVOID vs Q_EXPLORE reaches
p ≤ 0.0167 in **1 %** of simulated 8v8 draws (Q_WORLD AVOID mean 0.48, mostly zero seeds), and the verdict becomes
`FLOOR_MATCHED`. If the tuple splits finer than the drive's bands, every probe is a cold cell. That is the over-fine
strawman B1 lesson 4 forbids.

**Fix (a freeze blocker):** measure the tuple's split in the pilot and freeze one rule for both cases before any
gated data. Either freeze the binding key as world cluster × the DriveSpec bands, computed from bridge `food`, with
the live tuple reported; or, if the live tuple has more than one cluster per band, merge to the bands.

### S3. The twin is an accumulator, so its tie is by construction, and the tie is fragile

"TD(0) … target −intensity per pain" updates only on pain. Nothing updates on a pain-free surfacing, so it is an
exponential accumulator like Wire 4 (with a different asymptote: 0.75 after two pains against 1.0). Two results:

- **A real every-visit TD twin beats MAXIM on WANT.** One that also updates toward 0 on a surfaced terminal (the
  extinction the prereg names as missing) scores **AVOID 0, WANT 3** (eats at food 8 and 2, never pained).
- **The as-defined tie depends on the eat-failure link.** With eat booked only when it succeeds, the as-defined twin
  goes to **AVOID 3, WANT 6** while MAXIM stays 0/0.

**Fix:** define the twin as every-visit TD(0) with γ 1: pain −intensity, surfaced 0, a per-tick 0. Predict it in the
toy and run it. Keep the accumulator only as a reported row if wanted. Rewrite H-TWIN's interpretation ("the
TD-state-value-plus-prior structure, not Wire 4's accumulator"): the arm as drafted cannot separate those.

### S4. WANT counts bread eaten on the way to drowning, and the food-14 probe cannot express a want

- **Most of the floors' WANT comes from pained probes:** 64 % of Q_EXPLORE's bread and 96 % of RANDOM's.
- **"Eat twice and survive" is knife-edge or infeasible.** Eat, eat, escape takes 5.8 s against the 6.0 s pain, and
  6.6 s under 2 Hz ticks. The top of WANT's 0–6 scale is reachable only by taking the pain.
- **At the [13, 15] probe the need is 0.1–0.3**, below the 0.5 activation floor, so MAXIM's hunger drive is inert by
  construction there. The flat +1 still pays a sated floor for eating.

**Fix:**

- Add **RESOLVED** = probes with at least one bread eaten AND surfaced (0–3 per seed). It is the claim's own
  conjunction, not a weighted payoff, so no weights choose the winner. Toy values: MAXIM 0, Q_EXPLORE 0.57,
  Q_PRIOR 1.26, RANDOM 0.03.
- Gate RESOLVED, or at least report it beside AVOID and WANT.
- Compute WANT on the need-active probes only, or make the relief term ∝ Δneed (game-native, from the DriveSpec).
  Toy: Q_EXPLORE WANT 0.85 under the Δneed relief.
- Drop "MAXIM vs RANDOM" on WANT as a gate. RANDOM's "want" is eating while drowning.

### S5. The statistic

- **The `AVOID_ONLY` text asserts a direction that no test checks.** "Forgoes food a model-free learner keeps eating"
  and "the want is suppressed" both say floor > MAXIM, but the frozen tests run only one-sided in MAXIM's favour. A
  failed WANT test also covers a tie. Pre-register the reverse one-sided test (floor > MAXIM, power 1.0 in the toy)
  as the support for the "suppressed" wording; without it the row says only "not shown".
- **Holm over 4 does not match the verdicts.** Every verdict is a conjunction within a DV. Use intersection-union
  within each DV and α/2 across the two DVs, or keep Holm and say why.
- **The predicted pass is not certain under the toy.** Under the frozen Holm both AVOID tests pass in only 78 % of
  draws, so "the toy's effects are large" overstates the AVOID-vs-Q_EXPLORE margin. State the toy's power next to the
  prediction.
- **The Q_PRIOR bound has no power.** AVOID vs Q_PRIOR has 1.3 % power, so "bounded by Q_PRIOR" is descriptive only.
  Say so.
- **MAXIM's seeds sample rig noise only.** MAXIM is deterministic, so its 8 seeds sample rig noise, not a learner's
  variance. Say so.

### S6. MAXIM acts outside the armed window; the floors do not

Floors are armed water-only and idle otherwise (B1's `ArmedPolicy`). MAXIM's `recommend_action` runs whenever the loop
does, and B1's FEAR arm logged 13–14 shore `escape_water` calls per seed. In E1 that matters three ways:

- **(a)** Extra escape successes between episodes inflate the causal race behind WANT (D1).
- **(b)** With bread in hand and an eat link of +0.64, MAXIM can eat on the shore between `prepare` and the teleport.
  It then probes with fewer bread or a food level out of band, after the read-back has passed.
- **(c)** Shore calls count toward the consecutive same-tool cap (`agent_loop.py::_MAX_CONSECUTIVE_SAME_TOOL` = 5,
  identical params), which can cap MAXIM's first in-water escape. Nothing resets the cap at arming.

**Fix:**

- Put every arm, MAXIM, the twin and ABLATED included, behind the same armed window (disarmed = idle).
- Reset the same-tool count at arming.
- Stamp bread count and food at the teleport, not at `prepare`.
- Report MAXIM's off-window executed picks per seed.

### S7. "AVOID_ONLY predicted" is a replay confirmation, and should say so

Both of MAXIM's DVs are deterministic functions of the shipped weights, the schedule and the name sort. The argmax
replay and the toy already give 0/0. The live run therefore tests whether the loop reproduces the replay, and whether
the floors eat more than zero. It does not test a learning hypothesis about MAXIM.

**Fix:** state this. Pre-register what a deviation means. For example, MAXIM eating on a probe is diagnosed from that
decision's `score_components` (timing or booking difference) before any verdict, and does not count as
`WANT_ONLY`. A row that moves to PARTIAL on an analytically fixed outcome should say "replay-confirmed".

## NIT

- **N1.** Open Q1 claims K is insensitive "over 10–30", but `learner_sim.py` has no K grid. Commit one (mine: AVOID
  1.24 / 1.31 / 1.21 / 1.05 / 0.75 at K 6 / 12 / 24 / 48 / 96).
- **N2.** A Latin square over 3 probe orders with 8 seeds cannot balance. Use 6 or 12 seeds, or report per probe
  position: online learning carries probe 1's pain into probes 2–3 for the Q arms.
- **N3.** Freeze `drive_gate_enabled` for MAXIM, the twin and ABLATED. The replay shows it moves one corner.
- **N4.** "+1 per bread read from the food delta" is ambiguous near 20 (bread at food 19 gives a delta of 1). Count
  bread from the inventory decrement, for both the relief term and WANT.
- **N5.** Q_PRIOR adds `eat` to the prior set. `floor_policies.py::PRIOR_SUFFIXES` asserts exactly two tools; update
  it and pin the new set.
- **N6.** State the exact-tie arithmetic: 0.7 against 0.7 at need 1.0, and eat sorts first so it loses every exact
  tie. With `flee` dropped, causal off and eat renamed to sort last, MAXIM eats once at food 2: the tie decides, not
  the fear. On open Q4, dropping `flee` makes the fear's own contribution isolable (causal off: AVOID 0 from the fear
  alone). Recommend dropping it, or committing to the causal-off counterfactual in D1(b).
- **N7.** Delete "the relief term is what lets a floor win WANT": at relief weight 0, Q_EXPLORE still out-eats MAXIM
  (0.35 against 0). Note also that the relief makes returns positive, so B1's Q₀ = −2 is no longer the pain's value.
  It stays non-decisive (Q₀ sweep above).
