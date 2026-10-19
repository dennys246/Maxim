# E1 (DRAFT v2, 2026-10-18) — the want against the fear, learned at the food: does the conflict resolve, over floors that can win?

> **STATUS: DRAFT v2, paper only. Not frozen; no harness exists.** Session E of 1.3.3
> ([roadmap_1_3_x.md](../plans/roadmap_1_3_x.md) §1.3.3). v1 went through the four-lens design review on 2026-10-18
> (reports verbatim in [rationale/e1-conflict/](rationale/e1-conflict/)). Every lens returned DO-NOT-BUILD (8 in all),
> and v2 folds them (§What the review changed). **v1's predicted outcome is WITHDRAWN:** the review showed it rested
> on a toy that omitted a live credit channel. v2 makes an offline composed-loop replay a freeze prerequisite.
> Owner decisions OE1–OE4 taken 2026-10-18. Phase 0's `src/` (items 1a, 2, 4, and the arming gate below) is fenced until 1.3.6. Parent plan:
> [roadmap_1_4.md](../plans/roadmap_1_4.md) §Phase 2, §Release thresholds T1/T3. Floors:
> [baseline_floors.md](../plans/baseline_floors.md) (BF2, §Why). Lessons: [baseline-floors-b1.md](../lessons/baseline-floors-b1.md).

## Why

E1 v2 (the roadmap's) read the first choice of an agent placed submerged at the food, hungry, with a fear present.
Its argmax is a function of designer constants, so a same-prior floor learner matches it and E1 could not EARN under
BF2. **Owner decision 2026-10-16:** redesign E1 so a reasonable no-bio floor CAN lose, never by construction in
either direction. **OE1 (2026-10-18)** chose the redesign: a learning curve at the food, with separately gated DVs
and floors that can win either one.

## What the replays and the review found

**The argmax replay** ([rationale/e1-conflict/e1_argmax_replay.py](rationale/e1-conflict/e1_argmax_replay.py), committed with its json): on the real
`recommend_action`, the v2 crossing holds only for a fresh agent. A fear of 0.75 loses iff food ≤ 8; a cap fear (1.0)
never loses, until one successful `eat` (+0.5 causal) beats it at food ≤ 10. Under fear, `flee` wins every exact tie
by name sort (fear at the cap scores 0.70, the same as a full hunger need). The review adds what the replay left
out: it passes no `clusters` to `record_outcome`, so the **interoception cluster-reward channel** (±0.15 per event,
`update_cluster_reward`; the interoception cluster splits hunger at about food 9/10) is absent. The replay is a
fresh-agent and causal-history map, not a prediction of the loop.

**The hunger-state cosine replay** ([rationale/e1-conflict/e1_hunger_state_cosine_replay.py](rationale/e1-conflict/e1_hunger_state_cosine_replay.py)):
on the shipped body every game-native hungry probe misses a fed-trained fear (0.850 → 0.784, food 14 → 2) and hungry
shore/water merge (0.851–0.872); with `saturation` off the world modality the key holds to food 2 (≥ 0.915) and
shore/water separate (0.788–0.822). This stands and is OE2's evidence. `food` stays in the world modality, and a
world-keyed fear still holds across food 14 → 2 under OE2. Why `food` stays is to be recorded at the freeze (the
bio lens infers the relief credit is gated on `modality: world`; not yet verified).

**The v1 toy** ([rationale/e1-conflict/learner_sim.py](rationale/e1-conflict/learner_sim.py)) predicted "MAXIM avoids
and never eats". The review falsified that as a prediction:
- with the cluster-reward channel booked, MAXIM eats one bread at food 8 and two at food 2, pain-free (wiring DNB-1);
  the result then flips with the number of loop iterations outside each episode (≤ 16 eats; ≥ 24 decays back);
- MAXIM's AVOID pass needed a sated training band so `escape_water` books its first success before `eat` can
  compete; with both training bands hungry MAXIM takes the pain on every probe (confounding D2);
- a proper TD twin (every-visit, toward 0 on a pain-free surfacing: extinction) eats and avoids, so v1's "tie
  expected" was the toy's accumulator (bio D1, confounding S3);
- the live body publishes two oxygen pains per pained episode (0.5 at oxygen 13, 1.0 at 12), not one (bio S3,
  wiring SF-1).

The toy stays committed as v1's floor-can-lose evidence (both directions were live and stay so). Its numbers are not
a prediction.

## What the review changed (v1 → v2)

| v1 | Problem (lens) | v2 |
|---|---|---|
| "AVOID_ONLY predicted" | Rests on a toy without the cluster-reward channel, a sated band, one pain per episode, and no loop timing (wiring DNB-1, conf D2, bio S3) | **Withdrawn.** A composed-loop offline replay on `run_minecraft_aut` + a scripted food world produces the prediction before the freeze (§Freeze prerequisites); the run then tests whether the loop reproduces it, and says so (conf S7) |
| Neither DV isolates the fear | Causal links and the name sort decide; an "always escape at t = 0" policy ties MAXIM (conf D1) | Scripted anchors ALWAYS_ESCAPE and EAT_ONCE_THEN_ESCAPE run as reported rows; `causal_decisive` and `learned_decisive` counterfactuals join item 1a's `drive_decisive`; the claim text names which channel decided, from provenance |
| Training alternates a sated band | The AVOID result is an order-of-first-success effect of that band (conf D2) | The training schedule is a declared factor (OE3): the binding schedule and a reported contrast schedule are both frozen |
| WANT = bread eaten | Counts eating while drowning (64 % of Q_EXPLORE's) and eating when sated (bio D2, conf S4) | **RESOLVED** (ate ≥ 1 bread AND surfaced before the saturating pain) is the second gated DV, on the HUNGRY probes only; WANT reported; the sated probe is a control, and eaten(hungry) − eaten(sated) is reported |
| Held-out hunger bands | Not held out for any arm: MAXIM's fear key is need-blind (OE2); the interoception cluster and Q's key reuse trained clusters (conf S1, wiring SF-6) | "Generalization" leaves the claim. Probes are at FOOD LEVELS; their keys are recorded per probe |
| Twin = pain-only accumulator | Not TD; its tie was by construction (bio D1) | TD(0) with extinction, through a no-`src/` seam: a swapped Wire-4 pain subscriber writes `record_cluster_fear` so the store tracks V, updated at every episode end (pain → −1, surfaced → 0) |
| Pain terminal by harness oxygen read | Drive pain is published only at proposal time, so a rescue mid-`eat` can skip Wire 4's write (env DNB-1) | Every arm's episode ends on the bus publish of the saturating oxygen pain (intensity ≥ 1.0); only eats that RETURNED before it count |
| Hunger setter unspecified | The only setter in the repo needs 3.5–5 min per episode (≈ 70 h) (env DNB-2) | Two-stage drain (hunger amplifier 255, then 19) to exact point targets, RCON read-back, a difficulty preflight; bands avoid food ≤ 2 (starvation overshoot) |
| MAXIM unarmed | It acts on the shore and eats the bread before submersion; its executed-training loop has never run (wiring DNB-2, conf S6) | A typed arming gate in `src/` (behind the fence, with item 1a); MAXIM acts only while armed, like the floors |
| "Two breads fit" | Two breads + escape = 6.25–6.5 s against the pain at ≈ 6.0 s (env SF-1) | The pain-free ceiling is one bread; RESOLVED is defined accordingly; two are held so the choice to keep eating exists |
| Holm over 4, one direction | No test backs "suppressed"; conjunctive verdicts (conf S5) | Intersection-union within each DV, α/2 across DVs, and a pre-registered REVERSE test (floor > MAXIM) for any "suppressed" wording |
| Unpinned run config | `drive_gate_enabled` from the shared `~/.maxim`; `loop_hz` and window schedule free (wiring SF-7) | Frozen in the campaign fingerprint: `drive_gate_enabled` false, explore weight 0.0, `loop_hz` 2, the window schedule and the loop iterations outside episodes |

## Owner decisions

- **2026-10-16:** redesign E1 so a reasonable no-bio floor can lose (roadmap §Phase 2 banner).
- **OE1 (2026-10-18):** a learning curve at the food, two separately gated DVs, floors `Q_EXPLORE` (binding
  `tabular_q`) and `RANDOM` (binding), the twin RUN, others reported, no weighted payoff. v2 keeps OE1 and changes
  the second DV from WANT to RESOLVED (OE4).
- **OE2 (2026-10-18):** the variant body drops `saturation` from the world modality. `extends:` can express this only as
  `saturation: {modality: null}` (env SF-4, wiring SF-8): the bridge still serves it, and a read tool returning it
  remains. The rescue's satiation check moves to an RCON read-back, and E1 gets its own encoding record.
- **OE3 (2026-10-18), the binding training schedule: hungry only** (food 4 and 8 alternating), the strict option;
  the mixed sated-first schedule runs as MAXIM_CONTRAST, reported.
- **OE4 (2026-10-18): RESOLVED is the second gated DV** (WANT reported).

## The claim and the hypotheses

**Claim tested.** An agent that learned a situation fear from its own drowning pain, placed hungry in that situation
with food in hand, resolves the conflict: it avoids the saturating pain (AVOID) and, when hungry, eats and still
surfaces (RESOLVED), better than a model-free learner given the same executed experience.

- **H-AVOID (gated):** MAXIM's probe pain count is below Q_EXPLORE's AND below RANDOM's.
- **H-RESOLVED (gated):** MAXIM's resolved hungry probes exceed Q_EXPLORE's AND RANDOM's.
- **H-REVERSE (gated only for wording):** floor > MAXIM on RESOLVED, one-sided; only its pass licenses
  "the want is suppressed".
- **Reported:** the twin, ABLATED, the contrast schedule, Q_RAW, Q_PRIOR (descriptive: AVOID vs Q_PRIOR has ≈ 1 % power
  in the v1 toy), Q_WORLD, ALWAYS_ESCAPE, EAT_ONCE_THEN_ESCAPE; WANT; eaten(hungry) − eaten(sated); the decisive
  channel per decision (causal / learned / drive / fear) from item 1a's counterfactuals.

**Prediction:** none until the composed-loop replay runs (§Freeze prerequisites). The frozen prereg states it with the
replay's numbers and the channel the replay says decides. MAXIM is deterministic given its inputs, so its seeds sample
rig noise, not a learner's variance (stated beside every MAXIM result).

## Arms

Every arm: the Exp 60 pool on the E1 variant body (OE2), the substrate-primary loop at 2 Hz, the same episodes,
probes, refusals and arming. Floors run through B1's `SubstratePolicy` hook (#1236) and read no NAc.

| Arm | Policy | Role |
|---|---|---|
| MAXIM | `recommend_action`, Wire 4 attached, armed like the floors | The claim |
| MAXIM_CONTRAST | MAXIM on the mixed sated-first schedule (OE3) | Reported: the schedule's effect |
| TWIN | MAXIM with the Wire-4 subscriber swapped for TD(0) with extinction (above) | Mechanism twin, RUN |
| ABLATED | MAXIM, fear subscriber detached | Wire 4's contribution |
| ALWAYS_ESCAPE | Scripted: `escape_water` at the first armed select | Anchor: what the causal race alone can buy |
| EAT_ONCE_THEN_ESCAPE | Scripted: one `eat`, then `escape_water` | Anchor: the RESOLVED ceiling on this apparatus |
| Q_EXPLORE | Tabular Q on B1's key (the tuple of active modality clusters), full repertoire, B1's constants | **Binding `tabular_q`** |
| RANDOM | Uniform over the stamped tools | **Binding `random`** |
| Q_RAW | Q on raw sensors (is_in_water × hunger band × B1's oxygen bins) | Reported |
| Q_PRIOR | Q_RAW over `eat`, `flee`, `escape_water` | Reported |
| Q_WORLD | Q on the world cluster alone | Reported |

**Q_EXPLORE's key is frozen as B1's.** Whether it splits the hunger bands live is recorded (key coverage per probe),
never used to choose a floor after data (conf S2). The floors' code changes are additive: new parameters or classes
beside B1's, which stays byte-identical, since B1's committed verdict depends on it (wiring SF-4). `PRIOR_SUFFIXES`
gains `_eat` only in E1's policy, not in B1's.

**The floors' reward:** B1's oxygen cost (rule 2), plus +1 per bread whose `eat` call returned before the terminal
(read from the bridge's `food` delta and the inventory count), terminal 0 at surfacing and B1's −2 at the saturating
pain. `drive:food` deprivation pains (food ≤ 6) are recorded and excluded from the floors' reward. The +1 weight is
frozen; the review found it never changes MAXIM's outcome over weights 0–5.

## Protocol

- **Body:** `bodies/minecraft_e1` (`extends: bodies/minecraft_player`, `saturation: {modality: null}`); its tool
  signatures are renamed `minecraft_e1_*` (no tie moves among body tools, replay); `flee` kept (it wins capped ties
  by name sort and fails underwater; the count of name-sort-decided ties is reported). The R3 A+C variant baseline is
  frozen on it before the pilot.
- **Hunger:** two-stage drain to an exact food target, read back over RCON; the episode is refused before arming if
  read-back food is off target by more than 1. Targets: training per OE3; probes food 8 and food 4 (hungry, scored),
  food 14 (sated control; need 0.2, below the 0.5 activation floor). Starvation never reached.
- **Bread:** `clear`, then `give` 2, then a read-back, at every arming.
- **Episode:** teleport submerged at Exp 60's depth, arm, run. Terminals: surfaced (`is_in_water` 0); the saturating
  oxygen pain published on the bus (intensity ≥ 1.0); a cap. RCON rescue, bridge `stop`, loop stop. Any `drive:health`
  pain refuses the arm-seed. Both oxygen publishes (0.5, 1.0) and every `drive:*` publish are recorded.
- **Training:** K = 12 executed episodes, alternating food 4 and food 8 (OE3); MAXIM_CONTRAST alternates food 13 and food 4, sated first. **Probes:** food 8, 4, 14, order a balanced Latin
  square (12 seeds make the square balance); every arm keeps learning through the probes.
- **Seeds:** 12 per stochastic arm; scripted and deterministic arms 12 each (rig noise). Arm order: a Williams design.
  Fresh throwaway persistence per arm-seed. Two `--resume` sessions (the campaign is ≈ 8–9 h of rig at amplifier 255).

## DVs, gates and statistics

- **AVOID** per seed = probes (of 3) that ended in the saturating pain.
- **RESOLVED** per seed = hungry probes (of 2) with ≥ 1 bread eaten (call returned) AND surfaced before the pain.
- **WANT** (reported) = bread eaten across the hungry probes; **SATED_EAT** (reported) = bread eaten on the food-14 probe.
- **Tests:** exact permutation tests, one-sided, per seed. Within each DV the two binding floors form an
  intersection-union test; the two DVs at α = 0.025 each. H-REVERSE at 0.05, only for wording.
- **Sanity gates:** MAXIM's fear must form in training (a saturating pain), else INCOMPLETE; the per-probe NODE gate
  (each probe's world cluster is the trained one, read loop-off as in Exp 62).
- **Verdicts:** `FLOORS_BEATEN` (both DVs over both floors); `AVOID_ONLY`; `RESOLVED_ONLY`; `FLOOR_MATCHED`;
  `INCOMPLETE`.

## The claim, stated per outcome (in advance)

- `FLOORS_BEATEN`: "a learned situation fear and an innate need resolve a conflict at the food: the agent avoids the
  pain and, when hungry, still eats, better than a model-free learner with the same experience", with the channel
  that decided (from provenance) and the twin's and anchors' contrasts. EARNED requires every frozen gate.
- `AVOID_ONLY`: `PARTIAL (narrow: avoidance only)`; "the want is suppressed" only if H-REVERSE passes.
- `RESOLVED_ONLY`, `FLOOR_MATCHED`: recorded as such, with the curves.
- Whatever the verdict, if an anchor (ALWAYS_ESCAPE or EAT_ONCE_THEN_ESCAPE) matches MAXIM on a DV, the claim text
  says the DV was carried by that action order, not by the conflict.

## Freeze prerequisites

1. **The composed-loop offline replay** (the toy-before-rig rule, upgraded): MAXIM, the twin and the floors through
   `run_minecraft_aut` against a scripted water world with food and `eat` (`ScriptedWaterBridge` extended: it serves
   food 20 and answers `did eat` today), the frozen loop timing, both oxygen publishes, and every live booking
   (causal links, interoception cluster reward, temporal credit, decay). It produces the prediction, a power figure
   for each gate, and the schedule sweep. If it shows a floor that cannot lose or a MAXIM that cannot lose on a gated
   DV, the design returns to review.
2. **Item 1a** with `drive_by_need`, `drive_decisive`, and the two new counterfactuals `causal_decisive` and
   `learned_decisive`; **item 1b's `NAc_CREDIT` event** promoted into E1's instrument (the credit that decided must
   be readable at the write seam).
3. **The arming gate** (typed, `src/`), so MAXIM acts only while armed.
4. **Item 2** (campaign core) with E1's harness as a consumer; item 5's learning-curve rows; per-probe interoception
   cluster and bias reads; loop iterations per window.
5. The hunger setter, bread handling, the variant body and its encoding record, the RCON satiation read-back.

## Ledger triggers this work fires (wiring SF-9)

| Change | Fires |
|---|---|
| Item 1a's recommend-event fields | T1-7 and Exp 45 by their wording (discharged by the byte-identical-selection guard) |
| The arming gate | T1-6; T1-13 → T1-14/T1-15 if it lives in the loop's idle gate |
| The TD twin, if it ever moves into `src/` | T1-13/14/15 (the no-`src/` subscriber seam fires none) |
| Edits to `water_trial.py` or `scripted_water.py` | T1-14/T1-15's regression guards |

## Not claimed

Depth (one cluster by design); a graded trade-off (no shipped mechanism grades the fear by oxygen); generalization
to untrained states (v2 tests food levels, not held-out keys); anything on the LLM path; any comparison with Exp
60/61/62's rows.

## Open decisions

None open (OE1–OE4 taken 2026-10-18). The freeze waits on §Freeze prerequisites, and the frozen version gets its own
design-review round on the delta (the composed replay's prediction).
