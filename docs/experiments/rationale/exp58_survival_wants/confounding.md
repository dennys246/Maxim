# Exp 58 — CONFOUNDING lens (four-lens design review, 2026-09-14)

Reviewed: `docs/experiments/exp58_survival_wants_prereg.md` (DRAFT, no harness, no mechanism).
Consulted: `docs/wiring/substrate-learning-channels.md`, `docs/wiring/pain-needs-declared-failure-modes.md`,
`docs/experiments/r2_learned_bias_v2_prereg.md` §Outcome, `src/maxim/decisions/nac.py::recommend_action`
(+ `_DRIVE_TOOL_AFFINITIES`, `NACConfig.substrate_explore_bonus_weight`),
`src/maxim/runtime/agent_loop.py::_read_drive_states` / `_DRIVE_CORRECTIVE_NEEDS`,
`src/maxim/embodiment/sem.py::corrective_need_intensity`, `src/maxim/runtime/gating.py` (Wire-2 read side).

Charter question: does the metric isolate the claimed cause? Could a positive OR a null arise for a
reason other than the claim?

---

## DO-NOT-BUILD (as drafted — each is fixable, but the drafted gates would return a false verdict)

### D1. Claim B's primary DV `P(enter dark)` is likely NOT expressible by the designed read path — a structural null-confound, and the falsifier would misname the break

**The mechanism reads valence off the ACTIVE cluster** ("when an ACTIVE cluster carries valence
≤ −θ, contribute a threat need"). The world cluster is the encoding of the CURRENT situation:
while the agent stands in the LIT area deciding whether to enter the dark, the active world
cluster is the LIT one (valence ≈ 0 by the specificity gate itself) — the dark cluster only
becomes active once the agent is already inside. So the designed read path produces **escape**
(threat need fires inside the dark → shortened visits → lower fraction-of-time-in-dark) but has
no channel to produce **entry-avoidance** (`P(enter dark)` at the threshold, decided from a
lit-cluster state). Unless the encoder happens to flip to the dark cluster in the boundary/approach
region (light decays gradually near the opening — an unmeasured instrument property), the primary
gate `P(enter dark) ≤ 0.5 × baseline` nulls **while the mechanism works exactly as built**, and the
prereg's falsifier ("the valence write does not reach behaviour") would name the wrong break — the
write and the read both work; the DV asked for a prediction the architecture doesn't make.

Second half of the same break: the threat need is a **scalar, direction-free** boost. Its read path
is `_DRIVE_CORRECTIVE_NEEDS: health→threat` → `_DRIVE_TOOL_AFFINITIES["threat"] = (flee, hide,
retreat, escape, withdraw, defend, shelter)` — keyword name-matches on tool names. Avoidance is
only expressible if the repertoire contains a tool whose NAME matches one of those keywords AND
whose params/semantics move the agent away from dark specifically. A generic `move_to` matches no
threat keyword (verified against the table); `mine_block`/`turn` don't either. As drafted the
prereg never names the avoidance affordance, so the behavioural endpoint may be unreachable from
the score vector regardless of valence.

**Fix shape (three parts, all pre-build, all cheap):**
1. **Boundary-activation probe** added to the instrument check: sweep agent positions from lit →
   entrance → interior and record which world cluster is active at each; this settles empirically
   whether `P(enter)` is a valid DV or whether the primary must be fraction-of-time-in-dark /
   escape-latency (which the escape dynamics CAN drive). Pick the primary from the measurement,
   before freeze.
2. **Name the avoidance affordance in the prereg** (tool name + how the threat keywords reach it +
   how directionality (toward lit) gets into its params) — or add a `retreat_to_light`-class
   affordance to the classroom repertoire as a declared apparatus element.
3. **Extend the instrument check to the READ path**: with a hand-set valence −cap on the dark
   cluster and the dark cluster forced active, verify `recommend_action` actually shifts selection
   to the named avoidance tool, offline, before any live trial. The drafted instrument check (§ item 4)
   verifies only the WRITE (readouts flip negative) — exactly the half R2 already proved works
   differently from the half that decides behaviour.

### D2. Claim A's satiated gate `P(select eat | satiated) ≤ 0.2` contradicts the already-measured state-blind causal link — a predictable false-NULL (or, with the obvious "fix", a near-tautology)

`docs/wiring/substrate-learning-channels.md` measured directly that after a few successful eats
the state-blind causal link (`tool:eat`, no cluster, no drive conditioning) reaches ~0.89 and
**selects eat on its own** ("WITHOUT clusters -> eat"). In the drafted design (30 cycles × 5 seeds,
states alternated, one continuous agent per seed), the first few deficit cycles train that link;
every later satiated cycle then scores eat at causal_pos ≈ 0.9 with no drive term to outscore it
(satiation ⇒ no hunger need; nothing in `recommend_action` penalizes eat under satiation — verified
in the scoring loop: causal + reward_bias + cluster_bias + drive + explore, no satiation-side
suppressor). Whether eat wins depends only on what else is in the dining-hall repertoire and its
accumulated links — with a typical roster, `P(select eat | satiated) ≤ 0.2` fails from cycle ~5
onward, and the run reports a composition failure **while the composition works precisely as the
wiring doc says it must**. This is the same wall v1/v2 R2 hit, one gate over.

The tempting patch — reset the substrate every cycle so no causal link ever forms — makes the
satiated gate pass trivially but collapses Claim A into a measurement of the hand-written
`food→hunger→eat` affinity keyword table (code, not behaviour): near-tautological, and the prereg's
own honesty about being "prior-driven" doesn't rescue a gate that can only measure the prior table.

**Fix shape:** decide the learning regime explicitly and match the statistic to it:
- If **continuous** (learning on): drop the absolute satiated ceiling; gate on the
  **deficit-minus-satiated selection gap** with the satiated side reported descriptively, and
  pre-declare that the state-blind causal link is expected to lift the satiated floor over time
  (report `P(select eat | satiated)` as a trajectory, early vs late cycles). The deficit-contingency
  claim then rides the GAP, which the deficit-gated prior genuinely produces.
- If **reset-per-cycle**: rename the claim to what it measures ("the innate prior + need-derivation
  reach `eat` through the live loop") and accept that the satiated gate is then structural, not
  behavioural — and say so in the claim text.
Either is defensible; the drafted hybrid (continuous agent + absolute ≤ 0.2 ceiling) is not.

---

## SHOULD-FIX

### S1. Free-roam training couples the damage DOSE to the arm — the classic self-selected-treatment confound (charter item: episode-count asymmetry)

Under "free-roam exposure with K ≥ 10 dark-damage episodes", the FEAR arm's own learning curtails
its later exposure (it starts avoiding/escaping the dark mid-training) while ABLATED keeps walking
in. Three uncontrolled arm-differences follow: (a) total pain/damage episodes (dose ≠ dose),
(b) **Wire-2 aversion magnitude** — it accumulates per pain event under
`(entity_class='minecraft_player', 'drive:health')` and its read side lifts percept salience
(`gating.py` saturating mix), so the two arms enter the probe with different perceptual gating,
(c) **novelty/visit-count state** — `substrate_explore_bonus_weight`'s decaying bonus
(`weight/(1+visits)`) differs per tool per arm because the arms took different actions during
training. Any of these can produce FEAR-vs-ABLATED probe differences with the new valence store
playing no role. Worse, the `fewer than K usable episodes` stop rule fires preferentially in the
FEAR arm precisely when the mechanism works BEST — the design refuses its own strongest positive.

**Fix shape:** yoked, harness-scheduled exposure trials (declared, same disclosed-lane framing as
Claim A's hunger induction): K identical forced dark-encounters per arm, identical lit-recovery
interleave; free-roam reserved for the PROBE windows only. This equalizes pain dose, Wire-2
accumulation, and (approximately) visit counts, so the ablation difference isolates the subscriber.

### S2. Probe windows must be mob-free (spawning off) and full-heal — three confounds die at once

As drafted, probes run with spawning enabled inside the dark zone and a `health ≥ 18` filter:
1. **Reactive-avoidance positive confound:** a zombie visible at the dark entrance triggers
   avoidance through perception/salience (Wire-2-lifted, present in BOTH arms but dose-asymmetric
   per S1) — indistinguishable in the DV from anticipatory cluster-fear.
2. **Outcome-conditioned selection bias:** an agent that DOES enter the dark gets bitten, drops
   below 18, and its subsequent ticks/windows are excluded by the health filter — the measurement
   discards data as a function of the behaviour being measured, differentially by arm (ABLATED
   enters more → is censored more).
3. **Pre/post drift:** mob density in the dark zone at post-training probes depends on each arm's
   training trajectory (kills, despawns, spawn-cap history) — the pre-training baseline was taken
   against a different mob population than the post one, contaminating the within-agent
   `≤ 0.5 × pre` gate.

**Fix shape:** probe windows = spawning off + `/effect instant_health`-style full heal (disclosed
lane), so health stays ≥ 18 by construction, no mobs are perceivable, and the only path to
avoidance is the situation-keyed store. This also makes the healthy filter vacuous (good — filters
that never fire can't select).

### S3. θ, alpha, cap unspecified — a null becomes uninterpretable arithmetic

The read gates on `valence ≤ −θ`, the write books `−intensity·alpha` per episode, clamped to
`[−cap, 0]`, and the prereg names none of the three. If `K·alpha·1.0 < θ` the mechanism nulls by
parameter choice, not by architecture, and the shipped "read-path is the break" verdict is again
wrong. **Fix shape:** pre-register θ, alpha, cap with the arithmetic shown (`K` episodes at the
measured intensity ~1.0 must cross θ with ≥ 2× margin), and have the Step-2 probe re-run confirm
the accumulated magnitude, not just the sign.

### S4. Claim B has no N, no seeds, no statistic — as drafted it is n=1 per arm with point-estimate ratio gates

"Fresh agent per arm" (singular) + gates expressed as raw ratios (`≤ 0.5 ×`) with no variance
model, no number of probe windows, no per-seed replication, no test. A single-run difference in
either direction is uninterpretable; the previous line's discipline (per-seed gates + permutation
test, `r2_learned_bias_v2_prereg.md`) is the house pattern and is absent here. **Fix shape:**
S seeds per arm (fresh agent per seed), pre-registered probe-window count and length, gates
evaluated per-seed (median) + pooled, and a one-sided permutation test on per-seed
`P(enter dark)` (or the S15-corrected primary) for FEAR vs ABLATED.

### S5. Claim A's effective n is ~5 seeds, not 150 samples — deterministic argmax makes within-seed cycles non-independent

`recommend_action` is deterministic given (learning state, drive state, roster) — the v2 prereg
itself recorded this ("determinism → binary picks"). Within a seed, repeated same-state cycles are
near-copies; the real replication unit is the seed. The drafted "per-seed medians and pooled" is
directionally right — **state it as the unit of analysis** and size the seed count accordingly
(5 binary-ish seeds cannot support a 0.8/0.2 two-sided gate with any margin; 8–10 seeds or a
declared "all seeds individually pass" rule would).

### S6. Exploration bonus and config fingerprint not pinned

`substrate_explore_bonus_weight` defaults 0.0 but is settable via `config.json::sim.…`; if > 0 it
adds a dominating never-tried bonus that (a) inflates satiated-state eat selection early in Claim A
and (b) diverges across arms in Claim B (S1c). The v1 R2 guards (frozen-apparatus fingerprint,
absolute asserts, env unset) are not cited by this prereg. **Fix shape:** pin explore weight
(state its value, presumably 0.0), carry over the frozen-apparatus fingerprint + provenance
guards by explicit reference, and record the full NAc config in the run record.

### S7. Hunger runs uncontrolled through Claim B — a second live drive and a second learning channel in the "fear" measurement

Over long training + probe windows food drains; the derived `hunger` need then competes with the
threat need at selection time, and any eating during training books **positive** cluster-reward
credit onto whatever world cluster is co-active (this channel is NOT ablated — the ablation removes
only the pain subscriber). If food/eating happens mostly in the lit area, both arms learn a
lit-positive bias through the reward channel; symmetric only if schedules match (S1), and either
way it shrinks the headroom for the fear effect and adds a hunger-driven movement pattern to the
DV. **Fix shape:** keep the Claim-B agent satiated by disclosed induction throughout training and
probes (the same labelled lane Claim A uses), so the hunger need and the relief/reward channel are
quiescent and the only live learning channel difference between arms is the one being ablated.

### S8. "Usable damage episode" must be defined as *pain published*, not health delta

The drive-pain publish is LATCHED: it fires on band entry and re-fires only on a deepening breach;
the latch clears only when an evaluation observes recovery (`pain-needs-declared-failure-modes.md`,
instrument lesson 3). Repeated bites without full recovery + a healthy-state `evaluate_failures`
tick publish ONE pain event for several health drops → effective dose ≪ K while the episode
counter says K. The drafted stop rule checks the comfort-band breach (good — the band-edge trap is
covered) but counts episodes by damage, not by publish. **Fix shape:** count an episode as usable
iff `pain_bus.get_stats().total_published` incremented for it, and require the recovery interleave
to tick `evaluate_failures` while healthy (both already-documented instrument lessons — cite them
in the prereg so the harness review checks for them).

---

## NIT

- **Two primaries listed for Claim B** (`P(enter dark)` AND fraction-of-time-in-dark): pick one
  primary pre-freeze (D1's boundary probe decides which is valid), demote the other to secondary —
  two primaries invite post-hoc choice.
- **"Lit-area activity level unchanged between arms"** — metric undefined (distance moved? actions
  taken? per window?) and no threshold; define it or demote to descriptive.
- **Pre-training baseline windows for Claim B**: specify count/length and that they run under the
  same mob-free probe protocol (S2), else pre vs post differ in more than training.
- **Claim A deficit boundary**: food ≤ 6 vs satiated ≥ 16 — state the entropic thresholds
  (satisfaction/deprivation) of the body's food drive alongside, so "deficit" provably clears the
  `corrective_need_intensity` gate and "satiated" provably returns None (the v1 cluster-membership
  diagnostic showed adjacent food values can land in different clusters; one sentence of disclosure
  closes it).
- **`setdefault` collision note**: the learned anticipatory threat need and the innate
  `health→threat` need share the name `threat` in the drive dict; `_read_drive_states` merges
  derived needs by max and `setdefault`s into `drives`. Fine in healthy probes (innate = 0), but
  state the combination rule (max) in the mechanism PR so training-time behaviour is defined.

---

## Verified clean (checked, holds)

1. **Selection-not-execution DV for Claim A** does dodge the mechanical eat-refusal confound: the
   food-20 refusal is an execution-stage game rule; the DV reads the recommendation stage upstream
   of it. The residual satiated-side problem is D2 (causal link), not mechanics.
2. **The design does NOT repeat the R2 messenger trap.** The new valence store feeds the drive/need
   vector (break-1 channel), which R2 measured as a channel that genuinely drives selection — it
   does not require `cluster_reward_bias` to be a behavioural cause, and adds no fifth scoring term
   to `recommend_action`. This is the correct reading of `substrate-learning-channels.md`.
3. **Booking to the WORLD cluster only** (not interoception) correctly avoids the tautological
   hurt-fear key — the interoception cluster at pain time encodes the breach itself.
4. **The ablation shape** (subscriber not attached; pain still publishes; Wire 2 still fires)
   targets exactly the new mechanism rather than pain generally — valid GIVEN dose matching (S1);
   without S1 the "Wire 2 fires in BOTH arms" symmetry claim is only nominal.
5. **B8 action-blame suppression** means dark-damage writes no negative causal links onto in-flight
   actions in either arm — arm differences cannot ride action-blame; and the documented
   bystander-positive-link pathology (eat causal_pos 0.78 during damage) is arm-symmetric under
   matched schedules.
6. **The healthy-probe threshold ≥ 18 does exclude the innate `health→threat` need at read time:**
   `corrective_need_intensity` for a homeostatic drive returns None unless
   `value − set_point < −comfort_band` (set_point 20, band 6 → fires below 14). The leak is only
   via mid-window damage — handled by S2, not by the threshold.
7. **Cluster-ID identification across arms** is handled acceptably by the Phase-0 separability
   record + the instrument-drift stop rule (re-check on classroom geometry) — each fresh agent's
   dark/lit cluster ids are established by its own probe, not assumed shared.
8. **Falsifier direction honest**: FEAR ≈ ABLATED ships as a null naming the read path — right
   instinct; D1/S3 exist to make sure that verdict would be TRUE when shipped.
9. **The mechanism-before-harness ordering** (own reviewed `src/` PR, probe re-run gating the
   boundary) matches the fix-ships-with-a-caller discipline; the probe re-run as write-side
   instrument check is sound as far as it goes (D1 extends it to the read side).

## Bottom line

The two-claims split is honest and the mechanism placement (feed the need vector, not the score
table) is the right lesson from R2. But as drafted, both headline gates can return false verdicts
for reasons already measured elsewhere in this repo: Claim B's primary DV asks the read path for a
prediction it structurally may not make (D1), and Claim A's satiated ceiling re-collides with the
state-blind causal link that killed R2 v1/v2 (D2). Both are fixable pre-build with cheap offline
probes and gate rewording; S1/S2 (yoked dose + mob-free probes) are the difference between an
ablation that isolates the subscriber and one that isolates "whatever diverged during training."
