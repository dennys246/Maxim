# R2 learned-bias v2 — Bio-faithful lens review

**Verdict: SHOULD-FIX (multiple), one DO-NOT-BUILD-adjacent.** The core premise is faithful — the
drive-relief cluster credit's real job *is* state-conditioned competitive selection, and the prereg
targets the right channel (`cluster_reward_bias`, not the recognition-modulator `reward_bias`) and the
right consumer (`recommend_action` argmax). **But** the design's selection model is a caricature of the
actual mechanism in three load-bearing ways, and it does not pin two hard gates + one hard threshold
that can force a null for reasons unrelated to the credit. The pilot MUST characterize these before
freeze, or the confirmatory run risks a mechanism-artifact null.

Grounded against `src/maxim/decisions/nac.py::recommend_action` (lines ~2036–2260),
`::decay_cluster_reward_biases`, `::apply_wall_clock_decay`, `src/maxim/embodiment/sem.py::corrective_need_intensity`,
`_data/components/bodies/minecraft_player.yaml` (food drive spec), and `docs/agents/bio-memory.md`.

---

## DO-NOT-BUILD (adjacent) — a mechanism artifact can force the null

### D1. Two hard gates AND one hard threshold sit UPSTREAM of the additive-score argmax; the prereg pins none of them, and any one can mask the credit.

`recommend_action` is not a single argmax over a smooth score. Three cliffs precede the pick:

1. **Explore-FIRST hard gate** (`substrate_explore_bonus_weight > 0.0`, lines ~2224–2229): if any
   scored tool was never selected this session, selection is **restricted to untried tools** —
   the learned cluster credit is entirely bypassed.
2. **Drive gate** (`drive_gate_enabled` + `max_drive_intensity > drive_gate_threshold(0.5)`,
   lines ~2253–2258): restricts selection to the drive-relevant subset. `hunger` maps to `("eat",
   "pick_up", "food", "consume", "feed")` in `_DRIVE_TOOL_AFFINITIES`, so eat is drive-relevant and
   the competitors are not → the gate collapses selection to `{eat}` **in both arms**, forcing
   `P(eat|hungry)=1` in LEARNING *and* NO-CREDIT → `isolated_effect ≈ 0` → forced NULL even if the
   credit works. This is the v1 causal-link-saturation failure re-homed in the drive gate.
3. **The 0.5 drive-relevance cliff** (line 2105, `if drive_value <= 0.5: continue`): the innate
   drive-prior term is a hard step, not the graded contributor the prereg assumes (see D2).

Defaults are safe (`substrate_explore_bonus_weight=0.0`, `drive_gate_enabled=False` — NACConfig lines
459/507), **but both are wired from `config.json::sim.*` at `build_bio_stack`**, and the survival world
(1.3) is exactly where `drive_gate` (the Exp-42 motivated-attention mechanism) is likely turned ON.
The prereg's guard list pins `min_confidence=0.0` and determinism but is silent on these two.

**Consequence:** a confirmatory run under a survival-world config with `drive_gate_enabled=True`
returns a null that says nothing about the credit.

**Fix:** add `drive_gate_enabled=False` and `substrate_explore_bonus_weight=0.0` to the frozen-apparatus
fingerprint and assert them with the other absolute asserts. If the design *wants* the drive gate on
(to be faithful to the survival loop), then the probe state must be one where the gate provably does not
engage — which is D2.

---

## SHOULD-FIX

### S1. Food 11's corrective need is EXACTLY 0.5 — a knife-edge that contradicts the prereg's own rationale.

`food` is `EntropicDriveSpec(drift_direction="down", satisfaction_threshold=16, deprivation_threshold=6)`.
`corrective_need_intensity` returns `(16 − value)/10` below satisfaction. So:

| food | corrective need | drive-relevance (`>0.5`?) | drive gate (`>0.5`?) |
|---|---|---|---|
| 4 (train) | 1.0 | yes | fires |
| **11 (hungry probe)** | **0.5** | **no (`<=0.5` skip)** | **no (`0.5 > 0.5` False)** |
| 12 | 0.4 | no | no |
| 18 (satiated) | None | no | no |

Food 11 lands *precisely* on the hard threshold. This is (accidentally) the clean regime — at need 0.5
the drive-prior term is skipped and the gate stays off, so the cluster credit is the **sole**
differentiator between eat and the competitors. That is actually a CLEANER isolation than the prereg
describes. But two problems:

- **It's a knife-edge.** Any float drift, any retune of the 0.5 floor, or a probe at food 10
  (need 0.6 → drive term fires + gate engages) flips the mechanism into D1's forced-null regime. The
  design is silently relying on `0.5` being on the "off" side of `<=`. Fragile and undocumented.
- **It refutes the prereg's stated mechanism.** The doc says "eat preferred at a hungry probe"
  (Question §1), "the innate prior favours eat when hungry (both arms)" (§"Why…"), and "The prior
  contributes to the gap in every arm" (Known-limits). At food 11 the prior contributes **zero** —
  the drive term is skipped. The isolation math still works (`gap_LEARNING − gap_NO-CREDIT`), but the
  written model of *why* is wrong, which will mislead interpretation of the result.

**Consequence:** if a reviewer or a later re-run "fixes" food 11 → food 10 to make the probe "more
clearly hungry," they walk straight into the forced null.

**Fix:** state explicitly that food 11 gives corrective need = 0.5 and that this is why the prior does
NOT contaminate the probe; pin the probe food value in the fingerprint; and in the pilot LOG the actual
`drives` dict + the per-tool score `components` at the food-11 probe to confirm the `drive` component is
0.0 and no gate engaged. Reconcile the prose (drop "the prior contributes to the gap" — it doesn't here).

### S2. The "headroom ~1/(K+1)" titration rationale is a proportional-selection caricature of a deterministic argmax.

`recommend_action` selects `max(scores, key=lambda t: (scores[t], t))` — a hard argmax over **additive**
scores with a **name-sort tiebreak**, not a proportional/softmax pick. There is no `1/(K+1)` baseline:

- eat's baseline P(eat) is `P(eat_score > every competitor_score)`, driven by per-seed causal-link
  noise, not `1/(K+1)`.
- The tiebreak picks the alphabetically **largest** name on exact ties, so `turn`/`move_to`/`mine_block`
  all beat `eat` on a tie — the baseline is systematically biased *against* eat, not toward `1/(K+1)`.
- Adding competitors raises the bar the credit must clear: eat must beat `max` of K competitor scores,
  and `max` of more i.i.d. draws is stochastically larger. So "more K = more room for the credit" is
  backwards for the argmax — more K makes the credit's job **harder**, not roomier.

The pilot audit ("raw P(eat) drops with K") partially protects against shipping on a false premise, but
the *inference* the prereg draws from that audit ("headroom grew, so the credit had more room") does not
follow from the mechanism.

**Consequence:** the dose-response direction the primary decision-rule expects (non-decreasing
`isolated_effect(K)`) may not materialize even for a perfectly-working credit; a genuine credit could
produce a *decreasing* curve purely from argmax `max`-of-K dynamics, and the rule would misreport it as
"not choice-space-scaling."

**Fix:** re-ground the titration in the argmax: the meaningful axis is the **eat-to-best-competitor
score margin** and how the credit shifts eat across that margin as K grows. Have the pilot measure that
margin directly (S3) and re-derive the expected `isolated_effect(K)` shape from it, rather than from a
`1/(K+1)` proportional model.

### S3. Binary argmax is faithful to the credit's BEHAVIOURAL purpose but is a threshold readout of a graded quantity — add the score margin as a graded secondary.

The credit is a continuous additive term in `[-1, +1]` per cluster (line ~2082). The behaviour it drives
IS the argmax pick, so measuring the pick via the real consumer is faithful — good, no hand-composed
shortcut. **But** a binary pick only moves when the credit pushes eat's score across the best
competitor's; a credit that lifts eat from 0.30-below to 0.05-below the leader reads as **zero effect**,
and once eat wins, further credit is invisible. The current secondaries (cluster-bias *magnitude*; raw
P(eat)) tell you the credit **formed**, not whether it was competitively **decisive**.

**Fix:** record the pre-gate `scores[eat] − max(scores[competitor])` per (arm, K, state) as a graded
secondary. It distinguishes "credit inert" from "credit moved eat but not across the argmax boundary"
and is the faithful graded readout the prereg's binary primary cannot give. (The `components` dict is
already computed per tool at lines ~2031/2171 — the margin is cheap to emit alongside the existing
`sim_recommend_action` event.)

### S4. The cluster credit is keyed AT ACTION TIME; training that lets food rise smears the credit off the probe cluster.

`update_cluster_reward` / the measured-relief path (break 2) credit the interoception cluster **active
when eat ran**. Training happens at food ≤ 4 (cluster `fd0ae83c`), but a successful eat RAISES food
(`self_effect food: 4.0` in the YAML, live-owned), so a second eat in the same episode fires at food 8,
11, … — different clusters — and its credit lands off `fd0ae83c`. The probe reads only `fd0ae83c`.

**Consequence:** eat's credit in the probe cluster is diluted by however many eats fired after food
climbed out of the training band, weakening the very signal the experiment measures — and doing so
*differentially by episode length*, adding variance across seeds.

**Fix:** reset food to the training band before each rewarded eat (one eat per hunger dip), or otherwise
pin that every credited eat fires while `fd0ae83c` is the active cluster. Verify in the pilot by logging
the cluster id at each `update_cluster_reward`. (Cross-ref wiring/environment lenses.)

---

## NIT

### N1. `record_outcome` writes THREE learned traces, not the two the wiring doc names.

Besides the causal link (state-blind) and `cluster_reward_bias` (state-conditioned), `recommend_action`
Component 2 adds `reward_bias(agent_id, event_sig)` — a per-(agent, tool) trace, **state-blind**, capped
at `max_reward_bias` (0.20). It forms on every eat success. It is harmless for isolation (state-blind →
cancels within-arm in the contingency gap and again in the arm-difference), but the "two channels"
framing in `docs/wiring/substrate-learning-channels.md` and this prereg is incomplete. Confirm the
NO-CREDIT ablation suppresses **only** `cluster_reward_bias` and leaves `reward_bias` present in both
arms (so it cancels rather than confounds).

### N2. Decay timescales are respected by the same-session, fresh-substrate design — but verify no save/load between train and probe.

In-session `decay_cluster_reward_biases` uses `cluster_reward_bias_decay_tau=300` ticks (~0.33%/tick);
`apply_wall_clock_decay` uses `cluster_bias_wall_decay_half_life_s=86400` (1 day) and fires **only on
`load()`**. "Every (arm, K, seed) is a fresh substrate," train→probe in one session with no reload, so
wall-clock decay never applies — good. Two residual cautions: (a) the tick decay runs on **every** tick
including competitor episodes, so at higher K the eat cluster bias decays over more intervening ticks
between reinforcements → the plateau is **lower at higher K**, quietly co-varying the credit magnitude
with the titration axis the design wants to hold constant. The raw-P(eat) secondary audits headroom but
NOT that eat's cluster-bias magnitude is equal across K — add that to the secondary (measure
`cluster_reward_bias` magnitude per K and confirm it is flat, else the titration confounds credit
magnitude with choice-space size). (b) Ensure the probe fires with few intervening ticks after the last
eat reinforcement.

### N3. Faithful elements — verified, keep.

- **Naming clean:** `NAc`, `cluster_reward_bias`, `EC`, interoception cluster, `drive:hunger` — no
  removed identifiers (`NucleusAccumbens` etc.) reintroduced.
- **Credit routing faithful:** the credit keys on the interoception cluster from
  `SensorEncoder.encode_sensors`; the food-11-shares-`fd0ae83c` / food-18-different-cluster design
  correctly exploits the cluster keying to make the credit state-contingent. (Known-limit re-confirm of
  the cluster identity is rightly flagged.)
- **Competitors building causal links is faithful, not artificial:** competing actions genuinely coexist
  as separate additive argmax terms; giving eat's causal link real competition is exactly the mechanism,
  and under relief-only the competitors correctly form NO cluster bias. ("Always-successful/always-
  executable" is an environment-lens concern.)
- **Right channel targeted:** the prereg isolates `cluster_reward_bias` (competitive selection), not the
  `reward_bias` recognition-modulator — the credit's real job, faithfully.
