# Exp 57 — The dose–response ladder: does collective learning scale? (pre-registration)

**Status:** PRE-REGISTERED (pending merge) — freezes at the merge commit of this file,
before any harness run. A two-lens review round (design/methodology + code-accuracy) ran on
the draft BEFORE freeze, matching the Exp 56 discipline; its cross-confirmed findings were
folded, NOT discovered at campaign price: **G = 8 contradicted the 4 frozen slots → G = 4**
(both lenses); **coverage-widening was assumed under an identical-schedule spec that would
force a flat curve → independent per-contributor seed/order + a Phase-0 divergence tripwire**
(design lens; the link-channel analog); **δ_eff was a gate margin slated to be set from
pilot data → frozen at 0 now** (design lens); the backwards censoring-conservativeness
argument was corrected and gated on two artifact guards; the fold arithmetic was restated
from "the mean" to the shipped left-associative convex combination (code lens); and the DV
was made deterministic (bias-decisive from provenance) to remove ε-noise. The 1.2 **second** claim — the four
arms ([Exp 56](exp56_four_arm_sharing_preregistration.md), EARNED 2026-09-06 —
write-up `../56_four_arm_sharing.md` lands with PR #648) asked *does a taught want
transfer*; this asks *does pooling taught wants scale*, which is the claim Oasis actually
rests on. Design authority: [minecraft_benchmark.md](../../plans/minecraft_benchmark.md)
§"The dose–response ladder"; roadmap [§"The 1.2 benchmark"](../../plans/roadmap_1_1_to_1_3.md).
The gate set and the arm/condition structure below are carried from those frozen
documents; this pre-registration operationalizes the dependent measure they left open and
freezes everything else.

**Merge semantics under test are the REAL 1.2 semantics** (as Exp 56): pooling runs through
`maxim substrate ingest` — the V1–V10 adapter + `substrate_merge` (aligned re-key,
tighten-only clamp) — folding N contributor bundles into one fresh receiver, never bare
`nac_merge`. The N→1 fold is the shipped `substrate_merge` applied left-associatively over
the N bundles.

**Lineage:** Exp 42/45 (cross-session) → Exp 52 (the want is learned) → Exp 53/53b
(cross-context, on a body) → Exp 56 (cross-agent: one donor moves one receiver) → **this**
(N donors' partial learning pools into a receiver that reaches criterion in fewer per-agent
trials than one donor could). The scripted precedent is
`5_operant_creche_federation.py` / `7_graded_creche_federation.py`.

**Owner intent (recorded so the claim cannot drift):** the claim is that pooling the
partial learning of N genuinely-independent agents (D44 sense: distinct `agent_id`,
separately constructed `EntorhinalCortex` + `SensorEncoder`, cluster ids disjoint by
construction, meeting ONLY through the merge) lets a fresh receiver reach a criterion
behaviour with a SMALLER PER-AGENT budget as N grows — and does so because the merge
combines *learning* (coverage of taught contingencies) rather than accumulating noise or
mere magnitude. Nothing here claims hardware (the robot replicates ONE rung, not the
ladder — its own prereg on Exp 54's body), nothing claims a from-scratch sequence model
(1.3), nothing claims cross-layout generalization.

---

## The mechanism, named precisely (before any arms — the Exp 56 lesson)

Exp 56's first review critical was a MISREAD mechanism. The same discipline applies here,
and the mechanism decides the entire apparatus, so it is pinned first against the shipped
code, not assumed.

**How pooling scales is COVERAGE, not magnitude — verified in `hivemind/merge.py`.**
`cluster_reward_bias` folds via `_merge_mean_clamped`: **shared keys are combined** (two
contributors that learned the SAME `(agent, cluster, tool)` bias key produce a convex
combination of their values — NOT their sum), and **keys unique to one contributor are
kept** (zero-prior rule). Two mechanism facts, both against the shipped ingest path:

- The N→1 fold on the foreign path is `substrate_merge` applied **left-associatively**
  (`ingest_bundle` writes merged state back, so the next fold sees it). `_merge_mean_clamped`
  averages **pairwise**, so three-plus contributors on one shared key get an
  ORDER-DEPENDENT convex combination — for three, weights (¼, ¼, ½); for four,
  (⅛, ⅛, ¼, ½) — the last-folded contributor weighted ½, bounded by the max contributor
  value. (This is NOT the equal-weight 1/N fold: `nac_merge_many` does that, but it is
  trusted-local-only, does no EC alignment, and is deliberately OFF the foreign ingest path.
  The ladder measures the shipped foreign semantics.)
- Because the combination is convex (bounded above by the largest contributor), pooling
  **cannot scale by "louder wants"** on a single contingency — N partial-but-correct biases
  on ONE contingency stay bounded by the strongest single learner and can even DILUTE (a
  decisive +0.5 averaged with a weak +0.2 → +0.35, which may fall below the decisiveness
  bar — the dilution counter-mechanism the gates must observe, not assume away).
- What pooling grows is **coverage**: contributor *i*, sample-limited to a per-agent budget
  K, gets a decisive bias on only SOME of the world's taught contingencies; the union over
  N contributors covers MORE contingencies, so the merged receiver reads out correctly on
  more of them. This is the precedent's own mechanism (Exp 45d cell-starvation: one agent
  covers some state cells, N agents cover more, the merge unions them) and the reason the
  scripted crèche reached `single_full` while a single partial learner did not.

**Consequence for the apparatus (the load-bearing design decision):** the world must
present **G distinct taught contingencies** (situations each with their own correct
affordance), not Exp 56's single target, AND the N contributors must genuinely DIFFER in
which contingencies they cover first (independent per-contributor seed + presentation
order, §Apparatus/S5) — coverage cannot widen if every contributor covers the same set in
the same order. The dependent measure is **coverage** — the fraction of the G contingencies
the merged receiver reads out correctly — and trials-to-criterion is the per-agent budget at
which pooled coverage first crosses the criterion. A single-contingency world (Exp 56's), or
byte-identical contributors, would each guarantee a false-flat curve for a reason readable
in advance; both are designed against below and tripwired in Phase 0.

**The independent-agent wrinkle (gate 6 / D43 territory, now closed).** Independent agents
encode the SAME situation to DIFFERENT cluster UUIDs; `substrate_merge` calls
`ec_merge_aligned` (which returns an `id_map`, unlike bare `ec_merge`) and re-keys the
biases through `rekey_nac_state`. So two contributors that BOTH learned contingency *g* have
their *g*-clusters aligned → their bias keys become shared → convex-combined (coverage of
*g* confirmed, magnitude bounded); two contributors that learned DIFFERENT contingencies have
non-aligning clusters → kept as a union (coverage widened). The ladder therefore exercises
exactly the re-keyed merge Exp 56 validated at N = 1. This makes the `ec_merge_aligned`
alignment rate load-bearing — it governs the widening-vs-dilution race — so Phase 0 check 2
verifies alignment behaves (same-situation clusters align, distinct-situation clusters stay
distinct) AND reports the quantitative shared-vs-union split on the G-contingency world
before any rung runs: an over-aligning merge collapses coverage (false flat), an
under-aligning merge inflates it (a union that never averages).

## The question

N genuinely-independent agents are each taught, on a scripted balanced schedule with a
harness teacher (the Exp 56 A-phase, unchanged), against a world of G taught contingencies,
each with a per-agent budget of at most K_max trials. After each per-agent trial their
substrates are snapshotted. For a per-agent checkpoint t, the N contributors' t-th
snapshots are folded into a fresh receiver through the real 1.2 path, and the receiver's
**coverage** — the fraction of the G contingencies it reads out correctly at first contact
— is measured. As N grows from 1 to 8, does the per-agent budget needed to reach a criterion
coverage strictly decrease?

Four things must hold for the answer to be "collective learning scales":

1. median per-agent trials-to-criterion **strictly decreases** across N = 1 → 8
   (**MONOTONICITY**, primary);
2. at each rung, pooling is at least as efficient in TOTAL experience as one agent given
   all of it (**NOT-JUST-MORE-DATA**);
3. an untaught crèche never reaches criterion at any rung (**NOISE-FLOOR**);
4. a flat curve is not hidden — it ships as the honest result (**FALSIFIER**, stated so a
   null is a finding, not a failure).

## Dependent measure (frozen)

**Coverage at a checkpoint (read DETERMINISTICALLY from provenance — the review's ε-noise
fix).** For a merged receiver, present EACH of the G contingencies once and read the
`NAc_RECOMMEND` decision-provenance record. A contingency counts as covered iff the
**LEARNED-BIAS component makes the taught affordance the argmax** (the Exp 56 bias-decisive
property, but read as a property of the merged SUBSTRATE, not of an ε-greedy emitted
action). Coverage = (# contingencies covered) / G. Reading the decisive component directly —
rather than the ε-greedy selector's stochastic choice — removes the ~ε per-contingency miss
that would otherwise make a single-receiver, G-Bernoulli coverage read ε-noise-limited and
conflate "budget grew" with "dither landed" across the sustained window. The ε-greedy
selector is retained ONLY where L2 de-phase-locking needs it — the noise-floor arm's readout
(§Conditions) — where there is no learned-bias component to be decisive. Read at the real
consumer's provenance (the score the loop actually computes), never from dict contents
(D44's rule).

**Per-agent trials-to-criterion τ(N) (the ladder DV).** Each of the N contributors trains
its own balanced schedule; after each of its trials t ∈ {1..K_max} its `(NAc, EC)` snapshot
is recorded. For each t, fold the N contributors' t-th snapshots into a fresh receiver
(`substrate_merge`, N→1) and measure coverage d(N, t). **τ(N) = the smallest t such that
the sliding window of the last W checkpoints all satisfy d(N, t′) ≥ C** (sustained, so a
single noisy checkpoint cannot trip it). If no such t ≤ K_max exists, τ(N) is
**right-censored** at the sentinel K_max + 1 (recorded as censored; a single partial learner
failing to reach criterion is itself informative and is the expected N = 1 outcome).

Contributors are offline snapshots — they never interact during learning; they meet ONLY
through the merge. The receiver never learns (it is re-created fresh at every checkpoint):
the rate lives entirely in the CONTRIBUTOR budget, which is what "sample efficiency per
agent" means and what Oasis sells.

**Secondary (reported, never gated):** endpoint coverage d(N, K_max); per-contingency
coverage curves; the winning score component per contingency (the §Links audit); the
number of shared-vs-union bias keys after each fold (the coverage mechanism, made visible);
`ec_merge_aligned` alignment counts per fold; the fraction of τ values censored per rung
(the censoring-artifact guard, §Gates).

## Conditions per rung (frozen; the precedent's controls, generalized to a curve)

N ∈ **{1, 2, 4, 8}**. At each rung, three conditions, all reading out through the identical
probe battery:

| condition | construction | question |
|---|---|---|
| **creche(N)** | N taught contributors × per-agent trials, folded N→1 at each checkpoint | the test — does pooled partial learning reach criterion, and at what per-agent t |
| **single_matched(N)** | ONE taught contributor given up to N×K_max trials; coverage read through the IDENTICAL 1→1 re-key fold + probe battery as creche (see below); τ measured in ITS OWN trial index | is pooling at least as efficient in TOTAL experience as one agent seeing everything? (the precedent's `single_full`, made a rate) |
| **creche_none(N)** | N contributors on the balanced schedule with the teacher WITHHELD (zero credits by mechanism — the Exp 56 satiated construction), folded N→1 | noise floor at that rung — pooling uncredited substrate must not manufacture coverage |

**single_matched reads through the same merge path (the review's apples-to-apples fix):**
the single agent's coverage is read through a 1→1 `substrate_merge` re-key fold + the same
probe battery creche uses — NOT directly off its own substrate — so the ONLY difference
between creche(N) and single_matched(N) is *N-pooling* vs *one-agent-more-trials*, never
*merged-vs-unmerged readout*. Otherwise the merge's own clamp/align cost would land only on
creche and bias NOT-JUST-MORE-DATA against pooling. (This 1→1 re-key fold is exactly the
anti-vacuity kit's persist-expected "re-keyed-alone" variant.)

N = 1 is the shared baseline: creche(1) ≡ single_matched(1) ≡ one taught contributor at
its own budget, both read through the 1→1 fold. `creche_none` generalizes Exp 56's satiated/dangling controls to the ladder
(the falsifier's job here is the flat-curve clause, not a separate dangling arm — the
dangling-half falsifier was discharged at N = 1 by Exp 56 and is not re-litigated).

## Pre-registered gates (constants carried from the frozen design; numeric apparatus constants set in Phase 0, §Phase 0)

| gate | rule |
|---|---|
| **MONOTONICITY** (primary) | the per-seed τ(creche(N)) values fall across N = 1 → 8 by the **Jonckheere–Terpstra** ordered-alternative trend test (decreasing alternative), **p < 0.05** one-sided, evaluated with a **permutation null** (an entire rung is expected tied at the censoring sentinel, so the normal approximation degrades — the permutation null is the frozen form, not tie-corrected-normal). JT — not "median strictly decreases" — IS the gate; per-rung medians with bootstrap CIs are reported descriptively (with N = 1 and N = 2 both expected censored, their medians tie, so a "strictly decreasing median" criterion would fail a real effect and is deliberately NOT used). JT is the matched statistic for four ordered levels with replicate seeds (match the statistic to the design). **Censoring-artifact guard (the review's correction):** right-censored τ (did not reach C within K_max) rank as the largest value, which — because censoring concentrates at SMALL N (check 4 sets K_max so N = 1 is expected fully censored) — **reinforces the decreasing alternative rather than opposing it**. That is anti-conservative, so a JT pass is not sufficient alone: it MUST ALSO survive (i) a sensitivity re-test with fully-censored rungs dropped, AND (ii) a graded trend in the UNCENSORED secondary — endpoint coverage d(N, K_max) rising with N — so the claim rests on graded coverage widening, not on a two-point censored step at the bottom rung. The per-rung censored fraction is reported. |
| **NOT-JUST-MORE-DATA** | at each rung N ≥ 2, creche(N) TOTAL experience to criterion (N × τ(creche(N))) ≤ single_matched(N) trials-to-criterion **within the frozen margin δ_eff = 0** (pooling must be *no worse* than one agent seeing everything; the margin is frozen HERE, not set from pilot data — setting a gate threshold after seeing the N = 8 pilot is the forbidden retune). A strict-positive slack is admissible ONLY if justified mechanically (a named `substrate_merge` clamp/align overhead) in a pre-data amendment, never from the measured effect. If pooling costs MORE total experience than one agent seeing everything, that is an honest merge-cost finding and ships as one — it does not fail the primary, it qualifies it. |
| **NOISE-FLOOR** | creche_none(N) coverage stays at the floor (never reaches C) at EVERY rung, **checked especially at the TOP rung N = 8** where any residual link-channel contribution is largest. Two facts make this safe and are asserted, not assumed: the coverage DV counts a contingency only on a LEARNED-BIAS-decisive win (link-component wins are excluded by construction), and `_merge_link_pair` caps merged link `confidence = max(left, right)` (merge.py) — N untaught contributors' links do NOT accumulate to super-confidence with N. Deterministic coverage of creche_none is therefore ≈ 0. |
| **FALSIFIER** (stated, not a pass/fail row) | τ(creche(N)) flat in N (JT not significant, or the sensitivity/endpoint guards fail) means collective learning buys nothing on this apparatus. A flat curve is a RESULT and ships as one, with the audit target being the coverage mechanism (shared-vs-union key counts + the contributor-divergence Jaccard, §Phase 0) and `ec_merge_aligned` alignment — a coverage that never widens with N points at over-alignment or identical contributors. |
| **ANTI-VACUITY** (apparatus; no verdict without it) | the analyzer's no-op merge kit (Exp 56's, generalized): folding with `substrate_merge` replaced by *receiver-unchanged* and *empty-fold* variants MUST collapse coverage to the floor at every rung; a *contributors-re-keyed-alone* fold (one contributor, re-keyed, no averaging partner) is RECORDED and expected to persist (D62 / Exp 56 amendment 1). A gate that cannot fail is not a gate. |

The scaling claim requires MONOTONICITY (JT + both censoring-artifact guards) **and**
NOT-JUST-MORE-DATA **and** NOISE-FLOOR, with ANTI-VACUITY + the L2 gate (§Phase 0) clean.
MONOTONICITY passing while NOT-JUST-MORE-DATA fails is a PARTIAL ("pooling scales in
per-agent trials but costs total experience"), named and shipped or held at the release
checkpoint — owner call, not a silent pass.

**Power:** the confirmatory seed count is fixed A-PRIORI at ≥ **20 independent cohorts per
rung** (each cohort a fully independent draw of the N contributors — distinct per-contributor
seeds and presentation orders, §Apparatus/S5 — plus the single_matched agent and the
untaught crèche at that rung). It is NOT set from pilot data (one pilot cohort gives n = 1
τ/rung and no variance estimate); the Phase-0 pilot (§check 5) only confirms the effect is
obtainable and sanity-checks the τ spread. Report per-rung median τ with a bootstrap 95% CI;
the JT permutation test uses the per-seed τ values.

## Apparatus (what must exist at the frozen commit; owed items in the sign-off)

Exp 56's teacher, balanced schedule, ε-greedy selector regime (ε = 0.2,
`min_confidence = 0.3`, `substrate_explore_bonus_weight = 0` at the probe, per-pair name
permutation), `MAXIM_OPERANT_ONLY_CREDIT=1`, the export/ingest flags (no
`--allow-unstamped-geometry`, no `--force-digest`, no `--inherent-trust`), and the S1–S8
declarations are reused by reference. New or generalized for the ladder — and the world is
NOT unchanged (below):

- **G taught contingencies (not one) — G = 4, matching the 4 FROZEN slots.** Exp 56's
  amendment 2 froze **exactly four** far+high slots — (88,112,0)/(−88,112,8)/(8,112,88)/
  (−8,112,−88) — and validated each separates from REST (cos 0.19). The bench body's **8**
  is the *affordance roster* (k = 8), NOT the situation count; there are only 4 slots, so
  **G = 4** situation→correct-affordance contingencies, one per frozen slot (the per-cohort
  permutation maps slots→affordances so the alphabet is not load-bearing). Coverage is over
  the 4, so coverage ∈ {0, ¼, ½, ¾, 1} and the criterion is C = ¾ = 3 of 4 (confirmed in
  Phase 0). **A raised G would require ADDING slots** — new far+high coordinates,
  `setup_world.py` + `FROZEN["contingency_slots"]` extended, each re-validated — which is an
  explicit apparatus change, pre-registered by amendment, NOT "reuse unchanged"; it is not
  taken unless Phase 0 shows G = 4 gives too coarse a curve. **Not yet established (a real
  Phase-0 risk):** amendment 2 validated slot-vs-REST separation, NOT slot-vs-slot; Phase 0
  check 1 (widened to PAIRWISE separation over the 4) is the gate for it, and a failure is
  an apparatus fix, not a silent proceed.
- **Contributors must genuinely differ (the review's coverage-widening fix).** Each of the N
  contributors draws an **independent seed** (distinct ε-stream AND an independent per-
  contributor permutation of the contingency presentation ORDER — the precedent's
  `seed=int(rng.integers(1<<30))` per infant, which the first draft's "identical seeded
  schedule" wrongly collapsed). The balanced-exposure STRUCTURE is identical across
  contributors (each contingency scheduled equally often — the link-neutralization); only
  the seed and order differ, so contributors cover different contingencies first and the
  union can genuinely widen with N. Byte-identical contributors would force creche(N) ≡
  creche(1) and a guaranteed flat curve; Phase 0 check 2b tripwires it.
- **N-AUT generalization of the 1.1.4 two-AUT harness.** The A-phase runs N contributor
  sessions **sequentially against the live bridge** (one client per server/bridge, the Exp
  56 constraint; instances parallelize across ports). Contributors are snapshotted per
  trial; the B-phase folds and probes offline against the same seeded world script. Nothing
  needs N agents in one world at once (the shared-world fake-bridge gap is off this path,
  as in Exp 56).
- **Independent contributors (D44), asserted per cohort (S3):** all N `agent_id` distinct
  and distinct from the receiver; each contributor's EC constructed separately; pre-fold the
  receiver holds zero contributor cluster ids; post-fold, landed bias keys carry the
  RECEIVER's agent id. The `ec_merge_aligned` `id_map` is recorded per fold (the coverage
  mechanism's audit surface).
- **Loop:** substrate-primary, **no LLM in the action path**; `consolidation="full"`;
  `MAXIM_OPERANT_ONLY_CREDIT=1` everywhere (the Exp 56 apparatus, unchanged).
- **Harness compliance:** `assert_repo_interpreter` before the first session;
  `preflight_gated_record` (clean tree; `--allow-dirty` disallowed for the confirmatory
  campaign); `executed_code_provenance` per record; every record carries `ts`; `--mock`
  smoke + `--resume`; results route through `evidence_out_paths`
  (`--write-experiment-results` to touch committed paths); analyzer verdict constants frozen
  in `scripts/analyze_exp57.py` in the same PR as the harness, extended never retuned. One
  harness at a time, not on the leader machine (S8/L02). `require_semantic_encoder` is
  deliberately NOT required — the path under test is `SensorEncoder` end to end.

## Phase 0 — instrument checks + apparatus-constant calibration (gate the campaign)

Run gated on the frozen body + G = 4 world script before any rung; recorded and committed
like any gated record. The APPARATUS constants the design leaves open (K_max, C, W) are
CHOSEN here and frozen by a pre-campaign amendment with the disclosure the Exp 52/56
precedent set; the GATE constants — δ_eff = 0, p < 0.05, N ∈ {1,2,4,8}, the ≥ 20 seed
count — are frozen by THIS file and are NOT retuned from Phase-0 data. Every reading is
disclosed. All six checks must pass or the campaign does not start:

1. **G = 4 discriminability, PAIRWISE (L11/A4).** The 4 frozen slots must separate from REST
   (≥ 0.70 on each rest→situation contrast, stability ≥ 0.70 on re-probe) AND **from each
   other pairwise** (every slot-pair cos below the single-cluster bound). This is Exp 56
   check 1 widened from rest-vs-one to the full pairwise set — amendment 2 established only
   rest-vs-slot, so slot-vs-slot is genuinely unestablished and this is its gate. A failure
   is an apparatus fix (re-place slots / re-derive geometry), recorded, not a proceed.
2. **Merge-alignment sanity + the quantitative split (the independent-agent wrinkle).** Two
   independent contributors that each learned the SAME contingency fold to a SHARED (convex-
   combined) bias key (`ec_merge_aligned` aligned them); two that learned DIFFERENT
   contingencies fold to a UNION (kept distinct). Asserted on a pilot fold from provenance
   (`biases_rekeyed`, the `id_map`, the **shared-vs-union key counts reported quantitatively**
   so the widening-vs-dilution balance is observed, not assumed) — an over-aligning merge
   (collapses coverage → false flat) or under-aligning merge (a union that never averages)
   is an apparatus failure caught here.
2b. **Contributor divergence (the coverage-widening tripwire — the review's second blocker).**
   At the pilot per-agent budget, compute the **mean pairwise Jaccard of the contributors'
   decisively-covered contingency SETS** and assert the UNION of covered contingencies grows
   from N = 1 to the pilot N. If independent-seed contributors all cover the same set in the
   same order (Jaccard ≈ 1, union flat), the ladder is guaranteed flat for a reason readable
   in advance — an apparatus failure (widen the per-contributor order variance) fixed before
   any rung, never discovered at campaign price.
3. **L12 zero-prior.** `score_components["drive"] == 0.0` on every Phase-0 probe decision
   (asserted, as Exp 56); the campaign asserts the same per probe (S3).
4. **Calibration — K_max, C, W (the "partway / below-ceiling" requirement).** On a single
   taught contributor, measure the coverage-vs-per-agent-trial curve. Choose K_max so a
   single contributor at K_max sits **partway** — coverage strictly below C, a partial
   learner that does NOT saturate (design target: N = 1 reaches ≈ ½ = 2 of 4) — and C below
   the coverage ceiling so pooled crèches cross it with headroom (design target **C = ¾ = 3
   of 4**). W smooths a single noisy checkpoint (design target **W = 3**). Because coverage
   is read DETERMINISTICALLY (bias-decisive from provenance, §DV), the L2 concern is not
   ε-dither reaching the agent but **cohort-to-cohort coverage variance** — the isolated
   (N = 1) coverage must NOT be seed-invariant across cohorts (varying contributor seeds pick
   different covered sets); asserted here and, per the L2 gate, at every rung in the
   confirmatory analysis. Exact numbers set from this curve and disclosed.
5. **Pilot ladder (obtainability + spread, NOT count-setting).** **≥ 5 cohorts** end-to-end
   at N ∈ {1, 8} through the real export→fold→probe path (plus the anti-vacuity variants),
   confirming τ(8) < τ(1) is OBTAINABLE and the plumbing holds, and giving a τ-variance
   sanity check against the a-priori-fixed ≥ 20 count (one cohort would give n = 1/rung and
   no spread — the count is fixed by this file, not chosen from the pilot). The pilot's
   per-contingency readout must be bias-decisive with `|learned_margin| > 0.11` (L1's floor)
   on the contingencies it covers.

Two Phase-0 failures with NEW modes → stop, bird's-eye audit (the divergence rule); 1.2
does not ship this claim on schedule.

## Outcome tree (decided now)

| outcome | reading | action |
|---|---|---|
| Phase 0 fails check 1/2 | the 4 slots do not separate pairwise, or the merge mis-aligns | campaign does not run; apparatus/design fix (re-place slots / re-derive geometry); re-run Phase 0; each retry recorded |
| Phase 0 fails check 2b | contributors cover the same set in the same order — coverage cannot widen with N | apparatus fix (widen per-contributor seed/order variance); re-run; a ladder that cannot widen is flat by construction, not by finding |
| Phase 0 fails check 3 | L12 leak | body/name fix; re-run |
| Phase 0 fails check 4 | no partway/below-ceiling window exists at G = 4 (single agent saturates or floors, or coverage too coarse) | re-choose K_max/C, or pre-register ADDING slots (G > 4, apparatus amendment); re-run; if no window exists the coverage DV is wrong for this body → bird's-eye |
| Phase 0 fails check 5 | plumbing broken or effect not obtainable at N = 8 vs 1 | fix plumbing; if τ(8) ≈ τ(1) in the pilot, that is an early falsifier signal — recorded, campaign may still run to quantify it |
| MONOTONICITY (JT + censoring-artifact guards) + NOT-JUST-MORE-DATA + NOISE-FLOOR pass, ANTI-VACUITY + L2 clean | **the 1.2 scaling claim, earned.** Pooling independent partial learners reaches criterion in fewer per-agent trials as N grows, via coverage, at no worse total-experience cost than one agent. | new graduation row (Earned) citing the committed data; interpretation write-up in a separate later PR (structure-or-time); the two-Reachy one-rung replication prereg is unblocked |
| JT significant but a censoring-artifact guard fails (effect vanishes when censored rungs are dropped, or endpoint coverage does not rise with N) | the "trend" was a two-point censored step, not graded coverage widening | NOT a pass — recorded as an artifact; audit target is K_max (too small → over-censored) and the endpoint-coverage curve |
| MONOTONICITY passes, NOT-JUST-MORE-DATA fails | pooling scales per-agent trials but costs total experience (merge overhead) | PARTIAL, named; the audit target is the merge-cost source (convex-combination dilution, alignment loss); owner call at the release checkpoint |
| MONOTONICITY flat (JT n.s.) | collective learning buys nothing on this apparatus | FALSIFIER — ships as a result; audit target is the coverage mechanism (shared-vs-union counts, the check-2b Jaccard) + `ec_merge_aligned` alignment; a coverage that never widens with N points at over-alignment or insufficiently-divergent contributors |
| NOISE-FLOOR fails (untaught crèche gains coverage) | pooling manufactures coverage from uncredited substrate | STOP — not a science result; find the leak (link channel? the Exp 56 §Links containment); fix with a guard test; re-run once |
| ANTI-VACUITY / L2 fails | the gate cannot fail / effective n collapsed | no verdict; fix + the one permitted re-run |

**Stop rule:** the confirmatory campaign runs ONCE (plus the single re-run the
NOISE-FLOOR-leak / L2 / contaminated-anti-vacuity branches allow — one re-run total). A
second campaign-level divergence ends Exp 57 for 1.2: the result ships as it stands.

## What this experiment does NOT claim

- **Hardware.** The robot replicates ONE rung, not the ladder (N = 8 is affordable only in
  Minecraft). The two-Reachy one-rung replication is its own prereg on Exp 54's body.
- **A from-scratch sequence model / substrate-primary bootstrap.** 1.3.
- **Cross-layout generalization.** Contributors train and the receiver probes against the
  SAME seeded G-contingency world; "the crèche learns a situation it never saw" is not
  claimed.
- **Aversion / negative-valence pooling.** Positive folds only (tighten-only exercised as
  byte-untouched positive pass-through, as Exp 56). Exp 55, 1.3-line.
- **A specific scaling LAW.** The gate is monotone decrease, not a functional form (log,
  1/N, etc.); fitting a curve shape is post-hoc description, reported not gated.
- **Limits declared inapplicable by name** (the ledger discipline): L4 (`safe_pref`
  saturation — the whole point of this experiment is a RATE measure replacing the saturated
  endpoint; coverage is chosen to be unsaturated by Phase-0 check 4); L6 (prior-agreement
  ceiling — no LLM in the action path; the L12 channel is the applicable analog and is
  gated); L9/L10 (DoA instruments — none here).

## Apparatus declarations (S1–S8)

- **S1:** none change mechanism — the campaign consumes shipped, guarded paths
  (`substrate_merge` + the V1–V10 adapter, the operant credit path, the Exp 56 apparatus).
  The G-contingency world script, the N-AUT sequencing, and `scripts/exp57/` are
  experiment-local assembly reusing Exp 56's `scripts/exp56/common.py`. Any `src/` mechanism
  change proves necessary → declared amendment + its own reviewed PR before the campaign.
- **S3 in-sim assertions (refusal exit ≠ 0):** the Exp 56 donor-sanity set per contributor;
  per-probe `score_components["drive"] == 0.0`; the independence set (all N + receiver ids
  distinct, disjoint pre-fold clusters, receiver-id post-fold keys); the merge-alignment
  assertion (check 2, per fold); NOISE-FLOOR contributors carry zero credits.
- **S4:** per-checkpoint JSONL (N, t, coverage, per-contingency detail, fold reports,
  alignment maps) + Phase-0 records committed under `docs/experiments/data/57_*` with
  provenance stamps; workdirs on durable storage.
- **S5 exposure contract:** every contributor (taught and untaught) executes the identical
  balanced-exposure SCHEDULE STRUCTURE (each contingency scheduled equally often — the
  link-neutralization) against the identical G = 4 world, but with an **independent
  per-contributor seed AND presentation order** (§Apparatus — this is what lets coverage
  widen; an identical seed would make contributors byte-identical and force a flat curve);
  single_matched runs the same schedule extended to N×K_max; the receiver faces the
  identical per-cohort probe battery across all conditions and rungs, read through the same
  fold path (creche N→1, single_matched 1→1).
- **S6:** no fidelity toggles differ between conditions; `MAXIM_OPERANT_ONLY_CREDIT=1`
  everywhere; the harness refuses ambient `MAXIM_*` disagreement (the Exp 52 exit-3 pattern).
- **S7:** the criterion C sits below the coverage ceiling and above the floor (Phase-0
  check 4); censored τ concentrates at SMALL N and ranks largest, which REINFORCES the
  decreasing alternative (anti-conservative), so the JT pass is gated on the two
  censoring-artifact guards (drop-censored-rungs sensitivity + a graded uncensored endpoint
  trend, §Gates), never on JT alone; δ_eff = 0 is frozen here, not fit to the pilot;
  NOT-JUST-MORE-DATA compares TOTAL experience through the SAME fold path so "N agents did
  more" and "merged-vs-unmerged readout" cannot masquerade as efficiency.
- **S8:** one harness at a time; not co-located with a leader.

## Amendment rule

Amendments after first data (Phase 0's included) are permitted only for *structural
invalidity* — harness bug, degenerate metric, an apparatus-gate failure, or the Phase-0
apparatus-constant calibration this file explicitly defers — never for effect size; every
amendment is its own PR merged before the data it governs, with the read-state disclosure
the Exp 52/56 precedent set. The GATE constants are frozen at the commit that lands this
file — the four gate rows, **δ_eff = 0**, p < 0.05, the rung set {1,2,4,8}, and the ≥ 20
seed count; the APPARATUS constants left open (K_max, C, W, and G if raised above 4 by a
slot-adding amendment) are frozen by the Phase-0 amendment, disclosed. δ_eff and the seed
count are deliberately NOT in the Phase-0 list — setting a gate margin or a power target
from pilot data is the forbidden retune.

## Runbook (shape; exact flags frozen with the harness PR)

```bash
export PYTHONPATH=$PWD/src
# Phase 0 (instrument checks + calibration; commits 57_phase0.json, sets apparatus constants)
python scripts/exp57/instrument_check.py --write-experiment-results

# Confirmatory ladder (N ∈ {1,2,4,8} × ≥20 cohorts; --mock for the smoke)
python scripts/exp57/run_ladder.py --rungs 1,2,4,8 --cohorts 20 --seed-base 42 \
  --out docs/experiments/data/57_ladder.jsonl --write-experiment-results
python scripts/analyze_exp57.py --in docs/experiments/data/57_ladder.jsonl --gate v1 \
  --assert-noop-fails
```

## Sign-off (fills before the campaign; each box its own merged PR where marked)

- [ ] This pre-registration merged to `main` via merge commit (never squash) — hash: `____`
- [x] Two-lens review round on THIS draft folded pre-freeze (design/methodology +
      code-accuracy) — the Exp 56 discipline; both lenses returned FIX-FIRST, all three
      cross-/single-lens blockers + the HIGH/MODERATE refinements folded (see the Status
      header for the folded findings)
- [ ] Harness PR merged with guard tests (`scripts/exp57/` reusing `scripts/exp56/common.py`;
      the G = 4 world script + the N-AUT contributor sequencing with independent per-contributor
      seed/order; `scripts/analyze_exp57.py` with frozen gate constants incl. the JT
      permutation trend test + the two censoring-artifact guards, δ_eff = 0, the noise floor,
      and `--assert-noop-fails`; `--mock`/`--resume`) — hash: `____`; frozen GATE constants
      recorded: rungs `{1,2,4,8}`, δ_eff `0`, p `0.05`, cohorts `≥20`; apparatus constants
      set in Phase 0: G `4` (or the amended slot count), K_max `__`, C `__` (target ¾), W `__` (target 3)
- [ ] Phase 0 run + committed (`57_phase0.json`): checks 1, 2, 2b, 3, 4, 5 PASS; apparatus
      constants set + disclosed in the Phase-0 amendment (gated, clean tree, main-reachable)
- [ ] Confirmatory ladder run ONCE from a clean tree at a main-reachable commit; data PR
      merge-committed; interpretation in a separate later PR (structure-or-time rule)
