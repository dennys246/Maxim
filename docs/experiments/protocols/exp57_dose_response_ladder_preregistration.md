# Exp 57 — The dose–response ladder: does collective learning scale? (pre-registration)

**Status:** DRAFT — NOT YET FROZEN. Freezes at the merge commit of this file, before any
harness run, AFTER a two-lens review round (design/methodology + code-accuracy) folds
pre-freeze, matching the Exp 56 discipline (its two cross-confirmed criticals were
designed against, not discovered at campaign price). The 1.2 **second** claim — the four
arms ([Exp 56](../56_four_arm_sharing.md), EARNED 2026-09-06) asked *does a taught want
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
`cluster_reward_bias` folds via `_merge_mean_clamped`: **shared keys average** (two
contributors that learned the SAME `(agent, cluster, tool)` bias key produce the mean of
their two values — NOT their sum), and **keys unique to one contributor are kept**
(zero-prior rule). Therefore:

- Averaging N partial-but-correct biases on ONE contingency does **not** grow the
  magnitude on that contingency — pooling cannot scale by "louder wants".
- What pooling grows is **coverage**: contributor *i*, sample-limited to a per-agent budget
  K, gets a decisive bias on only SOME of the world's taught contingencies; the union over
  N contributors covers MORE contingencies, so the merged receiver reads out correctly on
  more of them. This is the precedent's own mechanism (Exp 45d cell-starvation: one agent
  covers some state cells, N agents cover more, the merge unions them) and the reason the
  scripted crèche reached `single_full` while a single partial learner did not.

**Consequence for the apparatus (the load-bearing design decision):** the world must
present **G distinct taught contingencies** (situations each with their own correct
affordance), not Exp 56's single target. The dependent measure is **coverage** — the
fraction of the G contingencies the merged receiver reads out correctly — and
trials-to-criterion is the per-agent budget at which pooled coverage first crosses the
criterion. A single-contingency world (Exp 56's) would make the ladder measure magnitude,
which the merge does not scale, guaranteeing a false-flat curve for a reason readable in
advance.

**The independent-agent wrinkle (gate 6 / D43 territory, now closed).** Independent agents
encode the SAME situation to DIFFERENT cluster UUIDs; `ec_merge` (inside `substrate_merge`)
aligns them by cosine and re-keys the biases. So two contributors that BOTH learned
contingency *g* have their *g*-clusters aligned → their bias keys become shared → averaged
(coverage of *g* confirmed, magnitude unchanged); two contributors that learned DIFFERENT
contingencies have non-aligning clusters → kept as a union (coverage widened). The ladder
therefore exercises exactly the re-keyed merge Exp 56 validated at N = 1; Phase 0 check 2
below verifies alignment behaves (same-situation clusters align, distinct-situation
clusters stay distinct) on the G-contingency world before any rung runs — an
over-aligning merge would collapse coverage and a false-flat curve would follow.

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

**Coverage at a checkpoint.** For a merged receiver, probe it once at first contact against
EACH of the G contingencies (the Exp 56 first-contact readout, per contingency: the frozen
ε-greedy selector's choice from the `NAc_RECOMMEND` provenance, counted correct only when
the taught affordance wins AND the LEARNED-BIAS component is decisive — the §Links
assertion, carried from Exp 56). Coverage = (# contingencies read out correctly) / G. Read
at the REAL consumer (the action the loop proposes), never from dict contents (D44's rule).

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
`ec_merge` alignment counts per fold.

## Conditions per rung (frozen; the precedent's controls, generalized to a curve)

N ∈ **{1, 2, 4, 8}**. At each rung, three conditions, all reading out through the identical
probe battery:

| condition | construction | question |
|---|---|---|
| **creche(N)** | N taught contributors × per-agent trials, folded N→1 at each checkpoint | the test — does pooled partial learning reach criterion, and at what per-agent t |
| **single_matched(N)** | ONE taught contributor given up to N×K_max trials; τ measured in ITS OWN trial index | is pooling at least as efficient in TOTAL experience as one agent seeing everything? (the precedent's `single_full`, made a rate) |
| **creche_none(N)** | N contributors on the balanced schedule with the teacher WITHHELD (zero credits by mechanism — the Exp 56 satiated construction), folded N→1 | noise floor at that rung — pooling uncredited substrate must not manufacture coverage |

N = 1 is the shared baseline: creche(1) ≡ single_matched(1) ≡ one taught contributor at
its own budget. `creche_none` generalizes Exp 56's satiated/dangling controls to the ladder
(the falsifier's job here is the flat-curve clause, not a separate dangling arm — the
dangling-half falsifier was discharged at N = 1 by Exp 56 and is not re-litigated).

## Pre-registered gates (constants carried from the frozen design; numeric apparatus constants set in Phase 0, §Phase 0)

| gate | rule |
|---|---|
| **MONOTONICITY** (primary) | median τ(creche(N)) strictly decreases across N = 1 → 8, tested by the **Jonckheere–Terpstra** ordered-alternative trend test (decreasing alternative) over the per-seed τ values at the four rungs, **p < 0.05** one-sided. Right-censored τ (did not reach C within K_max) rank as the largest value (a conservative direction for a decreasing-trend test: censoring at small N works AGAINST the alternative). JT is chosen over Spearman because N has four ordered levels with replicate seeds — the ordered-alternative form is the matched statistic ([[match-the-statistic-to-the-baseline]] discipline). |
| **NOT-JUST-MORE-DATA** | at each rung N ≥ 2, creche(N) TOTAL experience to criterion (N × τ(creche(N))) ≤ single_matched(N) trials-to-criterion **within a declared margin δ_eff** (frozen in Phase 0). If pooling costs MORE total experience than one agent seeing everything, that is an honest merge-cost finding and ships as one — it does not fail the primary, it qualifies it. |
| **NOISE-FLOOR** | creche_none(N) coverage stays at chance (never reaches C) at EVERY rung; per-contingency it sits at the ε-greedy floor ≈ 1/k. |
| **FALSIFIER** (stated, not a pass/fail row) | τ(creche(N)) flat in N (JT not significant) means collective learning buys nothing on this apparatus. A flat curve is a RESULT and ships as one, with the audit target being the coverage mechanism (shared-vs-union key counts per fold) and `ec_merge` alignment. |
| **ANTI-VACUITY** (apparatus; no verdict without it) | the analyzer's no-op merge kit (Exp 56's, generalized): folding with `substrate_merge` replaced by *receiver-unchanged* and *empty-fold* variants MUST collapse coverage to the floor at every rung; a *contributors-re-keyed-alone* fold (one contributor, re-keyed, no averaging partner) is RECORDED and expected to persist (D62 / Exp 56 amendment 1). A gate that cannot fail is not a gate. |

The scaling claim requires MONOTONICITY **and** NOT-JUST-MORE-DATA **and** NOISE-FLOOR, with
ANTI-VACUITY + the L2 gate (§Phase 0) clean. MONOTONICITY passing while NOT-JUST-MORE-DATA
fails is a PARTIAL ("pooling scales in per-agent trials but costs total experience"), named
and shipped or held at the release checkpoint — owner call, not a silent pass.

**Power:** ≥ **20 independent seed-paired cohorts per rung** (design target; each cohort a
fully independent draw of the N contributors + the single_matched agent + the untaught
crèche at that rung). Report per-rung median τ with a bootstrap 95% CI; the JT test uses the
per-seed τ values. The seed count is confirmed adequate in Phase 0 by a power check against
the pilot effect (§Phase 0 check 5); it is not retuned from confirmatory data.

## Apparatus (what must exist at the frozen commit; owed items in the sign-off)

Everything Exp 56 froze is reused UNCHANGED and by reference — the body
`bodies/minecraft_bench{,_satiated}`, the harness teacher + balanced schedule, the ε-greedy
selector regime (ε = 0.2, `min_confidence = 0.3`, `substrate_explore_bonus_weight = 0` at
the probe, per-pair name permutation), `MAXIM_OPERANT_ONLY_CREDIT=1` in every arm and phase,
the export/ingest flags (no `--allow-unstamped-geometry`, no `--force-digest`, no
`--inherent-trust`), and the S1–S8 declarations. New or generalized for the ladder:

- **G taught contingencies (not one).** The bench body's 8 opaque affordances become
  **G = 8** situation→correct-affordance contingencies: each of the world's frozen far+high
  slots is a distinct situation whose correct action is a distinct affordance (the
  per-pair permutation maps slots→affordances so the alphabet is not load-bearing). The
  teacher credits the correct affordance for whichever contingency is active. Coverage is
  read out over all G. The world script presents the G contingencies in a seeded order; the
  slot geometry stays the far+high placement Exp 56's amendment 2 fixed (each situation
  separates from rest and from the other situations — Phase 0 check 1 verifies pairwise
  separation over the G, not just rest-vs-one).
- **N-AUT generalization of the 1.1.4 two-AUT harness.** The A-phase runs N contributor
  sessions **sequentially against the live bridge** (one client per server/bridge, the Exp
  56 constraint; instances parallelize across ports). Contributors are snapshotted per
  trial; the B-phase folds and probes offline against the same seeded world script. Nothing
  needs N agents in one world at once (the shared-world fake-bridge gap is off this path,
  as in Exp 56).
- **Independent contributors (D44), asserted per cohort (S3):** all N `agent_id` distinct
  and distinct from the receiver; each contributor's EC constructed separately; pre-fold the
  receiver holds zero contributor cluster ids; post-fold, landed bias keys carry the
  RECEIVER's agent id. The `ec_merge` alignment map is recorded per fold (the coverage
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

Run gated on the frozen body + G-contingency world script before any rung; recorded and
committed like any gated record; the numeric apparatus constants (K_max, C, W, δ_eff, and
the seed count's adequacy) are CHOSEN here and frozen by a pre-campaign amendment with the
disclosure the Exp 52/56 precedent set. **Gate constants are NOT retuned from Phase-0 data**
— only the apparatus constants the design left open, and every reading is disclosed. All
five must pass or the campaign does not start:

1. **G-contingency discriminability (L11/A4).** Over the G situations, every pair separates
   (pairwise cos below the single-cluster bound; separation ≥ 0.70 on each rest→situation
   contrast) and repeated probes re-complete into the SAME cluster (stability ≥ 0.70). This
   is Exp 56 check 1 widened from one situation to G — a G-contingency world is only a
   coverage instrument if the G situations are actually distinguishable.
2. **Merge-alignment sanity (the independent-agent wrinkle).** Two independent contributors
   that each learned the SAME contingency fold to a SHARED (averaged) bias key
   (`ec_merge` aligned them); two that learned DIFFERENT contingencies fold to a UNION (kept
   distinct). Asserted on a pilot fold from provenance (`biases_rekeyed`, alignment map, the
   shared-vs-union key counts) — an over-aligning or under-aligning merge is an apparatus
   failure that would corrupt coverage, caught here before any rung.
3. **L12 zero-prior.** `score_components["drive"] == 0.0` on every Phase-0 probe decision
   (asserted, as Exp 56); the campaign asserts the same per probe (S3).
4. **Calibration — K_max, C, W (the design's "partway / below-ceiling" requirement).** On a
   single taught contributor, measure the coverage-vs-per-agent-trial curve. Choose K_max so
   a single contributor at K_max sits **partway** (coverage strictly below C — a partial
   learner that does NOT saturate), and C **below the ε-greedy coverage ceiling** so pooled
   crèches can cross it with headroom. W is set so a single noisy checkpoint cannot trip the
   criterion (design target W = 3). The isolated (N = 1) curve must be non-seed-invariant
   (the L2 dither check — carried from Exp 56 check 4). Design targets, to be confirmed and
   frozen here: **G = 8, C = 0.75 coverage, K_max ≈ the per-agent budget at which N = 1
   reaches ≈ 0.4 coverage, W = 3** — the exact numbers are set from this curve and disclosed.
5. **Pilot ladder + power.** ONE cohort end-to-end at N ∈ {1, 8} through the real
   export→fold→probe path (plus the anti-vacuity variants), confirming τ(8) < τ(1) is
   OBTAINABLE and the plumbing holds, and a power check fixes the confirmatory seed count
   (≥ 20 design target) against the pilot's τ spread so the JT test is adequately powered.
   The pilot's per-contingency readout must be bias-decisive with `|learned_margin| > 0.11`
   (L1's floor) on the contingencies it covers.

Two Phase-0 failures with NEW modes → stop, bird's-eye audit (the divergence rule); 1.2
does not ship this claim on schedule.

## Outcome tree (decided now)

| outcome | reading | action |
|---|---|---|
| Phase 0 fails check 1/2 | the G-contingency instrument cannot see the contingencies apart, or the merge mis-aligns | campaign does not run; apparatus/design fix; re-run Phase 0; each retry recorded |
| Phase 0 fails check 3 | L12 leak | body/name fix; re-run |
| Phase 0 fails check 4 | no partway/below-ceiling window exists (single agent saturates or floors) | re-choose K_max/C/G; re-run; if no window exists the coverage DV is wrong for this body → bird's-eye |
| Phase 0 fails check 5 | plumbing broken or effect not obtainable at N = 8 vs 1 | fix plumbing; if τ(8) ≈ τ(1) in the pilot, that is an early falsifier signal — recorded, campaign may still run to quantify it |
| MONOTONICITY + NOT-JUST-MORE-DATA + NOISE-FLOOR pass, ANTI-VACUITY + L2 clean | **the 1.2 scaling claim, earned.** Pooling independent partial learners reaches criterion in fewer per-agent trials as N grows, via coverage, at no worse total-experience cost than one agent. | new graduation row (Earned) citing the committed data; interpretation write-up in a separate later PR (structure-or-time); the two-Reachy one-rung replication prereg is unblocked |
| MONOTONICITY passes, NOT-JUST-MORE-DATA fails | pooling scales per-agent trials but costs total experience (merge overhead) | PARTIAL, named; the audit target is the merge-cost source (averaging dilution, alignment loss); owner call at the release checkpoint |
| MONOTONICITY flat (JT n.s.) | collective learning buys nothing on this apparatus | FALSIFIER — ships as a result; audit target is the coverage mechanism (shared-vs-union counts) + `ec_merge` alignment; a coverage that never widens with N points at over-alignment |
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
- **S5 exposure contract:** every contributor (taught and untaught) executes the IDENTICAL
  seeded balanced schedule against the identical G-contingency world; single_matched runs
  the same schedule extended to N×K_max; the receiver faces the identical per-cohort probe
  battery across all conditions and rungs.
- **S6:** no fidelity toggles differ between conditions; `MAXIM_OPERANT_ONLY_CREDIT=1`
  everywhere; the harness refuses ambient `MAXIM_*` disagreement (the Exp 52 exit-3 pattern).
- **S7:** the criterion C sits below the ε-greedy coverage ceiling and above the floor
  (Phase-0 check 4); censored τ ranks conservatively against the decreasing alternative;
  NOT-JUST-MORE-DATA compares TOTAL experience so "N agents did more" cannot masquerade as
  efficiency.
- **S8:** one harness at a time; not co-located with a leader.

## Amendment rule

Amendments after first data (Phase 0's included) are permitted only for *structural
invalidity* — harness bug, degenerate metric, an apparatus-gate failure, or the Phase-0
apparatus-constant calibration this file explicitly defers — never for effect size; every
amendment is its own PR merged before the data it governs, with the read-state disclosure
the Exp 52/56 precedent set. The GATE constants (the four rows above) are frozen at the
commit that lands this file; the APPARATUS constants (K_max, C, W, G, δ_eff, seed count) are
frozen by the Phase-0 amendment, disclosed.

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
- [ ] Two-lens review round on THIS draft folded pre-freeze (design/methodology +
      code-accuracy; both reports in the round record) — the Exp 56 discipline
- [ ] Harness PR merged with guard tests (`scripts/exp57/` reusing `scripts/exp56/common.py`;
      the G-contingency world script; `scripts/analyze_exp57.py` with frozen gate constants
      incl. the JT trend test, the NOT-JUST-MORE-DATA margin, the noise floor, and
      `--assert-noop-fails`; `--mock`/`--resume`) — hash: `____`; frozen constants recorded:
      rungs `{1,2,4,8}`, G `__`, K_max `__`, C `__`, W `__`, δ_eff `__`, cohorts `__`
- [ ] Phase 0 run + committed (`57_phase0.json`): checks 1–5 PASS; apparatus constants set +
      disclosed in the Phase-0 amendment (gated, clean tree, main-reachable)
- [ ] Confirmatory ladder run ONCE from a clean tree at a main-reachable commit; data PR
      merge-committed; interpretation in a separate later PR (structure-or-time rule)
