# Exp 57 — The dose–response ladder: pooling scales per-agent learning, at a total-experience cost

**Status: PARTIAL 2026-09-08.** The 1.2 "Oasis" scaling claim, run as its pre-registered
may-fail second claim. Pooling the partial learning of N genuinely-independent agents lets
each agent reach criterion in **fewer of its own trials as N grows** — the per-participant
sample-efficiency the collective sells — but the pool spends **more total experience** than
one agent given all of it. That conjunction is the pre-registered PARTIAL branch: an honest
merge-cost qualification, named, not a silent pass.

- **Pre-registration** (GATE constants frozen before any run):
  [protocols/exp57_dose_response_ladder_preregistration.md](protocols/exp57_dose_response_ladder_preregistration.md).
  Amendment 1 PRE-DATA (the direction-aware `minecraft_bench57` world channel); amendments
  2–3 POST-DATA (the Phase-0 shakedown harness fixes + the K_max/C/W freeze). The GATE
  constants — rungs {1,2,4,8}, δ_eff = 0, p < 0.05, ≥ 20 cohorts — were frozen at the
  prereg merge, not from Phase-0 data.
- **Data** (gated, committed): [data/57_ladder.jsonl](data/57_ladder.jsonl) (9,200 rows,
  4 rungs × 3 conditions × 20 cohorts) + [data/57_ladder_verdict.json](data/57_ladder_verdict.json)
  (analyzer verdict) + [data/57_phase0.json](data/57_phase0.json) (the six Phase-0 checks).
  Confirmatory campaign run ONCE at `main`-reachable `d6f12b1f` (the freeze commit), clean
  tree, `mock: false`, ~91 min on the operator's big-mac-mini.
- **Harness:** [scripts/exp57/](../../scripts/exp57/) (`run_ladder.py` / `instrument_check.py`
  / `common57.py`, reusing [scripts/exp56/common.py](../../scripts/exp56/common.py)) +
  [scripts/analyze_exp57.py](../../scripts/analyze_exp57.py) (frozen gate constants) + body
  [minecraft_bench57.yaml](../../src/maxim/_data/components/bodies/minecraft_bench57.yaml).

## The question

Exp 56 proved a taught want transfers from one agent to one other through the shipped 1.2
ingestion path. Exp 57 asks the scaling question directly: if N independent agents each
learn the SAME contingency only *partially* (a single agent at K_max sits below criterion —
it covers some of the situation, not all of it), does folding their substrates together let
a receiver reach criterion in fewer per-agent trials as N grows — and does pooling cost no
more *total* experience than one agent seeing everything?

The two halves are separate gates by pre-registration, because they can come apart:

- **MONOTONICITY** (primary): per-agent trials-to-criterion τ(creche(N)) fall across
  N = 1 → 8, by the Jonckheere–Terpstra ordered-trend test with a permutation null, plus
  two censoring-artifact guards.
- **NOT-JUST-MORE-DATA**: at each rung, the crèche's TOTAL experience to criterion
  (N × τ(creche(N))) is no worse than `single_matched(N)` — one agent given N × K_max
  trials — within the frozen margin δ_eff = 0. If pooling costs MORE total experience, the
  prereg says that "does not fail the primary, it qualifies it."

The contributors are independent in D44's sense (distinct `agent_id`, separately
constructed EC + SensorEncoder, disjoint cluster ids). When their g-clusters align through
`ec_merge_aligned`, their bias keys become shared and are **convex-combined** by the
left-associative `substrate_merge` fold — for three contributors, weights (¼, ¼, ½). Because
the combination is convex (bounded above by the largest contributor), pooling cannot scale
by "louder wants": it can only scale by **coverage** — N partial-but-correct biases spanning
more of the situation than any one alone. The widening-vs-dilution race is the whole game,
and it is what the gates observe rather than assume.

## Apparatus (as run)

- **World:** the LIVE Minecraft bridge (frozen NDJSON protocol) against the real offline
  Paper 1.16.5 server, superflat/void, the four frozen contingency slots placed by RCON per
  the seeded script — the same apparatus Exp 56 proved, reused.
- **Body:** `bodies/minecraft_bench57` — the offsets-only world channel (`offset_x` /
  `offset_z`, signed position from spawn) that makes the four slots pairwise-distinct through
  the real SensorEncoder (margin 0.478), with opaque affordances `aff_a`…`aff_h` and opaque
  drive `d1` preserved (the L12 zero-prior channel intact).
- **Contributors:** each of N donors runs a seeded, exposure-balanced schedule to K_max = 20
  per-agent trials; teacher-minted relief-signed operant credit lands each on
  `(agent, WORLD-cluster, tool:<target>)`. A single contributor at K_max is calibrated to
  sit **partway** — below C — so pooling has something to add.
- **Fold:** contributors fold N → 1 at each checkpoint through the real `substrate_merge`
  ingest path (the same left-associative ¼¼½ convex combination the product ships), never
  bare `nac_merge`.
- **Conditions (per rung):** `creche(N)` (the test — pooled partial learners); `single_matched(N)`
  (one agent given N × K_max trials — the total-experience baseline); `creche_none` (the
  noise floor — pooled agents that learned nothing).
- **DV:** per-agent trials-to-criterion τ = first checkpoint at which bias-decisive endpoint
  coverage reaches C = 0.75 over a window W = 3, read from the `NAc_RECOMMEND`
  decision-provenance record (the real consumer), right-censored at the sentinel K_max + 1.

## Phase 0 — instrument checks (all six PASS, then frozen)

Run gated on the live apparatus ([data/57_phase0.json](data/57_phase0.json), `main`-reachable
`db6fd598`): (1) discriminability — the four slots form four pairwise-distinct clusters
(separation 1.0, stability 1.0); (2) alignment/divergence — folded contributors share the
taught key while their unique coverage unions (divergence Jaccard 0.33); (3) drive-zero —
`score_components["drive"] == 0` on every probe; (4) calibration — the coverage curve puts a
single contributor partway at K_max (0.25 @ ~12, 0.5 @ ~20, 0.75 @ ~34, 1.0 @ ~60);
(5) pilot ladder — τ₈ median 12 « τ₁ median 46. On the strength of these, the prereg's
Amendment 3 froze **K_max = 20, C = 0.75, W = 3** for the confirmatory ladder.

## Results — 20 cohorts per rung × condition (censor sentinel = K_max + 1 = 21)

| N | crèche τ (per-agent, median) | endpoint coverage d(N, K_max) | `single_matched` τ | N × crèche τ vs `single_matched` |
|---|---|---|---|---|
| 1 | 21.0 — never reaches criterion (fully censored) | 0.25 | 21.0 (censored) | — |
| 2 | 21.0 (mostly censored) | 0.50 | 41.0 | 42 vs 41 |
| 4 | **15.5** | 0.75 | 46.0 | 62 vs 46 |
| 8 | **10.5** | 0.75 | ~43 (reaches at 43) | 84 vs 43 |

A lone agent (N = 1) covers only ¼ of the situation at K_max and **never** reaches the
3-of-4 criterion in 20 trials. Pooling widens coverage — 0.25 → 0.50 → 0.75 → 0.75 — and an
agent inside a crèche of 8 reaches criterion in ~10.5 of its own trials. Meanwhile one agent
given all the trials reaches criterion at ~43–46 (uncensored once its budget ≥ 46).

## The gates

| gate | rule | result | verdict |
|---|---|---|---|
| **MONOTONICITY** (primary) | τ(creche(N)) falls across N by JT (permutation null, p < 0.05) **and** both censoring-artifact guards | JT p ≈ 1e-4 (stat 2056.5); **guard i** drop-censored p ≈ 1e-4 (stat 1086.5); **guard ii** endpoint coverage rises 0.25→0.50→0.75→0.75 (Spearman ρ 0.949) | ✅ PASS |
| **NOT-JUST-MORE-DATA** | N × τ(creche(N)) ≤ `single_matched(N)` at δ_eff = 0, every rung N ≥ 2 | 42 > 41, 62 > 46, 84 > 43 | ❌ FAIL |
| **NOISE-FLOOR** | `creche_none` coverage stays below C | 0.0 at every rung | ✅ PASS |
| **L2_SEED_VARIANCE** | per-seed spread present, N = 1 concentration ≤ 0.9 | distinct {0.0, 0.25, 0.5}, concentration 0.75 | ✅ PASS |
| **ANTI-VACUITY** | must-collapse no-op merge variants collapse; rekeyed-alone recorded | `kit_pass: true` (rekeyed-alone coverage 0.25) | ✅ PASS |

The primary passes in its **robust** form: not a two-point censored step at the bottom rung,
but graded coverage widening (guard ii) that survives dropping the fully-censored rungs
(guard i). `verdict: PARTIAL, problems: []`.

## What this shows — and what it does NOT

**Shows:** collective learning is a **per-participant win and a parallelism win**. Joining a
crèche roughly quarters the trials each agent personally needs (τ 21→10.5 from N=1 to N=8),
by widening the coverage of a single shared contingency through the real convex-combination
fold — the primary claim the apparatus was built to test, earned in its robust form.

**Does NOT (the pre-registered qualification):** pooling is **not a total-sample free lunch**.
At every rung the crèche spends more aggregate experience to reach criterion than one agent
given all of it (N × τ = 42/62/84 vs 43–46). The cost is the **convex-combination merge
overhead** the prereg named as the audit target: averaging N partial biases (weights bounded
above by the largest contributor) recovers coverage but loses some of the signal one agent
accumulates undiluted. This is a real property of the shipped `substrate_merge` fold, not an
apparatus artifact — the fold is left-associative ¼¼½ convex combination by design.

**Honest scope in one line:** one campaign, one world layout, substrate-primary, in
Minecraft, 20 cohorts/rung — pooling buys each participant faster learning and parallelism,
at a total-experience cost; whether a different merge (coverage-preserving rather than
mean-clamped) closes that cost is 1.3-line work, not claimed here.

Note the framing the metric fixes: NOT-JUST-MORE-DATA compares *total serial experience*.
The crèche's 84 agent-trials at N = 8 are spent across 8 agents in parallel (10.5 each),
while the baseline's 43 are one agent serially — so in wall-clock/parallel terms pooling can
still win. The gate is deliberately the stricter *sample-cost* comparison; the parallelism
advantage is real but is not what this gate measures.

## Owner call at the release checkpoint

The prereg makes MONOTONICITY-passes-while-NOT-JUST-MORE-DATA-fails an explicit PARTIAL:
"named; the audit target is the merge-cost source (convex-combination dilution, alignment
loss); owner call at the release checkpoint." Decision (2026-09-08): **ship in 1.2 as a named
PARTIAL** — the per-participant scaling win is earned in its robust form, and the
merge-cost qualification is a stronger, more honest story than an unqualified pass. The
audit target (a coverage-preserving fold) is 1.3-line work, tracked, not back-fitted here.

## Provenance

The confirmatory campaign ran ONCE (the stop rule), clean tree at `main`-reachable
`d6f12b1f` — the freeze commit, so K_max/C/W were on `main` before the run — every row
stamped with the executed git hash and `ts`, `mock: false`, `working_tree_dirty_src_scripts:
false`. prereg-precedes-data lint clean: the governing base pre-registration and the PRE-DATA
Amendment 1 precede the first row; amendments 2–3 are POST-DATA disclosures (reported, not
judged).

## Regression guard

**Re-run on:** `substrate_merge` / `ec_merge_aligned` / `_merge_mean_clamped` change (the
fold whose cost this measures), `NAc.credit_operant_reward` / world-cluster credit routing
change, `recommend_action(current_clusters=)` change, `SensorEncoder` / EC world-modality
change, `minecraft_bench57` body change, Minecraft bridge protocol change, minor-version
heartbeat.

**Guard:** [scripts/exp57/](../../scripts/exp57/) (`instrument_check.py` /
`run_ladder.py --resume`, the balanced schedule + fold + selector in `common57.py`) +
[scripts/analyze_exp57.py](../../scripts/analyze_exp57.py) (`--gate v1 --assert-noop-fails`,
frozen gate constants: rungs {1,2,4,8}, δ_eff = 0, p < 0.05, ≥ 20 cohorts) + guard tests
[tests/unit/test_exp57_harness.py](../../tests/unit/test_exp57_harness.py) + committed data
[data/57_ladder.jsonl](data/57_ladder.jsonl) /
[data/57_ladder_verdict.json](data/57_ladder_verdict.json) /
[data/57_phase0.json](data/57_phase0.json) + the pre-registration.
