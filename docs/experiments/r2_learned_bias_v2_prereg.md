# R2 learned-bias v2 — does drive-relief credit make eating STATE-CONTINGENT, and how does it scale with choice-space size?

> **STATUS: PRE-REGISTRATION DRAFT (2026-09-12) — for review, NOT frozen, NO confirmatory data taken.**
> Supersedes the v1 marginal-probe design (`r2_learned_bias_prereg.md`): a run + offline experiment
> proved v1 cannot isolate the drive-relief credit — in a single-corrective-action world the
> state-blind tool-success causal link saturates the flip (`docs/wiring/substrate-learning-channels.md`).
> v2 gives the credit a discriminative job AND titrates the choice-space. This doc freezes AFTER the
> exploratory pilot (below) picks the count set + N; the frozen confirmatory doc must be on `main`
> before the first confirmatory data timestamp. Open params: `⟨DECIDE⟩`.

## Question

The drive-relief cluster credit's purpose is **state-conditioned, competitive selection** — choose
`eat` over other viable actions **specifically when hungry**. Two coupled questions:

1. **Isolation:** does the credit make eating state-contingent (eat preferred at a hungry probe but
   not a satiated one) BEYOND the innate prior and the state-blind causal link?
2. **Dose-response (titration):** how does that effect scale with the number of competing
   actions `K`? Headroom grows with `K` — eat's causal baseline is ~1/(K+1), so a state-conditioned
   credit has more room to lift eat-when-hungry as the choice space grows. A single `K` is a weak
   test (a null at `K=1` may be pure low-headroom); the **curve** is the robust measure.

## Why state-contingency isolates the credit (v1 lesson)

The causal link is state-BLIND (`tool:eat`, no cluster) → favours eat equally hungry/satiated. The
innate prior favours eat when hungry (both arms). The cluster credit is the ONLY learned AND
state-conditioned channel. Its signature is a hungry-vs-satiated gap that is larger in LEARNING than
the ablation:

    contingency_gap(arm, K) = P(eat over the K competitors | hungry) − P(eat | satiated)
    isolated drive-relief effect(K) = gap_LEARNING(K) − gap_NO-CREDIT(K)   (both share prior + causal)

## Apparatus (shared with v1 except roster + probe + titration)

Survival world, `bodies/minecraft_player`, substrate-primary (no LLM), real `record_outcome` path,
on the bridge box. All v1 guards carry over (enforced refusals; frozen-apparatus fingerprint +
absolute asserts + `MAXIM_OPERANT_ONLY_CREDIT`-unset; bridge-connect retry; incremental JSONL;
`recommend_action` determinism → K_probe=1 binary picks; `min_confidence=0.0`; relief-only enforced).

- **Competitor pool** ⟨DECIDE⟩: non-corrective, always-executable, always-successful tools that build
  their own causal links and have NO food/health effect — candidate ordered pool
  `[move_to, mine_block, turn]`; condition `K` uses the first `K`.
- **Balanced training** ⟨DECIDE⟩: eat trained identically across all `K` (same cluster bias); each
  competitor gets comparable successful runs via round-robin between eat episodes
  (`eat, c1, eat, c2, …`). The raw-P(eat) secondary audits that eat's baseline actually drops with `K`
  (else headroom didn't grow and the titration is uninformative).

## Arms (three; interleaved within seed; controls cycle-matched) × conditions (K ∈ ⟨DECIDE⟩ {1,2,3})

LEARNING / NO-CREDIT (cluster credit suppressed before `record_outcome`, relief-only) / SATIATED
(kept satiated, competitors run, eat gives no relief). Every (arm, K, seed) is a fresh substrate.

## Metrics

- **Probe states:** hungry = **food 11** (the only `None`-band state SHARING the food≤4 training
  cluster `fd0ae83c`; v1 diagnostic — food 12/13 are a different cluster). satiated = ⟨DECIDE⟩
  food 18. Both present the roster `{eat} ∪ {first K competitors}`; the pick is deterministic (K_probe=1).
- **Primary — the dose-response curve:** `isolated_effect(K) = gap_LEARNING(K) − gap_NO-CREDIT(K)`
  across `K`. Structural: "the hungry-minus-satiated eat-preference gap, arm-differenced, as a
  function of competitor count."
- **Secondary:** cluster-bias magnitude (forms in LEARNING, 0 in controls); raw per-(arm,K,state)
  P(eat) — audits the headroom-grows-with-K premise; the v1 marginal probe (with-vs-without clusters
  at food 11, competitors present) as a cross-check.

## Decision rule (frozen AFTER the pilot — no post-hoc motion once confirmatory data starts)

`N`/cell ⟨DECIDE from pilot⟩; `M` ⟨DECIDE⟩ (min meaningful isolated effect, candidate 0.20).
**PREMISE-HELD (drive-relief credit drives state-contingent selection) iff BOTH:**
1. `isolated_effect(K*) ≥ M` at the highest-headroom pre-registered `K*` by one-sided permutation on
   per-seed gaps, `p < 0.05`, AND
2. a non-negative dose-response: `isolated_effect(K)` does not DECREASE with `K` across the
   pre-registered set (a positive trend is the affirmative dose-response; flat-positive still HELD at
   `K*`; decreasing → the effect is not choice-space-scaling → report as such).

Otherwise **PREMISE-NULL**. `gap_LEARNING ≈ gap_NO-CREDIT` at all `K` → state-contingency is the
prior/causal link, not the credit → NULL. Refused seeds (all v1 flags) → REFUSED-UNVERIFIED.

## Pilot, then confirm

1. **PILOT (exploratory, NOT gated, no `--write-experiment-results`):** small `N≈5` across
   `K ∈ {1,2,3}` — locate the effect + confirm the headroom-grows-with-K premise (raw P(eat) drops
   with K) + that the instrument is clean live at scale. A pilot null at EVERY K (with headroom
   confirmed) means don't spend the confirmatory run — reconsider the design.
2. **FREEZE** this doc with the pilot-informed `K` set, `K*`, `N`, `M`; merge to `main`.
3. **CONFIRMATORY gated run** (well-powered `N`, `--write-experiment-results`) → data PR (merge-commit,
   tag waits a day) → verdict.

## Known-limit acknowledgments

- **Causal-link balance is the live risk** (v1 lesson, one level out): if eat's causal link dominates
  the competitors', the causal link re-saturates → no isolation → null even if the credit works.
  Mitigated by balanced round-robin training + audited by the raw-P(eat) secondary.
- **Cluster generalization:** the hungry probe MUST share the training cluster — food 11 does, food
  12/13 don't (v1 diagnostic); a pre-freeze disclosure re-confirms.
- **The prior contributes to the gap in every arm** — isolated only by `gap_LEARNING − gap_NO-CREDIT`.
- Rung-1 game-native, NO injection (DECISIONS.md 2026-09-12).

## Open decisions (settle at freeze, pilot-informed)

1. Competitor pool + ordering + the balanced round-robin rule.
2. Count set `{1,2,3}` and `K*`; satiated probe food (18?).
3. `N`/cell (from the pilot's variance) and `M` (0.20?).
4. Whether the dose-response primary is "positive at `K*`" alone or "positive trend" (candidate: both,
   as rule 1+2 above).

## Harness reuse

Reuses `scripts/survival_world/r2_learned_bias.py` machinery + all guards; deltas: a `--competitors K`
param (roster + round-robin training), the two-state (hungry/satiated) probe + gap metric, and the
outer `K` loop. Same pipeline: pilot → freeze → two-lens → confirmatory run → data PR.

## Outcome — SUPERSEDED at the design-review gate (2026-09-12); NO harness built, NO live run

The four-lens experiment-DESIGN review (rationale in `docs/experiments/rationale/r2_learned_bias_v2/`)
returned DO-NOT-BUILD on three cross-confirmed grounds — the satiated gap collapses (food-18 is a
different cluster → bias 0 in both arms → the gap ≡ the v1 food-11 marginal), the round-robin
schedule re-introduces v1's causal saturation, and the K-titration axis is broken under the
deterministic argmax selector (P(eat) binary, does not fall with K). A decisive offline experiment
(replicating the learned NAc with K∈{1,2,3} balanced `turn`-competitors) then settled the underlying
question empirically: the competitors build causal links equal to eat's (0.89), so at food-11 **eat
is selected WITHOUT the cluster credit** (causal link + tiebreak) at every K; the credit only adds a
near-tautological score margin (≈ the bias magnitude, which we already know reaches 1.0) or breaks a
tie in the razor-thin case the tiebreak disfavours eat.

**Finding (the cycle-divergence bird's-eye audit resolving after v1 + v2 hit the same wall):** on this
substrate the drive-relief *cluster* credit FORMS (bias=1.0, mechanism established in the break-3
dry-run) but is a behavioural **messenger, not a cause** — action selection is driven by the innate
prior + the state-blind tool-success causal link (itself learned from hunger-driven eating), which
the cluster credit is redundant with. R2's behavioural premise ("do the world drives measurably move
behaviour toward corrective affordances") is carried by those channels, not the cluster-credit
channel specifically; R2 stays PREMISE-NULL for the cluster-credit-specific behavioural claim.

No confirmatory data was taken (nothing to gate). The reusable lesson is recorded in
`docs/wiring/substrate-learning-channels.md`; the mechanism (credit forms) stands. Redirected to the
1.3 survival-world build, where the *composed* loop (prior + causal + credit) works without requiring
the credit to be the sole driver. Establishing this cost ~minutes of offline experiments instead of
a ~30-hour live run — the design-review gate paid for itself on its first use.
