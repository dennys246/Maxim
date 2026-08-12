# Decision provenance — recording WHY the substrate chose an action

**Status:** Stages 1+2 SHIPPED (2026-08-12); Stages 3+4 open.
`sim_recommend_action` now carries `score_components` (causal / reward_bias /
learned_bias / drive / explore for the selected tool), `runner_up_score`,
`n_candidates`, `visit_count`, `explore_decisive`, and `learned_margin` on every
path (None where uncomputable; `n_candidates` 0-vs-None mirrors the
`_consulted_on_empty` no-scores/no-tools distinction). The counterfactual
compares against the ACTUAL outcome (None when the gate fails), so the
explore-first-gate-selects-a-sub-threshold-tool case reads decisive=True.
Guards: `tests/unit/test_decision_provenance.py` (9 tests incl. the
byte-identical-selection sequence with telemetry on vs off, and the
gate-override negative `learned_margin`). Originally PROPOSED 2026-08-11,
motivated by the Exp 48 apparatus investigation.
**Owns:** the per-decision attribution surface on `NAc.recommend_action` /
`propose_via_substrate`.
**Companions:** [simulation_apparatus_standards.md](simulation_apparatus_standards.md)
(S2 canaries consume this), [behavioral_graduation_candidates.md](behavioral_graduation_candidates.md)
(rows whose claims rest on "the substrate chose it").

## The motivating failure

Exp 48's 1.1 re-run took **four configurations across two machines and two
narrator models** to establish what one structured field would have shown
immediately: **exploration was outscoring the learned signal roughly 20:1, and
the learned term never exceeded the novelty gap.**

The specific findings that were expensive to recover, and would have been free:

- `novelty = explore_weight / (1 + visits)` at weight 1.5 produces a
  steady-state gap of ~0.11 between the just-picked and not-just-picked action.
  Any learned `cluster_reward_bias` below that is **invisible** — a fact
  discovered by sweeping the bias offline, not by reading a log.
- On a driveless body the exploration bonus is the *sole source of action*
  (weight 0 → zero actions, because a driveless affordance scores 0 and is
  dropped by the `score > 0` filter). Nothing in the telemetry says "this
  action happened because of exploration, not learning."
- The operant credit lands on `_pending_operant_action` — the last action —
  which under alternation is a coin flip. Nothing records that the credited
  action was uncorrelated with the behaviour being taught.

**The generalisable claim:** several graduated rows assert *"the substrate
drives action selection"*. We record what was chosen and what the outcome was.
We do not record **why** — so "the substrate chose it" and "the exploration
bonus chose it" are indistinguishable in every artifact we keep.

## What exists today (audit before building)

`decisions/nac.py::_emit_recommend_action_event` already emits one
`sim_recommend_action` event per `recommend_action` call — including the
early-return paths (Stage 0c discipline). It carries:

| field | what it tells you |
|---|---|
| `best_tool`, `best_score` | what won, and its total |
| `cluster_reward_bias_consulted` | the learned bias **sum** |
| `consulted_bias_by_modality` | learned bias per modality (the extero/intero seam) |
| `current_cluster_id`, `current_clusters` | which clusters were active |
| `min_confidence`, `passed_gate` | whether the gate fired |

So the **learned** half is already structured and per-modality. That is real
coverage and this plan does not rebuild it.

## The gap

Three things are missing, all of them cheap:

1. **The score decomposition is not recorded.** `best_score` is a sum of named
   components (causal confidence, drive relevance, learned cluster bias,
   exploration novelty — `nac.py` ~:1990-2042) that are computed separately and
   then collapsed. Only the total survives.
2. **The margin is not recorded.** A win by 0.001 and a win by 0.9 are
   indistinguishable, so a coin-flip selection looks identical to a decisive one.
3. **There is no counterfactual.** The single most valuable question —
   *would the learned signal alone have chosen the same action?* — is
   computable at the decision site for free (the components are already in
   hand) and is never asked.

## Relationship to S1 (`feat/annotation-credit-provenance`) — ACCEPTED join

S1 built the **write** side independently and concurrently: *why a learned value
exists*, recorded at `update_cluster_reward` (`NAc._cluster_reward_source`, keyed
identically to `_cluster_reward_bias`; closed `CREDIT_SOURCES` vocabulary;
one-way promotion to `mixed`; optional `source=` so existing callers are
untouched; absent in pre-S1 files). This plan owns the **read** side: *whether
that value decided anything*, recorded at `recommend_action`.

**The join (accepted):** Stage 1 threads `score_components.learned_bias` as a
float; S1 annotates it with its source. One event then reads *"learned_bias
0.42, deposited by operant credit, and it was decisive."* Full chain in one
record, no new mechanism on either side.

**Conventions adopted from S1** rather than reinvented: optional field so no
caller changes; closed vocabulary mirroring real branches (a new value should
mean a new branch exists, not a new adjective); unknown values dropped so a typo
degrades to "no provenance"; pre-existing files load empty rather than
fabricating history.

**S1's honest limit — why Stage 2 is not lowered in priority.** S1 records
*which mechanism* deposited a value, not whether that mechanism attributed
*correctly*. The Exp 48 failure — operant credit landing on
`_pending_operant_action`, a coin flip under alternation — tags cleanly as
`operant` and looks healthy in S1. Only `explore_decisive` catches it. Neither
substitutes for the other.

**Stage 2 is now on someone else's critical path.** PR #499's
`MAXIM_PLACE_CODE_EXTEROCEPTION` (default OFF) spreads azimuth over ~7 EC nodes
instead of ~3, so the same learning divides across ~2.3× more nodes and
per-node `cluster_reward_bias` necessarily drops. Measured against the ~0.11
novelty visibility floor, the place code can push learned bias **under the
floor** — so a place-code gate run is uninterpretable until `explore_decisive` /
`learned_margin` exist. The flag-ON arm is held pending this stage.

**The ~0.11 floor is a design constraint, not an Exp 48 artifact.** Any encoding
change that subdivides a cluster must keep per-node learned bias above the
novelty gap at the operating explore weight, or the learning becomes
unexpressible regardless of whether it is correct.

## Front-gate scope pressure

Per CLAUDE.md Principle 3 — does this need to be its own mechanism?

| Candidate | Sufficient? |
|---|---|
| `sim_recommend_action` event (exists, one per call, tick-aligned to `sim_ec_activation`) | **Yes — this plan is additional FIELDS on it.** No new channel, no new bus, no new writer |
| The `reasoning` free-text string | Carries some of it unstructured; a string is not queryable and drifts silently. Promote to fields, keep the string |
| Per-run JSONL records (`fade`, harness output) | Wrong altitude — aggregate outcomes, not per-decision attribution |
| A new provenance bus | **Rejected.** The repo already carries four buses plus a deferred unified one; a fifth for a field addition moves the wrong way |

**Verdict:** rides entirely on existing infrastructure. Additive fields on one
existing event, plus one derived boolean.

## Stages

### Stage 1 — Structure the score decomposition (the core)

Thread the already-computed components into `_emit_recommend_action_event` for
the **winning** tool:

```
score_components: {causal: float, reward_bias: float, learned_bias: float,
                   drive: float, explore: float}
visit_count:      float          # the novelty driver
runner_up_score:  float | None   # margin = best_score - runner_up_score
n_candidates:     int
```

**Why `reward_bias` is a separate fifth component** (review fold): the node-keyed
`reward_bias` (capped 0.20, recognition modulator) is a different mechanism from the
cluster-keyed `cluster_reward_bias`, and S1's `_cluster_reward_source` annotates only
the latter. Folding both into `learned_bias` would make the S1 join incoherent —
"learned_bias 0.42, deposited by operant" would include a term S1 never sources. Do
NOT merge them for simplicity; `score_components.learned_bias` must equal exactly the
cluster-bias sum S1 annotates.

Keyword-only, all optional with `None` defaults so no existing caller breaks.
The components are local variables at the scoring site today — this is
plumbing, not computation.

### Stage 2 — The counterfactual field (the highest-value item)

One derived boolean, computed at the decision site where the components are
already in scope:

```
explore_decisive: bool   # would argmax WITHOUT the explore term have picked a
                         # different tool?
```

**This single field is the deliverable.** `explore_decisive = true` on ~95% of
decisions is the entire Exp 48 finding, visible in one `jq` query on day one
rather than after a four-configuration investigation. Cheap corollary:
`learned_margin` — the winner's learned-bias lead over the runner-up, which is
what must exceed the novelty gap for learning to be expressible.

### Stage 3 — Make it queryable + one worked query

A helper (`scripts/analyze_decision_provenance.py`) that reads a run's JSONL and
reports: fraction of decisions where `explore_decisive`, the distribution of
`learned_margin` vs the novelty gap, and per-tool selection shares with their
attribution. Cite it from the runbook so a walk can ask "was this row's
behaviour learned or explored?" as a routine check.

**S1-join note (review fold):** join provenance on the persisted
`(agent_id, cluster_id, tool_sig)` triple — the event carries `current_clusters`
+ `best_tool` for exactly this. Do NOT join through the aggregated
`get_cluster_reward_sources` read surface: it unions across ALL clusters
(`"mixed"` on disagreement) while `learned_bias` sums the ACTIVE clusters only,
so the tool-level aggregate can over-coarsen the attribution.

### Stage 4 — Wire into the S2 canaries (read side AND write side)

The apparatus canaries then assert on attribution rather than only on outcome.
**Two assertions, catching different failures:**

* **Read side (this plan):** *"in a substrate-primary row claiming learned
  selection, `explore_decisive` must be below X% by the final act."* Catches
  learning that is correct but never expressed. Two canary-authoring notes
  (review fold): (a) pick the denominator explicitly — `passed_gate=True`
  events vs all non-None `explore_decisive` events give different rates;
  (b) a run with exploration accidentally DISABLED reads 0% decisive and
  trivially passes — rows that claim exploration was live must co-assert
  `substrate_explore_bonus_weight > 0` (or nonzero `score_components.explore`
  somewhere in the run).
* **Write side (S1):** *"in an operant-only row, `cluster_reward_source` must be
  100% `operant`, never `mixed`."* Catches a tool-success floor leaking into a
  row whose design excludes it — a different failure, invisible to
  `explore_decisive`, and directly relevant since Exp 48 sets
  `MAXIM_OPERANT_ONLY_CREDIT=1`.

Together these turn the standards doc's S2 from "detect gross apparatus change"
into "detect the claim quietly becoming false."

## Explicit non-goals

- **Not a new bus, not a new persisted artifact.** Fields on an existing event.
- **Not a behaviour change.** Pure observation; if any stage changes a
  selection, it has a bug. Pin this with a byte-identical-selection test.
- **Not retroactive.** Existing runs cannot be re-attributed; the original
  Exp 48 data is gone regardless (S4).
- **Not the fix for the exploration/control conflict.** That is a separate
  design question (count-based novelty over a 2-actuator continuous axis is
  definitionally "reverse direction"). This plan makes the conflict *visible*;
  it does not resolve it.

## Sizing

| Stage | Cost |
|---|---|
| 1 decomposition fields | ~½ day (plumbing + event schema) |
| 2 counterfactual + margin | ~½ day (the highest value/cost ratio in the plan) |
| 3 analyzer + runbook citation | ~½ day |
| 4 canary wiring | folds into S2's per-row cost |

## Risks

| Risk | Mitigation |
|---|---|
| Event volume — one decision per 0.5 s per agent | Fields are small; gate the verbose ones behind the existing trace env-var pattern if a run's JSONL becomes unwieldy |
| Field drift vs the `reasoning` string | The string stays human-facing; fields are the queryable truth. If they disagree, fields win — state it in the docstring |
| Scope creep into a general "explainability" layer | Non-goals above are binding; this answers one question (learned vs explored) for one caller |

## Definition of done

- `sim_recommend_action` carries the score decomposition, margin, visit count,
  and `explore_decisive` for every decision, on every early-return path too
  (Stage 0c's existing discipline extends to the new fields).
- A byte-identical-selection test proves the instrumentation changed no choice.
- `analyze_decision_provenance.py` answers "learned or explored?" for a run in
  one command, cited from
  [heartbeat_rerun_runbook.md](../experiments/protocols/heartbeat_rerun_runbook.md).
- At least one graduated substrate-primary row has its attribution measured and
  recorded — including, honestly, if the answer is "explored."
