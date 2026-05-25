# 29 — Roy-3c bisect: which 0.9.1 wire merge introduced the priming-side cluster_reward_bias regression?

**Date:** 2026-05-24
**Branch:** `feat/0-9-1-roy-3c-bisect`
**Status:** Complete — verdict reframes the Roy-3 priming-side question.

## Motivation

`docs/experiments/23_roy_3.md` § "Priming-side substrate" surfaced a
regression in the post-priming `cluster_reward_bias` map between the
historical Roy-2/2c/4/5a iterations (2026-05-12 → 2026-05-13) and the
Roy-3a/3b runs that landed in 0.9.1 (2026-05-23):

| Iteration | priming `cluster_reward_bias` entries | values |
|---|---|---|
| Roy-2  (2026-05-12) | 6 | all `sense_food_source` × +1.0000 |
| Roy-2c (2026-05-13) | 6 | all `sense_food_source` × +1.0000 |
| Roy-4  (2026-05-13) | 6 | all `sense_food_source` × +1.0000 |
| Roy-5a (2026-05-13) | 6 | all `sense_food_source` × +1.0000 |
| Roy-3a (2026-05-23) | 2 | `sense_food_source` × {+0.18, +0.98} |
| Roy-3b (2026-05-23) | 2 | `sense_food_source` × {+0.21, +0.98} |

The Roy-3 writeup hypothesised that one of the four 0.9.1 wire merges
(PRs #253 / #255 / #256 / #257, landed 2026-05-15 → 2026-05-17)
introduced the regression. This iteration bisects the wire merges to
name the introducing PR.

## Bisect protocol (executed)

Six commits in linear merge order, EC threshold pinned at 0.40
(pre-EC-drift-fix) at every step for apples-to-apples comparison with
the 5/13 baseline:

| Step | Commit | PR | Wire |
|---|---|---|---|
| 0 | `cd51be5` | #252 | pre-wires (jepa docs) |
| 1 | `242235a` | #254 | Stages 0b+0c telemetry |
| 2 | `629745a` | #253 | **Wire-A** (cluster-bias annotation) |
| 3 | `51dfd38` | #255 | Wire 3 (embodiment tool filter) |
| 4 | `704acb0` | #256 | Wire 2 (Pavlovian PainBus + decay) |
| 5 | `6610566` | #257 | Wire 1 (NAc variance accumulator) |

**Spec:** `scenarios/roy/roy_2_iteration.yaml`. The Roy-3a spec doesn't
exist at the bisect commits (authored 2026-05-23, after window). Roy-2's
spec has byte-identical multi-arc priming structure (2 × cradle_prelinguistic +
2 × cradle + 1 × cradle_prelinguistic, 50 turns total) and IS the
historical 5/12 entry that produced the 6-key baseline.

**Runner env:** `PYTHONPATH=src MAXIM_SUBSTRATE_PATH=1` at every step.
The substrate-path env var is load-bearing per
[feedback_substrate_path_env_var_for_roy.md](../../.claude/projects/-Users-dennyschaedig-Scripts-Maxim/memory/feedback_substrate_path_env_var_for_roy.md).
**This is itself a probable contributor to the apparent regression** —
see the verdict section.

**Per-step procedure:**
1. `git checkout <commit>` (detached HEAD from bisect branch)
2. Sanity: `ECConfig().pattern_complete_threshold == 0.40` ✓ at every step
3. `rm -rf ~/.maxim/roy/roy-2` (clean slate per step)
4. `PYTHONPATH=src MAXIM_SUBSTRATE_PATH=1 maxim roy run scenarios/roy/roy_2_iteration.yaml`
5. Read `priming.final_session_id` from result.json, count keys + record values
   from `~/.maxim/sim_reports/$sid/aut_nac.json`
6. Backup to `~/.maxim/roy/roy-2-bisect-<commit_short>/` for audit

## Results

| Step | Commit | PR | key_count | values |
|---|---|---|---:|---|
| Historical | (5/12 baseline) | — | 6 | all +1.0 |
| 0 | cd51be5 | #252 docs | **2** | +1.0, +1.0 |
| 1 | 242235a | #254 telemetry | **2** | +1.0, +1.0 |
| 2 | 629745a | #253 Wire-A | **2** | **+0.22, +0.98** |
| 3 | 51dfd38 | #255 Wire 3 | **2** | +0.12, +0.98 |
| 4 | 704acb0 | #256 Wire 2 | **2** | +0.16, +0.98 |
| 5 | 6610566 | #257 Wire 1 | **2** | +0.15, +0.98 |
| Roy-3a (reference) | main+EC drift | — | 2 | +0.18, +0.98 |
| Roy-3b (reference) | main+EC drift | — | 2 | +0.21, +0.98 |

Both keys at every step are `tool:sense_food_source` on different EC
cluster UUIDs. Both UUIDs are **interoception-modality** EC nodes at
every step (confirmed by cross-referencing
`aut_nac.json::cluster_reward_bias` keys against
`aut_ec.json::substrate_nodes[<uuid>].modality`). The EC substrate at
every step contains 14 nodes: 12 text-modality + 2 interoception —
identical distribution at all six steps. The 2 reward-bias entries are
exclusively the interoception nodes; none of the 12 text-modality
nodes carry `cluster_reward_bias`. This matches PR #251's earlier
analysis verbatim: *"NAc's cluster_reward_bias still keys food clusters
exclusively to interoception node IDs."*

## Verdict — the regression decomposes into two axes

The kickoff treated the Roy-3 observation (6 saturated → 2 partial) as
a single phenomenon. The bisect data splits it into two distinct
effects with two distinct causes, both **outside** the wire-merge
window:

### Axis 1 — key count (6 → 2): pre-window environmental shift

The key-count drop is already present at Step 0 (the docs-only PR #252
merge). All five subsequent wire merges hold at 2 keys. The cause is
**before** cd51be5, somewhere between Roy-5a's last 6-key run
(2026-05-13) and cd51be5 (2026-05-14).

`git log --since=2026-05-13 --until=2026-05-15 -- src/maxim/decisions/
src/maxim/proprioception/ src/maxim/similarity/` returns one commit
(`0ec3f95`, PR #248) — and that commit only adds sim_reports
persistence write-paths, no runtime substrate logic. PR #252 is docs.
PR #251 is also docs (it just *documents* `MAXIM_SUBSTRATE_PATH=1` as
load-bearing). **No runtime code change in the window can explain the
key-count drop.**

**Initial suspect (REFUTED by A1):** the `MAXIM_SUBSTRATE_PATH=1`
env-var regime change. PR #251 documented it as load-bearing on 5/14,
after the historical 5/12 Roy-2 baseline. Plausible on paper: the
5/12 baseline almost certainly ran *without* the env var, and the env
var changes what gets EC-encoded.

**A1 ran this hypothesis (see "Disambiguating diagnostics" section
below) and refuted it.** Roy-2 at cd51be5 with the env var explicitly
removed still produces 2 saturated reward-bias keys. The env var DOES
add 12 text-modality EC nodes (vs only 2 interoception nodes with it
off), but NAc reward attribution lands on the same 2 interoception
clusters either way — because no text-modality cluster is ever
food-bearing in this priming sequence.

**Revised suspect (post-A1): LLM narrator drift on the leader between
5/13 and 5/23.** The cradle narrator is LLM-driven (qwen2.5-14b on
the user's leader per CLAUDE.md hardware notes). Different narrator
output → different food-bearing percept strings → different EC
clusters → different reward attribution. Not testable by code bisect;
testable by snapshotting cradle narrator outputs across historical
and current runs and diffing the content.

### Axis 2 — bias magnitude (saturated → partial): Wire-A's intentional bio-fidelity fix

Step 2 (Wire-A merge) is where the `{+1.0, +1.0}` → `{partial, +0.98}`
shift appears. Steps 3, 4, 5 hold the pattern; the partial values
(+0.12 / +0.16 / +0.15) all fall within run-to-run variance from
LLM-driven cradle narrator content (the priming uses substrate-primary
AUT but the narrator is LLM-driven).

A2 (see "Disambiguating diagnostics" section) ran each of cd51be5 /
242235a / 629745a 3 times. Pre-decay commits cd51be5 + 242235a
produce saturated `+1.0/+1.0` across all 7 runs (4+3); post-decay
629745a produces `{partial, +0.98}` across all 3 runs (partial value
+0.15 / +0.21 / +0.22). Zero overlap between the pre- and post-decay
magnitude distributions over 10 independent runs.

**This is not a regression. It is the intended bio-fidelity correction
added by Wire-A's pre-merge fold commit (`bee42ca`).**

The fold commit's own description:

> "Critical (bio-fidelity): `_cluster_reward_bias` had no per-tick
> decay. Without it, Wire-A's annotation becomes a permanent fossil of
> every reward the substrate ever saw — claiming 'from prior
> experience' while actually being 'from forever ago.' The bio reviewer
> correctly flagged this as a by-accretion contamination of the
> substrate-voice thesis.
>
> Fix: `NAc.decay_cluster_reward_biases()` mirrors
> `decay_goal_reward_biases` (bidirectional, abs-value prune below
> 0.001), wired into the per-tick decay block in `agent_loop.py`
> alongside `decay_reward_biases()` and `decay_goal_reward_biases()`."

The decay applies every tick during priming. Clusters that get
rewarded early in the 50-turn priming sequence decay back toward zero
as the priming progresses; the cluster that gets rewarded near the end
of priming saturates near +1.0. The `{partial, +0.98}` shape is
exactly what per-tick decay + sparse rewards produce.

This was a Critical bio-fidelity finding folded BEFORE Wire-A's PR
opened, per `feedback_review_before_ship.md`. Treating the partial-bias
observation as a regression to fix would *undo* a load-bearing
bio-fidelity correction. The thesis-erosion pattern in
[feedback_interim_contamination.md](../../.claude/projects/-Users-dennyschaedig-Scripts-Maxim/memory/feedback_interim_contamination.md)
warns specifically against this kind of unwind.

## Implications for the post-Roy-3 agenda

The pre-bisect kickoff routed three possible follow-up decisions on
the assumption that the priming regression was the symptom of a
wire-introduced bug:

1. **Affordance-replay bridge** (pre-1.0 vs deferred) — premised on
   re-saturating biases through repetition to recover the 6-key
   saturated pattern.
2. **Wire-A "strongly rewarding" render floor** — premised on biases
   needing to clear the >0.5 band threshold to render in the prompt.
3. **Roy-3 re-validation** — premised on a fix landing and being
   measured.

The bisect changes the routing for all three:

1. **Affordance-replay bridge is no longer load-bearing for the
   priming-side count.** The 6→2 drop is environmental (originally
   hypothesised as `MAXIM_SUBSTRATE_PATH=1`, but A1 below refutes that
   specific suspect — see "Disambiguating diagnostics" section. The
   updated suspect is LLM narrator drift on the leader between 5/13
   and 5/23). Either way, not a learning-rate problem. Building a
   replay bridge to re-saturate biases would be addressing a symptom
   of a fix-time decay rule, not a real cross-session learning gap.
   The bridge stays a 1.1+ candidate only if a separate diagnostic
   demonstrates a real text-modality cluster-reward gap.
2. **The "strongly rewarding" render floor question reframes from
   "biases regressed below threshold" to "given per-tick decay, what
   priming repetition / reward magnitude / decay-rate balance keeps
   the final-priming clusters above the >0.5 band?"** This is a tuning
   question about priming length × reward strength × decay rate, not a
   structural NAc fix. The +0.98 entry at every Step 2+ already clears
   the threshold; the partial entry doesn't, but its existence as a
   distinct cluster is the diagnostic signal Wire-A intended.
3. **Roy-3 re-validation against the bisect verdict is unnecessary.**
   The 2-key partial pattern Roy-3a/3b observed is the bio-correct
   substrate state under `MAXIM_SUBSTRATE_PATH=1` + Wire-A decay. The
   re-validation question becomes "can Wire-A's annotation render the
   substrate-voice signal effectively given this state shape?" — which
   is a prompt-render question, not a substrate-fix question.

## Disambiguating diagnostics (RAN — see results below)

Both A1 and A2 were executed against the initial verdict. **A1 refuted
the env-var hypothesis on the key-count axis. A2 confirmed Wire-A's
decay as the magnitude-axis cause.**

### A1 — env-var hypothesis (REFUTED)

Re-ran Roy-2 at cd51be5 with `MAXIM_SUBSTRATE_PATH` removed from the
environment (via `env -u MAXIM_SUBSTRATE_PATH`):

| Run | env var | key_count | values |
|---|---|---:|---|
| Step 0 (original) | ON | 2 | +1.0, +1.0 |
| **A1** | **OFF** | **2** | **+1.0, +1.0** |

The env-var removal did NOT restore the 6-key count. Both runs produce
exactly 2 saturated reward-bias keys on the same 2 interoception EC
clusters.

The env-var WAS confirmed to affect the EC substrate shape — with the
env var ON, 12 text-modality nodes are added to substrate_nodes; with
it OFF, only the 2 interoception nodes exist. But NAc reward
attribution lands on the 2 interoception clusters in both cases (no
text-modality cluster is ever food-bearing under this priming, per PR
#251's earlier analysis). So the env var is necessary for text-modality
EC encoding but is NOT the cause of the cluster_reward_bias key-count
gap.

**Revised hypothesis for the key-count axis:** the 6→2 drop must be
attributable to environmental drift in the cradle narrator's LLM
output. The narrator is LLM-driven (qwen2.5-14b on the user's leader
per CLAUDE.md hardware notes) and produces variable cradle scene
content per run. If the leader's LLM model state shifted between the
5/12-5/13 historical baseline runs and the 5/23-5/24 runs (model
swap, weights cache refresh, sampler config change, prompt template
tweak), the narrator would generate different food-bearing percept
strings, producing different EC clusters, producing different reward
attribution. **This is testable by snapshotting the cradle narrator
output across runs and diffing.** The bisect doesn't disambiguate
this further because the bisect varies code, not LLM state.

### A2 — magnitude axis (Wire-A decay CONFIRMED)

Re-ran cd51be5 / 242235a / 629745a 2 more times each (n=3 per step,
all with `MAXIM_SUBSTRATE_PATH=1`):

| Commit | Wire-A decay | Runs | All key counts | All magnitudes |
|---|---|---:|---|---|
| cd51be5 (Step 0 + A1 + r2 + r3) | OFF | 4 | all 2 | all +1.0, +1.0 |
| 242235a (Step 1 + r2 + r3) | OFF | 3 | all 2 | all +1.0, +1.0 |
| 629745a (Step 2 + r2 + r3) | **ON** | 3 | all 2 | **all {partial, +0.98}** |

Pre-decay (4+3 = 7 runs at commits cd51be5/242235a) produce saturated
`+1.0/+1.0` every time. Post-decay (3 runs at 629745a) produce
`{partial, ~+0.98}` every time, with the partial value varying
+0.15 / +0.21 / +0.22 (run-to-run narrator variance changes which
cluster gets the last reward + how recently the secondary cluster
was rewarded; the saturated +0.98 is stable because by-construction
it's the cluster rewarded just before priming ends — minimal decay
window).

Zero overlap between pre-decay and post-decay magnitude distributions
across 10 independent runs. **Wire-A's bee42ca fold introducing
`NAc.decay_cluster_reward_biases()` is the causal mechanism for the
magnitude axis.**

### What's still open after A1+A2

- **Key-count axis cause is now narrower but still un-pinpointed.** Not
  env var, not wire merges, not in-window code changes, not arc
  definitions. Most likely LLM narrator drift on the leader between
  5/13 and 5/23. Confirmation requires either (a) snapshotting cradle
  narrator outputs across the historical and current runs, OR (b)
  reverting the leader's LLM config to its 5/12 state and re-running.
  Both are out of scope for the bisect.
- **Magnitude axis is fully closed.** Wire-A's decay does what its
  fold commit description said it does. The partial-bias observation
  in Roy-3a/3b is the intentional bio-fidelity correction surfaced
  by the decay rule.

### The user's "latent issue surfaced by the fix" hypothesis (post-mortem)

The user asked during execution whether the bio-fidelity decay could
have surfaced a pre-existing under-reinforcement issue rather than
introduced new behavior. A2's data answers this directly for both
axes:

- **Count axis: NO.** Steps 0/1 run on the codebase BEFORE decay was
  wired and still produce 2 keys (n=7 runs). Decay can't be what's
  pruning 6 down to 2 — without decay, the count is also 2. The
  6→2 gap is environmental.
- **Magnitude axis: YES.** Without decay (Steps 0/1), every rewarded
  cluster sits at +1.0 forever — including clusters that were
  rewarded ONCE at turn 5 and never again. Decay makes visible the
  fact that priming under-reinforces most clusters: clusters rewarded
  once mid-priming decay back toward zero by the time priming ends.
  The partial-bias clusters are the substrate honestly representing
  "I saw this cluster once 40 turns ago and it hasn't fired since"
  rather than "this cluster is currently +1.0 strong." That IS new
  visible information that no-decay was hiding. Whether this counts
  as "surfacing a latent issue" or "the substrate finally telling
  the truth" is a framing question; either way the partial values
  are bio-correct and should be preserved.

## What this bisect does NOT decide

- It does NOT decide whether the bio-correct partial-bias pattern is
  the *useful* substrate state for Wire-A to render. That's a separate
  question about prompt assembly + LLM read-through behavior, owned
  by the Roy-3 re-validation iteration after this verdict is folded
  into the plan.
- It does NOT decide whether Maxim should ship a separate priming
  regime for the cross-modal text-to-interoception cluster binding
  question. That's a JEPA / scope-1.1+ question, not affected by this
  bisect.
- It does NOT touch the EC drift fix (PRs #259-#264). The bisect ran
  at threshold 0.40 throughout. The 0.44 retest at current main is a
  separate iteration that lands after this verdict.

## Files / artifacts

- Backups: `~/.maxim/roy/roy-2-bisect-<commit_short>/` for all six
  steps (cd51be5, 242235a, 629745a, 51dfd38, 704acb0, 6610566)
- Report dirs: `~/.maxim/sim_reports/bisect-<commit>-<sid>/`
- Per-step logs: `/tmp/roy_3c_step{0..5}_<commit_short>.log`
- Extractor: `/tmp/roy_3c_extract.sh`
- Bisect window inspection: PR #252 (docs only, 380 LOC docs),
  PR #251 (docs + 2 spec YAMLs, no runtime code),
  PR #248 (orchestrator.py + report.py persistence only),
  PR #249 (Roy harness analyzer only)
- Wire-A producer-site diff:
  `src/maxim/runtime/agent_loop.py` lines ~2766-2787 at c63216d
- Wire-A bio-fidelity fold commit: `bee42ca` (adds
  `NAc.decay_cluster_reward_biases()` + per-tick wiring)

## Reproduction

```bash
git fetch origin
git checkout feat/0-9-1-roy-3c-bisect

# Per step (substitute commit + label):
git checkout <commit>
rm -rf ~/.maxim/roy/roy-2
PYTHONPATH=src MAXIM_SUBSTRATE_PATH=1 maxim roy run scenarios/roy/roy_2_iteration.yaml
sid=$(jq -r '.priming.final_session_id' ~/.maxim/roy/roy-2/result.json)
jq '.cluster_reward_bias | length' ~/.maxim/sim_reports/$sid/aut_nac.json
jq '.cluster_reward_bias' ~/.maxim/sim_reports/$sid/aut_nac.json
```

## Verdict summary (one-line headline, A1+A2 refined)

**The 0.9.1 wire merges did not introduce the Roy-3 priming-side
"regression." The 6→2 key-count drop is environmental and (per A1's
refutation of the env-var hypothesis) most likely LLM narrator drift
on the leader between 5/13 and 5/23, NOT the `MAXIM_SUBSTRATE_PATH=1`
regime change. The saturated→partial magnitude shift is Wire-A's
intentional bio-fidelity decay correction (`bee42ca`,
`NAc.decay_cluster_reward_biases()`), confirmed behaviorally by A2 —
not a regression.**
