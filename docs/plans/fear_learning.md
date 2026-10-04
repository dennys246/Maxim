# Fear learning: relief as a reinforcer, and extinction

**Status: PROPOSED (2026-10-04).** Tracking issue: [#1072](https://github.com/dennys246/Maxim/issues/1072).
Prompted by #840's deep dive (owner request). Nothing here is built. Each experiment gets a full four-lens design
review (`docs/experiments/DESIGN_REVIEW.md`) before any harness, and everything enters as `[engineering]`.

## What the deep dive found

**FearCircuitBridge is not the thing to revive.** It was added on 2026-02-06 (9c890b0a) for LLM tool-safety
review. Its job was to learn false-positive and true-positive rates per `(DangerCategory, pattern)` for
`FearAgent`'s shell, file and URL reviewer. It was never live:
- its NAc write failed (`record_event` with a float and `metadata=`);
- its NAc read failed (`predict_outcome`, which NAc never had);
- its write key (`pattern_hash[:8]`) and read key (`pattern`) differ;
- the hub façades it serves have never had a non-test caller.

Nothing was declared its replacement, but its jobs are covered elsewhere:

| Intended function | Live coverage today |
|---|---|
| Learn which action patterns predict harm | `ToolPainBridge`/`ToolHarmPredictor` and the pain→NAc subscriber (negative causal links); Wire 4 per situation |
| Feed NAc | Four auto-wired pain subscribers and bridges |
| Block actions | `FearGatedExecutor` (static), the DefaultNetwork movement gate |
| Learn from false alarms | **Nothing.** As the bridge was built, this isn't learnable either: nobody can know whether a blocked call would have done harm |

It is marked `Dormant since 2026-10-04` (#840). The only fear mechanism with earned ledger rows is **NAc Wire 4
situation fear**: T1-13 (Exp 60), T1-14 (Exp 61, fear transfers between agents) and T1-15 (Exp 62).

## Already modelled vs missing

| Bio function | Maxim | Status |
|---|---|---|
| Pavlovian situation→fear (BLA) | Wire 4, `NAc.record_cluster_fear` | **EARNED** (Exp 60, 62) |
| CeA output → defensive state | `anticipatory_threat_need` → `drives["threat"]` | earned with Wire 4 |
| Social fear transfer | fear bundles | **EARNED** (Exp 61) |
| Innate defensive repertoire | `_DRIVE_TOOL_AFFINITIES["threat"]` (flee/hide/escape/…) | hard-coded **innate prior**, chosen by tool name |
| Immediate instrumental punishment | `update_cluster_reward` from `drive_potential_diff` | live; only the drive change during the act |

**Missing:**
1. **Fear relief as a reinforcer** (two-factor avoidance, factor 2). Leaving a feared situation never credits the
   act that did it, so the keyword table decides which defensive act fires. Rename `escape_water` and fear has no
   channel to action. Exp 60 never separated "the escape made me safe" from "the tool succeeded" (after the first
   surfacing, `escape_water` carried 47–51 positive tool-success links).
2. **Extinction.** `_cluster_fear` only ever deepens. The only way back is the slow 7-day wall decay. So a false
   alarm, a fear shared by Exp 61's transfer or one spread by Exp 62's generalization can never be corrected. This
   is the biological version of the bridge's "remember false positives" idea.
3. *(Out of scope: prospective avoidance, i.e. not entering a feared place. That is the R4 delayed-credit gap,
   roadmap 1.4 Phase 5.)*

## Scope pressure

- **Relief credit is the roadmap's reserved relief store under another name.** Its write, a positive relief
  credit keyed to a situation cluster, is exactly what roadmap 1.4 says does not exist yet ("relief never writes
  a world cluster", R4). Phase 5 reserves a **cluster-keyed relief store** for it, entered through its own plan
  and four-lens review, which also reserves the opposite sign so "no second store is ever created". So
  Experiment A must **front-gate against that store in writing**: either A is its first consumer and enters
  through its review, or this plan shows why the existing `cluster_reward` table is enough and corrects the
  roadmap. Until then the sketch below names the existing pieces it would touch:
  - The write is `NAc.update_cluster_reward(agent, feared_cluster, tool, +r, source="threat_relief")`, with a new
    `CREDIT_SOURCES` entry.
  - The trigger is one seam in `agent_loop` beside the Wire-4 read: the threat need on the chosen-in cluster drops
    below θ after an act.
  - The read is `recommend_action`'s existing `cluster_reward_bias`.
  - No new bus. Whether it is a new store is the front-gate question above.
  - Guard needed: credit only exits observed by bridge truth within Δ of the act, so that a rescue or a world
    change is not credited.
- **Extinction needs a small mechanism of its own.** `_cluster_fear` is non-positive and min-merged, so a positive
  update would be erasure, not extinction. Two options:
  - **(2a) erasure.** Cheap, but only honest with a claim scoped to "fear declines with exposure".
  - **(2b) a separate safety map.** `need = max(0, |fear| − safety)` keeps the original fear intact, so savings,
    renewal and reinstatement remain possible. It touches persistence (`_format_version`, CC3) and raises the
    question of whether safety travels in bundles. Recommended: local-only in v1, since extinction memory is
    context-bound and individual.

## Candidate experiments (sketches for the design review; numbers assigned at prereg)

Both run in the **Exp 60 water classroom**: its apparatus, rescue, the US-free 4.3 s window and the bridge-truth
surfacing measure are already built and proven live.

### A. "Fear relief teaches WHICH act"

- **Hypothesis:** with the escape act stripped of its innate name affinity, a conditioned agent learns to pick the
  act that **ends** the feared situation over equally successful acts that don't, only if fear reduction is
  credited.
- **Roster:** `swim_up` (the existing escape actuator, renamed; the only act that surfaces), `swim_forward`,
  `turn`, `sneak`. All four must succeed as tools underwater (preflight). Exploration is on, identically in every
  arm.
- **Phases:**
  1. Exp 60 conditioning.
  2. An instrumental phase with **no US**, so only fear termination can reinforce.
  3. A probe with exploration off.
- **Arms** (8 seeds each): RELIEF; FEAR-ONLY (the decisive ablation); ABLATED-FEAR; and, at n=3, an Exp 60
  roster positive control.
- **Primary measure:** P(first act = `swim_up`) per seed (chance 0.25).
- **Gates:** RELIEF median ≥ 0.67; FEAR-ONLY ≤ 0.33; exact permutation p < 0.05.
- **Confounds to guard:**
  - oxygen-relief credit must be exactly 0 inside the window;
  - rescue exits are not credited;
  - tool-success links must be flat;
  - exploration visit counts.

### B. "Fear can be unlearned"

- **Design:** CS-alone exposure under the game's `water_breathing` effect, with **response prevention**: an agent
  that escapes never experiences the CS alone.
- **Arms:** EXTINCTION, NO-EXTINCTION (predicted flat) and ABLATED-FEAR.
- **Primary measure:** post-extinction P(surface before the US).
- **Gates:** NO-EXT median ≥ 0.8; EXTINCTION ≤ 0.33; p < 0.05.
- **Mechanism measure:** the threat need falls below θ only in EXTINCTION.
- **Secondary, under 2b only:** savings.

## Owner decisions (asked at the start of the work)

0. Whether Experiment A rides roadmap 1.4's Phase 5 cluster-keyed relief store (entering through its review) or
   the existing `cluster_reward` table (with the roadmap corrected).
1. Experiment A's roster rename and exploration-on (a fingerprint change versus Exp 60).
2. Whether relief credit requires a bridge-truth exit within Δ of the act.
3. Extinction as 2a or 2b, and whether safety travels in bundles.

## Order

1. FearCircuitBridge Dormant (#840; this PR).
2. Experiment A first: it is the action-level question and reuses a proven apparatus. It enters through the
   relief-store front gate above.
3. Experiment B after the 2a/2b decision.

Side findings (other dead fear-side wiring) are tracked in [#1074](https://github.com/dennys246/Maxim/issues/1074).
