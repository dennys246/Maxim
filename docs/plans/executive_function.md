# Executive function: a staged, bio-faithful design

**Status: PROPOSED (2026-10-04).** Tracking issue: [#1073](https://github.com/dennys246/Maxim/issues/1073).
Prompted by #841: plan_manager's replan hint has never worked (owner: "an opportunity to explore an executive
function and frontal lobe system"). Nothing here is built. Everything enters as `[engineering]`, and each stage
gets its own plan and a four-lens design review.

## What exists today (inventory, 2026-10-04)

Nothing executive has behavioural validation. Of the 27 components checked, by non-test callers:

- **Live, on the LLM path (`run_agentic_loop`):**
  - "PFC" multi-cycle deliberation (`ready_to_act`, Jaccard convergence) and the PFC preamble;
  - `ThoughtGate` (its energy check never runs);
  - NAc per-goal reward bias shifting ThoughtGate's threshold;
  - `WorkingMemorySet` (a capacity-64 FIFO; `receive_valence` has no caller);
  - `FearGatedExecutor` (a fixed veto) and `AutonomyController`;
  - the Default Network gate, arbiter and reflex inhibition;
  - the Statistician (nothing executive reads it).
- **Built but never reached:**
  - `PlanManager`: no plan is ever created; the replan hint is Dormant (#841);
  - `DecisionEngine`/`AdaptivePlanner`: stored, never read, which contradicts DECISIONS.md 2026-01-04;
  - `PreemptionCircuit`, `GoalAgent` and `PlanHistoryBridge`'s read side;
  - `ExecAgent.recall_deep` (#845).

  Tracked in [#1075](https://github.com/dennys246/Maxim/issues/1075).
- **Absent:** prospective memory, task-set switching, and gating of what enters or stays in working memory.

## The gap, against the code

The substrate action path (`NAc.recommend_action`) is a **one-tick argmax**: causal links, reward bias, cluster
reward bias summed over the active clusters, and drive affinity. Nothing carries from one tick to the next except
learned tables. In neuroscience terms it is a striatum with no PFC above it, and it explains three gaps the 1.4
roadmap already measured:
- "a shore start never descends": nothing holds "I am going for the food" across the dive's steps;
- R4: trace credit reaches step three, but lands on a different interoception cluster at each step (no stable key);
- E1: nothing gives hysteresis, so near the eat/escape crossing the choice can flip every tick.

The missing prefrontal contribution is **context**: a maintained goal that biases action competition even where
the current sensory state carries no value, and that gives the striatum a stable key to learn on across a
multi-step path.

## Map: frontal function → Maxim

| Function (region) | Maxim today | Stage |
|---|---|---|
| Goal / task-set maintenance (dlPFC) | absent on the substrate path; `WMEntry.goal_tag` is an unused label | **1** |
| Gating PFC updates (BG, PBWM) | `ThoughtGate` (LLM path); the Exp 42 drive gate (momentary) | 1 (innate rule), 2 (learned) |
| Conflict / error monitoring (ACC) | LLM-path stall detection; plan_manager replan (dead) | 3 |
| Outcome expectancy (OFC) | Wire-4 fear (a step predictor); dormant pre-activation and cerebellum | 4 (the roadmap's graded predictor; do not duplicate) |
| Situation value (vmPFC) | `cluster_reward_bias` | no new piece |
| Prospective memory, options | none; motor engrams (#909) lack a caller | 5 |

## Stage 1: a gated goal slot

While a drive is held as the active goal, the NAc sees one extra, **stable** context key (`goal:hunger`) when it
scores and credits actions. So actions on the way to the food build goal-keyed value that stays the same at the
surface, mid-column and at the alcove.

- **Slot:** one per agent; its content is a drive name.
- **Gate in (innate prior):** the drive's need crosses the existing 0.5 floor (reuse it, don't add a constant), and
  the slot is empty or holds a weaker goal by a margin.
- **Gate out:**
  - relief: need below a lower release threshold;
  - timeout: N ticks;
  - switch: a stronger drive exceeds the held one by the margin.

  The two thresholds give hysteresis.
- **Read:** the slot's content is passed to `recommend_action` as its own typed keyword (e.g. `goal_context=`),
  which feeds the existing additive cluster-bias sum. No scorer change.
- **Write:** on consummation, the existing eligibility trace's recent tools are credited by
  `update_cluster_reward(agent, "goal:<drive>", tool, …, source="goal_relief")`. This is the R4 routing audit with
  the stable key it lacks.
- **Wiring trap:** the same clusters feed pain keying, the Wire-4 fear read and the situation cue. The goal key must
  never reach them, or the agent learns a "fear of being hungry" everywhere. So it is a typed keyword, with a guard
  test that pain keying never sees it.
- **Behaviour tiers:** the gate thresholds are an innate prior (declared, with a Stage 2 follow-up to learn them);
  the goal-keyed values are learned.
- **Naming:** "PFC deliberation" already means the LLM think loop. Use `GoalSlot`/`TaskSet` in code.

**Front gate.** Stage 1 is a new rule, and R4's first work is "route existing trace credit to the selection surface
**before any new rule**". So Stage 1 is only a **candidate R4 consumer**. It enters if the routing audit (after
#888/#889) shows routed credit fails for lack of a stable key across a dive, and then through its own four-lens
review. That failure is the written reason a goal key would be needed.

**The goal key's type (the write side, not only the read).** `update_cluster_reward` documents `cluster_id` as an
EC node id, so a `goal:<drive>` string in that table is the same silent-failure magnet as a string modality. The
plan must name:
- a typed key, or a separate goal-keyed table beside it;
- how decay and pruning treat it;
- its bundle policy. Recommended: local-only in v1, never in fear or Exp 56 bundles, until a transfer experiment
  earns it.

### Candidate experiment: "goal-held descent" (a sketch for the four-lens review)

The protocol runs on E1's treasure-dive column, with no LLM in the action path. A harness demonstrates a hungry dive
(propose-only), then probes with fresh starts at the surface.

- **Arms:**
  - **GOAL;**
  - **SLOT-ABLATED:** the shipped path, predicted never to descend;
  - **READ-ONLY:** maintenance without goal-keyed credit;
  - **SHUFFLED-GOAL:** credited under another drive.
- **Primary measure:** per seed, P(descend and eat), **hungry minus sated**.
- **Proposed gates:** GOAL ≥ 0.5; ABLATED and READ-ONLY ≤ 0.1; permutation p < 0.05.
- **Mechanism measure:** the winner's `learned_bias` is attributed to `goal`.
- **Offline replays first:**
  1. Does the shipped table already descend? If so, the experiment is moot.
  2. What is the hungry interoception cosine, surface vs submerged? If ≥ 0.85, the interoception key is the cause.
  3. Does `causal_pos(sink)` dominate even when sated?
- **Confounds:**
  - fear coupling from oxygen pain;
  - regen–hunger coupling;
  - the 5-consecutive same-tool cap.
- **Not claimed:** planning, model-based search or anticipation. The claim is that a maintained goal lets
  one-shot-demonstrated multi-step credit transfer to an unvisited start.

## Later stages (each enters only when a rung names it)

- **Stage 2, learned gating (PBWM):** thresholds learned from whether holding paid off. Earned on E1 by dithering
  near the crossing.
- **Stage 3, conflict/error monitor (ACC):** a small top-2 margin, or a goal timeout, releases the slot or raises
  exploration. It is the substrate twin of plan_manager's replan, and on the LLM path it could wake deliberation.
- **Stage 4, outcome expectancy (OFC):** the roadmap's graded predictor; stays under that plan.
- **Stage 5, options and prospective memory:** a consolidated goal-keyed sequence becomes an option, executed as a
  motor engram (#909); a second slot holds a pending goal (surface for air, then resume the food).

## What this means for existing pieces

- **Revives** `WMEntry.goal_tag`.
- **May subsume** the Exp 42 drive gate. Dormancy, not deletion, and only if Stage 1 earns and a replay shows the
  gate is redundant.
- **Leaves** `plan_manager` (the whole module Dormant since 2026-10-04: no plan is ever created, and the replan
  hint never worked; #841) and ThoughtGate's LLM deliberation alone for now.
  Later, a Stage 1 goal could be published one-way to plan_manager as its objective, so both paths share one goal.

## Owner decisions (asked when Stage 1 starts)

1. The gate priors: the gate-in margin, the release threshold and the timeout N. These are innate priors, with
   Stage 2 to learn them.
2. The goal key's home: a typed key in the cluster-reward table, or a separate goal table.
3. Whether goal-keyed values may ever cross agent bundles. Recommended: no, in v1.

## Risks

- The demonstration could be the whole story: "sink when hungry" with no position sense. A dry-room probe arm
  bounds the claim.
- Without a demonstration, nothing proposes descent. That is the "never sampled vs not credited" split the
  roadmap already names.
- A string "goal" modality is a silent-failure magnet. Type it.
