# Hybrid arbitration — the substrate acts by default, and the LLM deliberates when the substrate is unsure and may override it

> **PROPOSED 2026-11-11 — plan only, no code, no prereg; a PARALLEL line, not a 1.4 rung.** This was an owner idea
> on 2026-11-11, and the owner decided its direction the same day (DECISIONS.md, 2026-11-11 entry).
>
> - **Owner decisions:** HA-O1–HA-O16 were decided on 2026-11-11 and 2026-11-12.
> - **Reviews** (§Review record):
>   - a three-lens round plus its delta round;
>   - an owner fold;
>   - a final six-lens round: concurrency, hardware safety, adversarial authority, statistics, learning dynamics,
>     and scope.
> - **This file is Part A, the arbitration.** The owner's 2026-11-12 split moved four pieces into their own plans:
>
>   | Piece | Plan |
>   |---|---|
>   | Part B, the idle-time default mode network | [idle_dmn_coordinator.md](idle_dmn_coordinator.md) |
>   | Mid-action cancel | [sem_interruptibility.md](sem_interruptibility.md) |
>   | The learned hold window | [learned_hold_window.md](learned_hold_window.md) |
>   | The LLM → substrate advice layer | [foreign_advice_layer.md](foreign_advice_layer.md), shared with social_referencing |
>
>   The per-action chooser stamp joins the M1 provenance line (§Front-gate #3).
>
> **Now:** paper, plus Step 0 (§A.4), an analysis of logs that already exist.
> **No `src/` change for this line before the 1.4.0 cut.** That one rule also keeps `NAc.recommend_action`'s selection
> path free of changes while E1's mechanism lives there. Until a check exists it is mechanization backlog M89.
> **Never active in an E1–E3 arm** (unenforced until mechanization backlog M88 closes).
> **Claims:** a 1.5 *exploratory* result is the realistic target. A 1.5 EARNED claim is not, because the full design
> depends on about ten unbuilt plans or stages owned by five or more other lines (§A.4).

**One-line thesis.** Today a run has one controller, because `aut_mode` is either/or. The hybrid has both:
1. **The substrate is the automatic movement.** It proposes and acts on every decision, and it waits only briefly when
   it is unsure.
2. **The LLM deliberates on the uncertain decisions.** It may override what the substrate does next.
3. **A hard floor protects danger responses** from the LLM.

If the hybrid matches each controller where that controller is strong, at a fraction of the LLM calls, it becomes a
candidate for the default.

---

## Why (the evidence that points here)

| Record | What it shows for this plan | Status / caveat |
|---|---|---|
| [Exp 49](../experiments/49_two_joint_centering.md), arms B (LLM) and C (substrate) | **Substrate:** centred 5/10 in 4.65 s, but trapped by the linear array's front/back fold (far bins 0/4; 77 fold-divergent credits). **LLM (Qwen32B):** centred 4/10 in 86.8 s, at roughly 40–90 s per action, and it crosses the fold (far bins 2/4). The write-up calls this "the cleanest argument yet for the layered design". | COMPLETE 2026-08-04; n=10 per arm. It ran on `SimulatedController`, so a re-run needs no robot. Arm C ran with `MAXIM_NAC_MIN_CONFIDENCE=0` and without `focus_on_sound` (amendment 4). Its code predates the decomposition, and its dense-arm data is lost. |
| [Exp 37](../experiments/37_cross_session_graduation.md), [38](../experiments/38_counter_prior_substrate.md), [40](../experiments/40_counter_prior_goldilocks.md) | When the substrate reaches the LLM only through prompt annotation, the coupling follows the LLM's prior: it is either dominated by the prior or amplifies it. In Exp 38 under a wrong prior, R1 moved +2.25 SD the wrong way (+2.31 on re-check). The ablations lowered warming from 0.60 to 0.40/0.34/0.21, so the substrate carried part of that effect. | Bounded 2026-10-25 ([re-assessment](../experiments/narrator_defects_reassessment.md)); T1-2 STALE until B4. A 2026-11-02 code reading suggests Exp 38's hearth harm may never have reached the body in LLM mode (`embodiment=None`); a probe and correction are owed. Do not cite these as "language beat experience". |
| Exp 44 counterfactual, [44b](../experiments/44b_pilot.md) | Showing the substrate's view to the LLM moved its choices: 14/15 flipped toward safe, from offline re-queries. It has **never been shown to improve an outcome.** | Exploratory; 44b is a pilot (F1–F4 open). |
| [Exp 42](../experiments/42_substrate_primary_preference.md), Exp 60/61/62 | The substrate acting alone is near-deterministic and fast. Exp 60's median escape is 1.72 s, inside a 4.34 s window. *Conjecture, not measured:* a bare local LLM at Exp 49's per-action latency would miss that window. | Exp 60/61/62 PARTIAL (narrow: floor matched) since B1. |

No experiment has tested a hybrid. Everything above is motivation, not evidence for the claim.

## Front-gate (CLAUDE.md principle 3)

**Rides on what exists:**
- **The substrate proposal.** `runtime/substrate_proposal.py::propose_via_substrate` returns `None` when no tool
  clears `min_confidence`. `NAc.recommend_action` computes `runner_up_score` and `learned_margin` from raw scores,
  and every call emits them on the `NAc_RECOMMEND` provenance event whenever sim logging is active
  (`nac.py::_emit_recommend_action_event`).
- **The asynchronous LLM worker.** `agents/llm_worker.py::LLMWorker`.
- **ExecutiveFunction Stage 3, the ACC conflict monitor, as the arbiter's home** (HA-O10). The owner revived Stage 3
  for this line on 2026-11-12, and [executive_function.md](executive_function.md)'s banner records it. Stage 3's
  goal-slot release assumes Stage 1. This line does not need that part, and the banner says so.
- **One composition point.** [loop_stack_builder.md](loop_stack_builder.md) (1.3.6).
- **Queued-command withdrawal.** `embodied_runtime/media_loop.py::_clear_motor_queue` runs on the media thread, and
  the tickets it supersedes resolve as `superseded`.

**Not reused as the arbiter:**
- **ThoughtGate.** It scores only when percept text is present, and `should_think` mutates refractory state.
- **DN's ThalamicGate.** It handles visual percepts only.

**Net-new:**
1. **The arbiter.** The loop's branches are exclusive predicates: `runtime/loop_state.py::_substrate_tick_due`, and
   the `aut_mode != "substrate-primary"` submit guard in `runtime/agent_loop.py`.
2. **Loop structure for hold-then-override** (§A.1b; concurrency review, BLOCKER). Today the loop has one proposal
   slot (`ctrl.pending_proposal`) and one submit clock (`ctrl.last_llm_submit_time`, stamped by `substrate_tick` too).
   With that structure:
   - an escalation is never submitted while the substrate's proposal fills the slot;
   - each substrate tick pushes the next LLM submit back;
   - a landing LLM answer silently overwrites a held proposal at §2.

   The build needs four pieces, listed as build items:
   - a **held-proposal slot with a deadline**;
   - a **submit gate keyed on the worker's attempt state**, not on the slot;
   - **split cadence clocks**, because the same stamp also drives `_submitted_recently` and D13's liveness check;
   - **one typed replace site** that records each override.
3. **Per-action chooser stamp.** It joins the M1 provenance line. Its fields:
   - the **chooser**: a closed key `substrate | llm | substrate_advised`. It must not reuse
     `strategy_used="substrate-primary"`, and actions an LLM chains or runs in parallel inherit their chooser;
   - the **escalation reason**: a closed key `none | margin | novelty | rpe | text`;
   - **pre-emption events**: `danger | stale | override`;
   - `basis_seq`;
   - decision-start and dispatch timestamps, on the wall clock and the experience clock;
   - hold time, and whether the LLM answer landed inside *h*;
   - LLM calls and tokens attributed to the decision, including dropped stale proposals;
   - the size of the suppressed set.

   Run manifests also stamp the **resolved** `aut_mode`, LLM model, quantization, n_ctx, temperature/seed, the
   serving box and the lane placement (local, peer or cloud), each written at the point of writing. Today sim
   `report.json` stamps none of this. A new frozen stamp record takes CC3 path (a), with `extra`.
   `LLMProposal` is an in-process, runtime-ephemeral type, so CC3 applies to it only if it crosses a wire.
4. **The margin carried on the proposal.** Today `propose_via_substrate` drops every key except tool, params,
   reasoning and confidence. `SubstratePolicy.select` (the floors' contract) needs the same field.
5. **Explore-first bookkeeping at dispatch time.** `recommend_action` records `_ever_selected` and `_visit_count`
   when a proposal passes the gate, which is before the hold and before any override. If the LLM replaces an
   explore-first trial, the untried tool is marked tried but was never dispatched. A tool the LLM dispatches is never
   marked at all. The build moves this bookkeeping to dispatch time, the same split as Step 0's scoring read.

   This also changes substrate-primary. A gated proposal that confirmation, `ToolPolicy` or the fear gate refuses
   would no longer count as selected, and explore-first would retry it. That differs from what Exp 41, 42 and 60 ran,
   and it sits inside `recommend_action`, the E1 seam. So the old behaviour stays for substrate-primary behind the
   mode switch, and a **selection-identical guard test** pins it. Only the arbiter's mode uses dispatch-time
   bookkeeping.
6. **Prerequisites, unbuilt:**
   - **GL3.B7** in [thalamic_relay.md](thalamic_relay.md): pre-empting LLM-primary's in-flight plan, by dropping a
     stale *LLM* proposal through `basis_seq`. **Not GL3.B4**: B4 drops a pending *substrate* proposal. B7 waits for
     a rung that names it, so the first evidence slice (§A.4) does without it.
   - **An efference-copy exclusion.** "Stale" must exclude self-caused events (§A.1b.5).

## The biology, stated precisely

The default mode network is task-*negative*: active at rest, and the site of replay, prospection and self-referential
processing. That half of the idea now lives in [idle_dmn_coordinator.md](idle_dmn_coordinator.md).
`default_network/` is a salience / exogenous-attention reactive layer under a historical name (NAME-ONLY since PR
#492). HA-O5 keeps that name.

| Role | Biology | Maxim | Bio-mapping |
|---|---|---|---|
| Learned value | Ventral striatum / NAc | `NAc` value tables | ANALOG (existing) |
| Routine action | Habitual / model-free control (dorsolateral striatum) | Substrate selection | ANALOG; the module is named for the ventral system |
| Novel or complex action | Goal-directed / model-based control (PFC, dorsomedial striatum) | LLM | ANALOG (existing) |
| Who acts | Uncertainty-based arbitration (Daw, Niv & Dayan 2005); ACC conflict monitoring | The arbiter, as Stage 3 | ANALOG, and **one-sided**: it thresholds only the substrate's uncertainty |
| The hold | The subthalamic nucleus raises the decision threshold in proportion to conflict (Frank 2006) | *h*, scaled by conflict (§A.1b.1) | ANALOG; *h* is an engineering choice |
| Pain interrupts | Pain's interruptive function (Eccleston & Crombez 1999) | The danger trigger (§A.1b.4) | ANALOG |
| The override refractory | No biological basis: response inhibition is transient | §A.1b.3 | `[engineering]` only |

**Keramati, Dezfouli & Piray (2011)** model a choice between habit and deliberation. Their model supports *skipping
deliberation when time is costly*. It does not support acting and deliberating at once with a later override, which
is what this plan does. So it is the starting point for the learned hold ([learned_hold_window.md](learned_hold_window.md)),
not a justification for §A.1b.

## Relation to earlier designs

- **The 2026-07-17 HYBRID decision.** See [deferred/hybrid_substrate_reflex_runtime.md](deferred/hybrid_substrate_reflex_runtime.md),
  the DN reflex under LLM cognition, deferred until a second body exists. It rejected `aut_mode="hybrid"` for a
  reflex, and that is right for a reflex. This line is a per-decision choice made at the cognition layer. The two
  designs compose.
  - **SF-1 (one-way inhibition) carries over.** The deliberate controller can suppress the habit. A bottom-up signal
    acts only by escalation or interrupt, never by peer veto.
  - **BL-4 (NAc separation) does NOT carry over.** One shared NAc is the point of the design (HA-O8, HA-O9).
- **[thalamic_relay.md](thalamic_relay.md).** GL3.B7 is a prerequisite (above). GL3.B4's once-per-breach-episode
  livelock guard is the precedent for the refractory.
- **[advisor_mode.md](advisor_mode.md), the atonia gate.**
  - The arbiter's mode value cannot compose with the advisor/atonia profile.
  - `ToolPolicy` stays innermost for both controllers' dispatches.
- **[baseline_floors.md](baseline_floors.md) BF2.** It binds every claim of this line.
- **[social_referencing.md](social_referencing.md).** Its foreign layer is now [foreign_advice_layer.md](foreign_advice_layer.md),
  which both lines consume.

---

## Part A — arbitration

### A.1 Mechanism sketch

On each tick where the loop would act:
1. **The substrate proposes** (a NAc lookup).
2. **The arbiter escalates** if ANY of the following holds:
   - `recommend_action` returned `None`.
   - `learned_margin` < *m*, computed on the **unsuppressed** candidate set. A negative margin counts as below *m*; it
     occurs where the explore-first gate or the drive gate overrode the argmax.
   - **Novelty above *n*.** `bio_integration.py::situation_novelty` returns `None` when nothing is comparable, and a
     `None` escalates.
   - **Recent surprise above *r*.** This is a per-agent, decaying last-outcome RPE. Today's `NAc.last_rpe` is one float
     for the whole instance and never decays.
   - **Pending addressed text.** This is new code: a side-effect-free predicate extracted from
     `has_meaningful_input`'s address checks. The loop's `pre_tick_gate` locals are broader wake predicates.

     It is **budgeted per source**, so any chat participant cannot force the hybrid into llm-primary indefinitely
     (adversarial review S5).
3. **An escalation submits** to the LLM worker. In the S→L cells of HA-O3 the prompt carries the substrate's proposal
   plus Wire-A and Wire-1.
   - All percept and chat text in that prompt is spotlighted (`utils/content_safety.py::frame_tool_output`).
   - Ingested bundle content passes the bundle scrub (sharing_threat_model row L) before it can reach the prompt.
   - Both are prerequisites of the S→L cells.
4. **The substrate does not wait** (§A.1b). The held proposal dispatches when *h* expires.
5. **Credit is per chooser** (HA-O8):
   - `substrate`: substrate-primary's rule, including the tool-success floor.
   - `llm`: drive relief only.
   - `substrate_advised`: drive relief only, like an LLM choice. Without it, advice would reach the substrate's own
     values through dispatch credit.
     - **When it applies.** The outcome with advice differs from the outcome without it, and gate-crossing counts:
       if `recommend_action` returns `None` without advice, but the advised tool clears `min_confidence` with it,
       that action is `substrate_advised` too.
     - **The guard.** foreign_advice_layer.md's deletion-proven guard covers that `None`-to-tool case.

   Three credit paths ignore the chooser, and HA-1's prereg declares them as confounds unless the stamp is carried onto
   them:
   - `TemporalCreditDistributor`'s eligibility traces and 300 s anchors;
   - ToolPainBridge's pain outcomes;
   - `set_pending_operant_action`, which keeps the most recent action per agent. This path matters only under
     operant-only (cradle) credit.

   Without that, B's relief or pain can credit A's nodes.

**Fixed in advance.** *m*, *n*, *r*, the RPE decay, the text budget and *h* are frozen in the portfolio prereg and
committed by hash. They are calibrated on pilot bearings and seeds kept separate from evaluation. The pilot bearings
cover **both strata**, near and far of the fold. A sensitivity grid is fixed in advance.

**Behaviour tier.** The arbiter is an **innate prior**. The safety floor (§A.1d) is an **invariant**.

**Where the hybrid is not degenerate.** In text-percept sims every turn escalates, so the hybrid only means something
in sensor-driven worlds: the survival world, the robot, and Exp 49's simulated robot.

### A.1b Pre-emption: the substrate acts by default, and each controller may interrupt the other for a different reason

1. **The hold *h*.**
   - When the arbiter escalates, the substrate's proposal waits up to *h* and then dispatches. An LLM answer that
     lands inside *h* replaces it at the typed replace site.
   - *h* is a **deadline on the held slot**, checked once per pass. It is never a blocking wait: a sleep would stall
     intake and the pain drain.
   - *h* is **scaled by conflict** (1 − margin), in the spirit of the STN, and **capped at *h_max***, which is frozen in
     the prereg. A negative margin would otherwise make *h* exceed its base without limit.
   - ***h* = 0 for any protective or danger-selected proposal, in every version (v1 and learned).** This is part of
     the safety floor (§A.1d) and has its own guard test.
   - Its base value is per world, and stated in **one named clock per world**: the experience clock where the world
     has one, otherwise the wall clock. Both clocks are stamped on every decision.
   - For example, the base is ≈ 0 inside the survival fear window, and longer in Exp 49's centring.
2. **The LLM pre-empts on authority.** It replaces a held proposal, withdraws queued commands, and overrides the
   ongoing behaviour, meaning what comes next.
   - At local latency the LLM rarely lands during an action, so its effect is on the next action and on the policy.
   - Pre-emption is checked **between passes only**. In Minecraft, `call_action` blocks the loop for the whole
     action, so nothing pre-empts during it.
   - Mid-action cancel is [sem_interruptibility.md](sem_interruptibility.md).
3. **The override refractory** (`[engineering]`, no biological basis). The habit the LLM overrode is suppressed:
   - **Keyed by** (agent, situation cluster, tool), in memory only, never persisted.
   - **Cleared by** a change of situation or a world episode boundary. **Any danger event clears the whole
     refractory** (§A.1d item 2 restates this one rule). Episodes are
     defined per world: in Minecraft, death/respawn or a breach episode; in Exp 49, a trial; on the robot, a change
     of situation only.
   - **Capped** in how many tools can be suppressed at once. Without a cap, enough suppressed habits push every margin
     negative, and the hybrid silently becomes llm-primary.
   - **NAc decay** continues on a suppressed tool's value. The refractory therefore has a **maximum duration**
     *D_max*, measured in decisions on the world's named clock. It is derived from the NAc decay rate, so that a
     suppressed value cannot decay below its pre-suppression rank, and it is frozen in the prereg. The refractory lifts at
     *D_max* even with no change of situation. On the robot, a change of situation is the only other way it clears,
     so *D_max* is what stops a habit staying suppressed for a whole session.
   - The suppressed-set size is reported per decision.
4. **The substrate pre-empts only on typed triggers, never on disagreement.** The two triggers:
   - **danger:** a pain, fear or vital-drift event, through the escalation path SF-1 allows;
   - **staleness:** defined below.

   Under one of these triggers, the substrate drops the LLM's pending answer through the typed pre-emption handler,
   not `_drop_stale_proposal`'s planning-failure path (which counts toward `planning_failed` exhaustion). After a
   danger drop, only the substrate reselects, with no LLM re-arm in the same pass.
5. **Staleness, defined.**
   - An LLM answer is stale when a non-self-caused event on a tracked right arrives after its `basis_seq`.
   - **"Self-caused" is decided by attribution, not by a heuristic.** The mechanism is GL3.B2's invocation-pid join,
     or the TR23 writer-kind stamp ([thalamic_relay.md](thalamic_relay.md)). A tool's reafference carries the
     dispatching invocation's pid; a lagged world event at the same locus does not.
   - **In practice this makes staleness danger at a lower threshold.** The tracked rights are thalamic threshold
     tracks such as `nociceptive_fast`. A non-danger track counts only if the build plan names it.
   - Events caused by the substrate's own held-then-dispatched action are excluded (efference copy). Otherwise every
     in-flight LLM answer would be stale.
   - An answer computed before the substrate acted is applied to the next decision only, never retroactively.
6. **Two pre-emptions in one pass.** The refractory is applied only when an override actually dispatches. A danger
   drop in the same pass wins.

### A.1c Two-way coupling (HA-O3)

- **S→L.** The LLM sees the substrate's view. That view has moved LLM choices but has never been shown to improve
  outcomes. Here it is a hypothesis.
- **L→S.** The substrate takes the LLM's advice through [foreign_advice_layer.md](foreign_advice_layer.md). The advice
  is credit-isolated, never persisted, never promoted, carries a frozen trust weight, and has zero weight on
  protective actions. Advice-flipped actions are stamped `substrate_advised`.
- **The arbiter reads the margin before advice is added.** So L→S does not *directly* change the escalation rate.
  It still changes it *indirectly*, because advice changes what the agent experiences. Escalation is reported per
  cell.
- **What the both-on cell can claim.** It is a composite-system result, which is fine for efficiency. It is never
  reported as substrate capability.

### A.1d The safety floor (`[engineering]`, invariant tier)

The adversarial, hardware, learning and concurrency reviews each found this hole independently. A learned escape
(Exp 60's `escape_water` is a learned selection, not a reflex) could be overridden by one wrong or injected LLM
answer, and then suppressed by the refractory while the agent drowns. The floor that closes it:
1. **No LLM action can override, withdraw, cancel, delay or suppress** a protective action.
   - **The one definition of "protective" lives in [sem_interruptibility.md](sem_interruptibility.md)**, and this
     plan cites it.
   - It has a static part: affordances that the device owner declares protective **in code**, on the backend class.
     This covers escapes, wake, sleep and the stop path.
   - It has a dynamic part: any selection that a danger trigger makes or re-arms, whatever the tool is. This catches
     an escape carried out through a generic `move` or turn-away.
   - **No loaded or LLM-generated spec can claim protective membership.** Membership grants immunity, so spoofing
     it would be an attack: a harmful affordance named as an "escape" would then block the LLM's own correction.
2. **Any danger event clears the whole refractory** (the same rule as §A.1b.3).
3. **Override, withdrawal and suppression are effects.** They pass the same refusal path and logging as dispatches.
   The executor-side gates (`ToolPolicy`, `FearGatedExecutor`) refuse dispatches but never see these. This matters
   most when the LLM is a remote leader, whose answers now gate inactions too.
4. **Neither the advice layer nor any *h*, v1 or learned, may touch protective actions.** *h* = 0 for them.

**Regression guard (built with the arbiter, a build-exit item):** deletion-proven tests that each of 1–4 holds.
For example, an injected "stay in the water" escalation must leave the escape dispatchable under drowning pain.
Until the arbiter exists, this rule is mechanization backlog M89.

### A.2 The habit-formation prediction

LLM-chosen actions that pay off should become confident substrate selections. So the escalation rate for the
**margin** reason should fall in a stable world and rise again after a contingency change that comes with outcome
feedback. This is not a devaluation test, because habits are by definition insensitive to devaluation.

**Only the margin-reason curve is pre-registered.**
- Novelty falls as EC grows, whatever arbitration does.
- The RPE reason rises after a change by definition.
- The refractory shapes the selection frequencies.

**The baseline is a counterfactual arbiter** replayed over substrate-primary logs, which gives the decline that
arbitration does not cause. The curve is reported split by chooser, including `substrate_advised`.

**A floor on LLM-chosen dispatches per member** is pre-registered too. With the substrate acting by default and
*h* ≈ 0 in fast worlds, the LLM may choose almost nothing.

**The claim's name is "LLM-scaffolded habit formation"**, because credit on LLM-chosen actions is imitation through
reinforcement. The research-claim rule holds only as "the state lives in the substrate", never as "the substrate
learned independently".

### A.3 The claim and its bounds

All of this line's claims are bound by BF2 (none earns before B2) and by the project's pass-criteria rule: **effect
sizes with confidence intervals, beating the comparator by ≥ 2 SD across seeds, with no p-values and no
Bonferroni/Holm in pass criteria.**

1. **Primary: per-decision efficiency across a portfolio** (HA-O6).
   - **Per decision**, from the stamp: LLM calls and tokens, and median and p95 latency split by chooser, at matched
     **run-level** outcome. Delayed credit makes a per-decision outcome undefined.
   - **The verdict forms.** Each is computed on paired per-seed differences (hybrid − comparator), and the SD is the
     SD of those paired differences, never the pooled per-arm SD.
     - **IMPROVE:** the mean paired difference beats the comparator by ≥ 2 SD, and its CI excludes 0.
     - **NON-INFERIOR:** the lower CI bound of the paired difference is above −δᵢ.
     - **DEGRADE:** anything else.
     - **The zero-tolerance safety DEGRADE is a separate override.** Any death inside Exp 60's 4.34 s window is a
       DEGRADE, whatever the statistics say.
     - **The BF2 floor outcome is reported per member.** A member that does not beat BF2's floors on outcome
       returns its BF2 status, `PARTIAL (narrow: floor matched)`, not a clean NON-INFERIOR or IMPROVE. Exp 60 and 61
       are already floor matched.
   - **Each member pre-registers its own comparator role**, so the portfolio is not guaranteed to fail:

     | Member | Outcome role | Efficiency role |
     |---|---|---|
     | Exp 49 | Per stratum: non-inferior to the *better* controller in each fixed stratum (near vs far bins), never pooled into a single "beats both" (a Simpson's-paradox risk) | vs its own llm-primary arm |
     | R3 offline | Non-inferior to substrate-primary | vs llm-primary; INCOMPLETE until B5(a) |
     | Exp 60 | **Safety role.** It wins over the ceiling role: zero tolerance, and non-inferior to substrate-primary | vs llm-primary; INCOMPLETE until B5(a) |
     | Exp 62 rung A | Non-inferior to substrate-primary; the A.2 curve is secondary | vs llm-primary; INCOMPLETE until B5(a) |
     | Exp 61 | Ceiling member: non-inferior to substrate-primary | Relative to llm-primary, never to substrate-primary, which makes zero calls; INCOMPLETE until B5(a) |
     | Hardware orient | Per its prereg | Per its prereg |

   - **The efficiency half cannot be computed for any survival member until an llm-primary leg exists.** So the
     portfolio's efficiency claim waits for B2 and B5(a). A count table of survival NON-INFERIORs is **not** an
     efficiency result, and nobody may read it as one.
   - **Each member returns** IMPROVE, NON-INFERIOR, DEGRADE, PARTIAL (floor matched) or INCOMPLETE.
     - INCOMPLETE means the claim **waits**. It never counts as a pass.
     - Any safety DEGRADE vetoes the portfolio claim.
   - **The report is a count table.** There is no pooled statistic.
   - **Fewer calls proves nothing by itself.** Substrate-primary makes zero LLM calls, and so does an arbiter that
     never escalates, so the efficiency bound always comes with outcome non-inferiority.
   - **Selection bias.** Escalated decisions are harder by construction, so a split by chooser is descriptive.
   - **Every margin is frozen together.** All members' δᵢ, Xᵢ and roles are frozen in **one portfolio prereg** before
     member 1 runs.
2. **Secondary:** the A.2 margin-reason curve, and the 2×2 interaction. The interaction is descriptive unless powered.
   Here "powered" means the interaction's pre-registered CI half-width is ≤ δ/2 at the frozen n.
3. **Capability ("better than both"):** only over BF2's floors, after B2.

Typed aborts (D22) and records without the stamps cannot move any row. Until M1 lands, an operator-attested record
cannot be the sole evidence.

### A.4 Steps

0. **Step 0: calibration from existing logs** (now; no `src/` change, no LLM calls).
   - **The data.** Every `recommend_action` call emits `NAc_RECOMMEND` (`best_score`, `runner_up_score`,
     `learned_margin`, `passed_gate`) while sim logging is active. The committed data, counted 2026-11-11:
     - `56_four_arm.jsonl` is margin-rich: 180 non-null margins against 20 null. Whether its events join to
       per-decision outcomes is the first check.
     - `exp61_pairs.jsonl` has 24 non-null margins against 528 null, and `exp62_rows.jsonl` has 48 against 116.
     - Exp 60 and B1 hold none.
     - Exp 49 has none: its dense-arm data is lost.
   - **If the Exp 56 join fails or is too thin,** Step 0 becomes a substrate-only logging re-run under the
     harness-provenance and prereg rules.
   - **Question:** do the confidence and the margin predict a good outcome?
   - **Stop rule.** A flat curve counts only where the joined events reach a minimum count, written down before the
     join. A thin join is "not measured". If every sufficient world is flat, Part A stops, and that is recorded.
   - **Scope.** It is exploratory, and the join is a script under `scripts/`.
   - **The question Step 0 cannot answer: would the LLM have agreed?** That needs the offline label pass: logged
     states replayed to the LLM on a separate box. `scripts/exp44/rerun_ablated_offline.py` is the template. The
     same pass supplies the labels the echo metric needs (HA-O3).

     `propose_via_substrate` is not a side-effect-free scoring read. It writes EC, re-keys clusters, fires a
     situation cue, ticks drift and pain, and bumps visit counts. So no "shadow" call is allowed inside llm-primary.
1. **The design review.** The four-lens review of the portfolio prereg runs BEFORE any harness is built.
2. **The build, after 1.4.0.**
   - The arbiter is ExecutiveFunction Stage 3, composed in the loop-stack builder.
   - Net-new items #2–#5 are included.
   - The safety floor's guard tests are included.
   - The mode value enters through **exhaustive dispatch**: a typed value set where an unknown value raises.
     - Every `aut_mode ==/!=/in` comparison falls toward llm-primary today. That is 8 sites in `runtime/` plus the
       orchestrator, the bridge and `embodied_runtime/`; re-count them with the grep at build time.
     - `agentic_runtime.py` coerces an unknown value to llm-primary, so a robot run would silently be an LLM run.
     - The llm-path drift tick plus a substrate proposal would tick the body twice.
     - Each remaining site gets its own decision, listed in the build plan.
   - Docstrings that say "substrate-primary only" are updated.
3. **The first evidence slice** (HA-O14): an **Exp 49 re-run, boundary-only, with one hybrid cell (S→L on, L→S off)**.
   - **Arms:** substrate-primary, llm-primary, a floor, and that hybrid cell.
   - **What it does not need:** GL3.B7 and the advice layer. The LLM's answer applies at the next decision boundary,
     A report-only stamp, **`dispatched_since_submit`**, records whether an action was dispatched since the submit.
     It is never a drop trigger, and it is not §A.1b.5's staleness: it counts exactly the self-caused events that
     §A.1b.5 excludes.
   - **Status:** exploratory under BF2.
   - **Design requirements** (statistics review):
     - Trials are **paired** by bearing × seed across arms, and arms are **interleaved**: randomised or ABBA, never
       run in blocks.
     - A **continuous primary**: time-to-centre as censored survival data, or final |θ|.
     - **n per cell computed from δ** before freezing. n=10 is unpowered: the 95% CI for 4/10 is about [0.12, 0.74].
     - **One fixed action set** for every controller (amendment 4 dropped `focus_on_sound`).
     - The `nac.json`/`ec.json` import is pinned by hash. The import file must still exist; check this first.
     - A **deterministic-latency arm**, where the LLM's answer is injected at pass N + k. It is **diagnostic, not
       scored**: it makes the mechanism reproducible. Its answers are **replayed from a recorded LLM log**, never
       produced by a live call released late. llama.cpp sampling under batching is not bit-stable, so only a
       replay is reproducible. Under real latency jitter, results are reproducible only as
       distributions, so latency is a covariate and is stamped on every record.
   - **Unbuilt for Exp 49:**
     - **A floor policy.** B1's `tabular_q`/`random` (`scripts/survival_world/floor_policies.py`) exist only for the
       survival world, so Exp 49 needs a `SubstratePolicy` floor for its own action set.
     - **The LLM arms' cost.** The LLM arm is clock-bound: about 87 s to centre, against a 180 s cap. That is about
       30 min per 10 trials per LLM-bearing arm. The one-cell slice has two such arms, llm-primary and the hybrid
       cell. At a powered n (likely 30 or more paired trials), that is roughly 3 h or more of LLM time, on a box
       separate from the leader.
4. **The full portfolio** (HA-O7), each member a separately pre-registered derivative that owes the confounding and
   bio-faithful lenses, run in this order:
   1. **Exp 49 with the full 2×2,** once the advice layer and GL3.B7 exist.
   2. **R3 offline** (`ScriptedWaterBridge`, no rig).
   3. **Exp 60,** the safety member.
   4. **Exp 62 rung A.**
   5. **Exp 61.**
   6. **The hardware orient runs,** last.

   **What the survival llm-primary legs wait for.** B2 must run first. Then B5(a), which needs three things: a text
   renderer and `llm_worker` path, a latency preflight, and a **prior-free survival variant** ("only where priors do
   not help").

   **Excluded:**
   - **Text sims,** which are degenerate: every turn escalates.
   - **Exp 56,** which is Step 0's calibration source, not a member.
   - **E1–E3,** which may join only as declared campaigns after their claims settle and after M88 is built.
5. **The default** changes only by a later owner decision, after the portfolio claim holds.

---

## Risks

- **R1. Confidently wrong habits.** Exp 49's substrate was confident when the fold trapped it. That is why the
  arbiter reads the margin, novelty and RPE rather than confidence alone, and why Step 0 comes first.
- **R2. Oscillation (SF-1).** Pre-emption is asymmetric, and the refractory prevents re-selection.
- **R3. Wire-A contamination.** The hybrid's LLM sees biases taught by substrate-chosen actions. That is declared
  treatment (HA-O9).
- **R4. Rig contention.** The LLM arms need a separate box, and survival runs queue behind release-critical rig
  work.
- **R5. A metric that rewards silence.** Hence efficiency is never claimed without outcome non-inferiority.
- **R6. Silent mode mis-routing.** Hence exhaustive dispatch.
- **R7. Acting on the uncertain decision.** The substrate does not wait, so it can act on exactly the decision the
  LLM was asked about. *h* bounds this. Hold time and the override rate are reported.
- **R8. Echo.** Advice-flipped actions are stamped and credited under drive relief only, and the advice layer is never
  persisted or promoted. Echo is measured in the frozen post-test against offline labels, with a ceiling
  pre-registered (HA-O3).
- **R9. Injection becomes motor authority.** In the S→L cells, percept text and ingested bundle content reach a prompt
  whose answer can override behaviour. Mitigations: spotlighting, the bundle scrub, the safety floor and the per-source
  text budget (§A.1).
- **R10. DN inhibited most of the time.** `InhibitionMixin.inhibit` blanks all DN output, startle included, for each
  motion command. At the substrate's dispatch rate that could be most of the run. Robot members report DN's
  inhibited duty cycle against a pre-registered ceiling.

## Owner decisions (decided 2026-11-11 and 2026-11-12)

Each entry gives the decision, then the alternatives that were considered.

- **HA-O1 — the switch. DECIDED (a).** A third `aut_mode` value, named for the arbiter (e.g. `"arbitrated"`), not
  `"hybrid"`. It enters through exhaustive dispatch, and it is composed in the loop-stack builder.

  Considered:
  - (b) an `arbitration=` axis;
  - (c) a config flag on llm-primary.
- **HA-O2 — the substrate while the LLM deliberates. DECIDED (owner revision).** The substrate acts by default after
  the hold *h*, and pre-emption is asymmetric (§A.1b), under the safety floor (§A.1d).

  Considered:
  - fast-class-only, with the rest waiting;
  - everything holds;
  - symmetric pre-emption (struck: SF-1).
- **HA-O3 — two-way coupling. DECIDED (owner revision): a 2×2 of S→L × L→S, with the both-on cell as the primary
  arm.** The full 2×2 runs only in Exp 49, and only after the advice layer and GL3.B7 exist (HA-O14). The controls:
  - **A yoked sham** for each direction: another trial's proposal sequence, replayed in order. It keeps the marginal
    distribution and breaks the state contingency. A "prior from a different state" is not used as the sham, because
    in Exp 49 it acts as adversarial advice. A wrong-content cell may be added as its own arm.
  - **A frozen post-test with no LLM** (L→S off), alongside a substrate-primary arm trained for the same number of
    decisions and credit events.
  - **The echo metric**, measured in that post-test against labels from the offline label pass on held-out states,
    with its ceiling written before any run.
  - **The credit firewall** (foreign_advice_layer.md guards). Every argmax the advice flipped is logged.
  - **Harm verified to reach the body** by an instrumented `self_effect` probe.
  - **#1260/#1261 fixed** on any narrator path.

  Considered:
  - the LLM blind in the primary arm;
  - "sees it" everywhere;
  - a credit-writable L→S bias (rejected: the echo path).
- **HA-O4 — an LLM in the idle DMN lane? DECIDED (a): not in the first build** (now in idle_dmn_coordinator.md).
- **HA-O5 — Stage C naming of `default_network/`. DECIDED (a): keep the code names.** Decide again at the DMN
  coordinator's review.
- **HA-O6 — the claim. DECIDED (owner revision): per-decision efficiency across a portfolio** (§A.3).
- **HA-O7 — the world. DECIDED (owner revision): the portfolio of §A.4 step 4,** after the first slice.
- **HA-O8 — the credit rule. DECIDED (a): per chooser, with `substrate_advised` added in the final review.**
  - The closed key is `substrate | llm | substrate_advised`.
  - `_drive_relief_only` today is bound once in `loop_setup.py` and threaded through its builder helpers and
    `agent_loop.py`. So the rule is read at every outcome site.
  - **Consequence for A.2:** LLM-chosen actions become habits through drive relief only. Hence the floor on LLM-chosen
    dispatches and the check on relief events.

  Considered: (b) one rule per run.
- **HA-O9 — Wire-A under a shared NAc. DECIDED (a), absorbed by HA-O3.** Wire-A is part of the S→L factor. It is the
  treatment in the "on" cells and the yoked sham in the "off" cells.
- **HA-O10 — the arbiter's home. DECIDED (a): ExecutiveFunction Stage 3.** The owner revived it on 2026-11-12, and
  executive_function.md's banner records this.
- **HA-O11 — motion precedence. DECIDED (a).** Every loop-dispatched action counts as voluntary, whichever controller
  chose it, and it is subject to DN inhibition at the motor layer as today. `runtime/dn_controller.py::inhibit_for_tool`
  is unwired, and the motion paths call `InhibitionMixin.inhibit` themselves. Both controllers reach the motors only
  through `ReachyMiniController.goto_target`. See R10.
- **HA-O12 — mid-action cancel. DECIDED: staged, and declared per SEM component.** It now lives in
  [sem_interruptibility.md](sem_interruptibility.md). The hybrid pre-empts at boundaries only.
- **HA-O13 — the split. DECIDED (owner, 2026-11-12):** split into this plan plus four (banner), and the stamp joins M1.
  Considered: one plan.
- **HA-O14 — the first evidence slice. DECIDED (owner, 2026-11-12):** the one-cell Exp 49, boundary-only, exploratory
  (§A.4 step 3). Both-on stays the declared portfolio primary.

  Considered: wait for both-on.
- **HA-O15 — the advice layer's home. DECIDED (owner, 2026-11-12):** a shared split-out,
  [foreign_advice_layer.md](foreign_advice_layer.md), with LLM-source rules.

  Considered:
  - wait for social_referencing S1;
  - build it in this line.
- **HA-O16 — the safety floor. Required, not optional** (§A.1d). Recorded because it bounds HA-O2's grant of
  authority.

## Stated limits

- No experiment has tested a hybrid.
- The biological mappings are ANALOG, the arbitration is one-sided, and the refractory has no biological basis.
- Step 0's curves cover the substrate as it was trained, in those worlds.
- Pre-emption is checked at boundaries and between passes only. At local latency, the LLM mostly acts on the next
  action.
- S→L has moved choices but has never improved an outcome, so it is a hypothesis.
- A both-on gain is a composite-system result.
- Results under real LLM latency are reproducible only as distributions.

## Review record

The reports were given in the sessions and are not committed.

**Three-lens round, 2026-11-11** (Executor, Architecture, Wire integrity):
- **Blockers:**
  - the shadow log was not side-effect-free;
  - calibration against the LLM could not be observed;
  - the credit rule was undefined;
  - BL-4 was recorded as kept although the design breaks it.
- **Should-fix folds:** GL3.B4 marked as a prerequisite, the Stage 3 home, M88, a two-sided claim, the mode-site
  list, and signal definitions.

**Delta round, the same day:** a corrected inhibition path, Step 0's real corpus, the text gate as new code, and M88's
refusal predicate.

**Owner fold, 2026-11-12:** HA-O2, HA-O3, HA-O6 and HA-O7 were revised and HA-O12 was added, after a four-reader
audit (cancel paths, coupling, the experiment suite, a literature check).

**Final six-lens round, 2026-11-12.** No lens attacked the direction. The folds:
- **Adversarial (blocker):** the safety floor (§A.1d).
- **Adversarial (should-fix):**
  - `substrate_advised`;
  - LLM-source advice is never persisted or promoted;
  - bundle scrub and spotlighting;
  - override, withdrawal and suppression handled as effects;
  - a per-source text budget;
  - lane placement stamped;
  - the atonia rule.
- **Hardware (blockers):** a robot cancel goes `between_steps` only, ticket-bound; the retained target is rewritten
  on cancel; the protective class.
- **Hardware (should-fix):** deceleration, MP9 kept outside arbitration, DN duty cycle, wear, contract owners. These
  now live in sem_interruptibility.md, plus R10.
- **Scope (blocker):** the prerequisite is GL3.B7, not B4.
- **Scope (should-fix):**
  - the Stage 3 and advice-layer ownership claims;
  - Part B's swapped slice;
  - checks, now M89;
  - B5(a)'s prior-free variant;
  - the M88 row;
  - the Exp 49 floor;
  - costs.
- **Scope (decisions):** HA-O13 and HA-O14 settle the split and the first slice.
- **Concurrency (blockers):** the one-slot, one-clock loop (net-new #2); staleness under efference copy (§A.1b.5).
- **Concurrency (should-fix):** a non-blocking hold, one named clock, the episode definition, the refractory cap,
  two pre-emptions in one pass, thread ownership, and a deterministic-latency arm.
- **Statistics (blockers):** per-member comparator roles (the pass rule could not pass); powering and pairing Exp 49;
  the echo metric moved to the post-test.
- **Statistics (should-fix):**
  - the "LLM-scaffolded" claim name;
  - only the margin-reason A.2 curve, with a replay baseline;
  - calibration on disjoint pilot data;
  - a yoked sham;
  - one action set;
  - nuisance stamping;
  - the citation corrections.
- **Statistics (policy):** effect sizes with CIs at ≥ 2 SD, with no IUT or Holm. This is the project's pass-criteria
  rule.
- **Learning (blocker):** advice-flipped credit (`substrate_advised`).
- **Learning (should-fix):** explore-first bookkeeping at dispatch time, the refractory's value semantics, credit
  paths that ignore the chooser, the advice layer's persistence and clock, the A.2 dispatch floor, and the owner type
  for the SEM field.

**Delta round on the final fold, 2026-11-12** (all six lenses, scoped to the delta):
- **Hardware:** the stop path is outside the contract; the "retained target" rule is restated as a red-gate test;
  cancel is per ticket and covers chunked moves only; the robot has no danger caller yet.
- **Learning:** the `substrate_advised` trigger now covers gate-crossing; the SEM declaration is code-only;
  substrate-primary keeps its old bookkeeping behind a guard; the refractory has *D_max*; the advice-decay
  prerequisite and the operant confound are stated.
- **Statistics:**
  - roles for every member, with Exp 61's efficiency measured against llm-primary;
  - the survival efficiency claim waits for B5(a);
  - verdict forms on paired differences;
  - the BF2 floor outcome reported per member;
  - pilots cover both strata;
  - "powered" defined;
  - the latency arm is diagnostic.
