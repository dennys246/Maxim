# Benchmarking — 1.0 Gate Scoping

> **ARCHIVED (2026-07-15 plans audit):** ✅ CONCLUDED. The 1.0 benchmark gate fired and was dispositioned 2026-06-13 (Exp 37 + Exp 38 executed across 4 models; performance claim pulled from 1.0 framing, mechanism/persistence claims stand). Post-1.0 continuation is owned by [behavioral_graduation_candidates.md](../behavioral_graduation_candidates.md).


**Status:** Scoping doc. Defines what "passes the 1.0 benchmark" means. Implementation is out of scope here — that lives in follow-up plans once this scope is accepted.
**Created:** 2026-05-29.
**Sibling gate:** [behavioral_graduation_candidates.md](../behavioral_graduation_candidates.md) — mechanism validation ("do specific bio-mechanisms carry their claimed load").
**This gate:** agent performance ("does the system solve the thing that the bio-mechanisms are claimed to enable").
**Triggered by:** the [1.0 sequenced plan](v1_refinement.md#1-6-sequenced-1-0-plan) named benchmarking + behavioral graduation as the two parallel 1.0 gates. Behavioral graduation has its own living doc; benchmarking did not. This is the missing artifact.

## Front-gate scope pressure (CLAUDE.md Principle 3)

**Question:** does the 1.0 benchmarking gate need to be its own mechanism / artifact, or can it ride on existing infrastructure?

**Existing infrastructure surveyed:**

| Candidate | Why insufficient (or sufficient) |
|---|---|
| [behavioral_convergence_practice.md](../deferred/behavioral_convergence_practice.md) | **Living doc, not a gate.** Tracks ongoing behavioral hypotheses + experiments. Per its own framing: "Not a 1.0 gate. Behavioral change is a demonstration, not a pass/fail test." The 1.0 benchmarking gate needs a pass/fail commitment that this doc explicitly avoids. |
| [behavioral_graduation_candidates.md](../behavioral_graduation_candidates.md) | **The sibling gate** — mechanism validation. Both must pass for 1.0. Passing one without the other leaves a half-claim ("the mechanisms validate but the agent doesn't solve anything" or vice versa). |
| [minecraft_benchmark.md](../minecraft_benchmark.md) | **Explicitly 1.1 splash, not 1.0.** Per its own framing: "Cradle (B4) already provides the cross-session evidence 1.0 needs. Minecraft strengthens the story without gating it." The 1.0 gate has to land before Minecraft work matures. |
| [Experiment 10 — cross-session enrichment](../../experiments/10_cross_session_enrichment.md) | **Partial evidence.** Validates that enrichment surfaces prior-session memories in the prompt (3 memories/turn on resume). Does NOT measure whether that surfaces produces measurably different downstream behavior. The 1.0 gate needs the behavioral half. |
| [Experiment 11 — Cradle sensorimotor PoC](../../experiments/11_cradle_sensorimotor_poc.md) | **Infrastructure shipped; behavioral measurement pending.** Cradle is the substrate for the 1.0 cross-session claim, but the demonstration-grade benchmark run on top of it has not been formalized as a 1.0 gate. |
| Roy harness (Exp 16–36) | **Mechanism-level evidence, not agent-performance evidence.** Roy validates specific bio-mechanisms (EC pattern completion, NAc reward bias propagation, cluster annotation effects). It feeds behavioral graduation, not this gate. |

**Verdict:** yes-needs-own (small). The 1.0 benchmarking gate is the **commitment + acceptance criteria + pass/fail framing** that the surrounding artifacts deliberately avoid. The artifact is mostly a scoping decision; the *implementation* rides on existing infrastructure (Cradle as the substrate, sim_logger + report.json as the measurement plumbing, BioEnrichmentPipeline as the dependent variable).

**Specific reason this scoping doc exists:** without a documented gate, 1.0 ships with implicit benchmarking criteria that nobody can reproduce or audit. The discipline of writing it down forces the "what does pass mean?" question to be answered explicitly rather than assumed.

## What this gate is for

The 1.0 thesis (per [CLAUDE.md framing strategy](../../../CLAUDE.md) + [project_research_claim_non_negotiables](../../.claude/projects/-Users-dennyschaedig-Scripts-Maxim/memory/project_research_claim_non_negotiables.md)) is:

> **Cross-session learning without fine-tuning** — a bio-inspired LLM harness whose persistent substrate produces measurably different agent behavior across resumed sessions, with no weight updates, no prompt-engineering tricks, no fine-tuning.

That claim has two halves:

1. **Mechanism:** the bio-systems learn and persist (Hippocampus stores episodes, NAc accumulates reward biases, EC pattern-completes concepts, etc.). → [behavioral_graduation_candidates.md](../behavioral_graduation_candidates.md).
2. **Performance:** the agent **does something different** as a result. Tool choices shift, error rates drop, exploration patterns adapt, repeat-failure rates fall. → **this doc.**

Both halves must be defensible at 1.0 ship. A passing mechanism gate with unbacked behavioral claims still leaves the bio-inspired-LLM-harness framing exposed. A passing performance gate with broken mechanism validation isn't 1.0-shippable either ("works by accident").

## What this gate is NOT

- **Not a research paper.** The acceptance criteria are "defensible enough to ship 1.0 as a bio-inspired LLM harness," not "publishable at NeurIPS." Statistical rigor matches the artifact.
- **Not the Minecraft demo.** That's the 1.1 splash — Cradle is the 1.0 substrate. Minecraft strengthens the story; it doesn't enable 1.0.
- **Not a continuous CI gate.** Benchmarking runs are too expensive (LLM calls, multi-session setup) for every PR. The gate fires at: minor-version heartbeats pre-1.0, the 1.0 readiness review, and category-scoped triggers (substrate-pipeline changes, encoder swaps, bio-system refactors).
- **Not a replacement for behavioral graduation.** Graduation validates mechanisms; this validates agent performance. Both gate 1.0; they cover orthogonal failure modes.
- **Not a single-pass acceptance.** The gate is "ran the benchmark, recorded the result, met the threshold" — not "ran once, saw a good number, moved on." Each run produces a logged result with seed + model + config so the threshold is reproducible.

## Why a separate gate from behavioral graduation

Mechanism validation answers "does X bio-system do what we claim it does in isolation?" — measured at the substrate layer (cosine thresholds, cluster counts, reward-bias propagation rates).

Benchmarking answers "does the agent's *end-to-end behavior* reflect those mechanisms?" — measured at the agent's tool-call sequence, action diversity, repeat-failure rate.

The two can diverge:
- **Mechanism pass + behavior fail:** the bio-systems work but the LLM doesn't condition on them (prompt-construction bug, enrichment-routing gap, dose-too-low). [Exp 10 pre-fix](../../experiments/10_cross_session_enrichment.md) is the canonical example — enrichment retrieved memories but the prompt section was empty, so behavior was identical to fresh-start.
- **Mechanism fail + behavior pass:** the agent appears to "learn" but only because the LLM is doing the work (verbose system prompts, in-context recall, prompt-engineering). The bio-thesis is unsupported. This is the framing-strategy failure mode.

A single gate would let either failure mode hide. Two gates force both halves to land.

## Acceptance criteria for 1.0 (proposed)

The benchmark surface for 1.0 is the **cross-session learning differential** — does an agent resumed from a prior session behave measurably differently from a fresh-start agent on the same scenario, in ways consistent with the prior session's experience?

### Primary criterion (must pass)

**The fresh-vs-resume behavioral delta is non-zero, in the expected direction, and isolable to the bio-substrate.**

Concretely:

- **Scenario class:** a Cradle-derived embodied scenario where a specific failure or success pattern in session N produces predictable behavior change in session N+1.
- **Conditions compared:**
  - Arm A — Fresh agent on the scenario.
  - Arm B — Agent resumed from a prior session where the failure / success occurred.
  - Arm C (isolation) — Agent resumed from a prior session in a *different* scenario (negative-transfer control).
- **Primary metric (must include):** **repeat-failure-action rate** — fraction of turns where Arm B selects an action class that failed in the prior session. The gate requires Arm B's rate to be lower than Arm A's; this is the most direct operationalization of "the agent learned not to do the thing that hurt last time." This metric is non-negotiable for 1.0 because it's the most direct test of the load-bearing claim; the other metrics below corroborate but do not substitute.
- **Corroborating metrics (at least one in addition):**
  - Tool-class diversity shift (Arm B explores fewer dead-end tools after seeing them fail).
  - Time-to-goal on success scenarios (Arm B reaches goal in fewer turns).
  - Affordance-selection preference shift in the direction NAc reward bias would predict.
- **N:** ≥ 5 paired (Arm A, Arm B, Arm C) trials per metric.
- **Variance-survival rule:** Arm B's primary-metric mean must lie outside Arm A's 95th-percentile band (computed across the same N trials). For corroborating metrics, the looser rule "Arm B mean differs from Arm A mean by ≥ 1 SD of Arm A baseline" applies. The rule is concrete — eyeballing is not acceptable.
- **Isolation requirement (Arm C):** Arm C confirms the delta is scenario-specific (not "agent gets generally cautious after any prior session"). Arm C's primary-metric value must fall within Arm A's band — if Arm C also shows a delta, the result is the "general caution" confound and the gate fails. **Without Arm C the result is observed-but-not-isolated.**
- **(Future refinement, not 1.0-gating):** Arm D — resumed-from-same-scenario-but-success — would isolate valence direction (does the agent shift toward success-associated actions, not just away from failure-associated ones). Add when the implementation plan is mature; deferring keeps the 1.0 gate scope tight.

### Secondary criterion (should pass)

**The cross-session delta is mechanism-attributable.** When Wire-A annotation, Wire 1 variance annotation, EC pattern completion, or NAc reward bias are disabled, the delta shrinks measurably. This is the **bridge to behavioral_graduation_candidates** — passing the secondary criterion is evidence the gates are coupled, not orthogonal.

If the secondary criterion fails (delta survives ablation of all bio-mechanisms), the benchmark passes but the framing is exposed: the agent's behavior change isn't from the bio-substrate, it's from something else (LLM in-context learning, prompt artifacts). The 1.0 release notes would have to retract the bio-attribution.

### Tertiary criterion (informational)

**The delta replicates on at least two model backends.** Cross-session learning that's only visible on Claude Sonnet but not on local mistral-7b is suspicious — either the local model is too weak to surface it (acceptable but worth flagging) or the effect is LLM-prompt-driven (problematic).

## Out of scope for 1.0

- **Voyager / GITM / SPRING comparison runs.** That's the Minecraft 1.1 splash. The 1.0 gate measures Maxim against itself (fresh vs resume), not against external baselines. External comparison is a marketing artifact, not a thesis-validation artifact.
- **Statistical significance testing.** N ≥ 5 per arm + the variance-survival check is sufficient for a 1.0 ship decision. Significance testing is research-paper-grade discipline; this gate is engineering-grade.
- **Multi-task generalization.** The 1.0 claim is "cross-session learning works on a class of scenarios," not "the agent transfers learning across scenario classes." Transfer generalization is a 1.1+ research question.
- **Long-horizon retention.** "Does the agent remember after 30 days?" is a different question. 1.0 measures cross-session-within-the-session-pair. Long-horizon retention rides on the consolidation pipeline + persistence-format stability (CC1) but isn't gated here.
- **Adversarial scenarios.** Persona-driven adversarial sims surface different failure modes (covered by [persona_convergence_crucible.md](../deferred/persona_convergence_crucible.md)). Not part of the 1.0 benchmark.

## Inventory of existing evidence

| Evidence | What it shows | Gap for 1.0 |
|---|---|---|
| [Exp 10 — cross-session enrichment](../../experiments/10_cross_session_enrichment.md) | BioEnrichmentPipeline surfaces prior-session memories in the LLM prompt (3 memories/turn on resume post-fix). | Does NOT measure behavioral delta. Stops at "the memories are in the prompt." |
| [Exp 11 — Cradle sensorimotor PoC](../../experiments/11_cradle_sensorimotor_poc.md) | Cradle infrastructure runs end-to-end; narrator generates scenes; bio-substrate accumulates state across cradle phases. | Demonstration-grade, not benchmark-grade. No paired fresh-vs-resume measurement. |
| [Tier 1 behavioral graduation entries](../behavioral_graduation_candidates.md) | EC pattern completion EARNED; SEM pain cascade EARNED; cross-session learning PARTIAL; affordance transfer PARTIAL; substrate-primary PARTIAL. | Mechanism evidence — feeds the secondary criterion (ablation attribution) of this gate. |
| Roy harness | Mechanism-level isolation of specific bio-claims (cluster bias, variance annotation, EC drift, etc.). | Substrate-layer measurement. Doesn't directly produce agent-performance metrics. |
| sim_reports under `~/.maxim/sessions/` | Per-run logs with action sequences, enrichment traces, reward biases. | The measurement plumbing exists. The protocol that turns these into a benchmark result is what this gate prescribes. |

**Headline gap for 1.0:** the **paired fresh-vs-resume Cradle behavioral measurement** does not exist as a logged, reproducible experiment. Exp 10 has the enrichment half; Exp 11 has the substrate half. A "Exp 37+ — Cradle cross-session behavioral delta" would be the artifact this gate ultimately requires to pass.

## Relation to other plans

- **[v1_refinement.md](v1_refinement.md)** — names benchmarking as a 1.0 gate but does not specify acceptance criteria. This doc fills that gap.
- **[behavioral_graduation_candidates.md](../behavioral_graduation_candidates.md)** — the sibling 1.0 gate. The secondary criterion of this gate (mechanism-attributable delta) is what couples the two — passing this gate without the graduation gate means the bio-attribution is unsupported.
- **[behavioral_convergence_practice.md](../deferred/behavioral_convergence_practice.md)** — the living doc for ongoing behavioral hypotheses. Once the 1.0 benchmark passes, the doc keeps tracking newer hypotheses post-1.0 — same shape as the post-1.0 lifecycle in graduation candidates.
- **[memory_consolidation_practice.md](../deferred/memory_consolidation_practice.md)** — the long-horizon retention practice doc. Out of scope for 1.0 benchmarking but relevant for 1.1+ long-horizon questions.
- **[minecraft_benchmark.md](../minecraft_benchmark.md)** — the 1.1 showpiece. Builds on whatever protocol this gate establishes; Minecraft adapts it to a third-party-recognizable environment.

## Implementation plan (out of scope here)

This scoping doc deliberately stops at "what passes the gate." The follow-up work is:

1. **Pick the Cradle scenario(s).** Currently the embodiment fixtures are `bodies/infant_humanoid` + the cradle items (`items/cradle_*`). The benchmark scenario must produce a measurable failure / success pattern that's predictable across the fresh/resume pair.
2. **Lock the primary metric.** Repeat-failure-action rate is committed by the primary criterion above. Pick the specific failure class this scenario surfaces.
3. **Pick at least one corroborating metric.** Affordance-selection preference shift is the most mechanism-bridging; tool-class diversity is the easiest-to-measure.
4. **Write the experiment harness.** Likely lives in `tests/behavioral/` or a `scripts/benchmark_*.py` — extends the sim runner with paired-trial bookkeeping (Arms A/B/C).
5. **Pre-register the run.** Per [feedback_invariant_two_tier_tracking](../../.claude/projects/-Users-dennyschaedig-Scripts-Maxim/memory/feedback_invariant_two_tier_tracking.md) discipline, pre-register the scenario + metric + threshold *before* running so a passing result isn't a post-hoc fit. A pre-registration template lives in [docs/experiments/protocols/](../experiments/protocols/) or — if no parallel exists — gets drafted alongside this run.
6. **Run it.** Log the result as a new experiment doc (Exp 37+) with full reproducibility info (seed, model, config, prompt version, encoder version).
7. **Cross-confirm with graduation candidates.** The secondary criterion's ablation arms feed back into the Tier 1 entries' Earned status.

A reasonable implementation plan would be ~1-2 weeks wall-time, gated on the Cradle scenario selection (the slow step is iterating on a scenario that surfaces a reliable behavioral signal).

## What "shipping 1.0 without this gate" would look like

For documentation purposes, the failure mode this gate prevents:

- 1.0 ships with the bio-inspired-LLM-harness framing.
- A reviewer asks "where's the evidence the agent gets better across sessions?"
- We point at Exp 10 ("enrichment surfaces memories") and Cradle ("infrastructure runs").
- The reviewer notes neither directly measures behavioral change.
- We retract the cross-session-learning framing in a 1.0.1 patch or face credibility loss.

Shipping 1.0 without this gate means shipping a thesis we haven't measured. The gate exists to force the measurement *before* the framing goes out the door, not after.

## Execution & 1.0 disposition (2026-06-13)

**The gate has FIRED.** The "headline gap" this doc identified — *"the paired fresh-vs-resume Cradle behavioral measurement does not exist as a logged, reproducible experiment"* — is now closed. The full implementation plan (items 1–7 above) was executed:

- **Harness:** [`scripts/benchmark_cross_session.py`](../../scripts/benchmark_cross_session.py) (paired Arms A/B/C + ablations); **analyzer:** [`scripts/analyze_exp37.py`](../../scripts/analyze_exp37.py). Pre-registered per discipline.
- **Scenario / primary metric:** cradle `bodies/infant_humanoid` + `items/cradle_*`; repeat-failure-action rate (operationalised as the warm-self / failure-class engagement fraction), Arm-C isolation, ablation secondary — exactly the §"Acceptance criteria" committed above.
- **Two pre-registered experiments executed across the model set:**
  - **Exp 37 (prior-aligned)** — [37_cross_session_graduation.md](../../experiments/37_cross_session_graduation.md) + [cross-model](../../experiments/37_cross_model_results.md): a **Goldilocks zone** — the cross-session signal is detectable only when the base LLM's priors leave headroom (Qwen32B +1.43 SD PASS, R1-distill +2.11 SD PASS; Qwen14B null, Mistral24B ceiling). Reasoning models are better substrate consumers (R1 surfaces Wire-A as the carrying mechanism, +1.13 SD ablation).
  - **Exp 38 (counter-prior, the disambiguator)** — [38_counter_prior_substrate.md](../../experiments/38_counter_prior_substrate.md): a world where the prior is *wrong* (a hearth whose `warm_self` hurts). **Dominance across all 4 frontier models** (Sonnet / GPT-4o / DeepSeek-V3 / R1) — carried substrate does **not** override a wrong prior. R1 is the sharpest: substrate causally load-bearing (ablations drop hearth-warming *below* a fresh agent) but it **amplifies the prior, not the corrective experience**.

**Primary-criterion outcome:** the measurement exists, is reproducible, and shows the cross-session *behavioral improvement does not materialise under strong LLM priors*. **Secondary-criterion (mechanism attribution):** moot in the three chat models (no delta to attribute); in R1 the substrate IS ablation-attributable, but in the counter-prior its causal effect is *maladaptive* (reinforces the wrong prior).

**1.0 disposition — gate satisfied by execution, not by a green pass.** Per §"What this gate is NOT" ("ran the benchmark, recorded the result, met the threshold" — a documented outcome, not a single good number), the gate is satisfied: the thesis has been *measured before ship*, which is precisely the retraction-avoidance §"shipping 1.0 without this gate" describes. The honest 1.0 framing this forces: **bio-mechanisms are validated at the mechanism level** (the EARNED Tier 1 entries) **and the substrate is causally active in action selection (R1 ablation), but cross-session agent-performance *improvement* does not occur under LLM priors** — it is an open 1.1+ problem (substrate-primary action selection + the substrate-aware-reasoning direction R1 opened). The cross-session-learning *performance* claim is pulled from 1.0; the mechanism-reality and memory-persistence claims are not. This feeds directly into the [behavioral_graduation_candidates.md](../behavioral_graduation_candidates.md) Tier 1 #2 disposition (cross-confirm per item 7).

## Status

- **Scoping accepted:** 2026-05-30 (user explicit-accepted as-is during the [cross-session graduation kickoff](../../.claude/projects/-Users-dennyschaedig-Scripts-Maxim/memory/kickoff_1_0_graduation_cross_session.md); acceptance criteria above become the pre-registration template for downstream experiments).
- **Implementation:** pre-registered at [37_cross_session_graduation.md](../../experiments/37_cross_session_graduation.md) + [38_counter_prior_substrate.md](../../experiments/38_counter_prior_substrate.md). Owner: cross-session graduation worktree (`feat/1-0-graduation-cross-session`). Target: Phase 2 of [v1_refinement.md §1.6](v1_refinement.md).
- **EXECUTED 2026-06-11 / 2026-06-13** — Exp 37 (prior-aligned, 4 models) + Exp 38 (counter-prior, 4 models). Gate fired; result recorded in §"Execution & 1.0 disposition" above. Outcome: cross-session behavioral *improvement* does not materialise under LLM priors → the performance claim is pulled from 1.0 framing; mechanism-reality + memory-persistence claims stand.

## Cross-references

- [v1_refinement.md](v1_refinement.md) §6.1 — 1.0 plan rollup.
- [behavioral_graduation_candidates.md](../behavioral_graduation_candidates.md) — sibling 1.0 gate.
- [behavioral_convergence_practice.md](../deferred/behavioral_convergence_practice.md) — living behavioral practice doc.
- [minecraft_benchmark.md](../minecraft_benchmark.md) — 1.1 splash plan.
- [docs/experiments/10_cross_session_enrichment.md](../../experiments/10_cross_session_enrichment.md) — partial evidence.
- [docs/experiments/11_cradle_sensorimotor_poc.md](../../experiments/11_cradle_sensorimotor_poc.md) — substrate evidence.
- [CLAUDE.md framing strategy](../../../CLAUDE.md) — "bio-inspired LLM harness for AI engineers."
- [project_research_claim_non_negotiables](../../.claude/projects/-Users-dennyschaedig-Scripts-Maxim/memory/project_research_claim_non_negotiables.md) — the load-bearing 1.0 claim.
