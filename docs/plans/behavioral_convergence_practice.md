# Behavioral Convergence Practice — Living Doc

**Status:** Living document. Not a gated phase. Not on the critical path to 1.0.
**Kin:** [tool_refinement_plan.md](tool_refinement_plan.md) (same pattern — an ongoing practice, not a one-shot plan).
**Related:** [substrate_plan.md](substrate_plan.md) (this doc consumes substrate mechanisms; it does not define them).

## What this document is for

The substrate plan proves that **memory works** — nodes stabilize, episodes bind, reward modulates recognition, associations decay, consolidation strengthens rewarded episodes. Those are mechanism claims. They are evaluated by retrieval F1, pattern completion rates, and round-trip tests.

The separate question is whether **the agent gets better** — whether living through more sessions in a recurring scenario produces measurably improved decisions, fewer repeated errors, faster convergence to goals, or more consistent preferences. That is a **behavioral claim**. It is not mechanistic, it is not one-shot, and it cannot be cleanly gated as a version milestone.

This document is where that claim lives as an ongoing practice. It captures:

- **Behavioral hypotheses** — specific things the agent should get better at over repeated sessions.
- **Scenarios** — deterministic(-ish) test environments where a hypothesis can be observed.
- **Metrics** — what "better" means for a given scenario, operationalized.
- **Experiments** — runs with dated results, logged here for review.
- **Findings** — what we learned from each experiment, even (especially) the ones that failed.

## What this document is NOT

- **Not a 1.0 gate.** The 1.0 claim stays on substrate mechanisms. Behavioral change is a demonstration, not a pass/fail test. Gating 1.0 on noisy system-level behavior is how you ship late.
- **Not an evaluation harness.** Harness code lives in `tests/behavioral/` if and when it exists. This doc is the *program of questions*, not the framework.
- **Not a research paper.** Entries should be short and specific — a hypothesis, a scenario, a metric, a result. Not a literature review.
- **Not a replacement for substrate_plan.** Substrate proves the mechanism; this doc tracks what the mechanism enables. They are complementary.

## Why this is a living doc instead of a substrate phase

Behavioral change is observed, not mechanized. You do not need to write new substrate code to evaluate it — you run the substrate you already have in a scenario, measure what happened, and log the result. That pattern is a practice, not a milestone. Forcing it into a P-phase would either rush decisions (shipping before the scenarios are stable) or miss deadlines (waiting for clean results on something inherently noisy).

The microGPT tension also bears on this. Behavior change is hard to evaluate cleanly with external LLMs because LLM non-determinism contaminates the measurement — you can't always tell whether the agent improved or the LLM drifted. Two options:

1. **Measure at a level below LLM output** — tool-call sequences, rewarded-action rate, known-bad-path avoidance rate. These can survive LLM noise.
2. **Accept the noise** — run each scenario at high N (≥20 trials per condition) and look for effects that survive variance.

This doc defaults to option 1 where possible and option 2 where not, and either way avoids the trap of taking any single run as evidence.

## How to add an entry

Each experiment gets a short section under `## Experiments`, structured as:

```markdown
### YYYY-MM-DD — <short name>

**Hypothesis:** What we expected to observe, stated falsifiably.
**Scenario:** The deterministic setup — fixture file, seed, scenario script.
**Metric:** What we measured. How it was computed. Why it is not trivially gamed.
**Conditions compared:** Typically "fresh agent" vs "agent reloaded from session N." Sometimes "with vs without a specific substrate component."
**N:** Number of trials per condition.
**Result:** Raw numbers. Mean + std. No p-values.
**Interpretation:** What we think it means. Keep this separate from the result so a future reader can re-interpret if needed.
**Decision:** Did this change anything about how we think about the architecture, the substrate plan, or future experiments?
```

Entries are append-only. If a finding is later invalidated, add a new entry that supersedes it and link both directions — do not edit the old one.

## Initial seed hypotheses (to be tested as scenarios and substrate mature)

These are candidates, not commitments. They become experiments when a scenario exists and the substrate has the mechanisms to be evaluated.

### H1 — Repeated-scenario error reduction

**Claim:** An agent that has lived through a failing scenario N times makes fewer of the same tool-call errors on the N+1th attempt, because past-failure episodes are retrievable via P3a/P3b and feed into B4 replanning.

**Why it's interesting:** this is the clean intersection of Track A (episode retrieval) and Track B (replanning with memory). It's also the minimum viable "learning" claim.

**What's needed before it can be tested:** P3a, P3b, B4 shipped. A deterministic failing scenario. A metric that counts tool-call errors independently of LLM variance.

### H2 — Known-bad-path avoidance

**Claim:** An agent that was punished for taking action A in context C avoids A in the same C on later sessions, because the punishment is encoded as reward bias on the A-associated ATL node (P2) and persists across sessions (P3.5).

**Why it's interesting:** this is the simplest possible "cross-session learning" claim. It's also the claim P7 was originally going to gate, now living here as a demonstration.

**What's needed:** P2, P3.5 shipped. A scenario with a clear "bad" action. Per-agent bias isolation (F0.5) so multi-agent tests don't collide.

### H3 — NPC preference consistency across sessions

**Claim:** An NPC with a reward history on certain dialogue paths will preferentially select those paths in later sessions, producing visibly consistent character preferences to a human observer.

**Why it's interesting:** this is where Track A (memory) and Track B (acting coach) combine into something a human can evaluate without specialized metrics. Also the closest thing in the plan to "the NPC remembered me."

**What's needed:** P2, P3.5, B3 shipped. A multi-session DM campaign scenario. Blind A/B evaluation protocol (humans cannot be told which run has memory and which doesn't).

### H4 — Consolidation demonstrably helps

**Claim:** After P8 ships, running a sleep-phase between sessions produces measurable behavior change on a held-out scenario — the agent performs better in session N+1 with a sleep phase between N and N+1 than without.

**Why it's interesting:** this is the behavioral counterpart to P8's mechanism test. P8 shows retrieval F1 improves; H4 shows it matters for behavior.

**What's needed:** P8 shipped. Any H1/H2/H3 scenario that can be run with and without a sleep phase.

### H5 — Cross-modal transfer under partial input

**Claim:** After seeing text + vision of a mug together during session 1, the agent can correctly act on a vision-only mug cue in session 2 even when the task was originally described in text — because cross-modal binding (P4) retrieves the linguistic context.

**Why it's interesting:** this is the behavioral claim P4's retrieval F1 is a proxy for. It is also the closest thing in the plan to "the agent knows what it is looking at."

**What's needed:** P4 shipped with production vision. A scenario where a task is specified textually and executed in a visual context.

## How this feeds back into substrate_plan

If an experiment here produces a strong finding — positive or negative — that contradicts or extends a substrate commitment, the result should be promoted into the substrate plan as either:

- **Evidence for a new phase** (e.g., "H5 shows cross-modal transfer is brittle in these specific conditions — add a phase that targets those conditions").
- **Evidence to revisit a commitment** (e.g., "H2 shows reward bias does not persist usefully across sessions even with P3.5 — revisit commitment #4").
- **Evidence to seed a new practice-doc entry** (e.g., "H4 revealed that sleep-phase timing matters — seed a new consolidation-practice entry").

The promotion is manual and deliberate. Not every finding here needs to change the substrate plan. Most will just accumulate as evidence.

## Review cadence + forcing function

Each version bump (0.3 → 0.4, 0.4 → 0.5, 0.5 → 1.0) triggers a review of this document:

1. What experiments ran since the last version?
2. What did we learn?
3. Does any finding change our plan for the next version?
4. Are there new hypotheses worth seeding for the next version's substrate scope?

The review is a note in the version's release notes, not a separate meeting.

**Soft discipline (from substrate_plan's "Living-doc discipline" section):**

Try to log at least one new experiment entry per version bump, so the empirical base grows alongside the code. This is a soft discipline, not a hard gate — for a single-developer project, hard gates against yourself are LARPing. If a version ships without a new entry, that's a signal to ask why and be honest about the trade-off.

**The 1.0 release should have at least one entry in this doc.** Enforce that one on yourself at tag time.

## Experiments

*(No experiments yet. The first entry will land when P1 is green and the first hypothesis scenario is runnable.)*
