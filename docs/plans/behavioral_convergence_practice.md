# Behavioral Convergence Practice — Living Doc

**Status:** Living document. Not a gated phase. Not on the critical path to 1.0.
**Kin:** [tool_refinement_plan.md](tool_refinement_plan.md) (same pattern — an ongoing practice, not a one-shot plan).
**Related:** [substrate_recognition.md](substrate_recognition.md), [substrate_binding_persistence.md](archive/substrate_binding_persistence.md) (this doc consumes substrate mechanisms; it does not define them). Master reference: [archive/substrate_plan.md](archive/substrate_plan.md).

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

## Testing tiers

Behavioral convergence testing has three tiers, each building on the last:

| Tier | Training | Test | What it proves | Status |
|------|----------|------|----------------|--------|
| **Tier 1** | Scripted reactions | Substrate retrieval | Bio-systems learn and persist | Exp 1+2 PASS |
| **Tier 2** | Scripted episodes | LLM decisions | LLM acts on bio-system learning | Exp 3 PASS |
| **Tier 3** | LLM-driven sim | LLM decisions | Agent learns AND acts organically | Exp 4 PASS |

**Tier 1** is deterministic and fast (~0.5s). Proves the substrate stores and retrieves affective associations across sessions.

**Tier 2** uses scripted training (deterministic) but tests the LLM's actual decisions. The key question: does the LLM choose differently when it sees valence context? Uses masked entity names (e.g., "Vial A/B/C") to prevent language priors from confounding the measurement.

**Tier 3** is the ultimate 1.0 proof — no scripted training, the agent lives through a sim, builds bio-system state organically, and behaves differently in later sessions. Requires Tier 2 to pass first.

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

## Stimulus-agent experiments — isolation discipline

Some hypotheses here (especially H3 NPC preference consistency, H4 consolidation-helps-behavior, and scale-dependent versions of H1/H2) will eventually benefit from a generative stimulus agent rather than hand-authored fixtures. That's the Mother NPC pattern documented in [deferred/mother_npc_stimulus_plan.md](deferred/mother_npc_stimulus_plan.md).

**Hard rule:** any experiment entry in this doc that uses a second agent (Mother, another NPC, another Maxim instance) as a stimulus source must satisfy the isolation audit from the deferred Mother NPC plan *before results count*. The audit is a gate, not a checkbox — if it can't be demonstrated that Mother and Baby shared zero state beyond the percept stream, the experiment proves nothing about Baby's substrate, only about Mother's LLM leaking through.

Until the Mother NPC plan is revived, experiments here use hand-authored fixtures. When the Mother NPC plan revives (trigger: this doc has ≥2 successful experiments + 1 blocked-on-variety), new entries can use Mother but must document:
- Mode (deterministic replay vs. live seeded)
- LLM class for Mother and Baby (must be comparable — see same-class-LLM discipline in the deferred plan)
- Reward signal source (Mother-as-judge vs. rigid scenario rule vs. mixed)
- Isolation audit status for the specific runtime at experiment time (not a reference to an old audit — fresh verification)

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

### 2026-04-17 — Cross-session affective memory transfer (Exp 1)

**Hypothesis (H2 variant):** An agent that experienced pain, benefit, and disguised harm in Session 1 will show measurable affective differentiation in Session 2.

**Scenario:** Three SEM entities (rusty sword, healing potion, poison potion). Deterministic, no LLM. Tier 1 substrate-only.

**Metric:** Average retrieval valence per entity, NAc reward bias, EC threshold overrides. Experienced agent vs fresh control.

**N:** 1 (deterministic). Reproducible via `PYTHONPATH=src python scripts/behavioral_convergence_exp1.py`.

**Result:** 11/11 hypotheses confirmed.

| Entity | Experienced Agent | Fresh Control |
|---|---|---|
| Rusty sword | valence **-0.800** | 0.000 |
| Healing potion | valence **+0.195**, NAc bias **True**, EC widened | 0.000 |
| Poison potion | valence **-0.574** | 0.000 |

**Interpretation:** Affective memory transfers across sessions without fine-tuning. Shared "potion" concept carries mixed valence (healing+poison). Reward bias is asymmetric (positive only widens EC). Pain spikes create clean episode boundaries.

**Decision:** Tier 1 confirmed. Next: Tier 2 (LLM decisions based on valence). Blocked on [behavioral_convergence_wiring.md](archive/behavioral_convergence_wiring.md).

---

### 2026-04-17 — Energy-driven consumable learning (Exp 2)

**Hypothesis:** Agent learns to differentiate beneficial vs harmful consumables. Energy depletion triggers interoceptive reactions feeding the learning loop.

**Scenario:** Food ration, water flask, poison vial. Energy bridge fires hunger/fatigue. Tier 1 substrate-only.

**N:** 1 (deterministic).

**Result:** 13/13 hypotheses confirmed. Food +0.753, water +0.135, poison -0.495. Energy bridge: 1 hunger, 1 fatigue, 3 satiation. Key finding: environmental satiation creates background positive credit; discriminant is relative bias strength.

**Reproduction:** `PYTHONPATH=src python scripts/behavioral_convergence_exp2.py`

---

### 2026-04-17 — LLM acts on bio-system learning (Exp 3, Tier 2)

**Hypothesis (H2 extension, Tier 2):** An LLM given valence context from the bio-system will make different tool-selection decisions than a fresh LLM with no prior experience, when presented with masked/arbitrary item names that carry no semantic hints.

**Scenario:** Agent is poisoned (damage per turn). Three masked vials: Purple Hexagonal Glass (heals HP), Teal Cylindrical Ceramic (stops poison), Orange Triangular Crystal (more poison). Scripted training → persist → LLM test.

**Metric:** Vial selection rate per condition (experienced vs fresh), N=10 per condition.

**N:** 10 per condition. Model: qwen2.5-14b, temperature 0.3. Vial order shuffled per trial.

**Result:** 12/12 hypotheses confirmed (10/10 Tier 1 + 2/2 Tier 2).

| Vial | Experienced | Fresh |
|---|---|---|
| **Teal (stops poison)** | **10/10 (100%)** | **0/10 (0%)** |
| Purple (heals HP) | 0/10 | 7/10 (70%) |
| Orange (more poison) | 0/10 | 3/10 (30%) |

**Interpretation:** Bio-system learning changes LLM behavior. The experienced agent picked the optimal vial 100% of the time. The fresh agent had no preference and picked the harmful orange vial 30% of the time. Valence strength differentiation is critical — flat "GOOD/BAD" labels showed no effect. This is the first Tier 2 proof for the 1.0 claim: cross-session learning without fine-tuning affects agent decisions.

**What this does NOT prove (Tier 3):** The agent doesn't take the action during training — reactions are injected. Tier 3 tests organic learning.

**Reproduction:** `PYTHONPATH=src python scripts/behavioral_convergence_exp3_tier2.py --model qwen2.5-14b`
**Full protocol:** [experiments/protocols/behavioral_convergence_exp3_reproduction.md](../experiments/protocols/behavioral_convergence_exp3_reproduction.md)

---

### 2026-04-17 — Organic LLM learning (Exp 4, Tier 3)

**Hypothesis (H2 extension, Tier 3):** An agent running in a real sim with SEM entities will organically learn from its own actions — choosing a vial, experiencing the outcome via CerebellumModulator -> ReactionBus -> valence annotation, and making different choices in subsequent sessions. No scripted reactions. All learning comes from the agent's actual tool executions through the CerebellumModulator -> _emit_failure/success_reaction pathway.

**Scenario:** Agent is poisoned (damage per turn). Three masked vials: Purple Hexagonal Glass (heals HP), Teal Cylindrical Ceramic (stops poison), Orange Triangular Crystal (more poison). Multi-session organic training. Session 1: exploration. Session 2: early learning. Session 3: convergence. Fresh control comparison.

**Tier:** 3 (organic LLM training + LLM test — the ultimate 1.0 proof)

**Metric:** Teal vial selection rate across sessions + fresh control comparison.

**N:** 3 sessions (organic training) + 1 fresh control. Model: qwen2.5-14b, temperature 0.3.

**Result:** 5/5 hypotheses confirmed.

| Session | Teal Rate | Interpretation |
|---|---|---|
| **Session 1** (exploration) | **0%** | No prior knowledge, agent explores randomly |
| **Session 2** (early learning) | **25%** | Agent begins shifting toward learned associations |
| **Session 3** (convergence) | **100%** | Full convergence — agent picks antidote every time |
| **Fresh control** | **DIED** | No learning signal — agent never picks antidote |

**Interpretation:** The agent learns organically from its own actions. No scripted training, no injected reactions — the full CerebellumModulator -> ReactionBus -> hippocampus/NAc -> valence annotation -> PromptAssembler pipeline works end-to-end. The fresh control validates that learning is necessary — without bio-system state, the LLM has no reason to prefer any vial. The experienced agent escapes on turn 1 in Session 3; the fresh agent dies. This is the Tier 3 proof for the 1.0 claim: cross-session learning without fine-tuning, demonstrated with organic LLM-driven training.

**Decision:** All three testing tiers now PASS. 41/41 hypotheses confirmed across 4 experiments. The 1.0 research claim — cross-session learning without fine-tuning — is demonstrated at every tier. Version bump to 0.3.0.

**Reproduction:** `PYTHONPATH=src python scripts/behavioral_convergence_exp4_tier3.py --model qwen2.5-14b`
**Full protocol:** [experiments/protocols/behavioral_convergence_exp4_reproduction.md](../experiments/protocols/behavioral_convergence_exp4_reproduction.md)
