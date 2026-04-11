# Memory Consolidation Practice — Living Doc

**Status:** Living document. Kicks in once P8 ships in 0.5. Not a gated phase; refines the mechanism P8 puts in place.
**Kin:** [tool_refinement_plan.md](tool_refinement_plan.md), [behavioral_convergence_practice.md](behavioral_convergence_practice.md).
**Related:** [substrate_plan.md](substrate_plan.md) P8 (the minimum-viable sleep replay mechanism this doc extends).

## What this document is for

P8 in the substrate plan ships a **deliberately minimal** consolidation mechanism: during a sleep phase, replay the top-N most-rewarded episodes, strengthen Hebbian links, measure whether retrieval F1 on replayed probes improves. That is enough to prove offline learning *can* happen. It is not enough to prove it *should* happen the way the P8 minimum chose.

Consolidation is a broad research area with many open questions:

- Which episodes should be replayed? (Top-N by reward, Hebbian centrality, recency, novelty, prediction error?)
- How often should sleep phases fire?
- How should replay interact with forgetting (P6)? Is sleep a time for both strengthening and pruning?
- Does replay during wake (as in rodent hippocampal replay during rest between tasks) add value over sleep-only replay?
- What is the right update strength during replay — same as online, scaled down, or tapered by reward magnitude?
- When should replay-strengthened links promote to ATL as semantic concepts (hippocampus → cortex transfer)?
- How do we avoid over-fitting: the same episode replayed too often becomes dominant in retrieval regardless of real utility.
- Can consolidation interfere with itself — strengthening one episode weakens others?

These are research questions, not engineering decisions. They need to be run, measured, and iterated. This doc is where that iteration lives.

## What this document is NOT

- **Not a gate.** P8's minimum mechanism is the gate. This doc is the research program on top of it.
- **Not a substrate-plan phase.** Phases build mechanism. This doc refines the mechanism P8 built.
- **Not a replacement for P8's unit tests.** P8's determinism and round-trip tests stay in the substrate plan and are part of its pass criteria. This doc is for experiments beyond the minimum.
- **Not a place for design documents.** If an experiment would require a design document to run, the design goes in a substrate-plan fallback proposal, not here.

## Why this is a living doc + a shipped phase, not just a living doc

Behavioral convergence (the sibling practice doc) is purely a living doc because behavior is *observed* — you don't need new code to evaluate it. Consolidation is different: it is a *mechanism*, and mechanisms only exist if something ships. A "consolidation practice" doc with no consolidation mechanism would be a design document pretending to be a plan, and this repo's `deferred/` directory shows how that ends.

So the asymmetry is:

- **P8 ships the minimum mechanism** as a substrate phase in 0.5 — one replay strategy, one scheduling rule, one measurable improvement. The gate is "the mechanism works at all."
- **This doc hosts everything more ambitious** — alternative strategies, production tuning, interference analysis, promotion rules — as ongoing experiments with logged results.

The living doc can exist only because the mechanism already does.

## How to add an entry

Each experiment gets a section under `## Experiments`, structured as:

```markdown
### YYYY-MM-DD — <short name>

**Question:** The specific consolidation question being probed, stated as a hypothesis.
**Variant:** How this experiment's replay differs from the P8 minimum (e.g., "top-N by Hebbian centrality instead of top-N by reward").
**Baseline for comparison:** Typically the P8 minimum itself — does this variant beat the minimum on the same scenario?
**Metric:** Retrieval F1 delta, interference rate, over-fitting rate, or a behavioral metric if the experiment is run against a scenario from [behavioral_convergence_practice.md](behavioral_convergence_practice.md).
**Scenario:** Fixture file + seed + scenario script.
**N:** Trials per condition (≥10 for substrate metrics, ≥20 for behavioral).
**Result:** Raw numbers, mean + std. No p-values.
**Interpretation:** What we think it means.
**Decision:** Does this variant promote into the default P8 mechanism, stay as an option, or get discarded?
```

Entries are append-only. Superseded findings get new entries linking back, not edits.

## Promotion criteria — when a variant becomes the default

A variant earns a promotion into the default P8 mechanism (and possibly back into substrate_plan as the new minimum) when **all** of the following hold:

1. It beats the current default on the same scenario by a margin of ≥2 std across ≥10 seeds.
2. It does not introduce new failure modes not present in the default (checked via P8's persistence round-trip and determinism tests).
3. It is at most modestly more complex than the default. Complexity-for-marginal-gain is rejected here as sharply as in the substrate plan.
4. It has been running as a tracked entry in this doc for at least one version cycle.

A promotion triggers a PR to substrate_plan.md updating P8's minimum mechanism and this doc's "current default" section. The old default stays in this doc as an archived entry.

## Current default (starts with P8's minimum)

*This section will be updated as variants are promoted. At kickoff (when P8 ships), the default is P8's minimum:*

- **Replay source:** top-N episodes by cumulative reward in the current session
- **Replay count:** N = 10
- **Replay update strength:** same as online Hebbian update
- **Sleep trigger:** one sleep phase per sim run, fired at end-of-session
- **Scope:** within-session replay only (no cross-session replay at the minimum)

## Seed research questions (to be turned into experiments over time)

These are the first questions to chase, ordered loosely by expected payoff. Not commitments — a future reviewer might reorder them or drop them entirely.

### Q1 — Is top-N-by-reward the right selector?

**Why:** "Most rewarded" is the intuitive pick but might over-fit to rare high-reward events. Alternatives: top-N by Hebbian centrality (most-connected episodes), top-N by novelty (episodes with concepts not yet consolidated), top-N by prediction error (episodes that violated expectations, if/when prediction becomes a thing).

**First experiment:** Hebbian centrality vs reward. Same scenario, same N, swap selector. Measure post-sleep F1.

### Q2 — Does replay strength matter?

**Why:** The minimum uses online-equivalent update strength. That might over-strengthen replayed links relative to naturally-encountered ones. Tapering strength (e.g., 0.5× online) might produce more balanced consolidation.

**First experiment:** three conditions — 1.0×, 0.5×, 0.25× update strength. Measure F1 and interference rate.

### Q3 — Does cross-session replay add value?

**Why:** The minimum only replays episodes from the current session. Biologically, consolidation includes older memories. If cross-session replay strengthens older episodes, retention over multiple sessions should improve.

**First experiment:** minimum (within-session) vs cross-session (top-N from all sessions). Long-running sim over ≥5 sessions. Measure F1 on session-1 concepts in session 5.

### Q4 — What is the interaction with P6 forgetting?

**Why:** Consolidation strengthens, decay weakens. If they fire on the same cycle, their interaction determines net effect. Are they orthogonal, or does replay effectively reset decay for replayed episodes?

**First experiment:** enable/disable decay during replay phases. Measure retention curves for replayed vs non-replayed episodes over multiple sessions.

### Q5 — When should replay-strengthened links promote from hippocampus to ATL as semantic links?

**Why:** Biological consolidation moves memories from hippocampus to cortex over time. The substrate plan has ATL as the store but does not yet have a promotion rule from hippocampus-episode-links to durable ATL edges. This is the closest thing to "semantic memory formation" the plan is going to produce.

**First experiment:** this is a design experiment, not a run-and-measure. Propose a promotion rule, implement it as a variant, run it in isolation, see if downstream retrieval still works.

### Q6 — Does over-consolidation produce rigidity?

**Why:** The dark side of strengthening is that an over-strong link is hard to overwrite. An agent that consolidates too aggressively might fail to update beliefs when the world changes.

**First experiment:** run H2 (known-bad-path avoidance) from [behavioral_convergence_practice.md](behavioral_convergence_practice.md) with and without aggressive consolidation. Add a "world change" midway — what used to be punished is now rewarded. Measure how fast the agent updates.

## How this feeds back into substrate_plan

Three possible paths:

1. **Promotion** — a variant becomes the new default for P8 (see promotion criteria above). Substrate_plan is edited to reflect the new minimum.
2. **New fallback** — a finding shows P8's current mechanism has a limitation the current fallback language doesn't cover. "If the whole thing fails" gets a new entry.
3. **New phase proposal** — a finding suggests consolidation needs more than a mechanism refinement — e.g., a new data structure or a new bio-system. That triggers a proposal for a substrate-plan phase, reviewed separately.

## Review cadence + forcing function

Every version bump triggers a review of this doc. Same protocol as [behavioral_convergence_practice.md](behavioral_convergence_practice.md): what ran, what was learned, what changed, what's next. Notes go in the release notes.

**Soft discipline (from substrate_plan's "Living-doc discipline" section):**

Once P8 ships in 0.5, try to log at least one new consolidation experiment per version bump. This is a soft discipline, not a hard gate. If a version ships without a new entry, that's a signal to ask why and be honest about the trade-off.

**The 1.0 release should have at least one entry in this doc** (P8 has shipped by then, so the doc is active). Enforce that one on yourself at tag time.

## Experiments

*(No experiments yet. The first entry lands when P8 ships in 0.5.)*
