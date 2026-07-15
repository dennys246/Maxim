# Temporal Credit Validation — Experiment Results

**Plan:** [temporal_credit_integration.md](../plans/archive/temporal_credit_integration.md)
**Protocol:** [temporal_credit_validation.md](protocols/temporal_credit_validation.md)
**Date:** TBD
**Model:** qwen2.5-14b-instruct on RTX 5080

---

## Hypothesis

The temporal credit integration (Phases 1-7) enables three capabilities not present before:

1. **Cross-session affordance transfer**: negative reward bias on fire-related substrate nodes persists across sessions, causing `[DANGEROUS]` annotations on novel fire affordances without direct experience
2. **Goal-level deliberation learning**: `_goal_reward_bias` accumulates across turns, modulating ThoughtGate threshold bidirectionally (positive = deliberate more, negative = act faster)
3. **Temporal credit fallback**: after fast-decay eligibility traces expire, phase-similarity anchors still enable credit distribution via `TemporalCreditDistributor`

## Results

### Sim Set 1: Cross-session affordance transfer

| Metric | Session 1 (dragon) | Session 2 (mage) |
|--------|-------------------|-------------------|
| Duration | | |
| Turns | | |
| `[DANGEROUS]` annotations | N/A | |
| NAc fire node reward_bias | | |
| credit_goal calls | | |
| ThoughtGate fires | | |
| Cost | | |

**Cross-session transfer observed?** TBD

**Evidence:**
<!-- Paste relevant JSONL excerpts or grep output here -->

### Sim Set 2: Goal-level deliberation learning (Arena)

| Metric | Value |
|--------|-------|
| Duration | |
| Turns completed | |
| credit_goal calls | |
| Unique goals with non-zero bias | |
| ThoughtGate fires (early) | |
| ThoughtGate fires (late) | |
| Goal bias range | |

**Bidirectional learning observed?** TBD

### Sim Set 3: Multi-entity imagination

| Metric | Value |
|--------|-------|
| Duration | |
| Entities instantiated | |
| ComponentIndex hits | |
| Imagination designs (LLM) | |
| Substrate nodes formed | |
| Affordance annotations | |

### Sim Set 4: Sensory deprivation (Darkened Cavern)

| Metric | Value |
|--------|-------|
| Duration | |
| Pain events | |
| Cerebellum updates | |
| Enrichment sections (early) | |
| Enrichment sections (late) | |

## Summary

| Set | Hypothesis | Result |
|-----|-----------|--------|
| 1 | Cross-session fire transfer | TBD |
| 2 | Bidirectional goal bias | TBD |
| 3 | Multi-entity imagination | TBD |
| 4 | Sensory deprivation adaptation | TBD |

## Observations

<!-- Notable behaviors, surprises, or issues discovered during the run -->
