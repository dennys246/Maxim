# Temporal Credit Validation — Experiment Results

> **Status (audit 2026-09-13): STALE** — protocol + runner shipped ~2026-04 (commit 6dccd363)
> but the 4-set run was never executed (wrong: it ran and produced no evidence, see the 2026-10-08
> correction below); every Result cell still reads TBD and no data exists under `data/` or `results/`.
>
> **Corrected 2026-10-08 (1.4 grounding line, GL0 truth pass): it ran, and produced no evidence.**
> "Never executed" was wrong. The runner (`scripts/temporal_credit_validation.sh`) was executed about
> **19 times on the owner's Mac on 2026-04-24/25** while the script was being debugged (results
> directories `~/.maxim/experiments/temporal_credit_20260424_*` / `temporal_credit_20260425_*`, local
> only, never committed): mostly Sim Set 1 (dragon → mage), one run with Sets 2–3. **No result was ever
> recorded here.** A scan of the most complete runs' JSONL found **zero** temporal-credit,
> `credit_node`, `reward_bias` or `credit_goal` entries, and the reports carry `_llm_unavailable`
> fallbacks. The runs are unstamped (no code-under-test, model or context provenance), so they are weak
> evidence and **cannot hold a status** in either direction (CLAUDE.md, weak evidence never gates).
> Set 1's headline `[DANGEROUS]` criterion cannot pass by construction (#910, below). **What is owed:**
> a redesigned, pre-registered run of the SCN phase-similarity fallback credit (Hypothesis 3), with a
> four-lens design review first ([DESIGN_REVIEW.md](DESIGN_REVIEW.md)); [#1180](https://github.com/dennys246/Maxim/issues/1180). Until it lands,
> the `[behavioral]` SCN-coupling stub in [docs/agents/bio-memory.md](../agents/bio-memory.md) that
> cites this doc is marked **evidence PENDING** (owner decision 2026-10-08: run the experiment rather
> than demote the tag).

**Plan:** [temporal_credit_integration.md](../plans/archive/temporal_credit_integration.md)
**Protocol:** [temporal_credit_validation.md](protocols/temporal_credit_validation.md)
**Date:** TBD
**Model:** qwen2.5-14b-instruct on RTX 5080

---

## Hypothesis

The temporal credit integration (Phases 1-7) enables three capabilities not present before:

1. **Cross-session affordance transfer**: negative reward bias on fire-related substrate nodes persists across sessions, causing `[DANGEROUS]` annotations on novel fire affordances without direct experience. **(2026-10-04: this criterion cannot pass. `reward_bias` is clamped to ≥ 0, so the label never fired, and it was removed in [#910](https://github.com/dennys246/Maxim/issues/910); a re-run must not report its absence as a finding. Reading the stores that hold harm is #910's deferred option 1.)**
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
