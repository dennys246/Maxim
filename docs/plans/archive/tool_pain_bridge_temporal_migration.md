# ToolPainBridge Temporal Event Migration

**Status:** Shell plan (2026-04-24)
**Scope:** ~50 LOC across ToolPainBridge + bio_stack wiring
**Depends on:** [temporal_credit_integration.md](temporal_credit_integration.md) (Phases 1-7 shipped)
**Branch:** TBD (focused PR)

---

## Problem

ToolPainBridge has 4 `scn.register()` call sites (lines 153, 276, 328, 433) that write temporal signatures to SCN bins but never connect to NAc temporal credit.  These are one-way writes — the data goes in but never comes back out for credit attribution.

With the `TemporalCreditDistributor` shipped (Phase 4), these sites should also emit `TemporalEvent`s so tool outcomes and pain signals get temporal credit attribution via the distributor.

## What changes

1. **Thread the distributor into ToolPainBridge.**  Add `distributor: TemporalCreditDistributor | None = None` to the bridge constructor.  The distributor is wired via `build_executor` or the bio-stack construction path.

2. **At each SCN registration site, also emit a TemporalEvent.**  The event carries the same signature and temporal context, but routes through the distributor for eligibility + credit wiring.

3. **RE-3 fold:** Thread `Reaction.scn_tag` (CircadianContext) through the reaction subscriber.  Convert `CircadianContext` → `TemporalSignature` at the subscriber boundary (the distributor expects `TemporalSignature`).

## Key constraint

ToolPainBridge's existing NAc calls (`record_outcome`, `record_outcome_full`, `record_tool_embodiment_failure`) stay unchanged — they handle event→outcome causal links.  The temporal event emission is **additive** (SCN registration + eligibility anchoring for temporal credit fallback).

## Staging

Single PR.  The 4 sites are independent — each can be migrated and tested separately within the same PR.

## Validation

- Existing ToolPainBridge tests pass unchanged (causal link behavior is unaffected)
- New tests: verify TemporalEvents are emitted at each of the 4 sites
- IT-4 (temporal credit under decay) continues to pass
