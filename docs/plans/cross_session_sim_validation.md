# Cross-Session Sim Validation — proving pre-deliberation enrichment works

**Status:** Shell plan (2026-04-22)
**Scope:** ~50-100 LOC (mostly sim scripting + analysis)
**Priority:** Medium — validates the 1.0 research claim
**Depends on:** deliberation_observability.md (should ship first so we can see what's happening)
**Gates:** none, but results inform 1.0 confidence
**Target version:** 0.8.x validation

---

## Goal

Prove that Layer 1 pre-deliberation enrichment produces measurably different behavior when the agent has prior session history vs. a fresh start. This directly validates the 1.0 research claim: "cross-session learning without fine-tuning."

## The enrichment gap today

In the 2026-04-22 sim runs, pre-deliberation enrichment was wired but the agent was always starting fresh:
- Hippocampus: empty (no prior episodes to recall)
- NAc: empty (no causal links to predict from)
- ATL: empty (no semantic concepts formed)
- WMS: only has this-session turns (useful by turn 2+, but shallow)

After even one prior session, all four bio-systems have data. The enrichment pipeline should surface "you tried X last time and it failed" or "this entity is associated with negative valence."

## Experiment design

### Phase 1: Baseline (fresh session)
```bash
maxim --sim "escape a dungeon with a sleeping guard" \
  --interactive false --sim-max-turns 8
```
Record: session_id, action sequence, hippocampus captures, NAc links formed.

### Phase 2: Resume (cross-session)
```bash
maxim --sim "escape a dungeon with a sleeping guard" \
  --interactive false --sim-max-turns 8 \
  --resume-sim <session_id_from_phase_1>
```
Record: same metrics. Compare:
- Does the "WHAT YOUR EXPERIENCE TELLS YOU" section populate?
- Does the action sequence differ from phase 1?
- Are prior episode memories surfaced in the enrichment?
- Do NAc predictions appear (e.g., "base_humanoid_move → positive, high confidence")?

### Phase 3: Negative transfer test
Run a different scenario after the dungeon session:
```bash
maxim --sim "you are in a peaceful garden, enjoy the flowers" \
  --interactive false --sim-max-turns 5 \
  --resume-sim <session_id_from_phase_1>
```
Verify: dungeon-specific memories don't dominate the garden scenario. Enrichment should show low relevance scores or empty results for unrelated content.

## What to measure

| Metric | Fresh | Resume | Expected |
|--------|-------|--------|----------|
| Enrichment sections populated | 0-1 (WMS only) | 3-4 (memories + predictions + concepts + WMS) | Resume >> Fresh |
| Action diversity (unique tools / total actions) | ~0.3 | ~0.5+ | Resume shows more varied strategy |
| Repeated actions (same tool consecutively) | High | Lower | Prior experience prevents repetition |
| NAc predictions in prompt | 0 | 2-5 | Resume has learned causal links |
| Hippocampal recalls in prompt | 0 | 1-3 | Resume has episodes to retrieve |

## Prerequisite

**deliberation_observability.md must ship first.** Without sim_log visibility into the enrichment pipeline, we can't measure whether enrichment sections populated. We'd be back to inferring from indirect evidence.

## Key files

| File | Purpose |
|------|---------|
| `simulation/orchestrator.py` | Resume-sim wiring (already works) |
| `agents/exec_agent.py` | `_run_pre_deliberation` (already wired) |
| `integration/bio_enrichment.py` | `enrich()` (already accepts working_memory) |
| New: `docs/experiments/09_cross_session_enrichment.md` | Results doc |

## Success criteria

The experiment passes if:
1. Resume session shows 2+ enrichment sections populated that were empty in fresh session
2. Action sequence in resume session differs from fresh session in a way traceable to enrichment content
3. Negative transfer test shows enrichment doesn't dominate unrelated scenarios
