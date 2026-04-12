# Experiments — Lab Notebook

This directory is the auditable evidence trail for Maxim's substrate research claims. Every experiment that produces data referenced by a plan, a pass/fail decision, or the 1.0 release criteria gets an entry here.

## Why this exists

The 1.0 claim is "cross-session learning without LLM fine-tuning." That claim requires evidence, and evidence requires reproducibility. A result that can't be regenerated from the repo is not evidence — it's an anecdote.

## Structure

```
docs/experiments/
    README.md                           # this file
    results/                            # machine-readable JSON (programmatic comparison)
        p0_baseline_sweep.json          # raw numbers from P0 threshold sweep
        p1_mechanism_test.json          # (future) P1 pass/fail data
        ...
    p0_baseline_sweep.md                # methodology, results, decision, repro commands
    p1_mechanism_test.md                # (future) P1 experiment entry
    ...
```

## Entry template

Every experiment entry follows this structure:

```markdown
# [Phase] — [Experiment Name]

**Date:** YYYY-MM-DD
**Phase:** P0 / P1 / P2 / ...
**Status:** recorded / superseded / invalidated
**Code version:** git hash
**Decision:** what was decided based on this data

## Hypothesis

What we expected to see and why.

## Methodology

Exactly how the experiment was run — commands, parameters, hardware.

## Results

Tables, numbers, analysis.

## Reproduction

Copy-paste commands to regenerate the results from scratch.

## Raw data

Link to `results/<filename>.json` for machine-readable output.
```

## Rules

1. **Every entry must have a Reproduction section** with copy-paste commands. If it can't be reproduced, it doesn't belong here.
2. **Raw data goes in `results/`** as JSON. Markdown entries reference it. Future phases can load prior results programmatically.
3. **Entries are append-only.** Don't edit a recorded entry — if results change, add a new entry and mark the old one as superseded with a link to the replacement.
4. **Tie to git.** Every entry records the git hash it was run against. Results are only valid for that code version unless explicitly re-validated.
5. **Living practice docs link here.** [behavioral_convergence_practice.md](../plans/behavioral_convergence_practice.md) and [memory_consolidation_practice.md](../plans/memory_consolidation_practice.md) reference entries by filename when citing experimental evidence.
6. **Plan decisions link here.** When a plan phase passes or fails, the decision entry in the plan links to the experiment that produced the evidence.

## Index

| Entry | Phase | Date | Status | Decision |
|---|---|---|---|---|
| [p0_baseline_sweep.md](p0_baseline_sweep.md) | P0 | 2026-04-12 | recorded | Fixtures well-calibrated (78.5% @ best operating point). Proceed to P1. |
| [p1_recognition_sweep.md](p1_recognition_sweep.md) | P1 | 2026-04-12 | recorded | Recognition criteria met (93.5% collapse, 3.3% cross-cluster). Proceed to P2. |
