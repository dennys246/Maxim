# L2 — Deterministic-apparatus phase-locking (tracking doc)

**Ledger entry:** [README.md](README.md) §L2 · **Disposition:** MITIGATED (#514)
**Instrument:** cyclic scripted stimulus × deterministic agent — canonically
`cradle_mother` directedness; the class covers ANY closed loop of two periodic
deterministic processes scored by their alignment.

## The limit, precisely

When both the stimulus sequence and the agent policy are deterministic, the
closed loop falls into a phase-locked attractor. The metric then reports the
attractor's geometry — a small set of exact fractions — not the policy's
quality. Learned-bias changes move the metric only by *jumping between
attractors* (discontinuous), never gradually. This is the argmax
visibility-floor problem (L1) at whole-apparatus scale: below a threshold,
invisible; above it, a snap.

**Red-flag signature:** seed-invariant results on a "seeded" apparatus, and/or
metric values that are exact small rationals of the bin size.

## Measurement history

| date | apparatus | observation | source |
|---|---|---|---|
| 2026-08-11 | v1 (stopwatch regime) | Contest raises "directedness may measure a phase relationship" as a hypothesis: 50/50 L/R alternation, outcome autocorrelation 0.62, both stimulus and agent alternating deterministically | [48 §heartbeat re-run](../experiments/48_cradle_mother_seam.md) |
| 2026-08-17 | v2, ew 1.5 | taught late 8/12, control 2/12 — exact twelfths, 12/12 seeds identical | `data/48_rebaseline_v4.jsonl` |
| 2026-08-17 | v2, ew 1.0 | taught late 8/12, control 2/12 — plateau invariant to exploration share | `data/48_sweep_ew1.0.jsonl` |
| 2026-08-18 | v2, ew 0.75 | **taught 4/12, control 6/12 — arms INVERT; the control moved with zero teaching.** Hypothesis → demonstrated | `data/48_sweep_ew0.75.jsonl` |

## What raises the ceiling (mitigation lineage)

1. **#514 seeded stimulus-order shuffle** (`MotherScaffold.stimulus_order="shuffled"`):
   per-block permutation — exposure balanced (every stimulus once per block,
   S5 preserved), order unpredictable, deterministic per (seed, block) via int
   seeding. Dithers the measurement back into a graded function of the policy;
   seeds finally produce different trajectories on this apparatus.
2. Candidate, unmeasured: narrator temperature > 0 would also dither, but
   couples the dither source to the LLM (machine-dependent, cost-bearing) —
   the seeded shuffle is preferred because the dither is in the *stimulus*,
   fully reproducible, and exposure-balanced by construction.

## Open questions

- Does the shuffle fully de-quantize directedness, or do residual attractors
  survive within a block? → answered by the apparatus-v3 campaign's
  seed-variance (if v3 curves are still exact fractions, the agent side needs
  dither too).
- Does the qualified MOTHER-TAUGHT interpretation (credit-tipped attractor
  selection) convert back to graded teaching under dither? This is the v3
  campaign's core question.

## Re-measure on

- The apparatus-v3 campaign (`--stimulus-order shuffled` + the frozen v3
  gate) — its seed-variance IS the re-measurement of this limit.
- Any new scripted-stimulus arc scored by an alignment metric (habituation
  sequences, Exp 46-class crèche arcs) — check the red-flag signature before
  trusting its first campaign.
- Narrator temperature or decoding-strategy change (alters the agent-side
  determinism this limit depends on).
