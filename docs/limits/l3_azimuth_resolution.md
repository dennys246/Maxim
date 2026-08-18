# L3 — Azimuth representational resolution (tracking doc)

**Ledger entry:** [README.md](README.md) §L3 · **Disposition:** MITIGATED (#499, flag default OFF)
**Instrument:** exteroceptive azimuth encoding → EC clustering (what the
substrate can *represent*; contrast L2, which bounds what a *metric* can see).

## The limit, precisely

The raw scalar azimuth encode resolves ~3 distinct place-like nodes
(left/centre/right). A learned policy cannot be more graded than its
representation: graded-orient claims on the raw scalar are structurally
capped at 3-way discrimination regardless of learning quality.

## Measurement history

| date | apparatus | measured resolution | source |
|---|---|---|---|
| 2026-07-22 | Exp 46 creche, raw scalar | 2 clusters (graded orient 0.19 pre-fix) | [46_operant_orient_creche.md](../experiments/46_operant_orient_creche.md) |
| 2026-07-22 | Exp 46 + Gaussian place code (experiment-side) | 6/6 distinct clusters, graded orient 0.19 → 0.82 | same |
| 2026-08-11 | RSC pre-check, production path | **~3 nodes** — channel is RESOLUTION-bound, not frame-confused (refuted the predicted failure mode; re-sequenced the RSC plan) | `scripts/rsc_precheck.py`; [retrosplenial_spatial_frames.md](../plans/deferred/retrosplenial_spatial_frames.md) §2 |
| 2026-08-12 | production, `MAXIM_PLACE_CODE_EXTEROCEPTION=1` | **7 nodes** (3 → 7) | [modality_resolution_and_alignment.md](../plans/modality_resolution_and_alignment.md) §7 (#499) |

## What raises the ceiling (mitigation lineage)

1. **#499 Gaussian place code** (flag, default OFF): 3 → 7 nodes on the
   production path. Default-ON is gated (the 1.1.x roadmap item): D2 `ec`
   invalidate command, D4 hivemind merge dim-guard, `min_confidence`
   recalibration — and the **L1 interaction**: splitting one cluster into N
   divides per-node learned bias; the split signal must stay above the ~0.11
   visibility floor or higher resolution *reduces* behavioral legibility.
   (Instrument for checking: #504's `learned_margin`.)
2. Beyond 7: the cross-modal fabric's population-coding line (1.3) — not on
   any near-term path.

## Open questions

- Post-default-ON: does 7-node resolution survive hivemind merge with
  old-geometry (3-node) bundles? (D4 is the guard gap.)
- The Exp 48 re-runs the place-code flag note requires (48 + 49 H3/arm C)
  double as this limit's behavioral re-measurement.

## Re-measure on

- Place-code default-ON (the flag's own gate list) — re-measure node count
  AND per-node `learned_margin` vs the L1 floor.
- Encoder model swap (standing graduation trigger — resolution rides the
  encoder).
- Adding a `space` ModalityChannel (the RSC re-validation registry's trigger
  — cluster geometry changes upstream of everything cluster-keyed).
