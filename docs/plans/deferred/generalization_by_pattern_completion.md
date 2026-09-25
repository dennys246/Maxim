# Memory 2S-e (B) — generalization by pattern completion (parked)

> **PARKED 2026-09-25 (owner).** Memory-strength 2S-e's chosen consumer
> ([memory_strength_and_forgetting.md](../memory_strength_and_forgetting.md) §2S-e): in a situation the
> NAc has not keyed, complete to the most similar past situation and carry over what happened there.
> Parked because **its measured gap is gone**, not because the idea failed.

## Why parked

(B) was to fill R1's measured gap — "the fear misses in the night pool at 0.799, under the 0.85
threshold" — gated by Rung B's SUPPORT (outstanding O3). The SUPPORT design review (2026-09-25,
[rationale/rungb-support/](../../experiments/rationale/rungb-support/)) showed:

- The 0.799 is `time_of_day` 0.99, the minute before the clock wraps; at the fear place midnight reads
  0.903, inside the key. The miss is the linear encoding of a circular clock — a KEYING defect,
  [#899](https://github.com/dennys246/Maxim/issues/899) — which a graded read would only paper over.
- At the fear place a real day gives no SUPPORT: ~0.95 of it is inside the key, ~0.05 in the
  [0.75, 0.85) middle, and that middle is the wrap.
- The other measured wall, a lit surface pond at 0.588, is beyond any reasonable graded-read floor: a
  keying question (does `light_level` belong in the world channel), not a generalization one.

## The design as it stood (for whoever revives it)

Two readings of "carry over what happened there" were on the table:
1. **A graded NAc fear read** at the cluster boundary: when the world cluster is unkeyed, complete (EC
   read-only, excluding this tick's own new node — the EC discards the nearest id on a separation
   today) to the nearest **feared** node, not the nearest node (with the day cycle on, a day is a
   chain of clusters and a nearest-node read carries fear one hop), and fold a similarity-weighted
   fear into `drives["threat"]`. Needs no captures. The Exp 62 prereg's "Rung B remedy".
2. **The episodic route**: complete, recall with the stateless `PatternCompleter.recall_situation`
   (2S-d), derive threat from the recalled memories' outcomes, with outcome-gated credit. Needs a
   world whose memories are captured before the test read (no Exp 60–62 phase does that).

Inherited from 2S-d and still owed by any revival: outcome-gated credit, the ranking policy, the
stateless seam, loop captures in the validation world, the lossy-refs fallback trigger (see the
memory plan's 2S-e "Inherited from 2S-d" list). Its front-gate answer must still compare against a
substrate keying rule.

## Revive when

A **measured** generalization gap exists that keying does not own: the same danger missed across two
states that genuinely differ in the world channel but sit within reach of a graded read (cos in roughly
[0.75, 0.85) to a feared node), measured at the place the fear was learned — e.g. after #899's fix,
on an open-world trace with place and weather variation, OR when a rung needs carry-over of something
the NAc does not store (which action worked, relief).
