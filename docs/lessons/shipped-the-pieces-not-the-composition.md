# A fix with no callers (2026-09-02)

**D43 was declared fixed, merged, and changed nothing that runs.** PR #590
shipped `ec_merge_aligned`, `rekey_nac_state` and `nac_merge_many` — all three
correct, all three tested — and left their COMPOSITION to call sites. There
were zero. Every shipped consumer still called bare `nac_merge`, so through
every real path a merged foreign want still read out as exactly `0.0`, which
is the defect D43 names.

## How it got that far

The PR body claimed a measurement:

> A receiver that never saw the contingency: **0.0 → 1.0**, 4 of 4 merged bias
> keys naming a reachable cluster (was 0 of 4).

That number was real. It was taken on a sequence composed **by hand in the
verification script** — align, re-key, fold — which no shipped code path
performed. The measurement proved the pieces could be composed correctly, and
was presented as proof that they were.

Nothing in the normal battery could see it. 9,820 tests passed. Every lint was
clean. The architecture audit was clean. The new unit tests passed because they
called the new functions directly — which is exactly what a unit test does, and
exactly why they could not notice that nothing else did.

## What caught it

`test_d44_merge_behavioural_delta.py`'s three behavioural arms were
`xfail(strict=True)`. After D43 merged they were expected to XPASS and turn the
suite red, forcing the marker's removal.

**They kept failing.** That is the whole finding. A non-strict `xfail` would
have stayed quietly yellow through both merges.

The reflex at that moment is to delete the markers and get green — the fix
*is* merged, after all. The second reflex, barely better, is to re-point the
arms at the hand-composed sequence, which turns a ship gate into a test of a
recipe. Both produce a green suite over a system where the defect is live.

## The generalisable shape

This is the [green-PR-with-no-tests-run](green-pr-with-no-tests-run.md) family
stated at the level of source rather than CI:

> **A mechanism that does not run looks exactly like one that ran and found
> nothing.**

There, a `CONFLICTING` PR suppressed the whole `Tests` workflow and the page
rendered three green rows. Here, a merged fix had no callers and the suite
rendered 9,820 green tests. In both cases the *absence* is invisible and every
present signal is genuinely passing.

The specific trap for library-shaped fixes: **a defect that lives in a
COMPOSITION cannot be fixed by shipping better pieces.** D43's mechanism was
"`ec_merge` computes an alignment and discards it" — an ordering fact about how
two functions are called. Returning the map makes the fix *possible*; it does
not make it *happen*. When the bug is in the seam, the fix belongs in the seam,
as one callable thing.

## Rules

1. **A fix ships with a caller, or it has not shipped.** Before declaring a
   defect fixed, grep the new symbols across `src/` and `scripts/` excluding
   tests. Zero non-test callers means the fix is a capability, not a fix.
2. **Say where a number was measured.** "Measured end to end" must name the
   entry point. If the sequence was composed in the verification script, it is
   a measurement of a possibility.
3. **Write red gates `strict=True`.** The strictness is the entire mechanism:
   a non-strict xfail that silently keeps failing is indistinguishable from one
   that silently starts passing.
4. **When a red gate does not flip, that is data — do not remove the marker.**
   The gate is reporting on the merge, and it is the only thing that can.
5. **Anti-vacuity guards must cover the fix's own new degrees of freedom.**
   Wiring the composition made EC ingestion part of the gate, which introduced
   a fresh way to pass hollowly: ingesting the donor's clusters relocates which
   cluster the receiver lands on, so "score went up" could be relocation rather
   than transfer. Every mechanism added to a gate's path needs its own
   null arm.

## Collateral found by wiring it

Composing the path surfaced a defect neither the sweep nor the design brief
predicted, because both reasoned about the merge and this lives after it:
**there was no way to load merged EC nodes into a live EC.**
`register_substrate_node` hardcodes `self._substrate_node_counts[node_id] = 1`,
discarding the `member_count`-weighted centroid mean the merge had just
computed (merge design decision 3 — "a node with 10 members weighs 10x a
1-member node"). Each federation round would have erased the previous round's
evidence weighting. `EC.load()` restores counts but is file-based and clobbers
signatures and LSH tables. Hence `EC.ingest_substrate_nodes`.

Generalisation: **an ingestion has two halves, and the second one is easy to
forget because the first one returns a value that looks like the answer.**
`substrate_merge` returns merged nodes; the receiver is not changed until
something loads them.
