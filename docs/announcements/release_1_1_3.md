# pymaxim 1.1.3 — "Reachability"

**Released 2026-09-03.** `pip install --upgrade pymaxim`

1.1.2 closed four gates that could not fail. 1.1.3 is what happened when we kept pulling that
thread, and the name is literal: **every headline defect in this release was a mechanism that
existed and could not be reached.** A merged want that read out as zero. A regression guard that
passed against a stub. A provenance stamp nothing read. A fix with no callers. A test that was
structurally impossible.

None of them looked broken. That is the point — and it is the same shape each time: *a mechanism
that does not run looks exactly like one that ran and found nothing.*

## D43 — a shared want read out as exactly 0.0, and the success indicator moved the wrong way

`ec_merge` computed a right→left node alignment and **discarded it**, while `nac_merge` folded
`cluster_reward_bias` on exact string keys. A donor's biases therefore landed under cluster ids
that are not nodes in the receiver's EC — structurally unreachable at readout — while the merge
reported success and the bias dict *grew*.

That growth is the tell. `len(cluster_reward_bias)` is `|left ∪ right|`, which is **maximal
exactly when nothing aligns**: the number people were reading as success was inversely correlated
with it.

Measured on a receiver that never saw the contingency: **0.0 → 1.0**, with **4 of 4** merged bias
keys naming a reachable cluster (was 0 of 4).

One half of the fix appeared in no plan document. `ec_merge`'s `0.44` threshold is tuned for
paraphrase-mpnet **text**; interoception clusters — the ones that key `cluster_reward_bias` — form
at **0.85**. Returning the alignment without retuning would have collapsed every donor cluster
onto whichever receiver node scored highest: a *confidently wrong* alignment replacing an honestly
missing one, which is strictly worse.

## The lesson of the release: a fix that merged and changed nothing

The D43 fix shipped three correct, tested functions — and left their **composition** to call sites,
of which there were **zero**. Every consumer still called the old path. The defect stayed live
behind 9,820 passing tests, clean lints and a clean architecture audit.

It was caught only because the ship gate had been written red with `xfail(strict=True)` and, after
the fix merged, **did not flip to XPASS**. A non-strict marker would have stayed quietly yellow.

So: **a fix ships with a caller, or it has not shipped.** Before declaring a defect fixed, grep the
new symbols across `src/` and `scripts/` *excluding tests* — zero non-test callers means capability,
not fix. Full write-up:
[`docs/lessons/shipped-the-pieces-not-the-composition.md`](https://github.com/dennys246/Maxim/blob/main/docs/lessons/shipped-the-pieces-not-the-composition.md).

## D62 — a regression guard that passed against `return left`

`orient_merge_arm.py` is cited as the regression guard for an **earned** behavioural result, under a
`Re-run on: nac_merge semantics change` trigger. You could have gutted `nac_merge` and it stayed
green: the gate read `correctness` only, never the `magnitude_appropriateness` it printed on every
run, and both recorded parents were already perfect — so `merged >= max(parents)` sat at ceiling and
carried no information.

Three vacuities, and only one was a threshold problem. `--assert-noop-fails` is now the guard on the
guard: it re-runs the gauntlet with `nac_merge` replaced by `return left` / `return right` /
`return {}` / a naive dict update, and **exits non-zero if any still passes.**

## Gates 1 and 2 — compatibility checks that could not fire

**Gate 1 (D1).** `encoder_provenance` was recorded, persisted, reloaded — and compared against
nothing. Its only readers were the bundle export. A geometry change loaded old-geometry centroids
and cosine-scanned them against new embeddings.

**Gate 2 (D3/D4).** `_cosine` refuses a *dimension* mismatch, catching a 384-vs-768 encoder swap. It
cannot see a **same-dimension** change: a place code adds sensor names, so the basis set changes and
the length does not. Because `audio` is a frozen-centroid modality the centroid never moves, so the
only symptom is inflated counts.

EC nodes now carry a `geometry` tag derived from what actually makes two vectors comparable. Two
nodes declaring different geometries never fold — **at cosine 1.0 included**, because similarity
across spaces is not a small number, it is undefined.

## The review round earned its keep

A three-lens pre-merge round ran against that green suite and returned **two blockers, both
cross-confirmed by independent lenses**:

1. **The geometry tag named the reading, not the space.** It hashed the per-tick sensor keys — but a
   corrective `cold` need appears only while a thermal drive is outside its comfort band. Measured: a
   warm infant and a cold one hashed to *different* geometries, their clusters became mutually
   unreachable, and the guard fired **in both directions during routine thermoregulation**. A
   contingency learned while warm was invisible the moment the body got cold.
2. **`geometry` was an optional kwarg whose omission silently disabled the guard** — and a live
   caller had already omitted it, on a path that *mutates centroids*.

Both were folded in before merge, not filed after. The round also found gate 1 was **inert for every
existing installation** (no `ec.json` has a geometry tag, and completion never stamped one), which is
why nodes now stamp on first touch.

## Evidence

All three `Stale` graduation rows are discharged. **Exp 52** re-ran in simulation and reproduced
GRADUATE (taught 0.837 / satiated 0.413 / no_feed 0.413, all gates PASS). **Exp 53b** re-ran **on the
robot** and reproduced its original exactly (taught 1.00 / satiated 0.00 / no_feed 0.50).

Two honest notes travel with that. Exp 53b's platform emitted **85 actuator-degradation warnings
across 180 trials** — roll and pitch, never yaw, and azimuth readout rides on yaw, which is why the
result stands rather than being retracted. And one sentence of Exp 52's original write-up is
**retracted**: per-seed values are not reproducible run-to-run, so the "one weak seed" narrative was a
post-hoc story about a number that does not hold still. The arm-level means replicate; the per-seed
ones do not.

## Owed

Gate 1's specification says *"reject **or migrate** incompatible state."* Only **reject** shipped. An
`ec invalidate` / re-encode path is still owed, and the warning text currently promises a remedy that
does not yet exist.

---

**Full changelog:**
[CHANGELOG.md](https://github.com/dennys246/Maxim/blob/main/CHANGELOG.md) ·
**Defect ledger:** [docs/bugs/README.md](https://github.com/dennys246/Maxim/blob/main/docs/bugs/README.md)
