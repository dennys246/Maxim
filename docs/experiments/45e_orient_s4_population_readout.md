# Exp 45e — Orient magnitude S4: population-vector readout resolves far-bin starvation

**Status:** COMPLETE (2026-07-27). Live Reachy Mini, substrate-primary (`live_3_learn.py --perturb`, no LLM in the action path). Resolves the single-far-bin **cell-starvation** ceiling that [Exp 45d](45d_magnitude_replication.md) characterized, via the population-vector readout 45d named as the forward path.
**Scripts:** [`scripts/orient_backbone/live_3_learn.py`](../../scripts/orient_backbone/live_3_learn.py) (`--readout population`).
**Prior:** [45b](45b_orient_magnitude.md)/[45c](45c_flip_bins.md) (magnitude via the derived bin boundary, n=1 each) · [45d](45d_magnitude_replication.md) (replication + magnitude cross-session; starvation ceiling characterized).
**Plan:** [orient_magnitude_learning.md](../plans/deferred/orient_magnitude_learning.md) (S4).
**Data provenance:** operator hardware session 2026-07-27, NAc at `~/.maxim/orient_live/nac_reachy_argmax_seed3.json` + `--log` JSONL. **Durable-data follow-up:** commit the seed-3 session JSONL to `docs/experiments/data/45e_*.jsonl` to complete the regression-guard citation (held on the operator's box at write time).

## Why this run exists

45d closed magnitude replication but characterized a residual ceiling: per seed, **exactly one far bin's big-turn cell never gets a positive exploration sample**, so per-cell tabular argmax reads the wrong magnitude there (seed 3: `far_left` learned `turn_left` at +0.21 while `turn_left_big` stayed ~0.03). 45d named the fix — a **continuous readout that shares evidence across same-eccentricity bins** (population-vector / cerebellar gain adaptation) — but did not run it. S4 is that readout. Seed 3 (the documented starved seed) is the test case.

## The S4 arc (seed 3, one hardware session per arm)

| arm | readout | session | direction | magnitude | note |
|---|---|---|---|---|---|
| **s1** | argmax | fresh | **1.00** | 0.75 | `far_left` starved: `turn_left` 0.21 vs `turn_left_big` 0.03 → argmax reads small |
| **s2** | argmax | resume (no `--fresh`) | 1.00 | 0.75 | **cross-session:** first-10 greedy turned-toward = **1.0** (starts correct); biases compound (`far_right` `turn_right_big` 0.62→1.0) |
| **s3-pop** | population | resume | 1.00 | **1.00** | population readout resolves `far_left` → `turn_left_big` |
| **s4** | argmax | resume (on s3 file) | 1.00 | **1.00** | **readout-independent:** `far_left` `turn_left_big` **0.93** > `turn_left` 0.42 under plain argmax |

## Result 1 — the population readout is an exploration BOOTSTRAP, not a readout cosmetic

The load-bearing finding, and it is subtler than "our readout patches the number." On the starved biases, the population readout does **not** fix magnitude at readout time: at probe 10 and probe 15 it still reads `far_left → turn_left` (magnitude 0.75). What it changes is the **executed greedy action** — it pools the "big" magnitude from `far_left`'s mirror (`far_right`'s learned `turn_right_big` = 1.0) and *executes* `turn_left_big` at `far_left` (trials 15/20/21, relief +0.43 / +0.50 / +0.42). That big turn then **earns its own credit**, `turn_left_big`'s raw bias climbs 0.10 → 0.49 → and overtakes `turn_left`. Magnitude flips to 1.00 at probe 20.

So the mechanism is: **population readout directs exploration to the starved cell's borrowed magnitude → the cell earns real credit → the learning consolidates.** (The script's help text calling the two readouts "learning-unchanged, greedy-readout-only differs" is imprecise: because the *executed* action differs, the *credited* outcome differs, so learning downstream differs. Corrected here.)

## Result 2 — the fix is readout-independent (the confirm)

s4 re-runs with plain `--readout argmax` on the s3-trained NAc. `far_left` reads `turn_left_big` (bias **0.93** ≫ `turn_left` 0.42) → magnitude **1.00 under the discrete tabular readout**, no pooling involved. The population readout's job was to *seed* the bias flip; once seeded, argmax holds it. This is the strong version of the claim: the starvation resolution **outlived the readout that caused it**.

## Result 3 — cross-session transfer (re-confirmed)

s2 loads the s1 NAc and probes correct from trial 0 (first-10 greedy turned-toward = 1.0 vs s1's 0.625), with biases compounding across sessions rather than merely reloading (`far_right turn_right_big` 0.62 → 1.0, `near_left turn_left` 0.21 → 0.49). Consistent with the 45d cross-session arm; no new claim, corroboration.

## Honest footnotes (read before citing)

- **n=1, seed 3 only.** Every arm is a single hardware session on the one seed 45d documented as starved. This demonstrates the bootstrap *resolves the seed-3 starvation instance*; **multi-seed replication of the bootstrap is outstanding** (same evidential caveat the 45b/c/d rows carry).
- **Centering did not improve.** S4 residual `|az|` settled `0.157` (population) vs `0.113` (s2 argmax) — inflated by exploration trials in the last 10 (bad-outcome explores at trials 31/33/36), noise not regression. Population fixed *magnitude-appropriateness*, nothing else; do not claim it tightened centering.
- **The `far_left` asymmetry has a hardware component.** Seed 3's left/right starvation asymmetry is consistent with a known left-side motor issue (replacement motor pending); the population readout resolves the *learning* side of it, but the underlying actuation asymmetry is not a substrate property.
- Apparatus gain estimate drifted 0.52 → 0.61 az/rad across the four arms (warm-up / thermal); direction and magnitude arms are sign/bin-based and robust to it.

## Disposition

**Extends the [Exp 45 graduation row](../plans/behavioral_graduation_candidates.md) (real-hardware substrate sensorimotor learning).** It does not graduate a new claim — it *resolves the far-bin starvation ceiling* that 45d flagged, on the seed 45d documented, via the Layer-2 continuous readout 45d anticipated, and shows the resolution is readout-independent. The magnitude claim moves from "0.75 ceiling from tabular-argmax coverage, per 45d" toward "1.00 achievable via population-readout exploration bootstrap (seed 3, n=1; multi-seed outstanding)." No LLM in the action path throughout.
