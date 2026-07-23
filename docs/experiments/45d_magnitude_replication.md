# Exp 45d — Orient magnitude: replication + cross-session transfer of the full policy

**Status:** COMPLETE (2026-07-23). Live Reachy Mini, substrate-primary (`live_3_learn.py --perturb`, no LLM in the action path). Closes the "replication outstanding" flag on the Exp 45 graduation row.
**Scripts:** [`scripts/orient_backbone/live_3_learn.py`](../../scripts/orient_backbone/live_3_learn.py)
**Prior:** [Exp 45](45_reachy_orient_live.md) (direction, cross-session, merge — n=1 magnitude) · [45b](45b_orient_magnitude.md)/[45c](45c_flip_bins.md) (magnitude via the derived bin boundary, n=1 each).
**Data:** [data/45d_magnitude_replication.jsonl](data/45d_magnitude_replication.jsonl).

## Why this run exists

[45c](45c_flip_bins.md) earned magnitude 1.00 on a **single** hardware session by deriving the near/far bin boundary from the measured gain so no bin straddles the point where the optimal turn magnitude flips (`gain·(|Δbig|+|Δnormal|)/2 = 0.33`). The graduation row flagged the honest limitation: **n=1**, on a metric quantized to {0, .25, .5, .75, 1.0} over 4 bins, where the sim itself scored 0.75 on 2 of 6 seeds — so one draw had ~⅓ chance of landing PARTIAL. This run does the outstanding replication (3 clean seeds) AND adds the arm 45/45b/45c never ran: **cross-session transfer of the *magnitude* policy** (Exp 45's cross-session arm carried direction only).

All runs freeze the boundary at the canonical gain (`--flip-bins --az-gain 0.55` → boundary 0.330), so the "optimal" labels are identical across seeds and the scores are directly comparable; only the exploration RNG (`--seed`) varies. 40 credited trials, ε=0.25, fixed physical source placement.

## Result 1 — learning-curve replication (3 seeds)

Fresh NAc each run (`--fresh`); final-probe correctness (direction) and magnitude_appropriateness:

| seed | direction | magnitude | starved cell |
|------|-----------|-----------|--------------|
| 0 | **1.00** | 0.75 | far_left × big |
| 2 | **1.00** | **1.00** | none |
| 3 | **1.00** | 0.75 | far_left × big |

- **Direction is unanimous 1.00** across every completed run (also across the earlier aborted/mislabeled attempts) — the Layer-1 direction result is decisive, boundary-independent.
- **Magnitude lands 0.75–1.00, mode 0.75 (mean 0.83)** — matching the sim's "0.75 on ~⅔ of seeds." Every fresh run starts at trial-0 probe **0.00/0.00** (empty NAc) and rises, so this is learning, not a servo.

## Result 2 — the failure mode is coverage, not capability

The magnitude miss is dead consistent: **exactly one FAR bin's big-turn cell starves per seed.** In seed 3, far_**right** learned `turn_right_big` at cluster-bias **+0.585** (strong, decisive) while far_**left** never learned `turn_left_big` (bias exactly **0.0** — never positively sampled). The identical "far offset → big turn" structure is learned confidently on one side and completely absent on its mirror image. Both far bins must independently discover the big-turn cell, and 40 trials at ε=0.25 doesn't guarantee every (far bin × big) cell gets a positive sample. Seed 2 proves 1.00 is fully reachable when both happen to be covered — so this is a **coverage limit of per-cell tabular argmax**, not a capability limit of the substrate.

This is the data-grounded motivation for the Layer-2 continuous-readout work (`substrate_native_orienting` Layer 2): a **population-vector** (or **cerebellar gain-adaptation**) readout would let the two symmetric far bins **share** the "far → big" evidence instead of each rediscovering it, and would code magnitude in log/Weber-Fechner space (the parietal magnitude geometry, cf. the IPS Approximate Number System) rather than uniform linear bins.

## Result 3 — cross-session transfer of the FULL policy

Loaded seed 2's trained NAc (the 1.00/1.00 policy) into a **fresh session** (`s2`, no `--fresh`) and probed before any new learning:

- **Trial-0 probe: 1.00 / 1.00** — the complete direction *and* magnitude policy transferred intact, correct from the first probe.
- **All 9 probes across the session: 1.00 / 1.00** — the loaded policy held perfectly stable through 40 trials of continued exploration; it did not drift or degrade.

Exp 45's cross-session arm showed *direction* transferring at trial 0; this extends it to **magnitude**. The substrate persists a learned continuous-ish motor policy across sessions on real hardware.

## Disposition

- **Direction:** EARNED decisively (unanimous 1.00, replicated).
- **Magnitude:** EARNED with characterized ceiling — 0.75–1.00 (mode 0.75) at 40 trials, capped by single-far-bin cell-starvation, with 1.00 demonstrably reachable (seed 2) and transferring cross-session. The starvation mechanism is documented as the Layer-2 motivation rather than papered over.
- Closes the Exp 45 graduation-row "replication outstanding" flag.

## Method notes

- The `$S`-driven command (seed drives session + `--nac-path` + `--log` together) is the fix for the mislabel footgun that polluted an earlier seed-1 file (a `--seed` change without the path changes wrote seed-2 data into seed-1 files). The `--log` is append-mode — `rm` a seed's log before re-running or runs stack.
- Two early attempts died on `ConnectionError` (daemon/WS drops under rapid `goto`); a daemon restart cleared it. Aborted runs produce no final-probe result and are excluded.
- Gain drifts run-to-run (0.45–0.64 measured); freezing `--az-gain 0.55` keeps the boundary and the optimal-action labels stable so the magnitude scores stay comparable.
