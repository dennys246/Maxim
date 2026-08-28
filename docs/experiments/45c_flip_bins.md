# Exp 45c — does a DERIVED bin boundary take magnitude from 0.75 to 1.00?

**Status:** **PASS** (hardware, 2026-07-16). **Magnitude 1.00** — the pre-registered
prediction, hit exactly and held for 13 consecutive probes (trials 40→100).
**Plan:** [orient_magnitude_learning.md](../plans/deferred/orient_magnitude_learning.md) S1.
**Parent:** [Exp 45b](45b_orient_magnitude.md) (PASS: direction 1.00, magnitude 0.75).

## Claim under test

Exp 45b's magnitude ceiling of 0.75 was **not** an exploration or credit limitation —
it was a **state-representation** one: the `near` bin straddles the point where the
correct action magnitude changes, so it holds two opposite correct answers. Placing
the boundary where the physics puts it should lift magnitude to **1.00**.

## The derivation (this is the whole experiment)

A correct-direction step of shift `S = |delta| * gain` takes `|az| → |az − S|`, so its
relief is `az − |az − S|`. Step A beats step B exactly when `|az − S_A| < |az − S_B|` —
i.e. when **az is nearer A's shift**. So the boundary is their midpoint:

    az_boundary = gain * (|delta_big| + |delta_normal|) / 2
                = 0.546 * (0.9 + 0.3) / 2
                = 0.328

The optimal magnitude policy is **nearest-neighbour quantization of |az| onto the
available shifts**. `az_bin`'s legacy `near` = 0.1–0.5 straddles 0.328; `--flip-bins`
sets the boundary to the derived value and widens placements to near [0.16, 0.27] /
far [0.39, 0.80] (the old |az| ≤ 0.65 cap was chasing a head-bug artifact; the
post-headfix sweep is monotonic and linear to |az| ≈ 0.87).

**The boundary is derived per robot from its own measured gain and its own declared
action magnitudes** — not a constant. That is also the strongest surviving argument for
S2 (a robot that calibrates its gain derives its own state boundaries).

## Sim validation (6 seeds, gain 0.55, stationary source, 100 trials, eps 0.5)

| arm | magnitude per seed | mean |
|---|---|---|
| legacy boundary 0.5 | 0.25, 0.50, 0.25, 0.50, 0.50, 0.25 | **0.375** |
| derived boundary 0.328 (`--flip-bins`) | 0.75, 1.00, 1.00, 0.75, 1.00, 1.00 | **0.92** |

Direction stays 1.00 in both arms.

**A sim-fidelity bug was fixed to get this**: `DryRig` teleported the source
(`jump_prob=0.04` per *read*, ~10 reads/trial → ~1-in-3 trials disturbed). That models an
operator relocating the source — correct for manual mode, pure noise in `--perturb`
mode where the source is fixed and the robot moves. It had been degrading every sim
sweep, and it **masked this effect**: with the phantom noise, legacy scored 0.70 and the
derived boundary 0.80 (no clear separation); with honest physics, 0.375 vs 0.92. Noise
was hiding a systematic conflict.

## Pre-registered verdicts

- **PASS** — magnitude **1.00** (direction stays 1.00). The 45b ceiling was
  representational; the boundary was the limiter.
- **PARTIAL** — magnitude 0.75, unchanged from 45b. The boundary was NOT the limiter →
  next suspect is exploration/accumulation (`update_cluster_reward` accumulates rather
  than averages, so sample count competes with mean reward — the Exp 41 fixation
  dynamic). Diagnostic: a bin whose better action has bias ≈ 0.0 was never sampled.
- **FAIL** — magnitude < 0.75 or direction < 1.00 (regression; the narrower near bin has
  lower relief magnitudes, so a real-noise SNR problem would show here).

## Protocol

```bash
~/Envs/maxim-env/bin/python scripts/orient_backbone/live_3_learn.py \
    --host <ip> --perturb --flip-bins --session flip1 --fresh \
    --epsilon 0.5 --trials 100 --nac-path ~/.maxim/orient_live/nac_reachy_flip.json
```
~30 min. Prerequisites: daemon healthy (`backend_status.ready: true`), head-frame fix in
place (`yaw_verify.py` → `d(head)/d(body)` ≈ +1.0), continuous speech source in front.

**Note:** this changes the STATE SPACE, so the resulting policy is not comparable
bin-for-bin with a 45b NAc (same bin names, different az ranges). Fresh NAc required;
queen-mind bundles from 45b and 45c are different substrate generations.

## Results (flip1, 2026-07-16): **PASS — magnitude 1.00**

| metric | pre-registered | measured |
|---|---|---|
| direction correctness | 1.00 | **1.00** |
| magnitude appropriateness | **1.00** (PASS bar) | **1.00**, stable trials 40→100 (13 probes) |
| derived boundary | — | 0.330 (from live gain 0.55; metric gain settled 0.563) |
| apparatus gain | — | 0.56/rad — matches sweep 0.546-0.578 and mag2's 0.58 |

Every bin learned the derived-optimal action, with decisive margins:

| bin | argmax | optimal | winning bias | runner-up |
|---|---|---|---|---|
| far_left | `turn_left_big` | ✓ | **+0.870** | turn_left +0.053 |
| near_left | `turn_left` | ✓ | **+0.250** | turn_left_big −0.060 |
| near_right | `turn_right` | ✓ | **+0.352** | turn_right_big −0.570 |
| far_right | `turn_right_big` | ✓ | **+1.000** (clamped) | turn_right +0.080 |

**Learning curve: greedy turned-toward 0.286 → 1.000.** It began *below chance* — the
untrained substrate was actively wrong — and converged to perfect. The strongest
learned-vs-servo separation in the orient line to date.

**The near bins are the claim.** The substrate did not merely prefer the small step; it
learned the big one is *harmful* there (near_right: normal +0.352 vs big **−0.570**).
Trial 78 is the mechanism in one line: `near_right turn_left_big → az +0.16 → +0.69,
relief −0.533` — tried, punished, abandoned.

**Verdict: PASS.** Exp 45b's 0.75 ceiling was **representational, not motivational** —
the bin straddled the decision boundary. Fixing the state representation (2 derived
constants, no new mechanism, no effort cost, no exploration change) took magnitude
0.75 → 1.00. Sim predicted 0.92 (4/6 seeds perfect); hardware delivered 1.00.

### What made this reachable

Every link was load-bearing and none was available before the head fix:
1. `head=None` counter-rotated the mics → six phantom sensor hypotheses.
2. Head fixed → DoA revealed as near-linear (R²=0.998) → **honest gain 0.55**.
3. Honest gain → the boundary becomes **derivable**: gain*(|Δbig|+|Δnormal|)/2 = 0.33.
4. Derived boundary → no bin straddles it → **magnitude 1.00**.

A contaminated gain cannot produce a correct boundary; a wrong boundary caps magnitude
at 0.75 no matter how much exploration or credit you throw at it.

**Data:** [data/45c_flip_bins_sessions.jsonl](data/45c_flip_bins_sessions.jsonl).
**Substrate:** `~/.maxim/orient_live/nac_reachy_flip.json` (queen-mind v0.3 candidate —
a different state generation from 45b's; not bin-comparable).

## Method note

Found while implementing: the learner's own `az_bin` call was left on the legacy
boundary while the apparatus and metric used the derived one — a silent frame mismatch
that made `near` bins see |az| up to 0.49 against a 0.33 boundary. Caught by inspecting
what each bin *actually saw* rather than trusting the config. Same class as the
head-frame bug: **two components disagreeing about a frame, silently.**
