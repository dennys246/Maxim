# Exp 62 — CONFOUNDING lens (four-lens design review, 2026-09-17)

Reviewed: `docs/experiments/exp62_pressure_interoception_prereg.md` (DRAFT v1, uncommitted on
`exp62/design-draft`). Charter: `docs/experiments/DESIGN_REVIEW.md` — does the metric isolate the
claimed cause; could a positive OR a null arise for a reason other than the claim; controls;
statistic matched to the baseline; floor. Fed by `docs/wiring/cosine-separation-is-directional.md`
(corollaries 1, 3, 4, 6, 7), `r1_cross_layout.md`, and the Exp 61 / R3 confounding files (not
re-derived here: propose-only training writes only `cluster_fear`; Wire-4 is auto-wired; a fresh
agent can learn inside an UNRESCUED dive; `recommend_action` returns `None` when nothing scores).

Verdict line at the bottom. Two DO-NOT-BUILD, five SHOULD-FIX, three NIT. **The draft's own build
step 2 (the offline replay, corollary 3) was run here on the LIVE gate-(ii) vectors, and it answers
the design before any bridge line is written.**

## Verified first (evidence, not the prereg's assertions)

**V1 — The replay, on the real encoder and the real captured vectors.** Base vectors are the
per-sensor normalized `v_safe` / `v_dark` (shore / submerged) of
`docs/experiments/data/exp60_geometry_2026-09-15b.json` — the run-authorizing gate-(ii) record,
captured LIVE at the Exp 60 pool (shore y 40, floor y 35, 69.17 blocks from world spawn, light 0,
day frozen, saturation at rest). Embedded through the SHIPPED `maxim.similarity.encoder._sensor_embed`
with unit ranges (normalized inputs) and `gain_exponent=3.0`, exactly the shape of
`exp60_saturation_rest_check.py`. Sanity: cos(shore1, sub1) reproduces the record's **0.7874**.
Pool 2 = the same submerged reading with `y_altitude` and `distance_from_spawn` re-normalized to
the candidate placement (`y/128`, `(d+128)/256`); everything else identical because the classroom
is a sealed stone shell (light 0 at any y), the day is frozen (`water_trial.GAMERULES`
`doDaylightCycle false`), and the settle guard fixes saturation/food/health. Script at the end of
this file.

Gain weights at pool-1 submerged: `is_in_water` **1.0**, `light_level` **1.0**, `time_of_day`
**0.77**, `distance_from_spawn` **0.16**, `y_altitude` **0.09**, `oxygen` 0.04. The two "absolute"
sensors together carry a quarter of the mass of ONE of the three full-weight constants.

**A) SHIPPED roster — cos(pool-1 submerged, pool-2 submerged). A MISS (the draft's arm-1 floor) needs < 0.85.**

| shore_y \ dist | 0 | 20 | 40 | 69 | **90 (bound)** | 110 | 128 (cap) |
|---|---|---|---|---|---|---|---|
| 8 | 0.905 | 0.906 | 0.907 | 0.911 | 0.910 | 0.891 | **0.839** |
| 20 | 0.977 | 0.977 | 0.978 | 0.981 | 0.977 | 0.949 | 0.880 |
| 40 (pool 1) | 0.997 | 0.997 | 0.998 | 1.000 | 0.994 | 0.962 | 0.886 |
| 64 | 0.996 | 0.996 | 0.997 | 0.999 | 0.993 | 0.960 | 0.884 |
| 100 | 0.996 | 0.996 | 0.997 | 0.999 | 0.993 | 0.960 | 0.885 |
| 120 | 0.967 | 0.967 | 0.968 | 0.969 | 0.964 | 0.934 | 0.865 |
| 128 | 0.917 | 0.917 | 0.918 | 0.920 | 0.915 | 0.891 | **0.831** |

Inside the Exp 60 spawn bound (`setup_world.WATER_MAX_DIST_FROM_SPAWN = 90`, gated live by W1)
the shipped roster **never misses** — the minimum over the whole admissible band is 0.905
(y 8, the bottom of the builder's range). The only two cells that miss need `distance_from_spawn`
AT THE 128 CAP **and** the altitude at an extreme.

**B) The live cluster-distinct preflight AT pool 2 — cos(pool-2 shore, pool-2 submerged), needs < 0.85:**
at the two missing cells above it reads **0.873** and **0.873** (> 0.85): the pool-2 shore and
submerged readings collapse into one cluster — corollary 6 exactly (the capped distance is a
full-weight constant in both situations). `WaterTrial.check_clusters_distinct` refuses the pool.
So **D1 is unsatisfiable**: there is no placement that both misses on the shipped roster and passes
the apparatus's own preflight. The premise sentence of §The claim ("the shipped roster's absolute
features put pool 2's submerged reading in a different world cluster") is false on arithmetic for
every admissible pool.

**C) The three bodies at candidate placements** (arm 1 / arm 2 / arm 4; last column = pool-2 preflight):

| pool 2 | shipped | + `pressure` 4 of [−12,12] | + `pressure` at an extreme (4 of [−4,4]) | absolutes at gain 0 | pool-2 shore/sub |
|---|---|---|---|---|---|
| y 64, dist 109 (the brief's "+24 / +40") | 0.9631 | 0.9630 | 0.9720 | **1.0000** | 0.814 |
| y 64, dist 69 | 0.9991 | 0.9991 | 0.9993 | 1.0000 | 0.787 |
| y 8, dist 0 | 0.9054 | 0.9052 | 0.9238 | 1.0000 | 0.835 |
| y 128, dist 0 | 0.9173 | 0.9172 | 0.9366 | 1.0000 | 0.823 |
| y 128, dist 128 (inadmissible) | 0.8315 | 0.8315 | 0.8653 | 1.0000 | 0.873 (refuses) |

**V2 — `pressure` as declared is silent.** Range `[−12, 12]`, rest 0: at the Exp 60 pool the eye
block sits under 4 blocks of water → v = 16/24 = 0.667 → gain weight **(0.333)³ = 0.037**. By
depth: 3 → 0.016, 5 → 0.072, 8 → 0.30, 10 → 0.58, 12 → 1.0. Adding it moves cos(pool 1, pool 2) by
< 0.001 in every row of table C. And its sign is the wrong one for the draft's story: `pressure`
reads the SAME at both pools, so it is a SHARED component — it can only RAISE the cross-pool
cosine (0.963 → 0.972 when forced to an extreme), never lower it, and it cannot separate anything
at pool 1 either (cos(pressure 4, pressure 0) at pool 1 = 0.9999; only 4 → 12 gives 0.848). The
draft's corollary-7 hedge ("one added feature moves the direction by its gain-weighted share")
is correct and the share is 0.037.

**V3 — Making `pressure` loud breaks the DV window.** Under `[−12, 12]` full weight needs a
12-deep column (`WATER_MAX_DEPTH`). The measured ascent is 1.2–1.83 s for a 4-block head rise
(`exp60_water_apparatus.json` w4 `t_surface`; Exp 61 min 1.20 / max 1.57 s) ≈ 2.2–3.3 blocks/s; an
11-block rise is ≈ 3.3–5 s, plus the 1.4–1.9 s first-call latency (Exp 61 §Outcome) → 4.7–6.9 s
against the **4.335 s** US-free cap, which is depth-independent (oxygen drains at the game's rate
regardless of depth). The behavioural DV would censor a WORKING fear. (`escape_water` also holds
jump for a hard 8 s cap — R3 lens F12.) Re-declaring the range to `[−4, 4]` to make depth 4 an
extreme is tuning the sensor to the apparatus and still does nothing for the claim (table C).

**V4 — Arm 4 is an identity, not a control.** With `y_altitude` and `distance_from_spawn` at gain 0
every remaining sensor is identical between the pools by construction (sealed shell, frozen day,
settle guard, same depth), so cos = **1.0000** in every row. Arm 4 does not ask "does removing
place do the same job as adding pressure?"; it asserts that two identical readings are one cluster.

**V5 — What CAN make pool 2 miss, and it is not place.** The full-weight constants: a LIT pool 2
(`light_level` 0 → 7) gives cos **0.785** (a miss); light 15 → 0.59; `time_of_day` 0.04 → 0.5 gives
0.892; saturation drained at placement gives **0.8499** (a miss by 0.0001 — the exact Exp 60
gate-(ii) run-1 failure shape, corollary 6). None of these is controlled by any drafted arm: a
pool 2 built with a skylight, a day-cycle drift, or a settle that leaves saturation off its rest
would manufacture a "cache wall" that the design would attribute to altitude/spawn distance.

**V6 — The first-contact read at pool 2 is clean of bookings (as the brief asks to confirm).**
`WaterTrial.placement`: rescue (loop OFF) → `loop_window` starts the loop ON THE SHORE for
`loop_warm_s` → `submerge` → 4 Hz samples → rescue-first at the cap. On the shore the loop's only
NAc write is `note_active_clusters`; `anticipatory_threat_need` on the shore node is 0 (table E:
cos(pool-1 water node, pool-2 shore) 0.73–0.79 in the admissible band — no completion into the
feared node) → `recommend_action` returns `None` → nothing executes → no link. The cap
(pain edge − 0.75 s) is US-free and `placement` marks `dirty` on any US or damage. The only
pre-read write in the window is the `flee` negative link of the tie-break (Exp 60/61's recorded
shape). The R3 finding (a fresh agent learns inside an unrescued dive) does not apply here: every
window is rescued before the pain edge. The loop-OFF representation gate at pool 2 uses the
ENCODE path (`water_trial.encode_world_cluster` → `agent_loop._encode_current_clusters` →
`SensorEncoder.encode_sensors` → `EC.pattern_complete_or_separate`), so it CREATES the pool-2 node
on a miss and increments the count on a hit — no NAc write, but see F4.

**V7 — `pressure` cannot be a route.** `grep -rn "pressure" src/maxim/decisions/nac.py
src/maxim/runtime/agent_loop.py` → nothing: it is in no `_DRIVE_TOOL_AFFINITIES` row and no
`_DRIVE_CORRECTIVE_NEEDS` pair; v1 declares no drive on it, so it can produce no need. With no
need and no links, `recommend_action` returns `None` (Exp 60 ABLATED 0/60 placements, Exp 61
arm 3 0/12 — both with the auto-wired Wire-4 detached or unfed). Arm 5 is 0/12 by construction.

**V8 — Delta gate.** `SensorEncoder._max_delta` compares RAW values against `min_delta` 0.05; a
24-block altitude move re-encodes. Not a confound.

## Findings

### DO-NOT-BUILD (as drafted)

**F1 — The floor is not a floor: on the shipped roster pool 2 pattern-completes into the trained
pool-1 node at every placement the apparatus admits (cos 0.905–1.000, table A), and the only
placements that miss fail the pool-2 cluster-distinct preflight (table B).** Failure scenario: the
campaign runs, arm 1 surfaces 12/12 at pool 2, ABOVE-WALL reads 1.0 − 1.0 = 0 and the experiment
records a NULL for `pressure` — while the actual finding (the survival body's water fear ALREADY
fires at a second pool) is buried as "the floor failed". Or worse, D1 is "set by the replay" to
the y 8 / dist 128 cell, the pool-2 preflight refuses, and the campaign never starts. Evidence:
V1 A/B, `setup_world.WATER_MAX_DIST_FROM_SPAWN`, corollary 6. The premise of §The claim is
arithmetically false; a positive in arm 2 would arise for a reason other than the claim (the
absolutes carry 0.09 + 0.16 of gain mass against three constants at 1.0/1.0/0.77 — they were
never loud enough to place-key the fear). This is the R2 shape: knowable from the mechanism,
without a live campaign.

**F2 — `pressure` as declared cannot do the claimed job, and the arm that isolates it (arm 4) is
an identity.** Failure scenario (had F1 not held): arm 2 = arm 1 to three decimals (V2), so the
arm-2 − arm-1 contrast is 0 whatever the floor; and arm 4 reads 1.0000 by construction (V4), so
"arm 4 ≥ arm 2 → the claim is written as 'relative features'" is guaranteed before anything runs.
Making `pressure` loud needs depth 12 and pushes surfacing past the 4.335 s window (V3), so the
loud version censors its own DV. The draft's "prediction to be checked" is honest about arm 2 but
assumed arm 1 — and the arithmetic falsifies the assumption, not the hedge.

### SHOULD-FIX (for any version that proceeds — see the salvage below)

**F3 — Pin the full-weight constants per pool, or a MISS is manufactured by lighting/settle, not
place.** Failure scenario: pool 2 is built one band higher, a sky-light gap in the shell reads
light 7, cos drops to 0.785, and the outcome attributes the wall to altitude. Evidence: V5. Fix:
gate `light_level == 0`, `time_of_day` equal to pool 1's recorded value, and the settle guard
(`saturation == 10`, `is_raining == 0`, `nearest_player_dist == 64`, `nearest_hostile_dist == 64`)
on the pool-2 shore AND submerged samples, exactly as Exp 60 W1 gates pool 1; record both pools'
per-sensor gain-weight tables in the fingerprint.

**F4 — The replay must run on LIVE-captured pool-2 vectors, and the REPLAY-consistency gate must
not convert a finding into a refusal.** Failure scenario: the offline replay (synthetic offsets)
predicts a miss, the live pool-2 reading hits (because a constant differed from the synthetic
assumption), and the rule "a predicted miss that hits live → INCOMPLETE" files the substantive
result (no wall) as an instrument disagreement. Evidence: the draft's step 2 uses "a synthetic
pool-2 submerged reading … shifted by the planned offsets" — the "estimated on a different
geometry" trap Exp 60 §Apparatus names; and the loop-OFF representation gate is the ENCODE path on
the same EC as the live placement (V6), so gate-vs-placement "consistency" is tautological — the
offline replay is the only independent read. Fix: build pool 2, capture 30 shore + 30 submerged
samples with `l11_geometry_probe` (the gate-(ii) shape), replay THOSE against the committed pool-1
vectors, and make disagreement between the offline-on-live-vectors replay and the live EC a named
refusal (`replay_disagrees`) while a predicted-hit-that-hits is the result, not INCOMPLETE.

**F5 — Arm 5 is structurally 0 and costs 12 trainings.** Failure scenario: 36+ min of yoked
conditioning to re-measure Exp 60 ABLATED. Evidence: V7. Fix: replace with a 1–2 row anti-vacuity
kit — one NAIVE `pressure`-body agent (no training) at pool 2, zero executor calls in the window
— which tests the same proposition ("the sensor is not itself a route") at zero training cost.
If the owner wants "exposure without fear", Exp 60's ABLATED arm already is that.

**F6 — Gate shore-fear at pool 2 explicitly (Exp 61 F4, now at a NEW shore).** Failure scenario:
a pool-2 shore reading that completes into the pool-1 water node (it does not in the admissible
band — table E 0.73–0.79 — but a lit or drained shore moves it) makes the receiver flee during the
loop-ON warm-up and surface for a non-specific reason. Fix: `cluster_fear(pool-2 shore node) == 0`
and pool-2 shore id ≠ trained id, loop OFF, before every first contact; zero calls in the warm-up.

**F7 — "Location-invariant" is an overclaim for what any version can show, and the claim must say
what "different place" means.** The DV is R1's own comparison (does the new reading clear 0.85
against the trained key), and R1 recorded that "generalizes" and "genuinely different" are that
one comparison with opposite sign. Exp 62 escapes the trap ONLY because "different" is defined by
game coordinates, not by cosine — i.e. the body's roster decides what counts as the same
situation, which is the 1.3 thesis ("portability comes from the BODY", `survival_world_1_3.md`
§World topology). Fix: write the claim as "the survival body's water fear is invariant across the
admissible classroom band (shore y 8–128, ≤ 90 blocks from spawn) because its absolute sensors
carry ≤ 0.16 gain mass against a 1.0 cue" — a property of the SHIPPED body, earned by the offline
replay plus a live confirmation, not by a new sensor; and never "anywhere" (a lit surface pool, a
far-from-spawn pool, or a drained agent all miss, V5/table A).

### NIT

**F8 — Statistic.** Fisher on per-agent binaries is matched to Exp 60/61 (12/12 v 0/12 → 3.7e-7);
Wilson on 0/12 is [0, 0.24], on 12/12 [0.76, 1]. Fine — but the arithmetic predicts 12/12 v 12/12
for arm 2 v arm 1, where no statistic helps. If the salvage runs, the informative number is the
per-agent latency at pool 2 vs Exp 61's 1.38–1.86 s call / 2.70–3.53 s air, with intervals.

**F9 — The first loop-ON placement at pool 2 books nothing before the read** (V6) — confirmed as
drafted; keep Exp 61's decision-DV + `calls == []`-is-a-refusal rules (Exp 61 F2/F7) verbatim.

**F10 — Frozen prototype vs later reads.** The trained node is the FIRST submerged observation
(oxygen ≈ 0.5, silent); later reads sit at oxygen 0.325 (w 0.04): cos 0.9998 — immaterial. The
`min_delta` gate compares raw units (V8) — immaterial.

## Answers to the brief's questions, one line each

1. *Arm 4 sufficient / a third explanation?* Arm 4 is an identity (1.0000 by construction, V4).
   The third explanation is the whole result: pool 2 lands in the trained node because the
   absolutes are near-silent, `pressure` or not (V1). `pressure` cannot create a distinct pool-1
   node (cos 0.9999 at w 0.037), so "arm 3 passes on a different node" does not arise.
2. *Cosine arithmetic?* At +24 altitude / +40 distance the shipped roster reads 0.963 (a HIT), and
   `pressure` moves it by 0.0001 (table C). Arm 2 is predicted to EQUAL arm 1, not to fail; arm 4
   is not "the real experiment", it is an identity. The hedge was honest; the floor assumption was
   not checked.
3. *Arm 1 floor?* Not a floor: min cos 0.905 in the admissible band (vs the 0.787 shore/submerged
   contrast, pool 2 submerged is FAR closer to pool 1 submerged than pool 1's own shore is). The DV
   is then unmeasurable for `pressure` and the honest outcome is "already invariant" (F1, F7).
4. *US-free window clean?* Yes (V6); the pool-2 placement books only the `flee` tie-break link.
5. *Statistic / replay gate?* Fisher matched; the replay gate pre-decides only if the replay is
   synthetic — run it on live-captured pool-2 vectors and keep disagreement a named refusal (F4).
6. *Arm 5?* Vacuous: no drive, no affinity keyword, no corrective need → `None` (V7). Kit row (F5).
7. *Secretly R1 / overclaim?* It IS R1's comparison, legitimately escaped by defining "different"
   by coordinates; "location-invariant" must be bounded to the admissible band and attributed to
   the shipped body (F7).

## Salvage (what is worth doing instead — the R1/R2 mold)

1. **Do not build `pressure` for this claim.** The claim it was built for is answered offline: no
   place-keyed wall exists inside the apparatus's bounds. If `pressure` is wanted, it is for Exp 62b
   (a drive), which needs its own design against R3's frozen baseline.
2. **Promote the replay in this file to `docs/experiments/data/exp62_replay_*.py`** (the draft's
   step 2), extended with live-captured pool-2 vectors once a pool 2 exists (F4).
3. **A 3-seed LIVE confirmation of arm 1 only** — shipped roster, train pool 1 (Exp 60 protocol),
   probe pool 2 at a placement well inside the band (e.g. shore y 64, ≤ 90 from spawn, light 0),
   loop-OFF representation gate first — ≈ 15 min. Predicted 3/3 with the trained node id at pool 2.
   That closes Exp 60/61's "fears water anywhere" §Not claimed line as a bounded, structural +
   live-confirmed result ("invariant across the admissible band; the absolutes carry < 0.17 mass"),
   ships as a null-shaped record in the R1 mold, and adds a `docs/wiring/` entry: *a place sensor
   only place-keys a fear when it sits near an extreme; near its rest it is silent by design.*
4. **If a real wall is wanted to test a remedy against**, the lever is a full-weight constant that
   differs between pools — a LIT surface pool (cos 0.785, V5) — and then the claim is about
   `light_level`, and the remedy is a body question (which constants belong in the world
   channel), not a pressure question.

## What I verified

- `docs/experiments/DESIGN_REVIEW.md`; the Exp 62 draft in full; `cosine-separation-is-directional.md`
  (all seven corollaries); `r1_cross_layout.md`; `minecraft_benchmark.md` §R1 pointers;
  `survival_world_1_3.md` §World topology; Exp 60 §Apparatus, §Gate (ii), §Outcome (incl. caveat +
  not-claimed); Exp 61 §Not claimed, §Outcome; the Exp 61 and R3 confounding files (V1–V9 there).
- Code: `similarity/encoder.py` (`_stable_basis`, `_normalize_value`, `_sensor_embed`'s gain
  `(|v−0.5|·2)^p`, `sensor_geometry_fields`, `encode_sensors` incl. the D2 zero-vector branch and
  `_max_delta` raw units); `similarity/ec.py::pattern_complete_or_separate` (best-match ≥
  threshold, frozen centroid for `world`, first observation is the prototype, ENCODE path);
  `hivemind/merge.py::SENSOR_MODALITY_THRESHOLDS["world"] = 0.85`; `runtime/agent_loop.py::
  _encode_current_clusters`; `_data/components/bodies/minecraft_player.yaml` (every world range,
  rest, gain; `escape_water`/`flee`; the oxygen-drive comment on innate reflexes);
  `scripts/survival_world/water_trial.py` (`GAMERULES`, `rescue` + settle guard, `submerge`,
  `encode_world_cluster`, `check_clusters_distinct`, `loop_window` rescue-first, `placement` dirty
  rule, `probe`, `train` propose-only, `live_g2`); `scripts/survival_world/setup_world.py`
  (`WATER_SHORE_Y`, `WATER_MIN/MAX_DEPTH` 3–12, `WATER_MAX_DIST_FROM_SPAWN` 90,
  `water_classroom_geometry`); `grep pressure` over `decisions/nac.py` + `runtime/agent_loop.py`.
- Data: `exp60_geometry_2026-09-15b.json` (the live per-sensor vectors, cos 0.7874 reproduced),
  `exp60_geometry_2026-09-15.json` (the saturation-extreme 0.8502 failure), `l11_geometry_2026-09-15.json`,
  `exp60_water_apparatus.json` (`t_surface` 1.665, depth 5, distance 69.17, damage onset 16.2 s),
  `exp60_spawn_distance_check.py` / `exp60_saturation_rest_check.py` (replay shape reused).

**Not verified (other lenses / owner):** whether a sealed shell at shore y 8 or 128 is buildable
(bedrock / build limit on Paper 1.20.4) — table A's extreme rows are arithmetic, not placements;
the bridge's `blockAt` eye-column transform for `pressure` (moot under F1/F2); anything about a
`pressure` DRIVE (Exp 62b); the ascent rate at depths > 5 (V3 extrapolates from the 4-block
measurement).

## Verdict

**DO-NOT-BUILD (as drafted).** The experiment's floor does not exist inside its own apparatus
bounds (F1), the sensor it adds is silent at the declared range and cannot lower a cross-pool
cosine in principle (F2), and its isolating control is an identity (F2). None of this is fixable by
n, arms or statistic; it is the geometry. The salvage is cheap and positive: promote this replay,
run a 3-seed live arm-1 confirmation, and close "fears water anywhere" as a bounded structural
result of the shipped body — no `pressure` sensor required for that claim.

## Appendix — the replay (run from the repo root with `PYTHONPATH=src`)

```python
"""Exp 62 confounding-lens replay: pool-1 submerged (LIVE gate-(ii) vectors) vs synthetic pool-2."""
import json, math, sys
sys.path.insert(0, "src")
from maxim.similarity.encoder import _sensor_embed
P = 3.0
rec = json.load(open("docs/experiments/data/exp60_geometry_2026-09-15b.json"))
SUB1 = {s["sensor"]: s["v_dark"] for s in rec["per_sensor"]}   # normalized, pool-1 submerged (live)
SHORE1 = {s["sensor"]: s["v_safe"] for s in rec["per_sensor"]}

def embed(vmap, drop=()):
    vm = {k: v for k, v in vmap.items() if k not in drop}
    return _sensor_embed(vm, ranges={k: (0.0, 1.0) for k in vm}, gain_exponent=P)

def cos(a, b):
    d = sum(x*y for x, y in zip(a, b)); na = math.sqrt(sum(x*x for x in a)); nb = math.sqrt(sum(y*y for y in b))
    return d/(na*nb) if na and nb else 0.0

def w(v): return (abs(v-0.5)*2)**P
def y_v(y): return y/128.0
def d_v(d): return (d+128)/256.0
def press_v(depth, R=12): return (depth+R)/(2*R)
y1_shore, d1 = 40, 69.17

def pool(y_shore, dist, *, submerged, pressure=None, R=12, depth=5):
    base = dict(SUB1 if submerged else SHORE1)
    base["y_altitude"] = y_v(y_shore - depth if submerged else y_shore)
    base["distance_from_spawn"] = d_v(dist)
    if pressure is not None:
        base["pressure"] = press_v(pressure if submerged else 0, R)
    return base

assert abs(cos(embed(SHORE1), embed(SUB1)) - 0.7874) < 5e-4          # reproduces the gate-(ii) record
print({k: round(w(v), 4) for k, v in SUB1.items() if w(v) > 0})       # per-sensor gain weights at pool 1
print({d: round(w(press_v(d)), 4) for d in (1, 2, 3, 4, 5, 8, 10, 12)})  # pressure weight by depth
for ys in (8, 20, 40, 64, 80, 100, 120, 128):                          # table A / B
    print(ys, [round(cos(embed(pool(y1_shore, d1, submerged=True)), embed(pool(ys, d, submerged=True))), 3)
               for d in (0, 20, 40, 69, 90, 110, 128)],
              [round(cos(embed(pool(ys, d, submerged=False)), embed(pool(ys, d, submerged=True))), 3)
               for d in (0, 20, 40, 69, 90, 110, 128)])
for ys, d in ((64, 109), (64, 69), (8, 0), (128, 0), (128, 128)):     # table C
    s1 = pool(y1_shore, d1, submerged=True); s2 = pool(ys, d, submerged=True)
    print(ys, d, round(cos(embed(s1), embed(s2)), 4),
          round(cos(embed(pool(y1_shore, d1, submerged=True, pressure=4)), embed(pool(ys, d, submerged=True, pressure=4))), 4),
          round(cos(embed(pool(y1_shore, d1, submerged=True, pressure=4, R=4)), embed(pool(ys, d, submerged=True, pressure=4, R=4))), 4),
          round(cos(embed(s1, drop=("y_altitude", "distance_from_spawn")), embed(s2, drop=("y_altitude", "distance_from_spawn"))), 4))
for label, mut in (("light 7", {"light_level": 7/15}), ("noon", {"time_of_day": 0.5}), ("drained", {"saturation": 0.0})):  # V5
    p2 = pool(40, 69, submerged=True); p2.update(mut)
    print(label, round(cos(embed(pool(y1_shore, d1, submerged=True)), embed(p2)), 4))
```
