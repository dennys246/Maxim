# Set-point-aware neutral — CONFOUNDING lens (four-lens design review, 2026-09-16)

**Reviewer lens:** does the proposed VALIDATION isolate the claimed cause? Could a positive OR a null
arise for a reason other than "set-point-aware neutral makes small off-rest moves rotate the
embedding"? Right controls, a statistic matched to the baseline, no hand-built demo passing while the
real thing fails (D43/D44).
**Under review:** `docs/plans/setpoint_aware_neutral.md` (the L11 line's "B"), specifically its
§Approach bullet "Diagnostic-first" (the offline replay) and open question 2.
**Read-only except this file.** Every number below was recomputed in this session from the committed
records through the production `_stable_basis` / `_sensor_embed`; the scripts are inline in the
findings so the fold can rerun them.

**Verdict line: DO-NOT-BUILD as written — FIX-THEN-BUILD is available, but only after the
pre-registration in DNB-1 is frozen and the range-fix control in DNB-2 has been run FIRST.**

---

## Verified first

**The primitive the plan would change.** `similarity/encoder.py::_sensor_embed` weights each sensor
`w = (|v − 0.5|·2)^p` (p = `SensorEncoderConfig.gain_exponent` = 3.0, world only) and sums
`w·((1−v)·basis_low + v·basis_high)`. Its docstring carries decision D1 ("no set-point plumbing …
an unmeasured set-point-aware variant must not be improvised"). `SensorEncoder.encode_sensors`
implements D2: a gained call whose embedding is the zero vector returns `None` and records a
"designed rest" — **a situation at which every sensor sits at its neutral has NO cluster**. This is
load-bearing for DNB-3.

**The set-point source the plan names.** `initial:` is read by `embodiment/spec.py` to seed
`entity.vital_metrics` — it is a SPAWN value, not a measured rest. On the current body
(`_data/components/bodies/minecraft_player.yaml`, 17 `modality: world` sensors after #719 added
`is_in_water`) `initial` normalizes to exactly the range midpoint 0.5 for **15 of 17** sensors, by
design — every range comment reads "rest = midpoint = A4-neutral". The only two where
`initial ≠ midpoint` are `light_level` (7/15 = 0.467) and `time_of_day` (0.0). So on this body a
"YAML-initial set-point" is a change to two sensors — the two constant diluters Slice-1 named.

**The captured vectors that exist** (what the plan calls "the real captured vectors we already have"):

| record | real? | what is persisted | roster / ranges at capture | replayable byte-for-byte today? |
|---|---|---|---|---|
| `docs/experiments/data/l11_geometry_2026-09-15.json` (Slice-1 = the Exp 58 safe/dark vectors) | bridge-captured, 30+30 consecutive settled samples | **per-sensor MEANS of NORMALIZED values only** (`per_sensor[].v_safe/v_dark`); no raw rows; `provenance.world_ranges` is **absent** (the probe's writer at `scripts/survival_world/l11_geometry_probe.py::analyze` drops it) | 16 sensors at code `de7e343f`, `saturation` range `[0,10]`, no `is_in_water` | **NO.** Raw values cannot be recovered from the record; under the current `[0,20]` saturation range the same raw readings normalize differently (5/10 = 0.5 silent → 5/20 = 0.25, w = 0.125 loud), and the roster differs |
| `docs/experiments/data/exp60_geometry_2026-09-15.json` / `_15b.json` (shore/submerged) | bridge-captured, 30+30 | same shape, means only, 17 sensors | `_15`: saturation `[0,10]` (cos 0.8502); `_15b`: `[0,20]` (cos 0.7874) | same limitation; `_15b` is the current-body case |
| `docs/experiments/data/l11_world_trace_2026-09-04.jsonl` (the L11 remeasure) | **the only REAL raw trace**: 1,070 deduped snapshots, 159 onsets, one 10-min night session, one world | raw bridge states + events | 16 sensors at `b6549a7b`; `saturation` `[0,10]` at capture, but `scripts/l11_real_trace_remeasure.py::_declared_world_ranges` **re-reads the body YAML at run time** | yes for the raw values; the NUMBERS move with every YAML edit |
| `docs/experiments/data/encoding_bakeoff_*.json` (the bake-off) | **synthetic** (`s0…sN`, uncorrelated SHA bases, rest ~ U(0.30, 0.70) drawn per trial, iid 2 % jitter) | summary rows | no body, no set-point concept; its `_embed` is a MIRROR with unsalted `f"{name}:low"` bases, not production's `salt="low"` | n/a |

**The existing offline checks and what each computes.** All five (`l11_slice2_cosine_check.py`,
`exp60_oxygen_separation_check.py`, `exp60_oxygen_window_check.py`, `exp60_spawn_distance_check.py`,
`exp60_saturation_rest_check.py`) take the Slice-1 (or reconstructed Exp 60) MEAN vector as a base,
hand-set one to three sensor values, embed through the production bases (`_stable_basis` or
`_sensor_embed`), and print one cosine against the fixed 0.85 threshold. Every one of them uses a
`cos()` helper that returns **0.0 when an operand has zero norm** — i.e. prints "SEPARATES" for a
zero vector. They are semi-synthetic: a real base plus hand-set perturbations. I re-ran
`l11_slice2_cosine_check.py`: 0.9766 / 0.9959 / 0.9913 / 0.9920 / 0.9911, matching the committed text.

**What made the Slice-2 replay decisive** (`docs/plans/l11_slice2_channel_split.md` §status,
`docs/experiments/rationale/l11-slice2/`): (1) the remedy had **zero free parameters** — channel
membership was written in the plan before the number was computed; (2) the question was one number
vs a fixed threshold; (3) a sanity anchor (reconstructed 0.9766 vs the probe's 0.9767) proved the
replay reproduced production; (4) every candidate landed on the WRONG side by a wide margin. A search
was not possible, so the result could not be fitted. **The set-point plan's replay has none of these
properties**: the form is free ("exact form is the measured question"), the set-point source is free
(`setpoint`/`rest`/`initial`), the steepness `k` is free, and the vectors it tunes on are the vectors
it validates on.

**The ledger's status quo** (`docs/limits/l11_sensor_dilution.md`, 2026-09-15 row): A4 separation
**0.0881**, stability **1.0**, discrimination **0.6852** on the 09-04 trace at HEAD, vs 0.0566 /
0.9984 / 0.6309 as committed on 09-04 (`l11_remeasure_verdict_2026-09-04.json`). `git log` shows
**no committed record for the 09-15 re-run** — those numbers exist only in the ledger prose. The
ledger's own reading: "the delta is that one sensor [`saturation`], moving favourably." That row is
the confound in the flesh: a 56 % relative gain in separation with **no mechanism change**.

**My replays on the Exp 58 vectors** (Slice-1 means; weight-only set-point form
`w = (min(1, |v − s|·k))^p`, p = 3; `k = 2` reduces exactly to today when `s = 0.5`):

| variant | k | cos(safe, dark) | note |
|---|---|---|---|
| status quo midpoint A4 | 2 | **0.9766** | matches the record |
| set-point := YAML `initial` (light 0.467, time 0.0, rest 0.5) | 2 / 3 / 4 / 6 / 8 / 12 | 0.9593 / 0.9900 / 0.9826 / 0.9712 / 0.9319 / 0.8666 | never clears; **non-monotone in k** |
| same, with the basis mix re-centred so v = s maps to 0.5 | 2 / 4 / 8 | 0.9593 / 0.9826 / 0.9319 | re-centring changes nothing here |
| set-point := measured SAFE rest, all sensors | 2 | **|safe| = 0 → cos undefined** | safe is the zero vector; `encode_sensors` returns `None` (D2); the shared `cos()` helper prints 0.0 = "SEPARATES" |
| set-point := measured safe rest for all sensors EXCEPT `nearest_hostile_dist` | 2 / 4 | 0.9918 / 0.9878 | WORSE than status quo |
| set-point := measured safe rest for `light_level` + `time_of_day` only | 2 | 0.9911 | = the Slice-2 "rangefix" arm, already rejected |
| leave-one-in, set-point at safe rest for a single sensor | 2 | only `light_level` (0.9477) and `time_of_day` (0.9695) move the number; `nearest_hostile_dist` → **0.9996** (worse) | the plan's headline "small one-sided mover" gets LESS visible under a rest-anchored weight, not more |
| **fitted**: safe-rest set-points on light, time, `y_altitude`, `distance_from_spawn` | 2 / 4 / 6 / **8** / 12 | 0.9918 / 0.9878 / 0.9465 / **0.7947** / 0.6794 | **a search finds a pass** — carried by silencing the classroom's SITE constants plus a steep k, i.e. place-fear fitted to the pit |

Analytic form of the claim once a mover is absent at rest and the background B is constant in both
situations: cos = 1/√(1 + (w/|B|)²), so separation ⇔ **w_mover / |B_background| > 0.620**. The claim
is therefore not "direction" in this regime — it is the mover's set-point-weighted mass against the
residual background mass, and `k` scales the numerator directly. Any single case can be made to pass
by choosing `k`.

---

## DO-NOT-BUILD

### DNB-1 — The validation is a search, not a test: form, set-point source and `k` are all chosen on the vectors they are validated on

**Finding.** The plan's "exact form is the measured question" + "REPLAY … on the real captured
vectors we already have … show it (a) separates the cases the midpoint gain can't" describes
selecting the variant on the Slice-1/Exp 58 vectors and then reporting its separation on the same
vectors. That is fitting. The table above shows the degrees of freedom are sufficient to manufacture
a pass: a YAML-initial set-point at the identity `k = 2` fails (0.9593); set-points at the measured
safe rest of four site sensors at `k = 8` passes (0.7947). Nothing in the plan freezes any of source,
form, `k`, or threshold before the vectors are touched, and the `k`-curve is non-monotone (0.9593 →
0.9900 → 0.9826 …), so a threshold crossing at one `k` is noise-shaped, not a signal.

**Evidence.** `docs/plans/setpoint_aware_neutral.md` §Approach ("and/or … Exact form is the measured
question"), §Build order step 2 ("NOMINATES"); the replay table above; the D43/D44 lesson in
CLAUDE.md ("a measurement of a possibility, presented as proof it happened").

**Concrete change — the pre-registration, frozen on `main` BEFORE any replay touches a test vector:**

1. **One form, one source, one `k`.** Weight-only: `w = (min(1, |v − s|·2))^p`, `p` unchanged (3.0),
   **`k = 2` fixed** — the only value that reduces byte-identically to today at `s = 0.5`, so the
   identity control in DNB-2 is structural. No basis re-centring in v1 (it changed nothing above and
   doubles the search space). Set-point source: an **explicit YAML `setpoint:` key only** — never
   inferred from `initial:` (a spawn value; `light_level: 7` is invented; `saturation: 5` was wrong)
   and never inferred from a running mean in v1 (an adaptive set-point fits itself to whatever rest
   it sees — that is a second mechanism with its own review; see DNB-2 on why the plan conflates the
   two).
2. **Train / test split by RECORD, not by re-sampling.**
   - *Form-selection / exploratory (allowed to be searched, never cited for a verdict):* the
     bake-off (after SF-1) and the **first half by time** of `l11_world_trace_2026-09-04.jsonl`.
   - *Test (touched once, after the freeze):* the **second half** of the 09-04 trace through
     `l11_real_trace_remeasure.py::analyze` with ranges PINNED to a named commit; the Exp 60 `_15b`
     shore/submerged pair (positive control, must stay < 0.85 AND fresh-EC distinct); a **fresh**
     Exp 58 safe/dark capture under the current 17-sensor body (the 09-15 means are not replayable —
     SF-2 — and were used to motivate the mechanism, so they are contaminated as a test);
     held-out BODIES: every bundled body that declares no `setpoint:` (byte-identity, SF-1/DNB-2).
3. **Frozen thresholds.** For each test pair: cos < 0.85 AND fresh-EC ids distinct for ≥ 30/30
   samples per situation AND neither situation encodes to the zero vector (DNB-3). For the trace:
   PRIMARY ≥ the COMMITTED status-quo PRIMARY (SF-2) with stability ≥ 0.99 and economy reported. A
   miss on any test item is a NULL for the mechanism; no re-picking `k`.
4. **Attribution.** Any pass must be decomposed per sensor (the probe already computes gain-weighted
   contributions) and the pre-registered discriminator must be the dominant contributor — a pass
   carried by `light_level`/`time_of_day`/`y_altitude`/`distance_from_spawn` is a place/apparatus
   pass and FAILS (the fitted row above is exactly that shape).

### DNB-2 — On this body a declared set-point is a two-sensor range re-declaration in a mechanism costume; the replay as designed cannot tell "mechanism works" from "one range was mis-declared"

**Finding.** The range-recentring hack the ledger already uses ("ranges re-centered so REST sits at
the A4 neutral") IS a static set-point declaration by other means: declaring `[lo, hi]` so that
rest = midpoint is identical, for A4, to declaring `setpoint = rest`. The body has already been
recentred for 15/17 sensors, so a static set-point mechanism has **nothing left to change on this
body except `light_level` and `time_of_day`** — and any "separation gain" it shows on the Exp 58 or
09-04 vectors is indistinguishable from re-declaring those two ranges with zero substrate code. The
ledger's 09-15 row proves the confound is live: `saturation` alone moved separation 0.0566 → 0.0881
(and Exp 60's gate 0.8502 → 0.7874) with no mechanism. The plan then says the mechanism fixes
premise 1 — the small one-sided `nearest_hostile_dist` move — but a static set-point CANNOT touch
that case: its YAML rest (64 = cap = midpoint) is already the neutral, and the 0.179 "rest" at the
safe site is an apparatus artifact (a forceloaded hostile ~23 blocks away at "safe"), not a body
set-point. The only set-point that rescues premise 1 is a **site-adapted** one (habituation to the
safe chamber's readings) — a different, stateful mechanism (per-agent running rest, time constant,
persistence) that the plan never names, and which is intrinsically self-fitting (it adapts to the
very rest it is then tested against — the "set-point := measured safe rest" rows).

**Evidence.** YAML range comments ("rest = midpoint = A4-neutral") on 15/17 sensors; leave-one-in
table (only light/time move the number; hostile_dist gets WORSE at 0.9996); the ledger 09-15 row +
`exp60_saturation_rest_check.py`; Slice-2 bio-faithful lens ("even set-point neutrality would not
help here, because the contrast location is itself dark — the apparatus, not the encoder").

**Concrete change.** (a) The plan must state WHICH mechanism it proposes — *static declared set-point*
(which on a recentred body is the shipped range hack and is a plumbing nicety, not a separation fix)
or *adaptive/habituating set-point* (a new stateful mechanism needing its own front-gate scope
argument and its own confound analysis: what window it adapts over, and that the test window is
disjoint from the adaptation window). (b) **Run the range-fix control FIRST, before any code**:
re-declare `light_level` and `time_of_day` so their measured rest is the midpoint (the same hack
already applied to `saturation`), replay Exp 58 + 09-04 + Exp 60 `_15b` — that number is the bar the
mechanism must BEAT, not merely reach. If the mechanism only reproduces the range-fix numbers, the
honest outcome is "no mechanism needed; declare ranges correctly." (c) Include at least one test case
where a set-point genuinely cannot be expressed as a range — a sensor whose rest is asymmetric with
both tails meaningful (e.g. `light_level` rest 15 at the surface, 0 underground, where going darker
AND lighter both matter) — otherwise the mechanism has no case that distinguishes it from a YAML
edit.

### DNB-3 — The natural set-point source makes the rest situation the ZERO vector, and every existing replay helper reports that as "SEPARATES"

**Finding.** "A sensor at its set-point contributes nothing" + "the rest situation is, by definition,
every sensor at its set-point" ⇒ the rest situation embeds to the zero vector. Today that only
happens when a body sits exactly at every range midpoint (rare, and D2 already handles it); under a
set-point anchored at measured rest it happens **by construction** for the baseline of every
contrast. Consequences for the validation: (i) `cos(0, x)` is undefined — every `cos()` helper in
`docs/experiments/data/*_check.py` returns 0.0 and prints "SEPARATES", a vacuous positive of exactly
the D43 shape; (ii) `encode_sensors` returns `None` for the rest situation, so "fresh-EC ids
distinct" cannot be evaluated (one side has no id) and the bake-off / remeasure **stability** leg
(jitter must complete onto the SAME cluster as rest) has no rest cluster to complete onto; (iii) the
Exp 58 gates G2/G3 and the R2 credit path all key on cluster ids of the resting state. The plan's
composite `min(sep, stab, disc)` would be computed over a degenerate baseline and could read as a
pass.

**Evidence.** Row "set-point := measured safe rest, all sensors": |safe| = 0.0000, |dark| = 0.1319;
`encoder.py::SensorEncoder.encode_sensors` D2 branch (`all(x == 0.0 …) → return None`); the `cos()`
definition in `l11_slice2_cosine_check.py` et al. (`return d / (na * nb) if na and nb else 0.0`).

**Concrete change.** The replay harness must (a) classify a zero-norm operand as its own outcome
(`rest-zero`, the way the remeasure has `refuted-blind`) that can never count toward separation;
(b) require that BOTH situations receive a fresh-EC cluster id before any pair is scored; (c) state
in the prereg whether "rest has no cluster" is an intended property of the mechanism — if it is, the
bio-faithful and wiring lenses must rule on what Wire-4 fear-read, G2/G3 and R2 do with a `None`
baseline, because the confound here is that the validation metric silently rewards it.

---

## SHOULD-FIX

### SF-1 — "Does not regress the bake-off" is either vacuous or undefined as the bake-off stands

The bake-off (`scripts/encoding_bakeoff.py::run_arm`) draws each trial's rest from U(0.30, 0.70) and
has no set-point concept. A set-point arm must define `s`: `s := 0.5` reproduces A4 bit-for-bit (a
non-regression that measures nothing); `s := the trial's rest` makes every rest state the zero
vector (DNB-3: separation vacuous, stability undefined). And its `_embed` mirrors production with
different bases (`_stable_basis(f"{name}:low")` unsalted vs production `salt="low"`), so a variant
added there is measured on a mirror, contrary to the plan's own "through the real encoder" rule.
**Fix:** either define a bake-off arm with declared per-sensor set-points drawn ≠ midpoint AND
background sensors resting OFF their set-points (so rest is non-zero and the w/|B| condition is
actually exercised), or drop the bake-off from the validation claim and rely on the real trace.

### SF-2 — The baseline is not pinned: the status-quo numbers move with the YAML, the 09-15 numbers are uncommitted, and the geometry records cannot be replayed byte-for-byte

`l11_real_trace_remeasure.py::_declared_world_ranges` reads the live body YAML, so "does not regress
the remeasure" compares against a moving target (it already moved 09-04 → 09-15). The 09-15 numbers
(0.0881 / 1.0 / 0.6852) have no committed record. The geometry probe persists only normalized means
and omits `world_ranges`, so the Slice-1 and Exp 60 records cannot be re-normalized under the current
roster (`is_in_water` absent; `saturation` `[0,10]` vs `[0,20]`) and their raw readings are
unrecoverable without a checkout at `de7e343f`. **Fix (before any replay):** commit a HEAD baseline
record for the 09-04 trace with the ranges it used embedded in the record; change the probe to
persist raw per-sample rows + ranges + roster + code hash (the remeasure trace already does this
right); re-capture Exp 58 safe/dark under the current body before it is used as a test item. A
replay whose baseline cannot be reproduced is not a control.

### SF-3 — The composite and its sample are not matched to the claim

`min(sep, stab, disc)` was frozen for a *many-sensor, N-sweep* question (which arm survives a big
body), not for "does a small off-rest move separate." On the 09-04 trace separation sits at the floor
(0.0881): any variant that allocates more clusters raises separation AND discrimination
(different-kind onsets landing on different ids rises with churn), so the composite can rise from
cluster proliferation alone; only the stability leg resists, and it is gated on RAW delta ≤ 0.10
which under a steep `k` still maps to a loud weight ((0.10·8)³ = 0.51). The samples are not
independent: one 10-minute session, consecutive 0.5 s snapshots for the 642 quiet pairs, and 30
CONSECUTIVE settled samples per situation in the geometry records — effective n ≈ 1 session per
record, so "stability" has no between-session estimate at all. **Fix:** (a) pre-register the claim's
own statistic — the per-pair w_mover/|B| condition and the cluster-id outcome — alongside the
composite, not instead of it; (b) require stability ≥ status quo (1.0 on this trace) as a hard leg,
economy reported and capped (a variant that doubles clusters per 100 snapshots is a different
substrate); (c) time-split the trace (tune on half, test on half) and require at least one second
independent capture (a different night / different world seed) before any PRIMARY delta is cited.

### SF-4 — The Exp 58 dark/safe re-test is apparatus-confounded in BOTH directions; as framed it is a post-hoc target, not an honest close-out

The Exp 58 vectors motivated the mechanism, so they cannot also validate it (DNB-1). Independently,
the classroom cannot express dark=danger on any sensor: `light_level` = 0 in both situations (the
"safe" chamber at y = 40 is underground), and `nearest_hostile_dist` reads 0.179 at "safe" because a
hostile is forceloaded within range there. A **null** is uninterpretable (apparatus or mechanism); a
**positive** at a swept `k` is carried by the site coordinates (the fitted row: light, time, y,
spawn), i.e. place-fear. **Fix:** either (a) drop the Exp 58 re-test from the mechanism's validation
and test on a DESIGNED contrast where the cue is a real off-rest move on the pre-registered
discriminator (surface-lit safe vs dark, or hostile-far vs hostile-near with `light_level` matched),
or (b) keep it only under DNB-1's terms: fresh capture, frozen form/`k`/threshold, per-sensor
attribution with `nearest_hostile_dist` dominant, and a pre-declared statement that a null here says
nothing about the mechanism. "The honest re-test of the blocked claim" is only honest if the claim
can be false for the right reason.

---

## NIT

- **N-1 — Inventory honesty.** The plan's "Slice-1 trace + the L11 remeasure traces" is one real raw
  trace plus three mean-vector records; say so, and say which is train and which is test.
- **N-2 — The shared `cos()` helper is a trap.** Every `docs/experiments/data/*_check.py` copies a
  helper that returns 0.0 on a zero-norm operand. A new replay copied from them inherits DNB-3's
  vacuous pass; make the helper raise or return NaN and print `rest-zero`.
- **N-3 — Report curves, not points.** If `k` (or any parameter) is swept exploratively, the whole
  curve goes in the record; the YAML-initial curve above (0.9593, 0.9900, 0.9826, 0.9712, 0.9319,
  0.8666) shows single-point crossings are not evidence.
- **N-4 — `initial:` is not a rest.** `spec.py` uses it to seed spawn-time `vital_metrics`;
  `light_level: 7` has no measured basis and `saturation: 5` was measured wrong. The plan's "or infer
  from `initial`" should be struck; inference from a spawn value is the saturation bug generalized.

---

## Verdict

**DO-NOT-BUILD as written; FIX-THEN-BUILD once the pre-registration in DNB-1 is frozen on `main`,
the range-fix control in DNB-2 has been run first and recorded, and the replay harness carries the
`rest-zero` outcome from DNB-3.** The plan's validation cannot currently distinguish three things a
positive could mean — the mechanism works, two ranges were re-declared, or the form was fitted to
the classroom — and its most natural set-point source produces a positive that is vacuous by
construction. The most likely honest outcome of the controls is that a *static* declared set-point
is the already-shipped range hack under a new name (a plumbing nicety with byte-identical default,
fine to ship as such, no separation claim), and that the *adaptive* set-point the L11 line actually
wants is a different mechanism that needs its own design review with the fitting problem addressed
at the design stage (disjoint adaptation and test windows).

**What I did NOT verify:** the bio-plausibility of any form (bio-faithful lens); whether an adaptive
set-point can be designed with disjoint adaptation/test windows; the live re-encode path and what
Wire-4 / G2 / G3 / R2 do with a `None` baseline (wiring lens); I did not re-run
`l11_real_trace_remeasure.py analyze` at HEAD (the 0.0881 row is taken from the ledger, which is
part of SF-2's point); I did not inspect the geometry probe's settle/sample loop beyond confirming
the samples are consecutive; and my replays use the Slice-1 MEANS (the only thing persisted), so
their absolute values inherit that record's limitations — the direction of every finding does not
depend on them.
