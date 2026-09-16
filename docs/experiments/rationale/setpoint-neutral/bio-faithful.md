# Set-point-aware neutral — BIO-FAITHFUL lens (four-lens design review, 2026-09-16)

Reviewer lens: does "set-point-aware neutral" (`docs/plans/setpoint_aware_neutral.md`, the L11
line's "B") do the substrate's REAL job — make a situation the world distinguishes land in a
distinct EC cluster — or does it re-spell a lever the body already has while leaving the
measured failure untouched? Read against the code, the owning brief, the L11 ledger, the
direction lesson, and the offline replay checks; every number below was recomputed through the
shipped SHA bases (`similarity/encoder.py::_stable_basis`) on the real Slice-1 safe/dark vectors
(`docs/experiments/data/l11_geometry_2026-09-15.json`), the same way
`docs/experiments/data/l11_slice2_cosine_check.py` decided Slice-2.

**Verdict up front: DO-NOT-BUILD as proposed.** The plan's headline case is not fixable by ANY
placement of the neutral (measured), the static form of the primitive is numerically identical
to the range declaration the body already uses (measured), and the adaptive form collapses
stability 20/20 (measured). Details, evidence and the concrete replacement follow.

---

## Verified first

- `similarity/encoder.py::_sensor_embed` — contribution per sensor is
  `w · ((1−v)·basis_low + v·basis_high)` with `v = _normalize_value(value, range)` and
  `w = (|v−0.5|·2)^p` when gained (`p = SensorEncoderConfig.gain_exponent = 3.0`, applied only
  to `gain_modalities = {"world"}`). The D1 deferral comment is there verbatim ("there is no
  set-point plumbing here, and an unmeasured set-point-aware variant must not be improvised").
  `None` gain → `w = 1.0`, byte-identical pre-A4 sum. Zero vector → `encode_sensors` returns
  `None` (D2 designed-rest branch, `_designed_rest` stash).
- `similarity/encoder.py::_stable_basis` — bases are SHA-fixed per `(name, salt)`; nothing in any
  candidate form touches them, so basis (near-)orthogonality is not at stake in this review.
- `similarity/encoder.py::SensorEncoder.encode_sensors` — the delta gate keys on values AND
  `_last_ranges` identity (a ranges flip must not return the cached node); the geometry tag
  (`encoding_geometry_tag`) hashes `declared_sensors` + normalization mode + `gain` (added ONLY
  when gain applies — the "do not gratuitously re-stale" precedent). `docs/agents/bio-memory.md`
  §threshold table: sensor threshold **0.85**, world/interoception/audio are
  **frozen-centroid** (`similarity/ec.py::ECConfig.frozen_centroid_modalities`), so the
  running-mean drift lesson (`docs/lessons/ec-centroid-drift.md`) does not apply to `world`
  encodings — but frozen means the FIRST vector fixes a node's prototype forever.
- Ranges reach the encoder from the body walk: `runtime/agent_loop.py::_read_declared_modality_ranges`
  reads `reading_schema["range"]` per `modality: world` sensor, lockstep with
  `_read_declared_modality_states` (the same-set invariant in the brief). **No reader touches
  `initial:` on the encode path.** `initial:` is consumed by `embodiment/spec.py::_parse_entity`
  into `SpecSensor._initial`, whose only runtime use is the `SpecSensor.read` fallback when
  `vital_metrics` has no value — it is a boot/start value, with no rest semantics anywhere in
  `src/maxim/`.
- The drive layer owns the only declared set-point: `embodiment/sem.py::HomeostaticDriveSpec.set_point`
  (sensor units; SHAPE-FROZEN CC3), consumed by `drive_pain_for_value`,
  `corrective_need_intensity`, `drive_comfort_progress`, `embodiment/body.py::evaluate_failures`
  (pain latch) and `Body.tick_vital_drift` (drift toward set-point). Entropic drives have NO
  set-point (thresholds instead). Exteroceptive sensors (`light_level`, `time_of_day`, `speed`,
  `distance_from_spawn`, …) have no drive and therefore no set-point in the model.
- `bodies/minecraft_player.yaml` "RANGE PRINCIPLE" block: every world range is declared so the
  sensor's RESTING value sits at normalized 0.5 — e.g. `health [0,40]` rest 20, `oxygen [0,40]`
  rest 20, `nearest_hostile_dist [0,128]` rest = bridge cap 64, `saturation [0,20]` rest = clamp
  10, `distance_from_spawn [-128,128]` rest 0, `is_in_water [-1,1]` rest 0. For every
  homeostatic drive on this body, `drive.set_point == range midpoint` already.
- `docs/wiring/cosine-separation-is-directional.md` (the measured lesson): cosine sees direction;
  the separating moves are neutral→extreme or extreme→opposite extreme; a small same-side move
  does not rotate the sum; corollary 4 — adding a rest-at-neutral sensor is safe, changing an
  existing sensor's encoding is destructive. `docs/plans/l11_slice2_channel_split.md` §Decision
  names B as "the mechanism-faithful fix" but its own bio-faithful lens
  (`rationale/l11-slice2/bio-faithful.md`, DO-NOT-BUILD 2) already noted in parentheses: "even
  set-point neutrality would not help here" for the dark/safe pair.
- `docs/limits/l11_sensor_dilution.md` 2026-09-15 row and `exp60_saturation_rest_check.py`: a
  declared rest the world never visits is a constant, not a neutral (saturation declared rest 5,
  real rest 10). `docs/experiments/data/exp60_spawn_distance_check.py`: capped sensors go
  full-weight constant at the cap.

---

## Findings

### DO-NOT-BUILD 1 — The plan's headline case (`nearest_hostile_dist` 0.179→0.087) is NOT fixable by set-point neutral, under any candidate form. Measured: 0.955–0.959, threshold 0.85.

**Finding.** Plan §"The problem it fixes" item 1 says the small one-sided move "IS off the
sensor's operating point, but the midpoint-relative weight doesn't reflect that", and that
set-point neutral "makes a move OFF baseline rotate the embedding regardless of where rest
sits". Both halves are wrong for this sensor, and the second is wrong in general.

**Evidence (worked on the encoding).** A sensor contributes a vector in the 2-D plane spanned by
its `basis_low`/`basis_high`, at direction angle `θ(v) = atan2(v, 1−v)` — the whole range
`v ∈ [0,1]` sweeps only the 90° quadrant from pure-low to pure-high. Two readings of ONE sensor
can differ in direction by at most the arc between their two `θ`s; magnitude (`w`) is invisible
to cosine. For `nearest_hostile_dist`: `θ(0.1786) = 12.3°`, `θ(0.0871) = 5.4°` → single-sensor
direction cosine **0.9929**. Re-centering the mix so a set-point `sp` maps to the equal-mix
point (`d = clamp((v−sp)·2)`, coefficients `(1−d)/2, (1+d)/2`) with `sp = 0.3` (between the
cap and the readings) gives `θ = 31.4°` vs `21.9°` → **0.9865**. With `sp = 0.5` (= the
declared rest, since the bridge cap 64 is this sensor's real rest and the range principle already
put it at 0.5) the map is the identity → 0.9929 again. **No affine re-mapping of `v` that keeps
both readings on the same side of the set-point can open that arc past a few degrees**; the only
way a single sensor rotates the SUM by the ~32° that `cos < 0.85` needs is to swing across the
set-point (sign flip in `d`) with dominant mass — the direction lesson's "neutral→extreme" case,
which this pair does not have. The plan's sentence "regardless of where rest sits" is exactly
the claim the direction lesson refutes: re-locating the neutral only helps when the neutral was
in the WRONG place; it cannot manufacture an arc between two same-side readings.

Full 16-sensor replay through the real bases, Slice-1 safe vs dark, set-point = each sensor's
DECLARED `initial` normalized (the plan's own proposed source; 0.5 for 14 sensors,
`light_level` 7/15, `time_of_day` 0.0):

| form | cos(safe, dark) | clears 0.85? |
|---|---|---|
| status quo A4 (midpoint weight + midpoint mix) | 0.9766 | no (Slice-1's own number) |
| C1 set-point in the WEIGHT only, mix unchanged | 0.9593 | no |
| C2 re-centered mix + set-point weight | 0.9576 | no |
| C3 signed deviation on `(basis_high − basis_low)`, null at set-point | 0.9550 | no |

The small improvement is entirely `light_level`'s weight dropping (rest 7/15 ≈ 0.467 instead of
the midpoint) — the constant-diluter effect — not any rotation of the mover. The plan's build
step 5 ("re-test the Exp 58 dark/safe vectors under set-point as the honest close-out") would
return **NOT SEPARATED** for every form; that is knowable now, for free, and the prior lens
already said so in passing. Building the primitive on this motivation builds it for a case it
cannot fix.

**Concrete change to the plan.** Strike item 1 from §"The problem it fixes" and record the
replay table above (or its equivalent under `docs/experiments/data/`) as the reason. Bank
dark=danger-via-`nearest_hostile_dist` as **representation-limited by the two-basis arc** (a
single graded sensor has ≤90° of direction to give, and a same-side excursion uses a few
degrees of it) — the honest fixes remain the ones the Slice-2 decision already named: an
apparatus whose cue crosses neutral, or a binary in-state flag at rest-neutral (the Exp 60
`is_in_water` pattern, corollary 2 of the direction lesson). Do not carry "B would make the
small-move wants representable" (plan Q5) into any prereg.

### DO-NOT-BUILD 2 — A STATIC per-sensor set-point is numerically identical to the range declaration the body already uses; the primitive adds no representational capability, so the plan's own front-gate answer flips to "rides existing infrastructure — build nothing".

**Finding.** Plan §"Front-gate scope" says the change "rides existing infrastructure" because it
is a parameter on `_sensor_embed`. The stronger fact is that existing infrastructure ALREADY
EXPRESSES it: a piecewise-linear normalization that sends `[lo, sp] → [0, 0.5]` and
`[sp, hi] → [0.5, 1]` is, for every sensor on `minecraft_player`, exactly what the declared
"phantom half-range" produces through the existing `_normalize_value`.

**Evidence.** Through the shipped `_normalize_value`:

| `saturation` raw | range-principle `[0,20]` (shipped) | piecewise `sp=10` on `[0,10]` |
|---|---|---|
| 0 / 2.5 / 5 / 7.5 / 10 | 0.000 / 0.125 / 0.250 / 0.375 / 0.500 | 0.000 / 0.125 / 0.250 / 0.375 / 0.500 |

| `nearest_hostile_dist` raw | range-principle `[0,128]` (shipped) | piecewise `sp=64` on `[0,64]` |
|---|---|---|
| 0 / 11 / 23 / 40 / 64 | 0.000 / 0.086 / 0.180 / 0.313 / 0.500 | 0.000 / 0.086 / 0.180 / 0.313 / 0.500 |

Identical to four decimals (the shipped map is exact linear algebra, so identical in every
digit). The plan even says so ("the same idea the L11 remeasure already used at the *range*
level"), but draws the conclusion "promote to a substrate primitive" where the front-gate
principle in CLAUDE.md draws the opposite one: *if it can ride on existing, choose that path even
when less architecturally elegant.* What the primitive would add over the range lever is only
(a) an explicit declaration instead of an implicit midpoint — a hygiene/lint question, not an
encoding question — and (b) a MOVING set-point, which is DO-NOT-BUILD 3.

Also note the cost the plan under-states: any sensor that opts in with `sp ≠ midpoint` changes
the space (geometry tag must move — same-dim hole, Gate 2/D4) and re-encodes every persisted
world node for that body; "opt-in" avoids orphaning only for bodies that never opt in, i.e. the
primitive is safe exactly where it is unused.

**Concrete change to the plan.** Replace "build a `setpoint` parameter on `_sensor_embed`" with
a declaration-hygiene deliverable that needs NO encoder change: (1) a body-YAML lint (CI, like
`scripts/lint_*`) asserting for every gained sensor with a homeostatic drive that
`drive.set_point == (lo+hi)/2` within tolerance, and for every drive-less gained sensor that the
YAML carries a `# rest = …` justification for its midpoint (the `minecraft_player` convention,
made checkable); (2) the L11 ledger row "a declared rest the world never visits is a constant"
promoted to that lint's docstring. This keeps the byte-identical default trivially true (no
code path changes) and makes the Exp 60 `saturation` class of error a CI failure instead of a
live-run discovery.

### DO-NOT-BUILD 3 — The ADAPTIVE reading of "set-point" (habituation; rest inferred from the resting reading) collapses stability 20/20 and zeroes the rest vector; it is the wrong layer for habituation.

**Finding.** The plan's text slides between two mechanisms: a declared homeostatic set-point
(interoceptive; what `HomeostaticDriveSpec.set_point` is) and an adapted baseline ("rest /
expected value", "the body infers", the Slice-2 decision's word "habituation"). Biology keeps
these distinct: homeostatic set-points are regulated targets with error signals (the drive
layer models them faithfully — pain, drift, relief credit); exteroceptive adaptation is
sensory-gain/novelty modulation that suppresses a CONSTANT background so a CHANGE is salient —
it does not move where the encoding says "zero". The plan needs one or the other; the adaptive
one fails on the measurement.

**Evidence.** Set-point = the Slice-1 SAFE reading itself (what an adapted baseline would be
after resting in the safe chamber), form C2, 2% uniform jitter on every sensor, 20 pairs of
"the same safe situation, re-sampled":

| neutral | jitter-pair cos min / mean | pairs `< 0.85` (noise SEPARATES) |
|---|---|---|
| A4 midpoint (today) | 0.993 / 0.998 | 0 / 20 |
| set-point = rest reading | **0.175 / 0.478** | **20 / 20** |

Mechanism: at an adapted rest every sensor's `d ≈ 0`, so the surviving mass is whatever
jittered most this tick, and each tick points in a different sensor's basis direction — the
bake-off's N=6 stability collapse (0.62) in its limit form. And the un-jittered rest reading
itself embeds to the ZERO vector (the D2 branch: `encode_sensors` → `None`), so `cos = 0.0`
against it — which a naive replay reads as "separates". Any offline harness for this plan that
does not special-case zero-norm sides will report a vacuous pass (the same family as the
green-PR/absent-check lesson: a degenerate measurement looks exactly like a strong one).

There is also a persistence reason this cannot live in the encoder: `world` is frozen-centroid,
and NAc `cluster_reward_bias` is keyed on EC node ids. A set-point that moves with experience
makes the SAME world state encode to a DIFFERENT direction over time, so persisted prototypes and
their biases become unreachable — and because the set-point is state, not config, the geometry
tag cannot even name the space a node was written in. The codebase already has habituation in
the right layer: `tools/novelty.py` / `attention/salience_map.py` (Exp 47,
`docs/experiments/47_habituation_novel_in_noise.md`) modulate salience by cluster familiarity
DOWNSTREAM of a stationary encoding. That is the bio-faithful placement.

**Concrete change to the plan.** Delete the adaptive/inferred reading entirely. If exteroceptive
habituation is wanted for the world channel, file it as a separate plan on the salience layer
(familiar cluster → lower orient/threat salience), with its own design review; it must not touch
`_sensor_embed`.

### SHOULD-FIX 4 — `initial:` is not a rest value and must not be the set-point source; the only declared set-point is `drive.set_point`, and two rest concepts already exist.

**Finding.** Plan §Approach: "Each sensor declares (or the body infers) a `setpoint`/`rest`
value (the YAML already carries `initial:`)". `initial:` is a start value: `SpecSensor._initial`
is read once as the fallback when `vital_metrics` is empty (`spec.py::SpecSensor.read`) and
serialized back out (`sem.py`); nothing treats it as an equilibrium.

**Evidence from the shipped bodies.** `infant_humanoid.yaml` `core_temperature: initial: -0.15
# slightly cool — blanket provides relief` with `drive.set_point: 0.0` — the start is
DELIBERATELY off set-point so the blanket has something to relieve; inferring rest from
`initial` would make "slightly cold" the silent state and warmth the excursion — the inverse of
the drive's own definition. `minecraft_player.yaml` `time_of_day: initial: 0.0` — dawn; a cyclic
sensor has no rest at all (the YAML's own wrap note). `light_level: initial: 7` — neither the
surface day value (15) nor the cave value (0); a rest nobody visits. Exp 60 `saturation`: the
old `initial: 5` was the declared-but-never-visited rest whose constant mass broke gate (ii)
(`exp60_saturation_rest_check.py`); the fix moved BOTH `range` and `initial` because the two are
kept in agreement by hand, not because `initial` means rest. The plan proposes inferring the
neutral from precisely the field that was wrong there.

The codebase already holds two rest concepts: `HomeostaticDriveSpec.set_point` (sensor units,
homeostatic drives only, the regulated target) and the range midpoint (the encoding neutral for
every gained sensor, by the RANGE PRINCIPLE). A per-sensor `setpoint:` would be a third, able to
disagree with both. Entropic drives (hunger, food) have no set-point by design — their "rest" is
a threshold band — and exteroceptive sensors have none; a single primitive named "set-point"
does not faithfully cover them.

**Concrete change to the plan.** If any declaration surface survives DO-NOT-BUILD 2, it is a
derivation rule, not a new field: encoding neutral := `drive.set_point` when the sensor carries a
homeostatic drive, else the declared range midpoint; never `initial`; and the lint in
DO-NOT-BUILD 2 enforces the equality for homeostatic sensors. State in the plan that entropic and
exteroceptive sensors have NO set-point in the model and keep the midpoint convention.

### SHOULD-FIX 5 — "Byte-identical default" is true for the weight-only form only, and only under specific implementation constraints; the re-centered-mix form is NOT byte-identical.

**Finding.** Plan §Approach promises a default "byte-identical to today when no set-point is
declared (like `gain_exponent=None` is today)". Traced:

- Weight-only (C1) with a default `sp = 0.5`: `(abs(v - sp) * 2.0) ** p` performs the same float
  operations as today's `(abs(v - 0.5) * 2.0) ** p` → byte-identical vectors. Any other spelling
  (e.g. `abs(v - sp) / (0.5)`) is not.
- Re-centered mix (C2) with `sp = 0.5`: coefficients `(1−d)/2` with `d = (v−0.5)·2` equal `1−v`
  algebraically but not in IEEE arithmetic → ULP-level differences in every gained vector. The
  tag would not move (it hashes config, not vectors) while the vectors do — a same-tag,
  different-vector space, which is precisely what the tag exists to forbid. Pinned today by
  `tests/unit/test_world_channel.py::test_undeclared_bodies_are_byte_identical` and the Gate-2
  tag tests; those must be the acceptance tests, not prose.
- Plumbing that is required for the claim to hold even for C1: the per-sensor set-point map must
  join the delta-gate identity (as `_last_ranges` does — otherwise a set-point flip with still
  values returns the stale node, the exact P1 bug), a third lockstep walk
  (`_read_declared_modality_setpoints`) must emit the same sensor set as the states/ranges walks
  (the brief's same-set invariant), and the geometry tag gains a field ONLY when a set-point is
  declared (the `gain` precedent — adding `setpoint=None` to every tag re-stales every existing
  node for a space that did not change).

**Concrete change to the plan.** If anything is built: commit to C1 only, name the three plumbing
obligations above, and name the two pinned tests as the byte-identical gate. C2/C3 cannot claim
byte-identity.

### SHOULD-FIX 6 — The offline replay (build step 2) must guard the zero-vector degeneracy and carry the stability leg with jitter around rest, or it will pass vacuously.

**Finding.** The plan correctly demands the composite `min(sep, stab, disc)` bar, but a
set-point-relative encoding produces zero-norm vectors at rest by construction (D2), and
`cos(0, x)` reads 0.0 in every replay script under `docs/experiments/data/` (`cos` returns 0.0
when a norm is 0). A harness that scores "separation" as `cos < 0.85` will count every rest
comparison as a perfect separation.

**Concrete change to the plan.** Specify: (1) a zero-norm side is scored UNDEFINED and the pair
is excluded from separation AND counted as "no cluster for this situation" (which is a
representational failure for any contingency that must key on the rest state); (2) the stability
leg is measured as jitter-pairs of the SAME situation around its rest (the table in
DO-NOT-BUILD 3 is the template), not only between distinct situations; (3) the isolated-vs-
sequential measurement from the centroid-drift rule (brief §3) — `world` is frozen so drift is
not expected, but the rule is "always measure both".

### NIT 7 — Form C3 (signed deviation on `basis_high − basis_low`) changes the per-sensor geometry from a 90° arc to a 0°/180° flip; the 0.85 threshold was calibrated on the arc and would not transfer.

`SensorEncoderConfig.pattern_threshold`'s own comment calibrates 0.85 on "a single-sensor swing
0→1 leaves cos≈0.83 with the baseline" under the two-basis interpolation. Under C3 a sensor has
exactly two directions (±u); graded readings on one side are perfectly collinear per sensor and
carry information only through cross-sensor mass ratios. It is a different encoding family, not
a parameter on this one, and would need its own bake-off. Drop C3 from the candidate list.

### NIT 8 — Orthogonality, frozen centroids, hivemind threshold: nothing breaks, record it.

None of the forms touch `_stable_basis`; SHA bases stay fixed and their pairwise near-
orthogonality in 384-d is unaffected (only coefficients change). `world` is frozen-centroid, so
the running-mean drift path is not exercised; `hivemind/merge.py`'s per-modality threshold table
pins `world: 0.85` and would not need to move for C1. Worth one sentence in the plan so the
"does re-centering break the orthogonal-basis geometry" question (plan Q1) is closed with a
reason rather than left open.

### NIT 9 — Plan Q5 and the Slice-2 §Decision (B) overclaim what set-point neutral buys; fix the prose at both sites.

`l11_slice2_channel_split.md` §Decision (B): "makes a sensor's contribution relative to its
rest value, so a small move off baseline rotates the embedding — directly addressing the
direction problem." DO-NOT-BUILD 1 shows the second clause is false for same-side moves. The
direction lesson's corollary 5 carries the same sentence. Both should be amended to: set-point
neutral relocates WHERE the null is; it does not enlarge the arc a graded sensor can sweep, and
the range declaration already relocates the null.

---

## Verdict

**DO-NOT-BUILD** (as proposed). The primitive is motivated by a case it measurably cannot fix
(0.955–0.959 vs 0.85 across all three forms, because two same-side readings of one graded
sensor are near-collinear under ANY placement of the neutral), its static form is numerically
identical to the range declaration `minecraft_player.yaml` already applies (front-gate answer:
ride existing), and its adaptive form destroys stability (20/20 jitter pairs separate) and the
cluster-identity persistence NAc biases depend on. What survives is declaration hygiene — derive
the encoding neutral from `drive.set_point` where a homeostatic drive exists and lint
`set_point == range midpoint`, never from `initial:` — which needs no change to `_sensor_embed`
and keeps D1's deferral intact for the right reason: there is nothing here for the encoder to do.
If exteroceptive habituation is wanted, it belongs on the salience layer (Exp 47 machinery), as
its own reviewed plan.

**Not verified:** I did not run the bake-off (`scripts/encoding_bakeoff.py`) or
`l11_real_trace_remeasure.py analyze` under any set-point form on the 09-04 trace — the replay
here is the 16-sensor Slice-1 pair plus a jitter stability probe through the real bases, the
same instrument class that decided Slice-2 and Exp 60, not a full composite-bar run; the
numerical identity in DO-NOT-BUILD 2 was checked on `saturation` and `nearest_hostile_dist`, not
swept over every declared sensor (the shipped map is exact linear algebra, so I expect no
exception, but I did not enumerate); I did not audit hivemind bundle migration for an opt-in
body (wiring lens); and the biology claims (homeostatic set-point vs sensory adaptation as
distinct mechanisms) are stated from the model the codebase already commits to in
`embodiment/sem.py` and Exp 47, not from a fresh literature pass.
