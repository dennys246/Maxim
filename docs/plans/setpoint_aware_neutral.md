# Set-point-aware neutral — the mechanism-faithful fix for the direction problem (L11 "B")

**Status: DEFERRED 2026-09-16 after the four-lens design review — DO-NOT-BUILD as proposed (see
§Four-lens review outcome, folded below; the four lens reports are preserved verbatim under
`docs/experiments/rationale/setpoint-neutral/`). Revive trigger: a roadmap want that STRUCTURALLY
needs a small, same-side move of one graded sensor to separate — and then only as the adaptive
(salience-layer) design, not this one.** The draft below is kept verbatim as the artifact the review
read; where it conflicts with the outcome section, the outcome governs.

*Original status (2026-09-15): DESIGN PLAN DRAFT, for design review BEFORE any substrate code.* The
foundational fix that the L11 line kept pointing at: the Slice-2 rejection
(`docs/plans/l11_slice2_channel_split.md`) proved that cosine separation is a DIRECTION problem and
that no channel-regrouping fixes a near-collinear pair; the bio-faithful lens named the real
upstream fix as **set-point-aware neutral**, which `_sensor_embed` explicitly DEFERS ("there is no
set-point plumbing here, and an unmeasured set-point-aware variant must not be improvised — plan
decision D1", `encoder.py`). This plan proposes to measure and build it. **Nothing here authorizes a
build**; it is the artifact the design review reads.

## Four-lens review outcome (2026-09-16): DO-NOT-BUILD as proposed → DEFERRED; three hygiene deliverables ship instead

Four parallel lenses read this plan, the encoder (`similarity/encoder.py::_sensor_embed`), the
owning brief, the L11 ledger and every offline replay check under `docs/experiments/data/`.
Verdicts: **bio-faithful DO-NOT-BUILD** (3 DNB / 3 SF / 3 NIT); **confounding DO-NOT-BUILD as
written** (3 / 4 / 4); **wiring FIX-THEN-BUILD** (2 / 6 / 3); **regression FIX-THEN-BUILD
conditional on its divergence check, DO-NOT-BUILD as written** (2 / 6 / 4). Two lenses
RE-MEASURED the claim on the committed vectors through the production bases rather than arguing
it. Findings below are grouped by how many lenses reached them independently.

**Cross-confirmed by two lenses, measured (these decide the outcome).**

1. **The motivating case cannot be fixed by ANY static set-point** (bio-faithful DNB-1, confounding
   DNB-1). A sensor contributes in the 2-D plane of its two SHA bases at angle `θ(v) = atan2(v, 1−v)`;
   the whole range sweeps one 90° quadrant, and two SAME-SIDE readings of one graded sensor differ
   by a few degrees whatever the neutral (`nearest_hostile_dist` 0.179→0.087: 6.9° today, at best
   9.5° re-centred). Replayed on the Slice-1 safe/dark vectors with set-point = declared `initial`:
   status quo 0.9766; weight-only 0.9593; re-centred mix 0.9576; signed deviation 0.9550 — none
   clears 0.85, and the small gain is `light_level`'s constant weight dropping, not the mover
   rotating. The only replay that passes (0.7947) fits set-points to four SITE sensors at k = 8 —
   a place-fear fit whose k-curve is non-monotone. Item 1 of §"The problem it fixes" and the
   "regardless of where rest sits" claim are struck; dark=danger via a graded distance sensor is
   banked as **representation-limited by the two-basis arc** (new corollary 7 in
   `docs/wiring/cosine-separation-is-directional.md`).
2. **A STATIC declared set-point is numerically identical to the range declaration the body already
   uses** (bio-faithful DNB-2, confounding DNB-2). Piecewise `[lo,sp]→[0,0.5], [sp,hi]→[0.5,1]`
   reproduces the shipped "phantom half-range" tables for `saturation` and `nearest_hostile_dist`
   digit for digit; on `minecraft_player` 15/17 sensors already have `initial == midpoint`, so
   the primitive would be a two-sensor range re-declaration in a mechanism costume. The saturation
   range fix (#726: cos 0.850→0.787 with NO mechanism change) is that confound in the flesh. The
   front-gate answer flips: it rides existing infrastructure — **build nothing in the encoder.**
3. **The ADAPTIVE reading (set-point = the resting reading, "habituation") fails on measurement and
   is the wrong layer** (bio-faithful DNB-3, confounding DNB-3). Under a rest-anchored neutral,
   2 % jitter on the SAME situation separates 20/20 pairs (min cos 0.175 vs 0.993 today); the rest
   situation itself embeds to the ZERO vector (D2 → `encode_sensors` returns `None`), which every
   existing `cos()` replay helper prints as 0.0 = "SEPARATES" — a vacuous positive by
   construction. A moving neutral also strands frozen `world` prototypes and the NAc biases keyed on
   them, and cannot be named by a geometry tag (it is state, not config). Habituation already lives
   in the right layer (`tools/novelty.py`, `attention/salience_map.py`, Exp 47) DOWNSTREAM of a
   stationary encoding; if wanted for the world channel it is a separate salience-layer plan with
   its own review and disjoint adaptation/test windows.
4. **"Infer from `initial:`" is neither byte-identical nor a rest value** (bio-faithful SF-4, wiring
   D2, regression R5): `light_level` initial 7 on [0,15] ≠ midpoint 7.5; archetype sensors are
   `initial: 1.0` on [0,1]; the infant's `core_temperature` initial is deliberately off its
   set-point; Exp 60's saturation `initial: 5` was exactly this field being wrong. Struck.
5. **The replay as designed is a search, not a test** (confounding DNB-1; wiring D1): form, source
   and k are free and chosen on the validation vectors; the step-2 replay precedes the step-4
   primitive so it must be a hand-copied `embed()` (four of five existing `*_check.py` already
   are), and no step names the live caller (`ModalityChannel` walk + both `agent_loop` encode sites).
6. **The guards the plan leans on are not real yet** (wiring S1, regression R1/R2): there is no golden
   pin for `_sensor_embed` (the byte-identical test mirrors the formula in-process with `approx`),
   and the geometry tag hashes sensor NAMES and normalization MODE but not range VALUES — the
   saturation change proved the hole is live (same tag, different mapping, committed L11 numbers
   silently re-printed 0.0566→0.0881).
7. **Divergence** (regression R8): B is the FOURTH remedy aimed at "safe and dark do not separate"
   (Exp 58 → diagnostic → Slice-2 → B) after the divergence rule fired, and the contingency it was
   meant to unlock was EARNED without it (Exp 60, water=drowning). No roadmap want currently needs a
   small same-side move; Phase 2 carries the drowning want.

**Decision (owner's call recorded here): DEFER the primitive. Ship the three things the review
found that stand on their own, none touching `_sensor_embed`:**

- **(H1) Body-YAML rest lint, CI.** For every gained sensor with a homeostatic drive,
  `drive.set_point == (lo+hi)/2` within tolerance; for every drive-less gained sensor, a checkable
  `rest:` justification for its midpoint (the `minecraft_player` phantom-half-range convention made
  explicit). Makes the Exp 60 saturation class of error a CI failure instead of a live-run discovery.
  (bio-faithful DNB-2's concrete change.)
- **(H2) Range VALUES enter the geometry tag** (regression R2, wiring S1): the tag gains a field only
  when ranges are declared (the `gain` pattern, so undeclared tags stay byte-identical), derived by
  ONE helper shared with the `ec.py` migrate path (the obligation the code already records). Closes
  the live same-dimension hole the saturation change walked through. Needs its own two-lens code
  review; re-run bill priced before merge (rows naming `SensorEncoder`/`_sensor_embed`: Exp 42, 48,
  53b, 56, 57, 60 — dated discharge annotations where the geometry is provably untouched).
- **(H3) Golden pin for `_sensor_embed`** (regression R1): a fixture at the current commit — fixed
  sensor dicts × {gain None, 3.0} × {range-aware, range-blind, partial} → full 384-float vectors
  (exact equality) + the LITERAL geometry-tag strings for the three shipped spaces, two-process with
  differing `PYTHONHASHSEED`, with an anti-vacuity arm that perturbs one weight and must FAIL.

**If the trigger fires (build order, folded from the wiring + confounding + regression lenses).**
(0) H3 on main first. (1) Primitive + passthrough + identity, inert: `_sensor_embed(setpoints=None)`
per-call dict beside `ranges` (NOT a `SensorEncoderConfig` field — per-modality, one live builder,
~12 harness defaults); provenance + tag + D66 derivation through one helper; golden unchanged.
(2) The caller in the SAME PR: `spec._build_reading_schema` validates an explicit `setpoint:` key
(inside range, gained modalities only, parse-time raise), `ModalityChannel` reads it, both loop
encode sites pass it; strict red gate: a body with one `setpoint:` produces a different node through
`propose_via_substrate` than without. (3) Replay through the REAL path only, under a pre-registration
frozen on main first: one form (weight-only, k = 2), explicit `setpoint:` only, record-level
train/test split, frozen thresholds (cos < 0.85 AND fresh-EC distinct ≥ 30/30 AND neither side the
zero vector), per-sensor attribution with the pre-registered discriminator dominant, the range-fix
control run FIRST as the bar to beat. (4) Opt in via a body VARIANT or per-run override, never the
shared `minecraft_player.yaml`; frozen fingerprints gain the geometry tag; re-run bill paid.

**Dismissed / not adopted, with reasons.** The signed-deviation form (bio NIT-7: flips per-sensor
geometry from a 90° arc to 0°/180°, the 0.85 calibration would not transfer). Re-testing the
09-15 Exp 58 vectors as the "honest close-out" (confounding SF-4: apparatus-confounded both ways,
not byte-replayable under the current body — the close-out is corollary 7, which is knowable
without a run). The plan's Q5 sentence that B "would make the small-move wants representable"
(struck at all three sites: here, the Slice-2 §Decision (B), the direction lesson's corollary 5).

## The problem it fixes

`_sensor_embed`'s A4 gain weights each sensor by `w = (|v − 0.5|·2)^p` — magnitude relative to the
literal range **midpoint** 0.5. Two consequences the L11 work measured:
1. A sensor whose meaningful moves are **small and one-sided** (e.g. `nearest_hostile_dist`
   0.179→0.087, both below 0.5) barely rotates the embedding — cosine can't see it. The move IS off
   the sensor's operating point, but the midpoint-relative weight doesn't reflect that.
2. A sensor **resting at an extreme** (e.g. `light_level`=0) carries maximal constant weight and
   zero contrast, drowning the movers.

Set-point-aware neutral makes a sensor's contribution relative to its **rest / expected value**, so a
move OFF baseline rotates the embedding regardless of where rest sits in range — directly the
direction fix. It is the same idea the L11 remeasure already used at the *range* level ("ranges
re-centered so rest sits at the A4 neutral"), promoted from a per-body YAML hack to a measured
substrate primitive.

## Front-gate scope — does it need its own mechanism?

It is a change to the EXISTING encode primitive (`_sensor_embed` + `SensorEncoderConfig`), not a new
bus/bridge/bio-system — one gain/normalization policy parameterized by a per-sensor set-point. So it
rides existing infrastructure. BUT it touches the single most load-bearing substrate function, so the
bar is measurement + review, not elegance.

## Approach (candidate — the review + measurement decide the final form)

- **Set-point source.** Each sensor declares (or the body infers) a `setpoint`/`rest` value (the YAML
  already carries `initial:` — e.g. oxygen `initial: 20`). The weight becomes
  `w = (clamp(|v − setpoint_norm|) · k)^p`, and/or the basis mix re-centers so `v = setpoint` maps to
  the null contribution. Exact form is the measured question.
- **Backward compatibility.** Default MUST remain byte-identical to today when no set-point is
  declared (like `gain_exponent=None` is today), so existing bodies/substrate are untouched unless
  they opt in.
- **Diagnostic-first (the L11 lesson, applied to B).** Before shipping, REPLAY the set-point variant
  offline on the real captured vectors we already have (Slice-1 trace + the L11 remeasure traces) and
  on the bake-off, and show it (a) separates the cases the midpoint gain can't, AND (b) does not
  regress the bake-off's composite `min(sep,stab,disc)` or the cases that already work. Measure, then
  build — do not improvise the variant (D1's explicit warning).

## Open design questions (for the design review)

1. **Bio-faithful — the exact weight/normalization form**, and whether set-point belongs in the gain
   weight, the normalization, or both. Does re-centering break the orthogonal-basis geometry the
   encoder relies on? Reads `docs/agents/bio-memory.md` + the L11 ledger + `_sensor_embed`.
2. **Confounding — validation that isolates the fix.** The offline replay must show the SET-POINT
   change (not some other knob) produces the separation, with the composite bar and the anti-noise
   stability/discrimination legs, on real vectors — not a hand-built demo (D43/D44).
3. **Wiring / blast radius.** It changes ALL gained modalities (`world`) and the geometry tag →
   persisted substrate migration + Exp 56/57 re-baseline (the same gate Slice-2 faced, but here the
   change is a config/plumbing addition with a byte-identical default, so opt-in avoids orphaning
   until a body declares set-points). Grep every consumer; ship with callers.
4. **Regression — Exp 56/57 + the whole graduated line.** `_sensor_embed` underlies every substrate
   result. The behavioral-graduation suite must re-run; a set-point default that shifts existing
   encodings is a DO-NOT-SHIP. Opt-in default is the guard.
5. **Interaction with the demo (C).** Exp 60 (drowning) separates WITHOUT B (oxygen swings full
   range). So B is NOT gated on C and C is NOT gated on B — but B, once measured, is what would make
   the *small-move* wants (like the dark=danger family, or subtle survival cues) representable. B's
   validation could REUSE Exp 58's captured dark/safe vectors as a real test case (does set-point
   make them separate? — the honest re-test of the blocked claim).

## Build order (after review folds)

1. Set-point declaration surface (YAML `setpoint`, or infer from `initial`) + `SensorEncoderConfig`
   parameter; byte-identical default.
2. Offline replay harness: set-point variant vs midpoint, on the Slice-1 + remeasure + bake-off
   vectors — the composite bar. NOMINATES (like the L11 diagnostic).
3. Two-lens code review → fold.
4. If the offline replay clears: build the primitive change; behavioral-graduation re-run + Exp 56/57
   re-baseline + persisted-substrate migration for any body that opts in.
5. Re-test the Exp 58 dark/safe vectors under set-point as the honest close-out of that claim.

## Deliverable of the review

Four lenses (bio-faithful / confounding / wiring / regression) → DO-NOT-BUILD / SHOULD-FIX / NIT into
`docs/experiments/rationale/setpoint-neutral/<lens>.md`; folded here before any substrate code.
