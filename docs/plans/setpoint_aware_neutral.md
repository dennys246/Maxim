# Set-point-aware neutral — the mechanism-faithful fix for the direction problem (L11 "B")

**Status: DESIGN PLAN DRAFT (2026-09-15), for design review BEFORE any substrate code.** The
foundational fix that the L11 line kept pointing at: the Slice-2 rejection
(`docs/plans/l11_slice2_channel_split.md`) proved that cosine separation is a DIRECTION problem and
that no channel-regrouping fixes a near-collinear pair; the bio-faithful lens named the real
upstream fix as **set-point-aware neutral**, which `_sensor_embed` explicitly DEFERS ("there is no
set-point plumbing here, and an unmeasured set-point-aware variant must not be improvised — plan
decision D1", `encoder.py`). This plan proposes to measure and build it. **Nothing here authorizes a
build**; it is the artifact the design review reads.

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
