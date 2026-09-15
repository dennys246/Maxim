# L11 Slice-2 channel-split — CONFOUNDING lens (design review)

**Reviewer lens:** does the design isolate the claimed cause (per-type grouping restores the
danger/safe contrast), or does it manufacture the result / tune the instrument so Exp 58 passes?
**Reads:** `docs/plans/l11_slice2_channel_split.md` (under review), the Slice-1 diagnostic plan +
result (`docs/experiments/data/l11_geometry_2026-09-15.json`), the live-L11 wiring doc, and the
L11 limits ledger (`docs/limits/l11_sensor_dilution.md`).

**Verdict: DO-NOT-BUILD as drawn.** The channel-split *concept* is defensible independent of
Exp 58 (sensors should declare their own sub-modality — 1.1.4 owed that generalisation anyway).
But three design choices, as written, would let a later Exp 58 "pass" for reasons that are not the
danger signal: the specific partition lumps position coordinates into the "threat" channel, the
build gate has no per-sensor attribution requirement, and the split is not controlled against the
simpler already-known remedy (fix the mis-declared ranges of the two constant diluters). Fix those
before any substrate code.

---

## DO-NOT-BUILD

### DNB-1 — The `threat` channel lumps the danger sensor with position coordinates; separation could rest on "where the pit is," not "danger is here"
**Design flaw.** The proposed `world:threat` = {`nearest_hostile_dist`, `hostile_count`,
`y_altitude`, `distance_from_spawn`, `nearest_player_dist`}. Slice-1 shows exactly one of these
carries real contrast — `nearest_hostile_dist` (norm_delta 0.092, gain mass 0.27→0.56). The other
four are position/spatial or entity-count sensors: `y_altitude` (norm_delta 0.037, `moved=false`),
`distance_from_spawn` (0.025, `moved=false`), `hostile_count` (0.024, `carries_mass=false`),
`nearest_player_dist` (0.0, dead-neutral). Putting `y_altitude` and `distance_from_spawn` — pure
position coordinates — into a channel *labelled* "threat" is not a sensor-semantics cut. `y_altitude`
is depth; `distance_from_spawn` is a spawn offset. They land in the "threat" group only because
Exp 58's danger location (the deep pit) happens to differ from safe in depth and offset.

**Confound it causes.** The dark pit sits at a fixed `(x, y, z)`. If the split "works," the danger
cluster may separate from safe on `y_altitude`/`distance_from_spawn` — i.e. the agent learns
"fear when I am at position P," a coordinate artifact, not "fear when a hostile is near." That is a
learned *place* fear masquerading as danger fear, and it would not transfer to a hostile encountered
anywhere else — which is the whole point of a *transferable* dark-fear want (the 1.3 headline). A
reviewer shown only the sensor list would draw threat = entity-proximity (`nearest_hostile_dist`,
`hostile_count`, `nearest_player_dist`) and a SEPARATE spatial channel (`y_altitude`,
`distance_from_spawn`, `on_ground`); the merged threat+spatial cut is the tuning tell.

**Minimal fix.** (a) Pre-register the partition from sensor semantics with Exp 58's needs held out —
threat/entity-proximity, spatial/position, env, interoception/self, motor — do NOT co-locate
position coordinates with entity-proximity in one "threat" channel. (b) The partition boundary must
be reproducible by someone who has never seen Exp 58; write that derivation into the plan. See DNB-2
for the gate that enforces the right sensor is doing the work.

### DNB-2 — The build gate checks "clusters distinct," not "distinct *because of the danger sensor*" — no per-sensor attribution requirement
**Design flaw.** Q7 / diagnostic §5 authorise the build on a live re-encode giving *distinct*
safe/dark clusters at the composite bar. Distinctness alone does not say WHICH sub-channel dimension
produced it. With DNB-1's membership, distinctness can come entirely from a position coordinate.

**Confound it causes.** A green gate that is silent on attribution blesses a split whose separation
is carried by `y_altitude`/`distance_from_spawn`. This is the same "measured a possibility, presented
as proof" shape the D43/D44 lesson and the codebase's four-lens discipline exist to catch, and the
same "is the mechanism the cause or the messenger?" question the divergence-audit rule forces — here
answered by measurement, not assumed.

**Minimal fix.** Add to the pre-registered build gate a **per-sensor attribution** requirement:
decompose the live safe/dark separation in `world:threat` (gain-weighted per-sensor contribution,
the exact metric Slice-1 already computes) and require that `nearest_hostile_dist` (the danger
signal) is the dominant contributor. If the separation is carried by a position coordinate, the gate
FAILS even with distinct clusters. Freeze this criterion before the split is built.

### DNB-3 — The split is not controlled against the simpler, already-known remedy: fixing the mis-declared ranges of the two constant diluters
**Design flaw.** Slice-1's own reading is that the block is **constant-sensor mass**, not small-N:
`light_level` (gain weight **1.0** both, Δ0 — "safe" chamber at y=40 is also underground) and
`time_of_day` (**0.77** both, Δ0) carry maximal mass with zero contrast and out-vote the one working
sensor. But those weights are 1.0/0.77 *because the sensors rest at the extreme of their declared
range* — precisely the mis-declared-range failure the L11 ledger already named and already fixed once
for this world ("rest-at-extreme range declarations make the gained background maximally loud … ranges
re-centered so REST sits at the A4 neutral"). A sensor resting at neutral gets ~0 gain weight (see
`health`, `saturation`, `on_ground`, all silenced to ~0 in the Slice-1 table). So the two diluters
dominate because of a fixable declaration bug, not because of channel size.

**Confound it causes.** If the split ships and Exp 58 later passes, the success is confounded: the
split *excluded* the two mis-declared sensors from the danger channel, but so would re-centering their
ranges — with no structural change at all. The design would credit "per-type grouping" for an effect
actually produced by removing an apparatus artifact. This is a band-aid in the CLAUDE.md sense (a
structural change that hides mis-configured sensors in another channel rather than fixing the
declaration), and it is a divergence-audit trigger ("have any non-code dependencies moved — range
declarations?") going unasked.

**Minimal fix.** Add a **range-fix control arm** to the offline replay and the live gate: re-encode
safe/dark on the *unsplit* 16-sensor channel with `light_level`/`time_of_day` ranges re-centered so
rest sits near the A4-neutral 0.5. If the full channel separates once the two diluters are silenced,
the structural split is not justified by this experiment (or is justified only by the independent
declaration-driven-modality rationale, which must then be argued on its own, not on Exp 58). The split
ships only if it separates AND the range-fix control does not — i.e. only if grouping does work the
range fix cannot.

---

## SHOULD-FIX

### SF-1 — The Slice-2 plan does not restate/freeze the numeric PASS criterion for its build gate; it inherits a pre-reg written for the *diagnostic*
**Flaw.** The diagnostic plan pre-registered the composite `min(sep, stab, disc)` bar, sample N, a
boundary-crossing CI, situations grid, and an apparatus stop-rule — *for Slice-1's measurement*. The
Slice-2 plan says only "distinct clusters at the composite bar" and pins no numbers, no N, no CI for
the build gate. An unpinned gate can be read post-hoc to bless the split.

**Fix.** Port the diagnostic's pre-registration verbatim into the Slice-2 plan as the **build gate's**
frozen criterion — numeric thresholds for each composite leg, sample N, the boundary CI, the exact
situations grid, and the apparatus stop-rule pinned to the Slice-1 classroom fingerprint — and freeze
it (commit) BEFORE the split code lands. The remedy→apparatus→contrast circularity the diagnostic
already flagged means the live re-encode must run on the *same* classroom Slice-1 measured, not a
rebuilt one whose geometry silently differs.

### SF-2 — Zero-/near-zero-contrast sensors inside the threat channel reintroduce dilution and enable a trivial 1-sensor pass
**Flaw.** Two of the five proposed threat sensors carry no contrast: `nearest_player_dist` (0.5/0.5,
Δ0, dead-neutral) and `hostile_count` (identical safe/dark — the bridge counts ALL loaded mobs, so a
forceloaded pit mob shows at "safe" too; flagged in both the diagnostic and wiring docs). Under A0
they add 1/N constant mass; under A4 `nearest_player_dist` is gain-silenced but is still declared
membership. This is the *same* dilution mechanism one level down — and it forces a dilemma the plan
does not resolve: strip the channel to `nearest_hostile_dist` alone and it separates trivially on one
mover (maximal tuning-to-apparatus, DNB-1); keep the passengers and dilution partly returns.

**Fix.** Decide `hostile_count` explicitly — fix the apparatus so it counts only nearby/visible
hostiles (making it a real discriminator, the diagnostic's open item #3) OR exclude it and record the
artifact — do not silently ship a known-broken sensor in the discriminating channel. Drop
`nearest_player_dist` from `threat` unless a player-proximity contingency actually exists. The
composite bar's **discrimination** leg (computed *within* the sub-channel) is the guard against a
1-sensor trivial pass; SF-3 makes sure it is actually applied.

### SF-3 — Confirm the anti-noise composite is applied AT the live gate, with the discrimination leg computed inside the sub-channel
**Flaw.** The bake-off is explicit that small per-channel N lets NOISE separate (A2 grouping 0.00 @
N=100; A3 grouping+threshold WORSE than threshold alone, stability 0.56–0.62). A small threat channel
makes separation easier for ANY contrast, spurious included. The plan cites the composite bar but does
not state that stability and discrimination are evaluated on the *live re-encode* of the *sub-channel*
(not globally, not offline-only). Separation-alone at small N is exactly the failure the bake-off
recorded.

**Fix.** State in the plan that the live gate scores `min(separation, stability, discrimination)` on
the `world:threat` sub-channel, with stability from repeated live samples (the diagnostic already saw
a spot split 6/4 across two ids — jitter is real) and discrimination measuring that the RIGHT sensor
moved. This is the mechanical anti-tuning guard; DNB-2's attribution requirement is its per-sensor
sharpening.

---

## NIT

### N-1 — Offline replay is in-sample; label it and require fresh live samples at the gate
The build-order step 5 replays "the Slice-1 captured vectors, re-grouped per sub-channel." That is
evaluation on the same sample used to nominate the remedy — fine as a nomination (the diagnostic
already labels channel-split "offline-only"), but the plan should state that the offline number is
in-sample and that the live re-encode gate (step 6) captures FRESH vectors, never a replay of the
Slice-1 capture.

### N-2 — A4 gain has already shrunk the effective channel; the "small-N channel" benefit is smaller than framed
Slice-1 shows A4 already silences most of the 16 sensors to ~0 gain weight; effective mass sits on
~4 sensors (`light_level`, `time_of_day`, `nearest_hostile_dist`, `y_altitude`). The channel is
*already* effectively small-N and still fails — because two of those four are the constant diluters
(DNB-3). Framing the split as "back into the small-N regime" overstates the mechanism; the honest
framing is "remove the two constant-mass diluters from the danger channel," which is exactly why the
range-fix control (DNB-3) is the necessary comparison.

---

## Summary for the fold

The split concept can be justified without Exp 58 (declaration-driven sub-modality), but the design as
drawn cannot be built: (DNB-1) the `threat` partition co-locates position coordinates with the danger
sensor, (DNB-2) the gate never checks the danger sensor is the one separating, and (DNB-3) the split is
uncontrolled against re-centering the two mis-declared constant sensors that Slice-1 identifies as the
actual cause. Any Exp 58 pass under this design would be ambiguous between "danger fear," "place-fear
on a position coordinate," and "an apparatus range-fix wearing a channel-split costume." Fix the
partition, add per-sensor attribution + the range-fix control to a frozen numeric build gate, resolve
the `hostile_count`/`nearest_player_dist` passengers, and the experiment becomes able to isolate the
claimed cause.
