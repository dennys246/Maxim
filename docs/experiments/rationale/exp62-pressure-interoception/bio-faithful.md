# Exp 62 — bio-faithful lens (design review of DRAFT v1, 2026-09-17)

Charter: does the design test the mechanism's REAL job, not a caricature; does the manipulation
respect how the substrate / body / drives actually work. Read: `docs/agents/bio-memory.md` (Wire-4,
A4 range principle, H1/H2), `docs/agents/embodiment.md`, `docs/wiring/cosine-separation-is-directional.md`,
`survival_world_1_3.md` (D1 + world topology), R3 §D3, `bodies/minecraft_player.yaml`,
`scripts/lint_body_rest_neutral.py`, `similarity/encoder.py`, `decisions/nac.py`,
`proprioception/pain_bus.py::create_pain_cluster_fear_subscriber`, `runtime/agent_loop.py`,
`scripts/minecraft_bridge/index.js`, the Exp 60 live gate (ii) record and apparatus record.

Where the charter needed a number rather than an opinion, I replayed the draft's sensor on the
SHIPPED embed (`_sensor_embed`, p = 3.0, the Exp 60 live gate (ii) normalized values from
`exp60_geometry_2026-09-15b.json`, apparatus depth 5 → ≈ 4 water blocks above the eye block). The
scratch script is not committed; its output is quoted inline and is reproducible from those two
committed records in ~30 lines (corollary 3 of the cosine wiring doc).

## Findings

### DO-NOT-BUILD

**DNB-1 — The floor the claim stands on does not exist on this encoder, and every placement that
manufactures it breaks the apparatus at pool 2.** The claim's premise is that the shipped roster's
absolute features (`y_altitude`, `distance_from_spawn`) put pool 2's submerged reading in a
different world cluster from pool 1's. Replayed on the shipped embed with everything else held at
the Exp 60 live values (dark underground pool, `light_level` 0, `time_of_day` 0.0417, `is_in_water`
1.0, oxygen at the dive mean), cos(pool-1 submerged, pool-2 submerged) is **0.9966** at y = 64 /
36 blocks from spawn, **0.9919** at y = 100 / 90 blocks (Exp 60's spawn bound), **0.9587** at
y = 10 / 90, **0.9049** at y = 10 / 120 (already PAST the bound). All ≥ 0.85: the shipped roster
HITS the trained key at pool 2. The absolutes carry at most w ≈ 0.6 (y = 10) against three shared
full-weight constants (`is_in_water` 1.0, `light_level` 1.0, `time_of_day` 0.77) that make the two
pools the same direction. The ONLY placements where the shipped roster misses push BOTH absolutes
to their range extremes — y = 0 AND `distance_from_spawn` at the 128 cap gives cos 0.8195 — and at
every one of those placements pool 2 fails Exp 60's own gate (ii): cos(shore-2, submerged-2) =
**0.8535–0.8869 ≥ 0.85** (corollary 6, the capped-constant law — this is the 0.8525 @128 Exp 60
already measured). So arm 1 cannot be a floor without pool 2 being a place where shore and water
are one cluster, i.e. where the fear cannot be booked or read as a WATER fear at all. *Failure
scenario:* D1 is set "so the replay misses"; the only satisfying offsets are the cap; the harness's
pool-2 cluster-distinct preflight (Exp 60's live preflight, inherited "verbatim") refuses every
row, or — if that preflight is dropped to make it run — arm 2 fires on a cluster that is also the
shore cluster and the outcome is written as cross-context transfer. Evidence: the replay above;
`exp60_drowning_avoidance_prereg.md` §Apparatus second placement guard (0.8525 @128);
`cosine-separation-is-directional.md` corollary 6. The draft's own §Why-the-replay-comes-first
predicts this hedge ("the shipped roster misses" is stated as the prediction to CHECK) — the check
is answerable now, and the answer is that D1 has no in-bounds solution.

**DNB-2 — `pressure` as declared is inert at the depth the experiment runs at, and at the one depth
where it is loud it generalizes by SWAMPING, which is the caricature of what a body-carried cue
should do.** Range `[-12, 12]`, rest 0, negative half unobservable: depth d maps to v = (d+12)/24,
so the sensor sweeps only the upper half-quadrant, and under the cubic gain its weight is w =
(d/12)^3 — **0.0006 at 1 block, 0.037 at 4 (Exp 60's placement), 0.125 at 6, 1.0 only at 12.** At
the Exp 60 depth it moves cos(shore, submerged) from 0.7874 to **0.7879** and cos(pool-1, pool-2)
by **0.0000** in every case above. This is corollary 7 verbatim — a graded sensor that never
crosses its neutral is representation-limited, and the draft cites corollary 7 and then builds
five arms on the sensor anyway. At depth 12 (w = 1.0) the sensor DOES move things — the wrong way
for the claim's honesty: it is the same value in both pools, so it adds a shared full-weight basis
and RAISES cross-pool cosine (0.9049 → 0.9238 at the y=10/dist-120 case) — and it raises the
daylit-vs-underground case too (0.5973 → 0.6988), i.e. it is on its way to merging a lit lake with
a dark cave pool. That is not "the fear keys on a relative feature"; it is "one more constant that
makes every submerged reading look alike", the mass-not-direction confusion corollary 1 exists to
name. *Failure scenario:* v2 fixes DNB-1 by deepening both pools to ~12 blocks so `pressure` is
loud; arm 2 then beats arm 1 and the outcome reads "a relative feature made the fear
location-invariant", when the mechanism was dilution of whatever differed — the finding would
"transfer" a water fear to any 12-deep water regardless of every other feature, and would equally
swamp a real difference (a hostile-guarded pool) that the fear should NOT generalize over. The
structural cause is H1 + one-sided observability + p = 3: a rest-at-midpoint declaration for a
quantity the world only ever visits on one side spends half the arc on nothing (the `hostile_count`
shape — which was never asked to be THE cue). There is no range re-declaration that fixes it under
the lint, and a binary re-cut would simply duplicate `is_in_water`.

Together: the draft's key sentence "no shipped sensor is RELATIVE to the water surface" is false —
`is_in_water` is computed from the EYE block (`index.js` lines 107–120: "HEAD-block only … tracks
the DROWNING situation"), which is exactly a surface-relative cue at rest-neutral with a full-weight
swing; it is why Exp 60 worked. `pressure` adds a graded version of the same relativity at 3.7 % of
its weight. The body already carries the bio-faithful cue; what it ALSO carries is place.

### SHOULD-FIX

**SF-1 — The body's real job here is cued fear, and the experiment that tests it is arm 4, not
arm 2.** `survival_world_1_3.md` §World topology already made the bio-faithful commitment: "the
survival body senses the contingency FEATURES … not place; offsets added back only for the
spatial/farming rung". In the biology the draft borrows from, cued fear (amygdala, keyed on the
proximal CS) and contextual fear (hippocampal place code) are separate systems; this body sums
place (`y_altitude`, `distance_from_spawn`) and cue (`is_in_water`, `oxygen`) into ONE world vector
read by one exact-key threshold. The honest cross-context question for this body is therefore
"does a survival body that carries NO place in its world channel fire the water fear at a new pool"
— the draft's arm 4, which it demotes to a control. The replay says arm 4 hits (cos 1.0000 at every
dark pool 2) — but so does the shipped roster, so at in-bounds placements arm 4 and arm 1 are both
ceilings and the only real contrast in the table is DNB-1's. Recommend: drop `pressure` from the
claim, promote the ablated-absolutes body to the manipulation, and — more useful — ask the
question the replay actually raises (SF-2).

**SF-2 — The cross-context blockers on this body are the OTHER situation features, and that is not
a body problem.** The one placement that separated the pools in the replay was lighting: a daylit
surface pool reads cos **0.5973** against the cave pool (shipped), and NEITHER `pressure` (0.5966 at
depth 4) NOR removing the absolutes (0.5881) bridges it. A lit lake and a dark cave pool ARE
different situations to a feature-first body, and biology generalizes across them by GRADED cue
similarity — which is exactly the exact-key read's limit (R1's cache wall) and Phase 4's job, not
something to move into the body. The draft's D1 ("altitude band" / "distance from spawn") never
says whether pool 2 is lit, above ground, or at the same frozen time — light_level parity must be
DECLARED in D1, or the contrast is about light (and time_of_day drift, w 0.77 at the Exp 60
freeze, is a second such feature: if the campaign spans game hours between pools the key moves for
a reason that is neither place nor pressure). If v2 survives DNB-1/2 it should state which
situation features are held equal across pools and which are allowed to differ, and the claim's
scope is "invariant to <those>", never "location-invariant".

**SF-3 — Modality honesty: D3 is not a choice, it is forced twice, and the name should follow the
fact.** (a) `spec.py` REFUSES `modality: interoception` on any sensor (interoception membership comes
ONLY from a `drive:` block), so without a drive `pressure` cannot enter the interoception encode at
all; `modality: world` is the only way a drive-less sensor enters any encode. (b) Even with a drive,
Wire-4 books fear on the WORLD cluster only (`create_pain_cluster_fear_subscriber`:
`nac.active_clusters(agent_id).get("world")`, "fear keyed to the interoception cluster is
tautological"); `anticipatory_threat_need` reads across every noted modality but nothing ever
writes an interoception fear. So an interoception declaration would change nothing for the fear and
is not declarable anyway — confirmed. Given that, `pressure` is interoception in name only: it is a
world-owned situation feature (blocks of water above the eye) exactly like `is_in_water`. The §Not
claimed disclaimer is not enough because the NAME persists where the prose does not — in the YAML,
the geometry tag's `declared_sensors`, the frozen fingerprint, and every exported bundle's
provenance. Name it what it measures (`submersion_depth` / `water_above_eye`, unit blocks) and
reserve `pressure` for a drive that publishes pain (Exp 62b), so a future reader of a bundle cannot
mistake a world feature for a felt drive.

**SF-4 — A gained sensor is a geometry-tag change, the Exp 60/61 ledger triggers fire by their
letter, and the frozen fingerprint would NOT catch it.** `sensor_geometry_fields` puts the declared
sensor NAMES and (under gain) the range VALUES in the tag, so adding `pressure` re-tags the entire
`minecraft_player` world space — every persisted world node of every existing minecraft_player
agent re-stales on load (H2 invalidate; fine for fresh agents, but say so for any operator
persistence). Both ledger rows carry "`minecraft_player` sensor-range change" and "SensorEncoder /
EC world-modality change" as Re-run triggers; a roster addition is the former by its letter. The
draft answers this with arm 3 (pressure, same pool — Exp 60 at n=12), which is the right
re-baseline, but it should be NAMED as the discharge of both rows' triggers in the outcome, and note
the gap: `exp60_run.py`'s frozen fingerprint pins `sensor_ranges` for only `is_in_water`/`oxygen`/
`saturation` — it does NOT pin the roster, so Exp 60's own harness would run unrefused under an
18-sensor body with a different tag. The mechanism PR should pin the declared roster (or the tag
string) in the fingerprint, or the "frozen apparatus" stop rule is blind to exactly this change. On
the arithmetic: at depth 4 the change to gate (ii) is +0.0005 (no re-run needed on the NUMBER);
the re-run is owed on the TAG.

**SF-5 — Extinction: state the same limit as Exp 61, and its cross-context sharpening.** There is no
positive writer on `_cluster_fear`, no tick decay, wall decay only on `load()` (7-day class). A
fear that reaches at pool 2 is never unlearned at pool 2 by safe exposure — the only thing safe
exposure writes is a positive `escape_water` link. Cross-context makes this bite harder than in
Exp 61: if the key is reached by shared mass (DNB-2's depth-12 regime) it is reached at EVERY
deep-enough water, and no amount of surviving there discounts it. §Not claimed should carry the
Exp 61 sentence verbatim plus that corollary.

**SF-6 — Arm 5 is guaranteed by construction; keep it as anti-vacuity, but do not present it as a
control that could have gone the other way.** A drive-less `modality: world` sensor has exactly one
consumer: the world encode. It is not in `_read_drive_states` (no `drive:` block), it has no entry
in `_DRIVE_CORRECTIVE_NEEDS` (temp/thermal/food/health only), it feeds no affinity, and yoked
propose-only training books no links (`reward_bias == {}` is pinned). With the fear subscriber
detached the agent carries: a re-tagged world cluster, hippocampal captures, Wire-2 valence — and
nothing that can select `escape_water`. NOT-ROUTE (≤ 0.10) cannot fail unless the harness is
broken, which is a fine thing to check at n = 3, not n = 12. The meaningful sensor-control is arm 3
(does the sensor break the within-pool result), and the meaningful drive-control does not exist
until 62b.

### NIT

**N-1 — The 62b deferral holds bio-faithfully, and is stronger than the draft says.** A pressure
DRIVE would be homeostatic (sem.py's `HomeostaticDriveSpec` docstring names "pressure recovery" as
the archetype: equilibrium at the surface, the environment pushes the value away, `set_point: 0` =
range midpoint passes H1 with no `rest:`; `drift_rate: 0` because the bridge writes truth, as
`oxygen` does). Its pain would be `drive:pressure` — which is OUTSIDE `DEFAULT_CLUSTER_FEAR_FAILURE_MODES`
(`{drive:health, drive:oxygen}`), so under the shipped allowlist it would write NO fear (a silent
no-op by design, W-5); admitting it is an allowlist change that re-fires both ledger rows and the
Exp 61 transport allowlist. And the D1 pain half is not a rule to route around: Minecraft deals no
pressure damage; the game's only depth cost is a longer ascent, which is more air-hunger — R3 D3's
"oxygen is the game's pressure analog" is the bio-faithful statement, since air hunger (not
baroreception) is also what drives surfacing in the animal. A pressure drive would be a hand-coded
pain for a cost the game already charges through a sensed drive. Recommend 62b be gated on a
game-native pressure COST existing (none does in 1.20.4), not merely on 62's result.

**N-2 — "Nothing in `src/` beyond the YAML line" understates the surface.** The bridge JS
(dev-side), `FakeBridgeServer` (lockstep-pinned to emit every sensor), the encoder golden
(`test_encoder_golden_v1` pins literal tags for every shipped space), the rest-neutral lint's
shipped-roster pin, and the fingerprint (SF-4) all move. The draft lists golden-pin regen; list the
rest so the mechanism PR's reviewers know the blast radius.

**N-3 — `12 = the classroom builder's maximum column depth`** is the right ceiling for the range
under the rules, but it fixes the weight curve of DNB-2: the sensor is only loud in the deepest
pool the builder can make. If the line ever needs a graded depth feature, the honest declaration
question is whether the world ever visits both sides of a rest — it does not for depth — and the
answer is that this quantity is a poor fit for the A4 world channel by construction, not by tuning.

## What I verified

- Wire-4 keys fear on the WORLD cluster only: `pain_bus.py::create_pain_cluster_fear_subscriber`
  reads `active_clusters(agent_id).get("world")`; `anticipatory_threat_need` mins across all noted
  modalities but nothing writes an interoception fear. `spec.py` raises on `modality: interoception`.
- `is_in_water` is eye-block-relative (`index.js` 107–120), i.e. already a surface-relative cue.
- `_sensor_embed` weight `w = (|v−0.5|·2)^3` for world; a `[-12,12]` rest-0 sensor at depth 4 has
  w = 0.037 (computed and replayed on the shipped function, not a mirror).
- Exp 60 live gate (ii) values (`exp60_geometry_2026-09-15b.json`): cos 0.7874; live contributors
  `is_in_water` 1.0, `light_level` 1.0, `time_of_day` 0.77, `distance_from_spawn` 0.16, `y_altitude`
  0.09; apparatus depth 5 (`exp60_water_apparatus.json`).
- Replay: cos(shore, submerged) with `pressure` at depth 4/5/12 = 0.7879 / 0.7881 / 0.6785;
  cos(pool-1 sub, pool-2 sub), shipped roster, dark pool 2: 0.9966 / 0.9919 / 0.9587 / 0.9049 /
  0.8804 (y=0) / 0.8657 (y=128) / 0.8878 (dist cap) / 0.8195 (y=0 + dist cap); pool-2 gate (ii)
  at the four extreme placements 0.8576 / 0.8559 / 0.8535 / 0.8869 — all FAIL; daylit pool 2:
  shipped 0.5973, +pressure@4 0.5966, +pressure@12 0.6988, ablated absolutes 0.5881.
- `lint_body_rest_neutral.py`: `[-12,12]` with `rest: 0` passes (midpoint 0); a homeostatic
  `set_point: 0` would also pass without `rest:`.
- `sensor_geometry_fields`: `declared_sensors` + `ranges` in the gained tag → roster addition
  re-tags the space; `exp60_run.py` fingerprint pins only three sensors' ranges, not the roster.
- `_DRIVE_CORRECTIVE_NEEDS` = temp/thermal/food/health; `DEFAULT_CLUSTER_FEAR_FAILURE_MODES` =
  {drive:health, drive:oxygen}; Exp 61 §Not claimed extinction wording; both ledger rows' Re-run
  triggers.
- R3 §D3's recording of the idea ("re-keys every cluster … as a DRIVE a route to air … held out on
  the confound") and `survival_world_1_3.md` D1 + topology ("features, not place").

## Verdict

**DO-NOT-BUILD** as drafted — the floor arm and the apparatus are mutually exclusive on the shipped
encoder (DNB-1), and the sensor is inert at the experiment's depth and a swamping constant at the
only depth where it is not (DNB-2). The replay the draft schedules as step 2 answers this today;
run it before anything else, expect it to confirm, and re-scope: the body already carries the
surface-relative cue, the honest body change is removing PLACE (arm 4, per `survival_world_1_3.md`),
and the real cross-context blockers (lighting, time) are graded cue-similarity — Phase 4's problem,
not a sensor's.
