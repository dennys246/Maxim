# Set-point-aware neutral — WIRING lens (four-lens design review, 2026-09-16)

**Design under review:** `docs/plans/deferred/setpoint_aware_neutral.md` (L11 line "B"). **Remit:** real
consumers and real seams (D43 — a fix ships with a CALLER), blast radius, persisted-substrate
compatibility, and whether any replay could pass while the live path stays unchanged.
**Read first:** `docs/experiments/DESIGN_REVIEW.md`, `docs/wiring/README.md`,
`cosine-separation-is-directional.md`, `harness-loop-must-be-proven-live.md`,
`sensor-range-clamps.md`, `cluster-dilution-blocks-situation-fear.md`.

## Verified first (what the wiring actually is today)

Every claim below was read in the source on branch `docs/exp60-outcome-earned`; nothing was
executed.

**The primitive and its config.**
- `src/maxim/similarity/encoder.py::_sensor_embed(sensors, ranges, dim, gain_exponent)` — the weight
  is `w = (abs(v - 0.5) * 2.0) ** gain_exponent`; `v = _normalize_value(value, ranges.get(name))`.
  There is NO per-sensor parameter of any kind: the only per-sensor, per-body information that
  reaches this function is the `ranges` dict.
- `encoder.py::SensorEncoderConfig` has five fields: `embedding_dim`, `min_delta`,
  `pattern_threshold`, `gain_exponent`, `gain_modalities`. It is a **per-modality** policy object,
  not a per-sensor one. A set-point is per-sensor-per-body. The plan's step 1 ("`SensorEncoderConfig`
  parameter") puts the value in the wrong object — see D2.
- `encoder.py::SensorEncoder.encode_sensors` applies gain only when `modality in
  config.gain_modalities` (`{"world"}`), stamps `ec.record_encoder_provenance("sensor:<modality>",
  {embedding_dim, sensor_names, normalization, declared_sensors, gain_exponent})`, and builds the
  geometry tag from `tag_fields = {encoder, modality, declared_sensors, normalization,
  embedding_dim} (+ gain="p3.0" when gained)` via `encoding_geometry_tag`. The delta gate bypasses
  only when values AND the `ranges` identity are unchanged (`self._last_ranges`).
- `src/maxim/similarity/ec.py` (D66 migration, the method whose docstring begins "Gate 1's
  specification is reject **or migrate**") re-derives a sensor tag from provenance in a SECOND
  site, and its own comment says: *"If `tag_fields` grows again, derive both from one helper — a
  wrong migrated tag is worse than none."* A set-point field grows `tag_fields`.

**The ONE live builder, and the ~12 harness builders.**
- Live: `src/maxim/runtime/agent_loop.py::run_agentic_loop` builds
  `SensorEncoder(ec=_ec, atl=..., nac=_loop_nac)` with **no `config=`** — the default
  `SensorEncoderConfig()`. Nothing in `runtime/bio_stack.py`, `runtime/bootstrap.py`,
  `runtime/agent_factory.py`, `embodied_runtime/`, `simulation/`, or `integration/` constructs a
  `SensorEncoder` or a `SensorEncoderConfig` (grep: zero hits). `simulation/minecraft_harness.py`
  documents this: "the loop builds its own `SensorEncoder` from `memory_hub.ec`". So a
  `SensorEncoderConfig` field has NO plumbing from any config surface to the live loop — it would
  be reachable only by editing the default.
- Harness/replay builders, all `SensorEncoderConfig()` default or no config:
  `scripts/survival_world/common.py::make_fresh_encoder`, `scripts/exp56/common.py` (BenchSession),
  `scripts/exp57/common57.py` (B-phase encode), `scripts/l11_real_trace_remeasure.py::_resolve_stream`,
  `scripts/survival_world/l11_geometry_probe.py`, `scripts/survival_world/r2_learned_bias.py`,
  `scripts/orient_backbone/exp53_cross_context_readout.py`, `scripts/rsc_precheck.py`,
  `scripts/r1_cross_layout_probe.py`, and seven `scripts/orient_substrate/*.py`. This is the
  shared-builder-activation shape: a config-field set-point would be silently OFF in all of them.

**Where per-sensor declarations flow today (YAML → schema → walk → encoder).**
- `src/maxim/embodiment/spec.py::_build_reading_schema` copies EXACTLY `range`, `shape`, `dtype`,
  `type`, `initial`, `modality` into `reading_schema`. **Any other key on a sensor is silently
  dropped** — there is no unknown-key validation. A `setpoint:` line in a body YAML today is a
  no-op that looks like it worked (the exact failure the `modality:` validator's comment names:
  "a sensor silently belonging to no channel — indistinguishable from working").
- `spec.py::_parse_entity` consumes `initial:` ONLY to seed `entity.vital_metrics[sensor] =
  float(initial)`, falling back to `(lo + hi) / 2` when absent. So `initial` IS the de-facto rest
  value, and its absent-default IS the A4 neutral. That coincidence is what makes today's
  "re-center the range around rest" lever work, and it is exactly what breaks "infer from
  `initial`" (D2 below). Also: `embodiment/archetype.py` synthesizes archetype sensors as
  `{"range": [0, 1], "initial": 1.0}` — rest at an EXTREME.
- Live walk: `agent_loop._read_declared_modality_ranges(executor, modality)` reads
  `reading_schema["range"]` (skips anything that is not `len(rng) == 2`), exposed as
  `_read_world_ranges` and registered on `ModalityChannel(WORLD_TAG, _read_world_states,
  _read_world_ranges)` in `agent_loop._SUBSTRATE_CHANNELS`. `ModalityChannel`
  (`embodiment/sensory_streams.py`) has exactly `tag`, `read_values`, `read_ranges` — no third
  reader. Both `propose_via_substrate` and `_encode_current_clusters` call
  `encode_sensors(..., ranges=ch.read_ranges(executor) or None)`.
- **Four independent range walks exist**, three of which are harness re-implementations:
  production `_read_declared_modality_ranges`; `scripts/exp56/common.py::sensor_ranges(root, names)`
  (reads `reading_schema["range"]`, `len == 2`); `scripts/exp57/common57.py::world_ranges()`
  (instantiates `BODY_REF57` via `ComponentRegistry`, same `len == 2` filter);
  `scripts/l11_real_trace_remeasure.py::_declared_world_ranges()` (raw `yaml.safe_load` of
  `minecraft_player.yaml`, bypasses `spec.py` entirely). `exp60_run.py`, `instrument_check.py`,
  `l11_geometry_probe.py` use the production `_read_world_ranges` (good).

**Persisted substrate and geometry.**
- EC (`ec.py::pattern_complete_or_separate`) masks stored nodes whose non-None geometry differs
  from the live tag and warns once per `(modality, stored, live)` via `_note_geometry_mismatch`,
  pointing at `maxim substrate invalidate --drop-geometry`. `hivemind/cli.py` invalidate prunes NAc
  cluster biases for dropped nodes (`prune_nac_cluster_biases`) and tombstones them.
- `hivemind/merge.py::ec_merge_aligned`: two nodes that BOTH carry a geometry and DIFFER never
  fold; unstamped fold by default; `strict_geometry=True` refuses unstamped. `hivemind/ingest.py`
  refuses unstamped foreign nodes at admission unless `allow_unstamped_geometry`, and gates on
  `body_ref` by NAME (`assert_bundle_body_compatible`) — the manifest carries
  `encoder_provenance` (`bundle.py`) but ingest does not compare encoder geometry to the
  receiver's live geometry.
- Bench bodies `minecraft_bench.yaml` / `minecraft_bench57.yaml` declare their OWN sensors (no
  `extends:` of `minecraft_player`), so an opt-in on `minecraft_player` does not propagate to
  Exp 56/57's bodies by inheritance. But `minecraft_player.yaml` is the body of Exp 58, Exp 60, R2,
  the L11 remeasure, `instrument_check`, `dark_danger_probe`, `loop_tick_probe`.

**Frozen fingerprints.**
- `docs/experiments/data/exp60_trials.jsonl` (20 rows) `frozen.fingerprint` = `{cluster_fear_alpha,
  max_cluster_fear, cluster_fear_threshold, cluster_fear_failure_modes, encoder_pattern_threshold,
  substrate_explore_bonus_weight, oxygen_drive{set_point, comfort_band}, sensor_ranges{is_in_water,
  oxygen, saturation}}`. **No `gain_exponent`, no geometry tag, no set-point field.**
  `exp60_run.py` computes `fingerprint_drift(live_fp, FROZEN["fingerprint"])` over those keys only.
- `scripts/exp56/common.py::FROZEN` and `scripts/exp57/common57.py` (via `C.FROZEN`) carry selector
  constants and slots — **no encoder field at all.**

**Replay scripts — shipped primitive or re-implementation?**
- Re-implementations (import only `_stable_basis`, hand-write `w = (abs(v - 0.5) * 2.0) ** P`):
  `docs/experiments/data/l11_slice2_cosine_check.py`, `exp60_oxygen_separation_check.py`,
  `exp60_oxygen_window_check.py`, `exp60_spawn_distance_check.py`.
- Shipped: `exp60_saturation_rest_check.py` calls `_sensor_embed` directly (with unit ranges);
  `l11_real_trace_remeasure._resolve_stream` goes through `SensorEncoder.encode_sensors` (real EC,
  register-on-separate) but with YAML-parsed ranges; `l11_geometry_probe.py` calls both
  `_sensor_embed` and `encode_sensors` with production `_read_world_ranges`.

**Ledger re-run triggers naming the encoder** (`docs/plans/behavioral_graduation_candidates.md`,
row by table line): 188 Exp 42 substrate-primary ("SensorEncoder / EC-interoception change");
191 Exp 52 operant orienting ("SensorEncoder/EC-modality change"); 193 Exp 53 cross-context
hardware readout ("`_encode_current_clusters` / `_sensor_embed` / EC `pattern_complete_or_separate`
change"); 194 Exp 56 ("`SensorEncoder` / EC world-modality change"); 195 Exp 57 (same);
196 Exp 60 ("`SensorEncoder` / EC world-modality change, `minecraft_player` sensor-range change").
Rows 183/185/187 say "encoder swap" (text encoder; not fired by this).

**Decision D1** the plan and the docstring cite lives in `docs/plans/archive/world_seam_1_1_4.md`
§Decisions ("ship the measured equation literally … a body whose set point is far from its range
midpoint rests loud; the body author's lever today is declaring the range around the set point"),
not in `docs/limits/l11_sensor_dilution.md` (which mentions set point only in bake-off prose).

---

## Findings

### DO-NOT-BUILD (as written)

**D1. The build order ships the pieces before the composition, and the live caller is in no
step.** The plan's five steps are: (1) declaration surface + `SensorEncoderConfig` parameter,
(2) offline replay harness, (3) code review, (4) "build the primitive change", (5) re-test Exp 58
vectors. Two composition defects:

- *The step-2 replay runs BEFORE the step-4 primitive exists.* A replay that precedes the
  primitive cannot call `_sensor_embed(setpoints=…)` — it must re-implement the weight, exactly as
  four of the five existing `*_check.py` scripts already do (`w = (abs(v - 0.5) * 2.0) ** P` copied
  by hand, importing only `_stable_basis`). A hand-copied variant can clear the composite bar while
  the shipped function is byte-for-byte unchanged. That is the shipped-the-pieces failure in its
  purest form (D43/D44), and the plan's own Q2 forbids it ("not a hand-built demo").
- *No step names the caller.* The only per-sensor information reaching `_sensor_embed` today is
  the `ranges` dict, produced by `agent_loop._read_declared_modality_ranges` and carried by
  `ModalityChannel.read_ranges`. A declared set-point reaches the LIVE encode only if (a)
  `spec._build_reading_schema` copies and validates it (today it is silently dropped), (b) a body
  walk reads it — a third `ModalityChannel` reader or a widened ranges walk, (c) `propose_via_substrate`
  and `_encode_current_clusters` pass it to `encode_sensors`, (d) `encode_sensors` passes it to
  `_sensor_embed` and into the provenance stamp + geometry tag. Steps 1 and 4 cover (a) and (d);
  (b) and (c) — the seam — appear nowhere. Grepping the new symbol across `src/` after steps 1–4
  would find one definition and zero non-test callers.

*Concrete change to the plan.* Reorder and merge into composition-shaped PRs:
1. **Primitive + passthrough + identity, inert:** `_sensor_embed(…, setpoints: dict[str,float] |
   None = None)` (normalized rest per sensor; `None`/absent sensor → today's 0.5, byte-identical),
   `encode_sensors(…, setpoints=None)` passthrough, provenance stamp + geometry tag + the `ec.py`
   D66 derivation ALL through ONE `sensor_geometry_fields(...)` helper (S1), and the golden
   byte-identical test (below) written against current `main` FIRST and required to pass unchanged.
2. **The caller:** `spec._build_reading_schema` reads + validates `setpoint:` (S6);
   `ModalityChannel` gains `read_setpoints` (or `read_ranges` returns a richer per-sensor record —
   pick one and change every reader); `_read_declared_modality_setpoints` for `world`; both
   `agent_loop` encode sites pass it. A test body YAML with one `setpoint:` sensor, encoded through
   `propose_via_substrate` on a fake executor, must produce a DIFFERENT node than the same body
   without the declaration — the strict red gate for "the declaration reaches the live encode".
3. **The replay, through the real path:** replay the Slice-1 / remeasure / Exp 58 vectors by
   calling `encode_sensors(setpoints=<read by the PRODUCTION walk from a body YAML>)`, never a
   hand-typed dict and never a re-implemented `embed()`. Only then decide.
4. Opt `minecraft_player` in (if the replay clears) + fingerprints (S2) + re-runs (S3).
Steps 1–2 are one PR or two adjacent PRs on one branch; they are not separable "capabilities".

**D2. "Infer from `initial:`" violates the plan's own DO-NOT-SHIP rule, and `SensorEncoderConfig`
is the wrong locus for the value.** The plan offers "YAML `setpoint`, or infer from `initial`" and
separately declares "a set-point default that shifts existing encodings is a DO-NOT-SHIP". Those
conflict:
- `initial` ≠ midpoint on shipping bodies: `minecraft_player.yaml` `light_level` range `[0, 15]`,
  `initial: 7` (normalized 0.467 vs neutral 0.5); every archetype-synthesized sensor is
  `range [0,1], initial 1.0` (`archetype.py`) — rest at an extreme. Inferring set-point from
  `initial` flips those sensors' weights with no declaration by the body author, so the world
  channel of every body with any off-midpoint `initial` re-encodes and its persisted world nodes
  orphan — the opposite of "byte-identical unless a body opts in".
- If the set-point also enters NORMALIZATION / the basis mix (the plan's "and/or the basis mix
  re-centers"), inference from `initial` touches the UNGAINED modalities too (interoception
  drives all carry `initial`; audio azimuth `initial: 0`), and the ungained geometry tags that
  `test_ec_vectorized_scan.py::test_ungained_modality_geometry_tag_is_byte_identical_to_pre_a4`
  pins would move — re-firing rows 188/191/193 (Exp 42/52/53, the last on the Reachy).
- `SensorEncoderConfig` is per-modality and has ONE live construction (`run_agentic_loop`, default
  config, no plumbing from `maxim config` or `bio_stack`) plus ~12 harness constructions that all
  take the default. A set-point field there is either a global default (not opt-in) or dead.

*Concrete change.* Strike "or infer from `initial`". The declaration surface is an explicit
`setpoint:` key on an entity-level sensor, required to be inside `range`, and NOTHING is inferred.
The value travels with the per-call body-derived dict (alongside `ranges`), not in
`SensorEncoderConfig`. Keep `initial` as what it is (the vital-metrics seed); document in the YAML
that `initial` and `setpoint` are different declarations that usually coincide.

### SHOULD-FIX

**S1. Geometry tag, provenance stamp, and the D66 migration must move together through one helper;
opting in MUST bump the tag; and the plan's "opt-in avoids orphaning" is wrong for the body that
opts in.** A set-point-weighted vector at the same `declared_sensors`/dim/gain is a different
SPACE — precisely the same-dimension hole `encoding_geometry_tag` exists to close. If the tag does
not include the set-point declaration, new-space embeddings pattern-complete onto old-space
centroids; `world` is frozen-centroid, so (quoting the D4 comment) "nothing is observably wrong".
So the tag must carry e.g. `setpoints={name: normalized_sp}` (sorted, float-normalized like
`gain="p3.0"`), the provenance stamp must record the same, and `ec.py`'s D66 derivation must emit
it too — via one helper, per that code's own instruction. Consequence the plan must state plainly:
the moment `minecraft_player` declares a set-point, every persisted `world` node of every
`minecraft_player` session is masked (Gate 1) and must be dropped with `maxim substrate invalidate
--drop-geometry <old>` (which also prunes their NAc cluster biases). That is contained (one body,
one modality) but it IS orphaning, and Exp 60's persisted sessions are in scope.

**S2. The frozen fingerprints cannot see this change — the stop rules would not fire.** Exp 60's
`FROZEN["fingerprint"]` carries `encoder_pattern_threshold` and three `sensor_ranges` but no
`gain_exponent`, no set-points, no geometry tag; `fingerprint_drift` compares only frozen keys.
Exp 56/57 `FROZEN` carry no encoder field. A `setpoint:` added to `minecraft_player.yaml` would let
`exp60_run.py` run to a verdict under a changed encoder with `fingerprint_live` reporting no drift.
*Change:* before any body opts in, add the live world geometry tag (computed through the same
helper as S1, from `_read_world_ranges` + the new setpoint walk) to the frozen fingerprints of
Exp 56, 57, and 60 so a declaration change REFUSES; and add `docs/wiring/` stub
`frozen-apparatus-hygiene.md`'s "config surface that silently changes substrate selection" entry
for set-points.

**S3. Price the re-run bill by row, and decide the ungained-modality question before building.**
Two distinct bills:
- *Byte-identical default, `minecraft_player` opts in, gain-weight only:* row 196 (Exp 60 — fires
  on BOTH "`SensorEncoder` change" and "`minecraft_player` sensor-range change"; 10 trials + the
  probe→run gate); row 193 (Exp 53 — its trigger names `_sensor_embed` change LITERALLY, so it fires
  even for an inert parameter; needs the Reachy, operator time; the ledger must either re-run or
  record a reasoned waiver citing the golden test); rows 194/195 (Exp 56 ~51 min + Exp 57 ladder —
  their literal trigger is "`SensorEncoder` … change"; the bench bodies do not opt in, so the
  honest record is "re-baseline OR waiver-on-golden-test", decided up front, not after).
- *If set-point enters normalization / basis mix (ungained modalities move):* add rows 188 (Exp 42,
  20 runs/config), 191 (Exp 52, embodied), and 193 for real, plus every interoception-keyed
  persisted substrate (cradle, infant_operant, reachy_mini) orphaning. The plan currently leaves
  this as open question 1; wiring-wise it is the difference between a contained change and a
  whole-ledger re-validation. Decide it as a design constraint: gain-weight only in v1.

**S4. Three harness range walks are copies; the replay measures the walk it uses.** `exp56
sensor_ranges`, `exp57 world_ranges`, and the remeasure's raw-YAML `_declared_world_ranges` each
re-implement `_read_declared_modality_ranges` with a `len(rng) == 2` filter. If set-points are
carried as a third tuple element, all three silently drop them; if carried as a separate dict, all
three never read them. Either way, those harnesses would encode the midpoint function while the live
loop encodes the set-point function — for the very experiments (56/57) the plan names as its
regression bar. *Change:* route the copies through the production reader (export
`_read_declared_modality_ranges`/`_read_declared_modality_setpoints` from `agent_loop`, or move them
to `embodiment/sensory_streams.py` where `ModalityChannel` lives), and extend
`test_place_code_wiring.py::test_value_and_range_walks_stay_in_lockstep`'s pattern with a
setpoints ⊆ ranges lockstep test.

**S5. Ingest cannot see a set-point mismatch between donor and receiver; the failure is silent
non-transfer.** `assert_bundle_body_compatible` compares `body_ref` by NAME; a bundle from
`minecraft_player`-with-setpoints ingested into `minecraft_player`-without passes gate 7. Then
`ec_merge_aligned` never folds (tags differ) — nodes INSERT as foreign; the receiver's live encodes
land in its own geometry, so the taught bias keyed on the donor's node is never reached. The only
signal is one deduped EC WARNING on the receiver's next encode. This is Exp 56's "coverage 0" shape
with no refusal. *Change:* the manifest already carries `encoder_provenance`; have ingest compare
the bundle's per-modality sensor geometry (derived via the S1 helper) against the receiver body's
live geometry and refuse (or require an explicit `--allow-geometry-mismatch`) on a difference.

**S6. Fail-loud seams for the three silent-break cases in the brief (plus two more).** Each must
raise at `spec._build_reading_schema` (parse time, `ValueError`, the same class as the
`modality:` typo guard), never downstream:
- `setpoint` outside `range` → normalized rest outside `[0,1]`; the sensor rests LOUD (weight > 0 at
  rest), the exact defect being fixed, invisibly. Raise.
- `setpoint` without `range` → the legacy range-blind map folds values and the set-point has no unit
  to normalize in. Raise ("setpoint requires range").
- `setpoint` on a modulator sub-sensor → the channel walk reads only entity-level sensors; silent
  no-op. Raise (mirror the existing sub-sensor `modality:` guard).
- `setpoint` on a sensor whose modality is UNGAINED (`interoception` via `drive:`, `audio`) → if v1
  is gain-weight only, a declared set-point does nothing. Raise unless the sensor is `modality:
  world` (or whatever `gain_modalities` names), so a body author cannot believe it is wired.
- Bundle mismatch → S5 (ingest), not parse time.

### NIT

**N1.** The plan cites "plan decision D1" through `encoder.py`; the decision text is in
`docs/plans/archive/world_seam_1_1_4.md` §Decisions, not in `docs/limits/l11_sensor_dilution.md`. Cite the
file so the bio-faithful lens reads the recorded consequence ("the body author's lever today is
declaring the range around the set point") rather than the bake-off prose.

**N2.** The plan's deliverable lists lenses "bio-faithful / confounding / wiring / regression";
`DESIGN_REVIEW.md`'s four are confounding / bio-faithful / wiring / environment. The "regression"
material is S3's re-run bill — fine, but name it as such so `rationale/` has the expected four
files (or an explicit fifth).

**N3.** The `min_delta` gate keys on raw values plus the `ranges` identity (`_last_ranges`). If
set-points ride in a dict passed alongside `ranges`, extend that identity check to it (a ranges-
unchanged, setpoints-changed call must not return the cached node). Carrying set-points INSIDE the
per-sensor record `read_ranges` returns gets this for free — one more reason to prefer that shape.

---

## Answers to the brief's direct questions

**Minimal caller set so a declared set-point reaches the LIVE encode (not just a replay):**
`spec.py::_build_reading_schema` (read + validate) → `agent_loop::_read_declared_modality_setpoints`
(or a widened `_read_declared_modality_ranges`) registered on `ModalityChannel` in
`agent_loop._SUBSTRATE_CHANNELS` → both `agent_loop::propose_via_substrate` and
`agent_loop::_encode_current_clusters` passing it → `SensorEncoder.encode_sensors` (accept, stamp
provenance, tag) → `_sensor_embed` (weight). For persisted correctness add the `ec.py` D66 derivation
via the shared helper. For the experiments to measure the same function: the three harness range
copies (S4) and the three frozen fingerprints (S2).

**Is "byte-identical default" mechanically checkable? Yes — propose:**
`tests/unit/test_setpoint_byte_identity.py`, written on current `main` BEFORE the primitive changes
and required to pass unchanged after: (i) a golden fixture of ~6 fixed sensor dicts (the Exp 60
shore/submerged vmaps from `exp60_saturation_rest_check.py`, a `[-1,1]` signed sensor, a range-blind
dict, an all-neutral dict) encoded through `_sensor_embed` with and without `gain_exponent`, pinning
`hashlib.sha256(repr(vec).encode())` per case (bases are SHA-derived, so the floats are process-
stable); (ii) the same dicts through `encode_sensors` for `world`/`interoception`/`audio` pinning
the exact geometry-tag STRING for each (today's `g........`), so an undeclared body's tag cannot
move; (iii) a declared-setpoint case asserting the tag DIFFERS and the D66 derivation from the
recorded provenance reproduces the live tag exactly (one helper, two call sites, equal). Tag rule:
undeclared → identical string; declared → must bump. The lockstep test (S4) and the parse-time
raises (S6) are separate red gates.

**What breaks silently, and where to fail loud:** see S6 (parse time) and S5 (ingest).

**Does the build order ship pieces before the composition?** Yes — D1. Step 1 is a config field with
no live plumbing, step 2 is a replay that must be a re-implementation because step 4 has not
happened, and the caller (the `ModalityChannel` walk + the two `agent_loop` encode sites) is in no
step at all.

---

## Verdict

**FIX-THEN-BUILD.** The mechanism is wire-able on existing infrastructure (the plan's front-gate
answer is right), but the plan as written would ship a per-modality config field with zero live
callers, validate it with a hand-copied replay that cannot call the shipped primitive, and offers an
`initial:`-inference option that breaks its own byte-identical rule — while under-pricing the
geometry-tag/D66, frozen-fingerprint, ingest, and ledger-re-run surfaces. Fold D1 (reorder so the
primitive, the caller, and a real-path replay land as one composition with a strict red gate) and D2
(explicit `setpoint:` only, carried with the per-call body dict, not `SensorEncoderConfig`), then
S1–S6, and it is buildable.

**Not verified:** I executed no encoder code and computed no tags or vectors (all claims are from
reading source on this branch); I did not open every `scripts/orient_substrate/*.py` range source
(they encode interoception/audio and are out of the gained path unless normalization changes); I did
not inspect any `~/.maxim/sessions/*` persisted state to count the world nodes an opt-in would
orphan; I did not evaluate the bio-faithful form of the weight (whether set-point belongs in the gain
or the basis mix) — only its wiring consequences under each answer; and I did not check whether
`minecraft_bench57.yaml` reaches `minecraft_player` through the component registry's `archetype`
resolution beyond confirming it declares no `extends:`.
