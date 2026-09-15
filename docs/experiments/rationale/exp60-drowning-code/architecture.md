# Exp 60 drowning-substrate — ARCHITECTURE lens (pre-merge CODE review)

Branch `feat/exp60-drowning-substrate` vs `main`. Lens: does the change fit the
codebase's contracts, invariants, and layering — ship with its callers, respect
bio-system boundaries, avoid silent divergence.

Diff: `bodies/minecraft_player.yaml`, `decisions/nac.py`, `simulation/minecraft_harness.py`,
`scripts/minecraft_bridge/index.js`, `tests/unit/test_exp60_drowning_substrate.py`,
`tests/unit/test_minecraft_harness.py`, `tests/unit/test_minecraft_seam.py`.

## Verdict: MERGEABLE after the two SHOULD-FIXes. No DO-NOT-MERGE.

The composition is genuinely wired end to end (not the D43 "shipped the pieces"
trap), the encoding change is provably inert at rest, the drive follows an
existing precedent exactly, and every lockstep fixture that enumerates the world
roster was updated. The gaps are a stale load-bearing comment, a dangling prereg
citation, and an undocumented (but correct) mechanism asymmetry.

---

## What is CORRECT (verified, not assumed)

**Composition ships with its callers (D43).** The learned-fear path is complete:
`minecraft_player.yaml` oxygen drive → `body.py::_publish` emits `failure_mode="drive:oxygen"`
(`f"drive:{drive_name}"`, body.py:411 — same unsuffixed form `drive:health` uses and
Exp 58 proved lands in `context["failure_mode"]`) → `pain_bus.py::create_pain_cluster_fear_subscriber`
passes it verbatim to `NAc.record_cluster_fear`, whose allowlist now contains
`drive:oxygen` (nac.py:403) → on a later underwater tick `agent_loop.py:1557` calls
`anticipatory_threat_need`, `agent_loop.py:1562` folds it into `drives["threat"]` by max →
`recommend_action` scores `_DRIVE_TOOL_AFFINITIES["threat"]` which now contains
`"surface"` (nac.py:578, substring-matches `minecraft_player_surface`). Every seam
between "drowning pain" and "surface action scored" has a real caller. This is NOT
the Exp-58 dead-read-path recurrence.

**Encoding is inert at rest — no persisted-substrate / frozen-centroid invalidation.**
`encoder.py::encode_sensors` → `_sensor_embed` sums a per-name SHA basis into a FIXED
`embedding_dim` (encoder.py:811-816). Adding a 17th sensor does NOT change vector
dimension. `is_in_water` rests at range-midpoint (declared `[-1,1]`, initial 0 → normalized
0.5), and under the A4 gain a neutral sensor contributes the ZERO vector
(encoder.py:817-834, the `_designed_rest` path). So for every non-water body/scenario the
world embedding is byte-identical to before — existing clusters are not perturbed, and any
frozen centroid stays valid. This is precisely NOT the class of concern that sank Slice-2
(that was a representation change to the discriminating sensors; this adds a discriminator
that is silent unless submerged).

**Missing-key tolerance — scripted bridges that don't serve `is_in_water` don't break.**
World values come from `Entity.vital_metrics` via `_read_declared_modality_states`
(declaration-driven from the body YAML), not from the raw bridge dict. A sensor a bridge
snapshot omits retains its `initial` (0 = neutral = silent). So `scripts/survival_world/dark_danger_probe.py::LIT_SAFE`
(serves `oxygen`, omits `is_in_water`) and any other scripted bridge is unaffected — no
KeyError, no divergence. The oxygen drive at value 20 = set_point (comfort_band 6) publishes
no pain there, so dark_danger_probe behaviour is unchanged.

**Bench bodies (Exp 56/57) are untouched and correctly so.** Those use separate
`minecraft_bench.yaml` / `minecraft_bench57.yaml`; only `minecraft_player.yaml` changed.
They should NOT get `is_in_water` — they have no water apparatus and a neutral sensor there
would be pure dead weight against the very cluster-dilution failure (L11) that body already
fights.

**Modality tag is legal.** `is_in_water` declares `modality: world` = `WORLD_TAG`, in
`sensory_streams.py::DECLARABLE_MODALITY_TAGS`. No name/tag allow-list rejects it.

**Drive shape fits the CC3-frozen spec.** `spec.py::_parse_drive_spec` reads exactly the
`homeostatic` fields the YAML provides (`set_point`/`drift_rate`/`comfort_band`/`pain_scale`);
`HomeostaticDriveSpec` is SHAPE-FROZEN at 1.0 and the YAML adds no new field. `drive:oxygen`
in the `frozenset` default on `NACConfig` respects the config's forward-compat posture (it's
a value in an existing frozenset field, not a shape change).

**No forbidden imports.** `pain_bus` still keys the `"world"` literal by value (the
documented proprioception-must-not-import-embodiment guard, pain_bus.py:607-610) — untouched.
The affinity table (`decisions/nac.py`), the affordance (body YAML), and the implementation
(bridge JS) sit in the correct three layers.

**Bio-faithful.** oxygen-as-world-sensor-plus-homeostatic-drive is the EXACT `health`
precedent (both are world-owned truth the bridge writes, `drift_rate: 0.0`, `comfort_band 6.0`,
homeostatic). "air-hunger" / "surface" are consistent framing. It does not newly blur the
world/interoception seam — that seam is already walked by `health` and the codebase documents
"a world sensor that ALSO carries a drive appears in both encodes" (agent_loop.py `_read_world_states`).

---

## SHOULD-FIX

**SF-1 — Stale load-bearing roster comment. `minecraft_player.yaml` lines 74-76.**
The comment reads "16 world sensors total; the bridge snapshot and FakeBridgeServer emit
every one (lockstep-pinned in tests). All bridge-written: no drives, no drift." This PR makes
BOTH clauses false: the channel is now 17 sensors, and `oxygen` (declared under this very
section) now carries a homeostatic drive. This is the exact silent-divergence shape the lens
guards — the comment is the human-facing anchor for the lockstep fixtures, and a future editor
counting "16" or trusting "no drives" will be misled. Minimal fix: update `16 → 17` and
correct/scope the "no drives" clause (oxygen is now a drive-bearing world sensor, mirroring
health/food above).

**SF-2 — Dangling prereg + missing four-lens design rationale.** The test docstring cites
`docs/experiments/exp60_drowning_avoidance_prereg.md`; neither it nor
`docs/experiments/rationale/exp60-drowning/` exists anywhere in the tree (tracked or
untracked). Per the project's experiment-provenance discipline (prereg on `main` before first
data; four-lens DESIGN review committed to `rationale/<slug>/<lens>.md` BEFORE the harness),
these substrate additions are the harness's foundation and their prereg should land with (or
before) them. A test that cites a nonexistent prereg is a provenance gap. Fix: land the prereg
and the four-lens design rationale in this PR series, or correct the citation to where they
actually live.

---

## NIT

**N-1 — Undocumented (but correct) innate-vs-learned asymmetry. `agent_loop.py::_DRIVE_CORRECTIVE_NEEDS`
vs the oxygen drive.** `health` has BOTH an innate corrective-need path (`("health","threat")`
in `_DRIVE_CORRECTIVE_NEEDS`) AND the learned cluster-fear path. `oxygen` deliberately has ONLY
the learned path — `_corrective_need_for("oxygen")` returns None, so a naive drowning agent
emits no innate "threat" need and must LEARN to fear the underwater cluster. This is the RIGHT
design (an innate oxygen→threat reflex would confound the anticipatory-learning claim Exp 60
tests), but nothing records the decision. Because the two drives look parallel for `health`, a
future reader may "helpfully" add `("oxygen","threat")` and silently confound the experiment.
Fix (cheap): one comment near the `_DRIVE_TOOL_AFFINITIES["threat"]` edit or the oxygen drive
stating the surfacing is intentionally LEARNED, not an innate corrective need. (This finding
sits on the SHOULD-FIX/NIT boundary — the confound it prevents is real.)

**N-2 — Weak test assertions. `test_exp60_drowning_substrate.py`.**
`test_surface_declared_on_the_body` asserts `"surface" in yaml.dump(data)` — a substring over
the whole dumped body, which would also pass on any description text containing "surface"
(e.g. "surfaced"). Prefer asserting the affordance key exists on the entity's affordance map.
`test_threat_need_matches_surface`'s `any(kw in tool for kw in ...["threat"])` is near-tautological
given the prior line. Minor; the structural guards are otherwise the right cheap locks.

---

## Rosters checked for lockstep (all others confirmed safe)
- `scripts/minecraft_bridge/index.js::snapshot` — UPDATED ✓
- `simulation/minecraft_harness.py::FakeBridgeServer` — UPDATED ✓
- `tests/unit/test_minecraft_seam.py` (2 rosters) — UPDATED ✓
- `tests/unit/test_minecraft_harness.py` (2 synthetic traces) — UPDATED ✓
- `scripts/survival_world/dark_danger_probe.py::LIT_SAFE` — NOT updated, and correctly so
  (declaration-driven read fills the missing sensor from `initial=0`, silent).
- `scripts/survival_world/l11_geometry_probe.py` — dynamic (`len(ranges)` from the body), no pin.
- `tests/unit/test_world_channel.py` — synthetic bodies, not this roster.
- bench bodies (`minecraft_bench*.yaml`) — separate rosters, intentionally excluded.
