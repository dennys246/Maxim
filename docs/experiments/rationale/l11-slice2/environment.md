# L11 Slice 2 (channel-split) — ENVIRONMENT lens

**Design review of `docs/plans/l11_slice2_channel_split.md` (2026-09-15).**
Lens: is the thing the design must ultimately demonstrate REACHABLE and MEASURABLE in the
live world, game-natively (D1), and is the sole build-authorizing gate (Q7 live re-encode)
actually runnable? Read against the Slice-1 result
(`docs/experiments/data/l11_geometry_2026-09-15.json`), the classroom builder
(`scripts/survival_world/setup_world.py`), the Exp 58 harness
(`scripts/survival_world/exp58_run.py`), and the two live-sensor wiring lessons.

Verdict summary: the classroom is genuinely re-stageable, but the **gate as pointed cannot
tell "the split worked" from "the propped mob was adjacent this encode"**, and the **Exp 56/57
re-baseline (step 7) is a separate apparatus the plan treats as a re-run**. Two DO-NOT-BUILD,
three SHOULD-FIX, two NIT.

---

## DO-NOT-BUILD

### DNB-1 — The sole build gate rests on ONE propped sensor, with no control that isolates it from the apparatus

Q7 points the build authorization at "the exact cluster-distinct preflight that refused
Exp 58." That preflight is a **single-staging pairwise `dark_cluster != safe_cluster`**
(`exp58_run.py` L288–292) — not the composite `min(separation, stability, discrimination)`
bar the diagnostic prereg promised (`l11_world_channel_diagnostic.md` §"Pre-registration",
§5), which the diagnostic itself declared *necessary-but-not-sufficient* and error-prone in
both directions (fold-point 1).

Worse, the Slice-1 data shows the proposed `world:threat` channel
(`nearest_hostile_dist, hostile_count, y_altitude, distance_from_spawn, nearest_player_dist`)
is **~1 varying sensor + 4 constant** on this classroom:

| threat-channel sensor | norm Δ (safe→dark) | moves? | note |
|---|---|---|---|
| `nearest_hostile_dist` | 0.0915 | yes | the ONLY discriminator — and it *is* the hand-placed persistent clustermob (~19 blk safe → ~2 blk dark) |
| `y_altitude` | 0.037 | no (< MOVE_EPS) | 12-blk pit; clamped over [0,128] → barely moves |
| `distance_from_spawn` | 0.0254 | no | safe/dark only ~21 blk apart in x |
| `hostile_count` | 0.0245 | no | counts-all-loaded-through-walls artifact (see NIT-2) |
| `nearest_player_dist` | 0.0 | no | 0.5 constant — there is no second player in solo survival, ever |

So a green live gate would certify the split on the strength of **one sensor whose contrast
is manufactured by a prop we placed** (`summon … Tags:[exp58clustermob]`), with no
hostile-absent arm to show the channel reads *danger* rather than *this mob's coordinates*.
That is the Exp-58 / Phase-0 false-confidence shape ("validated an easier problem than the
live classroom") reproduced one level down — the exact trap the diagnostic-first discipline
was created to stop.

**Consequence:** the split could pass its sole gate and still be reading the apparatus, not a
transferable dark=danger want; the L11 line's whole point (a want that transfers) would rest
on a certified artefact.

**Minimal fix:** redefine the Q7 gate to the **composite `min(sep,stab,disc)` bar over ≥N
repeated stagings AND a hostile-present vs hostile-absent contrast** (the prereg grid
`× {hostile present, absent}` that Slice-1 never actually executed — it captured only
safe/dark, 30 each). If `world:threat` separation collapses when the clustermob is removed,
the channel is reading the prop and the build is not authorized. Do **not** ship the pairwise
`!=` as the authorization.

### DNB-2 — Exp 56/57 re-baseline (step 7) is a separate apparatus + a new design decision, not a re-run

The bench bodies the taught-want experiments use **do not contain the sensors the split is
built around**:

- `minecraft_bench` (Exp 56) world sensors = `y_altitude, distance_from_spawn, speed,
  on_ground, time_of_day`.
- `minecraft_bench57` (Exp 57) world sensors = `offset_x, offset_z`.
- **Neither has `nearest_hostile_dist`, `hostile_count`, or `nearest_player_dist`** — the
  entire `world:threat` channel is empty there.

So step 7 is not "re-run under the split." It requires: (a) a **new sub-grouping design** for
the bench sensor rosters (the threat/env/self partition is undefined when threat is empty and
`offset_x/offset_z` fit no proposed sub-tag); (b) standing up the **1.16.5 / temurin@11**
bench servers — a wholly separate apparatus from the 1.20.4 / temurin@17 survival classroom
(`setup_world.py` header: "`scripts/exp56/` stays 1.16.5-pinned until its own re-baseline
port"); (c) the persisted-substrate migration the geometry-tag change forces on the
`world`-keyed taught wants (diagnostic W5/DNB-3); (d) ~51 min/arm × two experiments of live
time. The plan compresses all of this into one build-order line after the survival gate.

**Consequence:** the regression gate the entire L11 line has been gating on is unbudgeted and
under-designed; the realistic failure is the split shipping with a hand-waved or skipped
Exp 56/57 check — exactly the "necessary, not sufficient" gap the diagnostic flagged.

**Minimal fix:** promote step 7 to its **own scoped sub-plan drafted before the split
merges** — naming the bench sub-grouping, the 1.16.5 apparatus stand-up, and the migration —
and state explicitly that survival separation alone does not authorize the ship. This is the
plan's own Q6 escalated: treat it as C's first gate, not its last line.

---

## SHOULD-FIX

### SF-1 — The gate runs on the classroom box, and the split must be DEPLOYED there first

The Slice-1 record's provenance shows capture ran on `/Users/dennys/RMSrv/scripts/Maxim`
(the big-mac-mini classroom box), **not** the dev checkout. The live re-encode gate therefore
executes wherever the standing server + bridge + `~/.maxim/exp58_classroom.json` live. The
split's re-tag + the `.get("world")` rewires (SF-2) must be **deployed to that box**
(`maxim peer update && restart`, or git pull) before the gate can measure anything. Plan
step 6 never says so.

**Consequence:** the gate silently runs the pre-split code (universal "same cluster" refusal,
mis-read as "apparatus can't separate"), or a half-deployed mix. Same family as the
run-the-wrong-checkout provenance lesson.

**Minimal fix:** make step 6 assert the running box's `executed_git_hash` equals the split
commit before the gate result counts (the probe already stamps `executed_git_hash` /
`in_process_code_provenance` — assert it, don't just record it).

### SF-2 — The gate is un-runnable until the `.get("world")` rewire targets `world:threat`, and the failure is a SILENT refusal

After the split, `_encode_current_clusters(...)` returns keys `world:threat / world:env /
world:self` and **no `world`**. Eight live-gate-relevant sites read the old key:
six in `exp58_run.py` (L280, L284, L323, L411, L452, L456), plus
`instrument_check.py:235` and `dark_danger_probe.py:224`, plus the geometry probe's own
`analyze` (`modality="world"`, L294). Each returns `None` → exp58's `if not dark_cluster_pre`
refuses **every** seed. This is the plan's Q5 (wiring) but its *environmental* consequence is
that the sole gate degrades to a silent universal refusal that looks like a world problem, not
a wiring problem.

**Consequence:** the gate either no-ops to "nothing separates" or, half-rewired, reads the
wrong sub-channel — indistinguishable from a real negative.

**Minimal fix:** step 3 must re-point exp58's preflight (and the probe's analyze) at the
**`world:threat`** sub-tag specifically, and step 6 must smoke-assert that
`_encode_current_clusters` returns the expected sub-keys before the gate result is trusted.

### SF-3 — "Re-encode safe vs dark" under-states the full apparatus the exp58 preflight needs

The plan phrases Q7 as a light re-encode, but "the exact preflight that refused Exp 58" is
embedded in a trial that also requires: the offline gates record with `all_pass=true`
(`exp58_offline_gates.json`), the frozen-fingerprint match, the bridge started **with
`--flee_x/--flee_z`**, `setup_world.py prepare` (food ≥16 satiation settle), a flee-actuation
climb, and the forceload+clustermob-verify from the classroom build. A missing gates record or
flee anchor refuses *before* the cluster check ever runs.

**Consequence:** operator under-scopes the gate; a settle/anchor/gates-record miss reads as a
gate failure.

**Minimal fix:** prefer extracting a **standalone live cluster-distinct re-encode** — server +
bridge + classroom only, the live agent's fresh EC, the composite bar from DNB-1 — as the gate
(the cluster check at exp58 L280–292 needs no food/flee), and enumerate the full staging if the
embedded-in-trial form is kept.

---

## NIT

### NIT-1 (reassuring) — Re-staging IS reachable; the clustermob-relocation failure is closed

`setup_world.py classroom` is idempotent and rebuilds **in place at the recorded anchor**;
safe/dark staging is teleport-only, so **no new world-building** is needed for the re-encode.
The Exp-58-saga relocation bug (clustermob teleporting to world spawn) is closed for the
re-encode: the build `forceload add`s the pit chunks *before* the summon, and an in-build
`execute if entity … distance=..3` **refuses the build** if the clustermob isn't within 3
blocks of its pit cell. The mob is `NoAI:1b, PersistenceRequired:1b, Silent:1b` — it will not
wander or despawn. Re-stageability is solid; the risk is measurability (DNB-1, SF-2), not the
world.

### NIT-2 — Drop `hostile_count` from the "discriminating" channel (or flag it in the record)

`hostile_count` counts all loaded mobs through walls (`world-light-sensing.md` trap;
Slice-1 Δnorm 0.0245, no move) — identical at safe/dark, an environment artefact. Leaving it
in `world:threat` is dead weight and mislabels the channel as richer than it is. The
diagnostic's own open-item #3 already flagged this. Exclude it, or note the artefact in the
decision record. (Same caution for `nearest_player_dist`: permanently 0.5 in solo survival —
it can never carry contrast in this apparatus.)

### NIT-3 — Capture-arm sweep gap (only if the gate uses the probe, not exp58's preflight)

`l11_geometry_probe.py capture` does **not** sweep, so spawner-spawned AI zombies accumulate
and (having AI) can path toward the safe chamber, adding noise to `nearest_hostile_dist` at
safe. exp58's preflight sweeps first (sparing the clustermob), so it is cleaner. If the gate
is built on a fresh probe capture rather than exp58's swept preflight, add a per-sample sweep
so the safe reading is the clustermob at ~19 blk, not a wandering spawn.
