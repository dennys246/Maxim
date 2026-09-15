# Exp 60 drowning-substrate — EXECUTOR-lens code review

Branch `feat/exp60-drowning-substrate` vs `main`. Lens: does the code DO what it
claims, correctly, at runtime? Findings ranked, `file::line` cited from the diff /
current tree.

## DO-NOT-MERGE
None.

## SHOULD-FIX

### S1 — `is_in_water` fires on feet-wet contact, contradicting the head-submerged intent (and disagrees with `surface()`'s own water test)
`scripts/minecraft_bridge/index.js` snapshot (`const inWater = (me && me.isInWater) || (headBlock && ...)`).
The comment claims the signal "tracks the drowning situation, not just feet
getting wet" and the primary cue is the HEAD block — but the `me.isInWater ||`
disjunct is not a rare version fallback: mineflayer's `entity.isInWater` is true
for ANY bounding-box/water intersection, i.e. it fires when the bot is merely
wading (feet wet, head in air, oxygen == 20, NOT drowning). Result: a shore-
adjacent shallow-water state reads `is_in_water = 1` with full oxygen, which is
exactly the cluster-contamination the L11 separability work is trying to avoid —
the "underwater cluster" would then include non-drowning wading states.

Worse, the two "in water" definitions on this branch DISAGREE: `surface()`'s
`headWater()` helper checks the head block ONLY (no `isInWater` OR), while the
snapshot sensor ORs in `isInWater`. So the sensor can read 1 while `surface()`
returns "already at surface" for the same pose.

Minimal fix: drop the `me.isInWater ||` disjunct so the sensor is head-block-only
(matching both the stated intent and `surface()`); if `isInWater` is genuinely
needed for flowing/waterlogged edge cases, reconcile the comment and apply the
same rule inside `surface()`. Practical blast radius for the intended deep-pool
apparatus is small (the bot is dry-shore or fully-submerged, rarely wading), so
this is SHOULD-FIX, not a blocker — but the intent/behavior mismatch is real.

### S2 — `"surface"` threat keyword false-matches the `climb_surface` affordance
`src/maxim/decisions/nac.py::_DRIVE_TOOL_AFFINITIES` (`"threat": (..., "surface")`).
The substring match (`nac.py` `if keyword in tool_lower`) is name-only, and no
minecraft tool other than `surface` contains the substring — so for Exp 60 this
is correct. BUT the affinity table is GLOBAL, and `climb_surface`
(`src/maxim/_data/components/creatures/alien_xenomorph.yaml::climb_surface`)
contains `"surface"`. On any xenomorph body with a threat need, the threat need
would score `climb_surface` (a locomotion affordance) as a defensive action.
This is the exact false-match class the table's own comment already guards
against ("No 'block' keyword — it false-matches place_block/mine_block").
(Checked: `giant_spider` uses `climb` with a `surface` PARAM — params don't enter
`tool_lower`, so no match there; only `climb_surface` collides.)

Minimal fix: anchor the match for this keyword (e.g. exact-name affinity, or
require the tool to end in `_surface`), or rename the body affordance to something
without the collision. At minimum, note the collision in the table comment.

### S3 — `surface` no-op returns a success string ("already at surface") → Exp-58 mis-credit risk in a tool-success crediting path
`scripts/minecraft_bridge/index.js` surface case (`if (!headWater()) return "already at surface";`).
All three returns ("already at surface", "surfaced", "surface: still submerged
(capped)") are non-exception → tool-success. If the confined/propose-only
training path credits a proposed action on tool-success (rather than on drive
progress), then proposing `surface` on dry shore (a no-op) is rewarded as though
it did something — the Exp-58 self-poisoning family. Mitigant already present:
value-based `drive_comfort_progress` (embodiment/sem.py) is self-correcting — a
dry-land surface yields ZERO oxygen deviation-reduction, so a progress-based
credit path gives 0. So this only bites if the (unbuilt) harness credits on
tool-success. Recommend the harness credit oxygen-progress, or return a
distinguishable no-op marker so "already at surface" cannot be scored as a
successful threat response.

## NIT

- **N1** `surface()` treats `bot.entity == null` and `blockAt == null` (unloaded
  chunk) as "head clear" → `headWater()` returns false → loop stops early and the
  final `return headWater() ? ... : "surfaced"` can report "surfaced" spuriously.
  Safe (jump IS released in `finally`), but a transient null mid-dive ends the
  swim-up early. For a bot actively submerged near spawn the chunk is loaded, so
  low-risk.
- **N2** `FakeBridgeServer` sets `is_in_water: float(r.random() > 0.9)`
  (`src/maxim/simulation/minecraft_harness.py`) uncorrelated with `oxygen`. Fine
  for seam/shape tests; any test expecting an underwater→low-oxygen coupling would
  need a real fixture, not this fake.
- **N3** `test_drive_oxygen_books_fear_on_the_underwater_cluster` calls
  `record_cluster_fear` directly, so it does NOT exercise the real
  PainSignal→allowlist→booking path. I verified that path by hand (see V2); the
  test would not catch a regression where the PainSignal `context["failure_mode"]`
  string drifted away from `"drive:oxygen"`.

## VERIFIED-CORRECT (checked, no action)

- **V1 — oxygen scale: NO mismatch.** The drive-pain path reads the RAW sensor
  value (`embodiment/body.py` `current = readings.get(ds_name)`) and
  `drive_pain_for_value` (`embodiment/sem.py`:
  `min(1, (|value-set_point|-comfort_band)*pain_scale)`) uses `set_point=20`
  against that raw value. The bridge writes `oxygen: bot.oxygenLevel ?? 20`
  (`index.js`), i.e. 0–20, so rest (surfaced) == 20 sits EXACTLY at `set_point`.
  The body `range: [0,40]` is used only for substrate encoding (midpoint 20 =
  A4-neutral → surfaced is silent); it does NOT enter the pain formula. Only the
  lower half [0,20] of the range is ever visited — by design, like
  `hostile_count`'s negative half. `comfort_band=6` → pain fires below 14 bubbles;
  `pain_scale=0.5` → saturates to 1.0 at oxygen 0 (|0-20|-6=14, ×0.5=7 → clamp 1).
  Sensible for a 20→0 depletion. The cluster-fear subscriber's 0.3 intensity floor
  → booking begins around oxygen ≤ 13.4 (early in a ~15s dive). Correct.
- **V2 — `drive:oxygen` allowlist booking works.** `_publish_drive_pain`
  (`embodiment/body.py`) ships `context["failure_mode"] = f"drive:{drive_name}"`
  == `"drive:oxygen"`, which matches `NACConfig.cluster_fear_failure_modes` and
  passes the gate in `pain_bus.py::create_pain_cluster_fear_subscriber` →
  `NAc.record_cluster_fear` (allowlist enforced in `record_cluster_fear`). The
  `":discomfort"` suffix is only on the FailureEvent NAME and the (dormant) SCN
  event — NOT on the PainSignal `failure_mode` — so there is no suffix mismatch.
  Mirrors `drive:health` exactly.
- **V3 — swim-up control.** `bot.setControlState("jump", true)` IS swim-up while
  submerged in mineflayer, and it is ALWAYS released — the release is in a
  `finally` wrapping the await; the `setControlState(true)` sits before the `try`,
  so a throw there simply never sets it. The 8s cap (`waited >= 8000`) prevents a
  hang in a walled/air-unreachable column.
- **V4 — head cell.** `offset(0, 1, 0)` selects the block spanning y∈[1,2], which
  contains a 1.8-tall bot's eyes (~1.62) — the correct cell for Minecraft
  breathing. `bubble_column` is correctly counted as underwater. Both the sensor
  and `surface()` null-guard `blockAt` (`headBlock && ...` / `b && ...`) and
  `bot.entity` (`me ?` / `if (!e) return false`).
- **V5 — reachability of `surface`.** Not scored via an `oxygen` affinity (there
  is none); it is reached through the `threat` need raised by learned cluster-fear
  on the underwater cluster, which the drive:oxygen pain books. Chain is intact.
