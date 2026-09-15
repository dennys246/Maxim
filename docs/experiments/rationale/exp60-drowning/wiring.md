# Exp 60 (learned drowning-avoidance) — WIRING lens, DESIGN review

**Reviewer:** wiring lens (D43: a fix/experiment ships with its REAL caller + credit path;
verify with the real consumer, not a hand-composed probe).
**Date:** 2026-09-15. **Verdict: DO-NOT-BUILD as written** — two hard dead-path gaps
(no oxygen pain producer; no vertical/surface actuation) plus a cue-timing gap that makes the
"anticipatory" DV unmeasurable on today's wiring. Every one is the *exact* class Exp 58 hit
(dead read path, XZ-only flee, absent write channel). None is fatal to the idea, but all must be
built + LIVE-verified before freeze.

Files read: `docs/experiments/exp60_drowning_avoidance_prereg.md`; `docs/wiring/*`;
`scripts/survival_world/exp58_run.py`, `scripts/minecraft_bridge/index.js`,
`scripts/survival_world/setup_world.py`, `scripts/survival_world/common.py`;
`src/maxim/runtime/agent_loop.py::propose_via_substrate`;
`src/maxim/decisions/nac.py` (`recommend_action`, `anticipatory_threat_need`,
`record_cluster_fear`, `note_active_clusters`, `NACConfig`);
`src/maxim/proprioception/pain_bus.py::create_pain_cluster_fear_subscriber`;
`src/maxim/embodiment/body.py::_publish_drive_pain`;
`src/maxim/_data/components/bodies/minecraft_player.yaml`.

---

## DO-NOT-BUILD

### DNB-1 — No pain producer for oxygen deficit: the fear WRITE channel is dead
**Gap.** The `oxygen` sensor in `bodies/minecraft_player.yaml` (lines 94–98) declares **no
`drive:` block** — only `health` (homeostatic) and `food` (entropic) carry drives. Drive pain is
published exclusively by `body.py::_publish_drive_pain`, which fires only for a sensor with a
drive spec and stamps `failure_mode = f"drive:{sensor_name}"` (body.py:411). No oxygen drive →
no `PainSignal` for drowning → `create_pain_cluster_fear_subscriber` (pain_bus.py:591) is never
called → `record_cluster_fear` never runs → `anticipatory_threat_need` returns 0.0 forever. Both
FEAR and ABLATED arms would be behaviourally identical; the live-G2 analog (fear readable on the
probe cluster) can never pass.

The prereg calls this "a config addition to verify, not a new Wire" (§Why-viable bullet 2). That
**understates it**: it is a body-spec change that adds a drive to a world-owned sensor, and the
pain-topology lesson (`docs/wiring/pain-needs-declared-failure-modes.md`) is precisely that this
write path is subtle and was mis-diagnosed twice.

**Consequence.** The headline mechanism does not fire; the experiment measures nothing.

**Minimal fix.**
1. Add a homeostatic `drive:` block to the `oxygen` sensor: `set_point: 20`, `drift_rate: 0.0`
   (world-owned — the bridge writes truth, the drift loop must not fight it), a `comfort_band`
   below 20 (e.g. 4 → pain when bubbles < 16), `pain_scale`. Mirror `health`.
2. Add `"drive:oxygen"` to `NACConfig.cluster_fear_failure_modes` (currently
   `frozenset({"drive:health"})`, nac.py:399) — the allowlist is enforced inside
   `record_cluster_fear` (nac.py:3136), so without this entry the subscriber's write is a silent
   no-op by design.
3. Update the harness fingerprint (`exp58_run.py::FROZEN["fingerprint"]
   ["cluster_fear_failure_modes"]`) to the new allowlist, or the frozen-apparatus assert refuses.
4. LIVE-verify pain publishes: submerge the bot, tick `evaluate_failures`, assert
   `pain_bus.get_stats()["total_published"]` increments **while** `active_clusters(agent_id)
   ["world"]` is non-empty (the W-4 encode-before-pain ordering already holds in
   `propose_via_substrate`: `note_active_clusters` at agent_loop.py:1528 precedes
   `evaluate_failures` at 1543).

### DNB-2 — No executable SURFACE action: `flee` is XZ-only and can book "fled" while still drowning
**Gap.** The corrective act for drowning is **vertical** (get the head into air). The only flight
affordance is `flee` (bridge index.js:183), which pathfinds `GoalNearXZ(anchor.x, anchor.z, 2)` —
an **X/Z goal that ignores Y**. `move_to` is likewise XZ. Two failure modes:
- If the air/shore anchor is within ~2 blocks horizontally of the submersion cell, `GoalNearXZ`
  is *already satisfied* → the action returns `"fled to anchor"` instantly while the bot's head is
  still underwater. This is the vertical twin of the very trap the flee comment warns about
  ("a goto-to-where-you-stand resolves instantly and would book flight SUCCESS for doing
  nothing", index.js:192) — here it books success for *not surfacing*.
- `fm.canDig = false` (index.js:198) means the bot cannot break the water column's cap to escape;
  mineflayer pathfinder's vertical ascent through deep water via jump/swim controls is
  notoriously flaky and is not what `GoalNearXZ` optimises for.

So "reuse flee-up" (prereg §claim, §open-Q3) **does not exist** — flee never expresses "up".

**Front-gate (design-time scope pressure).** Does this need a *new* affordance? **Yes.** Existing
infrastructure (`flee`/`move_to`, both XZ) structurally cannot command a vertical escape, and the
read path selects an affordance by name — there is no place to hang "surface" on the current set.
A new param-free `surface` affordance is justified and in scope (it rides the same Wire-4 threat
read; it is not a new mechanism).

**Consequence.** Read path emits `flee`; bot doesn't leave the water; DV is noise; arms don't
separate. Exactly Exp 58's first-review dead-flee, replayed in the Y axis.

**Minimal fix.**
1. Add a param-free `surface` bridge action that swims straight up until the head block is air
   (`bot.setControlState('jump', true)` + optional horizontal nudge toward a recorded air anchor;
   poll head-block == air, cap the duration), and a matching `surface` affordance on
   `minecraft_player.yaml`.
2. **Wire it into the read path**: the fear need arrives as `drives["threat"]`
   (agent_loop.py:1562) and is matched against `_DRIVE_TOOL_AFFINITIES["threat"]`
   `= ("flee","hide","retreat","escape","withdraw","defend","shelter")` (nac.py:574). The string
   `"surface"` matches **none** of these — so a bare `surface` affordance scores zero, the DNB-2
   analog of Exp 58's "threat matched ZERO affordances". Either name the affordance to contain a
   listed substring (`escape_water`) **or** add `"surface"`/`"ascend"`/`"swim"` to the threat
   affinity tuple.
3. Ship an **actuation preflight** (the exp58 flee-check analog, exp58_run.py:293–306): one real
   executor `surface` call from a submerged start must raise `oxygen` back above the deficit / put
   the head block in air within a deadline, or the seed refuses before any measurement.

---

## SHOULD-FIX

### SF-1 — The only separable cue coincides with the deficit: the "anticipatory" DV can't isolate learning
**Gap (wiring dimension of an L11 problem — cross-ref confounding + bio-faithful).** The prereg's
viability rests on `oxygen` being a clean half-range swing (0.5→0.0) that separates the underwater
cluster. But in Minecraft oxygen stays **full (20) for the first ~15 s** submerged, then depletes;
drowning damage begins at air 0. So the *fresh-submerged, full-oxygen* state reads oxygen = 20 =
the surfaced value — it separates from shore only on the L11-diluted small-swing axes
(`on_ground` w=0.125, `y_altitude` ~0.09/range, `speed`) that `cluster-dilution-blocks-situation-
fear.md` already showed **cannot** clear 0.85. The cluster that *does* separate is the
**low-oxygen** one — which is the deficit/pain state itself, not a pre-deficit warning. Fear
booked there fires only once the bot is already low on air, so FEAR cannot surface "before deep
deficit" relative to ABLATED on cue onset; the learned component is at best a faster reaction to
the low-oxygen cue, not anticipation of it. The primary DV as framed ("surface earlier, before
deficit") is not measurable on this wiring.

**Consequence.** Even with DNB-1/DNB-2 fixed, the experiment may measure innate reactive
damage-avoidance, not learned anticipation — the confound the prereg's own §open-Q1 flags.

**Minimal fix.** (a) Add a MEASURED live cluster-geometry preflight (the
`scripts/survival_world/l11_geometry_probe.py` analog) that reports cos(shore, fresh-submerged)
AND cos(shore, low-oxygen) and identifies **which** cluster carries the booked fear — do not
assume oxygen separates a pre-deficit state. (b) Consider an apparatus that makes "in water"
separable **before** oxygen drops (e.g. a deep pool at a distinct `y_altitude` + a hostile-free
contrast so depth is the discriminator, keeping oxygen as the pain source but not the sole cue) so
the anticipatory cluster exists ahead of the deficit. This is a design change, not a tuning knob;
resolve it with the confounding lens before freeze.

### SF-2 — Apparatus not built: a `water_classroom` is a new setup_world.py build with its own instrument check
**Gap.** `setup_world.py::_classroom` builds a **dry** stone cave (`fill … minecraft:air` /
`minecraft:stone`); there is no water. A submersion apparatus needs `fill … minecraft:water`, a
forceload (the world-spawn relocation bug is already handled for the cave — reuse it), a recorded
submersion teleport target, and a reachable **air anchor** (shore/ledge) for the surface goal.
RCON `fill` + `forceload` support this; the mechanics are reusable. But three live unknowns must
be checked, not assumed:
- Teleporting the bot into a water cell depletes `bot.oxygenLevel` **only if the head block is
  water** (an air pocket at head height leaves oxygen full) — verify the head cell is water.
- Oxygen **recovers** on teleport to the air anchor (needed so the training latch can clear).
- Water physics don't bob the bot out of a 1-wide column on their own (would confound "surfaced").

**Consequence.** Without a build + instrument check, a silent apparatus failure (head in an air
gap; bot floats out) masquerades as a null.

**Minimal fix.** Add a `water_classroom` subcommand + a 60-second instrument check
(the `world-light-sensing.md` / `sensor-range-clamps.md` discipline): assert `bot.oxygenLevel`
drops below the deficit band within N seconds submerged and returns to full at the anchor, read
through `settle_until` on the SENSED value (range-clamp aware: `oxygen` range is [0,40], rest 20;
compare against `_read_world_ranges`, not raw truth), before any gated run.

### SF-3 — Drowning death-cap + rescue must gate on OXYGEN recovery, not just heal
**Gap.** exp58 trains propose-only (no execution) so the corrective read path isn't poisoned by
arm-asymmetric negative credit — reuse that. But drowning kills fast (~10 s after air 0), and a
propose-only bot never surfaces itself, so every training episode races the death cap
(`FROZEN["death_cap"]`). exp58's between-episode `_heal()` + teleport-to-safe + **settle the
SENSED health ≥ 18 before ticking healthy** (exp58_run.py:424–434) is the right shape, but here
the rescue must teleport to the **air anchor** and `settle_until(oxygen recovered)` — the oxygen
homeostatic latch clears only on an evaluation observing recovery (same hysteresis as health, per
`pain-needs-declared-failure-modes.md`), so a heal that doesn't restore oxygen leaves the latch
stuck and every subsequent "episode" mis-counts.

**Minimal fix.** Rescue = teleport to air anchor + `settle_until(vm["oxygen"] >= set_point − ε)` +
a few healthy propose ticks, before the next submersion; keep `keepInventory` /
`doImmediateRespawn` (already in `_GAMERULES`). Verify the death cap is reachable within the
episode timeout given drowning's damage rate.

---

## NIT

- **N-1 (measurement signal).** Define "out of water" as **head block == air** (immediate,
  read by the surface action itself), not "oxygen back to full" (bubbles refill lazily over
  ~seconds, so an oxygen-recovery DV inherits a lag that inflates latency). Measure from a real
  bot-state read (bridge snapshot `oxygen`, or an RCON head-block query), never a timer.
- **N-2 (self-poisoning audit carries over).** exp58 records `flee_negative_links`
  (exp58_run.py:497) to catch training self-poisoning; add the `surface`-affordance analog
  (`surface_negative_links`) to the record.
- **N-3 (positive finding — the threat read is body-agnostic and already wired).** The fear→
  `drives["threat"]`→`recommend_action` injection (agent_loop.py:1556–1562) needs **no**
  oxygen-specific affinity key: fear surfaces as the generic `threat` need regardless of which
  drive produced the pain. Only the *affordance side* needs the DNB-2 keyword. Do not add an
  `"oxygen"` entry to `_DRIVE_TOOL_AFFINITIES`.
- **N-4 (θ boundary, inherited).** `anticipatory_threat_need` returns `magnitude` iff
  `magnitude ≥ θ` (=0.5), and `recommend_action`'s activation floor is `drive_value > 0.5`
  (strict). A need of exactly 0.5 clears θ but not the floor — a 1-tick dead zone. K=10 × α=0.5
  saturates at 1.0 (2× margin), so this doesn't bite at the pre-registered episode count; keep K
  and α unchanged so fear lands well above 0.5.

---

## Build-order recommendation (cheapest gate first, per the L11 lesson)
1. Add the oxygen drive + allowlist entry (DNB-1) and the `surface` action + affinity wiring
   (DNB-2) — both are prerequisites for *anything* to fire.
2. Build `water_classroom` + instrument check (SF-2); run the live cluster-geometry probe (SF-1)
   BEFORE committing to the DV — if no pre-deficit cluster separates, redesign the apparatus or
   the DV rather than proceeding.
3. Actuation preflight (surface removes the bot from water) + death-cap/rescue on oxygen recovery
   (SF-3) — the exp58 preflight discipline, re-verified for the vertical act.
Only then freeze the prereg and build the trial harness.
