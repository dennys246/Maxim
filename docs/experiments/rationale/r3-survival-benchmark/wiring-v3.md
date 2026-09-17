# R3 — WIRING lens, DELTA re-run on v3 (2026-09-17)

Reviewed: `docs/experiments/r3_survival_benchmark_prereg.md` (DRAFT v3, branch `r3/v3` @ `e78dc2e9`)
against the pilot that v3 folded — `scripts/survival_world/r3_pilot.py`, its five rows
`docs/experiments/data/r3_pilot_2026-09-17.jsonl` (stamped `executed_git_hash` `5d1e6a62`), the offline
test `tests/unit/test_r3_pilot.py`, `scripts/survival_world/scripted_water.py`, and the apparatus the
harness will extend (`water_trial.py`, `agent_loop.py`, `simulation/minecraft.py`, `minecraft_bridge/index.js`).
Predecessor: `wiring.md` (v1). This is the DELTA the v1 review asked for: what the pilot discharged,
what the harness still owes, and what the pilot's own code teaches `WaterTrial.lethal_event`.

The short version: the pilot discharged the two DO-NOT-BUILDs' factual halves and SF-3's *continuation*
half (the loop ticks through death; respawn is read 0.4 s after; `deaths` +1 exactly). It did NOT
discharge SF-4 (the silent zero is unchanged), and its `live_window` — the prototype v3 names for
`lethal_event` — carries four things the harness must not inherit: the `finally` order (join BEFORE the
shore teleport, the opposite of `loop_window`), a tick clock that is ≈ 0.75 s off the window clock and
unrecorded, a same-tool-cap story that the rows contradict, and a scripted bridge that cannot reach the
death branch at all. One provenance fact for the freeze: #755 was squash-merged, so the hash every pilot
row stamps is reachable only from a side branch, and `r3/v3` itself forked BEFORE #755 — the pilot file
on this branch is not the file that produced the rows.

---

## Findings

### DO-NOT-BUILD

None at the design level. v3's unit (one lethal event per fresh agent), arms, DVs and the depth axis are
consistent with what the pilot measured and with the apparatus as built. Everything below is harness.

### SHOULD-FIX

**SF-A — `live_window`'s `finally` joins the loop BEFORE the shore teleport; a surface-ended event leaves
an agent that sinks back within ~1 s and is exposed for the whole join. `lethal_event` must use
`loop_window`'s order (teleport first), minus the heal, with NO linger.**
*Failure scenario:* arm-B agent surfaces at 8.4 s (the DV), the harness lingers/joins, the agent sinks
(mineflayer sinks without input), takes oxygen pain or drowning damage AFTER the DV, `fear_after` /
`links_after` / the persisted object carry post-DV learning, and in the worst case `deaths()` at the end
reads +1 on an event whose samples say "surfaced" — two detectors disagreeing silently.
*Evidence:*
- `r3_pilot.py::Pilot.live_window` `finally`: `stop.set()` → `loop.join(timeout=20.0)` → `stop_motion()` →
  `rcon.teleport(shore)` → `reopen_hub_session()`. `water_trial.py::WaterTrial.loop_window` does the
  opposite on purpose: `teleport(shore)` "rescue BEFORE the loop drains", then `stop.set()` + join.
- The re-sink is MEASURED in both lethal rows: lethal_B `in_water` False at 8.377 (y 38.85), True again
  at 9.502 (y 37.69, oxygen 13 → 12 at 10.26); lethal_A False at 27.839 (health 9, oxygen 0), True again
  at 28.980 (y 37.75, oxygen 3, health 10). lethal_A was ≈ 2 hp-seconds from a post-DV death during the
  linger; the join that followed (in-flight `escape_water` ≤ 8.6 s + the `consolidation="full"` close)
  is unstamped.
- The row's `deaths_delta` is computed AFTER the join (`trial.deaths() - deaths0` at return), while
  `t_surface`/`t_death` come from the samples — two reads of one fact on two clocks.
- The 2 s `linger_after_stop_s` is what let the post-surface oscillation run: the 3 "extra" escape
  successes per lethal row (8.85/9.81; 28.44/29.36) are NOT `"already at surface"` no-ops — they are
  re-escapes from re-submersion (the samples show the agent back in water between them). The 7 positive
  links after one event (`links_after`) are 1 real escape + 2–3 re-escapes, not linger no-ops.
*Fix (harness):* `WaterTrial.lethal_event` ends the window at the FIRST sample whose `until` holds; on
SURFACE → `rcon.teleport(shore)` immediately (before `stop.set()`), no heal, no linger; on DEATH → the
respawn already did it (assert position = shore); on CAP → teleport + Refusal. DV fields come from the
samples ONLY; a post-window `deaths()` rise is an `InstrumentError` ("death after the DV"), never a DV.
Stamp `t_join_s` and the close duration per event (v1 SF-7's owed join budget). A `call` whose return
time is after the teleport and whose `detail` is `"surfaced"` is flagged `surfaced_by_harness` (see SF-B).

**SF-B — The same-tool cap did NOT fire "under the stone cap"; it fired on the post-respawn shore
snowball. Under the cap the limiter is the 8 s BLOCKING call, and the escape in flight at death returned
`"surfaced"` `ok: true` — a success booked on a call that ended in death.**
*Evidence, the rows:* drown_on `escape_water` calls at 7.158 / 15.284 / 23.376 / 31.505 — 8.1 s apart,
i.e. each is the bridge's 8 s jump cap (`index.js` `escape_water`: `waited >= 8000` → `"surface: still
submerged (capped)"`), and the loop thread sat inside `MinecraftClient.call_action` for all of it (drown
rows: 25 ticks in ~39 s vs 49 in 29.7 s for lethal_A). Only FOUR escapes ran before death at 32.988; the
one started at 31.505 was IN FLIGHT at death and returned ≈ 33.6 (next call 33.666) — respawn put the head
in air, `headWater()` read false, the bridge returned `"surfaced"`, the spy recorded `success: True`, and
the NAc booked a positive link on a drowning. The cap then fired on the 6th consecutive escape = the
SECOND post-respawn call: the gap 33.666 → 35.469 (1.8 s vs 0.75 s between neighbours) is exactly one
dropped proposal; drown_off shows the same gap 26.652 → 28.173. v3 §Pilot item 4 ("under the stone cap
… a capped escape that keeps failing loses proposals to that cap") mis-times this.
*What the cap does* (`agent_loop.py` execution stage, `_MAX_CONSECUTIVE_SAME_TOOL = 5`): counts
consecutive proposals reaching execution with the same `(tool_name, hash(params))`; when the count
EXCEEDS 5 (the 6th), it drops THAT proposal (`ctrl.pending_proposal = None`), appends a
`consecutive_tool_cap` outcome to `recent_outcomes`, resets all three counters to zero, `continue`s. So:
5 executed, 1 dropped, 5 executed … — a 5/6 duty cycle (v1 N-3), never a chain reset. A different tool
in between (the failed `flee`) resets it.
*Can it bite a real event at ≤ 12 blocks?* No. The first escape at depth 12 needs ≈ 12 × 0.43 = 5.2 s +
0.6 s hold < 8 s, returns `"surfaced"` in one call; the cap needs five prior identical executions. It can
only bite a CAPPED escape (depth > ~17 blocks, unbuildable: `WATER_MAX_DEPTH` 12) — and there the 8 s
blocking call, not the cap, is what removes proposals.
*Fix (harness):* `attach_instruments`' spy must record the bridge `detail` per call (v1 N-5: a `capped`
escape is a SUCCESS to the executor) AND the call's RETURN time; `lethal_event` flags any call whose
return is after `t_death`/`t_teleport` (`surfaced_by_respawn` / `surfaced_by_harness`) and reports the
positive link it booked as apparatus, not behaviour. Declare in the prereg that a capped escape holds the
loop for 8.6 s with no ticks (the "no proposals" window in a drowning is the call, not the cap).

**SF-C — SF-4 is NOT discharged: `deaths()` still returns 0 on a parse failure; the scripted bridge
always answers 0 and deals no damage, so `lethal_event`'s death branch and the InstrumentError cannot be
exercised offline — the #755 test asserts `capped` on both no-damage rows.**
*Evidence:* `water_trial.py::WaterTrial.deaths` — `except (IndexError, ValueError): return 0` unchanged;
`scripted_water.py::ScriptedWaterControl.command` returns `"<bot> has 0 [<obj>]"` for every scoreboard
read; `ScriptedWaterBridge._snapshot` pins `health: 20.0`; the #755 test (on `main`, not on this branch)
asserts `a["capped"] is True` and `d["capped_window"] is True` — the SURFACE and CAP branches are proven
offline, the DEATH branch is live-only. On the live rig the pilot's read happened to parse, so the silent
zero was never seen. With the zero, a death reads as a SURFACE (`is_in_water` 0 at respawn), giving
`t_surface` ≈ 33 s and `survived` True — not "survived to H" as v1 said, but worse: a plausible latency.
*The order inside a sample is load-bearing:* `sample_full` syncs the snapshot FIRST, then reads
`deaths()`; a respawn snapshot therefore never arrives with a stale `deaths_delta` 0. Reversed, a death
becomes a surface. Pin it.
*Fix (harness):* `deaths()` raises `InstrumentError` on an unparseable reply; preflight: `scoreboard
objectives list` contains `deaths_objective`, then `scoreboard players set <bot> <obj> 0` and a READ-BACK
== 0 (proves the parser on a known value, not on whatever the server happens to say); the death detector
is `deaths_delta > 0` corroborated by the respawn discontinuity (`health` 20 after < 20, `is_in_water` 0,
`y == shore_y`) within one sample — disagreement is an `InstrumentError`; add a scripted drowning to
`ScriptedWaterBridge` (oxygen 0 → health −2/s → death → anchor to shore, `deaths` 1, health/food/oxygen
reset) so the death branch and the parse failure are red-gated offline (`harness-loop-must-be-proven-live`
rung 5: reproduce the live condition offline).

**SF-D — Two clocks, one row: `_telemetry_ticks` ignores its `t0_monotonic` argument (dead parameter);
ticks are relative to the loop's FIRST telemetry row, calls/samples to the teleport; the offset is
≈ 0.75 s, NOT `loop_warm_s` (1.0), and nothing records it.**
*Evidence:* `water_trial.py::_telemetry_ticks(path, t0_monotonic)` never references `t0_monotonic`; it
subtracts the first row's wall-clock `ts`. Matching each proposal to its executor call gives the offset:
lethal_A flee proposal 26.606 vs flee call 25.846 (0.76), escape 27.399 vs 26.605 (0.79); lethal_B 7.000
vs 6.259 (0.74), 7.727 vs 7.027 (0.70). `row_drown`'s `ticks_after_death` uses `death + loop_warm_s` and
says "approximate". v3's headline numbers (flee 25.9 s, escape 26.6 s; 6.3 / 7.0 s) are all from `calls`
(window clock) and are comparable to each other and to `t_surface`; but the row's `first_proposals`
entry `[26.606, flee]` (tick clock) coincides numerically with the escape CALL at 26.605 (window clock) —
any reader will mis-pair them. Also: `first_proposals[*].threat` is `None` on every row because
`_telemetry_ticks` filters `drives` to `("threat","oxygen","health")` while
`substrate_telemetry._drive_snapshot` keys drives by the body's `drive_specs` sensor names — a dead
column; the prereg's decision provenance must come from `RecommendCapture` (Exp 61 step 5), not here.
*Fix (harness):* stamp `t0_wall = time.time()` at the teleport (or have `enter()` return both), convert
ticks with `ts − t0_wall`, delete the dead parameter, carry `t0_wall`/`first_tick_wall`/offset in the
row, keep the FULL `ticks` list per event (one event per agent — the pilot dropped it to `n_ticks` + 6
proposals, which is why Q1's "any percept-path effect visible in the ticks" cannot be answered from the
rows).

**SF-E — The flee anchor is start-time-only on the bridge, the water anchor record carries none, and the
pilot's preflight result ("No path to the goal!") would be REFUSED by v3's own rule.**
*Evidence:* `index.js` parses `--flee_x/--flee_z` once into `FLEE_X/FLEE_Z` consts (lines 29–41); no
runtime setter; fallback `bot.spawnPoint` = WORLD spawn; `flee` fails fast when submerged and throws
`"no flee anchor"` when neither is set. `setup_world.water_anchor_record` writes `shore`, `submerged`,
`surface_y`, `depth`, `pool`, `deaths_objective` — no `flee_x/flee_z` (only the depth-cave record has
them, `setup_world.py` ~393–400). The pilot's `flee_preflight` (through the BRIDGE, correctly) returned
`{"ok": false, "detail": "No path to the goal!", "latency_s": 0.015}` — the running bridge's anchor is
not reachable from the shore; v3 §Refusals requires `"fled to anchor"` ≤ 0.5 s. The pilot recorded and
did not refuse. Tie-break cost, symmetric and inside `t_surface` in every arm: `flee` first while
submerged, bridge-side failure, then `escape_water` ≈ 0.7–0.8 s later (6.259 → 7.027; 25.846 → 26.605).
*Fix (harness):* put `flee_x/flee_z` (= the shore x/z) in the water anchor record or the gauntlet file;
the preflight REFUSES on "No path"/"no flee anchor" and records latency + detail per campaign start and
per row (15 ms — cheap); since the bridge cannot be re-anchored at runtime, the preflight is the only
verification that the operator restarted it with the anchor (bridge-restart-after-sensor-change); record
`t_first_call` and `t_escape_call` per event so the tie-break's ≈ 0.75 s is visible beside `t_surface`.

**SF-F — The 1.20.4 gamerule is `drowningDamage` (camelCase), added 1.15 (19w36a), default `true`;
`doDrowningDamage` was a wrong NAME, not a missing rule. v3 drops it; restore it under the right name
with a refusal on "Incorrect argument", never a recorded absence.**
*Evidence:* the pilot's `read_rules` got `"Incorrect argument for command\ngamerule doDrowningDamage<--[HERE]"`
(row `apparatus.rules`); minecraft.wiki `Game_rule` (fetched 2026-09-17): `drowningDamage` (Java, 1.15 /
19w36a, default true; renamed `drowning_damage` in 1.21.11), alongside `fallDamage`, `fireDamage`
(1.15) and `freezeDamage` (1.17); the environment lens's `doDrowningDamage` (its SF-5) was the wrong
spelling. The rule governs the DV (drowning death); "evidently ON (two deaths)" is a behavioural
inference, not a roster read.
*Fix (harness):* `check_gamerules` extended with `("naturalRegeneration","true")`,
`("drowningDamage","true")`, `("doInsomnia","false")`; reply `"currently set to: true"` → record;
`"false"` → Refusal; a reply containing `"Incorrect argument"` → `InstrumentError("unknown gamerule name")`.
Every `set` (the pilot's `set_regen`) is followed by a READ-BACK — `row_drown.regen_restored` is the SET
echo (`"is now set to"`), and the offline test asserts that echo; a failed restore would carry the next
campaign at regeneration off with no other symptom.

**SF-G — Provenance and the gauntlet file: #755 was SQUASH-merged, so the hash every pilot row stamps
(`5d1e6a62`) is reachable only from `r3/pilot-fix2`; `r3/v3` forked from `d43d73cb` (#754) and carries the
PRE-#755 pilot + test, so the prereg on this branch describes code the branch does not contain.**
*Evidence:* `gh api …/commits/542858e3` → one parent (`d43d73cb`); `git merge-base --is-ancestor 5d1e6a62
HEAD` → 1, `… origin/main` → 1; `git branch --contains 5d1e6a62` → `r3/pilot-fix2` only. On this branch
`r3_pilot.py::Pilot.__init__` still reads `self.geom["surface"]` (the builder writes `surface_y`; #755's
diff fixes exactly that — a `KeyError` against any real anchor) and `test_r3_pilot.py` is the
construct-only stub-RCON test, not the "runs every row on the scripted bridge" test v3 cites. My review
of `live_window` is unaffected (#755 touched `__init__` geometry + the test only).
*Fix:* merge `main` into `r3/v3` before the freeze PR. For the harness: v1 SF-9's `merge-base
--is-ancestor <cal_code_hash> origin/main` refusal is only satisfiable if the cal PR is MERGE-committed —
write "no squash for the cal/bench PRs" into §Build order (the process rule, now with a mechanical
consequence); `in_process_code_provenance(..., allow_dirty=False)` for cal/bench (the pilot exposes
`--allow-dirty`). Add to the gauntlet file what the pilot stamps per row and v1 SF-9 did not list:
`provenance` (`executed_git_hash`, `executed_maxim_file`, `python`, `pythonpath` — the
`assert_repo_interpreter` family), `rss_mb` (pilot 72.0 → 76.4 MB over five builds in one process,
≈ 1 MB per row — one event per agent makes SF-7's per-episode subprocess unnecessary for memory; keep
the stamp), and `exp60_run.FROZEN` frozen BY VALUE (sha256 of the sorted JSON) — the pilot imports it
live, so there is no literal copy for an `exp60_frozen_matches` analogue to compare. The pilot's
`close_and_stage(aut, pump, None)` stages NOTHING and the "persisted files exist" check lives inside the
`stage_dir is not None` branch — v3's "the agent's persistence is closed and kept as the row's evidence"
needs a stage dir per row.

**SF-H — Per-sample `deaths()` over RCON costs ≈ 0.14 s per sample: the pilot sampled at 2.6 Hz, not
the 4 Hz the prereg states, and the bridge cadence was never re-measured inside a window.**
*Evidence:* every `sample_full` window has sample-interval median 0.39 s (min 0.255, max 0.41; `sleep(0.25)`
+ sync + one synchronous `RconControl.command` round trip, `exp56/common.py`). No committed Exp 60/61
window keeps its samples, so the RCON share is inferred from the only blocking call added; `t_surface`/
`t_death` resolution is therefore ± 0.4 s. No cost to the loop: 1.65 Hz ticks in both lethal rows
(49 / 29.7 s; 17 / 10.3 s) vs 1.78 Hz at rest (107 / 60 s); 8 ticks in the 6 s after death with six
blocking escape calls inside them. `bridge_state_interval_s` 0.101 was measured at preflight only; v3
says "cadence ≤ 0.15 s at every sample" — the pilot did not.
*Fix (harness):* `sleep(max(0, 0.25 − elapsed))` to hold 4 Hz (or declare 2.5 Hz); stamp
`state_age_s` per sample (`client.state_age_s()` is already read for the stale guard — record it) and
refuse > 0.15 s; report the achieved sample cadence per event.

### NIT

- **N-1** The settle guard: the pilot's `settle_guard` (`is_raining` 0, `nearest_player_dist` 64) is
  enforced only inside `rescue()` (v1 N-4 unchanged); `live_window` never checks it; `hostile_count == 0`
  (v3's third key) is nowhere. `sample_full` should carry the guard keys and `lethal_event` refuse on any
  sample that breaks one (after the DV read, a break is "dirty", before it a refusal).
- **N-2** `row_lethal.escaped_before_damage` compares to the ANCHOR's `t_damage_onset_min_s` (16.067),
  not the event's own first health drop; lethal rows record no `t_first_damage` (drown rows do). Record
  both; an event whose own onset is earlier than the anchor minimum with `oxygen > 0` is v3's "hostile
  leaked / fall damage" refusal — implement it from the samples.
- **N-3** The death/damage percepts: `[minecraft:damage] took damage (health N)` per `entityHurt` (~1 Hz
  while drowning) and `[minecraft:death] the player died` — none match `HARD_STOP_TRIGGERS`
  (`autonomy.py`: stop / halt / maxim stop / maxim halt / emergency stop / emergency_button), so the
  controller is never paused (confirmed). They DO wake the loop (`MinecraftPerceptSource.has_pending`)
  → extra non-idle iterations → the per-tick NAc decay unit runs more often during a damage stream.
  `cluster_fear` has no per-tick decay and `reward_bias == {}` at the boundary, so no DV effect; record
  the count of `[minecraft:damage]` percepts consumed per event so the arm-A rows carry it.
- **N-4** `_spy_execute` records `t` at CALL START only. With 8 s blocking calls, "when did it return"
  is the fact that dates the in-flight-at-death success (SF-B); record both.
- **N-5** `LETHAL_CAP_S = 45` is fine (death 25.3–33.0 s); keep `POST_DEATH_S` linger for the diagnostic
  script only — it has no place in `lethal_event`.
- **N-6** Q1's sanity: `deaths0` is read after `loop_warm_s`, before `enter()` — right place; keep it,
  and add the reset + read-back (SF-C) in front of it.

---

## Charter questions, answered (the delta)

1. **SF-3 (death/respawn seam).** Discharged for the CONTINUATION: loop ticks through death (8 in the 6 s
   after, with six blocking calls inside), respawn snapshot 0.4 s after (`y` 40.1 shore, health 20,
   food 20, oxygen 20, saturation 5, `is_in_water` 0), `deaths` +1 exactly, both regen settings. Text
   events: no HARD_STOP match; no visible cadence effect (1.65 Hz through a ~12-event damage stream vs
   1.65 Hz without one) — but the rows dropped the per-tick list, so "visible in the ticks" is not
   answerable from the record (SF-D). Per-sample RCON `deaths()` is the right detector shape (order:
   snapshot then scoreboard — SF-C) and costs ≈ 0.14 s per sample → 2.6 Hz effective (SF-H); nothing
   visible on `bridge_state_interval` (not re-measured) or tick counts.
2. **SF-4.** Not discharged: `water_trial.py::WaterTrial.deaths` unchanged; the pilot relied on it
   silently; the scripted control cannot exercise it. Owed in the harness PR at `WaterTrial.deaths`
   (raise), the preflight (objective exists + set-0 + read-back) and `scripted_water.py` (a scripted
   death) — SF-C.
3. **SF-1 (`live_window` as the prototype).** The `until` predicate `(not in_water) or deaths_delta > 0`
   and the surface/death disambiguation (`surf` requires `deaths_delta == 0`) are right; the stale guard
   (8 consecutive stale polls at `STALE_STATE_S` 1.5 s) is inherited correctly; the cap is fine. What
   must change: the `finally` ORDER (SF-A), the linger (none), the DV-from-samples-only rule, the join
   stamp, one wall-clock `t0` for the ticks (SF-D), the settle guard per sample (N-1), `detail` + return
   time per call (SF-B/N-4), a stage dir per row (SF-G). Instruments: `attach_instruments` once per
   object, as the pilot does; `_detach_fear_subscriber` before or after is safe (`_record_pain`'s
   qualname does not contain `cluster_fear`). Hub: `reopen_hub_session` after the join, as inherited.
   v3's 26.6 s vs 25.85 s are BOTH window-clock (`calls`) numbers and comparable; the tick-clock offset
   (≈ 0.75 s) is unrecorded and is the trap.
4. **The same-tool cap.** SF-B: the 6th consecutive identical proposal is dropped (counters reset), a
   5/6 duty cycle; it fired post-respawn on the shore, not under the stone cap; it cannot bite the first
   escape at any buildable depth; the post-surface snowball is real re-escapes (the agent sinks back)
   plus the linger that allowed them.
5. **The flee anchor.** Start-time only (`--flee_x/--flee_z` → consts); the water anchor record carries
   none; the pilot's "No path" would be refused by v3's rule — SF-E.
6. **Drowning gamerule.** `drowningDamage` (1.15+, default true) — SF-F; read it, refuse on
   "Incorrect argument", refuse on false, verify per cell.
7. **Gauntlet file + two-record apparatus.** Unchanged in shape; add the pilot's `provenance`, `rss_mb`,
   the FROZEN-by-value hash, the flee anchor, the stage-dir rule, and the merge-commit rule — SF-G.
8. **Pieces without a caller, in the pilot:** `_telemetry_ticks`'s `t0_monotonic` (dead); the `threat`
   drive column (wrong key space, always None); the end-of-window `deaths_delta` beside the sample
   detector (two reads, one can disagree); `set_regen`'s echo (no read-back); the settle guard only in
   `rescue()`; the scripted bridge's `deaths` 0 / no damage (the death branch has no offline caller);
   `close_and_stage(…, None)` (the persistence check has no caller when stage_dir is None).

## What I verified

- Read in full: v1 `wiring.md`; `DESIGN_REVIEW.md`; `harness-loop-must-be-proven-live.md`; prereg v3;
  `r3_pilot.py` (489 lines); `test_r3_pilot.py` on this branch AND `gh pr diff 755` (the version that
  produced the rows); `scripted_water.py`; `water_trial.py` `GAMERULES`, `_detach_fear_subscriber`,
  `_telemetry_ticks`, `WaterTrial.__init__/attach_instruments/detach_instruments/reopen_hub_session/
  deaths/rescue/stop_motion/sample/submerge/check_bridge/check_liveness/check_gamerules/
  positive_escape_links/loop_window/placement/check_death_cap/probe/final_rescue`; `exp60_run.FROZEN`
  and record paths; `exp61_run.close_and_stage`; `setup_world.water_classroom_geometry/
  water_anchor_record` and the depth-cave flee printout; `survival_world/common.py`;
  `exp60_water_check` constants; `exp56/common.py::RconControl`.
- Traced in `agent_loop.py`: `_substrate_tick_due`, the substrate branch, the consecutive-same-tool cap
  (count / drop / reset / `continue`), `_start_bio_session`, the stop-event check; in
  `minecraft_harness.py`: `_loop_kwargs` (AUTONOMOUS, `consolidation="full"`), `run_minecraft_aut`'s
  `finally: bio.on_session_end()`; in `simulation/minecraft.py`: `call_action` (blocking, 15 s timeout,
  `unknown` on timeout), `_handle_line` (event queue, `action_result` state absorbed), `pop_event`,
  `MinecraftPerceptSource.has_pending/next_percept`; in `autonomy.py`: `HARD_STOP_TRIGGERS` +
  `check_hard_stop`; in `substrate_telemetry.py`: `record` (`ts = time.time()`), `_drive_snapshot`
  key space; in `index.js`: `entityHurt`/`death` events, `flee` (anchor consts, submerged fail-fast),
  `escape_water` (head check, 600 ms hold, 8 s cap, `"already at surface"`), the `spawn` handler.
- Computed from the five rows: per-row calls with timestamps, sample-interval medians (0.38–0.39 s in
  every `sample_full` window), the surface/re-sink transitions, health series around death, the
  proposal-vs-call offsets (0.70–0.79 s), the 8.1 s call spacing under the cap, the 1.8 s / 1.5 s
  dropped-proposal gaps after respawn, tick rates (1.65 / 1.65 / 1.78 Hz).
- Provenance: `git cat-file -t 5d1e6a62` (commit), `merge-base --is-ancestor` vs HEAD and origin/main
  (both 1), `branch --contains` (`r3/pilot-fix2` only), `gh api` on the #755 merge commit (one parent →
  squash), `gh pr view 755` (MERGED 2026-09-17T21:21Z; files: `r3_pilot.py`, `test_r3_pilot.py`).
- Fetched minecraft.wiki `Game_rule` for the drowning rule's name, version and default.
- Not run: anything live; the per-sample RCON cost is inferred (no committed window keeps samples);
  the ticks' percept-path content is unreadable from the rows (dropped to counts).

## Verdict

**FIX-THEN-BUILD.** The pilot discharged the design-level doubts (naive route, hunger, the death seam's
continuation) and nothing here reopens v3's design. The harness PR must not inherit `live_window` as-is:
teleport-before-join with no linger (SF-A), `detail` + return time per call with the in-flight-at-death
success flagged (SF-B), `deaths()` that raises + a scripted death so the death branch has an offline
caller (SF-C), one clock (SF-D), a recorded/refused flee anchor (SF-E), `drowningDamage` under its real
name with read-back after every set (SF-F), merge-commit provenance + the extra gauntlet fields and a
stage dir per row (SF-G), 4 Hz held or declared with per-sample `state_age_s` (SF-H). Merge `main` into
`r3/v3` first — the branch's pilot is not the pilot that produced the rows.
