# R3 survival benchmark — ENVIRONMENT lens, DELTA re-run on v3 (2026-09-17, after the PILOT)

**Charter (docs/experiments/DESIGN_REVIEW.md):** does the world game-natively afford it, are the
needed states/acts reachable + measurable, does the bridge/world behave. **This is the re-run the v1
lens asked for.** The v1 report (`environment.md`, same dir) listed eight things it could not verify
offline; the pilot (`scripts/survival_world/r3_pilot.py`, rows
`docs/experiments/data/r3_pilot_2026-09-17.jsonl`, hash `5d1e6a62`, Paper 1.20.4 on big-mac-mini)
ran, and v3 of `r3_survival_benchmark_prereg.md` folded it. This report grades the DELTA only: which
of those eight are now measured and what the rows say; which world facts v3 states are actually
supported by the rows; what remains unmeasured. It does not re-derive v1's DNB-1/DNB-2 (both folded
into v3 and confirmed by the rows: the naive agent does not drown, hunger does not move at rest).

Read: DESIGN_REVIEW.md; the prereg v3 (§Pilot, §The gauntlet, §Goldilocks, D5); the five rows
(every sample, not only the summary fields); `r3_pilot.py`; `setup_world.py`
(`water_classroom_geometry`, `_GAMERULES`, `SERVER_PROPERTIES`, `water_anchor_record`);
`water_trial.py` (`GAMERULES`, `heal`, `rescue`, `deaths`, `submerge`); `minecraft_bridge/index.js`
(`flee`, `escape_water`); `substrate_telemetry.py::_drive_snapshot`; `sem.py::
corrective_need_intensity`; `minecraft_player.yaml::health`; the Exp 60 apparatus record; and, for
the one fact the repo cannot answer, the Minecraft wiki's game-rule history.

**Verdict line: FIX-THEN-BUILD — no DO-NOT-BUILD.** The pilot measured what the v1 lens asked for,
and the rows support v3's load-bearing facts (the two routes to air, the two death edges, the regen
cancel, zero exhaustion at rest, the death/respawn seam). Eight SHOULD-FIXes, all foldable into v3
text or the harness: one gamerule was dropped from the roster because of a misspelling, not because
it does not exist; one event-end predicate as written never fires; two numbers in v3 are arithmetic
presented beside measurements; the stone-cap restore has no observer; and three v1 items are still
unmeasured (the shore-anchor `flee` latency, the regen-off detached escape, the drive read at the
proposal tick).

---

## The v1 live list, item by item

| # | v1 asked for | Status | Value (row / field) |
|---|---|---|---|
| 1 | Innate threat need executes `escape_water` at hp < 14, detached, unrescued | **MEASURED** (behaviour); attribution by timing, not by a drive read (SF-7) | `lethal_A`: `detached_count` 1, `fear_before`/`fear_after` both `{}`; health 14.83 @ 24.86 s → 13.0 @ 25.11 s (the strict `< 14` breach, `sem.py::corrective_need_intensity` `deviation < -comfort_band`, band 6.0); `calls`: `flee` 25.846 s (fails, submerged), `escape_water` 26.605 s; `in_water` 0 @ 27.839 s (feet y 38.68); `min_health` 9.0; `escaped_before_damage` false; survived (`t_death` null). Oxygen pains at 5.27/6.52/7.07 s produced NO call (no fear booked) — the 20 s of silence between the last oxygen pain and the first call is the detached-subscriber signature. |
| 2 | Untrained Wire-4-ATTACHED agent surfaces ≈ 9–10 s in; `cluster_fear_dump` after | **MEASURED** — faster than predicted | `lethal_B`: pains 5.345 (0.5), 5.918 (1.0), 6.645 (1.0); `flee` 6.259 s (fails), `escape_water` 7.027 s; `in_water` 0 @ 8.377 s (feet 38.85); `min_oxygen` 9; `min_health` 20; `escaped_before_damage` true; `fear_after` `{<cluster>|drive:oxygen: -1.0}` (three publishes × α 0.5 = 1.25 → cap 1.0, so the cap is reached in ONE dive, not "one more publish away" as v1 said); `links_after` 7. |
| 3 | Drowning/regen timeline under the frozen regen + start state | **MEASURED** (n = 1 per setting, sampler ≈ 0.4 s) | `drown_off`: first damage 16.145 s, then 2 hp per ≈ 1.02 s (18 @ 16.14, 16 @ 17.32, 14 @ 18.09, 12 @ 19.27, 10 @ 20.05, 8 @ 21.23, 6 @ 22.31, 4 @ 23.09, 2 @ 24.28), death 25.318 s = the 10th hit. `drown_on`: first damage 16.25 s; health oscillates 18–20 (back to 20 @ 19.96) until ≈ 21 s = fast regen (1 hp / 10 ticks while sat ≥ 6) exactly cancelling 2 hp/s; fractional hp in sixths (18.83, 17.5, 16.33, 14.83, 13.17) 21.5–25.6 s = the `min(sat,6)/6` phase as saturation runs 6 → 0; then −2 hp/1.02 s with slow regen +1 hp/80 ticks (the +1 @ 29.77) → 0.33 @ 32.26, death 32.988 s. Regen buys 32.99 − 25.32 = **7.7 s**. Start state: sensed food 20, saturation 10 (the bridge clamp); the regen-cancel duration (≈ 4.7 s × 3 sat/s ≈ 14 sat above 6) is consistent with actual saturation ≈ 20 at the first hit, i.e. the `heal()` injection filled it. The respawn state (sat 5) was NOT drowned from — D4 keeps the heal state, so not needed. |
| 4 | Exhaustion at rest, 60 s, loop live | **MEASURED** | `apparatus.rest`: food 20 → 20, saturation 10 → 10, 160 samples, 107 ticks, `calls` []. Caveat: saturation is sensed at the clamp (10), so up to 40 exhaustion would have been invisible; every game source at rest is 0, so the conclusion stands, but the row cannot see the top half (N3). |
| 5 | Death detection: health-0 visibility, respawn latency/position, scoreboard per sample | **MEASURED** | Both drown rows: `deaths_delta` 0 on every sample until one sample reads 1; that SAME sample already reads health 20 (health 0 never visible at 4 Hz, as predicted under `doImmediateRespawn true`); `respawn_first_sample` (the next sample, +0.39 s): y 40.1, `in_water` false, health 20, food 20, oxygen 20, saturation 5, `deaths_delta` 1 exactly; loop ticked 8× in the 6 s after. So death and respawn fall inside ONE sampler period; the latency is < 0.4 s and unresolved (N1). Scoreboard read per sample worked on 187 samples with no parse failure (the pilot's `deaths()` still returns 0 on a parse failure — N2). |
| 6 | `flee` NoPath duration with the world-spawn anchor; "fled to anchor" latency with the right one | **MEASURED** (first half) / **NOT MEASURED** (second half) | `apparatus.flee_preflight`: "No path to the goal!" in 0.015 s. The reply proves the bridge ran WITHOUT `--flee_x/--flee_z` (a shore anchor within 2 blocks of the standing bot resolves without a search). Why fast: mineflayer-pathfinder's A* exhausts its open set in a sealed room — the chamber is 9×3×3 cells, liquid nodes hard-return, `canDig=false` — and returns `noPath` in milliseconds; v1's "multi-second" assumed the 5 s `thinkTimeout`, which only bounds a search whose frontier is NOT exhausted. So the fast fail is CONDITIONAL on the seal (a depth-8/12 rebuild keeps it: `shell` spans `pool_floor_y − 2 … shore_y + 5`). The shore-anchor latency v3 gates on ("fled to anchor ≤ 0.5 s") has never been measured (SF-8). |
| 7 | Long-window liveness / telemetry growth over 600 s | **MOOT** (one event per agent) | The longest live windows are now 60 s (`rest`, 107 ticks) and ≈ 39 s (`drown_on` incl. 6 s linger, 25 ticks) — both longer than any window proven before (15 s) and longer than the 45 s cap + linger the R3 event needs. RSS 72.0 → 76.4 MB across five agents. |
| 8 | Every gamerule read live; phantoms; difficulty | **MEASURED**, with one CONTRADICTION of v3's fold (SF-1) | `apparatus.rules`: `doMobSpawning false`, `doDaylightCycle false`, `doWeatherCycle false`, `doImmediateRespawn true`, `keepInventory true`, `naturalRegeneration true`, `doInsomnia true`, `difficulty` → "The difficulty is Normal". `doDrowningDamage` → "Incorrect argument for command" — that SPELLING is not a rule; the rule exists (see SF-1). `doInsomnia` read true; v3 says setup sets it false — no code does (SF-6). Settle guard (`is_raining` 0, `nearest_player_dist` 64) passed at every pilot rescue (≥ 8 rescues, no refusal), so "weather clear" is verified live even though `_prepare` never issues it. `hostile_count` was not sampled (N6). |

---

## World facts v3 now states — supported by the rows, or not

- **Ascent 0.43 s/block** — supported as a two-point differential: `ascent_from_floor` 1.809 s (feet 35 → eye-block 40 = 3.4 blocks), `ascent_from_floor_plus_2` 0.953 s (feet 37, 1.4 blocks). The floor+2 placement IS well-defined live: `submerge_to` refuses unless `is_in_water` ≥ 0.5 within 3 s, and the row has `refusal: null` with a real `t_surface`, so the eye at y 38.62 read water (pool spans y 35–39). But each `t_surface` is quantised at the sampler period (≈ 0.4 s: `sample` + 0.25 s sleep), so the slope is 0.43 ± ≈ 0.3 s/block, with a ≈ 0.35 s start latency. The in-window ascents corroborate: lethal_A 3.4 blocks in 1.23 s from the call (y 35.97 @ 27.05 → 38.68 @ 27.84), lethal_B 1.35 s — ≈ 0.30–0.35 s/block once moving (SF-4).
- **Depth 12 adds ≈ 3 s** — arithmetic (7 × 0.43), not measured; at the interval bounds it is 1–5 s. No column deeper than 5 has been built. v3's own "gate (ii) re-run per cell" is the measurement that supersedes this; the prediction "arm A survives every cell with regen on" (5.2 s margin vs ≈ 3 s) is likely but not safe at the pessimistic bound (SF-4).
- **The death edges 25.3 / 33.0 s** — measured, n = 1 each, ± 0.4 s, from the heal start state, with `doImmediateRespawn true`, at depth 5 under a stone cap (depth does not move the edge: air depletes at the same rate anywhere).
- **"The head clears at y ≥ 40" (the respawn y 40.1)** — NOT what the rows show, and the conflation is dangerous for the harness. `is_in_water` is an EYE-block sensor: it flipped to 0 at FEET y 38.68 (lethal_A) and 38.85 (lethal_B), i.e. eye ≈ 40.3–40.5; the feet never exceeded 39.89 in either window. The respawn y 40.1 is the SHORE spawnpoint's feet position, unrelated to surfacing. v3 §The gauntlet's event-end clause "`is_in_water` at eye height 0 with position above the water line" fires only if "position" means eye y ≥ 40 (feet ≥ 38.38); read as feet ≥ 40 it never fires and every event becomes a 45 s refusal (SF-2). Also measured: after the 600 ms `SURFACE_HOLD_MS` the agent bobs back under (`in_water` 1 again at 28.98 s / 9.50 s, feet 37.7) — the event must end at the FIRST clear sample and the cleanup teleport must follow.
- **Regen buys 7.7 s** — supported (32.988 − 25.318), n = 1 each.
- **The stone cap worked** — supported: `calls` in both drown rows show `escape_water` at 7.16 / 15.28 / 23.38 / 31.5 s (drown_on) and 7.49 / 15.70 / 23.81 s (drown_off) — 8.1 s apart = the 8000 ms hard cap returning "still submerged (capped)" as `success: true` — and the agent died. **Was the cap restored? NOT verified by any row.** `cap_pool(False)` discards the `fill … air` reply; the next row (`drown_off`) re-capped before its window and entered the pool by TELEPORT, so drown_on's restore is unobservable; drown_off is the last row, so the classroom's final surface cell has no observer at all. The claim "the next row's submerge/escape succeeded, which proves it" does not hold — no row after a restore exercised the surface cell (SF-5).
- **"Health crosses 14 at ≈ 25.5 s"** — rows say 24.86 (14.83) → 25.11 (13.0): ≈ 25.0 s (N8).
- **"Respawn read 0.4 s after death"** — 0.4 s is the sampler period; death and respawn are in the same sample (N1).
- **"With regeneration off the margin is ≈ 2 s (escape ≈ 23 s)"** — not derivable from the rows; the rows give ≈ 3.3 s (SF-3).
- **Item 4 of §Pilot "the same-tool cap (6 identical calls) broke the chain"** — the drown rows show four capped calls each occupying the executor for 8 s (no chain to break) followed by 6–7 dry successes after respawn; I cannot see the claimed cap effect in the data (N5; wiring lens to adjudicate).

---

## SHOULD-FIX

### SF-1 — `drowningDamage` IS a Java game rule (1.15, snapshot 19w36a); the pilot misspelled it and v3 dropped a real rule from the roster
- The pilot's `EXTRA_RULES` queries `doDrowningDamage` (v1's spelling); the server replied "Incorrect argument" because Java's name carries no `do` prefix: `drowningDamage`, alongside `fallDamage`, `fireDamage` (19w36a / 1.15) and `freezeDamage` (20w48a / 1.17). Source: the Minecraft wiki game-rule history (fetched; the page now lists the post-1.21.9 snake_case names `drowning_damage` etc., and the 1.15 history line). The live read of `doInsomnia` ("is currently set to: true") confirms this server is in the camelCase era. What the repo cannot verify: the exact string on THIS Paper 1.20.4 build — one RCON `gamerule drowningDamage` settles it (expected "…currently set to: true").
- v3 §The gauntlet: "`doDrowningDamage` is not a 1.20.4 gamerule name and is dropped from the roster" — the spelling half is right, the drop is wrong. A false value makes every event a 45 s alive-underwater refusal (loud, not silent — but a whole campaign's worth of refusals). v1's original concern stands with the corrected name.
- **Fix:** replace the pilot's and v3's `doDrowningDamage` with `drowningDamage`, read it live once, and put `("drowningDamage", "true")` into `water_trial.GAMERULES` beside `naturalRegeneration`.

### SF-2 — The event-end predicate "position above the water line" never fires if "position" is the feet
- Rows: `is_in_water` 0 first at feet y 38.68 / 38.85; max feet y in-window 39.89; water top is y 39, air from y 40; the eye sits 1.62 above the feet. "Above the water line" must be defined as EYE y ≥ 40 ⇔ feet y ≥ 38.38 — or the position clause dropped and `is_in_water` (bridge truth, eye block) used alone, as `r3_pilot.py` did.
- The bob-back after the 600 ms hold (in_water 1 again ≈ 1.1 s after the first clear) means: end the event at the first clear sample; teleport to the shore in the cleanup promptly; and record `t_surface` at the sampler's resolution (≈ 0.4 s: the pilot's `sample_full` includes an RCON round trip per sample, so the 4 Hz nominal is really ≈ 2.5 Hz — state the actual period in the harness, or move the `deaths` read off the sample path).

### SF-3 — v3's regen-off floor margin ("≈ 2 s", "escape ≈ 23 s") is arithmetic that the rows do not support; the derivable number is ≈ 3.3 s
- Strict breach (`deviation < -6.0` → hp < 14.0 exactly): with regen off health is integer, so the breach is the 12-hp hit at 19.27 s (`drown_off`), not the 14-hp hit at 18.09. Measured breach → air interval: 25.0 → 27.84 s = 2.8 s (`lethal_A`: tick wait + `flee` tie-break 0.74 s + `escape_water` + 1.2 s ascent). Predicted regen-off to-air ≈ 22.1 s vs death 25.32 → margin ≈ 3.2 s at depth 5; depth 12's extra ascent (1–5 s, central 3.0) lands ON that edge. "With regeneration off depth CAN cross it" is a coin flip, not a fact.
- **Fix:** state "≈ 3 s, derived from one row each, not measured detached with regen off"; any declared regen-off sweep measures arm A at depth 5 FIRST (n ≥ 3) before a depth cell is chosen.

### SF-4 — The per-block ascent rate carries ± 0.3 s/block; do not pre-commit the depth-12 arithmetic
- Two placements, each `t_surface` quantised at ≈ 0.4 s → slope 0.43 ± ≈ 0.3 s/block; the in-window ascents give ≈ 0.30–0.35 s/block once moving. Depth 12 (10.4 blocks of eye rise) → ascent 3.5–5.5 s from the floor; still under the 8 s jump cap at every buildable depth (v3's "8 s cap floors the floor" branch is not reachable at ≤ 12).
- Under regen on, arm A's depth-12 margin is 5.2 − (1…5) s: likely positive, possibly ≈ 0 at the pessimistic bound; health at surfacing ≈ 13 − 1.75 × ascent ≈ 3–6 hp. v3's "likely" wording is right; the R3-cal cell at 12 (n = 12) is the measurement. **Fix:** cite the interval, and make gate (ii) at each rebuilt depth record `t_surface` from the floor at ≥ 3 cycles (the Exp 60 water check's shape) so the per-block rate becomes a measured per-cell number.

### SF-5 — The stone-cap restore has no observer; verify the surface cell before the classroom is used again
- `Pilot.cap_pool` discards the `fill` reply; no row after a restore exercised the pool's top air cell (drown_off re-capped and entered by teleport; nothing ran after drown_off). If the final `fill … minecraft:air` did not land, every subsequent Exp 60/61/R3 dive on this classroom is a capped drowning.
- **Fix (operator, before R3-cal or any Exp 60/61 re-baseline):** one RCON `execute if block <surface x y z> minecraft:air` (the builder's own `water_classroom_verifications` line) or a full `exp60_water_check` (its w4 bridge-side escape refuses on a stuck cap). **Fix (pilot script):** record both fill replies in the row and refuse on "No blocks were filled" for the restore.

### SF-6 — `doInsomnia false` is stated as a setup fact but no code sets it; the row read true
- Neither `setup_world._GAMERULES` (five rules) nor `water_trial.GAMERULES` (five rules) names it; `_prepare` never issues it. Reachable trivially (`gamerule doInsomnia false`; the name is valid — the read succeeded). Phantoms are already blocked three ways (PhantomSpawner is gated on the spawn-enemies flag, `doMobSpawning false` read live; frozen day; no sky access under the shell), so this is a belt — but v3 says "set at setup", and a stated-not-done roster line is exactly the shape the v1 lens flagged for `naturalRegeneration`. **Fix:** add to `_GAMERULES` and to the verified roster in the same PR as SF-1.

### SF-7 — The innate-route attribution in `lethal_A` rests on timing; the drive read at the proposal tick was not persisted
- `first_proposals[*][2]` is `None` in both lethal rows because `_telemetry_ticks` keys `drives` by SENSOR name (`_drive_snapshot`: `drives[ds_name]` → `health`, `oxygen`) and the pilot reads `.get("threat")` (a NEED name). The rig's telemetry JSONL (`~/r3_pilot/r3pilot_lethal_A/telemetry_lethal_A_*.jsonl`) carries `drives.health` per tick; the committed row does not.
- The timing case is strong (breach 24.86–25.11 s → `flee` 25.85 → `escape_water` 26.61; oxygen at 0 since 14.98 s with no call; `fear_after` `{}`; `detached_count` 1 — no other game-native trigger exists in that window), but v3 demands of arm C "the object read paired with the behaviour, never the object alone"; arm A's anti-vacuity row deserves the same. **Fix:** the harness persists the drive intensities and `RecommendCapture`/`decision_decisive` per proposal (v3 §DVs already promises it); fix the pilot's key so the re-run inside the campaign records `drives.health` at the `escape_water` proposal.

### SF-8 — The shore-anchor `flee` preflight v3 gates on has never run; and with the anchor at the shore it is a free success
- The pilot's bridge ran with the world-spawn anchor (item 6). With `--flee_x/--flee_z` = the shore and the bot standing there, `GoalNearXZ(anchor, 2)` is satisfied at the start, `goto` resolves at once, and the bridge returns "fled to anchor" — the bridge's own comment on the removed position fallback warns that a goto-to-where-you-stand "would book flight SUCCESS for doing nothing". In R3 the event is over by the time a threat need stands on the shore, so this cannot change a DV, but a standing need (arm A leaves the water at 9 hp, below the band until slow regen lifts it past 14 — ≈ 20 s at 1 hp/4 s) will then execute `flee` (name tie-break) as an always-success at the 5/6 duty cycle instead of `escape_water`, changing `links_after` and the post-event snowball's composition. **Fix:** measure the shore-anchor latency live once (the ≤ 0.5 s gate is then a real number); record that dry `flee` is a free success like dry `escape_water`; wiring lens to decide whether post-event links are reported at all.

---

## NIT

- **N1** — "respawn read 0.4 s after death": the sample that first shows `deaths_delta` 1 already reads health 20 and is the respawn; 0.4 s is the sampler period. Word it "death and respawn within one sample (< 0.4 s); health 0 never visible". The drown rows persist only `(t, health)` per sample, so the death sample's `y`/`in_water` are not in the record.
- **N2** — `WaterTrial.deaths()` still returns 0 on a parse failure (v1 SF-2); v3 promises `InstrumentError` in the harness — owed, not yet in code.
- **N3** — The rest row's saturation is at the bridge clamp (10); a drain of ≤ 40 exhaustion in 60 s would be invisible. All game sources at rest are 0, so the conclusion stands; say the clamp.
- **N4** — The regen-on curve is exactly 1.20.4's `FoodData.tick`: fast regen `min(sat,6)/6` hp per 10 ticks spending `min(sat,6)` exhaustion (≈ 3 saturation/s) cancels drowning while sat ≥ 6, then the sixths phase, then slow regen 1 hp / 80 ticks while food ≥ 18 (the +1 at 29.77 s). Corollary: the regen-on death edge also depends on FOOD ≥ 18, constant under D4 (food 20 → 19 only after surfacing in `lethal_A`, 29.36 s — the first exhaustion the pilot ever sensed, spent by regen).
- **N5** — `calls` entries carry no `detail`; capped vs surfaced `escape_water` is distinguishable only by the 8.1 s spacing. The harness must persist `detail` (v3 says so). v3 §Pilot item 4's "same-tool cap broke the chain" is not visible in the rows (four 8 s-blocking capped calls, then 6–7 dry successes post-respawn).
- **N6** — `hostile_count` was not sampled; v3 adds it per sample. `is_raining` 0 and `nearest_player_dist` 64 were enforced at every pilot rescue and passed.
- **N7** — Liveness through the window: proposal ticks ran at ≈ 1.6–1.8/s (`rest` 107 / 60 s, `lethal_A` 49 / 30 s), below the nominal 2 Hz proposal cadence — fine for one event, but the per-window liveness refusal should be stated per 10 s bucket, not per window.
- **N8** — "Health crosses 14 at ≈ 25.5 s" → ≈ 25.0 s (14.83 @ 24.86, 13.0 @ 25.11). The breach is strict (`< 14.0`); with integer health (regen off) it is the 12-hp hit.
- **N9** — Fear cap reached in ONE dive (three oxygen publishes, −1.0), not −0.75 as v1/v3 §Pilot's prediction column say; v3's measured column has it right. Arm B's carried value after event 1 equals arm C's.

---

## What I verified

- Every field cited above, from the five rows (all 78 + 28 samples of the lethal rows; all 104 + 83 `health_series` points of the drown rows; the `calls`, `first_pain`, `first_proposals`, `fear_*`, `links_*`, `rules`, `rest`, ascent and flee fields of the apparatus row; provenance: hash `5d1e6a62c87b`, clean tree, rig interpreter).
- `r3_pilot.py`: `live_window` never rescues inside the window; `sample_full` does one RCON `deaths()` per sample; `submerge_to` refuses unless `is_in_water` reflects the teleport; `bridge_escape_timing` calls the bridge, never the executor; `cap_pool` discards the fill reply; `read_rules` spells `doDrowningDamage`; `first_proposals` reads `drives.threat`.
- `setup_world.py`: `water_classroom_geometry` (depth 3–12, pool y `shore_y − depth … shore_y − 1`, surface cell y 40, shell `pool_floor_y − 2 … shore_y + 5`, chamber 9×3×3), `_GAMERULES` (five, no `doInsomnia`), `SERVER_PROPERTIES` (difficulty normal), `water_anchor_record` (`deaths_objective`, depth, no flee anchor).
- `water_trial.py`: `GAMERULES` (five; no `naturalRegeneration`, no `drowningDamage`), `heal` (instant_health + saturation, amp 1, 10 s), `rescue` (settle guard enforced there), `deaths` (0 on parse failure), `submerge`.
- `index.js`: `flee` fails fast when `isInWater`, anchor `--flee_x/z` else `bot.spawnPoint`, `GoalNearXZ(anchor, 2)`, `canDig=false`; `escape_water` eye-block test, 600 ms hold, 8000 ms cap, "still submerged (capped)" vs "surfaced".
- `sem.py::corrective_need_intensity` (strict `deviation < -comfort_band`), `minecraft_player.yaml::health` (set_point 20, band 6.0), `agent_loop.py::_DRIVE_CORRECTIVE_NEEDS` (`health → threat`), `substrate_telemetry.py::_drive_snapshot` (keys by drive-spec name).
- Minecraft wiki game-rule page (fetched): `drowningDamage`/`fallDamage`/`fireDamage` added to Java in 19w36a (1.15), `freezeDamage` 20w48a (1.17), `naturalRegeneration` 13w23a, `doInsomnia` 19w36a — displayed under their post-rename snake_case names.

## What remains unmeasured (live, before R3-cal)

1. `gamerule drowningDamage` on the rig — expected "currently set to: true" (SF-1); then it enters the verified roster.
2. The pool's surface cell is AIR after the pilot (SF-5) — one `execute if block`, or a full `exp60_water_check`.
3. `flee` with the SHORE anchor: latency and reply on the shore (SF-8) — the bridge must be restarted with `--flee_x/--flee_z` from the anchor record, which currently does not carry one (`water_anchor_record` has no `flee_x/z`; the Exp 58 record does).
4. Depth cells 8 and 12: each rebuild is a NEW anchor record (the pool floor moves to y 32 / 28; the shell deepens); the Exp 60 apparatus check + gate (ii) re-run per cell, including the ascent from the floor at ≥ 3 cycles (SF-4) and the cosine replay of the submerged reading (v3 corollary 3; `y_altitude` changes by −3 / −7).
5. The regen-off detached escape at depth 5, n ≥ 3, if a regen-off sweep is ever declared (SF-3) — v3 already marks it unmeasured; the rows' derived margin is ≈ 3 s, not 2.
6. The innate threshold vs the regen curve at n > 1 under regen on: the breach time (≈ 25.0 s) depends on the saturation the heal actually leaves (the clamp hides it) and on the 2 Hz proposal phase; arm A's R3-cal cell at depth 5 (n = 12) is that measurement — the anti-vacuity row should persist `drives.health` at the proposal (SF-7).
7. `hostile_count` per sample and the death-with-`oxygen > 0` refusal have never fired (nothing to measure until the harness exists; N6).

## Verdict

**FIX-THEN-BUILD.** The pilot answered the v1 lens's list: both routes to air are real and ordered as predicted (in-situ fear 8.4 s, innate reflex 27.8 s), both death edges are measured (25.3 / 33.0 s, regen buys 7.7 s), exhaustion at rest is zero, the death/respawn seam holds with the loop live, and `flee` fails fast for a reason the rows explain. Fold before the harness: rename and re-roster `drowningDamage` (SF-1), define the event-end as the eye block (SF-2), replace the two arithmetic numbers with the rows' derived values and their intervals (SF-3/4), verify the classroom's surface cell (SF-5), put `doInsomnia` into the code the prereg says sets it (SF-6), persist the drive read at the proposal (SF-7), and measure the shore-anchor `flee` once (SF-8). Nothing here needs `src/`.
