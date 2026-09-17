# R3 survival benchmark — ENVIRONMENT lens (four-lens design review, 2026-09-17, on DRAFT v1)

**Charter (docs/experiments/DESIGN_REVIEW.md):** does the world game-natively afford it (D1 — no
synthetic sensor/reward), are the needed states/acts reachable and measurable, does the bridge/world
behave — the lens that exists to catch hunger-drains-too-slowly, food-caps-at-20, the eat-lag. Read:
`r3_survival_benchmark_prereg.md` (draft v1, uncommitted), `survival_world_1_3.md` (D1, topology,
version), `exp60_drowning_avoidance_prereg.md` (§Apparatus, §Design (iii), Amendments 3–7, §Outcome),
`exp61_shared_fear_prereg.md` (settle guard), `docs/wiring/harness-loop-must-be-proven-live.md`,
`docs/wiring/sensor-range-clamps.md`, `scripts/survival_world/setup_world.py`, `water_trial.py`,
`exp60_run.py`, `scripts/minecraft_bridge/index.js`, `bodies/minecraft_player.yaml`,
`runtime/agent_loop.py`, `decisions/nac.py`, `embodiment/sem.py`, `embodiment/body.py`,
`proprioception/pain_bus.py`, `simulation/minecraft.py`, the Exp 60 apparatus + trial records, the
prior environment lenses (exp58, r2_learned_bias_v2, exp61) and the break-3 memory.

**Verdict line: DO-NOT-BUILD as drafted.** Two design-level facts about this world and this agent
invert the draft's calibration premise: (1) the kill hazard IS answered by two mechanisms the naive
arm carries by default — the innate `health→threat→escape_water` prior and the auto-wired Wire-4
subscriber booking fear from the agent's OWN first dive — so the naive floor on an unrescued
submersion is ≈ 1.0 survival, not ≈ 0, and every arm sits at the ceiling; (2) hunger is not a
pressure on this gauntlet at all (zero exhaustion at rest, a sealed 5-block shore, starvation
non-lethal on Normal), so D3 option (c) has no axis. Both are foldable (re-cut the DV to the
pre-damage window; drop or re-found the hunger leg; decide per arm what "naive" carries) and the
remainder is FIX-THEN-BUILD on the SF list. Nothing here needs a new `src/` mechanism.

---

## DO-NOT-BUILD

### DNB-1 — The naive agent does NOT drown: the world's kill hazard is answered by two mechanisms every arm carries, and the draft's "0 calls" evidence was generated inside windows built to keep both from firing

**Failure scenario.** R3-cal runs naive agents unrescued; every naive agent surfaces before any
damage (Wire-4 attached) or 5–8 s before death (Wire-4 detached); no cell of any `(H, T)` sweep ever
leaves the ceiling; the "naive ≈ 26 s death" floor the entire D3 argument rests on never appears;
the arms are indistinguishable at survival = `H`.

**Evidence, path (a) — in-episode Wire-4 fear from the agent's own first dive.**
- The Wire-4 fear subscriber is AUTO-WIRED on every NAc-bearing pain bus, deliberately so a
  fear-ablated agent can never be the accidental default: `proprioception/pain_bus.py` (the bus
  builder around line 756: `bus.subscribe(create_pain_cluster_fear_subscriber(nac))`, reached via
  `runtime/bio_stack.py::build_bio_stack` → `build_minecraft_aut`). The Exp 60 harness has to
  actively DETACH it for the ablated arm (`water_trial.py::_detach_fear_subscriber`). A "fresh
  persistence, no training" agent therefore learns from its own submersion the moment nothing
  rescues it.
- The oxygen drive (`minecraft_player.yaml::oxygen`, set_point 20, comfort_band 6, pain_scale 0.5)
  publishes at 13 bubbles with intensity (7−6)·0.5 = **0.5** (measured edge 5.09–5.44 s,
  `exp60_water_apparatus.json::w2_dive.t_pain_edge`) and again at 12 bubbles (series: 5.98–6.29 s)
  with intensity (8−6)·0.5 = **1.0** — the breach latch re-publishes when severity deepens by more
  than `max(1e-3, 0.05·6)` (`body.py::_BREACH_DEEPEN_FRACTION`, `_BREACH_MIN_EPS`), which each
  bubble does. Exp 60 measured exactly this: "20 `drive:oxygen` pain signals" over 10 episodes
  capped at oxygen ≤ 12 = 2 per dive (`exp60_trials.jsonl::training.oxygen_pain_signals`).
- Fear update: `nac.py::NAc.record_cluster_fear` — `valence −= α·intensity`, α = 0.5 (frozen
  fingerprint) → after the two publishes of ONE dive the water cluster reads **−0.75**;
  `NAc.anticipatory_threat_need` returns the magnitude when ≥ θ = 0.5 → need **0.75**. The loop
  combines it by max with the innate threat need and scores `escape_water` at 0.75·0.7 = 0.525
  through `_DRIVE_TOOL_AFFINITIES["threat"] ∋ "escape"` (`nac.py`), which clears
  `_DEFAULT_SUBSTRATE_MIN_CONFIDENCE` = 0.3 (`agent_loop.py::_resolve_min_confidence`) and the
  0.5 drive-activation floor (`drive_value <= 0.5 → skip`).
- Timeline on the measured apparatus: need live at ≈ 6.3 s → next 2 Hz substrate tick → `flee`
  (name tie-break, fails fast in water) → `escape_water` ≈ 1.5 s later (Exp 60 post placement 1:
  flee 0.87 s, escape 1.55 s after need onset) → head in air ≈ 1.5–1.8 s after the call
  (`w4_escape.t_surface`). **Surfaced ≈ 9–10 s from the teleport, minimum oxygen ≈ 9–10 bubbles,
  zero damage, no regen, no food cost.** The agent is now a fear carrier (−0.75; the cap −1.0 is one
  more publish away) for every later event.

**Evidence, path (b) — the innate health→threat prior (present even with Wire-4 detached).**
- `agent_loop.py::_DRIVE_CORRECTIVE_NEEDS` contains `("health", "threat")` (break 1, #683).
  `sem.py::corrective_need_intensity` (homeostatic) returns `min(1, |deviation|)` once the value is
  below set_point − comfort_band, i.e. at hp < 14 the threat need is **1.0** — a full-strength need,
  not a graded one. Exp 60 §Design (iii) "Explicitly NOT claimed" says it in so many words: the
  US-free window is where "the innate `health→threat` reaction cannot fire". R3 removes the window.
- Drowning damage (Java 1.20.4): 2 hp every 20 ticks once air < −20, first hit 1 s after air 0
  (measured 16.07–16.65 s). Hits 18, 16, 14, **12** → hp < 14 at ≈ 19.1 s with regen off (≈ 25 s
  with regen on and a satiated start, SF-1) → need 1.0 → `escape_water` → surfaced ≈ 21–22 s at
  10–12 hp, 3–4 s before the 10th hit. The need is STATE-based: it stays 1.0 while hp < 14, so on
  every later teleport the injured agent proposes `escape_water` on arrival (≈ 1.5–2.5 s, Exp 60
  placements 2–6 timing) and takes no further damage — behaviourally identical to a fear carrier
  with no fear learned. With regen on it heals back to ≥ 14 on the shore (costing food, SF-1) and
  the reflex re-arms at the next event's 4th hit — still never dying.

**Why the draft's evidence does not cover this.** "0 calls in 30/30 pre-training placements; 0 in
60/60 control windows" are Exp 60/61 probe placements capped at `probe_cap_s` = 4.335 s
(`exp60_run.py::FROZEN["probe_cap_margin_s"]`, pain edge − 0.75 s), designed US-free precisely so
neither path (a) nor (b) could fire, and refused on any pain inside the window. They say nothing
about seconds 5–26. No harness on this path has ever let drowning damage land with the loop live
(Exp 60 §Outcome: 0 `drive:health` signals, 0 deaths, every seed). The draft's "Phase 1b (an innate
reaction below deliberation) is unbuilt and not in any arm" is false for the interoceptive
health→threat prior: it is built, on by default, and cannot be removed game-natively — it is the
agent, not the world.

**What this does to the design.** Death is unreachable for every arm on this gauntlet, so the
primary DV (survival time censored at `H`) saturates at the CEILING in all four arms — the opposite
saturation from the one the draft argues about, with the same consequence (D2/D3 vacuous). The only
quantities the world separates between a learned-fear agent and a naive one live in the pre-damage
window: escape latency from the teleport (fear: 1.3–3.3 s; in-episode learner: ≈ 9–10 s on event 1,
then fear-like; innate-only: ≈ 21 s on every event), oxygen-pain seconds, health lost per event.
That is a re-cut DV (Exp 60/61's secondary DV plus an in-episode learning curve), and it changes
the arms: the review must decide, per arm, whether the Wire-4 subscriber is attached during the
gauntlet (attached = "naive" is a one-shot learner after event 1; detached = the innate reflex is
the floor) — a confounding/bio decision, flagged here because the world forces it. Re-run the
four-lens on the re-cut design.

### DNB-2 — Hunger is not a pressure on this gauntlet; D3 option (c) has no axis; starvation cannot kill on this world

**Failure scenario.** R3-cal sweeps `H` and "food supply" on fear-carrying agents whose food never
moves; every cell reads survival 1.0; the calibration cannot land; or a harness "food supply" knob is
declared that changes nothing.

**Evidence.**
- Java 1.20.4 exhaustion sources: sprinting 0.1/m, swimming 0.01/m, jumping 0.05, sprint-jump 0.2,
  attacking 0.1, taking damage 0.1/hit, mining 0.005, hunger effect 0.005/tick/level, **regeneration
  6 per hp**; walking, sneaking and standing add 0. Every 4 exhaustion removes 1 saturation, then
  1 food. From the Exp 60 start state (food 20, saturation 20 after `effect give saturation 1 10`)
  reaching food 0 needs 160 exhaustion; from the respawn state (food 20, sat 5) 100.
- The shore is a dry 5×3 strip inside a sealed 3-high stone chamber
  (`setup_world.py::water_classroom_geometry`: chamber x [ax−3, ax+5], y [40, 42]); `move_to`
  requires `x/z` params the substrate-primary path never emits (`GoalNearXZ(undefined)`); the
  loop's only game-native food sinks reachable here are drowning hits (0.1 each) and regen after
  them (6/hp). An agent that never takes damage — every fear carrier, every arm after event 1 — has
  food 20 at `H` for any `H`. The break-3 smoke needed `effect give minecraft:hunger 1000 20`
  (amplifier 20, re-applied every 1.5 s) to move food at all, and the memory records "the first
  run's false negative was food stuck at ~20, satiated" (`break3_smoke.py` docstring;
  `project_1_3_break3_loop_closes.md`). At rest it is not slow; it is zero.
- Starvation on `difficulty=normal` (`setup_world.py::SERVER_PROPERTIES` + `_prepare` `difficulty
  normal`, re-checked by `_verify`) stops at 1 hp; it kills only on Hard. "An agent that starves
  between submersions has not survived" is not a game-native sentence on this world, and the
  refusal predicate "`deaths` rose while `oxygen > 0` and `food > 0`" has a `food > 0` leg that is
  always true.
- The `eat` prior fires only at food < 11 (`sem.py::corrective_need_intensity` entropic:
  (16 − food)/10 > 0.5), pain (`drive:food:deprived`) at ≤ 6 (`minecraft_player.yaml::food`
  deprivation_threshold 6, satisfaction 16): neither is reachable in `H` = 300 s. Bread 64 =
  320 food points — as a "supply" knob it is unbounded.
- The one game-native hunger interaction this world DOES afford: regeneration burns 1.5 food per hp
  healed and stops below food 18, so a repeatedly-half-drowned agent (innate-reflex-only, regen on)
  loses regen after ~2 events and then accumulates damage; `eat` (+5 food, +6 sat) restores regen —
  "eat to heal". D1-clean, but it presses ONLY agents that take damage, i.e. never a fear carrier,
  so it cannot be the calibration axis for the learned arms either.

**Fold.** Drop hunger as a calibration axis and as a survival leg for v1 (keep `food` recorded as a
drive-integrity term); or found a hunger leg on the regen-costs-food loop with the DV re-cut per
DNB-1 and the drain measured live first (SF-1). A difficulty change to Hard would make starvation
lethal but still would not make food drain at rest.

---

## SHOULD-FIX

### SF-1 — `naturalRegeneration` is unset, unverified, and moves every number in §The gauntlet; the episode-start interoceptive state is a harness injection that must be declared

- Nobody sets or verifies `naturalRegeneration` (grep over `scripts/`, `src/`, `docs/experiments`,
  `docs/plans`: zero hits; `water_trial.py::GAMERULES` verifies five rules, not this one). Default
  true. With regen on, fast regen (1 hp / 10 ticks while food = 20 ∧ sat > 0, +6 exhaustion each)
  exactly cancels drowning's 2 hp/s until saturation is spent (sat 20 → 6.7 s; sat 5 → 1.7 s), then
  slow regen (1 hp / 80 ticks while food ≥ 18) gives net −1.75 hp/s until food < 18. Naive death,
  if the reflexes were absent: ≈ 25.1 s regen off (10th hit at 16.07 + 9), ≈ 27–28 s regen on from
  the respawn state (sat 5), ≈ 31–33 s regen on from the Exp 60 rescue state (sat 20). The draft's
  "≈ 26 s" and "2 hp/s → death ≈ 10 s after onset" hold only with regen OFF.
- The Exp 60 start state comes from `water_trial.py::WaterTrial.heal` (`effect give instant_health
  1 10 true` + `effect give saturation 1 10 true`): saturation ≈ 20, which the bridge CLAMPS to 10
  (`index.js::snapshot` `Math.min(10, bot.foodSaturation)`, `minecraft_player.yaml::saturation`
  range [0, 20], rest 10) — the harness can verify ≥ 10, never 20 (`docs/wiring/sensor-range-
  clamps.md`). A `saturation` effect is a harness injection of interoceptive state; the respawn /
  fresh-join state (food 20, sat 5, hp 20, air 300) is the game-native one.
- **Fix:** freeze `naturalRegeneration` (either value, but named, verified-not-toggled in the
  harness's `GAMERULES`), freeze the episode-start state and HOW it is set (declare the effect, or
  use the respawn state), stamp the sensed `health/food/saturation` per episode, and re-measure the
  death time at calibration under the frozen setting (the draft already promises the pain-edge and
  onset re-measure; add the death edge and the regen-cancel window).

### SF-2 — Death is the DV and the draft has no death-detection instrument; the three obvious ones are each wrong

- `doImmediateRespawn true` + mineflayer's default auto-respawn make health-0 a ≤ 1–2-tick state; a
  100 ms snapshot can miss it entirely (the stream reads 20 → … → 20 with a position jump).
- The bridge `death` event (`index.js` `bot.on("death")`) lands in `MinecraftClient._events`, which
  the loop's own percept source CONSUMES (`simulation/minecraft.py::MinecraftPerceptSource.
  next_percept` → `pop_event`); with the loop live the harness races the loop for it.
- `water_trial.py::WaterTrial.deaths` returns **0 on any parse failure** of `scoreboard players get`
  — acceptable for Exp 60's death CAP, vacuous when death is the DV (a dead RCON reads as "no
  death": the mechanism-that-did-not-run shape).
- **Fix:** poll `scoreboard players get <user> exp60_deaths` on every 4 Hz sample (RCON is local,
  milliseconds); parse failure = `InstrumentError`; `scoreboard players set <user> exp60_deaths 0`
  at episode start (or the `deaths0` baseline); corroborate with the respawn discontinuity (a
  submerged → shore position change with health 20 and no harness teleport). Survival-time
  resolution is then the sample period (±0.25 s) — adequate for a 20–40 s DV; state it.
- Respawn facts (1.20.4): `/spawnpoint <user> x y z` (`setup_world.py::water_classroom_commands`)
  sets a FORCED respawn → the shore, no bed needed; on respawn health 20, food 20, saturation 5,
  air 300, potion effects cleared, inventory kept (`keepInventory true`). `bot.spawnPoint` (world
  spawn, the login packet) is untouched, so `distance_from_spawn` keeps its basis. The episode ends
  at death (draft) — fine — but the SAME bot carries hp/food/sat/effects into the next episode
  unless the harness resets it (fresh persistence resets the agent, not the game entity); the
  reset is the SF-1 injection and must be declared.

### SF-3 — `flee`'s anchor is unfrozen for the water classroom and becomes a hazard the moment a threat need stands on the shore

- The threat affinity's name tie-break picks `flee` before `escape_water` (`nac.py::
  _DRIVE_TOOL_AFFINITIES["threat"]`, Exp 60 §Outcome). On the shore `flee` runs
  `pathfinder.goto(GoalNearXZ(anchor, 2))` with `canDig=false`, anchor = `--flee_x/--flee_z` if
  the bridge was started with them, else `bot.spawnPoint` = WORLD spawn ≈ 69 blocks away
  (`index.js` `case "flee"`; `w1_shore.distance_from_spawn` 69.17). The Exp 58 classroom builder
  prints its own anchor to pass; the water classroom builder prints none
  (`setup_world.py::_water_classroom`), and neither Exp 60/61 runbook mentions flee args — the
  anchor the Exp 60/61 bridge ran with is unrecorded. From a sealed stone room a goto to a far
  anchor is a NoPath after a multi-second search, inside a BLOCKING executor call (client
  `action_timeout_s` 15 s, `minecraft_harness.py::build_minecraft_aut`); a teleport scheduled during
  it lands with no `escape_water` proposable until `flee` returns.
- In Exp 60/61 this never mattered: shore fear was 0 and no damage ever landed, so no threat need
  ever stood on the shore (post roams: 14 `escape_water` "already at surface" successes, no `flee`).
  In R3 an injured agent (hp < 14 with regen off) or an in-episode learner whose fear generalizes
  has a standing need between events.
- **Fix:** the gauntlet file freezes the bridge's flee anchor = the shore; a preflight through the
  BRIDGE (never the executor — it books a positive link) calls `flee` on the shore and requires
  "fled to anchor" within 0.5 s; the event scheduler asserts no executor call is in flight at the
  teleport (the executor spy already records every call) and `T` ≥ the `escape_water` hard cap
  (8 s, `index.js SURFACE_HOLD_MS`/cap) so a held `jump` can never carry a fresh placement up.
  Record, as Exp 60 did, that dry `escape_water` is a free always-success: under a standing need the
  loop executes it at a 5/6 duty cycle (`_MAX_CONSECUTIVE_SAME_TOOL`), ≈ hundreds of positive
  links per 300 s episode.

### SF-4 — A 300–600 s continuous loop window is 20–60× longer than any window ever proven live

Every live-proven window is ≤ 15.07 s (train cap) or 10 s (shore roam); liveness is proven for 3 s
(`FROZEN["loop_liveness_s"]`). R3 needs one continuous window with teleports injected mid-window
and the hub session open throughout (`water_trial.py::loop_window` opens/closes a session per
window; `reopen_hub_session` after). Unknowns only a pilot answers: liveness across the window
(ticks per 10 s bucket), telemetry JSONL growth at 4 Hz for 600 s, NAc decay/consolidation across
one long session, the 15 s action timeout under a stuck `flee`, bridge socket health. **Fix:** add
to the build order a single 600 s naive pilot episode with per-bucket liveness before R3-cal; refuse
any bucket < 4 ticks; make the per-bucket liveness a per-episode refusal in the harness.

### SF-5 — The world config the gauntlet file must freeze (the draft names `H`, `T`, food, geometry only)

- difficulty `normal` (server.properties + `/difficulty`, `_verify`); gamerules verified-not-
  toggled: the five in `water_trial.py::GAMERULES` + `naturalRegeneration` (SF-1) +
  `doDrowningDamage true` (exists since 1.15, default true, never verified — a false value makes
  drowning non-lethal with no other symptom) + `mobGriefing false` (`_GAMERULES`) + `doInsomnia
  false` as a belt (N1); the Phase-0 instrument check restores `doMobSpawning` to `true` on exit
  (`setup_world.py::_water_classroom` printout) — verify, never assume.
- `weather clear` at setup: the survival `prepare` never issues it (Exp 56's `setup_world.py:170`
  does; the Exp 58 environment lens asked for it); `doWeatherCycle false` freezes whatever weather
  the world had, and the settle guard (`is_raining == 0`) would then refuse every episode.
- `time set day` (1000 ticks → `time_of_day` 0.0417 constant, `light_level` 0 in the sealed room —
  `w1_shore`), spawnpoint = shore, bread ×64 (state it is unbounded on this horizon), bridge
  `--state_interval_ms=100` + the flee anchor (SF-3), the anchor record
  (`~/.maxim/exp60_water_classroom.json`: shore (−393, 40, −312), submerged (−388, 35, −312),
  depth 5) and the apparatus edges (pain edge, onset, plus the death edge and regen window, SF-1).
- The settle guard (`is_raining == 0`, `nearest_player_dist == 64`) is enforced at RESCUES
  (`WaterTrial.rescue`); R3 has none. Move it to the per-event teleport preflight and check both
  per sample (a spectator arriving mid-episode changes the cluster; `hostile_count` per sample is
  the early warning for a spawn leak).

---

## NIT

- **N1 — Phantoms/daylight are not a hazard, but say why.** `doDaylightCycle false` at day + the
  custom spawners (phantom included) sitting inside the `doMobSpawning` branch of chunk ticking,
  and phantoms needing night sky darkness; insomnia needs ≥ 72 000 ticks (1 h) awake, which a 4–6 h
  campaign DOES exceed, so `doInsomnia false` is a cheap belt. Verify by reading the rules live.
- **N2 — Drowning arithmetic.** Air 300 ticks → 0 in 15 s (measured 15.08–15.27), first hit 1 s
  later (16.07–16.65), 2 hp per 20 ticks; death = 10th hit ≈ 25.1 s regen off. Write "≈ 25–33 s
  depending on regen and start saturation", not "≈ 26 s" and "≈ 10 s after onset".
- **N3 — "Free on the shore" is not free.** A sealed 5-block strip; `move_to` is param-less-dead;
  `flee` goes to the anchor (SF-3); no fall damage possible (3-high chamber); nothing to explore.
- **N4 — `eat` at full food** never fires (need > 0.5 only below food 11); harmless. The eat-lag
  poll (`index.js` `case "eat"`) is in place.
- **N5 — `oxygen`** reads `bot.oxygenLevel` (bubbles, 0–20) and resets on respawn; `is_in_water`
  reads the EYE block (Amendment 3) — the submerged target's head at y = 36 in water, air at 40.
- **N6 — Refusal predicate.** Replace "`oxygen > 0 and food > 0`" with "`oxygen > 0`" (food is
  always > 0, DNB-2) and add "`hostile_count == 0` on every sample" as the spawn-leak guard.
- **N7 — Budget.** D4's 4–6 h omits R3-cal (cells × `n_cal` × `H`) and the SF-4 pilot.

---

## What I verified (offline, `file::symbol`)

- Innate corrective needs: `runtime/agent_loop.py::_DRIVE_CORRECTIVE_NEEDS` (`health→threat`,
  `food→hunger`); `_read_drive_states` emits them at `sem.py::corrective_need_intensity` (homeostatic
  → 1.0 at hp < 14; entropic food → (16 − food)/10); `decisions/nac.py::_DRIVE_TOOL_AFFINITIES
  ["threat"]` ∋ `"flee"`, `"escape"` → `escape_water` scores need × 0.7; activation floor 0.5 and
  `min_confidence` 0.3 (`agent_loop.py::_DEFAULT_SUBSTRATE_MIN_CONFIDENCE`); `NAc.anticipatory_
  threat_need` max-combines with the innate need at θ 0.5.
- Fear booking: `proprioception/pain_bus.py` auto-wires `create_pain_cluster_fear_subscriber`
  (intensity threshold 0.3) whenever an NAc is present; `NAc.record_cluster_fear` α 0.5, cap 1.0,
  allowlist {`drive:health`, `drive:oxygen`} (frozen fingerprint); breach re-publish epsilon
  `body.py::_BREACH_DEEPEN_FRACTION` 0.05 × band / `_BREACH_MIN_EPS`; pain intensity
  `sem.py::drive_pain_for_value` → 0.5 at 13 bubbles, 1.0 at ≤ 12; Exp 60 measured 2 oxygen
  publishes per dive and −1.0 after training (`exp60_trials.jsonl`).
- Apparatus: `setup_world.py::water_classroom_geometry/_commands/_verifications`, `_GAMERULES`,
  `SERVER_PROPERTIES` (difficulty normal, level-type normal, spawn-monsters true), `_prepare`
  (`give bread 64`, `time set day`, no `weather clear`), `_verify`; `water_trial.py::GAMERULES`
  (five rules), `WaterTrial.heal/rescue/submerge/deaths/loop_window/placement`, `FROZEN` in
  `exp60_run.py` (probe cap 4.335 s = 5.085 − 0.75; train cap 15.067; death_cap 2; loop 4 Hz,
  proposal 2 Hz); measured edges in `exp60_water_apparatus.json` (t_pain_edge 5.085–5.443,
  t_oxygen_zero 15.08–15.27, t_damage_onset 16.07–16.65, health_at_rescue 18, t_surface 1.45–1.83,
  w1 light 0 / time 0.0417 / distance_from_spawn 69.17).
- Bridge: `index.js` — `snapshot()` roster incl. `saturation` clamped at 10, `oxygen`, `is_in_water`
  at eye height; `escape_water` holds `jump` ≤ 8 s, releases in `finally`, "already at surface"
  when dry; `flee` fast-fails submerged, anchor = `--flee_x/z` else `bot.spawnPoint`, `canDig=false`;
  `eat` polls `bot.food` (eat-lag fix); actions run concurrently (no queue); `bot.on("death")` →
  event; `STATE_INTERVAL_MS` default 500 (harness requires 100).
- Client: `simulation/minecraft.py::MinecraftClient.call_action` (id-matched, blocking, 15 s
  default), `_handle_line` (events queued), `MinecraftPerceptSource.next_percept` pops the queue.
- Exp 60/61 records: all "0 calls" evidence is from US-free windows ≤ 4.335 s; 0 `drive:health`
  publishes and 0 deaths in every seed; post roams: 14 dry `escape_water` successes per 10 s;
  Exp 61 settle guard `{is_raining: 0, nearest_player_dist: 64}` enforced at rescues only.

## What I could NOT verify offline (must be measured live, in this order)

1. **That the innate threat need executes `escape_water` live at hp < 14** — structurally the same
   path that executed under the fear need in Exp 60, but never observed (no harness has let damage
   land with the loop live). One naive, Wire-4-DETACHED, unrescued dive with window telemetry.
2. **That an untrained Wire-4-ATTACHED agent surfaces ≈ 9–10 s into its first unrescued dive** (the
   in-episode fear path). One dive, same telemetry; read `cluster_fear_dump` after.
3. The drowning/regen timeline under the frozen `naturalRegeneration` and start state (SF-1): death
   edge, regen-cancel window, food/sat after one full regen.
4. Exhaustion at rest on the shore with the loop live for 60 s: sensed `food`/`saturation` unchanged
   (DNB-2's premise; also catches a hidden exhaustion source such as `flee` pathing).
5. Death detection: health-0 sample visibility at 100 ms, respawn latency and position, the
   scoreboard read per sample (SF-2).
6. `flee` NoPath duration from the sealed shore with a wrong anchor, and "fled to anchor" latency
   with the right one (SF-3).
7. Long-window liveness and telemetry growth over 600 s (SF-4).
8. The gamerule/phantom claims by reading every rule live (SF-5, N1), and mineflayer's
   `consume()` behaviour at full food (N4, harmless either way).

## Verdict

**DO-NOT-BUILD** as drafted: DNB-1 (the naive floor is a ceiling — two default mechanisms answer
the kill hazard, and the cited "0 calls" evidence was generated inside windows designed to keep
them from firing) and DNB-2 (no hunger axis exists on this world). Both are design folds — re-cut
the DV to the pre-damage window, decide per arm what "naive" carries, drop or re-found the hunger
leg — after which the SF list makes it FIX-THEN-BUILD with no new `src/` mechanism.
