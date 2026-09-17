# R3 — WIRING lens (four-lens design review, 2026-09-17)

Reviewed: `docs/experiments/r3_survival_benchmark_prereg.md` (DRAFT v1, uncommitted) against the
code on branch `r3/design-draft` at HEAD `4e25b4754219` (#747, Exp 61 frozen). Charter: real
consumers + a real credit path (D43), the right encoding/seams, no hand-composed shortcut that
passes while the loop fails. Read first: `docs/experiments/DESIGN_REVIEW.md`, every `docs/wiring/*.md`
(esp. `harness-loop-must-be-proven-live.md`, `cosine-separation-is-directional.md`,
`substrate-learning-channels.md`), then the apparatus (`water_trial.py`, `exp61_run.py`,
`exp60_run.py`, `setup_world.py`, `minecraft_harness.py`, `agent_loop.py`, `nac.py`, `body.py`,
`scripts/minecraft_bridge/index.js`) and the committed Exp 60/61 records
(`docs/experiments/data/exp60_trials.jsonl`, `exp61_dryrun2_2026-09-17.jsonl`).

The short version: the draft reuses the Exp 60/61 apparatus "verbatim" but its two load-bearing
premises — *the naive agent never surfaces* and *hunger presses every arm the same* — were both
measured (or assumed) under conditions the gauntlet removes. Removing the rescue also removes the
US-free cap that produced the naive zero, and removing the heal also removes the fed interoceptive
state every fear read was taken in. Neither premise transfers; the calibration decision D3 is
built on both.

---

## Findings

### DO-NOT-BUILD

**DNB-1 — The "naive floor" is an artefact of the US-free cap; with the loop live through the pain a
fresh agent learns the fear inside its FIRST submersion, before damage.**
*Failure scenario:* R3-cal (c) calibrates on fear-carrying agents against a naive floor that is not
a floor; arm 1 is reported as "nothing learned, nothing carried" while it is in fact learning at
event 1 with the same subscriber the other arms use, and every contrast in the draft inherits a
mislabelled baseline.
*Evidence:*
- The two numbers the draft leans on ("Exp 60: 0 calls in 30/30 pre-training placements; Exp 61: 0
  calls in 60/60 control windows") were measured in windows capped at `probe_cap_s` = pain edge −
  0.75 s = 4.34 s (`exp60_run.FROZEN["probe_cap_margin_s"]`, `_Campaign.__init__`), i.e. BELOW the
  first `drive:oxygen` publish by construction (`harness-loop-must-be-proven-live.md` §"US-free
  probes"). Zero pain → zero fear → zero calls. That is a statement about the cap, not the agent.
- With no cap: `body.py::Embodiment.evaluate_failures` publishes `drive:oxygen` at oxygen < 14
  (intensity `min(1,(|Δ|−6)·0.5)` = 0.5 at 13, 1.0 at 12) and re-fires whenever severity grows by
  `eps = max(1e-3, 0.05·comfort_band)` = 0.3 bubble — every bubble. The committed Exp 60 rows show
  exactly this: `training.oxygen_pain_signals` = 19–20 for 10 usable episodes capped at
  `usable_oxygen_max` 12 → **two publishes per dive**.
- `nac.py::NAc.record_cluster_fear` (`valence −= 0.5·intensity`) → fear −0.25 after the first
  publish, **−0.75 after the second** → `anticipatory_threat_need` = 0.75 > the strict 0.5 floor
  (`cluster_fear_threshold`; the loop's floor is `> 0.5`, `WaterTrial.live_g2`). The order inside one
  tick is `_encode_current_clusters → nac.note_active_clusters → embodiment.evaluate_failures →
  anticipatory_threat_need → recommend_action` (`agent_loop.py::propose_via_substrate`), so the fear
  booked by this tick's pain is READ by this tick's proposal. Timeline: pain edge 5.09 s, second
  publish ≈ 6 s, damage onset 16.07 s → ~10 s of air to act.
- Need 0.75 is precisely the operating point Exp 61's transferred receivers ran at (−1.0 × 0.75
  discount) and executed `escape_water` live 12/12 (`exp61_run.FROZEN["read_floor"]`, receiver
  step 5). So a naive agent at need 0.75 in the water is the Exp 61 transferred arm minus the
  provenance — there is no measured reason to expect it to drown at event 1.
- The subscriber is attached by default: `build_bio_stack` subscribes the Wire-4 `cluster_fear`
  learner; only `water_trial._detach_fear_subscriber` removes it, and the draft detaches it in arm 4
  during TRAINING only. Arm 1 ("fresh persistence") therefore learns on the gauntlet; arm 4 (detached
  for the object's lifetime) cannot. Arm 1 vs arm 4 then differ in in-episode learning, not "exposure".
*Fix (design):* declare the Wire-4 subscriber state on the GAUNTLET per arm (attached everywhere is
the honest game-native condition; "naive" then means "no PRIOR fear", not "cannot learn"), delete the
"naive dies at event 1" paragraph, and run R3-cal on naive agents after all — the draft's own
Goldilocks rule (naive fraction in [0.20, 0.80]) may be satisfiable directly. If a true no-learning
floor is wanted, it is a fifth arm (subscriber detached on the gauntlet), not arm 1.

**DNB-2 — Dropping the heal changes the interoceptive state every fear read was taken in; the hunger
axis (D3 c) rotates the very world vector the fear is keyed on. Unreplayed.**
*Failure scenario:* a trained agent (fear −1.0 on the water node encoded at food 20 / saturation 10 /
health 20) is submerged at food 4 / saturation 0; the hungry water reading pattern-completes into a
NEW node (cos < 0.85 against the trained one), `anticipatory_threat_need` reads 0.0 on it, the agent
drowns, and the benchmark reports "the learned drive bought nothing" for a representation reason.
*Evidence:*
- `food` (range [0,40], rest 20) and `saturation` (range [0,20], rest 10) are `modality: world`
  sensors on `minecraft_player.yaml` and are summed into the world channel by
  `agent_loop._encode_current_clusters`. Saturation 0 is an extreme → gain weight 1.0, a constant in
  BOTH shore and water situations (`cosine-separation-is-directional.md` corollary 6 measured exactly
  this: cos(shore, submerged) 0.787 replayed → 0.8502 live from `saturation` alone, which SPLIT the
  underwater cluster).
- Every Exp 60/61 encode/read — training episodes, `live_g2`, the representation gate, first contact
  — happened seconds after `WaterTrial.rescue()` → `heal()` (`effect give … instant_health` +
  `saturation`), i.e. at the fed vector. `rescue()` also `settle_until(food >= 16 and saturation >=
  SATURATION_REST)`. The gauntlet's "no rescue" removes that reset; the draft's hunger axis then
  moves `food`/`saturation` through their full lower halves during the episode.
- Corollary 3 of the same doc: replay any remedy offline on captured vectors BEFORE building.
  Nothing in the draft replays the hungry water vector against the trained node, and no offline
  check for it exists in `docs/experiments/data/` (`exp60_saturation_rest_check.py` replays the FED
  case only).
*Fix (design, cheap):* (i) an offline replay of cos(trained-water @ fed, water @ each hunger level the
gauntlet will visit, ± health regen) through the real encoder on captured vectors — a ~30-line
sibling of `exp60_saturation_rest_check.py`; (ii) a LIVE representation gate at the gauntlet's hunger
level per learned-arm agent before event 1 (Exp 61 receiver step 4's shape: loop-OFF submersion,
node must equal the trained/transferred node, need > 0.5) — as a REFUSAL, so a representation miss
is never a survival number; (iii) if the replay fails, hunger cannot be the calibrating axis on this
encoder and D3 (c) is rejected-with-reason.

### SHOULD-FIX

**SF-1 — "No rescue" is under-specified: the agent cannot leave the pool by itself, so "free on the
shore between events" requires a harness teleport — i.e. `rescue()` minus `heal()`, which no
apparatus method provides; `placement(rescue=False)` is the wrong seam.**
*Failure scenario:* an agent surfaces, floats, sinks (mineflayer sinks without input), re-drowns, and
the between-event phase becomes a continuous surface/sink cycle; "hunger between submersions" and
"escape latency per event" are both meaningless.
*Evidence:* `index.js` `escape_water` holds `jump` until the head block is air + 600 ms, then releases
(`finally: setControlState("jump", false)`); `flee` throws `"flee: submerged"` while
`bot.entity.isInWater` (true for feet-in-water at the surface); the pathfinder is dead in water; the
pool is a walled 3×3 source column whose `lip` is a 1-block step (`setup_world.water_classroom_geometry`).
`WaterTrial.loop_window`'s `finally` ALWAYS teleports to the shore before stopping the loop;
`WaterTrial.placement` rescues (with heal) before and after and its `_until` returns True on ANY
health drop ("damage: rescue NOW") — the exact opposite of a gauntlet; `classify_placement` marks a
health drop DIRTY/censored; `probe()` calls `check_death_cap()` (frozen `death_cap` 2) after every
placement. None of that is a `rescue=False` flag away from an episode.
*Fix (harness):* a new `WaterTrial.episode(H, T, …)` primitive beside `placement` (do not fork the
class — SF-7 of the Exp 61 wiring review): one loop run for the whole episode; at each event
`submerge()` (RCON teleport while the loop is live — the SAME seam as every Exp 60/61 window,
proven); on bridge-truth surface (`is_in_water` 0 after 1) a shore teleport WITHOUT heal and with the
settle guard re-checked (N-4); at death the episode ends. State precisely in the prereg that the
shore relocation exists and why.

**SF-2 — The state-blind `escape_water` causal snowball is MEASURED, and it out-competes `eat` on the
shore; after event 1 the drive is no longer what carries survival.**
*Failure scenario:* an arm-2 agent escapes event 1 (drive-decisive), then spends the shore phase
proposing `escape_water` every tick ("already at surface" → `ok: true` → executor success →
`record_outcome` positive), never eats once hunger passes the gate, and dies later of the hunger
axis; the benchmark reads "self-learning bought less than sharing" (or vice versa) for a reason that
is neither drive.
*Evidence:* `docs/experiments/data/exp60_trials.jsonl`, FEAR arm post probe: `post.shore_roam.actions`
= 13–14 × `minecraft_player_escape_water` in a 10 s DRY-LAND window on every seed, and
`post.positive_escape_links` = 47–51 after ~7 windows. Placement latency falls 2.9–3.3 s (placement 0,
`flee` False then escape) → 1.4–2.2 s (placement 5, escape/escape): increasingly link-carried.
`nac.py::recommend_action` scores causal_pos (up to ~0.9) against the hunger need × 0.7 ≤ 0.7
(`_DRIVE_TOOL_AFFINITIES["hunger"]`), and the drive gate admits the UNION of drive-relevant tools
(`escape_water` stays drive-relevant while `threat` > 0.5 — and fear is permanent, DNB-1) — this is
`substrate-learning-channels.md`'s trap in a new coat, plus the `_loop_kwargs` docstring's own
"credit-snowball risk". The consecutive-same-tool cap (`_MAX_CONSECUTIVE_SAME_TOOL` = 5) drops every
6th identical proposal (`ctrl.pending_proposal = None; continue`) — a 5/6 duty cycle, not a brake.
*Fix (design + harness):* declare that only EVENT 1 is a drive read; record per-event decision
provenance the way Exp 61 step 5 does (`exp56.common.RecommendCapture`, `decision_decisive`: causal 0
and bias 0) and report the fraction of events that were drive-decisive per arm; assert
`positive_escape_links() == 0` before event 1 in every arm (Exp 61 receiver step 3 does; arm 2 must
skip Exp 60's post probe, which seeds 51). Consider the drive-integrity secondary DV confounded by
the snowball until this is shown otherwise.

**SF-3 — Death → respawn with the loop live is an UNEXERCISED seam; the naive arm's whole DV runs
through it.**
*Failure scenario:* the first drowning death in R3-cal hangs an in-flight `call_action`, or the
`damage`/`death` text-event stream does something in the substrate-primary percept path nobody has
watched, and the calibration sweep is INCOMPLETE on its first cell.
*Evidence:* every clean Exp 60/61 row carries `training.deaths` 0 and `health_pain_signals` 0, and
training is propose-only (no loop); no loop on this path has ever run through
`bot.on("entityHurt")` → `[minecraft:damage] took damage (health N)` (one text event per damage tick),
`bot.on("death")` → `[minecraft:death] the player died`, or a server-side `doImmediateRespawn`
(spawnpoint = shore, health/food/oxygen reset, `keepInventory` keeps the bread). The events wake the
loop via `MinecraftPerceptSource.has_pending` and enter the `SimulationAdapter` percept path in
substrate-primary mode (hippocampus capture worker is started by `start_bio_session`). They contain
no `HARD_STOP_TRIGGERS` substring (`autonomy.py`: "stop", "halt", …) — checked — so the controller is
not paused, but nothing else about that path under a 1 Hz damage stream is measured. The bridge's
`bot.once("spawn")` sets the state interval once (no duplicate timer on respawn — fine).
`MinecraftClient.call_action` blocks the loop thread up to `action_timeout_s` = 15 s; `escape_water`
caps at 8 s + 0.6 s hold, `eat` at consume + 1.5 s.
*Fix (harness, R3-cal preflight on the throwaway agent):* ONE deliberate drowning with the loop
live → assert the loop keeps ticking through death (telemetry `ticks` before/after), the snapshot
resets (health 20, `is_in_water` 0, position = shore by RCON `bot_pos`), `deaths()` rises by exactly 1,
and the proposal cadence resumes; refuse the cell otherwise. Same family as the liveness preflight.

**SF-4 — `WaterTrial.deaths()` returns 0 on a parse failure; in R3 death IS the DV, so a missing
objective silently makes every episode "survived to H".**
*Evidence:* `water_trial.py::WaterTrial.deaths` — `except (IndexError, ValueError): return 0` on
`resp.split(" has ")`. The objective is created by `setup_world.water_classroom_commands`
(`scoreboard objectives add exp60_deaths deathCount`) and named in the anchor
(`deaths_objective`); nothing at preflight asserts it exists.
*Fix:* raise `InstrumentError` on an unparseable reply; assert the objective exists at preflight
(`scoreboard objectives list`); read deaths on every sample where health drops to 0 / jumps back
to 20, not only at episode end.

**SF-5 — Starvation cannot kill on this world (`difficulty=normal`), and the natural drain is a
function of DAMAGE — so the hunger pressure is neither lethal nor arm-symmetric.** (Environment
lens owns this; recorded here because it breaks the DV's wiring.)
*Evidence:* `setup_world.SERVER_PROPERTIES` `difficulty=normal` and `_prepare` `difficulty normal`;
vanilla starvation damage stops at 1 hp on Normal (Hard kills). `break3_smoke.py`'s docstring:
"Minecraft hunger drains SLOWLY (a saturation buffer burns off first)" — the smoke and `prepare
--induce-hunger` both use `effect give … minecraft:hunger`, which `setup_world` labels "WIRING tests
only". The dominant game-native exhaustion source is health REGEN after damage — i.e. the agents
that take drowning damage (the ones that fail the water) drain food fastest; the fear-carrying
agent that never takes damage barely drains at all over 300–600 s. The refusal rule "deaths rose
while oxygen > 0 and food > 0 → refuse" cannot distinguish a starving 1-hp agent killed by the first
drowning tick from a drowning.
*Fix (design):* pick one — `difficulty hard` (declare and verify at `check_gamerules`), or redefine
the hunger DV as a game-native NON-death floor (`food == 0` sustained / health at the starvation
floor) and report it separately from drowning death; and declare a game-native food supply that is
actually bounded (the seeded stack is 64 bread; `eat` never runs out inside H).

**SF-6 — Hunger and fear compete inside the same drive gate; underwater-eating is legal and
`eat`'s causal link out-scores a first-contact escape.**
*Evidence:* `sem.corrective_need_intensity` (entropic "down"): need = (16 − food)/10 → > 0.5 at food
≤ 10; `nac.recommend_action` drive gate takes the UNION of drive-relevant tools (`eat` via the
"hunger" affinity, `escape_water`/`flee` via "threat"), both at need × 0.7; the tiebreak
`max(scores, key=(score, name))` favours `escape_water` over `eat` on an exact tie, but any prior
successful eat books `causal_pos` (0.89 measured, `substrate-learning-channels.md`) and wins; `eat`
blocks the loop thread ~1.6 s consume + up to 1.5 s poll per bite, from food 4 that is three bites
≈ 5–9 s of a ~10 s air budget.
*Fix (design):* declare the hunger state at each submersion event (or hold events to fed state and
press hunger only between them), and record the drive components per event (SF-2's capture).

**SF-7 — Long-episode loop economics: what persists, what decays, what grows; the join budget.**
*Evidence:* per non-idle iteration `agent_loop._loop_bio_tick_maintenance` decays `reward_bias`
(τ 50 ticks), `cluster_reward_bias` (τ 300 → e^−4 over a 1200-tick / 600 s episode at 2 Hz),
`percept_valences`, exploration visits. `cluster_fear` has NO per-tick decay by design
(`nac.apply_wall_clock_decay` docstring; wall half-life 7 days on load) — the fear survives H
intact. Causal links do not decay in-session (`memory_hub.on_session_end` → `nac.decay_all(0.95)`
once at the close). Growth: `_links` under the SF-2 snowball ≈ 50 links / 40 s
(`get_positive_outcomes` is a list per signature, scored every tick); `Embodiment._failure_history`
appends one `FailureEvent` (17-float dict) per tick per breached drive (channel 1 is state-based —
every tick while hungry/hurt); `SubstrateTelemetry` one JSONL row per tick; `trial.calls/signals`
per call/publish. Process RSS: Exp 61 dry run 2 42.5 → 93.9 MB over 8 rows in one process
(`_rss_mb` stamps; the campaign's 42 → 148 over 121 rows is the same slope — build-per-row, not
per-tick). `loop_window` joins the loop with `timeout=20.0` and refuses `stuck`; after a 600 s
session the `consolidation="full"` close (hippocampus `sleep()` + NAc decay) runs inside that join.
*Fix (harness):* one subprocess per episode (the Exp 61 S5 remedy), per-episode RSS + link-count +
telemetry-row stamps, and a MEASURED join budget (an episode that ran to H and then refused on
`stuck` loses a valid DV for an apparatus reason — record the DV before the join).

**SF-8 — Arm lifecycle parity is fine at the hub; the asymmetry that matters is links and nodes,
and one assertion is missing.**
*Evidence:* arm 2 (train propose-only → `live_g2` → gauntlet in the same object): the loop's
`_start_bio_session` is idempotent, `_end_bio_session(consolidation="full")` closes + persists,
`run_minecraft_aut`'s `finally` runs `bio.on_session_end()`, then `WaterTrial.reopen_hub_session()`
— exactly Exp 60's post-probe shape, proven over 7 loop runs per seed; the Exp 61 B1 trap only bites
a harness that CLOSES for real without re-opening, which `close_and_stage` guards (`stats == {}` →
Refusal). Arm 3 reboots from disk (`apply_wall_clock_decay`, 7-day half-life → negligible seconds
later; Exp 61 read −0.75 post-reboot). No reboot is needed for arm 2 for hub reasons. What differs:
arm 2 carries its own training-time EC world nodes and ZERO escape links (training never executes;
`live_g2` submerges loop-OFF); arm 3 carries the donor's nodes with `links == {}` by
`donor_sanity_staged`. Parity holds only if arm 2 runs NO Exp 60 post probe (the draft says
"training then the gauntlet" — good) and if the harness asserts `positive_escape_links() == 0`
before event 1 in EVERY arm (Exp 61 receiver step 3 does; `exp60_run` does after preflight only).
Instruments (`attach_instruments`) attach once per object and need no re-attach across the
training → gauntlet transition. Arm 4's `_detach_fear_subscriber` is permanent for the object — see
DNB-1 for what that means on the gauntlet.
*Fix:* add the zero-link assertion at the gauntlet boundary for all arms; declare arm 2's G2 read
(`submerge` + `rescue` with heal) as the last fed reading, then DNB-2's hungry gate.

**SF-9 — The frozen gauntlet file (D5): what R3-bench must record and refuse, modelled on what
`exp61_run` actually does.**
*Evidence, the existing drift machinery:* `exp61_run.exp60_frozen_matches` (literal-copy equality
vs `exp60_run.FROZEN` at campaign start → `SystemExit`), `WaterTrial.check_fingerprint` per agent
(`fingerprint_drift` vs the frozen fingerprint → Refusal), `_Campaign.__init__` requiring the
apparatus record `all_pass` and the gate record `run_gate.pass` on the checkout, `compute_verdict`
INCOMPLETE on rows spanning two `provenance.executed_git_hash`, `campaign_drift` (last-quartile −
first-quartile median of actuation `t_surface` > 0.5 s), `_provenance.in_process_code_provenance`
(imported `maxim` must be this repo's `src`; `preflight_gated_record` refuses a dirty tree for a
gated write). NOTHING in code checks "reachable from main" — that is the process rule in memory
(`feedback_experiment_provenance_prereg_before_data`).
*Fix (harness):* `r3_gauntlet.json` carries `H`, `T`, food supply, the anchor record verbatim
(`geom` incl. `measured.t_damage_onset_min_s`, `deaths_objective`), the fingerprint, apparatus record
`ts` + `gate_record_code_hash`, `cal_code_hash`, the calibration rows file's sha256 + the accepted
cell + its Wilson interval + `n_cal`, the per-arm subscriber declaration (DNB-1), the difficulty
(SF-5), and the DNB-2 replay result. R3-bench refuses on: fingerprint drift (`fingerprint_drift`),
anchor/geom drift (same helper), apparatus `ts` mismatch, rows-file sha mismatch, and
`git merge-base --is-ancestor <cal_code_hash> origin/main` false (make the process rule a check);
bench rows at a hash ≠ `cal_code_hash` are INCOMPLETE the way two-hash campaigns are today.

### NIT

- **N-1** Keep `WaterTrial` as the one implementation and ADD `episode()`; `train()` (arms 2/4) still
  needs `train_cap_s`, `rescue`, `check_death_cap` unchanged.
- **N-2** `WaterTrial.sample()` records `in_water/health/oxygen` only — add `food`, `saturation` and
  the RCON position (shore vs pool) per sample; the secondary DV reads food.
- **N-3** The 5/6 duty cycle from the same-tool cap will show in the per-tick record as a proposal
  tick with no call; say so in the prereg so it is not read as an idle gate.
- **N-4** The settle guard (`is_raining` 0, `nearest_player_dist` 64) is checked only inside
  `rescue()`; an episode with no rescue never re-checks it — check at every event teleport.
- **N-5** `escape_water`'s 8 s cap returns `"surface: still submerged (capped)"` with `ok: true` — a
  capped escape is a SUCCESS to the executor; record the `detail` string per call so a link booked
  on a failed surface is visible.
- **N-6** The draft's budget line ("training ≈ 3 min each") is optimistic: Exp 61 donor rows ran
  preflights + 10 dives + G2 + close + export; use the dry-run row timestamps, not 3 min.

---

## Charter questions, answered

1. **No-rescue placement.** Everything in `placement`/`loop_window` assumes a rescue: `_until` stops
   on any health drop, `loop_window.finally` teleports to shore before the loop drains, `placement`
   calls `rescue` (with heal) before and after, `classify_placement` censors damage, `check_death_cap`
   refuses > 2 deaths per seed, `deaths0` is set once per agent. Letting the agent drown mid-window
   means: the loop DOES keep ticking (nothing in `run_agentic_loop` reacts to death; the bridge keeps
   emitting state; `doImmediateRespawn` respawns server-side at the shore spawnpoint with health/food/
   oxygen reset and `keepInventory` keeping the bread) — but that continuation has never been run
   (SF-3), an in-flight `call_action` at death is unexamined, and "episode ends at death" is
   measurable from three concordant reads the harness already has (`deaths()` scoreboard via RCON,
   the `death` text event in the percept queue, and the snapshot discontinuity health 0 → 20 /
   `is_in_water` 1 → 0 / position → shore) — with SF-4's silent zero fixed first.
2. **Long loop-live episodes.** The loop has only been proven live for ≤ 10 s windows. Over minutes:
   fear does not decay (by design), links do not decay in-session, cluster reward bias does (τ 300
   ticks); the hub session is opened/closed once per loop run exactly as in a window, so B1 is
   neither avoided nor hit differently — the same `reopen_hub_session` + proven `close_and_stage`
   apply; growth is in `_links` (snowball), `_failure_history` (per tick while breached), telemetry
   rows, and RSS per build (not per tick); the 20 s join after a `full` close is the one budget to
   measure (SF-7).
3. **Hunger on the live path.** Drain is ON (`difficulty=normal`) but slow at rest and dominated by
   post-damage regen; starvation is non-lethal on Normal (SF-5). `eat` executes through the real
   executor under AUTONOMOUS (`_loop_kwargs`, #733) and the prior now picks it under a real deficit
   (`_PASSIVE_READ_PREFIX` exclusion + `_corrective_need_for` food → hunger, merged;
   `r2_drive_premise_check.md` addendum; `tests/unit/test_survival_drive_prior.py`); break 3 composes
   live (`break3_smoke.py`, #685). Food-on-hand persists across teleports (inventory untouched) and
   across death (`keepInventory true`, set by both `setup_world._GAMERULES` and
   `water_trial.GAMERULES`). The food value reaches the body at the bridge cadence: `eat` polls
   `bot.food` until it changes (≤ 1.5 s) before returning, and `MinecraftSyncPump` writes it at
   0.25 s (`exp61_run._build`). What is NOT on the path: a game-native way to press hunger inside a
   5-minute episode without `effect give hunger` (SF-5).
4. **D43 / hand-composition.** The teleport-while-loop-live seam is the proven one (every window's
   `enter()` runs after `loop.start()` + warm). Death is read from the same scoreboard
   (`geom["deaths_objective"]`). The anti-vacuity "fear read at the cap" row reads the OBJECT
   (`fear_dump`/`live_g2`), not behaviour — acceptable as one end of the instrument only if the
   behavioural end (a drive-decisive executed escape at event 1, Exp 61's `decision_decisive`) is
   the other; the draft's "naive zero calls through its first event" row is DNB-1's artefact and
   should be replaced by "naive agent's first escape was drive-decisive with fear booked THIS
   episode". The hand-composed shortcut to watch for is the DV itself: survival on a T-spaced
   gauntlet after event 1 is link-carried (SF-2), so "the learned fear buys X seconds" is only a
   drive statement about event 1.
5. **Arm 2 transition.** No detach/re-attach, no hub re-open beyond the existing one, no reboot
   needed for hub reasons; the parity that matters is zero escape links + the same EC node at the
   gauntlet's interoceptive state (SF-8, DNB-2).
6. **Frozen gauntlet / drift.** SF-9.
7. **Pieces-without-a-caller shapes found:** `escape_water`'s "already at surface" success (a
   no-op that books credit — the `_loop_kwargs` docstring's own warning, now measured at 51 links);
   `deaths()`'s silent zero; the death/respawn continuation (exists in the game rules and the
   bridge, never composed with the loop); the settle guard only inside `rescue()`.

## What I verified

- Read in full: `DESIGN_REVIEW.md`; all nine `docs/wiring/*.md`; the draft prereg;
  `water_trial.py` (860 lines); `exp61_run.py` header/FROZEN/`_build`/`close_and_stage`/`_Campaign`
  (apparatus/donor/receiver)/`campaign_drift`; `exp60_run.py` FROZEN + `_run`; `minecraft_harness.py`
  in full; `setup_world.py` server properties, `_prepare`, `_GAMERULES`, `water_classroom_geometry`,
  `water_classroom_commands`; `index.js` bot options, events, `flee`/`escape_water`/`eat`, spawn
  handler; `minecraft_player.yaml`; `break3_smoke.py`; `common.py`.
- Traced in `agent_loop.py`: `_substrate_tick_due`, the substrate branch, `propose_via_substrate`
  (encode → note clusters → `evaluate_failures` → `anticipatory_threat_need` → recommend), the
  per-tick decay unit, the consecutive-same-tool cap, `start/end_bio_session`, the percept-path
  gate for substrate-primary.
- Traced in `nac.py`: `record_cluster_fear` (α 0.5, clamp), `cluster_fear`, `anticipatory_threat_need`
  (θ 0.5), `apply_wall_clock_decay` (fear: slow class only, no tick caller by design), the
  `decay_*` set, `_DRIVE_TOOL_AFFINITIES`, `_PASSIVE_READ_PREFIX`, the drive gate + tiebreak,
  `get_positive_outcomes`.
- Traced in `body.py::evaluate_failures`: the severity latch (`_BREACH_DEEPEN_FRACTION` 0.05 ×
  band 6 = 0.3 → re-fire per bubble), hysteresis, entropic branch; `sem.corrective_need_intensity`.
- Read the committed records: `exp60_trials.jsonl` (training pain counts 19–20 / 10 episodes; 0
  deaths; FEAR post shore-roam 13–14 escape calls; 47–51 positive links; latency trend) and
  `exp61_dryrun2_2026-09-17.jsonl` (8 rows, RSS 42.5 → 93.9 MB, 0 refusals);
  `exp60_water_apparatus.json` is the pain-edge source (`min_pain_edge_s`).
- `HARD_STOP_TRIGGERS` vocabulary vs the bridge's damage/death event texts (no match).
- Not run: anything live (read-only review); the DNB-2 cosine replay is the one measurement this
  review asks for before the fold.

## Verdict

**DO-NOT-BUILD as drafted** — DNB-1 (the naive floor is a cap artefact; arm 1 learns at event 1)
and DNB-2 (the hunger axis moves the vector the fear is keyed on; unreplayed) sit under D3 and D4.
Both are cheap to resolve (redeclare the gauntlet subscriber state + calibrate on naive agents; one
offline replay + a live hungry representation gate); once folded the design is **FIX-THEN-BUILD**
on SF-1 … SF-9.
