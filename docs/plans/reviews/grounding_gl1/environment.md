# GL1 four-lens design review: ENVIRONMENT lens

Plans reviewed: `docs/plans/grounding.md` (umbrella) and `docs/plans/autonomic_layer.md` (GL2a/b/c), at
worktree `.worktrees/gl1` = `origin/main` c80d619b. Charter: `docs/experiments/DESIGN_REVIEW.md`.
Lens question: do the named worlds (the cradle sim, Paper 1.20.4 Minecraft through the mineflayer
bridge, the scripted water trial) natively afford what each stage needs, without synthetic sensors or
rewards (D1)? Are the needed states and actions reachable and measurable, and does the bridge/world behave?
Owner decisions G1–G8 and the 2026-10-09 GL2a decisions (tool-path record only, measured drop-oldest
bound, unweighted valence, urgency = pressure only) are taken as given. Nothing below re-opens them.

**Verdict: 2 DO-NOT-BUILD / 9 SHOULD-FIX / 4 NIT.**

- DNB-1 (GL2a): the record's `drive_delta` straddles wall-clock drift. `evaluate_failures` applies
  `tick_vital_drift(now − _last_poll)` inside the tool, so on LLM-primary runs the drift accumulated over the
  whole LLM turn is booked as the action's consequence. The GL2a gate cannot see this.
- DNB-2 (GL2c): on the earned survival rows, the satiation crossings come from the apparatus (rescue
  teleports, `/effect` heals, respawn, teacher and mother feeds), not from the agent's own `escape_water`.
  The record would stamp them `experienced`, and in Exp 60 they would fall unevenly across the two arms.

---

## Findings

### DNB-1 — GL2a: `drive_delta` straddles wall-clock drift, so the "consequence" is partly time passing

- **Stage:** GL2a (the record), and through it GL4 S0b/S2 (it is their target).
- **Issue:** `Embodiment.evaluate_failures` (`embodiment/body.py::Embodiment.evaluate_failures`) starts by
  applying drift for the wall-clock time since the previous evaluation: `now = time.time()`,
  `tick_vital_drift(now − self._last_poll)`. On the tool path the executor reads the "before" state in
  `runtime/executor.py::Executor._run_started` (`pressure_before = self._drive_pressure_snapshot()`, just
  before `tool.run`). Inside the tool, `embodiment/tool_bridge.py::ModulatorAffordanceTool.execute` applies the
  `self_effect` and then calls `evaluate_failures()`, which applies **all** drift accumulated since the last
  evaluation. A record built from executor-before / post-tool-after therefore books that drift as the
  action's consequence.
  - On **LLM-primary** the last evaluation is `runtime/loop_gates.py`'s once-per-iteration embodiment tick
    (its docstring: "advances wall-clock drift"), which runs before the LLM call. So the drift interval is
    the whole LLM latency.
  - Shipped rates: `infant_humanoid` hunger 0.006/s and thirst 0.008/s; `infant_humanoid_chilled` `cold`
    0.08/s (Exp 42); `arms.thermal` 0.0008/s. A 20 s LLM turn on the chilled body therefore adds +1.6 `cold`
    (clamped at 1.0), which swamps `warm_self`'s −0.3. That inverts the sign of "relief" on the very
    dissociation pairs GL4/GL5 rely on.
  - Substrate-primary is better but not clean: drift still covers the encode/propose/dispatch span of one pass.
  - The existing record-only relief (`tool_bridge.py::_drive_progress_by_drive`) does **not** have this
    problem, because it diffs around `_apply_sensor_deltas` before `evaluate_failures`. But GL2a is told not
    to use it while #1161 keeps it blind to modulator drives, which pushes GL2a onto the drift-straddling bracket.
- **Why it matters:** the record's stated meaning is "what this invocation did to my body", and §5.5 says the
  plan "only guarantees the record exists and is honest". On the cradle runtime that produces the narrator
  data (LLM-primary `--sim`, Exp 37/38), and on any body with a fast entropic drive, the record is mostly drift.
  The GL2a gate cannot catch it: the scripted sequence (`feel` ×2, `warm_self` ×2, `touch`) runs in a unit
  test with milliseconds between calls, so drift there is about 1e-6. S0b would then count "consequence
  classes" and "dissociation pairs" that are artefacts of LLM latency.
- **Fix (plan text, before GL2a is built):** state where "before" and "after" are read and how drift is
  separated. One option inside the exempt set (`body.py`, `sem.py`, `executor.py`):
  1. Record the drift interval on the record (`extra["drift_dt_s"]`, i.e. `time.time() − _last_poll` read at
     `pressure_before`, via a read-only `Embodiment` accessor).
  2. Compute the drift share from the declared specs with a pure helper factored out of `tick_vital_drift`'s
     arithmetic. That arithmetic is linear with a set-point and [0,1] clamp; the helper needs no behaviour change.
  3. Report `drive_delta` net of it, keeping the observed delta in `extra`.

  Advancing drift early from the executor is **not** acceptable: it moves pain-publication cadence
  (`loop_gates.py`'s cadence caveat; `deferred/transition_based_drive_pain.md`'s trigger).

  Add to the GL2a gate a `_StepClock` case (`tests/unit/_loop_harness.py::_StepClock` patches the global
  `time`, so `body.py` sees it) that advances 20 s between the loop tick and the tool call on
  `infant_humanoid_chilled`. Its hand-computed table must show `warm_self`'s `cold` delta as −0.3 net of drift.

### DNB-2 — GL2c: on the earned survival rows, the satiation crossings are caused by the apparatus

- **Stage:** GL2c (and the out-of-band producer that lands after the fence, which mints the same crossings
  as records).
- **Issue:** G7's rationale (umbrella G7; autonomic §1.2 item 2, §3.5) says the `oxygen` latch clear fires
  "at Exp 60's own `escape_water` contingency, since its training trials breach". The code and the frozen
  record say otherwise. The crossings land where the **experimenter** acts:
  - **Exp 60/61 training is propose-only.** `scripts/survival_world/water_trial.py::WaterTrial.train` runs
    `propose_via_substrate` with no execution. A usable episode requires `oxygen ≤ usable_oxygen_max` (12)
    with `drive:oxygen` pain, i.e. a latched breach (the breach starts below 14). Each episode then ends in
    `WaterTrial.rescue`:
    - an RCON teleport to shore;
    - `settle_until(oxygen ≥ RECOVER_OXYGEN_MIN = 19)`;
    - `heal()` = `/effect instant_health` + `/effect saturation`;
    - four "healthy ticks: the latch observes recovery", in which the clear actually happens. The homeostatic
      clear point is `deviation ≤ comfort_band·(1−_BREACH_HYSTERESIS)` = 4.8, i.e. oxygen ≥ 15.2.

    So **every** training crossing (K = 10 per agent) is caused by the rescue teleport.
  - **Exp 60 probes never breach.** The frozen record `docs/experiments/data/exp60_trials.jsonl` has
    `probe_cap_s` 4.335 < `pain_edge_min_s` 5.085: probes end before oxygen crosses 14, by design ("no-damage
    probes capped at the measured onset"). A FEAR-arm escape therefore happens with **no** latched breach and
    mints no satiation.
  - **R3 (lethal window):** every death respawns at health 20 / oxygen 20
    (`scripted_water.py::ScriptedWaterBridge._snapshot` models the live respawn). That clears both latches,
    so dying would mint a positive satiation.
  - **Teacher and mother feeds:** Exp 56/57's teacher (`scripts/exp56/common.py`: `_apply_sensor_deltas(d1:
    −0.5)` then `nac.credit_operant_reward`) and Exp 52's mother
    (`simulation/cradle_mother.py::reactive_mother_tick`, the same shape) write the drive directly. A feed
    from `d1` ≥ 0.7 lands ≤ 0.2 < 0.3 = `satisfaction_threshold`, so the next `evaluate_failures` clears the
    latch. That is a second positive on an event already credited by `credit_operant_reward`. §3.5's
    double-credit note names only channel 3.
- **Why it matters:**
  1. G6's provenance set has no kind for apparatus or teacher writes. These run on the loop/harness thread
     outside the narrated scope, so they would be stamped `experienced`: a D1 violation (a synthetic reward
     delivered as world-native).
  2. In Exp 60 the positive lands at the shore cluster after every training episode and after every
     **censored** probe (rescue). Censoring is more common in the arm that did *not* escape, so a positive
     producer is **differential by arm** on the comparison T1-13 measures. "Batched re-run, PASS or BROKEN"
     cannot tell a mechanism effect from this apparatus effect.
  3. The restated refusals (`exp61_run.py::donor_sanity_staged`, `r3_run.py::_R3._boundary`) would trip on
     apparatus events, not on agent experience.
- **Fix (plan text, before GL2c's joint review):**
  - Correct the G7 rationale's mechanism. G7's decision stands: the rows fire and the re-run is batched. Only
    the *where* is wrong.
  - Require an **apparatus/teacher provenance** for writes made by harness RCON (teleport, `/effect`),
    respawn discontinuities and teacher/mother feeds. G6 leaves additional kinds open at GL3.B1, so this does
    not re-litigate it. Such writes must be excluded from satiation delivery, or carried at a declared
    discount like `narrated`.
  - Give the producer a scope API the harnesses enter around `rescue`/`heal`/respawn (the same thread-local
    pattern as the narrated scope).
  - Add to GL2c's flag-on gate: "a rescue-caused crossing delivers nothing, or the declared discount".
  - Extend §3.5's double-credit rule to `credit_operant_reward`.
  - Have the batched re-run report satiation counts per arm and per cause (agent action / apparatus / respawn).

### SF-1 — GL2a gate: one cited "executing" check does not execute, the other is a known wall-clock flake

- **Stage:** GL2a (also GL2c's flag-off gate).
- **Issue:** the gate cites `tests/unit/test_water_trial_smoke.py` and `tests/unit/test_exp61_run.py` as
  survival checks that **execute** the producer.
  - `test_exp61_run.py` runs no loop. Its tests are verdict/donor-sanity/statistics over hand-built files,
    plus an anti-vacuity fold, so it never reaches `Executor._stamp_invocation`.
  - In `test_water_trial_smoke.py`, only `test_water_trial_ticks_acts_and_the_staging_close_persists_fear`
    executes `escape_water` through the executor. That test is still on **wall time**: `train_cap_s=8.0`,
    `pl1["latency_s"] < 3.0` inside `probe_cap_s=3.0`, `t_surface < 2.5`. It is named in **open #954**
    ("same race class as #951"); only `test_donor_sequence…` moved to `StepClock`.
  - GL2a adds per-invocation and per-capture work (snapshot reads, the `RLock`, the sequencer mint,
    `extra["interoception"]` serialization), which narrows those margins.
  - GL2c's flag-off gate also cites `test_r3_run.py`, which #954 lists in the same class.
- **Why it matters:** "passes unchanged" proves nothing about the producer when a test never calls it. A
  wall-time flake read as a GL2a regression (or a lucky pass) is the #951 failure shape again.
- **Fix:**
  - Cite only executing tests, and add a positive assertion that the `escape_water` invocation's `ToolOutput`
    carries a record with a `pid`, so the deletion probe re-reds it.
  - Land #954 for the cited tests first (move them onto `StepClock`/`LockstepTime` with a slowdown probe), or
    state their margins.
  - Drop `test_exp61_run.py` from the "executing" list.

### SF-2 — Minecraft: the tool-path record sees the world at actuator return, so escape relief and every oxygen crossing are out-of-band

- **Stage:** GL2a (strict tool-path default), GL2c, GL4 S0b (Minecraft floor), GL5 (Minecraft arm).
- **Issue:** `embodiment/backends/minecraft.py::MinecraftWorldBackend.execute` syncs the action_result
  snapshot, and that snapshot is the record's "after".
  - **Live bridge:** `scripts/minecraft_bridge/index.js` `escape_water` returns 600 ms (`SURFACE_HOLD_MS`)
    after the eyes clear. Air refills about 4/300 per game tick (≈ 5.3 oxygen units/s on the bridge's 0–20
    scale), so the latch clear point of 15.2 is usually reached **after** the tool returns. This is game
    mechanics, UNVERIFIED against a live trace.
  - **Scripted bridge:** `scripted_water.py::ScriptedWaterBridge._do_action` returns "surfaced" at once and
    moves the anchor `escape_delay` later. At return the bot is still submerged, so the record shows Δoxygen
    ≤ 0, i.e. *harm*.
  - Pathfinding is dead in water (`flee`, `move_to`: NoPath per the YAML and bridge comments), so underwater
    `move_to` invocations fail fast and carry no Δoxygen spread.
  - Under the owner's tool-path-only GL2a decision, no Minecraft record ever has `satiated` non-empty. Red
    gate (d) is reachable only on the cradle, where SEM effects are synchronous.
- **Why it matters:**
  - GL2c's oxygen crossing is minted at a later loop-thread `evaluate_failures`. GL2c's stated dependency is
    "GL2a" only, but GL2c needs the out-of-band producer **and** the narrated scope, both after the fence.
    Without the scope, a narrator-caused cradle satiation (§3.5 "narrated satiation") would be delivered at
    full weight.
  - GL4 S0b's Minecraft floor ("≥ 100 executed `escape_water` / `move_to` … non-zero Δoxygen spread") cannot
    be met from tool-path records: `move_to` underwater refuses, and `escape_water`'s delta is the swim-up,
    not the recovery.
- **Fix:**
  - Add "out-of-band producer + narrated scope (after the fence)" to GL2c's Depends-on, and to the S0b/GL5
    Minecraft arm.
  - Specify a consequence window: the next out-of-band record within N passes carries `cause_pid` = the
    invocation's `pid` (`CauseRef.cause_pid` already exists for this).
  - Write the GL2a Minecraft assertions so they do not expect relief on `escape_water`'s tool-path record.
  - Replace `move_to` in the S0b floor with an action that acts underwater, or with dry-land graded `health`.

### SF-3 — Cradle burn reachability depends on state; the calibration constraint has a narrow or empty solution in the shipped scenes

- **Stage:** GL2b(ii) (and the GL1 census).
- **Issue:** SEM effects are additive deltas clamped to range (`tool_bridge.py::_apply_sensor_deltas`), and
  the nociceptor reads sensor **state**. The constraint "+0.6 contact *from rest* ≥ 0.4, single `warm_self`
  → 0" is stated from rest, but the cradle scenes do not touch from rest:
  - `cradle_cool_air.feel` takes `arms.thermal` −0.15 per call (the GL2a sequence itself reaches −0.30). A
    touch from −0.30 lands at 0.30.
  - The arc instructs the narrator to "use set_entity_sensor to increase arms.thermal toward 0.8" on approach
    (`simulation/arcs.py`; `cradle_fire_pit.yaml`: "proximity effects are handled by orchestrator sensor
    writes"). A touch from 0.8 clamps at 1.0 (delta +0.2).

  For a touch from a chilled arm to be noxious, the threshold must be < 0.3. For a single `warm_self` from
  rest to give 0, it must be > 0.2. In that band two stacked `warm_self` (0.4) are noxious, and the
  nociceptor fires **below** the thermal comfort band (0.5), which inverts the biology of noxious onset above
  innocuous warm.
- **Why it matters:** in the named world the gate may pass at rest and still fail its purpose. Most cradle
  burns will either not fire (cold arm) or be narrated (narrator writes), which G6 discounts.
- **Fix:** have the GL1 census report the `arms.thermal` distribution at `touch`/`warm_self` time over the
  committed cradle data (`docs/experiments/data/37_*`, `38_*`, `40_*`, and the Exp 41/42 rows). Then do one of:
  - state the GL2b(ii) gate over those starting states (e.g. "noxious from ≥ X"), or
  - give the variant body's scene contact semantics that are not additive (an authored `touch` that sets a
    floor, declared as a new affordance on the variant, not an edit to a shipped one).

  Also name the narrated share: the burns the arc produces through `SetEntitySensorTool`.

### SF-4 — Heat need: a shipped cooling act exists in the cradle scene, so GL2b(i) is not latent there

- **Stage:** GL2b(i).
- **Issue:** §8 recommends "say in the PR that no shipped scene yet offers a cooling act", and §1.3 R-2 calls
  the impact latent. But `items/cradle_cool_air.yaml` `draft.feel` (core −0.2, arms −0.15) and `shelter` are
  cooling acts, and `simulation/arcs.py` puts `items/cradle_cool_air` beside the fire pit in the `exploration`
  and `pain_consequence` arcs. Whether the heat need gets a consumer depends only on the affinity keywords:
  "cool" would match the `cool_air_*` tool signatures.
- **Why it matters:** with a matching keyword, a substrate-primary cradle run whose `arms.thermal` or core
  goes above the band (touch takes arms to 0.7) gains a new selection pull toward `cool_air`. That is a live
  behaviour change, not a latent one. (No Exp 41/42 harness scene includes `cool_air`: grep finds it only in
  `arcs.py` and two tests, so T1-6 itself is unaffected.)
- **Fix:** correct the recommendation text. State the keyword decision as the switch between latent and live,
  and add the arc scene to GL2b(i)'s walk.

### SF-5 — Minecraft `food` and the scripted "eat-after-hunger" gate are not reachable as written

- **Stage:** GL2c (flag-on gate; G7's food leg).
- **Issue:**
  - **Live:** `minecraft_player` `food` breaches at ≤ 6 and needs ≥ 16 to clear. In Paper 1.20.4, hunger
    drains only through exhaustion (4.0 per food point, after saturation is spent; walking is about 0.01/m),
    which is far beyond campaign length. The harness also refills saturation in every `rescue`. A native
    deprivation is unreachable in Exp 60/61/62/R3 time (the "hunger drains too slowly / food caps at 20"
    lesson the charter cites). Forcing it (`/effect hunger`) is an apparatus write (DNB-2).
  - **Offline:** `ScriptedWaterBridge._snapshot` serves `"food": 20.0` constant, and `_do_action("eat")`
    returns "did eat" with no state change. The ScriptedWaterControl `foodLevel` read is "20".
  - So GL2c's "scripted Minecraft … eat-after-hunger sequence" gate cannot run on the instrument named. Per
    #954's note, changing `ScriptedWaterBridge` is a change to a cited regression guard of the Exp 61/62 rows.
- **Fix:**
  - State in G7's rationale and §3.5 that the `food` leg is not reachable natively in the earned campaigns.
    The rows still fire by wording (G7 stands).
  - Run the entropic deprive→satisfy gate on the cradle: `infant_humanoid` hunger; `cradle_food.eat`
    hunger −0.4; satisfaction 0.3. Note that one feed clears only from hunger ≤ 0.7 + ε, so a later feed
    needs two.
  - Or build the Minecraft eat case on a **new** scripted fixture class, never by editing `ScriptedWaterBridge`.

### SF-6 — In `--sim`, a narrator write can land inside an AUT tool's before/after interval and be stamped `experienced`

- **Stage:** GL2a (strict default), and the §10 invariant "a narrator consequence is never `experienced`".
- **Issue:** `simulation/tools.py::SetEntitySensorTool.execute` writes through `tool_bridge._write_sensor` on
  the orchestrator thread and then calls `evaluate_failures()`. The GL2a `RLock` guards only the latch,
  snapshot and mint section **inside** `evaluate_failures`. It does not cover the executor's
  before-read → `tool.run` → after-read span on the AUT thread. So a narrator write (the arc's "raise
  arms.thermal toward 0.8 when the infant approaches") that lands in that span becomes part of an
  `experienced` tool-path record. The window is small for SEM tools (milliseconds) but not zero, and it is
  correlated with the AUT's own fire-pit actions, by the arc's instruction.
- **Why it matters:** §10 states the invariant as structural, but under the tool-path-only decision nothing
  enforces it, and the strict default's own rationale ("no provenance is ever guessed") does not hold.
- **Fix:** do one of the following:
  - hold the body `RLock` across the executor's bracket for SEM tools (but not for bridge tools, which can
    block for up to 8 s);
  - keep a per-body write epoch, bumped by any write outside the executor's bracket, and mark the record
    contaminated when it changes inside the bracket (this needs `_write_sensor`, outside the exempt set, so it
    lands with the narrated scope);
  - or scope §10's invariant to "out-of-band records" until the scope lands, and say so.

### SF-7 — Minecraft: cause identity is not available from the bridge

- **Stage:** GL2b(iii), GL5 (Minecraft arm), the umbrella's bio mapping.
- **Issue:** the bridge's damage signal is `event("damage", "took damage (health …)")` from `entityHurt`, with
  no attacker or source. The snapshot carries no cause either. Every Minecraft world consequence therefore has
  `cause=None`.
- **Why it matters:** cause-keyed valence cannot be produced in the game world. Adding a source field is a
  bridge protocol change, which `scripted_water.py`'s own note says "fires the re-run trigger on four EARNED
  ledger rows".
- **Fix:** state in §3.4 and in the GL2b(iii) gate that cause rows are cradle/`--sim`-only, and that a
  Minecraft cause needs a protocol change with its own ledger walk.

### SF-8 — GL5's jet triad has no world: none of its three affordances produces a body consequence anywhere shipped

- **Stage:** GL5 (prereg arm), GL4 S0b keys.
- **Issue:**
  - `flame_jet` (mage) and `water_jet` (fountain) exist only as test fixtures
    (`tests/integration/test_affordance_transfer.py`).
  - `fire_breath` (`creatures/dragon.yaml`) declares no `self_effect`/`target_effect`; the plan says so in
    §5.1(c).
  - If they are delivered through an actor, the path is `OrchestratorActorTool`, so every pair is `narrated`
    (G6), and the GL5 "without narrated data" report would be empty for that arm.
- **Why it matters:** "train `flame_jet`, probe `fire_breath` / `water_jet`" needs authored physics on all
  three. That puts the arm wholly inside Risk 9 (circular authored physics), which the blind author only
  partly mitigates.
- **Fix:** state in GL5's row that the triad needs new AUT-invoked entities (experienced, `self_effect` on
  the AUT), authored blind. Count them in S0b's keys.

### SF-9 — GL4 S0b's cradle key floor is reachable only with a purpose-built all-items scene

- **Stage:** GL4 S0b.
- **Issue:** all cradle-family items together declare 34 consequence-bearing (entity, affordance) keys
  (counted by YAML over `_data/components/items/*`: blanket, cool_air, false_hearth, fire_pit, food,
  sharp_rock, green/purple hearth/flame (+`_b`), warmth_alpha/beta safe/harm). The arc cradle scene
  (`simulation/arcs.py`: fire_pit, food, cool_air) has 5.
  - "≥ 30 keys" therefore needs nearly every item in one scene, which `infant_humanoid_chilled.yaml` warns
    about (the all-components tool-name collision; the `_b` twins exist for it).
  - The `cold` classes need a body with a `cold` sensor (chilled), because `warm_self`'s `cold: −0.3` changes
    nothing on `infant_humanoid`.
- **Fix:** name the S0b capture scene and body in the GL4 start decision. Count reachable keys and classes
  per body before the capture.

### NIT-1 — Stale dependency text

#1125 is **CLOSED** (2026-10-08, #1164, "the drive records read modulator drives through the one
resolver"), and `Executor._drive_pressure_snapshot` already reads `sem._read_sensor_value`. Update
autonomic §1.1, §3.1.4, §5.2, §8 and the umbrella's GL2a row. `_resolve_sensor_slot` lives in
`embodiment/sem.py`, not `tool_bridge.py` (§3.1.4).

### NIT-2 — Name the gate's tools and pin its table

The GL2a sequence "cool_air ×2" is `cool_air`'s `draft.feel`. The hand-computed table from `infant_humanoid`
initial values:

| Sensor | Start | after `feel` | after `feel` | after `warm_self` | after `warm_self` | after `touch` |
|---|---|---|---|---|---|---|
| `arms.thermal` | 0 | −0.15 | −0.30 | −0.10 | +0.10 | +0.70 (band breach; drive pain 0.08) |
| `core_temperature` | −0.15 | −0.35 | −0.55 | −0.35 | −0.15 | 0.00 |

The `core_temperature` latch clears on the second `warm_self` (deviation 0.15 ≤ 0.25·0.8). So the sequence
already contains a **homeostatic satiation crossing** on the tool path; assert it in the table. Run the test
under `_StepClock` or with a stated tolerance (wall drift is about 1e-6).

### NIT-3 — G7 text: state the clear points

Breach `oxygen` < 14; clear ≥ 15.2 (`_BREACH_HYSTERESIS` 0.2); `health` the same. Natural regeneration
needs food ≥ 18 (`naturalRegeneration` true in `water_trial.GAMERULES`). Exp 60/61 refuse training damage,
so the `health` leg occurs only in R3 (respawn) and through `heal()`.

### NIT-4 — Snapshot consistency under the sync pump

`MinecraftSyncPump._run` writes through `world_set_axis` on the `mc-sync-*` thread without the GL2a lock.
That is correct: it never calls `evaluate_failures`, as the umbrella states. The out-of-band snapshot must
store the values the latch was evaluated on, never a re-read. The record should also carry the bridge
`state_age_s` at before and after, so a consumer can bound pump-sample staleness (the pump interval defaults
to 0.5 s).

---

## What I verified (code at c80d619b)

- `body.py::Embodiment.evaluate_failures` applies wall-clock drift on entry (`time.time()`, `_last_poll`).
  Homeostatic clear uses `_BREACH_HYSTERESIS` = 0.2; entropic clear is `satisfaction_threshold`.
  `tick_vital_drift` formula confirmed.
- `loop_gates.py`'s LLM-primary tick calls `evaluate_failures()` once per iteration. In
  `substrate_proposal.py` the substrate tick does the same.
- `executor.py::_run_started` reads `pressure_before` before `tool.run`. `_stamp_invocation` stamps only the
  `rpe`/`pressure`/`relief`/`pain` trio, with no after-read today.
- `tool_bridge.py::ModulatorAffordanceTool.execute` applies the effect and then `evaluate_failures`. Its
  relief diff runs before drift; its pre-values are root-only and blind (#1161 OPEN).
- YAMLs read: `infant_humanoid` (drift rates, bands), `infant_humanoid_chilled` (`cold` 0.08/s),
  `cradle_fire_pit` (`warm_self` +0.2, `touch` +0.6), `cradle_cool_air` (`feel`, `shelter`), `cradle_food`
  (`eat` −0.4), `minecraft_player` (`oxygen`/`health`/`food` drives, world-owned, drift 0; `escape_water`,
  `eat`), `minecraft_bench` (`d1` MODELED, wall-clock drift, teacher feed).
- `simulation/arcs.py` cradle arcs list `cool_air` and instruct narrator `set_entity_sensor` writes to
  `arms.thermal` 0.8. `SetEntitySensorTool.execute` writes through `_write_sensor` and then
  `evaluate_failures`.
- `minecraft_harness.py::MinecraftSyncPump._run` calls only `sync_world_sensors`.
  `backends/minecraft.py::execute` syncs the post-action snapshot.
- `scripts/minecraft_bridge/index.js`: `escape_water` 600 ms hold, 8 s cap; `eat` polls until `bot.food`
  changes (1.5 s cap); the `damage` event has no source.
- `scripted_water.py`: `food` constant 20; `eat` is a no-op; escape returns before the anchor moves; the
  respawn resets vitals.
- `water_trial.py`: `train` is propose-only and ends in `rescue` (teleport, oxygen ≥ 19, `/effect` heal,
  four healthy ticks); `heal`; `GAMERULES`.
- `docs/experiments/data/exp60_trials.jsonl`: `probe_cap_s` 4.335 < `pain_edge_min_s` 5.085; training rows
  show 10 usable episodes with `oxygen_pain_signals` > 0 and `health_pain_signals` 0.
- `scripts/exp56/common.py` teacher feed and `simulation/cradle_mother.py::reactive_mother_tick`: direct
  `_apply_sensor_deltas`, then `credit_operant_reward`.
- `tests/unit/test_exp61_run.py` runs no loop. In `test_water_trial_smoke.py`, only the donor test is on
  `StepClock`; #951 is CLOSED and #954 is OPEN (it names the wall-time tests). `_loop_harness._StepClock`
  patches global `time`.
- #1125 CLOSED 2026-10-08 (#1164); #1161 OPEN.
- `fire_breath` declares no effect; `flame_jet`/`water_jet` are test fixtures only. 34 consequence keys
  across the cradle-family items.

**UNVERIFIED:**
- The live air-refill timing relative to `escape_water`'s return. It is derived from 1.20.4 air mechanics
  (+4 air/tick, 300 = 20 units); no live trace was read.
- Whether the orchestrator's own executor has an `embodiment` attached (that belongs to the wiring lens).
- Whether hivemind bundles carry Hippocampus traces, which would matter for the seq-resume rule's
  agent_id filter.
