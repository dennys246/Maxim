# GL3 design review — ENVIRONMENT lens (thalamic_relay.md)

Plan reviewed: `docs/plans/thalamic_relay.md` (GL3: the Receptor registry and AfferentTracks), read in full
at worktree `.worktrees/e-t8b` = `origin/main` d9dffe6e. Charter: `docs/experiments/DESIGN_REVIEW.md`. Also
read: `docs/plans/grounding.md` (owner decisions G1–G20 taken as settled), the GL2b section of
`docs/plans/autonomic_layer.md` (§3.2, `NociceptorSpec`), the GL1 environment review (format), `docs/wiring/`
(`sensor-range-clamps.md`, `world-light-sensing.md`, `body-and-word-worlds.md`,
`pain-needs-declared-failure-modes.md`, `harness-loop-must-be-proven-live.md`, `cradle-sim-arms-and-resume.md`),
and the embodiment and simulation-experiments briefs. The lens question: do the worlds and bodies the plan
names (the Minecraft harness and bridge, the `_loop_harness` scripted world, the cradle/narrator sims, the Reachy
body, the word world) natively afford what each stage needs, without synthetic sensors or rewards (D1)? Can the
needed states and acts be reached and measured, and do the bridges and worlds behave the way the plan assumes?
Every code claim below was checked at `file::symbol`. One read-only probe was run: re-measuring the GL3.B0
latency characterization on the current tree (see Verified). No sim, rig, robot or LLM was touched.

**Verdict: 3 DO-NOT-BUILD / 9 SHOULD-FIX / 6 NIT.**

Reading order: D1 (GL3.B4 cannot see a world breach before §4 executes), D2 (narrated provenance is
overtaken by loop-thread transductions), D3 (GL3.B3's burn has no real-loop harness and no hook on the path it
takes). Then S1 (DRIVE pain, the drowning US, moves off the W-4 order when the flag is on), S2 (what a "pass" is),
S4 (what the latency record measures on CI vs on the rig), and the rest in order.

---

## DO-NOT-BUILD

### D1 — GL3.B4: no transduction runs between selection and §4, so `preempted_since` cannot see a game-native breach

- **Stage:** GL3.B4 (and red gate (d), as GL3.B4 would flip it).
- **Issue.** H2's check is `preempted_since(proposal.basis_seq, "pending_proposal")`: "has any event … arrived
  after `basis_seq`?" An event exists only once something transduces the breach and emits through the
  scheduler. On substrate-primary, nothing transduces between §6b (selection) and the next §4 (dispatch):
  - `runtime/loop_gates.py::tick_embodiment_drift` returns immediately on `aut_mode == "substrate-primary"`.
  - `runtime/substrate_proposal.py::propose_via_substrate` is the substrate path's only per-pass
    `evaluate_failures` caller, and it cannot run while a proposal is pending, because
    `runtime/loop_state.py::_substrate_tick_due` requires `ctrl.pending_proposal is None` (this is L1).
  - In Minecraft the breach arrives as a world write: the `mc-sync-<agent>` pump
    (`simulation/minecraft_harness.py::MinecraftSyncPump._run` → `MinecraftWorldBackend.sync_world_sensors`)
    writes `vital_metrics` and never calls `evaluate_failures`. In `_loop_harness`, `_ScriptedWorld` re-syncs
    on every sleep, which has the same effect. The pump becomes an edge that posts into the inbox only at
    **GL3.B8** (rig only).
  - H0 drains edges, and in GL3.B4 there are none. H1 is percepts only.
  - So in the frozen damage arm (health 20 → 8 in the sleep after pass 0), no event with `seq > basis_seq`
    exists at pass 1's H2. GL3.B4 as specified (`basis_seq` + H2 + the preemption right + the handler) cannot
    flip (d), and cannot lower the frozen +3-pass latency (gates 2 and 3).
  - The only ways to put an event there are the ones §4.2 forbids as special cases ("re-running
    `evaluate_failures` in a second spot before §4"), or the L1 seam fix (#1176, still OPEN). The plan leaves
    it to "whichever the L1 issue decides", which hides the dependency.
- **The rig makes it worse, not better.** The frozen damage arm's stale action (`mine_block` with no
  coordinates) is **refused** before the tool runs, which is why its breach waits for the next substrate tick
  (the characterization's docstring says so). On the rig, a stale action that really runs is a blocking
  `MinecraftClient.call_action` (15 s timeout, `_DEFAULT_ACTION_TIMEOUT_S`; `escape_water` holds up to 8 s in
  `scripts/minecraft_bridge/index.js`). `MinecraftWorldBackend.execute` re-syncs post-action, and
  `embodiment/tool_bridge.py::ModulatorAffordanceTool.execute` then calls `evaluate_failures`. So the breach is
  transduced **inside the stale action's own call, after the hold**. That is after H2, by construction.
- **Fold (plan edit, before GL3.B4's gate design pass), strict first:**
  1. State GL3.B4's transduction dependency explicitly. Options, strict first:
     - (a) GL3.B4 waits for the pump to become an edge whose H0 drain transduces (GL3.B8's pump slice moved
       ahead of GL3.B4, sim side only, without DoA);
     - (b) GL3.B4 waits for #1176's seam fix, which must transduce once per pass at one point before §4 (the
       same "transduction separated from selection" shape §4.2 gives for L2). Then name that seam as H2's
       event source.
  2. Add a negative control to GL3.B4's gate: on the damage arm, H2 with no pre-§4 transduction does **not**
     flip (d). This shows the source in (1) is load-bearing.
  3. Record in §5.6 that on the rig the dominant breach-to-action term is the stale action's hold, which
     GL3.B4 explicitly does not cover ("Held action … Not in GL3.B4"). A rung arm that declares GL3.B4 must not
     expect a rig-visible latency change from it alone (S4).

### D2 — GL3.B3: the narrator's `narrated` stamp rides the deferred transduction, but loop-thread evaluations transduce the write first

- **Stage:** GL3.B3 (H0, the orchestrator-thread declared edge; gate 6). It also touches GL3.B1's §5.7 item 1.
- **Issue.** The plan moves only the narrator tools' **transduction** behind the inbox ("orchestrator-thread
  transduction no longer calls `evaluate_failures` directly: it posts into the edge inbox"). The **write**
  still happens at once, on the orchestrator thread: `simulation/tools.py::SetEntitySensorTool.execute`,
  `DamageComponentTool.execute` and `OrchestratorActorTool.execute` write the AUT's sensors or integrity, then
  evaluate. Between that write and the next H0, the loop thread transduces it itself:
  - **LLM-primary, the cradle's default** (`orchestrator.py` `aut_mode: str = "llm-primary"`; Exp 37/38):
    `runtime/loop_gates.py::_loop_live_tick` calls `tick_embodiment_drift` → `evaluate_failures()` on every
    live pass. The plan places H0 **right after** `_loop_live_tick`. So the drift tick sees the narrator's
    write, latches the breach (`Entity.drive_breach_severity`) and publishes it, unstamped. When H0 then runs
    the narrated transduction, the latch is already set, so it publishes nothing.
  - **Both modes:** an AUT tool running in §4 (`ModulatorAffordanceTool.execute` → `evaluate_failures`) or a
    substrate tick (`propose_via_substrate`) that falls between the write and H0 does the same thing. GL2a's
    own record already names this residual ("a narrator WRITE that lands during `tool.run` still shows in the
    after-read", embodiment brief).
  - **The same structure covers the apparatus.** The cradle reactive mother
    (`simulation/cradle_mother.py::reactive_mother_tick`) writes hunger with `_apply_sensor_deltas` from `main`
    and never evaluates. The DM cascade writes on `sim.dm`. Each is transduced by whichever loop evaluation
    comes next.
  - **What this violates.** Every `AfferentEvent` needs a provenance, and the loop's own evaluation has no way
    to know the write was narrated, so the burn is minted `experienced` (or apparatus is minted
    `experienced`). That violates G6 ("a narrated record can never be relabelled experienced") and G16.
  - **Why gate 6 misses it.** Gate 6 asserts thread identity and "the record carries `narrated`" under one
    scripted narrator write. It passes whenever the test drains H0 before any other evaluation, which is
    exactly the case that does not happen live.
- **Fold (plan edit), strict first:**
  - (a) Route the **write**, not the transduction, through the inbox. The narrator tools (and the mother and
    cascade writers, with `apparatus` / `narrated`) post their sensor deltas with provenance. H0 applies them
    on the loop thread and transduces them in place, so no other evaluation can see the write first.
  - (b) Alternatively, stamp last-writer provenance per sensor **at write time** (`world_set_axis` already
    takes `owner=`), and have every `evaluate_failures` caller read it when it mints the event.
  - Either way, GL3.B3 gate 6 runs **on llm-primary with the drift tick live**, and adds a second case where
    the write lands during an AUT `tool.run`. Both must yield `narrated`.
  - §5.7 item 1's "mixed state" rule then has one mechanism behind it, not two.

### D3 — GL3.B3: the first slice's burn has no real-loop harness, and no hook on the path the burn takes

- **Stage:** GL3.B3 (gates 2, 3, 5).
- **Issue.** The first slice is "the thermal dual-track fan-out", using GL2b's `NociceptorSpec`. The
  environment that affords that burn does not meet the harness or the hooks the plan names:
  1. **Body.** The nociceptor is declared only on a **cradle variant body** (`autonomic_layer.md` §3.2: "never
     by editing `infant_humanoid.yaml`"). `minecraft_player.yaml` has no thermal sensor, and the bridge's
     `entityHurt` event carries no cause (`"took damage (health …)"`).
  2. **Harness.** The plan says "every scheduler gate in this plan runs through this harness" (`_loop_harness`).
     That harness is Minecraft only (`ARMS = shore / fear_water`; `_ScriptedWorld` hard-codes `health 20`). The
     cradle lockstep driver that does exist, `tests/unit/_cradle_loop_driver.py`, **replaces the substrate
     proposer with a script**, so `propose_via_substrate` (H3) never runs in it.
  3. **Path.** An experienced burn is the agent's own `cradle_fire_pit.touch`, transduced in §4 inside
     `ModulatorAffordanceTool.execute` → `evaluate_failures`. No GL3.B3 hook covers that call:
     - H3 is defined only "inside `propose_via_substrate`";
     - H2 belongs to GL3.B4;
     - H4 drains slow tracks only.

     On llm-primary the drift tick's evaluation (pre-tick) has no fast drain either. So the `nociceptive_fast`
     delivery (latency 0) has nowhere to go in the cradle.
  4. **Provenance.** On **LLM-primary** cradle runs every burn is narrated. The orchestrator passes
     `embodiment=None` to the scene tools unless `aut_mode == "substrate-primary"` or
     `MAXIM_DETERMINISTIC_SCENE_EMBODIMENT` is set (`simulation/orchestrator.py`, the scene-embodiment block;
     `generative_runner.py::_load_world_entities` docstring: "LLM-AUT passes `None` so scene tools stay
     inert"). Under that flag, the orchestrator's own comment warns that the narrator "could still add harm
     (double-count)". One contact would then be **two** physical events with two pids, and `caused_by` cannot
     link them, because the narrator's write holds no handle to the tool-path pid.

  So gate 2 ("one scripted burn yields exactly two track deliveries with one pid") can be met only by a
  hand-composed `evaluate_failures` call. That is the D43 shape.
- **Fold (plan edit), strict first:**
  - (a) Name the harness: `_cradle_loop_driver.py` on the GL2b variant body, substrate-primary, an experienced
    `touch` from a starting state inside GL2b's calibrated band (GL2b §3.2's "noxious from ≥ X").
  - Add a hook, "H3t: tool-path transduction drains latency-0 tracks in place inside the executing call", so
    T1-4's synchronous tool-coupled pain is preserved, and state what the llm-primary drift tick drains.
  - Gate 5's flag-on golden must include that arm, not only the Minecraft ones.
  - State that GL3.B3's thermal gate runs substrate-primary only. LLM-primary burns are `narrated` by
    construction, and `MAXIM_DETERMINISTIC_SCENE_EMBODIMENT` is excluded from any GL3 arm until the
    double-count is resolved.
  - (b) The non-strict alternative (owner): build GL3.B3's fan-out first on `drive:health`, which is already
    `PainKind.NOCICEPTIVE` (`proprioception/pain.py::TISSUE_DAMAGE_DRIVES`). It runs in `_loop_harness`'s damage
    arm and needs no GL2b. The thermal receptor would follow as a second gate.

---

## SHOULD-FIX

### S1 — GL3.B3: DRIVE pain (the drowning US) moves off the W-4 order when the flag is on

- **Stage:** GL3.B3, flag on.
- **Issue.** §5.4 routes `PainKind.DRIVE` breaches to `affective_slow` only (latency 2), and lists that track's
  destinations as autonomic, credit and prompt, with **no `pain_bus`**. §5.8 says PainBus is "the `pain_bus`
  destination of … `affective_slow` (DRIVE)". The two sections disagree.
  - Either reading breaks H3's promise that the W-4 order "stays byte-identical". `drive:oxygen` is DRIVE
    (`failure_pain_kind`; only `drive:health` is tissue damage) and is allowlisted for Wire-4
    (`decisions/nac.py::DEFAULT_CLUSTER_FEAR_FAILURE_MODES = {drive:health, drive:oxygen}`). It is the US of
    Exp 60/61/62, the only EARNED survival rows.
  - With the flag on, the Wire-4 drowning-fear write would land two passes later, or never if no `pain_bus`
    destination exists.
  - On the rig, two passes can span an `escape_water` hold, so the fear lands on the surfaced cluster. That is
    exactly the Exp 58 W-4 failure ("the fear lands on the wrong cluster").
  - The plan's flag-on gates pin only a new golden. None checks the Wire-4 target.
- **Fold:**
  - Decide in the plan that DRIVE breaches keep their synchronous PainBus publish at H3, so `affective_slow`
    adds deliveries and moves none. Otherwise, declare that flag-on fires T1-13/14/15.
  - Add a flag-on gate on the `fear_water` arm: the Wire-4 write lands on the same cluster id, in the same
    pass, as flag-off.
  - Make §5.4 and §5.8 agree.

### S2 — GL3.B3/B4: "pass" is undefined, and its wall equivalent differs by up to four orders of magnitude across worlds

- **Issue.** The loop's iteration counter (`for step_num in step_iter`, `agent_loop.py`) counts idle passes.
  The GL3.B0 characterization counts them too ("idle passes count"). But the idle gate `continue`s before §8.5
  (`agent_loop.py`, `if _gate is GateOutcome.IDLE: continue`), so H4 (the slow drain) and NAc decay run only on
  RUN passes, while H0 sits before the idle gate.
  - So `due_pass = pass_index + 2` is ambiguous. If it counts iterations, a slow delivery can come due on an
    idle pass and wait for the next wake (H5 is deliberately unbuilt).
  - Wall equivalents per world:

    | World | Wall time per pass |
    |---|---|
    | `_loop_harness` | 0.05 s idle, 0.25 s running |
    | Live Minecraft | a pass contains any blocking hold (up to 8 s for `escape_water`, 15 s action timeout) |
    | LLM-primary | up to 300 s per deliberation cycle (L3); the AUT turn window is 30 s (`sim.aut_turn_timeout_s`) |

  - In the turn-based cradle, `experience_us` advances only per narrator turn
    (`runtime/experience_time.py::ExperienceClockDriver.on_live_pass`, the `experience_turns` branch). The fast
    and slow deliveries of one burn therefore usually share one `experience_us`.
  - The `AfferentEvent` is one shared object (§5.2), so its `pass_index` is the ingest pass, and **nothing
    records the delivery pass**. Gate 2 ("the slow delivery lands `latency_passes` later") has no field to read.
- **Fold:**
  - Define `pass_index` as one named counter (strict: RUN passes, the NAc-decay clock), and say whether H4 runs
    on idle passes.
  - Add a delivery log of `(pid, track, delivery_pass, experience_us_at_delivery)` that gate 2 reads.
  - Add a per-world table of pass wall-equivalents to §5.4's "ordinal, not metric" note, and state that GL4
    orders deliveries by `(seq, delivery_pass)`, never by `experience_us`, on turn worlds.

### S3 — GL3.B4: game-native damage does not look like the frozen damage arm, and the livelock probe is vacuous against it

- **Issue.** On `minecraft_player`, `health` has set point 20, `comfort_band` 6.0 and `pain_scale` 0.5
  (`minecraft_player.yaml`), so pain = `min(1, (dev − 6)·0.5)`:
  - at band entry (health just under 14) the intensity is near 0;
  - it is 0.5 at 13 hp;
  - it saturates at 1.0 only at 12 hp or below.

  With `preempt_threshold 1.0` ("at/above"), preemption fires only on saturated pain. The latch re-publishes
  on deepening by more than `max(1e-3, 0.05·6) = 0.3` hp (`embodiment/body.py::_BREACH_DEEPEN_FRACTION`), so
  every whole-hp game damage step re-publishes, and on the plan's design each publish is a new physical event
  with a new pid.
  - The frozen arm's breach is one 12-hp drop in one 0.25-s sleep. In game terms that is a creeper or a long
    fall. Typical hazards (lava, fire, drowning damage at air 0, mob hits) step down 1–4 hp at a time, so the
    first two or three publishes sit below threshold. (The per-hazard rates are game knowledge, UNVERIFIED in
    this repo.)
  - An RCON `/damage` reproduction would be `apparatus` (G16), never credited.
  - The livelock probe (gate 6) uses "a standing breach over 20 passes". A standing breach publishes **nothing**
    after entry, because the latch holds, so the probe passes trivially. The live hazard is a **deepening
    stream**, where "at most once per pid" bounds nothing, because every step has its own pid. Once a
    transduction source exists before §4 (D1), that stream can preempt a freshly reselected protective
    proposal before it executes.
- **Fold:**
  - Gate 6 drives a deepening stream (1 hp per pass from 14 to 4 hp), and adds "the protective action
    executes within k passes of the first above-threshold event".
  - The declaring rung's prereg names a game-native hazard and states its damage time course.
  - §5.6 says whether a deepening re-publish of one locus may preempt again inside the refractory.

### S4 — GL3.B0 record / GL3.B4: the latency record is a step-clock harness fact, and the rig adds sensor staleness the record has none of

- **Issue.** `tests/fixtures/afferent_latency_characterization_v1.json` is measured on `_StepClock` with
  `_ScriptedWorld`, which re-syncs the body on every loop sleep ("the pump's job live"). So the body sees a
  breach with **zero staleness**, and the record is identical on a CI runner and a laptop. It measures nothing
  about the rig. On the rig:
  - the pump runs at `interval_s` (0.5 s default; 0.25 s in `exp58_run.py` / `b1_floor_run.py`);
  - the bridge's state interval is 500 ms by default (100 ms only where a harness passes
    `--state_interval_ms=100`);
  - the stale action holds the loop thread (D1).

  GL3.B4's whole possible gain on the damage arm is ≤ 3 passes (0.55 step-clock s). That is the same size as
  one default bridge interval. Separately, the record's equality check runs only when `HEAD:src` equals its
  `src_tree`, so on CI it has been shape-only for 137 `src/` commits. I re-measured it at d9dffe6e and both
  arms are IDENTICAL to the frozen numbers, but nothing in the plan makes GL3.B4 re-check that before it
  compares.
- **Fold:**
  - GL3.B4 gate 1 adds: the flag-off measurement on the build tree equals the frozen record. If not, stop and
    record the drift; never compare a flag-on number against a drifted baseline.
  - §6 GL3.B0 says the record is a harness fact. A rig arm measures breach → protective action in wall time
    and bridge ticks, carries `MinecraftClient.state_age_s()` into `AfferentEvent.extra` at ingest, and inherits
    the Exp 60 preflights (bridge cadence ≤ 0.15 s, `_measure_bridge_cadence`; pump ≤ 0.25 s).

### S5 — GL3.B3: deferring narrator transduction changes what the narrator LLM is told, and lets the AUT act into the burn

- **Issue.** The narrator tools return the result of their synchronous `evaluate_failures` to the orchestrator
  LLM:
  - `DamageComponentTool` returns `pain_triggered` and `failure_modes`;
  - `OrchestratorActorTool` returns `side_effects["embodiment_failures"]`, "annotating … AUT-side failures for
    the orchestrator to narrate" (`simulation/tools.py`).

  With the flag on, those come back empty, so the narrator narrates no pain. Meanwhile the AUT executes its
  §4 proposal in the same pass, before the next H0, against an untransduced burn. Drift also runs between the
  write and the deferred evaluation: the chilled body's `cold` drifts at 0.08/s, so a 30-s turn can erase a
  narrated warming before it is transduced.
- **Fold:**
  - In GL3.B3, state the narrator tool-output contract under flag-on: channel-1 failures can be computed
    state-based, with no latch and no drift, because they are not latched today either.
  - List "narrator narration and the AUT's next action see the burn later" in the flag-on "new condition".

### S6 — GL3.B1: in Minecraft the apparatus reaches the body through the same writer as experience, so the last-writer rule cannot see it

- **Issue.**
  - §5.7 item 1's mixed-state rule names only the narrator as a last writer.
  - In Minecraft, every world write has one owner, `minecraft_bridge` (the pump's `world_set_axis`). The rescue
    teleport, the `/effect` heal and the respawn (G16's apparatus, `WaterTrial.rescue` / `heal`, R3) act in the
    world and come back as bridge truth.
  - The `world.state` and `body.drives` pull receptors would therefore stamp an apparatus-caused recovery as
    `experienced`.
  - The autonomic plan already designs an apparatus **window** (`autonomic_layer.md`, "the apparatus scope is a
    window, not a thread-local"), but GL3.B1 never consumes it.
- **Fold:**
  - GL3.B1's mixed-state rule consumes the apparatus window. Strict: every pull read inside the window, and
    until the first snapshot taken after it closes, is stamped `apparatus`.
  - Add a red gate that drives `WaterTrial.rescue` on `ScriptedWaterBridge` and expects `apparatus` on the
    surfacing read's EC node.

### S7 — §9 / G4: the re-keyed fabric trigger fires before the join key it needs exists (owner)

- **Issue.**
  - The fabric revives when "GL3's registry+provenance stage ships AND a 1.4 rung needs cross-modal binding",
    because "GL3.B1 carries the event identity … the binding's join key needs".
  - The only live receptor with two lanes is Reachy DoA, and its lanes get one pid only at **GL3.B8**
    (rig only; red gate (c)). The text↔sensor join is GL3.B5's `caused_by`.
  - GL3.B1 adds pids per pull read and joins nothing across modalities, so a rung could revive the fabric with
    no join key in place.
  - The supporting "five live sources with different … encoders, spaces and clocks" overstates what exists:
    - there are two encoders (`sensor_sum`, linguistic);
    - "vision" is a sentence embedding of percept text (§2 gap 8);
    - DN vision is live only on a robot with detections;
    - Minecraft text reaches EC only with `MAXIM_SUBSTRATE_PATH=1`.
- **Fold (owner decision; G4's split stands, only the trigger wording is in question), strict first:**
  - (a) Amend to "… AND the join the binding needs exists (GL3.B8 for DoA's two lanes, GL3.B5 for text↔sensor)".
  - (b) Keep the wording, and add to the fabric banner that a revived fabric starts without its join key.
  - Either way, correct the "five sources" sentence to the counts above.

### S8 — GL3.B5: the `caused_by` join has little to join to in the game as it is

- **Issue.**
  - The bridge's damage event is `event("damage", "took damage (health ${bot.health})")`
    (`scripts/minecraft_bridge/index.js`, `entityHurt`). It has no cause and no snapshot sequence, and it
    arrives on the event queue, separately from the state snapshot and on its own cadence.
  - A "health-drop sensor event" exists as a physical event only when the latch publishes (below 14 hp, or on
    deepening). Ordinary in-band hits (20 → 17) produce no sensor-side event at all.
  - So the GL3.B5 join rate on a recorded trace measures the latch, not the mechanism.
- **Fold:**
  - Define the cause target: a `world.state` read event in which `health` decreased, which needs a per-read
    pid and delta detection. Or restrict the join to band-crossing damage.
  - Add the bridge's snapshot counter to each event line (a game-native field, D1-clean).
  - Report the join rate stratified by in-band vs breach damage.

### S9 — GL3.B8: held-action preemption and the DoA push receptor need the hardware-safety facts on the page

- **Issue.**
  - **Minecraft hold.** No backend declares `interruptible`. The Minecraft bridge has a `stop` action that
    clears pathfinder goals only (`bot.pathfinder.setGoal(null)`); `dig`, `consume` and the `escape_water`
    surface loop are not cancelled by it. Bridge actions run as concurrent promises, so a mid-hold `stop` could
    be delivered, but `MinecraftClient.call_action` blocks the loop thread until `action_result`, so the loop
    cannot send one.
  - **Reachy.** The controllers (`hardware/reachy/controller.py`, `hardware/simulation/controller.py`) have
    `goto_target` and no cancel. Any held-motion preemption on Reachy must go through
    `ReachyMiniController.goto_target`, the single clamped and locked dispatch point (embodiment brief §1), and
    must define what the retained-axes stash holds after an interrupted goto: a commanded pose the head never
    reached.
  - **DoA push receptor.**
    - The `reachy.doa` reading is head-relative at capture. Its stamp trails the samples by up to the ~2 s gate
      cycle, and `DoAFeed` carries `window_start` for exactly that reason
      (`embodiment/audio_localization.py::DoAFeed.latest`).
    - An `AfferentEvent` stamped only with `experience_us` at H0 ingest loses the capture frame. The plan's "no
      wall time on the event" rule must not strip the capture tuple.
    - The sensor lane writes nothing unless `robots.yaml` declares a body with an `azimuth` sensor
      (`agentic_runtime.py`, "DoA feed started (… + azimuth sensor)"; `world_set_azimuth` returns False
      otherwise).
- **Fold:**
  - Replace §5.6's held-action UNVERIFIED with the facts above.
  - GL3.B8 adds a hardware-safety item: no held-motion preemption on Reachy without a controller-level abort
    behind the clamp and the lock, plus a deletion-probed test in `test_reachy_workspace_safety.py`.
  - The `reachy.doa` payload carries `(window_start, capture_head_yaw, capture_body_yaw)` as capture-frame
    data, never as a scheduling input.
  - Red gate (c) runs on a declared body.

---

## NIT

- **N1 (§8).** It cites `tests/unit/test_exp61_run.py` as a check that "executes the producer". It runs no loop:
  zero hits for `run_agentic_loop` / `propose_via_substrate` / `WaterTrial` / `ScriptedWaterBridge`, and
  grounding.md's GL2a row already says it "runs no loop and is not cited". Cite
  `tests/unit/test_water_trial_smoke.py::test_water_trial_ticks_acts_and_the_staging_close_persists_fear`
  instead. Note that `scripted_water.py` paces a socket bridge with the loop on wall time (#951), so it is
  timing-sensitive on a loaded CI runner.
- **N2 (§3.2).** "`mc-sync-<agent>` (0.5 s)" is only the default. The harnesses run 0.25 s
  (`exp58_run.py`, `b1_floor_run.py`) and 0.05 s (`dark_danger_probe.py`, `exp58_offline_gates.py`). The bridge
  state interval is 500 ms by default and 100 ms in the Exp 60–62 / R3 harnesses. State both, because S4
  depends on them.
- **N3 (§5.2).** `ReceptorSpec.clock: Literal["realtime","turn"]` is authored, while the run's clock kind is
  already decided by `ExperienceClockDriver` from `percept_source.experience_turns` (its `kind`). Derive it, as
  TR5 derives the body specs, or two sources of truth can disagree.
- **N4 (GL3.B1 gate 1).** The selection golden over `reachy_mini` should run with
  `MAXIM_PLACE_CODE_EXTEROCEPTION` both off and on, because the audio channel's encoding differs between them
  (§2's DoA row).
- **N5 (GL3.B2).** "On a recorded cradle and survival trace": no pid-bearing trace exists yet. Name the
  producers: the `_loop_harness` arms and `_cradle_loop_driver.py`, both offline, with no rig or LLM spend.
  Note that cradle-mother traces (`benchmark_cradle_mother.py`) are `apparatus` consequences under G16.
- **N6 (GL3.B3 blast radius).** Track publishes "bypass" PainBus's 0.5 s wall refractory in favour of one pass
  (0.25 s in the harness, about 0.16–0.25 s on the rig per `harness-loop-must-be-proven-live.md`'s ~50
  iterations / 8 s). On a deepening stream that is up to twice the publishes per second into the learning
  subscribers. Add it to the flag-on "new condition".

---

## Verified

- `proprioception/pain.py::TISSUE_DAMAGE_DRIVES = {"drive:health"}`; `classify_pain` and `failure_pain_kind`
  make every other `drive:` source DRIVE. So `drive:oxygen` is DRIVE, `drive:health` is NOCICEPTIVE, and
  `drive:arms.thermal:noxious` is DRIVE until GL2b adds its rule (autonomic §3.2 says so).
- `decisions/nac.py::DEFAULT_CLUSTER_FEAR_FAILURE_MODES = {drive:health, drive:oxygen}`.
- `minecraft_player.yaml`: `health` and `oxygen` homeostatic (set point 20, `comfort_band` 6.0, `pain_scale`
  0.5), both declared `modality: world` and `drive:` (the double entry); `health` range [0, 40]; no thermal or
  fire sensor.
- `embodiment/body.py::evaluate_failures`: the latch, `_BREACH_DEEPEN_FRACTION = 0.05`, `_BREACH_MIN_EPS`,
  `_BREACH_HYSTERESIS = 0.2`; channel 1 state-based, channel 2 latched.
- `runtime/loop_gates.py::tick_embodiment_drift` is a no-op on substrate-primary; `_loop_live_tick` runs it
  before the experience driver, and `pre_tick_gate` calls `_loop_live_tick` before the idle gate.
  `runtime/loop_state.py::_substrate_tick_due` requires `pending_proposal is None`. The `evaluate_failures`
  callers in `src/` are `substrate_proposal.py`, `loop_gates.py`, `tool_bridge.py`, `simulation/tools.py`
  (×4), `foundry.py` and `percepts.py`.
- `agent_loop.py`: `for step_num in step_iter` counts idle passes; `GateOutcome.IDLE` → `continue` before
  `_loop_bio_tick_maintenance`.
- `simulation/minecraft_harness.py::MinecraftSyncPump` (default 0.5 s, 5 s staleness gate);
  `embodiment/backends/minecraft.py::MinecraftWorldBackend.execute` re-syncs after `call_action`;
  `simulation/minecraft.py` (`_DEFAULT_ACTION_TIMEOUT_S = 15.0`, `state_age_s`, a `next_percept` with no
  `agent_id` argument); `scripts/minecraft_bridge/index.js` (`STATE_INTERVAL_MS` 500, `entityHurt` text with no
  cause, `stop` = `setGoal(null)`, `escape_water` 8 s cap, actions as concurrent promises).
- `simulation/tools.py`: `DamageComponentTool` publishes a `PainSignal`, then `evaluate_failures`, and returns
  `pain_triggered` / `failure_modes`; `OrchestratorActorTool` evaluates the AUT body and returns
  `embodiment_failures` "for the orchestrator to narrate".
- `simulation/orchestrator.py` scene-embodiment block: `embodiment=None` on LLM-primary unless
  `MAXIM_DETERMINISTIC_SCENE_EMBODIMENT`; the double-count is a known follow-up. Default `aut_mode` is
  `"llm-primary"`.
- `tests/unit/_loop_harness.py`: Minecraft only, substrate-primary, `_ScriptedWorld` health fixed at 20.
  `tests/unit/_cradle_loop_driver.py`: the substrate proposer replaced by a script.
- `runtime/experience_time.py::ExperienceClockDriver`: turn-based branch from `experience_turns`, 300 s
  per-pass cap. `simulation/conversational_source.py` declares `experience_turns`.
- `embodiment/audio_localization.py::DoAFeed`: one gated reading per cycle, `window_start` vs `ts`, a
  head/body yaw stamp, the sensor lane via `world_set_azimuth`. `embodied_runtime/agentic_runtime.py` wires
  `percept_sink=adapter.carry_percept` and needs an azimuth sensor. No `cancel` or `interruptible` exists on
  any `hardware/*/controller.py`.
- **Probe:** re-ran `test_afferent_latency_characterization.measure_all` at d9dffe6e (`HEAD:src` a63e5712, not
  the frozen `src_tree` 74a0fa82; 137 `src/` commits since 7d18347e): **damage IDENTICAL, drowning IDENTICAL**
  to the frozen fixture.
- Defect issues #1176 (L1), #1177 (L2), #1178 (L5): OPEN (`gh issue view`).
- Every file the plan cites as a gate or tool exists: `selection_dynamics_rebaseline.py`,
  `test_encoder_golden_v1.py`, `test_place_code_wiring.py`, `scripted_water.py`, `test_receptor_census.py`, the
  red-gate files, `test_water_trial_smoke.py` and `scripts/orient_backbone/yaw_verify.py`.
  `tests/unit/test_exp61_run.py` exists but executes no producer (N1).
- **UNVERIFIED:** the per-hazard Minecraft damage rates in S3 (game knowledge, not in this repo); whether the
  mineflayer version in use exposes a damage cause on `entityHurt` (S8).
