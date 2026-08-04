# SEM Motor Binding — real body-turns through the Reachy component composition

**Status:** DESIGNED (three-lens parallel pre-implementation review, 2026-08-04). Awaiting owner green-light for Phase 1.
**Owner ask:** a real body-turn capability, "unique to the Reachy SEM component composition," after the live LLM hallucinated a `turn_body` tool when `focus_on_sound`'s honesty note recommended a body turn no toolset offered. Plus sim validation that the runtime *uses* the capability appropriately.

## Decision record (what the three lenses agreed on)

All three lenses (bio-architecture, executor/wiring, substrate-credit) independently converged:

1. **Bind the EXISTING affordance names** — `turn_left` / `turn_right` / `turn_left_big` / `turn_right_big` in `bodies/reachy_mini.yaml`. Do NOT add a `turn_body` learning identity. NAc `cluster_reward_bias` is exact-match on `(agent_id, cluster_id, tool_signature)` — a new name inherits nothing, orphans the Exp 45/46 trained policy that `maxim substrate merge-nac` imports, and re-runs far-bin starvation from scratch. The YAML itself warns renames invalidate substrate keyed to these names.
2. **The seam already exists and was built for this:** `SpecModulator._backend` / `attach_backends(entity, modulator_factory=...)` (`embodiment/spec.py`) — zero production callers today; the module docstring's own example is a `motor:` modulator. Front-gate satisfied: ride existing infra.
3. **The phantom-credit guard stays untouched.** Even with real actuation, the *modeled* `azimuth` self_effect is a fabrication — the measured shift arrives at the next DoA reading. `live_world_set_sensors` keeps filtering `azimuth`; measured credit is a separate, later slice.
4. **No schema change.** `AffordanceSchema` stays SHAPE-FROZEN (CC3); the binding is `(modulator name → backend)` at attach time. Hardware dispatch does NOT go in the body YAML (portable template); if an explicit declaration is ever wanted it belongs in `robots.yaml` (operator layer).
5. **The backend is honest by measurement:** dispatches `goto_target(MotionTarget(body_yaw=current±Δ, duration≈2s))` — the controller composes the head to ride along automatically (head-frame invariant machinery; `head=None` with a `body_yaw` target already re-adds the target body yaw to the current relative head yaw) — then reads the pose back and returns REAL success/failure. This also ends a live bug found during review: **virtual turns currently lie `success=True` and book +1 tool-success into NAc for motion that never happened.**

**The one divergence, resolved by phasing:** bio-architecture + wiring lenses said defer ALL measured relief credit to the Stage-5 act-now-credit-later pending map; the substrate-credit lens showed a synchronous post-motion re-read (wait for a DoA reading with `ts > t_end + ~0.3 s`, ≤ ~1.5–2 s budget inside the blocking execute) is a genuine measurement and safe on the serialized executor paths. Resolution: **Phase 1 ships motor binding with NO relief credit** (like `focus_on_sound` today); **Phase 2 ships measured credit** with its own two-lens round. The pending map remains mandatory before any NON-blocking dispatcher (Stage-5 DN reflex) ships.

## Phase 1 — motor bind (single PR)

Seam list (merged from all lenses):

1. **New `src/maxim/hardware/reachy/motor_backend.py`** — `ReachyOrientMotorBackend.execute(affordance, params)`:
   - maps `turn_*` → signed body-yaw delta read from the affordance's own declared `self_effect["head_yaw"]` (±0.3/±0.9 rad; YAML numbers ARE the motor command, so retuning the YAML retunes model+motor together);
   - **DN inhibit + motor-queue clear** around the goto (the `turn_around` defense — DN gaze behaviors on their own thread would fight the turn), then `goto_target`, then `sync_head_position()` immediately post-motion (the 3 s periodic sync is too slow) and a pose readback;
   - world-sets `head_yaw`/`body_yaw` from the readback and claims them in `live_world_set_sensors` (kills the modeled-`head_yaw` SEM sensor drift that exists today);
   - returns `ModulatorResult(success=real_ok, metadata={"achieved_body_yaw": ...})` — a rejected motion becomes `ToolOutput(success=False)`.
2. **`runtime/bootstrap.py::build_executor`** — optional keyword-only `modulator_factory=None`, threaded to `attach_backends` between entity instantiation and tool generation. `None` = byte-identical everywhere (per the push-silent-no-ops-into-types lesson: the wiring is visible at the canonical site, not a forgettable post-hoc walk).
3. **`embodied_runtime/agentic_runtime.py`** — build the factory from the registry robot controller when a body is wired; pass it; loud INFO naming which modulators got motor-bound.
4. **Approval gate:** union generated `sem-modulator-derived` tool names into `SupervisionPolicy.allowed_tools` at the live-policy construction site — kills the "requires approval" dead-end (and the live-confirmation deadlock path) for the agent's OWN body. Do NOT widen the frozen `ALWAYS_ALLOWED_TOOLS`.
5. **Prompt surface:** union embodiment-derived tools into `available_tools` past the mode filter when `executor.embodiment` is present (body ownership is not a mode privilege) — otherwise `reachy_mini_turn_*` surface as bare tokens (the 'move' lesson). Reword the four YAML affordance descriptions for the LLM audience ("rotate the whole body left ~17°; use when a sound is beyond the neck's reach"). `_focus_result_note` names the real tool for the reported side.
6. **DoA capture stamp gains `body_yaw` (4-tuple)** — the `(az, ts, head_yaw)` stamp is frame-incomplete once the body can turn between capture and execute; `focus_on_sound` already tolerates tuple growth. Backend refreshes/invalidates the stale reading after a turn.
7. **Floor suppression for motor-bound live turns** (substrate-credit lens, non-negotiable even in Phase 1): a motor-bound affordance touching a live-owned drive sensor must NOT fall through to the flat +1 tool-success cluster floor — silence-heavy substrate-primary sessions would otherwise mint direction-blind +1s (the probe-3 floor-drowning failure, one cluster over).

Guard tests: sim byte-identity (no factory ⇒ stub path, `ToolOutput` byte-equal); fake-SDK live test pinning the body turn's head-ride-along + DN inhibit + post-motion sync; backend failure ⇒ `ToolOutput(success=False)`; no-cluster-credit-on-live-turn (credit-mill regression, incl. floor suppression); approval-gate unioned; prompt renders descriptions not bare tokens; 4-tuple stamp back-compat.

Operational (with Phase 1): audit runtime `nac.json` for `tool:focus_on_sound`/`tool:move*` success links written 2026-08-03→04 pre-#459 (phantom-frame sessions recorded mechanical success for actuation that never happened) — reset with backup, the `learned_bounds.json` treatment.

## Phase 2 — measured relief credit — IMPLEMENTED 2026-08-04 (branch feat/sem-motor-credit, two-lens round pending)

Delivered exactly as specified below: the backend measures (`_read_azimuth` at entry with a 10 s staleness gate; post-motion poll for a reading stamped past `t_end + 0.3 s` settle, 2 s timeout → None, never modeled), the bio layer computes (`tool_bridge` reads `metadata["measured_drive_transitions"]`, applies `drive_comfort_progress`, adds the sensor to `accounted_sensors`, replaces the withheld marker), and the consumer routes (`drive_relief_channel: "exteroceptive"` → the operant/audio cluster; the tool-success floor NEVER routes extero). Registry key `drive_relief_channel` since 1.0.6. Guards: `TestMeasuredReliefCredit` (8 — measured ±, timeout→withheld, stale-before→withheld, audio routing, modeled-stays-intero, floor-never-audio, same-sensor gate).

### Original Phase 2 spec (for reference)

- `az_before` from DoA `latest` at execute entry; post-motion wait for reading with `ts > t_end + ~0.3 s`; `diff = drive_comfort_progress(spec, az_before, az_after)`; emit `side_effects["drive_potential_diff"]` as today.
- **Timeout → `None`, never modeled fallback** (sparsity acceptable; fabricated sign is not).
- **`accounted_sensors` must include azimuth in the measured path** — else the same-sensor collateral gate nulls every still-off-center turn and the feature is silently self-defeating.
- **Routing decision (owner call, recommended option b):** route measured azimuth relief to the **audio/operant cluster** (the trained policy's keys) rather than the default intero routing — measured exteroceptive relief is source-attributable, the case the probe-3 carve-out exists for. Without this, live experience coexists with the imported policy instead of compounding it.
- Discard rule (pre-registered now): a pending measurement is discarded on ANY interleaved motion command (`focus_on_sound`, another turn, reflex) — the three-writers-on-one-axis mis-credit case.

## Phase 3 — sim validation arms (owner-designed, pre-registration to be written before running)

Both arms run the SAME YAML affordances the live runtime motor-binds (sim = modeled self_effect; live = real actuation), so policy/signatures transfer. Honest physics required: azimuth = f(source bearing − head world yaw), head rides body, neck clamps at the measured ~±22°, source moves continuously (never teleports).

- **Arm 1 — two-joint centering (the gate):** fixed source at bearings beyond neck reach (±40–160°, counterbalanced), reward = azimuth drive relief. Claim: head-only agents plateau at the neck limit; agents with body affordances center everything. Metrics: centering rate, time-to-center, body-turn usage on far bins. LLM-primary arm answers "does the runtime use it appropriately"; substrate-primary arm validates the policy.
- **Arm 2 — Weeping Angel (sustained orienting under threat):** an entity advances only while unobserved (|bearing| outside gaze cone → approaches; inside → frozen); contact fires the pain cascade. Tests sustained facing + body-turn necessity when the threat circles past the neck limit; engages fear circuit, harm-avoidance valence, severity latch. Rides existing sim machinery (creature component YAML + orchestrator Layer-2 proximity writes + narrative acts — the cradle pattern).
- **Hardware calibration step before/alongside Arm 1 live replication:** measure the delivered azimuth shift of a motor-bound turn (body+head-ride ≈ 1.0 gain vs the 0.57 head-only calibration the YAML's ±0.17/0.50 deltas were derived under). If delivered shifts change, the ≈0.33 decision boundary moves and Exp 45b/c re-open — the YAML comment already warns about exactly this.

## Phase 1 implementation review round (two-lens, 2026-08-04) — folds + conscious narrowings

Both lenses confirmed the architecture, guard integrity, head-frame compliance, and B8. Folds landed in the same branch:
- Prompt/allowlist unions consume `always_active_sem_tools` (the reflexive turn_*+listen vocabulary), not all ~30 SEM tools; SEM tools are ALSO registered into the LearnedToolIndex post-`build_executor` (the passive-mode filtered renderer partitions the index's own universe — without this the prompt union was dead code on live's default mode).
- Motor factory gates on the DoA feed's own preconditions (reader present + no `audio_localization` opt-out): without a measurement stream owning `azimuth`, the modeled credit books −1 for every real turn (the mill's mirror).
- Backend refuses to guess an unreadable pre-pose (`body_yaw` missing → failure) and world-sets BOTH `head_yaw` and `body_yaw` from the readback.
- Four new bare exception swallows converted to logged; `build_executor` fails fast on `modulator_factory` without `entity_ref`.
- Conscious narrowings recorded: the DoA-reading invalidation after a turn was NOT implemented (the 4-tuple correction + next-reading refresh compensate; readings captured DURING the blocking turn carry a stale body stamp — their azimuth is mid-rotation garbage regardless, Phase 2's pending map owns this window). `execute_parallel_actions` does not read the withheld marker (covered today: llm-primary sets drive_relief_only; substrate-primary emits single actions — comment at the site per the #437 lesson). A `--simulation` Reachy runtime (SimulatedController) also gets the factory — intended: the sim controller honors the same MotionTarget contract. Sign convention (+body_yaw = LEFT) is offline-unverified — the Phase 3 hardware calibration step gates the live policy claim.

## Risks (ranked, from the lenses)

1. **Three writers on one axis** (`focus_on_sound` + motor-bound turns + future reflex): serialized today by the executor; the Phase-2 discard rule + Stage 5's voluntary-suppresses-reflex handle the rest. Mis-credit across consecutive different-strategy turns is the case the discard rule kills.
2. **Blocking turn + DN contention:** DN inhibit + queue clear is mandatory in the backend (turn_around's defense); the periodic sync reading a mid-motion pose is why the backend syncs explicitly post-motion.
3. **Credit-semantics temptation:** do NOT drop `azimuth` from `live_world_set_sensors` "because the shift is real now" — the modeled intra-execute diff is still a model; measurement only exists after the motion, which is Phase 2's job.
4. **Gain/boundary drift:** composite body+head turns deliver different azimuth shifts than the head-only calibration — the calibration step gates the live policy claim.

## Invariant checklist

- No new frozen dataclasses; `AffordanceSchema` untouched (CC3).
- `build_executor` gains an optional keyword-only param — canonical wiring site preserved.
- Phantom-credit guard (`live_world_set_sensors`) retained and extended (head_yaw/body_yaw claimed by the backend).
- B8 stays fed the FILTERED self_effect (direction-blind-B8-on-signed-drives stays moot on live).
- Head-frame invariant: body turns go through `ReachyMiniController.goto_target`'s compose (pinned by `test_reachy_head_frame.py`); the backend never calls raw SDK `goto_target(body_yaw=..., head=None)`.
- Tool results flow through the agent bus; the bio layer (`Embodiment`) gains no robot-controller reference — the backend attaches at the modulator layer.
