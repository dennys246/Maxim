# #1499 motion-path model (approach note v8): no too-fast, wrongly-referenced or mid-path-unsafe command reaches a motor

Session S, 1.3.4 exit item (HIGH, hardware safety).

**Status: v8, final for the design phase.** It went through the v6 three-lens design review, the v7 delta round, and the v8 scoped rounds, all on 2026-10-29.
- All three scoped rounds returned BUILD AFTER FOLDS, with **no further design round owed**. Their folds are local and fail-closed: the abort token, `mark_accepted`, the `hardware` layer rule, `turn_around`'s ticket and the atomic bridge retire.
- The build starts with the red gates, and the code review gets a different reader on the abort fence.

The v7 delta round returned BUILD AFTER FOLDS on all three lenses. Its findings are folded in place (§11, second table). The owner decided the ABC question (O7).

The v6 review also returned BUILD AFTER FOLDS on all three lenses:
- **Executor:** one DO-NOT-BUILD (D1: the σ rule misses roll and pitch), plus S1 to S7.
- **Architecture:** one DO-NOT-BUILD (D1: refusals poison the learners in PR 1), plus S1 to S9.
- **Wire integrity:** findings 1 to 12.

Every finding is folded below. §11 maps each one to its home. The owner re-decided O3, O4 and O6 in light of the review (§1.2). v8 gets the scoped rounds the v7 delta asked for, each on the exact delta from v7:
- Executor, on the per-chunk re-check;
- Architecture, on the ticket and ABC seams;
- Wire, on D1 to D5. §12 lists the choices made in the folds themselves, for that round to check.

## 0. What #1496 already shipped (merged 2026-10-29, PR #1512; do not redesign)

- **Validators** in `hardware/controller.py`: `require_finite`, `require_duration` (range [0.1, 10] s), `floor_duration`, `clamp_finite` and `normalize_method`. `MotionTarget.__post_init__` and `PixelTarget.__post_init__` run them.
- **The Reachy last mile:** `hardware/reachy/controller.py::check_sdk_command`, run before every SDK motion call.
- **`maxim.exceptions.MotionRefused(kind)`**, kinds `non_finite | duration | method | range | invalid`.
- **Antennas** clamp to ±`ANTENNA_LIMIT_RAD` (90°, provisional) and always interpolate. `cartoon` is refused.
- **`scripts/lint_motion_safety.py`** bans the clamp idiom and raw SDK motion on any `…mini` handle chain.
- **`SyncResult`**, plus the motor worker's handling of `NON_FINITE` and of refusals.

## 1. Owner decisions

### 1.1 Carried (2026-10-26)

1. **Rate cap: 180°/s average** (amended by O2).
2. **A default or derived duration stretches; an explicit one that is too fast is refused.** The validator only refuses.
3. **Stash divergence is refused**, and the stash for that axis is forgotten.
4. **At the body limit, no turn:** the head does the rest. An explicit `turn_around` that cannot fit is refused with `kind="limit"`.
5. **`scripts/` is in the raw-SDK lint's scope**, with the live harnesses allow-listed by name.

### 1.2 New and re-decided (2026-10-29)

- **O1. Path rule.** A dispatch whose sampled path exceeds the neck-path limit (R5) is handled by its duration policy:
  - Default or derived duration: it is split into chunks, each run after the last completes.
  - Explicit duration: it is refused with `kind="path"`.
- **O2. Rate.** The average speed is at most 180°/s, and each method's peak, by its peak factor, is at most 340°/s.
- **O3 (re-decided).** The fresh readback is the daemon's public `GET /api/state/full`, a synchronous call answered with the daemon's own timestamp. The SDK's private events are not used.
- **O4 (re-decided).**
  - The motor enqueue returns a ticket. Tools learn the real outcome and the reason, and DN records on completion.
  - **No refusal kind feeds the gaze or bounds learners.** They see only completed moves and the existing IK-failure detection.
- **O5.** Extend the lint to `wake_up`, `goto_sleep`, `enable_wobbling`, `set_automatic_body_yaw`, `cancel_move` and `ReachyMini(`. Lifecycle motion goes through the controller.
  - The no-motion facade for `Selfy.mini` is a follow-up issue. Its reason is pymaxim-internal: media, audio and DoA reach the SDK through that handle. Pulse does NOT read `maxim.mini`.
- **O6 (re-decided).**
  - A configured `automatic_body_yaw: true` is a typed config error at connect, and nothing is constructed.
  - The controller asserts `set_automatic_body_yaw(False)` at connect, after `wake_up`, and on adopting a handle it did not build (pulse's app framework builds with the SDK default, `True`).
  - Every `ReachyMini(` passes the argument explicitly, enforced by the lint.
  - The single-client caveat is stated in §8.
- **O7 (2026-10-29). The ABC stays 1.x-compatible.**
  - `RobotController.goto_target` is annotated `-> bool | MotionOutcome`.
  - One normalizer, `as_outcome(result) -> MotionOutcome` in `hardware/controller.py`, turns a `bool` into an outcome with no reason (`kind=None`). Every consumer that reads a field goes through it.
  - Plugins, and the test fakes that return `bool`, keep working (docs/user/extension_api.md: "will not break in 1.x").
  - Recorded as an MP decision.

## 2. SDK and daemon facts (reachy-mini 1.8.3; verified 2026-10-26 and 2026-10-29)

- **F1.** `look_at_image` → `look_at_world` → `goto_target(head, …)`, and the `body_yaw` default is `0.0`, so the body is sent to 0. This is live in `ReachyMiniController.look_at_pixel` and in Selfy's SDK look-at path.
- **F2.** With `body_yaw=None`, `GotoMove` holds the body at the daemon's present body.
- **F3.** `play_move` drops a move while another is running, and the client's task still reports completed.
- **F4.** The client's `goto_target` blocks for `duration + 1` s (`wait_for_task_completion`). If it times out (`TimeoutError`), the daemon may still be moving.
- **F5.** Body yaw and antennas interpolate in joint space.
- **F6.** The head interpolates on the world SO(3) geodesic: `linear_pose_interpolation`, via `as_rotvec`, the short way round. The method only re-times `s`; the path in s ∈ [0, 1] is the same for every allowed method.
  - **Verified with the SDK's own function (2026-10-29):**
    - center_vision from body 160°, relative 30°, to 0/0: the neck yaw passes 179.8°.
    - body −55.7° → 73.9°, relative 10.2° → 30° (σ = 149.6°), with roll −25.6° → 35.9° and pitch −34.5° → −29.3°: the neck yaw passes **179.7°**.
    - The same move with roll and pitch held: 30.0°.
  - So a σ threshold alone cannot be the rule.
- **F7.** `create_head_pose` is extrinsic "xyz" (Rz·Ry·Rx), so world = Rz(body)·R_rel.
- **F8. Peak factors:** `minimum_jerk` 1.875×, `ease_in_out` 2.0×, `linear` 1.0×.
- **F9.** The SDK sleep pose puts the antennas at ±3.05 rad and the head at pitch about 24°, z −44 mm, x −21 mm (`SLEEP_HEAD_POSE`). Neither the SDK nor the daemon declares antenna limits.
- **F10.** `ReachyMini(...)` defaults `automatic_body_yaw=True` and pushes it to the daemon in `__init__`.
  - The setting is daemon-global.
  - With it on, `AnalyticalKinematics.ik` uses `inverse_kinematics_safe`, which moves the body itself.
  - There is no client getter.
- **F11.** `wake_up` and `goto_sleep` call `goto_target(INIT/SLEEP_HEAD_POSE, antennas, duration=2)` with the body default of 0.
- **F12. Freshness.**
  - The WS client's pose and joint messages carry no timestamp, and a non-finite pose is dropped silently (the cache keeps the last good one).
  - `GET /api/state/full` returns `head_pose`, `body_yaw`, `antennas`, `control_mode` and `timestamp` (UTC).
    - The pose is the backend's last control-tick FK (`current_head_pose`), so it is at most one tick old. It is not computed when the request arrives.
    - With `use_pose_matrix=true` it returns the 4×4 matrix; the default is an `XYZRPYPose` built by an Euler round trip.
    - It needs no auth, and it is served on the same host and port as the WS. The REST DoA reader already uses it in tunnel mode.
    - `control_mode` is one of `enabled`, `disabled` or `gravity_compensation`. The mujoco backend always reports `enabled`.
    - **The shape varies by version**: `delivered_shift_block.py::daemon_body_yaw` searches nested dicts for `body_yaw`.
  - The WS `GetStateCmd` also carries `is_move_running`, but `send_command` has no response routing in the client.
  - `GET /api/move/running` lists only REST-started moves, so it is not usable to see a WS goto.
- **F13. Maxim's motor queue drops commands silently in four places:**
  - `_enqueue_motor` drops the oldest pending command (get-then-put);
  - its `queue.Full` branch drops the **new** one;
  - `connection.py`'s soft reconnect drains the queue with raw `get_nowait()`;
  - `MediaLoopMixin.live` teardown sets `_motor_queue = None` without draining.
- **F14. Raw lifecycle motion in `src/`:**
  - `agentic_runtime.py` enqueues the SDK's `mini.wake_up`;
  - `connection.py`'s soft reconnect calls `self._robot.wake_up()`;
  - the controller's `wake_up`/`goto_sleep` call the SDK's methods without re-centring.

## 3. The reference model

**R1. The reference for every check is a fresh daemon readback (O3); the stash fills unspecified head axes only.**
- **The read** (Wire D5; Architecture N2; Executor N2, N3).
  - `_fresh_readback()` calls `GET /api/state/full?use_pose_matrix=true&with_head_pose=true&with_body_yaw=true&with_antenna_positions=true&with_control_mode=true`. It goes to the host and port the controller connected to, through `maxim_http.fetch_url`: the existing pattern of `ReachyMiniController._daemon_status` and the REST DoA reader, so this adds no new endpoint registration.
    - It passes `address_policy="private"` and an explicit short `TimeoutPolicy`, whose pool, connect and read phases each have a bound and sum to the deadline.
    - The default goes through the shared `_external` endpoint (a pool of 10, `TimeoutPolicy.long()`), which the DoA polling and the web tools also use. Pool waits there would show up as spurious `reference` refusals (Wire scoped NIT).
  - **The parser is tolerant of the version** (the shape varies, F12). It accepts the 4×4 matrix at `head_pose` (or `head_pose.m`), and `body_yaw` at the top level or one level down. Anything else is a missing field.
  - **A contract test at the SDK floor and latest**, in the R2 reachy cell, runs against the real daemon app, so the parser is pinned to a real response.
  - **An explicit overall deadline.** `TimeoutPolicy` is per phase, so the call is bounded by a total deadline: 100 ms on the LAN, 300 ms in tunnel mode, both measured on the rig before release. There are no retries inside it.
  - A transport error, a non-200 status, a missing field, a non-finite value, or a `control_mode` other than `enabled` raises `MotionRefused(kind="reference")`.
  - At connect, one probe read validates the shape. On a mismatch the controller logs one ERROR naming the SDK version, and `maxim doctor` reports it.
- **What the readback supplies:** the body used to compose a head-only move (F2), the start of every rate delta (R4), and the start of every path check (R5).
- **What the stash keeps:** `_last_commanded` fills only the head axes the caller did not specify (roll, pitch, relative yaw). This is the H1 F1 ratchet rule, unchanged. The body stash is no longer used to compose a move; it is used only for the divergence check.
- **Divergence (decision 3).**
  - Compare on the retained axes only, plus the body when a body stash exists. Relative yaw is compared through R_rel.
  - Above `_STASH_DIVERGENCE_REFUSE_RAD` (5°, provisional; measured on the rig as end-of-move lag plus a margin), raise `reference` and forget that axis.
  - The observational `_DIVERGENCE_WARN_RAD` warning stays.
- **What the read proves.** It is a fresh daemon-side reading at the time of the request. It is not a guarantee that nothing moved in the milliseconds after it. The rate cap bounds how far the robot can move in that window.

**R2. An axis with no readable reference is refused.**
- The `current.get(…, 0.0)` fallbacks go, including `target_body = …get("body_yaw", 0.0)`.
- An impossible body or neck-relative value (beyond the limit plus 5°) is refused. World yaw is never dropped, since it legitimately reaches 225°.
- Antennas are checked for finiteness only (F9).
- `get_current_pose()`, the WS cache, stays for observation (sync, the divergence warning, DN) and no longer feeds a dispatch.

**R3. Frames.**
- Body and antenna deltas are raw joint differences (F5).
- Head comparisons use R_rel = Rz(−body)·R_world (a left multiply) and the geodesic angle between rotations.
- No Euler subtraction remains on any path. That covers the seed `current["yaw"] - current["body_yaw"]`, `_warn_commanded_vs_achieved`, and FocusOnSound's diagnostic subtraction.
- x, y and z are not rate-checked (§8). Lifecycle re-centres hold the readback's translation (5.3).

**R4. The rate check (O2).**
- `required = max(Δ) · max(1 / 180°/s, peak(method) / 340°/s)`, computed per chunk.
- The Δ terms are the body, the head geodesic in R_rel, and each antenna, all measured from the fresh readback.
- `"exact"` means a duration below `required` is refused (`rate`). `"at_least"` means duration = max(requested, required).
- **Under a split** (Executor N1; delta S2), the requested duration is divided across the chunks in proportion to each chunk's Δ. Each share is a derived value, so it is floored with `floor_duration` (0.1 s) before the chunk is resolved on its own.
  - Example: `turn_around`'s 5 s over a 300° body chunk and a 5° yaw chunk gives a yaw share of 0.08 s, which floors to 0.1 s and is not refused.
  - The lifecycle 2 s floor (5.3) applies to the whole re-centre and is shared the same way.
- Each chunk passes `require_duration`. Over 10 s is refused, never clamped.

**R5. The path rule (O1; Executor D1, S1).**
- **The measure is the neck twist about the body's z axis.** Take R_rel(s) = Rz(−body(s))·R_world(s), decompose it into swing and twist about z, and use the twist angle. Euler yaw is not used: Executor S1 showed it overshoots mid-path whenever roll and pitch both change, even with the twist held.
- **The check.** `neck_path_ok(start, target)` samples s ∈ [0, 1] at 32 points. It uses the SDK's own `linear_pose_interpolation` for the head and the joint-space line for the body: the same function the daemon runs, imported, not re-implemented.
  - It passes when the maximum |twist| along the path is at most max(|twist| at the start, |twist| at the target) + 10°, and never more than 90°.
  - 90° is far above the 65° neck plus the measured overshoot, and far below the 180° sweep.
  - The 10° and the 90° are fold-made choices (§12); the build records the overshoot the property sweep measures.
  - Cost: 32 slerps, well under a millisecond.
- **σ stays as a cheap pre-filter only.** σ = |Δbody + Δrel_yaw| > 150° always fails. σ ≤ 150° still runs the sampler, because of F6's 179.7° case.
- **The split order** (Executor-measured worst neck twist of about 65° for each stage):
  1. **Body chunks:** ⌈|Δbody| / 150°⌉ chunks, with roll, pitch **and** relative yaw all held at the start value;
  2. one **relative-yaw chunk**, with roll and pitch held;
  3. one **roll-and-pitch chunk**.

  Stages with zero change are skipped. **Every chunk is run through the sampler before anything is dispatched.** If any chunk fails, the whole move is refused with `path` before chunk 1 moves (fail-closed). A split is never started that cannot finish.
- **Re-check at every chunk (delta S1, fail-open closed).** Before chunk k > 1 is dispatched:
  1. take a fresh readback;
  2. require it to be within `_STASH_DIVERGENCE_REFUSE_RAD` (the rig-measured end-of-move lag plus a margin; provisionally 5°) of chunk k − 1's target on every moved axis;
  3. re-run `neck_path_ok` from that readback to chunk k's target.

  Either failure stops the move with `partial`. This catches a chunk the daemon dropped silently (F3) while the client reported it complete: without the check, the next chunk would run as a 300° body move.
- **`"exact"`.** A move that fails the sampler as one dispatch is refused with `path`, naming the measured twist.

## 4. Types and placement (Architecture N3, S4)

- **In `hardware/controller.py` (robot-agnostic):**
  - `MotionRefusalKind` gains `rate | reference | path | limit`.
  - `MotionTarget` and `PixelTarget` gain `duration_policy: Literal["exact", "at_least"] = "at_least"` (trailing, with a default; no positional constructor exists anywhere, per the Wire check).
  - **`MotionOutcome`**, a frozen dataclass that is never persisted or sent over a wire (CC3, stated in its docstring):
    - `ok: bool`
    - `kind: MotionRefusalKind | Literal["superseded", "cancelled", "partial", "error"] | None`
    - `reason: str`
    - `duration_s: float | None`
    - `chunks_planned: int`
    - `chunks_done: int`
    - `__bool__` returns `ok`.
  - **`as_outcome(result: bool | MotionOutcome) -> MotionOutcome`** (O7). A `bool` becomes `MotionOutcome(ok=result, kind=None if result else "error", reason="" if result else "the controller returned False", …)`. Every consumer that reads a field calls it: the tools, the motor worker, `motor_backend.execute`, Selfy and DN.
  - **`MotionTicket`** (Architecture delta S2) lives here, beside `MotionOutcome`, so `hardware/` never imports from `embodied_runtime/`. It is two `concurrent.futures.Future`s, `accepted` and `completed`, plus the pain event id (5.6).
  - **`current_motion_ticket`**, a `contextvars.ContextVar[MotionTicket | None]` (default `None`). The motor worker binds it around each callable with `token = var.set(t)` and calls `var.reset(token)` in `finally`, so a raising callable never leaves its ticket bound. The controller calls `mark_accepted()` through it when 5.1 steps 1 to 4 pass. With nothing bound (a synchronous caller or a plugin) that call does nothing, and the worker resolves `accepted` itself when the callable returns. So a plugin still resolves both stages.
  - **`MotionTicket.mark_accepted()` is the only way `accepted` resolves** (Architecture scoped S2). It does three things in order:
    1. runs the ticket's `on_accept` callable (Selfy stores the pain start there), and stores the event id it returns;
    2. resolves `accepted`;
    3. is idempotent: a second call does nothing.

    A raising `on_accept` is logged in the Stage-1 form (`log_swallowed_exception`) and `accepted` still resolves. No `Future.add_done_callback` carries logic, because those swallow exceptions. The worker's plugin fallback calls the same `mark_accepted()`. Since `on_accept` is a callable carried on the ticket, `hardware/` never imports `bridges/`.
  - **The layering is enforced, not assumed** (Architecture scoped S1). `utils/audit.py::LAYER_RULES` has no `"hardware"` key today. PR 1 adds `"hardware": {"must_not_import": ["embodied_runtime", "tools", "runtime", "agents", "planning"]}` (`hardware/` is clean against it today). The stale `conscience` → `embodied_runtime` naming in the other rules is a follow-up issue.
  - Any `-> bool` function that returns a controller result goes through `bool(...)` or `as_outcome(...)`. The mypy per-file ratchet flags an unwrapped one.
  - `ConfigurationError`'s subclass `InvalidConfigError` (it already exists in `maxim.exceptions`) is the O6 error.
- **One API, 1.x-compatible (Architecture S4; delta S1; Wire D1; O7).**
  - The ABC's `goto_target` is annotated `-> bool | MotionOutcome`. The Reachy and simulated controllers return `MotionOutcome`; plugins may keep returning `bool`. There is no second `dispatch()` API.
  - The ABC's `look_at_pixel`, `center_vision`, `wake_up` and `goto_sleep` get the same `bool | MotionOutcome` annotation, and their consumers go through `as_outcome` too (this resolves the §4 vs 5.3 disagreement).
  - The `hardware/registry.py` broadcasts keep their public `dict[str, bool]`, as `bool(as_outcome(r))`, so the registry's own contract does not move.
  - The docstrings in docs/user/extension_api.md and robot-setup.md say that a `MotionOutcome` is preferred and a `bool` is still accepted.
- **In `hardware/reachy/controller.py` (SDK-specific):**
  - the constants `MAX_RATE_AVG_DEG_S = 180`, `MAX_RATE_PEAK_DEG_S = 340`, `_PEAK_FACTOR`, `PATH_SIGMA_PREFILTER_DEG = 150`, `NECK_PATH_SLACK_DEG = 10` and `NECK_PATH_CEILING_DEG = 90`;
  - `_STASH_DIVERGENCE_REFUSE_RAD` and the fresh-read timeouts;
  - `resolve_duration(...)` and `neck_path_ok(...)`, which describe the SDK's interpolation.

## 5. Boundaries

**5.1 The controller dispatch** (`ReachyMiniController`). `_goto_target_locked` becomes phase helpers (Rule T, Architecture S8). It is 177 lines today, and these phases would put it over 200.

The phases run in this order:
1. `_fresh_readback`
2. `_check_divergence`
3. `_compose` (against the readback body and the stash-filled head axes)
4. `_plan_chunks` (R5 split plus sampler, all chunks checked before any moves)
5. for each chunk:
   - `_resolve_rate` (R4)
   - the last mile
   - the SDK `goto_target`
   - the stash write for that chunk
   - checking the abort event
6. return the `MotionOutcome`

Rules around the loop:
- **The abort token (Architecture S5; Executor scoped round).** A plain event cleared "at the start of each dispatch" is lost: `goto_sleep` sets it mid-split, the split stops and releases the lock, and the worker's next queued command (the `RLock` is not fair) takes the lock, clears the event and moves. `goto_sleep` then waits behind it and can time out, so motion continues after a stop and the robot never sleeps. Instead:
  - **A generation counter.** `request_abort()` increments `_abort_gen`. Each dispatch records the generation at its start and checks it between chunks: a change stops the move with `cancelled`, carrying `chunks_done`. Nothing but the stop/sleep path ever resets it, so a later dispatch cannot consume someone else's abort.
  - **A stop fence.** `goto_sleep`, `disconnect` and the runtime's stop path call `request_abort()` **before** they try to take the lock. The stop path also calls `MotorQueue.close()` first, so no queued command can be dispatched behind a stop.
  - **No timeout for the stop path.** The lifecycle and stop path wait for the lock without the 25 s timeout. The fence makes that wait bounded: at most the current chunk, ≤ 11 s.
- **Mid-split failure (Executor S4).** A chunk k > 1 can fail on a fresh read, a last mile or an SDK `TimeoutError`. Then:
  - The outcome is `partial`, with `chunks_done = k − 1`; the learners ignore it.
  - The stash describes the last completed chunk, because each chunk's stash write follows its own dispatch.
  - After an SDK `TimeoutError`, the controller refuses further dispatches with `reference` until the move is **settled** (delta N4; `is_move_running` cannot be read over REST, F12).
    - Settled means a fresh readback within 2° of the timed-out chunk's target on every moved axis, **or** readbacks stable within 0.5° over a full second, meaning the move ended somewhere else.
    - Two reads 100 ms apart are not enough: a slow minimum-jerk move near its ends moves less than 5°/s.
- **Gate 15, reworded:** a refusal before chunk 1 leaves the stash identical.

**5.2 `look_at`, both paths (F1; Executor N2).**
- The helper takes the fresh readback, then calls `self._mini.look_at_image(u, v, perform_movement=False)`. That call reads the WS cache's head pose for the camera ray itself (one more read, at most 20 ms older; stated, and the rate cap bounds the skew).
- R_rel is then taken against the fresh readback's body, and the result is dispatched as `MotionTarget(roll, pitch, head_yaw=rel_yaw, body_yaw=None, duration, duration_policy)` through 5.1, under the same lock hold.
- The body never moves. `ReachyMiniController.look_at_pixel` and Selfy's SDK look-at (`_enqueue_sdk_look_at`/`_checked_look_at`) both use this helper, and the raw `look_at_image` call leaves the allow-list.
- **Stated behaviour change:** look_at no longer recentres the body. `_maybe_turn_around` is covered by 5.8.

**5.3 Lifecycle motion (F11, F14; Executor S5, S6; Architecture N1).**
- **`wake_up` and `goto_sleep`, under `_motion_lock`:**
  - If |body| > 2° or |neck twist| > 2° on the fresh read, re-centre first through 5.1:
    - body to 0 and relative yaw to 0;
    - **holding the readback's roll, pitch and x/y/z translation**, so the head is never lifted out of the sleep pose. (The SDK moves translation along a straight world chord, so over a 150° body chunk a held x/y offset passes inside its arc mid-path: about 16 mm on a 21 mm offset, with z unaffected. Stated, mm-scale.) Today the controller always composes zero translation, and `MotionTarget` has no x/y/z (delta N1). So a controller-internal `_hold_translation` argument to the compose phase carries the readback's translation **in the body frame**: rotated with the body, as `move_head` does, so a world-held x/y does not slide sideways during a body re-centre. It is not a public `MotionTarget` field;
    - `"at_least"`, with a floor of 2 s;
    - split by R5 as needed.
  - Then call the SDK's canned move. Its body starts at 0, so it cannot sweep.
  - With no fresh readback, or a refused re-centre, there is no canned move, and the call returns a failed outcome.
- **Torque off.** The daemon's `enable_motors` pins its targets to the present pose, and the pose is the forward kinematics of the present joints, so the read is valid (Executor-checked).
- **Torque-on race (delta S3).** The SDK's `enable_motors()` is a fire-and-forget WS `SetTorqueCmd`, so the next REST read can still say `control_mode: "disabled"`. `wake_up` therefore polls the REST `control_mode` until it reads `enabled`, bounded at 1 s, before the re-centre and the canned move. On timeout it reports the failure, with no motion. Without this, a cold boot (the daemon starts with `--no-wake-up-on-start`) would never wake.
- **The `enable_motors()` failure branch commands no motion** (Executor S6). It reports the failure and calls `disable_motors()`, best-effort and logged. The old safety `goto_sleep` there goes away.
- **Canned antennas.** The canned sleep move takes the antennas to ±3.05 rad, beyond the provisional 90° limit. The limit governs commands Maxim composes; the canned poses are the vendor's mechanical rest, stated and re-checked when pulse's hand-measured range arrives.
- **Raw callers move to the controller.** These are `agentic_runtime.py`'s `mini.wake_up` and `connection.py`'s soft-reconnect `wake_up`.
  - **Stated behaviour change:** a reconnect while the body is turned re-centres it (split, at most 4 chunks), and the abort event can interrupt that.
  - Selfy's sleep and wake paths go through the controller too.
- The torque-off "not enabled" message in Exp 53 is reworded to distinguish "refused" from "torque".

**5.4 `motion/movement.py` (Architecture S3; Executor S7).**
- When a controller is present, `move_head` and `move_antenna` **delegate** to the controller's dispatch. No second copy of R1 to R5 exists.
- The standalone path (scripts with no controller) calls the same phase helpers through one module function, `check_standalone_motion(mini, …)`. It still takes `motion_lock=` as a required keyword argument, so forgetting it is a `TypeError`.
- The IK-recovery `move_head` inside `motor_worker` goes through the controller.
- The ~18 `move_head(` callers in `src/`, `scripts/` and `tests/` are updated in the same PR; the review maps each one.
- `move_antenna`'s first move after sleep stretches under `"at_least"`, from the clamped target.

**5.5 Tools and Selfy (round-5 S4, S5; Wire 9).**
- **The explicit-duration key per tool:**

  | Tool | Key that means "explicit" |
  |---|---|
  | MoveTool, FocusOnSoundTool | `duration` in kwargs |
  | MaximCommandTool | `duration` in `kwargs["params"]` (what `_motion_number(params, …)` reads) |
  | TrackTool, NoveltyTrackTool | `duration_s` |

  When the key is present the policy is `"exact"`; otherwise it is `"at_least"`.
- **Other policies:**
  - `turn_around`'s hard-coded 5.0 is `"at_least"`.
  - **`goto_pose`'s pose-table duration is authored, not chosen by the model, so it is `"at_least"`.** Treating it as explicit would refuse authored poses.
- Selfy's `move`, `move_relative`, `turn_around`, `look_at_image`, `move_antenna`, `goto_pose` and `center_vision` gain `duration_policy=`.
- Derived durations are floored with `floor_duration` before a target is built (S5).
- The schemas drop their advertised duration defaults.
- **`media_loop.py::act`** waits for the ticket's `completed` (5.6) instead of sleeping the requested time. Today a stretched move overlaps the next one, which the latest-wins queue then drops.

**5.6 Outcomes (O4; Architecture S1, S2, S9; Executor S2, S3; Wire 1 to 5).**
- **A `MotorQueue` type owns the queue** (in `embodied_runtime/workers.py`). It is the CLAUDE.md "push silent no-ops into types" rule.
  - **Operations:** `put` (latest-wins), `clear`, `drain` and `close`. It resolves every ticket it removes:
    - a replaced command → `superseded`;
    - the `Full` case → `superseded`;
    - a command removed by `clear` or by the reconnect drain → `superseded`;
    - one still queued at `close` → `cancelled`.
  - **Raw queue access leaves** `connection.py`, `media_loop.py` and the tests. The `live()` teardown calls `close()`, which also takes the edits out of `MediaLoopMixin.live` (pinned at 425).
  - With no queue (the synchronous path), the call runs inline and returns an already-resolved ticket.
- **`MotionTicket` has two stages** (Architecture S2; Executor S3):
  - `accepted` resolves when the checks of 5.1 steps 1 to 4 pass and the first SDK call starts, or with the refusal.
  - `completed` resolves with the final `MotionOutcome`.
  - The worker calls `set_running_or_notify_cancel()` before it runs a callable, so a cancelled ticket never moves.
- **Tools wait on `accepted` only.** That is when the reason is known. The wait is at most 25 s, under `Tool.timeout` and `per_tool_timeout` (30 s).
  - A timeout cancels the ticket and returns "not started: the motor is busy", never success.
  - MoveTool and FocusOnSound call the controller synchronously on the tool thread. They get the outcome directly (through `as_outcome`), and no ticket is involved.
  - **Their lock wait is bounded (delta N5).** The controller acquires `_motion_lock` with `acquire(timeout=25)` on a dispatch that does not come from the motor thread. A timeout returns `MotionOutcome(ok=False, kind="cancelled", reason="motor busy")`, so the tool never runs past `per_tool_timeout`.
- **The motor worker** reads the callable's `MotionOutcome`:
  - **The success branch** (`record_gaze(success=True)` and `record_movement_outcome`) **runs only when `outcome.ok` and the move completed** (Wire 1; Architecture D1).
  - The generic-exception branch no longer feeds the learners.
  - Only the existing IK-failure detection reaches `record_gaze(success=False)` and the bounds learner, as today.
- **No refusal kind feeds a learner (O4).** That covers `rate`, `reference`, `path`, `limit`, `non_finite`, `duration`, `method`, `superseded`, `cancelled`, `partial` and `error`. It also covers DN's `spatial_map.record_movement` in `_record_movement_result` (Wire 5) and `gaze_manager.py::look_opposite`'s success-at-enqueue (Wire, table).
- **DN (Architecture S9; Wire 5).** DN never runs a callback on the motor thread. The ticket's `completed` posts the outcome to a DN-owned inbox, a `queue.SimpleQueue` drained on DN's own tick. That tick then updates:
  - `executed` and the stats;
  - `novelty_tracker.focus`;
  - `gaze_history.record_gaze`;
  - `_record_movement_result`;
  - `_publish_action`.

  No `concurrent.futures` callback carries logic that could raise, because exceptions in those callbacks are swallowed.
- **The pain bridge (Wire 2; delta D2).**
  - **The start is recorded when the move starts, not when it is queued.** Today Selfy calls `_record_pain_action_start` on DN's thread before the enqueue. So when A is moving and B is queued, B's start retires A, A's pain is booked against B, and the detector judges A's motion against B's target.
  - Now the ticket carries the start's arguments, and the motor thread records the start when `accepted` resolves. That records the start, sets the detector's target, and stores the returned event id on the ticket.
  - `record_action_complete(..., event_id=None)` keeps its keyword default (18 call sites in `test_pain_circuit_booking_1209.py`). Given an id, it completes that event only, so a superseded or stale completion can never book the move that replaced it.
  - A ticket resolved `superseded` or `cancelled` before `accepted` never recorded a start, so it completes nothing.
  - **`turn_around` (synchronous, on DN's or a tool's thread) binds its own ticket** (Wire scoped S1). It sets `current_motion_ticket` around its controller dispatches, with `on_accept` carrying its pain start. So the controller records the start inside `_motion_lock`, at the first chunk's `accepted`, after any in-flight motor-thread move has released the lock. Recording before the dispatch would bring D2 back: the turn waits behind move A and books A's pain against the turn. `_maybe_turn_around` goes the same way.
  - **The bridge has two producer threads now, so retire-then-install is atomic** (Wire scoped S2). `bridges/pain_bridge.py::record_action_start` reads `_pending` for `_retire` outside `_pending_lock` today, so two concurrent starts can both retire the same old event, and the other new one is overwritten unretired (NAc's `discard_pending_event` never runs). The retire and the install move under one `_pending_lock` hold. The other `_pending` readers are unchanged.
- **The public Selfy methods that queue motion return the ticket.** These are `look_at_image`, `move`, `move_relative`, `goto_pose`, `center_vision`, `move_antenna` and `media_loop.act` (Wire NIT).
  - Their early returns return a resolved ticket: `ok` with zero chunks for a no-op (`move`'s no-change return), or `limit` for "unreachable" (`look_at_image`'s reachability return).
  - This is a public contract change, listed in §6.
- **`turn_around` is synchronous, so it returns a `MotionOutcome`, not a ticket** (delta D3). A ticket is never `False`, so the two identity checks would go silent on one. Both move to truthiness:
  - `tools/reachy.py` MaximCommandTool's `if turned is False:` becomes `if not turned:`;
  - `embodied_runtime/movement.py::_maybe_turn_around`'s `success=turned is not False` becomes `success=bool(turned)`.

  The refused or `limit` turn then reports failure to the model and books failure.

**5.7 `automatic_body_yaw` (O6).**
- The controller's `__init__` **keeps the `automatic_body_yaw` parameter** and raises `InvalidConfigError` when it is `True` (delta D4).
  - The parameter has to stay: `hardware/registry.py::connect_robot` filters config keys to the constructor's signature and drops the rest at DEBUG. Removing it would make a `robots.yaml` `automatic_body_yaw: true` silently ignored, not refused.
  - Its docstring says `True` is refused, and why.
  - No shipped YAML or doctor check sets it (Wire-checked).
- At connect, after `wake_up`, and in `adopt(sdk)` (embodied_device.md §2), the controller asserts `set_automatic_body_yaw(False)`. If that assertion raises, the controller refuses motion (`reference`) until an assertion succeeds.
- `scripts/reachy_streamer.py`, `audio_localization.py` (the `mini=None` branch), `live_common.py` (both constructions) and `live_1_smoke.py` (both) pass `False`. `live_1b_controller_smoke.py`'s raw `set_automatic_body_yaw` joins the allow-list with its reason.
- The note to pulse (M70) asks that the app framework's handle be built with `automatic_body_yaw=False`, or that pulse accept pymaxim's assertion on adopt.

**5.8 Turning at the body limit (decision 4; Wire 4).**
- `_maybe_turn_around` reads the body from the fresh readback. It returns False when the turn cannot fit, and look_at proceeds with the head clamped.
- An explicit `turn_around` that cannot fit raises `limit`, and the tool returns that reason. **`limit` feeds no learner** (O4): a body limit is not a head-yaw bound.

**5.9 Named callers.**
- `RobotController.center_vision` sends body 0 and head 0 under `"at_least"`, through R5.
- DN tracking passes its 0.15 s base as `requested` under `"at_least"`.
- `hardware/reachy/motor_backend.py::execute` (the orient affordance, Wire) reads the `MotionOutcome` and returns its reason instead of "rejected by controller" (Wire 11).
- **Provenance notes** go in the same PR (Wire 10):
  - `scripts/exp49/run_trials.py`: 0.3 s steps, checks the outcome; recorded in Exp 49's provenance.
  - `delivered_shift_block.py::recenter`: now splits from ±160°.
  - Exp 53's `goto_body_yaw`: now splits from ±160°.
  - `live_1b_controller_smoke.py`: checks its outcome.

## 6. Contracts this changes (for the delta round's Wire lens)

| Contract | Change | Producers and consumers |
|---|---|---|
| `RobotController.goto_target`, `look_at_pixel`, `center_vision`, `wake_up`, `goto_sleep` (ABC) | `bool` → `bool \| MotionOutcome`, read through `as_outcome` (O7); the registry broadcasts keep `dict[str, bool]` | plugins (`maxim.robots` entry points) and bool fakes stay valid; Reachy and simulated controllers, the registry broadcast, the tools, `motor_backend.execute`, Selfy, DN, scripts |
| the motor queue | raw `queue.Queue` → `MotorQueue` with tickets | `media_loop._enqueue_motor`/`_clear_motor_queue`/`live` teardown, `connection.py` reconnect, `workers.motor_worker`, `tests/unit/test_media_loop.py` |
| Selfy public queued-motion methods (+ `move_antenna`, `media_loop.act`) | `None`/`bool` → `MotionTicket`; `turn_around` → `MotionOutcome` (sync), with both identity checks moved to truthiness | the tools, DN `_dispatch_action_to_motor`, `gaze_manager.look_opposite`, `media_loop.act` |
| `MotionRefusalKind` | + `rate reference path limit` | DN's `network.py` handler, the tools, the worker. Nothing reads `.kind` today (Wire check) |
| `MotionTarget`/`PixelTarget` | + `duration_policy` | 18 constructors in src and scripts, 8 test files, all keyword-only |
| `move_head`/`move_antenna` | delegate to the controller; standalone needs `motion_lock=` | ~18 callers |
| `look_at` | the body no longer recentres | `_maybe_turn_around`, the experiments that read the body after look_at |
| `PainCircuitBridge` | start recorded on the motor thread at `accepted`; complete keyed by `event_id=None` keyword | Selfy, DN, `test_pain_circuit_booking_1209.py` (18 sites, default kept) |
| the `automatic_body_yaw` config | `True` is a config error | robots.yaml, the controller, the scripts, pulse's handle (note to pulse) |
| the retained-axes semantics | the body stash no longer composes | §7, embodiment.md §1, the CLAUDE.md Reachy invariant |

## 7. Tests and invariant text this design inverts (Executor S7; Wire 7)

**Shared fake.** One shared fake SDK and daemon fixture (`tests/unit/_reachy_fakes.py`, extending the one #1496 added) serves `GET /api/state/full` from its modelled state and models F1 to F3 and F11. Every Reachy test file moves to it. About 100 controller motion calls in `test_reachy_retained_axes.py`, `test_reachy_workspace_safety.py`, `test_reachy_head_frame.py`, `test_motion_finite_1496.py` and `test_reachy_connection_options.py` would otherwise refuse with `reference`.

**Inverted, each rewritten as a strict xfail in the red-gate commit, never deleted:**
- `test_reachy_retained_axes.py`:
  - `test_head_only_commands_use_last_commanded_body_for_composition` → the readback body composes;
  - `test_body_stash_unblocks_unreadable_pose` → refuses;
  - `test_stash_survives_unreadable_pose_after_first_command` → refuses, and the stash survives;
  - `test_look_at_pixel_keeps_body_stash` → rewritten per 5.2;
  - `test_look_at_pixel_clears_head_axes`;
  - `TestAutomaticBodyYawMode::test_body_is_never_stashed` and `::test_head_only_composes_against_readback_body` → a config error (O6);
  - `TestLifecycleClearWiring` wake/sleep → re-centre per 5.3.
- `test_reachy_workspace_safety.py::test_body_read_failure_falls_back_to_zero` → refuses.
- `test_motion_finite_1496.py::test_a_failed_pose_read_counts_as_missing_not_refused` → refuses.
- `test_move_head_without_controller_still_dispatches`, and `test_workers_recovery_recenter_is_body_relative` → standalone with `motion_lock=`, and recovery through the controller.
- `test_agentic_runtime.py` (wake through the controller), and the `test_media_loop.py` enqueue tests (`MotorQueue`).
- **The 43 `... is True` assertions on controller motion calls** in the Reachy tests (delta D3). Each becomes a truthiness assert, or reads `.ok`.
- `test_reachy_connection_options.py`: constructing with `True` now raises `InvalidConfigError`, and the old-SDK `AttributeError` case is rewritten (delta D4).
- **The test stubs where `_enqueue_motor` returns `None`** (test_movement.py, test_agentic_runtime.py, test_motion_finite_1496.py, test_reachy_workspace_safety.py), and test_sem_motor_binding.py's assert that `_clear_motor_queue` is called (Wire NIT), move to `MotorQueue`.
- **The bool-returning fakes stay as they are** (test_sem_motor_binding.py, test_focus_on_sound.py, test_multi_robot_extensibility.py::_Minimal, test_robots_yaml_connect_config.py::_RecordingController, test_move_tool_gaze.py). Through `as_outcome` they are O7's compatibility proof.

**Text.** The `docs/agents/embodiment.md` §1 and CLAUDE.md wording "composed onto the last COMMANDED body" is rewritten to R1, in the red-gate commit. So are two more embodiment.md §1 stubs (Architecture delta N3):
- `reachy-motion-dispatch-safety`: "the only other sanctioned SDK primitives", "readback seeds an axis exactly once" and `note_external_head_motion`;
- the #1496 stub's "a stale pose cache is #1499's". The CLAUDE.md Reachy invariant keeps its two hard rules, gains the reference and path rules, and cites its regression guards.

## 8. Stated out of scope and assumptions

- **Single client.** Moves, the drop rule (F3) and `automatic_body_yaw` are daemon-global. Another client can bypass or undo everything here: a LAN script on `0.0.0.0:8000`, a second Maxim, or pulse's app handle before pymaxim's assertion on adopt. That belongs to the embodied-device plan (one owner per device).
- **x, y and z translation rates** are not checked, except that lifecycle re-centres hold the readback's translation.
- **Raw harness motion stays allow-listed and bypasses rate and path** (stated, Wire 10): `scripts/orient_backbone/live_common.py::goto_body_yaw`, used by `doa_sweep`, `doa_settle`, `live_2_reactive` and `yaw_verify` at 0.5 to 0.6 s for any angle.
- **The `Selfy.mini` facade** is a follow-up issue, filed with PR 1 (O5).
- **The daemon's IK NaN behaviour** is not counted.

## 9. Red gates (strict xfail, written first; each drives the real path against the shared fake)

1. **Path sampler (R5).** A table plus a seeded property sweep over legal starts and targets with roll and pitch moving. It includes:
   - F6's 179.7° case;
   - center_vision from 160°/30°;
   - wake from body 150°;
   - `turn_around(170)`.

   Every dispatched chunk keeps the sampled twist within the R5 bound (the sampler uses the SDK's `linear_pose_interpolation`). The same moves under `"exact"` are refused with `path`, and no chunk is dispatched when any planned chunk fails.
2. **Rate.**
   - 160° in 0.5 s, `"exact"` → `rate`;
   - `"at_least"` → at least 0.889 s for minimum_jerk;
   - `ease_in_out` → at least 0.941 s;
   - 30° in 0.3 s passes;
   - DN 40° from 0.15 s → at least 0.222 s;
   - `turn_around` 5 s over a split is shared out, not multiplied.
3. **Fresh read (O3).** The fake REST answers 500, times out, omits `body_yaw`, returns NaN, or reports torque off → each is `reference` with no SDK motion call.
4. **Divergence.**
   - Stash body 150°, readback 0° → `reference`; the stash is forgotten and the next move succeeds.
   - A diverged roll stash refuses only when roll is retained.
5. **No reference.** A missing `body_yaw` → refused. A world yaw of 200° is not dropped.
6. **Wrap.** World 170°, body 160°, target world 175° → passes, with a relative command near 15°.
7. **look_at.** Body 1.0 rad → no body command, on both paths.
8. **Lifecycle.**
   - Body 150° → re-centre chunks hold the readback's translation and last at least 2 s; the canned move sees body 0.
   - Asleep at body 0 → no re-centre.
   - `enable_motors` raising → no motion.
   - Reconnect and the agentic wake go through the controller.
9. **Auto yaw.** `automatic_body_yaw=True` → `InvalidConfigError` at construction, including through `registry.connect_robot` from a `robots.yaml` (not silently filtered). `set_automatic_body_yaw(False)` is asserted at connect, after wake and on adopt. A planted `ReachyMini()` without the argument fails the lint.
10. **Queue.** Each of the following resolves its ticket and never leaves one pending:
    - replaced → `superseded`;
    - `Full` → `superseded`;
    - `clear` → `superseded`;
    - the reconnect drain → `superseded`;
    - close → `cancelled`.
11. **Two-stage ticket.**
    - A refused move's reason reaches MaximCommandTool within the 25 s wait.
    - A busy motor beyond 25 s → "not started"; the ticket is cancelled and never moves.
    - MoveTool and FocusOnSound get the outcome synchronously.
12. **Learners.**
    - **PR 1:** a refused or partial move, and any non-ok outcome, never reaches the motor worker's `record_gaze(success=True)`, `record_movement_outcome` or `_contract_bound`. The bounds file is unchanged after a burst of refusals.
    - **PR 2** (Architecture delta S3): DN's `spatial_map.record_movement` and `executed` follow the outcome. Until PR 2, DN's success-at-enqueue stays as it is on `main`. That inflates reachability but never contracts a bound.
13. **DN.** The outcome is applied on DN's tick, not on the motor thread (a thread-identity assert). A raising DN handler is logged, not swallowed.
14. **Pain bridge.** While A moves, B is queued, and B's start is not recorded until B is accepted, so A's pain is booked against A. A's completion does not complete B. A ticket superseded before `accepted` completes nothing.
    - **Turn variant:** a `turn_around` issued while A moves records its start only after A releases the lock.
    - **A raising `on_accept`** is logged, and `accepted` still resolves.
    - **Two threads** starting concurrently retire each old event exactly once, with no leaked pending event.
15. **Stash.** A refusal before chunk 1 leaves `_last_commanded` identical. A `partial` stash equals the last completed chunk.
16. **Abort.** `goto_sleep` during a split → `cancelled`, with `chunks_done < chunks_planned`, and no chunk after the abort. **A queued command behind a stop is never dispatched**, and `goto_sleep` runs after the split stops, never "motor busy" (an unfair-lock race, forced in the test).
17. **Timeout.** An SDK `TimeoutError` at chunk 2 → `partial`, and further dispatches are refused until settled.
18. **Delegation.** `move_head` with a controller calls the controller's dispatch. The standalone `move_head` without `motion_lock=` raises `TypeError`.
19. **Lint (O5).** Planted `mini.wake_up()`, `m = self._mini; m.goto_sleep()`, `getattr(mini, "enable_wobbling")`, `set_automatic_body_yaw`, `cancel_move`, `mini.client.send_command` and `send_task_request` outside the allow-list are each caught (Executor N6).
20. **Plugin compatibility (O7).** A bool-only plugin controller driven through the tools, the motor worker and `motor_backend.execute`: no `AttributeError`, and False reads as `kind="error"`.
21. **The ticket seam.** With `current_motion_ticket` bound, the controller resolves `accepted` before the SDK call. With it unbound (a plugin), the worker resolves both stages. `hardware/` imports nothing from `embodied_runtime/` (pinned by the architecture audit).
22. **Per-chunk re-check (delta S1).** The fake daemon drops chunk 1 silently and reports it completed. Chunk 2 is never dispatched, and the outcome is `partial` with `chunks_done=1`.
23. **Torque-on race (delta S3).** The fake reports `disabled` for 300 ms after `enable_motors()`: `wake_up` waits and wakes. For 1.5 s: `wake_up` fails with no motion.
24. **REST read.** A contract test against the real daemon app at the SDK floor and latest (R2 reachy cell) pins the parser. Shape variants (a nested `body_yaw`, the `head_pose.m` matrix) parse. An unknown shape → `reference`, plus one ERROR at connect naming the SDK version.
25. **Synchronous lock bound (delta N5).** A tool-thread dispatch behind a 30 s in-flight move returns "motor busy" within 25 s.
26. **Turn identity (delta D3).** A refused `turn_around` reaches the model as an error and books failure.

**Deletion probes after the build.** Remove each of: R1 to R5, the REST read, the sampler, the split pre-check, the auto-yaw error, the `MotorQueue` resolutions, the learner gate, the inbox and the event-id pairing. Each must turn its gate red.

## 10. Rule T and function lengths (Architecture S8)

| Function | Lines today | Plan |
|---|---|---|
| `ReachyMiniController._goto_target_locked` | 177 | phase helpers (5.1); it shrinks |
| `workers.motor_worker` | 188 | outcome handling and the learner gate move to `_apply_outcome(...)` |
| `FocusOnSoundTool.execute` | 199 | the outcome read goes into `_aim` |
| `Selfy.turn_around` | 189 | the ticket and policy plumbing go into a helper |
| `Selfy.move` | 173 | the ticket and policy plumbing go into a helper |
| `MediaLoopMixin.live` | pinned 425 | teardown becomes `MotorQueue.close()`; it shrinks |
| `_stop_agentic_runtime` | pinned 207 | the abort-event set is extracted |
| `Maxim.__init__` | pinned 312 | untouched, or the touch is extracted |
| `ReachyMiniController.connect` | 162 | gains the O6 assertion and the shape probe; it has headroom, and is listed so it stays under 200 |

Every pinned function the build edits must shrink by at least the lines touched. The lint's numbers are checked at the red-gate commit and at the end.

## 11. Cross-reference: the v6 review → where each finding lives

| Finding | Home |
|---|---|
| Executor D1 (σ misses roll and pitch) | R5 sampler + split order, F6, gate 1 |
| Executor S1 (Euler measure) | R5 twist measure, §12 |
| Executor S2 / Architecture S1 / Wire 3 (orphaned futures) | 5.6 `MotorQueue`, gate 10 |
| Executor S3 / Architecture S2 (wait too long, timeout is not cancel) | 5.6 two-stage ticket, gate 11 |
| Executor S4 (mid-split failure) | 5.1, gates 15 and 17 |
| Executor S5 (re-centre from the sleep pose) | 5.3, gate 8 |
| Executor S6 (no motion when `enable_motors` fails) | 5.3, gate 8 |
| Executor S7 / Wire 7 (inverted tests) | §7 |
| Executor N1 (requested under a split) | R4 |
| Executor N2 (two reads in look_at) | 5.2 |
| Executor N3 (liveness, not recency) | O3 re-decided (REST), R1 |
| Executor N4 | no finding |
| Executor N5 / Architecture S9 / Wire 5 (DN off its thread) | 5.6 DN inbox, gate 13 |
| Executor N6 (`send_command`) | gate 19 |
| Architecture D1 / Wire 1 (learner poisoning in PR 1) | 5.6, gate 12, §13 (PR 1 carries it) |
| Architecture S3 (two copies of the gate) | 5.4 delegation, gate 18 |
| Architecture S4 / Wire 8 (two APIs; ABC) | §4 one API |
| Architecture S5 (lock held 41 s, no abort) | 5.1 abort event, gate 16 |
| Architecture S6 (private SDK state) | O3 re-decided (REST) |
| Architecture S7 (the option name lies) | O6 re-decided, 5.7, gate 9 |
| Architecture S8 (Rule T) | §10 |
| Architecture N1 (reconnect re-centre) | 5.3 |
| Architecture N2 (decisions) | §13: DECISIONS.md MP1 onward |
| Architecture N3 (placement) | §4 |
| Wire 2 (pain bridge) | 5.6, gate 14 |
| Wire 4 (`limit` has no route) | O4 re-decided, 5.8 |
| Wire 6 (pulse) | O5 reason corrected, O6, 5.7, the note to pulse |
| Wire 9 (`params`, `goto_pose`, `act`) | 5.5 |
| Wire 10 (harnesses) | §8, 5.9 |
| Wire 11 (motor_backend's reason) | 5.9 |
| Wire 12 (lock during reconnect) | 5.1 abort event, 5.3 |
| Wire table (missed sites) | 5.3, 5.5, 5.6, §6 |

**The v7 delta round → where each finding lives**

| Finding | Home |
|---|---|
| Executor delta S1 (re-check per chunk; fail-open) | R5 re-check, gate 22 |
| Executor delta S2 (the share falls below 0.1 s) | R4 split shares |
| Executor delta S3 (torque-on race) | 5.3, gate 23 |
| Executor delta N1 (no translation hold) | 5.3 `_hold_translation` (body frame) |
| Executor delta N2 (pose matrix; FK tick) | R1 read, F12 |
| Executor delta N3 (REST details; overall deadline) | R1 read, F12 |
| Executor delta N4 (settled) | 5.1 |
| Executor delta N5 (sync lock unbounded) | 5.6, gate 25 |
| Architecture delta S1 / Wire D1 (the ABC is a 1.x surface) | O7, §4, gate 20 |
| Architecture delta S2 (the `accepted` seam) | §4 `MotionTicket` + `current_motion_ticket`, gate 21 |
| Architecture delta S3 (gate 12 vs PR shape) | gate 12 split, §13 |
| Architecture delta N1 (`InvalidConfigError`) | §4, 5.7 |
| Architecture delta N2 (REST endpoint) | R1 read (existing `fetch_url` pattern) |
| Architecture delta N3 (embodiment.md stubs) | §7 Text |
| Architecture delta N4 (`connect` length) | §10 |
| Wire D2 (the pain start is recorded at enqueue) | 5.6 pain bridge |
| Wire D3 (identity checks on `turn_around`) | 5.6, §7, gate 26 |
| Wire D4 (the registry drops unknown keys) | 5.7, §7 |
| Wire D5 (the REST shape varies) | R1 read, gate 24 |
| Wire NITs (`move_antenna`/`act`; test stubs) | 5.6, §7 |

## 12. Choices made in these folds (for the delta round to check)

These were not put to the owner as separate questions. Each is the fail-closed or established-precedent option:
1. **The R5 measure and bounds:** twist about body z; endpoint maximum + 10°, never over 90°; 32 samples.
   - **Measured 2026-10-29** with the SDK's `linear_pose_interpolation`, 4,000 random moves per stage, legal starts and targets:
     - each split stage (body / relative yaw / roll and pitch) shows **0.00°** of excess over its endpoints (path maximum 59° to 60°);
     - F6's 179.7° case shows **135°** of excess (path twist 174.7°), so it is caught with a wide margin.
   - **The twist is a sweep detector, not the neck joint limit.** With roll and pitch non-zero it reads below the commanded relative yaw. The ±65° neck limit stays enforced at the endpoints by the existing clamp.
2. **The split order**: body, then relative yaw, then roll and pitch, as the Executor measured. The whole move is refused if any planned chunk fails.
3. **The fresh-read timeouts:** 100 ms LAN, 300 ms tunnel, measured on the rig before release. `control_mode` must be enabled.
4. **"Settled" after an SDK timeout:** two reads 100 ms apart within 0.5°.
5. **The tool wait:** at most 25 s, on `accepted`. A timeout cancels the ticket.
6. **One API** (`goto_target -> MotionOutcome`, truthy iff ok), following the `SyncResult` precedent.
7. **The lifecycle re-centre:** 2° deadband, 2 s floor, translation held. Canned sleep antennas are exempt from our limit.
8. **The `enable_motors` failure:** no motion, `disable_motors()` best-effort.
9. **Selfy's early returns:** `ok`/0 chunks for a no-op; `limit` for "unreachable".
10. **The per-chunk re-check tolerance:** `_STASH_DIVERGENCE_REFUSE_RAD` (one rig-measured constant, provisionally 5°).
17. **The final-target check:** one fresh read after the last chunk (and after every single dispatch). It is within the same tolerance, or the outcome is `partial`, not ok.
18. **Settled is cleared by a reconnect.**
11. **Settled:** within 2° of the timed-out target, or stable within 0.5° for 1 s.
12. **The torque-on poll:** bounded at 1 s.
13. **The synchronous lock wait:** 25 s, then "motor busy".
14. **The registry broadcasts keep `dict[str, bool]`.**
15. **The REST parser accepts the two known shape variants**; anything else refuses.
16. **A bool `False` from a plugin maps to `kind="error"`.**

## 13. PR shape (revised, Architecture D1)

1. **PR 1, the controller and safety:**
   - R1 to R5, the REST read, look_at and lifecycle;
   - the auto-yaw config error;
   - the one-API `MotionOutcome`;
   - **the motor worker's learner gate** (the success branch only on ok), which makes PR 1 safe to land alone;
   - the §7 rewrites;
   - DECISIONS.md MP1 onward for O1 to O7 (O7 records that the ABC's return type was widened, not replaced).

   PR 1 alone removes the live sweep and cannot poison the learners.
2. **PR 2, outcomes:** `MotorQueue`, the two-stage ticket, the tools, the DN inbox, the pain-bridge event ids and `media_loop.act`. Until it lands, a queued refusal is logged and ignored by the learners, but the model still hears "queued" as success, which is the state on `main` today.
3. **PR 3, the lint extension**, plus the facade follow-up issue.

#1499 closes on PR 3. 1.3.4 does not publish until all three have merged (exit item).
