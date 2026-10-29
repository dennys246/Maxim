# #1499 motion-path model (approach note v5, inherited from #1496): no non-finite, instant, too-fast, wrongly-referenced, or mid-path-unsafe command ever reaches a motor

Session S, 1.3.4 exit item.

**Status: v5, a consolidated rewrite, for a FULL design review.**
- Rounds 1–4 each returned DO-NOT-BUILD.
- Rounds 3 and 4 found blockers that the previous round's *folds* had introduced. That is the divergence trigger, so the owner chose (2026-10-26) a single rewrite on one reference model instead of more override layers.
- Every finding from rounds 1–4 is folded into the body below. The cross-reference at the end maps each finding to its home.

## 1. Owner decisions (all rounds, 2026-10-26)

1. **Scope.** #1496 is on 1.3.4's exit list and ships as **one PR closing the class**.
2. **Durations** lie in [0.1 s, 10 s], inclusive. A non-finite duration is always refused.
3. **The orient affordance's body-yaw delta** is capped at ±90°.
4. **Rate cap: 180°/s *average* per check.** The peak under minimum-jerk is about 340°/s, which the owner accepted knowingly.
5. **Antennas** are limited to ±90° provisionally, pending pulse's hand-measured range (requested 2026-10-26). They always interpolate; the instant `set_target` path goes.
6. **Refusals.** A non-finite value is a typed refusal, never a clamp. A finite value that is out of range is clamped, as today.
7. **Default vs explicit durations.**
   - A default or derived duration *stretches* to meet the rate cap.
   - An explicit LLM or user duration that is too fast is *refused*.
   - The validator itself only refuses.
8. **Grep scope.** `scripts/` is in scope for the raw-SDK grep; the existing live harnesses are allow-listed by name.
9. **Stash divergence.** A head-only move whose stashed body disagrees with the readback is **refused**, and the stash is forgotten.
10. **Turning at the body limit.** When no turn toward the target fits, there is **no turn**: the head does the rest, since body ±160° plus neck ±65° covers the full circle. An explicit LLM `turn_around` that cannot fit gets a typed refusal naming the limit.
11. **Fence exceptions** (named):
    - `tools/reachy.py` (a shared hunk with V);
    - `embodied_runtime/movement.py`;
    - `embodied_runtime/workers.py`;
    - `default_network/`;
    - `spatial/` (the recorder skip);
    - the CI greps.

## 2. SDK and daemon facts the design rests on (reachy-mini 1.8.3, verified)

**F1. `look_at_world` sends the body to 0.** It calls `goto_target(target_head_pose, duration=duration)`, and `goto_target`'s `body_yaw` defaults to `0.0`. Every SDK `look_at_*` therefore drives the body to 0, unchecked. With a duration of 0 or NaN it takes the instant `set_target` path instead.

**F2. A head-only move pins the body at the present readback.** When `body_yaw=None`, `motion/goto.py::GotoMove` sets `target_body_yaw = start_body_yaw`, which is the daemon's **present** body. Head-only IK is solved against that readback, never against our stash.

**F3. Overlapping moves are dropped silently.** `daemon/backend/abstract.py::play_move` returns early when a move is already running (`_try_start_move`), and the client task still reports `completed`.

**F4. The client `goto_target` blocks.** It waits on `wait_for_task_completion(timeout=duration + 1)`. A controller dispatch therefore holds `_motion_lock` for the whole move.

**F5. Body yaw and antennas are joint-space values, not circle values.** They interpolate as `start + (target − start)·s`, so a move from 160° to −160° physically turns 320°.

**F6. `create_head_pose` is SciPy extrinsic "xyz", i.e. Rz·Ry·Rx.** So world = Rz(body)·R_rel, and the controller's atan2 extraction is the matching inverse. `utils/interpolation.py::delta_angle_between_mat_rot` gives the geodesic angle between two rotations.

**F7. No interpolation method is limited.**
- `cartoon` overshoots by 10% and peaks at 5.6× its average speed.
- `ease_in_out` peaks at 2.0×, `minjerk` at 1.875×, and `linear` at 1.0× (but with an unlimited acceleration step).

**F8. The SDK sleep pose puts the antennas at ±3.05 rad.**

**F9. The SDK and daemon declare no antenna limits.** The daemon's IK NaN check depends on its version and is not counted.

## 3. The reference model: one rule underneath every check

**R1. The readback is the reference.**
- Every rate check, every head composition with `body_yaw=None`, and every derived duration uses the **present readback** (`get_current_pose`) as the starting point. This follows from F2 and F3.
- The retained-axes stash (`_last_commanded`) keeps one job only: retaining the axes the caller didn't specify.
- Before dispatching, the controller compares stash and readback on each axis.
  - If |stash − readback| exceeds `_STASH_DIVERGENCE_REFUSE_RAD` (new; the observational `_DIVERGENCE_WARN_RAD` stays untouched), it raises `MotionRefused(kind="reference")` and forgets that axis's stash.
  - So one divergence costs exactly one refusal.
  - The constant's value is measured on the rig, as end-of-move servo lag plus a margin, before release. Provisionally it is 5°.

**R2. An axis with no readable reference is refused.** Every axis that moves, or is composed against, needs a finite readback.
- The `current.get(..., 0.0)` fallbacks go.
- `get_current_pose` drops non-finite keys. It also drops impossible body and head values (|value| > limit + tolerance).
- Antenna readbacks are exempt from the impossible-value drop, because of F8. They are still checked for finiteness.

**R3. Frames.**
- **Body and antennas:** deltas are raw joint differences and are never wrapped (F5).
- **Head:** compared only on rotation matrices, in the body-relative frame. R_rel = Rz(−body_readback)·R_world, a **left** multiply. Δ_head is the geodesic `delta_angle_between_mat_rot(R_rel_start, R_rel_target)`.
- No Euler angle is subtracted anywhere. The seed path (`current["yaw"] - current["body_yaw"]`) and `_warn_commanded_vs_achieved` move to matrices, or to the extraction of R_rel.
- **x, y and z translation** are not rate-checked; only the workspace clamp applies. This is stated as a known gap.

**R4. The rate check.**
- Per dispatch, compute:
  - Δ_body = |target body − readback body|, when a body is commanded;
  - Δ_head (R3);
  - Δ_antenna = |target − readback| for each antenna.
- A move needs a duration of at least max(Δ) / `MAX_RATE`.
- Resolution under `duration_policy` (§4):
  - `"exact"`: refuse if the duration is too short;
  - `"at_least"`: duration = max(requested, max(Δ)/MAX_RATE).
- In both cases the result goes through `require_duration`: above 10 s it is **refused**, never clamped.
- The check runs inside `_goto_target_locked`, under `_motion_lock`, before any stash write. A refused target leaves the stash identical, which a test pins.

## 4. The validator module: `hardware/controller.py`, beside `MotionTarget` (one copy; #1497 reuses it)

**Constants**
- `MOTION_DURATION_MIN_S = 0.1`, `MOTION_DURATION_MAX_S = 10.0`
- `MAX_RATE_DEG_S = 180.0`, together with a recorded peak factor for each method
- `ANTENNA_LIMIT_RAD = radians(90)` (provisional)
- `ALLOWED_METHODS = {"minimum_jerk", "linear", "ease_in_out"}`, compared after normalisation (lower-cased and stripped, with `minjerk` mapped to `minimum_jerk`). `cartoon` is refused in every spelling.

**`class MotionRefused(ValueError)`**
- It carries `kind ∈ {"non_finite", "duration", "rate", "reference", "limit", "method"}` and a reason string.
- Only `"limit"` says anything about the target's location (§6).

**Helper functions**
- **`require_finite(field, value) -> float`**
  - Rejects `bool` and `str`.
  - Converts with `float()` inside a try, mapping TypeError, ValueError and OverflowError to a refusal.
  - Then requires `math.isfinite`.
  - Returns a plain float, so Decimal and numpy values do not survive.
- **`require_duration(value) -> float`**: finite, and within [MIN, MAX] inclusive.
- **`clamp_finite(field, value, lo, hi) -> float`**: refuses a non-finite value, then clamps. It replaces every motion-path clamp idiom (§5.4).
- **`resolve_duration(delta_rad, requested, policy) -> float`**: the R4 resolution. It is the single place a stretch happens. Callers that need the resolved number (sleeps, inhibit windows) get it from the dispatch result (§6), never by re-deriving it.

**`MotionTarget`** (a plain, non-frozen `@dataclass`, never serialised, so CC3 does not apply)
- `__post_init__`:
  - angles and extras go through `require_finite`, with None still meaning "don't change";
  - `duration` goes through `require_duration` (it stays a float, so every existing float consumer keeps working);
  - `method` is checked against `ALLOWED_METHODS`.
- New field: `duration_policy: Literal["exact", "at_least"] = "at_least"`. The default is the fail-safe direction: a caller that forgets it gets a *slower* move, never a faster one.

**`PixelTarget`** (`slots=True`, never persisted) gets the same treatment: u and v must be finite integers, plus `duration` and `duration_policy`.

## 5. Boundaries

**5.1 Entry points: Selfy** (`move`, `move_relative`, `goto_pose`, `turn_around`, `look_at_image`, `move_antenna`, centre/recenter).
- Each validates synchronously, *before mutating any state*, so a refusal never poisons the mirror (`self.x…self.yaw`, `self.duration`).
- `turn_around`:
  - validates both derived steps at entry;
  - skips step 2 if step 1 is refused or fails;
  - resets the `self.yaw = 0.0` mirror only when `ok1 and ok2`.

**5.2 Tools** (`tools/reachy.py`: MaximCommandTool `move`, `turn_around`, `move_antenna`; MoveTool; FocusOnSoundTool; the look-at tools).
- Parameters go through `require_finite`.
- When the model supplies `duration_s` in its kwargs, the tool passes `duration_policy="exact"`; otherwise it passes `"at_least"`. `tools/base.py` never injects schema defaults, so `"duration_s" in kwargs` is a reliable test.
- The schemas drop the advertised `default` values for durations, so a model can't echo one back as an explicit value. The description names the derived behaviour instead.
- The refusal reason goes back to the model.
- The normalised ±1.0 clamps (target_x, target_y, azimuth) become `clamp_finite`.

**5.3 Upstream data sources**
- A component YAML `head_yaw` that is NaN or inf is refused at load.
- `default_poses.json` NaN or Infinity is refused at `goto_pose`.
- An orient delta that is non-finite, a bool, or larger than |radians(90)| is refused in `ReachyOrientMotorBackend.__init__`. That covers both the factory and the exp53 direct build. The factory reports the refusal instead of a silent `continue`.
- `execute` refuses a `current_body` that is non-finite or impossible, and its `min(3.0, max(1.0, …))` duration clamp becomes `require_duration` plus the policy.

**5.4 Clamp sites.** Every motion-path `max(-L, min(L, x))`, `min(L, max(-L, x))` and `np.clip(` (about 27 sites) becomes `clamp_finite`, or `require_duration` for duration clamps.
- A CI grep bans those forms in `embodied_runtime/movement.py`, `motion/`, `hardware/`, `tools/reachy.py` and `motor_backend.py`.
- Pixel-bound clamps such as `max(bounds[...]` are excluded by pattern.

**5.5 Dispatch: `ReachyMiniController._goto_target_locked`.** In order:
1. readback (R2);
2. divergence check (R1);
3. compose (R3);
4. rate check and duration resolution (R4);
5. last-mile check;
6. SDK call;
7. stash write.

The last-mile check: `require_finite` runs on every scalar handed to the SDK (body_yaw, the 16 matrix entries, antennas, duration) immediately before `self._mini.goto_target`.

**5.6 `look_at`, both paths: `Selfy.look_at_image` and `ReachyMiniController.look_at_pixel`** (F1).
- **The compute helper.** One helper in `hardware/reachy/controller.py` calls `self._mini.look_at_image(u, v, perform_movement=False)`, which returns world-frame T_world_head from `get_current_head_pose()`.
- **The dispatch.** Under the same `_motion_lock` hold, it:
  - takes R_rel = Rz(−body_readback)·R_world, using the same readback the SDK used;
  - extracts roll, pitch and relative yaw;
  - dispatches `MotionTarget(roll, pitch, head_yaw=rel_yaw, body_yaw=None, duration, duration_policy)` through 5.5.
- **The result.** The body never moves. The neck is clamped to its relative limit, which is the world-frame rule. Composition is against the readback body (R1/F2), so the stash cannot inject a body mismatch.
- **Selfy's path.** `_enqueue_sdk_look_at` enqueues the controller method, not the raw SDK. Selfy's branch with no controller (`mini` set but `_robot is None`) refuses with `MotionRefused`; there is no raw fallback.
- **Behaviour change, stated:** `look_at` no longer recentres the body. The one implicit dependant, `_maybe_turn_around`, is handled in 5.8.
- `test_look_at_pixel_keeps_body_stash` is rewritten. Its fake SDK models F1, so a regression shows up as a body command.

**5.7 `motion/movement.py` (`move_head`, `move_antenna`).**
- These already compose against the readback, consistent with R1.
- `_current_body_yaw_deg` refuses instead of returning 0.0 or NaN.
- Both functions run the R4 check against the readback, plus the last-mile check.
- `move_antenna`:
  - applies `clamp_finite` to ±`ANTENNA_LIMIT_RAD` on the composed target;
  - always uses interpolated `goto_target`, with the `set_target` branch deleted;
  - restricts `method` to `ALLOWED_METHODS` (it has its own normalisation, which is aligned with the module's);
  - its first move after sleep (from ±3.05 rad) stretches under `"at_least"`.

**5.8 Turning at the body limit** (owner decision 10).
- `_maybe_turn_around` returns False when |body + step| toward the target exceeds the body limit. `look_at` then proceeds, with the head clamped.
- An explicit `turn_around` (the LLM tool) that cannot fit returns a typed `MotionRefused(kind="limit")` naming the limit. It never reverses silently.

**5.9 Other named callers**
- `RobotController.center_vision` (`hardware/registry.py`) sends body 0 under `"at_least"`.
- DN tracking passes its 0.15 s base as `requested` under `"at_least"`, so its cadence holds for small moves and stretches only for large ones.
- `scripts/exp49/run_trials.py`: 0.3 s steps, and the return value is checked.
- `live_1b_controller_smoke.py` checks its return value.

## 6. The refusal channel

- **Internal raise, public bool.** `ReachyMiniController._goto_target_checked` (internal) raises `MotionRefused`. Public `goto_target` wraps it, logs at ERROR and returns False, so every existing bool caller keeps working.
- **Motor-thread callables.** These are `move_head` and the controller-backed `look_at`. They call the raising variant, so a refusal is never confused with a disconnected False.
- **Dispatch result.** The raising variant returns the resolved duration, so `turn_around`'s sleep and `default_network.inhibit(...)` use the real duration.
- **`workers.py::motor_worker`** catches `MotionRefused` *before* the generic handler. It:
  - calls `sync_head_position()`;
  - counts no connection failure (so `gpu_compat.is_connection_error`'s substring check never sees it);
  - for `kind == "limit"` only, records `record_gaze(success=False)` and a bounds outcome;
  - for every other kind (rate, reference, non_finite, duration, method), skips both recorders, because those say nothing about the target's location and must not contract the persisted workspace (`spatial/bounds_learner.py::_contract_bound`).
- **The blind-reset branch** moves its `maxim.yaw`/`pitch = 0` writes after a successful `move_head`.
- **DN's `network.py` handler** catches `MotionRefused` by type, before its substring classification ("ik", "not achievable").

## 7. CI greps

**Raw SDK.**
- Bans any reference on an SDK handle (receiver pattern `\bmini\.`, `_mini.`, `reachy_mini.`, plus `getattr(<handle>, "<name>")`) to:
  - `goto_target`;
  - `look_at_image`, `look_at_world`;
  - `set_target*`;
  - `_goto_joint_positions`, `_set_joint_positions`;
  - `play_move`, `async_play_move`.
- That includes alias assignments.
- Scope is `src/maxim/` **and** `scripts/`.
- The allow-list:
  - the controller's dispatch and compute helper;
  - `motion/movement.py`'s sanctioned primitives;
  - `scripts/orient_backbone/live_common.py`, `scripts/orient_backbone/live_1_smoke.py`, each with a one-line reason.
- Selfy's own `maxim.look_at_image(` method is excluded by receiver name.

**Clamp idiom.** As 5.4.

Both greps get a planted-violation test.

## 8. Stated out of scope

- The daemon listens on `0.0.0.0:8000`, so a LAN client bypasses all of this. That belongs to the embodied-device plan.
- The SDK's own `wake_up` and `goto_sleep` are uncovered, but stay under the cap: 20° in 0.2 s, and at most 160°/s.
- x, y and z translation rates are not checked (R3).

## 9. Red gates (strict xfail, written first; each runs the real path with a fake SDK that models F1–F3)

1. A NaN or inf component `head_yaw`, and a 120° orient delta, are refused at load. NaN never reaches `goto_target`.
2. MaximCommandTool `turn_around(angle="nan")` and `move(yaw="nan")` are refused: no SDK call, and Selfy's mirror is unchanged.
3. `default_poses.json` with NaN, through `goto_pose`, is refused.
4. `MotionTarget` with duration 0, −1, nan or 11, with body_yaw nan, or with `method="cartoon"` (in any case), raises. `PixelTarget(duration=0)` raises.
5. `_clamp(nan)` raises. A NaN body readback refuses the move in both the controller and `move_head`.
6. `move_antenna`:
   - duration 0 is refused;
   - a NaN angle is refused;
   - 200° is clamped to the limit;
   - `set_target` is never called.

   `look_at_image(duration=0)` is refused.
7. Rate:
   - body 160° in 0.5 s, `"exact"` → refused; `"at_least"` → dispatched with a duration of at least 0.89 s;
   - a 30° look-at in 0.3 s passes;
   - a DN 40° track from a 0.15 s base is dispatched with a duration of at least 0.22 s.
8. A refused target leaves the retained-axes stash identical.
9. Planted violations are caught by both CI greps, including an alias and a `getattr`.
10. Body at 1.0 rad; `look_at_pixel` and `look_at_image` → the fake SDK receives no body command. A centre pixel gives a commanded relative yaw within 1° of the current relative yaw.
11. **Joint space (C-1):**
    - body 160° → −160° in 0.5 s, `"exact"` → refused;
    - the same under `"at_least"` → duration of at least 1.78 s;
    - antennas −3.05 → +3.05 rad stretch accordingly.
12. **Divergence (R1):** stash body 150°, readback 0 → `look_at_pixel` and a head-only goto are both refused with `kind="reference"`, the stash body is forgotten, and the next move succeeds.
13. **No reference (R2):** a readback missing `roll` → a move that changes roll is refused, never composed against 0.
14. **Wrap (R3):** world yaw 170°, body 160°, target world 175° → no refusal, and a relative command near 15°.
15. `turn_around` with step 1 refused → step 2 is never dispatched, and `self.yaw` is unchanged.
16. A `MotionRefused(kind="rate")` raised on the motor thread, through both `move_head` and `look_at`:
    - the mirror equals the readback;
    - no connection failure is counted;
    - neither `record_gaze` nor the bounds learner is called.

    With `kind="limit"`, both are called.
17. Turning at the limit: body 150°, a turn of +45° toward the target → `_maybe_turn_around` returns False and no body command is sent. The tool's `turn_around` at the same pose returns `kind="limit"`.
18. Selfy with no controller: `look_at_image` is refused, and no raw SDK call is made.

## 10. Cross-reference: rounds 1–4 → where each finding lives

| Round | Findings | Home |
|---|---|---|
| 1 (v1) | D1–D4, S1–S7 | §4, §5.1–5.4, §5.7 |
| 2 (v2) | D1 look_at body | §5.6, F1 |
| 2 (v2) | S1 ref | R1 |
| 2 (v2) | S2 no ref | R2 |
| 2 (v2) | S3 frames | R3 |
| 2 (v2) | S4 method | §4 |
| 2 (v2) | S5 callers | §5.1, §5.2, §5.9 |
| 2 (v2) | S6 mirror | §6 |
| 2 (v2) | N1 grep forms | §5.4 |
| 2 (v2) | N2 antennas | R2, §5.7 |
| 2 (v2) | N3, N4 | §8 |
| 2 (v2) | N5 scripts | §7 |
| 3 (v3) | A stash body | R1 |
| 3 (v3) | B conversion | §5.6 |
| 3 (v3) | C wrap | R3 |
| 3 (v3) | D duration placement | §4 `duration_policy` |
| 3 (v3) | E refusal visibility | §6 |
| 3 (v3) | F grep, raw fallback | §5.6, §7 |
| 3 (v3) | K turn | §5.8 |
| 3 (v3) | G, H, I, J | §4, R4, §5.9, §5.6 |
| 4 (v4) | C-1 joint-space wrap | R3 + gate 11 |
| 4 (v4) | A-1 compose vs readback | R1 + §5.6 |
| 4 (v4) | A-2 separate constant | R1 |
| 4 (v4) | B-1 left multiply | R3 |
| 4 (v4) | D-1 None breaks consumers | float + policy, §4 |
| 4 (v4) | D-2 floor lost | `requested` kept, R4 |
| 4 (v4) | E-1 recorder poisoning | §6 kinds |
| 4 (v4) | E-2 raising variant | §6 |
| 4 (v4) | E-3 blind reset | §6 |
| 4 (v4) | K-1, K-2, K-3 | §5.8 |
| 4 (v4) | F-1 getattr | §7 |
| 4 (v4) | G-1 normalisation | §4, §5.7 |

## 11. Round 5 (full review of v5): DO-NOT-BUILD. Owed into v6 before any build.

On 2026-10-26 the owner split #1496 here: the non-finite/duration/clamp class ships as #1496 (`motion_finite_1496.md`), and this note became #1499.

- **D1 (fails open). Mid-move neck sweep.**
  - The head interpolates on the world SO(3) geodesic (short way round) while the body interpolates in joint space, so whenever |Δbody + Δrel_yaw| > 180° the neck-relative yaw sweeps through ±180° mid-path.
  - Reproduced with the SDK's `linear_pose_interpolation`: 160→−160 with the head riding along → 180°; center_vision from body 160°, rel 30° → 178.5°.
  - Fix: a path rule (|Δbody + Δrel_yaw| ≤ 180° − margin, else segment under `at_least` or refuse under `exact`); recentre through the controller before SDK `wake_up`/`goto_sleep`; rewrite gate 11; add a gate that samples the real SDK path.
- **S1 (fails open). `automatic_body_yaw`.** It is daemon-global. `scripts/reachy_streamer.py`, and `audio_localization.make_reachy_doa_reader` when `mini=None`, construct `ReachyMini()` with the SDK default. Refuse dispatch while it is on, grep for it, and state the single-client assumption.
- **S2 (fails open, narrow). Lock-free movers.**
  - The IK-recovery `move_head` in `motor_worker`, `motion/movement.py::move_antenna`, and controller `wake_up`/`goto_sleep` run without `_motion_lock`.
  - Fix: make `motion_lock` a required keyword argument.
- **S3. Refusals recorded as success.**
  - MaximCommandTool's queued commands return success at enqueue.
  - `Selfy.turn_around` and `goto_pose` swallow refusals.
  - DN `_dispatch_action_to_motor` sets `executed=True` at enqueue, and `_record_movement_result` records success.
  - The public bool `goto_target` loses the reason for MoveTool and FocusOnSound.
- **S4. The explicit-duration key.** It is `duration` in MoveTool, FocusOnSound and MaximCommandTool (`turn_around` hard-codes 5.0), and `duration_s` only in Track and NoveltyTrack. Selfy needs a `duration_policy` parameter to carry it through.
- **S5. Derived durations under 0.1 s are refused before any stretch** (`turn_around` step 1 = min(2, 0.3·d)). Stretch derived values before constructing the target.
- **S6. Narrow R1 and R2.**
  - Divergence is checked only on *retained* axes, compared through R_rel (plus body, per decision 9).
  - The impossible-value drop never applies to world yaw.
- **S7. Guard tests and invariant text the design inverts:**
  - `test_reachy_retained_axes.py`: `test_head_only_commands_use_last_commanded_body_for_composition`, `test_body_stash_unblocks_unreadable_pose`, `test_stash_survives_unreadable_pose_after_first_command`;
  - `test_reachy_workspace_safety.py::test_body_read_failure_falls_back_to_zero`;
  - the embodiment.md §1 / CLAUDE.md "last COMMANDED body" text.
- **S8. `ease_in_out` peaks at 2.0×, which is 360°/s, above the accepted 340°/s.** Either apply the peak factor in R4, or ask the owner.
- **S9. Grep gaps:** `wake_up`, `goto_sleep`, `enable_wobbling`, `set_automatic_body_yaw`, `cancel_move`, `ReachyMini(`, and handle aliases (`m = self._mini`). A structural alternative: `Selfy.mini` returns a facade with no motion methods.
- **NITs:**
  - gate 11's antennas contradict the ±90° clamp;
  - gate 16's `limit` kind can't happen on the motor thread;
  - §5.6 should say "one snapshot under the lock";
  - gate 18 is unreachable (`Selfy.mini` is None when `_robot` is None);
  - `np.bool_` gets past the bool check;
  - `_maybe_turn_around`'s body source is unspecified;
  - exp49's step change must be recorded in that experiment's provenance;
  - FocusOnSound has a diagnostic Euler subtraction.
