# #1496 approach note (PR 1 of the split): no non-finite, out-of-range-duration, or instant command ever reaches a motor

Session S, 1.3.4 exit item.

**Status.** Round 1 (short pass, BUILD-WITH-FOLDS) is folded in place; the delta round's site list and the bodiless decision are folded; the site-list round's findings are folded too (decision 8 rebuilt via `validate_orient_deltas` in body wiring); the decision-8 delta round is folded (instantiate on the shared registry, exact placement, `MotionRefused` moved to `maxim/exceptions.py`, a tri-state sync). The last delta round is folded (placement structure, the present-pose latch, sync caller edits). Latch round: DO-NOT-BUILD on the latch as a hardware latch (the WS transport drops non-finite poses). The owner revised decision 9 to defence in depth, with the stale cache going to #1499; the latch-scoped folds are applied. The final §4.9 check returned BUILD-WITH-FOLDS with no further round owed; its folds are applied. **Design cleared for build (2026-10-27).** Scoped after the owner's split (2026-10-26).
- Five design rounds on the combined note found a deeper, pre-existing class: frames, references, rates and the mid-move neck sweep. That class moved to **#1499** (`motion_path_model.md`), which is also HIGH and on the 1.3.4 exit list.
- This note keeps only the class #1496 was filed for. Its design has drawn no blocker since round 1's folds.
- It gets one short adversarial pass before build.

## 1. Owner decisions that apply here (2026-10-26)

1. **Durations** lie in [0.1 s, 10 s], inclusive. A non-finite duration is always refused.
2. **The orient affordance's body-yaw delta** is capped at ±90°.
3. **Antennas** are limited to ±90° provisionally, pending pulse's hand-measured range. They always interpolate: the instant `set_target` path goes.
4. **Refusals.** A non-finite value is a typed refusal, never a clamp. A finite value that is out of range is clamped, as today.
5. **Derived and default durations** are lifted to the floor, never refused, because they are not the caller's explicit choice. An explicit duration outside the range is refused.
6. **`scripts/`** is in scope for the raw-SDK grep; the existing live harnesses are allow-listed by name.
7. **Fence exceptions:**
   - `tools/reachy.py` (a shared hunk with V);
   - `embodied_runtime/movement.py`;
   - `embodied_runtime/workers.py`;
   - `default_network/`;
   - `embodiment/spec.py` (the `head_yaw` load check);
   - `embodied_runtime/selfy.py` (the centred-pose parse);
   - `embodied_runtime/agentic_runtime.py::_resolve_body_wiring`, a shared hunk with V (the bodiless catch);
   - the CI greps.
8. **A refused orient delta means bodiless plus ERROR** (see §4.3).
9. **A non-finite present pose is refused at the last mile as defence in depth.** It cannot fire over the WS transport, which drops non-finite messages. The hardware stale-cache signal is #1499's (revised 2026-10-27, §4.9).

The rate cap, the `look_at` body frame, the readback-as-reference model, the refusal-kind recorder split and the turn-at-limit policy are **#1499's, not this PR's.**

## 2. The class

There are about 27 motion-path clamps of the form `max(-L, min(L, x))`, and each one turns NaN into +L. Several sit upstream of any check:
- LLM tool parameters, via `float("nan")`: MaximCommandTool `move`/`turn_around`/`move_antenna`, MoveTool, FocusOnSoundTool;
- `json.load` of `default_poses.json`, which accepts NaN and Infinity;
- a component YAML's `head_yaw` (a 160° swing in 1 s);
- `default_actions.json` (NaN-capable through `json.load`, reached via `media_loop.act` → `move()`, so `move()`'s validation covers it).

**Durations.** They are unbounded, and the SDK refuses only `<= 0`, so NaN passes. Its `look_at_world` takes the instant `set_target` path when the duration is 0 or NaN. Antennas take an instant `set_target` when the duration is 0 or None.

**Readbacks.** A NaN readback flows into the composed target after the clamps (`_goto_target_locked`'s `target_body`, and `motion/movement.py::_current_body_yaw_deg`).

**Interpolation.** `cartoon` overshoots the target by 10%: a clamped ±160° body target lands at about 176°.

## 3. Validator module: `hardware/controller.py`, beside `MotionTarget` (one copy; #1499 and #1497 reuse it)

**Constants**
- `MOTION_DURATION_MIN_S = 0.1`, `MOTION_DURATION_MAX_S = 10.0`
- `ANTENNA_LIMIT_RAD = radians(90)` (provisional)
- **Interpolation methods.** One canonical-to-SDK map is used at *every* SDK site: the controller dispatch and `motion/movement.py::move_antenna`. It reuses `ReachyMiniController._sdk_method`.
  - Inputs are normalised: lower-cased, stripped, `-` turned into `_`.
  - The aliases are kept with today's meanings: `minjerk`, `min_jerk` and `minimum_jerk` map to SDK `"minjerk"`; `ease` and `ease_in_out` map to `"ease_in_out"` (as `move_antenna` maps them today); `linear` maps to itself.
  - Anything else is refused with `kind="method"`, including `cartoon` in every spelling, because it overshoots any clamp.
  - `"cartoon"` is also removed from `_METHOD_MAP`, **and** `ReachyMiniController._sdk_method`'s `.get(method, "minjerk")` default is removed, so an unknown method raises `MotionRefused(kind="method")` instead of becoming minjerk silently. Today the controller maps `ease` to minjerk; under the one map it becomes `ease_in_out`, as `move_antenna` already does. That is a stated behaviour change on the controller path only.
  - The SDK never receives `minimum_jerk`, which its `InterpolationTechnique` rejects.

**`class MotionRefused(ValueError)`**
- It carries `kind ∈ {"non_finite", "duration", "method"}` and a reason. #1499 will add more kinds.
- It subclasses `ValueError`, and that is a **hazard**, not a convenience.
- **The site rule: validation never sits inside a `try` whose handler substitutes a default, a fallback motion, or a success.** It runs on the raw value *before* that `try`, and the default applies only when the key or value is *absent*. Re-raising inside such a handler is not enough: a plain `ValueError` from `float("abc")` would still take the default.
- Each site below is named. Each gets a red gate (§7, gate 13) or an existing gate as noted.

| Site | Today | Required |
|---|---|---|
| `tools/reachy.py::MaximCommandTool.execute`, turn_around **angle** | `except (TypeError, ValueError): angle = 90.0` | Validate the raw parameter outside the try. Default only when absent. Garbage → tool error. |
| same, turn_around **duration** | `except (TypeError, ValueError): duration = 5.0` | Same. A present but out-of-range value → `require_duration` refuses (decision 5). |
| `embodied_runtime/movement.py::goto_pose`, pose parse | `except Exception: vals = None` → `_default_head_pose` | Validate outside the try. A refusal → error, no motion (gate 3). |
| same, second handler | `try: self.move(...) except Exception: warn(...)` then `time.sleep(float(duration))` | The duration is validated **before the `self.move` try**. Re-raise `MotionRefused`. No sleep after a refusal. |
| `motion/movement.py::_current_body_yaw_deg` | `except Exception: pass; return 0.0` | The finite check sits outside the try. Non-finite → raise, never 0.0, because 0.0 would drive the neck past ±65° while the body is turned (gates 5, 10). |
| `embodied_runtime/movement.py::turn_around`, body readback | `except Exception: current_body_yaw = None` → `self.body_yaw` mirror | A non-finite readback refuses the turn. Only a *missing* readback keeps today's fallback (#1499 owns that). |
| same, STEP 1/STEP 2 handlers | `except Exception as e1/e2` → log; turn returns normally | Re-raise `MotionRefused`, so step 2 is skipped and the tool reports the refusal (gate 7). |
| same, outer handler | `except Exception` | Re-raise `MotionRefused`. |
| `hardware/reachy/motor_backend.py::make_reachy_orient_factory` | `except (TypeError, ValueError): continue` → stub semantics | **Delete the except.** `_affordance_schema` already converts `self_effect` with `float()`, and `validate_orient_deltas` covers the rest. |
| `motion/movement.py::load_poses` | `except Exception: vec = None` (drops the pose, so `goto_pose` falls back to the default pose and the tool reports success) | Keep unparseable entries as an **invalid sentinel**. `goto_pose` refuses a known-but-invalid name with an error and no motion. NaN and Infinity survive the load and are refused at `goto_pose`. A short list (`len < 6`, checked outside the try) is still dropped; that name is then unknown and refused by the unknown-name row. |
| `embodied_runtime/selfy.py::Selfy.__init__`, centred-pose parse (a second reader of `default_poses.json`) | `except (TypeError, ValueError, IndexError): log_swallowed_exception`; it seeds `self.x…self.yaw` and `_default_head_pose` with no finite check, and can half-assign | `require_finite` on every component **before any assignment**. The `load_poses` invalid sentinel is recognised explicitly (not by the `isinstance(list)` gate). A non-finite or invalid centred pose → ERROR, and the existing 0.01 seed values are kept. `goto_pose("centered")` then refuses for that run (stated, intended), while `_default_head_pose` stays finite. |
| `embodied_runtime/workers.py::motor_worker`, IK-recovery blind reset | When `sync_head_position()` is False → blind `move_head(0…, 0.5)` from an unknown pose | Branch on the tri-state (§4.9): on `SyncResult.NON_FINITE`, **skip the blind reset**, log ERROR, and reset `ik_failure_count`/`last_ik_reset_time` so it does not retry every move. `FAILED` keeps today's reset. |
| `_start_agentic_runtime` motor-binding try | `except Exception: warning` → `_motor_factory=None`, stub body | No validation runs inside it (the helper runs in body wiring, §4.3). |
| `tools/reachy.py::TrackTargetTool.execute` | `float(kwargs.get("duration_s", 0.3) or 0.3)`: a present 0 becomes 0.3 | Decision 5: absent **or `None`** → floor; any other present value → `require_duration`. Same rule for `novelty_track`. |
| `embodied_runtime/movement.py::goto_pose`, unknown name | An LLM-chosen unknown name moves to the default pose and reports success | Strip the name (as `load_poses` does). An unknown name then returns an error naming the known poses, with no motion. No caller relies on the fallback: `"centered"` is always set by `load_poses`. |
| `tools/reachy.py::FocusOnSoundTool.execute`, body-frame correction | `except` → debug log | Keep `require_finite` out of it. The NaN reaches the envelope clamp, where `clamp_finite` refuses it. |
| `tools/reachy.py` focus_interests `look_at_fn` handler | warn; the tool reports success | Re-raise `MotionRefused`, and the tool reports an error. |

- **Verified fine (they fail closed already):**
  - MaximCommandTool's outer and per-command handlers;
  - the outer handlers of MoveTool, track_target and novelty_track;
  - `ReachyMiniController.goto_target`/`look_at_pixel` (False plus ERROR);
  - `ReachyOrientMotorBackend.execute` (`success=False`);
  - `motor_worker`'s handlers;
  - the sync paths;
  - DN `_dispatch_action_to_motor`/`gaze_manager.look_opposite` (caught by type, §4.10);
  - `input_handlers._run_action_spec`.

**Helper functions**
- **`require_finite(field, value) -> float`**
  - Rejects `bool`, `np.bool_` and `str`. The `str` rejection is for internal entries only. At the tool boundary (§4.2), LLM strings such as `"30"` are converted with `float()` and then checked with `isfinite`, which still rejects `"nan"` and `"inf"`.
  - Converts with `float()` inside a try, mapping TypeError, ValueError and OverflowError to `kind="non_finite"`.
  - Requires `math.isfinite`.
  - Returns a plain float.
- **`require_duration(value) -> float`**: finite, and within [MIN, MAX] inclusive; otherwise `kind="duration"`.
- **`floor_duration(value) -> float`**: for derived or default durations only. It refuses a non-finite value, then returns `max(MIN, min(MAX, value))`. A derived value is lifted, never refused (decision 5).
- **`clamp_finite(field, value, lo, hi) -> float`**: refuses a non-finite value, or a non-finite `lo`/`hi`, then clamps.
  - It covers *command* clamps only. *Measurement* clamps instead discard a non-finite value to None and never refuse a move, for example `motor_backend.py::_frame_corrected_before`, a DoA credit measurement. They are listed by name in the grep's exemption list.
  - The `if abs(v) > L` clamps (`hardware/reachy/controller.py::_clamp`, and the sync clamps) also become `clamp_finite`, because NaN passes through `abs(v) > L` unchanged.

**`MotionTarget`** (plain `@dataclass`, never serialised) and **`PixelTarget`** (`slots=True`, never persisted)
- `__post_init__`:
  - angles and extras go through `require_finite`, with None still meaning "don't change";
  - `duration` goes through `require_duration`;
  - `method` is checked against `ALLOWED_METHODS`;
  - PixelTarget's u and v must be finite integers.

## 4. Boundaries

1. **Selfy entry points:** `move`, `move_relative`, `goto_pose`, `turn_around`, `look_at_image`, `move_antenna`, centre/recenter.
   - Each validates synchronously, *before mutating any state*, so a refusal never poisons the mirror (`self.x…self.yaw`, `self.duration`).
   - `turn_around` runs `require_finite(angle)` and `require_duration(duration)` **first**, before `_clear_motor_queue()` and `default_network.inhibit(...)`. A NaN `duration` is laundered by `min(2.0, nan*0.3)` = 2.0 and `max(3.0, nan)` = 3.0, so it must be refused before that arithmetic. After that, the derived step durations go through `floor_duration`.
   - If step 1 returns False or raises, step 2 is skipped, and `self.yaw = 0.0` is reset only on `ok1 and ok2`.
2. **Tools** (`tools/reachy.py`):
   - Parameters are converted with `float()` and checked with `isfinite`, then go through `require_duration` (explicit) or `floor_duration` (tool defaults).
   - Today's substitutions are stated: MoveTool's `float(duration) if duration else 1.0` and FocusOnSoundTool's `<= 0 → 1.0`. Under decision 5:
     - a value that is *absent* is a default and gets floored;
     - a value that is *present* is explicit and gets `require_duration`, so a present 0 is refused, not swapped for 1.0.
     - `goto_pose`'s JSON `pose[6]` is data, so it gets `require_duration`.
   - The refusal reason is returned to the model as a tool error, not as a success.
   - The normalised ±1.0 clamps (target_x/target_y, azimuth) become `clamp_finite`.
3. **Upstream data:**
   - A component YAML `head_yaw` that is non-finite is refused at load.
   - A non-finite value in `default_poses.json` is refused at `goto_pose`.
   - `ReachyOrientMotorBackend.__init__` refuses a delta that is non-finite, a bool, or larger than |radians(90)| (covering both the factory and the exp53 direct build).
   - **Owner decision 8 (2026-10-26): bodiless plus ERROR.** The factory lets `MotionRefused` propagate. `agentic_runtime.py::_resolve_body_wiring` catches it under its documented rule ("a declared but unusable body does NOT crash the live robot: fall back to bodiless"), logs one ERROR naming the affordance and the value, and the robot runs bodiless: no orient tools, but the agentic runtime still starts.
   - It never returns None into stub semantics, which would apply the NaN `self_effect` to the sensors.
   - **How it is built (site-list round F1; Session S chose this form while folding, and it is listed for the delta round):**
     - The factory is lazy. Both refusals (the spec.py parse and the backend `__init__`) would otherwise fire inside `build_executor`, after body wiring has returned.
     - So **one shared pure helper**, `hardware/reachy/motor_backend.py::validate_orient_deltas(entity) -> dict[str, float]`, takes the **parsed `Entity`**.
       - It walks the entity exactly as `embodiment/spec.py::attach_backends` does: `entity.walk()`, then each entity's `modulators["orient"]`, then `SpecModulator._affordances[*].self_effect["head_yaw"]`. A narrower walk would let a nested orient escape into `build_executor`.
       - It applies `require_finite` plus the |radians(90)| cap.
       - It is the only copy of the cap: `ReachyOrientMotorBackend.__init__` uses the same constants and predicate.
     - **Structure in `agentic_runtime.py::_resolve_body_wiring`.** Today `return body_ref, registry` sits *inside* the `"Failed to build ComponentRegistry"` try. The build restructures it:
       1. The existing try/except keeps registry construction and the `has()` check, unchanged. It no longer contains the return.
       2. **After** that try, a new block:
          ```python
          try:
              entity = registry.instantiate(body_ref)
              validate_orient_deltas(entity)
          except MotionRefused:
              # one ERROR with the affordance and value
              return None, None
          ```
          - It calls `registry.instantiate(body_ref)` on the same `registry` instance that is returned to `build_executor`, so both read one cached spec.
          - It catches `MotionRefused` **only** → one ERROR naming the affordance and value → `return None, None` (the existing bodiless contract).
          - The imports of `validate_orient_deltas` and `MotionRefused` (from `maxim.exceptions`, its canonical home) come **before** the new `try`. An ImportError there now crashes startup instead of falling back to bodiless; that is acceptable, because it is a code bug, not body config. If they sat inside it and failed, evaluating `except MotionRefused` would raise `NameError`.
       3. `return body_ref, registry`, outside both.
       - Any other parse error propagates, as it does today from `build_executor`. That is not a regression.
       - Calling `instantiate` twice is safe: the spec is cached, the copy is a deep copy, `renote` de-duplicates, and the throwaway entity has no identity coupling.
       - A test pins that `spec.py` has no `except ValueError` that would swallow `MotionRefused` before it reaches this catch.
     - The `head_yaw` finite check also lives in `embodiment/spec.py::_affordance_schema`. That function already raises `ValueError` for `speech_param`, so every other spec consumer handles a raise from it.
       - It raises `MotionRefused`, which `instantiate` above surfaces into the same catch.
       - `MotionRefused` moves to `maxim/exceptions.py` (re-exported from `hardware/controller.py`) so that `embodiment/` does not import from `hardware/`.
     - `_start_agentic_runtime` is not edited.
   - **Gating, stated:** body wiring runs before the motor-binding gates, so a bad delta makes sims, headless runs and the Exp 49 `motor_binding: false` arm bodiless too. This is intended: a bad delta is a config error wherever it is found, and the outcome fails closed.
   - **The motor-binding try** in `_start_agentic_runtime` (`except Exception: log.warning("SEM motor binding unavailable")`, after which `_motor_factory` is None and the body gets stub semantics) never runs validation. That is the site rule; it holds because the helper ran earlier in body wiring.
   - Nothing reaches `media_loop.live`'s `warn_unless_refusal`.
   - `execute`'s `min(3.0, max(1.0, …))` becomes `floor_duration`-style finite-checked bounds.
4. **Clamp sites.** Every motion-path `max(-L, min(L, x))`, `min(L, max(-L, x))` and `np.clip(` becomes `clamp_finite`, or `floor_duration`/`require_duration` for duration clamps.
   - A CI grep bans those forms in `embodied_runtime/movement.py`, `motion/`, `hardware/reachy/`, `hardware/controller.py`, `tools/reachy.py` and `motor_backend.py`.
   - `hardware/simulation/` is excluded (simulated streams only).
5. **Readback.** In `_goto_target_locked` and `motion/movement.py`:
   - any readback value *used* for composition passes `require_finite`;
   - a non-finite value refuses the move (`kind="non_finite"`) instead of composing;
   - `_current_body_yaw_deg` raises instead of returning NaN.

   Today's fall-back-to-0 for a *missing* readback is unchanged here; #1499 owns it.
6. **The last mile.**
   - `require_finite` runs on every scalar handed to the SDK: body_yaw, the 16 head-matrix entries, and the antennas.
   - **`require_duration`** (range as well as finiteness) runs on the duration.
   - The method goes through the canonical map.
   - In `look_at_pixel` the check runs before `_forget_commanded`, so a refused PixelTarget leaves the stash identical.
   - This happens immediately before each SDK motion call:
     - `ReachyMiniController._goto_target_locked`;
     - `ReachyMiniController.look_at_pixel`;
     - `embodied_runtime/movement.py::_enqueue_sdk_look_at`'s callable;
     - `motion/movement.py::move_head` and `move_antenna`.
   - Reason: `MotionTarget` is a mutable dataclass, so a post-construction write such as `target.duration = 0.01` bypasses `__post_init__`; and `move_head`/`move_antenna` take raw values from any caller.
7. **Antennas** (`motion/movement.py::move_antenna` and the controller antenna path):
   - `clamp_finite` to ±`ANTENNA_LIMIT_RAD`;
   - the duration goes through `floor_duration` when defaulted and `require_duration` when explicit;
   - the `set_target` branch is deleted, so antennas always use interpolated `goto_target`, which **keeps `body_yaw=None`** (the SDK default is 0.0, which would turn the body to 0);
   - `method` goes through the canonical map;
   - there is no controller antenna path: `ReachyMiniController._goto_target_locked` ignores `extras`, and none is added here.
8. **`look_at` durations.** `require_duration` runs synchronously in `Selfy.look_at_image` and `ReachyMiniController.look_at_pixel` (`floor_duration` for defaults), so the SDK never takes its instant `set_target` branch.
9. **The motor thread.** `workers.py::motor_worker` catches `MotionRefused` before its generic handler. It:
   - calls `sync_head_position()`. Sync (`embodied_runtime/movement.py::sync_head_position` and `_sync_head_position_from_controller`) passes **every** readback value through `require_finite` **before any arithmetic and before any mirror write**. The SDK path writes `self.body_yaw` before it reads the head matrix, so all reads come first. Otherwise a NaN slips past the `if abs(x) > LIMIT` clamps into the mirror, and an infinite body hangs the `while yaw_deg < -180` normalising loop. 
   - **The sync contract is a tri-state; sync never raises for this.** It returns a `SyncResult` with three values:
     - `OK`, which is truthy;
     - `FAILED` and `NON_FINITE`, both falsy (`__bool__`).
   - On `NON_FINITE` it keeps the mirror and logs ERROR once.
   - Existing bool callers treat `NON_FINITE` exactly like a failure, so they need no change:
     - `ReachyOrientMotorBackend.execute` post-motion;
     - the DN initial and periodic sync;
     - `look_at_image`;
     - `turn_around` STEP 3.
   - Callers that ignore the result also include the worker's two bounds-learner records (an IK failure and a success). On `NON_FINITE` they record the stale but finite mirror; `_get_workspace_limits` applies learned bounds with `min()`, so they can only tighten. This is stated as acceptable.
   - The motor worker alone branches on it (§3 table), with `synced is SyncResult.NON_FINITE` (identity, so a plain `bool` from a test stub falls through to the FAILED/OK paths). `SyncResult` is a plain `Enum` with `__bool__`, not an `IntEnum`.
   - "Logs ERROR once" means once per **episode**: logged when a NON_FINITE sync follows an OK one, and again on each recurrence.
   - The annotations change from `-> bool` to `-> SyncResult` on both functions. Five test assertions that use `is True`/`is False` move to truthiness or `is SyncResult.OK`: two in `tests/unit/test_exp49_harness_seams.py` and three in `tests/unit/test_body_yaw_joint_index.py`.
   - Because sync never raises, the `except MotionRefused` handler cannot raise from inside itself and kill the motor thread.
   - The handler counts no connection failure (gate 10) and logs at ERROR (gate 10 also asserts the ERROR record).
   - The blind-reset branch moves its mirror zeroing after a successful `move_head`.
   - **The present-pose finiteness check (owner decision 9, revised 2026-10-27): defence in depth, not a hardware latch.**
     - **The fact it rests on (verified on reachy-mini 1.8.3).** The daemon publishes poses with `model_dump_json()`, which serialises NaN and Inf as `null`. `WSClient._recv_loop` fails validation on a `null` and `continue`s, dropping the message silently. So **over the WS transport the client only ever caches the last *finite* pose.**
     - A finiteness check on the readback therefore **cannot fire on the real robot**. The same is true of §4.5's non-finite readback refusal and the `SyncResult.NON_FINITE` branch. The note and the PR say so plainly.
     - **Why it ships anyway.** It guards fakes, sims and any future transport that passes non-finite values through, and it costs one read.
     - **The rule.** Every move is refused with `MotionRefused(kind="non_finite")` when a *returned* present pose contains a non-finite value.
       - An exception, a `None`, or a non-numeric object (such as a `MagicMock`) counts as **missing**, and falls through to today's behaviour unchanged (owned by #1499). Examples: `test_body_read_failure_falls_back_to_zero`, `_NoJointsMini`, the `MagicMock` mini in `test_media_loop.py`.
       - On a partial read, each component is judged on its own.
     - **One shared helper: `hardware/controller.py::require_present_pose_finite(mini) -> None`**, beside the §3 validators. It raises `MotionRefused(kind="non_finite")`.
       - It reads `mini.get_current_head_pose()`, the raw 4×4 matrix with all 16 entries.
       - It reads `mini.get_current_joint_positions()`, which gives `(head[7], antennas[2])`; the antennas come from this same call.
       - Each component is judged on its own:
         - a read that raises → missing;
         - a `None` return → missing;
         - a conversion that raises TypeError or ValueError → missing;
         - the wrong shape (head not (4, 4), or an empty joint vector) → missing;
         - a `None` *inside* a returned matrix or vector → **refused** (fail-closed);
         - any other non-finite entry → refused.
       - "Missing" never refuses.
       - The catch uses the Stage-1 handle-and-log form (`log_swallowed_exception`), as `lint_no_silent_swallows` requires.
       - A `MagicMock` converts to an empty array or to 1.0, so the shape rule is what classifies it as missing, not finiteness. It **must not** reuse `ReachyMiniController.get_current_pose()`, which:
       - runs only conditionally;
       - collapses Inf to finite values through `atan2`;
       - swallows joint errors;
       - writes `_update_state`.
     - **Call sites**, each at motor-thread dispatch time immediately before the SDK call:
       - `ReachyMiniController._goto_target_locked` and `look_at_pixel` (before `_forget_commanded`, so a refusal leaves the stash identical);
       - `motion/movement.py::move_head` and `move_antenna`;
       - **both** branches of `embodied_runtime/movement.py::_enqueue_sdk_look_at`, inside the enqueued callable, whenever `perform_movement` is true;
       - `scripts/orient_backbone/live_common.py::_goto` and `live_1_smoke.py`.
     - **Handed to #1499 (owner, 2026-10-27).** On hardware, the real signal is a **stale** pose cache, when messages stop arriving. That needs the client's receipt state, and it belongs with #1499's missing- and stale-readback reference model.
     - **Our own process cannot put NaN into the daemon after this PR.** The last mile refuses it, and the daemon's head IK refuses NaN joints. What remains comes only from another client, which is out of scope (the embodied-device plan).
     - **Stated residuals (#1499):**
       - `wake_up`/`goto_sleep` (in `controller.py`, `selfy.py`, `connection.py`, `media_loop.py`, and `agentic_runtime`'s `mini.wake_up`) start from the present pose without this check;
       - the cache can be up to about 20 ms old (50 Hz), which leaves a window against the daemon's own read;
       - lock-free movers.
10. **DN.**
    - The handlers in `network.py` and `gaze_manager.py` catch `MotionRefused` by type, before their substring classification.
    - DN-derived durations (`network.py::_dispatch_action_to_motor`, `gaze_manager.py::look_opposite`) are floored in DN through `floor_duration`, because they are defaults derived from config, not explicit choices.

**Contracts**
- Public `goto_target` keeps returning False on a refusal, with an ERROR log.
- A refused target leaves `_last_commanded` untouched, and a test pins that.

## 5. CI grep (raw SDK)

The grep also covers `set_target`, `set_target_*`, `_goto_joint_positions`, `_set_joint_positions`, `look_at_world` and `play_move`/`async_play_move`.
- It matches attribute references on an SDK handle (`mini.`, `_mini.`, `reachy_mini.`), aliases, and `getattr(<handle>, "<name>")`.
- Its scope is `src/maxim/` **and** `scripts/`.
- The allow-list is: the controller dispatch, `motion/movement.py`'s sanctioned primitives, and `scripts/orient_backbone/live_common.py` plus `live_1_smoke.py`, each with a one-line reason.
- `look_at_image` stays allowed in this PR. #1499 reroutes it.
- The allow-list includes the existing `sdk_look_at = self.mini.look_at_image` line in `embodied_runtime/movement.py`.
- The lint scripts' own pattern literals are excluded from the `scripts/` scan.
- Stated residual: handle-name matching (`mini.`) misses other variable names, which #1499 closes structurally.
- The allow-listed `scripts/orient_backbone/live_common.py::_goto` and `live_1_smoke.py` call `require_finite`, `require_duration` and the method map before their raw dispatch, so the note's "ever" holds for them too.

## 6. Stated out of scope (owned by #1499)

- The rate cap.
- The mid-move neck sweep.
- The `look_at` body-to-0.
- Readback-as-reference and stash divergence.
- `automatic_body_yaw`.
- Lock-free movers.
- The refusal-kind split for recorders.
- Turning at the limit.
- The wider grep (`wake_up`, `goto_sleep`, `ReachyMini(`).

Also: the daemon listens on `0.0.0.0:8000`, which belongs to the embodied-device plan.

## 7. Red gates (strict xfail, written first; each runs the real path with a fake SDK)

1. A non-finite component `head_yaw`, and a 120° orient delta, are refused at load. Through the shipped path (`_start_agentic_runtime` → `_resolve_body_wiring`), the result is bodiless plus one ERROR, the agentic runtime starts, and NaN never reaches `goto_target`.
2. MaximCommandTool `turn_around(angle="nan")` and `move(yaw="nan")` are refused: no SDK call, the mirror is unchanged, and the tool result is an error.
3. `default_poses.json` with NaN, through `goto_pose`, is refused. No SDK call is made (in particular, no fall-through to `_default_head_pose`), and the tool result is an error. `turn_around(angle="nan")` through MaximCommandTool makes no SDK call; in particular, there is no 90° default turn.
4. These raise:
   - `MotionTarget` with duration 0, −1, nan, 11 or True;
   - body_yaw nan;
   - `method="Cartoon "`;
   - `PixelTarget(duration=0)`.
5. `_clamp(nan)` raises. A NaN body readback refuses the move, in both the controller and `move_head`.
6. `move_antenna`:
   - an explicit duration of 0 is refused;
   - a NaN angle is refused;
   - 200° is clamped to the limit;
   - `set_target` is never called;
   - the SDK receives `method="minjerk"` for the `min_jerk` and `ease` aliases;
   - the SDK receives `body_yaw=None`.

   `look_at_image(duration=0)` is refused, and the SDK `set_target` is never called.
7. `turn_around(duration=0.2)` dispatches step 1 at a duration of at least 0.1 s, not refused. `turn_around(duration=nan)` is refused before `inhibit` or a queue clear. With step 1 refused, step 2 is never dispatched.
8. A refused target leaves the retained-axes stash identical.
9. Planted violations are caught by both CI greps (the clamp idiom, and a raw `set_target` including an alias and a `getattr`).
10. A NaN, and separately a ±inf, body readback driven through the real `motion/movement.py::move_head` on `workers.py::motor_worker`:
    - the move is refused;
    - sync keeps the last finite mirror (no NaN, no hang);
    - no connection failure is counted.
11. Last mile: a `MotionTarget` whose `duration` is set to 0.01 after construction is refused at dispatch.
13. One gate per named site in the §3 table whose gate is not already listed. In each, a garbage or non-finite value produces an error result and **no SDK call**:
    - the explicit turn_around duration "abc" and 1e6;
    - goto_pose's second handler (no sleep);
    - the turn_around body readback;
    - the focus_interests `look_at_fn`;
    - the Selfy centred pose with NaN (finite default used, no NaN mirror);
    - the motor_worker blind reset skipped on a non-finite readback;
    - `goto_pose` with an invalid or unknown name (error, no motion);
    - TrackTargetTool `duration_s=0` refused.
14. **Present-pose check** (fake transport):
    - With the present head pose NaN (and separately Inf in a translation entry), `move_head`, `move_antenna`, a controller `goto_target`, `look_at_pixel`, and both `_enqueue_sdk_look_at` branches are each refused, and no SDK motion call is made.
    - Once the pose reads finite again, the next move dispatches.
    - A head-pose read that **raises** leaves today's behaviour unchanged: `test_body_read_failure_falls_back_to_zero` stays green.
    - A documentation test pins the WS fact: a NaN `HeadPoseMsg` and a NaN `JointPositionsMsg`, round-tripped through `model_dump_json` and the client's adapter validation, are both rejected. If a future SDK starts passing non-finite values through, this test flips, and the check becomes live on hardware.
12. DN: a `MotionRefused` from a gaze dispatch is handled by type (no IK classification), and a config `base_duration` of 0.05 is floored, not refused.
