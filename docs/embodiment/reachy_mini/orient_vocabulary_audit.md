# Orient-vocabulary audit — every path that commands orientation

**Date:** 2026-08-07 (1.1 cut-line item 8; see
[roadmap_1_1_to_1_3.md](../../plans/roadmap_1_1_to_1_3.md)).
**Trigger:** the hardware note — motors 2+3 were destroyed when Maxim commanded
a pose beyond the platform's physical capability. The breakage spanned
essentially the entire 1.0+ era. Two of the paths below bypassed
`ReachyMiniController.goto_target` entirely, and with it the head-frame
composition and any workspace clamping.

**The headline:** there are **nine** paths that command orientation, not the
three the plan docs talked about (the original draft of this audit found
seven; the two-lens pre-merge review found two more — which is the point of
the round) — plus a family of SEM `motion` affordances that are advertised to
the LLM while being motorless no-ops.

---

## The fix that shipped with this audit (2026-08-07 safety fold + review fold)

1. **Physical-limit clamps live in `ReachyMiniController.goto_target`** — the
   canonical dispatch point. Roll/pitch ±40°, head yaw (BODY-RELATIVE frame —
   that is where the neck constraint physically lives) ±65°, body yaw ±160°.
   Clamping logs a WARNING, records the axes on
   `controller.last_clamped_axes`, and `MoveTool` surfaces them to the LLM —
   echoing the commanded angles after a silent clamp is the "accepted
   dispatch is a promise, not a motion" dishonesty class PR #459 fixed.
2. **`_motion_lock` (RLock) spans get_current_pose() → head-matrix
   composition → SDK dispatch** (TOCTOU on live kinematic state).
   `look_at_pixel` takes the same lock.
3. **`Selfy.turn_around` routes through the controller** instead of the raw
   SDK. Its pre-fold STEP 2 shipped `head=None` while rotating the body up to
   ±160° — the daemon held the head at its RETAINED world pose, commanding a
   head-relative angle far past the ±65° neck capability. **This is the
   likeliest concrete mechanism of the motor destruction.**
4. **`Selfy.move()` clamps via `_clamp_to_workspace_6d`** before enqueueing.
   Pre-fold it applied only the per-call max-step limiter, whose defaults are
   all `0.0` (= disabled), then shipped the raw ABSOLUTE pose — so the
   MoveTool no-`robot_id` fallback sent an LLM-passed `yaw=<anything>`
   straight to the motors.
5. **`move_head` composes the WORLD pose from its BODY-RELATIVE yaw at
   dispatch time, and mirrors the rotation clamps** (review fold — BLOCKING
   finding). The Selfy layer's frame contract is body-relative
   (`sync_head_position`: `yaw = world − body`) but the SDK matrix is world;
   pre-fold `move_head` shipped relative AS world with `body_yaw=None`, so
   after any SEM body turn a "centered" gaze command dragged the head toward
   WORLD 0 — a head-relative demand of up to ±160°: the destruction
   mechanism, reincarnated one layer up. Fixing the primitive repairs
   `Selfy.move`, `goto_pose`, and the workers.py IK-recovery recenter at
   once. Body-frame translation intent is likewise rotated into world.
6. **`goto_pose`/`center_vision` route through `Selfy.move()`** (review fold)
   so config-authored poses get the workspace clamp instead of a direct
   enqueue.
7. **The bounds learner is tighten-only against the hardcoded safe limits**
   (review fold): `_get_workspace_limits` takes `min(safe, learned)` — a
   config-inflated learner ceiling can no longer widen a safety clamp.
8. **`SimulatedController` defaults its limits to the real controller's
   constants** (review fold, sim-hardware parity) and clamps roll/pitch too.
   Explicit tighter limits (Exp 49's measured ~±22°) still win.
9. **CI grep** restricts raw `mini.(goto_target|set_target|look_at_image)`
   dispatch to the sanctioned primitives (structural enforcement — the next
   hand-rolled dispatch fails CI, not a hardware session).

Regression guard: [tests/unit/test_reachy_workspace_safety.py](../../../tests/unit/test_reachy_workspace_safety.py)
(verified to fail against the pre-fold code) +
[tests/unit/test_reachy_head_frame.py](../../../tests/unit/test_reachy_head_frame.py) +
the CI grep in `.github/workflows/test.yml`.

### The 65° provenance conflict (flagged, resolved by H1)

The ±65° head-relative-yaw clamp is the **vendor-doc capability ceiling**
(SDK docs: "Yaw delta max 65°"). Exp 49's sim seams pin a **measured ~±22°**
— but that measurement was taken on the degraded motors-2+3 platform, and the
CLAUDE.md head-frame invariant separately mentions "~±15-18°" platform-own
travel. These cannot all be right. The clamp uses the vendor ceiling (a
command inside vendor spec must not destroy motors; the incident involved
~160°-scale relative demands, far past any candidate value). **H1's Part A
measures the true achievable relative yaw on healthy hardware** and settles
it; if the healthy envelope comes back materially below 65°, tighten the
constant then, citing the measurement.

---

## The nine orientation-command paths

| # | Path | Entry | Dispatch | Clamped? | Head-frame correct? | DN inhibit? |
|---|------|-------|----------|----------|--------------------:|-------------|
| 1 | **Controller** `ReachyMiniController.goto_target` | all sanctioned callers | SDK `goto_target` with composed head matrix | **YES (this fold)** — roll/pitch ±40°, rel-yaw ±65°, body ±160° | YES (2026-07-16 invariant) + motion lock (this fold) | n/a (callee) |
| 2 | **`FocusOnSoundTool`** (`tools/reachy.py`) | LLM tool call | controller `goto_target` | YES — own ±45° head-yaw envelope, tightened by learned bounds | YES (via controller) | **NO** |
| 3 | **SEM `orient` affordances** (`turn_left/right[_big]`) via `ReachyOrientMotorBackend` (`hardware/reachy/motor_backend.py`) | LLM tool call (entity-prefixed) | controller `goto_target(body_yaw=…)` | YES — own ±160° body clamp + controller | YES (via controller; refuses on unreadable pose) | **YES** (`_inhibit_dn`) |
| 4 | **`MoveTool` WITH `robot_id`** | LLM tool call | controller `goto_target` | Gaze params (`target_x/y`) clamped to ±45°/±30°; **raw `yaw`/`pitch` were UNCLAMPED until this fold** (now clamped at controller) | YES (via controller) | **NO** |
| 5 | **`MoveTool` WITHOUT `robot_id`** → `Selfy.move()` | LLM tool call (the default!) | `_enqueue_motor(move_head)` → raw SDK | **WAS NO — fixed this fold** (`_clamp_to_workspace_6d` in `move()`) | Head-only pose matrix; `body_yaw=None` on a head-only command is safe | **NO** |
| 6 | **`Selfy.turn_around`** — reached from BOTH `look_at_image`'s `_maybe_turn_around` AND the DN `TurnAround` behavior proposal | vision tracking / DN | **WAS raw `self.mini.goto_target` — fixed this fold** (now controller) | WAS hand-rolled ±160° only — now also controller-clamped | **WAS BROKEN** (STEP-2 `head=None` counter-rotation + neck-limit breach) — fixed | YES (inhibits DN for the turn) |
| 7 | **`Selfy.look_at_image` / `move_relative`** (DN visual tracking: `FocusInterestsTool`, `TrackTargetTool`, scan behaviors) | vision loop | `_enqueue_sdk_look_at` → SDK `look_at_image` (SDK-internal IK) / `move()` | Pixel bounds (`_LOOK_AT_PIXEL_BOUNDS`) + reachability gate + pain-risk restriction; `move_relative` clamps 6D | SDK-managed IK for look_at; `move()` head-only | Uses its own gates; not DN-inhibited (it IS the DN's effector) |
| 8 | **`Selfy.goto_pose` / `center_vision`** (found by the review round) | LLM-invocable via `RobotCommandTool._ALLOWED` + internal callers | **WAS a direct `_enqueue_motor(move_head)` with NO clamp — fixed** (routes through `Selfy.move()`) | Config-authored pose values; now workspace-clamped + `move_head` rotation-clamped | Now correct via `move_head`'s body-frame composition | **NO** |
| 9 | **workers.py IK-failure recovery recenter** (found by the review round) | fires automatically on repeated IK failures | raw `move_head(mini, 0,…,0)` | Rotation-clamped at `move_head` (fixed) | **WAS BROKEN**: commanded WORLD zero with body retained — with body at ±160° that demands relative ∓160° *while the platform is already in IK distress*. Fixed by `move_head`'s frame composition: 0 now means centered-on-body. | n/a (recovery path) |

**Antennas:** `move_antenna` (`motion/movement.py`) dispatches raw
`mini.set_target/goto_target(antennas=…)` with no range clamp. Different
motors than the incident, same commanding-beyond-range family — LLM-invocable
via `RobotCommandTool`. Disposition: documented; antenna range clamping is a
follow-up (the antenna motors are low-torque and the SDK tolerates
over-range, but "the SDK tolerates it" is the assumption class this incident
falsified — do not leave it undocumented).

### Notes on the table

- **Path 5 is the default MoveTool path.** `robot_id` is optional and the LLM
  essentially never passes it, so the *unclamped* branch was the one in
  production use.
- **Path 6 is the best candidate for the destruction mechanism.** STEP 1
  centered the head at **world** 0, then STEP 2 rotated the body up to ±160°
  with `head=None` — the daemon actively counter-rotates to hold the head at
  world 0, which demands a head-relative angle of up to 160° against a ±65°
  physical capability. "Commanded a pose beyond its physical capability;
  the motors glitched; the head snapped to the opposite extreme" is exactly
  what an IK solver fighting an impossible constraint looks like.
- Paths 2/3 carry their own pre-clamps AND now inherit the controller's. The
  double clamp is intentional (defense in depth); the per-tool envelopes are
  *tighter* than physical limits and shape behavior, the controller's are
  capability limits and prevent damage.

---

## The motorless-affordance dishonesty (documented, NOT fixed here)

`reachy_mini.yaml` declares a `motion` modulator with affordances
**`look_at`, `goto_pose`, `recenter`, `nod`, `shake_head`** — all advertised
to the LLM as tools. But `make_reachy_orient_factory` binds motor backends
ONLY for `mod_name == "orient"`, so every `motion` affordance is a
**motorless no-op that reports success**. This is the same dishonesty class
PR #459 fixed for `focus_on_sound` ("accepted" ≠ "moved"): the LLM plans
around gestures that never happen and learns from success signals for actions
that had no effect.

**Disposition (1.1):** documented here; fixing belongs with the SEM
motor-binding line (either bind them in `motor_backend.py` or stop
advertising them — the roadmap's Phase-3 gesture work decides which).
Do NOT silently extend the orient factory to bind them without the
Phase-2/3 measured-credit design — an unmeasured gesture path would
re-open the phantom-credit hole `drive_credit_withheld` closed.

## Coordination gaps (documented, NOT fixed here)

- **`focus_on_sound` recommends the SEM turn tool by name when clamped**
  (`_focus_result_note`), with no coordinator, refractory, or mutual
  exclusion between paths 2 and 3. A single ~45° sound can produce ~97° of
  world rotation (head to envelope + big body turn). The reflex
  canonicalization decision (roadmap: DN is the canonical home; arbitration
  via `PriorityArbiter`) is the 1.3 answer; until then the honest state is:
  two LLM-triggered orient paths exist and the LLM is the only arbiter.
- **Only paths 3 and 6 inhibit the DefaultNetwork during motion.** Paths 2,
  4, 5 can have DN tracking commands interleave with their motion. The
  motion lock serializes read→compose→dispatch for the controller's
  `goto_target`, `look_at_pixel`, AND `Selfy.move()`'s motor-queue dispatch
  (`move_head` takes the controller's lock when Selfy can supply it).
  **Residual unserialized dispatchers:** `_enqueue_sdk_look_at` (SDK-managed
  IK, FOV-bounded) and the workers.py recovery recenter (lock-less
  `move_head` — degrades to a fresh body read with a small residual window).
  Behavioral-level arbitration remains 1.3 scope.
- **Lock hold time:** the SDK's `goto_target` blocks until motion completes,
  so the lock is held up to `duration + 1 s`; a `focus_on_sound` issued
  during a turn waits it out (intended serialization). Knock-on: registry
  operations holding `_robots_lock` can stall behind an in-flight motion —
  strict `_robots_lock → _motion_lock` ordering, so a stall, never a
  deadlock.

## What this audit deliberately did not do

- No new mechanism (1.1 rule). The fixes route existing paths through the
  existing canonical dispatch point and add clamps + a lock there.
- No `world_set_axis` owner-gating (the reflex-canonicalization structural
  half) — tracked separately in the roadmap's reflex section.
- No change to SEM YAML affordance declarations.
