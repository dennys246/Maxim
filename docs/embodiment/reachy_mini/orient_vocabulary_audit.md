# Orient-vocabulary audit — every path that commands orientation

**Date:** 2026-08-07 (1.1 cut-line item 8; see
[roadmap_1_1_to_1_3.md](../../plans/roadmap_1_1_to_1_3.md)).
**Trigger:** the hardware note — motors 2+3 were destroyed when Maxim commanded
a pose beyond the platform's physical capability. The breakage spanned
essentially the entire 1.0+ era. Two of the paths below bypassed
`ReachyMiniController.goto_target` entirely, and with it the head-frame
composition and any workspace clamping.

**The headline:** there are **seven** paths that command orientation, not the
three the plan docs talked about — plus a family of SEM `motion` affordances
that are advertised to the LLM while being motorless no-ops.

---

## The fix that shipped with this audit (2026-08-07 safety fold)

1. **Physical-limit clamps live in `ReachyMiniController.goto_target`** — the
   single canonical dispatch point. Roll/pitch ±40°, head yaw (BODY-RELATIVE
   frame — that is where the 65° neck constraint physically lives) ±65°,
   body yaw ±160°. Clamping logs a WARNING; it never silently succeeds at a
   different pose without saying so.
2. **`_motion_lock` (RLock) spans get_current_pose() → head-matrix
   composition → SDK dispatch.** Two overlapping callers previously each read
   the pose and composed a head matrix against a body yaw the other was
   concurrently changing (TOCTOU on live kinematic state).
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

Regression guard: [tests/unit/test_reachy_workspace_safety.py](../../../tests/unit/test_reachy_workspace_safety.py)
(verified to fail 10/14 against the pre-fold code) +
[tests/unit/test_reachy_head_frame.py](../../../tests/unit/test_reachy_head_frame.py).

---

## The seven orientation-command paths

| # | Path | Entry | Dispatch | Clamped? | Head-frame correct? | DN inhibit? |
|---|------|-------|----------|----------|--------------------:|-------------|
| 1 | **Controller** `ReachyMiniController.goto_target` | all sanctioned callers | SDK `goto_target` with composed head matrix | **YES (this fold)** — roll/pitch ±40°, rel-yaw ±65°, body ±160° | YES (2026-07-16 invariant) + motion lock (this fold) | n/a (callee) |
| 2 | **`FocusOnSoundTool`** (`tools/reachy.py`) | LLM tool call | controller `goto_target` | YES — own ±45° head-yaw envelope, tightened by learned bounds | YES (via controller) | **NO** |
| 3 | **SEM `orient` affordances** (`turn_left/right[_big]`) via `ReachyOrientMotorBackend` (`hardware/reachy/motor_backend.py`) | LLM tool call (entity-prefixed) | controller `goto_target(body_yaw=…)` | YES — own ±160° body clamp + controller | YES (via controller; refuses on unreadable pose) | **YES** (`_inhibit_dn`) |
| 4 | **`MoveTool` WITH `robot_id`** | LLM tool call | controller `goto_target` | Gaze params (`target_x/y`) clamped to ±45°/±30°; **raw `yaw`/`pitch` were UNCLAMPED until this fold** (now clamped at controller) | YES (via controller) | **NO** |
| 5 | **`MoveTool` WITHOUT `robot_id`** → `Selfy.move()` | LLM tool call (the default!) | `_enqueue_motor(move_head)` → raw SDK | **WAS NO — fixed this fold** (`_clamp_to_workspace_6d` in `move()`) | Head-only pose matrix; `body_yaw=None` on a head-only command is safe | **NO** |
| 6 | **`Selfy.turn_around`** — reached from BOTH `look_at_image`'s `_maybe_turn_around` AND the DN `TurnAround` behavior proposal | vision tracking / DN | **WAS raw `self.mini.goto_target` — fixed this fold** (now controller) | WAS hand-rolled ±160° only — now also controller-clamped | **WAS BROKEN** (STEP-2 `head=None` counter-rotation + neck-limit breach) — fixed | YES (inhibits DN for the turn) |
| 7 | **`Selfy.look_at_image` / `move_relative`** (DN visual tracking: `FocusInterestsTool`, `TrackTargetTool`, scan behaviors) | vision loop | `_enqueue_sdk_look_at` → SDK `look_at_image` (SDK-internal IK) / `move()` | Pixel bounds (`_LOOK_AT_PIXEL_BOUNDS`) + reachability gate + pain-risk restriction; `move_relative` clamps 6D | SDK-managed IK for look_at; `move()` head-only | Uses its own gates; not DN-inhibited (it IS the DN's effector) |

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
  controller's motion lock (this fold) serializes the *dispatches* so
  composition is no longer corrupted, but behavioral-level arbitration
  remains 1.3 scope.

## What this audit deliberately did not do

- No new mechanism (1.1 rule). The fixes route existing paths through the
  existing canonical dispatch point and add clamps + a lock there.
- No `world_set_axis` owner-gating (the reflex-canonicalization structural
  half) — tracked separately in the roadmap's reflex section.
- No change to SEM YAML affordance declarations.
