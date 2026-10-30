- **No too-fast, wrongly-referenced or mid-path-unsafe command reaches a Reachy motor** (#1499 PR 1; decisions
  MP1–MP12, [docs/plans/motion_path_model.md](docs/plans/motion_path_model.md)). The SDK moves the head the short way
  round the world rotation while the body follows its joint line, so a move whose endpoints are both legal could
  swing the neck through about 180° mid-way: from a turned body through `center_vision`, every SDK `look_at`, wake
  and sleep, and large turns. `ReachyMiniController` now:
  - reads its reference fresh from the daemon (`GET /api/state/full`) before every dispatch and between chunks, and
    refuses (`reference`) on a failed, malformed, torque-off, impossible or diverged read, with no `0.0` fallbacks;
  - caps the rate (180°/s average, 340°/s peak), stretching a default duration and refusing a too-fast explicit one;
  - samples the SDK's own path and splits a move that would sweep the neck (body, then neck yaw, then roll and
    pitch), checking every chunk before any moves, re-checking the pose between chunks, and stopping cleanly on an
    abort, an SDK timeout or a dropped chunk;
  - returns a `MotionOutcome` (truthy iff the move completed; `bool` stays accepted from plugins via `as_outcome`).
  `look_at` turns the head only. Wake and sleep re-centre a turned robot first and run the SDK's steps as clamped
  dispatches, so the sleep antennas stop at ±90° instead of the SDK's ±175°; sleep with torque already off returns
  ok without moving. After a sleep, new head and body moves are refused (and the robot reports not awake) until a
  wake succeeds, even across a reconnect; after a disconnect, until the next connect.
  `automatic_body_yaw: true` is now a config error. `move_head` delegates to the controller, and standalone callers
  must pass `motion_lock=`. The motor worker books the gaze and bounds learners only on a completed move. Not yet
  covered (#1499 PR 2): `move_antenna` still dispatches straight to the SDK (clamped, but with no rate cap or lock),
  and the tools do not yet mark model-chosen durations as explicit, so those are stretched rather than refused.
