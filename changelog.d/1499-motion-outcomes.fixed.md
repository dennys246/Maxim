- **A Reachy move's outcome now reaches the model, not just "queued"** (#1499 PR 2a,
  [docs/plans/motion_path_model.md](docs/plans/motion_path_model.md) §5.4–5.8). Before, a move that went through
  Selfy's motor queue was fire-and-forget: a refused move (too fast, an unreadable pose, a sweeping path) was reported
  to the model as success, a replaced or cleared command vanished without a word, and `act` slept the requested time
  while the controller had stretched the move.
  - A new `MotorQueue` resolves every command it removes: `superseded` when replaced, dropped as full, cleared or
    drained on a reconnect; `cancelled` when a stop closes the queue (`sleep`, the live teardown), and for every
    later command. Nothing queued behind a stop runs.
  - Selfy's queued-motion methods (`move`, `move_relative`, `look_at_image`, `goto_pose`, `center_vision`,
    `move_antenna`, `act`) return a two-stage `MotionTicket`: `accepted` when the checks pass and the first SDK call
    starts (or with the refusal), `completed` with the final `MotionOutcome`. `goto_pose` no longer sleeps, and `act`
    waits for each move to complete.
  - The tools wait for `accepted` (at most 20 s, then "not started: the motor is busy"; a cancel holds until the move
    is accepted, so it never runs behind that answer) and return the controller's reason (`rate`, `path`,
    `reference`, `limit`…) instead of "Motion command failed". A duration the model chose is `exact` and refused
    when too fast; an absent one is stretched. `track_target` and `novelty_track` no longer advertise a 0.3 s
    default, which models copied into every call.
  - `turn_around` returns a `MotionOutcome`, plans from the daemon's fresh body yaw (never the mirror), and refuses
    a turn that would end past ±160° (`limit`) instead of shrinking it; the automatic turn in `look_at` then looks
    with the head clamped and books nothing with the pain bridge, and the Default Network's own turn is no longer
    counted as executed. A move that never started no longer leaves Selfy's pose mirror at its target.
  - `move_antenna` goes through the controller: the motion lock, the fresh readback and the rate cap.
  - `SimulatedController` returns `MotionOutcome` from `goto_target`, `look_at_pixel`, `wake_up` and `goto_sleep`.
  - Controller plugins: `RobotController.read_body_yaw()` (optional override; the default needs `body_yaw` in
    `get_current_pose()`) and `mark_motion_accepted()` (call it before the first actuator command) are documented
    in docs/user/extension_api.md.
