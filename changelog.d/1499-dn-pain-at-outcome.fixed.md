- **The Default Network and the pain bridge now book what a move did, when it did it** (#1499 PR 2b,
  [docs/plans/motion_path_model.md](docs/plans/motion_path_model.md) §5.6). Before, DN counted a move as done, and
  taught its gaze history, spatial map and attention network, the moment it was queued, so a refused or replaced move
  still taught "reachable". The pain bridge recorded a look-at's start at the same moment, so a queued move retired
  the one still running and that move's pain was booked against the wrong action.
  - DN's bookings run on DN's own tick from the move's final outcome (a new motion inbox), and they run while DN is
    inhibited and once more as it stops. `DNActionProposal.was_executed` and the action callbacks now arrive when the
    move finishes, and DN's current gaze position is where a move arrived, not where one is headed. Only a completed
    move or an IK failure teaches; `look_opposite` books the same way, and its `True` now means the move was issued.
  - DN's pacing does not change: the dwell and inhibition of return still start when a move is issued, for a move that
    later fails too (`GazeHistory.note_move_commanded`). A command that raised paces nothing, also as before.
  - A look-at's pain start is recorded when the move starts, inside the motion lock. A move replaced first records
    nothing. `record_action_complete(event_id=)` completes that event only, and the bridge's retire-and-install is
    one atomic step.
  - The automatic turn binds its own ticket, so its start is recorded behind any move in flight. A turn the
    controller refuses before its first SDK call now books nothing, and a look-at whose turn failed now reports
    the failure instead of success.
  - IK failures have their own outcome kind, `ik`, for the failures the client can see, whether the SDK raised or
    logged them: one path through the failure count, the mirror re-read, the bounds learner and recovery, and never
    a connection failure. It replaces DN's and `look_opposite`'s substring checks with a narrower match, so "like"
    or a port collision no longer counts. Plugins report `kind="ik"` themselves.
  - `move` answering "already at the requested pose" while that pose is still queued now returns the queued move's
    ticket, so a retry of a move about to fail no longer reads as done.
  - Nothing moves the robot after the live loop stops. The teardown used to drop the queue, which made the next move
    run synchronously while DN was still issuing them, and a body turn never went through the queue at all. The
    teardown now leaves a closed queue and, first of all, sets the controller's stop latch
    (`RobotController.stop_motion()`, a no-op by default for plugins), which a successful wake or a new connect
    clears. A soft reconnect that finishes after the stop re-latches and wakes nothing. The sleep pose still runs,
    and a stop that fails is logged without costing the rest of the teardown. A command the motor worker takes as it
    stops does not run either.
  Queued moves still book no pain completion, as before: the completion clears the detector's target, which would
  end its movement-failure check, and it would add a positive NAc link per look-at.
