- **The Default Network and the pain bridge now book what a move did, when it did it** (#1499 PR 2b,
  [docs/plans/motion_path_model.md](docs/plans/motion_path_model.md) §5.6). Before, DN counted a move as done, and
  taught its gaze history, spatial map and attention network, the moment it was queued, so a refused or replaced move
  still taught "reachable". The pain bridge recorded a look-at's start at the same moment, so a queued move retired
  the one still running and that move's pain was booked against the wrong action.
  - DN's bookings run on DN's own tick from the move's final outcome (a new motion inbox), and they run while DN is
    inhibited. `DNActionProposal.was_executed` and the action callbacks now arrive when the move finishes. Only a
    completed move or an IK failure teaches; `look_opposite` books the same way, and its `True` now means the move
    was issued.
  - DN's dwell still starts when a move is issued, so it does not re-issue a move every tick while one is running
    (`GazeHistory.note_move_commanded`). A failed move still uses up the dwell, as before.
  - A look-at's pain start is recorded when the move starts, inside the motion lock. A move replaced first records
    nothing. `record_action_complete(event_id=)` completes that event only, and the bridge's retire-and-install is
    one atomic step.
  - `turn_around` binds its own ticket (a new `ticket=` keyword), so the automatic turn's start is recorded behind
    any move in flight. A turn the controller refuses before its first SDK call now books nothing.
  - IK failures have their own outcome kind, `ik`, which replaces DN's and `look_opposite`'s substring checks. The
    word match is stricter too: "ik" no longer matches inside "like".
  - `move` answering "already at the requested pose" while that pose is still queued now returns the queued move's
    ticket, so a retry of a move about to fail no longer reads as done.
  Queued moves still book no pain completion, as before: the completion clears the detector's target, which would
  end its movement-failure check, and it would add a positive NAc link per look-at.
