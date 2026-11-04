- **A Reachy move is refused when the head sits past the neck clamp from the daemon's previous body target**
  (#1563). On the first control tick of a move, the daemon composes the present head against the body target its
  last move left. The ±65° clamp is checked against the present body instead. SDK 1.8.3 never serves that target
  (`GET /api/state/full` drops `target_body_yaw`), so the controller tracks the target its own last move left.
  Before each chunk it refuses (`reference`) a head more than 1° (provisional, for servo noise) past both the
  clamp and the twist the robot already holds. The tracked target is forgotten after a disconnect, after a wake
  from torque off, and when the body is found 5° or more from it, which refuses once. So the refusal never locks
  the robot out. Neither a target set by another client nor the first move after a forget is covered.
