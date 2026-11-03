- **A Reachy move is refused when the head sits past the neck clamp from the daemon's previous body target**
  (#1563). On the first control tick of a move, the daemon composes the present head against the body target its
  last move left, not against the present body that the ±65° clamp is checked against. SDK 1.8.3 never serves that
  target (`GET /api/state/full` drops `target_body_yaw`), so the controller tracks the target its own last move
  left. Before each chunk it refuses (`reference`) a head more than 66° from that target (65° plus a provisional
  1° for servo noise). A target set by another client is not covered, and neither is the first move after a
  connect.
