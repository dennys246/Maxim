- **A Reachy move no longer starts against a stale body target** (#1563). On the first control tick of a move, the
  daemon composes the present head against the body target its last move left. The ±65° clamp is checked against
  the present body instead. SDK 1.8.3 never serves that target (`GET /api/state/full` drops `target_body_yaw`), so
  it cannot be checked. Instead, before each chunk the controller re-pins the daemon's head and body targets to
  the pose it just read, so the first tick holds the pose the robot is already in. The frame is skipped when the
  target the controller's last command left is already within 1° of the present body, so tracking adds almost no
  traffic. The daemon ignores the re-pin while a move another client started is running.
