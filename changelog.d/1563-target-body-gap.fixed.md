- **A Reachy move is refused when the daemon's body target and the present body disagree** (#1563). The
  daemon's IK composes the head against its body target, not the body the controller's ±65° neck clamp was
  checked against, and the first control tick of a move can still see the previous target. The read before each
  dispatch (a move's first read, each chunk's re-read and each lifecycle step) now also reads `target_body_yaw` and
  refuses (`reference`) when it is more than 2° from the present body. A daemon that has set no target yet reads as
  0°, the value its IK uses, and a readback without the field refuses, which the connect-time probe now reports as
  one ERROR.
