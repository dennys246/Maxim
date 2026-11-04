- **The Reachy controller turns the daemon's automatic body yaw off again before every discrete move** (#1499 PR 3).
  The setting is daemon-wide and has no getter, so a second SDK client connecting mid-session could turn it back
  on, and the daemon would then rotate the body behind Maxim. `goto_target` and `goto_sleep` now re-assert it under
  the motion lock first; if the call fails, the move is refused (`reference`) and nothing moves until an assertion
  succeeds. The look-at tracking stream does not re-assert, which would cost about 10 SDK calls a second. The
  bundled `src/tests/basic_move.sh` no longer turns the setting back on for every client, and the motion lint now
  flags any `ReachyMini(` in `src/` or `scripts/` built without `automatic_body_yaw=False`.
