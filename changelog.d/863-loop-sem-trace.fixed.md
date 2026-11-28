- **The agent loop's perception traces no longer hide the caller's faults, and its perception failures are reported**
  (#863): the four trace lines in `runtime/loop_perception.py` (imagination skipped, auto-sense, audio-orient, the
  orienting reflex) sat in an inner `try` inside each section's broad handler. `sim_log` has never raised since #868,
  so those wraps only caught faults in the caller's own argument construction. Each line now runs after its section's
  `try`, so such a fault ends the agent run (owner decision 2026-11-27: fail loud). The three section handlers
  (`imagine`, `auto_sense`, `orient_to_audio`), and `auto_sense`'s per-tool and self-sense handlers, now report a
  failure of their real logic as a structured `swallowed_exception` event (WARNING on a site's first fire), where
  before it was a DEBUG line with no event.
- **A sub-simulation whose AUT loop dies now says so**: `spawn_sub_simulation` logs the failure at ERROR and names it
  in the sub-report as `aut_error`, where before the exception went to a list nothing read and the narrator saw only
  a timeout.
