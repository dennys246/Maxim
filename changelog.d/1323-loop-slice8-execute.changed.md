- **The agent loop's pending-action execution moved into `runtime/loop_execute.py`** (#1323): section 4 moved verbatim
  into `execute_pending_tick`. That covers the #1451 set-aside of a run-ending answer, the live-situation stamp, the
  consecutive same-tool cap (now `_same_tool_cap`), the parallel batch, the plan gate, and the autonomy dispatch (run,
  queue for approval, park a confirmation, or book a hard rejection). Its two log helpers moved with it. Behaviour is
  unchanged and pinned by a new characterization through the public loop. The body shrinks from 1,078 to 942 lines,
  under the 1.4.0 gate's 1,000 (1.3.6 loop phase 2, slice 8).
