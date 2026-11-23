- **A caller's time of exactly 0.0 is used, not swapped for the wall clock** (#1684): `SalienceMap.update`,
  `SalienceNetwork.update_from_detections` and `CostTracker.record` / `get_totals` / `get_per_provider_totals` read
  `x or time.time()`, so a host on a clock that starts at zero (a stepped world, a replay, a test clock) got wall time
  on its first tick. `SalienceMap` also counted a visit at 0.0 as "never visited"; the visit count decides that now.
