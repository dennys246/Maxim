- **A caller's time of exactly 0.0 is used, not swapped for the wall clock** (#1684): `SalienceMap.update`,
  `SalienceNetwork.update_from_detections` and `CostTracker.record` / `get_totals` / `get_per_provider_totals` read
  `x or time.time()`, so a host on a clock that starts at zero (a stepped world, a replay, a test clock) got wall time
  on its first tick. Two more places read a 0.0 time as "never": `SalienceMap`'s visited flag (the visit count decides
  now) and `CostTracker`'s spend-rate average (the sample count decides now). `SalienceNetwork`'s readers
  (`get_salience`, `get_novelty`, `get_top_salient`, `get_objects_at_position`, `query`, `to_context_str`) take an
  optional `now`, so a host can read on the clock it writes with; they default to the wall clock as before.
