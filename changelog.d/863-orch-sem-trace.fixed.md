- **A defect building a simulation trace line is no longer swallowed** (#863): nine `sim_log` emits in
  `maxim.simulation` sat inside `except Exception: pass`. They are the three `SEM_TRACE` lines in
  `start_simulation_mode` (the SEM tool listing, ImaginationTrigger ACTIVE / FAILED), the bridge's turn-entry trace,
  the fixture pre-trigger's failure trace, the reactive mother's per-turn line, and the damage, actor and sensor tool
  traces. `sim_log` has never raised since #868, so each wrap only hid faults in the caller's own argument
  construction. The wraps are dropped, and such a fault now raises from the trace line. The mother's line sits inside its tick's own
  handler, so its fault is logged there at DEBUG. The fixture pre-trigger's line sits in its fail-soft handler, so a
  fault building it now ends the fixture run. A fault in a setup-time trace
  now stops the run before the AUT starts, where before only the line was lost. A test holds `maxim.simulation` at zero
  such wraps and ratchets the rest of `src/maxim/` (#863's remaining sites).
