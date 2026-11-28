- **A defect building a simulation's SEM trace lines is no longer swallowed** (#863): `start_simulation_mode`
  wrapped its three `SEM_TRACE` emits (the SEM tool listing, ImaginationTrigger ACTIVE / FAILED) in
  `except Exception: pass`. `sim_log` has never raised since #868, so the wrap only hid faults in the caller's own
  argument construction; it is dropped, and such a fault now surfaces.
