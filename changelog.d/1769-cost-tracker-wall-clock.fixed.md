- **The cost ceiling can no longer be reset by a caller's clock** (#1769): `CostTracker` runs every window (the
  calendar month, the hourly and daily rolling windows, the spend-rate EMAs) on the wall clock. A caller's
  `record(timestamp=)` or `get_totals(now=)` / `get_per_provider_totals(now=)` is now ignored with a
  `DeprecationWarning`. Before, a caller on another clock (a stepped world starting near 0.0) reset the month and pruned
  its own spend out of the windows, so the ceiling stopped protecting the session. The router still passes
  `timestamp=` with the wall's value, so its behaviour is unchanged; it stops passing it at A-post.
