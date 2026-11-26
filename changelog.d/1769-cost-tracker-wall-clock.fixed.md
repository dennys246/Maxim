- **The cost ceiling can no longer be reset by a caller's clock** (#1769): `CostTracker` runs every window (the
  calendar month, the hourly and daily rolling windows, the spend-rate EMAs) on the wall clock. A caller's
  `record(timestamp=)` or `get_totals(now=)` / `get_per_provider_totals(now=)` is now ignored with a
  `DeprecationWarning`. Before, a caller on another clock (an embedder's stepped world starting near 0.0) could reset
  the month and prune its own spend out of the windows, so the ceiling stopped protecting the session; Maxim's own
  router always passed the wall's value. The router now passes neither keyword (same values), so a warnings-as-errors
  embedder no longer turns every routing decision into an error or every record into unbooked spend.
