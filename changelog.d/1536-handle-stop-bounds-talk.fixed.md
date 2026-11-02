- **`MaximHandle.stop(deadline=)` now bounds the Talk loop's own session end** (#1536). An embedder stopping one handle
  makes no process-wide stop request. Until now, the Talk loop's full consolidation (sleep, replay, promotion, ATL/AG)
  ran past the embedder's deadline. The handle now gives each Talk loop it builds a bound that `stop()` sets, including
  for a loop still ending from `rest()` or a rebuild. The bound cuts only the consolidation steps: everything the
  session learned is still flushed and saved, and a campaign's session is untouched.
