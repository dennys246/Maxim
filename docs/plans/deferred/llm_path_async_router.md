# Deferred: LLM Path — Async Router for Concurrent Per-Lane Dispatch

**Status:** Deferred shell plan — scope not fully designed
**Revive when:** multi-agent workloads show significant head-of-line blocking on `LLMRouter._inference_lock` under post-Plan-2 stress tests, AND `llama.cpp --parallel` batching does not sufficiently hide the blocking.
**Estimated scope:** ~800-1,200 LOC (large refactor)
**Depends on:** [llm_path_fast_failover.md](../llm_path_fast_failover.md) (Plan 2) — `_MaximPeerBackend` must exist first
**Related deferred:** [llm_path_fair_scheduling.md](llm_path_fair_scheduling.md)

## Why this was deferred

The existing `LLMRouter._inference_lock` serializes inference calls within a lane. Under multi-agent workloads (AgentPool with N NPCs), agent A's in-progress call blocks agent B's call until A completes.

**Before Plan 2:** this was catastrophic. One failed call held the lock for ~52 seconds (the `_OpenAIBackend` retry loop). All other agents waited.

**After Plan 2:** the worst case shortens to ~2-5 seconds (the `_MaximPeerBackend` one-call-no-retry pattern). Typical case is the actual call duration: 100ms-5s depending on model + prompt length. **This may be acceptable.**

**When it becomes unacceptable:**
- More than ~3 concurrent agents per lane
- Streaming responses (~seconds of lock hold per call)
- User-interactive latency requirements under multi-agent load

**How to know:** post-Plan-2 stress tests measure `backend_call_duration_seconds` under concurrent load AND measure the **wait time** between an agent submitting a request and the backend call actually starting. If wait time p99 > backend call p99, the lock is the bottleneck.

## What this plan would change

This is a significant refactor. The shell below is a starting point, not a commitment.

### Approach A — Make the router async

Convert `LLMRouter._complete_text_locked` to an async method. Replace `threading.Lock` with `asyncio.Lock` (or drop the lock entirely if `ProviderState` mutations can be made atomic). Every backend's `complete_with_usage` becomes `async def`. Python GIL still applies but I/O-bound waits no longer serialize.

**Pros:** natural fit for HTTP-bound inference. httpx already supports async.
**Cons:** viral change — every caller up the stack becomes async or uses `asyncio.run` bridges. Big refactor across `agent_loop.py`, `executor.py`, `agents/*`.

### Approach B — Per-provider lock instead of per-router lock

Keep the router sync but use a `Lock` per `provider_key` instead of one `_inference_lock`. Agent A calling `lane-large-rtx-leader` and agent B calling `lane-large-mac-peer` don't block each other.

**Pros:** small change. No caller impact.
**Cons:** doesn't help when both agents hit the same provider (the common case in Plan 3 with reactive overflow). Also doesn't parallelize calls to the same backend instance.

### Approach C — Thread pool per lane

Dispatch `LLMRouter` calls to a thread pool with N workers per lane. Each worker has its own `LLMRouter` instance (shared config). `_inference_lock` becomes per-worker. N concurrent calls proceed in parallel.

**Pros:** retains sync semantics from callers' perspective. Bounded parallelism.
**Cons:** N copies of router state. Backoff tracking duplicated (though `ProviderState` could be shared via a reader-writer lock). Worker-thread debugging is harder than async.

### Recommendation (not yet committed)

Approach B first (small, safe, fixes the 2-peer case). If stress tests still show wait-time as bottleneck, Approach C (bounded parallelism per lane). Approach A only if the whole codebase is moving to async for other reasons.

## Open design questions (need answers at revive time)

1. How does `ProviderState` backoff handle concurrent mutations? Reader-writer lock, atomic counters, or immutable snapshot?
2. Does `_maxim_peer_backend.complete_with_usage()` thread safety hold under concurrent calls? (httpx client is thread-safe; our wrapper needs verification.)
3. How does streaming interact with parallelism? Two streaming responses share the lane's throughput.
4. Do we need fairness between agents within a lane, or is FIFO OK? (Fair-share is [llm_path_fair_scheduling.md](llm_path_fair_scheduling.md).)
5. Impact on `CostTracker` under concurrent cloud provider calls?

## Revive trigger checklist

- [ ] Plan 2 shipped, stress test complete
- [ ] Phase B or E stress test shows wait-time p99 > backend call p99 (lock is bottleneck)
- [ ] `llama.cpp --parallel` batching already exhausted as cheaper alternative
- [ ] Workload has ≥3 concurrent agents per lane OR significant streaming traffic
- [ ] Measurable user impact from the head-of-line blocking

## Related docs

- **Plan:** [../llm_path_fast_failover.md](../llm_path_fast_failover.md) — shortens the blocking duration
- **Deferred sibling:** [llm_path_fair_scheduling.md](llm_path_fair_scheduling.md) — fairness layer that would sit on top
- **Deferred sibling:** [llm_path_multi_peer_dispatch.md](llm_path_multi_peer_dispatch.md) — distribution alternative
