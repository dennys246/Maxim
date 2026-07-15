# Deferred: LLM Path — Fair-Share Scheduling Across Agents

**Status:** Deferred shell plan
**Revive when:** multi-agent workloads consistently show one agent starving others even with per-agent rate limits, OR operators need priority classes (e.g., "interactive > batch > background").
**Estimated scope:** ~300-500 LOC
**Depends on:** [llm_path_operator_visibility.md](../archive/llm_path_operator_visibility.md) (Plan 3) — which introduces per-agent rate limiting
**Related deferred:** [llm_path_async_router.md](llm_path_async_router.md) — likely prerequisite for meaningful fair-share

## Why this was deferred

Plan 3 added **per-agent rate limiting** — a hard cap per agent via `mesh.yml::agent_rate_limits`. This prevents runaway agents from starving others. But it's a blunt tool:

- **No priority classes.** All agents are equal under rate limits. Can't say "interactive agents get priority over batch agents."
- **No work-conserving behavior.** If agent A is under its limit and agent B is idle, A can't use B's share.
- **No dynamic adjustment.** Limits are static in config; don't adapt to observed load.

Fair-share scheduling addresses all three. It's a bigger refactor because it requires:
1. A scheduler (token bucket → weighted queue)
2. Priority class configuration
3. Work-conserving fairness (probably deficit round-robin or similar)
4. Per-agent quota accounting (already partially there via Plan 3 admin API)
5. Async router or worker pool (otherwise `_inference_lock` defeats fairness)

**Plan 3's simpler rate limiting handles the 80% case** (runaway agent protection) with 70 LOC. Fair-share handles the remaining 20% but costs ~400+ LOC. Defer until the simple approach is proven insufficient.

## What this plan would add

Shell design, not committed:

### R-fair.1 — Scheduler abstraction

`LLMRouter` gets a pluggable scheduler that dispatches queued requests according to a policy:
- **FIFO** (current behavior, equivalent to no scheduling)
- **Round-robin** (one request per agent before repeating)
- **Weighted round-robin** (priority-based)
- **Deficit round-robin** (work-conserving fairness)

Policy configured via `mesh.yml::scheduler_policy`. Default: FIFO (current behavior).

### R-fair.2 — Priority classes

`mesh.yml::agent_classes` maps agent IDs to classes with weights:
```yaml
agent_classes:
  npc-mother: {class: interactive, weight: 10}
  npc-background: {class: background, weight: 1}
scheduler_policy: weighted_round_robin
```

### R-fair.3 — Queue depth + wait time metrics

Per-agent, per-class queue depth + wait time histograms. Observability is the point — fair-share without observability is impossible to debug.

### R-fair.4 — Quota rollover

Unused quota within a window rolls over to the next window (bounded). Work-conserving.

## Why this depends on async router

Fair-share scheduling is pointless if the router serializes everything under `_inference_lock`. Priority class A can only "preempt" class B if there's actual parallelism to allocate. Without async routing, fair-share reduces to "pick the next lucky agent" — which is basically what Plan 3's rate limiter already does.

**Practical implication:** revive `llm_path_async_router.md` first (or concurrently), then add fair-share on top.

## Revive trigger checklist

- [ ] Plan 3 rate limiting shipped and observed in production
- [ ] Stress tests show rate limiting alone is insufficient (specific scenario documented)
- [ ] Operators have explicit priority requirements (interactive vs batch)
- [ ] Async router is either shipped or being shipped concurrently
- [ ] Budget for ~400 LOC + observability tooling

## Open design questions (need answers at revive time)

1. Is FIFO within a priority class OK, or do we need fairness within a class too?
2. How are ad-hoc agents (not in `agent_classes`) handled? Default class, or rejected?
3. Does the scheduler preempt in-flight requests, or only affect queued ones? (Preempting an LLM call is hard.)
4. Metric cardinality impact — per-agent labels already bounded in Plan 3; this plan may add per-class labels.
5. Integration with cost tracking for cloud providers — does fair-share apply equally or cloud gets separate treatment?

## Design aspiration: bio-inspired scheduling, not just Kubernetes

Per user note: when this plan revives, we're **not required to stick with industry patterns** (Kubernetes CPU quotas, Linux cgroups, DRR). Maxim is a bio-inspired cognitive architecture; its scheduler can be too.

**Problems with Kubernetes-style quotas (the things NOT to copy):**

- **Wasted quota under idle peers.** K8s CPU limits don't redistribute unused capacity to busy pods in the same period.
- **Hard starvation under burst patterns.** A bursty agent can exhaust its quota in 100ms then starve for the rest of the window.
- **No awareness of LLM-specific cost.** A 100-token call and a 4000-token call count the same in req/min.
- **Blind to tail latency.** Industry fair-share historically optimizes for mean fairness; P99 can be terrible.
- **Context-free.** No notion of why an agent is making a call, just that it did.

**Bio-inspired alternatives worth exploring at revive time:**

- **Dopamine-like reward modulation.** Agents whose recent outputs produced measurable reward (via NAc feedback) get a latency boost — they've shown they're productive, they get prioritized. Ties directly into substrate P2's reward mechanism.
- **Energy budget from `maxim.energy`.** Every agent has a per-window energy budget that depletes on LLM calls (weighted by token cost) and regenerates during idle time. Zero energy = wait. Unlike token buckets, energy can be "spent" on other things too (memory operations, etc.) for a unified cost model.
- **Fatigue / habituation.** An agent making the same request repeatedly (cache-hit-style) gets deprioritized — the architecture implicitly says "this isn't novel, try something else." Fresh requests get priority.
- **Attention-weighted scheduling.** Priority derived from an attention signal (what the user is currently focused on, what the agent's goal relevance is). Inspired by selective attention in biological cognition.
- **Sleep-cycle batching.** Low-priority work (sleep replay, memory consolidation, background embedding generation) deferred to "sleep" phases where the agent isn't processing interactive input. Aligns with substrate P8.
- **Per-lane homeostasis.** Each tier (large/medium/small) maintains a target load — if it drifts high, scheduler pressures agents toward smaller tiers (or deferred execution) via natural feedback, not hard caps.

**Design constraint:** whatever scheduling approach we pick must still answer the basic questions a production scheduler answers (fairness, predictability, tail-latency SLOs). Bio-inspired is additive to those requirements, not a replacement.

**At revive time:** pick at least ONE bio-inspired mechanism to integrate alongside whatever baseline fair-share we implement. The goal isn't to be purely novel — it's to not throw away Maxim's identity when we reach for industry patterns.

## Related docs

- **Plan introducing simpler rate limiting:** [../llm_path_operator_visibility.md](../archive/llm_path_operator_visibility.md)
- **Prerequisite (deferred):** [llm_path_async_router.md](llm_path_async_router.md)
- **Meta plan:** [../llm_path_refinement.md](../archive/llm_path_refinement.md)
