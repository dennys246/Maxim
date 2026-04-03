# Realtime Refinement Plan

> **Status:** Not started. Consolidates observation-gated work from multiple plans into one cohesive system for watching, measuring, and tuning Maxim's behavior in real time.

## Vision

A unified approach to observing Maxim in operation (live, simulation, or headless) and iteratively refining its behavior based on what you see. This isn't a feature — it's a practice supported by tooling across several subsystems.

Currently, tuning requires reading logs after the fact. This plan creates a feedback loop: observe in real time → identify issues → adjust → observe the effect.

---

## Components

### 1. Simulation Agent Tuning (from Simulation Agent Phase 4)

Run the simulation agent against the AUT, observe orchestrator behavior, and refine:

- **Persona prompt iteration:** Run adversarial persona → observe if LLM follows escalation pattern → adjust context_prompt wording → re-run
- **Tool usage patterns:** Does the orchestrator call check_completion too often? Not enough? Does it use analyze_results between phases?
- **Settle detection tuning:** Is 2s settle_s too long (slow simulations) or too short (misses multi-action responses)?
- **Campaign decomposition quality:** Does the campaign persona actually decompose goals into phases, or does it just run random probes?

**How to observe:** `--sim agent --sim-debug` shows all tool calls, percept injections, and LLM reasoning. `/status` shows turn counts and action summaries. `/report` triggers interim analysis.

**What to tune:** Persona context_prompts in `simulation/personas.py`. Settle timeouts in `SimulationBridge` defaults. CheckCompletion heuristics.

### 2. Intelligent Context Refinement (from Intelligent Context Upgrade Parts 1-2)

The remaining observation-gated work from the context upgrade plan:

- **Part 1 v3-v4: Edit disambiguation prompt tuning**
  - Observe: How often does the LLM use `context_before`/`context_after`?
  - Observe: When it uses them, does disambiguation accuracy improve?
  - Tune: Adjust prompt instructions based on usage patterns
  - Auto-suggest: When 3+ matches found, include suggested context in error messages

- **Part 2 v2-v4: LLM-driven turn pinning**
  - Observe: With v1 (always pin turn 1), does the LLM still contradict earlier decisions?
  - If contradictions persist: Add `pin_turns` field to LLMProposal, basic prompt instruction
  - Observe: Does the LLM pin at all? What does it pin? Is it over-pinning?
  - Tune: Refine pinning instructions based on pin rate and survival rate

**Metrics to track:**
- Retry rate (edit disambiguation failures per session)
- Disambiguation usage rate
- Contradiction rate (before/after pinning)
- Pin rate and over-pinning rate (>50% turns pinned = problem)

### 3. Per-Lane LLM Metrics (from Multi-LLM Scaling Phase 8)

Once multi-model is running, observe per-lane performance:

- **LaneMetrics:** jobs completed, dropped, avg latency, remote ratio, failover count
- **Which lane is bottlenecking?** If `infer` queue is always full but `review` is idle, rebalance
- **Remote vs local:** What percentage of calls go to remote? Is tunnel latency acceptable?
- **Model quality comparison:** Does the GPU model produce better plans than the CPU model?

**How to observe:** CLI `--status` or tool call showing lane metrics. Provenance traces per-request routing decisions.

### 4. NAc Causal Learning Observation

The NAc learns from every tool execution. Observe what it's learning:

- **Existing tools:** `predict_outcome` and `causal_links` introspection tools
- **What to watch:** Are causal links forming correctly? Are confidence scores reasonable?
- **Simulation agent feedback:** The orchestrator's NAc learns from probe outcomes — check if "approach X always gets blocked" actually converges

### 5. Provenance & Tracing

The provenance system (implemented) already traces execution. Use it for refinement:

- **ExplainTool:** Query what happened in a recent cycle
- **Session JSONL logs:** Post-hoc analysis of LLM reasoning chains
- **Sim logger:** Bio-subsystem traces during simulation (PERCEPT, HIPPOCAMPUS, FEAR, PAIN, etc.)

---

## Implementation Approach

This plan doesn't have phases — it's an ongoing practice. The infrastructure is mostly built:

| Component | Status | What's needed |
|-----------|--------|---------------|
| Simulation agent observation | **Ready** | Use `--sim-debug`, `/status`, `/report` |
| Edit disambiguation metrics | **Mechanism built** | Add logging of usage/retry/accuracy rates |
| Turn pinning | **v1 done** | Add `pin_turns` to LLMProposal (Part 2 v2) |
| LaneMetrics | **Not built** | Implement with Multi-LLM Phase 8 |
| NAc introspection | **Ready** | Use existing `predict_outcome`, `causal_links` tools |
| Provenance tracing | **Ready** | Use `ExplainTool`, session logs |

**When to start:** After running the simulation agent a few times and after multi-LLM Phases 1-3 are live. The observation data needs to exist before you can refine from it.

---

## Metrics Dashboard (future)

A lightweight CLI or web view showing:
- Active simulation: turn count, blocked rate, persona effectiveness
- LLM performance: per-lane latency, queue depth, failover events
- Memory health: hippocampus size, NAc link count, consolidation timing
- Context quality: edit retry rate, disambiguation usage, pin rate

This is a nice-to-have, not a prerequisite. Start with `--sim-debug` output and log analysis.
