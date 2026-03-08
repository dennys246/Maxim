# Contemplation Plan: Local Chain-of-Thought for ExecAgent

## Problem

Extended thinking is an Anthropic-specific feature. When running on local LLMs (llama.cpp, PyTorch) or OpenAI, ExecAgent generates plans in a single pass — no internal reasoning before committing. For complex multi-step goals, this produces lower-quality plans: missed steps, wrong ordering, inappropriate priority, and poor sub_goal decomposition.

## Insight

Extended thinking is just structured self-dialogue: draft → critique → refine → commit. The local LLM can do the same thing across multiple calls. The key difference is latency (1 API call vs 2-3 local calls), not capability. A 1.7B model that struggles to produce a perfect plan in one shot can often improve its plan when shown its own draft and asked to evaluate it.

## Design: Contemplation Loop

### Core Concept

A bounded iterative refinement loop inside `_propose_goal()` that runs only when:
1. The provider does NOT support native extended thinking (i.e., not Anthropic with thinking enabled)
2. The initial plan is complex enough to warrant refinement

When Anthropic extended thinking IS available, skip contemplation entirely — Claude's internal reasoning is faster and higher quality.

### Architecture

```
_propose_goal(ctx)
  │
  ├── Pass 1: DRAFT — Generate initial plan (existing code path, any provider)
  │   └── Returns: {goal_description, priority, tool_name, tool_params, sub_goals, reasoning}
  │
  ├── Complexity gate: skip refinement if plan is simple
  │   └── Simple = no sub_goals, or priority <= LOW, or IDLE response
  │   └── Defensive: sub_goals checked via _safe_sub_goal_count(response) — handles None, missing key, non-list
  │
  ├── Preemption check: if _work_available.is_set(), abort contemplation and return draft
  │
  ├── Pass 2: CRITIQUE — Self-evaluate the draft
  │   └── Prompt: "You proposed this plan: [JSON]. Evaluate it."
  │   └── Returns: {confidence: 0-1, issues: [...], suggestions: [...]}
  │   └── On invalid JSON: accept draft as-is (fallback)
  │
  ├── Decision gate: accept or refine
  │   └── confidence >= threshold → accept draft as-is
  │   └── confidence < threshold → one refinement pass
  │
  ├── Preemption check: if _work_available.is_set(), abort contemplation and return draft
  │
  └── Pass 3: REFINE (conditional) — Improve the plan
      └── Prompt: "Original plan: [JSON]. Issues found: [critique]. Produce improved plan."
      └── Returns: revised {goal_description, priority, tool_name, tool_params, sub_goals, reasoning}
      └── On invalid JSON: accept draft as-is (fallback)
```

### Constraints

- **Max 3 passes total** (draft + critique + refine). Never more.
- **Only for complex plans**: Gate on `_safe_sub_goal_count(response) >= min_sub_goals_to_trigger` OR `priority in (HIGH, CRITICAL)`. Simple single-tool actions skip straight to execution.
- **Skip when extended thinking is active**: If `thinking_cfg` is set (Anthropic with thinking enabled), the contemplation loop is redundant. Claude already reasons internally.
- **Respect rate limiting**: Each pass goes through the same `_call_llm_with_timeout` path, so existing rate limits and provider semaphores apply.
- **Total token budget**: The critique pass uses `max_tokens=384` and the refine pass uses `max_tokens=512`, since they're working with existing context, not generating from scratch.
- **Preemption**: Between passes, check `self._work_available.is_set()`. If a new percept arrived (especially a voice command), abort contemplation early and return the current best draft. This prevents percept starvation.
- **Fallback on parse failure**: If any contemplation pass returns invalid JSON, accept the original draft unchanged. Never discard a valid draft because a critique/refine failed to parse.

### Fallback Strategy

Small local models (1.7B) frequently produce malformed JSON. Every contemplation pass must be wrapped in try/except with explicit fallback:

```python
def _critique_plan(self, draft, ctx, ...) -> dict | None:
    """Pass 2: critique. Returns parsed critique dict, or None on failure."""
    try:
        response = self._contemplation_llm_call(system=CRITIQUE_SYSTEM, user=critique_prompt, max_tokens=384)
        if not isinstance(response, dict) or "confidence" not in response:
            return None  # Invalid critique → caller accepts draft as-is
        return response
    except Exception:
        return None  # Parse failure → accept draft

def _refine_plan(self, draft, critique, ctx, ...) -> dict | None:
    """Pass 3: refine. Returns revised plan dict, or None on failure."""
    try:
        response = self._contemplation_llm_call(system=REFINE_SYSTEM, user=refine_prompt, max_tokens=512)
        if not isinstance(response, dict) or not response.get("goal_description"):
            return None  # Invalid refinement → caller keeps draft
        return response
    except Exception:
        return None  # Parse failure → keep draft
```

Rule: **contemplation can only improve a draft, never destroy it.** Any failure mode returns the original draft.

### Defensive Complexity Gate

```python
def _safe_sub_goal_count(self, response: dict) -> int:
    """Safely count sub_goals, handling None, missing key, non-list."""
    sub_goals = response.get("sub_goals")
    if not isinstance(sub_goals, list):
        return 0
    return len(sub_goals)

def _should_contemplate(self, response: dict) -> bool:
    """Complexity gate: only contemplate plans that warrant it."""
    # Reject timeout/error dicts from _call_llm_with_timeout
    if response.get("_timeout") or not response.get("goal_description"):
        return False
    cfg = self._contemplation_config()
    if not cfg.get("enabled", True):
        return False
    priority = str(response.get("priority", "")).upper()
    if priority in ("IDLE",):
        return False
    if cfg.get("trigger_on_high_priority", True) and priority in ("HIGH", "CRITICAL"):
        return True
    min_sg = int(cfg.get("min_sub_goals_to_trigger", 2))
    return self._safe_sub_goal_count(response) >= min_sg
```

### Prompt Design

#### Pass 2: Critique Prompt

```
SYSTEM: You are evaluating a proposed action plan. Be concise and specific.

USER:
You previously proposed this plan:
{draft_json}

Context that led to this plan:
- Root goal: {ctx.root_goal}
- Current mode: {ctx.mode}
- Active goal: {ctx.active_goal}

Evaluate the plan by answering:
1. Are the sub_goals in the right execution order?
2. Are any critical steps missing?
3. Is the priority level appropriate for the situation?
4. Will the tool_params actually work for the chosen tool?
5. Could this be accomplished with fewer steps?

Respond with ONLY valid JSON:
{
    "confidence": <0.0 to 1.0>,
    "issues": ["issue 1", "issue 2"],
    "suggestions": ["fix 1", "fix 2"]
}
```

Note: `accept` field removed from critique output. The confidence threshold is enforced in code (`confidence >= threshold`), not by asking the LLM to redundantly evaluate its own confidence against a threshold it doesn't know.

#### Pass 3: Refine Prompt

```
SYSTEM: You are refining an action plan based on self-critique. Produce an improved version.

USER:
Original plan:
{draft_json}

Self-critique found these issues:
{critique_issues}

Suggestions for improvement:
{critique_suggestions}

Context:
- Root goal: {ctx.root_goal}
- Mode: {ctx.mode}

Produce a corrected plan. Respond with ONLY valid JSON:
{
    "goal_description": "...",
    "priority": "CRITICAL|HIGH|MEDIUM|LOW|IDLE",
    "tool_name": "...",
    "tool_params": {...},
    "reasoning": "...",
    "sub_goals": [...]
}
```

### Where It Lives

**Inside ExecAgent._propose_goal()**, after the initial LLM call and before the ProposedGoal construction. Not in LLMWorker — this is ExecAgent-specific deliberation, not a general LLM feature.

```python
# In _propose_goal(), after getting initial `response` (works for ALL code paths):

if (
    response
    and not thinking_cfg  # Skip if extended thinking handled it
    and self._should_contemplate(response)
):
    response = self._contemplate(response, ctx)
```

New methods on ExecAgent:
- `_contemplation_config() -> dict` — Reads contemplation config from LLMConfig, returns defaults if missing
- `_safe_sub_goal_count(response) -> int` — Defensive sub_goal counting
- `_should_contemplate(response) -> bool` — Complexity gate
- `_contemplate(draft, ctx) -> dict` — Runs critique + optional refine, uses `self._llm_worker` / `self._router` directly
- `_contemplation_llm_call(system, user, max_tokens) -> dict | None` — Shared LLM call path for critique/refine, routes through whichever backend is available (LLMWorker → Router → ChatLLMAgent)
- `_critique_plan(draft, ctx) -> dict | None` — Pass 2
- `_refine_plan(draft, critique, ctx) -> dict | None` — Pass 3

### Multi-Path LLM Routing

The contemplation loop must work through all three code paths in `_propose_goal()`, not just LLMWorker. A shared helper routes calls through whichever backend is available:

```python
def _contemplation_llm_call(self, *, system: str, user: str, max_tokens: int) -> dict | None:
    """Route a contemplation LLM call through the best available backend."""
    llm_worker = self._llm_worker
    router = self._router

    if llm_worker is not None:
        return llm_worker.generate_json_direct(
            system=system,
            user=user,
            temperature=0.2,
            max_tokens=max_tokens,
            request_id=f"contemplate-{uuid.uuid4()}",
            agent_name=self.agent_name,
        )
    elif router is not None:
        return router.generate_json(
            user,
            temperature=0.2,
            system_override=system,
            request_context={"agent": self.agent_name, "lane": "infer"},
        )
    else:
        llm = self._ensure_llm()
        if llm is not None:
            return llm.generate_json(user, system_prompt=system, temperature=0.2)
    return None
```

Note: contemplation calls do NOT pass `tools` or `thinking` — these are plain JSON generation calls. Native tool use and extended thinking are only for the initial draft pass.

### Preemption (Percept Starvation Fix)

The worker loop is single-threaded. During a 3-pass contemplation, incoming percepts (including voice commands) queue behind it. To prevent unacceptable delays:

```python
def _contemplate(self, draft: dict, ctx: StructuredContext) -> dict:
    """Run critique + optional refine. Returns best available plan."""
    # Check for preemption before critique
    if self._work_available.is_set():
        return draft  # New percept arrived — skip contemplation

    critique = self._critique_plan(draft, ctx)
    if critique is None:
        return draft  # Parse failure — accept draft

    cfg = self._contemplation_config()
    threshold = float(cfg.get("confidence_threshold", 0.7))
    confidence = float(critique.get("confidence", 1.0))

    if confidence >= threshold:
        return draft  # Draft is good enough

    # Check for preemption before refine
    if self._work_available.is_set():
        return draft  # New percept arrived — skip refine

    refined = self._refine_plan(draft, critique, ctx)
    if refined is None:
        return draft  # Parse failure — keep draft

    return refined
```

This ensures:
- Voice commands ("Hey Maxim") during contemplation are delayed by at most one pass (~4-10s), not the full 3-pass cycle
- The worst-case delay for a preempting percept is the duration of a single LLM call, not 3x

### Configuration

**Important**: `LLMConfig` is `@dataclass(frozen=True, slots=True)`, so mutable defaults like `dict` are not allowed. Use a frozen mapping type:

```python
# In LLMConfig dataclass — use tuple of key-value pairs (frozen-compatible):
contemplation: tuple[tuple[str, Any], ...] = ()

# Helper to access as dict:
def _contemplation_config(self) -> dict[str, Any]:
    """Read contemplation config from LLMConfig, with sane defaults."""
    defaults = {
        "enabled": True,
        "confidence_threshold": 0.7,
        "min_sub_goals_to_trigger": 2,
        "trigger_on_high_priority": True,
        "max_passes": 3,
        "critique_max_tokens": 384,
        "refine_max_tokens": 512,
    }
    router = self._router
    if router is not None:
        raw = dict(getattr(router.cfg, "contemplation", ()) or ())
        defaults.update(raw)
    return defaults

# In load_llm_config(), after other field extraction:
contemplation_raw = raw.get("contemplation")
if isinstance(contemplation_raw, dict):
    kwargs["contemplation"] = tuple(contemplation_raw.items())
```

Config in `llm.json` under a new `contemplation` key:

```json
{
  "contemplation": {
    "enabled": true,
    "confidence_threshold": 0.7,
    "min_sub_goals_to_trigger": 2,
    "trigger_on_high_priority": true,
    "max_passes": 3,
    "critique_max_tokens": 384,
    "refine_max_tokens": 512
  }
}
```

Defaults are sane — enabled by default, 0.7 confidence threshold, triggers on 2+ sub_goals or HIGH/CRITICAL priority. All defaults are defined in `_contemplation_config()` so the system works without any config file entry.

### How It Interacts with Existing Infrastructure

| Component | Interaction |
|---|---|
| **LLMWorker** | Each contemplation pass calls `generate_json_direct()` — same path as the initial draft. Rate limits, timeouts, provider semaphores all apply. |
| **Router (direct)** | If no LLMWorker, falls through to `router.generate_json()`. Contemplation works identically. |
| **ChatLLMAgent (legacy)** | Last resort fallback. Contemplation calls `llm.generate_json(user, system_prompt=system)` — uses the keyword-arg signature, not string concatenation. |
| **PromptBudgeter** | Not directly involved. Contemplation prompts are self-contained (draft JSON + short context), not full StructuredContext rebuilds. |
| **Energy tracking** | Each pass is a separate LLM call, so energy is tracked per-pass. NAc learns that contemplation costs ~2-3x a single call. |
| **Cost tracking** | Each pass records cost via CostTracker. Budget enforcement applies — if budget hits warning tier mid-contemplation, the refine pass uses a downgraded model. |
| **Extended thinking** | Mutually exclusive. If `thinking_cfg` is set, `_should_contemplate()` returns False. |
| **Native tool use** | Compatible. If Anthropic tool use is active, the draft comes back as structured tool input. Contemplation still works — it evaluates the JSON structure regardless of how it was generated. |
| **Rate limiting** | ExecAgent's `_min_interval` rate limit applies to the overall `_propose_goal()` call, not per-pass. So a 3-pass contemplation counts as one proposal cycle. |
| **Preemption** | `_work_available` is checked between passes. If a new percept arrived, contemplation aborts early and returns the current best draft. |

### Latency Impact

On a local 1.7B model at ~50 tokens/sec:
- Pass 1 (draft): ~500 tokens out → ~10s
- Pass 2 (critique): ~250 tokens out → ~5s
- Pass 3 (refine): ~500 tokens out → ~10s
- **Typical total: ~25s** (vs ~10s without contemplation)

**Worst case (timeout scenario):** Each pass has a 30s timeout via `_call_llm_with_timeout`. If the model is slow or loaded, worst case is 3 × 30s = **90s total**. This is mitigated by:
1. Only triggers for complex plans (2+ sub_goals or HIGH/CRITICAL)
2. Simple plans (single tool, LOW/MEDIUM priority) skip entirely
3. If critique confidence is high (>= 0.7), Pass 3 is skipped → ~15s typical
4. Preemption checks between passes abort early if new percepts arrive
5. Each pass has individual timeout protection — a single hung pass doesn't block forever

**Percept delay (with preemption):** A voice command arriving mid-contemplation waits at most for the current pass to complete (~5-10s typical, 30s worst case), then contemplation aborts and the worker loop processes the new percept.

### What the LLM Learns

Over time, the NAc learns:
- "Contemplation on complex plans produces better outcomes" (success rate improves)
- "Contemplation on simple plans wastes energy" (no outcome improvement, more energy spent)
- The complexity gate self-tunes: if contemplation rarely helps for 2-sub_goal plans but always helps for 4+, the threshold could be adjusted (future work, not Phase 1).

### Biological Analogy

This mirrors the prefrontal cortex's deliberative process:
- **Fast path** (System 1): Simple percept → immediate action. No contemplation needed.
- **Slow path** (System 2): Complex situation → generate candidate plan → internal simulation / critique → revised plan → commit.

The confidence threshold acts like the "feeling of knowing" — when the initial plan feels right (high confidence), skip deliberation. When it feels uncertain, engage the slower reflective process.

The 3-pass cap prevents rumination — the pathological case where deliberation loops indefinitely without converging. Biological systems have the same constraint: you can only deliberate for so long before the situation changes and the context becomes stale.

The preemption check mirrors attentional interrupts — even during deep deliberation, a salient stimulus (voice command, sudden threat) can interrupt the process and redirect attention.

## Implementation Phases

### Phase 1: Core Loop (Minimal)
1. Add `contemplation: dict[str, Any]` field to `LLMConfig` dataclass.
2. Update `load_llm_config()` to read `contemplation` from raw JSON.
3. Add `_contemplation_config()` — reads config with sane defaults.
4. Add `_safe_sub_goal_count()` — defensive sub_goal counting.
5. Add `_should_contemplate()` — complexity gate based on sub_goals count and priority.
6. Add `_contemplation_llm_call()` — shared LLM call path (LLMWorker → Router → ChatLLMAgent).
7. Add `_critique_plan()` — Pass 2 prompt, parse confidence + issues, fallback to None.
8. Add `_refine_plan()` — Pass 3 prompt, parse revised plan, fallback to None.
9. Add `_contemplate()` — orchestrator with preemption checks.
10. Wire into `_propose_goal()` after initial response, gated on `not thinking_cfg`.
11. Tests: mock LLM responses to verify the loop triggers correctly, respects gates, caps at 3 passes, falls back on invalid JSON, and respects preemption.

### Phase 2: Quality Metrics ✓ IMPLEMENTED
12. ✓ Log contemplation outcomes via `_on_goal_completed()` — subscribes to `GoalCompleted` on AgentBus, correlates with `_contemplation_log` metadata.
13. ✓ Track `contemplation_improvement_rate()` — returns success rates for contemplated vs uncontemplated goals with improvement delta.
14. ✓ Feed signal to NAc via `nac.observe()` — event_type="contemplation", event_signature="contemplation:refined" or "contemplation:draft", outcome_valence=POSITIVE/NEGATIVE. Late-wired via `wire_nac()` from MaximAgent.wire_memory_hub().

### Phase 3: Adaptive Threshold ✓ IMPLEMENTED
15. ✓ Added `_adaptive_thresholds()` — queries NAc for `contemplation:refined` and `contemplation:draft` links, computes weighted success rates from `predicted_value × observation_count`, derives improvement delta.
16. ✓ When improvement < -0.1 (contemplation hurts): raises `confidence_threshold` (harder to trigger refine) and increases `min_sub_goals_to_trigger` (tighter gate).
17. ✓ When improvement > 0.1 (contemplation helps): lowers `confidence_threshold` (easier to trigger refine) and decreases `min_sub_goals_to_trigger` (looser gate).
18. ✓ Integrated into `_contemplation_config()` — adaptive adjustments applied after static config, gated on `adaptive_enabled` (default True) and `adaptive_min_observations` (default 10).
19. ✓ Configurable bounds: `adaptive_confidence_floor` (0.3), `adaptive_confidence_ceiling` (0.95), `adaptive_min_sub_goals_floor` (1), `adaptive_min_sub_goals_ceiling` (5). User config always constrains the adaptive range.
20. ✓ Graceful degradation: NAc missing, query exceptions, or insufficient observations → returns None, static config used unchanged.
21. ✓ When only one event signature has data (e.g., only refined or only draft), improvement is computed against a 0.5 baseline.
22. ✓ Logs adaptive decisions via `log_structured("contemplation_adaptive", ...)` at DEBUG level.

### Phase 4: Fast Contemplation Mode ✓ IMPLEMENTED
23. ✓ Added `_contemplate_fast()` — single LLM call that combines critique + refine. Prompt asks for both evaluation metadata (`confidence`, `issues`) and corrected `plan` in one JSON response.
24. ✓ Confidence gate preserved: if `confidence >= threshold`, original draft is used. Only applies the embedded `plan` when confidence is low.
25. ✓ `_contemplate()` dispatches to `_contemplate_standard()` or `_contemplate_fast()` based on `contemplation.mode` config (`"standard"` default, `"fast"` optional).
26. ✓ Fast mode uses `fast_max_tokens` (default 640) since the response contains both evaluation and full plan.
27. ✓ All failure modes return the original draft unchanged (LLM error, missing confidence, missing/invalid plan).

### Phase 5: Smart Preemption ✓ IMPLEMENTED
28. ✓ Added `_urgent_work_available` Event alongside existing `_work_available`.
29. ✓ `_trigger_proposal(urgent=False)` — always sets `_work_available`; when `urgent=True`, also sets `_urgent_work_available`.
30. ✓ Urgent sources: CLI input, comms (SMS/voice), voice keywords (`has_maxim_keyword`), filtered percepts with `urgency >= 0.7`.
31. ✓ Non-urgent sources: high salience+novelty vision percepts, low-urgency filtered percepts.
32. ✓ Contemplation loop (`_contemplate_standard` and `_contemplate_fast`) checks `_urgent_work_available` instead of `_work_available`. Normal percepts queue up and are processed after contemplation finishes.
33. ✓ `_worker_loop` clears both events when processing. `on_stop` sets both events to unblock shutdown.

## Resolved Questions

- **Critique prompt sees minimal context** (draft + root goal + mode), not full StructuredContext. The draft already encodes the LLM's interpretation of the full context, so re-sending it is redundant and wastes tokens.
- **Contemplation is ExecAgent-only**, not available on LLMWorker's standard prompt path. LLMWorker handles fast routing where latency matters more than plan quality.
- **`accept` field removed from critique prompt.** Confidence threshold is enforced in code. Asking the LLM to evaluate its own confidence against a threshold it doesn't know creates inconsistencies.
- **`critique_max_tokens` bumped to 384** (from 256). A critique with 5 evaluation points + issues + suggestions in JSON format needs room.
- **Multi-path support confirmed.** `_contemplation_llm_call()` routes through LLMWorker, Router, or ChatLLMAgent — whichever is available. Contemplation works regardless of which code path produced the initial draft.
- **`LLMConfig` is `frozen=True`** — can't use `dict` field. Config stored as `tuple[tuple[str, Any], ...]` and converted to dict in `_contemplation_config()`.
- **ChatLLMAgent signature** uses `generate_json(user, system_prompt=system)` keyword-arg form, not string concatenation.
- **Timeout dict rejection** — `_should_contemplate()` explicitly rejects `{"_timeout": True}` dicts from `_call_llm_with_timeout` and any response missing `goal_description`.

## Resolved Open Questions

- **Fast contemplation mode** — Implemented in Phase 4. Combined critique+refine in a single prompt, with confidence gate preserved. Configurable via `contemplation.mode: "fast"`. The confidence gate is NOT lost — the combined response includes both `confidence` and `plan`, and the gate is checked in code before accepting the improved plan.
- **Smart preemption** — Implemented in Phase 5. Added `_urgent_work_available` Event that only gets set for CLI, comms, voice keywords, and high-urgency (>= 0.7) filtered percepts. Normal vision percepts no longer interrupt contemplation. Worst-case delay for urgent percepts is one LLM call (~5-10s standard, ~8-12s fast).