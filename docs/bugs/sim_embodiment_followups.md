# Follow-up: Sim Embodiment (E0) Known Issues

**Status:** Open — tracked for E1+ or next sim prompt pass
**Severity:** Low-Medium — functionality works but LLM behavior + latent risks need attention
**Affects:** `maxim --sim "..." --embodiment <ref>`
**Discovered:** 2026-04-19 during E0 PoC simulation

## Issue 1: LLM enters respond loop instead of using affordance tools repeatedly

**Severity:** Medium (prompt engineering)
**Observed:** The 14B model (qwen2.5-14b) calls `rusty_sword_slash` once, then enters a `respond` loop asking the user for guidance instead of continuing to test the sword at different force levels or with manipulated durability.

**Root cause:** The AUT prompt doesn't emphasize that affordance tools exist or guide the agent toward exploratory SEM interaction. The orchestrator's probe message ("The rusty sword has been slashed once with high force. What should be the next action?") is addressed to the AUT as if it were a human, triggering the `respond` / `request_interaction` pattern.

**Fix direction:**
- Update orchestrator probing to be more directive with embodiment goals (e.g., "Now test the sword at lower durability. Call rusty_sword_slash again with maximum force.")
- Consider injecting embodiment tool descriptions into the AUT's system prompt when entity_ref is set (similar to how the prompt assembler includes bio-system state)
- May fold into prompt_b3_b5_track.md (Acting Coach) — the coach could guide tool exploration

## Issue 2: No automatic durability degradation in PoC sim

**Severity:** Low (expected behavior)
**Observed:** The sword starts at durability=1.0 (default), so all slashes succeed. The 1Hz Embodiment poll tick runs `evaluate_failures()` but durability doesn't degrade automatically unless a modulator explicitly drains it.

**Root cause:** The `rusty_sword` spec's `slash` modulator updates `strain` but doesn't degrade `durability` on success. Durability only matters for failure triggers (broken_blade fires when durability < 0.1). In real gameplay, durability degrades through narrative events or DM intervention, not from the slash action itself.

**Fix direction:**
- For PoC stress testing, either (a) pre-set durability to 0.0 via orchestrator intervention, or (b) add a `wear` sensor that the slash modulator increments, with a failure trigger at high wear
- The Asset Foundry (E1) should generate components with more interesting feedback loops (e.g., each slash reduces durability by 0.05)
- Not a code bug — the existing component works as designed

## Issue 3: Latent bridge x subscriber double-recording risk

**Severity:** Low (latent, not active)
**Observed:** When entity_ref is set, `pain_bus=aut_pain_bus` is passed to `build_executor`, causing the ToolPainBridge to subscribe to the same bus that `create_pain_nac_subscriber` is already subscribed to.

**Root cause:** Documented in `pain_bus_bridge_subscriber_unification.md`. Correctness is load-bearing on the context-similarity mismatch (0/1 = 0.0 < 0.5 threshold), not on any guard. If `record_tool_start`'s pending-event context is enriched in the future, double-counting will start silently.

**Fix direction:** Already tracked in `docs/plans/pain_bus_bridge_subscriber_unification.md`. The tripwire test `test_subscriber_does_not_link_pending_tool_event` guards against this. No action needed now.

## Issue 4: ComponentRegistry construction repeated at each call site

**Severity:** Low (code quality)
**Observed:** Both `cli.py` (non-sim path) and `orchestrator.py` (sim path) do the same `if entity_ref: ComponentRegistry()` dance before passing to `build_executor`.

**Root cause:** `build_executor` requires both `entity_ref` and `component_registry` because the registry can be configured with `campaign_dir` by some callers. But most callers just use the default.

**Fix direction:** Consider defaulting `component_registry` inside `build_executor` when `entity_ref` is set but no registry is provided. This would eliminate the repeated pattern. However, it changes the fail-fast precondition contract — evaluate trade-offs before implementing. May fold into `agent_factory_canonicalization.md` when the factory becomes the single construction door.
