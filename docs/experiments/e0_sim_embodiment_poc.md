# E0 Sim Embodiment PoC — Results

**Date:** 2026-04-19
**Plan:** [asset_foundry_plan.md](../plans/archive/asset_foundry_plan.md) Stage 0
**Reproduction:** [protocols/e0_sim_embodiment_reproduction.md](protocols/e0_sim_embodiment_reproduction.md)

## Hypothesis

When `--embodiment weapons/rusty_sword` is passed with `--sim`, the AUT's executor loads the entity via `ComponentRegistry`, registers affordance tools (e.g., `rusty_sword_slash`, `rusty_sword_parry`), and the AUT can invoke them through the normal tool dispatch path. The full pain-cascade learning (SEM affordance -> ToolPainBridge -> NAc) operates in sim mode.

## Method

1. **Integration tests** (10 tests) verify the wiring, preconditions, and pain cascade at the construction level
2. **PoC simulation** runs `maxim --sim "test sword combat" --embodiment weapons/rusty_sword --interactive false --sim-max-turns 5`
3. **JSONL log** captured via `MAXIM_LOG_FILE`

## Results

### Integration tests: 10/10 PASS

| Test | Result |
|------|--------|
| `test_entity_ref_produces_embodiment` | PASS |
| `test_entity_ref_registers_affordance_tools` | PASS |
| `test_entity_ref_registers_sense_tools` | PASS |
| `test_no_entity_ref_no_embodiment` | PASS |
| `test_no_entity_ref_pain_bus_none` | PASS |
| `test_entity_ref_without_nac_raises` | PASS |
| `test_entity_ref_without_pain_bus_raises` | PASS |
| `test_nonexistent_entity_ref_raises` | PASS |
| `test_tool_failure_produces_nac_link` | PASS |
| `test_successful_tool_produces_positive_link` | PASS |

### PoC simulation: PASS

Key evidence from the sim output:

1. **Entity loaded:** `AUT ComponentRegistry created for entity_ref='weapons/rusty_sword'`
2. **Affordance tool invoked by LLM:** `LLM raw response parsed: tool=rusty_sword_slash` — the AUT's LLM independently decided to call the sword affordance
3. **NAc direct attribution:** `Causal link: tool:rusty_sword_slash -> positive (RPE=0.50, confidence=0.50)` — the ToolPainBridge recorded the tool outcome
4. **Hippocampus capture:** `Captured: rusty_sword_slash (salience=0.75)` — the event was stored as an episodic memory
5. **ATL concept formation:** `Concept formed: "rusty_sword_slash" -> action` — concept extraction created a semantic node

### Full test suite: 5229 passed, 1 skipped

No regressions introduced. The default path (no entity_ref) is unchanged.

## Conclusion

The sim orchestrator correctly wires `entity_ref` to `build_executor` when `--embodiment` is passed. Affordance tools are registered, the LLM invokes them, and the full bio-pipeline (NAc learning, hippocampus capture, ATL concept formation) operates. The architectural contract ("only difference between sim and robot path is where percepts come from") is restored for the embodiment dimension.

## Known limitations

- The PoC ran with full durability (1.0), so the slash succeeded and produced a positive NAc link. A zero-durability run would produce failure events and negative links (verified by `test_tool_failure_produces_nac_link`).
- The LLM (qwen2.5-14b) entered a respond loop after the first slash, not continuing to test at different durabilities. This is a prompt engineering concern, not an embodiment wiring issue.
- The pain_bus is passed to build_executor when entity_ref is set, creating the documented latent bridge x subscriber double-recording risk (see `pain_bus_bridge_subscriber_unification.md`). Same risk as CLI non-sim path.
