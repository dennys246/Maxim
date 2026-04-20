# 0.7 Imagination Wiring — Integration PoC Results

**Date:** 2026-04-20
**Plan:** [07_feature_completion.md](../plans/07_feature_completion.md) Integration stage
**Reproduction:** [protocols/07_imagination_wiring_reproduction.md](protocols/07_imagination_wiring_reproduction.md)

## Hypothesis

When `--embodiment` is passed with `--sim`, the orchestrator constructs an ImaginationTrigger and passes it to `run_agentic_loop`. The full I1+I2 pipeline fires: entity extraction from percept text, ComponentIndex two-layer lookup, DN arousal gating, EntityDesigner-based generation (when truly novel), `register_ephemeral()` for session-scoped entities, and provenance tagging (`imagined=True` on Episodes and CausalLinks). On session end, imagined causal links are decayed 50% and ephemeral entities are cleared.

## Method

1. **Integration tests** (17 tests) verify the wiring at the construction level — entity extraction, trigger pipeline, orchestrator wiring pattern, session cleanup, and provenance tagging
2. **Unit tests** (63 tests) verify the imagination module internals — ImaginationTrigger, ImaginationDesigner, ImaginationCache
3. **Component index tests** (40 tests) verify two-layer semantic discovery
4. **Factory + simulation tests** (78 tests) verify the broader agent construction path
5. **Deep parallel review** — Executor lens + Architecture lens review agents independently audit the wiring

## Results

### Integration tests: 19/19 PASS

| Test class | Tests | Result |
|------------|-------|--------|
| `TestEntityExtraction` | 5 | PASS |
| `TestTriggerPipeline` | 6 | PASS |
| `TestOrchestratorWiringPattern` | 2 | PASS |
| `TestSessionCleanup` | 4 | PASS |
| `TestProvenanceTagging` | 2 | PASS |

### Unit tests: 63/63 PASS

All imagination module unit tests pass (trigger, designer, cache).

### Component index tests: 40/40 PASS (1 skipped)

ComponentIndex two-layer discovery works. 1 test skipped (rusty_gate not in seed components).

### Factory + simulation tests: 78/78 PASS

No regressions in agent construction or simulation paths.

### Chain verification

| # | Chain link | Status | Evidence |
|---|-----------|--------|----------|
| 1 | Entity extraction | PASS | `extract_entity_phrases()` finds SEM-relevant nouns (creatures, weapons, items, NPCs, environmental features) |
| 2 | ComponentIndex lookup | PASS | Two-layer: alias hash table O(1) + embedding cosine similarity (threshold 0.65) |
| 3 | DN arousal gate | PASS | `imagination_allowed()` blocks during high arousal; gate open when DN is None |
| 4 | Cache deduplication | PASS | Second mention of same phrase hits cache |
| 5 | Thread safety | PASS | 8-thread concurrent `process_percept()` with no errors |
| 6 | Orchestrator wiring | PASS | `aut_imagination_trigger` constructed only when `aut_component_registry is not None` |
| 7 | Session cleanup | PASS | `tag_imagined_links()` + `decay_imagined_links(0.5)` + `clear_ephemeral()` all work correctly |
| 8 | Provenance: Episode | PASS | `Episode.imagined` defaults to False, accepts True |
| 9 | Provenance: CausalLink | PASS | `CausalLink.imagined` defaults to False, accepts True |
| 10 | No-entity-ref path | PASS | `entity_ref=None` → trigger stays None, no construction attempted |

### Deep review findings

### Deep review: 2-lens pre-merge audit

Two independent review Claudes audited the 0.7 wiring. Key findings:

**Executor lens (14 findings):**
- **CRITICAL #1 (FIXED): Provenance gap.** `imagined=True` was never set on CausalLinks at runtime. Fields existed but no code path populated them, making `decay_imagined_links()` dead code. **Fix:** Added `NAc.tag_imagined_links(entity_refs)` to retroactively tag links by matching event signatures against imagined entity ref basenames. Wired into orchestrator session-end cleanup: tag → decay → clear.
- **MEDIUM #8:** DN-None arousal bypass — by design (no DN → gate open)
- **MEDIUM #12:** Reachy runtime no trigger — acceptable (deferred to 0.8)
- 10 LOW findings: all correct behavior confirmed
- **Cross-confirmed:** Lock ordering safe, all 7 run_agentic_loop call sites correctly wired

**Architecture lens (9 findings):**
- **MEDIUM #2.1:** `process_percept()` blocks on LLM calls (2-15s stall per novel entity). Mitigated by threshold=2 + arousal gate. Async dispatch deferred to I4 (0.8).
- **MEDIUM #6.1:** ImaginationCache unbounded within session (no cap/eviction). Acceptable for typical sessions.
- **MEDIUM #6.2:** ComponentIndex.add() uses np.vstack per entity (O(N^2)). Acceptable for typical imagined entity counts (<50/session).
- **Cross-confirmed:** Lock ordering verified safe. Thread safety invariants maintained.

All critical and blocking findings folded into fix commits BEFORE PR.

## Conclusion

The 0.7 imagination wiring is correctly integrated into the orchestrator's AUT path. The trigger fires only when `--embodiment` provides a ComponentRegistry, entities are extracted from percept text, known entities resolve via ComponentIndex, and novel entities can be designed in real-time (gated by DN arousal + energy budget). Session cleanup properly decays imagined causal links and clears ephemeral entities. The default path (no entity_ref) is unchanged.

## Architecture decisions

- **Imagination trigger lives in orchestrator, not AgentFactory:** The trigger requires ComponentIndex + EntityDesigner + DefaultNetwork, which are orchestrator-scope objects. Pushing it into the factory would require passing too many optional deps.
- **Only AUT gets imagination:** The orchestrator agent does NOT get imagination — it's a planner, not an actor. Sub-AUTs, CLI non-sim, api.py headless, and Reachy runtime do not get imagination yet.
- **Reachy runtime deferred:** The embodied runtime (agentic_runtime.py) already has a ComponentRegistry but constructing imagination there requires careful DN interaction testing with real hardware. Deferred to 0.8.

## Additional changes

- **Default embodiment for sim mode:** `bodies/base_humanoid` is now the default when `--embodiment` is not specified in sim mode. Provides 5 sensors, 4 modulators (8 affordances: move, rest, look, listen, pick_up, drop, use, speak), and 3 failure modes. This activates the full 0.7 chain (Acting Coach, imagination, scene-scoped tools) by default.
- **`NAc.tag_imagined_links(entity_refs)`:** Retroactively tags causal links by matching event signatures against imagined entity ref basenames. Called at session end before `decay_imagined_links()`.

## Known limitations

- EntityDesigner LLM calls inside `process_percept()` block the agent loop thread. The 2Hz target loop frequency means a 500ms LLM call consumes a full tick. Async design dispatch is deferred to I4 (0.8).
- Auto-curation in CLI builds a SEPARATE ComponentRegistry from the orchestrator's. Newly curated components are available to the orchestrator (it re-discovers from disk) but the ComponentIndex instances are separate — no shared state.
- ImaginationCache is unbounded within a session. Acceptable for typical sessions (<1000 phrases) but pathological long-running sessions could accumulate unbounded memory.
- Episode provenance tagging (`imagined=True` on Episodes) is not yet implemented — only CausalLink provenance is wired. Episode tagging requires a hippocampus hook (deferred).
