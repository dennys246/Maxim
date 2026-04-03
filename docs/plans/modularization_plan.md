# Modularization Plan

Systematic decomposition of monolithic files and removal of dead code. Goal: no single file exceeds ~800 lines, each file has one clear responsibility, and unused code is eliminated.

---

## Motivation

Several core files have grown past 1500-2500 lines with mixed concerns. This makes them hard to navigate, test in isolation, and reason about. Single-author risk amplifies this — if you can't quickly re-orient on a file after time away, the architecture discipline elsewhere is undermined.

The repo also has ~6 dead files (never imported, empty, or deprecated) that add noise.

---

## Phase 0: Dead Code Removal

Remove files that are never imported, empty, or explicitly deprecated. Low risk, immediate clarity gain.

| File | Lines | Reason | Action |
|------|-------|--------|--------|
| `src/maxim/bridges/energy_bridge.py` | 361 | Complete impl, zero imports anywhere | DELETE |
| `src/maxim/bridges/math_bridge.py` | 193 | Complete impl, zero imports anywhere | DELETE |
| `src/maxim/comms/encryption.py` | 92 | AES-256-GCM helpers, never called | DELETE |
| `src/maxim/data/audio/sound.py` | 27 | Backward-compat re-export, nothing imports it | DELETE |
| `src/maxim/environment/metrics.py` | 0 | Empty file | DELETE |
| `src/maxim/memory/store/sqlite_store.py` | 0 | Empty placeholder | DELETE |

**Review before deleting:**
| File | Lines | Reason | Action |
|------|-------|--------|--------|
| `src/maxim/evaluation/llm_benchmark.py` | 83 | Standalone `python -m` script, never imported | KEEP if benchmarking is still useful, else DELETE |

**Total removal: ~673 lines, 6 files**

---

## Phase 1: Critical Splits (1500+ lines)

### 1A. `runtime/agent_loop.py` (2591 lines)

Current responsibilities:
- Approval intent detection (keyword sets, `detect_approval_intent()`)
- State persistence utilities (`_persist_state_json`, `_get_failure_strategy`, `_build_replan_context`)
- Main agent loop (`run_agent_loop()`)
- Agentic loop mode (`run_agentic_loop()`)

**Split plan:**

| New file | Extracted from | Contents | ~Lines |
|----------|---------------|----------|--------|
| `runtime/approval.py` | Lines 55-103 | `_APPROVAL_YES`, `_APPROVAL_NO`, `detect_approval_intent()` | ~60 |
| `runtime/loop_state.py` | Scattered utilities | `_persist_state_json()`, `_get_failure_strategy()`, `_get_plan_depth()`, `_build_replan_context()` | ~120 |
| `runtime/agent_loop.py` | Remainder | Core loop orchestration | ~2400 |

**Note:** 2400 is still large. After Phase 1, evaluate whether `run_agent_loop()` and `run_agentic_loop()` can be further separated. They likely share enough state that forcing a split would create worse coupling than keeping them together.

### 1B. `memory/hippocampus.py` (2280 lines)

Current responsibilities:
- Configuration (`HippocampusConfig`)
- Capture pipeline (async worker, request processing)
- Retrieval (`recall()`, `recall_similar()`, `recall_associated()`, spreading activation)
- Persistence (`save()`, `load()`, `load_with_recovery()`, `save_with_backup()`)
- Sleep consolidation (`sleep()`, `_sleep()`, `sleep_with_clustering()`)
- Memory promotion (`consolidate()`, `_promote_to_long_term()`, `_should_promote()`)
- Consistency (`check_consistency()`, `repair_consistency()`)

**Split plan:**

| New file | Contents | ~Lines |
|----------|----------|--------|
| `memory/hippocampus_persistence.py` | `save()`, `load()`, `load_with_recovery()`, `save_with_backup()`, version migration, atomic writes | ~300 |
| `memory/hippocampus_consolidation.py` | `sleep()`, `_sleep()`, `sleep_with_clustering()`, `consolidate()`, `_promote_to_long_term()`, retention scoring | ~400 |
| `memory/hippocampus_retrieval.py` | `recall()`, `recall_similar()`, `recall_associated()`, spreading activation, association graph formation | ~350 |
| `memory/hippocampus.py` | `HippocampusConfig`, core `Hippocampus` class, capture pipeline, basic store/get/remove, consistency checks — delegates to above modules | ~800 |

**Migration:** `Hippocampus` class stays in `hippocampus.py`. No public API changes — all imports of `Hippocampus` continue to work.

**Extraction strategy (important):** The methods targeted for extraction (`save()`, `load()`, `recall()`, `_sleep()`) all directly access 8+ private attributes (`_rwlock`, `_memories`, `_context_index`, `_graph`, `_stats`, `_compressed_count`, `_memory_contexts`) and call internal helpers (`_compress_memory()`, `_remove_memory()`, `_get_memory_strategy()`). Free-function extraction would require passing 6+ private attributes as parameters, breaking encapsulation.

Instead, use a **mixin approach**: extracted modules define mixin classes (`PersistenceMixin`, `ConsolidationMixin`, `RetrievalMixin`) that `Hippocampus` inherits from. Mixins access `self` directly — same attribute access pattern, just organized into separate files. The `Hippocampus` class becomes:

```python
# hippocampus.py
class Hippocampus(PersistenceMixin, ConsolidationMixin, RetrievalMixin):
    """Core hippocampus — config, capture, store/get/remove, consistency."""
    ...
```

This keeps all methods as instance methods with full `self` access while splitting the file along responsibility boundaries. Each mixin file is independently navigable and testable via the composed class.

### 1C. `models/language/router.py` (2187 lines)

Current responsibilities:
- Token counting (`TokenCounter`, `CharEstimateCounter`, `LlamaCppTokenCounter`, `_LazyTokenCounter`)
- Model profiles (`_BUILTIN_PROFILES` with 20+ configs)
- Prompt formatting (9 prompt style functions)
- JSON extraction (`_sanitize_json_string`, `_find_first_json_object`, `_extract_json_object`)
- Quantization management
- llama.cpp backend (`_LlamaCppBackend`)
- LLM routing (`LLMRouter`, `RoutingPolicy`, `ProviderState`, cost tracking)

**Split plan:**

| New file | Contents | ~Lines |
|----------|----------|--------|
| `models/language/token_counter.py` | `TokenCounter` protocol, `CharEstimateCounter`, `LlamaCppTokenCounter`, `_LazyTokenCounter` | ~70 |
| `models/language/prompt_formats.py` | All 9 prompt style functions, `list_prompt_styles()` | ~160 |
| `models/language/json_parser.py` | `_sanitize_json_string()`, `_find_first_json_object()`, `_extract_json_object()` | ~110 |
| `models/language/llama_backend.py` | `_LlamaCppBackend` class | ~160 |
| `models/language/router.py` | `LLMConfig`, `LLMRouter`, `RoutingPolicy`, `ProviderState`, profiles, quantization | ~1600 |

### 1D. `tools/reachy.py` (1701 lines)

Current responsibilities:
- Novelty tracking data structures (`NoveltyRecord`, `NoveltyInfo`)
- Robot state detection (posture, joint state, body rotation)
- Motion/trajectory definitions
- Tool class wrappers

**Split plan:**

| New file | Contents | ~Lines |
|----------|----------|--------|
| `tools/novelty.py` | `NoveltyRecord`, `NoveltyInfo`, novelty scoring | ~150 |
| `tools/robot_state.py` | State detection, posture checks, joint reading | ~200 |
| `tools/reachy.py` | Tool classes, motion definitions, orchestration | ~1350 |

### 1E. `agents/exec_agent.py` (1576 lines)

**Split plan:**

| New file | Contents | ~Lines |
|----------|----------|--------|
| `agents/exec_prompts.py` | System prompt template, prompt builder functions, tool guidance text | ~250 |
| `agents/exec_agent.py` | `ExecAgent` class, goal proposal logic, autonomy reasoning | ~1300 |

### 1F. `default_network/network.py` (1551 lines)

**Split plan:**

| New file | Contents | ~Lines |
|----------|----------|--------|
| `default_network/gaze_manager.py` | Salience-gated gaze control, exploration gaze | ~250 |
| `default_network/inhibition.py` | DN inhibition from deliberative layer | ~150 |
| `default_network/network.py` | `DefaultNetworkConfig`, main coordination, behavior arbitration, attention filtering | ~1100 |

### 1G. `runtime/prefetch.py` (1454 lines)

**Split plan:**

| New file | Contents | ~Lines |
|----------|----------|--------|
| `runtime/file_patterns.py` | `FILE_PATTERNS`, keyword sets, `detect_file_references()` | ~100 |
| `runtime/fetch_cache.py` | TTL-based cache with file mtime validation | ~200 |
| `runtime/prefetch.py` | Main prefetch orchestration and discovery | ~1100 |

---

## Phase 2: Medium Splits (1000-1500 lines)

### 2A. `integration/memory_hub.py` (1386 lines)
**Assessment:** Acceptable. Central coordinator by design — splitting would create artificial seams. **No action.**

### 2B. `agents/memory_agent.py` (1287 lines)

| New file | Contents | ~Lines |
|----------|----------|--------|
| `memory/association_index.py` | `AssociationIndex` class (keyword indexing, similarity lookup) | ~150 |
| `agents/memory_agent.py` | `MemoryAgent` class and lifecycle management | ~1100 |

### 2C. `modes/definitions.py` (1254 lines)
**Assessment:** Pure configuration/data definitions. Monolithic but each element is specific and self-contained. **No action.**

### 2D. `agents/bus.py` (1231 lines)
**Assessment:** Pure message types, enums, and protocol definitions. This IS the shared contract. Splitting would scatter the API surface. **No action.**

### 2E. `conscience/movement.py` (1223 lines)

| New file | Contents | ~Lines |
|----------|----------|--------|
| `conscience/kinematics.py` | Workspace calculations, coordinate transforms, 6-DOF blending | ~250 |
| `conscience/movement.py` | `MovementMixin`, head tracking, body rotation, trajectory | ~950 |

### 2F. `cli.py` (1124 lines)

| New file | Contents | ~Lines |
|----------|----------|--------|
| `cli_parser.py` | `_build_parser()`, all argument definitions | ~180 |
| `cli.py` | Main entry logic, mode dispatch, event loop | ~940 |

### 2G. `math/angular_gyrus.py` (1040 lines)
**Assessment:** Single well-defined responsibility. Math computation + memory for mathematical knowledge. **No action.**

---

## Phase 3: Validation

After each split:

1. **Import check:** `python -c "import maxim"` — no import errors
2. **Architecture audit:** `python -m maxim --audit-architecture` — no new violations
3. **Test suite:** `pytest tests/unit/ -x` — all passing
4. **Re-export check:** If the original module was imported externally, add re-exports in `__init__.py` so existing imports don't break

---

## Summary

| Phase | Files touched | Lines moved | New files created | Files deleted |
|-------|--------------|-------------|-------------------|---------------|
| 0 (dead code) | 6 deleted | -673 | 0 | 6 |
| 1 (critical) | 7 split | ~2800 extracted | 14 new modules | 0 |
| 2 (medium) | 3 split | ~580 extracted | 3 new modules | 0 |
| **Total** | **16** | **~3380 moved** | **17 new** | **6 deleted** |

**Execution order:** Phase 0 first (quick wins, no risk). Then Phase 1 files in order of most benefit: hippocampus > router > agent_loop > reachy > exec_agent > network > prefetch. Phase 2 last.

Each split should be one commit. Don't batch — if a split introduces a regression, you want a clean revert target.
