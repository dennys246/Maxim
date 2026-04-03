# Router Modularization Plan

> **Status:** Not started. Prerequisite for Multi-LLM Scaling Phase 3.

Split `models/language/router.py` (1,721 LOC) into focused modules. The file currently mixes config data, type definitions, config loading, token counting, and the router class itself. Multi-LLM scaling adds `LaneBackendManager` — without splitting first, router.py grows to ~2,000 LOC.

---

## Current Structure (router.py)

| Section | Lines | LOC | What it contains |
|---------|-------|-----|------------------|
| Config data | 46-227 | ~180 | `QUANTIZATION_LEVELS`, `_PROFILE_ALIASES`, `_BUILTIN_PROFILES`, helper functions |
| Type definitions | 290-365 | ~75 | `LLMResponse`, `LLMConfig`, `RoutingPolicy`, `ProviderState`, `_DEFAULT_PRICING` |
| Config loading | 385-600 | ~215 | `_as_bool`, `_read_json`, `load_llm_config()` |
| Token counters | (imported) | ~170 | `CharEstimateCounter`, `LlamaCppTokenCounter` — already in separate file |
| LLMRouter class | 603-1721 | ~1,120 | 42 methods: init, provider mgmt, routing, LLM calls, warmup, introspection |

---

## Target Structure

```
models/language/
  router.py           → ~1,120 LOC  (LLMRouter class only)
  config.py           → ~400 LOC   (LLMConfig, load_llm_config, profiles, quantization)
  types.py            → ~80 LOC    (LLMResponse, RoutingPolicy, ProviderState, pricing)
  llama_backend.py    → exists (161 LOC)
  openai_backend.py   → exists (289 LOC)
  transformers_backend.py → exists (479 LOC)
  token_counter.py    → exists (already split)
```

After multi-LLM scaling:
```
  lane_manager.py     → NEW (~200 LOC, Multi-LLM Phase 3)
```

---

## Migration Steps

### Step 1: Extract `types.py` (~80 LOC)

Move from router.py:
- `LLMResponse` dataclass
- `RoutingPolicy` dataclass
- `ProviderState` dataclass
- `ModelPricing` (if it exists as a class) or `_DEFAULT_PRICING` dict

These are pure data types with no logic. Zero risk.

### Step 2: Extract `config.py` (~400 LOC)

Move from router.py:
- `QUANTIZATION_LEVELS` dict + `list_quantization_levels()` + `get_quantization_info()`
- `_PROFILE_ALIASES` dict + `_BUILTIN_PROFILES` dict
- `_normalize_profile()` + `normalize_llm_profile()` + `list_llm_profiles()`
- `build_model_path()`
- `LLMConfig` dataclass
- `_as_bool()` + `_read_json()` helper functions
- `load_llm_config()` function

`LLMConfig` is the most-imported type from router.py. Moving it to config.py means updating imports, but a re-export in router.py's `__init__` preserves backward compat.

### Step 3: Add re-exports to router.py

```python
# Backward compatibility — these moved to config.py and types.py
from maxim.models.language.config import (  # noqa: F401
    LLMConfig,
    load_llm_config,
    QUANTIZATION_LEVELS,
    # ... etc
)
from maxim.models.language.types import (  # noqa: F401
    LLMResponse,
    RoutingPolicy,
    ProviderState,
)
```

All existing `from maxim.models.language.router import LLMConfig` continues to work. Over time, update callers to import from the new locations.

### Step 4: Verify

- `python -c "import maxim"` — no import errors
- `pytest tests/unit/ -x -q -m "not slow"` — all tests pass
- Grep for old import paths — all resolve via re-exports

---

## What Stays in router.py

The `LLMRouter` class (~1,120 LOC) stays. It's a single cohesive class — splitting its methods into separate files would scatter related logic without improving readability. The goal is to remove the data/config that doesn't belong, not to break apart the router's internal structure.

---

## Import Dependency Order

```
types.py       ← no internal deps (pure dataclasses)
config.py      ← imports types.py (LLMConfig references LLMResponse? No — they're independent)
router.py      ← imports config.py (LLMConfig, load_llm_config) + types.py (LLMResponse, etc.)
lane_manager.py ← imports config.py (LLMConfig) + router.py (LLMRouter) [future, Multi-LLM Phase 3]
```

No circular dependencies. Clean DAG.

---

## Risk

**Low.** This is a mechanical move-and-re-export refactoring. No behavior changes. The re-exports ensure zero breakage for existing callers. The only risk is missing an import, which tests catch immediately.

---

## Effort

~1-2 hours. Mostly find-move-reexport for each section, then run tests.
