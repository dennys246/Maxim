# Router Modularization Plan

> **Status:** Not started. Prerequisite for Multi-LLM Scaling Phase 3.

Split `models/language/router.py` (1,721 LOC) into focused modules. The file currently mixes config data, type definitions, config loading, token counting, and the router class itself. Multi-LLM scaling adds `LaneBackendManager` — without splitting first, router.py grows to ~2,000 LOC.

---

## Current Structure (router.py)

| Section | Lines | LOC | What it contains |
|---------|-------|-----|------------------|
| Imports + re-exports | 1-43 | ~43 | stdlib, cost_tracker, token_counter, prompt_formats, json_parser, llama_backend |
| Config data | 46-282 | ~235 | `QUANTIZATION_LEVELS`, `DEFAULT_QUANTIZATION`, `_PROFILE_ALIASES`, `_BUILTIN_PROFILES`, helper functions, `build_model_path()` |
| Type definitions | 289-382 | ~95 | `LLMResponse`, `LLMConfig`, `RoutingPolicy`, `ProviderState`, `_DEFAULT_PRICING`, `_MODEL_DOWNGRADE_MAP` |
| Config loading | 385-600 | ~215 | `_as_bool`, `_read_json`, `load_llm_config()` |
| LLMRouter class | 603-1721 | ~1,120 | 42 methods: init, provider mgmt, routing, LLM calls, warmup, introspection |

---

## Target Structure

```
models/language/
  router.py           → ~1,200 LOC  (LLMRouter class + private pricing/downgrade data)
  config.py           → ~470 LOC   (LLMConfig, load_llm_config, profiles, quantization)
  types.py            → ~50 LOC    (LLMResponse, RoutingPolicy, ProviderState)
  llama_backend.py    → exists (161 LOC, unchanged)
  openai_backend.py   → exists (289 LOC, unchanged)
  anthropic_backend.py → exists (unchanged)
  transformers_backend.py → exists (479 LOC, unchanged)
  cost_tracker.py     → exists (ModelPricing lives here, unchanged)
  token_counter.py    → exists (already split, unchanged)
  prompt_formats.py   → exists (already split, unchanged)
  json_parser.py      → exists (already split, unchanged)
  __init__.py         → update re-exports to point at new locations
```

After multi-LLM scaling:
```
  lane_manager.py     → NEW (~200 LOC, Multi-LLM Phase 3)
```

---

## What Goes Where

### → `types.py` (~50 LOC) — pure data types, no logic

- `LLMResponse` dataclass (lines 289-301)
- `RoutingPolicy` dataclass (lines 340-356)
- `ProviderState` dataclass (lines 358-365)

These are used by router.py and by backends (openai_backend, anthropic_backend). Moving them breaks no internal coupling.

**NOT moved:** `_DEFAULT_PRICING`, `_MODEL_DOWNGRADE_MAP` — these are private to LLMRouter (used in `_load_pricing_table` and `_model_for_tier`). They stay in router.py. `ModelPricing` already lives in `cost_tracker.py`.

### → `config.py` (~470 LOC) — config loading + profile data

- `QUANTIZATION_LEVELS` dict (lines 46-62)
- `DEFAULT_QUANTIZATION` constant (line 63)
- `list_quantization_levels()` + `get_quantization_info()` (lines 66-74)
- `_PROFILE_ALIASES` dict (lines 77-107)
- `_BUILTIN_PROFILES` dict (lines 109-227)
- `_normalize_profile()` + `normalize_llm_profile()` (lines 230-238)
- `list_llm_profiles()` (lines 241-267)
- `build_model_path()` (lines 271-282)
- `LLMConfig` dataclass (lines 310-337)
- `_as_bool()` + `_read_json()` helpers (lines 385-406)
- `load_llm_config()` (lines 409-600)

`config.py` has no internal deps on `types.py` — `LLMConfig` and `LLMResponse` are independent. `load_llm_config()` uses `os.path.dirname(__file__)` for repo root detection; since config.py is in the same directory as router.py, the relative path (`../../../..`) still resolves correctly.

### → stays in `router.py` (~1,200 LOC) — LLMRouter + private data

- All existing imports (re-pointed to config.py and types.py)
- `_DEFAULT_PRICING` dict (lines 368-375) — private, used by `_load_pricing_table()`
- `_MODEL_DOWNGRADE_MAP` dict (lines 377-382) — private, used by `_model_for_tier()`
- `LLMRouter` class (lines 603-1721) — all 42 methods
- Re-exports for backward compatibility

---

## Migration Steps

### Step 1: Create `types.py` (~50 LOC)

Move `LLMResponse`, `RoutingPolicy`, `ProviderState`. These have no internal dependencies — pure dataclasses with stdlib-only imports.

### Step 2: Create `config.py` (~470 LOC)

Move everything listed above. `config.py` imports:
- `json`, `os` (stdlib)
- `dataclasses.dataclass, field` (stdlib)

No imports from `types.py` or `router.py`. Fully independent.

### Step 3: Update `router.py` imports

Replace the moved definitions with imports from their new homes:

```python
# At top of router.py, replace inline definitions with:
from maxim.models.language.config import (
    LLMConfig,
    load_llm_config,
    QUANTIZATION_LEVELS,
    DEFAULT_QUANTIZATION,
    _BUILTIN_PROFILES,
    _PROFILE_ALIASES,
    _normalize_profile,
    normalize_llm_profile,
    list_llm_profiles,
    list_quantization_levels,
    get_quantization_info,
    build_model_path,
    _as_bool,
    _read_json,
)
from maxim.models.language.types import (
    LLMResponse,
    RoutingPolicy,
    ProviderState,
)
```

These serve double duty: router.py uses them internally AND re-exports them for backward compatibility (existing `from maxim.models.language.router import LLMConfig` keeps working).

### Step 4: Update backend imports (optional, not required)

`openai_backend.py` and `anthropic_backend.py` import `LLMConfig` and `LLMResponse` from router.py. The re-exports make this work without changes. Optionally update to import from the canonical locations:

```python
# openai_backend.py — optional update
from maxim.models.language.config import LLMConfig
from maxim.models.language.types import LLMResponse
```

### Step 5: Update `__init__.py`

The package `__init__.py` already re-exports from router.py. Since router.py re-exports from config.py, this still works. No change required. Optionally update to import from canonical locations for clarity.

### Step 6: Verify

1. `python -c "import maxim"` — no import errors
2. `python -c "from maxim.models.language.router import LLMConfig, load_llm_config, LLMResponse"` — re-exports work
3. `python -c "from maxim.models.language.config import LLMConfig"` — new path works
4. `pytest tests/unit/ -x -q -m "not slow"` — all tests pass
5. Grep for imports — all resolve

---

## Import Dependency Order (verified)

```
types.py       ← stdlib only (dataclasses)
config.py      ← stdlib only (json, os, dataclasses) — independent from types.py
router.py      ← imports config.py + types.py + cost_tracker.py + backends
```

No circular dependencies. `config.py` and `types.py` don't import from each other or from router.py.

---

## Callers That Import from router.py (23 import sites)

All continue to work via re-exports. No caller changes required.

| Caller | What it imports | Status after split |
|--------|----------------|-------------------|
| `llm_worker.py` | `CharEstimateCounter` | Already re-exported from token_counter.py |
| `openai_backend.py` | `LLMConfig`, `LLMResponse` | Re-exported from config.py/types.py |
| `anthropic_backend.py` | `LLMConfig`, `LLMResponse` | Re-exported from config.py/types.py |
| `transformers_backend.py` | `LLMConfig` (TYPE_CHECKING) | Re-exported, TYPE_CHECKING only |
| `exec_agent.py` | `LLMRouter`, `load_llm_config` | Router stays, config re-exported |
| `llm_agent.py` | `_BUILTIN_PROFILES`, `_normalize_profile` | Re-exported from config.py |
| `agentic_runtime.py` | `LLMRouter`, `load_llm_config` | Router stays, config re-exported |
| `cli.py` | `list_llm_profiles`, `normalize_llm_profile` | Re-exported from config.py |
| `__init__.py` | Multiple | Re-exported, no change needed |
| Tests (5 files) | Various | Re-exported, no change needed |

---

## Risk

**Low.** Mechanical move-and-re-export. No behavior changes. Re-exports ensure zero breakage. Tests catch missing imports immediately.

**One thing to watch:** `load_llm_config()` uses `os.path.dirname(__file__)` to find the repo root (line 419). After moving to `config.py`, `__file__` changes from `router.py` to `config.py` — but both are in the same directory (`models/language/`), so the relative path `../../../..` still resolves to the repo root. Verified.

---

## Effort

~1-2 hours. Mostly mechanical: create files, move code blocks, add imports, run tests.
